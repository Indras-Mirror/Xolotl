#!/usr/bin/env python3
"""compare_ernie_graph.py - gate the ERNIE ggml-graph dumps against a reference dump.

    python3 tools/v41/engine/compare_ernie_graph.py --mine DIR --ref DIR [--tol 1e-3] [--bands]
                                                   [--only a,b] [--skip prefix,...] [--json out.json]

Both dumps are DIR/<name>.f32 + DIR/manifest.json in the same layout (test_ernie_graph, ernie_dense's
test_ernie_dense, or the llama.cpp PR's ernie_dump).  Per tensor it reports the relative L2 and max abs diff.

The routed-expert selections (ffn_moe_topk-N, dumped as floats) are gated separately: the fraction of tokens whose
top-k SET is identical must be >= 99 % (the reference's own top-k order is not canonical).

--bands uses the tolerances the CPU runner (ernie-s3) established against the PR dumps: 1e-3 per tensor, and for the
residual (l_last / residual-all / ffn_out at layer 0) the 4e-3 the reference's own arithmetic scatters (Q8_0 activation
rounding), plus cosine >= 0.99999.
"""
import argparse
import json
import os
import sys

import numpy as np

TOPK_PREFIX = "ffn_moe_topk-"
RESIDUAL_NAMES = ("l_last-", "residual-all", "ffn_out-")
BAND_TOL = {"residual": 4e-3, "default": 1e-3, "cosine": 0.99999}


def load(d):
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    out = {}
    for t in m["tensors"]:
        key = t["name"] + (f"__{t['seq']}" if t.get("seq", 1) > 1 else "")
        path = os.path.join(d, t["file"])
        if not os.path.exists(path):
            continue
        a = np.fromfile(path, dtype="<f4")
        if a.size:
            out[key] = a
    return out, m


def rel(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    den = float(np.sqrt(np.sum(b.astype(np.float64) ** 2)))
    num = float(np.sqrt(np.sum(d * d)))
    r = num / den if den > 0 else num
    cos = float(np.dot(a.astype(np.float64), b.astype(np.float64)) /
                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return r, float(np.max(np.abs(d))) if d.size else 0.0, cos


def topk_sets(a, b, n_used):
    """a / b: flat float ids of [n_used, nt] (ggml order: id fastest).  Returns (set-agree, order-agree)."""
    if a.size != b.size or a.size % n_used:
        return float("nan"), float("nan")
    nt = a.size // n_used
    sa = np.sort(a.reshape(nt, n_used).astype(np.int64), axis=1)
    sb = np.sort(b.reshape(nt, n_used).astype(np.int64), axis=1)
    same_set = int(np.sum(np.all(sa == sb, axis=1)))
    same_ord = int(np.sum(np.all(a.reshape(nt, n_used).astype(np.int64) == b.reshape(nt, n_used).astype(np.int64), axis=1)))
    return same_set / nt, same_ord / nt


def ids_from_probs(probs, weights, n_expert, n_used):
    """Recover the selection from ffn_moe_probs [n_expert, nt] and ffn_moe_weights [1, n_used, nt]: the final score
    of a selected expert equals its weight, and the match is unique (ernie-s4's recovery)."""
    if probs.size % n_expert:
        return None
    nt = probs.size // n_expert
    if weights.size != nt * n_used:
        return None
    ids = np.zeros((nt, n_used), dtype=np.int64)
    for t in range(nt):
        p = probs[t * n_expert:(t + 1) * n_expert]
        for k in range(n_used):
            v = weights[k + n_used * t]
            j = int(np.argmin(np.abs(p - v)))
            if abs(float(p[j]) - float(v)) > 1e-6:
                return None
            ids[t, k] = j
    return ids


def set_agreement(ia, ib):
    nt = min(len(ia), len(ib))
    if nt == 0:
        return float("nan"), float("nan")
    sa = np.sort(ia[:nt], axis=1)
    sb = np.sort(ib[:nt], axis=1)
    return float(np.mean(np.all(sa == sb, axis=1))), float(np.mean(np.all(ia[:nt] == ib[:nt], axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--json")
    ap.add_argument("--router-from-probs", action="store_true",
                    help="recover the reference's ids from ffn_moe_probs + ffn_moe_weights (the PR dumper's own "
                         "ffn_moe_topk is misaligned after token 0; ernie-s4's finding), mine stays ffn_moe_topk")
    ap.add_argument("--n-used", type=int, default=10)
    ap.add_argument("--n-expert", type=int, default=512)
    args = ap.parse_args()
    only = [x for x in args.only.split(",") if x]
    skip = [x for x in args.skip.split(",") if x]

    mine, _ = load(args.mine)
    ref, _ = load(args.ref)

    rows, fails, worst = [], 0, 0.0
    for name in sorted(mine):
        if only and name not in only:
            continue
        if any(name.startswith(s) for s in skip):
            continue
        if name not in ref:
            continue
        a, b = mine[name], ref[name]
        if a.size != b.size:
            rows.append((name, float("nan"), float("nan"), float("nan"), "size mismatch"))
            fails += 1
            continue
        r, mx, cos = rel(a, b)
        if name.startswith(TOPK_PREFIX):
            if args.router_from_probs:
                ln = name[len(TOPK_PREFIX):]
                rp, rw = ref.get(TOPK_PREFIX.replace("topk", "probs") + ln), ref.get("ffn_moe_weights-" + ln)
                if rp is None or rw is None:
                    rows.append((name, r, mx, cos, "no probs/weights in the reference to recover from"))
                    continue
                rid = ids_from_probs(rp, rw, args.n_expert, args.n_used)
                if rid is None:
                    rows.append((name, r, mx, cos, "reference id recovery failed"))
                    fails += 1
                    continue
                mid = mine[name].reshape(-1, args.n_used)[:len(rid)]
                ss, so = set_agreement(mid.astype(np.int64), rid)
                ok = ss >= 0.99
                rows.append((name, r, mx, ss, f"top-{args.n_used} set agree {ss:.4f} (ordered {so:.4f}; ref ids "
                                              f"recovered from probs+weights)"))
                fails += 0 if ok else 1
                continue
            ss, so = topk_sets(a, b, args.n_used)
            ok = ss >= 0.99
            rows.append((name, r, mx, ss, f"top-{args.n_used} set agree {ss:.4f} (ordered {so:.4f})"))
            fails += 0 if ok else 1
            continue
        tol = args.tol
        if args.bands:
            tol = BAND_TOL["residual"] if name.startswith(RESIDUAL_NAMES) else BAND_TOL["default"]
        ok = r <= tol and (not args.bands or cos >= BAND_TOL["cosine"] or not name.startswith(RESIDUAL_NAMES))
        if not np.isnan(r):
            worst = max(worst, r)
        rows.append((name, r, mx, cos, "" if ok else "FAIL"))
        fails += 0 if ok else 1

    missing = [n for n in sorted(ref) if n not in mine and not only]
    width = max([len(r[0]) for r in rows] + [10])
    print(f"{'tensor':<{width}}  {'rel L2':>10}  {'max abs':>10}  {'cosine':>9}  note")
    for name, r, mx, cos, note in rows:
        print(f"{name:<{width}}  {r:>10.3e}  {mx:>10.3e}  {cos:>9.6f}  {note}")
    if missing:
        print(f"\n{len(missing)} reference tensors have no counterpart here (not gated): " + ", ".join(missing[:12]) +
              (" ..." if len(missing) > 12 else ""))
    print(f"\n{len(rows)} tensors compared, worst rel L2 {worst:.3e}, tol {args.tol:.1e} -> "
          f"{'PASS' if rows and fails == 0 else 'FAIL'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": [{"name": n, "rel_l2": r, "max_abs": m, "cosine": c, "note": s}
                                for n, r, m, c, s in rows], "worst": worst, "tol": args.tol,
                       "pass": bool(rows) and fails == 0}, f, indent=1)
    return 0 if rows and fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
