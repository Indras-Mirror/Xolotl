#!/usr/bin/env python3
"""compare_ernie.py - diff two ERNIE dense dumps (test_ernie_dense vs the llama.cpp PR's ernie_dump).

Both sides write DIR/<name>.f32 (float32, ggml order: ne[0] fastest) and DIR/manifest.json, and both use the same
ggml tensor names, so a comparison is name -> element-wise.  A relative L2 is reported per tensor; the gate passes
when every compared tensor is under --tol (default 1e-3).

    python3 tools/v41/engine/compare_ernie.py --mine dump/mine32 --ref /tmp/s3_pr32 [--tol 1e-3]
    ... [--only l_last-0,attn_out-1] [--skip ffn_out] [--npz ref.npz] [--json out.json]
    ... [--router] [--indexer]

--indexer compares the DSA indexer's selected sets per MLA layer (indexer_top_k) and its finite score entries
(indexer_score); both taps are written by test_ernie_dense and the PR's ernie_dump.

--npz compares against an ernie_ref.py .npz instead (arrays named like the ggml tensors, transformed to numpy's
row-major order; only the tensors present on both sides are compared).
"""
import argparse
import json
import os
import sys

import numpy as np


def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    out = {}
    for t in m["tensors"]:
        key = t["name"] + (f"__{t['seq']}" if t.get("seq", 1) > 1 else "")
        out[key] = t
    return out, m


def load_f32(d, t):
    a = np.fromfile(os.path.join(d, t["file"]), dtype="<f4")
    ne = [int(x) for x in t["ne"]]
    n = int(np.prod(ne))
    if a.size != n:
        raise SystemExit(f"{t['file']}: {a.size} floats, manifest says {ne} = {n}")
    # ggml order -> C order (the reverse of the ne list), then flatten: values are identical by construction
    return a


def rel(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    den = float(np.sqrt(np.sum(b.astype(np.float64) ** 2)))
    num = float(np.sqrt(np.sum(d * d)))
    return (num / den if den > 0 else num), float(np.max(np.abs(d)))


def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den else float("nan")


def rel_masked(a, b):
    """rel() over the entries finite on both sides; the +-inf pattern (the DSA score's window/future mask) is checked
    separately.  Returns (rel, max_abs, note)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not (fa.all() and fb.all()):
        if not np.array_equal(fa, fb):
            return float("nan"), float("nan"), (f"non-finite pattern differs: {int(fa.sum())} vs {int(fb.sum())} "
                                                f"finite of {a.size}")
        a, b = a[fa], b[fb]
        if a.size == 0:
            return 0.0, 0.0, ""
    r, mx = rel(a, b)
    return r, mx, ""


def router_ids(d, names, il, n_tok, n_expert, top_k):
    """The router's (token, slot) -> expert ids for layer `il`, as an [n_tok, top_k] int array.

    Preferred source is the `ffn_moe_topk-<il>` tap.  The PR's own dumper writes that tensor through a *view* of its
    argsort output, so everything but token 0's slots is misaligned garbage in the file; when the dump also has
    `ffn_moe_probs-<il>` (the full [n_expert, n_tok] weights) and `ffn_moe_weights-<il>` ([1, top_k, n_tok]), the ids
    are recovered exactly: each selected weight equals probs[expert] * scale / sum, so the matching expert is the
    unique argmin of |probs - weight|.  Returns (ids [n_tok, top_k], weights or None, source) or (None, None, reason).
    """
    probs_name, w_name, topk_name = (f"ffn_moe_probs-{il}", f"ffn_moe_weights-{il}", f"ffn_moe_topk-{il}")
    if probs_name in names and w_name in names:
        p = load_f32(d, names[probs_name]).reshape(n_tok, n_expert)
        w = load_f32(d, names[w_name]).reshape(n_tok, top_k)
        ids = np.zeros((n_tok, top_k), dtype=np.int64)
        for t in range(n_tok):
            for k in range(top_k):
                ids[t, k] = int(np.argmin(np.abs(p[t] - w[t, k])))
        return ids, w, "probs+weights"
    if topk_name in names:
        return load_f32(d, names[topk_name]).reshape(n_tok, top_k).round().astype(np.int64), None, "topk tap"
    return None, None, "no router taps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--npz")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--router", action="store_true",
                    help="also report router id agreement per MoE layer (ids recovered from probs+weights when the "
                         "dump has them: the PR dumps ffn_moe_topk through a view, so only slot 0 of token 0 is real)")
    ap.add_argument("--indexer", action="store_true",
                    help="also report the DSA indexer's selected-set agreement per MLA layer (indexer_top_k entries "
                         "with 0 <= index <= token position; the row is padded, and ggml's top_k pads -inf cells)")
    ap.add_argument("--json")
    args = ap.parse_args()

    only = [x for x in args.only.split(",") if x]
    skip = [x for x in args.skip.split(",") if x]

    mine, _ = load_manifest(args.mine)
    rows = []
    if args.npz:
        z = np.load(args.npz)
        for name, t in mine.items():
            if name not in z:
                continue
            if only and name not in only:
                continue
            if any(name.startswith(s) for s in skip):
                continue
            b = np.asarray(z[name], dtype=np.float32).reshape(-1)
            a = load_f32(args.mine, t)
            if a.size != b.size:
                rows.append((name, float("nan"), float("nan"), float("nan"), f"size {a.size} vs {b.size}"))
                continue
            r, mx = rel(a, b)
            rows.append((name, r, mx, cosine(a, b), ""))
    else:
        ref, _ = load_manifest(args.ref)
        for name, t in mine.items():
            if name not in ref:
                continue
            if only and name not in only:
                continue
            if any(name.startswith(s) for s in skip):
                continue
            if "(view)" in name or "(reshaped)" in name or "(permuted)" in name:
                continue   # the PR dumper's renamed views of the same data
            a = load_f32(args.mine, t)
            b = load_f32(args.ref, ref[name])
            if a.size != b.size:
                rows.append((name, float("nan"), float("nan"), float("nan"), f"size {a.size} vs {b.size}"))
                continue
            r, mx, note = rel_masked(a, b)
            rows.append((name, r, mx, cosine(a, b) if not note else float("nan"), note))

    width = max([len(r[0]) for r in rows] + [10])
    print(f"{'tensor':<{width}}  {'rel L2':>10}  {'max abs':>10}  {'cos':>10}  note")
    worst = 0.0
    bad = 0
    for name, r, mx, cs, note in rows:
        flag = ""
        if note:
            flag = "  <-- " + note
            bad += 1
        elif r > args.tol:
            flag = "  <-- FAIL"
            bad += 1
        if not np.isnan(r):
            worst = max(worst, r)
        print(f"{name:<{width}}  {r:>10.3e}  {mx:>10.3e}  {cs:>10.8f}{flag}")
    print(f"\n{len(rows)} tensors compared, worst rel L2 {worst:.3e}, tol {args.tol:.1e} -> "
          f"{'PASS' if bad == 0 and rows else 'FAIL'}")

    # ---- the router's ids, layer by layer (the gate's "top-10 ids identical on >= 99% of (token, layer) slots")
    if args.router and not args.npz and rows:
        ref, _ = load_manifest(args.ref)
        n_tok = len(json.load(open(os.path.join(args.mine, "manifest.json")))["tokens"])
        print(f"\nrouter (top-k ids, {n_tok} tokens):")
        for name in sorted(mine):
            if not name.startswith("ffn_moe_topk-"):
                continue
            il = name.rsplit("-", 1)[-1]
            # shapes: ffn_moe_probs-<il> is [n_expert, n_tok]; the weights [1, top_k, n_tok]
            pt = mine.get(f"ffn_moe_probs-{il}")
            wt = mine.get(f"ffn_moe_weights-{il}")
            if not pt or not wt:
                continue
            n_expert = int(pt["ne"][0])
            top_k = int(wt["ne"][1])
            ra, wa, src_a = router_ids(args.mine, mine, il, n_tok, n_expert, top_k)
            rb, wb, src_b = router_ids(args.ref, ref, il, n_tok, n_expert, top_k)
            if ra is None or rb is None:
                print(f"  layer {il}: ids unavailable on one side (mine: {src_a}, ref: {src_b})")
                continue
            slot = float(np.mean(ra == rb))
            sets = float(np.mean([len(set(x) & set(y)) / max(1, len(x)) for x, y in zip(ra, rb)]))
            exact = float(np.mean([set(x) == set(y) for x, y in zip(ra, rb)]))
            line = (f"  layer {il}: slot-exact {slot:.4f}  set-overlap {sets:.4f}  tokens with identical set "
                    f"{exact:.4f}")
            if wa is not None and wb is not None:
                # the gate's "weights within 1e-2": everywhere, and on the slots whose expert id agrees (a slot that
                # selected a different expert has O(1) different weight by construction, so it is reported apart)
                both = ra == rb
                dw = np.abs(wa - wb)
                line += (f"  |dw| all max {dw.max():.3e}  on agreeing slots max {dw[both].max() if both.any() else 0:.3e}")
            print(line + f"  (ids from {src_a})")

    # ---- the DSA indexer's selected sets, layer by layer
    if args.indexer and not args.npz and rows:
        ref, _ = load_manifest(args.ref)
        print("\nindexer (top-k selected sets, single pass: valid = 0 <= index <= token position):")
        for name in sorted(mine):
            if not name.startswith("indexer_top_k-") or name not in ref:
                continue
            il = name.rsplit("-", 1)[-1]
            # [n_top, nt] rows are padded; only entries with 0 <= index <= token position are real

            def grid(d, t):
                a = load_f32(d, t).reshape(int(t["ne"][1]), int(t["ne"][0]))
                return a.round().astype(np.int64), int(t["ne"][0])
            A, ca = grid(args.mine, mine[name])
            B, _ = grid(args.ref, ref[name])
            nt = A.shape[0]
            ov, exact, na, nb = [], 0, [], []
            for t in range(nt):
                a = {int(x) for x in A[t] if 0 <= x <= t}
                b = {int(x) for x in B[t] if 0 <= x <= t}
                na.append(len(a))
                nb.append(len(b))
                ov.append(len(a & b) / max(1, len(a)))
                exact += int(a == b)
            line = (f"  layer {il}: set-overlap {np.mean(ov):.4f}  tokens with identical set "
                    f"{exact / max(1, nt):.4f}  selected {np.mean(na):.1f} (mine) / {np.mean(nb):.1f} (ref) of "
                    f"{ca} top-k row cells")
            # the masked score: finite entries only; the +inf window / -inf future pattern must match.  The reference
            # scores over its whole padded cache (n_ctx rounded up), mine over the written cells; compare the prefix.
            sn = f"indexer_score-{il}"
            if sn in mine and sn in ref:
                sa = load_f32(args.mine, mine[sn])
                sb = load_f32(args.ref, ref[sn])
                na_, nb_ = int(mine[sn]["ne"][0]), int(ref[sn]["ne"][0])
                nt_ = int(mine[sn]["ne"][1])
                n = min(na_, nb_)
                sa = sa.reshape(nt_, na_)[:, :n].reshape(-1)
                sb = sb.reshape(nt_, nb_)[:, :n].reshape(-1)
                r, mx, note = rel_masked(sa, sb)
                line += f"  score rel {r:.3e} (max {mx:.3e})"
                if note:
                    line += f"  [{note}]"
            print(line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": [{"name": n, "rel_l2": r, "max_abs": m, "cos": c, "note": s}
                                for n, r, m, c, s in rows],
                       "worst": worst, "tol": args.tol, "pass": bad == 0 and bool(rows)}, f, indent=1)
    return 0 if bad == 0 and rows else 1


if __name__ == "__main__":
    sys.exit(main())
