#!/usr/bin/env python3
"""abl_sensitivity.py - does the abliteration edit change the per-expert tier ranking?

Test §4.1 of private/notes/PLAN_2026-10-28_QWEN38_XQ_V2.md.  For one layer and a sample of
experts, measure the DOWN-projection output error at each candidate tier twice - pristine FP8
weights, and abliterated weights `W - lam*u(u^T W)` from the real adapter - on the same
activations, rows and quantiser.  Reports per-expert error, argmin agreement and Spearman, plus
alpha/signal (how much of the edit each tier actually stores).

Why it matters: the 2026-10-27/28 run screened pristine weights.  If the ranking is stable, the
down numbers are reusable; if it flips, the screen has to be redone with the edit applied.

  python3 tools/qwen38/abl_sensitivity.py --layer 16 --experts 12 \
      --adapter /media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf

Note: uses a UNIFORM down imatrix (identity), identical in both arms, so the comparison is fair;
screen.py fits the imatrix from data instead.  Only the relative ranking is being tested here.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import screen as S            # noqa: E402  FP8Model, GGML, dumps, expert math
import alpha_preflight as P   # noqa: E402  load_adapter (adapter -> per-family directions)


def spearman(a, b):
    """Spearman rho without scipy: ranks, then Pearson on the ranks."""
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties
        vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            sums = np.zeros(len(vals)); np.add.at(sums, inv, r)
            r = (sums / cnt)[inv]
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8")
    ap.add_argument("--dump", default="/media/mal/NVME1TB/qwen38/dumps")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", type=int, default=12, help="experts sampled (evenly spread)")
    ap.add_argument("--rows", type=int, default=256, help="token rows per expert (0 = all)")
    ap.add_argument("--tiers", default="q2_0,iq4_nl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    G = S.GGML()
    model = S.FP8Model(a.model_dir)

    dirs = P.load_adapter(a.adapter)
    key = f"blk.{a.layer}.ffn_down_exps.weight"
    if key not in dirs:
        near = [k for k in dirs if f".{a.layer}." in k]
        raise SystemExit(f"adapter has no {key}; layer {a.layer} entries: {near[:8]}")
    u, lam = dirs[key]["u"], dirs[key]["lam"]
    if u.ndim == 1:
        u = u[:, None]
    print(f"adapter: {key}  u={u.shape}  lam={lam}  rank={dirs[key]['rank']}")

    dump = S.dump_files(a.dump, a.layer)
    if not dump:
        raise SystemExit(f"no dumps for layer {a.layer} in {a.dump}")
    X, ids = S.read_dump(dump)
    T, N = X.shape
    K = ids.shape[1]

    router = model.router(a.layer)
    logits = X @ router.T
    top = np.argpartition(-logits, K - 1, axis=1)[:, :K]
    z = np.take_along_axis(logits, top, axis=1)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    w = p / p.sum(axis=1, keepdims=True)
    w_full = np.zeros((T, router.shape[0]), dtype=np.float32)
    w_full[np.arange(T)[:, None], top] = w

    # experts that actually receive tokens, evenly spread over the layer
    counts = np.bincount(ids.ravel(), minlength=router.shape[0])
    live = np.where(counts > 0)[0]
    pick = live[np.linspace(0, len(live) - 1, min(a.experts, len(live))).astype(int)]

    n_eff = u.shape[1]
    # A single column means ONE direction shared by every expert in the layer (measured: the
    # adapter's rank-512 LoRA collapses to one normalised vector, u = [2560, 1]).  Broadcast it.
    shared = n_eff == 1
    print(f"layer {a.layer}: {T} tokens, {len(live)} live experts, "
          f"adapter covers {n_eff} expert direction(s)"
          f"{' - SHARED by all experts' if shared else ''}")

    rows_out = []
    for e in pick:
        if e >= n_eff and not shared:
            print(f"  expert {e}: no adapter direction (u has {n_eff} columns) - skipped")
            continue
        rows = np.where(ids == e)[0]
        if rows.size == 0:
            continue
        if a.rows and rows.size > a.rows:
            rows = rows[:: max(1, rows.size // a.rows)][: a.rows]
        wgt = w_full[rows, e]
        Xe = X[rows]
        Wg = model.expert(a.layer, e, S.FGATE)
        Wu = model.expert(a.layer, e, S.FUP)
        Wd = model.expert(a.layer, e, S.FDOWN)

        ue = np.ascontiguousarray(u[:, 0 if shared else e], dtype=np.float32)
        d_int = -lam * np.outer(ue, ue @ Wd)
        Wd_abl = Wd + d_int

        im = np.ones(Wd.shape[1], dtype=np.float32)
        Y = S.expert_out(Xe, Wg, Wu, Wd) * wgt[:, None]
        y_en = float(np.square(Y).sum())
        rec = {"expert": int(e), "rows": int(rows.size), "y_energy": y_en}
        for t in tiers:
            # pristine
            Wp, _ = G.quantize_dequantize(Wd, t, im)
            Yp = S.expert_out(Xe, Wg, Wu, Wp) * wgt[:, None]
            rec[f"e_{t}"] = float(np.square(Yp - Y).sum()) / y_en
            # abliterated
            Wa, _ = G.quantize_dequantize(Wd_abl, t, im)
            Ya = S.expert_out(Xe, Wg, Wu, Wa) * wgt[:, None]
            rec[f"a_{t}"] = float(np.square(Ya - Y).sum()) / y_en
            # how much of the edit this tier stores
            d_real = Wa - Wd
            den = float((d_int * d_int).sum())
            alpha = float((d_real * d_int).sum() / den) if den else float("nan")
            rec[f"alpha_{t}"] = alpha
            nz = float((d_real * d_real).sum())
            rec[f"signal_{t}"] = (alpha * alpha * den / nz * 100.0) if nz else float("nan")
        rows_out.append(rec)

    if not rows_out:
        raise SystemExit("no experts measured")

    print(f"\n{'expert':>7} " + " ".join(f"{t:>10} {'abl':>10}" for t in tiers)
          + "   alpha/signal%")
    for r in rows_out:
        cells = " ".join(f"{r[f'e_{t}']:10.5f} {r[f'a_{t}']:10.5f}" for t in tiers)
        if len(tiers) >= 1:
            cells += "   " + " ".join(f"{r[f'alpha_{t}']:.3f}/{r[f'signal_{t}']:.1f}" for t in tiers)
        print(f"{r['expert']:>7} {cells}")

    summary = {"layer": a.layer, "tiers": tiers, "n_experts": len(rows_out),
               "adapter_expert_dirs": int(n_eff), "lam": lam, "rows": [r["rows"] for r in rows_out]}
    print()
    for t in tiers:
        pr = [r[f"e_{t}"] for r in rows_out]
        ab = [r[f"a_{t}"] for r in rows_out]
        same = int(np.sum(np.argmin(np.array([pr, ab]), axis=0) == 0))  # only meaningful across tiers
        rho = spearman(pr, ab)
        ratio = float(np.mean(np.array(ab) / np.array(pr)))
        # how often the tier ranking across the two tiers changes per expert
        summary[t] = {"spearman_pristine_vs_ablated": rho, "mean_abl_over_pristine": ratio,
                      "mean_alpha": float(np.mean([r[f"alpha_{t}"] for r in rows_out])),
                      "mean_signal_pct": float(np.mean([r[f"signal_{t}"] for r in rows_out]))}
        print(f"{t:>8}: spearman(pristine, ablated) {rho:+.3f}   "
              f"mean ablated/pristine error {ratio:.3f}   "
              f"alpha {summary[t]['mean_alpha']:.3f}  signal {summary[t]['mean_signal_pct']:.1f}%")

    if len(tiers) >= 2:
        flip_pr = sum(1 for r in rows_out if np.argmin([r[f"e_{t}"] for t in tiers]) != 0)
        flip_ab = sum(1 for r in rows_out if np.argmin([r[f"a_{t}"] for t in tiers]) != 0)
        summary["argmin_flips"] = {"pristine": int(flip_pr), "ablated": int(flip_ab),
                                   "of": len(rows_out)}
        print(f"\nargmin ({tiers[0]} vs {tiers[1]}): pristine picks "
              f"{tiers[0]} for {len(rows_out)-flip_pr}/{len(rows_out)}, "
              f"ablated for {len(rows_out)-flip_ab}/{len(rows_out)}")

    if a.out:
        json.dump({"summary": summary, "experts": rows_out}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
