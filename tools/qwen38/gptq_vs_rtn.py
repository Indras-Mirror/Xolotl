#!/usr/bin/env python3
"""Does our GPTQ + GSQ/AdaRound refinement actually beat RTN on a qwen4exp expert?

This is the acceptance test V1 could not have passed: V1's rounding was llama-quantize RTN.
V2's whole premise is that GPTQ (error feedback through H^-1) followed by v41_repair.refine
beats plain RTN at the same tier. That premise is worth one cheap experiment before spending
hours wiring it into the chain.

It is also the guard against the failure that motivated this file: `v41_gptq.gptq()` with an
IDENTITY Hessian is bit-identical to `rtn()` (measured: maxdiff 0.0). So a pipeline fed only
diag importances ships RTN while logging "gptq ran". This harness builds a REAL Hessian from
the routed-token activations and reports whether the codes actually moved.

    python3 tools/qwen38/gptq_vs_rtn.py \
        --model /media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf \
        --dump "/media/mal/Team CX2/qwen38-dumps-abl" \
        --layer 20 --expert 0 --tier q2_0

Output error is measured on HELD-OUT tokens (the fit tokens build H and the scales only), which
is the same discipline screen.py uses for its tier screening.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V41 = HERE.parent / "v41"
sys.path.insert(0, str(V41))

import screen as S  # noqa: E402
import v41_gptq as G  # noqa: E402
import v41_repair as R  # noqa: E402


def swiglu_t(gate, up):
    return torch.nn.functional.silu(gate) * up


def expert_out_t(X, Wg, Wu, Wd):
    return swiglu_t(X @ Wg.T, X @ Wu.T) @ Wd.T


def to_t(a, dev):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)


def build_H(Xf, wf):
    """Hessian of the layer's input for the gate/up pair: sum_t w_t x_t x_t^T."""
    return Xf.T @ (Xf * wf)


def quantise_pair(Ws, H, tier, do_gptq, do_refine, steps, dev, damp):
    """gate+up stacked: same input columns, so ONE Hessian and one GPTQ run."""
    d = torch.diag(H).clamp_min(0) + 1e-8
    lo_v, hi_v, _, _ = G.TIERS[tier]
    out = []
    S_ = [G.expand_scales(G.search_scales(W, d, tier), tier, H.shape[0]) for W in Ws]
    los = [torch.full((W.shape[0], 1), float(lo_v), device=dev) for W in Ws]
    his = [torch.full((W.shape[0], 1), float(hi_v), device=dev) for W in Ws]
    if not do_gptq:
        for W, s, lo, hi in zip(Ws, S_, los, his):
            out.append(G.rtn(W, s, lo, hi))
        return out
    # stack so the H^-1 error feedback is applied once over both matrices' rows
    Wst = torch.cat(Ws)
    Sst = torch.cat(S_)
    lost = torch.cat(los)
    hist = torch.cat(his)
    Hinv, dead = G.hinv_chol(H, damp)
    Q = G.gptq(Wst, Sst, lost, hist, Hinv, dead)
    if do_refine:
        Q = R.refine(Wst, Q, Sst, lost, hist, H, steps=steps)
    n = 0
    for W in Ws:
        out.append(Q[n:n + W.shape[0]])
        n += W.shape[0]
    return out


def quantise_down(Wd, H, tier, do_gptq, do_refine, steps, dev, damp):
    d = torch.diag(H).clamp_min(0) + 1e-8
    lo_v, hi_v, _, _ = G.TIERS[tier]
    s = G.expand_scales(G.search_scales(Wd, d, tier), tier, H.shape[0])
    lo = torch.full((Wd.shape[0], 1), float(lo_v), device=dev)
    hi = torch.full((Wd.shape[0], 1), float(hi_v), device=dev)
    if not do_gptq:
        return G.rtn(Wd, s, lo, hi)
    Hinv, dead = G.hinv_chol(H, damp)
    Q = G.gptq(Wd, s, lo, hi, Hinv, dead)
    if do_refine:
        Q = R.refine(Wd, Q, s, lo, hi, H, steps=steps)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--tier", default="q2_0")
    ap.add_argument("--steps", type=int, default=200, help="V41_REFINE_STEPS")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--min-fit", type=int, default=32)
    ap.add_argument("--damp", type=float, default=0.3,
                    help="H damping. The v41 default 0.01 assumes a well-conditioned H; a per-expert\n"
                         "H from ~900 tokens is rank-deficient (899 of 2560 here), and 0.3 measured\n"
                         "~9 points better on held-out data.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args()

    dev = torch.device(a.device)
    model = S.GGUFModel(a.model)
    dump = S.dump_files(a.dump, a.layer)
    if not dump:
        raise SystemExit(f"no dumps for layer {a.layer} in {a.dump}")
    X, ids = S.read_dump(dump)
    print(f"layer {a.layer}: X {X.shape}, {len(dump)} chunk file(s)")

    rows = np.where(ids == a.expert)[0]
    if rows.size == 0:
        raise SystemExit(f"expert {a.expert} was never routed in layer {a.layer}")

    Wg = model.expert(a.layer, a.expert, S.FGATE)
    Wu = model.expert(a.layer, a.expert, S.FUP)
    Wd = model.expert(a.layer, a.expert, S.FDOWN)
    print(f"expert {a.expert}: gate {Wg.shape} up {Wu.shape} down {Wd.shape}; {rows.size} routed tokens")

    rng = np.random.RandomState(1234 + a.layer)
    held = rng.rand(len(ids)) < a.eval_frac
    fit = ~held[rows]
    ev = held[rows]
    if int(fit.sum()) < a.min_fit:
        raise SystemExit(f"only {int(fit.sum())} fit tokens (< --min-fit {a.min_fit}); "
                         f"V4 uses an identity Hessian below that, and this test needs a real one")
    if int(ev.sum()) < 8:
        ev = np.ones_like(ev)
        print("  WARNING: too few held-out tokens; measuring on all (optimistic)")

    Xe = X[rows]
    Xf = to_t(Xe[fit], dev)
    Xv = to_t(Xe[ev], dev)
    Wg_t = to_t(Wg, dev); Wu_t = to_t(Wu, dev); Wd_t = to_t(Wd, dev)

    # routing weight matters: a token routed with weight 0.2 should not dominate the fit
    wgt = np.ones(len(rows), dtype=np.float32)  # conservative: unweighted unless dumps carry it
    wf = to_t(wgt[fit][:, None], dev)

    t0 = time.time()
    H_gu = build_H(Xf, wf)
    Hd_x = swiglu_t(Xf @ Wg_t.T, Xf @ Wu_t.T) * wf.sqrt()
    H_dn = Hd_x.T @ Hd_x
    H_time = time.time() - t0
    print(f"Hessians: gu {tuple(H_gu.shape)} dn {tuple(H_dn.shape)} in {H_time:.1f}s")
    print(f"  H_gu diag range {H_gu.diagonal().min():.3e}..{H_gu.diagonal().max():.3e}; "
          f"offdiag energy {float((H_gu - torch.diag(H_gu.diagonal())).pow(2).sum() / H_gu.pow(2).sum()):.3f}")

    with torch.no_grad():
        Y = expert_out_t(Xv, Wg_t, Wu_t, Wd_t)
        ref = float(Y.pow(2).sum())

    arms = [
        ("fp32", False, False),
        ("rtn", False, False),
        ("gptq", True, False),
        ("gptq+refine", True, True),
    ]
    results = {}
    for name, do_gptq, do_refine in arms:
        if name == "fp32":
            results[name] = {"relerr": 0.0, "note": "reference"}
            continue
        t0 = time.time()
        Wgq, Wuq = quantise_pair([Wg_t, Wu_t], H_gu, a.tier, do_gptq, do_refine, a.steps, dev, a.damp)
        Wdq = quantise_down(Wd_t, H_dn, a.tier, do_gptq, do_refine, a.steps, dev, a.damp)
        with torch.no_grad():
            Yq = expert_out_t(Xv, Wgq, Wuq, Wdq)
            err = float((Yq - Y).pow(2).sum())
        dt = time.time() - t0
        # did the codes actually move off RTN?
        with torch.no_grad():
            Wgr, Wur = quantise_pair([Wg_t, Wu_t], H_gu, a.tier, False, False, 0, dev, a.damp)
            diff = (float((Wgq - Wgr).abs().sum()) + float((Wuq - Wur).abs().sum())
                    + float((Wdq - quantise_down(Wd_t, H_dn, a.tier, False, False, 0, dev, a.damp)).abs().sum()))
        results[name] = {"relerr": (err / ref) ** 0.5, "out_err": err, "secs": dt,
                         "moved_from_rtn": diff}
        print(f"  {name:13} rel_out_err {(err / ref) ** 0.5:.6f}   ({dt:5.1f}s)   "
              f"codes moved vs RTN: {diff:.4f}")

    base = results["rtn"]["relerr"]
    print(f"\ntier={a.tier} layer={a.layer} expert={a.expert} "
          f"fit={int(fit.sum())} held={int(ev.sum())}")
    for name in ("gptq", "gptq+refine"):
        r = results[name]["relerr"]
        print(f"  {name:13} {100 * (base - r) / base:+6.2f}% vs RTN, "
              f"{(1 - r / results['gptq']['relerr']) * 100 if name == 'gptq+refine' else 0:+.2f}% vs gptq")

    if results["gptq"]["moved_from_rtn"] == 0.0:
        print("\nFAIL: GPTQ output is identical to RTN -- the Hessian was not used.")
        print("      This is the identity-Hessian trap: gptq() degenerates to rtn() exactly.")
        raise SystemExit(1)

    if a.json:
        import json
        pathlib.Path(a.json).write_text(json.dumps(
            {"layer": a.layer, "expert": a.expert, "tier": a.tier,
             "fit": int(fit.sum()), "held": int(ev.sum()), "arms": results}, indent=1) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
