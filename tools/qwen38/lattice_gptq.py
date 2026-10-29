#!/usr/bin/env python3
"""GPTQ over ggml's IQ4_NL codebook instead of an integer grid.

WHY. `v41_gptq.gptq` rounds to a uniform integer grid, so it only works on the formats that ARE
uniform grids (ptq1, q2_0, q3_k). The formats that actually win on qwen4exp are ggml's codebook
types, and they are not grids. Measured held-out expert output error (QFN-V2-PIPELINE §11d):

    RTN q2_0 1.234   GPTQ q2_0 0.320   RTN ptq1 0.719   GPTQ ptq1 0.408
    IQ2_XS   0.400   IQ3_XXS   0.248   IQ4_NL   0.122

So GPTQ helps enormously but only where a grid exists, and the grids are worse formats than the
codebooks at the same size. This module closes that: the same error-feedback loop, with the
rounding operator replaced by "nearest codebook entry".

TWO THINGS THAT MAKE IT WORK, BOTH LEARNED THE HARD WAY:

1. `d` MUST be held fixed. The IQ types derive the block scale from the block maximum, so
   `dequantize(x)` is not a fixed point of `quantize()` (repeated snapping drifts 0.18419 ->
   0.21133 on iq3_xxs) and a mixture of old and new entries is not representable. Quantising a
   perturbed W, or GPTQ-in-float followed by a single snap, is 42-176% WORSE than plain RTN.
2. Codes must be written back directly (iq4nl_codec.encode), never re-quantised. Holding `d` and
   emitting codes makes the result exactly representable by construction.

Measured on real qwen4exp experts vs RTN IQ4_NL: +71.5% (L5/e0), +40.7% (L20/e0), +38.1% (L30/e7),
and +44.3% on an independent held-out split. 6.6% of codes change. ~2.3 s per expert for all three
matrices.

    python3 tools/qwen38/lattice_gptq.py --selftest
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v41"))

import iq4nl_codec as C          # noqa: E402
import screen as S               # noqa: E402
import v41_gptq as G             # noqa: E402


def _kvals(dev):
    return torch.as_tensor(C.KVALS.astype(np.float32), device=dev)


def rtn_codes(SH, W, type_name="iq4_nl"):
    """Quantise with ggml (RTN) -> (d, codes) at `type_name`. Only iq4_nl has a codec so far."""
    if type_name != "iq4_nl":
        raise SystemExit(f"no codec for {type_name}; only iq4_nl so far")
    raw = C._quantize_bytes(SH, W.detach().cpu().numpy(), type_name)
    return C.parse(raw, W.shape[0], W.shape[1])


def lattice_gptq(SH, W, H, damp=0.3, block=128, type_name="iq4_nl"):
    """GPTQ error feedback with a nearest-codebook rounding operator, holding d fixed.

    W [rows, cols] fp32 (original weights), H [cols, cols] the input Hessian (fit tokens only).
    Returns (d [rows, nblk] fp32, codes [rows, cols] uint8) -- a legal IQ4_NL tensor.
    """
    d_np, codes0 = rtn_codes(SH, W, type_name)
    d = torch.as_tensor(d_np, device=W.device)
    rows, cols = W.shape
    KV = _kvals(W.device)
    # per-element scale: every 32 consecutive columns share one block scale
    S_eff = d[:, torch.arange(cols, device=W.device) // C.QK4_NL]
    Hinv, dead = G.hinv_chol(H, damp)
    Wc = W.clone()
    Wc[:, dead] = 0
    codes = codes0.copy()
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = Wc[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            w = W1[:, i]
            s = S_eff[:, c]
            # nearest entry of the 16-value codebook, in units of the fixed block scale
            j = ((w / s).unsqueeze(1) - KV.unsqueeze(0)).abs().argmin(dim=1)
            q = s * KV[j]
            codes[:, c] = j.detach().cpu().numpy().astype(np.uint8)
            e = (w - q) / Hi1[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = e
        Wc[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return d.detach().cpu().numpy(), codes


def selftest(shim="/tmp/libgqshim.so"):
    """Prove the output is writable AND better, on real weights and real routed tokens."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SH = S.GGML(shim)
    model = S.GGUFModel("/media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf")
    model_out = lambda X, a, b, c: (torch.nn.functional.silu(X @ a.T) * (X @ b.T)) @ c.T

    L, ex = 20, 0
    X, ids = S.read_dump(S.dump_files("/media/mal/Team CX2/qwen38-dumps-abl", L))
    rows = np.where(ids == ex)[0]
    Xe = X[rows]
    Wv = [torch.as_tensor(model.expert(L, ex, p), device=dev)
          for p in (S.FGATE, S.FUP, S.FDOWN)]

    rng = np.random.RandomState(1234 + L)
    held = rng.rand(len(ids)) < 0.25
    fit = ~held[rows]
    Xf = torch.as_tensor(Xe[fit], device=dev)
    H = Xf.T @ Xf

    d0, c0 = rtn_codes(SH, Wv[0])
    d1, c1 = lattice_gptq(SH, Wv[0], H)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    check("d is unchanged from RTN (block scales untouched)", np.array_equal(d0, d1))
    check("codes actually changed", not np.array_equal(c0, c1))
    check("all codes in 0..15", int(c1.min()) >= 0 and int(c1.max()) <= 15)
    raw = C.encode(d1, c1, *Wv[0].shape)
    via_ggml = torch.as_tensor(C._dequant(SH, raw, Wv[0].shape[1], Wv[0].shape[0]), device=dev)
    mine = torch.as_tensor(C.dequantize(d1, c1), device=dev)
    # THE shippability check: the bytes we would write must decode through ggml to what we measured
    check("encoded bytes decode through ggml to our values", torch.equal(via_ggml, mine))
    check("byte count matches row_bytes",
          raw.size == C.row_bytes(Wv[0].shape[1]) * Wv[0].shape[0])

    # independent split: rules out the improvement coming from fitting and scoring the same rows
    rng2 = np.random.RandomState(999)
    h2 = rng2.rand(len(ids)) < 0.25
    f2, e2 = ~h2[rows], h2[rows]
    if int(e2.sum()) < 8:
        e2 = np.ones_like(e2)
    Xf2 = torch.as_tensor(Xe[f2], device=dev)
    Xv2 = torch.as_tensor(Xe[e2], device=dev)
    H2 = Xf2.T @ Xf2
    hdn = torch.nn.functional.silu(Xf2 @ Wv[0].T) * (Xf2 @ Wv[1].T)
    Hdn2 = hdn.T @ hdn

    def score(assign):
        with torch.no_grad():
            Y = model_out(Xv2, *Wv)
            ref = float(Y.pow(2).sum())
            Wq = [torch.as_tensor(C.dequantize(*a), device=dev) for a in assign]
            return float(((model_out(Xv2, *Wq) - Y).pow(2).sum() / ref) ** 0.5)

    rtn_assign = [rtn_codes(SH, w) for w in Wv]
    lat_assign = [lattice_gptq(SH, Wv[0], H2), lattice_gptq(SH, Wv[1], H2),
                  lattice_gptq(SH, Wv[2], Hdn2)]
    a, b = score(rtn_assign), score(lat_assign)
    print(f"  independent split: RTN {a:.5f} -> lattice {b:.5f} ({100 * (a - b) / a:+.2f}%)")
    check("lattice beats RTN on an independent split", b < a)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shim", default="/tmp/libgqshim.so")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest(a.shim))
    ap.print_help()


if __name__ == "__main__":
    main()
