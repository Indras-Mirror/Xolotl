#!/usr/bin/env python3
"""lite_calib.py - ONE calibration pass for HXQ3K-lite (Q3_K / Q2_0 mixed routed experts), CPU-friendly.

The teacher chain is v41_calib's layer-streaming forward (DeepSeek's model.py, torch_kernels stand-in, FP4 experts
exactly as trained).  For EVERY routed expert a worker process:
  * fits GPTQ Q2_0 (v41_quant.quant: static diag(H)-weighted scales, damp 1.0) for gate+up on the input Hessian and
    for down on the Hessian of the clamped SwiGLU x routing weight, on the FIT sequences only, and writes its GGUF
    block bytes to OUT/q2/LNN.bin (expert e at e * EXB: gate rows, up rows, down rows - v41_gguf --experts layout);
  * reads the SAME expert's Q3_K bytes from the llama.cpp heretic Q3_K GGUF (= the HXQ3K experts byte for byte);
  * measures the held-out output error sum ||f(x;W) - f(x;Q)||^2 of the four (gate+up, down) combinations
    e33 / e23 / e32 / e22 (first digit gate+up, second down; 3 = Q3_K, 2 = Q2_0) plus the teacher energy y_energy.
OUT/stats/LNN.json collects them; lite_alloc.py picks the map for any size from these numbers (no second pass).

  V41_SRC=<checkpoint> V41_WORK=<local work dir> python3 lite_calib.py run --out OUT [--layers 0-39] [--workers 24]
       [--worker-threads 6] [--threads 64] [--gc]
  (`init` first: v41_calib init = h/L00 from the embedding.)
"""
import argparse, json, os, sys, time, types
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V41 = os.path.dirname(HERE)
sys.path.insert(0, V41)
from lite_config import Q3K_GGUF, ENGRAM, ENGRAM_SMALL
D, FF = 5120, 2304
Q2_ROW_GU, Q2_ROW_DN = D // 64 * 18, FF // 64 * 18          # 1440 / 648 bytes
EXB = 2 * FF * Q2_ROW_GU + D * Q2_ROW_DN                     # 9,953,280 bytes per expert
Q3_MAT = D * FF // 256 * 110                                 # 5,068,800 bytes per matrix


def q3k_offsets():
    """{layer: {"gate": off, "up": off, "down": off}} - absolute file offsets of the stacked Q3_K expert tensors."""
    import v41_gguf as W
    _, tensors, start = W.read_gguf(Q3K_GGUF)
    out = {}
    for name, shape, qtype, offset in tensors:
        p = name.split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            if tuple(shape) not in ((D, FF, 384), (FF, D, 384)) or qtype != 11:
                raise ValueError(f"{name}: expected stacked Q3_K experts, got {shape}, type {qtype}")
            out.setdefault(int(p[1]), {})[p[2].split("_")[1]] = start + offset
    if not out:
        raise ValueError("No stacked Q3_K expert tensors in LITE_Q3K_GGUF")
    return out


# ------------------------------------------------------------------------------------------------ worker side
_W = {}


def _winit(threads, out, q3off):
    import torch
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float32)
    import v41_calib as C
    import v41_quant as Q
    import v41_formats as F
    import v41_gptq as G
    _orig = G.search_scales

    def search_scales_blocked(W, d, tier, rb=128):
        """v41_gptq.search_scales on 128-row blocks (rows are independent): the same scales (checked equal on
        4608x5120), 18x faster on CPU - the 44 candidate passes stay in cache instead of streaming 94 MB each."""
        if W.shape[0] <= rb:
            return _orig(W, d, tier)
        return torch.cat([_orig(W[i:i + rb], d, tier) for i in range(0, W.shape[0], rb)])
    G.search_scales = search_scales_blocked
    _W.update(torch=torch, C=C, Q=Q, F=F, out=out, q3off=q3off, fd={}, q3fd=os.open(Q3K_GGUF, os.O_RDONLY))


def _q3(L, ex):
    F, torch = _W["F"], _W["torch"]
    off = _W["q3off"][L]
    m = []
    for role, (R, Cc) in (("gate", (FF, D)), ("up", (FF, D)), ("down", (D, FF))):
        raw = os.pread(_W["q3fd"], Q3_MAT, off[role] + ex * Q3_MAT)
        assert len(raw) == Q3_MAT
        m.append(torch.from_numpy(F.decode(F.Q3_K, raw, R * Cc).reshape(R, Cc).astype(np.float32)))
    return m


def _work(t):
    """One expert. t: L, ex, packed FP4 weights + scales, routed rows (bf16) / weights / fit mask or None, lim."""
    torch, C, Q, F = _W["torch"], _W["C"], _W["Q"], _W["F"]
    t0 = time.time()
    L, ex, lim = t["L"], t["ex"], t["lim"]
    K = C.K
    T = torch.from_numpy
    E8 = lambda a: T(a).view(torch.float8_e8m0fnu)
    W1 = K.dequant_fp4_weight(T(t["w1"]), E8(t["s1"])).float()
    W3 = K.dequant_fp4_weight(T(t["w3"]), E8(t["s3"])).float()
    W2 = K.dequant_fp4_weight(T(t["w2"]), E8(t["s2"])).float()
    st = {"n": 0}
    xe = t.get("xe")
    teacher = None
    if xe is not None:
        xe = T(xe).view(torch.bfloat16)
        ge, fit = T(t["ge"]), T(t["fit"])
        # the teacher output exactly as v41_calib computes it (DeepSeek's Expert.forward on the FP4 weights, the
        # torch_kernels FP8-activation path, bf16 default dtype, ROWS chunks) - here instead of the main process
        E = _W.get("expert")
        if E is None or E.swiglu_limit != lim:
            E = _W["expert"] = C.M.Expert(D, FF, dtype=torch.float4_e2m1fn_x2, swiglu_limit=lim)
        for lin, w, sc in ((E.w1, "w1", "s1"), (E.w3, "w3", "s3"), (E.w2, "w2", "s2")):
            lin.weight.data.view(torch.uint8).copy_(T(t[w]))
            lin.scale.data.view(torch.uint8).copy_(T(t[sc]))
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.inference_mode():
                teacher = torch.cat([E(xe[c:c + C.ROWS], ge[c:c + C.ROWS]) for c in range(0, len(xe), C.ROWS)])
        finally:
            torch.set_default_dtype(torch.float32)
        xf, gf = xe[fit], ge[fit]
        n_fit = int(fit.sum())
        st.update(n=int(len(xe)), n_fit=n_fit, n_eval=int((~fit).sum()))
    do = xe is not None and st["n_fit"] >= t["min_fit"]
    if do:
        H1, H2 = C.act_hessians(xf, W1, W3, gf, lim)
    else:
        H1, H2 = torch.eye(D), torch.eye(FF)
    (g1, g3) = Q.quant([W1, W3], H1, "q2_0", 1.0, do)
    (g2,) = Q.quant([W2], H2, "q2_0", 1.0, do)
    b = [F.encode_tier(q.numpy(), s.numpy(), "q2_0").tobytes() for q, s in (g1, g3, g2)]
    assert [len(x) for x in b] == [FF * Q2_ROW_GU, FF * Q2_ROW_GU, D * Q2_ROW_DN]
    fd = _W["fd"].get(L)
    if fd is None:
        fd = _W["fd"][L] = os.open(os.path.join(_W["out"], "q2", f"L{L:02d}.bin"), os.O_WRONLY)
    os.pwrite(fd, b"".join(b), ex * EXB)
    st["gptq"] = bool(do)
    if xe is not None:
        r1, r3, r2 = _q3(L, ex)
        xv, gv = xe[~fit], ge[~fit]
        if len(xv) == 0:              # nothing held out: measured on the fit tokens (flagged)
            xv, gv, st["eval_on_fit"] = xf, gf, True
        Y = C.expert_f(xv, W1, W3, W2, gv, lim)
        st["y_energy"] = float(Y.square().sum())
        q1, q3, q2 = g1[0], g3[0], g2[0]
        for k, (a1, a3, a2) in {"e33": (r1, r3, r2), "e23": (q1, q3, r2), "e32": (r1, r3, q2), "e22": (q1, q3, q2)}.items():
            st[k] = C.sq_err(xv, a1, a3, a2, gv, lim, Y)
        # weight-space control (no activations): relative Frobenius error of each format
        st["w_q3"] = float(((r1 - W1).square().sum() + (r3 - W3).square().sum() + (r2 - W2).square().sum()) /
                           (W1.square().sum() + W3.square().sum() + W2.square().sum()))
        st["w_q2"] = float(((q1 - W1).square().sum() + (q3 - W3).square().sum() + (q2 - W2).square().sum()) /
                           (W1.square().sum() + W3.square().sum() + W2.square().sum()))
    st["sec"] = round(time.time() - t0, 2)
    return L, ex, st, (None if teacher is None else teacher.contiguous().view(torch.int16).numpy())


# ------------------------------------------------------------------------------------------------ main side
def setup_calib():
    import v41_calib as C
    C.ABL = types.SimpleNamespace(manifest_id=lambda: "heretic-1017 (baked into the checkpoint)")
    orig = C.shard_file

    def shard_file(name):
        return ENGRAM_SMALL.get(name) or orig(name)
    C.shard_file = shard_file
    big = {v: ENGRAM[k] for k, v in ENGRAM_SMALL.items()}

    class Emb(C.MemmapEngramEmbedding):   # cmd_layer sets SRC from shard_file (the small file): rows from the tables
        def __init__(self, *a, **k):
            p, wn, sn = type(self).SRC
            type(self).SRC = (big.get(p, p), wn, sn)
            super().__init__(*a, **k)
    C.MemmapEngramEmbedding = Emb
    return C


def cmd_init(a):
    C = setup_calib()
    C.cmd_init(a)


def cmd_run(a):
    import torch
    torch.set_num_threads(a.threads)
    C = setup_calib()
    os.makedirs(os.path.join(a.out, "q2"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "stats"), exist_ok=True)
    q3off = q3k_offsets()
    lo, hi = (int(x) for x in a.layers.split("-"))
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_winit, initargs=(a.worker_threads, a.out, q3off))
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)
    for L in range(lo, hi + 1):
        sj = os.path.join(a.out, "stats", f"L{L:02d}.json")
        nxt = os.path.join(a.out, "stats", f"L{L + 1:02d}.json")
        if os.path.exists(sj) and (os.path.exists(C.wpath("h", f"L{L + 1:02d}.bin")) or os.path.exists(nxt)):
            log(f"lite L{L}: done already")
            continue
        qb = os.path.join(a.out, "q2", f"L{L:02d}.bin")
        with open(qb, "ab") as f:
            f.truncate(384 * EXB)
        futs, done_ex, t0 = [], set(), time.time()

        def submit(L, ex, exp, xe=None, ge=None, fit=None, lim=0.0):
            # plain numpy across the process boundary (pickled through the pipe; no torch shared-memory handles)
            t = {"L": L, "ex": ex, "lim": float(lim), "min_fit": a.min_fit,
                 "w1": exp.w1.weight.data.view(torch.uint8).numpy().copy(), "s1": exp.w1.scale.data.view(torch.uint8).numpy().copy(),
                 "w3": exp.w3.weight.data.view(torch.uint8).numpy().copy(), "s3": exp.w3.scale.data.view(torch.uint8).numpy().copy(),
                 "w2": exp.w2.weight.data.view(torch.uint8).numpy().copy(), "s2": exp.w2.scale.data.view(torch.uint8).numpy().copy()}
            if xe is not None:
                t.update(xe=xe.contiguous().view(torch.int16).numpy().copy(), ge=ge.float().numpy().copy(),
                         fit=fit.numpy().copy())
            while sum(not f.done() for f in futs) >= 2 * a.workers:   # bound the rows in flight
                time.sleep(0.05)
            futs.append(pool.submit(_work, t))
            done_ex.add(ex)

        toks = {}

        def hook(L, ex, exp, xe, ge, fit, lim, out, tok=None):
            if not toks:   # the MoE phase: the main process only gathers rows, the workers need the cores
                torch.set_num_threads(a.moe_threads)
            toks[ex] = ns.cur_tok
            submit(L, ex, exp, xe, ge, fit, lim)
            return None, {}      # defer_teacher: the worker's teacher output is added to Y in moe_done

        def moe_done(L, blk, stats):
            for ex in range(384):
                if ex not in done_ex:
                    submit(L, ex, blk.ffn.experts[ex])
            res = {}
            Y = ns.moe_Y
            for f in futs:
                _, ex, st, y = f.result()
                res[ex] = st
                if y is not None:
                    Y.index_add_(0, toks[ex], torch.from_numpy(y).view(torch.bfloat16).float())
            for ex, st in res.items():
                stats[ex].update(st)
            e = {k: sum(s.get(k, 0.0) for s in res.values()) for k in ("e33", "e23", "e32", "e22", "y_energy")}
            y = max(e["y_energy"], 1e-30)
            log(f"lite L{L}: experts done in {time.time() - t0:.0f} s; held-out rel err Q3_K {e['e33'] / y:.5f}, "
                f"gu Q2 {e['e23'] / y:.5f}, dn Q2 {e['e32'] / y:.5f}, all Q2 {e['e22'] / y:.5f}; "
                f"worker s/expert {np.mean([s['sec'] for s in res.values()]):.1f}")

        ns = argparse.Namespace(layer=L, batch=a.batch, device="cpu", expert_hook=hook, moe_done_hook=moe_done,
                                no_gptq=True, gptq_experts=0, expert_stride=1, min_fit=a.min_fit, damps=[1.0],
                                screen_only=None, gc=a.gc, defer_teacher=True)
        torch.set_num_threads(a.threads)   # the attention half: every core (the workers are idle)
        C.cmd_layer(ns)
        # v41_calib wrote WORK/stats/LNN.json (with our per-expert numbers merged in): copy it next to the q2 bytes
        with open(qb, "rb") as qfile:
            os.fsync(qfile.fileno())
        j = json.load(open(C.wpath("stats", f"L{L:02d}.json")))
        tmp = sj + ".tmp"
        json.dump(j, open(tmp, "w"))
        os.replace(tmp, sj)
        log(f"lite L{L}: layer done in {time.time() - t0:.0f} s")
    pool.shutdown()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0-39")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--worker-threads", type=int, default=6)
    p.add_argument("--threads", type=int, default=64, help="main process torch threads in the attention half")
    p.add_argument("--moe-threads", type=int, default=16, help="main process torch threads in the MoE phase")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--min-fit", type=int, default=32)
    p.add_argument("--gc", action="store_true")
    a = ap.parse_args()
    {"init": cmd_init, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
