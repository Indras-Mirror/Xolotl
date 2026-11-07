#!/usr/bin/env python3
"""v41_repair.py - quant-repair adapter PROTOTYPE: how much of the GSQ experts' error can a small correction remove?

Mal 2026-10-17: the quant (not the abliteration / API revision) is what costs API agreement - on DwarfStar's 100 short
fixtures the FP8 original scores 93.65 % top-1, the GSQ quant 86.13 % (exact routing) - and the quant's thinking
over-verifies (abc398_f: the algorithm by 43 % of a 52K-token trace, then hand-checks).  Before building an adapter,
measure what each shape could buy, on the layers that matter (card: 4-19 and 32 carry the error).

A STUDENT chain through DeepSeek's model.py (v41_calib.cmd_layer, layer by layer): the routed experts are the GSQ
file's own bytes (decoded on the GPU, bit-identical to v41_formats.decode), the dense half the FP8 source (the GGUF's
Q8_0 is rel 5e-3 of it).  At --measure layers, on the student's own inputs, against the FP4 teacher experts:
  base     held-out routed-output error  sum ||Y_teacher - Y_gsq||^2 / sum ||Y_teacher||^2   (the card's metric)
  lqer_r   per-expert low-rank weight correction (LQER: SVD of (W_fp4 - W_gsq) diag(s), s = rms of the expert's FIT
           inputs; the down projection's s from its SwiGLU input), rank r - an adapter inside each expert
  dense_r  one linear correction per layer on the MoE input, reduced-rank ridge fit E ~ X A B on FIT tokens - what an
           adapter on the always-on side (shared expert / attention output) could at best do linearly
All errors on HELD-OUT sequences only.

  v41_repair.py prep  --rwork DIR [--eval 4 --fit 6]      a subset of the calibration sequences (CPU)
  v41_repair.py init  --rwork DIR                         h/L00 from the embedding
  v41_repair.py layer L --rwork DIR [--measure 13,32] [--ranks 8,32] [--gc]
  v41_repair.py selftest                                   GPU decoders == v41_formats.decode on real experts
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v41_calib as C  # noqa: E402
import v41_formats as F  # noqa: E402
import v41_gptq as G  # noqa: E402
import v41_quant as VQ  # noqa: E402

GGUF = os.environ.get("V41_GSQ_GGUF", "/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/V41-GSQ.gguf")   # the GSQ file whose experts are read
CALIB = "/media/Crucial1TB/v41-calib"
FF = 2304


# ------------------------------------------------------------------------------------------- GPU decoders
def _f16(b):
    return b.contiguous().view(torch.float16).float()


def decode_t(t, raw):
    """uint8 tensor (any device) -> float32 flat, the same math as v41_formats.decode."""
    if t == F.Q2_0:
        b = raw.view(-1, 18)
        d = _f16(b[:, :2])
        qs = b[:, 2:].int()
        q = torch.stack([(qs >> (2 * k)) & 3 for k in range(4)], -1).reshape(-1, 64).float() - 1
        return (q * d).reshape(-1)
    if t == F.PTQ1_0:
        b = raw.view(-1, 28)
        d = _f16(b[:, 26:28])
        p3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=raw.device)

        def trits(x, k):
            return ((((x[:, None, :].int() * p3[:k, None]) & 0xFF) * 3) >> 8).float()
        v = torch.empty((b.shape[0], 128), dtype=torch.float32, device=raw.device)
        v[:, 0:80] = trits(b[:, 0:16], 5).reshape(-1, 80)
        v[:, 80:120] = trits(b[:, 16:24], 5).reshape(-1, 40)
        v[:, 120:128] = trits(b[:, 24:26], 4).reshape(-1, 8)
        return ((v - 1) * d).reshape(-1)
    if t == F.Q3_K:
        b = raw.view(-1, 110)
        hm, qs, raw12 = b[:, 0:32].int(), b[:, 32:96].reshape(-1, 2, 32).int(), b[:, 96:108].int()
        d = _f16(b[:, 108:110])
        a, c, tmp = raw12[:, 0:4], raw12[:, 4:8], raw12[:, 8:12]
        s = torch.cat([(a & 15) | ((tmp & 3) << 4), (c & 15) | (((tmp >> 2) & 3) << 4),
                       (a >> 4) | (((tmp >> 4) & 3) << 4), (c >> 4) | (((tmp >> 6) & 3) << 4)], 1)
        y = torch.empty((b.shape[0], 256), dtype=torch.float32, device=raw.device)
        for h in range(2):
            for j in range(4):
                low = (qs[:, h, :] >> (2 * j)) & 3
                hb = (hm >> (4 * h + j)) & 1
                e0 = 128 * h + 32 * j
                y[:, e0:e0 + 32] = low.float() - torch.where(hb == 1, 0, 4).float()
        return (y * d * (torch.repeat_interleave(s, 16, dim=1) - 32).float()).reshape(-1)
    raise ValueError(t)


class GsqLayer:
    """One layer's GSQ experts, read from the GGUF at the offsets v41_gguf wrote (<gguf>.experts.json)."""

    def __init__(self, L, dev, path=None):
        path = path or GGUF
        lay = json.load(open(path + ".experts.json"))["layers"][str(L)]
        self.gt, self.dt, self.off, self.dev = lay["gu_type"], lay["d_type"], lay["off"], dev
        self.fd = os.open(path, os.O_RDONLY)

    def weights(self, ex):
        o, out = self.off[str(ex)], []
        for role, t, shape in (("gate", self.gt[ex], (FF, C.D)), ("up", self.gt[ex], (FF, C.D)),
                               ("down", self.dt[ex], (C.D, FF))):
            n = F.nbytes(t, (shape[1], shape[0]))
            raw = torch.frombuffer(bytearray(os.pread(self.fd, n, o[role])), dtype=torch.uint8).to(self.dev)
            out.append(decode_t(t, raw).view(shape))
        return out


# ------------------------------------------------------------------------------------------- measurement
N_PROTO = int(os.environ.get("V41_PROTO", "24"))   # experts per measure layer for the refine / Hadamard prototypes


def refine(W, Q, S, lo, hi, H, steps=200):
    """GSQ/AdaRound-style refinement of a GPTQ result: per weight, a learned choice between its code c and the next
    code towards the original value, Gumbel-free (rectified sigmoid + annealed binary regulariser), loss = the
    output-error quadratic tr(dW H dW^T).  Returns the refined weights (still on the grid)."""
    with torch.inference_mode(False), torch.enable_grad():   # cmd_layer runs under inference_mode
        return _refine(W.clone(), Q.clone(), S.clone(), lo, hi, H.clone(), steps)


def _refine(W, Q, S, lo, hi, H, steps):
    torch.backends.cuda.matmul.allow_tf32 = True
    H = H + 1.0 * torch.diag(H).mean() * torch.eye(H.shape[0], device=H.device)   # GPTQ's damp 1.0: no overfit to the fit tokens
    c = torch.round(Q / S)
    r = W / S
    d = torch.sign(r - c)
    c2 = (c + d).clamp(lo, hi)
    step = (c2 - c) * S                                    # 0 where no move is possible
    V = torch.full_like(W, -2.0, requires_grad=True)      # h(V) ~ 0.04: start at the GPTQ answer
    opt = torch.optim.Adam([V], lr=0.05)
    norm = float(torch.einsum("ij,jk,ik->", W, H, W))
    for t in range(steps):
        h = (torch.sigmoid(V) * 1.2 - 0.1).clamp(0, 1)
        dW = W - (Q + h * step)
        loss = torch.einsum("ij,jk,ik->", dW, H, dW) / norm
        if t > steps // 5:
            beta = 20 - 18 * (t - steps // 5) / (steps - steps // 5)
            loss = loss + 1e-4 * (1 - (2 * h - 1).abs().pow(beta)).sum() / h.numel() * 100
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = (torch.sigmoid(V) * 1.2 - 0.1).clamp(0, 1)
        return Q + (h >= 0.5).float() * step


def hadamard_rot(n, dev, seed=0):
    """Block-diagonal randomized Hadamard (128 blocks, +-1 signs), orthonormal [n, n]."""
    Hb = torch.ones(1, 1)
    while Hb.shape[0] < 128:
        Hb = torch.cat([torch.cat([Hb, Hb], 1), torch.cat([Hb, -Hb], 1)], 0)
    Hb = Hb / Hb.shape[0] ** 0.5
    g = torch.Generator().manual_seed(seed)
    sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    R = torch.block_diag(*[Hb] * (n // 128)) * sg[:, None]
    return R.to(dev)


def lqer(W, Wq, s, r):
    """Rank-r correction of Wq towards W, weighted by the input scale s [in]: (W - Wq) ~ (U S V^T) / s."""
    U, S, Vh = torch.linalg.svd((W - Wq) * s, full_matrices=False)
    return Wq + (U[:, :r] * S[:r]) @ (Vh[:r] / s)


class Measure:
    """expert_hook for C.cmd_layer: student = GSQ experts (propagated); at a measure layer also the error stats."""

    def __init__(self, a, L):
        self.a, self.L, self.gsq = a, L, GsqLayer(L, a.device)
        self.on = L in a.measure
        self.ranks = a.ranks
        self.acc = {"base": 0.0, "energy": 0.0, **{f"lqer{r}": 0.0 for r in self.ranks}}
        self.proto = {"n": 0, "energy": 0.0}
        self.R1, self.R2 = (hadamard_rot(C.D, a.device), hadamard_rot(FF, a.device, 1)) if self.on else (None, None)
        self.E = None

    def expert(self, L, ex, exp, xe, ge, fit, lim, out_teacher):
        W1q, W3q, W2q = self.gsq.weights(ex)
        out = torch.empty_like(out_teacher)
        for c in range(0, len(xe), C.ROWS):
            out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], W1q, W3q, W2q, ge[c:c + C.ROWS], lim).to(out.dtype)
        st = {"gu": int(self.gsq.gt[ex]), "dn": int(self.gsq.dt[ex])}
        if self.on:
            X, wts, idx = self.a.moe_ctx
            tok, _ = torch.where(idx == ex)
            if self.E is None:
                self.E = torch.zeros((X.shape[0], C.D), dtype=torch.float32, device=X.device)
                self.Yt = torch.zeros_like(self.E)
            self.E.index_add_(0, tok, out_teacher.float() - out.float())
            self.Yt.index_add_(0, tok, out_teacher.float())
            hv = ~fit
            if hv.any():
                xv, gv, Yv = xe[hv], ge[hv], out_teacher[hv].float()
                self.acc["energy"] += float(Yv.square().sum())
                self.acc["base"] += float((Yv - out[hv].float()).square().sum())
                if fit.sum() >= 8:
                    W1, W3, W2 = (C.K.dequant_fp4_weight(m.weight, m.scale) for m in (exp.w1, exp.w3, exp.w2))
                    xf = xe[fit].float()
                    s1 = xf.square().mean(0).sqrt().clamp_min(1e-6)
                    g1, u1 = xf @ W1.T, xf @ W3.T
                    if lim > 0:
                        g1, u1 = g1.clamp(max=lim), u1.clamp(-lim, lim)
                    hf = torch.nn.functional.silu(g1) * u1 * ge[fit].float()     # W2's input, as expert_f
                    s2 = hf.square().mean(0).sqrt().clamp_min(1e-6)
                    del g1, u1, hf
                    if ex in (0, 1, 2):   # orientation / decode sanity: the GSQ tiers' rel weight error is 0.18-0.45
                        st["w_rel_err"] = [float((W - Wq).norm() / W.norm()) for W, Wq in ((W1, W1q), (W3, W3q), (W2, W2q))]
                        C.log(f"L{L} e{ex}: rel weight err gate/up/down {st['w_rel_err']}")
                    for r in self.ranks:
                        Yr = C.expert_f(xv, lqer(W1, W1q, s1, r), lqer(W3, W3q, s1, r), lqer(W2, W2q, s2, r), gv, lim)
                        self.acc[f"lqer{r}"] += float((Yv - Yr.float()).square().sum())
                    if self.proto["n"] < N_PROTO and fit.sum() >= 64 and len(xv) >= 16:
                        self.prototypes(ex, W1, W3, W2, xe[fit], ge[fit], xv, gv, Yv, lim)
                else:
                    for r in self.ranks:
                        self.acc[f"lqer{r}"] += float((Yv - out[hv].float()).square().sum())
        return out, st

    def prototypes(self, ex, W1, W3, W2, xf, gf, xv, gv, Yv, lim):
        """Same expert, same fit / held-out tokens: GPTQ at each tier (this file's method) vs + refinement vs on a
        Hadamard-rotated basis; held-out output error of each, summed over the prototype experts."""
        P, e = self.proto, lambda Y: float((Yv - Y.float()).square().sum())
        P["n"] += 1
        P["energy"] += float(Yv.square().sum())
        H1, H2 = C.act_hessians(xf, W1, W3, gf, lim)
        R1, R2 = self.R1, self.R2
        H1r, H2r = R1.T @ H1 @ R1, R2.T @ H2 @ R2
        for tier in ("ptq1", "q2_0", "q3_k"):
            lo, hi, _, _ = G.TIERS[tier]
            (a1, s1), (a3, s3) = VQ.quant([W1, W3], H1, tier, 1.0, True)
            ((a2, s2),) = VQ.quant([W2], H2, tier, 1.0, True)
            P[f"gptq_{tier}"] = P.get(f"gptq_{tier}", 0.0) + e(C.expert_f(xv, a1, a3, a2, gv, lim))
            S13 = torch.cat([G.expand_scales(s1, tier, C.D), G.expand_scales(s3, tier, C.D)])
            S2 = G.expand_scales(s2, tier, FF)
            Wst, Qst = torch.cat([W1, W3]), torch.cat([a1, a3])
            r13 = refine(Wst, Qst, S13, lo, hi, H1)
            r2 = refine(W2, a2, S2, lo, hi, H2)
            P[f"refine_{tier}"] = P.get(f"refine_{tier}", 0.0) + e(C.expert_f(xv, r13[:FF], r13[FF:], r2, gv, lim))
            (b1, _), (b3, _) = VQ.quant([W1 @ R1, W3 @ R1], H1r, tier, 1.0, True)
            ((b2, _),) = VQ.quant([W2 @ R2], H2r, tier, 1.0, True)
            P[f"hadamard_{tier}"] = P.get(f"hadamard_{tier}", 0.0) + e(C.expert_f(xv, b1 @ R1.T, b3 @ R1.T, b2 @ R2.T, gv, lim))

    def finish(self, L, blk, stats):
        if not self.on:
            return
        X, _, _ = self.a.moe_ctx
        S = int(np.load(C.wpath("tokens.npy"), mmap_mode="r").shape[1])
        ev = torch.tensor(json.load(open(C.wpath("meta.json")))["is_eval"], device=X.device).repeat_interleave(S)
        res = {k: v / self.acc["energy"] for k, v in self.acc.items() if k != "energy"}
        P = self.proto
        proto = {k: v / P["energy"] for k, v in P.items() if k not in ("n", "energy")} if P["n"] else {}
        # dense_r: reduced-rank ridge E ~ X W on fit tokens, held-out error
        Xf, Ef = X[~ev].float(), self.E[~ev]
        Xv, Ev, Yv = X[ev].float(), self.E[ev], self.Yt[ev]
        G = Xf.T @ Xf
        lam = 1e-3 * float(torch.diag(G).mean())
        Wfull = torch.linalg.solve(G + lam * torch.eye(C.D, device=X.device), Xf.T @ Ef)
        Pf = Xf @ Wfull                                                     # reduced-rank: top directions of the fitted values
        _, V = torch.linalg.eigh(Pf.T @ Pf)
        Vh = V.flip(1).T
        del Pf
        en = float(Yv.square().sum())
        res["base_check"] = float(Ev.square().sum()) / en
        for r in self.ranks + [256]:
            Wr = Wfull @ Vh[:r].T @ Vh[:r]
            res[f"dense{r}"] = float((Ev - Xv @ Wr).square().sum()) / en
        res["dense_full"] = float((Ev - Xv @ Wfull).square().sum()) / en
        if proto:
            C.log(f"repair L{L} prototypes ({P['n']} experts): " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(proto.items())))
        json.dump({"layer": L, "held_out_rel_err": res, "prototypes": proto, "proto_experts": P["n"], "tokens_fit": int((~ev).sum()), "tokens_eval": int(ev.sum())},
                  open(C.wpath("stats", f"repair_L{L:02d}.json"), "w"), indent=1)
        C.log(f"repair L{L}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()))


# ------------------------------------------------------------------------------------------- commands
def set_work(d):
    C.WORK = d
    os.makedirs(d, exist_ok=True)


def cmd_prep(a):
    meta = json.load(open(os.path.join(CALIB, "meta.json")))
    ev = [i for i, e in enumerate(meta["is_eval"]) if e][:a.eval]
    ft = [i for i, e in enumerate(meta["is_eval"]) if not e][:a.fit]
    rows = ft + ev
    tok = np.load(os.path.join(CALIB, "tokens.npy"))[rows]
    hs = np.load(os.path.join(CALIB, "hashes.npy"), mmap_mode="r")[rows]
    np.save(C.wpath("tokens.npy"), tok)
    np.save(C.wpath("hashes.npy"), np.ascontiguousarray(hs))
    json.dump({"n": len(rows), "seq": int(tok.shape[1]), "domains": [meta["domains"][i] for i in rows],
               "is_eval": [i in ev for i in rows], "corpus": meta.get("corpus"), "calib_rows": rows},
              open(C.wpath("meta.json"), "w"))
    C.log(f"prep: {len(ft)} fit + {len(ev)} eval sequences of {tok.shape[1]} from {CALIB}: "
          f"{[meta['domains'][i] for i in rows]}")


def cmd_layer(a):
    m = Measure(a, a.layer)
    a.expert_hook, a.moe_done_hook, a.want_ctx = m.expert, m.finish, True
    a.no_gptq, a.screen_only, a.damps, a.expert_stride, a.min_fit, a.gptq_experts = True, None, [1.0], 1, 32, 0
    C.cmd_layer(a)


def cmd_selftest(a):
    gl = GsqLayer(32, "cuda")
    for ex in (0, 1, 5, 100, 383):
        o = gl.off[str(ex)]
        for role, t, shape in (("gate", gl.gt[ex], (FF, C.D)), ("down", gl.dt[ex], (C.D, FF))):
            n = F.nbytes(t, (shape[1], shape[0]))
            raw = os.pread(gl.fd, n, o[role])
            ref = F.decode(t, raw, shape[0] * shape[1])
            got = decode_t(t, torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()).cpu().numpy()
            assert ref.shape == got.shape and np.array_equal(ref, got), (ex, role, t)
            print(f"L32 e{ex} {role}: type {F.TYPE_NAME[t]} {shape} bit-identical")
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--rwork", required=True); p.add_argument("--eval", type=int, default=4)
    p.add_argument("--fit", type=int, default=6)
    p = sub.add_parser("init"); p.add_argument("--rwork", required=True)
    p = sub.add_parser("layer"); p.add_argument("layer", type=int); p.add_argument("--rwork", required=True)
    p.add_argument("--measure", type=lambda v: [int(x) for x in v.split(",") if x], default=[])
    p.add_argument("--ranks", type=lambda v: [int(x) for x in v.split(",")], default=[8, 32])
    p.add_argument("--device", default="cuda"); p.add_argument("--batch", type=int, default=0)
    p.add_argument("--gc", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd != "selftest":
        set_work(a.rwork)
    {"prep": cmd_prep, "init": C.cmd_init, "layer": cmd_layer, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
