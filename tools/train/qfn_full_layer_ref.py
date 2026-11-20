#!/usr/bin/env python3
"""Whole-layer numpy reference for one QFN gated-delta-net layer, checked against REAL captured activations.

usage: qfn_full_layer_ref.py <pack.gguf> <capture-dir> <layer> [--next FILE]

Reads the residual entering <layer> from <capture-dir>/L<layer>.bin (STRATA_KVA_CAPTURE, fp32 with
STRATA_KVA_CAPTURE_FP32=1: [T, hc*n_embd], stream-major) and recomputes the layer exactly as
llama.cpp's qwen4exp graph builds it (src/models/qwen4exp.cpp, non-fused paths):

    mixed, inj = hc_mix(res, hc_attn_*)          grouped RMSNorm -> silu(down/hc) -> up -> sigmoid gate -> stream mean
    a          = gdn(mixed)                      qkv/z, conv4+silu, l2norm, k->v head tile, chunked GDN, rmsnorm*sigmoid(z), out
    res        = res + a * 2*sigmoid(inj)
    mixed, inj = hc_mix(res, hc_ffn_*)
    f          = moe(mixed) + sigmoid(shexp_gate.mixed) * shexp(mixed)   softmax top-10, renormalised
    res        = res + f * 2*sigmoid(inj)

and compares the result with <capture-dir>/L<layer+1>.bin (or --next FILE). Only the routed experts are dequantized.
This is the gate that says the trainer's understanding of a layer matches the SERVED model on real data.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../third_party/llama.cpp/gguf-py")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import gdn_ref  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


class Pack:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}

    def kv(self, k):
        return self.r.fields[k].contents()

    def f32(self, name):
        """Dequantized tensor as numpy with numpy shape = reversed ggml ne (i.e. W[out, in] for a matrix)."""
        t = self.t[name]
        d = t.data if t.tensor_type.name == "F32" else dequantize(t.data, t.tensor_type)
        ne = [int(v) for v in t.shape]
        return np.asarray(d, dtype=np.float32).reshape(ne[::-1]).astype(np.float64)

    def expert(self, name, e):
        """One expert's matrix [out, in] from a 3-D expert tensor (ggml ne [in, out, n_expert])."""
        t = self.t[name]
        ne = [int(v) for v in t.shape]
        raw = np.asarray(t.data)
        per = raw.reshape(ne[2], -1)[e]
        d = per if t.tensor_type.name == "F32" else dequantize(per.reshape(ne[1], -1), t.tensor_type)
        return np.asarray(d, dtype=np.float32).reshape(ne[1], ne[0]).astype(np.float64)


def rms(x, eps, axis=-1):
    return x / np.sqrt((x * x).mean(axis=axis, keepdims=True) + eps)


def hc_mix(p, pre, res, hc, D, eps):
    """res [T, hc, D] -> mixed [T, D], inject [T, hc]."""
    T = res.shape[0]
    g = p.f32(pre + "_norm.weight").reshape(hc, D)
    xn = rms(res, eps) * g[None]                                   # grouped RMSNorm over each stream
    xf = xn.reshape(T, hc * D)
    lo = silu((xf @ p.f32(pre + "_down.weight").T) / hc)            # [T, 320]
    gate = (lo @ p.f32(pre + "_up.weight").T).reshape(T, hc, D)    # [T, hc, D]
    mixed = (xn * sigmoid(gate)).mean(axis=1)
    if pre + "_inject.weight" not in p.t:                           # the head mixer has no scatter
        return mixed, None
    inj = (xf @ p.f32(pre + "_inject.weight").T) / hc              # [T, hc]
    return mixed, inj


def hc_combine(res, block, inj):
    return res + block[:, None, :] * (2.0 * sigmoid(inj))[:, :, None]


def gdn(p, pre, x, hd, nk, nv, d_conv, eps):
    """x [T, D] -> [T, D]: the qwen4exp linear-attention block."""
    T = x.shape[0]
    n_qkv = 2 * hd * nk + hd * nv
    qkv = x @ p.f32(pre + "attn_qkv.weight").T                       # [T, n_qkv]
    z = x @ p.f32(pre + "attn_gate.weight").T                        # [T, hd*nv]
    convw = p.f32(pre + "ssm_conv1d.weight")                         # [n_qkv, d_conv]
    cin = np.concatenate([np.zeros((d_conv - 1, n_qkv)), qkv], 0)
    y = np.zeros((T, n_qkv))
    for k in range(d_conv):
        y += cin[k:k + T] * convw[:, k][None]
    y = silu(y)

    def heads(a, n):
        return a.reshape(T, n, hd).transpose(2, 1, 0)                # [hd, n, T]

    q = heads(y[:, :hd * nk], nk)
    k_ = heads(y[:, hd * nk:2 * hd * nk], nk)
    v = heads(y[:, 2 * hd * nk:], nv)

    def l2(t):
        return t / np.sqrt((t ** 2).mean(axis=0, keepdims=True) + eps / hd) * (1.0 / np.sqrt(hd))

    q = np.tile(l2(q), (1, nv // nk, 1))                             # ggml_repeat_4d TILES: v head j <- k head j % nk
    k_ = np.tile(l2(k_), (1, nv // nk, 1))
    beta = sigmoid(x @ p.f32(pre + "ssm_beta.weight").T).T           # [nv, T]
    alpha = (x @ p.f32(pre + "ssm_alpha.weight").T).T                # [nv, T]
    gate = np.log1p(np.exp(alpha + p.f32(pre + "ssm_dt.bias")[:, None])) * p.f32(pre + "ssm_a")[:, None]
    o = gdn_ref.gdn_forward(q, k_, v, gate[None], beta[None], hd, nv, T)   # [hd, nv, T]
    zh = heads(z, nv)
    on = rms(o, eps, axis=0) * p.f32(pre + "ssm_norm.weight")[:, None, None] * sigmoid(zh)
    flat = on.transpose(2, 1, 0).reshape(T, nv * hd)
    return flat @ p.f32(pre + "ssm_out.weight").T


def rope_neox(x, pos, n_rot, base):
    """x [T, H, hd]: NEOX rotation of the first n_rot dims (pairs i, i + n_rot/2). For text tokens every IMRoPE
    section carries the same position, so qwen4exp's IMRoPE reduces to exactly this."""
    half = n_rot // 2
    inv = base ** (-np.arange(half) * 2.0 / n_rot)
    ang = pos[:, None] * inv[None]                                   # [T, half]
    c, s_ = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    y = x.copy()
    a, b = x[..., :half], x[..., half:n_rot]
    y[..., :half] = a * c - b * s_
    y[..., half:n_rot] = a * s_ + b * c
    return y


def qsa_dense(p, pre, x, n_head, n_kv, hd, n_rot, base, eps):
    """x [T, D] -> [T, D]: qwen4exp full-attention layer, DENSE path (what QSA reduces to while every pool is kept,
    i.e. T <= indexer_top_k: n_top_pool = min(n_pool, top_k / kpool))."""
    T = x.shape[0]
    qf = (x @ p.f32(pre + "attn_q.weight").T).reshape(T, n_head, 2 * hd)   # per head: [q | gate]
    q, gate = qf[..., :hd], qf[..., hd:]
    q = rms(q, eps) * p.f32(pre + "attn_q_norm.weight")
    k = rms((x @ p.f32(pre + "attn_k.weight").T).reshape(T, n_kv, hd), eps) * p.f32(pre + "attn_k_norm.weight")
    v = (x @ p.f32(pre + "attn_v.weight").T).reshape(T, n_kv, hd)
    pos = np.arange(T, dtype=np.float64)
    q = rope_neox(q, pos, n_rot, base)
    k = rope_neox(k, pos, n_rot, base)
    g = n_head // n_kv                                               # ggml GQA broadcast: q head h -> kv head h // g
    out = np.zeros((T, n_head, hd))
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for h in range(n_head):
        sc = (q[:, h] @ k[:, h // g].T) / np.sqrt(hd) + mask
        sc = np.exp(sc - sc.max(1, keepdims=True))
        sc /= sc.sum(1, keepdims=True)
        out[:, h] = sc @ v[:, h // g]
    out = out * sigmoid(gate)
    return out.reshape(T, n_head * hd) @ p.f32(pre + "attn_output.weight").T


def moe(p, pre, x, n_expert, k_used):
    T = x.shape[0]
    logits = x @ p.f32(pre + "ffn_gate_inp.weight").T                # [T, E]
    pr = np.exp(logits - logits.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    sel = np.argsort(-pr, axis=1)[:, :k_used]
    w = np.take_along_axis(pr, sel, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 6.103515625e-5)     # norm_w with the F16-min clamp
    out = np.zeros_like(x)
    for e in np.unique(sel):
        rows, slot = np.nonzero(sel == e)
        xe = x[rows]
        g = xe @ p.expert(pre + "ffn_gate_exps.weight", e).T
        u = xe @ p.expert(pre + "ffn_up_exps.weight", e).T
        d = (silu(g) * u) @ p.expert(pre + "ffn_down_exps.weight", e).T
        out[rows] += d * w[rows, slot][:, None]
    sh = (silu(x @ p.f32(pre + "ffn_gate_shexp.weight").T) * (x @ p.f32(pre + "ffn_up_shexp.weight").T)) \
        @ p.f32(pre + "ffn_down_shexp.weight").T
    sg = sigmoid(x @ p.f32(pre + "ffn_gate_inp_shexp.weight").reshape(-1))
    return out + sh * sg[:, None], len(np.unique(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("capdir")
    ap.add_argument("layer", type=int, help="first layer; with --to-final, run every layer from here to the logits")
    ap.add_argument("--next", default=None)
    ap.add_argument("--to-final", action="store_true")
    ap.add_argument("--tokens", default=None, help="prompt token file (for the CE check with --to-final)")
    a = ap.parse_args()

    p = Pack(a.pack)
    A = "qwen4exp."
    D = int(p.kv(A + "embedding_length"))
    hc = int(p.kv(A + "hyper_connection.count"))
    hd = int(p.kv(A + "ssm.state_size"))
    nk = int(p.kv(A + "ssm.group_count"))
    nv = int(p.kv(A + "ssm.time_step_rank"))
    d_conv = int(p.kv(A + "ssm.conv_kernel"))
    eps = float(p.kv(A + "attention.layer_norm_rms_epsilon"))
    E = int(p.kv(A + "expert_count"))
    K = int(p.kv(A + "expert_used_count"))
    NL = int(p.kv(A + "block_count"))
    fai = int(p.kv(A + "full_attention_interval"))
    n_head = int(p.kv(A + "attention.head_count"))
    n_kv = int(p.kv(A + "attention.head_count_kv"))
    ahd = int(p.kv(A + "attention.key_length"))
    n_rot = int(p.kv(A + "rope.dimension_count"))
    base = float(p.kv(A + "rope.freq_base"))
    top_k = int(p.kv(A + "attention.indexer.top_k"))
    print(f"[ref] D {D} hc {hc} hd {hd} nk {nk} nv {nv} eps {eps:g} E {E} K {K} | attn {n_head}/{n_kv}x{ahd} "
          f"rot {n_rot} base {base:g} top_k {top_k}", flush=True)

    def load(path, T=None):
        if path.endswith(".bf16"):
            u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
            r = u.view(np.float32).astype(np.float64)
        else:
            r = np.fromfile(path, dtype=np.float32).astype(np.float64)
        return r.reshape(-1, hc, D)

    def rel(x, y):
        return float(np.linalg.norm(x - y) / np.linalg.norm(y))

    def layer(L, res):
        pre = f"blk.{L}."
        T = res.shape[0]
        mixed, inj = hc_mix(p, pre + "hc_attn", res, hc, D, eps)
        if (L + 1) % fai == 0:
            if T > top_k:
                raise SystemExit(f"T {T} > indexer top_k {top_k}: QSA selection is not dense, replay STRATA_QSA_DUMP")
            att = qsa_dense(p, pre, mixed, n_head, n_kv, ahd, n_rot, base, eps)
        else:
            att = gdn(p, pre, mixed, hd, nk, nv, d_conv, eps)
        res1 = hc_combine(res, att, inj)
        mixed2, inj2 = hc_mix(p, pre + "hc_ffn", res1, hc, D, eps)
        ffn, n_used = moe(p, pre, mixed2, E, K)
        return hc_combine(res1, ffn, inj2), n_used

    def report(tag, res, out, ref):
        dr, do = ref - res, out - res
        cos = float((dr * do).sum() / (np.linalg.norm(dr) * np.linalg.norm(do)))
        print(f"[ref] {tag}: update rel err {rel(do, dr):.3e} cos {cos:.6f} | out rel {rel(out, ref):.3e} | "
              f"identity {rel(res, ref):.3e}", flush=True)

    L0 = a.layer
    res = load(os.path.join(a.capdir, f"L{L0}.bin"))
    T = res.shape[0]
    print(f"[ref] T {T}, input rms {np.sqrt((res ** 2).mean()):.4f}", flush=True)
    if not a.to_final:
        ref = load(a.next or os.path.join(a.capdir, f"L{L0 + 1}.bin"))
        out, n_used = layer(L0, res)
        report(f"layer {L0} ({n_used} experts) vs captured L{L0 + 1}", res, out, ref)
        return

    # --to-final: chain OUR outputs (not the captures) through every layer, report each against its capture
    cur = res
    for L in range(L0, NL):
        nxt = os.path.join(a.capdir, f"L{L + 1}.bin") if L + 1 < NL else os.path.join(a.capdir, "final.bf16")
        out, n_used = layer(L, cur)
        report(f"layer {L} {'QSA' if (L + 1) % fai == 0 else 'GDN'} (chained) vs {os.path.basename(nxt)}",
               cur, out, load(nxt))
        cur = out
    final_cap = load(os.path.join(a.capdir, "final.bf16"))

    def head(r):
        mixed, _ = hc_mix(p, "output_hc", r, hc, D, eps)
        W = p.f32("output.weight")                                   # [vocab, D]
        return mixed @ W.T

    lg_ours, lg_cap = head(cur), head(final_cap)
    print(f"[ref] logits ours vs from captured final: rel {rel(lg_ours, lg_cap):.3e}, "
          f"top-1 agree {int((lg_ours.argmax(1) == lg_cap.argmax(1)).sum())}/{T}")
    if a.tokens:
        toks = np.array([int(t) for t in open(a.tokens).read().replace(",", " ").split()])
        tgt = toks[1:T + 1]

        def ce(lg):
            m = lg.max(1, keepdims=True)
            lse = (m[:, 0] + np.log(np.exp(lg - m).sum(1)))
            return lse - lg[np.arange(len(tgt)), tgt]

        c1, c2 = ce(lg_ours), ce(lg_cap)
        print(f"[ref] next-token CE over {len(tgt)} positions: ours {c1.mean():.4f}, from captured final {c2.mean():.4f}; "
              f"top-1 = next token {int((lg_cap.argmax(1) == tgt).sum())}/{len(tgt)}")
        print("[ref] predicted (from captured final):", " ".join(str(int(t)) for t in lg_cap.argmax(1)))
        print("[ref] actual next                    :", " ".join(str(int(t)) for t in tgt))


if __name__ == "__main__":
    main()
