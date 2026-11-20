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

sys.path.insert(0, "/media/Crucial1TB/xolotl-train/llama.cpp/gguf-py")
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
    ap.add_argument("layer", type=int)
    ap.add_argument("--next", default=None)
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
    L = a.layer
    if (L + 1) % int(p.kv(A + "full_attention_interval")) == 0:
        raise SystemExit(f"layer {L} is a QSA layer; this reference covers the GDN layers")
    pre = f"blk.{L}."
    print(f"[ref] layer {L}: D {D} hc {hc} hd {hd} nk {nk} nv {nv} conv {d_conv} eps {eps:g} E {E} K {K}", flush=True)

    res = np.fromfile(os.path.join(a.capdir, f"L{L}.bin"), dtype=np.float32).astype(np.float64)
    T = res.size // (hc * D)
    res = res.reshape(T, hc, D)
    nxt_path = a.next or os.path.join(a.capdir, f"L{L + 1}.bin")
    ref = np.fromfile(nxt_path, dtype=np.float32).astype(np.float64).reshape(T, hc, D)
    print(f"[ref] T {T}, input rms {np.sqrt((res ** 2).mean()):.4f}, target rms {np.sqrt((ref ** 2).mean()):.4f}",
          flush=True)

    mixed, inj = hc_mix(p, pre + "hc_attn", res, hc, D, eps)
    att = gdn(p, pre, mixed, hd, nk, nv, d_conv, eps)
    res1 = hc_combine(res, att, inj)
    mixed2, inj2 = hc_mix(p, pre + "hc_ffn", res1, hc, D, eps)
    ffn, n_used = moe(p, pre, mixed2, E, K)
    out = hc_combine(res1, ffn, inj2)

    def rel(x, y):
        return float(np.linalg.norm(x - y) / np.linalg.norm(y))

    delta_ref = ref - res
    delta_out = out - res
    cos = float((delta_ref * delta_out).sum() / (np.linalg.norm(delta_ref) * np.linalg.norm(delta_out)))
    print(f"[ref] experts touched {n_used}")
    print(f"[ref] rel L2 |out - captured| / |captured|           = {rel(out, ref):.3e}")
    print(f"[ref] rel L2 of the layer's UPDATE (out-in vs cap-in) = {rel(delta_out, delta_ref):.3e}  cos {cos:.6f}")
    print(f"[ref] baseline: identity (in vs captured)            = {rel(res, ref):.3e}")
    per_t = np.linalg.norm((out - ref).reshape(T, -1), axis=1) / np.linalg.norm(delta_ref.reshape(T, -1), axis=1)
    print("[ref] per-position update error:", " ".join(f"{v:.3f}" for v in per_t))


if __name__ == "__main__":
    main()
