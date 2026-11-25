#!/usr/bin/env python3
"""ernie_ref.py - numpy reference forward for ERNIE-5.5-Flash (HCA layers), CPU only.

Reads the blockwise-FP8 safetensors directly (128x128 dequant), runs embedding +
trunk layers 0..L-1 for a short token list, and dumps the per-layer residual
streams, the attention/router intermediates and the router top-10 to an .npz.
The math follows the vLLM-ERNIE model code (`src/vllm_ernie/models/ernie5_5/`,
the upstream vLLM 0.32 `DeepseekV4Attention` + `DeepseekV4ScalingRotaryEmbedding`
that it subclasses) and the llama.cpp ERNIE 5.5 PR #31925 graph.

    python3 tools/ernie/ernie_ref.py SRC --layers 3 --tokens 30000,30001,30002 --out ref.npz

`--vha vllm` (default) applies the head mix as vLLM does (`out = m o`), which is also what the
llama.cpp PR #31925 computes; `--vha llamacpp` is an alias (an earlier einsum here summed the wrong index) so the
rest of the graph can be cross-checked against the PR (see README_REF.md).

Bounded memory: only the tensors of the requested layers are read, experts are
dequantised one at a time and only the ones the router selected, and every
layer's big matrices are released before the next layer starts.

Sites captured (same ggml tensor names the C++ dumper writes, see
tools/ernie/ernie_dump.cpp):
    hc_init, l_last-{il}, hc_attn_pre-{il}, hc_ffn_pre-{il}, attn_norm-{il},
    ffn_norm-{il}, qr-{il}, qr_norm-{il}, q-{il}, kv-{il}, attn_hca-{il}, attn_derope-{il},
    attn_wo_a-{il}, attn_gate-{il}, attn_out-{il}, ffn_out-{il},
    ffn_shexp-{il}, ffn_routed_up-{il}, ffn_moe_probs-{il},
    ffn_moe_topk-{il}, ffn_moe_weights-{il}

"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ernie_gguf import Store  # noqa: E402  (safetensors index + FP8 block dequant)

# --------------------------------------------------------------------------- model constants
D = 4096
N_HEAD = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
SWA = 128
HCA_RATIO = 128
EPS = 1e-6
HC = 4
HC_ITERS = 20
HC_EPS = 1e-6
HC_POST_MULT = 2.0
SITU_BETA = 4.0
SITU_LB = 25.0
N_EXPERT = 512
TOP_K = 10
LATENT = 2048
MOE_FF = 2048
COMPRESS_ROPE_BASE = 160000.0
YARN_FACTOR = 16.0
YARN_ORIG_CTX = 65536
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- numpy primitives
def rmsnorm(x, w=None, eps=EPS):
    x = np.asarray(x, np.float32)
    y = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))
    if w is not None:
        y = y * w
    return y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def situ(g, u):
    """beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)  (llama.cpp LLM_FFN_SITU)."""
    a = SITU_BETA * np.tanh(g * (1.0 / SITU_BETA)) * sigmoid(g)
    b = SITU_LB * np.tanh(u * (1.0 / SITU_LB))
    return a * b


def yarn_correction_dim(num_rotations, dim, base, orig_ctx):
    return dim * math.log(orig_ctx / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_correction_range(low_rot, high_rot, dim, base, orig_ctx):
    # vLLM `yarn_find_correction_range` (truncate=True) and ggml's
    # `ggml_rope_yarn_corr_dims` both floor/ceil the two correction dims.
    low = math.floor(yarn_correction_dim(low_rot, dim, base, orig_ctx))
    high = math.ceil(yarn_correction_dim(high_rot, dim, base, orig_ctx))
    return max(low, 0), min(high, dim - 1)


def build_inv_freq(rope_dim, base):
    """HCA compress rope: YaRN, factor 16, beta 32/1, mscale 1.

    vLLM (`build_deepseek_v4_rope` -> `DeepseekV4ScalingRotaryEmbedding`, mscale
    and mscale_all_dim = 0) and the llama.cpp PR agree: the low-frequency tail
    is interpolated by the YaRN factor, the head of the dims keeps the raw
    frequency.  For ERNIE: base 160000, orig ctx 65536 -> correction dims
    floor/ceil = 15 and 25.
    """
    pos_freqs = np.power(base, np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim)
    extrap = 1.0 / pos_freqs
    interp = 1.0 / (YARN_FACTOR * pos_freqs)
    low, high = yarn_correction_range(YARN_BETA_FAST, YARN_BETA_SLOW, rope_dim, base, YARN_ORIG_CTX)
    ramp = np.clip((np.arange(rope_dim // 2, dtype=np.float64) - low) / max(0.001, high - low), 0.0, 1.0)
    inv = interp * ramp + extrap * (1.0 - ramp)
    return inv.astype(np.float32)


def rope_apply(x, positions, inv_freq, inverse=False):
    """GPT-J / interleaved partial RoPE on the last 2*len(inv_freq) channels.

    x: [nt, ..., head_dim].  mscale is 1.0 for the ERNIE compress rope (vLLM
    passes mscale=0/mscale_all_dim=0; llama.cpp's yarn_attn_factor cancels).
    """
    rd = 2 * len(inv_freq)
    freqs = np.asarray(positions, np.float64)[:, None] * inv_freq[None, :].astype(np.float64)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    if inverse:
        sin = -sin
    cos = np.repeat(cos, 2, axis=-1).astype(np.float32)
    sin = np.repeat(sin, 2, axis=-1).astype(np.float32)
    while cos.ndim < x.ndim:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    xr = x[..., -rd:]
    pairs = xr.reshape(xr.shape[:-1] + (rd // 2, 2))
    rot = np.empty_like(pairs)
    rot[..., 0] = -pairs[..., 1]
    rot[..., 1] = pairs[..., 0]
    rot = rot.reshape(xr.shape)
    out = np.concatenate([x[..., :-rd], xr * cos + rot * sin], axis=-1)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- mHC
def hc_pre(x, fn, scale, base, pre_only=False):
    """ERNIE mHC pre: 1/(sqrt(mean(x^2)) + eps) over the flattened 4*D vector.

    Returns (collapsed [nt,D], post [nt,4], comb [nt,4,4]).
    """
    nt = x.shape[0]
    flat = x.reshape(nt, HC * D).astype(np.float32)
    rms = np.sqrt(np.mean(flat * flat, axis=-1)) + HC_EPS
    mixes = (flat @ fn.T) / rms[:, None]

    pre = sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC]) * HC_POST_MULT
    comb = mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]
    comb = softmax(comb.reshape(nt, HC, HC), axis=-1) + HC_EPS
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + HC_EPS)
    for _ in range(HC_ITERS - 1):
        comb = comb / (np.sum(comb, axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + HC_EPS)

    collapsed = np.einsum("th,thd->td", pre, x, optimize=True).astype(np.float32)
    return collapsed, post, comb


def hc_post(x, residual, post, comb):
    # out[t,dst,:] = post[t,dst]*x[t,:] + sum_src comb[t,src,dst]*residual[t,src,:]
    out = np.einsum("tsk,tsd->tkd", comb, residual, optimize=True).astype(np.float32)
    out += x[:, None, :] * post[:, :, None]
    return out


# --------------------------------------------------------------------------- model
class ErnieRef:
    def __init__(self, src, n_layers=3, vha_mode="vllm"):
        self.store = Store(src, max_open=4)
        with open(os.path.join(src, "config.json")) as f:
            self.cfg = json.load(f)
        self.vha_mode = vha_mode
        self.n_layers = n_layers
        self.inv_hca = build_inv_freq(ROPE_DIM, COMPRESS_ROPE_BASE)
        log(f"HCA rope inv_freq[0,8,15,16,24,31] {np.round(self.inv_hca[[0, 8, 15, 16, 24, 31]], 6)}")
        self.is_hca = [max(r, 0) != 0 for r in self.cfg["compress_ratios"]]

    # ---- tensor helpers ------------------------------------------------
    def g(self, name):
        return self.store.f32(name).reshape(self.store.index[name].shape).astype(np.float32)

    def g1(self, name):
        return self.g(name).reshape(-1)

    def hc_pair(self, il, kind):
        p = f"model.layers.{il}.{kind}_hyper_connection."
        fn = self.g(p + "mapping_proj.weight").T.astype(np.float32)      # [24, 16384]
        base = self.g1(p + "bias")                                        # [24]
        scale = np.array([self.g1(p + "alpha_pre")[0],
                          self.g1(p + "alpha_post")[0],
                          self.g1(p + "alpha_res")[0]], np.float32)
        return fn, scale, base

    def layer_weights(self, il):
        """Load the non-expert tensors of one layer (bounded: released after use)."""
        p = f"model.layers.{il}."
        T = {}
        T["attn_norm"] = self.g1(p + "input_layernorm.weight")
        T["ffn_norm"] = self.g1(p + "post_attention_layernorm.weight")
        T["hc_attn"] = self.hc_pair(il, "self_attention")
        T["hc_ffn"] = self.hc_pair(il, "mlp")

        if self.is_hca[il]:
            a = p + "self_attn."
            T["sinks"] = self.g1(a + "core_attention.attn_sink")
            T["vha_u"] = self.g(a + "vha_postmix_U")
            T["vha_v"] = self.g(a + "vha_postmix_V")
            T["q_a"] = self.g(a + "linear_q_down_proj.weight")
            T["q_a_norm"] = self.g1(a + "q_layernorm.weight")
            T["q_b"] = self.g(a + "linear_q_up_proj.weight")
            T["kv_w"] = self.g(a + "linear_kv_proj.weight")
            T["kv_norm"] = self.g1(a + "kv_layernorm.weight")
            T["wo_a"] = self.g(a + "linear_o_group_proj").reshape(O_GROUPS, O_LORA, -1)
            T["wo_b"] = self.g(a + "o_proj.weight")
            T["o_gate"] = self.g(a + "gate_proj.weight")
            T["c_kv"] = self.g(a + "core_attention.compressor.linear_wkv.weight")
            T["c_gate"] = self.g(a + "core_attention.compressor.linear_wgate.weight")
            T["c_ape"] = self.g(a + "core_attention.compressor.ape")
            T["c_norm"] = self.g1(a + "core_attention.compressor.norm.weight")
        else:
            raise NotImplementedError(
                f"layer {il} is MLA+DSA (compress ratio 0); its tensors live in shard 64 and the "
                "DSA path is not part of this slice")

        if il < self.cfg["first_k_dense_replace"]:
            m = p + "mlp."
            T["ffn_gate"] = self.g(m + "gate_proj.weight")
            T["ffn_up"] = self.g(m + "up_proj.weight")
            T["ffn_down"] = self.g(m + "down_proj.weight")
            return T

        b = p + "block_sparse_moe."
        T["router0"] = self.g(b + "gate.weight")
        T["router1"] = self.g(b + "gate.weight_1")
        T["exp_bias"] = self.g1(b + "e_score_correction_bias")
        T["exp_scale"] = self.g1(b + "gate.routed_scaling_factor_param")
        T["fc1"] = self.g(b + "fc1_latent_proj.weight")
        T["fc2"] = self.g(b + "fc2_latent_proj.weight")
        T["latent_norm"] = self.g1(b + "latent_norm.weight")
        T["sh_gate"] = self.g(b + "shared_experts.w1.weight")
        T["sh_down"] = self.g(b + "shared_experts.w2.weight")
        T["sh_up"] = self.g(b + "shared_experts.w3.weight")
        return T

    def expert(self, il, e, role):
        """One routed expert projection, dequantised on demand."""
        n = f"model.layers.{il}.block_sparse_moe.experts.{e}.{role}.weight"
        return self.g(n)

    # ---- attention -----------------------------------------------------
    def attention(self, il, cur, T, positions, cap):
        nt = cur.shape[0]
        q_a = cur @ T["q_a"].T
        cap[f"qr-{il}"] = q_a
        q_c = rmsnorm(q_a, T["q_a_norm"])
        cap[f"qr_norm-{il}"] = q_c

        gate = sigmoid(q_c @ T["o_gate"].T)
        cap[f"attn_gate-{il}"] = gate

        q = (q_c @ T["q_b"].T).reshape(nt, N_HEAD, HEAD_DIM)
        # per-head unweighted QK-norm (vLLM fused_qnorm kernel / llama.cpp ggml_rms_norm on q)
        q = q * (1.0 / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + EPS))
        q = rope_apply(q, positions, self.inv_hca)
        cap[f"q-{il}"] = q

        kv = rmsnorm(cur @ T["kv_w"].T, T["kv_norm"])
        kv = rope_apply(kv[:, None, :], positions, self.inv_hca)[:, 0, :]
        cap[f"kv-{il}"] = kv

        # ---- compressor: one row per complete 128-token window
        ckv = cur @ T["c_kv"].T
        cgate = cur @ T["c_gate"].T
        nblk = nt // HCA_RATIO
        comp = []
        for b in range(nblk):
            s = slice(b * HCA_RATIO, (b + 1) * HCA_RATIO)
            w = softmax(cgate[s] + T["c_ape"], axis=0)
            c = np.sum(ckv[s] * w, axis=0)
            c = rmsnorm(c, T["c_norm"])
            comp.append(rope_apply(c[None, :], np.array([b * HCA_RATIO]), self.inv_hca)[0])
        comp = np.stack(comp) if comp else np.zeros((0, HEAD_DIM), np.float32)

        allk = np.concatenate([kv, comp], axis=0)  # K = V
        scale = 1.0 / math.sqrt(HEAD_DIM)
        S = np.einsum("thd,jd->thj", q, allk, optimize=True) * scale

        idx = np.arange(nt)
        allow = np.zeros((nt, allk.shape[0]), bool)
        allow[:, :nt] = (idx[None, :] <= positions[:, None]) & (idx[None, :] > positions[:, None] - SWA)
        if comp.shape[0]:
            allow[:, nt:] = np.arange(comp.shape[0])[None, :] < ((positions[:, None] + 1) // HCA_RATIO)
        S = np.where(allow[:, None, :], S, -np.inf)

        sink = T["sinks"].astype(np.float32)
        m = np.maximum(np.max(S, axis=-1), sink[None, :])
        E = np.exp(S - m[..., None])
        den = np.sum(E, axis=-1) + np.exp(sink[None, :] - m)
        out = np.einsum("thj,jd->thd", E, allk, optimize=True) / den[..., None]
        out = out.astype(np.float32)
        cap[f"attn_hca-{il}"] = out

        # ---- inverse RoPE, VHA head mix, grouped low-rank out, sigmoid gate
        out = rope_apply(out, positions, self.inv_hca, inverse=True)
        cap[f"attn_derope-{il}"] = out

        # vLLM VHAPostMix: m = I + V U^T, out[h] = sum_g m[h,g] o[g] (torch.matmul(m, o) with o
        # [T, H, D]).  llama.cpp's build_head_mix computes the same m o (ggml_mul_mat(mix, x) with
        # the row-major [h][g] tensor).  --vha llamacpp is kept as an alias: both conventions agree.
        m_vha = np.eye(N_HEAD, dtype=np.float32) + T["vha_v"] @ T["vha_u"].T
        mixed = np.einsum("hg,tgd->thd", m_vha, out, optimize=True).astype(np.float32)

        grouped = mixed.reshape(nt, O_GROUPS, (N_HEAD // O_GROUPS) * HEAD_DIM)
        oa = np.einsum("tgi,goi->tgo", grouped, T["wo_a"], optimize=True).astype(np.float32)
        oa = oa.reshape(nt, O_GROUPS * O_LORA)
        cap[f"attn_wo_a-{il}"] = oa

        oa = oa * gate
        out = oa @ T["wo_b"].T
        cap[f"attn_out-{il}"] = out
        return out.astype(np.float32)

    # ---- FFNs ----------------------------------------------------------
    def dense_ffn(self, cur, T, cap, il):
        g = cur @ T["ffn_gate"].T
        u = cur @ T["ffn_up"].T
        h = situ(g, u)
        out = h @ T["ffn_down"].T
        cap[f"ffn_out-{il}"] = out
        return out.astype(np.float32)

    def latent_moe(self, il, cur, T, cap):
        nt = cur.shape[0]
        scores = sigmoid(cur @ T["router0"].T) + sigmoid(cur @ T["router1"].T)
        cap[f"ffn_moe_probs_split-{il}"] = scores

        sel = scores + T["exp_bias"]
        ids = np.argsort(-sel, axis=1, kind="stable")[:, :TOP_K]
        w = np.take_along_axis(scores, ids, axis=1)
        w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-20)
        w = w * T["exp_scale"][ids]
        cap[f"ffn_moe_topk-{il}"] = ids.astype(np.int32)
        cap[f"ffn_moe_weights-{il}"] = w.astype(np.float32)

        # dense per-expert routing probs (llama.cpp ffn_moe_probs: renorm * scale)
        w_sum = np.maximum(np.sum(np.take_along_axis(scores, ids, axis=1), axis=1, keepdims=True), 1e-20)
        cap[f"ffn_moe_probs-{il}"] = (scores * T["exp_scale"][None, :] / w_sum).astype(np.float32)

        routed = cur @ T["fc1"].T
        out = np.zeros((nt, LATENT), np.float32)
        for e in np.unique(ids):
            rows, ks = np.where(ids == e)
            if rows.size == 0:
                continue
            x = routed[rows]
            g = x @ self.expert(il, int(e), "w1").T
            u = x @ self.expert(il, int(e), "w3").T
            h = situ(g, u)
            d = h @ self.expert(il, int(e), "w2").T
            out[rows] += w[rows, ks][:, None] * d
            del g, u, h, d
        out = rmsnorm(out, T["latent_norm"])
        out = out @ T["fc2"].T
        cap[f"ffn_routed_up-{il}"] = out

        # shared expert sees the original (normalised) hidden state
        sg = cur @ T["sh_gate"].T
        su = cur @ T["sh_up"].T
        sh = (situ(sg, su) @ T["sh_down"].T).astype(np.float32)
        cap[f"ffn_shexp-{il}"] = sh

        out = out + sh
        cap[f"ffn_out-{il}"] = out
        return out.astype(np.float32)

    # ---- forward -------------------------------------------------------
    def forward(self, token_ids, cap):
        nt = len(token_ids)
        positions = np.arange(nt, dtype=np.int64)
        # rows only: the full 201216 x 4096 BF16 table would be 1.65 GB as float32
        rows = {}
        emb = np.empty((nt, D), np.float32)
        for i, t in enumerate(token_ids):
            if t not in rows:
                rows[t] = self.store.f32("model.embed_tokens.weight", int(t), int(t) + 1).reshape(D)
            emb[i] = rows[t]
        x = np.repeat(emb[:, None, :], HC, axis=1).astype(np.float32)  # [nt, 4, D]
        cap["hc_init"] = x
        del emb, rows

        for il in range(self.n_layers):
            if not self.is_hca[il]:
                raise NotImplementedError(f"layer {il}: MLA+DSA path not implemented in this slice")
            T = self.layer_weights(il)

            residual = x
            cur, post, comb = hc_pre(x, *T["hc_attn"])
            cap[f"hc_attn_pre-{il}"] = cur
            cur = rmsnorm(cur, T["attn_norm"])
            cap[f"attn_norm-{il}"] = cur
            cur = self.attention(il, cur, T, positions, cap)
            x = hc_post(cur, residual, post, comb)

            residual = x
            cur, post, comb = hc_pre(x, *T["hc_ffn"])
            cap[f"hc_ffn_pre-{il}"] = cur
            cur = rmsnorm(cur, T["ffn_norm"])
            cap[f"ffn_norm-{il}"] = cur
            if il < self.cfg["first_k_dense_replace"]:
                cur = self.dense_ffn(cur, T, cap, il)
            else:
                cur = self.latent_moe(il, cur, T, cap)
            x = hc_post(cur, residual, post, comb)
            cap[f"l_last-{il}"] = x

            del T, cur, residual, post, comb
            gc.collect()
            log(f"layer {il} done ({'dense' if il < self.cfg['first_k_dense_replace'] else 'moe'})")
        return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="FP8 repo dir (config.json + model-*.safetensors)")
    ap.add_argument("--layers", type=int, default=3, help="run layers 0..N-1 (default 3)")
    ap.add_argument("--tokens", default="1001,337,42,8192,7,12345,200000,512",
                    help="comma separated token ids")
    ap.add_argument("--tokens-file", help="file with token ids (comma/whitespace separated)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--vha", choices=("vllm", "llamacpp"), default="vllm",
                    help="head-mix convention: both vllm and llamacpp compute out = m o (alias kept for old commands)")
    args = ap.parse_args(argv)

    if args.tokens_file:
        with open(args.tokens_file) as f:
            ids = [int(t) for t in f.read().replace(",", " ").split()]
    else:
        ids = [int(t) for t in args.tokens.split(",") if t.strip()]
    if not ids:
        ap.error("no tokens")

    t0 = time.time()
    ref = ErnieRef(args.src, n_layers=args.layers, vha_mode=args.vha)
    cap = {}
    ref.forward(ids, cap)
    np.savez(args.out, **cap)
    dt = time.time() - t0
    log(f"wrote {args.out}: {len(cap)} arrays, {len(ids)} tokens, {dt:.1f}s")
    ref.store.report_io()
    ref.store.drop()


if __name__ == "__main__":
    main()
