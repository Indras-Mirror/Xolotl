"""Independent small F32-model reference, float64 math with explicit F16 KV storage.

This is a synthetic graph check, not the real UD-IQ3_S acceptance oracle.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_reader import GGUFFile


def oai(gate, up):
    gate = np.minimum(gate, 7)
    return gate / (1 + np.exp(-1.702 * gate)) * (np.clip(up, -7, 7) + 1)


def forward(path: Path, ids, trace=None):
    file = GGUFFile(path)
    m = file.metadata
    w = {t.name: np.memmap(path, mode="r", dtype="<f4", offset=file.data_start + t.offset,
                         shape=tuple(reversed(t.shape))).astype(np.float64) for t in file.tensors}
    key = lambda s: m["minimax-m3." + s]
    d, h, kv = key("embedding_length"), key("attention.head_count"), key("attention.head_count_kv")
    hd, rot = key("attention.key_length"), key("rope.dimension_count")
    block, top = key("attention.indexer.block_size"), key("attention.indexer.top_k")
    eps, base = key("attention.layer_norm_rms_epsilon"), key("rope.freq_base")
    n = len(ids)

    def norm(x, scale):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * scale

    def rope(x):
        angle = np.arange(n)[:, None] / base ** (np.arange(0, rot, 2)[None, :] / rot)
        if x.ndim == 3:
            angle = angle[:, None, :]
        a, b = x[..., :rot // 2], x[..., rot // 2:rot]
        return np.concatenate([a * np.cos(angle) - b * np.sin(angle),
                               a * np.sin(angle) + b * np.cos(angle), x[..., rot:]], -1)

    x = w["token_embd.weight"][ids]
    for l in range(key("block_count")):
        p = f"blk.{l}."
        wt = lambda name: w[p + name]
        xn = norm(x, wt("attn_norm.weight"))
        if trace: trace(f"attn_norm-{l}", xn)
        q = rope(norm((xn @ wt("attn_q.weight").T).reshape(n, h, hd), wt("attn_q_norm.weight")))
        k = rope(norm((xn @ wt("attn_k.weight").T).reshape(n, kv, hd), wt("attn_k_norm.weight")))
        v = (xn @ wt("attn_v.weight").T).reshape(n, kv, hd)
        iq = rope(norm((xn @ wt("indexer.q_proj.weight").T).reshape(n, kv, hd), wt("indexer.q_norm.weight")))
        ik = rope(norm(xn @ wt("indexer.k_proj.weight").T, wt("indexer.k_norm.weight")))
        k, v, ik = (a.astype(np.float16).astype(np.float64) for a in (k, v, ik))
        # FA rounds Q to F16; the indexer uses an F32 gather of its F16 cache.
        q = q.astype(np.float16).astype(np.float64)
        attn = np.empty((n, h, hd))
        for t in range(n):
            nblocks = (t + block) // block
            score = np.full((nblocks * block, kv), -np.inf)
            score[:t + 1] = ik[:t + 1] @ iq[t].T
            bs = score.reshape(nblocks, block, kv).max(1)
            for local in range(key("attention.indexer.local_blocks")):
                b = t // block - local
                if b >= 0:
                    bs[b] += 1e30
            chosen = np.argsort(-bs, axis=0, kind="stable")[:min(top, nblocks)]
            for group in range(kv):
                rows = (chosen[:, group, None] * block + np.arange(block)).reshape(-1)
                rows = rows[rows <= t]
                for head in range(group * (h // kv), (group + 1) * (h // kv)):
                    s = k[rows, group] @ q[t, head] / np.sqrt(hd)
                    pr = np.exp(s - s.max())
                    pr /= pr.sum()
                    attn[t, head] = pr @ v[rows, group]
        residual = x + attn.reshape(n, h * hd) @ wt("attn_output.weight").T
        if trace: trace(f"kqv_out-{l}", attn.reshape(n, h * hd))
        if trace: trace(f"ffn_inp-{l}", residual)
        xn = norm(residual, wt("ffn_norm.weight"))
        if l < key("leading_dense_block_count"):
            ff = oai(xn @ wt("ffn_gate.weight").T, xn @ wt("ffn_up.weight").T) @ wt("ffn_down.weight").T
        else:
            shared = oai(xn @ wt("ffn_gate_shexp.weight").T, xn @ wt("ffn_up_shexp.weight").T) @ wt("ffn_down_shexp.weight").T
            prob = 1 / (1 + np.exp(-(xn @ wt("ffn_gate_inp.weight").T)))
            routed_ids = np.argsort(-(prob + wt("exp_probs_b.bias")), axis=1, kind="stable")[:, :key("expert_used_count")]
            ff = np.zeros_like(xn)
            for t in range(n):
                weights = prob[t, routed_ids[t]]
                if key("expert_weights_norm"):
                    weights = weights / max(weights.sum(), 6.103515625e-5)
                weights *= key("expert_weights_scale")
                for e, scale in zip(routed_ids[t], weights):
                    expert = oai(wt("ffn_gate_exps.weight")[e] @ xn[t], wt("ffn_up_exps.weight")[e] @ xn[t])
                    ff[t] += scale * (wt("ffn_down_exps.weight")[e] @ expert)
            ff += shared
        x = residual + ff
        if trace: trace(f"l_out-{l}", x)
    return norm(x, w["output_norm.weight"]) @ w["output.weight"].T
