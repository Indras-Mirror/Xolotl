#!/usr/bin/env python3
"""Tiny F32 M3.1 with MSA also on the leading dense layer; no production model changes."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_writer import GGUFWriter, _Tensor

D, L, H, HKV, HD, ROT = 32, 4, 4, 2, 16, 8
FF, EF, E, USED, VOCAB = 64, 32, 4, 2, 64
BLOCK, TOP_BLOCKS, CTX = 4, 2, 768


def make_fixture(path: Path, missing_dense_index=False, unknown_tensor=False, block=BLOCK, overflow_float=None):
    w = GGUFWriter()
    w.add("general.architecture", "minimax-m3")
    meta = {
        "block_count": L, "embedding_length": D, "context_length": CTX,
        "attention.head_count": H, "attention.head_count_kv": HKV,
        "attention.key_length": HD, "attention.value_length": HD,
        "rope.dimension_count": ROT, "feed_forward_length": FF,
        "expert_feed_forward_length": EF, "expert_count": E, "expert_used_count": USED,
        "expert_shared_count": 1, "leading_dense_block_count": 1,
        "expert_gating_func": 2, "attention.indexer.head_count": HKV,
        "attention.indexer.key_length": HD, "attention.indexer.block_size": block,
        "attention.indexer.top_k": TOP_BLOCKS, "attention.indexer.local_blocks": 1,
    }
    for key, value in meta.items():
        w.add("minimax-m3." + key, value, "u32")
    for key, value in {"attention.layer_norm_rms_epsilon": 1e-6,
                       "rope.freq_base": 10000., "expert_weights_scale": 2.}.items():
        w.add("minimax-m3." + key, 1e300 if key == overflow_float else value,
              "f64" if key == overflow_float else "f32")
    w.add("minimax-m3.expert_weights_norm", True)
    w.add("tokenizer.ggml.normalizer.nfc", True)
    rng = np.random.default_rng(31192)

    def add(name, ne, norm=False):
        # GGUF dim0 is contiguous: raw data is C-contiguous with reversed dimensions.
        a = rng.normal(0, .08, tuple(reversed(ne))).astype("<f4")
        if norm:
            a += 1
        if "ffn_gate" in name and "inp" not in name:
            a.reshape(-1)[0] = 18.0  # force the OAI gate clamp for some inputs
        if "ffn_up" in name:
            a.reshape(-1)[0] = -19.0
        w.tensors.append(_Tensor(name, list(ne), "F32", a.tobytes()))

    add("token_embd.weight", [D, VOCAB])
    add("output.weight", [D, VOCAB])
    add("output_norm.weight", [D], True)
    for l in range(L):
        p = f"blk.{l}."
        for name in ("attn_norm", "ffn_norm"):
            add(p + name + ".weight", [D], True)
        add(p + "attn_q.weight", [D, H * HD])
        for role in ("k", "v"):
            add(p + f"attn_{role}.weight", [D, HKV * HD])
        add(p + "attn_output.weight", [H * HD, D])
        for role in ("q", "k"):
            add(p + f"attn_{role}_norm.weight", [HD], True)
        if not (missing_dense_index and l == 0):
            add(p + "indexer.q_proj.weight", [D, HKV * HD])
            add(p + "indexer.k_proj.weight", [D, HD])
            for role in ("q", "k"):
                add(p + f"indexer.{role}_norm.weight", [HD], True)
        if l == 0:
            for role in ("gate", "up"):
                add(p + f"ffn_{role}.weight", [D, FF])
            add(p + "ffn_down.weight", [FF, D])
        else:
            add(p + "ffn_gate_inp.weight", [D, E])
            add(p + "exp_probs_b.bias", [E])
            for role in ("gate", "up", "down"):
                ne = [EF, D] if role == "down" else [D, EF]
                add(p + f"ffn_{role}_shexp.weight", ne)
                add(p + f"ffn_{role}_exps.weight", ne + [E])
    if unknown_tensor:
        add("unrecognized.weight", [D])
    w.write(path)


if __name__ == "__main__":
    make_fixture(Path(sys.argv[1]))
