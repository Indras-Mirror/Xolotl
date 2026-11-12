"""Structural test for tools/qwen4_hc_fixture.py's dense HC + full-attention model.

The fixture exists so the dense engine's gated-residual path (and now a mixed GDN/attention layer
stack with the Qwen4-27B QSA head ratio) has a file it can load.  This test checks the FILE, not the
engine: it reads the GGUF back with the repo's own header reader and pins the metadata and the tensor
set/shapes the fixture promises.  Running the engine is deliberately not part of it - `--check` needs
a GPU - so a shape regression here is caught without one.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gguf_reader import GGUFFile  # noqa: E402
from qwen4_hc_fixture import E, LAYERS, VOCAB, build  # noqa: E402

HC, HC_LR, ATT_HEAD, ATT_KV, ATT_DIM, ATT_ROT, FF = 4, 64, 12, 2, 64, 16, 384


@pytest.fixture()
def mixed(tmp_path):
    path = build(tmp_path / "hc-qsa.gguf", hc=HC, hc_lr=HC_LR, attn_every=4,
                 attn_heads=ATT_HEAD, attn_kv=ATT_KV, attn_dim=ATT_DIM, attn_rot=ATT_ROT, ff=FF)
    return GGUFFile(path)


def tensor(gguf: GGUFFile, name: str):
    found = [t for t in gguf.tensors if t.name == name]
    assert len(found) == 1, f"{name}: found {len(found)}"
    return found[0]


def test_metadata_are_dense_with_hyper_connections(mixed: GGUFFile):
    m = mixed.metadata
    assert m["general.architecture"] == "qwen35"
    assert m["qwen35.block_count"] == LAYERS
    assert m["qwen35.embedding_length"] == E
    assert m["qwen35.feed_forward_length"] == FF
    assert m["qwen35.hyper_connection.count"] == HC
    assert m["qwen35.hyper_connection.low_rank"] == HC_LR
    # the QSA ratio: 12 query heads over 2 KV heads == 6
    assert m["qwen35.attention.head_count"] == ATT_HEAD
    assert m["qwen35.attention.head_count_kv"] == ATT_KV
    assert m["qwen35.attention.head_count"] % m["qwen35.attention.head_count_kv"] == 0
    assert m["qwen35.attention.head_count"] // m["qwen35.attention.head_count_kv"] == 6
    assert m["qwen35.attention.key_length"] == ATT_DIM
    assert m["qwen35.rope.dimension_count"] == ATT_ROT
    # a dense file: no routed-expert metadata at all
    assert "qwen35.expert_count" not in m
    assert "qwen35.expert_used_count" not in m
    assert "qwen35.expert_feed_forward_length" not in m
    assert len(m["tokenizer.ggml.tokens"]) == VOCAB


def test_one_attention_layer_beside_gdn_layers(mixed: GGUFFile):
    for layer in range(LAYERS):
        names = {t.name.split(".", 2)[2] for t in mixed.tensors if t.name.startswith(f"blk.{layer}.")}
        if (layer + 1) % 4 == 0:          # the fixture's attn_every=4
            assert "attn_q.weight" in names and "attn_qkv.weight" not in names, layer
            assert {"attn_k.weight", "attn_v.weight", "attn_output.weight",
                    "attn_q_norm.weight", "attn_k_norm.weight"} <= names, layer
        else:
            assert "attn_qkv.weight" in names and "attn_q.weight" not in names, layer
            assert {"attn_gate.weight", "ssm_out.weight", "ssm_conv1d.weight",
                    "ssm_alpha.weight", "ssm_beta.weight", "ssm_norm.weight"} <= names, layer


def test_shapes_are_the_engine_layout(mixed: GGUFFile):
    # every layer: the dense FFN and the gated residual
    for layer in range(LAYERS):
        ln = f"blk.{layer}."
        assert tensor(mixed, ln + "ffn_gate.weight").shape == [E, FF]
        assert tensor(mixed, ln + "ffn_up.weight").shape == [E, FF]
        assert tensor(mixed, ln + "ffn_down.weight").shape == [FF, E]
        hd = HC * E
        for mixer in ("attn", "ffn"):
            assert tensor(mixed, ln + f"hc_{mixer}_norm.weight").shape == [hd]
            assert tensor(mixed, ln + f"hc_{mixer}_down.weight").shape == [hd, HC_LR]
            assert tensor(mixed, ln + f"hc_{mixer}_up.weight").shape == [HC_LR, hd]
            assert tensor(mixed, ln + f"hc_{mixer}_inject.weight").shape == [hd, HC]
    # the attention layer: q holds q | gate, K/V are the KV-head width, O maps back
    a = f"blk.{LAYERS - 1}."
    assert tensor(mixed, a + "attn_q.weight").shape == [E, ATT_HEAD * 2 * ATT_DIM]
    assert tensor(mixed, a + "attn_k.weight").shape == [E, ATT_KV * ATT_DIM]
    assert tensor(mixed, a + "attn_v.weight").shape == [E, ATT_KV * ATT_DIM]
    assert tensor(mixed, a + "attn_output.weight").shape == [ATT_HEAD * ATT_DIM, E]
    assert tensor(mixed, a + "attn_q_norm.weight").shape == [ATT_DIM]
    assert tensor(mixed, a + "attn_k_norm.weight").shape == [ATT_DIM]
    # the output side: the final mixer, and the tied embedding/head
    assert tensor(mixed, "output_hc_down.weight").shape == [HC * E, HC_LR]
    assert tensor(mixed, "output_hc_up.weight").shape == [HC_LR, HC * E]
    assert tensor(mixed, "token_embd.weight").shape == [E, VOCAB]
    # no MoE tensors anywhere
    assert not [t for t in mixed.tensors if "_exps" in t.name or "ffn_gate_inp" in t.name]


def test_default_fixture_is_still_gdn_only(tmp_path):
    gguf = GGUFFile(build(tmp_path / "hc-tiny.gguf"))
    assert not [t for t in gguf.tensors if t.name.endswith("attn_q.weight")]
    for layer in range(LAYERS):
        assert [t for t in gguf.tensors if t.name == f"blk.{layer}.attn_qkv.weight"]
