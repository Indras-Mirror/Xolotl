"""A tiny SYNTHETIC hyper-connection model: the dense engine's gated-residual path, exercised.

This is not a model and its weights mean nothing.  It exists because the hyper-connection path in
src/core/dense_model.cpp cannot be reached by any real file we have - Qwen4-27B is unreleased, and
every Qwen3.8 file uses a plain pre-norm residual.  The fixture is the smallest GGUF that makes the
engine load the gated residual, run a forward pass and check itself:

    python3 tools/qwen4_hc_fixture.py /tmp/hc-tiny.gguf
    ./build-xolotl/xolotl-dense --check /tmp/hc-tiny.gguf
    # expect "hyper-connections: N streams of E, rank R", a bit-identical chunk comparison, and
    #        "check: all passed"

Shape choices are the ENGINE's constraints, not a real model's: GDN layers only (so no attention
kernel is involved), embedding/FF widths a multiple of 256, ssm.state_size 128 and conv_kernel 4
(the GDN kernels' compiled shape), and an even hc*n_embd.  Every layer is GDN, which no real file
is - that keeps the fixture down to one mixer implementation.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter  # noqa: E402

E, VOCAB, FF = 256, 24000, 256     # the vocab clears --check's random token ids (1000..21000)
SSM_STATE, SSM_K, SSM_V, CONV_K = 128, 1, 2, 4
HC, HC_LR = 2, 8
LAYERS = 4
CONV = 2 * SSM_STATE * SSM_K + SSM_STATE * SSM_V
VDIM = SSM_STATE * SSM_V


def build(path: pathlib.Path, hc: int = HC, hc_lr: int = HC_LR, seed: int = 7) -> pathlib.Path:
    rng = np.random.default_rng(seed)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def q(w, name, rows, cols):          # GGUF shape [cols, rows]; Q2_0 blocks run along cols
        w.add_q2_0(name, r(rows, cols), [cols, rows])

    w = GGUFWriter()
    a = "qwen35"
    p = a + "."
    w.add("general.architecture", a, "string")
    for k, v in [("block_count", LAYERS), ("embedding_length", E),
                 ("feed_forward_length", FF), ("attention.head_count", 8),
                 ("attention.head_count_kv", 1), ("attention.key_length", 32),
                 ("rope.dimension_count", 8), ("ssm.conv_kernel", CONV_K),
                 ("ssm.state_size", SSM_STATE), ("ssm.group_count", SSM_K),
                 ("ssm.time_step_rank", SSM_V),
                 ("hyper_connection.count", hc), ("hyper_connection.low_rank", hc_lr)]:
        w.add(p + k, v, "u32")
    w.add("tokenizer.ggml.tokens", [f"t{i}" for i in range(VOCAB)], "array:string")
    w.add("tokenizer.ggml.eos_token_id", 0, "u32")

    q(w, "token_embd.weight", VOCAB, E)          # tied head: the engine reuses it
    hd = hc * E
    w.add_f32("output_hc_norm.weight", np.ones(hd, dtype=np.float32))
    w.add_f32("output_hc_down.weight", r(hd, hc_lr))
    w.add_f32("output_hc_up.weight", r(hc_lr, hd))

    for layer in range(LAYERS):
        ln = f"blk.{layer}."
        q(w, ln + "attn_qkv.weight", CONV, E)
        q(w, ln + "attn_gate.weight", VDIM, E)
        q(w, ln + "ssm_out.weight", E, VDIM)
        w.add_f32(ln + "ssm_conv1d.weight", r(CONV_K, CONV))
        w.add_f32(ln + "ssm_alpha.weight", r(E, SSM_V))
        w.add_f32(ln + "ssm_beta.weight", r(E, SSM_V))
        w.add_f32(ln + "ssm_a", np.full(SSM_V, -1.0, dtype=np.float32))
        w.add_f32(ln + "ssm_dt.bias", np.zeros(SSM_V, dtype=np.float32))
        w.add_f32(ln + "ssm_norm.weight", np.ones(SSM_STATE, dtype=np.float32))
        q(w, ln + "ffn_gate.weight", FF, E)
        q(w, ln + "ffn_up.weight", FF, E)
        q(w, ln + "ffn_down.weight", E, FF)
        for mixer in ("attn", "ffn"):
            w.add_f32(ln + f"hc_{mixer}_norm.weight", np.ones(hd, dtype=np.float32))
            w.add_f32(ln + f"hc_{mixer}_down.weight", r(hd, hc_lr))
            w.add_f32(ln + f"hc_{mixer}_up.weight", r(hc_lr, hd))
            w.add_f32(ln + f"hc_{mixer}_inject.weight", r(hd, hc))
    return w.write(path)


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hc-tiny.gguf")
    print(build(out), f"({out.stat().st_size / 1048576:.1f} MiB)")
