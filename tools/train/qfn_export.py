#!/usr/bin/env python3
"""Export one QFN GDN layer's tensors from the pack to a flat F32 file for the C++ trainer.

usage: qfn_export.py <pack.gguf> <layer> <out.bin>

Writes, in this order, F32 little-endian:
    attn_qkv [D, 2*hd*nk + hd*nv]      (in=D, ggml ne order [in, out])
    attn_gate [D, hd*nv]               (z)
    ssm_conv1d [d_conv, channels]
    ssm_beta [D, nv]  ssm_alpha [D, nv]
    ssm_a [nv]  ssm_dt [nv]  ssm_norm [hd]
    ssm_out [hd*nv, D]
followed by a 5 x int32 header at the END (D, hd, nk, nv, d_conv) so the reader can seek back.

The pack stores the separate qkv/z and beta/alpha projections (not the *_cat variants); the trainer concatenates
qkv + z and beta + alpha, which is exactly the row order the model's *_cat path uses.
"""
import sys

import numpy as np

sys.path.insert(0, "/media/Crucial1TB/xolotl-train/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from gguf import GGML_QUANT_SIZES  # noqa: E402


def tensor_f32(reader, name):
    t = next((x for x in reader.tensors if x.name == name), None)
    if t is None:
        raise SystemExit(f"missing tensor {name}")
    data = t.data
    if t.tensor_type.name != "F32":
        data = dequantize(data, t.tensor_type)
    return np.asarray(data, dtype=np.float32)      # ne0-fastest, i.e. ggml order


def main(pack, layer, out):
    r = GGUFReader(pack)
    f = r.fields

    def kv(name):
        return f[name].contents()

    D = int(kv("qwen4exp.embedding_length"))
    hd = int(kv("qwen4exp.ssm.state_size"))
    nk = int(kv("qwen4exp.ssm.group_count"))
    nv = int(kv("qwen4exp.ssm.time_step_rank"))
    d_conv = int(kv("qwen4exp.ssm.conv_kernel"))
    pre = f"blk.{layer}."

    order = [("attn_qkv.weight", None), ("attn_gate.weight", None), ("ssm_conv1d.weight", None),
             ("ssm_beta.weight", None), ("ssm_alpha.weight", None),
             ("ssm_a", None), ("ssm_dt.bias", None), ("ssm_norm.weight", None), ("ssm_out.weight", None)]
    total = 0
    with open(out, "wb") as fh:
        for name, _ in order:
            a = tensor_f32(r, pre + name)
            fh.write(a.tobytes())
            total += a.size
            print(f"{pre + name:26s} {a.size:9d} floats {a.dtype}")
        fh.write(np.asarray([D, hd, nk, nv, d_conv], dtype=np.int32).tobytes())
    print(f"wrote {out}: {total} floats + header  (D {D} hd {hd} nk {nk} nv {nv} d_conv {d_conv})")


def export_embd(pack, ids, D, out):
    """Write [D, len(ids)] F32 embedding rows for the given token ids (ggml order: D on dim0)."""
    r = GGUFReader(pack)
    t = next(x for x in r.tensors if x.name == "token_embd.weight")
    blk, tsize = GGML_QUANT_SIZES[t.tensor_type]
    bpr = D // blk * tsize                      # bytes per row (one token)
    raw = np.memmap(pack, dtype=np.uint8, mode="r")
    rows = []
    for i in ids:
        lo = t.data_offset + i * bpr
        rows.append(np.asarray(dequantize(raw[lo:lo + bpr], t.tensor_type), dtype=np.float32))
    a = np.stack(rows, axis=1)                  # [D, len(ids)]
    a.tofile(out)
    print(f"wrote {out}: {a.size} floats ({len(ids)} tokens x {D}), rms {float(np.sqrt((a**2).mean())):.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4]:
        ids = [int(v) for v in sys.argv[4].split(",")]
        r = GGUFReader(sys.argv[1])
        D = int(r.fields["qwen4exp.embedding_length"].contents())
        export_embd(sys.argv[1], ids, D, sys.argv[3] + ".embd.f32")
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0,
             sys.argv[3] if len(sys.argv) > 3 else "/tmp/qfn_l0.f32")
