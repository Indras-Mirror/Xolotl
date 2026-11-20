#!/usr/bin/env python3
"""Check that an Inkling-2 GGUF keeps the official expert weights exactly.

NVFP4 layers: dequant(GGUF NVFP4 block) * GGUF `.scale`[e]  ==  E2M1 * E4M3 * amax[e]/denominator  (official formula).
MXFP8 layers: GGUF BF16 == E4M3 * 2^(UE8M0-127)  (BF16 holds every such value exactly).
gate_up rows: the checkpoint interleaves gate/up (row 2i = gate i, 2i+1 = up i); the GGUF stores [gate; up].

usage: check_nvfp4_exact.py <official dir> <gguf> [layer ...]   (default layers 0 19 20 23, experts 0 63 127)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/Crucial1TB/llama.cpp-inkling/gguf-py")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from safetensors import safe_open  # noqa: E402

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def e4m3_table():
    c = np.arange(256, dtype=np.uint8)
    e = ((c & 0x7F) >> 3).astype(np.int32)
    m = (c & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1 + m / 8, e - 7)).astype(np.float32)
    v = np.where(c & 0x80, -v, v)
    v[(c & 0x7F) == 0x7F] = np.nan
    return v.astype(np.float32)


E4M3 = e4m3_table()


def main():
    src, gg = Path(sys.argv[1]), sys.argv[2]
    layers = [int(x) for x in sys.argv[3:]] or [0, 19, 20, 23]
    experts = [0, 63, 127]
    r = gguf.GGUFReader(gg)
    t = {x.name: x for x in r.tensors}
    import torch
    ok = True
    with safe_open(src / "model-00001-of-00001.safetensors", "pt") as f:
        keys = set(f.keys())
        for L in layers:
            for hf, gn in (("w13_weight", "ffn_gate_up_exps"), ("w2_weight", "ffn_down_exps")):
                base = f"model.llm.layers.{L}.mlp.experts.{hf}"
                gt = t[f"blk.{L}.{gn}.weight"]
                nvfp4 = base + "_amax" in keys
                W = f.get_tensor(base)
                S = f.get_tensor(base + "_scale")
                if nvfp4:
                    amax = f.get_tensor(base + "_amax").float().numpy()
                    den = np.float32(f.get_tensor(base + "_outer_scale_denominator").item())
                    gs = t[f"blk.{L}.{gn}.scale"].data.astype(np.float32)
                for e in experts:
                    w = W[e].view(torch.uint8).numpy()
                    s = S[e].view(torch.uint8).numpy()
                    if nvfp4:
                        ref = np.empty((w.shape[0], w.shape[1] * 2), np.float32)
                        ref[:, 0::2] = E2M1[w & 15]
                        ref[:, 1::2] = E2M1[w >> 4]
                        ref.reshape(ref.shape[0], -1, 16)[:] *= E4M3[s][:, :, None]
                        ref *= np.float32(amax[e]) / den
                    else:
                        ref = E4M3[w].copy()
                        ref.reshape(ref.shape[0], -1, 32)[:] *= np.ldexp(np.ones(s.shape, np.float32), s.astype(np.int32) - 127)[:, :, None]
                    if hf == "w13_weight":
                        ref = np.concatenate([ref[0::2], ref[1::2]], axis=0)
                    raw = np.asarray(gt.data)[e]
                    if gt.tensor_type == gguf.GGMLQuantizationType.NVFP4:
                        got = dequantize(raw, gt.tensor_type).reshape(ref.shape).astype(np.float32) * gs[e]
                    else:
                        got = dequantize(raw, gt.tensor_type).reshape(ref.shape).astype(np.float32)
                    bad = int(np.count_nonzero(got != ref))
                    md = float(np.max(np.abs(got - ref)))
                    print(f"L{L:2d} {gn:16s} e{e:3d} {gt.tensor_type.name:6s} mismatches {bad:9d} / {ref.size}  max|d| {md:.3g}")
                    ok &= bad == 0
    print("EXACT" if ok else "NOT EXACT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
