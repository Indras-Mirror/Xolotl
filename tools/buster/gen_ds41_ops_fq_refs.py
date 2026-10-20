#!/usr/bin/env python3
"""Write test_ds41_ops reference vectors (x.f32, fp8.f32, fp4e.f32, fp4p.f32) using tools/v41/torch_kernels.

Deterministic (seed 42); no GGUF or 1017 shards required. Optional: pass a directory of FP8 activations and
this script will sample the first 512*R floats from a .safetensors tensor instead of random data.

  python3 tools/buster/gen_ds41_ops_fq_refs.py /tmp/ds41-fq-refs
  build-buster/test_ds41_ops /tmp/ds41-fq-refs
  build-buster/test_ds41_ops /tmp/ds41-fq-refs --cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "v41", "torch_kernels"))

import torch  # noqa: E402
import kernel as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--rows", type=int, default=64, help="number of 512-wide rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    W = 512
    R = args.rows
    x = torch.randn(R, W, dtype=torch.float32) * 0.25
    x0 = x.clone()
    x_fp8 = x.clone()
    K.act_quant(x_fp8, block_size=128, scale_fmt="ue8m0", inplace=True)
    x_fp4e = x.clone()
    # Match ds41_dense FP4_E4M3_16 (block size 16) / FP4_POW2_32 (block size 32).
    K.fp4_act_quant(x_fp4e, block_size=16, inplace=True, scale_dtype=torch.float8_e4m3fn)
    x_fp4p = x.clone()
    K.fp4_act_quant(x_fp4p, block_size=32, inplace=True, scale_dtype=torch.float8_e8m0fnu)
    for name, t in [("x", x0), ("fp8", x_fp8), ("fp4e", x_fp4e), ("fp4p", x_fp4p)]:
        path = os.path.join(args.out_dir, f"{name}.f32")
        t.detach().cpu().numpy().astype(np.float32).tofile(path)
        print(path, t.numel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
