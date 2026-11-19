#!/usr/bin/env python3
"""Write the Hy4-flash Engram tables as the second shard of a split GGUF, streamed in row chunks.

The public release ships the four tables as BF16 (660,006,416 rows x 96 together, 127 GB); llama.cpp's converter
loads and concatenates them whole, which does not fit in RAM. This streams them instead, in the encoding Tencent's
AngelSlim/Hy4-flash-GGUF uses: F8_E4M3, each row scaled so its max |x| is 448, in BF16 arithmetic
(x / (rowmax / 448)). Byte-exact against Tencent's shard on the rows checked (see --check). The graph RMS-norms every
looked-up row, so the per-row scale does not change the model.

    hy4_engram_gguf.py MODEL_DIR OUT-00002-of-00002.gguf --tensors-total 1380 [--check ROW:FILE ...]

The first shard is the converter's output with HY4_ENGRAM_SPLIT=1 (no table, split.* keys set).
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

ALIGN = 32
E4M3_MAX = 448.0
GGML_TYPE_F8_E4M3 = 43
NAME = "per_layer_token_embd.weight"


def gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def header(rows: int, dim: int, total: int) -> bytes:
    h = b"GGUF" + struct.pack("<IQQ", 3, 1, 3)
    h += gguf_str("split.no") + struct.pack("<IH", 2, 1)              # UINT16
    h += gguf_str("split.count") + struct.pack("<IH", 2, 2)           # UINT16
    h += gguf_str("split.tensors.count") + struct.pack("<Ii", 5, total)  # INT32
    h += gguf_str(NAME) + struct.pack("<I", 2) + struct.pack("<QQ", dim, rows) + struct.pack("<IQ", GGML_TYPE_F8_E4M3, 0)
    return h + b"\0" * ((-len(h)) % ALIGN)


def encode(x: torch.Tensor) -> tuple[np.ndarray, int]:
    """BF16 rows -> E4M3 bytes, per-row max scaled to 448 (BF16 arithmetic, as Tencent's GGUF). Zero rows stay 0."""
    am = x.abs().amax(1, keepdim=True)
    zero = am == 0
    y = x / torch.where(zero, torch.ones_like(am), am / E4M3_MAX)
    return y.to(torch.float8_e4m3fn).view(torch.uint8).numpy(), int(zero.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--tensors-total", type=int, required=True, help="split.tensors.count (tensors in both shards)")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--check", nargs="*", default=[], help="ROW:FILE - Tencent's E4M3 bytes starting at global row ROW")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    layers = [int(i) - 1 for i in sorted(int(i) for i in tc["engram_layer_ids"])]   # 1-based in the config
    wmap = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    names = [f"model.layers.{l}.engram_layer.multi_head_embedding.embedding.weight" for l in layers]

    shapes = []
    for n in names:
        with safe_open(os.path.join(a.model_dir, wmap[n]), "pt") as f:
            shapes.append(f.get_slice(n).get_shape())
    dim = shapes[0][1]
    assert all(s[1] == dim for s in shapes), shapes
    rows = sum(s[0] for s in shapes)
    print(f"engram layers {layers} rows {[s[0] for s in shapes]} = {rows} x {dim}", flush=True)

    checks = []
    for c in a.check:
        r, p = c.split(":", 1)
        ref = np.fromfile(p, dtype=np.uint8).reshape(-1, dim)
        checks.append((int(r), ref))

    tmp = a.out + ".part"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    hdr = header(rows, dim, a.tensors_total)
    os.write(fd, hdr)
    base_row, zeros, t0, written = 0, 0, time.time(), 0
    mism = {i: [0, 0] for i in range(len(checks))}
    for n, (nrows, _) in zip(names, shapes):
        with safe_open(os.path.join(a.model_dir, wmap[n]), "pt") as f:
            s = f.get_slice(n)
            for r0 in range(0, nrows, a.chunk_rows):
                r1 = min(nrows, r0 + a.chunk_rows)
                b, z = encode(s[r0:r1])
                zeros += z
                g0, g1 = base_row + r0, base_row + r1
                for i, (cr, ref) in enumerate(checks):
                    lo, hi = max(g0, cr), min(g1, cr + len(ref))
                    if lo < hi:
                        ours, theirs = b[lo - g0:hi - g0], ref[lo - cr:hi - cr]
                        mism[i][0] += int((ours != theirs).sum())
                        mism[i][1] += ours.size
                mv = memoryview(np.ascontiguousarray(b)).cast("B")
                while mv:
                    k = os.write(fd, mv)
                    mv = mv[k:]
                written += b.size
                if (g1 // a.chunk_rows) % 20 == 0:
                    os.fdatasync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    el = time.time() - t0
                    print(f"  {g1}/{rows} rows  {written / 2**30:.1f} GiB  {written / 2**20 / el:.0f} MiB/s", flush=True)
        base_row += nrows
    os.fdatasync(fd)
    os.close(fd)
    assert base_row == rows
    size = os.path.getsize(tmp)
    assert size == len(hdr) + rows * dim, (size, len(hdr), rows * dim)
    print(f"zero rows {zeros}", flush=True)
    bad = False
    for i, (cr, ref) in enumerate(checks):
        d, t = mism[i]
        print(f"check row {cr}: {t} bytes compared, {d} differ", flush=True)
        bad |= d != 0 or t == 0
    if bad:
        print("CHECK FAILED - leaving", tmp, flush=True)
        sys.exit(1)
    os.rename(tmp, a.out)
    print(f"done {a.out} {size} bytes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
