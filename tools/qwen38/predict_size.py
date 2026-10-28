#!/usr/bin/env python3
"""Predict a llama-quantize output size from the --tensor-type-file it will be given.

llama-quantize writes as it goes; if the volume fills at 80% the run is lost and the
partial file can still look plausible. This resolves each tensor the way
llama-quant.cpp does (first matching rule wins, else the final ftype argument) and
sums the bytes, so the space gate runs BEFORE a byte is written.

  predict_size.py --types TT.txt --gguf MODEL.gguf --ftype Q8_0 [--ceiling-gib 77]
"""
import argparse
import json
import re
import struct
import sys

BPW = {"Q8_0": 8.5, "IQ4_NL": 4.5, "Q2_0": 2.25, "PTQ1_0": 1.75,
       "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5,
       "IQ3_XXS": 3.0625, "IQ3_S": 3.4375, "Q3_K": 3.4375, "F32": 32.0, "F16": 16.0,
       "BF16": 16.0}
BLOCK = {"Q8_0": 32, "IQ4_NL": 32, "Q2_0": 64, "PTQ1_0": 128,
         "IQ2_XXS": 256, "IQ2_XS": 256, "IQ2_S": 256,
         "IQ3_XXS": 256, "IQ3_S": 256, "Q3_K": 256}
# types llama-quantize leaves alone: norms/1-D tensors and the router stay F32
UNTYPED = {"F32", "F16", "BF16"}


def read_dir(path):
    """(name, n_elements, current_type) for every tensor, without loading payloads."""
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    _, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    sc = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(n_kv):
        f.read(struct.unpack("<Q", f.read(8))[0])
        t, = struct.unpack("<I", f.read(4))
        if t == 8:
            f.read(struct.unpack("<Q", f.read(8))[0])
        elif t == 9:
            at, = struct.unpack("<I", f.read(4))
            al, = struct.unpack("<Q", f.read(8))
            if at == 8:
                for _ in range(al):
                    f.read(struct.unpack("<Q", f.read(8))[0])
            else:
                f.read(sc[at] * al)
        else:
            f.read(sc[t])
    out = []
    for _ in range(n_tensors):
        n = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        tt, _off = struct.unpack("<IQ", f.read(12))
        n_el = 1
        for d in dims:
            n_el *= int(d)
        out.append((n, n_el, nd, tt))
    return out


def rules(path):
    r = []
    for line in open(path):
        line = line.strip()
        if line and "=" in line:
            pat, ty = line.rsplit("=", 1)
            r.append((re.compile(pat, re.I), ty.upper()))
    return r


def predict(types_path, gguf, ftype, ptq1_targets=()):
    rs = rules(types_path)
    total = 0
    per_type = {}
    untyped = 0
    for name, n_el, nd, _cur in read_dir(gguf):
        if nd < 2 or not name.endswith("weight"):
            total += n_el * 4
            untyped += 1
            continue
        ty = ftype.upper()
        for pat, t in rs:
            if pat.search(name):
                ty = t
                break
        if ty in UNTYPED:
            total += n_el * 4
            untyped += 1
            continue
        if ty not in BLOCK:
            raise SystemExit(f"{name}: unknown type {ty}")
        if n_el % BLOCK[ty]:
            raise SystemExit(f"{name}: {n_el} elements is not a multiple of {ty}'s block {BLOCK[ty]}")
        b = n_el * BPW[ty] / 8
        total += b
        per_type[ty] = per_type.get(ty, 0) + b
    return int(total), per_type, untyped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ftype", default="Q8_0")
    ap.add_argument("--ceiling-gib", type=float, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    total, per, untyped = predict(a.types, a.gguf, a.ftype)
    gib = total / 2**30
    if a.json:
        print(json.dumps({"bytes": total, "gib": gib,
                          "per_type_gib": {k: v / 2**30 for k, v in sorted(per.items())},
                          "untyped_tensors": untyped}))
    else:
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<8} {v / 2**30:8.3f} GiB")
        print(f"  {'F32 1-D/other':<8} ({untyped} tensors)")
        print(f"  PREDICTED OUTPUT: {gib:.3f} GiB ({total} bytes)")
    if a.ceiling_gib and gib > a.ceiling_gib:
        raise SystemExit(f"predicted {gib:.3f} GiB exceeds the {a.ceiling_gib} GiB ceiling")


if __name__ == "__main__":
    main()
