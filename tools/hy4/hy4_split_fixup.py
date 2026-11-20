#!/usr/bin/env python3
"""Join the converter's N main shards and the streamed Engram shard into one split set of N+1.

    hy4_split_fixup.py DIR PREFIX [--apply]

DIR holds PREFIX-000kk-of-000NN.gguf (k = 1..N, from convert_hf_to_gguf --split-max-size, split.count N,
split.tensors.count = main tensors) and PREFIX-00002-of-00002.gguf (hy4_engram_gguf.py, 1 tensor). The split.* values
are patched in place (same widths: UINT16 / UINT16 / INT32) and the files renamed to -of-(N+1), the Engram one last.
Without --apply it only prints the plan. Refuses to run while any .part file is left.
"""
import argparse
import glob
import os
import re
import struct
import sys


def kv_offsets(path):
    """{key: (value_offset, type)} for the scalar KVs of a GGUF header, plus the tensor count."""
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_val(t):
            if t == 8:
                n, = struct.unpack("<Q", f.read(8)); f.seek(n, 1)
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    for _ in range(n):
                        m, = struct.unpack("<Q", f.read(8)); f.seek(m, 1)
                else:
                    f.seek(sizes[et] * n, 1)
            else:
                f.seek(sizes[t], 1)

        for _ in range(n_kv):
            kl, = struct.unpack("<Q", f.read(8))
            k = f.read(kl).decode()
            t, = struct.unpack("<I", f.read(4))
            out[k] = (f.tell(), t)
            skip_val(t)
    return out, n_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("prefix")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if glob.glob(os.path.join(a.dir, a.prefix + "-*.part")):
        sys.exit("unfinished .part files left - let the conversion finish first")
    eng = os.path.join(a.dir, f"{a.prefix}-00002-of-00002.gguf")
    pat = re.compile(re.escape(a.prefix) + r"-(\d{5})-of-(\d{5})\.gguf$")
    main_files = sorted(p for p in glob.glob(os.path.join(a.dir, a.prefix + "-*-of-*.gguf"))
                        if pat.search(p) and not p.endswith("-00002-of-00002.gguf"))
    if not main_files:
        sys.exit("no main shards")
    n = int(pat.search(main_files[0]).group(2))
    if len(main_files) != n or not os.path.exists(eng):
        sys.exit(f"expected {n} main shards + {eng}, found {len(main_files)} main")
    total_main = 0
    for i, p in enumerate(main_files):
        kv, n_t = kv_offsets(p)
        assert int(pat.search(p).group(1)) == i + 1
        total_main += n_t
    _kv, n_eng = kv_offsets(eng)
    total = total_main + n_eng
    count = n + 1
    print(f"{n} main shards ({total_main} tensors) + Engram ({n_eng}) -> {count} shards, {total} tensors")

    plan = [(p, i) for i, p in enumerate(main_files)] + [(eng, n)]
    for p, no in plan:
        dst = os.path.join(a.dir, f"{a.prefix}-{no + 1:05d}-of-{count:05d}.gguf")
        print(f"  {os.path.basename(p)} -> {os.path.basename(dst)}  split.no {no} count {count} tensors {total}")
        if not a.apply:
            continue
        kv, _ = kv_offsets(p)
        want = {"split.no": (2, "<H", no), "split.count": (2, "<H", count), "split.tensors.count": (5, "<i", total)}
        with open(p, "r+b") as f:
            for k, (t, fmt, v) in want.items():
                off, typ = kv[k]
                assert typ == t, (p, k, typ)
                f.seek(off)
                f.write(struct.pack(fmt, v))
            f.flush()
            os.fsync(f.fileno())
        os.rename(p, dst)
    if not a.apply:
        print("plan only; re-run with --apply")


if __name__ == "__main__":
    main()
