"""Merge llama.cpp-style GGUF split shards (<name>-0000N-of-0000M.gguf) into ONE GGUF file, byte for byte.

The DeepSeek-V4.1 engine's expert tier reads a single-file GGUF (Ds41FileBlobs), while some publishers ship splits
(e.g. lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit). llama.cpp's gguf-split --merge needs every tensor type's block size,
and these files carry Strata's own expert types (PTQ1_0 = 143, Q2_0 maps). This tool never decodes a type: each
tensor's byte length comes from the shard layout itself (the next tensor's offset, or the end of the shard), so the
tensor data is copied exactly. Metadata comes from shard 1 minus the split.* keys; every tensor is re-placed at
general.alignment. Writes <out>.partial and renames it when done, then spot-checks tensors against the shards.

  python gguf_merge_shards.py <shard-00001-of-0000N.gguf> <out.gguf>
"""
import os
import random
import re
import struct
import sys
import time

FIRST, OUT = sys.argv[1], sys.argv[2]
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64())


def skip_value(r, t, raw):
    """Read one KV value, appending its raw bytes to `raw`; returns the decoded value for scalars/strings."""
    if t in SCALAR:
        b = r.read(SCALAR[t])
        raw += b
        return struct.unpack("<" + {1: "B", 2: "H", 4: "I", 8: "Q"}[SCALAR[t]], b)[0] if t not in (6, 12) else None
    if t == 8:
        n = r.read(8)
        s = r.read(struct.unpack("<Q", n)[0])
        raw += n + s
        return s.decode("utf-8", "replace")
    if t == 9:
        hdr = r.read(12)
        raw += hdr
        at, n = struct.unpack("<IQ", hdr)
        for _ in range(n):
            skip_value(r, at, raw)
        return None
    raise ValueError(f"unknown GGUF value type {t}")


def parse(path):
    r = Reader(path)
    if r.read(4) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    ver, nt, nkv = r.u32(), r.u64(), r.u64()
    kvs = []   # (key, raw bytes of key+type+value, decoded value)
    for _ in range(nkv):
        kb = r.read(8)
        key = r.read(struct.unpack("<Q", kb)[0])
        t = r.read(4)
        raw = bytearray(kb + key + t)
        val = skip_value(r, struct.unpack("<I", t)[0], raw)
        kvs.append((key.decode(), bytes(raw), val))
    tensors = []   # (name bytes, ndims, dims bytes, type, offset)
    for _ in range(nt):
        name = r.string()
        nd = r.u32()
        dims = r.read(8 * nd)
        typ = r.u32()
        off = r.u64()
        tensors.append((name, nd, dims, typ, off))
    header_end = r.f.tell()
    r.f.close()
    align = next((v for k, _, v in kvs if k == "general.alignment"), 32) or 32
    data_start = (header_end + align - 1) // align * align
    size = os.path.getsize(path)
    order = sorted(range(nt), key=lambda i: tensors[i][4])
    lengths = [0] * nt
    for j, i in enumerate(order):
        end = tensors[order[j + 1]][4] if j + 1 < nt else size - data_start
        lengths[i] = end - tensors[i][4]
    return ver, kvs, tensors, lengths, data_start, align


m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", FIRST)
if not m:
    sys.exit("first argument must be the -00001-of-0000N.gguf shard")
count = int(m.group(3))
shards = [f"{m.group(1)}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
parsed = [parse(p) for p in shards]
ver, kvs0, _, _, _, align = parsed[0]
kv_out = [raw for k, raw, _ in kvs0 if not k.startswith("split.")]
entries = []   # (shard index, tensor tuple, length)
for si, (_, _, tensors, lengths, _, _) in enumerate(parsed):
    for t, n in zip(tensors, lengths):
        entries.append((si, t, n))
names = [t[0] for _, t, _ in entries]
assert len(names) == len(set(names)), "duplicate tensor names across shards"

# new offsets: each tensor at the next alignment boundary (relative to the data section)
offsets, at = [], 0
for _, _, n in entries:
    offsets.append(at)
    at = (at + n + align - 1) // align * align
header = bytearray(b"GGUF" + struct.pack("<IQQ", ver, len(entries), len(kv_out)))
for raw in kv_out:
    header += raw
for (si, (name, nd, dims, typ, _), _), off in zip(entries, offsets):
    header += struct.pack("<Q", len(name)) + name + struct.pack("<I", nd) + dims + struct.pack("<IQ", typ, off)
data_start = (len(header) + align - 1) // align * align
total = data_start + at
free = os_free = None
try:
    import shutil
    os_free = shutil.disk_usage(os.path.dirname(os.path.abspath(OUT))).free
except OSError:
    pass
if os_free is not None and os_free < total + (1 << 30):
    sys.exit(f"need {total / 1e9:.1f} GB free next to {OUT}, have {os_free / 1e9:.1f} GB")
print(f"{len(shards)} shards, {len(entries)} tensors, {len(kv_out)} keys -> {OUT} ({total / 1e9:.1f} GB)", flush=True)

t0, done = time.time(), 0
handles = [open(p, "rb") for p in shards]
with open(OUT + ".partial", "wb") as w:
    w.write(header)
    w.write(b"\0" * (data_start - len(header)))
    for k, ((si, t, n), off) in enumerate(zip(entries, offsets)):
        pos = w.tell() - data_start
        if pos < off:
            w.write(b"\0" * (off - pos))
        src = handles[si]
        src.seek(parsed[si][4] + t[4])
        left = n
        while left:
            b = src.read(min(left, 64 << 20))
            if not b:
                raise EOFError(f"short read in {shards[si]} for {t[0].decode()}")
            w.write(b)
            left -= len(b)
        done += n
        if k % 100 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(entries)} tensors, {done / 1e9:.1f} GB, {done / max(el, 1e-9) / 1e9:.2f} GB/s", flush=True)
    pos = w.tell() - data_start
    if pos < at:
        w.write(b"\0" * (at - pos))
for h in handles:
    h.close()
os.replace(OUT + ".partial", OUT)

# spot-check: 64 random tensors' first and last MiB against their shard
random.seed(0)
with open(OUT, "rb") as w:
    for k in random.sample(range(len(entries)), min(64, len(entries))):
        si, t, n = entries[k]
        with open(shards[si], "rb") as src:
            for o in sorted({0, max(0, n - (1 << 20))}):
                ln = min(1 << 20, n - o)
                src.seek(parsed[si][4] + t[4] + o)
                w.seek(data_start + offsets[k] + o)
                if src.read(ln) != w.read(ln):
                    sys.exit(f"MISMATCH in {t[0].decode()} at +{o}")
print(f"OK: merged {done / 1e9:.1f} GB in {time.time() - t0:.0f} s; 64 random tensors match their shards")
