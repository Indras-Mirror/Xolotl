#!/usr/bin/env python3
"""Build the DSpark (MTP) drafter into an Inkling-2-Mini model GGUF, so one file carries both.

    python tools/inkling/embed_dspark.py MODEL.gguf DRAFTER.gguf OUT.gguf [--q8]

The model's header, keys and tensor data are copied byte for byte (any tensor type, including ones gguf-py does not
know); the drafter's 54 tensors are appended as `dspark.<name>` and its `dflash.*` keys are added beside the model's.
`xolotl-inkling --mtp` loads the drafter from such a file (`--dspark DRAFTER.gguf` still works with a separate file).
--q8 stores the drafter's 2-D weights as Q8_0 (about half the size); norms, biases and the 4x4 fuse stay F32.
"""
import argparse
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
# value type -> struct format for the fixed-size scalars
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
T_STRING, T_ARRAY = 8, 9
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


class Reader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"{self.path}: truncated header")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8")

    def skip_value(self, t):
        if t in SCALAR:
            self.read(struct.calcsize(SCALAR[t]))
        elif t == T_STRING:
            self.read(self.u64())
        elif t == T_ARRAY:
            at, n = self.u32(), self.u64()
            for _ in range(n):
                self.skip_value(at)
        else:
            raise ValueError(f"{self.path}: unknown GGUF value type {t}")


def parse(path):
    """-> dict(version, kvs=[(key, raw bytes of key+type+value, type, value or None)], tensors=[...], data_start,
    alignment, reader).  Tensor: dict(name, dims, type, offset)."""
    r = Reader(path)
    if r.read(4) != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    version = r.u32()
    if version < 2:
        raise ValueError(f"{path}: GGUF v{version} is too old")
    n_tensors, n_kv = r.u64(), r.u64()
    kvs, alignment = [], 32
    for _ in range(n_kv):
        start = r.f.tell()
        key = r.string()
        t = r.u32()
        vstart = r.f.tell()
        r.skip_value(t)
        end = r.f.tell()
        r.f.seek(start)
        raw = r.read(end - start)
        val = None
        if t in SCALAR:
            val = struct.unpack(SCALAR[t], raw[vstart - start:])[0]
        if key == "general.alignment":
            alignment = int(val)
        kvs.append((key, raw, t, val))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    hdr_end = r.f.tell()
    data_start = (hdr_end + alignment - 1) // alignment * alignment
    return {"version": version, "kvs": kvs, "tensors": tensors, "data_start": data_start, "alignment": alignment,
            "reader": r}


def enc_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def tensor_info(name, dims, ttype, off):
    return enc_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + \
        struct.pack("<I", ttype) + struct.pack("<Q", off)


def to_f32(raw, ttype, n):
    if ttype == GGML_F32:
        return np.frombuffer(raw, dtype=np.float32, count=n).copy()
    if ttype == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)
    if ttype == GGML_BF16:
        return (np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"drafter tensor type {ttype} is not F32/F16/BF16")


def q8_0(x):
    """ggml's quantize_row_q8_0_ref: per 32 values, d = amax/127 (stored f16), q = round(x/d)."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    idd = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0).astype(np.float32)
    q = np.round(b * idd[:, None]).astype(np.int8)   # roundf: half away from zero; np.round is half-to-even -
    # the two differ only at exact .5 ties, harmless for a drafter (it is not bit-compared against ggml)
    out = np.empty(b.shape[0], dtype=[("d", "<f2"), ("q", "i1", 32)])
    out["d"] = d.astype(np.float16)
    out["q"] = q
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("drafter")
    ap.add_argument("out")
    ap.add_argument("--q8", action="store_true", help="store the drafter's 2-D weights as Q8_0")
    ap.add_argument("--prefix", default="dspark.")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")

    m = parse(a.model)
    d = parse(a.drafter)
    d_arch = None
    for k, raw, t, v in d["kvs"]:
        if k == "general.architecture":
            r = raw[len(enc_string(k)) + 4:]
            d_arch = r[8:].decode()
    if d_arch != "dflash":
        sys.exit(f"{a.drafter} is not a DSpark drafter (general.architecture = {d_arch})")
    if any(t["name"].startswith(a.prefix) for t in m["tensors"]):
        sys.exit(f"{a.model} already carries {a.prefix}* tensors")
    mkeys = {k for k, *_ in m["kvs"]}
    add_kvs = [kv for kv in d["kvs"] if kv[0].startswith("dflash.")]
    clash = [k for k, *_ in add_kvs if k in mkeys]
    if clash:
        sys.exit(f"key clash: {clash[:5]}")
    al = m["alignment"]

    # the model's data blob: everything from its data start to the end of the file, unchanged
    m_blob = m["reader"].size - m["data_start"]
    base = (m_blob + al - 1) // al * al

    # the drafter's tensors, (re)encoded, laid out after the model's data
    dr = d["reader"]
    dt = sorted(d["tensors"], key=lambda t: t["offset"])
    ends = [t["offset"] for t in dt[1:]] + [dr.size - d["data_start"]]
    new = []   # (name, dims, type, bytes)
    for t, end in zip(dt, ends):
        dr.f.seek(d["data_start"] + t["offset"])
        raw = dr.f.read(end - t["offset"])
        n = int(np.prod(t["dims"]))
        ttype, data = t["type"], None
        if a.q8 and len(t["dims"]) == 2 and t["dims"][0] % 32 == 0 and t["type"] in (GGML_F32, GGML_F16, GGML_BF16):
            ttype, data = GGML_Q8_0, q8_0(to_f32(raw, t["type"], n))
        else:
            esz = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2}.get(t["type"])
            if esz is None:
                sys.exit(f"drafter tensor {t['name']} has type {t['type']}; only F32/F16/BF16 drafters are supported")
            data = raw[:n * esz]
        new.append((a.prefix + t["name"], t["dims"], ttype, data))
    offs, cur = [], 0
    for _, _, _, b in new:
        offs.append(cur)
        cur = (cur + len(b) + al - 1) // al * al

    hdr = bytearray()
    hdr += GGUF_MAGIC + struct.pack("<I", m["version"])
    hdr += struct.pack("<QQ", len(m["tensors"]) + len(new), len(m["kvs"]) + len(add_kvs) + 1)
    for _, raw, _, _ in m["kvs"]:
        hdr += raw
    for _, raw, _, _ in add_kvs:
        hdr += raw
    hdr += enc_string("dspark.embedded") + struct.pack("<I", 7) + struct.pack("<?", True)
    for t in m["tensors"]:
        hdr += tensor_info(t["name"], t["dims"], t["type"], t["offset"])
    for (name, dims, ttype, _), o in zip(new, offs):
        hdr += tensor_info(name, dims, ttype, base + o)
    pad = (len(hdr) + al - 1) // al * al - len(hdr)
    hdr += b"\0" * pad

    tmp = a.out + ".tmp"
    mr = m["reader"]
    with open(tmp, "wb") as f:
        f.write(hdr)
        mr.f.seek(m["data_start"])
        left = m_blob
        while left:
            chunk = mr.f.read(min(left, 64 << 20))
            if not chunk:
                sys.exit("short read on the model data")
            f.write(chunk)
            left -= len(chunk)
        f.write(b"\0" * (base - m_blob))
        for (_, _, _, b), o in zip(new, offs):
            assert f.tell() == len(hdr) + base + o
            f.write(b)
            f.write(b"\0" * ((len(b) + al - 1) // al * al - len(b)))
    os.replace(tmp, a.out)
    added = sum(len(b) for *_, b in new)
    print(f"wrote {a.out}: {len(m['tensors'])} model tensors (bytes unchanged) + {len(new)} drafter tensors "
          f"({added / 1e6:.1f} MB{', Q8_0' if a.q8 else ''}), {len(add_kvs)} dflash.* keys")


if __name__ == "__main__":
    main()
