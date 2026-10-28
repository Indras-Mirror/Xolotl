#!/usr/bin/env python3
"""Graft the IQ4_NL PLE table into a text GGUF that was converted with PLE deferred.

Why this is needed
------------------
`convert_ple_deferred.py` deliberately omits `per_layer_token_embd.weight` because the FP8
version of that table is 51.2e9 values and materialising it needs ~205 GB of RAM.  That is fine
for the Strata engine, which takes the table from a second file at serve time (`--ple-gguf`).

It is NOT fine for any llama.cpp tool.  `llama_model_load` calls require_weight() on that
tensor, so `llama-imatrix`, `llama-quantize` and `llama-server` all die at load:

    require_weight: tensor 'per_layer_token_embd.weight' not found

Confirmed 2026-10-27 by loading the converted Q8_0 with llama-imatrix (build 517).  So the
calibration/quantisation half of the pipeline needs the tensor physically in the file.

This appends it, without re-reading or rewriting any existing tensor.

How it stays fast
-----------------
GGUF tensor `offset` fields are relative to the start of the tensor-data section, not to the
file.  Appending one tensor therefore leaves every existing offset valid: we rewrite the header
(tensor count + one new info block), pad to alignment, copy the whole old data section verbatim,
and append the new tensor's bytes at the end.  No tensor is decoded, re-laid-out or re-quantised.

The copy is a straight byte stream, so memory stays flat regardless of file size.

  python3 tools/qwen38/graft_ple.py \
      --model /media/mal/NVME1TB/qwen38/qwen38-q8_0.gguf \
      --ple   "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8/ple/Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf" \
      --out   /media/Crucial1TB/qwen38/qwen38-full-q8_0.gguf
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
ALIGN_DEFAULT = 32
CHUNK = 64 << 20


class Hdr:
    """Header + KV + tensor-infos of a GGUF, with the byte ranges needed to copy them verbatim."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                raise SystemExit(f"{path}: not a GGUF")
            self.version, = struct.unpack("<I", f.read(4))
            self.n_tensors, = struct.unpack("<Q", f.read(8))
            self.n_kv, = struct.unpack("<Q", f.read(8))
            self.kv_start = f.tell()
            self.alignment = ALIGN_DEFAULT
            for _ in range(self.n_kv):
                k = self._rstr(f)
                t, = struct.unpack("<I", f.read(4))
                v = self._rval(f, t)
                if k == "general.alignment" and isinstance(v, int):
                    self.alignment = v
            self.kv_end = f.tell()

            # one pass over the tensor infos; record name/type/dims/offset plus the raw byte
            # range so the new file can copy each info block untouched
            self.infos = []          # (start, end, name, type, dims, offset)
            for _ in range(self.n_tensors):
                start = f.tell()
                name = self._rstr(f)
                nd, = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
                tt, = struct.unpack("<I", f.read(4))
                off, = struct.unpack("<Q", f.read(8))
                self.infos.append((start, f.tell(), name, tt, dims, off))
            self.infos_end = f.tell()

        # GGUF pads the tensor-data section to `general.alignment`
        self.data_start = (self.infos_end + self.alignment - 1) // self.alignment * self.alignment

    # -- primitive readers -------------------------------------------------
    @staticmethod
    def _rstr(f):
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    @staticmethod
    def _rval(f, t):
        # GGUF metadata value types, per gguf-py gguf/constants.py:
        # 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool(1 byte!),
        # 8 str, 9 array, 10 u64, 11 i64, 12 f64
        if t == 0: return struct.unpack("<B", f.read(1))[0]
        if t == 1: return struct.unpack("<b", f.read(1))[0]
        if t == 2: return struct.unpack("<H", f.read(2))[0]
        if t == 3: return struct.unpack("<h", f.read(2))[0]
        if t == 4: return struct.unpack("<I", f.read(4))[0]
        if t == 5: return struct.unpack("<i", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0] != 0
        if t == 8: return Hdr._rstr(f)
        if t == 9:
            at, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [Hdr._rval(f, at) for _ in range(n)]
        if t == 10: return struct.unpack("<Q", f.read(8))[0]
        if t == 11: return struct.unpack("<q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        raise SystemExit(f"graft_ple: unknown KV type {t}")


def tensor_bytes(h: Hdr, name: str) -> tuple[int, int, int, tuple]:
    """(absolute data offset, nbytes, type, dims) for one tensor."""
    for _s, _e, n, tt, dims, off in h.infos:
        if n == name:
            return h.data_start + off, _nbytes(tt, dims), tt, dims
    raise SystemExit(f"graft_ple: {name} not found in {h.path}")


def _nbytes(tt: int, dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    if tt == 20:      # IQ4_NL: 32 values per 18-byte block
        assert n % 32 == 0, n
        return n // 32 * 18
    raise SystemExit(f"graft_ple: unsupported type {tt} for size computation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the PLE-deferred text GGUF")
    ap.add_argument("--ple", required=True, help="the one-tensor IQ4_NL PLE GGUF")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = Hdr(a.model)
    p = Hdr(a.ple)

    if any(n == "per_layer_token_embd.weight" for _s, _e, n, _t, _d, _o in m.infos):
        raise SystemExit("graft_ple: the model already carries per_layer_token_embd.weight")

    ple_off, ple_nbytes, ple_type, ple_dims = tensor_bytes(p, "per_layer_token_embd.weight")
    print(f"model : {m.path}\n  tensors {m.n_tensors}  kv {m.n_kv}  alignment {m.alignment}  "
          f"data starts at {m.data_start}")
    print(f"ple   : {p.path}\n  type {ple_type}  dims {ple_dims}  {ple_nbytes/1e9:.2f} GB")

    # the new tensor info may not fit in the padding that followed the old ones, so recompute
    # where the data section now starts.  GGUF tensor offsets are relative to the data section,
    # so moving it wholesale leaves every existing offset valid.
    nm = b"per_layer_token_embd.weight"
    new_info_len = (8 + len(nm) + 4 + 8 * len(ple_dims) + 4 + 8)
    new_infos_end = m.infos_end + new_info_len
    new_data_start = (new_infos_end + m.alignment - 1) // m.alignment * m.alignment

    old_data_size = os.path.getsize(a.model) - m.data_start
    new_off = (old_data_size + m.alignment - 1) // m.alignment * m.alignment
    out_size = new_data_start + new_off + ple_nbytes
    print(f"  header grows {m.data_start} -> {new_data_start} "
          f"(+{new_data_start - m.data_start} bytes of padding)")

    need = out_size
    free = os.statvfs(os.path.dirname(os.path.abspath(a.out))).f_bavail * \
        os.statvfs(os.path.dirname(os.path.abspath(a.out))).f_frsize
    print(f"out   : {a.out}\n  {out_size/1e9:.2f} GB needed, {free/1e9:.2f} GB free")
    if free < need:
        raise SystemExit("graft_ple: not enough free space")

    # --- header --------------------------------------------------------
    hdr = bytearray()
    hdr += GGUF_MAGIC
    hdr += struct.pack("<I", m.version)
    hdr += struct.pack("<Q", m.n_tensors + 1)
    hdr += struct.pack("<Q", m.n_kv)

    with open(m.path, "rb") as mf, open(a.out, "wb") as of:
        # KV verbatim
        mf.seek(m.kv_start)
        hdr += mf.read(m.kv_end - m.kv_start)
        # old tensor infos verbatim
        for (s, e, *_rest) in m.infos:
            mf.seek(s)
            hdr += mf.read(e - s)
        # the new tensor info
        hdr += struct.pack("<Q", len(nm)) + nm
        hdr += struct.pack("<I", len(ple_dims))
        for d in ple_dims:
            hdr += struct.pack("<Q", int(d))
        hdr += struct.pack("<I", ple_type)
        hdr += struct.pack("<Q", new_off)
        # pad to the (possibly moved) data section
        assert len(hdr) <= new_data_start, (len(hdr), new_data_start)
        hdr += b"\x00" * (new_data_start - len(hdr))
        of.write(hdr)

        # --- old tensor data, verbatim ---------------------------------
        print("  copying model tensor data ...", flush=True)
        mf.seek(m.data_start)
        left = old_data_size
        while left:
            b = mf.read(min(CHUNK, left))
            if not b:
                raise SystemExit("graft_ple: model data truncated")
            of.write(b)
            left -= len(b)
        # alignment padding before the appended tensor
        pad = new_off - old_data_size
        if pad:
            of.write(b"\x00" * pad)

        # --- the PLE table ---------------------------------------------
        print("  copying PLE table ...", flush=True)
        with open(p.path, "rb") as pf:
            pf.seek(ple_off)
            left = ple_nbytes
            while left:
                b = pf.read(min(CHUNK, left))
                if not b:
                    raise SystemExit("graft_ple: PLE data truncated")
                of.write(b)
                left -= len(b)

    got = os.path.getsize(a.out)
    print(f"wrote {a.out}: {got/1e9:.2f} GB (expected {out_size/1e9:.2f} GB)")
    if got != out_size:
        raise SystemExit("graft_ple: size mismatch")
    print("  tensor count", m.n_tensors + 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
