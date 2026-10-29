#!/usr/bin/env python3
"""IQ4_NL codec: parse a quantised block row into (scales, codes) and write it back.

WHY THIS EXISTS. Lattice refinement (the GSQ/AdaRound analogue for ggml's codebook formats)
cannot be done in weight space: the IQ types derive `d` per block from that block's maximum, so
`dequantize(x)` is NOT a fixed point of `quantize()` (measured: repeated snapping drifts
0.18419 -> 0.21133 on iq3_xxs) and a mixture of "old" and "new" codebook entries is not
representable at all (~0% of entries). Perturbing W and re-quantising therefore re-scales every
block and is worse than doing nothing (measured: -41% to -176% vs plain RTN).

So refinement has to work on the codes directly, holding `d` fixed. That needs exactly two things
per type: read the codes out of the quantised bytes, and write modified codes back. This module is
that for IQ4_NL, which carries 47.9 of the ~70 GiB in the V2 (arm B) plan -- `ffn_down_exps`
(21.09 GiB) plus the PLE table (26.82 GiB).

Layout (ggml-common.h, ggml-quants.c:dequantize_row_iq4_nl):

    block_iq4_nl { ggml_half d; uint8_t qs[16]; }          // 32 values in 18 bytes
    y[j]      = d * kvalues_iq4nl[qs[j] & 0xf]      j in 0..15
    y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4]       j in 0..15

    python3 tools/qwen38/iq4nl_codec.py --selftest
"""
import argparse
import sys
import pathlib

import numpy as np

QK4_NL = 32
BLOCK_BYTES = 18                      # 2 (fp16 d) + 16 (nibbles)

# ggml-common.h:GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
KVALS = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                  1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int16)


def row_bytes(cols):
    """Bytes for one row of `cols` values, or 0 when the block does not divide it."""
    if cols % QK4_NL:
        return 0
    return cols // QK4_NL * BLOCK_BYTES


def parse(raw, rows, cols):
    """Raw quantised bytes -> (d [rows, nblk] fp32, codes [rows, cols] uint8 0..15).

    codes are in the block's own order, i.e. codes[:, 0:16] came from the low nibbles of the
    block's first 16 bytes and codes[:, 16:32] from the high nibbles.
    """
    nblk = cols // QK4_NL
    rb = row_bytes(cols)
    if rb == 0:
        raise ValueError(f"{cols} is not a multiple of {QK4_NL}")
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.size != rows * rb:
        raise ValueError(f"expected {rows * rb} bytes, got {raw.size}")
    b = raw.reshape(rows, nblk, BLOCK_BYTES)

    d = b[:, :, :2].copy().reshape(rows, nblk, 2).view(np.float16).reshape(rows, nblk)
    d = d.astype(np.float32)

    qs = b[:, :, 2:]                                   # [rows, nblk, 16]
    lo = qs & 0x0F
    hi = (qs >> 4) & 0x0F
    codes = np.concatenate([lo, hi], axis=2)           # [rows, nblk, 32]
    return d, codes.reshape(rows, nblk * QK4_NL).astype(np.uint8)


def encode(d, codes, rows, cols):
    """(d, codes) -> raw quantised bytes. Inverse of parse()."""
    nblk = cols // QK4_NL
    if codes.shape != (rows, nblk * QK4_NL):
        raise ValueError(f"codes shape {codes.shape} != {(rows, nblk * QK4_NL)}")
    c = codes.reshape(rows, nblk, QK4_NL)
    if c.size and (c.max() > 15 or c.min() < 0):
        raise ValueError("codes must be 0..15")
    lo = c[:, :, :16].astype(np.uint8)
    hi = c[:, :, 16:].astype(np.uint8)
    qs = lo | (hi << 4)                                # [rows, nblk, 16]
    dh = np.ascontiguousarray(d.astype(np.float16).reshape(rows, nblk, 1)).view(np.uint8).reshape(rows, nblk, 2)
    b = np.concatenate([dh, qs], axis=2)
    return np.ascontiguousarray(b.reshape(rows, -1))


def dequantize(d, codes):
    """Values from (d [rows, nblk], codes [rows, nblk*32])."""
    rows, total = codes.shape
    nblk = total // QK4_NL
    vals = KVALS[codes.reshape(rows, nblk, QK4_NL)]    # [rows, nblk, 32] int16
    return (vals.astype(np.float32) * d[:, :, None]).reshape(rows, total)


def selftest(shim_path="/tmp/libgqshim.so"):
    """Does our parse/encode/dequantize agree with ggml byte for byte?"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import screen as S

    SH = S.GGML(shim_path)
    rng = np.random.RandomState(0)
    ok = True
    for rows, cols in ((8, 2560), (8, 640), (3, 32), (2, 320)):
        if row_bytes(cols) == 0:
            continue
        # real-ish weights: gaussian, and a heavy-tailed one to exercise the scale spread
        for label, W in (("gauss", rng.randn(rows, cols).astype(np.float32) * 0.05),
                         ("t-dist", (rng.standard_t(3, (rows, cols)) * 0.05).astype(np.float32))):
            raw = _quantize_bytes(SH, W, "iq4_nl")
            d, codes = parse(raw, rows, cols)

            # 1. our dequantize must equal ggml's
            mine = dequantize(d, codes)
            theirs = SH.quantize_dequantize(W, "iq4_nl", None)[0]
            if not np.array_equal(mine, theirs):
                print(f"  FAIL dequant rows={rows} cols={cols} {label}: "
                      f"maxdiff {float(np.abs(mine - theirs).max()):.3e}")
                ok = False
                continue

            # 2. encode(parse(x)) must reproduce the bytes exactly. encode() returns
            # (rows, row_bytes) while _quantize_bytes returns a flat stream -- compare flattened,
            # not shape-to-shape, or a correct codec reports as a byte mismatch.
            back = encode(d, codes, rows, cols)
            if not np.array_equal(back.reshape(-1), np.asarray(raw).reshape(-1)):
                print(f"  FAIL encode rows={rows} cols={cols} {label}: bytes differ")
                ok = False
                continue

            # 3. a CODE EDIT must survive: change one code, encode, re-parse, and confirm the
            #    value moved to the neighbouring codebook entry. This is the operation the
            #    refinement stage needs, so it is what must be verified.
            #    (Deliberately NOT testing "re-quantising the dequantised tensor returns the same
            #    thing": that is false for IQ types by design -- the block scale is derived from
            #    the block max, so a lattice tensor is not a fixed point. See QFN-V2-PIPELINE §11e.)
            c2 = codes.copy()
            r0, b0 = 0, 0
            old = c2[r0, b0]
            new = (old + 1) % 16
            c2[r0, b0] = new
            v = dequantize(d, c2)
            want = float(d[r0, 0]) * float(KVALS[new])
            got = float(v[r0, b0])
            if got != want:
                print(f"  FAIL code-edit rows={rows} cols={cols} {label}: {got} != {want}")
                ok = False
                continue
            # and the edited stream must still decode through ggml to that same value
            ed = encode(d, c2, rows, cols)
            via_ggml = _dequant(SH, ed, cols, rows)[r0, b0]
            if float(via_ggml) != want:
                print(f"  FAIL code-edit/ggml rows={rows} cols={cols} {label}: "
                      f"{via_ggml} != {want}")
                ok = False
                continue
            print(f"  ok  rows={rows} cols={cols:4d} {label}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _quantize_bytes(SH, W, type_name):
    """Quantise and return the RAW bytes.

    screen.py's quantize_dequantize already allocates this buffer and then discards it; code-level
    refinement needs it. Kept here so screen.py does not have to change until the stage lands.
    """
    import ctypes
    rows, cols = W.shape
    rb = SH.row_bytes(type_name, cols)
    if rb <= 0:
        raise SystemExit(f"{type_name}: bad row bytes for cols={cols}")
    buf = ctypes.create_string_buffer(rb * rows)
    t = SH.id(type_name)
    im = np.ones(cols, dtype=np.float32)
    n = SH.L.gq_quantize(t, W.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         buf, rows, cols, im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if n != rb * rows:
        raise SystemExit(f"quantize wrote {n} bytes, expected {rb * rows}")
    return np.frombuffer(buf.raw, dtype=np.uint8).copy()


def _dequant(SH, raw, cols, rows):
    import ctypes
    # exact-size buffer: create_string_buffer(init, size) insists on size >= len(init)+1 for the
    # NUL terminator, which makes a byte-for-byte buffer of length N impossible via that path.
    flat = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8).reshape(-1))
    buf = (ctypes.c_uint8 * flat.size).from_buffer_copy(flat.tobytes())
    out = np.empty(rows * cols, dtype=np.float32)
    SH.L.gq_dequantize(SH.id("iq4_nl"), buf,
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), cols * rows)
    return out.reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shim", default="/tmp/libgqshim.so")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest(a.shim))
    ap.print_help()


if __name__ == "__main__":
    main()
