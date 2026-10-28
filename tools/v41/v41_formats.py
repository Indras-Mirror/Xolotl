"""v41_formats.py - the GGUF block encoders of the V4.1 export (numpy, CPU): integer codes + stored-format scales ->
the exact block bytes the engine (and ggml / llama-tq) dequantizes.

Types (ggml ids):
  Q8_0    8   32 weights:  fp16 d, int8 q[32]                                  34 B   (dense: attention / shared / head)
  Q2_0   42   64 weights:  fp16 d, qs[16] 2-bit (00=-1 01=0 10=+1 11=+2)       18 B   (the default expert tier)
  Q3_K   11  256 weights:  hmask[32] qs[64] scales[12] (6-bit, sc+32) fp16 d  110 B   (sensitive experts)
  PTQ1_0 143 128 weights:  qs[24] base-3 (5 trits/byte), qh[2] (4), fp16 d    28 B   (weakest experts; llama-tq's type)

Every expert encoder takes what v41_gptq produces: the quantized VALUES Q [R, C] (codes x scale, exactly
representable) and the per-group scales S [R, C/G] in their stored format (v41_gptq.search_scales).  The codes are
recovered as Q / S and checked to be integers in the tier's range, so a scale/value mismatch fails loudly instead of
writing a silently different weight.  `encode_rtn` is the round-to-nearest path (all-Q2_0 control, tests).

Layouts are bit-checked against ggml's dequantizers (Q8_0/Q2_0/Q3_K) and Strata's ptq1_0.cpp by
tools/v41/tests/test_v41_formats.py.
"""
import numpy as np

Q8_0, Q3_K, Q2_0, PTQ1_0 = 8, 11, 42, 143
F32, F16, I8, I32 = 0, 1, 24, 26
TIER_TYPE = {"ptq1": PTQ1_0, "q2_0": Q2_0, "q3_k": Q3_K}
TYPE_TIER = {v: k for k, v in TIER_TYPE.items()}
# (weights per block, bytes per block)
BLOCK = {F32: (1, 4), F16: (1, 2), I8: (1, 1), I32: (1, 4), Q8_0: (32, 34), Q2_0: (64, 18), Q3_K: (256, 110),
         PTQ1_0: (128, 28)}
TYPE_NAME = {F32: "F32", F16: "F16", I8: "I8", I32: "I32", Q8_0: "Q8_0", Q2_0: "Q2_0", Q3_K: "Q3_K",
             PTQ1_0: "PTQ1_0"}


def row_bytes(t, n):
    qk, b = BLOCK[t]
    assert n % qk == 0, (t, n)
    return n // qk * b


def nbytes(t, shape):
    """shape in ggml order (ne0 = row length first)."""
    n = 1
    for s in shape[1:]:
        n *= s
    return n * row_bytes(t, shape[0])


def _f16(x):
    return np.asarray(x, np.float32).astype(np.float16)


def _codes(Q, S, G, lo, hi):
    """Q [R, C], S [R, C/G] -> int8 codes [R, C] (exact)."""
    Q = np.asarray(Q, np.float32)
    S = np.asarray(S, np.float32)
    R, C = Q.shape
    Se = np.repeat(S, G, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(np.abs(Se) < 1e-20, 0.0, Q / Se)
    r = np.rint(c)
    if not (np.all(np.abs(c - r) < 1e-3) and r.min() >= lo and r.max() <= hi):
        raise ValueError(f"values are not codes x scales (max frac {np.abs(c - r).max():.3g}, range {r.min()}..{r.max()})")
    return r.astype(np.int8)


# ------------------------------------------------------------------ Q2_0
def pack_q2_0(codes, d):
    """codes [R, C] in -1..2, d [R, C/64] (rounded to fp16 here) -> bytes [R, C/64*18]."""
    R, C = codes.shape
    q = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(R, C // 64, 16, 4)
    qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    out = np.empty((R, C // 64, 18), np.uint8)
    out[..., :2] = _f16(d).reshape(R, C // 64, 1).view(np.uint8)
    out[..., 2:] = qs
    return out.reshape(R, -1)


# ------------------------------------------------------------------ PTQ1_0
def _ptq1_byte(digits):
    """digits [..., k] most significant first (k = 5, or 4 shifted up one trit) -> ceil(v * 256 / 243)."""
    v = np.zeros(digits.shape[:-1], np.uint32)
    for i in range(digits.shape[-1]):
        v = v * 3 + digits[..., i]
    if digits.shape[-1] == 4:
        v = v * 3
    return ((v * 256 + 242) // 243).astype(np.uint8)


def pack_ptq1_0(codes, d):
    """codes [R, C] in -1..1, d [R, C/128] -> bytes [R, C/128*28] (Strata ptq1_0.cpp pack_block)."""
    R, C = codes.shape
    g = (codes.astype(np.int16) + 1).astype(np.uint32).reshape(R, C // 128, 128)
    out = np.empty((R, C // 128, 28), np.uint8)
    # qs[m] (m < 16): values m + 16 n, n = 0..4;  qs[16 + m] (m < 8): 80 + m + 8 n;  qh[h]: 120 + h + 2 n, n = 0..3
    out[..., 0:16] = _ptq1_byte(g[..., 0:80].reshape(R, -1, 5, 16).transpose(0, 1, 3, 2))
    out[..., 16:24] = _ptq1_byte(g[..., 80:120].reshape(R, -1, 5, 8).transpose(0, 1, 3, 2))
    out[..., 24:26] = _ptq1_byte(g[..., 120:128].reshape(R, -1, 4, 2).transpose(0, 1, 3, 2))
    out[..., 26:28] = _f16(d).reshape(R, C // 128, 1).view(np.uint8)
    return out.reshape(R, -1)


# ------------------------------------------------------------------ Q3_K
def q3k_split(S):
    """v41_gptq's Q3_K scales S [R, C/16] (= dsup * sc, dsup fp16 per 256, sc integer in -32..31) -> (dsup [R, C/256]
    fp16-exact, sc [R, C/16] int).  dsup is recovered as fp16(max|S| / 31) (search_scales makes max|sc| = 31) and the
    split is checked exact; an all-zero super-block (S = 1e-30 placeholders) gets dsup 0, sc 0."""
    S = np.asarray(S, np.float32)
    R, n16 = S.shape
    Sb = S.reshape(R, n16 // 16, 16)
    amax = np.abs(Sb).max(-1)
    dsup = _f16(amax / 31.0).astype(np.float32)
    dead = dsup == 0
    dsafe = np.where(dead, 1.0, dsup)
    sc = np.rint(Sb / dsafe[..., None])
    sc = np.where(dead[..., None], 0, sc)
    bad = ~dead[..., None] & ((np.abs(sc * dsafe[..., None] - Sb) > 1e-6 * np.abs(Sb)) | (sc < -32) | (sc > 31))
    if bad.any():
        # rare: amax/31 does not round-trip (fp16-subnormal d of a tiny super-block, or max|sc| < 31).  Search the fp16
        # d values near amax/k for one that splits the block exactly; failing that, re-round the block (one tiny-
        # magnitude block written within its own rounding of the GPTQ value - logged, never a crash)
        for r, b in zip(*np.nonzero(bad.any(-1))):
            row = Sb[r, b]
            best = None
            for k in range(31, 0, -1):
                d0 = np.float16(amax[r, b] / k)
                for d in (d0, np.nextafter(d0, np.float16(np.inf)), np.nextafter(d0, np.float16(0))):
                    d = np.float32(d)
                    if d <= 0:
                        continue
                    q = np.rint(row / d)
                    if (q >= -32).all() and (q <= 31).all() and (np.abs(q * d - row) <= 1e-6 * np.abs(row)).all():
                        best = (d, q)
                        break
                if best:
                    break
            if best is None:
                d = np.float32(_f16(np.float32(amax[r, b] / 31.0)))
                d = d if d > 0 else np.float32(np.finfo(np.float16).smallest_subnormal)
                best = (d, np.clip(np.rint(row / d), -32, 31))
                print(f"q3k_split: row {r} block {b} re-rounded (max rel {float(np.abs(best[1] * d - row).max() / max(amax[r, b], 1e-30)):.2e})",
                      flush=True)
            dsup[r, b], sc[r, b] = best
    return dsup, sc.reshape(R, n16).astype(np.int16)


def pack_q3_k(codes, dsup, sc):
    """codes [R, C] in -4..3, dsup [R, C/256], sc [R, C/16] in -32..31 -> bytes [R, C/256*110]."""
    R, C = codes.shape
    nb = C // 256
    q = (codes.astype(np.int16) + 4).astype(np.uint8).reshape(R, nb, 2, 4, 32)   # [half, j, l]: element 128h+32j+l
    low, high = q & 3, q >> 2
    qs = (low[:, :, :, 0] | (low[:, :, :, 1] << 2) | (low[:, :, :, 2] << 4) | (low[:, :, :, 3] << 6))   # [R,nb,2,32]
    hm = np.zeros((R, nb, 32), np.uint8)
    for h in range(2):
        for j in range(4):
            hm |= high[:, :, h, j, :] << (4 * h + j)
    s = (sc.astype(np.int16) + 32).astype(np.uint8).reshape(R, nb, 16)
    raw = np.empty((R, nb, 12), np.uint8)
    raw[..., 0:4] = (s[..., 0:4] & 15) | ((s[..., 8:12] & 15) << 4)
    raw[..., 4:8] = (s[..., 4:8] & 15) | ((s[..., 12:16] & 15) << 4)
    raw[..., 8:12] = (s[..., 0:4] >> 4) | ((s[..., 4:8] >> 4) << 2) | ((s[..., 8:12] >> 4) << 4) | ((s[..., 12:16] >> 4) << 6)
    out = np.empty((R, nb, 110), np.uint8)
    out[..., 0:32] = hm
    out[..., 32:96] = qs.reshape(R, nb, 64)
    out[..., 96:108] = raw
    out[..., 108:110] = _f16(dsup).reshape(R, nb, 1).view(np.uint8)
    return out.reshape(R, -1)


# ------------------------------------------------------------------ Q8_0 (dense)
def _roundf(x):
    """C roundf (half away from zero), exact in float32."""
    t = np.trunc(x)
    return np.where(np.abs(x - t) >= 0.5, t + np.sign(x), t)


def encode_q8_0(W):
    """W [R, C] float -> bytes [R, C/32*34], ggml quantize_row_q8_0_ref (d = amax/127, q = roundf(x * (1/d)))."""
    W = np.asarray(W, np.float32)
    R, C = W.shape
    g = W.reshape(R, C // 32, 32)
    d = (np.abs(g).max(-1) / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = _roundf((g * idv[..., None]).astype(np.float32)).astype(np.int8)
    out = np.empty((R, C // 32, 34), np.uint8)
    out[..., :2] = _f16(d)[..., None].view(np.uint8)
    out[..., 2:] = q.view(np.uint8)
    return out.reshape(R, -1)


# ------------------------------------------------------------------ expert tiers
def encode_tier(Q, S, tier):
    """GPTQ/GSQ result of one matrix -> GGUF bytes [R, row_bytes] at `tier` (ptq1 / q2_0 / q3_k)."""
    if tier == "q2_0":
        return pack_q2_0(_codes(Q, S, 64, -1, 2), S)
    if tier == "ptq1":
        return pack_ptq1_0(_codes(Q, S, 128, -1, 1), S)
    if tier == "q3_k":
        dsup, sc = q3k_split(S)
        Se = (np.repeat(dsup, 16, axis=1) * sc).astype(np.float32)
        return pack_q3_k(_codes(Q, Se, 16, -4, 3), dsup, sc)
    raise ValueError(tier)


def encode_rtn(W, tier, importance=None):
    """Round-to-nearest at `tier` with the writer's scale search.

    `importance` is one nonnegative activation-importance value per input
    column, matching ggml's imatrix convention. None preserves the original
    uniform-importance control. The same routine is used by screening and the
    final writer, so a measured tier is the tier actually emitted.
    """
    import torch
    import v41_gptq as G
    Wt = torch.from_numpy(np.ascontiguousarray(W, np.float32))
    if importance is None:
        d = torch.ones(Wt.shape[1])
    else:
        d = torch.as_tensor(np.asarray(importance, np.float32).ravel())
        if d.numel() != Wt.shape[1] or not bool(torch.all(torch.isfinite(d))) or bool(torch.any(d < 0)):
            raise ValueError("importance must be finite, nonnegative, and one value per input column")
        d = d.to(Wt.device)
    S = G.search_scales(Wt, d, tier)
    lo, hi, grp, _ = G.TIERS[tier]
    Q = G._rtn(Wt, G.expand_scales(S, tier, Wt.shape[1]), lo, hi)
    return encode_tier(Q.numpy(), S.numpy(), tier)


# ------------------------------------------------------------------ decoders (tests, validation)
def decode(t, raw, n):
    """bytes of one row of n weights -> float32 [n] (the reference form of each layout)."""
    raw = np.frombuffer(bytes(raw), np.uint8)
    if t == Q2_0:
        b = raw.reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], -1).reshape(-1, 64).astype(np.float32) - 1
        return (q * d).reshape(-1)
    if t == PTQ1_0:
        b = raw.reshape(-1, 28)
        d = b[:, 26:28].copy().view(np.float16).astype(np.float32)
        p3 = np.array([1, 3, 9, 27, 81], np.uint16)

        def trits(x, k):   # x [nb, m] -> [nb, k, m]
            return ((((x[:, None, :].astype(np.uint16) * p3[:k, None]) & 0xFF) * 3) >> 8).astype(np.float32)
        v = np.empty((b.shape[0], 128), np.float32)
        v[:, 0:80] = trits(b[:, 0:16], 5).reshape(-1, 80)
        v[:, 80:120] = trits(b[:, 16:24], 5).reshape(-1, 40)
        v[:, 120:128] = trits(b[:, 24:26], 4).reshape(-1, 8)
        return ((v - 1) * d).reshape(-1)
    if t == Q3_K:
        b = raw.reshape(-1, 110)
        hm, qs, raw12 = b[:, 0:32], b[:, 32:96].reshape(-1, 2, 32), b[:, 96:108]
        d = b[:, 108:110].copy().view(np.float16).astype(np.float32)
        s = np.empty((b.shape[0], 16), np.int32)
        a, c, tmp = raw12[:, 0:4].astype(np.int32), raw12[:, 4:8].astype(np.int32), raw12[:, 8:12].astype(np.int32)
        s[:, 0:4] = (a & 15) | ((tmp & 3) << 4)
        s[:, 4:8] = (c & 15) | (((tmp >> 2) & 3) << 4)
        s[:, 8:12] = (a >> 4) | (((tmp >> 4) & 3) << 4)
        s[:, 12:16] = (c >> 4) | (((tmp >> 6) & 3) << 4)
        y = np.empty((b.shape[0], 256), np.float32)
        for h in range(2):
            for j in range(4):
                low = (qs[:, h, :] >> (2 * j)) & 3
                hb = (hm >> (4 * h + j)) & 1
                e0 = 128 * h + 32 * j
                y[:, e0:e0 + 32] = low.astype(np.float32) - np.where(hb == 1, 0, 4)
        return (y * d * (np.repeat(s, 16, axis=1) - 32)).reshape(-1)
    if t == Q8_0:
        b = raw.reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        return (b[:, 2:].copy().view(np.int8).astype(np.float32) * d).reshape(-1)
    if t == F16:
        return raw.view(np.float16).astype(np.float32)
    if t == F32:
        return raw.view(np.float32).copy()
    raise ValueError(t)
