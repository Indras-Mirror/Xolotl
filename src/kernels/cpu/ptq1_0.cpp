// src/kernels/cpu/ptq1_0.cpp - PTQ1_0 ternary rows on the CPU.  Layout and contract: the header.
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"   // cpu_avx2_ok

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_PTQ1_X86 1
#endif

namespace strata::kernels::cpu {
namespace {

// fp16 <-> fp32 without a ggml dependency (IEEE half, round-to-nearest-even on the way in)
float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int k = -1; uint32_t mm = m; do { mm <<= 1; ++k; } while (!(mm & 0x400)); bits = s | ((uint32_t) (127 - 15 - k) << 23) | ((mm & 0x3ff) << 13); }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000; int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t) (s | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000; const int sh = 14 - e; uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13); const uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t) (s | r);
}

constexpr uint8_t kPow3[6] = {1, 3, 9, 27, 81, 243};
inline int trit(uint8_t b, int n) { return (int) ((((uint16_t) (uint8_t) (b * kPow3[n])) * 3) >> 8); }   // 0,1,2

// position of value v (0..127) in the block: (byte, trit)
struct Pos { uint8_t byte, n; bool hi; };
Pos where(int v) {
    if (v < 80) return {(uint8_t) (v % 16), (uint8_t) (v / 16), false};
    if (v < 120) return {(uint8_t) (16 + (v - 80) % 8), (uint8_t) ((v - 80) / 8), false};
    return {(uint8_t) ((v - 120) % 2), (uint8_t) ((v - 120) / 2), true};
}

// digits[128] in {0,1,2} for value order 0..127
void block_digits(const uint8_t* blk, uint8_t* dg) {
    for (int n = 0; n < 5; ++n)
        for (int m = 0; m < 16; ++m) dg[m + 16 * n] = (uint8_t) trit(blk[m], n);
    for (int n = 0; n < 5; ++n)
        for (int m = 0; m < 8; ++m) dg[80 + m + 8 * n] = (uint8_t) trit(blk[16 + m], n);
    for (int n = 0; n < 4; ++n)
        for (int h = 0; h < 2; ++h) dg[120 + h + 2 * n] = (uint8_t) trit(blk[24 + h], n);
}

void pack_block(const int8_t* t, float d, uint8_t* blk) {
    // 5-trit bytes: accumulate most significant first, then ceil(q * 256 / 243)
    for (int m = 0; m < 16; ++m) {
        unsigned q = 0;
        for (int n = 0; n < 5; ++n) q = q * 3 + (unsigned) (t[m + 16 * n] + 1);
        blk[m] = (uint8_t) ((q * 256 + 242) / 243);
    }
    for (int m = 0; m < 8; ++m) {
        unsigned q = 0;
        for (int n = 0; n < 5; ++n) q = q * 3 + (unsigned) (t[80 + m + 8 * n] + 1);
        blk[16 + m] = (uint8_t) ((q * 256 + 242) / 243);
    }
    for (int h = 0; h < 2; ++h) {
        unsigned q = 0;
        for (int n = 0; n < 4; ++n) q = q * 3 + (unsigned) (t[120 + h + 2 * n] + 1);
        q *= 3;   // the first value to the most significant of 5 trits
        blk[24 + h] = (uint8_t) ((q * 256 + 242) / 243);
    }
    const uint16_t hd = f2h(d);
    std::memcpy(blk + 26, &hd, 2);
}

struct Q8 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(Q8) == 34, "block_q8_0 layout");

}  // namespace

void ptq1_0_pack(const int8_t* t, const float* d, uint8_t* dst, int64_t n) {
    for (int64_t b = 0; b < n / kPtq1Block; ++b) pack_block(t + b * kPtq1Block, d[b], dst + b * kPtq1Bytes);
}

void ptq1_0_quantize_ref(const float* x, uint8_t* dst, int64_t n) {
    int8_t t[kPtq1Block];
    for (int64_t b = 0; b < n / kPtq1Block; ++b) {
        const float* xb = x + b * kPtq1Block;
        float amax = 0.f;
        for (int j = 0; j < kPtq1Block; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float id = amax ? 1.f / amax : 0.f;
        for (int j = 0; j < kPtq1Block; ++j) t[j] = (int8_t) std::lround(xb[j] * id);
        pack_block(t, amax, dst + b * kPtq1Bytes);
    }
}

void ptq1_0_unpack(const uint8_t* src, int8_t* t, float* d, int64_t n) {
    uint8_t dg[kPtq1Block];
    for (int64_t b = 0; b < n / kPtq1Block; ++b) {
        const uint8_t* blk = src + b * kPtq1Bytes;
        block_digits(blk, dg);
        for (int j = 0; j < kPtq1Block; ++j) t[b * kPtq1Block + j] = (int8_t) (dg[j] - 1);
        uint16_t hd; std::memcpy(&hd, blk + 26, 2);
        d[b] = h2f(hd);
    }
}

void ptq1_0_dequant(const uint8_t* src, float* y, int64_t n) {
    uint8_t dg[kPtq1Block];
    for (int64_t b = 0; b < n / kPtq1Block; ++b) {
        const uint8_t* blk = src + b * kPtq1Bytes;
        block_digits(blk, dg);
        uint16_t hd; std::memcpy(&hd, blk + 26, 2);
        const float d = h2f(hd);
        for (int j = 0; j < kPtq1Block; ++j) y[b * kPtq1Block + j] = (float) ((int) dg[j] - 1) * d;
    }
}

void ptq1_0_vec_dot_q8_0_scalar(int n, float* s, const void* vx, const void* vy) {
    const uint8_t* x = (const uint8_t*) vx;
    const Q8* y = (const Q8*) vy;
    uint8_t dg[kPtq1Block];
    float acc = 0.f;
    for (int b = 0; b < n / kPtq1Block; ++b) {
        const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
        block_digits(blk, dg);
        uint16_t hd; std::memcpy(&hd, blk + 26, 2);
        const float dw = h2f(hd);
        float sb = 0.f;
        for (int k = 0; k < 4; ++k) {
            const Q8& a = y[b * 4 + k];
            int sum = 0;
            for (int i = 0; i < 32; ++i) sum += ((int) dg[k * 32 + i] - 1) * a.qs[i];
            sb += h2f(a.d) * (float) sum;
        }
        acc += dw * sb;
    }
    *s = acc;
}

#if STRATA_PTQ1_X86
// GCC/Clang compile these functions for AVX2 (+FMA/F16C) on their own; MSVC needs no attribute to use the
// intrinsics (the same pattern as iq_avx2.cpp's STRATA_AVXVNNI_FN).
#if defined(__GNUC__) || defined(__clang__)
#define STRATA_PTQ_AVX2 __attribute__((target("avx2")))
#define STRATA_PTQ_AVX2_FMA __attribute__((target("avx2,fma,f16c")))
#else
#define STRATA_PTQ_AVX2
#define STRATA_PTQ_AVX2_FMA
#endif
#if defined(__GNUC__) || defined(__clang__)
STRATA_PTQ_AVX2_FMA inline float ptq_h2f(uint16_t h) { return _cvtsh_ss(h); }
#else   // MSVC has no scalar F16C intrinsic: the same conversion through the vector one
inline float ptq_h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
#endif
namespace {
// 16 bytes -> for trit n: 16 digits (uint8 0..2) of trit n of each byte
STRATA_PTQ_AVX2 inline __m128i trits16(__m128i b, int n) {
    const __m256i w = _mm256_cvtepu8_epi16(b);
    const __m256i p = _mm256_mullo_epi16(w, _mm256_set1_epi16(kPow3[n]));
    const __m256i lo = _mm256_and_si256(p, _mm256_set1_epi16(0xff));
    const __m256i dg = _mm256_srli_epi16(_mm256_mullo_epi16(lo, _mm256_set1_epi16(3)), 8);
    return _mm_packus_epi16(_mm256_castsi256_si128(dg), _mm256_extracti128_si256(dg, 1));
}
// sum over 32 lanes of digits (u8) * q (i8) and of q, as int32 (both exact)
STRATA_PTQ_AVX2 inline void dot32(__m256i dg, __m256i q, int& dq, int& sq) {
    const __m256i one16 = _mm256_set1_epi16(1);
    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(dg, q), one16);                     // digits <= 2: no i16 saturation
    const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), q), one16);
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    __m128i c = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    a = _mm_hadd_epi32(a, a); a = _mm_hadd_epi32(a, a);
    c = _mm_hadd_epi32(c, c); c = _mm_hadd_epi32(c, c);
    dq = _mm_cvtsi128_si32(a); sq = _mm_cvtsi128_si32(c);
}
STRATA_PTQ_AVX2 void dot_avx2(int n, float* s, const uint8_t* x, const Q8* y) {
    alignas(32) uint8_t dg[kPtq1Block];
    float acc = 0.f;
    for (int b = 0; b < n / kPtq1Block; ++b) {
        const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
        const __m128i b16 = _mm_loadu_si128((const __m128i*) blk);
        for (int t = 0; t < 5; ++t) _mm_store_si128((__m128i*) (dg + 16 * t), trits16(b16, t));
        // the 8 + 2 tail bytes: scalar (40 + 8 digits)
        for (int t = 0; t < 5; ++t)
            for (int m = 0; m < 8; ++m) dg[80 + m + 8 * t] = (uint8_t) trit(blk[16 + m], t);
        for (int t = 0; t < 4; ++t)
            for (int h = 0; h < 2; ++h) dg[120 + h + 2 * t] = (uint8_t) trit(blk[24 + h], t);
        uint16_t hd; std::memcpy(&hd, blk + 26, 2);
        float sb = 0.f;
        for (int k = 0; k < 4; ++k) {
            const Q8& a = y[b * 4 + k];
            int dq, sq;
            dot32(_mm256_load_si256((const __m256i*) (dg + 32 * k)), _mm256_loadu_si256((const __m256i*) a.qs), dq, sq);
            sb += h2f(a.d) * (float) (dq - sq);
        }
        acc += h2f(hd) * sb;
    }
    *s = acc;
}
// ---- ptq1_0_rows_q8_0's AVX2 kernel: a block's 128 digits decoded once (16-bit lanes: digit n of a byte b is
// ((uint8)(b * 3^n) * 3) >> 8), as -1 / 0 / +1 bytes; per token and 32-weight sub-block sum(sign(q, t - 1)) in int32
// - exactly the scalar dot's integer - then * d_a in fp32, * d_w per block, one horizontal sum a row.
STRATA_PTQ_AVX2_FMA inline __m256i rtrit16(__m256i w, __m256i pw) {
    const __m256i p = _mm256_and_si256(_mm256_mullo_epi16(w, pw), _mm256_set1_epi16(0xff));
    return _mm256_srli_epi16(_mm256_add_epi16(p, _mm256_add_epi16(p, p)), 8);
}
STRATA_PTQ_AVX2_FMA inline __m256i rpack(__m256i a, __m256i b) {
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
}
// one block -> 4 x 32 signed trits in element order (qs[0..15]: 16 n + m; qs[16..23]: 80 + m + 8 n; qh: 120 + h + 2 n)
STRATA_PTQ_AVX2_FMA inline void runpack(const uint8_t* blk, __m256i dg[4]) {
    const __m256i w = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) blk));
    const __m256i t0 = rtrit16(w, _mm256_set1_epi16(1)), t1 = rtrit16(w, _mm256_set1_epi16(3));
    const __m256i t2 = rtrit16(w, _mm256_set1_epi16(9)), t3 = rtrit16(w, _mm256_set1_epi16(27));
    const __m256i t4 = rtrit16(w, _mm256_set1_epi16(81));
    const __m128i v8 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (blk + 16)));
    const __m256i w2 = _mm256_set_m128i(v8, v8);
    const __m256i u01 = rtrit16(w2, _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i u23 = rtrit16(w2, _mm256_setr_epi16(9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27));
    const uint32_t h0 = blk[24], h1 = blk[25];
    const __m256i w3 = _mm256_set_m128i(_mm_set1_epi32((int) (h0 | h1 << 16)), v8);
    const __m256i u4 = rtrit16(w3, _mm256_setr_epi16(81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27));
    const __m256i one8 = _mm256_set1_epi8(1);
    dg[0] = _mm256_sub_epi8(rpack(t0, t1), one8);
    dg[1] = _mm256_sub_epi8(rpack(t2, t3), one8);
    dg[2] = _mm256_sub_epi8(rpack(t4, u01), one8);
    dg[3] = _mm256_sub_epi8(rpack(u23, u4), one8);
}
STRATA_PTQ_AVX2_FMA inline float rhsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}
template <int NT>
STRATA_PTQ_AVX2_FMA void rows_avx2(const uint8_t* w, size_t rb, int n, const Q8* const* y, float* const* out, int r0, int r1) {
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * rb;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < n / kPtq1Block; ++b) {
            const uint8_t* blk = x + (size_t) b * kPtq1Bytes;
            __m256i dg[4];
            runpack(blk, dg);
            uint16_t hd;
            std::memcpy(&hd, blk + 26, 2);
            const __m256 vdw = _mm256_set1_ps(ptq_h2f(hd));   // F16C: exact, as h2f
            for (int t = 0; t < NT; ++t) {
                const Q8* a = y[t] + b * 4;
                __m256 bs = _mm256_setzero_ps();
                for (int k = 0; k < 4; ++k) {
                    const __m256i sq = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) a[k].qs), dg[k]);
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, sq), ones16);   // |q| <= 127
                    bs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(ptq_h2f(a[k].d)), bs);
                }
                acc[t] = _mm256_fmadd_ps(bs, vdw, acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = rhsum(acc[t]);
    }
}
}  // namespace
#endif

void ptq1_0_rows_q8_0(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
#if STRATA_PTQ1_X86
    static const bool avx2 = cpu_avx2_ok();   // AVX, AVX2, FMA and F16C (expert_layout.hpp)
    if (avx2) {
        // tokens in groups of <= 4 (the accumulators stay in registers)
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int g = nt - t0 < 4 ? nt - t0 : 4;
            const Q8* y[4];
            for (int t = 0; t < g; ++t) y[t] = (const Q8*) act[t0 + t];
            switch (g) {
                case 1: rows_avx2<1>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 2: rows_avx2<2>(w, row_bytes, n, y, out + t0, r0, r1); break;
                case 3: rows_avx2<3>(w, row_bytes, n, y, out + t0, r0, r1); break;
                default: rows_avx2<4>(w, row_bytes, n, y, out + t0, r0, r1); break;
            }
        }
        return;
    }
#endif
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) ptq1_0_vec_dot_q8_0(n, &out[t][r], 0, w + (size_t) r * row_bytes, 0, act[t], 0, 1);
}

void ptq1_0_vec_dot_q8_0(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
#if STRATA_PTQ1_X86
    static const bool avx2 = cpu_avx2_ok();
    if (avx2) { dot_avx2(n, s, (const uint8_t*) vx, (const Q8*) vy); return; }
#endif
    ptq1_0_vec_dot_q8_0_scalar(n, s, vx, vy);
}

}  // namespace strata::kernels::cpu
