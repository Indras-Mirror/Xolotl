// src/kernels/cpu/iq3_one.cpp - IQ3_XXS / IQ3_S expert rows for the calls ggml-cpu's vec_dot serves today (a lone
// decode token's gate / up, every IQ3_S down), ported (xolotl, 2026-10-27) from Project Maya's src/ds4/ds4_iq_cpu.cpp
// (v1.1.28, 6afa525; MIT - Copyright (c) 2026 Peasant Smith and the Project Maya contributors, Copyright (c) 2026
// Niko1221 and the Strata contributors).
//
// Every grid byte of these two formats is one of 8 levels, so a grid entry is four 3-bit level codes (a u16) and the
// whole grid fits in zmm registers - IQ3_XXS's 256 entries in 8, IQ3_S's 512 in 16.  A block's 64 indices are looked
// up 32 at a time with two-register word permutes and blends (ggml-cpu's AVX2 dot builds each 32 values from eight
// scalar table loads and inserts); each 64 values' 16 codes are spread to their bytes by a word permute, a multishift,
// an and and a byte shuffle; IQ3_XXS's signs come from pdep and a parity fold, IQ3_S's are a mask register's bits.  The
// int32 lanes are ggml's exact sums, folded to its 8 lanes, then ggml's float steps (one fma per block, hsum_float_8,
// IQ3_XXS times 0.25): the same numbers to the bit, which iq3_one_selftest checks against ggml-cpu's own dot here.
//
// Compiled /arch:AVX512 (any code in this file may use AVX-512): reached only behind iq3_one_ok() in native_expert.cpp,
// which checks AVX-512 F / BW / VL / VNNI / VBMI and BMI2 first.
#include "strata/kernels/cpu/iq3_one.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) {   // F16 -> F32, exactly
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;   // m 2^-24 (exact)
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline float hsum_ggml(__m256 x) {   // ggml's hsum_float_8: the same additions in the same order
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

inline uint32_t signs4(uint32_t w) {   // ksigns_iq2xs for the four 7-bit fields of w: i | parity(i) << 7
    const uint64_t s = _pdep_u64(w & 0x0FFFFFFFu, 0x7F7F7F7Full);
    uint64_t p = s ^ ((s >> 4) & 0x0F0F0F0Full);
    p ^= (p >> 2) & 0x03030303ull;
    p ^= (p >> 1) & 0x01010101ull;
    return (uint32_t) (s | ((p & 0x01010101ull) << 7));
}

alignas(64) uint16_t g_code3x[256], g_code3s[512];
alignas(64) uint16_t g_scale_idx[32];   // int16 lane j <- word j / 8 of a broadcast qword (scales4)
alignas(64) uint16_t g_pidx[2][32];     // a 64-value group h's 32 codes: output qword m <- entries 16 h + 2 m, + 1
bool g_codes_ok = false;
std::once_flag g_codes_once;

// each grid entry as four 3-bit level codes; false if a byte is not one of the 8 levels (a grid this file does not know)
bool build_codes3() {
    static const int lx[8] = {4, 12, 20, 28, 36, 44, 52, 62}, ls[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    auto code = [](uint32_t e, const int* lv) -> int {
        int c = 0;
        for (int k = 0; k < 4; ++k) {
            const int b = (int) ((e >> (8 * k)) & 255u);
            int l = -1;
            for (int j = 0; j < 8; ++j)
                if (lv[j] == b) l = j;
            if (l < 0) return -1;
            c |= l << (3 * k);
        }
        return c;
    };
    for (int i = 0; i < 256; ++i) {
        const int c = code(iq3xxs_grid[i], lx);
        if (c < 0) return false;
        g_code3x[i] = (uint16_t) c;
    }
    for (int i = 0; i < 512; ++i) {
        const int c = code(iq3s_grid[i], ls);
        if (c < 0) return false;
        g_code3s[i] = (uint16_t) c;
    }
    for (int j = 0; j < 32; ++j) g_scale_idx[j] = (uint16_t) (j / 8);
    for (int h = 0; h < 2; ++h)
        for (int m = 0; m < 8; ++m) {
            g_pidx[h][4 * m] = g_pidx[h][4 * m + 1] = (uint16_t) (16 * h + 2 * m);
            g_pidx[h][4 * m + 2] = g_pidx[h][4 * m + 3] = (uint16_t) (16 * h + 2 * m + 1);
        }
    return true;
}

inline __m256i fold(__m512i a) {   // a block's 16 int32 lanes -> ggml's 8 (lane k + lane k + 8), exact
    return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
}

inline __m512i scales4(int s0, int s1, int s2, int s3) {   // int16 lanes 0-7 s0, 8-15 s1, 16-23 s2, 24-31 s3
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(_mm512_load_si512((const void*) g_scale_idx), _mm512_set1_epi64((long long) packed));
}

// the codes of 32 word indices: 64 entries a register pair, NP pairs, blended by the index's bits 6, 7 (and 8)
template <int NP>
inline __m512i lookup(const __m512i* T, __m512i idx) {
    __m512i r[NP];
    for (int j = 0; j < NP; ++j) r[j] = _mm512_permutex2var_epi16(T[2 * j], idx, T[2 * j + 1]);
    const __mmask32 m6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 m7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    __m512i a = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[0], r[1]), _mm512_mask_blend_epi16(m6, r[2], r[3]));
    if constexpr (NP == 8) {
        const __mmask32 m8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
        const __m512i b = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[4], r[5]), _mm512_mask_blend_epi16(m6, r[6], r[7]));
        a = _mm512_mask_blend_epi16(m8, a, b);
    }
    return a;
}

template <int TY, int NT>
void iq3_rows_512(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    constexpr int NTAB = TY == 18 ? 8 : 16;
    constexpr size_t BB = TY == 18 ? sizeof(block_iq3_xxs) : sizeof(block_iq3_s);
    __m512i T[NTAB];
    for (int k = 0; k < NTAB; ++k) T[k] = _mm512_load_si512((const void*) ((TY == 18 ? g_code3x : g_code3s) + 32 * k));
    // code -> grid byte (IQ3_XXS 4 .. 62, IQ3_S 1 .. 15) in every 128-bit lane; byte b of an output qword <- 3 bits at
    // [0, 3, 6, 9] (its first entry's code, word 0) or [32, 35, 38, 41] (the second's, word 2)
    const __m512i lut = TY == 18 ? _mm512_set4_epi32(0, 0, 0x3E342C24, 0x1C140C04) : _mm512_set4_epi32(0, 0, 0x0F0D0B09, 0x07050301);
    const __m512i ctl = _mm512_set1_epi64(0x2926232009060300LL);
    const __m512i seven = _mm512_set1_epi8(7), zero = _mm512_setzero_si512();
    const __m512i pidx[2] = {_mm512_load_si512((const void*) g_pidx[0]), _mm512_load_si512((const void*) g_pidx[1])};
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            __m512i codes[2];
            if constexpr (TY == 18) {
                const __m512i q = _mm512_loadu_si512((const void*) (blk + 2));
                codes[0] = lookup<4>(T, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(q)));
                codes[1] = lookup<4>(T, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(q, 1)));
            } else {
                const uint8_t* qh = blk + 2 + QK_K / 4;
                const __m512i hi = _mm512_set1_epi16(256);
                const __m512i ia = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2)));
                const __m512i ib = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 34)));
                // (index k of chunk c: qs[8 c + k] | bit k of qh[c] << 8 - the 32 bits of qh[0..3] are chunks 0..3's)
                codes[0] = lookup<8>(T, _mm512_or_si512(ia, _mm512_maskz_mov_epi16((__mmask32) u32(qh), hi)));
                codes[1] = lookup<8>(T, _mm512_or_si512(ib, _mm512_maskz_mov_epi16((__mmask32) u32(qh + 4), hi)));
            }
            __m512i acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
            for (int p = 0; p < 4; ++p) {
                const __m512i rep = _mm512_permutexvar_epi16(pidx[p & 1], codes[p >> 1]);
                const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, rep), seven));
                uint64_t sm;
                int sa, sb;
                if constexpr (TY == 18) {
                    const uint8_t* sw = blk + 2 + QK_K / 4 + 8 * p;
                    const uint32_t wa = u32(sw), wb = u32(sw + 4);
                    sm = (uint64_t) signs4(wa) | ((uint64_t) signs4(wb) << 32);
                    sa = 2 * (int) (wa >> 28) + 1;
                    sb = 2 * (int) (wb >> 28) + 1;
                } else {
                    std::memcpy(&sm, blk + 2 + QK_K / 4 + QK_K / 32 + 8 * p, 8);
                    const int sc = blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + p];
                    sa = 2 * (sc & 0xf) + 1;
                    sb = 2 * (sc >> 4) + 1;
                }
                const __mmask64 m = _cvtu64_mask64(sm);
                const __m512i S = scales4(sa, sa, sb, sb);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * p));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                    acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), S));
                }
            }
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = (TY == 18 ? 0.25f : 1.0f) * hsum_ggml(accf[t]);
    }
}

template <int TY>
void rows_n(const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {   // 4 tokens a pass: their accumulators stay in registers
        const block_q8_K* y[4];
        float* o[4];
        const int c = nt - t0 < 4 ? nt - t0 : 4;
        for (int t = 0; t < c; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            o[t] = out[t0 + t];
        }
        switch (c) {
            case 1: iq3_rows_512<TY, 1>(w, rb, nb, y, o, r0, r1); break;
            case 2: iq3_rows_512<TY, 2>(w, rb, nb, y, o, r0, r1); break;
            case 3: iq3_rows_512<TY, 3>(w, rb, nb, y, o, r0, r1); break;
            default: iq3_rows_512<TY, 4>(w, rb, nb, y, o, r0, r1); break;
        }
    }
}

bool codes_ready() {
    std::call_once(g_codes_once, [] { g_codes_ok = build_codes3(); });
    return g_codes_ok;
}

void random_rows(ggml_type t, int n, int rows, std::vector<uint8_t>& w, uint64_t& st) {   // random blocks, small scales
    auto next = [&st] {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return st;
    };
    const size_t rb = ggml_row_size(t, n), bb = ggml_type_size(t);
    w.resize(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) next();
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        std::memcpy(&w[o], &d, sizeof d);
    }
}

}  // namespace

void iq3_one_rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out,
                  int r0, int r1) {
    if (type == GGML_TYPE_IQ3_XXS) rows_n<18>(w, rb, n, act, nt, out, r0, r1);
    else rows_n<21>(w, rb, n, act, nt, out, r0, r1);
}

bool iq3_one_selftest(int type, int n_embd, double& us_ggml, double& us_ours) {
    us_ggml = us_ours = 0;
    if ((type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ3_S) || !codes_ready()) return false;
    ggml_cpu_init();
    const ggml_type t = (ggml_type) type;
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto dot = ggml_get_type_traits_cpu(t)->vec_dot;
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto next = [&st] {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return st;
    };
    // bit for bit against ggml-cpu's dot: 13 rows of 8 blocks, Q8_K activations from ggml's quantizer, 1 / 2 / 3 / 5 tokens
    {
        const int n = 2048, rows = 13;
        const size_t rb = ggml_row_size(t, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
        std::vector<uint8_t> w;
        random_rows(t, n, rows, w, st);
        for (int nt : {1, 2, 3, 5}) {
            std::vector<uint8_t> a(ab * (size_t) nt);
            std::vector<float> x((size_t) n), got((size_t) nt * rows), ref((size_t) nt * rows);
            std::vector<const void*> ap((size_t) nt);
            std::vector<float*> op((size_t) nt);
            for (int k = 0; k < nt; ++k) {
                for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
                q8->from_float(x.data(), a.data() + ab * (size_t) k, n);
                ap[(size_t) k] = a.data() + ab * (size_t) k;
                op[(size_t) k] = got.data() + (size_t) k * rows;
            }
            iq3_one_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows);
            for (int k = 0; k < nt; ++k)
                for (int r = 0; r < rows; ++r)
                    dot(n, &ref[(size_t) k * rows + r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) k], 0, 1);
            if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) != 0) return false;
        }
    }
    // one token's time on 32 rows of n_embd columns (an expert's gate or up slice), the best of 6 rounds after a warm-up
    const int n = std::max(QK_K, n_embd / QK_K * QK_K), rows = 32;
    const size_t rb = ggml_row_size(t, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> w, a(ab);
    random_rows(t, n, rows, w, st);
    std::vector<float> x((size_t) n), y((size_t) rows);
    for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
    q8->from_float(x.data(), a.data(), n);
    const void* ap = a.data();
    float* yp = y.data();
    double bg = 1e30, bo = 1e30;
    for (int round = 0; round < 7; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 8; ++rep)
            for (int r = 0; r < rows; ++r) dot(n, &y[(size_t) r], 0, w.data() + (size_t) r * rb, 0, ap, 0, 1);
        const double g = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 8; ++rep) iq3_one_rows(type, w.data(), rb, n, &ap, 1, &yp, 0, rows);
        const double o = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (round > 0) { bg = std::min(bg, g); bo = std::min(bo, o); }
    }
    us_ggml = bg / 8;
    us_ours = bo / 8;
    return true;
}

}  // namespace strata::kernels::cpu
