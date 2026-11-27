// src/kernels/cpu/tq_avx2.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX2 + FMA (tq_cpu.hpp).
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU.
//
// Per 128-weight block: the 32 states in scalar code (tq_states.hpp: one 32-bit read, a shift and a mask each, used
// as table indices straight away), four 8-byte state-table rows per YMM register as scalar loads, then per token
// VPMADDWD against the unpacked int16 activations and a VPHADDD tree that leaves the four exact sub-block sums
// S[k] in one XMM, so the float part is the scalar kernel's, lane for lane.  The weights are decoded once per block
// for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block, one token, AVX2): ~48 with this scheme; computing the states with byte
// shuffles instead (VPSHUFB / VPSRLVD per 8 states) and moving them to GPRs or through memory was slower (60-80),
// and ggml-cpu's AVX2 vec_dot for these types (agentionai llama.cpp, the same state table) takes about 2.3x as long.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu::tq_detail {
namespace {

// four state-table rows (8 bytes each) into one YMM: a load and three broadcast loads blended in.  Broadcast loads
// and VPBLENDD avoid the shuffle port that VPINSRQ / VINSERTI128 need (the two measure the same on Zen 5).
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

template <int KB>
void rows_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    const long long* tab = (const long long*) table;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            __m256i wv[8];   // wv[m] = weights 16m .. 16m+15 (states 4m .. 4m+3)
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
            for (int m = 0; m < 8; ++m)
                wv[m] = rows4(tab, tq_state<KB>(qs, wr, 4 * m), tq_state<KB>(qs, wr, 4 * m + 1),
                              tq_state<KB>(qs, wr, 4 * m + 2), tq_state<KB>(qs, wr, 4 * m + 3));
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                const __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(wv[0], _mm256_load_si256(q + 0)),
                                                    _mm256_madd_epi16(wv[1], _mm256_load_si256(q + 1)));
                const __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(wv[2], _mm256_load_si256(q + 2)),
                                                    _mm256_madd_epi16(wv[3], _mm256_load_si256(q + 3)));
                const __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(wv[4], _mm256_load_si256(q + 4)),
                                                    _mm256_madd_epi16(wv[5], _mm256_load_si256(q + 5)));
                const __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(wv[6], _mm256_load_si256(q + 6)),
                                                    _mm256_madd_epi16(wv[7], _mm256_load_si256(q + 7)));
                // per 128-bit lane [p0, p1, p2, p3] partial sums, then the two lanes added: S[0..3], exact
                const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

// TQP (147-149): 64 pair-steps per 128-value block.  A step's two weights are the FIRST pair of its state
// table row, so 8 states make one YMM of 16 int16 weights, and a 32-value group (16 steps) is two YMM of
// weights against two YMM of activations: VPMADDWD, one exact int32 sum per group, then the four group sums
// go into the same per-token __m128 the other kernels use, so the float part is the scalar kernel's.
inline __m256i pairs8(const int16_t* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4,
                      uint32_t s5, uint32_t s6, uint32_t s7) {
    const __m128i a0 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s0)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s1)));
    const __m128i a1 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s2)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s3)));
    const __m128i a2 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s4)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s5)));
    const __m128i a3 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s6)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s7)));
    return _mm256_set_m128i(_mm_unpacklo_epi64(a2, a3), _mm_unpacklo_epi64(a0, a1));
}

template <int KB>
void rows_tqp_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * KB;
    const int16_t* tab = tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            uint32_t st[64];
            tq_states_tqp<KB>(blk + 2, st);
            __m256i wv[8];   // wv[2g], wv[2g+1]: the weights of the 32-value group g
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                const uint32_t* s = st + 16 * g;
                wv[2 * g] = pairs8(tab, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
                wv[2 * g + 1] = pairs8(tab, s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]);
            }
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                int32_t sum[4];
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wv[2 * g], _mm256_load_si256(q + 2 * g)),
                                                       _mm256_madd_epi16(wv[2 * g + 1], _mm256_load_si256(q + 2 * g + 1)));
                    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
                    s = _mm_hadd_epi32(s, s);
                    s = _mm_hadd_epi32(s, s);
                    sum[g] = _mm_cvtsi128_si32(s);
                }
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                const __m128i s4 = _mm_set_epi32(sum[3], sum[2], sum[1], sum[0]);
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s4), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

// ---- TQP fast kernel (25 Nov): point table, vector state windows, two passes per row --------------------------
// State t of a TQP block = the 16 stream bits from bit (63 - t)*KB, masked to 15 bits; its two weights are the
// pair of codebook point (s * 0x9e3779b1) >> 21 (tq_point_table, 8 KB, L1-resident; the 256 KB state table thrashed
// L1/L2).  The 8 states t = 8h .. 8h+7 lie inside a <= 11-byte window at a compile-time byte offset: one 16-byte
// load broadcast to both lanes, VPSHUFB (each 32-bit lane takes its 4 bytes), VPSRLVD (its bit shift), mask,
// VPMULLD + shift = 8 hashed points.  Pass 1 writes a row's points to an L1 buffer; pass 2 loads the pairs with
// scalar loads (Zen 3's 8-wide VPGATHERDD costs ~12 cycles, more than everything else) and does the dots.  Two
// passes because one pass chains window -> hash -> store -> reload -> table load -> madd (~50 cycles) and the reorder
// window holds about one block.  Same int16 numbers and the same exact int32 group sums as the scalar kernel: the
// float results are bit-identical (tqp_avx2_exact_test).
// Zen 3 (5700X, 1 thread, ERNIE expert 2048 -> 2x2048 + down, 1 token): 6.6 ms (state table) -> 2.0 ms; tried and
// dropped: one 8-wide gather (3.2-3.8), two 4-wide (3.7), one pass + scalar loads (2.7), u32 index reloads (2.4-3.6),
// pairs joined in GPRs (3.3), a block copy for the wrap (4.5: store-forwarding stalls).

template <int KB>
struct TqpCtl {
    alignas(32) uint8_t shuf[8][32];
    alignas(32) uint32_t sh[8][8];
    int base[8];   // the window's first stream byte
    int need[8];   // one past the last stream byte the window's states use (may pass 8*KB: the circular wrap)
    constexpr TqpCtl() : shuf{}, sh{}, base{}, need{} {
        for (int h = 0; h < 8; ++h) {
            const int bmin = ((56 - 8 * h) * KB) >> 3;   // the lowest offset is t = 8h+7
            base[h] = bmin;
            int nd = 0;
            for (int j = 0; j < 8; ++j) {
                const int rel = (63 - 8 * h - j) * KB - 8 * bmin;
                const int b = rel >> 3;
                sh[h][j] = (uint32_t) (rel & 7);
                for (int k = 0; k < 4; ++k) shuf[h][4 * j + k] = (uint8_t) (b + k);
                const int e = bmin + ((rel + 15 + 7) >> 3);   // bytes holding bits [rel, rel+15)
                nd = e > nd ? e : nd;
            }
            need[h] = nd;
        }
    }
};

template <int KB>
inline constexpr TqpCtl<KB> kTqpCtl{};

// the 16 stream bytes from c.base[H] - straight from the block, never outside it: a plain load where it fits, the
// block's last 16 bytes shifted down near the end, and those with the block's first bytes appended (VPALIGNR) for the
// circular windows.  Bytes past `need` are never used, so the zeros VPSRLDQ shifts in do not matter.
template <int KB, int H>
inline __m128i tqp_window(const uint8_t* qs) {
    constexpr TqpCtl<KB> c = kTqpCtl<KB>;
    constexpr int NB = 8 * KB, base = c.base[H];
    if constexpr (base + 16 <= NB) {
        return _mm_loadu_si128((const __m128i*) (qs + base));
    } else if constexpr (c.need[H] <= NB) {
        return _mm_srli_si128(_mm_loadu_si128((const __m128i*) (qs + NB - 16)), base - (NB - 16));
    } else {
        static_assert(c.need[H] - NB <= base - (NB - 16), "the wrap fits one VPALIGNR");
        return _mm_alignr_epi8(_mm_loadu_si128((const __m128i*) qs), _mm_loadu_si128((const __m128i*) (qs + NB - 16)),
                               base - (NB - 16));
    }
}

template <int KB, int H>
inline __m256i tqp_points8(const uint8_t* qs) {
    const TqpCtl<KB>& c = kTqpCtl<KB>;
    __m256i v = _mm256_broadcastsi128_si256(tqp_window<KB, H>(qs));
    v = _mm256_shuffle_epi8(v, _mm256_load_si256((const __m256i*) c.shuf[H]));
    v = _mm256_srlv_epi32(v, _mm256_load_si256((const __m256i*) c.sh[H]));
    v = _mm256_and_si256(v, _mm256_set1_epi32(0x7fff));
    return _mm256_srli_epi32(_mm256_mullo_epi32(v, _mm256_set1_epi32((int) 0x9e3779b1u)), 21);
}

// the 8 pairs of 8 stored points: the indices read back as u64 pairs (fewer load uops), VMOVD + unpacks
inline __m256i tqp_pairs_from(const int* pts, const uint32_t* ix) {
    uint64_t i0, i1, i2, i3;
    std::memcpy(&i0, ix + 0, 8);
    std::memcpy(&i1, ix + 2, 8);
    std::memcpy(&i2, ix + 4, 8);
    std::memcpy(&i3, ix + 6, 8);
    auto ld = [pts](uint32_t i) { return _mm_cvtsi32_si128(pts[i]); };
    const __m128i a0 = _mm_unpacklo_epi32(ld((uint32_t) i0), ld((uint32_t) (i0 >> 32)));
    const __m128i a1 = _mm_unpacklo_epi32(ld((uint32_t) i1), ld((uint32_t) (i1 >> 32)));
    const __m128i a2 = _mm_unpacklo_epi32(ld((uint32_t) i2), ld((uint32_t) (i2 >> 32)));
    const __m128i a3 = _mm_unpacklo_epi32(ld((uint32_t) i3), ld((uint32_t) (i3 >> 32)));
    return _mm256_set_m128i(_mm_unpacklo_epi64(a2, a3), _mm_unpacklo_epi64(a0, a1));
}

// per-lane pair sums of a and b as VPHADDD gives them, from two shuffles and an add (exact int32; ~3 % faster at
// 8 tokens on Zen 3)
inline __m256i tqp_hadd(__m256i a, __m256i b) {
    const __m256 fa = _mm256_castsi256_ps(a), fb = _mm256_castsi256_ps(b);
    return _mm256_add_epi32(_mm256_castps_si256(_mm256_shuffle_ps(fa, fb, 0x88)),
                            _mm256_castps_si256(_mm256_shuffle_ps(fa, fb, 0xDD)));
}

template <int KB>
void rows_tqp3_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                 int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * KB;
    const int* pts = (const int*) tq_point_table();
    alignas(64) uint32_t ix[kTqMaxN / 2];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = x + (size_t) i * bsz + 2;
            __m256i* d = (__m256i*) (ix + 64 * i);
            _mm256_store_si256(d + 0, tqp_points8<KB, 0>(qs));
            _mm256_store_si256(d + 1, tqp_points8<KB, 1>(qs));
            _mm256_store_si256(d + 2, tqp_points8<KB, 2>(qs));
            _mm256_store_si256(d + 3, tqp_points8<KB, 3>(qs));
            _mm256_store_si256(d + 4, tqp_points8<KB, 4>(qs));
            _mm256_store_si256(d + 5, tqp_points8<KB, 5>(qs));
            _mm256_store_si256(d + 6, tqp_points8<KB, 6>(qs));
            _mm256_store_si256(d + 7, tqp_points8<KB, 7>(qs));
        }
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint32_t* bi = ix + 64 * i;
            __m256i wv[8];
#pragma GCC unroll 8
            for (int h = 0; h < 8; ++h) wv[h] = tqp_pairs_from(pts, bi + 8 * h);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                const __m256i p0 = _mm256_add_epi32(_mm256_madd_epi16(wv[0], _mm256_load_si256(q + 0)),
                                                    _mm256_madd_epi16(wv[1], _mm256_load_si256(q + 1)));
                const __m256i p1 = _mm256_add_epi32(_mm256_madd_epi16(wv[2], _mm256_load_si256(q + 2)),
                                                    _mm256_madd_epi16(wv[3], _mm256_load_si256(q + 3)));
                const __m256i p2 = _mm256_add_epi32(_mm256_madd_epi16(wv[4], _mm256_load_si256(q + 4)),
                                                    _mm256_madd_epi16(wv[5], _mm256_load_si256(q + 5)));
                const __m256i p3 = _mm256_add_epi32(_mm256_madd_epi16(wv[6], _mm256_load_si256(q + 6)),
                                                    _mm256_madd_epi16(wv[7], _mm256_load_si256(q + 7)));
                const __m256i hh = tqp_hadd(tqp_hadd(p0, p1), tqp_hadd(p2, p3));
                const __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(hh), _mm256_extracti128_si256(hh, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s4), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

bool tqp_old_kernel() {
    static const bool old = [] { const char* e = std::getenv("STRATA_TQP_OLD"); return e && *e == '1'; }();
    return old;
}

}  // namespace

void tq_rows_avx2_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                      float* const* out, int r0, int r1) {
    if (tqp_old_kernel()) {   // A/B only: the state-table kernel (b9b4ae011)
        if (kb == 5) rows_tqp_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
        else if (kb == 6) rows_tqp_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
        else rows_tqp_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
        return;
    }
    if (kb == 5) rows_tqp3_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_tqp3_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_tqp3_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

void tq_rows_avx2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
    if (kb == 8) rows_t<8>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 5) rows_t<5>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK5X
    else if (kb == 4) rows_t<4>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK4X
    else rows_t<7>(table, w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu::tq_detail
