// include/strata/kernels/cpu/ptq1_0.hpp - PTQ1_0 ternary weights on the CPU (DeepSeek-V4.1 mixed-tier quant).
//
// PTQ1_0 is llama-tq's group-128 ternary type (github.com/LL4nc33/llama-tq, MIT; GGML type id 143): weights
// {-1, 0, +1} * d with one fp16 d per 128 weights, packed in base 3 like ggml's TQ1_0.  28 bytes per 128 weights
// (1.75 bpw).  This file is Strata's own implementation of that layout, bit-compatible with llama-tq's
// quantize_row_ptq1_0_ref / dequantize_row_ptq1_0:
//   qs[0..15]  : 5 trits per byte, value m + 16 n (m = byte, n = trit, most significant first)  -> values   0..79
//   qs[16..23] : 5 trits per byte, value 80 + m + 8 n                                          -> values  80..119
//   qh[0..1]   : 4 trits per byte, value 120 + h + 2 n (stored shifted to the top trit)         -> values 120..127
//   d          : fp16 scale
// A byte b holds trits as round-up(t0 t1 t2 t3 t4 (base 3) * 256 / 243); trit n is ((uint8)(b * 3^n) * 3) >> 8.
//
// Activations are Q8_0 (ggml's block_q8_0: fp16 d + 32 int8), i.e. a 128-weight block meets 4 activation blocks.
// The dot product is d_w * sum_k d_a[k] * sum_i (t_i - 1) q_i, computed as sum t q - sum q on the {0,1,2} digits.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

inline constexpr int kPtq1Type = 143;      ///< GGML type id in llama-tq (not in upstream ggml: GGML_TYPE_COUNT is 43)
inline constexpr int kPtq1Block = 128;     ///< weights per block
inline constexpr int kPtq1Bytes = 28;      ///< bytes per block
inline constexpr int kQ2_0Type = 42;      ///< ggml's Q2_0 ({-1,0,1,2} x fp16 per 64; the V4.1 XQ middle tier)

/// Bytes of one PTQ1_0 row of n weights (n a multiple of 128).
inline size_t ptq1_0_row_bytes(int64_t n) { return (size_t) (n / kPtq1Block) * kPtq1Bytes; }

/// Packs ternary codes t in {-1, 0, +1} (n of them) with one scale per 128 into PTQ1_0 blocks (GSQ export).
void ptq1_0_pack(const int8_t* t, const float* d, uint8_t* dst, int64_t n);
/// Round-to-nearest reference quantizer (amax scale), bit-identical to llama-tq's quantize_row_ptq1_0_ref.
void ptq1_0_quantize_ref(const float* x, uint8_t* dst, int64_t n);
/// Codes back out of a row (t in {-1,0,+1}) and the per-block scales.
void ptq1_0_unpack(const uint8_t* src, int8_t* t, float* d, int64_t n);
void ptq1_0_dequant(const uint8_t* src, float* y, int64_t n);

/// ggml_vec_dot_t-compatible: s[0] = PTQ1_0 row vx . Q8_0 row vy, n weights (nrc must be 1).
void ptq1_0_vec_dot_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
/// The scalar form of the same (the parity reference for the AVX2 one).
void ptq1_0_vec_dot_q8_0_scalar(int n, float* s, const void* vx, const void* vy);
/// Rows [r0, r1) of a PTQ1_0 matrix (row stride `row_bytes`, n weights a row) against nt Q8_0 rows (nt <= 8):
/// out[t][r] = row r . act[t].  The AVX2 path decodes each 128-weight block ONCE for all tokens (every digit in SIMD)
/// and keeps each 32-weight sub-block's sum exact in int32 (as the scalar dot) - 3.9x the per-token dot at one token,
/// 6-8x at 2-3 (s11 2026-10-15, one Zen 3 core).  Without AVX2: the per-token dot.
void ptq1_0_rows_q8_0(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1);
/// Rows [r0, r1) of a ggml Q2_0 matrix (18-byte blocks of 64 weights) against nt Q8_0 rows: out[t][r] = row r .
/// act[t].  ggml-cpu has only the scalar _generic Q2_0 dot on x86; the AVX2 path decodes each block once for all
/// tokens with the same integer sums.  Without AVX2: the scalar arithmetic of ggml's _generic dot.
void q2_0_rows_q8_0(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
