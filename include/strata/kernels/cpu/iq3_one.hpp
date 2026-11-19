// include/strata/kernels/cpu/iq3_one.hpp - IQ3_XXS / IQ3_S expert rows for the calls ggml-cpu's vec_dot serves
// (src/kernels/cpu/iq3_one.cpp: Project Maya's register-codebook AVX-512 rows, the same bits as that dot).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Rows [r0, r1) of an IQ3_XXS (18) or IQ3_S (21) matrix (row stride rb, n columns) against nt tokens' Q8_K
/// activations: out[t][r - r0], bit for bit ggml-cpu's vec_dot.  Compiled /arch:AVX512: call only where
/// iq3_one_ok(type) (native_expert.cpp) holds.
void iq3_one_rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out,
                  int r0, int r1);

/// Once per type on this CPU (the caller has checked AVX-512 F / BW / VBMI and BMI2): these rows against ggml-cpu's
/// dot on random blocks (1, 2, 3 and 5 tokens) - true when every output is the same float - and both timed on 32 rows
/// of n_embd columns for one token (microseconds, the best of several rounds).
bool iq3_one_selftest(int type, int n_embd, double& us_ggml, double& us_ours);

/// The gate the callers use (native_expert.cpp): STRATA_IQ3_ONE=1 takes these rows where they are bit-exact here and
/// at least 10 % faster than ggml-cpu's dot, =2 wherever they are bit-exact; unset / 0: never.
bool iq3_one_ok(int type);

}  // namespace strata::kernels::cpu
