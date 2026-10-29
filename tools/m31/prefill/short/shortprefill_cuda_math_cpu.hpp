// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::m31::cuda_math_cpu {

// CUDA MMQ's D4 layout has F32 scales per 32 values. It is deliberately
// different from both CPU Q8_K (256 values) and ordinary Q8_1 (F16 scales).
struct Q8Mmq32 {
    float d = 0;
    int8_t qs[32]{};
};

bool supported(ggml_type type);
void quantize_mmq_d4_into(const float * input, size_t columns, Q8Mmq32 * output);
void quantize_mmq_d4(const float * input, size_t columns, std::vector<Q8Mmq32> & output);
int32_t integer_dot32(const int8_t * weight, const int8_t * activation);

// Decode one group of 32 original GGUF values into signed integer values and
// its F32 scale, matching mmq-load-tiles.cuh's IQ3_XXS/IQ4_XS loaders.
void unpack_weight32(ggml_type type, const void * weight_row, size_t group,
                     int8_t * values, float & scale);

// Scalar arithmetic reference for native NVIDIA MMQ: integer dot per 32
// values, then float(intdot)*weight_scale, then FMA with activation scale.
// Split-K may change the final reduction order on the actual GPU.
float dot_mmq(ggml_type type, const void * weight_row, size_t columns,
              const Q8Mmq32 * activation);

// A small assignment tile reuses each decoded weight group across all its
// activations. Output[token*output_token_stride+row], tokens must be <=16.
// Each activation points at columns/32 contiguous Q8Mmq32 groups.
// Optional corrections[token][group] must equal 128*sum(qs[32]) for the
// corresponding activation. They avoid recomputing this integer term per tile;
// the Q8Mmq32 byte format and floating-point arithmetic remain unchanged.
void mul_mat_quantized_tile(ggml_type type, const void * weights, size_t weight_row_stride,
                            size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                            size_t tokens, float * output, size_t output_token_stride,
                            const int32_t * const * corrections = nullptr);

// Previous VNNI implementation, retained for bitwise regression comparisons.
void mul_mat_quantized_tile_vnni_reference(ggml_type type, const void * weights, size_t weight_row_stride,
                                          size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                                          size_t tokens, float * output, size_t output_token_stride);

// Regression/performance comparison with the previous AVX2 assignment tile.
void mul_mat_quantized_tile_avx2_reference(ggml_type type, const void * weights, size_t weight_row_stride,
                                          size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                                          size_t tokens, float * output, size_t output_token_stride);

// One expert matrix, output [rows,tokens], x [columns,tokens]. This is a
// correctness probe, not a threaded executor; no GPU/backend is created.
void mul_mat_rows(ggml_type type, const void * weights, size_t weight_row_stride,
                  size_t rows, size_t columns, const float * x, size_t tokens,
                  std::vector<float> & output);

} // namespace strata::m31::cuda_math_cpu
