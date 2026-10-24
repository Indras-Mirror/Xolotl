// tools/v41/engine/ds41_embd.hpp - one token_embd row to floats, whatever the file's type: F16 (ours) or a ggml quant
// type (ISTA-DASLab's GGUFs store it Q5_K).  The trunk's embedding and DSpark's both read rows off the mmap with this.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace strata::v41 {

/// true if ds41_embd_row can read rows of `type` (F16, F32, BF16, or any type with a to_float)
inline bool ds41_embd_type_ok(int type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) return false;
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_get_type_traits((ggml_type) type)->to_float != nullptr;
}

/// row `id` of a [D, vocab] embedding of `type` at `base` -> out[0..D)
inline void ds41_embd_row(const uint8_t * base, int type, int64_t D, int64_t id, float * out) {
    const size_t row = ggml_row_size((ggml_type) type, D);
    const uint8_t * src = base + (size_t) id * row;
    if (type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, out, D); return; }
    if (type == GGML_TYPE_F32) { for (int64_t i = 0; i < D; ++i) out[i] = ((const float *) src)[i]; return; }
    ggml_get_type_traits((ggml_type) type)->to_float(src, out, D);
}

}  // namespace strata::v41
