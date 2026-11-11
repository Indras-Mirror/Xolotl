#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::v41 {

// The prediction graph selects 16 expert IDs per token.
inline ggml_tensor* ds41_prediction_topk(ggml_context* context, ggml_tensor* scores) {
    return ggml_argsort_top_k(context, scores, 16);
}

inline void ds41_prediction_readback(ggml_tensor* prediction, int32_t* ids,
                                     std::vector<int32_t>& scratch) {
    GGML_ASSERT(prediction->type == GGML_TYPE_I32 && prediction->ne[0] == 16);
    GGML_ASSERT(prediction->ne[1] > 0 && prediction->ne[2] == 1 && prediction->ne[3] == 1);
    const size_t rows = (size_t) prediction->ne[1];
    const size_t row_bytes = 16 * sizeof(int32_t);
    if (ggml_is_contiguous(prediction)) {
        ggml_backend_tensor_get(prediction, ids, 0, rows * row_bytes);
        return;
    }
    // Top-k is a view of the full argsort. Read its span once, then retain
    // each row's selected prefix in the packed host payload.
    GGML_ASSERT(prediction->nb[1] >= row_bytes && prediction->nb[1] % sizeof(int32_t) == 0);
    const size_t span_bytes = prediction->nb[1] * (rows - 1) + row_bytes;
    scratch.resize(span_bytes / sizeof(int32_t));
    ggml_backend_tensor_get(prediction, scratch.data(), 0, span_bytes);
    const size_t stride = prediction->nb[1] / sizeof(int32_t);
    for (size_t row = 0; row < rows; ++row) {
        std::copy_n(scratch.data() + row * stride, 16, ids + row * 16);
    }
}

}  // namespace strata::v41
