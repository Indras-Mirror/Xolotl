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
    (void) scratch;
    ggml_backend_tensor_get(prediction, ids, 0, (size_t) prediction->ne[1] * 16 * sizeof(int32_t));
}

}  // namespace strata::v41
