#pragma once

#include "ggml.h"

namespace strata::v41 {

// Prediction consumers read 16 expert IDs per token in one flat transfer.
inline ggml_tensor* ds41_prediction_topk(ggml_context* context, ggml_tensor* scores) {
    return ggml_argsort_top_k(context, scores, 16);
}

}  // namespace strata::v41
