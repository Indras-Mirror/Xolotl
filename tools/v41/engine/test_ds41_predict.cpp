// Exercise the production predictor's top-k graph and its flat D2H payload.
#include "ds41_predict.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kExperts = 384;
constexpr int kPredictions = 16;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct GraphResources {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~GraphResources() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (context) {
            ggml_free(context);
        }
    }
};

void check_width(ggml_backend_t backend, int width) {
    GraphResources resources;
    const size_t metadata_bytes = ggml_tensor_overhead() * 32 +
                                  ggml_graph_overhead_custom(16, false);
    const ggml_init_params parameters = {metadata_bytes, nullptr, true};
    resources.context = ggml_init(parameters);
    require(resources.context != nullptr, "GGML context allocation failed");

    ggml_tensor* scores = ggml_new_tensor_2d(resources.context, GGML_TYPE_F32, kExperts, width);
    ggml_set_input(scores);
    ggml_tensor* prediction = strata::v41::ds41_prediction_topk(resources.context, scores);
    ggml_set_output(prediction);
    ggml_cgraph* graph = ggml_new_graph_custom(resources.context, 16, false);
    ggml_build_forward_expand(graph, prediction);
    for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, node)),
                "predictor graph contains an unsupported CUDA operation");
    }
    resources.buffer = ggml_backend_alloc_ctx_tensors(resources.context, backend);
    require(resources.buffer != nullptr, "CUDA tensor allocation failed");

    for (int iteration = 0; iteration < 4; ++iteration) {
        std::vector<float> values((size_t) width * kExperts);
        std::vector<int32_t> expected((size_t) width * kPredictions);
        for (int token = 0; token < width; ++token) {
            // Multiplication by 37 permutes all 384 scores without ties.
            // Each row and execution puts different experts in its top 16.
            for (int expert = 0; expert < kExperts; ++expert) {
                const int permutation = (expert * 37 + token * 53 + iteration * 71) % kExperts;
                values[(size_t) token * kExperts + expert] =
                    (float) (permutation - 192) * 0.25f + (float) token * 0.03125f;
            }
            std::array<int32_t, kExperts> order;
            std::iota(order.begin(), order.end(), 0);
            const float* row = values.data() + (size_t) token * kExperts;
            std::sort(order.begin(), order.end(), [&](int32_t first, int32_t second) {
                return row[first] > row[second];
            });
            std::copy_n(order.begin(), kPredictions,
                        expected.begin() + (size_t) token * kPredictions);
        }

        ggml_backend_tensor_set(scores, values.data(), 0, values.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                "CUDA predictor graph execution failed");
        // predict_multi and the queued actor consume this exact flat layout.
        std::vector<int32_t> actual((size_t) width * kPredictions, -1);
        ggml_backend_tensor_get(prediction, actual.data(), 0, actual.size() * sizeof(int32_t));
        for (size_t index = 0; index < actual.size(); ++index) {
            if (actual[index] != expected[index]) {
                std::fprintf(stderr,
                             "width=%d iteration=%d token=%zu rank=%zu expected=%d actual=%d stride=%zu\n",
                             width, iteration, index / kPredictions, index % kPredictions,
                             (int) expected[index], (int) actual[index], prediction->nb[1]);
                throw std::runtime_error("flat predictor payload differs from the per-row oracle");
            }
        }
        require(ggml_is_contiguous(prediction), "prediction payload must be contiguous");
        require(width == 1 || prediction->nb[1] == kPredictions * sizeof(int32_t),
                "prediction row stride differs from its flat consumer contract");
    }
    std::printf("PASS CUDA prediction width=%d executions=4\n", width);
}

}  // namespace

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::fprintf(stderr, "SKIP no CUDA device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend || !ggml_backend_is_cuda(backend)) {
        std::fprintf(stderr, "FAIL CUDA backend initialization\n");
        if (backend) {
            ggml_backend_free(backend);
        }
        return 1;
    }
    int status = 0;
    try {
        for (int width : {1, 2, 3, 4, 5, 6, 8}) {
            check_width(backend, width);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL predictor payload regression %s\n", error.what());
        status = 1;
    }
    ggml_backend_free(backend);
    return status;
}
