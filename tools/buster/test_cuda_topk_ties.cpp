// Test selected index sets through the production CUDA TOP_K operation.
// The set comparison and indexer shape follow llama.cpp PR #31084.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

struct Shape {
    int columns;
    int rows;
    int selected;
};

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

struct Results {
    int executions = 0;
    int mismatched_rows = 0;
    int changed_sets = 0;
    int invalid_indices = 0;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> make_scores(const Shape& shape, int pattern) {
    std::vector<float> values((size_t) shape.columns * shape.rows, 0.0f);
    if (pattern == 0) {
        return values;
    }
    // Upstream's tied-input pattern uses integer division, followed by a
    // per-row shuffle. Fixed seeds make the control and corrected runs match.
    int denominator = std::max(1, std::min(10, shape.selected / 2));
    if (shape.columns == 1024 && shape.rows <= 16) {
        // These small-row controls can use bitonic, whose partial-tie index
        // order is outside the radix fix. Use distinct shuffled values here.
        denominator = 1;
    }
    for (int row = 0; row < shape.rows; ++row) {
        float* data = values.data() + (size_t) row * shape.columns;
        for (int column = 0; column < shape.columns; ++column) {
            data[column] = (float) (column / denominator - shape.columns / denominator / 2);
        }
        std::mt19937 random(31084u + (uint32_t) row * 17u + (uint32_t) shape.columns);
        std::shuffle(data, data + shape.columns, random);
    }
    return values;
}

std::vector<int32_t> expected_sets(const Shape& shape, const std::vector<float>& values) {
    std::vector<int32_t> expected((size_t) shape.rows * shape.selected);
    std::vector<int32_t> order((size_t) shape.columns);
    for (int row = 0; row < shape.rows; ++row) {
        std::iota(order.begin(), order.end(), 0);
        const float* scores = values.data() + (size_t) row * shape.columns;
        std::sort(order.begin(), order.end(), [&](int32_t first, int32_t second) {
            if (scores[first] != scores[second]) {
                return scores[first] > scores[second];
            }
            return first < second;
        });
        int32_t* output = expected.data() + (size_t) row * shape.selected;
        std::copy_n(order.begin(), shape.selected, output);
        std::sort(output, output + shape.selected);
    }
    return expected;
}

Results check_shape(ggml_backend_t backend, const Shape& shape) {
    GraphResources resources;
    const size_t metadata_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
    const ggml_init_params parameters = {metadata_bytes, nullptr, true};
    resources.context = ggml_init(parameters);
    require(resources.context != nullptr, "GGML context allocation failed");
    ggml_tensor* input = ggml_new_tensor_2d(resources.context, GGML_TYPE_F32, shape.columns, shape.rows);
    ggml_set_input(input);
    ggml_tensor* output = ggml_top_k(resources.context, input, shape.selected);
    require(output->op == GGML_OP_TOP_K, "test must exercise the production TOP_K operation");
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph_custom(resources.context, 16, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_supports_op(backend, output), "CUDA backend does not support TOP_K");
    resources.buffer = ggml_backend_alloc_ctx_tensors(resources.context, backend);
    require(resources.buffer != nullptr, "CUDA tensor allocation failed");

    Results results;
    for (int pattern = 0; pattern < 2; ++pattern) {
        const auto scores = make_scores(shape, pattern);
        const auto expected = expected_sets(shape, scores);
        std::vector<int32_t> reference;
        ggml_backend_tensor_set(input, scores.data(), 0, scores.size() * sizeof(float));
        for (int repeat = 0; repeat < 4; ++repeat) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                    "CUDA TOP_K execution failed");
            std::vector<int32_t> actual((size_t) shape.rows * shape.selected);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(int32_t));
            for (int row = 0; row < shape.rows; ++row) {
                int32_t* begin = actual.data() + (size_t) row * shape.selected;
                std::sort(begin, begin + shape.selected);
                const int32_t* want = expected.data() + (size_t) row * shape.selected;
                const bool invalid = begin[0] < 0 || begin[shape.selected - 1] >= shape.columns ||
                                     std::adjacent_find(begin, begin + shape.selected) != begin + shape.selected;
                results.invalid_indices += invalid;
                if (!std::equal(begin, begin + shape.selected, want)) {
                    if (results.mismatched_rows == 0) {
                        const auto mismatch = std::mismatch(begin, begin + shape.selected, want);
                        std::fprintf(stderr,
                                     "first mismatch columns=%d rows=%d k=%d pattern=%d repeat=%d row=%d "
                                     "expected_index=%d actual_index=%d\n",
                                     shape.columns, shape.rows, shape.selected, pattern, repeat, row,
                                     (int) *mismatch.second, (int) *mismatch.first);
                    }
                    ++results.mismatched_rows;
                }
            }
            if (repeat == 0) {
                reference = actual;
            } else if (actual != reference) {
                ++results.changed_sets;
            }
            ++results.executions;
        }
    }
    std::printf("TOP_K columns=%d rows=%d k=%d executions=%d mismatched_rows=%d changed_sets=%d invalid_indices=%d\n",
                shape.columns, shape.rows, shape.selected, results.executions,
                results.mismatched_rows, results.changed_sets, results.invalid_indices);
    return results;
}

}  // namespace

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::fprintf(stderr, "SKIP no CUDA device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend || !ggml_backend_is_cuda(backend)) {
        if (backend) {
            ggml_backend_free(backend);
        }
        std::fprintf(stderr, "FAIL CUDA backend initialization\n");
        return 1;
    }
    const std::array<Shape, 9> shapes = {{
        {1024, 3, 512}, {1024, 16, 512}, {1024, 512, 512},
        {4096, 3, 512}, {4096, 16, 512}, {4096, 512, 512},
        {4096, 16, 16}, {4096, 3, 2048}, {1088, 2048, 512}
    }};
    int status = 0;
    try {
        for (const auto& shape : shapes) {
            const Results result = check_shape(backend, shape);
            if (result.mismatched_rows || result.changed_sets || result.invalid_indices) {
                status = 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA TOP_K tie regression %s\n", error.what());
        status = 1;
    }
    ggml_backend_free(backend);
    std::printf("%s CUDA TOP_K cutoff ties and repeated selected sets\n", status == 0 ? "PASS" : "FAIL");
    return status;
}
