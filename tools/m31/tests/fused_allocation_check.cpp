// A linker-only failure seam keeps allocator fault injection out of the engine.
#include "cpu/fused_decode.hpp"
#include "m31_engine.hpp"
#include <atomic>
#include <iostream>

namespace {
std::atomic<bool> fail_next_allocation{false};
std::atomic<bool> fail_next_routed{false};
std::atomic<unsigned> routed_failures{0};
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
struct Backend {
    ggml_backend_t value = ggml_backend_cpu_init();
    ~Backend() { if (value) ggml_backend_free(value); }
};
struct Pool {
    ggml_threadpool_t value = nullptr;
    explicit Pool(int threads) {
        auto params = ggml_threadpool_params_default(threads);
        value = ggml_threadpool_new(&params);
    }
    ~Pool() { if (value) ggml_threadpool_free(value); }
};
struct Weights {
    ggml_context * context = ggml_init({65536, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, 3> tensors{};
    explicit Weights(ggml_backend_t backend) {
        check(context != nullptr, "weight context");
        for (auto & tensor : tensors) tensor = ggml_new_tensor_3d(context, GGML_TYPE_IQ4_XS, 256, 256, 2);
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        check(buffer != nullptr, "weight allocation");
        for (size_t role = 0; role < tensors.size(); ++role) {
            std::vector<float> values(size_t(ggml_nelements(tensors[role]))), importance(256, 1.f);
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i * 17 + role * 7) % 101) - 50) * .002f;
            std::vector<uint8_t> packed(ggml_nbytes(tensors[role]));
            check(ggml_quantize_chunk(GGML_TYPE_IQ4_XS, values.data(), packed.data(), 0,
                int64_t(values.size() / 256), 256, importance.data()) == packed.size(), "weight quantization");
            ggml_backend_tensor_set(tensors[role], packed.data(), 0, packed.size());
        }
    }
    ~Weights() { if (buffer) ggml_backend_buffer_free(buffer); if (context) ggml_free(context); }
};
}
extern "C" bool __real_ggml_gallocr_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph * graph);
extern "C" bool __wrap_ggml_gallocr_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph * graph) {
    if (fail_next_allocation.exchange(false)) return false;
    // The CPU routed graph ends in a contiguous rank reduction. Dense and
    // output-head graphs have different terminal operations.
    if (fail_next_routed.load() && ggml_graph_n_nodes(graph) > 0 &&
        ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1)->op == GGML_OP_CONT &&
        fail_next_routed.exchange(false)) {
        ++routed_failures;
        return false;
    }
    return __real_ggml_gallocr_alloc_graph(allocator, graph);
}
int main(int argc, char ** argv) {
    try {
        Backend backend; check(backend.value != nullptr, "CPU backend");
        Pool pool(2); check(pool.value != nullptr, "CPU pool");
        ggml_backend_cpu_set_threadpool(backend.value, pool.value);
        {
            Weights weights(backend.value);
            strata::m31::CpuFusedDecode retry(backend.value, pool.value, 2);
            strata::m31::CpuFusedDecode fresh(backend.value, pool.value, 2);
            std::vector<float> input(256), actual, expected;
            for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 13) - 6) * .1f;
            auto run = [&](auto & executor, auto & output) {
                executor.forward(weights.tensors[0], weights.tensors[1], weights.tensors[2],
                    nullptr, nullptr, nullptr, input, {1, 0}, {.375f, .625f}, output);
            };
            for (int failure = 0; failure < 2; ++failure) {
                fail_next_allocation.store(true);
                bool rejected = false;
                try { run(retry, actual); }
                catch (const std::runtime_error & e) {
                    rejected = std::string(e.what()).find("graph allocation failed") != std::string::npos;
                }
                check(rejected && !fail_next_allocation.load(), "lazy graph allocation failure was not rejected");
            }
            run(retry, actual); run(fresh, expected);
            check(actual.size() == expected.size() && std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0,
                  "retry after allocation failures changed output bytes");
            run(retry, actual);
            check(std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0, "recovered graph was not reusable");
        }
        ggml_backend_cpu_set_threadpool(backend.value, nullptr);
        if (argc == 2) {
            strata::m31::Config cfg;
            cfg.context = 32; cfg.max_chunk = 1; cfg.threads = 2; cfg.cpu_fused = false;
            strata::m31::Engine retry, fresh;
            std::string error;
            check(retry.init(argv[1], cfg, error), error.c_str());
            check(fresh.init(argv[1], cfg, error), error.c_str());
            const int32_t token = 17;
            for (int failure = 0; failure < 2; ++failure) {
                fail_next_routed.store(true);
                check(!retry.forward(&token, 1, error) && !fail_next_routed.load(),
                      "CPU routed allocation failure was not reached/rejected");
                retry.reset();
            }
            check(routed_failures.load() == 2, "wrong routed failure count");
            check(retry.forward(&token, 1, error), error.c_str());
            check(fresh.forward(&token, 1, error), error.c_str());
            std::vector<float> actual, expected;
            check(retry.logits(0, 1, actual, error), error.c_str());
            check(fresh.logits(0, 1, expected, error), error.c_str());
            check(actual.size() == expected.size() && std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0,
                  "reset after CPU routed allocation failures changed head bytes");
        }
        std::cout << "two failed lazy allocations, retry and warm reuse passed bitwise\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
