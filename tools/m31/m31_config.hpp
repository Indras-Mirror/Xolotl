#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace strata::m31 {
// Accepted native batch/decode math and automatic short-chunk CPU/GPU split.
// Algorithm thresholds are fixed; only resource budgets are knobs.
struct Config {
    ggml_backend_t backend = nullptr; // borrowed; nullptr selects CPU
    int threads = int(std::max(1u, std::min(32u, std::thread::hardware_concurrency())));
    int64_t context = 4096;
    int max_chunk = 32768;
    double arena_gib = 0; // CPU: mapped weights; CUDA: exact complete arena budget
    double gpu_experts_gib = 0; // CUDA: automatic bounded current/next-layer pool
    int gpu_experts_min_tokens = 32;
    bool reuse_allocations = true;
    bool persistent_cpu_pool = true;
    bool cpu_residual = true;
    bool cpu_decode_cache = true;
    bool dense_decode_cache = true;
    bool profile = false;
    bool cpu_fused = true;
    int cpu_poll = -1; // -1 preserves ggml's default
    std::string cpu_affinity;
    bool register_arena = false; // explicit opt-in; never pins the full model implicitly
    uint64_t register_limit_bytes = 0;
    uint64_t register_chunk_bytes = uint64_t(1) << 30;
    unsigned register_threads = 4;
    // Threads that copy the GGUF expert tensors into the RAM arena. With
    // registration, each filled piece is registered behind the fill.
    unsigned arena_fill_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    unsigned gpu_pack_threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    unsigned upload_depth = 8;
    unsigned upload_streams = 1;
    unsigned prefetch_limit = 128;
    std::function<bool()> cancelled;
    std::function<void(int, int, const std::vector<int32_t> &, const std::vector<float> &)> route_observer;
    std::function<void(const std::string &, int64_t, const ggml_tensor *)> tap;
};
// Run before allocating model/backend resources. Throws invalid_argument.
void validate_config(const Config & config, bool initializing_cuda = false);
// Entry-point only: set fusion guard in this engine process, before backend init.
// A caller that explicitly requested fusion gets an error instead of a silent override.
void prepare_engine_environment();
std::string effective_config(const Config & config);
} // namespace strata::m31
