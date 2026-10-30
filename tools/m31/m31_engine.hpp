#pragma once

#include "m31_geometry.hpp"
#include "m31_config.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::m31 {

// MiniMax-M3.1 MSA engine. See docs/MINIMAX_M31.md for prefill and numerical
// contracts. Single-token decode preserves the original ordered rank reduction.
class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    bool init(const std::string & gguf, const Config & cfg, std::string & error);
    const Geometry & geometry() const;
    // Append one chunk. A preflight rejection preserves valid state. Once execution
    // starts, an error requires reset(): caches may be partially written.
    bool forward(const int32_t * tokens, int n, std::string & error);
    // Linear decoder positions: -1 tokens select rows of the F32 vision table;
    // text tokens have visual_rows[i] == -1. Requires STRATA_M31_VISION.
    // Validation occurs before mutation, with the same failure/reset contract.
    bool forward_mixed(const int32_t * tokens, const int32_t * visual_rows,
                       const float * visual_embeddings, int visual_count,
                       int n, std::string & error);
    // Exact whole-prefix checkpoint: owned host copies of valid F16 K/V/index
    // rows only. Checkpoints belong to this initialized Engine; no model math,
    // weights, live graph pointers or expert-cache entries are serialized.
    class PrefixCheckpoint {
    public:
        ~PrefixCheckpoint() = default;
        PrefixCheckpoint(const PrefixCheckpoint &) = delete;
        PrefixCheckpoint & operator=(const PrefixCheckpoint &) = delete;
        size_t bytes() const { return bytes_.size(); }
        int64_t position() const { return position_; }
    private:
        friend class Engine;
        PrefixCheckpoint() = default;
        std::shared_ptr<const void> owner_;
        int64_t position_ = 0;
        std::vector<uint8_t> bytes_;
    };
    using PrefixCheckpointPtr = std::shared_ptr<const PrefixCheckpoint>;
    size_t prefix_checkpoint_bytes() const;
    bool checkpoint_prefix(size_t maximum_bytes, PrefixCheckpointPtr & out, std::string & error);
    // On success only forward() may consume the restored prefix; logits() has
    // no last chunk. The caller retains the exact last-prefix logits separately.
    // Rejected foreign/invalid checkpoints leave the current valid state alone.
    bool restore_prefix(const PrefixCheckpointPtr & checkpoint, std::string & error);
    // Reuse completed KV and MSA-index rows at a 256-token boundary.
    bool rewind(int64_t position, std::string & error);
    bool release_prefill_workspace(std::string & error);
    // Logits for a consecutive set of rows in the last chunk. Small row slices avoid vocab*n scratch.
    bool logits(int row, int count, std::vector<float> & out, std::string & error);
    void reset();
    int64_t position() const;
    struct PrefillStats {
        uint64_t hybrid_layers = 0, cpu_routes = 0, gpu_routes = 0, cold_gpu_bytes = 0;
        uint64_t quant_calls = 0, quant_runtime_allocations = 0;
        size_t quant_device_bytes = 0;
    };
    // Cumulative counters; requesting them performs no allocation or model work.
    PrefillStats prefill_stats() const;
    // Deep sleep releases every device allocation of this engine: dense
    // weights, the expert cache, KV/index rows, graph scratch and the arena's
    // registration. Host RAM keeps the arena, a byte copy of the dense device
    // buffers, the valid KV/index rows and the expert-cache residency. After
    // sleep() the borrowed backend is unused; the caller may free it and
    // release the device context. wake() takes a new backend for the same
    // device and restores the exact pre-sleep state, so the next forward
    // continues the conversation without reading it again. A failed sleep
    // still releases the device but may drop the conversation (position 0).
    // A failed wake releases what it allocated and stays asleep.
    // The first sleep copies the dense buffers to host RAM; the copy is kept
    // (the weights are immutable), so later sleeps only copy the KV rows.
    // wake() returns once dense weights, KV rows and the expert cache are on
    // the device; the arena then registers on a background thread, only while
    // no forward/logits ran for 250 ms, and a later forward switches expert
    // uploads from the staging ring to direct DMA.
    struct SleepStats {
        double seconds = 0, kv_seconds = 0, dense_seconds = 0;
        int64_t kv_tokens = 0;
        size_t kv_bytes = 0, dense_bytes = 0, cached_experts = 0;
        bool dense_copied = false, dense_pinned = false;
    };
    struct WakeStats {
        double seconds = 0, dense_seconds = 0, kv_seconds = 0, experts_seconds = 0;
        int64_t kv_tokens = 0;
        size_t cached_experts = 0;
        uint64_t expert_bytes = 0;
        bool dense_pinned = false, registration_background = false;
    };
    bool sleep(SleepStats & stats, std::string & error);
    bool wake(ggml_backend_t backend, WakeStats & stats, std::string & error);
    bool asleep() const;
    // Cap every backend at the MSA int32 limit. CUDA also applies a
    // conservative scratch-memory plan to the remaining prompt chunk.
    // In vision-enabled CUDA builds, cap by live VRAM to account for the helper.
    // If retained scratch prevents any chunk, release it and plan once again:
    // KV/position survive, but the previous chunk's logits are invalidated.
    // Consume that head before planning the next append. No new rows run here.
    // Returns 0 with error on invalid state/insufficient budget; forward never changes n.
    int prefill_chunk(int requested, std::string & error) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::m31
