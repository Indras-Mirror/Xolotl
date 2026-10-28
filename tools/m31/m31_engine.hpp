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
    // Cap every backend at the MSA int32 limit. CUDA also applies a
    // conservative scratch-memory plan to the remaining prompt chunk.
    // Returns0 with error on invalid state/insufficient budget; forward never changes n.
    int prefill_chunk(int requested, std::string & error) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace strata::m31
