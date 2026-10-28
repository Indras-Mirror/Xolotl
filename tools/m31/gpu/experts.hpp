// MiniMax-M3.1 expert execution. Original GGUF quants, one cache shared by all layers.
#pragma once
#include "../m31_arena.hpp"
#include "../m31_config.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::m31 {

// Borrows the backend and arena, which must outlive this object. Serialized use only.
// The VRAM budget covers one fixed expert-slot pool; graph scratch/dense/KV are separate.
// A cache miss copies one complete expert slot through a small pinned staging buffer.
class GpuExperts {
public:
    struct Stats {
        uint64_t hits = 0, misses = 0, evictions = 0, upload_bytes = 0;
        uint64_t graph_batches = 0, forwards = 0;
        // Wall time on the calling thread. Async upload_submit can overlap prior
        // DMA; execute includes device completion and must not be called pure kernel time.
        double prepare_seconds = 0, pack_seconds = 0, upload_submit_seconds = 0;
        double upload_wait_seconds = 0, execute_seconds = 0;
        double gate_seconds = 0, up_seconds = 0, oai_seconds = 0;
        double down_seconds = 0, reduce_seconds = 0;
        size_t resident_bytes = 0, peak_resident_bytes = 0, budget_bytes = 0;
        size_t scratch_bytes = 0, minimum_budget_bytes = 0;
        size_t slot_stride = 0, slot_count = 0, pool_bytes = 0, occupied_slots = 0;
        bool pinned_staging = false;
        bool q8_activations = false;
        uint64_t prefetch_batches = 0, prefetched_experts = 0, prefetch_bytes = 0;
        double prefetch_wait_seconds = 0;
        double prefetch_pack_seconds = 0, prefetch_staging_wait_seconds = 0, prefetch_submit_seconds = 0;
    };
    GpuExperts();
    ~GpuExperts();
    GpuExperts(const GpuExperts &) = delete;
    GpuExperts & operator=(const GpuExperts &) = delete;

    // Called after dense weights and KV are resident, before allocating the
    // fixed expert pool. Uses the selected backend's exact quant padding.
    static uint64_t automatic_budget_bytes(const GgufModel & model, const Geometry & geometry,
                                           ggml_backend_t backend);
    // Throws on invalid budget/geometry/backend. Allocates a fixed bounded pool once;
    // no weights are uploaded until forward(). A batch must fit all its distinct experts.
    void init(const GgufModel & model, const Geometry & geometry, const ExpertArena & arena,
              ggml_backend_t backend, double cache_gib, int max_chunk, bool q8_activations = false, const Config & config = Config{});

    // x: [embd,n]; ids and weights: [used,n], all in contiguous GGML order.
    // Runs stock ggml GPU math. Contributions preserve original top-k order.
    // Uses OAI SwiGLU(alpha=1.702,limit=7) and each role's external NVFP4 scale.
    std::vector<float> forward(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & ids,
                               const std::vector<float> & weights);
    // Reuse caller-owned host capacity across layers/chunks. Output must not be
    // the same vector object as x or weights; on failure the caller must discard it.
    void forward_into(int layer, int n, const std::vector<float> & x,
                      const std::vector<int32_t> & ids, const std::vector<float> & weights,
                      std::vector<float> & output);
    // Device-only prefill. All tensors are contiguous GPU F32 [embd,n]; dense
    // input/shared/residual descriptors and buffers stay alive until return.
    // The destination is separate. Only IDs/route weights/scales cross H2D.
    // Completes the borrowed compute backend before returning.
    void forward_device(int layer,int n,ggml_tensor * normalized,
                        const std::vector<int32_t> & ids,const std::vector<float> & weights,
                        ggml_tensor * shared,ggml_tensor * residual,ggml_tensor * destination);
    // Readable during a live HTTP worker; no destructor or signal shutdown needed.
    void print_profile(const char * label = "live") const;
    bool native_gpu_math() const;
    size_t workspace_bytes() const;
#ifdef M31_ENGINE_VISION
    // Actual current gallocr allocation, including a failed-growth shrink.
    // workspace_bytes()/stats retain their existing observational contract.
    size_t retained_workspace_bytes() const;
#endif
    void release_prefill_workspace();
    const Stats & stats() const;
    void reset_stats(); // Keeps warm cache entries; counters restart at current residency.
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::m31
