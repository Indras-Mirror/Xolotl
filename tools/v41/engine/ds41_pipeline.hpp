#pragma once

#include "ds41_dense.hpp"
#include "ds41_moe.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

struct Ds41Stage {
    int cuda_device = 0;
    int64_t layer_lo = 0;
    int64_t layer_hi = 0;
    std::vector<int64_t> import_layers;   ///< shadow layers copied from the previous stage at handoff
    bool import_cand = false;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    std::unique_ptr<Ds41Dense> dense;
    std::unique_ptr<Ds41MoeTier> tier;
};

/// Output head (+ DSpark) on a device that may differ from the tail trunk stage (or share a stage's backend).
struct Ds41HeadStage {
    int cuda_device = -1;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    std::unique_ptr<Ds41Dense> dense;
#ifdef STRATA_ENABLE_CUDA
    void * cuda_stream = nullptr;   ///< cudaStream_t
#endif
};

struct Ds41Pipeline {
    std::vector<Ds41Stage> stages;
    int split_layer = 0;   ///< first boundary (legacy; == first split when enabled)
    std::vector<int> split_layers;   ///< increasing layer indices; stage s is [bound(s), bound(s+1))
    std::vector<float> handoff_host;   ///< pinned-capable host buffer for cross-GPU handoff
    std::unique_ptr<Ds41HeadStage> head;

    ~Ds41Pipeline();

    bool enabled() const { return stages.size() > 1; }
    bool head_offload() const { return head && head->dense; }
    bool head_logits(int row0, int nrows, float * out, std::string & err);
    bool sync_head_state(int col0, int n, std::string & err);
    /// Handoff state from stage `from` to `from + 1` (adjacent only).
    bool sync_handoff_between(int from_stage, int to_stage, int col0, int n, std::string & err);
    bool sync_handoff(int col0, int n, std::string & err) { return sync_handoff_between(0, 1, col0, n, err); }
    int n_stages() const { return (int) stages.size(); }

    Ds41Dense & dense(int s) { return *stages[(size_t) s].dense; }
    Ds41MoeTier & tier(int s) { return *stages[(size_t) s].tier; }
    int64_t layer_lo(int s) const { return stages[(size_t) s].layer_lo; }
    int64_t layer_hi(int s) const { return stages[(size_t) s].layer_hi; }

    size_t handoff_bytes(int n) const;
};

std::vector<std::pair<int32_t, int32_t>> filter_ranked_layers(const std::vector<std::pair<int32_t, int32_t>> & ranked,
                                                            int64_t layer_lo, int64_t layer_hi);

/// `layer_split`: one int (e.g. "20") or comma list ("9,25") — must have one more stage device than split count.
/// `stage_pcie`: per-stage pcie_frac; empty = use mc_template.pcie_frac for every stage.
bool ds41_build_pipeline(const std::string & model, const std::string & layer_split, const std::vector<int> & split_devices,
                         int head_cuda_device, double vram_margin_gib, const std::vector<double> & stage_pcie,
                         Ds41DenseConfig dc_template, Ds41MoeConfig mc_template, Ds41FileBlobs & shared_blobs, Ds41Pipeline & out,
                         std::string & err);

/// One-time stderr lines for each stage boundary (and tail->head when offloaded).
void ds41_log_pipeline_handoffs(const Ds41Pipeline & pipe);
/// Round-trip a dummy column across every boundary; for STRATA_SELFTEST_HANDOFF=1 at load.
bool ds41_selftest_handoffs(Ds41Pipeline & pipe, std::string & err);

}  // namespace strata::v41
