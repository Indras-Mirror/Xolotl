// tools/v41/engine/ernie_dense.hpp - ERNIE-5.5-Flash (`ernie5_5`): the CPU dense path for HCA and MLA + DSA layers.
//
// This is the ERNIE analogue of ds41_dense's dense half, written as a plain F32 CPU runner rather than a ggml graph so
// that a mid-stack layer can be gated on its own.  It handles, per layer:
//
//   mHC pre/post (4 streams, 20 Sinkhorn iterations, ERNIE's 1/(sqrt(mean(x^2))+eps) pre-RMS)
//   HCA attention: q_lora -> q_b (64 heads x 512), per-head RMS, YaRN(160000, factor 16) on the last 64 dims,
//     one MQA K/V row, a 128-token raw window plus the per-layer gated 128-pool compressor with the APE on the gate,
//     attention sinks, inverse RoPE, the VHA head mix (I + V U^T folded at conversion)
//   MLA + DSA attention: q/kv low-rank projections, the absorbed latent K row [kv_c | k_pe | indexer k] kept per
//     position (F16), plain rope_theta RoPE on q_pe/k_pe, the token-level lightning indexer (64 heads x 128 dims,
//     LayerNorm + NEOX RoPE on its key, relu(q.k)*w scores, top-2048 of the history plus the forced 128-cell window),
//     sparse attention over the selected cells + the per-head sink, per-head V expansion (v_b), VHA, the sigmoid gate
//     and the full-rank wo
//   grouped low-rank output (wo_a 8 x 1024 / wo_b) with the sigmoid gate from the normalised q latent (HCA)
//   layer 0: the dense SiTU FFN; MoE layers: the latent MoE (split-feature router, 512 experts top-10 in a 2048-d
//     latent, per-expert exp_scale, SiTU-GLU, latent RMS norm + up projection, and the shared expert on the hidden)
//
// A pass is `n` consecutive tokens at positions pos0..pos0+n-1 (n = 1 is a decode step); the per-layer raw window and
// compressor state persist between passes, so chunked prefill and decode both work.  `run()` records named
// intermediates in ggml order (ne[0] fastest) so they can be diffed against the llama.cpp PR's cpu dump.
//
// The geometry is the one artifact header the GGUF gate uses (include/strata/artifact/ernie_geometry.hpp): this
// runner reads its numbers through `strata::ernie_read_geometry`, so there is one ERNIE geometry in the tree.
#pragma once

#include "strata/artifact/ernie_geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

/// One named intermediate: `ne` is the ggml shape (ne[0] varies fastest), `data` is row-major in that order.
struct ErnieTap {
    std::string name;
    std::vector<int64_t> ne;   ///< up to 4 dims; missing dims are 1
    std::vector<float> data;
};

struct ErnieDenseConfig {
    int n_threads = 8;
    /// Zero the FFN of MoE layers (layers >= leading_dense_block_count) - reproduces the S3 gate, where the MoE was
    /// out of scope.  Default: run the latent MoE.
    bool stub_moe_ffn = false;
    /// Store the window K rows and the compressed rows at F16 precision - the reference's KV cache element type
    /// (llama.cpp's default F16).  The model math is otherwise F32; false keeps the cache F32.
    bool kv_fp16 = true;
};

class ErnieDense {
public:
    struct Impl;
    ErnieDense();
    ~ErnieDense();
    ErnieDense(const ErnieDense &) = delete;
    ErnieDense & operator=(const ErnieDense &) = delete;

    /// Open the GGUF, require `general.architecture == "ernie5_5"`, read the geometry (no tensor data).
    bool init(const std::string & gguf, const ErnieDenseConfig & cfg, std::string & err);
    const ErnieGeometry & geom() const;
    /// Dequantise the dense weights of layers [lo, hi) (and the token embedding) to F32.  Call once before a pass.
    /// The routed experts are NOT dequantised: they are read per used expert from their stored type at run time.
    bool prepare(int lo, int hi, std::string & err);

    /// Start a pass of `n` tokens at positions pos0..: the residual is the token embedding replicated over the 4
    /// mHC streams.  pos0 must equal the position after the last pass (0 for the first).
    bool begin(const int * tokens, int n, int pos0, std::string & err);
    /// Start a pass from a given residual ([n_embd, hc, n] ggml order) instead of the embeddings - the seed used to
    /// gate a mid-stack layer against the reference's `l_last-{il-1}`.  Advances the pass position like begin().
    bool begin_seeded(const float * x, int n, int pos0, std::string & err);

    /// Run layers [lo, hi) of the current pass.  `taps` (optional) receives the named intermediates.
    bool run(int lo, int hi, std::vector<ErnieTap> * taps, std::string & err);

    /// The residual after the last run, [n_embd, hc, n] ggml order.
    const std::vector<float> & residual() const;
    int pass_size() const;
    int pass_pos0() const;
    int64_t layer_lo() const, layer_hi() const;
    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
