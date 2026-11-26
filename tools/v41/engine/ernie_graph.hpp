// tools/v41/engine/ernie_graph.hpp - ERNIE-5.5-Flash (`ernie5_5`): the dense half as per-layer ggml graphs.
//
// The ERNIE analogue of ds41_dense.hpp: one graph per layer per pass on any ggml backend (CPU / CUDA), built once and
// reused, with the same call shape the DS4.1 driver uses with Ds41Dense:
//
//     begin(tids, n, pos0)
//     for il:  attn_router(il, ids, wts, &fn)        -> the tier (or a host loop) computes the routed sum
//              finish(il, routed_sum)
//     logits(row0, nrows, out)
//
// Per layer the graphs are (see ernie_graph.cpp):
//   attn_router: mHC pre-mixes (ERNIE's 1/(sqrt(mean x^2)+eps)), HCA attention (q-lora, per-head RMS, YaRN on the last
//                64 dims, the 128-token window ring, the per-layer 128-pool compressor with its APE, sinks, inverse
//                RoPE, the VHA head mix, the grouped low-rank output with the sigmoid gate), mHC post, the FFN mixes,
//                ffn_norm, then the router + shared expert (MoE layers) or the dense SiTU FFN (the leading dense block).
//                ids/wts = the top-`expert_used_count` experts; fn = the 2048-wide latent rows the routed experts
//                consume (fc1_latent_proj of ffn_norm), or the D-wide ffn_norm activation on a dense block.
//   finish:      latent_norm(routed sum) -> fc2_latent_proj -> + shared expert -> mHC post -> the residual state.
//   logits:      the head mixer (hc_head_fn), output_norm, output.weight and MultiMax.
//
// MLA+DSA layers (compress ratio 0) are refused at init with a precise message (slice S5).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ernie_geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

/// One named intermediate of a pass, in ggml order (ne[0] fastest) - the same shape/name convention the CPU runner
/// (ernie_dense) dumps, so test_ernie_graph's output diffs against the llama.cpp PR's dumps directly.
struct ErnieGraphTap {
    std::string name;
    std::vector<int64_t> ne;   ///< up to 4 dims; missing dims are 1
    std::vector<float> data;
};

class ErnieGraph;

struct ErnieGraphConfig {
    ggml_backend_t backend = nullptr;   ///< NULL = create and own a CPU backend
    int      n_threads = 8;
    int64_t  ctx_max = 65536;           ///< positions the window / compressor caches cover (compressed rows = ctx_max/128)
    int      max_pass = 512;            ///< tokens a pass may carry (the prefill chunk size)
    bool     kv_f16 = true;             ///< the window K rows and compressed rows at F16 (the reference's cache type)
    /// The MLA layers' cache rows ([kv_c | k_pe | idx_k], one per position: ~96 % of the KV at long context).
    /// GGML_TYPE_F16 (default, = kv_f16), GGML_TYPE_Q8_0 or GGML_TYPE_Q4_0.  A quantised cache is written as is and
    /// dequantised (to F16) over the cells a pass reads.  `mla_hadamard`: each row segment (kv_c, k_pe, idx_k) stored
    /// Hadamard-rotated (the queries rotated alike, the attention output rotated back): dot products are unchanged
    /// and the 4-bit blocks see no outlier channels.  Default: on for Q4_0, off otherwise.
    ggml_type mla_kv_type = GGML_TYPE_F16;
    int      mla_hadamard = -1;         ///< -1 = auto (Q4_0 only), 0 = off, 1 = on
    /// Optional directional ablation at load: a file of
    /// int32 n_layer, int32 n_embd, then float r[n_layer][n_embd] (unit vectors; a zero row = that layer untouched).
    /// Every matrix that writes the residual in layer l (attn_output_b / MLA attn_output, ffn_routed_up, ffn_down_shexp; ffn_down on
    /// the dense block) becomes W - abl_scale * r_l (r_l^T W) on its output side: dequantised, edited in F32,
    /// re-quantised to its own type.  The routed experts are untouched (they write through ffn_routed_up), and the
    /// mHC post / comb mixes are per-stream scalars and stream sums, so scale 1 removes r_l exactly from what each
    /// writer adds to the residual.
    std::string abl_file;
    float    abl_scale = 1.0f;
    /// DSpark DRAFT mode (ERNIE-5.5's speculative drafter, docs/ernie/ERNIE_DSPARK_SPEC.md): `gguf` is then the DSpark
    /// sidecar (`dspark.*` tensors, 3 stages) and this graph runs its stages instead of a model.  The geometry, the
    /// token embedding and output.weight come from this trunk (initialised first, same backend; it must outlive the
    /// draft).  The stages are HCA layers without the compressor whose window is the committed context written by
    /// dspark_commit(); a pass (begin) is one draft block, every query sees the window + ALL block rows (non-causal).
    const ErnieGraph * dspark_trunk = nullptr;
};

class ErnieGraph {
public:
    struct Impl;
    ErnieGraph();
    ~ErnieGraph();
    ErnieGraph(const ErnieGraph &) = delete;
    ErnieGraph & operator=(const ErnieGraph &) = delete;

    /// Open the GGUF, require `general.architecture == "ernie5_5"` and HCA layers, load the dense weights.
    bool init(const std::string & gguf, const ErnieGraphConfig & cfg, std::string & err);
    const ErnieGeometry & geom() const;
    int64_t n_layer() const;
    ggml_backend_t backend() const;

    /// Start a pass of n consecutive tokens at positions pos0.. (n <= max_pass; pos0 must follow the last pass).
    bool begin(const int * tids, int n, int pos0);

    /// Layer il of the current pass: attention + mHC + ffn_norm + router + shared expert (see the file comment).
    /// ids / wts: n * n_used (token-major); *fn_host: n * (expert_latent_length on a MoE layer, n_embd on the dense
    /// block) floats, valid until the next call.  `taps` (optional): the named intermediates of this graph.
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, std::vector<ErnieGraphTap> * taps = nullptr);
    /// routed_sum: n * latent_length floats (ignored, may be null, on the leading dense block), the tier's weighted
    /// sum of the selected experts' SiTU-GLU outputs in the latent space.
    bool finish(int il, const float * routed_sum, std::vector<ErnieGraphTap> * taps = nullptr);

    /// Guessed top kPredK experts of MoE layer il from the current residual (call after finish(il-1), before
    /// attn_router(il)): the LAST pass column's guess, for Ds41MoeTier::prefetch / spec_read.  False on a dense layer.
    static constexpr int kPredK = 16;
    bool predict(int il, int * top);

    /// Logits of pass columns row0..row0+nrows-1: nrows * n_vocab floats.
    bool logits(int row0, int nrows, float * out, std::vector<ErnieGraphTap> * taps = nullptr);

    /// The residual of pass columns [c0, c0+n): n * hc * n_embd floats, token-major.
    bool read_state(int c0, int n, std::vector<float> & x);
    /// The residual before the last attn_router's graph (the mHC pre mix) - gate tap.
    bool read_pre(std::vector<float> & pre);

    /// Conversation reuse: every cache is position-indexed except the window ring and the pending 128-block of
    /// compressor projections, so a snapshot is those two (~0.8 MB) + the position.  `slot` (0..kSnapSlots-1) keeps
    /// several at once (serve mode: position 0 and the last prompt's end).  A slot is only valid while the positions
    /// before it still hold the tokens it was taken over: restoring an EARLIER slot and reading past it invalidates it.
    static constexpr int kSnapSlots = 4;
    bool snapshot(int pos, int slot = 0);
    bool restore(int slot = 0);
    int snapshot_pos(int slot = 0) const;

    // ---- DSpark speculative decoding
    /// Draft mode only.  The window rows of COMMITTED positions [pos0, pos0 + k), k <= 128, from their main_hidden: k
    /// rows of 3 * n_embd floats, token-major, each the trunk's target layers' outputs (mHC-stream means) concatenated.
    /// Commits must come in position order (the ring keeps the last 128); begin(block, n, P) needs P == committed.
    bool dspark_commit(const float * main_hidden, int64_t pos0, int k);
    int64_t dspark_committed() const;
    /// Draft mode, after a block pass's stages (begin + attn_router/finish for every stage): the greedy drafts.  The
    /// head = hc_head, norm, the trunk's output.weight + MultiMax per slot, then the Markov chain from `anchor`:
    /// prev = anchor; draft[t] = argmax(base[t] + head . embed[prev]); prev = draft[t].  out: pass_size() ids.
    bool dspark_draft(int anchor, int * out);
    /// Trunk side.  Before a verify pass of n tokens at pos0: keep the cache rows that pass overwrites but rejected
    /// positions must not (the window ring and pending-block slots of positions pos0..pos0+n-1).  After the accept:
    /// spec_rollback(keep_end) restores them for positions [keep_end, pos0 + n) and sets the next position to keep_end
    /// (every other cache is position-indexed, or rewritten before it is read again).
    bool spec_save(int64_t pos0, int n);
    bool spec_rollback(int64_t keep_end);
    /// The mHC-stream mean of the residual of pass columns [c0, c0 + n) (after finish(il): layer il's output):
    /// n * n_embd floats, token-major.  DSpark's main_hidden rows (layers 40, 41, 42).
    bool read_state_mean(int c0, int n, float * out);

    int pass_size() const;
    int pass_pos0() const;
    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
