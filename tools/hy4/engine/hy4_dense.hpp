// tools/hy4/engine/hy4_dense.hpp - Hy4-flash (`hy_v4_flash`): the dense (non-routed-expert) half.
//
// Math = llama.cpp PR #32517 `src/models/hy-v4-flash.cpp` (the reference, run on the same GGUF; its per-layer tensors
// dumped by ~/v41-work/hy4/tools/hy4_golden are this file's gate, tools/hy4/engine/hy4_check.cpp).
//
// One PASS = n consecutive tokens at positions pos0..pos0+n-1 (decode n = 1, prefill n = a chunk).  Per layer:
//   attn_router(il): [Engram add] -> hc_pre -> attn_norm -> MLA attention (absorbed; K row = kv_lora + rope, V = the
//                    latent; sinks; sliding-window mask on window layers) -> wv_b -> sigmoid gate -> wo ->
//                    post_attention_norm -> hc_post; then hc_pre -> ffn_norm and either the dense FFN (layer 0, the
//                    layer is then finished) or the router (sigmoid + bias top-k, normalised x scale), routed_down
//                    (the latent MoE input) and the shared expert
//   finish(il, routed_latent_sum): routed_up + shared -> post_ffw_norm -> hc_post
// then logits().  Every layer keeps its own latent K cache [kv_lora + rope, ctx_max].
//
// The DSA layers' k-pool indexer (P2b): the top 511 pools of 4 + the current group.  Below position 2048 every pool is
// selected (exact full-causal); past it the selection runs (kpool_maintain / kpool_score).  Checked 20 Nov against
// llama.cpp on a 3407-token prompt: layer-0 selection 99.9 % Jaccard, deeper layers 92-99.8 % (activation drift),
// prefill argmax match KL 5e-5, the 24 greedy tokens identical.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "hy4_geometry.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4 {

struct Hy4DenseConfig {
    ggml_backend_t backend = nullptr;   ///< required (CUDA or CPU)
    int     n_threads = 16;             ///< CPU backend threads
    int64_t ctx_max = 4096;             ///< positions the K caches hold
    int     max_pass = 512;             ///< tokens a pass may carry
    bool    taps = false;               ///< keep named intermediates (attn_out, ...) for the golden gate
};

class Hy4Dense {
public:
    struct Impl;
    Hy4Dense();
    ~Hy4Dense();
    Hy4Dense(const Hy4Dense &) = delete;
    Hy4Dense & operator=(const Hy4Dense &) = delete;

    bool init(const std::string & gguf, const Hy4DenseConfig & cfg, std::string & err);
    const Hy4Geometry & geom() const;
    const GgufModel & model() const;
    /// Highest position (exclusive) at which every pool is selected (the k-pool selection changes nothing below it).
    int64_t kpool_exact_limit() const;

    /// Start a pass: tokens tids[0..n) at positions pos0...  Embeddings, Engram rows (the hash reads the 2 tokens
    /// before each position from the position-indexed history).
    bool begin(const int * tids, int n, int pos0);
    /// Layer il (see the file comment).  For a MoE layer: ids / wts get n * n_used (token-major), *fn_host the n *
    /// n_latent routed-expert inputs (valid until the next call), and *dense = false.  For a dense-FFN layer the layer
    /// is finished here and *dense = true (call finish() only for MoE layers).
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, bool * dense);
    /// routed: n * n_latent floats, the routed experts' weighted sum in the latent space (host).
    bool finish(int il, const float * routed);
    /// Logits of pass rows row0..row0+nrows-1 after the last layer: nrows * vocab floats.
    bool logits(int row0, int nrows, float * out);

    /// The hc state after the last attn_router (dense layer) / finish: n * hc * n_embd floats ([n_embd, hc, n]).
    bool read_state(std::vector<float> & x);
    /// Overwrite the hc state (n * hc * n_embd floats) - the golden gate's teacher forcing.
    bool write_state(const float * x);
    /// cfg.taps: the named intermediates of the last layer call (name -> floats, ggml layout).
    const std::map<std::string, std::vector<float>> & taps() const;
    const std::string & last_error() const;

    // ---- LoRA (docs/train/LORA_PLAN.md).  Adapters on 2-D dense weights by GGUF tensor name: W x -> W x + s B (A x),
    // s = alpha / rank, A [in, rank] small uniform, B [rank, out] zero (an untrained adapter changes nothing).  Every
    // graph (inference and training) applies them.  Call once, after init.
    bool lora_add(const std::vector<std::string> & names, int rank, float alpha, uint64_t seed, std::string & err);
    int  lora_count() const;
    const std::string & lora_name(int i) const;
    void lora_shape(int i, int64_t & n_in, int64_t & n_out, int64_t & rank) const;
    int  lora_layer(int i) const;   ///< blk.N -> N, -1 for a non-layer weight
    void lora_get(int i, std::vector<float> & A, std::vector<float> & B) const;
    void lora_set(int i, const float * A, const float * B);

    // ---- training (LoRA).  The sequence = the last begin() (positions 0..n-1, n <= kpool_exact_limit(): full causal,
    // sliding window on window layers, every DSA pool selected).
    /// Layer il as ONE differentiable graph, the routed experts inside it (their GGUF slices uploaded for the step):
    /// x_in = the layer's input saved by the forward (n * hc * n_embd), g_out = d loss / d layer output, ids = the
    /// forward's routing (n * n_used).  Out: g_in = d loss / d x_in; gA / gB[i] += the gradient of every adapter of
    /// this layer (sized by the caller to the adapters' A / B).
    bool train_layer(int il, const float * x_in, const float * g_out, const int * ids, float * g_in,
                     std::vector<std::vector<float>> & gA, std::vector<std::vector<float>> & gB,
                     const float * expect_out = nullptr, double * out_rel_err = nullptr);
    ///   expect_out (optional): the forward's output of this layer; *out_rel_err = the training graph's output vs it
    /// The head: x_last (the state after the last layer), targets[t] = the token expected at t + 1 (-1 = no loss there).
    /// Out: the mean cross-entropy over the counted positions and g_x = d loss / d x_last.
    bool train_head(const float * x_last, const int * targets, float * loss, float * g_x,
                    std::vector<std::vector<float>> & gA, std::vector<std::vector<float>> & gB);

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::hy4
