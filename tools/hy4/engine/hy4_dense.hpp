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
// NOT YET (P2b): the DSA layers' k-pool indexer.  It selects the top 511 pools of 4 + the current group, which is every
// earlier position while fewer than 512 groups exist, so attention is exact full-causal below position 2044 and the
// engine refuses positions past that (kpool_exact_limit()) instead of silently running a different model.
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
    /// Highest position (exclusive) at which the missing k-pool selection is still exact (all pools selected).
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
    /// cfg.taps: the named intermediates of the last layer call (name -> floats, ggml layout).
    const std::map<std::string, std::vector<float>> & taps() const;
    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::hy4
