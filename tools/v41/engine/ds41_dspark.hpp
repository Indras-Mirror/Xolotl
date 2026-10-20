// tools/v41/engine/ds41_dspark.hpp - DeepSeek-V4.1's DSpark draft stack (the checkpoint's mtp.0-2) on ggml.
//
// Convention (docs/ds4/V41_DSPARK_AUDIT.md, verified against SGLang / vLLM / DeepSeek's model.py):
//   main_hidden(p) = the HC-mean (mean over the hc streams) of the trunk residual ENTERING layers 37 / 38 / 39 at
//                    position p, concatenated 37|38|39 -> [3 * n_embd]
//   main_x         = main_norm(main_proj(main_hidden))                          (block 0's weights, shared by all)
//   window         = each block's OWN kv_norm(wkv(main_x)), RoPE'd at its position, FP8-rounded, for the last 128
//                    ACCEPTED positions (a ring per block)
//   draft at p     = block [bonus token (the trunk's token for p+1), 4 x noise id] at positions p+1..p+5; three
//                    Blocks (hc mixing, window attention over the ring + all 5 slots - non-causal, sinks -, a
//                    128-expert top-3 MoE + shared expert); head = trunk output.weight on hc_pre -> norm; then the
//                    Markov head adds, left to right, a logit bias from the token picked at the previous slot (slot 0:
//                    the bonus); confidence = proj(cat(pre-norm hidden, markov embed)) (a logit, not a probability)
// Output: 5 draft ids for positions p+2..p+6 and their 5 confidences.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

struct Ds41DSparkConfig {
    ggml_backend_t backend = nullptr;   ///< the backend to run on (the engine's); required
    std::string sidecar;                ///< V41-MTP-Vision.gguf (mtp.0-2.*)
    std::string main_gguf;              ///< the trunk GGUF (geometry, output.weight, token_embd)
    int n_used = 3;                     ///< DSpark experts per token (config dspark_n_activated_experts)
    int noise_id = 128799;              ///< config dspark_noise_token_id
    int block = 5;                      ///< config dspark_block_size
    int max_append = 8;                 ///< main_hidden rows one draft call may append to the window (k <= this)
    /// The trunk's output.weight already on `backend` (Ds41Dense::weight): shared instead of a second copy (-0.7 GB).
    /// Any type (a requantized dense head only moves acceptance - drafts are verified).  nullptr = load from main_gguf.
    ggml_tensor * shared_head = nullptr;
    int cuda_device = -1;   ///< CUDA ordinal for cudaSetDevice around graph compute (layer-split head GPU)
};

class Ds41DSpark {
public:
    struct Impl;
    Ds41DSpark();
    ~Ds41DSpark();
    bool init(const Ds41DSparkConfig & cfg, std::string & err);
    int64_t main_dim() const;           ///< 3 * n_embd
    /// Append k main_hidden rows (k * main_dim floats, positions pos[0..k)) to every block's window, then - when
    /// `bonus` >= 0 - draft the block after the LAST appended position: ids[0..block) = tokens for pos+2.., conf
    /// likewise.  `bonus` < 0: append only (seeding the window from a prompt).  Positions must be increasing and each
    /// appended position is "accepted" (it stays in the window until 128 later positions push it out).
    bool step(const float * main_hidden, const int * pos, int k, int bonus, int * ids, float * conf);
    /// Forget the window (a new conversation / a rewind past it).
    void reset();
    double last_ms() const;             ///< wall time of the last step's graph
    size_t weight_bytes() const;        ///< device bytes of the stack's weights
    const std::string & last_error() const;
private:
    std::unique_ptr<Impl> p_;
};
