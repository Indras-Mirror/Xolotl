// tools/v41/engine/ds41_dfly2.hpp - the DFly2 block drafter for DeepSeek-V4.1-Flash (AngelSpec dfly2-b6) on ggml.
//
// Math = AngelSpec's own forward (angelspec/models/dflash.py, draft/dflash.py, draft/dsv41_mla.py,
// draft/dsv41_attentive_correction.py; its offline harness ~/.quetza-data/dfly2/harness/accept.py measured tau 4.02 on
// our quantised trunk).  One draft, anchor a = the trunk's token for position a (not fed yet), a-1 = the last accepted
// position (its pass column is what the trunk state taps below come from):
//   block ids [x_a, MASK x5] at positions a..a+5, x = E(ids); slot 0 += gate * eh_proj(enorm(E(x_a)) :: hnorm(h)),
//   h = output_norm(hc_pre(trunk state at a-1)) (the MTP fusion)
//   three mHC blocks as DSpark's (latent MQA, sinks, 128-expert top-3 MoE + shared expert), block i reading target
//   layer L_i = 28 / 32 / 36's context: its 512 indexer picks for row a-1 (rows of its kv source's compressed KV,
//   layer 20) + its own 128-position window ring (a-128..a-1) + the block's own 6 latents (non-causal)
//   draft_hidden = final_norm(hc_pre(x)); then slot by slot j = 1..5 the attentive hidden correction:
//   cat = hidden_norm(dh_j) :: embed_norm(E(prev_j)) (prev_1 = x_a, prev_j = proposal_{j-1}), h'_j = dh_j +
//   swiglu MLP(cat) + attention(cat) over target layer 36's 640 context keys (+ a gated causal read of the block's own
//   correction keys 0..j), proposal_j = argmax(output.weight h'_j)
// RoPE everywhere = the trunk's compressed-layer YaRN (compress_rope_theta, factor 16).  No FP8 rounding of the
// drafter's own latents (trained in float).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

struct Ds41DFly2Config {
    ggml_backend_t backend = nullptr;   ///< the trunk dense half's backend (the drafter reads its caches in place)
    std::string sidecar;                ///< DFly2-B6-ds41.gguf (tools/v41/dfly2_sidecar.py)
    std::string main_gguf;              ///< the trunk GGUF (geometry, token_embd, and output head when not shared)
    ggml_tensor * shared_head = nullptr;       ///< the trunk's output.weight on `backend` (nullptr = load a copy)
    ggml_tensor * shared_out_norm = nullptr;   ///< the trunk's output_norm.weight on `backend` (nullptr = load a copy)
    int cuda_device = -1;
};

/// The trunk state one draft reads (after the pass that fed position a-1).  Device tensors live on cfg.backend.
struct Ds41DFly2Ctx {
    ggml_tensor * raw[3] = { nullptr, nullptr, nullptr };    ///< window K rings [DH, RAW] of target layers 28/32/36
    ggml_tensor * comp[3] = { nullptr, nullptr, nullptr };   ///< those layers' kv source's compressed KV [DH, rows]
    const int32_t * sel[3] = { nullptr, nullptr, nullptr };  ///< host: the indexer picks of row a-1 (compressed row ids)
    const float * selm[3] = { nullptr, nullptr, nullptr };   ///< host: 0 / -inf per pick
    int K[3] = { 0, 0, 0 };                                  ///< valid picks per layer (<= 512)
    const float * x = nullptr;     ///< host: the hc state of row a-1 after the last layer (HC * n_embd, [hc][d])
    const float * pre = nullptr;   ///< host: its pre mix (HC)
};

class Ds41DFly2 {
public:
    struct Impl;
    Ds41DFly2();
    ~Ds41DFly2();
    bool init(const Ds41DFly2Config & cfg, std::string & err);
    /// the target layers whose caches the drafter reads (28, 32, 36)
    const std::vector<int> & target_layers() const;
    int n_draft() const;   ///< block - 1 = 5
    /// Draft n_draft() tokens for positions a+1..: a = the anchor's position, `anchor` its token, `prev` the token at a-1.
    /// `depth` <= n_draft(): stop the correction unroll early (the remaining ids are not computed).
    /// prob (optional, n_draft() floats): each proposal's probability under the corrected head (0 past the unroll).
    bool draft(const Ds41DFly2Ctx & c, int a, int anchor, int prev, int depth, int * ids, float * prob = nullptr);
    double last_ms() const;
    double last_main_ms() const;   ///< the blocks + slot 1 graph of the last draft
    size_t weight_bytes() const;
    const std::string & last_error() const;
private:
    std::unique_ptr<Impl> p_;
};
