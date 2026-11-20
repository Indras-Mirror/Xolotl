// tools/hy4/engine/hy4_moe_tier.hpp - Hy4-flash routed experts on the ds41 expert tier (P3).
//
// The latent experts (blk.L.ffn_{gate,up,down}_exps, n_latent x n_ff_exp per expert, top-16 of 512) through
// strata::v41::Ds41MoeTier: VRAM slots, the RAM arena, the NVMe file tier, the CPU pool - the same tier XQ-quantised
// experts will load through.  Same interface as Hy4RefMoe, so hy4_check / hy4_generate switch with HY4_TIER=1.
// The tier's geometry is built by hand: n_embd = the LATENT width (ds41_moe_geom_from_gguf would read the residual
// width), per-layer types (layer 0 is dense: -1).  Needs ds41_moe.cpp built with -DDS41_MOE_MAX_K=16.
#pragma once

#include "hy4_geometry.hpp"

#include <memory>
#include <string>

namespace strata::hy4 {

class Hy4TierMoe {
public:
    struct Impl;
    Hy4TierMoe();
    ~Hy4TierMoe();
    /// gguf: the first shard.  Env: HY4_TIER_CPU=1 (CPU pool only), HY4_SLOTS (VRAM expert slots, default 600),
    /// HY4_ARENA_GIB (RAM arena, default 16), HY4_THREADS (pool workers).
    bool init(const std::string & gguf, const Hy4Geometry & G, int n_threads, std::string & err);
    /// x: n * n_latent (token-major), ids / wts: n * n_used.  out: n * n_latent, the weighted sum.
    bool run(int il, const float * x, const int * ids, const float * wts, int n, float * out);
    void report() const;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::hy4
