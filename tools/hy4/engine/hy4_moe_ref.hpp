// tools/hy4/engine/hy4_moe_ref.hpp - Hy4-flash routed experts, the plain reference path (P2 correctness only).
//
// The latent experts (ffn_{gate,up,down}_exps, [n_latent, n_ff_exp, n_expert]) are bound in place in the GGUF mmap and
// run on ggml-cpu: up/gate = mul_mat_id, swiglu_clamp (routed only, swiglu_clamp_exp), down = mul_mat_id, weighted sum.
// ref: build_moe_ffn (llama-graph.cpp).  P3 replaces this with the ds41 expert tier (RAM / NVMe tiers, prefetch).
#pragma once

#include "hy4_geometry.hpp"

#include <memory>
#include <string>

namespace strata::hy4 {

class Hy4RefMoe {
public:
    struct Impl;
    Hy4RefMoe();
    ~Hy4RefMoe();
    bool init(const GgufModel & model, const Hy4Geometry & G, int n_threads, std::string & err);
    /// x: n * n_latent (token-major), ids / wts: n * n_used.  out: n * n_latent, the weighted sum.
    bool run(int il, const float * x, const int * ids, const float * wts, int n, float * out);

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::hy4
