// tools/hy4/engine/hy4_moe_tier.cpp - see hy4_moe_tier.hpp.
#include "hy4_moe_tier.hpp"

#include "ds41_moe.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::hy4 {

using strata::v41::Ds41GgufBlobs;
using strata::v41::Ds41MoeConfig;
using strata::v41::Ds41MoeGeom;
using strata::v41::Ds41MoeTier;

struct Hy4TierMoe::Impl {
    const Hy4Geometry * G = nullptr;
    std::unique_ptr<Ds41GgufBlobs> src;
    Ds41MoeTier tier;
};

Hy4TierMoe::Hy4TierMoe() : p_(new Impl) {}
Hy4TierMoe::~Hy4TierMoe() = default;

static double env_d(const char * k, double d) { const char * v = std::getenv(k); return v && *v ? std::atof(v) : d; }

bool Hy4TierMoe::init(const std::string & gguf, const Hy4Geometry & G, int n_threads, std::string & err) {
    Impl & im = *p_;
    im.G = &G;
    Ds41MoeGeom g;
    g.n_embd = G.n_latent;
    g.n_ff = G.n_ff_exp;
    g.n_layers = G.n_layer;
    g.n_experts = G.n_expert;
    g.top_k = G.n_used;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0 (the Q8_0 GGUF; XQ builds will read the types from the file)
    g.gu_type = g.d_type = kQ8_0;
    g.gate_types.assign((size_t) G.n_layer, -1);
    g.up_types.assign((size_t) G.n_layer, -1);
    g.down_types.assign((size_t) G.n_layer, -1);
    double lim = 0.0;
    for (int64_t l = 0; l < G.n_layer; ++l) {
        if (!G.moe(l)) continue;
        g.gate_types[(size_t) l] = g.up_types[(size_t) l] = g.down_types[(size_t) l] = kQ8_0;
        const double c = (size_t) l < G.swiglu_clamp.size() ? G.swiglu_clamp[(size_t) l] : 0.0;
        if (lim == 0.0) lim = c;
        if (c != lim) { err = "swiglu_clamp_exp differs between routed layers (the tier has one limit)"; return false; }
    }
    if (lim > 1e-6) g.swiglu_limit = (float) lim;
    im.src.reset(new Ds41GgufBlobs(g, gguf));
    if (!im.src->open(err)) return false;
    Ds41MoeConfig cfg;
    cfg.cpu_only = env_d("HY4_TIER_CPU", 0) != 0;
    cfg.threads = n_threads;
    cfg.slots = (int64_t) env_d("HY4_SLOTS", 600);
    cfg.arena_gib = cfg.max_arena_gib = env_d("HY4_ARENA_GIB", 16.0);
    cfg.pf_b = 0;
    if (!im.tier.init(g, im.src.get(), cfg, err)) return false;
    if (!cfg.cpu_only) {
        // no Hy4 routing profile yet: an unranked spread (expert-major, so every layer gets a share of the slots and
        // the arena); correct, only slower than a ranked seed
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int64_t e = 0; e < G.n_expert; ++e)
            for (int64_t l = 0; l < G.n_layer; ++l)
                if (G.moe(l)) ranked.emplace_back((int32_t) l, (int32_t) e);
        if (!im.tier.seed_from_ranked(ranked, err)) return false;
    }
    std::fprintf(stderr, "hy4_moe_tier: %s, %lld VRAM-resident, %lld in the arena (%.1f GiB), blob %lld B, top_k %lld\n",
                 im.tier.mode(), (long long) im.tier.resident(), (long long) im.tier.arena_experts(), im.tier.arena_gib(),
                 (long long) im.tier.blob_bytes(G.n_dense_lead), (long long) g.top_k);
    return true;
}

bool Hy4TierMoe::run(int il, const float * x, const int * ids, const float * wts, int n, float * out) {
    Impl & im = *p_;
    static_assert(sizeof(int) == sizeof(int32_t), "ids");
    if (n == 1) return im.tier.run(il, (const int32_t *) ids, wts, x, out);
    return im.tier.run_chunk(il, n, (const int32_t *) ids, wts, x, out);
}

void Hy4TierMoe::report() const {
    const auto & s = p_->tier.stats();
    std::fprintf(stderr, "hy4_moe_tier: hits %lld cpu %lld pcie %lld file %lld\n", (long long) s.hits, (long long) s.cpu,
                 (long long) s.pcie, (long long) s.file_tier);
}

}  // namespace strata::hy4
