// tools/hy4/engine/hy4_moe.hpp - the routed-expert path the Hy4 drivers use: the reference CPU experts (Hy4RefMoe),
// or, in a build with -DHY4_TIER and HY4_TIER=1 at run time, the ds41 expert tier (Hy4TierMoe).
#pragma once

#include "hy4_moe_ref.hpp"
#ifdef HY4_TIER
#include "hy4_moe_tier.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::hy4 {

class Hy4Moe {
public:
    bool init(const GgufModel & model, const std::string & gguf, const Hy4Geometry & G, int n_threads, std::string & err) {
#ifdef HY4_TIER
        tier_on_ = std::getenv("HY4_TIER") && std::getenv("HY4_TIER")[0] == '1';
        if (tier_on_) return tier_.init(gguf, G, n_threads, err);
#else
        (void) gguf;
        if (std::getenv("HY4_TIER") && std::getenv("HY4_TIER")[0] == '1') {
            err = "HY4_TIER=1 needs a build with -DHY4_TIER";
            return false;
        }
#endif
        std::fprintf(stderr, "hy4_moe: reference CPU experts\n");
        return ref_.init(model, G, n_threads, err);
    }
    bool run(int il, const float * x, const int * ids, const float * wts, int n, float * out) {
#ifdef HY4_TIER
        if (tier_on_) return tier_.run(il, x, ids, wts, n, out);
#endif
        return ref_.run(il, x, ids, wts, n, out);
    }
    void report() const {
#ifdef HY4_TIER
        if (tier_on_) tier_.report();
#endif
    }

private:
    Hy4RefMoe ref_;
#ifdef HY4_TIER
    Hy4TierMoe tier_;
    bool tier_on_ = false;
#endif
};

}  // namespace strata::hy4
