// tools/v41/engine/test_ds41_window_policy.cpp - the serve pass's MTP/lookup window seam (ds41_window_policy.hpp).
//
// Simulated rounds with a measured-cost shape check that
//   1. with no lookup copy proposed the DSpark window is untouched;
//   2. a copy never widens the window past the engine's cap;
//   3. a size whose cost is still a guess is probed a few times when the copy's match is confident;
//   4. once a lookup round is measured expensive and the DSpark round cheap, the DSpark window is kept;
//   5. the other way round, the copy is taken;
//   6. the context bands keep their own cost tables (a row at 32K costs more than one at 1K).
#include "ds41_window_policy.hpp"

#include <cstdio>

using strata::ds41::WindowInput;
using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("ds41_window_policy_test\n");
    {
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        const WindowInput in{5, 0, 0};
        const auto c = strata::ds41::choose_window(p, in, 4);
        check(!c.lookup && c.drafts == 4, "no copy proposed: the DSpark window (4 drafts)");
    }
    {
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        const WindowInput in{6, 12, 40};
        const auto c = strata::ds41::choose_window(p, in, 3);
        check(c.drafts <= 3, "a copy is cut to the engine's cap (3 drafts)");
    }
    {
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        const auto c = strata::ds41::choose_window(p, WindowInput{5, 8, 40}, 7);
        check(c.lookup && c.drafts == 7, "an unmeasured size is probed on a confident copy (7 drafts)");
    }
    {
        // 4 drafts of DSpark at ~20 ms a round; a copy of 7 drafts at ~90 ms
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        for (int i = 0; i < 30; ++i) p.observe(false, 5, 3, 0, 20.0);
        for (int i = 0; i < 30; ++i) p.observe(true, 8, 6, 40, 90.0);
        const auto c = strata::ds41::choose_window(p, WindowInput{5, 8, 40}, 7);
        check(!c.lookup && c.drafts == 4, "a copy measured at 90 ms a round loses to DSpark at 20 ms");
    }
    {
        // the same window sizes, the costs the other way round
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        for (int i = 0; i < 30; ++i) p.observe(false, 5, 1, 0, 40.0);
        for (int i = 0; i < 30; ++i) p.observe(true, 8, 7, 40, 10.0);
        const auto c = strata::ds41::choose_window(p, WindowInput{5, 8, 40}, 7);
        check(c.lookup && c.drafts >= 5, "a copy measured at 10 ms a round is taken (>= 5 drafts)");
    }
    {
        // context bands: T2 at 12 ms below 6K, at 40 ms past 24K
        DraftPolicy p(DraftPolicy::kMaxT);
        p.set_robust(true);
        const double low[3] = {8.0, 12.0, 15.0};
        const double high[3] = {30.0, 40.0, 52.0};
        p.seed_costs(low, 3, 6.0, 0);
        p.seed_costs(high, 3, 6.0, 2);
        p.set_context(1000);
        const double a = p.cost_ms(2);
        p.set_context(32000);
        const double b = p.cost_ms(2);
        check(a < 20.0 && b > 30.0, "context bands: T2 costs 12 ms at 1K and 40 ms at 32K");
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
