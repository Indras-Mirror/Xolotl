// tools/v41/engine/ds41_window_policy.hpp - the serve pass's draft window: DSpark's own drafts, or a prompt-lookup copy?
//
// Off by default (STRATA_DSPARK_COST=1 turns it on).  A pass is priced by what it measured: DSpark's window pays for
// `mtp_rows` rows, a lookup copy for `lookup_k + 1`, and a round's wall time (verify + sample + the next DSpark step)
// is learned per window size and per context band (< 6K, < 24K, longer - a row costs more at 32K than at 1K).  The
// window with the best expected committed tokens per millisecond wins, by a margin, with a few probes of a size whose
// cost is still a guess.  The rule is Project Maya's MIT-licensed draft policy (include/strata/spec/draft_policy.hpp,
// the same one the 27B MTP path uses); this header is only the seam the serve loop calls, so the choice is testable
// without a model.  It chooses which drafts are verified - the output is unchanged, the verify width is all it moves.
#pragma once

#include "strata/spec/draft_policy.hpp"

#include <algorithm>

namespace strata::ds41 {

struct WindowInput {
    int mtp_rows = 1;       ///< DSpark's window as the confidence / depth / cumulative cuts left it (1 = a plain step)
    int lookup_k = 0;       ///< the lookup copy's length in drafts (0 = no copy proposed)
    int lookup_match = 0;   ///< how many tokens the copy's n-gram match held
};

struct WindowChoice {
    bool lookup = false;
    int drafts = 0;         ///< drafts to verify (0 = a plain step; DSpark's window is mtp_rows - 1)
};

/// `cap_drafts` = the widest window the pass may take (the engine's depth / context / output caps applied already).
inline WindowChoice choose_window(strata::spec::DraftPolicy& pol, const WindowInput& in, int cap_drafts) {
    WindowChoice c;
    const int mtp_rows = std::clamp(in.mtp_rows, 1, strata::spec::DraftPolicy::kMaxT);
    c.drafts = std::clamp(mtp_rows - 1, 0, cap_drafts);
    if (in.lookup_k <= 0) return c;
    const int lookup_k = std::min(in.lookup_k, cap_drafts);
    if (lookup_k <= 0) return c;
    const strata::spec::DraftPolicy::Pick p = pol.choose(mtp_rows, lookup_k, in.lookup_match);
    if (p.lookup) {
        c.lookup = true;
        c.drafts = std::clamp(p.t - 1, 1, lookup_k);
    }
    return c;
}

}  // namespace strata::ds41
