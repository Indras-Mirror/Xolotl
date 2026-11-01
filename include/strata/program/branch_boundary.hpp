#pragma once

#include <cstdint>

namespace strata::program {

// This is a NEW snapshot made after a real read, never permission to reuse KV
// by token equality alone. Small shared prefixes and tiny replay savings do not
// pay for a recurrent-state snapshot. The existing cache cap bounds retention.
inline int64_t branch_checkpoint_boundary(int64_t common, int64_t resume, int64_t prompt_tokens,
        int64_t min_prefix = 8192, int64_t min_replay = 64) {
    if (resume < 0 || min_prefix < 0 || min_replay <= 0 || prompt_tokens < 2 ||
        common < min_prefix || common <= resume || common >= prompt_tokens - 1 ||
        common - resume < min_replay) return -1;
    return common;
}

} // namespace strata::program
