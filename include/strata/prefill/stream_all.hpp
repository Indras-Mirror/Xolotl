// include/strata/prefill/stream_all.hpp - the chunk size from which a prompt chunk streams every non-resident expert.
//
// Ported from jesdga95/basalt (MIT), PR #107; see docs/BASALT_PORTS.md.  Below this size a chunk stages only the
// experts its tokens route to (and, with the CPU share armed, hands some to the idle decode pool); from it on every
// non-resident expert streams through the ring.
//
// THE ECONOMICS.  Streaming copies ALL the non-resident experts over the link whatever the chunk, so its cost is a
// constant in tokens.  The routed-only path costs extra per token instead.  The crossover is therefore
//
//     tokens_crossover = copy_ms / routed_extra_ms_per_token
//
// and it moves with the link: a card on a slow link pays more to stream them all, so it should stay on the
// routed-only path for longer.  Our threshold was a constant, tuned once on one card, and so was blind to this.
//
// WHAT THIS DOES NOT DO.  It never LOWERS the threshold below kStreamAllFloor: 1024 is this engine's measured floor
// (below ~1,000 tokens the output changed on a Q2_0 pack, because a smaller chunk takes other kernels - see the note
// at prefill.cpp's stream_all_min).  It only raises the threshold on a slow link.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::prefill {

/// This engine's measured floor: below ~1,000 tokens a smaller chunk changes the output on a Q2_0 pack, so the
/// threshold is never priced below it (prefill.cpp's stream_all_min note).
constexpr int64_t kStreamAllFloor = 1024;
/// The cap.  Basalt's default on a 5090; kept as the upper bound here.
constexpr int64_t kStreamAllCap = 6144;
/// The routed-only path's time per token beyond the streamed path's, in milliseconds.
///
/// UNCALIBRATED ON THIS HARDWARE.  The value is Basalt's, measured on their box.  Until it is re-measured here this
/// function is opt-in (STRATA_PREFILL_STREAM_PRICED=1) and the constant threshold is used.
constexpr double kRoutedExtraMs = 0.24;

/// The stream-all chunk size for `bytes` of non-resident experts over a host->device link of `gbps` GB/s: the
/// smallest chunk on the 256-token grid whose extra routed-only time outlasts copying them all, clamped to
/// [kStreamAllFloor, kStreamAllCap].  Returns 0 - "no price, keep the constant" - when either input is unknown, so a
/// caller can leave the existing threshold alone.
inline int64_t stream_all_min_for(uint64_t bytes, double gbps) {
    if (gbps <= 0.0 || bytes == 0) return 0;
    const double copy_ms = (double) bytes / (gbps * 1e6);   // GB/s over a byte count
    const double t = std::ceil(copy_ms / kRoutedExtraMs / 256.0) * 256.0;
    return (int64_t) std::min(std::max(t, (double) kStreamAllFloor), (double) kStreamAllCap);
}

}  // namespace strata::prefill
