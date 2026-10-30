// Scheduling only: preserves every original route, input row and tiled flag.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
struct PrefillRouteWorklist {
    std::vector<int32_t> route;       // scheduled item -> ORIGINAL route index
    std::vector<int32_t> histogram;   // FULL original batch, never a subset
    std::vector<int32_t> tiled;       // indexed by ORIGINAL route
    size_t original_expert_switches = 0, scheduled_expert_switches = 0;
};

inline PrefillRouteWorklist prefill_route_worklist(const std::vector<int32_t> & ids,
                                                   size_t slots, int used,
                                                   bool tiled_enabled = true,
                                                   bool tiled_forced = false) {
    if (slots == 0 || slots > 1048576 || used <= 0 || used > 4096 ||
        ids.empty() || ids.size() % size_t(used) || ids.size() > size_t(INT32_MAX))
        throw std::runtime_error("M31 route worklist: invalid slots/used/routes");
    PrefillRouteWorklist out;
    out.histogram.resize(slots);
    out.route.resize(ids.size());
    out.tiled.resize(ids.size());
    for (const auto id : ids) {
        if (id < 0 || size_t(id) >= slots)
            throw std::runtime_error("M31 route worklist: invalid expert slot");
        ++out.histogram[size_t(id)];
    }
    // Stable counting sort. Only scheduling changes; model IDs stay untouched.
    std::vector<size_t> offset(slots);
    size_t sum = 0;
    for (size_t slot = 0; slot < slots; ++slot) {
        offset[slot] = sum;
        sum += size_t(out.histogram[slot]);
    }
    for (size_t original = 0; original < ids.size(); ++original) {
        const size_t expert = size_t(ids[original]);
        out.route[offset[expert]++] = int32_t(original);
        out.tiled[original] = tiled_enabled && (tiled_forced || out.histogram[expert] >= 8);
        if (original && ids[original] != ids[original-1]) ++out.original_expert_switches;
    }
    for (size_t scheduled = 1; scheduled < out.route.size(); ++scheduled)
        if (ids[size_t(out.route[scheduled])] != ids[size_t(out.route[scheduled-1])])
            ++out.scheduled_expert_switches;
    return out;
}
} // namespace strata::m31
