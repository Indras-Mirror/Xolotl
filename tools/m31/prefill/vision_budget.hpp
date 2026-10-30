// Account for a persistent external encoder without sharing its ggml ABI.
#pragma once
#include "plan.hpp"

namespace strata::m31 {
// Four independent native owners that the next plan already charges. Never
// credit resident weights/KV/expert slots, high-water statistics or vendor pools.
struct VisionPrefillMemory {
    uint64_t free_bytes = 0, total_bytes = 0;
    uint64_t dense_bytes = 0, routed_bytes = 0, hidden_bytes = 0, attention_bytes = 0;
    uint64_t owned_bytes() const {
        return prefill_sum(prefill_sum(dense_bytes, routed_bytes),
                           prefill_sum(hidden_bytes, attention_bytes));
    }
};
inline PrefillBudget cap_vision_prefill_budget(PrefillBudget budget, const VisionPrefillMemory & memory) {
    if (!memory.total_bytes || memory.free_bytes > memory.total_bytes)
        throw std::runtime_error("vision VRAM planning requires valid device memory counters");
    const auto owned = memory.owned_bytes();
    if (owned > memory.total_bytes - memory.free_bytes)
        throw std::runtime_error("vision VRAM scratch ownership exceeds device usage");
    // Adding these current allocations back prevents charging them twice when
    // plan_prefill includes retained dense/routed/hidden or replacement masks.
    budget.free_after_resident = std::min(budget.free_after_resident,
                                         prefill_sum(memory.free_bytes, owned));
    budget.retained_dense_bytes = memory.dense_bytes;
    budget.retained_routed_bytes = memory.routed_bytes;
    budget.retained_hidden_bytes = memory.hidden_bytes;
    return budget;
}
struct VisionPrefillPlan {
    PrefillPlan plan;
    PrefillBudget budget;
    VisionPrefillMemory memory;
    bool reclaimed = false;
};
// Only a refused proposal can release the previous chunk's scratch. This is
// before any new rows execute, never a retry after a partial forward. Callers
// must consume the previous head before asking for the next append's plan.
// Read again after release: internal CUDA pools may retain memory we cannot
// credit, and external users may have changed their allocations meanwhile.
template<class ReadMemory, class ReleaseScratch>
VisionPrefillPlan plan_vision_prefill(const Geometry & geometry, const PrefillBudget & initial,
                                     ReadMemory read_memory, ReleaseScratch release_scratch) {
    VisionPrefillPlan result;
    auto propose = [&] {
        result.memory = read_memory();
        result.budget = cap_vision_prefill_budget(initial, result.memory);
        result.plan = plan_prefill(geometry, result.budget);
    };
    propose();
    if (!result.plan.chunk && result.memory.owned_bytes()) {
        release_scratch(); // Failure propagates; no rows or alternate recipe ran.
        result.reclaimed = true;
        propose();
    }
    return result;
}
} // namespace strata::m31
