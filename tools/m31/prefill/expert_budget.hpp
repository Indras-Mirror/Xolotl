#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::m31 {
// A portable default for the native streaming tier. Leave two thirds of the
// post-dense/KV free memory for graph scratch and vendor pools, keep at most the
// current and next complete layers, and never pretend a partial layer is enough
// for the default >=32-token threshold (4 routes can touch all 128 experts).
inline uint64_t choose_expert_budget(uint64_t free_after_dense_kv,
                                    uint64_t slot_stride, uint64_t experts) {
    if(!slot_stride||!experts||experts>UINT64_MAX/2||slot_stride>UINT64_MAX/(experts*2))
        throw std::invalid_argument("M31 automatic expert budget: invalid geometry or overflow");
    const uint64_t slots=std::min(free_after_dense_kv/3/slot_stride,experts*2);
    if(slots<experts)
        throw std::runtime_error("M31 automatic expert budget cannot fit one complete expert layer while reserving two thirds of free VRAM; reduce context or set explicit memory budgets");
    return slots*slot_stride;
}
} // namespace strata::m31
