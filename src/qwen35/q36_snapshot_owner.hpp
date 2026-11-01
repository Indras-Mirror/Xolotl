#pragma once
// Snapshots are numerical state, not merely a token-prefix memo. An anchor may
// travel with an unchanged KV prefix through a checked-out conversation fork;
// an equal semantic hash from another conversation is not that certificate.
#include "q36_cache_hub.hpp"

namespace strata { namespace q36 { namespace snapshot_owner {
using Snapshot = cache::Snapshot;

inline bool ready_owned(const Snapshot* state) {
    return state && state->host && state->refs > 0 && !state->writing && state->pos > 0;
}

inline Snapshot* at(const std::vector<Snapshot*>& owned, int64_t pos, uint64_t hash) {
    for (auto* state : owned)
        if (ready_owned(state) && state->pos == pos && state->hash == hash) return state;
    return nullptr;
}

// prefix_hash[p] describes cells [0,p). The final prompt anchor is excluded:
// this engine still needs to evaluate its final row to obtain next-token logits.
inline Snapshot* best(const std::vector<Snapshot*>& owned, int64_t limit,
                      const std::vector<uint64_t>& prefix_hash) {
    Snapshot* result = nullptr;
    if (limit <= 0 || prefix_hash.size() < 2) return result;
    const size_t prompt_cells = prefix_hash.size() - 1;
    for (auto* state : owned)
        if (ready_owned(state) && state->pos <= limit && (uint64_t) state->pos < prompt_cells &&
            state->hash == prefix_hash[(size_t) state->pos] && (!result || state->pos > result->pos)) result = state;
    return result;
}

// Before replay, every anchor beyond the restored boundary becomes ineligible,
// even if replayed tokens are unchanged. A different PP geometry can produce
// different floating-point state. Return removed OWNED references to the caller
// for release under its existing metadata lock; never mutate another record.
inline std::vector<Snapshot*> prune(std::vector<Snapshot*>& owned, int64_t retained,
                                    const std::vector<uint64_t>& prefix_hash) {
    std::vector<Snapshot*> removed;
    size_t kept = 0;
    for (auto* state : owned) {
        const bool valid = ready_owned(state) && state->pos <= retained &&
            (uint64_t) state->pos < prefix_hash.size() && state->hash == prefix_hash[(size_t) state->pos];
        if (valid) owned[kept++] = state;
        else removed.push_back(state);
    }
    owned.resize(kept);
    return removed;
}

}}} // namespace strata::q36::snapshot_owner
