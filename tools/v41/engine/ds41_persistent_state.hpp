// Exact prompt-boundary image for the persistent prefix tier. The catalogue owns
// tokens/image identities and checksums; this adapter owns the native tensor layout.
#pragma once
#include "ds41_dense.hpp"
#include "ds41_persistent_wire.hpp"
#include "ds41_persistent_identity.hpp"
#include "ggml-backend.h"
#include "strata/core/persistent_prefix.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace strata::v41 {

inline std::vector<uint64_t> ds41_persistent_cells(const std::vector<int> & tokens,
                                                  const std::vector<uint64_t> & identities, size_t n) {
    std::vector<uint64_t> cells; cells.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        cells.push_back((uint32_t) tokens[i]);
        cells.push_back(i < identities.size() ? identities[i] : 0);
    }
    return cells;
}

struct Ds41PersistentBoundary {
    int64_t pos = 0;
    std::vector<float> logits, seed;
    int seed_rows = 0;
    bool replay = false; // logits-free source-KV anchor; caller must compute native ER suffix
};

namespace ds41_persistent_detail {
using persistent_wire::u64;
inline void append(std::vector<uint8_t> & out, const void * p, size_t n) {
    if (!n) return;
    const auto * b = static_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);
}
inline bool little_endian() { const uint32_t one = 1; return *(const uint8_t *) &one == 1; }
} // namespace ds41_persistent_detail

// Capture only while the owning slot is bound and the prompt snapshot is final.
// Disk workers receive these immutable bytes, never device pointers or live state.
inline bool ds41_capture_persistent(Ds41Dense & dense, const Ds41PersistentBoundary & boundary,
                                    std::vector<uint8_t> & out, std::string & error,
                                    const std::function<bool()> & cancelled = {}) {
    using namespace ds41_persistent_detail;
    auto stopped = [&] {
        if (cancelled && cancelled()) { error = "persistent capture cancelled"; return true; }
        return false;
    };
    if (stopped()) return false;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
    if (!little_endian() || boundary.pos < 1 || dense.snapshot_pos() != boundary.pos ||
        (!boundary.replay && boundary.logits.empty()) ||
        (boundary.replay && (!boundary.logits.empty() || boundary.seed_rows != 0 || !boundary.seed.empty())) ||
        boundary.seed_rows < 0 || boundary.seed_rows > 128 ||
        boundary.seed.size() != (size_t) boundary.seed_rows * 3 * (size_t) dense.geom().n_embd) {
        error = "persistent boundary is incomplete"; return false;
    }
    const auto regions = dense.state_regions(boundary.pos);
    if (regions.empty()) { error = "persistent boundary has no regions"; return false; }
    out.clear();
    u64(out, 0x3154534950525344ull); // DSRPIST1, adapter version independent of catalogue
    u64(out, (uint64_t) boundary.pos);
    u64(out, boundary.logits.size());
    u64(out, (uint64_t) boundary.seed_rows);
    u64(out, boundary.seed.size());
    u64(out, regions.size());
    append(out, boundary.logits.data(), boundary.logits.size() * sizeof(float));
    append(out, boundary.seed.data(), boundary.seed.size() * sizeof(float));
    if (stopped()) return false;
    ggml_backend_synchronize(dense.backend());
    for (const auto & region : regions) {
        if (stopped()) return false;
        u64(out, (uint64_t) region.t->type);
        for (unsigned d = 0; d < GGML_MAX_DIMS; ++d) {
            u64(out, (uint64_t) region.t->ne[d]); u64(out, region.t->nb[d]);
        }
        u64(out, region.off); u64(out, region.bytes);
        if (!persistent_wire::capture_region(out, region.bytes, stopped,
                [&](uint8_t * target, size_t offset, size_t chunk) {
                    ggml_backend_tensor_get(region.t, target, region.off + offset, chunk);
                })) return false;
    }
    return !stopped();
}

// All bytes, native layouts and metadata are checked before the first write.
// Cancellation is checked before this short native commit; it must not interrupt
// the commit and leave a partially restored slot available to another request.
inline bool ds41_restore_persistent(Ds41Dense & dense, const std::vector<uint8_t> & bytes,
                                    int64_t expected_pos, size_t expected_logits, bool replay,
                                    const std::function<bool()> & cancelled,
                                    Ds41PersistentBoundary & restored, std::string & error) {
    using namespace ds41_persistent_detail;
    if (!little_endian() || expected_pos < 1) { error = "incompatible persistent boundary platform"; return false; }
    const auto regions = dense.state_regions(expected_pos);
    std::vector<persistent_wire::Region> expected;
    expected.reserve(regions.size());
    static_assert(GGML_MAX_DIMS == 4, "persistent adapter dimension format");
    for (const auto & native : regions) {
        persistent_wire::Region region;
        region.type = (uint64_t) native.t->type;
        for (unsigned d = 0; d < 4; ++d) { region.ne[d] = (uint64_t) native.t->ne[d]; region.nb[d] = native.t->nb[d]; }
        region.offset = native.off; region.bytes = native.bytes;
        expected.push_back(region);
    }
    persistent_wire::Plan plan;
    if (!persistent_wire::validate(bytes, (uint64_t) expected_pos, replay ? 0 : expected_logits,
                                   (uint64_t) dense.geom().n_embd, expected, plan, error)) return false;
    if (replay && (plan.seed_rows != 0 || plan.seed != 0)) { error = "persistent replay anchor contains draft state"; return false; }
    Ds41PersistentBoundary result;
    result.replay = replay;
    result.pos = expected_pos; result.seed_rows = (int) plan.seed_rows;
    result.logits.resize((size_t) plan.logits); result.seed.resize((size_t) plan.seed);
    if (plan.logits) std::memcpy(result.logits.data(), bytes.data() + plan.logits_offset, (size_t) plan.logits * sizeof(float));
    if (plan.seed) std::memcpy(result.seed.data(), bytes.data() + plan.seed_offset, (size_t) plan.seed * sizeof(float));
    if (cancelled && cancelled()) { error = "persistent restore cancelled"; return false; }
    ggml_backend_synchronize(dense.backend());
    for (size_t i = 0; i < regions.size(); ++i)
        ggml_backend_tensor_set(regions[i].t, bytes.data() + plan.regions[i], regions[i].off, regions[i].bytes);
    dense.rearm((int) expected_pos);
    if (!dense.restore()) { error = "persistent native restore failed"; return false; }
    restored = std::move(result);
    return true;
}

} // namespace strata::v41
