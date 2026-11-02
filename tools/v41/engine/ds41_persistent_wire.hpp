// Host-only fail-closed decoder for a DS41 prompt-boundary image.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::v41::persistent_wire {
constexpr uint64_t magic = 0x3154534950525344ull;
// Read an immutable native region into private capture bytes. The caller owns
// the bound slot; no disk worker receives a tensor or callback into live state.
template<class Cancel,class Read>
inline bool capture_region(std::vector<uint8_t>& out,size_t bytes,Cancel cancelled,Read read) {
    for(size_t copied=0;copied<bytes;) {
        if(cancelled())return false;
        const size_t chunk=std::min<size_t>(1<<20,bytes-copied),offset=out.size();
        if(chunk>out.max_size()-offset)throw std::length_error("persistent capture length");
        out.resize(offset+chunk);
        read(out.data()+offset,copied,chunk);
        copied+=chunk;
    }
    return !cancelled();
}
struct Region {
    uint64_t type = 0;
    std::array<uint64_t, 4> ne{}, nb{};
    uint64_t offset = 0, bytes = 0;
};
struct Plan {
    uint64_t pos = 0, seed_rows = 0, logits = 0, seed = 0;
    size_t logits_offset = 0, seed_offset = 0;
    std::vector<size_t> regions;
};
inline void u64(std::vector<uint8_t> & out, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) out.push_back((uint8_t) (v >> (i * 8)));
}
struct Reader {
    const std::vector<uint8_t> & bytes;
    size_t at = 0;
    bool u64(uint64_t & v) {
        if (at > bytes.size() || bytes.size() - at < 8) return false;
        v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= (uint64_t) bytes[at++] << (i * 8);
        return true;
    }
    bool skip(uint64_t count, size_t & offset) {
        if (at > bytes.size() || count > bytes.size() - at) return false;
        offset = at; at += (size_t) count; return true;
    }
};
// Produces an offset-only plan; failure never returns a partially validated plan.
// The native adapter must not write any tensor until this function succeeds.
inline bool validate(const std::vector<uint8_t> & bytes, uint64_t position, uint64_t logits,
                     uint64_t embedding, const std::vector<Region> & expected, Plan & out, std::string & error) {
    Reader reader{bytes};
    Plan plan;
    uint64_t value, count;
    if (!reader.u64(value) || value != magic || !reader.u64(plan.pos) || plan.pos != position || !position ||
        !reader.u64(plan.logits) || plan.logits != logits || !reader.u64(plan.seed_rows) || plan.seed_rows > 128 ||
        embedding > UINT64_MAX / (128 * 3) || !reader.u64(plan.seed) || plan.seed != plan.seed_rows * 3 * embedding ||
        !reader.u64(count) || count != expected.size() || expected.empty() ||
        plan.logits > bytes.size() / 4 || plan.seed > bytes.size() / 4) {
        error = "incompatible persistent boundary header"; return false;
    }
    if (!reader.skip(plan.logits * 4, plan.logits_offset) || !reader.skip(plan.seed * 4, plan.seed_offset)) {
        error = "truncated persistent boundary metadata"; return false;
    }
    plan.regions.reserve(expected.size());
    for (const auto & region : expected) {
        if (!reader.u64(value) || value != region.type) { error = "persistent tensor type mismatch"; return false; }
        for (unsigned d = 0; d < 4; ++d) {
            if (!reader.u64(value) || value != region.ne[d] || !reader.u64(value) || value != region.nb[d]) {
                error = "persistent tensor shape mismatch"; return false;
            }
        }
        if (!reader.u64(value) || value != region.offset || !reader.u64(value) || value != region.bytes) {
            error = "persistent tensor region mismatch"; return false;
        }
        size_t offset;
        if (!reader.skip(region.bytes, offset)) { error = "truncated persistent tensor region"; return false; }
        plan.regions.push_back(offset);
    }
    if (reader.at != bytes.size()) { error = "persistent boundary has trailing bytes"; return false; }
    out = std::move(plan);
    return true;
}
} // namespace strata::v41::persistent_wire
