// Standalone CPU validation of the wire plan, not a GPU/native numerical gate.
// c++ -std=c++17 -O2 tools/v41/engine/test_ds41_persistent_wire.cpp -o test_ds41_persistent_wire
#include "ds41_persistent_wire.hpp"
#include <cstdio>
#include <limits>
using namespace strata::v41::persistent_wire;
namespace {
int failures = 0, checks = 0;
void require(bool ok, const char * message) { ++checks; if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", message); } }
void write64(std::vector<uint8_t> & data, size_t at, uint64_t n) {
    for (unsigned i = 0; i < 8; ++i) data[at + i] = (uint8_t) (n >> (i * 8));
}
}
int main() {
    Region first; first.type = 1; first.ne = {4, 8, 1, 1}; first.nb = {4, 16, 128, 128}; first.bytes = 16;
    Region spare = first; spare.offset = 112; spare.bytes = 16;
    const std::vector<Region> regions{first, spare};
    std::vector<uint8_t> bytes;
    u64(bytes, magic); u64(bytes, 31); u64(bytes, 3); u64(bytes, 2); u64(bytes, 24); u64(bytes, regions.size());
    float values[27];
    for (unsigned i = 0; i < 27; ++i) values[i] = (float) (i + 1);
    const auto * p = reinterpret_cast<const uint8_t *>(values);
    bytes.insert(bytes.end(), p, p + sizeof(values));
    std::vector<size_t> descriptors;
    for (const auto & r : regions) {
        descriptors.push_back(bytes.size());
        u64(bytes, r.type);
        for (unsigned d = 0; d < 4; ++d) { u64(bytes, r.ne[d]); u64(bytes, r.nb[d]); }
        u64(bytes, r.offset); u64(bytes, r.bytes);
        for (unsigned i = 0; i < r.bytes; ++i) bytes.push_back((uint8_t) (r.offset + i));
    }
    std::string error; Plan plan;
    require(validate(bytes, 31, 3, 4, regions, plan, error), "round trip accepted");
    require(plan.pos == 31 && plan.seed_rows == 2 && plan.logits == 3 && plan.seed == 24, "metadata retained");
    require(plan.regions.size() == 2 && bytes[plan.regions[0]] == 0 && bytes[plan.regions[1]] == 112, "spare region uses its own payload offset");
    require(plan.logits_offset == 48 && plan.seed_offset == 60, "logits and three-tap draft seed kept distinct");
    auto short_seed = bytes; write64(short_seed, 32, 8);
    require(!validate(short_seed, 31, 3, 4, regions, plan, error), "single-tap seed rejected: DSpark requires three taps");
    for (size_t n = 0; n < bytes.size(); ++n) {
        auto truncated = bytes; truncated.resize(n);
        Plan sentinel; sentinel.pos = 999; sentinel.regions = {123};
        require(!validate(truncated, 31, 3, 4, regions, sentinel, error), "every truncation rejected");
        require(sentinel.pos == 999 && sentinel.regions == std::vector<size_t>{123}, "failure preserves output plan");
    }
    for (size_t i = 0; i < 6; ++i) {
        auto bad = bytes; write64(bad, i * 8, UINT64_MAX);
        require(!validate(bad, 31, 3, 4, regions, plan, error), "overflowing header rejected");
    }
    for (const auto at : descriptors) for (unsigned field = 0; field < 11; ++field) {
        auto bad = bytes; bad[at + field * 8] ^= 1;
        require(!validate(bad, 31, 3, 4, regions, plan, error), "layout mismatch rejected before native mutation");
    }
    std::vector<uint8_t> anchor(bytes.begin(), bytes.begin() + 48);
    write64(anchor, 16, 0); write64(anchor, 24, 0); write64(anchor, 32, 0);
    anchor.insert(anchor.end(), bytes.begin() + descriptors.front(), bytes.end());
    require(validate(anchor, 31, 0, 4, regions, plan, error) && !plan.logits && !plan.seed_rows,
            "explicit logits-free native replay anchor accepted");
    require(!validate(anchor, 31, 3, 4, regions, plan, error), "replay anchor cannot masquerade as complete regeneration state");
    auto trailing = bytes; trailing.push_back(0);
    require(!validate(trailing, 31, 3, 4, regions, plan, error), "trailing garbage rejected");
    require(!validate(bytes, 32, 3, 4, regions, plan, error), "wrong boundary rejected");
    require(!validate(bytes, 31, 4, 4, regions, plan, error), "wrong vocabulary rejected");
    require(!validate(bytes, 31, 3, 5, regions, plan, error), "wrong draft embedding rejected");
    require(!validate(bytes, 31, 3, UINT64_MAX, regions, plan, error), "overflowing embedding rejected");
    std::printf("DS41 persistent wire: %d checks, %d failures (CPU parser only)\n", checks, failures);
    return failures ? 1 : 0;
}
