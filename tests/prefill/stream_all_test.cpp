// The stream-all threshold priced from the link and the non-resident bytes (stream_all.hpp): no model, no GPU.
// Ported from jesdga95/basalt (MIT) PR #107; see docs/BASALT_PORTS.md.
#include <cstdio>

#include "strata/prefill/stream_all.hpp"

using strata::prefill::kStreamAllCap;
using strata::prefill::kStreamAllFloor;
using strata::prefill::stream_all_min_for;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // Unknown inputs: no price at all, so the caller keeps the constant threshold.  Not the floor - 0 is "leave it
    // alone", and a caller that read it as a chunk size would stream everything from token one.
    CHECK(stream_all_min_for(0, 25.0) == 0);
    CHECK(stream_all_min_for(60ull << 30, 0.0) == 0);
    CHECK(stream_all_min_for(0, 0.0) == 0);

    // A huge non-resident set on a fast link: streaming pays almost immediately, but the measured floor holds it up.
    CHECK(stream_all_min_for(60ull << 30, 400.0) == kStreamAllFloor);

    // The same set on a slow link: copying all of it takes long enough that the routed-only path wins for longer.
    const int64_t slow = stream_all_min_for(60ull << 30, 4.0);
    CHECK(slow > kStreamAllFloor);
    CHECK(slow <= kStreamAllCap);

    // Monotonic in the link: faster link -> never a larger threshold, never below the floor.
    int64_t prev = kStreamAllCap + 1;
    for (double gbps = 1.0; gbps <= 64.0; gbps *= 2.0) {
        const int64_t t = stream_all_min_for(40ull << 30, gbps);
        CHECK(t >= kStreamAllFloor && t <= kStreamAllCap);
        CHECK(t <= prev);
        prev = t;
    }

    // The result is always on the 256-token grid, so a chunk size is never split awkwardly.
    for (double gbps : {2.0, 6.0, 12.0, 25.0}) {
        const int64_t t = stream_all_min_for(30ull << 30, gbps);
        CHECK(t % 256 == 0);
    }

    // A tiny non-resident set is always the floor: there is nothing worth waiting for a big chunk to amortise.
    CHECK(stream_all_min_for(1ull << 20, 25.0) == kStreamAllFloor);

    if (failed == 0) std::printf("stream_all_test: ok\n");
    return failed == 0 ? 0 : 1;
}
