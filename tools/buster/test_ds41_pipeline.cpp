// CPU-only: layer-split parsing helpers and handoff byte geometry (no GPU).
#include "ds41_pipeline.hpp"

#include <cstdio>
#include <cstdlib>

static int fails = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++fails;
    }
}

int main() {
    using strata::v41::filter_ranked_layers;
    std::vector<std::pair<int32_t, int32_t>> ranked = {{0, 1}, {5, 2}, {19, 3}, {20, 4}, {25, 5}};
    auto lo = filter_ranked_layers(ranked, 0, 20);
    check(lo.size() == 3, "layers [0,20) filter");
    auto hi = filter_ranked_layers(ranked, 20, 40);
    check(hi.size() == 2, "layers [20,40) filter");
    strata::v41::Ds41Pipeline pipe;
    check(!pipe.enabled(), "empty pipeline not split");
    check(pipe.handoff_bytes(4) == 0, "no stages -> zero handoff");
    std::fprintf(stderr, "test_ds41_pipeline: %d failure(s)\n", fails);
    return fails == 0 ? 0 : 1;
}
