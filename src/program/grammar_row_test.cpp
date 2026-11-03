// Exercise the production classification helper with explicit candidate masks.
// This tests finite/empty intersection and numerical-fault precedence, not a
// claim that a particular pinned llama.cpp grammar reaches an empty state.
#include "strata/program/grammar_row.hpp"
#include <cstdio>
#include <limits>

int main() {
    using strata::program::GrammarRowStatus;
    using strata::program::mask_grammar_row;
    const float inf = std::numeric_limits<float>::infinity();
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
        failures += !ok;
    };
    auto only_second = [&](float* values, size_t) { values[0] = -inf; };
    float excluded[] = {10.f, -inf}; // grammar excludes the only finite candidate
    check(mask_grammar_row(excluded, 2, only_second) == GrammarRowStatus::starved,
          "finite raw candidate with empty grammar intersection is request starvation");
    float valid[] = {10.f, 2.f};
    check(mask_grammar_row(valid, 2, only_second) == GrammarRowStatus::ready && valid[1] == 2.f,
          "an allowed finite candidate remains usable");
    bool called = false;
    auto mask_everything = [&](float* values, size_t count) {
        called = true;
        for (size_t i = 0; i < count; ++i) values[i] = -inf;
    };
    float nan[] = {10.f, std::numeric_limits<float>::quiet_NaN()};
    check(mask_grammar_row(nan, 2, mask_everything) == GrammarRowStatus::invalid_logits && !called,
          "NaN remains a numerical failure before any grammar decision");
    float positive_inf[] = {10.f, inf};
    check(mask_grammar_row(positive_inf, 2, mask_everything) == GrammarRowStatus::invalid_logits && !called,
          "positive infinity remains a numerical failure before any grammar decision");
    check(mask_grammar_row(nullptr, 2, mask_everything) == GrammarRowStatus::invalid_logits && !called,
          "invalid row is not reported as grammar starvation");
    return failures ? 1 : 0;
}
