// Classification of a target row after the request's native grammar mask.
// This is independent of CUDA and of the grammar parser. The caller first honors
// its window's existing raw-logit fault flags; -infinity here means an excluded
// sampling candidate, whereas NaN and positive infinity are numerical failures.
#pragma once
#include <cmath>
#include <cstddef>

namespace strata::program {
enum class GrammarRowStatus { ready, starved, invalid_logits };

template<class Mask>
GrammarRowStatus mask_grammar_row(float* values, std::size_t count, Mask&& mask) {
    if (!values || !count) return GrammarRowStatus::invalid_logits;
    for (std::size_t i = 0; i < count; ++i)
        if (std::isnan(values[i]) || (values[i] > 0 && !std::isfinite(values[i])))
            return GrammarRowStatus::invalid_logits;
    mask(values, count);
    for (std::size_t i = 0; i < count; ++i)
        if (std::isfinite(values[i])) return GrammarRowStatus::ready;
    return GrammarRowStatus::starved;
}
} // namespace strata::program
