// DeepSeek's existing host sampler and deterministic-proposal acceptance.
// Grammar masks are request-local and applied before the sampling chain.
#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include "strata/program/grammar_row.hpp"

inline int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

// the distribution sample_p draws from, as (token, probability) pairs summing to 1 (temp 0: the argmax alone) - the
// trunk's p for --serve's speculative sampling (same candidates, same weights as sample_p)
inline void sample_dist(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p,
                 std::vector<std::pair<int, double>> & out) {
    out.clear();
    if (temp <= 0.0f) { out.push_back({ (int) (std::max_element(logits, logits + n_vocab) - logits), 1.0 }); return; }
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    for (int i = 0; i < n; ++i) out.push_back({ idx[(size_t) i], p[(size_t) i] / z });
}

// Grammar is applied before truncation. Keep the original sampler and RNG calls
// untouched when the request has no grammar.
// Internal mask status: positive candidate count, zero starvation, -1 raw numerical fault.
template<class Grammar>
int grammar_logits(Grammar & grammar, const float * logits, int nv, std::vector<float> & masked) {
    masked.assign(logits, logits + nv);
    std::vector<int> ids((size_t) nv);
    for (int i = 0; i < nv; ++i) ids[(size_t) i] = i;
    const auto state = strata::program::mask_grammar_row(masked.data(), masked.size(),
        [&](float * values, size_t count) { grammar.mask(values, ids.data(), (int) count); });
    if (state == strata::program::GrammarRowStatus::invalid_logits) return -1;
    if (state == strata::program::GrammarRowStatus::starved) return 0;
    return (int) std::count_if(masked.begin(), masked.end(), [](float x) { return std::isfinite(x); });
}
// A sampled token, -1 grammar starvation, or -2 raw numerical fault.
template<class Grammar>
int grammar_sample(Grammar & grammar, const float * logits, int nv, float temp, int tk, float tp, float mp,
                   std::mt19937_64 & rng) {
    if (!grammar.active()) return sample_p(logits, nv, temp, tk, tp, mp, rng);
    std::vector<float> masked;
    const int count = grammar_logits(grammar, logits, nv, masked);
    if (count < 0) return -2;
    if (!count) return -1;
    return sample_p(masked.data(), nv, temp, std::min(count, tk > 0 ? tk : 256), tp, mp, rng);
}
// Distribution status: 1 ready, 0 grammar starvation, -1 raw numerical fault.
template<class Grammar>
int grammar_dist(Grammar & grammar, const float * logits, int nv, float temp, int tk, float tp, float mp,
                  std::vector<std::pair<int, double>> & dist) {
    if (!grammar.active()) { sample_dist(logits, nv, temp, tk, tp, mp, dist); return true; }
    std::vector<float> masked;
    const int count = grammar_logits(grammar, logits, nv, masked);
    if (count <= 0) return count;
    sample_dist(masked.data(), nv, temp, std::min(count, tk > 0 ? tk : 256), tp, mp, dist);
    return true;
}

// Speculative sampling with a DETERMINISTIC draft d (DSpark drafts by argmax: q = a point mass on d).  Leviathan /
// Chen: accept with min(1, p(d) / q(d)) = p(d); on rejection sample from norm(max(0, p - q)) = p without d.  The
// emitted token is then distributed exactly as p - the same as plain sampling.  true = d accepted; false = `out` is
// the replacement token.
inline bool spec_accept(const std::vector<std::pair<int, double>> & dist, int d, std::mt19937_64 & rng, int & out) {
    double pd = 0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (pd > 0 && u(rng) < pd) return true;
    double r = u(rng) * (1.0 - pd);
    int last = -1;
    for (const auto & c : dist) {
        if (c.first == d) continue;
        last = c.first;
        if ((r -= c.second) <= 0) { out = c.first; return false; }
    }
    out = last >= 0 ? last : d;   // rounding: the last candidate (d only if it was the sole candidate - then pd was 1)
    return false;
}

