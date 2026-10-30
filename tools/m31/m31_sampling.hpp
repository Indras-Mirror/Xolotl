// Sampling shared by the native HTTP worker and independent greedy regression probe.
#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
static int sample(const std::vector<float> & logits, double temperature, double top_p, int top_k, double min_p,
                  const std::set<int> & excluded, std::mt19937_64 & rng, const std::set<int> * allowed = nullptr) {
    if (temperature <= 0) {
        int best = -1;
        float value = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            // Validate every logit, including tokens outside the sampling support.
            if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            if (!excluded.count(int(i)) && (!allowed || allowed->count(int(i))) &&
                (best < 0 || logits[i] > value)) {
                best = int(i);
                value = logits[i];
            }
        }
        if (best < 0) throw std::runtime_error("empty sampling support");
        return best; // Strict > preserves the first (lowest ID) among equal maxima.
    }
    std::vector<std::pair<float, int>> values;
    values.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
        if (!excluded.count(int(i)) && (!allowed || allowed->count(int(i)))) values.emplace_back(logits[i], int(i));
    }
    if (values.empty()) throw std::runtime_error("empty sampling support");
    const size_t keep = top_k > 0 ? std::min(size_t(top_k), values.size()) : values.size();
    std::partial_sort(values.begin(), values.begin() + keep, values.end(), [](auto a, auto b) { return a.first > b.first; });
    values.resize(keep);
    // llama's usual top-k -> top-p -> min-p -> temperature order is kept explicit.
    // Probabilities for top-p/min-p use unscaled logits; temperature is applied last.
    std::vector<double> weights(keep);
    const double peak = values.front().first;
    double total = 0;
    for (size_t i = 0; i < keep; ++i) total += weights[i] = std::exp(double(values[i].first) - peak);
    size_t cutoff = 0;
    double cumulative = 0;
    do { cumulative += weights[cutoff++] / total; } while (cutoff < keep && cumulative < top_p);
    while (cutoff > 1 && weights[cutoff - 1] < min_p * weights.front()) --cutoff;
    weights.resize(cutoff);
    for (size_t i = 0; i < cutoff; ++i) weights[i] = std::exp((double(values[i].first) - peak) / temperature);
    return values[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)].second;
}
