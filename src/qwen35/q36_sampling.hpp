// Host sampling shared by the CUDA engine and its CPU distribution checks.
#pragma once
#include "q36_grammar.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

/// The sampled path on a logits row's kTopK candidates (descending; see topk_rows): the distribution sample() draws
/// from - the penalties of the last last_n tokens of `hist`, the top_k cut, the temperature, then min_p / top_p - as
/// (id, probability) pairs.  Exact when every penalty lowers a logit (repeat >= 1, freq / present >= 0) and the cut
/// (top_k) plus the penalized tokens fit in the candidates: a token outside them cannot enter the cut then.  Returns
/// false when that does not hold (the caller uses the whole row).
inline bool cand_dist(const float* cv, const int* ci, int nc, bool whole, const Sampling& s, const std::vector<int32_t>& hist,
               std::vector<std::pair<int, double>>& out, const Q36Grammar* grammar = nullptr) {
    out.clear();
    const bool constrained = grammar && grammar->active();
    // Full rows are indexed by token ID. A memoized full-vocabulary mask can
    // represent a narrow state as its complete allowed-ID list: this is exact,
    // including tokens outside the device top-128, with no additional cutoff.
    if (constrained && whole && nc == grammar->vocab_size()) {
        if (const auto * allowed = grammar->allowed_ids(); allowed && allowed->size() < (size_t) nc && allowed->size() <= 4096) {
            if (allowed->empty()) throw std::runtime_error("grammar has no allowed finite-logit tokens");
            bool indexed = true;
            for (int i = 0; i < nc; ++i) if (ci[i] != i) { indexed = false; break; }
            if (indexed) {
                std::vector<float> values;
                values.reserve(allowed->size());
                for (int id : *allowed) values.push_back(cv[id]);
                return cand_dist(values.data(), allowed->data(), (int) allowed->size(), true, s, hist, out, grammar);
            }
        }
    }
    const int kk = std::max(1, s.top_k > 0 ? s.top_k : 64);
    std::map<int, int> cnt;
    const bool pen = s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f);
    if (pen) {
        if (!whole && (s.repeat < 1.f || s.freq < 0.f || s.present < 0.f)) return false;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
    }
    if (!constrained && !whole && kk + (int) cnt.size() > nc) return false;
    std::vector<float> masked;
    const float floor = cv[nc - 1];
    if (constrained) {
        masked.assign(cv, cv + nc);
        grammar->mask(masked.data(), ci, nc);
        cv = masked.data();
    }
    std::vector<std::pair<float, int>> c;
    c.reserve(nc);
    for (int i = 0; i < nc; ++i) {
        float v = cv[i];
        if (constrained && !std::isfinite(v)) continue;
        if (pen) {
            auto it = cnt.find(ci[i]);
            if (it != cnt.end()) {
                if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
                v -= (float) it->second * s.freq + s.present;
            }
        }
        c.emplace_back(v, ci[i]);
    }
    const int wanted = s.temperature <= 0.f ? 1 : kk;
    if (c.empty()) {
        if (!whole) return false;
        throw std::runtime_error("grammar has no allowed finite-logit tokens");
    }
    if (constrained && !whole && (int) c.size() < wanted) return false;
    const int cut = std::min(wanted, (int) c.size());   // a greedy request with penalties: their argmax
    std::partial_sort(c.begin(), c.begin() + cut, c.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    c.resize((size_t) cut);
    // Outside the raw top-128 every logit is <= floor. Lowering penalties cannot
    // promote it above this cut. Strict comparison also resolves boundary ties.
    if (constrained && !whole && c.back().first <= floor) return false;
    if (cut == 1) {
        out.emplace_back(c[0].second, 1.0);
        return true;
    }
    std::vector<double> p(c.size());
    const double mx = c[0].first;
    double sum = 0;
    for (size_t i = 0; i < c.size(); ++i) sum += (p[i] = std::exp((c[i].first - mx) / s.temperature));
    size_t keep = c.size();
    double cum = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    for (size_t i = 0; i < keep; ++i) out.emplace_back(c[i].second, p[i] / tot);
    return true;
}
/// A draw from (id, probability) pairs; `skip`: an id taken out (its mass removed, the rest renormalized).
inline int draw(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double tot = 0;
    for (const auto& [id, q] : d)
        if (id != skip) tot += q;
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    int last = -1;
    for (const auto& [id, q] : d) {
        if (id == skip) continue;
        last = id;
        if ((r -= q) <= 0) return id;
    }
    return last;
}

