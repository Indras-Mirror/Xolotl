#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ds41_sampling.hpp"
#include <cassert>
#include <limits>

// A grammar whose next token depends on the complete accepted prefix. This
// checks sampling/acceptance without a model or a second grammar implementation.
struct Grammar {
    bool enabled = true;
    std::vector<int> accepted;
    bool active() const { return enabled; }
    void accept(int token) { accepted.push_back(token); }
    void mask(float * values, const int * ids, int n) {
        for (int i = 0; i < n; ++i) {
            const bool allowed = accepted.empty() ? ids[i] == 1 || ids[i] == 2 : ids[i] == 3;
            if (!allowed) values[i] = -INFINITY;
        }
    }
};
int main() {
    float logits[] = {100.f, 0.f, 0.f, 99.f};
    Grammar grammar;
    std::mt19937_64 rng(17), reference(17);
    std::vector<std::pair<int, double>> dist;
    assert(grammar_dist(grammar, logits, 4, 1.f, 4, 1.f, 0.f, dist));
    assert(dist.size() == 2 && dist[0].second == .5 && dist[1].second == .5);
    // A forbidden deterministic proposal has probability zero, so rejection
    // samples the constrained target rather than leaking the high raw logit.
    int replacement = -1;
    assert(!spec_accept(dist, 0, rng, replacement));
    assert(replacement == 1 || replacement == 2);
    Grammar trial = grammar;
    trial.accept(replacement);
    assert(grammar_sample(trial, logits, 4, 0.f, 4, 1.f, 0.f, rng) == 3);
    assert(grammar.accepted.empty()); // trial cannot advance the committed parser
    grammar.accept(replacement);
    assert(grammar_sample(grammar, logits, 4, 1.f, 4, 1.f, 0.f, rng) == 3);
    Grammar independent;
    assert(grammar_sample(independent, logits, 4, 0.f, 4, 1.f, 0.f, rng) != 3);
    // Distribution preservation for allowed and forbidden point-mass drafts.
    for (int draft : {0, 1, 2, 3}) {
        int counts[4] = {};
        for (int i = 0; i < 40000; ++i) {
            int token = draft;
            if (spec_accept(dist, draft, rng, token)) token = draft;
            ++counts[token];
        }
        assert(counts[0] == 0 && counts[3] == 0);
        assert(std::abs(counts[1] - 20000) < 600);
    }
    // Absent grammar uses exactly the original sampler/RNG sequence.
    Grammar plain; plain.enabled = false;
    rng.seed(17); reference.seed(17);
    for (int i = 0; i < 100; ++i) {
        assert(grammar_sample(plain, logits, 4, .8f, 4, .9f, .01f, rng) ==
               sample_p(logits, 4, .8f, 4, .9f, .01f, reference));
    }
    // No legal finite candidate and numerical faults return a failure sentinel.
    float impossible[] = {0.f, -INFINITY, -INFINITY, 0.f};
    assert(grammar_sample(independent, impossible, 4, 0.f, 4, 1.f, 0.f, rng) == -1);
    // Raw numerical faults take priority even when the faulty token would be
    // excluded by the grammar. Negative infinity is a valid excluded candidate.
    for (float fault : {NAN, INFINITY}) {
        float invalid[] = {fault, 0.f, 0.f, 0.f};
        assert(grammar_sample(independent, invalid, 4, 0.f, 4, 1.f, 0.f, rng) == -2);
        assert(grammar_dist(independent, invalid, 4, 1.f, 4, 1.f, 0.f, dist) < 0);
    }
    float excluded[] = {-INFINITY, -INFINITY, 0.f, -INFINITY};
    assert(grammar_sample(independent, excluded, 4, 1.f, 4, 1.f, 0.f, rng) == 2);
    assert(grammar_dist(independent, excluded, 4, 1.f, 4, 1.f, 0.f, dist) == 1);
    assert(dist.size() == 1 && dist[0].first == 2 && dist[0].second == 1.);
}
