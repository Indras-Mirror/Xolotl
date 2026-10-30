// Request-local GBNF state. llama.cpp owns parsing, UTF-8 and grammar semantics.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef STRATA_Q36_GRAMMAR
#include "llama.h"
bool q36_grammar_state_key(const llama_sampler *, std::string &);
class Q36Grammar {
public:
    // An exact memo of llama.cpp decisions, keyed by its complete pushdown state.
    // Clones share immutable masks, never mutable sampler state. Bounds apply
    // per grammar; the vocabulary keeps at most eight grammars.
    struct Node {
        bool known = false;
        std::vector<uint64_t> bits;
        std::vector<int> ids;
    };
    struct Cache {
        std::map<std::string, std::unique_ptr<Node>> states;
        size_t bytes = 0;
        // Never accept tokens into this sampler. Every request owns a native clone.
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> initial{nullptr, llama_sampler_free};
        int eos = -1;
        uint64_t used = 0;
    };
    struct Caches {
        std::map<std::string, std::shared_ptr<Cache>> entries;
        uint64_t clock = 0, compiles = 0, hits = 0;
        const llama_vocab * owner = nullptr;
    };
private:
    using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    Sampler sampler{nullptr, llama_sampler_free};
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<Cache> cache;
    mutable Node * node = nullptr;
    mutable bool node_checked = false;
    int after = -1, eos = -1;
    bool waiting = false;
    Node * state_node() const {
        if (node_checked) return node;
        node_checked = true;
        std::string key;
        if (!cache || !q36_grammar_state_key(sampler.get(), key)) return nullptr;
        auto it = cache->states.find(key);
        if (it != cache->states.end()) node = it->second.get();
        else if (cache->states.size() < 8192) {
            auto fresh = std::make_unique<Node>();
            node = fresh.get();
            cache->states.emplace(std::move(key), std::move(fresh));
        }
        return node;
    }
public:
    Q36Grammar() = default;
    Q36Grammar(const Q36Grammar & other)
        : sampler(other.sampler ? llama_sampler_clone(other.sampler.get()) : nullptr, llama_sampler_free),
          vocab(other.vocab), cache(other.cache), node(other.node), node_checked(other.node_checked),
          after(other.after), eos(other.eos), waiting(other.waiting) {
        if (other.sampler && !sampler) throw std::runtime_error("cannot clone grammar");
    }
    void init(const llama_vocab * v, const std::string & path, int trigger, Caches & caches) {
        sampler.reset(); vocab = v; after = trigger; waiting = trigger >= 0;
        cache.reset(); node = nullptr; node_checked = false;
        if (path.empty()) {
            if (waiting) throw std::runtime_error("grammar_after requires grammar");
            return;
        }
        if (trigger < -1 || trigger >= llama_vocab_n_tokens(v)) throw std::runtime_error("invalid grammar_after");
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot read " + path);
        const std::string text((std::istreambuf_iterator<char>(file)), {});
        if (text.empty() || text.find('\0') != std::string::npos) throw std::runtime_error("empty grammar or NUL byte");
        if (caches.owner && caches.owner != v) throw std::runtime_error("grammar cache vocabulary changed");
        caches.owner = v;
        auto it = caches.entries.find(text);
        if (it == caches.entries.end()) {
            auto fresh = std::make_shared<Cache>();
            fresh->initial.reset(llama_sampler_init_grammar(v, text.c_str(), "root"));
            if (!fresh->initial) throw std::runtime_error("cannot compile GBNF (root rule required); see engine log");
            const char end[] = "<|im_end|>";
            llama_token token = -1;
            if (llama_tokenize(v, end, sizeof(end) - 1, &token, 1, false, true) != 1)
                throw std::runtime_error("missing im_end token");
            fresh->eos = token;
            if (caches.entries.size() >= 8) {
                auto oldest = std::min_element(caches.entries.begin(), caches.entries.end(), [](const auto & a, const auto & b) {
                    return a.second->used < b.second->used;
                });
                caches.entries.erase(oldest);   // live requests retain their shared cache and pristine sampler
            }
            it = caches.entries.emplace(text, std::move(fresh)).first;
            ++caches.compiles;
        } else {
            ++caches.hits;
        }
        cache = it->second;
        cache->used = ++caches.clock;
        eos = cache->eos;
        sampler.reset(llama_sampler_clone(cache->initial.get()));
        if (!sampler) throw std::runtime_error("cannot clone compiled grammar");
    }
    bool active() const { return bool(sampler); }
    int vocab_size() const { return vocab ? llama_vocab_n_tokens(vocab) : 0; }
    const std::vector<int> * allowed_ids() const {
        if (!sampler || waiting) return nullptr;
        Node * state = state_node();
        return state && state->known ? &state->ids : nullptr;
    }
    void mask(float * values, const int * ids, int n) const {
        if (!sampler || waiting) return;
        state_node();
        if (node && node->known) {
            for (int i = 0; i < n; ++i)
                if (!(node->bits[(size_t) ids[i] / 64] & (uint64_t(1) << (ids[i] % 64)))) values[i] = -INFINITY;
            return;
        }
        std::vector<llama_token_data> data;
        data.reserve(n);
        // Grammar decisions do not depend on logits. Zeroes also let a full
        // mask be memoized when the caller already has non-finite logits.
        for (int i = 0; i < n; ++i) data.push_back({ids[i], 0.f, 0.f});
        llama_token_data_array a{data.data(), data.size(), -1, false};
        llama_sampler_apply(sampler.get(), &a);
        bool full = node && n == vocab_size();
        if (full) for (int i = 0; i < n; ++i) if (ids[i] != i) { full = false; break; }
        std::vector<int> allowed;
        for (int i = 0; i < n; ++i) {
            const bool ok = std::isfinite(data[i].logit) && (!llama_vocab_is_eog(vocab, ids[i]) || ids[i] == eos);
            if (!ok) values[i] = -INFINITY;
            else if (full) allowed.push_back(ids[i]);
        }
        const size_t words = ((size_t) n + 63) / 64;
        const size_t bytes = words * sizeof(uint64_t) + allowed.capacity() * sizeof(int);
        if (full && cache->bytes + bytes <= (32u << 20)) {
            node->bits.assign(words, 0);
            for (int id : allowed) node->bits[(size_t) id / 64] |= uint64_t(1) << (id % 64);
            node->ids = std::move(allowed);
            node->known = true;
            cache->bytes += bytes;
        }
    }
    void accept(int token) {
        if (!sampler) return;
        if (waiting) { if (token == after) waiting = false; return; }
        llama_sampler_accept(sampler.get(), token);
        // Accepting EOG leaves llama.cpp's grammar state unchanged.
        if (token != eos) { node = nullptr; node_checked = false; }
    }
};
class Q36Vocabulary {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    Q36Grammar::Caches caches;
public:
    explicit Q36Vocabulary(const std::string & path) {
        llama_backend_init();
        auto p = llama_model_default_params();
        p.vocab_only = true; p.n_gpu_layers = 0; p.load_mode = LLAMA_LOAD_MODE_MMAP;
        model.reset(llama_model_load_from_file(path.c_str(), p));
        if (!model) throw std::runtime_error("cannot load grammar vocabulary");
    }
    void prepare(Q36Grammar & grammar, const std::string & path, int after) {
        grammar.init(llama_model_get_vocab(model.get()), path, after, caches);
    }
};
#else
class Q36Grammar {
public:
    bool active() const { return false; }
    int vocab_size() const { return 0; }
    const std::vector<int> * allowed_ids() const { return nullptr; }
    void mask(float *, const int *, int) const {}
    void accept(int) {}
};
class Q36Vocabulary {
public:
    explicit Q36Vocabulary(const std::string &) {}
    void prepare(Q36Grammar &, const std::string & path, int after) const {
        if (!path.empty() || after >= 0) throw std::runtime_error("grammar unavailable in this build");
    }
};
#endif
