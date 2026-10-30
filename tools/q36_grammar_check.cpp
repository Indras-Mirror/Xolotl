// CPU-only checks against the real GGUF vocabulary and llama.cpp grammar API.
#include "q36_sampling.hpp"
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <numeric>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: q36-grammar-check MODEL SCENE_GBNF");
        Q36Vocabulary vocabulary(argv[1]);
        Q36Grammar grammar;
        vocabulary.prepare(grammar, argv[2], -1);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        require(bool(model), "vocabulary load failed");
        auto * vocab = llama_model_get_vocab(model.get());
        std::vector<int> ids(llama_vocab_n_tokens(vocab));
        std::iota(ids.begin(), ids.end(), 0);
        std::vector<float> logits(ids.size(), 0.f);
        auto tokenize = [&](const char * text) {
            std::vector<llama_token> result(256);
            int n = llama_tokenize(vocab, text, (int) std::strlen(text), result.data(), result.size(), false, true);
            require(n > 0, "tokenization failed");
            result.resize(n);
            return result;
        };
        Q36Grammar clone(grammar), text_state(grammar), choice_state(grammar);
        for (int id : tokenize("speak=false")) {
            float value = 0;
            clone.mask(&value, &id, 1);
            require(std::isfinite(value), "scene branch rejected");
            clone.accept(id);
        }
        for (int id : tokenize("speak=true;what=")) text_state.accept(id);
        for (int id : tokenize("speak=")) choice_state.accept(id);
        clone.mask(logits.data(), ids.data(), ids.size());
        int count = 0, eos = tokenize("<|im_end|>").front();
        for (size_t i = 0; i < ids.size(); ++i) if (std::isfinite(logits[i])) {
            require(ids[i] == eos, "non-EOS allowed after completed scene");
            ++count;
        }
        require(count == 1, "completed grammar must allow exactly im_end");
        float value = 0;
        grammar.mask(&value, &eos, 1);
        require(!std::isfinite(value), "clone mutated the original sampler");
        Q36Grammar lazy;
        int trigger = tokenize("</think>").front();
        vocabulary.prepare(lazy, argv[2], trigger);
        value = 0;
        lazy.mask(&value, &eos, 1);
        require(std::isfinite(value), "lazy grammar masked reasoning");
        lazy.accept(trigger);
        value = 0;
        lazy.mask(&value, &eos, 1);
        require(!std::isfinite(value), "lazy grammar did not activate");

        // Compare exact fast cuts and full-row fallback, including raising penalties.
        std::mt19937_64 rng(6541);
        std::vector<int> order(ids);
        int exact = 0, fallback = 0;
        for (int round = 0; round < 40; ++round) {
            for (float & x : logits) x = std::uniform_real_distribution<float>(-20, 20)(rng);
            std::partial_sort(order.begin(), order.begin() + 128, order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            std::vector<float> values;
            for (int i = 0; i < 128; ++i) values.push_back(logits[order[i]]);
            Sampling s;
            s.temperature = round % 3 == 0 ? 0.f : .9f;
            s.top_k = round % 2 ? 20 : 1;
            s.top_p = .9f;
            s.present = round % 4 == 0 ? -1.5f : 1.5f;
            s.last_n = 64;
            std::vector<int32_t> history{order[0], order[1], order[1]};
            auto * state = round % 2 ? &text_state : &clone;
            std::vector<std::pair<int, double>> fast, full;
            bool ok = cand_dist(values.data(), order.data(), 128, false, s, history, fast, state);
            cand_dist(logits.data(), ids.data(), ids.size(), true, s, history, full, state);
            if (ok) { require(fast == full, "fast grammar distribution differs from full row"); ++exact; }
            else ++fallback;
        }
        require(exact && fallback, "distribution checks must cover both paths");
        // Different free-text values must reuse the same native-state mask,
        // without merging states that have different future grammar rules.
        Q36Grammar text_a(text_state), text_b(text_state);
        for (int id : tokenize("alpha")) text_a.accept(id);
        for (int id : tokenize("beta")) text_b.accept(id);
        std::fill(logits.begin(), logits.end(), 0.f);
        text_a.mask(logits.data(), ids.data(), ids.size());
        require(text_a.allowed_ids() && text_a.allowed_ids() == text_b.allowed_ids(),
                "equivalent free-text states did not share their exact mask");
        // A cached narrow mask must preserve probabilities on permuted full
        // rows too; only identity-indexed rows can use the allowed-ID gather.
        std::vector<int> reversed(ids.rbegin(), ids.rend());
        for (int round = 0; round < 10; ++round) {
            for (float & x : logits) x = std::uniform_real_distribution<float>(-20, 20)(rng);
            Sampling s;
            s.temperature = .9f; s.top_p = .9f;
            std::vector<std::pair<int, double>> indexed, permuted;
            cand_dist(logits.data(), ids.data(), ids.size(), true, s, {}, indexed, &choice_state);
            std::vector<float> backwards(logits.rbegin(), logits.rend());
            cand_dist(backwards.data(), reversed.data(), reversed.size(), true, s, {}, permuted, &choice_state);
            require(indexed == permuted, "cached full mask depends on candidate order");
        }
        // A compiled-cache hit must clone pristine state, not a previous request's
        // accepted tokens; state-mask reuse must preserve the exact distribution.
        const std::string scene((std::istreambuf_iterator<char>(std::ifstream(argv[2]).rdbuf())), {});
        char scratch[] = "/tmp/q36-grammar-cache-XXXXXX";
        require(mkdtemp(scratch) != nullptr, "cannot create grammar fixture directory");
        std::vector<std::string> files, texts;
        for (int i = 0; i < 9; ++i) {
            files.push_back(std::string(scratch) + "/case-" + std::to_string(i) + ".gbnf");
            texts.push_back(scene + "\n# compiled-cache fixture " + std::to_string(i) + "\n");
            std::ofstream(files.back()) << texts.back();
        }
        Q36Grammar::Caches compiled;
        Q36Grammar original, restored;
        original.init(vocab, files[0], -1, compiled);
        std::vector<float> cold_mask(ids.size(), 0.f), warm_mask(ids.size(), 0.f);
        original.mask(cold_mask.data(), ids.data(), ids.size());
        for (int id : tokenize("speak=false")) original.accept(id);
        restored.init(vocab, files[0], -1, compiled);
        restored.mask(warm_mask.data(), ids.data(), ids.size());
        require(cold_mask == warm_mask && compiled.compiles == 1 && compiled.hits == 1,
                "compiled hit was not pristine or did not reuse compilation");
        value = 0;
        restored.mask(&value, &eos, 1);
        require(!std::isfinite(value), "compiled template acquired accepted tokens");
        Sampling cached_sampling;
        cached_sampling.temperature = .9f; cached_sampling.top_p = .9f;
        for (float & x : logits) x = std::uniform_real_distribution<float>(-20, 20)(rng);
        Q36Grammar::Caches uncached;
        Q36Grammar fresh;
        fresh.init(vocab, files[0], -1, uncached);
        std::vector<std::pair<int, double>> cold_dist, warm_dist;
        cand_dist(logits.data(), ids.data(), ids.size(), true, cached_sampling, {}, cold_dist, &fresh);
        cand_dist(logits.data(), ids.data(), ids.size(), true, cached_sampling, {}, warm_dist, &restored);
        require(cold_dist == warm_dist, "compiled-cache hit changed sampling distribution");
        for (int i = 1; i < 8; ++i) {
            Q36Grammar temp;
            temp.init(vocab, files[i], -1, compiled);
        }
        Q36Grammar touch;
        touch.init(vocab, files[0], -1, compiled); // keep oldest lexical key hot
        Q36Grammar ninth;
        ninth.init(vocab, files[8], -1, compiled);
        require(compiled.entries.size() == 8 && compiled.entries.count(texts[0]) == 1 &&
                compiled.entries.count(texts[1]) == 0, "compiled cache is not bounded LRU");
        // Evict the original cache entry while its live request still owns it.
        for (int i = 2; i < 9; ++i) {
            Q36Grammar temp;
            temp.init(vocab, files[i], -1, compiled);
        }
        Q36Grammar reinsert;
        reinsert.init(vocab, files[1], -1, compiled);
        require(compiled.entries.count(texts[0]) == 0, "fixture did not evict the live entry");
        value = 0;
        original.mask(&value, &eos, 1);
        require(std::isfinite(value), "LRU eviction invalidated a live request");
        value = 0;
        restored.mask(&value, &eos, 1);
        require(!std::isfinite(value), "LRU eviction changed a pristine clone");
        // Report actual CPU prepare cost, independently of first-mask construction.
        // Cache-hit timings include reading the grammar file; raw compile uses the
        // already-read string. This intentionally does not credit the cache for I/O.
        constexpr int repetitions = 100;
        const auto begin_compile = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> direct(
                llama_sampler_init_grammar(vocab, scene.c_str(), "root"), llama_sampler_free);
            require(bool(direct), "direct compile failed");
        }
        const auto begin_hits = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            Q36Grammar hit;
            hit.init(vocab, files[1], -1, compiled);
        }
        const auto finished_hits = std::chrono::steady_clock::now();
        std::cout << "compile_vs_cached_prepare_us: raw="
                  << std::chrono::duration<double, std::micro>(begin_hits - begin_compile).count() / repetitions
                  << " cached_with_file_read="
                  << std::chrono::duration<double, std::micro>(finished_hits - begin_hits).count() / repetitions
                  << " fixture_dir=" << scratch << '\n';
        std::cout << "Q36 grammar checks passed; exact=" << exact << " fallback=" << fallback << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
