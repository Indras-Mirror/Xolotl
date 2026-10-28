// Exact PR GBNF lazy sampler, isolated from Xolotl's ggml process.
#pragma once
#include "common.h"
#include "sampling.h"
#include "reasoning-budget.h"
#include <cmath>

class ServerGrammar {
    using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    Sampler grammar{nullptr, llama_sampler_free}, reasoning{nullptr, llama_sampler_free};
    const llama_vocab * vocab = nullptr;
    bool lazy = false;
    bool applies() const {
        if (!grammar) return false;
        if (!lazy || !reasoning) return true;
        const auto state = common_reasoning_budget_get_state(reasoning.get());
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
public:
    bool active() const { return bool(grammar); }
    void init(const llama_vocab * v, const common_chat_params & chat) {
        grammar.reset(); reasoning.reset(); vocab = v; lazy = chat.grammar_lazy;
        if (chat.grammar.empty()) return;
        common_params_sampling params;
        params.grammar_lazy = lazy;
        common_sampling_add_preserved_tokens(params, vocab, chat.preserved_tokens);
        common_sampling_add_grammar_triggers(params, vocab, chat.grammar_triggers);
        std::vector<std::string> patterns;
        std::vector<llama_token> tokens;
        for (const auto & t : params.grammar_triggers) {
            switch (t.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD: patterns.push_back(regex_escape(t.value)); break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: patterns.push_back(t.value); break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                    patterns.push_back(t.value.empty() ? "^$" : (t.value.front() == '^' ? "" : "^") + t.value + (t.value.back() == '$' ? "" : "$")); break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: tokens.push_back(t.token); break;
                default: throw std::runtime_error("unknown grammar trigger");
            }
        }
        std::vector<const char *> raw;
        for (const auto & p : patterns) raw.push_back(p.c_str());
        grammar.reset(lazy ? llama_sampler_init_grammar_lazy_patterns(vocab, chat.grammar.c_str(), "root", raw.data(), raw.size(), tokens.data(), tokens.size())
                           : llama_sampler_init_grammar(vocab, chat.grammar.c_str(), "root"));
        if (!grammar) throw std::runtime_error("PR tool grammar initialization failed");
        const auto prefill = tokenize(vocab, chat.generation_prompt, false, true);
        if (!lazy) for (auto t : prefill) llama_sampler_accept(grammar.get(), t);
        if (lazy && !chat.thinking_start_tag.empty() && !chat.thinking_end_tags.empty()) {
            std::vector<llama_tokens> ends;
            for (const auto & s : chat.thinking_end_tags) ends.push_back(tokenize(vocab, s, false, true));
            reasoning.reset(common_reasoning_budget_init(vocab, {tokenize(vocab, chat.thinking_start_tag, false, true)}, ends, {}, INT_MAX));
            for (auto t : prefill) llama_sampler_accept(reasoning.get(), t);
        }
    }
    bool allowed(llama_token token) const {
        if (!applies()) return true;
        llama_token_data item{token, 0.0f, 0.0f};
        llama_token_data_array a{&item, 1, -1, false};
        llama_sampler_apply(grammar.get(), &a);
        return std::isfinite(item.logit);
    }
    std::vector<int> allowed_ids() const {
        std::vector<llama_token_data> items;
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) items.push_back({i, 0.0f, 0.0f});
        llama_token_data_array a{items.data(), items.size(), -1, false};
        if (applies()) llama_sampler_apply(grammar.get(), &a);
        std::vector<int> result;
        for (const auto & t : items) if (std::isfinite(t.logit)) result.push_back(t.id);
        return result;
    }
    void accept(llama_token token) {
        const bool apply = applies();
        if (reasoning) {
            llama_sampler_accept(reasoning.get(), token);
            if (grammar && !apply && common_reasoning_budget_get_state(reasoning.get()) == REASONING_BUDGET_DONE) {
                const auto * end = common_reasoning_budget_get_end_match(reasoning.get());
                if (end) for (auto t : *end) llama_sampler_accept(grammar.get(), t);
            }
        }
        if (apply) llama_sampler_accept(grammar.get(), token);
    }
};
