// Vocab-only template/grammar support from the same pinned llama.cpp as the engine's ggml.
#pragma once
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef M31_HAVE_GRAMMAR
#include "llama.h"
#include "chat.h"
#include "sampling.h"
#include "reasoning-budget.h"
#include "json.hpp"
#include <climits>
#include <cmath>
namespace strata::m31 {
inline std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special, bool parse) {
    if (text.size() > INT_MAX) throw std::runtime_error("text too large");
    int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, special, parse);
    if (n > 0) throw std::runtime_error("invalid tokenizer length");
    std::vector<llama_token> ids(size_t(-n), 0);
    n = llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), int(ids.size()), special, parse);
    if (n < 0) throw std::runtime_error("tokenize failed");
    ids.resize(size_t(n));
    return ids;
}
class GrammarSampler {
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
        if (!grammar) throw std::runtime_error("tool grammar initialization failed");
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
inline std::string decode_chat_base64(const std::string & value) {
    std::string out;
    uint32_t bits = 0;
    int count = 0;
    bool padded = false;
    for (const auto c : value) {
        if (c == '=') { padded = true; continue; }
        if (padded) throw std::runtime_error("invalid chat_b64 padding");
        int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 :
                c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1;
        if (v < 0) throw std::runtime_error("invalid chat_b64");
        bits = (bits << 6) | uint32_t(v);
        count += 6;
        if (count >= 8) { count -= 8; out.push_back(char((bits >> count) & 255)); }
    }
    if (count == 6 || (count && (bits & ((1u << count) - 1)))) throw std::runtime_error("invalid chat_b64 tail");
    return out;
}
class ChatGrammar {
    using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    Model model_{nullptr, llama_model_free};
    common_chat_templates_ptr templates_;
    GrammarSampler grammar_;
    const llama_vocab * vocab_ = nullptr;
public:
    explicit ChatGrammar(const std::string & path) {
        // Uses the existing ggml registry; vocab_only never places model weights on a device.
        llama_backend_init();
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_.reset(llama_model_load_from_file(path.c_str(), params));
        if (!model_) throw std::runtime_error("cannot load grammar vocabulary");
        vocab_ = llama_model_get_vocab(model_.get());
        templates_ = common_chat_templates_init(model_.get(), "");
    }
    int vocab_size() const { return llama_vocab_n_tokens(vocab_); }
    std::set<int> stops() const {
        std::set<int> ids;
        for (int i = 0; i < vocab_size(); ++i) if (llama_vocab_is_eog(vocab_, i)) ids.insert(i);
        return ids;
    }
    std::set<int> suppress() const {
        int32_t count = 0;
        const auto * ids = llama_vocab_get_suppress_tokens(vocab_, &count);
        return count ? std::set<int>(ids, ids + count) : std::set<int>();
    }
    void prepare(const std::string & encoded, const std::vector<int32_t> & expected) {
        common_chat_params chat;
        if (encoded.empty()) { grammar_.init(vocab_, chat); return; }
        using json = nlohmann::ordered_json;
        const auto request = json::parse(decode_chat_base64(encoded));
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(request.at("messages").dump()));
        if (request.contains("tools")) inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(request["tools"].dump()));
        inputs.enable_thinking = request.value("enable_thinking", true);
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        inputs.parallel_tool_calls = request.value("parallel_tool_calls", true);
        inputs.add_generation_prompt = true;
        if (request.contains("json_schema")) inputs.json_schema = request["json_schema"].is_string()
            ? request["json_schema"].get<std::string>() : request["json_schema"].dump();
        if (request.contains("grammar"))
            throw std::runtime_error("raw GBNF grammar unavailable; use response_format or tool schemas");
        std::string choice = "auto";
        if (request.contains("tool_choice")) {
            if (!request["tool_choice"].is_string()) throw std::runtime_error("named tool_choice is not supported by the pinned grammar API");
            choice = request["tool_choice"].get<std::string>();
        }
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(choice);
        if (request.contains("chat_template_kwargs")) for (auto it = request["chat_template_kwargs"].begin(); it != request["chat_template_kwargs"].end(); ++it)
            inputs.chat_template_kwargs[it.key()] = it.value().dump();
        chat = common_chat_templates_apply(templates_.get(), inputs);
        const auto ids = tokenize(vocab_, chat.prompt.str(), false, true);
        if (ids.size() > expected.size() || !std::equal(ids.begin(), ids.end(), expected.begin()))
            throw std::runtime_error("chat template/tokenizer mismatch between server and native grammar");
        grammar_.init(vocab_, chat);
        // The common server may continue a turn after an injected reasoning
        // terminator or forced-tool prefix. Replay the already generated suffix
        // into a fresh request-local grammar instead of keeping mutable state
        // shared with a different HTTP request.
        if (grammar_.active()) for (size_t i = ids.size(); i < expected.size(); ++i) {
            if (!grammar_.allowed(expected[i])) throw std::runtime_error("assistant continuation violates grammar");
            grammar_.accept(expected[i]);
        }
    }
    bool active() const { return grammar_.active(); }
    bool allowed(int token) const { return grammar_.allowed(token); }
    std::set<int> allowed_ids() const {
        const auto ids = grammar_.allowed_ids();
        return std::set<int>(ids.begin(), ids.end());
    }
    void accept(int token) { grammar_.accept(token); }
};
} // namespace strata::m31
#else
namespace strata::m31 {
class ChatGrammar {
public:
    explicit ChatGrammar(const std::string &) {}
    int vocab_size() const { return 0; }
    std::set<int> stops() const { return {}; }
    std::set<int> suppress() const { return {}; }
    void prepare(const std::string & request, const std::vector<int32_t> &) {
        if (!request.empty()) throw std::runtime_error("grammar unavailable in this build");
    }
    bool active() const { return false; }
    bool allowed(int) const { return true; }
    std::set<int> allowed_ids() const { return {}; }
    void accept(int) {}
};
} // namespace strata::m31
#endif
