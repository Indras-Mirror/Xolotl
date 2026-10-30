// Vocab-only helper. Link against the audited PR's llama-common and llama archives,
// NEVER against Xolotl's ggml in the same process. No model inference or GPU access.
#include "llama.h"
#include "chat.h"
#include "json.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
static void send(const json & value) { std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush; }
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special, bool parse) {
    if (text.size() > INT_MAX) throw std::runtime_error("text too large");
    int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, special, parse);
    if (n > 0) throw std::runtime_error("invalid tokenizer length");
    std::vector<llama_token> ids(size_t(-n), 0);
    n = llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), int(ids.size()), special, parse);
    if (n < 0) throw std::runtime_error("tokenize failed");
    ids.resize(size_t(n));
    return ids;
}
static void check_ids(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    for (auto token : ids) if (token < 0 || token >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("token out of vocabulary");
}
#include "server_grammar.hpp"
int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: m31_tokenizer MODEL.gguf\n"); return 2; }
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    llama_backend_init();
    try {
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("cannot load vocabulary");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto templates = common_chat_templates_init(model.get(), "");
        common_chat_params chat_params;
        ServerGrammar grammar;
        common_chat_msg previous_message;
        bool have_chat = false;
        std::vector<int> eog;
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) if (llama_vocab_is_eog(vocab, i)) eog.push_back(i);
        int32_t n_suppress = 0;
        const auto * suppress_data = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        std::vector<int> suppress;
        if (n_suppress > 0) suppress.assign(suppress_data, suppress_data + n_suppress);
        send({{"event", "ready"}, {"vocab", llama_vocab_n_tokens(vocab)}, {"eog", eog}, {"suppress", suppress},
              {"add_bos", llama_vocab_get_add_bos(vocab)}, {"template", llama_model_chat_template(model.get(), nullptr)}});
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto req = json::parse(line);
                const auto op = req.at("op").get<std::string>();
                if (op == "tokenize") {
                    auto ids = tokenize(vocab, req.at("content").get<std::string>(), req.value("add_special", false), req.value("parse_special", false));
                    send({{"tokens", ids}});
                } else if (op == "detokenize") {
                    auto ids = req.at("tokens").get<std::vector<llama_token>>();
                    check_ids(vocab, ids);
                    const bool remove = req.value("remove_special", false), unparse = req.value("unparse_special", true);
                    int n = llama_detokenize(vocab, ids.data(), int(ids.size()), nullptr, 0, remove, unparse);
                    if (n > 0) throw std::runtime_error("invalid detokenizer size");
                    std::string text(size_t(-n), '\0');
                    n = llama_detokenize(vocab, ids.data(), int(ids.size()), text.data(), int(text.size()), remove, unparse);
                    if (n < 0) throw std::runtime_error("detokenize failed");
                    text.resize(size_t(n));
                    send({{"content", text}});
                } else if (op == "piece") {
                    const int token = req.at("token").get<int>();
                    check_ids(vocab, {token});
                    std::vector<char> bytes(256);
                    int n = llama_token_to_piece(vocab, token, bytes.data(), int(bytes.size()), 0, true);
                    if (n < 0) {
                        bytes.resize(size_t(-n));
                        n = llama_token_to_piece(vocab, token, bytes.data(), int(bytes.size()), 0, true);
                    }
                    if (n < 0) throw std::runtime_error("piece failed");
                    std::vector<unsigned> raw;
                    for (int i = 0; i < n; ++i) raw.push_back(static_cast<unsigned char>(bytes[size_t(i)]));
                    send({{"bytes", raw}});
                } else if (op == "chat") {
                    common_chat_templates_inputs inputs;
                    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(req.at("messages").dump()));
                    if (req.contains("tools")) inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(req["tools"].dump()));
                    inputs.enable_thinking = req.value("enable_thinking", true);
                    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
                    inputs.parallel_tool_calls = req.value("parallel_tool_calls", true);
                    inputs.add_generation_prompt = true;
                    const std::string choice = req.value("tool_choice", std::string("auto"));
                    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(choice);
                    if (req.contains("chat_template_kwargs")) for (auto it = req["chat_template_kwargs"].begin(); it != req["chat_template_kwargs"].end(); ++it)
                        inputs.chat_template_kwargs[it.key()] = it.value().dump();
                    chat_params = common_chat_templates_apply(templates.get(), inputs);
                    grammar.init(vocab, chat_params);
                    have_chat = true;
                    previous_message = {};
                    previous_message.role = "assistant";
                    const std::string prompt = chat_params.prompt.str();
                    auto ids = tokenize(vocab, prompt, false, true);
                    send({{"tokens", ids}, {"prompt", prompt}, {"generation_prompt", chat_params.generation_prompt},
                          {"additional_stops", chat_params.additional_stops}, {"format", common_chat_format_name(chat_params.format)},
                          {"grammar_active", grammar.active()}, {"grammar_lazy", chat_params.grammar_lazy}});
                } else if (op == "grammar_check") {
                    if (!have_chat || !grammar.active()) throw std::runtime_error("chat tool grammar is not active");
                    const auto token = req.at("token").get<int>();
                    check_ids(vocab, {token});
                    const bool accepted = grammar.allowed(token);
                    if (accepted) {
                        if (req.value("accept", true)) grammar.accept(token);
                        send({{"accepted", true}});
                    } else send({{"accepted", false}, {"allowed", grammar.allowed_ids()}});
                } else if (op == "parse") {
                    if (!have_chat) throw std::runtime_error("chat must precede parse");
                    common_chat_parser_params pp(chat_params);
                    // The convenience constructor copies format/prefix only, NOT the PEG arena.
                    // Match common_chat_session's explicit parser+token-aware input setup.
                    pp.parser = chat_params.parser;
                    pp.generation_prompt = common_chat_input_tokenize(vocab, chat_params.generation_prompt);
                    auto input = common_chat_input_tokenize(vocab, req.at("content").get<std::string>());
                    auto msg = common_chat_parse(input, req.value("partial", false), pp);
                    if (!msg.empty()) previous_message = std::move(msg);
                    previous_message.role = "assistant";
                    send({{"message", json::parse(previous_message.to_json_oaicompat().dump())}});
                } else throw std::runtime_error("unknown tokenizer operation");
            } catch (const std::exception & e) { send({{"error", e.what()}}); }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "m31_tokenizer: %s\n", e.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
