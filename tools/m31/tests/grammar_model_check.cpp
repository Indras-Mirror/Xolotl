// Optional vocabulary-only integration check. No decoder state or model weights are loaded.
#include "m31_grammar.hpp"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: m31_grammar_model_check MODEL.gguf CASES.jsonl");
        strata::m31::ChatGrammar grammar(argv[1]);
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open grammar cases");
        std::string line;
        size_t count = 0;
        while (std::getline(input, line)) {
            using json = nlohmann::ordered_json;
            const auto test = json::parse(line);
            const auto name = test.at("name").get<std::string>();
            bool rejected = false;
            try {
                grammar.prepare(test.at("chat_b64").get<std::string>(), test.at("ids").get<std::vector<int32_t>>());
            } catch (const std::exception & error) {
                if (!test.value("expect_error", false)) throw std::runtime_error(name + ": " + error.what());
                rejected = true;
            }
            if (test.value("expect_error", false) != rejected) throw std::runtime_error(name + ": expected prepare rejection");
            if (!rejected) {
                if (grammar.active() != test.value("active", false)) throw std::runtime_error(name + ": grammar activation differs");
                if (test.value("reject_eog_before", false)) for (auto stop : grammar.stops())
                    if (grammar.allowed(stop)) throw std::runtime_error(name + ": incomplete constrained output allowed EOG");
                const auto suffix = test.value("accept_ids", std::vector<int32_t>());
                for (size_t i = 0; i < suffix.size(); ++i) {
                    const auto token = suffix[i];
                    if (token < 0 || token >= grammar.vocab_size() || !grammar.allowed(token))
                        throw std::runtime_error(name + ": valid teacher-forced output rejected at token " + std::to_string(i));
                    grammar.accept(token);
                }
                if (test.value("accept_eog_after", false)) {
                    bool accepts_stop = false;
                    for (auto stop : grammar.stops()) accepts_stop = accepts_stop || grammar.allowed(stop);
                    if (!accepts_stop) throw std::runtime_error(name + ": completed output does not allow EOG");
                }
            }
            std::cout << json{{"case", name}, {"ok", true}, {"rejected", rejected}}.dump() << '\n';
            ++count;
        }
        if (!count) throw std::runtime_error("no grammar cases");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "native grammar check: " << error.what() << '\n';
        return 1;
    }
}
