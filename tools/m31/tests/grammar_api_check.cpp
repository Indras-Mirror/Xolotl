// No model or backend is needed: validate the native schema/GBNF dependency.
#include "json-schema-to-grammar.h"
#include "common.h"
#include "llama.h"
#include "m31_grammar.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
        const auto schema = common_json::parse(R"({"type":"object","properties":{"tool":{"type":"object","properties":{"city":{"type":"string","enum":["東京","München"]}},"required":["city"],"additionalProperties":false}},"required":["tool"],"additionalProperties":false})");
        const auto grammar = json_schema_to_grammar(schema);
        using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
        Sampler parsed(llama_sampler_init_grammar(nullptr, grammar.c_str(), "root"), llama_sampler_free);
        if (!parsed) throw std::runtime_error("nested Unicode schema generated invalid GBNF");
        Sampler malformed(llama_sampler_init_grammar(nullptr, "root ::= [", "root"), llama_sampler_free);
        if (malformed) throw std::runtime_error("malformed GBNF was accepted");
        const auto escaped = regex_escape("<tool>(a|b).*");
        if (escaped == "<tool>(a|b).*") throw std::runtime_error("literal trigger was not escaped");
        using strata::m31::decode_chat_base64;
        if (decode_chat_base64("5p2x5Lqs") != "東京" ||
            decode_chat_base64("--__") != std::string("\xfb\xef\xff", 3))
            throw std::runtime_error("chat payload decoder changed Unicode or URL-safe bytes");
        for (const char * invalid : {"A", "AB", "!", "AA=A"}) {
            bool rejected = false;
            try { (void) decode_chat_base64(invalid); }
            catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error("malformed chat payload was accepted");
        }
        std::cout << "M31 native schema/GBNF API checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
