// Strata's text engine protocol. Parsing and sampling do not depend on a model or a GPU.
#pragma once
#include "m31_sampling.hpp"
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

namespace strata::m31::protocol {
inline int64_t integer(const std::string & value, int64_t low, int64_t high, const char * name) {
    size_t end = 0;
    long long n;
    try { n = std::stoll(value, &end); } catch (...) { throw std::runtime_error(std::string("invalid ") + name); }
    if (end != value.size() || n < low || n > high) throw std::runtime_error(std::string("invalid ") + name);
    return n;
}
inline uint64_t unsigned_integer(const std::string & value, const char * name) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("invalid ") + name);
    size_t end = 0;
    unsigned long long n;
    try { n = std::stoull(value, &end); } catch (...) { throw std::runtime_error(std::string("invalid ") + name); }
    if (end != value.size()) throw std::runtime_error(std::string("invalid ") + name);
    return uint64_t(n);
}
inline double real(const std::string & value, double low, double high, const char * name) {
    size_t end = 0;
    double n;
    try { n = std::stod(value, &end); } catch (...) { throw std::runtime_error(std::string("invalid ") + name); }
    if (end != value.size() || !std::isfinite(n) || n < low || n > high) throw std::runtime_error(std::string("invalid ") + name);
    return n;
}
inline std::vector<std::string> split(const std::string & text, char sep) {
    std::vector<std::string> out;
    size_t pos = 0;
    do {
        const auto end = text.find(sep, pos);
        out.push_back(text.substr(pos, end == std::string::npos ? end : end - pos));
        if (end == std::string::npos) break;
        pos = end + 1;
    } while (true);
    return out;
}
inline std::vector<int32_t> token_ids(const std::string & text, int vocab, bool visual = false) {
    std::vector<int32_t> result;
    if (text.empty()) throw std::runtime_error("empty token list");
    for (const auto & value : split(text, ',')) result.push_back(int32_t(integer(value, visual ? -1 : 0, vocab - 1, "token ID")));
    return result;
}
struct Request {
    int slot = -1, max_new = 0;
    double temperature = 0, top_p = 1, min_p = 0;
    int top_k = 0, penalty_last_n = 64;
    double penalty_repeat = 1, penalty_freq = 0, penalty_present = 0;
    uint64_t seed = 0;
    bool seeded = false, cache_prompt = true, checkpoint = true, ignore_eos = false;
    std::string chat_b64, embeddings;
    bool vision = false;
    std::vector<int32_t> ids;
    std::map<int, double> bias;
    std::set<int> stop, allowed;
    bool masked = false;
};
inline Request parse_request(const std::string & line, int vocab, int64_t context, bool vision_available = false) {
    std::istringstream input(line);
    std::string command, word;
    input >> command;
    const bool vision = command == "GENI" || command == "BGENI";
    if (vision && !vision_available) throw std::runtime_error("vision unavailable in this build");
    if (command != "GEN" && command != "BGEN" && !vision) throw std::runtime_error("unsupported command");
    Request r;
    r.vision = vision;
    if (command == "BGEN" || command == "BGENI") {
        if (!(input >> word)) throw std::runtime_error("BGEN: missing slot");
        r.slot = int(integer(word, 0, 1023, "batch slot"));
    }
    if (!(input >> word)) throw std::runtime_error("missing max_new");
    // Zero is an empty generation; it never means unbounded decoding.
    r.max_new = int(integer(word, 0, std::numeric_limits<int>::max(), "max_new"));
    std::set<std::string> seen;
    bool have_ids = false;
    while (input >> word) {
        const auto eq = word.find('=');
        if (eq == std::string::npos) {
            if (vision && r.embeddings.empty()) { r.embeddings = word; continue; }
            if (have_ids) throw std::runtime_error("expected exactly one prompt token list");
            r.ids = token_ids(word, vocab, vision); have_ids = true;
            continue;
        }
        if (have_ids || !r.embeddings.empty()) throw std::runtime_error("sampling keys must precede prompt tokens/embeddings");
        const auto key = word.substr(0, eq), value = word.substr(eq + 1);
        if (!seen.insert(key).second) throw std::runtime_error("duplicate sampling key: " + key);
        if (key == "temperature") r.temperature = real(value, 0, 100, "temperature");
        else if (key == "top_p") r.top_p = real(value, std::numeric_limits<double>::min(), 1, "top_p");
        else if (key == "min_p") r.min_p = real(value, 0, 1, "min_p");
        else if (key == "top_k") r.top_k = int(integer(value, 0, std::numeric_limits<int>::max(), "top_k"));
        else if (key == "penalty_repeat") r.penalty_repeat = real(value, std::numeric_limits<double>::min(), 100, "penalty_repeat");
        else if (key == "penalty_freq") r.penalty_freq = real(value, -100, 100, "penalty_freq");
        else if (key == "penalty_present") r.penalty_present = real(value, -100, 100, "penalty_present");
        else if (key == "penalty_last_n") r.penalty_last_n = int(integer(value, 0, std::numeric_limits<int>::max(), "penalty_last_n"));
        else if (key == "seed") { r.seed = unsigned_integer(value, "seed"); r.seeded = true; }
        else if (key == "cache_prompt") r.cache_prompt = integer(value, 0, 1, "cache_prompt") != 0;
        else if (key == "ignore_eos") r.ignore_eos = integer(value, 0, 1, "ignore_eos") != 0;
        else if (key == "ckpt") r.checkpoint = integer(value, 0, 1, "ckpt") != 0;
        else if (key == "stop" || key == "allowed_ids") {
            const auto ids = token_ids(value, vocab);
            auto & dst = key == "stop" ? r.stop : r.allowed;
            dst.insert(ids.begin(), ids.end());
            if (key == "allowed_ids") r.masked = true;
        }
        else if (key == "logit_bias" || key == "bias") {
            for (const auto & entry : split(value, ',')) {
                const auto pair = split(entry, ':');
                if (pair.size() != 2) throw std::runtime_error("invalid logit_bias pair");
                const int id = int(integer(pair[0], 0, vocab - 1, "logit_bias token"));
                if (!r.bias.emplace(id, real(pair[1], -100, 100, "logit_bias")).second)
                    throw std::runtime_error("duplicate logit_bias token");
            }
        }
        else if (key == "chat_b64") r.chat_b64 = value;
        else if (key == "img_keys" && vision) {
            // The common server supplies source hashes. Reuse below compares
            // actual F32 rows, so no hash collision can alias visual contents.
            for (const auto & part : split(value, ','))
                if (part.empty() || part.size() > 16 || part.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
                    throw std::runtime_error("invalid img_keys");
        }
        else if (key == "grammar" || key == "grammar_file") throw std::runtime_error("GBNF grammar unavailable; no grammar was applied");
        else throw std::runtime_error("unsupported sampling key: " + key);
    }
    if (!have_ids || r.ids.empty()) throw std::runtime_error("empty prompt");
    if (int64_t(r.ids.size()) + r.max_new > context) throw std::runtime_error("prompt plus max_new exceeds context");
    if (r.masked && r.allowed.empty()) throw std::runtime_error("empty allowed_ids support");
    return r;
}
inline int sample_token(const std::vector<float> & raw, const Request & r, const std::vector<int32_t> & history,
                        const std::set<int> & suppress, std::mt19937_64 & rng) {
    std::vector<float> logits = raw;
    std::set<int> excluded = suppress;
    for (const auto & entry : r.bias) {
        if (entry.second <= -100) excluded.insert(entry.first);
        else logits.at(size_t(entry.first)) += float(entry.second);
    }
    if (r.penalty_repeat != 1 || r.penalty_freq != 0 || r.penalty_present != 0) {
        std::unordered_map<int32_t, int> counts;
        const auto count = std::min(size_t(r.penalty_last_n), history.size());
        for (size_t i = history.size() - count; i < history.size(); ++i) if (history[i] >= 0) ++counts[history[i]];
        for (const auto & item : counts) {
            auto & value = logits.at(size_t(item.first));
            value = float(value <= 0 ? value * r.penalty_repeat : value / r.penalty_repeat);
            value -= float(item.second * r.penalty_freq + r.penalty_present);
        }
    }
    return ::sample(logits, r.temperature, r.top_p, r.top_k, r.min_p, excluded, rng, r.masked ? &r.allowed : nullptr);
}
inline int reusable_prefix(const std::vector<int32_t> & history, const std::vector<int32_t> & prompt) {
    size_t same = 0;
    while (same < history.size() && same < prompt.size() && history[same] == prompt[same]) ++same;
    // Re-read at least the last token so its logits have the fresh request's head recipe.
    same = std::min(same, prompt.size() - 1);
    return int(same / 256 * 256);
}
} // namespace strata::m31::protocol
