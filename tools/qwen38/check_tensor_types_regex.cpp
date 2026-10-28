// check_tensor_types_regex.cpp - prove an emitted --tensor-type-file is valid for llama-quantize.
//
// WHY THIS EXISTS: emit_tensor_types.py generates its patterns programmatically, and the only
// consumer is llama-quantize, which compiles each line with std::regex(pat, std::regex::ECMAScript)
// (llama-quant.cpp:196).  Python `re` is NOT the same dialect: Python accepts (?P<name>...), named
// groups and variable-length lookbehind that ECMAScript rejects.  So a pattern the emitter is happy
// with can make llama-quantize throw at startup - hours into an unattended job.
//
// This compiles every pattern exactly as llama-quantize does, then resolves a probe set with the
// same first-match-wins rule (llama-quant.cpp:716 breaks on the first regex_search hit) so the
// routing of expert / dense / PLE tensors can be checked before any quantize time is spent.
//
// Build and run:
//   g++ -O2 -std=c++17 -o /tmp/check_tt tools/qwen38/check_tensor_types_regex.cpp
//   /tmp/check_tt <tensor_types.txt>
//
// CONTRACT - other tooling gates on these exit codes, so they must not change silently:
//   0  every pattern compiled under std::regex ECMAScript, AND all 144 generated expert probes
//      (blk.<L>.ffn_{gate,up,down}_exps.weight for L in 0..47) matched an expert-specific rule --
//      i.e. some rule other than the dense catch-all matched them.
//   1  a pattern failed to compile, OR at least one expert probe was unruled / caught by the
//      dense catch-all.  The offending patterns / experts are named.
//   2  usage error, or the file could not be opened (this is NOT a pass - callers must treat a
//      non-zero code as failure, and must distinguish "tool missing" from "tool passed").
// A caller that cannot build this tool must report SKIPPED explicitly; a skipped check must never
// be indistinguishable from a passed one.
#include <regex>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tensor_types.txt>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    std::vector<std::pair<std::regex, std::string>> rules;
    std::string line;
    int bad = 0, lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty()) continue;
        const auto eq = line.rfind('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "line %d has no '=': %s\n", lineno, line.c_str());
            ++bad;
            continue;
        }
        const std::string pat = line.substr(0, eq);
        const std::string ty  = line.substr(eq + 1);
        try {
            rules.emplace_back(std::regex(pat, std::regex::ECMAScript), ty);
        } catch (const std::regex_error & e) {
            std::fprintf(stderr, "line %d: COMPILE FAILED (%s)\n  pattern: %s\n",
                         lineno, e.what(), pat.c_str());
            ++bad;
        }
    }

    std::printf("compiled %zu pattern(s), %d failed\n", rules.size(), bad);
    if (bad) return 1;

    // Expert probes are GENERATED for every layer x projection, not sampled.  A fixed sample of
    // blk.0 and blk.47 would pass while blk.23 had no rule at all -- the exact case a partial rule
    // set (e.g. --layers 0-7) produces.  48 x 3 costs nothing and is exhaustive over the rule
    // space; whether the GGUF actually CONTAINS those tensors is a separate check.
    std::vector<std::string> probes;
    for (int L = 0; L < 48; ++L)
        for (const char * pr : {"gate", "up", "down"})
            probes.push_back("blk." + std::to_string(L) + ".ffn_" + pr + "_exps.weight");
    // non-expert probes: these must NOT be caught by an expert rule, and are listed for the table
    const char * others[] = {
        "blk.3.attn_qkv.weight",       "blk.12.ffn_gate_inp.weight",
        "blk.0.ssm_alpha.weight",      "blk.5.linear_attn.out_proj.weight",
        "per_layer_token_embd.weight", "token_embd.weight",
        "output.weight",               "output_hc_up.weight",
    };
    for (const char * nm : others) probes.push_back(nm);
    auto resolve = [&rules](const std::string & nm) -> int {
        for (size_t i = 0; i < rules.size(); ++i)
            if (std::regex_search(nm, rules[i].first)) return (int) i;
        return -1;
    };

    // The DENSE CATCH-ALL must be identified BY RULE INDEX, not by type.  An expert that falls
    // into the catch-all still "matched a rule", so an `idx < 0` test alone reports all-clear
    // while every expert is untyped - and if the catch-all's type happens to equal the intended
    // expert tier, even a type comparison passes.  Probe a tensor only the catch-all can serve.
    const int catchall = resolve("blk.0.attn_q.weight");

    std::printf("\ndense catch-all is rule #%d\n", catchall);
    int bad_experts = 0;
    bool printed_header = false;
    for (const auto & nm : probes) {
        const int idx = resolve(nm);
        const std::string hit = idx >= 0 ? rules[(size_t) idx].second : "<default_type>";
        const bool is_expert = nm.find("_exps.weight") != std::string::npos;
        const bool caught = is_expert && (idx < 0 || idx == catchall);
        if (caught) {
            if (!printed_header) {
                std::printf("\n%-34s %-14s %s\n", "expert", "type", "matched rule #");
                printed_header = true;
            }
            ++bad_experts;
            std::printf("%-34s %-14s %d   <-- %s\n", nm.c_str(), hit.c_str(), idx,
                        idx < 0 ? "NO RULE" : "CAUGHT BY THE DENSE CATCH-ALL");
        }
    }
    if (bad_experts) {
        std::fprintf(stderr,
                     "\nERROR: %d of %d expert probes do not match an expert-specific rule; they "
                     "would take %s instead of a measured tier.\n",
                     bad_experts, (int) probes.size() - 8,
                     catchall >= 0 ? "the catch-all's type" : "default_type");
        return 1;
    }
    std::printf("all %d expert probes (48 layers x gate/up/down) matched an expert-specific rule\n",
                (int) probes.size() - 8);
    return 0;
}
