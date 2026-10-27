#include "m31_protocol.hpp"
#include <functional>
#include <iostream>
#include <limits>
using namespace strata::m31::protocol;
void check(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
void rejects(const std::function<void()> & operation) {
    try { operation(); } catch (const std::exception &) { return; }
    throw std::runtime_error("expected rejection");
}
int main() {
    const auto request = parse_request("GEN 3 temperature=1 top_p=0.95 seed=42 cache_prompt=0 logit_bias=2:-100,3:5 1,2", 8, 8);
    check(request.ids == std::vector<int32_t>({1,2}), "prompt parsing");
    check(request.max_new == 3 && !request.cache_prompt && request.seeded && request.seed == 42, "request settings");
    check(request.bias.at(2) == -100 && request.bias.at(3) == 5, "bias parsing");
    check(parse_request("GEN 1 top_k=99 1",8,8).top_k == 99, "wide top_k is bounded by sampling support");
    check(parse_request("GEN 1 seed=0 1",8,8).seeded, "zero seed must be explicit");
    check(parse_request("GEN 1 seed=18446744073709551615 1",8,8).seed == std::numeric_limits<uint64_t>::max(), "uint64 seed");
    check(parse_request("BGEN 2 0 1",8,8).slot == 2, "batch parsing and zero bound");
    for (const auto * bad : {"GEN", "GEN -1 1", "GEN 8 1", "GEN 1 8", "GEN 1 -1", "GEN 1 1,", "GEN 1 1 2", "GEN 1 x=3 1",
             "GEN 1 temperature=nan 1", "GEN 1 top_p=0 1", "GEN 1 top_k=-1 1", "GEN 1 seed=1junk 1", "GEN 1 seed=-1 1", "GEN 1 seed=18446744073709551616 1", "GEN 1 cache_prompt=2 1",
             "GEN 1 logit_bias=1:101 1", "GEN 1 logit_bias=1:2,1:3 1", "GEN 1 temperature=1 temperature=2 1", "GEN 1 1 seed=3",
             "GEN 1 grammar=unhandled 1", "GENI 1 1", "BGENI 1 1 1"})
        rejects([&] { parse_request(bad,8,8); });
    auto greedy = parse_request("GEN 1 allowed_ids=1,3 logit_bias=3:5 0",4,8);
    std::mt19937_64 random(3), original = random;
    check(sample_token({9,2,3,1},greedy,{}, {}, random) == 3, "allowed mask and bias");
    check(random == original, "greedy consumed RNG");
    auto penalized = parse_request("GEN 1 penalty_repeat=2 penalty_freq=1 penalty_present=1 penalty_last_n=2 0",4,8);
    check(sample_token({4,3,2,1},penalized,{0,0},{},random) == 1, "request penalties");
    auto banned = parse_request("GEN 1 logit_bias=0:-100,1:-100,2:-100,3:-100 0",4,8);
    rejects([&] { sample_token({4,3,2,1},banned,{}, {}, random); });
    auto masked = parse_request("GEN 1 allowed_ids=0 0",4,8);
    rejects([&] { sample_token({0,std::numeric_limits<float>::quiet_NaN(),0,0},masked,{}, {},random); });
    std::vector<int32_t> history(600,1), prompt(700,1);
    check(reusable_prefix(history,prompt)==512, "append prefix boundary");
    prompt[300]=2;
    check(reusable_prefix(history,prompt)==256, "branch edit prefix boundary");
    check(reusable_prefix(history,std::vector<int32_t>(256,1))==0, "must reread final token");
    check(reusable_prefix(history,std::vector<int32_t>(257,1))==256, "257 boundary");
    std::cout << "protocol parse/masks/penalties/bounds/prefix checks passed\n";
}
