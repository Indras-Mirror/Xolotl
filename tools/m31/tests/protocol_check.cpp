#include "m31_protocol.hpp"
#include "m31_vision.hpp"
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
    const auto vision = parse_request("GENI 1 img_keys=0123,abcd image.sve1 1,-1,-1,2,-1",8,8,true);
    check(vision.vision && vision.embeddings == "image.sve1" && vision.ids[1] == -1, "GENI common positional format");
    check(parse_request("BGENI 3 1 image.sve1 1,-1",8,8,true).slot == 3, "BGENI slot");
    for (const auto * bad : {"GENI 1 image.sve1", "GENI 1 image.sve1 1,-2", "GENI 1 img_keys=xyz image.sve1 1,-1",
                            "GENI 1 image.sve1 seed=3 1,-1", "GEN 1 img_keys=abc 1"})
        rejects([&] { parse_request(bad,8,8,true); });
    namespace fs = std::filesystem;
    const auto test_dir = fs::temp_directory_path() / ("m31-vision-protocol-" + std::to_string(std::random_device{}()));
    check(fs::create_directory(test_dir), "temporary vision fixture directory");
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); } } cleanup{test_dir};
    const auto blob = test_dir / "input.sve1";
    auto write_blob = [&](const std::vector<int> & sizes, int width = 4, bool nonfinite = false) {
        std::ofstream file(blob, std::ios::binary | std::ios::trunc);
        for (auto n : sizes) {
            const int32_t header[] = {0x31455653, n, n, 1, width};
            file.write(reinterpret_cast<const char *>(header), sizeof header);
            std::vector<float> values(size_t(n * width), .25f);
            if (nonfinite) values[0] = std::numeric_limits<float>::infinity();
            file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
        }
    };
    write_blob({2,1});
    const auto loaded = strata::m31::load_vision_embeddings(blob, vision.ids, 4);
    check(loaded.visual_count == 3 && loaded.visual_rows == std::vector<int32_t>({-1,0,1,-1,2}), "SVE1 span placement");
    rejects([&] { strata::m31::load_vision_embeddings(blob, vision.ids, 8); });
    rejects([&] { strata::m31::load_vision_embeddings(test_dir, vision.ids, 4); });
    rejects([&] { strata::m31::load_vision_embeddings(test_dir / "absent", vision.ids, 4); });
    write_blob({3}); // A record cannot cross text between visual spans.
    rejects([&] { strata::m31::load_vision_embeddings(blob, vision.ids, 4); });
    write_blob({2,1},4,true);
    rejects([&] { strata::m31::load_vision_embeddings(blob, vision.ids, 4); });
    write_blob({2,1}); fs::resize_file(blob, fs::file_size(blob)-1);
    rejects([&] { strata::m31::load_vision_embeddings(blob, vision.ids, 4); });
    std::vector<int32_t> visual_ids(700,1); visual_ids[300] = -1;
    auto old_media = loaded, new_media = loaded;
    old_media.visual_rows.assign(700,-1); old_media.visual_rows[300] = 0;
    new_media.visual_rows = old_media.visual_rows;
    check(strata::m31::reusable_visual_prefix(512, visual_ids,old_media,new_media,4)==512,"same media lost prefix");
    new_media.embeddings[0] += .5f;
    check(strata::m31::reusable_visual_prefix(512, visual_ids,old_media,new_media,4)==256,"changed media reused stale KV");
    // Placeholder rows are never vocabulary entries for repetition penalties.
    auto visual_penalty = parse_request("GENI 1 penalty_repeat=2 x.sve1 -1,0",4,8,true);
    std::mt19937_64 visual_rng(1);
    check(sample_token({4,3,2,1},visual_penalty,{-1,0},{},visual_rng)==1,"visual placeholder entered penalties");
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
