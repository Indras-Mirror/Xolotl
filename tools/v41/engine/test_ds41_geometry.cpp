// test_ds41_geometry <file.gguf> - P1 gate: geometry + full tensor map + expert tiers of a v41_gguf.py file, and
// mutation checks (a missing layer / a layer the file does not hold must fail the map check).
#include "ds41_geometry.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata;
using namespace strata::v41;

int main(int argc, char ** argv) {
    const char * gguf = nullptr;
    if (argc >= 2 && argv[1][0] != '\0') gguf = argv[1];
    else if (const char * env = std::getenv("DS41_TEST_GGUF")) {
        if (env[0] != '\0') gguf = env;
    }
    if (!gguf) {
        std::fprintf(stderr, "test_ds41_geometry: skipped (pass a .gguf path or set DS41_TEST_GGUF)\n");
        return 77;
    }
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & detail = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  ", detail.c_str());
        fails += !ok;
    };
    GgufModel m = GgufModel::open(gguf);
    Ds41Geometry G;
    std::string e = ds41_read_geometry(m, G);
    check("geometry", e.empty(), e);
    if (!e.empty()) return 1;
    std::printf("D %lld layers %lld heads %lld d_head %lld rope %lld q_lora %lld o %lldx%lld swa %lld experts %lld/%lld ff %lld "
                "hc %lld idx %lldx%lld top %lld cand L%lld %lldx%lld eps %g vocab %lld ctx %lld clamp %g\n",
                (long long) G.n_embd, (long long) G.n_layer, (long long) G.n_head, (long long) G.d_head, (long long) G.d_rope,
                (long long) G.r_q, (long long) G.o_groups, (long long) G.o_lora, (long long) G.n_swa, (long long) G.n_expert,
                (long long) G.n_used, (long long) G.n_ff, (long long) G.hc, (long long) G.idx_heads, (long long) G.idx_dim,
                (long long) G.idx_topk, (long long) G.cand_layer, (long long) G.cand_blocks, (long long) G.cand_bsize,
                G.rms_eps, (long long) G.vocab, (long long) G.ctx_len, G.swiglu_limit);
    std::printf("engram layers %zu heads %lld dim %lld ngram %lld pad %lld cvocab %lld token_map %zu primes %zu mult %zu\n",
                G.engram_layers.size(), (long long) G.engram_heads, (long long) G.engram_dim, (long long) G.engram_ngram,
                (long long) G.engram_pad, (long long) G.engram_cvocab, G.engram_token_map.size(), G.engram_primes.size(),
                G.engram_mult.size());
    check("V4.1 shape", G.n_embd == 5120 && G.n_layer == 40 && G.n_expert == 384 && G.n_used == 6 && G.n_ff == 2304 &&
                        G.r_q == 1280 && G.d_head == 512 && G.idx_heads == 32 && G.idx_dim == 128 && G.hc == 4);
    uint32_t mx = 0;
    for (uint32_t v : G.engram_token_map) mx = std::max(mx, v);
    check("engram token map within compressed vocab", (int64_t) mx < G.engram_cvocab && mx + 1 == (uint32_t) G.engram_cvocab);
    std::printf("sources: kv_src/idx_src per layer:");
    for (int64_t l = 0; l < G.n_layer; ++l) std::printf(" %lld:%lld/%lld", (long long) l, (long long) G.kv_src(l), (long long) G.idx_src(l));
    std::printf("\n");
    const std::vector<int64_t> layers = ds41_file_layers(m, G);
    e = ds41_check_tensors(m, G, layers);
    check("full tensor map", e.empty(), e);
    for (int64_t l : layers) {
        std::map<std::pair<int, int>, int> pairs;
        const auto & X = G.experts[(size_t) l];
        for (int64_t x = 0; x < G.n_expert; ++x) pairs[{ X.gu_type[(size_t) x], X.d_type[(size_t) x] }]++;
        std::printf("L%lld tiers:", (long long) l);
        for (const auto & [p, n] : pairs) std::printf(" %s/%s x%d", ggml_type_name((uint32_t) p.first), ggml_type_name((uint32_t) p.second), n);
        std::printf("\n");
    }
    // mutations: a layer missing from the expectation -> its tensors are unexpected; one the file lacks -> missing
    if (layers.size() > 1) {
        std::vector<int64_t> fewer(layers.begin() + 1, layers.end());
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, fewer);
        check("mutation: unpredicted layer rejected", e.rfind("unexpected tensor", 0) == 0, e);
    }
    if ((int64_t) layers.size() < G.n_layer) {
        std::vector<int64_t> more = layers;
        for (int64_t l = 0; l < G.n_layer; ++l) if (std::find(layers.begin(), layers.end(), l) == layers.end()) { more.push_back(l); break; }
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, more);
        check("mutation: absent layer rejected", e.find("missing") != std::string::npos, e);
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
