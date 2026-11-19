// test_hy4_geometry <file.gguf> - Hy4 P1 gate: geometry + full tensor map of a hy_v4_flash GGUF (ours or Tencent's),
// plus mutation checks (a dropped layer / a renamed tensor must fail the map check).
#include "hy4_geometry.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata;
using namespace strata::hy4;

int main(int argc, char ** argv) {
    const char * gguf = argc >= 2 && argv[1][0] ? argv[1] : std::getenv("HY4_TEST_GGUF");
    if (!gguf || !gguf[0]) {
        std::fprintf(stderr, "test_hy4_geometry: skipped (pass a .gguf path or set HY4_TEST_GGUF)\n");
        return 77;
    }
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & d = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, d.empty() ? "" : "  ", d.c_str());
        fails += !ok;
    };
    GgufModel m = GgufModel::open(gguf);
    Hy4Geometry G;
    std::string e = hy4_read_geometry(m, gguf, G);
    check("geometry", e.empty(), e);
    if (!e.empty()) return 1;
    int n_swa = 0, n_idx = 0;
    for (int64_t l = 0; l < G.n_layer; ++l) { n_swa += G.swa(l); n_idx += G.indexed(l); }
    std::printf("D %lld layers %lld (+%lld nextn) swa %d (window %lld) dsa %d heads %lld/%lld q_lora %lld kv_lora %lld "
                "qk %lld v %lld rope %lld theta %g/%g\n",
                (long long) G.n_embd, (long long) G.n_layer, (long long) G.n_nextn, n_swa, (long long) G.n_swa, n_idx,
                (long long) G.heads(0), (long long) G.heads(2), (long long) G.q_lora, (long long) G.kv_lora,
                (long long) G.d_qk, (long long) G.d_v, (long long) G.d_rope, G.rope_base, G.rope_base_swa);
    std::printf("idx %lldx%lld top %lld kpool %lld tail %d | moe %lld/%lld ff %lld latent %lld shared %lld scale %g norm %d "
                "gate %lld clamp %g | dense lead %lld ff %lld | hc %lld eps %g mag %g\n",
                (long long) G.idx_heads, (long long) G.idx_dim, (long long) G.idx_topk, (long long) G.idx_kpool,
                (int) G.idx_select_tail, (long long) G.n_used, (long long) G.n_expert, (long long) G.n_ff_exp,
                (long long) G.n_latent, (long long) G.n_shared, G.weights_scale, (int) G.weights_norm,
                (long long) G.gating_func, G.swiglu_clamp[1], (long long) G.n_dense_lead, (long long) G.n_ff,
                (long long) G.hc, G.hc_eps, G.hc_magnitude);
    std::printf("engram layers");
    for (int64_t l : G.engram_layers) std::printf(" %lld", (long long) l);
    std::printf(" | %lld-gram x %lld heads x %lld | rows %lld | token_map %zu pad %lld\n", (long long) G.engram_ngram,
                (long long) G.engram_heads_per_ngram, (long long) G.engram_dim, (long long) G.engram_rows(),
                G.engram_token_map.size(), (long long) G.engram_pad);
    check("Hy4-flash shape", G.n_embd == 3072 && G.n_layer == 42 && G.n_expert == 512 && G.n_used == 16 &&
                             G.n_latent == 1536 && G.hc == 4 && n_swa == 30 && n_idx == 12 && G.heads(0) == 32 &&
                             G.heads(2) == 64 && G.kv_lora == 512 && G.d_rope == 64 && G.d_qk == 256 && G.d_v == 256);
    check("engram rows = 660006416", G.engram_rows() == 660006416);
    e = hy4_check_tensors(m, G);
    check("full tensor map", e.empty(), e);
    // mutations: the map must notice a layer it does not have and a layer it lost
    {
        Hy4Geometry X = G;
        X.engram_layers.back() += 1;
        check("mutation: moved Engram layer fails", !hy4_check_tensors(m, X).empty());
        Hy4Geometry Y = G;
        Y.has_indexer[2] = true;
        check("mutation: extra indexer layer fails", !hy4_check_tensors(m, Y).empty());
        Hy4Geometry Z = G;
        Z.n_dense_lead = 2;
        check("mutation: dense lead 2 fails", !hy4_check_tensors(m, Z).empty());
    }
    std::printf(fails ? "FAILED (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
