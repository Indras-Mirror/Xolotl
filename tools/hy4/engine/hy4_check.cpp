// hy4_check <model-00001-of-N.gguf> <golden dir> - the Hy4 P2 gate: run the golden prompt through the native dense half
// (CUDA) + the reference experts (CPU) in one pass and compare every layer with the llama.cpp reference dump
// (~/v41-work/hy4/tools/hy4_golden: <name>-<layer>.f32 files + index.tsv + tokens.i32 + logits_last.f32).
//
// Per layer: rel L2 error of l_out (the hc streams after the layer) and attn_out, the overlap of the chosen experts
// with ffn_moe_topk, then the last row's logits: argmax match, max |diff|, KL(ref || ours).
// Env: HY4_CPU=1 (dense half on the CPU too), HY4_LAYERS=N (stop after N layers), HY4_THREADS.
#include "hy4_dense.hpp"
#include "hy4_moe.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef HY4_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::hy4;

template <class T> static bool load(const std::string & p, std::vector<T> & v) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto sz = (size_t) f.tellg();
    v.resize(sz / sizeof(T));
    f.seekg(0);
    f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    return true;
}

static double relerr(const std::vector<float> & a, const std::vector<float> & b, size_t off = 0, size_t n = 0) {
    if (!n) n = b.size();
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[off + i] - b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hy4_check <model.gguf> <golden dir>\n");
        return 2;
    }
    const std::string gguf = argv[1], gd = argv[2];
    std::vector<int32_t> toks;
    if (!load(gd + "/tokens.i32", toks) || toks.empty()) { std::fprintf(stderr, "no %s/tokens.i32\n", gd.c_str()); return 2; }
    const int n = (int) toks.size();
    const int nthr = std::getenv("HY4_THREADS") ? std::atoi(std::getenv("HY4_THREADS")) : 16;

    ggml_backend_t be = nullptr;
#ifdef HY4_CUDA
    if (!(std::getenv("HY4_CPU") && std::getenv("HY4_CPU")[0] == '1')) be = ggml_backend_cuda_init(0);
#endif
    if (!be) {
        be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, nthr);
    }
    Hy4DenseConfig cfg;
    cfg.backend = be;
    cfg.ctx_max = std::max(2048, n);
    cfg.max_pass = n;
    cfg.taps = true;
    Hy4Dense dense;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!dense.init(gguf, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const Hy4Geometry & G = dense.geom();
    Hy4Moe moe;
    if (!moe.init(dense.model(), gguf, G, nthr, err)) { std::fprintf(stderr, "moe init: %s\n", err.c_str()); return 1; }
    std::printf("hy4_check: %d tokens, init %.1f s\n", n,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    if (!dense.begin(toks.data(), n, 0)) return 1;
    int n_layers = (int) G.n_layer;
    if (std::getenv("HY4_LAYERS")) n_layers = std::min(n_layers, std::atoi(std::getenv("HY4_LAYERS")));
    std::vector<int> ids((size_t) (n * G.n_used));
    std::vector<float> wts(ids.size()), routed((size_t) (n * G.n_latent)), st;
    double worst = 0;
    std::printf("%5s %10s %10s %10s %8s %8s\n", "layer", "l_out", "attn_out", "engram", "topk%", "sec");
    for (int il = 0; il < n_layers; ++il) {
        auto ts = std::chrono::steady_clock::now();
        const float * fn = nullptr;
        bool dense_layer = false;
        // HY4_FORCE=1: start each layer from the reference's previous l_out, so every row is that layer's own error
        if (std::getenv("HY4_FORCE") && il > 0) {
            std::vector<float> prev;
            if (!load(gd + "/l_out-" + std::to_string(il - 1) + ".f32", prev)) { std::fprintf(stderr, "no golden l_out\n"); return 1; }
            dense.write_state(prev.data());
        }
        if (!dense.attn_router(il, ids.data(), wts.data(), &fn, &dense_layer)) return 1;
        auto taps = dense.taps();   // copy: finish() replaces them
        double overlap = -1;
        if (!dense_layer) {
            std::vector<int32_t> ref;
            if (load(gd + "/ffn_moe_topk-" + std::to_string(il) + ".i32", ref) && ref.size() == ids.size()) {
                size_t hit = 0;
                for (int t = 0; t < n; ++t) {
                    std::set<int> a(ids.begin() + t * G.n_used, ids.begin() + (t + 1) * G.n_used);
                    for (int u = 0; u < G.n_used; ++u) hit += a.count(ref[(size_t) (t * G.n_used + u)]);
                }
                overlap = 100.0 * (double) hit / (double) ref.size();
            }
            if (!moe.run(il, fn, ids.data(), wts.data(), n, routed.data())) return 1;
            if (!dense.finish(il, routed.data())) return 1;
        }
        dense.read_state(st);
        std::vector<float> ref;
        const std::string sl = std::to_string(il);
        const double e_l = load(gd + "/l_out-" + sl + ".f32", ref) && ref.size() == st.size() ? relerr(st, ref) : -1;
        const double e_a = load(gd + "/attn_out-" + sl + ".f32", ref) && taps.count("attn_out") && ref.size() == taps["attn_out"].size()
                               ? relerr(taps["attn_out"], ref) : -1;
        const double e_e = taps.count("engram_out") && load(gd + "/engram_out-" + sl + ".f32", ref) && ref.size() == taps["engram_out"].size()
                               ? relerr(taps["engram_out"], ref) : -1;
        worst = std::max(worst, e_l);
        if (const char * sd = std::getenv("HY4_SAVE")) {   // HY4_SAVE=dir: our l_out of every layer, for offline analysis
            std::ofstream f(std::string(sd) + "/ours_l_out-" + sl + ".f32", std::ios::binary);
            f.write((const char *) st.data(), (std::streamsize) (st.size() * 4));
            for (auto & kv : taps) {
                std::ofstream g(std::string(sd) + "/ours_" + kv.first + "-" + sl + ".f32", std::ios::binary);
                g.write((const char *) kv.second.data(), (std::streamsize) (kv.second.size() * 4));
            }
        }
        // HY4_PERTOK=1: per-token rel err of l_out every 10 layers and at the last (where does the drift sit?)
        if (std::getenv("HY4_PERTOK") && (il % 10 == 0 || il == n_layers - 1) && load(gd + "/l_out-" + sl + ".f32", ref) &&
            ref.size() == st.size()) {
            const size_t tok = (size_t) (G.hc * G.n_embd);
            std::printf("  pertok l%d:", il);
            for (int t = 0; t < n; ++t) {
                std::vector<float> rt(ref.begin() + t * tok, ref.begin() + (t + 1) * tok);
                double nr = 0;
                for (float v : rt) nr += (double) v * v;
                std::printf(" %d:%.3f(|%.0f|)", t, relerr(st, rt, t * tok, tok), std::sqrt(nr));
            }
            std::printf("\n");
        }
        std::printf("%5d %10.3e %10.3e %10.3e %8.2f %8.2f\n", il, e_l, e_a, e_e, overlap,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
        std::fflush(stdout);
    }
    if (n_layers < G.n_layer) {
        std::printf("stopped after %d layers (HY4_LAYERS); worst l_out rel err %.3e\n", n_layers, worst);
        return 0;
    }
    std::vector<float> lg((size_t) G.vocab), lref;
    if (!dense.logits(n - 1, 1, lg.data())) return 1;
    if (!load(gd + "/logits_last.f32", lref) || lref.size() != lg.size()) { std::fprintf(stderr, "no logits_last.f32\n"); return 1; }
    const int am = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int ar = (int) (std::max_element(lref.begin(), lref.end()) - lref.begin());
    double mx = 0, zr = 0, zo = 0;
    const double mr = lref[(size_t) ar], mo = lg[(size_t) am];
    for (size_t i = 0; i < lg.size(); ++i) {
        mx = std::max(mx, (double) std::fabs(lg[i] - lref[i]));
        zr += std::exp(lref[i] - mr);
        zo += std::exp(lg[i] - mo);
    }
    double kl = 0;
    for (size_t i = 0; i < lg.size(); ++i) {
        const double pr = std::exp(lref[i] - mr) / zr;
        const double lpo = (lg[i] - mo) - std::log(zo), lpr = (lref[i] - mr) - std::log(zr);
        kl += pr * (lpr - lpo);
    }
    std::printf("logits: argmax ours %d ref %d %s | max|diff| %.4f | KL(ref||ours) %.3e | worst l_out rel err %.3e\n", am, ar,
                am == ar ? "MATCH" : "DIFFER", mx, kl, worst);
    return am == ar ? 0 : 1;
}
