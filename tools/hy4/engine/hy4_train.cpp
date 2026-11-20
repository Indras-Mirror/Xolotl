// hy4_train <model-00001-of-N.gguf> <tokens.i32> [steps] - LoRA training on Hy4-flash (docs/train/LORA_PLAN.md, T1).
//
// One step over one sequence (the token file, cut to HY4_TRAIN_N tokens):
//   forward  the inference path (Hy4Dense + the expert tier or the reference experts, adapters applied), saving each
//            layer's input state and routing on the host
//   head     cross-entropy of every position against the next token -> d loss / d (last state)
//   backward layer by layer from the top down to the lowest adapted layer: Hy4Dense::train_layer rebuilds the layer as
//            one differentiable graph (experts inside it) and returns d loss / d (layer input) + the adapters' grads
//   update   AdamW on the host copies, uploaded back
// Env: HY4_LORA_LAYERS "a-b" (default the top 4), HY4_LORA_TARGETS (comma list of per-layer weight suffixes, default
// attn_q_a,attn_q_b,attn_kv_a_mqa,attn_output), HY4_LORA_RANK (8), HY4_LORA_ALPHA (16), HY4_LR (1e-3), HY4_TRAIN_N
// (64), HY4_TRAIN_FD=1 (finite-difference check of one adapter element before training), HY4_TRAIN_SAVE=<file.gguf|file.bin>
// (the adapters after the last step: .gguf = llama.cpp-format LoRA GGUF, else raw F32 name + shapes).  Expert env as hy4_generate.
#include "hy4_dense.hpp"
#include "hy4_moe.hpp"
#include "../../train/lora_gguf.hpp"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::hy4;

namespace {

std::vector<int32_t> load_i32(const std::string & f) {
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    std::vector<int32_t> v;
    if (!in) return v;
    v.resize((size_t) in.tellg() / 4);
    in.seekg(0);
    in.read((char *) v.data(), (std::streamsize) (v.size() * 4));
    return v;
}
const char * env(const char * k, const char * d) { const char * v = std::getenv(k); return v && *v ? v : d; }

struct Step {   // what a forward leaves for the backward
    std::vector<std::vector<float>> x_in;   // per layer
    std::vector<std::vector<int>> ids;      // per MoE layer (n * n_used)
    std::vector<float> x_last;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hy4_train <model.gguf> <tokens.i32> [steps]\n");
        return 2;
    }
    const std::string gguf = argv[1];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 20;
    std::vector<int32_t> toks = load_i32(argv[2]);
    const int n = std::min((int) toks.size(), std::atoi(env("HY4_TRAIN_N", "64")));
    if (n < 2) { std::fprintf(stderr, "need >= 2 tokens\n"); return 2; }
    toks.resize((size_t) n);
    const int nthr = std::atoi(env("HY4_THREADS", "16"));

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "no CUDA\n"); return 1; }
    Hy4DenseConfig cfg;
    cfg.backend = be;
    cfg.ctx_max = std::max(2048, (n + 3) / 4 * 4);
    cfg.max_pass = n;
    Hy4Dense dense;
    std::string err;
    if (!dense.init(gguf, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const Hy4Geometry & G = dense.geom();
    if (n > dense.kpool_exact_limit()) { std::fprintf(stderr, "HY4_TRAIN_N above the exact k-pool limit\n"); return 2; }

    // adapters
    int l0 = (int) G.n_layer - 4, l1 = (int) G.n_layer - 1;
    std::sscanf(env("HY4_LORA_LAYERS", ""), "%d-%d", &l0, &l1);
    std::vector<std::string> suff;
    {
        std::string s = env("HY4_LORA_TARGETS", "attn_q_a,attn_q_b,attn_kv_a_mqa,attn_output");
        for (size_t a = 0, b; a <= s.size(); a = b + 1) {
            b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            if (b > a) suff.push_back(s.substr(a, b - a));
        }
    }
    std::vector<std::string> names;
    for (int l = l0; l <= l1; ++l)
        for (const auto & sf : suff) names.push_back("blk." + std::to_string(l) + "." + sf + ".weight");
    const int rank = std::atoi(env("HY4_LORA_RANK", "8"));
    const float alpha = (float) std::atof(env("HY4_LORA_ALPHA", "16"));
    if (!dense.lora_add(names, rank, alpha, 20261120, err)) { std::fprintf(stderr, "lora: %s\n", err.c_str()); return 1; }
    const int NL = dense.lora_count();
    size_t n_par = 0;
    for (int i = 0; i < NL; ++i) { int64_t a, b, r; dense.lora_shape(i, a, b, r); n_par += (size_t) (r * (a + b)); }
    int lowest = (int) G.n_layer;
    for (int i = 0; i < NL; ++i) if (dense.lora_layer(i) >= 0) lowest = std::min(lowest, dense.lora_layer(i));
    std::printf("hy4_train: %d tokens, %d adapters on layers %d-%d (%zu params, rank %d, alpha %.0f), backward down to layer %d\n",
                n, NL, l0, l1, n_par, rank, alpha, lowest);

    Hy4Moe moe;
    if (!moe.init(dense.model(), gguf, G, nthr, err)) { std::fprintf(stderr, "moe: %s\n", err.c_str()); return 1; }

    std::vector<int> targets((size_t) n, -1);
    for (int t = 0; t + 1 < n; ++t) targets[(size_t) t] = toks[(size_t) t + 1];
    const size_t SX = (size_t) (G.n_embd * G.hc * n);

    std::vector<int> ids((size_t) (n * G.n_used));
    std::vector<float> wts(ids.size()), routed((size_t) (n * G.n_latent));
    auto forward = [&](Step & st) -> bool {
        st.x_in.assign((size_t) G.n_layer, {});
        st.ids.assign((size_t) G.n_layer, {});
        if (!dense.begin(toks.data(), n, 0)) return false;
        for (int il = 0; il < G.n_layer; ++il) {
            if (il >= lowest) dense.read_state(st.x_in[(size_t) il]);
            const float * fn = nullptr;
            bool dl = false;
            if (!dense.attn_router(il, ids.data(), wts.data(), &fn, &dl)) return false;
            if (dl) continue;
            if (il >= lowest) st.ids[(size_t) il] = ids;
            if (!moe.run(il, fn, ids.data(), wts.data(), n, routed.data())) return false;
            if (!dense.finish(il, routed.data())) return false;
        }
        return dense.read_state(st.x_last);
    };
    std::vector<std::vector<float>> gA, gB;
    std::vector<float> gx(SX), gin(SX);
    auto loss_only = [&](float & loss) -> bool {   // forward + head (the head's grads are discarded)
        Step st;
        if (!forward(st)) return false;
        std::vector<std::vector<float>> a, b;
        return dense.train_head(st.x_last.data(), targets.data(), &loss, gx.data(), a, b);
    };
    auto step_grads = [&](float & loss, double & t_f, double & t_b) -> bool {
        auto t0 = std::chrono::steady_clock::now();
        Step st;
        if (!forward(st)) return false;
        gA.assign((size_t) NL, {});
        gB.assign((size_t) NL, {});
        if (!dense.train_head(st.x_last.data(), targets.data(), &loss, gx.data(), gA, gB)) return false;
        auto t1 = std::chrono::steady_clock::now();
        const bool chk = env("HY4_TRAIN_CHECK", "0")[0] == '1';
        for (int il = (int) G.n_layer - 1; il >= lowest; --il) {
            // HY4_TRAIN_CHECK=1: the training graph's layer output vs the forward's (the next layer's saved input)
            const float * expect = chk ? (il + 1 < G.n_layer ? st.x_in[(size_t) il + 1].data() : st.x_last.data()) : nullptr;
            double re = -1;
            if (!dense.train_layer(il, st.x_in[(size_t) il].data(), gx.data(), st.ids[(size_t) il].data(), gin.data(), gA, gB,
                                   expect, &re)) return false;
            if (chk) {
                double gn = 0;
                for (float v : gin) gn += (double) v * v;
                std::printf("  layer %2d: train-graph output vs forward rel err %.3e | |g_in| %.3e\n", il, re, std::sqrt(gn));
            }
            std::swap(gx, gin);
        }
        auto t2 = std::chrono::steady_clock::now();
        t_f = std::chrono::duration<double>(t1 - t0).count();
        t_b = std::chrono::duration<double>(t2 - t1).count();
        return true;
    };

    // host masters + AdamW state
    std::vector<std::vector<float>> A((size_t) NL), B((size_t) NL), mA((size_t) NL), vA((size_t) NL), mB((size_t) NL), vB((size_t) NL);
    for (int i = 0; i < NL; ++i) {
        dense.lora_get(i, A[(size_t) i], B[(size_t) i]);
        mA[(size_t) i].assign(A[(size_t) i].size(), 0.0f); vA[(size_t) i] = mA[(size_t) i];
        mB[(size_t) i].assign(B[(size_t) i].size(), 0.0f); vB[(size_t) i] = mB[(size_t) i];
    }

    if (env("HY4_TRAIN_FD", "0")[0] == '1') {
        // one element of the top adapter's B: analytic vs central difference through the whole model (the as-served
        // forward rounds activations in its quantized products, so expect agreement to ~10 %, not 1e-4)
        float loss = 0;
        double tf, tb;
        if (!step_grads(loss, tf, tb)) { std::fprintf(stderr, "step: %s\n", dense.last_error().c_str()); return 1; }
        int i = NL - 1;
        size_t j = 0;
        for (size_t k = 0; k < gB[(size_t) i].size(); ++k) if (std::fabs(gB[(size_t) i][k]) > std::fabs(gB[(size_t) i][j])) j = k;
        const float eps = (float) std::atof(env("HY4_FD_EPS", "0.02"));
        float lp = 0, lm = 0;
        auto Bp = B[(size_t) i];
        Bp[j] += eps; dense.lora_set(i, A[(size_t) i].data(), Bp.data()); loss_only(lp);
        Bp[j] -= 2 * eps; dense.lora_set(i, A[(size_t) i].data(), Bp.data()); loss_only(lm);
        dense.lora_set(i, A[(size_t) i].data(), B[(size_t) i].data());
        std::printf("FD check %s B[%zu]: analytic %+.6f numeric %+.6f (loss %.5f)\n", dense.lora_name(i).c_str(), j,
                    gB[(size_t) i][j], (lp - lm) / (2 * eps), loss);
        // HY4_FD_DIR="e1,e2,...": directional derivative along the unit gradient over ALL adapter params. Analytic
        // = |g|; the loss change is eps*|g| per side, far above the activation-rounding noise of one element.
        // The same eps list is also used for the single element above (per-element numeric at each eps).
        const char * dl = env("HY4_FD_DIR", "");
        if (*dl) {
            double gn2 = 0;
            for (int q = 0; q < NL; ++q) {
                for (float g : gA[(size_t) q]) gn2 += (double) g * g;
                for (float g : gB[(size_t) q]) gn2 += (double) g * g;
            }
            const double gnorm = std::sqrt(gn2);
            auto set_dir = [&](double s) {   // params + s * g / |g|
                for (int q = 0; q < NL; ++q) {
                    auto a = A[(size_t) q], b = B[(size_t) q];
                    if (!gA[(size_t) q].empty()) for (size_t k = 0; k < a.size(); ++k) a[k] += (float) (s * gA[(size_t) q][k] / gnorm);
                    if (!gB[(size_t) q].empty()) for (size_t k = 0; k < b.size(); ++k) b[k] += (float) (s * gB[(size_t) q][k] / gnorm);
                    dense.lora_set(q, a.data(), b.data());
                }
            };
            for (const char * p = dl; *p;) {
                const double e = std::atof(p);
                float dp = 0, dm = 0;
                set_dir(+e); loss_only(dp);
                set_dir(-e); loss_only(dm);
                set_dir(0);
                float ep = 0, em = 0;
                auto Bq = B[(size_t) i];
                Bq[j] += (float) e; dense.lora_set(i, A[(size_t) i].data(), Bq.data()); loss_only(ep);
                Bq[j] -= (float) (2 * e); dense.lora_set(i, A[(size_t) i].data(), Bq.data()); loss_only(em);
                dense.lora_set(i, A[(size_t) i].data(), B[(size_t) i].data());
                std::printf("FD eps %.4g: direction analytic %.6f numeric %.6f (ratio %.3f; L+ %.5f L- %.5f) | element numeric %+.6f (ratio %.3f)\n",
                            e, gnorm, (dp - dm) / (2 * e), (dp - dm) / (2 * e) / gnorm, dp, dm,
                            (ep - em) / (2 * e), (ep - em) / (2 * e) / gB[(size_t) i][j]);
                std::fflush(stdout);
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
    }

    const float lr = (float) std::atof(env("HY4_LR", "1e-3")), b1 = 0.9f, b2 = 0.999f, wd = 0.0f, aeps = 1e-8f;
    for (int s = 1; s <= steps; ++s) {
        float loss = 0;
        double tf = 0, tb = 0;
        if (!step_grads(loss, tf, tb)) { std::fprintf(stderr, "step %d: %s\n", s, dense.last_error().c_str()); return 1; }
        double gn = 0;
        for (int i = 0; i < NL; ++i) {
            for (float g : gA[(size_t) i]) gn += (double) g * g;
            for (float g : gB[(size_t) i]) gn += (double) g * g;
        }
        gn = std::sqrt(gn);
        const float clip = gn > 1.0 ? (float) (1.0 / gn) : 1.0f;   // global grad-norm clip at 1
        const float c1 = 1.0f - std::pow(b1, (float) s), c2 = 1.0f - std::pow(b2, (float) s);
        auto adam = [&](std::vector<float> & p, std::vector<float> & m, std::vector<float> & v, const std::vector<float> & g) {
            if (g.empty()) return;
            for (size_t k = 0; k < p.size(); ++k) {
                const float gk = g[k] * clip;
                m[k] = b1 * m[k] + (1 - b1) * gk;
                v[k] = b2 * v[k] + (1 - b2) * gk * gk;
                p[k] -= lr * ((m[k] / c1) / (std::sqrt(v[k] / c2) + aeps) + wd * p[k]);
            }
        };
        for (int i = 0; i < NL; ++i) {
            adam(A[(size_t) i], mA[(size_t) i], vA[(size_t) i], gA[(size_t) i]);
            adam(B[(size_t) i], mB[(size_t) i], vB[(size_t) i], gB[(size_t) i]);
            dense.lora_set(i, A[(size_t) i].data(), B[(size_t) i].data());
        }
        std::printf("step %3d  loss %.5f  |grad| %.3e  forward %.1f s  backward %.1f s\n", s, loss, gn, tf, tb);
        std::fflush(stdout);
    }
    {
        float loss = 0;
        if (loss_only(loss)) std::printf("final loss %.5f\n", loss);
    }
    if (const char * sv = std::getenv("HY4_TRAIN_SAVE"); sv && std::string(sv).size() > 5 &&
        std::string(sv).compare(std::string(sv).size() - 5, 5, ".gguf") == 0) {
        // llama.cpp-format LoRA GGUF (tools/train/lora_gguf.hpp)
        std::vector<xolotl::train::LoraTensorPair> ad((size_t) NL);
        for (int i = 0; i < NL; ++i) {
            auto & p = ad[(size_t) i];
            dense.lora_shape(i, p.n_in, p.n_out, p.rank);
            p.name = dense.lora_name(i);
            p.A = A[(size_t) i];
            p.B = B[(size_t) i];
        }
        const MetaValue * av = dense.model().meta().get("general.architecture");
        const std::string arch = av && !av->s.empty() ? av->s : "hy4";
        std::string werr;
        if (!xolotl::train::write_lora_gguf(sv, arch, alpha, ad, werr)) { std::fprintf(stderr, "save: %s\n", werr.c_str()); return 1; }
        std::printf("saved %d adapters to %s (LoRA GGUF, arch %s, alpha %.0f)\n", NL, sv, arch.c_str(), alpha);
    } else if (sv) {
        std::ofstream o(sv, std::ios::binary);
        for (int i = 0; i < NL; ++i) {
            int64_t a, b, r;
            dense.lora_shape(i, a, b, r);
            const std::string & nm = dense.lora_name(i);
            const uint32_t len = (uint32_t) nm.size();
            o.write((const char *) &len, 4);
            o.write(nm.data(), len);
            o.write((const char *) &a, 8); o.write((const char *) &b, 8); o.write((const char *) &r, 8);
            o.write((const char *) A[(size_t) i].data(), (std::streamsize) (A[(size_t) i].size() * 4));
            o.write((const char *) B[(size_t) i].data(), (std::streamsize) (B[(size_t) i].size() * 4));
        }
        std::printf("saved %d adapters to %s\n", NL, sv);
    }
    moe.report();
    return 0;
}
