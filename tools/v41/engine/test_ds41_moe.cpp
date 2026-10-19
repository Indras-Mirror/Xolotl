// test_ds41_moe <file.gguf> [--layers 0,1] [--tokens 8] [--gpu] - the V4.1 expert tier against a dequant + F32
// reference on REAL mixed-tier experts: for random activations and router picks (forced to cover every format class of
// the layer), Ds41MoeTier::run's weighted sum vs sum_k w_k * expert_k(x) with every row dequantized (ggml to_float /
// PTQ1_0 dequant) and an F32 SwiGLU (clamped at the model's limit).  CPU-only by default (no CUDA call); --gpu runs the
// CUDA tier with a small VRAM cache + the arena so hits, PCIe and CPU entries all occur (binary from build-ds4-gpu).
// The only permitted difference is the kernels' own 8-bit activation rounding (~1e-2 relative), as test_ds4_moe.
#include "ds41_moe.hpp"

#include "strata/kernels/cpu/ptq1_0.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    block_geometry((uint32_t) type, be, bb);
    return (size_t) (n / be) * (size_t) bb;
}

void deq(int type, const uint8_t * src, float * y, int64_t n) {
    if (type == 143) kernels::cpu::ptq1_0_dequant(src, y, n);
    else ggml_get_type_traits((ggml_type) type)->to_float(src, y, n);
}

// out = expert(blob) applied to x, F32 (the blob is [gate | up | down] of class c)
void ref_expert(const Ds41MoeGeom & g, int c, const uint8_t * blob, const float * x, float * out) {
    const int64_t D = g.n_embd, FF = g.n_ff;
    const int gt = g.cls_gu[(size_t) c], dt = g.cls_d[(size_t) c];
    const size_t rg = row_bytes(gt, D), rd = row_bytes(dt, FF);
    std::vector<float> row((size_t) std::max(D, FF)), a((size_t) FF);
    const float lim = g.swiglu_limit;
    for (int64_t i = 0; i < FF; ++i) {
        double sg = 0, su = 0;
        deq(gt, blob + (size_t) i * rg, row.data(), D);
        for (int64_t k = 0; k < D; ++k) sg += (double) row[(size_t) k] * x[k];
        deq(gt, blob + (size_t) FF * rg + (size_t) i * rg, row.data(), D);
        for (int64_t k = 0; k < D; ++k) su += (double) row[(size_t) k] * x[k];
        float gg = (float) sg, uu = (float) su;
        if (std::isfinite(lim)) { uu = std::min(std::max(uu, -lim), lim); gg = std::min(gg, lim); }
        a[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu;
    }
    for (int64_t i = 0; i < D; ++i) {
        deq(dt, blob + 2 * (size_t) FF * rg + (size_t) i * rd, row.data(), FF);
        double s = 0;
        for (int64_t k = 0; k < FF; ++k) s += (double) row[(size_t) k] * a[(size_t) k];
        out[i] = (float) s;
    }
}

double rel(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_ds41_moe <file.gguf> [--layers 0,1] [--tokens 8] [--gpu] [--chunk N] [--chunk-mmq] [--multi N] [--remote N [--remote-gib G]]\n"); return 2; }
    std::string path = argv[1];
    int nchunk = 0;   // --chunk N: also N tokens at once through run_chunk (the prompt path), each row vs the reference
    std::vector<int> layers = { 0 };
    int ntok = 8;
    bool gpu = false, chunk_mmq = false;
    int remote = -1, nmulti = 0;   // --remote N [--remote-gib G]: a second card holds experts after the VRAM head;
    double remote_gib = 2.0;       // --multi N: N tokens at once through run_multi (the serve / verify path)
    bool noref = false;            // --no-ref: the chunk pass is only timed (3 runs), not checked
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc) {
            layers.clear();
            std::stringstream ss(argv[++i]);
            std::string x;
            while (std::getline(ss, x, ',')) layers.push_back(std::atoi(x.c_str()));
        } else if (a == "--tokens" && i + 1 < argc) ntok = std::atoi(argv[++i]);
        else if (a == "--gpu") gpu = true;
        else if (a == "--chunk" && i + 1 < argc) nchunk = std::atoi(argv[++i]);
        else if (a == "--chunk-mmq") chunk_mmq = true;   // the prompt path's MMQ (PTQ1_0 via its Q2_0 transcode)
        else if (a == "--remote" && i + 1 < argc) remote = std::atoi(argv[++i]);
        else if (a == "--remote-gib" && i + 1 < argc) remote_gib = std::atof(argv[++i]);
        else if (a == "--multi" && i + 1 < argc) nmulti = std::atoi(argv[++i]);
        else if (a == "--no-ref") noref = true;
    }
    Ds41FileBlobs src(path);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & g = src.geom();
    std::printf("%lld layers, %zu classes:", (long long) g.n_layers, g.cls_gu.size());
    for (size_t c = 0; c < g.cls_gu.size(); ++c) std::printf(" %d/%d", g.cls_gu[c], g.cls_d[c]);
    std::printf("\n");
    Ds41MoeTier tier;
    Ds41MoeConfig cfg;
    cfg.cpu_only = !gpu;
    cfg.chunk_mmq = chunk_mmq;
    cfg.threads = 8;
    cfg.slots = 64;              // a small cache: most picks miss -> CPU / PCIe paths run too
    cfg.arena_gib = 4.0;
    cfg.max_arena_gib = 4.0;
    cfg.pcie_frac = 0.5;
    cfg.tier_batch_pcie = 0.5;   // run_multi too: half the arena misses go to the CPU pool
    cfg.pf_b = 0;
    cfg.remote_device = remote;
    cfg.remote_gib = remote_gib;
    if (!tier.init(g, &src, cfg, err)) { std::fprintf(stderr, "tier init: %s\n", err.c_str()); return 1; }
    if (gpu) {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int l : layers) for (int e = 0; e < g.n_experts; ++e) ranked.emplace_back(l, e);
        if (!tier.seed_from_ranked(ranked, err)) { std::fprintf(stderr, "seed: %s\n", err.c_str()); return 1; }
        std::printf("tier %s: %lld resident, %lld remote, %lld arena\n", tier.mode(), (long long) tier.resident(),
                    (long long) tier.remote_experts(), (long long) tier.arena_experts());
    }
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t D = g.n_embd, K = g.top_k;
    int fails = 0;
    double worst = 0;
    std::vector<uint8_t> blob(src.blob_bytes());
    for (int l : layers) {
        // per class, the experts of that class
        std::vector<std::vector<int>> by((size_t) g.cls_gu.size());
        for (int e = 0; e < g.n_experts; ++e) by[(size_t) g.cls(l, e)].push_back(e);
        for (int t = 0; t < ntok; ++t) {
            std::vector<float> x((size_t) D);
            for (auto & v : x) v = nd(rng) * 0.3f;
            // picks: cycle through the layer's classes so every class (and mixtures) appear
            std::vector<int32_t> ids((size_t) K);
            std::vector<float> w((size_t) K);
            int ci = t;
            for (int64_t k = 0; k < K; ++k) {
                int tries = 0;
                while (by[(size_t) (ci % by.size())].empty() && tries++ < 16) ++ci;
                const auto & v = by[(size_t) (ci % by.size())];
                int32_t e;
                do { e = v[(size_t) (rng() % v.size())]; } while (std::find(ids.begin(), ids.begin() + k, e) != ids.begin() + k && v.size() > (size_t) k);
                ids[(size_t) k] = e;
                w[(size_t) k] = 0.1f + 0.4f * (float) (rng() % 1000) / 1000.0f;
                ++ci;
            }
            std::vector<float> got((size_t) D), want((size_t) D, 0.0f), one((size_t) D);
            if (!tier.run(l, ids.data(), w.data(), x.data(), got.data())) { std::fprintf(stderr, "run failed\n"); return 1; }
            for (int64_t k = 0; k < K; ++k) {
                src.read(l, ids[(size_t) k], blob.data());
                ref_expert(g, g.cls(l, ids[(size_t) k]), blob.data(), x.data(), one.data());
                for (int64_t i = 0; i < D; ++i) want[(size_t) i] += w[(size_t) k] * one[(size_t) i];
            }
            const double r = rel(got, want);
            worst = std::max(worst, r);
            const bool ok = r < 3e-2;
            fails += !ok;
            std::printf("%s L%d tok %d classes", ok ? "ok  " : "FAIL", l, t);
            for (int64_t k = 0; k < K; ++k) std::printf(" %d", g.cls(l, ids[(size_t) k]));
            std::printf("  rel %.3e\n", r);
        }
    }
    // ---- serve / verify path: N tokens per layer through run_multi (random picks over all experts)
    for (int l : (nmulti > 0 ? layers : std::vector<int>{})) {
        const int nm = std::min(nmulti, Ds41MoeTier::kMaxMulti);
        std::vector<float> X((size_t) nm * D), W((size_t) nm * K), OUT((size_t) nm * D);
        std::vector<int32_t> IDS((size_t) nm * K);
        for (int rep = 0; rep < 8; ++rep) {
            for (auto & v : X) v = nd(rng) * 0.3f;
            for (int t = 0; t < nm; ++t)
                for (int64_t k = 0; k < K; ++k) {
                    int32_t e;
                    do { e = (int32_t) (rng() % (uint64_t) g.n_experts); }
                    while (std::find(IDS.begin() + (size_t) t * K, IDS.begin() + (size_t) t * K + k, e) != IDS.begin() + (size_t) t * K + k);
                    IDS[(size_t) t * K + k] = e;
                    W[(size_t) t * K + k] = 0.1f + 0.4f * (float) (rng() % 1000) / 1000.0f;
                }
            if (!tier.run_multi(l, nm, IDS.data(), W.data(), X.data(), OUT.data())) { std::fprintf(stderr, "run_multi failed\n"); return 1; }
            int nrem = 0;
            for (int32_t e : IDS) nrem += tier.in_remote(l, e);
            double mw = 0;
            for (int t = 0; t < nm; ++t) {
                std::vector<float> got(OUT.begin() + (size_t) t * D, OUT.begin() + (size_t) (t + 1) * D), want((size_t) D, 0.0f), one((size_t) D);
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t e = IDS[(size_t) t * K + k];
                    src.read(l, e, blob.data());
                    ref_expert(g, g.cls(l, e), blob.data(), X.data() + (size_t) t * D, one.data());
                    for (int64_t i = 0; i < D; ++i) want[(size_t) i] += W[(size_t) t * K + k] * one[(size_t) i];
                }
                mw = std::max(mw, rel(got, want));
            }
            worst = std::max(worst, mw);
            fails += !(mw < 3e-2);
            std::printf("%s L%d multi %d tokens (%d remote entries): worst rel %.3e\n", mw < 3e-2 ? "ok  " : "FAIL", l, nm, nrem, mw);
        }
    }
    // ---- prompt path: N tokens per layer through run_chunk (random picks over all experts: every class, mixed)
    for (int l : (nchunk > 0 ? layers : std::vector<int>{})) {
        std::vector<float> X((size_t) nchunk * D), W((size_t) nchunk * K), OUT((size_t) nchunk * D);
        std::vector<int32_t> IDS((size_t) nchunk * K);
        for (auto & v : X) v = nd(rng) * 0.3f;
        for (int t = 0; t < nchunk; ++t)
            for (int64_t k = 0; k < K; ++k) {
                int32_t e;
                do { e = (int32_t) (rng() % (uint64_t) g.n_experts); }
                while (std::find(IDS.begin() + (size_t) t * K, IDS.begin() + (size_t) t * K + k, e) != IDS.begin() + (size_t) t * K + k);
                IDS[(size_t) t * K + k] = e;
                W[(size_t) t * K + k] = 0.1f + 0.4f * (float) (rng() % 1000) / 1000.0f;
            }
        if (noref) {
            for (int rep = 0; rep < 3; ++rep) {
                const auto c0 = std::chrono::steady_clock::now();
                if (!tier.run_chunk(l, nchunk, IDS.data(), W.data(), X.data(), OUT.data())) { std::fprintf(stderr, "run_chunk failed\n"); return 1; }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
                int nrem = 0;
                for (int32_t e : IDS) nrem += tier.in_remote(l, e);
                std::printf("time L%d chunk %d tokens (%d remote entries): %.1f ms\n", l, nchunk, nrem, ms);
            }
            continue;
        }
        if (!tier.run_chunk(l, nchunk, IDS.data(), W.data(), X.data(), OUT.data())) { std::fprintf(stderr, "run_chunk failed\n"); return 1; }
        int cf = 0, nan_rows = 0;
        double cw = 0;
        for (int t = 0; t < nchunk; ++t) {
            std::vector<float> got(OUT.begin() + (size_t) t * D, OUT.begin() + (size_t) (t + 1) * D), want((size_t) D, 0.0f), one((size_t) D);
            bool fin = true;
            for (float v : got) fin &= std::isfinite(v);
            nan_rows += !fin;
            for (int64_t k = 0; k < K; ++k) {
                const int32_t e = IDS[(size_t) t * K + k];
                src.read(l, e, blob.data());
                ref_expert(g, g.cls(l, e), blob.data(), X.data() + (size_t) t * D, one.data());
                for (int64_t i = 0; i < D; ++i) want[(size_t) i] += W[(size_t) t * K + k] * one[(size_t) i];
            }
            const double r = fin ? rel(got, want) : 1e9;
            if (!(r < 3e-2)) {
                std::printf("  row %d rel %.3e:", t, r);
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t e = IDS[(size_t) t * K + k];
                    std::printf(" e%d/c%d", e, g.cls(l, e));
                }
                std::printf("\n");
            }
            cw = std::max(cw, r);
            cf += !(r < 3e-2);
        }
        worst = std::max(worst, cw);
        fails += cf;
        std::printf("%s L%d chunk %d tokens: %d rows off, %d non-finite, worst rel %.3e\n", cf ? "FAIL" : "ok  ", l, nchunk, cf, nan_rows, cw);
    }
    const Ds41MoeStats st = tier.stats();
    std::printf("stats: %lld hit, %lld cpu, %lld pcie, %lld file; worst rel %.3e -> %s\n", (long long) st.hits, (long long) st.cpu,
                (long long) st.pcie, (long long) st.file_tier, worst, fails ? "FAIL" : "PASS");
    tier.close();
    return fails ? 1 : 0;
}
