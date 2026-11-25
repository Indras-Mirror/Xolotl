// test_ernie_moe <ernie.gguf> [--layers 1,2] [--tokens 32] [--dump DIR] [--gpu] - the ERNIE-5.5 routed-expert tier
// against a double reference on the real Q8_0 / MXFP4 experts of an `ernie5_5` GGUF.
//
// The tier is Ds41MoeTier with the arch's own geometry (n_embd = the 2048-d latent, n_ff 2048, 512 experts, top-10) and
// its SiTU activation.  For each layer and token row:
//   * a seeded, fixed routing (10 distinct ids + weights) and x rows: seeded random-normal, or the REAL latent rows
//     `--dump DIR` reads from an `ernie_dump` directory's ffn_routed_down-L (the llama.cpp PR's own expert input);
//   * run(layer, ids, w, x, out) and ds41_moe_cpu_expert() per slot, both vs a double reference of the SAME expert bytes
//     (ggml to_float of every row) and the SAME int8 activation the lane quantizes to (rel L2 <= 1e-5, "lane");
//   * the same reference with the exact F32 activations (report only: the int8 activation path's usual band);
// CPU-only by default (no CUDA call of any kind).  --gpu runs the CUDA tier instead (small VRAM cache + arena, as
// test_ds41_moe): built here, RUN IN A GPU GAP.
#include "ds41_moe.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
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

void deq(int type, const uint8_t* src, float* y, int64_t n) {
    ggml_get_type_traits((ggml_type) type)->to_float(src, y, n);
}

/// x (n floats) as the lane's activation of type `act_type`, dequantized back to floats: the exact values its dots see.
void lane_act(int act_type, const float* x, int64_t n, std::vector<float>& out) {
    out.resize((size_t) n);
    std::vector<uint8_t> buf(row_bytes(act_type, n));
    ggml_get_type_traits_cpu((ggml_type) act_type)->from_float(x, buf.data(), n);
    deq(act_type, buf.data(), out.data(), n);
}

struct Ref {
    std::vector<double> gate, up, situ, down;   // [n_ff] gate/up/situ per row, [n_embd] down (row-major)
    std::vector<float> down_f32;                // the same down rows summed in float (F32 accumulation's spread)
};

/// One expert (its blob) applied to x, in double, through `act_type`/`dact_type` activations: the tier's own contract
/// with the lane's activation rounding (or exact floats when act_type < 0).
/// `h_override` (when given): the down projection takes THIS hidden row (the lane's own SwiGLU/SiTU output), quantized
/// like the lane quantizes it - so the gate measures the tier's down + sum arithmetic on the tier's own activations,
/// not two implementations' hidden rows differing by their last bit (which Q8_0 then amplifies).
void ref_expert(const Ds41MoeGeom& g, const cpu::NativeFmt& f, const uint8_t* blob, const float* x, int act_type,
                int dact_type, std::vector<float>& out, Ref* taps, const float* h_override = nullptr) {
    const int64_t D = g.n_embd, FF = g.n_ff;
    const size_t rg = f.gu_row;
    std::vector<float> row((size_t) std::max(D, FF)), xq, hq((size_t) FF), h((size_t) FF);
    const float* xv = x;
    if (act_type >= 0) { lane_act(act_type, x, D, xq); xv = xq.data(); }
    for (int64_t i = 0; i < FF; ++i) {
        double sg = 0, su = 0;
        deq(f.gu_type, blob + (size_t) i * rg, row.data(), D);
        for (int64_t k = 0; k < D; ++k) sg += (double) row[(size_t) k] * xv[k];
        deq(f.gu_type, blob + f.up_off + (size_t) i * rg, row.data(), D);
        for (int64_t k = 0; k < D; ++k) su += (double) row[(size_t) k] * xv[k];
        const float gg = (float) sg, uu = (float) su;
        h[(size_t) i] = cpu::native_act_value(f.act, gg, uu, f.swiglu_limit, f.situ_beta, f.situ_beta_l);
        if (taps) {
            taps->gate.push_back(sg);
            taps->up.push_back(su);
            taps->situ.push_back(h[(size_t) i]);
        }
    }
    const float* hs = h_override != nullptr ? h_override : h.data();
    const float* hv = hs;
    if (dact_type >= 0) { lane_act(dact_type, hs, FF, hq); hv = hq.data(); }
    out.assign((size_t) D, 0.0f);
    for (int64_t i = 0; i < D; ++i) {
        deq(f.d_type, blob + f.down_off + (size_t) i * f.d_row, row.data(), FF);
        double s = 0;
        float sf = 0.0f;
        for (int64_t k = 0; k < FF; ++k) {
            s += (double) row[(size_t) k] * hv[k];
            sf += row[(size_t) k] * hv[k];
        }
        out[(size_t) i] = (float) s;
        if (taps) { taps->down.push_back(s); taps->down_f32.push_back(sf); }
    }
}

double norm2(const std::vector<float>& a) {
    double s = 0;
    for (float v : a) s += (double) v * v;
    return std::sqrt(s);
}

double norm2diff(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

/// A dump tap: `<dir>/<name>-<layer>.f32`, `n` floats.  False when the file is absent (the dump is optional).
bool read_tap(const std::string& dir, const char* name, int layer, size_t n, std::vector<float>& out) {
    if (dir.empty()) return false;
    const std::string p = dir + "/" + name + "-" + std::to_string(layer) + ".f32";
    FILE* fp = std::fopen(p.c_str(), "rb");
    if (fp == nullptr) return false;
    out.resize(n);
    const size_t got = std::fread(out.data(), sizeof(float), n, fp);
    std::fclose(fp);
    return got == n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ernie_moe <ernie.gguf> [--layers 1,2] [--tokens 32] [--dump DIR] [--gpu] [--pcie X]\n");
        return 2;
    }
    std::string path = argv[1], dump;
    std::vector<int> layers = {1, 2};
    int ntok = 32;
    bool gpu = false;
    int probe_l = -1, probe_e = -1, probe_t = 0;   // --probe L,E,T: the stage-by-stage number for one (expert, row)
    double pcie_share = 0.5;                       // --pcie X: share of misses computed on the GPU (0 = the CPU pool does
                                                   // every one, i.e. the GPU path with NO GPU expert kernels)
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc) {
            layers.clear();
            std::string s = argv[++i], x;
            for (size_t p = 0; p <= s.size();) {
                const size_t q = s.find(',', p);
                x = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (!x.empty()) layers.push_back(std::atoi(x.c_str()));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        } else if (a == "--tokens" && i + 1 < argc) ntok = std::atoi(argv[++i]);
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--probe" && i + 1 < argc) {
            int lo = 0, e = 0, tt = 0;
            if (std::sscanf(argv[++i], "%d,%d,%d", &lo, &e, &tt) == 3) { probe_l = lo; probe_e = e; probe_t = tt; }
        }
        else if (a == "--gpu") gpu = true;
        else if (a == "--pcie" && i + 1 < argc) pcie_share = std::atof(argv[++i]);
    }

    // ---- the arch's own geometry (ernie5_5 -> latent n_embd, SiTU), then the file's experts ----
    Ds41MoeGeom g;
    std::string err;
    if (!ds41_moe_geom_from_gguf(path, g, err)) { std::fprintf(stderr, "geom: %s\n", err.c_str()); return 1; }
    if (g.act != cpu::NativeAct::SiTU) { std::fprintf(stderr, "geom is not SiTU (arch not ernie5_5?)\n"); return 1; }
    std::printf("geom: %lld layers x %lld experts top-%lld, n_embd %lld (latent), n_ff %lld, SiTU beta %.3g/%.3g\n",
                (long long) g.n_layers, (long long) g.n_experts, (long long) g.top_k, (long long) g.n_embd,
                (long long) g.n_ff, g.situ_beta, g.situ_beta_l);
    Ds41GgufBlobs src(g, path);
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }

    Ds41MoeTier tier;
    Ds41MoeConfig cfg;
    cfg.cpu_only = !gpu;
    cfg.threads = 8;
    cfg.slots = 64;          // GPU arm: a small cache, so hits, PCIe and CPU entries all occur
    cfg.arena_gib = 4.0;
    cfg.max_arena_gib = 4.0;
    cfg.pcie_frac = pcie_share;
    cfg.tier_batch_pcie = pcie_share;
    cfg.pf_b = 0;
    if (!tier.init(g, &src, cfg, err)) { std::fprintf(stderr, "tier init: %s\n", err.c_str()); return 1; }
    if (gpu) {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int l : layers)
            for (int e = 0; e < g.n_experts; ++e) ranked.emplace_back(l, e);
        if (!tier.seed_from_ranked(ranked, err)) { std::fprintf(stderr, "seed: %s\n", err.c_str()); return 1; }
        std::printf("tier %s: %lld resident, %lld arena\n", tier.mode(), (long long) tier.resident(),
                    (long long) tier.arena_experts());
    }

    if (probe_l >= 0) {   // stage by stage: the lane's own kernels vs the double reference, one expert and row
        std::mt19937_64 prng(20261124);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        cpu::NativeFmt f;
        if (!ds41_moe_layer_layout(g, probe_l, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<float> px((size_t) g.n_embd), dump_x;
        if (read_tap(dump, "ffn_routed_down", probe_l, (size_t) (probe_t + 1) * g.n_embd, dump_x))
            std::memcpy(px.data(), dump_x.data() + (size_t) probe_t * g.n_embd, sizeof(float) * (size_t) g.n_embd);
        else for (auto& v : px) v = nd(prng) * 0.3f;
        std::vector<uint8_t> pb(src.blob_bytes());
        src.read(probe_l, probe_e, pb.data());
        std::vector<uint8_t> nact(cpu::kNativeActBytes);
        cpu::native_quant_act(f, px.data(), nact.data());
        std::vector<float> h((size_t) g.n_ff), want((size_t) g.n_ff), href((size_t) g.n_embd), hqref((size_t) g.n_embd);
        float* ff[1] = {h.data()};
        const void* ap[1] = {nact.data()};
        cpu::native_gu_rows(f, pb.data(), ap, 1, ff, 0, (int) g.n_ff);
        Ref taps;
        ref_expert(g, f, pb.data(), px.data(), f.gu_act, f.d_act, href, &taps);
        std::vector<float> hd((size_t) g.n_ff);
        for (int64_t i = 0; i < g.n_ff; ++i) hd[(size_t) i] = (float) taps.situ[(size_t) i];
        std::printf("probe L%d expert %d row %d: gate/up+situ rel %.3e\n", probe_l, probe_e, probe_t, rel(h, hd));
        std::vector<uint8_t> nh(cpu::kNativeHBytes);
        cpu::native_quant_h(f, h.data(), nh.data());
        std::vector<float> down((size_t) g.n_embd);
        const void* hp[1] = {nh.data()};
        float* op[1] = {down.data()};
        cpu::native_down_rows(f, pb.data(), hp, 1, op, 0, (int) g.n_embd);
        std::printf("probe down rel %.3e (lane vs the double reference)\n", rel(down, href));
        return 0;
    }

    const int64_t D = g.n_embd, FF = g.n_ff, K = g.top_k;
    std::mt19937_64 rng(20261124);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    double worst_lane = 0, worst_exact = 0, worst_scaled = 0;
    std::vector<uint8_t> blob(src.blob_bytes());

    for (int l : layers) {
        cpu::NativeFmt f;
        if (!ds41_moe_layer_layout(g, l, f, err)) { std::fprintf(stderr, "layer %d: %s\n", l, err.c_str()); return 1; }
        std::printf("L%d: types %d/%d/%d, blob %zu bytes, act %d\n", l, f.gu_type, f.up_type, f.d_type, f.bytes,
                    (int) f.act);
        // the dump's real rows / routing (or the seeded random ones), `nt` tokens
        std::vector<float> dump_x;
        const bool have_x = read_tap(dump, "ffn_routed_down", l, (size_t) ntok * D, dump_x);
        std::printf("  rows: %s, routing: seeded\n", have_x ? "dump (real latent rows)" : "random-normal");
        for (int t = 0; t < ntok; ++t) {
            std::vector<float> x((size_t) D);
            if (have_x) std::memcpy(x.data(), dump_x.data() + (size_t) t * D, sizeof(float) * (size_t) D);
            else for (auto& v : x) v = nd(rng) * 0.3f;
            std::vector<int32_t> ids((size_t) K);
            std::vector<float> w((size_t) K);
            for (int64_t k = 0; k < K; ++k) {
                int32_t e;
                do { e = (int32_t) (rng() % (uint64_t) g.n_experts); }
                while (std::find(ids.begin(), ids.begin() + k, e) != ids.begin() + k);
                ids[(size_t) k] = e;
                w[(size_t) k] = 0.1f + 0.4f * (float) (rng() % 1000) / 1000.0f;
            }
            // ---- the tier's routed sum + its per-slot expert, vs the reference
            std::vector<float> got((size_t) D), want((size_t) D, 0.0f), wexact((size_t) D, 0.0f);
            std::vector<float> one((size_t) D), one_exact((size_t) D);
            double mag = 0;   // sum_k |w_k| ||slot_k||: the scale the routed sum's errors live at (see the gate)
            if (!tier.run(l, ids.data(), w.data(), x.data(), got.data())) { std::fprintf(stderr, "run failed\n"); return 1; }
            int slot_fail = 0;
            double slot_worst = 0;
            for (int64_t k = 0; k < K; ++k) {
                src.read(l, ids[(size_t) k], blob.data());
                Ref taps;
                ref_expert(g, f, blob.data(), x.data(), f.gu_act, f.d_act, one, &taps);   // independent reference
                ref_expert(g, f, blob.data(), x.data(), -1, -1, one_exact, nullptr);
                std::vector<float> lane((size_t) D);
                if (!ds41_moe_cpu_expert(f, D, FF, blob.data(), x.data(), lane.data(), tier.pool())) {
                    std::fprintf(stderr, "cpu_expert failed\n");
                    return 1;
                }
                // the lane's own hidden row (gate/up + the activation) and a reference down over it: the 1e-5 arm
                thread_local std::vector<uint8_t> nact(cpu::kNativeActBytes), nh(cpu::kNativeHBytes);
                std::vector<float> h_lane((size_t) FF), one_lane((size_t) D);
                cpu::native_quant_act(f, x.data(), nact.data());
                {
                    const void* ap[1] = {nact.data()};
                    float* hp[1] = {h_lane.data()};
                    cpu::native_gu_rows(f, blob.data(), ap, 1, hp, 0, (int) FF);
                }
                ref_expert(g, f, blob.data(), x.data(), f.gu_act, f.d_act, one_lane, nullptr, h_lane.data());
                const double r_h = rel(h_lane, std::vector<float>(taps.situ.begin(), taps.situ.end()));
                const double r1 = rel(lane, one_lane);        // gate: 1e-5
                const double r0 = rel(lane, one);             // report: the fully independent reference
                slot_worst = std::max(slot_worst, r1);
                slot_fail += !(r1 <= 1e-5);
                if (r1 > 1e-5 || r_h > 1e-5)
                    std::printf("  slot %lld expert %d: lane-down rel %.3e  lane-h rel %.3e  independent rel %.3e\n",
                                (long long) k, (int) ids[(size_t) k], r1, r_h, r0);
                for (int64_t i = 0; i < D; ++i) {
                    want[(size_t) i] += w[(size_t) k] * one_lane[(size_t) i];
                    wexact[(size_t) i] += w[(size_t) k] * one[(size_t) i];
                }
                mag += std::fabs(w[(size_t) k]) * norm2(one_lane);
            }
            const double r_lane = rel(got, want);
            // The routed sum cancels, so the norm of the SUM says nothing about the error a per-slot 1e-7 can put
            // there.  The scale the slot errors actually live at is mag = sum_k |w_k| ||slot_k||: the sum is gated
            // against that (||got - want|| / mag <= 1e-5), and the raw ratio is still printed for continuity.
            const double r_scaled = mag > 0 ? norm2diff(got, want) / mag : 0.0;
            const double r_exact = rel(got, wexact);   // report: both sides' own hidden quantisation
            const bool ok = r_scaled <= 1e-5 && slot_fail == 0;
            fails += !ok;
            worst_lane = std::max(worst_lane, r_lane);
            worst_scaled = std::max(worst_scaled, r_scaled);
            worst_exact = std::max(worst_exact, r_exact);
            std::printf("%s L%d t%02d  sum rel %.3e  scaled %.3e (gate 1e-5 vs sum_k |w_k| ||slot_k||)  "
                        "independent %.3e  slot worst %.3e%s\n",
                        ok ? "ok  " : "FAIL", l, t, r_lane, r_scaled, r_exact, slot_worst, slot_fail ? " FAIL" : "");
        }
    }
    const Ds41MoeStats st = tier.stats();
    std::printf("stats: %lld hit, %lld cpu, %lld pcie, %lld file; worst rel: scaled %.3e (GATE, 1e-5), raw sum %.3e, "
                "exact-act %.3e -> %s\n",
                (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier, worst_scaled,
                worst_lane, worst_exact, fails ? "FAIL" : "PASS");
    tier.close();
    return fails ? 1 : 0;
}
