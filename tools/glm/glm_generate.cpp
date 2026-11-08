// tools/glm/glm_generate.cpp - the Strata GLM-5.3-Flash engine end to end: GlmDense (attention + router + shared
// expert/dense FFN, CUDA or CPU ggml backend) + Ds4MoeTier (routed experts: VRAM cache, prefetch, CPU pool, PCIe share,
// host arena) + greedy/temperature sampling.  Token ids in, token ids out (one per line on stdout).
//
//   glm_generate -m model.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                [--slots auto|N] [--arena-gib 60] [--profile routes.bin] [--vram-lru] [--route-bias 0.05]
//                [--ctx 2048] [--temp 0] [--stop id,id] [--ppl] [--dump-logits f.f32] [--dump-routes f.bin]
//
// Per token, per layer: predict(l) -> tier.prefetch(l); attn_router(l); tier.run(l) (MoE layers only); finish_layer(l).
// The prompt goes through the same one-token path (phase 1: no prompt chunks yet).
// Load the real model ONLY through tools/ds4/memguard.sh.
//
// --dump-routes FILE: every token's routed ids, [512-token block][routed layer][token][top_k] u16 (the format
// Ds4MoeTier::seed_from_routes(file, 512) ranks) - a GLM routing profile for --profile.  Without --profile the
// arena and VRAM cache are filled in (layer, expert) index order.
#include "glm_dense.hpp"
#include "glm_lora.hpp"
#include "ds4_moe.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model, ids_csv, ids_file, profile, dump_logits, dump_routes, lora, saliency, prune;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;   // 0 = auto
    double vram_grow_keep = 0.0, vram_margin_gib = 1.0, pcie = 0.25, pf_b = 1.43, arena_gib = 60.0;
    int threads = 0;
    int64_t ctx = 0;
    float route_bias_file = 0.0f;   // --route-bias-file X: DECODE-only selection penalty -X on file-tier experts
                                    // (neither VRAM cache nor RAM arena) - soft, every expert stays reachable and keeps
                                    // its LoRA delta (unlike --skip-*), the ds41 decode recipe (route-bias-file 0.3)
    float temp = 0.0f, route_bias = 0.0f, prune_penalty = 0.0f, skip_miss = 0.0f, skip_file = 0.0f, skip_file_chunk = 0.0f, arena_admit = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool serve = false;              // --serve: Strata's engine line protocol on stdin/stdout (serve/server.py)
    bool lora_exps = false;          // --lora-exps: also apply the adapter's routed-expert (ffn_*_exps) deltas
    bool renorm_skip = false;        // --renorm-skip: rescale the MoE sum by the surviving weight fraction after a drop
    bool arena_lazy = false;         // --arena-lazy: fill the RAM arena in the background; serve (file tier) meanwhile
    bool ppl = false, vram_lru = false, arena_adapt = false, arena_skip = false;
    int prefill_chunk = 0;           // --prefill-chunk N: the prompt in passes of N tokens (0 = the decode loop)
    bool chunk_mmq = false, chunk_prestage = false, allow_long = false;
    int dense_requant = -1;          // --dense-requant q4_k|q5_k: the dense half's Q6_K matrices at load
    std::string census;              // --census FILE: persistent per-(layer, expert) decode use counts (helios-style)
    float vram_pin = 0.0f;           // --vram-pin F: with --vram-lru, the top F of the seeded slots are never evicted
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

bool parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--vram-grow") a.vram_grow_keep = std::atof(next().c_str());
        else if (k == "--pcie") { const std::string v = next(); a.pcie = v == "auto" ? -1.0 : std::atof(v.c_str()); }   // auto: balance the card and the pool (FreeToken)
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--profile") a.profile = next();
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-routes") a.dump_routes = next();
        else if (k == "--lora") a.lora = next();
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--route-bias-file") a.route_bias_file = (float) std::atof(next().c_str());
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--serve") a.serve = true;
        else if (k == "--lora-exps") a.lora_exps = true;
        else if (k == "--arena-skip-resident") a.arena_skip = true;
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--chunk-prestage") a.chunk_prestage = true;
        else if (k == "--allow-long-ctx") a.allow_long = true;
        else if (k == "--saliency") a.saliency = next();
        else if (k == "--prune") a.prune = next();
        else if (k == "--skip-miss") a.skip_miss = (float) std::atof(next().c_str());
        else if (k == "--skip-file") a.skip_file = (float) std::atof(next().c_str());
        else if (k == "--skip-file-prefill") a.skip_file_chunk = (float) std::atof(next().c_str());
        else if (k == "--renorm-skip") a.renorm_skip = true;
        else if (k == "--arena-lazy") a.arena_lazy = true;
        else if (k == "--census") a.census = next();
        else if (k == "--dense-requant") { const std::string v = next(); a.dense_requant = v == "q4_k" ? (int) GGML_TYPE_Q4_K : v == "q5_k" ? (int) GGML_TYPE_Q5_K : -1; }
        else if (k == "--vram-pin") a.vram_pin = (float) std::atof(next().c_str());
        else if (k == "--arena-admit") a.arena_admit = (float) std::atof(next().c_str());
        else if (k == "--prune-penalty") a.prune_penalty = (float) std::atof(next().c_str());
        else { std::fprintf(stderr, "glm_generate: unknown argument %s\n", k.c_str()); return false; }
    }
    // (--serve keeps --vram-grow since s26: the cache shrinks before every long prompt and grows back after it -
    // Ds4MoeTier::shrink_cache / grow_cache - so each request's chunks get their VRAM and its decode the full cache)
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int sample(const float * logits, int n_vocab, float temp, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const float mx = *std::max_element(logits, logits + n_vocab);
    std::vector<double> p((size_t) n_vocab);
    double z = 0;
    for (int i = 0; i < n_vocab; ++i) z += (p[(size_t) i] = std::exp((double) (logits[i] - mx) / temp));
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n_vocab; ++i) if ((r -= p[(size_t) i]) <= 0) return i;
    return n_vocab - 1;
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best)
int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    double z = 0;
    for (int i = 0; i < K; ++i) z += (p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp));
    int n = K;
    if (min_p > 0.0f)
        while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        double c = 0;
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("GLM_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: glm_generate -m model.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu]\n"
                             "       [--experts gpu|cpu] [--slots auto|N] [--arena-gib 60] [--profile routes.bin] [--vram-lru]\n"
                             "       [--route-bias X] [--route-bias-file X] [--ctx N] [--temp 0] [--stop id,id] [--ppl] [--dump-logits f] [--dump-routes f]\n"
                             "       [--lora adapter.gguf] [--prefill-chunk N [--chunk-mmq] [--chunk-prestage]] [--allow-long-ctx]\n"
                             "       [--prune f [--prune-penalty X]] [--arena-adapt [--arena-admit HALF_LIFE_TOKENS]]\n"
                             "       [--skip-miss T] [--skip-file T] [--skip-file-prefill T] [--renorm-skip] [--arena-lazy] [--vram-grow KEEP_GIB]\n"
                             "       [--lora-exps] (with --lora: also apply the routed-expert ffn_*_exps deltas)\n");
        return 2;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "glm_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        const size_t n = (size_t) f.tellg() / 4;
        f.seekg(0);
        prompt.resize(n);
        f.read((char *) prompt.data(), (std::streamsize) (n * 4));
    }
    if (prompt.empty() && !a.serve) { std::fprintf(stderr, "glm_generate: empty prompt\n"); return 2; }

    ggml_backend_t be = ggml_backend_init_by_type(a.backend == "cuda" ? GGML_BACKEND_DEVICE_TYPE_GPU
                                                                      : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "glm_generate: no %s backend in this build\n", a.backend.c_str()); return 2; }
    std::fprintf(stderr, "dense backend: %s\n", ggml_backend_name(be));

    const double t_load0 = now_ms();
    strata::glm::LoraAdapter lora_ad;
    if (!a.lora.empty()) {
        std::string lerr;
        if (!lora_ad.load(a.lora, lerr)) { std::fprintf(stderr, "glm_generate: lora: %s\n", lerr.c_str()); return 1; }
        std::fprintf(stderr, "lora: %s - dense half will apply %sattn_output + %sffn_down_shexp; routed-expert exps are "
                             "NOT applied yet (CPU-path slice pending)\n",
                     a.lora.c_str(), lora_ad.any(strata::glm::LoraAdapter::ATTN_OUT) ? "" : "(no) ",
                     lora_ad.any(strata::glm::LoraAdapter::SHEXP_DOWN) ? "" : "(no) ");
    }
    GlmDense dense;
    GlmDenseConfig dc;
    dc.backend = be;
    dc.lora = a.lora.empty() ? nullptr : &lora_ad;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 8;
    dc.max_tokens = std::max(1, a.prefill_chunk);
    dc.allow_long_ctx = a.allow_long;
    dc.dense_requant = a.dense_requant;
    std::string err;
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "glm_generate: dense init: %s\n", err.c_str()); return 1; }
    const GlmGeometry & G = dense.geom();
    std::fprintf(stderr, "glm5-next: %lld layers (%lld KDA, %lld MLA), %lld experts top-%lld, vocab %lld; dense half loaded "
                         "in %.1f s\n", (long long) G.n_layer,
                 (long long) std::count(G.is_kda.begin(), G.is_kda.end(), true),
                 (long long) std::count(G.is_kda.begin(), G.is_kda.end(), false), (long long) G.n_expert,
                 (long long) G.n_expert_used, (long long) G.vocab, (now_ms() - t_load0) / 1000.0);
    // --lora-exps: the adapter's routed-expert (ffn_gate/up/down_exps) deltas, flattened per layer for the tier.
    // GlmDense applies only the solo targets (attn_output, ffn_down_shexp) via dc.lora; without this the 78
    // routed-expert pairs - the bulk of the abliteration - are silently dropped.  These arrays live for the whole
    // run (the tier copies them to the device at init); a layer with no expert deltas stays all-null.
    std::vector<strata::ds4::Ds4MoeLoraHost> lora_host;
    std::deque<std::vector<float>> lora_buf;   // deque: emplace_back never moves the earlier arrays
    if (a.lora_exps) {
        if (a.lora.empty()) { std::fprintf(stderr, "glm_generate: --lora-exps needs --lora ADAPTER\n"); return 2; }
        const int64_t NE = G.n_expert;
        lora_host.assign((size_t) G.n_layer, strata::ds4::Ds4MoeLoraHost());
        auto flat = [&](int64_t l, strata::glm::LoraAdapter::Mod m, bool is_a) -> const float* {
            const strata::glm::Lora1* v0 = lora_ad.exps(l, m, 0);
            if (!v0) return nullptr;
            const int64_t k = (int64_t) (is_a ? v0->a.size() : v0->b.size());
            lora_buf.emplace_back((size_t) (NE * k));
            std::vector<float>& out = lora_buf.back();
            for (int64_t e = 0; e < NE; ++e) {
                const strata::glm::Lora1* v = lora_ad.exps(l, m, e);
                if (!v) return nullptr;
                const std::vector<float>& src = is_a ? v->a : v->b;
                if ((int64_t) src.size() != k) return nullptr;
                std::copy(src.begin(), src.end(), out.begin() + (size_t) (e * k));
            }
            return out.data();
        };
        int nl = 0;
        for (int64_t l = 0; l < G.n_layer; ++l) {
            if (!lora_ad.has(l, strata::glm::LoraAdapter::GATE)) continue;
            strata::ds4::Ds4MoeLoraHost& h = lora_host[(size_t) l];
            h.a_g = flat(l, strata::glm::LoraAdapter::GATE, true);
            h.b_g = flat(l, strata::glm::LoraAdapter::GATE, false);
            h.a_u = flat(l, strata::glm::LoraAdapter::UP, true);
            h.b_u = flat(l, strata::glm::LoraAdapter::UP, false);
            h.a_d = flat(l, strata::glm::LoraAdapter::DOWN, true);
            h.b_d = flat(l, strata::glm::LoraAdapter::DOWN, false);
            if (!h.a_g || !h.b_g || !h.a_u || !h.b_u || !h.a_d || !h.b_d) {
                std::fprintf(stderr, "glm_generate: --lora-exps: layer %lld has an incomplete expert delta\n",
                             (long long) l);
                return 2;
            }
            h.n_experts = NE;
            ++nl;
        }
        std::fprintf(stderr, "lora: routed-expert deltas for %d layer(s) (n_experts %lld)\n", nl, (long long) NE);
    }
    double slot_gib_auto = 0, slot_gib_max = 0;
    if (a.slots <= 0 && a.experts != "cpu") {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        // one expert: gate+up+down of the largest layer type (Q4_K ~ 14.2 MB, Q2_K ~ 8.3 MB): size by the tier's
        // blob_bytes after init would be exact; use the Q4_K size here (conservative), the tier caps at what fits
        const double blob = 3.0 * (double) G.n_embd * G.n_ff_exp * 0.5625;
        a.slots = std::max<int64_t>(0, (int64_t) (((double) fr - a.vram_margin_gib * 1073741824.0) / blob));
        // the cache is sized by BYTES (slot_gib, per-layer blob sizes): most GLM layers are Q2_K (~8.3 MB), so a
        // Q4_K-sized count would leave ~40% of the budget empty
        slot_gib_auto = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        std::fprintf(stderr, "slots auto: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f GiB)\n",
                     fr / 1073741824.0, slot_gib_auto, a.vram_margin_gib);
        // --vram-grow KEEP: open the cache as if the margin were KEEP (elastic: mapped up to slot_gib for the
        // prompt, grown after it - see Ds4MoeConfig::slot_gib_max)
        if (a.vram_grow_keep > 0 && a.vram_grow_keep < a.vram_margin_gib)
            slot_gib_max = std::max(0.0, ((double) fr - a.vram_grow_keep * 1073741824.0) / 1073741824.0);
    }

    namespace ds4 = strata::ds4;
    ds4::Ds4MoeTier tier;
    ds4::Ds4MoeConfig mc;
    mc.slots = slot_gib_auto > 0 ? G.n_layer * G.n_expert : a.slots;
    mc.slot_gib = slot_gib_auto;
    mc.slot_gib_max = slot_gib_max;
    mc.pcie_frac = a.pcie;
    mc.pf_b = a.pf_b;
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.cpu_only = a.experts == "cpu";
    mc.vram_lru = a.vram_lru;
    if (slot_gib_max > 0 && mc.vram_lru) {   // the elastic regrow re-admits in the seed's order: slots must not move
        mc.vram_lru = false;
        std::fprintf(stderr, "vram-grow: --vram-lru off (the elastic cache needs the static seed)\n");
    }
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prestage = a.chunk_prestage;
    mc.saliency = !a.saliency.empty();
    mc.skip_file_chunk = a.skip_file_chunk;   // prefill chunks: same rule
    mc.skip_file = a.skip_file;   // decode: same, only for experts the arena does not hold (file tier)
    mc.arena_admit = a.arena_admit;   // arena_adapt gate: heat half-life in tokens (0 = promote on first read)
    if (a.arena_admit > 0.0f) mc.arena_adapt = true;
    mc.skip_miss = a.skip_miss;   // decode: drop a VRAM-miss expert weighing < skip_miss x the token's weight sum
    mc.renorm_skip = a.renorm_skip;   // rescale the sum by the surviving weight fraction (default off)
    mc.arena_lazy = a.arena_lazy;     // fill the arena in the background (serve off the file tier meanwhile)
    mc.vram_pin_frac = a.vram_pin;    // --vram-lru: the census-hot core of the seed is never evicted
    mc.lora = lora_host.empty() ? nullptr : lora_host.data();   // routed-expert deltas (--lora-exps)
    if (a.lora_exps) {
        // The CPU pool applies the down delta too now (ExpertJobMulti::lora_a/b), so --pcie is the user's again;
        // GLM_LORA_PCIE1=1 keeps the old forced 1.0 (every miss on the GPU grouped path) for A/B.
        if (std::getenv("GLM_LORA_PCIE1")) mc.pcie_frac = 1.0;
        // (--chunk-mmq stays: the MMQ chunk path adds the down delta too - native_expert_lora_down after each down
        // product - so a chunk and a decode are ablated the same.  GLM_LORA_MMQ_OFF=1: the old behaviour, for A/B.)
        if (mc.chunk_mmq && std::getenv("GLM_LORA_MMQ_OFF")) {
            mc.chunk_mmq = false;
            std::fprintf(stderr, "lora-exps: --chunk-mmq off (GLM_LORA_MMQ_OFF)\n");
        }
        std::fprintf(stderr, "lora-exps: pcie_frac %.2f (GPU grouped, MMQ chunk and CPU pool paths all apply the "
                             "down delta)\n", mc.pcie_frac);
        // The skip heuristics drop a low-weight expert READ on the miss path - a dropped expert contributes nothing
        // AND loses its delta, so the ablation stops being all-or-nothing (same failure mode as the MMQ chunk above:
        // the model degenerates into repetition loops, e.g. "import LPVOID, LPVOID, ..." and never terminates).
        // Force them off: every routed expert must be ablated or none of it.
        if (!a.renorm_skip && (mc.skip_miss > 0.0f || mc.skip_file > 0.0f || mc.skip_file_chunk > 0.0f)) {
            mc.skip_miss = mc.skip_file = mc.skip_file_chunk = 0.0f;
            std::fprintf(stderr, "lora-exps: skip-miss/skip-file -> 0 (a skipped expert loses its delta; the "
                                 "ablation must be all-or-nothing or the model loops; pass --renorm-skip to test "
                                 "the rescaled sum instead)\n");
        }
    }
    if (mc.saliency && a.prefill_chunk <= 0) { std::fprintf(stderr, "glm_generate: --saliency needs --prefill-chunk\n"); return 2; }
    // --prune FILE: "layer expert" lines; those experts are never routed (selection bias -1e30, REAP-style pruning)
    // and never take arena/VRAM space, so a pruned set that fits RAM+VRAM never touches the file tier.
    // --prune-penalty X: soft pruning - the bias is -X instead, so a pruned expert still wins when the router wants
    // it by more than X (it is then read from the file tier); 0 = the hard mask.
    std::vector<char> pruned((size_t) (G.n_layer * G.n_expert), 0);
    int64_t n_pruned = 0;
    if (!a.prune.empty()) {
        std::FILE * pf = std::fopen(a.prune.c_str(), "r");
        if (!pf) { std::fprintf(stderr, "glm_generate: cannot open %s\n", a.prune.c_str()); return 2; }
        long l, e;
        while (std::fscanf(pf, "%ld %ld", &l, &e) == 2)
            if (l >= 0 && l < G.n_layer && e >= 0 && e < G.n_expert && G.routed(l) && !pruned[(size_t) (l * G.n_expert + e)]) {
                pruned[(size_t) (l * G.n_expert + e)] = 1;
                ++n_pruned;
            }
        std::fclose(pf);
        std::fprintf(stderr, "prune: %lld (layer, expert) pairs masked out of routing (%s)\n", (long long) n_pruned, a.prune.c_str());
    }
    auto is_pruned = [&](int64_t l, int64_t e) { return pruned[(size_t) (l * G.n_expert + e)] != 0; };
    // --census FILE (helios' .census): long-run decode use per (layer, expert), read to rank the VRAM slots and the
    // arena, counted during decode, written back after every request.  The in-process heat (arena_adapt, vram_lru)
    // follows the last few tokens; a prompt floods it.  The census is what the model actually uses, across restarts.
    std::vector<double> census((size_t) (G.n_layer * G.n_expert), 0.0);
    int64_t census_n = 0;
    if (!a.census.empty()) {
        if (std::FILE * cf = std::fopen(a.census.c_str(), "r")) {
            char line[256];
            long l, e;
            double c;
            while (std::fgets(line, sizeof line, cf))
                if (line[0] != '#' && std::sscanf(line, "%ld %ld %lf", &l, &e, &c) == 3 && l >= 0 && l < G.n_layer &&
                    e >= 0 && e < G.n_expert && c > 0) { census[(size_t) (l * G.n_expert + e)] = c; ++census_n; }
            std::fclose(cf);
        }
        std::fprintf(stderr, "census: %lld (layer, expert) counts from %s\n", (long long) census_n, a.census.c_str());
    }
    auto save_census = [&]() {
        if (a.census.empty()) return;
        const std::string tmp = a.census + ".tmp";
        std::FILE * cf = std::fopen(tmp.c_str(), "w");
        if (!cf) return;
        std::fprintf(cf, "# glm_generate census: layer expert decode-uses\n");
        for (int64_t l = 0; l < G.n_layer; ++l)
            for (int64_t e = 0; e < G.n_expert; ++e)
                if (census[(size_t) (l * G.n_expert + e)] > 0)
                    std::fprintf(cf, "%lld %lld %.0f\n", (long long) l, (long long) e, census[(size_t) (l * G.n_expert + e)]);
        if (std::fclose(cf) == 0) std::rename(tmp.c_str(), a.census.c_str());
    };
    if (!tier.init(a.model, mc, err)) { std::fprintf(stderr, "glm_generate: tier init: %s\n", err.c_str()); return 1; }
    const ds4::Ds4MoeGeom & TG = tier.geom();
    if (!mc.cpu_only) {
        bool ok;
        if (!a.profile.empty() && n_pruned == 0) ok = tier.seed_from_routes(a.profile, 512, err);
        else {   // no profile (or a prune mask): index order over the kept experts (GLM routing is near-flat)
            // smallest blobs first: with near-flat routing a VRAM slot's worth is one expert whatever its size, so
            // the Q2_K layers fill the cache (most experts per GiB) and the arena takes the rest
            std::vector<int64_t> lorder;
            for (int64_t l = 0; l < TG.n_layers; ++l) if (TG.routed(l)) lorder.push_back(l);
            std::stable_sort(lorder.begin(), lorder.end(), [&](int64_t x, int64_t y) { return tier.blob_bytes(x) < tier.blob_bytes(y); });
            std::vector<std::pair<int32_t, int32_t>> ranked;
            for (int64_t l : lorder)
                for (int64_t e = 0; e < TG.n_experts; ++e)
                    if (!is_pruned(l, e)) ranked.emplace_back((int32_t) l, (int32_t) e);
            if (census_n > 0)   // the census first: uses per byte (a VRAM slot costs its blob), unseen experts after
                std::stable_sort(ranked.begin(), ranked.end(), [&](const auto & x, const auto & y) {
                    const double cx = census[(size_t) (x.first * G.n_expert + x.second)] / (double) tier.blob_bytes(x.first);
                    const double cy = census[(size_t) (y.first * G.n_expert + y.second)] / (double) tier.blob_bytes(y.first);
                    return cx > cy;
                });
            // soft pruning (s28): a pruned expert the router still picks is read from NVMe (~2 ms, 4 ms/token measured).
            // The arena budget the kept set leaves (72 GiB budget vs ~58 GiB kept) takes the most-used pruned experts,
            // census-ranked, after every kept one - the VRAM seed's head is unchanged.
            if (n_pruned > 0 && a.prune_penalty > 0 && census_n > 0 && !std::getenv("GLM_NO_PRUNED_ARENA")) {
                std::vector<std::pair<int32_t, int32_t>> pr;
                for (int64_t l : lorder)
                    for (int64_t e = 0; e < TG.n_experts; ++e)
                        if (is_pruned(l, e) && census[(size_t) (l * G.n_expert + e)] > 0) pr.emplace_back((int32_t) l, (int32_t) e);
                std::stable_sort(pr.begin(), pr.end(), [&](const auto & x, const auto & y) {
                    return census[(size_t) (x.first * G.n_expert + x.second)] > census[(size_t) (y.first * G.n_expert + y.second)];
                });
                std::fprintf(stderr, "arena: %zu soft-pruned experts with census uses ranked after the kept set\n", pr.size());
                ranked.insert(ranked.end(), pr.begin(), pr.end());
            }
            ok = tier.seed_from_ranked(ranked, err);
        }
        if (!ok) { std::fprintf(stderr, "glm_generate: seed: %s\n", err.c_str()); return 1; }
    }
    std::fprintf(stderr, "tier: %s, %lld resident slots, arena %.1f GiB (%lld experts), file tier %lld; load %.1f s\n",
                 tier.mode(), (long long) tier.resident(), tier.arena_gib(), (long long) tier.arena_experts(),
                 (long long) tier.file_tier(), (now_ms() - t_load0) / 1000.0);

    const int n_layer = (int) G.n_layer;
    const int top_k = (int) TG.top_k;
    const int kPredW = ds4::Ds4MoeConfig::kPredW;
    const int64_t n_embd = G.n_embd;
    std::vector<int> ids_i((size_t) top_k), pred_i((size_t) kPredW);
    std::vector<int32_t> ids32((size_t) top_k), pred32((size_t) kPredW);
    std::vector<float> w((size_t) top_k), routed((size_t) n_embd), rb((size_t) G.n_expert);
    std::mt19937_64 rng(a.seed);

    // --dump-routes: [512-token block][routed layer][token][top_k] u16
    std::vector<int> routed_layers;
    for (int l = 0; l < n_layer; ++l) if (G.routed(l)) routed_layers.push_back(l);
    const int RB = 512;
    std::vector<uint16_t> rblock;
    int rblock_n = 0;
    std::FILE * rfile = a.dump_routes.empty() ? nullptr : std::fopen(a.dump_routes.c_str(), "wb");
    auto flush_routes = [&](bool partial) {
        if (!rfile || rblock_n == 0 || (rblock_n < RB && !partial)) return;
        if (rblock_n < RB) return;   // the ranker reads whole blocks only
        std::fwrite(rblock.data(), 2, rblock.size(), rfile);
        rblock_n = 0;
        std::fill(rblock.begin(), rblock.end(), 0);
    };
    if (rfile) rblock.assign((size_t) RB * routed_layers.size() * top_k, 0);

    // selection-only bias per layer, the ds41 decode recipe: -1e30 / -prune_penalty on pruned experts always; in
    // DECODE only (decode_bias) + route_bias on the experts in the VRAM cache and -route_bias_file on file-tier ones
    // (neither VRAM nor RAM arena).  The prompt routes exactly (prune part only).  Soft: every expert stays reachable
    // and a chosen expert keeps its full LoRA delta - unlike --skip-*, which drop it (the corrupted tool calls, s5).
    bool decode_bias = false;
    const bool dbias = (a.route_bias != 0.0f || a.route_bias_file > 0.0f) && !mc.cpu_only;
    auto set_bias = [&]() {
        const bool on = dbias && decode_bias;
        for (int l = 0; l < n_layer; ++l) {
            if (!G.routed(l)) continue;
            for (int64_t e = 0; e < G.n_expert; ++e) {
                const bool res = on && tier.resident(l, e);
                rb[(size_t) e] = is_pruned(l, e) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (on && a.route_bias != 0.0f && res) ? a.route_bias
                               : (on && a.route_bias_file > 0.0f && !res && !tier.in_arena(l, e)) ? -a.route_bias_file : 0.0f;
            }
            dense.set_route_bias(l, rb.data());
        }
    };
    if (dbias) std::fprintf(stderr, "decode route bias: +%.2f VRAM-resident, -%.2f file-tier (the prompt routes exactly)\n",
                            a.route_bias, a.route_bias_file);
    if (n_pruned > 0) set_bias();
    double t_ph[5] = { 0, 0, 0, 0, 0 };
    bool timing = false;
    auto step = [&](int tid, int pos) -> bool {
        if (dbias && decode_bias) set_bias();
        if (!dense.begin_token(tid)) return false;
        int ri = 0;
        for (int l = 0; l < n_layer; ++l) {
            const bool moe = G.routed(l);
            double t0 = timing ? now_ms() : 0, t1;
            if (moe && !mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {
                int n = 0;
                for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) n++] = pred_i[(size_t) i];
                if (n > 0) tier.prefetch(l, pred32.data(), n);
            }
            if (timing) { t1 = now_ms(); t_ph[0] += t1 - t0; t0 = t1; }
            const float * x = nullptr;
            if (!dense.attn_router_n(l, pos, 1, ids_i.data(), w.data(), &x)) return false;
            if (timing) { t1 = now_ms(); t_ph[1] += t1 - t0; t0 = t1; }
            if (moe) {
                for (int k = 0; k < top_k; ++k) ids32[(size_t) k] = ids_i[(size_t) k];
                if (!a.census.empty())
                    for (int k = 0; k < top_k; ++k)
                        if (ids_i[(size_t) k] >= 0) census[(size_t) (l * G.n_expert + ids_i[(size_t) k])] += 1.0;
                if (rfile) {
                    uint16_t * dst = rblock.data() + ((size_t) ri * RB + rblock_n) * top_k;
                    for (int k = 0; k < top_k; ++k) dst[k] = (uint16_t) ids_i[(size_t) k];
                }
                if (!tier.run(l, ids32.data(), w.data(), x, routed.data())) {
                    std::fprintf(stderr, "glm_generate: tier.run refused at layer %d\n", l);
                    return false;
                }
                ++ri;
            }
            if (timing) { t1 = now_ms(); t_ph[2] += t1 - t0; t0 = t1; }
            if (!dense.finish_layer_n(l, 1, moe ? routed.data() : nullptr)) return false;
            if (timing) { t1 = now_ms(); t_ph[3] += t1 - t0; }
        }
        if (rfile && ++rblock_n == RB) flush_routes(false);
        return true;
    };

    // ---- serve mode: Strata's engine line protocol (~/AI/Strata/serve/server.py StrataEngine).  In: GEN <max_new>
    // [key=value ...] <ids csv> | STOP | QUIT.  Out: READY <ctx> stop, RESUME <reused>, PP <read> <total> <ms> <tok/s>,
    // T <id> per token, DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> 0 0 <reused> <hits> <lookups> 0
    // <file blobs> 0 <prompt read>, ERR <text>.  Conversation reuse: the state is snapshotted at the end of every
    // prompt (GlmDense::snapshot); a request that extends what was fed continues, one that shares the last prompt
    // restores the snapshot, anything else starts over (KDA state cannot rewind further).
    if (a.serve) {
        const int ctxv = (int) dc.ctx;
        std::mutex qm;
        std::condition_variable qc;
        std::deque<std::string> q;
        bool eof = false;
        std::thread([&]() {
            std::string ln;
            while (std::getline(std::cin, ln)) {
                std::lock_guard<std::mutex> lk(qm);
                q.push_back(ln);
                qc.notify_all();
            }
            std::lock_guard<std::mutex> lk(qm);
            eof = true;
            qc.notify_all();
        }).detach();
        auto next_line = [&](std::string & out) -> bool {
            std::unique_lock<std::mutex> lk(qm);
            qc.wait(lk, [&] { return !q.empty() || eof; });
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        auto stop_pending = [&]() -> bool {
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end(); ++it)
                if (*it == "STOP") { q.erase(it); return true; }
            return false;
        };
        const int NP = a.prefill_chunk;
        std::vector<int> cids;
        std::vector<int32_t> cids32;
        std::vector<float> cw, crouted, last_logits((size_t) G.vocab);
        if (NP > 0) {
            cids.resize((size_t) NP * top_k);
            cids32.resize((size_t) NP * top_k);
            cw.resize((size_t) NP * top_k);
            crouted.resize((size_t) NP * n_embd);
        }
        std::string ferr;
        // the prompt ids[0, n) at positions pos0.. -> last_logits = the last one's
        // snap_rel (0 < snap_rel < n): the dense state is snapshotted after the first snap_rel tokens (snap_taken = true)
        bool snap_taken = false;
        auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress, int snap_rel = -1) -> bool {
            snap_taken = false;
            auto snap_here = [&]() {
                if (dense.snapshot()) snap_taken = true;
                else std::fprintf(stderr, "serve: snapshot failed: %s\n", dense.last_error().c_str());
            };
            // xolotl: with the elastic cache grown to its decode size (slot_gib_max > 0), the chunk path's staging
            // (2 x c_stage of >= 1 GiB) only fits after shrink_cache.  The shrink used to run only for prompts of
            // more than 256 tokens, so every shorter prompt - a short chat, a short follow-up turn - died at
            // "ds4_moe: c_stage: out of memory" (exit 134; production too, at 512K).  Now: a few tokens go through
            // the decode loop (no staging), and every chunked prompt shrinks first.
            if (NP <= 0 || (slot_gib_max > 0 && n <= 16)) {
                for (int i = 0; i < n; ++i) {
                    if (!step(ids[i], pos0 + i)) { ferr = dense.last_error(); return false; }
                    if (i + 1 == snap_rel) snap_here();
                    if ((i + 1) % 64 == 0 || i + 1 == n) progress(i + 1);
                }
                const float * lg = nullptr;
                int nv = 0;
                if (!dense.logits_n(1, &lg, &nv)) { ferr = dense.last_error(); return false; }
                std::copy(lg, lg + nv, last_logits.begin());
                return true;
            }
            // the per-request elastic cache: a long prompt borrows the cache's tail for its chunk scratch
            const bool elastic_req = slot_gib_max > 0;   // every chunked prompt (see feed's note above)
            if (elastic_req) {
                std::string serr;
                const int64_t gone = tier.shrink_cache(serr);
                if (gone < 0) { ferr = "shrink_cache: " + serr; return false; }
                if (gone > 0) std::fprintf(stderr, "serve: cache shrunk by %lld slots for a %d-token prompt\n", (long long) gone, n);
            }
            for (int c0 = 0, m = 0; c0 < n; c0 += m) {
                m = dense.safe_chunk(pos0 + c0, std::min(NP, n - c0));
                if (snap_rel > c0 && snap_rel < c0 + m) m = snap_rel - c0;   // a chunk ends where the snapshot goes
                if (!dense.begin_tokens(ids + c0, m)) { ferr = dense.last_error(); return false; }
                for (int l = 0; l < n_layer; ++l) {
                    const bool moe = G.routed(l);
                    const float * x = nullptr;
                    if (!dense.attn_router_n(l, pos0 + c0, m, cids.data(), cw.data(), &x)) { ferr = dense.last_error(); return false; }
                    if (moe) {
                        for (int k = 0; k < m * top_k; ++k) cids32[(size_t) k] = cids[(size_t) k];
                        if (!tier.run_chunk(l, m, cids32.data(), cw.data(), x, crouted.data())) { ferr = "tier.run_chunk refused"; return false; }
                    }
                    if (!dense.finish_layer_n(l, m, moe ? crouted.data() : nullptr)) { ferr = dense.last_error(); return false; }
                }
                if (c0 + m == n && !dense.logits_rows(m - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                if (c0 + m == snap_rel) snap_here();
                progress(c0 + m);
            }
            tier.release_chunk();
            if (elastic_req) dense.release_big();   // the chunk compute buffer too, before the cache grows into it
            if (elastic_req) {
                std::string gerr;
                if (tier.grow_cache(a.vram_grow_keep, gerr) < 0) { ferr = "grow_cache: " + gerr; return false; }
            }
            return true;
        };
        auto is_stop = [&](int t) { return std::find(a.stop.begin(), a.stop.end(), t) != a.stop.end(); };
        std::vector<int> hist;   // the ids the state has seen, in order
        int snap_len = -1;       // hist's length at the snapshot (the end of the last prompt)
        if (slot_gib_max > 0) {   // start at the decode size: a short first request (no shrink) must not decode small
            std::string gerr;
            if (tier.grow_cache(a.vram_grow_keep, gerr) < 0) std::fprintf(stderr, "serve: grow_cache: %s\n", gerr.c_str());
        }
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) tier.resident(),
                    (long long) tier.arena_experts());
        std::printf("READY %d stop\n", ctxv);
        std::fflush(stdout);
        std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d)\n", ctxv, NP);
        std::string line;
        while (next_line(line)) {
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv;
            int max_new = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            int top_k = 0;
            ss >> w >> max_new;
            while (ss >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) { csv = w; continue; }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                if (k == "temperature") temp = (float) std::atof(v.c_str());
                else if (k == "top_p") top_p = (float) std::atof(v.c_str());
                else if (k == "top_k") top_k = std::atoi(v.c_str());
                else if (k == "min_p") min_p = (float) std::atof(v.c_str());
                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
            }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
            if ((int64_t) ids.size() + std::max(0, max_new) > ctxv) {
                std::printf("ERR prompt (%zu) + max tokens (%d) exceed the context (%d)\n", ids.size(), max_new, ctxv);
                std::fflush(stdout);
                continue;
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            int reused = 0;
            if (!hist.empty() && lcp == hist.size() && lcp < ids.size()) reused = (int) lcp;
            else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len < ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else {
                dense.reset();
                hist.clear();
                snap_len = -1;
            }
            // why a follow-up did or did not reuse: lcp = the shared prefix, hist = what the state has seen, snap =
            // where the snapshot sits (the end of the last prompt; -1 = no snapshot).  0 reused on a follow-up whose
            // prefix matches means the snapshot was unavailable (e.g. the device copy failed under VRAM pressure).
            std::fprintf(stderr, "serve: reuse lcp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\n",
                         lcp, hist.size(), snap_len, ids.size(), reused);
            if (reused > 0) { std::printf("RESUME %d\n", reused); std::fflush(stdout); }
            tier.reset_stats();
            const double tp0 = now_ms();
            const int total = (int) ids.size();
            // The snapshot goes before the prompt's last <|assistant|> (the generation prompt), not after it: a client
            // that re-renders the end of the prompt - the turn after a cut-off or cancelled reply, a retry - then
            // resumes there.  It used to sit at the very end, so a retry that differed 2 tokens before the end read
            // a 26,357-token conversation again (52 s, 2026-11-09).  A follow-up reads those few tokens again.
            // STRATA_GLM_SNAP_TOKEN: the tag's id (GLM-5's <|assistant|> = 154828; 0 = the end of the prompt, as before).
            static const int snap_tok = [] { const char * e = std::getenv("STRATA_GLM_SNAP_TOKEN"); return e ? std::atoi(e) : 154828; }();
            int snap_at = total;
            if (snap_tok > 0)
                for (int i = total - 1; i > reused; --i)
                    if (ids[(size_t) i] == snap_tok) { snap_at = i; break; }
            const bool ok = feed(ids.data() + reused, total - reused, reused, [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            }, snap_at < total ? snap_at - reused : -1);
            if (!ok) {
                std::printf("ERR prefill failed: %s\n", ferr.c_str());
                std::fflush(stdout);
                dense.reset();
                hist.clear();
                snap_len = -1;
                continue;
            }
            hist = ids;
            if (snap_at < total) snap_len = snap_taken ? snap_at : -1;
            else if (dense.snapshot()) snap_len = (int) hist.size();
            else std::fprintf(stderr, "serve: snapshot failed: %s\n", dense.last_error().c_str());
            const double prompt_ms = now_ms() - tp0;
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused) in %.2f s = %.1f tok/s\n", total, reused,
                         prompt_ms / 1000.0, 1000.0 * (total - reused) / std::max(prompt_ms, 1e-9));
            tier.reset_stats();
            const double td0 = now_ms();
            int gen = 0, pos = total;
            const char * finish = "length";
            decode_bias = true;
            while (gen < max_new || max_new <= 0) {
                if (stop_pending()) { finish = "cancel"; break; }
                const float * lg = last_logits.data();
                int nv = (int) G.vocab;
                if (gen > 0 && !dense.logits_n(1, &lg, &nv)) { ferr = dense.last_error(); finish = "error"; break; }
                const int tok = sample_p(lg, nv, temp, top_k, top_p, min_p, rng);
                std::printf("T %d\n", tok);
                std::fflush(stdout);
                ++gen;
                if (is_stop(tok)) { finish = "stop"; break; }
                if ((max_new > 0 && gen >= max_new) || pos + 1 >= ctxv) break;
                if (!step(tok, pos)) { ferr = dense.last_error(); finish = "error"; break; }
                hist.push_back(tok);
                ++pos;
            }
            const double dec_ms = now_ms() - td0;
            if (decode_bias && dbias) { decode_bias = false; set_bias(); }   // the next prompt routes exactly
            decode_bias = false;
            const ds4::Ds4MoeStats st = tier.stats();
            if (std::strcmp(finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                dense.reset();
                hist.clear();
                snap_len = -1;
                finish = "length";
            }
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s)\n", gen, dec_ms / 1000.0,
                         1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), finish);
            save_census();
            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish,
                        reused, (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
        }
        tier.close();
        ggml_backend_free(be);
        return 0;
    }

    // ---- prefill (decode loop over the prompt)
    tier.reset_stats();
    const double t_pf0 = now_ms();
    double nll = 0;
    int n_scored = 0;
    std::vector<float> first_logits;   // chunked prefill: the prompt's last row
    double pc_attn = 0, pc_exp = 0, pc_fin = 0, pc_head = 0;
    if (a.prefill_chunk > 0) {
        const int NP = a.prefill_chunk;
        std::vector<int> cids((size_t) NP * top_k);
        std::vector<int32_t> cids32((size_t) NP * top_k);
        std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), lrows;
        const int LR = 16;
        int n = 0;
        for (size_t c0 = 0; c0 < prompt.size(); c0 += (size_t) n) {
            n = dense.safe_chunk((int) c0, (int) std::min<size_t>((size_t) NP, prompt.size() - c0));
            if (!dense.begin_tokens(prompt.data() + c0, n)) { std::fprintf(stderr, "glm_generate: %s\n", dense.last_error().c_str()); return 1; }
            int ri = 0;
            for (int l = 0; l < n_layer; ++l) {
                const bool moe = G.routed(l);
                double t0 = now_ms();
                const float * x = nullptr;
                if (!dense.attn_router_n(l, (int) c0, n, cids.data(), cw.data(), &x)) {
                    std::fprintf(stderr, "glm_generate: chunk at %zu layer %d: %s\n", c0, l, dense.last_error().c_str());
                    return 1;
                }
                double t1 = now_ms(); pc_attn += t1 - t0; t0 = t1;
                if (moe) {
                    for (int k = 0; k < n * top_k; ++k) cids32[(size_t) k] = cids[(size_t) k];
                    if (rfile)
                        for (int t = 0; t < n; ++t) {
                            if (rblock_n + t >= RB) break;
                            uint16_t * dst = rblock.data() + ((size_t) ri * RB + rblock_n + t) * top_k;
                            for (int k = 0; k < top_k; ++k) dst[k] = (uint16_t) cids[(size_t) (t * top_k + k)];
                        }
                    if (!tier.run_chunk(l, n, cids32.data(), cw.data(), x, crouted.data())) {
                        std::fprintf(stderr, "glm_generate: tier.run_chunk refused at layer %d\n", l);
                        return 1;
                    }
                    ++ri;
                }
                t1 = now_ms(); pc_exp += t1 - t0; t0 = t1;
                if (!dense.finish_layer_n(l, n, moe ? crouted.data() : nullptr)) { std::fprintf(stderr, "glm_generate: %s\n", dense.last_error().c_str()); return 1; }
                pc_fin += now_ms() - t0;
            }
            if (std::getenv("GLM_VRAM_TRACE")) {   // per-chunk device headroom (long-prompt scratch growth)
                size_t fr = 0, tot = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
                std::fprintf(stderr, "vram-trace: chunk end pos %zu: %.0f MiB free\n", c0 + (size_t) n, fr / 1048576.0);
            }
            if (rfile) {   // whole 512-token blocks only (a chunk may straddle a block edge: the excess is dropped)
                rblock_n = std::min(RB, rblock_n + n);
                if (rblock_n == RB) flush_routes(false);
            }
            const double th = now_ms();
            if (a.ppl) {   // rows whose next token is in the prompt
                const int last = (int) std::min<size_t>((size_t) n, prompt.size() - 1 - c0);
                for (int r0 = 0; r0 < last; r0 += LR) {
                    const int nr = std::min(LR, last - r0);
                    lrows.resize((size_t) nr * (size_t) G.vocab);
                    if (!dense.logits_rows(r0, nr, lrows.data())) { std::fprintf(stderr, "glm_generate: %s\n", dense.last_error().c_str()); return 1; }
                    for (int r = 0; r < nr; ++r) {
                        const float * l = lrows.data() + (size_t) r * G.vocab;
                        const float mx = *std::max_element(l, l + G.vocab);
                        double z = 0;
                        for (int64_t v = 0; v < G.vocab; ++v) z += std::exp((double) (l[v] - mx));
                        nll += (std::log(z) + mx) - l[prompt[c0 + (size_t) (r0 + r) + 1]];
                        ++n_scored;
                    }
                }
            }
            if (c0 + (size_t) n == prompt.size()) {
                first_logits.resize((size_t) G.vocab);
                if (!dense.logits_rows(n - 1, 1, first_logits.data())) { std::fprintf(stderr, "glm_generate: %s\n", dense.last_error().c_str()); return 1; }
            }
            pc_head += now_ms() - th;
            {   // progress, llama.cpp-style: tokens done, rate so far, ETA
                const double el = (now_ms() - t_pf0) / 1000.0, done = (double) (c0 + (size_t) n);
                const double rate = done / std::max(el, 1e-9);
                std::fprintf(stderr, "\rprefill %zu/%zu tokens | %.1f tok/s | %.0f s elapsed, ETA %.0f s   ",
                             c0 + (size_t) n, prompt.size(), rate, el, ((double) prompt.size() - done) / std::max(rate, 1e-9));
                if (c0 + (size_t) n == prompt.size()) std::fprintf(stderr, "\n");
            }
        }
        if (!a.saliency.empty()) {   // [i32 n_layer][i32 n_expert][f64 sum x L*E][i64 count x L*E]
            const std::vector<double> & ss = tier.saliency_sum();
            const std::vector<int64_t> & sc = tier.saliency_count();
            std::FILE * sf = std::fopen(a.saliency.c_str(), "wb");
            if (!sf || ss.empty()) { std::fprintf(stderr, "glm_generate: cannot write saliency %s\n", a.saliency.c_str()); return 1; }
            const int32_t hdr[2] = { (int32_t) TG.n_layers, (int32_t) TG.n_experts };
            std::fwrite(hdr, 4, 2, sf);
            std::fwrite(ss.data(), 8, ss.size(), sf);
            std::fwrite(sc.data(), 8, sc.size(), sf);
            std::fclose(sf);
            std::fprintf(stderr, "saliency: %s (%zu entries)\n", a.saliency.c_str(), ss.size());
        }
        tier.release_chunk();
    }
    if (slot_gib_max > 0) dense.release_big();   // the chunk compute buffer's VRAM too
    if (slot_gib_max > 0) {   // elastic cache: the prompt's VRAM is back - give it to the expert cache for decode
        std::string gerr;
        if (tier.grow_cache(a.vram_grow_keep, gerr) < 0) { std::fprintf(stderr, "glm_generate: %s\n", gerr.c_str()); return 1; }
    }
    // GLM_BIAS_PROMPT=1 (one-shot, --prefill-chunk 0): the decode biases on the prompt too - a teacher-forced quality
    // proxy for --route-bias-file (with --ppl), since the served prompt never takes them
    if (std::getenv("GLM_BIAS_PROMPT") && a.prefill_chunk <= 0) { decode_bias = true; std::fprintf(stderr, "GLM_BIAS_PROMPT: decode biases on the prompt\n"); }
    for (size_t i = 0; a.prefill_chunk <= 0 && i < prompt.size(); ++i) {
        if (!step(prompt[i], (int) i)) {
            std::fprintf(stderr, "glm_generate: prefill failed at %zu: %s\n", i, dense.last_error().c_str());
            return 1;
        }
        if (a.ppl && i + 1 < prompt.size()) {
            const float * l = nullptr;
            int nv = 0;
            if (!dense.logits_n(1, &l, &nv)) return 1;
            const float mx = *std::max_element(l, l + nv);
            double z = 0;
            for (int v = 0; v < nv; ++v) z += std::exp((double) (l[v] - mx));
            nll += (std::log(z) + mx) - l[prompt[i + 1]];
            ++n_scored;
        }
    }
    if (a.ppl && n_scored)
        std::fprintf(stderr, "ppl: %d positions, mean NLL %.5f, perplexity %.4f\n", n_scored, nll / n_scored, std::exp(nll / n_scored));
    const double pf_ms = now_ms() - t_pf0;

    if (a.skip_file_chunk > 0.0f)
        std::fprintf(stderr, "prefill file-tier entries skipped: %lld\n", (long long) tier.stats().skipped_file);
    // ---- decode
    tier.reset_stats();
    int pos = (int) prompt.size(), n_gen = 0;
    const double t_dec0 = now_ms();
    decode_bias = true;   // --route-bias-file: decode only
    timing = true;
    static const bool prof_dec = std::getenv("GLM_PROFILE_DECODE") != nullptr;   // nsys --capture-range=cudaProfilerApi
    if (prof_dec) ds4::Ds4MoeTier::profiler(true);
    for (; n_gen < a.n_predict; ++n_gen) {
        const double th0 = now_ms();
        const float * lg = nullptr;
        int n_vocab = 0;
        if (n_gen == 0 && !first_logits.empty()) { lg = first_logits.data(); n_vocab = (int) first_logits.size(); }
        else if (!dense.logits_n(1, &lg, &n_vocab)) { std::fprintf(stderr, "glm_generate: logits: %s\n", dense.last_error().c_str()); return 1; }
        t_ph[4] += now_ms() - th0;
        if (n_gen == 0 && !a.dump_logits.empty())
            if (std::FILE * f = std::fopen(a.dump_logits.c_str(), "wb")) { std::fwrite(lg, 4, (size_t) n_vocab, f); std::fclose(f); }
        const int tok = sample(lg, n_vocab, a.temp, rng);
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) { ++n_gen; break; }
        if (n_gen + 1 == a.n_predict) { ++n_gen; break; }
        if (n_gen > 0 && n_gen % 8 == 0)
            std::fprintf(stderr, "\rdecode %d/%d tokens | %.2f tok/s   ", n_gen, a.n_predict,
                         1000.0 * n_gen / std::max(now_ms() - t_dec0, 1e-9));
        if (!step(tok, pos++)) {
            std::fprintf(stderr, "glm_generate: decode failed at pos %d: %s\n", pos - 1, dense.last_error().c_str());
            return 1;
        }
    }
    if (prof_dec) ds4::Ds4MoeTier::profiler(false);
    if (rfile) { flush_routes(true); std::fclose(rfile); }
    const double dec_ms = now_ms() - t_dec0;
    const ds4::Ds4MoeStats st = tier.stats();
    const int dec_steps = std::max(1, n_gen - 1);
    std::fprintf(stderr, "\nprefill: %zu tokens in %.2f s = %.2f tok/s (%s)\n", prompt.size(),
                 pf_ms / 1000.0, 1000.0 * (double) prompt.size() / pf_ms, a.prefill_chunk > 0 ? "chunked" : "decode-loop prefill");
    if (a.prefill_chunk > 0)
        std::fprintf(stderr, "prefill chunks of %d: attention+router %.0f ms, experts %.0f ms, finish %.0f ms, head/ppl %.0f ms\n",
                     a.prefill_chunk, pc_attn, pc_exp, pc_fin, pc_head);
    std::fprintf(stderr, "decode : %d tokens, %d forward passes in %.2f s = %.2f tok/s\n", n_gen, dec_steps,
                 dec_ms / 1000.0, 1000.0 * dec_steps / dec_ms);
    const double look = (double) std::max<int64_t>(1, st.lookups());
    std::fprintf(stderr, "experts/token: hit %.1f%% (prefetched-useful %.1f/token of %.1f issued), cpu %.1f%%, pcie %.1f%%, "
                         "file tier %lld, skipped %.1f/token\n", 100.0 * (double) st.hits / look, (double) st.prefetched_useful / dec_steps,
                 (double) st.prefetch_issued / dec_steps, 100.0 * (double) st.cpu / look, 100.0 * (double) st.pcie / look,
                 (long long) st.file_tier, (double) st.skipped / dec_steps);
    if (a.vram_lru || st.file_tier > 0)
        std::fprintf(stderr, "tier moves: vram_lru swaps %.2f/pass (demoted %lld), arena swaps %lld (admit rejects %lld), "
                             "file reads %.2f ms/pass, file-tier skipped %lld\n",
                     (double) st.vram_swaps / dec_steps, (long long) st.vram_demotes, (long long) st.arena_swaps,
                     (long long) st.admit_rejects, st.file_ms / dec_steps, (long long) st.skipped_file);
    std::fprintf(stderr, "decode ms/token: predict+prefetch %.2f, attention+router %.2f, experts %.2f, finish %.2f, head %.2f\n",
                 t_ph[0] / dec_steps, t_ph[1] / dec_steps, t_ph[2] / dec_steps, t_ph[3] / dec_steps, t_ph[4] / dec_steps);
    std::fprintf(stderr, "tier ms/token: wall %.2f (gpu hits %.2f, pcie %.2f, cpu pool %.2f)\n", st.wall_ms / dec_steps,
                 st.hit_ms / dec_steps, st.pcie_ms / dec_steps, st.cpu_ms / dec_steps);
    {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        std::fprintf(stderr, "VRAM free at the end: %.2f GiB\n", fr / 1073741824.0);
    }
    save_census();
    tier.close();
    ggml_backend_free(be);
    return 0;
}
