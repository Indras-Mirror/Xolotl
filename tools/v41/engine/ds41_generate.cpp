// tools/v41/engine/ds41_generate.cpp - the Strata DeepSeek-V4.1-Flash engine end to end: Ds41Dense (attention,
// Engram, router, shared expert; CUDA or CPU ggml backend) + Ds41MoeTier (routed experts with per-expert formats:
// VRAM cache, pinned RAM arena, file tier, CPU pool, PCIe share) + sampling.  Token ids in, token ids out (text is
// the server's job: tools/v41/serve/serve_ds41.py).
//
//   ds41_generate -m V41-GSQ.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                 [--engram DIR:DIR:DIR] [--engram-hot hot.bin] [--slots auto|N] [--vram-margin GiB] [--arena-gib 60] [--rank r.bin]
//                 [--vram-lru] [--arena-adapt] [--arena-skip-resident] [--remote-device N [--remote-gib G]] [--pcie 0.25] [--tier-batch-pcie 1.0] [--prefill-chunk N] [--prompt-batch B] [--chunk-mmq] [--ctx N]
//                 [--temp 0] [--seed 1] [--stop id,id] [--ppl] [--no-actquant] [--kv-fp4] [--serve] [--quiet]
//                 [--enc-skip|--no-enc-skip] [--enc-replay N] [--dump-last F]
// --enc-skip (layer-major batches only; DEFAULT for one-shot runs, --no-enc-skip = full path): prompt tokens before the last --enc-replay (default 2560 = 128 * 20) stop
// after layer 20's persistent-KV writes; only the last 2560 tokens run layers 21-39.  Same last-token logits / decode.
//
// --serve speaks Strata's engine line protocol (~/AI/Strata/serve/server.py StrataEngine, as glm_generate):
//   in:  GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. seed=..] <ids csv> | STOP | QUIT
//        GENI <max_new> [k=v ..] <image embeddings file> <ids csv>   (GEN with vision; see --image-token)
//   out: READY <ctx> stop, RESUME <reused>, PP <read> <total> <ms> <tok/s>, T <id>, DONE ..., ERR <text>
// --batch-slots N (or Strata's --batch N, which serve/server.py passes for "parallel": N): N conversations decode
// together, one row each per batch window (Strata's slot protocol, serve/server.py generate_batched; INFO batch_slots=N):
//   in:  BGEN <slot> <max_new> [k=v ..] <ids csv> | BGENI <slot> <max_new> [k=v ..] <file> <ids csv> | BSTOP <slot> |
//        BYIELD <slot>
//   out: BGEN reads the prompt into <slot> as a GEN 1 (RESUME / PP / T / DONE on the control lines), then
//        BADM <slot> <1 = it decodes on in the slot | 0 = done>; the slot's tokens are BT <slot> <id> lines and its end
//        BDONE <slot> <produced> <stop|length|cancel|nonfinite> <ms>.  BSTOP ends a slot before its next window;
//        BYIELD <slot> stops a long prompt read at its next layer-major batch: the part read stays in <slot>,
//        YIELDED <slot> <tokens> + DONE .. cancel (+ BADM <slot> 0), and the read goes on from there when it is sent again.
// A window takes every decoding slot of one routing class (the decode biases are global: -exact and -fast requests never
// share a window) plus up to 16 - that rows of a short prompt being read (<= STRATA_BATCH_MIX_MAX new tokens); a longer
// prompt, or one with images, is read alone between windows (layer-major) and the slots wait.  With one slot left a window
// is the old single-sequence step.  GEN (alone: the server sends it when no slot is busy) runs as before, DSpark included,
// in the idle slot that holds the longest start of its prompt.  A slot's conversation stays in it after BDONE (and goes to
// the conversation pool then); the next BGEN / GEN picks the slot (swap_slots) or the pool record that holds the most.
// Conversation reuse: a request that extends what the state has seen continues; one that shares the last prompt
// restores the snapshot taken at its end (Ds41Dense::snapshot: every other cache is position-indexed); the same prompt
// again (regenerate) restores it and samples from the saved last-row logits without a prefill.  VRAM: a multi-token
// prompt runs with the expert cache closed (its buffers need that VRAM) and the decode after it re-seeds the cache from
// the measured free memory (~0.6 s), as a one-shot run; the decode biases (--route-bias / --route-bias-file) apply only
// while decoding.
//
// Load the real model only through tools/ds4/memguard.sh (RAM cap + the shared GPU lock).
#include "ds41_conversation.hpp"
#include "ds41_cuda_policy.hpp"
#include "ds41_dense.hpp"
#include "ds41_exec.hpp"
#include "ds41_moe.hpp"
#include "ds41_conv_reuse.hpp"
#include "ds41_batch_slots.hpp"
#include "ds41_batch.hpp"
#include "ds41_pipeline.hpp"
#if defined(DS41_HAVE_DSPARK)
#include "ds41_dspark.hpp"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, rank, dump_logits, dump_last;
    std::string build_mirror;      // --build-mirror F [--mirror-gib N]: write the blob mirror (coldest experts by --rank) and exit
    double mirror_gib = 85.0;
    std::string prune;            // --prune F: "layer expert" lines (coldest first - v41_prune_list.py), soft or hard pruned
    float prune_penalty = 0.0f;   // --prune-penalty X: selection bias -X on the listed experts (0 = masked out of routing)
    float route_bias_file = 0.6f;    // --route-bias-file X: selection penalty -X on file-tier experts (neither VRAM nor RAM arena);
                                     // DEFAULT 0.6 (Mal: max decode w/o too much loss): 21.1 vs 15.1 tok/s, code ppl 2.67->2.87, agentic 6.88->6.39
    int   bias_prompt = 0;          // --bias-prompt N: quality proxy - apply route bias during the prompt, treating the first N
                                    // ranked experts as VRAM-resident (what the deferred seed will cache)
    float route_bias_remote = 0.0f;  // --route-bias-remote X: selection bonus +X on the experts the remote card (--remote-device)
                                     // holds - computed there beside the main card, off the decode's critical path (0 = as before)
    float route_bias = 1.0f;      // --route-bias X: selection bonus +X on the experts resident in VRAM at the seed (DEFAULT 1.0, Mal 2026-10-11:
                                  // ~4x decode; 4096-tok ppl code 2.38 -> 2.67, agentic 15.1 -> 6.9; --route-bias 0 = exact routing)
    std::vector<std::string> engram;
    std::string engram_hot;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;            // 0 = auto (free VRAM after the dense half, minus --vram-margin)
    // defaults = the measured best decode config on the 4090 (2026-10-11): --pcie 0.75 --pf-b 1.43 --vram-lru
    double vram_margin_gib = 1.0, pcie = 0.75, arena_gib = 60.0, pf_b = 1.43;
    double tier_batch_pcie = -1.0;   // < 0 = STRATA_TIER_BATCH_PCIE / 1.0
    int remote_device = -1;          // --remote-device N: a second card holds the experts ranked after the VRAM head and computes them
    double remote_gib = 0.0, remote_reserve_gib = 1.5;   // --remote-gib G (0 = its free memory less --remote-reserve-gib)
    bool arena_skip_resident = false;   // --arena-skip-resident: the VRAM head stays out of the RAM arena
    int threads = 0, prefill_chunk = 0, prompt_batch = 8192;   // prompt_batch: layer-major batch B (0 = the chunk path)
    int64_t ctx = 0;
    float temp = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool ppl = false, serve = false, quiet = false, vram_lru = true, arena_adapt = false, chunk_mmq = true;
    bool act_quant = true, chunk_prefetch = true, kv_fp4 = false;
    int  kv_type = -1;            // --kv-q8 / --kv-type N: the compressed KV + indexer keys' ggml type (-1 = BF16 with act_quant, else F32)
    int  enc_skip = -1;           // -1 = default (on for one-shot layer-major runs), 1 = --enc-skip, 0 = --no-enc-skip; only run the tokens before the replay window through layers 21-39
    int  enc_replay = 2560;       // --enc-replay N: the replay window (window_size * (n_layer - cand_layer) = 128*20)
    float skip_miss = 0.0f, skip_file = 0.0f;
    double chunk_prefetch_gib = 0.0;
    int image_token = 129264;     // --image-token N: the model's <|image|> id (each image is one run of it in the prompt)
    bool vision = false;
    // --serve's conversation pool: --conversation-cache-mib N (default 8192; 0 = the single park of STRATA_SERVE_PARK)
    // of host memory for conversations saved at their turn boundary, at most --conversation-cache-slots of them
    int conv_mib = 8192, conv_slots = 64;
    // --batch-slots N | auto (--batch N: the server's "parallel": N): conversation states held at once for the batch
    // windows (ds41_batch.hpp; the protocol in the serve loop); 1 = the single-sequence engine as before, 0 = auto
    // (Ds41DenseConfig::n_slots)
    int batch_slots = 1;
    std::string layer_split;
    std::string split_device;
    std::string stage_devices;   ///< alias for --split-device (V100 stage ordinals only)
    std::string head_device;     ///< CUDA ordinal for output head + DSpark (--head-device / STRATA_HEAD_DEVICE)
    std::string stage_pcie;      ///< optional per-stage pcie_frac list (comma-separated)
    bool split_skip_if_fits = false;
    int64_t conversation_cache_min_tokens = 0;
    int64_t conversation_cache_min_free_mib = 2560;
    int prompt_cache = 1;
    int prompt_cache_every = 0;
    std::string prompt_cache_root;
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

// The image-embedding file of a GENI line: one or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then
// n * n_embd float32) in prompt order (serve/server.py combined_embeddings_path; generate.cpp reads the same).
static bool read_sve1(const std::string & path, std::vector<std::vector<float>> & rows, std::vector<int64_t> & counts,
                      int64_t n_embd, int64_t max_rows, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open the embeddings file"; return false; }
    int32_t hdr[5];
    int64_t total = 0;
    for (;;) {
        f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
        if (f.gcount() == 0 && f.eof()) break;
        if (f.gcount() != sizeof(hdr)) { err = "short embeddings header"; return false; }
        if (hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || hdr[4] != (int32_t) n_embd
            || (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[1] > max_rows - total) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            return false;
        }
        const int64_t n = hdr[1];
        total += n;
        std::vector<float> r((size_t) (n * n_embd));
        if (f.read(reinterpret_cast<char *>(r.data()), (std::streamsize) (r.size() * 4)).gcount()
                != (std::streamsize) (r.size() * 4)) { err = "short embeddings file"; return false; }
        rows.push_back(std::move(r));
        counts.push_back(n);
    }
    if (rows.empty()) { err = "the embeddings file has no records"; return false; }
    return true;
}

// Map records onto image-token runs, using the row counts to split adjacent images within the same run.
static bool map_vision(const std::vector<int> & ids, int image_token, const std::vector<std::vector<float>> & rows,
                       const std::vector<int64_t> & counts, std::vector<float> & vis_rows,
                       std::vector<int64_t> & vis_pos, std::string & err) {
    size_t k = 0;
    for (size_t i = 0; i < ids.size(); ) {
        if (ids[i] != image_token) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == image_token) ++j;
        const int64_t run = (int64_t) (j - i);
        if (k >= rows.size()) { err = "the prompt has more images than the embeddings file"; return false; }
        int64_t matched = 0;
        while (matched < run && k < rows.size()) {
            if (counts[k] > run - matched) { err = "an image span does not match its embeddings"; return false; }
            matched += counts[k];
            vis_rows.insert(vis_rows.end(), rows[k].begin(), rows[k].end());
            ++k;
        }
        if (matched != run) { err = "an image span is missing embeddings"; return false; }
        for (size_t t = i; t < j; ++t) vis_pos.push_back((int64_t) t);
        i = j;
    }
    if (k != rows.size()) { err = "the embeddings file has more images than the prompt"; return false; }
    return true;
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
        else if (k == "--engram-hot") a.engram_hot = next();
#ifdef _WIN32
        // the Windows path-list separator: a drive letter's ':' must not split "D:\a;E:\b"
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ';')) a.engram.push_back(d); }
#else
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ':')) a.engram.push_back(d); }
#endif
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--rank") a.rank = next();
        else if (k == "--build-mirror") a.build_mirror = next();
        else if (k == "--check-mirror") { a.build_mirror = next(); a.mirror_gib = -1; }
        else if (k == "--mirror-gib") a.mirror_gib = std::atof(next().c_str());
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--no-vram-lru") a.vram_lru = false;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--arena-skip-resident") a.arena_skip_resident = true;
        else if (k == "--remote-device") a.remote_device = std::atoi(next().c_str());
        else if (k == "--remote-gib") a.remote_gib = std::atof(next().c_str());
        else if (k == "--remote-reserve-gib") a.remote_reserve_gib = std::atof(next().c_str());
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--tier-batch-pcie") a.tier_batch_pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());
        // prompt experts on MMQ (int8 tensor cores; PTQ1_0 through its lossless Q2_0 transcode) - DEFAULT since
        // 2026-10-12: 8K prompt 219-220 -> 244-248 tok/s, 4096 code ppl 2.3799 -> 2.3790, KL 7.6e-3, deterministic
        // (8K --dump-last md5 ee31507a0e88; the MMVQ path's was 4d229102872a).  --no-chunk-mmq = the MMVQ kernels.
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--no-chunk-mmq") a.chunk_mmq = false;
        else if (k == "--no-chunk-prefetch") a.chunk_prefetch = false;
        else if (k == "--chunk-prefetch") a.chunk_prefetch_gib = std::atof(next().c_str());
        else if (k == "--skip-miss") a.skip_miss = (float) std::atof(next().c_str());
        else if (k == "--skip-file") a.skip_file = (float) std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--no-actquant") a.act_quant = false;
        else if (k == "--kv-fp4") a.kv_fp4 = true;
        // 8-bit cache: the same compressed-KV / indexer-key tensors as Q8_0 instead of the default BF16 (act_quant).
        // ggml's flash-attn takes Q8_0 K/V (its kv-type list), set_rows quantizes on the way in and the ring's
        // ggml_cast dequantizes on the way out, so this halves the state's bytes at 8 bits a weight - the 4-bit
        // grids (--kv-fp4) are the aggressive ones and need the packers of tools/v41/ggml/kv-fp4.patch.
        else if (k == "--kv-q8") a.kv_type = (int) GGML_TYPE_Q8_0;
        else if (k == "--kv-type") a.kv_type = std::atoi(next().c_str());
        else if (k == "--serve") a.serve = true;
        else if (k == "--prune") a.prune = next();
        else if (k == "--prune-penalty") a.prune_penalty = (float) std::atof(next().c_str());
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--bias-prompt") a.bias_prompt = std::atoi(next().c_str());
        else if (k == "--route-bias-file") a.route_bias_file = (float) std::atof(next().c_str());
        else if (k == "--route-bias-remote") a.route_bias_remote = (float) std::atof(next().c_str());
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-last") a.dump_last = next();
        else if (k == "--enc-skip") a.enc_skip = 1;
        else if (k == "--no-enc-skip") a.enc_skip = 0;
        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
        else if (k == "--image-token") a.image_token = std::atoi(next().c_str());
        else if (k == "--vision") a.vision = true;
        else if (k == "--conversation-cache-mib") a.conv_mib = std::max(0, std::atoi(next().c_str()));
        else if (k == "--conversation-cache-slots") a.conv_slots = std::max(1, std::atoi(next().c_str()));
        else if (k == "--batch-slots" || k == "--batch") { const std::string v = next(); a.batch_slots = v == "auto" ? 0 : std::max(1, std::atoi(v.c_str())); }
        // Strata serve / router config compatibility (the ones ds41_generate does not implement; the HTTP server and
        // generate.cpp implement them for Qwen).  Accept the flags so the same JSON drops in without edits.
        else if (k == "--conversation-cache-min-tokens") a.conversation_cache_min_tokens = std::atoll(next().c_str());
        else if (k == "--conversation-cache-min-free-mib") a.conversation_cache_min_free_mib = std::atoll(next().c_str());
        else if (k == "--prompt-cache") a.prompt_cache = std::atoi(next().c_str());
        else if (k == "--prompt-cache-every") a.prompt_cache_every = std::atoi(next().c_str());
        else if (k == "--prompt-cache-root") a.prompt_cache_root = next();
        else if (k == "--layer-split") a.layer_split = next();
        else if (k == "--split-device") a.split_device = next();
        else if (k == "--stage-devices") a.stage_devices = next();
        else if (k == "--head-device") a.head_device = next();
        else if (k == "--stage-pcie") a.stage_pcie = next();
        else if (k == "--split-skip-if-fits") a.split_skip_if_fits = true;
        else if (k == "--pool-workers" || k == "--max-context" || k == "--kv-resident" ||
                 k == "--spec" || k == "--spec-min-p" || k == "--batch-mtp" ||
                 k == "--vram-reserve-mib" || k == "--vram-elastic" || k == "--vram-segment-mib" ||
                 k == "--peer-device" || k == "--expert-profile" ||
                 k == "--expert-profile-save" || k == "--expert-profile-save-every" || k == "--calibrate")
            (void) next();
        else if (k == "--prompt-cache-tail" || k == "--no-vision") { /* flag only (--no-vision: vision is off unless --vision) */ }
        else if (k == "--pcie-frac") a.pcie = std::atof(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty() || !a.build_mirror.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best); temp 0 = argmax
int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

// the distribution sample_p draws from, as (token, probability) pairs summing to 1 (temp 0: the argmax alone) - the
// trunk's p for --serve's speculative sampling (same candidates, same weights as sample_p)
void sample_dist(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p,
                 std::vector<std::pair<int, double>> & out) {
    out.clear();
    if (temp <= 0.0f) { out.push_back({ (int) (std::max_element(logits, logits + n_vocab) - logits), 1.0 }); return; }
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    for (int i = 0; i < n; ++i) out.push_back({ idx[(size_t) i], p[(size_t) i] / z });
}

// Speculative sampling with a DETERMINISTIC draft d (DSpark drafts by argmax: q = a point mass on d).  Leviathan /
// Chen: accept with min(1, p(d) / q(d)) = p(d); on rejection sample from norm(max(0, p - q)) = p without d.  The
// emitted token is then distributed exactly as p - the same as plain sampling.  true = d accepted; false = `out` is
// the replacement token.
bool spec_accept(const std::vector<std::pair<int, double>> & dist, int d, std::mt19937_64 & rng, int & out) {
    double pd = 0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (pd > 0 && u(rng) < pd) return true;
    double r = u(rng) * (1.0 - pd);
    int last = -1;
    for (const auto & c : dist) {
        if (c.first == d) continue;
        last = c.first;
        if ((r -= c.second) <= 0) { out = c.first; return false; }
    }
    out = last >= 0 ? last : d;   // rounding: the last candidate (d only if it was the sole candidate - then pd was 1)
    return false;
}

double log_softmax_at(const float * lg, int nv, int tok) {
    const float mx = *std::max_element(lg, lg + nv);
    double z = 0;
    for (int i = 0; i < nv; ++i) z += std::exp((double) lg[i] - mx);
    return (double) lg[tok] - mx - std::log(z);
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("DS41_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: see tools/v41/engine/ds41_generate.cpp\n");
        return 2;
    }
    ds41_apply_mmq_cc_policy(a.backend == "cuda", a.chunk_mmq);
    if (!a.build_mirror.empty()) {   // the second-drive blob mirror (STRATA_BLOB_MIRROR): coldest experts first
        std::string err;
        if (a.mirror_gib < 0) {   // --check-mirror F: open it with every blob compared against the model file
            ds41_sys::setenv("STRATA_BLOB_MIRROR", a.build_mirror.c_str(), 1);
            ds41_sys::setenv("STRATA_BLOB_MIRROR_CHECK", "0", 1);
            Ds41FileBlobs chk(a.model);
            if (!chk.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
            return chk.mirrored() > 0 ? 0 : 1;
        }
        Ds41FileBlobs src(a.model);
        if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
        if (a.rank.empty()) { std::fprintf(stderr, "ds41_generate: --build-mirror needs --rank\n"); return 2; }
        std::vector<std::pair<int32_t, int32_t>> order = ds41_read_ranked(a.rank, err);
        if (order.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        std::reverse(order.begin(), order.end());
        if (!src.build_mirror(a.build_mirror, order, (uint64_t) (a.mirror_gib * 1073741824.0), err)) {
            std::fprintf(stderr, "ds41_generate: build-mirror: %s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        prompt.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) prompt.data(), (std::streamsize) (prompt.size() * 4));
    }
    std::mt19937_64 rng(a.seed);

    // ---- dense half (device 0 in the process's visible GPU set; layer-split stages use ggml_backend_cuda_init per stage)
    ggml_backend_t be = nullptr;
    if (a.backend == "cuda") {
#ifdef STRATA_ENABLE_CUDA
        be = ggml_backend_cuda_init(0);
#else
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
#endif
    } else {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!be) { std::fprintf(stderr, "ds41_generate: no %s backend\n", a.backend.c_str()); return 1; }
    if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, a.threads > 0 ? a.threads : 8);
    int64_t ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 64;
    ctx = std::max<int64_t>(4096, (ctx + 255) / 256 * 256);
    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
    // the layer-major batch B: --prompt-batch, capped to the prompt length for a one-shot run (the state tensors hold
    // B columns, so a short prompt must not pay for the flag's full size; --serve sizes for the largest request)
    const int PB = a.prompt_batch > 0 ? (a.serve ? a.prompt_batch : (int) std::min<int64_t>(a.prompt_batch, (int64_t) prompt.size())) : 0;
    // the dense sub-batch size S: --prefill-chunk, or 512 with a layer-major prompt batch and no chunk size given
    const int S = a.prefill_chunk > 0 ? a.prefill_chunk : (a.prompt_batch > 0 ? 512 : 1);
    dc.max_pass = S;
    dc.prompt_batch = PB;
    // --serve: elastic state columns (STRATA_ELASTIC_COLS=0 = the fixed prompt_batch columns): decode keeps S columns and
    // the ~0.31 MiB a column of a big prompt batch (2.6 GiB at 8192) goes to VRAM expert slots; a layer-major batch
    // grows them (the VRAM cache is released for it anyway) and gives them back.  s11: 2.6 GiB of slots = 10-20% of a
    // verify pass (one-shot, margin 1.0 / 2.3 / 3.6: python 131 / 136 / 145 ms, code 89 / 93 / 107 ms).
    dc.elastic_cols = a.serve && PB > 1 && !(std::getenv("STRATA_ELASTIC_COLS") && std::getenv("STRATA_ELASTIC_COLS")[0] == '0');
    // encoder-only prompt prefill (enc-skip): tokens before the last `enc_replay` need their layers past layer 20 only
    // through the per-layer 128-token window, so for a long prompt they can stop after the KV-writing part of layer
    // 20 (the layer-major batch path only; the chunk path has no per-layer column split, --ppl/--dump-logits need
    // every row's logits, which the skipped rows do not have).
    // default ON (byte-identical last-row logits vs the full path, 8192 tokens, 2026-10-11) for layer-major runs, --serve
    // included (a continuation / snapshot restore only reads per-position caches + the last window of every layer,
    // which the replay window computes in full)
    bool enc_skip = a.enc_skip == 1 || (a.enc_skip < 0 && PB > 1);
    if (enc_skip && a.enc_replay < 1) { std::fprintf(stderr, "ds41_generate: --enc-replay must be >= 1\n"); return 2; }
    if (enc_skip && (a.ppl || !a.dump_logits.empty())) {
        if (a.enc_skip == 1) std::fprintf(stderr, "note: --enc-skip disabled (--ppl/--dump-logits need every row's logits, the old full path)\n");
        enc_skip = false;
    }
    if (enc_skip && PB <= 1) { std::fprintf(stderr, "ds41_generate: --enc-skip needs the layer-major prompt path (--prompt-batch >= 2)\n"); return 2; }
    int64_t ER = enc_skip ? (int64_t) a.enc_replay : 0;
    dc.engram_dirs = a.engram;
    dc.engram_hot = a.engram_hot;
    dc.vision = a.vision;
    dc.engram_hot = a.engram_hot;
    dc.act_quant = a.act_quant;
    // --kv-fp4 / STRATA_KV_FP4=1: the compressed KV + indexer keys packed to their exact FP4 grids (Ds41DenseConfig::kv_fp4)
    const bool kv_fp4_env = std::getenv("STRATA_KV_FP4") && std::getenv("STRATA_KV_FP4")[0] == '1';
    if ((a.kv_fp4 || kv_fp4_env) && a.kv_type >= 0) {
        std::fprintf(stderr, "ds41_generate: --kv-fp4 and --kv-q8/--kv-type are mutually exclusive "
                             "(FP4 packs the caches; --kv-type picks one ggml type for them)\n");
        return 2;
    }
    dc.kv_fp4 = a.kv_fp4 || kv_fp4_env;
    if (dc.kv_fp4) std::fprintf(stderr, "kv: compressed KV NVFP4 + indexer keys MXFP4 (--kv-fp4, exact)\n");
    dc.kv_type = a.kv_type;
    if (a.kv_type >= 0) std::fprintf(stderr, "kv: compressed KV + indexer keys as %s (--kv-q8/--kv-type)\n",
                                     ggml_type_name((ggml_type) a.kv_type));
    // --batch-slots N | auto: decided before the dense half loads (its per-slot states are allocated with it).  Off,
    // with the reason in the log, when they cannot run: the server then says "N asked, the engine runs one at a time".
    int NS = 0;   // the batch slots the serve loop runs (0 = off: one request at a time, as before)
    const bool slots_stub = std::getenv("STRATA_BATCH_STUB") && std::getenv("STRATA_BATCH_STUB")[0] == '1';
    bool slots_on = false;
    if (a.batch_slots != 1) {
        const char * off = nullptr;
        const bool park_env_off = std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0';
        if (!a.serve) off = "--serve only";
        else if (!a.layer_split.empty()) off = "a layer split runs one request at a time (the slots and the pool hold one stage's state)";
        else if (std::getenv("STRATA_BATCH_SLOTS") && std::getenv("STRATA_BATCH_SLOTS")[0] == '0') off = "STRATA_BATCH_SLOTS=0";
        else if (a.conv_mib <= 0 || park_env_off) off = "they need the conversation pool (--conversation-cache-mib > 0)";
        else if (!ds41_core_slots() && !slots_stub)
            off = "this build's dense half has no per-slot states (STRATA_BATCH_STUB=1 runs them one by one, for tests)";
        else if (a.batch_slots == 0 && (slots_stub || !ds41_core_slots())) off = "--batch-slots auto needs the dense half's slots";
        const std::string asked = a.batch_slots == 0 ? std::string("auto") : std::to_string(a.batch_slots);
        if (off) std::fprintf(stderr, "batch slots: --batch-slots %s asked, OFF: %s - one request at a time\n", asked.c_str(), off);
        else {
            slots_on = true;
            NS = a.batch_slots == 0 ? 0 : std::min(a.batch_slots, (int) Ds41MoeTier::kMaxMulti);
            if (a.batch_slots > NS)
                std::fprintf(stderr, "batch slots: %d asked, %d run (a window is at most %d rows, one per slot)\n",
                             a.batch_slots, NS, (int) Ds41MoeTier::kMaxMulti);
        }
    }
    const bool slots_core = slots_on && ds41_core_slots() && !slots_stub;
    ds41_k_config(dc, slots_core ? NS : 1);   // (0 = auto: the core sizes them)
    std::string err;
    const double t_load = now_ms();
    int head_cuda_device = -1;
    if (const char * hd = std::getenv("STRATA_HEAD_DEVICE")) head_cuda_device = std::atoi(hd);
    if (!a.head_device.empty()) head_cuda_device = std::atoi(a.head_device.c_str());
    std::vector<int> split_devs;
    if (!a.layer_split.empty()) {
        auto parse_ints = [](const std::string & str, std::vector<int> & out) -> bool {
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back(std::atoi(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        int n_dev = 1;
#ifdef STRATA_ENABLE_CUDA
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
#endif
        const std::string & stage_list = !a.stage_devices.empty() ? a.stage_devices : a.split_device;
        if (!stage_list.empty()) {
            if (!parse_ints(stage_list, split_devs)) {
                std::fprintf(stderr, "ds41_generate: --stage-devices needs comma-separated GPU indices\n");
                return 2;
            }
        } else if (head_cuda_device >= 0) {
            std::fprintf(stderr, "ds41_generate: --head-device requires --stage-devices (the trunk GPUs, not the head)\n");
            return 2;
        } else {
            // one GPU per stage, in order: "24" = 2 stages, "7,14,20,27,34" = 6 (it took the first 2 GPUs whatever the
            // split, so a 3+ stage split without --stage-devices failed with "need N stage devices", issue #33)
            const int want = a.layer_split == "auto" ? 2 : 2 + (int) std::count(a.layer_split.begin(), a.layer_split.end(), ',');
            for (int d = 0; d < n_dev && (int) split_devs.size() < want; ++d) split_devs.push_back(d);
        }
        if (head_cuda_device >= 0 && split_devs.size() < 2) {
            std::fprintf(stderr, "ds41_generate: --head-device needs a multi-stage layer split\n");
            return 2;
        }
        if (a.layer_split != "auto" && split_devs.size() < 2) {
            for (int d = 1; d < n_dev && (int) split_devs.size() < 2; ++d)
                if (d != head_cuda_device) split_devs.push_back(d);
        }
    }
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    Ds41MoeConfig mc;
    mc.cpu_only = a.experts == "cpu" || (be && ggml_backend_is_cpu(be));
    // slots=0 sizes the VRAM cache from free GiB at seed time (slot_gib); the full expert count made the byte budget huge.
    mc.slots = a.slots > 0 ? a.slots : 0;
    mc.slot_gib = 0.0;
    mc.pcie_frac = a.pcie;
    mc.tier_batch_pcie = a.tier_batch_pcie;
    mc.pf_b = a.pf_b;   // predicted prefetch (Ds41Dense::predict -> tier.prefetch), experts per layer
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip_resident;
    mc.remote_device = a.remote_device;
    mc.remote_gib = a.remote_gib;
    mc.remote_reserve_gib = a.remote_reserve_gib;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prefetch = a.chunk_prefetch;
    mc.chunk_prefetch_gib = a.chunk_prefetch_gib;
    mc.skip_miss = a.skip_miss;
    mc.skip_file = a.skip_file;
    // soft / hard pruning (the GLM recipe, as tools/ds4/ds4_generate): listed experts take no VRAM slot and no arena space;
    // with --prune-penalty X they keep a selection bias -X (still chosen when the router prefers them by more than X, then
    // read from the file tier), without it they are never routed
    std::vector<char> pruned;
    int64_t n_pruned = 0;
    if (!a.prune.empty()) {
        pruned.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        std::FILE * pf = std::fopen(a.prune.c_str(), "r");
        if (!pf) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.prune.c_str()); return 2; }
        long l, e;
        while (std::fscanf(pf, "%ld %ld", &l, &e) == 2)
            if (l >= 0 && l < TG.n_layers && e >= 0 && e < TG.n_experts && !pruned[(size_t) (l * TG.n_experts + e)]) {
                pruned[(size_t) (l * TG.n_experts + e)] = 1;
                ++n_pruned;
            }
        std::fclose(pf);
        std::fprintf(stderr, "prune: %lld (layer, expert) pairs %s (%s)\n", (long long) n_pruned,
                     a.prune_penalty > 0 ? ("penalised -" + std::to_string(a.prune_penalty)).c_str() : "masked out of routing", a.prune.c_str());
        mc.exclude = pruned;
    }
    std::vector<double> stage_pcie_fracs;
    if (!a.stage_pcie.empty()) {
        std::stringstream ss(a.stage_pcie);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            stage_pcie_fracs.push_back(std::atof(tok.c_str()));
        }
    }
    Ds41Pipeline pipe;
    if (!ds41_build_pipeline(a.model, a.layer_split, split_devs, head_cuda_device, a.vram_margin_gib, stage_pcie_fracs, dc, mc, src,
                            pipe, err)) {
        std::fprintf(stderr, "ds41_generate: pipeline: %s\n", err.c_str());
        return 1;
    }
    if (std::getenv("STRATA_SELFTEST_HANDOFF")) {
        if (!ds41_selftest_handoffs(pipe, err)) {
            std::fprintf(stderr, "ds41_generate: handoff self-test failed: %s\n", err.c_str());
            return 1;
        }
    }
    Ds41Dense & dense = ds41_head_dense(pipe);
    if (slots_core) {
        const int k = ds41_k_n(dense);
        if (NS > 0 && k != NS) std::fprintf(stderr, "batch slots: %d asked, the dense half holds %d\n", NS, k);
        NS = k < 2 ? 0 : std::min(k, (int) Ds41MoeTier::kMaxMulti);
        if (NS == 0) std::fprintf(stderr, "batch slots: OFF (the dense half holds %d) - one request at a time\n", k);
    }
    if (NS > 0)
        std::fprintf(stderr, "batch slots: %d (%s)\n", NS, slots_core ? "per-slot states in the dense half"
                                                                       : "STUB: one state, the slots run one by one");
    // gate tap: the single-sequence paths on another batch slot (they must not care which one)
    if (const char * e = std::getenv("DS41_TEST_SOLO_SLOT"); e && !dense.bind_slot(std::atoi(e))) {
        std::fprintf(stderr, "ds41_generate: DS41_TEST_SOLO_SLOT: %s\n", dense.last_error().c_str());
        return 2;
    }
    const Ds41Geometry & G = pipe.dense(0).geom();
    const int n_layer = (int) G.n_layer, n_embd = (int) G.n_embd, top_k = (int) G.n_used, nv = (int) G.vocab;
    int CL = 0;
    for (int64_t s : G.kv_sources) CL = std::max<int>(CL, (int) s);
    const int64_t reach = (int64_t) G.n_swa * (int64_t) (n_layer - CL);
    if (enc_skip && ER < reach) {
        std::fprintf(stderr, "note: --enc-replay %lld < the window reach %lld (swa %lld x %d layers past %d); raised\n",
                     (long long) ER, (long long) reach, (long long) G.n_swa, n_layer - CL, CL);
        ER = reach;
    }
    if (enc_skip) std::fprintf(stderr, "enc-skip: cut at layer %d (last kv source), replay window %lld (reach %lld)\n", CL, (long long) ER, (long long) reach);
    if (a.engram.empty()) std::fprintf(stderr, "WARNING: no --engram dirs: the Engram add is skipped (not the real model)\n");
    if (a.serve && a.conv_mib > 0 && a.prompt_cache == 0)
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     "--prompt-cache 0");
    for (int st = 0; st < pipe.n_stages(); ++st) {
        pipe.dense(st).set_oom_hook([st, &pipe]() { return pipe.tier(st).release_prestage(); });
        std::fprintf(stderr, "ds41: stage %d layers [%lld,%lld) CUDA device %d\n", st, (long long) pipe.layer_lo(st),
                     (long long) pipe.layer_hi(st), pipe.stages[(size_t) st].cuda_device);
    }
    Ds41MoeTier & tier = pipe.tier(0);
    // defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the cache opens into
    // it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots.  --serve does this
    // per request: a multi-token prompt closes the cache (release_cache), the decode after it re-seeds (~0.6 s)
    const bool defer_vram = !mc.cpu_only && (a.prefill_chunk > 1 || a.prompt_batch > 1);
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!mc.cpu_only) {
        if (!a.rank.empty()) {
            ranked = ds41_read_ranked(a.rank, err);
            if (ranked.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        } else {
            for (int64_t l = 0; l < TG.n_layers; ++l)
                for (int64_t e = 0; e < TG.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        }
        const double t0 = now_ms();
        for (int st = 0; st < pipe.n_stages(); ++st) {
            auto sub = filter_ranked_layers(ranked, pipe.layer_lo(st), pipe.layer_hi(st));
            Ds41MoeTier & stier = pipe.tier(st);
            if (defer_vram ? !stier.build_arena_ranked(sub, err) : !stier.seed_from_ranked(sub, err)) {
                std::fprintf(stderr, "ds41_generate: seed stage %d: %s\n", st, err.c_str());
                return 1;
            }
        }
        long long res = 0, arena = 0;
        for (int st = 0; st < pipe.n_stages(); ++st) {
            res += pipe.tier(st).resident();
            arena += pipe.tier(st).arena_experts();
        }
        std::fprintf(stderr, "tier: %lld experts in VRAM, %lld on the remote card (%.1f GiB), %lld in the RAM arena (%.1f GiB%s), "
                     "%lld file tier; seeded in %.1f s\n",
                     res, (long long) tier.remote_experts(), tier.remote_gib(), arena, tier.arena_gib(),
                     pipe.n_stages() > 1 ? " per stage" : "", (long long) tier.file_tier(), (now_ms() - t0) / 1000.0);
    }
    // the selection-only bias per layer: the prune penalty / mask, + route_bias on the experts resident in VRAM now
    std::vector<char> proxy_res;   // --bias-prompt: the would-be VRAM set (first N ranked) until the real seed
    if (a.bias_prompt > 0) {
        proxy_res.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        int got = 0;
        for (const auto & le : ranked) {
            if (got >= a.bias_prompt) break;
            char & r = proxy_res[(size_t) ((int64_t) le.first * TG.n_experts + le.second)];
            if (!r) { r = 1; ++got; }
        }
    }
    // STRATA_DS41_REMOTE_BIAS=1 (with STRATA_DS41_REMOTE): the helper's experts take --route-bias's bonus too, as the
    // ones in this card's VRAM do (they cost a round trip that runs under the layer's own work, not a PCIe copy)
    static const bool remote_bias = std::getenv("STRATA_DS41_REMOTE_BIAS") && std::getenv("STRATA_DS41_REMOTE_BIAS")[0] == '1';
    auto resident_now = [&](int64_t l, int64_t e) {
        if (remote_bias && ds41_tier_for(pipe, (int) l).remote_holds(l, e)) return true;
        if (!proxy_res.empty()) return proxy_res[(size_t) (l * TG.n_experts + e)] != 0;
        return ds41_tier_for(pipe, (int) l).resident(l, e);
    };
    // bias_now: route-bias / file penalty are DECODE levers - before the VRAM seed (the prompt) only the prune part
    // applies, unless --bias-prompt measures them (the prompt's file tier is prefetched; a penalty there buys nothing)
    bool bias_now = a.bias_prompt > 0 || !defer_vram;
    bool bias_set = false;   // a nonzero bias is in the dense half now (--serve turns it off again for the next prompt)
    auto apply_bias = [&]() {
        const bool want = n_pruned > 0 || (bias_now && (a.route_bias != 0.0f || a.route_bias_file != 0.0f || a.route_bias_remote != 0.0f));
        if (!want && !bias_set) return;
        bias_set = want;
        std::vector<float> rb((size_t) TG.n_experts);
        for (int64_t l = 0; l < TG.n_layers; ++l) {
            for (int64_t e = 0; e < TG.n_experts; ++e)
                rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) (l * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (bias_now && a.route_bias != 0.0f && !mc.cpu_only && resident_now(l, e)) ? a.route_bias
                               : (bias_now && a.route_bias_remote != 0.0f && tier.in_remote(l, e)) ? a.route_bias_remote
                               : (bias_now && a.route_bias_file != 0.0f && !resident_now(l, e) &&
                                  !ds41_tier_for(pipe, (int) l).in_arena(l, e)) ? -a.route_bias_file : 0.0f;
            for (int st = 0; st < pipe.n_stages(); ++st) {
                if (l < pipe.layer_lo(st) || l >= pipe.layer_hi(st)) continue;
                pipe.dense(st).set_route_bias((int) l, rb.data());
            }
        }
    };
    apply_bias();   // the prune part now (the prompt routes with it); route_bias again once the VRAM cache is seeded
    std::fprintf(stderr, "loaded in %.1f s: %d layers, ctx %lld, prefill chunk %d, experts %s, %zu classes\n",
                 (now_ms() - t_load) / 1000.0, n_layer, (long long) ctx, a.prefill_chunk, tier.mode(), TG.cls_gu.size());

    // ---- the forward pieces
    std::vector<int> ids1((size_t) top_k);
    std::vector<float> w1((size_t) top_k), routed1((size_t) n_embd), logits1((size_t) nv);
    // --serve's short-turn passes (STRATA_SERVE_PASS_M tokens per dense pass; 1 = single decode steps)
    const int pass_max = std::max(1, std::getenv("STRATA_SERVE_PASS_M") ? std::atoi(std::getenv("STRATA_SERVE_PASS_M")) : 16);
    std::vector<float> crouted_m((size_t) std::max(pass_max, 17) * n_embd);   // 17: pass_v's DSpark seeding / verify passes
    std::string ferr;
    auto head_logits = [&](int row0, int nrows, float * out) -> bool {
        if (!ds41_head_logits(pipe, row0, nrows, out, err)) { ferr = err; return false; }
        return true;
    };
    // STRATA_DEC_PROF=1: per-stage wall time of the decode step (begin / predict+prefetch / attn_router / tier / finish)
    static const bool dprof = std::getenv("STRATA_DEC_PROF") != nullptr;
    // decode: finish(l) and predict(l+1) as one graph (identical results; STRATA_NO_FUSE_PREDICT=1 = separate graphs)
    static const bool fuse_predict = std::getenv("STRATA_NO_FUSE_PREDICT") == nullptr;
    // STRATA_ADVISE_AHEAD=k (0 = off): at layer l, predict layer l+k's routing from the current state and hint its
    // top STRATA_ADVISE_TOP (8) file-tier experts to the kernel (page-cache readahead; the SSD is idle in decode)
    const int advise_ahead = std::getenv("STRATA_ADVISE_AHEAD") ? std::atoi(std::getenv("STRATA_ADVISE_AHEAD")) : 0;
    const int advise_top = std::getenv("STRATA_ADVISE_TOP") ? std::atoi(std::getenv("STRATA_ADVISE_TOP")) : 8;
    // STRATA_FILE_PRED=1 (measurement only, slower): at layer l, predict layers l+1..l+4 from the current state and count
    // how many of the file-tier misses each layer actually routes were in that prediction's top 4 / 8 / 16 - the hit
    // rate a speculative O_DIRECT read k layers ahead would get.  Printed after the decode line.
    // speculative file-tier reads (decode, OPT-IN): finish(l)'s fused prediction graph also predicts layers
    // l+2..l+1+SPEC_AHEAD; the file-tier experts among each prediction's top STRATA_SPEC_TOP are read ahead
    // (tier.spec_read).  Measured 2026-10-12 (p256 x 256, --slots 1500, tokens identical): 61-81% of the file misses
    // come from the stash and the tier gets ~1.8 ms/token faster, but the bigger finish graph (+1.5 ms) and issuing
    // (+1.3 ms) eat it - net 24.97-25.10 vs 25.56 tok/s off.  Default 0 = off; try STRATA_SPEC_TOP=4.
    const int spec_top = std::getenv("STRATA_SPEC_TOP") ? std::atoi(std::getenv("STRATA_SPEC_TOP")) : 0;
    const int spec_ahead = std::getenv("STRATA_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_SPEC_AHEAD")) : 2;
    // STRATA_SPEC_L1=1: also read the next layer's predicted file-tier experts (one layer of lead: mostly late)
    const bool spec_l1 = std::getenv("STRATA_SPEC_L1") && std::getenv("STRATA_SPEC_L1")[0] == '1';
    // STRATA_DUMP_MAIN=F (measurement only, slower): per decode step append {int32 token, int32 pos, 3 x n_embd f32} to F
    // = the step's input token and DeepSeek model.py's DSpark main_hidden: the HC-mean of the state ENTERING target
    // layers 37/38/39 ("the MTP head reads the attention input of its target layers, not their output") = the output
    // of layers 36/37/38 - for the offline DSpark acceptance test (tools/v41/dspark_accept.py).  The one-shot run then
    // feeds the last 128 prompt tokens (DSpark's attention window) as steps too.
    const char * dump_main_path = std::getenv("STRATA_DUMP_MAIN");
    std::FILE * dump_main = dump_main_path ? std::fopen(dump_main_path, "wb") : nullptr;
    const int dm_layers[3] = { 36, 37, 38 };   // outputs of these = inputs of the targets 37 / 38 / 39
    std::vector<float> dm_x, dm_pre, dm_row;
    int dm_tok = 0, dm_pos = 0;
    const bool file_pred = std::getenv("STRATA_FILE_PRED") != nullptr;
    constexpr int kFP = 4;
    std::vector<int> fp_ids(file_pred ? (size_t) kFP * (size_t) n_layer * 16 : 0, -1);
    long long fp_miss[kFP + 1] = {0}, fp_hit[kFP + 1][3] = {{0}}, fp_cand[kFP + 1][3] = {{0}};
    // decode: the tier sums the routed experts on the card straight into the dense half's routed tensor (same bits as
    // the host sum; no parts readback / routed upload per layer).  STRATA_DEC_HOSTSUM=1 = the host round trip
    // the routed tensor's device address: re-read after every dense.set_columns (elastic columns reallocate it)
    float * rdev1 = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? pipe.dense(0).routed_dev() : nullptr;
    double dp[6] = {0, 0, 0, 0, 0, 0};
    long dp_n = 0;
    auto step = [&](int tok, int pos) -> bool {   // one token through every layer; logits via dense.logits
        double ta = dprof ? now_ms() : 0;
        if (dump_main) {
            dm_tok = tok;
            dm_pos = pos;
            dm_row.assign((size_t) 3 * n_embd, 0.0f);
        }
        if (!pipe.dense(0).begin(&tok, 1, pos)) { ferr = pipe.dense(0).last_error(); return false; }
        if (dprof) { const double t = now_ms(); dp[0] += t - ta; ta = t; }
        int32_t top16[16];
        bool have_pred = false;
        auto one_layer = [&](int l) -> bool {
            Ds41Dense & d = ds41_dense_for(pipe, l);
            Ds41MoeTier & t = ds41_tier_for(pipe, l);
            const int stage_end = (int) pipe.layer_hi(ds41_stage_for_layer(pipe, l));
            float * rdev = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? d.routed_dev() : nullptr;
            if (file_pred) {
                for (int k = 1; k <= kFP; ++k) {
                    if (l + k >= stage_end) continue;
                    int * const pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) (l + k)) * 16;
                    if (!d.predict(l + k, pk, 16)) std::fill(pk, pk + 16, -1);
                    for (int i = 0; i < 16; ++i)
                        if (pk[i] >= 0 && !t.resident(l + k, pk[i]) && !t.in_arena(l + k, pk[i]))
                            for (int c = 0; c < 3; ++c) fp_cand[k][c] += i < (4 << c);
                }
            }
            if (advise_ahead > 0 && l + advise_ahead < stage_end) {
                int ta_ids[16];
                if (d.predict(l + advise_ahead, ta_ids, 16)) t.advise_file(l + advise_ahead, ta_ids, advise_top);
            }
            if (mc.pf_b > 0 && !mc.cpu_only) {
                int t16[16];
                if (have_pred) t.prefetch(l, top16, 16);
                else if (d.predict(l, t16, 16)) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    t.prefetch(l, top16, 16);
                }
                have_pred = false;
            }
            if (dprof) { const double t1 = now_ms(); dp[1] += t1 - ta; ta = t1; }
            const float * fn = nullptr;
            if (!d.attn_router(l, ids1.data(), w1.data(), &fn)) { ferr = d.last_error(); return false; }
            if (dprof) { const double t1 = now_ms(); dp[2] += t1 - ta; ta = t1; }
            if (file_pred) {
                for (int j = 0; j < top_k; ++j) {
                    const int e = ids1[(size_t) j];
                    if (e < 0 || t.resident(l, e) || t.in_arena(l, e)) continue;
                    ++fp_miss[0];
                    for (int k = 1; k <= kFP; ++k) {
                        if (l - k < 0) continue;
                        const int * pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) l) * 16;
                        ++fp_miss[k];
                        int at = 16;
                        for (int i = 0; i < 16; ++i) if (pk[i] == e) { at = i; break; }
                        if (at < 4) ++fp_hit[k][0];
                        if (at < 8) ++fp_hit[k][1];
                        if (at < 16) ++fp_hit[k][2];
                    }
                }
            }
            const bool tok_ok = rdev ? t.run_to_dev(l, ids1.data(), w1.data(), fn, rdev)
                                     : t.run(l, ids1.data(), w1.data(), fn, routed1.data());
            if (!tok_ok) { ferr = "tier.run failed"; return false; }
            if (dprof) { const double t1 = now_ms(); dp[3] += t1 - ta; ta = t1; }
            const float * rsum = rdev ? nullptr : routed1.data();
            if (fuse_predict && mc.pf_b > 0 && !mc.cpu_only && l + 1 < stage_end) {
                int t16[16];
                if (spec_top > 0 && spec_ahead > 0) {
                    int ahead[16 * 8];
                    const int na = std::max(0, std::min({spec_ahead, 8, stage_end - l - 2}));
                    if (!d.finish_predict_ahead(l, rsum, l + 1, t16, 16, na, ahead)) { ferr = d.last_error(); return false; }
                    if (dprof) { const double t1 = now_ms(); dp[4] += t1 - ta; ta = t1; }
                    if (spec_l1 && l + 1 < stage_end) t.spec_read(l + 1, t16, std::min(spec_top, 16));
                    for (int ai = 0; ai < na; ++ai)
                        if (l + 2 + ai < stage_end) t.spec_read(l + 2 + ai, ahead + ai * 16, std::min(spec_top, 16));
                    if (dprof) { const double t1 = now_ms(); dp[5] += t1 - ta; ta = t1; }
                } else if (!d.finish_predict(l, rsum, l + 1, t16, 16)) { ferr = d.last_error(); return false; }
                for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                have_pred = true;
            } else if (!d.finish(l, rsum)) { ferr = d.last_error(); return false; }
            if (dprof) { const double t1 = now_ms(); dp[4] += t1 - ta; ta = t1; }
            for (int j = 0; dump_main && j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                d.read_state(dm_x, dm_pre);
                const size_t HC = dm_x.size() / (size_t) n_embd;
                for (size_t h = 0; h < HC; ++h)
                    for (int64_t di = 0; di < n_embd; ++di) dm_row[(size_t) j * n_embd + di] += dm_x[h * n_embd + di] / (float) HC;
                if (j == 2) {
                    std::fwrite(&dm_tok, 4, 1, dump_main);
                    std::fwrite(&dm_pos, 4, 1, dump_main);
                    std::fwrite(dm_row.data(), 4, dm_row.size(), dump_main);
                }
            }
            return true;
        };
        int cur_st = 0;
        for (int l = 0; l < n_layer; ++l) {
            const int st = ds41_stage_for_layer(pipe, l);
            if (pipe.enabled() && st != cur_st) {
                if (!ds41_sync_stages(pipe, cur_st, st, 0, 1, ferr)) return false;
                if (!ds41_begin_stage_continued(pipe, st, &tok, 1, pos, false, ferr)) return false;
                cur_st = st;
            }
            if (!one_layer(l)) return false;
        }
        if (dprof && ++dp_n % 64 == 0)
            std::fprintf(stderr, "decode step ms/token: begin %.2f predict %.2f attn_router %.2f tier %.2f finish %.2f spec %.2f (%ld steps)\n",
                         dp[0] / dp_n, dp[1] / dp_n, dp[2] / dp_n, dp[3] / dp_n, dp[4] / dp_n, dp[5] / dp_n, dp_n);
        return true;
    };
    // a few tokens in ONE dense pass (the dense weights - ~8 GB - read once for m tokens, not per token), the routed
    // experts token by token from the VRAM cache (run_to_dev into the pass's routed rows): --serve's short turns.
    // Row-batched matmuls: the same model, not the same bits as m single steps.
    double fd_dense = 0, fd_tier = 0, fd_logits = 0;   // prompt wall split (printed with the prompt line)
    std::vector<int> mids;
    std::vector<float> mw;
    auto pass_m = [&](const int * toks, int m, int pos) -> bool {
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        double t0 = now_ms();
        if (!pipe.dense(0).begin(toks, m, pos)) { ferr = pipe.dense(0).last_error(); return false; }
        auto run_pm = [&](int l) -> bool {
            Ds41Dense & d = ds41_dense_for(pipe, l);
            Ds41MoeTier & tr = ds41_tier_for(pipe, l);
            float * rdev = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? d.routed_dev() : nullptr;
            const float * fn = nullptr;
            if (!d.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = d.last_error(); return false; }
            const double t1 = now_ms();
            fd_dense += t1 - t0;
            t0 = t1;
            for (int ti = 0; ti < m; ++ti) {
                const bool ok = rdev ? tr.run_to_dev(l, mids.data() + (size_t) ti * top_k, mw.data() + (size_t) ti * top_k,
                                                     fn + (size_t) ti * n_embd, rdev + (size_t) ti * n_embd)
                                     : tr.run(l, mids.data() + (size_t) ti * top_k, mw.data() + (size_t) ti * top_k,
                                              fn + (size_t) ti * n_embd, routed1.data());
                if (!ok) { ferr = "tier.run failed"; return false; }
                if (!rdev) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) ti * n_embd);
            }
            const double t2 = now_ms();
            fd_tier += t2 - t1;
            t0 = t2;
            if (!d.finish(l, rdev ? nullptr : crouted_m.data())) { ferr = d.last_error(); return false; }
            t0 = now_ms();
            return true;
        };
        int cur_st = 0;
        for (int l = 0; l < n_layer; ++l) {
            const int st = ds41_stage_for_layer(pipe, l);
            if (pipe.enabled() && st != cur_st) {
                if (!ds41_sync_stages(pipe, cur_st, st, 0, m, ferr)) return false;
                if (!ds41_begin_stage_continued(pipe, st, toks, m, pos, false, ferr)) return false;
                cur_st = st;
            }
            if (!run_pm(l)) return false;
        }
        return true;
    };
    // pass_m + DSpark's main_hidden: vh[t * 3D + j * D ..] = the HC-mean of the state leaving layer dm_layers[j] (=
    // entering the DSpark target layer 37 + j) for pass token t (the STRATA_DUMP_MAIN math, token-major read_state);
    // vlog = the pass's m rows of logits.
    std::vector<float> vh, vlog, vx, vpre;
    std::vector<int> pv_pred, pv_rank;                 // D3 prefetch: per-token predictions, the ranked union
    std::vector<std::pair<int, int>> pv_score;
    double pv_ms[4] = {0, 0, 0, 0};   // pass_v split: dense (begin/attn_router/finish), tier, taps (read_state), logits
    long pv_n = 0, pv_tok = 0;
    auto pass_v = [&](const int * toks, int m, int pos) -> bool {
        // D3 (the batched per-class verify tier) is the default: correct (bias-off verify-vs-prefill gate, near-ties
        // only) and faster (verify 167.6 -> ~100 ms/pass).  STRATA_TIER_BATCH=0 = the per-token run_to_dev path.
        static const bool no_tier_batch = [] { const char* v = std::getenv("STRATA_TIER_BATCH"); return v && v[0] == '0'; }();
        double tq = now_ms();
        auto lap = [&](int i) { const double t = now_ms(); pv_ms[i] += t - tq; tq = t; };
        ++pv_n; pv_tok += m;
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        vh.assign((size_t) m * 3 * n_embd, 0.0f);
        if (!pipe.dense(0).begin(toks, m, pos)) { ferr = pipe.dense(0).last_error(); return false; }
        // D3 prefetch (STRATA_D3_PF = slots per layer, default 1, 0 = off): before each layer's attention, predict
        // every pass token's routing (predict_multi), rank the union (score 16 - rank, summed over tokens) and DMA the
        // likely misses into the prefetch slots on the prefetch stream - they land under the attention.  PCIe is the
        // wall (nsys: ~100 expert DMAs per verify pass, 1.4 s of a 4.2 s decode), so a wrong guess costs a full blob:
        // docs prompt, same text: slots 0/1/2/3/6 -> verify 104.4/101.7/104.5/111.1/124.1 ms (precision 40% at 1).
        // Its row cap stays 8 (the old kMaxMulti, i.e. the passes it was measured on): since kMaxMulti 16 a 9-16-row
        // pass is ONE tier call too, but it ran without the prefetch before and still does (a separate experiment).
        static const int d3_pf = std::getenv("STRATA_D3_PF") ? std::atoi(std::getenv("STRATA_D3_PF")) : 1;
        constexpr int kD3PfRows = 8;
        static_assert(kD3PfRows <= Ds41MoeTier::kMaxMulti, "the prefetch is consumed by ONE tier call per layer");
        auto pv_layer = [&](int l) -> bool {
            Ds41Dense & d = ds41_dense_for(pipe, l);
            Ds41MoeTier & tr = ds41_tier_for(pipe, l);
            const int stage_end = (int) pipe.layer_hi(ds41_stage_for_layer(pipe, l));
            float * rdev = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? d.routed_dev() : nullptr;
            const bool pf_now = rdev && !no_tier_batch && d3_pf > 0 && m <= kD3PfRows;
            if (pf_now) {
                pv_pred.resize((size_t) m * 16);
                if (d.predict_multi(l, pv_pred.data())) {
                    pv_score.clear();
                    for (int t = 0; t < m; ++t)
                        for (int i = 0; i < 16; ++i) {
                            const int e = pv_pred[(size_t) t * 16 + i];
                            if (e < 0) continue;
                            auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & p) { return p.first == e; });
                            if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                        }
                    std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
                    pv_rank.clear();
                    for (const auto & p : pv_score) pv_rank.push_back(p.first);
                    tr.prefetch_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_pf);
                }
            }
            // D3 speculative NVMe reads (STRATA_D3_SPEC=n, default 0 = off; STRATA_D3_SPEC_AHEAD=k, default 2): the
            // pass's routing for layer l+k predicted from the state entering l (predict_multi, bias included), its
            // union ranked as above, and the NVMe-tier experts among the top n read into the stash by background
            // readers while layers l..l+k-1 run - the verify pass otherwise waits on them (nsys x0.5: pthread_join
            // 7.9 ms/token on the decode thread).  A wrong guess costs SSD bandwidth only.
            static const int d3_spec = std::getenv("STRATA_D3_SPEC") ? std::atoi(std::getenv("STRATA_D3_SPEC")) : 0;
            static const int d3_spec_ahead = std::getenv("STRATA_D3_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_D3_SPEC_AHEAD")) : 2;
            if (rdev && !no_tier_batch && d3_spec > 0 && m <= Ds41MoeTier::kMaxMulti && l + d3_spec_ahead < stage_end) {
                pv_pred.resize((size_t) m * 16);
                if (d.predict_multi(l + d3_spec_ahead, pv_pred.data())) {
                    pv_score.clear();
                    for (int t = 0; t < m; ++t)
                        for (int i = 0; i < 16; ++i) {
                            const int e = pv_pred[(size_t) t * 16 + i];
                            if (e < 0) continue;
                            auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & q) { return q.first == e; });
                            if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                        }
                    std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
                    pv_rank.clear();
                    for (const auto & q : pv_score) pv_rank.push_back(q.first);
                    tr.spec_read(l + d3_spec_ahead, pv_rank.data(), std::min(d3_spec, (int) pv_rank.size()), true);
                }
            }
            const float * fn = nullptr;
            if (!d.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = d.last_error(); return false; }
            lap(0);
            if (rdev && !no_tier_batch) {
                // D3: ONE batched per-class tier call for the pass's m tokens (the union of their experts is served
                // once), not m run_to_dev calls.  Split at Ds41MoeTier::kMaxMulti (16: a depth-4 pass and a 16-row
                // lookup verify / prompt pass are one call each).
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int chunk = std::min(kMM, m - off);
                    if (!tr.run_multi_to_dev(l, chunk, mids.data() + (size_t) off * top_k, mw.data() + (size_t) off * top_k,
                                               fn + (size_t) off * n_embd, rdev + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi_to_dev failed"; return false;
                    }
                }
            } else {
                for (int t = 0; t < m; ++t) {
                    const bool ok = rdev ? tr.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                            fn + (size_t) t * n_embd, rdev + (size_t) t * n_embd)
                                          : tr.run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                     fn + (size_t) t * n_embd, routed1.data());
                    if (!ok) { ferr = "tier.run failed"; return false; }
                    if (!rdev) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
                }
            }
            lap(1);
            if (!d.finish(l, rdev ? nullptr : crouted_m.data())) { ferr = d.last_error(); return false; }
            lap(0);
            for (int j = 0; j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                if (!d.read_state(vx, vpre)) { ferr = d.last_error(); return false; }
                const size_t HC = vx.size() / ((size_t) m * n_embd);
                for (int t = 0; t < m; ++t)
                    for (size_t h = 0; h < HC; ++h) {
                        const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                        float * dst = vh.data() + (size_t) t * 3 * n_embd + (size_t) j * n_embd;
                        for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                    }
                lap(2);
            }
            return true;
        };
        int cur_st = 0;
        for (int l = 0; l < n_layer; ++l) {
            const int st = ds41_stage_for_layer(pipe, l);
            if (pipe.enabled() && st != cur_st) {
                if (!ds41_sync_stages(pipe, cur_st, st, 0, m, ferr)) return false;
                if (!ds41_begin_stage_continued(pipe, st, toks, m, pos, false, ferr)) return false;
                cur_st = st;
            }
            if (!pv_layer(l)) return false;
        }
        vlog.resize((size_t) m * nv);
        if (!head_logits(0, m, vlog.data())) return false;
        lap(3);
        if (dprof && pv_n % 16 == 0)
            std::fprintf(stderr, "verify pass ms/token: dense %.2f tier %.2f taps %.2f logits %.2f (%ld passes, %ld tokens)\n",
                         pv_ms[0] / pv_tok, pv_ms[1] / pv_tok, pv_ms[2] / pv_tok, pv_ms[3] / pv_tok, pv_n, pv_tok);
        return true;
    };
    const int NP = std::max(1, a.prefill_chunk);
    const int SB = std::max(1, S);                 // the dense sub-batch size
    std::vector<int> cids((size_t) NP * top_k);
    std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), clog;
    const bool routed_host = std::getenv("STRATA_ROUTED_HOST") != nullptr;
    std::vector<float> brouted;                    // the layer-major path's routed sums (n_embd * B)
    if (PB > 0) brouted.resize((size_t) n_embd * (size_t) PB);
    // a single run_chunk call streams layer l's experts once for its n tokens; the card's chunk buffers (dominated by
    // c_parts = n * top_k * n_embd * 4) must fit ~1 GiB, so a larger batch is run in pieces (each streams once again)
    // budget 3 GB by default (STRATA_CHUNK_BUDGET_GB): at B=8192 the old 1 GB cap split every layer into 2 pieces and each
    // piece re-uploaded the whole layer over PCIe (1.97M vs 0.98M expert copies); the prompt has ~12 GiB of VRAM free
    // (the VRAM expert cache is only seeded after the prompt, from the memory measured then)
    const double chunk_budget = std::getenv("STRATA_CHUNK_BUDGET_GB") ? std::atof(std::getenv("STRATA_CHUNK_BUDGET_GB")) * 1e9 : 3.0e9;
    const int piece = (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) n_embd * 4 * (top_k + 3)));
    // the prompt ids[0, n) at positions pos0..: last_logits = the last row; all_logits (if given) every row
    std::vector<float> last_logits((size_t) nv);
    // the deferred VRAM seed after a prompt: the prompt's buffers go back, the cache opens into the free space, and the
    // decode biases (route bias + file penalty) switch on.  Returns the free GiB measured (for the log).
    double seed_free = 0, seed_gib = 0, seed_s = 0;
    // STRATA_DSPARK=<sidecar> (one-shot, greedy): DSpark speculative decoding - the checkpoint's mtp.0-2 draft stack
    // (ds41_dspark) drafts STRATA_DSPARK_DEPTH (default 4, <= 5) tokens, one dense pass verifies them (pass_v).  The
    // drafter loads here, before the VRAM expert cache measures the free memory, so the cache sizes around it.
    // --serve: STRATA_SERVE_DSPARK=<sidecar> (opt-in) - speculative SAMPLING (spec_accept), same output distribution
    const char * dsp_path = a.serve ? std::getenv("STRATA_SERVE_DSPARK") : std::getenv("STRATA_DSPARK");
    if (a.serve && dsp_path && (!defer_vram || S < 17)) {
        std::fprintf(stderr, "serve: STRATA_SERVE_DSPARK needs the GPU layer-major prompt path - DSpark off\n");
        dsp_path = nullptr;
    }
#if defined(DS41_HAVE_DSPARK)
    std::unique_ptr<Ds41DSpark> dsp;
#endif
    auto open_cache = [&]() -> bool {
        for (int st = 0; st < pipe.n_stages(); ++st) pipe.dense(st).release_pass();
#if defined(DS41_HAVE_DSPARK)
        if (dsp_path && !dsp) {
            Ds41DSparkConfig sc;
            // Layer split: the trunk head (shared output.weight) lives on the last stage; DSpark must run there too.
            sc.backend = dense.backend();
            if (pipe.head_offload()) sc.cuda_device = pipe.head->cuda_device;
            sc.sidecar = dsp_path;
            sc.main_gguf = a.model;
            sc.max_append = 16;
            if (!std::getenv("STRATA_DSPARK_OWN_HEAD")) sc.shared_head = dense.weight("output.weight");   // -0.7 GB
#ifdef STRATA_ENABLE_CUDA
            if (pipe.head_offload()) cudaSetDevice(pipe.head->cuda_device);
            else if (pipe.enabled()) cudaSetDevice(pipe.stages[(size_t) (pipe.n_stages() - 1)].cuda_device);
#endif
            dsp = std::make_unique<Ds41DSpark>();
            const double ti = now_ms();
            if (!dsp->init(sc, err)) {
                if (a.serve) {
                    std::fprintf(stderr, "serve: DSpark disabled (%s) — decoding without drafts\n", err.c_str());
                    dsp.reset();
                } else {
                    err = "DSpark init: " + err;
                    return false;
                }
            } else {
                std::fprintf(stderr, "\nDSpark: %s, %.2f GB of weights on the card (%.1f s)", dsp_path, dsp->weight_bytes() / 1e9,
                             (now_ms() - ti) / 1000.0);
            }
        }
#endif
        const double min_gib = 0.25;
        const bool explicit_slots = a.slots > 0;
        const double ts = now_ms();
        for (int st = 0; st < pipe.n_stages(); ++st) {
            ggml_backend_t st_be = pipe.stages[(size_t) st].backend;
#ifdef STRATA_ENABLE_CUDA
            cudaSetDevice(pipe.stages[(size_t) st].cuda_device);
#endif
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(st_be), &fr, &tot);
            const double free_gib = fr / 1073741824.0;
            seed_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
            if (st == 0) seed_free = free_gib;
            const bool cache_room = explicit_slots || seed_gib >= min_gib;
            Ds41MoeTier & stier = pipe.tier(st);
            stier.set_cache_enabled(cache_room);
            stier.set_slot_gib(explicit_slots ? 0.0 : seed_gib);
            auto sub = filter_ranked_layers(ranked, pipe.layer_lo(st), pipe.layer_hi(st));
            if (!cache_room)
                std::fprintf(stderr, "serve: stage %d VRAM %.2f GiB free - no room for the expert cache\n", st, free_gib);
            if (!stier.seed_from_ranked(sub, err)) {
                if (cache_room) {
                    std::fprintf(stderr, "serve: stage %d cache seed failed (%s) - seeding with VRAM cache off\n", st,
                                 err.c_str());
                    stier.set_cache_enabled(false);
                    if (!stier.seed_from_ranked(sub, err)) return false;
                } else return false;
            }
        }
        seed_s = (now_ms() - ts) / 1000.0;
        // every seed (the DSpark tail opens the cache silently otherwise): what another process on the card - the
        // image encoder - costs the decode is the slot count here
        if (std::getenv("STRATA_SEED_LOG"))
            std::fprintf(stderr, "seed: %.2f GiB free (margin %.2f) -> %.2f GiB of slots: %lld experts in VRAM (%.2f s)\n",
                         seed_free, a.vram_margin_gib, seed_gib, (long long) tier.resident(), seed_s);
        proxy_res.clear();   // the real VRAM set from here on
        bias_now = true;     // decode: route bias + file penalty on
        apply_bias();
        return true;
    };
    bool enc_reported = false;
    // ---- env-gated per-layer dumps (determinism bisect): STRATA_LAYER_DUMP=<dir> writes, for every layer of the
    // sequential chunk path, the routed experts' input (fn, the dense half's output) and their weighted sum, plus an
    // index source layer's top-k ids.  Two runs' files diff at the first layer where a run-to-run difference appears,
    // and fn-vs-routed tells whether it is the MoE tier or the attention / indexer half.
    const char * dump_dir = std::getenv("STRATA_LAYER_DUMP");
    auto dumpf = [&](const char * name, const void * p, size_t bytes) {
        std::ofstream f(std::string(dump_dir) + "/" + name, std::ios::binary);
        f.write((const char *) p, (std::streamsize) bytes);
    };
    // tail capture (STRATA_SERVE_DUMP_TAIL, the quality scorer): the last `tail_keep` rows of a feed's logits into
    // tail_logits (row n - tail_keep first) - a teacher-forced continuation's rows without the whole prompt's (a 23K-token
    // prompt's full logits would be 12 GB).  0 = off.
    int tail_keep = 0;
    std::vector<float> tail_logits;
    auto capture_tail = [&](int n, int row0, int nrows, int dense_row0) -> bool {   // rows [row0, row0+nrows) of this feed
        if (tail_keep <= 0) return true;
        const int t0 = n - tail_keep, lo = std::max(row0, t0), hi = row0 + nrows;
        if (lo >= hi) return true;
        if (tail_logits.size() != (size_t) tail_keep * (size_t) nv) tail_logits.assign((size_t) tail_keep * (size_t) nv, 0.0f);
        if (!head_logits(dense_row0 + (lo - row0), hi - lo, tail_logits.data() + (size_t) (lo - t0) * nv)) return false;
        return true;
    };
    int bcap_n = 0, bcap_got = 0;  // serve DSpark seed captured from a layer-major batch (see the layer loop below)
    std::vector<float> bcap_vh;    // [bcap_n][3 * n_embd], as pass_v's vh
    // the layer-major batch width for THIS turn: PB, clamped per request by the VRAM actually free (the --serve loop
    // sets it; a one-shot run keeps PB)
    int pb_cap = PB;
    // STRATA_VRAM_TRACE=1: every allocation-relevant step prints the free VRAM (debugging a shared card)
    const bool vram_trace = std::getenv("STRATA_VRAM_TRACE") != nullptr;
    auto vram_log = [&](const char * where, int a) {
        if (!vram_trace) return;
        size_t tf = 0, tt = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
        std::fprintf(stderr, "vram trace: %s %d free %.2f GiB\n", where, a, tf / 1073741824.0);
    };
    // --batch-slots' BYIELD: asked before each layer-major batch of a read (c0, its width B) whether that batch is the
    // read's last one; then its enc-skip cut ends at the batch's own end (the replay window before it is computed in
    // full, so the read can go on from there) and feed() stops after it with feed_stopped = the tokens fed.
    std::function<bool(int, int)> feed_yield;
    int feed_stopped = -1;
    // The serve stdin reader can receive STOP while a long layer-major prompt is
    // running. Poll only between completed graph/tier operations; one-shot has no
    // callback and keeps the existing path.
    std::function<bool()> prefill_stop;
    bool prefill_cancelled = false;
    auto stop_prefill = [&]() -> bool {
        if (prefill_cancelled || (prefill_stop && prefill_stop())) {
            prefill_cancelled = true;
            ferr = "cancelled";
            return true;
        }
        return false;
    };
    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,
                    std::vector<float> * all_logits) -> bool {
        feed_stopped = -1;
        for (int c0 = 0; c0 < n; ) {
            if (stop_prefill()) return false;
            bool yield_here = false;
            if (PB > 1 && n - c0 >= 2) {
                // ---- layer-major: up to B tokens of every layer, the experts run once per layer over the whole batch
                int B = std::min(pb_cap, n - c0);
                // The batch must FIT: its state columns (~0.31 MiB a column) and its graph (~0.17 MiB a column,
                // 8192 -> 1424 MiB) come out of whatever is left after everything else sharing the card (a
                // desktop's STT server, DSpark's 2.9 GB, the dense half).  A batch that does not fit is retried at
                // HALF the width (twice the sweeps over the file tier, the same answer) down to SB, so a tight card
                // makes a prompt slower instead of failing the request (s13 run8: "prefill failed: graph allocation
                // failed" on a 21K prompt with Whisper's 2.5 GB resident).
                Ds41Dense & batch_dense = pipe.enabled() ? pipe.dense(0) : dense;
                for (;;) {
                    const auto set_cols = [&](int cols) -> bool {
                        if (!pipe.enabled()) {
                            if (cols <= dense.columns()) return true;
                            if (!dense.set_columns(cols)) { ferr = dense.last_error(); return false; }
                            if (rdev1) rdev1 = dense.routed_dev();
                            return true;
                        }
                        for (int st = 0; st < pipe.n_stages(); ++st) {
                            if (cols <= pipe.dense(st).columns()) continue;
                            if (!pipe.dense(st).set_columns(cols)) {
                                ferr = pipe.dense(st).last_error();
                                return false;
                            }
                        }
                        if (rdev1) rdev1 = pipe.dense(0).routed_dev();
                        return true;
                    };
                    if (!set_cols(B)) {
                        if (B > SB) {
                            std::fprintf(stderr, "ds41_generate: %d state columns did not fit (%s) - retrying at %d\n",
                                         B, ferr.c_str(), std::max(SB, B / 2));
                            B = std::max(SB, B / 2);
                            continue;
                        }
                        return false;
                    }
                    if (batch_dense.begin_batch(ids + c0, B, pos0 + c0)) break;
                    if (B > SB) {
                        std::fprintf(stderr, "ds41_generate: batch of %d did not fit (%s) - retrying at %d\n",
                                     B, dense.last_error().c_str(), std::max(SB, B / 2));
                        B = std::max(SB, B / 2);
                        continue;
                    }
                    ferr = dense.last_error();
                    return false;
                }
                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20
                // are never read for them), rounded DOWN to a sub-batch boundary so a straddling sub-batch still runs
                // the full path (more work, never wrong); the last sub-batches holding the final token are always full.
                if (feed_yield && c0 + B < n) yield_here = feed_yield(c0, B);
                const int n_cut = yield_here ? c0 + B : n;   // (the read's end, as far as enc-skip is concerned)
                int cut = 0;
                if (enc_skip) {
                    const int64_t gpos = (int64_t) pos0 + c0;
                    const int64_t ta = (int64_t) pos0 + n_cut - ER - gpos;
                    cut = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                    cut = (cut / SB) * SB;
                    if (!enc_reported) { enc_reported = true; std::fprintf(stderr, "enc-skip: first batch B=%d, cut=%d (phase A %d cols, phase B %d)\n", B, cut, cut, B - cut); }
                }
                double t0 = now_ms();
                Ds41Dense * cur_dense = &pipe.dense(0);
                Ds41MoeTier * cur_tier = &pipe.tier(0);
                int cur_st = 0;
                for (int l = 0; l < n_layer; ++l) {
                    if (stop_prefill()) return false;
                    const int st = ds41_stage_for_layer(pipe, l);
                    if (pipe.enabled() && st != cur_st) {
                        if (!ds41_sync_stages(pipe, cur_st, st, 0, B, ferr)) return false;
                        if (!ds41_begin_stage_continued(pipe, st, ids + c0, B, pos0 + c0, true, ferr)) return false;
                        cur_dense = &pipe.dense(st);
                        cur_tier = &pipe.tier(st);
                        cur_st = st;
                    }
                    if (l < 4) vram_log("batch layer", l);
                    // layer 20 is the last kv source: for phase A only its persistent KV writes run; layers 21-39 run
                    // nothing at all for phase A (their output columns are never read, and no later layer reads state
                    // a phase-A position would leave past layer 20)
                    const int d0 = (cut > 0 && l >= CL) ? cut : 0;
                    if (cut > 0 && l == CL) {
                        for (int s = 0; s < cut; s += SB) {
                            const int sn = std::min(SB, cut - s);
                            if (!cur_dense->attn_router_span_kv(l, s, sn)) { ferr = cur_dense->last_error(); return false; }
                        }
                    }
                    for (int s = d0; s < B; s += SB) {   // sub-batches in order: the layer's KV state advances as today
                        const int sn = std::min(SB, B - s);
                        if (!cur_dense->attn_router_span(l, s, sn)) { ferr = cur_dense->last_error(); return false; }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    // GPU: the tier writes the routed sums straight into the dense half's routed tensor (no 2 x n*D*4
                    // host round trip per layer); CPU / STRATA_ROUTED_HOST=1: through the host as before
                    float * rdev = routed_host ? nullptr : cur_dense->routed_dev();
                    if (l < 2) vram_log("pre-tier", l);
                    for (int s = d0; s < B; s += piece) {   // the experts for the dense rows (in pieces if the buffers cap)
                        const int pn = std::min(piece, B - s);
                        const bool ok = rdev ? cur_tier->run_chunk_dev(l, pn, cur_dense->batch_ids() + (size_t) s * top_k,
                                                                  cur_dense->batch_wts() + (size_t) s * top_k,
                                                                  cur_dense->batch_fn() + (size_t) s * n_embd,
                                                                  rdev + (size_t) s * n_embd)
                                             : cur_tier->run_chunk(l, pn, cur_dense->batch_ids() + (size_t) s * top_k,
                                                              cur_dense->batch_wts() + (size_t) s * top_k,
                                                              cur_dense->batch_fn() + (size_t) s * n_embd,
                                                              brouted.data() + (size_t) s * n_embd);
                        if (!ok) { ferr = "tier run refused"; return false; }
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!rdev && !cur_dense->upload_routed(brouted.data(), B)) { ferr = cur_dense->last_error(); return false; }
                    for (int s = d0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!cur_dense->finish_span(l, s, sn)) { ferr = cur_dense->last_error(); return false; }
                    }
                    // serve DSpark seed (bcap_n > 0): the main_hidden rows of the batch's last bcap_n tokens - pass_v's
                    // HC-mean of the state leaving dm_layers[j] - read here, so the tail need not re-run as passes
                    if (bcap_n > 0 && c0 + B == n) {
                        for (int j = 0; j < 3; ++j) {
                            if (l != dm_layers[j]) continue;
                            const int cn = std::min(bcap_n, B);
                            bcap_got = cn;
                            if (!cur_dense->read_state_cols(B - cn, cn, vx)) { ferr = cur_dense->last_error(); return false; }
                            const size_t HC = vx.size() / ((size_t) cn * n_embd);
                            for (int t = 0; t < cn; ++t) {
                                float * dst = bcap_vh.data() + (size_t) (bcap_n - cn + t) * 3 * n_embd + (size_t) j * n_embd;
                                for (size_t h = 0; h < HC; ++h) {
                                    const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                                    for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                                }
                            }
                        }
                    }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)
                    all_logits->resize((size_t) (c0 + B) * (size_t) nv);
                    for (int s = 0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!head_logits(s, sn, all_logits->data() + (size_t) (c0 + s) * nv)) return false;
                    }
                    if (c0 + B == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + B == n && !head_logits(B - 1, 1, last_logits.data())) return false;
                for (int s = 0; tail_keep > 0 && s < B; s += SB)   // one sub-batch at a time, as above
                    if (!capture_tail(n, c0 + s, std::min(SB, B - s), s)) return false;
                fd_logits += now_ms() - tl;
                if (vram_trace) {   // the batch's peak is its last layers' (the pass graph only grows within a prompt)
                    size_t tf = 0, tt = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                    std::fprintf(stderr, "vram trace: batch at %d (B %d) done: free %.2f GiB; %s\n", pos0 + c0, B,
                                 tf / 1073741824.0, dense.vram_ledger().c_str());
                }
                c0 += B;
            } else {
                // ---- the sequential chunk path (unchanged): one pass of up to NP tokens through every layer
                const int m = std::min(NP, n - c0);
                double t0 = now_ms();
                if (!pipe.dense(0).begin(ids + c0, m, pos0 + c0)) { ferr = pipe.dense(0).last_error(); return false; }
                Ds41Dense * cur_dense = &pipe.dense(0);
                Ds41MoeTier * cur_tier = &pipe.tier(0);
                int cur_st = 0;
                for (int l = 0; l < n_layer; ++l) {
                    if (stop_prefill()) return false;
                    const int st = ds41_stage_for_layer(pipe, l);
                    if (pipe.enabled() && st != cur_st) {
                        if (!ds41_sync_stages(pipe, cur_st, st, 0, m, ferr)) return false;
                        if (!ds41_begin_stage_continued(pipe, st, ids + c0, m, pos0 + c0, false, ferr)) return false;
                        cur_dense = &pipe.dense(st);
                        cur_tier = &pipe.tier(st);
                        cur_st = st;
                    }
                    const float * fn = nullptr;
                    if (!cur_dense->attn_router(l, cids.data(), cw.data(), &fn)) { ferr = cur_dense->last_error(); return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_fn.f32", l);
                        dumpf(nb, fn, (size_t) m * (size_t) n_embd * 4);
                        std::vector<int> sel; std::vector<float> smask; int K = 0;
                        if (cur_dense->read_sel(l, sel, smask, &K) && K > 0 && !sel.empty()) {
                            std::snprintf(nb, sizeof(nb), "L%02d_sel.i32", l);
                            dumpf(nb, sel.data(), sel.size() * 4);
                        }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    const bool ok = m == 1 ? cur_tier->run(l, cids.data(), cw.data(), fn, crouted.data())
                                           : cur_tier->run_chunk(l, m, cids.data(), cw.data(), fn, crouted.data());
                    if (!ok) { ferr = "tier run refused"; return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_routed.f32", l);
                        dumpf(nb, crouted.data(), (size_t) m * (size_t) n_embd * 4);
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!cur_dense->finish(l, crouted.data())) { ferr = cur_dense->last_error(); return false; }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {
                    all_logits->resize((size_t) (c0 + m) * (size_t) nv);
                    if (!head_logits(0, m, all_logits->data() + (size_t) c0 * nv)) return false;
                    if (c0 + m == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + m == n && !head_logits(m - 1, 1, last_logits.data())) return false;
                if (!capture_tail(n, c0, m, 0)) return false;
                fd_logits += now_ms() - tl;
                c0 += m;
            }
            progress(c0);
            if (yield_here) { feed_stopped = c0; break; }
        }
        if (dc.elastic_cols) {   // back to S columns
            if (!dense.set_columns(0)) { ferr = dense.last_error(); return false; }
            if (rdev1) rdev1 = dense.routed_dev();
        }
        tier.release_chunk();
        return true;
    };

    // ---- serve
    if (a.serve) {
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
        prefill_stop = stop_pending;
        auto is_stop = [&](int t) { return std::find(a.stop.begin(), a.stop.end(), t) != a.stop.end() || t == (int) G.eos; };
        const int ctxv = (int) ctx;
        std::vector<int> hist;
        int snap_len = -1;
        std::vector<float> snap_logits;   // the last prompt row's logits at the snapshot (a repeated prompt samples from them)
        // s6: DSpark's seed rows (main_hidden) of the snapshot's prompt tail - a repeated prompt (regenerate / a client
        // retry) feeds no new tokens, so without them DSpark had nothing to seed from and the whole answer decoded
        // without drafts (measured: 21.9 vs 31.8 tok/s on the same 1200-token edit).  cur_seed = this prompt's tail.
        std::vector<float> snap_seed, cur_seed;
        int snap_seed_n = 0, cur_seed_n = 0;
        int n_req = 0;
        // STRATA_SERVE_PARK (default on): an agent harness sends short UNRELATED requests between the turns of its main
        // conversation (titles, summaries).  Each wrote its own positions from 0 over the main conversation's caches, so
        // the next main turn re-read the whole prompt (2026-10-15 harness: a 454-token side call -> the 20,736-token
        // system prompt again, 69.5 s).  Now the main conversation is parked (Ds41Dense::park: the snapshot rings + the
        // rows the side request can write, to host memory) and unparked when a request continues it.
        // STRATA_SERVE_PARK_MIN: only conversations of >= this many tokens (default 2048); STRATA_SERVE_PARK_MAX: the
        // most positions a side request may write (prompt + max tokens, default 65536 ~ 1 GB of host memory).
        // conversation caching: one stage = the pinned conversation pool (#20, below); a layer split across cards =
        // buster's Ds41ConversationServe (it captures and restores every stage's state).  Same flags.
        Ds41ConversationServe conv_cache(pipe.n_stages() > 1 ? a.conv_mib : 0, a.conv_slots, a.conversation_cache_min_tokens,
                                         a.prompt_cache_every, a.prompt_cache_root);
        const bool conv_on = pipe.n_stages() > 1 && conv_cache.enabled() && a.prompt_cache > 0;
        const bool park_on = conv_on ? false : !(std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0');
        // (batch slots: 256 - a slot's conversation that another request takes over goes to the pool, and agents' turns
        // under 2K tokens were simply lost then: 20.10 card, conversations of 1.5-1.9K read again after an eviction)
        const int park_min = std::getenv("STRATA_SERVE_PARK_MIN") ? std::atoi(std::getenv("STRATA_SERVE_PARK_MIN")) : NS > 0 ? 256 : 2048;
        const int64_t park_max = std::getenv("STRATA_SERVE_PARK_MAX") ? std::atoll(std::getenv("STRATA_SERVE_PARK_MAX")) : 65536;
        // STRATA_SERVE_PARK_FLOOR (default 8192): the window a parked conversation reserves for the side requests that
        // FOLLOW it, not only for the one that triggered the park.  An agent harness sends several of them between two
        // turns: s1 2026-10-17 a 1,013-token title call parked the 92,792-token conversation with the 1,125 positions
        // THAT call needs, the next call (5,188 tokens) fell past that window, drop_park() threw the cache away, and
        // the main turn re-read 94,320 tokens: 250.71 s (the log: "park dropped (request reaches past 1125
        // positions)" -> "prompt=94320 -> reused=0").  Widening the window is nearly free: park() copies the snapshot
        // rings whole whatever this is, and only the compressor caches grow with it (~16 KiB a position, so 8192 is
        // ~130 MB on a 1 TB box).  0 = the old behaviour: the window of the one request that triggered the park.
        static const int64_t park_floor = std::getenv("STRATA_SERVE_PARK_FLOOR")
            ? std::atoll(std::getenv("STRATA_SERVE_PARK_FLOOR")) : 8192;
        std::vector<int> park_hist;       // the parked conversation's tokens (its snapshot length)
        std::vector<float> park_logits;
        int64_t park_rows = 0;            // positions saved: a request writing past this invalidates the park
        bool cache_open = !defer_vram;   // deferred: the arena only until the first decode
        // STRATA_SERVE_DSPARK leaves the live state DIRTY when its last verify pass rejected drafts: the ratio-2
        // compressor state ring then holds the rejected positions' states, so a request that continues the live
        // history at an odd position (its group's first half is the last accepted token) must re-feed that token -
        // the same rule as the in-request refeed (`dirty && (P + 1) & 1`).
        bool live_dirty = false;
        // ---- the conversation pool (--conversation-cache-mib, default 8192 MiB; 0 = the single park above): N agents
        // that take turns on one engine (each conversation differs only by its tokens - there is no session id)
        // re-read their whole prompt at every switch with one park, and a park needed the NEXT request's prompt +
        // max tokens to fit STRATA_SERVE_PARK_MAX (an agent's max_tokens 32K kept it from parking at all).  Now the
        // live conversation goes to the pool whenever a request does not continue it (at its turn boundary: the
        // snapshot at its prompt's end, every position-indexed row before it), and a request resumes from whichever
        // holds the longest prefix of it - the live state or a pool record (ties: the live one, no copy).  A position
        // matches when its token AND its identity do (conv_img_ident: which picture row; 0 = text), so image prompts
        // reuse like text.  STRATA_SERVE_PARK_MIN is the smallest conversation worth a record.  KV with --kv-fp4 is
        // ~890 bytes a position (+ ~6 MB of rings a record): 8 GiB holds ~9M positions, i.e. ~150 agents at 60K.
        const bool pool_on = park_on && a.conv_mib > 0 && pipe.n_stages() == 1;   // (the pool holds one stage's state)
        Ds41ConvStore pool;
        if (pool_on) {
            std::string perr;
            if (!pool.init(dense, ctx, (size_t) a.conv_mib << 20, (size_t) a.conv_slots, !ggml_backend_is_cpu(be), perr)) {
                std::fprintf(stderr, "ds41_generate: conversation pool: %s\n", perr.c_str());
                return 1;
            }
        }
        std::vector<uint64_t> live_id;   // hist's position identities (its prompt part; past it: text = 0)
        // the live conversation's turn boundary into the pool (logged as serve/server.py ConvCacheLog reads it)
        auto save_live = [&](const char * why) {
            if (!pool_on || snap_len < std::max(1, park_min) || hist.size() < (size_t) snap_len) return;
            const double t0 = now_ms();
            std::string perr;
            const int64_t id = pool.save(dense, hist.data(), live_id.data(), live_id.size(), snap_len,
                                         snap_logits.empty() ? nullptr : snap_logits.data(), snap_logits.size(), perr);
            std::fprintf(stderr, "strata serve: conversation cache: %s %d tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu (%s%s%s)\n",
                         id ? "parked" : "skipped", snap_len, now_ms() - t0, pool.size(), pool.bytes(), pool.evictions(), why,
                         id ? "" : ": ", id ? "" : perr.c_str());
        };
#if defined(DS41_HAVE_DSPARK)
        // STRATA_SERVE_DSPARK: DSpark's window (main_hidden of the last <= 128 positions) is seeded by running a
        // request's last min(128, new) prompt tokens through pass_v instead of the passes / batch (no extra work), and
        // it carries over when the next request continues this one (dsp_next = the position its ring expects next);
        // any other reuse resets it (lower acceptance for a while, never a different output).
        int dsp_next = -1;
        const int sp_depth_serve = std::getenv("STRATA_DSPARK_DEPTH") ? std::max(1, std::min(5, std::atoi(std::getenv("STRATA_DSPARK_DEPTH")))) : 4;
        long sv_pass = 0, sv_acc = 0;
#endif
        // STRATA_SERVE_STEP_MAX: new prompt tokens up to which a request runs on the decode side - passes of
        // STRATA_SERVE_PASS_M tokens with the VRAM cache and the decode biases (~40 tok/s) - instead of the exact
        // layer-major batch (~16.8 s + n / 250 tok/s; measured crossover ~765 tokens); 0 = always the batch (every prompt
        // token exact)
        const int step_max = std::getenv("STRATA_SERVE_STEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_STEP_MAX")) : 768;
        // STRATA_SERVE_PASS_BIAS=0: those passes route exactly (the decode biases only while decoding)
        const bool pass_bias = !(std::getenv("STRATA_SERVE_PASS_BIAS") && std::getenv("STRATA_SERVE_PASS_BIAS")[0] == '0');
        // Dual-V100: fill the VRAM expert cache at load (Qwen-like) instead of waiting for the first decode.
        // Layer-major batches still release/re-seed via the paths below. Opt out: STRATA_SERVE_SEED_AT_READY=0.
        if (defer_vram && !(std::getenv("STRATA_SERVE_SEED_AT_READY") && std::getenv("STRATA_SERVE_SEED_AT_READY")[0] == '0')) {
            if (!open_cache()) {
                std::fprintf(stderr, "serve: VRAM seed at ready failed: %s\n", err.c_str());
            } else {
                cache_open = true;
                int64_t resid = 0, arena_n = 0;
                for (int st = 0; st < pipe.n_stages(); ++st) {
                    resid += pipe.tier(st).resident();
                    arena_n += pipe.tier(st).arena_experts();
                }
                std::fprintf(stderr, "serve: VRAM ready seed -> %lld experts in VRAM, %lld arena (seed %.2f s)\n",
                             (long long) resid, (long long) arena_n, seed_s);
            }
        }
        int64_t resid_all = 0, arena_all = 0;
        for (int st = 0; st < pipe.n_stages(); ++st) {
            resid_all += pipe.tier(st).resident();
            arena_all += pipe.tier(st).arena_experts();
        }
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld", ctxv, (long long) resid_all, (long long) arena_all);
        if (pool_on) std::printf(" conversation_cache_mib=%d conversation_cache_slots=%d", a.conv_mib, a.conv_slots);
        if (NS > 0) std::printf(" batch_slots=%d slot_cache=1", NS);   // serve/server.py: the slot protocol (generate_batched)
        std::printf("\n");
        if (pool_on) std::fprintf(stderr, "serve: conversation pool %d MiB, %d slots, records >= %d tokens\n", a.conv_mib,
                                  a.conv_slots, std::max(1, park_min));
        if (pool_on) std::fprintf(stderr, "serve: conversation pool store: %s\n", pool.kind());
        std::printf("READY %d stop\n", ctxv);
        std::fflush(stdout);
        std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d)\n", ctxv, NP);
        // per-request decode biases (GEN keys route_bias= / route_bias_file=, from the server's "strata_tune"): a request
        // that names neither gets the --route-bias / --route-bias-file the engine started with, so one server can answer
        // exact-routing requests (both 0) beside the fast default without a restart.  apply_bias() writes zeros over a
        // bias it set earlier, so an exact request never inherits the previous one's.
        const float cfg_route_bias = a.route_bias, cfg_route_bias_file = a.route_bias_file, cfg_route_bias_remote = a.route_bias_remote;
        // the smart router's ranks (route_protect= / route_pool=, engines built with STRATA_RB_SMART / _PROTECT / _POOL):
        // a request that names neither gets the env's
        auto env_rank = [](const char * k) { const char * e = std::getenv(k); return e ? std::max(0, std::atoi(e)) : 0; };
        const int cfg_rprot = env_rank("STRATA_RB_PROTECT"), cfg_rpool = env_rank("STRATA_RB_POOL");
        int cur_rprot = cfg_rprot, cur_rpool = cfg_rpool;
        static const bool rewind1 = std::getenv("STRATA_SERVE_REWIND1") && std::getenv("STRATA_SERVE_REWIND1")[0] == '1';
        std::string line;
        bool previous_vision = false;
        // ---- batch slots (--batch-slots N; the protocol is at the top of this file).  Slot s's conversation is a Conv
        // (what the request path calls hist / live_id / snap_len / ..: those locals belong to the slot the request on
        // the control lines works in, `gslot`, and are swapped in and out around it) plus its device state (the core's
        // slot s, or the stub's copy).  A decoding slot is a BSlot.
        struct Conv {
            std::vector<int> hist;
            std::vector<uint64_t> live_id;
            int snap_len = -1;
            std::vector<float> snap_logits, snap_seed;
            int snap_seed_n = 0;
            bool live_dirty = false;
            double used = 0;
        };
        struct RouteCls {
            float rb = 0, rbf = 0;
            int prot = 0, pool = 0;
            bool operator==(const RouteCls & o) const { return rb == o.rb && rbf == o.rbf && prot == o.prot && pool == o.pool; }
        };
        struct BSlot {
            bool active = false, stop = false;
            int x = 0, p = 0;                  // the token the next window feeds, at position p
            int64_t produced = 0, max_new = 0;
            double t0 = 0;
            float temp = 0, top_p = 1, min_p = 0;
            int top_k = 0;
            std::mt19937_64 rng;
            RouteCls cls;
        };
        std::vector<Conv> cv((size_t) NS);
        std::vector<BSlot> bs((size_t) NS);
        std::vector<RouteCls> cls_seen;      // routing classes by id (the planner compares ids)
        auto cls_id = [&](const RouteCls & c) -> int {
            for (size_t i = 0; i < cls_seen.size(); ++i) if (cls_seen[i] == c) return (int) i;
            cls_seen.push_back(c);
            return (int) cls_seen.size() - 1;
        };
        int gslot = -1;                      // the slot of the request on the control lines (-1: none)
        int64_t batch_epoch = 0;             // bumped by every window / admission / swap (DSpark's window is the solo path's)
#if defined(DS41_HAVE_DSPARK)
        int dsp_slot = -1;
        int64_t dsp_epoch = -1;
#endif
        auto conv_pos = [&](int s) -> int64_t {   // the positions slot s's conversation has in its state
            const std::vector<int> & h = s == gslot ? hist : cv[(size_t) s].hist;
            const int sl = s == gslot ? snap_len : cv[(size_t) s].snap_len;
            return std::max<int64_t>((int64_t) h.size(), sl);
        };
        Ds41SlotsStub stub;
        if (NS > 0 && !slots_core)
            stub.init(dense, NS, conv_pos, [&]() { if (pool_on) pool.fence(); });
        auto bind = [&](int s) -> bool {
            if (slots_core ? !ds41_k_bind(dense, s) : !stub.bind_slot(s, ferr)) {
                if (slots_core) ferr = dense.last_error();
                return false;
            }
            return true;
        };
        auto conv_swap = [&](Conv & c) {   // the request path's locals <-> a slot's record
            std::swap(hist, c.hist);
            std::swap(live_id, c.live_id);
            std::swap(snap_len, c.snap_len);
            std::swap(snap_logits, c.snap_logits);
            std::swap(snap_seed, c.snap_seed);
            std::swap(snap_seed_n, c.snap_seed_n);
            std::swap(live_dirty, c.live_dirty);
        };
        // exchange two idle slots' conversations (device state and records): a conversation moves to the slot id the
        // server asked for, or the one a yielded read waits in
        auto swap_conv = [&](int x, int y) -> bool {
            if (x == y) return true;
            if (slots_core ? !ds41_k_swap(dense, x, y) : !stub.swap_slots(x, y, ferr)) {
                if (slots_core) ferr = dense.last_error();
                return false;
            }
            std::swap(cv[(size_t) x], cv[(size_t) y]);
            ++batch_epoch;
            return true;
        };
        // what slot s's live state offers a prompt (the request path's own rules): its history continued, or its
        // snapshot restored
        auto live_offer = [&](const Conv & c, const std::vector<int> & q, const std::vector<uint64_t> & qid) -> int {
            const size_t l = conv_common_prefix(c.hist.data(), c.live_id.data(), c.live_id.size(), c.hist.size(), q.data(),
                                                qid.data(), qid.size(), q.size());
            if (!c.hist.empty() && l == c.hist.size() && l < q.size()) return (int) l;
            if (!c.hist.empty() && l == c.hist.size() && l == q.size() && l >= 2 && !c.live_dirty &&
                (c.snap_len < 0 || (size_t) c.snap_len < q.size()))
                return (int) l - 1;   // (cont_last below)
            if (c.snap_len > 0 && l >= (size_t) c.snap_len && (size_t) c.snap_len <= q.size()) return c.snap_len;
            return 0;
        };
        auto any_active = [&]() { for (const BSlot & b : bs) if (b.active) return true; return false; };
        // a conversation that has ended in its slot goes to the pool (its turn boundary, as save_live)
        auto save_conv = [&](int s, const char * why) {
            Conv & c = cv[(size_t) s];
            if (!pool_on || c.snap_len < std::max(1, park_min) || c.hist.size() < (size_t) c.snap_len) return;
            if (!bind(s)) { std::fprintf(stderr, "strata batch: slot %d: %s\n", s, ferr.c_str()); return; }
            const double t0 = now_ms();
            std::string perr;
            const int64_t id = pool.save(dense, c.hist.data(), c.live_id.data(), c.live_id.size(), c.snap_len,
                                         c.snap_logits.empty() ? nullptr : c.snap_logits.data(), c.snap_logits.size(), perr);
            std::fprintf(stderr, "strata serve: conversation cache: %s %d tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu (%s, slot %d%s%s)\n",
                         id ? "parked" : "skipped", c.snap_len, now_ms() - t0, pool.size(), pool.bytes(), pool.evictions(), why,
                         s, id ? "" : ": ", id ? "" : perr.c_str());
        };
        auto try_line = [&](std::string & out) -> bool {
            std::lock_guard<std::mutex> lk(qm);
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        // the slot lines that may come while a prompt is read: BSTOP is taken at once; BYIELD (`yield`: the slot it
        // names, else -1) - the rest wait their turn
        auto take_slot_lines = [&](int * yield) {
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end();) {
                if (it->rfind("BSTOP ", 0) == 0) {
                    const int b = std::atoi(it->c_str() + 6);
                    if (b >= 0 && b < NS) bs[(size_t) b].stop = true;
                    it = q.erase(it);
                } else if (it->rfind("BYIELD", 0) == 0) {
                    if (yield) *yield = it->size() > 7 ? std::atoi(it->c_str() + 7) : -1;
                    it = q.erase(it);
                } else {
                    ++it;
                }
            }
        };
        auto set_class = [&](const RouteCls & c) {   // the dense half's decode biases for this window's rows
            if (a.route_bias != c.rb || a.route_bias_file != c.rbf || !bias_now) {
                a.route_bias = c.rb;
                a.route_bias_file = c.rbf;
                bias_now = true;
                apply_bias();
            }
            if ((c.prot != cur_rprot || c.pool != cur_rpool) && dense.set_route_select(c.prot, c.pool)) {
                cur_rprot = c.prot;
                cur_rpool = c.pool;
            }
        };
        static const bool btrace = std::getenv("STRATA_BATCH_TRACE") != nullptr;
        // STRATA_BATCH_MIX_MAX: a BGEN prompt of at most this many new tokens is read in the windows (its rows beside
        // the decoding slots'); a longer one alone (layer-major), the slots waiting.  Default: STRATA_SERVE_STEP_MAX.
        const int mix_max = std::getenv("STRATA_BATCH_MIX_MAX") ? std::atoi(std::getenv("STRATA_BATCH_MIX_MAX")) : step_max;
        constexpr int kRows = Ds41MoeTier::kMaxMulti;
        // the prompt being read in the windows (its slot, its rows still to go, its class)
        struct Mix {
            int slot = -1, pos = 0, left = 0;
            const int * ids = nullptr;            // the whole prompt (Engram history: the 3 tokens before a pass)
            const uint64_t * rid = nullptr;
            size_t nrid = 0;
            RouteCls cls;
        };
        std::vector<float> wlog, wlog2;
        std::vector<int> wslot, wpos, wtok, wwant;
        std::vector<Ds41Dense::Row> wrows;
        Ds41RowsStep rows_step(dense, tier);
        // STRATA_BATCH_SERIAL=1: every window slot by slot through the old single-sequence path (bind + step / pass_m)
        // on the core's per-slot states - the solo engine's numbers exactly (the protocol checks), not its speed
        static const bool slots_serial = std::getenv("STRATA_BATCH_SERIAL") && std::getenv("STRATA_BATCH_SERIAL")[0] == '1';
        int win_rr = 0;
        int64_t bw_windows = 0, bw_rows = 0, bw_read = 0, bw_tokens = 0;
        double bw_ms = 0, bw_start = 0;
        // the stub's Engram history: one history for every slot, so a slot's pass at p rewrites the 3 tokens before it
        auto engram_tail = [&](int p, const int * toks, const uint64_t * idn, size_t nidn) {
            if (slots_core || p <= 0) return;
            const int h0 = std::max(0, p - 3);
            uint8_t dead[3] = { 0, 0, 0 };
            for (int i = h0; i < p; ++i) dead[i - h0] = (size_t) i < nidn && idn[(size_t) i] != 0;
            dense.set_engram_history(h0, p - h0, toks + h0, dead);
        };
        auto end_slot = [&](int b, const char * fin) {
            BSlot & sl = bs[(size_t) b];
            std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, now_ms() - sl.t0);
            sl.active = false;
            sl.stop = false;
            if (std::strcmp(fin, "nonfinite") == 0) {   // its state is not worth keeping
                cv[(size_t) b].hist.clear();
                cv[(size_t) b].live_id.clear();
                cv[(size_t) b].snap_len = -1;
            } else {
                save_conv(b, "slot done");
            }
            cv[(size_t) b].used = now_ms();
        };
        // one batch window: every decoding slot of one routing class + rows of the prompt being read (`mx`); the rows
        // of each slot through the dense half: the core's begin_rows when several slots share it, else the old step /
        // pass_m on the bound slot (one slot left = the solo engine).  The read's last row's logits -> last_logits.
        auto run_window = [&](Mix * mx) -> bool {
            for (int b = 0; b < NS; ++b)   // BSTOPped slots end before their next row
                if (bs[(size_t) b].active && bs[(size_t) b].stop) end_slot(b, "cancel");
            std::vector<Ds41PlanSlot> ps((size_t) NS);
            for (int b = 0; b < NS; ++b) ps[(size_t) b] = { bs[(size_t) b].active, bs[(size_t) b].active ? cls_id(bs[(size_t) b].cls) : -1 };
            const Ds41Plan pl = ds41_plan_window(ps, mx ? mx->slot : -1, mx ? cls_id(mx->cls) : -1, mx ? mx->left : 0, kRows, win_rr);
            if (pl.cls < 0) { std::fflush(stdout); return true; }
            const double tw0 = now_ms();
            if (bw_windows == 0) bw_start = tw0;
            set_class(cls_seen[(size_t) pl.cls]);
            if (defer_vram && !cache_open) {   // a long read released the cache: it opens into what is free now
                if (!open_cache()) { std::printf("ERR batch window: VRAM seed failed: %s\n", err.c_str()); return false; }
                cache_open = true;
                set_class(cls_seen[(size_t) pl.cls]);
            }
            const int A = (int) pl.dec.size(), R = pl.read_rows, n = A + R;
            wslot.assign((size_t) n, 0); wpos.assign((size_t) n, 0); wtok.assign((size_t) n, 0);
            for (int i = 0; i < A; ++i) {
                const BSlot & sl = bs[(size_t) pl.dec[(size_t) i]];
                wslot[(size_t) i] = pl.dec[(size_t) i]; wpos[(size_t) i] = sl.p; wtok[(size_t) i] = sl.x;
            }
            for (int i = 0; i < R; ++i) {
                wslot[(size_t) (A + i)] = mx->slot; wpos[(size_t) (A + i)] = mx->pos + i; wtok[(size_t) (A + i)] = mx->ids[mx->pos + i];
            }
            const bool read_ends = R > 0 && R == mx->left;
            wlog.resize((size_t) std::max(n, 1) * (size_t) nv);
            int groups = 0;
            for (int i = 0; i < n; ++i) groups += i == 0 || wslot[(size_t) i] != wslot[(size_t) i - 1];
            bool ok = true;
            if (slots_core && groups > 1 && !slots_serial) {
                // ---- one pass over every row (the core's per-slot states, Ds41RowsStep): the dense half batched over the
                // rows, attention per slot, the tier once per layer over the union of the rows' experts
                wrows.resize((size_t) n);
                for (int i = 0; i < n; ++i) wrows[(size_t) i] = Ds41Dense::Row{ wslot[(size_t) i], wpos[(size_t) i], wtok[(size_t) i] };
                wwant.clear();
                for (int i = 0; i < A; ++i) wwant.push_back(i);
                if (read_ends) wwant.push_back(n - 1);
                if (!rows_step.run(wrows.data(), n, wwant.data(), (int) wwant.size(), rdev1, wlog2, ferr)) ok = false;
                for (size_t k = 0; ok && k < wwant.size(); ++k)
                    std::copy_n(wlog2.data() + k * (size_t) nv, nv, wlog.data() + (size_t) wwant[k] * nv);
            } else {
                // ---- slot by slot through the old single-sequence path (the stub; or one slot in the window): the
                // read's rows last, so its slot is the bound one afterwards
                for (int i0 = 0; ok && i0 < n;) {
                    int i1 = i0 + 1;
                    while (i1 < n && wslot[(size_t) i1] == wslot[(size_t) i0]) ++i1;
                    const int s = wslot[(size_t) i0], m = i1 - i0, p0 = wpos[(size_t) i0];
                    if (!bind(s)) { ok = false; break; }
                    if (i0 < A) {
                        const Conv & c = cv[(size_t) s];
                        engram_tail(p0, c.hist.data(), c.live_id.data(), c.live_id.size());
                    } else {
                        engram_tail(p0, mx->ids, mx->rid, mx->nrid);
                    }
                    if (!(m == 1 ? step(wtok[(size_t) i0], p0) : pass_m(wtok.data() + i0, m, p0))) { ok = false; break; }
                    const bool want = i0 < A || (read_ends && i1 == n);
                    if (want && !dense.logits(m - 1, 1, wlog.data() + (size_t) (i1 - 1) * nv)) { ferr = dense.last_error(); ok = false; }
                    i0 = i1;
                }
            }
            if (!ok) {
                std::printf("ERR batch window: %s\n", ferr.c_str());
                std::fflush(stdout);
                return false;
            }
            ++batch_epoch;
            for (int i = 0; i < A; ++i) {
                const int b = pl.dec[(size_t) i];
                BSlot & sl = bs[(size_t) b];
                const float * lg = wlog.data() + (size_t) i * nv;
                bool finite = true;
                for (int v = 0; v < nv && finite; ++v) finite = std::isfinite(lg[v]);
                if (!finite) {
                    std::fprintf(stderr, "strata batch: non-finite logits in slot %d at position %d: it ends (BDONE nonfinite)\n", b, sl.p);
                    end_slot(b, "nonfinite");
                    continue;
                }
                const int y = sample_p(lg, nv, sl.temp, sl.top_k, sl.top_p, sl.min_p, sl.rng);
                cv[(size_t) b].hist.push_back(sl.x);   // the fed token is in the slot's state now
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                ++bw_tokens;
                const char * fin = is_stop(y) ? "stop" : sl.stop ? "cancel" : (sl.produced >= sl.max_new && sl.max_new > 0) ? "length"
                                 : sl.p + 2 > ctxv ? "length" : nullptr;
                if (fin) { end_slot(b, fin); continue; }
                sl.x = y;
                sl.p += 1;
            }
            if (R > 0) {
                if (read_ends) std::copy(wlog.begin() + (size_t) (n - 1) * nv, wlog.begin() + (size_t) n * nv, last_logits.begin());
                mx->pos += R;
                mx->left -= R;
            }
            std::fflush(stdout);
            tier.rebalance_tick();
            const double wms = now_ms() - tw0;
            ++bw_windows;
            bw_rows += A;
            bw_read += R;
            bw_ms += wms;
            if (btrace) {
                std::string who;
                for (int i = 0; i < A; ++i) who += (i ? "," : "") + std::to_string(pl.dec[(size_t) i]);
                std::fprintf(stderr, "strata batch: window %lld class %d (bias %.2f/%.2f) decode [%s] read %d%s - %.1f ms\n",
                             (long long) bw_windows, pl.cls, cls_seen[(size_t) pl.cls].rb, cls_seen[(size_t) pl.cls].rbf,
                             who.c_str(), R, R ? (" of slot " + std::to_string(mx->slot)).c_str() : "", wms);
            }
            if (!any_active() && !(mx && mx->left > 0)) {   // all idle: one line for the run of windows
                const double wall = now_ms() - bw_start;
                std::fprintf(stderr, "strata batch: %lld windows, %lld decode rows + %lld prompt rows in %.2f s (windows %.2f s) = "
                                     "%.1f tok/s decoded, %.2f rows/window\n", (long long) bw_windows, (long long) bw_rows,
                             (long long) bw_read, wall / 1000.0, bw_ms / 1000.0, 1000.0 * bw_tokens / std::max(wall, 1e-9),
                             (double) (bw_rows + bw_read) / std::max<int64_t>(1, bw_windows));
                if (slots_core && !slots_serial) std::fprintf(stderr, "strata batch: rows passes: %s\n", rows_step.stats_line(true).c_str());
                bw_windows = bw_rows = bw_read = bw_tokens = 0;
                bw_ms = 0;
            }
            return true;
        };
        // the slot of a request: an admission (BGEN) takes its slot - first exchanged with an idle slot whose live state
        // offers more of the prompt; a solo request (GEN) the idle slot that offers the most, else an empty one, else
        // the one used longest ago.  -1: no slots.
        auto pick_slot = [&](int admit, const std::vector<int> & q, const std::vector<uint64_t> & qid) -> int {
            if (NS <= 0) return -1;
            auto empty = [&](int k) { return cv[(size_t) k].hist.empty() && cv[(size_t) k].snap_len < 0; };
            int target = admit, best = admit >= 0 ? live_offer(cv[(size_t) admit], q, qid) : -1, bk = -1;
            for (int k = 0; k < NS; ++k) {
                if (k == admit || bs[(size_t) k].active) continue;
                const int o = live_offer(cv[(size_t) k], q, qid);
                if (admit >= 0) {
                    if (o > best) { best = o; bk = k; }
                } else if (target < 0 || o > best ||
                           (o == best && (empty(k) != empty(target) ? empty(k) : cv[(size_t) k].used < cv[(size_t) target].used))) {
                    target = k;
                    best = o;
                }
            }
            if (admit >= 0 && bk >= 0) {
                if (swap_conv(admit, bk)) std::fprintf(stderr, "strata batch: slot %d takes slot %d's conversation (%d tokens of the prompt)\n", admit, bk, best);
                else std::fprintf(stderr, "strata batch: swap %d <-> %d failed: %s\n", admit, bk, ferr.c_str());
            }
            return target;
        };
        if (NS > 0)
            std::fprintf(stderr, "serve: batch slots %d (%s), prompts of <= %d new tokens read in the windows\n", NS,
                         !slots_core ? "stub" : slots_serial ? "core, windows slot by slot" : "core", mix_max);
        for (;;) {
            if (NS > 0 && any_active()) {   // slots decoding: a window whenever no line is waiting
                if (!try_line(line)) {
                    if (!run_window(nullptr)) { std::fflush(stdout); std::fflush(stderr); std::_Exit(1); }
                    continue;
                }
            } else if (!next_line(line)) {
                break;
            }
            if (line == "QUIT") break;
            if (line == "STOP" || line.empty()) continue;
            // the slot lines: BSTOP ends a slot at its next window; BYIELD for a read that has ended meanwhile; BGEN
            // <slot> <max_new> .. reads the prompt into <slot> as a GEN 1, then the slot decodes on (BADM)
            int admit = -1;
            long long admit_max = 0;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < NS) bs[(size_t) b].stop = true;
                continue;
            }
            if (line.rfind("BYIELD", 0) == 0) continue;
            if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
                const bool bimg = line.rfind("BGENI ", 0) == 0;
                char * e1 = nullptr;
                const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
                char * e2 = nullptr;
                const long long mn = std::strtoll(e1, &e2, 10);
                if (b < 0 || b >= NS || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                    std::printf("ERR BGEN: no such free slot (--batch-slots %d) or a bad max_new\n", NS);
                    std::fflush(stdout);
                    continue;
                }
                admit = (int) b;
                admit_max = mn;
                line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
            }
            dense.clear_vision();
            if (previous_vision && !pool_on) {   // the single park: token-only reuse, an image prompt is never continued
                hist.clear();
                live_id.clear();
                snap_len = -1;
                previous_vision = false;
            }
            // GENI = GEN with an image-embedding file: GENI <max_new> [k=v ..] <embeddings file> <ids csv> (the file
            // holds one strata-vision record per image, in prompt order; the engine lands them on the image_token runs).
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
            std::istringstream ss(line);
            std::string w, csv, emb_file;
            int max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            float req_rb = cfg_route_bias, req_rbf = cfg_route_bias_file, req_rbr = cfg_route_bias_remote;
            int req_rprot = cfg_rprot, req_rpool = cfg_rpool;
            std::vector<uint64_t> req_img_keys;   // img_keys=<hex>,<hex>.. (GENI): the server's key per picture, in prompt order
            ss >> w >> max_new;
            while (ss >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) {   // the two positionals: GENI's embeddings file then the csv (GEN: just the csv)
                    if (geni && emb_file.empty()) emb_file = w; else csv = w;
                    continue;
                }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                if (k == "temperature") temp = (float) std::atof(v.c_str());
                else if (k == "top_p") top_p = (float) std::atof(v.c_str());
                else if (k == "top_k") tk = std::atoi(v.c_str());
                else if (k == "min_p") min_p = (float) std::atof(v.c_str());
                else if (k == "seed") rng.seed((uint64_t) std::atoll(v.c_str()));
                else if (k == "route_bias") req_rb = std::max(0.0f, (float) std::atof(v.c_str()));
                else if (k == "route_bias_file") req_rbf = std::max(0.0f, (float) std::atof(v.c_str()));
                else if (k == "route_bias_remote") req_rbr = std::max(0.0f, (float) std::atof(v.c_str()));
                else if (k == "route_protect") req_rprot = std::max(0, std::atoi(v.c_str()));
                else if (k == "route_pool") req_rpool = std::max(0, std::atoi(v.c_str()));
                else if (k == "img_keys") {
                    std::stringstream ks(v);
                    std::string h;
                    while (std::getline(ks, h, ',')) {
                        char * e = nullptr;
                        const unsigned long long x = std::strtoull(h.c_str(), &e, 16);
                        if (h.empty() || !e || *e != '\0') { req_img_keys.clear(); break; }
                        req_img_keys.push_back((uint64_t) x);
                    }
                }
            }
            // a changed bias is applied again just before the decode: a repeated prompt (regenerate) runs no prompt pass,
            // and the passes are what turn the decode bias on - without this it decoded with the previous request's
            bool bias_changed = false;
            if (req_rb != a.route_bias || req_rbf != a.route_bias_file || req_rbr != a.route_bias_remote) {
                a.route_bias = req_rb;
                a.route_bias_file = req_rbf;
                a.route_bias_remote = req_rbr;
                bias_changed = true;
                std::fprintf(stderr, "serve: decode route bias %.2f, remote %.2f, file penalty %.2f%s\n", req_rb, req_rbr, req_rbf,
                             (req_rb == 0.0f && req_rbf == 0.0f && req_rbr == 0.0f) ? " (exact routing)" : "");
            }
            if ((req_rprot != cur_rprot || req_rpool != cur_rpool) && dense.set_route_select(req_rprot, req_rpool)) {
                cur_rprot = req_rprot;
                cur_rpool = req_rpool;
                std::fprintf(stderr, "serve: smart routing: exact top %d always kept, picks inside the exact top %d%s\n",
                             req_rprot, req_rpool, (req_rprot == 0 && req_rpool == 0) ? " (off: the flat bias)" : "");
            }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); continue; }
            if ((int64_t) ids.size() + std::max(0, max_new) > ctxv) {
                std::printf("ERR prompt (%zu) + max tokens (%d) exceed the context (%d)\n", ids.size(), max_new, ctxv);
                std::fflush(stdout);
                continue;
            }
            // vision: resolve this request's image rows onto its image_token runs and hand them to the dense layers
            std::vector<float> vis_rows;
            std::vector<int64_t> vis_pos;
            std::vector<uint64_t> rid;   // this prompt's position identities (empty: all text)
            const bool is_vision = geni;
            if (is_vision) {
                if (!dense.vision()) { std::printf("ERR image input requires --vision\n"); std::fflush(stdout); continue; }
                std::string verr;
                std::vector<std::vector<float>> recs;
                std::vector<int64_t> counts;
                const int64_t ne = dense.geom().n_embd;
                if (!read_sve1(emb_file, recs, counts, ne, (int64_t) ids.size(), verr)
                    || !map_vision(ids, a.image_token, recs, counts, vis_rows, vis_pos, verr)) {
                    std::printf("ERR %s\n", verr.c_str()); std::fflush(stdout); continue;
                }
                if (vis_rows.empty()) { std::printf("ERR GENI prompt has no image tokens\n"); std::fflush(stdout); continue; }
                if (pool_on) {
                    // every image span is the same run of image_token: the positions' identities tell the pictures
                    // apart - the server's key for picture k (img_keys=, its normalized bytes' hash) or, without
                    // keys, a hash of its rows - so reuse compares (token, identity) and an image prompt reuses too
                    rid.assign(ids.size(), 0);
                    const bool keyed = req_img_keys.size() == recs.size();
                    size_t r = 0;
                    for (size_t k = 0; k < recs.size(); ++k) {
                        uint64_t key = 0;
                        if (keyed) key = conv_mix64(req_img_keys[k]);
                        else {
                            key = conv_fnv1a(&counts[k], sizeof(counts[k]));
                            key = conv_fnv1a(recs[k].data(), recs[k].size() * sizeof(float), key);
                        }
                        for (int64_t j = 0; j < counts[k] && r < vis_pos.size(); ++j, ++r)
                            rid[(size_t) vis_pos[r]] = conv_img_ident(key, j);
                    }
                } else {
                    // the single park: an image prompt prefills fresh (a token-only reuse test would "continue" a
                    // different image: same ids, different embeddings).  Empty hist / no snapshot / no park.
                    dense.drop_park();
                    park_hist.clear();
                    hist.clear();
                    live_id.clear();
                    snap_len = -1;
                }
                dense.set_vision(vis_rows.data(), vis_pos.data(), (int) vis_pos.size());
                previous_vision = true;
            } else {
                dense.clear_vision();
            }
            // batch slots: the request works in its slot (an admission's, or the idle one that offers the most): the dense
            // half is bound to it, and hist / live_id / snap_len / .. below are its conversation until this iteration ends
            const int my_slot = pick_slot(admit, ids, rid);
            struct SlotScope {
                std::function<void()> f;
                ~SlotScope() { if (f) f(); }
            } slot_scope;
            if (my_slot >= 0) {
                if (!bind(my_slot)) {
                    std::printf("ERR batch slot %d: %s\n", my_slot, ferr.c_str());
                    std::fflush(stdout);
                    continue;
                }
                conv_swap(cv[(size_t) my_slot]);
                gslot = my_slot;
                if (btrace) std::fprintf(stderr, "strata batch: request in slot %d (%s)\n", my_slot, admit >= 0 ? "BGEN" : "GEN");
                slot_scope.f = [&, my_slot]() {
                    conv_swap(cv[(size_t) my_slot]);
                    cv[(size_t) my_slot].used = now_ms();
                    gslot = -1;
                    feed_yield = nullptr;
                    dense.clear_vision();   // a window's rows are text (and these rows die with the request)
                };
#if defined(DS41_HAVE_DSPARK)
                if (admit < 0 && (my_slot != dsp_slot || batch_epoch != dsp_epoch)) dsp_next = -1;   // another conversation
#endif
                if (admit >= 0) ++batch_epoch;
            }
            size_t lcp = conv_common_prefix(hist.data(), live_id.data(), live_id.size(), hist.size(), ids.data(), rid.data(),
                                            rid.size(), ids.size());
            // batch slots: the live state holds the WHOLE prompt (a solo request STOPped for a promotion fed its last token
            // before it saw the STOP): continue it by re-feeding that one token instead of restoring the prompt's snapshot
            // and re-reading everything generated since (the ring and ratio-2 states are position-indexed: re-feeding the
            // last position writes what is there)
            const bool cont_last = NS > 0 && !hist.empty() && lcp == hist.size() && lcp == ids.size() && lcp >= 2 && !live_dirty &&
                                   (snap_len < 0 || (size_t) snap_len < ids.size());
            // ... and such a request (the prompt is the slot's own history - its turn and what it generated - plus at most
            // the last generated token) keeps the turn's snapshot at the prompt's end: a promotion is not a turn boundary,
            // and the client's next turn (or the same prompt again) resumes at that end, not 2 tokens into the reply
            // (20.10 card: snapshots taken at prompt + 2 left 5 of 8 conversations unreachable at N=8)
            const bool keep_snap = NS > 0 && snap_len > 0 && !hist.empty() && lcp + 1 >= hist.size() &&
                                   ids.size() <= hist.size() + 1 && (size_t) snap_len < ids.size();
            if (conv_on && hist.empty()) {   // a layer split: the best parked conversation in buster's cache
                const int restored = conv_cache.try_restore(pipe, ids, hist, snap_len, snap_logits, err);
                if (restored > 0) lcp = (size_t) restored;
            }
            bool unparked = false;
            int reuse_override = -1;   // a partial unpark: resume at the longest common prefix (see below)
            size_t lp_show = 0;        // how far this request actually continues the PARKED conversation (logged)
            bool pool_regen = false;   // a pool record's boundary IS this prompt: sample from its saved logits
            if (pool_on) {
                // what the live state offers (the rules below): its history continued, or its snapshot restored
                int live = 0;
                if (!hist.empty() && lcp == hist.size() && lcp < ids.size()) live = (int) lcp;
                else if (cont_last) live = (int) lcp - 1;
                else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len <= ids.size()) live = snap_len;
                const ConvMatch m = pool.best_prefix(ids.data(), rid.data(), rid.size(), (int64_t) ids.size());
                int64_t pr = m.resume();
                const bool exact = m.pos > 0 && m.pos == (int64_t) ids.size();
                if (exact && !m.has_logits) pr = 0;                      // nothing to sample the first token from
                if (!exact && pr >= (int64_t) ids.size()) pr = 0;        // a partial resume must feed something
                lp_show = (size_t) m.lcp;
                if (pr > live) {
                    save_live("switch");
                    ConvRestore rs;
                    std::string perr;
                    const double t0 = now_ms();
                    if (pool.restore(dense, m, (int64_t) std::min<size_t>(lcp, (size_t) pr), rs, perr)) {
                        dense.rearm((int) rs.pos);
                        dense.restore();
                        // positions [0, pr) are this prompt's own (the match is strict on token + identity)
                        hist.assign(ids.begin(), ids.begin() + pr);
                        live_id.assign(rid.begin(), rid.begin() + std::min<size_t>(rid.size(), (size_t) pr));
                        if (pr == rs.pos) { snap_len = (int) rs.pos; snap_logits = rs.logits; }
                        else snap_len = -1;
                        snap_seed_n = 0;   // no DSpark seed rows in a record: a repeat decodes its first tokens draftless
                        live_dirty = false;
                        reuse_override = (int) pr;
                        pool_regen = exact;
                        if (exact) last_logits = rs.logits;
                        lcp = (size_t) pr;
                        unparked = true;
#if defined(DS41_HAVE_DSPARK)
                        dsp_next = -1;
#endif
                        std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                                     (long long) pr, pr == rs.pos ? "pool" : "pool, partial", now_ms() - t0, pool.size(), pool.bytes());
                    } else {
                        std::fprintf(stderr, "strata serve: conversation cache: restore failed (%s)\n", perr.c_str());
                        if (rs.live_clobbered) { hist.clear(); live_id.clear(); snap_len = -1; lcp = 0; }
                    }
                }
            } else if (dense.parked()) {   // does this request continue the parked conversation (better than the live one)?
                size_t lp = 0;
                while (lp < park_hist.size() && lp < ids.size() && park_hist[lp] == ids[lp]) ++lp;
                lp_show = lp;
                // STRATA_SERVE_PARK_PARTIAL (default on): a request that continues MOST of the parked conversation is
                // worth restoring, not only one that continues ALL of it.  A chat template that re-renders its tail -
                // or an EMPTY assistant turn, as the s13 harness produces - leaves lp just under park_hist.size(), and
                // requiring an exact prefix threw away a 21,349-token cache over a 27-token difference: run8/run15 both
                // logged "reuse lcp=27 hist=0 snap=-1 prompt=23531 -> reused=0" and re-read the whole prompt.
                // Positional caches make a partial resume safe: unpark() restores positions [0, park_size) and the
                // forward pass then re-feeds from lp, overwriting the stale tail (that is the documented fix: "resume
                // at the longest common prefix even when it is below the snapshot").
                static const bool park_partial = !(std::getenv("STRATA_SERVE_PARK_PARTIAL") && std::getenv("STRATA_SERVE_PARK_PARTIAL")[0] == '0');
                const size_t lp_min = (size_t) std::max<int64_t>(park_min, (int64_t) park_hist.size() * 3 / 4);
                // ...but NOT every cache is positional: the window ring at the park's end P holds positions [P - RAW, P)
                // (RAW = SWA + 16) and the ratio-2 compressor state P - 2, P - 1.  A resume at s < P reads the window
                // [s - SWA + 1, s) from the ring and, at an odd s, the state of s - 1: exact only for an even s with
                // P - s <= RAW - SWA = 16 - and only after restore() (the live rings are the side request's).  The
                // old path resumed at any lp with the side request's rings (tools/v41/engine/check_conv_reuse.py
                // park_partial, CPU: 20 back -> KL 2.1 against a fresh prefill, top-1 wrong).
                const size_t ps = lp & ~(size_t) 1;
                if (park_partial && lp < park_hist.size() && lp >= lp_min && ps > lcp && park_hist.size() - ps <= 16
                    && dense.unpark() && dense.restore()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation at %zu (lcp %zu; partial: re-feeds from there)\n",
                                 park_hist.size(), ps, lp);
                    hist = park_hist;
                    snap_len = -1;        // the tail under the park's end is re-fed, so no snapshot survives this
                    reuse_override = (int) ps;
                    lcp = ps;
                    unparked = true;
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;        // DSpark's window holds the side request's states
#endif
                } else if (lp == park_hist.size() && lp > lcp && dense.unpark()) {
                    std::fprintf(stderr, "serve: unparked the %zu-token conversation\n", park_hist.size());
                    hist = park_hist;
                    snap_len = (int) park_hist.size();
                    snap_logits = park_logits;
                    snap_seed_n = 0;
                    lcp = lp;
                    unparked = true;   // the live rings are the side request's: resume through restore()
#if defined(DS41_HAVE_DSPARK)
                    dsp_next = -1;     // DSpark's window holds the side request's states
#endif
                } else if ((int64_t) ids.size() + std::max(0, max_new) > park_rows) {
                    dense.drop_park();   // this request writes past the saved rows
                    std::fprintf(stderr, "serve: park dropped (request reaches past %lld positions)\n", (long long) park_rows);
                }
            }
            int reused = 0;
            bool regen = false;   // the snapshot's own prompt again: nothing new to feed
            if (reuse_override >= 0) { reused = reuse_override; regen = pool_regen; }   // a partial unpark / a pool record
            else if (!unparked && !hist.empty() && lcp == hist.size() && lcp < ids.size()) {
                reused = (int) lcp;
                if (live_dirty && (reused & 1)) {   // re-feed the last accepted token (its group's ratio-2 state)
                    --reused;
                    hist.resize((size_t) reused);
                    std::fprintf(stderr, "serve: the live state is dirty (DSpark) at an odd position: resuming at %d\n", reused);
                }
            }
            else if (!unparked && cont_last) {
                reused = (int) lcp - 1;
                hist.resize((size_t) reused);
            }
            else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len == ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);   // the same prompt again (regenerate): its end state + saved logits
                reused = snap_len;
                last_logits = snap_logits;
                regen = true;
            } else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len < ids.size() && dense.restore()) {
                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else if (rewind1 && !unparked && snap_len > 1 && lcp + 1 == (size_t) snap_len && lcp < ids.size() && dense.restore()) {
                // STRATA_SERVE_REWIND1=1 (opt-in): the request matches the snapshot's prompt except its LAST token -
                // DeepSeek's template, in thinking mode, re-renders the previous turn's "<think>" generation prompt as
                // "</think>" once a new user message follows (drop_thinking), so every new user message re-read the whole
                // conversation.  Restore the snapshot and re-feed from that token: one token back is exact - after
                // restore() the window ring holds [P - RAW, P) and the ratio-2 compressor state P - 2, P - 1, so the
                // re-fed token's window and group partner are intact; KV / compressed / indexer rows and Engram's history
                // are positional and are overwritten.  Two tokens back would not be (see the partial unpark above).
                hist.resize(lcp);
                if (live_id.size() > lcp) live_id.resize(lcp);
                reused = (int) lcp;
                std::fprintf(stderr, "serve: rewound the snapshot one token (%d -> %zu)\n", snap_len, lcp);
                snap_len = -1;
            } else if (pool_on) {
                // an unrelated request: the conversation it overwrites goes to the pool
                save_live("unrelated");
                hist.clear();
                live_id.clear();
                snap_len = -1;
            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK)
                int64_t rows = (int64_t) ids.size() + (max_new > 0 ? max_new : park_max) + 64;
                if (rows < park_floor) rows = std::min(park_floor, park_max);   // the window for the NEXT side call
                if (conv_on && snap_len >= park_min && hist.size() >= (size_t) snap_len && rows <= park_max)
                    conv_cache.park_live(pipe, hist, snap_len, rows, snap_logits, err);
                else if (park_on && !dense.parked() && snap_len >= park_min && hist.size() >= (size_t) snap_len &&
                         rows <= park_max && dense.park(rows)) {
                    park_hist.assign(hist.begin(), hist.begin() + snap_len);
                    park_logits = snap_logits;
                    park_rows = rows;
                    std::fprintf(stderr, "serve: parked the %d-token conversation (%lld positions saved)\n", snap_len, (long long) rows);
                }
                hist.clear();
                live_id.clear();
                snap_len = -1;
            }
            live_dirty = false;
            if (pool_on) {   // a save queued above reads positions this request is about to overwrite
                std::string ferr2;
                if (!pool.fence(&ferr2)) std::fprintf(stderr, "strata serve: conversation cache: fence: %s\n", ferr2.c_str());
            }
            if (pool_on) std::fprintf(stderr, "serve: reuse lcp=%zu pool=%zu lp=%zu hist=%zu snap=%d prompt=%zu images=%zu -> reused=%d\n",
                                      lcp, pool.size(), lp_show, hist.size(), snap_len, ids.size(), vis_pos.size(), reused);
            else
            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d\n", lcp,
                         park_hist.size(), lp_show, hist.size(), snap_len, ids.size(), reused);
            if (reused > 0) {
                // the Engram hashes of the first new positions read the 3 tokens before `reused` from the history, which
                // is kept by position: an unrelated request since (a parked conversation's side request) wrote its own
                // tokens there.  The reused prefix is this prompt's own, so write it back (a no-op when unchanged).
                const int h0 = std::max(0, reused - 3);
                uint8_t dead[3] = { 0, 0, 0 };
                for (int p = h0; p < reused; ++p) dead[p - h0] = (size_t) p < rid.size() && rid[(size_t) p] != 0;
                dense.set_engram_history(h0, reused - h0, ids.data() + h0, dead);
                std::printf("RESUME %d\n", reused);
                std::fflush(stdout);
            }
            const int total = (int) ids.size();
            // a long prompt runs layer-major without the VRAM cache (its pass / chunk buffers need that VRAM) and with
            // the prompt's routing (no decode bias).  A short one (<= step_max new tokens: a chat / agent turn's tool
            // result) runs on the decode side instead - small dense passes, the cache and the decode biases: the
            // layer-major path streams nearly every expert of every layer whatever the batch size (~18 s for 40 tokens)
            const int delta = total - reused;
            int sp_tail = 0;   // DSpark: the last new prompt tokens, fed by pass_v (seeds the window)
#if defined(DS41_HAVE_DSPARK)
            if (dsp_path && admit < 0) {   // (an admission decodes on in a slot: no drafts there)
                if (!(reused > 0 && reused == dsp_next) && dsp) dsp->reset();
                dsp_next = -1;
                sp_tail = std::min(128, delta);
            }
#endif
            int dh = delta - sp_tail;   // the part the passes / batch feed
            bool by_step = defer_vram && dh >= 1 && dh <= step_max;
            // STRATA_SERVE_BATCH_TAIL (default on, =0 the old way): a turn that runs as a layer-major batch anyway takes
            // its DSpark tail INTO the batch and the seed rows are read off the batch's state (bcap_vh) - the tail as
            // 16-token exact passes was half of every turn (s11 2026-10-15: +273 tokens 33.5 s = batch 17.1 + tail 15.8).
            // DSpark drafts are verified, so only the seed's numerics (batch vs pass kernels) change, never the output.
            static const bool batch_tail = !(std::getenv("STRATA_SERVE_BATCH_TAIL") && std::getenv("STRATA_SERVE_BATCH_TAIL")[0] == '0');
            int bcap = 0;
            if (batch_tail && sp_tail > 0 && !by_step && dh >= 2 && PB > 1) { bcap = sp_tail; dh = delta; }
            // STRATA_SERVE_BATCH_MIN (tokens; default 128, 0 = off): a turn of at least this many new tokens runs as a
            // layer-major batch even when the passes would take it (every turn up to 128 + STEP_MAX = 224 tokens did:
            // the DSpark tail + the step path).  The batch's floor fell with the routed-only reads.  s11, 512K, prompt s
            // (+ 32-token decode) passes / batch: +30 5.0 (1.7) / 6.0 (2.0), +60 6.6 (1.7) / 7.0 (2.0), +90 9.7 (1.0) /
            // 7.7 (1.7), +150 17.6 (1.4) / 9.8 (1.6).  Decode right after a batch starts colder (the VRAM cache is
            // re-seeded), hence the margin above the ~80-token crossover.
            static const int batch_min = std::getenv("STRATA_SERVE_BATCH_MIN") ? std::atoi(std::getenv("STRATA_SERVE_BATCH_MIN")) : 128;
            bool force_batch = false;
            if (batch_min > 0 && batch_tail && sp_tail > 0 && bcap == 0 && delta >= batch_min && PB > 1 && delta >= 2) {
                bcap = sp_tail; dh = delta; force_batch = true; by_step = false;
            }
            (void) force_batch;
            // batch slots: a short BGEN prompt is read in the windows, its rows beside the decoding slots' (feed_mixed);
            // an image prompt, or a longer one, alone (the slots wait)
            const bool mix_read = NS > 0 && admit >= 0 && !is_vision && dh >= 1 && dh <= mix_max && any_active();
            if (mix_read && defer_vram) by_step = true;
            if (by_step && !cache_open) {
                if (!open_cache()) { std::printf("ERR VRAM seed failed: %s\n", err.c_str()); std::fflush(stdout); continue; }
                cache_open = true;
            }
            // exact passes (STRATA_SERVE_PASS_BIAS=0): the decode-side prompt tokens route with the model's own
            // selection (no route bias / file penalty) - only the batch shape differs from the exact batch path
            const bool exact_pass = by_step && !pass_bias;
            if (exact_pass && bias_set) {
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            // STRATA_SERVE_KEEP_CACHE=1 (s6, opt-in until measured): a batch turn of at most STRATA_SERVE_KEEP_MAX new
            // tokens (default 4096) gives back only the VRAM its batch needs (the calibrated need(B) below) - the cache's
            // lowest-ranked tail - instead of the whole cache, so the hot experts serve the batch from VRAM (no PCIe) and
            // the decode after it skips the re-seed (regrow_cache refills just the tail).  The prompt still routes exactly.
            // s6 A/B: it never engages with the default --vram-lru (the LRU reorders slots, so "the tail" is not the
            // coldest experts and a batch-time admission could duplicate a saved one) - it needs --no-vram-lru for now.
            bool kept = false;
            if (Ds41MoeTier::keep_cache_mode() && !by_step && dh >= 2 && !(defer_vram && cache_open))
                std::fprintf(stderr, "serve: keep-cache: not applicable (deferred VRAM %d, cache open %d)\n", (int) defer_vram,
                             (int) cache_open);
            if (!by_step && defer_vram && cache_open && dh >= 2) {
                static const int keep_max = std::getenv("STRATA_SERVE_KEEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_KEEP_MAX")) : 4096;
                if (Ds41MoeTier::keep_cache_mode() && dh <= keep_max && !mc.cpu_only) {
                    const double need = (4.3 * 1073741824.0 + 0.95 * 1048576.0 * (double) std::min(dh, PB)) +
                                        a.vram_margin_gib * 1073741824.0;
                    std::string kerr;
                    int64_t g = 0; bool shrink_ok = true;
                    for (int st = 0; st < pipe.n_stages(); ++st) {
                        std::string sk;
                        const int64_t gs = pipe.tier(st).shrink_cache_for((int64_t) need, sk);
                        if (gs < 0) { shrink_ok = false; kerr = sk.empty() ? kerr : sk; g = -1; break; }
                        g += gs;
                    }
                    if (shrink_ok && g >= 0) {
                        kept = true;
                        int64_t stay = 0; for (int st = 0; st < pipe.n_stages(); ++st) stay += pipe.tier(st).resident();
                        std::fprintf(stderr, "serve: cache kept for a %d-token batch (%lld tail slots out, %lld experts stay)\n",
                                     dh, (long long) g, (long long) stay);
                    } else if (!kerr.empty()) {
                        std::fprintf(stderr, "serve: keep-cache: %s - releasing the cache\n", kerr.c_str());
                    }
                }
                if (!kept) {
                    for (int st = 0; st < pipe.n_stages(); ++st) pipe.tier(st).release_cache();
                    cache_open = false;
#if defined(DS41_HAVE_DSPARK)
                    // STRATA_SERVE_DSPARK_EVICT=N (s6, opt-in until measured; 0 = off): a batch of >= N new tokens also
                    // frees DSpark's ~2.9 GB of weights for its duration - the prompt batch is sized from free VRAM
                    // (s6: 8731 columns before DSpark loads, 5434 after), so a long prompt takes fewer expert sweeps.
                    // open_cache() re-creates it for the tail (~1-4 s); the window is re-seeded by the tail as always.
                    static const int dsp_evict = std::getenv("STRATA_SERVE_DSPARK_EVICT") ? std::atoi(std::getenv("STRATA_SERVE_DSPARK_EVICT")) : 0;
                    if (dsp && dsp_evict > 0 && dh >= dsp_evict) {
                        dsp.reset();
                        std::fprintf(stderr, "serve: DSpark weights freed for a %d-token batch\n", dh);
                    }
#endif
                }
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            tier.rebalance_begin_prompt();
            tier.reset_stats();
            if (const char* rl = std::getenv("STRATA_ROUTE_LOG")) {   // a turn marker for the route log (see ds41_moe.cpp)
                if (FILE* f = std::fopen(rl, "a")) { std::fprintf(f, "# turn %d new %d %s\n", n_req, dh, by_step ? "passes" : "batch"); std::fclose(f); }
            }
            const double tp0 = now_ms();
            auto pp = [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            };
            prefill_cancelled = false;
            auto feed_steps = [&]() -> bool {   // the short-turn path: passes of up to pass_max tokens (a lone token: a step)
                int last_rows = 1;
                for (int i = 0; i < dh; ) {
                    if (stop_prefill()) return false;
                    const int m = std::min(pass_max, dh - i);
                    if (m == 1) {
                        const double ts0 = now_ms();
                        if (!step(ids[(size_t) (reused + i)], reused + i)) return false;
                        fd_dense += now_ms() - ts0;
                    } else if (!pass_m(ids.data() + reused + i, m, reused + i)) return false;
                    if (!capture_tail(delta, i, m, 0)) return false;   // this pass's rows, before the next pass replaces them
                    const int before = i;
                    i += m;
                    last_rows = m;
                    if (i / 64 != before / 64 && i < dh) pp(i);
                }
                if (!head_logits(last_rows - 1, 1, last_logits.data())) return false;
                pp(dh);
                return true;
            };
            static const int dump_tail = std::getenv("STRATA_SERVE_DUMP_TAIL") ? std::atoi(std::getenv("STRATA_SERVE_DUMP_TAIL")) : 0;
            tail_keep = dump_tail > 0 ? std::min(dump_tail, delta) : 0;
            tail_logits.clear();
            // VRAM-aware batch width for this turn (feed() halves it again if even this does not fit): the batch's
            // state columns + graph are ~0.48 MiB a column, and the card is shared with whatever else the desktop
            // runs.  This is what lets a big --prompt-batch stay OPT-IN: the engine takes what is actually free.
            pb_cap = PB;
            if (!by_step && dh >= 2 && PB > 1 && !mc.cpu_only) {
                size_t frb = std::numeric_limits<size_t>::max(), totb = 0;
                for (int st = 0; st < pipe.n_stages(); ++st) {
#ifdef STRATA_ENABLE_CUDA
                    cudaSetDevice(pipe.stages[(size_t) st].cuda_device);
#endif
                    size_t fr = 0, tot = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(pipe.stages[(size_t) st].backend), &fr, &tot);
                    frb = std::min(frb, fr);
                    totb = std::max(totb, tot);
                }
                if (frb == std::numeric_limits<size_t>::max()) frb = 0;
                // The layer-major batch's own VRAM, CALIBRATED on four traced points (STRATA_VRAM_TRACE=1, s13
                // run9/run13/run14, a 4 GiB foreign process): free 9.28 GiB with B=8192 failed, free 9.26 with
                // B=5210 passed, free 5.92 with B=2411 failed, a free card with B=8192 passed - all four fit
                //     need(B) ~= 4.3 GiB + 0.95 MiB * B
                // 4.3 GiB fixed = the tier's two 1 GiB stage halves + a ~2.19 GiB graph buffer that is the SAME
                // SIZE at B=2411 and B=8192 (so it is not the piece buffers; identity unconfirmed - see the
                // residuals in V41_START_HERE).  0.95 MiB a column = columns 0.32 + graph 0.17 + part buffers 0.41,
                // measured directly.  Per-tensor arithmetic undercounted the tier 2x and let a 2.19 GiB allocation
                // fail while cudaMemGetInfo still said 6.83 GiB free.  A free card (>= 13 GiB) still gets its full PB.
                // run20 (2026-10-16) fixes the GRAPH term: the pass-graph buffer is B-INDEPENDENT - the B=16384 leg
                // OOM'd on "allocating 1424.31 MiB", the very size this comment records for 8192, so "graph 0.17 MiB
                // a column" is a mis-derivation (1424/8192) and the decomposition is really ~5.7 GiB fixed + 0.78 a
                // column.  Same line, same fit - but it matters: a wide batch's per-column STATE is what starves the
                // fixed graph buffer (16384 = 5.14 GiB of state, 1.03 GiB short), not the graph itself.
                // 2026-10-20: the GRAPH also grows with the CONTEXT.  The indexer's score / mask tensors are
                // [visible compressed rows x queries] (build_attn), and the visible rows are a power of two: a
                // ratio-1 layer past 2^19 rows builds a 9.06 GiB span graph untiled (5.56 with build_attn's indexer
                // tile) instead of ~1.2, and it is built first by
                // the prompt's LAST batch (under enc-skip the only one that runs layers 21-39) - long after this width
                // took the columns.  The line above budgets the graph as its own decomposition says, 2.19 GiB + 0.17
                // MiB a column (4.2 GiB at 12288 columns): room for some growth over the 1.16 GiB a short context
                // builds, none for 9 GiB.  So the width must also fit the batch's memory WITHOUT that allowance plus
                // the graph this prompt's last sub-batch will really build, as the allocator plans it:
                //     need(B) = max(4.3 GiB + 0.95 MiB * B,  2.5 GiB + 0.78 MiB * B + graph)
                // The second term is a 5090 trace (2026-10-20, 12288 columns, 18.74 GiB free at the cap): a 512K
                // prompt's batches took 13.01 GiB with a 1.16 GiB graph - 11.85 GiB besides it = 2.5 GiB + run20's
                // 0.78 MiB a column - and its last batch's 4.56 GiB graph left 2.33 GiB free, as the term says.  Up to
                // the allowance (here every context up to 512K at the full width) the cap is the one it was.
                // STRATA_PB_GRAPH=0: the line alone (diagnostics).
                static const bool pb_graph = !(std::getenv("STRATA_PB_GRAPH") && std::atoi(std::getenv("STRATA_PB_GRAPH")) == 0);
                const double g_need = pb_graph ? (double) dense.span_graph_bytes(total, SB) : 0.0;   // memoized per row count
                const double room = (double) frb - a.vram_margin_gib * 1073741824.0 - 4.3 * 1073741824.0;
                const double room_g = (double) frb - a.vram_margin_gib * 1073741824.0 - 2.5 * 1073741824.0 - g_need;
                const int fit_line = (int) std::max<int64_t>(2, (int64_t) (room / (0.95 * 1048576.0)));
                const int fit_g = pb_graph ? (int) std::max<int64_t>(2, (int64_t) (room_g / (0.78 * 1048576.0))) : fit_line;
                const int fit = std::min(fit_line, fit_g);
                pb_cap = std::min(PB, std::max(SB, fit));
                if (vram_trace) std::fprintf(stderr, "vram trace: cap free %.2f GiB room %.2f GiB fit %d, graph %.2f GiB fit %d -> pb_cap %d\n",
                                             frb / 1073741824.0, room / 1073741824.0, fit_line, g_need / 1073741824.0, fit_g, pb_cap);
                if (fit_g < fit_line && fit_g < PB)
                    std::fprintf(stderr, "serve: a %d-token context's pass graph needs %.2f GiB -> prompt batch %d (of %d)\n",
                                 total, g_need / 1073741824.0, pb_cap, PB);
                else if (pb_cap < PB) std::fprintf(stderr, "serve: VRAM %.2f GiB free -> prompt batch %d (of %d)\n",
                                                   frb / 1073741824.0, pb_cap, PB);
            }
            const double tph0 = now_ms();
            const double fd0[3] = { fd_dense, fd_tier, fd_logits };   // this request's batch split (stderr)
            bcap_n = bcap; bcap_got = 0;
            if (bcap > 0) bcap_vh.assign((size_t) bcap * 3 * n_embd, 0.0f);
            auto feed_mixed = [&]() -> bool {   // the read's rows go into the windows, up to 16 - the decoding rows each
                Mix mx;
                mx.slot = gslot;
                mx.pos = reused;
                mx.left = dh;
                mx.ids = ids.data();
                mx.rid = rid.data();
                mx.nrid = rid.size();
                const RouteCls own{ a.route_bias, a.route_bias_file, cur_rprot, cur_rpool };
                mx.cls = own;
                if (exact_pass) mx.cls.rb = mx.cls.rbf = 0.0f;
                while (mx.left > 0) {
                    take_slot_lines(nullptr);   // BSTOPs now; a short read gives way to nobody (BYIELD not taken)
                    const int before = dh - mx.left;
                    if (!run_window(&mx)) { std::fflush(stdout); std::fflush(stderr); std::_Exit(1); }
                    if (!bind(gslot)) return false;
                    const int done = dh - mx.left;
                    if (done / 64 != before / 64 && mx.left > 0) pp(done);
                }
                pp(dh);
                set_class(own);
                return true;
            };
            // BYIELD during a long read (a solo one, or this admission's): at the next layer-major batch boundary the
            // part read stays in the named slot (the admission's own, or a free one the server reserved for a solo read)
            int yield_slot = -1;
            if (NS > 0 && !is_vision)
                feed_yield = [&](int c0, int B) -> bool {
                    int ys = -1;
                    take_slot_lines(&ys);
                    if (ys < 0) return false;
                    const int rest = dh - (c0 + B);
                    const bool can = ys < NS && !bs[(size_t) ys].active && (admit < 0 || ys == admit) && rest > std::max(SB, 16) &&
                                     (!enc_skip || B >= ER || c0 == 0);
                    if (!can) {
                        std::fprintf(stderr, "strata batch: BYIELD %d not taken at %d of %d new tokens\n", ys, c0, dh);
                        return false;
                    }
                    yield_slot = ys;
                    std::fprintf(stderr, "strata batch: the prompt read gives way after %d of %d new tokens; slot %d holds it\n",
                                 c0 + B, dh, ys);
                    return true;
                };
            dense.reset_span_width();   // a sub-batch narrowed by the last prompt's tight card starts wide again
            bool ok = total == reused || dh == 0 || (mix_read ? feed_mixed() : by_step ? feed_steps() : feed(ids.data() + reused, dh, reused, pp, nullptr));
            feed_yield = nullptr;
            bcap_n = 0;
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                for (int st = 0; st < pipe.n_stages(); ++st) pipe.dense(st).release_pass();
                bool regrow_hard_fail = false;
                for (int st = 0; st < pipe.n_stages(); ++st) {
                    std::string sk;
                    if (pipe.tier(st).regrow_cache(sk) < 0) {
                        std::fprintf(stderr, "serve: keep-cache regrow stage %d: %s\n", st, sk.c_str());
                        regrow_hard_fail = true;
                        kerr = sk;
                    }
                }
                if (regrow_hard_fail) {
                    std::fprintf(stderr, "serve: keep-cache regrow hard-fail - re-seeding\n");
                    for (int st = 0; st < pipe.n_stages(); ++st) pipe.tier(st).release_cache();
                    cache_open = false;
                } else {
                    bias_now = true;
                    apply_bias();
                }
            }
            if (ok && feed_stopped >= 0 && yield_slot >= 0 && !by_step && !mix_read) {   // BYIELD taken: the read goes on later
                const int qn = reused + feed_stopped;
                feed_stopped = -1;
                hist.assign(ids.begin(), ids.begin() + qn);
                live_id.assign(rid.begin(), rid.begin() + std::min<size_t>(rid.size(), (size_t) qn));
                snap_len = -1;
                live_dirty = false;
                std::printf("YIELDED %d %d\n", yield_slot, qn);
                std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d 0 0 0 0 0.0 %d\n", total, now_ms() - tp0, reused, qn - reused);
                if (admit >= 0) std::printf("BADM %d 0\n", admit);
                std::fflush(stdout);
                if (yield_slot != gslot) {   // a solo read: into the slot the server reserved for it
                    const int from = gslot;
                    slot_scope.f();
                    slot_scope.f = nullptr;
                    if (!swap_conv(yield_slot, from))
                        std::fprintf(stderr, "strata batch: swap %d <-> %d failed: %s\n", yield_slot, from, ferr.c_str());
                }
                ++batch_epoch;
                continue;
            }
            feed_stopped = -1;
            double ph_feed = now_ms() - tph0, ph_open = 0, ph_tail = 0;   // the request's prompt phases (stderr)
            int sp_t = -1;   // DSpark: the first decode token (sampled from the last prompt row), its drafts in sp_ids
            int sp_ids[5] = {0}, sp_tmp[5];
            float sp_conf[5];
#if defined(DS41_HAVE_DSPARK)
            // the cache opens before dsp is tested: open_cache() re-creates DSpark when STRATA_SERVE_DSPARK_EVICT freed it
            // for this prompt's batch (otherwise the tail is skipped and the turn decodes without drafts)
            if (ok && sp_tail > 0 && defer_vram && !cache_open) {
                const double to0 = now_ms();
                if (!open_cache()) { ferr = "VRAM seed failed: " + err; ok = false; }
                else cache_open = true;
                ph_open = now_ms() - to0;
            }
            if (ok && dsp && sp_tail > 0) {   // the tail through pass_v (chunks of <= 16); routing as the passes
                bias_now = pass_bias;
                apply_bias();
                const double tt0 = now_ms();
                std::vector<int> spos;
                if (bcap > 0) {   // the seed rows came off the batch (bcap_vh, its last bcap_got tokens): DSpark steps only
                    const int c0s = total - bcap_got;
                    for (int i0 = c0s; ok && i0 < total; i0 += 16) {
                        if (stop_prefill()) { ok = false; break; }
                        const int m = std::min(16, total - i0);
                        spos.resize((size_t) m);
                        for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                        const bool last = i0 + m == total;
                        if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);   // the batch's last row
                        if (i0 == c0s) cur_seed_n = 0;
                        cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                        std::copy_n(bcap_vh.data() + (size_t) (i0 - c0s) * 3 * n_embd, (size_t) m * 3 * n_embd,
                                    cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                        cur_seed_n += m;
                        if (!dsp->step(bcap_vh.data() + (size_t) (i0 - c0s) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                       last ? sp_ids : sp_tmp, sp_conf)) {
                            ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                        }
                    }
                    if (bcap_got < 1) { ferr = "DSpark batch seed captured nothing"; ok = false; }
                }
                for (int i0 = total - sp_tail; ok && bcap == 0 && i0 < total; i0 += 16) {
                        if (stop_prefill()) { ok = false; break; }
                    const int m = std::min(16, total - i0);
                    if (!pass_v(ids.data() + i0, m, i0)) { ok = false; break; }
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) {
                        std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                        sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    }
                    if (i0 == total - sp_tail) cur_seed_n = 0;
                    cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                    std::copy_n(vh.data(), (size_t) m * 3 * n_embd, cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                    cur_seed_n += m;
                    if (!dsp->step(vh.data(), spos.data(), m, last ? sp_t : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                    }
                }
                if (ok) { bias_now = true; apply_bias(); }
                ph_tail = now_ms() - tt0;
                pp(delta);
            } else if (ok && regen && dsp && snap_seed_n > 0 && admit < 0) {   // a repeated prompt: re-seed DSpark from the snapshot's rows
                const double tt0 = now_ms();
                dsp->reset();
                std::vector<int> spos;
                const int s0 = total - snap_seed_n;
                for (int i0 = s0; i0 < total; i0 += 16) {
                    const int m = std::min(16, total - i0);
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    if (!dsp->step(snap_seed.data() + (size_t) (i0 - s0) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                   last ? sp_ids : sp_tmp, sp_conf)) {
                        std::fprintf(stderr, "serve: DSpark re-seed: %s (decoding without drafts)\n", dsp->last_error().c_str());
                        dsp->reset();
                        sp_t = -1;
                        break;
                    }
                }
                ph_tail = now_ms() - tt0;
                std::fprintf(stderr, "serve: repeated prompt - DSpark re-seeded from %d saved rows (%.2f s)\n", snap_seed_n, ph_tail / 1000.0);
            }
#endif
            dense.prof_report("turn");
            std::fprintf(stderr, "serve: prompt phases: %d new = %d %s (%.2f s: dense %.2f, experts %.2f, logits %.2f) + DSpark tail %d%s "
                         "(cache open %.2f s, seed %.2f s)\n", delta, dh, by_step ? "passes" : "batch", ph_feed / 1000.0,
                         (fd_dense - fd0[0]) / 1000.0, (fd_tier - fd0[1]) / 1000.0, (fd_logits - fd0[2]) / 1000.0,
                         bcap > 0 ? bcap_got : sp_tail, bcap > 0 ? " from the batch" : " as passes", ph_open / 1000.0, ph_tail / 1000.0);
            if (tail_keep > 0 && ok) {   // the quality scorer's rows: req%03d.tail.f32, [tail_keep x n_vocab], last prompt row last
                if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {
                    char nb[40];
                    std::snprintf(nb, sizeof(nb), "/req%03d.tail.f32", n_req);
                    std::ofstream f(std::string(dd) + nb, std::ios::binary);
                    f.write((const char *) tail_logits.data(), (std::streamsize) (tail_logits.size() * 4));
                }
            }
            tail_keep = 0;
            if (!ok) {
                if (prefill_cancelled) {
                    // No partial prompt can be reused as a valid conversation.
                    snap_seed_n = cur_seed_n = 0;
                    // feed returned before its ordinary elastic-column/chunk cleanup.
                    dense.release_pass();
                    const bool cleaned = !dc.elastic_cols || dense.set_columns(0);
                    if (rdev1) rdev1 = dense.routed_dev();
                    tier.release_chunk();
                    tier.rebalance_tick(true);
                    if (cleaned) std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d 0 0 0 0 0 0\n",
                                             total, now_ms() - tp0, reused);
                    else std::printf("ERR cancelled prefill cleanup: %s\n", dense.last_error().c_str());
                } else std::printf("ERR prefill failed: %s\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();
                live_id.clear();
                snap_len = -1;
                continue;
            }
            hist = ids;
            live_id = rid;
            // the single park: no snapshot for an image prompt (a text request with the same ids would restore its
            // embeddings); the pool compares identities, so an image prompt is snapshotted like any other
            if (keep_snap && !unparked) {
                if (btrace) std::fprintf(stderr, "strata batch: the turn's snapshot stays at %d (this request re-feeds its own output)\n", snap_len);
            } else if (total > reused && (!is_vision || pool_on) && dense.snapshot(total)) {
                snap_len = total;
                snap_logits = last_logits;
                snap_seed.swap(cur_seed);   // this prompt's DSpark tail rows (0 rows when DSpark is off / no tail ran)
                snap_seed_n = cur_seed_n;
            }
            cur_seed_n = 0;
            if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {   // exactness checks: request N's last-row logits
                char nb[32];
                std::snprintf(nb, sizeof(nb), "/req%03d.f32", n_req);
                std::ofstream f(std::string(dd) + nb, std::ios::binary);
                f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
            }
            ++n_req;
            const double prompt_ms = now_ms() - tp0;
            if (exact_pass && cache_open) {   // decode: the biases back on, from the VRAM set as it is now
                bias_now = true;
                apply_bias();
            }
            const Ds41MoeStats pst = tier.stats();
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused, %s) in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / "
                                 "%lld pcie / %lld file)\n", total, reused, by_step ? (exact_pass ? "exact passes" : "passes") : "batch", prompt_ms / 1000.0,
                         1000.0 * (total - reused) / std::max(prompt_ms, 1e-9), (long long) pst.hits, (long long) pst.cpu,
                         (long long) pst.pcie, (long long) pst.file_tier);
            if (defer_vram && !cache_open) {   // decode next: the VRAM cache opens into what the prompt freed
                if (!open_cache()) {
                    std::printf("ERR VRAM seed failed: %s\n", err.c_str());
                    std::fflush(stdout);
                    continue;
                }
                cache_open = true;
                { int64_t resid = 0; for (int st = 0; st < pipe.n_stages(); ++st) resid += pipe.tier(st).resident();
                std::fprintf(stderr, "serve: VRAM %.2f GiB free -> %lld experts in VRAM (seed %.2f s)\n", seed_free,
                             (long long) resid, seed_s); }
            }
            if (ok) tier.rebalance_plan();
            if (bias_changed) {   // decode: this request's route bias + file penalty (see bias_changed)
                bias_now = true;
                apply_bias();
            }
            tier.reset_stats();
            dense.route_meter(true);
            const double td0 = now_ms();
            int gen = 0, pos = total;
            const char * finish = "length";
            std::vector<float> lg = last_logits;
#if defined(DS41_HAVE_DSPARK)
            // ---- DSpark speculative sampling (as the one-shot loop: P = last fed position, t = the token for P+1,
            // sp_ids = drafts for P+2..; each verify row's distribution is sample_p's, spec_accept keeps it exact)
            long rq_pass = 0, rq_acc = 0;
            // serve pass split (ms summed over this request's passes): verify = pass_v (its own split from pv_ms),
            // sample = sample_dist/spec_accept/sample_p over the verify rows, draft = DSpark step; the rest of the
            // decode wall (emit, lookup, rebalance_tick) is reported as "other"
            double sq_ms[3] = {0, 0, 0};
            const double pv0[4] = {pv_ms[0], pv_ms[1], pv_ms[2], pv_ms[3]};
            if (sp_t >= 0) {
                int P = total - 1, t = sp_t, prev = ids.back();
                bool dirty = false;
                std::vector<int> vt;
                std::vector<std::pair<int, double>> dist;
                // STRATA_SERVE_LOOKUP=L (s6, opt-in until measured; mab776/Strata-DS4 M11's design): when the context's
                // last G+1 tokens (STRATA_LOOKUP_NGRAM=G, default 3) occurred earlier in this request (prompt + output),
                // the tokens that followed them there are the verify pass's drafts instead of DSpark's - an agent's output
                // copies its context (edits, rewrites, quoted tool results).  spec_accept treats every draft as a
                // deterministic token (accept with p, else resample without it), so the output distribution is unchanged.
                // A match of M tokens drafts M - G + 1 (<= L, <= the pass width).  STRATA_LOOKUP_AGREE (default 1): only
                // when the copy's first token is DSpark's own first draft.  A lookup pass with less than half accepted
                // sits the lookup out 1, 2, 4 .. 64 passes.  DSpark steps on the verified rows either way.
                // kVerify (s6): rows one verify pass may take - 16, as the prompt passes (pass_v splits the tier calls at
                // kMaxMulti - 8 then, 16 now - and the pass buffers hold 17).  A copy is accepted ~96% on edits, so the
                // draft length was the limit at the old kMaxMulti - 2 = 6 (s6: 932/972 accepted in 169 passes, only +3%
                // over DSpark's own).
                constexpr int kVerify = 16;
                static const int lk_max = std::getenv("STRATA_SERVE_LOOKUP") ? std::max(0, std::min(kVerify - 2, std::atoi(std::getenv("STRATA_SERVE_LOOKUP")))) : 0;
                static const int lk_g = std::getenv("STRATA_LOOKUP_NGRAM") ? std::max(1, std::atoi(std::getenv("STRATA_LOOKUP_NGRAM"))) : 3;
                static const bool lk_agree = !(std::getenv("STRATA_LOOKUP_AGREE") && std::getenv("STRATA_LOOKUP_AGREE")[0] == '0');
                int lk_cool = 0, lk_back = 1;
                long lk_pass = 0, lk_dr = 0, lk_acc = 0;
                int dr[kVerify] = {0};
                auto emit = [&](int tok) -> bool {   // false = stop here
                    std::printf("T %d\n", tok);
                    std::fflush(stdout);
                    ++gen;
                    if (is_stop(tok)) { finish = "stop"; return false; }
                    if (max_new > 0 && gen >= max_new) return false;
                    return true;
                };
                while (emit(t)) {
                    if (stop_pending()) { finish = "cancel"; break; }
                    int k = sp_depth_serve;
                    if (max_new > 0) k = std::min(k, max_new - gen);
                    k = std::min(k, ctxv - (P + 3));   // the pass writes P+1 .. P+1+k
                    if (k < 0) break;
                    // STRATA_DSPARK_CONF=<tau>, as the one-shot loop: verify only the leading drafts whose confidence-head
                    // probability is >= tau (the emitted tokens are still the trunk's - only the verify width changes)
                    static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
                    if (conf_tau_s >= 0) {
                        int kk = 0;
                        while (kk < k && 1.0f / (1.0f + std::exp(-sp_conf[kk])) >= conf_tau_s) ++kk;
                        k = kk;
                    }
                    const bool refeed = dirty && ((P + 1) & 1);
                    for (int i = 0; i < k; ++i) dr[i] = sp_ids[i];
                    bool is_lk = false;
                    if (lk_max > 0 && lk_cool > 0) --lk_cool;
                    else if (lk_max > 0) {
                        // the sequence is hist + t (t = the token for P+1, not in hist yet); find the latest earlier q with
                        // seq[q-G .. q] == seq[n-G .. n] (G+1 tokens, the last being t), then copy seq[q+1 ..]
                        const int n = (int) hist.size();
                        auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
                        int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                        if (max_new > 0) cap = std::min(cap, max_new - gen);
                        cap = std::min(cap, ctxv - (P + 3));
                        if (cap >= 2 && n > lk_g + 1) {
                            for (int q = n - 1; q >= lk_g; --q) {
                                if (hist[(size_t) q] != t) continue;
                                int m = 1;
                                while (m < 64 && q - m >= 0 && hist[(size_t) (q - m)] == sq(n - m)) ++m;
                                if (m < lk_g + 1) continue;
                                const int L = std::min({cap, m - lk_g + 1, n - q});
                                if (L < 2) break;   // the latest match copies too little: not worth a pass
                                if (lk_agree && sq(q + 1) != sp_ids[0]) break;
                                for (int i = 0; i < L; ++i) dr[i] = sq(q + 1 + i);
                                k = L;
                                is_lk = true;
                                break;
                            }
                        }
                    }
                    vt.clear();
                    if (refeed) vt.push_back(prev);
                    vt.push_back(t);
                    for (int i = 0; i < k; ++i) vt.push_back(dr[i]);
                    const int r0 = refeed ? 1 : 0, p0 = refeed ? P : P + 1;
                    const double tq0 = now_ms();
                    if (!pass_v(vt.data(), (int) vt.size(), p0)) { finish = "error"; break; }
                    const double tq1 = now_ms();
                    int acc = 0, tn = -1;
                    for (; acc < k; ++acc) {
                        sample_dist(vlog.data() + (size_t) (r0 + acc) * nv, nv, temp, tk, top_p, min_p, dist);
                        if (!spec_accept(dist, dr[acc], rng, tn)) break;
                    }
                    if (acc == k) tn = sample_p(vlog.data() + (size_t) (r0 + k) * nv, nv, temp, tk, top_p, min_p, rng);
                    const double tq2 = now_ms();
                    ++rq_pass; rq_acc += acc;
                    if (is_lk) {
                        ++lk_pass; lk_dr += k; lk_acc += acc;
                        if (2 * acc < k) { lk_cool = lk_back; lk_back = std::min(64, lk_back * 2); }
                        else lk_back = 1;
                    }
                    int spos[kVerify + 1];
                    for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
                    int drafts[kVerify];
                    for (int i = 0; i < k; ++i) drafts[i] = dr[i];
                    if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, sp_ids, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); finish = "error"; break;
                    }
                    sq_ms[0] += tq1 - tq0; sq_ms[1] += tq2 - tq1; sq_ms[2] += now_ms() - tq2;
                    hist.push_back(t);
                    bool more = true;
                    for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); more = emit(drafts[i]); }
                    prev = acc == 0 ? t : drafts[acc - 1];
                    dirty = acc < k;
                    P += 1 + acc;
                    t = tn;
                    tier.rebalance_tick();
                    if (!more) break;
                }
                if (lk_pass > 0)
                    std::fprintf(stderr, "serve: lookup %ld/%ld drafts accepted in %ld passes\n", lk_acc, lk_dr, lk_pass);
                dsp_next = P + 1;   // the ring holds every position <= P (the last emitted token is not fed)
                if (std::strcmp(finish, "error") == 0) dsp_next = -1;
                live_dirty = dirty;   // the last pass rejected drafts: the ratio-2 state ring holds rejected positions
                sv_pass += rq_pass; sv_acc += rq_acc;
            }
#endif
            int last_tok = -1;   // (an admission's first token: its slot feeds it first)
            while (sp_t < 0 && (gen < max_new || max_new <= 0)) {
                if (stop_pending()) { finish = "cancel"; break; }
                const int tok = sample_p(lg.data(), nv, temp, tk, top_p, min_p, rng);
                last_tok = tok;
                std::printf("T %d\n", tok);
                std::fflush(stdout);
                ++gen;
                if (is_stop(tok)) { finish = "stop"; break; }
                if ((max_new > 0 && gen >= max_new) || pos + 1 >= ctxv) break;
                if (!step(tok, pos) || !head_logits(0, 1, lg.data())) { finish = "error"; break; }
                hist.push_back(tok);
                ++pos;
                tier.rebalance_tick();
            }
            const double dec_ms = now_ms() - td0;
            const Ds41MoeStats st = tier.stats();
            if (std::strcmp(finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                hist.clear();
                live_id.clear();
                snap_len = -1;
                finish = "length";
            }
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file",
                         gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), finish,
                         (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier);
            if (st.remote > 0) std::fprintf(stderr, ", %lld remote", (long long) st.remote);   // STRATA_DS41_REMOTE
            std::fprintf(stderr, "\n");
            if (vram_trace) {   // who holds the card while decoding (the expert slots are the rest)
                size_t tf = 0, tt = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                double dsp_gib = 0.0;
#if defined(DS41_HAVE_DSPARK)
                if (dsp) dsp_gib = (double) dsp->weight_bytes() / 1073741824.0;
#endif
                std::fprintf(stderr, "vram ledger: %s; DSpark weights %.2f GiB; free %.2f of %.2f GiB\n",
                             dense.vram_ledger().c_str(), dsp_gib, tf / 1073741824.0, tt / 1073741824.0);
            }
            if (const Ds41Dense::RouteMeter rm = dense.route_meter(false); rm.on && rm.slots > 0)
                std::fprintf(stderr, "serve: routing vs exact: %.2f%% of expert picks swapped, %.2f%% of router weight kept (%lld slots)\n",
                             100.0 * (double) rm.swaps / (double) rm.slots, 100.0 * rm.weight_kept, rm.slots);
            if (vram_trace) {   // who holds the card while decoding (the expert slots are the rest)
                size_t tf = 0, tt = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                double dsp_gib = 0.0;
#if defined(DS41_HAVE_DSPARK)
                if (dsp) dsp_gib = (double) dsp->weight_bytes() / 1073741824.0;
#endif
                std::fprintf(stderr, "vram ledger: %s; DSpark weights %.2f GiB; free %.2f of %.2f GiB\n",
                             dense.vram_ledger().c_str(), dsp_gib, tf / 1073741824.0, tt / 1073741824.0);
            }
#if defined(DS41_HAVE_DSPARK)
            if (rq_pass > 0)
                std::fprintf(stderr, "serve: DSpark %ld passes, %.2f tokens/pass (all requests %.2f)\n", rq_pass,
                             1.0 + (double) rq_acc / rq_pass, 1.0 + (double) sv_acc / std::max(1L, sv_pass));
            if (rq_pass > 0) {
                const double n = (double) rq_pass;
                std::fprintf(stderr, "serve: pass split %.2f ms/pass = verify %.2f (dense %.2f, tier %.2f, taps %.2f, logits %.2f) "
                             "+ sample %.2f + draft %.2f + other %.2f\n", dec_ms / n, sq_ms[0] / n, (pv_ms[0] - pv0[0]) / n,
                             (pv_ms[1] - pv0[1]) / n, (pv_ms[2] - pv0[2]) / n, (pv_ms[3] - pv0[3]) / n, sq_ms[1] / n,
                             sq_ms[2] / n, (dec_ms - sq_ms[0] - sq_ms[1] - sq_ms[2]) / n);
            }
#endif
            tier.rebalance_tick(true);   // the gap until the next request: retire a large batch for the reader
            std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", gen, total, prompt_ms, dec_ms, finish, reused,
                        (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, total - reused);
            std::fflush(stdout);
            if (admit >= 0) {   // BADM: the request decodes on in its slot (the windows), or it is done
                const bool cont = gen == 1 && last_tok >= 0 && admit_max > 1 && std::strcmp(finish, "length") == 0 &&
                                  (int) hist.size() == total && total + 1 < ctxv;
                if (cont) {
                    BSlot & sl = bs[(size_t) admit];
                    sl = BSlot();
                    sl.active = true;
                    sl.x = last_tok;
                    sl.p = total;
                    sl.produced = 1;
                    sl.max_new = admit_max;
                    sl.t0 = now_ms();
                    sl.temp = temp; sl.top_k = tk; sl.top_p = top_p; sl.min_p = min_p;
                    sl.rng = rng;
                    sl.cls = RouteCls{ a.route_bias, a.route_bias_file, cur_rprot, cur_rpool };
                    std::fprintf(stderr, "strata batch: slot %d decodes on from %d tokens (%d reused; route bias %.2f/%.2f)\n",
                                 admit, total, reused, a.route_bias, a.route_bias_file);
                }
                std::printf("BADM %d %d\n", admit, cont ? 1 : 0);
                std::fflush(stdout);
                ++batch_epoch;
                if (!cont) save_live("turn end");   // (a slot that decodes on goes to the pool at its BDONE)
            }
            // batch slots: every turn's boundary into the pool at once (an async copy of the new tail).  With one live
            // state the pool got it when the next request overwrote it; with slots a side request takes another slot,
            // nothing is overwritten, and an earlier boundary of this conversation (an edited turn's resume point) would
            // be gone with the slot's next snapshot (20.10 card: c3edit read 8185 tokens again instead of resuming at 8081)
            else if (my_slot >= 0) {
                save_live("turn end");
#if defined(DS41_HAVE_DSPARK)
                dsp_slot = my_slot;         // a solo request: DSpark's window is this slot's conversation now
                dsp_epoch = batch_epoch;
#endif
            }
        }
        tier.close();
        // the stdin reader thread (detached) sits in getline holding cin's lock: a normal exit's stream teardown waits
        // on it until the client closes the pipe - leave without the static destructors
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    }

    // ---- one-shot: prompt (+ perplexity), then generate
    if (prompt.empty()) { std::fprintf(stderr, "ds41_generate: empty prompt\n"); return 2; }
    tier.reset_stats();
    const double tp0 = now_ms();
    std::vector<float> all;
    const int dm_tail = dump_main ? std::min<int>(128, (int) prompt.size() - 1) : 0;   // DSpark's window, as steps
    bool spec = dsp_path != nullptr;
#if !defined(DS41_HAVE_DSPARK)
    if (spec) { std::fprintf(stderr, "ds41_generate: STRATA_DSPARK: built without ds41_dspark\n"); return 2; }
#endif
    const int spec_depth = std::getenv("STRATA_DSPARK_DEPTH") ? std::atoi(std::getenv("STRATA_DSPARK_DEPTH")) : 4;
    if (spec && (!defer_vram || S < 17 || dump_main || spec_depth < 0 || spec_depth > 5 || std::getenv("STRATA_FORCE_IDS"))) {
        std::fprintf(stderr, "ds41_generate: STRATA_DSPARK needs the GPU layer-major prompt path (S >= 17), depth 1..5, and no "
                             "STRATA_DUMP_MAIN / STRATA_FORCE_IDS\n");
        return 2;
    }
    if (spec && a.temp > 0) std::fprintf(stderr, "note: STRATA_DSPARK decodes greedily (temp ignored)\n");
    // the DSpark window needs per-position main_hidden, which the layer-major prompt does not produce: the last
    // min(128, L-1) prompt tokens run through pass_v (chunks of <= 16) after the VRAM cache opens
    const int sp_tail = spec ? std::min<int>(128, (int) prompt.size() - 1) : 0;
    const bool ok = feed(prompt.data(), (int) prompt.size() - dm_tail - sp_tail, 0, [&](int done) {
        if (!a.quiet) std::fprintf(stderr, "\rprompt %d/%zu", done, prompt.size());
    }, a.ppl || !a.dump_logits.empty() ? &all : nullptr);
    if (!ok) { std::fprintf(stderr, "\nds41_generate: prefill: %s\n", ferr.c_str()); return 1; }
    if (dm_tail > 0 && defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
    for (int i = (int) prompt.size() - dm_tail; i < (int) prompt.size(); ++i)
        if (!step(prompt[(size_t) i], i) || !head_logits(0, 1, last_logits.data())) {
            std::fprintf(stderr, "ds41_generate: dump-main prompt step: %s\n", ferr.c_str());
            return 1;
        }
#if defined(DS41_HAVE_DSPARK)
    int sp_bonus = -1;   // the trunk's token for position L (argmax of the last prompt row) and DSpark's first drafts
    int sp_ids[5] = {0}, sp_tmp[5];
    float sp_conf[5];
    if (spec) {
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        bias_now = false;   // the tail is PROMPT: it routes like the rest of the prompt (no decode biases), as plain decode
        apply_bias();
        std::vector<int> spos;
        for (int i0 = (int) prompt.size() - sp_tail; i0 < (int) prompt.size(); i0 += 16) {
            const int m = std::min<int>(16, (int) prompt.size() - i0);
            if (!pass_v(prompt.data() + i0, m, i0)) { std::fprintf(stderr, "ds41_generate: DSpark seed: %s\n", ferr.c_str()); return 1; }
            spos.resize((size_t) m);
            for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
            const bool last = i0 + m == (int) prompt.size();
            if (last) {
                std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                sp_bonus = (int) (std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
            }
            if (!dsp->step(vh.data(), spos.data(), m, last ? sp_bonus : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark seed step: %s\n", dsp->last_error().c_str());
                return 1;
            }
        }
        bias_now = true;    // decode: route bias + file penalty on (open_cache's state)
        apply_bias();
    }
#endif
    const double prompt_ms = now_ms() - tp0;
    if (defer_vram && dm_tail == 0 && !spec) {   // the prompt is done: its buffers go back, the VRAM expert cache opens into the free space
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "\nVRAM after the prompt: %.2f GiB free -> %.2f GiB of expert slots: %lld experts in VRAM (%.1f s)",
                     seed_free, seed_gib, (long long) tier.resident(), seed_s);
    }
    const Ds41MoeStats pst = tier.stats();
    std::fprintf(stderr, "\nprompt: %zu tokens in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / %lld pcie / %lld file)\n",
                 prompt.size(), prompt_ms / 1000.0, 1000.0 * prompt.size() / prompt_ms, (long long) pst.hits,
                 (long long) pst.cpu, (long long) pst.pcie, (long long) pst.file_tier);
    std::fprintf(stderr, "prompt split: dense %.1f s, experts %.1f s (file wait %.1f s), logits %.1f s\n", fd_dense / 1000.0,
                 fd_tier / 1000.0, pst.file_ms / 1000.0, fd_logits / 1000.0);
    if (a.ppl) {
        double nll = 0;
        int n = 0, top1 = 0;
        for (size_t t = 0; t + 1 < prompt.size(); ++t) {
            const float * lg = all.data() + t * (size_t) nv;
            nll -= log_softmax_at(lg, nv, prompt[t + 1]);
            top1 += (int) (std::max_element(lg, lg + nv) - lg) == prompt[t + 1];
            ++n;
        }
        std::printf("PPL %.4f  mean NLL %.4f over %d tokens, top-1 %.3f\n", std::exp(nll / std::max(1, n)), nll / std::max(1, n), n,
                    (double) top1 / std::max(1, n));
    }
    if (!a.dump_logits.empty()) {
        std::ofstream f(a.dump_logits, std::ios::binary);
        f.write((const char *) all.data(), (std::streamsize) (all.size() * 4));
    }
    if (!a.dump_last.empty()) {   // the last row's logits - the one row enc-skip still computes exactly
        std::ofstream f(a.dump_last, std::ios::binary);
        f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
    }
    tier.reset_stats();
    // STRATA_PROFILE_DECODE=1: nsys --capture-range=cudaProfilerApi captures exactly the decode
    static const bool prof_decode = std::getenv("STRATA_PROFILE_DECODE") != nullptr;
    if (prof_decode) Ds41MoeTier::profiler(true);
    const double td0 = now_ms();
    std::vector<float> lg = last_logits;
    int pos = (int) prompt.size(), gen = 0;
    // STRATA_FORCE_IDS=<file>: decode this token sequence (one id per line - a previous run's stdout) in place of the
    // sampled tokens, so kernel A/Bs whose variants round differently still do the same work (greedy decoding flips on
    // near-ties, and a different text means different experts and a different VRAM hit rate)
    std::vector<int> forced;
    if (const char * fp = std::getenv("STRATA_FORCE_IDS")) {   // lines that are not a bare id (a memguard banner) skipped
        std::ifstream f(fp);
        for (std::string line; std::getline(f, line);) {
            char * end = nullptr;
            const long t = std::strtol(line.c_str(), &end, 10);
            if (end != line.c_str() && *end == '\0') forced.push_back((int) t);
        }
        std::fprintf(stderr, "STRATA_FORCE_IDS: %zu forced tokens from %s\n", forced.size(), fp);
    }
#if defined(DS41_HAVE_DSPARK)
    // ---- DSpark: P = the last fed position, t = the trunk's token for P+1 (not fed yet), d = drafts for P+2..
    // A pass verifies [t, d0..d(k-1)] at P+1..P+1+k; g_i = the argmax of row P+1+i (the trunk's token for P+2+i);
    // a = #leading d_i == g_i; d0..d(a-1) + t' = g_a are emitted, DSpark appends the a+1 accepted rows and drafts
    // after t'.  Every cache is position-indexed (rejected rows are overwritten when re-fed; the window ring is sized
    // for it), except the ratio-2 compressor's 2-slot state ring: a pass writes its LAST 2 tokens' states, so after a
    // rejection a pass that starts at an odd position re-feeds the last accepted token (its group's state then comes
    // from inside the pass).
    // STRATA_ROUTE_TRACE=<file> (measurement only): the verify passes' routing + acceptance, for
    // tools/v41/sim/route_sim.py (session 13: in-pass draft pruning and the VRAM replacement policy, replayed offline)
    if (const char * rt = std::getenv("STRATA_ROUTE_TRACE")) tier.trace_open(rt);
    long sp_pass = 0, sp_acc = 0, sp_refeed = 0, sp_full = 0;
    const bool sp_noacc = std::getenv("STRATA_DSPARK_NOACC") != nullptr;
    double sp_vms = 0, sp_dms = 0;
    long sp_prof_n = 0;
    if (spec) {
        int P = (int) prompt.size() - 1, t = sp_bonus, prev = prompt.back();
        bool dirty = false;   // the last pass rejected something
        std::vector<int> vt;
        auto emit = [&](int tok) -> bool {   // false = stop here
            std::printf("%d\n", tok);
            std::fflush(stdout);
            ++gen;
            return !(std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos || gen >= a.n_predict);
        };
        // STRATA_DSPARK_CONF=<tau>: verify only the leading drafts whose confidence-head probability (sigmoid of
        // conf) is >= tau - a draft that will be rejected still costs its experts' PCIe / NVMe bytes in the verify
        // pass, while every emitted token is still the trunk's own (lossless: only the verify width changes).
        // STRATA_DSPARK_CONFLOG=<file>: per pass "k acc verify_ms p0 p1 p2 p3 p4" (the offline threshold screen).
        static const float conf_tau = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
        static FILE * conf_log = std::getenv("STRATA_DSPARK_CONFLOG") ? std::fopen(std::getenv("STRATA_DSPARK_CONFLOG"), "w") : nullptr;
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        while (emit(t)) {
            int k = std::min(spec_depth, a.n_predict - gen);   // drafts past the budget are never emitted
            if (conf_tau >= 0) {
                int kk = 0;
                while (kk < k && sig(sp_conf[kk]) >= conf_tau) ++kk;
                k = kk;
            }
            const bool refeed = dirty && ((P + 1) & 1);
            vt.clear();
            if (refeed) vt.push_back(prev);
            vt.push_back(t);
            for (int i = 0; i < k; ++i) vt.push_back(sp_ids[i]);
            const int r0 = refeed ? 1 : 0, p0 = refeed ? P : P + 1;
            const double tv = now_ms();
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { std::fprintf(stderr, "ds41_generate: verify: %s\n", ferr.c_str()); return 1; }
            sp_vms += now_ms() - tv;
            // STRATA_DSPARK_DUMP=<file> (correctness tap): every verify row as {pos, m, row, refeed, nv floats} - row
            // `pos` predicts the token at pos+1, so it lines up with row `pos` of a --dump-logits prefill of the stream
            static FILE * sp_dump = std::getenv("STRATA_DSPARK_DUMP") ? std::fopen(std::getenv("STRATA_DSPARK_DUMP"), "wb") : nullptr;
            if (sp_dump) {
                for (int i = 0; i < (int) vt.size(); ++i) {
                    const int32_t hdr[4] = { p0 + i, (int32_t) vt.size(), i, refeed ? 1 : 0 };
                    std::fwrite(hdr, 4, 4, sp_dump);
                    std::fwrite(vlog.data() + (size_t) i * nv, 4, (size_t) nv, sp_dump);
                }
                std::fflush(sp_dump);
            }
            int g[6];
            for (int i = 0; i <= k; ++i) {
                const float * row = vlog.data() + (size_t) (r0 + i) * nv;
                g[i] = (int) (std::max_element(row, row + nv) - row);
            }
            int acc = 0;
            while (acc < k && sp_ids[acc] == g[acc]) ++acc;
            if (sp_noacc) acc = 0;   // bisect tap: every draft rejected (the stream must stay plain greedy)
            ++sp_pass; sp_acc += acc; sp_refeed += refeed; sp_full += acc == k;
            if (conf_log) {
                // acceptance of EVERY draft against the trunk rows this pass computed (beyond k: unknown, -1)
                std::fprintf(conf_log, "%d %d %.2f", k, acc, now_ms() - tv);
                for (int i = 0; i < spec_depth; ++i) std::fprintf(conf_log, " %.4f", sig(sp_conf[i]));
                std::fprintf(conf_log, "\n");
                std::fflush(conf_log);
            }
            {   // STRATA_ROUTE_TRACE: 'P' {m, k, acc, refeed, verify ms, conf p0..p4}
                float pm[10] = { (float) vt.size(), (float) k, (float) acc, (float) refeed, (float) (now_ms() - tv) };
                for (int i = 0; i < 5; ++i) pm[5 + i] = i < spec_depth ? sig(sp_conf[i]) : 0.0f;
                tier.trace_mark(pm, 10);
            }
            int spos[6];
            for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
            const int drafts[5] = { sp_ids[0], sp_ids[1], sp_ids[2], sp_ids[3], sp_ids[4] };
            const double td = now_ms();
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, g[acc], sp_ids, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark step: %s\n", dsp->last_error().c_str());
                return 1;
            }
            sp_dms += now_ms() - td;
            bool more = true;
            for (int i = 0; i < acc && more; ++i) more = emit(drafts[i]);
            if (!more) break;
            prev = acc == 0 ? t : drafts[acc - 1];
            dirty = acc < k;
            P += 1 + acc;
            t = g[acc];
            if (dprof) {
                ++sp_prof_n;
                if (sp_prof_n % 32 == 0 && sp_pass > 0)
                    std::fprintf(stderr, "DSpark ms/token: verify %.2f draft %.2f acc %.2f (%ld passes)\n", sp_vms / sp_pass,
                                 sp_dms / sp_pass, (double) sp_acc / sp_pass, sp_pass);
            }
        }
        if (dprof && sp_pass > 0)
            std::fprintf(stderr, "DSpark totals: verify %.2f ms/pass draft %.2f ms/pass acc %.2f (%ld passes, %ld tokens)\n",
                         sp_vms / sp_pass, sp_dms / sp_pass, (double) sp_acc / sp_pass, sp_pass, gen);
    }
#endif
    for (; !spec && gen < a.n_predict; ++gen) {
        int tok = sample_p(lg.data(), nv, a.temp, 0, 1.0f, 0.0f, rng);
        if (gen < (int) forced.size()) {
            // STRATA_DEC_MARGIN=1 (gate tap, with STRATA_FORCE_IDS): per step the argmax, its logit and the forced
            // token's logit - a forced stream that leaves the greedy path at a near-tie shows a tiny gap there
            static const bool margin = std::getenv("STRATA_DEC_MARGIN") != nullptr;
            if (margin && tok != forced[(size_t) gen])
                std::fprintf(stderr, "margin: step %d argmax %d (%.4f) forced %d (%.4f) gap %.4f\n", gen, tok, lg[(size_t) tok],
                             forced[(size_t) gen], lg[(size_t) forced[(size_t) gen]], lg[(size_t) tok] - lg[(size_t) forced[(size_t) gen]]);
            tok = forced[(size_t) gen];
        }
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos) { ++gen; break; }
        if (gen + 1 == a.n_predict) { ++gen; break; }
        if (!step(tok, pos) || !head_logits(0, 1, lg.data())) { std::fprintf(stderr, "ds41_generate: decode: %s\n", ferr.c_str()); return 1; }
        ++pos;
    }
    const double dec_ms = now_ms() - td0;
    if (prof_decode) Ds41MoeTier::profiler(false);
    const Ds41MoeStats st = tier.stats();
    std::fprintf(stderr, "decode: %d tokens in %.2f s = %.2f tok/s; experts %lld hit / %lld cpu / %lld pcie / %lld file "
                 "(%.1f%% VRAM hits), cpu %.0f ms, file %.0f ms; spec reads %lld issued / %lld used; prefetch %lld issued / "
                 "%lld outside the arena / %lld used\n",
                 gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), (long long) st.hits,
                 (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                 100.0 * st.hits / std::max<int64_t>(1, st.lookups()), st.cpu_ms, st.file_ms,
                 (long long) st.spec_issued, (long long) st.spec_hits, (long long) st.prefetch_issued,
                 (long long) st.prefetch_dummy, (long long) st.prefetched_useful);
    if (spec)
        std::fprintf(stderr, "DSpark: %ld verify passes, %.3f accepted drafts/pass = %.3f tokens/pass, %.1f%% full, %ld re-fed; "
                     "verify %.1f ms/pass, draft %.2f ms/pass (depth %d)\n", sp_pass, (double) sp_acc / std::max(1L, sp_pass),
                     1.0 + (double) sp_acc / std::max(1L, sp_pass), 100.0 * sp_full / std::max(1L, sp_pass), sp_refeed,
                     sp_vms / std::max(1L, sp_pass), sp_dms / std::max(1L, sp_pass), spec_depth);
    if (spec)
        std::fprintf(stderr, "pass_v split (all %ld passes, %.2f tokens each): dense %.1f ms, tier %.1f ms, taps %.1f ms, "
                     "logits %.1f ms per pass\n", pv_n, (double) pv_tok / std::max(1L, pv_n), pv_ms[0] / std::max(1L, pv_n),
                     pv_ms[1] / std::max(1L, pv_n), pv_ms[2] / std::max(1L, pv_n), pv_ms[3] / std::max(1L, pv_n));
    if (dump_main) std::fclose(dump_main);
    if (file_pred) {
        std::fprintf(stderr, "file-pred: %lld file-tier routes", fp_miss[0]);
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " | k=%d: %lld scored, in top4 %.1f%% top8 %.1f%% top16 %.1f%%", k, fp_miss[k],
                         100.0 * fp_hit[k][0] / std::max(1LL, fp_miss[k]), 100.0 * fp_hit[k][1] / std::max(1LL, fp_miss[k]),
                         100.0 * fp_hit[k][2] / std::max(1LL, fp_miss[k]));
        std::fprintf(stderr, "\nfile-pred candidates (file-tier experts in the top 4/8/16, all layers):");
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " k=%d %lld/%lld/%lld", k, fp_cand[k][0], fp_cand[k][1], fp_cand[k][2]);
        std::fprintf(stderr, "\n");
    }
    tier.close();
    return 0;
}
