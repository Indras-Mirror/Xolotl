#include <sstream>
#include <fstream>
// src/program/inkling_main.cpp - `xolotl-inkling --serve --native <inkling-2-mini.gguf> [--context N] [--ubatch N] [--cpu-exps L]`
//
// Inkling-2-Mini (Thinking Machines Lab, Apache 2.0) in Xolotl.  The forward pass is the Inkling 2 graph of llama.cpp
// PR #32694 (src/models/inkling2.cpp + inkling2-loop.cpp, by Pedro Cuenca, MIT) re-built on ggml-backend (CUDA, with
// the CPU backend as the scheduler's fallback), so it reads every GGUF type llama.cpp reads: the upstream converter's
// dequantised GGUFs, llama.cpp quants, and Xolotl's exact parent (NVFP4 experts kept, with per-expert `.scale`).
// The engine skeleton (state, prefix reuse, checkpoint, park/unpark, the serve protocol) is xolotl-glimmer's
// (src/program/glimmer_main.cpp).
//
// It speaks Xolotl's serve line protocol (the header of src/program/dense_main.cpp), so serve/server.py drives it
// through its unchanged StrataEngine:
//   out INFO ... | READY <context> stop | PP <pos> <prompt> <ms> <tok/s> | T <id> | SCORE ... |
//       DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in  GEN <max_new> [key=value ...] id,id,... | GENI <max_new> [key=value ...] <embeddings file> id,id,... |
//       STOP | QUIT | PARK | VRAM [mib]
//
// GENI carries an image: the file is strata-vision's records (int32 'SVE1', n, nx, ny, n_embd followed by
// float32 [n][n_embd], in prompt order) and its rows replace the embeddings of the prompt's image tokens
// (`<|unused_200054|>`, one row per token).  The reference skips the token-embedding norm for image rows, so they are
// fed raw while text rows keep the rms*norm (llama.cpp inkling2.cpp: "do not normalize raw embeddings input"); each
// image token is one position, as in llama.cpp's mtmd (Inkling has no M-RoPE).
//
// The model (GGUF arch `inkling2_loop`, or `inkling2` without the loop): 24 physical layers; layers [loop.start,
// loop.end) (6-17) run loop.count (3) passes, so a token sees 48 layer applications and the state has 48 cache slots
// (slot = physical layer for pass 1, n_phys + (pass-2)*12 + (L-6) for passes 2-3).  No RoPE: a learned relative-position
// bias per head (projected from the query input, gathered by distance) plus log-scaled positions on the global layers.
// Sliding window 1024 with a global layer every 6th.  Short causal convolutions (k=4) on K, V, the attention output and
// the FFN output carry a recurrent state per slot.  Block attention residuals mix the earlier residual streams (a new
// stream starts every attn_res.block_size layers) with learned depth queries.  Passes 2-3 read the previous pass
// through per-layer low-rank controllers.  MoE: 128 routed experts, top-8, sigmoid router + selection bias, SiTU.
//
// Fixed against the reference: slots 24-47 use the SAME per-expert scales as their physical layer (the PR attached the
// scales by slot index after copying the layers, so NVFP4-kept GGUFs ran passes 2-3 unscaled; Xolotl 21 Nov 2026).
//
// State per slot: K/V (global slots: the whole context, append-only, cell = position; sliding slots: a ring of
// window + ubatch cells padded to 256) and the conv state F32 [3, 2*1024 + 2*2304].  The conv state is recurrent, so a
// request resumes only by appending to what the state has read, or from the checkpoint one token before the end of the
// previous prompt (sliding rings + every conv state), else from token 0.  No speculative drafting (a rejected draft
// would leave the conv state past it).
//
// Attention is the non-flash path (K.Q + mask + per-head relative bias, softmax with a zero sink logit): the chunk
// length is capped so the [n_kv, T, n_head] score tensors stay within --attn-budget elements per head-row.
#include "strata/kernels/sampler.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>

// --stats (teacher statistics for our private quantisation pipeline) needs inkling_stats.hpp, which is not part of
// the public tree; without it the engine builds without --stats (the graph taps stay, unused, behind m.st == nullptr).
#if __has_include("inkling_stats.hpp")
#include "inkling_stats.hpp"   // --stats: host accumulators + on-disk formats (header-only)
#define XOLOTL_INKLING_STATS 1
#else
#define XOLOTL_INKLING_STATS 0
namespace inkling_stats { enum { ST_Q = 0, ST_K, ST_V, ST_O, ST_F, ST_ACT, ST_IDS, ST_W, ST_NKIND }; struct Stats; }
#endif
#include "inkling_fused.h"
#include "inkling_dspark.h"    // --dspark: the DSpark block drafter (speculative decoding)
#include "inkling_vision.h"    // --mmproj: the inkling2v image encoder, in-process (GENI/ENC)

#include <signal.h>
#include "strata/platform/win_compat.hpp"   // <unistd.h>; on MSVC also the 64-bit off_t, fseeko, __popcnt, M_PI, setenv

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void die(const std::string& m) { throw std::runtime_error(m); }
struct AllocFail : std::runtime_error { AllocFail() : std::runtime_error("graph allocation failed (VRAM full)") {} };

constexpr int kGraphNodes = 32768;
constexpr int kSpecT = 8;   // longest speculative verify chunk (next + up to 7 drafts)
// cell chunks per attention window at most (keeps the node count flat); XOLOTL_INKLING_ATTN_CHUNKS forces a smaller
// value, which is how the chunked online softmax is self-checked against the single-shot path at a short prompt.
int attn_chunks() {
    static const int n = std::getenv("XOLOTL_INKLING_ATTN_CHUNKS") ? std::max(1, std::atoi(std::getenv("XOLOTL_INKLING_ATTN_CHUNKS"))) : 32;
    return n;
}

// ---------------------------------------------------------------------------------------------------------- GGUF keys
struct Gguf {
    gguf_context* g = nullptr;
    int64_t key(const std::string& k, bool need = true) const {
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0 && need) die("GGUF key missing: " + k);
        return id;
    }
    static int64_t as_int(gguf_type t, const void* p) {
        switch (t) {
            case GGUF_TYPE_UINT8: return *(const uint8_t*) p;
            case GGUF_TYPE_INT8: return *(const int8_t*) p;
            case GGUF_TYPE_UINT16: return *(const uint16_t*) p;
            case GGUF_TYPE_INT16: return *(const int16_t*) p;
            case GGUF_TYPE_UINT32: return *(const uint32_t*) p;
            case GGUF_TYPE_INT32: return *(const int32_t*) p;
            case GGUF_TYPE_UINT64: return (int64_t) *(const uint64_t*) p;
            case GGUF_TYPE_INT64: return *(const int64_t*) p;
            case GGUF_TYPE_BOOL: return *(const int8_t*) p ? 1 : 0;
            case GGUF_TYPE_FLOAT32: return (int64_t) *(const float*) p;
            default: die("GGUF value is not an integer");
        }
    }
    std::vector<int64_t> arr(const std::string& k) const {
        const int64_t id = key(k, false);
        std::vector<int64_t> out;
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return out;
        const gguf_type t = gguf_get_arr_type(g, id);
        const size_t sz = t == GGUF_TYPE_UINT8 || t == GGUF_TYPE_INT8 || t == GGUF_TYPE_BOOL ? 1
                        : t == GGUF_TYPE_UINT16 || t == GGUF_TYPE_INT16 ? 2
                        : t == GGUF_TYPE_UINT64 || t == GGUF_TYPE_INT64 ? 8 : 4;
        const auto* d = (const uint8_t*) gguf_get_arr_data(g, id);
        for (size_t j = 0; j < gguf_get_arr_n(g, id); ++j) out.push_back(as_int(t, d + j * sz));
        return out;
    }
    // a scalar, or the first element of a per-layer array (the caller checks the array is uniform)
    int64_t i(const std::string& k) const {
        const int64_t id = key(k);
        if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
            const auto a = arr(k);
            if (a.empty()) die("GGUF key is an empty array: " + k);
            for (int64_t v : a) if (v != a[0]) die("per-layer values differ (not supported): " + k);
            return a[0];
        }
        return as_int(gguf_get_kv_type(g, id), gguf_get_val_data(g, id));
    }
    int64_t i_or(const std::string& k, int64_t d) const { return key(k, false) < 0 ? d : i(k); }
    float f(const std::string& k) const {
        const int64_t id = key(k);
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
            case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(g, id);
            case GGUF_TYPE_ARRAY: case GGUF_TYPE_STRING: die("GGUF key is not a float: " + k);
            default: return (float) as_int(gguf_get_kv_type(g, id), gguf_get_val_data(g, id));
        }
    }
    float f_or(const std::string& k, float d) const { return key(k, false) < 0 ? d : f(k); }
};

// -------------------------------------------------------------------------------------------------------- the model
struct Hp {
    int n_phys = 0, n_embd = 0, n_head = 0, n_head_kv = 0, hd = 0, n_vocab = 0, n_vocab_pad = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_rel = 0, rel_extent = 0, n_swa = 0, block = 0, n_conv = 0;
    int loop_start = 0, loop_end = 0, n_pass = 1, n_slot = 0;
    int64_t n_ctx_train = 0, log_floor = 1;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_alpha = 0, situ_beta = 1, situ_lb = 0, w_scale = 0, softcap = 0;
    std::vector<bool> swa;     // per physical layer
    int n_q() const { return hd * n_head; }
    int n_kgqa() const { return hd * n_head_kv; }
    int n_chan() const { return 2 * n_kgqa() + 2 * n_embd; }
    bool is_loop(int ip) const { return ip >= loop_start && ip < loop_end && n_pass > 1; }
    int phys_of(int slot) const { return slot < n_phys ? slot : loop_start + (slot - n_phys) % (loop_end - loop_start); }
};

struct Layer {                 // one physical layer
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *gate = nullptr;
    ggml_tensor* wo_b = nullptr;   // optional F32 [n_embd] attention-output bias, from the side file (steering edits)
    ggml_tensor *q_clip = nullptr, *sink_proj = nullptr, *rel = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    ggml_tensor* rel_bias_pad = nullptr;   // rel_bias + a zero row (built at load; was a pad kernel per visit)
    // one-token decode: the projections that read the same input as one mat-vec (rows stacked at load, exact per row):
    // pqrg = [wq ; rel ; gate], pall = [wq ; rel ; gate ; wk ; wv] (sliding layers, where q/k/v share one input)
    ggml_tensor *pqrg = nullptr, *pall = nullptr;
    ggml_tensor *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    ggml_tensor *attn_norm = nullptr, *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;
    ggml_tensor *k_conv_proj = nullptr, *v_conv_proj = nullptr;
    ggml_tensor *attn_res_query = nullptr, *ffn_res_query = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate_inp = nullptr, *probs_b = nullptr;
    ggml_tensor *gate_up_exps = nullptr, *down_exps = nullptr, *gate_up_s = nullptr, *down_s = nullptr;
    // --vram-budget-mib: the gate/up and down arenas are split hot (VRAM) / cold (pinned host RAM); slot 0 of each
    // arena is a zero expert ("not here").  All null when the cache is off, which is the untouched path.  The split
    // moves expert BYTES only: ids, routing weights and the per-expert .scale stay the baseline's (so a layer with
    // no .scale, 20-23 in this quant, never gets one invented for it).
    ggml_tensor *gu_hot = nullptr, *gu_cold = nullptr, *dn_hot = nullptr, *dn_cold = nullptr;
    ggml_tensor *hot_tab = nullptr, *cold_tab = nullptr;                  // I32 [n_expert] -> slot (+1) or 0
    // I32 [k, n_expert]: [e][i] = slot if expert e is in this arena, else a zero row of its own for top-k position i
    // (0 for i = 0, n + i above).  MMQ's id helper (prompt chunks) assumes a token never names one row twice.
    ggml_tensor *hot_tab2 = nullptr, *cold_tab2 = nullptr;
    int n_hot = 0, n_cold = 0;
    // loop controllers (loop layers only): Q, K, V, FFN
    ggml_tensor *msg_query = nullptr, *ctrl_down[4] = {}, *ctrl_gain[4] = {}, *ctrl_forget[4] = {};
    // stacked loop controllers (built at load, exact): q/k/v down [2E, r, 3], q/k/v gain|forget [r, 2E, 3], q/k/v input norms [E, 1, 3]
    ggml_tensor *c3_down = nullptr, *c3_gf = nullptr, *n3 = nullptr;
    // the same controllers in BF16 (the checkpoint's type, so exact) for the fused decode op: q/k/v stacked, and the
    // FFN one; up = [gain rows ; forget rows]
    ggml_tensor *c3_down_h = nullptr, *c3_up_h = nullptr, *c1_down_h = nullptr, *c1_up_h = nullptr;
    ggml_tensor* router_h = nullptr;   // gate_inp in BF16 (the checkpoint's type, exact) for decode / verify
};

struct Slot {                  // one cache slot
    int phys = 0;
    bool swa = true;
    ggml_type k_type = GGML_TYPE_F16, v_type = GGML_TYPE_F16;   // the full slots take --kv-type; the rings stay f16
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr, *conv = nullptr;
    ggml_tensor* conv_hist = nullptr;   // --dspark: [n_conv-1+kSpecT, n_chan] = [state ; chunk inputs] of the last verify
    int64_t cells = 0;
};

// XOLOTL_INKLING_KV_ROT: a full-context K or V leg stored as q4_0 goes through a randomised Hadamard rotation
// before the cache store (default 1; 0 = off; 2 = every leg, f16 and q8_0 too: the exactness check).  A rotated K
// leg rotates q with it (q.k unchanged); a rotated V leg rotates the attention output back once per head.  The
// rotation spreads outlier channels so 4-bit blocks lose less; q8_0 legs stay unrotated (near-lossless without it,
// Mal 22 Nov: "skip q8 rot"), as in the main engine (Q4 K/V rotated, STRATA_KV_ROT INT8 opt-in).
inline int kv_rot_mode() { static const int v = std::getenv("XOLOTL_INKLING_KV_ROT") ? std::atoi(std::getenv("XOLOTL_INKLING_KV_ROT")) : 1; return v; }
inline bool kv_rot_leg(ggml_type t) {
    const int r = kv_rot_mode();
    return r >= 2 || (r == 1 && t == GGML_TYPE_Q4_0);
}
struct InkLora { ggml_tensor *a = nullptr, *b = nullptr; };   // a [n_in, r], b [r, n_out]: y += scale * b (a x)

struct Model {
    Hp hp;
    Gguf gg;
    std::string arch;
    ggml_context* wctx = nullptr;
    ggml_context* sctx = nullptr;
    ggml_tensor *hrot_f = nullptr, *hrot_i = nullptr;   // [128,128] randomised Hadamard M and M^T (quantised K/V rotation)
    ggml_tensor* zsink = nullptr;    // [n_head] zeros: the attention sink logit (state buffer, cleared once; was a fill per layer visit)
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, hbuf = nullptr, sbuf = nullptr;   // GPU weights, host weights, state
    ggml_context* xctx = nullptr; ggml_backend_buffer_t xbuf = nullptr;   // derived weights (stacked controllers)
    ggml_context* rctx = nullptr; ggml_backend_buffer_t rbuf = nullptr;   // derived weights (BF16 routers)
    ggml_context* bctx = nullptr; ggml_backend_buffer_t bbuf = nullptr;   // side-file biases (blk.N.attn_output.bias)
    ggml_context* pctx = nullptr; ggml_backend_buffer_t pbuf = nullptr;   // derived weights (padded rel bias tables)
    bool ubatch_auto = false;   // --ubatch not given: load() sizes it from the free VRAM
    size_t extra_vram = 0;      // VRAM that loads after load() (the DSpark drafter), for the auto ubatch
    bool pattn_ok = false;   // every layer can take the prompt attention (padded bias tables built, hd 128, n_rel 16, 3 q heads per kv head)
    ggml_context* qctx = nullptr; ggml_backend_buffer_t qbuf = nullptr;   // derived weights (stacked projections)
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor *tok_embd = nullptr, *tok_norm = nullptr, *output = nullptr, *output_norm = nullptr, *output_res_query = nullptr;
    std::vector<Layer> L;
    std::vector<Slot> S;
    std::vector<ggml_tensor*> gpu_w, host_wt;   // host: the token table + --cpu-exps experts
    int64_t n_ctx = 0;
    int ubatch = 512;
    int64_t attn_budget = 1 << 25;   // n_kv * T cap for one chunk's attention scores (and for a prefill chunk)
    int64_t ring = 0;
    std::vector<int64_t> ring_pos;   // position held by each ring cell (-1 = empty); one map for every sliding slot
    std::vector<uint8_t> graph_buf;
    size_t weight_bytes = 0, host_bytes = 0, state_bytes = 0;
    ggml_type kv_k = GGML_TYPE_F16, kv_v = GGML_TYPE_F16;   // the full-context K/V cache types
    struct Step* last = nullptr;     // the last built + allocated graph, reused while its shape holds
    int64_t last_key[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    Dspark* dspark = nullptr;        // --dspark: the drafter (its own K/V caches); spec = conv history + taps allocated
    bool spec = false;
    struct GSlot { ggml_backend_t gpu = nullptr; ggml_backend_sched_t sched = nullptr; struct Step* last = nullptr; int64_t key[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1}; std::vector<uint8_t> buf; };
    std::vector<GSlot> vg;           // --dspark: one cached graph (+ its own scheduler) per verify length 2..kSpecT
    uint8_t* host_w = nullptr;       // pinned copy of the GPU weights (park / unpark)
    std::vector<size_t> host_w_at;
    std::vector<uint8_t> host_state;
    bool parked = false;
    inkling_stats::Stats* st = nullptr;   // --stats: accumulate from the graph's taps
    FILE* route_fp = nullptr;        // --route-log: per-visit routing record sink (off: null, no tap, no cost)
    // --lora: one LoRA adapter (xolotl-inkling --train output) applied on top of the base weights in G::mm()
    std::unordered_map<const ggml_tensor*, InkLora> lora;
    float lora_scale = 1.0f;
    ggml_context* lora_ctx = nullptr;
    ggml_backend_buffer_t lora_buf = nullptr;
    int64_t route_records = 0;
    // --vram-budget-mib
    int64_t vram_budget_mib = 0;     // 0: off
    ggml_context* ectx = nullptr;    // the hot + cold expert arenas and their tables/scales
    ggml_context* ectx_host = nullptr;
    ggml_backend_buffer_t ehot_buf = nullptr, ecold_buf = nullptr;
    size_t ehot_bytes = 0, ecold_bytes = 0;
    int64_t ehot_experts = 0, ecold_experts = 0;
    std::vector<std::vector<int32_t>> ehot_of, ecold_of;   // [layer][expert] -> slot (+1) or -1
};

ggml_tensor* get(Model& m, const std::string& name, bool need = true) {
    ggml_tensor* t = ggml_get_tensor(m.wctx, name.c_str());
    if (!t && need) die("tensor missing: " + name);
    return t;
}

void alloc_gpu_weights(Model& m) {
    ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(m.gpu);
    const size_t al = ggml_backend_buft_get_alignment(bt);
    size_t total = 0;
    for (ggml_tensor* t : m.gpu_w) total += GGML_PAD(ggml_backend_buft_get_alloc_size(bt, t), al);
    m.wbuf = ggml_backend_buft_alloc_buffer(bt, total + al);
    if (!m.wbuf) die("cannot allocate the weights in VRAM (" + std::to_string(total >> 20) + " MiB; try --cpu-exps)");
    ggml_backend_buffer_set_usage(m.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tallocr ta = ggml_tallocr_new(m.wbuf);
    for (ggml_tensor* t : m.gpu_w)
        if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) die(std::string("cannot place weight ") + t->name);
    m.weight_bytes = total;
}

// "20-23,5" -> {5, 20, 21, 22, 23}
std::vector<int> parse_layers(const std::string& s, int n) {
    std::vector<int> out;
    if (s.empty() || s == "none") return out;
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(',', p);
        if (e == std::string::npos) e = s.size();
        const std::string part = s.substr(p, e - p);
        const size_t d = part.find('-');
        const int a = std::atoi(part.substr(0, d).c_str()), b = d == std::string::npos ? a : std::atoi(part.substr(d + 1).c_str());
        for (int x = a; x <= b; ++x) if (x >= 0 && x < n) out.push_back(x);
        p = e + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// The three attention-input controllers of a loop layer (q, k, v) run as ONE batched chain: their weights stacked
// along dim 2 (gain and forget rows side by side), so a pass-2/3 visit issues ~14 kernels instead of ~48. Rows are
// computed exactly as before (same kernels per row). XOLOTL_INKLING_CTRL_REF=1 keeps the per-controller chains.
// gate_inp (the router) in BF16: the checkpoint stores it as BF16 and the GGUF's F32 copy holds exactly those values,
// so decode / verify read half the bytes for the same logits (ggml's mat-vec keeps the activations F32)
void build_router_bf16(Model& m) {
    if (std::getenv("XOLOTL_INKLING_ROUTER_F32")) return;
    int n = 0;
    for (int l = 0; l < m.hp.n_phys; ++l) if (m.L[l].gate_inp && m.L[l].gate_inp->type == GGML_TYPE_F32) ++n;
    if (!n) return;
    ggml_init_params ip{(size_t) (n + 2) * ggml_tensor_overhead(), nullptr, true};
    m.rctx = ggml_init(ip);
    for (int l = 0; l < m.hp.n_phys; ++l) {
        Layer& y = m.L[l];
        if (y.gate_inp && y.gate_inp->type == GGML_TYPE_F32)
            y.router_h = ggml_new_tensor_2d(m.rctx, GGML_TYPE_BF16, y.gate_inp->ne[0], y.gate_inp->ne[1]);
    }
    m.rbuf = ggml_backend_alloc_ctx_tensors(m.rctx, m.gpu);
    if (!m.rbuf) { for (int l = 0; l < m.hp.n_phys; ++l) m.L[l].router_h = nullptr; return; }
    std::vector<float> f;
    std::vector<ggml_bf16_t> hb;
    for (int l = 0; l < m.hp.n_phys; ++l) {
        Layer& y = m.L[l];
        if (!y.router_h) continue;
        f.resize(ggml_nelements(y.gate_inp));
        ggml_backend_tensor_get(y.gate_inp, f.data(), 0, f.size() * sizeof(float));
        hb.resize(f.size());
        bool exact = true;
        for (size_t i = 0; i < f.size(); ++i) { hb[i] = ggml_fp32_to_bf16(f[i]); exact = exact && ggml_bf16_to_fp32(hb[i]) == f[i]; }
        if (!exact) { y.router_h = nullptr; continue; }   // not BF16 values after all: keep the F32 router
        ggml_backend_tensor_set(y.router_h, hb.data(), 0, hb.size() * sizeof(ggml_bf16_t));
    }
}

void build_stacked_controllers(Model& m) {
    if (std::getenv("XOLOTL_INKLING_CTRL_REF")) return;
    const Hp& h = m.hp;
    std::vector<int> ls;
    for (int l = 0; l < h.n_phys; ++l) {
        const Layer& y = m.L[l];
        if (!y.ctrl_down[0]) continue;
        bool ok = true;
        for (int k = 0; k < 3; ++k)
            ok = ok && y.ctrl_down[k]->type == GGML_TYPE_F32 && y.ctrl_gain[k]->type == GGML_TYPE_F32 &&
                 y.ctrl_forget[k]->type == GGML_TYPE_F32 && ggml_are_same_shape(y.ctrl_down[k], y.ctrl_down[0]) &&
                 ggml_are_same_shape(y.ctrl_gain[k], y.ctrl_gain[0]) && ggml_are_same_shape(y.ctrl_forget[k], y.ctrl_gain[0]);
        if (!h.swa[l]) ok = ok && y.q_inp_norm && y.k_inp_norm && y.v_inp_norm && y.q_inp_norm->type == GGML_TYPE_F32 &&
                            y.k_inp_norm->type == GGML_TYPE_F32 && y.v_inp_norm->type == GGML_TYPE_F32;
        if (ok) ls.push_back(l);
    }
    if (ls.empty()) return;
    ggml_init_params ip{(ls.size() * 8 + 4) * ggml_tensor_overhead(), nullptr, true};
    m.xctx = ggml_init(ip);
    for (int l : ls) {
        Layer& y = m.L[l];
        const int64_t E2 = y.ctrl_down[0]->ne[0], r = y.ctrl_down[0]->ne[1], E = y.ctrl_gain[0]->ne[1];
        y.c3_down = ggml_new_tensor_3d(m.xctx, GGML_TYPE_F32, E2, r, 3);
        y.c3_gf = ggml_new_tensor_3d(m.xctx, GGML_TYPE_F32, r, 2 * E, 3);
        if (!h.swa[l]) y.n3 = ggml_new_tensor_3d(m.xctx, GGML_TYPE_F32, E, 1, 3);
        y.c3_down_h = ggml_new_tensor_3d(m.xctx, GGML_TYPE_BF16, E2, r, 3);
        y.c3_up_h = ggml_new_tensor_3d(m.xctx, GGML_TYPE_BF16, r, 2 * E, 3);
        if (y.ctrl_down[3] && y.ctrl_down[3]->type == GGML_TYPE_F32 && ggml_are_same_shape(y.ctrl_down[3], y.ctrl_down[0]) &&
            y.ctrl_gain[3] && y.ctrl_forget[3] && ggml_are_same_shape(y.ctrl_gain[3], y.ctrl_gain[0]) &&
            ggml_are_same_shape(y.ctrl_forget[3], y.ctrl_gain[0])) {
            y.c1_down_h = ggml_new_tensor_3d(m.xctx, GGML_TYPE_BF16, E2, r, 1);
            y.c1_up_h = ggml_new_tensor_3d(m.xctx, GGML_TYPE_BF16, r, 2 * E, 1);
        }
    }
    m.xbuf = ggml_backend_alloc_ctx_tensors(m.xctx, m.gpu);
    if (!m.xbuf) {
        for (int l : ls) m.L[l].c3_down = m.L[l].c3_gf = m.L[l].n3 = m.L[l].c3_down_h = m.L[l].c3_up_h = m.L[l].c1_down_h = m.L[l].c1_up_h = nullptr;
        return;
    }
    std::vector<uint8_t> b;
    auto copy = [&](ggml_tensor* src, ggml_tensor* dst, size_t off) {
        b.resize(ggml_nbytes(src));
        ggml_backend_tensor_get(src, b.data(), 0, b.size());
        ggml_backend_tensor_set(dst, b.data(), off, b.size());
    };
    for (int l : ls) {
        Layer& y = m.L[l];
        for (int k = 0; k < 3; ++k) {
            copy(y.ctrl_down[k], y.c3_down, (size_t) k * y.c3_down->nb[2]);
            copy(y.ctrl_gain[k], y.c3_gf, (size_t) k * y.c3_gf->nb[2]);
            copy(y.ctrl_forget[k], y.c3_gf, (size_t) k * y.c3_gf->nb[2] + ggml_nbytes(y.ctrl_gain[k]));
        }
        // BF16 copies: the F32 values are BF16 values (the converter widened them), so the rounding is exact
        std::vector<float> f;
        std::vector<ggml_bf16_t> hb;
        auto copy_h = [&](ggml_tensor* src, ggml_tensor* dst, size_t off_elems) {
            f.resize(ggml_nelements(src));
            ggml_backend_tensor_get(src, f.data(), 0, f.size() * sizeof(float));
            hb.resize(f.size());
            for (size_t i = 0; i < f.size(); ++i) hb[i] = ggml_fp32_to_bf16(f[i]);
            ggml_backend_tensor_set(dst, hb.data(), off_elems * sizeof(ggml_bf16_t), hb.size() * sizeof(ggml_bf16_t));
        };
        const size_t dsz = (size_t) ggml_nelements(y.ctrl_down[0]), gsz = (size_t) ggml_nelements(y.ctrl_gain[0]);
        for (int k = 0; k < 3; ++k) {
            copy_h(y.ctrl_down[k], y.c3_down_h, k * dsz);
            copy_h(y.ctrl_gain[k], y.c3_up_h, k * 2 * gsz);
            copy_h(y.ctrl_forget[k], y.c3_up_h, k * 2 * gsz + gsz);
        }
        if (y.c1_down_h) {
            copy_h(y.ctrl_down[3], y.c1_down_h, 0);
            copy_h(y.ctrl_gain[3], y.c1_up_h, 0);
            copy_h(y.ctrl_forget[3], y.c1_up_h, gsz);
        }
        if (y.n3) {
            copy(y.q_inp_norm, y.n3, 0);
            copy(y.k_inp_norm, y.n3, y.n3->nb[2]);
            copy(y.v_inp_norm, y.n3, 2 * y.n3->nb[2]);
        }
    }
}

// rel_bias [n_rel, Ext] -> [n_rel, Ext+1] with a zero last row, once at load (exact: the graph padded it every visit)
void build_padded_rel_bias(Model& m) {
    std::vector<int> ls;
    for (int l = 0; l < m.hp.n_phys; ++l)
        if (m.L[l].rel_bias && m.L[l].rel_bias->type == GGML_TYPE_F32 && ggml_is_contiguous(m.L[l].rel_bias)) ls.push_back(l);
    if (ls.empty()) return;
    ggml_init_params ip{(ls.size() + 2) * ggml_tensor_overhead(), nullptr, true};
    m.pctx = ggml_init(ip);
    for (int l : ls) {
        const ggml_tensor* rb = m.L[l].rel_bias;
        m.L[l].rel_bias_pad = ggml_new_tensor_2d(m.pctx, GGML_TYPE_F32, rb->ne[0], rb->ne[1] + 1);
    }
    m.pbuf = ggml_backend_alloc_ctx_tensors(m.pctx, m.gpu);
    if (!m.pbuf) { for (int l : ls) m.L[l].rel_bias_pad = nullptr; return; }
    std::vector<uint8_t> b;
    for (int l : ls) {
        ggml_tensor *rb = m.L[l].rel_bias, *pd = m.L[l].rel_bias_pad;
        b.assign(ggml_nbytes(pd), 0);
        ggml_backend_tensor_get(rb, b.data(), 0, ggml_nbytes(rb));
        ggml_backend_tensor_set(pd, b.data(), 0, b.size());
    }
}

size_t vram_free_mib();

// rows of the projections that share an input, stacked once at load for the one-token decode graph (~0.45 GB on the XQ
// quant; skipped when that would leave < 2.5 GB free, or with XOLOTL_INKLING_STACKQ=0)
void build_stacked_proj(Model& m) {
    if (const char* e = std::getenv("XOLOTL_INKLING_STACKQ")) if (std::string(e) == "0") return;
    const Hp& h = m.hp;
    auto same = [](const ggml_tensor* a, const ggml_tensor* b) {
        return a && b && a->type == b->type && a->ne[0] == b->ne[0] && ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
               a->buffer && b->buffer && !ggml_backend_buffer_is_host(a->buffer) && !ggml_backend_buffer_is_host(b->buffer);
    };
    std::vector<int> ls;
    size_t bytes = 0;
    for (int l = 0; l < h.n_phys; ++l) {
        const Layer& y = m.L[l];
        if (!same(y.wq, y.rel) || !same(y.wq, y.gate) || ggml_is_quantized(y.wq->type) == false) continue;
        ls.push_back(l);
        bytes += ggml_nbytes(y.wq) + ggml_nbytes(y.rel) + ggml_nbytes(y.gate);
        if (h.swa[l] && same(y.wq, y.wk) && same(y.wq, y.wv)) bytes += ggml_nbytes(y.wk) + ggml_nbytes(y.wv);
    }
    if (ls.empty() || vram_free_mib() < (bytes >> 20) + 2560) return;
    ggml_init_params ip{(ls.size() * 2 + 2) * ggml_tensor_overhead(), nullptr, true};
    m.qctx = ggml_init(ip);
    for (int l : ls) {
        Layer& y = m.L[l];
        const bool all = h.swa[l] && same(y.wq, y.wk) && same(y.wq, y.wv);
        const int64_t rows = y.wq->ne[1] + y.rel->ne[1] + y.gate->ne[1] + (all ? y.wk->ne[1] + y.wv->ne[1] : 0);
        ggml_tensor* t = ggml_new_tensor_2d(m.qctx, y.wq->type, y.wq->ne[0], rows);
        (all ? y.pall : y.pqrg) = t;
    }
    m.qbuf = ggml_backend_alloc_ctx_tensors(m.qctx, m.gpu);
    if (!m.qbuf) { for (int l : ls) m.L[l].pall = m.L[l].pqrg = nullptr; return; }
    std::vector<uint8_t> b;
    for (int l : ls) {
        Layer& y = m.L[l];
        ggml_tensor* t = y.pall ? y.pall : y.pqrg;
        size_t off = 0;
        for (ggml_tensor* src : {y.wq, y.rel, y.gate, y.pall ? y.wk : nullptr, y.pall ? y.wv : nullptr}) {
            if (!src) continue;
            b.resize(ggml_nbytes(src));
            ggml_backend_tensor_get(src, b.data(), 0, b.size());
            ggml_backend_tensor_set(t, b.data(), off, b.size());
            off += b.size();
        }
        if (y.pall) y.pqrg = ggml_view_2d(m.qctx, y.pall, y.pall->ne[0], y.wq->ne[1] + y.rel->ne[1] + y.gate->ne[1], y.pall->nb[1], 0);
    }
    std::fprintf(stderr, "xolotl-inkling: stacked q/rel/gate(/k/v) projections for decode on %zu layers (%.0f MiB)\n",
                 ls.size(), bytes / 1048576.0);
}

// --vram-budget-mib: split every layer's experts into a hot VRAM arena and a cold pinned-host arena (slot 0 of each
// is a zero expert, the "not here" destination of the remapped ids).  The selection is the top exp_probs_b.bias
// experts per layer until a measured profile from --route-log replaces it; the loop core (visited 3x) is filled
// first (see ~/.quetza-data/conductor/specs/ink-lowvram-SPEC.md).  False: the budget covers everything, cache off.
bool setup_expert_cache(Model& m, const std::string& path) {
    Hp& h = m.hp;
    if (m.vram_budget_mib <= 0) return false;
    const int64_t NE = h.n_expert;
    // non-expert VRAM: every GPU weight except the expert tensors and the (always host) token table
    size_t expert_bytes = 0;
    for (int l = 0; l < h.n_phys; ++l) expert_bytes += ggml_nbytes(m.L[l].gate_up_exps) + ggml_nbytes(m.L[l].down_exps);
    size_t other = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.wctx); t; t = ggml_get_next_tensor(m.wctx, t)) {
        if (t == m.tok_embd) continue;
        bool exp = false;
        for (int l = 0; l < h.n_phys && !exp; ++l) exp = t == m.L[l].gate_up_exps || t == m.L[l].down_exps;
        if (!exp) other += GGML_PAD(ggml_nbytes(t), 256);
    }
    // KV: the same shapes load() allocates (full slots take --kv-type; the sliding rings are f16)
    size_t kv = 0;
    {
        const int64_t cells_ring = std::min<int64_t>(m.n_ctx, ((h.n_swa + m.ubatch + 255) / 256) * 256);
        size_t kfull = ggml_row_size(m.kv_k, h.n_kgqa()), vfull = ggml_row_size(m.kv_v, h.n_kgqa());
        size_t ksw = ggml_row_size(GGML_TYPE_F16, h.n_kgqa()), vsw = ksw;
        int n_full = 0, n_swa = 0;
        for (int s = 0; s < h.n_slot; ++s) (h.swa[h.phys_of(s)] ? n_swa : n_full)++;
        kv = (size_t) n_full * (size_t) m.n_ctx * (kfull + vfull) + (size_t) n_swa * (size_t) cells_ring * (ksw + vsw);
    }
    const size_t total = (size_t) m.vram_budget_mib << 20;
    const size_t reserve = 768ull << 20;   // stacked controllers/projections, CUDA context, graph + scratch
    if (other + kv + reserve >= total) die("--vram-budget-mib: the budget leaves no room for experts (" +
                                           std::to_string(other >> 20) + " MiB weights + " + std::to_string(kv >> 20) + " MiB KV + " +
                                           std::to_string(reserve >> 20) + " MiB reserve)");
    size_t hot_budget = total - other - kv - reserve;
    if (hot_budget >= expert_bytes) {
        std::fprintf(stderr, "xolotl-inkling: --vram-budget-mib %lld: all experts fit (%zu of %zu MiB) - cache off\n",
                     (long long) m.vram_budget_mib, hot_budget >> 20, expert_bytes >> 20);
        return false;
    }

    // the per-expert blob of each projection
    std::vector<size_t> bgu(h.n_phys), bdn(h.n_phys);
    for (int l = 0; l < h.n_phys; ++l) {
        bgu[l] = ggml_nbytes(m.L[l].gate_up_exps) / NE;
        bdn[l] = ggml_nbytes(m.L[l].down_exps) / NE;
    }
    // quotas: round-robin, loop core first (it is visited n_pass times), until the budget is spent
    std::vector<int> q(h.n_phys, 0);
    std::vector<int> order;
    for (int l = h.loop_start; l < h.loop_end && h.n_pass > 1; ++l) order.push_back(l);
    for (int l = 0; l < h.n_phys; ++l) if (!(h.n_pass > 1 && h.is_loop(l))) order.push_back(l);
    auto fill_quotas = [&](size_t budget) {
        q.assign(h.n_phys, 0);
        size_t used = 0;
        for (bool more = true; more;) {
            more = false;
            for (int l : order) {
                if (q[l] >= NE) continue;
                if (used + bgu[l] + bdn[l] > budget) continue;
                ++q[l]; used += bgu[l] + bdn[l];
                more = true;
            }
        }
        return used;
    };
    size_t used = fill_quotas(hot_budget);
    if (used == 0) die("--vram-budget-mib: too small for one expert");
    {   // the staged path (T >= 2) needs a VRAM copy of ONE layer's cold arena at a time; take it out of the hot
        // budget so the graph allocation cannot fail on it, then recompute the quotas once
        size_t max_cold = 0;
        for (int l = 0; l < h.n_phys; ++l) {
            const int64_t nc = NE - q[l];
            if (nc > 0) max_cold = std::max(max_cold, (size_t) (nc + h.n_used) * (bgu[l] + bdn[l]));
        }
        if (max_cold > 0 && hot_budget > max_cold + (128ull << 20)) {
            const size_t used2 = fill_quotas(hot_budget - max_cold - (32ull << 20));
            std::fprintf(stderr, "xolotl-inkling: --vram-budget-mib: staging reserve %.0f MiB for one layer's cold arena "
                         "(hot experts %zu -> %zu MiB)\n", max_cold / 1048576.0, used2 >> 20, used >> 20);
            used = used2;
        }
    }

    // exp_probs_b.bias -> the ranking (one 512-byte read per layer; the main loop reads the tensors themselves)
    // XOLOTL_INKLING_ARENA_IDENTITY=1: slot = expert, everything hot.  A probe mode: the arena then computes the
    // same graph as the baseline over different tensor objects, so a mismatch isolates the plumbing (buffer,
    // scheduler, fill) from the permutation.
    static const bool identity = std::getenv("XOLOTL_INKLING_ARENA_IDENTITY") != nullptr;
    // XOLOTL_INKLING_COLD1=L,E: ONE expert cold (everything else hot, identity slots).  The isolation probe:
    // if this faults at T >= 2, the fault is the cold mechanism; if it runs, it is a count/capacity effect.
    int cold1_l = -1, cold1_e = -1;
    if (const char* c1 = std::getenv("XOLOTL_INKLING_COLD1")) {
        const char* comma = std::strchr(c1, ',');
        if (comma) { cold1_l = std::atoi(c1); cold1_e = std::atoi(comma + 1); }
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);
    std::vector<float> probe;
    const size_t data_off = gguf_get_data_offset(m.gg.g);
    for (int l = 0; l < h.n_phys; ++l) {
        if (cold1_l >= 0) {   // one cold expert: slot = expert for all hot ones, the named one goes to the cold arena
            std::vector<int32_t> hot_of(NE, -1), cold_of(NE, -1);
            int nh = 0;
            for (int e = 0; e < NE; ++e) {
                if (l == cold1_l && e == cold1_e) cold_of[e] = 1;
                else hot_of[e] = ++nh;
            }
            m.ehot_of.push_back(std::move(hot_of));
            m.ecold_of.push_back(std::move(cold_of));
            m.L[l].n_hot = nh;
            m.L[l].n_cold = NE - nh;
            continue;
        }
        if (identity) {
            std::vector<int32_t> hot_of(NE), cold_of(NE, -1);
            for (int e = 0; e < NE; ++e) hot_of[e] = e + 1;
            m.ehot_of.push_back(std::move(hot_of));
            m.ecold_of.push_back(std::move(cold_of));
            m.L[l].n_hot = NE;
            m.L[l].n_cold = 0;
            continue;
        }
        const int64_t ti = m.L[l].probs_b ? gguf_find_tensor(m.gg.g, m.L[l].probs_b->name) : -1;
        if (ti < 0) die("--vram-budget-mib needs blk.*.exp_probs_b.bias");
        probe.resize(NE);
        if (fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(m.gg.g, ti)), SEEK_SET) != 0 ||
            std::fread(probe.data(), sizeof(float), NE, f) != (size_t) NE)
            die(std::string("--vram-budget-mib: short read on ") + m.L[l].probs_b->name);
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return probe[a] > probe[b]; });
        std::vector<int32_t> hot_of(NE, -1), cold_of(NE, -1);
        for (int i = 0; i < q[l]; ++i) hot_of[idx[i]] = i + 1;   // slot 0 is the zero expert
        for (int i = q[l]; i < NE; ++i) cold_of[idx[i]] = i - q[l] + 1;
        m.ehot_of.push_back(std::move(hot_of));
        m.ecold_of.push_back(std::move(cold_of));
        m.L[l].n_hot = q[l];
        m.L[l].n_cold = NE - q[l];
    }
    std::fclose(f);

    // contexts: hot arenas + tables on the device, cold arenas in pinned host memory (CUDA host buft)
    ggml_init_params dcp{(size_t) (h.n_phys * 6 + 8) * ggml_tensor_overhead(), nullptr, true};
    const int64_t KD = h.n_used;   // zero rows per arena: one per top-k position (see hot_tab2)
    ggml_init_params hcp{(size_t) (h.n_phys * 2 + 8) * ggml_tensor_overhead(), nullptr, true};
    m.ectx = ggml_init(dcp);
    ggml_context* hctx = ggml_init(hcp);
    for (int l = 0; l < h.n_phys; ++l) {
        Layer& y = m.L[l];
        if (!y.gate_up_exps || (y.n_hot == 0 && y.n_cold == 0)) continue;
        if (y.n_hot > 0) {
            y.gu_hot = ggml_new_tensor_3d(m.ectx, y.gate_up_exps->type, y.gate_up_exps->ne[0], y.gate_up_exps->ne[1], y.n_hot + KD);
            y.dn_hot = ggml_new_tensor_3d(m.ectx, y.down_exps->type, y.down_exps->ne[0], y.down_exps->ne[1], y.n_hot + KD);
        }
        y.hot_tab = ggml_new_tensor_1d(m.ectx, GGML_TYPE_I32, NE);
        y.hot_tab2 = ggml_new_tensor_2d(m.ectx, GGML_TYPE_I32, KD, NE);
        if (y.n_cold > 0) {
            y.gu_cold = ggml_new_tensor_3d(hctx, y.gate_up_exps->type, y.gate_up_exps->ne[0], y.gate_up_exps->ne[1], y.n_cold + KD);
            y.dn_cold = ggml_new_tensor_3d(hctx, y.down_exps->type, y.down_exps->ne[0], y.down_exps->ne[1], y.n_cold + KD);
            y.cold_tab = ggml_new_tensor_1d(m.ectx, GGML_TYPE_I32, NE);
            y.cold_tab2 = ggml_new_tensor_2d(m.ectx, GGML_TYPE_I32, KD, NE);
        }
    }
    m.ehot_buf = ggml_backend_alloc_ctx_tensors(m.ectx, m.gpu);
    ggml_backend_buffer_type_t host_bt = ggml_backend_cuda_host_buffer_type();
    m.ecold_buf = ggml_backend_alloc_ctx_tensors_from_buft(hctx, host_bt);
    // The CUDA host buft falls back to a PAGEABLE cpu buffer when cudaMallocHost fails (ggml-cuda.cu), and a
    // pageable arena is not device-addressable - every read of it from a kernel would fault.  It also advertises
    // get_max_size = 4 GiB while a 12 GB budget's cold tier can exceed that.  Say which one we actually got.
    if (m.ecold_buf) {
        const bool pinned = ggml_backend_buffer_get_type(m.ecold_buf) == host_bt;
        std::fprintf(stderr, "xolotl-inkling: cold arena buffer: %s (%s)\n", ggml_backend_buffer_name(m.ecold_buf),
                     pinned ? "CUDA host (pinned, device-addressable)" : "NOT the CUDA host buft - pageable fallback");
        if (std::getenv("XOLOTL_INKLING_REQUIRE_PINNED") && !pinned)
            die("--vram-budget-mib: the cold arena is not pinned memory; a kernel read of it would fault");
    }
    if (!m.ehot_buf || !m.ecold_buf) die("--vram-budget-mib: cannot allocate the expert arenas");
    ggml_backend_buffer_set_usage(m.ehot_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(m.ecold_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.ehot_bytes = ggml_backend_buffer_get_size(m.ehot_buf);
    m.ecold_bytes = ggml_backend_buffer_get_size(m.ecold_buf);
    for (int l = 0; l < h.n_phys; ++l) {
        Layer& y = m.L[l];
        if (y.gu_hot) m.ehot_experts += y.n_hot;
        if (y.gu_cold) m.ecold_experts += y.n_cold;
    }
    std::fprintf(stderr, "xolotl-inkling: --vram-budget-mib %lld: %lld experts hot (%.0f MiB VRAM), %lld cold (%.0f MiB pinned RAM); "
                 "budget hot %.0f of %.0f MiB experts\n",
                 (long long) m.vram_budget_mib, (long long) m.ehot_experts, m.ehot_bytes / 1048576.0,
                 (long long) m.ecold_experts, m.ecold_bytes / 1048576.0, hot_budget / 1048576.0, expert_bytes / 1048576.0);
    // stash the cold context for the fill; freed with the model's lifetime
    m.ectx_host = hctx;
    return true;
}

// XOLOTL_INKLING_SELFTEST=1: does mul_mat_id over the arena reproduce mul_mat_id over the ORIGINAL expert tensor?
// The original tensor is read into a throwaway device buffer for ONE layer (~4 MB), a random x and a fixed routing
// row go through both, and the two outputs are compared.  Same kernel, same bytes => the only difference is the
// remap, so a mismatch localizes the bug to the table/slot mapping rather than to the model graph.
void arena_selftest(Model& m, FILE* f, size_t base, int l) {
    const Hp& h = m.hp;
    const int64_t E = h.n_embd, NE = h.n_expert, k = h.n_used, T = 2;
    Layer& y = m.L[l];
    if (!y.gu_hot) { std::fprintf(stderr, "selftest: blk.%d has no hot arena - skipped\n", l); return; }
    const size_t blob = ggml_nbytes(y.gate_up_exps) / NE;
    const int64_t ti = gguf_find_tensor(m.gg.g, y.gate_up_exps->name);
    const size_t t_off = gguf_get_tensor_offset(m.gg.g, ti);

    ggml_init_params op{(size_t) 16 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* oc = ggml_init(op);
    ggml_tensor* orig = ggml_new_tensor_3d(oc, y.gate_up_exps->type, y.gate_up_exps->ne[0], y.gate_up_exps->ne[1], NE);
    ggml_backend_buffer_t ob = ggml_backend_alloc_ctx_tensors(oc, m.gpu);
    if (!ob) die("selftest: cannot allocate the reference tensor");
    std::vector<uint8_t> b(blob);
    for (int e = 0; e < NE; ++e) {
        if (fseeko(f, (off_t) (base + t_off + (size_t) e * blob), SEEK_SET) != 0 || std::fread(b.data(), 1, blob, f) != blob)
            die("selftest: short read");
        ggml_backend_tensor_set(orig, b.data(), (size_t) e * blob, blob);
    }

    std::vector<uint8_t> gb(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    ggml_init_params gp{gb.size(), gb.data(), true};
    ggml_context* c = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, 1, T);
    ggml_tensor* idt = ggml_new_tensor_2d(c, GGML_TYPE_I32, k, T);
    ggml_tensor* ref = ggml_mul_mat_id(c, orig, x, idt);
    ggml_tensor* hid = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_2d(c, y.hot_tab, 1, NE),
                                     ggml_reshape_2d(c, ggml_cont(c, idt), k * T, 1)), k, T);
    ggml_tensor* tst = ggml_mul_mat_id(c, y.gu_hot, x, hid);
    ggml_set_output(ref);
    ggml_set_output(tst);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, tst);

    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    ggml_backend_sched_t sc = ggml_backend_sched_new(bes, nullptr, 2, 4096, false, true);
    ggml_backend_sched_reset(sc);
    if (!ggml_backend_sched_alloc_graph(sc, gf)) die("selftest: graph alloc failed");
    std::vector<float> xv((size_t) E * T);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = 0.05f * std::sin((float) i * 0.37f) + 0.01f * std::cos((float) i);
    std::vector<int32_t> idv((size_t) k * T);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) idv[(size_t) j + (size_t) t * k] = (j * 17 + t * 5 + 3) % NE;   // spread over the table
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_tensor_set(idt, idv.data(), 0, idv.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sc, gf) != GGML_STATUS_SUCCESS) die("selftest: compute failed");
    std::vector<float> a((size_t) E * k * T), bb((size_t) E * k * T);
    ggml_backend_tensor_get(ref, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(tst, bb.data(), 0, bb.size() * sizeof(float));
    double mx = 0, mag = 0;
    size_t first = SIZE_MAX;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) bb[i]);
        if (d > mx) { mx = d; }
        if (d > 1e-6 && first == SIZE_MAX) first = i;
        mag = std::max(mag, std::fabs((double) a[i]));
    }
    std::fprintf(stderr, "selftest blk.%d gate_up: arena vs original ids=%d,%d.. k=%lld: max|d| %.3e (|ref| %.3e) %s\n",
                 l, idv[0], idv[1], (long long) k, mx, mag, mx == 0.0 ? "BIT-IDENTICAL" : (mx < 1e-5 ? "numerically equal" : "MISMATCH"));
    if (first != SIZE_MAX) std::fprintf(stderr, "  first differing element %zu (expert j=%zu t=%zu channel=%zu)\n",
                                        first, first % (size_t) k, (first / (size_t) k) % (size_t) T, first / ((size_t) k * T));
    ggml_backend_sched_free(sc);
    ggml_free(c);
    ggml_backend_buffer_free(ob);
    ggml_free(oc);
}

// `--arena-selftest <gguf>`: the remap question alone, without loading the model.  One layer's gate/up experts are
// read into a throwaway device tensor (~4 MB), a permuted slot table is built, and mul_mat_id over the permuted
// arena is compared with mul_mat_id over the original ids - both identity and reversed permutations.  This is the
// probe that separates "storage/plumbing" from "remap" at a cost of a few MB of VRAM and one CUDA context.
int arena_selftest_main(const std::string& path, int layer) {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) die("selftest: CUDA backend failed to start");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_context* wctx = nullptr;
    gguf_init_params gp{true, &wctx};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) die("selftest: cannot read " + path);
    const std::string pre = "blk." + std::to_string(layer) + ".";
    ggml_tensor* t = ggml_get_tensor(wctx, (pre + "ffn_gate_up_exps.weight").c_str());
    if (!t) die("selftest: no " + pre + "ffn_gate_up_exps.weight");
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], NE = t->ne[2];
    const size_t blob = ggml_nbytes(t) / NE;
    const int64_t ti = gguf_find_tensor(g, t->name);
    const size_t t_off = gguf_get_tensor_offset(g, ti);
    const size_t base = gguf_get_data_offset(g);
    const int k = 8, T = 2;
    std::fprintf(stderr, "selftest: blk.%d gate_up %lldx%lldx%lld type %s, blob %.1f MiB, k=%d T=%d\n", layer,
                 (long long) ne0, (long long) ne1, (long long) NE, ggml_type_name(t->type), blob / 1048576.0, k, T);

    ggml_init_params op{(size_t) 32 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* oc = ggml_init(op);
    ggml_tensor* orig = ggml_new_tensor_3d(oc, t->type, ne0, ne1, NE);
    ggml_tensor* hot = ggml_new_tensor_3d(oc, t->type, ne0, ne1, NE + 1);
    ggml_tensor* tab = ggml_new_tensor_1d(oc, GGML_TYPE_I32, NE);
    ggml_backend_buffer_t ob = ggml_backend_alloc_ctx_tensors(oc, gpu);
    if (!ob) die("selftest: cannot allocate the probe tensors");
    {   // both tensors get the SAME bytes: expert e at slot e (orig) and at slot e+1 (arena, slot 0 = zero expert)
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) die("selftest: cannot open " + path);
        std::vector<uint8_t> b(blob);
        for (int64_t e = 0; e < NE; ++e) {
            if (fseeko(f, (off_t) (base + t_off + (size_t) e * blob), SEEK_SET) != 0 || std::fread(b.data(), 1, blob, f) != blob)
                die("selftest: short read");
            ggml_backend_tensor_set(orig, b.data(), (size_t) e * blob, blob);
            ggml_backend_tensor_set(hot, b.data(), (size_t) (e + 1) * blob, blob);
        }
        std::fclose(f);
    }
    std::vector<uint8_t> zb(blob, 0);
    ggml_backend_tensor_set(hot, zb.data(), 0, blob);

    std::vector<uint8_t> gb(ggml_tensor_overhead() * 128 + ggml_graph_overhead());
    ggml_init_params gip{gb.size(), gb.data(), true};
    std::vector<float> xv((size_t) ne0 * T);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = 0.05f * std::sin((float) i * 0.37f) + 0.01f * std::cos((float) i);
    std::vector<int32_t> idv((size_t) k * T);
    for (int tt = 0; tt < T; ++tt)
        for (int j = 0; j < k; ++j) idv[(size_t) j + (size_t) tt * k] = (int) ((j * 17 + tt * 5 + 3) % NE);

    int n_bad = 0;
    for (int mode = 0; mode < 2; ++mode) {   // 0 = identity (slot = expert+1), 1 = reversed permutation
        std::vector<int32_t> tb(NE);
        if (mode == 0) for (int64_t e = 0; e < NE; ++e) tb[e] = (int32_t) (e + 1);
        else {
            // expert e lives in slot (NE - e): a permutation that keeps every expert in a valid, distinct slot
            std::vector<int32_t> slot(NE);
            for (int64_t e = 0; e < NE; ++e) slot[e] = (int32_t) (NE - e);
            // rewrite the arena under the reversed layout so slot s holds the expert the table claims
            ggml_backend_buffer_t ob2 = ob;   // reuse: refill hot with the permuted order
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) die("selftest: cannot open " + path);
            std::vector<uint8_t> b(blob);
            for (int64_t e = 0; e < NE; ++e) {
                if (fseeko(f, (off_t) (base + t_off + (size_t) e * blob), SEEK_SET) != 0 || std::fread(b.data(), 1, blob, f) != blob)
                    die("selftest: short read");
                ggml_backend_tensor_set(hot, b.data(), (size_t) slot[e] * blob, blob);
            }
            std::fclose(f);
            ggml_backend_tensor_set(hot, zb.data(), 0, blob);
            for (int64_t e = 0; e < NE; ++e) tb[e] = slot[e];
            (void) ob2;
        }
        ggml_backend_tensor_set(tab, tb.data(), 0, (size_t) NE * sizeof(int32_t));

        ggml_context* c = ggml_init(gip);
        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, ne0, 1, T);
        ggml_tensor* idt = ggml_new_tensor_2d(c, GGML_TYPE_I32, k, T);
        ggml_tensor* ref = ggml_mul_mat_id(c, orig, x, idt);
        ggml_tensor* hid = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_2d(c, tab, 1, NE),
                                         ggml_reshape_2d(c, ggml_cont(c, idt), k * T, 1)), k, T);
        ggml_tensor* tst = ggml_mul_mat_id(c, hot, x, hid);
        ggml_set_output(ref);
        ggml_set_output(tst);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, tst);
        ggml_backend_t bes[2] = {gpu, cpu};
        ggml_backend_sched_t sc = ggml_backend_sched_new(bes, nullptr, 2, 4096, false, true);
        ggml_backend_sched_reset(sc);
        if (!ggml_backend_sched_alloc_graph(sc, gf)) die("selftest: graph alloc failed");
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(idt, idv.data(), 0, idv.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sc, gf) != GGML_STATUS_SUCCESS) die("selftest: compute failed");
        std::vector<float> a(xv.size() * k), bb(xv.size() * k);
        ggml_backend_tensor_get(ref, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(tst, bb.data(), 0, bb.size() * sizeof(float));
        double mx = 0, mag = 0;
        size_t first = SIZE_MAX;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = std::fabs((double) a[i] - (double) bb[i]);
            if (d > mx) mx = d;
            if (d > 1e-6 && first == SIZE_MAX) first = i;
            mag = std::max(mag, std::fabs((double) a[i]));
        }
        std::fprintf(stderr, "selftest %s: max|d| %.3e (|ref| %.3e) -> %s%s\n", mode == 0 ? "identity" : "reversed", mx, mag,
                     mx == 0.0 ? "BIT-IDENTICAL" : (mx < 1e-5 ? "numerically equal" : "MISMATCH"),
                     first != SIZE_MAX ? " (first diff shown next)" : "");
        if (first != SIZE_MAX) {
            std::fprintf(stderr, "  first differing element %zu: expert j=%zu t=%zu channel=%zu  ref %.6f arena %.6f\n",
                         first, first % (size_t) k, (first / (size_t) k) % (size_t) T, first / ((size_t) k * T), a[first], bb[first]);
            ++n_bad;
        }
        ggml_backend_sched_free(sc);
        ggml_free(c);
    }
    ggml_backend_buffer_free(ob);
    ggml_free(oc);
    gguf_free(g);
    ggml_free(wctx);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::fprintf(stderr, "selftest: %s\n", n_bad ? "FAIL - the remap changes the result" : "PASS - the remap is transparent");
    return n_bad ? 1 : 0;
}

inline bool prompt_attn_on();
void load(Model& m, const std::string& path, int64_t want_ctx, int threads, const std::string& cpu_exps, bool park_copy) {
    gguf_init_params gp{true, &m.wctx};
    m.gg.g = gguf_init_from_file(path.c_str(), gp);
    if (!m.gg.g) die("cannot read GGUF " + path);
    m.arch = gguf_get_val_str(m.gg.g, m.gg.key("general.architecture"));
    if (m.arch != "inkling2_loop" && m.arch != "inkling2") die("not an Inkling 2 GGUF (general.architecture = " + m.arch + ")");
    const std::string a = m.arch + ".";
    Hp& h = m.hp;
    h.n_phys = (int) m.gg.i(a + "block_count");
    h.n_embd = (int) m.gg.i(a + "embedding_length");
    h.n_head = (int) m.gg.i(a + "attention.head_count");
    h.n_head_kv = (int) m.gg.i(a + "attention.head_count_kv");
    h.hd = (int) m.gg.i_or(a + "attention.key_length", h.n_embd / h.n_head);
    if (m.gg.i_or(a + "attention.value_length", h.hd) != h.hd) die("key_length != value_length");
    h.eps = m.gg.f(a + "attention.layer_norm_rms_epsilon");
    h.qk_clip = m.gg.f(a + "attention.qk_clip");
    h.n_rel = (int) m.gg.i(a + "attention.relative_length");
    h.rel_extent = (int) m.gg.i(a + "attention.relative_extent");
    h.rel_clip = m.gg.f(a + "attention.relative_clip");
    h.log_floor = m.gg.i(a + "attention.log_scale_floor");
    h.log_alpha = m.gg.f(a + "attention.log_scale_alpha");
    h.n_swa = (int) m.gg.i(a + "attention.sliding_window");
    h.block = (int) m.gg.i(a + "attn_res.block_size");
    h.n_conv = (int) m.gg.i(a + "shortconv.l_cache");
    h.n_expert = (int) m.gg.i(a + "expert_count");
    h.n_used = (int) m.gg.i(a + "expert_used_count");
    h.n_ff_exp = (int) m.gg.i(a + "expert_feed_forward_length");
    h.situ_beta = m.gg.f(a + "activation.situ_beta");
    h.situ_lb = m.gg.f(a + "activation.situ_linear_beta");
    h.w_scale = m.gg.f_or(a + "expert_weights_scale", 0.0f);
    h.softcap = m.gg.f_or(a + "final_logit_softcapping", 0.0f);
    h.n_vocab_pad = (int) m.gg.i(a + "vocab_size");
    h.n_vocab = (int) m.gg.i(a + "vocab_size_unpadded");
    h.n_ctx_train = m.gg.i_or(a + "context_length", 131072);
    {
        const auto pat = m.gg.arr(a + "attention.sliding_window_pattern");
        if ((int) pat.size() != h.n_phys) die("sliding_window_pattern has the wrong length");
        for (int l = 0; l < h.n_phys; ++l) h.swa.push_back(pat[l] != 0);
    }
    if (m.arch == "inkling2_loop") {
        h.loop_start = (int) m.gg.i(a + "loop.start");
        h.loop_end = (int) m.gg.i(a + "loop.end");
        h.n_pass = (int) m.gg.i(a + "loop.count");
        if (const char* np = std::getenv("XOLOTL_INKLING_PASSES")) {   // research: run the loop core N times (trained: 3)
            h.n_pass = std::atoi(np);
            std::fprintf(stderr, "xolotl-inkling: XOLOTL_INKLING_PASSES=%d (trained %lld)\n", h.n_pass, (long long) m.gg.i(a + "loop.count"));
        }
        if (m.gg.i_or(a + "loop.varying", 0)) die("a varying loop (loop.varying = true) is not supported");
        if (h.n_pass < 2 || h.loop_start <= 0 || h.loop_start >= h.loop_end || h.loop_end > h.n_phys ||
            h.loop_start % h.block || h.loop_end % h.block) die("invalid loop core");
    }
    h.n_slot = h.n_phys + (h.n_pass - 1) * (h.loop_end - h.loop_start);
    if (h.n_conv < 2 || h.n_head % h.n_head_kv) die("invalid dimensions");
    m.n_ctx = want_ctx > 0 ? want_ctx : std::min<int64_t>(h.n_ctx_train, 32768);

    m.tok_embd = get(m, "token_embd.weight");
    m.tok_norm = get(m, "token_embd_norm.weight");
    m.output = get(m, "output.weight");
    m.output_norm = get(m, "output_norm.weight");
    m.output_res_query = get(m, "output_res_query.weight");
    if (m.tok_embd->ne[1] < h.n_vocab || m.output->ne[1] < h.n_vocab) die("token table smaller than the vocabulary");
    m.L.resize(h.n_phys);
    for (int l = 0; l < h.n_phys; ++l) {
        Layer& y = m.L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.wq = get(m, p + "attn_q.weight"); y.wk = get(m, p + "attn_k.weight"); y.wv = get(m, p + "attn_v.weight");
        y.wo = get(m, p + "attn_output.weight"); y.gate = get(m, p + "attn_gate.weight");
        y.wo_b = get(m, p + "attn_output.bias", false);   // optional steering bias baked into a release file (a side
                                                          // file's bias of the same layer still replaces it)
        if (y.wo_b && (y.wo_b->type != GGML_TYPE_F32 || ggml_nelements(y.wo_b) != h.n_embd))
            die(p + "attn_output.bias must be F32 [n_embd]");
        y.q_clip = get(m, p + "attn_q_clip.weight"); y.sink_proj = get(m, p + "attn_sink_proj.weight");
        y.rel = get(m, p + "attn_rel.weight"); y.rel_bias = get(m, p + "attn_rel_bias.weight"); y.rel_clip = get(m, p + "attn_rel_clip.weight");
        y.k_conv = get(m, p + "attn_k_conv.weight"); y.v_conv = get(m, p + "attn_v_conv.weight");
        y.attn_post_conv = get(m, p + "attn_post_conv.weight"); y.ffn_post_conv = get(m, p + "ffn_post_conv.weight");
        if (h.swa[l]) y.attn_norm = get(m, p + "attn_norm.weight");
        else {
            y.q_inp_norm = get(m, p + "attn_q_inp_norm.weight"); y.k_inp_norm = get(m, p + "attn_k_inp_norm.weight");
            y.v_inp_norm = get(m, p + "attn_v_inp_norm.weight");
            y.k_conv_proj = get(m, p + "attn_k_conv_proj.weight"); y.v_conv_proj = get(m, p + "attn_v_conv_proj.weight");
        }
        y.attn_res_query = get(m, p + "attn_res_query.weight", l > 0);
        y.ffn_res_query = get(m, p + "ffn_res_query.weight");
        y.ffn_norm = get(m, p + "ffn_norm.weight"); y.gate_inp = get(m, p + "ffn_gate_inp.weight");
        y.probs_b = get(m, p + "exp_probs_b.bias", false);
        y.gate_up_exps = get(m, p + "ffn_gate_up_exps.weight"); y.down_exps = get(m, p + "ffn_down_exps.weight");
        y.gate_up_s = get(m, p + "ffn_gate_up_exps.scale", false); y.down_s = get(m, p + "ffn_down_exps.scale", false);
        if (h.is_loop(l)) {
            static const char* cn[4] = {"q", "k", "v", "ffn"};
            y.msg_query = get(m, p + "loop_msg_query.weight");
            for (int c = 0; c < 4; ++c) {
                y.ctrl_down[c] = get(m, p + "loop_" + cn[c] + "_ctrl_down.weight");
                y.ctrl_gain[c] = get(m, p + "loop_" + cn[c] + "_ctrl_gain.weight");
                y.ctrl_forget[c] = get(m, p + "loop_" + cn[c] + "_ctrl_forget.weight");
            }
        }
    }

    // backends: weights on the GPU but the token table (and --cpu-exps experts) in host RAM
    m.gpu = ggml_backend_cuda_init(0);
    if (!m.gpu) die("ggml CUDA backend failed to start");
    m.cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m.cpu, threads);
    std::vector<bool> host_exp(h.n_phys, false);
    for (int l : parse_layers(cpu_exps, h.n_phys)) host_exp[l] = true;
    if (!host_exp.empty() && std::any_of(host_exp.begin(), host_exp.end(), [](bool b) { return b; }) && m.vram_budget_mib > 0)
        die("--cpu-exps and --vram-budget-mib are mutually exclusive");
    setup_expert_cache(m, path);   // --vram-budget-mib: hot VRAM + cold pinned-RAM arenas (off by default)
    for (ggml_tensor* t = ggml_get_first_tensor(m.wctx); t; t = ggml_get_next_tensor(m.wctx, t)) {
        bool exp = false;
        for (int l = 0; l < h.n_phys && !exp; ++l) exp = t == m.L[l].gate_up_exps || t == m.L[l].down_exps;
        if (exp && m.ectx) continue;   // --vram-budget-mib: the arenas take these bytes (filled below)
        bool host = t == m.tok_embd;
        for (int l = 0; l < h.n_phys && !host; ++l)
            host = host_exp[l] && (t == m.L[l].gate_up_exps || t == m.L[l].down_exps);
        (host ? m.host_wt : m.gpu_w).push_back(t);
    }
    alloc_gpu_weights(m);
    {
        ggml_backend_buffer_type_t bt = ggml_backend_cpu_buffer_type();
        const size_t al = ggml_backend_buft_get_alignment(bt);
        size_t total = 0;
        for (ggml_tensor* t : m.host_wt) total += GGML_PAD(ggml_backend_buft_get_alloc_size(bt, t), al);
        m.hbuf = ggml_backend_buft_alloc_buffer(bt, total + al);
        if (!m.hbuf) die("cannot allocate the host weights in RAM");
        ggml_backend_buffer_set_usage(m.hbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(m.hbuf);
        for (ggml_tensor* t : m.host_wt)
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) die(std::string("cannot place host weight ") + t->name);
        m.host_bytes = total;
    }
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) die("cannot open " + path);
        size_t total = 0;
        for (ggml_tensor* t : m.gpu_w) total += GGML_PAD(ggml_nbytes(t), 64);
        if (!park_copy || cudaHostAlloc((void**) &m.host_w, total, cudaHostAllocDefault) != cudaSuccess) m.host_w = nullptr;
        std::vector<uint8_t> buf;
        size_t at = 0;
        const size_t base = gguf_get_data_offset(m.gg.g);
        std::vector<size_t> at_of(gguf_get_n_tensors(m.gg.g), SIZE_MAX);
        std::vector<bool> is_host(gguf_get_n_tensors(m.gg.g), false);
        for (ggml_tensor* t : m.host_wt) is_host[gguf_find_tensor(m.gg.g, t->name)] = true;
        // XOLOTL_INKLING_OVERRIDE=<side.gguf>: tensors of the same name, type and size are read from that file instead
        // (A/B of weight edits - e.g. other attention-output edits - without a second 14 GB model file)
        gguf_context* ov = nullptr;
        FILE* fo = nullptr;
        int n_ov = 0;
        if (const char* op = std::getenv("XOLOTL_INKLING_OVERRIDE")) {
            gguf_init_params og{true, nullptr};
            ov = gguf_init_from_file(op, og);
            fo = ov ? std::fopen(op, "rb") : nullptr;
            if (!ov || !fo) die(std::string("XOLOTL_INKLING_OVERRIDE: cannot read ") + op);
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(m.gg.g); ++i) {
            ggml_tensor* t = ggml_get_tensor(m.wctx, gguf_get_tensor_name(m.gg.g, i));
            bool exp = false;
            for (int l = 0; l < h.n_phys && !exp; ++l) exp = t == m.L[l].gate_up_exps || t == m.L[l].down_exps;
            if (exp && m.ectx) continue;   // --vram-budget-mib: read into the hot/cold arenas after this loop
            const size_t n = ggml_nbytes(t);
            uint8_t* dst = nullptr;
            if (is_host[i]) dst = (uint8_t*) t->data;                                  // straight into the host buffer
            else if (m.host_w) { dst = m.host_w + at; at_of[i] = at; at += GGML_PAD(n, 64); }
            else { buf.resize(n); dst = buf.data(); }
            const int64_t oi = ov ? gguf_find_tensor(ov, t->name) : -1;
            if (oi >= 0) {
                if (gguf_get_tensor_type(ov, oi) != t->type || gguf_get_tensor_size(ov, oi) != n)
                    die(std::string("XOLOTL_INKLING_OVERRIDE: ") + t->name + " differs in type or size");
                if (fseeko(fo, (off_t) (gguf_get_data_offset(ov) + gguf_get_tensor_offset(ov, oi)), SEEK_SET) != 0 ||
                    std::fread(dst, 1, n, fo) != n) die(std::string("XOLOTL_INKLING_OVERRIDE: short read on ") + t->name);
                ++n_ov;
            } else if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(m.gg.g, i)), SEEK_SET) != 0 || std::fread(dst, 1, n, f) != n) {
                std::fclose(f);
                die(std::string("short read on tensor ") + t->name);
            }
            if (!is_host[i]) ggml_backend_tensor_set(t, dst, 0, n);
        }
        std::fclose(f);
        if (ov) {
            // extra tensors the base file lacks: blk.N.attn_output.bias (F32 [n_embd]), added to the attention output
            // after its post-conv on every visit of layer N - an additive steering vector, still a weight in the side file
            std::vector<std::pair<int, int64_t>> bl;
            for (int l = 0; l < m.hp.n_phys; ++l) {
                const int64_t oi = gguf_find_tensor(ov, ("blk." + std::to_string(l) + ".attn_output.bias").c_str());
                if (oi < 0) continue;
                if (gguf_get_tensor_type(ov, oi) != GGML_TYPE_F32 || gguf_get_tensor_size(ov, oi) != (size_t) m.hp.n_embd * 4)
                    die("XOLOTL_INKLING_OVERRIDE: blk." + std::to_string(l) + ".attn_output.bias must be F32 [n_embd]");
                bl.push_back({l, oi});
            }
            if (!bl.empty()) {
                ggml_init_params sp{(bl.size() + 2) * ggml_tensor_overhead(), nullptr, true};
                m.bctx = ggml_init(sp);
                for (auto& [l, oi] : bl) m.L[l].wo_b = ggml_new_tensor_1d(m.bctx, GGML_TYPE_F32, m.hp.n_embd);
                m.bbuf = ggml_backend_alloc_ctx_tensors(m.bctx, m.gpu);
                if (!m.bbuf) die("XOLOTL_INKLING_OVERRIDE: cannot allocate the bias tensors");
                std::vector<float> v(m.hp.n_embd);
                for (auto& [l, oi] : bl) {
                    if (fseeko(fo, (off_t) (gguf_get_data_offset(ov) + gguf_get_tensor_offset(ov, oi)), SEEK_SET) != 0 ||
                        std::fread(v.data(), 4, v.size(), fo) != v.size()) die("XOLOTL_INKLING_OVERRIDE: short read on a bias");
                    ggml_backend_tensor_set(m.L[l].wo_b, v.data(), 0, v.size() * 4);
                }
                std::fprintf(stderr, "xolotl-inkling: %zu attn_output biases from XOLOTL_INKLING_OVERRIDE\n", bl.size());
            }
            std::fprintf(stderr, "xolotl-inkling: %d tensors from XOLOTL_INKLING_OVERRIDE (%lld in that file)\n", n_ov,
                         (long long) gguf_get_n_tensors(ov));
            std::fclose(fo);
            gguf_free(ov);
        }
        if (m.host_w)
            for (ggml_tensor* t : m.gpu_w) m.host_w_at.push_back(at_of[gguf_find_tensor(m.gg.g, t->name)]);
    }
    if (m.ectx) {   // --vram-budget-mib: fill the hot (VRAM) and cold (pinned RAM) arenas expert by expert, write the
                    // tables, zero slot 0 of every arena (the "not here" destination), then READ BACK a sample and
                    // compare it to the file bytes - a slot table that is right about indices and wrong about bytes
                    // produces a plausible token, which is the failure this project keeps paying for
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) die("cannot open " + path);
        const size_t base = gguf_get_data_offset(m.gg.g);
        std::vector<uint8_t> bounce, zeros;
        int64_t n_checked = 0, n_bad = 0;
        const int n_verify = std::getenv("XOLOTL_INKLING_VERIFY") ? std::atoi(std::getenv("XOLOTL_INKLING_VERIFY")) : 8;
        for (int l = 0; l < h.n_phys; ++l) {
            Layer& y = m.L[l];
            for (int which = 0; which < 2; ++which) {
                ggml_tensor* src = which == 0 ? y.gate_up_exps : y.down_exps;
                ggml_tensor* hot = which == 0 ? y.gu_hot : y.dn_hot;
                ggml_tensor* cold = which == 0 ? y.gu_cold : y.dn_cold;
                const size_t blob = ggml_nbytes(src) / h.n_expert;
                const int64_t ti = gguf_find_tensor(m.gg.g, src->name);
                const size_t t_off = gguf_get_tensor_offset(m.gg.g, ti);
                // per-layer blob sizes differ (NVFP4/Q2_0/IQ3_*/IQ4_*/Q8_0 in this file), so size the zero filler
                // to THIS tensor, not once
                if (zeros.size() != blob) zeros.assign(blob, 0);
                for (int e = 0; e < h.n_expert; ++e) {
                    const off_t at = (off_t) (base + t_off + (size_t) e * blob);
                    if (fseeko(f, at, SEEK_SET) != 0) die(std::string("--vram-budget-mib: seek on ") + src->name);
                    const int hs = m.ehot_of[l][e], cs = m.ecold_of[l][e];
                    if (hs > 0 && hot) {
                        bounce.resize(blob);
                        if (std::fread(bounce.data(), 1, blob, f) != blob) die(std::string("--vram-budget-mib: short read on ") + src->name);
                        ggml_backend_tensor_set(hot, bounce.data(), (size_t) hs * blob, blob);
                    } else if (cs > 0 && cold) {
                        if (std::fread((uint8_t*) cold->data + (size_t) cs * blob, 1, blob, f) != blob)
                            die(std::string("--vram-budget-mib: short read on ") + src->name);
                    }
                }
                // the zero rows: slot 0 and the top-k positions' own rows above the last slot (n + 1 .. n + k - 1)
                for (ggml_tensor* a : {hot, cold}) {
                    if (!a) continue;
                    const int64_t n = a->ne[2] - h.n_used;
                    ggml_backend_tensor_set(a, zeros.data(), 0, blob);
                    for (int64_t i = 1; i < h.n_used; ++i) ggml_backend_tensor_set(a, zeros.data(), (size_t) (n + i) * blob, blob);
                }
                // readback: the first n_verify experts of the layer, hot or cold, compared byte for byte to the file
                std::vector<uint8_t> got;
                for (int e = 0; e < std::min<int>(n_verify, h.n_expert); ++e) {
                    const int hs = m.ehot_of[l][e], cs = m.ecold_of[l][e];
                    bounce.resize(blob);
                    if (fseeko(f, (off_t) (base + t_off + (size_t) e * blob), SEEK_SET) != 0 ||
                        std::fread(bounce.data(), 1, blob, f) != blob) die(std::string("--vram-budget-mib: readback seek on ") + src->name);
                    got.resize(blob);
                    if (hs > 0 && hot) ggml_backend_tensor_get(hot, got.data(), (size_t) hs * blob, blob);
                    else if (cs > 0 && cold) std::memcpy(got.data(), (const uint8_t*) cold->data + (size_t) cs * blob, blob);
                    else { ++n_bad; continue; }
                    ++n_checked;
                    if (std::memcmp(got.data(), bounce.data(), blob) != 0) {
                        size_t first = 0;
                        while (first < blob && got[first] == bounce[first]) ++first;
                        ++n_bad;
                        std::fprintf(stderr, "xolotl-inkling: VERIFY FAIL blk.%d %s expert %d (slot %d): first diff at byte %zu of %zu\n",
                                     l, which == 0 ? "gate_up" : "down", e, hs > 0 ? hs : cs, first, blob);
                    }
                }
            }
            // tables: expert -> slot (+1) or 0
            std::vector<int32_t> htab(h.n_expert, 0), ctab(h.n_expert, 0);
            for (int e = 0; e < h.n_expert; ++e) {
                if (m.ehot_of[l][e] > 0) htab[e] = m.ehot_of[l][e];
                if (m.ecold_of[l][e] > 0) ctab[e] = m.ecold_of[l][e];
            }
            ggml_backend_tensor_set(y.hot_tab, htab.data(), 0, (size_t) h.n_expert * sizeof(int32_t));
            if (y.cold_tab) ggml_backend_tensor_set(y.cold_tab, ctab.data(), 0, (size_t) h.n_expert * sizeof(int32_t));
            for (int which = 0; which < 2; ++which) {   // hot_tab2 / cold_tab2: a zero row per top-k position
                ggml_tensor* t2 = which == 0 ? y.hot_tab2 : y.cold_tab2;
                ggml_tensor* a = which == 0 ? y.gu_hot : y.gu_cold;
                if (!t2 || !a) continue;
                const std::vector<int32_t>& tab = which == 0 ? htab : ctab;
                const int64_t K = h.n_used, n = a->ne[2] - K;
                std::vector<int32_t> t2v((size_t) K * h.n_expert);
                for (int e = 0; e < h.n_expert; ++e)
                    for (int64_t i = 0; i < K; ++i)
                        t2v[(size_t) e * K + i] = tab[e] > 0 ? tab[e] : (i == 0 ? 0 : (int32_t) (n + i));
                ggml_backend_tensor_set(t2, t2v.data(), 0, t2v.size() * sizeof(int32_t));
            }
            if (std::getenv("XOLOTL_INKLING_SELFTEST") && (l == 0 || l == h.loop_start)) arena_selftest(m, f, base, l);
        }
        std::fclose(f);
        if (n_bad) die("--vram-budget-mib: " + std::to_string(n_bad) + " of " + std::to_string(n_checked) +
                       " expert slots do not hold the bytes they claim");
        std::fprintf(stderr, "xolotl-inkling: arena verify: %lld expert blobs read back, all match the file\n", (long long) n_checked);
    }
    build_stacked_controllers(m);
    build_router_bf16(m);
    build_padded_rel_bias(m);
    m.pattn_ok = h.hd == 128 && h.n_rel == 16 && h.n_head == 3 * h.n_head_kv;
    for (int l = 0; l < h.n_phys; ++l) m.pattn_ok = m.pattn_ok && m.L[l].rel_bias_pad != nullptr;
    build_stacked_proj(m);

    // state
    m.n_ctx = ((m.n_ctx + 255) / 256) * 256;   // first: positions run to the rounded n_ctx - 1, so a ring clamped to n_ctx never wraps
    if (m.ubatch_auto) {
        const size_t freeb = (size_t) vram_free_mib() << 20;
        int n_full = 0, n_sw = 0;
        for (int sl = 0; sl < h.n_slot; ++sl) (h.swa[h.phys_of(sl)] ? n_sw : n_full)++;
        const size_t kfull = ggml_row_size(m.kv_k, h.n_kgqa()) + ggml_row_size(m.kv_v, h.n_kgqa());
        const size_t ksw = 2 * ggml_row_size(GGML_TYPE_F16, h.n_kgqa());
        int pick = 256;
        for (int cand : {2048, 1024, 512}) {
            if (cand > 512 && !prompt_attn_on()) continue;   // the chain's score tensors halve big chunks anyway
            const int64_t ring_c = std::min<int64_t>(m.n_ctx, (((int64_t) h.n_swa + cand + 255) / 256) * 256);
            const size_t need = (size_t) n_full * (size_t) m.n_ctx * kfull + (size_t) n_sw * (size_t) ring_c * ksw +
                                (size_t) cand * (768u << 10) + m.extra_vram + (384ull << 20);   // t16: parent 16K at 512 keeps 656 MiB after an 8K prompt
            if (freeb >= need) { pick = cand; break; }
        }
        m.ubatch = pick;
        std::fprintf(stderr, "xolotl-inkling: auto ubatch %d (%zu MiB free after the weights)\n", pick, freeb >> 20);
    }
    m.ring = std::min<int64_t>(m.n_ctx, (((int64_t) h.n_swa + m.ubatch + 255) / 256) * 256);
    m.ring_pos.assign(m.ring, -1);
    ggml_init_params sp{(size_t) (4 * h.n_slot + 4) * ggml_tensor_overhead(), nullptr, true};
    m.sctx = ggml_init(sp);
    m.zsink = ggml_new_tensor_1d(m.sctx, GGML_TYPE_F32, h.n_head);
    if (h.hd == 128 && (kv_rot_leg(m.kv_k) || kv_rot_leg(m.kv_v))) {
        m.hrot_f = ggml_new_tensor_2d(m.sctx, GGML_TYPE_F32, h.hd, h.hd);
        m.hrot_i = ggml_new_tensor_2d(m.sctx, GGML_TYPE_F32, h.hd, h.hd);
        ggml_format_name(m.hrot_f, "kv_hrot"); ggml_format_name(m.hrot_i, "kv_hrot_t");
    }
    m.S.resize(h.n_slot);
    for (int s = 0; s < h.n_slot; ++s) {
        Slot& z = m.S[s];
        z.phys = h.phys_of(s);
        z.swa = h.swa[z.phys];
        z.cells = z.swa ? m.ring : m.n_ctx;
        // the sliding rings are small (window + ubatch cells) and read every token: keep them f16, fast and exact;
        // the full-context slots carry the long KV and take --kv-type.
        z.k_type = z.swa ? GGML_TYPE_F16 : m.kv_k;
        z.v_type = z.swa ? GGML_TYPE_F16 : m.kv_v;
        z.k_cache = ggml_new_tensor_3d(m.sctx, z.k_type, h.hd, h.n_head_kv, z.cells);
        z.v_cache = ggml_new_tensor_3d(m.sctx, z.v_type, h.hd, h.n_head_kv, z.cells);
        z.conv = ggml_new_tensor_2d(m.sctx, GGML_TYPE_F32, h.n_conv - 1, h.n_chan());
        ggml_format_name(z.k_cache, "k_cache.%d", s);
        ggml_format_name(z.v_cache, "v_cache.%d", s);
        ggml_format_name(z.conv, "conv.%d", s);
        if (m.spec) {
            z.conv_hist = ggml_new_tensor_2d(m.sctx, GGML_TYPE_F32, h.n_conv - 1 + kSpecT, h.n_chan());
            ggml_format_name(z.conv_hist, "conv_hist.%d", s);
        }
    }
    m.sbuf = ggml_backend_alloc_ctx_tensors(m.sctx, m.gpu);
    if (!m.sbuf) die("cannot allocate the K/V cache in VRAM (try a smaller --context, or --cpu-exps)");
    ggml_backend_buffer_clear(m.sbuf, 0);
    if (m.hrot_f) {   // M = H diag(s) / sqrt(128): Sylvester H, fixed pseudo-random signs (the same every run)
        const int n = (int) h.hd;
        std::vector<float> sg(n), f((size_t) n * n), t((size_t) n * n);
        uint64_t st = 0x9E3779B97F4A7C15ull;
        for (int k = 0; k < n; ++k) { st = st * 6364136223846793005ull + 1442695040888963407ull; sg[k] = (st >> 63) ? -1.f : 1.f; }
        const float sc = 1.0f / std::sqrt((float) n);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k) {
                const float v = (__builtin_popcount((unsigned) (i & k)) & 1 ? -sc : sc) * sg[k];   // M[i][k]
                f[(size_t) i * n + k] = v;   // mul_mat(hrot_f, x)_i = sum_k M[i][k] x_k
                t[(size_t) k * n + i] = v;   // mul_mat(hrot_i, x)_i = sum_k M[k][i] x_k
            }
        ggml_backend_tensor_set(m.hrot_f, f.data(), 0, f.size() * sizeof(float));
        ggml_backend_tensor_set(m.hrot_i, t.data(), 0, t.size() * sizeof(float));
        std::fprintf(stderr, "xolotl-inkling: Hadamard rotation on the full-context %s%s%s legs (XOLOTL_INKLING_KV_ROT=0 turns it off)\n",
                     kv_rot_leg(m.kv_k) ? "K" : "", kv_rot_leg(m.kv_k) && kv_rot_leg(m.kv_v) ? " + " : "", kv_rot_leg(m.kv_v) ? "V" : "");
    }
    m.state_bytes = ggml_backend_buffer_get_size(m.sbuf);

    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, kGraphNodes, false, true);
    m.graph_buf.resize(ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false));
}

// ------------------------------------------------------------------------------------------------------- the graph
// one cell chunk of the attention window: the mask and the relative-distance index of the cells [j0, j0+cells), host-filled
// XOLOTL_INKLING_FA=1 (needs a ggml with per-head flash-attention masks, llama.cpp #32217): short chunks (decode and
// speculative verify, T <= 16) read the whole K/V window in place with ggml_flash_attn_ext, the relative bias folded
// into a per-head f16 mask, instead of transposing/dequantising V every step. Prefill keeps the chunked path (its
// per-head mask would be [cells, T, heads]).
// XOLOTL_INKLING_WSUM_REF=1: the MoE weighted sum as the batched f32 mat-vec (the reference) instead of inkling_moe_wsum
inline bool wsum_ref() { static const bool on = std::getenv("XOLOTL_INKLING_WSUM_REF") != nullptr; return on; }
// prompt chunks through inkling_attn_prompt: default on (22 Nov S9: 8K prefill GPU 3.68 -> 1.40 s, 32K wall ~27 -> 6.1 s on a
// 4090, prefill-path NLL vs the parent gated); XOLOTL_INKLING_PROMPT_ATTN=0 = the chain
inline bool prompt_attn_on() { static const bool on = !(std::getenv("XOLOTL_INKLING_PROMPT_ATTN") && std::atoi(std::getenv("XOLOTL_INKLING_PROMPT_ATTN")) == 0); return on; }
struct Model;
bool kv_fused_type(ggml_type t);
bool pattn_use(const Model& m, int T);   // prompt_attn_on() and this model/chunk can take it (one predicate for build() and the layer)
inline bool fa_on() { static const bool on = std::getenv("XOLOTL_INKLING_FA") && std::atoi(std::getenv("XOLOTL_INKLING_FA")) != 0; return on; }
struct Model;
bool fa_ok(const Model& m, int T);
bool decattn_on();
bool dec_kernel_use(const Model& m, int T);

struct ChunkIn {
    ggml_tensor* mask = nullptr;   // [cells, T] F32 (0 / -inf, the causal or window test)
    ggml_tensor* rp = nullptr;     // [cells, T] I32 (the bias row: relative distance, clamped to the extent)
    int64_t j0 = 0, cells = 0;
};

struct Step {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *tokens = nullptr, *posf = nullptr;
    ggml_tensor *media = nullptr, *media_sel = nullptr;   // GENI: encoder rows [n_embd, T] and where they go [1, T]
    std::vector<ChunkIn> ck_full, ck_swa;   // the visible window in cell chunks (one entry when it fits the budget)
    bool pos_full = false;   // the global slots' decode attention takes the positions (idx_full): ck_full has no mask/rp
    ggml_tensor *out_ids = nullptr, *logits = nullptr;
    ggml_tensor *idx_full = nullptr, *idx_swa = nullptr;   // the cells this step writes (set_rows: fixed pointers -> CUDA graphs)
    std::vector<ggml_tensor*> taps;   // --dump: l_out of every slot
    std::vector<ggml_tensor*> st_taps;   // --stats: [slot][ST_NKIND] the tensors to accumulate
    std::vector<ggml_tensor*> dtaps;     // --dspark: l_out of slots 4, 28, 46 (visits 4, 22, 40) + result_norm, [n_embd, T]
    std::vector<ggml_tensor*> rt_ids, rt_w;   // --route-log: the top-k ids (strided argsort view) and w, per slot
    // --vram-budget-mib: tensors to force onto the CUDA backend.  cold_w = the pinned-host weight leaves,
    // cold_ops = the mmid and (staged mode) its copy nodes, cold_stage = the graph-allocated VRAM staging tensors.
    std::vector<ggml_tensor*> cold_ops, cold_w, cold_stage;
    bool hist = false;                   // --dspark verify chunk: every convolve() also saves [state ; inputs]
    int64_t n_kv_full = 0, n_kv_swa = 0;
};

// fused decode ops (inkling_fused.cu, needs the ggml CUDA custom-op hook): on by default when built in;
// XOLOTL_INKLING_FUSED=0 keeps the reference op chains for A/B
// --train (inkling_train.cpp): set once at start; every fused (custom, non-differentiable) op stays off
inline bool& inkling_train_mode() { static bool v = false; return v; }

static bool fused_on() {
    static const bool on = inkling_fused_available() && !(std::getenv("XOLOTL_INKLING_FUSED") && std::string(std::getenv("XOLOTL_INKLING_FUSED")) == "0");
    return on && !inkling_train_mode();
}

struct G {                     // graph-building context
    Model& m;
    const Hp& h;
    ggml_context* c;
    ggml_cgraph* gf;
    Step& s;
    int T;
    int64_t p0;
    ggml_tensor* tau = nullptr;
    inkling_stats::Stats* st = nullptr;   // --stats: remember these tensors as graph outputs
    bool route = false;                   // --route-log: keep the routing tensors readable after the compute
    int il_cur = 0;

    // --train (inkling_train.cpp): the same op sequence over one chunk, with no cache or state writes.  The slot's conv
    // history and the K/V before the chunk come in as inputs (zeros / none when the chunk starts at position 0).
    using Lora = InkLora;
    bool train = false;
    ggml_tensor* tconv = nullptr;                        // [n_conv-1, n_chan], the slot's conv history
    ggml_tensor *tk_pre = nullptr, *tv_pre = nullptr;    // [hd, nkv, n_pre] carried K (clipped) / V (normed), or null
    ggml_tensor *tpad = nullptr;   // --train: [hd, nkv, n] zero cells after the window (masked), so the cell count is the engine's 256-multiple
    const std::unordered_map<const ggml_tensor*, Lora>* lora = nullptr;
    float lora_scale = 1.0f;
    ggml_tensor* fixed_ids = nullptr;    // train gates: replay these top-k ids [k, T] (frozen routing = a smooth loss)
    ggml_tensor** ids_out = nullptr;     // train gates: record the top-k ids this visit chose
    // a projection by a base weight, plus its adapter when one is attached (inference: lora == null, plain mul_mat)
    ggml_tensor* mm(ggml_tensor* W, ggml_tensor* x) {
        ggml_tensor* y = ggml_mul_mat(c, W, x);
        if (lora) {
            auto it = lora->find(W);
            if (it != lora->end()) y = ggml_add(c, y, ggml_scale(c, ggml_mul_mat(c, it->second.b, ggml_mul_mat(c, it->second.a, x)), lora_scale));
        }
        return y;
    }

    // --stats: keep a tensor alive past the graph so the host can read it back after the compute
    void tap(ggml_tensor* t, int kind) {
        ggml_set_output(t);
        s.st_taps[(size_t) il_cur * inkling_stats::ST_NKIND + kind] = t;
    }

    ggml_tensor* rms(ggml_tensor* x, ggml_tensor* w) { return ggml_mul(c, ggml_rms_norm(c, x, h.eps), w); }

    // scale each row to an L2 norm of at most thr, after the optional per-channel scale
    // == thr * x / max(|x|, thr) = thr * l2_norm(x, eps = thr): two kernels instead of nine (the reference adds 1e-6
    // to |x|, a rounding-level difference); XOLOTL_INKLING_CLIP_REF=1 keeps the reference op chain for A/B
    ggml_tensor* clip(ggml_tensor* x, ggml_tensor* scale, float thr) {
        if (scale) x = ggml_mul(c, x, scale);
        static const bool ref = std::getenv("XOLOTL_INKLING_CLIP_REF") != nullptr;
        if (!ref && thr > 0.0f) return ggml_scale(c, ggml_l2_norm(c, x, thr), thr);
        ggml_tensor* n = ggml_sqrt(c, ggml_sum_rows(c, ggml_sqr(c, x)));
        n = ggml_scale_bias(c, n, 1.0f, 1e-6f);
        ggml_tensor* fac = ggml_div(c, ggml_fill(c, n, thr), n);
        return ggml_mul(c, x, ggml_clamp(c, fac, 0.0f, 1.0f));
    }

    // attention over the depth: earlier streams + cur, normed, softmax-weighted by learned queries -> [n_embd, nq, T]
    ggml_tensor* mix(const std::vector<ggml_tensor*>& blocks, ggml_tensor* cur, ggml_tensor* query) {
        const int64_t E = h.n_embd;
        static const bool mix_fused = std::getenv("XOLOTL_INKLING_MIX_REF") == nullptr;   // A/B: =1 keeps the op chain
        if (fused_on() && mix_fused) {   // the whole depth attention as one kernel
            std::vector<ggml_tensor*> v(blocks.rbegin(), blocks.rend());
            v.push_back(cur);
            if (ggml_tensor* r = inkling_mix(c, v.data(), (int) v.size(), query)) return r;
        }
        ggml_tensor* st = nullptr;
        if (fused_on() && !blocks.empty() && blocks.size() + 1 <= GGML_MAX_SRC) {   // [b_n-1 .. b_0, cur] in one copy
            std::vector<ggml_tensor*> v(blocks.rbegin(), blocks.rend());
            v.push_back(cur);
            st = inkling_stack(c, v.data(), (int) v.size());
        }
        if (!st) {
            st = ggml_reshape_3d(c, cur, E, 1, T);
            for (ggml_tensor* b : blocks) st = ggml_concat(c, ggml_reshape_3d(c, b, E, 1, T), st, 1);
        }
        st = ggml_rms_norm(c, st, 1e-6f);
        const int64_t nq = ggml_n_dims(query) > 1 ? query->ne[1] : 1;
        ggml_tensor* q = T == 1 && ggml_is_contiguous(query) ? ggml_reshape_3d(c, query, E, nq, 1)
                                                              : ggml_repeat(c, query, ggml_new_tensor_3d(c, GGML_TYPE_F32, E, nq, T));
        ggml_tensor* sc = ggml_mul_mat(c, st, q);                                  // [nS, nq, T]
        sc = ggml_soft_max_ext(c, sc, nullptr, 1.0f / std::sqrt((float) E), 0.0f);
        return ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, st)), sc);
    }

    // causal depthwise conv (k = n_conv) with a residual; the last n_conv-1 inputs live in the slot's conv state at
    // channel offset `off`; dyn (global K/V) adds input-dependent taps per head
    ggml_tensor* convolve(ggml_tensor* x, ggml_tensor* w, ggml_tensor* dyn, ggml_tensor* state, int64_t off) {
        const int64_t K = h.n_conv, C = x->ne[0];
        // --dspark verify (s.hist): the fused kernel also writes [state ; inputs] into conv_hist (it used to drop the
        // whole verify graph to the op chain: ~192 unfused convolutions per verify forward).  XOLOTL_INKLING_HIST_REF=1:
        // the chain, as before.
        static const bool hist_ref = std::getenv("XOLOTL_INKLING_HIST_REF") != nullptr;
        ggml_tensor* hview = nullptr;
        if (s.hist && !hist_ref) {
            for (Slot& z : m.S) if (z.conv == state) { if (z.conv_hist) hview = ggml_view_2d(c, z.conv_hist, K - 1 + T, C, z.conv_hist->nb[1], (size_t) off * z.conv_hist->nb[1]); break; }
        }
        if (fused_on() && (!s.hist || hview) && K <= 8 && ggml_is_contiguous(x) && x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(w) && (!dyn || (ggml_is_contiguous(dyn) && dyn->ne[0] % K == 0 && C % (dyn->ne[0] / K) == 0))) {
            ggml_tensor* st = ggml_view_2d(c, state, K - 1, C, state->nb[1], (size_t) off * state->nb[1]);
            const int nh = dyn ? (int) (dyn->ne[0] / K) : 0;
            ggml_tensor* y = inkling_conv_step(c, x, w, st, dyn, (int) K, nh, 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * K + 1)), hview);
            if (hview) ggml_build_forward_expand(gf, y);
            return y;
        }
        ggml_tensor* past = ggml_view_3d(c, state, K - 1, C, 1, state->nb[1], state->nb[1] * C, (size_t) off * state->nb[1]);
        ggml_tensor* xt = T == 1 && ggml_is_contiguous(x) ? ggml_reshape_3d(c, x, 1, C, 1)   // one token: same memory
                                                          : ggml_transpose(c, ggml_reshape_3d(c, x, C, T, 1));
        ggml_tensor* X = ggml_concat(c, past, xt, 0);                                     // [K-1+T, C, 1]
        if (!train) {   // the last K-1 inputs back into the state
            ggml_tensor* src = ggml_view_2d(c, X, K - 1, C, X->nb[1], (size_t) T * X->nb[0]);
            ggml_tensor* dst = ggml_view_2d(c, state, K - 1, C, state->nb[1], (size_t) off * state->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(c, src, dst));
        }
        if (s.hist) {   // --dspark: keep [state ; chunk inputs] so a rejected draft tail can be rolled back exactly
            ggml_tensor* hist = nullptr;
            for (Slot& z : m.S) if (z.conv == state) { hist = z.conv_hist; break; }
            if (hist) {
                ggml_tensor* dst = ggml_view_2d(c, hist, K - 1 + T, C, hist->nb[1], (size_t) off * hist->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, X, K - 1 + T, C), dst));
            }
        }
        ggml_tensor* out = ggml_reshape_2d(c, ggml_ssm_conv(c, X, w), C, T);
        if (dyn) {
            const int64_t nh = dyn->ne[0] / K;
            dyn = ggml_reshape_3d(c, dyn, K, nh, T);
            for (int64_t lag = 0; lag < K; ++lag) {
                ggml_tensor* tap = ggml_view_3d(c, dyn, 1, nh, T, dyn->nb[1], dyn->nb[2], (size_t) lag * dyn->nb[0]);
                ggml_tensor* win = ggml_view_3d(c, X, T, C, 1, X->nb[1], X->nb[2], (size_t) (K - 1 - lag) * X->nb[0]);
                win = ggml_reshape_3d(c, ggml_cont(c, ggml_transpose(c, win)), C / nh, nh, T);
                out = ggml_add(c, out, ggml_reshape_2d(c, ggml_mul(c, win, tap), C, T));
            }
        }
        return ggml_scale(c, ggml_add(c, x, out), 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * K + 1)));
    }

    // (1 - forget) * cur + gain * msg, both gated on rms(cur) ++ rms(msg)
    ggml_tensor* control(ggml_tensor* cur, ggml_tensor* msg, const Layer& y, int k) {
        ggml_tensor* g = ggml_concat(c, ggml_rms_norm(c, cur, 1e-6f), ggml_rms_norm(c, msg, 1e-6f), 0);
        g = ggml_mul_mat(c, y.ctrl_down[k], g);
        ggml_tensor* gain = ggml_tanh(c, ggml_mul_mat(c, y.ctrl_gain[k], g));
        ggml_tensor* forget = ggml_tanh(c, ggml_mul_mat(c, y.ctrl_forget[k], g));
        return ggml_add(c, ggml_sub(c, cur, ggml_mul(c, forget, cur)), ggml_mul(c, gain, msg));
    }

    ggml_tensor* store_kv(ggml_tensor* src, ggml_tensor* dst, bool swa) {   // returns the store (a consumer that reads
        ggml_tensor* d2 = ggml_reshape_2d(c, dst, dst->ne[0] * dst->ne[1], dst->ne[2]);   // the cache in place lists it as a src)
        ggml_tensor* s2 = ggml_reshape_2d(c, src, src->ne[0] * src->ne[1], T);
        ggml_tensor* st = ggml_set_rows(c, d2, s2, swa ? s.idx_swa : s.idx_full);
        ggml_build_forward_expand(gf, st);
        return st;
    }

    ggml_tensor *moe_d = nullptr, *moe_wk = nullptr;   // moe() -> layer(): the fused finish's inputs
    ggml_tensor *moe_ds = nullptr, *moe_ids = nullptr;  // + the per-expert down scale and the ids it is gathered by
    bool want_finish = false;
    ggml_tensor* moe(ggml_tensor* x, const Layer& y) {
        const int64_t E = h.n_embd, NE = h.n_expert, k = h.n_used;
        moe_d = moe_wk = moe_ds = moe_ids = nullptr;
        // [NE, T]; decode / verify read the BF16 copy (same values, half the bytes)
        ggml_tensor* logits = ggml_mul_mat(c, T <= 16 && y.router_h && !st ? y.router_h : y.gate_inp, x);
        ggml_tensor* probs = ggml_sigmoid(c, logits);
        ggml_tensor* sel = y.probs_b ? ggml_add(c, probs, y.probs_b) : probs;
        ggml_tensor* ids = ggml_argsort_top_k(c, sel, (int) k);                  // [k, T]
        if (train && fixed_ids) ids = fixed_ids;
        if (train && ids_out) *ids_out = ids;
        ggml_tensor* w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, NE, T), ids);   // [1, k, T]
        w = ggml_reshape_2d(c, w, k, T);
        // clamp (not + eps): the exact op sequence ggml-cuda fuses into one top-k MoE router kernel (llama.cpp's
        // build_moe_ffn); a sum of 8 sigmoids is never near 6e-5, so the result is unchanged
        static const bool router_ref = std::getenv("XOLOTL_INKLING_ROUTER_REF") != nullptr;   // A/B: the unfused form
        w = router_ref ? ggml_div(c, w, ggml_scale_bias(c, ggml_sum_rows(c, w), 1.0f, 1e-20f))
                       : ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        w = ggml_reshape_3d(c, w, 1, k, T);
        if (h.w_scale != 0.0f && h.w_scale != 1.0f) w = ggml_scale(c, w, h.w_scale);
        ggml_build_forward_expand(gf, w);
        if (route) {   // --route-log: both are in the graph already; output keeps them readable after the compute
            ggml_set_output(ids);
            ggml_set_output(w);
            s.rt_ids[il_cur] = ids;
            s.rt_w[il_cur] = w;
        }
        auto exp_scale = [&](ggml_tensor* r, ggml_tensor* sc) {                   // per-expert scale of NVFP4-kept experts
            if (!sc) return r;
            // one token: the [1, NE, 1] scale is already the right shape (the repeat was a copy kernel per call)
            ggml_tensor* s = T == 1 ? ggml_reshape_3d(c, sc, 1, NE, 1) : ggml_repeat_4d(c, ggml_reshape_3d(c, sc, 1, NE, 1), 1, NE, T, 1);
            return ggml_mul(c, r, ggml_get_rows(c, s, ids));
        };
        // --vram-budget-mib: expert id -> arena slot (+1; 0 = the zero expert).  get_rows wants ne0 = the row width,
        // so the table is read through a [1, NE] view and the ids flattened ([k*T, 1]); ids is a strided view over
        // the full argsort when T > 1, hence the cont.
        // expert id -> arena slot, with a distinct zero row per top-k position for experts not in this arena:
        // rows of tab2 [k, NE] gathered per (token, position) -> [k, k*T], then the diagonal (position i, column i)
        auto slot_ids = [&](ggml_tensor* tab) {
            ggml_tensor* tab2 = tab == y.hot_tab ? y.hot_tab2 : y.cold_tab2;
            ggml_tensor* ic = ggml_cont(c, ids);
            ggml_tensor* iflat = ggml_reshape_1d(c, ic, k * T);
            ggml_tensor* g = ggml_get_rows(c, tab2, iflat);   // [k, k*T]
            ggml_tensor* d = ggml_view_3d(c, g, 1, k, T, (size_t) (k + 1) * sizeof(int32_t), (size_t) k * k * sizeof(int32_t), 0);
            return ggml_reshape_2d(c, ggml_cont(c, d), k, T);
        };
        // --vram-budget-mib, the cold arena and the two kernel regimes.  mul_mat_id uses MMVQ (reads each expert's
        // weights in place) only up to a per-type batch cap (ggml-cuda/mmvq.cu:177); above it, grouped MMQ runs,
        // and MMQ stages weight blocks through shared memory with async copies - it cannot read a pinned-host
        // pointer (measured: illegal access in launch_mul_mat_q).  So:
        //   T == 1 (decode, verify): zero-copy - the mmid reads the pinned arena over PCIe where it lies.
        //   T >= 2 (prefill): stage that layer's cold arena into a VRAM tensor first.  The staged tensor is a GRAPH
        //     tensor, so ggml-alloc hands the same memory to the next layer once this layer's nodes are done - one
        //     layer's worth of scratch, not the whole cold set.  The copy is a plain cudaMemcpy (UVA pinned source
        //     is device-addressable), and everything is pinned to CUDA so the scheduler inserts no second copy.
        static const char* stage_env = std::getenv("XOLOTL_INKLING_STAGE");
        const std::string stage_mode = stage_env ? stage_env : "";
        const bool zero_copy = stage_mode == "always" ? false : (stage_mode == "never" ? true : T == 1);
        // the tensor the cold mmid must read: the pinned arena itself (zero-copy) or its staged VRAM copy
        auto cold_weights = [&](ggml_tensor* cold) -> ggml_tensor* {
            s.cold_w.push_back(cold);
            if (zero_copy) return cold;
            // a DUP node, not a copy into a new leaf: ggml-alloc never frees leaves (every layer's staging stayed
            // resident: graph buffers of 13-24 GB at 128K), and MMQ refuses a padded src0 that is a view
            ggml_tensor* st = ggml_dup(c, cold);
            s.cold_ops.push_back(st);
            return st;
        };
        // the mmid over a cold weight: recorded so forward() can pin it to CUDA
        static const bool dbg_cold = std::getenv("XOLOTL_INKLING_DBG_COLD") != nullptr;
        auto cold_mmid = [&](ggml_tensor* w, ggml_tensor* a, ggml_tensor* sid) {
            ggml_tensor* r = ggml_mul_mat_id(c, w, a, sid);
            if (dbg_cold) {   // confirms src0 really is what we think: the staged view (view_src set) or the arena
                std::fprintf(stderr, "cold mmid T=%d slot=%d: src0=%s type=%s view_src=%p data=%p\n", T, il_cur,
                             w->name, ggml_type_name(w->type), (void*) w->view_src, w->data);
            }
            s.cold_ops.push_back(r);
            return r;
        };
        ggml_tensor* x3 = ggml_reshape_3d(c, x, E, 1, T);
        ggml_tensor* gu = nullptr;
        ggml_tensor* gu_act = nullptr;   // SiTU with the gate/up scale folded in
        ggml_tensor* hid = nullptr, *cid = nullptr;
        if (y.gu_hot || y.gu_cold) {
            // The split moves expert BYTES; it changes nothing else.  Routing ids, w and the per-expert .scale are
            // the baseline's ops over the baseline's ids - so layers 20-23 (no .scale in this quant) apply none,
            // exactly as the baseline does, and the scale is never re-derived per slot.
            if (y.gu_hot) {
                hid = slot_ids(y.hot_tab);
                gu = ggml_mul_mat_id(c, y.gu_hot, x3, hid);
            }
            if (y.gu_cold) {
                cid = slot_ids(y.cold_tab);
                ggml_tensor* gc = cold_mmid(cold_weights(y.gu_cold), x3, cid);
                gu = gu ? ggml_add(c, gu, gc) : gc;
            }
            gu = exp_scale(gu, y.gate_up_s);
        } else {
            gu = ggml_mul_mat_id(c, y.gate_up_exps, x3, ids);
            // the per-expert gate/up scale goes into the SiTU kernel (gather + mul folded in) when it is fused
            static const bool fold_ref = std::getenv("XOLOTL_INKLING_SCALE_FOLD") && std::atoi(std::getenv("XOLOTL_INKLING_SCALE_FOLD")) == 0;
            ggml_tensor* folded = y.gate_up_s && fused_on() && !fold_ref && !st && ggml_is_contiguous(gu) && gu->type == GGML_TYPE_F32
                                      ? inkling_situ(c, gu, h.situ_beta, h.situ_lb, y.gate_up_s, ids) : nullptr;
            if (folded) gu_act = folded;
            else gu = exp_scale(gu, y.gate_up_s);
        }
        const int64_t F = gu->ne[0] / 2;
        ggml_tensor* act;
        if (gu_act) {
            act = gu_act;
        } else if (fused_on() && ggml_is_contiguous(gu) && gu->type == GGML_TYPE_F32) {
            act = inkling_situ(c, gu, h.situ_beta, h.situ_lb);   // the chain below as one kernel
        } else {
            ggml_tensor* g = ggml_cont(c, ggml_view_3d(c, gu, F, k, T, gu->nb[1], gu->nb[2], 0));
            ggml_tensor* u = ggml_cont(c, ggml_view_3d(c, gu, F, k, T, gu->nb[1], gu->nb[2], (size_t) F * gu->nb[0]));
            // SiTU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
            act = ggml_scale(c, ggml_tanh(c, ggml_scale(c, g, 1.0f / h.situ_beta)), h.situ_beta);
            act = ggml_mul(c, act, ggml_sigmoid(c, g));
            if (h.situ_lb > 0.0f) u = ggml_scale(c, ggml_tanh(c, ggml_scale(c, u, 1.0f / h.situ_lb)), h.situ_lb);
            act = ggml_mul(c, act, u);
        }
        if (st) {   // --stats: the down input per (expert slot, token), the routing, and w after its normalisation
            tap(act, inkling_stats::ST_ACT);
            tap(ids, inkling_stats::ST_IDS);
            tap(w, inkling_stats::ST_W);
        }
        ggml_tensor* d = nullptr;
        if (y.dn_hot || y.dn_cold) {
            if (y.dn_hot) {
                if (!hid) hid = slot_ids(y.hot_tab);
                d = ggml_mul_mat_id(c, y.dn_hot, act, hid);
            }
            if (y.dn_cold) {
                if (!cid) cid = slot_ids(y.cold_tab);
                ggml_tensor* dc = cold_mmid(cold_weights(y.dn_cold), act, cid);
                d = d ? ggml_add(c, d, dc) : dc;
            }
        } else {
            d = ggml_mul_mat_id(c, y.down_exps, act, ids);                               // [E, k, T]
        }
        static const bool views_sum = std::getenv("XOLOTL_INKLING_MOE_VIEWS") != nullptr;    // A/B: the reference's sum
        if (views_sum) {
            d = ggml_mul(c, exp_scale(d, y.down_s), w);
            ggml_tensor* out = nullptr;
            for (int64_t i = 0; i < k; ++i) {
                ggml_tensor* v = ggml_view_2d(c, d, E, T, d->nb[2], (size_t) i * d->nb[1]);
                ggml_build_forward_expand(gf, v);
                out = out ? ggml_add(c, out, v) : v;
            }
            return k == 1 ? ggml_cont(c, out) : out;
        }
        // fused finish (layer() does the weighted sum + ffn post-conv + residual in one kernel): hand d, w and (folded
        // into the kernel) the per-expert down scale over
        static const bool finish_ref = std::getenv("XOLOTL_INKLING_MOE_FINISH_REF") != nullptr;
        static const bool fold_ref2 = std::getenv("XOLOTL_INKLING_SCALE_FOLD") && std::atoi(std::getenv("XOLOTL_INKLING_SCALE_FOLD")) == 0;
        if (want_finish && fused_on() && !finish_ref && !fold_ref2 && !st && y.down_s && d->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(d) && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w)) {
            moe_d = d; moe_wk = w; moe_ds = y.down_s; moe_ids = ids;
            return nullptr;
        }
        // the weighted sum over the k experts as one batched mat-vec; the per-expert down scale folds into the weights
        ggml_tensor* wk = y.down_s ? exp_scale(w, y.down_s) : w;                           // [1, k, T]
        if (want_finish && fused_on() && !finish_ref && d->type == GGML_TYPE_F32 && ggml_is_contiguous(d) &&
            wk->type == GGML_TYPE_F32 && ggml_is_contiguous(wk)) {
            moe_d = d; moe_wk = wk;
            return nullptr;
        }
        if (!wsum_ref() && fused_on()) if (ggml_tensor* ws = inkling_moe_wsum(c, d, wk)) return ws;
        ggml_tensor* dT = ggml_cont(c, ggml_permute(c, d, 1, 0, 2, 3));                    // [k, E, T]
        return ggml_reshape_2d(c, ggml_mul_mat(c, dT, ggml_reshape_3d(c, wk, k, 1, T)), E, T);
    }

    // one layer application: slot il, physical ip; messages (passes 2+) = [n_embd, 4, T]
    ggml_tensor* layer(int il, int ip, ggml_tensor* cur, std::vector<ggml_tensor*>& blocks, ggml_tensor* messages) {
        const Layer& y = m.L[ip];
        Slot& z = m.S[il];
        ggml_tensor* cst = train ? tconv : z.conv;   // the conv history this visit reads (and, inference, writes)
        il_cur = il;
        const bool swa = h.swa[ip];
        const int64_t E = h.n_embd, nh = h.n_head, nkv = h.n_head_kv, hd = h.hd, kgqa = h.n_kgqa();
        auto message = [&](int k) { return ggml_cont(c, ggml_view_2d(c, messages, E, T, messages->nb[2], (size_t) k * messages->nb[1])); };

        ggml_tensor* inpSA = cur;
        ggml_tensor* in[3] = {cur, cur, cur};
        ggml_tensor* mxq = y.attn_res_query ? mix(blocks, cur, y.attn_res_query) : nullptr;
        // the three controllers as one fused op (decode / verify): strided views in, no copies
        static const bool ctrl_fused = !(std::getenv("XOLOTL_INKLING_CTRL_FUSED") && std::atoi(std::getenv("XOLOTL_INKLING_CTRL_FUSED")) == 0);
        ggml_tensor* out3f = nullptr;
        if (messages && y.c3_down && (swa || y.n3) && fused_on() && ctrl_fused && !st && T <= 16) {
            ggml_tensor* cur3 = mxq && !swa ? ggml_view_3d(c, mxq, E, T, 3, mxq->nb[2], mxq->nb[1], 0)
                                            : (mxq ? ggml_view_2d(c, mxq, E, T, mxq->nb[2], 0) : cur);
            ggml_tensor* msg3 = ggml_view_3d(c, messages, E, T, 3, messages->nb[2], messages->nb[1], 0);
            out3f = y.c3_down_h ? inkling_ctrl(c, cur3, msg3, y.c3_down_h, y.c3_up_h, swa ? y.attn_norm : y.n3, h.eps) : nullptr;
            if (out3f) for (int j = 0; j < 3; ++j) in[j] = ggml_view_2d(c, out3f, E, T, out3f->nb[1], (size_t) j * out3f->nb[2]);
        }
        if (out3f) {
        } else if (messages && y.c3_down && (swa || y.n3)) {   // the three controllers as one batched chain: [E, T, 3]
            ggml_tensor* cur3;
            if (mxq && !swa) cur3 = ggml_cont(c, ggml_permute(c, mxq, 0, 2, 1, 3));                       // [E,3,T] -> [E,T,3]
            else {
                ggml_tensor* base = mxq ? ggml_cont(c, ggml_view_2d(c, mxq, E, T, mxq->nb[2], 0)) : cur;
                cur3 = ggml_repeat(c, ggml_reshape_3d(c, base, E, T, 1), ggml_new_tensor_3d(c, GGML_TYPE_F32, E, T, 3));
            }
            ggml_tensor* msg3 = ggml_cont(c, ggml_permute(c, ggml_view_3d(c, messages, E, 3, T, messages->nb[1], messages->nb[2], 0), 0, 2, 1, 3));
            ggml_tensor* g = ggml_concat(c, ggml_rms_norm(c, cur3, 1e-6f), ggml_rms_norm(c, msg3, 1e-6f), 0);   // [2E, T, 3]
            ggml_tensor* gf = ggml_tanh(c, ggml_mul_mat(c, y.c3_gf, ggml_mul_mat(c, y.c3_down, g)));            // [2E, T, 3]
            ggml_tensor* gain = ggml_view_3d(c, gf, E, T, 3, gf->nb[1], gf->nb[2], 0);
            ggml_tensor* forget = ggml_view_3d(c, gf, E, T, 3, gf->nb[1], gf->nb[2], (size_t) E * gf->nb[0]);
            ggml_tensor* out3 = ggml_add(c, ggml_sub(c, cur3, ggml_mul(c, forget, cur3)), ggml_mul(c, gain, msg3));
            out3 = swa ? rms(out3, y.attn_norm) : ggml_mul(c, ggml_rms_norm(c, out3, h.eps), y.n3);
            for (int j = 0; j < 3; ++j) in[j] = ggml_view_2d(c, out3, E, T, out3->nb[1], (size_t) j * out3->nb[2]);
        } else {
            if (mxq)
                for (int j = 0; j < 3; ++j) in[j] = ggml_cont(c, ggml_view_2d(c, mxq, E, T, mxq->nb[2], (size_t) (swa ? 0 : j) * mxq->nb[1]));
            if (messages)
                for (int j = 0; j < 3; ++j) in[j] = control(in[j], message(j), y, j);
            if (swa && messages) for (int j = 0; j < 3; ++j) in[j] = rms(in[j], y.attn_norm);
            else if (swa) { in[0] = rms(in[0], y.attn_norm); in[1] = in[2] = in[0]; }
            else { in[0] = rms(in[0], y.q_inp_norm); in[1] = rms(in[1], y.k_inp_norm); in[2] = rms(in[2], y.v_inp_norm); }
        }
        if (st) {      // --stats: the attention inputs after their norms (on SWA layers all three are one tensor)
            tap(in[0], inkling_stats::ST_Q);
            tap(in[1], inkling_stats::ST_K);
            tap(in[2], inkling_stats::ST_V);
        }

        // one token: q/rel/gate (and k/v on sliding layers fed one input) as one stacked mat-vec, sliced by row views
        ggml_tensor *Q, *R, *Kp = nullptr, *Vp = nullptr, *Gp = nullptr;
        const bool one_in = in[1] == in[0] && in[2] == in[0];
        if (T == 1 && y.pqrg && !lora) {   // --lora: per-weight projections, so each adapter applies
            const bool all = y.pall && one_in;
            ggml_tensor* P = ggml_mul_mat(c, all ? y.pall : y.pqrg, in[0]);
            const int64_t nq = y.wq->ne[1], nr = y.rel->ne[1], ng = y.gate->ne[1];
            auto rows = [&](int64_t r0, int64_t n) { return ggml_view_2d(c, P, n, 1, (size_t) n * P->nb[0], (size_t) r0 * P->nb[0]); };
            Q = rows(0, nq); R = rows(nq, nr); Gp = rows(nq + nr, ng);
            if (all) { Kp = rows(nq + nr + ng, y.wk->ne[1]); Vp = rows(nq + nr + ng + y.wk->ne[1], y.wv->ne[1]); }
        } else {
            Q = mm(y.wq, in[0]);
            R = mm(y.rel, in[0]);
        }
        if (!Kp) Kp = mm(y.wk, in[1]);
        if (!Vp) Vp = mm(y.wv, in[2]);
        // fused attention prepare (K/V conv + key clip + value rms + both cache writes, one kernel per visit);
        // XOLOTL_INKLING_PREP_REF=1 keeps the chain.  The --stats path keeps the chain (it taps nothing here, but
        // stays on the validated graph).
        static const bool prep_ref = std::getenv("XOLOTL_INKLING_PREP_REF") != nullptr;
        bool prepped = false;
        ggml_tensor *Kc = nullptr, *Vc = nullptr;
        // rotated legs: the prep kernel rotates K / V itself (XOLOTL_INKLING_PREP_ROT_REF=1: those layers take the chain)
        const bool rot_k = !train && !swa && m.hrot_f && !st && kv_rot_leg(z.k_cache->type);
        const bool rot_v = !train && !swa && m.hrot_f && !st && kv_rot_leg(z.v_cache->type);
        const bool rot = rot_k || rot_v;
        static const bool prep_rot_ref = std::getenv("XOLOTL_INKLING_PREP_ROT_REF") != nullptr;
        auto hrot = [&](ggml_tensor* x, ggml_tensor* M) {
            ggml_tensor* x2 = ggml_reshape_2d(c, ggml_is_contiguous(x) ? x : ggml_cont(c, x), hd, ggml_nelements(x) / hd);
            return ggml_reshape_3d(c, ggml_mul_mat(c, M, x2), hd, x->ne[1], x->ne[2]);
        };
        if (fused_on() && !prep_ref && !st && T <= 16 && !(rot && prep_rot_ref)) {   // decode + verify (the per-token loop is serial in T)
            ggml_tensor* dk = swa ? nullptr : ggml_mul_mat(c, y.k_conv_proj, in[1]);
            ggml_tensor* dv = swa ? nullptr : ggml_mul_mat(c, y.v_conv_proj, in[2]);
            ggml_tensor* hview = nullptr;
            const bool hist_ok = !s.hist || (z.conv_hist && !std::getenv("XOLOTL_INKLING_HIST_REF"));
            if (s.hist && hist_ok) hview = z.conv_hist;
            const int64_t K = h.n_conv;
            ggml_tensor* pr = hist_ok ? inkling_attn_prep(c, Kp, Vp, y.k_conv, y.v_conv, z.conv, (int) kgqa, dk, dv, (int) K,
                                                          1.0f / std::sqrt((float) ((dk ? 2 : 1) * K + 1)), swa ? s.idx_swa : s.idx_full,
                                                          z.k_cache, z.v_cache, h.qk_clip, h.eps, hview, rot_k ? m.hrot_f : nullptr,
                                                          rot_v ? m.hrot_f : nullptr) : nullptr;
            if (pr) { ggml_build_forward_expand(gf, pr); prepped = true; }
            else {
                Kc = convolve(Kp, y.k_conv, dk, cst, 0);
                Vc = convolve(Vp, y.v_conv, dv, cst, kgqa);
            }
        } else {
            Kc = convolve(Kp, y.k_conv, swa ? nullptr : ggml_mul_mat(c, y.k_conv_proj, in[1]), cst, 0);
            Vc = convolve(Vp, y.v_conv, swa ? nullptr : ggml_mul_mat(c, y.v_conv_proj, in[2]), cst, kgqa);
        }
        {   // clip (+ the global layers' log-position scale) as one kernel each when fused; else the op chains
            static const bool clip_ref = std::getenv("XOLOTL_INKLING_CLIP_REF") != nullptr;
            ggml_tensor* tq = swa ? nullptr : tau;
            ggml_tensor *Qf = nullptr, *Kf = nullptr, *Rf = nullptr;
            static const bool clip_pair = !(std::getenv("XOLOTL_INKLING_CLIP_PAIR") && std::atoi(std::getenv("XOLOTL_INKLING_CLIP_PAIR")) == 0);
            ggml_tensor* qr = fused_on() && !clip_ref && clip_pair
                ? inkling_clip2(c, ggml_reshape_3d(c, Q, hd, nh, T), y.q_clip, h.qk_clip,
                                ggml_reshape_3d(c, R, h.n_rel, nh, T), y.rel_clip, h.rel_clip, tq) : nullptr;
            if (qr) {   // q and rel clips in one launch: views of the one output
                Qf = ggml_view_3d(c, qr, hd, nh, T, hd * 4, hd * nh * 4, 0);
                Rf = ggml_view_3d(c, qr, h.n_rel, nh, T, h.n_rel * 4, h.n_rel * nh * 4, (size_t) hd * nh * T * 4);
                if (!prepped) Kf = inkling_clip(c, ggml_reshape_3d(c, Kc, hd, nkv, T), nullptr, h.qk_clip, nullptr);
            } else if (fused_on() && !clip_ref) {
                Qf = inkling_clip(c, ggml_reshape_3d(c, Q, hd, nh, T), y.q_clip, h.qk_clip, tq);
                if (!prepped) Kf = inkling_clip(c, ggml_reshape_3d(c, Kc, hd, nkv, T), nullptr, h.qk_clip, nullptr);
                Rf = inkling_clip(c, ggml_reshape_3d(c, R, h.n_rel, nh, T), y.rel_clip, h.rel_clip, tq);
            }
            Q = Qf ? Qf : clip(ggml_reshape_3d(c, Q, hd, nh, T), y.q_clip, h.qk_clip);
            if (!prepped) Kc = Kf ? Kf : clip(ggml_reshape_3d(c, Kc, hd, nkv, T), nullptr, h.qk_clip);
            R = Rf ? Rf : clip(ggml_reshape_3d(c, R, h.n_rel, nh, T), y.rel_clip, h.rel_clip);
            if (!swa && !Qf) Q = ggml_mul(c, Q, tau);
            if (!swa && !Rf) R = ggml_mul(c, R, tau);
        }
        if (rot_k) { Q = hrot(Q, m.hrot_f); if (!prepped) Kc = hrot(Kc, m.hrot_f); }
        ggml_tensor *kst = nullptr, *vst = nullptr;   // the cache stores (the prompt attention orders itself after them)
        if (!prepped) {
            Vc = ggml_rms_norm(c, ggml_reshape_3d(c, Vc, hd, nkv, T), h.eps);
            if (rot_v) Vc = hrot(Vc, m.hrot_f);
            if (!train) {
                kst = store_kv(Kc, z.k_cache, swa);
                vst = store_kv(Vc, z.v_cache, swa);
            }
        }
        const std::vector<ChunkIn>& plan = swa ? s.ck_swa : s.ck_full;

        // relative bias: project R on the (zero-padded) bias table, subtract the per-query sink, gather by distance
        ggml_tensor* sinks = ggml_mul_mat(c, y.sink_proj, in[0]);                         // [nh, T]
        ggml_tensor* table_raw = ggml_mul_mat(c, y.rel_bias_pad ? y.rel_bias_pad : ggml_pad(c, y.rel_bias, 0, 1, 0, 0), R);   // [Ext+1, nh, T]
        // the per-head mask (bias + causal/window mask) as one kernel when fused (XOLOTL_INKLING_RELBIAS_REF=1: the chain)
        static const bool relbias_ref = std::getenv("XOLOTL_INKLING_RELBIAS_REF") != nullptr;
        ggml_tensor* table = nullptr;
        auto get_table = [&]() {
            if (!table) {
                table = ggml_sub(c, table_raw, ggml_reshape_3d(c, sinks, 1, nh, T));
                table = ggml_cont(c, ggml_permute(c, table, 1, 0, 2, 3));                  // [nh, Ext+1, T]
            }
            return table;
        };
        auto fused_mask = [&](const ChunkIn& ci, ggml_type ty) -> ggml_tensor* {
            return fused_on() && !relbias_ref ? inkling_relbias(c, table_raw, sinks, ci.rp, ci.mask, ty) : nullptr;
        };
        ggml_tensor* q = ggml_permute(c, Q, 0, 2, 1, 3);                                    // [hd, T, nh]
        // The visible window is read in cell chunks so the score tensor [cells, T, nh] and the per-head mask stay
        // bounded at 512K context. One chunk (short context) is the single-shot softmax the engine always used; more
        // than one runs an online softmax whose initial (max 0, sum 1) is exactly the zero sink logit.
        ggml_tensor *acc = nullptr, *lse = nullptr, *mx = nullptr;
        ggml_tensor* o = nullptr;
        // decode / verify (T <= 16): the split-K attention kernel reads the f16 K/V window in place (no V transpose
        // copy per step).  Default on: quality-neutral vs the chain against the NVFP4 parent (paired NLL / KL-to-parent, 22 Nov S8); XOLOTL_INKLING_DECATTN=0 = the chain
        bool gated = false;   // the output gate already applied inside the attention kernel
        const bool decattn = decattn_on();
        // prompt chunks (T > 16): flash attention on the tensor cores over the whole f16 cache, bias and sink in-kernel
        // (port of Basalt attn_prompt.cu) - no [cells, T, nh] scores, masks or V transposes. XOLOTL_INKLING_PROMPT_ATTN=1
        if (train) {   // --train: the chunk's own K/V (after any carried prefix), host masks, one softmax - the reference's
                       // single-chunk path below with the cache views replaced by the fresh tensors
            const ChunkIn& ci = plan.at(0);
            ggml_tensor* Ka = Kc;   // [hd, nkv, T] (clipped)
            ggml_tensor* Va = Vc;   // [hd, nkv, T] (normed)
            if (tk_pre) { Ka = ggml_concat(c, tk_pre, Ka, 2); Va = ggml_concat(c, tv_pre, Va, 2); }
            // pad to the engine's cell layout (a multiple of 256, masked): a ragged cell count makes the f16 attention
            // products take another kernel / reduction blocking than the engine's, which broke --check-prefix exactness
            if (tpad) { Ka = ggml_concat(c, Ka, tpad, 2); Va = ggml_concat(c, Va, tpad, 2); }
            GGML_ASSERT(Ka->ne[2] == ci.cells && Va->ne[2] == ci.cells);
            // XOLOTL_INKLING_TRAIN_KV16=1 (the --check gate): K/V as f16 tensors, so both attention products run the cache path's precision
            static const bool kv16 = std::getenv("XOLOTL_INKLING_TRAIN_KV16") != nullptr;
            if (kv16) { Ka = ggml_cast(c, Ka, GGML_TYPE_F16); Va = ggml_cast(c, Va, GGML_TYPE_F16); }   // f16 operands, as the cache path
            ggml_tensor* kq_b = ggml_get_rows(c, get_table(), ci.rp);                       // [nh, cells, T]
            kq_b = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, kq_b, nh, ci.cells, T, 1), 2, 0, 1, 3));   // [cells, T, nh]
            ggml_tensor* mask_h = ggml_add(c, kq_b, ci.mask);
            ggml_tensor* Kv = ggml_permute(c, Ka, 0, 2, 1, 3);                                // [hd, cells, nkv]
            ggml_tensor* vt = ggml_cont(c, ggml_transpose(c, ggml_permute(c, Va, 0, 2, 1, 3)));   // [cells, hd, nkv]
            ggml_tensor* kq = ggml_mul_mat(c, Kv, q);                                       // [cells, T, nh]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(c, kq, mask_h, 1.0f / std::sqrt((float) hd), 0.0f);
            ggml_soft_max_add_sinks(kq, m.zsink);
            ggml_tensor* at = ggml_mul_mat(c, vt, kq);                                      // [hd, T, nh]
            static const bool at32 = std::getenv("XOLOTL_INKLING_TRAIN_ATTN_F32") != nullptr;   // f32 accumulation (layout-order test)
            if (at32) ggml_prec_set_acc(at, GGML_PREC_F32);
            o = ggml_cont_2d(c, ggml_permute(c, at, 0, 2, 1, 3), (int64_t) hd * nh, T);
        }
        if (!o && pattn_use(m, (int) T) && kst && vst && y.rel_bias_pad && kv_fused_type(z.k_cache->type) && kv_fused_type(z.v_cache->type)) {
            ggml_tensor* pa = inkling_attn_prompt(c, Q, z.k_cache, z.v_cache, R, sinks, y.rel_bias_pad, kst, vst, p0,
                                                  swa ? h.n_swa : 0, 1.0f / std::sqrt((float) hd));
            if (pa) o = ggml_reshape_2d(c, pa, (int64_t) hd * nh, T);
        }
        if (!o && plan.empty()) GGML_ABORT("xolotl-inkling: prompt attention refused a layer that pattn_use() admitted (a bug; XOLOTL_INKLING_PROMPT_ATTN=0 works around it)");
        if (!o && decattn && !fa_ok(m, (int) T) && T <= 16 && plan.size() == 1 && !m.st &&
            kv_fused_type(z.k_cache->type) && kv_fused_type(z.v_cache->type)) {
            const ChunkIn& ci = plan[0];
            // the relative bias computed inside the attention kernel (no table mat-mul, no [cells, T, nh] mask);
            // XOLOTL_INKLING_DECATTN_REL=0 keeps the relbias mask
            static const bool rel_in = !(std::getenv("XOLOTL_INKLING_DECATTN_REL") && std::atoi(std::getenv("XOLOTL_INKLING_DECATTN_REL")) == 0);
            static const bool gate_ref0 = std::getenv("XOLOTL_INKLING_DECATTN_GATE_REF") != nullptr;
            if (rel_in && y.rel_bias_pad) {
                ggml_tensor* Kd = ggml_view_3d(c, z.k_cache, hd, nkv, ci.cells, z.k_cache->nb[1], z.k_cache->nb[2], (size_t) ci.j0 * z.k_cache->nb[2]);
                ggml_tensor* Vd = ggml_view_3d(c, z.v_cache, hd, nkv, ci.cells, z.v_cache->nb[1], z.v_cache->nb[2], (size_t) ci.j0 * z.v_cache->nb[2]);
                ggml_tensor* Gt = gate_ref0 || rot_v ? nullptr : (Gp ? Gp : mm(y.gate, in[0]));
                ggml_tensor* ad = !swa && s.pos_full
                    ? inkling_attn_dec_rel_pos(c, Q, Kd, Vd, y.rel_bias_pad, R, sinks, s.idx_full, h.rel_extent, 1.0f / std::sqrt((float) hd), Gt)
                    : inkling_attn_dec_rel(c, Q, Kd, Vd, y.rel_bias_pad, R, sinks, ci.rp, ci.mask, 1.0f / std::sqrt((float) hd), Gt);
                if (!ad && !swa && s.pos_full) GGML_ABORT("xolotl-inkling: positional decode attention refused (inkling_dec_pos_ok out of sync)");
                if (ad) { o = ggml_reshape_2d(c, ad, (int64_t) hd * nh, T); gated = Gt != nullptr; }
            }
            ggml_tensor* mask_h = o ? nullptr : fused_mask(ci, GGML_TYPE_F32);
            if (mask_h) {
                ggml_tensor* Kd = ggml_view_3d(c, z.k_cache, hd, nkv, ci.cells, z.k_cache->nb[1], z.k_cache->nb[2], (size_t) ci.j0 * z.k_cache->nb[2]);
                ggml_tensor* Vd = ggml_view_3d(c, z.v_cache, hd, nkv, ci.cells, z.v_cache->nb[1], z.v_cache->nb[2], (size_t) ci.j0 * z.v_cache->nb[2]);
                static const bool gate_ref = std::getenv("XOLOTL_INKLING_DECATTN_GATE_REF") != nullptr;   // A/B: gate as ggml ops
                ggml_tensor* Gt = gate_ref || rot_v ? nullptr : (Gp ? Gp : mm(y.gate, in[0]));
                ggml_tensor* ad = inkling_attn_dec(c, Q, Kd, Vd, mask_h, 1.0f / std::sqrt((float) hd), Gt);
                if (ad) { o = ggml_reshape_2d(c, ad, (int64_t) hd * nh, T); gated = Gt != nullptr; }
            }
        }
        if (!o && fa_ok(m, (int) T) && plan.size() == 1) {
            const ChunkIn& ci = plan[0];
            ggml_tensor* mask_h = fused_mask(ci, GGML_TYPE_F16);
            if (!mask_h) {
                ggml_tensor* kq_b = ggml_get_rows(c, get_table(), ci.rp);                   // [nh, cells, T]
                kq_b = ggml_permute(c, ggml_reshape_4d(c, kq_b, nh, ci.cells, T, 1), 2, 0, 1, 3);   // [cells, T, nh]
                mask_h = ggml_cast(c, ggml_add(c, kq_b, ci.mask), GGML_TYPE_F16);           // per-head f16 mask
            }
            ggml_tensor* Kv = ggml_permute(c, ggml_view_3d(c, z.k_cache, hd, nkv, ci.cells, z.k_cache->nb[1], z.k_cache->nb[2], 0), 0, 2, 1, 3);
            ggml_tensor* Vv = ggml_permute(c, ggml_view_3d(c, z.v_cache, hd, nkv, ci.cells, z.v_cache->nb[1], z.v_cache->nb[2], 0), 0, 2, 1, 3);
            ggml_tensor* fa = ggml_flash_attn_ext(c, q, Kv, Vv, mask_h, 1.0f / std::sqrt((float) hd), 0.0f, 0.0f);
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            ggml_flash_attn_ext_add_sinks(fa, m.zsink);   // zero sink logit
            o = ggml_reshape_2d(c, fa, (int64_t) hd * nh, T);                               // [hd, nh, T] -> rows per token
        }
        for (const ChunkIn& ci : plan) {
            if (o) break;
            ggml_tensor* mask_h = fused_mask(ci, GGML_TYPE_F32);
            if (!mask_h) {
                ggml_tensor* kq_b = ggml_get_rows(c, get_table(), ci.rp);                   // [nh, cells, T]
                kq_b = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, kq_b, nh, ci.cells, T, 1), 2, 0, 1, 3));   // [cells, T, nh]
                mask_h = ggml_add(c, kq_b, ci.mask);
            }
            ggml_tensor* Kv = ggml_permute(c, ggml_view_3d(c, z.k_cache, hd, nkv, ci.cells, z.k_cache->nb[1], z.k_cache->nb[2],
                                                          (size_t) ci.j0 * z.k_cache->nb[2]), 0, 2, 1, 3);
            ggml_tensor* Vv = ggml_permute(c, ggml_view_3d(c, z.v_cache, hd, nkv, ci.cells, z.v_cache->nb[1], z.v_cache->nb[2],
                                                          (size_t) ci.j0 * z.v_cache->nb[2]), 0, 2, 1, 3);
            ggml_tensor* vt;
            if (z.v_type == GGML_TYPE_F16) {
                vt = ggml_cont(c, ggml_transpose(c, Vv));                                   // [cells, hd, nkv]
            } else {   // a quantised cache is dequantised to f32: CUDA's CPY has no quantised -> f16 pair
                ggml_tensor* vf = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, ci.cells, nkv);
                ggml_build_forward_expand(gf, ggml_cpy(c, Vv, vf));
                vt = ggml_cont(c, ggml_transpose(c, vf));
            }
            ggml_tensor* kq = ggml_mul_mat(c, Kv, q);                                       // [cells, T, nh]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            if (plan.size() == 1) {
                kq = ggml_soft_max_ext(c, kq, mask_h, 1.0f / std::sqrt((float) hd), 0.0f);
                ggml_soft_max_add_sinks(kq, m.zsink);   // zero sink logit
                acc = ggml_mul_mat(c, vt, kq);                                              // [hd, T, nh]
                break;
            }
            ggml_tensor* zc = ggml_add(c, ggml_scale(c, kq, 1.0f / std::sqrt((float) hd)), mask_h);
            ggml_tensor* mc = ggml_pool_1d(c, zc, GGML_OP_POOL_MAX, (int) ci.cells, (int) ci.cells, 0);   // [1, T, nh]
            if (!mx) {
                mx = ggml_clamp(c, mc, 0.0f, INFINITY);                                     // the sink logit joins the max
                ggml_tensor* p = ggml_exp(c, ggml_sub(c, zc, mx));
                lse = ggml_add(c, ggml_sum_rows(c, p), ggml_exp(c, ggml_scale(c, mx, -1.0f)));
                acc = ggml_mul_mat(c, vt, p);
            } else {
                ggml_tensor* m_new = ggml_pool_1d(c, ggml_concat(c, mx, mc, 0), GGML_OP_POOL_MAX, 2, 2, 0);   // max(mx, mc)
                ggml_tensor* f = ggml_exp(c, ggml_sub(c, mx, m_new));                        // rescale of the running sums
                ggml_tensor* p = ggml_exp(c, ggml_sub(c, zc, m_new));
                lse = ggml_add(c, ggml_mul(c, lse, f), ggml_sum_rows(c, p));
                acc = ggml_add(c, ggml_mul(c, acc, f), ggml_mul_mat(c, vt, p));
                mx = m_new;
            }
        }
        if (!o) {
            o = lse ? ggml_div(c, acc, lse) : acc;                                             // [hd, T, nh]  (normalise once)
            o = ggml_cont_2d(c, ggml_permute(c, o, 0, 2, 1, 3), (int64_t) hd * nh, T);
        }
        if (rot_v) {   // the values were stored rotated: o = M^T o' per head, before the (elementwise) gate
            GGML_ASSERT(!gated);
            o = ggml_reshape_2d(c, ggml_mul_mat(c, m.hrot_i, ggml_reshape_2d(c, ggml_is_contiguous(o) ? o : ggml_cont(c, o), hd, nh * T)), (int64_t) hd * nh, T);
        }
        if (!gated) o = ggml_mul(c, o, ggml_sigmoid(c, Gp ? Gp : mm(y.gate, in[0])));
        if (st) tap(o, inkling_stats::ST_O);            // --stats: the o_proj input, exactly what wo consumes
        o = mm(y.wo, o);
        o = convolve(o, y.attn_post_conv, nullptr, cst, 2 * kgqa);
        if (y.wo_b) o = ggml_add(c, o, y.wo_b);

        if (ip % h.block == 0) { blocks.push_back(inpSA); cur = o; }     // a new residual stream starts here
        else cur = ggml_add(c, inpSA, o);
        ggml_tensor* ffn_inp = cur;

        ggml_tensor* x = ggml_reshape_2d(c, mix(blocks, cur, y.ffn_res_query), E, T);
        ggml_tensor* xf = nullptr;   // the FFN controller + ffn_norm as one fused op (decode / verify)
        if (messages && y.c1_down_h && fused_on() && !st && T <= 16 &&
            !(std::getenv("XOLOTL_INKLING_CTRL_FUSED") && std::atoi(std::getenv("XOLOTL_INKLING_CTRL_FUSED")) == 0)) {
            ggml_tensor* msg1 = ggml_view_3d(c, messages, E, T, 1, messages->nb[2], messages->nb[1], (size_t) 3 * messages->nb[1]);
            xf = inkling_ctrl(c, x, msg1, y.c1_down_h, y.c1_up_h, y.ffn_norm, h.eps);
            if (xf) xf = ggml_reshape_2d(c, xf, E, T);
        }
        if (xf) x = xf;
        else {
            if (messages) x = control(x, message(3), y, 3);
            x = rms(x, y.ffn_norm);
        }
        if (st) tap(x, inkling_stats::ST_F);            // --stats: the MoE input (gate/up + the router)
        want_finish = T <= 16;   // decode + verify; prefill keeps the sum and gets the token-parallel conv
        x = moe(x, y);
        want_finish = false;
        if (!x) {   // fused MoE finish: weighted sum over k + ffn post-conv + residual (+ the verify conv history)
            const int64_t K = h.n_conv, off = 2 * kgqa + E;
            ggml_tensor* stv = ggml_view_2d(c, z.conv, K - 1, E, z.conv->nb[1], (size_t) off * z.conv->nb[1]);
            ggml_tensor* hview = (s.hist && z.conv_hist) ? ggml_view_2d(c, z.conv_hist, K - 1 + T, E, z.conv_hist->nb[1], (size_t) off * z.conv_hist->nb[1]) : nullptr;
            static const bool hist_ref = std::getenv("XOLOTL_INKLING_HIST_REF") != nullptr;
            ggml_tensor* f = (s.hist && (!hview || hist_ref)) ? nullptr
                           : inkling_moe_finish(c, moe_d, moe_wk, y.ffn_post_conv, stv, ffn_inp, (int) K, 1.0f / std::sqrt((float) (K + 1)), hview,
                                                    moe_ds, moe_ids);
            if (f) {
                if (hview) ggml_build_forward_expand(gf, f);
                cur = f;
            } else {   // the reference weighted sum, then the usual convolve + add
                if (moe_ds) {   // the down scale was left for the fused finish: apply it here (as moe()'s exp_scale)
                    const int64_t NE = h.n_expert;
                    ggml_tensor* s3 = T == 1 ? ggml_reshape_3d(c, moe_ds, 1, NE, 1)
                                             : ggml_repeat_4d(c, ggml_reshape_3d(c, moe_ds, 1, NE, 1), 1, NE, T, 1);
                    moe_wk = ggml_mul(c, moe_wk, ggml_get_rows(c, s3, moe_ids));
                    moe_ds = moe_ids = nullptr;
                }
                x = wsum_ref() ? nullptr : inkling_moe_wsum(c, moe_d, moe_wk);
                if (!x) {
                    ggml_tensor* dT = ggml_cont(c, ggml_permute(c, moe_d, 1, 0, 2, 3));
                    x = ggml_reshape_2d(c, ggml_mul_mat(c, dT, ggml_reshape_3d(c, moe_wk, h.n_used, 1, T)), E, T);
                }
                x = convolve(x, y.ffn_post_conv, nullptr, cst, off);
                cur = ggml_add(c, ffn_inp, x);
            }
        } else {
            x = convolve(x, y.ffn_post_conv, nullptr, cst, 2 * kgqa + E);
            cur = ggml_add(c, ffn_inp, x);
        }
        ggml_format_name(cur, "l_out-%d", il);
        if (!s.taps.empty()) { ggml_set_output(cur); s.taps[il] = cur; }
        if (!s.dtaps.empty() && (il == 4 || il == 28 || il == 46)) { ggml_set_output(cur); s.dtaps[il == 4 ? 0 : il == 28 ? 1 : 2] = cur; }
        return cur;
    }
};

// cells per attention chunk: the graph stays flat (kAttnChunks chunks per slot at most) and one chunk's score
// tensor [cells, T, n_head] stays within --attn-budget elements unless the chunk cap would need more of them.
int64_t attn_cell_chunk(const Model& m, int T, int64_t nk, int nh) {
    if (nk <= 0) return 1;
    const int64_t chunks = attn_chunks();
    int64_t C = std::max<int64_t>(64, (nk + chunks - 1) / chunks);
    // the budget may merge chunks (fewer nodes, one bigger score tensor) but never split below the chunk cap
    if (!std::getenv("XOLOTL_INKLING_ATTN_CHUNKS"))
        C = std::max(C, std::min<int64_t>(nk, m.attn_budget / std::max<int64_t>(1, (int64_t) T * nh)));
    C = std::min(C, nk);
    return std::min(nk, (C + 63) / 64 * 64);   // 64-cell chunks keep the K/V row views block aligned
}

bool fa_ok(const Model& m, int T) { return fa_on() && T <= 16 && !m.st; }
bool kv_fused_type(ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0; }
// the split-K decode attention kernel (decode + verify): XOLOTL_INKLING_DECATTN=0 = the chain
bool decattn_on() {
    static const bool v = !(std::getenv("XOLOTL_INKLING_DECATTN") && std::atoi(std::getenv("XOLOTL_INKLING_DECATTN")) == 0);
    return v;
}
// decode / verify steps the split-K kernel takes: it reads the K/V window in place with no score tensor, so its plan is
// ONE chunk at any context (split by --attn-budget, a 477K-cell window became several chunks, which the kernel refuses,
// and the chain fallback asked for a 17 GB graph buffer)
bool dec_kernel_use(const Model& m, int T) {
    return decattn_on() && fa_ok(m, T) == false && T <= 16 && !m.st && kv_fused_type(m.kv_k) && kv_fused_type(m.kv_v);
}
bool pattn_use(const Model& m, int T) {
    return prompt_attn_on() && T > 16 && !m.st && m.pattn_ok && kv_fused_type(m.kv_k) && kv_fused_type(m.kv_v);
}

// T tokens at positions p0 .. p0+T-1; logits for the last n_out of them
Step build(Model& m, int T, int64_t p0, int n_out, bool dump, bool dspark = false, bool vision = false) {
    const Hp& h = m.hp;
    Step s;
    ggml_init_params ip{m.graph_buf.size(), m.graph_buf.data(), true};
    s.ctx = ggml_init(ip);
    ggml_context* c = s.ctx;
    s.gf = ggml_new_graph_custom(c, kGraphNodes, false);
    if (dump) s.taps.assign(h.n_slot, nullptr);
    if (m.st) s.st_taps.assign((size_t) h.n_slot * inkling_stats::ST_NKIND, nullptr);
    if (m.route_fp) { s.rt_ids.assign(h.n_slot, nullptr); s.rt_w.assign(h.n_slot, nullptr); }
    if (dspark) { s.dtaps.assign(4, nullptr); s.hist = m.spec && T <= kSpecT; }

    s.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.tokens);
    s.posf = ggml_new_tensor_1d(c, GGML_TYPE_F32, T); ggml_set_input(s.posf);
    if (vision) {
        s.media = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, T); ggml_set_input(s.media);
        s.media_sel = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T); ggml_set_input(s.media_sel);
    }
    s.n_kv_full = std::min<int64_t>(m.n_ctx, ((p0 + T + 255) / 256) * 256);
    {
        int64_t hi = -1;
        for (int64_t j = 0; j < m.ring; ++j) if (m.ring_pos[j] >= 0) hi = j;
        for (int t = 0; t < T; ++t) hi = std::max<int64_t>(hi, (p0 + t) % m.ring);
        s.n_kv_swa = std::min<int64_t>(m.ring, ((hi + 1 + 255) / 256) * 256);
    }
    {   // the visible window in cell chunks; one chunk at short context == the old single mask/rel-pos pair
        // positional global slots: the decode kernel derives mask and distance from idx_full, so the plan carries no
        // [cells, T] host arrays for them (3.8 MB of fill + upload a token at 477K)
        static const bool rel_in_env = !(std::getenv("XOLOTL_INKLING_DECATTN_REL") && std::atoi(std::getenv("XOLOTL_INKLING_DECATTN_REL")) == 0);
        s.pos_full = dec_kernel_use(m, T) && rel_in_env && !m.L.empty() && m.L[0].rel_bias_pad &&
                     inkling_dec_pos_ok(h.hd, h.n_head, h.n_head_kv, m.kv_k, m.kv_v, h.n_rel, ggml_row_size(m.kv_k, h.hd),
                                        ggml_row_size(m.kv_k, h.hd) * h.n_head_kv);
        auto plan = [&](int64_t nk, bool positional) {
            std::vector<ChunkIn> v;
            // --stats keeps the single-shot attention it was gated with (its score tensor is a few tens of MB here)
            const int64_t C = (m.st && nk * T * h.n_head * 4 <= (256 << 20)) || fa_ok(m, T) || dec_kernel_use(m, T) ? nk
                                                                                                                  : attn_cell_chunk(m, T, nk, h.n_head);
            for (int64_t j0 = 0; j0 < nk; j0 += C) {
                ChunkIn ci;
                ci.j0 = j0; ci.cells = std::min(C, nk - j0);
                if (!positional) {
                    ci.mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, ci.cells, T); ggml_set_input(ci.mask);
                    ci.rp = ggml_new_tensor_2d(c, GGML_TYPE_I32, ci.cells, T); ggml_set_input(ci.rp);
                }
                v.push_back(ci);
            }
            return v;
        };
        // the prompt attention computes the causal/window test and the distance in-kernel: no host masks
        if (!pattn_use(m, T)) {
            s.ck_full = plan(s.n_kv_full, s.pos_full);
            s.ck_swa = plan(s.n_kv_swa, false);
        }
    }
    if (n_out < T) { s.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_out); ggml_set_input(s.out_ids); }
    s.idx_full = ggml_new_tensor_1d(c, GGML_TYPE_I64, T); ggml_set_input(s.idx_full);
    s.idx_swa = ggml_new_tensor_1d(c, GGML_TYPE_I64, T); ggml_set_input(s.idx_swa);

    G g{m, h, c, s.gf, s, T, p0};
    g.st = m.st;
    if (!m.lora.empty()) { g.lora = &m.lora; g.lora_scale = m.lora_scale; }
    g.route = m.route_fp != nullptr;
    // log-scaled positions for the global layers: 1 + alpha*log(max(1, (pos + 1)/floor))
    ggml_tensor* tau = ggml_scale_bias(c, s.posf, 1.0f, 1.0f);
    tau = ggml_log(c, ggml_clamp(c, ggml_scale(c, tau, 1.0f / (float) h.log_floor), 1.0f, INFINITY));
    g.tau = ggml_reshape_3d(c, ggml_scale_bias(c, tau, h.log_alpha, 1.0f), 1, 1, T);

    ggml_tensor* cur = ggml_get_rows(c, m.tok_embd, s.tokens);
    if (s.media) {
        // text rows: rms_norm * tok_norm; image rows: the projector's output raw (the reference's normed-embedding
        // table IS normed, then the image rows overwrite it, so never normalize them - llama.cpp inkling2.cpp)
        ggml_tensor* normed = g.rms(cur, m.tok_norm);
        ggml_tensor* sel = s.media_sel;                                        // [1, T], broadcasts over n_embd
        cur = ggml_add(c, ggml_mul(c, normed, ggml_sub(c, ggml_fill(c, sel, 1.0f), sel)), ggml_mul(c, s.media, sel));
    } else {
        cur = g.rms(cur, m.tok_norm);
    }
    std::vector<ggml_tensor*> blocks;
    const int start = h.n_pass > 1 ? h.loop_start : h.n_phys, end = h.n_pass > 1 ? h.loop_end : h.n_phys;
    for (int il = 0; il < start; ++il) cur = g.layer(il, il, cur, blocks, nullptr);
    ggml_tensor* prelude = cur;
    const std::vector<ggml_tensor*> prelude_blocks = blocks;
    for (int pass = 0; pass < h.n_pass && start < end; ++pass) {
        std::vector<ggml_tensor*> bank;
        ggml_tensor* bank_out = nullptr;
        if (pass > 0) {   // the previous pass: its streams after the first core stream, and its output
            bank.assign(blocks.begin() + (long) prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ipy = start; ipy < end; ++ipy) {
            const int il = pass == 0 ? ipy : h.n_phys + (pass - 1) * (end - start) + (ipy - start);
            ggml_tensor* msgs = pass == 0 ? nullptr : g.mix(bank, bank_out, m.L[ipy].msg_query);
            cur = g.layer(il, ipy, cur, blocks, msgs);
        }
    }
    for (int il = end; il < h.n_phys; ++il) cur = g.layer(il, il, cur, blocks, nullptr);

    cur = ggml_reshape_2d(c, g.mix(blocks, cur, m.output_res_query), h.n_embd, T);
    cur = g.rms(cur, m.output_norm);
    if (!s.dtaps.empty()) { ggml_set_output(cur); s.dtaps[3] = cur; ggml_build_forward_expand(s.gf, cur); }
    if (s.out_ids) cur = ggml_get_rows(c, cur, s.out_ids);
    cur = ggml_mul_mat(c, m.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    if (h.softcap != 0.0f) cur = ggml_scale(c, ggml_tanh(c, ggml_scale(c, cur, 1.0f / h.softcap)), h.softcap);
    cur = ggml_cont(c, ggml_view_2d(c, cur, h.n_vocab, cur->ne[1], cur->nb[1], 0));
    s.logits = cur;
    ggml_set_output(s.logits);
    ggml_build_forward_expand(s.gf, s.logits);
    return s;
}

#if XOLOTL_INKLING_STATS
// --stats: read the taps of a finished forward back to the host and accumulate (Hessians on the CPU).
// The per-token routing and the expert squared sums land in the sink; the MoE inputs of a held-out row
// also go to its dump buffer.
void stats_consume(Model& m, Step& s, int T, int64_t p0) {
    const Hp& h = m.hp;
    inkling_stats::Stats& st = *m.st;
    const int K = h.n_used, E = h.n_embd, F = h.n_ff_exp, NQ = h.n_q();
    std::vector<float> xq((size_t) E * T), xk, xv, xo((size_t) NQ * T), xf((size_t) E * T), act((size_t) F * K * T);
    std::vector<int32_t> idraw((size_t) h.n_expert * T), idb((size_t) K * T);
    std::vector<float> wb((size_t) K * T);
    auto get = [&](ggml_tensor* t, void* dst) { ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t)); };
    for (int il = 0; il < h.n_slot; ++il) {
        ggml_tensor** tp = s.st_taps.data() + (size_t) il * inkling_stats::ST_NKIND;
        if (!tp[inkling_stats::ST_IDS]) continue;
        const int key = st.key_of(il);
        // ids is a strided view over the full [n_expert, T] argsort (nb[1] = n_expert * 4); the top-k of
        // each token are its first k rows
        get(tp[inkling_stats::ST_IDS], idraw.data());
        for (int t = 0; t < T; ++t)
            for (int kk = 0; kk < K; ++kk) idb[(size_t) kk + (size_t) t * K] = idraw[(size_t) kk + (size_t) t * h.n_expert];
        get(tp[inkling_stats::ST_W], wb.data());
        get(tp[inkling_stats::ST_F], xf.data());
        st.route(il, idb.data(), wb.data(), T, K);
        if (st.dump_row) st.dump_add(il, xf.data(), p0, T);
        if (!st.fit_row) continue;           // held-out rows: route + dump only (no Hessians, no imatrix)
        st.hess(inkling_stats::ST_F, key, xf.data(), T);
        const auto te = Clock::now();
        get(tp[inkling_stats::ST_ACT], act.data());
        st.ms_get += ms_since(te);
        const auto tx = Clock::now();
        st.experts(key, xf.data(), idb.data(), T, K);
        st.experts_down(key, act.data(), idb.data(), T, K);
        st.ms_exp += ms_since(tx);
        get(tp[inkling_stats::ST_Q], xq.data());
        bool same_kv = tp[inkling_stats::ST_K] == tp[inkling_stats::ST_Q];
        if (!same_kv) {
            xk.resize((size_t) E * T); xv.resize((size_t) E * T);
            get(tp[inkling_stats::ST_K], xk.data());
            get(tp[inkling_stats::ST_V], xv.data());
        }
        st.hess_qkv(key, xq.data(), same_kv ? nullptr : xk.data(), same_kv ? nullptr : xv.data(), T);
        get(tp[inkling_stats::ST_O], xo.data());
        st.hess(inkling_stats::ST_O, key, xo.data(), T);
        st.tok[key] += T;
    }
    if (st.dump_row && p0 + T >= st.seq) st.dump_flush_row();   // only held-out rows reach the dump files
}
#endif

// --route-log: append this forward's per-slot routing to the record file. The ids tensor is a strided view over the
// full [n_expert, T] argsort (nb[1] = n_expert * 4); a flat read of ggml_nbytes() therefore starts at token 0's row
// and runs k + (T-1)*n_expert entries, exactly the layout stats_consume() reads (inkling_main.cpp:1119-1123).
// Record: int32 slot, phys, pass, T, k, 0; int64 p0; int32 ids[k*T]; float w[k*T].
void route_consume(Model& m, Step& s, int T, int64_t p0) {
    const Hp& h = m.hp;
    const int K = h.n_used;
    std::vector<int32_t> idraw((size_t) h.n_expert * T);
    std::vector<int32_t> idb((size_t) K * T);
    std::vector<float> wb((size_t) K * T);
    for (int il = 0; il < h.n_slot; ++il) {
        if (!s.rt_ids[il]) continue;
        ggml_backend_tensor_get(s.rt_ids[il], idraw.data(), 0, ggml_nbytes(s.rt_ids[il]));
        ggml_backend_tensor_get(s.rt_w[il], wb.data(), 0, ggml_nbytes(s.rt_w[il]));
        for (int t = 0; t < T; ++t)
            for (int kk = 0; kk < K; ++kk) idb[(size_t) kk + (size_t) t * K] = idraw[(size_t) kk + (size_t) t * h.n_expert];
        const int32_t phys = h.phys_of(il);
        const int32_t pass = il < h.n_phys ? 0 : 1 + (il - h.n_phys) / std::max(1, h.loop_end - h.loop_start);
        const int32_t hdr[6] = {il, phys, pass, T, K, 0};
        std::fwrite(hdr, sizeof(hdr), 1, m.route_fp);
        std::fwrite(&p0, sizeof(p0), 1, m.route_fp);
        std::fwrite(idb.data(), sizeof(int32_t), idb.size(), m.route_fp);
        std::fwrite(wb.data(), sizeof(float), wb.size(), m.route_fp);
        ++m.route_records;
    }
    std::fflush(m.route_fp);
}

// one forward pass; host logits for the last n_out tokens; taps (dump): l_out per slot, [T][n_embd] each
void drop_graph(Model& m) {
    if (m.last) { ggml_free(m.last->ctx); delete m.last; m.last = nullptr; }
    for (int64_t& k : m.last_key) k = -1;
}

void forward(Model& m, const int32_t* ids, int T, int64_t p0, int n_out, std::vector<float>& logits,
             std::vector<std::vector<float>>* taps = nullptr, std::vector<float>* dtap = nullptr,
             const float* media_rows = nullptr, const float* media_sel = nullptr) {
    static const bool dbg = std::getenv("XOLOTL_INKLING_DEBUG") != nullptr;
    // speculative verify chunks keep their own graph per length, so a changing draft length never rebuilds a graph
    Model::GSlot* gs = nullptr;
    if (dtap && m.spec && T > 1 && T <= kSpecT && n_out == T && !taps) {
        if (m.vg.empty()) m.vg.resize(kSpecT + 1);
        gs = &m.vg[(size_t) T];
        if (!gs->sched) {   // its own CUDA backend too: ggml-cuda keeps one CUDA graph per backend (no re-capture on switch)
            if (!gs->gpu) gs->gpu = ggml_backend_cuda_init(0);
            ggml_backend_t bes[2] = {gs->gpu ? gs->gpu : m.gpu, m.cpu};
            gs->sched = ggml_backend_sched_new(bes, nullptr, 2, kGraphNodes, false, true);
            gs->buf.resize(m.graph_buf.size());
        }
    }
    struct SlotSwap {
        Model& m; Model::GSlot* g;
        void sw() { std::swap(m.sched, g->sched); std::swap(m.last, g->last); std::swap_ranges(m.last_key, m.last_key + 9, g->key); m.graph_buf.swap(g->buf); }
        SlotSwap(Model& mm, Model::GSlot* gg) : m(mm), g(gg) { if (g) sw(); }
        ~SlotSwap() { if (g) sw(); }
    } slot_swap{m, gs};
    // the graph's shape: T, n_out, the global and sliding K/V windows (each padded to 256), dump taps
    const int64_t n_kv_full = std::min<int64_t>(m.n_ctx, ((p0 + T + 255) / 256) * 256);
    int64_t n_kv_swa;
    {
        int64_t hi = -1;
        for (int64_t j = 0; j < m.ring; ++j) if (m.ring_pos[j] >= 0) hi = j;
        for (int t = 0; t < T; ++t) hi = std::max<int64_t>(hi, (p0 + t) % m.ring);
        n_kv_swa = std::min<int64_t>(m.ring, ((hi + 1 + 255) / 256) * 256);
    }
    const bool vision = media_rows != nullptr;
    // the prompt attention kernel bakes pos0 into its op params, so its graph is only reusable at the same p0 (a chunk
    // with the same T in the same 256-cell bucket reran with the old position: wrong attention, 22 Nov S13)
    const int64_t key[9] = {T, n_out, n_kv_full, n_kv_swa, taps != nullptr, m.st != nullptr, dtap != nullptr, vision,
                            pattn_use(m, T) ? p0 : -1};
    const auto tb = Clock::now();
    const bool reuse = m.last && std::equal(key, key + 9, m.last_key);
    if (!reuse) {
        drop_graph(m);
        m.last = new Step(build(m, T, p0, n_out, taps != nullptr, dtap != nullptr, vision));
        ggml_backend_sched_reset(m.sched);
        // --vram-budget-mib: the cold arena lives in pinned host memory, which a discrete card does not accept as a
        // weight buft (ggml-cuda's supports_buft is integrated-only), so the scheduler would copy it to VRAM.  Pin
        // the cold mmid nodes AND their weight leaves to CUDA, after the reset (which clears ids) and before alloc
        // (a user assignment is never overwritten, and no copy is inserted when src and node share a backend).
        // --vram-budget-mib: pin every cold-related tensor to CUDA - the pinned-host weight leaves, the mmid nodes,
        // and (staged mode) the copy nodes with their graph-allocated VRAM destinations.  With all of them on one
        // backend the scheduler inserts no copy of its own and the cold path never falls to the CPU backend, which
        // is what the pinning is for; the regime (zero-copy at T == 1, staged at T >= 2) is chosen in G::moe().
        if (!m.last->cold_ops.empty()) {
            for (ggml_tensor* t : m.last->cold_w) ggml_backend_sched_set_tensor_backend(m.sched, t, m.gpu);
            for (ggml_tensor* t : m.last->cold_stage) ggml_backend_sched_set_tensor_backend(m.sched, t, m.gpu);
            for (ggml_tensor* t : m.last->cold_ops) ggml_backend_sched_set_tensor_backend(m.sched, t, m.gpu);
        }
        if (!ggml_backend_sched_alloc_graph(m.sched, m.last->gf)) { drop_graph(m); throw AllocFail(); }
        std::copy(key, key + 9, m.last_key);
    }
    Step& s = *m.last;
    const double ms_build = ms_since(tb);
    const double ms_alloc = 0;
    const auto tc = Clock::now();
    ggml_backend_tensor_set(s.tokens, ids, 0, (size_t) T * sizeof(int32_t));
    if (s.media) {
        ggml_backend_tensor_set(s.media, media_rows, 0, (size_t) T * m.hp.n_embd * sizeof(float));
        ggml_backend_tensor_set(s.media_sel, media_sel, 0, (size_t) T * sizeof(float));
    }
    {
        std::vector<float> pf(T);
        for (int t = 0; t < T; ++t) pf[t] = (float) (p0 + t);
        ggml_backend_tensor_set(s.posf, pf.data(), 0, pf.size() * sizeof(float));
    }
    for (int t = 0; t < T; ++t) m.ring_pos[(p0 + t) % m.ring] = p0 + t;
    {
        std::vector<int64_t> a(T), b(T);
        for (int t = 0; t < T; ++t) { a[t] = p0 + t; b[t] = (p0 + t) % m.ring; }
        ggml_backend_tensor_set(s.idx_full, a.data(), 0, a.size() * sizeof(int64_t));
        ggml_backend_tensor_set(s.idx_swa, b.data(), 0, b.size() * sizeof(int64_t));
    }
    const Hp& h = m.hp;
    for (const ChunkIn& ci : s.ck_full) {   // global slots: cell j = position j, masked when j > p0 + t
        if (!ci.mask) continue;   // positional (the kernel derives both from idx_full)
        std::vector<float> mk((size_t) ci.cells * T);
        std::vector<int32_t> rp((size_t) ci.cells * T);
        const int32_t ext = h.rel_extent;
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            for (int64_t jj = 0; jj < ci.cells; ++jj) {
                const int64_t j = ci.j0 + jj;
                const int64_t d = p - j;
                mk[(size_t) t * ci.cells + jj] = d >= 0 ? 0.0f : -INFINITY;
                rp[(size_t) t * ci.cells + jj] = d >= 0 && d < ext ? (int32_t) d : ext;
            }
        }
        ggml_backend_tensor_set(ci.mask, mk.data(), 0, mk.size() * sizeof(float));
        ggml_backend_tensor_set(ci.rp, rp.data(), 0, rp.size() * sizeof(int32_t));
    }
    for (const ChunkIn& ci : s.ck_swa) {    // sliding slots: ring cells, masked when empty, in the future or outside the window
        std::vector<float> mk((size_t) ci.cells * T);
        std::vector<int32_t> rp((size_t) ci.cells * T);
        const int32_t ext = h.n_swa;
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            for (int64_t jj = 0; jj < ci.cells; ++jj) {
                const int64_t kp = m.ring_pos[ci.j0 + jj];
                const int64_t d = kp >= 0 ? p - kp : -1;
                mk[(size_t) t * ci.cells + jj] = d >= 0 && d < h.n_swa ? 0.0f : -INFINITY;
                rp[(size_t) t * ci.cells + jj] = d >= 0 && d < ext ? (int32_t) d : ext;
            }
        }
        ggml_backend_tensor_set(ci.mask, mk.data(), 0, mk.size() * sizeof(float));
        ggml_backend_tensor_set(ci.rp, rp.data(), 0, rp.size() * sizeof(int32_t));
    }
    if (s.out_ids) {
        std::vector<int32_t> o(n_out);
        for (int i = 0; i < n_out; ++i) o[i] = T - n_out + i;
        ggml_backend_tensor_set(s.out_ids, o.data(), 0, o.size() * sizeof(int32_t));
    }
    const double ms_inputs = ms_since(tc);   // host-side input fill + upload (masks, positions, ids)
    if (ggml_backend_sched_graph_compute(m.sched, s.gf) != GGML_STATUS_SUCCESS) { drop_graph(m); die("graph compute failed"); }
    if (m.route_fp && !s.rt_ids.empty()) route_consume(m, s, T, p0);
#if XOLOTL_INKLING_STATS
    if (m.st && !s.st_taps.empty()) stats_consume(m, s, T, p0);
#endif
    {   // XOLOTL_INKLING_NODES=1: the op histogram of the first one-token graph (where fusion would cut launches)
        static bool shown = false;
        static const bool want = std::getenv("XOLOTL_INKLING_NODES") != nullptr;
        if (want && !shown && T == 1) {
            shown = true;
            std::map<std::string, int> hist;
            for (int i = 0; i < ggml_graph_n_nodes(s.gf); ++i) {
                ggml_tensor* n = ggml_graph_node(s.gf, i);
                std::string k = ggml_op_desc(n);
                if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) k += " (no kernel)";
                hist[k]++;
            }
            std::vector<std::pair<int, std::string>> v;
            for (auto& kv : hist) v.push_back({kv.second, kv.first});
            std::sort(v.rbegin(), v.rend());
            std::fprintf(stderr, "xolotl-inkling: one-token graph %d nodes:\n", ggml_graph_n_nodes(s.gf));
            for (auto& e : v) std::fprintf(stderr, "  %5d  %s\n", e.first, e.second.c_str());
            if (std::atoi(std::getenv("XOLOTL_INKLING_NODES")) >= 2)   // =2: every kernel node in order (op, name, shape, srcs)
                for (int i = 0; i < ggml_graph_n_nodes(s.gf); ++i) {
                    ggml_tensor* n = ggml_graph_node(s.gf, i);
                    if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
                    std::fprintf(stderr, "  #%04d %-10s %-28s [%lld,%lld,%lld]", i, ggml_op_desc(n), n->name, (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2]);
                    for (int k = 0; k < 4 && n->src[k]; ++k) std::fprintf(stderr, " <%s:%s>", n->src[k]->name, ggml_type_name(n->src[k]->type));
                    std::fprintf(stderr, "\n");
                }
        }
    }
    if (dbg) std::fprintf(stderr, "xolotl-inkling: T=%d nodes=%d splits=%d %s %.1f ms inputs %.2f ms compute %.1f ms\n", T,
                          ggml_graph_n_nodes(s.gf), ggml_backend_sched_get_n_splits(m.sched), reuse ? "reused" : "built+alloc", ms_build + ms_alloc,
                          ms_inputs, ms_since(tc) - ms_inputs);
    logits.resize((size_t) n_out * h.n_vocab);
    ggml_backend_tensor_get(s.logits, logits.data(), 0, logits.size() * sizeof(float));
    if (taps) {
        taps->assign(h.n_slot, {});
        for (int i = 0; i < h.n_slot; ++i) {
            (*taps)[i].resize((size_t) T * h.n_embd);
            ggml_backend_tensor_get(s.taps[i], (*taps)[i].data(), 0, (*taps)[i].size() * sizeof(float));
        }
    }
    if (dtap) {   // [T][4][n_embd]: the drafter's tap order (slots 4, 28, 46, result_norm)
        const size_t E = (size_t) h.n_embd;
        std::vector<float> one((size_t) T * E);
        dtap->resize((size_t) T * 4 * E);
        for (int j = 0; j < 4; ++j) {
            ggml_backend_tensor_get(s.dtaps[j], one.data(), 0, one.size() * sizeof(float));
            for (int t = 0; t < T; ++t) std::memcpy(dtap->data() + ((size_t) t * 4 + j) * E, one.data() + (size_t) t * E, E * sizeof(float));
        }
    }
}

// ---- speculative decoding (--dspark, greedy only)
// The conv state after the first n tokens of a verify chunk = rows [n, n + K-1) of the saved [state ; inputs] (the
// convolve() history), so a rejected tail is undone exactly with no second forward.  Sliding rings: the cells the
// rejected positions wrote held positions >= window behind (ring = window + ubatch), so they only need marking empty.
// Global K/V is positional: a rejected cell is masked (causal) until it is rewritten.
void conv_rollback(Model& m, int n) {
    const int64_t K1 = m.hp.n_conv - 1, C = m.hp.n_chan();
    for (Slot& z : m.S) {
        const size_t sp = (size_t) z.conv_hist->nb[1], dp = (size_t) z.conv->nb[1];
        cudaMemcpy2D(z.conv->data, dp, (const char*) z.conv_hist->data + (size_t) n * sizeof(float), sp,
                     (size_t) K1 * sizeof(float), (size_t) C, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
}

struct SpecStats { long long steps = 0, drafted = 0, accepted = 0; long long hist[kSpecT + 1] = {}; double ms_draft = 0, ms_verify = 0, ms_post = 0;
                   long long plain = 0, switches = 0; double ms_plain = 0; };

// One verify step.  `fed` = committed tokens (positions 0..P-1, the drafter holds their taps); `next` = the target's
// greedy token at P, not yet fed.  Commits next + the accepted drafts to `fed`, returns how many (>= 1), and leaves
// the logits row after the last committed token in `lg` (its argmax is the next step's `next`).
// `pick` (null = greedy argmax) chooses the target's token from a logits row.  Sampled speculation is exact: the draft
// is deterministic, so drawing s ~ p from the verify row and accepting iff s == draft accepts with probability p(draft)
// and, on a reject, s is already a draw from p restricted to != draft (the residual) - it is returned in *resample and
// MUST be the next token (re-drawing from the same row would be biased).  *resample = -1 when every draft was accepted.
int spec_step(Model& m, std::vector<int32_t>& fed, int32_t next, std::vector<float>& lg, SpecStats& ss,
              const std::function<int32_t(const float*)>& pick = nullptr, int32_t* resample = nullptr) {
    if (resample) *resample = -1;
    const int64_t P = (int64_t) fed.size();
    const int64_t nv = m.hp.n_vocab;
    int32_t d[16] = {};
    const auto ta = Clock::now();
    static const bool exact = std::getenv("XOLOTL_SPEC_EXACT") != nullptr;   // off: 2..4-column reductions differ only in rounding
    int nall = 0;
    const int nd = dspark_draft(*m.dspark, fed.back(), P, d, &nall);   // d[0] predicts `next`, d[i] the token after d[i-1]
    ss.ms_draft += ms_since(ta);
    int V = std::min<int>(kSpecT, nd);
    // bit-exact shapes: ggml's quantised mat-vec reduces 2..4 columns in a different order than 1 or 5..8, which can
    // flip a greedy near-tie vs plain decoding.  Exact mode pads 2..4 up to 5 (the extra drafts are simply rejected).
    if (exact && V >= 2 && V <= 4) V = std::min<int>(5, std::max(V, nall));
    if (exact && V >= 2 && V <= 4) V = 1;
    if (P + V >= m.n_ctx) V = 1;
    std::vector<int32_t> chunk((size_t) V);
    chunk[0] = next;
    for (int i = 1; i < V; ++i) chunk[(size_t) i] = d[i];
    std::vector<float> lgs, dt;
    const auto tv = Clock::now();
    forward(m, chunk.data(), V, P, V, lgs, nullptr, &dt);
    ss.ms_verify += ms_since(tv);
    const auto tp = Clock::now();
    int n = 1;
    while (n < V) {
        const float* r = lgs.data() + (size_t) (n - 1) * nv;
        const int32_t t = pick ? pick(r) : (int32_t) (std::max_element(r, r + nv) - r);
        if (t != chunk[(size_t) n]) { if (resample) *resample = t; break; }
        ++n;
    }
    if (n < V) {
        conv_rollback(m, n);
        for (int t = n; t < V; ++t) m.ring_pos[(P + t) % m.ring] = -1;
    }
    dspark_add_ctx(*m.dspark, dt.data(), n, P);
    fed.insert(fed.end(), chunk.begin(), chunk.begin() + n);
    lg.assign(lgs.begin() + (size_t) (n - 1) * nv, lgs.begin() + (size_t) n * nv);
    ss.ms_post += ms_since(tp);
    ++ss.steps; ss.drafted += V - 1; ss.accepted += n - 1; ++ss.hist[n];
    return n;
}

// Adaptive drafting (XOLOTL_DSPARK_ADAPT=1, opt-in; default drafts every step).  A verify step pays the drafter and the
// post step (~4.5 ms on the 4090) plus ~2 ms per extra verified row (each row routes to new experts), so it only wins
// when enough drafts are accepted: on wiki prose every draft cap lost to plain decode (t17, 22 Nov: 91-96 vs 109 tok/s)
// while code kept +20 %.  The policy measures committed tokens per ms of both modes (EMA) and runs plain steps while
// they are faster, probing the drafter again after 16, 32 .. 256 plain steps.  A plain step is a T = 1 forward that
// still returns the drafter taps; they are queued and fed to the drafter in one batch before its next draft, so the
// drafter never misses a position.  Output tokens are unchanged (greedy: the same argmax; sampled: the same sampler on
// the same row - a plain step is the non-speculative decode).
struct SpecPolicy {
    // windowed rates (tokens and ms summed over the last kWin steps of each mode): one verify commits 1..8 tokens, so
    // a per-step average swings far more than the difference it has to detect
    static constexpr int kWin = 16, kProbe = 8;
    struct Win {
        double n[kWin] = {}, ms[kWin] = {}; int k = 0, len = 0;
        void add(double tok, double t) { n[k] = tok; ms[k] = t; k = (k + 1) % kWin; len = std::min(len + 1, kWin); }
        double rate() const { double a = 0, b = 0; for (int i = 0; i < len; ++i) { a += n[i]; b += ms[i]; } return b > 0 ? a / b : 0; }
        void clear() { len = 0; k = 0; }
    } spec, plain;
    int plain_left = 0, probe_left = kProbe, backoff = 16;
    long long spec_since_plain = 0;
    bool plain_warm = false;                   // the first plain step builds its graph: not a rate sample
    std::vector<float> pend;                   // queued taps [n][4][n_embd] of plain steps
    int64_t pend_p0 = -1;
    int pend_n = 0;
};

static void spec_flush(Model& m, SpecPolicy& pol) {
    if (pol.pend_n == 0) return;
    dspark_add_ctx(*m.dspark, pol.pend.data(), pol.pend_n, pol.pend_p0);
    pol.pend.clear(); pol.pend_n = 0; pol.pend_p0 = -1;
}

int adaptive_step(Model& m, std::vector<int32_t>& fed, int32_t next, std::vector<float>& lg, SpecStats& ss, SpecPolicy& pol,
                  const std::function<int32_t(const float*)>& pick = nullptr, int32_t* resample = nullptr) {
    // opt-in: t22 (22 Nov, 4096 greedy thinking tokens) always-on 127.7 tok/s, adaptive 125.9, plain 113.3; it only
    // helped prose (106 vs 101, plain 111) - drafting every step stays the default until a policy wins everywhere
    static const bool on = std::getenv("XOLOTL_DSPARK_ADAPT") && std::string(std::getenv("XOLOTL_DSPARK_ADAPT")) == "1";
    static const bool dbg = std::getenv("XOLOTL_DSPARK_ADAPT_DEBUG") != nullptr;
    if (!on) return spec_step(m, fed, next, lg, ss, pick, resample);
    const int64_t P = (int64_t) fed.size();
    if (pol.plain_left > 0) {
        if (resample) *resample = -1;
        const auto t0 = Clock::now();
        std::vector<float> lgs, dt;
        forward(m, &next, 1, P, 1, lgs, nullptr, &dt);
        if (pol.pend_n == 0) pol.pend_p0 = P;
        pol.pend.insert(pol.pend.end(), dt.begin(), dt.end());
        if (++pol.pend_n >= 16) spec_flush(m, pol);
        fed.push_back(next);
        lg.swap(lgs);
        const double ms = ms_since(t0);
        if (pol.plain_warm) pol.plain.add(1, ms);
        pol.plain_warm = true;
        ++ss.plain; ss.ms_plain += ms;
        if (--pol.plain_left == 0) { pol.probe_left = SpecPolicy::kProbe; pol.spec.clear(); pol.spec_since_plain = 0; }
        return 1;
    }
    const auto t0 = Clock::now();
    spec_flush(m, pol);
    const int n = spec_step(m, fed, next, lg, ss, pick, resample);
    pol.spec.add(n, ms_since(t0));
    ++pol.spec_since_plain;
    if (pol.probe_left > 0 && --pol.probe_left > 0) return n;
    if (pol.plain.len == 0 || pol.spec_since_plain >= 512) {   // (re)measure plain decode: 4 steps (5 the first time)
        pol.plain_left = pol.plain_warm ? 4 : 5; pol.spec_since_plain = 0;
        return n;
    }
    const double rs = pol.spec.rate(), rp = pol.plain.rate();
    if (rs < 0.95 * rp) {
        if (dbg) std::fprintf(stderr, "dspark adapt @%lld: spec %.1f < plain %.1f tok/s -> %d plain steps\n", (long long) P,
                              rs * 1e3, rp * 1e3, pol.backoff);
        pol.plain_left = pol.backoff; pol.backoff = std::min(256, pol.backoff * 2); ++ss.switches;
    } else if (rs > 1.05 * rp) {
        pol.backoff = 16;
    }
    return n;
}

// the longest chunk at p whose attention scores fit the budget (global slots read p + T cells)
int chunk_len(const Model& m, int64_t p, int64_t left) {
    int T = (int) std::min<int64_t>(m.ubatch, left);
    while (!pattn_use(m, T) && T > 16 && ((p + T + 255) / 256) * 256 * (int64_t) T > m.attn_budget) T /= 2;   // no score tensor with it
    // a chunk's sliding keys (n_swa - 1 before it + its T) must all be in the ring at once
    if (m.ring < m.n_ctx) T = (int) std::min<int64_t>(T, std::max<int64_t>(1, m.ring - m.hp.n_swa + 1));
    return std::max(1, T);
}

// ---- checkpoint: the sliding rings + every conv state (global K/V is positional)
struct Ckpt {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> k, v, cv;
    std::vector<int64_t> ring_pos;
    // serve: the tokens 0..pos-1 the checkpoint stands for, the drafter at that position, and an LRU stamp
    std::vector<int32_t> toks;
    DsparkSnap ds;
    bool has_ds = false;
    uint64_t used = 0;
};
void ckpt_take(Model& m, Ckpt& c, int64_t pos) {
    c.pos = pos;
    const int n = m.hp.n_slot;
    c.k.resize(n); c.v.resize(n); c.cv.resize(n);
    auto grab = [](ggml_tensor* t, std::vector<uint8_t>& out) { out.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, out.data(), 0, out.size()); };
    for (int s = 0; s < n; ++s) {
        if (m.S[s].swa) { grab(m.S[s].k_cache, c.k[s]); grab(m.S[s].v_cache, c.v[s]); }
        grab(m.S[s].conv, c.cv[s]);
    }
    c.ring_pos = m.ring_pos;
}
void ckpt_restore(Model& m, const Ckpt& c) {
    for (int s = 0; s < m.hp.n_slot; ++s) {
        if (m.S[s].swa) {
            ggml_backend_tensor_set(m.S[s].k_cache, c.k[s].data(), 0, c.k[s].size());
            ggml_backend_tensor_set(m.S[s].v_cache, c.v[s].data(), 0, c.v[s].size());
        }
        ggml_backend_tensor_set(m.S[s].conv, c.cv[s].data(), 0, c.cv[s].size());
    }
    m.ring_pos = c.ring_pos;
}
void state_reset(Model& m) {
    std::fill(m.ring_pos.begin(), m.ring_pos.end(), -1);
    for (Slot& z : m.S) ggml_backend_tensor_memset(z.conv, 0, 0, ggml_nbytes(z.conv));   // the conv history starts at zero
}

// ---- park / unpark
void detach(ggml_context* ctx) {
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) { t->data = nullptr; t->buffer = nullptr; }
}
void park(Model& m) {
    if (m.parked) return;
    if (!m.host_w) die("park: no pinned host copy of the weights (cudaHostAlloc failed at load, or --no-park-copy)");
    drop_graph(m);
    for (Model::GSlot& g : m.vg) {   // the per-length verify graphs are rebuilt on demand after unpark
        if (g.last) { ggml_free(g.last->ctx); delete g.last; g.last = nullptr; }
        if (g.sched) { ggml_backend_sched_free(g.sched); g.sched = nullptr; }
        for (int64_t& k : g.key) k = -1;
    }
    m.host_state.resize(m.state_bytes);
    size_t at = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.sctx); t; t = ggml_get_next_tensor(m.sctx, t)) {
        ggml_backend_tensor_get(t, m.host_state.data() + at, 0, ggml_nbytes(t));
        at += ggml_nbytes(t);
    }
    ggml_backend_sched_free(m.sched); m.sched = nullptr;
    ggml_backend_buffer_free(m.sbuf); m.sbuf = nullptr;
    ggml_backend_buffer_free(m.wbuf); m.wbuf = nullptr;
    detach(m.sctx);
    for (ggml_tensor* t : m.gpu_w) { t->data = nullptr; t->buffer = nullptr; }
    ggml_backend_free(m.gpu); m.gpu = nullptr;
    m.parked = true;
}
void unpark(Model& m) {
    if (!m.parked) return;
    m.gpu = ggml_backend_cuda_init(0);
    if (!m.gpu) die("unpark: the CUDA backend failed to start");
    alloc_gpu_weights(m);
    for (size_t i = 0; i < m.gpu_w.size(); ++i) ggml_backend_tensor_set(m.gpu_w[i], m.host_w + m.host_w_at[i], 0, ggml_nbytes(m.gpu_w[i]));
    m.sbuf = ggml_backend_alloc_ctx_tensors(m.sctx, m.gpu);
    if (!m.sbuf) die("unpark: not enough VRAM for the K/V cache");
    size_t at = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.sctx); t; t = ggml_get_next_tensor(m.sctx, t)) {
        ggml_backend_tensor_set(t, m.host_state.data() + at, 0, ggml_nbytes(t));
        at += ggml_nbytes(t);
    }
    std::vector<uint8_t>().swap(m.host_state);
    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, kGraphNodes, false, true);
    m.parked = false;
}

size_t vram_free_mib() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
    return f >> 20;
}

// ----------------------------------------------------------------------------------------------------- protocol
struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    bool eof = false;
    void push(std::string s) { std::lock_guard<std::mutex> l(m); q.push_back(std::move(s)); cv.notify_one(); }
    bool pop(std::string& out, int64_t wait_ms = 0, bool* timed_out = nullptr) {
        std::unique_lock<std::mutex> l(m);
        if (timed_out) *timed_out = false;
        if (wait_ms > 0) {
            if (!cv.wait_for(l, std::chrono::milliseconds(wait_ms), [&] { return !q.empty() || eof; })) {
                if (timed_out) *timed_out = true;
                return false;
            }
        } else cv.wait(l, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front()); q.pop_front();
        return true;
    }
};

bool parse_ids(const char* p, std::vector<int32_t>& ids) {
    ids.clear();
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        ids.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != '\r' && *p != '\n') return false;
        else break;
    }
    return !ids.empty();
}

// The encoder's file (tools/vision, llama.cpp's mtmd): records of int32 {tag, n, nx, ny, n_embd} + float32 [n][n_embd],
// 'SVE1' an image, in prompt order.  Rows are kept as float32; the graph blends them into the token embeddings.
bool read_media(const std::string& path, int n_embd, std::vector<float>& rows, std::string& err) {
    rows.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open the embeddings file " + path; return false; }
    for (;;) {
        int32_t hd[5];
        const size_t got = std::fread(hd, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hd[0] != 0x31455653 || hd[1] <= 0 || hd[4] != n_embd) {
            std::fclose(f);
            err = "bad embeddings record (want SVE1 with n_embd " + std::to_string(n_embd) + ")";
            return false;
        }
        const size_t n = (size_t) hd[1] * (size_t) n_embd, at = rows.size();
        rows.resize(at + n);
        if (std::fread(rows.data() + at, sizeof(float), n, f) != n) { std::fclose(f); err = "short embeddings record"; return false; }
    }
    std::fclose(f);
    return true;
}

std::vector<int32_t> token_ids_named(const Model& m, std::initializer_list<const char*> names) {
    std::vector<int32_t> out;
    const int64_t tk = m.gg.key("tokenizer.ggml.tokens", false);
    if (tk < 0) return out;
    for (size_t i = 0; i < gguf_get_arr_n(m.gg.g, tk); ++i) {
        const char* s = gguf_get_arr_str(m.gg.g, tk, i);
        for (const char* n : names) if (!std::strcmp(s, n)) out.push_back((int32_t) i);
    }
    return out;
}

// --lora <adapter.gguf> [--lora-scale S]: a llama.cpp-format LoRA adapter (what --train writes, tools/train/lora_gguf.hpp)
// on the attention projections. Each "<w>.lora_a" [n_in, r] / "<w>.lora_b" [r, n_out] pair attaches to the model
// tensor <w>; the delta is (alpha / r) * S * B (A x), exactly the trainer's.
void load_lora(Model& m, const std::string& path, float user_scale) {
    ggml_context* meta = nullptr;
    gguf_init_params gp = {true, &meta};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) die("--lora: cannot read " + path);
    auto str = [&](const char* k) -> std::string { const int64_t i = gguf_find_key(g, k); return i >= 0 ? gguf_get_val_str(g, i) : ""; };
    if (str("adapter.type") != "lora") die("--lora: " + path + " is not a LoRA adapter (adapter.type)");
    if (!str("general.architecture").empty() && str("general.architecture") != m.arch)
        die("--lora: adapter is for " + str("general.architecture") + ", the model is " + m.arch);
    const int64_t ka = gguf_find_key(g, "adapter.lora.alpha");
    std::vector<std::pair<std::string, std::pair<ggml_tensor*, ggml_tensor*>>> pairs;
    for (ggml_tensor* t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        const std::string n = ggml_get_name(t);
        if (n.size() < 7 || n.compare(n.size() - 7, 7, ".lora_a") != 0) continue;
        const std::string base = n.substr(0, n.size() - 7);
        ggml_tensor* b = ggml_get_tensor(meta, (base + ".lora_b").c_str());
        if (!b) die("--lora: " + n + " has no .lora_b");
        pairs.push_back({base, {t, b}});
    }
    if (pairs.empty()) die("--lora: no .lora_a/.lora_b pairs in " + path);
    const int64_t rank = pairs[0].second.first->ne[1];
    const float alpha = ka >= 0 ? gguf_get_val_f32(g, ka) : (float) rank;
    ggml_init_params ip = {ggml_tensor_overhead() * (pairs.size() * 2 + 4), nullptr, true};
    m.lora_ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> dev;   // (device tensor, file tensor)
    for (auto& [base, ab] : pairs) {
        ggml_tensor *a = ab.first, *b = ab.second;
        ggml_tensor* W = get(m, base, false);
        if (!W) die("--lora: the model has no tensor " + base);
        if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) die("--lora: " + base + ": adapters must be F32");
        if (a->ne[1] != rank || b->ne[0] != rank) die("--lora: mixed ranks are not supported (" + base + ")");
        if (W->ne[0] != a->ne[0] || W->ne[1] != b->ne[1]) die("--lora: shape mismatch for " + base);
        ggml_tensor* da = ggml_dup_tensor(m.lora_ctx, a);
        ggml_tensor* db = ggml_dup_tensor(m.lora_ctx, b);
        ggml_set_name(da, ggml_get_name(a)); ggml_set_name(db, ggml_get_name(b));
        m.lora[W] = {da, db};
        dev.push_back({da, a}); dev.push_back({db, b});
    }
    m.lora_buf = ggml_backend_alloc_ctx_tensors(m.lora_ctx, m.gpu ? m.gpu : m.cpu);
    if (!m.lora_buf) die("--lora: cannot allocate the adapter tensors");
    ggml_backend_buffer_set_usage(m.lora_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("--lora: cannot open " + path);
    std::vector<uint8_t> buf;
    for (auto& [d, t] : dev) {
        const int64_t ti = gguf_find_tensor(g, ggml_get_name(t));
        buf.resize(ggml_nbytes(t));
        if (fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti)), SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, buf.size(), f) != buf.size()) die("--lora: short read in " + path);
        ggml_backend_tensor_set(d, buf.data(), 0, buf.size());
    }
    std::fclose(f);
    m.lora_scale = alpha / (float) rank * user_scale;
    std::fprintf(stderr, "xolotl-inkling: LoRA %s: %zu adapted weights, rank %lld, alpha %.3g, scale %.4g (%.1f MiB)\n",
                 path.c_str(), pairs.size(), (long long) rank, alpha, m.lora_scale, ggml_backend_buffer_get_size(m.lora_buf) / 1048576.0);
    gguf_free(g);
    ggml_free(meta);
}

void usage() {
    std::fprintf(stderr,
                 "usage: xolotl-inkling --serve --native <inkling-2-mini.gguf> [--context N] [--ubatch N] [--threads N]\n"
                 "       [--cpu-exps L] [--kv-type f16|q8_0|k8v4] [--dspark F | --mtp] [--attn-budget N] [--park-idle SECS] [--peer-vram URL]\n"
                 "       [--no-park-copy] [--route-log PATH] [--lora ADAPTER.gguf [--lora-scale S]]\n"
                 "  --lora applies a LoRA adapter (as written by --train) to the attention projections at serve time.\n"
                 "  --route-log appends per-visit routing records (slot, phys, pass, ids, w) for expert-usage analysis;\n"
                 "  off by default, and off costs nothing (no tap tensor is marked).  XOLOTL_INKLING_ROUTE_LOG=PATH too.\n"
                 "  --vram-budget-mib N (off by default) keeps each layer's most-routed experts in VRAM and the rest in\n"
                 "  pinned host RAM, so a 12 GB card runs the 14 GB quant; the loop core (visited 3x) is kept first.\n"
                 "  --kv-type quantises the full-context K/V cache (k8v4 = K q8_0, V q4_0; the sliding-window rings stay f16).\n"
                 "  --attn-budget caps one prefill chunk's n_kv*T scores and sizes the attention cell chunks (default 33554432).\n"
                 "       [--mmproj <mmproj.gguf> [--vision-device gpu|cpu]]\n"
                 "  --mmproj loads Inkling-2-Mini's image tower (convert_hf_to_gguf.py --mmproj, inkling2v) and answers\n"
                 "  `ENC <image> <out>` (\"OK <n> <nx> <ny> <ms>\") with the SVE1 records a GENI request reads; ~640 MB\n"
                 "  of VRAM while one picture is encoded, released after.  --vision-device cpu keeps it off the GPU.\n"
                 "       xolotl-inkling --encode <image> <out.sve1> --mmproj <mmproj.gguf> [--vision-device cpu|gpu]\n"
                 "  One picture through the in-process encoder, no model: the gate for the tower alone.\n"
                 "       xolotl-inkling --dump <inkling-2-mini.gguf> --ids a,b,c,... --out-dir DIR [--gen N]\n"
                 "  Inkling-2-Mini behind Xolotl's server protocol.  --cpu-exps 20-23 keeps those physical layers' experts in\n"
                 "  host RAM (the CPU computes them).  --dump writes logits.f32 [T][n_vocab] and l_out-<slot>.f32 [T][n_embd]\n"
                 "  for the parity check, and with --gen N prints N greedy token ids.\n"
                 "       xolotl-inkling --stats <inkling-2-mini.gguf> --tokens <tokens.npy> --out DIR [--rows N] [--seq L]\n"
                 "       [--row-start A] [--heap-per-visit] [--with-down-hess] [--stats-chunk N] [--stats-batch N] [--meta meta.json]\n"
                 "  Held-out rows (meta.json is_eval) feed only DIR/route and DIR/dumps; the Hessians and imatrix are fit rows.\n"
                 "  --stats accumulates the teacher statistics into DIR/hess, DIR/imatrix, DIR/route and DIR/dumps\n"
                 "  (see src/program/inkling_stats.hpp for every format).  The down projection keeps its imatrix\n"
                 "  diagonal (exp_dn_sq); --with-down-hess is not produced by this slice.\n");
}

}  // namespace

int inkling_train_main(int argc, char** argv);   // --train: inkling_train.cpp (included at the end)

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--train") return inkling_train_main(argc, argv);
    std::string gguf_path, ids_arg, out_dir, cpu_exps, dspark_path, dspark_prefix;
    bool mtp = false;   // --mtp: the DSpark (MTP) drafter built into the model GGUF (tools/inkling/embed_dspark.py)
    std::string mmproj_path, vision_device = "auto";   // --mmproj: the image tower
    std::vector<std::string> encode_imgs, encode_outs;   // --encode pairs (one Encoder, N pictures)
    bool serve = false, dump = false, park_copy = true;
    ggml_type kv_k = GGML_TYPE_F16, kv_v = GGML_TYPE_F16;
    int64_t ctx = 0, attn_budget = 1 << 25;
    int ubatch = 0, threads = 8, gen_n = 0;   // ubatch 0 = auto (2048 with the prompt attention on a >= 20 GB card, else 512)
    double park_idle_s = 0;
    std::string peer_vram, route_log, lora_path;
    float lora_user_scale = 1.0f;
    int64_t vram_budget_mib = 0;
    // --stats
    bool stats = false, per_visit = false, with_down_hess = false, dump_eval = true;
    std::string tokens_path, stats_out, meta_path;
    int64_t s_rows = -1, s_seq = -1, s_row_start = 0;
    int s_chunk = 128, s_batch = 4096;
    double hess_max_gb = 24.0, dump_max_gb = 9.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--arena-selftest") {   // the remap probe alone: a few MB of VRAM, no model load
            const std::string p = val();
            const int l = i + 1 < argc ? std::atoi(argv[i + 1]) : 0;
            if (i + 1 < argc && argv[i + 1][0] != '-') ++i;
            return arena_selftest_main(p, l);
        }
        if (a == "--serve") serve = true;
        else if (a == "--dump") { dump = true; gguf_path = val(); }
        else if (a == "--stats") {
            if (!XOLOTL_INKLING_STATS) { std::fprintf(stderr, "xolotl-inkling: --stats is not in this build\n"); return 2; }
            stats = true; gguf_path = val();
        }
        else if (a == "--tokens") tokens_path = val();
        else if (a == "--out") stats_out = val();
        else if (a == "--rows") s_rows = std::atoll(val().c_str());
        else if (a == "--seq") s_seq = std::atoll(val().c_str());
        else if (a == "--row-start") s_row_start = std::atoll(val().c_str());
        else if (a == "--stats-chunk") s_chunk = std::max(16, std::atoi(val().c_str()));
        else if (a == "--stats-batch") s_batch = std::max(16, std::atoi(val().c_str()));
        else if (a == "--heap-per-visit") per_visit = true;
        else if (a == "--with-down-hess") with_down_hess = true;
        else if (a == "--no-dump-eval") dump_eval = false;
        else if (a == "--meta") meta_path = val();
        else if (a == "--hess-max-gb") hess_max_gb = std::atof(val().c_str());
        else if (a == "--dump-max-gb") dump_max_gb = std::atof(val().c_str());
        else if (a == "--native") gguf_path = val();
        else if (a == "--context" || a == "--max-context") ctx = std::atoll(val().c_str());
        else if (a == "--ubatch") ubatch = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--cpu-exps") cpu_exps = val();
        else if (a == "--vram-budget-mib") vram_budget_mib = std::atoll(val().c_str());
        else if (a == "--attn-budget") attn_budget = std::max<int64_t>(4096, std::atoll(val().c_str()));
        else if (a == "--park-idle") park_idle_s = std::atof(val().c_str());
        else if (a == "--peer-vram") peer_vram = val();
        else if (a == "--no-park-copy") park_copy = false;
        else if (a == "--route-log") route_log = val();
        else if (a == "--lora") lora_path = val();
        else if (a == "--lora-scale") lora_user_scale = (float) std::atof(val().c_str());
        else if (a == "--mmproj") mmproj_path = val();
        else if (a == "--vision-device") {
            vision_device = val();
            if (vision_device != "cpu" && vision_device != "gpu" && vision_device != "auto") { std::fprintf(stderr, "xolotl-inkling: --vision-device takes cpu or gpu\n"); return 2; }
        }
        else if (a == "--encode") { encode_imgs.push_back(val()); encode_outs.push_back(val()); }
        else if (a == "--ids") {   // a,b,c,... or @file holding that list (long contexts overflow argv)
            ids_arg = val();
            if (!ids_arg.empty() && ids_arg[0] == '@') { std::ifstream f(ids_arg.substr(1)); std::stringstream ss; ss << f.rdbuf(); ids_arg = ss.str(); while (!ids_arg.empty() && std::isspace((unsigned char) ids_arg.back())) ids_arg.pop_back(); }
        }
        else if (a == "--out-dir") out_dir = val();
        else if (a == "--gen") gen_n = std::atoi(val().c_str());
        else if (a == "--dspark") dspark_path = val();
        else if (a == "--mtp") mtp = true;
        else if (a == "--kv-type") {
            const std::string v = val();
            if (v == "f16") { kv_k = kv_v = GGML_TYPE_F16; }
            else if (v == "q8_0") { kv_k = kv_v = GGML_TYPE_Q8_0; }
            else if (v == "k8v4") { kv_k = GGML_TYPE_Q8_0; kv_v = GGML_TYPE_Q4_0; }
            else { std::fprintf(stderr, "xolotl-inkling: --kv-type takes f16, q8_0 or k8v4 (got %s)\n", v.c_str()); return 2; }
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else std::fprintf(stderr, "xolotl-inkling: ignoring unknown option %s\n", a.c_str());
    }
    // --encode: the image tower alone (no model, no GPU unless asked): its own CPU/gpu backends
    if (!encode_imgs.empty() || !encode_outs.empty()) {
        if (encode_imgs.size() != encode_outs.size() || mmproj_path.empty()) { usage(); return 2; }
        ggml_backend_t cpu = ggml_backend_cpu_init();
        // the tower alone defaults to the CPU: this mode is also the gate that must run beside a busy card
        ggml_backend_t gpu = vision_device == "gpu" ? ggml_backend_cuda_init(0) : nullptr;
        std::string verr;
        inkling_vision::Encoder* ve = inkling_vision::load(mmproj_path, gpu, cpu, gpu != nullptr, verr);
        if (!ve) { std::printf("ERR %s\n", verr.c_str()); return 1; }
        int rc = 0;
        for (size_t i = 0; i < encode_imgs.size(); ++i) {
            const auto t0 = Clock::now();
            const int rows = inkling_vision::encode(*ve, encode_imgs[i], encode_outs[i], 0, verr);
            if (rows <= 0) { std::printf("ERR %s\n", verr.c_str()); rc = 1; break; }
            std::printf("OK %d %d 1 %.0f\n", rows, rows, ms_since(t0));
            std::fflush(stdout);
        }
        inkling_vision::free(ve);
        if (gpu) ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return rc;
    }
    if (gguf_path.empty() || (!serve && !dump && !stats)) { usage(); return 2; }
    if (stats && (tokens_path.empty() || stats_out.empty())) { usage(); return 2; }
    if (with_down_hess)
        std::fprintf(stderr, "xolotl-inkling: stats: --with-down-hess is not produced by this slice "
                             "(exp_dn_sq, the per-expert down diagonal, is); continuing with the diagonal only\n");
#ifdef SIGPIPE                             // MSVC's signal.h has no SIGPIPE (a write to a closed pipe returns an error there)
    signal(SIGPIPE, SIG_IGN);
#endif

    if (route_log.empty()) if (const char* e = std::getenv("XOLOTL_INKLING_ROUTE_LOG")) route_log = e;

    Model m;
    // ubatch 0 = auto: load() picks the biggest prompt chunk (2048 / 1024 / 512 / 256) that the VRAM left after the
    // weights holds with its KV, compute buffer and the drafter (bigger chunks read each expert once for more rows:
    // 8K prefill GPU 1.84 s at 512 -> 1.40 s at 2048; compute ~0.75 MiB a row, measured t16)
    m.ubatch_auto = ubatch <= 0 && !stats;
    if (ubatch <= 0) ubatch = 512;
    m.ubatch = std::max(16, stats ? std::min(ubatch, s_chunk) : ubatch);
    m.attn_budget = attn_budget;
    m.kv_k = kv_k;
    m.kv_v = kv_v;
    m.vram_budget_mib = vram_budget_mib;
    if (!route_log.empty()) {   // --route-log: file header, then route_consume() appends one record per slot visit
        m.route_fp = std::fopen(route_log.c_str(), "wb");
        if (!m.route_fp) { std::fprintf(stderr, "xolotl-inkling: cannot write --route-log %s\n", route_log.c_str()); return 2; }
        static char rb[1 << 20];
        std::setvbuf(m.route_fp, rb, _IOFBF, sizeof rb);
        const uint32_t rh[3] = {0x54525851u /* "XQRT" */, 1u, 0u};   // magic, version, reserved
        std::fwrite(rh, sizeof(rh), 1, m.route_fp);
    }
    try {
        const int64_t want_ctx = stats ? std::max<int64_t>(s_seq > 0 ? s_seq + 16 : 0, 1024)
                             : dump ? std::max<int64_t>(ctx, 4096) : ctx;
        if (mtp && dspark_path.empty()) {
            if (!dspark_in_gguf(gguf_path)) die("--mtp: " + gguf_path + " has no built-in drafter (dspark.*); use --dspark <drafter.gguf>");
            dspark_path = gguf_path;
            dspark_prefix = "dspark.";
        }
        m.spec = !dspark_path.empty();
        if (m.spec) {   // the drafter loads after the target: hold its file size (+ its buffers) out of the auto ubatch
            FILE* df = std::fopen(dspark_path.c_str(), "rb");
            if (df) { std::fseek(df, 0, SEEK_END); m.extra_vram = (size_t) std::ftell(df) + (512ull << 20); std::fclose(df); }
            if (dspark_path == gguf_path) m.extra_vram = 1536ull << 20;   // --mtp: built in, ~0.6 GB of drafter tensors
        }
        if (!lora_path.empty()) m.extra_vram += 128ull << 20;   // the adapter tensors (a r16 q/k/v/o/gate/rel set is ~40 MB)
        load(m, gguf_path, want_ctx, threads, cpu_exps, park_copy && serve);
        if (!lora_path.empty()) load_lora(m, lora_path, lora_user_scale);
        if (m.spec) {
            DsparkTarget tgt{m.tok_embd, m.tok_norm, m.output};
            m.dspark = dspark_load(dspark_path, m.gpu, tgt, m.n_ctx, dspark_prefix);
            std::fprintf(stderr, "xolotl-inkling: DSpark drafter loaded (%s), speculative decoding on for greedy requests\n", dspark_path.c_str());
        }
    } catch (const std::exception& e) {
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    const int n_vocab = m.hp.n_vocab;

    if (dump) {
        std::vector<int32_t> ids;
        if (!ids_arg.empty() && ids_arg[0] == '@') {   // --ids @file: the comma list from a file (32K+ ids exceed argv limits)
            FILE* fi = std::fopen(ids_arg.c_str() + 1, "rb");
            if (!fi) die("cannot read " + ids_arg.substr(1));
            std::string b; char tmp[65536]; size_t r;
            while ((r = std::fread(tmp, 1, sizeof tmp, fi)) > 0) b.append(tmp, r);
            std::fclose(fi);
            ids_arg = b;
        }
        if (!parse_ids(ids_arg.c_str(), ids) || out_dir.empty()) { usage(); return 2; }
        std::vector<float> lg;
        std::vector<std::vector<float>> taps;
        try {
            const auto t0 = Clock::now();
            // XOLOTL_INKLING_DUMP_T=N: forwards of at most N tokens (1 = every logit from the one-token decode graph)
            static const int dump_t = std::getenv("XOLOTL_INKLING_DUMP_T") ? std::max(1, std::atoi(std::getenv("XOLOTL_INKLING_DUMP_T"))) : 1 << 30;
            // XOLOTL_INKLING_DUMP_TAIL=N: the last chunk is exactly the last N tokens (its taps = those N rows);
            // XOLOTL_INKLING_DUMP_LAST=1: keep/write only the final logit row (a 13K-token prompt's [T][n_vocab] is 10 GB)
            static const int64_t dump_tail = std::getenv("XOLOTL_INKLING_DUMP_TAIL") ? std::max(0, std::atoi(std::getenv("XOLOTL_INKLING_DUMP_TAIL"))) : 0;
            static const bool dump_last = std::getenv("XOLOTL_INKLING_DUMP_LAST") != nullptr;
            std::vector<int64_t> tap_rows;
            if (const char* tr = std::getenv("XOLOTL_INKLING_TAP_ROWS")) {
                std::string a = tr;
                if (!a.empty() && a[0] == '@') {
                    FILE* fi = std::fopen(a.c_str() + 1, "rb");
                    if (!fi) die("cannot read " + a.substr(1));
                    a.clear(); char tb[65536]; size_t r;
                    while ((r = std::fread(tb, 1, sizeof tb, fi)) > 0) a.append(tb, r);
                    std::fclose(fi);
                }
                for (const char* q = a.c_str(); *q;) {
                    char* e = nullptr; const long long v = std::strtoll(q, &e, 10);
                    if (e == q) { ++q; continue; }
                    if (v >= 0 && v < (long long) ids.size()) tap_rows.push_back(v);
                    q = e;
                }
                std::sort(tap_rows.begin(), tap_rows.end());
                tap_rows.erase(std::unique(tap_rows.begin(), tap_rows.end()), tap_rows.end());
            }
            auto dchunk = [&](int64_t p, int64_t left) {
                int64_t T = std::min(chunk_len(m, p, left), dump_t);
                const int64_t cut = (int64_t) ids.size() - std::min<int64_t>(dump_tail, (int64_t) ids.size());
                if (dump_tail > 0 && p < cut) T = std::min<int64_t>(T, cut - p);
                return (int) T;
            };
            const bool chunked = ids.size() > 256 || dump_t < (int) ids.size();   // long prompts: chunked like serve, logits only
            std::vector<float> dt;
            // XOLOTL_INKLING_DUMP_NLL=1: no logits file; nll.f32 [T-1] = -log p(ids[t+1] | ids[..t]) computed per chunk,
            // so a long-context perplexity run (32K+ tokens) needs no [T][n_vocab] buffer
            static const bool nll_only = std::getenv("XOLOTL_INKLING_DUMP_NLL") != nullptr;
            if (nll_only) {
                std::vector<float> part, nll;
                for (size_t p = 0; p < ids.size();) {
                    const int T = dchunk((int64_t) p, (int64_t) (ids.size() - p));
                    forward(m, ids.data() + p, T, (int64_t) p, T, part, nullptr, nullptr);
                    for (int t = 0; t < T && p + t + 1 < ids.size(); ++t) {
                        const float* r = part.data() + (size_t) t * n_vocab;
                        const float mx = *std::max_element(r, r + n_vocab);
                        double se = 0.0;
                        for (int v = 0; v < n_vocab; ++v) se += std::exp((double) r[v] - mx);
                        nll.push_back((float) (mx + std::log(se) - r[ids[p + t + 1]]));
                    }
                    p += T;
                }
                FILE* f = std::fopen((out_dir + "/nll.f32").c_str(), "wb");
                if (!f) die("cannot write " + out_dir + "/nll.f32");
                std::fwrite(nll.data(), sizeof(float), nll.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "xolotl-inkling: %zu tokens in %.0f ms (nll)\n", ids.size(), ms_since(t0));
                return 0;
            }
            if (!chunked) forward(m, ids.data(), (int) ids.size(), 0, (int) ids.size(), lg, &taps, m.dspark ? &dt : nullptr);
            else {
                std::vector<float> part;
                lg.clear();
                for (size_t p = 0; p < ids.size();) {
                    const int T = dchunk((int64_t) p, (int64_t) (ids.size() - p));
                    // the last chunk keeps its slot taps (l_out-<slot>.f32 = that chunk's T rows); with
                    // XOLOTL_INKLING_TAP_ROWS (sorted positions, "a,b,c" or @file) every chunk that holds a listed
                    // position keeps its taps and l_out-<slot>.f32 = those rows, in list order
                    const bool last = p + (size_t) T == ids.size();
                    bool want = last && tap_rows.empty();
                    if (!tap_rows.empty()) {
                        auto it = std::lower_bound(tap_rows.begin(), tap_rows.end(), (int64_t) p);
                        want = it != tap_rows.end() && *it < (int64_t) (p + T);
                    }
                    std::vector<std::vector<float>> ctaps;
                    // DUMP_LAST: one logit row per chunk, like serve's prefill (all T rows = [T][n_vocab] to the host per chunk)
                    forward(m, ids.data() + p, T, (int64_t) p, dump_last ? 1 : T, part, want ? (tap_rows.empty() ? &taps : &ctaps) : nullptr,
                            m.dspark ? &dt : nullptr);
                    if (want && !tap_rows.empty()) {
                        const int64_t E = m.hp.n_embd;
                        if (taps.empty()) taps.assign(m.hp.n_slot, {});
                        for (auto it = std::lower_bound(tap_rows.begin(), tap_rows.end(), (int64_t) p);
                             it != tap_rows.end() && *it < (int64_t) (p + T); ++it)
                            for (int sl = 0; sl < m.hp.n_slot; ++sl)
                                taps[sl].insert(taps[sl].end(), ctaps[sl].begin() + (*it - (int64_t) p) * E,
                                                ctaps[sl].begin() + (*it - (int64_t) p + 1) * E);
                    }
                    if (m.dspark) dspark_add_ctx(*m.dspark, dt.data(), T, (int64_t) p);
                    if (!dump_last) lg.insert(lg.end(), part.begin(), part.end());
                    else if (last) lg.assign(part.end() - n_vocab, part.end());
                    p += T;
                }
            }
            std::fprintf(stderr, "xolotl-inkling: %zu tokens in %.0f ms\n", ids.size(), ms_since(t0));
            auto wr = [&](const std::string& name, const std::vector<float>& v) {
                FILE* f = std::fopen((out_dir + "/" + name).c_str(), "wb");
                if (!f) die("cannot write " + out_dir + "/" + name);
                std::fwrite(v.data(), sizeof(float), v.size(), f);
                std::fclose(f);
            };
            wr("logits.f32", lg);
            if (!taps.empty()) for (int s = 0; s < m.hp.n_slot; ++s) wr("l_out-" + std::to_string(s) + ".f32", taps[s]);
            if (m.dspark && !chunked) dspark_add_ctx(*m.dspark, dt.data(), (int) ids.size(), 0);
            if (gen_n > 0 && m.dspark) {   // greedy continuation, speculative (must print the same ids as without --dspark)
                std::vector<float> row(lg.end() - n_vocab, lg.end());
                std::vector<int32_t> fed(ids.begin(), ids.end());
                SpecStats ss;
                SpecPolicy pol;
                std::printf("GEN");
                const auto t1 = Clock::now();
                int out = 0;
                bool stop = false;
                while (out < gen_n && !stop) {
                    const int32_t nx = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
                    const size_t f0 = fed.size();
                    adaptive_step(m, fed, nx, row, ss, pol);
                    for (size_t k = f0; k < fed.size() && out < gen_n; ++k) {
                        std::printf(" %d", fed[k]); ++out;
                        if (fed[k] == 200006) { stop = true; break; }
                    }
                    std::fflush(stdout);
                }
                std::printf("\n");
                const double ms = ms_since(t1);
                std::fprintf(stderr, "xolotl-inkling: %d tokens in %lld verify steps, %.1f tok/s, %.2f accepted drafts/step, "
                             "commit histogram", out, ss.steps, out * 1000.0 / ms, ss.steps ? (double) ss.accepted / ss.steps : 0.0);
                for (int k = 1; k <= kSpecT; ++k) std::fprintf(stderr, " %d:%lld", k, ss.hist[k]);
                std::fprintf(stderr, " | per step ms: draft %.2f verify %.2f post %.2f, drafted %.2f | plain steps %lld (%.2f ms), "
                             "switches %lld\n", ss.ms_draft / std::max(1LL, ss.steps), ss.ms_verify / std::max(1LL, ss.steps),
                             ss.ms_post / std::max(1LL, ss.steps), (double) ss.drafted / std::max(1LL, ss.steps), ss.plain,
                             ss.plain ? ss.ms_plain / ss.plain : 0.0, ss.switches);
            } else if (gen_n > 0) {   // greedy continuation (the parity text check)
                std::vector<float> row(lg.end() - n_vocab, lg.end());
                std::printf("GEN");
                int64_t p = (int64_t) ids.size();
                const auto t1 = Clock::now();
                for (int i = 0; i < gen_n; ++i) {
                    int32_t nx = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
                    std::printf(" %d", nx);
                    std::fflush(stdout);
                    if (nx == 200006) break;
                    forward(m, &nx, 1, p++, 1, row);
                }
                std::printf("\n");
                std::fprintf(stderr, "xolotl-inkling: %d decode steps, %.1f tok/s\n", gen_n, gen_n * 1000.0 / ms_since(t1));
            }
        } catch (const std::exception& e) { std::fprintf(stderr, "xolotl-inkling: %s\n", e.what()); return 1; }
        std::fprintf(stderr, "xolotl-inkling: wrote %zu x %d logits + %d slot dumps to %s\n", ids.size(), n_vocab, m.hp.n_slot, out_dir.c_str());
        return 0;
    }

#if XOLOTL_INKLING_STATS
    if (stats) {
        std::vector<int32_t> ids;
        int64_t n_rows = 0, n_seq = 0;
        std::vector<float> lg;
        try {
            inkling_stats::load_tokens_npy(tokens_path, ids, n_rows, n_seq);
            const int64_t rows = s_rows > 0 ? std::min(s_rows, n_rows - s_row_start) : n_rows - s_row_start;
            const int64_t seq = s_seq > 0 ? std::min(s_seq, n_seq) : n_seq;
            if (rows <= 0 || seq <= 0) die("--rows/--seq select nothing from " + tokens_path);
            for (const char* d : {"hess", "imatrix", "route", "dumps"}) {
                const std::string p = stats_out + "/" + d;
                if (system(("mkdir -p '" + p + "'").c_str()) != 0) die("cannot create " + p);
            }
            inkling_stats::Stats st;
            st.out = stats_out;
            st.per_visit = per_visit;
            st.max_gb = hess_max_gb;
            st.dump_max_gb = dump_max_gb;
            st.chunk = m.ubatch;
            st.batch_tok = s_batch;
            st.n_embd = m.hp.n_embd; st.n_q = m.hp.n_q(); st.n_ff = m.hp.n_ff_exp;
            st.n_expert = m.hp.n_expert; st.n_used = m.hp.n_used;
            st.n_phys = m.hp.n_phys; st.n_slot = m.hp.n_slot; st.n_pass = m.hp.n_pass;
            st.slot_phys.resize(m.hp.n_slot);
            for (int s = 0; s < m.hp.n_slot; ++s) st.slot_phys[s] = m.hp.phys_of(s);
            st.row_start = s_row_start; st.rows = rows; st.seq = seq;
            st.dump_eval = dump_eval;
            meta_path = meta_path.empty() ? tokens_path.substr(0, tokens_path.find_last_of('/')) + "/meta.json" : meta_path;
            st.eval_rows = inkling_stats::read_eval_rows(meta_path);
            st.init();
            st.dump_plan();
            m.st = &st;
            const std::string dump_note = st.dump_rows.empty() ? std::string("none (no held-out row in range)")
                                                               : std::to_string(st.dump_rows.size()) + " rows";
            std::fprintf(stderr, "xolotl-inkling: stats: %lld rows x %lld tokens starting at row %lld -> %s (hessians %lld GiB, "
                                 "chunk %d, eval dump %s)\n", (long long) rows, (long long) seq, (long long) s_row_start,
                         stats_out.c_str(), (long long) st.hess_gib, (int) m.ubatch, dump_note.c_str());
            const auto t0 = Clock::now();
            int64_t tok = 0;
            for (int64_t r = 0; r < rows; ++r) {
                state_reset(m);
                st.cur_row = (int) (s_row_start + r);
                st.fit_row = !st.is_eval(st.cur_row);
                st.dump_row = !st.fit_row && st.is_dumped(st.cur_row);
                (st.fit_row ? st.fit_rows_done : st.eval_rows_done) += 1;
                const int32_t* row = ids.data() + (s_row_start + r) * n_seq;
                for (int64_t p = 0; p < seq;) {
                    const int T = chunk_len(m, p, seq - p);
                    st.cur_off = p;
                    forward(m, row + p, T, p, 1, lg);
                    p += T;
                    tok += T;
                }
                std::fprintf(stderr, "\rxolotl-inkling: stats: row %lld/%lld (%.1f tok/s)   ", (long long) (r + 1),
                             (long long) rows, tok * 1000.0 / ms_since(t0));
            }
            std::fprintf(stderr, "\n");
            st.sanity();
            st.write_all();
            std::fprintf(stderr, "xolotl-inkling: stats: host ms: transpose %.0f gram %.0f fp64-add %.0f experts %.0f act-readback %.0f\n",
                         st.ms_push, st.ms_gram, st.ms_add, st.ms_exp, st.ms_get);
            const double secs = ms_since(t0) / 1000.0;
            std::printf("STATS DONE rows=%lld tokens=%lld seconds=%.1f tok_s=%.1f out=%s\n", (long long) rows, (long long) tok,
                        secs, tok / secs, stats_out.c_str());
            std::fflush(stdout);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xolotl-inkling: stats: %s\n", e.what());
            return 1;
        }
        return 0;
    }
#endif

    // stop ids: eos / eot (<|content_model_end_sampling|>, the end of the model's turn).  <|end_message|> ends one
    // message of a turn (the thinking, a tool call) and is not a stop.
    std::vector<int32_t> stop_ids = token_ids_named(m, {"<|content_model_end_sampling|>"});
    if (m.gg.key("tokenizer.ggml.eos_token_id", false) >= 0) stop_ids.push_back((int32_t) m.gg.i("tokenizer.ggml.eos_token_id"));
    if (m.gg.key("tokenizer.ggml.eot_token_id", false) >= 0) stop_ids.push_back((int32_t) m.gg.i("tokenizer.ggml.eot_token_id"));
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());
    auto is_stop = [&](int32_t t) { return std::binary_search(stop_ids.begin(), stop_ids.end(), t); };

    float* d_logits = nullptr;
    int* d_next = nullptr;
    int* d_hist = nullptr;
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_logits, (size_t) n_vocab * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_next, sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hist, (size_t) kMaxHist * sizeof(int)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    // --mmproj: the image tower, in-process (an `ENC <image> <out>` line answers `OK <n> <nx> <ny> <ms>` and writes
    // the SVE1 record the next GENI reads)
    inkling_vision::Encoder* vis = nullptr;
    if (!mmproj_path.empty()) {
        std::string verr;
        const bool use_gpu = vision_device != "cpu";
        vis = inkling_vision::load(mmproj_path, m.gpu, m.cpu, use_gpu, verr);
        if (!vis) { std::printf("ERR %s\n", verr.c_str()); std::fflush(stdout); return 1; }
        std::fprintf(stderr, "xolotl-inkling: image tower loaded (%s, %s)\n", mmproj_path.c_str(),
                     use_gpu ? "GPU per picture" : "CPU");
    }
    std::printf("INFO engine=inkling-0.1 model=%s layers=%d slots=%d n_embd=%d weights_mib=%llu host_mib=%llu kv_mib=%llu kv_type=%s-%s "
                "context=%lld ubatch=%d ring=%lld cpu_exps=%s vram_free_mib=%zu\n",
                m.arch.c_str(), m.hp.n_phys, m.hp.n_slot, m.hp.n_embd, (unsigned long long) (m.weight_bytes >> 20),
                (unsigned long long) (m.host_bytes >> 20), (unsigned long long) (m.state_bytes >> 20),
                ggml_type_name(m.kv_k), ggml_type_name(m.kv_v), (long long) m.n_ctx, m.ubatch,
                (long long) m.ring, cpu_exps.empty() ? "none" : cpu_exps.c_str(), vram_free_mib());
    std::printf("READY %lld stop\n", (long long) m.n_ctx);
    std::fflush(stdout);
    std::fprintf(stderr, "xolotl-inkling: ready - %.2f GiB of weights in VRAM (+%.2f GiB in RAM), %.2f GiB of state, context %lld\n",
                 m.weight_bytes / 1073741824.0, m.host_bytes / 1073741824.0, m.state_bytes / 1073741824.0, (long long) m.n_ctx);

    Lines in;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line == "STOP") { in.stop.store(true); continue; }
            in.push(line);
        }
        std::lock_guard<std::mutex> l(in.m);
        in.eof = true;
        in.cv.notify_all();
    });
    reader.detach();

    std::vector<int32_t> fed;     // every token the state has consumed
    // Prompt checkpoints, several (XOLOTL_INKLING_CKPTS, default 4): a thinking turn closed by the budget sends a second
    // request whose checkpoint replaced the only one, and the next turn (the history re-rendered) then resumed below it
    // and read its whole prompt again (Gauntlet 22 Nov: a 38K prompt with 0 reused).  A checkpoint at pos stands for
    // its tokens 0..pos-1; the global K/V caches are positional, so it stays valid only while cells 0..pos-1 hold those
    // tokens - ck_settle() drops any whose cells a request rewrote with something else.
    std::vector<Ckpt> cks;
    const size_t ck_cap = (size_t) std::max(1, std::getenv("XOLOTL_INKLING_CKPTS") ? std::atoi(std::getenv("XOLOTL_INKLING_CKPTS")) : 4);
    uint64_t ck_clock = 0;
    auto ck_take = [&](const std::vector<int32_t>& toks, int64_t p, bool with_ds) {
        Ckpt* t = nullptr;
        for (Ckpt& c : cks) if (c.pos == p) { t = &c; break; }
        if (!t && cks.size() < ck_cap) { cks.emplace_back(); t = &cks.back(); }
        if (!t) { t = &cks[0]; for (Ckpt& c : cks) if (c.used < t->used) t = &c; }
        ckpt_take(m, *t, p);
        t->toks.assign(toks.begin(), toks.begin() + (long) p);
        t->has_ds = with_ds && m.dspark;
        if (t->has_ds) dspark_snap(*m.dspark, t->ds);
        t->used = ++ck_clock;
    };
    // after a request wrote cells resume .. fed.size() (+ up to kSpecT rejected draft cells past it)
    auto ck_settle = [&](size_t resume) {
        std::vector<Ckpt> keep;
        for (Ckpt& c : cks) {
            const size_t q = (size_t) c.pos;
            bool ok = c.pos > 0 && (q <= resume ||
                      (q <= fed.size() && std::equal(c.toks.begin() + (long) resume, c.toks.begin() + (long) q, fed.begin() + (long) resume)));
            if (ok) keep.push_back(std::move(c));
        }
        cks.swap(keep);
    };
    // the image token: the chat template writes `<|content_image|>` then one `<|unused_200054|>` per encoder row
    const std::vector<int32_t> media_ids = token_ids_named(m, {"<|unused_200054|>"});
    auto is_media = [&](int32_t t) { return std::find(media_ids.begin(), media_ids.end(), t) != media_ids.end(); };
    size_t fed_media = SIZE_MAX;  // the first image token the state has read (its rows are not in `fed`)
    std::vector<float> lg;
    std::string line;
    const size_t need_mib = (m.weight_bytes >> 20) + (m.state_bytes >> 20) + 1200;
    auto peer = [&](const char* reserve) {
        if (peer_vram.empty()) return;
        const std::string cmd = "curl -s -m 150 -X POST '" + peer_vram + "' -H 'Content-Type: application/json' -d '{\"reserve_mib\": " +
                                reserve + "}' > /dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "xolotl-inkling: the peer's /v1/vram did not answer\n");
    };
    auto do_park = [&](const char* why) {
        if (m.parked) return;
        const auto t0 = Clock::now();
        park(m);
        peer("null");
        std::fprintf(stderr, "xolotl-inkling: parked (%s) in %.0f ms - %zu MiB of state in RAM, %zu MiB VRAM free\n", why, ms_since(t0),
                     m.host_state.size() >> 20, vram_free_mib());
    };
    auto do_unpark = [&]() {
        if (!m.parked) return;
        const auto t0 = Clock::now();
        peer(std::to_string(need_mib).c_str());
        unpark(m);
        std::fprintf(stderr, "xolotl-inkling: back on the GPU in %.0f ms\n", ms_since(t0));
    };
    for (;;) {
        bool idle = false;
        if (!in.pop(line, park_idle_s > 0 && !m.parked ? (int64_t) (park_idle_s * 1000) : 0, &idle)) {
            if (idle) {
                try { do_park("idle"); } catch (const std::exception& e) { std::fprintf(stderr, "xolotl-inkling: %s\n", e.what()); }
                continue;
            }
            break;
        }
        if (line == "QUIT") { std::fflush(stdout); std::fflush(stderr); std::_Exit(0); }
        if (line.rfind("ENC ", 0) == 0) {   // the picture, read by the in-process tower (--mmproj)
            if (!vis) { std::printf("ERR this engine was started without --mmproj\n"); std::fflush(stdout); continue; }
            std::string img, out;
            const size_t sp = line.find(' ', 4);
            if (sp == std::string::npos) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
            img = line.substr(4, sp - 4);
            out = line.substr(sp + 1);
            const auto t0 = Clock::now();
            std::string verr;
            const int rows = inkling_vision::encode(*vis, img, out, 0, verr);
            if (rows <= 0) std::printf("ERR %s\n", verr.c_str());
            else std::printf("OK %d %d 1 %.0f\n", rows, rows, ms_since(t0));
            std::fflush(stdout);
            continue;
        }
        if (line == "PARK" || line.rfind("VRAM", 0) == 0) {
            try {
                const bool want_park = line == "PARK" || std::atoll(line.c_str() + 4) > 0;
                if (want_park) do_park("asked"); else do_unpark();
                std::printf("VRAM parked=%d vram_free_mib=%zu weights_mib=%zu state_mib=%zu\n", m.parked ? 1 : 0, vram_free_mib(),
                            m.weight_bytes >> 20, m.state_bytes >> 20);
            } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); }
            std::fflush(stdout);
            continue;
        }
        try { do_unpark(); } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); std::fflush(stdout); continue; }
        in.stop.store(false);
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] <embeddings file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        std::vector<int32_t> score_ids;
        bool score_bad = false;
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") sp.top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist);
            else if (key == "penalty_repeat") sp.penalty_repeat = fv;
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
            else if (key == "score") {
                score_bad = !parse_ids(tok.c_str() + eq + 1, score_ids) || score_ids.size() > 64;
                for (const int32_t id : score_ids) score_bad = score_bad || id >= n_vocab;
            }
        }
        if (score_bad) { std::printf("ERR score: expected 1-64 token ids below the vocabulary size\n"); std::fflush(stdout); continue; }
        std::string media_path;
        if (geni) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            media_path.assign(start, (size_t) (endp - start));
            if (media_path.empty()) { std::printf("ERR bad request: GENI wants an embeddings file\n"); std::fflush(stdout); continue; }
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > m.n_ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) m.n_ctx);
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;

        // ---- the image rows (GENI): one per image token, in prompt order
        std::vector<float> media;
        std::vector<int64_t> media_row(ids.size(), -1);
        size_t first_media = ids.size();
        if (geni) {
            std::string err;
            if (!read_media(media_path, m.hp.n_embd, media, err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }
            int64_t k = 0;
            for (size_t i = 0; i < ids.size(); ++i)
                if (is_media(ids[i])) { media_row[i] = k++; first_media = std::min(first_media, i); }
            if ((size_t) k * m.hp.n_embd != media.size()) {
                std::printf("ERR the prompt has %lld image tokens but the embeddings file %zu rows\n", (long long) k, media.size() / m.hp.n_embd);
                std::fflush(stdout);
                continue;
            }
        }

        // ---- what of the state can be kept (the conv state is recurrent: append, or the checkpoint, or restart)
        size_t common = 0;
        {
            const size_t lim = std::min(fed.size(), ids.size());
            while (common < lim && fed[common] == ids[common]) ++common;
        }
        // the state's token history does not carry the encoder rows: only reuse up to the first image token and up to
        // the first media token the state already read (a new picture with the same pad ids is a different prompt)
        common = std::min(common, first_media);
        common = std::min(common, fed_media);
        size_t resume = 0;
        const DsparkSnap* ds_restored = nullptr;
        try {
            if (!fed.empty() && fed.size() < ids.size() && common == fed.size()) resume = fed.size();
            else {
                Ckpt* best = nullptr;   // the furthest checkpoint inside the new prompt
                for (Ckpt& c : cks)
                    if (c.pos > 0 && (size_t) c.pos < ids.size() && (size_t) c.pos <= common && (!best || c.pos > best->pos) &&
                        std::equal(c.toks.begin(), c.toks.end(), ids.begin())) best = &c;
                if (best) {
                    ckpt_restore(m, *best); resume = (size_t) best->pos; best->used = ++ck_clock;
                    if (best->has_ds) ds_restored = &best->ds;
                } else state_reset(m);
            }
        } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); std::fflush(stdout); continue; }
        fed.assign(ids.begin(), ids.begin() + (long) resume);
        bool spec_ok = false;
        if (m.dspark && ds_restored) { dspark_restore(*m.dspark, *ds_restored); spec_ok = true; }   // the drafter as it was there
        else spec_ok = m.dspark && dspark_rewind(*m.dspark, (int64_t) resume);
        if (m.dspark && !spec_ok) {   // the drafter is behind the target: restart both (rare: a drafter-less request ran)
            state_reset(m); dspark_reset(*m.dspark); resume = 0; fed.clear(); spec_ok = true;
        }
        std::vector<float> dt_pp;
        fed_media = first_media < ids.size() ? first_media : SIZE_MAX;

        // ---- the prompt (a chunk whose compute buffer does not fit is read in halves)
        std::vector<float> chunk_media, chunk_sel;
        std::function<void(size_t, int)> run = [&](size_t at, int T) {
            try {
                bool any = false;
                for (int t = 0; t < T && !any; ++t) any = media_row[at + t] >= 0;
                if (!any) {
                    forward(m, ids.data() + at, T, (int64_t) at, 1, lg, nullptr, spec_ok ? &dt_pp : nullptr);
                } else {
                    const int ne = m.hp.n_embd;
                    chunk_media.assign((size_t) T * ne, 0.0f);
                    chunk_sel.assign(T, 0.0f);
                    for (int t = 0; t < T; ++t)
                        if (media_row[at + t] >= 0) {
                            std::memcpy(chunk_media.data() + (size_t) t * ne, media.data() + (size_t) media_row[at + t] * ne, (size_t) ne * sizeof(float));
                            chunk_sel[t] = 1.0f;
                        }
                    forward(m, ids.data() + at, T, (int64_t) at, 1, lg, nullptr, spec_ok ? &dt_pp : nullptr,
                            chunk_media.data(), chunk_sel.data());
                }
                if (spec_ok) dspark_add_ctx(*m.dspark, dt_pp.data(), T, (int64_t) at);
            } catch (const AllocFail&) {
                if (T <= 16) throw;
                std::fprintf(stderr, "xolotl-inkling: no VRAM for a %d-token chunk at %zu (%zu MiB free): reading it in halves\n", T, at, vram_free_mib());
                run(at, T / 2);
                run(at + T / 2, T - T / 2);
            }
        };
        const auto t_prompt = Clock::now();
        bool failed = false;
        size_t p = resume;
        try {
            while (p < ids.size()) {
                const int T = chunk_len(m, (int64_t) p, (int64_t) (ids.size() - p));
                const bool last = p + T == ids.size();
                if (last && T > 1) {   // the checkpoint sits one token before the end
                    run(p, T - 1);
                    fed.insert(fed.end(), ids.begin() + (long) p, ids.begin() + (long) (p + T - 1));
                    p += T - 1;
                    ck_take(ids, (int64_t) p, spec_ok);
                    run(p, 1);
                    fed.push_back(ids[p]);
                    p += 1;
                } else {
                    if (last) ck_take(ids, (int64_t) p, spec_ok);
                    run(p, T);
                    fed.insert(fed.end(), ids.begin() + (long) p, ids.begin() + (long) (p + T));
                    p += T;
                }
                const double ms = ms_since(t_prompt);
                std::printf("PP %zu %zu %.0f %.1f\n", p, ids.size(), ms, ms > 0 ? (p - resume) * 1000.0 / ms : 0.0);
                std::fflush(stdout);
                if (in.stop.load()) break;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xolotl-inkling: %s\n", e.what());
            std::printf("ERR %s\n", e.what()); std::fflush(stdout);
            fed.clear(); cks.clear();
            failed = true;
        }
        if (failed) continue;
        const double prompt_ms = ms_since(t_prompt);
        if (in.stop.load()) {
            ck_settle(resume);
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu\n", ids.size(), prompt_ms, resume);
            std::fflush(stdout);
            continue;
        }
        if (!score_ids.empty()) {
            double mx = -INFINITY;
            for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, (double) lg[v]);
            double se = 0;
            int top = 0;
            for (int v = 0; v < n_vocab; ++v) { se += std::exp((double) lg[v] - mx); if (lg[v] > lg[top]) top = v; }
            const double lse = mx + std::log(se);
            std::printf("SCORE");
            for (const int32_t id : score_ids) std::printf(" %d:%.6f", id, (double) lg[id] - lse);
            std::printf(" top=%d:%.6f\n", top, (double) lg[top] - lse);
            std::fflush(stdout);
        }

        // ---- decode
        const auto t_dec = Clock::now();
        long long gen = 0;
        const char* why = "length";
        std::vector<int32_t> hist;
        static const bool spec_sampled = !(std::getenv("XOLOTL_SPEC_SAMPLED") && std::string(std::getenv("XOLOTL_SPEC_SAMPLED")) == "0");
        const bool spec_req = spec_ok && (sp.greedy || spec_sampled) && !sp.logit_bias &&
                              (sp.penalty_last_n <= 0 || (sp.penalty_repeat == 1.0f && sp.penalty_freq == 0.0f && sp.penalty_present == 0.0f));
        SpecStats ss;
        try {
            if (spec_req) {   // speculative decode (greedy, or exact sampled - see spec_step): one verify commits 1..kSpecT
                uint64_t draws = 0;
                std::function<int32_t(const float*)> pick = nullptr;
                if (!sp.greedy) pick = [&](const float* row) {   // the request's sampler on one row (no penalties here)
                    cudaMemcpy(d_logits, row, (size_t) n_vocab * sizeof(float), cudaMemcpyHostToDevice);
                    sp.counter = draws++;
                    int t = 0;
                    strata::kernels::sample_tokens(d_logits, 1, n_vocab, nullptr, 0, sp, d_next, nullptr);
                    cudaMemcpy(&t, d_next, sizeof(int), cudaMemcpyDeviceToHost);
                    return (int32_t) t;
                };
                int32_t carry = -1;   // a rejected verify row's draw: it IS the next token
                SpecPolicy pol;
                for (;;) {
                    const int32_t nx = carry >= 0 ? carry
                                     : pick ? pick(lg.data()) : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
                    if ((int64_t) fed.size() + 1 >= m.n_ctx) { std::printf("T %d\n", nx); ++gen; std::fflush(stdout); break; }
                    const size_t f0 = fed.size();
                    adaptive_step(m, fed, nx, lg, ss, pol, pick, &carry);
                    bool end = false;
                    for (size_t k = f0; k < fed.size(); ++k) {
                        std::printf("T %d\n", fed[k]);
                        ++gen;
                        if (is_stop(fed[k])) { why = "stop"; end = true; break; }
                        if (gen >= max_new) { end = true; break; }
                    }
                    std::fflush(stdout);
                    if (end) break;
                    if (in.stop.load()) { why = "cancel"; break; }
                }
                std::fprintf(stderr, "xolotl-inkling: spec %lld steps, %.2f accepted drafts/step, plain %lld steps, %lld switches\n",
                             ss.steps, ss.steps ? (double) ss.accepted / ss.steps : 0.0, ss.plain, ss.switches);
            } else for (;;) {
                cudaMemcpy(d_logits, lg.data(), (size_t) n_vocab * sizeof(float), cudaMemcpyHostToDevice);
                int hl = 0;
                if (sp.penalty_last_n > 0) {
                    hl = std::min<int>(sp.penalty_last_n, (int) fed.size());
                    hist.assign(fed.end() - hl, fed.end());
                    cudaMemcpy(d_hist, hist.data(), (size_t) hl * sizeof(int), cudaMemcpyHostToDevice);
                }
                sp.counter = (uint64_t) gen;
                int next = 0;
                strata::kernels::sample_tokens(d_logits, 1, n_vocab, hl ? d_hist : nullptr, hl, sp, d_next, nullptr);
                cudaMemcpy(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost);
                std::printf("T %d\n", next);
                std::fflush(stdout);
                ++gen;
                if (is_stop(next)) { why = "stop"; break; }
                if (gen >= max_new) break;
                if ((int64_t) fed.size() + 1 >= m.n_ctx) break;
                if (in.stop.load()) { why = "cancel"; break; }
                forward(m, &next, 1, (int64_t) fed.size(), 1, lg, nullptr, spec_ok ? &dt_pp : nullptr);
                if (spec_ok) dspark_add_ctx(*m.dspark, dt_pp.data(), 1, (int64_t) fed.size());   // keep the drafter in step
                fed.push_back(next);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xolotl-inkling: %s\n", e.what());
            fed.clear(); cks.clear();
            why = "cancel";
        }
        ck_settle(resume);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", gen, ids.size(), prompt_ms, ms_since(t_dec), why, resume);
        std::fflush(stdout);
    }
    return 0;
}

#include "inkling_train.cpp"   // --train (LoRA training on the served GGUF)
