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
//   in  GEN <max_new> [key=value ...] id,id,... | STOP | QUIT | PARK | VRAM [mib]
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
#include "inkling_dspark.h"    // --dspark: the DSpark block drafter (speculative decoding)

#include <signal.h>
#include <unistd.h>

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
    ggml_tensor *q_clip = nullptr, *sink_proj = nullptr, *rel = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    ggml_tensor *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    ggml_tensor *attn_norm = nullptr, *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;
    ggml_tensor *k_conv_proj = nullptr, *v_conv_proj = nullptr;
    ggml_tensor *attn_res_query = nullptr, *ffn_res_query = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate_inp = nullptr, *probs_b = nullptr;
    ggml_tensor *gate_up_exps = nullptr, *down_exps = nullptr, *gate_up_s = nullptr, *down_s = nullptr;
    // loop controllers (loop layers only): Q, K, V, FFN
    ggml_tensor *msg_query = nullptr, *ctrl_down[4] = {}, *ctrl_gain[4] = {}, *ctrl_forget[4] = {};
};

struct Slot {                  // one cache slot
    int phys = 0;
    bool swa = true;
    ggml_type k_type = GGML_TYPE_F16, v_type = GGML_TYPE_F16;   // the full slots take --kv-type; the rings stay f16
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr, *conv = nullptr;
    ggml_tensor* conv_hist = nullptr;   // --dspark: [n_conv-1+kSpecT, n_chan] = [state ; chunk inputs] of the last verify
    int64_t cells = 0;
};

struct Model {
    Hp hp;
    Gguf gg;
    std::string arch;
    ggml_context* wctx = nullptr;
    ggml_context* sctx = nullptr;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, hbuf = nullptr, sbuf = nullptr;   // GPU weights, host weights, state
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
    int64_t last_key[7] = {-1, -1, -1, -1, -1, -1, -1};
    Dspark* dspark = nullptr;        // --dspark: the drafter (its own K/V caches); spec = conv history + taps allocated
    bool spec = false;
    struct GSlot { ggml_backend_t gpu = nullptr; ggml_backend_sched_t sched = nullptr; struct Step* last = nullptr; int64_t key[7] = {-1, -1, -1, -1, -1, -1, -1}; std::vector<uint8_t> buf; };
    std::vector<GSlot> vg;           // --dspark: one cached graph (+ its own scheduler) per verify length 2..kSpecT
    uint8_t* host_w = nullptr;       // pinned copy of the GPU weights (park / unpark)
    std::vector<size_t> host_w_at;
    std::vector<uint8_t> host_state;
    bool parked = false;
    inkling_stats::Stats* st = nullptr;   // --stats: accumulate from the graph's taps
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
    for (ggml_tensor* t = ggml_get_first_tensor(m.wctx); t; t = ggml_get_next_tensor(m.wctx, t)) {
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
        for (int64_t i = 0; i < gguf_get_n_tensors(m.gg.g); ++i) {
            ggml_tensor* t = ggml_get_tensor(m.wctx, gguf_get_tensor_name(m.gg.g, i));
            const size_t n = ggml_nbytes(t);
            uint8_t* dst = nullptr;
            if (is_host[i]) dst = (uint8_t*) t->data;                                  // straight into the host buffer
            else if (m.host_w) { dst = m.host_w + at; at_of[i] = at; at += GGML_PAD(n, 64); }
            else { buf.resize(n); dst = buf.data(); }
            if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(m.gg.g, i)), SEEK_SET) != 0 || std::fread(dst, 1, n, f) != n) {
                std::fclose(f);
                die(std::string("short read on tensor ") + t->name);
            }
            if (!is_host[i]) ggml_backend_tensor_set(t, dst, 0, n);
        }
        std::fclose(f);
        if (m.host_w)
            for (ggml_tensor* t : m.gpu_w) m.host_w_at.push_back(at_of[gguf_find_tensor(m.gg.g, t->name)]);
    }

    // state
    m.ring = std::min<int64_t>(m.n_ctx, (((int64_t) h.n_swa + m.ubatch + 255) / 256) * 256);
    m.n_ctx = ((m.n_ctx + 255) / 256) * 256;
    m.ring_pos.assign(m.ring, -1);
    ggml_init_params sp{(size_t) (4 * h.n_slot + 4) * ggml_tensor_overhead(), nullptr, true};
    m.sctx = ggml_init(sp);
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
    m.state_bytes = ggml_backend_buffer_get_size(m.sbuf);

    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, kGraphNodes, false, true);
    m.graph_buf.resize(ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false));
}

// ------------------------------------------------------------------------------------------------------- the graph
// one cell chunk of the attention window: the mask and the relative-distance index of the cells [j0, j0+cells), host-filled
struct ChunkIn {
    ggml_tensor* mask = nullptr;   // [cells, T] F32 (0 / -inf, the causal or window test)
    ggml_tensor* rp = nullptr;     // [cells, T] I32 (the bias row: relative distance, clamped to the extent)
    int64_t j0 = 0, cells = 0;
};

struct Step {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *tokens = nullptr, *posf = nullptr;
    std::vector<ChunkIn> ck_full, ck_swa;   // the visible window in cell chunks (one entry when it fits the budget)
    ggml_tensor *out_ids = nullptr, *logits = nullptr;
    ggml_tensor *idx_full = nullptr, *idx_swa = nullptr;   // the cells this step writes (set_rows: fixed pointers -> CUDA graphs)
    std::vector<ggml_tensor*> taps;   // --dump: l_out of every slot
    std::vector<ggml_tensor*> st_taps;   // --stats: [slot][ST_NKIND] the tensors to accumulate
    std::vector<ggml_tensor*> dtaps;     // --dspark: l_out of slots 4, 28, 46 (visits 4, 22, 40) + result_norm, [n_embd, T]
    bool hist = false;                   // --dspark verify chunk: every convolve() also saves [state ; inputs]
    int64_t n_kv_full = 0, n_kv_swa = 0;
};

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
    int il_cur = 0;

    // --stats: keep a tensor alive past the graph so the host can read it back after the compute
    void tap(ggml_tensor* t, int kind) {
        ggml_set_output(t);
        s.st_taps[(size_t) il_cur * inkling_stats::ST_NKIND + kind] = t;
    }

    ggml_tensor* rms(ggml_tensor* x, ggml_tensor* w) { return ggml_mul(c, ggml_rms_norm(c, x, h.eps), w); }

    // scale each row to an L2 norm of at most thr, after the optional per-channel scale
    ggml_tensor* clip(ggml_tensor* x, ggml_tensor* scale, float thr) {
        if (scale) x = ggml_mul(c, x, scale);
        ggml_tensor* n = ggml_sqrt(c, ggml_sum_rows(c, ggml_sqr(c, x)));
        n = ggml_scale_bias(c, n, 1.0f, 1e-6f);
        ggml_tensor* fac = ggml_div(c, ggml_fill(c, n, thr), n);
        return ggml_mul(c, x, ggml_clamp(c, fac, 0.0f, 1.0f));
    }

    // attention over the depth: earlier streams + cur, normed, softmax-weighted by learned queries -> [n_embd, nq, T]
    ggml_tensor* mix(const std::vector<ggml_tensor*>& blocks, ggml_tensor* cur, ggml_tensor* query) {
        const int64_t E = h.n_embd;
        ggml_tensor* st = ggml_reshape_3d(c, cur, E, 1, T);
        for (ggml_tensor* b : blocks) st = ggml_concat(c, ggml_reshape_3d(c, b, E, 1, T), st, 1);
        st = ggml_rms_norm(c, st, 1e-6f);
        const int64_t nq = ggml_n_dims(query) > 1 ? query->ne[1] : 1;
        ggml_tensor* q = ggml_repeat(c, query, ggml_new_tensor_3d(c, GGML_TYPE_F32, E, nq, T));
        ggml_tensor* sc = ggml_mul_mat(c, st, q);                                  // [nS, nq, T]
        sc = ggml_soft_max_ext(c, sc, nullptr, 1.0f / std::sqrt((float) E), 0.0f);
        return ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, st)), sc);
    }

    // causal depthwise conv (k = n_conv) with a residual; the last n_conv-1 inputs live in the slot's conv state at
    // channel offset `off`; dyn (global K/V) adds input-dependent taps per head
    ggml_tensor* convolve(ggml_tensor* x, ggml_tensor* w, ggml_tensor* dyn, ggml_tensor* state, int64_t off) {
        const int64_t K = h.n_conv, C = x->ne[0];
        ggml_tensor* past = ggml_view_3d(c, state, K - 1, C, 1, state->nb[1], state->nb[1] * C, (size_t) off * state->nb[1]);
        ggml_tensor* X = ggml_concat(c, past, ggml_transpose(c, ggml_reshape_3d(c, x, C, T, 1)), 0);   // [K-1+T, C, 1]
        {   // the last K-1 inputs back into the state
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

    void store_kv(ggml_tensor* src, ggml_tensor* dst, bool swa) {
        ggml_tensor* d2 = ggml_reshape_2d(c, dst, dst->ne[0] * dst->ne[1], dst->ne[2]);
        ggml_tensor* s2 = ggml_reshape_2d(c, src, src->ne[0] * src->ne[1], T);
        ggml_build_forward_expand(gf, ggml_set_rows(c, d2, s2, swa ? s.idx_swa : s.idx_full));
    }

    ggml_tensor* moe(ggml_tensor* x, const Layer& y) {
        const int64_t E = h.n_embd, NE = h.n_expert, k = h.n_used;
        ggml_tensor* logits = ggml_mul_mat(c, y.gate_inp, x);                    // [NE, T] (router F32)
        ggml_tensor* probs = ggml_sigmoid(c, logits);
        ggml_tensor* sel = y.probs_b ? ggml_add(c, probs, y.probs_b) : probs;
        ggml_tensor* ids = ggml_argsort_top_k(c, sel, (int) k);                  // [k, T]
        ggml_tensor* w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, NE, T), ids);   // [1, k, T]
        w = ggml_reshape_2d(c, w, k, T);
        w = ggml_div(c, w, ggml_scale_bias(c, ggml_sum_rows(c, w), 1.0f, 1e-20f));
        w = ggml_reshape_3d(c, w, 1, k, T);
        if (h.w_scale != 0.0f && h.w_scale != 1.0f) w = ggml_scale(c, w, h.w_scale);
        ggml_build_forward_expand(gf, w);
        auto exp_scale = [&](ggml_tensor* r, ggml_tensor* sc) {                   // per-expert scale of NVFP4-kept experts
            if (!sc) return r;
            ggml_tensor* s = ggml_repeat_4d(c, ggml_reshape_3d(c, sc, 1, NE, 1), 1, NE, T, 1);
            return ggml_mul(c, r, ggml_get_rows(c, s, ids));
        };
        ggml_tensor* gu = exp_scale(ggml_mul_mat_id(c, y.gate_up_exps, ggml_reshape_3d(c, x, E, 1, T), ids), y.gate_up_s);
        const int64_t F = gu->ne[0] / 2;
        ggml_tensor* g = ggml_cont(c, ggml_view_3d(c, gu, F, k, T, gu->nb[1], gu->nb[2], 0));
        ggml_tensor* u = ggml_cont(c, ggml_view_3d(c, gu, F, k, T, gu->nb[1], gu->nb[2], (size_t) F * gu->nb[0]));
        // SiTU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
        ggml_tensor* act = ggml_scale(c, ggml_tanh(c, ggml_scale(c, g, 1.0f / h.situ_beta)), h.situ_beta);
        act = ggml_mul(c, act, ggml_sigmoid(c, g));
        if (h.situ_lb > 0.0f) u = ggml_scale(c, ggml_tanh(c, ggml_scale(c, u, 1.0f / h.situ_lb)), h.situ_lb);
        act = ggml_mul(c, act, u);
        if (st) {   // --stats: the down input per (expert slot, token), the routing, and w after its normalisation
            tap(act, inkling_stats::ST_ACT);
            tap(ids, inkling_stats::ST_IDS);
            tap(w, inkling_stats::ST_W);
        }
        ggml_tensor* d = ggml_mul_mat_id(c, y.down_exps, act, ids);                      // [E, k, T]
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
        // the weighted sum over the k experts as one batched mat-vec; the per-expert down scale folds into the weights
        ggml_tensor* wk = y.down_s ? exp_scale(w, y.down_s) : w;                           // [1, k, T]
        ggml_tensor* dT = ggml_cont(c, ggml_permute(c, d, 1, 0, 2, 3));                    // [k, E, T]
        return ggml_reshape_2d(c, ggml_mul_mat(c, dT, ggml_reshape_3d(c, wk, k, 1, T)), E, T);
    }

    // one layer application: slot il, physical ip; messages (passes 2+) = [n_embd, 4, T]
    ggml_tensor* layer(int il, int ip, ggml_tensor* cur, std::vector<ggml_tensor*>& blocks, ggml_tensor* messages) {
        const Layer& y = m.L[ip];
        Slot& z = m.S[il];
        il_cur = il;
        const bool swa = h.swa[ip];
        const int64_t E = h.n_embd, nh = h.n_head, nkv = h.n_head_kv, hd = h.hd, kgqa = h.n_kgqa();
        auto message = [&](int k) { return ggml_cont(c, ggml_view_2d(c, messages, E, T, messages->nb[2], (size_t) k * messages->nb[1])); };

        ggml_tensor* inpSA = cur;
        ggml_tensor* in[3] = {cur, cur, cur};
        if (y.attn_res_query) {
            ggml_tensor* mx = mix(blocks, cur, y.attn_res_query);
            for (int j = 0; j < 3; ++j) in[j] = ggml_cont(c, ggml_view_2d(c, mx, E, T, mx->nb[2], (size_t) (swa ? 0 : j) * mx->nb[1]));
        }
        if (messages)
            for (int j = 0; j < 3; ++j) in[j] = control(in[j], message(j), y, j);
        if (swa && messages) for (int j = 0; j < 3; ++j) in[j] = rms(in[j], y.attn_norm);
        else if (swa) { in[0] = rms(in[0], y.attn_norm); in[1] = in[2] = in[0]; }
        else { in[0] = rms(in[0], y.q_inp_norm); in[1] = rms(in[1], y.k_inp_norm); in[2] = rms(in[2], y.v_inp_norm); }
        if (st) {      // --stats: the attention inputs after their norms (on SWA layers all three are one tensor)
            tap(in[0], inkling_stats::ST_Q);
            tap(in[1], inkling_stats::ST_K);
            tap(in[2], inkling_stats::ST_V);
        }

        ggml_tensor* Q = ggml_mul_mat(c, y.wq, in[0]);
        ggml_tensor* Kc = convolve(ggml_mul_mat(c, y.wk, in[1]), y.k_conv, swa ? nullptr : ggml_mul_mat(c, y.k_conv_proj, in[1]), z.conv, 0);
        ggml_tensor* Vc = convolve(ggml_mul_mat(c, y.wv, in[2]), y.v_conv, swa ? nullptr : ggml_mul_mat(c, y.v_conv_proj, in[2]), z.conv, kgqa);
        ggml_tensor* R = ggml_mul_mat(c, y.rel, in[0]);
        Q = clip(ggml_reshape_3d(c, Q, hd, nh, T), y.q_clip, h.qk_clip);
        Kc = clip(ggml_reshape_3d(c, Kc, hd, nkv, T), nullptr, h.qk_clip);
        Vc = ggml_rms_norm(c, ggml_reshape_3d(c, Vc, hd, nkv, T), h.eps);
        R = clip(ggml_reshape_3d(c, R, h.n_rel, nh, T), y.rel_clip, h.rel_clip);
        if (!swa) { Q = ggml_mul(c, Q, tau); R = ggml_mul(c, R, tau); }

        store_kv(Kc, z.k_cache, swa);
        store_kv(Vc, z.v_cache, swa);
        const std::vector<ChunkIn>& plan = swa ? s.ck_swa : s.ck_full;

        // relative bias: project R on the (zero-padded) bias table, subtract the per-query sink, gather by distance
        ggml_tensor* sinks = ggml_mul_mat(c, y.sink_proj, in[0]);                         // [nh, T]
        ggml_tensor* table = ggml_mul_mat(c, ggml_pad(c, y.rel_bias, 0, 1, 0, 0), R);       // [Ext+1, nh, T]
        table = ggml_sub(c, table, ggml_reshape_3d(c, sinks, 1, nh, T));
        table = ggml_cont(c, ggml_permute(c, table, 1, 0, 2, 3));                          // [nh, Ext+1, T]
        ggml_tensor* q = ggml_permute(c, Q, 0, 2, 1, 3);                                    // [hd, T, nh]
        // The visible window is read in cell chunks so the score tensor [cells, T, nh] and the per-head mask stay
        // bounded at 512K context. One chunk (short context) is the single-shot softmax the engine always used; more
        // than one runs an online softmax whose initial (max 0, sum 1) is exactly the zero sink logit.
        ggml_tensor *acc = nullptr, *lse = nullptr, *mx = nullptr;
        for (const ChunkIn& ci : plan) {
            ggml_tensor* kq_b = ggml_get_rows(c, table, ci.rp);                             // [nh, cells, T]
            kq_b = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, kq_b, nh, ci.cells, T, 1), 2, 0, 1, 3));   // [cells, T, nh]
            ggml_tensor* mask_h = ggml_add(c, kq_b, ci.mask);
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
                ggml_soft_max_add_sinks(kq, ggml_fill(c, ggml_new_tensor_1d(c, GGML_TYPE_F32, nh), 0.0f));   // zero sink logit
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
        ggml_tensor* o = lse ? ggml_div(c, acc, lse) : acc;                                    // [hd, T, nh]  (normalise once)
        o = ggml_cont_2d(c, ggml_permute(c, o, 0, 2, 1, 3), (int64_t) hd * nh, T);
        o = ggml_mul(c, o, ggml_sigmoid(c, ggml_mul_mat(c, y.gate, in[0])));
        if (st) tap(o, inkling_stats::ST_O);            // --stats: the o_proj input, exactly what wo consumes
        o = ggml_mul_mat(c, y.wo, o);
        o = convolve(o, y.attn_post_conv, nullptr, z.conv, 2 * kgqa);

        if (ip % h.block == 0) { blocks.push_back(inpSA); cur = o; }     // a new residual stream starts here
        else cur = ggml_add(c, inpSA, o);
        ggml_tensor* ffn_inp = cur;

        ggml_tensor* x = ggml_reshape_2d(c, mix(blocks, cur, y.ffn_res_query), E, T);
        if (messages) x = control(x, message(3), y, 3);
        x = rms(x, y.ffn_norm);
        if (st) tap(x, inkling_stats::ST_F);            // --stats: the MoE input (gate/up + the router)
        x = moe(x, y);
        x = convolve(x, y.ffn_post_conv, nullptr, z.conv, 2 * kgqa + E);
        cur = ggml_add(c, ffn_inp, x);
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

// T tokens at positions p0 .. p0+T-1; logits for the last n_out of them
Step build(Model& m, int T, int64_t p0, int n_out, bool dump, bool dspark = false) {
    const Hp& h = m.hp;
    Step s;
    ggml_init_params ip{m.graph_buf.size(), m.graph_buf.data(), true};
    s.ctx = ggml_init(ip);
    ggml_context* c = s.ctx;
    s.gf = ggml_new_graph_custom(c, kGraphNodes, false);
    if (dump) s.taps.assign(h.n_slot, nullptr);
    if (m.st) s.st_taps.assign((size_t) h.n_slot * inkling_stats::ST_NKIND, nullptr);
    if (dspark) { s.dtaps.assign(4, nullptr); s.hist = m.spec && T <= kSpecT; }

    s.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.tokens);
    s.posf = ggml_new_tensor_1d(c, GGML_TYPE_F32, T); ggml_set_input(s.posf);
    s.n_kv_full = std::min<int64_t>(m.n_ctx, ((p0 + T + 255) / 256) * 256);
    {
        int64_t hi = -1;
        for (int64_t j = 0; j < m.ring; ++j) if (m.ring_pos[j] >= 0) hi = j;
        for (int t = 0; t < T; ++t) hi = std::max<int64_t>(hi, (p0 + t) % m.ring);
        s.n_kv_swa = std::min<int64_t>(m.ring, ((hi + 1 + 255) / 256) * 256);
    }
    {   // the visible window in cell chunks; one chunk at short context == the old single mask/rel-pos pair
        auto plan = [&](int64_t nk) {
            std::vector<ChunkIn> v;
            // --stats keeps the single-shot attention it was gated with (its score tensor is a few tens of MB here)
            const int64_t C = (m.st && nk * T * h.n_head * 4 <= (256 << 20)) ? nk : attn_cell_chunk(m, T, nk, h.n_head);
            for (int64_t j0 = 0; j0 < nk; j0 += C) {
                ChunkIn ci;
                ci.j0 = j0; ci.cells = std::min(C, nk - j0);
                ci.mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, ci.cells, T); ggml_set_input(ci.mask);
                ci.rp = ggml_new_tensor_2d(c, GGML_TYPE_I32, ci.cells, T); ggml_set_input(ci.rp);
                v.push_back(ci);
            }
            return v;
        };
        s.ck_full = plan(s.n_kv_full);
        s.ck_swa = plan(s.n_kv_swa);
    }
    if (n_out < T) { s.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_out); ggml_set_input(s.out_ids); }
    s.idx_full = ggml_new_tensor_1d(c, GGML_TYPE_I64, T); ggml_set_input(s.idx_full);
    s.idx_swa = ggml_new_tensor_1d(c, GGML_TYPE_I64, T); ggml_set_input(s.idx_swa);

    G g{m, h, c, s.gf, s, T, p0};
    g.st = m.st;
    // log-scaled positions for the global layers: 1 + alpha*log(max(1, (pos + 1)/floor))
    ggml_tensor* tau = ggml_scale_bias(c, s.posf, 1.0f, 1.0f);
    tau = ggml_log(c, ggml_clamp(c, ggml_scale(c, tau, 1.0f / (float) h.log_floor), 1.0f, INFINITY));
    g.tau = ggml_reshape_3d(c, ggml_scale_bias(c, tau, h.log_alpha, 1.0f), 1, 1, T);

    ggml_tensor* cur = ggml_get_rows(c, m.tok_embd, s.tokens);
    cur = g.rms(cur, m.tok_norm);
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

// one forward pass; host logits for the last n_out tokens; taps (dump): l_out per slot, [T][n_embd] each
void drop_graph(Model& m) {
    if (m.last) { ggml_free(m.last->ctx); delete m.last; m.last = nullptr; }
    for (int64_t& k : m.last_key) k = -1;
}

void forward(Model& m, const int32_t* ids, int T, int64_t p0, int n_out, std::vector<float>& logits,
             std::vector<std::vector<float>>* taps = nullptr, std::vector<float>* dtap = nullptr) {
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
        void sw() { std::swap(m.sched, g->sched); std::swap(m.last, g->last); std::swap_ranges(m.last_key, m.last_key + 7, g->key); m.graph_buf.swap(g->buf); }
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
    const int64_t key[7] = {T, n_out, n_kv_full, n_kv_swa, taps != nullptr, m.st != nullptr, dtap != nullptr};
    const auto tb = Clock::now();
    const bool reuse = m.last && std::equal(key, key + 7, m.last_key);
    if (!reuse) {
        drop_graph(m);
        m.last = new Step(build(m, T, p0, n_out, taps != nullptr, dtap != nullptr));
        ggml_backend_sched_reset(m.sched);
        if (!ggml_backend_sched_alloc_graph(m.sched, m.last->gf)) { drop_graph(m); throw AllocFail(); }
        std::copy(key, key + 7, m.last_key);
    }
    Step& s = *m.last;
    const double ms_build = ms_since(tb);
    const double ms_alloc = 0;
    const auto tc = Clock::now();
    ggml_backend_tensor_set(s.tokens, ids, 0, (size_t) T * sizeof(int32_t));
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
    if (ggml_backend_sched_graph_compute(m.sched, s.gf) != GGML_STATUS_SUCCESS) { drop_graph(m); die("graph compute failed"); }
#if XOLOTL_INKLING_STATS
    if (m.st && !s.st_taps.empty()) stats_consume(m, s, T, p0);
#endif
    if (dbg) std::fprintf(stderr, "xolotl-inkling: T=%d nodes=%d splits=%d %s %.1f ms compute %.1f ms\n", T,
                          ggml_graph_n_nodes(s.gf), ggml_backend_sched_get_n_splits(m.sched), reuse ? "reused" : "built+alloc", ms_build + ms_alloc, ms_since(tc));
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

struct SpecStats { long long steps = 0, drafted = 0, accepted = 0; long long hist[kSpecT + 1] = {}; double ms_draft = 0, ms_verify = 0, ms_post = 0; };

// One verify step.  `fed` = committed tokens (positions 0..P-1, the drafter holds their taps); `next` = the target's
// greedy token at P, not yet fed.  Commits next + the accepted drafts to `fed`, returns how many (>= 1), and leaves
// the logits row after the last committed token in `lg` (its argmax is the next step's `next`).
int spec_step(Model& m, std::vector<int32_t>& fed, int32_t next, std::vector<float>& lg, SpecStats& ss) {
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
        if ((int32_t) (std::max_element(r, r + nv) - r) != chunk[(size_t) n]) break;
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

// the longest chunk at p whose attention scores fit the budget (global slots read p + T cells)
int chunk_len(const Model& m, int64_t p, int64_t left) {
    int T = (int) std::min<int64_t>(m.ubatch, left);
    while (T > 16 && ((p + T + 255) / 256) * 256 * (int64_t) T > m.attn_budget) T /= 2;
    return std::max(1, T);
}

// ---- checkpoint: the sliding rings + every conv state (global K/V is positional)
struct Ckpt {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> k, v, cv;
    std::vector<int64_t> ring_pos;
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

void usage() {
    std::fprintf(stderr,
                 "usage: xolotl-inkling --serve --native <inkling-2-mini.gguf> [--context N] [--ubatch N] [--threads N]\n"
                 "       [--cpu-exps L] [--kv-type f16|q8_0|k8v4] [--attn-budget N] [--park-idle SECS] [--peer-vram URL]\n"
                 "       [--no-park-copy]\n"
                 "  --kv-type quantises the full-context K/V cache (k8v4 = K q8_0, V q4_0; the sliding-window rings stay f16).\n"
                 "  --attn-budget caps one prefill chunk's n_kv*T scores and sizes the attention cell chunks (default 33554432).\n"
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

int main(int argc, char** argv) {
    std::string gguf_path, ids_arg, out_dir, cpu_exps, dspark_path;
    bool serve = false, dump = false, park_copy = true;
    ggml_type kv_k = GGML_TYPE_F16, kv_v = GGML_TYPE_F16;
    int64_t ctx = 0, attn_budget = 1 << 25;
    int ubatch = 512, threads = 8, gen_n = 0;
    double park_idle_s = 0;
    std::string peer_vram;
    // --stats
    bool stats = false, per_visit = false, with_down_hess = false, dump_eval = true;
    std::string tokens_path, stats_out, meta_path;
    int64_t s_rows = -1, s_seq = -1, s_row_start = 0;
    int s_chunk = 128, s_batch = 4096;
    double hess_max_gb = 24.0, dump_max_gb = 9.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
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
        else if (a == "--attn-budget") attn_budget = std::max<int64_t>(4096, std::atoll(val().c_str()));
        else if (a == "--park-idle") park_idle_s = std::atof(val().c_str());
        else if (a == "--peer-vram") peer_vram = val();
        else if (a == "--no-park-copy") park_copy = false;
        else if (a == "--ids") ids_arg = val();
        else if (a == "--out-dir") out_dir = val();
        else if (a == "--gen") gen_n = std::atoi(val().c_str());
        else if (a == "--dspark") dspark_path = val();
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
    if (gguf_path.empty() || (!serve && !dump && !stats)) { usage(); return 2; }
    if (stats && (tokens_path.empty() || stats_out.empty())) { usage(); return 2; }
    if (with_down_hess)
        std::fprintf(stderr, "xolotl-inkling: stats: --with-down-hess is not produced by this slice "
                             "(exp_dn_sq, the per-expert down diagonal, is); continuing with the diagonal only\n");
    signal(SIGPIPE, SIG_IGN);

    Model m;
    m.ubatch = std::max(16, stats ? std::min(ubatch, s_chunk) : ubatch);
    m.attn_budget = attn_budget;
    m.kv_k = kv_k;
    m.kv_v = kv_v;
    try {
        const int64_t want_ctx = stats ? std::max<int64_t>(s_seq > 0 ? s_seq + 16 : 0, 1024)
                             : dump ? std::max<int64_t>(ctx, 4096) : ctx;
        m.spec = !dspark_path.empty();
        load(m, gguf_path, want_ctx, threads, cpu_exps, park_copy && serve);
        if (m.spec) {
            DsparkTarget tgt{m.tok_embd, m.tok_norm, m.output};
            m.dspark = dspark_load(dspark_path, m.gpu, tgt, m.n_ctx);
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
        if (!parse_ids(ids_arg.c_str(), ids) || out_dir.empty()) { usage(); return 2; }
        std::vector<float> lg;
        std::vector<std::vector<float>> taps;
        try {
            const auto t0 = Clock::now();
            const bool chunked = ids.size() > 256;   // long prompts: chunked like serve, logits only (no slot taps)
            std::vector<float> dt;
            if (!chunked) forward(m, ids.data(), (int) ids.size(), 0, (int) ids.size(), lg, &taps, m.dspark ? &dt : nullptr);
            else {
                std::vector<float> part;
                lg.clear();
                for (size_t p = 0; p < ids.size();) {
                    const int T = chunk_len(m, (int64_t) p, (int64_t) (ids.size() - p));
                    forward(m, ids.data() + p, T, (int64_t) p, T, part, nullptr, m.dspark ? &dt : nullptr);
                    if (m.dspark) dspark_add_ctx(*m.dspark, dt.data(), T, (int64_t) p);
                    lg.insert(lg.end(), part.begin(), part.end());
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
            if (!chunked) for (int s = 0; s < m.hp.n_slot; ++s) wr("l_out-" + std::to_string(s) + ".f32", taps[s]);
            if (m.dspark && !chunked) dspark_add_ctx(*m.dspark, dt.data(), (int) ids.size(), 0);
            if (gen_n > 0 && m.dspark) {   // greedy continuation, speculative (must print the same ids as without --dspark)
                std::vector<float> row(lg.end() - n_vocab, lg.end());
                std::vector<int32_t> fed(ids.begin(), ids.end());
                SpecStats ss;
                std::printf("GEN");
                const auto t1 = Clock::now();
                int out = 0;
                bool stop = false;
                while (out < gen_n && !stop) {
                    const int32_t nx = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
                    const size_t f0 = fed.size();
                    spec_step(m, fed, nx, row, ss);
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
                std::fprintf(stderr, " | per step ms: draft %.2f verify %.2f post %.2f, drafted %.2f\n", ss.ms_draft / ss.steps,
                             ss.ms_verify / ss.steps, ss.ms_post / ss.steps, (double) ss.drafted / ss.steps);
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
    Ckpt ck;
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
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
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
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > m.n_ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) m.n_ctx);
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;

        // ---- what of the state can be kept (the conv state is recurrent: append, or the checkpoint, or restart)
        size_t common = 0;
        {
            const size_t lim = std::min(fed.size(), ids.size());
            while (common < lim && fed[common] == ids[common]) ++common;
        }
        size_t resume = 0;
        try {
            if (!fed.empty() && fed.size() < ids.size() && common == fed.size()) resume = fed.size();
            else if (ck.pos > 0 && (size_t) ck.pos <= common && (size_t) ck.pos < ids.size()) { ckpt_restore(m, ck); resume = (size_t) ck.pos; }
            else state_reset(m);
        } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); std::fflush(stdout); continue; }
        fed.assign(ids.begin(), ids.begin() + (long) resume);
        bool spec_ok = m.dspark && dspark_rewind(*m.dspark, (int64_t) resume);
        if (m.dspark && !spec_ok) {   // the drafter is behind the target: restart both (rare: a drafter-less request ran)
            state_reset(m); dspark_reset(*m.dspark); resume = 0; fed.clear(); spec_ok = true;
        }
        std::vector<float> dt_pp;

        // ---- the prompt (a chunk whose compute buffer does not fit is read in halves)
        std::function<void(size_t, int)> run = [&](size_t at, int T) {
            try {
                forward(m, ids.data() + at, T, (int64_t) at, 1, lg, nullptr, spec_ok ? &dt_pp : nullptr);
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
                    ckpt_take(m, ck, (int64_t) p);
                    run(p, 1);
                    fed.push_back(ids[p]);
                    p += 1;
                } else {
                    if (last) ckpt_take(m, ck, (int64_t) p);
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
            fed.clear(); ck.pos = -1;
            failed = true;
        }
        if (failed) continue;
        const double prompt_ms = ms_since(t_prompt);
        if (in.stop.load()) {
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
        const bool spec_req = spec_ok && sp.greedy && !sp.logit_bias &&
                              (sp.penalty_last_n <= 0 || (sp.penalty_repeat == 1.0f && sp.penalty_freq == 0.0f && sp.penalty_present == 0.0f));
        SpecStats ss;
        try {
            if (spec_req) {   // speculative greedy decode: one verify forward commits 1..kSpecT tokens
                for (;;) {
                    const int32_t nx = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
                    if ((int64_t) fed.size() + 1 >= m.n_ctx) { std::printf("T %d\n", nx); ++gen; std::fflush(stdout); break; }
                    const size_t f0 = fed.size();
                    spec_step(m, fed, nx, lg, ss);
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
                std::fprintf(stderr, "xolotl-inkling: spec %lld steps, %.2f accepted drafts/step\n", ss.steps,
                             ss.steps ? (double) ss.accepted / ss.steps : 0.0);
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
            fed.clear(); ck.pos = -1;
            why = "cancel";
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", gen, ids.size(), prompt_ms, ms_since(t_dec), why, resume);
        std::fflush(stdout);
    }
    return 0;
}
