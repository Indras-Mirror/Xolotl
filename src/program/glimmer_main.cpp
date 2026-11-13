// src/program/glimmer_main.cpp - `xolotl-glimmer --serve --native <muse-glimmer.gguf> [--context N] [--ubatch N] [--kv-type T]`
//
// Muse Glimmer 30B (Meta Superintelligence Labs, Apache 2.0) in Xolotl.  The forward pass is llama.cpp's Muse Glimmer
// graph (src/models/muse-glimmer.cpp at Xolotl's pinned llama.cpp 6050b467, MIT; support added upstream in #26841) built
// on ggml-backend (CUDA, with the CPU backend as the scheduler's fallback), so it reads every GGUF quant type llama.cpp
// reads.  The engine skeleton (state, prefix reuse, park/unpark, the serve protocol) is xolotl-lfm3's
// (src/program/lfm3_main.cpp).
//
// It speaks Xolotl's serve line protocol (the header of src/program/dense_main.cpp), so serve/server.py drives it
// through its unchanged StrataEngine:
//   out INFO ... | READY <context> stop | PP <pos> <prompt> <ms> <tok/s> | T <id> | SCORE ... |
//       DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in  GEN <max_new> [key=value ...] id,id,... | GENI <max_new> [key=value ...] <embeddings file> id,id,... |
//       STOP | QUIT | PARK | VRAM [mib]
//
// The model (GGUF arch `muse-glimmer`): 52 dense layers, n_embd 6656, GQA 32 query / 2 K/V heads of 128, SwiGLU FFN
// 19968.  Layers il % 4 != 3 are sliding-window (window 2048) and RoPE'd (NORM rope, base 500000); every 4th layer is a
// GLOBAL layer with NO position encoding (NoPE).  The embeddings get an unweighted RMS norm before layer 0; attention
// has QK-norm (the q_norm weight carries the model's qk scale) and a sigmoid gate per output dim (attn_gate @ the
// pre-attention normed input) between the attention and o_proj; sandwich norms whose post norms use eps 1e-8; a
// separate lm head, then logit_scale (0.196) and a tanh softcap of 20.
//
// Long context: the 13 global layers are NoPE, and llama.cpp gives the sliding layers their TRAINED rope scale
// (llama_model::get_rope_freq_scale -> rope_freq_scale_train_swa = 1.0) whatever --rope-scaling says, so YaRN changes
// nothing for this model (with freq_scale 1 the YaRN ramp is the identity and its mscale is 1).  A context past the
// trained 131072 is the global layers' K/V sized to it - what Mal's llama.cpp wrapper does at 512K.
//
// State: K/V for the 13 global layers over the whole context, K/V for the 39 sliding layers in a ring of window +
// ubatch cells (padded to 256).  A request that only appends to what the state has read resumes where it stopped;
// otherwise the state resumes from the checkpoint taken one token before the end of the previous prompt (the sliding
// rings, ~100 MiB; global K/V is positional and append-only), or restarts from token 0.
#include "strata/kernels/sampler.hpp"
#include <unordered_map>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>

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
// the compute buffer for a graph did not fit in VRAM: the caller may retry with fewer tokens (nothing has changed yet)
struct AllocFail : std::runtime_error { AllocFail() : std::runtime_error("graph allocation failed (VRAM full)") {} };

const char* ARCH = "muse-glimmer";

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
            default: die("GGUF value is not an integer");
        }
    }
    int64_t i(const std::string& k) const {
        const int64_t id = key(k);
        if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) die("GGUF key is an array: " + k);
        return as_int(gguf_get_kv_type(g, id), gguf_get_val_data(g, id));
    }
    int64_t i_or(const std::string& k, int64_t d) const { return key(k, false) < 0 ? d : i(k); }
    float f(const std::string& k) const {
        const int64_t id = key(k);
        if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
        if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(g, id);
        die("GGUF key is not a float: " + k);
    }
    float f_or(const std::string& k, float d) const { return key(k, false) < 0 ? d : f(k); }
    // an integer per layer: an array, or nothing (-> empty)
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
};

// -------------------------------------------------------------------------------------------------------- the model
struct Hp {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_swa = 0, n_vocab = 0;
    int64_t n_ctx_train = 0;
    float eps = 1e-5f, rope_base_swa = 5e5f, logit_scale = 1.0f, softcap = 0.0f;
    std::vector<bool> swa;
};

struct Layer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr, *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr, *gate = nullptr;
    ggml_tensor *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr;
    int64_t cells = 0;             // global: n_ctx; sliding: ring size
};

struct Model {
    Hp hp;
    Gguf gg;
    ggml_context* wctx = nullptr;      // weight metadata (from gguf_init, no_alloc)
    ggml_context* sctx = nullptr;      // state tensors
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, ebuf = nullptr, sbuf = nullptr;   // GPU weights, host token table, state
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor *tok_embd = nullptr, *output_norm = nullptr, *output = nullptr;
    std::vector<Layer> L;
    std::vector<ggml_tensor*> gpu_w;   // every weight but the token table (which stays in host RAM, as llama.cpp keeps it)
    int64_t n_ctx = 0;
    int ubatch = 512;
    int64_t ring = 0;                  // sliding-window ring cells
    std::vector<int64_t> ring_pos;     // position held by each ring cell (-1 = empty); one map for every sliding layer
    std::vector<uint8_t> graph_buf;
    size_t weight_bytes = 0, embd_bytes = 0, state_bytes = 0;
    // --kv-type: f16 | q8 (q8_0 K and V) | k8v4 (q8_0 K + q4_0 V; needs ggml built with GGML_CUDA_FA_ALL_QUANTS=ON)
    ggml_type kv_k = GGML_TYPE_F16, kv_v = GGML_TYPE_F16;
    // park / unpark: a pinned host image of the GPU weights, filled at load; parked, the K/V sits in host RAM and every
    // VRAM buffer is freed (weights, state, the scheduler's compute)
    uint8_t* host_w = nullptr;
    std::vector<size_t> host_w_at;     // gpu_w[i] -> offset in host_w
    std::vector<uint8_t> host_state;
    bool parked = false;
    struct Dflash* dft = nullptr;      // --dflash: the block drafter whose K/V each forward pass fills (nullptr = none)
};

// ---- DFlash block drafter (Meta's Muse-Glimmer-30B-assistant, GGUF arch "dflash"; llama.cpp src/models/dflash.cpp).
// Five sliding-window layers of the target's width.  Every target forward pass taps the residual stream at the input of
// layers dflash.target_layers (2, 14, 26, 38, 50), fuses the five rows (fc + RMS norm) and writes each drafter layer's
// K/V for those positions into the drafter's ring.  A draft is one non-causal pass over [last token, MASK x (B-1)] at
// the next B positions, reading that ring; the argmax at block rows 1..B-1 (through the target's own output matrix) is
// the proposal.  Token table and output matrix are the target's.
struct DLayer {
    ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr;
};
struct Dflash {
    std::string path;
    Gguf gg;
    ggml_context *wctx = nullptr, *sctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, sbuf = nullptr;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_swa = 0, block = 16;
    int64_t n_ctx_orig = 131072;
    float eps = 1e-5f, rope_base = 5e5f;
    int32_t mask_id = -1;
    std::vector<int> taps;             // target layers whose input rows are fused
    std::vector<bool> is_tap;          // per target layer
    ggml_tensor *fc = nullptr, *enc_norm = nullptr, *out_norm = nullptr;
    std::vector<DLayer> L;
    int64_t ring = 0;
    std::vector<int64_t> ring_pos;
    size_t weight_bytes = 0, state_bytes = 0;
};
void dflash_alloc(Model& m, Dflash& d);
void dflash_free(Dflash& d);

ggml_tensor* need(Model& m, const std::string& name) {
    ggml_tensor* t = ggml_get_tensor(m.wctx, name.c_str());
    if (!t) die("tensor missing: " + name);
    return t;
}

// every GPU weight into one VRAM buffer (load and unpark)
void alloc_gpu_weights(Model& m) {
    ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(m.gpu);
    const size_t al = ggml_backend_buft_get_alignment(bt);
    size_t total = 0;
    for (ggml_tensor* t : m.gpu_w) total += GGML_PAD(ggml_backend_buft_get_alloc_size(bt, t), al);
    m.wbuf = ggml_backend_buft_alloc_buffer(bt, total + al);
    if (!m.wbuf) die("cannot allocate the weights in VRAM (" + std::to_string(total >> 20) + " MiB)");
    ggml_backend_buffer_set_usage(m.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tallocr ta = ggml_tallocr_new(m.wbuf);
    for (ggml_tensor* t : m.gpu_w)
        if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) die(std::string("cannot place weight ") + t->name);
    m.weight_bytes = total;
}

void load(Model& m, const std::string& path, int64_t want_ctx, int threads) {
    gguf_init_params gp{true, &m.wctx};
    m.gg.g = gguf_init_from_file(path.c_str(), gp);
    if (!m.gg.g) die("cannot read GGUF " + path);
    const std::string arch = gguf_get_val_str(m.gg.g, m.gg.key("general.architecture"));
    if (arch != ARCH) die("not a Muse Glimmer GGUF (general.architecture = " + arch + ")");
    const std::string a = std::string(ARCH) + ".";
    Hp& h = m.hp;
    h.n_layer = (int) m.gg.i(a + "block_count");
    h.n_embd = (int) m.gg.i(a + "embedding_length");
    h.n_ff = (int) m.gg.i(a + "feed_forward_length");
    h.n_head = (int) m.gg.i(a + "attention.head_count");
    h.n_head_kv = (int) m.gg.i(a + "attention.head_count_kv");
    h.head_dim = (int) m.gg.i_or(a + "attention.key_length", h.n_embd / h.n_head);
    if (m.gg.i_or(a + "attention.value_length", h.head_dim) != h.head_dim) die("key_length != value_length");
    h.n_rot = (int) m.gg.i_or(a + "rope.dimension_count", h.head_dim);
    h.n_swa = (int) m.gg.i(a + "attention.sliding_window");
    h.n_ctx_train = m.gg.i(a + "context_length");
    h.eps = m.gg.f(a + "attention.layer_norm_rms_epsilon");
    // llama.cpp: rope_freq_base_train_swa defaults to the main base unless the GGUF names its own
    h.rope_base_swa = m.gg.f_or(a + "rope.freq_base_swa", m.gg.f(a + "rope.freq_base"));
    h.logit_scale = m.gg.f_or(a + "logit_scale", 1.0f);
    h.softcap = m.gg.f_or(a + "final_logit_softcapping", 0.0f);
    {   // the per-layer pattern (true = sliding), else llama.cpp's load_swa_pattern(ml, 4): every 4th layer global
        const auto pat = m.gg.arr(a + "attention.sliding_window_pattern");
        if (!pat.empty() && (int) pat.size() != h.n_layer) die("sliding_window_pattern has the wrong length");
        for (int l = 0; l < h.n_layer; ++l) h.swa.push_back(pat.empty() ? (l % 4) < 3 : pat[l] != 0);
    }
    m.n_ctx = want_ctx > 0 ? want_ctx : h.n_ctx_train;

    m.tok_embd = need(m, "token_embd.weight");
    h.n_vocab = (int) m.tok_embd->ne[1];
    m.output_norm = need(m, "output_norm.weight");
    m.output = need(m, "output.weight");
    m.L.resize(h.n_layer);
    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.attn_norm = need(m, p + "attn_norm.weight");
        y.attn_post_norm = need(m, p + "post_attention_norm.weight");
        y.ffn_norm = need(m, p + "ffn_norm.weight");
        y.ffn_post_norm = need(m, p + "post_ffw_norm.weight");
        y.wq = need(m, p + "attn_q.weight");
        y.wk = need(m, p + "attn_k.weight");
        y.wv = need(m, p + "attn_v.weight");
        y.wo = need(m, p + "attn_output.weight");
        y.q_norm = need(m, p + "attn_q_norm.weight");
        y.k_norm = need(m, p + "attn_k_norm.weight");
        y.gate = need(m, p + "attn_gate.weight");
        y.ffn_gate = need(m, p + "ffn_gate.weight");
        y.ffn_up = need(m, p + "ffn_up.weight");
        y.ffn_down = need(m, p + "ffn_down.weight");
    }
    if (h.n_embd != h.n_head * h.head_dim) std::fprintf(stderr, "xolotl-glimmer: note: n_embd %d != n_head*head_dim %d\n", h.n_embd, h.n_head * h.head_dim);

    // backends: weights on the GPU but the token table in host RAM (get_rows reads T rows of it); the CPU backend is
    // also the scheduler's fallback
    m.gpu = ggml_backend_cuda_init(0);
    if (!m.gpu) die("ggml CUDA backend failed to start");
    m.cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m.cpu, threads);
    for (ggml_tensor* t = ggml_get_first_tensor(m.wctx); t; t = ggml_get_next_tensor(m.wctx, t))
        if (t != m.tok_embd) m.gpu_w.push_back(t);
    alloc_gpu_weights(m);
    {
        ggml_backend_buffer_type_t bt = ggml_backend_cpu_buffer_type();
        m.embd_bytes = ggml_backend_buft_get_alloc_size(bt, m.tok_embd);
        m.ebuf = ggml_backend_buft_alloc_buffer(bt, m.embd_bytes + ggml_backend_buft_get_alignment(bt));
        if (!m.ebuf) die("cannot allocate the token table in RAM");
        ggml_backend_buffer_set_usage(m.ebuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(m.ebuf);
        if (ggml_tallocr_alloc(&ta, m.tok_embd) != GGML_STATUS_SUCCESS) die("cannot place the token table");
    }
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) die("cannot open " + path);
        size_t total = 0;
        for (ggml_tensor* t : m.gpu_w) total += GGML_PAD(ggml_nbytes(t), 64);
        if (cudaHostAlloc((void**) &m.host_w, total, cudaHostAllocDefault) != cudaSuccess) m.host_w = nullptr;  // no park without it
        std::vector<uint8_t> buf;
        size_t at = 0;
        const size_t base = gguf_get_data_offset(m.gg.g);
        std::vector<size_t> at_of(gguf_get_n_tensors(m.gg.g), SIZE_MAX);
        for (int64_t i = 0; i < gguf_get_n_tensors(m.gg.g); ++i) {
            ggml_tensor* t = ggml_get_tensor(m.wctx, gguf_get_tensor_name(m.gg.g, i));
            const size_t n = ggml_nbytes(t);
            uint8_t* dst = nullptr;
            if (t != m.tok_embd && m.host_w) { dst = m.host_w + at; at_of[i] = at; at += GGML_PAD(n, 64); }
            else { buf.resize(n); dst = buf.data(); }
            if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(m.gg.g, i)), SEEK_SET) != 0 || std::fread(dst, 1, n, f) != n) {
                std::fclose(f);
                die(std::string("short read on tensor ") + t->name);
            }
            ggml_backend_tensor_set(t, dst, 0, n);
        }
        std::fclose(f);
        if (m.host_w) {   // gpu_w order -> offsets
            for (ggml_tensor* t : m.gpu_w) {
                const int64_t i = gguf_find_tensor(m.gg.g, t->name);
                m.host_w_at.push_back(at_of[i]);
            }
        }
    }

    // state: K/V per layer (global = n_ctx cells, sliding = a ring of window + ubatch cells padded to 256: ggml's CUDA
    // flash attention reads K/V in 256-row tiles, llama.cpp pads its caches the same way)
    m.ring = std::min<int64_t>(m.n_ctx, (((int64_t) h.n_swa + m.ubatch + 255) / 256) * 256);
    if (const char* e = std::getenv("XOLOTL_GLIMMER_RING"); e && std::atoll(e) > 0) m.ring = std::min<int64_t>(m.n_ctx, std::atoll(e));   // diagnostics only
    m.n_ctx = ((m.n_ctx + 255) / 256) * 256;
    m.ring_pos.assign(m.ring, -1);
    ggml_init_params sp{(size_t) (2 * h.n_layer + 4) * ggml_tensor_overhead(), nullptr, true};
    m.sctx = ggml_init(sp);
    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        y.cells = h.swa[l] ? m.ring : m.n_ctx;
        y.k_cache = ggml_new_tensor_3d(m.sctx, m.kv_k, h.head_dim, h.n_head_kv, y.cells);
        y.v_cache = ggml_new_tensor_3d(m.sctx, m.kv_v, h.head_dim, h.n_head_kv, y.cells);
        ggml_format_name(y.k_cache, "k_cache.%d", l);
        ggml_format_name(y.v_cache, "v_cache.%d", l);
    }
    m.sbuf = ggml_backend_alloc_ctx_tensors(m.sctx, m.gpu);
    if (!m.sbuf) die("cannot allocate the K/V cache in VRAM (try a smaller --context or --kv-type q8)");
    ggml_backend_buffer_clear(m.sbuf, 0);
    m.state_bytes = ggml_backend_buffer_get_size(m.sbuf);

    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, 8192, false, true);
    m.graph_buf.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
}

// ------------------------------------------------------------------------------------------------------- the graph
struct Step {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *tokens = nullptr, *pos = nullptr, *mask_full = nullptr, *mask_swa = nullptr, *out_ids = nullptr, *logits = nullptr;
    ggml_tensor *media = nullptr, *media_sel = nullptr;   // image rows from the encoder, and where they go
    int64_t n_kv_full = 0;
    int64_t n_kv_swa = 0;      // sliding layers read ring cells [0, n_kv_swa): the used ones, padded to 256 (llama.cpp's n_kv)
};

ggml_tensor* rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) { return ggml_mul(c, ggml_rms_norm(c, x, eps), w); }

// src [hd, nkv, T] -> ring cells (p0 + t) % ring of dst (at most one wrap)
void ring_store(ggml_context* c, ggml_cgraph* gf, ggml_tensor* src, ggml_tensor* dst, int64_t ring, int64_t p0, int T) {
    const size_t row = dst->nb[2];
    const int64_t start = p0 % ring;
    const int64_t n1 = std::min<int64_t>(T, ring - start);
    ggml_tensor* s1 = ggml_view_3d(c, src, src->ne[0], src->ne[1], n1, src->nb[1], src->nb[2], 0);
    ggml_tensor* d1 = ggml_view_3d(c, dst, dst->ne[0], dst->ne[1], n1, dst->nb[1], dst->nb[2], (size_t) start * row);
    ggml_build_forward_expand(gf, ggml_cpy(c, s1, d1));
    if (n1 < T) {
        ggml_tensor* s2 = ggml_view_3d(c, src, src->ne[0], src->ne[1], T - n1, src->nb[1], src->nb[2], (size_t) n1 * src->nb[2]);
        ggml_tensor* d2 = ggml_view_3d(c, dst, dst->ne[0], dst->ne[1], T - n1, dst->nb[1], dst->nb[2], 0);
        ggml_build_forward_expand(gf, ggml_cpy(c, s2, d2));
    }
}

// the drafter's K/V for T target positions from the tapped residual rows (llama.cpp dflash: encoder, then injection)
void dflash_inject(ggml_context* c, ggml_cgraph* gf, const Dflash& d, const std::vector<ggml_tensor*>& taps, ggml_tensor* pos, int64_t p0, int T) {
    ggml_tensor* feat = taps[0];
    for (size_t i = 1; i < taps.size(); ++i) feat = ggml_concat(c, feat, taps[i], 0);   // [n_taps * n_embd, T]
    ggml_tensor* g = rms(c, ggml_mul_mat(c, d.fc, feat), d.enc_norm, d.eps);
    for (const DLayer& y : d.L) {
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, y.wk, g), d.head_dim, d.n_head_kv, T);
        ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, y.wv, g), d.head_dim, d.n_head_kv, T);
        k = rms(c, k, y.k_norm, d.eps);
        k = ggml_rope_ext(c, k, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, (int) d.n_ctx_orig, d.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ring_store(c, gf, k, y.k_cache, d.ring, p0, T);
        ring_store(c, gf, v, y.v_cache, d.ring, p0, T);
    }
}

// T tokens at positions p0 .. p0+T-1; logits for the last `n_out` of them
Step build(Model& m, int T, int64_t p0, int n_out, bool media) {
    const Hp& h = m.hp;
    Step s;
    ggml_init_params ip{m.graph_buf.size(), m.graph_buf.data(), true};
    s.ctx = ggml_init(ip);
    ggml_context* c = s.ctx;
    s.gf = ggml_new_graph_custom(c, 8192, false);

    s.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.tokens);
    s.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.pos);
    // global layers read cells [0, n_kv_full), padded to 256 so the graph shape holds for 256 decode steps
    s.n_kv_full = std::min<int64_t>(m.n_ctx, ((p0 + T + 255) / 256) * 256);
    s.mask_full = ggml_new_tensor_2d(c, GGML_TYPE_F16, s.n_kv_full, T); ggml_set_input(s.mask_full);
    {   // the highest ring cell in use once this step's cells are written (until the ring wraps: p0 + T)
        int64_t hi = -1;
        for (int64_t j = 0; j < m.ring; ++j) if (m.ring_pos[j] >= 0) hi = j;
        for (int t = 0; t < T; ++t) hi = std::max<int64_t>(hi, (p0 + t) % m.ring);
        s.n_kv_swa = std::min<int64_t>(m.ring, ((hi + 1 + 255) / 256) * 256);
    }
    s.mask_swa = ggml_new_tensor_2d(c, GGML_TYPE_F16, s.n_kv_swa, T); ggml_set_input(s.mask_swa);
    if (n_out < T) { s.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_out); ggml_set_input(s.out_ids); }

    ggml_tensor* cur = ggml_get_rows(c, m.tok_embd, s.tokens);   // [n_embd, T]
    if (media) {
        // the image tokens' rows come from the encoder, the rest from the table: cur + sel * (media - cur)
        s.media = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, T); ggml_set_input(s.media);
        s.media_sel = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T); ggml_set_input(s.media_sel);
        cur = ggml_add(c, cur, ggml_mul(c, ggml_sub(c, s.media, cur), s.media_sel));
    }
    cur = ggml_rms_norm(c, cur, h.eps);                           // muse-glimmer.cpp: embd_norm, no weight
    const float kq_scale = 1.0f / std::sqrt((float) h.head_dim);
    const float post_eps = 1e-8f;                                 // the post-attention / post-FFN norms
    const int nkv = h.n_head_kv;
    const int64_t n_q = (int64_t) h.head_dim * h.n_head;

    std::vector<ggml_tensor*> taps;
    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        const bool swa = h.swa[l];
        if (m.dft && m.dft->wbuf && m.dft->is_tap[l]) {
            taps.push_back(cur);
            if (taps.size() == m.dft->taps.size()) dflash_inject(c, s.gf, *m.dft, taps, s.pos, p0, T);
        }
        ggml_tensor* res = cur;
        ggml_tensor* x = rms(c, cur, y.attn_norm, h.eps);
        ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, y.wq, x), h.head_dim, h.n_head, T);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, y.wk, x), h.head_dim, nkv, T);
        ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, y.wv, x), h.head_dim, nkv, T);
        ggml_tensor* g = ggml_mul_mat(c, y.gate, x);                                  // [n_head*head_dim, T]
        q = rms(c, q, y.q_norm, h.eps);
        k = rms(c, k, y.k_norm, h.eps);
        if (swa) {   // RoPE only on the sliding layers (NORM type); the global layers are NoPE
            q = ggml_rope_ext(c, q, s.pos, nullptr, h.n_rot, GGML_ROPE_TYPE_NORMAL, (int) h.n_ctx_train, h.rope_base_swa, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_ext(c, k, s.pos, nullptr, h.n_rot, GGML_ROPE_TYPE_NORMAL, (int) h.n_ctx_train, h.rope_base_swa, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }
        // store K/V: global layers at cells p0.., sliding layers at (p0..) % ring (at most one wrap)
        const size_t row_k = y.k_cache->nb[2], row_v = y.v_cache->nb[2];
        auto store = [&](ggml_tensor* src, ggml_tensor* dst, size_t row) {
            if (!swa) {
                ggml_tensor* d = ggml_view_3d(c, dst, h.head_dim, nkv, T, dst->nb[1], dst->nb[2], (size_t) p0 * row);
                ggml_build_forward_expand(s.gf, ggml_cpy(c, src, d));
                return;
            }
            const int64_t start = p0 % m.ring;
            const int64_t n1 = std::min<int64_t>(T, m.ring - start);
            ggml_tensor* s1 = ggml_view_3d(c, src, h.head_dim, nkv, n1, src->nb[1], src->nb[2], 0);
            ggml_tensor* d1 = ggml_view_3d(c, dst, h.head_dim, nkv, n1, dst->nb[1], dst->nb[2], (size_t) start * row);
            ggml_build_forward_expand(s.gf, ggml_cpy(c, s1, d1));
            if (n1 < T) {
                ggml_tensor* s2 = ggml_view_3d(c, src, h.head_dim, nkv, T - n1, src->nb[1], src->nb[2], (size_t) n1 * src->nb[2]);
                ggml_tensor* d2 = ggml_view_3d(c, dst, h.head_dim, nkv, T - n1, dst->nb[1], dst->nb[2], 0);
                ggml_build_forward_expand(s.gf, ggml_cpy(c, s2, d2));
            }
        };
        store(k, y.k_cache, row_k);
        store(v, y.v_cache, row_v);
        const int64_t nk = swa ? s.n_kv_swa : s.n_kv_full;
        ggml_tensor* K = ggml_view_3d(c, y.k_cache, h.head_dim, nkv, nk, y.k_cache->nb[1], y.k_cache->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, y.v_cache, h.head_dim, nkv, nk, y.v_cache->nb[1], y.v_cache->nb[2], 0);
        ggml_tensor* o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, K, 0, 2, 1, 3),
                                             ggml_permute(c, V, 0, 2, 1, 3), swa ? s.mask_swa : s.mask_full, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        // o: [head_dim, n_head, T] -> [n_head*head_dim, T]; out = wo @ (o * sigmoid(gate))
        o = ggml_mul(c, ggml_reshape_2d(c, o, n_q, T), ggml_sigmoid(c, g));
        x = ggml_mul_mat(c, y.wo, o);
        x = rms(c, x, y.attn_post_norm, post_eps);
        if (l == h.n_layer - 1 && s.out_ids) {
            x = ggml_get_rows(c, x, s.out_ids);
            res = ggml_get_rows(c, res, s.out_ids);
        }
        cur = ggml_add(c, x, res);
        ggml_tensor* f = rms(c, cur, y.ffn_norm, h.eps);
        f = ggml_mul_mat(c, y.ffn_down, ggml_swiglu_split(c, ggml_mul_mat(c, y.ffn_gate, f), ggml_mul_mat(c, y.ffn_up, f)));
        f = rms(c, f, y.ffn_post_norm, post_eps);
        cur = ggml_add(c, f, cur);
    }
    cur = rms(c, cur, m.output_norm, h.eps);
    cur = ggml_mul_mat(c, m.output, cur);                          // [n_vocab, n_out]
    cur = ggml_scale(c, cur, h.logit_scale);
    if (h.softcap != 0.0f) {
        cur = ggml_scale(c, cur, 1.0f / h.softcap);
        cur = ggml_tanh(c, cur);
        cur = ggml_scale(c, cur, h.softcap);
    }
    s.logits = cur;
    ggml_set_output(s.logits);
    ggml_build_forward_expand(s.gf, s.logits);
    return s;
}

// one forward pass; host logits for the last n_out tokens.  media (optional): T x n_embd rows, used where sel[t] = 1
void forward(Model& m, const int32_t* ids, int T, int64_t p0, int n_out, std::vector<float>& logits,
             const float* media = nullptr, const float* sel = nullptr) {
    Step s = build(m, T, p0, n_out, media != nullptr);
    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, s.gf)) { ggml_free(s.ctx); throw AllocFail(); }
    ggml_backend_tensor_set(s.tokens, ids, 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = (int32_t) (p0 + t);
    ggml_backend_tensor_set(s.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    // sliding ring: the cells this step writes now hold p0+t
    for (int t = 0; t < T; ++t) m.ring_pos[(p0 + t) % m.ring] = p0 + t;
    if (m.dft && m.dft->wbuf)
        for (int t = 0; t < T; ++t) m.dft->ring_pos[(p0 + t) % m.dft->ring] = p0 + t;
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    {   // causal: cell j visible to token t when j <= p0 + t (row-wise fills: at 512K this is the big host write)
        thread_local std::vector<ggml_fp16_t> mk;
        mk.resize((size_t) s.n_kv_full * T);
        for (int t = 0; t < T; ++t) {
            ggml_fp16_t* r = mk.data() + (size_t) t * s.n_kv_full;
            const int64_t vis = std::min<int64_t>(s.n_kv_full, p0 + t + 1);
            std::fill(r, r + vis, z);
            std::fill(r + vis, r + s.n_kv_full, ninf);
        }
        ggml_backend_tensor_set(s.mask_full, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
    }
    {
        std::vector<ggml_fp16_t> mk((size_t) s.n_kv_swa * T);
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            for (int64_t j = 0; j < s.n_kv_swa; ++j) {
                const int64_t kp = m.ring_pos[j];
                // llama_hparams::is_masked_swa (standard): masked when p - kp >= n_swa
                const bool ok = kp >= 0 && kp <= p && p - kp < m.hp.n_swa;
                mk[(size_t) t * s.n_kv_swa + j] = ok ? z : ninf;
            }
        }
        ggml_backend_tensor_set(s.mask_swa, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
    }
    if (s.media) {
        ggml_backend_tensor_set(s.media, media, 0, (size_t) T * m.hp.n_embd * sizeof(float));
        ggml_backend_tensor_set(s.media_sel, sel, 0, (size_t) T * sizeof(float));
    }
    if (s.out_ids) {
        std::vector<int32_t> o(n_out);
        for (int i = 0; i < n_out; ++i) o[i] = T - n_out + i;
        ggml_backend_tensor_set(s.out_ids, o.data(), 0, o.size() * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(m.sched, s.gf) != GGML_STATUS_SUCCESS) { ggml_free(s.ctx); die("graph compute failed"); }
    logits.resize((size_t) n_out * m.hp.n_vocab);
    ggml_backend_tensor_get(s.logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_free(s.ctx);
}

// ---- one DFlash block: [last, MASK x (B-1)] at positions n..n+B-1, non-causal over the drafter's ring; the argmax of
// block rows 1..B-1 go to out.  Returns how many (0 when the compute buffer does not fit: decode just goes on).
int dflash_draft(Model& m, Dflash& d, int32_t last, int64_t n, int B, int32_t* out) {
    if (B < 2 || !d.wbuf) return 0;
    for (int i = 0; i < B; ++i) d.ring_pos[(n + i) % d.ring] = n + i;   // the block's own cells, while it runs
    auto clear_block = [&] { for (int i = 0; i < B; ++i) d.ring_pos[(n + i) % d.ring] = -1; };
    int64_t hi = -1;
    for (int64_t j = 0; j < d.ring; ++j) if (d.ring_pos[j] >= 0) hi = j;
    const int64_t n_kv = std::min<int64_t>(d.ring, ((hi + 1 + 255) / 256) * 256);
    ggml_init_params ip{m.graph_buf.size(), m.graph_buf.data(), true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(c, 2048, false);
    ggml_tensor* tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, B); ggml_set_input(tokens);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, B); ggml_set_input(pos);
    ggml_tensor* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, B); ggml_set_input(mask);
    ggml_tensor* cur = ggml_get_rows(c, m.tok_embd, tokens);                 // the target's table, no norm (as llama.cpp)
    const float kq_scale = 1.0f / std::sqrt((float) d.head_dim);
    for (const DLayer& y : d.L) {
        ggml_tensor* x = rms(c, cur, y.attn_norm, d.eps);
        ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, y.wq, x), d.head_dim, d.n_head, B);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, y.wk, x), d.head_dim, d.n_head_kv, B);
        ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, y.wv, x), d.head_dim, d.n_head_kv, B);
        q = rms(c, q, y.q_norm, d.eps);
        k = rms(c, k, y.k_norm, d.eps);
        q = ggml_rope_ext(c, q, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, (int) d.n_ctx_orig, d.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, (int) d.n_ctx_orig, d.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ring_store(c, gf, k, y.k_cache, d.ring, n, B);
        ring_store(c, gf, v, y.v_cache, d.ring, n, B);
        ggml_tensor* K = ggml_view_3d(c, y.k_cache, d.head_dim, d.n_head_kv, n_kv, y.k_cache->nb[1], y.k_cache->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, y.v_cache, d.head_dim, d.n_head_kv, n_kv, y.v_cache->nb[1], y.v_cache->nb[2], 0);
        ggml_tensor* o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, K, 0, 2, 1, 3),
                                             ggml_permute(c, V, 0, 2, 1, 3), mask, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        o = ggml_mul_mat(c, y.wo, ggml_reshape_2d(c, o, (int64_t) d.head_dim * d.n_head, B));
        ggml_tensor* ffn_inp = ggml_add(c, o, cur);
        ggml_tensor* f = rms(c, ffn_inp, y.ffn_norm, d.eps);
        f = ggml_mul_mat(c, y.ffn_down, ggml_swiglu_split(c, ggml_mul_mat(c, y.ffn_gate, f), ggml_mul_mat(c, y.ffn_up, f)));
        cur = ggml_add(c, f, ffn_inp);
    }
    cur = rms(c, cur, d.out_norm, d.eps);
    cur = ggml_view_2d(c, cur, cur->ne[0], B - 1, cur->nb[1], cur->nb[1]);   // rows 1..B-1 (the MASK slots)
    ggml_tensor* ids = ggml_argmax(c, ggml_mul_mat(c, m.output, cur));      // the target's output matrix
    ggml_set_output(ids);
    ggml_build_forward_expand(gf, ids);
    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) { ggml_free(c); clear_block(); return 0; }
    std::vector<int32_t> tk(B, d.mask_id), ps(B);
    tk[0] = last;
    for (int i = 0; i < B; ++i) ps[i] = (int32_t) (n + i);
    ggml_backend_tensor_set(tokens, tk.data(), 0, tk.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pos, ps.data(), 0, ps.size() * sizeof(int32_t));
    {   // non-causal: committed cells inside the window, and every cell of the block
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mk((size_t) n_kv * B);
        for (int i = 0; i < B; ++i) {
            const int64_t p = n + i;
            for (int64_t j = 0; j < n_kv; ++j) {
                const int64_t kp = d.ring_pos[j];
                const bool ok = kp >= 0 && (kp < n ? p - kp < d.n_swa : kp < n + B);
                mk[(size_t) i * n_kv + j] = ok ? z : ninf;
            }
        }
        ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(m.sched, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); die("drafter graph compute failed"); }
    ggml_backend_tensor_get(ids, out, 0, (size_t) (B - 1) * sizeof(int32_t));
    ggml_free(c);
    clear_block();   // noise rows, not real ones: the verify pass writes positions n.. from the target's taps
    return B - 1;
}

// ---- checkpoint: the sliding rings at the end of a prompt (global K/V is positional)
struct Ckpt {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> k, v;
    std::vector<int64_t> ring_pos;
    std::vector<std::vector<uint8_t>> dk, dv;     // the drafter's ring (--dflash)
    std::vector<int64_t> dring_pos;
};
void ckpt_take(Model& m, Ckpt& c, int64_t pos) {
    c.pos = pos;
    c.k.resize(m.hp.n_layer);
    c.v.resize(m.hp.n_layer);
    auto grab = [](ggml_tensor* t, std::vector<uint8_t>& out) { out.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, out.data(), 0, out.size()); };
    for (int l = 0; l < m.hp.n_layer; ++l)
        if (m.hp.swa[l]) { grab(m.L[l].k_cache, c.k[l]); grab(m.L[l].v_cache, c.v[l]); }
    c.ring_pos = m.ring_pos;
    if (m.dft && m.dft->wbuf) {
        Dflash& d = *m.dft;
        c.dk.resize(d.n_layer); c.dv.resize(d.n_layer);
        for (int l = 0; l < d.n_layer; ++l) { grab(d.L[l].k_cache, c.dk[l]); grab(d.L[l].v_cache, c.dv[l]); }
        c.dring_pos = d.ring_pos;
    } else c.dring_pos.clear();
}
void ckpt_restore(Model& m, const Ckpt& c) {
    for (int l = 0; l < m.hp.n_layer; ++l)
        if (m.hp.swa[l]) {
            ggml_backend_tensor_set(m.L[l].k_cache, c.k[l].data(), 0, c.k[l].size());
            ggml_backend_tensor_set(m.L[l].v_cache, c.v[l].data(), 0, c.v[l].size());
        }
    m.ring_pos = c.ring_pos;
    if (m.dft && m.dft->wbuf) {
        Dflash& d = *m.dft;
        if (c.dring_pos.size() == d.ring_pos.size()) {
            for (int l = 0; l < d.n_layer; ++l) {
                ggml_backend_tensor_set(d.L[l].k_cache, c.dk[l].data(), 0, c.dk[l].size());
                ggml_backend_tensor_set(d.L[l].v_cache, c.dv[l].data(), 0, c.dv[l].size());
            }
            d.ring_pos = c.dring_pos;
        } else std::fill(d.ring_pos.begin(), d.ring_pos.end(), -1);   // drafts get worse for a window, never wrong
    }
}
void state_reset(Model& m) {
    std::fill(m.ring_pos.begin(), m.ring_pos.end(), -1);
    if (m.dft) std::fill(m.dft->ring_pos.begin(), m.dft->ring_pos.end(), -1);
}

// ---- park / unpark
void detach(ggml_context* ctx) {
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) { t->data = nullptr; t->buffer = nullptr; }
}
void park(Model& m) {
    if (m.parked) return;
    if (!m.host_w) die("park: no pinned host copy of the weights (cudaHostAlloc failed at load)");
    m.host_state.resize(m.state_bytes);
    size_t at = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.sctx); t; t = ggml_get_next_tensor(m.sctx, t)) {
        ggml_backend_tensor_get(t, m.host_state.data() + at, 0, ggml_nbytes(t));
        at += ggml_nbytes(t);
    }
    ggml_backend_sched_free(m.sched); m.sched = nullptr;
    if (m.dft) dflash_free(*m.dft);     // reloaded from its file on unpark; its ring refills from the next prompt
    ggml_backend_buffer_free(m.sbuf); m.sbuf = nullptr;
    ggml_backend_buffer_free(m.wbuf); m.wbuf = nullptr;
    detach(m.sctx);
    for (ggml_tensor* t : m.gpu_w) { t->data = nullptr; t->buffer = nullptr; }   // the token table stays in RAM
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
    if (m.dft) dflash_alloc(m, *m.dft);
    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, 8192, false, true);
    m.parked = false;
}
// weights + ring into VRAM (load and unpark); the ring starts empty
void dflash_alloc(Model& m, Dflash& d) {
    d.wbuf = ggml_backend_alloc_ctx_tensors(d.wctx, m.gpu);
    if (!d.wbuf) die("cannot allocate the DFlash drafter in VRAM");
    ggml_backend_buffer_set_usage(d.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    d.weight_bytes = ggml_backend_buffer_get_size(d.wbuf);
    FILE* f = std::fopen(d.path.c_str(), "rb");
    if (!f) die("cannot open " + d.path);
    std::vector<uint8_t> buf;
    const size_t base = gguf_get_data_offset(d.gg.g);
    for (int64_t i = 0; i < gguf_get_n_tensors(d.gg.g); ++i) {
        ggml_tensor* t = ggml_get_tensor(d.wctx, gguf_get_tensor_name(d.gg.g, i));
        buf.resize(ggml_nbytes(t));
        if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(d.gg.g, i)), SEEK_SET) != 0 || std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            std::fclose(f);
            die(std::string("short read on drafter tensor ") + t->name);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
    }
    std::fclose(f);
    d.sbuf = ggml_backend_alloc_ctx_tensors(d.sctx, m.gpu);
    if (!d.sbuf) die("cannot allocate the DFlash drafter's K/V in VRAM");
    ggml_backend_buffer_clear(d.sbuf, 0);
    d.state_bytes = ggml_backend_buffer_get_size(d.sbuf);
    std::fill(d.ring_pos.begin(), d.ring_pos.end(), -1);
}
void dflash_free(Dflash& d) {
    if (d.wbuf) { ggml_backend_buffer_free(d.wbuf); d.wbuf = nullptr; }
    if (d.sbuf) { ggml_backend_buffer_free(d.sbuf); d.sbuf = nullptr; }
    detach(d.wctx);
    detach(d.sctx);
    std::fill(d.ring_pos.begin(), d.ring_pos.end(), -1);
}
void dflash_load(Model& m, Dflash& d, const std::string& path) {
    d.path = path;
    gguf_init_params gp{true, &d.wctx};
    d.gg.g = gguf_init_from_file(path.c_str(), gp);
    if (!d.gg.g) die("cannot read GGUF " + path);
    const std::string arch = gguf_get_val_str(d.gg.g, d.gg.key("general.architecture"));
    if (arch != "dflash") die("--dflash: not a DFlash drafter GGUF (general.architecture = " + arch + ")");
    d.n_layer = (int) d.gg.i("dflash.block_count");
    d.n_embd = (int) d.gg.i("dflash.embedding_length");
    d.n_head = (int) d.gg.i("dflash.attention.head_count");
    d.n_head_kv = (int) d.gg.i("dflash.attention.head_count_kv");
    d.head_dim = (int) d.gg.i_or("dflash.attention.key_length", d.n_embd / d.n_head);
    d.n_swa = (int) d.gg.i_or("dflash.attention.sliding_window", 0);
    d.block = (int) d.gg.i_or("dflash.block_size", 16);
    d.n_ctx_orig = d.gg.i_or("dflash.context_length", 131072);
    d.eps = d.gg.f("dflash.attention.layer_norm_rms_epsilon");
    d.rope_base = d.gg.f("dflash.rope.freq_base");
    d.mask_id = (int32_t) d.gg.i("tokenizer.ggml.mask_token_id");
    for (int64_t t : d.gg.arr("dflash.target_layers")) d.taps.push_back((int) t);
    const auto pat = d.gg.arr("dflash.attention.sliding_window_pattern");
    for (size_t l = 0; l < pat.size(); ++l)
        if (!pat[l]) die("--dflash: this drafter has a full-attention layer; only all-sliding drafters are supported");
    if (d.n_swa <= 0) die("--dflash: this drafter has no sliding window; only all-sliding drafters are supported");
    if (d.n_embd != m.hp.n_embd) die("--dflash: the drafter's width " + std::to_string(d.n_embd) + " is not the model's " + std::to_string(m.hp.n_embd));
    if (d.taps.empty()) die("--dflash: no dflash.target_layers");
    d.is_tap.assign(m.hp.n_layer, false);
    for (int t : d.taps) {
        if (t < 0 || t >= m.hp.n_layer) die("--dflash: target layer out of range");
        d.is_tap[t] = true;
    }
    if (!std::is_sorted(d.taps.begin(), d.taps.end())) die("--dflash: target layers must be ascending");
    if (d.mask_id < 0 || d.mask_id >= m.hp.n_vocab) die("--dflash: bad mask_token_id");
    auto need_d = [&](const std::string& n) { ggml_tensor* t = ggml_get_tensor(d.wctx, n.c_str()); if (!t) die("drafter tensor missing: " + n); return t; };
    d.fc = need_d("fc.weight");
    d.enc_norm = need_d("enc.output_norm.weight");
    d.out_norm = need_d("output_norm.weight");
    if (d.fc->ne[0] != (int64_t) d.taps.size() * m.hp.n_embd) die("--dflash: fc.weight does not match the target layers");
    d.L.resize(d.n_layer);
    for (int l = 0; l < d.n_layer; ++l) {
        DLayer& y = d.L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.attn_norm = need_d(p + "attn_norm.weight");
        y.wq = need_d(p + "attn_q.weight"); y.wk = need_d(p + "attn_k.weight"); y.wv = need_d(p + "attn_v.weight");
        y.wo = need_d(p + "attn_output.weight");
        y.q_norm = need_d(p + "attn_q_norm.weight"); y.k_norm = need_d(p + "attn_k_norm.weight");
        y.ffn_norm = need_d(p + "ffn_norm.weight");
        y.ffn_gate = need_d(p + "ffn_gate.weight"); y.ffn_up = need_d(p + "ffn_up.weight"); y.ffn_down = need_d(p + "ffn_down.weight");
    }
    // ring: window + the larger of a prompt chunk and a block, padded to 256 (as the target's)
    d.ring = (((int64_t) d.n_swa + std::max(m.ubatch, d.block) + 255) / 256) * 256;
    d.ring_pos.assign(d.ring, -1);
    ggml_init_params sp{(size_t) (2 * d.n_layer + 4) * ggml_tensor_overhead(), nullptr, true};
    d.sctx = ggml_init(sp);
    for (int l = 0; l < d.n_layer; ++l) {
        d.L[l].k_cache = ggml_new_tensor_3d(d.sctx, GGML_TYPE_F16, d.head_dim, d.n_head_kv, d.ring);
        d.L[l].v_cache = ggml_new_tensor_3d(d.sctx, GGML_TYPE_F16, d.head_dim, d.n_head_kv, d.ring);
        ggml_format_name(d.L[l].k_cache, "dflash_k.%d", l);
        ggml_format_name(d.L[l].v_cache, "dflash_v.%d", l);
    }
    dflash_alloc(m, d);
    m.dft = &d;
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
    // false with `timed_out` set when nothing came in `wait_ms` (0 = wait for ever)
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

// The encoder's file (tools/vision on llama.cpp's mtmd): records of int32 {tag, n, nx, ny, n_embd} + float32 [n][n_embd],
// 'SVE1' an image, in prompt order.
bool read_media(const std::string& path, int n_embd, std::vector<float>& rows, std::string& err) {
    rows.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open the embeddings file " + path; return false; }
    for (;;) {
        int32_t hd[5];
        const size_t got = std::fread(hd, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || (hd[0] != 0x31455653 && hd[0] != 0x31415653) || hd[1] <= 0 || hd[4] != n_embd) {
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

// ---- prompt lookup drafting (--ngram K): the tokens that followed the latest earlier occurrence of the last n
// tokens (n = 4, then 3, then 2) are proposed and checked in one forward pass.  Every row is sampled with the
// counter plain decode would use, and a draft is kept only when it equals that sample, so the output is the same
// tokens plain decode samples (up to batch-size rounding in the logits).
struct NgramIndex {
    static constexpr int kMin = 2, kMax = 4;
    std::unordered_map<uint64_t, int64_t> at[kMax + 1];   // hash of seq[i-n+1..i] -> i + 1 (the latest)
    size_t done = 0;                                        // n-grams ending before index `done` are indexed
    static uint64_t key(const int32_t* t, int n) {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t) n;
        for (int i = 0; i < n; ++i) { h ^= (uint32_t) t[i]; h *= 0x100000001B3ull; h ^= h >> 29; }
        return h;
    }
    void clear() { for (auto& x : at) x.clear(); done = 0; }
    // index every n-gram whose continuation index exists (ends at index <= seq.size() - 2)
    void update(const std::vector<int32_t>& seq) {
        for (; done + 1 < seq.size(); ++done)
            for (int n = kMin; n <= kMax; ++n)
                if ((int64_t) done + 1 >= n) at[n][key(seq.data() + done + 1 - n, n)] = (int64_t) done + 1;
    }
    // up to k drafts following seq's suffix
    int draft(const std::vector<int32_t>& seq, int k, int32_t* out) {
        update(seq);
        const int64_t L = (int64_t) seq.size();
        for (int n = kMax; n >= kMin; --n) {
            if (L < n + 1) continue;
            auto it = at[n].find(key(seq.data() + L - n, n));
            if (it == at[n].end()) continue;
            const int64_t c = it->second;
            if (c < n || !std::equal(seq.begin() + (c - n), seq.begin() + c, seq.begin() + (L - n))) continue;
            int got = 0;
            while (got < k && c + got < L) { out[got] = seq[c + got]; ++got; }
            if (got) return got;
        }
        return 0;
    }
};

void usage() {
    std::fprintf(stderr,
                 "usage: xolotl-glimmer --serve --native <muse-glimmer.gguf> [--context N] [--ubatch N] [--threads N]\n"
                 "       [--kv-type f16|q8|k8v4] [--park-idle SECS] [--peer-vram http://127.0.0.1:8099/v1/vram] [--ngram K]\n"
                 "       xolotl-glimmer --dump <muse-glimmer.gguf> --ids a,b,c,... --out logits.f32 [--context N] [--ubatch N]\n"
                 "                      [--stride S]   (write every S-th position's logits, and always the last)\n"
                 "  Muse Glimmer 30B behind Xolotl's server protocol; --dump writes logits for the parity check.\n"
                 "  --ngram K: draft up to K tokens by prompt lookup and check them in one pass (0 = off; a request's ngram=K\n"
                 "  overrides, up to the --ngram value).\n"
                 "  --dflash <dflash.gguf> [--dflash-max K]: Meta's DFlash block drafter (up to block_size-1 = 15 drafts a pass;\n"
                 "  a request's dflash=K lowers it, 0 turns it off; n-gram drafting is used when it is off).\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path, ids_arg, out_path;
    bool serve = false, dump = false;
    int64_t ctx = 32768;
    int ubatch = 512, threads = 8, stride = 1, ngram_k = 0, dflash_max = -1;
    std::string dflash_path;
    double park_idle_s = 0;
    std::string peer_vram, kv_type = "f16";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--serve") serve = true;
        else if (a == "--dump") { dump = true; gguf_path = val(); }
        else if (a == "--native") gguf_path = val();
        else if (a == "--context" || a == "--max-context") ctx = std::atoll(val().c_str());
        else if (a == "--ubatch") ubatch = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--kv-type") kv_type = val();
        else if (a == "--park-idle") park_idle_s = std::atof(val().c_str());
        else if (a == "--peer-vram") peer_vram = val();
        else if (a == "--ids") ids_arg = val();
        else if (a == "--out") out_path = val();
        else if (a == "--stride") stride = std::max(1, std::atoi(val().c_str()));
        else if (a == "--ngram") ngram_k = std::max(0, std::min(16, std::atoi(val().c_str())));
        else if (a == "--dflash") dflash_path = val();
        else if (a == "--dflash-max") dflash_max = std::max(0, std::atoi(val().c_str()));
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else std::fprintf(stderr, "xolotl-glimmer: ignoring unknown option %s\n", a.c_str());
    }
    if (gguf_path.empty() || (!serve && !dump)) { usage(); return 2; }
    signal(SIGPIPE, SIG_IGN);

    Model m;
    m.ubatch = std::max(16, ubatch);
    if (kv_type == "f16") {}
    else if (kv_type == "q8") m.kv_k = m.kv_v = GGML_TYPE_Q8_0;
    else if (kv_type == "k8v4") { m.kv_k = GGML_TYPE_Q8_0; m.kv_v = GGML_TYPE_Q4_0; }
    else { std::fprintf(stderr, "xolotl-glimmer: --kv-type is f16, q8 or k8v4 (not %s)\n", kv_type.c_str()); return 2; }
    try {
        load(m, gguf_path, ctx, threads);
    } catch (const std::exception& e) {
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    const int n_vocab = m.hp.n_vocab;
    Dflash dft;
    if (!dflash_path.empty() && !dump) {
        try {
            dflash_load(m, dft, dflash_path);
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            return 1;
        }
        dflash_max = dflash_max < 0 ? dft.block - 1 : std::min(dflash_max, dft.block - 1);
        std::fprintf(stderr, "xolotl-glimmer: DFlash drafter %s: %d layers, %.0f MiB + %.0f MiB ring (%lld cells), taps", dflash_path.c_str(),
                     dft.n_layer, dft.weight_bytes / 1048576.0, dft.state_bytes / 1048576.0, (long long) dft.ring);
        for (int t : dft.taps) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, ", up to %d drafts a pass\n", dflash_max);
    } else dflash_max = 0;
    if (m.n_ctx > m.hp.n_ctx_train)
        std::fprintf(stderr, "xolotl-glimmer: context %lld is past the trained %lld (the global layers are NoPE and the sliding "
                     "layers only see 2048 back, as with the llama.cpp wrapper's 512K)\n", (long long) m.n_ctx, (long long) m.hp.n_ctx_train);

    if (dump) {
        std::vector<int32_t> ids;
        if (!parse_ids(ids_arg.c_str(), ids)) { usage(); return 2; }
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
        std::vector<float> lg;
        size_t written = 0;
        try {
            for (size_t p = 0; p < ids.size(); p += m.ubatch) {
                const int T = (int) std::min<size_t>(m.ubatch, ids.size() - p);
                forward(m, ids.data() + p, T, (int64_t) p, T, lg);
                for (int t = 0; t < T; ++t) {
                    const size_t at = p + t;
                    if (at % stride == 0 || at + 1 == ids.size()) {
                        std::fwrite(lg.data() + (size_t) t * n_vocab, sizeof(float), n_vocab, f);
                        ++written;
                    }
                }
            }
        } catch (const std::exception& e) { std::fprintf(stderr, "xolotl-glimmer: %s\n", e.what()); return 1; }
        std::fclose(f);
        std::fprintf(stderr, "xolotl-glimmer: wrote %zu x %d logits to %s (stride %d)\n", written, n_vocab, out_path.c_str(), stride);
        return 0;
    }

    // stop ids: <|end_of_text|> (eos) and <|eot|> (the end of a turn).  <|eom|> ends one message of a turn (the
    // reasoning, a tool call followed by another) and is not a stop.
    std::vector<int32_t> stop_ids = token_ids_named(m, {"<|eot|>", "<|end_of_text|>"});
    if (m.gg.key("tokenizer.ggml.eos_token_id", false) >= 0) stop_ids.push_back((int32_t) m.gg.i("tokenizer.ggml.eos_token_id"));
    if (m.gg.key("tokenizer.ggml.eot_token_id", false) >= 0) stop_ids.push_back((int32_t) m.gg.i("tokenizer.ggml.eot_token_id"));
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());
    auto is_stop = [&](int32_t t) { return std::binary_search(stop_ids.begin(), stop_ids.end(), t); };

    float* d_logits = nullptr;
    int* d_next = nullptr;
    int* d_hist = nullptr;
    constexpr int kMaxHist = 4096;
    const int win_max = std::max(ngram_k, dflash_max) + 1;   // rows of one verify window
    if (cudaMalloc((void**) &d_logits, (size_t) win_max * n_vocab * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_next, (size_t) win_max * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hist, (size_t) win_max * kMaxHist * sizeof(int)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=glimmer-0.1 model=muse-glimmer layers=%d n_embd=%d weights_mib=%llu embd_ram_mib=%llu kv_mib=%llu context=%lld "
                "ubatch=%d ring=%lld kv=%s/%s ngram=%d dflash=%d vram_free_mib=%zu\n",
                m.hp.n_layer, m.hp.n_embd, (unsigned long long) (m.weight_bytes >> 20), (unsigned long long) (m.embd_bytes >> 20),
                (unsigned long long) (m.state_bytes >> 20), (long long) m.n_ctx, m.ubatch, (long long) m.ring, ggml_type_name(m.kv_k),
                ggml_type_name(m.kv_v), ngram_k, dflash_max, vram_free_mib());
    std::printf("READY %lld stop\n", (long long) m.n_ctx);
    std::fflush(stdout);
    std::fprintf(stderr, "xolotl-glimmer: ready - %.2f GiB of weights in VRAM (+%.2f GiB token table in RAM), %.2f GiB of K/V, context %lld\n",
                 m.weight_bytes / 1073741824.0, m.embd_bytes / 1073741824.0, m.state_bytes / 1073741824.0, (long long) m.n_ctx);

    Lines in;
    NgramIndex ngram_idx;
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

    // the image placeholders the server expands to one per encoder row (the chat template writes <|patch|>)
    const std::vector<int32_t> media_ids = token_ids_named(m, {"<|patch|>"});
    auto is_media = [&](int32_t t) { return std::find(media_ids.begin(), media_ids.end(), t) != media_ids.end(); };
    size_t fed_media = SIZE_MAX;  // the first media token the state has read (its rows are not in `fed`)
    std::vector<int32_t> fed;     // every token the state has consumed
    Ckpt ck;                      // the state one token before the end of the last prompt
    std::vector<float> lg;
    std::string line;
    const size_t need_mib = (m.weight_bytes >> 20) + (m.state_bytes >> 20) + 1200;
    auto peer = [&](const char* reserve) {
        if (peer_vram.empty()) return;
        const std::string cmd = "curl -s -m 150 -X POST '" + peer_vram + "' -H 'Content-Type: application/json' -d '{\"reserve_mib\": " +
                                reserve + "}' > /dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "xolotl-glimmer: the peer's /v1/vram did not answer\n");
    };
    auto do_park = [&](const char* why) {
        if (m.parked) return;
        const auto t0 = Clock::now();
        park(m);
        peer("null");
        std::fprintf(stderr, "xolotl-glimmer: parked (%s) in %.0f ms - %zu MiB of state in RAM, %zu MiB VRAM free\n", why, ms_since(t0),
                     m.host_state.size() >> 20, vram_free_mib());
    };
    auto do_unpark = [&]() {
        if (!m.parked) return;
        const auto t0 = Clock::now();
        peer(std::to_string(need_mib).c_str());
        unpark(m);
        std::fprintf(stderr, "xolotl-glimmer: back on the GPU in %.0f ms\n", ms_since(t0));
    };
    for (;;) {
        bool idle = false;
        if (!in.pop(line, park_idle_s > 0 && !m.parked ? (int64_t) (park_idle_s * 1000) : 0, &idle)) {
            if (idle) {
                try { do_park("idle"); } catch (const std::exception& e) { std::fprintf(stderr, "xolotl-glimmer: %s\n", e.what()); }
                continue;
            }
            break;
        }
        if (line == "QUIT") {
            // _Exit, not return: exit() flushes stdin too, whose lock the reader thread holds inside getline (lfm3_main.cpp)
            std::fflush(stdout); std::fflush(stderr);
            std::_Exit(0);
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
        int req_ngram = ngram_k, req_dflash = dflash_max;
        bool dflash_fixed = false;
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
            else if (key == "ngram") req_ngram = std::max(0, std::min(ngram_k, std::atoi(tok.c_str() + eq + 1)));
            else if (key == "dflash_fixed") dflash_fixed = std::atoi(tok.c_str() + eq + 1) != 0;
            else if (key == "dflash") req_dflash = std::max(0, std::min(dflash_max, std::atoi(tok.c_str() + eq + 1)));
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
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
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
        if ((long long) ids.size() + 1 > m.n_ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) m.n_ctx);
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;

        // ---- what of the state can be kept
        size_t common = 0;
        {
            const size_t lim = std::min(fed.size(), ids.size());
            while (common < lim && fed[common] == ids[common]) ++common;
            common = std::min(common, first_media);
            common = std::min(common, fed_media);
        }
        size_t resume = 0;
        try {
            if (!fed.empty() && fed.size() < ids.size() && common == fed.size()) resume = fed.size();
            else if (ck.pos > 0 && (size_t) ck.pos <= common && (size_t) ck.pos < ids.size()) { ckpt_restore(m, ck); resume = (size_t) ck.pos; }
            else state_reset(m);
        } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); std::fflush(stdout); continue; }
        fed.assign(ids.begin(), ids.begin() + (long) resume);

        // ---- the prompt
        std::vector<float> chunk_media, chunk_sel;
        auto run_one = [&](size_t at, int T) {
            bool any = false;
            for (int t = 0; t < T && !any; ++t) any = media_row[at + t] >= 0;
            if (!any) { forward(m, ids.data() + at, T, (int64_t) at, 1, lg); return; }
            const int ne = m.hp.n_embd;
            chunk_media.assign((size_t) T * ne, 0.0f);
            chunk_sel.assign(T, 0.0f);
            for (int t = 0; t < T; ++t)
                if (media_row[at + t] >= 0) {
                    std::memcpy(chunk_media.data() + (size_t) t * ne, media.data() + (size_t) media_row[at + t] * ne, (size_t) ne * sizeof(float));
                    chunk_sel[t] = 1.0f;
                }
            forward(m, ids.data() + at, T, (int64_t) at, 1, lg, chunk_media.data(), chunk_sel.data());
        };
        // a chunk whose compute buffer does not fit (another process took VRAM since the last one) is read in halves
        std::function<void(size_t, int)> run = [&](size_t at, int T) {
            try {
                run_one(at, T);
            } catch (const AllocFail&) {
                if (T <= 16) throw;
                std::fprintf(stderr, "xolotl-glimmer: no VRAM for a %d-token chunk at %zu (%zu MiB free): reading it in halves\n", T, at, vram_free_mib());
                run(at, T / 2);
                run(at + T / 2, T - T / 2);
            }
        };
        fed_media = first_media < ids.size() ? first_media : SIZE_MAX;
        const auto t_prompt = Clock::now();
        bool failed = false;
        size_t p = resume;
        try {
            while (p < ids.size()) {
                const int T = (int) std::min<size_t>(m.ubatch, ids.size() - p);
                const bool last = p + T == ids.size();
                if (last && T > 1) {
                    // the checkpoint sits one token before the end: everything but the final token, then that token
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
            std::fprintf(stderr, "xolotl-glimmer: %s\n", e.what());
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
        long long n_verify = 0, n_drafted = 0, n_kept = 0;
        // adaptive DFlash length: about 3 past the recent accepted count, doubled after a fully kept block
        // (prose keeps ~1-2 of 15, a copied passage nearly all); dflash_fixed=1 drafts req_dflash every pass
        double d_ema = 4.0;
        int d_len = std::min(req_dflash, 7);
        bool used_dflash = false;
        try {
            // `next` is sampled from `lg` unless a verify window already sampled it (have_next)
            int next = 0;
            bool have_next = false;
            std::vector<int32_t> seq;                     // fed + next: what the drafts are looked up in
            if (req_ngram > 0) { ngram_idx.clear(); seq = fed; }
            if (!m.dft) req_dflash = 0;
            std::vector<int32_t> win(win_max), picks(win_max);
            for (;;) {
                if (!have_next) {
                    cudaMemcpy(d_logits, lg.data(), (size_t) n_vocab * sizeof(float), cudaMemcpyHostToDevice);
                    int hl = 0;
                    if (sp.penalty_last_n > 0) {
                        hl = std::min<int>(sp.penalty_last_n, (int) fed.size());
                        hist.assign(fed.end() - hl, fed.end());
                        cudaMemcpy(d_hist, hist.data(), (size_t) hl * sizeof(int), cudaMemcpyHostToDevice);
                    }
                    sp.counter = (uint64_t) gen;
                    strata::kernels::sample_tokens(d_logits, 1, n_vocab, hl ? d_hist : nullptr, hl, sp, d_next, nullptr);
                    cudaMemcpy(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost);
                }
                have_next = false;
                std::printf("T %d\n", next);
                std::fflush(stdout);
                ++gen;
                if (is_stop(next)) { why = "stop"; break; }
                if (gen >= max_new) break;
                if ((int64_t) fed.size() + 1 >= m.n_ctx) break;
                if (in.stop.load()) { why = "cancel"; break; }
                // drafts: never past max_new or the context
                int k = 0;
                const long long room = std::min<long long>(max_new - gen, m.n_ctx - (long long) fed.size() - 2);
                if (req_ngram > 0) seq.push_back(next);
                used_dflash = false;
                if (req_dflash > 0 && room > 0) {
                    const int kk = (int) std::min<long long>(dflash_fixed ? req_dflash : d_len, room);
                    used_dflash = true;
                    k = dflash_draft(m, *m.dft, next, (int64_t) fed.size(), kk + 1, win.data() + 1);
                } else if (req_ngram > 0 && room > 0) {
                    k = ngram_idx.draft(seq, (int) std::min<long long>(req_ngram, room), win.data() + 1);
                }
                if (k == 0) {
                    forward(m, &next, 1, (int64_t) fed.size(), 1, lg);
                    fed.push_back(next);
                    continue;
                }
                // verify window [next, d1..dk] at fed.size()..; row t's pick follows win[t]
                const int T = k + 1;
                const int64_t p0 = (int64_t) fed.size();
                win[0] = next;
                forward(m, win.data(), T, p0, T, lg);
                cudaMemcpy(d_logits, lg.data(), (size_t) T * n_vocab * sizeof(float), cudaMemcpyHostToDevice);
                int hl = 0;
                if (sp.penalty_last_n > 0) {
                    hl = sp.penalty_last_n;
                    hist.resize((size_t) T * hl);
                    strata::kernels::penalty_rows(fed.data(), (int64_t) fed.size(), win.data(), T, hl, hist.data());
                    cudaMemcpy(d_hist, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice);
                }
                sp.counter = (uint64_t) gen;
                strata::kernels::sample_tokens(d_logits, T, n_vocab, hl ? d_hist : nullptr, hl, sp, d_next, nullptr);
                cudaMemcpy(picks.data(), d_next, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost);
                ++n_verify; n_drafted += k;
                // keep drafts while they equal the picks; win[0..a] then hold real tokens, picks[a] is the next one
                int a = 0;
                bool ended = false;
                while (a < k && picks[a] == win[a + 1]) {
                    ++a; ++n_kept;
                    std::printf("T %d\n", win[a]);
                    std::fflush(stdout);
                    ++gen;
                    seq.push_back(win[a]);
                    if (is_stop(win[a])) { why = "stop"; ended = true; break; }
                    if (gen >= max_new || in.stop.load()) { if (in.stop.load()) why = "cancel"; ended = true; break; }
                }
                fed.insert(fed.end(), win.begin(), win.begin() + a + 1);
                if (used_dflash) {
                    d_ema = 0.5 * d_ema + 0.5 * a;
                    d_len = a == k ? std::min(req_dflash, std::max(d_len, k) * 2) : std::max(3, std::min(req_dflash, (int) std::ceil(d_ema) + 3));
                }
                for (int t = a + 1; t < T; ++t) {   // rejected cells (the old ones were outside the window)
                    m.ring_pos[(p0 + t) % m.ring] = -1;
                    if (m.dft) m.dft->ring_pos[(p0 + t) % m.dft->ring] = -1;
                }
                if (ended) break;
                std::memmove(lg.data(), lg.data() + (size_t) a * n_vocab, (size_t) n_vocab * sizeof(float));
                lg.resize(n_vocab);
                next = picks[a];
                have_next = true;               // seq == fed here; `next` is pushed at the top
            }
            if (n_verify) std::fprintf(stderr, "xolotl-glimmer: %s %lld windows, %lld drafted, %lld kept (%.0f %%), %.2f tokens a pass\n", req_dflash > 0 ? "dflash" : "ngram",
                                       n_verify, n_drafted, n_kept, 100.0 * n_kept / std::max(1LL, n_drafted), 1.0 + (double) n_kept / n_verify);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xolotl-glimmer: %s\n", e.what());
            fed.clear(); ck.pos = -1;
            why = "cancel";
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", gen, ids.size(), prompt_ms, ms_since(t_dec), why, resume);
        std::fflush(stdout);
    }
    return 0;
}
