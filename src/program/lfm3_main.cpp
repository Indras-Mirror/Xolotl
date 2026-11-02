// src/program/lfm3_main.cpp - `xolotl-lfm3 --serve --native <lfm3.gguf> [--context N] [--ubatch N] [--kv-ram]`
//
// LFM3 (Liquid AI's LFM3 / LFM3-Omni text model) in Xolotl.  The forward pass is llama.cpp's LFM3 graph
// (src/models/lfm3.cpp + src/models/lfm-base.cpp, upstream f3586e6, MIT) built on ggml-backend (CUDA, with the CPU
// backend as the scheduler's fallback), so it reads every GGUF quant type llama.cpp reads, our XQ mixes included.
//
// It speaks Xolotl's serve line protocol (the header of src/program/dense_main.cpp), so serve/server.py drives it
// through its unchanged StrataEngine:
//   out INFO ... | READY <context> stop | PP <pos> <prompt> <ms> <tok/s> | T <id> | SCORE ... |
//       DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in  GEN <max_new> [key=value ...] id,id,... | STOP | QUIT
//   speech (LFM3-Omni): GENA <max_frames> [temperature= top_k= penalty_repeat= penalty_last_n= seed= codes=1] id,id,...
//       reads the prompt (the model's "tts" template, ending in <|audio_start|>), then per audio frame
//       out A <n_samples> <base64 PCM, 16-bit signed little-endian mono at 24 kHz>  (and C <12 codes> with codes=1)
//       until the end-of-audio code, max_frames or STOP; then DONE <frames> ... as for GEN.  The depthformer + codec run
//       in the xolotl-lfm3-speech child (tools/lfm3/lfm3_speech.cpp; --speech-exe / --speech-mmproj), started at the
//       first GENA and ended when the engine parks.
//
// The model: 24 layers, 14 short-conv blocks (in_proj -> B*x -> causal depthwise conv over L_cache=7 -> C*y ->
// out_proj) and 10 attention blocks (GQA 16/4, head 128, q/k RMS norm, a sigmoid gate per head on the output; 5 full
// layers with rope base 1e6 and rope_freqs = half the dims rotated, 5 sliding-window layers, window 256, base 1e4),
// sandwich norms, SwiGLU FFN, tied output.
//
// State: K/V for the full layers over the whole context, K/V for the sliding layers in a ring of window + ubatch
// cells, and a [L_cache-1, n_embd] conv state per conv layer.  --kv-ram keeps the K/V in host RAM (attention then
// runs on the CPU backend, like llama.cpp's -nkvo).  A request that only appends to what the state has read resumes
// where it stopped; otherwise the state resumes from the checkpoint taken at the end of the previous prompt (conv
// state + the sliding rings, a few MiB), or restarts from token 0.
#include "strata/kernels/sampler.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>

#include <signal.h>
#include <sys/wait.h>
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
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void die(const std::string& m) { throw std::runtime_error(m); }

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
    // an integer per layer: an array, or one scalar for every layer
    std::vector<int64_t> per_layer(const std::string& k, int n) const {
        const int64_t id = key(k);
        std::vector<int64_t> out;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) { out.assign(n, i(k)); return out; }
        const gguf_type t = gguf_get_arr_type(g, id);
        const size_t sz = t == GGUF_TYPE_UINT8 || t == GGUF_TYPE_INT8 || t == GGUF_TYPE_BOOL ? 1
                        : t == GGUF_TYPE_UINT16 || t == GGUF_TYPE_INT16 ? 2
                        : t == GGUF_TYPE_UINT64 || t == GGUF_TYPE_INT64 ? 8 : 4;
        const auto* d = (const uint8_t*) gguf_get_arr_data(g, id);
        for (size_t j = 0; j < gguf_get_arr_n(g, id); ++j) out.push_back(as_int(t, d + j * sz));
        if ((int) out.size() != n) die("GGUF array " + k + " has " + std::to_string(out.size()) + " entries, want " + std::to_string(n));
        return out;
    }
};

// -------------------------------------------------------------------------------------------------------- the model
struct Hp {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, head_dim = 0, n_rot = 0, n_swa = 0, l_cache = 0, n_vocab = 0;
    int64_t n_ctx_train = 0;
    float eps = 1e-5f, rope_base = 1e6f, rope_base_swa = 1e4f;
    std::vector<int> n_head_kv;   // 0 = conv layer
    std::vector<bool> swa;
    bool is_conv(int l) const { return n_head_kv[l] == 0; }
};

struct Layer {
    ggml_tensor *attn_norm = nullptr, *ffn_norm = nullptr, *attn_post_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr, *gate = nullptr;
    ggml_tensor *conv = nullptr, *in_proj = nullptr, *out_proj = nullptr;
    // state
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr, *conv_state = nullptr;
    int cells = 0;                 // full: n_ctx; sliding: ring size
};

struct Model {
    Hp hp;
    Gguf gg;
    ggml_context* wctx = nullptr;      // weight metadata (from gguf_init, no_alloc)
    ggml_context* sctx = nullptr;      // state tensors
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, sbuf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor *tok_embd = nullptr, *output_norm = nullptr, *rope_freqs = nullptr;
    std::vector<Layer> L;
    int64_t n_ctx = 0;
    int ubatch = 512;
    int ring = 0;                      // sliding-window ring cells
    std::vector<int64_t> ring_pos;     // position held by each ring cell (-1 = empty); one map for every sliding layer
    std::vector<uint8_t> graph_buf;
    size_t weight_bytes = 0, state_bytes = 0;
    bool kv_ram = false;
    // --kv-type: the K/V cache types (f16 | q8 = q8_0/q8_0 | k8v4 = q8_0 K + q4_0 V; k8v4 needs ggml built with
    // GGML_CUDA_FA_ALL_QUANTS=ON for the mixed flash-attention kernel).  Only the 5 full-attention layers hold n_ctx cells.
    ggml_type kv_k = GGML_TYPE_F16, kv_v = GGML_TYPE_F16;
    // park / unpark (Mal: the GPU only while it works): a pinned host image of the weights, filled at load; parked,
    // the K/V + conv state sit in host RAM and every VRAM buffer is freed (weights, state, the scheduler's compute)
    uint8_t* host_w = nullptr;
    std::vector<std::pair<ggml_tensor*, size_t>> host_w_at;   // tensor -> offset in host_w
    std::vector<uint8_t> host_state;
    bool parked = false;
    int threads = 8;
};

ggml_tensor* need(Model& m, const std::string& name) {
    ggml_tensor* t = ggml_get_tensor(m.wctx, name.c_str());
    if (!t) die("tensor missing: " + name);
    return t;
}
ggml_tensor* maybe(Model& m, const std::string& name) { return ggml_get_tensor(m.wctx, name.c_str()); }

void load(Model& m, const std::string& path, int64_t want_ctx, int threads) {
    gguf_init_params gp{true, &m.wctx};
    m.gg.g = gguf_init_from_file(path.c_str(), gp);
    if (!m.gg.g) die("cannot read GGUF " + path);
    const int64_t aid = m.gg.key("general.architecture");
    const std::string arch = gguf_get_val_str(m.gg.g, aid);
    if (arch != "lfm3") die("not an LFM3 GGUF (general.architecture = " + arch + ")");
    Hp& h = m.hp;
    h.n_layer = (int) m.gg.i("lfm3.block_count");
    h.n_embd = (int) m.gg.i("lfm3.embedding_length");
    h.n_ff = (int) m.gg.i("lfm3.feed_forward_length");
    h.n_head = (int) m.gg.i("lfm3.attention.head_count");
    for (int64_t v : m.gg.per_layer("lfm3.attention.head_count_kv", h.n_layer)) h.n_head_kv.push_back((int) v);
    for (int64_t v : m.gg.per_layer("lfm3.attention.sliding_window_pattern", h.n_layer)) h.swa.push_back(v != 0);
    h.head_dim = (int) m.gg.i_or("lfm3.attention.key_length", h.n_embd / h.n_head);
    h.n_rot = (int) m.gg.i_or("lfm3.rope.dimension_count", h.head_dim);
    h.n_swa = (int) m.gg.i("lfm3.attention.sliding_window");
    h.l_cache = (int) m.gg.i("lfm3.shortconv.l_cache");
    h.n_ctx_train = m.gg.i("lfm3.context_length");
    h.eps = m.gg.f("lfm3.attention.layer_norm_rms_epsilon");
    h.rope_base = m.gg.f("lfm3.rope.freq_base");
    h.rope_base_swa = m.gg.f("lfm3.rope.freq_base_swa");
    m.n_ctx = want_ctx > 0 ? std::min<int64_t>(want_ctx, h.n_ctx_train) : h.n_ctx_train;

    m.tok_embd = need(m, "token_embd.weight");
    h.n_vocab = (int) m.tok_embd->ne[1];
    m.output_norm = need(m, "output_norm.weight");
    m.rope_freqs = maybe(m, "rope_freqs.weight");
    m.L.resize(h.n_layer);
    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.attn_norm = need(m, p + "attn_norm.weight");
        y.ffn_norm = need(m, p + "ffn_norm.weight");
        y.attn_post_norm = maybe(m, p + "post_attention_norm.weight");
        y.ffn_post_norm = maybe(m, p + "post_ffw_norm.weight");
        y.ffn_gate = need(m, p + "ffn_gate.weight");
        y.ffn_up = need(m, p + "ffn_up.weight");
        y.ffn_down = need(m, p + "ffn_down.weight");
        if (h.is_conv(l)) {
            y.conv = need(m, p + "shortconv.conv.weight");
            y.in_proj = need(m, p + "shortconv.in_proj.weight");
            y.out_proj = need(m, p + "shortconv.out_proj.weight");
        } else {
            y.wq = need(m, p + "attn_q.weight");
            y.wk = need(m, p + "attn_k.weight");
            y.wv = need(m, p + "attn_v.weight");
            y.wo = need(m, p + "attn_output.weight");
            y.q_norm = need(m, p + "attn_q_norm.weight");
            y.k_norm = need(m, p + "attn_k_norm.weight");
            y.gate = need(m, p + "attn_gate.weight");
        }
    }

    // backends: weights on the GPU; the CPU backend is the scheduler's fallback (and holds the K/V with --kv-ram)
    m.gpu = ggml_backend_cuda_init(0);
    if (!m.gpu) die("ggml CUDA backend failed to start");
    m.cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(m.cpu, threads);
    m.wbuf = ggml_backend_alloc_ctx_tensors(m.wctx, m.gpu);
    if (!m.wbuf) die("cannot allocate the weights in VRAM");
    ggml_backend_buffer_set_usage(m.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.weight_bytes = ggml_backend_buffer_get_size(m.wbuf);
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) die("cannot open " + path);
        std::vector<uint8_t> buf;
        size_t total = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(m.gg.g); ++i)
            total += GGML_PAD(ggml_nbytes(ggml_get_tensor(m.wctx, gguf_get_tensor_name(m.gg.g, i))), 64);
        if (cudaHostAlloc((void**) &m.host_w, total, cudaHostAllocDefault) != cudaSuccess) m.host_w = nullptr;  // no park without it
        size_t at = 0;
        const size_t base = gguf_get_data_offset(m.gg.g);
        for (int64_t i = 0; i < gguf_get_n_tensors(m.gg.g); ++i) {
            ggml_tensor* t = ggml_get_tensor(m.wctx, gguf_get_tensor_name(m.gg.g, i));
            const size_t n = ggml_nbytes(t);
            buf.resize(n);
            if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(m.gg.g, i)), SEEK_SET) != 0 || std::fread(buf.data(), 1, n, f) != n) {
                std::fclose(f);
                die(std::string("short read on tensor ") + t->name);
            }
            ggml_backend_tensor_set(t, buf.data(), 0, n);
            if (m.host_w) {
                std::memcpy(m.host_w + at, buf.data(), n);
                m.host_w_at.push_back({t, at});
                at += GGML_PAD(n, 64);
            }
        }
        std::fclose(f);
    }

    // state: K/V per attention layer (full = n_ctx cells, sliding = a ring), a conv state per conv layer
    // diagnostics only: XOLOTL_LFM3_NSWA overrides the window (parity experiments; never set it when serving)
    if (const char* e = std::getenv("XOLOTL_LFM3_NSWA"); e && std::atoi(e) > 0) h.n_swa = std::atoi(e);
    // window + ubatch cells, padded to 256: ggml's CUDA flash attention reads K/V in 256-row tiles (llama.cpp pads its
    // caches the same way); a 320-cell ring at ubatch 64 read past the ring and broke parity from position 0
    m.ring = std::min<int64_t>(m.n_ctx, (((int64_t) h.n_swa + m.ubatch + 255) / 256) * 256);
    if (const char* e = std::getenv("XOLOTL_LFM3_RING"); e && std::atoi(e) > 0) m.ring = std::min<int64_t>(m.n_ctx, std::atoi(e));   // diagnostics only
    m.ring_pos.assign(m.ring, -1);
    ggml_init_params sp{(size_t) (3 * h.n_layer + 4) * ggml_tensor_overhead(), nullptr, true};
    m.sctx = ggml_init(sp);
    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        if (h.is_conv(l)) {
            y.conv_state = ggml_new_tensor_2d(m.sctx, GGML_TYPE_F32, h.l_cache - 1, h.n_embd);
            ggml_format_name(y.conv_state, "conv_state.%d", l);
        } else {
            y.cells = h.swa[l] ? m.ring : (int) m.n_ctx;
            y.k_cache = ggml_new_tensor_3d(m.sctx, m.kv_k, h.head_dim, h.n_head_kv[l], y.cells);
            y.v_cache = ggml_new_tensor_3d(m.sctx, m.kv_v, h.head_dim, h.n_head_kv[l], y.cells);
            ggml_format_name(y.k_cache, "k_cache.%d", l);
            ggml_format_name(y.v_cache, "v_cache.%d", l);
        }
    }
    if (m.kv_ram) die("--kv-ram: not in this build yet");
    m.sbuf = ggml_backend_alloc_ctx_tensors(m.sctx, m.gpu);
    if (!m.sbuf) die("cannot allocate the K/V cache in VRAM (try a smaller --context)");
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
    ggml_tensor *media = nullptr, *media_sel = nullptr;   // image/audio rows from the encoders, and where they go
    ggml_tensor* hidden = nullptr;                        // the output-normed hidden state (speech: the depthformer's input)
    int64_t n_kv_full = 0;
    int64_t n_kv_swa = 0;      // sliding layers read ring cells [0, n_kv_swa): the used ones, padded to 256 (llama.cpp's n_kv)
};

ggml_tensor* rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) { return ggml_mul(c, ggml_rms_norm(c, x, eps), w); }

// T tokens at positions p0 .. p0+T-1; logits for the last `n_out` of them
Step build(Model& m, int T, int64_t p0, int n_out, bool media, bool want_h = false) {
    const Hp& h = m.hp;
    Step s;
    ggml_init_params ip{m.graph_buf.size(), m.graph_buf.data(), true};
    s.ctx = ggml_init(ip);
    ggml_context* c = s.ctx;
    s.gf = ggml_new_graph_custom(c, 8192, false);

    s.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.tokens);
    s.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(s.pos);
    // full layers read cells [0, n_kv_full), padded to 256 so the graph shape holds for 256 decode steps
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
        // the media tokens' rows come from the encoder (mtmd's LFM3 vision / audio towers), the rest from the table:
        // cur + sel * (media - cur), sel = 1 on a media token, 0 elsewhere
        s.media = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, T); ggml_set_input(s.media);
        s.media_sel = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T); ggml_set_input(s.media_sel);
        cur = ggml_add(c, cur, ggml_mul(c, ggml_sub(c, s.media, cur), s.media_sel));
    }
    const float kq_scale = 1.0f / std::sqrt((float) h.head_dim);
    const int64_t d_conv = h.l_cache - 1;

    for (int l = 0; l < h.n_layer; ++l) {
        Layer& y = m.L[l];
        ggml_tensor* res = cur;
        ggml_tensor* x = rms(c, cur, y.attn_norm, h.eps);
        if (h.is_conv(l)) {
            // lfm-base.cpp build_lfm_shortconv (one sequence)
            ggml_tensor* bcx = ggml_mul_mat(c, y.in_proj, x);                     // [3*n_embd, T]
            const int64_t cs = bcx->ne[0] / 3;
            ggml_tensor* b = ggml_view_2d(c, bcx, cs, T, bcx->nb[1], 0);
            ggml_tensor* cc = ggml_view_2d(c, bcx, cs, T, bcx->nb[1], 1 * cs * ggml_element_size(bcx));
            ggml_tensor* xx = ggml_view_2d(c, bcx, cs, T, bcx->nb[1], 2 * cs * ggml_element_size(bcx));
            ggml_tensor* bx = ggml_transpose(c, ggml_mul(c, b, xx));               // [T, n_embd]
            bx = ggml_reshape_3d(c, ggml_cont(c, bx), T, h.n_embd, 1);
            ggml_tensor* st = ggml_reshape_3d(c, y.conv_state, d_conv, h.n_embd, 1);
            bx = ggml_concat(c, st, bx, 0);                                        // [d_conv+T, n_embd, 1]
            ggml_tensor* last = ggml_view_3d(c, bx, d_conv, h.n_embd, 1, bx->nb[1], bx->nb[2], (bx->ne[0] - d_conv) * ggml_element_size(bx));
            ggml_build_forward_expand(s.gf, ggml_cpy(c, last, ggml_reshape_3d(c, y.conv_state, d_conv, h.n_embd, 1)));
            ggml_tensor* co = ggml_ssm_conv(c, bx, y.conv);                        // [n_embd, T, 1]
            ggml_tensor* yy = ggml_mul(c, ggml_reshape_3d(c, ggml_cont(c, cc), cs, T, 1), co);
            x = ggml_reshape_2d(c, ggml_mul_mat(c, y.out_proj, yy), h.n_embd, T);
        } else {
            // lfm3.cpp build_attn_block
            const int nkv = h.n_head_kv[l];
            const bool swa = h.swa[l];
            ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, y.wq, x), h.head_dim, h.n_head, T);
            ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, y.wk, x), h.head_dim, nkv, T);
            ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, y.wv, x), h.head_dim, nkv, T);
            q = rms(c, q, y.q_norm, h.eps);
            k = rms(c, k, y.k_norm, h.eps);
            ggml_tensor* rf = swa ? nullptr : m.rope_freqs;
            const float base = swa ? h.rope_base_swa : h.rope_base;
            q = ggml_rope_ext(c, q, s.pos, rf, h.n_rot, GGML_ROPE_TYPE_NEOX, (int) h.n_ctx_train, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_ext(c, k, s.pos, rf, h.n_rot, GGML_ROPE_TYPE_NEOX, (int) h.n_ctx_train, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            // store K/V: full layers at cells p0.., sliding layers at (p0..) % ring (at most one wrap)
            const size_t row = y.k_cache->nb[2];
            auto store = [&](ggml_tensor* src, ggml_tensor* dst) {
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
            store(k, y.k_cache);
            store(v, y.v_cache);
            const int64_t nk = swa ? s.n_kv_swa : s.n_kv_full;
            ggml_tensor* K = ggml_view_3d(c, y.k_cache, h.head_dim, nkv, nk, y.k_cache->nb[1], y.k_cache->nb[2], 0);
            ggml_tensor* V = ggml_view_3d(c, y.v_cache, h.head_dim, nkv, nk, y.v_cache->nb[1], y.v_cache->nb[2], 0);
            ggml_tensor* o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, K, 0, 2, 1, 3),
                                                 ggml_permute(c, V, 0, 2, 1, 3), swa ? s.mask_swa : s.mask_full, kq_scale, 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            // o: [head_dim, n_head, T]
            ggml_tensor* g = ggml_sigmoid(c, ggml_mul_mat(c, y.gate, x));         // [n_head, T]
            o = ggml_mul(c, o, ggml_reshape_3d(c, g, 1, h.n_head, T));
            x = ggml_mul_mat(c, y.wo, ggml_reshape_2d(c, o, h.head_dim * h.n_head, T));
        }
        if (y.attn_post_norm) x = rms(c, x, y.attn_post_norm, h.eps);
        if (l == h.n_layer - 1 && s.out_ids) {
            x = ggml_get_rows(c, x, s.out_ids);
            res = ggml_get_rows(c, res, s.out_ids);
        }
        cur = ggml_add(c, res, x);
        ggml_tensor* f = rms(c, cur, y.ffn_norm, h.eps);
        f = ggml_mul_mat(c, y.ffn_down, ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, y.ffn_gate, f)), ggml_mul_mat(c, y.ffn_up, f)));
        if (y.ffn_post_norm) f = rms(c, f, y.ffn_post_norm, h.eps);
        cur = ggml_add(c, cur, f);
    }
    cur = rms(c, cur, m.output_norm, h.eps);
    if (want_h) {                                                                 // llama.cpp's t_embd (result_norm)
        s.hidden = cur;
        ggml_set_output(s.hidden);
        ggml_build_forward_expand(s.gf, s.hidden);
    }
    s.logits = ggml_mul_mat(c, m.tok_embd, cur);                                  // [n_vocab, n_out]
    ggml_set_output(s.logits);
    ggml_build_forward_expand(s.gf, s.logits);
    return s;
}

// one forward pass; host logits for the last n_out tokens
// media (optional): T x n_embd rows, used where sel[t] = 1
// hidden (optional): the last n_out rows of the output-normed hidden state
void forward(Model& m, const int32_t* ids, int T, int64_t p0, int n_out, std::vector<float>& logits,
             const float* media = nullptr, const float* sel = nullptr, std::vector<float>* hidden = nullptr) {
    Step s = build(m, T, p0, n_out, media != nullptr, hidden != nullptr);
    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, s.gf)) { ggml_free(s.ctx); die("graph allocation failed"); }
    ggml_backend_tensor_set(s.tokens, ids, 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = (int32_t) (p0 + t);
    ggml_backend_tensor_set(s.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    // sliding ring: the cells this step writes now hold p0+t
    for (int t = 0; t < T; ++t) m.ring_pos[(p0 + t) % m.ring] = p0 + t;
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    {
        std::vector<ggml_fp16_t> mk((size_t) s.n_kv_full * T);
        for (int t = 0; t < T; ++t)
            for (int64_t j = 0; j < s.n_kv_full; ++j) mk[(size_t) t * s.n_kv_full + j] = j <= p0 + t ? z : ninf;
        ggml_backend_tensor_set(s.mask_full, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
    }
    {
        std::vector<ggml_fp16_t> mk((size_t) s.n_kv_swa * T);
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            for (int j = 0; j < s.n_kv_swa; ++j) {
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
    if (hidden) {
        hidden->resize((size_t) n_out * m.hp.n_embd);
        ggml_backend_tensor_get(s.hidden, hidden->data(), 0, hidden->size() * sizeof(float));
    }
    ggml_free(s.ctx);
}

// ---- checkpoint: the conv states + the sliding rings at the end of a prompt (full-layer K/V is positional)
struct Ckpt {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> conv, k, v;
    std::vector<int64_t> ring_pos;
};
void ckpt_take(Model& m, Ckpt& c, int64_t pos) {
    c.pos = pos;
    c.conv.assign(m.hp.n_layer, {});
    c.k.assign(m.hp.n_layer, {});
    c.v.assign(m.hp.n_layer, {});
    for (int l = 0; l < m.hp.n_layer; ++l) {
        Layer& y = m.L[l];
        auto grab = [](ggml_tensor* t, std::vector<uint8_t>& out) { out.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, out.data(), 0, out.size()); };
        if (y.conv_state) grab(y.conv_state, c.conv[l]);
        else if (m.hp.swa[l]) { grab(y.k_cache, c.k[l]); grab(y.v_cache, c.v[l]); }
    }
    c.ring_pos = m.ring_pos;
}
void ckpt_restore(Model& m, const Ckpt& c) {
    for (int l = 0; l < m.hp.n_layer; ++l) {
        Layer& y = m.L[l];
        if (y.conv_state) ggml_backend_tensor_set(y.conv_state, c.conv[l].data(), 0, c.conv[l].size());
        else if (m.hp.swa[l]) {
            ggml_backend_tensor_set(y.k_cache, c.k[l].data(), 0, c.k[l].size());
            ggml_backend_tensor_set(y.v_cache, c.v[l].data(), 0, c.v[l].size());
        }
    }
    m.ring_pos = c.ring_pos;
}
void state_reset(Model& m) {
    for (int l = 0; l < m.hp.n_layer; ++l)
        if (m.L[l].conv_state) {
            std::vector<uint8_t> z(ggml_nbytes(m.L[l].conv_state), 0);
            ggml_backend_tensor_set(m.L[l].conv_state, z.data(), 0, z.size());
        }
    std::fill(m.ring_pos.begin(), m.ring_pos.end(), -1);
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
    ggml_backend_buffer_free(m.sbuf); m.sbuf = nullptr;
    ggml_backend_buffer_free(m.wbuf); m.wbuf = nullptr;
    detach(m.sctx);
    detach(m.wctx);
    ggml_backend_free(m.gpu); m.gpu = nullptr;           // its pools and streams too
    m.parked = true;
}
void unpark(Model& m) {
    if (!m.parked) return;
    m.gpu = ggml_backend_cuda_init(0);
    if (!m.gpu) die("unpark: the CUDA backend failed to start");
    m.wbuf = ggml_backend_alloc_ctx_tensors(m.wctx, m.gpu);
    if (!m.wbuf) die("unpark: not enough VRAM for the weights");
    ggml_backend_buffer_set_usage(m.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [t, off] : m.host_w_at) ggml_backend_tensor_set(t, m.host_w + off, 0, ggml_nbytes(t));
    m.sbuf = ggml_backend_alloc_ctx_tensors(m.sctx, m.gpu);
    if (!m.sbuf) die("unpark: not enough VRAM for the K/V cache");
    size_t at = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.sctx); t; t = ggml_get_next_tensor(m.sctx, t)) {
        ggml_backend_tensor_set(t, m.host_state.data() + at, 0, ggml_nbytes(t));
        at += ggml_nbytes(t);
    }
    ggml_backend_t bes[2] = {m.gpu, m.cpu};
    m.sched = ggml_backend_sched_new(bes, nullptr, 2, 8192, false, true);
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

// The encoders' files (xolotl-lfm3-vision / -audio = tools/vision, tools/audio built on llama.cpp's mtmd): records of
// int32 {tag, n, nx, ny, n_embd} + float32 [n][n_embd], 'SVE1' an image, 'SVA1' audio, in prompt order.
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
            err = "bad embeddings record (want SVE1/SVA1 with n_embd " + std::to_string(n_embd) + ")";
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

// ---- speech: the xolotl-lfm3-speech child (depthformer + codec through llama.cpp's mtmd, tools/lfm3/lfm3_speech.cpp)
struct SpeechChild {
    std::string exe, mmproj, model;
    bool gpu = true;
    pid_t pid = -1;
    FILE *to = nullptr, *from = nullptr;
    int n_embd = 0, rate = 0, n_cb = 0;
    bool configured() const { return !exe.empty() && !mmproj.empty(); }
    bool alive() const { return pid > 0; }
    void start() {
        int a[2], b[2];
        if (pipe(a) != 0) die("speech: pipe failed");
        if (pipe(b) != 0) { close(a[0]); close(a[1]); die("speech: pipe failed"); }
        pid = fork();
        if (pid < 0) die("speech: fork failed");
        if (pid == 0) {                                   // the child: stdin <- a, stdout -> b, stderr = the engine's log
            dup2(a[0], 0);
            dup2(b[1], 1);
            close(a[0]); close(a[1]); close(b[0]); close(b[1]);
            std::vector<const char*> av = {exe.c_str(), "--mmproj", mmproj.c_str(), "--model", model.c_str()};
            if (gpu) av.push_back("--gpu");
            av.push_back(nullptr);
            execv(exe.c_str(), const_cast<char* const*>(av.data()));
            std::fprintf(stderr, "xolotl-lfm3: cannot run %s\n", exe.c_str());
            _exit(127);
        }
        close(a[0]);
        close(b[1]);
        to = fdopen(a[1], "wb");
        from = fdopen(b[0], "rb");
        int32_t hdr[4];
        if (!r(hdr, sizeof hdr) || hdr[0] != 0x3150534c) { kill_now(); die("speech: xolotl-lfm3-speech did not start (its reason is in this log)"); }
        n_embd = hdr[1]; rate = hdr[2]; n_cb = hdr[3];
    }
    void stop() {                                         // QUIT and wait (park, exit)
        if (pid <= 0) return;
        const int32_t q = 4;
        if (to) { std::fwrite(&q, sizeof q, 1, to); std::fclose(to); to = nullptr; }
        if (from) { std::fclose(from); from = nullptr; }
        int st = 0;
        waitpid(pid, &st, 0);
        pid = -1;
    }
    void kill_now() {                                     // after a protocol error
        if (pid <= 0) return;
        ::kill(pid, SIGKILL);
        if (to) { std::fclose(to); to = nullptr; }
        if (from) { std::fclose(from); from = nullptr; }
        int st = 0;
        waitpid(pid, &st, 0);
        pid = -1;
    }
    bool w(const void* p, size_t n) { return std::fwrite(p, 1, n, to) == n; }
    bool r(void* p, size_t n) { return std::fread(p, 1, n, from) == n; }
    [[noreturn]] void broke() { kill_now(); die("speech: the xolotl-lfm3-speech process stopped answering"); }
    void reset(uint32_t seed, float temp, int32_t top_k, float rep, int32_t rep_n) {
        const int32_t cmd = 1;
        if (!w(&cmd, 4) || !w(&seed, 4) || !w(&temp, 4) || !w(&top_k, 4) || !w(&rep, 4) || !w(&rep_n, 4) || std::fflush(to) != 0) broke();
        int32_t st = 0;
        if (!r(&st, 4)) broke();
        if (st != 0) die("speech: the speech state did not reset");
    }
    // 0 = a frame (codes, fb, pcm), 1 = end of audio (codes)
    int frame(const float* h, int32_t* codes, std::vector<float>& fb, std::vector<float>& pcm) {
        const int32_t cmd = 2;
        if (!w(&cmd, 4) || !w(h, (size_t) n_embd * sizeof(float)) || std::fflush(to) != 0) broke();
        int32_t st = 0;
        if (!r(&st, 4) || !r(codes, (size_t) n_cb * sizeof(int32_t))) broke();
        if (st < 0) die("speech: a depthformer or codec pass failed (see this log)");
        if (st == 1) return 1;
        fb.resize(n_embd);
        int32_t n = 0;
        if (!r(fb.data(), fb.size() * sizeof(float)) || !r(&n, 4) || n < 0) broke();
        pcm.resize(n);
        if (n && !r(pcm.data(), (size_t) n * sizeof(float))) broke();
        return 0;
    }
    void flush(std::vector<float>& pcm) {
        const int32_t cmd = 3;
        if (!w(&cmd, 4) || std::fflush(to) != 0) broke();
        int32_t n = 0;
        if (!r(&n, 4) || n < 0) broke();
        pcm.resize(n);
        if (n && !r(pcm.data(), (size_t) n * sizeof(float))) broke();
    }
};

// `A <n> <base64>`: 16-bit signed little-endian PCM
void emit_pcm(const std::vector<float>& pcm) {
    if (pcm.empty()) return;
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> b(pcm.size() * 2);
    for (size_t i = 0; i < pcm.size(); ++i) {
        const float x = std::max(-1.0f, std::min(1.0f, pcm[i]));
        const int16_t v = (int16_t) std::lrintf(x * 32767.0f);
        b[2 * i] = (uint8_t) (v & 0xff);
        b[2 * i + 1] = (uint8_t) ((uint16_t) v >> 8);
    }
    std::string out;
    out.reserve((b.size() + 2) / 3 * 4 + 32);
    out += "A " + std::to_string(pcm.size()) + " ";
    for (size_t i = 0; i < b.size(); i += 3) {
        const uint32_t n = (uint32_t) b[i] << 16 | (i + 1 < b.size() ? (uint32_t) b[i + 1] << 8 : 0) | (i + 2 < b.size() ? b[i + 2] : 0);
        out += tbl[n >> 18 & 63];
        out += tbl[n >> 12 & 63];
        out += i + 1 < b.size() ? tbl[n >> 6 & 63] : '=';
        out += i + 2 < b.size() ? tbl[n & 63] : '=';
    }
    std::printf("%s\n", out.c_str());
}

void usage() {
    std::fprintf(stderr,
                 "usage: xolotl-lfm3 --serve --native <lfm3.gguf> [--context N] [--ubatch N] [--threads N] [--kv-type f16|q8|k8v4]\n"
                 "       [--park-idle SECS] [--peer-vram http://127.0.0.1:8099/v1/vram]\n"
                 "       [--speech-exe xolotl-lfm3-speech --speech-mmproj <mmproj.gguf> [--speech-cpu]]\n"
                 "       xolotl-lfm3 --dump <lfm3.gguf> --ids a,b,c,... --out logits.f32 [--context N] [--ubatch N]\n"
                 "  LFM3 / LFM3-Omni (text) behind Xolotl's server protocol; --dump writes every position's logits (parity).\n");
}

}  // namespace

// ---- --fa-selftest: the same masked flash attention (sliding window 256, causal) over n_kv = 512 / 768 / 1024 cells
// (the extra cells masked), on CUDA and on the CPU backend, against a float64 reference.  Diagnostics for parity.
int fa_selftest() {
    const int D = 128, H = 16, HKV = 4, T = 512, W = 256;   // LFM3-Omni-2B: 16 query heads, 4 K/V heads
    const float scale = 1.0f / std::sqrt((float) D);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t) D * T * H), k((size_t) D * T * HKV), v((size_t) D * T * HKV);
    for (auto& x : q) x = 3.0f * nd(rng);
    for (auto& x : k) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng)));
    for (auto& x : v) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng)));
    // q [D, T, H], k/v [D, n_kv, HKV] (cells >= T are zero and masked), out [D, H, T]
    std::vector<double> ref((size_t) D * H * T);
    for (int h = 0; h < H; ++h) {
        const int hk = h / (H / HKV);
        for (int t = 0; t < T; ++t) {
            std::vector<double> w(T, 0.0); double mx = -1e300;
            for (int j = 0; j <= t; ++j) {
                if (t - j >= W) continue;
                double d = 0; for (int i = 0; i < D; ++i) d += (double) q[((size_t) h * T + t) * D + i] * k[((size_t) hk * T + j) * D + i];
                w[j] = d * scale; mx = std::max(mx, w[j]);
            }
            double sum = 0; for (int j = 0; j <= t; ++j) { if (t - j >= W) { w[j] = 0; continue; } w[j] = std::exp(w[j] - mx); sum += w[j]; }
            for (int i = 0; i < D; ++i) {
                double o = 0; for (int j = 0; j <= t; ++j) o += w[j] * v[((size_t) hk * T + j) * D + i];
                ref[((size_t) t * H + h) * D + i] = o / sum;
            }
        }
    }
    for (int nkv : {512, 768, 1024}) {
        for (int be = 0; be < 2; ++be) {
            ggml_backend_t b = be == 0 ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
            ggml_init_params ip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
            ggml_context* c = ggml_init(ip);
            // the engine's layout: q [D, H, T] and caches [D, HKV, cells], permuted to FA's [D, rows, heads] views
            ggml_tensor* Q = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, H, T);
            ggml_tensor* K = ggml_new_tensor_3d(c, GGML_TYPE_F16, D, HKV, nkv);
            ggml_tensor* V = ggml_new_tensor_3d(c, GGML_TYPE_F16, D, HKV, nkv);
            ggml_tensor* M = ggml_new_tensor_2d(c, GGML_TYPE_F16, nkv, T);
            ggml_tensor* O = ggml_flash_attn_ext(c, ggml_permute(c, Q, 0, 2, 1, 3), ggml_permute(c, K, 0, 2, 1, 3),
                                                 ggml_permute(c, V, 0, 2, 1, 3), M, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(O, GGML_PREC_F32);
            ggml_cgraph* g = ggml_new_graph(c);
            ggml_build_forward_expand(g, O);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, b);
            std::vector<ggml_fp16_t> kh((size_t) D * nkv * HKV, ggml_fp32_to_fp16(0.0f)), vh(kh.size(), ggml_fp32_to_fp16(0.0f));
            for (int hk = 0; hk < HKV; ++hk) for (int j = 0; j < T; ++j) for (int i = 0; i < D; ++i) {
                kh[((size_t) j * HKV + hk) * D + i] = ggml_fp32_to_fp16(k[((size_t) hk * T + j) * D + i]);
                vh[((size_t) j * HKV + hk) * D + i] = ggml_fp32_to_fp16(v[((size_t) hk * T + j) * D + i]);
            }
            std::vector<ggml_fp16_t> mh((size_t) nkv * T);
            for (int t = 0; t < T; ++t) for (int j = 0; j < nkv; ++j)
                mh[(size_t) t * nkv + j] = ggml_fp32_to_fp16(j < T && j <= t && t - j < W ? 0.0f : -INFINITY);
            std::vector<float> qh(q.size());
            for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) for (int i = 0; i < D; ++i)
                qh[((size_t) t * H + h) * D + i] = q[((size_t) h * T + t) * D + i];
            ggml_backend_tensor_set(Q, qh.data(), 0, qh.size() * 4);
            ggml_backend_tensor_set(K, kh.data(), 0, kh.size() * 2);
            ggml_backend_tensor_set(V, vh.data(), 0, vh.size() * 2);
            ggml_backend_tensor_set(M, mh.data(), 0, mh.size() * 2);
            if (be == 1) ggml_backend_cpu_set_n_threads(b, 8);
            ggml_backend_graph_compute(b, g);
            std::vector<float> o((size_t) D * H * T);
            ggml_backend_tensor_get(O, o.data(), 0, o.size() * 4);
            double mx = 0, mx_lo = 0, mx_hi = 0;
            for (int t = 0; t < T; ++t) for (size_t i = 0; i < (size_t) D * H; ++i) {
                const double e = std::fabs(o[(size_t) t * D * H + i] - ref[(size_t) t * D * H + i]);
                mx = std::max(mx, e); (t < W ? mx_lo : mx_hi) = std::max(t < W ? mx_lo : mx_hi, e);
            }
            std::printf("fa-selftest %-4s n_kv %4d: max |err| %.3e  (queries < %d: %.3e, >= %d: %.3e)\n", be == 0 ? "cuda" : "cpu", nkv, mx, W, mx_lo, W, mx_hi);
            ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(b);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string gguf_path, ids_arg, out_path;
    bool serve = false, dump = false, kv_ram = false;
    int64_t ctx = 32768;
    int ubatch = 512, threads = 8;
    double park_idle_s = 0;
    std::string peer_vram, kv_type = "f16";
    SpeechChild speech;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } (void) what; return argv[++i]; };
        if (a == "--fa-selftest") return fa_selftest();
        if (a == "--serve") serve = true;
        else if (a == "--dump") { dump = true; gguf_path = val("gguf"); }
        else if (a == "--native") gguf_path = val("gguf");
        else if (a == "--context" || a == "--max-context") ctx = std::atoll(val("n").c_str());
        else if (a == "--ubatch") ubatch = std::atoi(val("n").c_str());
        else if (a == "--threads") threads = std::atoi(val("n").c_str());
        else if (a == "--kv-ram") kv_ram = true;
        else if (a == "--kv-type") kv_type = val("type");
        else if (a == "--park-idle") park_idle_s = std::atof(val("s").c_str());
        else if (a == "--peer-vram") peer_vram = val("url");
        else if (a == "--speech-exe") speech.exe = val("path");
        else if (a == "--speech-mmproj") speech.mmproj = val("path");
        else if (a == "--speech-cpu") speech.gpu = false;
        else if (a == "--ids") ids_arg = val("ids");
        else if (a == "--out") out_path = val("path");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else std::fprintf(stderr, "xolotl-lfm3: ignoring unknown option %s\n", a.c_str());
    }
    if (gguf_path.empty() || (!serve && !dump)) { usage(); return 2; }
    speech.model = gguf_path;
    signal(SIGPIPE, SIG_IGN);                              // a speech child that died is an error line, not the engine's end

    Model m;
    m.ubatch = std::max(16, ubatch);
    m.kv_ram = kv_ram;
    if (kv_type == "f16") {}
    else if (kv_type == "q8") m.kv_k = m.kv_v = GGML_TYPE_Q8_0;
    else if (kv_type == "k8v4") { m.kv_k = GGML_TYPE_Q8_0; m.kv_v = GGML_TYPE_Q4_0; }
    else { std::fprintf(stderr, "xolotl-lfm3: --kv-type is f16, q8 or k8v4 (not %s)\n", kv_type.c_str()); return 2; }
    try {
        load(m, gguf_path, ctx, threads);
    } catch (const std::exception& e) {
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    const int n_vocab = m.hp.n_vocab;

    if (dump) {
        std::vector<int32_t> ids;
        if (!parse_ids(ids_arg.c_str(), ids)) { usage(); return 2; }
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
        std::vector<float> lg;
        for (size_t p = 0; p < ids.size(); p += m.ubatch) {
            const int T = (int) std::min<size_t>(m.ubatch, ids.size() - p);
            forward(m, ids.data() + p, T, (int64_t) p, T, lg);
            std::fwrite(lg.data(), sizeof(float), lg.size(), f);
        }
        std::fclose(f);
        std::fprintf(stderr, "xolotl-lfm3: wrote %zu x %d logits to %s\n", ids.size(), n_vocab, out_path.c_str());
        return 0;
    }

    // stop ids: eos + the chat end markers from the vocabulary
    std::vector<int32_t> stop_ids;
    {
        const int64_t e = m.gg.key("tokenizer.ggml.eos_token_id", false);
        if (e >= 0) stop_ids.push_back((int32_t) m.gg.i("tokenizer.ggml.eos_token_id"));
        const int64_t tk = m.gg.key("tokenizer.ggml.tokens", false);
        if (tk >= 0)
            for (size_t i = 0; i < gguf_get_arr_n(m.gg.g, tk); ++i) {
                const char* s = gguf_get_arr_str(m.gg.g, tk, i);
                if (!std::strcmp(s, "<|im_end|>") || !std::strcmp(s, "<|endoftext|>") || !std::strcmp(s, "<|text_end|>")) stop_ids.push_back((int32_t) i);
            }
        std::sort(stop_ids.begin(), stop_ids.end());
        stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());
    }
    auto is_stop = [&](int32_t t) { return std::binary_search(stop_ids.begin(), stop_ids.end(), t); };

    float* d_logits = nullptr;
    int* d_next = nullptr;
    int* d_hist = nullptr;
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_logits, (size_t) n_vocab * sizeof(float)) != cudaSuccess || cudaMalloc((void**) &d_next, sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hist, kMaxHist * sizeof(int)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=lfm3-0.1 model=lfm3 layers=%d n_embd=%d weights_mib=%llu kv_mib=%llu context=%lld ubatch=%d ring=%d kv=%s/%s\n",
                m.hp.n_layer, m.hp.n_embd, (unsigned long long) (m.weight_bytes >> 20), (unsigned long long) (m.state_bytes >> 20),
                (long long) m.n_ctx, m.ubatch, m.ring, ggml_type_name(m.kv_k), ggml_type_name(m.kv_v));
    std::printf("READY %lld stop\n", (long long) m.n_ctx);
    std::fflush(stdout);
    std::fprintf(stderr, "xolotl-lfm3: ready - %.2f GiB of weights, %.2f GiB of state, context %lld\n", m.weight_bytes / 1073741824.0,
                 m.state_bytes / 1073741824.0, (long long) m.n_ctx);

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

    // the media placeholders the chat template writes (expanded by the server to one per encoder row)
    std::vector<int32_t> media_ids;
    {
        const int64_t tk = m.gg.key("tokenizer.ggml.tokens", false);
        if (tk >= 0)
            for (size_t i = 0; i < gguf_get_arr_n(m.gg.g, tk); ++i) {
                const char* s = gguf_get_arr_str(m.gg.g, tk, i);
                if (!std::strcmp(s, "<|image|>") || !std::strcmp(s, "<|audio|>") || !std::strcmp(s, "<|video|>")) media_ids.push_back((int32_t) i);
            }
    }
    auto is_media = [&](int32_t t) { return std::find(media_ids.begin(), media_ids.end(), t) != media_ids.end(); };
    size_t fed_media = SIZE_MAX;  // the first media token the state has read (its rows are not in `fed`)
    std::vector<int32_t> fed;   // every token the state has consumed
    bool stale = false;         // the state has read rows past `fed` (a GENA's audio frames): resume from the checkpoint
    Ckpt ck;                    // the state at the end of the last prompt
    std::vector<float> lg;
    std::string line;
    // park / unpark: idle for --park-idle seconds -> everything leaves the GPU (state to pinned RAM); the next
    // request brings it back.  --peer-vram URL (another Xolotl server's POST /v1/vram, e.g. QFN's): asked to keep
    // room free before an unpark, and told it may take the VRAM back after a park.
    const size_t need_mib = (m.weight_bytes >> 20) + (m.state_bytes >> 20) + 700;
    auto peer = [&](const char* reserve) {
        if (peer_vram.empty()) return;
        const std::string cmd = "curl -s -m 150 -X POST '" + peer_vram + "' -H 'Content-Type: application/json' -d '{\"reserve_mib\": " +
                                reserve + "}' > /dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "xolotl-lfm3: the peer's /v1/vram did not answer\n");
    };
    auto do_park = [&](const char* why) {
        if (m.parked) return;
        const auto t0 = Clock::now();
        speech.stop();                                     // its VRAM too; the next GENA starts it again
        park(m);
        peer("null");
        std::fprintf(stderr, "xolotl-lfm3: parked (%s) in %.0f ms - %zu MiB of state in RAM, %zu MiB VRAM free\n", why, ms_since(t0),
                     m.host_state.size() >> 20, vram_free_mib());
    };
    auto do_unpark = [&]() {
        if (!m.parked) return;
        const auto t0 = Clock::now();
        peer(std::to_string(need_mib).c_str());
        unpark(m);
        std::fprintf(stderr, "xolotl-lfm3: back on the GPU in %.0f ms\n", ms_since(t0));
    };
    for (;;) {
        bool idle = false;
        if (!in.pop(line, park_idle_s > 0 && !m.parked ? (int64_t) (park_idle_s * 1000) : 0, &idle)) {
            if (idle) {
                try { do_park("idle"); } catch (const std::exception& e) { std::fprintf(stderr, "xolotl-lfm3: %s\n", e.what()); }
                continue;
            }
            break;
        }
        if (line == "QUIT") {
            // _Exit, not return: exit() flushes every stdio stream, stdin too, whose lock the reader thread holds inside
            // getline while the server keeps the pipe open - QUIT hung until stdin closed (park test, 2026-11-02)
            speech.stop();
            std::fflush(stdout); std::fflush(stderr);
            std::_Exit(0);
        }
        if (line == "PARK" || line.rfind("VRAM", 0) == 0) {
            // VRAM <mib> (the server's POST /v1/vram, elastic): any reserve parks; VRAM alone: back on the GPU
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
        const bool gena = line.rfind("GENA ", 0) == 0;
        if (!geni && !gena && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>, GENI <max_new> [key=value ...] <embeddings file> <id,id,...> "
                        "or GENA <max_frames> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni || gena ? 5 : 4), &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        std::vector<int32_t> score_ids;
        bool score_bad = false;
        // GENA: only what the request names goes to the speech child (the rest = the checkpoint's defaults)
        float a_temp = -1.0f, a_rep = -1.0f;
        int32_t a_top_k = -1, a_rep_n = -1;
        bool seed_given = false, want_codes = false;
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
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; a_temp = fv; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") { sp.top_k = std::atoi(tok.c_str() + eq + 1); a_top_k = sp.top_k; }
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") { sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist); a_rep_n = std::atoi(tok.c_str() + eq + 1); }
            else if (key == "penalty_repeat") { sp.penalty_repeat = fv; a_rep = fv; }
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") { seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10); seed_given = true; }
            else if (key == "codes") want_codes = std::atoi(tok.c_str() + eq + 1) != 0;
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
        // media rows, one per placeholder token (<|image|>, <|audio|>, <|video|>) in prompt order
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
                std::printf("ERR the prompt has %lld media tokens but the embeddings file %zu rows\n", (long long) k, media.size() / m.hp.n_embd);
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
            if (!stale && !fed.empty() && fed.size() < ids.size() && common == fed.size()) resume = fed.size();
            else if (ck.pos > 0 && (size_t) ck.pos <= common && (size_t) ck.pos < ids.size()) { ckpt_restore(m, ck); resume = (size_t) ck.pos; }
            else state_reset(m);
        } catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); std::fflush(stdout); continue; }
        fed.assign(ids.begin(), ids.begin() + (long) resume);
        stale = false;

        // ---- the prompt
        std::vector<float> chunk_media, chunk_sel, hid;
        auto run = [&](size_t at, int T) {
            bool any = false;
            for (int t = 0; t < T && !any; ++t) any = media_row[at + t] >= 0;
            if (!any) { forward(m, ids.data() + at, T, (int64_t) at, 1, lg, nullptr, nullptr, gena ? &hid : nullptr); return; }
            const int ne = m.hp.n_embd;
            chunk_media.assign((size_t) T * ne, 0.0f);
            chunk_sel.assign(T, 0.0f);
            for (int t = 0; t < T; ++t)
                if (media_row[at + t] >= 0) {
                    std::memcpy(chunk_media.data() + (size_t) t * ne, media.data() + (size_t) media_row[at + t] * ne, (size_t) ne * sizeof(float));
                    chunk_sel[t] = 1.0f;
                }
            forward(m, ids.data() + at, T, (int64_t) at, 1, lg, chunk_media.data(), chunk_sel.data(), gena ? &hid : nullptr);
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
            std::fprintf(stderr, "xolotl-lfm3: %s\n", e.what());
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

        if (gena) {
            // ---- speech: hidden state -> the child's frame (12 codes, its feedback row, PCM) -> the row is the next
            // input; as llama.cpp's lfm3a_gen_audio_pipeline::step_gen in TTS mode (end of audio ends the request)
            const auto t_dec = Clock::now();
            long long frames = 0;
            const char* why = "length";
            try {
                if (!speech.configured()) die("this engine was started without --speech-exe and --speech-mmproj (no speech output)");
                if (!speech.alive()) {
                    const auto t0 = Clock::now();
                    speech.start();
                    std::fprintf(stderr, "xolotl-lfm3: speech process up in %.0f ms (%d Hz, %d codebooks)\n", ms_since(t0), speech.rate, speech.n_cb);
                }
                if (speech.n_embd != m.hp.n_embd) die("speech: the speech process has n_embd " + std::to_string(speech.n_embd));
                speech.reset(seed_given ? (uint32_t) seed : UINT32_MAX, a_temp, a_top_k, a_rep, a_rep_n);
                int64_t pos = (int64_t) fed.size();
                std::vector<int32_t> codes(speech.n_cb);
                std::vector<float> fb, pcm;
                const float one = 1.0f;
                const int32_t row_tok = 0;                      // any id: the feedback row replaces its embedding
                for (;;) {
                    const int st = speech.frame(hid.data(), codes.data(), fb, pcm);
                    if (want_codes) {
                        std::string c = "C";
                        for (int i = 0; i < speech.n_cb; ++i) c += (i ? "," : " ") + std::to_string(codes[i]);
                        std::printf("%s\n", c.c_str());
                    }
                    if (st == 1) { why = "stop"; std::fflush(stdout); break; }
                    emit_pcm(pcm);
                    std::fflush(stdout);
                    ++frames;
                    if (frames >= max_new) break;
                    if (pos + 1 >= m.n_ctx) break;
                    if (in.stop.load()) { why = "cancel"; break; }
                    forward(m, &row_tok, 1, pos, 1, lg, fb.data(), &one, &hid);
                    ++pos;
                }
                speech.flush(pcm);                              // the codec's held-back tail
                emit_pcm(pcm);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "xolotl-lfm3: %s\n", e.what());
                std::printf("ERR %s\n", e.what());
                std::fflush(stdout);
                fed.clear(); ck.pos = -1;
                continue;
            }
            // the state now holds audio rows past the prompt: the next request resumes from the prompt's checkpoint
            fed.resize(ck.pos > 0 ? (size_t) ck.pos : 0);
            if (ck.pos <= 0) fed.clear();
            stale = true;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", frames, ids.size(), prompt_ms, ms_since(t_dec), why, resume);
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
        try {
            for (;;) {
                cudaMemcpy(d_logits, lg.data(), (size_t) n_vocab * sizeof(float), cudaMemcpyHostToDevice);
                int hl = 0;
                if (sp.penalty_last_n > 0) {
                    hl = std::min<int>(sp.penalty_last_n, (int) fed.size());
                    hist.assign(fed.end() - hl, fed.end());
                    cudaMemcpy(d_hist, hist.data(), (size_t) hl * sizeof(int), cudaMemcpyHostToDevice);
                }
                sp.counter = (uint64_t) gen;
                strata::kernels::sample_tokens(d_logits, 1, n_vocab, hl ? d_hist : nullptr, hl, sp, d_next, nullptr);
                int next = 0;
                cudaMemcpy(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost);
                std::printf("T %d\n", next);
                std::fflush(stdout);
                ++gen;
                if (is_stop(next)) { why = "stop"; break; }
                if (gen >= max_new) break;
                if ((int64_t) fed.size() + 1 >= m.n_ctx) break;
                if (in.stop.load()) { why = "cancel"; break; }
                forward(m, &next, 1, (int64_t) fed.size(), 1, lg);
                fed.push_back(next);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xolotl-lfm3: %s\n", e.what());
            fed.clear(); ck.pos = -1;
            why = "cancel";
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", gen, ids.size(), prompt_ms, ms_since(t_dec), why, resume);
        std::fflush(stdout);
    }
    return 0;
}
