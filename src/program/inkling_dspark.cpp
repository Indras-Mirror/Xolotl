// src/program/inkling_dspark.cpp - the DSpark block drafter for Inkling-2-Mini (GGUF arch `dflash`).
//
// See inkling_dspark.h for the conventions and the integration points.  The executable spec is
// a torch reference of the MTP drafter (6/7 accepted on golden p0); this file is a
// line-by-line port of it:
//
//   ctx K/V  = fuse(taps, row=li) -> k_proj_ctx/v_proj_ctx -> k_norm_ctx -> neox rope at the position
//   block    = row 0 rms(embed(anchor), embed_norm), rows 1..7 mask_embd; positions T..T+7
//   3 layers = rms -> q/k/v (+per-head norms, rope) -> bidirectional attention over ctx+block -> o_proj
//              -> residual -> rms -> silu-glu MLP -> residual
//   hidden   = rms(x, output_norm); base = hidden @ unembed^T  (the TARGET's LM head)
//   markov   = fp32 on the host: raw = ctx_i + [state, w1[prev]] @ joint[:, :512]^T
//              gate/cand/state, latent; logits = 30*tanh((base_i + fold_w@latent + fold_b)/30)
//
// The one precomputed fusion is `fold_w = unembed @ dspark_hidden_out.weight` (once, on the GPU at load): it turns
// every markov bias into a 256-wide matvec instead of a full 2304-wide LM matmul.  The markov recurrence itself is
// host fp32, as the validated dry run does it; only the 256->vocab projection runs on the device.
#include "inkling_dspark.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include "strata/platform/win_compat.hpp"   // MSVC: the 64-bit off_t and fseeko the seeks below are cast through

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNodes = 4096;
constexpr int kLayers = 3;

[[noreturn]] void dfail(const std::string& m) { throw std::runtime_error(m); }

}  // namespace

// -------------------------------------------------------------------------------------------------- GGUF keys
struct Keys {
    gguf_context* g = nullptr;
    int64_t key(const std::string& k, bool need = true) const {
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0 && need) dfail("drafter GGUF key missing: " + k);
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
            default: dfail("drafter GGUF value is not an integer");
        }
    }
    int64_t i(const std::string& k) const {
        const int64_t id = key(k);
        return as_int(gguf_get_kv_type(g, id), gguf_get_val_data(g, id));
    }
    int64_t i_or(const std::string& k, int64_t d) const { return key(k, false) < 0 ? d : i(k); }
    float f(const std::string& k) const {
        const int64_t id = key(k);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) dfail("drafter GGUF key is not a float: " + k);
        return gguf_get_val_f32(g, id);
    }
};

// -------------------------------------------------------------------------------------------------- the drafter
struct DLayer {
    ggml_tensor *dflare = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *wk_ctx = nullptr, *wv_ctx = nullptr;
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
    bool local = true;
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr;   // [hd, nkv, cells] F16
};

struct Dspark {
    // gguf + weights
    Keys k;
    ggml_context* wctx = nullptr;
    ggml_context* sctx = nullptr;   // K/V caches (cleared by dspark_reset)
    ggml_context* fctx = nullptr;   // the folded LM head (survives dspark_reset)
    ggml_backend_buffer_t wbuf = nullptr, sbuf = nullptr, fbuf = nullptr;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> gbuf;      // graph arena (caches/block/markov-fold graphs)
    size_t weight_bytes = 0, state_bytes = 0;

    // hyperparameters
    int n_embd = 0, n_head = 0, n_head_kv = 0, hd = 0, ff = 0, block = 0, n_swa = 0;
    int vpad = 0, vocab = 0, markov_rank = 256;
    float eps = 1e-6f, rope_base = 1e6f, softcap = 30.0f;
    int64_t n_ctx = 0, n_local = 0, ctx_end = 0, n_glob = 0;
    DLayer L[kLayers];

    ggml_tensor *fuse = nullptr, *mask_embd = nullptr, *norm = nullptr, *markov_norm = nullptr;
    ggml_tensor *ho_w = nullptr, *ho_b = nullptr;   // dspark_hidden_out
    ggml_tensor *w1 = nullptr;                      // markov_w1
    ggml_tensor *joint = nullptr, *joint_b = nullptr;
    ggml_tensor *fold_w = nullptr, *fold_b = nullptr;
    ggml_tensor *hoT = nullptr;                     // dspark_hidden_out transposed: [n_embd, 256] f32
    ggml_tensor *jctx = nullptr;
    ggml_tensor *jst = nullptr, *jpv = nullptr;     // joint_proj[:, 0:256] / [:, 256:512] as [256, 768] f32 (markov, on the GPU)
    ggml_tensor *ctx_buf = nullptr;                 // [768, block] f32: the block's markov ctx term stays on the GPU
    ggml_tensor *base_buf = nullptr;                // [vocab, block] f32: the block's base logits stay on the GPU                    // joint_proj[:, 512:] as [2*n_embd, 768] f32 (the ctx part, on the GPU)
    float tau = 0.0f;                               // XOLOTL_DSPARK_TAU: stop drafting once the cumulative confidence < tau
    int max_draft = 7;                              // XOLOTL_DSPARK_MAXD

    // target
    ggml_tensor *embed = nullptr, *embed_norm = nullptr, *unembed = nullptr;

    // host copies (fp32)
    std::vector<float> h_alpha;       // [4][4] softmax(fuse)
    std::vector<float> h_mask;        // [n_embd]
    std::vector<float> h_norm;        // output_norm (= torch norm.weight)
    std::vector<float> h_markov_norm; // dspark_norm  (= torch markov_head.dflare_norm.weight)
    std::vector<float> h_embed_norm;  // target token_embd_norm
    std::vector<float> h_ho;          // dspark_hidden_out.weight, fp32 in the torch [2304][256] order
    std::vector<float> h_joint;       // [5120][768] fp32, [in][out]
    std::vector<float> h_joint_b;     // [768]
    std::vector<float> h_w1;          // [vocab][256]

    // committed context
    std::vector<float> anchor_taps;   // [4][n_embd] at position ctx_end-1
    std::vector<float> trow;          // scratch: one embedding row
    std::vector<float> f_rnn;         // scratch: markov fused tap
};

int64_t cells_local(const Dspark& d) { return d.n_swa + 1; }
// The global layer keeps the last n_glob positions in a ring (XOLOTL_DSPARK_WINDOW, default 32768; 0 = the whole
// context, the old positional cache).  A full-context f16 cache was 2 GiB at 512K and each draft cast all of it to
// f32 (O(T) per step: below plain decode at 38K, an allocation failure at 110K).  The drafter only proposes; the target
// verifies every token, so the window changes acceptance, never the output.  kGlobSlack extra cells keep rewinds of up
// to that many tokens exact.
constexpr int64_t kGlobSlack = 512;
int64_t cells_global(const Dspark& d) { return d.n_glob > 0 ? d.n_glob + kGlobSlack : d.n_ctx; }
bool glob_ring(const Dspark& d) { return d.n_glob > 0; }

ggml_tensor* need(ggml_context* c, const std::string& n) {
    ggml_tensor* t = ggml_get_tensor(c, n.c_str());
    if (!t) dfail("drafter tensor missing: " + n);
    return t;
}

std::string s_pfx;   // the tensor-name prefix of a drafter built into another GGUF (set by dspark_load)

std::vector<uint8_t> read_bytes(gguf_context* g, FILE* f, const ggml_tensor* t) {
    const int64_t idx = gguf_find_tensor(g, (s_pfx + t->name).c_str());
    if (idx < 0) dfail(std::string("drafter tensor missing from file: ") + t->name);
    std::vector<uint8_t> b(ggml_nbytes(t));
    if (fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, idx)), SEEK_SET) != 0 ||
        std::fread(b.data(), 1, b.size(), f) != b.size())
        dfail(std::string("short read on drafter tensor ") + t->name);
    return b;
}

std::vector<float> to_f32(const std::vector<uint8_t>& b, ggml_type type, int64_t n) {
    std::vector<float> o((size_t) n);
    if (type == GGML_TYPE_F32) { std::memcpy(o.data(), b.data(), (size_t) n * 4); return o; }
    if (type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < n; ++i) o[(size_t) i] = ggml_fp16_to_fp32(((const ggml_fp16_t*) b.data())[i]);
        return o;
    }
    if (type == GGML_TYPE_BF16) {
        for (int64_t i = 0; i < n; ++i) o[(size_t) i] = ggml_bf16_to_fp32(((const ggml_bf16_t*) b.data())[i]);
        return o;
    }
    const ggml_type_traits* tt = ggml_get_type_traits(type);   // quantised token tables (q8_0 in the XQ quant)
    if (!tt || !tt->to_float) dfail("drafter tensor has an unsupported host type");
    tt->to_float(b.data(), o.data(), n);
    return o;
}

std::vector<float> read_vec(gguf_context* g, FILE* f, ggml_context* wctx, const std::string& name) {
    const ggml_tensor* t = need(wctx, name);
    return to_f32(read_bytes(g, f, t), t->type, ggml_nelements(t));
}

void rms_row(float* x, const float* w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float s = (float) (1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) x[i] = x[i] * s * (w ? w[i] : 1.0f);
}

// softmax(fuse_weight, dim=-1)
void softmax_rows(const std::vector<float>& in, std::vector<float>& out, int rows, int cols) {
    out.resize((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + (size_t) r * cols;
        float mx = x[0];
        for (int j = 1; j < cols; ++j) mx = std::max(mx, x[j]);
        double se = 0;
        for (int j = 0; j < cols; ++j) se += std::exp((double) x[j] - mx);
        for (int j = 0; j < cols; ++j) out[(size_t) r * cols + j] = (float) (std::exp((double) x[j] - mx) / se);
    }
}

// rms(sum_j alpha[row][j] * taps[j]) over one [4][n_embd] tap row
void mix_host(const Dspark& d, const float* taps, int row, const float* wnorm, float* out) {
    const int E = d.n_embd;
    for (int e = 0; e < E; ++e) {
        double s = 0;
        for (int j = 0; j < 4; ++j) s += (double) d.h_alpha[(size_t) row * 4 + j] * taps[(size_t) j * E + e];
        out[e] = (float) s;
    }
    rms_row(out, wnorm, E, d.eps);
}

namespace {

// ---------------------------------------------------------------------------------------------- graph plumbing
struct Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
};

Graph graph_begin(Dspark& d) {
    Graph gr;
    ggml_init_params ip{(size_t) d.gbuf.size(), d.gbuf.data(), true};
    gr.ctx = ggml_init(ip);
    if (!gr.ctx) dfail("cannot create the drafter graph context");
    gr.gf = ggml_new_graph_custom(gr.ctx, kNodes, false);
    return gr;
}

void graph_compute(Dspark& d, Graph& gr) {
    ggml_backend_sched_reset(d.sched);
    if (!ggml_backend_sched_alloc_graph(d.sched, gr.gf)) { ggml_free(gr.ctx); dfail("drafter graph allocation failed (VRAM full)"); }
    if (ggml_backend_sched_graph_compute(d.sched, gr.gf) != GGML_STATUS_SUCCESS) { ggml_free(gr.ctx); dfail("drafter graph compute failed"); }
    ggml_free(gr.ctx);
    gr.ctx = nullptr;
    gr.gf = nullptr;
}

ggml_tensor* g_rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    x = ggml_rms_norm(c, x, eps);
    return w ? ggml_mul(c, x, w) : x;
}

ggml_tensor* g_rope(ggml_context* c, const Dspark& d, ggml_tensor* x, ggml_tensor* pos) {
    return ggml_rope_ext(c, x, pos, nullptr, d.hd, GGML_ROPE_TYPE_NEOX, 1048576, d.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

// attention over ctx+block, bidirectional: q [hd, nh, T], K/V [hd, nkv, nk] -> o [hd*nh, T]
ggml_tensor* g_attend(ggml_context* c, const Dspark& d, ggml_tensor* q, ggml_tensor* K, ggml_tensor* V) {
    ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);                       // [hd, T, nh]
    ggml_tensor* Kv = ggml_permute(c, K, 0, 2, 1, 3);                       // [hd, nk, nkv]
    ggml_tensor* kq = ggml_mul_mat(c, Kv, qp);                              // [nk, T, nh]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / std::sqrt((float) d.hd), 0.0f);
    ggml_tensor* Vv = ggml_permute(c, V, 0, 2, 1, 3);                       // [hd, nk, nkv]
    ggml_tensor* vt = ggml_cont(c, ggml_transpose(c, Vv));                  // [nk, hd, nkv]
    ggml_tensor* o = ggml_mul_mat(c, vt, kq);                               // [hd, T, nh]
    return ggml_cont_2d(c, ggml_permute(c, o, 0, 2, 1, 3), (int64_t) d.hd * d.n_head, q->ne[2]);
}

void g_store_kv(ggml_context* c, ggml_cgraph* gf, ggml_tensor* src, ggml_tensor* cache, ggml_tensor* idx, int t) {
    ggml_tensor* d2 = ggml_reshape_2d(c, cache, cache->ne[0] * cache->ne[1], cache->ne[2]);
    ggml_tensor* s2 = ggml_reshape_2d(c, src, src->ne[0] * src->ne[1], t);
    ggml_build_forward_expand(gf, ggml_set_rows(c, d2, s2, idx));
}

}  // namespace

// -------------------------------------------------------------------------------------------------------- load
bool dspark_in_gguf(const std::string& path, const std::string& prefix) {
    gguf_init_params gp{true, nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return false;
    const bool yes = gguf_find_tensor(g, (prefix + "markov_w1.weight").c_str()) >= 0 &&
                     gguf_find_key(g, "dflash.block_count") >= 0;
    gguf_free(g);
    return yes;
}

Dspark* dspark_load(const std::string& path, ggml_backend_t gpu, const DsparkTarget& tgt, int64_t n_ctx,
                    const std::string& prefix) {
    if (!gpu) dfail("dspark_load: no GPU backend");
    if (!tgt.embed || !tgt.embed_norm || !tgt.unembed) dfail("dspark_load: missing target tensors");
    Dspark* d = new Dspark();
    d->gpu = gpu;
    d->cpu = ggml_backend_cpu_init();
    if (!d->cpu) { delete d; dfail("dspark_load: CPU backend failed to start"); }
    ggml_backend_cpu_set_n_threads(d->cpu, 4);
    d->embed = tgt.embed; d->embed_norm = tgt.embed_norm; d->unembed = tgt.unembed;

    try {
        s_pfx = prefix;
        ggml_context* meta = nullptr;   // built-in drafter: every tensor of the host GGUF, metadata only
        gguf_init_params gp{true, prefix.empty() ? &d->wctx : &meta};
        d->k.g = gguf_init_from_file(path.c_str(), gp);
        if (!d->k.g) dfail("cannot read the drafter GGUF " + path);
        gguf_context* g = d->k.g;
        if (prefix.empty()) {
            const std::string arch = gguf_get_val_str(g, d->k.key("general.architecture"));
            if (arch != "dflash") dfail("not a DSpark drafter GGUF (general.architecture = " + arch + ")");
        } else {   // keep only `<prefix>*`, under the drafter's own names
            std::vector<ggml_tensor*> src;
            for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
                const char* nm = gguf_get_tensor_name(g, i);
                if (std::strncmp(nm, prefix.c_str(), prefix.size()) == 0) src.push_back(ggml_get_tensor(meta, nm));
            }
            if (src.empty()) { ggml_free(meta); dfail("no built-in drafter (" + prefix + "*) in " + path); }
            ggml_init_params ip{(src.size() + 1) * ggml_tensor_overhead(), nullptr, true};
            d->wctx = ggml_init(ip);
            for (ggml_tensor* t : src) {
                ggml_tensor* c = ggml_new_tensor(d->wctx, t->type, GGML_MAX_DIMS, t->ne);
                ggml_set_name(c, t->name + prefix.size());
            }
            ggml_free(meta);
        }
        d->n_embd = (int) d->k.i("dflash.embedding_length");
        d->n_head = (int) d->k.i("dflash.attention.head_count");
        d->n_head_kv = (int) d->k.i("dflash.attention.head_count_kv");
        d->hd = (int) d->k.i_or("dflash.attention.key_length", d->n_embd / d->n_head);
        d->ff = (int) d->k.i_or("dflash.feed_forward_length", 9216);
        d->block = (int) d->k.i_or("dflash.block_size", 8);
        d->n_swa = (int) d->k.i_or("dflash.attention.sliding_window", 1024);
        d->eps = d->k.f("dflash.attention.layer_norm_rms_epsilon");
        d->rope_base = d->k.f("dflash.rope.freq_base");
        d->softcap = d->k.f("dflash.final_logit_softcapping");
        d->vpad = (int) d->k.i("dflash.vocab_size");
        d->vocab = (int) d->k.i("dflash.vocab_size_unpadded");
        d->markov_rank = (int) d->k.i_or("dflash.dspark.markov_rank", 256);
        if ((int) d->k.i("dflash.block_count") != kLayers) dfail("the drafter is not the 3-layer DSpark block");
        if (d->markov_rank != 256) dfail("unsupported markov rank " + std::to_string(d->markov_rank));
        if (d->block < 2) dfail("unsupported dflash.block_size");
        const int64_t pat_id = d->k.key("dflash.attention.sliding_window_pattern");
        if ((int64_t) gguf_get_arr_n(g, pat_id) != kLayers) dfail("sliding_window_pattern has the wrong length");
        const int64_t pat_type = gguf_get_arr_type(g, pat_id);
        if (pat_type != GGUF_TYPE_BOOL && pat_type != GGUF_TYPE_UINT8 && pat_type != GGUF_TYPE_INT8)
            dfail("sliding_window_pattern has an unexpected type");

        d->n_local = (int64_t) d->n_swa + 1;   // the python's ck[max(0, anchor - SWA):] is up to SWA + 1 cells
        d->n_ctx = std::max<int64_t>(n_ctx, 2048);
        {
            const char* w = std::getenv("XOLOTL_DSPARK_WINDOW");
            const int64_t win = w ? std::atoll(w) : 32768;
            d->n_glob = (win > 0 && win + kGlobSlack < d->n_ctx) ? win : 0;
        }

        auto need_w = [&](const std::string& n) { return need(d->wctx, n); };
        d->fuse = need_w("dspark_fuse.weight");
        d->mask_embd = need_w("mask_embd.weight");
        // NOTE: the converter's names are reversed relative to the torch ones (verified against mtp.safetensors):
        //   GGUF output_norm.weight  == torch norm.weight                  (the drafter's final norm)
        //   GGUF dspark_norm.weight  == torch markov_head.dflare_norm.weight
        d->norm = need_w("output_norm.weight");
        d->markov_norm = need_w("dspark_norm.weight");
        d->ho_w = need_w("dspark_hidden_out.weight");
        d->ho_b = need_w("dspark_hidden_out.bias");
        d->w1 = need_w("markov_w1.weight");
        d->joint = need_w("joint_proj.weight");
        d->joint_b = need_w("joint_proj.bias");
        const int8_t* pat = (const int8_t*) gguf_get_arr_data(g, pat_id);
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            DLayer& y = d->L[l];
            y.dflare = need_w(p + "dflare_norm.weight");
            y.attn_norm = need_w(p + "attn_norm.weight");
            y.ffn_norm = need_w(p + "ffn_norm.weight");
            y.q_norm = need_w(p + "attn_q_norm.weight");
            y.k_norm = need_w(p + "attn_k_norm.weight");
            y.k_norm_ctx = need_w(p + "attn_k_ctx_norm.weight");
            y.wq = need_w(p + "attn_q.weight");
            y.wk = need_w(p + "attn_k.weight");
            y.wv = need_w(p + "attn_v.weight");
            y.wo = need_w(p + "attn_output.weight");
            y.wk_ctx = need_w(p + "attn_k_ctx.weight");
            y.wv_ctx = need_w(p + "attn_v_ctx.weight");
            y.gate = need_w(p + "ffn_gate.weight");
            y.up = need_w(p + "ffn_up.weight");
            y.down = need_w(p + "ffn_down.weight");
            y.local = pat[l] != 0;
        }
        if (d->unembed->ne[0] != d->n_embd) dfail("the target LM head width does not match the drafter");
        if (d->unembed->ne[1] < d->vpad) dfail("the target LM head has fewer rows than the drafter's vocabulary");
        if (d->embed->ne[0] != d->n_embd) dfail("the target token table width does not match the drafter");

        // the 54 drafter tensors into VRAM
        d->wbuf = ggml_backend_alloc_ctx_tensors(d->wctx, d->gpu);
        if (!d->wbuf) dfail("cannot allocate the drafter in VRAM");
        ggml_backend_buffer_set_usage(d->wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        d->weight_bytes = ggml_backend_buffer_get_size(d->wbuf);
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) dfail("cannot open " + path);
        {
            std::vector<uint8_t> buf;
            const size_t base = gguf_get_data_offset(g);
            for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
                const char* nm = gguf_get_tensor_name(g, i);
                if (std::strncmp(nm, prefix.c_str(), prefix.size()) != 0) continue;   // the host model's own tensors
                ggml_tensor* t = ggml_get_tensor(d->wctx, nm + prefix.size());
                if (!t) dfail(std::string("drafter tensor not set up: ") + nm);
                buf.resize(ggml_nbytes(t));
                if (fseeko(f, (off_t) (base + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0 ||
                    std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
                    std::fclose(f);
                    dfail(std::string("short read on drafter tensor ") + t->name);
                }
                ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
            }
        }

        // host copies the markov head needs
        std::vector<float> fuse = read_vec(g, f, d->wctx, "dspark_fuse.weight");
        softmax_rows(fuse, d->h_alpha, 4, 4);
        d->h_mask = read_vec(g, f, d->wctx, "mask_embd.weight");
        d->h_norm = read_vec(g, f, d->wctx, "output_norm.weight");
        d->h_markov_norm = read_vec(g, f, d->wctx, "dspark_norm.weight");
        // the converter reversed dspark_hidden_out's dims (ggml ne = (256, 2304)), while the reference applies it
        // WITHOUT a transpose (fold_w = unembed @ hidden_out.weight).  Keep the torch order and upload the transpose.
        d->h_ho = read_vec(g, f, d->wctx, "dspark_hidden_out.weight");
        {
            std::vector<float> w1 = read_vec(g, f, d->wctx, "markov_w1.weight");   // [vpad][256]
            d->h_w1.assign(w1.begin(), w1.begin() + (long) d->vocab * d->markov_rank);
        }
        {
            const std::vector<float> j = read_vec(g, f, d->wctx, "joint_proj.weight");   // [768][5120]
            const int64_t nin = (int64_t) j.size() / 768;
            if (nin != 2 * d->markov_rank + 2 * d->n_embd) dfail("joint_proj width does not match the markov head");
            d->h_joint.assign((size_t) nin * 768, 0.0f);
            for (int64_t o = 0; o < 768; ++o)
                for (int64_t i = 0; i < nin; ++i) d->h_joint[(size_t) (i * 768 + o)] = j[(size_t) (o * nin + i)];
        }
        d->h_joint_b = read_vec(g, f, d->wctx, "joint_proj.bias");
        std::fclose(f);

        // the target's embedding norm (the block row 0 is normalised on the host)
        {
            std::vector<uint8_t> raw(ggml_nbytes(d->embed_norm));
            ggml_backend_tensor_get(d->embed_norm, raw.data(), 0, raw.size());
            d->h_embed_norm = to_f32(raw, d->embed_norm->type, d->n_embd);
        }

        // caches (cleared by dspark_reset) and the folded LM head (persistent)
        {
            ggml_init_params sp{(size_t) (2 * kLayers + 1) * ggml_tensor_overhead(), nullptr, true};
            d->sctx = ggml_init(sp);
            if (!d->sctx) dfail("cannot create the drafter state context");
            for (int l = 0; l < kLayers; ++l) {
                const int64_t cells = d->L[l].local ? cells_local(*d) : cells_global(*d);
                d->L[l].k_cache = ggml_new_tensor_3d(d->sctx, GGML_TYPE_F16, d->hd, d->n_head_kv, cells);
                d->L[l].v_cache = ggml_new_tensor_3d(d->sctx, GGML_TYPE_F16, d->hd, d->n_head_kv, cells);
                ggml_format_name(d->L[l].k_cache, "dspark_k.%d", l);
                ggml_format_name(d->L[l].v_cache, "dspark_v.%d", l);
            }
            d->sbuf = ggml_backend_alloc_ctx_tensors(d->sctx, d->gpu);
            if (!d->sbuf) dfail("cannot allocate the drafter's K/V in VRAM");
            ggml_backend_buffer_clear(d->sbuf, 0);
            d->state_bytes = ggml_backend_buffer_get_size(d->sbuf);

            ggml_init_params fp{10 * ggml_tensor_overhead(), nullptr, true};
            d->fctx = ggml_init(fp);
            if (!d->fctx) dfail("cannot create the drafter fold context");
            // fold_w is read in full by every draft slot (7 x per draft): F16 halves that traffic.  Draft quality only
            // moves the acceptance rate, never the output (verify is exact).  XOLOTL_DSPARK_FOLD_F32=1 keeps fp32.
            const ggml_type fold_t = std::getenv("XOLOTL_DSPARK_FOLD_F32") ? GGML_TYPE_F32 : GGML_TYPE_F16;
            d->fold_w = ggml_new_tensor_2d(d->fctx, fold_t, d->markov_rank, d->vocab);
            d->fold_b = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, d->vocab, 1);
            d->hoT = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, d->n_embd, d->markov_rank);
            ggml_format_name(d->fold_w, "dspark_fold_w");
            ggml_format_name(d->fold_b, "dspark_fold_b");
            ggml_format_name(d->hoT, "dspark_hoT");
            d->jctx = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, 2 * d->n_embd, 768);
            ggml_format_name(d->jctx, "dspark_jctx");
            d->base_buf = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, d->vocab, d->block);
            ggml_format_name(d->base_buf, "dspark_base");
            d->jst = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, d->markov_rank, 768);
            d->jpv = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, d->markov_rank, 768);
            d->ctx_buf = ggml_new_tensor_2d(d->fctx, GGML_TYPE_F32, 768, d->block);
            ggml_format_name(d->jst, "dspark_jst");
            ggml_format_name(d->jpv, "dspark_jpv");
            ggml_format_name(d->ctx_buf, "dspark_ctxm");
            d->fbuf = ggml_backend_alloc_ctx_tensors(d->fctx, d->gpu);
            if (!d->fbuf) dfail("cannot allocate the folded LM head in VRAM");
            ggml_backend_buffer_clear(d->fbuf, 0);
            std::vector<float> t((size_t) d->n_embd * d->markov_rank);
            for (int64_t f = 0; f < d->n_embd; ++f)
                for (int64_t l = 0; l < d->markov_rank; ++l)
                    t[(size_t) (f + l * d->n_embd)] = d->h_ho[(size_t) (f * d->markov_rank + l)];
            ggml_backend_tensor_set(d->hoT, t.data(), 0, t.size() * sizeof(float));
            {   // h_joint is [in][out] host fp32; jctx row o = inputs 512..5119 of output o
                const int64_t R2 = 2 * (int64_t) d->markov_rank, I = 2 * (int64_t) d->n_embd;
                std::vector<float> jc((size_t) I * 768);
                for (int64_t o = 0; o < 768; ++o)
                    for (int64_t k = 0; k < I; ++k) jc[(size_t) (o * I + k)] = d->h_joint[(size_t) ((R2 + k) * 768 + o)];
                ggml_backend_tensor_set(d->jctx, jc.data(), 0, jc.size() * sizeof(float));
                const int64_t R = d->markov_rank;   // the recurrence's two 256-wide inputs, [out][in] for ggml_mul_mat
                std::vector<float> js((size_t) R * 768), jp((size_t) R * 768);
                for (int64_t o = 0; o < 768; ++o)
                    for (int64_t e = 0; e < R; ++e) {
                        js[(size_t) (o * R + e)] = d->h_joint[(size_t) (e * 768 + o)];
                        jp[(size_t) (o * R + e)] = d->h_joint[(size_t) ((R + e) * 768 + o)];
                    }
                ggml_backend_tensor_set(d->jst, js.data(), 0, js.size() * sizeof(float));
                ggml_backend_tensor_set(d->jpv, jp.data(), 0, jp.size() * sizeof(float));
            }
            if (const char* e = std::getenv("XOLOTL_DSPARK_TAU")) d->tau = (float) std::atof(e);
            else d->tau = 0.35f;
            if (const char* e = std::getenv("XOLOTL_DSPARK_MAXD")) d->max_draft = std::max(1, std::min(7, std::atoi(e)));
        }

        d->gbuf.resize(ggml_tensor_overhead() * kNodes + ggml_graph_overhead_custom(kNodes, false));
        {
            ggml_backend_t bes[2] = {d->gpu, d->cpu};
            d->sched = ggml_backend_sched_new(bes, nullptr, 2, kNodes, false, true);
            if (!d->sched) dfail("cannot create the drafter scheduler");
        }
        d->anchor_taps.assign((size_t) 4 * d->n_embd, 0.0f);
        d->trow.assign((size_t) d->n_embd, 0.0f);
        d->f_rnn.assign((size_t) d->n_embd, 0.0f);

        // fold_w = (unembed @ dspark_hidden_out.weight)[:vocab] and fold_b = unembed @ hidden_out.bias, one-time.
        // fold_w's layout is ggml [markov_rank, vocab] (ne0 = the 256-rank latent), 118 GMAC on the GPU, chunked
        // over the vocabulary so the temporary stays at ~4 MB.
        {
            Graph gr = graph_begin(*d);
            ggml_context* c = gr.ctx;
            const int64_t E = d->n_embd, CH = 4096;
            for (int64_t v0 = 0; v0 < d->vocab; v0 += CH) {
                const int64_t n = std::min<int64_t>(CH, d->vocab - v0);
                ggml_tensor* us = ggml_view_2d(c, d->unembed, E, n, d->unembed->nb[1], (size_t) v0 * d->unembed->nb[1]);
                ggml_tensor* r = ggml_mul_mat(c, d->hoT, us);                 // [256, n] = ho^T @ unembed[:, v0:]
                const size_t es = ggml_element_size(d->fold_w);
                ggml_tensor* dst = ggml_view_2d(c, d->fold_w, d->markov_rank, n, (size_t) d->markov_rank * es,
                                                (size_t) v0 * d->markov_rank * es);
                ggml_build_forward_expand(gr.gf, ggml_cpy(c, r, dst));
            }
            ggml_tensor* fb = ggml_mul_mat(c, d->unembed, d->ho_b);           // [vpad, 1]
            ggml_tensor* fbv = ggml_view_2d(c, fb, d->vocab, 1, fb->nb[1], 0);
            ggml_build_forward_expand(gr.gf, ggml_cpy(c, ggml_cont(c, fbv), d->fold_b));
            graph_compute(*d, gr);
        }
        return d;
    } catch (...) {
        dspark_free(d);
        throw;
    }
}

void dspark_free(Dspark* d) {
    if (!d) return;
    if (d->sched) ggml_backend_sched_free(d->sched);
    if (d->wbuf) ggml_backend_buffer_free(d->wbuf);
    if (d->sbuf) ggml_backend_buffer_free(d->sbuf);
    if (d->fbuf) ggml_backend_buffer_free(d->fbuf);
    if (d->sctx) ggml_free(d->sctx);
    if (d->fctx) ggml_free(d->fctx);
    if (d->wctx) ggml_free(d->wctx);
    if (d->k.g) gguf_free(d->k.g);
    if (d->cpu) ggml_backend_free(d->cpu);
    delete d;
}

bool dspark_rewind(Dspark& d, int64_t p) {
    if (p > d.ctx_end) return false;
    d.ctx_end = p;   // positional caches: the cells >= p are rewritten by the next add_ctx
    return true;
}

int64_t dspark_ctx_end(const Dspark& d) { return d.ctx_end; }

void dspark_snap(const Dspark& d, DsparkSnap& s) {
    s.ctx_end = d.ctx_end;
    s.k.resize(kLayers); s.v.resize(kLayers);
    for (size_t l = 0; l < (size_t) kLayers; ++l) {
        if (!d.L[l].local && !glob_ring(d)) { s.k[l].clear(); s.v[l].clear(); continue; }
        s.k[l].resize(ggml_nbytes(d.L[l].k_cache)); s.v[l].resize(ggml_nbytes(d.L[l].v_cache));
        ggml_backend_tensor_get(d.L[l].k_cache, s.k[l].data(), 0, s.k[l].size());
        ggml_backend_tensor_get(d.L[l].v_cache, s.v[l].data(), 0, s.v[l].size());
    }
    s.anchor = d.anchor_taps;
}

void dspark_restore(Dspark& d, const DsparkSnap& s) {
    if (s.ctx_end < 0 || s.k.size() != (size_t) kLayers) dfail("dspark_restore: empty or foreign snapshot");
    for (size_t l = 0; l < (size_t) kLayers; ++l) {
        if (!d.L[l].local && s.k[l].empty()) continue;
        ggml_backend_tensor_set(d.L[l].k_cache, s.k[l].data(), 0, s.k[l].size());
        ggml_backend_tensor_set(d.L[l].v_cache, s.v[l].data(), 0, s.v[l].size());
    }
    d.anchor_taps = s.anchor;
    d.ctx_end = s.ctx_end;
}

void dspark_reset(Dspark& d) {
    if (d.sbuf) ggml_backend_buffer_clear(d.sbuf, 0);   // the K/V caches only; the folded LM head survives
    d.ctx_end = 0;
    std::fill(d.anchor_taps.begin(), d.anchor_taps.end(), 0.0f);
}

// ----------------------------------------------------------------------------------------------- context K/V
void dspark_add_ctx(Dspark& d, const float* taps, int T, int64_t p0) {
    if (T <= 0) return;
    if (p0 != d.ctx_end) dfail("dspark_add_ctx: positions " + std::to_string(p0) + "+" + std::to_string(T) +
                               " do not continue the committed context (next = " + std::to_string(d.ctx_end) + ")");
    if (p0 + T > d.n_ctx) dfail("dspark_add_ctx: the context exceeds the drafter's allocated window");
    const int E = d.n_embd;
    Graph gr = graph_begin(d);
    ggml_context* c = gr.ctx;

    ggml_tensor* tin = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, 4, T);
    ggml_set_input(tin);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    ggml_tensor* idxl = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);   // ring cells (local layers)
    ggml_set_input(idxl);
    ggml_tensor* idxg = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);   // position cells (the global layer)
    ggml_set_input(idxg);

    for (int l = 0; l < kLayers; ++l) {
        DLayer& y = d.L[l];
        // c = rms(sum_j alpha[l][j] * taps[j], dflare_norm)
        ggml_tensor* cm = nullptr;
        for (int j = 0; j < 4; ++j) {
            // tap j of every token: a strided 2-D view (rows nb[2] apart) - reshape needs contiguous, scale does not
            // ggml_cont: the CUDA scale kernel reads its source as contiguous (a bare strided view mixed the taps for t > 0)
            ggml_tensor* tv = ggml_cont(c, ggml_view_2d(c, tin, E, T, tin->nb[2], (size_t) j * tin->nb[1]));
            ggml_tensor* sc = ggml_scale(c, tv, d.h_alpha[(size_t) l * 4 + j]);
            cm = cm ? ggml_add(c, cm, sc) : sc;
        }
        cm = g_rms(c, cm, y.dflare, d.eps);
        // k/v over the whole chunk, no mask: every tap is committed and visible to the block
        ggml_tensor* k = ggml_mul_mat(c, y.wk_ctx, cm);                 // [nkv*hd, T]
        k = ggml_reshape_3d(c, k, d.hd, d.n_head_kv, T);
        k = g_rms(c, k, y.k_norm_ctx, d.eps);
        k = g_rope(c, d, k, pos);
        ggml_tensor* idx = y.local ? idxl : idxg;
        g_store_kv(c, gr.gf, k, y.k_cache, idx, T);
        ggml_tensor* v = ggml_mul_mat(c, y.wv_ctx, cm);
        v = ggml_reshape_3d(c, v, d.hd, d.n_head_kv, T);
        g_store_kv(c, gr.gf, v, y.v_cache, idx, T);
    }

    ggml_backend_sched_reset(d.sched);
    if (!ggml_backend_sched_alloc_graph(d.sched, gr.gf)) { ggml_free(gr.ctx); dfail("drafter ctx graph allocation failed"); }
    {   // host inputs
        std::vector<int32_t> p(T);
        std::vector<int64_t> il(T), ig(T);
        for (int t = 0; t < T; ++t) {
            p[t] = (int32_t) (p0 + t);
            il[t] = (p0 + t) % cells_local(d);
            ig[t] = (p0 + t) % cells_global(d);
        }
        ggml_backend_tensor_set(tin, taps, 0, (size_t) T * 4 * E * sizeof(float));
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
        ggml_backend_tensor_set(idxl, il.data(), 0, il.size() * sizeof(int64_t));
        ggml_backend_tensor_set(idxg, ig.data(), 0, ig.size() * sizeof(int64_t));
    }
    if (ggml_backend_sched_graph_compute(d.sched, gr.gf) != GGML_STATUS_SUCCESS) { ggml_free(gr.ctx); dfail("drafter ctx compute failed"); }
    ggml_free(gr.ctx);
    d.ctx_end = p0 + T;
    std::memcpy(d.anchor_taps.data(), taps + (size_t) (T - 1) * 4 * E, (size_t) 4 * E * sizeof(float));
}

// ------------------------------------------------------------------------------------------------- the block
namespace {

// one embedding row from the target's token table, as fp32
void read_embed_row(const Dspark& d, int32_t tok, float* out) {
    if (tok < 0 || tok >= d.embed->ne[1]) dfail("dspark: anchor token outside the target token table");
    const size_t rb = ggml_row_size(d.embed->type, d.n_embd);
    std::vector<uint8_t> raw(rb);
    ggml_backend_tensor_get(d.embed, raw.data(), (size_t) tok * d.embed->nb[1], rb);
    const std::vector<float> row = to_f32(raw, d.embed->type, d.n_embd);
    std::memcpy(out, row.data(), (size_t) d.n_embd * sizeof(float));
}

}  // namespace

int dspark_draft(Dspark& d, int32_t anchor, int64_t T, int32_t* out, int* n_all) {
    const int B = d.block;
    if (T <= 0 || T != d.ctx_end) dfail("dspark_draft: the committed context ends at " + std::to_string(d.ctx_end) +
                                        ", not " + std::to_string(T));
    if (T > d.n_ctx) dfail("dspark_draft: context beyond the allocated window");
    const int E = d.n_embd, nkv = d.n_head_kv, hd = d.hd;
    const int64_t nloc = std::min<int64_t>(T, d.n_local);
    const int64_t nglob = glob_ring(d) ? std::min<int64_t>(T, d.n_glob) : T;

    Graph gr = graph_begin(d);
    ggml_context* c = gr.ctx;
    ggml_tensor* xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, B);
    ggml_set_input(xin);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    ggml_set_input(pos);
    ggml_tensor* idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, nloc);
    ggml_set_input(idx);
    ggml_tensor* idxg = nullptr;
    if (glob_ring(d)) { idxg = ggml_new_tensor_1d(c, GGML_TYPE_I32, nglob); ggml_set_input(idxg); }

    ggml_tensor* x = xin;
    std::vector<std::pair<std::string, ggml_tensor*>> dbg;
    const bool dbgon = std::getenv("XOLOTL_DSPARK_DUMP") != nullptr;
    for (int l = 0; l < kLayers; ++l) {
        DLayer& y = d.L[l];
        ggml_tensor* h = g_rms(c, x, y.attn_norm, d.eps);
        ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, y.wq, h), hd, d.n_head, B);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, y.wk, h), hd, nkv, B);
        ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, y.wv, h), hd, nkv, B);
        q = g_rope(c, d, g_rms(c, q, y.q_norm, d.eps), pos);
        k = g_rope(c, d, g_rms(c, k, y.k_norm, d.eps), pos);

        ggml_tensor* Kc = nullptr;
        ggml_tensor* Vc = nullptr;
        if (y.local) {   // gather the last nloc ring cells in chronological order
            ggml_tensor* k2 = ggml_reshape_2d(c, y.k_cache, (int64_t) hd * nkv, cells_local(d));
            ggml_tensor* v2 = ggml_reshape_2d(c, y.v_cache, (int64_t) hd * nkv, cells_local(d));
            Kc = ggml_cast(c, ggml_get_rows(c, k2, idx), GGML_TYPE_F32);
            Vc = ggml_cast(c, ggml_get_rows(c, v2, idx), GGML_TYPE_F32);
            Kc = ggml_reshape_3d(c, Kc, hd, nkv, nloc);
            Vc = ggml_reshape_3d(c, Vc, hd, nkv, nloc);
        } else if (idxg) {   // the global ring: the last nglob positions in chronological order
            ggml_tensor* k2 = ggml_reshape_2d(c, y.k_cache, (int64_t) hd * nkv, cells_global(d));
            ggml_tensor* v2 = ggml_reshape_2d(c, y.v_cache, (int64_t) hd * nkv, cells_global(d));
            Kc = ggml_reshape_3d(c, ggml_cast(c, ggml_get_rows(c, k2, idxg), GGML_TYPE_F32), hd, nkv, nglob);
            Vc = ggml_reshape_3d(c, ggml_cast(c, ggml_get_rows(c, v2, idxg), GGML_TYPE_F32), hd, nkv, nglob);
        } else {
            Kc = ggml_cast(c, ggml_view_3d(c, y.k_cache, hd, nkv, T, y.k_cache->nb[1], y.k_cache->nb[2], 0), GGML_TYPE_F32);
            Vc = ggml_cast(c, ggml_view_3d(c, y.v_cache, hd, nkv, T, y.v_cache->nb[1], y.v_cache->nb[2], 0), GGML_TYPE_F32);
        }
        ggml_tensor* K = ggml_concat(c, Kc, k, 2);
        ggml_tensor* V = ggml_concat(c, Vc, v, 2);
        ggml_tensor* att = g_attend(c, d, q, K, V);
        ggml_tensor* o = ggml_mul_mat(c, y.wo, att);
        x = ggml_add(c, x, o);
        if (dbgon) {
            for (auto pr : {std::make_pair("q", q), std::make_pair("K", K), std::make_pair("V", V), std::make_pair("att", att), std::make_pair("x", x)}) {
                ggml_tensor* t = ggml_cont(c, pr.second);
                ggml_set_output(t); ggml_build_forward_expand(gr.gf, t);
                dbg.push_back({std::string(pr.first) + std::to_string(l), t});
            }
        }
        ggml_tensor* f = g_rms(c, x, y.ffn_norm, d.eps);
        ggml_tensor* act = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, y.gate, f)), ggml_mul_mat(c, y.up, f));
        x = ggml_add(c, x, ggml_mul_mat(c, y.down, act));
    }
    ggml_tensor* hidden = g_rms(c, x, d.norm, d.eps);
    ggml_set_output(hidden);
    ggml_tensor* frnn_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, 1);
    ggml_set_input(frnn_in);
    // ctx_i = [hidden_i, f_rnn] @ joint[:, 512:]^T + b   (was ~28 MMAC on one host core per draft)
    ggml_tensor* ctxg = ggml_add(c, ggml_mul_mat(c, d.jctx, ggml_concat(c, hidden, ggml_repeat(c, frnn_in, hidden), 0)), d.joint_b);
    ggml_set_output(ctxg);
    ggml_build_forward_expand(gr.gf, ctxg);
    ggml_build_forward_expand(gr.gf, ggml_cpy(c, ctxg, d.ctx_buf));     // read by the device markov graph
    ggml_tensor* base = ggml_mul_mat(c, d.unembed, hidden);                        // [vpad, B]
    ggml_tensor* bview = ggml_view_2d(c, base, d.vocab, B, base->nb[1], 0);
    base = ggml_cont(c, bview);
    ggml_set_output(base);
    ggml_build_forward_expand(gr.gf, base);
    ggml_build_forward_expand(gr.gf, ggml_cpy(c, base, d.base_buf));   // read by the markov graph (no host copy)
    ggml_build_forward_expand(gr.gf, hidden);

    ggml_backend_sched_reset(d.sched);
    if (!ggml_backend_sched_alloc_graph(d.sched, gr.gf)) { ggml_free(gr.ctx); dfail("drafter block allocation failed"); }
    {   // block rows: rms(embed(anchor), embed_norm) then the mask vector
        std::vector<float> xb((size_t) E * B);
        read_embed_row(d, anchor, d.trow.data());
        rms_row(d.trow.data(), d.h_embed_norm.data(), E, d.eps);
        std::memcpy(xb.data(), d.trow.data(), (size_t) E * sizeof(float));
        for (int r = 1; r < B; ++r) std::memcpy(xb.data() + (size_t) r * E, d.h_mask.data(), (size_t) E * sizeof(float));
        std::vector<int32_t> p(B);
        for (int i = 0; i < B; ++i) p[i] = (int32_t) (T + i);
        std::vector<int32_t> idxh((size_t) nloc);
        for (int64_t j = 0; j < nloc; ++j) idxh[(size_t) j] = (int32_t) ((T - nloc + j) % cells_local(d));
        ggml_backend_tensor_set(xin, xb.data(), 0, xb.size() * sizeof(float));
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
        ggml_backend_tensor_set(idx, idxh.data(), 0, idxh.size() * sizeof(int32_t));
        if (idxg) {
            std::vector<int32_t> ig((size_t) nglob);
            for (int64_t j = 0; j < nglob; ++j) ig[(size_t) j] = (int32_t) ((T - nglob + j) % cells_global(d));
            ggml_backend_tensor_set(idxg, ig.data(), 0, ig.size() * sizeof(int32_t));
        }
        mix_host(d, d.anchor_taps.data(), 3, d.h_markov_norm.data(), d.f_rnn.data());
        ggml_backend_tensor_set(frnn_in, d.f_rnn.data(), 0, (size_t) E * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(d.sched, gr.gf) != GGML_STATUS_SUCCESS) { ggml_free(gr.ctx); dfail("drafter block compute failed"); }
    static const bool host_mk = std::getenv("XOLOTL_DSPARK_HOST_MARKOV") != nullptr;
    std::vector<float> ctxm_gpu(host_mk ? (size_t) 768 * B : 0);   // the device markov reads ctx_buf instead
    if (host_mk) ggml_backend_tensor_get(ctxg, ctxm_gpu.data(), 0, ctxm_gpu.size() * sizeof(float));
    const bool dumping = std::getenv("XOLOTL_DSPARK_DUMP") != nullptr;
    std::vector<float> hbase(dumping ? (size_t) d.vocab * B : 0), hid((size_t) E * B);
    if (dumping) ggml_backend_tensor_get(base, hbase.data(), 0, hbase.size() * sizeof(float));
    if (dumping) ggml_backend_tensor_get(hidden, hid.data(), 0, hid.size() * sizeof(float));
    if (const char* dd = std::getenv("XOLOTL_DSPARK_DUMP")) {   // debug: block hidden + base logits for the torch diff
        auto dump = [&](const char* n, const std::vector<float>& v) {
            const std::string p = std::string(dd) + "/" + n;
            if (FILE* f = std::fopen(p.c_str(), "wb")) { std::fwrite(v.data(), sizeof(float), v.size(), f); std::fclose(f); }
        };
        dump("hidden.f32", hid);
        dump("base.f32", hbase);
    }
    if (dbgon) {
        for (auto& pr : dbg) {
            std::vector<float> v((size_t) ggml_nelements(pr.second));
            ggml_backend_tensor_get(pr.second, v.data(), 0, v.size() * sizeof(float));
            const std::string p = std::string(std::getenv("XOLOTL_DSPARK_DUMP")) + "/" + pr.first + ".f32";
            if (FILE* f = std::fopen(p.c_str(), "wb")) { std::fwrite(v.data(), sizeof(float), v.size(), f); std::fclose(f); }
            std::fprintf(stderr, "dump %s ne %lld %lld %lld\n", pr.first.c_str(), (long long) pr.second->ne[0], (long long) pr.second->ne[1], (long long) pr.second->ne[2]);
        }
    }
    ggml_free(gr.ctx);

    // ---- the markov head, as dspark_dry_validated.py lines 132-153 (ctx part computed in the block graph)
    const int R = d.markov_rank, J = 768, W = 768;   // raw width = 3*256
    const std::vector<float>& ctxm = ctxm_gpu;       // [B][768]

    // DEVICE markov (default): the whole 7-slot recurrence as ONE graph - prev = argmax feeds the next slot's
    // w1 row through get_rows on the GPU, so there is no host recurrence and no per-slot sync (the host path did a
    // sched reset + graph launch + two D2H per slot).  XOLOTL_DSPARK_HOST_MARKOV=1 = the validated host path below.
    static const bool host_markov = std::getenv("XOLOTL_DSPARK_HOST_MARKOV") != nullptr;
    if (!host_markov) {
        Graph mg = graph_begin(d);
        ggml_context* mc = mg.ctx;
        ggml_tensor* aid = ggml_new_tensor_1d(mc, GGML_TYPE_I32, 1);
        ggml_set_input(aid);
        ggml_tensor *prev = aid, *state = nullptr;
        std::vector<ggml_tensor*> ams, ses;
        for (int i = 1; i < B; ++i) {
            ggml_tensor* pe = ggml_get_rows(mc, d.w1, prev);                                   // [R, 1] f32 = w1[prev]
            ggml_tensor* raw = ggml_mul_mat(mc, d.jpv, pe);                                    // [768, 1]
            if (state) raw = ggml_add(mc, raw, ggml_mul_mat(mc, d.jst, state));
            raw = ggml_add(mc, raw, ggml_view_2d(mc, d.ctx_buf, J, 1, d.ctx_buf->nb[1], (size_t) i * d.ctx_buf->nb[1]));
            raw = ggml_cont(mc, raw);
            ggml_tensor* gt = ggml_sigmoid(mc, ggml_view_1d(mc, raw, R, 0));
            ggml_tensor* cd = ggml_tanh(mc, ggml_view_1d(mc, raw, R, (size_t) R * sizeof(float)));
            ggml_tensor* lt = ggml_tanh(mc, ggml_view_1d(mc, raw, R, (size_t) 2 * R * sizeof(float)));
            ggml_tensor* gc = ggml_mul(mc, gt, cd);
            state = state ? ggml_add(mc, ggml_sub(mc, cd, gc), ggml_mul(mc, gt, state)) : ggml_sub(mc, cd, gc);   // g*s + (1-g)*c
            ggml_tensor* t = ggml_mul_mat(mc, d.fold_w, lt);                                   // [vocab, 1]
            t = ggml_add(mc, ggml_add(mc, t, d.fold_b),
                         ggml_view_2d(mc, d.base_buf, d.vocab, 1, d.base_buf->nb[1], (size_t) i * d.base_buf->nb[1]));
            if (d.softcap > 0.0f) t = ggml_scale(mc, ggml_tanh(mc, ggml_scale(mc, t, 1.0f / d.softcap)), d.softcap);
            ggml_tensor* am = ggml_argmax(mc, t);                                                // [1] i32
            ggml_tensor* mx = ggml_get_rows(mc, ggml_reshape_2d(mc, t, 1, d.vocab), am);
            ggml_tensor* se = ggml_sum(mc, ggml_exp(mc, ggml_sub(mc, t, ggml_repeat(mc, mx, t))));
            ggml_set_output(am); ggml_set_output(se);
            ggml_build_forward_expand(mg.gf, am);
            ggml_build_forward_expand(mg.gf, se);
            ams.push_back(am); ses.push_back(se);
            prev = am;
        }
        ggml_backend_sched_reset(d.sched);
        if (!ggml_backend_sched_alloc_graph(d.sched, mg.gf)) { ggml_free(mc); dfail("drafter markov graph allocation failed"); }
        ggml_backend_tensor_set(aid, &anchor, 0, sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(d.sched, mg.gf) != GGML_STATUS_SUCCESS) { ggml_free(mc); dfail("drafter markov compute failed"); }
        int n = 0, nconf = -1;
        double cum = 1.0;
        for (int i = 1; i < B; ++i) {
            int32_t best = 0;
            float sexp = 1.0f;
            ggml_backend_tensor_get(ams[(size_t) i - 1], &best, 0, sizeof(int32_t));
            ggml_backend_tensor_get(ses[(size_t) i - 1], &sexp, 0, sizeof(float));
            out[n++] = best;
            if (n >= 2 && (d.tau > 0.0f || n > d.max_draft)) {   // same stopping rule as the host path
                if (n > d.max_draft) { --n; break; }
                cum *= 1.0 / std::max(1.0f, sexp);
                if (cum < d.tau && nconf < 0) {
                    nconf = n - 1;
                    if (!n_all) break;
                }
            }
        }
        ggml_free(mc);
        if (n_all) *n_all = n;
        return nconf < 0 ? n : nconf;
    }

    // the 256 -> vocab projection on the device, one slot at a time (the recurrence is serial)
    Graph fgr = graph_begin(d);
    {
        ggml_context* fc = fgr.ctx;
        ggml_tensor* lat = ggml_new_tensor_1d(fc, GGML_TYPE_F32, R);
        ggml_set_input(lat);
        ggml_tensor* ridx = ggml_new_tensor_1d(fc, GGML_TYPE_I32, 1);
        ggml_set_input(ridx);
        ggml_tensor* t = ggml_mul_mat(fc, d.fold_w, lat);        // [vocab, 1]
        t = ggml_add(fc, ggml_add(fc, t, d.fold_b), ggml_get_rows(fc, d.base_buf, ridx));
        if (d.softcap > 0.0f) t = ggml_scale(fc, ggml_tanh(fc, ggml_scale(fc, t, 1.0f / d.softcap)), d.softcap);
        ggml_tensor* am = ggml_argmax(fc, t);                                                       // [1] i32
        ggml_tensor* mx = ggml_get_rows(fc, ggml_reshape_2d(fc, t, 1, d.vocab), am);                // [1, 1]
        ggml_tensor* se = ggml_sum(fc, ggml_exp(fc, ggml_sub(fc, t, ggml_repeat(fc, mx, t))));      // sum exp(l - max)
        ggml_set_output(am); ggml_set_output(se);
        ggml_build_forward_expand(fgr.gf, am);
        ggml_build_forward_expand(fgr.gf, se);
        ggml_backend_sched_reset(d.sched);
        if (!ggml_backend_sched_alloc_graph(d.sched, fgr.gf)) { ggml_free(fc); dfail("drafter markov fold allocation failed"); }
        ggml_tensor* latent_in = lat;
        std::vector<float> state((size_t) R, 0.0f), cand((size_t) R), gate((size_t) R), latent((size_t) R);
        std::vector<float> raw((size_t) W);
        int32_t prev = anchor;
        int n = 0, nconf = -1;
        double cum = 1.0;
        for (int i = 1; i < B; ++i) {
            // raw = ctx_i + [state, w1[prev]] @ joint[:, :512]^T
            std::fill(raw.begin(), raw.end(), 0.0f);
            for (int e = 0; e < R; ++e) {
                const float a = state[(size_t) e];
                const float* wa = d.h_joint.data() + (size_t) e * W;
                for (int o = 0; o < W; ++o) raw[(size_t) o] += a * wa[o];
                const float b = d.h_w1[(size_t) prev * R + e];
                const float* wb = d.h_joint.data() + (size_t) (R + e) * W;
                for (int o = 0; o < W; ++o) raw[(size_t) o] += b * wb[o];
            }
            for (int o = 0; o < W; ++o) raw[(size_t) o] += ctxm[(size_t) i * J + o];
            for (int e = 0; e < R; ++e) {
                gate[(size_t) e] = 1.0f / (1.0f + std::exp(-raw[(size_t) e]));
                cand[(size_t) e] = std::tanh(raw[(size_t) (R + e)]);
                state[(size_t) e] = gate[(size_t) e] * state[(size_t) e] + (1.0f - gate[(size_t) e]) * cand[(size_t) e];
                latent[(size_t) e] = std::tanh(raw[(size_t) (2 * R + e)]);
            }
            ggml_backend_tensor_set(latent_in, latent.data(), 0, latent.size() * sizeof(float));
            const int32_t ri = i;
            ggml_backend_tensor_set(ridx, &ri, 0, sizeof(int32_t));
            if (ggml_backend_sched_graph_compute(d.sched, fgr.gf) != GGML_STATUS_SUCCESS) dfail("drafter markov fold compute failed");
            int32_t best = 0;
            float sexp = 1.0f;
            ggml_backend_tensor_get(am, &best, 0, sizeof(int32_t));
            ggml_backend_tensor_get(se, &sexp, 0, sizeof(float));
            prev = best;
            out[n++] = best;
            // dynamic draft length: draft 0 predicts the token the target already chose (never verified); from draft 1
            // on, stop once the product of the drafter's top-1 probabilities falls below tau (cheaper verifies on a MoE)
            if (n >= 2 && (d.tau > 0.0f || n > d.max_draft)) {
                if (n > d.max_draft) { --n; break; }
                cum *= 1.0 / std::max(1.0f, sexp);
                if (cum < d.tau && nconf < 0) {
                    nconf = n - 1;
                    if (!n_all) break;              // n_all: keep drafting (the caller may pad to a bit-exact shape)
                }
            }
        }
        ggml_free(fgr.ctx);
        if (n_all) *n_all = n;
        return nconf < 0 ? n : nconf;
    }
}

// ================================================================================================ standalone gate
// Built only into `xolotl-inkling-dspark` (-DXOLOTL_DSPARK_TOOL): the drafter alone, against target taps on disk.
#if defined(XOLOTL_DSPARK_TOOL)
#include <chrono>

namespace {

[[noreturn]] void tfail(const std::string& m) { std::fprintf(stderr, "xolotl-inkling-dspark: %s\n", m.c_str()); std::exit(1); }

std::vector<int32_t> parse_csv(const char* p) {
    std::vector<int32_t> out;
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p) tfail(std::string("bad id list at '") + p + "'");
        out.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p == '\0' || *p == '\r' || *p == '\n') break;
        else tfail(std::string("bad id list at '") + p + "'");
    }
    return out;
}

// read a tensor's bytes from the target GGUF straight into its backend buffer, in chunks
void read_tensor_from_file(FILE* f, gguf_context* g, ggml_tensor* t, const std::string& name) {
    const int64_t idx = gguf_find_tensor(g, name.c_str());
    if (idx < 0) tfail("target tensor missing: " + name);
    if (ggml_nbytes(t) != gguf_get_tensor_size(g, idx))
        tfail("size mismatch on target tensor " + name);
    const size_t n = ggml_nbytes(t);
    if (fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, idx)), SEEK_SET) != 0)
        tfail("seek failed on " + name);
    constexpr size_t kChunk = 64u << 20;
    std::vector<uint8_t> buf(std::min(n, kChunk));
    size_t at = 0;
    while (at < n) {
        const size_t want = std::min(n - at, buf.size());
        if (std::fread(buf.data(), 1, want, f) != want) tfail("short read on " + name);
        ggml_backend_tensor_set(t, buf.data(), at, want);
        at += want;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string target, drafter, taps_dir, ids_arg, gen_arg;
    int64_t ctx = 0;
    int64_t anchor = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) tfail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--target") target = val();
        else if (a == "--dspark") drafter = val();
        else if (a == "--taps") taps_dir = val();
        else if (a == "--ids") ids_arg = val();
        else if (a == "--gen") gen_arg = val();
        else if (a == "--ctx") ctx = std::atoll(val().c_str());
        else if (a == "--anchor") anchor = std::atoll(val().c_str());
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "usage: xolotl-inkling-dspark --target <inkling.gguf> --dspark <drafter.gguf> "
                                 "--taps <dir> --ids a,b,c [--gen a,b,c,...] [--ctx N] [--anchor N]\n");
            return 0;
        } else tfail("unknown option " + a);
    }
    if (target.empty() || drafter.empty() || taps_dir.empty() || ids_arg.empty())
        tfail("--target, --dspark, --taps and --ids are required");
    const std::vector<int32_t> ids = parse_csv(ids_arg.c_str());
    const std::vector<int32_t> gen = gen_arg.empty() ? std::vector<int32_t>() : parse_csv(gen_arg.c_str());
    if (ids.empty()) tfail("empty prompt");
    const int64_t T = (int64_t) ids.size();

    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) tfail("CUDA backend failed to start");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);

    // target GGUF: metadata + the three tensors the drafter borrows
    ggml_context* gctx = nullptr;
    gguf_init_params gp{true, &gctx};
    gguf_context* g = gguf_init_from_file(target.c_str(), gp);
    if (!g) tfail("cannot read " + target);
    FILE* f = std::fopen(target.c_str(), "rb");
    if (!f) tfail("cannot open " + target);

    ggml_context* hctx = nullptr;
    ggml_context* dctx = nullptr;
    ggml_backend_buffer_t hbuf = nullptr, dbuf = nullptr;
    {
        const int64_t ti = gguf_find_tensor(g, "token_embd.weight");
        const int64_t ni = gguf_find_tensor(g, "token_embd_norm.weight");
        const int64_t oi = gguf_find_tensor(g, "output.weight");
        if (ti < 0 || ni < 0 || oi < 0) tfail("the target GGUF lacks token_embd / token_embd_norm / output");

        ggml_init_params ip{16 * ggml_tensor_overhead(), nullptr, true};
        hctx = ggml_init(ip);
        dctx = ggml_init(ip);
        const int64_t* en = gguf_get_tensor_ne(g, ti);
        ggml_tensor* embed = ggml_new_tensor_2d(hctx, gguf_get_tensor_type(g, ti), en[0], en[1]);
        const int64_t* nn = gguf_get_tensor_ne(g, ni);
        ggml_tensor* embed_norm = ggml_new_tensor_1d(hctx, gguf_get_tensor_type(g, ni), nn[0]);
        const int64_t* on = gguf_get_tensor_ne(g, oi);
        ggml_tensor* unembed = ggml_new_tensor_2d(dctx, gguf_get_tensor_type(g, oi), on[0], on[1]);
        hbuf = ggml_backend_alloc_ctx_tensors(hctx, cpu);
        dbuf = ggml_backend_alloc_ctx_tensors(dctx, gpu);
        if (!hbuf || !dbuf) tfail("cannot allocate the target tensors");
        read_tensor_from_file(f, g, embed, "token_embd.weight");
        read_tensor_from_file(f, g, embed_norm, "token_embd_norm.weight");
        read_tensor_from_file(f, g, unembed, "output.weight");
        std::fclose(f);

        // taps: l_out-4, l_out-28, l_out-46, result_norm -> [T][4][E]
        const char* names[4] = {"l_out-4.f32", "l_out-28.f32", "l_out-46.f32", "result_norm.f32"};
        const int64_t E = embed->ne[0];
        std::vector<float> taps((size_t) T * 4 * E);
        for (int j = 0; j < 4; ++j) {
            const std::string p = taps_dir + "/" + names[j];
            FILE* tf = std::fopen(p.c_str(), "rb");
            if (!tf) tfail("cannot open tap " + p);
            std::vector<float> row((size_t) T * E);
            if (std::fread(row.data(), sizeof(float), row.size(), tf) != row.size()) tfail("short tap " + p);
            std::fclose(tf);
            for (int64_t t = 0; t < T; ++t)
                std::memcpy(taps.data() + ((size_t) t * 4 + j) * (size_t) E, row.data() + (size_t) t * E,
                            (size_t) E * sizeof(float));
        }

        const auto t0 = std::chrono::steady_clock::now();
        DsparkTarget tgt{embed, embed_norm, unembed};
        Dspark* d = dspark_load(drafter, gpu, tgt, ctx > 0 ? ctx : T + 64);
        const auto t1 = std::chrono::steady_clock::now();
        dspark_add_ctx(*d, taps.data(), (int) T, 0);
        const auto t2 = std::chrono::steady_clock::now();
        int32_t draft[16] = {};
        const int32_t anc = anchor >= 0 ? (int32_t) anchor : ids.back();
        const int n = dspark_draft(*d, anc, T, draft);
        const auto t3 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

        std::printf("anchor %d rms_norm_embed anchor_taps_ok\n", anc);
        std::printf("draft ");
        for (int i = 0; i < n; ++i) std::printf("%d ", draft[i]);
        std::printf("\n");
        if (!gen.empty()) {
            std::printf("greedy");
            for (size_t i = 0; i < gen.size(); ++i) std::printf(" %d", gen[i]);
            std::printf("\n");
            int acc = 0;
            for (int i = 0; i < n && i < (int) gen.size(); ++i) {
                if (draft[i] != gen[(size_t) i]) break;
                ++acc;
            }
            std::printf("accepted prefix: %d/%d\n", acc, n);
        }
        std::fprintf(stderr, "xolotl-inkling-dspark: load+fold %.0f ms, ctx %.0f ms, draft %.1f ms (T=%lld)\n",
                     ms(t0, t1), ms(t1, t2), ms(t2, t3), (long long) T);
        dspark_free(d);
    }
    if (hctx) ggml_free(hctx);
    if (dctx) ggml_free(dctx);
    if (hbuf) ggml_backend_buffer_free(hbuf);
    if (dbuf) ggml_backend_buffer_free(dbuf);
    if (gctx) ggml_free(gctx);
    if (g) gguf_free(g);
    if (cpu) ggml_backend_free(cpu);
    if (gpu) ggml_backend_free(gpu);
    return 0;
}
#endif
