// tools/glm/glm_dense.cpp - see glm_dense.hpp and docs/glm/ARCHITECTURE.md.
//
// One ggml graph per (layer, pass size n, MLA capacity) for attention + FFN input + router, one per (layer, n) for the
// finish (hc_post), plus init/head/predict.  The op sequence mirrors neurall/llama.cpp src/models/glm5-next.cpp
// (commit 2e0435a) so its intermediate tensors can be compared layer by layer.  Infrastructure (weight upload, one
// input upload per pass, one router readback per layer, graph reuse) follows tools/ds4/ds4_dense.cpp.

#include "glm_dense.hpp"
#include "glm_lora.hpp"   // the run-time rank-1 adapter (solo targets: attn_output, ffn_down_shexp)

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"   // ggml_cgraph::uid + ggml_graph_next_uid (graph reuse)
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"   // ggml_backend_cuda_graph_cache_clear: drop cached CUDA graphs when the pool is rebuilt
#endif

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

using namespace strata;

namespace {

constexpr float NEG_INF = -INFINITY;

// Each graph has its own allocator and never changes after it is built: allocate once, then stamp a uid so ggml-cuda
// reuses the captured CUDA graph without re-checking node properties (tools/ds4/ds4_dense.cpp alloc_graph).
bool alloc_graph(ggml_gallocr_t a, ggml_cgraph * g) {
    static const bool reuse = [] { const char * e = std::getenv("GLM_GRAPH_REUSE"); return !e || std::atoi(e) != 0; }();
    if (reuse && g->uid != 0) return true;
    if (!ggml_gallocr_alloc_graph(a, g)) return false;
    if (reuse) g->uid = ggml_graph_next_uid();
    return true;
}

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ------------------------------------------------------------------ metadata

const MetaValue * meta(const GgufFile & f, const std::string & k) { return f.get(k); }

bool meta_num(const GgufFile & f, const std::string & k, double & out) {
    const MetaValue * v = meta(f, k);
    if (!v) return false;
    if (v->is_num()) { out = v->num(); return true; }
    if (v->type == MetaType::ARRAY && !v->items.empty() && v->items[0].is_num()) { out = v->items[0].num(); return true; }
    return false;
}

bool meta_arr(const GgufFile & f, const std::string & k, std::vector<double> & out) {
    const MetaValue * v = meta(f, k);
    if (!v || v->type != MetaType::ARRAY) return false;
    out.clear();
    for (const MetaValue & e : v->items) out.push_back(e.is_num() ? e.num() : 0.0);
    return true;
}

std::string read_geometry(const GgufFile & f, GlmGeometry & g) {
    const MetaValue * a = meta(f, "general.architecture");
    if (!a || a->s != "glm5-next") return "not a glm5-next GGUF (general.architecture = " + (a ? a->s : "?") + ")";
    const std::string p = "glm5-next.";
    struct K { const char * key; int64_t * i; double * d; bool req; };
    double tmp = 0;
    const K keys[] = {
        { "block_count", &g.n_layer, nullptr, true },        { "embedding_length", &g.n_embd, nullptr, true },
        { "attention.head_count", &g.n_head, nullptr, true }, { "vocab_size", &g.vocab, nullptr, false },
        { "feed_forward_length", &g.n_ff, nullptr, true },   { "hyper_connection.count", &g.hc, nullptr, true },
        { "hyper_connection.sinkhorn_iterations", &g.hc_sinkhorn_iters, nullptr, true },
        { "hyper_connection.epsilon", nullptr, &g.hc_eps, true },
        { "attention.layer_norm_rms_epsilon", nullptr, &g.rms_eps, true },
        { "kda.head_dim", &g.kda_head_dim, nullptr, true },  { "ssm.conv_kernel", &g.d_conv, nullptr, true },
        { "kda.gate_lower_bound", nullptr, &g.kda_gate_lower_bound, false },
        { "attention.q_lora_rank", &g.q_lora, nullptr, true }, { "attention.kv_lora_rank", &g.kv_lora, nullptr, true },
        { "attention.key_length_mla", &g.k_mla, nullptr, true }, { "attention.value_length_mla", &g.v_mla, nullptr, true },
        { "attention.indexer.top_k", &g.idx_top_k, nullptr, true }, { "attention.indexer.kpool", &g.idx_kpool, nullptr, true },
        { "attention.indexer.head_count", &g.idx_heads, nullptr, true },
        { "attention.indexer.key_length", &g.idx_dim, nullptr, true },
        { "attention.layer_norm_epsilon", nullptr, &g.ln_eps, false },
        { "expert_count", &g.n_expert, nullptr, true },      { "expert_used_count", &g.n_expert_used, nullptr, true },
        { "expert_feed_forward_length", &g.n_ff_exp, nullptr, true },
        { "leading_dense_block_count", &g.n_dense_lead, nullptr, true },
        { "expert_weights_scale", nullptr, &g.expert_weights_scale, true },
        { "context_length", &g.context_length, nullptr, false },
    };
    for (const K & k : keys) {
        if (!meta_num(f, p + k.key, tmp)) {
            if (k.req) return "missing metadata " + p + k.key;
            continue;
        }
        if (k.i) *k.i = (int64_t) tmp;
        else *k.d = tmp;
    }
    // block_count includes the NextN draft block when the file ships it (Maya-S-v2: 46 = 45 decoder layers + blk.45);
    // the per-layer arrays below then have block_count entries, of which the first n_layer are the decoder's
    if (meta_num(f, p + "nextn_predict_layers", tmp) && tmp > 0 && (int64_t) tmp < g.n_layer) g.n_layer -= (int64_t) tmp;
    if (const MetaValue * v = meta(f, p + "expert_weights_norm")) g.expert_weights_norm = v->u != 0;
    if (const MetaValue * gf = meta(f, p + "expert_gating_func"); gf && gf->num() != 2)
        return "expert_gating_func " + std::to_string((int) gf->num()) + " (only sigmoid = 2 is implemented)";
    if (const MetaValue * r = meta(f, p + "rope.dimension_count"); r && r->num() != 0)
        return "rope.dimension_count != 0: GLM5-Next MLA is nope-only";
    std::vector<double> kv;
    if (!meta_arr(f, p + "attention.head_count_kv", kv) || (int64_t) kv.size() < g.n_layer)
        return "attention.head_count_kv must be a per-layer array (0 = KDA layer)";
    g.is_kda.assign((size_t) g.n_layer, false);
    for (int64_t l = 0; l < g.n_layer; ++l) g.is_kda[(size_t) l] = kv[(size_t) l] == 0;
    if (!meta_arr(f, p + "swiglu_clamp_exp", g.swiglu_clamp_exp)) g.swiglu_clamp_exp.assign((size_t) g.n_layer, 0.0);
    if (!meta_arr(f, p + "swiglu_clamp_shexp", g.swiglu_clamp_shexp)) g.swiglu_clamp_shexp = g.swiglu_clamp_exp;
    if ((int64_t) g.swiglu_clamp_exp.size() < g.n_layer || (int64_t) g.swiglu_clamp_shexp.size() < g.n_layer)
        return "swiglu_clamp arrays shorter than block_count";
    if (g.hc != 4) return "hyper_connection.count != 4 (the fused mHC ops assume 4)";
    if (g.vocab == 0) {
        const TensorInfo * te = f.find("output.weight");
        if (!te) return "no output.weight";
        g.vocab = (int64_t) te->shape.at(1);
    }
    return "";
}

// ------------------------------------------------------------------ weights (tools/ds4/ds4_dense.cpp load_weights)

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (ggml_backend_buffer_t b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "[glm_dense] missing weight tensor %s\n", n.c_str());
            std::abort();
        }
        return it->second;
    }
};

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Not loaded: the routed experts (the tier's), the MTP block.  Off-CPU also token_embd (looked
// up on the host).
bool skip_weight(const std::string & n, bool cpu, bool skip_experts, int64_t n_layer) {
    if (skip_experts && ends_with(n, "_exps.weight")) return true;
    if (n.rfind("blk.", 0) == 0 && std::atoll(n.c_str() + 4) >= n_layer) return true;
    return !cpu && n == "token_embd.weight";
}

bool load_weights(const GgufModel & model, ggml_backend_t backend, WStore & w, bool skip_experts, int64_t n_layer,
                  std::string & err, int dense_requant = -1) {
    ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    w.ctx = ggml_init(ip);
    if (!w.ctx) { err = "ggml_init(weights) failed"; return false; }
    const bool cpu = ggml_backend_is_cpu(backend);
    if (cpu) {   // bind every tensor onto the GGUF mmap
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & tensors = sh.tensors();
            if (tensors.empty()) continue;
            const uint8_t * any = sh.tensor_data(tensors[0]);
            uint8_t * base = const_cast<uint8_t *>(any) - sh.data_start() - tensors[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            if (!buf) { err = "cannot wrap " + sh.path(); return false; }
            w.bufs.push_back(buf);
            for (const auto & ti : tensors) {
                if (skip_weight(ti.name, true, skip_experts, n_layer)) continue;
                if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
                ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
                ggml_set_name(t, ti.name.c_str());
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
        return true;
    }
    // off-CPU: one backend buffer, payloads uploaded.  BF16 matrices -> Q8_0 (a BF16 mul_mat goes through cuBLAS on
    // CUDA, whose kernels land in VRAM after the expert cache was sized - tools/ds4/ds4_dense.cpp)
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip_weight(ti.name, false, skip_experts, n_layer)) continue;
            if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_type ty = (ggml_type) ti.type;
            if (ty == GGML_TYPE_BF16 && ti.shape.size() >= 2 && ne[0] % 32 == 0) ty = GGML_TYPE_Q8_0;
            if (dense_requant >= 0 && ty == GGML_TYPE_Q6_K && ti.shape.size() >= 2 && ne[0] % 256 == 0)
                ty = (ggml_type) dense_requant;   // --dense-requant (e.g. Q4_K: Maya-S-v2 -> Maya-S24's dense)
            ggml_tensor * t = ggml_new_tensor(w.ctx, ty, (int) ti.shape.size(), ne);
            ggml_set_name(t, ti.name.c_str());
            w.t[ti.name] = t;
        }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip_weight(ti.name, false, skip_experts, n_layer)) continue;
            ggml_tensor * t = w.t[ti.name];
            const uint8_t * src = model.shard(s).tensor_data(ti);
            if ((int) t->type == (int) ti.type) { ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t)); continue; }
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        }
    return true;
}

}  // namespace

// ================================================================== Impl

struct GlmDense::Impl {
    GlmGeometry g;
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    int  n_threads = 8;
    bool cpu = true;
    bool gate_taps = false;
    bool allow_long = false;
    WStore w;
    const strata::glm::LoraAdapter * lora = nullptr;   // run-time rank-1 adapter (solo targets), or NULL
    ggml_context * lctx = nullptr;                     // LoRA A/B tensors (F32, on `backend`)
    ggml_backend_buffer_t lbuf = nullptr;
    std::string err;

    int64_t D = 0, HC = 0, NH = 0, SK = 0, DI = 0, DC = 0, KVL = 0, KM = 0, VM = 0;
    int64_t NEXP = 0, NUSED = 0, NT = 1, CAPMAX = 0;
    int64_t KP = 4, IDXD = 128, IDXH = 32, PCAP = 0, NNEW = 1, MCAP = 0;   // indexer; MCAP = rows of i_mask per token

    ggml_context * sctx = nullptr;   // persistent state (sbuf)
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context * ictx = nullptr;   // input span + router output blocks
    ggml_backend_buffer_t ibuf = nullptr, obuf = nullptr, pbuf = nullptr;
    // s28 fused predict: finish(il) on one token also runs predict(il+1) into pred_sel (its own graph launch and sync
    // gone); pred_ready = the layer whose selection scores pred_sel holds now (-1: none - predict() falls back)
    ggml_tensor * pred_sel = nullptr;
    int pred_ready = -1;
    bool fused_pred = true;
    ggml_context * gctx = nullptr;   // graph node structs
    size_t gctx_bytes = 0;           // the graph pool's size (see reset_graph_cache)
    int64_t gctx_resets = 0;         // how many times the cache was rebuilt (logged; 0 in a short session)
    bool predict_stale = false;      // reset_graph_cache dropped build_predict's graphs; ensure_n builds them again

    // pass state
    ggml_tensor * x_state = nullptr;     // [D, HC, NT]
    ggml_tensor * routed_sum = nullptr;  // [D, NT]
    ggml_tensor * i_emb = nullptr;       // [D, NT]
    const uint8_t * embd_data = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<float> host_emb;

    // inputs (one upload per pass): MLA write slots + causal mask, per-layer route bias
    ggml_tensor * i_slot = nullptr;      // I32 [NT]
    ggml_tensor * i_mask = nullptr;      // F16 [MCAP * NT], viewed [cap, n] (dense attention only)
    ggml_tensor * i_tpos = nullptr;      // F32 [NT] positions (indexer pool visibility)
    ggml_tensor * i_newidx = nullptr;    // I32 [KP * NNEW] members of the pools this pass completes (into [tail | pass])
    ggml_tensor * i_newdst = nullptr;    // I32 [NNEW] their pool rows (PCAP = dump)
    ggml_tensor * i_tail = nullptr;      // F32 [(KP - 1) * NT] incomplete-pool cells per token (dump when absent)
    ggml_tensor * i_span = nullptr;
    std::vector<uint8_t> in_host;
    bool bias_dirty = true;             ///< a route bias changed since the last upload (the first upload sends them)
    uint8_t * in_base = nullptr;
    int in_pos = -1, in_n = 0;
    int64_t in_cap = -1;
    bool in_dirty = true;
    int next_pos = 0;                    // the position the recurrent state expects next
    ggml_context * snap_ctx = nullptr;   // snapshot(): copies of the recurrent state tensors
    ggml_backend_buffer_t snap_buf = nullptr;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> snap_pairs;   // (live, copy)
    int snap_pos = -1;

    int64_t o_blk = 0, o_ids_off = 0, o_wts_off = 0;

    struct Var { int64_t cap = 0, n = 1; bool idx = false; ggml_cgraph * gf = nullptr; ggml_gallocr_t allo = nullptr; };
    struct Layer {
        bool kda = false, routed = false;
        // KDA
        ggml_tensor * conv[3] = {};      // [DC-1, DI] q/k/v conv state (time innermost)
        ggml_tensor * S = nullptr;       // [SK, SK, NH] recurrent state
        // MLA
        ggml_tensor * kvc = nullptr;     // F16 [KVL, CAPMAX] latent cache
        // lightning indexer: pooled keys (row p = pool p's key, row PCAP = dump) and the last kpool-1 tokens'
        // [key | gate] rows (the members a later pass completes a pool with)
        ggml_tensor * ipool = nullptr;   // F16 [IDXD, PCAP + 1]
        ggml_tensor * itail = nullptr;   // F32 [2 * IDXD, KP - 1]
        // hand-offs
        ggml_tensor * hap = nullptr;     // [D, HC, NT]
        ggml_tensor * post_f = nullptr;  // [HC, NT]
        ggml_tensor * comb_f = nullptr;  // [HC, HC, NT]
        ggml_tensor * ffo = nullptr;     // [D, NT] shared expert (MoE layers) or dense FFN output
        ggml_tensor * t_lout = nullptr;  // [D, HC, NT] tap
        ggml_tensor * rbias = nullptr;   // F32 [NEXP] (input span)
        // run-time rank-1 LoRA for this layer's whole-module targets (NULL when the adapter has none here):
        //   _attn: `attn_output` -> A [in,1], B [1,D];   _sh: `ffn_down_shexp` -> A [n_ff_exp,1], B [1,D]
        ggml_tensor * lo_attn_a = nullptr, * lo_attn_b = nullptr;
        ggml_tensor * lo_sh_a = nullptr, * lo_sh_b = nullptr;
        ggml_tensor * o_f = nullptr, * o_i = nullptr, * o_span = nullptr;   // router output block
        std::vector<uint8_t> host_out;
        std::vector<float> host_fn, host_lout;
        std::vector<Var> vars;
        std::vector<ggml_cgraph *> gf_finish;      // [NT + 1]
        std::vector<ggml_gallocr_t> allo_finish;   // [NT + 1] (NULL for n > kSmallN: allo_big)
        ggml_cgraph * gf_predict = nullptr;
        ggml_tensor * predict_out = nullptr;
        bool fused_next = false;   // gf_finish[1] also computes predict(il+1) into pred_sel
        ggml_gallocr_t allo_predict = nullptr;
    };
    std::vector<Layer> ly;
    std::vector<ggml_cgraph *> gf_init, gf_head;
    std::vector<ggml_gallocr_t> allo_init, allo_head;
    std::vector<ggml_tensor *> logits_t;
    std::vector<float> host_logits;

    // Graphs of up to kSmallN tokens (decode, verify) keep an allocator each (no re-planning per call); prompt-chunk
    // graphs share ONE allocator (they run one after another, and one buffer per (layer, n) would not fit the card).
    static constexpr int64_t kSmallN = 4;
    ggml_gallocr_t allo_big = nullptr;
    ggml_gallocr_t new_allo() const { return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); }
    // a shared allocator re-plans every call (its buffer may grow and move, and the graphs overlap in it); the
    // per-graph ones plan once (alloc_graph's uid stamp lets ggml-cuda reuse the captured CUDA graph)
    bool alloc(ggml_gallocr_t a, ggml_cgraph * g) {
        if (a == allo_big || a == allo_rows) {
            // gallocr treats a tensor with data as allocated elsewhere: forget the pointers the last plan gave this
            // graph's own tensors (they point into a compute buffer that a later, bigger plan may have freed)
            // (compare pointers only: a stale buffer pointer must not be dereferenced)
            auto persistent = [&](ggml_backend_buffer_t bb) {
                if (bb == sbuf || bb == ibuf || bb == obuf || bb == pbuf || bb == lbuf) return true;
                for (ggml_backend_buffer_t wb : w.bufs) if (bb == wb) return true;
                return false;
            };
            auto forget = [&](ggml_tensor * t) {
                if (t->buffer && !persistent(t->buffer)) {
                    t->data = nullptr;
                    t->buffer = nullptr;
                }
            };
            for (int i = 0; i < g->n_nodes; ++i) forget(g->nodes[i]);
            for (int i = 0; i < g->n_leafs; ++i) forget(g->leafs[i]);
            return ggml_gallocr_alloc_graph(a, g);
        }
        return alloc_graph(a, g);
    }
    ggml_gallocr_t allo_for(int64_t n) {
        if (n <= kSmallN) return new_allo();
        if (!allo_big) allo_big = new_allo();
        return allo_big;
    }
    // shared per-pass hand-offs (every layer's attn -> finish runs back to back, so one copy serves all layers)
    std::vector<uint8_t> host_out;
    std::vector<float> host_fn;
    // logits_rows: rows copied out of x_state into hx, then the head over them
    static constexpr int64_t kRowsMax = 16;
    ggml_tensor * hx = nullptr;                 // [D, HC, kRowsMax]
    std::vector<ggml_cgraph *> gf_rows;         // [kRowsMax + 1]
    std::vector<ggml_tensor *> rows_t;
    ggml_gallocr_t allo_rows = nullptr;
    std::vector<uint8_t> host_x;

    ~Impl() {
        for (Layer & L : ly) {
            for (Var & v : L.vars) if (v.allo && v.allo != allo_big) ggml_gallocr_free(v.allo);
            for (ggml_gallocr_t a : L.allo_finish) if (a && a != allo_big) ggml_gallocr_free(a);
            if (L.allo_predict) ggml_gallocr_free(L.allo_predict);
        }
        for (ggml_gallocr_t a : allo_init) if (a && a != allo_big) ggml_gallocr_free(a);
        for (ggml_gallocr_t a : allo_head) if (a && a != allo_big) ggml_gallocr_free(a);
        if (allo_rows) ggml_gallocr_free(allo_rows);
        if (allo_big) ggml_gallocr_free(allo_big);
        if (snap_buf) ggml_backend_buffer_free(snap_buf);
        if (snap_ctx) ggml_free(snap_ctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        if (pbuf) ggml_backend_buffer_free(pbuf);
        if (lbuf) ggml_backend_buffer_free(lbuf);
        if (lctx) ggml_free(lctx);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }

    /// Free every cached graph and start the graph context over.  `ensure_n` caches one graph set per prompt length
    /// and per attention capacity for the whole session; a long conversation reaches many of them and the single
    /// fixed pool fills up.  ggml's allocator then returns NULL from ggml_new_object and the engine died with a NULL
    /// dereference (2026-10-22: "not enough space in the context's memory pool (needed 536919568, available
    /// 536870912)" then exit 139).  Rebuilding a graph costs milliseconds, so the cache is dropped when the pool is
    /// over half full, and the graphs a request needs are built again by ensure_n.  Only safe at a request's start,
    /// which is where ensure_n is called.
    ///
    /// The CUDA backend caches one captured graph per first-node pointer.  Freeing gctx frees the pool those
    /// pointers live in, and re-creating it at the same size hands back the same addresses, so a cached graph whose
    /// kernels still point at the freed buffers would be replayed as-is: 2026-10-22 "ds4_moe: chunk sync: an illegal
    /// memory access was encountered" and a SIGSEGV in ggml_backend_cuda_graph_compute at pool 6.5 of 512 MiB.  The
    /// cache has to be dropped here, before the pool goes away, so the next build captures afresh.
    void reset_graph_cache() {
        #ifdef GGML_USE_CUDA
        if (backend) ggml_backend_cuda_graph_cache_clear(backend);
        #endif
        for (Layer & L : ly) {
            for (Var & v : L.vars) if (v.allo && v.allo != allo_big) ggml_gallocr_free(v.allo);
            L.vars.clear();
            for (ggml_gallocr_t a : L.allo_finish) if (a && a != allo_big) ggml_gallocr_free(a);
            std::fill(L.allo_finish.begin(), L.allo_finish.end(), nullptr);
            std::fill(L.gf_finish.begin(), L.gf_finish.end(), nullptr);
            // build_predict runs once at load time and ensure_n never rebuilds it: after gctx goes away
            // L.gf_predict and L.predict_out point into freed memory, so the next predict() would hand
            // ggml_backend_cuda_graph_compute a cgraph whose nodes[0] is unmapped (2026-10-22: SEGV_MAPERR at
            // graph_compute +0xbae446).  Drop them; ensure_n builds them again in the new pool.
            if (L.allo_predict && L.allo_predict != allo_big) ggml_gallocr_free(L.allo_predict);
            L.allo_predict = nullptr;
            L.gf_predict = nullptr;
            L.predict_out = nullptr;
            L.fused_next = false;
        }
        for (ggml_gallocr_t a : allo_init) if (a && a != allo_big) ggml_gallocr_free(a);
        for (ggml_gallocr_t a : allo_head) if (a && a != allo_big) ggml_gallocr_free(a);
        std::fill(gf_init.begin(), gf_init.end(), nullptr);
        std::fill(gf_head.begin(), gf_head.end(), nullptr);
        std::fill(allo_init.begin(), allo_init.end(), nullptr);
        std::fill(allo_head.begin(), allo_head.end(), nullptr);
        std::fill(logits_t.begin(), logits_t.end(), nullptr);
        // logits_rows caches its graphs and output tensors in gf_rows/rows_t too.  clear() (not fill) so the next
        // call sees empty() and creates a fresh allocator; the graphs themselves are rebuilt lazily there.
        gf_rows.clear();
        rows_t.clear();
        if (allo_rows && allo_rows != allo_big) ggml_gallocr_free(allo_rows);
        allo_rows = nullptr;
        predict_stale = true;
        if (gctx) ggml_free(gctx);
        ggml_init_params gp = { gctx_bytes, nullptr, true };
        gctx = ggml_init(gp);
        if (!gctx) { err = "the graph cache could not be re-created"; return; }
        ++gctx_resets;
        std::fprintf(stderr, "glm: graph cache rebuilt (%lld time(s); the pool held %.1f MiB of %.1f MiB)\n",
                     (long long) gctx_resets, (double) gctx_bytes / 1048576.0, (double) gctx_bytes / 1048576.0);
    }

    // MLA through the lightning indexer once a token can see more than idx_top_k earlier positions (below that the
    // selection is every position, i.e. the dense causal attention - exact, and cheaper)
    bool use_idx(int64_t pos_last) const { return !allow_long && pos_last >= g.idx_top_k; }
    int64_t n_top_pools(int64_t cap) const { return std::min<int64_t>(cap / KP, g.idx_top_k / KP); }
    int64_t cap_for(int64_t pos_last) const {
        const int64_t c = std::max<int64_t>(cpu ? 16 : 256, next_pow2(pos_last + 1));
        return std::min<int64_t>(c, CAPMAX);
    }
    bool ensure_n(int64_t n);
};

// ================================================================== graph helpers

namespace {

struct B {
    GlmDense::Impl * im = nullptr;
    ggml_context * c = nullptr;
    const GlmGeometry * g = nullptr;

    ggml_tensor * W(const std::string & n) const { return im->w.get(n); }
    ggml_tensor * BL(int il, const std::string & s) const { return im->w.get("blk." + std::to_string(il) + "." + s); }
    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * v2(ggml_tensor * a, int64_t n0, int64_t n1, size_t nb1, size_t off) const {
        return ggml_view_2d(c, a, n0, n1, nb1, off);
    }
    ggml_tensor * fill_f32(int64_t n0, int64_t n1, float v) const {
        return ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, n0, n1), v);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wgt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wgt);
    }
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) const {
        return ggml_add(c, ggml_mul(c, x, scale), base);
    }
    // glm5-next.cpp build_hc_pre (fused ops, as with cparams.fused_dsv4_hc_*)
    ggml_tensor * hc_pre(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                         ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(c, x, hc * D, nt);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, (float) g->rms_eps));
        ggml_tensor * pre = v2(mixes, hc, nt, mixes->nb[1], 0);
        pre = ggml_sigmoid(c, hc_affine(pre, v1(sc, 1, 0), v1(bs, hc, 0)));
        pre = ggml_scale_bias(c, pre, 1.0f, (float) g->hc_eps);
        *post = v2(mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
        *post = ggml_sigmoid(c, hc_affine(*post, v1(sc, 1, sizeof(float)), v1(bs, hc, (size_t) hc * 4)));
        *post = ggml_scale(c, *post, 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_sinkhorn_iters);
        return ggml_dsv4_hc_pre(c, x, pre);
    }
    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        return ggml_dsv4_hc_post(c, x, residual, post, comb);
    }
    // y += B (A x) - the rank-1 adapter merge (A is [in,1], b is [1,out]); a no-op when the adapter has no pair.
    ggml_tensor * lora_add(ggml_tensor * y, ggml_tensor * a, ggml_tensor * b_, ggml_tensor * x) const {
        if (!a || !b_) return y;
        return ggml_add(c, y, ggml_mul_mat(c, b_, ggml_mul_mat(c, a, x)));
    }
    // SwiGLU FFN with GLM's clamp (ggml_swiglu_clamp: gate -> min(gate, lim), up -> clamp(up, +-lim)).
    // a_d/b_d (optional) = a run-time rank-1 adapter on the down projection (ffn_down_shexp).
    ggml_tensor * ffn(ggml_tensor * x, ggml_tensor * up_w, ggml_tensor * gate_w, ggml_tensor * down_w, float lim,
                      ggml_tensor * a_d = nullptr, ggml_tensor * b_d = nullptr) const {
        ggml_tensor * up = ggml_mul_mat(c, up_w, x);
        ggml_tensor * gate = ggml_mul_mat(c, gate_w, x);
        ggml_tensor * z = lim > 1e-6f ? ggml_swiglu_clamp(c, gate, up, lim) : ggml_swiglu_split(c, gate, up);
        return lora_add(ggml_mul_mat(c, down_w, z), a_d, b_d, z);
    }
    // FLA's l2 norm (glm5-next.cpp build_gdn_l2_norm)
    ggml_tensor * l2(ggml_tensor * x, float eps) const {
        const float n = (float) x->ne[0];
        return ggml_scale(c, ggml_rms_norm(c, x, eps / n), 1.0f / std::sqrt(n));
    }
};

// KDA: causal depthwise conv of one projection; the conv state keeps the last DC-1 inputs (time innermost)
ggml_tensor * kda_conv(B & b, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                       ggml_tensor * state, int64_t n) {
    GlmDense::Impl & im = *b.im;
    ggml_context * c = b.c;
    ggml_tensor * xp = ggml_mul_mat(c, proj_w, x);                                   // [DI, n]
    ggml_tensor * cx = ggml_concat(c, state, ggml_transpose(c, xp), 0);              // [DC-1+n, DI]
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_view_2d(c, cx, im.DC - 1, im.DI, cx->nb[1],
                                                           (size_t) n * ggml_element_size(cx)), state));
    ggml_tensor * w2 = ggml_reshape_2d(c, conv_w, im.DC, im.DI);
    ggml_tensor * y = ggml_ssm_conv(c, ggml_reshape_3d(c, cx, im.DC - 1 + n, im.DI, 1), w2);   // [DI, n, 1]
    y = ggml_silu(c, ggml_reshape_2d(c, y, im.DI, n));
    return ggml_reshape_4d(c, y, im.SK, im.NH, n, 1);
}

}  // namespace

// attention (KDA or MLA) + ffn hyper-connection + ffn_norm + router / shared expert (or dense FFN), n tokens
static ggml_cgraph * build_attn(GlmDense::Impl & im, int il, int64_t cap, int64_t n, bool idx) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 2048, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC, NH = im.NH;

    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * cur = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                 b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * xn = b.rms_w(cur, b.BL(il, "attn_norm.weight"));                    // [D, n]
    ggml_build_forward_expand(gf, xn);

    ggml_tensor * attn_out = nullptr;
    if (L.kda) {
        ggml_tensor * q = kda_conv(b, gf, xn, b.BL(il, "attn_q.weight"), b.BL(il, "ssm_conv1d_q.weight"), L.conv[0], n);
        ggml_tensor * k = kda_conv(b, gf, xn, b.BL(il, "attn_k.weight"), b.BL(il, "ssm_conv1d_k.weight"), L.conv[1], n);
        ggml_tensor * v = kda_conv(b, gf, xn, b.BL(il, "attn_v.weight"), b.BL(il, "ssm_conv1d_v.weight"), L.conv[2], n);
        // per-channel decay: g = lb * sigmoid(exp(A_log) * (f_b f_a x + dt_bias)); ssm_a holds -exp(A_log)
        ggml_tensor * g1 = ggml_mul_mat(gc, b.BL(il, "ssm_f_b.weight"), ggml_mul_mat(gc, b.BL(il, "ssm_f_a.weight"), xn));
        g1 = ggml_add(gc, g1, b.BL(il, "ssm_dt.bias"));
        g1 = ggml_reshape_3d(gc, g1, im.SK, NH, n);
        g1 = ggml_mul(gc, g1, ggml_reshape_3d(gc, b.BL(il, "ssm_a"), 1, NH, 1));
        g1 = ggml_sigmoid(gc, ggml_scale(gc, g1, -1.0f));
        g1 = ggml_scale(gc, g1, (float) im.g.kda_gate_lower_bound);
        g1 = ggml_reshape_4d(gc, g1, im.SK, NH, n, 1);
        ggml_tensor * beta = ggml_mul_mat(gc, b.BL(il, "ssm_beta.weight"), xn);
        beta = ggml_sigmoid(gc, ggml_reshape_4d(gc, beta, 1, NH, n, 1));
        q = b.l2(q, 1e-6f);
        k = b.l2(k, 1e-6f);
        ggml_tensor * s = ggml_reshape_4d(gc, L.S, im.SK, im.SK, NH, 1);
        ggml_tensor * res = ggml_gated_delta_net(gc, q, k, v, g1, beta, s, 1);
        ggml_tensor * o = ggml_view_4d(gc, res, im.SK, NH, n, 1, ggml_row_size(res->type, im.SK),
                                       ggml_row_size(res->type, im.SK * NH), ggml_row_size(res->type, im.SK * NH * n), 0);
        ggml_tensor * s_new = ggml_view_1d(gc, res, im.SK * im.SK * NH, ggml_row_size(res->type, im.SK * NH * n));
        ggml_build_forward_expand(gf, ggml_cpy(gc, s_new, ggml_reshape_1d(gc, L.S, im.SK * im.SK * NH)));
        // output gate: rmsnorm(o) * w * sigmoid(g_b g_a x)
        ggml_tensor * g2 = ggml_mul_mat(gc, b.BL(il, "ssm_g_b.weight"), ggml_mul_mat(gc, b.BL(il, "ssm_g_a.weight"), xn));
        g2 = ggml_reshape_3d(gc, g2, im.SK, NH, n);
        ggml_tensor * on = b.rms_w(ggml_reshape_3d(gc, ggml_cont(gc, o), im.SK, NH, n), b.BL(il, "ssm_norm.weight"));
        ggml_tensor * gated = ggml_mul(gc, on, ggml_sigmoid(gc, g2));
        ggml_tensor * ao_in = ggml_cont_2d(gc, gated, im.DI, n);
        attn_out = b.lora_add(ggml_mul_mat(gc, b.BL(il, "attn_output.weight"), ao_in), L.lo_attn_a, L.lo_attn_b, ao_in);
    } else {
        // nope MLA over every earlier position (phase 1: no indexer - exact up to dense_attn_ctx())
        ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, b.BL(il, "attn_q_a.weight"), xn), b.BL(il, "attn_q_a_norm.weight"));
        ggml_tensor * q = ggml_mul_mat(gc, b.BL(il, "attn_q_b.weight"), qr);          // [KM*NH, n]
        q = ggml_cont(gc, ggml_permute(gc, ggml_reshape_3d(gc, q, im.KM, NH, n), 0, 2, 1, 3));   // [KM, n, NH]
        ggml_tensor * q_abs = ggml_mul_mat(gc, b.BL(il, "attn_k_b.weight"), q);       // [KVL, n, NH]
        ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, b.BL(il, "attn_kv_a_mqa.weight"), xn), b.BL(il, "attn_kv_a_norm.weight"));
        ggml_tensor * kvc2 = ggml_set_rows(gc, L.kvc, kv, ggml_view_1d(gc, im.i_slot, n, 0));
        ggml_tensor * kview = ggml_view_3d(gc, kvc2, im.KVL, cap, 1, kvc2->nb[1], kvc2->nb[1] * cap, 0);
        // ---- lightning indexer (glm5-next.cpp build_kpool_select).  The pooled-key cache is kept up to date on
        // every pass, the selection only runs past idx_top_k positions.
        // NW: the pools a pass of n tokens can complete (<= n/KP + 1), not NNEW (sized by the largest chunk): decode used
        // to gather / pool / scatter max_tokens/KP + 1 pools (1025 at chunk 4096) per MLA layer per token, all but one
        // of them padding into the dump row (s26: attention +2.3 ms/token at chunk 4096).  The host fills the real
        // entries first, so the first NW are all of them.
        const int64_t KP = im.KP, ID = im.IDXD, NW = std::min<int64_t>(im.NNEW, n / im.KP + 1);
        ggml_tensor * ik = ggml_norm(gc, ggml_mul_mat(gc, b.BL(il, "indexer.attn_k.weight"), xn), (float) im.g.ln_eps);
        ik = ggml_add(gc, ggml_mul(gc, ik, b.BL(il, "indexer.k_norm.weight")), b.BL(il, "indexer.k_norm.bias"));
        ggml_tensor * ig = ggml_mul_mat(gc, b.BL(il, "indexer_compressor_gate.weight"), xn);
        ggml_tensor * kg = ggml_concat(gc, L.itail, ggml_concat(gc, ik, ig, 0), 1);         // [2ID, KP-1+n]
        ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_view_2d(gc, kg, 2 * ID, KP - 1, kg->nb[1], (size_t) n * kg->nb[1]),
                                               L.itail));
        ggml_tensor * rows = ggml_get_rows(gc, kg, ggml_view_1d(gc, im.i_newidx, KP * NW, 0));   // [2ID, KP*NW]
        ggml_tensor * pk = ggml_cont(gc, ggml_view_2d(gc, rows, ID, KP * NW, rows->nb[1], 0));
        ggml_tensor * pg = ggml_cont(gc, ggml_view_2d(gc, rows, ID, KP * NW, rows->nb[1], (size_t) ID * ggml_element_size(rows)));
        ggml_tensor * ape = b.BL(il, "indexer_compressor_ape.weight");
        if (ape->type != GGML_TYPE_F32) ape = ggml_cast(gc, ape, GGML_TYPE_F32);   // (the mini fixture's is F16)
        ggml_tensor * lg = ggml_add(gc, ggml_reshape_3d(gc, pg, ID, KP, NW), ape);
        lg = ggml_cont(gc, ggml_permute(gc, lg, 1, 0, 2, 3));                                   // [KP, ID, NW]
        ggml_tensor * pr = ggml_reshape_3d(gc, ggml_soft_max(gc, ggml_reshape_2d(gc, lg, KP, ID * NW)), KP, ID, NW);
        ggml_tensor * pkt = ggml_cont(gc, ggml_permute(gc, ggml_reshape_3d(gc, pk, ID, KP, NW), 1, 0, 2, 3));
        ggml_tensor * pooled_new = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, pr, pkt)), ID, NW);
        ggml_tensor * ipool2 = ggml_set_rows(gc, L.ipool, pooled_new, ggml_view_1d(gc, im.i_newdst, NW, 0));
        ggml_tensor * mask = nullptr;
        if (!idx) {
            ggml_build_forward_expand(gf, ipool2);
            mask = ggml_reshape_2d(gc, ggml_view_1d(gc, im.i_mask, cap * n, 0), cap, n);
        } else {
            const int64_t npc = cap / KP, ntop = im.n_top_pools(cap), nsel = KP * ntop + KP - 1;
            ggml_tensor * iq = ggml_reshape_3d(gc, ggml_mul_mat(gc, b.BL(il, "indexer.attn_q_b.weight"), qr), ID, im.IDXH, n);
            ggml_tensor * wts = ggml_scale(gc, ggml_mul_mat(gc, b.BL(il, "indexer.proj.weight"), xn),
                                           1.0f / std::sqrt((float) (ID * im.IDXH)));       // [IDXH, n]
            ggml_tensor * pv = ggml_view_3d(gc, ipool2, ID, 1, npc, ipool2->nb[1], ipool2->nb[1], 0);
            // pool p is visible to the token at t iff its last member KP*p+KP-1 <= t
            ggml_tensor * pend = ggml_arange(gc, (float) (KP - 1), (float) (KP * npc), (float) KP);   // [npc]
            ggml_tensor * tp = ggml_reshape_2d(gc, ggml_view_1d(gc, im.i_tpos, n, 0), 1, n);
            ggml_tensor * vis = ggml_sub(gc, ggml_repeat_4d(gc, ggml_reshape_2d(gc, pend, npc, 1), npc, n, 1, 1), tp);
            ggml_tensor * pmask = ggml_cast(gc, ggml_scale(gc, ggml_step(gc, vis), -65504.0f), GGML_TYPE_F16);
            ggml_tensor * score = ggml_lightning_indexer(gc, iq, pv, wts, pmask);                  // [npc, n]
            ggml_tensor * top = ggml_top_k(gc, score, (int) ntop);                                 // [ntop, n]
            // live = the selected pool is visible (a token that sees fewer than ntop pools picks masked ones too);
            // a dead slot writes its own dump row cap + slot, so every token's scatter indices stay unique
            ggml_tensor * ss = ggml_get_rows(gc, ggml_reshape_3d(gc, score, 1, npc, n), top);    // [1, ntop, n]
            ggml_tensor * live = ggml_step(gc, ggml_scale_bias(gc, ss, 1.0f, 30000.0f));
            ggml_tensor * cells = ggml_scale(gc, ggml_reshape_3d(gc, ggml_cast(gc, top, GGML_TYPE_F32), 1, ntop, n), (float) KP);
            cells = ggml_add(gc, ggml_repeat_4d(gc, cells, KP, ntop, n, 1),
                             ggml_reshape_3d(gc, ggml_arange(gc, 0.0f, (float) KP, 1.0f), KP, 1, 1));
            ggml_tensor * dump = ggml_reshape_3d(gc, ggml_arange(gc, (float) cap, (float) (cap + KP * ntop), 1.0f), KP, ntop, 1);
            ggml_tensor * sel = ggml_add(gc, ggml_mul(gc, ggml_sub(gc, cells, dump), ggml_repeat_4d(gc, live, KP, ntop, n, 1)), dump);
            sel = ggml_concat(gc, ggml_reshape_2d(gc, sel, KP * ntop, n),
                              ggml_reshape_2d(gc, ggml_view_1d(gc, im.i_tail, (KP - 1) * n, 0), KP - 1, n), 0);   // [nsel, n]
            sel = ggml_cast(gc, sel, GGML_TYPE_I32);
            // rows padded to 8 halfs: CUDA flash_attn (head 512) needs every mask stride % 16 bytes == 0
            ggml_tensor * mall = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F16, 1, cap + (nsel + 7) / 8 * 8, n), NEG_INF);
            ggml_tensor * zeros = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, 1, nsel, n), 0.0f);
            mall = ggml_set_rows(gc, mall, zeros, ggml_reshape_3d(gc, sel, nsel, n, 1));
            mask = ggml_view_2d(gc, mall, cap, n, mall->nb[2], 0);
            if (n > 1) mask = ggml_cont(gc, mask);   // flash_attn_ext wants a contiguous mask (rows are cap + nsel apart)
        }
        ggml_tensor * out = ggml_flash_attn_ext(gc, q_abs, kview, kview, mask, 1.0f / std::sqrt((float) im.KM), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);                              // [KVL, NH, n]
        out = ggml_cont(gc, ggml_permute(gc, out, 0, 2, 1, 3));                        // [KVL, n, NH]
        out = ggml_mul_mat(gc, b.BL(il, "attn_v_b.weight"), out);                      // [VM, n, NH]
        out = ggml_cont_2d(gc, ggml_permute(gc, out, 0, 2, 1, 3), im.VM * NH, n);      // [VM*NH, n]
        attn_out = b.lora_add(ggml_mul_mat(gc, b.BL(il, "attn_output.weight"), out), L.lo_attn_a, L.lo_attn_b, out);
    }

    ggml_tensor * hap = b.hc_post(attn_out, xin, post_a, comb_a);
    ggml_tensor * post_f = nullptr, * comb_f = nullptr;
    ggml_tensor * fpre = b.hc_pre(hap, b.BL(il, "hc_ffn_fn.weight"), b.BL(il, "hc_ffn_scale.weight"),
                                  b.BL(il, "hc_ffn_base.weight"), &post_f, &comb_f);
    ggml_tensor * fn = b.rms_w(fpre, b.BL(il, "ffn_norm.weight"));                    // [D, n]

    const size_t ob = (size_t) im.o_blk;
    const float lim_sh = (float) im.g.swiglu_clamp_shexp[(size_t) il];
    ggml_tensor * ffo = nullptr;
    if (L.routed) {
        ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fn);
        ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
        ggml_tensor * probs = ggml_sigmoid(gc, rlog);                                  // [NEXP, n]
        ggml_tensor * sel = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
        sel = ggml_add(gc, sel, ggml_reshape_2d(gc, L.rbias, im.NEXP, 1));             // cache-aware bias (0 = model)
        ggml_tensor * selected = ggml_argsort_top_k(gc, sel, (int) im.NUSED);         // [NUSED, n]
        ggml_tensor * wts = ggml_get_rows(gc, ggml_reshape_3d(gc, probs, 1, im.NEXP, n), selected);
        wts = ggml_reshape_2d(gc, wts, im.NUSED, n);
        if (im.g.expert_weights_norm) {
            ggml_tensor * wsum = ggml_clamp(gc, ggml_sum_rows(gc, wts), 6.103515625e-5f, INFINITY);
            wts = ggml_div(gc, wts, wsum);
        }
        wts = ggml_scale(gc, wts, (float) im.g.expert_weights_scale);
        ggml_build_forward_expand(gf, ggml_cpy(gc, selected, ggml_view_2d(gc, L.o_i, im.NUSED, n, ob, (size_t) im.o_ids_off)));
        ggml_build_forward_expand(gf, ggml_cpy(gc, wts, ggml_view_2d(gc, L.o_f, im.NUSED, n, ob, (size_t) im.o_wts_off)));
        ffo = b.ffn(fn, b.BL(il, "ffn_up_shexp.weight"), b.BL(il, "ffn_gate_shexp.weight"),
                    b.BL(il, "ffn_down_shexp.weight"), lim_sh, L.lo_sh_a, L.lo_sh_b);
    } else {
        ffo = b.ffn(fn, b.BL(il, "ffn_up.weight"), b.BL(il, "ffn_gate.weight"), b.BL(il, "ffn_down.weight"), lim_sh);
    }
    auto cols2 = [&](ggml_tensor * t, int64_t n0) { return ggml_view_2d(gc, t, n0, n, t->nb[1], 0); };
    auto cols3 = [&](ggml_tensor * t, int64_t n0, int64_t n1) { return ggml_view_3d(gc, t, n0, n1, n, t->nb[1], t->nb[2], 0); };
    ggml_build_forward_expand(gf, ggml_cpy(gc, fn, ggml_view_2d(gc, L.o_f, D, n, ob, 0)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_3d(gc, hap, D, HC, n), cols3(L.hap, D, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, post_f, cols2(L.post_f, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_3d(gc, comb_f, HC, HC, n), cols3(L.comb_f, HC, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ffo, cols2(L.ffo, D)));

    GlmDense::Impl::Var v;
    v.cap = cap;
    v.n = n;
    v.idx = idx;
    v.gf = gf;
    v.allo = im.allo_for(n);
    L.vars.push_back(v);
    return gf;
}

// ffn_out = ffo (+ routed sum on MoE layers); l_out = hc_post(ffn_out, hap, post_f, comb_f)
static ggml_cgraph * build_finish(GlmDense::Impl & im, int il, int64_t n) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * ffo = ggml_view_2d(gc, L.ffo, D, n, L.ffo->nb[1], 0);
    if (L.routed) ffo = ggml_add(gc, ffo, ggml_view_2d(gc, im.routed_sum, D, n, im.routed_sum->nb[1], 0));
    ggml_tensor * hap = ggml_view_3d(gc, L.hap, D, HC, n, L.hap->nb[1], L.hap->nb[2], 0);
    ggml_tensor * pf = ggml_view_2d(gc, L.post_f, HC, n, L.post_f->nb[1], 0);
    ggml_tensor * cf = ggml_view_3d(gc, L.comb_f, HC, HC, n, L.comb_f->nb[1], L.comb_f->nb[2], 0);
    ggml_tensor * lout = ggml_reshape_3d(gc, b.hc_post(ffo, hap, pf, cf), D, HC, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, lout, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                                  im.x_state->nb[2], 0)));
    if (im.gate_taps)
        ggml_build_forward_expand(gf, ggml_cpy(gc, lout, ggml_view_3d(gc, L.t_lout, D, HC, n, L.t_lout->nb[1],
                                                                      L.t_lout->nb[2], 0)));
    // fused predict (s28): the next routed layer's selection scores from this layer's output - build_predict's ops on
    // the values x_state now receives (lout), so the scores are the same; GlmDense::predict(il+1) only reads them
    if (n == 1 && im.pred_sel && il + 1 < (int) im.ly.size() && im.ly[(size_t) il + 1].routed) {
        const int nl = il + 1;
        ggml_tensor * post = nullptr, * comb = nullptr;
        ggml_tensor * pre = b.hc_pre(lout, b.BL(nl, "hc_attn_fn.weight"), b.BL(nl, "hc_attn_scale.weight"),
                                     b.BL(nl, "hc_attn_base.weight"), &post, &comb);
        ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(nl, "ffn_gate_inp.weight"), b.rms_w(pre, b.BL(nl, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
        ggml_tensor * sc = ggml_add(gc, ggml_sigmoid(gc, rlog), b.BL(nl, "exp_probs_b.bias"));
        ggml_build_forward_expand(gf, ggml_cpy(gc, sc, ggml_reshape_2d(gc, im.pred_sel, im.NEXP, 1)));
        L.fused_next = true;
    }
    L.gf_finish[(size_t) n] = gf;
    L.allo_finish[(size_t) n] = im.allo_for(n);
    return gf;
}

// predict: the layer's router on rms_norm(hc_attn_pre) * ffn_norm.weight for token 0 (selection scores)
static ggml_cgraph * build_predict(GlmDense::Impl & im, int il) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, im.D, im.HC, 1, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post = nullptr, * comb = nullptr;
    ggml_tensor * pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                 b.BL(il, "hc_attn_base.weight"), &post, &comb);
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), b.rms_w(pre, b.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * out = ggml_add(gc, ggml_sigmoid(gc, rlog), b.BL(il, "exp_probs_b.bias"));
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    L.predict_out = out;
    L.gf_predict = gf;
    L.allo_predict = im.new_allo();
    return gf;
}

static ggml_cgraph * build_init(GlmDense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    ggml_tensor * x = ggml_view_3d(gc, im.i_emb, im.D, 1, n, im.i_emb->nb[1], im.i_emb->nb[1], 0);
    x = ggml_repeat_4d(gc, x, im.D, im.HC, n, 1);
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, ggml_view_3d(gc, im.x_state, im.D, im.HC, n, im.x_state->nb[1],
                                                               im.x_state->nb[2], 0)));
    return gf;
}

// head: mean of the hc streams -> output_norm -> output (glm5-next.cpp: build_hc_mean, no hc_head weights)
static ggml_cgraph * build_head(GlmDense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    B b { &im, gc, &im.g };
    ggml_tensor * x = ggml_view_3d(gc, im.x_state, im.D, im.HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * acc = ggml_view_2d(gc, x, im.D, n, x->nb[2], 0);
    for (int64_t s = 1; s < im.HC; ++s) acc = ggml_add(gc, acc, ggml_view_2d(gc, x, im.D, n, x->nb[2], (size_t) s * x->nb[1]));
    acc = ggml_scale(gc, acc, 1.0f / (float) im.HC);
    ggml_tensor * lg = ggml_mul_mat(gc, b.W("output.weight"), b.rms_w(acc, b.W("output_norm.weight")));
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    im.logits_t[(size_t) n] = lg;
    return gf;
}

bool GlmDense::Impl::ensure_n(int64_t n) {
    if (n < 1 || n > NT) { err = "pass size " + std::to_string(n) + " out of range 1.." + std::to_string(NT); return false; }
    if (!gf_init[(size_t) n] && gctx_bytes && ggml_used_mem(gctx) * 2 > gctx_bytes) reset_graph_cache();
    if (!gctx) return false;
    if (predict_stale) {   // reset_graph_cache dropped build_predict's graphs (they lived in the old pool)
        for (size_t il = 0; il < ly.size(); ++il)
            if (ly[il].routed) build_predict(*this, (int) il);
        predict_stale = false;
    }
    if (!gf_init[(size_t) n]) {
        gf_init[(size_t) n] = build_init(*this, n);
        allo_init[(size_t) n] = allo_for(n);
        if (n <= kSmallN) {   // a prompt chunk reads its logits through logits_rows
            gf_head[(size_t) n] = build_head(*this, n);
            allo_head[(size_t) n] = new_allo();
        }
    }
    for (int il = 0; il < (int) ly.size(); ++il)
        if (!ly[(size_t) il].gf_finish[(size_t) n]) build_finish(*this, il, n);
    return true;
}

// ================================================================== public API

GlmDense::GlmDense() : p_(new Impl()) {}
GlmDense::~GlmDense() = default;

const GlmGeometry & GlmDense::geom() const { return p_->g; }
int64_t GlmDense::n_layer() const { return p_->g.n_layer; }
int GlmDense::max_tokens() const { return (int) p_->NT; }
ggml_backend_t GlmDense::backend() const { return p_->backend; }
const std::string & GlmDense::last_error() const { return p_->err; }

bool GlmDense::init(const std::string & model_path, const GlmDenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.n_threads = cfg.n_threads;
    im.gate_taps = cfg.gate_taps;
    im.allow_long = cfg.allow_long_ctx;
    im.backend = cfg.backend;
    if (!im.backend) {
        im.backend = ggml_backend_cpu_init();
        im.own_backend = true;
        if (!im.backend) { err = "cannot create a CPU backend"; return false; }
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (im.cpu) ggml_backend_cpu_set_n_threads(im.backend, im.n_threads);

    im.model = std::make_unique<GgufModel>(GgufModel::open(model_path));
    if (im.model->size() == 0) { err = "cannot open " + model_path; return false; }
    if (!(err = read_geometry(im.model->shard(0), im.g)).empty()) return false;
    const GlmGeometry & g = im.g;
    if (!load_weights(*im.model, im.backend, im.w, cfg.skip_routed_experts, g.n_layer, err, cfg.dense_requant)) return false;
    {
        size_t sh = 0;
        const TensorInfo * te = im.model->find("token_embd.weight", &sh);
        if (!te) { err = "no token_embd.weight"; return false; }
        im.embd_data = im.model->shard(sh).tensor_data(*te);
        im.embd_type = (int) te->type;
        im.embd_row = ggml_row_size((ggml_type) te->type, (int64_t) te->shape.at(0));
    }
    im.D = g.n_embd; im.HC = g.hc; im.NH = g.n_head; im.SK = g.kda_head_dim; im.DI = g.kda_head_dim * g.n_head;
    im.DC = g.d_conv; im.KVL = g.kv_lora; im.KM = g.k_mla; im.VM = g.v_mla;
    im.NEXP = g.n_expert; im.NUSED = g.n_expert_used;
    im.NT = std::max(1, cfg.max_tokens);
    // the MLA cache holds ctx rounded up to 256 (was: to a power of two - --ctx 307200 then allocated 512K, ~2.2 GiB of
    // VRAM the expert slots never got).  The per-step buckets (cap_for) stay powers of two below it; only the last
    // bucket is CAPMAX itself.  256 keeps PCAP = CAPMAX / KP whole and the KV length a multiple of FA's 256 stride.
    im.CAPMAX = std::max<int64_t>(im.cpu ? 16 : 256, (std::max<int64_t>(1, cfg.ctx) + 255) / 256 * 256);
    im.KP = g.idx_kpool; im.IDXD = g.idx_dim; im.IDXH = g.idx_heads;
    im.PCAP = im.CAPMAX / im.KP;
    im.NNEW = std::max<int64_t>(1, cfg.max_tokens) / im.KP + 1;
    // the dense attention mask is only needed while the whole context fits the indexer's selection
    im.MCAP = im.allow_long ? im.CAPMAX : std::min<int64_t>(im.CAPMAX, std::max<int64_t>(im.cpu ? 16 : 256, g.idx_top_k));
    if (im.NT > 8192) { err = "max_tokens > 8192"; return false; }   // 8192: helios prefills in 8K chunks

    // ---- persistent state
    ggml_init_params ip = { 64ull * 1024 * 1024, nullptr, true };
    im.sctx = ggml_init(ip);
    im.ictx = ggml_init(ip);
    if (!im.sctx || !im.ictx) { err = "ggml_init(state) failed"; return false; }
    const int64_t NT = im.NT;
    im.x_state = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
    im.routed_sum = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
    im.i_emb = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
    im.ly.resize((size_t) g.n_layer);
    for (int64_t il = 0; il < g.n_layer; ++il) {
        Impl::Layer & L = im.ly[(size_t) il];
        L.kda = g.kda(il);
        L.routed = g.routed(il);
        if (L.kda) {
            for (auto & cs : L.conv) cs = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.DC - 1, im.DI);
            L.S = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.SK, im.SK, im.NH);
        } else {
            L.kvc = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F16, im.KVL, im.CAPMAX);
            L.ipool = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F16, im.IDXD, im.PCAP + 1);
            L.itail = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, 2 * im.IDXD, im.KP - 1);
        }
        if (il == 0) {   // one set for all layers: layer l's attn -> experts -> finish completes before l+1 starts
            L.hap = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
            L.post_f = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.HC, NT);
            L.comb_f = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.HC, im.HC, NT);
            L.ffo = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
        } else {
            const Impl::Layer & L0 = im.ly[0];
            L.hap = L0.hap; L.post_f = L0.post_f; L.comb_f = L0.comb_f; L.ffo = L0.ffo;
        }
        L.gf_finish.assign((size_t) NT + 1, nullptr);
        L.allo_finish.assign((size_t) NT + 1, nullptr);
        if (im.gate_taps) {
            L.t_lout = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
            L.host_lout.resize((size_t) (im.D * im.HC));
        }
    }
    im.host_fn.resize((size_t) (im.D * NT));
    im.hx = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, Impl::kRowsMax);

    // ---- run-time rank-1 adapter: the dense half's whole-module targets (`attn_output`, `ffn_down_shexp`).  Each
    // pair is a bare linear merge y += B (A x), so it is two tiny mul_mats added in the graph; the A [in,1] / B [1,out]
    // payloads sit in their own backend buffer on the same backend, so the merge stays on device.
    im.lora = cfg.lora;
    if (im.lora) {
        ggml_init_params lp = { 8ull * 1024 * 1024, nullptr, true };
        im.lctx = ggml_init(lp);
        if (!im.lctx) { err = "ggml_init(lora) failed"; return false; }
        auto mk_pair = [&](const strata::glm::Lora1 * p, ggml_tensor ** pa, ggml_tensor ** pb) {
            if (!p) { *pa = *pb = nullptr; return; }
            *pa = ggml_new_tensor_2d(im.lctx, GGML_TYPE_F32, (int64_t) p->a.size(), 1);
            *pb = ggml_new_tensor_2d(im.lctx, GGML_TYPE_F32, 1, (int64_t) p->b.size());
        };
        for (int64_t il = 0; il < g.n_layer; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            mk_pair(im.lora->solo(il, strata::glm::LoraAdapter::ATTN_OUT), &L.lo_attn_a, &L.lo_attn_b);
            mk_pair(im.lora->solo(il, strata::glm::LoraAdapter::SHEXP_DOWN), &L.lo_sh_a, &L.lo_sh_b);
        }
        im.lbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.lctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!im.lbuf) { err = "cannot allocate the LoRA tensors"; return false; }
        for (int64_t il = 0; il < g.n_layer; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            const strata::glm::Lora1 * p;
            if ((p = im.lora->solo(il, strata::glm::LoraAdapter::ATTN_OUT))) {
                ggml_backend_tensor_set(L.lo_attn_a, p->a.data(), 0, p->a.size() * sizeof(float));
                ggml_backend_tensor_set(L.lo_attn_b, p->b.data(), 0, p->b.size() * sizeof(float));
            }
            if ((p = im.lora->solo(il, strata::glm::LoraAdapter::SHEXP_DOWN))) {
                ggml_backend_tensor_set(L.lo_sh_a, p->a.data(), 0, p->a.size() * sizeof(float));
                ggml_backend_tensor_set(L.lo_sh_b, p->b.data(), 0, p->b.size() * sizeof(float));
            }
        }
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.backend));
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);

    // ---- input span: i_slot, i_mask, every layer's rbias, back to back
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
    const size_t al = std::max<size_t>(64, ggml_backend_buft_get_alignment(buft));
    auto up = [&](size_t n) { return (n + al - 1) / al * al; };
    std::vector<ggml_tensor *> in_list;
    im.i_slot = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, NT);
    im.i_mask = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F16, im.MCAP * NT);
    im.i_tpos = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, NT);
    im.i_newidx = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, im.KP * im.NNEW);
    im.i_newdst = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, im.NNEW);
    im.i_tail = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, (im.KP - 1) * NT);
    in_list = { im.i_slot, im.i_mask, im.i_tpos, im.i_newidx, im.i_newdst, im.i_tail };
    for (Impl::Layer & L : im.ly) { L.rbias = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, im.NEXP); in_list.push_back(L.rbias); }
    size_t in_bytes = 0;
    for (ggml_tensor * t : in_list) in_bytes += up(ggml_nbytes(t));
    im.ibuf = ggml_backend_buft_alloc_buffer(buft, in_bytes + al);
    if (!im.ibuf) { err = "cannot allocate the input span"; return false; }
    im.in_base = (uint8_t *) ggml_backend_buffer_get_base(im.ibuf);
    {
        size_t off = 0;
        for (ggml_tensor * t : in_list) {
            if (ggml_backend_tensor_alloc(im.ibuf, t, im.in_base + off) != GGML_STATUS_SUCCESS) { err = "cannot place an input"; return false; }
            off += up(ggml_nbytes(t));
        }
        im.i_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) in_bytes);
        if (ggml_backend_tensor_alloc(im.ibuf, im.i_span, im.in_base) != GGML_STATUS_SUCCESS) { err = "cannot place the input span"; return false; }
        im.in_host.assign(in_bytes, 0);
    }
    // ---- router output blocks: per layer NT x [fn | ids | wts], one readback per layer
    const size_t o1 = up(ggml_row_size(GGML_TYPE_F32, im.D)), o2 = o1 + up(4 * im.NUSED), blk = o2 + up(4 * im.NUSED);
    im.o_blk = (int64_t) blk; im.o_ids_off = (int64_t) o1; im.o_wts_off = (int64_t) o2;
    const size_t lblk = blk * (size_t) NT;
    im.obuf = ggml_backend_buft_alloc_buffer(buft, lblk + al);   // shared by every layer (see the hand-offs)
    if (!im.obuf) { err = "cannot allocate the router output blocks"; return false; }
    {
        uint8_t * ob = (uint8_t *) ggml_backend_buffer_get_base(im.obuf);
        for (int64_t il = 0; il < g.n_layer; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            uint8_t * b0 = ob;
            L.o_f = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, (int64_t) (lblk / 4));
            L.o_i = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, (int64_t) (lblk / 4));
            L.o_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) lblk);
            for (ggml_tensor * t : { L.o_f, L.o_i, L.o_span })
                if (ggml_backend_tensor_alloc(im.obuf, t, b0) != GGML_STATUS_SUCCESS) { err = "cannot place a router block"; return false; }
        }
        im.host_out.assign(lblk, 0);
    }
    im.fused_pred = std::getenv("GLM_NO_FUSED_PREDICT") == nullptr;
    if (im.fused_pred) {
        im.pbuf = ggml_backend_buft_alloc_buffer(buft, (size_t) im.NEXP * 4 + al);
        if (!im.pbuf) { err = "cannot allocate the fused predict output"; return false; }
        im.pred_sel = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, im.NEXP);
        if (ggml_backend_tensor_alloc(im.pbuf, im.pred_sel, ggml_backend_buffer_get_base(im.pbuf)) != GGML_STATUS_SUCCESS) {
            err = "cannot place the fused predict output";
            return false;
        }
    }

    ggml_init_params gp = { 512ull * 1024 * 1024, nullptr, true };
    im.gctx_bytes = (size_t) gp.mem_size;
    im.gctx = ggml_init(gp);
    if (!im.gctx) { err = "ggml_init(graph) failed"; return false; }
    im.gf_init.assign((size_t) NT + 1, nullptr);
    im.gf_head.assign((size_t) NT + 1, nullptr);
    im.allo_init.assign((size_t) NT + 1, nullptr);
    im.allo_head.assign((size_t) NT + 1, nullptr);
    im.logits_t.assign((size_t) NT + 1, nullptr);
    if (!im.ensure_n(1)) { err = im.err; return false; }
    for (int64_t il = 0; il < g.n_layer; ++il)
        if (im.ly[(size_t) il].routed) build_predict(im, (int) il);
    im.host_logits.resize((size_t) (g.vocab * NT));
    return true;
}

void GlmDense::reset() {
    Impl & im = *p_;
    ggml_backend_synchronize(im.backend);
    ggml_backend_buffer_clear(im.sbuf, 0);
    im.pred_ready = -1;
    im.next_pos = 0;
    im.in_pos = -1;
    im.snap_pos = -1;
}

bool GlmDense::snapshot() {
    Impl & im = *p_;
    if (!im.snap_buf) {
        std::vector<ggml_tensor *> live;
        for (Impl::Layer & L : im.ly) {
            if (L.kda) { live.push_back(L.S); for (ggml_tensor * c : L.conv) live.push_back(c); }
            else if (L.itail) live.push_back(L.itail);
        }
        ggml_init_params ip = { ggml_tensor_overhead() * (live.size() + 1), nullptr, true };
        im.snap_ctx = ggml_init(ip);
        if (!im.snap_ctx) { im.err = "snapshot: ggml_init failed"; return false; }
        for (ggml_tensor * t : live) im.snap_pairs.emplace_back(t, ggml_dup_tensor(im.snap_ctx, t));
        im.snap_buf = ggml_backend_alloc_ctx_tensors_from_buft(im.snap_ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!im.snap_buf) { im.err = "snapshot: cannot allocate the state copy"; return false; }
    }
    ggml_backend_synchronize(im.backend);
    for (auto & pr : im.snap_pairs) ggml_backend_tensor_copy(pr.first, pr.second);
    ggml_backend_synchronize(im.backend);
    im.snap_pos = im.next_pos;
    return true;
}

bool GlmDense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) { im.err = "restore: no snapshot"; return false; }
    ggml_backend_synchronize(im.backend);
    for (auto & pr : im.snap_pairs) ggml_backend_tensor_copy(pr.second, pr.first);
    ggml_backend_synchronize(im.backend);
    im.next_pos = im.snap_pos;
    im.in_pos = -1;
    return true;
}

int GlmDense::snapshot_pos() const { return p_->snap_pos; }

bool GlmDense::begin_tokens(const int * tids, int n) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!im.allo_init[(size_t) n]) im.allo_init[(size_t) n] = im.allo_for(n);   // after release_big
    if (!im.alloc(im.allo_init[(size_t) n], im.gf_init[(size_t) n])) { im.err = "gallocr(init) failed"; return false; }
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint8_t * row = im.embd_data + (size_t) tids[t] * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_synchronize(im.backend);   // host_emb's previous upload / the previous token's work
    im.pred_ready = -1;
    ggml_backend_tensor_set_async(im.backend, im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (ggml_backend_graph_compute_async(im.backend, im.gf_init[(size_t) n]) != GGML_STATUS_SUCCESS) {
        im.err = "init graph compute failed";
        return false;
    }
    return true;
}

bool GlmDense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n_top <= 0) { im.err = "predict: bad args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;
    if (!L.routed) return false;
    if (!L.gf_predict || !L.predict_out) { im.err = "predict: graph missing"; return false; }
    std::vector<float> sel((size_t) ggml_nelements(L.predict_out));
    if (im.pred_ready == il) {   // finish(il-1) already computed it
        ggml_backend_tensor_get_async(im.backend, im.pred_sel, sel.data(), 0, sel.size() * 4);
    } else {
        if (!im.alloc(L.allo_predict, L.gf_predict)) { im.err = "gallocr(predict) failed"; return false; }
        if (ggml_backend_graph_compute_async(im.backend, L.gf_predict) != GGML_STATUS_SUCCESS) { im.err = "predict compute failed"; return false; }
        ggml_backend_tensor_get_async(im.backend, L.predict_out, sel.data(), 0, sel.size() * 4);
    }
    im.pred_ready = -1;
    ggml_backend_synchronize(im.backend);
    std::vector<int> idx(sel.size());
    for (size_t i = 0; i < sel.size(); ++i) idx[i] = (int) i;
    const int k = std::min<int>(n_top, (int) sel.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    for (int i = 0; i < k; ++i) top_ids[i] = idx[(size_t) i];
    return true;
}

// The prompt chunks' shared compute buffer (allo_big, sized by the largest chunk) back to the device: after a prompt
// its VRAM is the expert cache's for decode (s26 per-request elastic cache).  The cached graphs keep their plans;
// the next chunk re-acquires an allocator (allo_for) and alloc() re-plans it (it forgets the stale pointers).
void GlmDense::release_big() {
    Impl & im = *p_;
    if (!im.allo_big) return;
    ggml_backend_synchronize(im.backend);
    for (Impl::Layer & L : im.ly) {
        for (Impl::Var & v : L.vars) if (v.allo == im.allo_big) v.allo = nullptr;
        for (ggml_gallocr_t & a : L.allo_finish) if (a == im.allo_big) a = nullptr;
    }
    for (ggml_gallocr_t & a : im.allo_init) if (a == im.allo_big) a = nullptr;
    ggml_gallocr_free(im.allo_big);
    im.allo_big = nullptr;
}

int GlmDense::safe_chunk(int pos0, int want) const {
    const Impl & im = *p_;
    static const int64_t lim = [] {
        const char * e = std::getenv("GLM_CHUNK_CAPN");
        return e ? (int64_t) std::atoll(e) : (int64_t) 65536 * 4096;
    }();
    int m = want;
    while (m > 256 && im.cap_for((int64_t) pos0 + m - 1) * (int64_t) m > lim) m /= 2;
    return m;
}

void GlmDense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return;
    Impl::Layer & L = im.ly[(size_t) il];
    std::memcpy(im.in_host.data() + ((uint8_t *) L.rbias->data - im.in_base), bias, (size_t) im.NEXP * 4);
    im.in_dirty = true;
    im.bias_dirty = true;
}

bool GlmDense::attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w, const float ** ffn_norm_host) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "attn_router: bad layer"; return false; }
    if (n < 1 || n > im.NT) { im.err = "attn_router: pass size out of range"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int64_t pos_last = (int64_t) pos0 + n - 1;
    if (il == 0) {
        if (pos0 != im.next_pos) {
            im.err = "attn_router: position " + std::to_string(pos0) + " but the recurrent state is at " +
                     std::to_string(im.next_pos) + " (KDA state cannot rewind; reset() to start over)";
            return false;
        }
        if (pos_last >= im.CAPMAX) {
            im.err = pos_last >= im.g.dense_attn_ctx() && !im.allow_long
                ? "context beyond the MLA cache (" + std::to_string(im.CAPMAX) + " rows; raise ctx)"
                : "context beyond the MLA cache (" + std::to_string(im.CAPMAX) + " rows; raise ctx)";
            return false;
        }
    }
    const int64_t cap = L.kda ? 0 : im.cap_for(pos_last);
    const bool idx = !L.kda && im.use_idx(pos_last);
    Impl::Var * var = nullptr;
    for (Impl::Var & v : L.vars) if (v.cap == cap && v.n == n && v.idx == idx) { var = &v; break; }
    if (!var) { build_attn(im, il, cap, n, idx); var = &L.vars.back(); }
    if (!var->allo) var->allo = im.allo_for(n);   // after release_big
    if (!im.alloc(var->allo, var->gf)) { im.err = "gallocr(attn) failed"; return false; }

    // inputs: one upload per pass (positions or capacity changed, or a route bias was set)
    const int64_t mcap = im.cap_for(pos_last);
    if (im.in_dirty || pos0 != im.in_pos || n != im.in_n || mcap != im.in_cap) {
        auto in_ptr = [&](ggml_tensor * t) { return im.in_host.data() + ((uint8_t *) t->data - im.in_base); };
        int32_t * slot = (int32_t *) in_ptr(im.i_slot);
        ggml_fp16_t * mask = (ggml_fp16_t *) in_ptr(im.i_mask);
        float * tpos = (float *) in_ptr(im.i_tpos);
        int32_t * nidx = (int32_t *) in_ptr(im.i_newidx);
        int32_t * ndst = (int32_t *) in_ptr(im.i_newdst);
        float * tail = (float *) in_ptr(im.i_tail);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(NEG_INF);
        const bool dense = !im.use_idx(pos_last);
        const int64_t KP = im.KP, ntop = im.n_top_pools(mcap);
        for (int t = 0; t < n; ++t) {
            const int64_t pos = pos0 + t;
            slot[t] = (int32_t) pos;
            tpos[t] = (float) pos;
            if (dense && mcap <= im.MCAP)
                for (int64_t j = 0; j < mcap; ++j) mask[(size_t) t * mcap + j] = j <= pos ? z : ninf;
            // the incomplete pool (kpool_select_tail): positions pos, pos-1, ... of it; dead slots -> own dump row
            const int64_t n_tail = (pos + 1) % KP;
            for (int64_t k = 0; k < KP - 1; ++k)
                tail[(size_t) t * (KP - 1) + k] = k < n_tail ? (float) (pos - k) : (float) (mcap + KP * ntop + k);
        }
        // pools completed by this pass: members index [tail (KP-1 earlier tokens) | this pass]
        int64_t q = 0;
        for (int64_t pp = pos0 / KP; pp <= (pos_last) / KP; ++pp) {
            const int64_t end = KP * pp + KP - 1;
            if (end < pos0 || end > pos_last) continue;
            for (int64_t k = 0; k < KP; ++k) nidx[q * KP + k] = (int32_t) (KP * pp + k - pos0 + (KP - 1));
            ndst[q++] = (int32_t) pp;
        }
        for (; q < im.NNEW; ++q) {   // padding: any members, written to the dump row
            for (int64_t k = 0; k < KP; ++k) nidx[q * KP + k] = 0;
            ndst[q] = (int32_t) im.PCAP;
        }
        // upload what this pass reads, not the whole span: the span is sized for the largest chunk (the mask alone is
        // MCAP x NT F16 = 16 MB at NT 4096) and a decode step used to send all of it every token (s26: +2.7 ms/token)
        ggml_backend_synchronize(im.backend);   // put() is a plain (cudaStreamPerThread) upload
        auto put = [&](ggml_tensor * t, size_t bytes) {
            const size_t off = (size_t) ((uint8_t *) t->data - im.in_base);
            ggml_backend_tensor_set(im.i_span, im.in_host.data() + off, off, std::min(bytes, ggml_nbytes(t)));
        };
        put(im.i_slot, (size_t) n * 4);
        if (dense && mcap <= im.MCAP) put(im.i_mask, (size_t) n * (size_t) mcap * 2);
        put(im.i_tpos, (size_t) n * 4);
        put(im.i_newidx, ggml_nbytes(im.i_newidx));
        put(im.i_newdst, ggml_nbytes(im.i_newdst));
        put(im.i_tail, (size_t) n * (size_t) (KP - 1) * 4);
        if (im.bias_dirty) {
            for (Impl::Layer & Lb : im.ly) put(Lb.rbias, ggml_nbytes(Lb.rbias));
            im.bias_dirty = false;
        }
        im.in_pos = pos0; im.in_n = n; im.in_cap = mcap; im.in_dirty = false;
    }
    if (ggml_backend_graph_compute_async(im.backend, var->gf) != GGML_STATUS_SUCCESS) { im.err = "attn graph compute failed"; return false; }
    if (il == (int) im.ly.size() - 1) im.next_pos = pos0 + n;

    // one readback: n blocks [fn | ids | wts]
    const size_t ob = (size_t) im.o_blk;
    ggml_backend_tensor_get_async(im.backend, L.o_span, im.host_out.data(), 0, ob * (size_t) n);
    ggml_backend_synchronize(im.backend);
    for (int t = 0; t < n; ++t) {
        const uint8_t * blk = im.host_out.data() + (size_t) t * ob;
        std::memcpy(im.host_fn.data() + (size_t) t * im.D, blk, (size_t) im.D * 4);
        if (L.routed) {
            const int32_t * ids = (const int32_t *) (blk + im.o_ids_off);
            const float * w = (const float *) (blk + im.o_wts_off);
            for (int k = 0; k < im.NUSED; ++k) {
                routed_ids[t * im.NUSED + k] = ids[k];
                routed_w[t * im.NUSED + k] = w[k];
            }
        }
    }
    if (ffn_norm_host) *ffn_norm_host = im.host_fn.data();
    return true;
}

bool GlmDense::finish_layer_n(int il, int n, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n < 1 || n > im.NT) { im.err = "finish_layer: bad args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (L.routed) {
        if (!routed_sum) { im.err = "finish_layer: routed sum missing"; return false; }
        ggml_backend_tensor_set_async(im.backend, im.routed_sum, routed_sum, 0, (size_t) (im.D * n) * 4);
    }
    if (!im.ensure_n(n)) return false;
    if (!L.allo_finish[(size_t) n]) L.allo_finish[(size_t) n] = im.allo_for(n);   // after release_big
    if (!im.alloc(L.allo_finish[(size_t) n], L.gf_finish[(size_t) n])) { im.err = "gallocr(finish) failed"; return false; }
    if (ggml_backend_graph_compute_async(im.backend, L.gf_finish[(size_t) n]) != GGML_STATUS_SUCCESS) { im.err = "finish compute failed"; return false; }
    im.pred_ready = n == 1 && L.fused_next ? il + 1 : -1;
    if (im.gate_taps) ggml_backend_synchronize(im.backend);
    if (im.gate_taps)
        ggml_backend_tensor_get(L.t_lout, L.host_lout.data(), (size_t) (n - 1) * L.t_lout->nb[2], L.host_lout.size() * 4);
    return true;
}

bool GlmDense::logits_rows(int r0, int nr, float * out) {
    Impl & im = *p_;
    if (nr < 1 || nr > Impl::kRowsMax || r0 < 0 || r0 + nr > im.NT) { im.err = "logits_rows: bad rows"; return false; }
    if (im.gf_rows.empty()) {
        im.gf_rows.assign((size_t) Impl::kRowsMax + 1, nullptr);
        im.rows_t.assign((size_t) Impl::kRowsMax + 1, nullptr);
        im.allo_rows = im.new_allo();
    }
    if (!im.gf_rows[(size_t) nr]) {   // the head over hx's first nr rows (build_head's math)
        ggml_context * gc = im.gctx;
        ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
        B b { &im, gc, &im.g };
        ggml_tensor * x = ggml_view_3d(gc, im.hx, im.D, im.HC, nr, im.hx->nb[1], im.hx->nb[2], 0);
        ggml_tensor * acc = ggml_view_2d(gc, x, im.D, nr, x->nb[2], 0);
        for (int64_t s = 1; s < im.HC; ++s) acc = ggml_add(gc, acc, ggml_view_2d(gc, x, im.D, nr, x->nb[2], (size_t) s * x->nb[1]));
        acc = ggml_scale(gc, acc, 1.0f / (float) im.HC);
        ggml_tensor * lg = ggml_mul_mat(gc, b.W("output.weight"), b.rms_w(acc, b.W("output_norm.weight")));
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, lg);
        im.gf_rows[(size_t) nr] = gf;
        im.rows_t[(size_t) nr] = lg;
    }
    const size_t slab = im.x_state->nb[2];
    ggml_backend_synchronize(im.backend);   // the async finish graphs wrote x_state
    im.host_x.resize(slab * (size_t) nr);
    ggml_backend_tensor_get(im.x_state, im.host_x.data(), slab * (size_t) r0, slab * (size_t) nr);
    ggml_backend_tensor_set(im.hx, im.host_x.data(), 0, slab * (size_t) nr);
    if (!im.alloc(im.allo_rows, im.gf_rows[(size_t) nr])) { im.err = "gallocr(rows) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_rows[(size_t) nr]) != GGML_STATUS_SUCCESS) { im.err = "rows head failed"; return false; }
    ggml_backend_tensor_get(im.rows_t[(size_t) nr], out, 0, (size_t) (im.g.vocab * nr) * 4);
    return true;
}

bool GlmDense::logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!im.gf_head[(size_t) n]) { im.err = "logits_n: n > 4 (prompt chunk): use logits_rows"; return false; }
    if (!im.alloc(im.allo_head[(size_t) n], im.gf_head[(size_t) n])) { im.err = "gallocr(head) failed"; return false; }
    if (ggml_backend_graph_compute_async(im.backend, im.gf_head[(size_t) n]) != GGML_STATUS_SUCCESS) { im.err = "head compute failed"; return false; }
    ggml_backend_tensor_get_async(im.backend, im.logits_t[(size_t) n], im.host_logits.data(), 0, (size_t) (im.g.vocab * n) * 4);
    ggml_backend_synchronize(im.backend);
    *out = im.host_logits.data();
    *n_vocab = (int) im.g.vocab;
    return true;
}

const float * GlmDense::tap_l_out(int il) const {
    const Impl & im = *p_;
    if (!im.gate_taps || il < 0 || il >= (int) im.ly.size()) return nullptr;
    return im.ly[(size_t) il].host_lout.data();
}
