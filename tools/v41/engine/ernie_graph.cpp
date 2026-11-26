// tools/v41/engine/ernie_graph.cpp - ERNIE-5.5-Flash (`ernie5_5`) dense half as per-layer ggml graphs.  See the header.
//
// The structure follows tools/v41/engine/ds41_dense.cpp (the DeepSeek-V4.1 engine): one ggml graph per layer per pass,
// on any ggml backend, built once and reused; the state tensors (the 4-stream residual, the attention pre-mix, the
// window ring, the compressor's pending block) live in their own context/buffer and the graphs read/write them; the
// graph allocator is a ggml_gallocr shared by the graphs (Allo).  What is ERNIE's own:
//
//   mHC        the pre-RMS is 1/(sqrt(mean(x^2)) + eps) (not rsqrt(mean + eps)), as the llama.cpp PR's
//              build_hc_pre_ernie; the Sinkhorn comb is ggml_dsv4_hc_comb (softmax over dst, +eps, one column norm,
//              then 19 x (row, column) - the same math the CPU runner was gated on).
//   HCA        q_lora -> per-head unweighted RMS -> YaRN(compress base 160000, factor 16) on the last 64 dims; one MQA
//              K/V row; a 128-token window ring; a per-layer gated 128-pool compressor with a learned APE on the gate
//              (the pending block's wkv/wgate projections are a state tensor, so a pass may start or end mid-block);
//              attention over [window | every completed compressed row] with per-head sinks; inverse RoPE; the VHA
//              head mix I+V U^T (folded at conversion); the grouped low-rank output times sigmoid(o_gate . q_latent).
//   FFN        the leading dense block: SiTU-GLU at feed_forward_length.  MoE layers: two sigmoid routers summed, the
//              top-k chosen on score + exp_probs_b, the weights = plain scores x exp_scale renormalised, the shared
//              expert on the hidden state, and the routed experts on fc1_latent_proj(fn) (the 2048-wide latent rows
//              this engine hands to the tier).  finish: latent_norm -> fc2_latent_proj(+ shared) -> mHC post.
//   head       hc_head_fn mixer (standard RMS), output_norm, output.weight, MultiMax.
//
//   MLA+DSA    (compress ratio 0, G4) q_lora -> q_b [nope 192 | rope 64] per head, plain RoPE (8e6, NORMAL) on the
//              rope slice; kv_a_mqa -> [kv_c 512 (RMS) | k_pe 64 (RoPE)]; K absorption q_abs[h] = k_b[h] q_nope[h];
//              the indexer key (LayerNorm w+b, NEOX RoPE on its first 64); one cache row per position
//              [kv_c | k_pe | idx_k] (704, F16 with kv_f16).  Attention reads the row's first 576 as K and first 512
//              as V for ALL cached positions with a DSA selection mask (exactly the sparse top-k attention: unselected
//              cells get -inf), per-head sinks, scale 1/sqrt(256); then v_b[h], the VHA mix, the sigmoid gate, wo.
//              The mask: causal; when the context exceeds top_k + window (2176) the indexer score
//              sum_h relu(q_h . k_j) w_h with the window forced in picks the top 2176 (ties at the threshold kept).
#include "ernie_graph.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <strings.h>
#include <thread>
#include <tuple>
#include <vector>

// ERNIE_PREC_F32=1 (gate/debug, opt-in): every matmul of the graph asks for F32 compute (cuBLAS F32 on CUDA for F16/F32
// weights), so a gate can compare passes of different widths without the backend's width-dependent F16 accumulation.
// Off (default): plain ggml_mul_mat, the graph is unchanged.
static bool ernie_prec_f32() {
    static const bool on = [] { const char * e = std::getenv("ERNIE_PREC_F32"); return e && *e && *e != '0'; }();
    return on;
}
static ggml_tensor * ernie_mm(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = ggml_mul_mat(ctx, a, b);
    if (ernie_prec_f32()) ggml_mul_mat_set_prec(t, GGML_PREC_F32);
    return t;
}


using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;
constexpr int64_t kHcaRatio = 128;

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// ds41_dense.cpp's Allo: one gallocr serves every cached graph (they run one at a time); a graph re-allocated after
// the buffer grew (the epoch) is re-planned.  Same failure-mode handling (a failed allocation leaves the gallocr
// planned with no buffer, so the next allocation of that graph must re-plan).
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    bool failed = false;
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        if (failed && !ggml_gallocr_reserve(a, g)) { ++epoch; return false; }
        if (!ggml_gallocr_alloc_graph(a, g)) { failed = true; ++epoch; return false; }
        failed = false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ernie_graph] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
};

bool is_routed(const std::string & n) { return n.find("_exps.") != std::string::npos; }

}  // namespace

// ================================================================== Impl

struct ErnieGraph::Impl {
    ErnieGraphConfig cfg;
    ErnieGeometry g;
    std::unique_ptr<GgufModel> model;
    WStore w;
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    std::string err;

    // dims
    int64_t D = 0, HC = 0, DH = 0, NH = 0, OG = 0, OL = 0, OGD = 0, QL = 0, RAW = 0, LAT = 0, NE = 0, NU = 0,
            FE = 0, SE = 0, NC = 0, cmax = 0;
    int64_t nl = 0;

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;    // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;  // [HC, NC]  the FFN mixes' pre (the next attention's pre-mix)
    ggml_tensor * hap = nullptr;        // [D, HC, NC] the residual after the attention's hc_post
    ggml_tensor * post_f = nullptr;     // [HC, NC]
    ggml_tensor * comb_f = nullptr;     // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;      // [D, NC]   the shared expert (or the dense FFN) output
    ggml_tensor * o_fn = nullptr;       // [D, NC]   ffn_norm activation (the dense block)
    ggml_tensor * o_lat = nullptr;      // [LAT, NC] the routed experts' latent input (fc1_latent_proj)
    ggml_tensor * o_ids = nullptr;      // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;      // [NU, NC]
    ggml_tensor * o_pred = nullptr;     // I32 [kPredK, NC] predict(): the next layer's guessed top experts
    ggml_tensor * routed_in = nullptr;  // [LAT, NC] the host's routed sum in the latent space
    ggml_tensor * neg_ramp = nullptr;   // F32 [cmax + 1]  -i
    ggml_tensor * pos_ramp = nullptr;   // F32 [cmax + 1]  +i
    ggml_tensor * j_ramp = nullptr;     // F32 [crows]     position index of each MLA cache row
    int64_t crows = 0;                  // MLA cache rows: ctx_max rounded up to 256 (CUDA flash-attn key multiple)
    bool mla_rot = false;               // the MLA cache rows are Hadamard-rotated (ErnieGraphConfig::mla_hadamard)
    ggml_tensor * had_kl = nullptr, * had_dr = nullptr, * had_id = nullptr;   // normalised Sylvester H [n, n], F32
    struct Layer {
        ggml_tensor * raw = nullptr;      // [DH, RAW]      window K = V, post-RoPE
        ggml_tensor * pend_kv = nullptr;  // [DH, 128]      the pending block's wkv projections
        ggml_tensor * pend_sc = nullptr;  // [DH, 128]      ... and its wgate projections (+ APE)
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1] compressed KV rows (row cmax = spare)
        ggml_tensor * mla = nullptr;      // MLA layers: [704, crows] one cache row per position
    };
    std::vector<Layer> ly;
    struct Snap {
        std::vector<std::vector<uint8_t>> raw, kv, sc;
        int pos = -1;
    };
    Snap snaps[ErnieGraph::kSnapSlots];

    // ---- per-pass inputs (ictx / ibuf)
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_tensor * i_pos = nullptr;    // I32 [NC]
    ggml_tensor * i_widx = nullptr;   // I32 [RAW * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;  // F32 [RAW, NC]
    ggml_tensor * i_wslot = nullptr;  // I32 [min(NC, RAW)]
    ggml_tensor * i_pslot = nullptr;  // I32 [NC]   the pending-block slot each token writes
    ggml_tensor * i_cpos = nullptr;   // I32 [NC]   the RoPE position of each compressed row written
    ggml_tensor * i_cslot = nullptr;  // I32 [NC]   the compressed row each block-completing token writes
    ggml_tensor * i_cnvis = nullptr;  // F32 [NC]   compressed rows visible per query
    ggml_tensor * i_posf = nullptr;   // F32 [NC]   positions as floats (the MLA mask)

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t next_pos = 0;
    struct Seg { int c0, k; bool closes; int64_t block; };
    std::vector<Seg> segs;
    int64_t ncomp = 0;   // completed compressed rows visible to the pass's last query
    int64_t nblk = 0;    // compressed rows the pass writes

    std::vector<float> host_fn, host_wts;
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_pred;

    // ---- graphs
    ggml_context * gctx = nullptr;
    Allo allo_dec, allo_pass;
    struct GKey {
        int il = -1, kind = 0;
        int64_t aux = 0, n = 0;
        bool operator<(const GKey & o) const { return std::tie(il, kind, aux, n) < std::tie(o.il, o.kind, o.aux, o.n); }
    };
    std::map<GKey, ggml_cgraph *> graphs;
    std::map<GKey, ggml_tensor *> head_out;
    std::map<GKey, std::vector<ggml_tensor *>> draft_out;   // dspark_draft: the Markov chain's argmax nodes

    // ---- DSpark (ErnieGraphConfig::dspark_trunk)
    bool draft = false;                   // this graph runs the sidecar's draft stages
    const Impl * trunk = nullptr;         // draft mode: the trunk (geometry, token_embd, output.weight)
    const GgufModel * emb_model = nullptr;   // where token_embd.weight is read (the trunk's file in draft mode)
    int64_t committed = 0;                // draft mode: positions whose window rows dspark_commit wrote
    int64_t DM = 0;                       // draft mode: keys per block query, RAW + NC (padded to 256 off the CPU)
    ggml_tensor * i_dmask = nullptr;      // F32 [DM, NC] the block's mask (the same for every query)
    ggml_tensor * i_mh = nullptr;         // F32 [3 D, RAW] main_hidden rows to commit
    ggml_tensor * i_cwpos = nullptr;      // I32 [RAW] their positions
    ggml_tensor * i_cwslot = nullptr;     // I32 [RAW] their ring slots
    ggml_tensor * i_anchor = nullptr;     // I32 [1]  the Markov chain's first prev token
    struct Spec { int64_t pos0 = -1; int n = 0; std::vector<std::vector<uint8_t>> raw, kv, sc; };
    Spec spec;                            // trunk: spec_save / spec_rollback

    ~Impl() {
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (gctx) ggml_free(gctx);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    bool compute(ggml_cgraph * gf, bool cached);
    void reset_graphs();
    bool gctx_room() const;
};

namespace {

// ---- ggml helpers ---------------------------------------------------------

ggml_tensor * sit_u(ggml_context * gc, ggml_tensor * gate, ggml_tensor * up, float beta, float lb) {
    ggml_tensor * act = ggml_scale(gc, ggml_tanh(gc, ggml_scale(gc, gate, 1.0f / beta)), beta);
    ggml_tensor * cur = ggml_mul(gc, act, ggml_sigmoid(gc, gate));
    if (lb > 0.0f) cur = ggml_mul(gc, cur, ggml_scale(gc, ggml_tanh(gc, ggml_scale(gc, up, 1.0f / lb)), lb));
    return cur;
}

// YaRN at the compress base on the last d_rope dims of each ne[0] row (ds41_dense's rope() with ratio > 0).
ggml_tensor * rope_hca(ggml_context * gc, const ErnieGeometry & g, ggml_tensor * a, ggml_tensor * pos, bool back) {
    const float base = (float) g.compress_rope_base;
    const float fscale = (float) (1.0 / g.rope_factor);
    const float attn = 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale));
    const int nd = (int) g.d_rope;
    ggml_tensor * r = back ? ggml_rope_ext_back(gc, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, (int) g.orig_ctx, base,
                                                fscale, 1.0f, attn, (float) g.beta_fast, (float) g.beta_slow)
                           : ggml_rope_ext(gc, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, (int) g.orig_ctx, base,
                                           fscale, 1.0f, attn, (float) g.beta_fast, (float) g.beta_slow);
    return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
}

struct B {
    ErnieGraph::Impl * im;
    ggml_context * c;
    std::vector<ggml_tensor *> * taps = nullptr;
    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) im->g.rms_eps), wt);
    }
    // a named graph tap: a contiguous copy marked as an output (expanded by the builder at the end)
    void tap(const std::string & name, ggml_tensor * t) const {
        if (!taps) return;
        ggml_tensor * k = t->type == GGML_TYPE_I32 ? ggml_cast(c, t, GGML_TYPE_F32) : ggml_cont(c, t);
        ggml_set_output(k);
        ggml_set_name(k, name.c_str());
        taps->push_back(k);
    }
};

// mHC mixes (model.py Block.hc_mixes + the PR's build_hc_pre_ernie): x [D, HC, n] -> pre [HC, n], post [HC, n],
// comb [HC(dst), HC(src), n].
void hc_mixes(ErnieGraph::Impl & im, ggml_context * gc, ggml_tensor * x, ggml_tensor * fn_w, ggml_tensor * sc,
              ggml_tensor * bs, ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb, B & b,
              const std::string & pre_name, const std::string & post_name) {
    const int64_t HC = im.HC, nt = x->ne[2], hc_dim = HC * im.D;
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(gc, x);
    ggml_tensor * flat = ggml_reshape_2d(gc, xc, hc_dim, nt);
    // ERNIE: mixes = (hc_fn . flat) / (sqrt(mean(flat^2)) + eps)   -- not rsqrt(mean + eps)
    ggml_tensor * ss = ggml_sum_rows(gc, ggml_sqr(gc, flat));                      // [1, n]
    ggml_tensor * rms = ggml_sqrt(gc, ggml_scale(gc, ss, 1.0f / (float) hc_dim));
    rms = ggml_scale_bias(gc, rms, 1.0f, (float) im.g.hc_eps);                     // + eps
    ggml_tensor * mixes = ernie_mm(gc, fn_w, flat);                            // [(2+HC)HC, n]
    ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
    mixes = ggml_div(gc, mixes, rms);
    ggml_tensor * m_pre = ggml_view_2d(gc, mixes, HC, nt, mixes->nb[1], 0);
    *pre = ggml_add(gc, ggml_mul(gc, m_pre, b.v1(sc, 1, 0)), b.v1(bs, HC, 0));
    *pre = ggml_scale_bias(gc, ggml_sigmoid(gc, *pre), 1.0f, (float) im.g.hc_eps);
    ggml_tensor * m_post = ggml_view_2d(gc, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
    *post = ggml_add(gc, ggml_mul(gc, m_post, b.v1(sc, 1, sizeof(float))),
                     b.v1(bs, HC, (size_t) HC * sizeof(float)));
    *post = ggml_scale(gc, ggml_sigmoid(gc, *post), 2.0f);
    *comb = ggml_dsv4_hc_comb(gc, mixes, sc, bs, (float) im.g.hc_eps, (int32_t) im.g.hc_iters);
    b.tap(pre_name, *pre);
    b.tap(post_name, *post);
}

// the ERNIE head mixer: mixes = hc_head_fn . rms_norm(flat(x)); pre = sigmoid(mixes*scale + base) + eps;
// out = sum_h pre_h x_h   (llama.cpp deepseek4.cpp build_hc_head, standard RMS)
ggml_tensor * hc_head(ErnieGraph::Impl & im, ggml_context * gc, ggml_tensor * x) {
    const int64_t HC = im.HC, nt = x->ne[2], hc_dim = HC * im.D;
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(gc, x);
    ggml_tensor * flat = ggml_reshape_2d(gc, xc, hc_dim, nt);
    ggml_tensor * fnorm = ggml_rms_norm(gc, flat, (float) im.g.rms_eps);
    ggml_tensor * mixes = ernie_mm(gc, im.W("output_hc_fn.weight"), fnorm);      // [HC, n]
    ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
    mixes = ggml_add(gc, ggml_mul(gc, mixes, im.W("output_hc_scale.weight")), im.W("output_hc_base.weight"));
    mixes = ggml_scale_bias(gc, ggml_sigmoid(gc, mixes), 1.0f, (float) im.g.hc_eps);
    return ggml_dsv4_hc_pre(gc, x, mixes);
}

ggml_tensor * multimax(ErnieGraph::Impl & im, ggml_context * gc, ggml_tensor * logits) {
    const double * r = im.g.multimax;
    const double * t = r + 4;
    bool on = false;
    for (int i = 0; i < 8; ++i) on = on || r[i] != 0.0;
    if (!on) return logits;
    ggml_tensor * below0 = ggml_relu(gc, ggml_scale_bias(gc, logits, -1.0f, (float) r[0]));
    ggml_tensor * above1 = ggml_relu(gc, ggml_scale_bias(gc, logits, 1.0f, -(float) r[1]));
    ggml_tensor * below2 = ggml_relu(gc, ggml_scale_bias(gc, logits, -1.0f, (float) r[2]));
    ggml_tensor * above3 = ggml_relu(gc, ggml_scale_bias(gc, logits, 1.0f, -(float) r[3]));
    ggml_tensor * out = logits;
    out = ggml_add(gc, out, ggml_scale(gc, below0, (float) t[0]));
    out = ggml_add(gc, out, ggml_scale(gc, above1, (float) t[1]));
    out = ggml_add(gc, out, ggml_scale(gc, ggml_sqr(gc, below2), (float) t[2]));
    out = ggml_add(gc, out, ggml_scale(gc, ggml_sqr(gc, above3), (float) t[3]));
    return out;
}

}  // namespace

// ================================================================== weights

// Draft mode: a sidecar tensor's name in the trunk's convention, so the layer graphs read the stages unchanged:
// dspark.blk.N.* -> blk.N.*, the head mixer / norm -> output_hc_* / output_norm; the rest (main_proj, main_norm,
// context_wkv_proj, markov_head.*, confidence_head.*) keep their dspark.* names.
static std::string draft_name(const std::string & n) {
    if (n.rfind("dspark.blk.", 0) == 0) return n.substr(7);
    if (n == "dspark.hc_head_fn.weight") return "output_hc_fn.weight";
    if (n == "dspark.hc_head_base.weight") return "output_hc_base.weight";
    if (n == "dspark.hc_head_scale.weight") return "output_hc_scale.weight";
    if (n == "dspark.norm.weight") return "output_norm.weight";
    return n;
}

static bool load_weights(ErnieGraph::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params wp = { 64ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(wp);
    if (!w.ctx) { err = "cannot create the weight context"; return false; }
    auto skip = [](const std::string & n) {
        return is_routed(n) || n == "token_embd.weight";   // the tier reads the experts; the embedding is host-side
    };
    auto key = [&](const std::string & n) { return im.draft ? draft_name(n) : n; };
    // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; OFF by default, the DS4.1 engine's switch): every
    // Q8_0 dense matrix the GPU graphs read (attention, shared expert, latent projections; output.weight unless
    // STRATA_DENSE_REQUANT_HEAD=0) is re-quantized in memory at load: rows dequantized, quantized to that type.  The
    // file is not touched.  The VRAM it frees goes to the expert cache when --slots is 0 (auto).
    ggml_type rq = GGML_TYPE_COUNT;
    if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e && !im.cpu)
        for (int t = 0; t < GGML_TYPE_COUNT; ++t)
            if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
    const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
    auto requant = [&](const TensorInfo & ti) {
        return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
               (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight") &&
               // tables read with ggml_get_rows keep their type (CUDA get_rows has no K-quant rows)
               ti.name.find("attn_compressor_ape") == std::string::npos && ti.name.find("markov_head") == std::string::npos;
    };
    auto make = [&](const TensorInfo & ti) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        const int nd = (int) std::min<size_t>(ti.shape.size(), GGML_MAX_DIMS);
        for (int d = 0; d < nd; ++d) ne[d] = (int64_t) ti.shape[(size_t) d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, requant(ti) ? rq : (ggml_type) ti.type, nd, ne);
        ggml_set_name(t, ti.name.c_str());
        return t;
    };
    if (im.cpu) {   // bind the mmapped tensors in place (no copy): the same bytes the host expert loop reads
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[key(ti.name)] = t;
            }
        }
    } else {
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[key(ti.name)] = make(ti);
            }
        // --- the optional directional ablation (ErnieGraphConfig::abl_file)
        std::vector<float> abl;
        int64_t abl_nl = 0;
        if (!im.cfg.abl_file.empty()) {
            std::FILE * f = std::fopen(im.cfg.abl_file.c_str(), "rb");
            int32_t hd[2] = { 0, 0 };
            if (!f || std::fread(hd, 4, 2, f) != 2 || hd[1] != (int32_t) im.D || hd[0] < 1) {
                if (f) std::fclose(f);
                err = "abl_file " + im.cfg.abl_file + ": bad header (want n_layer, n_embd " + std::to_string(im.D) + ")";
                return false;
            }
            abl_nl = hd[0];
            abl.resize((size_t) abl_nl * im.D);
            const size_t got = std::fread(abl.data(), 4, abl.size(), f);
            std::fclose(f);
            if (got != abl.size()) { err = "abl_file " + im.cfg.abl_file + ": short"; return false; }
        }
        auto abl_layer = [&](const std::string & n) -> int {   // the layer whose direction edits n, -1 = none
            if (abl.empty() || n.rfind("blk.", 0) != 0) return -1;
            const size_t dot = n.find('.', 4);
            const std::string role = n.substr(dot + 1);
            if (role != "attn_output_b.weight" && role != "attn_output.weight" && role != "ffn_routed_up.weight" && role != "ffn_down_shexp.weight" &&
                role != "ffn_down.weight")
                return -1;
            const int l = std::atoi(n.c_str() + 4);
            if (l >= abl_nl) return -1;
            const float * r = abl.data() + (size_t) l * im.D;
            for (int64_t i = 0; i < im.D; ++i) if (r[i] != 0.0f) return l;
            return -1;
        };
        int n_edit = 0;
        size_t rq_n = 0, rq_from = 0, rq_to = 0;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[key(ti.name)];
                const int al = im.draft ? -1 : abl_layer(ti.name);
                if (al < 0 && (int) t->type == (int) ti.type) {
                    ggml_backend_tensor_set(t, model.shard(s).tensor_data(ti), 0, ggml_nbytes(t));
                    continue;
                }
                if (al < 0) {   // STRATA_DENSE_REQUANT: the file's rows -> t->type, threads over rows
                    const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
                    const ggml_type st = (ggml_type) ti.type;
                    const size_t src_row = ggml_row_size(st, n0), dst_row = ggml_row_size(t->type, n0);
                    const uint8_t * src = model.shard(s).tensor_data(ti);
                    std::vector<uint8_t> dst((size_t) nr * dst_row);
                    const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
                    std::vector<std::thread> th;
                    for (int k = 0; k < nth; ++k)
                        th.emplace_back([&, k] {
                            std::vector<float> f((size_t) n0);
                            for (int64_t r = k; r < nr; r += nth) {
                                ggml_get_type_traits(st)->to_float(src + (size_t) r * src_row, f.data(), n0);
                                ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                            }
                        });
                    for (auto & x : th) x.join();
                    ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
                    ++rq_n;
                    rq_from += (size_t) nr * src_row;
                    rq_to += dst.size();
                    continue;
                }
                // W [in = ne0, out = ne1]: v = r^T W (length in), W[j][:] -= scale * r_j * v
                const int64_t in = t->ne[0], out = t->ne[1];
                if (out != im.D || ggml_nrows(t) != out) { err = "abl: " + ti.name + " does not write n_embd rows"; return false; }
                const ggml_type ty = t->type;   // the stored type (a requant target when STRATA_DENSE_REQUANT is set)
                const ggml_type sty = (ggml_type) ti.type;   // the file's type, what src holds
                const ggml_type_traits * tr = ggml_get_type_traits(sty);
                std::vector<float> wf((size_t) in * out);
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (sty == GGML_TYPE_F32) std::memcpy(wf.data(), src, wf.size() * 4);
                else if (tr && tr->to_float) tr->to_float(src, wf.data(), (int64_t) wf.size());
                else { err = "abl: no row reader for " + ti.name; return false; }
                const float * r = abl.data() + (size_t) al * im.D;
                std::vector<double> v((size_t) in, 0.0);
                for (int64_t j = 0; j < out; ++j) {
                    const double rj = r[j];
                    if (rj == 0.0) continue;
                    const float * row = wf.data() + (size_t) j * in;
                    for (int64_t i = 0; i < in; ++i) v[(size_t) i] += rj * row[i];
                }
                const double sc = im.cfg.abl_scale;
                for (int64_t j = 0; j < out; ++j) {
                    const double k = sc * r[j];
                    if (k == 0.0) continue;
                    float * row = wf.data() + (size_t) j * in;
                    for (int64_t i = 0; i < in; ++i) row[i] = (float) (row[i] - k * v[(size_t) i]);
                }
                std::vector<uint8_t> q(ggml_nbytes(t));
                if (ty == GGML_TYPE_F32) std::memcpy(q.data(), wf.data(), q.size());
                else ggml_quantize_chunk(ty, wf.data(), q.data(), 0, out, in, nullptr);
                ggml_backend_tensor_set(t, q.data(), 0, q.size());
                ++n_edit;
            }
        if (!abl.empty()) std::fprintf(stderr, "ernie_graph: ablation %s, scale %.3f: %d writer matrices edited\n",
                                       im.cfg.abl_file.c_str(), im.cfg.abl_scale, n_edit);
        if (rq != GGML_TYPE_COUNT)
            std::fprintf(stderr, "ernie_graph%s: STRATA_DENSE_REQUANT=%s: %zu Q8_0 matrices re-quantized, %.2f -> %.2f GB (+%d edited)\n",
                         im.draft ? " (draft)" : "", ggml_type_name(rq), rq_n, rq_from / 1e9, rq_to / 1e9, n_edit);
    }
    if (im.draft) {
        // the trunk's LM head (no copy: same backend), and each stage's wkv = its rows of context_wkv_proj (the three
        // stages' wkv stacked in stage order, one GEMM in the reference's CommitKvProj)
        w.t["output.weight"] = im.trunk->w.get("output.weight");
        ggml_tensor * cw = w.get("dspark.context_wkv_proj.weight");
        if (cw->ne[0] != im.D || cw->ne[1] != im.DH * im.nl) {
            err = "dspark.context_wkv_proj is not [n_embd, n_stages * d_head]";
            return false;
        }
        for (int64_t s = 0; s < im.nl; ++s) {
            ggml_tensor * v = ggml_view_2d(w.ctx, cw, cw->ne[0], im.DH, cw->nb[1], (size_t) (s * im.DH) * cw->nb[1]);
            if (ggml_backend_view_init(v) != GGML_STATUS_SUCCESS) { err = "cannot view context_wkv_proj"; return false; }
            const std::string n = "blk." + std::to_string(s) + ".attn_kv.weight";
            ggml_set_name(v, n.c_str());
            w.t[n] = v;
        }
    }
    return true;
}

// ================================================================== init

static size_t gctx_bytes() {
    return ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400;
}

bool ErnieGraph::Impl::gctx_room() const {
    if (!gctx) return false;
    const size_t need = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false) * 2;
    return ggml_used_mem(gctx) + need <= ggml_get_mem_size(gctx);
}

void ErnieGraph::Impl::reset_graphs() {
    graphs.clear();
    head_out.clear();
    draft_out.clear();
    for (Allo * al : { &allo_dec, &allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        al->failed = false;
        ++al->epoch;
    }
    if (gctx) ggml_free(gctx);
    ggml_init_params gp = { gctx_bytes(), nullptr, true };
    gctx = ggml_init(gp);
}

static bool alloc_state(ErnieGraph::Impl & im, std::string & err) {
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (im.nl * 8 + 64), nullptr, true };
    im.sctx = ggml_init(sp);
    if (!im.sctx) { err = "cannot create the state context"; return false; }
    auto T = [&](ggml_type t, std::initializer_list<int64_t> d) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        int i = 0;
        for (int64_t x : d) ne[i++] = x;
        return ggml_new_tensor(im.sctx, t, i, ne);
    };
    const int64_t NC = im.NC;
    im.x_state = T(GGML_TYPE_F32, { im.D, im.HC, NC });
    im.pre_state = T(GGML_TYPE_F32, { im.HC, NC });
    im.hap = T(GGML_TYPE_F32, { im.D, im.HC, NC });
    im.post_f = T(GGML_TYPE_F32, { im.HC, NC });
    im.comb_f = T(GGML_TYPE_F32, { im.HC, im.HC, NC });
    im.shexp = T(GGML_TYPE_F32, { im.D, NC });
    im.o_fn = T(GGML_TYPE_F32, { im.D, NC });
    im.o_lat = T(GGML_TYPE_F32, { im.LAT, NC });
    im.o_ids = T(GGML_TYPE_I32, { im.NU, NC });
    im.o_wts = T(GGML_TYPE_F32, { im.NU, NC });
    im.o_pred = T(GGML_TYPE_I32, { ErnieGraph::kPredK, NC });
    im.routed_in = T(GGML_TYPE_F32, { im.LAT, NC });
    im.neg_ramp = T(GGML_TYPE_F32, { im.cmax + 1 });
    im.pos_ramp = T(GGML_TYPE_F32, { im.cmax + 1 });
    const ggml_type kv_type = im.cfg.kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    im.ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        if (!im.g.is_hca(l)) {
            const ggml_type mt = im.cfg.mla_kv_type == GGML_TYPE_F16 ? kv_type : im.cfg.mla_kv_type;
            L.mla = T(mt, { im.g.kv_lora + im.g.d_rope + im.g.idx_key_dim, im.crows });
            continue;
        }
        L.raw = T(kv_type, { im.DH, im.RAW });
        L.pend_kv = T(GGML_TYPE_F32, { im.DH, kHcaRatio });
        L.pend_sc = T(GGML_TYPE_F32, { im.DH, kHcaRatio });
        L.comp = T(kv_type, { im.DH, im.cmax + 1 });
    }
    if (im.g.has_mla) im.j_ramp = T(GGML_TYPE_F32, { im.crows });
    im.mla_rot = im.g.has_mla && (im.cfg.mla_hadamard > 0 || (im.cfg.mla_hadamard < 0 && im.cfg.mla_kv_type == GGML_TYPE_Q4_0));
    if (im.mla_rot) {
        im.had_kl = T(GGML_TYPE_F32, { im.g.kv_lora, im.g.kv_lora });
        im.had_dr = T(GGML_TYPE_F32, { im.g.d_rope, im.g.d_rope });
        im.had_id = T(GGML_TYPE_F32, { im.g.idx_key_dim, im.g.idx_key_dim });
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "cannot allocate the state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    if (im.mla_rot) {
        // Sylvester H_n / sqrt(n): symmetric and orthogonal, its own inverse
        for (ggml_tensor * h : { im.had_kl, im.had_dr, im.had_id }) {
            const int64_t nn = h->ne[0];
            if (nn & (nn - 1)) { err = "mla_hadamard: segment width " + std::to_string(nn) + " is not a power of two"; return false; }
            std::vector<float> m((size_t) (nn * nn));
            const float sc = 1.0f / std::sqrt((float) nn);
            for (int64_t i = 0; i < nn; ++i)
                for (int64_t j = 0; j < nn; ++j) m[(size_t) (i * nn + j)] = (__builtin_popcountll((unsigned long long) (i & j)) & 1) ? -sc : sc;
            ggml_backend_tensor_set(h, m.data(), 0, m.size() * sizeof(float));
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 80, nullptr, true };
    im.ictx = ggml_init(ip);
    auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) {
        return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2);
    };
    im.i_pos = I(GGML_TYPE_I32, NC);
    im.i_widx = I(GGML_TYPE_I32, im.RAW * NC);
    im.i_wmask = I(GGML_TYPE_F32, im.RAW, NC);
    im.i_wslot = I(GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    im.i_pslot = I(GGML_TYPE_I32, NC);
    im.i_cpos = I(GGML_TYPE_I32, NC);
    im.i_cslot = I(GGML_TYPE_I32, NC);
    im.i_cnvis = I(GGML_TYPE_F32, NC);
    im.i_posf = I(GGML_TYPE_F32, NC);
    if (im.draft) {
        im.DM = im.RAW + NC;
        if (!im.cpu) im.DM = (im.DM + 255) / 256 * 256;   // ggml-cuda's flash-attn for d 512: keys a multiple of 256
        im.i_dmask = I(GGML_TYPE_F32, im.DM, NC);
        im.i_mh = I(GGML_TYPE_F32, 3 * im.D, im.RAW);
        im.i_cwpos = I(GGML_TYPE_I32, im.RAW);
        im.i_cwslot = I(GGML_TYPE_I32, im.RAW);
        im.i_anchor = I(GGML_TYPE_I32, 1);
    }
    im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
    if (!im.ibuf) { err = "cannot allocate the pass inputs"; return false; }
    ggml_backend_buffer_clear(im.ibuf, 0);
    {
        std::vector<float> v((size_t) (im.cmax + 1));
        for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
        ggml_backend_tensor_set(im.neg_ramp, v.data(), 0, v.size() * sizeof(float));
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
        ggml_backend_tensor_set(im.pos_ramp, v.data(), 0, v.size() * sizeof(float));
        if (im.j_ramp) {
            std::vector<float> j((size_t) im.crows);
            for (size_t i = 0; i < j.size(); ++i) j[i] = (float) i;
            ggml_backend_tensor_set(im.j_ramp, j.data(), 0, j.size() * sizeof(float));
        }
    }
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!im.allo_dec.a || !im.allo_pass.a) { err = "cannot create the graph allocators"; return false; }
    return true;
}

bool ErnieGraph::init(const std::string & gguf, const ErnieGraphConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model.reset(new GgufModel(GgufModel::open(gguf)));
    } catch (const std::exception & e) {
        err = std::string("open ") + gguf + ": " + e.what();
        return false;
    }
    if (cfg.dspark_trunk) {
        // draft mode: the trunk's geometry with the sidecar's stage count; every stage is an HCA MoE layer
        const Impl & T = *cfg.dspark_trunk->p_;
        if (T.draft || T.nl < 1) { err = "dspark_trunk is not an initialised trunk"; return false; }
        if (cfg.backend && cfg.backend != T.backend) { err = "the draft must run on the trunk's backend"; return false; }
        int64_t ns = 0;
        while (im.model->find("dspark.blk." + std::to_string(ns) + ".attn_norm.weight")) ++ns;
        if (ns < 1 || !im.model->find("dspark.main_proj.weight") || !im.model->find("dspark.markov_head.embed.weight")) {
            err = gguf + ": not an ERNIE DSpark sidecar (no dspark.blk.0 / main_proj / markov_head)";
            return false;
        }
        im.draft = true;
        im.trunk = &T;
        im.g = T.g;
        im.g.n_layer = ns;
        im.g.ratio.assign((size_t) ns, kHcaRatio);
        im.g.dense_lead = 0;
        im.g.has_mla = false;
        im.emb_model = T.model.get();
        im.cfg.backend = T.backend;
    } else {
        if (!im.g.load(*im.model, err)) return false;
        im.emb_model = im.model.get();
    }
    const ErnieGeometry & G = im.g;
    im.nl = G.n_layer;
    for (int64_t l = 0; l < im.nl; ++l) {
        if (G.is_hca(l)) continue;
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * n : { "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
                                "attn_output.weight", "attn_q_b.weight", "attn_gate.weight", "attn_sinks.weight",
                                "attn_vha.weight", "indexer.attn_k.weight", "indexer.k_norm.weight",
                                "indexer.k_norm.bias", "indexer.attn_q_b.weight", "indexer.proj.weight" })
            if (!im.model->find(b + n)) { err = "MLA layer " + std::to_string(l) + ": missing " + b + n; return false; }
        if (G.mla_nope() % 32 || G.kv_lora % 32 || G.idx_key_dim % 32) {
            err = "an MLA width is not a Q8_0 block multiple";
            return false;
        }
    }
    if (cfg.max_pass < 1 || cfg.ctx_max < kHcaRatio || cfg.ctx_max % kHcaRatio) {
        err = "ctx_max must be a multiple of 128; max_pass must be >= 1";
        return false;
    }
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.NH = G.n_head; im.OG = G.o_groups; im.OL = G.o_lora;
    im.OGD = G.o_group_dim(); im.QL = G.q_lora; im.RAW = G.n_swa; im.LAT = G.n_latent; im.NE = G.n_expert;
    im.NU = G.n_used; im.FE = G.n_ff_exp; im.SE = G.n_ff_exp * G.n_shared; im.NC = cfg.max_pass;
    im.cmax = cfg.ctx_max / kHcaRatio;
    im.crows = (cfg.ctx_max + 255) / 256 * 256;
    if (im.RAW != kHcaRatio) { err = "attention.sliding_window must be 128 for the HCA path"; return false; }
    if (im.D % 32 || im.QL % 32 || im.DH % 32 || im.LAT % 32 || im.FE % 32) {
        err = "a dense-path width is not a Q8_0 block multiple";
        return false;
    }
    if (im.draft) {
        for (const char * n : { "dspark.hc_head_fn.weight", "dspark.hc_head_base.weight", "dspark.hc_head_scale.weight",
                                "dspark.norm.weight", "dspark.main_norm.weight", "dspark.context_wkv_proj.weight",
                                "dspark.markov_head.head.weight" })
            if (!im.model->find(n)) { err = std::string("missing tensor ") + n; return false; }
    } else {
        for (const char * n : { "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight",
                                "output_norm.weight", "output.weight" }) {
            if (!im.model->find(n)) { err = std::string("missing tensor ") + n; return false; }
        }
    }
    if (im.cfg.backend) {
        im.backend = im.cfg.backend;
    } else {
        im.backend = ggml_backend_cpu_init();
        if (!im.backend) { err = "cannot create the CPU backend"; return false; }
        if (cfg.n_threads > 0) ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (!load_weights(im, err)) return false;
    if (!alloc_state(im, err)) return false;
    ggml_init_params gp = { gctx_bytes(), nullptr, true };
    im.gctx = ggml_init(gp);
    if (!im.gctx) { err = "cannot create the graph context"; return false; }
    im.host_fn.resize((size_t) (std::max(im.D, im.LAT) * im.NC));
    im.host_ids.resize((size_t) (im.NU * im.NC));
    im.host_pred.resize((size_t) (ErnieGraph::kPredK * im.NC));
    im.host_wts.resize((size_t) (im.NU * im.NC));
    return true;
}

const ErnieGeometry & ErnieGraph::geom() const { return p_->g; }
int64_t ErnieGraph::n_layer() const { return p_->nl; }
ggml_backend_t ErnieGraph::backend() const { return p_->backend; }
const std::string & ErnieGraph::last_error() const { return p_->err; }
int ErnieGraph::pass_size() const { return p_->n; }
int ErnieGraph::pass_pos0() const { return p_->pos0; }
int ErnieGraph::snapshot_pos(int slot) const {
    return slot >= 0 && slot < kSnapSlots ? p_->snaps[slot].pos : -1;
}

// ================================================================== begin

bool ErnieGraph::begin(const int * tids, int np, int p0) {
    Impl & im = *p_;
    if (np < 1 || np > im.NC) {
        im.err = "pass of " + std::to_string(np) + " tokens (max_pass " + std::to_string(im.NC) + ")";
        return false;
    }
    if (im.draft ? p0 != im.committed : p0 != im.next_pos) {
        im.err = "begin at position " + std::to_string(p0) + " but the state is at " +
                 std::to_string(im.draft ? im.committed : im.next_pos);
        return false;
    }
    if ((int64_t) p0 + np > im.cfg.ctx_max) { im.err = "pass past ctx_max"; return false; }
    im.pos0 = p0;
    im.n = np;

    // ---- the residual: the token embedding replicated over the HC streams; pre = one-hot (make_identity_pre_mix)
    size_t es = 0;
    const TensorInfo * emb = im.emb_model->find("token_embd.weight", &es);
    if (!emb) { im.err = "token_embd.weight is missing"; return false; }
    const ggml_type ety = (ggml_type) emb->type;
    const ggml_type_traits * etr = ggml_get_type_traits(ety);
    if (!etr || (ety != GGML_TYPE_F32 && !etr->to_float)) { im.err = "token_embd type has no row reader"; return false; }
    if ((size_t) im.D % ggml_blck_size(ety)) { im.err = "token_embd row is not a whole number of blocks"; return false; }
    const size_t emb_row = (size_t) im.D / ggml_blck_size(ety) * ggml_type_size(ety);
    const uint8_t * emb_base = im.emb_model->shard(es).tensor_data(*emb);
    std::vector<float> xh((size_t) np * im.HC * im.D);
    std::vector<float> preh((size_t) np * im.HC, 0.0f);
    for (int t = 0; t < np; ++t) {
        const int tok = tids[t];
        if (tok < 0 || tok >= im.g.vocab) {
            im.err = "token id " + std::to_string(tok) + " outside the vocabulary";
            return false;
        }
        float * row = xh.data() + (size_t) t * im.HC * im.D;
        if (ety == GGML_TYPE_F32) std::memcpy(row, emb_base + (size_t) tok * emb_row, (size_t) im.D * 4);
        else etr->to_float(emb_base + (size_t) tok * emb_row, row, im.D);
        for (int64_t s = 1; s < im.HC; ++s) std::memcpy(row + (size_t) s * im.D, row, (size_t) im.D * 4);
        preh[(size_t) t * im.HC] = 1.0f;
    }
    ggml_backend_tensor_set(im.x_state, xh.data(), 0, xh.size() * 4);
    ggml_backend_tensor_set(im.pre_state, preh.data(), 0, preh.size() * 4);

    if (im.draft) {
        // a draft block: positions p0..p0+np-1; every query sees the committed ring rows + all np block rows
        std::vector<int32_t> pos_i((size_t) np);
        for (int t = 0; t < np; ++t) pos_i[(size_t) t] = (int32_t) (p0 + t);
        std::vector<float> dm((size_t) im.DM * np, NEG_INF);
        for (int t = 0; t < np; ++t) {
            float * col = dm.data() + (size_t) im.DM * t;
            static const bool nowin = [] { const char * e = std::getenv("ERNIE_DSPARK_NOWIN"); return e && *e == '1'; }();
            for (int64_t j = 0; j < im.RAW; ++j) if (!nowin && (im.committed >= im.RAW || j < im.committed)) col[j] = 0.0f;
            for (int j = 0; j < np; ++j) col[im.RAW + j] = 0.0f;
        }
        ggml_backend_tensor_set(im.i_pos, pos_i.data(), 0, pos_i.size() * 4);
        ggml_backend_tensor_set(im.i_dmask, dm.data(), 0, dm.size() * 4);
        im.segs.clear();
        im.nblk = 0;
        im.ncomp = 0;
        return true;
    }

    // ---- positions / window gather / compressor bookkeeping (host, as ds41_dense's upload_inputs)
    std::vector<int32_t> pos_i((size_t) np), pslot((size_t) np), widx((size_t) im.RAW * np);
    std::vector<float> wmask((size_t) im.RAW * np), cnvis((size_t) np), posf((size_t) np);
    for (int t = 0; t < np; ++t) {
        const int64_t p = p0 + t;
        pos_i[(size_t) t] = (int32_t) p;
        posf[(size_t) t] = (float) p;
        pslot[(size_t) t] = (int32_t) (p % im.RAW);
        cnvis[(size_t) t] = (float) ((p + 1) / kHcaRatio);
        for (int64_t j = 0; j < im.RAW; ++j) {
            const int64_t qpos = p - im.RAW + 1 + j;
            const bool vis = qpos >= 0 && qpos <= p;
            int64_t idx = 0;
            if (vis) idx = qpos < p0 ? qpos % im.RAW : im.RAW + (qpos - p0);
            widx[(size_t) j + (size_t) im.RAW * t] = (int32_t) idx;
            wmask[(size_t) j + (size_t) im.RAW * t] = vis ? 0.0f : NEG_INF;
        }
    }
    const int64_t nw = std::min<int64_t>(np, im.RAW);
    std::vector<int32_t> wslot((size_t) nw);
    for (int64_t r = 0; r < nw; ++r) wslot[(size_t) r] = (int32_t) ((p0 + np - nw + r) % im.RAW);

    // ---- the pass's compressor segments (split at 128-token block boundaries)
    im.segs.clear();
    im.nblk = 0;
    {
        int c = 0;
        int64_t slot = (int64_t) p0 % kHcaRatio;
        while (c < np) {
            const int take = (int) std::min<int64_t>(np - c, kHcaRatio - slot);
            const bool closes = slot + take == kHcaRatio;
            im.segs.push_back(ErnieGraph::Impl::Seg { c, take, closes, closes ? (p0 + c + take) / kHcaRatio - 1 : -1 });
            if (closes) ++im.nblk;
            c += take;
            slot = closes ? 0 : slot + take;
        }
    }
    std::vector<int32_t> cpos((size_t) im.nblk), cslot((size_t) im.nblk);
    {
        int64_t k = 0;
        for (const auto & s : im.segs)
            if (s.closes) {
                cslot[(size_t) k] = (int32_t) s.block;
                cpos[(size_t) k] = (int32_t) (s.block * kHcaRatio);
                ++k;
            }
    }
    im.ncomp = (int64_t) (p0 + np) / kHcaRatio;

    ggml_backend_tensor_set(im.i_pos, pos_i.data(), 0, pos_i.size() * 4);
    ggml_backend_tensor_set(im.i_pslot, pslot.data(), 0, pslot.size() * 4);
    ggml_backend_tensor_set(im.i_widx, widx.data(), 0, widx.size() * 4);
    ggml_backend_tensor_set(im.i_wmask, wmask.data(), 0, wmask.size() * 4);
    ggml_backend_tensor_set(im.i_wslot, wslot.data(), 0, wslot.size() * 4);
    ggml_backend_tensor_set(im.i_cnvis, cnvis.data(), 0, cnvis.size() * 4);
    ggml_backend_tensor_set(im.i_posf, posf.data(), 0, posf.size() * 4);
    if (im.nblk > 0) {
        ggml_backend_tensor_set(im.i_cpos, cpos.data(), 0, cpos.size() * 4);
        ggml_backend_tensor_set(im.i_cslot, cslot.data(), 0, cslot.size() * 4);
    }
    im.next_pos = (int64_t) p0 + np;
    return true;
}

// ================================================================== the layer graph

// MLA + DSA attention (compress ratio 0) for one pass: from the attention input xn [D, n] to the attention output
// [D, n] (before the mHC post).  Writes this pass's cache rows first, then attends over the cache's first `mla_cells`
// rows (a multiple of 256 covering every position up to the pass's last) under the DSA mask.  See the file header.
static int64_t mla_cells(const ErnieGraph::Impl & im) { return ((int64_t) im.pos0 + im.n + 255) / 256 * 256; }
static bool mla_topk_on(const ErnieGraph::Impl & im) {
    return (int64_t) im.pos0 + im.n > im.g.idx_top_k + im.g.n_swa;
}

static ggml_tensor * build_mla(ErnieGraph::Impl & im, ggml_context * gc, ggml_cgraph * gf, B & b, int il,
                               ggml_tensor * xn, ggml_tensor * pos) {
    const ErnieGeometry & G = im.g;
    const int64_t n = im.n, NH = im.NH, QKH = G.d_head_mla_k, NOPE = G.mla_nope(), DR = G.d_rope, DV = G.d_head_mla_v;
    const int64_t KL = G.kv_lora, KROW = KL + DR, ID = G.idx_key_dim, IH = G.idx_n_head;
    const int64_t ROW = KROW + ID, OJ = NH * DV;
    const int64_t M = mla_cells(im), NTOP = G.idx_top_k + G.n_swa;
    const bool topk = mla_topk_on(im);
    const std::string sl = std::to_string(il);
    const float base = (float) G.rope_base;
    auto rope = [&](ggml_tensor * a, int mode, int off) {   // plain RoPE (no YaRN) on dims [off, off + d_rope)
        ggml_tensor * r = ggml_rope_ext(gc, a, pos, nullptr, (int) DR, mode, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return off ? ggml_rope_set_offset(r, off) : r;
    };

    // ---- q latent, output gate, q = [nope | rope] per head
    ggml_tensor * qc = ernie_mm(gc, im.BL(il, "attn_q_a.weight"), xn);
    b.tap("q_c-" + sl, qc);
    ggml_tensor * qr = b.rms_w(qc, im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * gate = ggml_sigmoid(gc, ernie_mm(gc, im.BL(il, "attn_gate.weight"), qr));   // [OJ, n]
    b.tap("attn_gate-" + sl, gate);
    ggml_tensor * q = ggml_reshape_3d(gc, ernie_mm(gc, im.BL(il, "attn_q_b.weight"), qr), QKH, NH, n);
    q = rope(q, GGML_ROPE_TYPE_NORMAL, (int) NOPE);
    ggml_tensor * q_nope = ggml_view_3d(gc, q, NOPE, NH, n, q->nb[1], q->nb[2], 0);
    ggml_tensor * q_pe = ggml_cont(gc, ggml_view_3d(gc, q, DR, NH, n, q->nb[1], q->nb[2], (size_t) NOPE * q->nb[0]));
    b.tap("q_pe-" + sl, q_pe);

    // ---- kv latent: [kv_c (RMS) | k_pe (RoPE)]
    ggml_tensor * kvpe = ernie_mm(gc, im.BL(il, "attn_kv_a_mqa.weight"), xn);                 // [KROW, n]
    b.tap("kv_cmpr_pe-" + sl, kvpe);
    ggml_tensor * kvc = b.rms_w(ggml_cont(gc, ggml_view_2d(gc, kvpe, KL, n, kvpe->nb[1], 0)),
                                im.BL(il, "attn_kv_a_norm.weight"));
    b.tap("kv_cmpr-" + sl, kvc);
    ggml_tensor * kpe = ggml_cont(gc, ggml_view_2d(gc, kvpe, DR, n, kvpe->nb[1], (size_t) KL * kvpe->nb[0]));
    kpe = rope(ggml_reshape_3d(gc, kpe, DR, 1, n), GGML_ROPE_TYPE_NORMAL, 0);
    b.tap("k_pe-" + sl, kpe);

    // ---- K absorption: q_abs[h] = k_b[h] q_nope[h]  (k_b [NOPE, KL, NH])
    ggml_tensor * qn_p = ggml_cont(gc, ggml_permute(gc, q_nope, 0, 2, 1, 3));                  // [NOPE, n, NH]
    ggml_tensor * qabs = ernie_mm(gc, im.BL(il, "attn_k_b.weight"), qn_p);                     // [KL, n, NH]
    qabs = ggml_cont(gc, ggml_permute(gc, qabs, 0, 2, 1, 3));                                  // [KL, NH, n]
    b.tap("q_nope_absorbed-" + sl, qabs);
    // a rotated cache: rotate the queries alike (q.k is unchanged under an orthogonal H applied to both)
    auto hrot = [&](ggml_tensor * h, ggml_tensor * x) {   // H x over dim 0, any shape
        const int64_t w = x->ne[0];
        ggml_tensor * x2 = ggml_reshape_2d(gc, ggml_cont(gc, x), w, ggml_nelements(x) / w);
        return ggml_reshape(gc, ggml_mul_mat(gc, h, x2), x);
    };
    ggml_tensor * q_pe_r = q_pe;
    if (im.mla_rot) { qabs = hrot(im.had_kl, qabs); q_pe_r = hrot(im.had_dr, q_pe); }
    ggml_tensor * Qcur = ggml_concat(gc, qabs, q_pe_r, 0);                                     // [KROW, NH, n]
    b.tap("Qcur-" + sl, Qcur);

    // ---- indexer key: LayerNorm (w + b, eps 1e-6), NEOX RoPE on the first d_rope
    ggml_tensor * ik = ernie_mm(gc, im.BL(il, "indexer.attn_k.weight"), xn);                   // [ID, n]
    ik = ggml_add(gc, ggml_mul(gc, ggml_norm(gc, ik, 1e-6f), im.BL(il, "indexer.k_norm.weight")),
                  im.BL(il, "indexer.k_norm.bias"));
    ik = rope(ggml_reshape_3d(gc, ik, ID, 1, n), GGML_ROPE_TYPE_NEOX, 0);
    b.tap("indexer_k-" + sl, ik);

    // ---- this pass's cache rows [kv_c | k_pe | idx_k] at their positions; every read below goes through `cw`
    ggml_tensor * kpe2 = ggml_reshape_2d(gc, kpe, DR, n), * ik2 = ggml_reshape_2d(gc, ik, ID, n), * kvc2 = kvc;
    if (im.mla_rot) { kvc2 = hrot(im.had_kl, kvc); kpe2 = hrot(im.had_dr, kpe2); ik2 = hrot(im.had_id, ik2); }
    ggml_tensor * kcur = ggml_concat(gc, ggml_concat(gc, kvc2, kpe2, 0), ik2, 0);              // [ROW, n]
    b.tap("Kcur-" + sl, kcur);
    ggml_tensor * cache = im.ly[(size_t) il].mla;
    const bool mq = ggml_is_quantized(cache->type);
    ggml_tensor * cw = ggml_set_rows(gc, cache, (cache->type == GGML_TYPE_F32 || mq) ? kcur : ggml_cast(gc, kcur, cache->type),
                                     b.v1(im.i_pos, n, 0));
    ggml_build_forward_expand(gf, cw);
    if (mq)   // every read below wants F16 rows: dequantise the M cells this pass reads (Q8_0 / Q4_0 -> F32 -> F16)
        cw = ggml_cast(gc, ggml_cast(gc, ggml_view_2d(gc, cw, cw->ne[0], M, cw->nb[1], 0), GGML_TYPE_F32), GGML_TYPE_F16);

    // ---- the mask over the first M cells: d = p_t - j; causal s0 = [d >= 0]
    ggml_tensor * jr = ggml_reshape_2d(gc, ggml_view_1d(gc, im.j_ramp, M, 0), M, 1);
    ggml_tensor * pr = ggml_repeat(gc, ggml_reshape_2d(gc, b.v1(im.i_posf, n, 0), 1, n),
                                   ggml_new_tensor_2d(gc, GGML_TYPE_F32, M, n));
    ggml_tensor * d = ggml_sub(gc, pr, jr);                                                    // [M, n]
    ggml_tensor * s0 = ggml_step(gc, ggml_scale_bias(gc, d, 1.0f, 0.5f));
    ggml_tensor * keep = s0;
    if (topk || b.taps) {
        // indexer query / head weights; score[j, t] = sum_h relu(q_h . k_j) w_h
        ggml_tensor * iq = ggml_reshape_3d(gc, ernie_mm(gc, im.BL(il, "indexer.attn_q_b.weight"), qr), ID, IH, n);
        iq = rope(iq, GGML_ROPE_TYPE_NEOX, 0);
        if (im.mla_rot) iq = hrot(im.had_id, iq);   // the cached idx_k is rotated
        b.tap("indexer_q-" + sl, iq);
        ggml_tensor * iw = ggml_scale(gc, ernie_mm(gc, im.BL(il, "indexer.proj.weight"), xn),
                                      1.0f / std::sqrt((float) (ID * IH)));                    // [IH, n]
        b.tap("indexer_weights-" + sl, iw);
        ggml_tensor * kidx = ggml_cont(gc, ggml_view_2d(gc, cw, ID, M, cw->nb[1], (size_t) KROW * cw->nb[0]));
        // the [M, IH, n] F32 head scores (and their relu / weighted / permuted copies) grow with the context x the
        // chunk: at 8K cells x 256 rows they are ~2 GB, the graph failed to allocate beside the expert cache.  Build
        // them per column block of at most ~64 MiB each (the allocator reuses one block's buffers for the next).
        const int64_t NB = std::max<int64_t>(1, std::min<int64_t>(n, (64ll << 20) / std::max<int64_t>(1, M * IH * 4)));
        ggml_tensor * iq2 = ggml_reshape_3d(gc, ggml_cont(gc, iq), ID, IH, n);
        ggml_tensor * score = nullptr;
        for (int64_t c0 = 0; c0 < n; c0 += NB) {
            const int64_t nc = std::min<int64_t>(NB, n - c0);
            ggml_tensor * iqb = ggml_view_2d(gc, iq2, ID, IH * nc, iq2->nb[1], (size_t) c0 * iq2->nb[2]);
            ggml_tensor * iwb = ggml_view_3d(gc, iw, 1, IH, nc, iw->nb[0], iw->nb[1], (size_t) c0 * iw->nb[1]);
            ggml_tensor * dots = ernie_mm(gc, kidx, iqb);                                      // [M, IH*nc]
            ggml_mul_mat_set_prec(dots, GGML_PREC_F32);
            dots = ggml_mul(gc, ggml_reshape_3d(gc, ggml_relu(gc, dots), M, IH, nc), iwb);
            ggml_tensor * sb = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_cont(gc, ggml_permute(gc, dots, 1, 0, 2, 3))),
                                               M, nc);                                         // [M, nc]
            score = score ? ggml_concat(gc, score, sb, 1) : sb;
        }                                                                                      // [M, n]
        b.tap("indexer_score-" + sl, score);
        if (topk) {
            // window cells forced in, future cells out: + 1e30 * (2 s0 - s1 - 1), s1 = [d >= window]
            ggml_tensor * s1 = ggml_step(gc, ggml_scale_bias(gc, d, 1.0f, 0.5f - (float) G.n_swa));
            ggml_tensor * bonus = ggml_scale(gc, ggml_scale_bias(gc, ggml_sub(gc, ggml_scale(gc, s0, 2.0f), s1), 1.0f, -1.0f),
                                             1e30f);
            ggml_tensor * sel = ggml_add(gc, score, bonus);
            ggml_tensor * top = ggml_argsort_top_k(gc, sel, (int) NTOP);                       // sorted, descending
            b.tap("indexer_top_k-" + sl, top);
            ggml_tensor * kth = ggml_view_2d(gc, top, 1, n, top->nb[1], (size_t) (NTOP - 1) * top->nb[0]);
            ggml_tensor * thr = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, M, n), ggml_cont(gc, kth)),
                                                1, n);                                         // the k-th largest
            ggml_tensor * below = ggml_step(gc, ggml_sub(gc, ggml_scale(gc, sel, -1.0f), ggml_scale(gc, thr, -1.0f)));
            keep = ggml_mul(gc, ggml_scale_bias(gc, below, -1.0f, 1.0f), s0);                 // [sel >= thr] & causal
        }
    }
    ggml_tensor * mask = ggml_cast(gc, ggml_log(gc, keep), GGML_TYPE_F16);                     // 0 / -inf, [M, n]

    // ---- attention: K = rows' first KROW, V = rows' first KL, every cached cell under the mask
    ggml_tensor * K = ggml_view_4d(gc, cw, KROW, M, 1, 1, cw->nb[1], cw->nb[1] * M, cw->nb[1] * M, 0);
    ggml_tensor * V = ggml_view_4d(gc, cw, KL, M, 1, 1, cw->nb[1], cw->nb[1] * M, cw->nb[1] * M, 0);
    if (K->type != GGML_TYPE_F16) { K = ggml_cast(gc, K, GGML_TYPE_F16); V = ggml_cast(gc, V, GGML_TYPE_F16); }
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, Qcur, KROW, NH, n, 1), 0, 2, 1, 3);   // [KROW, n, NH, 1]
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, K, V, mask, 1.0f / std::sqrt((float) QKH), 0.0f, 0.0f);
    if (ernie_prec_f32()) ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, KL, NH, n);                                                     // the latent per head
    if (im.mla_rot) o = hrot(im.had_kl, o);   // V was the rotated kv_c: rotate the weighted sum back (H = H^-1)

    // ---- per-head V expansion v_b[h] (v_b [KL, DV, NH]), the VHA mix, the gate, wo
    ggml_tensor * op = ggml_cont(gc, ggml_permute(gc, o, 0, 2, 1, 3));                         // [KL, n, NH]
    ggml_tensor * vo = ernie_mm(gc, im.BL(il, "attn_v_b.weight"), op);                         // [DV, n, NH]
    vo = ggml_cont(gc, ggml_permute(gc, vo, 0, 2, 1, 3));                                      // [DV, NH, n]
    b.tap("attn_mla-" + sl, ggml_reshape_2d(gc, vo, OJ, n));
    {
        ggml_tensor * xt = ggml_reshape_2d(gc, ggml_cont(gc, ggml_permute(gc, vo, 1, 0, 2, 3)), NH, DV * n);
        xt = ggml_reshape_3d(gc, ernie_mm(gc, im.BL(il, "attn_vha.weight"), xt), NH, DV, n);
        vo = ggml_cont(gc, ggml_permute(gc, xt, 1, 0, 2, 3));                                  // [DV, NH, n]
    }
    b.tap("attn_head_mix-" + sl, vo);
    ggml_tensor * mixed = ggml_mul(gc, ggml_reshape_2d(gc, vo, OJ, n), gate);
    ggml_tensor * attn = ernie_mm(gc, im.BL(il, "attn_output.weight"), mixed);                 // [D, n]
    b.tap("attn_out-" + sl, attn);
    return attn;
}

// attn_router: from the residual to (ids, wts, fn) + shexp / hap / post_f / comb_f / pre_state.
static ggml_cgraph * build_attn(ErnieGraph::Impl & im, ggml_context * gc, int il,
                                std::vector<ggml_tensor *> * taps) {
    B b { &im, gc, taps };
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, OG = im.OG, OL = im.OL, OGD = im.OGD;
    const int64_t n = im.n, OJ = OG * OL;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    std::vector<ggml_tensor *> outs, late;
    ggml_tensor * pos = b.v1(im.i_pos, n, 0);
    const std::string sl = std::to_string(il);

    // ---- mHC pre-mixes of the attention
    ggml_tensor * x = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    hc_mixes(im, gc, x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"),
             im.BL(il, "hc_attn_base.weight"), &a_pre, &a_post, &a_comb, b, "hc_pre-" + sl, "hc_post-" + sl);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, a_pre);
    b.tap("hc_attn_pre-" + sl, xin);
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));
    b.tap("attn_norm-" + sl, xn);

    ggml_tensor * attn = nullptr, * kv16 = nullptr;
    if (!im.g.is_hca(il)) {
        attn = build_mla(im, gc, gf, b, il, xn, pos);
    } else {
    // ---- q (q-lora, per-head unweighted RMS, YaRN on the last 64) and the output gate's input
    ggml_tensor * qc = ernie_mm(gc, im.BL(il, "attn_q_a.weight"), xn);            // the q latent (pre-norm)
    b.tap("qr-" + sl, qc);
    ggml_tensor * qr = b.rms_w(qc, im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * gate = ggml_sigmoid(gc, ernie_mm(gc, im.BL(il, "attn_gate.weight"), qr));
    b.tap("attn_gate-" + sl, gate);
    ggml_tensor * q = ggml_reshape_3d(gc, ernie_mm(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = ggml_rms_norm(gc, q, (float) im.g.rms_eps);   // per head, unweighted
    b.tap("q_norm-" + sl, q);
    ggml_tensor * qr_rope = rope_hca(gc, im.g, q, pos, false);
    b.tap("q-" + sl, qr_rope);

    // ---- one MQA K/V row per token
    ggml_tensor * kv = b.rms_w(ernie_mm(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = rope_hca(gc, im.g, ggml_reshape_3d(gc, kv, DH, 1, n), pos, false);
    b.tap("kv-" + sl, kv);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    kv16 = im.cfg.kv_f16 ? ggml_cast(gc, ggml_cont(gc, kv2), GGML_TYPE_F16) : ggml_cont(gc, kv2);

    // ---- the per-layer compressor: projections into the pending 128-block, a pool on every block that closes here
    if (!im.draft) {
        ggml_tensor * skv = ernie_mm(gc, im.BL(il, "attn_compressor_kv.weight"), xn);     // [DH, n]
        ggml_tensor * ssc = ernie_mm(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
        ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
        ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
        ggml_tensor * ape_r = ggml_get_rows(gc, im.BL(il, "attn_compressor_ape.weight"), b.v1(im.i_pslot, n, 0));
        ssc = ggml_add(gc, ssc, b.flat2(ape_r, DH, n));                                         // + APE[pos % 128]
        b.tap("hca_state_score_ape-" + sl, ssc);
        int64_t blk_ord = 0;
        ggml_tensor * pend_kv = im.ly[(size_t) il].pend_kv, * pend_sc = im.ly[(size_t) il].pend_sc;
        for (const auto & s : im.segs) {
            ggml_tensor * pslot = b.v1(im.i_pslot, s.k, (size_t) s.c0 * 4);
            ggml_build_forward_expand(gf, ggml_set_rows(gc, pend_kv, ggml_cont(gc, b.cols2(skv, DH, s.k, s.c0)), pslot));
            ggml_build_forward_expand(gf, ggml_set_rows(gc, pend_sc, ggml_cont(gc, b.cols2(ssc, DH, s.k, s.c0)), pslot));
            if (!s.closes) continue;
            // softmax over the block's 128 positions per dim, weighted sum of the wkv rows
            ggml_tensor * V = ggml_cont(gc, ggml_permute(gc, pend_kv, 1, 0, 2, 3));             // [128, DH]
            ggml_tensor * S = ggml_cont(gc, ggml_permute(gc, pend_sc, 1, 0, 2, 3));
            ggml_tensor * W = ggml_soft_max(gc, S);
            ggml_tensor * pooled = ggml_sum_rows(gc, ggml_mul(gc, V, W));                       // [1, DH]
            pooled = ggml_cont(gc, ggml_permute(gc, pooled, 1, 0, 2, 3));                       // [DH, 1]
            b.tap("hca_compress-" + sl + "-" + std::to_string(blk_ord), pooled);
            pooled = b.rms_w(pooled, im.BL(il, "attn_compressor_norm.weight"));
            pooled = rope_hca(gc, im.g, ggml_reshape_3d(gc, pooled, DH, 1, 1),
                              b.v1(im.i_cpos, 1, (size_t) blk_ord * 4), false);
            ggml_tensor * val = im.cfg.kv_f16 ? ggml_cast(gc, pooled, GGML_TYPE_F16) : pooled;
            ggml_tensor * wnode = ggml_set_rows(gc, im.ly[(size_t) il].comp, ggml_cont_2d(gc, val, DH, 1),
                                               b.v1(im.i_cslot, 1, (size_t) blk_ord * 4));
            ggml_build_forward_expand(gf, wnode);
            ++blk_ord;
        }
    }

    // ---- attention over [window | every completed compressed row], per-head sinks
    {
        ggml_tensor * o = nullptr;
        if (im.draft) {
            // DSpark block: ONE key list for every query, [the committed ring | this block's rows], padded to DM
            // (i_dmask: invalid ring rows and the padding -inf).  Non-causal inside the block, as the reference.
            ggml_tensor * Kd = ggml_concat(gc, im.ly[(size_t) il].raw, kv16, 1);                // [DH, RAW + n]
            if (im.DM > im.RAW + n)
                Kd = ggml_concat(gc, Kd, ggml_fill(gc, ggml_new_tensor_2d(gc, Kd->type, DH, im.DM - im.RAW - n), 0.0f), 1);
            ggml_tensor * k4 = ggml_reshape_4d(gc, Kd, DH, im.DM, 1, 1);
            if (k4->type != GGML_TYPE_F16) k4 = ggml_cast(gc, k4, GGML_TYPE_F16);
            ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, qr_rope, DH, NH, n, 1), 0, 2, 1, 3);   // [DH, n, NH, 1]
            ggml_tensor * m4 = ggml_cast(gc, b.cols2(im.i_dmask, im.DM, n), GGML_TYPE_F16);   // [DM, n]
            o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        } else {
        ggml_tensor * wsrc = ggml_concat(gc, im.ly[(size_t) il].raw, kv16, 1);                 // [DH, RAW + n]
        ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, b.v1(im.i_widx, im.RAW * n, 0)),
                                           DH, im.RAW, n);
        ggml_tensor * Mw = b.cols2(im.i_wmask, im.RAW, n);
        ggml_tensor * Kc = nullptr, * Mc = nullptr;
        if (im.ncomp > 0) {
            ggml_tensor * ramp = ggml_reshape_2d(gc, ggml_view_1d(gc, im.pos_ramp, im.ncomp, 0), im.ncomp, 1);
            ggml_tensor * rep = ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.ncomp, n));
            ggml_tensor * cidx = ggml_cast(gc, ggml_reshape_1d(gc, rep, im.ncomp * n), GGML_TYPE_I32);
            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, im.ly[(size_t) il].comp, cidx), DH, im.ncomp, n);
            // mask: 0 for row j < nvis_t, -inf past it
            ggml_tensor * nramp = ggml_reshape_2d(gc, ggml_view_1d(gc, im.neg_ramp, im.ncomp, 0), im.ncomp, 1);
            ggml_tensor * r2 = ggml_repeat(gc, nramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.ncomp, n));
            ggml_tensor * nv = ggml_reshape_2d(gc, b.v1(im.i_cnvis, n, 0), 1, n);
            Mc = ggml_log(gc, ggml_step(gc, ggml_add(gc, r2, nv)));       // 0 / -inf
        }
        ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;
        ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;
        int64_t M = Kall->ne[1];
        if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
            const int64_t pad = 256 - M % 256;
            Kall = ggml_concat(gc, Kall, ggml_fill(gc, ggml_new_tensor_3d(gc, Kall->type, DH, pad, n), 0.0f), 1);
            Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
            M += pad;
        }
        ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, n)
                                                       : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, qr_rope, DH, NH, 1, n), 0, 2, 1, 3);
        ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
        o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        }
        if (ernie_prec_f32()) ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
        o = ggml_reshape_3d(gc, o, DH, NH, n);
        b.tap("attn_hca-" + sl, ggml_reshape_2d(gc, o, DH * NH, n));   // the reference's flat [d_head*n_head, n]
        o = rope_hca(gc, im.g, o, pos, true);                       // de-RoPE
        b.tap("attn_derope-" + sl, o);

        // ---- VHA: out[h] = sum_g (I + V U^T)[h, g] x[g]  (build_head_mix)
        {
            ggml_tensor * xt = ggml_cont(gc, ggml_permute(gc, o, 1, 0, 2, 3));          // [NH, DH, n]
            xt = ggml_reshape_2d(gc, xt, NH, DH * n);
            xt = ernie_mm(gc, im.BL(il, "attn_vha.weight"), xt);                    // [NH, DH*n]
            xt = ggml_reshape_3d(gc, xt, NH, DH, n);
            o = ggml_cont(gc, ggml_permute(gc, xt, 1, 0, 2, 3));                        // [DH, NH, n]
        }
        // ---- grouped low-rank output x the sigmoid gate, then wo_b
        ggml_tensor * op = ggml_permute(gc, ggml_reshape_3d(gc, o, OGD, OG, n), 0, 2, 1, 3);   // [OGD, n, OG]
        ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), OGD, OL, OG);
        ggml_tensor * oamm = ernie_mm(gc, wo_a, op);                                       // [OL, n, OG]
        b.tap("attn_wo_a-" + sl, oamm);                                                        // (the PR's layout)
        ggml_tensor * oa = ggml_permute(gc, oamm, 0, 2, 1, 3);                                 // [OL, OG, n]
        ggml_tensor * oaf = ggml_cont_2d(gc, oa, OJ, n);
        oaf = ggml_mul(gc, oaf, gate);
        attn = ernie_mm(gc, im.BL(il, "attn_output_b.weight"), oaf);
        b.tap("attn_out-" + sl, attn);
    }   // the HCA attention
    }   // HCA
    {
        // ---- mHC post (attention), then the FFN mixes
        ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);
        b.tap("hc_attn_post-" + sl, h);
        ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
        hc_mixes(im, gc, h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"),
                 im.BL(il, "hc_ffn_base.weight"), &f_pre, &f_post, &f_comb, b,
                 "hc_ffn_pre_mix-" + sl, "hc_ffn_post-" + sl);
        ggml_tensor * xffn = ggml_dsv4_hc_pre(gc, h, f_pre);
        b.tap("hc_ffn_pre-" + sl, xffn);
        ggml_tensor * fn = b.rms_w(xffn, im.BL(il, "ffn_norm.weight"));
        b.tap("ffn_norm-" + sl, fn);

        const bool dense = il < im.g.dense_lead;
        ggml_tensor * se = nullptr, * latent = nullptr, * ids = nullptr, * wts = nullptr;
        if (dense) {
            ggml_tensor * fg = ernie_mm(gc, im.BL(il, "ffn_gate.weight"), fn);
            ggml_tensor * fu = ernie_mm(gc, im.BL(il, "ffn_up.weight"), fn);
            ggml_tensor * act = sit_u(gc, fg, fu, (float) im.g.situ_beta, (float) im.g.situ_lb);
            b.tap("ffn_situ-" + sl, act);
            se = ernie_mm(gc, im.BL(il, "ffn_down.weight"), act);
        } else {
            // split-feature routing: the two [NE, D] scorers stacked in ffn_gate_inp
            ggml_tensor * rw = im.BL(il, "ffn_gate_inp.weight");                 // [D, 2*NE]
            ggml_tensor * r0 = ggml_view_2d(gc, rw, D, im.NE, rw->nb[1], 0);
            ggml_tensor * r1 = ggml_view_2d(gc, rw, D, im.NE, rw->nb[1], (size_t) im.NE * rw->nb[1]);
            ggml_tensor * probs = ggml_add(gc, ggml_sigmoid(gc, ernie_mm(gc, r0, fn)),
                                            ggml_sigmoid(gc, ernie_mm(gc, r1, fn)));
            b.tap("ffn_moe_probs_split-" + sl, probs);
            ggml_tensor * sel = ggml_add(gc, probs, im.BL(il, "exp_probs_b.bias"));   // the bias selects only
            ids = ggml_argsort_top_k(gc, sel, (int) im.NU);
            b.tap("ffn_moe_topk-" + sl, ids);
            ggml_tensor * wsel = ggml_get_rows(gc, ggml_reshape_3d(gc, probs, 1, im.NE, n), ids);   // [1, NU, n]
            ggml_tensor * wsum = ggml_clamp(gc, ggml_sum_rows(gc, ggml_reshape_2d(gc, wsel, im.NU, n)), 1e-20f, INFINITY);
            ggml_tensor * pf = ggml_div(gc, ggml_mul(gc, probs, im.BL(il, "exp_scale.weight")), wsum);
            b.tap("ffn_moe_probs-" + sl, pf);
            wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, pf, 1, im.NE, n), ids), im.NU, n);
            latent = ernie_mm(gc, im.BL(il, "ffn_routed_down.weight"), fn);     // the 2048-wide latent rows
            b.tap("ffn_routed_down-" + sl, latent);
            ggml_tensor * sg = ernie_mm(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
            ggml_tensor * su = ernie_mm(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
            se = ernie_mm(gc, im.BL(il, "ffn_down_shexp.weight"),
                              sit_u(gc, sg, su, (float) im.g.situ_beta, (float) im.g.situ_lb));
            b.tap("ffn_shexp-" + sl, se);
        }
        // ---- the layer's state writes (after every read of x_state / pre_state)
        outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, 0)));
        outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, 0)));
        outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, 0)));
        outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, 0)));
        outs.push_back(ggml_cpy(gc, se, b.cols2(im.shexp, D, n, 0)));
        if (dense) {
            outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, 0)));
        } else {
            outs.push_back(ggml_cpy(gc, latent, b.cols2(im.o_lat, im.LAT, n, 0)));
            outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, 0)));
            outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, 0)));
        }
        // ---- the window ring: the pass's last min(n, RAW) rows at their slots (every read of `raw` is above)
        if (kv16 && !im.draft) {   // (a draft block's rows are scratch: dspark_commit writes the committed ones)
            const int64_t k = std::min<int64_t>(n, im.RAW);
            ggml_tensor * last = ggml_view_2d(gc, kv16, DH, k, kv16->nb[1], (size_t) (n - k) * kv16->nb[1]);
            late.push_back(ggml_set_rows(gc, im.ly[(size_t) il].raw, ggml_cont(gc, last), b.v1(im.i_wslot, k, 0)));
        }
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    if (taps) for (ggml_tensor * t : *taps) ggml_build_forward_expand(gf, t);
    return gf;
}

// finish: latent_norm(routed sum) -> fc2_latent_proj -> + shared expert -> mHC post -> the residual state
// predict(il): layer il's router on the CURRENT residual (x_state = layer il-1's output, before il's attention):
// the FFN mHC pre-mix and ffn_norm of layer il applied to it straight away, then the split-feature scorers + the
// selection bias, top kPredK.  The DS4.1 engine's finish_predict trick (the attention changes the residual a little,
// so the guess is usually the true top-k plus neighbours); its ids feed Ds41MoeTier::prefetch / spec_read only and
// never change any output.
static ggml_cgraph * build_predict(ErnieGraph::Impl & im, ggml_context * gc, int il) {
    B b { &im, gc, nullptr };
    const int64_t D = im.D, HC = im.HC, n = im.n;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 1024, false);
    ggml_tensor * x = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    hc_mixes(im, gc, x, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"),
             im.BL(il, "hc_ffn_base.weight"), &f_pre, &f_post, &f_comb, b, "pred_pre", "pred_post");
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, x, f_pre), im.BL(il, "ffn_norm.weight"));
    ggml_tensor * rw = im.BL(il, "ffn_gate_inp.weight");
    ggml_tensor * r0 = ggml_view_2d(gc, rw, D, im.NE, rw->nb[1], 0);
    ggml_tensor * r1 = ggml_view_2d(gc, rw, D, im.NE, rw->nb[1], (size_t) im.NE * rw->nb[1]);
    ggml_tensor * sel = ggml_add(gc, ggml_add(gc, ggml_sigmoid(gc, ernie_mm(gc, r0, fn)), ggml_sigmoid(gc, ernie_mm(gc, r1, fn))),
                                 im.BL(il, "exp_probs_b.bias"));
    ggml_tensor * ids = ggml_argsort_top_k(gc, sel, (int) ErnieGraph::kPredK);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ids, b.cols2(im.o_pred, ErnieGraph::kPredK, n, 0)));
    return gf;
}

static ggml_cgraph * build_finish(ErnieGraph::Impl & im, ggml_context * gc, int il, std::vector<ggml_tensor *> * taps) {
    B b { &im, gc, taps };
    const int64_t D = im.D, HC = im.HC, n = im.n, LAT = im.LAT;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    const bool dense = il < im.g.dense_lead;
    ggml_tensor * se = ggml_view_2d(gc, im.shexp, D, n, im.shexp->nb[1], 0);
    ggml_tensor * out = se;
    if (!dense) {
        ggml_tensor * lat = ggml_view_2d(gc, im.routed_in, LAT, n, im.routed_in->nb[1], 0);
        ggml_tensor * ln = b.rms_w(lat, im.BL(il, "ffn_routed_norm.weight"));
        ggml_tensor * rup = ernie_mm(gc, im.BL(il, "ffn_routed_up.weight"), ln);
        b.tap("ffn_routed_up-" + std::to_string(il), rup);
        out = ggml_add(gc, rup, se);
    }
    b.tap("ffn_out-" + std::to_string(il), out);
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, ggml_view_3d(gc, im.hap, D, HC, n, im.hap->nb[1], im.hap->nb[2], 0),
                                        ggml_view_2d(gc, im.post_f, HC, n, im.post_f->nb[1], 0),
                                        ggml_view_3d(gc, im.comb_f, HC, HC, n, im.comb_f->nb[1], im.comb_f->nb[2], 0));
    b.tap("l_last-" + std::to_string(il), x);
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0)));
    if (taps) for (ggml_tensor * t : *taps) ggml_build_forward_expand(gf, t);
    return gf;
}

// the head: hc_head mixer, output_norm, output.weight, MultiMax
static ggml_cgraph * build_head(ErnieGraph::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg,
                                std::vector<ggml_tensor *> * taps = nullptr) {
    B b { &im, gc, taps };
    const int64_t D = im.D, HC = im.HC;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    ggml_tensor * xs = ggml_view_3d(gc, im.x_state, D, HC, nr, im.x_state->nb[1], im.x_state->nb[2],
                                    (size_t) row0 * im.x_state->nb[2]);
    ggml_tensor * h = hc_head(im, gc, xs);
    b.tap("hc_head", h);
    h = b.rms_w(h, im.W("output_norm.weight"));
    b.tap("result_norm", h);
    ggml_tensor * raw = ernie_mm(gc, im.W("output.weight"), h);
    b.tap("result_output_raw", raw);
    ggml_tensor * lg0 = multimax(im, gc, raw);
    b.tap("result_output", lg0);
    ggml_set_output(lg0);
    *lg = lg0;
    ggml_build_forward_expand(gf, lg0);
    if (taps) for (ggml_tensor * t : *taps) ggml_build_forward_expand(gf, t);
    return gf;
}

// ================================================================== compute

bool ErnieGraph::Impl::compute(ggml_cgraph * gf, bool cached) {
    Allo & a = cached ? allo_dec : allo_pass;
    if (!a.alloc(gf, cached && env_on("ERNIE_GRAPH_REUSE", true))) { err = "graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    if (!cached) ggml_backend_synchronize(backend);
    return true;
}

static void read_taps(std::vector<ggml_tensor *> & tp, std::vector<ErnieGraphTap> * out) {
    if (!out) return;
    for (ggml_tensor * t : tp) {
        ErnieGraphTap g;
        g.name = ggml_get_name(t);
        g.ne = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        g.data.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, g.data.data(), 0, g.data.size() * 4);
        out->push_back(std::move(g));
    }
}

// ================================================================== public API

bool ErnieGraph::attn_router(int il, int * ids, float * wts, const float ** fn_host, std::vector<ErnieGraphTap> * taps) {
    Impl & im = *p_;
    if (im.n < 1) { im.err = "no pass: call begin() first"; return false; }
    if (il < 0 || il >= im.nl) { im.err = "layer outside the model"; return false; }
    const bool dense = il < im.g.dense_lead;
    std::vector<ggml_tensor *> tp;
    ggml_cgraph * gf = nullptr;
    bool cached = false;
    std::vector<uint8_t> meta;
    // the graph's shape depends on the pass width, the number of visible compressed rows and the segment layout
    const int64_t layout = im.draft ? 0 : (im.n == 1) ? (im.pos0 % kHcaRatio == kHcaRatio - 1 ? 1 : 0) : (im.pos0 % kHcaRatio);
    if (taps) {
        meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        gf = build_attn(im, gc, il, &tp);
    } else {
        const bool mla = !im.g.is_hca(il);
        const Impl::GKey key { il, mla ? 5 : 1,
                               mla ? mla_cells(im) * 2 + (mla_topk_on(im) ? 1 : 0) : im.ncomp * 1024 + layout, im.n };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();
            it = im.graphs.find(key);
            if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, nullptr)).first;
        }
        gf = it->second;
        cached = true;
    }
    if (!im.compute(gf, cached)) return false;

    if (!dense) {
        ggml_backend_tensor_get(im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * im.n) * 4);
        ggml_backend_tensor_get(im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * im.n) * 4);
        std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * im.n) * 4);
        std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * im.n) * 4);
    } else {
        for (int64_t i = 0; i < im.NU * im.n; ++i) ids[i] = -1;
        std::memset(wts, 0, (size_t) (im.NU * im.n) * 4);
    }
    const int64_t fnw = dense ? im.D : im.LAT;
    ggml_backend_tensor_get(dense ? im.o_fn : im.o_lat, im.host_fn.data(), 0, (size_t) (fnw * im.n) * 4);
    *fn_host = im.host_fn.data();
    read_taps(tp, taps);
    return true;
}

bool ErnieGraph::finish(int il, const float * routed_sum, std::vector<ErnieGraphTap> * taps) {
    Impl & im = *p_;
    if (im.n < 1) { im.err = "no pass: call begin() first"; return false; }
    const bool dense = il < im.g.dense_lead;
    if (!dense) {
        if (!routed_sum) { im.err = "finish: layer " + std::to_string(il) + " needs the routed latent sum"; return false; }
        ggml_backend_tensor_set(im.routed_in, routed_sum, 0, (size_t) (im.LAT * im.n) * 4);
    }
    std::vector<ggml_tensor *> tp;
    ggml_cgraph * gf = nullptr;
    bool cached = false;
    std::vector<uint8_t> meta;
    if (taps) {
        meta.resize(ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(1024, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        gf = build_finish(im, gc, il, &tp);
    } else {
        const Impl::GKey key { il, dense ? 3 : 2, 0, im.n };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();
            it = im.graphs.find(key);
            if (it == im.graphs.end()) it = im.graphs.emplace(key, build_finish(im, im.gctx, il, nullptr)).first;
        }
        gf = it->second;
        cached = true;
    }
    if (!im.compute(gf, cached)) return false;
    read_taps(tp, taps);
    return true;
}

bool ErnieGraph::predict(int il, int * top) {
    Impl & im = *p_;
    if (im.n < 1) { im.err = "no pass: call begin() first"; return false; }
    if (il < im.g.dense_lead || il >= im.nl) return false;   // no router there: nothing to guess
    const Impl::GKey key { il, 6, 0, im.n };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        if (!im.gctx_room()) im.reset_graphs();
        it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_predict(im, im.gctx, il)).first;
    }
    if (!im.compute(it->second, true)) return false;
    ggml_backend_tensor_get(im.o_pred, im.host_pred.data(), (size_t) (im.n - 1) * kPredK * 4, (size_t) kPredK * 4);
    std::memcpy(top, im.host_pred.data(), (size_t) kPredK * 4);
    return true;
}

bool ErnieGraph::logits(int row0, int nrows, float * out, std::vector<ErnieGraphTap> * taps) {
    Impl & im = *p_;
    if (im.n < 1 || row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "logits: bad rows"; return false; }
    std::vector<ggml_tensor *> tp;
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = nullptr;
    bool cached = false;
    std::vector<uint8_t> meta;
    if (taps) {
        meta.resize(ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        gf = build_head(im, gc, row0, nrows, &lg, &tp);
    } else {
        const Impl::GKey key { -1, 4, row0, nrows };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();
            ggml_cgraph * g = build_head(im, im.gctx, row0, nrows, &lg);
            it = im.graphs.emplace(key, g).first;
            im.head_out[key] = lg;
        }
        gf = it->second;
        lg = im.head_out[key];
        cached = true;
    }
    if (!im.compute(gf, cached)) return false;
    ggml_backend_tensor_get(lg, out, 0, (size_t) (nrows * im.g.vocab) * 4);
    read_taps(tp, taps);
    return true;
}

bool ErnieGraph::read_state(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (im.n < 1 || c0 < 0 || n < 1 || c0 + n > im.n) { im.err = "read_state: bad rows"; return false; }
    x.resize((size_t) (n * im.HC * im.D));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * im.HC * im.D * 4, x.size() * 4);
    return true;
}

bool ErnieGraph::read_pre(std::vector<float> & pre) {
    Impl & im = *p_;
    if (im.n < 1) { im.err = "read_pre: no pass"; return false; }
    pre.resize((size_t) (im.n * im.HC));
    ggml_backend_tensor_get(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

static void get_bytes(ggml_tensor * t, std::vector<uint8_t> & dst) {
    dst.resize((size_t) ggml_nbytes(t));
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size());
}

static void set_bytes(ggml_tensor * t, const std::vector<uint8_t> & src) {
    ggml_backend_tensor_set(t, src.data(), 0, src.size());
}

bool ErnieGraph::snapshot(int pos, int slot) {
    Impl & im = *p_;
    if (slot < 0 || slot >= kSnapSlots) { im.err = "snapshot: slot " + std::to_string(slot) + " out of range"; return false; }
    Impl::Snap & sn = im.snaps[slot];
    sn.raw.resize(im.ly.size());
    sn.kv.resize(im.ly.size());
    sn.sc.resize(im.ly.size());
    for (size_t l = 0; l < im.ly.size(); ++l) {
        if (!im.ly[l].raw) continue;   // MLA: rows by absolute position, rewritten past the snapshot
        get_bytes(im.ly[l].raw, sn.raw[l]);
        get_bytes(im.ly[l].pend_kv, sn.kv[l]);
        get_bytes(im.ly[l].pend_sc, sn.sc[l]);
    }
    sn.pos = pos;
    return true;
}

bool ErnieGraph::restore(int slot) {
    Impl & im = *p_;
    if (slot < 0 || slot >= kSnapSlots) { im.err = "restore: slot " + std::to_string(slot) + " out of range"; return false; }
    const Impl::Snap & sn = im.snaps[slot];
    if (sn.pos < 0 || sn.raw.size() != im.ly.size()) { im.err = "restore: no snapshot in slot " + std::to_string(slot); return false; }
    for (size_t l = 0; l < im.ly.size(); ++l) {
        if (!im.ly[l].raw) continue;
        set_bytes(im.ly[l].raw, sn.raw[l]);
        set_bytes(im.ly[l].pend_kv, sn.kv[l]);
        set_bytes(im.ly[l].pend_sc, sn.sc[l]);
    }
    im.next_pos = sn.pos;
    im.n = 0;
    im.pos0 = sn.pos;
    return true;
}

// ================================================================== DSpark

// dspark_commit's graph: main_x = main_norm(main_proj(main_hidden)); per stage s: kv = kv_a_norm(wkv_s(main_x)) +
// the HCA rope at each row's position, written into stage s's ring at pos % 128 (the reference's CommitKvProj +
// set_swa_key_buffer_radix_fused_norm_rope).
static ggml_cgraph * build_commit(ErnieGraph::Impl & im, ggml_context * gc, int64_t k) {
    B b { &im, gc, nullptr };
    const int64_t D = im.D, DH = im.DH;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    ggml_tensor * x = ggml_view_2d(gc, im.i_mh, 3 * D, k, im.i_mh->nb[1], 0);
    ggml_tensor * mx = b.rms_w(ernie_mm(gc, im.W("dspark.main_proj.weight"), x), im.W("dspark.main_norm.weight"));
    ggml_tensor * pos = b.v1(im.i_cwpos, k, 0);
    ggml_tensor * slot = b.v1(im.i_cwslot, k, 0);
    for (int64_t s = 0; s < im.nl; ++s) {
        ggml_tensor * kv = b.rms_w(ernie_mm(gc, im.BL(s, "attn_kv.weight"), mx), im.BL(s, "attn_kv_a_norm.weight"));
        kv = rope_hca(gc, im.g, ggml_reshape_3d(gc, kv, DH, 1, k), pos, false);
        ggml_tensor * kv2 = ggml_cont_2d(gc, kv, DH, k);
        if (im.cfg.kv_f16) kv2 = ggml_cast(gc, kv2, GGML_TYPE_F16);
        ggml_build_forward_expand(gf, ggml_set_rows(gc, im.ly[(size_t) s].raw, kv2, slot));
    }
    return gf;
}

// dspark_draft's graph: the head per block slot (base logits), then the Markov chain as argmax nodes (outs[t]).
static ggml_cgraph * build_draft_head(ErnieGraph::Impl & im, ggml_context * gc, int64_t nr, std::vector<ggml_tensor *> & outs) {
    B b { &im, gc, nullptr };
    const int64_t D = im.D, HC = im.HC, V = im.g.vocab;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 1024, false);
    ggml_tensor * xs = ggml_view_3d(gc, im.x_state, D, HC, nr, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * h = b.rms_w(hc_head(im, gc, xs), im.W("output_norm.weight"));
    ggml_tensor * base = multimax(im, gc, ernie_mm(gc, im.W("output.weight"), h));   // [V, nr]
    ggml_tensor * emb = im.W("dspark.markov_head.embed.weight");                      // [rank, V]: row = a token's latent
    ggml_tensor * head = im.W("dspark.markov_head.head.weight");                      // [rank, V]: w2, out = V
    ggml_tensor * prev = im.i_anchor;
    outs.clear();
    for (int64_t t = 0; t < nr; ++t) {
        ggml_tensor * e = ggml_get_rows(gc, emb, prev);                                // [rank, 1]
        ggml_tensor * bias = ernie_mm(gc, head, e);                                    // [V, 1]
        static const bool nomarkov = [] { const char * e = std::getenv("ERNIE_DSPARK_NOMARKOV"); return e && *e == '1'; }();
        ggml_tensor * bt = ggml_view_2d(gc, base, V, 1, base->nb[1], (size_t) t * base->nb[1]);
        ggml_tensor * lt = nomarkov ? ggml_cont(gc, bt) : ggml_add(gc, bt, bias);   // probe: base logits only
        prev = ggml_argmax(gc, lt);                                                    // I32 [1]
        ggml_set_output(prev);
        ggml_build_forward_expand(gf, prev);
        outs.push_back(prev);
    }
    return gf;
}

bool ErnieGraph::dspark_commit(const float * mh, int64_t pos0, int k) {
    Impl & im = *p_;
    if (!im.draft) { im.err = "dspark_commit: not a draft graph"; return false; }
    if (k < 1 || k > im.RAW || !mh) { im.err = "dspark_commit: 1..128 rows"; return false; }
    if (pos0 < 0 || !(pos0 == im.committed || k == im.RAW)) {   // in order, or a whole new ring
        im.err = "dspark_commit: rows at " + std::to_string(pos0) + " but " + std::to_string(im.committed) +
                 " are committed (commit in order, or a full 128-row ring)";
        return false;
    }
    std::vector<int32_t> pos((size_t) k), slot((size_t) k);
    for (int i = 0; i < k; ++i) { pos[(size_t) i] = (int32_t) (pos0 + i); slot[(size_t) i] = (int32_t) ((pos0 + i) % im.RAW); }
    ggml_backend_tensor_set(im.i_mh, mh, 0, (size_t) k * 3 * im.D * 4);
    ggml_backend_tensor_set(im.i_cwpos, pos.data(), 0, pos.size() * 4);
    ggml_backend_tensor_set(im.i_cwslot, slot.data(), 0, slot.size() * 4);
    const Impl::GKey key { -1, 7, 0, k };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        if (!im.gctx_room()) im.reset_graphs();
        it = im.graphs.emplace(key, build_commit(im, im.gctx, k)).first;
    }
    if (!im.compute(it->second, true)) return false;
    im.committed = pos0 + k;
    return true;
}

int64_t ErnieGraph::dspark_committed() const { return p_->committed; }

bool ErnieGraph::dspark_draft(int anchor, int * out) {
    Impl & im = *p_;
    if (!im.draft) { im.err = "dspark_draft: not a draft graph"; return false; }
    if (im.n < 1) { im.err = "dspark_draft: no block pass"; return false; }
    if (anchor < 0 || anchor >= im.g.vocab) { im.err = "dspark_draft: anchor outside the vocabulary"; return false; }
    const int32_t a = anchor;
    ggml_backend_tensor_set(im.i_anchor, &a, 0, 4);
    const Impl::GKey key { -1, 8, 0, im.n };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        if (!im.gctx_room()) im.reset_graphs();
        std::vector<ggml_tensor *> outs;
        it = im.graphs.emplace(key, build_draft_head(im, im.gctx, im.n, outs)).first;
        im.draft_out[key] = outs;
    }
    if (!im.compute(it->second, true)) return false;
    const std::vector<ggml_tensor *> & outs = im.draft_out[key];
    for (int t = 0; t < im.n; ++t) {
        int32_t v = 0;
        ggml_backend_tensor_get(outs[(size_t) t], &v, 0, 4);
        out[t] = v;
    }
    return true;
}

// rows [slot0, slot0 + n) of a ring of `t->ne[1]` rows (wrapping) to / from buf
static void ring_rows(ggml_tensor * t, int64_t slot0, int n, uint8_t * buf, bool get) {
    const int64_t R = t->ne[1];
    const size_t rb = t->nb[1];
    int done = 0;
    while (done < n) {
        const int64_t s = (slot0 + done) % R;
        const int take = (int) std::min<int64_t>(n - done, R - s);
        if (get) ggml_backend_tensor_get(t, buf + (size_t) done * rb, (size_t) s * rb, (size_t) take * rb);
        else ggml_backend_tensor_set(t, buf + (size_t) done * rb, (size_t) s * rb, (size_t) take * rb);
        done += take;
    }
}

bool ErnieGraph::spec_save(int64_t pos0, int n) {
    Impl & im = *p_;
    if (im.draft) { im.err = "spec_save: a trunk call"; return false; }
    if (n < 1 || n > kHcaRatio || pos0 != im.next_pos) { im.err = "spec_save: 1..128 tokens at the next position"; return false; }
    Impl::Spec & sp = im.spec;
    sp.raw.resize(im.ly.size());
    sp.kv.resize(im.ly.size());
    sp.sc.resize(im.ly.size());
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const Impl::Layer & L = im.ly[l];
        if (!L.raw) continue;
        sp.raw[l].resize((size_t) n * L.raw->nb[1]);
        sp.kv[l].resize((size_t) n * L.pend_kv->nb[1]);
        sp.sc[l].resize((size_t) n * L.pend_sc->nb[1]);
        ring_rows(L.raw, pos0 % L.raw->ne[1], n, sp.raw[l].data(), true);
        ring_rows(L.pend_kv, pos0 % L.pend_kv->ne[1], n, sp.kv[l].data(), true);
        ring_rows(L.pend_sc, pos0 % L.pend_sc->ne[1], n, sp.sc[l].data(), true);
    }
    sp.pos0 = pos0;
    sp.n = n;
    return true;
}

bool ErnieGraph::spec_rollback(int64_t keep_end) {
    Impl & im = *p_;
    Impl::Spec & sp = im.spec;
    if (sp.pos0 < 0 || keep_end < sp.pos0 || keep_end > sp.pos0 + sp.n) {
        im.err = "spec_rollback: no saved verify span covers " + std::to_string(keep_end);
        return false;
    }
    const int i0 = (int) (keep_end - sp.pos0), cnt = sp.n - i0;
    if (cnt > 0)
        for (size_t l = 0; l < im.ly.size(); ++l) {
            const Impl::Layer & L = im.ly[l];
            if (!L.raw) continue;
            ring_rows(L.raw, keep_end % L.raw->ne[1], cnt, sp.raw[l].data() + (size_t) i0 * L.raw->nb[1], false);
            ring_rows(L.pend_kv, keep_end % L.pend_kv->ne[1], cnt, sp.kv[l].data() + (size_t) i0 * L.pend_kv->nb[1], false);
            ring_rows(L.pend_sc, keep_end % L.pend_sc->ne[1], cnt, sp.sc[l].data() + (size_t) i0 * L.pend_sc->nb[1], false);
        }
    im.next_pos = keep_end;
    im.n = 0;
    im.pos0 = (int) keep_end;
    sp.pos0 = -1;
    return true;
}

bool ErnieGraph::read_state_mean(int c0, int n, float * out) {
    Impl & im = *p_;
    if (im.n < 1 || c0 < 0 || n < 1 || c0 + n > im.n || !out) { im.err = "read_state_mean: bad rows"; return false; }
    std::vector<float> x((size_t) (n * im.HC * im.D));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * im.HC * im.D * 4, x.size() * 4);
    const float inv = 1.0f / (float) im.HC;
    for (int t = 0; t < n; ++t)
        for (int64_t i = 0; i < im.D; ++i) {
            float acc = 0.0f;
            for (int64_t h = 0; h < im.HC; ++h) acc += x[(size_t) ((t * im.HC + h) * im.D + i)];
            out[(size_t) t * im.D + i] = acc * inv;
        }
    return true;
}

ErnieGraph::ErnieGraph() : p_(new Impl()) {}
ErnieGraph::~ErnieGraph() {
    // the buffers and contexts go first (Impl's destructor), the backend we own last
    ggml_backend_t be = p_->own_backend ? p_->backend : nullptr;
    p_->own_backend = false;
    p_.reset();
    if (be) ggml_backend_free(be);
}
