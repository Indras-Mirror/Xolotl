// tools/hy4/engine/hy4_dense.cpp - see hy4_dense.hpp.  Every formula cites the reference function it mirrors
// (llama.cpp PR #32517, src/models/hy-v4-flash.cpp, "ref:").
#include "hy4_dense.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::hy4 {

namespace {

// ref: HY_V4_FLASH_ENGRAM_EPS - the reference Engram norms leave eps unset (fp32 machine epsilon)
constexpr float kEngramEps = 1.1920929e-07f;
constexpr int64_t kDead = -1;

// F8_E4M3 (e4m3fn: bias 7, no inf, S.1111.111 = NaN) -> float
float e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = std::ldexp((float) m / 8.0f, -6);
    else if (e == 15 && m == 7) v = NAN;
    else v = std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return s ? -v : v;
}

struct Layer {
    ggml_tensor *attn_norm = nullptr, *sinks = nullptr, *wq_a = nullptr, *q_a_norm = nullptr, *wq_b = nullptr,
                *wkv_a = nullptr, *kv_a_norm = nullptr, *wk_b = nullptr, *wv_b = nullptr, *wo = nullptr,
                *wgate = nullptr, *post_attn_norm = nullptr;
    ggml_tensor *hc_attn_fn = nullptr, *hc_attn_base = nullptr, *hc_attn_scale = nullptr;
    ggml_tensor *hc_ffn_fn = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
    ggml_tensor *ffn_norm = nullptr, *post_ffn_norm = nullptr;
    // dense lead FFN
    ggml_tensor *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    // MoE (dense half)
    ggml_tensor *router = nullptr, *probs_b = nullptr, *routed_down = nullptr, *routed_up = nullptr;
    ggml_tensor *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;
    // Engram
    ggml_tensor *e_kv = nullptr, *e_kv_b = nullptr, *e_gate = nullptr, *e_lookup_norm = nullptr, *e_pre_gamma = nullptr,
                *e_hc_scale = nullptr;
    ggml_tensor * kcache = nullptr;   ///< [kv_lora + rope, ctx_max] F32
    // k-pool indexer (DSA layers): weights, the per-position value|gate rows and the pooled keys of complete groups
    ggml_tensor *ix_k = nullptr, *ix_k_norm = nullptr, *ix_val = nullptr, *ix_gate = nullptr, *ix_ape = nullptr,
                *ix_cnorm = nullptr, *ix_qb = nullptr, *ix_proj = nullptr;
    ggml_tensor * vg = nullptr;       ///< [2 * idx_dim, ctx_max]
    ggml_tensor * pooled = nullptr;   ///< [idx_dim, ctx_max / kpool]
};

}  // namespace

struct Hy4Dense::Impl {
    Hy4DenseConfig cfg;
    std::unique_ptr<GgufModel> model;
    Hy4Geometry G;
    std::string err;
    ggml_backend_t be = nullptr;
    ggml_gallocr_t galloc = nullptr;

    ggml_context * wctx = nullptr;               // weights + caches (no_alloc), one backend buffer each
    ggml_backend_buffer_t wbuf = nullptr, cbuf = nullptr;
    ggml_context * cctx = nullptr;
    std::vector<Layer> L;
    ggml_tensor *out_norm = nullptr, *output = nullptr, *head_fn = nullptr, *head_base = nullptr, *head_scale = nullptr;

    // host
    const uint8_t * tok_embd = nullptr;          // Q8_0 rows in the mmap
    size_t tok_row_bytes = 0;
    const uint8_t * engram_tab = nullptr;        // F8_E4M3 rows of engram_dim bytes
    std::vector<int64_t> hist;                   // token id by position (Engram history)

    // pass state (device, re-made per begin when n changes)
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    int n = 0, pos0 = 0, pn_alloc = 0;
    ggml_tensor *x = nullptr, *x_mid = nullptr, *post_ffn = nullptr, *shexp = nullptr, *pos = nullptr;
    std::vector<ggml_tensor *> e_rows, e_valid;  // per Engram module: [dim, cols, n], [cols, n]
    std::vector<float> fn_host;
    std::map<std::string, std::vector<float>> taps;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (lbuf) ggml_backend_buffer_free(lbuf);
        if (lctx) ggml_free(lctx);
        if (pbuf) ggml_backend_buffer_free(pbuf);
        if (pctx) ggml_free(pctx);
        if (cbuf) ggml_backend_buffer_free(cbuf);
        if (cctx) ggml_free(cctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }

    bool fail(const std::string & e) { err = e; std::fprintf(stderr, "hy4_dense: %s\n", e.c_str()); return false; }

    // ---- LoRA adapters (W x -> W x + s B (A x)); A / B live on the backend, F32
    struct Lora { std::string name; ggml_tensor *W = nullptr, *A = nullptr, *B = nullptr; float s = 0; int layer = -1; };
    std::vector<Lora> lora;
    std::map<const ggml_tensor *, int> lora_of;
    ggml_context * lctx = nullptr;
    ggml_backend_buffer_t lbuf = nullptr;
    ggml_tensor * mm(ggml_context * c, ggml_tensor * W, ggml_tensor * x) {
        ggml_tensor * y = ggml_mul_mat(c, W, x);
        auto it = lora_of.find(W);
        if (it == lora_of.end()) return y;
        const Lora & l = lora[(size_t) it->second];
        return ggml_add(c, y, ggml_scale(c, ggml_mul_mat(c, l.B, ggml_mul_mat(c, l.A, x)), l.s));
    }

    // ---- one graph = a fresh context + the shared allocator
    struct Graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> in;
        std::vector<std::pair<std::string, ggml_tensor *>> tap;
        ~Graph() { if (ctx) ggml_free(ctx); }
    };
    void new_graph(Graph & g) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true };
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, 16384, false);
    }
    ggml_tensor * input(Graph & g, ggml_type t, int64_t n0, int64_t n1, const void * data, size_t bytes) {
        ggml_tensor * x = ggml_new_tensor_2d(g.ctx, t, n0, n1);
        ggml_set_input(x);
        g.in.emplace_back(x, std::vector<uint8_t>((const uint8_t *) data, (const uint8_t *) data + bytes));
        return x;
    }
    void tap(Graph & g, ggml_tensor * t, const std::string & name) {
        if (!cfg.taps) return;
        ggml_tensor * c = ggml_cont(g.ctx, t);
        ggml_set_output(c);
        ggml_build_forward_expand(g.gf, c);
        g.tap.emplace_back(name, c);
    }
    bool run(Graph & g) {
        if (!ggml_gallocr_alloc_graph(galloc, g.gf)) return fail("graph allocation failed");
        for (auto & i : g.in) ggml_backend_tensor_set(i.first, i.second.data(), 0, i.second.size());
        if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) return fail("graph compute failed");
        if (cfg.taps) {
            taps.clear();
            for (auto & t : g.tap) {
                std::vector<float> v((size_t) ggml_nelements(t.second));
                ggml_backend_tensor_get(t.second, v.data(), 0, v.size() * 4);
                taps[t.first] = std::move(v);
            }
        }
        return true;
    }

    // ---- building blocks (ref: build_norm LLM_NORM_RMS, build_hc_pre / build_hc_post / build_hc_reduce)
    ggml_tensor * rms(ggml_context * c, ggml_tensor * a, ggml_tensor * w) {
        return ggml_mul(c, ggml_rms_norm(c, a, (float) G.rms_eps), w);
    }
    ggml_tensor * hc_reduce(ggml_context * c, ggml_tensor * x, ggml_tensor * w) {   // x [D, hc, n], w [hc, n]
        const int64_t D = G.n_embd, nt = x->ne[2];
        ggml_tensor * r = nullptr;
        for (int64_t h = 0; h < G.hc; ++h) {
            ggml_tensor * xh = ggml_view_2d(c, x, D, nt, x->nb[2], (size_t) h * x->nb[1]);
            ggml_tensor * wh = ggml_view_2d(c, w, 1, nt, w->nb[1], (size_t) h * w->nb[0]);
            ggml_tensor * t = ggml_mul(c, xh, wh);
            r = r ? ggml_add(c, r, t) : t;
        }
        return r;
    }
    // returns the reduced input; *post = the post gate [hc, n]
    ggml_tensor * hc_pre(ggml_context * c, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
                         ggml_tensor ** post) {
        const int64_t hc = G.hc, nt = x->ne[2];
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, hc * G.n_embd, nt), (float) G.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);   // [2hc, n]
        ggml_tensor * s_pre = ggml_view_1d(c, scale, 1, 0), * s_post = ggml_view_1d(c, scale, 1, scale->nb[0]);
        ggml_tensor * b_pre = ggml_view_1d(c, base, hc, 0), * b_post = ggml_view_1d(c, base, hc, (size_t) hc * base->nb[0]);
        ggml_tensor * pre = ggml_view_2d(c, mixes, hc, nt, mixes->nb[1], 0);
        pre = ggml_scale_bias(c, ggml_sigmoid(c, ggml_add(c, ggml_mul(c, pre, s_pre), b_pre)), 1.0f, (float) G.hc_eps);
        ggml_tensor * po = ggml_view_2d(c, mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
        po = ggml_sigmoid(c, ggml_add(c, ggml_mul(c, po, s_post), b_post));
        po = ggml_scale_bias(c, ggml_scale(c, po, (float) G.hc_magnitude), 1.0f, (float) G.hc_eps);
        *post = po;
        return hc_reduce(c, x, pre);
    }
    // identity comb: stream i = residual_i + y * post_i   (y [D, n], residual [D, hc, n], post [hc, n])
    ggml_tensor * hc_post(ggml_context * c, ggml_tensor * y, ggml_tensor * residual, ggml_tensor * post) {
        const int64_t D = G.n_embd, nt = y->ne[1];
        ggml_tensor * out = nullptr;
        for (int64_t i = 0; i < G.hc; ++i) {
            ggml_tensor * ri = ggml_view_2d(c, residual, D, nt, residual->nb[2], (size_t) i * residual->nb[1]);
            ggml_tensor * pi = ggml_view_2d(c, post, 1, nt, post->nb[1], (size_t) i * post->nb[0]);
            ggml_tensor * t = ggml_reshape_3d(c, ggml_add(c, ri, ggml_mul(c, y, pi)), D, 1, nt);
            out = out ? ggml_concat(c, out, t, 1) : t;
        }
        return out;
    }
    ggml_tensor * swiglu_ffn(ggml_context * c, ggml_tensor * x, ggml_tensor * g, ggml_tensor * u, ggml_tensor * d) {
        return mm(c, d, ggml_swiglu_split(c, mm(c, g, x), mm(c, u, x)));
    }

    // ref: build_engram
    ggml_tensor * engram(ggml_context * c, const Layer & ly, ggml_tensor * x, ggml_tensor * rows, ggml_tensor * valid) {
        const int64_t hc = G.hc, nt = x->ne[2], D = G.n_embd, cols = G.engram_cols(), dim = G.engram_dim;
        ggml_tensor * r = ggml_rms_norm(c, rows, kEngramEps);                  // [dim, cols, n]
        r = ggml_mul(c, r, ly.e_lookup_norm);                                   // [dim, cols]
        r = ggml_mul(c, r, ggml_reshape_3d(c, valid, 1, cols, nt));
        r = ggml_reshape_2d(c, r, dim * cols, nt);
        ggml_tensor * kv = ggml_mul_mat(c, ly.e_kv, r);                         // [D*(hc+1), n]
        kv = ggml_add(c, ggml_reshape_3d(c, kv, D, hc + 1, nt), ly.e_kv_b);
        ggml_tensor * key = ggml_cont(c, ggml_view_3d(c, kv, D, hc, nt, kv->nb[1], kv->nb[2], 0));
        ggml_tensor * value = ggml_cont(c, ggml_view_3d(c, kv, D, 1, nt, kv->nb[1], kv->nb[2], (size_t) hc * kv->nb[1]));
        ggml_tensor * dot = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, x, kEngramEps), ggml_rms_norm(c, key, kEngramEps)), ly.e_gate);
        dot = ggml_scale(c, ggml_sum_rows(c, dot), 1.0f / std::sqrt((float) D));   // [1, hc, n]
        ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, dot), 1e-6f, INFINITY));
        ggml_tensor * sign = ggml_scale_bias(c, ggml_step(c, ggml_neg(c, dot)), -2.0f, 1.0f);
        ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, mag, sign));
        ggml_tensor * signal = ggml_mul(c, ggml_rms_norm(c, value, kEngramEps), ly.e_pre_gamma);   // [D, 1, n]
        ggml_tensor * scale = ggml_add(c, ggml_repeat_4d(c, gate, D, hc, nt, 1), ly.e_hc_scale);
        ggml_tensor * delta = ggml_mul(c, ggml_repeat_4d(c, signal, D, hc, nt, 1), scale);
        ggml_tensor * any = ggml_step(c, ggml_sum_rows(c, ggml_reshape_2d(c, valid, cols, nt)));
        delta = ggml_mul(c, delta, ggml_reshape_3d(c, any, 1, 1, nt));
        return ggml_add(c, x, delta);
    }

    // ref: build_indexer_rope - rotate only the trailing d_rope of each indexer head ([idx_dim, heads, n])
    ggml_tensor * idx_rope(ggml_context * c, ggml_tensor * x, ggml_tensor * p, float base) {
        const int64_t nr = G.d_rope, nf = G.idx_dim - nr;
        ggml_tensor * front = ggml_view_3d(c, x, nf, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        ggml_tensor * back = ggml_view_3d(c, x, nr, x->ne[1], x->ne[2], x->nb[1], x->nb[2], ggml_row_size(x->type, nf));
        back = ggml_rope_ext(c, ggml_cont(c, back), p, nullptr, (int) nr, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        return ggml_concat(c, ggml_cont(c, front), back, 0);
    }
    // the groups of kpool positions this pass completes: [g0, g1)
    void new_groups(int64_t & g0, int64_t & g1) const {
        // a group g is completed by this pass when its last member kp*g + kp - 1 lies in [pos0, pos0 + n)
        const int64_t kp = G.idx_kpool;
        g0 = (pos0 + 1) / kp + ((pos0 + 1) % kp != 0 ? 0 : 0);
        while (g0 * kp + kp - 1 < pos0) ++g0;
        while (g0 > 0 && (g0 - 1) * kp + kp - 1 >= pos0) --g0;
        g1 = std::max(g0, (int64_t) (pos0 + n) / kp);
    }
    // ref: build_kpool_select, the cache half: value|gate rows of this pass, then pool the groups it completes
    void kpool_maintain(Graph & g, int il, ggml_tensor * cur) {
        ggml_context * c = g.ctx;
        const Layer & ly = L[(size_t) il];
        const int64_t di = G.idx_dim, kp = G.idx_kpool, nt = n;
        const float base = (float) (G.swa(il) ? G.rope_base_swa : G.rope_base);
        ggml_tensor * ik = rms(c, ggml_mul_mat(c, ly.ix_k, cur), ly.ix_k_norm);             // [di, n]
        ggml_tensor * vgr = ggml_concat(c, ggml_mul_mat(c, ly.ix_val, ik), ggml_mul_mat(c, ly.ix_gate, ik), 0);   // [2di, n]
        ggml_tensor * dst = ggml_view_2d(c, ly.vg, 2 * di, nt, ly.vg->nb[1], (size_t) pos0 * ly.vg->nb[1]);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, vgr, dst));
        int64_t g0, g1;
        new_groups(g0, g1);
        if (g1 <= g0) return;
        const int64_t ng = g1 - g0;
        ggml_tensor * rows = ggml_view_3d(c, ly.vg, 2 * di, kp, ng, ly.vg->nb[1], ly.vg->nb[1] * kp, (size_t) (g0 * kp) * ly.vg->nb[1]);
        ggml_tensor * pk = ggml_view_3d(c, rows, di, kp, ng, rows->nb[1], rows->nb[2], 0);
        ggml_tensor * pg = ggml_view_3d(c, rows, di, kp, ng, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, di));
        ggml_tensor * lg = ggml_cont(c, ggml_permute(c, ggml_add(c, pg, ly.ix_ape), 1, 0, 2, 3));   // [kp, di, ng]
        ggml_tensor * pr = ggml_reshape_3d(c, ggml_soft_max(c, ggml_reshape_2d(c, lg, kp, di * ng)), kp, di, ng);
        ggml_tensor * pkt = ggml_cont(c, ggml_permute(c, pk, 1, 0, 2, 3));
        ggml_tensor * pooled = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_mul(c, pr, pkt)), di, ng);
        pooled = rms(c, pooled, ly.ix_cnorm);
        std::vector<int32_t> pp((size_t) ng);
        for (int64_t i = 0; i < ng; ++i) pp[(size_t) i] = (int32_t) ((g0 + i) * kp);   // rotated at the first member
        ggml_tensor * ppos = ggml_new_tensor_1d(c, GGML_TYPE_I32, ng);
        ggml_set_input(ppos);
        g.in.emplace_back(ppos, std::vector<uint8_t>((uint8_t *) pp.data(), (uint8_t *) pp.data() + pp.size() * 4));
        pooled = ggml_reshape_2d(c, idx_rope(c, ggml_reshape_3d(c, pooled, di, 1, ng), ppos, base), di, ng);
        ggml_tensor * pdst = ggml_view_2d(c, ly.pooled, di, ng, ly.pooled->nb[1], (size_t) g0 * ly.pooled->nb[1]);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, pooled, pdst));
    }
    // ref: build_kpool_select, the scoring half: top (top_k / kpool - 1) visible pools per token -> host (top [K, n])
    ggml_tensor * kpool_score(Graph & g, int il, ggml_tensor * cur, ggml_tensor * qr, int64_t n_pool, int64_t K) {
        ggml_context * c = g.ctx;
        const Layer & ly = L[(size_t) il];
        const int64_t di = G.idx_dim, nh = G.idx_heads, kp = G.idx_kpool, nt = n;
        const float base = (float) (G.swa(il) ? G.rope_base_swa : G.rope_base);
        ggml_tensor * iq = idx_rope(c, ggml_reshape_3d(c, ggml_mul_mat(c, ly.ix_qb, qr), di, nh, nt), pos, base);   // [di, nh, n]
        ggml_tensor * w = ggml_scale(c, ggml_mul_mat(c, ly.ix_proj, cur), 1.0f / std::sqrt((float) (di * nh)));     // [nh, n]
        ggml_tensor * pk = ggml_view_2d(c, ly.pooled, di, n_pool, ly.pooled->nb[1], 0);
        ggml_tensor * kq = ggml_mul_mat(c, pk, ggml_permute(c, iq, 0, 2, 1, 3));                  // [P, n, nh]
        kq = ggml_cont(c, ggml_permute(c, kq, 2, 1, 0, 3));                                       // [nh, n, P]
        ggml_tensor * sc = ggml_sum_rows(c, ggml_mul(c, ggml_relu(c, kq), w));                     // [1, n, P]
        sc = ggml_cont(c, ggml_permute(c, sc, 2, 1, 0, 3));                                       // [P, n, 1]
        sc = ggml_reshape_2d(c, sc, n_pool, nt);
        std::vector<float> pm((size_t) (n_pool * nt));
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t vis = (pos0 + t) / kp;   // pools ending before the token (ref: lower_bound on pool_end)
            for (int64_t q = 0; q < n_pool; ++q) pm[(size_t) (t * n_pool + q)] = q < vis ? 0.0f : -INFINITY;
        }
        sc = ggml_add(c, sc, input(g, GGML_TYPE_F32, n_pool, nt, pm.data(), pm.size() * 4));
        tap(g, sc, "indexer_score");
        ggml_tensor * top = ggml_top_k(c, sc, (int) K);                                            // [K, n] I32
        ggml_set_output(top);
        ggml_build_forward_expand(g.gf, top);
        (void) kp;
        return top;
    }

    // ref: build_attention (no indexer: full causal on DSA layers, see the header)
    ggml_tensor * attention(Graph & g, int il, ggml_tensor * cur, const std::vector<float> * sel_mask, bool maintain) {
        ggml_context * c = g.ctx;
        const Layer & ly = L[(size_t) il];
        const int64_t H = G.heads(il), dk = G.d_qk, dr = G.d_rope, dn = G.d_nope(), kvl = G.kv_lora, nt = n;
        const int64_t krow = kvl + dr, n_kv = pos0 + n;
        const float kq_scale = 1.0f / std::sqrt((float) dk);
        const float base = (float) (G.swa(il) ? G.rope_base_swa : G.rope_base);

        ggml_tensor * qr = rms(c, mm(c, ly.wq_a, cur), ly.q_a_norm);              // q_resid [q_lora, n]
        if (G.indexed(il) && maintain) kpool_maintain(g, il, cur);
        ggml_tensor * q = mm(c, ly.wq_b, qr);                                     // [H*dk, n]
        const size_t es = ggml_element_size(q);
        ggml_tensor * q_nope = ggml_view_3d(c, q, dn, H, nt, es * dk, es * dk * H, 0);
        ggml_tensor * q_pe = ggml_view_3d(c, q, dr, H, nt, es * dk, es * dk * H, es * dn);
        ggml_tensor * kvp = mm(c, ly.wkv_a, cur);                                 // [krow, n]
        ggml_tensor * kv_c = ggml_view_2d(c, kvp, kvl, nt, kvp->nb[1], 0);
        ggml_tensor * k_pe = ggml_view_3d(c, kvp, dr, 1, nt, kvp->nb[1], kvp->nb[1], es * kvl);
        q_pe = ggml_rope_ext(c, ggml_cont(c, q_pe), pos, nullptr, (int) dr, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k_pe = ggml_rope_ext(c, ggml_cont(c, k_pe), pos, nullptr, (int) dr, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        kv_c = rms(c, kv_c, ly.kv_a_norm);                                        // kv_cmpr [kvl, n]
        tap(g, kv_c, "kv_cmpr");

        ggml_tensor * qa = ggml_mul_mat(c, ly.wk_b, ggml_permute(c, q_nope, 0, 2, 1, 3));   // [kvl, n, H]
        qa = ggml_permute(c, qa, 0, 2, 1, 3);                                     // [kvl, H, n]
        ggml_tensor * Q = ggml_concat(c, qa, q_pe, 0);                            // [krow, H, n]
        ggml_tensor * K = ggml_concat(c, ggml_reshape_3d(c, kv_c, kvl, 1, nt), k_pe, 0);   // [krow, 1, n]

        // write this pass's rows, then attend over [0, n_kv)
        ggml_tensor * kdst = ggml_view_2d(c, ly.kcache, krow, nt, ly.kcache->nb[1], (size_t) pos0 * ly.kcache->nb[1]);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_2d(c, K, krow, nt), kdst));
        ggml_tensor * Kall = ggml_view_2d(c, ly.kcache, krow, n_kv, ly.kcache->nb[1], 0);
        ggml_tensor * Vt = ggml_cont(c, ggml_transpose(c, ggml_view_2d(c, ly.kcache, kvl, n_kv, ly.kcache->nb[1], 0)));  // [n_kv, kvl]

        // mask [n_kv, n]: causal, and the sliding window on window layers (ref: LLAMA_SWA_TYPE_STANDARD, p - pk >= n_swa)
        std::vector<float> m((size_t) (n_kv * nt));
        if (sel_mask) m = *sel_mask;
        else for (int64_t t = 0; t < nt; ++t) {
            const int64_t p = pos0 + t;
            for (int64_t k = 0; k < n_kv; ++k) {
                bool ok = k <= p;
                if (G.swa(il) && p - k >= G.n_swa) ok = false;
                m[(size_t) (t * n_kv + k)] = ok ? 0.0f : -INFINITY;
            }
        }
        ggml_tensor * mask = input(g, GGML_TYPE_F32, n_kv, nt, m.data(), m.size() * 4);

        ggml_tensor * Qp = ggml_permute(c, Q, 0, 2, 1, 3);                        // [krow, n, H]
        ggml_tensor * kq = ggml_mul_mat(c, Kall, Qp);                             // [n_kv, n, H]
        kq = ggml_soft_max_ext(c, kq, mask, kq_scale, 0.0f);
        ggml_soft_max_add_sinks(kq, ly.sinks);
        ggml_tensor * kqv = ggml_mul_mat(c, Vt, kq);                              // [kvl, n, H]
        ggml_tensor * a = ggml_mul_mat(c, ly.wv_b, kqv);                          // [d_v, n, H]
        a = ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3));                         // [d_v, H, n]
        a = ggml_reshape_2d(c, a, G.d_v * H, nt);
        tap(g, a, "attn_kqv");
        a = ggml_mul(c, a, ggml_sigmoid(c, mm(c, ly.wgate, cur)));
        ggml_tensor * o = rms(c, mm(c, ly.wo, a), ly.post_attn_norm);
        tap(g, o, "attn_out");
        return o;
    }

    // training: the same attention over the whole sequence of the pass (no cache: K / V are this pass's rows), `mask`
    // [n, n] = causal + sliding window (every DSA pool is selected below kpool_exact_limit)
    ggml_tensor * attention_train(ggml_context * c, int il, ggml_tensor * cur, ggml_tensor * mask) {
        const Layer & ly = L[(size_t) il];
        const int64_t H = G.heads(il), dk = G.d_qk, dr = G.d_rope, dn = G.d_nope(), kvl = G.kv_lora, nt = n;
        const int64_t krow = kvl + dr;
        const float kq_scale = 1.0f / std::sqrt((float) dk);
        const float base = (float) (G.swa(il) ? G.rope_base_swa : G.rope_base);
        ggml_tensor * qr = rms(c, mm(c, ly.wq_a, cur), ly.q_a_norm);
        ggml_tensor * q = mm(c, ly.wq_b, qr);
        const size_t es = ggml_element_size(q);
        ggml_tensor * q_nope = ggml_view_3d(c, q, dn, H, nt, es * dk, es * dk * H, 0);
        ggml_tensor * q_pe = ggml_view_3d(c, q, dr, H, nt, es * dk, es * dk * H, es * dn);
        ggml_tensor * kvp = mm(c, ly.wkv_a, cur);
        ggml_tensor * kv_c = ggml_view_2d(c, kvp, kvl, nt, kvp->nb[1], 0);
        ggml_tensor * k_pe = ggml_view_3d(c, kvp, dr, 1, nt, kvp->nb[1], kvp->nb[1], es * kvl);
        q_pe = ggml_rope_ext(c, ggml_cont(c, q_pe), pos, nullptr, (int) dr, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k_pe = ggml_rope_ext(c, ggml_cont(c, k_pe), pos, nullptr, (int) dr, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        kv_c = rms(c, ggml_cont(c, kv_c), ly.kv_a_norm);
        ggml_tensor * qa = ggml_mul_mat(c, ly.wk_b, ggml_cont(c, ggml_permute(c, ggml_cont(c, q_nope), 0, 2, 1, 3)));   // [kvl, n, H]
        qa = ggml_permute(c, qa, 0, 2, 1, 3);
        ggml_tensor * Q = ggml_concat(c, ggml_cont(c, qa), q_pe, 0);                               // [krow, H, n]
        ggml_tensor * K = ggml_reshape_2d(c, ggml_concat(c, ggml_reshape_3d(c, kv_c, kvl, 1, nt), k_pe, 0), krow, nt);
        ggml_tensor * Vt = ggml_cont(c, ggml_transpose(c, kv_c));                                   // [n, kvl]
        ggml_tensor * kq = ggml_mul_mat(c, K, ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3)));       // [n, n, H]
        kq = ggml_soft_max_ext(c, kq, mask, kq_scale, 0.0f);
        ggml_soft_max_add_sinks(kq, ly.sinks);
        ggml_tensor * kqv = ggml_mul_mat(c, Vt, kq);                                               // [kvl, n, H]
        ggml_tensor * a = ggml_mul_mat(c, ly.wv_b, kqv);                                           // [d_v, n, H]
        a = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, a, 0, 2, 1, 3)), G.d_v * H, nt);
        a = ggml_mul(c, a, ggml_sigmoid(c, mm(c, ly.wgate, cur)));
        return rms(c, mm(c, ly.wo, a), ly.post_attn_norm);
    }
};

Hy4Dense::Hy4Dense() : p_(new Impl) {}
Hy4Dense::~Hy4Dense() = default;
const Hy4Geometry & Hy4Dense::geom() const { return p_->G; }
const GgufModel & Hy4Dense::model() const { return *p_->model; }
const std::string & Hy4Dense::last_error() const { return p_->err; }
const std::map<std::string, std::vector<float>> & Hy4Dense::taps() const { return p_->taps; }
int64_t Hy4Dense::kpool_exact_limit() const {
    // a query at p sees floor(p / kpool) complete pools; all are selected while that is <= top_k / kpool - 1
    return (p_->G.idx_topk / p_->G.idx_kpool - (p_->G.idx_select_tail ? 1 : 0) + 1) * p_->G.idx_kpool;
}

bool Hy4Dense::init(const std::string & gguf, const Hy4DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    im.be = cfg.backend;
    if (!im.be) { err = "no backend"; return false; }
    try {
        im.model.reset(new GgufModel(GgufModel::open(gguf)));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    err = hy4_read_geometry(*im.model, gguf, im.G);
    if (!err.empty()) return false;
    const Hy4Geometry & G = im.G;
    const GgufModel & M = *im.model;

    // ---- weights: every tensor except the routed experts, the Engram table and the token embeddings
    auto is_host_only = [](const std::string & n) {
        return n == "token_embd.weight" || n == "per_layer_token_embd.weight" || n.find("_exps.weight") != std::string::npos;
    };
    ggml_init_params ip = { ggml_tensor_overhead() * 4096, nullptr, true };
    im.wctx = ggml_init(ip);
    std::map<std::string, std::pair<ggml_tensor *, std::pair<size_t, const TensorInfo *>>> made;
    for (size_t s = 0; s < M.size(); ++s)
        for (const auto & ti : M.shard(s).tensors()) {
            if (is_host_only(ti.name) || ti.name.rfind("blk.", 0) == 0 && std::atoll(ti.name.c_str() + 4) >= G.n_layer) continue;
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < 4; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_tensor * t = ggml_new_tensor(im.wctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
            ggml_set_name(t, ti.name.c_str());
            made[ti.name] = { t, { s, &ti } };
        }
    im.wbuf = ggml_backend_alloc_ctx_tensors(im.wctx, im.be);
    if (!im.wbuf) { err = "cannot allocate the dense weights"; return false; }
    size_t bytes = 0;
    for (auto & kv : made) {
        const size_t s = kv.second.second.first;
        const TensorInfo & ti = *kv.second.second.second;
        ggml_backend_tensor_set(kv.second.first, M.shard(s).tensor_data(ti), 0, ggml_nbytes(kv.second.first));
        bytes += ggml_nbytes(kv.second.first);
    }
    auto W = [&](const std::string & n, bool req = true) -> ggml_tensor * {
        auto it = made.find(n);
        if (it == made.end()) {
            if (req) err = "missing " + n;
            return nullptr;
        }
        return it->second.first;
    };
    im.L.resize((size_t) G.n_layer);
    for (int64_t l = 0; l < G.n_layer; ++l) {
        Layer & y = im.L[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.attn_norm = W(p + "attn_norm.weight"); y.sinks = W(p + "attn_sinks.weight");
        y.wq_a = W(p + "attn_q_a.weight"); y.q_a_norm = W(p + "attn_q_a_norm.weight"); y.wq_b = W(p + "attn_q_b.weight");
        y.wkv_a = W(p + "attn_kv_a_mqa.weight"); y.kv_a_norm = W(p + "attn_kv_a_norm.weight");
        y.wk_b = W(p + "attn_k_b.weight"); y.wv_b = W(p + "attn_v_b.weight"); y.wo = W(p + "attn_output.weight");
        y.wgate = W(p + "attn_gate.weight"); y.post_attn_norm = W(p + "post_attention_norm.weight");
        y.hc_attn_fn = W(p + "hc_attn_fn.weight"); y.hc_attn_base = W(p + "hc_attn_base.weight"); y.hc_attn_scale = W(p + "hc_attn_scale.weight");
        y.hc_ffn_fn = W(p + "hc_ffn_fn.weight"); y.hc_ffn_base = W(p + "hc_ffn_base.weight"); y.hc_ffn_scale = W(p + "hc_ffn_scale.weight");
        y.ffn_norm = W(p + "ffn_norm.weight"); y.post_ffn_norm = W(p + "post_ffw_norm.weight");
        if (!G.moe(l)) {
            y.ffn_gate = W(p + "ffn_gate.weight"); y.ffn_up = W(p + "ffn_up.weight"); y.ffn_down = W(p + "ffn_down.weight");
        } else {
            y.router = W(p + "ffn_gate_inp.weight"); y.probs_b = W(p + "exp_probs_b.bias");
            y.routed_down = W(p + "ffn_routed_down.weight"); y.routed_up = W(p + "ffn_routed_up.weight");
            y.sh_gate = W(p + "ffn_gate_shexp.weight"); y.sh_up = W(p + "ffn_up_shexp.weight"); y.sh_down = W(p + "ffn_down_shexp.weight");
        }
        if (G.indexed(l)) {
            y.ix_k = W(p + "indexer.attn_k.weight"); y.ix_k_norm = W(p + "indexer.k_norm.weight");
            y.ix_val = W(p + "indexer_compressor_value.weight"); y.ix_gate = W(p + "indexer_compressor_gate.weight");
            y.ix_ape = W(p + "indexer_compressor_ape.weight"); y.ix_cnorm = W(p + "indexer_compressor_norm.weight");
            y.ix_qb = W(p + "indexer.attn_q_b.weight"); y.ix_proj = W(p + "indexer.proj.weight");
        }
        if (G.engram_slot(l) >= 0) {
            y.e_kv = W(p + "engram_kv.weight"); y.e_kv_b = W(p + "engram_kv_b.weight"); y.e_gate = W(p + "engram_gate.weight");
            y.e_lookup_norm = W(p + "engram_lookup_norm.weight"); y.e_pre_gamma = W(p + "engram_pre_gamma.weight");
            y.e_hc_scale = W(p + "engram_hc_scale.weight");
        }
        if (!err.empty()) return false;
    }
    im.out_norm = W("output_norm.weight"); im.output = W("output.weight");
    im.head_fn = W("output_hc_fn.weight"); im.head_base = W("output_hc_base.weight"); im.head_scale = W("output_hc_scale.weight");
    if (!err.empty()) return false;
    if (im.L[1].e_gate == nullptr && G.engram_slot(1) >= 0) { err = "engram weights"; return false; }

    // ---- K caches
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.cctx = ggml_init(cp);
    for (size_t l = 0; l < im.L.size(); ++l) {
        Layer & y = im.L[l];
        y.kcache = ggml_new_tensor_2d(im.cctx, GGML_TYPE_F32, G.kv_lora + G.d_rope, cfg.ctx_max);
        if (G.indexed((int64_t) l)) {
            y.vg = ggml_new_tensor_2d(im.cctx, GGML_TYPE_F32, 2 * G.idx_dim, cfg.ctx_max);
            y.pooled = ggml_new_tensor_2d(im.cctx, GGML_TYPE_F32, G.idx_dim, cfg.ctx_max / G.idx_kpool + 1);
        }
    }
    im.cbuf = ggml_backend_alloc_ctx_tensors(im.cctx, im.be);
    if (!im.cbuf) { err = "cannot allocate the K caches"; return false; }
    ggml_backend_buffer_clear(im.cbuf, 0);

    // ---- host tables
    size_t si = 0;
    const TensorInfo * te = M.find("token_embd.weight", &si);
    if (!te || (ggml_type) te->type != GGML_TYPE_Q8_0) { err = "token_embd.weight must be Q8_0"; return false; }
    im.tok_embd = M.shard(si).tensor_data(*te);
    im.tok_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, G.n_embd);
    const TensorInfo * tt = M.find("per_layer_token_embd.weight", &si);
    // F8_E4M3: the reader retags upstream's type 43 to strata::kTypeF8E4M3 by size (43 is MXFP8 in older DS4.1 GGUFs);
    // a raw 43 is still accepted for a table too small for the size test to decide
    if (!tt || (tt->type != strata::kTypeF8E4M3 && tt->type != 43) || (int64_t) tt->shape[0] != G.engram_dim) { err = "per_layer_token_embd.weight must be F8_E4M3 [dim, rows]"; return false; }
    im.engram_tab = M.shard(si).tensor_data(*tt);
    im.hist.assign((size_t) cfg.ctx_max, -1);

    im.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.be));
    std::fprintf(stderr, "hy4_dense: %zu dense tensors, %.2f GB on %s; K caches %.2f GB (ctx %lld)\n", made.size(), bytes / 1e9,
                 ggml_backend_name(im.be), ggml_backend_buffer_get_size(im.cbuf) / 1e9, (long long) cfg.ctx_max);
    return true;
}

bool Hy4Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    if (n <= 0 || n > im.cfg.max_pass) return im.fail("pass size out of range");
    if (pos0 < 0 || pos0 + n > im.cfg.ctx_max) return im.fail("position past ctx_max");
    im.n = n;
    im.pos0 = pos0;
    for (int i = 0; i < n; ++i) im.hist[(size_t) (pos0 + i)] = tids[i];

    // pass tensors
    if (im.pn_alloc != n) {
        if (im.pbuf) ggml_backend_buffer_free(im.pbuf);
        if (im.pctx) ggml_free(im.pctx);
        ggml_init_params pp = { ggml_tensor_overhead() * 64, nullptr, true };
        im.pctx = ggml_init(pp);
        im.x = ggml_new_tensor_3d(im.pctx, GGML_TYPE_F32, G.n_embd, G.hc, n);
        im.x_mid = ggml_new_tensor_3d(im.pctx, GGML_TYPE_F32, G.n_embd, G.hc, n);
        im.post_ffn = ggml_new_tensor_2d(im.pctx, GGML_TYPE_F32, G.hc, n);
        im.shexp = ggml_new_tensor_2d(im.pctx, GGML_TYPE_F32, G.n_embd, n);
        im.pos = ggml_new_tensor_1d(im.pctx, GGML_TYPE_I32, n);
        im.e_rows.clear();
        im.e_valid.clear();
        for (size_t e = 0; e < G.engram_layers.size(); ++e) {
            im.e_rows.push_back(ggml_new_tensor_3d(im.pctx, GGML_TYPE_F32, G.engram_dim, G.engram_cols(), n));
            im.e_valid.push_back(ggml_new_tensor_2d(im.pctx, GGML_TYPE_F32, G.engram_cols(), n));
        }
        im.pbuf = ggml_backend_alloc_ctx_tensors(im.pctx, im.be);
        if (!im.pbuf) return im.fail("cannot allocate the pass state");
        im.pn_alloc = n;
    }
    // embeddings -> hc streams (ref: hc_init = repeat)
    {
        const auto * tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
        std::vector<float> row((size_t) G.n_embd), xs((size_t) (G.n_embd * G.hc * n));
        for (int i = 0; i < n; ++i) {
            if (tids[i] < 0 || tids[i] >= G.vocab) return im.fail("token id out of range");
            tr->to_float(im.tok_embd + (size_t) tids[i] * im.tok_row_bytes, row.data(), G.n_embd);
            for (int64_t h = 0; h < G.hc; ++h)
                std::memcpy(&xs[(size_t) ((i * G.hc + h) * G.n_embd)], row.data(), (size_t) G.n_embd * 4);
        }
        ggml_backend_tensor_set(im.x, xs.data(), 0, xs.size() * 4);
        std::vector<int32_t> p((size_t) n);
        for (int i = 0; i < n; ++i) p[(size_t) i] = pos0 + i;
        ggml_backend_tensor_set(im.pos, p.data(), 0, p.size() * 4);
    }
    // Engram ids (ref: engram_hash) and rows
    {
        const int64_t ng = G.engram_ngram, nh = G.engram_heads_per_ngram, cols = G.engram_cols(), dim = G.engram_dim;
        const auto & map = G.engram_token_map;
        auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= (int64_t) map.size() ? kDead : map[(size_t) t]; };
        const int64_t pad = map[(size_t) G.engram_pad];
        std::vector<uint64_t> toks((size_t) ng);
        for (size_t e = 0; e < G.engram_layers.size(); ++e) {
            const int64_t * mult = &G.engram_mult[e * (size_t) ng];
            const int64_t * offs = &G.engram_offsets[e * (size_t) cols];
            const int64_t * prim = &G.engram_buckets[e * (size_t) cols];
            std::vector<float> rows((size_t) (dim * cols * n)), valid((size_t) (cols * n));
            for (int i = 0; i < n; ++i) {
                const int64_t p = pos0 + i;
                int64_t c0 = cid(tids[i]);
                const bool dead = c0 == kDead;
                toks[0] = (uint64_t) (dead ? pad : c0);
                for (int64_t s = 1; s < ng; ++s) {
                    const int64_t cs = p - s < 0 ? kDead : cid(im.hist[(size_t) (p - s)]);
                    toks[(size_t) s] = (uint64_t) (cs == kDead ? pad : cs);
                }
                uint64_t rolling = toks[0] * (uint64_t) mult[0];
                for (int64_t o = 1; o < ng; ++o) {
                    rolling ^= toks[(size_t) o] * (uint64_t) mult[o];
                    for (int64_t h = 0; h < nh; ++h) {
                        const int64_t col = (o - 1) * nh + h;
                        const uint64_t id = rolling % (uint64_t) prim[col] + (uint64_t) offs[col];
                        const uint8_t * src = im.engram_tab + id * (uint64_t) dim;
                        float * dst = &rows[(size_t) ((i * cols + col) * dim)];
                        for (int64_t d = 0; d < dim; ++d) dst[d] = e4m3_to_f32(src[d]);
                        valid[(size_t) (i * cols + col)] = dead ? 0.0f : 1.0f;
                    }
                }
            }
            ggml_backend_tensor_set(im.e_rows[e], rows.data(), 0, rows.size() * 4);
            ggml_backend_tensor_set(im.e_valid[e], valid.data(), 0, valid.size() * 4);
        }
    }
    return true;
}

bool Hy4Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, bool * dense) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    const Layer & ly = im.L[(size_t) il];
    const int n = im.n;
    // DSA layer with more visible pools than the selection keeps: a pre-graph maintains the k-pool cache and picks the
    // top pools; the attention mask is then those pools' members + the current group (ref: tail_idxs, select_tail)
    std::vector<float> sel;
    const int64_t kp = G.idx_kpool, K = G.idx_topk / kp - (G.idx_select_tail ? 1 : 0);
    if (G.indexed(il) && (im.pos0 + n - 1) / kp > K) {
        Impl::Graph pg;
        im.new_graph(pg);
        ggml_context * c = pg.ctx;
        ggml_tensor * x = im.x;
        const int64_t e = G.engram_slot(il);
        if (e >= 0) x = im.engram(c, ly, x, im.e_rows[(size_t) e], im.e_valid[(size_t) e]);
        ggml_tensor * post_a = nullptr;
        ggml_tensor * cur = im.rms(c, im.hc_pre(c, x, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, &post_a), ly.attn_norm);
        ggml_tensor * qr = im.rms(c, im.mm(c, ly.wq_a, cur), ly.q_a_norm);
        im.kpool_maintain(pg, il, cur);
        const int64_t n_pool = (im.pos0 + n - 1) / kp;   // the most any token of the pass can see
        ggml_tensor * top = im.kpool_score(pg, il, cur, qr, n_pool, K);
        if (!im.run(pg)) return false;
        std::vector<int32_t> tk((size_t) (K * n));
        ggml_backend_tensor_get(top, tk.data(), 0, tk.size() * 4);
        const int64_t n_kv = im.pos0 + n;
        sel.assign((size_t) (n_kv * n), -INFINITY);
        for (int t = 0; t < n; ++t) {
            const int64_t p = im.pos0 + t, vis = p / kp;
            float * row = &sel[(size_t) (t * n_kv)];
            if (vis <= K) {
                for (int64_t k = 0; k <= p; ++k) row[k] = 0.0f;
                continue;
            }
            for (int64_t j = 0; j < K; ++j) {
                const int64_t q = tk[(size_t) (t * K + j)];
                if (q < 0 || q >= vis) continue;   // never a pool the token cannot see (they carry -inf scores)
                for (int64_t k = q * kp; k < q * kp + kp; ++k) row[k] = 0.0f;
            }
            for (int64_t k = vis * kp; k <= p; ++k) row[k] = 0.0f;   // the tail: the current group up to the token
        }
        if (const char * dd = std::getenv("HY4_SEL_DUMP")) {   // gate tap: the selection mask [n_kv, n] of this pass
            const std::string f = std::string(dd) + "/sel-" + std::to_string(il) + "-p" + std::to_string(im.pos0) + ".f32";
            if (std::FILE * fp = std::fopen(f.c_str(), "wb")) { std::fwrite(sel.data(), 4, sel.size(), fp); std::fclose(fp); }
        }
    }

    Impl::Graph g;
    im.new_graph(g);
    ggml_context * c = g.ctx;

    ggml_tensor * x = im.x;
    const int64_t e = G.engram_slot(il);
    if (e >= 0) {
        x = im.engram(c, ly, x, im.e_rows[(size_t) e], im.e_valid[(size_t) e]);
        im.tap(g, x, "engram_out");
    }
    ggml_tensor * post_a = nullptr;
    ggml_tensor * cur = im.rms(c, im.hc_pre(c, x, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, &post_a), ly.attn_norm);
    im.tap(g, cur, "attn_norm");
    cur = im.attention(g, il, cur, sel.empty() ? nullptr : &sel, sel.empty());
    ggml_tensor * xm = im.hc_post(c, cur, x, post_a);
    im.tap(g, xm, "hc_attn_out");
    ggml_tensor * post_f = nullptr;
    ggml_tensor * fin = im.rms(c, im.hc_pre(c, xm, ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base, &post_f), ly.ffn_norm);
    im.tap(g, fin, "ffn_norm");

    ggml_tensor * o_ids = nullptr, * o_wts = nullptr, * o_fn = nullptr;
    if (!G.moe(il)) {
        ggml_tensor * f = im.rms(c, im.swiglu_ffn(c, fin, ly.ffn_gate, ly.ffn_up, ly.ffn_down), ly.post_ffn_norm);
        im.tap(g, f, "ffn_out");
        ggml_build_forward_expand(g.gf, ggml_cpy(c, im.hc_post(c, f, xm, post_f), im.x));
    } else {
        // ref: build_latent_moe + build_moe_ffn (SIGMOID gating, exp_probs_b for selection only, norm_w, w_scale)
        ggml_tensor * logits = im.mm(c, ly.router, fin);                       // [n_expert, n]
        ggml_tensor * probs = ggml_sigmoid(c, logits);
        ggml_tensor * sel = ly.probs_b ? ggml_add(c, probs, ly.probs_b) : probs;
        o_ids = ggml_top_k(c, sel, (int) G.n_used);                            // [n_used, n] I32
        ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, G.n_expert, n), o_ids);   // [1, n_used, n]
        w = ggml_reshape_2d(c, w, G.n_used, n);
        if (G.weights_norm) w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        if (G.weights_scale != 0.0 && G.weights_scale != 1.0) w = ggml_scale(c, w, (float) G.weights_scale);
        o_wts = w;
        o_fn = im.mm(c, ly.routed_down, fin);                                  // [n_latent, n]
        ggml_build_forward_expand(g.gf, ggml_cpy(c, im.swiglu_ffn(c, fin, ly.sh_gate, ly.sh_up, ly.sh_down), im.shexp));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, xm, im.x_mid));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, post_f, im.post_ffn));
        for (ggml_tensor * t : { o_ids, o_wts, o_fn }) { ggml_set_output(t); ggml_build_forward_expand(g.gf, t); }
    }
    if (!im.run(g)) return false;
    *dense = !G.moe(il);
    if (*dense) return true;
    ggml_backend_tensor_get(o_ids, ids, 0, (size_t) (n * G.n_used) * 4);
    ggml_backend_tensor_get(o_wts, wts, 0, (size_t) (n * G.n_used) * 4);
    im.fn_host.resize((size_t) (n * G.n_latent));
    ggml_backend_tensor_get(o_fn, im.fn_host.data(), 0, im.fn_host.size() * 4);
    *fn_host = im.fn_host.data();
    return true;
}

bool Hy4Dense::finish(int il, const float * routed) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    const Layer & ly = im.L[(size_t) il];
    Impl::Graph g;
    im.new_graph(g);
    ggml_context * c = g.ctx;
    ggml_tensor * r = im.input(g, GGML_TYPE_F32, G.n_latent, im.n, routed, (size_t) (G.n_latent * im.n) * 4);
    ggml_tensor * f = ggml_add(c, im.mm(c, ly.routed_up, r), im.shexp);
    f = im.rms(c, f, ly.post_ffn_norm);
    im.tap(g, f, "ffn_out");
    ggml_build_forward_expand(g.gf, ggml_cpy(c, im.hc_post(c, f, im.x_mid, im.post_ffn), im.x));
    return im.run(g);
}

bool Hy4Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    if (row0 < 0 || nrows <= 0 || row0 + nrows > im.n) return im.fail("logits rows out of range");
    Impl::Graph g;
    im.new_graph(g);
    ggml_context * c = g.ctx;
    ggml_tensor * x = ggml_view_3d(c, im.x, G.n_embd, G.hc, nrows, im.x->nb[1], im.x->nb[2], (size_t) row0 * im.x->nb[2]);
    x = ggml_cont(c, x);
    // ref: build_hc_head
    ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, G.hc * G.n_embd, nrows), (float) G.rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(c, im.head_fn, flat);
    ggml_tensor * pre = ggml_sigmoid(c, ggml_add(c, ggml_mul(c, mixes, im.head_scale), im.head_base));
    pre = ggml_scale_bias(c, pre, 1.0f, (float) G.hc_eps);
    ggml_tensor * h = im.rms(c, im.hc_reduce(c, x, pre), im.out_norm);
    im.tap(g, h, "result_norm");
    ggml_tensor * lg = im.mm(c, im.output, h);
    ggml_set_output(lg);
    ggml_build_forward_expand(g.gf, lg);
    if (!im.run(g)) return false;
    ggml_backend_tensor_get(lg, out, 0, (size_t) (nrows * G.vocab) * 4);
    return true;
}

bool Hy4Dense::read_state(std::vector<float> & x) {
    Impl & im = *p_;
    x.resize((size_t) ggml_nelements(im.x));
    ggml_backend_tensor_get(im.x, x.data(), 0, x.size() * 4);
    return true;
}

bool Hy4Dense::write_state(const float * x) {
    Impl & im = *p_;
    ggml_backend_tensor_set(im.x, x, 0, ggml_nbytes(im.x));
    return true;
}

// ================================ LoRA ================================

bool Hy4Dense::lora_add(const std::vector<std::string> & names, int rank, float alpha, uint64_t seed, std::string & err) {
    Impl & im = *p_;
    if (im.lctx) { err = "lora_add: adapters are already set"; return false; }
    if (rank <= 0 || names.empty()) { err = "lora_add: rank and names required"; return false; }
    ggml_init_params ip = { ggml_tensor_overhead() * (2 * names.size() + 8), nullptr, true };
    im.lctx = ggml_init(ip);
    for (const auto & nm : names) {
        ggml_tensor * W = ggml_get_tensor(im.wctx, nm.c_str());
        if (!W || ggml_n_dims(W) != 2) { err = "lora_add: no 2-D dense weight " + nm; return false; }
        Impl::Lora l;
        l.name = nm;
        l.W = W;
        l.A = ggml_new_tensor_2d(im.lctx, GGML_TYPE_F32, W->ne[0], rank);
        l.B = ggml_new_tensor_2d(im.lctx, GGML_TYPE_F32, rank, W->ne[1]);
        ggml_set_name(l.A, (nm + ".lora_a").c_str());
        ggml_set_name(l.B, (nm + ".lora_b").c_str());
        l.s = alpha / (float) rank;
        l.layer = nm.rfind("blk.", 0) == 0 ? std::atoi(nm.c_str() + 4) : -1;
        im.lora_of[W] = (int) im.lora.size();
        im.lora.push_back(l);
    }
    im.lbuf = ggml_backend_alloc_ctx_tensors(im.lctx, im.be);
    if (!im.lbuf) { err = "lora_add: cannot allocate the adapters"; return false; }
    // A ~ U(-1/sqrt(in), 1/sqrt(in)), B = 0 (the usual LoRA start: the adapted model starts as the base)
    uint64_t st = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto u01 = [&]() { st = st * 6364136223846793005ull + 1442695040888963407ull; return (double) (st >> 11) / 9007199254740992.0; };
    for (auto & l : im.lora) {
        const double b = 1.0 / std::sqrt((double) l.A->ne[0]);
        std::vector<float> a((size_t) ggml_nelements(l.A)), z((size_t) ggml_nelements(l.B), 0.0f);
        for (auto & v : a) v = (float) ((u01() * 2.0 - 1.0) * b);
        ggml_backend_tensor_set(l.A, a.data(), 0, a.size() * 4);
        ggml_backend_tensor_set(l.B, z.data(), 0, z.size() * 4);
    }
    return true;
}
int Hy4Dense::lora_count() const { return (int) p_->lora.size(); }
const std::string & Hy4Dense::lora_name(int i) const { return p_->lora[(size_t) i].name; }
int Hy4Dense::lora_layer(int i) const { return p_->lora[(size_t) i].layer; }
void Hy4Dense::lora_shape(int i, int64_t & n_in, int64_t & n_out, int64_t & rank) const {
    const auto & l = p_->lora[(size_t) i];
    n_in = l.A->ne[0];
    rank = l.A->ne[1];
    n_out = l.B->ne[1];
}
void Hy4Dense::lora_get(int i, std::vector<float> & A, std::vector<float> & B) const {
    const auto & l = p_->lora[(size_t) i];
    A.resize((size_t) ggml_nelements(l.A));
    B.resize((size_t) ggml_nelements(l.B));
    ggml_backend_tensor_get(l.A, A.data(), 0, A.size() * 4);
    ggml_backend_tensor_get(l.B, B.data(), 0, B.size() * 4);
}
void Hy4Dense::lora_set(int i, const float * A, const float * B) {
    const auto & l = p_->lora[(size_t) i];
    ggml_backend_tensor_set(l.A, A, 0, ggml_nbytes(l.A));
    ggml_backend_tensor_set(l.B, B, 0, ggml_nbytes(l.B));
}

// ================================ training ================================

namespace {

// one training graph: a no-alloc context, every tensor allocated at once on the backend (no reuse: a training graph
// is built per step, its size is the step's), host inputs uploaded after the allocation
struct TrainGraph {
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> host;   // owned copies
    std::vector<std::pair<ggml_tensor *, const void *>> ptr;            // borrowed (valid until compute)
    ~TrainGraph() { if (buf) ggml_backend_buffer_free(buf); if (c) ggml_free(c); }
    void init(size_t n_tensors, size_t n_nodes) {
        ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_nodes, true), nullptr, true };
        c = ggml_init(ip);
        gf = ggml_new_graph_custom(c, n_nodes, true);
    }
    template <class T> ggml_tensor * input(ggml_type t, int64_t n0, int64_t n1, int64_t n2, const T * d, size_t count) {
        ggml_tensor * x = ggml_new_tensor_3d(c, t, n0, n1, n2);
        host.emplace_back(x, std::vector<uint8_t>((const uint8_t *) d, (const uint8_t *) d + count * sizeof(T)));
        return x;
    }
    bool run(ggml_backend_t be, ggml_tensor * loss) {
        ggml_set_loss(loss);
        ggml_build_forward_expand(gf, loss);
        ggml_build_backward_expand(c, gf, nullptr);
        buf = ggml_backend_alloc_ctx_tensors(c, be);
        if (!buf) return false;
        for (auto & h : host) ggml_backend_tensor_set(h.first, h.second.data(), 0, h.second.size());
        for (auto & h : ptr) ggml_backend_tensor_set(h.first, h.second, 0, ggml_nbytes(h.first));
        ggml_graph_reset(gf);
        return ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    }
};

void add_grad(ggml_cgraph * gf, ggml_tensor * t, std::vector<float> & acc) {
    ggml_tensor * g = ggml_graph_get_grad(gf, t);
    if (!g) return;   // the adapter is not on this graph's path
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(g, v.data(), 0, v.size() * 4);
    acc.resize(v.size(), 0.0f);
    for (size_t i = 0; i < v.size(); ++i) acc[i] += v[i];
}

}  // namespace

bool Hy4Dense::train_layer(int il, const float * x_in, const float * g_out, const int * ids, float * g_in,
                           std::vector<std::vector<float>> & gA, std::vector<std::vector<float>> & gB,
                           const float * expect_out, double * out_rel_err) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    const Layer & ly = im.L[(size_t) il];
    const int64_t n = im.n, D = G.n_embd, HC = G.hc;
    if (n <= 0 || im.pos0 != 0) return im.fail("train_layer: begin() the whole sequence first (pos0 0)");
    if (n > kpool_exact_limit()) return im.fail("train_layer: the sequence is longer than the exact k-pool limit");
    const bool moe = G.moe(il);
    // the routing of the forward: (token, k) pairs grouped by expert
    std::vector<std::vector<int32_t>> tok, pair;
    std::vector<int32_t> flat;
    if (moe) {
        tok.assign((size_t) G.n_expert, {});
        pair.assign((size_t) G.n_expert, {});
        flat.resize((size_t) (n * G.n_used));
        for (int64_t t = 0; t < n; ++t)
            for (int64_t k = 0; k < G.n_used; ++k) {
                const int e = ids[t * G.n_used + k];
                if (e < 0 || e >= G.n_expert) return im.fail("train_layer: routing id out of range");
                flat[(size_t) (t * G.n_used + k)] = (int32_t) (t * G.n_expert + e);
                tok[(size_t) e].push_back((int32_t) t);
                pair[(size_t) e].push_back((int32_t) (t * G.n_used + k));
            }
    }
    int64_t n_used_e = 0;
    for (const auto & v : tok) n_used_e += !v.empty();

    TrainGraph tg;
    tg.init((size_t) (4096 + 64 * n_used_e), (size_t) (32768 + 96 * n_used_e));
    ggml_context * c = tg.c;
    ggml_tensor * x = tg.input(GGML_TYPE_F32, D, HC, n, x_in, (size_t) (D * HC * n));
    ggml_set_param(x);
    ggml_tensor * gout = tg.input(GGML_TYPE_F32, D, HC, n, g_out, (size_t) (D * HC * n));
    for (auto & l : im.lora) if (l.layer == il) { ggml_set_param(l.A); ggml_set_param(l.B); }
    std::vector<float> m((size_t) (n * n));
    for (int64_t t = 0; t < n; ++t)
        for (int64_t k = 0; k < n; ++k)
            m[(size_t) (t * n + k)] = k <= t && !(G.swa(il) && t - k >= G.n_swa) ? 0.0f : -INFINITY;
    ggml_tensor * mask = tg.input(GGML_TYPE_F32, n, n, 1, m.data(), m.size());
    mask = ggml_reshape_2d(c, mask, n, n);

    ggml_tensor * xx = x;
    const int64_t e = G.engram_slot(il);
    if (e >= 0) xx = im.engram(c, ly, x, im.e_rows[(size_t) e], im.e_valid[(size_t) e]);
    ggml_tensor * post_a = nullptr;
    ggml_tensor * cur = im.rms(c, im.hc_pre(c, xx, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, &post_a), ly.attn_norm);
    cur = im.attention_train(c, il, cur, mask);
    ggml_tensor * xm = im.hc_post(c, cur, xx, post_a);
    ggml_tensor * post_f = nullptr;
    ggml_tensor * fin = im.rms(c, im.hc_pre(c, xm, ly.hc_ffn_fn, ly.hc_ffn_scale, ly.hc_ffn_base, &post_f), ly.ffn_norm);
    ggml_tensor * f = nullptr;
    if (!moe) {
        f = im.rms(c, im.swiglu_ffn(c, fin, ly.ffn_gate, ly.ffn_up, ly.ffn_down), ly.post_ffn_norm);
    } else {
        ggml_tensor * probs = ggml_sigmoid(c, im.mm(c, ly.router, fin));                           // [E, n]
        ggml_tensor * idf = ggml_reshape_1d(c, tg.input(GGML_TYPE_I32, n * G.n_used, 1, 1, flat.data(), flat.size()), n * G.n_used);
        ggml_tensor * w = ggml_get_rows(c, ggml_reshape_2d(c, probs, 1, G.n_expert * n), idf);   // [1, K*n]
        w = ggml_reshape_2d(c, w, G.n_used, n);
        if (G.weights_norm) w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        if (G.weights_scale != 0.0 && G.weights_scale != 1.0) w = ggml_scale(c, w, (float) G.weights_scale);
        ggml_tensor * wf = ggml_reshape_2d(c, ggml_cont(c, w), 1, n * G.n_used);
        ggml_tensor * fn = im.mm(c, ly.routed_down, fin);                                         // [L, n]
        // this layer's used experts: their GGUF slices, uploaded for the step
        const GgufModel & M = *im.model;
        const char * role[3] = { "gate", "up", "down" };
        const uint8_t * base[3];
        ggml_type type[3];
        int64_t ne0[3], ne1[3];
        size_t slice[3];
        for (int r = 0; r < 3; ++r) {
            size_t sh = 0;
            const std::string nm = "blk." + std::to_string(il) + ".ffn_" + role[r] + "_exps.weight";
            const TensorInfo * ti = M.find(nm, &sh);
            if (!ti) return im.fail("train_layer: missing " + nm);
            base[r] = M.shard(sh).tensor_data(*ti);
            type[r] = (ggml_type) ti->type;
            ne0[r] = (int64_t) ti->shape[0];
            ne1[r] = (int64_t) ti->shape[1];
            slice[r] = ggml_row_size(type[r], ne0[r]) * (size_t) ne1[r];
        }
        const double lim = (size_t) il < G.swiglu_clamp.size() ? G.swiglu_clamp[(size_t) il] : 0.0;
        ggml_tensor * routed = nullptr;
        for (int64_t ex = 0; ex < G.n_expert; ++ex) {
            const int64_t ne = (int64_t) tok[(size_t) ex].size();
            if (ne == 0) continue;
            ggml_tensor * W[3];
            for (int r = 0; r < 3; ++r) {
                W[r] = ggml_new_tensor_2d(c, type[r], ne0[r], ne1[r]);
                tg.ptr.emplace_back(W[r], base[r] + (size_t) ex * slice[r]);
            }
            ggml_tensor * ie = ggml_reshape_1d(c, tg.input(GGML_TYPE_I32, ne, 1, 1, tok[(size_t) ex].data(), (size_t) ne), ne);
            ggml_tensor * pe = ggml_reshape_1d(c, tg.input(GGML_TYPE_I32, ne, 1, 1, pair[(size_t) ex].data(), (size_t) ne), ne);
            ggml_tensor * xe = ggml_get_rows(c, fn, ie);                                          // [L, ne]
            ggml_tensor * gt = ggml_mul_mat(c, W[0], xe), * up = ggml_mul_mat(c, W[1], xe);
            ggml_tensor * a = lim > 1e-6 ? ggml_swiglu_clamp(c, gt, up, (float) lim) : ggml_swiglu_split(c, gt, up);
            ggml_tensor * ye = ggml_mul(c, ggml_mul_mat(c, W[2], a), ggml_reshape_2d(c, ggml_get_rows(c, wf, pe), 1, ne));
            ggml_tensor * sc = ggml_get_rows_back(c, ye, ie, fn);
            routed = routed ? ggml_add(c, routed, sc) : sc;
        }
        f = ggml_add(c, im.mm(c, ly.routed_up, routed), im.swiglu_ffn(c, fin, ly.sh_gate, ly.sh_up, ly.sh_down));
        f = im.rms(c, f, ly.post_ffn_norm);
    }
    ggml_tensor * out = im.hc_post(c, f, xm, post_f);
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, out, gout));   // the surrogate: d loss / d (params) = the layer's VJP
    ggml_set_output(out);
    if (!tg.run(im.be, loss)) return im.fail("train_layer: graph allocation or compute failed");
    if (expect_out && out_rel_err) {
        std::vector<float> o((size_t) (D * HC * n));
        ggml_backend_tensor_get(out, o.data(), 0, o.size() * 4);
        double num = 0, den = 0;
        for (size_t i = 0; i < o.size(); ++i) { const double d = o[i] - expect_out[i]; num += d * d; den += (double) expect_out[i] * expect_out[i]; }
        *out_rel_err = std::sqrt(num / std::max(den, 1e-30));
    }
    ggml_backend_tensor_get(ggml_graph_get_grad(tg.gf, x), g_in, 0, (size_t) (D * HC * n) * 4);
    gA.resize(im.lora.size());
    gB.resize(im.lora.size());
    for (size_t i = 0; i < im.lora.size(); ++i)
        if (im.lora[i].layer == il) { add_grad(tg.gf, im.lora[i].A, gA[i]); add_grad(tg.gf, im.lora[i].B, gB[i]); }
    for (auto & l : im.lora) if (l.layer == il) { l.A->flags &= ~GGML_TENSOR_FLAG_PARAM; l.B->flags &= ~GGML_TENSOR_FLAG_PARAM; }
    return true;
}

bool Hy4Dense::train_head(const float * x_last, const int * targets, float * loss_out, float * g_x,
                          std::vector<std::vector<float>> & gA, std::vector<std::vector<float>> & gB) {
    Impl & im = *p_;
    const Hy4Geometry & G = im.G;
    const int64_t n = im.n, D = G.n_embd, HC = G.hc;
    std::vector<int32_t> at;
    for (int64_t t = 0; t < n; ++t) if (targets[t] >= 0) at.push_back((int32_t) t);
    if (at.empty()) return im.fail("train_head: no target positions");
    const int64_t mcount = (int64_t) at.size();
    std::vector<float> lab((size_t) (G.vocab * mcount), 0.0f);
    for (int64_t j = 0; j < mcount; ++j) {
        const int tg_id = targets[at[(size_t) j]];
        if (tg_id >= G.vocab) return im.fail("train_head: target id out of range");
        lab[(size_t) (j * G.vocab + tg_id)] = 1.0f;
    }
    TrainGraph tg;
    tg.init(4096, 16384);
    ggml_context * c = tg.c;
    ggml_tensor * x = tg.input(GGML_TYPE_F32, D, HC, n, x_last, (size_t) (D * HC * n));
    ggml_set_param(x);
    for (auto & l : im.lora) if (l.layer < 0) { ggml_set_param(l.A); ggml_set_param(l.B); }
    // ref: build_hc_head (as logits())
    ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, n), (float) G.rms_eps);
    ggml_tensor * pre = ggml_sigmoid(c, ggml_add(c, ggml_mul(c, ggml_mul_mat(c, im.head_fn, flat), im.head_scale), im.head_base));
    pre = ggml_scale_bias(c, pre, 1.0f, (float) G.hc_eps);
    ggml_tensor * h = im.rms(c, im.hc_reduce(c, x, pre), im.out_norm);
    ggml_tensor * rows = ggml_reshape_1d(c, tg.input(GGML_TYPE_I32, mcount, 1, 1, at.data(), at.size()), mcount);
    ggml_tensor * lg = im.mm(c, im.output, ggml_get_rows(c, h, rows));                            // [V, m]
    ggml_tensor * labels = ggml_reshape_2d(c, tg.input(GGML_TYPE_F32, G.vocab, mcount, 1, lab.data(), lab.size()), G.vocab, mcount);
    ggml_tensor * loss = ggml_cross_entropy_loss(c, lg, labels);
    if (!tg.run(im.be, loss)) return im.fail("train_head: graph allocation or compute failed");
    ggml_backend_tensor_get(loss, loss_out, 0, 4);
    ggml_backend_tensor_get(ggml_graph_get_grad(tg.gf, x), g_x, 0, (size_t) (D * HC * n) * 4);
    gA.resize(im.lora.size());
    gB.resize(im.lora.size());
    for (size_t i = 0; i < im.lora.size(); ++i)
        if (im.lora[i].layer < 0) { add_grad(tg.gf, im.lora[i].A, gA[i]); add_grad(tg.gf, im.lora[i].B, gB[i]); }
    for (auto & l : im.lora) if (l.layer < 0) { l.A->flags &= ~GGML_TENSOR_FLAG_PARAM; l.B->flags &= ~GGML_TENSOR_FLAG_PARAM; }
    return true;
}

}  // namespace strata::hy4
