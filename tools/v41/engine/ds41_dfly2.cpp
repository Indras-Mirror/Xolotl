// tools/v41/engine/ds41_dfly2.cpp - see ds41_dfly2.hpp.  The block code follows ds41_dspark.cpp (same DS4.1 layers).
#include "ds41_embd.hpp"
#include "ds41_dfly2.hpp"

#include "ds41_geometry.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"

#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>

using namespace strata;
using namespace strata::v41;

namespace {
constexpr float NEG_INF = -INFINITY;
constexpr int64_t WIN = 128;    // the target window the drafter reads (positions a-128 .. a-1)
constexpr int64_t TOP = 512;    // indexer picks per layer
constexpr int64_t NCTX = TOP + WIN;   // 640 context keys per query
constexpr int64_t MP = 768;     // keys per query after padding (ggml-cuda flash-attn d 512 wants % 256): 640 + 6 + pad
constexpr int64_t MC = 768;     // the correction's context keys after padding (640 + pad)
constexpr int64_t NSLOT = 16;   // block_rel_bias / block_gate_bias columns (MAX_BLOCK_SLOTS)
}

struct Ds41DFly2::Impl {
    Ds41DFly2Config cfg;
    std::string err;
    std::unique_ptr<GgufModel> side, main;
    Ds41Geometry g;
    ggml_backend_t be = nullptr;
    int64_t D = 0, HC = 0, DH = 0, NH = 0, OG = 0, OL = 0, OGD = 0, NE = 0, NU = 3, V = 0, BS = 6;
    int mask_id = 128799;
    std::vector<int> tl { 28, 32, 36 };
    // weights
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    const uint8_t * embd = nullptr;
    int embd_type = GGML_TYPE_F16;
    size_t wbytes = 0;
    std::vector<float> rel_bias, gate_bias, block_sink;   // host copies (F32): [NH][NSLOT], [NH][NSLOT], [NH]
    // inputs + persistent state
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * i_emb = nullptr, * i_xs = nullptr, * i_pre1 = nullptr, * i_bpos = nullptr, * i_pre = nullptr;
    ggml_tensor * i_cidx[3] = {}, * i_mask[3] = {}, * i_widx = nullptr, * i_cmask = nullptr;
    ggml_tensor * i_eprev = nullptr;   // [D, 2]: E(x_{a-1}) (slot 0's correction key), E(prev_j) of the current slot
    ggml_tensor * i_rb = nullptr, * i_gb = nullptr;   // [BS + 1, NH] block logit bias (+ sink row), [NH] gate bias
    ggml_tensor * s_dh = nullptr, * s_bk = nullptr;   // [D, BS] draft hidden, [DH, BS] correction block keys
    // graphs: 0 = blocks + slot 1, j = 2..5 = slot j's correction; each its own allocator
    struct G { ggml_context * c = nullptr; ggml_cgraph * gf = nullptr; ggml_gallocr_t ga = nullptr; ggml_tensor * id = nullptr, * p = nullptr; };
    std::map<int, G> graphs;
    const void * bound[6] = {};   // the external tensors the cached graphs were built on
    double ms = 0, main_ms = 0;

    ~Impl() {
        for (auto & kv : graphs) { ggml_free(kv.second.c); if (kv.second.ga) ggml_gallocr_free(kv.second.ga); }
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
    ggml_tensor * W(const std::string & n) const {
        auto it = w.find(n);
        if (it == w.end()) { std::fprintf(stderr, "[ds41_dfly2] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    ggml_tensor * LW(int b, const std::string & n) const { return W("layers." + std::to_string(b) + "." + n); }
    ggml_tensor * CW(const std::string & n) const { return W("hidden_correction." + n); }
    bool load(std::string & e);
    G & graph(int j, const Ds41DFly2Ctx & c);
    void drop_graphs() {
        for (auto & kv : graphs) { ggml_free(kv.second.c); if (kv.second.ga) ggml_gallocr_free(kv.second.ga); }
        graphs.clear();
    }
};

Ds41DFly2::Ds41DFly2() : p_(new Impl) {}
Ds41DFly2::~Ds41DFly2() = default;
const std::vector<int> & Ds41DFly2::target_layers() const { return p_->tl; }
int Ds41DFly2::n_draft() const { return (int) p_->BS - 1; }
double Ds41DFly2::last_ms() const { return p_->ms; }
double Ds41DFly2::last_main_ms() const { return p_->main_ms; }
size_t Ds41DFly2::weight_bytes() const { return p_->wbytes; }
const std::string & Ds41DFly2::last_error() const { return p_->err; }

bool Ds41DFly2::Impl::load(std::string & e) {
    struct Src { const GgufModel * m; const TensorInfo * ti; std::string name; };
    std::vector<Src> src;
    for (size_t sh = 0; sh < side->size(); ++sh)
        for (const auto & ti : side->shard(sh).tensors()) src.push_back({ side.get(), &ti, ti.name });
    const TensorInfo * out = main->find("output.weight");
    const TensorInfo * onorm = main->find("output_norm.weight");
    const TensorInfo * te = main->find("token_embd.weight");
    if (!out || !onorm || !te) { e = "the trunk GGUF lacks output.weight / output_norm.weight / token_embd.weight"; return false; }
    if (!ds41_embd_type_ok((int) te->type)) { e = "token_embd type has no row reader"; return false; }
    embd_type = (int) te->type;
    if (!cfg.shared_head) src.push_back({ main.get(), out, "output.weight" });
    if (!cfg.shared_out_norm) src.push_back({ main.get(), onorm, "output_norm.weight" });
    // STRATA_DFLY2_EXP_TYPE=<ggml type name> (q4_K, q3_K, q2_K, ...): the routed experts (MXFP4, 7.2 GB) requantized on
    // load.  Drafts are verified by the trunk: precision moves only acceptance.
    ggml_type xt = GGML_TYPE_COUNT;
    if (const char * x = std::getenv("STRATA_DFLY2_EXP_TYPE")) {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * nm = ggml_type_name((ggml_type) i);
            if (nm && std::strcmp(nm, x) == 0) xt = (ggml_type) i;
        }
        if (xt == GGML_TYPE_COUNT) { e = std::string("STRATA_DFLY2_EXP_TYPE: unknown type ") + x; return false; }
        if (ggml_quantize_requires_imatrix(xt)) { e = std::string("STRATA_DFLY2_EXP_TYPE ") + x + " needs an imatrix"; return false; }
        ggml_quantize_init(xt);
    }
    auto is_exp = [](const std::string & n) { return n.find(".ffn.experts.") != std::string::npos; };
    ggml_init_params ip = { (src.size() + 8) * ggml_tensor_overhead(), nullptr, true };
    wctx = ggml_init(ip);
    for (auto & s : src) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < s.ti->shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) s.ti->shape[d];
        const ggml_type ty = (xt != GGML_TYPE_COUNT && is_exp(s.name)) ? xt : (ggml_type) s.ti->type;
        if (ty != (ggml_type) s.ti->type && ne[0] % ggml_blck_size(ty)) { e = s.name + ": row not a multiple of the block"; return false; }
        ggml_tensor * t = ggml_new_tensor(wctx, ty, (int) s.ti->shape.size(), ne);
        ggml_set_name(t, s.name.c_str());
        w[s.name] = t;
    }
    wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_get_default_buffer_type(be));
    if (!wbuf) { e = "cannot allocate the DFly2 weights on the backend"; return false; }
    for (auto & s : src) {
        size_t sh = 0;
        s.m->find(s.ti->name, &sh);
        ggml_tensor * t = w[s.name];
        const void * data = s.m->shard(sh).tensor_data(*s.ti);
        if (t->type == (ggml_type) s.ti->type) {
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        } else {
            const ggml_type st = (ggml_type) s.ti->type;
            const auto * tr = ggml_get_type_traits(st);
            if (!tr->to_float) { e = s.name + ": no dequantizer for " + ggml_type_name(st); return false; }
            const int64_t n0 = t->ne[0], rows = ggml_nrows(t);
            const size_t srow = ggml_row_size(st, n0), drow = ggml_row_size(t->type, n0);
            std::vector<uint8_t> o((size_t) rows * drow);
            const int nth = std::max(1u, std::thread::hardware_concurrency());
            const int64_t per = (rows + nth - 1) / nth;
            std::vector<std::thread> th;
            for (int i = 0; i < nth; ++i)
                th.emplace_back([&, i]() {
                    const int64_t r0 = i * per, r1 = std::min(rows, r0 + per);
                    std::vector<float> f((size_t) 256 * n0);
                    for (int64_t r = r0; r < r1; r += 256) {
                        const int64_t nr = std::min<int64_t>(256, r1 - r);
                        for (int64_t k = 0; k < nr; ++k)
                            tr->to_float((const uint8_t *) data + (size_t) (r + k) * srow, f.data() + (size_t) k * n0, n0);
                        ggml_quantize_chunk(t->type, f.data(), o.data() + (size_t) r * drow, 0, nr, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, o.data(), 0, o.size());
        }
        wbytes += ggml_nbytes(t);
    }
    auto share = [&](ggml_tensor * h, const TensorInfo * ti, const char * nm) -> bool {
        if (h->ne[0] != (int64_t) ti->shape[0] || (ti->shape.size() > 1 && h->ne[1] != (int64_t) ti->shape[1])) {
            e = std::string(nm) + ": shared tensor shape differs"; return false;
        }
        if (!h->buffer || ggml_backend_buffer_get_type(h->buffer) != ggml_backend_get_default_buffer_type(be)) {
            e = std::string(nm) + ": shared tensor is not on the DFly2 backend"; return false;
        }
        w[nm] = h;
        return true;
    };
    if (cfg.shared_head && !share(cfg.shared_head, out, "output.weight")) return false;
    if (cfg.shared_out_norm && !share(cfg.shared_out_norm, onorm, "output_norm.weight")) return false;
    size_t se = 0;
    main->find("token_embd.weight", &se);
    embd = main->shard(se).tensor_data(*te);
    // the gated block read's small tables on the host (the per-slot bias rows are uploaded per graph)
    auto host_f32 = [&](const char * nm, std::vector<float> & v, size_t n) -> bool {
        size_t sh = 0;
        const TensorInfo * ti = side->find(nm, &sh);
        if (!ti || ti->type != GGML_TYPE_F32) { e = std::string("DFly2 sidecar: missing F32 ") + nm; return false; }
        int64_t el = 1;
        for (auto x : ti->shape) el *= (int64_t) x;
        if ((size_t) el != n) { e = std::string("DFly2 sidecar: bad size of ") + nm; return false; }
        const float * d = (const float *) side->shard(sh).tensor_data(*ti);
        v.assign(d, d + n);
        return true;
    };
    if (!host_f32("hidden_correction.attention.block_rel_bias", rel_bias, (size_t) (NH * NSLOT))) return false;
    if (!host_f32("hidden_correction.attention.block_gate_bias", gate_bias, (size_t) (NH * NSLOT))) return false;
    if (!host_f32("hidden_correction.attention.block_sink", block_sink, (size_t) NH)) return false;
    return true;
}

bool Ds41DFly2::init(const Ds41DFly2Config & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    im.be = cfg.backend;
    if (!im.be) { err = "Ds41DFly2: a backend is required"; return false; }
    try {
        im.side = std::make_unique<GgufModel>(GgufModel::open(cfg.sidecar));
        im.main = std::make_unique<GgufModel>(GgufModel::open(cfg.main_gguf));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.main, im.g)).empty()) return false;
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.NH = G.n_head; im.OG = G.o_groups; im.OL = G.o_lora;
    im.OGD = G.n_head * G.d_head / G.o_groups; im.V = G.vocab;
    if (const MetaValue * v = im.side->meta().get("dfly2.block_size")) im.BS = (int64_t) v->u;
    if (const MetaValue * v = im.side->meta().get("dfly2.mask_token_id")) im.mask_id = (int) v->u;
    if (const MetaValue * v = im.side->meta().get("dfly2.n_expert_used")) im.NU = (int64_t) v->u;
    if (im.BS != 6) { err = "DFly2: block size " + std::to_string(im.BS) + " (this port is built for 6)"; return false; }
    if (!im.load(err)) return false;
    im.NE = im.LW(0, "ffn.experts.w1.weight")->ne[2];
    if (im.W("mtp_fusion.eh_proj.weight")->ne[0] != 2 * im.D) { err = "DFly2: mtp_fusion.eh_proj is not [2*n_embd, n_embd]"; return false; }
    ggml_init_params sp = { 48 * ggml_tensor_overhead(), nullptr, true };
    im.sctx = ggml_init(sp);
    auto T1 = [&](ggml_type t, int64_t a) { return ggml_new_tensor_1d(im.sctx, t, a); };
    auto T2 = [&](ggml_type t, int64_t a, int64_t b) { return ggml_new_tensor_2d(im.sctx, t, a, b); };
    im.i_emb = T2(GGML_TYPE_F32, im.D, im.BS);
    im.i_xs = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, 1);
    im.i_pre1 = T2(GGML_TYPE_F32, im.HC, 1);
    im.i_bpos = T1(GGML_TYPE_I32, im.BS);
    im.i_pre = T2(GGML_TYPE_F32, im.HC, im.BS);
    for (int i = 0; i < 3; ++i) { im.i_cidx[i] = T1(GGML_TYPE_I32, TOP); im.i_mask[i] = T2(GGML_TYPE_F32, MP, im.BS); }
    im.i_widx = T1(GGML_TYPE_I32, WIN);
    im.i_cmask = T2(GGML_TYPE_F32, MC, 1);
    im.i_eprev = T2(GGML_TYPE_F32, im.D, 2);
    im.i_rb = T2(GGML_TYPE_F32, im.BS + 1, im.NH);
    im.i_gb = T2(GGML_TYPE_F32, 1, im.NH);
    im.s_dh = T2(GGML_TYPE_F32, im.D, im.BS);
    im.s_bk = T2(GGML_TYPE_F32, im.DH, im.BS);
    im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.be));
    if (!im.sbuf) { err = "cannot allocate the DFly2 state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    std::vector<float> pre((size_t) (im.HC * im.BS), 0.0f);   // the first block's collapsed input = x itself
    for (int64_t t = 0; t < im.BS; ++t) pre[(size_t) (t * im.HC)] = 1.0f;
    ggml_backend_tensor_set(im.i_pre, pre.data(), 0, pre.size() * 4);
    return true;
}

// j = 0: the blocks + slot 1's proposal; j = 2..BS-1: slot j's correction + proposal.
Ds41DFly2::Impl::G & Ds41DFly2::Impl::graph(int j, const Ds41DFly2Ctx & cx) {
    const void * b[6] = { cx.raw[0], cx.raw[1], cx.raw[2], cx.comp[0], cx.comp[1], cx.comp[2] };
    if (std::memcmp(b, bound, sizeof b) != 0) { drop_graphs(); std::memcpy(bound, b, sizeof b); }
    G & gr = graphs[j];
    if (gr.gf) return gr;
    ggml_init_params ip = { 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * c = gr.c = ggml_init(ip);
    ggml_cgraph * gf = gr.gf = ggml_new_graph_custom(c, 8192, false);
    const Ds41Geometry & Gm = g;
    const float eps = (float) Gm.rms_eps;
    auto rms_w = [&](ggml_tensor * x, ggml_tensor * wt) { return ggml_mul(c, ggml_rms_norm(c, x, eps), wt); };
    // the trunk's compressed-layer RoPE (ds41_dense.cpp rope(.., ratio != 0)): compress_rope_theta + YaRN
    auto rope = [&](ggml_tensor * a, ggml_tensor * pos, bool back) {
        const float base = (float) Gm.compress_rope_base, fscale = (float) (1.0 / Gm.rope_factor);
        const float attn = 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale));
        const int n_ctx = (int) Gm.orig_ctx, nd = (int) Gm.d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, 1.0f, attn, (float) Gm.beta_fast, (float) Gm.beta_slow)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, 1.0f, attn, (float) Gm.beta_fast, (float) Gm.beta_slow);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    };
    auto hc_mixes = [&](ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                        ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) {
        const int64_t nt = x->ne[2];
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, ggml_view_1d(c, sc, 1, 0)), ggml_view_1d(c, bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) Gm.hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, ggml_view_1d(c, sc, 1, sizeof(float))), ggml_view_1d(c, bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) Gm.hc_eps, (int32_t) Gm.hc_iters);
    };
    // the context latents of target layer tl[i]: [DH, 640] F32 = its picks' compressed rows | its window
    auto ctx_keys = [&](int i) {
        ggml_tensor * kc = ggml_get_rows(c, cx.comp[i], i_cidx[i]);
        ggml_tensor * kw = ggml_get_rows(c, cx.raw[i], i_widx);
        if (kc->type != GGML_TYPE_F32) kc = ggml_cast(c, kc, GGML_TYPE_F32);
        if (kw->type != GGML_TYPE_F32) kw = ggml_cast(c, kw, GGML_TYPE_F32);
        return ggml_concat(c, kc, kw, 1);
    };
    // grouped output (inverse RoPE on the read, wo_a per group, wo_b): o [DH, NH, n] -> [D, n]
    auto grouped = [&](ggml_tensor * o, ggml_tensor * pos, ggml_tensor * wo_a_w, ggml_tensor * wo_b_w, int64_t n) {
        o = rope(o, pos, true);
        o = ggml_permute(c, ggml_reshape_3d(c, o, OGD, OG, n), 0, 2, 1, 3);                      // [OGD, n, OG]
        ggml_tensor * wo_a = ggml_reshape_3d(c, wo_a_w, OGD, OL, OG);
        ggml_tensor * oa = ggml_permute(c, ggml_mul_mat(c, wo_a, o), 0, 2, 1, 3);                // [OL, OG, n]
        return ggml_mul_mat(c, wo_b_w, ggml_cont_2d(c, oa, OL * OG, n));
    };
    const float scale = 1.0f / std::sqrt((float) DH);
    // slot jj's correction from its draft hidden dh [D,1] and prev embedding e [D,1]; prior block keys bk_prev [DH, jj]
    // (null for none) -> the proposal id; *bk_out = this slot's block key [DH, 1]
    auto correct = [&](int jj, ggml_tensor * dh, ggml_tensor * e, ggml_tensor * bk_prev, ggml_tensor * ctx36,
                       ggml_tensor ** bk_out, ggml_tensor ** p_out) -> ggml_tensor * {
        ggml_tensor * pos = ggml_view_1d(c, i_bpos, 1, (size_t) jj * 4);
        ggml_tensor * cat = ggml_concat(c, rms_w(dh, CW("hidden_norm.weight")), rms_w(e, CW("embed_norm.weight")), 0);   // [2D, 1]
        ggml_tensor * delta = ggml_mul_mat(c, CW("down_proj.weight"),
                                           ggml_swiglu_split(c, ggml_mul_mat(c, CW("gate_proj.weight"), cat),
                                                             ggml_mul_mat(c, CW("up_proj.weight"), cat)));
        ggml_tensor * q = rms_w(ggml_mul_mat(c, CW("attention.q_in.weight"), cat), CW("attention.q_norm.weight"));
        q = rope(ggml_reshape_3d(c, ggml_mul_mat(c, CW("attention.wq_b.weight"), q), DH, NH, 1), pos, false);   // [DH, NH, 1]
        ggml_tensor * bk = rms_w(ggml_mul_mat(c, CW("attention.kv_in.weight"), cat), CW("attention.kv_norm.weight"));
        bk = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, bk, DH, 1, 1), pos, false), DH, 1);
        *bk_out = bk;
        // context read: target layer 36's 640 keys, attn_sink
        ggml_tensor * K = ggml_pad(c, ctx36, 0, (int) (MC - NCTX), 0, 0);
        ggml_tensor * k4 = ggml_cast(c, ggml_reshape_4d(c, K, DH, MC, 1, 1), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, DH, NH, 1, 1), 0, 2, 1, 3);    // [DH, 1, NH, 1]
        ggml_tensor * m4 = ggml_cast(c, ggml_reshape_4d(c, i_cmask, MC, 1, 1, 1), GGML_TYPE_F16);
        ggml_tensor * o = ggml_flash_attn_ext(c, q4, k4, k4, m4, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, CW("attention.attn_sink"));
        ggml_tensor * read = ggml_reshape_2d(c, o, DH, NH);
        // gated block read over the correction keys 0..jj (causal), rel bias + its own sink, tanh gate per head
        ggml_tensor * keys = bk_prev ? ggml_concat(c, bk_prev, bk, 1) : bk;                     // [DH, jj+1]
        const int64_t nk = keys->ne[1];
        ggml_tensor * lg = ggml_mul_mat(c, keys, ggml_reshape_2d(c, q, DH, NH));                 // [nk, NH]
        ggml_mul_mat_set_prec(lg, GGML_PREC_F32);
        lg = ggml_scale(c, lg, scale);
        ggml_tensor * rb = ggml_view_2d(c, i_rb, nk, NH, i_rb->nb[1], 0);
        ggml_tensor * sk = ggml_view_2d(c, i_rb, 1, NH, i_rb->nb[1], (size_t) BS * 4);          // sink row
        ggml_tensor * wsm = ggml_soft_max(c, ggml_concat(c, ggml_add(c, lg, rb), sk, 0));        // [nk+1, NH]
        wsm = ggml_view_2d(c, wsm, nk, NH, wsm->nb[1], 0);
        ggml_tensor * bread = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, keys)), ggml_cont(c, wsm));   // [DH, NH]
        ggml_tensor * gate = ggml_tanh(c, ggml_add(c, ggml_reshape_2d(c, ggml_mul_mat(c, CW("attention.block_gate.weight"), cat), 1, NH), i_gb));
        read = ggml_add(c, read, ggml_mul(c, bread, gate));
        ggml_tensor * out = grouped(ggml_reshape_3d(c, read, DH, NH, 1), pos, CW("attention.wo_a.weight"), CW("attention.wo_b.weight"), 1);
        ggml_tensor * h = ggml_add(c, dh, ggml_add(c, delta, out));
        ggml_tensor * logits = ggml_mul_mat(c, W("output.weight"), h);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        ggml_tensor * am = ggml_argmax(c, logits);
        // the proposal's probability (the confidence the verify-width cuts read)
        *p_out = ggml_get_rows(c, ggml_reshape_2d(c, ggml_soft_max(c, logits), 1, V), am);
        return am;
    };

    if (j >= 2) {
        ggml_tensor * dh = ggml_view_2d(c, s_dh, D, 1, s_dh->nb[1], (size_t) j * s_dh->nb[1]);
        ggml_tensor * e = ggml_view_2d(c, i_eprev, D, 1, i_eprev->nb[1], i_eprev->nb[1]);
        ggml_tensor * bkp = ggml_view_2d(c, s_bk, DH, j, s_bk->nb[1], 0);
        ggml_tensor * bk = nullptr;
        ggml_tensor * pp = nullptr;
        ggml_tensor * id = correct(j, dh, e, bkp, ctx_keys(2), &bk, &pp);
        ggml_build_forward_expand(gf, ggml_cpy(c, bk, ggml_view_2d(c, s_bk, DH, 1, s_bk->nb[1], (size_t) j * s_bk->nb[1])));
        ggml_set_output(id);
        ggml_set_output(pp);
        ggml_build_forward_expand(gf, id);
        ggml_build_forward_expand(gf, pp);
        gr.id = id;
        gr.p = pp;
        gr.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        return gr;
    }

    // ---- the block input: E([x_a, MASK x5]) + the MTP fusion on slot 0
    ggml_tensor * bpos = i_bpos;
    ggml_tensor * e0 = ggml_view_2d(c, i_emb, D, 1, i_emb->nb[1], 0);
    ggml_tensor * hfin = rms_w(ggml_dsv4_hc_pre(c, i_xs, i_pre1), W("output_norm.weight"));       // [D, 1]
    ggml_tensor * fu = ggml_mul_mat(c, W("mtp_fusion.eh_proj.weight"),
                                    ggml_concat(c, rms_w(e0, W("mtp_fusion.enorm.weight")), rms_w(hfin, W("mtp_fusion.hnorm.weight")), 0));
    ggml_tensor * x0 = ggml_add(c, e0, ggml_mul(c, fu, W("mtp_fusion.gate")));
    ggml_tensor * xin = ggml_concat(c, x0, ggml_view_2d(c, i_emb, D, BS - 1, i_emb->nb[1], i_emb->nb[1]), 1);   // [D, BS]
    ggml_tensor * x = ggml_repeat(c, ggml_reshape_3d(c, xin, D, 1, BS), ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, BS));
    ggml_tensor * pre = i_pre;
    const float lim = (float) Gm.swiglu_limit;
    ggml_tensor * ctx36 = nullptr;
    for (int b = 0; b < 3; ++b) {
        ggml_tensor * a_pre, * a_post, * a_comb;
        hc_mixes(x, LW(b, "hc_attn_fn"), LW(b, "hc_attn_scale"), LW(b, "hc_attn_base"), &a_pre, &a_post, &a_comb);
        ggml_tensor * xn = rms_w(ggml_dsv4_hc_pre(c, x, pre), LW(b, "attn_norm.weight"));
        ggml_tensor * qr = rms_w(ggml_mul_mat(c, LW(b, "attn.wq_a.weight"), xn), LW(b, "attn.q_norm.weight"));
        ggml_tensor * q = rope(ggml_reshape_3d(c, ggml_mul_mat(c, LW(b, "attn.wq_b.weight"), qr), DH, NH, BS), bpos, false);
        ggml_tensor * kvb = rms_w(ggml_mul_mat(c, LW(b, "attn.wkv.weight"), xn), LW(b, "attn.kv_norm.weight"));
        kvb = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, kvb, DH, 1, BS), bpos, false), DH, BS);
        ggml_tensor * ck = ctx_keys(b);
        if (b == 2) ctx36 = ck;
        // keys = [640 context | the 6 block latents | pad], every slot sees all (non-causal)
        ggml_tensor * K = ggml_pad(c, ggml_concat(c, ck, kvb, 1), 0, (int) (MP - NCTX - BS), 0, 0);   // [DH, MP]
        ggml_tensor * k4 = ggml_cast(c, ggml_repeat(c, ggml_reshape_4d(c, K, DH, MP, 1, 1),
                                                    ggml_new_tensor_4d(c, GGML_TYPE_F32, DH, MP, 1, BS)), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, DH, NH, 1, BS), 0, 2, 1, 3);    // [DH, 1, NH, BS]
        ggml_tensor * m4 = ggml_cast(c, ggml_reshape_4d(c, i_mask[b], MP, 1, 1, BS), GGML_TYPE_F16);
        ggml_tensor * o = ggml_flash_attn_ext(c, q4, k4, k4, m4, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, LW(b, "attn.attn_sink"));
        ggml_tensor * attn = grouped(ggml_reshape_3d(c, o, DH, NH, BS), bpos, LW(b, "attn.wo_a.weight"), LW(b, "attn.wo_b.weight"), BS);
        ggml_tensor * h = ggml_dsv4_hc_post(c, attn, x, a_post, a_comb);
        ggml_tensor * f_pre, * f_post, * f_comb;
        hc_mixes(h, LW(b, "hc_ffn_fn"), LW(b, "hc_ffn_scale"), LW(b, "hc_ffn_base"), &f_pre, &f_post, &f_comb);
        ggml_tensor * fn = rms_w(ggml_dsv4_hc_pre(c, h, a_pre), LW(b, "ffn_norm.weight"));     // [D, BS]
        ggml_tensor * rl = ggml_mul_mat(c, LW(b, "ffn.gate.weight"), fn);
        ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
        ggml_tensor * sc = ggml_sqrt(c, ggml_softplus(c, rl));                                  // [NE, BS]
        ggml_tensor * ids = ggml_argsort_top_k(c, ggml_add(c, sc, LW(b, "ffn.gate.bias")), (int) NU);
        ggml_tensor * wts = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, sc, 1, sc->ne[0], BS), ids), NU, BS);
        if (Gm.norm_topk && NU > 1) wts = ggml_div(c, wts, ggml_scale_bias(c, ggml_sum_rows(c, wts), 1.0f, 1e-20f));
        wts = ggml_scale(c, wts, (float) Gm.weights_scale);
        ggml_tensor * x3 = ggml_reshape_3d(c, fn, D, 1, BS);
        ggml_tensor * ge = ggml_mul_mat_id(c, LW(b, "ffn.experts.w1.weight"), x3, ids);
        ggml_tensor * ue = ggml_mul_mat_id(c, LW(b, "ffn.experts.w3.weight"), x3, ids);
        if (lim > 0) { ue = ggml_clamp(c, ue, -lim, lim); ge = ggml_clamp(c, ge, NEG_INF, lim); }
        ggml_tensor * act = ggml_mul(c, ggml_swiglu_split(c, ge, ue), ggml_reshape_3d(c, wts, 1, NU, BS));
        ggml_tensor * de = ggml_mul_mat_id(c, LW(b, "ffn.experts.w2.weight"), act, ids);
        ggml_tensor * routed = ggml_view_2d(c, de, D, BS, de->nb[2], 0);
        for (int64_t u = 1; u < NU; ++u) routed = ggml_add(c, routed, ggml_view_2d(c, de, D, BS, de->nb[2], (size_t) u * de->nb[1]));
        ggml_tensor * up = ggml_mul_mat(c, LW(b, "ffn.shared_experts.w3.weight"), fn);
        ggml_tensor * gt = ggml_mul_mat(c, LW(b, "ffn.shared_experts.w1.weight"), fn);
        if (lim > 0) { up = ggml_clamp(c, up, -lim, lim); gt = ggml_clamp(c, gt, NEG_INF, lim); }
        ggml_tensor * se = ggml_mul_mat(c, LW(b, "ffn.shared_experts.w2.weight"), ggml_swiglu_split(c, gt, up));
        x = ggml_dsv4_hc_post(c, ggml_add(c, routed, se), h, f_post, f_comb);
        pre = f_pre;
    }
    ggml_tensor * dh = rms_w(ggml_dsv4_hc_pre(c, x, pre), W("final_norm.weight"));             // [D, BS]
    ggml_build_forward_expand(gf, ggml_cpy(c, dh, s_dh));
    // slot 0's correction key (prev = x_{a-1}) and slot 1's correction (prev = x_a)
    ggml_tensor * dh0 = ggml_view_2d(c, dh, D, 1, dh->nb[1], 0);
    ggml_tensor * dh1 = ggml_view_2d(c, dh, D, 1, dh->nb[1], dh->nb[1]);
    ggml_tensor * ep0 = ggml_view_2d(c, i_eprev, D, 1, i_eprev->nb[1], 0);
    ggml_tensor * cat0 = ggml_concat(c, rms_w(dh0, CW("hidden_norm.weight")), rms_w(ep0, CW("embed_norm.weight")), 0);
    ggml_tensor * bk0 = rms_w(ggml_mul_mat(c, CW("attention.kv_in.weight"), cat0), CW("attention.kv_norm.weight"));
    bk0 = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, bk0, DH, 1, 1), ggml_view_1d(c, i_bpos, 1, 0), false), DH, 1);
    ggml_tensor * bk1 = nullptr;
    ggml_tensor * pp = nullptr;
    ggml_tensor * id = correct(1, dh1, e0, bk0, ctx36, &bk1, &pp);
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_concat(c, bk0, bk1, 1), ggml_view_2d(c, s_bk, DH, 2, s_bk->nb[1], 0)));
    ggml_set_output(id);
    ggml_set_output(pp);
    ggml_build_forward_expand(gf, id);
    ggml_build_forward_expand(gf, pp);
    gr.id = id;
    gr.p = pp;
    gr.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    return gr;
}

bool Ds41DFly2::draft(const Ds41DFly2Ctx & cx, int a, int anchor, int prev, int depth, int * ids, float * prob) {
    Impl & im = *p_;
    for (int i = 0; i < 3; ++i)
        if (!cx.raw[i] || !cx.comp[i] || !cx.sel[i] || !cx.selm[i] || cx.K[i] < 0 || cx.K[i] > TOP) { im.err = "DFly2: incomplete taps"; return false; }
    if (!cx.x || !cx.pre) { im.err = "DFly2: no trunk state"; return false; }
    depth = std::max(1, std::min<int>(depth, (int) im.BS - 1));
#ifdef STRATA_ENABLE_CUDA
    if (im.cfg.cuda_device >= 0) cudaSetDevice(im.cfg.cuda_device);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t D = im.D, BS = im.BS;
    // inputs
    std::vector<float> emb((size_t) (D * BS));
    for (int64_t t = 0; t < BS; ++t) ds41_embd_row(im.embd, im.embd_type, D, t == 0 ? anchor : im.mask_id, emb.data() + (size_t) (t * D));
    std::vector<float> ep((size_t) (2 * D));
    ds41_embd_row(im.embd, im.embd_type, D, prev, ep.data());
    std::vector<int32_t> bp((size_t) BS);
    for (int64_t t = 0; t < BS; ++t) bp[(size_t) t] = a + (int) t;
    const int64_t RAW = cx.raw[0]->ne[1];
    std::vector<int32_t> widx((size_t) WIN);
    std::vector<float> wm((size_t) WIN);
    for (int64_t s = 0; s < WIN; ++s) {   // positions a-128 .. a-1
        const int64_t p = a - WIN + s;
        widx[(size_t) s] = p >= 0 ? (int32_t) (p % RAW) : 0;
        wm[(size_t) s] = p >= 0 ? 0.0f : NEG_INF;
    }
    ggml_backend_tensor_set(im.i_emb, emb.data(), 0, emb.size() * 4);
    ggml_backend_tensor_set(im.i_eprev, ep.data(), 0, (size_t) D * 4);
    ggml_backend_tensor_set(im.i_xs, cx.x, 0, (size_t) (im.HC * D) * 4);
    ggml_backend_tensor_set(im.i_pre1, cx.pre, 0, (size_t) im.HC * 4);
    ggml_backend_tensor_set(im.i_bpos, bp.data(), 0, bp.size() * 4);
    ggml_backend_tensor_set(im.i_widx, widx.data(), 0, widx.size() * 4);
    std::vector<int32_t> ci((size_t) TOP);
    std::vector<float> cm((size_t) TOP);
    std::vector<float> mask((size_t) (MP * BS));
    for (int i = 0; i < 3; ++i) {
        for (int64_t s = 0; s < TOP; ++s) {
            const bool ok = s < cx.K[i] && cx.selm[i][s] == 0.0f && cx.sel[i][s] >= 0;
            ci[(size_t) s] = ok ? cx.sel[i][s] : 0;
            cm[(size_t) s] = ok ? 0.0f : NEG_INF;
        }
        ggml_backend_tensor_set(im.i_cidx[i], ci.data(), 0, ci.size() * 4);
        for (int64_t t = 0; t < BS; ++t) {
            float * m = mask.data() + (size_t) (t * MP);
            std::copy(cm.begin(), cm.end(), m);
            std::copy(wm.begin(), wm.end(), m + TOP);
            for (int64_t k = NCTX; k < MP; ++k) m[k] = k < NCTX + BS ? 0.0f : NEG_INF;
        }
        ggml_backend_tensor_set(im.i_mask[i], mask.data(), 0, mask.size() * 4);
        if (i == 2) {   // the correction's context = layer 36's 640 keys (no block)
            std::vector<float> cmk((size_t) MC, NEG_INF);
            std::copy(cm.begin(), cm.end(), cmk.begin());
            std::copy(wm.begin(), wm.end(), cmk.begin() + TOP);
            ggml_backend_tensor_set(im.i_cmask, cmk.data(), 0, cmk.size() * 4);
        }
    }
    auto set_slot_bias = [&](int j) {   // rows: keys s = 0..j get block_rel_bias[h, j-s]; the sink row at BS
        std::vector<float> rb((size_t) ((BS + 1) * im.NH), NEG_INF);
        std::vector<float> gb((size_t) im.NH);
        for (int64_t h = 0; h < im.NH; ++h) {
            for (int s = 0; s <= j; ++s) rb[(size_t) (h * (BS + 1) + s)] = im.rel_bias[(size_t) (h * NSLOT + (j - s))];
            rb[(size_t) (h * (BS + 1) + BS)] = im.block_sink[(size_t) h];
            gb[(size_t) h] = im.gate_bias[(size_t) (h * NSLOT + j)];
        }
        ggml_backend_tensor_set(im.i_rb, rb.data(), 0, rb.size() * 4);
        ggml_backend_tensor_set(im.i_gb, gb.data(), 0, gb.size() * 4);
    };
    auto run = [&](int j) -> bool {
        Impl::G & gr = im.graph(j, cx);
        if (!ggml_gallocr_alloc_graph(gr.ga, gr.gf)) { im.err = "DFly2 graph allocation failed"; return false; }
        if (ggml_backend_graph_compute(im.be, gr.gf) != GGML_STATUS_SUCCESS) { im.err = "DFly2 graph compute failed"; return false; }
        return true;
    };
    set_slot_bias(1);
    if (!run(0)) return false;
    int32_t tok = 0;
    float pr = 0;
    ggml_backend_tensor_get(im.graphs[0].id, &tok, 0, 4);
    ggml_backend_tensor_get(im.graphs[0].p, &pr, 0, 4);
    ids[0] = tok;
    if (prob) prob[0] = pr;
    // STRATA_DFLY2_STOP=<p>: stop the unroll once a proposal's probability is under p (its later slots would be cut)
    static const float stop_p = std::getenv("STRATA_DFLY2_STOP") ? (float) std::atof(std::getenv("STRATA_DFLY2_STOP")) : -1.0f;
    float cum = pr;
    im.main_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int j = 2; j <= depth; ++j) {
        if (stop_p > 0 && cum < stop_p) { depth = j - 1; break; }
        ds41_embd_row(im.embd, im.embd_type, D, ids[j - 2], ep.data() + (size_t) D);
        ggml_backend_tensor_set(im.i_eprev, ep.data() + (size_t) D, (size_t) D * 4, (size_t) D * 4);
        set_slot_bias(j);
        if (!run(j)) return false;
        ggml_backend_tensor_get(im.graphs[j].id, &tok, 0, 4);
        ggml_backend_tensor_get(im.graphs[j].p, &pr, 0, 4);
        ids[j - 1] = tok;
        if (prob) prob[j - 1] = pr;
        cum *= pr;
    }
    for (int j = depth + 1; j < (int) BS; ++j) { ids[j - 1] = -1; if (prob) prob[j - 1] = 0.0f; }
    im.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
