// tools/v41/engine/ds41_dspark.cpp - see ds41_dspark.hpp.  Line references are to DeepSeek's inference/model.py
// (DSparkBlock / DSparkAttention / forward_embed / forward_head ~1021-1160, Block ~907-995, MoE / Gate ~792-905).
#include "ds41_dspark.hpp"

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
constexpr int64_t WIN = 128;           // config window_size (DSpark attention reach)
constexpr int64_t MP = 256;            // keys per query after padding (ggml-cuda flash-attn d 512 wants % 256)
}

struct Ds41DSpark::Impl {
    Ds41DSparkConfig cfg;
    std::string err;
    std::unique_ptr<GgufModel> side, main;
    Ds41Geometry g;
    ggml_backend_t be = nullptr;
    int64_t D = 0, HC = 0, DH = 0, NH = 0, OG = 0, OL = 0, OGD = 0, NE = 0, NU = 0, NFF = 0, V = 0, BS = 0, KM = 0;
    // weights
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    const uint8_t * embd = nullptr;    // trunk token_embd F16 rows on the mmap
    size_t wbytes = 0;
    // state: per block window ring [DH, WIN]; inputs
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    std::vector<ggml_tensor *> ring;
    ggml_tensor * i_main = nullptr, * i_mpos = nullptr, * i_mslot = nullptr, * i_emb = nullptr, * i_bpos = nullptr;
    ggml_tensor * i_mask = nullptr, * i_bonus = nullptr, * i_pre = nullptr;
    std::vector<int> filled;           // ring slot -> position held (-1 empty)
    // graphs (one per (k, draft?) shape), one allocator
    ggml_gallocr_t ga = nullptr;
    struct G { ggml_context * c = nullptr; ggml_cgraph * gf = nullptr; std::vector<ggml_tensor *> ids, conf, rids; };
    // STRATA_DSPARK_COUNT=<file>: every block's routed-expert picks are counted (draft AND feed graphs) and written at
    // exit as 3 lines of NE counts - the input of STRATA_DSPARK_KEEP.
    std::vector<std::vector<int64_t>> counts;
    // STRATA_DSPARK_KEEP=<counts file>:<K>[,<K1>,<K2>]: only each block's K most-used routed experts are loaded (gate rows, gate
    // bias and the experts sliced on load; the router then picks among those K).  The drafter's experts are ~4.15 GiB
    // of the card - VRAM the trunk's expert cache does not get.  Drafts are verified by the trunk, so this moves only
    // acceptance, never the output.
    std::vector<std::vector<int32_t>> keep;
    std::map<std::pair<int, bool>, G> graphs;
    std::pair<int, bool> last_key { -1, false };   ///< the graph ga was last planned for
    double ms = 0;

    ~Impl() {
        if (const char * cf = std::getenv("STRATA_DSPARK_COUNT"); cf && !counts.empty()) {
            if (FILE * f = std::fopen(cf, "w")) {
                for (auto & r : counts) { for (size_t i = 0; i < r.size(); ++i) std::fprintf(f, "%s%lld", i ? " " : "", (long long) r[i]); std::fputc('\n', f); }
                std::fclose(f);
            }
        }
        for (auto & kv : graphs) ggml_free(kv.second.c);
        if (ga) ggml_gallocr_free(ga);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
    ggml_tensor * W(const std::string & n) const {
        auto it = w.find(n);
        if (it == w.end()) { std::fprintf(stderr, "[ds41_dspark] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    ggml_tensor * BW(int b, const std::string & n) const { return W("mtp." + std::to_string(b) + "." + n); }
    bool load(std::string & e);
    G & graph(int k, bool draft);
};

Ds41DSpark::Ds41DSpark() : p_(new Impl) {}
Ds41DSpark::~Ds41DSpark() = default;
int64_t Ds41DSpark::main_dim() const { return 3 * p_->D; }
double Ds41DSpark::last_ms() const { return p_->ms; }
size_t Ds41DSpark::weight_bytes() const { return p_->wbytes; }
const std::string & Ds41DSpark::last_error() const { return p_->err; }

bool Ds41DSpark::Impl::load(std::string & e) {
    // every mtp.* tensor of the sidecar (bias_vl too: tiny, unused for text) + the trunk's output.weight
    std::vector<std::pair<const GgufModel *, const TensorInfo *>> src;
    for (size_t s = 0; s < side->size(); ++s)
        for (const auto & ti : side->shard(s).tensors())
            if (ti.name.rfind("mtp.", 0) == 0) src.push_back({ side.get(), &ti });
    size_t si = 0;
    const TensorInfo * out = main->find("output.weight", &si);
    const TensorInfo * te = main->find("token_embd.weight");
    if (!out || !te) { e = "the trunk GGUF lacks output.weight / token_embd.weight"; return false; }
    if (te->type != GGML_TYPE_F16) { e = "token_embd must be F16"; return false; }
    if (!cfg.shared_head) src.push_back({ main.get(), out });
    // STRATA_DSPARK_EXP_TYPE=<ggml type name> (q4_K, q3_K, q2_K, iq4_xs, ...): the routed experts
    // (mtp.*.ffn.experts.w1/w2/w3, MXFP4 in the sidecar) are requantized ON LOAD (dequantize -> quantize, CPU threads).
    // A drafter's precision only moves acceptance - every draft is verified by the trunk.
    ggml_type xt = GGML_TYPE_COUNT;
    if (const char * x = std::getenv("STRATA_DSPARK_EXP_TYPE")) {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * nm = ggml_type_name((ggml_type) i);
            if (nm && std::strcmp(nm, x) == 0) xt = (ggml_type) i;
        }
        if (xt == GGML_TYPE_COUNT) { e = std::string("STRATA_DSPARK_EXP_TYPE: unknown type ") + x; return false; }
        if (ggml_quantize_requires_imatrix(xt)) { e = std::string("STRATA_DSPARK_EXP_TYPE ") + x + " needs an imatrix"; return false; }
        ggml_quantize_init(xt);
    }
    auto is_exp = [](const std::string & n) { return n.find(".ffn.experts.") != std::string::npos; };
    // keep: per block, the K most-used experts (ascending index order)
    auto blk_of = [](const std::string & n) { return n.rfind("mtp.", 0) == 0 ? std::atoi(n.c_str() + 4) : -1; };
    auto sliced = [&](const std::string & n) -> int {   // 0 = no, 1 = experts (ne[2]), 2 = gate weight (ne[1]), 3 = gate bias (ne[0])
        if (keep.empty() || blk_of(n) < 0) return 0;
        if (is_exp(n)) return 1;
        auto ends = [&](const char * x) { const size_t l = std::strlen(x); return n.size() > l && n.compare(n.size() - l, l, x) == 0; };
        if (ends(".ffn.gate.weight")) return 2;
        if (ends(".ffn.gate.bias")) return 3;
        return 0;
    };
    if (const char * kp = std::getenv("STRATA_DSPARK_KEEP")) {
        std::string a(kp);
        const size_t c = a.rfind(':');
        if (c == std::string::npos) { e = "STRATA_DSPARK_KEEP=<counts file>:<K>"; return false; }
        std::vector<int> Ks;   // one K for every block, or a comma list per block (block 0's picks spread more widely)
        for (const char * q = a.c_str() + c + 1; *q;) { Ks.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
        FILE * f = std::fopen(a.substr(0, c).c_str(), "r");
        if (!f) { e = "STRATA_DSPARK_KEEP: cannot open " + a.substr(0, c); return false; }
        char line[65536];
        while (std::fgets(line, sizeof line, f)) {
            std::vector<std::pair<long long, int32_t>> v;
            char * q = line;
            for (int32_t i = 0;; ++i) { char * end; const long long x = std::strtoll(q, &end, 10); if (end == q) break; v.push_back({ x, i }); q = end; }
            if (v.empty()) continue;
            const int K = Ks.empty() ? 0 : Ks[std::min(keep.size(), Ks.size() - 1)];
            if (K < (int) cfg.n_used || K > (int) v.size()) { std::fclose(f); e = "STRATA_DSPARK_KEEP: K out of range"; return false; }
            std::stable_sort(v.begin(), v.end(), [](auto & x, auto & y) { return x.first > y.first; });
            std::vector<int32_t> k;
            for (int i = 0; i < K; ++i) k.push_back(v[(size_t) i].second);
            std::sort(k.begin(), k.end());
            keep.push_back(k);
        }
        std::fclose(f);
        if (keep.size() != 3) { e = "STRATA_DSPARK_KEEP: expected 3 lines of counts"; return false; }
    }
    ggml_init_params ip = { (src.size() + 8) * ggml_tensor_overhead(), nullptr, true };
    wctx = ggml_init(ip);
    for (auto & s : src) {
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < s.second->shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) s.second->shape[d];
        if (const int sl = sliced(s.second->name)) ne[sl == 1 ? 2 : sl == 2 ? 1 : 0] = (int64_t) keep[(size_t) blk_of(s.second->name)].size();
        const ggml_type ty = (xt != GGML_TYPE_COUNT && is_exp(s.second->name)) ? xt : (ggml_type) s.second->type;
        if (ty != (ggml_type) s.second->type && ne[0] % ggml_blck_size(ty)) { e = s.second->name + ": row not a multiple of the block"; return false; }
        ggml_tensor * t = ggml_new_tensor(wctx, ty, (int) s.second->shape.size(), ne);
        ggml_set_name(t, s.second->name.c_str());
        w[s.second->name] = t;
    }
    wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_get_default_buffer_type(be));
    if (!wbuf) { e = "cannot allocate the DSpark weights on the backend"; return false; }
    for (auto & s : src) {
        const GgufModel & m = *s.first;
        size_t sh = 0;
        m.find(s.second->name, &sh);
        ggml_tensor * t = w[s.second->name];
        const void * data = m.shard(sh).tensor_data(*s.second);
        std::vector<uint8_t> kept;   // STRATA_DSPARK_KEEP: the kept experts' / gate rows' source bytes, in order
        if (const int sl = sliced(s.second->name)) {
            const ggml_type st = (ggml_type) s.second->type;
            const int64_t n0 = (int64_t) s.second->shape[0];
            const size_t unit = sl == 1 ? ggml_row_size(st, n0) * (size_t) s.second->shape[1]   // one expert
                              : sl == 2 ? ggml_row_size(st, n0)                                   // one gate row
                                        : ggml_type_size(st);                                     // one bias value
            if (sl == 3 && ggml_blck_size(st) != 1) { e = s.second->name + ": quantized gate bias"; return false; }
            const auto & k = keep[(size_t) blk_of(s.second->name)];
            kept.resize(k.size() * unit);
            for (size_t i = 0; i < k.size(); ++i) std::memcpy(kept.data() + i * unit, (const uint8_t *) data + (size_t) k[i] * unit, unit);
            data = kept.data();
        }
        if (t->type == (ggml_type) s.second->type) {
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        } else {   // dequantize + requantize in row chunks over the CPU threads
            const ggml_type st = (ggml_type) s.second->type;
            const auto * tr = ggml_get_type_traits(st);
            if (!tr->to_float) { e = s.second->name + ": no dequantizer for " + ggml_type_name(st); return false; }
            const int64_t n0 = t->ne[0], rows = ggml_nrows(t);
            const size_t srow = ggml_row_size(st, n0), drow = ggml_row_size(t->type, n0);
            std::vector<uint8_t> out((size_t) rows * drow);
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
                        ggml_quantize_chunk(t->type, f.data(), out.data() + (size_t) r * drow, 0, nr, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, out.data(), 0, out.size());
        }
        wbytes += ggml_nbytes(t);
    }
    if (cfg.shared_head) {
        const ggml_tensor * h = cfg.shared_head;
        if (h->ne[0] != (int64_t) out->shape[0] || h->ne[1] != (int64_t) out->shape[1]) { e = "shared_head: shape differs from output.weight"; return false; }
        if (!h->buffer) { e = "shared_head: weight has no device buffer"; return false; }
        const ggml_backend_buffer_type_t head_buft = ggml_backend_buffer_get_type(h->buffer);
        const ggml_backend_buffer_type_t dspark_buft = ggml_backend_get_default_buffer_type(be);
        if (head_buft != dspark_buft) {
            e = "shared_head is not on the DSpark backend (use ds41_head_dense's backend for layer-split)";
            return false;
        }
        w["output.weight"] = cfg.shared_head;
    }
    size_t se = 0;
    main->find("token_embd.weight", &se);
    embd = main->shard(se).tensor_data(*te);
    return true;
}

bool Ds41DSpark::init(const Ds41DSparkConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    im.be = cfg.backend;
    if (!im.be) { err = "Ds41DSpark: a backend is required"; return false; }
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
    im.OGD = G.n_head * G.d_head / G.o_groups; im.NFF = G.n_ff; im.V = G.vocab; im.BS = cfg.block; im.KM = cfg.max_append;
    im.NU = cfg.n_used;
    if (!im.load(err)) return false;
    im.NE = im.BW(0, "ffn.experts.w1.weight")->ne[2];
    if (im.BW(0, "main_proj.weight")->ne[0] != 3 * im.D) { err = "mtp.0.main_proj is not [3*n_embd, n_embd]"; return false; }
    // state + inputs
    ggml_init_params sp = { 32 * ggml_tensor_overhead(), nullptr, true };
    im.sctx = ggml_init(sp);
    for (int b = 0; b < 3; ++b) im.ring.push_back(ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.DH, WIN));
    im.i_main = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, 3 * im.D, im.KM);
    im.i_mpos = ggml_new_tensor_1d(im.sctx, GGML_TYPE_I32, im.KM);
    im.i_mslot = ggml_new_tensor_1d(im.sctx, GGML_TYPE_I32, im.KM);
    im.i_emb = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, im.BS);
    im.i_bpos = ggml_new_tensor_1d(im.sctx, GGML_TYPE_I32, im.BS);
    im.i_mask = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, MP, im.BS);
    im.i_bonus = ggml_new_tensor_1d(im.sctx, GGML_TYPE_I32, 1);
    im.i_pre = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.HC, im.BS);
    im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.be));
    if (!im.sbuf) { err = "cannot allocate the DSpark state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    std::vector<float> pre((size_t) (im.HC * im.BS), 0.0f);   // model.py make_identity_pre_mix: one-hot on copy 0
    for (int64_t t = 0; t < im.BS; ++t) pre[(size_t) (t * im.HC)] = 1.0f;
    ggml_backend_tensor_set(im.i_pre, pre.data(), 0, pre.size() * 4);
    im.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.be));
    reset();
    return true;
}

void Ds41DSpark::reset() {
    p_->filled.assign(WIN, -1);
}

static ggml_tensor * fake_quant_fp8(ggml_context * c, ggml_tensor * x) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(c, x, Ds41Quant::FP8_POW2_32) : ggml_dsv41_fake_quant(c, x, (int) Ds41Quant::FP8_POW2_32);
#else
    return ds41_fake_quant(c, x, Ds41Quant::FP8_POW2_32);
#endif
}

Ds41DSpark::Impl::G & Ds41DSpark::Impl::graph(int k, bool draft) {
    G & gr = graphs[{ k, draft }];
    if (gr.gf) return gr;
    ggml_init_params ip = { 16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false), nullptr, true };
    ggml_context * c = gr.c = ggml_init(ip);
    ggml_cgraph * gf = gr.gf = ggml_new_graph_custom(c, 16384, false);
    const Ds41Geometry & Gm = g;
    auto rms_w = [&](ggml_tensor * x, ggml_tensor * wt) { return ggml_mul(c, ggml_rms_norm(c, x, (float) Gm.rms_eps), wt); };
    // RoPE over the last d_rope dims, NORMAL pairs; DSpark layers have compress ratio 0: rope_theta, no YaRN
    auto rope = [&](ggml_tensor * a, ggml_tensor * pos, bool back) {
        const int nd = (int) Gm.d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, 0, (float) Gm.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, 0, (float) Gm.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    };
    // Block.hc_mixes (the same composite as ds41_dense's B::hc_mixes)
    auto hc_mixes = [&](ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                        ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) {
        const int64_t nt = x->ne[2];
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) Gm.rms_eps);
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

    // ---- main_x and the window writes (model.py forward_embed: main_norm(main_proj(main_hidden)))
    ggml_tensor * main = ggml_view_2d(c, i_main, 3 * D, k, i_main->nb[1], 0);
    ggml_tensor * mx = rms_w(ggml_mul_mat(c, W("mtp.0.main_proj.weight"), main), W("mtp.0.main_norm.weight"));   // [D, k]
    ggml_tensor * mpos = ggml_view_1d(c, i_mpos, k, 0);
    ggml_tensor * mslot = ggml_view_1d(c, i_mslot, k, 0);
    std::vector<ggml_tensor *> ring_w(3);
    for (int b = 0; b < 3; ++b) {   // DSparkAttention: main_kv = kv_norm(wkv(main_x)), RoPE at its position, FP8-rounded
        ggml_tensor * mkv = rms_w(ggml_mul_mat(c, BW(b, "attn.wkv.weight"), mx), BW(b, "attn.kv_norm.weight"));
        mkv = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, mkv, DH, 1, k), mpos, false), DH, k);
        mkv = fake_quant_fp8(c, ggml_cont(c, mkv));
        ring_w[b] = ggml_set_rows(c, ring[(size_t) b], mkv, mslot);
    }
    if (!draft) {
        for (auto * t : ring_w) ggml_build_forward_expand(gf, t);
        return gr;
    }

    // ---- the block: [bonus, noise x4] at positions p+1..p+5, hc copies of the embedding, identity pre-mix
    ggml_tensor * bpos = i_bpos;
    ggml_tensor * x = ggml_repeat(c, ggml_reshape_3d(c, i_emb, D, 1, BS), ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, BS));
    ggml_tensor * pre = i_pre;
    ggml_tensor * m4 = ggml_cast(c, ggml_reshape_4d(c, i_mask, MP, 1, 1, BS), GGML_TYPE_F16);
    const float lim = (float) Gm.swiglu_limit;
    for (int b = 0; b < 3; ++b) {
        // attention (Block.forward: hc_mixes of the input, hc_pre with the previous FFN's pre-mix, attn_norm)
        ggml_tensor * a_pre, * a_post, * a_comb;
        hc_mixes(x, BW(b, "hc_attn_fn"), BW(b, "hc_attn_scale"), BW(b, "hc_attn_base"), &a_pre, &a_post, &a_comb);
        ggml_tensor * xn = rms_w(ggml_dsv4_hc_pre(c, x, pre), BW(b, "attn_norm.weight"));
        ggml_tensor * qr = rms_w(ggml_mul_mat(c, BW(b, "attn.wq_a.weight"), xn), BW(b, "attn.q_norm.weight"));
        ggml_tensor * q = rope(ggml_reshape_3d(c, ggml_mul_mat(c, BW(b, "attn.wq_b.weight"), qr), DH, NH, BS), bpos, false);
        ggml_tensor * kvb = rms_w(ggml_mul_mat(c, BW(b, "attn.wkv.weight"), xn), BW(b, "attn.kv_norm.weight"));
        kvb = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, kvb, DH, 1, BS), bpos, false), DH, BS);
        kvb = fake_quant_fp8(c, ggml_cont(c, kvb));
        // keys = [window ring (incl. this step's rows) | the 5 block rows | pad], every query sees all of them
        ggml_tensor * K = ggml_pad(c, ggml_concat(c, ring_w[(size_t) b], kvb, 1), 0, (int) (MP - WIN - BS), 0, 0);   // [DH, MP]
        ggml_tensor * k4 = ggml_cast(c, ggml_repeat(c, ggml_reshape_4d(c, K, DH, MP, 1, 1),
                                                    ggml_new_tensor_4d(c, GGML_TYPE_F32, DH, MP, 1, BS)), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, DH, NH, 1, BS), 0, 2, 1, 3);    // [DH, 1, NH, BS]
        ggml_tensor * o = ggml_flash_attn_ext(c, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, BW(b, "attn.attn_sink"));
        o = rope(ggml_reshape_3d(c, o, DH, NH, BS), bpos, true);
        o = ggml_permute(c, ggml_reshape_3d(c, o, OGD, OG, BS), 0, 2, 1, 3);                      // [OGD, BS, OG]
        ggml_tensor * wo_a = ggml_reshape_3d(c, BW(b, "attn.wo_a.weight"), OGD, OL, OG);
        ggml_tensor * oa = ggml_permute(c, ggml_mul_mat(c, wo_a, o), 0, 2, 1, 3);                 // [OL, OG, BS]
        ggml_tensor * attn = ggml_mul_mat(c, BW(b, "attn.wo_b.weight"), ggml_cont_2d(c, oa, OL * OG, BS));
        ggml_tensor * h = ggml_dsv4_hc_post(c, attn, x, a_post, a_comb);
        // FFN: its mixes give the next block's pre-mix; its input uses the attention's pre-mix
        ggml_tensor * f_pre, * f_post, * f_comb;
        hc_mixes(h, BW(b, "hc_ffn_fn"), BW(b, "hc_ffn_scale"), BW(b, "hc_ffn_base"), &f_pre, &f_post, &f_comb);
        ggml_tensor * fn = rms_w(ggml_dsv4_hc_pre(c, h, a_pre), BW(b, "ffn_norm.weight"));   // [D, BS]
        // Gate: sqrt(softplus), top-NU on score + bias, weights from the raw scores
        ggml_tensor * rl = ggml_mul_mat(c, BW(b, "ffn.gate.weight"), fn);
        ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
        ggml_tensor * sc = ggml_sqrt(c, ggml_softplus(c, rl));                                  // [NE, BS]
        ggml_tensor * ids = ggml_argsort_top_k(c, ggml_add(c, sc, BW(b, "ffn.gate.bias")), (int) NU);   // [NU, BS]
        if (std::getenv("STRATA_DSPARK_COUNT")) { ggml_set_output(ids); gr.rids.push_back(ids); ggml_build_forward_expand(gf, ids); }
        ggml_tensor * wts = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, sc, 1, sc->ne[0], BS), ids), NU, BS);
        if (Gm.norm_topk && NU > 1) wts = ggml_div(c, wts, ggml_scale_bias(c, ggml_sum_rows(c, wts), 1.0f, 1e-20f));
        wts = ggml_scale(c, wts, (float) Gm.weights_scale);
        // routed experts (Expert.forward: clamped SwiGLU, the routing weight applied before w2)
        ggml_tensor * x3 = ggml_reshape_3d(c, fn, D, 1, BS);
        ggml_tensor * ge = ggml_mul_mat_id(c, BW(b, "ffn.experts.w1.weight"), x3, ids);         // [NFF, NU, BS]
        ggml_tensor * ue = ggml_mul_mat_id(c, BW(b, "ffn.experts.w3.weight"), x3, ids);
        if (lim > 0) { ue = ggml_clamp(c, ue, -lim, lim); ge = ggml_clamp(c, ge, NEG_INF, lim); }
        ggml_tensor * act = ggml_mul(c, ggml_swiglu_split(c, ge, ue), ggml_reshape_3d(c, wts, 1, NU, BS));
        ggml_tensor * de = ggml_mul_mat_id(c, BW(b, "ffn.experts.w2.weight"), act, ids);        // [D, NU, BS]
        ggml_tensor * routed = ggml_view_2d(c, de, D, BS, de->nb[2], 0);
        for (int64_t u = 1; u < NU; ++u) routed = ggml_add(c, routed, ggml_view_2d(c, de, D, BS, de->nb[2], (size_t) u * de->nb[1]));
        // shared expert
        ggml_tensor * up = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w3.weight"), fn);
        ggml_tensor * gt = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w1.weight"), fn);
        if (lim > 0) { up = ggml_clamp(c, up, -lim, lim); gt = ggml_clamp(c, gt, NEG_INF, lim); }
        ggml_tensor * se = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w2.weight"), ggml_swiglu_split(c, gt, up));
        x = ggml_dsv4_hc_post(c, ggml_add(c, routed, se), h, f_post, f_comb);
        pre = f_pre;
    }
    // ---- forward_head: hc_pre -> norm -> trunk head; Markov bias left to right; confidence
    ggml_tensor * hn = ggml_dsv4_hc_pre(c, x, pre);                                             // [D, BS] (pre-norm)
    ggml_tensor * lg = ggml_mul_mat(c, W("output.weight"), rms_w(hn, W("mtp.2.norm.weight")));     // [V, BS]
    ggml_mul_mat_set_prec(lg, GGML_PREC_F32);
    ggml_tensor * tok = i_bonus;
    for (int64_t i = 0; i < BS; ++i) {
        ggml_tensor * e = ggml_get_rows(c, W("mtp.2.markov_head.embed.weight"), tok);            // [256, 1]
        ggml_tensor * bias = ggml_mul_mat(c, W("mtp.2.markov_head.head.weight"), e);             // [V, 1]
        ggml_mul_mat_set_prec(bias, GGML_PREC_F32);
        ggml_tensor * li = ggml_add(c, ggml_view_2d(c, lg, V, 1, lg->nb[1], (size_t) i * lg->nb[1]), bias);
        tok = ggml_argmax(c, li);                                                                 // I32 [1]
        ggml_tensor * hc = ggml_concat(c, ggml_view_2d(c, hn, D, 1, hn->nb[1], (size_t) i * hn->nb[1]), e, 0);
        ggml_tensor * cf = ggml_mul_mat(c, W("mtp.2.confidence_head.proj.weight"), hc);           // [1, 1]
        ggml_set_output(tok);
        ggml_set_output(cf);
        gr.ids.push_back(tok);
        gr.conf.push_back(cf);
        ggml_build_forward_expand(gf, tok);
        ggml_build_forward_expand(gf, cf);
    }
    return gr;
}

bool Ds41DSpark::step(const float * mh, const int * pos, int k, int bonus, int * ids, float * conf) {
    Impl & im = *p_;
    if (k < 1 || k > im.KM) { im.err = "DSpark step: k out of range"; return false; }
#ifdef STRATA_ENABLE_CUDA
    if (im.cfg.cuda_device >= 0) cudaSetDevice(im.cfg.cuda_device);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> mp(pos, pos + k), ms((size_t) k);
    for (int i = 0; i < k; ++i) { ms[(size_t) i] = pos[i] % (int) WIN; im.filled[(size_t) ms[(size_t) i]] = pos[i]; }
    ggml_backend_tensor_set(im.i_main, mh, 0, (size_t) k * (size_t) (3 * im.D) * 4);
    ggml_backend_tensor_set(im.i_mpos, mp.data(), 0, (size_t) k * 4);
    ggml_backend_tensor_set(im.i_mslot, ms.data(), 0, (size_t) k * 4);
    const bool draft = bonus >= 0;
    if (draft) {
        const int p = pos[k - 1];
        // embeddings of [bonus, noise x4] (trunk token_embd, F16 rows)
        std::vector<float> emb((size_t) (im.D * im.BS));
        for (int64_t t = 0; t < im.BS; ++t) {
            const int id = t == 0 ? bonus : im.cfg.noise_id;
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) (im.embd + (size_t) id * (size_t) im.D * 2), emb.data() + (size_t) (t * im.D), im.D);
        }
        std::vector<int32_t> bp((size_t) im.BS);
        for (int64_t t = 0; t < im.BS; ++t) bp[(size_t) t] = p + 1 + (int) t;
        // keys: window slots holding a position in (p - WIN, p] are visible; the 5 block rows always; pad never
        std::vector<float> mask((size_t) (MP * im.BS), NEG_INF);
        for (int64_t t = 0; t < im.BS; ++t) {
            float * m = mask.data() + (size_t) (t * MP);
            for (int64_t s = 0; s < WIN; ++s) { const int q = im.filled[(size_t) s]; if (q >= 0 && q <= p && q > p - WIN) m[s] = 0.0f; }
            for (int64_t j = 0; j < im.BS; ++j) m[WIN + j] = 0.0f;
        }
        const int32_t bo = bonus;
        ggml_backend_tensor_set(im.i_emb, emb.data(), 0, emb.size() * 4);
        ggml_backend_tensor_set(im.i_bpos, bp.data(), 0, bp.size() * 4);
        ggml_backend_tensor_set(im.i_mask, mask.data(), 0, mask.size() * 4);
        ggml_backend_tensor_set(im.i_bonus, &bo, 0, 4);
    }
    Impl::G & gr = im.graph(k, draft);
    // ONE gallocr serves every cached graph, and ggml_gallocr only places tensors whose data is NULL: a cached graph
    // kept the addresses of its FIRST allocation, which a later, bigger graph's reserve frees (the buffer is
    // reallocated) - --serve crashed (CUDA invalid argument in a cpy) when a 16-row seeding graph ran again after the
    // decode graphs had grown the buffer.  So: this graph's own tensors are re-placed every step (external weights /
    // state live in other contexts and keep theirs), and a graph change re-plans.
    for (ggml_tensor * t = ggml_get_first_tensor(gr.c); t; t = ggml_get_next_tensor(gr.c, t)) {
        t->data = nullptr;
        t->buffer = nullptr;
    }
    const std::pair<int, bool> key { k, draft };
    if (im.last_key != key) {
        if (!ggml_gallocr_reserve(im.ga, gr.gf)) { im.err = "DSpark graph reserve failed"; return false; }
        im.last_key = key;
    }
    if (!ggml_gallocr_alloc_graph(im.ga, gr.gf)) { im.err = "DSpark graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(im.be, gr.gf) != GGML_STATUS_SUCCESS) { im.err = "DSpark graph compute failed"; return false; }
    if (!gr.rids.empty()) {
        if (im.counts.empty()) im.counts.assign(gr.rids.size(), std::vector<int64_t>((size_t) im.NE, 0));
        for (size_t b = 0; b < gr.rids.size(); ++b) {
            std::vector<int32_t> v((size_t) ggml_nelements(gr.rids[b]));
            ggml_backend_tensor_get(gr.rids[b], v.data(), 0, v.size() * 4);
            for (int32_t x : v) if (x >= 0 && x < im.NE) ++im.counts[b][(size_t) x];
        }
    }
    if (draft) {
        for (size_t i = 0; i < gr.ids.size(); ++i) {
            ggml_backend_tensor_get(gr.ids[i], &ids[i], 0, 4);
            ggml_backend_tensor_get(gr.conf[i], &conf[i], 0, 4);
        }
    }
    im.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
