// qfn_harness_train <pack.gguf> <capture-root> <run-name> [steps]
//
// LoRA training of QFN (Swift 1.5 QFN-XQ-45GB) on QuetzaCodetl harness transcripts, ON THE 4090, from the served
// model's own activations: harness_capture.py wrote, per window, the residual entering layer 44 (x44.bf16), the
// window tokens + loss mask (tokens.json) and layer 47's engine selection (qsa.bin); the 20,135-token Quetza prefix's
// x44 rows are shared (prefix_x44.bf16).
//
// Trained: rank-r adapters on layers 44-47 (GDN: attn_qkv, attn_gate, ssm_out; QSA: attn_q, attn_k, attn_v,
// attn_output), base weights frozen and quantized exactly as served. Loss: next-token CE on the assistant tokens.
// Per step (one window), the hy4_train pattern - small per-layer graphs instead of one huge one:
//   1. prefix carries (GDN states + conv rows for 44-46, K/V for 47) computed with the CURRENT adapter, no gradient,
//      every QFN_PREFIX_EVERY steps (the prefix is identical for every window)
//   2. window forward 44 -> 47 (no grad): each layer's input + its MoE routing (host top-k, constant)
//   3. head: CE in chunks of 256 loss positions -> d loss / d x48
//   4. layers 47 -> 44: VJP graphs (surrogate sum(out * g_out)) -> d/d x_l and the adapter gradients
//   5. AdamW on the host (global-norm clip), adapters re-uploaded
// Layer 47 attends, per chunk of 256 queries, over the UNION of their engine-selected cells (prefix K/V constant,
// window K/V live), with an additive mask - exactly the served sparse attention.
// Metrics -> RUNS/<run>/metrics.jsonl (tools/train/lossplot.py draws them); adapter -> RUNS/<run>/adapter.gguf.
//
// env: QFN_LR (2e-4) QFN_RANK (16) QFN_ALPHA (32) QFN_WARMUP (20) QFN_PREFIX_EVERY (50) QFN_EVAL_EVERY (25)
//      QFN_EVAL_N (2: windows held out) QFN_CLIP (1.0) QFN_SAVE_EVERY (100) QFN_RUNS (runs)
//      QFN_ROPE_SCALE (2: the served YaRN) QFN_SEED (1)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "qfn_topk.hpp"
#include "lora_gguf.hpp"
#include "metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace xolotl::train;

static const char * env(const char * k, const char * d) { const char * v = std::getenv(k); return v && *v ? v : d; }

static std::vector<float> read_bf16(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> b((size_t) n / 2);
    if (std::fread(b.data(), 2, b.size(), f) != b.size()) std::exit(1);
    std::fclose(f);
    std::vector<float> v(b.size());
    for (size_t i = 0; i < b.size(); ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&v[i], &u, 4); }
    return v;
}

static std::vector<int> json_ints(const std::string & s, const std::string & key) {
    std::vector<int> v;
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return v;
    p = s.find('[', p);
    const size_t e = s.find(']', p);
    std::stringstream ss(s.substr(p + 1, e - p - 1));
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

// ---- one graph on the backend: a no-alloc context, host inputs uploaded after allocation ---------------------------
// Memory: ONE shared ggml_gallocr (liveness-based reuse, grows to the largest graph) - not alloc_ctx_tensors, which
// gives every intermediate its own memory (17.7 GB for one forward layer at W = 1551, 28 GB for the backward at 3063).
// Inputs (host uploads, gradient accumulators zeroed by graph_reset) are flagged INPUT so nothing computed earlier can
// share their memory; everything read back afterwards is flagged OUTPUT so it is not reused before we read it.
static ggml_gallocr_t g_alloc = nullptr;
static size_t g_alloc_peak = 0;
struct G1 {
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> up;
    explicit G1(size_t n_nodes = 1 << 17) {
        ggml_init_params ip = { ggml_tensor_overhead() * (n_nodes * 2) + ggml_graph_overhead_custom(n_nodes, true), nullptr, true };
        c = ggml_init(ip);
        gf = ggml_new_graph_custom(c, n_nodes, true);
    }
    ~G1() { if (c) ggml_free(c); }
    ggml_tensor * f32(const float * d, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
        ggml_tensor * t = ggml_new_tensor_4d(c, GGML_TYPE_F32, n0, n1, n2, n3);
        ggml_set_input(t);
        up.emplace_back(t, std::vector<uint8_t>((const uint8_t *) d, (const uint8_t *) d + (size_t) ggml_nbytes(t)));
        return t;
    }
    ggml_tensor * i32(const std::vector<int32_t> & d) {
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) d.size());
        ggml_set_input(t);
        up.emplace_back(t, std::vector<uint8_t>((const uint8_t *) d.data(), (const uint8_t *) (d.data() + d.size())));
        return t;
    }
    // grads_of: tensors whose gradient is read back after compute
    bool run(ggml_backend_t be, ggml_tensor * loss_or_null, const std::vector<ggml_tensor *> & outs,
             const std::vector<ggml_tensor *> & grads_of = {}) {
        for (ggml_tensor * o : outs) if (o) { ggml_set_output(o); ggml_build_forward_expand(gf, o); }
        if (loss_or_null) {
            ggml_set_loss(loss_or_null);
            ggml_build_forward_expand(gf, loss_or_null);
            ggml_build_backward_expand(c, gf, nullptr);
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * acc = ggml_graph_get_grad_acc(gf, ggml_graph_node(gf, i));
                if (acc) ggml_set_input(acc);
            }
            for (ggml_tensor * t : grads_of) {
                if (ggml_tensor * acc = ggml_graph_get_grad_acc(gf, t)) ggml_set_input(acc);
                if (ggml_tensor * gr = ggml_graph_get_grad(gf, t)) ggml_set_output(gr);
            }
        }
        if (!g_alloc) g_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(g_alloc, gf)) { std::fprintf(stderr, "graph allocation failed (VRAM)\n"); return false; }
        const size_t bs = ggml_gallocr_get_buffer_size(g_alloc, 0);
        if (bs > g_alloc_peak) {
            g_alloc_peak = bs;
            std::fprintf(stderr, "[alloc] compute buffer grew to %.2f GiB (%d nodes)\n", bs / 1073741824.0, ggml_graph_n_nodes(gf));
        }
        for (auto & h : up)   // an input this graph never reads (e.g. pos in a router-only pass) has no memory: skip it
            if (h.first->buffer) ggml_backend_tensor_set(h.first, h.second.data(), 0, h.second.size());
        if (loss_or_null) ggml_graph_reset(gf);
        return ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    }
};

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
    return v;
}

// ---- adapters ---------------------------------------------------------------------------------------------------
struct Ad {
    std::string name;            // base tensor, e.g. blk.44.attn_qkv.weight
    int64_t layer, n_in, n_out;
    ggml_tensor * A = nullptr, * B = nullptr;
    std::vector<float> a, b, ma, va, mb, vb, ga, gb;
};

struct Window {
    std::string dir;
    std::vector<int> tok, mask;
    int n_loss = 0;
};

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <pack.gguf> <capture-root> <run-name> [steps]\n", argv[0]); return 2; }
    const std::string root = argv[2], run = argv[3];
    const int steps = argc > 4 ? atoi(argv[4]) : 200;
    const float lr0 = (float) atof(env("QFN_LR", "2e-4"));
    const int64_t rank = atoll(env("QFN_RANK", "16"));
    const float alpha = (float) atof(env("QFN_ALPHA", "32"));
    const int warmup = atoi(env("QFN_WARMUP", "20")), prefix_every = atoi(env("QFN_PREFIX_EVERY", "50"));
    const int eval_every = atoi(env("QFN_EVAL_EVERY", "25")), eval_n = atoi(env("QFN_EVAL_N", "2"));
    const int save_every = atoi(env("QFN_SAVE_EVERY", "100"));
    const float clip = (float) atof(env("QFN_CLIP", "1.0"));
    const int64_t CH = 2048, CQ = 256, CL = 256, L0 = 44;
    if (!std::getenv("QFN_ROPE_SCALE")) setenv("QFN_ROPE_SCALE", "2", 1);   // the SERVED rope unless told otherwise

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "no CUDA\n"); return 1; }
    PackTensors p;
    if (!p.open(argv[1])) return 1;
    const QfnGeom G = qfn_geom(p);
    std::vector<std::string> names = { "output.weight", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight" };
    for (int64_t l = L0; l < G.NL; ++l) for (auto & n : qfn_layer_names(l)) names.push_back(n);
    auto tl0 = std::chrono::steady_clock::now();
    if (!p.load_backend(names, be)) { std::fprintf(stderr, "weight load failed\n"); return 1; }
    std::printf("[train] weights for layers %lld-%lld + head on the GPU (%.1f s); rope freq_scale %.2f ext %.0f\n",
                (long long) L0, (long long) G.NL - 1,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count(), G.rope_freq_scale, G.rope_ext);
    const int64_t row = G.hc * G.D;
    const float ls = alpha / (float) rank;

    // ---- data
    std::vector<Window> wins;
    if (DIR * d = opendir(root.c_str())) {
        while (dirent * e = readdir(d)) {
            const std::string nm = e->d_name;
            if (nm.size() < 2 || nm[0] != 'w') continue;
            Window w;
            w.dir = root + "/" + nm;
            std::ifstream tf(w.dir + "/tokens.json"), xf(w.dir + "/x44.bf16"), qf(w.dir + "/qsa.bin");
            if (!tf || !xf || !qf) continue;
            std::string js((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
            w.tok = json_ints(js, "tokens");
            w.mask = json_ints(js, "mask");
            for (size_t i = 1; i < w.mask.size(); ++i) w.n_loss += w.mask[i];
            wins.push_back(w);
        }
        closedir(d);
    }
    std::sort(wins.begin(), wins.end(), [](const Window & a, const Window & b) { return a.dir < b.dir; });
    if ((int) wins.size() <= eval_n) { std::fprintf(stderr, "need more than %d captured windows (have %zu)\n", eval_n, wins.size()); return 2; }
    std::vector<Window> evals(wins.end() - eval_n, wins.end());
    wins.resize(wins.size() - eval_n);
    const std::vector<float> prefix_x44 = read_bf16(root + "/prefix_x44.bf16");
    const int64_t P = (int64_t) (prefix_x44.size() / row);
    long loss_tok = 0;
    for (auto & w : wins) loss_tok += w.n_loss;
    std::printf("[train] %zu train windows (%ld loss tokens), %zu held out; prefix %lld positions\n", wins.size(), loss_tok,
                evals.size(), (long long) P);

    // ---- adapters (A ~ N(0, 1/n_in) like kaiming-ish, B = 0: the adapter starts as the identity)
    std::vector<Ad> ads;
    std::mt19937 rng((unsigned) atoi(env("QFN_SEED", "1")));
    for (int64_t l = L0; l < G.NL; ++l) {
        const std::vector<const char *> tg = G.is_qsa(l) ? std::vector<const char *>{ "attn_q", "attn_k", "attn_v", "attn_output" }
                                                         : std::vector<const char *>{ "attn_qkv", "attn_gate", "ssm_out" };
        for (const char * t : tg) {
            Ad a;
            a.name = "blk." + std::to_string((long long) l) + "." + t + ".weight";
            a.layer = l;
            ggml_tensor * W = p.get(a.name);
            a.n_in = W->ne[0];
            a.n_out = W->ne[1];
            std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float) a.n_in));
            a.a.resize((size_t) (a.n_in * rank));
            for (auto & x : a.a) x = nd(rng);
            a.b.assign((size_t) (rank * a.n_out), 0.0f);
            a.ma.assign(a.a.size(), 0); a.va.assign(a.a.size(), 0); a.mb.assign(a.b.size(), 0); a.vb.assign(a.b.size(), 0);
            ads.push_back(std::move(a));
        }
    }
    ggml_init_params lip = { ads.size() * 2 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * lctx = ggml_init(lip);
    for (auto & a : ads) {
        a.A = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, a.n_in, rank);
        a.B = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, rank, a.n_out);
    }
    ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, be);
    auto upload = [&]() { for (auto & a : ads) { ggml_backend_tensor_set(a.A, a.a.data(), 0, a.a.size() * 4);
                                                 ggml_backend_tensor_set(a.B, a.b.data(), 0, a.b.size() * 4); } };
    upload();
    size_t n_par = 0;
    for (auto & a : ads) n_par += a.a.size() + a.b.size();
    std::printf("[train] %zu adapters, rank %lld alpha %.0f, %zu params\n", ads.size(), (long long) rank, alpha, n_par);

    auto wire = [&](TopLayer & L) {   // this layer's adapters into its structs
        for (auto & a : ads) {
            if (a.layer != L.l) continue;
            const std::string t = a.name.substr(a.name.find('.', 4) + 1);
            if (t == "attn_qkv.weight") { L.gdn.lora_a_qkv = a.A; L.gdn.lora_b_qkv = a.B; }
            else if (t == "attn_gate.weight") { L.gdn.lora_a_gate = a.A; L.gdn.lora_b_gate = a.B; }
            else if (t == "ssm_out.weight") { L.gdn.lora_a_out = a.A; L.gdn.lora_b_out = a.B; }
            else if (t == "attn_q.weight") L.qsa_lora.q = { a.A, a.B };
            else if (t == "attn_k.weight") L.qsa_lora.k = { a.A, a.B };
            else if (t == "attn_v.weight") L.qsa_lora.v = { a.A, a.B };
            else if (t == "attn_output.weight") L.qsa_lora.o = { a.A, a.B };
        }
        L.gdn.lora_alpha = alpha;
        L.gdn.lora_rank = rank;
    };
    std::vector<float> idv((size_t) (kGdnChunk * kGdnChunk), 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) idv[(size_t) (i * kGdnChunk + i)] = 1.0f;
    auto posv = [](int64_t p0, int64_t n) { std::vector<int32_t> v((size_t) n); for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (int32_t) (p0 + i); return v; };

    // ---- carries
    struct HostT { std::vector<float> d; int64_t ne[4] = { 1, 1, 1, 1 }; };
    auto to_host = [](ggml_tensor * t, HostT & h) { h.d = get(t); for (int i = 0; i < 4; ++i) h.ne[i] = t->ne[i]; };
    struct Carry { HostT conv, s; std::vector<float> k, v; int64_t nkv = 0; };
    auto set_carry = [&](G1 & g, TopLayer & L, const Carry & cy) {
        if (L.qsa) {
            if (cy.nkv) {
                L.qsa_carry.k_in = ggml_reshape_3d(g.c, g.f32(cy.k.data(), G.ahd, G.n_kv, cy.nkv), G.ahd, G.n_kv, cy.nkv);
                L.qsa_carry.v_in = ggml_reshape_3d(g.c, g.f32(cy.v.data(), G.ahd, G.n_kv, cy.nkv), G.ahd, G.n_kv, cy.nkv);
            }
        } else if (!cy.s.d.empty()) {
            L.gdn_carry.conv_in = ggml_reshape_2d(g.c, g.f32(cy.conv.d.data(), cy.conv.ne[0], cy.conv.ne[1]), cy.conv.ne[0], cy.conv.ne[1]);
            L.gdn_carry.s_in = g.f32(cy.s.d.data(), cy.s.ne[0], cy.s.ne[1], cy.s.ne[2], cy.s.ne[3]);
        }
    };
    // MoE index tensors for a G1 graph
    auto moe_index = [&](G1 & g, const std::vector<int32_t> & ids, int64_t n) {
        std::vector<std::vector<int32_t>> tok_of, pair_of;
        moe_expert_lists(ids, G.K, n, G.E, tok_of, pair_of);
        MoeIndex ix;
        std::vector<int32_t> ws((size_t) (G.K * n)), ta, pa;
        for (int64_t t = 0; t < n; ++t) for (int64_t k = 0; k < G.K; ++k) ws[(size_t) (t * G.K + k)] = (int32_t) (t * G.E + ids[(size_t) (t * G.K + k)]);
        ix.wsel = g.i32(ws);
        ix.off.assign((size_t) G.E, 0); ix.cnt.assign((size_t) G.E, 0);
        for (int64_t e = 0; e < G.E; ++e) {
            ix.off[(size_t) e] = (int64_t) ta.size();
            ix.cnt[(size_t) e] = (int64_t) tok_of[(size_t) e].size();
            ta.insert(ta.end(), tok_of[(size_t) e].begin(), tok_of[(size_t) e].end());
            pa.insert(pa.end(), pair_of[(size_t) e].begin(), pair_of[(size_t) e].end());
        }
        ix.tok_all = g.i32(ta);
        ix.pair_all = g.i32(pa);
        ix.tok.assign((size_t) G.E, nullptr);
        ix.pair.assign((size_t) G.E, nullptr);
        return ix;
    };
    auto route = [&](const std::vector<float> & logits_te, int64_t n) {   // ggml [E, n] (t*E+e) -> ids
        std::vector<float> lt((size_t) (G.E * n));
        for (int64_t t = 0; t < n; ++t) for (int64_t e = 0; e < G.E; ++e) lt[(size_t) (e * n + t)] = logits_te[(size_t) (t * G.E + e)];
        std::vector<int32_t> ids;
        std::vector<float> wts;
        moe_route(lt.data(), G.E, n, G.K, ids, wts);
        return ids;
    };

    // ---- the layer-47 attention over each query chunk's selected cells (prefix K/V constant, window K/V live)
    struct Sel { std::vector<std::vector<int32_t>> cells; };   // per window query
    auto qsa_window_block = [&](G1 & g, ggml_tensor * x /* [D, W] mixed */, TopLayer & L, ggml_tensor * pos,
                                const Carry & pre, const Sel & sel, int64_t W) {
        const int64_t H = G.n_head, HK = G.n_kv, hd = G.ahd;
        ggml_tensor *q, *gate, *k, *v;
        qfn_qsa_qkv(g.c, x, p, L.l, G, pos, &L.qsa_lora, ls, &q, &gate, &k, &v);
        ggml_tensor * kp = ggml_reshape_3d(g.c, g.f32(pre.k.data(), hd, HK, pre.nkv), hd, HK, pre.nkv);
        ggml_tensor * vp = ggml_reshape_3d(g.c, g.f32(pre.v.data(), hd, HK, pre.nkv), hd, HK, pre.nkv);
        ggml_tensor * K2 = ggml_reshape_2d(g.c, ggml_concat(g.c, kp, k, 2), hd * HK, pre.nkv + W);   // rows = cells
        ggml_tensor * V2 = ggml_reshape_2d(g.c, ggml_concat(g.c, vp, v, 2), hd * HK, pre.nkv + W);
        ggml_tensor * out = nullptr;
        for (int64_t q0 = 0; q0 < W; q0 += CQ) {
            const int64_t n = std::min(CQ, W - q0);
            std::set<int32_t> us;
            for (int64_t t = 0; t < n; ++t) for (int32_t c : sel.cells[(size_t) (q0 + t)]) us.insert(c);
            std::vector<int32_t> U(us.begin(), us.end());
            std::map<int32_t, int32_t> at;
            for (size_t i = 0; i < U.size(); ++i) at[U[i]] = (int32_t) i;
            std::vector<float> m((size_t) (U.size() * n), -INFINITY);
            for (int64_t t = 0; t < n; ++t) for (int32_t c : sel.cells[(size_t) (q0 + t)]) m[(size_t) (t * U.size() + at[c])] = 0.0f;
            const int64_t nu = (int64_t) U.size();
            ggml_tensor * ui = g.i32(U);
            auto expand = [&](ggml_tensor * t2) {   // [hd*HK, nu] -> [hd, H, nu] contiguous kv groups
                ggml_tensor * t = ggml_reshape_4d(g.c, t2, hd, 1, HK, nu);
                t = ggml_repeat_4d(g.c, t, hd, H / HK, HK, nu);
                return ggml_reshape_3d(g.c, t, hd, H, nu);
            };
            ggml_tensor * ku = expand(ggml_get_rows(g.c, K2, ui));
            ggml_tensor * vu = expand(ggml_get_rows(g.c, V2, ui));
            ggml_tensor * qc = ggml_view_3d(g.c, q, hd, H, n, q->nb[1], q->nb[2], (size_t) q0 * q->nb[2]);
            ggml_tensor * qp = ggml_cont(g.c, ggml_permute(g.c, qc, 0, 2, 1, 3));      // [hd, n, H]
            ggml_tensor * kpp = ggml_cont(g.c, ggml_permute(g.c, ku, 0, 2, 1, 3));    // [hd, nu, H]
            ggml_tensor * vt = ggml_cont(g.c, ggml_permute(g.c, vu, 1, 2, 0, 3));     // [nu, hd, H]
            ggml_tensor * kq = ggml_mul_mat(g.c, kpp, qp);                             // [nu, n, H]
            ggml_tensor * mk = ggml_reshape_2d(g.c, g.f32(m.data(), nu, n), nu, n);
            kq = ggml_soft_max_ext(g.c, kq, mk, 1.0f / std::sqrt((float) hd), 0.0f);
            ggml_tensor * o = ggml_cont(g.c, ggml_permute(g.c, ggml_mul_mat(g.c, vt, kq), 0, 2, 1, 3));   // [hd, H, n]
            out = out ? ggml_concat(g.c, out, o, 2) : o;
        }
        out = ggml_mul(g.c, out, ggml_sigmoid(g.c, ggml_cont(g.c, gate)));
        Lora * lo = &L.qsa_lora.o;
        return lin(g.c, p.get("blk." + std::to_string((long long) L.l) + ".attn_output.weight"),
                   ggml_reshape_2d(g.c, out, hd * H, W), lo, ls);
    };

    // one window layer: x [D,hc,W] -> out; with `gout` it is the VJP graph (adapters of this layer as params)
    struct LayerIO { std::vector<float> out, gx; std::vector<int32_t> ids; };
    auto window_layer = [&](int64_t l, const std::vector<float> & x, int64_t W, const Carry & cy, const Sel * sel,
                            const std::vector<int32_t> * ids_in, const std::vector<float> * gout, bool want_gx, LayerIO & io) -> bool {
        G1 g;
        ggml_tensor * xin = g.f32(x.data(), G.D, G.hc, W);
        if (gout && want_gx) ggml_set_param(xin);
        TopLayer L = qfn_top_layer(g.c, p, l, G, g.f32(idv.data(), kGdnChunk, kGdnChunk));
        wire(L);
        ggml_tensor * pos = g.i32(posv(P, W));
        ggml_tensor * inj = nullptr;
        ggml_tensor * mixed = qfn_hc_mix(g.c, xin, L.hc_attn, G.hc, G.eps, &inj);
        ggml_tensor * att;
        if (L.qsa) att = qsa_window_block(g, mixed, L, pos, cy, *sel, W);
        else { set_carry(g, L, cy); att = qfn_gdn_block(g.c, mixed, L.gdn, G.hd, G.nk, G.nv, G.d_conv, G.eps, &L.gdn_carry); }
        ggml_tensor * res1 = qfn_hc_combine(g.c, xin, att, inj, G.hc);
        ggml_tensor * inj2 = nullptr;
        ggml_tensor * m2 = qfn_hc_mix(g.c, res1, L.hc_ffn, G.hc, G.eps, &inj2);
        const std::string b = "blk." + std::to_string((long long) l) + ".";
        if (!ids_in) {   // routing pass: just the router logits
            ggml_tensor * lg = ggml_mul_mat(g.c, p.get(b + "ffn_gate_inp.weight"), m2);
            if (!g.run(be, nullptr, { lg })) return false;
            io.ids = route(get(lg), W);
            return true;
        }
        MoeIndex ix = moe_index(g, *ids_in, W);
        ggml_tensor * y = qfn_hc_combine(g.c, res1, qfn_ffn(g.c, m2, p, l, ix), inj2, G.hc);
        if (!gout) {
            if (!g.run(be, nullptr, { y })) return false;
            io.out = get(y);
            return true;
        }
        std::vector<ggml_tensor *> want;
        for (auto & a : ads) if (a.layer == l) { ggml_set_param(a.A); ggml_set_param(a.B); want.push_back(a.A); want.push_back(a.B); }
        if (want_gx) want.push_back(xin);
        ggml_tensor * loss = ggml_sum(g.c, ggml_mul(g.c, y, g.f32(gout->data(), G.D, G.hc, W)));
        const bool ok = g.run(be, loss, {}, want);
        if (ok) {
            if (want_gx) io.gx = get(ggml_graph_get_grad(g.gf, xin));
            for (auto & a : ads) if (a.layer == l) {
                ggml_tensor * ga = ggml_graph_get_grad(g.gf, a.A), * gb = ggml_graph_get_grad(g.gf, a.B);
                if (ga) { auto v = get(ga); a.ga.resize(v.size(), 0); for (size_t i = 0; i < v.size(); ++i) a.ga[i] += v[i]; }
                if (gb) { auto v = get(gb); a.gb.resize(v.size(), 0); for (size_t i = 0; i < v.size(); ++i) a.gb[i] += v[i]; }
            }
        }
        for (auto & a : ads) if (a.layer == l) { a.A->flags &= ~GGML_TENSOR_FLAG_PARAM; a.B->flags &= ~GGML_TENSOR_FLAG_PARAM; }
        return ok;
    };

    // prefix carries with the current adapters (no gradient)
    std::vector<Carry> pre((size_t) (G.NL - L0));
    auto compute_prefix = [&]() -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> cur(prefix_x44), next;
        for (int64_t l = L0; l < G.NL; ++l) {
            Carry & cy = pre[(size_t) (l - L0)];
            cy = Carry{};
            if (G.is_qsa(l)) {   // K/V only
                for (int64_t p0 = 0; p0 < P; p0 += CH) {
                    const int64_t n = std::min(CH, P - p0);
                    G1 g;
                    ggml_tensor * x = g.f32(&cur[(size_t) (p0 * row)], G.D, G.hc, n);
                    TopLayer L = qfn_top_layer(g.c, p, l, G, g.f32(idv.data(), kGdnChunk, kGdnChunk));
                    wire(L);
                    ggml_tensor * mixed = qfn_hc_mix(g.c, x, L.hc_attn, G.hc, G.eps, nullptr);
                    ggml_tensor *k, *v;
                    qfn_qsa_qkv(g.c, mixed, p, l, G, g.i32(posv(p0, n)), &L.qsa_lora, ls, nullptr, nullptr, &k, &v);
                    if (!g.run(be, nullptr, { k, v })) return false;
                    auto kv = get(k), vv = get(v);
                    cy.k.insert(cy.k.end(), kv.begin(), kv.end());
                    cy.v.insert(cy.v.end(), vv.begin(), vv.end());
                    cy.nkv += n;
                }
                continue;
            }
            next.assign(cur.size(), 0.0f);
            for (int64_t p0 = 0; p0 < P; p0 += CH) {
                const int64_t n = std::min(CH, P - p0);
                std::vector<int32_t> ids;
                for (int phase = 0; phase < 2; ++phase) {
                    G1 g;
                    ggml_tensor * x = g.f32(&cur[(size_t) (p0 * row)], G.D, G.hc, n);
                    TopLayer L = qfn_top_layer(g.c, p, l, G, g.f32(idv.data(), kGdnChunk, kGdnChunk));
                    wire(L);
                    set_carry(g, L, cy);
                    ggml_tensor *m2, *inj2;
                    ggml_tensor * res1 = qfn_attn_half(g.c, x, L, p, G, g.i32(posv(p0, n)), ls, &m2, &inj2, true);
                    if (phase == 0) {
                        ggml_tensor * lg = ggml_mul_mat(g.c, p.get("blk." + std::to_string((long long) l) + ".ffn_gate_inp.weight"), m2);
                        if (!g.run(be, nullptr, { lg })) return false;
                        ids = route(get(lg), n);
                    } else {
                        MoeIndex ix = moe_index(g, ids, n);
                        ggml_tensor * y = qfn_hc_combine(g.c, res1, qfn_ffn(g.c, m2, p, l, ix), inj2, G.hc);
                        if (!g.run(be, nullptr, { y, L.gdn_carry.conv_out, L.gdn_carry.s_out })) return false;
                        auto yv = get(y);
                        std::memcpy(&next[(size_t) (p0 * row)], yv.data(), yv.size() * 4);
                        to_host(L.gdn_carry.conv_out, cy.conv);
                        to_host(L.gdn_carry.s_out, cy.s);
                    }
                }
            }
            cur.swap(next);
        }
        std::printf("[train] prefix carries (%lld positions, current adapters) in %.1f s\n", (long long) P,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return true;
    };

    // ---- one window: forward (+ CE) and optionally backward. Returns mean CE over its loss tokens.
    auto load_sel = [&](const Window & w, int64_t W) {
        Sel s;
        s.cells.resize((size_t) W);
        FILE * f = std::fopen((w.dir + "/qsa.bin").c_str(), "rb");
        int32_t h[4];
        while (f && std::fread(h, 4, 4, f) == 4) {
            std::vector<int32_t> ids((size_t) h[2] * h[3]);
            if (std::fread(ids.data(), 4, ids.size(), f) != ids.size()) break;
            for (int32_t t = 0; t < h[2]; ++t) {
                const int64_t wp = h[1] + t - P;
                if (wp >= 0 && wp < W) s.cells[(size_t) wp].assign(ids.begin() + (size_t) t * h[3], ids.begin() + (size_t) (t + 1) * h[3]);
            }
        }
        if (f) std::fclose(f);
        return s;
    };
    auto step_window = [&](const Window & w, bool train, double & ce_out, int & n_out) -> bool {
        std::vector<float> x = read_bf16(w.dir + "/x44.bf16");
        const int64_t W = (int64_t) (x.size() / row);
        const Sel sel = load_sel(w, W);
        std::vector<std::vector<float>> xs((size_t) (G.NL - L0 + 1));
        std::vector<std::vector<int32_t>> ids((size_t) (G.NL - L0));
        xs[0] = x;
        for (int64_t l = L0; l < G.NL; ++l) {
            LayerIO r, f;
            const Carry & cy = pre[(size_t) (l - L0)];
            if (!window_layer(l, xs[(size_t) (l - L0)], W, cy, &sel, nullptr, nullptr, false, r)) return false;
            ids[(size_t) (l - L0)] = r.ids;
            if (!window_layer(l, xs[(size_t) (l - L0)], W, cy, &sel, &r.ids, nullptr, false, f)) return false;
            xs[(size_t) (l - L0 + 1)] = std::move(f.out);
        }
        // head: CE over positions t whose NEXT token is a loss token
        std::vector<int32_t> lp;
        for (int64_t t = 0; t + 1 < W; ++t) if (w.mask[(size_t) t + 1]) lp.push_back((int32_t) t);
        const int64_t NLp = (int64_t) lp.size();
        std::vector<float> g48(train ? (size_t) (W * row) : 0, 0.0f);
        double ce = 0;
        for (int64_t c0 = 0; c0 < NLp; c0 += CL) {
            const int64_t n = std::min(CL, NLp - c0);
            G1 g;
            ggml_tensor * xh = g.f32(xs.back().data(), G.D, G.hc, W);
            if (train) ggml_set_param(xh);
            // the head mixer runs per position: only the rows of this chunk
            std::vector<int32_t> rows(lp.begin() + c0, lp.begin() + c0 + n);
            ggml_tensor * xr = ggml_reshape_3d(g.c, ggml_get_rows(g.c, ggml_reshape_2d(g.c, xh, row, W), g.i32(rows)), G.D, G.hc, n);
            ggml_tensor * lg = qfn_head(g.c, xr, p, G);                                   // [V, n]
            std::vector<float> oh((size_t) (G.n_vocab * n), 0.0f);
            for (int64_t t = 0; t < n; ++t) oh[(size_t) (t * G.n_vocab + w.tok[(size_t) rows[(size_t) t] + 1])] = 1.0f;
            ggml_tensor * loss = ggml_scale(g.c, ggml_cross_entropy_loss(g.c, lg, g.f32(oh.data(), G.n_vocab, n)),
                                            (float) n / (float) NLp);                     // chunk mean -> window mean
            if (!g.run(be, train ? loss : nullptr, { loss }, train ? std::vector<ggml_tensor *>{ xh } : std::vector<ggml_tensor *>{})) return false;
            ce += get(loss)[0];
            if (train) {
                auto gv = get(ggml_graph_get_grad(g.gf, xh));
                for (size_t i = 0; i < gv.size(); ++i) g48[i] += gv[i];
            }
        }
        ce_out = ce;
        n_out = (int) NLp;
        if (!train) return true;
        std::vector<float> gout = std::move(g48);
        for (int64_t l = G.NL - 1; l >= L0; --l) {
            LayerIO r;
            if (!window_layer(l, xs[(size_t) (l - L0)], W, pre[(size_t) (l - L0)], &sel, &ids[(size_t) (l - L0)], &gout, l > L0, r))
                return false;
            if (l > L0) gout = std::move(r.gx);
        }
        return true;
    };

    // ---- run
    const std::string runs = env("QFN_RUNS", "runs");
    char cfg[1024];
    std::snprintf(cfg, sizeof cfg, "{\"model\": \"Swift1.5-QFN-XQ-45GB\", \"layers\": \"%lld-%lld\", \"rank\": %lld, \"alpha\": %.0f, "
                  "\"lr\": %g, \"warmup\": %d, \"steps\": %d, \"train_windows\": %zu, \"eval_windows\": %zu, \"rope_scale\": \"%s\", "
                  "\"prefix_every\": %d, \"clip\": %g}", (long long) L0, (long long) G.NL - 1, (long long) rank, alpha, lr0, warmup, steps,
                  wins.size(), evals.size(), env("QFN_ROPE_SCALE", "2"), prefix_every, clip);
    Metrics M(runs, run, cfg);
    auto evaluate = [&]() {
        double s = 0; long n = 0;
        for (auto & w : evals) { double ce; int k; if (step_window(w, false, ce, k)) { s += ce * k; n += k; } }
        return n ? s / n : NAN;
    };
    auto save = [&](const std::string & path) {
        std::vector<LoraTensorPair> out;
        for (auto & a : ads) { LoraTensorPair pr; pr.name = a.name; pr.n_in = a.n_in; pr.n_out = a.n_out; pr.rank = rank; pr.A = a.a; pr.B = a.b; out.push_back(pr); }
        std::string err;
        if (!write_lora_gguf(path, "qwen4exp", alpha, out, err)) std::fprintf(stderr, "save: %s\n", err.c_str());
    };

    if (!compute_prefix()) return 1;
    const double ev0 = evaluate();
    std::printf("[train] step 0: eval CE %.4f (the base model, adapters at identity)\n", ev0);
    M.log(0, { { "eval_loss", ev0 }, { "lr", 0.0 }, { "tokens", 0.0 } });
    std::shuffle(wins.begin(), wins.end(), rng);
    long tokens = 0;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    for (int step = 1; step <= steps; ++step) {
        const auto ts = std::chrono::steady_clock::now();
        if (prefix_every > 0 && step > 1 && (step - 1) % prefix_every == 0 && !compute_prefix()) return 1;
        const Window & w = wins[(size_t) ((step - 1) % wins.size())];
        for (auto & a : ads) { a.ga.assign(a.a.size(), 0.0f); a.gb.assign(a.b.size(), 0.0f); }
        double ce; int nl;
        if (!step_window(w, true, ce, nl)) { std::fprintf(stderr, "step %d failed (%s)\n", step, w.dir.c_str()); return 1; }
        tokens += nl;
        double gn = 0;
        for (auto & a : ads) { for (float g : a.ga) gn += (double) g * g; for (float g : a.gb) gn += (double) g * g; }
        gn = std::sqrt(gn);
        const float sc = gn > clip ? (float) (clip / gn) : 1.0f;
        const float lr = step <= warmup ? lr0 * step / warmup
                                        : lr0 * (0.1f + 0.9f * 0.5f * (1.0f + std::cos(3.14159265f * (step - warmup) / std::max(1, steps - warmup))));
        const float bc1 = 1.0f - std::pow(b1, (float) step), bc2 = 1.0f - std::pow(b2, (float) step);
        auto adam = [&](std::vector<float> & w_, std::vector<float> & g_, std::vector<float> & m_, std::vector<float> & v_) {
            for (size_t i = 0; i < w_.size(); ++i) {
                const float g = g_[i] * sc;
                m_[i] = b1 * m_[i] + (1 - b1) * g;
                v_[i] = b2 * v_[i] + (1 - b2) * g * g;
                w_[i] -= lr * (m_[i] / bc1) / (std::sqrt(v_[i] / bc2) + eps);
            }
        };
        for (auto & a : ads) { adam(a.a, a.ga, a.ma, a.va); adam(a.b, a.gb, a.mb, a.vb); }
        upload();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        std::vector<std::pair<const char *, double>> kv = { { "loss", ce }, { "lr", lr }, { "gnorm", gn }, { "tokens", (double) tokens },
                                                            { "step_s", secs }, { "window_tokens", (double) nl } };
        double ev = NAN;
        if (eval_every > 0 && step % eval_every == 0) { ev = evaluate(); kv.push_back({ "eval_loss", ev }); }
        M.log(step, kv);
        std::printf("[train] step %d/%d  loss %.4f  gnorm %.3f  lr %.2e  %d tok  %.1f s%s\n", step, steps, ce, gn, lr, nl, secs,
                    std::isfinite(ev) ? (" | eval " + std::to_string(ev)).c_str() : "");
        std::fflush(stdout);
        if (save_every > 0 && step % save_every == 0) save(M.dir + "/adapter-step" + std::to_string(step) + ".gguf");
    }
    save(M.dir + "/adapter.gguf");
    std::printf("[train] done: adapter %s/adapter.gguf\n", M.dir.c_str());
    ggml_backend_buffer_free(lbuf);
    ggml_free(lctx);
    return 0;
}
