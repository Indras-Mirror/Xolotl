// qfn_train <layer.f32> <embed.f32> <out.gguf> [steps] [T] [lr] [rank]
//
// LoRA training run on REAL Qwen3.8-Flash-Next weights (one gated-delta-net layer exported by qfn_export.py).
// Frozen F32 base, adapters on blk.<l>.attn_qkv.weight (D -> 10240) and blk.<l>.ssm_out.weight (6144 -> D),
// loss = sum((layer(x_a) - target)^2), target = the same layer's output on a DIFFERENT real input x_b: a
// memorisation gate in the style of the Hy4 T1 run. AdamW on the host, 1 step = forward + backward + update.
// Prints the loss curve, then saves a llama.cpp-format LoRA GGUF and verifies by reloading it into a fresh graph
// (adapted output vs target, and the un-adapted base vs target for contrast).
#include "ggml.h"
#include "ggml-cpu.h"
#include "qfn_layer_train.hpp"
#include "lora_gguf.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace xolotl::train;

static std::vector<float> read_f32(const char * path, size_t n, size_t skip = 0) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    fseek(f, (long) (skip * 4), SEEK_SET);
    if (fread(v.data(), 4, n, f) != n) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    fclose(f);
    return v;
}

struct Trainable {
    std::vector<float> a, b;          // host copies: A [n_in, r], B [r, n_out]
    std::vector<float> ma, va, mb, vb;
    int64_t n_in = 0, n_out = 0;
    std::string name;
};

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <layer.f32> <embed.f32> <out.gguf> [steps] [T] [lr] [rank]\n", argv[0]); return 2; }
    const char * wpath = argv[1], * epath = argv[2], * gpath = argv[3];
    const int steps = argc > 4 ? atoi(argv[4]) : 100;
    const int T = argc > 5 ? atoi(argv[5]) : 16;
    const float lr = argc > 6 ? (float) atof(argv[6]) : 3e-4f;
    const int64_t rank = argc > 7 ? atoll(argv[7]) : 8;
    const int64_t layer = argc > 8 ? atoll(argv[8]) : 0;
    const float alpha = 16.0f;

    // ---- header is at the END of the export: 5 x int32 ----
    int hdr[5] = { 0, 0, 0, 0, 0 };
    FILE * f = fopen(wpath, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", wpath); return 1; }
    fseek(f, -20, SEEK_END);
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { std::fprintf(stderr, "bad header\n"); return 1; }
    fclose(f);
    const int64_t D = hdr[0], hd = hdr[1], nk = hdr[2], nv = hdr[3], d_conv = hdr[4];
    const int64_t n_qkv = 2 * hd * nk + hd * nv, n_z = hd * nv;
    std::printf("layer %lld: D %lld hd %lld nk %lld nv %lld d_conv %lld (channels %lld, z %lld)\n",
                (long long) layer, (long long) D, (long long) hd, (long long) nk, (long long) nv,
                (long long) d_conv, (long long) n_qkv, (long long) n_z);

    size_t off = 0;
    auto take = [&](size_t n) {
        auto v = read_f32(wpath, n, off);
        off += n;
        return v;
    };
    const auto wqkv  = take((size_t) D * n_qkv);          // attn_qkv [D, 10240]
    const auto wgate = take((size_t) D * n_z);            // attn_gate [D, 6144]
    const auto wconv = take((size_t) d_conv * n_qkv);     // ssm_conv1d
    const auto wbeta = take((size_t) D * nv);             // ssm_beta
    const auto walpha = take((size_t) D * nv);            // ssm_alpha
    const auto ssm_a  = take((size_t) nv);
    const auto dtv    = take((size_t) nv);
    const auto nrm    = take((size_t) hd);
    const auto wout   = take((size_t) hd * nv * D);       // ssm_out [6144, D]

    std::vector<float> w_ba((size_t) D * 2 * nv);
    for (int64_t i = 0; i < D; ++i) {
        std::memcpy(&w_ba[(size_t) i * 2 * nv],       &wbeta[(size_t) i * nv],  (size_t) nv * 4);
        std::memcpy(&w_ba[(size_t) i * 2 * nv + nv],  &walpha[(size_t) i * nv], (size_t) nv * 4);
    }

    // real inputs: two different token-embedding windows
    const auto emb = read_f32(epath, (size_t) D * 2 * T);
    std::vector<float> xa(emb.begin(), emb.begin() + (size_t) D * T);
    std::vector<float> xb(emb.begin() + (size_t) D * T, emb.end());
    std::printf("inputs %lld x %d tokens from %s\n", (long long) D, T, epath);

    ggml_init_params ip = { 1200ull << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    if (!c) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    auto mk = [&](ggml_type t, int64_t n0, int64_t n1) { return ggml_new_tensor_2d(c, t, n0, n1); };
    auto mkv = [&](int64_t n) { return ggml_new_tensor_1d(c, GGML_TYPE_F32, n); };
    QfnLayer w;
    w.wqkv   = mk(GGML_TYPE_F32, D, n_qkv);
    w.wgate  = mk(GGML_TYPE_F32, D, n_z);
    w.conv_w = mk(GGML_TYPE_F32, d_conv, n_qkv);
    w.w_ba   = mk(GGML_TYPE_F32, D, 2 * nv);
    w.ssm_a  = mkv(nv); w.dt = mkv(nv); w.norm_w = mkv(hd);
    w.w_out  = mk(GGML_TYPE_F32, hd * nv, D);
    w.ident  = mk(GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    w.lora_alpha = alpha; w.lora_rank = rank;
    w.lora_a_qkv  = mk(GGML_TYPE_F32, D, rank);  w.lora_b_qkv  = mk(GGML_TYPE_F32, rank, n_qkv);
    w.lora_a_gate = mk(GGML_TYPE_F32, D, rank);  w.lora_b_gate = mk(GGML_TYPE_F32, rank, n_z);
    w.lora_a_out = mk(GGML_TYPE_F32, hd * nv, rank); w.lora_b_out = mk(GGML_TYPE_F32, rank, D);
    memcpy(w.wqkv->data, wqkv.data(), wqkv.size() * 4);
    memcpy(w.wgate->data, wgate.data(), wgate.size() * 4);
    memcpy(w.conv_w->data, wconv.data(), wconv.size() * 4);
    memcpy(w.w_ba->data, w_ba.data(), w_ba.size() * 4);
    memcpy(w.ssm_a->data, ssm_a.data(), ssm_a.size() * 4);
    memcpy(w.dt->data, dtv.data(), dtv.size() * 4);
    memcpy(w.norm_w->data, nrm.data(), nrm.size() * 4);
    memcpy(w.w_out->data, wout.data(), wout.size() * 4);
    std::vector<float> ident((size_t) kGdnChunk * kGdnChunk, 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) ident[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
    memcpy(w.ident->data, ident.data(), ident.size() * 4);

    // adapters: A ~ U(+-1/sqrt(in)), B = 0 (llama.cpp's init: the model is unchanged at step 0)
    std::vector<Trainable> tr(3);
    tr[0].n_in = D;       tr[0].n_out = n_qkv;        tr[0].name = "blk." + std::to_string(layer) + ".attn_qkv.weight";
    tr[1].n_in = D;       tr[1].n_out = n_z;          tr[1].name = "blk." + std::to_string(layer) + ".attn_gate.weight";
    tr[2].n_in = hd * nv; tr[2].n_out = D;            tr[2].name = "blk." + std::to_string(layer) + ".ssm_out.weight";
    {
        unsigned s = 12345;
        auto rnd = [&]() { s = s * 1103515245u + 12345u; return ((float) ((s >> 8) & 0xffff) / 32768.0f) - 1.0f; };
        for (auto & t : tr) {
            const float sc = 1.0f / std::sqrt((float) t.n_in);
            t.a.resize((size_t) t.n_in * (size_t) rank);
            for (auto & v : t.a) v = sc * rnd();
            t.b.assign((size_t) rank * (size_t) t.n_out, 0.0f);
            t.ma.assign(t.a.size(), 0.0f); t.va.assign(t.a.size(), 0.0f);
            t.mb.assign(t.b.size(), 0.0f); t.vb.assign(t.b.size(), 0.0f);
        }
    }
    auto upload = [&]() {
        if (!w.lora_a_qkv) return;
        memcpy(w.lora_a_qkv->data,  tr[0].a.data(), tr[0].a.size() * 4);
        memcpy(w.lora_b_qkv->data,  tr[0].b.data(), tr[0].b.size() * 4);
        memcpy(w.lora_a_gate->data, tr[1].a.data(), tr[1].a.size() * 4);
        memcpy(w.lora_b_gate->data, tr[1].b.data(), tr[1].b.size() * 4);
        memcpy(w.lora_a_out->data,  tr[2].a.data(), tr[2].a.size() * 4);
        memcpy(w.lora_b_out->data,  tr[2].b.data(), tr[2].b.size() * 4);
    };
    upload();

    // ---- target: the same layer on the other input window ----
    ggml_tensor * txa = mk(GGML_TYPE_F32, D, T);
    ggml_tensor * txb = mk(GGML_TYPE_F32, D, T);
    memcpy(txa->data, xa.data(), xa.size() * 4);
    memcpy(txb->data, xb.data(), xb.size() * 4);
    std::vector<float> target;
    {
        w.lora_a_qkv = nullptr; w.lora_b_qkv = nullptr;   // base output for the target
        w.lora_a_gate = nullptr; w.lora_b_gate = nullptr;
        w.lora_a_out = nullptr; w.lora_b_out = nullptr;
        ggml_tensor * o = qfn_gdn_layer(c, txb, w, hd, nk, nv, d_conv, 1e-6f);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(c, g, 16);
        target.assign((float *) o->data, (float *) o->data + (size_t) D * T);
        w.lora_a_qkv  = mk(GGML_TYPE_F32, D, rank);  w.lora_b_qkv  = mk(GGML_TYPE_F32, rank, n_qkv);
    w.lora_a_gate = mk(GGML_TYPE_F32, D, rank);  w.lora_b_gate = mk(GGML_TYPE_F32, rank, n_z);
        w.lora_a_out = mk(GGML_TYPE_F32, hd * nv, rank); w.lora_b_out = mk(GGML_TYPE_F32, rank, D);
        upload();
        double n2 = 0;
        for (float v : target) n2 += (double) v * v;
        std::printf("target |o| = %.4f (rms %.4f)\n", std::sqrt(n2), std::sqrt(n2 / (double) target.size()));
    }
    ggml_tensor * tgt = mk(GGML_TYPE_F32, D, T);
    memcpy(tgt->data, target.data(), target.size() * 4);

    // ---- the training graph: rebuilt each step so a fresh context allocates cleanly ----
    const float b1 = 0.9f, b2 = 0.999f, aeps = 1e-8f, wd = 0.0f;
    std::vector<float> gA[3], gB[3];
    float loss0 = 0;
    for (int step = 0; step <= steps; ++step) {
        ggml_init_params pi = { 1200ull << 20, nullptr, false };
        ggml_context * sc = ggml_init(pi);
        QfnLayer sw = w;
        sw.wqkv  = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, n_qkv);
        sw.wgate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, n_z);
        sw.conv_w = ggml_new_tensor_2d(sc, GGML_TYPE_F32, d_conv, n_qkv);
        sw.w_ba = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, 2 * nv);
        sw.ssm_a = ggml_new_tensor_1d(sc, GGML_TYPE_F32, nv);
        sw.dt = ggml_new_tensor_1d(sc, GGML_TYPE_F32, nv);
        sw.norm_w = ggml_new_tensor_1d(sc, GGML_TYPE_F32, hd);
        sw.w_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, hd * nv, D);
        sw.ident = ggml_new_tensor_2d(sc, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
        sw.lora_a_qkv = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, rank);
        sw.lora_b_qkv = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, n_qkv);
        sw.lora_a_gate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, rank);
        sw.lora_b_gate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, n_z);
        sw.lora_a_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, hd * nv, rank);
        sw.lora_b_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, D);
        memcpy(sw.wqkv->data, wqkv.data(), wqkv.size() * 4);
        memcpy(sw.wgate->data, wgate.data(), wgate.size() * 4);
        memcpy(sw.conv_w->data, wconv.data(), wconv.size() * 4);
        memcpy(sw.w_ba->data, w_ba.data(), w_ba.size() * 4);
        memcpy(sw.ssm_a->data, ssm_a.data(), ssm_a.size() * 4);
        memcpy(sw.dt->data, dtv.data(), dtv.size() * 4);
        memcpy(sw.norm_w->data, nrm.data(), nrm.size() * 4);
        memcpy(sw.w_out->data, wout.data(), wout.size() * 4);
        memcpy(sw.ident->data, ident.data(), ident.size() * 4);
        memcpy(sw.lora_a_qkv->data,  tr[0].a.data(), tr[0].a.size() * 4);
        memcpy(sw.lora_b_qkv->data,  tr[0].b.data(), tr[0].b.size() * 4);
        memcpy(sw.lora_a_gate->data, tr[1].a.data(), tr[1].a.size() * 4);
        memcpy(sw.lora_b_gate->data, tr[1].b.data(), tr[1].b.size() * 4);
        memcpy(sw.lora_a_out->data,  tr[2].a.data(), tr[2].a.size() * 4);
        memcpy(sw.lora_b_out->data,  tr[2].b.data(), tr[2].b.size() * 4);
        ggml_tensor * x = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, T);
        ggml_tensor * tt = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, T);
        memcpy(x->data, xa.data(), xa.size() * 4);
        memcpy(tt->data, target.data(), target.size() * 4);
        ggml_set_param(sw.lora_a_qkv); ggml_set_param(sw.lora_b_qkv);
        ggml_set_param(sw.lora_a_gate); ggml_set_param(sw.lora_b_gate);
        ggml_set_param(sw.lora_a_out); ggml_set_param(sw.lora_b_out);
        ggml_tensor * o = qfn_gdn_layer(sc, x, sw, hd, nk, nv, d_conv, 1e-6f);
        ggml_tensor * d = ggml_sub(sc, o, tt);
        ggml_tensor * loss = ggml_sum(sc, ggml_mul(sc, d, d));
        ggml_set_loss(loss);
        ggml_cgraph * gf = ggml_new_graph_custom(sc, 262144, true);
        ggml_build_forward_expand(gf, loss);
        ggml_build_backward_expand(sc, gf, nullptr);
        ggml_graph_reset(gf);
        ggml_graph_compute_with_ctx(sc, gf, 16);
        const float L = ((float *) loss->data)[0];
        if (step == 0) loss0 = L;
        if (step % 10 == 0 || step == steps) {
            double rel = std::sqrt((double) L / ((double) D * T));
            std::printf("step %4d  loss %.6f  rms_err %.5f\n", step, L, rel);
            std::fflush(stdout);
        }
        if (step == steps) { ggml_free(sc); break; }
        auto grab = [&](ggml_tensor * p, std::vector<float> * dst) {
            ggml_tensor * g = ggml_graph_get_grad(gf, p);
            dst->assign((float *) g->data, (float *) g->data + ggml_nelements(p));
        };
        grab(sw.lora_a_qkv, &gA[0]); grab(sw.lora_b_qkv, &gB[0]);
        grab(sw.lora_a_gate, &gA[1]); grab(sw.lora_b_gate, &gB[1]);
        grab(sw.lora_a_out, &gA[2]); grab(sw.lora_b_out, &gB[2]);
        // AdamW
        const float bc1 = 1.0f - std::pow(b1, (float) (step + 1)), bc2 = 1.0f - std::pow(b2, (float) (step + 1));
        for (int t = 0; t < 3; ++t) {
            auto upd = [&](std::vector<float> & prm, std::vector<float> & m, std::vector<float> & v,
                           const std::vector<float> & g) {
                for (size_t i = 0; i < prm.size() && i < g.size(); ++i) {
                    m[i] = b1 * m[i] + (1 - b1) * g[i];
                    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
                    const float mh = m[i] / bc1, vh = v[i] / bc2;
                    prm[i] -= lr * (mh / (std::sqrt(vh) + aeps) + wd * prm[i]);
                }
            };
            upd(tr[t].a, tr[t].ma, tr[t].va, gA[t]);
            upd(tr[t].b, tr[t].mb, tr[t].vb, gB[t]);
        }
        ggml_free(sc);
    }

    // ---- save the adapter GGUF and verify it round-trips into a fresh graph ----
    std::vector<LoraTensorPair> ad(tr.size());
    for (size_t i = 0; i < tr.size(); ++i) {
        ad[i].name = tr[i].name; ad[i].n_in = tr[i].n_in; ad[i].n_out = tr[i].n_out; ad[i].rank = rank;
        ad[i].A = tr[i].a; ad[i].B = tr[i].b;
    }
    std::string err;
    if (!write_lora_gguf(gpath, "qwen4exp", alpha, ad, err)) { std::fprintf(stderr, "save: %s\n", err.c_str()); return 1; }
    std::printf("saved %s (%zu adapters, rank %lld, alpha %.0f)\n", gpath, ad.size(), (long long) rank, alpha);

    std::vector<LoraTensorPair> back;
    float fa = 0; std::string arch;
    if (!read_lora_gguf(gpath, back, fa, arch, err)) { std::fprintf(stderr, "reload: %s\n", err.c_str()); return 1; }
    double dmax = 0;
    for (size_t i = 0; i < back.size(); ++i) {
        for (size_t k = 0; k < back[i].A.size(); ++k) dmax = std::fmax(dmax, std::fabs(back[i].A[k] - ad[i].A[k]));
        for (size_t k = 0; k < back[i].B.size(); ++k) dmax = std::fmax(dmax, std::fabs(back[i].B[k] - ad[i].B[k]));
    }
    std::printf("reload: %zu adapters, alpha %.1f, arch %s, max |diff| %.3e\n", back.size(), fa, arch.c_str(), dmax);

    // adapted vs base, both from the reloaded file
    auto eval = [&](bool adapted) {
        ggml_init_params pi = { 1200ull << 20, nullptr, false };
        ggml_context * sc = ggml_init(pi);
        QfnLayer sw;
        sw.wqkv  = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, n_qkv);
        sw.wgate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, n_z);
        sw.conv_w = ggml_new_tensor_2d(sc, GGML_TYPE_F32, d_conv, n_qkv);
        sw.w_ba = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, 2 * nv);
        sw.ssm_a = ggml_new_tensor_1d(sc, GGML_TYPE_F32, nv);
        sw.dt = ggml_new_tensor_1d(sc, GGML_TYPE_F32, nv);
        sw.norm_w = ggml_new_tensor_1d(sc, GGML_TYPE_F32, hd);
        sw.w_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, hd * nv, D);
        sw.ident = ggml_new_tensor_2d(sc, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
        memcpy(sw.wqkv->data, wqkv.data(), wqkv.size() * 4);
        memcpy(sw.wgate->data, wgate.data(), wgate.size() * 4);
        memcpy(sw.conv_w->data, wconv.data(), wconv.size() * 4);
        memcpy(sw.w_ba->data, w_ba.data(), w_ba.size() * 4);
        memcpy(sw.ssm_a->data, ssm_a.data(), ssm_a.size() * 4);
        memcpy(sw.dt->data, dtv.data(), dtv.size() * 4);
        memcpy(sw.norm_w->data, nrm.data(), nrm.size() * 4);
        memcpy(sw.w_out->data, wout.data(), wout.size() * 4);
        memcpy(sw.ident->data, ident.data(), ident.size() * 4);
        if (adapted) {
            sw.lora_a_qkv = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, rank);
            sw.lora_b_qkv = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, n_qkv);
            sw.lora_a_gate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, rank);
            sw.lora_b_gate = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, n_z);
            sw.lora_a_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, hd * nv, rank);
            sw.lora_b_out = ggml_new_tensor_2d(sc, GGML_TYPE_F32, rank, D);
            sw.lora_alpha = fa; sw.lora_rank = rank;
            memcpy(sw.lora_a_qkv->data,  back[0].A.data(), back[0].A.size() * 4);
            memcpy(sw.lora_b_qkv->data,  back[0].B.data(), back[0].B.size() * 4);
            memcpy(sw.lora_a_gate->data, back[1].A.data(), back[1].A.size() * 4);
            memcpy(sw.lora_b_gate->data, back[1].B.data(), back[1].B.size() * 4);
            memcpy(sw.lora_a_out->data,  back[2].A.data(), back[2].A.size() * 4);
            memcpy(sw.lora_b_out->data,  back[2].B.data(), back[2].B.size() * 4);
        }
        ggml_tensor * x = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, T);
        memcpy(x->data, xa.data(), xa.size() * 4);
        ggml_tensor * o = qfn_gdn_layer(sc, x, sw, hd, nk, nv, d_conv, 1e-6f);
        ggml_cgraph * g = ggml_new_graph(sc);
        ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(sc, g, 16);
        double n2 = 0, m2 = 0;
        for (int64_t i = 0; i < D * T; ++i) {
            const double d = ((float *) o->data)[i] - target[(size_t) i];
            n2 += d * d;
            m2 += (double) target[(size_t) i] * target[(size_t) i];
        }
        ggml_free(sc);
        return std::sqrt(n2 / std::max(m2, 1e-30));
    };
    const double rel_base = eval(false), rel_adapted = eval(true);
    std::printf("verify: relative L2 vs target  base %.4f  adapted %.4f  (%.1fx)\n",
                rel_base, rel_adapted, rel_base / std::max(rel_adapted, 1e-12));
    std::printf("%s\n", rel_adapted < 0.5 * rel_base ? "TRAIN-RUN PASS" : "TRAIN-RUN WEAK (adapters did not move the layer enough)");
    (void) loss0;
    ggml_free(c);
    return 0;
}
