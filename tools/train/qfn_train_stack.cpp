// qfn_train_stack <layer0.f32> <layer1.f32> ... -- <embed.f32> <out.gguf> [steps] [T] [lr] [rank]
//
// LoRA training through a STACK of real Qwen3.8-Flash-Next gated-delta-net layers (qfn_export.py output).
// Frozen F32 base, rank-r adapters on every layer's attn_qkv / attn_gate / ssm_out, AdamW on the host.
// loss = ||stack(x_a) - stack_base(x_b)||^2 : the memorisation gate, now through depth (each layer's gradient has to
// travel back through the layers above it, which the single-layer run could not exercise).
// Saves one llama.cpp-format LoRA GGUF for the whole stack and verifies it by reloading into a fresh graph.
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

struct LayerW {
    std::vector<float> wqkv, wgate, conv, beta, alpha, a, dt, nrm, out;
    int64_t layer = 0;
};

static std::vector<float> read_f32(const char * path, size_t n, size_t skip = 0) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    if (skip) fseek(f, (long) (skip * 4), SEEK_SET);
    if (fread(v.data(), 4, n, f) != n) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    // split at "--"
    int sep = -1;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--")) { sep = i; break; }
    if (sep < 2 || argc < sep + 4) {
        std::fprintf(stderr, "usage: %s <layer0.f32> [layer1.f32 ...] -- <embed.f32> <out.gguf> [steps] [T] [lr] [rank]\n", argv[0]);
        return 2;
    }
    const int NL = sep - 1;
    const char * epath = argv[sep + 1], * gpath = argv[sep + 2];
    const int steps = argc > sep + 3 ? atoi(argv[sep + 3]) : 100;
    const int T = argc > sep + 4 ? atoi(argv[sep + 4]) : 8;
    const float lr = argc > sep + 5 ? (float) atof(argv[sep + 5]) : 3e-3f;
    const int64_t rank = argc > sep + 6 ? atoll(argv[sep + 6]) : 8;
    const float alpha = 16.0f;

    std::vector<LayerW> L((size_t) NL);
    int64_t D = 0, hd = 0, nk = 0, nv = 0, d_conv = 0;
    for (int i = 0; i < NL; ++i) {
        int hdr[5] = { 0, 0, 0, 0, 0 };
        FILE * f = fopen(argv[1 + i], "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1 + i]); return 1; }
        fseek(f, -20, SEEK_END);
        if (fread(hdr, sizeof(hdr), 1, f) != 1) { std::fprintf(stderr, "bad header %s\n", argv[1 + i]); return 1; }
        fclose(f);
        if (i == 0) { D = hdr[0]; hd = hdr[1]; nk = hdr[2]; nv = hdr[3]; d_conv = hdr[4]; }
        else if (hdr[0] != D || hdr[1] != hd || hdr[2] != nk || hdr[3] != nv || hdr[4] != d_conv) {
            std::fprintf(stderr, "layer %d geometry differs\n", i); return 1;
        }
        LayerW & w = L[(size_t) i];
        w.layer = i;
        size_t off = 0;
        auto take = [&](size_t n) { auto v = read_f32(argv[1 + i], n, off); off += n; return v; };
        const size_t n_qkv = (size_t) (2 * hd * nk + hd * nv), n_z = (size_t) (hd * nv);
        w.wqkv = take((size_t) D * n_qkv);
        w.wgate = take((size_t) D * n_z);
        w.conv = take((size_t) d_conv * n_qkv);
        w.beta = take((size_t) D * nv);
        w.alpha = take((size_t) D * nv);
        w.a = take((size_t) nv);
        w.dt = take((size_t) nv);
        w.nrm = take((size_t) hd);
        w.out = take((size_t) hd * nv * D);
    }
    const int64_t n_qkv = 2 * hd * nk + hd * nv, n_z = hd * nv;
    std::printf("%d layers: D %lld hd %lld nk %lld nv %lld d_conv %lld\n", NL,
                (long long) D, (long long) hd, (long long) nk, (long long) nv, (long long) d_conv);

    const auto emb = read_f32(epath, (size_t) D * 2 * T);
    std::vector<float> xa(emb.begin(), emb.begin() + (size_t) D * T);
    std::vector<float> xb(emb.begin() + (size_t) D * T, emb.end());

    struct Tr { std::vector<float> a, b, ma, va, mb, vb; int64_t n_in = 0, n_out = 0; std::string name; };
    std::vector<Tr> tr((size_t) NL * 3);
    const int64_t nin[3] = { D, D, hd * nv };
    const int64_t nout[3] = { n_qkv, n_z, D };
    const char * suf[3] = { ".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight" };
    {
        unsigned s = 12345;
        auto rnd = [&]() { s = s * 1103515245u + 12345u; return ((float) ((s >> 8) & 0xffff) / 32768.0f) - 1.0f; };
        for (int i = 0; i < NL; ++i)
            for (int j = 0; j < 3; ++j) {
                Tr & t = tr[(size_t) (i * 3 + j)];
                t.n_in = nin[j]; t.n_out = nout[j];
                t.name = "blk." + std::to_string(i) + suf[j];
                const float sc = 1.0f / std::sqrt((float) t.n_in);
                t.a.resize((size_t) t.n_in * (size_t) rank);
                for (auto & v : t.a) v = sc * rnd();
                t.b.assign((size_t) rank * (size_t) t.n_out, 0.0f);
                t.ma.assign(t.a.size(), 0.0f); t.va.assign(t.a.size(), 0.0f);
                t.mb.assign(t.b.size(), 0.0f); t.vb.assign(t.b.size(), 0.0f);
            }
    }

    // Builds the stack in context c. adapted=false -> base only.
    auto build = [&](ggml_context * c, ggml_tensor * x, bool adapted,
                     std::vector<ggml_tensor *> * la = nullptr, std::vector<ggml_tensor *> * lb = nullptr) {
        std::vector<float> ident((size_t) kGdnChunk * kGdnChunk, 0.0f);
        for (int i = 0; i < kGdnChunk; ++i) ident[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
        ggml_tensor * cur = x;
        for (int i = 0; i < NL; ++i) {
            const LayerW & w = L[(size_t) i];
            QfnLayer g;
            g.wqkv  = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_qkv);
            g.wgate = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_z);
            g.conv_w = ggml_new_tensor_2d(c, GGML_TYPE_F32, d_conv, n_qkv);
            g.w_ba  = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * nv);
            g.ssm_a = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
            g.dt    = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
            g.norm_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
            g.w_out = ggml_new_tensor_2d(c, GGML_TYPE_F32, hd * nv, D);
            g.ident = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
            memcpy(g.wqkv->data, w.wqkv.data(), w.wqkv.size() * 4);
            memcpy(g.wgate->data, w.wgate.data(), w.wgate.size() * 4);
            memcpy(g.conv_w->data, w.conv.data(), w.conv.size() * 4);
            std::vector<float> ba((size_t) D * 2 * nv);
            for (int64_t r = 0; r < D; ++r) {
                memcpy(&ba[(size_t) r * 2 * nv], &w.beta[(size_t) r * nv], (size_t) nv * 4);
                memcpy(&ba[(size_t) r * 2 * nv + nv], &w.alpha[(size_t) r * nv], (size_t) nv * 4);
            }
            memcpy(g.w_ba->data, ba.data(), ba.size() * 4);
            memcpy(g.ssm_a->data, w.a.data(), w.a.size() * 4);
            memcpy(g.dt->data, w.dt.data(), w.dt.size() * 4);
            memcpy(g.norm_w->data, w.nrm.data(), w.nrm.size() * 4);
            memcpy(g.w_out->data, w.out.data(), w.out.size() * 4);
            memcpy(g.ident->data, ident.data(), ident.size() * 4);
            g.lora_alpha = alpha; g.lora_rank = rank;
            if (adapted) {
                g.lora_a_qkv = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, rank);
                g.lora_b_qkv = ggml_new_tensor_2d(c, GGML_TYPE_F32, rank, n_qkv);
                g.lora_a_gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, rank);
                g.lora_b_gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, rank, n_z);
                g.lora_a_out = ggml_new_tensor_2d(c, GGML_TYPE_F32, hd * nv, rank);
                g.lora_b_out = ggml_new_tensor_2d(c, GGML_TYPE_F32, rank, D);
                const Tr & t0 = tr[(size_t) (i * 3 + 0)], & t1 = tr[(size_t) (i * 3 + 1)], & t2 = tr[(size_t) (i * 3 + 2)];
                memcpy(g.lora_a_qkv->data, t0.a.data(), t0.a.size() * 4);
                memcpy(g.lora_b_qkv->data, t0.b.data(), t0.b.size() * 4);
                memcpy(g.lora_a_gate->data, t1.a.data(), t1.a.size() * 4);
                memcpy(g.lora_b_gate->data, t1.b.data(), t1.b.size() * 4);
                memcpy(g.lora_a_out->data, t2.a.data(), t2.a.size() * 4);
                memcpy(g.lora_b_out->data, t2.b.data(), t2.b.size() * 4);
                if (la) {
                    la->insert(la->end(), { g.lora_a_qkv, g.lora_a_gate, g.lora_a_out });
                    lb->insert(lb->end(), { g.lora_b_qkv, g.lora_b_gate, g.lora_b_out });
                }
            }
            cur = qfn_gdn_layer(c, cur, g, hd, nk, nv, d_conv, 1e-6f);
        }
        return cur;
    };

    // ---- target: the base stack on the other window ----
    std::vector<float> target;
    {
        ggml_init_params ip = { 3ull << 30, nullptr, false };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        memcpy(x->data, xb.data(), xb.size() * 4);
        ggml_tensor * o = build(c, x, false, nullptr, nullptr);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(c, g, 16);
        target.assign((float *) o->data, (float *) o->data + (size_t) D * T);
        double n2 = 0;
        for (float v : target) n2 += (double) v * v;
        std::printf("target rms %.5f\n", std::sqrt(n2 / (double) target.size()));
        ggml_free(c);
    }

    // ---- training ----
    const float b1 = 0.9f, b2 = 0.999f, aeps = 1e-8f, wd = 0.0f;
    for (int step = 0; step <= steps; ++step) {
        ggml_init_params pi = { 3ull << 30, nullptr, false };
        ggml_context * sc = ggml_init(pi);
        ggml_tensor * x = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, T);
        ggml_tensor * tt = ggml_new_tensor_2d(sc, GGML_TYPE_F32, D, T);
        memcpy(x->data, xa.data(), xa.size() * 4);
        memcpy(tt->data, target.data(), target.size() * 4);
        std::vector<ggml_tensor *> la, lb;
        ggml_tensor * o = build(sc, x, true, &la, &lb);
        for (auto * t : la) ggml_set_param(t);
        for (auto * t : lb) ggml_set_param(t);
        ggml_tensor * d = ggml_sub(sc, o, tt);
        ggml_tensor * loss = ggml_sum(sc, ggml_mul(sc, d, d));
        ggml_set_loss(loss);
        ggml_cgraph * gf = ggml_new_graph_custom(sc, 1048576, true);
        ggml_build_forward_expand(gf, loss);
        ggml_build_backward_expand(sc, gf, nullptr);
        ggml_graph_reset(gf);
        ggml_graph_compute_with_ctx(sc, gf, 16);
        const float Loss = ((float *) loss->data)[0];
        if (step % 5 == 0 || step == steps)
            std::printf("step %4d  loss %.6f  rms_err %.5f\n", step, Loss, std::sqrt((double) Loss / ((double) D * T)));
        std::fflush(stdout);
        if (step == steps) { ggml_free(sc); break; }
        const float bc1 = 1.0f - std::pow(b1, (float) (step + 1)), bc2 = 1.0f - std::pow(b2, (float) (step + 1));
        for (size_t i = 0; i < la.size(); ++i) {
            auto upd = [&](std::vector<float> & prm, std::vector<float> & m, std::vector<float> & v, ggml_tensor * g) {
                const float * gp = (const float *) g->data;
                for (size_t k = 0; k < prm.size(); ++k) {
                    m[k] = b1 * m[k] + (1 - b1) * gp[k];
                    v[k] = b2 * v[k] + (1 - b2) * gp[k] * gp[k];
                    prm[k] -= lr * ((m[k] / bc1) / (std::sqrt(v[k] / bc2) + aeps) + wd * prm[k]);
                }
            };
            ggml_tensor * ga = ggml_graph_get_grad(gf, la[i]);
            ggml_tensor * gb = ggml_graph_get_grad(gf, lb[i]);
            upd(tr[i].a, tr[i].ma, tr[i].va, ga);
            upd(tr[i].b, tr[i].mb, tr[i].vb, gb);
        }
        ggml_free(sc);
    }

    // ---- save + reload + verify ----
    std::vector<LoraTensorPair> ad(tr.size());
    for (size_t i = 0; i < tr.size(); ++i) {
        ad[i].name = tr[i].name; ad[i].n_in = tr[i].n_in; ad[i].n_out = tr[i].n_out; ad[i].rank = rank;
        ad[i].A = tr[i].a; ad[i].B = tr[i].b;
    }
    std::string err;
    if (!write_lora_gguf(gpath, "qwen4exp", alpha, ad, err)) { std::fprintf(stderr, "save: %s\n", err.c_str()); return 1; }
    std::printf("saved %s (%zu adapters, rank %lld)\n", gpath, ad.size(), (long long) rank);

    std::vector<LoraTensorPair> back;
    float fa = 0; std::string arch;
    if (!read_lora_gguf(gpath, back, fa, arch, err)) { std::fprintf(stderr, "reload: %s\n", err.c_str()); return 1; }
    double dmax = 0;
    for (size_t i = 0; i < back.size(); ++i) {
        for (size_t k = 0; k < back[i].A.size(); ++k) dmax = std::fmax(dmax, std::fabs(back[i].A[k] - ad[i].A[k]));
        for (size_t k = 0; k < back[i].B.size(); ++k) dmax = std::fmax(dmax, std::fabs(back[i].B[k] - ad[i].B[k]));
    }
    std::printf("reload: %zu adapters, alpha %.1f, arch %s, max |diff| %.3e\n", back.size(), fa, arch.c_str(), dmax);

    auto eval = [&](bool adapted) {
        ggml_init_params pi = { 3ull << 30, nullptr, false };
        ggml_context * c = ggml_init(pi);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        memcpy(x->data, xa.data(), xa.size() * 4);
        ggml_tensor * o = build(c, x, adapted, nullptr, nullptr);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(c, g, 16);
        double n2 = 0, m2 = 0;
        for (int64_t i = 0; i < D * T; ++i) {
            const double dd = ((float *) o->data)[i] - target[(size_t) i];
            n2 += dd * dd; m2 += (double) target[(size_t) i] * (double) target[(size_t) i];
        }
        ggml_free(c);
        return std::sqrt(n2 / std::max(m2, 1e-30));
    };
    const double rb = eval(false), ra = eval(true);
    std::printf("verify: relative L2 vs target  base %.4f  adapted %.4f  (%.1fx)\n", rb, ra, rb / std::max(ra, 1e-12));
    std::printf("%s\n", ra < 0.5 * rb ? "TRAIN-STACK PASS" : "TRAIN-STACK WEAK");
    return 0;
}
