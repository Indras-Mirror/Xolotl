// test_qfn_hc [D] [hc] [hc_lr] [hd] [nk] [nv] [hc2] [eps] [stride] [dump.bin]
//   hc  = streams into the attention half, hc2 = streams into the FFN half (both 4 on QFN)
// FD + forward gate for the hyper-connection layer in tools/train/qfn_layer_train.hpp:
//   res [D, hc, T] --hc_mix--> mixed [D, T] --qfn_gdn_block--> attn_out [D, T] --hc_combine--> res' [D, hc, T]
//   ... and the same for an FFN half, which here is a plain one-layer MLP (the MoE is gated separately in
//   test_moe_train; this test is about the hyper-connection plumbing, in and out).
// loss = sum(res' * R); central differences over res, both halves' weights and the block weights.
#include "ggml.h"
#include "ggml-cpu.h"
#include "qfn_layer_train.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace xolotl::train;
using xolotl::train::kGdnChunk;

struct Inputs {
    std::vector<float> res, nrm, down, up, inj, ffn_g, ffn_u, ffn_d, r;
};

static double run(int D, int hc, int lr, int hd, int nk, int nv, int hc2, const Inputs & in,
                  std::vector<float> * g_res, std::vector<float> * g_nrm, std::vector<float> * g_down,
                  std::vector<float> * g_up, std::vector<float> * g_inj, std::vector<float> * g_fg,
                  std::vector<float> * g_fu, std::vector<float> * g_fd, std::vector<float> * out_o) {
    const int d_conv = 4;
    ggml_init_params ip = { 8u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    const int64_t n_qkv = 2 * hd * nk + hd * nv, n_z = hd * nv;
    const int T = 8;

    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, hc, T);
    QfnHc w1;
    w1.norm = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, hc);
    w1.down = ggml_new_tensor_2d(c, GGML_TYPE_F32, D * hc, lr);
    w1.up = ggml_new_tensor_2d(c, GGML_TYPE_F32, lr, D * hc);
    w1.inject = ggml_new_tensor_2d(c, GGML_TYPE_F32, D * hc, hc);
    QfnHc w2 = w1;   // the FFN half reuses the same shapes in this test
    QfnLayer g;
    g.wqkv = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_qkv);
    g.wgate = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_z);
    g.conv_w = ggml_new_tensor_2d(c, GGML_TYPE_F32, d_conv, n_qkv);
    g.w_ba = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * nv);
    g.ssm_a = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
    g.dt = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
    g.norm_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
    g.w_out = ggml_new_tensor_2d(c, GGML_TYPE_F32, hd * nv, D);
    g.ident = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    ggml_tensor * fg = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * D);
    ggml_tensor * fu = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * D);
    ggml_tensor * fd = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * D, D);
    ggml_tensor * rr = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, hc, T);

    memcpy(res->data, in.res.data(), in.res.size() * 4);
    memcpy(rr->data, in.r.data(), in.r.size() * 4);
    memcpy(w1.norm->data, in.nrm.data(), in.nrm.size() * 4);
    memcpy(w1.down->data, in.down.data(), in.down.size() * 4);
    memcpy(w1.up->data, in.up.data(), in.up.size() * 4);
    memcpy(w1.inject->data, in.inj.data(), in.inj.size() * 4);
    memcpy(fg->data, in.ffn_g.data(), in.ffn_g.size() * 4);
    memcpy(fu->data, in.ffn_u.data(), in.ffn_u.size() * 4);
    memcpy(fd->data, in.ffn_d.data(), in.ffn_d.size() * 4);
    std::vector<float> id((size_t) kGdnChunk * kGdnChunk, 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) id[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
    memcpy(g.ident->data, id.data(), id.size() * 4);

    const bool want = g_res != nullptr;
    if (want) {
        ggml_set_param(res); ggml_set_param(w1.norm); ggml_set_param(w1.down); ggml_set_param(w1.up);
        ggml_set_param(w1.inject); ggml_set_param(fg); ggml_set_param(fu); ggml_set_param(fd);
        ggml_set_param(g.wqkv); ggml_set_param(g.wgate); ggml_set_param(g.conv_w); ggml_set_param(g.w_ba);
        ggml_set_param(g.ssm_a); ggml_set_param(g.dt); ggml_set_param(g.norm_w); ggml_set_param(g.w_out);
    }
    // Stage-gated chain so a hang can be BISECTED instead of guessed at: HC_STAGES=1..6, default 6 (full).
    // Every stage writes the shape it produces into `tout`, so the tail below is identical for all of them.
    const int stages = getenv("HC_STAGES") ? atoi(getenv("HC_STAGES")) : 6;
    fprintf(stderr, "[stage] running %d of 6\n", stages); fflush(stderr);

    ggml_tensor * inj1 = nullptr;
    ggml_tensor * cur = qfn_hc_mix(c, res, w1, hc, 1e-6f, &inj1);          // [D, T]
    ggml_tensor * tout = ggml_reshape_3d(c, cur, D, 1, T);
    ggml_tensor * res2 = res;
    fprintf(stderr, "[stage] 1 mix1 built\n"); fflush(stderr);

    if (stages >= 2) {
        cur = qfn_gdn_block(c, cur, g, hd, nk, nv, d_conv, 1e-6f);        // [D, T]
        tout = ggml_reshape_3d(c, cur, D, 1, T);
        fprintf(stderr, "[stage] 2 block built\n"); fflush(stderr);
    }
    if (stages >= 3) {
        res2 = qfn_hc_combine(c, res, cur, inj1, hc);                     // [D, hc, T]
        tout = res2;
        fprintf(stderr, "[stage] 3 combine1 built\n"); fflush(stderr);
    }
    ggml_tensor * inj2 = nullptr;
    ggml_tensor * f = nullptr;
    if (stages >= 4) {
        cur = qfn_hc_mix(c, ggml_reshape_3d(c, res2, D, hc, T), w2, hc, 1e-6f, &inj2);   // [D, T]
        tout = ggml_reshape_3d(c, cur, D, 1, T);
        fprintf(stderr, "[stage] 4 mix2 built\n"); fflush(stderr);
    }
    if (stages >= 5) {
        ggml_tensor * gu = ggml_mul(c, ggml_mul_mat(c, fu, cur), ggml_silu(c, ggml_mul_mat(c, fg, cur)));
        f = ggml_mul_mat(c, fd, gu);
        tout = ggml_reshape_3d(c, f, D, 1, T);
        fprintf(stderr, "[stage] 5 mlp built\n"); fflush(stderr);
    }
    if (stages >= 6) {
        tout = qfn_hc_combine(c, res2, f, inj2, hc);
        fprintf(stderr, "[stage] 6 combine2 built\n"); fflush(stderr);
    }
    // partial stages produce [D,1,T]; the reference tensor is [D,hc,T] - repeat so the loss shape always matches
    if (tout->ne[1] != hc) tout = ggml_repeat_4d(c, tout, D, hc, T, 1);
    ggml_tensor * out = tout;

    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, out, rr));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, true);
    ggml_build_forward_expand(gf, loss);
    if (want) { ggml_build_backward_expand(c, gf, nullptr); ggml_graph_reset(gf); }
    ggml_graph_compute_with_ctx(c, gf, 1);
    const double L = ((float *) loss->data)[0];
    if (out_o) out_o->assign((float *) out->data, (float *) out->data + (size_t) D * hc * T);
    if (want) {
        auto grab = [&](ggml_tensor * t, std::vector<float> * dst) {
            ggml_tensor * q = ggml_graph_get_grad(gf, t);   // NULL when a partial stage never uses the tensor
            if (!q) { dst->assign(ggml_nelements(t), 0.0f); return; }
            dst->assign((float *) q->data, (float *) q->data + ggml_nelements(t));
        };
        grab(res, g_res); grab(w1.norm, g_nrm); grab(w1.down, g_down); grab(w1.up, g_up); grab(w1.inject, g_inj);
        grab(fg, g_fg); grab(fu, g_fu); grab(fd, g_fd);
    }
    ggml_free(c);
    return L;
}

int main(int argc, char ** argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 8;
    const int hc = argc > 2 ? atoi(argv[2]) : 3;
    const int lr = argc > 3 ? atoi(argv[3]) : 6;
    const int hd = argc > 4 ? atoi(argv[4]) : 4;
    const int nk = argc > 5 ? atoi(argv[5]) : 2;
    const int nv = argc > 6 ? atoi(argv[6]) : 2;
    const int eps_i = argc > 8 ? 8 : 8;
    (void) eps_i;
    const float eps = argc > 8 ? (float) atof(argv[8]) : 1e-2f;
    int stride = argc > 9 ? atoi(argv[9]) : 1;
    if (stride < 1) { std::fprintf(stderr, "[warn] stride %d -> 1 (i += 0 would loop forever)\n", stride); stride = 1; }
    const char * dump = argc > 10 ? argv[10] : nullptr;
    const int T = 8;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> U(-1, 1);
    Inputs in;
    in.res.assign((size_t) D * hc * T, 0.0f);  for (auto & v : in.res) v = 0.5f * U(rng);
    in.nrm.assign((size_t) D * hc, 1.0f);      for (auto & v : in.nrm) v = 1.0f + 0.2f * U(rng);
    in.down.assign((size_t) D * hc * lr, 0.0f);  for (auto & v : in.down) v = 2.0f * U(rng) / std::sqrt((float) (D * hc));
    in.up.assign((size_t) lr * D * hc, 0.0f);    for (auto & v : in.up) v = 2.0f * U(rng) / std::sqrt((float) lr);
    in.inj.assign((size_t) D * hc * hc, 0.0f);   for (auto & v : in.inj) v = 2.0f * U(rng) / std::sqrt((float) (D * hc));
    in.ffn_g.assign((size_t) D * 2 * D, 0.0f);   for (auto & v : in.ffn_g) v = 0.3f * U(rng) / std::sqrt((float) D);
    in.ffn_u.assign((size_t) D * 2 * D, 0.0f);   for (auto & v : in.ffn_u) v = 0.3f * U(rng) / std::sqrt((float) D);
    in.ffn_d.assign((size_t) 2 * D * D, 0.0f);   for (auto & v : in.ffn_d) v = 0.3f * U(rng) / std::sqrt((float) (2 * D));
    in.r.assign((size_t) D * hc * T, 0.0f);      for (auto & v : in.r) v = 0.5f * U(rng);

    std::vector<float> o, gr, gn, gd, gu, gi, gfg, gfu, gfd;
    run(D, hc, lr, hd, nk, nv, hc, in, &gr, &gn, &gd, &gu, &gi, &gfg, &gfu, &gfd, &o);
    if (dump) {
        FILE * f = fopen(dump, "wb");
        const int hdr[6] = { D, hc, lr, hd, nk, nv };
        fwrite(hdr, sizeof(hdr), 1, f);
        for (const auto * a : { &in.res, &in.nrm, &in.down, &in.up, &in.inj, &in.ffn_g, &in.ffn_u, &in.ffn_d, &o })
            fwrite(a->data(), a->size() * 4, 1, f);
        fclose(f);
    }
    auto slot = [](Inputs & s, int i) -> std::vector<float> & {
        switch (i) {
            case 0: return s.res; case 1: return s.nrm; case 2: return s.down; case 3: return s.up;
            case 4: return s.inj; case 5: return s.ffn_g; case 6: return s.ffn_u; default: return s.ffn_d;
        }
    };
    struct Case { const char * name; int which; std::vector<float> * grad; };
    Case cases[] = { { "res", 0, &gr }, { "hc_norm", 1, &gn }, { "hc_down", 2, &gd }, { "hc_up", 3, &gu },
                     { "hc_inject", 4, &gi }, { "ffn_gate", 5, &gfg }, { "ffn_up", 6, &gfu }, { "ffn_down", 7, &gfd } };
    for (auto & cs : cases) {
        const size_t n = slot(in, cs.which).size();
        double emax = 0, gmax = 0, se = 0, ss = 0;
        size_t nchk = 0, worst = 0;
        for (size_t i = 0; i < n; i += (size_t) stride) {
            Inputs p = in, m = in;
            slot(p, cs.which)[i] += eps;
            slot(m, cs.which)[i] -= eps;
            const double num = (run(D, hc, lr, hd, nk, nv, hc, p, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) -
                                run(D, hc, lr, hd, nk, nv, hc, m, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) / (2 * eps);
            const double e = std::fabs(num - (*cs.grad)[i]);
            if (e > emax) { emax = e; worst = i; }
            gmax = std::max(gmax, (double) std::fabs((*cs.grad)[i]));
            se += e; ss += std::fabs(num) + std::fabs((*cs.grad)[i]);
            ++nchk;
        }
        std::printf("d%-10s max abs err %.3e (idx %zu) | max |g| %.3e | mean rel %.2e\n",
                    cs.name, emax, worst, gmax, se / std::max(ss, 1e-30));
    }
    std::printf("D %d hc %d lr %d hd %d nk %d nv %d: loss %.6f, %zu checked per input (stride %d), eps %.1e\n",
                D, hc, lr, hd, nk, nv,
                run(D, hc, lr, hd, nk, nv, hc, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                1 + (in.res.size() - 1) / (size_t) stride, stride, eps);
    return 0;
}
