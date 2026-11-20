// test_qfn_layer [D] [hd] [nk] [nv] [T] [eps] [stride] [dump.bin]
// FD + forward gate for tools/train/qfn_layer_train.hpp (one QFN gated-delta-net layer, weights as parameters).
// loss = sum(layer_out * R); central differences over cur and every weight tensor.
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
    std::vector<float> cur, wqkv, conv, wba, a, dt, nrm, wout, r;
};

static double run(int D, int hd, int nk, int nv, int T, const Inputs & in,
                  std::vector<float> * g_cur, std::vector<float> * g_wqkv, std::vector<float> * g_conv,
                  std::vector<float> * g_wba, std::vector<float> * g_a, std::vector<float> * g_dt,
                  std::vector<float> * g_nrm, std::vector<float> * g_wout, std::vector<float> * out_o) {
    const int d_conv = 4;
    ggml_init_params ip = { 1024u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    const int64_t n_qkv = 2 * hd * nk + hd * nv, n_z = hd * nv;
    QfnLayer w;
    w.wqkv   = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_qkv);
    w.wgate  = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_z);
    w.conv_w = ggml_new_tensor_2d(c, GGML_TYPE_F32, d_conv, n_qkv);
    w.w_ba   = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * nv);
    w.ssm_a  = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
    w.dt     = ggml_new_tensor_1d(c, GGML_TYPE_F32, nv);
    w.norm_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
    w.w_out  = ggml_new_tensor_2d(c, GGML_TYPE_F32, hd * nv, D);
    w.ident  = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    ggml_tensor * cur = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    ggml_tensor * rr  = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    memcpy(w.wqkv->data, in.wqkv.data(), (size_t) D * n_qkv * 4);
    memcpy(w.wgate->data, in.wqkv.data() + (size_t) D * n_qkv, (size_t) D * n_z * 4);
    memcpy(w.conv_w->data, in.conv.data(), in.conv.size() * 4);
    memcpy(w.w_ba->data, in.wba.data(), in.wba.size() * 4);
    memcpy(w.ssm_a->data, in.a.data(), in.a.size() * 4);
    memcpy(w.dt->data, in.dt.data(), in.dt.size() * 4);
    memcpy(w.norm_w->data, in.nrm.data(), in.nrm.size() * 4);
    memcpy(w.w_out->data, in.wout.data(), in.wout.size() * 4);
    memcpy(cur->data, in.cur.data(), in.cur.size() * 4);
    memcpy(rr->data, in.r.data(), in.r.size() * 4);
    std::vector<float> id((size_t) kGdnChunk * kGdnChunk, 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) id[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
    memcpy(w.ident->data, id.data(), id.size() * 4);

    const bool want = g_cur != nullptr;
    if (want) {
        ggml_set_param(cur); ggml_set_param(w.wqkv); ggml_set_param(w.wgate); ggml_set_param(w.conv_w); ggml_set_param(w.w_ba);
        ggml_set_param(w.ssm_a); ggml_set_param(w.dt); ggml_set_param(w.norm_w); ggml_set_param(w.w_out);
    }
    ggml_tensor * o = qfn_gdn_layer(c, cur, w, hd, nk, nv, d_conv, 1e-6f);
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, o, rr));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 131072, true);
    ggml_build_forward_expand(gf, loss);
    if (want) { ggml_build_backward_expand(c, gf, nullptr); ggml_graph_reset(gf); }
    ggml_graph_compute_with_ctx(c, gf, 1);
    const double L = ((float *) loss->data)[0];
    if (out_o) out_o->assign((float *) o->data, (float *) o->data + (size_t) D * T);
    if (want) {
        auto grab = [&](ggml_tensor * t, std::vector<float> * dst) {
            ggml_tensor * g = ggml_graph_get_grad(gf, t);
            dst->assign((float *) g->data, (float *) g->data + ggml_nelements(t));
        };
        grab(cur, g_cur); grab(w.wqkv, g_wqkv);
        { ggml_tensor * gg2 = ggml_graph_get_grad(gf, w.wgate); std::vector<float> tmp((float *) gg2->data, (float *) gg2->data + ggml_nelements(w.wgate)); g_wqkv->insert(g_wqkv->end(), tmp.begin(), tmp.end()); } grab(w.conv_w, g_conv); grab(w.w_ba, g_wba);
        grab(w.ssm_a, g_a); grab(w.dt, g_dt); grab(w.norm_w, g_nrm); grab(w.w_out, g_wout);
    }
    ggml_free(c);
    return L;
}

int main(int argc, char ** argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 24;
    const int hd = argc > 2 ? atoi(argv[2]) : 4;
    const int nk = argc > 3 ? atoi(argv[3]) : 2;
    const int nv = argc > 4 ? atoi(argv[4]) : 6;
    const int T = argc > 5 ? atoi(argv[5]) : 20;
    const float eps = argc > 6 ? (float) atof(argv[6]) : 1e-2f;
    const int stride = argc > 7 ? atoi(argv[7]) : 1;
    const char * dump = argc > 8 ? argv[8] : nullptr;
    const int64_t n_qkv = 2 * hd * nk + hd * nv, n_z = hd * nv;

    std::mt19937 rng(41);
    std::uniform_real_distribution<float> U(-1, 1);
    Inputs in;
    in.cur.assign((size_t) D * T, 0.0f);     for (auto & v : in.cur) v = 0.6f * U(rng);
    in.wqkv.assign((size_t) D * (n_qkv + n_z), 0.0f); for (auto & v : in.wqkv) v = 0.5f * U(rng);
    in.conv.assign((size_t) 4 * n_qkv, 0.0f);         for (auto & v : in.conv) v = 0.5f * U(rng);
    in.wba.assign((size_t) D * 2 * nv, 0.0f);         for (auto & v : in.wba) v = 0.5f * U(rng);
    in.a.assign((size_t) nv, 0.0f);   for (auto & v : in.a) v = -std::exp(0.5f * U(rng));   // -A_log.exp()
    in.dt.assign((size_t) nv, 0.0f);  for (auto & v : in.dt) v = 0.3f * U(rng);
    in.nrm.assign((size_t) hd, 1.0f);
    in.wout.assign((size_t) hd * nv * D, 0.0f); for (auto & v : in.wout) v = 0.4f * U(rng);
    in.r.assign((size_t) D * T, 0.0f);          for (auto & v : in.r) v = 0.5f * U(rng);

    std::vector<float> o, gc, gw, gcv, gb, ga, gd, gn, go;
    run(D, hd, nk, nv, T, in, &gc, &gw, &gcv, &gb, &ga, &gd, &gn, &go, &o);
    if (dump) {
        FILE * f = fopen(dump, "wb");
        const int hdr[5] = { D, hd, nk, nv, T };
        fwrite(hdr, sizeof(hdr), 1, f);
        for (const auto * a : { &in.cur, &in.wqkv, &in.conv, &in.wba, &in.a, &in.dt, &in.nrm, &in.wout, &in.r, &o })
            fwrite(a->data(), a->size() * 4, 1, f);
        fclose(f);
    }
    auto slot = [](Inputs & s, int i) -> std::vector<float> & {
        switch (i) {
            case 0: return s.cur; case 1: return s.wqkv; case 2: return s.conv; case 3: return s.wba;
            case 4: return s.a;   case 5: return s.dt;   case 6: return s.nrm;  default: return s.wout;
        }
    };
    struct Case { const char * name; int which; std::vector<float> * grad; };
    Case cases[] = { { "cur", 0, &gc }, { "wqkv", 1, &gw }, { "conv", 2, &gcv }, { "w_ba", 3, &gb },
                     { "ssm_a", 4, &ga }, { "dt", 5, &gd }, { "norm", 6, &gn }, { "w_out", 7, &go } };
    for (auto & cs : cases) {
        const size_t n = slot(in, cs.which).size();
        double emax = 0, gmax = 0, se = 0, ss = 0;
        size_t nchk = 0, worst = 0;
        for (size_t i = 0; i < n; i += (size_t) stride) {
            Inputs p = in, m = in;
            slot(p, cs.which)[i] += eps;
            slot(m, cs.which)[i] -= eps;
            const double num = (run(D, hd, nk, nv, T, p, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) -
                                run(D, hd, nk, nv, T, m, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) / (2 * eps);
            const double e = std::fabs(num - (*cs.grad)[i]);
            if (e > emax) { emax = e; worst = i; }
            gmax = std::max(gmax, (double) std::fabs((*cs.grad)[i]));
            se += e; ss += std::fabs(num) + std::fabs((*cs.grad)[i]);
            ++nchk;
        }
        std::printf("d%-7s max abs err %.3e (idx %zu) | max |g| %.3e | mean rel %.2e\n",
                    cs.name, emax, worst, gmax, se / std::max(ss, 1e-30));
    }
    std::printf("D %d hd %d nk %d nv %d T %d: loss %.6f, %zu checked per input (stride %d), eps %.1e\n",
                D, hd, nk, nv, T, run(D, hd, nk, nv, T, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                1 + (in.cur.size() - 1) / (size_t) stride, stride, eps);
    return 0;
}
