// test_gdn_chunked [S] [H] [T] [eps] [stride] [dump.bin] [input_scale]
// Gate for tools/train/gdn_train.hpp (differentiable chunked gated delta net):
//   1) forward: o = gdn_chunked(q, k, v, g, b, I)  (dumped for the numpy reference, tools/train/gdn_ref.py)
//   2) backward: loss = sum(o * R), analytic (ggml autodiff through SOLVE_TRI/TRI/CUMSUM/PAD/...) vs central
//      finite differences over every element of q, k, v, g, b (stride > 1 samples).
// All F32 on the CPU. Expected: max abs err ~1e-4..1e-3 at eps 1e-2, well under the gradient magnitudes.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gdn_train.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

using xolotl::train::gdn_chunked;
using xolotl::train::kGdnChunk;

struct Inputs {
    std::vector<float> q, k, v, g, b, r;
};

// forward (and optionally the gradient of sum(o*R) w.r.t. every input)
static double run(int S, int H, int T, const Inputs & in, std::vector<float> * gq, std::vector<float> * gk,
                  std::vector<float> * gv, std::vector<float> * gg, std::vector<float> * gb, std::vector<float> * out_o) {
    ggml_init_params ip = { 512u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tq = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * tr = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * ti = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    memcpy(tq->data, in.q.data(), in.q.size() * 4);
    memcpy(tk->data, in.k.data(), in.k.size() * 4);
    memcpy(tv->data, in.v.data(), in.v.size() * 4);
    memcpy(tg->data, in.g.data(), in.g.size() * 4);
    memcpy(tb->data, in.b.data(), in.b.size() * 4);
    memcpy(tr->data, in.r.data(), in.r.size() * 4);
    std::vector<float> ident((size_t) kGdnChunk * kGdnChunk, 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) ident[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
    memcpy(ti->data, ident.data(), ident.size() * 4);

    const bool want_grad = gq != nullptr;
    if (want_grad) { ggml_set_param(tq); ggml_set_param(tk); ggml_set_param(tv); ggml_set_param(tg); ggml_set_param(tb); }
    ggml_tensor * o = gdn_chunked(c, tq, tk, tv, tg, tb, ti);
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, o, tr));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, true);
    ggml_build_forward_expand(gf, loss);
    if (want_grad) { ggml_build_backward_expand(c, gf, nullptr); ggml_graph_reset(gf); }
    ggml_graph_compute_with_ctx(c, gf, 1);
    const double L = ((float *) loss->data)[0];
    if (out_o) out_o->assign((float *) o->data, (float *) o->data + (size_t) S * H * T);
    if (want_grad) {
        auto grab = [&](ggml_tensor * t, std::vector<float> * dst) {
            ggml_tensor * gt = ggml_graph_get_grad(gf, t);
            dst->assign((float *) gt->data, (float *) gt->data + ggml_nelements(t));
        };
        grab(tq, gq); grab(tk, gk); grab(tv, gv); grab(tg, gg); grab(tb, gb);
    }
    ggml_free(c);
    return L;
}

int main(int argc, char ** argv) {
    const int S = argc > 1 ? atoi(argv[1]) : 16;
    const int H = argc > 2 ? atoi(argv[2]) : 2;
    const int T = argc > 3 ? atoi(argv[3]) : 100;
    const float eps = argc > 4 ? (float) atof(argv[4]) : 1e-2f;
    const int stride = argc > 5 ? atoi(argv[5]) : 1;
    const char * dump = argc > 6 ? argv[6] : nullptr;
    const float scale = argc > 7 ? (float) atof(argv[7]) : 1.0f;   // input scale (<1 keeps the loss O(1) at S=128 H=48)
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> U(-1, 1);
    Inputs in;
    in.q.resize((size_t) S * H * T); in.k.resize(in.q.size()); in.v.resize(in.q.size()); in.r.resize(in.q.size());
    in.g.resize((size_t) H * T); in.b.resize(in.g.size());
    for (auto & x : in.q) x = scale * U(rng);
    for (auto & x : in.k) x = scale * U(rng);
    for (auto & x : in.v) x = scale * U(rng);
    for (auto & x : in.r) x = 0.5f * U(rng);
    for (auto & x : in.g) x = -0.3f * std::fabs(U(rng));          // log decay, <= 0
    for (auto & x : in.b) x = 0.5f + 0.4f * U(rng);               // beta after sigmoid, in (0, 1)

    std::vector<float> o;
    std::vector<float> gq, gk, gv, gg, gb;
    run(S, H, T, in, &gq, &gk, &gv, &gg, &gb, &o);
    if (dump) {
        FILE * f = fopen(dump, "wb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", dump); return 1; }
        const int hdr[3] = { S, H, T };
        fwrite(hdr, sizeof(hdr), 1, f);
        for (const auto * a : { &in.q, &in.k, &in.v, &in.g, &in.b, &in.r, &o })
            fwrite(a->data(), a->size() * 4, 1, f);
        fclose(f);
    }

    auto slot = [](Inputs & x, int which) -> std::vector<float> & {
        return which == 0 ? x.q : which == 1 ? x.k : which == 2 ? x.v : which == 3 ? x.g : x.b;
    };
    struct Case { const char * name; int which; std::vector<float> * grad; };
    Case cases[] = {
        { "q", 0, &gq }, { "k", 1, &gk }, { "v", 2, &gv }, { "g", 3, &gg }, { "b", 4, &gb },
    };
    for (auto & cs : cases) {
        const size_t n = slot(in, cs.which).size();
        double emax = 0, gmax = 0, sum_rel = 0, sum_scale = 0;
        size_t nchk = 0, worst = 0;
        for (size_t i = 0; i < n; i += (size_t) stride) {
            Inputs ip = in, im = in;
            slot(ip, cs.which)[i] += eps;
            slot(im, cs.which)[i] -= eps;
            const double num = (run(S, H, T, ip, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) -
                                run(S, H, T, im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) / (2 * eps);
            const double err = std::fabs(num - (*cs.grad)[i]);
            if (err > emax) { emax = err; worst = i; }
            gmax = std::max(gmax, (double) std::fabs((*cs.grad)[i]));
            sum_rel += std::fabs(num - (*cs.grad)[i]);
            sum_scale += std::fabs(num) + std::fabs((*cs.grad)[i]);
            ++nchk;
        }
        std::printf("d%s: max abs err %.3e (idx %zu) | max |g| %.3e | mean abs err %.3e\n",
                    cs.name, emax, worst, gmax, sum_rel / std::max<size_t>(nchk, 1));
        std::printf("     relative (mean |err| / mean |g|): %.2e\n", sum_rel / std::max(sum_scale, 1e-30));
    }
    std::printf("loss %.6f, n_chunks %d, checked %zu per input (stride %d), eps %.1e\n",
                run(S, H, T, in, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                (T + kGdnChunk - 1) / kGdnChunk, in.q.size() / (size_t) stride, stride, eps);
    return 0;
}
