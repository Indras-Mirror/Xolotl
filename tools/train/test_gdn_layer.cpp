// test_gdn_layer [S] [n_k] [n_v] [T] [eps] [stride] [dump.bin]
// QFN layer core as ONE training graph (what qwen4exp.cpp build_layer_attn_linear does between the qkv projection
// and the output projection):
//   conv_input [d_conv-1+T, channels] (zero history, i.e. position 0) --ssm_conv(c)--> silu
//   -> split into q [S, n_k, T], k [S, n_k, T], v [S, n_v, T] (channel order q | k | v, as the model)
//   -> l2 norm q,k (rms_norm(x, eps/S) * 1/sqrt(S), models.h build_gdn_l2_norm)
//   -> tile q,k to n_v heads with ggml_repeat_4d (v head j reads k head j % n_k - the model's mapping)
//   -> gdn_chunked (tools/train/gdn_train.hpp)
//   loss = sum(o * R); central FD vs the analytic gradient for conv_input (x), the conv weight (c), g and b.
// Gate: every element (stride) of x, c, g, b. Expected: max abs err ~1e-4 at eps 1e-2.
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
    std::vector<float> x, c, g, b, r;
};

static double run(int S, int nk, int nv, int T, const Inputs & in, std::vector<float> * gx, std::vector<float> * gc,
                  std::vector<float> * gg, std::vector<float> * gb, std::vector<float> * out_o) {
    const int d_conv = 4;
    const int64_t channels = 2 * S * nk + S * nv;
    const int64_t n_cin = d_conv - 1 + T;
    ggml_init_params ip = { 512u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_cin, channels, 1);
    ggml_tensor * tc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, channels);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, nv, T, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, nv, T, 1);
    ggml_tensor * tr = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, nv, T, 1);
    ggml_tensor * ti = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    memcpy(tx->data, in.x.data(), in.x.size() * 4);
    memcpy(tc->data, in.c.data(), in.c.size() * 4);
    memcpy(tg->data, in.g.data(), in.g.size() * 4);
    memcpy(tb->data, in.b.data(), in.b.size() * 4);
    memcpy(tr->data, in.r.data(), in.r.size() * 4);
    std::vector<float> ident((size_t) kGdnChunk * kGdnChunk, 0.0f);
    for (int i = 0; i < kGdnChunk; ++i) ident[(size_t) i + (size_t) i * kGdnChunk] = 1.0f;
    memcpy(ti->data, ident.data(), ident.size() * 4);

    const bool want_grad = gx != nullptr;
    if (want_grad) { ggml_set_param(tx); ggml_set_param(tc); ggml_set_param(tg); ggml_set_param(tb); }

    ggml_tensor * y = ggml_silu(ctx, ggml_ssm_conv(ctx, tx, tc));          // [channels, T, 1]
    const size_t es = ggml_element_size(y), row = ggml_row_size(y->type, channels);
    ggml_tensor * q = ggml_view_4d(ctx, y, S, nk, T, 1, S * es, row, row * T, 0);
    ggml_tensor * k = ggml_view_4d(ctx, y, S, nk, T, 1, S * es, row, row * T, (size_t) S * nk * es);
    ggml_tensor * v = ggml_view_4d(ctx, y, S, nv, T, 1, S * es, row, row * T, (size_t) 2 * S * nk * es);
    const float eps = 1e-6f;
    auto l2 = [&](ggml_tensor * t) {
        return ggml_scale(ctx, ggml_rms_norm(ctx, t, eps / (float) S), 1.0f / std::sqrt((float) S));
    };
    q = l2(q); k = l2(k);
    q = ggml_repeat_4d(ctx, q, S, nv, T, 1);     // model: repeat tiles (v head j <- k head j % nk)
    k = ggml_repeat_4d(ctx, k, S, nv, T, 1);
    ggml_tensor * o = gdn_chunked(ctx, q, k, v, tg, tb, ti);
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, o, tr));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, true);
    ggml_build_forward_expand(gf, loss);
    if (want_grad) { ggml_build_backward_expand(ctx, gf, nullptr); ggml_graph_reset(gf); }
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    const double L = ((float *) loss->data)[0];
    if (out_o) out_o->assign((float *) o->data, (float *) o->data + (size_t) S * nv * T);
    if (want_grad) {
        auto grab = [&](ggml_tensor * t, std::vector<float> * dst) {
            ggml_tensor * gt = ggml_graph_get_grad(gf, t);
            dst->assign((float *) gt->data, (float *) gt->data + ggml_nelements(t));
        };
        grab(tx, gx); grab(tc, gc); grab(tg, gg); grab(tb, gb);
    }
    ggml_free(ctx);
    return L;
}

int main(int argc, char ** argv) {
    const int S = argc > 1 ? atoi(argv[1]) : 8;          // head dim
    const int nk = argc > 2 ? atoi(argv[2]) : 2;         // k heads
    const int nv = argc > 3 ? atoi(argv[3]) : 6;         // v heads
    const int T = argc > 4 ? atoi(argv[4]) : 96;
    const float eps = argc > 5 ? (float) atof(argv[5]) : 1e-2f;
    const int stride = argc > 6 ? atoi(argv[6]) : 1;
    const char * dump = argc > 7 ? argv[7] : nullptr;
    const int d_conv = 4;
    const int64_t channels = 2 * S * nk + S * nv, n_cin = d_conv - 1 + T;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> U(-1, 1);
    Inputs in;
    in.x.assign((size_t) n_cin * channels, 0.0f);          // d_conv-1 history rows stay 0 (position 0)
    for (size_t i = (size_t) (d_conv - 1) * channels; i < in.x.size(); ++i) in.x[i] = 0.5f * U(rng);
    in.c.resize((size_t) d_conv * channels); for (auto & x : in.c) x = 0.5f * U(rng);
    in.g.resize((size_t) nv * T); for (auto & x : in.g) x = -0.3f * std::fabs(U(rng));
    in.b.resize((size_t) nv * T); for (auto & x : in.b) x = 0.5f + 0.4f * U(rng);
    in.r.resize((size_t) S * nv * T); for (auto & x : in.r) x = 0.5f * U(rng);

    std::vector<float> o, gx, gc, gg, gb;
    run(S, nk, nv, T, in, &gx, &gc, &gg, &gb, &o);
    if (dump) {
        FILE * f = fopen(dump, "wb");
        const int hdr[5] = { S, nk, nv, T, d_conv };
        fwrite(hdr, sizeof(hdr), 1, f);
        for (const auto * a : { &in.x, &in.c, &in.g, &in.b, &in.r, &o }) fwrite(a->data(), a->size() * 4, 1, f);
        fclose(f);
    }
    auto slot = [](Inputs & x, int w) -> std::vector<float> & { return w == 0 ? x.x : w == 1 ? x.c : w == 2 ? x.g : x.b; };
    struct Case { const char * name; int which; std::vector<float> * grad; };
    Case cases[] = { { "x", 0, &gx }, { "c", 1, &gc }, { "g", 2, &gg }, { "b", 3, &gb } };
    for (auto & cs : cases) {
        const size_t n = slot(in, cs.which).size();
        double emax = 0, gmax = 0, sum_err = 0, sum_scale = 0;
        size_t nchk = 0, worst = 0;
        for (size_t i = 0; i < n; i += (size_t) stride) {
            Inputs ip = in, im = in;
            slot(ip, cs.which)[i] += eps;
            slot(im, cs.which)[i] -= eps;
            const double num = (run(S, nk, nv, T, ip, nullptr, nullptr, nullptr, nullptr, nullptr) -
                                run(S, nk, nv, T, im, nullptr, nullptr, nullptr, nullptr, nullptr)) / (2 * eps);
            const double err = std::fabs(num - (*cs.grad)[i]);
            if (err > emax) { emax = err; worst = i; }
            gmax = std::max(gmax, (double) std::fabs((*cs.grad)[i]));
            sum_err += err; sum_scale += std::fabs(num) + std::fabs((*cs.grad)[i]);
            ++nchk;
        }
        std::printf("d%s: max abs err %.3e (idx %zu) | max |g| %.3e | mean rel %.2e\n",
                    cs.name, emax, worst, gmax, sum_err / std::max(sum_scale, 1e-30));
    }
    std::printf("S %d nk %d nv %d T %d: loss %.6f, checked %zu per input (stride %d), eps %.1e\n",
                S, nk, nv, T, run(S, nk, nv, T, in, nullptr, nullptr, nullptr, nullptr, nullptr),
                in.x.size() / (size_t) stride, stride, eps);
    return 0;
}
