// test_moe_train [n_embd] [n_ff] [E] [K] [T] [eps] [stride] [dump.bin]
// FD gate for tools/train/moe_train.hpp (differentiable MoE expert loop, QFN's configuration: softmax gating,
// top-K of E, weight normalisation, SILU, shared-expert-style scale off). loss = sum(o * R).
// The routing (which experts, in what order) is host-computed and constant, exactly as in training; the weights are
// differentiated through probs.
#include "ggml.h"
#include "ggml-cpu.h"
#include "moe_train.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace xolotl::train;

struct Inputs {
    std::vector<float> x, router, up, gate, down, r;
};

static double run(int D, int F, int E, int K, int T, const Inputs & in, const std::vector<int32_t> & ids,
                  const std::vector<std::vector<int32_t>> & tok_e, const std::vector<std::vector<int32_t>> & pair_e,
                  std::vector<float> * gx, std::vector<float> * gr, std::vector<float> * gup,
                  std::vector<float> * ggate, std::vector<float> * gdown, std::vector<float> * out_o) {
    ggml_init_params ip = { 1024u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    ggml_tensor * tr = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, E);
    ggml_tensor * tup = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, F, E);
    ggml_tensor * tga = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, F, E);
    ggml_tensor * tdo = ggml_new_tensor_3d(c, GGML_TYPE_F32, F, D, E);
    ggml_tensor * trr = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    memcpy(tx->data, in.x.data(), in.x.size() * 4);
    memcpy(tr->data, in.router.data(), in.router.size() * 4);
    memcpy(tup->data, in.up.data(), in.up.size() * 4);
    memcpy(tga->data, in.gate.data(), in.gate.size() * 4);
    memcpy(tdo->data, in.down.data(), in.down.size() * 4);
    memcpy(trr->data, in.r.data(), in.r.size() * 4);
    MoeIndex ix = moe_make_index(c, ids, K, T, E, tok_e, pair_e);

    const bool want = gx != nullptr;
    if (want) { ggml_set_param(tx); ggml_set_param(tr); ggml_set_param(tup); ggml_set_param(tga); ggml_set_param(tdo); }
    ggml_tensor * o = moe_ffn(c, tx, tr, tup, tga, tdo, ix, true, 0.0f);
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, o, trr));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, true);
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
        grab(tx, gx); grab(tr, gr); grab(tup, gup); grab(tga, ggate); grab(tdo, gdown);
    }
    ggml_free(c);
    return L;
}

int main(int argc, char ** argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 16;
    const int F = argc > 2 ? atoi(argv[2]) : 12;
    const int E = argc > 3 ? atoi(argv[3]) : 6;
    const int K = argc > 4 ? atoi(argv[4]) : 2;
    const int T = argc > 5 ? atoi(argv[5]) : 10;
    const float eps = argc > 6 ? (float) atof(argv[6]) : 1e-2f;
    const int stride = argc > 7 ? atoi(argv[7]) : 1;
    const char * dump = argc > 8 ? argv[8] : nullptr;

    std::mt19937 rng(31);
    std::uniform_real_distribution<float> U(-1, 1);
    Inputs in;
    in.x.resize((size_t) D * T);      for (auto & v : in.x) v = 0.6f * U(rng);
    in.router.resize((size_t) D * E); for (auto & v : in.router) v = 0.8f * U(rng);
    in.up.resize((size_t) D * F * E);     for (auto & v : in.up) v = 0.4f * U(rng);
    in.gate.resize((size_t) D * F * E);   for (auto & v : in.gate) v = 0.4f * U(rng);
    in.down.resize((size_t) F * D * E);   for (auto & v : in.down) v = 0.4f * U(rng);
    in.r.resize((size_t) D * T);      for (auto & v : in.r) v = 0.5f * U(rng);

    // routing from the real router logits (host, constant for the graph)
    std::vector<float> logits((size_t) E * T, 0.0f);
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int d = 0; d < D; ++d) s += (double) in.router[(size_t) (d + e * D)] * in.x[(size_t) (d + t * D)];
            logits[(size_t) (e * T + t)] = (float) s;
        }
    std::vector<int32_t> ids; std::vector<float> wts;
    moe_route(logits.data(), E, T, K, ids, wts);
    std::vector<std::vector<int32_t>> tok_e, pair_e;
    moe_expert_lists(ids, K, T, E, tok_e, pair_e);

    std::vector<float> o, gx, gr, gup, gga, gdo;
    run(D, F, E, K, T, in, ids, tok_e, pair_e, &gx, &gr, &gup, &gga, &gdo, &o);
    if (dump) {
        FILE * f = fopen(dump, "wb");
        const int hdr[5] = { D, F, E, K, T };
        fwrite(hdr, sizeof(hdr), 1, f);
        for (const auto * a : { &in.x, &in.router, &in.up, &in.gate, &in.down, &in.r, &o })
            fwrite(a->data(), a->size() * 4, 1, f);
        for (const auto & v : wts) { float w = v; fwrite(&w, 4, 1, f); }
        for (int i = 0; i < K * T; ++i) { int32_t v = ids[(size_t) i]; fwrite(&v, 4, 1, f); }
        fclose(f);
    }

    auto slot = [](Inputs & s, int w) -> std::vector<float> & {
        return w == 0 ? s.x : w == 1 ? s.router : w == 2 ? s.up : w == 3 ? s.gate : s.down;
    };
    struct Case { const char * name; int which; std::vector<float> * grad; };
    Case cases[] = { { "x", 0, &gx }, { "router", 1, &gr }, { "up", 2, &gup }, { "gate", 3, &gga }, { "down", 4, &gdo } };
    for (auto & cs : cases) {
        const size_t n = slot(in, cs.which).size();
        double emax = 0, gmax = 0, sum_err = 0, sum_scale = 0;
        size_t nchk = 0, worst = 0;
        for (size_t i = 0; i < n; i += (size_t) stride) {
            Inputs a = in, b = in;
            slot(a, cs.which)[i] += eps;
            slot(b, cs.which)[i] -= eps;
            const double num = (run(D, F, E, K, T, a, ids, tok_e, pair_e, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) -
                                run(D, F, E, K, T, b, ids, tok_e, pair_e, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)) / (2 * eps);
            const double err = std::fabs(num - (*cs.grad)[i]);
            if (err > emax) { emax = err; worst = i; }
            gmax = std::max(gmax, (double) std::fabs((*cs.grad)[i]));
            sum_err += err; sum_scale += std::fabs(num) + std::fabs((*cs.grad)[i]);
            ++nchk;
        }
        std::printf("d%-7s max abs err %.3e (idx %zu) | max |g| %.3e | mean rel %.2e\n",
                    cs.name, emax, worst, gmax, sum_err / std::max(sum_scale, 1e-30));
    }
    std::printf("D %d F %d E %d K %d T %d: loss %.6f, %zu checked per input (stride %d), eps %.1e\n",
                D, F, E, K, T, run(D, F, E, K, T, in, ids, tok_e, pair_e, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                (size_t) 1 + (in.x.size() - 1) / (size_t) stride, stride, eps);
    return 0;
}
