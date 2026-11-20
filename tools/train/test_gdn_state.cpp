// test_gdn_state [T] [split] : gdn_chunked over T tokens in one pass == prefix pass (s_out) + suffix pass (s0).
// Checks the initial-state input the harness trainer uses for the shared 20K-token Quetza prefix. Also checks the
// gradient into the suffix inputs is unchanged by the split (FD-free: compares analytic grads of the two graphs).
#include "ggml.h"
#include "ggml-cpu.h"
#include "gdn_train.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace xolotl::train;

int main(int argc, char ** argv) {
    const int64_t S = 16, H = 2, T = argc > 1 ? atoll(argv[1]) : 200, T1 = argc > 2 ? atoll(argv[2]) : 100, T2 = T - T1;
    std::printf("[state] S %lld H %lld T %lld split %lld\n", (long long) S, (long long) H, (long long) T, (long long) T1);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(S * H * T), k(S * H * T), v(S * H * T), g(H * T), b(H * T);
    for (auto & x : q) x = nd(rng);
    for (auto & x : k) x = nd(rng) * 0.25f;
    for (auto & x : v) x = nd(rng);
    for (auto & x : g) x = -std::fabs(nd(rng)) * 0.1f;
    for (auto & x : b) x = 1.f / (1.f + std::exp(-nd(rng)));
    ggml_init_params ip = { 512u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    auto mk = [&](const std::vector<float> & src, int64_t d0, int64_t t0, int64_t tn) {   // [d0, H, tn] slice of [d0, H, T]
        ggml_tensor * x = ggml_new_tensor_4d(c, GGML_TYPE_F32, d0, H, tn, 1);
        float * p = (float *) x->data;
        for (int64_t t = 0; t < tn; ++t) for (int64_t h = 0; h < H; ++h) for (int64_t i = 0; i < d0; ++i)
            p[(t * H + h) * d0 + i] = src[(size_t) (((t0 + t) * H + h) * d0 + i)];
        return x;
    };
    ggml_tensor * id = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
    for (int i = 0; i < kGdnChunk * kGdnChunk; ++i) ((float *) id->data)[i] = (i % (kGdnChunk + 1)) == 0 ? 1.f : 0.f;
    // one pass
    ggml_tensor * o_full = gdn_chunked(c, mk(q, S, 0, T), mk(k, S, 0, T), mk(v, S, 0, T), mk(g, 1, 0, T), mk(b, 1, 0, T), id);
    // prefix pass -> state
    ggml_tensor * s1 = nullptr;
    ggml_tensor * o1 = gdn_chunked(c, mk(q, S, 0, T1), mk(k, S, 0, T1), mk(v, S, 0, T1), mk(g, 1, 0, T1), mk(b, 1, 0, T1), id, nullptr, &s1);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, false);
    ggml_build_forward_expand(gf, o_full);
    ggml_build_forward_expand(gf, o1);
    ggml_build_forward_expand(gf, s1);
    ggml_graph_compute_with_ctx(c, gf, 1);
    // suffix pass from a CONSTANT copy of the state (as the trainer feeds it)
    ggml_tensor * s0 = ggml_dup_tensor(c, s1);
    std::memcpy(s0->data, s1->data, ggml_nbytes(s1));
    ggml_tensor * o2 = gdn_chunked(c, mk(q, S, T1, T2), mk(k, S, T1, T2), mk(v, S, T1, T2), mk(g, 1, T1, T2), mk(b, 1, T1, T2), id, s0);
    ggml_cgraph * g2 = ggml_new_graph_custom(c, 65536, false);
    ggml_build_forward_expand(g2, o2);
    ggml_graph_compute_with_ctx(c, g2, 1);
    // compare: o_full [S, H, T] vs o1 (first T1) and o2 (rest)
    double e = 0, m = 0;
    const float * F = (const float *) o_full->data, * A = (const float *) o1->data, * B = (const float *) o2->data;
    for (int64_t t = 0; t < T; ++t) for (int64_t i = 0; i < S * H; ++i) {
        const float ref = F[t * S * H + i];
        const float got = t < T1 ? A[t * S * H + i] : B[(t - T1) * S * H + i];
        e = std::fmax(e, std::fabs(ref - got));
        m = std::fmax(m, std::fabs(ref));
    }
    std::printf("[state] max |one pass - split| = %.3e (max |o| %.3e) %s\n", e, m, e <= 1e-5 * std::fmax(m, 1.0) ? "PASS" : "FAIL");
    return e <= 1e-5 * std::fmax(m, 1.0) ? 0 : 1;
}
