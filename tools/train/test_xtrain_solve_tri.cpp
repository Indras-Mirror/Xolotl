// test_xtrain_solve_tri [offdiag diag_min n k batches eps [pos [verbose]]] (env B3 = 4th-dim split, NT = threads)
// SOLVE_TRI backward (Xolotl rule: dB = A^-T G via the new upper solve, dA = -tril(dB X^T)) vs central differences, max abs
// error (test-backend-ops grad mode uses a relative per-element metric that breaks on the zero upper triangle).
// 20 Nov: n 10/17/30, k 3/8/9, 3-D and 4-D batches, 1 and 16 threads: dA/dB agree to FD noise (4e-7..4e-5).
// SOLVE_TRI backward check: loss = sum(X * R), X = solve(A, B), A lower. Analytic (ggml backward) vs central FD.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>
int main(int argc, char ** argv) {
    const float offd = argc > 1 ? atof(argv[1]) : 0.3f, dmin = argc > 2 ? atof(argv[2]) : 1.0f;
    const int n = argc > 3 ? atoi(argv[3]) : 10, k = argc > 4 ? atoi(argv[4]) : 3, b2 = argc > 5 ? atoi(argv[5]) : 2;
    const float eps = argc > 6 ? atof(argv[6]) : 1e-2f;
    const bool pos = argc > 7;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1, 1);
    std::vector<float> A(n * n * b2), B(k * n * b2), R(k * n * b2);
    for (int b = 0; b < b2; ++b) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        A[b * n * n + i * n + j] = j < i ? (pos ? (0.1f + 0.45f * (U(rng) + 1)) / n : offd * U(rng)) : j == i ? dmin + 0.5f * (U(rng) + 1) : 0.0f;
    for (auto & v : B) v = U(rng);
    for (auto & v : R) v = pos ? 1.0f : U(rng);
    auto run = [&](const std::vector<float> & a, const std::vector<float> & bb, std::vector<float> * ga, std::vector<float> * gb) {
        ggml_init_params ip = { 64u << 20, nullptr, false };
        ggml_context * c = ggml_init(ip);
        const int b3 = getenv("B3") ? atoi(getenv("B3")) : 1, bb2 = b2 / b3;
        ggml_tensor * ta = ggml_new_tensor_4d(c, GGML_TYPE_F32, n, n, bb2, b3), * tb = ggml_new_tensor_4d(c, GGML_TYPE_F32, k, n, bb2, b3), * tr = ggml_new_tensor_4d(c, GGML_TYPE_F32, k, n, bb2, b3);
        memcpy(ta->data, a.data(), a.size() * 4); memcpy(tb->data, bb.data(), bb.size() * 4); memcpy(tr->data, R.data(), R.size() * 4);
        if (ga) { ggml_set_param(ta); ggml_set_param(tb); }
        ggml_tensor * loss = ggml_sum(c, ggml_mul(c, ggml_solve_tri(c, ta, tb, true, true, false), tr));
        ggml_set_loss(loss);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 4096, true);
        ggml_build_forward_expand(gf, loss);
        if (ga) { ggml_build_backward_expand(c, gf, nullptr); ggml_graph_reset(gf); }
        ggml_graph_compute_with_ctx(c, gf, getenv("NT") ? atoi(getenv("NT")) : 1);
        double L = ((float *) loss->data)[0];
        if (ga) {
            ggml_tensor * g = ggml_graph_get_grad(gf, ta); ga->assign((float *) g->data, (float *) g->data + a.size());
            g = ggml_graph_get_grad(gf, tb); gb->assign((float *) g->data, (float *) g->data + bb.size());
        }
        ggml_free(c);
        return L;
    };
    std::vector<float> gA, gB;
    run(A, B, &gA, &gB);
    double ea = 0, eb = 0, ma = 0, mb = 0, maa = 0; size_t nv = 0, nnan = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        auto ap = A, am = A; ap[i] += eps; am[i] -= eps;
        const double num = (run(ap, B, nullptr, nullptr) - run(am, B, nullptr, nullptr)) / (2 * eps);
        { const double as = (num - gA[i]) / (num + gA[i]); if (std::isfinite(as)) { maa += std::fabs(as); ++nv; } else ++nnan; }
        if (std::fabs(num - gA[i]) > ea) { ea = std::fabs(num - gA[i]); if (argc > 8) printf("  A[%zu] (b %zu row %zu col %zu) num %.5f ana %.5f\n", i, i/(n*n), (i%(n*n))/n, i%n, num, gA[i]); } ma = std::max(ma, std::fabs(num));
    }
    for (size_t i = 0; i < B.size(); ++i) {
        auto bp = B, bm = B; bp[i] += eps; bm[i] -= eps;
        const double num = (run(A, bp, nullptr, nullptr) - run(A, bm, nullptr, nullptr)) / (2 * eps);
        eb = std::max(eb, std::fabs(num - gB[i])); mb = std::max(mb, std::fabs(num));
    }
    std::printf("harness MAA over dA = %.4f (%zu valid, %zu NaN)\n", maa / std::max<size_t>(nv, 1), nv, nnan);
    std::printf("offdiag %.2f diag>=%.2f: dA max abs err %.3e (max |g| %.3e) | dB max abs err %.3e (max |g| %.3e)\n", offd, dmin, ea, ma, eb, mb);
}
