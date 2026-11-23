// Double-precision reference for the softmax gradients w.r.t. input a and mask m (Xolotl LoRA training).
// Build: g++ -O2 -std=c++17 smgrad_ref.cpp -I<train tree>/ggml/include -L<build-cpu>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build-cpu>/bin
// L = sum(softmax(scale*a + m [, sinks]) * w). Analytic: dz = y*(w - <y,w>_row), da = scale*dz, dm = dz summed over
// the dims the mask broadcasts on. Compares the CPU ggml backward against it (16 cases, pass = max abs err <= 1e-5 x (1 + max)).
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int run(int n0, int n1, int n2, int n3, bool bcast, bool sinks, float scale) {
    ggml_init_params ip = {256u << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, bcast ? 1 : n2, n3);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3);
    ggml_tensor * sk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n2);
    ggml_set_param(a);
    ggml_set_param(m);
    std::mt19937 rng(1234 + n0 + 7 * n2 + (bcast ? 100 : 0) + (sinks ? 1000 : 0));
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    for (ggml_tensor * t : {a, m, w, sk}) {
        float * p = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) p[i] = U(rng);
    }
    ggml_tensor * y = ggml_soft_max_ext(ctx, a, m, scale, 0.0f);
    if (sinks) ggml_soft_max_add_sinks(y, sk);
    ggml_tensor * L = ggml_sum(ctx, ggml_mul(ctx, y, w));
    ggml_set_loss(L);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, L);
    ggml_build_backward_expand(ctx, gf, nullptr);
    ggml_graph_reset(gf);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    const float * ga = (const float *) ggml_graph_get_grad(gf, a)->data;
    const float * gm = (const float *) ggml_graph_get_grad(gf, m)->data;

    // reference in double
    const float * A = (const float *) a->data, * M = (const float *) m->data, * W = (const float *) w->data,
                * S = (const float *) sk->data;
    const int m2 = bcast ? 1 : n2;
    std::vector<double> rda(ggml_nelements(a)), rdm(ggml_nelements(m), 0.0);
    for (int i3 = 0; i3 < n3; i3++) for (int i2 = 0; i2 < n2; i2++) for (int i1 = 0; i1 < n1; i1++) {
        const size_t ra = ((size_t)(i3 * n2 + i2) * n1 + i1) * n0;
        const size_t rm = ((size_t)(i3 * m2 + (bcast ? 0 : i2)) * n1 + i1) * n0;
        std::vector<double> z(n0), yy(n0);
        double mx = sinks ? (double) S[i2] : -INFINITY;
        for (int i = 0; i < n0; i++) { z[i] = (double) scale * A[ra + i] + M[rm + i]; mx = std::max(mx, z[i]); }
        double den = sinks ? std::exp((double) S[i2] - mx) : 0.0;
        for (int i = 0; i < n0; i++) { yy[i] = std::exp(z[i] - mx); den += yy[i]; }
        double dot = 0;
        for (int i = 0; i < n0; i++) { yy[i] /= den; dot += yy[i] * W[ra + i]; }
        for (int i = 0; i < n0; i++) {
            const double dz = yy[i] * (W[ra + i] - dot);
            rda[ra + i] = scale * dz;
            rdm[rm + i] += dz;
        }
    }
    double ea = 0, em = 0, na = 0, nm = 0;
    for (size_t i = 0; i < rda.size(); i++) { ea = std::max(ea, std::fabs(ga[i] - rda[i])); na = std::max(na, std::fabs(rda[i])); }
    for (size_t i = 0; i < rdm.size(); i++) { em = std::max(em, std::fabs(gm[i] - rdm[i])); nm = std::max(nm, std::fabs(rdm[i])); }
    const bool ok = ea <= 1e-5 * (1 + na) && em <= 1e-5 * (1 + nm);
    printf("ne=[%d,%d,%d,%d] bcast=%d sinks=%d scale=%.6f  da maxabs err %.3e (|da|max %.3f)  dm maxabs err %.3e (|dm|max %.3f)  %s\n",
           n0, n1, n2, n3, bcast, sinks, scale, ea, na, em, nm, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    for (bool bcast : {false, true}) for (bool sinks : {false, true}) for (float sc : {1.0f, 0.08838835f}) {
        fails += run(16, 8, 3, 1, bcast, sinks, sc);
        fails += run(33, 5, 4, 2, bcast, sinks, sc);
    }
    printf("FAILS %d / 16\n", fails);
    return fails ? 1 : 0;
}
