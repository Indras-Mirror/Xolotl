// tools/train/test_xtrain_grad.cpp - T0 gate for LoRA training (docs/train/LORA_PLAN.md): one mini MoE layer shaped like
// Hy4's, built the way the trainer will build it, differentiated by ggml autodiff on CPU and on CUDA.
//
//   h   = rms_norm(x) * nw
//   q   = Wq h + s * B (A h)                         Wq Q8_0 frozen; A, B the LoRA (params)
//   p   = sigmoid(Wr h)                              router (F32, frozen)
//   w   = get_rows(p, ids) / sum                     ids fixed for the step (routing is a constant of the step)
//   y_e = Wd_e swiglu_clamp(Wg_e x_e, Wu_e x_e, L)   x_e = get_rows(h, idx_e); experts Q8_0 frozen
//   out = q + sum_e get_rows_back(y_e * w_e, idx_e)  (scatter-add)
//   loss = sum(out * G)                              surrogate: G = the upstream gradient
//
// Checks: dA, dB, dx  CUDA vs CPU (rel err); CPU vs central finite differences on sampled elements, on an F32 copy of the
// dequantized weights (a Q8_0 matmul rounds its ACTIVATIONS to 8 bits on the CPU, so the as-served forward is a step
// function of x at fine scale and finite differences there measure the rounding); and the Q8_0 run's grads vs that copy's.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace {

constexpr int D = 256, F = 128, T = 12, E = 8, K = 2, R = 4;
constexpr float LIM = 1.0f, S_LORA = 0.5f;

struct Host {   // every input of the graph, identical for both backends
    std::vector<float> x, nw, A, B, Wr, G;
    std::vector<uint8_t> Wq, Wg[E], Wu[E], Wd[E];
    std::vector<int32_t> ids;                    // [K, T]
    std::vector<int32_t> ids_flat;               // t * E + ids[t, k]
    std::vector<std::vector<int32_t>> idx, pair; // per expert: token rows, flat (t * K + k) indices
};

std::vector<uint8_t> quant(const std::vector<float> & f, int64_t n0, int64_t n1) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, n0) * n1);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, n1, n0, nullptr);
    return q;
}

Host make_host() {
    std::mt19937 rng(20261120);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randn = [&](size_t n, float s) { std::vector<float> v(n); for (auto & e : v) e = nd(rng) * s; return v; };
    Host h;
    h.x = randn((size_t) D * T, 1.0f);
    h.nw = randn(D, 0.2f);
    for (auto & e : h.nw) e += 1.0f;
    h.A = randn((size_t) D * R, 0.1f);
    h.B = randn((size_t) R * D, 0.1f);
    h.Wr = randn((size_t) D * E, 0.1f);
    h.G = randn((size_t) D * T, 1.0f);
    h.Wq = quant(randn((size_t) D * D, 0.06f), D, D);
    for (int e = 0; e < E; ++e) {
        h.Wg[e] = quant(randn((size_t) D * F, 0.1f), D, F);
        h.Wu[e] = quant(randn((size_t) D * F, 0.1f), D, F);
        h.Wd[e] = quant(randn((size_t) F * D, 0.1f), F, D);
    }
    h.ids.resize((size_t) K * T);
    h.idx.assign(E, {});
    h.pair.assign(E, {});
    for (int t = 0; t < T; ++t) {
        const int a = (int) (rng() % E);
        int b = (int) (rng() % (E - 1));
        if (b >= a) ++b;
        h.ids[(size_t) t * K + 0] = a;
        h.ids[(size_t) t * K + 1] = b;
        for (int k = 0; k < K; ++k) {
            const int e = h.ids[(size_t) t * K + k];
            h.idx[e].push_back(t);
            h.pair[e].push_back(t * K + k);
        }
    }
    return h;
}

struct Result { std::vector<float> dA, dB, dx; float loss = 0; bool ok = false; };

// builds, allocates on `be`, computes; `perturb` edits the host inputs first (finite differences)
Result run(ggml_backend_t be, const Host & h0, bool backward, const std::function<void(Host &)> & perturb = nullptr, bool f32w = false) {
    Host h = h0;
    if (perturb) perturb(h);
    Result r;
    ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, true), nullptr, true };
    ggml_context * c = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const void *>> fill;
    std::vector<std::vector<float>> deq;   // f32w: dequantized weight copies (kept alive until the upload)
    deq.reserve(64);
    auto in = [&](ggml_type t, int64_t n0, int64_t n1, const void * d) {
        if (f32w && t == GGML_TYPE_Q8_0) {
            deq.emplace_back((size_t) (n0 * n1));
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(d, deq.back().data(), n0 * n1);
            t = GGML_TYPE_F32;
            d = deq.back().data();
        }
        ggml_tensor * x = ggml_new_tensor_2d(c, t, n0, n1);
        fill.emplace_back(x, d);
        return x;
    };
    ggml_tensor * x = in(GGML_TYPE_F32, D, T, h.x.data());
    ggml_tensor * A = in(GGML_TYPE_F32, D, R, h.A.data());
    ggml_tensor * B = in(GGML_TYPE_F32, R, D, h.B.data());
    ggml_set_param(x);
    ggml_set_param(A);
    ggml_set_param(B);
    ggml_tensor * nw = in(GGML_TYPE_F32, D, 1, h.nw.data());
    ggml_tensor * Wq = in(GGML_TYPE_Q8_0, D, D, h.Wq.data());
    ggml_tensor * Wr = in(GGML_TYPE_F32, D, E, h.Wr.data());
    ggml_tensor * G = in(GGML_TYPE_F32, D, T, h.G.data());
    // routing indices into the flat [1, E*T] probabilities (t * E + expert): a 2-D gather, which GET_ROWS' backward takes
    std::vector<int32_t> & flat = const_cast<std::vector<int32_t> &>(h.ids_flat);
    flat.resize((size_t) K * T);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) flat[(size_t) t * K + k] = t * E + h.ids[(size_t) t * K + k];
    ggml_tensor * ids = ggml_reshape_1d(c, in(GGML_TYPE_I32, (int64_t) K * T, 1, flat.data()), (int64_t) K * T);

    ggml_tensor * hn = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), nw);
    ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, Wq, hn), ggml_scale(c, ggml_mul_mat(c, B, ggml_mul_mat(c, A, hn)), S_LORA));
    ggml_tensor * p = ggml_sigmoid(c, ggml_mul_mat(c, Wr, hn));                                   // [E, T]
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_2d(c, p, 1, (int64_t) E * T), ids);        // [1, K*T]
    w = ggml_reshape_2d(c, w, K, T);
    w = ggml_div(c, w, ggml_sum_rows(c, w));
    ggml_tensor * wf = ggml_reshape_2d(c, ggml_cont(c, w), 1, (int64_t) K * T);                 // one weight per row
    ggml_tensor * out = q;
    for (int e = 0; e < E; ++e) {
        const int ne = (int) h.idx[e].size();
        if (ne == 0) continue;
        ggml_tensor * ie = in(GGML_TYPE_I32, ne, 1, h.idx[e].data());
        ggml_tensor * pe = in(GGML_TYPE_I32, ne, 1, h.pair[e].data());
        ie = ggml_reshape_1d(c, ie, ne);
        pe = ggml_reshape_1d(c, pe, ne);
        ggml_tensor * Wg = in(GGML_TYPE_Q8_0, D, F, h.Wg[e].data());
        ggml_tensor * Wu = in(GGML_TYPE_Q8_0, D, F, h.Wu[e].data());
        ggml_tensor * Wd = in(GGML_TYPE_Q8_0, F, D, h.Wd[e].data());
        ggml_tensor * xe = ggml_get_rows(c, hn, ie);                                             // [D, ne]
        ggml_tensor * a = ggml_swiglu_clamp(c, ggml_mul_mat(c, Wg, xe), ggml_mul_mat(c, Wu, xe), LIM);
        ggml_tensor * ye = ggml_mul(c, ggml_mul_mat(c, Wd, a), ggml_reshape_2d(c, ggml_get_rows(c, wf, pe), 1, ne));
        out = ggml_add(c, out, ggml_get_rows_back(c, ye, ie, hn));
    }
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, out, G));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, true);
    ggml_build_forward_expand(gf, loss);
    if (backward) ggml_build_backward_expand(c, gf, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); ggml_free(c); return r; }
    for (auto & f : fill) ggml_backend_tensor_set(f.first, f.second, 0, ggml_nbytes(f.first));
    if (backward) ggml_graph_reset(gf);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return r; }
    ggml_backend_tensor_get(loss, &r.loss, 0, 4);
    if (backward) {
        auto get = [&](ggml_tensor * t, std::vector<float> & v) {
            ggml_tensor * g = ggml_graph_get_grad(gf, t);
            v.resize((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(g, v.data(), 0, v.size() * 4);
        };
        get(A, r.dA);
        get(B, r.dB);
        get(x, r.dx);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    r.ok = true;
    return r;
}

double rel(const std::vector<float> & a, const std::vector<float> & b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (double) (a[i] - b[i]); d += (double) b[i] * b[i]; }
    return std::sqrt(n / std::max(d, 1e-30));
}

}  // namespace

int main() {
    const Host h = make_host();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    Result rc = run(cpu, h, true);                                     // Q8_0 weights, as served
    Result rf = run(cpu, h, true, nullptr, true);                      // the same weights in F32: a smooth forward
    if (!rc.ok || !rf.ok) return 1;
    std::printf("CPU  loss %.6f (F32-weight copy %.6f)\n", rc.loss, rf.loss);
    bool pass = true;
    {   // the Q8_0 run's gradients vs the smooth run's: they differ only by the forward's activation rounding
        const double ea = rel(rc.dA, rf.dA), eb = rel(rc.dB, rf.dB), ex = rel(rc.dx, rf.dx);
        std::printf("CPU  Q8_0 vs F32-weight grads: dA %.2e dB %.2e dx %.2e %s\n", ea, eb, ex,
                    ea < 5e-2 && eb < 5e-2 && ex < 5e-2 ? "OK" : "FAIL");
        pass = pass && ea < 5e-2 && eb < 5e-2 && ex < 5e-2;
    }

    // finite differences on the CPU (F32 activations; the Q8_0 weights are fixed, so the function is smooth away from
    // the clamp kinks)
    std::mt19937 rng(7);
    const float eps = 1e-2f;
    auto fd = [&](const char * name, std::vector<float> Host::*field, const std::vector<float> & grad) {
        double worst = 0;
        for (int s = 0; s < 8; ++s) {
            const size_t i = rng() % grad.size();
            const float lp = run(cpu, h, false, [&](Host & hh) { (hh.*field)[i] += eps; }, true).loss;
            const float lm = run(cpu, h, false, [&](Host & hh) { (hh.*field)[i] -= eps; }, true).loss;
            const double num = (lp - lm) / (2.0 * eps);
            const double e = std::fabs(num - grad[i]) / std::max(1e-3, std::fabs(num) + std::fabs((double) grad[i]));
            if (std::getenv("XT_VERBOSE")) std::printf("   %s[%zu] analytic %+.6f numeric %+.6f\n", name, i, grad[i], num);
            worst = std::max(worst, e);
        }
        std::printf("CPU  finite differences %-2s: worst rel err %.2e %s\n", name, worst, worst < 2e-2 ? "OK" : "FAIL");
        pass = pass && worst < 2e-2;
    };
    fd("x", &Host::x, rf.dx);   // finite differences need the smooth forward (see above)
    fd("A", &Host::A, rf.dA);
    fd("B", &Host::B, rf.dB);

    ggml_backend_t cu = ggml_backend_cuda_init(0);
    if (!cu) { std::printf("no CUDA backend\n"); return pass ? 0 : 1; }
    Result rg = run(cu, h, true);
    if (!rg.ok) return 1;
    const double ea = rel(rg.dA, rc.dA), eb = rel(rg.dB, rc.dB), ex = rel(rg.dx, rc.dx);
    std::printf("CUDA loss %.6f (CPU %.6f) | grad rel err vs CPU: dA %.2e dB %.2e dx %.2e\n", rg.loss, rc.loss, ea, eb, ex);
    const bool ok = ea < 2e-2 && eb < 2e-2 && ex < 2e-2;   // the CUDA forward rounds activations to q8_1 for its dots
    std::printf("%s\n", pass && ok ? "PASS" : "FAIL");
    ggml_backend_free(cu);
    ggml_backend_free(cpu);
    return pass && ok ? 0 : 1;
}
