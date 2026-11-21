// Fused decode ops for xolotl-inkling (see inkling_fused.h).  Built with -use_fast_math like ggml-cuda, and written
// to perform the same float operations in the same order as the ggml kernels they replace (fmaf where ggml's
// compiler contracts a*b+c, __fadd_rn/__fmul_rn where ggml runs separate elementwise kernels).
#include "inkling_fused.h"

#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#ifdef GGML_CUDA_CUSTOM_MAGIC   // the xolotl ggml patch with the CUDA custom-op hook

namespace {

// op_params[0..5] hold ggml_custom_op_params (fun, n_tasks, userdata); the op's own parameters start at int 8
constexpr int kP = 8;

template <typename T> T param(const ggml_tensor* t, int i) { T v; std::memcpy(&v, &t->op_params[kP + i], sizeof(T)); return v; }
template <typename T> void set_param(ggml_tensor* t, int i, T v) { std::memcpy(&t->op_params[kP + i], &v, sizeof(T)); }

// ------------------------------------------------------------------------------------------------ conv step
template <int K>
__global__ void k_conv_step(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ st,
                            const float* __restrict__ dyn, float* __restrict__ y, int C, int T, int nh, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float r[K], wr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) wr[j] = w[(size_t) c * K + j];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) r[j] = st[(size_t) c * (K - 1) + j];
    const int hh = nh > 0 ? c / (C / nh) : 0;
    for (int t = 0; t < T; ++t) {
        const float xv = x[(size_t) t * C + c];
        r[K - 1] = xv;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < K; ++j) s = fmaf(r[j], wr[j], s);            // ssm_conv: sumf += x*w (contracted)
        if (nh > 0) {
            const float* d = dyn + (size_t) t * K * nh + (size_t) hh * K;
#pragma unroll
            for (int lag = 0; lag < K; ++lag) s = __fadd_rn(s, __fmul_rn(r[K - 1 - lag], d[lag]));   // mul, then add
        }
        y[(size_t) t * C + c] = __fmul_rn(scale, __fadd_rn(xv, s));      // add, then scale
#pragma unroll
        for (int j = 0; j < K - 1; ++j) r[j] = r[j + 1];
    }
#pragma unroll
    for (int j = 0; j < K - 1; ++j) st[(size_t) c * (K - 1) + j] = r[j];
}

void conv_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *dyn = dst->src[3];
    const int K = param<int>(dst, 0), nh = param<int>(dst, 1);
    const float scale = param<float>(dst, 2);
    const int C = (int) x->ne[0], T = (int) x->ne[1];
    const dim3 b(128), g((C + 127) / 128);
    cudaStream_t s = (cudaStream_t) stream;
    const float* d = dyn ? (const float*) dyn->data : nullptr;
#define CONV_CASE(k) case k: k_conv_step<k><<<g, b, 0, s>>>((const float*) x->data, (const float*) w->data, (float*) st->data, d, (float*) dst->data, C, T, nh, scale); break;
    switch (K) { CONV_CASE(2) CONV_CASE(3) CONV_CASE(4) CONV_CASE(5) CONV_CASE(6) CONV_CASE(7) CONV_CASE(8) default: break; }
#undef CONV_CASE
}

void conv_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *dyn = dst->src[3];
    const int K = param<int>(dst, 0), nh = param<int>(dst, 1);
    const float scale = param<float>(dst, 2);
    const int C = (int) x->ne[0], T = (int) x->ne[1];
    const float *xp = (const float*) x->data, *wp = (const float*) w->data, *dp = dyn ? (const float*) dyn->data : nullptr;
    float *sp = (float*) st->data, *yp = (float*) dst->data;
    for (int c = ith; c < C; c += nth) {
        float r[8];
        for (int j = 0; j < K - 1; ++j) r[j] = sp[(size_t) c * (K - 1) + j];
        const int hh = nh > 0 ? c / (C / nh) : 0;
        for (int t = 0; t < T; ++t) {
            const float xv = xp[(size_t) t * C + c];
            r[K - 1] = xv;
            float s = 0.0f;
            for (int j = 0; j < K; ++j) s += r[j] * wp[(size_t) c * K + j];
            if (nh > 0)
                for (int lag = 0; lag < K; ++lag) s += r[K - 1 - lag] * dp[(size_t) t * K * nh + (size_t) hh * K + lag];
            yp[(size_t) t * C + c] = scale * (xv + s);
            for (int j = 0; j < K - 1; ++j) r[j] = r[j + 1];
        }
        for (int j = 0; j < K - 1; ++j) sp[(size_t) c * (K - 1) + j] = r[j];
    }
}

// ------------------------------------------------------------------------------------------------ SiTU
__global__ void k_situ(const float* __restrict__ gu, float* __restrict__ y, int F, int64_t n, float beta, float ibeta,
                       float lb, float ilb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / F, f = i - row * F;
    const float g = gu[row * 2 * F + f];
    float u = gu[row * 2 * F + F + f];
    float a = fmaf(beta, tanhf(fmaf(ibeta, g, 0.0f)), 0.0f);            // scale(tanh(scale(g)))  (scale = s*x + 0)
    a = __fmul_rn(a, 1.0f / (1.0f + expf(-g)));                          // * sigmoid(g)
    if (lb > 0.0f) u = fmaf(lb, tanhf(fmaf(ilb, u, 0.0f)), 0.0f);
    y[i] = __fmul_rn(a, u);
}

void situ_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor* gu = dst->src[0];
    const float beta = param<float>(dst, 0), lb = param<float>(dst, 1);
    const int F = (int) dst->ne[0];
    const int64_t n = ggml_nelements(dst);
    k_situ<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const float*) gu->data, (float*) dst->data, F, n,
                                                                          beta, 1.0f / beta, lb, lb > 0.0f ? 1.0f / lb : 0.0f);
}

void situ_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* gu = dst->src[0];
    const float beta = param<float>(dst, 0), lb = param<float>(dst, 1);
    const int64_t F = dst->ne[0], rows = ggml_nelements(dst) / F;
    const float* gp = (const float*) gu->data;
    float* yp = (float*) dst->data;
    for (int64_t r = ith; r < rows; r += nth)
        for (int64_t f = 0; f < F; ++f) {
            const float g = gp[r * 2 * F + f];
            float u = gp[r * 2 * F + F + f];
            float a = beta * std::tanh(g * (1.0f / beta)) * (1.0f / (1.0f + std::exp(-g)));
            if (lb > 0.0f) u = lb * std::tanh(u * (1.0f / lb));
            yp[r * F + f] = a * u;
        }
}


// ------------------------------------------------------------------------------------------------ stack
// dst [E, S, T] row s of token t = src_s [E, T] (S <= 10): the n concats that build mix()'s stream stack, in one copy
struct StackPtrs { const float* p[GGML_MAX_SRC]; };
__global__ void k_stack(StackPtrs sp, float* __restrict__ y, int E, int S, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) E * S * T;
    if (i >= n) return;
    const int e = (int) (i % E), s = (int) ((i / E) % S), t = (int) (i / ((int64_t) E * S));
    y[i] = sp.p[s][(int64_t) t * E + e];
}

void stack_cuda(ggml_tensor* dst, void* stream, void*) {
    const int E = (int) dst->ne[0], S = (int) dst->ne[1], T = (int) dst->ne[2];
    StackPtrs sp{};
    for (int s = 0; s < S; ++s) sp.p[s] = (const float*) dst->src[s]->data;
    const int64_t n = (int64_t) E * S * T;
    k_stack<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(sp, (float*) dst->data, E, S, T);
}

void stack_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const int64_t E = dst->ne[0], S = dst->ne[1], T = dst->ne[2];
    float* y = (float*) dst->data;
    for (int64_t r = ith; r < S * T; r += nth) {
        const int64_t s = r % S, t = r / S;
        std::memcpy(y + r * E, (const float*) dst->src[s]->data + t * E, (size_t) E * sizeof(float));
    }
}

ggml_cuda_custom_op g_conv{GGML_CUDA_CUSTOM_MAGIC, conv_cuda, nullptr};
ggml_cuda_custom_op g_situ{GGML_CUDA_CUSTOM_MAGIC, situ_cuda, nullptr};
ggml_cuda_custom_op g_stack{GGML_CUDA_CUSTOM_MAGIC, stack_cuda, nullptr};

}  // namespace

ggml_tensor* inkling_conv_step(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* state, ggml_tensor* dyn,
                               int K, int nh, float scale) {
    ggml_tensor* args[4] = {x, w, state, dyn};
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, args, dyn ? 4 : 3, conv_cpu,
                                    GGML_N_TASKS_MAX, &g_conv);
    set_param<int>(y, 0, K);
    set_param<int>(y, 1, dyn ? nh : 0);
    set_param<float>(y, 2, scale);
    return y;
}

ggml_tensor* inkling_situ(ggml_context* c, ggml_tensor* gu, float beta, float lb) {
    ggml_tensor* args[1] = {gu};
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, gu->ne[0] / 2, gu->ne[1], gu->ne[2], gu->ne[3], args, 1, situ_cpu,
                                    GGML_N_TASKS_MAX, &g_situ);
    set_param<float>(y, 0, beta);
    set_param<float>(y, 1, lb);
    return y;
}

ggml_tensor* inkling_stack(ggml_context* c, ggml_tensor* const* srcs, int n) {
    if (n < 1 || n > GGML_MAX_SRC) return nullptr;
    const int64_t E = srcs[0]->ne[0], T = ggml_nelements(srcs[0]) / E;
    for (int i = 0; i < n; ++i)
        if (srcs[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(srcs[i]) || srcs[i]->ne[0] != E || ggml_nelements(srcs[i]) != E * T)
            return nullptr;
    ggml_tensor* args[GGML_MAX_SRC];
    for (int i = 0; i < n; ++i) args[i] = srcs[i];
    return ggml_custom_4d(c, GGML_TYPE_F32, E, n, T, 1, args, n, stack_cpu, GGML_N_TASKS_MAX, &g_stack);
}

bool inkling_fused_available() { return true; }

#else   // a ggml without the hook: the custom ops would run on the CPU, so the engine keeps its op chains

ggml_tensor* inkling_conv_step(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, int, int, float) { return nullptr; }
ggml_tensor* inkling_situ(ggml_context*, ggml_tensor*, float, float) { return nullptr; }
ggml_tensor* inkling_stack(ggml_context*, ggml_tensor* const*, int) { return nullptr; }
bool inkling_fused_available() { return false; }

#endif
