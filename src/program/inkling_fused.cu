// Fused decode ops for xolotl-inkling (see inkling_fused.h).  Built with -use_fast_math like ggml-cuda, and written
// to perform the same float operations in the same order as the ggml kernels they replace (fmaf where ggml's
// compiler contracts a*b+c, __fadd_rn/__fmul_rn where ggml runs separate elementwise kernels).
#include "inkling_fused.h"

#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <tuple>
#include <vector>

#ifdef GGML_CUDA_CUSTOM_MAGIC   // the xolotl ggml patch with the CUDA custom-op hook

namespace {

// op_params[0..5] hold ggml_custom_op_params (fun, n_tasks, userdata); the op's own parameters start at int 8
constexpr int kP = 8;

template <typename T> T param(const ggml_tensor* t, int i) { T v; std::memcpy(&v, &t->op_params[kP + i], sizeof(T)); return v; }
template <typename T> void set_param(ggml_tensor* t, int i, T v) { std::memcpy(&t->op_params[kP + i], &v, sizeof(T)); }

// ------------------------------------------------------------------------------------------------ conv step
template <int K>
__global__ void k_conv_step(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ st,
                            const float* __restrict__ dyn, float* __restrict__ y, int C, int T, int nh, float scale,
                            float* __restrict__ hist, long hs, const float* __restrict__ md, const float* __restrict__ mw,
                            int mk, const float* __restrict__ res) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float r[K], wr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) wr[j] = w[(size_t) c * K + j];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) r[j] = st[(size_t) c * (K - 1) + j];
    float* hr = hist ? hist + (size_t) c * hs : nullptr;   // --dspark verify: [state ; inputs] for the exact rollback
    if (hr) {
#pragma unroll
        for (int j = 0; j < K - 1; ++j) hr[j] = r[j];
    }
    const int hh = nh > 0 ? c / (C / nh) : 0;
    for (int t = 0; t < T; ++t) {
        float xv;
        if (md) {   // MoE finish: the input is the routing-weighted sum of the k expert outputs d [C, mk, T]
            xv = 0.0f;
            for (int i = 0; i < mk; ++i) xv = fmaf(md[((size_t) t * mk + i) * C + c], mw[(size_t) t * mk + i], xv);
        } else {
            xv = x[(size_t) t * C + c];
        }
        r[K - 1] = xv;
        if (hr) hr[K - 1 + t] = xv;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < K; ++j) s = fmaf(r[j], wr[j], s);            // ssm_conv: sumf += x*w (contracted)
        if (nh > 0) {
            const float* d = dyn + (size_t) t * K * nh + (size_t) hh * K;
#pragma unroll
            for (int lag = 0; lag < K; ++lag) s = __fadd_rn(s, __fmul_rn(r[K - 1 - lag], d[lag]));   // mul, then add
        }
        const float yv = __fmul_rn(scale, __fadd_rn(xv, s));             // add, then scale
        y[(size_t) t * C + c] = res ? __fadd_rn(res[(size_t) t * C + c], yv) : yv;   // + the residual (moe finish)
#pragma unroll
        for (int j = 0; j < K - 1; ++j) r[j] = r[j + 1];
    }
#pragma unroll
    for (int j = 0; j < K - 1; ++j) st[(size_t) c * (K - 1) + j] = r[j];
}


// prefill (T large): the same convolution parallel over (channel, token) - each output reads only its own K-tap window
// (old state for negative positions), so the result is bit-identical to k_conv_step; the state (and verify history)
// update runs as a second tiny kernel after all outputs have read the old state.  k_conv_step walked T serially per
// channel on C/128 blocks (8 blocks for C = 1024): ~20 % of prefill GPU time.
template <int K>
__global__ void k_conv_par(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ st,
                           const float* __restrict__ dyn, float* __restrict__ y, int C, int T, int nh, float scale,
                           const float* __restrict__ res) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    float r[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int i = t - (K - 1) + j;
        r[j] = i < 0 ? st[(size_t) c * (K - 1) + (K - 1 + i)] : x[(size_t) i * C + c];
    }
    const float xv = r[K - 1];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < K; ++j) s = fmaf(r[j], w[(size_t) c * K + j], s);
    if (nh > 0) {
        const int hh = c / (C / nh);
        const float* d = dyn + (size_t) t * K * nh + (size_t) hh * K;
#pragma unroll
        for (int lag = 0; lag < K; ++lag) s = __fadd_rn(s, __fmul_rn(r[K - 1 - lag], d[lag]));
    }
    const float yv = __fmul_rn(scale, __fadd_rn(xv, s));
    y[(size_t) t * C + c] = res ? __fadd_rn(res[(size_t) t * C + c], yv) : yv;
}
template <int K>
__global__ void k_conv_state(const float* __restrict__ x, float* __restrict__ st, int C, int T, float* __restrict__ hist, long hs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float r[K - 1];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) r[j] = st[(size_t) c * (K - 1) + j];
    if (hist) {
        float* hr = hist + (size_t) c * hs;
#pragma unroll
        for (int j = 0; j < K - 1; ++j) hr[j] = r[j];
        for (int t = 0; t < T; ++t) hr[K - 1 + t] = x[(size_t) t * C + c];
    }
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        const int i = T - (K - 1) + j;   // the last K-1 inputs (old state when T < K-1)
        st[(size_t) c * (K - 1) + j] = i < 0 ? r[K - 1 + i] : x[(size_t) i * C + c];
    }
}
void conv_cuda(ggml_tensor* dst, void* stream, void*) {
    const int K = param<int>(dst, 0), nh = param<int>(dst, 1), has_h = param<int>(dst, 3);
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *dyn = nh > 0 ? dst->src[3] : nullptr;
    const ggml_tensor* hi = has_h ? dst->src[nh > 0 ? 4 : 3] : nullptr;
    const int mk = param<int>(dst, 4), has_r = param<int>(dst, 5);
    int si = 3 + (nh > 0) + (has_h ? 1 : 0);
    const ggml_tensor* mwt = mk > 0 ? dst->src[si++] : nullptr;
    const ggml_tensor* rst = has_r ? dst->src[si++] : nullptr;
    const float* mdp = mk > 0 ? (const float*) x->data : nullptr;
    const float* mwp = mwt ? (const float*) mwt->data : nullptr;
    const float* rsp = rst ? (const float*) rst->data : nullptr;
    float* hp = hi ? (float*) hi->data : nullptr;
    const long hs = hi ? (long) (hi->nb[1] / sizeof(float)) : 0;
    const float scale = param<float>(dst, 2);
    const int C = (int) dst->ne[0], T = (int) dst->ne[1];
    const dim3 b(128), g((C + 127) / 128);
    cudaStream_t s = (cudaStream_t) stream;
    const float* d = dyn ? (const float*) dyn->data : nullptr;
    static const bool par_ref = std::getenv("XOLOTL_INKLING_CONV_PAR_REF") != nullptr;   // A/B: the serial kernel
    if (T >= 32 && mk == 0 && !par_ref) {   // prefill: parallel over tokens, then the state/history pass
        const dim3 gp((C + 127) / 128, T);
#define CONV_PAR(k) case k: k_conv_par<k><<<gp, b, 0, s>>>((const float*) x->data, (const float*) w->data, (const float*) st->data, d, \
            (float*) dst->data, C, T, nh, scale, rsp); k_conv_state<k><<<g, b, 0, s>>>((const float*) x->data, (float*) st->data, C, T, hp, hs); break;
        switch (K) { CONV_PAR(2) CONV_PAR(3) CONV_PAR(4) CONV_PAR(5) CONV_PAR(6) CONV_PAR(7) CONV_PAR(8) default: break; }
#undef CONV_PAR
        return;
    }
#define CONV_CASE(k) case k: k_conv_step<k><<<g, b, 0, s>>>((const float*) x->data, (const float*) w->data, (float*) st->data, d, (float*) dst->data, C, T, nh, scale, hp, hs, mdp, mwp, mk, rsp); break;
    switch (K) { CONV_CASE(2) CONV_CASE(3) CONV_CASE(4) CONV_CASE(5) CONV_CASE(6) CONV_CASE(7) CONV_CASE(8) default: break; }
#undef CONV_CASE
}

void conv_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const int K = param<int>(dst, 0), nh = param<int>(dst, 1), has_h = param<int>(dst, 3);
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *dyn = nh > 0 ? dst->src[3] : nullptr;
    const ggml_tensor* hi = has_h ? dst->src[nh > 0 ? 4 : 3] : nullptr;
    const int mk = param<int>(dst, 4), has_r = param<int>(dst, 5);
    int si = 3 + (nh > 0) + (has_h ? 1 : 0);
    const float* mwp = mk > 0 ? (const float*) dst->src[si++]->data : nullptr;
    const float* rsp = has_r ? (const float*) dst->src[si++]->data : nullptr;
    const float scale = param<float>(dst, 2);
    const int C = (int) dst->ne[0], T = (int) dst->ne[1];
    const float *xp = (const float*) x->data, *wp = (const float*) w->data, *dp = dyn ? (const float*) dyn->data : nullptr;
    float *sp = (float*) st->data, *yp = (float*) dst->data;
    for (int c = ith; c < C; c += nth) {
        float r[8];
        for (int j = 0; j < K - 1; ++j) r[j] = sp[(size_t) c * (K - 1) + j];
        float* hr = hi ? (float*) ((char*) hi->data + (size_t) c * hi->nb[1]) : nullptr;
        if (hr) for (int j = 0; j < K - 1; ++j) hr[j] = r[j];
        const int hh = nh > 0 ? c / (C / nh) : 0;
        for (int t = 0; t < T; ++t) {
            float xv = 0.0f;
            if (mk > 0) for (int i = 0; i < mk; ++i) xv += xp[((size_t) t * mk + i) * C + c] * mwp[(size_t) t * mk + i];
            else xv = xp[(size_t) t * C + c];
            r[K - 1] = xv;
            if (hr) hr[K - 1 + t] = xv;
            float s = 0.0f;
            for (int j = 0; j < K; ++j) s += r[j] * wp[(size_t) c * K + j];
            if (nh > 0)
                for (int lag = 0; lag < K; ++lag) s += r[K - 1 - lag] * dp[(size_t) t * K * nh + (size_t) hh * K + lag];
            yp[(size_t) t * C + c] = (rsp ? rsp[(size_t) t * C + c] : 0.0f) + scale * (xv + s);
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


// ------------------------------------------------------------------------------------------------ clip
// y = thr * l2_norm(x [* s], eps = thr) [* tau]: rows of hd = x->ne[0], one warp per row like ggml's l2_norm_f32 for
// rows < 1024 (same strided FMA partial sums, same xor-shuffle reduction, rsqrtf(fmaxf(sum, eps^2))), so the result
// equals the mul + l2_norm + scale (+ mul) chain.  x [hd, nh, T] contiguous, s [hd, nh] or null, tau [T] or null.
__global__ void k_clip(const float* __restrict__ x, const float* __restrict__ sc, const float* __restrict__ tau,
                       float* __restrict__ y, int hd, int nh, float thr) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int h = row % nh, t = row / nh;
    const float* xr = x + (size_t) row * hd;
    float tmp = 0.0f;
    for (int col = lane; col < hd; col += 32) {
        float v = xr[col];
        if (sc) v = __fmul_rn(v, sc[(size_t) h * hd + col]);
        tmp = fmaf(v, v, tmp);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, off, 32);
    const float scale = rsqrtf(fmaxf(tmp, thr * thr));
    for (int col = lane; col < hd; col += 32) {
        float v = xr[col];
        if (sc) v = __fmul_rn(v, sc[(size_t) h * hd + col]);
        float o = __fmul_rn(thr, __fmul_rn(scale, v));
        if (tau) o = __fmul_rn(o, tau[t]);
        y[(size_t) row * hd + col] = o;
    }
}

void clip_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *x = dst->src[0], *sc = dst->src[1], *tau = dst->src[2];
    const int has_s = param<int>(dst, 1), has_t = param<int>(dst, 2);
    const int hd = (int) x->ne[0], nh = (int) x->ne[1], rows = (int) (ggml_nelements(x) / hd);
    k_clip<<<rows, 32, 0, (cudaStream_t) stream>>>((const float*) x->data, has_s ? (const float*) sc->data : nullptr,
                                                   has_t ? (const float*) tau->data : nullptr, (float*) dst->data, hd, nh,
                                                   param<float>(dst, 0));
}

void clip_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *x = dst->src[0], *sc = dst->src[1], *tau = dst->src[2];
    const int has_s = param<int>(dst, 1), has_t = param<int>(dst, 2);
    const float thr = param<float>(dst, 0);
    const int hd = (int) x->ne[0], nh = (int) x->ne[1], rows = (int) (ggml_nelements(x) / hd);
    for (int row = ith; row < rows; row += nth) {
        const int h = row % nh, t = row / nh;
        const float* xr = (const float*) x->data + (size_t) row * hd;
        double sum = 0.0;
        for (int c = 0; c < hd; ++c) { float v = xr[c] * (has_s ? ((const float*) sc->data)[(size_t) h * hd + c] : 1.0f); sum += (double) v * v; }
        const float scale = 1.0f / std::sqrt(std::max((float) sum, thr * thr));
        for (int c = 0; c < hd; ++c) {
            float v = xr[c] * (has_s ? ((const float*) sc->data)[(size_t) h * hd + c] : 1.0f);
            float o = thr * (scale * v);
            if (has_t) o *= ((const float*) tau->data)[t];
            ((float*) dst->data)[(size_t) row * hd + c] = o;
        }
    }
}


// ------------------------------------------------------------------------------------------------ mix
// depth attention in one kernel: streams x_s [E, T] (s < S <= 9), queries q [E, nq] (nq <= 4):
//   n_s = rms_norm(x_s, 1e-6);  p[:, j] = softmax_s(<n_s, q_j> / sqrt(E));  y[:, j, t] = sum_s p[s, j] n_s
// == stack + rms_norm + mul_mat + soft_max + transpose + mul_mat (6 kernels), in fp32 with a different summation order
// (rounding-level differences, like any reduction change).  One block per token.
constexpr int kMixS = 9, kMixQ = 4, kMixBS = 512;
struct MixPtrs { const float* p[kMixS]; };
template <int S, int NQ>
__global__ void __launch_bounds__(kMixBS) k_mix(MixPtrs xs, const float* __restrict__ q, float* __restrict__ y, int E, float isq) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int NV = S * (1 + NQ);
    float part[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) part[i] = 0.0f;
    for (int e = tid; e < E; e += kMixBS) {
        float qe[NQ];
#pragma unroll
        for (int j = 0; j < NQ; ++j) qe[j] = q[(size_t) j * E + e];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float v = xs.p[s][(size_t) t * E + e];
            part[s] = fmaf(v, v, part[s]);
#pragma unroll
            for (int j = 0; j < NQ; ++j) part[S + s * NQ + j] = fmaf(v, qe[j], part[S + s * NQ + j]);
        }
    }
    __shared__ float red[kMixBS / 32][NV];
    __shared__ float pw[S][NQ];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        float v = part[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off, 32);
        if (lane == 0) red[warp][i] = v;
    }
    __syncthreads();
    if (warp == 0) {   // lane i owns value i (NV <= 32): the totals, then one lane per query does the softmax
        float tot = 0.0f;
        if (lane < NV)
#pragma unroll
            for (int w = 0; w < kMixBS / 32; ++w) tot += red[w][lane];
        float r[S];
#pragma unroll
        for (int s = 0; s < S; ++s) r[s] = rsqrtf(__shfl_sync(0xffffffff, tot, s, 32) / (float) E + 1e-6f);
        float sc[S];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < NQ; ++j) { const float dj = __shfl_sync(0xffffffff, tot, S + s * NQ + j, 32); if (j == lane) d = dj; }
            sc[s] = d * r[s] * isq;
        }
        if (lane < NQ) {
            float mx = -INFINITY;
#pragma unroll
            for (int s = 0; s < S; ++s) mx = fmaxf(mx, sc[s]);
            float ev[S], sum = 0.0f;
#pragma unroll
            for (int s = 0; s < S; ++s) { ev[s] = expf(sc[s] - mx); sum += ev[s]; }
#pragma unroll
            for (int s = 0; s < S; ++s) pw[s][lane] = ev[s] / sum * r[s];   // the stream's rms scale folded in
        }
    }
    __syncthreads();
    for (int e = tid; e < E; e += kMixBS) {
        float acc[NQ];
#pragma unroll
        for (int j = 0; j < NQ; ++j) acc[j] = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float v = xs.p[s][(size_t) t * E + e];
#pragma unroll
            for (int j = 0; j < NQ; ++j) acc[j] = fmaf(v, pw[s][j], acc[j]);
        }
#pragma unroll
        for (int j = 0; j < NQ; ++j) y[((size_t) t * NQ + j) * E + e] = acc[j];
    }
}

template <int S>
void mix_launch(int nq, MixPtrs xs, const float* q, float* y, int E, int T, cudaStream_t st) {
    const float isq = 1.0f / std::sqrt((float) E);
    switch (nq) {
        case 1: k_mix<S, 1><<<T, kMixBS, 0, st>>>(xs, q, y, E, isq); break;
        case 2: k_mix<S, 2><<<T, kMixBS, 0, st>>>(xs, q, y, E, isq); break;
        case 3: k_mix<S, 3><<<T, kMixBS, 0, st>>>(xs, q, y, E, isq); break;
        case 4: k_mix<S, 4><<<T, kMixBS, 0, st>>>(xs, q, y, E, isq); break;
        default: break;
    }
}

void mix_cuda(ggml_tensor* dst, void* stream, void*) {
    const int S = param<int>(dst, 0), E = (int) dst->ne[0], nq = (int) dst->ne[1], T = (int) dst->ne[2];
    MixPtrs xs{};
    for (int s = 0; s < S; ++s) xs.p[s] = (const float*) dst->src[s]->data;
    const float* q = (const float*) dst->src[S]->data;
    float* y = (float*) dst->data;
    cudaStream_t st = (cudaStream_t) stream;
    switch (S) {
        case 1: mix_launch<1>(nq, xs, q, y, E, T, st); break;
        case 2: mix_launch<2>(nq, xs, q, y, E, T, st); break;
        case 3: mix_launch<3>(nq, xs, q, y, E, T, st); break;
        case 4: mix_launch<4>(nq, xs, q, y, E, T, st); break;
        case 5: mix_launch<5>(nq, xs, q, y, E, T, st); break;
        case 6: mix_launch<6>(nq, xs, q, y, E, T, st); break;
        default: break;
    }
}

void mix_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const int S = param<int>(dst, 0), E = (int) dst->ne[0], nq = (int) dst->ne[1], T = (int) dst->ne[2];
    const float* q = (const float*) dst->src[S]->data;
    for (int t = ith; t < T; t += nth) {
        float r[kMixS], sc[kMixS][kMixQ];
        for (int s = 0; s < S; ++s) {
            const float* x = (const float*) dst->src[s]->data + (size_t) t * E;
            double ss = 0; for (int e = 0; e < E; ++e) ss += (double) x[e] * x[e];
            r[s] = 1.0f / std::sqrt((float) (ss / E) + 1e-6f);
            for (int j = 0; j < nq; ++j) { double d = 0; for (int e = 0; e < E; ++e) d += (double) x[e] * q[(size_t) j * E + e]; sc[s][j] = (float) d * r[s] / std::sqrt((float) E); }
        }
        for (int j = 0; j < nq; ++j) {
            float mx = -INFINITY, sum = 0; for (int s = 0; s < S; ++s) mx = std::max(mx, sc[s][j]);
            float pw[kMixS]; for (int s = 0; s < S; ++s) { pw[s] = std::exp(sc[s][j] - mx); sum += pw[s]; }
            float* yo = (float*) dst->data + ((size_t) t * nq + j) * E;
            for (int e = 0; e < E; ++e) { float a = 0; for (int s = 0; s < S; ++s) a += ((const float*) dst->src[s]->data)[(size_t) t * E + e] * r[s] * pw[s] / sum; yo[e] = a; }
        }
    }
}

// ------------------------------------------------------------------------------------------------ relative-bias mask
// M[j, t, h] = (table[rp[j, t], h, t] - sink[h, t]) + mask[j, t]  ->  [cells, T, nh] (F32, or F16 for flash attention)
// == sub (table - sinks) + permute/cont + get_rows (rows of nh floats) + permute/cont + add (+ cast): the same two IEEE
// operations per element in the same order, so the mask is bit-identical; one block per (t, h) row, coalesced in j.
// table [Ext1, nh, T] contiguous F32 (the bias-table mat-mul output), sink [nh, T], rp [cells, T] I32, mask [cells, T].
template <typename To>
__global__ void k_relbias(const float* __restrict__ table, const float* __restrict__ sink, const int* __restrict__ rp,
                          const float* __restrict__ mask, To* __restrict__ out, int cells, int T, int nh, int ext1,
                          long rp_s1, long mask_s1) {
    const int t = blockIdx.x, h = blockIdx.y;
    const float* tr = table + ((size_t) t * nh + h) * ext1;
    const float sk = sink[(size_t) t * nh + h];
    const int* rr = rp + (size_t) t * rp_s1;
    const float* mr = mask + (size_t) t * mask_s1;
    To* o = out + ((size_t) h * T + t) * cells;
    for (int j = threadIdx.x; j < cells; j += blockDim.x) {
        const float v = __fadd_rn(__fsub_rn(tr[rr[j]], sk), mr[j]);
        if constexpr (sizeof(To) == 2) o[j] = __float2half(v); else o[j] = v;
    }
}

void relbias_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *tb = dst->src[0], *sk = dst->src[1], *rp = dst->src[2], *mk = dst->src[3];
    const int cells = (int) dst->ne[0], T = (int) dst->ne[1], nh = (int) dst->ne[2], ext1 = (int) tb->ne[0];
    const dim3 grid(T, nh);
    const int threads = cells >= 512 ? 256 : 128;
    const long rs = (long) (rp->nb[1] / sizeof(int)), ms = (long) (mk->nb[1] / sizeof(float));
    if (dst->type == GGML_TYPE_F16)
        k_relbias<__half><<<grid, threads, 0, (cudaStream_t) stream>>>((const float*) tb->data, (const float*) sk->data,
            (const int*) rp->data, (const float*) mk->data, (__half*) dst->data, cells, T, nh, ext1, rs, ms);
    else
        k_relbias<float><<<grid, threads, 0, (cudaStream_t) stream>>>((const float*) tb->data, (const float*) sk->data,
            (const int*) rp->data, (const float*) mk->data, (float*) dst->data, cells, T, nh, ext1, rs, ms);
}

void relbias_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *tb = dst->src[0], *sk = dst->src[1], *rp = dst->src[2], *mk = dst->src[3];
    const int cells = (int) dst->ne[0], T = (int) dst->ne[1], nh = (int) dst->ne[2], ext1 = (int) tb->ne[0];
    for (int r = ith; r < T * nh; r += nth) {
        const int t = r % T, h = r / T;
        const float* tr = (const float*) tb->data + ((size_t) t * nh + h) * ext1;
        const float s = ((const float*) sk->data)[(size_t) t * nh + h];
        const int* rr = (const int*) ((const char*) rp->data + (size_t) t * rp->nb[1]);
        const float* mr = (const float*) ((const char*) mk->data + (size_t) t * mk->nb[1]);
        for (int j = 0; j < cells; ++j) {
            const float v = (tr[rr[j]] - s) + mr[j];
            if (dst->type == GGML_TYPE_F16) ((ggml_fp16_t*) dst->data)[((size_t) h * T + t) * cells + j] = ggml_fp32_to_fp16(v);
            else ((float*) dst->data)[((size_t) h * T + t) * cells + j] = v;
        }
    }
}


// ------------------------------------------------------------------------------------------------ decode attention
// Split-K flash decode over the f16 K/V window, read in place (the reference chain makes a contiguous transposed copy
// of V every step, a cost that grows with the context).  One block per (kv head, token, 128-cell split); the R query
// heads of a kv head share the block, so K and V are read once.  Cells whose mask is -inf for all R heads are skipped
// (the sliding window, future cells).  Partials [HD + 2, nh, T, nsplit] = (sum_j p_j v_j, max, sum_j p_j); the reduce
// joins the splits with the zero sink logit.  fp32 math throughout (the chain rounds q and p to f16 for the GEMMs).
constexpr int kDecChunk = 64, kDecThreads = 128, kDecWarps = kDecThreads / 32, kDecU = 4;   // kDecU cells per warp step

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// TT tokens per block (verify): RR = TT * R query rows (row = (token t0 + rr / R, head g * R + rr % R)) share each K/V
// load.  Per-row arithmetic is identical to TT = 1, so the partials are bit-identical whatever TT is.
template <int HD, int R, int TT>
__global__ void __launch_bounds__(kDecThreads) k_attn_dec_part(const float* __restrict__ q, const char* __restrict__ K,
        const char* __restrict__ V, const float* __restrict__ mask, float* __restrict__ part, int cells, int T, int nh,
        long k_nb1, long k_nb2, long v_nb1, long v_nb2, float scale, int f16r,
        const float* __restrict__ relb, const float* __restrict__ Rt, const float* __restrict__ sinks,
        const int* __restrict__ rp, long rp_s1, long mask_s1, int nrel) {
    constexpr int RR = R * TT;
    const int g = blockIdx.x, t0 = blockIdx.y * TT, sp = blockIdx.z;
    const int j0 = sp * kDecChunk, n = min(cells - j0, kDecChunk);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    constexpr int PL = HD / 32;                      // dims per lane in the score pass (even)
    __shared__ float s[RR][kDecChunk];
    __shared__ float red[RR][kDecWarps];
    __shared__ float stat[2][RR];
    __shared__ float Rs[RR][32], sks[RR];   // relb mode: the block's relative states and sink logits
    auto row_t = [&](int rr) { return t0 + rr / R; };
    auto row_h = [&](int rr) { return g * R + rr % R; };
    if (relb) {
        for (int i = threadIdx.x; i < RR * nrel; i += kDecThreads) {
            const int rr = i / nrel, k = i % nrel;
            Rs[rr][k] = row_t(rr) < T ? Rt[((size_t) row_t(rr) * nh + row_h(rr)) * nrel + k] : 0.0f;
        }
        if (threadIdx.x < RR) sks[threadIdx.x] = row_t(threadIdx.x) < T ? sinks[(size_t) row_t(threadIdx.x) * nh + row_h(threadIdx.x)] : 0.0f;
        __syncthreads();
        // every (cell, row) bias of this split in parallel, into s[][] (phase 1 reads it as the mask, then
        // overwrites it with the score)
        for (int idx = threadIdx.x; idx < n * RR; idx += kDecThreads) {
            const int i = idx / RR, rr = idx % RR, t = row_t(rr);
            float mvv = -INFINITY;
            if (t < T) {
                const float m0 = mask[(size_t) t * mask_s1 + j0 + i];
                if (m0 != -INFINITY) {
                    const float* tb = relb + (size_t) rp[(size_t) t * rp_s1 + j0 + i] * nrel;
                    float b = 0.0f;
                    for (int k = 0; k < nrel; ++k) b = fmaf(tb[k], Rs[rr][k], b);
                    mvv = __fadd_rn(__fsub_rn(b, sks[rr]), m0);
                }
            }
            s[rr][i] = mvv;
        }
        __syncthreads();
    }
    float qr[RR][PL];
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
        const int t = min(row_t(rr), T - 1);
        const float* qh = q + ((size_t) t * nh + row_h(rr)) * HD + lane * PL;
#pragma unroll
        for (int i = 0; i < PL; ++i) qr[rr][i] = f16r ? __half2float(__float2half(qh[i])) : qh[i];   // f16r: the chain's f16 q
    }
    // 1. scores: one warp per cell, kDecU cells per step with their K loads issued together
    for (int i0 = w; i0 < n; i0 += kDecWarps * kDecU) {
        half2 kv[kDecU][PL / 2];
        float mv[kDecU][RR];
        bool live[kDecU];
#pragma unroll
        for (int u = 0; u < kDecU; ++u) {
            const int i = i0 + u * kDecWarps;
            live[u] = false;
            if (i < n) {
                if (relb) {   // the prologue left the full bias + mask in s[][]
#pragma unroll
                    for (int rr = 0; rr < RR; ++rr) { mv[u][rr] = s[rr][i]; live[u] |= mv[u][rr] != -INFINITY; }
                } else {
#pragma unroll
                    for (int rr = 0; rr < RR; ++rr) {
                        const int t = row_t(rr);
                        mv[u][rr] = t < T ? mask[((size_t) row_h(rr) * T + t) * cells + j0 + i] : -INFINITY;
                        live[u] |= mv[u][rr] != -INFINITY;
                    }
                }
            }
            if (live[u]) {
                const half2* kr = (const half2*) (K + (size_t) (j0 + i) * k_nb2 + (size_t) g * k_nb1) + lane * (PL / 2);
#pragma unroll
                for (int i2 = 0; i2 < PL / 2; ++i2) kv[u][i2] = kr[i2];
            }
        }
#pragma unroll
        for (int u = 0; u < kDecU; ++u) {
            const int i = i0 + u * kDecWarps;
            if (i >= n) break;
            if (!live[u]) { if (lane == 0) { for (int rr = 0; rr < RR; ++rr) s[rr][i] = -INFINITY; } continue; }
            float d[RR];
#pragma unroll
            for (int rr = 0; rr < RR; ++rr) d[rr] = 0.0f;
#pragma unroll
            for (int i2 = 0; i2 < PL / 2; ++i2) {
                const float2 f = __half22float2(kv[u][i2]);
#pragma unroll
                for (int rr = 0; rr < RR; ++rr) d[rr] = fmaf(qr[rr][2 * i2 + 1], f.y, fmaf(qr[rr][2 * i2], f.x, d[rr]));
            }
#pragma unroll
            for (int rr = 0; rr < RR; ++rr) d[rr] = warp_sum(d[rr]);
            if (lane == 0) {
#pragma unroll
                for (int rr = 0; rr < RR; ++rr) s[rr][i] = mv[u][rr] == -INFINITY ? -INFINITY : fmaf(d[rr], scale, mv[u][rr]);
            }
        }
    }
    __syncthreads();
    // 2. per-row max, p = exp(s - max), sum
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < n; i += kDecThreads) m = fmaxf(m, s[rr][i]);
        m = warp_max(m);
        if (lane == 0) red[rr][w] = m;
    }
    __syncthreads();
    if (threadIdx.x < RR) {
        float m = -INFINITY;
        for (int k = 0; k < kDecWarps; ++k) m = fmaxf(m, red[threadIdx.x][k]);
        stat[0][threadIdx.x] = m;
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
        const float m = stat[0][rr];
        float l = 0.0f;
        for (int i = threadIdx.x; i < n; i += kDecThreads) {
            const float p = m == -INFINITY ? 0.0f : __expf(s[rr][i] - m);
            s[rr][i] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) red[rr][w] = l;
    }
    __syncthreads();
    if (threadIdx.x < RR) {
        float l = 0.0f;
        for (int k = 0; k < kDecWarps; ++k) l += red[threadIdx.x][k];
        stat[1][threadIdx.x] = l;
    }
    // 3. sum_j p_j v_j: thread = (dim pair, cell group); groups split the cells, then join through shared memory
    constexpr int NP = HD / 2, NG = kDecThreads / NP > 0 ? kDecThreads / NP : 1;
    __shared__ float accs[NG > 1 ? NG - 1 : 1][RR][HD];
    const int pr = threadIdx.x % NP, cg = threadIdx.x / NP;
    float a0[RR], a1[RR];
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) { a0[rr] = 0.0f; a1[rr] = 0.0f; }
    if (cg < NG) {
#pragma unroll 4
        for (int i = cg; i < n; i += NG) {
            bool any = false;
#pragma unroll
            for (int rr = 0; rr < RR; ++rr) any |= s[rr][i] != 0.0f;
            if (!any) continue;
            const float2 f = __half22float2(((const half2*) (V + (size_t) (j0 + i) * v_nb2 + (size_t) g * v_nb1))[pr]);
#pragma unroll
            for (int rr = 0; rr < RR; ++rr) {
                const float pv = f16r ? __half2float(__float2half(s[rr][i])) : s[rr][i];
                a0[rr] = fmaf(pv, f.x, a0[rr]); a1[rr] = fmaf(pv, f.y, a1[rr]);
            }
        }
    }
    __syncthreads();   // stat[1] visible; s no longer written
    if (NG > 1 && cg > 0 && cg < NG) {
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) { accs[cg - 1][rr][2 * pr] = a0[rr]; accs[cg - 1][rr][2 * pr + 1] = a1[rr]; }
    }
    __syncthreads();
    if (cg == 0) {
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) {
            const int t = row_t(rr);
            if (t >= T) continue;
            for (int k = 0; k + 1 < NG; ++k) { a0[rr] += accs[k][rr][2 * pr]; a1[rr] += accs[k][rr][2 * pr + 1]; }
            float* o = part + (((size_t) sp * T + t) * nh + row_h(rr)) * (HD + 2);
            o[2 * pr] = a0[rr];
            o[2 * pr + 1] = a1[rr];
            if (pr == 0) { o[HD] = stat[0][rr]; o[HD + 1] = stat[1][rr]; }
        }
    }
}

template <int HD>
__global__ void k_attn_dec_reduce(const float* __restrict__ part, float* __restrict__ out, int T, int nh, int nsplit,
                                  const float* __restrict__ gate, long gs) {
    const int h = blockIdx.x, t = blockIdx.y;
    auto P = [&](int sp) { return part + (((size_t) sp * T + t) * nh + h) * (HD + 2); };
    float M = 0.0f;                                   // the zero sink logit joins the max
    for (int sp = 0; sp < nsplit; ++sp) M = fmaxf(M, P(sp)[HD]);
    float den = __expf(-M);                           // the sink's exp(0 - M)
    for (int sp = 0; sp < nsplit; ++sp) { const float m = P(sp)[HD]; if (m != -INFINITY) den += P(sp)[HD + 1] * __expf(m - M); }
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float a = 0.0f;
        for (int sp = 0; sp < nsplit; ++sp) { const float m = P(sp)[HD]; if (m != -INFINITY) a = fmaf(P(sp)[d], __expf(m - M), a); }
        float o = a / den;
        if (gate) {   // the attention output gate, o * sigmoid(g) (ggml's sigmoid, then mul)
            const float gv = gate[(size_t) t * gs + (size_t) h * HD + d];
            o = __fmul_rn(o, 1.0f / (1.0f + expf(-gv)));
        }
        out[((size_t) t * nh + h) * HD + d] = o;
    }
}

template <int HD, int TT>
void attn_dec_part_launch_tt(int R, dim3 grid, cudaStream_t st, const float* q, const char* K, const char* V, const float* mk,
                          float* part, int cells, int T, int nh, long a, long b, long c, long d, float scale, int f16r,
                          const float* relb, const float* Rt, const float* sk, const int* rp, long rps, long mks, int nrel) {
    switch (R) {
        case 1: k_attn_dec_part<HD, 1, TT><<<grid, kDecThreads, 0, st>>>(q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel); break;
        case 2: k_attn_dec_part<HD, 2, TT><<<grid, kDecThreads, 0, st>>>(q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel); break;
        case 3: k_attn_dec_part<HD, 3, TT><<<grid, kDecThreads, 0, st>>>(q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel); break;
        default: k_attn_dec_part<HD, 4, TT><<<grid, kDecThreads, 0, st>>>(q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel); break;
    }
}

// verify rows per block: XOLOTL_INKLING_DECATTN_TT (1, 2 or 4; default 1: bit-identical, but 2/4 measured 1-2 % slower in t9); decode (T = 1) always 1
static int dec_tt(int T) {
    static const int tt = [] { const char* e = std::getenv("XOLOTL_INKLING_DECATTN_TT"); const int v = e ? std::atoi(e) : 1; return v >= 4 ? 4 : v >= 2 ? 2 : 1; }();
    return T <= 1 ? 1 : tt;
}

template <int HD>
void attn_dec_part_launch(int R, dim3 grid, cudaStream_t st, const float* q, const char* K, const char* V, const float* mk,
                          float* part, int cells, int T, int nh, long a, long b, long c, long d, float scale, int f16r,
                          const float* relb, const float* Rt, const float* sk, const int* rp, long rps, long mks, int nrel) {
    const int tt = dec_tt(T);
    grid.y = (T + tt - 1) / tt;
    if (tt == 4) attn_dec_part_launch_tt<HD, 4>(R, grid, st, q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel);
    else if (tt == 2) attn_dec_part_launch_tt<HD, 2>(R, grid, st, q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel);
    else attn_dec_part_launch_tt<HD, 1>(R, grid, st, q, K, V, mk, part, cells, T, nh, a, b, c, d, scale, f16r, relb, Rt, sk, rp, rps, mks, nrel);
}

void attn_dec_part_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mk = dst->src[3];
    const int HD = (int) q->ne[0], nh = (int) q->ne[1], T = (int) q->ne[2], nkv = (int) k->ne[1], cells = (int) k->ne[2];
    const int nsplit = (int) dst->ne[3];
    const float scale = param<float>(dst, 0);
    const int rmode = param<int>(dst, 2);   // relb mode: srcs q, k, v, mask[cells,T], relb, R, sinks, rp
    const float* relb = rmode ? (const float*) dst->src[4]->data : nullptr;
    const float* Rt = rmode ? (const float*) dst->src[5]->data : nullptr;
    const float* skp = rmode ? (const float*) dst->src[6]->data : nullptr;
    const int* rpp = rmode ? (const int*) dst->src[7]->data : nullptr;
    const long rps = rmode ? (long) (dst->src[7]->nb[1] / sizeof(int)) : 0, mks = rmode ? (long) (mk->nb[1] / sizeof(float)) : 0;
    const int nrel = rmode ? (int) dst->src[4]->ne[0] : 0;
    const dim3 grid(nkv, T, nsplit);
    cudaStream_t st = (cudaStream_t) stream;
    const auto args = std::make_tuple((const float*) q->data, (const char*) k->data, (const char*) v->data, (const float*) mk->data, (float*) dst->data);
    if (HD == 64) attn_dec_part_launch<64>(nh / nkv, grid, st, std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args), std::get<4>(args), cells, T, nh, (long) k->nb[1], (long) k->nb[2], (long) v->nb[1], (long) v->nb[2], scale, param<int>(dst, 1), relb, Rt, skp, rpp, rps, mks, nrel);
    else if (HD == 128) attn_dec_part_launch<128>(nh / nkv, grid, st, std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args), std::get<4>(args), cells, T, nh, (long) k->nb[1], (long) k->nb[2], (long) v->nb[1], (long) v->nb[2], scale, param<int>(dst, 1), relb, Rt, skp, rpp, rps, mks, nrel);
    else attn_dec_part_launch<256>(nh / nkv, grid, st, std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args), std::get<4>(args), cells, T, nh, (long) k->nb[1], (long) k->nb[2], (long) v->nb[1], (long) v->nb[2], scale, param<int>(dst, 1), relb, Rt, skp, rpp, rps, mks, nrel);
}

void attn_dec_reduce_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor* pt = dst->src[0];
    const ggml_tensor* gt = param<int>(dst, 0) ? dst->src[1] : nullptr;
    const float* gp = gt ? (const float*) gt->data : nullptr;
    const long gs = gt ? (long) (gt->nb[1] / sizeof(float)) : 0;
    const int HD = (int) dst->ne[0], nh = (int) dst->ne[1], T = (int) dst->ne[2], nsplit = (int) pt->ne[3];
    const dim3 grid(nh, T);
    cudaStream_t st = (cudaStream_t) stream;
    if (HD == 64) k_attn_dec_reduce<64><<<grid, 64, 0, st>>>((const float*) pt->data, (float*) dst->data, T, nh, nsplit, gp, gs);
    else if (HD == 128) k_attn_dec_reduce<128><<<grid, 128, 0, st>>>((const float*) pt->data, (float*) dst->data, T, nh, nsplit, gp, gs);
    else k_attn_dec_reduce<256><<<grid, 128, 0, st>>>((const float*) pt->data, (float*) dst->data, T, nh, nsplit, gp, gs);
}

// CPU fallbacks: the same partials / reduce in plain loops
void attn_dec_part_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mk = dst->src[3];
    const int HD = (int) q->ne[0], nh = (int) q->ne[1], T = (int) q->ne[2], nkv = (int) k->ne[1], cells = (int) k->ne[2];
    const int nsplit = (int) dst->ne[3], R = nh / nkv;
    const float scale = param<float>(dst, 0);
    std::vector<float> sc(kDecChunk);
    for (int u = ith; u < nh * T * nsplit; u += nth) {
        const int h = u % nh, t = (u / nh) % T, sp = u / (nh * T), g = h / R;
        const int j0 = sp * kDecChunk, n = std::min(cells - j0, kDecChunk);
        const float* qh = (const float*) q->data + ((size_t) t * nh + h) * HD;
        float m = -INFINITY;
        for (int i = 0; i < n; ++i) {
            float mv;
            if (param<int>(dst, 2)) {   // relb mode
                const ggml_tensor *rb = dst->src[4], *Rt = dst->src[5], *sk = dst->src[6], *rpt = dst->src[7];
                const int nrel = (int) rb->ne[0];
                const float m0 = *(const float*) ((const char*) mk->data + (size_t) t * mk->nb[1] + (size_t) (j0 + i) * 4);
                if (m0 == -INFINITY) mv = -INFINITY;
                else {
                    const int e = *(const int*) ((const char*) rpt->data + (size_t) t * rpt->nb[1] + (size_t) (j0 + i) * 4);
                    float b = 0.0f;
                    for (int kk = 0; kk < nrel; ++kk) b += ((const float*) rb->data)[(size_t) e * nrel + kk] * ((const float*) Rt->data)[((size_t) t * nh + h) * nrel + kk];
                    mv = (b - ((const float*) sk->data)[(size_t) t * nh + h]) + m0;
                }
            } else {
                mv = ((const float*) mk->data)[((size_t) h * T + t) * cells + j0 + i];
            }
            float d = 0.0f;
            if (mv != -INFINITY) {
                const ggml_fp16_t* kr = (const ggml_fp16_t*) ((const char*) k->data + (size_t) (j0 + i) * k->nb[2] + (size_t) g * k->nb[1]);
                for (int e = 0; e < HD; ++e) d += qh[e] * ggml_fp16_to_fp32(kr[e]);
            }
            sc[i] = mv == -INFINITY ? -INFINITY : d * scale + mv;
            m = std::max(m, sc[i]);
        }
        float* o = (float*) dst->data + (((size_t) sp * T + t) * nh + h) * (HD + 2);
        float l = 0.0f;
        for (int e = 0; e < HD; ++e) o[e] = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float p = m == -INFINITY ? 0.0f : std::exp(sc[i] - m);
            l += p;
            if (p == 0.0f) continue;
            const ggml_fp16_t* vr = (const ggml_fp16_t*) ((const char*) v->data + (size_t) (j0 + i) * v->nb[2] + (size_t) g * v->nb[1]);
            for (int e = 0; e < HD; ++e) o[e] += p * ggml_fp16_to_fp32(vr[e]);
        }
        o[HD] = m; o[HD + 1] = l;
    }
}
void attn_dec_reduce_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* pt = dst->src[0];
    const int HD = (int) dst->ne[0], nh = (int) dst->ne[1], T = (int) dst->ne[2], nsplit = (int) pt->ne[3];
    for (int u = ith; u < nh * T; u += nth) {
        const int h = u % nh, t = u / nh;
        auto P = [&](int sp) { return (const float*) pt->data + (((size_t) sp * T + t) * nh + h) * (HD + 2); };
        float M = 0.0f;
        for (int sp = 0; sp < nsplit; ++sp) M = std::max(M, P(sp)[HD]);
        float den = std::exp(-M);
        for (int sp = 0; sp < nsplit; ++sp) if (P(sp)[HD] != -INFINITY) den += P(sp)[HD + 1] * std::exp(P(sp)[HD] - M);
        float* o = (float*) dst->data + ((size_t) t * nh + h) * HD;
        for (int e = 0; e < HD; ++e) {
            float a = 0.0f;
            for (int sp = 0; sp < nsplit; ++sp) if (P(sp)[HD] != -INFINITY) a += P(sp)[e] * std::exp(P(sp)[HD] - M);
            o[e] = a / den;
            if (param<int>(dst, 0)) {
                const ggml_tensor* gt = dst->src[1];
                const float gv = *(const float*) ((const char*) gt->data + (size_t) t * gt->nb[1] + (size_t) (h * HD + e) * sizeof(float));
                o[e] *= 1.0f / (1.0f + std::exp(-gv));
            }
        }
    }
}


// ------------------------------------------------------------------------------------------------ attention prepare
// One kernel per visit for the key/value side: K and V convolutions (state, optional dynamic taps, optional verify
// history), the key clip thr*x/max(|x|,thr) per head row, the value RMS norm per head row, and both f16 cache writes at
// the step's cells.  Block = (kv head, K|V), thread = channel.  The row reductions are done by warp 0 in exactly the
// order of k_clip / ggml's 32-thread rms_norm, so the values match the chain they replace.
template <int K>
__global__ void k_attn_prep(const float* __restrict__ x, long xs, const float* __restrict__ w, float* __restrict__ state,
                            int off, const float* __restrict__ dyn, int nhd, float scale, const int64_t* __restrict__ idx,
                            char* __restrict__ cache, long c1, long c2, float* __restrict__ hist, long hs, int T, float thr,
                            float eps, int isv) {
    const int g = blockIdx.x, tid = threadIdx.x, hd = blockDim.x;
    const int C = gridDim.x * hd, ch = g * hd + tid, srow = off + ch;
    __shared__ float yb[256];
    __shared__ float nrm;
    float r[K], wr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) wr[j] = w[(size_t) ch * K + j];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) r[j] = state[(size_t) srow * (K - 1) + j];
    float* hr = hist ? hist + (size_t) srow * hs : nullptr;
    if (hr) {
#pragma unroll
        for (int j = 0; j < K - 1; ++j) hr[j] = r[j];
    }
    const int hh = dyn ? ch / (C / nhd) : 0;
    for (int t = 0; t < T; ++t) {
        const float xval = x[(size_t) t * xs + ch];
        r[K - 1] = xval;
        if (hr) hr[K - 1 + t] = xval;
        float sacc = 0.0f;
#pragma unroll
        for (int j = 0; j < K; ++j) sacc = fmaf(r[j], wr[j], sacc);
        if (dyn) {
            const float* d = dyn + (size_t) t * K * nhd + (size_t) hh * K;
#pragma unroll
            for (int lag = 0; lag < K; ++lag) sacc = __fadd_rn(sacc, __fmul_rn(r[K - 1 - lag], d[lag]));
        }
        const float y = __fmul_rn(scale, __fadd_rn(xval, sacc));
#pragma unroll
        for (int j = 0; j < K - 1; ++j) r[j] = r[j + 1];
        yb[tid] = y;
        __syncthreads();
        if (tid < 32) {
            float tmp = 0.0f;
            for (int col = tid; col < hd; col += 32) tmp = fmaf(yb[col], yb[col], tmp);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, o, 32);
            if (tid == 0) nrm = isv ? rsqrtf(tmp / hd + eps) : rsqrtf(fmaxf(tmp, thr * thr));
        }
        __syncthreads();
        const float o = isv ? __fmul_rn(nrm, y) : __fmul_rn(thr, __fmul_rn(nrm, y));
        ((__half*) (cache + (size_t) idx[t] * c2 + (size_t) g * c1))[tid] = __float2half(o);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < K - 1; ++j) state[(size_t) srow * (K - 1) + j] = r[j];
}

void attn_prep_cuda(ggml_tensor* dst, void* stream, void*) {
    // srcs: x, w, state, idx, cache, [dyn], [hist]
    const int K = param<int>(dst, 0), nhd = param<int>(dst, 1), has_h = param<int>(dst, 2), off = param<int>(dst, 3);
    const float scale = param<float>(dst, 4), thr = param<float>(dst, 5), eps = param<float>(dst, 6);
    const int isv = param<int>(dst, 7);
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *ix = dst->src[3], *cc = dst->src[4];
    const ggml_tensor* dy = nhd > 0 ? dst->src[5] : nullptr;
    const ggml_tensor* hi = has_h ? dst->src[nhd > 0 ? 6 : 5] : nullptr;
    const int hd = (int) cc->ne[0], nkv = (int) cc->ne[1], T = (int) ix->ne[0];
    cudaStream_t s = (cudaStream_t) stream;
#define PREP_CASE(k) case k: k_attn_prep<k><<<nkv, hd, 0, s>>>((const float*) x->data, (long) (x->nb[1] / 4), (const float*) w->data, \
        (float*) st->data, off, dy ? (const float*) dy->data : nullptr, nhd, scale, (const int64_t*) ix->data, (char*) cc->data, \
        (long) cc->nb[1], (long) cc->nb[2], hi ? (float*) hi->data : nullptr, hi ? (long) (hi->nb[1] / 4) : 0, T, thr, eps, isv); break;
    switch (K) { PREP_CASE(2) PREP_CASE(3) PREP_CASE(4) PREP_CASE(5) PREP_CASE(6) PREP_CASE(7) PREP_CASE(8) default: break; }
#undef PREP_CASE
}

void attn_prep_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const int K = param<int>(dst, 0), nhd = param<int>(dst, 1), has_h = param<int>(dst, 2), off = param<int>(dst, 3);
    const float scale = param<float>(dst, 4), thr = param<float>(dst, 5), eps = param<float>(dst, 6);
    const int isv = param<int>(dst, 7);
    const ggml_tensor *x = dst->src[0], *w = dst->src[1], *st = dst->src[2], *ix = dst->src[3], *cc = dst->src[4];
    const ggml_tensor* dy = nhd > 0 ? dst->src[5] : nullptr;
    const ggml_tensor* hi = has_h ? dst->src[nhd > 0 ? 6 : 5] : nullptr;
    const int hd = (int) cc->ne[0], nkv = (int) cc->ne[1], T = (int) ix->ne[0], C = hd * nkv;
    for (int g = ith; g < nkv; g += nth) {
        std::vector<float> rr((size_t) hd * K), y(hd);
        float* sp = (float*) st->data;
        for (int tid = 0; tid < hd; ++tid) {
            const int srow = off + g * hd + tid;
            for (int j = 0; j < K - 1; ++j) rr[(size_t) tid * K + j] = sp[(size_t) srow * (K - 1) + j];
            if (hi) for (int j = 0; j < K - 1; ++j) ((float*) ((char*) hi->data + (size_t) srow * hi->nb[1]))[j] = rr[(size_t) tid * K + j];
        }
        for (int t = 0; t < T; ++t) {
            for (int tid = 0; tid < hd; ++tid) {
                const int ch = g * hd + tid, srow = off + ch;
                float* r = &rr[(size_t) tid * K];
                const float xval = *(const float*) ((const char*) x->data + (size_t) t * x->nb[1] + (size_t) ch * 4);
                r[K - 1] = xval;
                if (hi) ((float*) ((char*) hi->data + (size_t) srow * hi->nb[1]))[K - 1 + t] = xval;
                float sacc = 0.0f;
                for (int j = 0; j < K; ++j) sacc += r[j] * ((const float*) w->data)[(size_t) ch * K + j];
                if (dy) {
                    const int hh = ch / (C / nhd);
                    for (int lag = 0; lag < K; ++lag) sacc += r[K - 1 - lag] * ((const float*) dy->data)[(size_t) t * K * nhd + (size_t) hh * K + lag];
                }
                y[tid] = scale * (xval + sacc);
                for (int j = 0; j < K - 1; ++j) r[j] = r[j + 1];
            }
            float tmp = 0.0f;
            for (int tid = 0; tid < hd; ++tid) tmp += y[tid] * y[tid];
            const float nr = isv ? 1.0f / std::sqrt(tmp / hd + eps) : 1.0f / std::sqrt(std::max(tmp, thr * thr));
            const int64_t cell = ((const int64_t*) ix->data)[t];
            for (int tid = 0; tid < hd; ++tid) {
                const float o = isv ? nr * y[tid] : thr * (nr * y[tid]);
                ((ggml_fp16_t*) ((char*) cc->data + (size_t) cell * cc->nb[2] + (size_t) g * cc->nb[1]))[tid] = ggml_fp32_to_fp16(o);
            }
        }
        for (int tid = 0; tid < hd; ++tid) {
            const int srow = off + g * hd + tid;
            for (int j = 0; j < K - 1; ++j) sp[(size_t) srow * (K - 1) + j] = rr[(size_t) tid * K + j];
        }
    }
}

ggml_cuda_custom_op g_conv{GGML_CUDA_CUSTOM_MAGIC, conv_cuda, nullptr};
ggml_cuda_custom_op g_situ{GGML_CUDA_CUSTOM_MAGIC, situ_cuda, nullptr};
ggml_cuda_custom_op g_stack{GGML_CUDA_CUSTOM_MAGIC, stack_cuda, nullptr};
ggml_cuda_custom_op g_clip{GGML_CUDA_CUSTOM_MAGIC, clip_cuda, nullptr};
ggml_cuda_custom_op g_mix{GGML_CUDA_CUSTOM_MAGIC, mix_cuda, nullptr};
ggml_cuda_custom_op g_relbias{GGML_CUDA_CUSTOM_MAGIC, relbias_cuda, nullptr};
ggml_cuda_custom_op g_attn_prep{GGML_CUDA_CUSTOM_MAGIC, attn_prep_cuda, nullptr};

// ------------------------------------------------------------------------------------------------ prompt attention
// Port of Basalt's attn_prompt.cu (Jesus Gulfo, MIT; src/models/inkling2/kernels/attn_prompt.cu): a chunk's rows against
// every visible key, flash style on the fp16 tensor cores (mma.sync m16n8k16, fp32 accumulation, online softmax in base 2),
// the relative bias as one more product (R times the band of the bias table the tile's distances span), the sink beside
// the keys in the denominator.  A block = 64 positions x one kv head (its 3 query heads); keys stream through shared
// memory in 32-key tiles straight from our f16 cache (position p in cell p % cap: the global cache with cap = its cells,
// a sliding ring with cap = the ring).  Adapted: runtime nh/nkv/extent/window, our bias table layout [n_rel, Ext+1]
// (element (k, d) at d * n_rel + k), our sinks (bias - sink with a zero logit == a sink logit beside the keys), one cache
// (our rings hold window + ubatch, so the chunk's keys are already in place - no stage).  Needs hd 128, n_rel 16, 3 q
// heads per kv head, f16 K/V.
namespace pa {
constexpr int kHD = 128, kRel = 16, kGroup = 3;
constexpr int kPos = 64, kWarpsPerHead = kPos / 16, kWarps = kGroup * kWarpsPerHead, kThreads = kWarps * 32, kKeys = 32;
constexpr int kLd = kHD + 8, kBand = 96, kBandLd = kRel + 8, kGCols = 48, kGLd = 55;
constexpr float kLog2e = 1.4426950408889634f;
static_assert(kBand >= kPos + kKeys - 1 && kGCols >= 16 + kKeys - 1 && kGCols % 8 == 0 && kGLd >= kGCols, "band sizes");
struct KvTile { __half k[kKeys][kLd]; __half v[kKeys][kLd]; };
struct Smem {
    KvTile t0;
    __half band[kBand][kBandLd];
    union { KvTile t1; float gx[kWarps][16][kGLd]; } u;
};
static_assert(sizeof(KvTile) <= sizeof(float) * kWarps * 16 * kGLd, "the second tile fits in the exchange");
struct Args {
    const float *q, *r, *sink, *rel;   // q [T][nh][128], r [T][nh][16], sink [T][nh], rel [Ext+1][16]
    const __half *k, *v;               // [cap][nkv][128]
    long long pos0, cap;
    int nh, ext, window;
    float scale;
};
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) { const __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<const unsigned*>(&h); }
__device__ __forceinline__ float quad_max(float v) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1)); return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2)); }
__device__ __forceinline__ float quad_sum(float v) { v += __shfl_xor_sync(0xffffffffu, v, 1); return v + __shfl_xor_sync(0xffffffffu, v, 2); }

__device__ __forceinline__ void load_tile(const Args& a, KvTile& sm, int kvh, int nkv, long long k0, long long lo, long long hi, int tid) {
    constexpr int kPieces = kKeys * kHD / 8;
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem / (kHD / 8), col = (rem % (kHD / 8)) * 8;
        const long long key = k0 + j;
        const bool ok = key >= lo && key <= hi;
        const __half* src = (isv ? a.v : a.k) + ((size_t) (ok ? key % a.cap : 0) * nkv + kvh) * kHD + col;
        cp16(isv ? &sm.v[j][col] : &sm.k[j][col], src, ok);
    }
}

__global__ void __launch_bounds__(kThreads, 1) k_attn_prompt(const Args a, const int rows, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int kvh = blockIdx.y, nkv = gridDim.y, nh = a.nh;
    const int i0 = (int) (gridDim.x - 1 - blockIdx.x) * kPos;   // the blocks with the most keys first
    const long long q0 = a.pos0 + i0;
    const int head = kvh * kGroup + w / kWarpsPerHead;
    const int jw = (w % kWarpsPerHead) * 16;
    const long long pw = q0 + jw;
    const int ia = i0 + jw + g, ib = ia + 8;
    const bool va = ia < rows, vb = ib < rows;
    const long long hi = a.pos0 + min(rows, i0 + kPos) - 1;
    const long long lo = a.window > 0 ? max(0LL, q0 - a.window + 1) : 0;
    const int ext = a.ext;

    uint32_t qf[kHD / 16][4], rf[4];
    {
        const float* qa = a.q + ((size_t) (va ? ia : 0) * nh + head) * kHD + 2 * tq;
        const float* qb = a.q + ((size_t) (vb ? ib : 0) * nh + head) * kHD + 2 * tq;
#pragma unroll
        for (int kk = 0; kk < kHD / 16; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(qa + 16 * kk);
            const float2 x1 = *reinterpret_cast<const float2*>(qb + 16 * kk);
            const float2 x2 = *reinterpret_cast<const float2*>(qa + 16 * kk + 8);
            const float2 x3 = *reinterpret_cast<const float2*>(qb + 16 * kk + 8);
            qf[kk][0] = pack_h2(x0.x, x0.y); qf[kk][1] = pack_h2(x1.x, x1.y);
            qf[kk][2] = pack_h2(x2.x, x2.y); qf[kk][3] = pack_h2(x3.x, x3.y);
        }
        const float* ra = a.r + ((size_t) (va ? ia : 0) * nh + head) * kRel + 2 * tq;
        const float* rb = a.r + ((size_t) (vb ? ib : 0) * nh + head) * kRel + 2 * tq;
        rf[0] = pack_h2(ra[0], ra[1]); rf[1] = pack_h2(rb[0], rb[1]);
        rf[2] = pack_h2(ra[8], ra[9]); rf[3] = pack_h2(rb[8], rb[9]);
    }
    float o[kHD / 8][4];
#pragma unroll
    for (int n = 0; n < kHD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    const float scale2 = a.scale * kLog2e;
    float (*gx)[kGLd] = sm.u.gx[w];
    const auto far = [&](long long k0) { return q0 - k0 - (kKeys - 1) >= ext; };
    int cur = 0;
    bool ready = false;
    for (long long k0 = lo / kKeys * kKeys; k0 <= hi; k0 += kKeys) {
        __syncthreads();
        if (!ready) {
            cur = 0;
            load_tile(a, sm.t0, kvh, nkv, k0, lo, hi, tid);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        const bool next = far(k0) && k0 + kKeys <= hi && far(k0 + kKeys);
        if (next) {
            load_tile(a, cur ? sm.t0 : sm.u.t1, kvh, nkv, k0 + kKeys, lo, hi, tid);
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        const KvTile& kt = cur ? sm.u.t1 : sm.t0;
        ready = next;
        if (next) cur ^= 1;
        // the bias band: row b = the table's distance d = q0 - k0 - (kKeys - 1) + b, zero outside [0, ext)
        if (!far(k0))
            for (int idx = tid; idx < kBand * kRel; idx += kThreads) {
                const int b = idx / kRel, t = idx % kRel;
                const long long d = q0 - k0 - (kKeys - 1) + b;
                sm.band[b][t] = __float2half_rn(d >= 0 && d < ext ? a.rel[(size_t) d * kRel + t] : 0.f);
            }
        __syncthreads();
        const long long p_first = pw, p_last = pw + 15;
        if (k0 > p_last || (a.window > 0 && k0 + kKeys - 1 <= p_first - a.window)) continue;
        const bool whole = k0 + kKeys - 1 <= p_first && (a.window <= 0 || k0 > p_last - a.window);
        float s[kKeys / 8][4];
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < kHD / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < kKeys / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &kt.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                mma16816(s[n + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
            }
        }
        if (p_first - (k0 + kKeys - 1) < ext) {
            float ga[kGCols / 8][4];
#pragma unroll
            for (int nt = 0; nt < kGCols / 8; ++nt) {
                ga[nt][0] = ga[nt][1] = ga[nt][2] = ga[nt][3] = 0.f;
                unsigned b0, b1;
                ldsm_x2(b0, b1, &sm.band[jw + nt * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
                mma16816(ga[nt], rf[0], rf[1], rf[2], rf[3], b0, b1);
            }
#pragma unroll
            for (int nt = 0; nt < kGCols / 8; ++nt) {
                gx[g][nt * 8 + 2 * tq] = ga[nt][0];
                gx[g][nt * 8 + 2 * tq + 1] = ga[nt][1];
                gx[g + 8][nt * 8 + 2 * tq] = ga[nt][2];
                gx[g + 8][nt * 8 + 2 * tq + 1] = ga[nt][3];
            }
            __syncwarp();
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = g + (e >> 1) * 8, kk = n * 8 + 2 * tq + (e & 1);
                    s[n][e] = fmaf(s[n][e], scale2, gx[j][j - kk + kKeys - 1] * kLog2e);
                }
            __syncwarp();
        } else {
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[n][e] *= scale2;
        }
        if (!whole) {
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const long long key = k0 + n * 8 + 2 * tq + (e & 1), p = pw + g + (e >> 1) * 8;
                    if (key > p || (a.window > 0 && key <= p - a.window)) s[n][e] = -INFINITY;
                }
        }
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) { mx_a = fmaxf(mx_a, fmaxf(s[n][0], s[n][1])); mx_b = fmaxf(mx_b, fmaxf(s[n][2], s[n][3])); }
        mx_a = quad_max(mx_a); mx_b = quad_max(mx_b);
        const float ca = mx_a == -INFINITY ? 1.f : exp2f(m_a - mx_a), cb = mx_b == -INFINITY ? 1.f : exp2f(m_b - mx_b);
        m_a = mx_a; m_b = mx_b;
        float ps_a = 0.f, ps_b = 0.f;
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) {
            s[n][0] = s[n][0] == -INFINITY ? 0.f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] == -INFINITY ? 0.f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] == -INFINITY ? 0.f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] == -INFINITY ? 0.f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1]; ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * ca + ps_a; l_b = l_b * cb + ps_b;
#pragma unroll
        for (int n = 0; n < kHD / 8; ++n) { o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb; }
        unsigned pf[kKeys / 16][4];
#pragma unroll
        for (int j = 0; j < kKeys / 16; ++j) {
            pf[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pf[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pf[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pf[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < kHD / 8; n += 2)
#pragma unroll
            for (int j = 0; j < kKeys / 16; ++j) {
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, &kt.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816(o[n], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                mma16816(o[n + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
            }
    }
    l_a = quad_sum(l_a); l_b = quad_sum(l_b);
    const float sa = va ? a.sink[(size_t) ia * nh + head] * kLog2e : 0.f;
    const float sb = vb ? a.sink[(size_t) ib * nh + head] * kLog2e : 0.f;
    const float ma = fmaxf(m_a, sa), mb = fmaxf(m_b, sb);
    const float fa = exp2f(m_a - ma), fb = exp2f(m_b - mb);
    const float inv_a = fa / (l_a * fa + exp2f(sa - ma)), inv_b = fb / (l_b * fb + exp2f(sb - mb));
#pragma unroll
    for (int n = 0; n < kHD / 8; ++n) {
        const int col = n * 8 + 2 * tq;
        if (va) *reinterpret_cast<float2*>(out + ((size_t) ia * nh + head) * kHD + col) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        if (vb) *reinterpret_cast<float2*>(out + ((size_t) ib * nh + head) * kHD + col) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
    }
}
}  // namespace pa

// srcs: q [128, nh, T], k/v caches [128, nkv, cap] f16, R [16, nh, T], sinks [nh, T], relb [16, Ext+1], + the two cache
// stores (ordering only).  params: 0 pos0, 1 cap, 2 window, 3 ext, 4 scale
void attn_prompt_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *R = dst->src[3], *sk = dst->src[4], *rb = dst->src[5];
    pa::Args a;
    a.q = (const float*) q->data; a.r = (const float*) R->data; a.sink = (const float*) sk->data; a.rel = (const float*) rb->data;
    a.k = (const __half*) k->data; a.v = (const __half*) v->data;
    a.pos0 = param<int>(dst, 0); a.cap = param<int>(dst, 1); a.window = param<int>(dst, 2); a.ext = param<int>(dst, 3);
    a.scale = param<float>(dst, 4); a.nh = (int) q->ne[1];
    const int rows = (int) q->ne[2], nkv = (int) k->ne[1];
    static const bool sized = [] { cudaFuncSetAttribute(pa::k_attn_prompt, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(pa::Smem)); return true; }();
    (void) sized;
    const dim3 grid((unsigned) ((rows + pa::kPos - 1) / pa::kPos), (unsigned) nkv);
    pa::k_attn_prompt<<<grid, pa::kThreads, sizeof(pa::Smem), (cudaStream_t) stream>>>(a, rows, (float*) dst->data);
}

// CPU reference: the same attention in plain fp32 loops
void attn_prompt_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *R = dst->src[3], *sk = dst->src[4], *rb = dst->src[5];
    const int nh = (int) q->ne[1], T = (int) q->ne[2], nkv = (int) k->ne[1], G = nh / nkv;
    const long long pos0 = param<int>(dst, 0), cap = param<int>(dst, 1);
    const int window = param<int>(dst, 2), ext = param<int>(dst, 3);
    const float scale = param<float>(dst, 4);
    std::vector<float> sc;
    for (int u = ith; u < T * nh; u += nth) {
        const int t = u / nh, h = u % nh, g = h / G;
        const long long p = pos0 + t, lo = window > 0 ? std::max(0LL, p - window + 1) : 0;
        const float* qh = (const float*) q->data + ((size_t) t * nh + h) * 128;
        const float* rh = (const float*) R->data + ((size_t) t * nh + h) * 16;
        const float sink = ((const float*) sk->data)[(size_t) t * nh + h];
        sc.assign((size_t) (p - lo + 1), 0.0f);
        float mx = sink;
        for (long long key = lo; key <= p; ++key) {
            const ggml_fp16_t* kr = (const ggml_fp16_t*) k->data + ((size_t) (key % cap) * nkv + g) * 128;
            float d = 0.0f;
            for (int i = 0; i < 128; ++i) d += qh[i] * ggml_fp16_to_fp32(kr[i]);
            float b = 0.0f;
            const long long dist = p - key;
            if (dist < ext) for (int i = 0; i < 16; ++i) b += rh[i] * ((const float*) rb->data)[(size_t) dist * 16 + i];
            sc[(size_t) (key - lo)] = d * scale + b;
            mx = std::max(mx, sc[(size_t) (key - lo)]);
        }
        float den = std::exp(sink - mx);
        float* o = (float*) dst->data + ((size_t) t * nh + h) * 128;
        for (int i = 0; i < 128; ++i) o[i] = 0.0f;
        for (long long key = lo; key <= p; ++key) {
            const float e = std::exp(sc[(size_t) (key - lo)] - mx);
            den += e;
            const ggml_fp16_t* vr = (const ggml_fp16_t*) v->data + ((size_t) (key % cap) * nkv + g) * 128;
            for (int i = 0; i < 128; ++i) o[i] += e * ggml_fp16_to_fp32(vr[i]);
        }
        for (int i = 0; i < 128; ++i) o[i] /= den;
    }
}
ggml_cuda_custom_op g_attn_prompt{GGML_CUDA_CUSTOM_MAGIC, attn_prompt_cuda, nullptr};


// ------------------------------------------------------------------------------------------------ MoE weighted sum
// prompt chunks: x[c, t] = sum_i d[c, i, t] * w[i, t] over the k experts (the decode finish's order), replacing the
// per-token batched f32 mat-vec over a transposed copy of d (~0.5 ms a visit at T = 512, now bandwidth-bound)
__global__ void k_moe_wsum(const float* __restrict__ d, const float* __restrict__ w, float* __restrict__ y, int C, int k, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    float x = 0.0f;
    for (int i = 0; i < k; ++i) x = fmaf(d[((size_t) t * k + i) * C + c], w[(size_t) t * k + i], x);
    y[(size_t) t * C + c] = x;
}
void moe_wsum_cuda(ggml_tensor* dst, void* stream, void*) {
    const ggml_tensor *d = dst->src[0], *w = dst->src[1];
    const int C = (int) d->ne[0], k = (int) d->ne[1], T = (int) d->ne[2];
    k_moe_wsum<<<dim3((C + 255) / 256, T), 256, 0, (cudaStream_t) stream>>>((const float*) d->data, (const float*) w->data, (float*) dst->data, C, k, T);
}
void moe_wsum_cpu(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor *d = dst->src[0], *w = dst->src[1];
    const int C = (int) d->ne[0], k = (int) d->ne[1], T = (int) d->ne[2];
    const float *dp = (const float*) d->data, *wp = (const float*) w->data;
    for (int t = ith; t < T; t += nth)
        for (int c = 0; c < C; ++c) {
            float x = 0.0f;
            for (int i = 0; i < k; ++i) x = std::fma(dp[((size_t) t * k + i) * C + c], wp[(size_t) t * k + i], x);
            ((float*) dst->data)[(size_t) t * C + c] = x;
        }
}
ggml_cuda_custom_op g_moe_wsum{GGML_CUDA_CUSTOM_MAGIC, moe_wsum_cuda, nullptr};

ggml_cuda_custom_op g_attn_dec_part{GGML_CUDA_CUSTOM_MAGIC, attn_dec_part_cuda, nullptr};
ggml_cuda_custom_op g_attn_dec_reduce{GGML_CUDA_CUSTOM_MAGIC, attn_dec_reduce_cuda, nullptr};

}  // namespace

ggml_tensor* inkling_conv_step(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* state, ggml_tensor* dyn,
                               int K, int nh, float scale, ggml_tensor* hist) {
    ggml_tensor* args[5] = {x, w, state, nullptr, nullptr};
    int n = 3;
    if (dyn) args[n++] = dyn;
    if (hist) args[n++] = hist;
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, args, n, conv_cpu,
                                    GGML_N_TASKS_MAX, &g_conv);
    set_param<int>(y, 0, K);
    set_param<int>(y, 1, dyn ? nh : 0);
    set_param<float>(y, 2, scale);
    set_param<int>(y, 3, hist ? 1 : 0);
    set_param<int>(y, 4, 0);
    set_param<int>(y, 5, 0);
    return y;
}

ggml_tensor* inkling_moe_finish(ggml_context* c, ggml_tensor* d, ggml_tensor* wk, ggml_tensor* w, ggml_tensor* state,
                                ggml_tensor* res, int K, float scale, ggml_tensor* hist) {
    const int64_t C = d->ne[0], k = d->ne[1], T = d->ne[2];
    if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || wk->type != GGML_TYPE_F32 || !ggml_is_contiguous(wk) ||
        ggml_nelements(wk) != k * T || res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != C * T ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || K > 8) return nullptr;
    ggml_tensor* args[6] = {d, w, state, nullptr, nullptr, nullptr};
    int n = 3;
    if (hist) args[n++] = hist;
    args[n++] = wk;
    args[n++] = res;
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, C, T, 1, 1, args, n, conv_cpu, GGML_N_TASKS_MAX, &g_conv);
    set_param<int>(y, 0, K);
    set_param<int>(y, 1, 0);
    set_param<float>(y, 2, scale);
    set_param<int>(y, 3, hist ? 1 : 0);
    set_param<int>(y, 4, (int) k);
    set_param<int>(y, 5, 1);
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

ggml_tensor* inkling_clip(ggml_context* c, ggml_tensor* x, ggml_tensor* scale, float thr, ggml_tensor* tau) {
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] >= 1024 || thr <= 0.0f) return nullptr;
    if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || scale->ne[0] != x->ne[0] ||
                  ggml_nelements(scale) != x->ne[0] * x->ne[1])) return nullptr;
    if (tau && (tau->type != GGML_TYPE_F32 || !ggml_is_contiguous(tau) || ggml_nelements(tau) != x->ne[2] || x->ne[3] != 1)) return nullptr;
    ggml_tensor* args[3] = {x, scale ? scale : x, tau ? tau : x};
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 3, clip_cpu,
                                    GGML_N_TASKS_MAX, &g_clip);
    set_param<float>(y, 0, thr);
    set_param<int>(y, 1, scale ? 1 : 0);
    set_param<int>(y, 2, tau ? 1 : 0);
    return y;
}

ggml_tensor* inkling_mix(ggml_context* c, ggml_tensor* const* srcs, int n, ggml_tensor* query) {
    if (n < 1 || n > 6 || query->type != GGML_TYPE_F32 || !ggml_is_contiguous(query)) return nullptr;
    const int64_t E = srcs[0]->ne[0], T = ggml_nelements(srcs[0]) / E, nq = ggml_nelements(query) / E;
    if (query->ne[0] != E || nq < 1 || nq > kMixQ) return nullptr;
    for (int i = 0; i < n; ++i)
        if (srcs[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(srcs[i]) || srcs[i]->ne[0] != E || ggml_nelements(srcs[i]) != E * T)
            return nullptr;
    ggml_tensor* args[GGML_MAX_SRC];
    for (int i = 0; i < n; ++i) args[i] = srcs[i];
    args[n] = query;
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, E, nq, T, 1, args, n + 1, mix_cpu, GGML_N_TASKS_MAX, &g_mix);
    set_param<int>(y, 0, n);
    return y;
}

ggml_tensor* inkling_relbias(ggml_context* c, ggml_tensor* table, ggml_tensor* sinks, ggml_tensor* rp, ggml_tensor* mask,
                             ggml_type out_type) {
    if (table->type != GGML_TYPE_F32 || !ggml_is_contiguous(table) || sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) ||
        rp->type != GGML_TYPE_I32 || mask->type != GGML_TYPE_F32 || rp->nb[0] != sizeof(int32_t) || mask->nb[0] != sizeof(float) ||
        (out_type != GGML_TYPE_F32 && out_type != GGML_TYPE_F16)) return nullptr;
    const int64_t nh = table->ne[1], T = table->ne[2], cells = rp->ne[0];
    if (ggml_nelements(sinks) != nh * T || rp->ne[1] != T || mask->ne[0] != cells || mask->ne[1] != T) return nullptr;
    ggml_tensor* args[4] = {table, sinks, rp, mask};
    return ggml_custom_4d(c, out_type, cells, T, nh, 1, args, 4, relbias_cpu, GGML_N_TASKS_MAX, &g_relbias);
}

static ggml_tensor* attn_prep_one(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* state, int off, ggml_tensor* dyn,
                                  int K, float scale, ggml_tensor* idx, ggml_tensor* cache, float thr, float eps, ggml_tensor* hist, int isv,
                                  ggml_tensor* dep = nullptr) {
    const int64_t hd = cache->ne[0], nkv = cache->ne[1], C = hd * nkv, T = idx->ne[0];
    if (cache->type != GGML_TYPE_F16 || hd > 256 || hd % 32 != 0 || K < 2 || K > 8 || idx->type != GGML_TYPE_I64 ||
        !ggml_is_contiguous(idx) || x->type != GGML_TYPE_F32 || x->nb[0] != 4 || x->ne[0] != C || x->ne[1] != T ||
        x->nb[1] % 4 != 0 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != C * K ||
        state->type != GGML_TYPE_F32 || !ggml_is_contiguous(state)) return nullptr;
    int nhd = 0;
    if (dyn) {
        if (!ggml_is_contiguous(dyn) || dyn->type != GGML_TYPE_F32 || dyn->ne[0] % K != 0) return nullptr;
        nhd = (int) (dyn->ne[0] / K);
        if (C % nhd != 0) return nullptr;
    }
    ggml_tensor* args[8] = {x, w, state, idx, cache, nullptr, nullptr, nullptr};
    int n = 5;
    if (dyn) args[n++] = dyn;
    if (hist) args[n++] = hist;
    if (dep) args[n++] = dep;   // ordering only (the K op before the V op): the kernel never reads it
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, 1, 1, 1, 1, args, n, attn_prep_cpu, GGML_N_TASKS_MAX, &g_attn_prep);
    set_param<int>(y, 0, K);
    set_param<int>(y, 1, nhd);
    set_param<int>(y, 2, hist ? 1 : 0);
    set_param<int>(y, 3, off);
    set_param<float>(y, 4, scale);
    set_param<float>(y, 5, thr);
    set_param<float>(y, 6, eps);
    set_param<int>(y, 7, isv);
    return y;
}
ggml_tensor* inkling_attn_prep(ggml_context* c, ggml_tensor* xk, ggml_tensor* xv, ggml_tensor* wk, ggml_tensor* wv,
                               ggml_tensor* state, int off_v, ggml_tensor* dynk, ggml_tensor* dynv, int K, float scale,
                               ggml_tensor* idx, ggml_tensor* kcache, ggml_tensor* vcache, float thr, float eps, ggml_tensor* hist) {
    if ((dynk == nullptr) != (dynv == nullptr)) return nullptr;
    ggml_tensor* a = attn_prep_one(c, xk, wk, state, 0, dynk, K, scale, idx, kcache, thr, eps, hist, 0);
    if (!a) return nullptr;
    // the V op lists the K op as a source, so expanding the V op (the caller does, before the attention) runs both
    return attn_prep_one(c, xv, wv, state, off_v, dynv, K, scale, idx, vcache, thr, eps, hist, 1, a);
}
ggml_tensor* inkling_moe_wsum(ggml_context* c, ggml_tensor* d, ggml_tensor* wk) {
    const int64_t C = d->ne[0], k = d->ne[1], T = d->ne[2];
    if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || wk->type != GGML_TYPE_F32 || !ggml_is_contiguous(wk) ||
        ggml_nelements(wk) != k * T) return nullptr;
    ggml_tensor* args[2] = {d, wk};
    return ggml_custom_4d(c, GGML_TYPE_F32, C, T, 1, 1, args, 2, moe_wsum_cpu, GGML_N_TASKS_MAX, &g_moe_wsum);
}

ggml_tensor* inkling_attn_prompt(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* R,
                                 ggml_tensor* sinks, ggml_tensor* relb, ggml_tensor* kst, ggml_tensor* vst, int64_t pos0,
                                 int window, float scale) {
    const int64_t nh = q->ne[1], T = q->ne[2], nkv = k->ne[1], cap = k->ne[2];
    if (q->type != GGML_TYPE_F32 || !ggml_is_contiguous(q) || q->ne[0] != 128 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(k) || !ggml_is_contiguous(v) || k->ne[0] != 128 || v->ne[0] != 128 || v->ne[1] != nkv || v->ne[2] != cap ||
        nkv < 1 || nh != 3 * nkv || R->type != GGML_TYPE_F32 || !ggml_is_contiguous(R) || R->ne[0] != 16 || ggml_nelements(R) != 16 * nh * T ||
        sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || ggml_nelements(sinks) != nh * T ||
        relb->type != GGML_TYPE_F32 || !ggml_is_contiguous(relb) || relb->ne[0] != 16 || relb->ne[1] < 2 ||
        pos0 + T > INT32_MAX || cap > INT32_MAX) return nullptr;
    ggml_tensor* a[8] = {q, k, v, R, sinks, relb, kst, vst};
    const int na = kst && vst ? 8 : 6;
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, 128, nh, T, 1, a, na, attn_prompt_cpu, GGML_N_TASKS_MAX, &g_attn_prompt);
    set_param<int>(y, 0, (int) pos0);
    set_param<int>(y, 1, (int) cap);
    set_param<int>(y, 2, window);
    set_param<int>(y, 3, (int) relb->ne[1] - 1);   // Ext (the pad row is the zero row)
    set_param<float>(y, 4, scale);
    return y;
}

ggml_tensor* inkling_attn_dec_rel(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* relb,
                                  ggml_tensor* Rt, ggml_tensor* sinks, ggml_tensor* rp, ggml_tensor* mask, float scale,
                                  ggml_tensor* gate) {
    const int64_t HD = q->ne[0], nh = q->ne[1], T = q->ne[2], nkv = k->ne[1], cells = k->ne[2];
    if (q->type != GGML_TYPE_F32 || !ggml_is_contiguous(q) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return nullptr;
    if ((HD != 64 && HD != 128 && HD != 256) || k->ne[0] != HD || v->ne[0] != HD || v->ne[1] != nkv || v->ne[2] != cells ||
        k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t) || nkv < 1 || nh % nkv != 0 || nh / nkv > 4 ||
        T > 64 || cells < 1 || k->nb[1] % 4 != 0 || k->nb[2] % 4 != 0 || v->nb[1] % 4 != 0 || v->nb[2] % 4 != 0) return nullptr;
    if (relb->type != GGML_TYPE_F32 || !ggml_is_contiguous(relb) || relb->ne[0] > 32 || Rt->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(Rt) || Rt->ne[0] != relb->ne[0] || ggml_nelements(Rt) != relb->ne[0] * nh * T ||
        sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || ggml_nelements(sinks) != nh * T ||
        rp->type != GGML_TYPE_I32 || rp->nb[0] != sizeof(int32_t) || rp->ne[0] != cells || rp->ne[1] != T ||
        mask->type != GGML_TYPE_F32 || mask->nb[0] != sizeof(float) || mask->ne[0] != cells || mask->ne[1] != T) return nullptr;
    if (gate && (gate->type != GGML_TYPE_F32 || gate->nb[0] != sizeof(float) || gate->ne[0] != HD * nh || gate->ne[1] != T)) return nullptr;
    const int64_t nsplit = (cells + kDecChunk - 1) / kDecChunk;
    ggml_tensor* a1[8] = {q, k, v, mask, relb, Rt, sinks, rp};
    ggml_tensor* part = ggml_custom_4d(c, GGML_TYPE_F32, HD + 2, nh, T, nsplit, a1, 8, attn_dec_part_cpu, GGML_N_TASKS_MAX, &g_attn_dec_part);
    set_param<float>(part, 0, scale);
    set_param<int>(part, 1, 0);
    set_param<int>(part, 2, 1);
    ggml_tensor* a2[2] = {part, gate};
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, HD, nh, T, 1, a2, gate ? 2 : 1, attn_dec_reduce_cpu, GGML_N_TASKS_MAX, &g_attn_dec_reduce);
    set_param<int>(y, 0, gate ? 1 : 0);
    return y;
}
ggml_tensor* inkling_attn_dec(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale,
                              ggml_tensor* gate) {
    const int64_t HD = q->ne[0], nh = q->ne[1], T = q->ne[2], nkv = k->ne[1], cells = k->ne[2];
    if (q->type != GGML_TYPE_F32 || !ggml_is_contiguous(q) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F32 || !ggml_is_contiguous(mask)) return nullptr;
    if ((HD != 64 && HD != 128 && HD != 256) || k->ne[0] != HD || v->ne[0] != HD || v->ne[1] != nkv || v->ne[2] != cells ||
        k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t) || nkv < 1 || nh % nkv != 0 || nh / nkv > 4 ||
        mask->ne[0] != cells || mask->ne[1] != T || mask->ne[2] != nh || T > 64 || cells < 1) return nullptr;
    if (k->nb[1] % 4 != 0 || k->nb[2] % 4 != 0 || v->nb[1] % 4 != 0 || v->nb[2] % 4 != 0) return nullptr;   // half2 loads
    const int64_t nsplit = (cells + kDecChunk - 1) / kDecChunk;
    ggml_tensor* a1[4] = {q, k, v, mask};
    ggml_tensor* part = ggml_custom_4d(c, GGML_TYPE_F32, HD + 2, nh, T, nsplit, a1, 4, attn_dec_part_cpu, GGML_N_TASKS_MAX, &g_attn_dec_part);
    set_param<float>(part, 0, scale);
    static const int f16r = std::getenv("XOLOTL_INKLING_DECATTN_F16R") ? 1 : 0;   // A/B: round q and p to f16 like the chain
    set_param<int>(part, 1, f16r);
    set_param<int>(part, 2, 0);
    if (gate && (gate->type != GGML_TYPE_F32 || gate->nb[0] != sizeof(float) || gate->ne[0] != HD * nh || gate->ne[1] != T)) return nullptr;
    ggml_tensor* a2[2] = {part, gate};
    ggml_tensor* y = ggml_custom_4d(c, GGML_TYPE_F32, HD, nh, T, 1, a2, gate ? 2 : 1, attn_dec_reduce_cpu, GGML_N_TASKS_MAX, &g_attn_dec_reduce);
    set_param<int>(y, 0, gate ? 1 : 0);
    return y;
}
bool inkling_fused_available() { return true; }

#else   // a ggml without the hook: the custom ops would run on the CPU, so the engine keeps its op chains

ggml_tensor* inkling_conv_step(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, int, int, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_moe_finish(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, int, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_situ(ggml_context*, ggml_tensor*, float, float) { return nullptr; }
ggml_tensor* inkling_stack(ggml_context*, ggml_tensor* const*, int) { return nullptr; }
ggml_tensor* inkling_clip(ggml_context*, ggml_tensor*, ggml_tensor*, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_mix(ggml_context*, ggml_tensor* const*, int, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_relbias(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_type) { return nullptr; }
ggml_tensor* inkling_attn_dec(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_attn_dec_rel(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_attn_prep(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, int, ggml_tensor*, ggml_tensor*, int, float, ggml_tensor*, ggml_tensor*, ggml_tensor*, float, float, ggml_tensor*) { return nullptr; }
ggml_tensor* inkling_attn_prompt(ggml_context*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, int64_t, int, float) { return nullptr; }
ggml_tensor* inkling_moe_wsum(ggml_context*, ggml_tensor*, ggml_tensor*) { return nullptr; }
bool inkling_fused_available() { return false; }

#endif
