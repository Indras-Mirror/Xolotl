// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / GENI / HAVE / RESUME / PP / T / DONE / STOP; the held
// state is reused at the longest common start, from a GDN state snapshot when the request leaves it; GENI's pictures
// take M-RoPE positions), --tokens IDS [--n N] [--force IDS] [--embeddings F] [--dump-full-logits F]
// (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "strata/qwen35/flash.hpp"
#include "q36_sampling.hpp"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "q36_cache_hub.hpp"
#include "q36_resident_prefix.hpp"
#include "q36_resident_verify.hpp"
#ifndef __HIP_PLATFORM_AMD__
#include "q36_kv_vmm_cuda.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <future>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double wall_ms() {   // the epoch in ms (log lines a client compares with its own clock)
    return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
}
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// M-RoPE (ggml's IMROPE, rope.dimension_sections): which position stream (0 t, 1 h, 2 w, 3 e) rotary pair i takes.
/// Set once at load from the GGUF's sections; with t = h = w (text) every stream is the cell's own position.
__constant__ int8_t c_mrope_stream[64];

/// The attention layer's per-token preparation for T rows at cells *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the FP16 cache, the gate
/// kept.  `mrope` (nullable): the per-cell rotary positions [cells][3] (t, h, w) - an image's cells share t and differ
/// in h / w, the text after it continues past them; null = the cell index.  grid (H + HKV, T), block 256 (HD <= 256).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, __half* kc, __half* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* mrope) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mrope ? mrope[(size_t) pos * 3 + min(2, (int) c_mrope_stream[p])] : pos;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        kc[((size_t) kh * ctx + pos) * HD + i] = __float2half(v);
        vc[((size_t) kh * ctx + pos) * HD + i] = __float2half(vx[((size_t) t * HKV + kh) * HD + i]);
    }
}

/// Split-K decode attention over the FP16 cache, a KV head's G query heads per block (the cells read once for all of
/// them): block (chunk c of 256 cells, query t0 + y / HKV, KV head y % HKV), 8 warps.  Scores: warp w takes cells
/// w, w + 8, ..; its 32 lanes each 8 dims of a cell (coalesced 512-byte rows), the G dot products reduced across the
/// warp.  Then per head the chunk's max / sum, and out[d] = sum_p e_p V[p][d] with thread d.  Writes the same partials
/// as before (row t * H + h): the merge is unchanged.  G <= 16, HD == 256.
template <int G>
__global__ void __launch_bounds__(256) k_attn_part_gqa(const float* q, const __half* kc, const __half* vc,
                                                       const int32_t* pos0, int t0, int H, int HKV, int ctx, float scale,
                                                       int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256;
    __shared__ __align__(16) float qs[G][HD];
    __shared__ float sc[G][kChunkCells];
    __shared__ float smx[G], ssum[G];
    const int c = blockIdx.x, t = t0 + blockIdx.y / HKV, kvh = blockIdx.y % HKV;
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int n = min(kChunkCells, npos - p0);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    for (int i = tid; i < G * HD; i += 256) qs[i / HD][i % HD] = q[((size_t) t * H + kvh * G + i / HD) * HD + i % HD];
    __syncthreads();
    const __half* kbase = kc + ((size_t) kvh * ctx + p0) * HD;
    for (int k = warp; k < kChunkCells; k += 8) {
        float a[G];
#pragma unroll
        for (int g = 0; g < G; ++g) a[g] = 0.f;
        if (k < n) {
            const uint4 raw = *(const uint4*) (kbase + (size_t) k * HD + lane * 8);
            const __half2* h2 = (const __half2*) &raw;
            float kv[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h2[j]);
                kv[2 * j] = f.x;
                kv[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float4 q0 = *(const float4*) &qs[g][lane * 8], q1 = *(const float4*) &qs[g][lane * 8 + 4];
                a[g] = q0.x * kv[0] + q0.y * kv[1] + q0.z * kv[2] + q0.w * kv[3] + q1.x * kv[4] + q1.y * kv[5] +
                       q1.z * kv[6] + q1.w * kv[7];
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g)
            for (int m = 16; m > 0; m >>= 1) a[g] += __shfl_xor_sync(0xffffffffu, a[g], m);
        if (lane < G) {
            float v = 0.f;
#pragma unroll
            for (int g = 0; g < G; ++g)
                if (g == lane) v = a[g];
            sc[lane][k] = k < n ? v * scale : -INFINITY;
        }
    }
    __syncthreads();
    for (int g = warp; g < G; g += 8) {   // per head: the chunk's max, the exponentials, their sum
        float mx = -INFINITY;
        for (int k = lane; k < kChunkCells; k += 32) mx = fmaxf(mx, sc[g][k]);
        for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
        float s = 0.f;
        for (int k = lane; k < kChunkCells; k += 32) {
            const float e = k < n ? expf(sc[g][k] - mx) : 0.f;
            sc[g][k] = e;
            s += e;
        }
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) {
            smx[g] = mx;
            ssum[g] = s;
        }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    const __half* vp = vc + ((size_t) kvh * ctx + p0) * HD + tid;
    for (int k = 0; k < n; ++k) {
        const float v = __half2float(vp[(size_t) k * HD]);
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = fmaf(sc[g][k], v, acc[g]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const size_t o = (size_t) ((blockIdx.y / HKV) * H + kvh * G + g) * n_chunk + c;   // rows from t0, as the merge
        pacc[o * HD + tid] = acc[g];
        if (tid == 0) {
            pm[o] = smx[g];
            pl[o] = ssum[g];
        }
    }
}

/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

// ---- the sampled path's candidates: each logits row's kTopK largest (value, id), in descending order (ties: the lower
// id first), found on the device so that a sampled token (and a speculative window's rows) costs kTopK values over
// PCIe instead of the vocabulary.  Stage 1: kTopBlocks blocks per row, each sorts its slice (<= 4096 cells) in
// shared memory (bitonic); stage 2: one block per row sorts the kTopBlocks x kTopK candidates.
constexpr int kTopK = 128, kTopBlocks = 64, kTopSeg = 4096;
__device__ __forceinline__ bool top_before(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }
template <int N>
__device__ void bitonic_desc(float* v, int* id) {
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int l = i ^ j;
                if (l > i) {
                    const bool up = (i & k) == 0;   // this pair's run sorts descending when `up`
                    const bool sw = up ? top_before(v[l], id[l], v[i], id[i]) : top_before(v[i], id[i], v[l], id[l]);
                    if (sw) {
                        const float tv = v[i]; v[i] = v[l]; v[l] = tv;
                        const int ti = id[i]; id[i] = id[l]; id[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
}
__global__ void __launch_bounds__(512) k_topk1(const float* lg, int V, float* pv, int* pi) {
    __shared__ float v[kTopSeg];
    __shared__ int id[kTopSeg];
    const int row = blockIdx.y, b = blockIdx.x;
    const int seg = (V + kTopBlocks - 1) / kTopBlocks, a0 = b * seg;
    const float* x = lg + (size_t) row * V;
    for (int i = threadIdx.x; i < kTopSeg; i += blockDim.x) {
        const int c = a0 + i;
        const bool ok = i < seg && c < V;
        v[i] = ok ? x[c] : -INFINITY;
        id[i] = ok ? c : 0x7fffffff;
    }
    __syncthreads();
    bitonic_desc<kTopSeg>(v, id);
    for (int i = threadIdx.x; i < kTopK; i += blockDim.x) {
        pv[((size_t) row * kTopBlocks + b) * kTopK + i] = v[i];
        pi[((size_t) row * kTopBlocks + b) * kTopK + i] = id[i];
    }
}
__global__ void __launch_bounds__(1024) k_topk2(const float* pv, const int* pi, float* ov, int* oi) {
    extern __shared__ float sm[];
    constexpr int M = kTopBlocks * kTopK;
    float* v = sm;
    int* id = (int*) (sm + M);
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        v[i] = pv[(size_t) row * M + i];
        id[i] = pi[(size_t) row * M + i];
    }
    __syncthreads();
    bitonic_desc<M>(v, id);
    for (int i = threadIdx.x; i < kTopK; i += blockDim.x) {
        ov[(size_t) row * kTopK + i] = v[i];
        oi[(size_t) row * kTopK + i] = id[i];
    }
}

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write) {
    __shared__ float part[4];
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write) return;   // a verify window: the history is commit()'s to move
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write) {
    constexpr int S = 128, RG = 4, RPG = 32;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
        __syncthreads();
        if (rg == 0) {
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) t * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    bool has_mrope = false;                                   // rope.dimension_sections given (the vision path)
    int tok_im_start = -1, tok_im_end = -1, tok_image_pad = -1, tok_video_pad = -1;   // from the GGUF vocabulary (-1: none)
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

/// The ids of `names` in a GGUF's tokenizer.ggml.tokens (read straight from the file's header; absent ones left out).
std::map<std::string, int> vocab_ids(const std::string& path, const std::vector<std::string>& names) {
    std::map<std::string, int> out;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    auto u32 = [&]() { uint32_t v = 0; rd(&v, 4); return v; };
    auto u64 = [&]() { uint64_t v = 0; rd(&v, 8); return v; };
    auto str = [&]() { const uint64_t n = u64(); std::string s(n < (1u << 20) ? n : 0, '\0'); if (n < (1u << 20)) rd(s.data(), n); else std::fseek(f, (long) n, SEEK_CUR); return s; };
    auto scalar = [](uint32_t t) -> int { switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2;
                                                    case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; } };
    std::function<bool(uint32_t)> skip = [&](uint32_t t) -> bool {
        if (t == 8) { str(); return true; }
        if (t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            if (scalar(et)) return std::fseek(f, (long) (n * (uint64_t) scalar(et)), SEEK_CUR) == 0;
            for (uint64_t i = 0; i < n; ++i) if (!skip(et)) return false;
            return true;
        }
        const int b = scalar(t);
        return b > 0 && std::fseek(f, b, SEEK_CUR) == 0;
    };
    if (u32() != 0x46554747u) { std::fclose(f); return out; }   // 'GGUF'
    (void) u32();
    (void) u64();
    const uint64_t n_kv = u64();
    for (uint64_t k = 0; k < n_kv && !std::feof(f); ++k) {
        const std::string key = str();
        const uint32_t t = u32();
        if (key == "tokenizer.ggml.tokens" && t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            for (uint64_t i = 0; i < n && et == 8; ++i) {
                const std::string tok = str();
                for (const std::string& want : names)
                    if (tok == want) out[want] = (int) i;
            }
            break;
        }
        if (!skip(t)) break;
    }
    std::fclose(f);
    return out;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    const int64_t gqa = g.n_head_kv > 0 ? g.n_head / g.n_head_kv : 0;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15 ||
        g.n_head_kv < 1 || g.n_head % g.n_head_kv != 0 || (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 8 && gqa != 16)) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15, "
              "1 / 2 / 4 / 8 / 16 query heads per KV head)";
        return false;
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    {   // M-RoPE: ggml's IMROPE pair -> stream map from rope.dimension_sections (absent: every pair the t stream)
        int sec[4] = {0, 0, 0, 0};
        if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_sections"))
            for (size_t i = 0; i < 4 && i < v->items.size(); ++i) sec[i] = (int) v->items[i].u;
        const int dims = sec[0] + sec[1] + sec[2] + sec[3];
        int8_t map[64] = {};
        for (int pr = 0; pr < 64; ++pr) {
            const int sector = dims > 0 ? pr % dims : 0;
            map[pr] = dims == 0 ? 0 : (sector % 3 == 1 && sector < 3 * sec[1]) ? 1 : (sector % 3 == 2 && sector < 3 * sec[2]) ? 2
                    : (sector % 3 == 0 && sector < 3 * sec[0]) ? 0 : 3;
        }
        m.has_mrope = dims > 0;
        CK(cudaMemcpyToSymbol(c_mrope_stream, map, sizeof map));
    }
    if (m.n_rot > 128) {
        err = "rope.dimension_count above 128";
        return false;
    }
    {   // the chat / vision markers' ids, from the file's vocabulary (the reader keeps only a sample of the list)
        const std::map<std::string, int> ids = vocab_ids(path, {"<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|video_pad|>"});
        auto get = [&](const char* k) { auto it = ids.find(k); return it == ids.end() ? -1 : it->second; };
        m.tok_im_start = get("<|im_start|>");
        m.tok_im_end = get("<|im_end|>");
        m.tok_image_pad = get("<|image_pad|>");
        m.tok_video_pad = get("<|video_pad|>");
        std::fprintf(stderr, "strata-q36: <|im_start|> %d, <|image_pad|> %d (message-start snapshots, pictures)\n",
                     m.tok_im_start, m.tok_image_pad);
    }
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
    }
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

// Only prompt prefill uses these buffers. The batch admission gate is held
// until that prefill has fully drained; draft/decode never borrow them.
struct Q36SharedPrefill {
    uint16_t* dequant=nullptr;
    int64_t dequant_elems=0;
    void* workspace=nullptr;
    void* flash=nullptr;
    size_t flash_bytes=0;
    uint8_t* rollback=nullptr;
    size_t rollback_bytes=0;
    ~Q36SharedPrefill() {
        if(dequant) cudaFree(dequant); if(workspace) cudaFree(workspace);
        if(flash) cudaFree(flash); if(rollback) cudaFree(rollback);
    }
};
struct Engine {
    bool resident_prefix_reuse=false, resident_prefix_verify=false;
    strata::q36::resident::Prefix resident_prefix;
    int64_t resident_teacher_end=0, resident_ram_equal_end=0;
    std::vector<strata::q36::resident::PageStamp> resident_ram_stamps;
    std::vector<strata::q36::resident::PageStamp> resident_host_writes;
    void finish_resident_host_writes() {
        if(!resident_prefix_reuse)return;
        for(const auto& ticket:resident_host_writes)
            if(!cache_hub->finish_page_write(ticket))std::abort();
        resident_host_writes.clear();
    }
    strata::q36::resident::Identity resident_identity() const {
#ifndef __HIP_PLATFORM_AMD__
        if(kv_arena) return {uint64_t(reinterpret_cast<uintptr_t>(this)),
            uint64_t(reinterpret_cast<uintptr_t>(kv_arena->get())),kv_arena->mapping_generation()};
#endif
        return {};
    }
    void resident_before_write(int64_t first) {
        if(resident_prefix_reuse) {
            if(first<0){resident_prefix.invalidate();resident_ram_equal_end=0;}
            else {resident_prefix.before_write((size_t)first);resident_ram_equal_end=std::min(resident_ram_equal_end,first);}
        }
    }
    std::shared_ptr<strata::q36::cache::Hub> cache_hub;
    bool batch_coordinator=false;
    std::shared_ptr<Q36SharedPrefill> shared_prefill;
    uint8_t* borrowed_rollback=nullptr;
    uint8_t* rollback_data() const { return borrowed_rollback?borrowed_rollback:prefill_rollback.get(); }

#ifndef __HIP_PLATFORM_AMD__
    strata::q36::kv::Budget* kv_budget = nullptr;
    strata::q36::kv::Driver* kv_driver = nullptr;
    std::unique_ptr<strata::q36::kv::Arena> kv_arena;
#endif
    bool ensure_kv(size_t end) {
#ifndef __HIP_PLATFORM_AMD__
        return !kv_arena || kv_arena->ensure(end);
#else
        return true;
#endif
    }
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first)
    // Eager prefill is a transaction: one device copy of the recurrent state before each chunk.
    // KV writes at/above the chunk's logical end are discarded by leaving pos unchanged on abort.
    std::unique_ptr<uint8_t, decltype(&cudaFree)> prefill_rollback{nullptr, cudaFree};
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc = nullptr, *pm = nullptr, *pl = nullptr, *amv = nullptr;
    bool dynamic_attention = false;
    float *top_pv = nullptr, *top_ov = nullptr;   // the candidates (k_topk1 / k_topk2) and their host copy
    int *top_pi = nullptr, *top_oi = nullptr;
    uint8_t* top_h = nullptr;                     // pinned: [kVerify][kTopK] floats, then [kVerify][kTopK] ids
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S;           // per GDN layer (null for an attention layer)
    std::vector<__half*> kc, vc;           // per attention layer, FP16 [HKV][ctx][HD]; the MTP block's at n_layers
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    // Resident verify windows are split only at layer boundaries, with STOP checked
    // after each completed graph. Activations stay in the same x/h buffers throughout.
    static constexpr size_t kVerifyLayersPerGraph = 8;
    std::vector<cudaGraphExec_t> gv_parts[kVerify + 1];
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half* q16 = nullptr;             // the prompt attention's FP16 queries (K / V are read from the FP16 cache)
    int32_t* mrope = nullptr;          // [ctx][3] the cells' rotary positions (t, h, w): see build_positions
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    bool use_flash = false, flash_dumped = false;
    void* flash_work = nullptr;
    int flash_sms = 0;
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    void init(const Model& model, int ctx_, int tmax_) {
        m = &model;
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N); hlast = dalloc(N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        const char* fa = std::getenv("STRATA_Q36_FA");
        use_flash = (!fa || std::strcmp(fa, "0") != 0) && strata::qwen35::flash_available();
        if (use_flash) {
            int dev = 0;
            CK(cudaGetDevice(&dev));
            CK(cudaDeviceGetAttribute(&flash_sms, cudaDevAttrMultiProcessorCount, dev));
            const size_t bytes = strata::qwen35::flash_workspace_bytes(tmax, (int) g.n_head, (int) g.n_head_kv, ctx, flash_sms);
            if (bytes && !batch_coordinator) {
                if(shared_prefill) {
                    if(!shared_prefill->flash) {
                        CK(cudaMalloc(&shared_prefill->flash,bytes)); shared_prefill->flash_bytes=bytes;
                    }
                    if(bytes>shared_prefill->flash_bytes) { std::fprintf(stderr,"strata-q36: shared flash workspace too small\n"); std::exit(1); }
                    flash_work=shared_prefill->flash;
                } else CK(cudaMalloc(&flash_work,bytes));
            }
            kAttScoreFloats = 0;
            std::fprintf(stderr, "strata-q36: flash prefill (GQA, split-K), scratch %.1f MiB, %d SMs\n", bytes / 1048576.0, flash_sms);
        } else {
            att_s = dalloc(kAttScoreFloats);
            att_p16 = dalloc_t<__half>(kAttScoreFloats);
            q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
            std::fprintf(stderr, "strata-q36: cuBLAS prefill (STRATA_Q36_FA=0 or unsupported device)\n");
        }
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        mrope = dalloc_t<int32_t>((size_t) ctx * 3);
        {   // the identity until a request with pictures sets its own
            std::vector<int32_t> id3((size_t) ctx * 3);
            for (size_t c = 0; c < (size_t) ctx; ++c) id3[c * 3] = id3[c * 3 + 1] = id3[c * 3 + 2] = (int32_t) c;
            CK(cudaMemcpy(mrope, id3.data(), id3.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        use_mmq = !batch_coordinator && MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            bool gemm_ok=true;
            if(!batch_coordinator) {
                if(shared_prefill) {
                    if(!shared_prefill->dequant) {
                        CK(cudaMalloc((void**)&shared_prefill->dequant,(size_t)big*2));
                        shared_prefill->dequant_elems=big;
                        CK(cudaMalloc(&shared_prefill->workspace,size_t(32)<<20));
                    }
                    gemm_ok=big<=shared_prefill->dequant_elems && gemm.init_external(st,shared_prefill->dequant,
                        shared_prefill->dequant_elems,shared_prefill->workspace,size_t(32)<<20,gerr);
                } else gemm_ok=gemm.init(st,big,gerr);
            }
            if (!gemm_ok) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        top_pv = dalloc((size_t) kVerify * kTopBlocks * kTopK);
        top_pi = dalloc_t<int>((size_t) kVerify * kTopBlocks * kTopK);
        top_ov = dalloc((size_t) kVerify * kTopK);
        top_oi = dalloc_t<int>((size_t) kVerify * kTopK);
        CK(cudaHostAlloc((void**) &top_h, (size_t) kVerify * kTopK * 8, cudaHostAllocDefault));
        CK(cudaFuncSetAttribute(k_topk2, cudaFuncAttributeMaxDynamicSharedMemorySize, kTopBlocks * kTopK * 8));
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = ((dynamic_attention ? std::min(ctx, 128) : ctx) + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.n_head * (batch_coordinator ? 4 : kGemmT));   // the decode / verify path's rows (T <= kGemmT; a chunk runs GEMMs)
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        size_t kv_plane = 0;
#ifndef __HIP_PLATFORM_AMD__
        if (kv_budget) {
            size_t planes = m->has_mtp ? 2 : 0;
            for (const Layer& L : m->layers) if (L.attn) planes += 2;
            kv_arena = std::make_unique<strata::q36::kv::Arena>(*kv_budget, *kv_driver,
                planes * (size_t) g.n_head_kv, ctx, (size_t) g.head_dim * sizeof(__half));
            if (!kv_arena->init()) { std::fprintf(stderr, "strata-q36: KV virtual reservation failed\n"); std::exit(1); }
        }
#endif
        auto allocate_kv = [&]() -> __half* {
#ifndef __HIP_PLATFORM_AMD__
            if (kv_arena) {
                auto* p = reinterpret_cast<__half*>(kv_arena->get()) + kv_plane * (size_t) ctx * g.head_dim;
                kv_plane += (size_t) g.n_head_kv;
                return p;
            }
#endif
            return dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        };
        for (const Layer& L : m->layers) {
            if (L.attn) {
                conv.push_back(nullptr);
                S.push_back(nullptr);
                kc.push_back(allocate_kv());
                vc.push_back(allocate_kv());
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                conv.push_back(batch_coordinator?nullptr:dalloc(C * (g.ssm_d_conv - 1)));
                S.push_back(batch_coordinator?nullptr:dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                v_qkv.push_back(batch_coordinator?nullptr:dalloc((size_t) kVerify * C));
                v_hc.push_back(batch_coordinator?nullptr:dalloc((size_t) kVerify * C));
                v_alpha.push_back(batch_coordinator?nullptr:dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(batch_coordinator?nullptr:dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            conv.push_back(nullptr);
            S.push_back(nullptr);
            kc.push_back(allocate_kv());
            vc.push_back(allocate_kv());
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
    }

    // Batch owners have graph-free attention and can size partial reductions
    // for the admitted request, rather than reserve full-context scratch four
    // times. The caller owns this idle Engine and has fenced its stream.
    bool resize_attention(size_t cells) {
        if (!dynamic_attention) return true;
        const int chunks=(int)((std::max<size_t>(1,cells)+kChunkCells-1)/kChunkCells);
        if (chunks == n_chunk && pacc && pm && pl) return true;
        CK(cudaStreamSynchronize(st));
        CK(cudaFree(pacc)); CK(cudaFree(pm)); CK(cudaFree(pl));
        pacc=pm=pl=nullptr;
        const size_t rows=(size_t)att_rows*chunks;
        if (cudaMalloc((void**)&pacc, rows*(size_t)m->g.head_dim*4)!=cudaSuccess ||
            cudaMalloc((void**)&pm, rows*4)!=cudaSuccess || cudaMalloc((void**)&pl,rows*4)!=cudaSuccess) {
            (void)cudaGetLastError();
            if(pacc) cudaFree(pacc); if(pm) cudaFree(pm); if(pl) cudaFree(pl);
            pacc=pm=pl=nullptr; return false;
        }
        n_chunk=chunks; return true;
    }
    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// STRATA_Q36_MROPE_OFF=1 (a check only): every cell rotates by its index, pictures too (the positions ignored)
    static bool mrope_off() {
        static const bool off = std::getenv("STRATA_Q36_MROPE_OFF") != nullptr;
        return off;
    }
    /// The decode attention's partials for G query heads per KV head (load() admits G in {1, 2, 4, 8, 16}).
    void attn_part(int G, dim3 grid, const float* qq, const __half* k, const __half* v, const int32_t* dpos, int t0, int H,
                   int HKV, int ctx_, float scale, int nch) {
        switch (G) {
        case 1: k_attn_part_gqa<1><<<grid, 256, 0, st>>>(qq, k, v, dpos, t0, H, HKV, ctx_, scale, nch, pacc, pm, pl); break;
        case 2: k_attn_part_gqa<2><<<grid, 256, 0, st>>>(qq, k, v, dpos, t0, H, HKV, ctx_, scale, nch, pacc, pm, pl); break;
        case 4: k_attn_part_gqa<4><<<grid, 256, 0, st>>>(qq, k, v, dpos, t0, H, HKV, ctx_, scale, nch, pacc, pm, pl); break;
        case 8: k_attn_part_gqa<8><<<grid, 256, 0, st>>>(qq, k, v, dpos, t0, H, HKV, ctx_, scale, nch, pacc, pm, pl); break;
        default: k_attn_part_gqa<16><<<grid, 256, 0, st>>>(qq, k, v, dpos, t0, H, HKV, ctx_, scale, nch, pacc, pm, pl); break;
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    bool layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout, bool interruptible = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep(h, N, T, xq);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
            // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
            // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
            // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
            // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
            if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                  be_b, gdn_y, HK, HV, T);
                k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
            } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                       T, !verify);
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep(h, N, T, xq);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            resident_before_write(p0);
            k_attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, kc[li], vc[li], H,
                                                              HKV, HD, m->n_rot, dpos, ctx, m->rope_base, m->eps,
                                                              m->has_mrope && !mrope_off() ? mrope : nullptr);
            if (T > kGemmT) dump_flash_inputs(li, T, (int) p0);
            if (T > kGemmT && use_flash) {
                strata::qwen35::flash_prefill(q, kc[li], vc[li], gate, att, flash_work,
                                            T, H, HKV, ctx, (int) p0, flash_sms, st);
            } else if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = kc[li] + (size_t) kvh * ctx * HD;   // cells [0, P) of the cache
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = vc[li] + (size_t) kvh * ctx * HD;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    attn_part(H / HKV, dim3(nc, nt * HKV), q, kc[li], vc[li], dpos, t0, H, HKV, ctx,
                              1.0f / sqrtf((float) HD), n_chunk);
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att);
                }
            }
            mm(L.wo, att, y, T);
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        if (one) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, 1);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        // MMQ already fences at its router below. The grouped path has no host read there:
        // fence only for a prefill transaction, never during decode/verify graph capture.
        if (interruptible && !(T > kGemmT && use_mmq)) {
            CK(cudaStreamSynchronize(st));
            if (cancel && cancel->load()) return false;
        }
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (interruptible && cancel && cancel->load()) return false;
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            k_group<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, (int) g.n_expert, L.blobs, L.stride, grp_ptr, grp_start,
                                            n_groups, ent_dst, ent_tok);
            quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        apply(L.sh_gate, xq, sg, T);
        apply(L.sh_up, xq, su, T);
        if (T <= kGemmT) {   // SwiGLU and its q8_1 image in one launch
            K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);
            mmq(L.sh_down, q8, sy, T);
        } else {
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (!one) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, sy, one ? logits_r + g.n_expert : sgate, 1, wn, hout,
                                          N, m->eps);
        return true;
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    bool forward_span(int T, int64_t p0, int heads, bool graph_mode, bool verify, size_t begin, size_t end,
                      const float* const* img = nullptr, bool interruptible = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (begin == 0) {
            if (main_hybrid && T > hybrid_t && use_mmq) {
                std::vector<int> seq;
                for (size_t l = 0; l < m->layers.size(); ++l)
                    if (lhy(l)) seq.push_back((int) l);
                stream_begin(seq);
            }
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
            put_image_rows(img, T, x);
            k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        }
        for (size_t l = begin; l < end; ++l) {
            const bool last = l + 1 == m->layers.size();
            if (!layer(m->layers[l], l, T, p0, pos0, graph_mode, verify,
                       last ? m->out_norm : m->layers[l + 1].attn_norm, last ? hmain : h, interruptible)) return false;
        }
        if (interruptible && cancel && cancel->load()) return false;
        if (end == m->layers.size() && heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
        return true;
    }
    std::function<bool(int, int64_t, int, bool, const float* const*)> batch_dispatch;
    int batch_slot = -1;
    unsigned long long batch_epoch = 0;
    std::function<void()> batch_begin;
    std::function<void(long long,size_t)> batch_parsed;
    std::function<bool(size_t)> batch_admit;
    std::function<void()> batch_idle;
    bool forward(int T, int64_t p0, int heads, bool graph_mode, bool verify, const float* const* img = nullptr,
                 bool interruptible = false) {
        if (batch_dispatch && T <= kVerify && !interruptible)
            return batch_dispatch(T, p0, heads, verify, img);
        return forward_span(T, p0, heads, graph_mode, verify, 0, m->layers.size(), img, interruptible);
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    bool forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode,
                     const float* const* img = nullptr, bool interruptible = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        put_image_rows(img, T, y);   // an image cell's e input is its embedding row (llama.cpp's MTP does the same)
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        if (!layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp,
                   interruptible)) return false;
        if (interruptible && cancel && cancel->load()) return false;
        if (draft) head_rows(hmtp + (size_t) (T - 1) * N, 1, logits_m, pick_m);
        return true;
    }

    #include "q36_batch_forward.hpp"

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_route, st));
        k_group_hy<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, (int) g.n_expert, hy[li].slot_of_d, L.blobs, L.stride,
                                           grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
        quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) y] >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                       cudaMemcpyHostToDevice, st));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }

    // STRATA_Q36_FA_DUMP=path, _DUMP_P=P, _DUMP_LAYER=li captures one real prefill activation.
    // Compact binary: int32 {T,H,HKV,P}, float Q/gate, half K/V [HKV,P,256]. No graph/decode capture.
    void dump_flash_inputs(size_t li, int T, int p0) {
        const char* path = std::getenv("STRATA_Q36_FA_DUMP");
        if (!path || warming || flash_dumped) return;
        const char* ps = std::getenv("STRATA_Q36_FA_DUMP_P");
        const char* ls = std::getenv("STRATA_Q36_FA_DUMP_LAYER");
        const int P = p0 + T;
        if ((ps && P != std::atoi(ps)) || li != (size_t) (ls ? std::atoi(ls) : 3)) return;
        CK(cudaStreamSynchronize(st));
        FILE* f = std::fopen(path, "wb");
        if (!f) { std::perror(path); std::exit(1); }
        const int H = (int) m->g.n_head, HKV = (int) m->g.n_head_kv;
        const int32_t header[] = {T, H, HKV, P};
        std::fwrite(header, sizeof(header), 1, f);
        auto save = [&](const void* data, size_t bytes) {
            std::vector<unsigned char> host(bytes);
            CK(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost));
            if (std::fwrite(host.data(), 1, bytes, f) != bytes) { std::perror(path); std::exit(1); }
        };
        save(q, (size_t) T * H * 256 * 4);
        save(gate, (size_t) T * H * 256 * 4);
        for (const __half* cache : {kc[li], vc[li]})
            for (int h = 0; h < HKV; ++h) save(cache + (size_t) h * ctx * 256, (size_t) P * 256 * 2);
        std::fclose(f);
        flash_dumped = true;
        std::fprintf(stderr, "strata-q36: captured attention layer %zu, T=%d P=%d to %s\n", li, T, P, path);
    }

    size_t att_shrink_bytes() const {
        const size_t small = (size_t) 16 << 20;
        return kAttScoreFloats > small ? (kAttScoreFloats - small) * 6 : 0;
    }
    void shrink_att(size_t floats) {
        if (use_flash) return;
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (cudaGraphExec_t* g : {&g1, &gm1})
            if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
        for (auto& g : gv)
            if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        for (auto& parts : gv_parts)
            for (auto& g : parts)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    /// An image's rows in place of its cells' token embeddings: rows r of the batch with img[r] set get that host row
    /// (runs of consecutive rows of one picture in one copy).
    void put_image_rows(const float* const* img, int T, float* dst) {
        if (img == nullptr) return;
        const size_t N = (size_t) m->g.n_embd;
        for (int r = 0; r < T;) {
            if (img[r] == nullptr) { ++r; continue; }
            int e = r + 1;
            while (e < T && img[e] == img[e - 1] + N) ++e;
            CK(cudaMemcpyAsync(dst + (size_t) r * N, img[r], (size_t) (e - r) * N * 4, cudaMemcpyHostToDevice, st));
            r = e;
        }
    }

    // ---- state snapshots: the GDN layers' conv history and recurrence (the attention layers' K / V are per cell and
    // stay valid for every cell below the position) and hlast (the MTP seed), at chosen positions of a prompt.  A
    // snapshot is valid for a request whose first `pos` cells hash to `hash` while those cells are still the ones
    // the cache holds (pos <= the common start with `live`, checked by the caller).  Kept in pinned host memory.
    using Snap = strata::q36::cache::Snapshot;
    static constexpr uint64_t kPrioAge = 96;   // a snapshot of priority p is kept as if used p * kPrioAge saves later
    std::vector<Snap> snaps;
    const Snap* pinned_snap = nullptr;
    std::function<bool()> evict_snapshot_owner;   // evicts one whole LRU parked prefix (KV and state refs)
    std::function<void(Snap&)> snapshot_saved;    // bounded ownership of the current prefix's useful anchors
    int max_snaps = 128;
    uint64_t snap_clock = 0;
    long long snaps_saved = 0, snaps_restored = 0;
    std::vector<Snap>& snapshot_pool() { return cache_hub ? cache_hub->snaps : snaps; }
    uint64_t& snapshot_clock() { return cache_hub ? cache_hub->snap_clock : snap_clock; }
    std::vector<uint8_t*>& page_pool() { return cache_hub ? cache_hub->free_pages : free_pages; }
    size_t& host_page_count() { return cache_hub ? cache_hub->n_pages : n_pages; }
    std::unique_lock<std::recursive_mutex> host_lock() {
        return cache_hub ? std::unique_lock<std::recursive_mutex>(cache_hub->mutex)
                         : std::unique_lock<std::recursive_mutex>();
    }
    size_t snap_bytes() const {
        const auto& g = m->g;
        size_t b = (size_t) g.n_embd * 4;
        for (size_t l = 0; l < conv.size(); ++l)
            if (conv[l])
                b += (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4 +
                     (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        return b;
    }
    template <class Copy> void snap_walk(uint8_t* host, Copy cp) {
        const auto& g = m->g;
        const size_t cb = (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4;
        const size_t sb = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        size_t off = 0;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (!conv[l]) continue;
            cp((void*) conv[l], host + off, cb);
            off += cb;
            cp((void*) S[l], host + off, sb);
            off += sb;
        }
        cp((void*) hlast, host + off, (size_t) m->g.n_embd * 4);
    }
    void alloc_prefill_rollback() {
        if(shared_prefill) {
            if(!shared_prefill->rollback) {
                CK(cudaMalloc((void**)&shared_prefill->rollback,snap_bytes()));
                shared_prefill->rollback_bytes=snap_bytes();
            }
            if(shared_prefill->rollback_bytes!=snap_bytes()) { std::fprintf(stderr,"strata-q36: shared rollback shape mismatch\n"); std::exit(1); }
            borrowed_rollback=shared_prefill->rollback; return;
        }
        uint8_t* saved = nullptr;
        CK(cudaMalloc(&saved, snap_bytes()));   // every byte is overwritten before the first possible restore
        prefill_rollback.reset(saved);
        std::fprintf(stderr, "strata-q36: transactional prefill rollback %.1f MiB in VRAM\n",
                     snap_bytes() / 1048576.0);
    }
    void prefill_state_copy(bool restore) {
        snap_walk(rollback_data(), [&](void* state, uint8_t* saved, size_t bytes) {
            CK(cudaMemcpyAsync(restore ? state : saved, restore ? saved : state, bytes,
                               cudaMemcpyDeviceToDevice, st));
        });
    }
    void touch_snap(Snap& sn, int prio) {
        auto lock=host_lock();
        sn.prio = std::max(sn.prio, prio);
        sn.used = ++snapshot_clock();
    }
    // Only unowned states may be overwritten. A retained KV prefix and its GDN anchors
    // have one lifetime; priority only chooses among entries nobody currently owns.
    Snap* snapshot_victim() {
        Snap* v = nullptr;
        for (Snap& sn : snapshot_pool()) {
            if (&sn == pinned_snap || sn.refs != 0 || sn.pins != 0 || sn.writing) continue;
            if (sn.pos < 0) return &sn;
            if (!v || std::make_pair(sn.prio, sn.used) < std::make_pair(v->prio, v->used)) v = &sn;
        }
        return v;
    }
    /// The current state: reclaim an unowned slot, else evict an entire parked prefix.
    void save_snap(uint64_t hash, int prio = 0) {
        if(cache_hub) {
            if(max_snaps<=0) return;
            strata::q36::cache::Hub::SnapshotWrite writer;
            Snap* victim=nullptr;
            {
                auto lock=host_lock();
                for(Snap& sn:snapshot_pool())
                    if(!sn.writing && sn.pos==pos && sn.hash==hash) {
                        touch_snap(sn,prio); if(snapshot_saved) snapshot_saved(sn); return;
                    }
                victim=snapshot_victim();
                while(!victim && evict_snapshot_owner && evict_snapshot_owner()) victim=snapshot_victim();
                if(!victim) return;
                writer=cache_hub->reserve_snapshot(*victim);
                if(!writer || !writer.data()) { std::fprintf(stderr,"strata-q36: shared snapshot reservation failed\n"); std::abort(); }
            }
            // A writer reservation remains unavailable globally until every
            // D2H byte is complete. No CUDA wait is inside the metadata mutex.
            snap_walk(writer.data(),[&](void* device,uint8_t* host,size_t bytes) {
                CK(cudaMemcpyAsync(host,device,bytes,cudaMemcpyDeviceToHost,st));
            });
            CK(cudaStreamSynchronize(st));
            {
                auto lock=host_lock();
                if(!writer.commit(pos,hash,prio,[] { return true; })) std::abort();
                ++snaps_saved;
                if(snapshot_saved) snapshot_saved(*victim);
            }
            return;
        }
        if (max_snaps <= 0) return;
        for (Snap& sn : snapshot_pool())
            if (sn.pos == pos && sn.hash == hash) {
                touch_snap(sn, prio);
                if (snapshot_saved) snapshot_saved(sn);
                return;
            }
        if (snapshot_pool().empty()) snapshot_pool().resize((size_t) max_snaps);
        Snap* v = snapshot_victim();
        while (!v && evict_snapshot_owner && evict_snapshot_owner()) v = snapshot_victim();
        if (!v) return;   // e.g. a one-entry ring whose source is pinned; keep it correct
        if (!v->host && cudaHostAlloc((void**) &v->host, snap_bytes(), cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata-q36: no pinned memory for a state snapshot (%.0f MiB): snapshots off\n",
                         snap_bytes() / 1048576.0);
            max_snaps = 0;
            return;
        }
        snap_walk(v->host, [&](void* d, uint8_t* hst, size_t b) {
            CK(cudaMemcpyAsync(hst, d, b, cudaMemcpyDeviceToHost, st));
        });
        v->pos = pos;
        v->hash = hash;
        v->prio = prio;
        v->used = ++snapshot_clock();
        ++snaps_saved;
        if (snapshot_saved) snapshot_saved(*v);
    }
    void restore_snap(Snap& sn) {
        strata::q36::cache::Hub::SnapPin read_pin;
        if(cache_hub) read_pin=strata::q36::cache::Hub::SnapPin(*cache_hub,sn);
        CK(cudaStreamSynchronize(st));
        snap_walk(sn.host, [&](void* d, uint8_t* hst, size_t b) {
            CK(cudaMemcpyAsync(d, hst, b, cudaMemcpyHostToDevice, st));
        });
        if(cache_hub) CK(cudaStreamSynchronize(st));
        auto lock=host_lock();
        pos = sn.pos;
        sn.used = ++snapshot_clock();
        ++snaps_restored;
    }
    void drop_snaps() {
        auto lock=host_lock();
        for(Snap& sn:snapshot_pool()) if(!cache_hub || (sn.refs==0 && sn.pins==0 && !sn.writing)) sn.pos=-1;
    }
    /// The snapshot ring's pinned memory, all at once (at load: a 63 MiB cudaHostAlloc inside a prompt costs ~20 ms)
    void alloc_snaps() {
        if(cache_hub && cache_hub->snaps_initialized) {
            if(cache_hub->snapshot_bytes!=snap_bytes()) { std::fprintf(stderr,"strata-q36: shared snapshot shape mismatch\n"); std::abort(); }
            max_snaps=(int)snapshot_pool().size(); return;
        }
        if (max_snaps <= 0) return;
        const auto t0 = Clock::now();
        if(!cache_hub) snapshot_pool().resize((size_t) max_snaps);
        int got = 0;
        for (Snap& sn : snapshot_pool()) {
            if (!sn.host && cudaHostAlloc((void**) &sn.host, snap_bytes(), cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                break;
            }
            ++got;
        }
        if(cache_hub) {
            if(got!=max_snaps) { std::fprintf(stderr,"strata-q36: shared snapshot pool could not allocate requested budget\n"); std::abort(); }
            cache_hub->snapshot_bytes=snap_bytes(); cache_hub->snaps_initialized=true;
        }
        if (got < max_snaps) {
            snapshot_pool().resize((size_t) got);
            max_snaps = got;
        }
        std::fprintf(stderr, "strata-q36: %d state snapshots of %.1f MiB in pinned host memory (%.1f GiB) in %.1f s\n",
                     max_snaps, snap_bytes() / 1048576.0, max_snaps * (double) snap_bytes() / (1 << 30), ms_since(t0) / 1000.0);
    }

    // ---- parked conversations: every conversation's attention cells mirrored in pinned host pages, so that a request
    // of another conversation (another caller with its own long prompt) brings its cells back with one copy instead of
    // reading them again.  A page is kPage cells of every attention layer's K and V ([layer][K|V][HKV][kPage][HD]).
    static constexpr int kPage = 1024;
    std::vector<int> att_li;            // the attention layers' state indices (the MTP block's too)
    std::vector<uint8_t*> free_pages;   // the pool (allocated at load, --park-mib)
    size_t n_pages = 0;
    size_t page_bytes() const {
        return att_li.size() * 2 * (size_t) m->g.n_head_kv * kPage * (size_t) m->g.head_dim * sizeof(__half);
    }
    void alloc_pages(size_t budget) {
        att_li.clear();
        for (size_t l = 0; l < kc.size(); ++l)
            if (kc[l]) att_li.push_back((int) l);
        if(cache_hub && cache_hub->pages_initialized) {
            if(cache_hub->page_bytes!=page_bytes()) { std::fprintf(stderr,"strata-q36: shared KV page shape mismatch\n"); std::abort(); }
            return;
        }
        if (budget == 0 || att_li.empty()) return;
        const auto t0 = Clock::now();
        const size_t want = budget / page_bytes();
        for (size_t i = 0; i < want; ++i) {
            uint8_t* p = nullptr;
            if (cudaHostAlloc((void**) &p, page_bytes(), cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                break;
            }
            page_pool().push_back(p);
        }
        host_page_count() = page_pool().size();
        if(cache_hub) { cache_hub->page_bytes=page_bytes(); cache_hub->pages_initialized=true; }
        std::fprintf(stderr, "strata-q36: %zu conversation pages of %d cells (%.1f MiB each, %.1f GiB) in pinned host "
                     "memory in %.1f s\n", host_page_count(), kPage, page_bytes() / 1048576.0,
                     host_page_count() * (double) page_bytes() / (1 << 30), ms_since(t0) / 1000.0);
    }
    /// Cells [a, b) between the cache and `pages` (page i holds cells [i kPage, (i + 1) kPage)), on the stream.
    void kv_copy(const std::vector<uint8_t*>& pages, int64_t a, int64_t b, bool to_host) {
#ifndef __HIP_PLATFORM_AMD__
        if(kv_arena && (a<0 || b<a || b>ctx || (size_t)b>kv_arena->committed_tokens())) {
            std::fprintf(stderr,"strata-q36: parked KV copy outside mapped prefix\n"); std::abort();
        }
#endif
        const size_t HD = (size_t) m->g.head_dim, HKV = (size_t) m->g.n_head_kv, cb = HD * sizeof(__half);
        if(resident_prefix_reuse && to_host && b>a) {
            if(!resident_host_writes.empty())std::abort();
            for(size_t pg=(size_t)a/kPage;pg<=((size_t)b-1)/kPage;++pg)
                resident_host_writes.push_back(cache_hub->begin_page_write(pages[pg]));
        }
#ifndef __HIP_PLATFORM_AMD__
        if(kv_arena) {
            // Sparse mapped prefixes are separate regions. A pitched 2D
            // envelope includes unmapped gaps between heads; use validated
            // contiguous row ranges, with the exact existing host layout.
            for(size_t j=0;j<att_li.size();++j) for(int kv=0;kv<2;++kv) {
                auto* device=reinterpret_cast<uint8_t*>((kv?vc:kc)[(size_t)att_li[j]]);
                bool valid=strata::q36::kv::for_each_copy_row((size_t)ctx,cb,HKV,kPage,
                    kv_arena->committed_tokens(),(size_t)a,(size_t)b,
                    [&](size_t pg,size_t device_offset,size_t host_offset,size_t bytes) {
                        auto* host=pages[pg]+(j*2+(size_t)kv)*HKV*kPage*cb+host_offset;
                        if(to_host) CK(cudaMemcpyAsync(host,device+device_offset,bytes,cudaMemcpyDeviceToHost,st));
                        else CK(cudaMemcpyAsync(device+device_offset,host,bytes,cudaMemcpyHostToDevice,st));
                    });
                if(!valid) { std::fprintf(stderr,"strata-q36: invalid sparse KV copy plan\n"); std::abort(); }
            }
            return;
        }
#endif
        for (int64_t c = a; c < b;) {
            const int64_t pg = c / kPage, o = c - pg * kPage, e = std::min<int64_t>(b, (pg + 1) * kPage);
            uint8_t* page = pages[(size_t) pg];
            for (size_t j = 0; j < att_li.size(); ++j)
                for (int kv = 0; kv < 2; ++kv) {
                    __half* dev = (kv ? vc : kc)[(size_t) att_li[j]] + (size_t) c * HD;
                    uint8_t* host = page + ((j * 2 + (size_t) kv) * HKV * kPage + (size_t) o) * cb;
                    const size_t w = (size_t) (e - c) * cb;
                    if (to_host)
                        CK(cudaMemcpy2DAsync(host, kPage * cb, dev, (size_t) ctx * cb, w, HKV, cudaMemcpyDeviceToHost, st));
                    else
                        CK(cudaMemcpy2DAsync(dev, (size_t) ctx * cb, host, kPage * cb, w, HKV, cudaMemcpyHostToDevice, st));
                }
            c = e;
        }
    }

    bool verify_resident_prefix(const std::vector<uint8_t*>& pages,size_t end) {
#ifndef __HIP_PLATFORM_AMD__
        if(!resident_prefix_verify || !end)return true;
        const size_t HD=(size_t)m->g.head_dim,HKV=(size_t)m->g.n_head_kv,cb=HD*sizeof(__half);
        uint8_t* staging=nullptr;
        CK(cudaHostAlloc((void**)&staging,kPage*cb,cudaHostAllocDefault));
        struct HostBuffer { uint8_t* p;~HostBuffer(){if(p)cudaFreeHost(p);} } buffer{staging};
        std::vector<strata::q36::resident::Difference> differences(att_li.size()*2*HKV);
        for(size_t j=0;j<att_li.size();++j)for(int kv=0;kv<2;++kv) {
            const auto* device=reinterpret_cast<const uint8_t*>((kv?vc:kc)[(size_t)att_li[j]]);
            const bool valid=strata::q36::kv::for_each_copy_row((size_t)ctx,cb,HKV,kPage,
                kv_arena->committed_tokens(),0,end,
                [&](size_t pg,size_t device_offset,size_t host_offset,size_t bytes) {
                    const size_t head=host_offset/(kPage*cb);
                    const size_t first=(device_offset-head*(size_t)ctx*cb)/cb;
                    const auto* expected=pages[pg]+(j*2+(size_t)kv)*HKV*kPage*cb+host_offset;
                    CK(cudaMemcpyAsync(staging,device+device_offset,bytes,cudaMemcpyDeviceToHost,st));
                    // Diagnostic only: each bounded staging segment must be
                    // complete before CPU comparison or buffer reuse.
                    CK(cudaStreamSynchronize(st));
                    differences[(j*2+(size_t)kv)*HKV+head].observe(first,staging,expected,bytes,cb);
                });
            if(!valid)std::abort();
        }
        uint64_t bad_bytes=0;
        for(size_t plane=0;plane<differences.size();++plane) {
            const auto& diff=differences[plane];bad_bytes+=diff.different_bytes;
            const size_t head=plane%HKV,j=(plane/HKV)/2,kv=(plane/HKV)%2;
            std::fprintf(stderr,"Q36_RESIDENT_VERIFY {\"event\":\"plane\",\"slot_id\":%d,\"request_epoch\":%llu,\"layer\":%d,\"kv\":%zu,\"head\":%zu,\"cells\":%zu,\"bytes\":%llu,\"different_bytes\":%llu,\"different_cells\":%llu,\"first_cell\":%lld,\"first_byte_in_cell\":%zu,\"last_cell\":%lld,\"gpu_fnv1a64\":%llu,\"ram_fnv1a64\":%llu}\n",
                batch_slot,batch_epoch,att_li[j],kv,head,end,(unsigned long long)diff.bytes,
                (unsigned long long)diff.different_bytes,(unsigned long long)diff.different_cells,
                diff.same()?-1LL:(long long)diff.first_cell,diff.first_byte_in_cell,
                diff.same()?-1LL:(long long)diff.last_cell,(unsigned long long)diff.gpu_hash,(unsigned long long)diff.ram_hash);
        }
        std::fprintf(stderr,"Q36_RESIDENT_VERIFY {\"event\":\"decision\",\"slot_id\":%d,\"request_epoch\":%llu,\"proposed_skip_cells\":%zu,\"effective_skip_cells\":%zu,\"different_bytes\":%llu,\"fallback_full_ram\":%s}\n",
            batch_slot,batch_epoch,end,bad_bytes?size_t(0):end,(unsigned long long)bad_bytes,bad_bytes?"true":"false");
        return !bad_bytes;
#else
        (void)pages;(void)end;return false;
#endif
    }

    /// Set while a request runs (serve's STOP). Eager resident prefill rolls back an interrupted chunk;
    /// decode graphs and hybrid inference retain their complete-step / complete-chunk boundaries.
    const std::atomic<bool>* cancel = nullptr;
    bool prefill_aborted = false;   // no post-cancel host snapshot: the valid state remains live on the device

    /// A prompt chunk must end at this position, and the state there be kept as a snapshot of this hash.
    struct Cut {
        int64_t pos;
        uint64_t hash;
        int prio = 0;
    };

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  `img` (nullable,
    /// n entries): the host embedding row of an image cell, null for a token.  `cuts`: ascending positions in
    /// (pos, pos + n] where a chunk ends and a snapshot is taken.
    template <class F>
    void process(const int32_t* toks, int64_t n, F on_chunk, const float* const* img = nullptr,
                 const std::vector<Cut>* cuts = nullptr) {
        prefill_aborted = false;
        if(resident_prefix_reuse) resident_teacher_end=pos;
        if (!ensure_kv((size_t) (pos + n))) { std::fprintf(stderr, "strata-q36: unreserved KV operation\n"); std::abort(); }
        const int N = (int) m->g.n_embd;
        if (n == 1 && (img == nullptr || img[0] == nullptr) && (cuts == nullptr || cuts->empty())) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            rows_deferred = true;
            graph_run(g1, [&] { forward(1, pos, 1, use_graph, false); });
            rows_deferred = false;
            CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            if(resident_prefix_reuse && !(m->has_mtp && mtp_d>0))resident_teacher_end=pos;
            after_rows(1);
            on_chunk(1);
            return;
        }
        const int64_t start = pos;
        // Hybrid inference also mutates routing scores, the CPU pool and a prefetch stream.
        // Until those are transactional too, keep its existing whole-chunk cancellation.
        const bool transaction = cancel && rollback_data() && !m->hybrid;
        size_t ci = 0;
        while (cuts && ci < cuts->size() && (*cuts)[ci].pos <= start) ++ci;
        for (int64_t i = 0; i < n;) {
            if (cancel && cancel->load()) {
                // A preceding cache restore may still be queued even before the first chunk.
                CK(cudaStreamSynchronize(st));
                prefill_aborted = transaction;
                break;
            }
            int64_t end = std::min<int64_t>(i + tmax, n);
            if (cuts && ci < cuts->size() && (*cuts)[ci].pos - start < end) end = (*cuts)[ci].pos - start;
            const int T = (int) (end - i);
            const float* const* im = img ? img + i : nullptr;
            CK(cudaStreamSynchronize(st));
            if (transaction) prefill_state_copy(false);
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            bool complete = forward(T, pos, end == n ? 1 : 0, false, false, im, transaction);
            if (complete && transaction && cancel->load()) complete = false;
            if (complete && m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                complete = forward_mtp(T, pos, mtp_seed(), false, false, im, transaction);
            }
            if (complete) {
                CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (transaction) {
                    CK(cudaStreamSynchronize(st));
                    if (cancel->load()) complete = false;
                }
            }
            if (!complete) {
                // The same stream orders rollback after every already queued kernel. No head/MTP
                // continuation, published position or snapshot may escape a partial transaction.
                prefill_state_copy(true);
                CK(cudaStreamSynchronize(st));
                prefill_aborted = true;
                std::fprintf(stderr, "strata-q36: prefill rolled back to cell %lld at %.3f\n", (long long) pos, wall_ms());
                break;
            }
            pos += T;
            if(resident_prefix_reuse)resident_teacher_end=pos;
            i = end;
            while (cuts && ci < cuts->size() && (*cuts)[ci].pos <= pos) {
                if ((*cuts)[ci].pos == pos) save_snap((*cuts)[ci].hash, (*cuts)[ci].prio);
                ++ci;
            }
            CK(cudaStreamSynchronize(st));
            on_chunk(i);
            if (cancel && cancel->load() && i < n) {
                prefill_aborted = transaction;   // this complete chunk stays committed; no extra host snapshot
                break;
            }
        }
        if (m->hybrid && !(cancel && cancel->load())) rebalance(hybrid_swaps_after_prompt);
    }
    int hybrid_swaps_after_prompt = 2048;
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `judge` (sampled requests): decides the window from its rows' logits (in `logits`) instead of the greedy picks:
    /// returns the accepted count a and writes picks[a], the token after them.
    /// Returns -1 on STOP before commit: pos/GDN/hlast are unchanged, picks must not be read.
    const bool cancel_trace = std::getenv("STRATA_Q36_CANCEL_TRACE") != nullptr;
    double cancel_abort_at = 0;
    const char* cancel_abort_stage = "none";
    int64_t cancel_abort_pos = -1;
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::function<int(int, const std::vector<int32_t>&, int32_t*)>* judge = nullptr,
                  const std::function<int(int)>* draft_pick = nullptr) {
        if (!ensure_kv((size_t) (pos + D + 1))) { std::fprintf(stderr, "strata-q36: unreserved KV verify\n"); std::abort(); }
        const int N = (int) m->g.n_embd;
        // MTP attention only changes KV rows (uncommitted rows >=pos are ignored). Hybrid
        // routing/CPU work and a future recurrent MTP block would need their own rollback.
        // Serve warmup must capture the same segmented graphs before READY, before cancel is attached.
        const bool interruptible = (cancel || (warming && rollback_data())) && !m->hybrid && m->mtp.attn;
        auto stopped = [&](const char* phase = "draft") {
            if (!interruptible || !cancel || !cancel->load()) return false;
            CK(cudaStreamSynchronize(st));   // include all queued draft/verify work before reporting DONE
            // No synchronous log write on the cancellation path. The optional
            // summary after cancelled reports this already-drained abort boundary.
            if (cancel_trace) {
                cancel_abort_at = wall_ms();
                cancel_abort_stage = phase;
                cancel_abort_pos = pos;
            }
            return true;
        };
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        if (stopped("before_draft")) return -1;
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (stopped()) return -1;
            drafts[0] = draft_pick ? (*draft_pick)(hpin[2]) : hpin[2];
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            if (stopped()) return -1;
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (stopped()) return -1;
            drafts[(size_t) d] = draft_pick ? (*draft_pick)(hpin[2]) : hpin[2];
        }
        if (stopped("before_verify")) return -1;
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        if (batch_dispatch) {
            if (!batch_dispatch(T, pos, T, true, nullptr)) { rows_deferred = false; return -1; }
        } else if (interruptible) {
            auto& parts = gv_parts[T];
            const size_t np = (m->layers.size() + kVerifyLayersPerGraph - 1) / kVerifyLayersPerGraph;
            if (parts.empty()) parts.resize(np, nullptr);
            for (size_t part = 0; part < np; ++part) {
                const size_t begin = part * kVerifyLayersPerGraph;
                const size_t end = std::min(m->layers.size(), begin + kVerifyLayersPerGraph);
                graph_run(parts[part], [&] { forward_span(T, pos, T, use_graph, true, begin, end); });
                CK(cudaStreamSynchronize(st));
                if (stopped("verify")) { rows_deferred = false; return -1; }
            }
        } else {
            graph_run(gv[T], [&] { forward(T, pos, T, use_graph, true); });
        }
        rows_deferred = false;
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (stopped("after_verify")) return -1;
        int a = 0;
        if (judge) {
            a = (*judge)(T, drafts, picks);
        } else {
            while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
            for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        }
        if (stopped("before_commit")) return -1;
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        if(dynamic_attention && !resize_attention(std::min<size_t>((size_t)ctx,(size_t)tmax+kGemmT+2+2*kVerify))) {
            std::fprintf(stderr,"strata-q36: warmup attention workspace failed\n"); std::abort();
        }
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        if (tmax > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_cpu_ms = 0;
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    /// Logits rows [0, n) (n <= kVerify) -> each row's kTopK largest (value, id), descending, in `vals` / `ids`.
    void topk_rows(int n, std::vector<float>& vals, std::vector<int>& ids, const float* source = nullptr) {
        k_topk1<<<dim3(kTopBlocks, n), 512, 0, st>>>(source ? source : logits, (int) m->g.n_vocab, top_pv, top_pi);
        k_topk2<<<n, 1024, kTopBlocks * kTopK * 8, st>>>(top_pv, top_pi, top_ov, top_oi);
        CK(cudaMemcpyAsync(top_h, top_ov, (size_t) n * kTopK * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(top_h + (size_t) kVerify * kTopK * 4, top_oi, (size_t) n * kTopK * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        vals.assign((const float*) top_h, (const float*) top_h + (size_t) n * kTopK);
        ids.assign((const int*) (top_h + (size_t) kVerify * kTopK * 4), (const int*) (top_h + (size_t) kVerify * kTopK * 4) + (size_t) n * kTopK);
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 768 << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t slack = ((size_t) 512 << 20) + E.att_shrink_bytes();   // reserve + reclaimable scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack)) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                     "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                     (double) packed_bytes / (1 << 30));
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t att_save = E.att_shrink_bytes();   // zero for flash; att_s (4 B) + att_p16 (2 B) otherwise
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-q36: the experts nearly fit: %sthe VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n", E.use_flash ? "flash prefill keeps its scratch, " : "the attention scores buffer at 64 MB and ");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-q36: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.use_mmq ? E.tmax : E.tmax));
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    const size_t staging = E.use_mmq && E.main_hybrid ? 2 * (max_layer + kMmqTail) : 0;
    size_t slot_bytes = budget2 > staging ? budget2 - staging : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging + ((size_t) 512 << 20) ? fr - staging - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies)
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
        }
        H.score.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, T * Kp * N * 4));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ images
/// The engine's own copy of the pictures it was sent (pinned host rows), by the server's key: a later request can
/// name a picture by its key alone (an `SVR1` record) and the server need not encode it again.  Least recently used
/// go first past `budget` bytes; the pictures of the request being read are never dropped.
struct ImageCache {
    struct Entry {
        float* rows = nullptr;
        int64_t n = 0, nx = 0, ny = 0;
        uint64_t used = 0;
    };
    std::recursive_mutex mutex;
    bool parallel=false;  // batch1 retains the established image eviction order
    std::map<uint64_t, size_t> pinned;
    struct Pins {
        ImageCache& cache; std::vector<uint64_t> keys;
        explicit Pins(ImageCache& c): cache(c) {}
        void add(uint64_t key) { if(!cache.parallel) return; std::lock_guard<std::recursive_mutex> lock(cache.mutex); if (std::find(keys.begin(),keys.end(),key)==keys.end()) { keys.push_back(key); ++cache.pinned[key]; } }
        ~Pins() { std::lock_guard<std::recursive_mutex> lock(cache.mutex); for(auto key:keys) if (--cache.pinned[key]==0) cache.pinned.erase(key); }
    };
    std::map<uint64_t, Entry> m;
    size_t bytes = 0, budget = (size_t) 2048 << 20;
    uint64_t clock = 0;
    int64_t width = 0;
    Entry* find(uint64_t key) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto it = m.find(key);
        if (it == m.end()) return nullptr;
        it->second.used = ++clock;
        return &it->second;
    }
    Entry* put(uint64_t key, const float* rows, int64_t n, int64_t nx, int64_t ny, const std::vector<uint64_t>& keep) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (Entry* e = find(key)) {
            if (e->n == n && e->nx == nx && e->ny == ny) return e;
            if (parallel && pinned[key] > 1) return nullptr;
            drop(key);
        }
        const size_t b = (size_t) n * (size_t) width * 4;
        while (bytes + b > budget && !m.empty()) {   // the least recently used that this request does not hold
            auto v = m.end();
            for (auto it = m.begin(); it != m.end(); ++it)
                if (!pinned.count(it->first) && std::find(keep.begin(), keep.end(), it->first) == keep.end() && (v == m.end() || it->second.used < v->second.used))
                    v = it;
            if (v == m.end()) break;
            drop(v->first);
        }
        Entry e;
        if (cudaHostAlloc((void**) &e.rows, b, cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        std::memcpy(e.rows, rows, b);
        e.n = n; e.nx = nx; e.ny = ny; e.used = ++clock;
        bytes += b;
        return &(m[key] = e);
    }
    void drop(uint64_t key) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto it = m.find(key);
        if (it == m.end()) return;
        bytes -= (size_t) it->second.n * (size_t) width * 4;
        cudaFreeHost(it->second.rows);
        m.erase(it);
    }
};

uint64_t fnv_mix(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xff;
        h *= 1099511628211ull;
    }
    return h;
}
constexpr uint64_t kFnvSeed = 1469598103934665603ull;
constexpr uint64_t kImageCell = 1ull << 63;   // a cell value with this bit is an image row (its key and row), else a token

/// A picture of a request: its grid and key, and its rows (host), or null for a reference the engine does not hold.
struct Pic {
    int64_t n, nx, ny;
    uint64_t key;
    const float* rows;
};
/// A request's cells: each one's value (a token, or kImageCell | hash(picture key, row)), its image row (null for a
/// token) and picture index (-1).
struct Cells {
    std::vector<uint64_t> cell;
    std::vector<const float*> rowp;
    std::vector<int> pic_of;
};
/// The cells of `ids` with the pictures `pics` at their runs of <|image_pad|> (only with `geni`), and `table` [ctx][3]
/// their rotary positions: a token at the next text position p; picture k's row j at (p, p + j / nx, p + j % nx), the
/// text after it from p + max(nx, ny) (mtmd's M-RoPE positions); the cells after the prompt continue the text.
/// -> "" or why the request is refused.
std::string build_cells(const Model& m, const std::vector<int32_t>& ids, const std::vector<Pic>& pics, bool geni,
                        int ctx, Cells& cl, std::vector<int32_t>& table) {
    const int64_t n = (int64_t) ids.size(), N = m.g.n_embd;
    const int pad = m.tok_image_pad;
    cl.cell.assign((size_t) n, 0);
    cl.rowp.assign((size_t) n, nullptr);
    cl.pic_of.assign((size_t) n, -1);
    table.assign((size_t) ctx * 3, 0);
    int64_t rp = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (!geni || ids[(size_t) i] != pad) {
            cl.cell[(size_t) i] = (uint64_t) (uint32_t) ids[(size_t) i];
            table[(size_t) i * 3] = table[(size_t) i * 3 + 1] = table[(size_t) i * 3 + 2] = (int32_t) rp;
            ++rp;
            ++i;
            continue;
        }
        if (k >= pics.size()) return "the prompt has more pictures than the embeddings file";
        const Pic& pc = pics[k];
        for (int64_t j = 0; j < pc.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != pad)
                return "image " + std::to_string(k + 1) + " has " + std::to_string((long long) pc.n) +
                       " rows but fewer image-pad tokens";
        for (int64_t j = 0; j < pc.n; ++j) {
            const size_t c = (size_t) (i + j);
            cl.cell[c] = kImageCell | (fnv_mix(fnv_mix(kFnvSeed, pc.key), (uint64_t) j) >> 1);
            cl.rowp[c] = pc.rows ? pc.rows + (size_t) j * (size_t) N : nullptr;
            cl.pic_of[c] = (int) k;
            table[c * 3] = (int32_t) rp;
            table[c * 3 + 1] = (int32_t) (rp + j / pc.nx);
            table[c * 3 + 2] = (int32_t) (rp + j % pc.nx);
        }
        i += pc.n;
        rp += std::max(pc.nx, pc.ny);
        ++k;
    }
    if (k != pics.size()) return "the embeddings file has more pictures than the prompt";
    if (n > 0 && cl.pic_of[(size_t) (n - 1)] >= 0) return "the prompt cannot end in an image";
    for (int64_t c = n; c < ctx; ++c)
        table[(size_t) c * 3] = table[(size_t) c * 3 + 1] = table[(size_t) c * 3 + 2] = (int32_t) (rp + (c - n));
    return "";
}

/// An embeddings file (strata-vision records) read whole: (n, nx, ny, has rows) per record, and the rows.
std::string read_embeddings(const std::string& path, int64_t N, std::vector<std::tuple<int64_t, int64_t, int64_t, bool>>& heads,
                            std::vector<std::vector<float>>& bodies) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    for (;;) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        const bool ref = hdr[0] == 0x31525653;   // 'SVR1': a picture by its key alone (no rows)
        if (got != 5 || (hdr[0] != 0x31455653 && !ref) || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) N) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) N) + ")";
            break;
        }
        bodies.emplace_back();
        if (!ref) {
            bodies.back().resize((size_t) hdr[1] * (size_t) N);
            if (std::fread(bodies.back().data(), sizeof(float), bodies.back().size(), f) != bodies.back().size()) {
                ve = "short embeddings file";
                break;
            }
        }
        heads.emplace_back(hdr[1], hdr[2], hdr[3], !ref);
    }
    std::fclose(f);
    return ve;
}

// A per-worker protocol channel keeps the original single-request state machine
// intact. In batch mode a router owns stdin/stdout and adapts its slot lines.
struct Q36Channel {
    std::function<bool(std::string&)> read;
    std::function<void(const std::string&)> write;
};
thread_local Q36Channel* q36_channel = nullptr;
int q36_printf(const char* format, ...) {
    va_list args; va_start(args, format);
    if (!q36_channel) { int n = std::vprintf(format, args); va_end(args); return n; }
    va_list copy; va_copy(copy, args);
    int n = std::vsnprintf(nullptr, 0, format, copy); va_end(copy);
    std::vector<char> out((size_t) std::max(0, n) + 1);
    if (n >= 0) std::vsnprintf(out.data(), out.size(), format, args);
    va_end(args);
    if (n >= 0) q36_channel->write(std::string(out.data(), (size_t) n));
    return n;
}
void q36_flush() { if (!q36_channel) std::fflush(stdout); }
// ------------------------------------------------------------------------------------------------ serve
// The engine protocol (serve/server.py): READY <ctx> stop; GEN / GENI requests; HAVE <key,...> -> HAVE <n:nx:ny | 0,...>
// (which pictures the engine holds, so the server encodes only the others); STOP.  The held conversation is reused at
// the longest common start of the request and `live` the engine can return to: all of it when the request extends it,
// else the latest state snapshot inside the common start (the attention layers' cells are kept as they are, the
// GDN state comes back from the snapshot).  Snapshots are taken at the request's last message starts and every
// `snap_every` cells near its end.
int serve(Engine& E, ImageCache& icache, int snap_every, size_t park_bytes, const std::string& model_path) {
    Q36Vocabulary vocabulary(model_path);
#ifdef STRATA_Q36_GRAMMAR
    q36_printf("INFO grammar=1\n");
#endif
    const auto& g = E.m->g;
    const int64_t N = g.n_embd;
    E.alloc_snaps();
    E.alloc_pages(park_bytes);
    q36_printf("INFO engine=q36 img_refs=1 snapshots=%d snapshot_every=%d snapshot_mib=%.0f image_cache_mib=%zu n_embd=%lld park_pages=%zu\n",
                E.max_snaps, snap_every, E.snap_bytes() / 1048576.0, icache.budget >> 20, (long long) N, E.host_page_count());
    q36_printf("READY %d stop\n", E.ctx);
    q36_flush();
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    E.cancel = &stop;
    const bool phase_trace = std::getenv("STRATA_Q36_PHASE_TRACE") != nullptr;
    Q36Channel* channel = q36_channel;
    std::thread reader([&, channel] {
        std::string l;
        while (channel ? channel->read(l) : static_cast<bool>(std::getline(std::cin, l))) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                if (!stop.exchange(true))
                    std::fprintf(stderr, "strata-q36: STOP at %.3f\n", wall_ms());   // (the cancel latency, see DONE)
                continue;
            }
            const bool gen = l.rfind("GEN ", 0) == 0 || l.rfind("GENI ", 0) == 0;
            const double arrived_at = gen && phase_trace ? wall_ms() : 0;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (gen) stop.store(false);
                lines.push_back(l);
            }
            cv.notify_one();
            // Diagnostics must not delay publishing a ready request. The captured
            // timestamp is its arrival; log line ordering can lag engine execution.
            if (gen && phase_trace) std::fprintf(stderr, "strata-q36: GEN arrived at %.3f\n", arrived_at);
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    struct ReaderJoin { std::thread& reader; ~ReaderJoin() { if (reader.joinable()) reader.join(); } } reader_join{reader};
    // ---- conversations: the active one (live, its cells in the cache) and the parked ones (their cells only in host
    // pages).  Pages are shared between conversations for a common start (reference counts; a page written while
    // shared is copied first).
    using Conv = strata::q36::cache::Conversation;
    std::vector<Conv> local_parked;
    auto& parked=E.cache_hub?E.cache_hub->parked:local_parked;
    strata::q36::cache::Hub::RecordLease active_record;
    std::vector<Engine::Snap*> act_states;
    auto release_states = [&](std::vector<Engine::Snap*>& states) {
        if(E.cache_hub) { E.cache_hub->release_states(states); return; }
        for (Engine::Snap* sn : states) { assert(sn->refs > 0); --sn->refs; }
        states.clear();
    };
    // At most twelve anchors per retained prefix: explicit points, recently reused
    // divergence/end states, then the newest automatic grid/message states. Without
    // explicit marks, at most four non-grid anchors leave eight 512-cell grid states:
    // a 4K suffix window, enough to discover a changed 1-3K tail inside ONE message.
    // This is a RAM admission budget, not a caller ID or a device slot. Shared states
    // count once. An explicit 128-state pool covers six such prefixes plus replacement;
    // smaller user-selected pools still evict whole LRU prefixes rather than corrupt state.
    // Leave at least half of a tiny ring unowned for ordinary replacement. A one-entry
    // ring has no owner quota: it is still usable as the original best-effort cache.
    const size_t state_limit = std::min<size_t>(12, E.snapshot_pool().size() / 2);
    auto retain_state = [&](std::vector<Engine::Snap*>& states, Engine::Snap& sn) {
        auto lock=E.host_lock();
        if(sn.writing) return;
        if (state_limit == 0 || std::find(states.begin(), states.end(), &sn) != states.end()) return;
        states.push_back(&sn);
        ++sn.refs;
        // Do not let old end states crowd out the unchanged message boundary.
        // Cap metadata anchors so they cannot consume the automatic suffix window.
        // Explicit points already name the client's boundary and may use four entries;
        // divergence/end and message boundaries use at most two each.
        for (int kind = 3; kind >= 1; --kind) {
            const size_t cap = kind == 3 ? 4 : 2;
            for (;;) {
                size_t count = 0;
                auto worst = states.end();
                for (auto it = states.begin(); it != states.end(); ++it)
                    if (std::min((*it)->prio, 3) == kind) {
                        ++count;
                        // Structural anchors keep the latest positions, not the
                        // most globally used states: a shared root touched by another
                        // caller must not replace this owner's first-image boundary.
                        if (worst == states.end() ||
                            (kind == 1 ? std::make_pair((*it)->pos, (*it)->used) < std::make_pair((*worst)->pos, (*worst)->used)
                                       : (*it)->used < (*worst)->used)) worst = it;
                    }
                if (count <= cap) break;
                assert((*worst)->refs > 0); --(*worst)->refs; states.erase(worst);
            }
        }
        while (states.size() > state_limit) {
            auto worst = std::min_element(states.begin(), states.end(), [](const Engine::Snap* a, const Engine::Snap* b) {
                return std::make_pair(a->prio, a->used) < std::make_pair(b->prio, b->used);
            });
            assert((*worst)->refs > 0);
            --(*worst)->refs;
            states.erase(worst);
        }
    };
    std::map<uint8_t*, int> local_page_refs;
    auto& page_refs=E.cache_hub?E.cache_hub->page_refs:local_page_refs;
    std::vector<uint8_t*> act_pages;
    int64_t act_mirrored = 0;
    uint64_t local_conv_clock = 0;
    auto& conv_clock=E.cache_hub?E.cache_hub->conv_clock:local_conv_clock;
    const size_t max_parked = 16;
    const bool cache_trace = std::getenv("STRATA_Q36_CACHE_TRACE") != nullptr;
    auto trace_states = [&](const char* owner, int index, size_t cells, const std::vector<Engine::Snap*>& states) {
        if (!cache_trace) return;
        auto lock=E.host_lock();
        std::string detail;
        for(const Engine::Snap* sn:states) detail+=std::to_string(sn->pos)+"/"+std::to_string(sn->prio)+"/"+std::to_string(sn->refs)+",";
        if(lock.owns_lock()) lock.unlock();
        std::fprintf(stderr,"strata-q36: cache owner=%s index=%d cells=%zu states=%s\n",owner,index,cells,detail.c_str());
    };
    long long switches = 0, cells_brought = 0;
    auto unref = [&](std::vector<uint8_t*>& pages) {
        if(E.cache_hub) { E.cache_hub->release_pages(pages); return; }
        for (uint8_t* pg : pages)
            if (--page_refs[pg] == 0) {
                page_refs.erase(pg);
                E.page_pool().push_back(pg);
            }
        pages.clear();
    };
    auto drop_lru_parked = [&]() {
        if(E.cache_hub) return E.cache_hub->evict_lru();
        if (parked.empty()) return false;
        size_t v = 0;
        for (size_t i = 1; i < parked.size(); ++i)
            if (parked[i].used < parked[v].used) v = i;
        unref(parked[v].pages);
        release_states(parked[v].states);
        parked.erase(parked.begin() + (long) v);
        return true;
    };
    E.evict_snapshot_owner = drop_lru_parked;
    E.snapshot_saved = [&](Engine::Snap& sn) { retain_state(act_states, sn); };
    struct SnapshotCallbacks {
        Engine& e;
        std::function<void()> release;
        ~SnapshotCallbacks() {
            e.evict_snapshot_owner = {}; e.snapshot_saved = {};
            release();
        }
    } snapshot_callbacks{E, [&] {
        if(E.cache_hub) { CK(cudaStreamSynchronize(E.st)); if(E.cst) CK(cudaStreamSynchronize(E.cst)); }
        auto lock=E.host_lock();
        release_states(act_states);
        if(E.cache_hub) { unref(act_pages); active_record.reset(); }
        else for (Conv& c : parked) release_states(c.states);
    }};
    auto take_page = [&]() -> uint8_t* {
        auto lock=E.host_lock();
        while (E.page_pool().empty())
            if (!drop_lru_parked()) return nullptr;
        if(E.cache_hub) return E.cache_hub->take_page();
        uint8_t* pg = E.page_pool().back();
        E.page_pool().pop_back();
        page_refs[pg] = 1;
        return pg;
    };
    std::vector<int32_t> live;       // the tokens whose state the engine holds (cell i = live[i])
    std::vector<uint64_t> live_v;    // the same cells' values (a token, or an image row: kImageCell | hash(key, row))
    int64_t live_prefill_end = 0;   // committed prompt boundary, excluding any generated answer rows
    bool live_cancelled = false;    // engine-thread state: a new GEN resets stop, but must not erase the old outcome
    // the active conversation's cells [act_mirrored, pos) into its pages (a shared page copied before it is written)
    auto mirror_active = [&]() {
        if(E.cache_hub) {
            if(E.host_page_count()==0) return;
            const int64_t want=E.pos;
            if(E.resident_prefix_reuse && E.resident_ram_equal_end>0) {
                const auto current=E.cache_hub->page_stamps(act_pages);
                for(size_t start=0;start<(size_t)E.resident_ram_equal_end;start+=Engine::kPage) {
                    const size_t pg=start/Engine::kPage;
                    if(pg>=current.size() || pg>=E.resident_ram_stamps.size() || !current[pg].valid() ||
                       !(current[pg]==E.resident_ram_stamps[pg])){E.resident_ram_equal_end=(int64_t)start;break;}
                }
            }
            for(int64_t pg=act_mirrored/Engine::kPage;pg<(int64_t)act_pages.size() && pg*Engine::kPage<want;++pg) {
                uint8_t* source=act_pages[(size_t)pg]; bool shared=false;
                { auto lock=E.host_lock(); shared=page_refs.at(source)>1; }
                if(!shared) continue;
                uint8_t* own=nullptr;
                for(;;) {
                    own=E.cache_hub->cow_page(source,[](uint8_t* dst,const uint8_t* src,size_t bytes) {
                        std::memcpy(dst,src,bytes); return true;
                    });
                    if(own || !drop_lru_parked()) break;
                }
                if(!own) return;
                { auto lock=E.host_lock();
                  act_pages[(size_t)pg]=own;
                  std::vector<uint8_t*> previous{source}; E.cache_hub->release_pages(previous);
                }
            }
            while((int64_t)act_pages.size()*Engine::kPage<want) {
                uint8_t* page=take_page(); if(!page) break; act_pages.push_back(page);
            }
            const int64_t end=std::min<int64_t>(want,(int64_t)act_pages.size()*Engine::kPage);
            if(end>act_mirrored) {
                const int64_t begin=act_mirrored;
                E.kv_copy(act_pages,begin,end,true);
                CK(cudaStreamSynchronize(E.st));
                E.finish_resident_host_writes();
                if(E.resident_prefix_reuse && begin<=E.resident_ram_equal_end)
                    E.resident_ram_equal_end=std::max(E.resident_ram_equal_end,end);
                act_mirrored=end;
            }
            if(E.resident_prefix_reuse)E.resident_ram_stamps=E.cache_hub->page_stamps(act_pages);
            return;
        }
        if (E.host_page_count() == 0) return;
        const int64_t upto_want = E.pos;
        for (int64_t pg = act_mirrored / Engine::kPage; pg < (int64_t) act_pages.size() && pg * Engine::kPage < upto_want; ++pg)
            if (page_refs[act_pages[(size_t) pg]] > 1) {   // shared with a parked conversation: this one's own copy
                uint8_t* own = take_page();
                // A selected conversation can own this page outside parked while being restored.
                // No private page means no writes: retain only the already mirrored prefix.
                if (!own) return;
                if (page_refs[act_pages[(size_t) pg]] == 1) {
                    page_refs.erase(own);
                    E.page_pool().push_back(own);
                    continue;
                }
                std::memcpy(own, act_pages[(size_t) pg], E.page_bytes());
                --page_refs[act_pages[(size_t) pg]];
                act_pages[(size_t) pg] = own;
            }
        while ((int64_t) act_pages.size() * Engine::kPage < upto_want) {
            uint8_t* pg = take_page();
            if (!pg) break;
            act_pages.push_back(pg);
        }
        const int64_t upto = std::min<int64_t>(upto_want, (int64_t) act_pages.size() * Engine::kPage);
        if (upto > act_mirrored) {
            E.kv_copy(act_pages, act_mirrored, upto, true);
            CK(cudaStreamSynchronize(E.st));
            act_mirrored = upto;
        }
    };
    auto prefix_hash = [](const std::vector<uint64_t>& v, size_t n) {
        uint64_t h = kFnvSeed;
        for (size_t i = 0; i < n; ++i) h = fnv_mix(h, v[i]);
        return h;
    };
    // Park the active prefix without changing its device state. A cancelled caller may
    // retain an already-owned, mirrored checkpoint instead of copying a new end state.
    // share=true retains the active handles; share=false transfers only the retained subset.
    auto park_active = [&](bool share) {
        if (phase_trace) std::fprintf(stderr, "strata-q36: park start at %.3f cells=%zu cancelled=%d share=%d\n",
                     wall_ms(), live_v.size(), (int) live_cancelled, (int) share);
        if (live_v.empty() || E.host_page_count() == 0) return;
        auto park_lock=E.host_lock();
        Engine::Snap* checkpoint = nullptr;
        std::vector<Engine::Snap*> checkpoint_states;
        if (live_cancelled) {
            const size_t limit = std::min({live.size(), live_v.size(),
                (size_t) std::max<int64_t>(0, act_mirrored), act_pages.size() * (size_t) Engine::kPage});
            std::vector<Engine::Snap*> candidates;
            for (Engine::Snap* sn : act_states)
                if (sn->refs > 0 && sn->pos > 0 && (size_t) sn->pos <= limit) candidates.push_back(sn);
            std::sort(candidates.begin(), candidates.end(), [](const Engine::Snap* a, const Engine::Snap* b) {
                return a->pos < b->pos;
            });
            // One prefix-hash pass for all owned candidates, not one full pass per state.
            size_t at = 0;
            uint64_t hash = kFnvSeed;
            for (Engine::Snap* sn : candidates) {
                while (at < (size_t) sn->pos) hash = fnv_mix(hash, live_v[at++]);
                if (sn->hash == hash) { checkpoint_states.push_back(sn); checkpoint = sn; }
            }
            // Early cold prefill can have only a root checkpoint. Do not throw away a
            // large completed history just to avoid its snapshot: use the honest old path.
            constexpr size_t max_trim_cells = 512;
            // The replay bound protects the committed PROMPT. A long generated tail
            // may be discarded after cancellation without forcing a new GPU snapshot.
            if (checkpoint && live_prefill_end > checkpoint->pos &&
                (uint64_t) (live_prefill_end - checkpoint->pos) > max_trim_cells) checkpoint = nullptr;
        }
        const bool checkpoint_only = checkpoint != nullptr;
        const char* mode = checkpoint_only ? "owned-checkpoint" : "end-snapshot";
        Conv c;
        if (checkpoint_only) {
            const size_t kept = (size_t) checkpoint->pos;
            const size_t pages = (kept + Engine::kPage - 1) / Engine::kPage;
            c.tok.assign(live.begin(), live.begin() + kept);
            c.val.assign(live_v.begin(), live_v.begin() + kept);
            c.pages.assign(act_pages.begin(), act_pages.begin() + pages);
            c.states = checkpoint_states;
            c.mirrored = (int64_t) kept;
            if (share) {
                for (uint8_t* pg : c.pages) ++page_refs[pg];
                for (Engine::Snap* sn : c.states) ++sn->refs;
            } else {
                // No ref increment for the transferred subset; release the omitted tails.
                std::vector<uint8_t*> tail_pages(act_pages.begin() + pages, act_pages.end());
                unref(tail_pages);
                for (Engine::Snap* sn : act_states)
                    if (std::find(c.states.begin(), c.states.end(), sn) == c.states.end()) {
                        assert(sn->refs > 0); --sn->refs;
                    }
                act_pages.clear(); act_states.clear(); act_mirrored = 0;
            }
            if (phase_trace) std::fprintf(stderr, "strata-q36: park mirror end at %.3f mode=%s skipped=1 kept=%zu trimmed=%zu\n",
                         wall_ms(), mode, kept, live_v.size() - kept);
            if (phase_trace) std::fprintf(stderr, "strata-q36: park snapshot end at %.3f mode=%s skipped=1\n", wall_ms(), mode);
        } else {
            if(park_lock.owns_lock()) park_lock.unlock();
            mirror_active();
            if (phase_trace) std::fprintf(stderr, "strata-q36: park mirror end at %.3f mode=%s mirrored=%lld\n",
                         wall_ms(), mode, (long long) act_mirrored);
            if (act_mirrored < (int64_t) live_v.size()) {
                std::fprintf(stderr, "strata-q36: park declined at %.3f reason=insufficient-pages\n", wall_ms());
                return;
            }
            E.save_snap(prefix_hash(live_v, live_v.size()), 2);
            if(E.cache_hub) { CK(cudaStreamSynchronize(E.st)); park_lock.lock(); }
            c.tok = live;
            c.val = live_v;
            c.pages = act_pages;
            c.mirrored = act_mirrored;
            if (share) {
                c.states = act_states;
                for (Engine::Snap* sn : c.states) ++sn->refs;
                for (uint8_t* pg : c.pages) ++page_refs[pg];
            } else {
                c.states = std::move(act_states);
                act_states.clear(); act_pages.clear(); act_mirrored = 0;
            }
        }
        if(E.cache_hub) {
            const size_t needed=((size_t)c.mirrored+Engine::kPage-1)/Engine::kPage;
            if(c.pages.size()>needed) {
                std::vector<uint8_t*> tails(c.pages.begin()+needed,c.pages.end());
                unref(tails);
                c.pages.resize(needed);
            }
        }
        c.origin_slot=E.batch_slot;
        if(!E.cache_hub) c.used = ++conv_clock;
        // A longer retained prefix contains every KV cell of its exact ancestors.
        // Transfer their useful anchors, then drop those redundant records. Divergent
        // branches remain separate, even when they share the same explicit breakpoint.
        for (size_t i = 0; i < parked.size();) {
            Conv& old = parked[i];
            if ((!E.cache_hub || old.leases==0) && old.val.size() <= c.val.size() && std::equal(old.val.begin(), old.val.end(), c.val.begin())) {
                for (Engine::Snap* sn : old.states) retain_state(c.states, *sn);
                unref(old.pages);
                release_states(old.states);
                parked.erase(parked.begin() + (long) i);
            } else ++i;
        }
        // Ancestor admission may replace a low-priority grid entry. Keep the selected
        // logical endpoint owned even then; total ownership remains within state_limit.
        if (checkpoint_only && std::find(c.states.begin(), c.states.end(), checkpoint) == c.states.end()) {
            assert(state_limit > 0);
            if (c.states.size() >= state_limit) {
                auto victim = std::min_element(c.states.begin(), c.states.end(), [](const Engine::Snap* a, const Engine::Snap* b) {
                    return std::make_pair(a->prio, a->used) < std::make_pair(b->prio, b->used);
                });
                assert((*victim)->refs > 0); --(*victim)->refs; c.states.erase(victim);
            }
            c.states.push_back(checkpoint); ++checkpoint->refs;
        }
        if(E.cache_hub) E.cache_hub->publish(std::move(c));
        else parked.push_back(std::move(c));
        while (parked.size() > max_parked) if(!drop_lru_parked()) break;
        const auto published_mirrored=parked.empty()?0:parked.back().mirrored;
        const auto published_cells=parked.empty()?0:parked.back().val.size();
        const auto published_index=(int)parked.size()-1;
        const auto published_states=parked.empty()?std::vector<Engine::Snap*>{}:parked.back().states;
        if(park_lock.owns_lock()) park_lock.unlock();
        if (!checkpoint_only && !E.cache_hub) {
            CK(cudaStreamSynchronize(E.st));   // observe the old end-snapshot completion before new KV restore
            if (phase_trace) std::fprintf(stderr, "strata-q36: park snapshot end at %.3f mode=%s observed=1\n", wall_ms(), mode);
        }
        // The checkpoint-only branch enqueues no GPU work. Cancel DONE already drained
        // the old stream; neither E.pos nor live tokens were changed by parking.
        std::fprintf(stderr, "strata-q36: parked at %.3f cells=%zu mode=%s retained=%lld\n",
                     wall_ms(), live_v.size(), mode, (long long) published_mirrored);
        trace_states("parked-new", published_index, published_cells, published_states);
    };
    std::vector<float> lg;
    std::vector<int32_t> table;      // [ctx][3] the rotary positions of the request's cells (and the cells after them)
    const int pad = E.m->tok_image_pad;
    auto parse_hex = [](const std::string& h, uint64_t& out) {
        if (h.empty() || h.size() > 16) return false;
        uint64_t v = 0;
        for (char c : h) {
            const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0) return false;
            v = v * 16 + (uint64_t) d;
        }
        out = v;
        return true;
    };
    auto split = [](const std::string& v, char sep) {
        std::vector<std::string> out;
        size_t a = 0;
        while (a <= v.size()) {
            const size_t b = v.find(sep, a);
            out.push_back(v.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        return out;
    };
    for (;;) {
        const double phase_loop = phase_trace ? wall_ms() : 0; // prior iteration locals have been destroyed
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        const double phase_dequeue = phase_trace ? wall_ms() : 0;
        if (line.empty()) continue;
        if (line.rfind("HAVE", 0) == 0) {
            std::lock_guard<std::recursive_mutex> cache_lock(icache.mutex);   // HAVE <hex key>,... -> HAVE n:nx:ny or 0 per key, in order
            std::string out = "HAVE";
            const std::string rest = line.size() > 5 ? line.substr(5) : "";
            bool first = true;
            for (const std::string& k : split(rest, ',')) {
                if (k.empty()) continue;
                uint64_t key = 0;
                const ImageCache::Entry* e = parse_hex(k, key) ? icache.find(key) : nullptr;
                out += first ? " " : ",";
                first = false;
                out += e ? std::to_string((long long) e->n) + ":" + std::to_string((long long) e->nx) + ":" +
                               std::to_string((long long) e->ny)
                         : std::string("0");
            }
            q36_printf("%s\n", out.c_str());
            q36_flush();
            continue;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            q36_printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] "
                        "<file> <id,id,...>\n");
            q36_flush();
            continue;
        }
        if (E.batch_begin) E.batch_begin();
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
        Sampling smp;
        std::string grammar_path;
        int grammar_after = -1;
        bool grammar_bad = false;
        std::vector<uint64_t> img_keys;
        std::vector<int64_t> snap_at;   // snap=<cell>,...: the client's snapshot points (at most 4)
        bool keys_bad = false;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "snap") {
                for (const std::string& c : split(v, ','))
                    if (!c.empty() && snap_at.size() < 4) snap_at.push_back(std::atoll(c.c_str()));
            }
            else if (key == "grammar") {
                grammar_path = v;
                if (grammar_path.empty()) grammar_bad = true;
            } else if (key == "grammar_after") {
                char* end = nullptr;
                const long id = std::strtol(v, &end, 10);
                if (end == v || *end || id < 0 || id >= g.n_vocab) grammar_bad = true;
                else grammar_after = (int) id;
            }
            else if (key == "img_keys") {
                for (const std::string& h : split(v, ',')) {
                    uint64_t x = 0;
                    if (!parse_hex(h, x)) { keys_bad = true; break; }
                    img_keys.push_back(x);
                }
            }
        }
        std::string emb_path;
        if (geni) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            emb_path.assign(st0, (size_t) (p - st0));
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if(E.batch_parsed) E.batch_parsed(max_new,ids.size());
        auto fail = [&](const std::string& why) {
            q36_printf("ERR %s\n", why.c_str());
            q36_flush();
            if (E.batch_idle) E.batch_idle();
        };
        if (grammar_bad) { fail("grammar: invalid grammar path or grammar_after token id"); continue; }
        Q36Grammar grammar;
        try { vocabulary.prepare(grammar, grammar_path, grammar_after); }
        catch (const std::exception& e) { fail(std::string("grammar: ") + e.what()); continue; }
        if (max_new < 1 || ids.empty()) { fail("bad request: max_new or the ids"); continue; }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            fail("prompt (" + std::to_string(ids.size()) + " tokens) + max_new (" + std::to_string(max_new) +
                 ") exceeds the context (" + std::to_string(E.ctx) + ")");
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) g.n_vocab;
        if (bad) { fail("a token id is outside the vocabulary"); continue; }
        if (E.batch_admit && !E.batch_admit(std::min<size_t>(E.ctx, ids.size() + (size_t) max_new + E.mtp_d + 1))) {
            if (stop.load()) {
                q36_printf("DONE 0 %zu 0 0 cancel 0 0 0\n", ids.size());
                E.batch_idle();
            } else fail("BGEN: request exceeds physical KV budget");
            continue;
        }
        ImageCache::Pins image_pins(icache);
        for (auto key : img_keys) image_pins.add(key);
        const double phase_parse = phase_trace ? wall_ms() : 0;
        const int64_t n = (int64_t) ids.size();
        // ---- the pictures: the file's records (SVE1 rows, SVR1 a key the engine holds), each to its run of pads
        std::vector<Pic> pics;
        std::string ve;
        if (keys_bad) ve = "img_keys: hex keys separated by commas";
        if (geni && ve.empty()) {
            std::vector<std::tuple<int64_t, int64_t, int64_t, bool>> heads;   // n, nx, ny, has rows
            std::vector<std::vector<float>> bodies;
            ve = read_embeddings(emb_path, N, heads, bodies);
            if (ve.empty() && !img_keys.empty() && img_keys.size() != heads.size())
                ve = "img_keys names " + std::to_string(img_keys.size()) + " pictures, the file has " + std::to_string(heads.size());
            std::vector<uint64_t> keep;
            for (size_t k = 0; ve.empty() && k < heads.size(); ++k) {
                const auto [pn, nx, ny, has] = heads[k];
                uint64_t key = 0;
                if (!img_keys.empty()) key = img_keys[k];
                else if (has) {   // no key from the server: the grid and the rows' bits
                    key = fnv_mix(fnv_mix(fnv_mix(kFnvSeed, (uint64_t) pn), (uint64_t) nx), (uint64_t) ny);
                    const uint8_t* b = (const uint8_t*) bodies[k].data();
                    for (size_t i = 0; i < bodies[k].size() * 4; ++i) key = (key ^ b[i]) * 1099511628211ull;
                } else {
                    ve = "a picture by reference (SVR1) needs its key in img_keys";
                    break;
                }
                image_pins.add(key);
                keep.push_back(key);
                const float* rows = nullptr;
                if (has) {
                    const ImageCache::Entry* e = icache.put(key, bodies[k].data(), pn, nx, ny, keep);
                    if (!e) { ve = "no host memory for a picture's rows"; break; }
                    rows = e->rows;
                } else if (const ImageCache::Entry* e = icache.find(key)) {
                    if (e->n != pn || e->nx != nx || e->ny != ny) { ve = "a picture's reference does not match its grid"; break; }
                    rows = e->rows;
                }
                pics.push_back({pn, nx, ny, key, rows});
            }
        }
        // ---- the cells: their values, image rows and rotary positions
        Cells cl;
        if (ve.empty()) ve = build_cells(*E.m, ids, pics, geni, E.ctx, cl, table);
        if (!ve.empty()) { fail(ve); continue; }
        const double phase_cells = phase_trace ? wall_ms() : 0;
        const std::vector<uint64_t>& cell = cl.cell;
        std::vector<const float*>& rowp = cl.rowp;
        const std::vector<int>& pic_of = cl.pic_of;
        std::vector<uint64_t> ph((size_t) n + 1);   // ph[i]: the hash of cells [0, i)
        ph[0] = kFnvSeed;
        for (int64_t i = 0; i < n; ++i) ph[(size_t) i + 1] = fnv_mix(ph[(size_t) i], cell[(size_t) i]);
        auto selection_lock=E.host_lock();
        // Refresh personal breakpoints even when reuse selects a deeper periodic state.
        // A cache-control marker newly added to a held state can promote it without recompute.
        for (Engine::Snap& sn : E.snapshot_pool())
            if (!sn.writing && sn.pos > 0 && sn.pos < n && sn.hash == ph[(size_t) sn.pos]) {
                const bool explicit_now = std::find(snap_at.begin(), snap_at.end(), sn.pos) != snap_at.end();
                if (explicit_now || sn.prio >= 3) E.touch_snap(sn, explicit_now ? 3 : sn.prio);
            }
        // ---- what the engine can keep: the longest common start with the active conversation and with every parked
        // one, then the state that reaches furthest into it (the request extending the active one, else a snapshot)
        auto common = [&](const std::vector<uint64_t>& v, size_t len) {
            size_t k = 0;
            while (k < len && k < (size_t) n && v[k] == cell[k]) ++k;
            return k;
        };
        auto best_snap = [&](int64_t limit) {
            Engine::Snap* f = nullptr;
            for (Engine::Snap& sn : E.snapshot_pool())
                if (!sn.writing && sn.pos > 0 && sn.pos <= limit && sn.pos < n && sn.hash == ph[(size_t) sn.pos] &&
                    (f == nullptr || sn.pos > f->pos))
                    f = &sn;
            return f;
        };
        size_t L = common(live_v, live_v.size());
        size_t held_n = live_v.size();   // what the selected source of this request held before it
        // Learning a changed suffix is independent of finding a reusable state. A parked
        // caller can match far beyond its last surviving snapshot (or tie the active root).
        // Keep that real LCP even when the source chosen for restoration does not change.
        size_t max_lcp = L, max_lcp_held = held_n;
        int64_t reuse = 0;
        Engine::Snap* from = nullptr;
        int src = -1;                    // -1: the active conversation, else a parked one
        if (L == live_v.size() && L < (size_t) n && L == (size_t) E.pos) {
            reuse = (int64_t) L;         // the request extends the held state
        } else {
            from = best_snap((int64_t) L);
            reuse = from ? from->pos : 0;
        }
        for (size_t i = 0; i < parked.size(); ++i) {
            const size_t Lc = common(parked[i].val, (size_t) parked[i].mirrored);
            if (Lc > max_lcp) {
                max_lcp = Lc;
                max_lcp_held = parked[i].val.size();
            } else if (Lc == max_lcp) {
                max_lcp_held = std::max(max_lcp_held, parked[i].val.size());
            }
            if ((int64_t) Lc <= reuse) continue;
            Engine::Snap* f = best_snap((int64_t) Lc);
            if (f && f->pos > reuse) {
                src = (int) i;
                from = f;
                reuse = f->pos;
                L = Lc;
                held_n = parked[i].val.size();
            }
        }
        const double phase_match = phase_trace ? wall_ms() : 0;
        if (cache_trace && !E.cache_hub) {
            std::fprintf(stderr, "strata-q36: cache match cells=%lld reuse=%lld src=%d selected_lcp=%zu max_lcp=%zu max_held=%zu\n",
                         (long long) n, (long long) reuse, src, L, max_lcp, max_lcp_held);
            trace_states("active", -1, live_v.size(), act_states);
            for (size_t i = 0; i < parked.size(); ++i)
                trace_states("parked", (int) i, parked[i].val.size(), parked[i].states);
        }
        {   // the pictures that must be read again need their rows
            std::string need;
            int last = -1;
            for (int64_t c = reuse; c < n; ++c)
                if (pic_of[(size_t) c] >= 0 && rowp[(size_t) c] == nullptr && pic_of[(size_t) c] != last) {
                    last = pic_of[(size_t) c];
                    need += (need.empty() ? "" : ",") + std::to_string(last + 1);
                }
            if (!need.empty()) {
                if(selection_lock.owns_lock()) selection_lock.unlock();
                fail("NEED_IMAGES " + need + " (the engine does not hold these pictures: send their rows)"); continue;
            }
        }
        // One diagnostic line per accepted GEN, immediately before parking (or
        // restoration if no park is needed). No token contents or paths are logged.
        auto trace_phase = [&](bool will_park) {
            if (phase_trace)
                std::fprintf(stderr, "strata-q36: phases loop_top=%.3f dequeue_after=%.3f parse_validate_after=%.3f build_cells_after=%.3f match_after=%.3f park_before=%.3f park=%d cells=%lld\n",
                             phase_loop, phase_dequeue, phase_parse, phase_cells, phase_match, wall_ms(),
                             (int) will_park, (long long) n);
        };
        int64_t brought = 0;
        size_t resident_skipped=0;
        std::vector<strata::q36::resident::PageStamp> resident_selected_stamps;
        strata::q36::cache::Hub::SnapPin selected_pin;
        uint64_t restored_record_id=0;
        int restored_origin_slot=-1;
        if(E.cache_hub) {
            if(!live.empty() || !act_pages.empty() || !act_states.empty()) {
                std::fprintf(stderr,"strata-q36: shared admission inherited an unfinished active lease\n"); std::abort();
            }
            if(src>=0) {
                auto checkout=E.cache_hub->checkout((size_t)src);
                if(!checkout || !from) std::abort();
                selected_pin=strata::q36::cache::Hub::SnapPin(*E.cache_hub,*from);
                restored_record_id=checkout.lease.id();
                Conv copy=checkout.release_conv();
                restored_origin_slot=copy.origin_slot;
                active_record=std::move(checkout.lease);
                act_states=std::move(copy.states); live=std::move(copy.tok); live_v=std::move(copy.val);
                act_pages=std::move(copy.pages); act_mirrored=copy.mirrored;
                brought=reuse; ++switches; cells_brought+=brought;
            } else if(from || reuse) std::abort();
            selection_lock.unlock();
            if(src>=0) {
#ifndef __HIP_PLATFORM_AMD__
                if(E.resident_prefix_reuse) {
                    resident_selected_stamps=E.cache_hub->page_stamps(act_pages);
                    resident_skipped=E.resident_prefix.reusable(E.resident_identity(),cell,(size_t)reuse,
                        E.kv_arena->committed_tokens(),resident_selected_stamps,Engine::kPage);
                }
#endif
                if(E.resident_prefix_verify && resident_skipped && !E.verify_resident_prefix(act_pages,resident_skipped)) {
                    E.resident_prefix.invalidate();resident_skipped=0;
                }
                E.resident_before_write((int64_t)resident_skipped);
                if((int64_t)resident_skipped<reuse)E.kv_copy(act_pages,(int64_t)resident_skipped,reuse,false);
            }
            if(from) E.restore_snap(*from); else E.reset();
            if(E.resident_prefix_reuse) {
                // The skip was certified against THESE immutable pages; the
                // remaining range has now completed H2D at restore_snap's fence.
                E.resident_ram_equal_end=reuse;
                E.resident_ram_stamps=resident_selected_stamps;
                E.resident_prefix.publish(E.resident_identity(),cell,(size_t)reuse,resident_selected_stamps,Engine::kPage);
            }
            selected_pin.reset();
            if(src>=0 && std::getenv("STRATA_Q36_BATCH_TRACE")) {
                auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"cache_restore\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"record_id\":%llu,\"origin_slot\":%d,\"cached_cells\":%lld}\n",(long long)ns,E.batch_slot,E.batch_epoch,(unsigned long long)restored_record_id,restored_origin_slot,(long long)reuse);
            }
        } else {
        E.pinned_snap = from;
        if (src >= 0) {
            // another conversation: the active one parked (whole), the chosen one's cells beyond the common start of
            // the two brought from its pages, its state from the snapshot
            const size_t same = [&] {   // the cells the cache already holds for it: the two conversations' common start
                size_t k = 0;
                const Conv& c = parked[(size_t) src];
                while (k < live_v.size() && k < c.val.size() && live_v[k] == c.val[k]) ++k;
                return k;
            }();
            from->used = ++E.snapshot_clock();   // (the parked one's end snapshot must not take the slot it is restored from)
            Conv c = std::move(parked[(size_t) src]);
            parked.erase(parked.begin() + src);
            trace_phase(true);
            park_active(false);
            unref(act_pages);
            release_states(act_states);   // park can fail when its KV could not be mirrored
            act_states = std::move(c.states);
            live = std::move(c.tok);
            live_v = std::move(c.val);
            act_pages = std::move(c.pages);
            act_mirrored = c.mirrored;
            const int64_t lo = std::min<int64_t>((int64_t) same, reuse);
            if (reuse > lo) E.kv_copy(act_pages, lo, reuse, false);
            brought = reuse - lo;
            ++switches;
            cells_brought += brought;
        } else if ((int64_t) held_n - reuse >= 4096) {
            if (from) from->used = ++E.snapshot_clock();
            trace_phase(true);
            park_active(true);   // a long conversation the request leaves early: kept for its caller's next turn
        } else {
            trace_phase(false);
        }
        if (reuse == 0) {
            E.reset();
        } else if (from != nullptr) {
            E.restore_snap(*from);
        }
        }
        E.pinned_snap = nullptr;
        live_cancelled = false;   // the previous outcome was used by park_active; this request now owns the state
        live.resize((size_t) reuse);
        live_v.resize((size_t) reuse);
        act_mirrored = std::min<int64_t>(act_mirrored, reuse);
        // Release anchors outside the new prefix. The hashes describe the complete
        // incoming prompt: matching anchors ahead of reuse remain valid while we redo
        // that unchanged part. An abort prunes them to the committed position below.
        {
        auto metadata_lock=E.host_lock();
        for (auto it = act_states.begin(); it != act_states.end();) {
            Engine::Snap* sn = *it;
            if (sn->pos <= 0 || sn->pos > n || sn->hash != ph[(size_t) sn->pos]) {
                assert(sn->refs > 0); --sn->refs; it = act_states.erase(it);
            } else ++it;
        }
        // Admit useful states already in the shared pool (including a newly marked
        // held prefix), not only states created during this particular request.
        for (Engine::Snap& sn : E.snapshot_pool())
            if (!sn.writing && sn.pos > 0 && sn.pos <= n && sn.hash == ph[(size_t) sn.pos]) retain_state(act_states, sn);
        }
        if (E.m->has_mrope)
            CK(cudaMemcpyAsync(E.mrope + (size_t) reuse * 3, table.data() + (size_t) reuse * 3,
                               ((size_t) E.ctx - (size_t) reuse) * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, E.st));
        // the snapshots this prompt leaves: its last message starts and, near its end, every snap_every cells
        std::vector<Engine::Cut> cuts;
        {
            std::vector<std::pair<int64_t, int>> at;   // position, priority
            int bounds = 0;
            for (int64_t c = n - 1; c > reuse && bounds < 3; --c)
                if (ids[(size_t) c] == E.m->tok_im_start && pic_of[(size_t) c] < 0) { at.push_back({c, 1}); ++bounds; }
            // Preserve the text prefix of a multimodal message independently of its
            // photo memory. A removed/changed image can invalidate the whole suffix
            // window while every text cell before the first image is unchanged.
            // pic_of marks actual embedding rows (including cached references).
            // Look back eight cells, as for learned divergence, to precede image
            // delimiters and tokenization changes when text is appended before photos.
            for (int64_t c = 0; c < n; ++c)
                if (pic_of[(size_t) c] >= 0) {
                    const int64_t anchor = c > 8 ? c - 8 : c;
                    if (anchor > reuse) at.push_back({anchor, 1});
                    break;
                }
            if (snap_every > 0) {
                const int64_t window = (int64_t) std::max(0, E.max_snaps - 4) * snap_every;
                for (int64_t c = (reuse / snap_every + 1) * snap_every; c < n; c += snap_every)
                    if (c >= n - window) at.push_back({c, 0});
            }
            // where this request left the held one: a client that keeps a long start and replaces its tail every time
            // (a replaceable state message) leaves the next request at the same place - a snapshot a few cells before it
            // (the tokenizer can merge the first cells of a new tail differently) makes that one read its tail alone
            if ((int64_t) max_lcp > reuse + 8 && (int64_t) max_lcp < n && max_lcp < max_lcp_held)
                at.push_back({(int64_t) max_lcp - 8, 2});
            // the end of the last message's body (before its <|im_end|>): a client that grows that message (and sends
            // the same start again) continues from there
            for (int64_t c = n - 1; c > reuse; --c)
                if (ids[(size_t) c] == E.m->tok_im_end && pic_of[(size_t) c] < 0) { at.push_back({c, 0}); break; }
            for (int64_t c : snap_at)   // the client's own points (cache_control / prompt_cache_breakpoint)
                if (c > reuse && c < n) at.push_back({c, 3});
            std::sort(at.begin(), at.end());
            for (const auto& [c, pr] : at)   // one cut a position, the highest priority asked for it
                if (!cuts.empty() && cuts.back().pos == c) cuts.back().prio = std::max(cuts.back().prio, pr);
                else cuts.push_back({c, ph[(size_t) c], pr});
        }
        const auto t0 = Clock::now();
        int64_t fresh = n - reuse;
        q36_printf("RESUME %lld\n", (long long) reuse);
        q36_flush();
        std::fprintf(stderr, "strata-q36: prompt from cell %lld of %lld at %.3f\n", (long long) reuse, (long long) n, wall_ms());
        E.resident_before_write(reuse);
        E.process(ids.data() + reuse, fresh, [&](int64_t done) {
            // Each callback publishes only a fully committed chunk. Incremental host mirroring
            // prevents a later cancellation/switch from inheriting an entire prompt's KV copy.
            // mirror_active uses E.pos, not live: the vectors are appended once below process().
            mirror_active();
            q36_printf("PP %lld %lld %.0f %.1f\n", (long long) (reuse + done), (long long) n, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            q36_flush();
        }, geni ? rowp.data() + reuse : nullptr, &cuts);
        live_prefill_end = E.pos;   // committed full/partial prompt, before decoding any answer rows
        const bool cut_short = E.pos < n;   // STOP while the prompt was read: what it read stays held
        live.insert(live.end(), ids.begin() + (long) reuse, ids.begin() + (long) E.pos);
        live_v.insert(live_v.end(), cell.begin() + (long) reuse, cell.begin() + (long) E.pos);
        if (cut_short) {
            fresh = E.pos - reuse;
            // Transactional abort already restored a consistent live state. A host snapshot here
            // would enqueue another 63 MiB copy after rollback; park_active saves it only if needed.
            if (E.pos > 0 && !E.prefill_aborted) E.save_snap(ph[(size_t) E.pos], 2);
        }
        { auto metadata_lock=E.host_lock();
        for (auto it = act_states.begin(); it != act_states.end();) {
            if ((*it)->pos > E.pos) {
                assert((*it)->refs > 0); --(*it)->refs; it = act_states.erase(it);
            } else ++it;
        }
        }
        const double prompt_ms = ms_since(t0);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // with the MTP block, speculative steps of mtp_d drafts: a greedy request keeps the drafts the greedy picks
        // confirm (the text is the greedy text); a sampled one keeps draft d of a row with the probability the row's
        // sampling distribution p gives it and otherwise draws from p without d - exactly the distribution of drawing
        // every token in turn (the drafts are the block's greedy picks, so their own distribution is a point mass).
        // STRATA_Q36_SPEC_SAMPLED=0: sampled requests one token at a time, as before.
        static const bool spec_sampled = [] {
            const char* e = std::getenv("STRATA_Q36_SPEC_SAMPLED");
            return e == nullptr || std::strcmp(e, "0") != 0;
        }();
        const bool sampled = smp.needs_logits();
        const bool constrained = grammar.active();
        const bool spec = E.mtp_d > 0 && (!sampled || spec_sampled);
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        std::vector<float> cv;
        std::vector<int> ci, all_ids;
        std::vector<std::pair<int, double>> dist;
        long long grammar_fast = 0, grammar_full = 0;
        const bool greedy_unpenalized = smp.temperature <= 0.f &&
            (smp.last_n == 0 || (smp.repeat == 1.f && smp.freq == 0.f && smp.present == 0.f));
        auto allowed_by = [&](const Q36Grammar& gr, int id) {
            if (id < 0 || id >= g.n_vocab) return false;
            float value = 0.f;
            gr.mask(&value, &id, 1);
            return std::isfinite(value);
        };
        // row r of `logits` -> dist under history h: on its candidates when that is exact, else on the whole row
        auto row_dist = [&](int r, const std::vector<int32_t>& h, const Q36Grammar* gr) {
            if (cand_dist(cv.data() + (size_t) r * kTopK, ci.data() + (size_t) r * kTopK, kTopK, false, smp, h, dist, gr)) {
                if (constrained) ++grammar_fast;
                return;
            }
            if (constrained) ++grammar_full;
            std::vector<float> row((size_t) g.n_vocab);
            CK(cudaMemcpy(row.data(), E.logits + (size_t) r * g.n_vocab, row.size() * 4, cudaMemcpyDeviceToHost));
            if (all_ids.empty()) {
                all_ids.resize((size_t) g.n_vocab);
                for (size_t i = 0; i < all_ids.size(); ++i) all_ids[i] = (int) i;
            }
            cand_dist(row.data(), all_ids.data(), (int) g.n_vocab, true, smp, h, dist, gr);
        };
        auto next_sampled = [&]() {   // the token after the last processed row (logits row 0)
            if (constrained && greedy_unpenalized) {
                const int raw = E.greedy();
                if (allowed_by(grammar, raw)) {
                    ++grammar_fast;
                    dist.assign(1, {raw, 1.0});
                    return draw(dist, rng);
                }
            }
            E.topk_rows(1, cv, ci);
            row_dist(0, hist, &grammar);
            return draw(dist, rng);
        };
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const std::function<int(int, const std::vector<int32_t>&, int32_t*)> judge =
            [&](int T, const std::vector<int32_t>& dr, int32_t* pk) {
                bool have_topk = false;
                Q36Grammar window(grammar);
                std::vector<int32_t> h = hist;   // the emitted tokens (the window's first row, x, among them)
                for (int i = 0; i < T; ++i) {
                    // spec_step has already copied every row's raw argmax to
                    // hpin[2+i]. If it is allowed and no penalties apply, it is
                    // also the exact grammar-first greedy choice (including ties).
                    const int raw = E.hpin[2 + i];
                    if (constrained && greedy_unpenalized && allowed_by(window, raw)) {
                        ++grammar_fast;
                        dist.assign(1, {raw, 1.0});
                    } else {
                        if (!have_topk) { E.topk_rows(T, cv, ci); have_topk = true; }
                        row_dist(i, h, &window);
                    }
                    if (i == T - 1) { pk[i] = draw(dist, rng); return i; }
                    double pd = 0;
                    for (const auto& [id, q] : dist)
                        if (id == dr[(size_t) i]) pd = q;
                    if (smp.temperature <= 0.f ? pd == 1.0 : unit(rng) < pd) {
                        pk[i] = dr[(size_t) i];
                        h.push_back(dr[(size_t) i]); window.accept(dr[(size_t) i]); continue;
                    }
                    pk[i] = draw(dist, rng, dr[(size_t) i]);
                    return i;
                }
                return T - 1;
            };
        // The draft state starts AFTER nx, which emit() has already accepted.
        std::unique_ptr<Q36Grammar> draft_state;
        const std::function<int(int)> draft_pick = [&](int raw) {
            // The global argmax is also the best allowed top-128 candidate
            // whenever it is allowed. Avoid three extra device top-k passes
            // per window in the common case, without changing the proposal.
            if (allowed_by(*draft_state, raw)) {
                draft_state->accept(raw);
                return raw;
            }
            std::vector<float> values;
            std::vector<int> ids_m;
            E.topk_rows(1, values, ids_m, E.logits_m);
            const int fallback = ids_m.front();
            draft_state->mask(values.data(), ids_m.data(), (int) values.size());
            int chosen = fallback;
            for (size_t i = 0; i < values.size(); ++i)
                if (std::isfinite(values[i])) { chosen = ids_m[i]; break; }
            // A disallowed fallback will be rejected by the verifier. Do not feed
            // it to llama.cpp (which asserts on an impossible grammar transition).
            float allowed = 0.f;
            draft_state->mask(&allowed, &chosen, 1);
            if (std::isfinite(allowed)) draft_state->accept(chosen);
            return chosen;
        };
        const bool canceled = cut_short || stop.load();
        int nx = canceled ? -1 : sampled || constrained ? next_sampled() : E.greedy();
        if(E.resident_prefix_reuse && !canceled && !E.prefill_aborted) {
            const size_t valid=(size_t)std::min({E.pos,E.resident_teacher_end,act_mirrored,E.resident_ram_equal_end});
            E.resident_prefix.publish(E.resident_identity(),cell,valid,E.resident_ram_stamps,Engine::kPage);
        }
        if(E.resident_prefix_reuse && std::getenv("STRATA_Q36_BATCH_TRACE")) {
            const auto stamp=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
            std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"resident_prefix\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"mapping_generation\":%llu,\"reuse_cells\":%lld,\"skipped_cells\":%zu,\"copied_cells\":%lld,\"skipped_bytes\":%zu,\"copied_bytes\":%zu,\"valid_end\":%zu}\n",
                (long long)stamp,E.batch_slot,E.batch_epoch,(unsigned long long)E.resident_identity().generation,(long long)reuse,resident_skipped,(long long)reuse-(long long)resident_skipped,
                resident_skipped*(E.page_bytes()/Engine::kPage),((size_t)reuse-resident_skipped)*(E.page_bytes()/Engine::kPage),E.resident_prefix.valid_end());
        }
        bool done = canceled;
        if (canceled) finish = "cancel";
        auto keep = [&](int32_t t) { live.push_back(t); live_v.push_back((uint64_t) (uint32_t) t); };
        double cancel_returned = 0;
        if (E.cancel_trace) {
            E.cancel_abort_at = 0;
            E.cancel_abort_stage = "none";
            E.cancel_abort_pos = -1;
        }
        while (!done) {
            if (stop.load()) { finish = "cancel"; break; }
            auto emit = [&](int t) {   // false: the answer ends here
                q36_printf("T %d\n", t);
                q36_flush();
                ++produced;
                hist.push_back(t);
                grammar.accept(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (spec && E.pos + E.mtp_d + 1 <= E.ctx) {
                if (constrained) draft_state = std::make_unique<Q36Grammar>(grammar);
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks,
                                          sampled || constrained ? &judge : nullptr, constrained ? &draft_pick : nullptr);
                if (a < 0) {
                    if (E.cancel_trace) cancel_returned = wall_ms();
                    finish = "cancel"; break;   // no main-state commit, no live append or picks[-1]
                }
                offered += E.mtp_d;
                accepted += a;
                keep(nx);   // the window's kept rows are all in the state, whatever the answer keeps of them
                for (int i = 0; i < a; ++i) keep(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                for (int i = 0; i < a; ++i)
                    if (!emit(drafts[(size_t) i])) { done = true; break; }
                nx = picks[a];
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            keep(nx);
            fix.clear();
            nx = sampled || constrained ? next_sampled() : E.greedy();
        }
        const bool trace_cancel = E.cancel_trace && std::strcmp(finish, "cancel") == 0;
        const double cancel_sync_before = trace_cancel ? wall_ms() : 0;
        double cancel_st_after = 0, cancel_cst_after = 0;
        if (std::strcmp(finish, "cancel") == 0) {
            CK(cudaStreamSynchronize(E.st));   // a STOP after commit must not hide its queued GPU writes after DONE
            if (trace_cancel) cancel_st_after = wall_ms();
            if (E.cst) CK(cudaStreamSynchronize(E.cst));
            if (trace_cancel) cancel_cst_after = wall_ms();
        }
        live_cancelled = std::strcmp(finish, "cancel") == 0;
        const double decode_ms = ms_since(d0);
        if (constrained) std::fprintf(stderr, "strata-q36: grammar target rows: fast=%lld full=%lld\n", grammar_fast, grammar_full);
        const double cancel_done_before = trace_cancel ? wall_ms() : 0;
        q36_printf("DONE %lld %zu %.1f %.1f %s %lld %lld %lld 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, (long long) reuse, (long long) fresh);
        const double cancel_printf_after = trace_cancel ? wall_ms() : 0;
        q36_flush();
        const double cancel_flush_after = trace_cancel ? wall_ms() : 0;
        const double cancel_log_before = trace_cancel ? wall_ms() : 0;
        if (std::strcmp(finish, "cancel") == 0) std::fprintf(stderr, "strata-q36: cancelled at %.3f\n", wall_ms());
        const double cancel_log_after = trace_cancel ? wall_ms() : 0;
        if (trace_cancel)
            std::fprintf(stderr, "strata-q36: cancel trace abort_at=%.3f abort_stage=%s abort_pos=%lld caller_return=%.3f sync_before=%.3f st_after=%.3f cst_after=%.3f done_before=%.3f printf_after=%.3f flush_after=%.3f cancelled_log_before=%.3f cancelled_log_after=%.3f\n",
                         E.cancel_abort_at, E.cancel_abort_stage, (long long) E.cancel_abort_pos, cancel_returned, cancel_sync_before,
                         cancel_st_after, cancel_cst_after, cancel_done_before, cancel_printf_after,
                         cancel_flush_after, cancel_log_before, cancel_log_after);
        // Cancelled live state stays on the device. A switch prefers its existing owned
        // checkpoint; the fallback can still copy, so measure park completion separately.
        if (!live_cancelled) mirror_active();
        size_t parked_count=0, free_page_count=0;
        { auto metadata_lock=E.host_lock(); parked_count=parked.size(); free_page_count=E.page_pool().size(); }
        std::fprintf(stderr, "strata-q36: request %lld cells: reused %lld (%s%s), read %lld in %.1f ms, %zu pictures, %zu "
                     "snapshots taken; %lld tokens in %.1f ms (%s); conversations: %zu parked, %zu of %zu pages free\n",
                     (long long) n, (long long) reuse, reuse == 0 ? "fresh" : from ? "snapshot" : "held state",
                     src >= 0 ? (", another conversation: " + std::to_string((long long) brought) + " cells brought back").c_str() : "",
                     (long long) fresh, prompt_ms, pics.size(), cuts.size(), produced, decode_ms, finish, parked_count,
                     free_page_count, E.host_page_count());
        if (E.batch_idle) {
            park_active(false);
            {
                auto metadata_lock=E.host_lock();
                unref(act_pages); release_states(act_states);
                active_record.reset(); act_mirrored=0;
                live.clear(); live_v.clear();
            }
            E.reset();
            E.batch_idle();
        }
    }
}

#include "q36_parallel.hpp"

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, embeddings;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_at = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    int bench_reps = 3, batch = 1, batch_kv_mib = 6144, batch_prefill_chunk = 0;
    double expert_vram = -1.0;
    bool warm = true;
    int expert_slots = 0;
    int snapshots = 0, snap_every = 512, image_cache_mib = 2048, park_mib = 0;
    bool serve_mode = false, graph = true, resident_prefix_reuse = false, resident_prefix_verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--batch") batch = std::atoi(next().c_str());
        else if (a == "--batch-kv-mib") batch_kv_mib = std::atoi(next().c_str());
        else if (a == "--resident-prefix-reuse") resident_prefix_reuse = true;
        else if (a == "--resident-prefix-verify") resident_prefix_verify = true;
        else if (a == "--batch-prefill-chunk") batch_prefill_chunk = std::atoi(next().c_str());
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--embeddings") embeddings = next();   // --tokens with pictures: strata-vision records (GENI's file)
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-pp-at") bench_at = std::atoi(next().c_str());
        else if (a == "--bench-reps") bench_reps = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--snapshots") snapshots = std::atoi(next().c_str());              // state snapshots kept (0: off)
        else if (a == "--snapshot-every") snap_every = std::atoi(next().c_str());       // cells between snapshots
        else if (a == "--image-cache-mib") image_cache_mib = std::atoi(next().c_str()); // pictures held by key
        else if (a == "--park-mib") park_mib = std::atoi(next().c_str());   // conversations' cells in host memory (0: off)
        else if (a == "--vision") {}   // the serve path always takes GENI (the pictures' rows come from strata-vision)
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N [--bench-pp-at P0])\n"
                         "       --tokens @FILE reads comma/space separated token IDs from a file\n");
            return 2;
        }
    }
    if(resident_prefix_verify && !resident_prefix_reuse) {
        std::fprintf(stderr,"strata-q36: --resident-prefix-verify requires --resident-prefix-reuse\n");return 2;
    }
    if(resident_prefix_reuse && (!serve_mode || batch<2 || batch>4)) {
        std::fprintf(stderr,"strata-q36: --resident-prefix-reuse needs --serve --batch 2..4\n");return 2;
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_Q8_PACKED", "1", 0);   // the packed Q8_0 decode copy unless STRATA_Q8_PACKED is set otherwise
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    if (batch < 1 || batch > 4 || batch_kv_mib < 1 || batch_prefill_chunk < 0 ||
        (batch_prefill_chunk > 0 && (batch_prefill_chunk < Engine::kVerify || batch_prefill_chunk > 2048)) ||
        (batch > 1 && !serve_mode)) {
        std::fprintf(stderr, "strata-q36: --batch must be 1..4 and needs --serve; --batch-kv-mib must be positive; --batch-prefill-chunk must be 0(auto) or 8..2048\n"); return 2;
    }
#ifndef __HIP_PLATFORM_AMD__
    std::unique_ptr<strata::q36::kv::Budget> batch_budget;
    std::unique_ptr<strata::q36::kv::CudaDriver> batch_driver;
    if (batch > 1) {
        int device = 0; CK(cudaGetDevice(&device));
        batch_budget = std::make_unique<strata::q36::kv::Budget>((size_t) batch_kv_mib << 20);
        batch_driver = std::make_unique<strata::q36::kv::CudaDriver>(device);
    }
#else
    if (batch > 1) { std::fprintf(stderr, "strata-q36: native batch VMM currently requires CUDA\n"); return 2; }
#endif
    if(batch>1 && !m.packed.empty()) {
        CK(cudaDeviceSynchronize()); size_t returned=0;
        for(const auto& packed:m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(packed));
            returned+=std::get<2>(packed); CK(cudaFree(std::get<1>(packed)));
        }
        m.packed.clear();
        std::fprintf(stderr,"strata-q36: batch physical KV budget uses native Q8_0 layout; packed copies returned %.0fMiB (MTP unchanged)\n",returned/1048576.0);
    }
    Engine E;
    E.batch_coordinator=batch>1;
    E.resident_prefix_reuse=resident_prefix_reuse;
    E.resident_prefix_verify=resident_prefix_verify;
    if(batch>1) E.shared_prefill=std::make_shared<Q36SharedPrefill>();
#ifndef __HIP_PLATFORM_AMD__
    E.kv_budget = batch_budget.get(); E.kv_driver = batch_driver.get();
#endif
    E.use_graph = batch > 1 ? false : graph;
    if (chunk == 0) {
        // auto: 512 with every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (with %.2f packed), main experts %.2f GiB\n",
                         fr / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0);
    }
    if (batch > 1 && chunk > 128) chunk = 128;
    // Full-context PP followed by TG needs a small cache tail for the requested generated tokens.
    if (bench_at < 0 || bench_tg < 0 || bench_reps < 1 || (int64_t) bench_at + bench_pp > ctx || (int64_t) ctx + bench_tg > INT32_MAX) {
        std::fprintf(stderr, "strata-q36: benchmark prompt exceeds --max-context\n"); return 2;
    }
    E.init(m, ctx + ((bench_pp > 0 || bench_tg > 0) ? bench_tg : 0), batch > 1 ? 32 : chunk);
    if (serve_mode && batch == 1) E.alloc_prefill_rollback();   // include the rollback reservation in expert placement
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    E.max_snaps = std::max(0, snapshots);
    if (warm && batch == 1) E.warmup();
    if (serve_mode) {
        ImageCache icache;
        icache.parallel = batch>1;
        icache.width = m.g.n_embd;
        icache.budget = (size_t) std::max(64, image_cache_mib) << 20;
#ifndef __HIP_PLATFORM_AMD__
        if (batch > 1) return serve_parallel(E, icache, model, batch, snap_every,
            batch_prefill_chunk > 0 ? batch_prefill_chunk : (batch == 2 ? 256 : 64),
            ((size_t) std::max(0, park_mib) << 20),
            std::max(0, snapshots), *batch_budget, *batch_driver, warm);
#endif
        return serve(E, icache, snap_every, (size_t) std::max(0, park_mib) << 20, model);
    }
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        std::vector<int32_t> history((size_t) bench_at);
        for (auto& t : history) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < bench_reps; ++rep) {
            E.reset();
            if (!history.empty()) {
                E.process(history.data(), (int64_t) history.size(), [](int64_t) {});
                CK(cudaStreamSynchronize(E.st));
            }
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    if (!tokens.empty() && tokens[0] == '@') {
        const std::string path = tokens.substr(1);
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::perror(path.c_str()); return 2; }
        std::string contents;
        char buf[8192];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f))) contents.append(buf, n);
        const bool ok = !std::ferror(f);
        std::fclose(f);
        if (!ok) { std::fprintf(stderr, "strata-q36: cannot read token file %s\n", path.c_str()); return 2; }
        for (char& ch : contents) if (std::isspace((unsigned char) ch)) ch = ' ';
        tokens = std::move(contents);
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    std::vector<std::tuple<int64_t, int64_t, int64_t, bool>> heads;
    std::vector<std::vector<float>> bodies;
    Cells cl;
    if (!embeddings.empty()) {   // the pictures at their pads, their rows and M-RoPE positions as in the serve path
        std::string ve = read_embeddings(embeddings, m.g.n_embd, heads, bodies);
        std::vector<Pic> pics;
        for (size_t k = 0; ve.empty() && k < heads.size(); ++k) {
            if (!std::get<3>(heads[k])) ve = "a reference record (SVR1) in --embeddings";
            pics.push_back({std::get<0>(heads[k]), std::get<1>(heads[k]), std::get<2>(heads[k]), (uint64_t) k + 1,
                            bodies[k].data()});
        }
        std::vector<int32_t> table;
        if (ve.empty()) ve = build_cells(m, prompt, pics, true, E.ctx, cl, table);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 1; }
        CK(cudaMemcpy(E.mrope, table.data(), table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        std::fprintf(stderr, "strata-q36: %zu pictures in the prompt\n", pics.size());
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {}, embeddings.empty() ? nullptr : cl.rowp.data());
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            if (a < 0) break;
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
