// src/kernels/cuda/dense_kernels.cu - see include/strata/kernels/dense_kernels.hpp.
#include "strata/kernels/dense_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdlib>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kWarps = 8;            // warps per attention block: 8 cells in flight
constexpr int kChunk = 256;          // cells per attention block (one split)
constexpr int kMaxHeadDim = 256;     // the merge maps one thread to one output dim

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------------ RMSNorm
// One block per row; the row is read twice (sum of squares, then scale), which for 5120 floats stays in L1.
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                int cols, float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* orow = out + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += xr[i] * xr[i];
    acc = warp_sum(acc);
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
        scale = rsqrtf(s / (float) cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * scale * w[i];
}

// One block per value head: rms over `cols`, then * w[i] * silu(z[i]).
__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ w, float* __restrict__ y, int cols, float eps) {
    const float* orow = o + (size_t) blockIdx.x * cols;
    const float* zrow = z + (size_t) blockIdx.x * cols;
    float* yrow = y + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += orow[i] * orow[i];
    acc = warp_sum(acc);
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
        scale = rsqrtf(s / (float) cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float g = zrow[i];
        yrow[i] = orow[i] * scale * w[i] * (g / (1.0f + __expf(-g)));
    }
}

// One block of 1024 threads: pass 1 finds the largest logit (lowest index on a tie), pass 2 sums exp(l - max).
__global__ void argmax_prob_kernel(const float* __restrict__ logits, int n, int* __restrict__ out_id,
                                   float* __restrict__ out_p) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float best = -FLT_MAX;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }    // NaN compares false: never picked
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    const float gmax = sv[0];
    const int gid = si[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = logits[i];
        if (!isnan(v)) sum += __expf(v - gmax);
    }
    sv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const bool ok = gid != INT_MAX && sv[0] > 0.0f;
        *out_id = ok ? gid : 0;
        *out_p = ok ? 1.0f / sv[0] : 0.0f;
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o,
                              int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    o[i] = (a / (1.0f + __expf(-a))) * u[i];
}

// One warp per output row.
__global__ void gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32) acc += wr[i] * x[i];
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void fill_i32_kernel(int32_t* dst, int n, int32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

// ------------------------------------------------------------------------------------------------ KV cache
// One head's one cell is head_dim values in one of three layouts (see DenseKvFormat).  Q8 and Q4 are ggml's Q8_0 and
// Q4_0 blocks of 32 values, so lane l of a warp that owns the dims lane + 32 * i finds dim i's block at block i.
struct Q8Blk { __half d; int8_t q[32]; };      // x ~ q * d
struct Q4Blk { __half d; uint8_t q[16]; };     // x ~ (nibble - 8) * d; byte j holds value j (low) and j + 16 (high)
static_assert(sizeof(Q8Blk) == 34 && sizeof(Q4Blk) == 18, "KV block layout");

constexpr int kF16 = DENSE_KV_F16, kQ8 = DENSE_KV_Q8, kQ4 = DENSE_KV_Q4;

__host__ __device__ constexpr int cell_bytes(int fmt, int head_dim) {
    return fmt == kF16 ? head_dim * 2 : fmt == kQ8 ? head_dim / 32 * 34 : head_dim / 32 * 18;
}

// In-place orthonormal Walsh-Hadamard transform of 256 floats in shared memory, by the 256 threads of the block.
// It is symmetric and its own inverse: applying it twice gives the input back.
__device__ __forceinline__ void fwht256_shared(float* buf) {
    const int t = threadIdx.x;
    for (int len = 1; len < 256; len <<= 1) {
        __syncthreads();
        const float a = buf[t], b = buf[t ^ len];
        __syncthreads();
        buf[t] = (t & len) ? (b - a) : (a + b);
    }
    __syncthreads();
    buf[t] *= 0.0625f;                      // 1 / sqrt(256)
    __syncthreads();
}

// Stores buf[0..head_dim) as one cell.  One warp per block of 32 values; blockDim.x == head_dim.
template <int F>
__device__ __forceinline__ void kv_store_cell(uint8_t* cell, const float* buf) {
    const int t = threadIdx.x;
    if (F == kF16) { ((__half*) cell)[t] = __float2half(buf[t]); return; }
    const int warp = t >> 5, lane = t & 31;
    const float x = buf[t];
    if (F == kQ8) {
        float amax = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        Q8Blk* b = (Q8Blk*) cell + warp;
        b->q[lane] = (int8_t) rintf(x * id);
        if (lane == 0) b->d = __float2half(d);
    } else {
        // ggml's Q4_0 rule: d = (the value of largest magnitude, with its sign) / -8; q = min(15, trunc(x / d + 8.5)).
        // The pair (|x|, lane) is a total order, so every lane of the butterfly ends on the same winner.
        float mx = x, ax = fabsf(x);
        int idx = lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, mx, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            const float oa = fabsf(om);
            if (oa > ax || (oa == ax && oi < idx)) { mx = om; ax = oa; idx = oi; }
        }
        const float d = mx / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const int q = min(15, (int) (x * id + 8.5f));
        const int qh = __shfl_down_sync(0xffffffffu, q, 16);
        Q4Blk* b = (Q4Blk*) cell + warp;
        if (lane < 16) b->q[lane] = (uint8_t) (q | (qh << 4));
        if (lane == 0) b->d = __float2half(d);
    }
}

// The value of dim (lane + 32 * i) of a cell.
template <int F>
__device__ __forceinline__ float kv_val(const uint8_t* cell, int i, int lane) {
    if (F == kF16) return __half2float(((const __half*) cell)[lane + 32 * i]);
    if (F == kQ8) {
        const Q8Blk* b = (const Q8Blk*) cell + i;
        return (float) b->q[lane] * __half2float(b->d);
    }
    const Q4Blk* b = (const Q4Blk*) cell + i;
    const int by = b->q[lane & 15];
    return (float) ((lane < 16 ? (by & 15) : (by >> 4)) - 8) * __half2float(b->d);
}

template <int KF, int VF>
__global__ void kv_append_kernel(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, const float* __restrict__ k,
                                 const float* __restrict__ v, int pos, int head_dim, int max_ctx) {
    __shared__ float buf[kMaxHeadDim];
    const int h = blockIdx.x, t = threadIdx.x;
    const size_t cell = (size_t) h * max_ctx + pos;
    buf[t] = k[(size_t) h * head_dim + t];
    __syncthreads();
    if (KF == kQ4) fwht256_shared(buf);
    kv_store_cell<KF>(kc + cell * cell_bytes(KF, head_dim), buf);
    __syncthreads();
    buf[t] = v[(size_t) h * head_dim + t];
    __syncthreads();
    if (VF == kQ4) fwht256_shared(buf);
    kv_store_cell<VF>(vc + cell * cell_bytes(VF, head_dim), buf);
}

// ------------------------------------------------------------------------------------------------ attention
// Grid (n_head, n_splits), 256 threads = 8 warps.  A warp owns a cell at a time: its 32 lanes hold head_dim/32
// dims each (strided by 32 so the loads coalesce), the dot product is a warp reduction, and the warp's
// running (m, l, acc) is the online softmax over the cells it visited (cells t = begin + warp, +8, ...).
// With Q4 keys q is rotated (Walsh-Hadamard) first; the scores q . k are the same in the rotated space.
template <int KF, int VF>
__global__ void attn_partial_kernel(const float* __restrict__ q, const uint8_t* __restrict__ K,
                                    const uint8_t* __restrict__ V, float* __restrict__ part_acc,
                                    float* __restrict__ part_ml, int n_head, int n_kv, int head_dim, int n_ctx,
                                    int max_ctx, float scale, int n_splits) {
    const int h = blockIdx.x, split = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;                       // dims per lane, <= 8

    __shared__ float sq[kMaxHeadDim];
    float qv[8];
    if (KF == kQ4) {                                      // head_dim == 256: one thread per dim
        sq[threadIdx.x] = q[(size_t) h * head_dim + threadIdx.x];
        __syncthreads();
        fwht256_shared(sq);
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = i < per ? sq[lane + 32 * i] * scale : 0.0f;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = i < per ? q[(size_t) h * head_dim + lane + 32 * i] * scale : 0.0f;
    }

    const size_t kcell = (size_t) cell_bytes(KF, head_dim), vcell = (size_t) cell_bytes(VF, head_dim);
    const uint8_t* Kb = K + (size_t) kvh * max_ctx * kcell;
    const uint8_t* Vb = V + (size_t) kvh * max_ctx * vcell;
    const int begin = split * kChunk;
    const int end = min(n_ctx, begin + kChunk);

    float m = -FLT_MAX, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int t = begin + warp; t < end; t += kWarps) {
        const uint8_t* kcl = Kb + (size_t) t * kcell;
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) d += qv[i] * kv_val<KF>(kcl, i, lane);
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn);               // m == -FLT_MAX on the first cell: exp(-huge) = 0
        const float p = __expf(d - mn);
        l = l * corr + p;
        const uint8_t* vcl = Vb + (size_t) t * vcell;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * kv_val<VF>(vcl, i, lane);
        m = mn;
    }

    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < per) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    if (threadIdx.x < head_dim) {
        float gm = -FLT_MAX;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[w]);
        float a = 0.0f, L = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            if (sl[w] == 0.0f) continue;                 // this warp saw no cell of the chunk
            const float f = __expf(sm[w] - gm);
            a += sacc[w][threadIdx.x] * f;
            L += sl[w] * f;
        }
        part_acc[((size_t) h * n_splits + split) * head_dim + threadIdx.x] = a;
        if (threadIdx.x == 0) {
            part_ml[((size_t) h * n_splits + split) * 2] = L > 0.0f ? gm : -FLT_MAX;
            part_ml[((size_t) h * n_splits + split) * 2 + 1] = L;
        }
    }
}

// Grid n_head, head_dim threads: merges the chunks of one head.  With Q4 values the sum is in the rotated space:
// the rotation (its own inverse) takes it back.
template <int VF>
__global__ void attn_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                  float* __restrict__ out, int head_dim, int n_splits) {
    __shared__ float so[kMaxHeadDim];
    const int h = blockIdx.x;
    float gm = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, part_ml[((size_t) h * n_splits + s) * 2]);
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float l = part_ml[((size_t) h * n_splits + s) * 2 + 1];
        if (l == 0.0f) continue;
        const float f = __expf(part_ml[((size_t) h * n_splits + s) * 2] - gm);
        a += part_acc[((size_t) h * n_splits + s) * head_dim + threadIdx.x] * f;
        L += l * f;
    }
    float val = L > 0.0f ? a / L : 0.0f;
    if (VF == kQ4) {
        so[threadIdx.x] = val;
        __syncthreads();
        fwht256_shared(so);
        val = so[threadIdx.x];
    }
    out[(size_t) h * head_dim + threadIdx.x] = val;
}

int splits_for(int cells) { return (cells + kChunk - 1) / kChunk; }

// ================================================================================== the batched prompt kernels
// The same arithmetic as the per-token path above, with the token as the slow index, so a chunk of T tokens lands on
// the same logits as feeding those tokens through step() one at a time.

__global__ void split_q_kernel(const float* __restrict__ q_full, float* __restrict__ q, int head_dim) {
    const int row = blockIdx.x;                       // one row per (token, query head)
    const float* src = q_full + (size_t) row * 2 * head_dim;
    float* dst = q + (size_t) row * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) dst[i] = src[i];
}

// out[i] = attn[i] * sigmoid(gate) with the gate the second half of the head's [q | gate] block.
__global__ void gate_apply_kernel(const float* __restrict__ attn, const float* __restrict__ q_full,
                                  float* __restrict__ out, int head_dim, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / head_dim, d = i - row * head_dim;
    const float g = q_full[row * 2 * head_dim + head_dim + d];
    out[i] = attn[i] * (1.0f / (1.0f + __expf(-g)));
}

// The grid rounds rows up to a whole number of blocks, so the tail threads have to be turned off: a short final
// prompt chunk (or a decode pass whose n * n_head is not a multiple of the block size) would otherwise write past the
// end of the position buffer and into whatever allocation follows it.
__global__ void positions_kernel(int32_t* dst, int rows, int heads, int32_t pos0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    dst[i] = pos0 + (int32_t) (i / heads);
}

// grid (n_kv, T): one block per (KV head, token) writes that token's cell of that head's cache.  blockDim.x ==
// head_dim, the same requirement kv_append_kernel has: kv_store_cell wants one thread per value and fwht256_shared
// wants the 256 threads of a 256-wide head.
template <int KF, int VF>
__global__ void kv_append_rows_kernel(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, const float* __restrict__ k,
                                      const float* __restrict__ v, int pos0, int n_kv, int head_dim, int max_ctx) {
    __shared__ float buf[kMaxHeadDim];
    const int h = blockIdx.x, t = blockIdx.y;
    const size_t cell = (size_t) h * max_ctx + pos0 + t;
    const float* kh = k + ((size_t) t * n_kv + h) * head_dim;
    const float* vh = v + ((size_t) t * n_kv + h) * head_dim;
    buf[threadIdx.x] = kh[threadIdx.x];
    __syncthreads();
    if (KF == kQ4) fwht256_shared(buf);
    kv_store_cell<KF>(kc + cell * cell_bytes(KF, head_dim), buf);
    __syncthreads();
    buf[threadIdx.x] = vh[threadIdx.x];
    __syncthreads();
    if (VF == kQ4) fwht256_shared(buf);
    kv_store_cell<VF>(vc + cell * cell_bytes(VF, head_dim), buf);
}

// One block per (query head, token): the decode kernel's warp-per-cell online softmax, restricted to the causal
// range [0, pos0 + t].  No split-K: the (head, token) grid already gives n_head * T blocks.  The KV formats are the
// same the decode path reads, so a chunk and a token at a time land on the same cache.
template <int KF, int VF>
__global__ void attn_chunk_kernel(const float* __restrict__ q, const uint8_t* __restrict__ K,
                                  const uint8_t* __restrict__ V, float* __restrict__ out, int n_head, int n_kv,
                                  int head_dim, int pos0, int max_ctx, float scale) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;

    __shared__ float sq[kMaxHeadDim];
    float qv[8];
    if (KF == kQ4) {                                      // head_dim == 256: one thread per dim
        sq[threadIdx.x] = q[((size_t) t * n_head + h) * head_dim + threadIdx.x];
        __syncthreads();
        fwht256_shared(sq);
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = i < per ? sq[lane + 32 * i] * scale : 0.0f;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qv[i] = i < per ? q[((size_t) t * n_head + h) * head_dim + lane + 32 * i] * scale : 0.0f;
    }

    const size_t kcell = (size_t) cell_bytes(KF, head_dim), vcell = (size_t) cell_bytes(VF, head_dim);
    const uint8_t* Kb = K + (size_t) kvh * max_ctx * kcell;
    const uint8_t* Vb = V + (size_t) kvh * max_ctx * vcell;
    const int end = pos0 + t + 1;

    float m = -FLT_MAX, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int c = warp; c < end; c += kWarps) {
        const uint8_t* kcl = Kb + (size_t) c * kcell;
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) d += qv[i] * kv_val<KF>(kcl, i, lane);
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn);
        const float p = __expf(d - mn);
        l = l * corr + p;
        const uint8_t* vcl = Vb + (size_t) c * vcell;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * kv_val<VF>(vcl, i, lane);
        m = mn;
    }

    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    __shared__ float so[kMaxHeadDim];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < per) sacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    // A Q4 value cache means head_dim == 256, so every thread of the block is inside this branch and the rotation's
    // __syncthreads() stay aligned.
    if (threadIdx.x < head_dim) {
        float gm = -FLT_MAX;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[w]);
        float a = 0.0f, L = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            if (sl[w] == 0.0f) continue;
            const float f = __expf(sm[w] - gm);
            a += sacc[w][threadIdx.x] * f;
            L += sl[w] * f;
        }
        float val = L > 0.0f ? a / L : 0.0f;
        if (VF == kQ4) {                                  // back from the rotated space
            so[threadIdx.x] = val;
            __syncthreads();
            fwht256_shared(so);
            val = so[threadIdx.x];
        }
        out[((size_t) t * n_head + h) * head_dim + threadIdx.x] = val;
    }
}
// Same arithmetic as attn_chunk_kernel, but one block per (KV head, token) serving the G query heads that share that
// KV head: every K/V cell is loaded and decompressed once instead of G times.  Per head the operations and their
// order are the ones of attn_chunk_kernel, so the result is bit-identical.
template <int KF, int VF, int G>
__global__ void attn_chunk_g_kernel(const float* __restrict__ q, const uint8_t* __restrict__ K,
                                    const uint8_t* __restrict__ V, float* __restrict__ out, int n_head, int n_kv,
                                    int head_dim, int pos0, int max_ctx, float scale) {
    const int kvh = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per = head_dim >> 5;

    __shared__ float sq[kMaxHeadDim];
    float qv[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qp = q + ((size_t) t * n_head + kvh * G + g) * head_dim;
        if (KF == kQ4) {                                  // head_dim == 256: one thread per dim
            sq[threadIdx.x] = qp[threadIdx.x];
            __syncthreads();
            fwht256_shared(sq);
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[g][i] = i < per ? sq[lane + 32 * i] * scale : 0.0f;
            __syncthreads();                              // sq is rewritten by the next head
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[g][i] = i < per ? qp[lane + 32 * i] * scale : 0.0f;
        }
    }

    const size_t kcell = (size_t) cell_bytes(KF, head_dim), vcell = (size_t) cell_bytes(VF, head_dim);
    const uint8_t* Kb = K + (size_t) kvh * max_ctx * kcell;
    const uint8_t* Vb = V + (size_t) kvh * max_ctx * vcell;
    const int end = pos0 + t + 1;

    float m[G], l[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -FLT_MAX;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[g][i] = 0.0f;
    }

    for (int c = warp; c < end; c += kWarps) {
        const uint8_t* kcl = Kb + (size_t) c * kcell;
        float kk[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) kk[i] = i < per ? kv_val<KF>(kcl, i, lane) : 0.0f;
        float pg[G], cg[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (i < per) d += qv[g][i] * kk[i];
            d = warp_sum(d);
            const float mn = fmaxf(m[g], d);
            cg[g] = __expf(m[g] - mn);
            pg[g] = __expf(d - mn);
            l[g] = l[g] * cg[g] + pg[g];
            m[g] = mn;
        }
        const uint8_t* vcl = Vb + (size_t) c * vcell;
        float vv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) vv[i] = i < per ? kv_val<VF>(vcl, i, lane) : 0.0f;
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (i < per) acc[g][i] = acc[g][i] * cg[g] + pg[g] * vv[i];
    }

    // The cross-warp merge, one head at a time so the 8 x 256 float buffer is reused (G of them would not fit in 48 KiB).
    __shared__ float sm[G][kWarps], sl[G][kWarps];
    __shared__ float sacc[kWarps][kMaxHeadDim];
    __shared__ float so[kMaxHeadDim];
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) { sm[g][warp] = m[g]; sl[g][warp] = l[g]; }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) sacc[warp][lane + 32 * i] = acc[g][i];
        __syncthreads();
        if (threadIdx.x < head_dim) {
            float gm = -FLT_MAX;
            for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, sm[g][w]);
            float a = 0.0f, L = 0.0f;
            for (int w = 0; w < kWarps; ++w) {
                if (sl[g][w] == 0.0f) continue;
                const float f = __expf(sm[g][w] - gm);
                a += sacc[w][threadIdx.x] * f;
                L += sl[g][w] * f;
            }
            float val = L > 0.0f ? a / L : 0.0f;
            if (VF == kQ4) {                              // back from the rotated space
                so[threadIdx.x] = val;
                __syncthreads();
                fwht256_shared(so);
                val = so[threadIdx.x];
            }
            out[((size_t) t * n_head + kvh * G + g) * head_dim + threadIdx.x] = val;
        }
        __syncthreads();                                  // sacc is rewritten by the next head
    }
}
// ---- tensor-core chunk attention: block = (KV head, tile of 8 tokens) x the 6 query heads that share the KV head =
// 48 rows.  K/V cells are dequantized to half in shared memory once per block and multiplied with WMMA.
// Two passes over the cells: pass 1 finds each row's max, pass 2 recomputes S, forms P = exp(S - max) and accumulates
// P V in registers with no rescaling (WMMA does not expose which row an accumulator element belongs to).
// head_dim == 256, 256 threads, G == 6.  Dynamic shared memory: 54144 bytes.
constexpr int kTcSmem = 54144;

template <int F>
__device__ __forceinline__ void tc_load_tile(__half* dst, const uint8_t* Base, size_t cellb, int c0, int cmax) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;       // thread == dim (lane + 32 * warp)
    for (int c = 0; c < 32; ++c) {
        float v = 0.0f;
        if (c0 + c < cmax) v = kv_val<F>(Base + (size_t) (c0 + c) * cellb, warp, lane);
        dst[c * 264 + threadIdx.x] = __float2half(v);
    }
}

template <int KF, int VF>
__global__ void __launch_bounds__(256) attn_chunk_tc_kernel(const float* __restrict__ q, const uint8_t* __restrict__ K,
                                                            const uint8_t* __restrict__ V, float* __restrict__ out,
                                                            int T, int n_head, int pos0, int max_ctx, float scale) {
    using namespace nvcuda;
    constexpr int D = 256, G = 6, BQ = 8, R = 48, BK = 32, LD = 264, SL = 40;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __half* Qs = (__half*) smem_raw;                       // [48][264]
    __half* KVs = (__half*) (smem_raw + 25344);            // [32][264]  (reused as Os float [16][264] at the end)
    float* Os = (float*) KVs;
    float* Ss = (float*) (smem_raw + 42240);               // [48][40]
    __half* Ps = (__half*) (smem_raw + 49920);             // [48][40]
    float* ms = (float*) (smem_raw + 53760);
    float* ls = ms + R;

    const int kvh = blockIdx.x, t0 = blockIdx.y * BQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ntok = min(BQ, T - t0);
    const size_t kcell = (size_t) cell_bytes(KF, D), vcell = (size_t) cell_bytes(VF, D);
    const uint8_t* Kb = K + (size_t) kvh * max_ctx * kcell;
    const uint8_t* Vb = V + (size_t) kvh * max_ctx * vcell;

    // ---- Q (scaled, rotated for Q4 keys) -> half, row r = token * 6 + head-in-group
    for (int r = 0; r < R; ++r) {
        const int tl = r / G, g = r % G;
        float v = 0.0f;
        if (tl < ntok) v = q[((size_t) (t0 + tl) * n_head + kvh * G + g) * D + tid];
        if (KF == kQ4) {
            Ss[tid] = v;
            __syncthreads();
            fwht256_shared(Ss);
            v = Ss[tid];
            __syncthreads();
        }
        Qs[r * LD + tid] = __float2half(v * scale);
    }
    if (tid < R) { ms[tid] = -FLT_MAX; ls[tid] = 0.0f; }
    __syncthreads();

    const int cmax = pos0 + t0 + ntok;                          // cells [0, cmax) can be seen by some row of the tile
    const int ntiles = (cmax + BK - 1) / BK;

    auto compute_S = [&]() {
        if (warp < 6) {
            const int rt = warp >> 1, ct = warp & 1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0f);
            for (int k = 0; k < 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, Qs + rt * 16 * LD + k * 16, LD);
                wmma::load_matrix_sync(b, KVs + ct * 16 * LD + k * 16, LD);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(Ss + rt * 16 * SL + ct * 16, c, SL, wmma::mem_row_major);
        }
    };

    // ---- pass 1: row maxima
    for (int tile = 0; tile < ntiles; ++tile) {
        const int c0 = tile * BK;
        tc_load_tile<KF>(KVs, Kb, kcell, c0, cmax);
        __syncthreads();
        compute_S();
        __syncthreads();
        for (int j = 0; j < 6; ++j) {
            const int r = warp * 6 + j, tl = r / G;
            const int lim = tl < ntok ? pos0 + t0 + tl + 1 : 0;
            float s = (c0 + lane < lim) ? Ss[r * SL + lane] : -FLT_MAX;
            for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
            if (lane == 0) ms[r] = fmaxf(ms[r], s);
        }
    }
    __syncthreads();

    // ---- pass 2: P = exp(S - max), l = sum P, O += P V
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[3][2];
#pragma unroll
    for (int rt = 0; rt < 3; ++rt)
#pragma unroll
        for (int i = 0; i < 2; ++i) wmma::fill_fragment(oacc[rt][i], 0.0f);

    for (int tile = 0; tile < ntiles; ++tile) {
        const int c0 = tile * BK;
        tc_load_tile<KF>(KVs, Kb, kcell, c0, cmax);
        __syncthreads();
        compute_S();
        __syncthreads();
        for (int j = 0; j < 6; ++j) {
            const int r = warp * 6 + j, tl = r / G;
            const int lim = tl < ntok ? pos0 + t0 + tl + 1 : 0;
            const float p = (c0 + lane < lim) ? __expf(Ss[r * SL + lane] - ms[r]) : 0.0f;
            Ps[r * SL + lane] = __float2half(p);
            const float ps = warp_sum(p);
            if (lane == 0) ls[r] += ps;
        }
        tc_load_tile<VF>(KVs, Vb, vcell, c0, cmax);        // K is no longer needed: S is in Ss
        __syncthreads();
        for (int ks = 0; ks < 2; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[3];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[2];
#pragma unroll
            for (int rt = 0; rt < 3; ++rt) wmma::load_matrix_sync(a[rt], Ps + rt * 16 * SL + ks * 16, SL);
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(b[i], KVs + ks * 16 * LD + (warp * 2 + i) * 16, LD);
#pragma unroll
            for (int rt = 0; rt < 3; ++rt)
#pragma unroll
                for (int i = 0; i < 2; ++i) wmma::mma_sync(oacc[rt][i], a[rt], b[i], oacc[rt][i]);
        }
        __syncthreads();
    }

    // ---- output: stage 16 rows at a time through shared memory, normalise, (inverse rotation for Q4 values), store
    for (int rt = 0; rt < 3; ++rt) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::store_matrix_sync(Os + (warp * 2 + i) * 16, oacc[rt][i], LD, wmma::mem_row_major);
        __syncthreads();
        for (int i = 0; i < 16; ++i) {
            const int r = rt * 16 + i, tl = r / G, g = r % G;
            float* row = Os + i * LD;
            const float inv = ls[r] > 0.0f ? 1.0f / ls[r] : 0.0f;
            row[tid] *= inv;
            __syncthreads();
            if (VF == kQ4) fwht256_shared(row);
            if (tl < ntok) out[((size_t) (t0 + tl) * n_head + kvh * G + g) * D + tid] = row[tid];
        }
        __syncthreads();
    }
}

template <int KF, int VF>
void launch_attn_chunk_tc(const float* q, const uint8_t* K, const uint8_t* V, float* out, int T, int n_head, int n_kv,
                          int pos0, int max_ctx, float scale, cudaStream_t st) {
    static bool armed[64] = {};                            // one flag per GPU and instantiation
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return;
    if (!armed[dev]) {
        cudaFuncSetAttribute(attn_chunk_tc_kernel<KF, VF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTcSmem);
        armed[dev] = true;
    }
    attn_chunk_tc_kernel<KF, VF><<<dim3(n_kv, (T + 7) / 8), 256, kTcSmem, st>>>(q, K, V, out, T, n_head, pos0, max_ctx, scale);
}
// grid (ceil(n_out / 4), rows): one warp per output row of W for one input row.
__global__ void gemv_f32_rows_kernel(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ y,
                                     int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int col = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    const float* xr = X + (size_t) col * n_in;
    float acc = 0.0f;
    for (int i = lane; i < n_in; i += 32) acc += wr[i] * xr[i];
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) col * n_out + row] = acc;
}

// One thread per conv channel, walking the chunk inside the launch (the decode path's conv_silu with a token loop).
// The thread owns its channel's whole 4-tap window, so a snapshot of the history after column j is three stores from
// registers it already has: that is how a speculative pass gets its rollback points without a copy per column.
__global__ void gdn_conv_chunk_kernel(float* __restrict__ history, const float* __restrict__ qkv,
                                      const float* __restrict__ weights, float* __restrict__ h, int channels, int T,
                                      DenseGdnSnap snap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float v[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], 0.0f};
    const float* w = weights + (size_t) c * 4;
    for (int t = 0; t < T; ++t) {
        v[3] = qkv[(size_t) t * channels + c];
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += v[tap] * w[tap];
        // __expf, not expf: the decode path's conv_silu is compiled with --use_fast_math, which turns its expf
        // into this intrinsic.  Matching it keeps a chunk's conv output as close as possible to the token loop's.
        h[(size_t) t * channels + c] = sum / (1.0f + __expf(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) v[tap] = v[tap + 1];
        if (t < snap.n) {
            float* s = snap.slot[t] + (size_t) c * 3;
            s[0] = v[0];
            s[1] = v[1];
            s[2] = v[2];
        }
    }
    history[c * 3] = v[0];
    history[c * 3 + 1] = v[1];
    history[c * 3 + 2] = v[2];
}

// grid(rows_per_token, T): one block per q or k head of one token, the row starting at `base` inside that token's
// `channels` values.  The same scaling as native_gdn_l2_norm (which is handed epsilon / 128 and multiplies by
// 1 / sqrt(128) afterwards).
__global__ void gdn_l2_norm_kernel(float* x, float eps, int channels, int base) {
    constexpr int W = 128;
    const int col = threadIdx.x;
    x += (size_t) blockIdx.y * channels + base + (size_t) blockIdx.x * W;
    const float value = col < W ? x[col] : 0.0f;
    float partial = value * value;
    __shared__ float sums[4];
    partial = warp_sum(partial);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = partial;
    __syncthreads();
    const float total = sums[0] + sums[1] + sums[2] + sums[3];
    const float scale = rsqrtf(total / (float) W + eps / (float) W);
    if (col < W) x[col] = scale * value * (1.0f / sqrtf((float) W));
}

__global__ void gdn_gates_kernel(const float* __restrict__ alpha, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int heads, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int hh = (int) (i % heads);
    const float value = alpha[i] + dt[hh];
    const float softplus = value > 20.0f ? value : __logf(1.0f + __expf(value));
    gate[i] = softplus * ssm_a[hh];
    beta[i] = 1.0f / (1.0f + __expf(-beta[i]));
}

// The delta-rule recurrence over a chunk: one block per value head, its 128 state columns split over 4 row groups of
// 32 threads (the rows a warp owns stay in registers), the T tokens walked in order, and the output norm
// rms_norm(o) * gamma * SiLU(z) folded in - SiLU, which is what qwen35 does and the MoE path's sigmoid does not.
// The readout is scaled by 1/sqrt(128) exactly as native_gdn_step's `scale`, so the norm sees the same values.
__global__ void __launch_bounds__(128 * 4) gdn_rec_chunk_kernel(
        float* __restrict__ state, const float* __restrict__ h, const float* __restrict__ gate,
        const float* __restrict__ beta, const float* __restrict__ z, const float* __restrict__ gamma, float eps,
        float* __restrict__ y, int k_heads, int v_heads, int T, DenseGdnSnap snap) {
    constexpr int W = 128, RG = 4, RPG = W / RG;
    __shared__ float sq[W], sk[W], red[RG][W], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * W + col;
    const int qh = head % k_heads;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * v_heads + head) * W + col;
    const size_t rs = (size_t) v_heads * W;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    const float rts = 1.0f / sqrtf((float) W);      // native_gdn_step's readout scale, the same expression
    const size_t hstride = (size_t) (2 * k_heads + v_heads) * W;
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * hstride;
        __syncthreads();
        if (tid < W) {
            sq[tid] = ht[qh * W + tid];
            sk[tid] = ht[k_heads * W + qh * W + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * v_heads + head]);   // the decode step's fast-math expf
        float kv = 0.0f;
        // The expressions are native_gdn_step's, written the same way, so the compiler contracts them the same way
        // (native_gdn.cu is compiled with --use_fast_math like this file): a chunk lands on the same bits as the
        // token loop wherever the reduction order allows.
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv += s[r] * sk[rg * RPG + r];
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * k_heads * W + head * W + col] - g * kv_col) * beta[(size_t) t * v_heads + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = g * s[r] + sk[rg * RPG + r] * delta;
            o += s[r] * sq[rg * RPG + r];
        }
        // The state lives in registers for the whole chunk, which is the point of this kernel: a 3 MiB state is read
        // and written once per chunk instead of once per token.  The rollback snapshots come out of the same
        // registers, so a speculative pass pays one extra store per column and nothing else.
        if (t < snap.n) {
            float* sb = snap.slot[t] + ((size_t) (rg * RPG) * v_heads + head) * W + col;
#pragma unroll
            for (int r = 0; r < RPG; ++r) sb[r * rs] = s[r];
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rts;
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float gz = z[(size_t) t * v_heads * W + head * W + col];
            y[(size_t) t * v_heads * W + head * W + col] =
                oc * rsqrtf(ss / (float) W + eps) * g_col * (gz / (1.0f + __expf(-gz)));
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

}  // namespace

void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream) {
    rms_norm_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, out, cols, eps);
    check("dense_rms_norm");
}

void dense_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int heads, int cols, float eps,
                        void* stream) {
    gdn_out_norm_kernel<<<heads, 128, 0, (cudaStream_t) stream>>>(o, z, w, y, cols, eps);
    check("dense_gdn_out_norm");
}

void dense_argmax_prob(const float* logits, int n, int* out_id, float* out_p, void* stream) {
    argmax_prob_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(logits, n, out_id, out_p);
    check("dense_argmax_prob");
}

void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream) {
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, n);
    check("dense_swiglu");
}

void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream) {
    const int warps = 4;
    gemv_f32_kernel<<<(n_out + warps - 1) / warps, warps * 32, 0, (cudaStream_t) stream>>>(W, x, y, n_in, n_out);
    check("dense_gemv_f32");
}

void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream) {
    fill_i32_kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t) stream>>>(dst, n, value);
    check("dense_fill_i32");
}

uint64_t dense_kv_cell_bytes(int fmt, int head_dim) {
    if (fmt < kF16 || fmt > kQ4) throw std::runtime_error("dense_kv_cell_bytes: unknown KV format");
    return (uint64_t) cell_bytes(fmt, head_dim);
}

namespace {
void check_kv_formats(const char* who, int k_fmt, int v_fmt, int head_dim) {
    if (k_fmt < kF16 || k_fmt > kQ4 || v_fmt < kF16 || v_fmt > kQ4)
        throw std::runtime_error(std::string(who) + ": unknown KV format");
    if (head_dim <= 0 || head_dim % 32 != 0 || head_dim > kMaxHeadDim)
        throw std::runtime_error(std::string(who) + ": head_dim must be a multiple of 32 up to 256");
    if ((k_fmt == kQ4 || v_fmt == kQ4) && head_dim != 256)
        throw std::runtime_error(std::string(who) + ": the Q4 KV format rotates 256 values per head, so head_dim must be 256");
}
}  // namespace

void dense_kv_append_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v, int pos,
                         int n_kv, int head_dim, int max_ctx, void* stream) {
    if (pos < 0 || pos >= max_ctx) throw std::runtime_error("dense_kv_append: position outside the cache");
    check_kv_formats("dense_kv_append", k_fmt, v_fmt, head_dim);
    const cudaStream_t st = (cudaStream_t) stream;
    uint8_t* kc = (uint8_t*) k_cache;
    uint8_t* vc = (uint8_t*) v_cache;
#define STRATA_KVAPP(K, V) kv_append_kernel<K, V><<<n_kv, head_dim, 0, st>>>(kc, vc, k, v, pos, head_dim, max_ctx)
    switch (k_fmt * 3 + v_fmt) {
        case 0: STRATA_KVAPP(kF16, kF16); break;
        case 1: STRATA_KVAPP(kF16, kQ8); break;
        case 2: STRATA_KVAPP(kF16, kQ4); break;
        case 3: STRATA_KVAPP(kQ8, kF16); break;
        case 4: STRATA_KVAPP(kQ8, kQ8); break;
        case 5: STRATA_KVAPP(kQ8, kQ4); break;
        case 6: STRATA_KVAPP(kQ4, kF16); break;
        case 7: STRATA_KVAPP(kQ4, kQ8); break;
        default: STRATA_KVAPP(kQ4, kQ4); break;
    }
#undef STRATA_KVAPP
    check("dense_kv_append");
}

void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream) {
    dense_kv_append_fmt(k_cache, v_cache, kF16, kF16, k, v, pos, n_kv, head_dim, max_ctx, stream);
}

uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx) {
    const uint64_t splits = (uint64_t) splits_for(max_ctx);
    return (uint64_t) n_head * splits * ((uint64_t) head_dim + 2) * sizeof(float);
}

void dense_attn_decode_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                           float* scratch, int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale,
                           void* stream) {
    if (n_ctx < 1 || n_ctx > max_ctx) throw std::runtime_error("dense_attn_decode: n_ctx outside the cache");
    if (head_dim % 32 != 0 || head_dim > kMaxHeadDim || n_head % n_kv != 0)
        throw std::runtime_error("dense_attn_decode: head_dim must be a multiple of 32 up to 256 and n_head a multiple of n_kv");
    check_kv_formats("dense_attn_decode", k_fmt, v_fmt, head_dim);
    const int n_splits = splits_for(n_ctx);
    const int max_splits = splits_for(max_ctx);
    // the partials sit at a stride of n_splits of THIS call; the scratch was sized for max_splits
    float* part_acc = scratch;
    float* part_ml = scratch + (size_t) n_head * max_splits * head_dim;
    const cudaStream_t st = (cudaStream_t) stream;
    const uint8_t* K = (const uint8_t*) k_cache;
    const uint8_t* V = (const uint8_t*) v_cache;
    const dim3 grid(n_head, n_splits);
#define STRATA_ATT(KF, VF)                                                                                      \
    attn_partial_kernel<KF, VF><<<grid, kWarps * 32, 0, st>>>(q, K, V, part_acc, part_ml, n_head, n_kv,         \
                                                              head_dim, n_ctx, max_ctx, scale, n_splits)
    switch (k_fmt * 3 + v_fmt) {
        case 0: STRATA_ATT(kF16, kF16); break;
        case 1: STRATA_ATT(kF16, kQ8); break;
        case 2: STRATA_ATT(kF16, kQ4); break;
        case 3: STRATA_ATT(kQ8, kF16); break;
        case 4: STRATA_ATT(kQ8, kQ8); break;
        case 5: STRATA_ATT(kQ8, kQ4); break;
        case 6: STRATA_ATT(kQ4, kF16); break;
        case 7: STRATA_ATT(kQ4, kQ8); break;
        default: STRATA_ATT(kQ4, kQ4); break;
    }
#undef STRATA_ATT
    check("dense_attn_partial");
    if (v_fmt == kQ4) attn_merge_kernel<kQ4><<<n_head, head_dim, 0, st>>>(part_acc, part_ml, out, head_dim, n_splits);
    else attn_merge_kernel<kF16><<<n_head, head_dim, 0, st>>>(part_acc, part_ml, out, head_dim, n_splits);
    check("dense_attn_merge");
}

void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream) {
    dense_attn_decode_fmt(q, k_cache, v_cache, kF16, kF16, out, scratch, n_head, n_kv, head_dim, n_ctx, max_ctx, scale, stream);
}

// ============================================================================ the batched prompt path
void dense_split_q(const float* q_full, float* q, int rows, int head_dim, void* stream) {
    if (rows < 1) return;
    split_q_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(q_full, q, head_dim);
    check("dense_split_q");
}

void dense_gate_apply(const float* attn, const float* q_full, float* out, int rows, int n_head, int head_dim,
                      void* stream) {
    const int64_t n = (int64_t) rows * head_dim;
    if (n <= 0) return;
    (void) n_head;
    gate_apply_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out, head_dim, n);
    check("dense_gate_apply");
}

void dense_positions_i32(int32_t* dst, int rows, int heads, int32_t pos0, void* stream) {
    if (rows < 1) return;
    positions_kernel<<<(rows + 127) / 128, 128, 0, (cudaStream_t) stream>>>(dst, rows, heads, pos0);
    check("dense_positions_i32");
}

void dense_kv_append_rows_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v,
                              int T, int pos0, int n_kv, int head_dim, int max_ctx, void* stream) {
    if (T < 1 || pos0 < 0 || pos0 + T > max_ctx)
        throw std::runtime_error("dense_kv_append_rows: positions outside the cache");
    check_kv_formats("dense_kv_append_rows", k_fmt, v_fmt, head_dim);
    const cudaStream_t st = (cudaStream_t) stream;
    uint8_t* kc = (uint8_t*) k_cache;
    uint8_t* vc = (uint8_t*) v_cache;
#define STRATA_KVROWS(K, V) \
    kv_append_rows_kernel<K, V><<<dim3(n_kv, T), head_dim, 0, st>>>(kc, vc, k, v, pos0, n_kv, head_dim, max_ctx)
    switch (k_fmt * 3 + v_fmt) {
        case 0: STRATA_KVROWS(kF16, kF16); break;
        case 1: STRATA_KVROWS(kF16, kQ8); break;
        case 2: STRATA_KVROWS(kF16, kQ4); break;
        case 3: STRATA_KVROWS(kQ8, kF16); break;
        case 4: STRATA_KVROWS(kQ8, kQ8); break;
        case 5: STRATA_KVROWS(kQ8, kQ4); break;
        case 6: STRATA_KVROWS(kQ4, kF16); break;
        case 7: STRATA_KVROWS(kQ4, kQ8); break;
        default: STRATA_KVROWS(kQ4, kQ4); break;
    }
#undef STRATA_KVROWS
    check("dense_kv_append_rows");
}

void dense_kv_append_rows(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int T, int pos0,
                          int n_kv, int head_dim, int max_ctx, void* stream) {
    dense_kv_append_rows_fmt(k_cache, v_cache, kF16, kF16, k, v, T, pos0, n_kv, head_dim, max_ctx, stream);
}

void dense_attn_chunk_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                          int T, int pos0, int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream) {
    if (T < 1 || pos0 < 0 || pos0 + T > max_ctx)
        throw std::runtime_error("dense_attn_chunk: positions outside the cache");
        static const bool use_tc = !(std::getenv("STRATA_DENSE_ATTN_TC") && std::getenv("STRATA_DENSE_ATTN_TC")[0] == '0');
    if (use_tc && n_head / n_kv == 6 && head_dim == 256) {
        const cudaStream_t st = (cudaStream_t) stream;
        const uint8_t* K = (const uint8_t*) k_cache;
        const uint8_t* V = (const uint8_t*) v_cache;
#define STRATA_ATTTC(KF, VF) launch_attn_chunk_tc<KF, VF>(q, K, V, out, T, n_head, n_kv, pos0, max_ctx, scale, st)
        switch (k_fmt * 3 + v_fmt) {
            case 0: STRATA_ATTTC(kF16, kF16); break;
            case 1: STRATA_ATTTC(kF16, kQ8); break;
            case 2: STRATA_ATTTC(kF16, kQ4); break;
            case 3: STRATA_ATTTC(kQ8, kF16); break;
            case 4: STRATA_ATTTC(kQ8, kQ8); break;
            case 5: STRATA_ATTTC(kQ8, kQ4); break;
            case 6: STRATA_ATTTC(kQ4, kF16); break;
            case 7: STRATA_ATTTC(kQ4, kQ8); break;
            default: STRATA_ATTTC(kQ4, kQ4); break;
        }
#undef STRATA_ATTTC
        check("dense_attn_chunk_tc");
        return;
    }
    if (n_head % n_kv != 0)
        throw std::runtime_error("dense_attn_chunk: n_head must be a multiple of n_kv");
    check_kv_formats("dense_attn_chunk", k_fmt, v_fmt, head_dim);
    const cudaStream_t st = (cudaStream_t) stream;
    const uint8_t* K = (const uint8_t*) k_cache;
    const uint8_t* V = (const uint8_t*) v_cache;
    const dim3 grid(n_head, T);
    if (n_head / n_kv == 6) {                  // Qwen3.8-27B: 24 query heads over 4 KV heads
        const dim3 g6(n_kv, T);
#define STRATA_ATTG(KF, VF)                                                                 \
        attn_chunk_g_kernel<KF, VF, 6><<<g6, kWarps * 32, 0, st>>>(q, K, V, out, n_head, n_kv, \
                                                                   head_dim, pos0, max_ctx, scale)
        switch (k_fmt * 3 + v_fmt) {
            case 0: STRATA_ATTG(kF16, kF16); break;
            case 1: STRATA_ATTG(kF16, kQ8); break;
            case 2: STRATA_ATTG(kF16, kQ4); break;
            case 3: STRATA_ATTG(kQ8, kF16); break;
            case 4: STRATA_ATTG(kQ8, kQ8); break;
            case 5: STRATA_ATTG(kQ8, kQ4); break;
            case 6: STRATA_ATTG(kQ4, kF16); break;
            case 7: STRATA_ATTG(kQ4, kQ8); break;
            default: STRATA_ATTG(kQ4, kQ4); break;
        }
#undef STRATA_ATTG
        check("dense_attn_chunk");
        return;
    }
#define STRATA_ATTCH(KF, VF)                                                            \
    attn_chunk_kernel<KF, VF><<<grid, kWarps * 32, 0, st>>>(q, K, V, out, n_head, n_kv,  \
                                                            head_dim, pos0, max_ctx, scale)
    switch (k_fmt * 3 + v_fmt) {
        case 0: STRATA_ATTCH(kF16, kF16); break;
        case 1: STRATA_ATTCH(kF16, kQ8); break;
        case 2: STRATA_ATTCH(kF16, kQ4); break;
        case 3: STRATA_ATTCH(kQ8, kF16); break;
        case 4: STRATA_ATTCH(kQ8, kQ8); break;
        case 5: STRATA_ATTCH(kQ8, kQ4); break;
        case 6: STRATA_ATTCH(kQ4, kF16); break;
        case 7: STRATA_ATTCH(kQ4, kQ8); break;
        default: STRATA_ATTCH(kQ4, kQ4); break;
    }
#undef STRATA_ATTCH
    check("dense_attn_chunk");
}

void dense_attn_chunk(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int T, int pos0,
                      int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream) {
    dense_attn_chunk_fmt(q, k_cache, v_cache, kF16, kF16, out, T, pos0, n_head, n_kv, head_dim, max_ctx, scale, stream);
}

void dense_gemv_f32_rows(const float* W, const float* X, float* y, int n_in, int n_out, int rows, void* stream) {
    if (rows < 1) return;
    const int warps = 4;
    const dim3 grid((n_out + warps - 1) / warps, rows);
    gemv_f32_rows_kernel<<<grid, warps * 32, 0, (cudaStream_t) stream>>>(W, X, y, n_in, n_out);
    check("dense_gemv_f32_rows");
}

void dense_gdn_conv_chunk(float* history, const float* qkv, const float* conv_w, float* h, int channels, int T,
                          const DenseGdnSnap& snap, void* stream) {
    if (T < 1) return;
    if (snap.n > DenseGdnSnap::kMaxSlots || snap.n > T)
        throw std::runtime_error("dense_gdn_conv_chunk: snapshot slots outside the chunk");
    for (int j = 0; j < snap.n; ++j)
        if (!snap.slot[j]) throw std::runtime_error("dense_gdn_conv_chunk: a snapshot slot is null");
    gdn_conv_chunk_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, T, snap);
    check("dense_gdn_conv_chunk");
}

void dense_gdn_l2_norm(float* h, int T, int channels, int base, int rows_per_token, int cols, float eps,
                       void* stream) {
    if (T < 1 || rows_per_token < 1) return;
    if (cols != 128) throw std::runtime_error("dense_gdn_l2_norm: only the 128-wide GDN state rows are supported");
    if (base < 0 || base + (int64_t) rows_per_token * cols > channels)
        throw std::runtime_error("dense_gdn_l2_norm: the head rows are outside the token's channels");
    const dim3 grid(rows_per_token, T);
    gdn_l2_norm_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(h, eps, channels, base);
    check("dense_gdn_l2_norm");
}

void dense_gdn_gates(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int heads,
                     int rows, void* stream) {
    const int64_t n = (int64_t) rows * heads;
    if (n <= 0) return;
    gdn_gates_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(alpha, dt, ssm_a, gate, beta,
                                                                                      heads, n);
    check("dense_gdn_gates");
}

void dense_gdn_rec_chunk(float* state, const float* h, const float* gate, const float* beta, const float* z,
                         const float* gamma, float eps, float* y, int k_heads, int v_heads, int T,
                         const DenseGdnSnap& snap, void* stream) {
    if (T < 1 || k_heads < 1 || v_heads < 1 || v_heads % k_heads != 0)
        throw std::runtime_error("dense_gdn_rec_chunk: T >= 1 and v_heads a positive multiple of k_heads");
    if (snap.n > DenseGdnSnap::kMaxSlots || snap.n > T)
        throw std::runtime_error("dense_gdn_rec_chunk: snapshot slots outside the chunk");
    for (int j = 0; j < snap.n; ++j)
        if (!snap.slot[j]) throw std::runtime_error("dense_gdn_rec_chunk: a snapshot slot is null");
    const dim3 block(128, 4);
    gdn_rec_chunk_kernel<<<v_heads, block, 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y,
                                                                       k_heads, v_heads, T, snap);
    check("dense_gdn_rec_chunk");
}

}  // namespace strata::kernels
