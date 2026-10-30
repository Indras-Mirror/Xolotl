// Dense GQA prefill: FA2, FP16 tensor cores, FP32 online softmax and split-K merge.
#include "strata/qwen35/flash.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace strata::qwen35 {
namespace {
#ifndef STRATA_Q36_FA_ROWS
#define STRATA_Q36_FA_ROWS 128
#endif
constexpr int D = 256, BK = 32, ROWS = STRATA_Q36_FA_ROWS, WARPS = ROWS / 16;
static_assert(ROWS == 64 || ROWS == 128, "q36 flash tiles use 64 or 128 query rows");
constexpr int STRIDE = D + 8, PS = BK + 8;

#if !defined(__HIPCC__)
__device__ __forceinline__ uint32_t pack(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}
__device__ __forceinline__ void copy16(__half* dst, const __half* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr), "l"(src), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
__device__ __forceinline__ void load_b(uint32_t* b, const __half* ptr, bool transpose) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    if (transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
#endif
}
struct Shared {
    __half k[2][BK][STRIDE], v[2][BK][STRIDE];
    __half p[WARPS][16][PS];
};
__device__ __forceinline__ void stage(Shared& s, int buf, const __half* k, const __half* v, int base, int P) {
    for (int i = threadIdx.x; i < BK * D / 8; i += WARPS * 32) {
        const int row = i / (D / 8), d = (i % (D / 8)) * 8;
        // Clamp even the zero-fill source pointer: no out-of-allocation pointer on a partial tile.
        const size_t off = static_cast<size_t>(min(base + row, P - 1)) * D + d;
        copy16(&s.k[buf][row][d], k + off, base + row < P);
        copy16(&s.v[buf][row][d], v + off, base + row < P);
    }
    commit();
}

template<int G>
__global__ __launch_bounds__(WARPS * 32, 1)
void flash_part(const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                float* __restrict__ partial, int T, int H, int ctx, int p0, int splits, int tiles_per_split) {
    extern __shared__ __align__(16) unsigned char storage[];
    Shared& s = *reinterpret_cast<Shared*>(storage);
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int gr = lane / 4, tc = lane % 4;
    const int kvh = blockIdx.y, split = blockIdx.z;
    constexpr int QT = ROWS / G;
    const int tbase = blockIdx.x * QT;
    const int r0 = warp * 16 + gr, r1 = r0 + 8;
    const int t0 = tbase + r0 / G, t1 = tbase + r1 / G;
    const int h0 = kvh * G + r0 % G, h1 = kvh * G + r1 % G;
    const int P = p0 + T;
    const int begin = split * tiles_per_split * BK;
    const int end = min(min(begin + tiles_per_split * BK, P), p0 + min(tbase + QT, T));
    if (begin >= end) {
        for (int i = threadIdx.x; i < ROWS * (D + 2); i += WARPS * 32) {
            const int row = i / (D + 2), d = i % (D + 2);
            const int t = tbase + row / G, h = kvh * G + row % G;
            if (t < T)
                partial[((static_cast<size_t>(split) * T + t) * H + h) * (D + 2) + d] = d == D ? -1.e30f : 0.f;
        }
        return;
    }
    k += static_cast<size_t>(kvh) * ctx * D;
    v += static_cast<size_t>(kvh) * ctx * D;

    uint32_t qr[D / 16][4];
#pragma unroll
    for (int kd = 0; kd < D / 16; ++kd) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = j & 1 ? t1 : t0, h = j & 1 ? h1 : h0;
            const int d = kd * 16 + tc * 2 + (j / 2) * 8;
            const size_t off = (static_cast<size_t>(t) * H + h) * D + d;
            qr[kd][j] = t < T ? pack(q[off], q[off + 1]) : 0;
        }
    }
    float acc[D / 8][4] = {};
    float m0 = -1.e30f, m1 = -1.e30f, l0 = 0.f, l1 = 0.f;
    if (begin < end) {
        stage(s, 0, k, v, begin, P);
        wait();
        __syncthreads();
        for (int base = begin, tile = 0; base < end; base += BK, ++tile) {
            const int buf = tile & 1;
            if (base + BK < end) stage(s, buf ^ 1, k, v, base + BK, P);
            float score[BK / 8][4] = {};
#pragma unroll
            for (int kd = 0; kd < D / 16; ++kd) {
#pragma unroll
                for (int n = 0; n < BK / 8; ++n) {
                    uint32_t b[2];
                    load_b(b, &s.k[buf][n * 8 + lane % 8][kd * 16 + ((lane / 8) & 1) * 8], false);
                    mma(score[n], qr[kd], b);
                }
            }
            float mx0 = m0, mx1 = m1;
#pragma unroll
            for (int n = 0; n < BK / 8; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int cell = base + n * 8 + tc * 2 + (j & 1);
                    const int t = j < 2 ? t0 : t1;
                    // scale=1/sqrt(256); log2 scale makes the inner loop use exp2.
                    score[n][j] = cell <= p0 + t && cell < P && t < T
                                      ? score[n][j] * (0.0625f * 1.4426950408889634f) : -INFINITY;
                }
                mx0 = fmaxf(mx0, fmaxf(score[n][0], score[n][1]));
                mx1 = fmaxf(mx1, fmaxf(score[n][2], score[n][3]));
            }
#pragma unroll
            for (int d = 1; d <= 2; d *= 2) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, d));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, d));
            }
            const float a0 = exp2f(m0 - mx0), a1 = exp2f(m1 - mx1);
            m0 = mx0; m1 = mx1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int n = 0; n < BK / 8; ++n) {
                const float p00 = exp2f(score[n][0] - m0), p01 = exp2f(score[n][1] - m0);
                const float p10 = exp2f(score[n][2] - m1), p11 = exp2f(score[n][3] - m1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                *reinterpret_cast<uint32_t*>(&s.p[warp][gr][n * 8 + tc * 2]) = pack(p00, p01);
                *reinterpret_cast<uint32_t*>(&s.p[warp][gr + 8][n * 8 + tc * 2]) = pack(p10, p11);
            }
#pragma unroll
            for (int d = 1; d <= 2; d *= 2) {
                sum0 += __shfl_xor_sync(0xffffffff, sum0, d);
                sum1 += __shfl_xor_sync(0xffffffff, sum1, d);
            }
            l0 = l0 * a0 + sum0; l1 = l1 * a1 + sum1;
#pragma unroll
            for (int n = 0; n < D / 8; ++n) {
                acc[n][0] *= a0; acc[n][1] *= a0;
                acc[n][2] *= a1; acc[n][3] *= a1;
            }
            __syncwarp();
#pragma unroll
            for (int kd = 0; kd < BK / 16; ++kd) {
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t*>(&s.p[warp][gr][kd * 16 + tc * 2]);
                a[1] = *reinterpret_cast<const uint32_t*>(&s.p[warp][gr + 8][kd * 16 + tc * 2]);
                a[2] = *reinterpret_cast<const uint32_t*>(&s.p[warp][gr][kd * 16 + tc * 2 + 8]);
                a[3] = *reinterpret_cast<const uint32_t*>(&s.p[warp][gr + 8][kd * 16 + tc * 2 + 8]);
#pragma unroll
                for (int n = 0; n < D / 8; ++n) {
                    uint32_t b[2];
                    load_b(b, &s.v[buf][kd * 16 + lane % 16][n * 8], true);
                    mma(acc[n], a, b);
                }
            }
            wait();
            __syncthreads();  // next buffer ready; every warp finished reading the old buffer
        }
    }
    // [split,T,H,D+2], with log2(max) and the unrounded FP32 denominator.
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int t = rr ? t1 : t0, h = rr ? h1 : h0;
        if (t < T) {
            float* dst = partial + ((static_cast<size_t>(split) * T + t) * H + h) * (D + 2);
#pragma unroll
            for (int n = 0; n < D / 8; ++n) {
                dst[n * 8 + tc * 2] = acc[n][rr * 2];
                dst[n * 8 + tc * 2 + 1] = acc[n][rr * 2 + 1];
            }
            if (tc == 0) { dst[D] = rr ? m1 : m0; dst[D + 1] = rr ? l1 : l0; }
        }
    }
}
__global__ void merge(const float* __restrict__ partial, const float* __restrict__ gate,
                      float* __restrict__ out, int rows, int splits) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float* src = partial + static_cast<size_t>(row) * (D + 2);
    const size_t stride = static_cast<size_t>(rows) * (D + 2);
    float m = -1.e30f;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, src[s * stride + D]);
    float l = 0.f, a = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float w = exp2f(src[s * stride + D] - m);
        l += w * src[s * stride + D + 1];
        a += w * src[s * stride + d];
    }
    const size_t i = static_cast<size_t>(row) * D + d;
    out[i] = (a / l) / (1.f + expf(-gate[i]));
}
#endif
}  // namespace

bool flash_available() {
#if defined(__HIPCC__)
    return false;
#else
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    return prop.major >= 8 && prop.sharedMemPerBlockOptin >= sizeof(Shared);
#endif
}
int flash_splits(int T, int H, int HKV, int P, int sms) {
    const int blocks = ((T + ROWS / (H / HKV) - 1) / (ROWS / (H / HKV))) * HKV;
    return std::max(1, std::min({256, (2 * sms + blocks - 1) / blocks, (P + BK - 1) / BK}));
}
size_t flash_workspace_bytes(int max_T, int H, int HKV, int ctx, int sms) {
    size_t rows = 0;
    for (int t = 9; t <= max_T; ++t)
        rows = std::max(rows, static_cast<size_t>(t) * H * flash_splits(t, H, HKV, ctx, sms));
    return rows * (D + 2) * sizeof(float);
}
void flash_prefill(const float* q, const __half* k, const __half* v, const float* gate,
                   float* out, void* workspace, int T, int H, int HKV, int ctx, int p0,
                   int sms, cudaStream_t stream) {
#if !defined(__HIPCC__)
    const int G = H / HKV, splits = flash_splits(T, H, HKV, p0 + T, sms);
    const int tiles = ((p0 + T + BK - 1) / BK + splits - 1) / splits;
    const dim3 grid((T + ROWS / G - 1) / (ROWS / G), HKV, splits);
    auto launch = [&](auto g) {
        constexpr int group = decltype(g)::value;
        static const cudaError_t attr = cudaFuncSetAttribute(flash_part<group>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
        if (attr != cudaSuccess) throw std::runtime_error(cudaGetErrorString(attr));
        flash_part<group><<<grid, WARPS * 32, sizeof(Shared), stream>>>(q, k, v, static_cast<float*>(workspace),
                                                                     T, H, ctx, p0, splits, tiles);
    };
    switch (G) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        case 16: launch(std::integral_constant<int, 16>{}); break;
        default: throw std::invalid_argument("q36 flash: unsupported GQA");
    }
    merge<<<T * H, D, 0, stream>>>(static_cast<const float*>(workspace), gate, out, T * H, splits);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
#else
    throw std::runtime_error("q36 flash requires CUDA SM80+");
#endif
}
}  // namespace strata::qwen35
