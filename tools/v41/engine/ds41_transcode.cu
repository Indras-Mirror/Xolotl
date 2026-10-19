// ds41_transcode.cu - PTQ1_0 (ternary) -> Q2_0, losslessly, on the device: lets the prompt path's MMQ (int8 tensor
// cores, which know Q2_0 but not PTQ1_0) run the ternary experts too.
//
// PTQ1_0: 128 weights = {-1, 0, +1} x one fp16 d (28 B: qs[24] + qh[2] base-3, 5 / 4 trits per byte, then d).
// Q2_0:    64 weights = {-1, 0, +1, +2} x one fp16 d (18 B: d, then qs[16] 2-bit codes, element j at byte j/4 bits
//          2*(j%4), code = value + 1).
// One PTQ1_0 block becomes two Q2_0 blocks with the SAME d and code = trit + 1, so every weight is the same number
// (code 3 = +2 is never used).  Rows stay rows: a row of K weights is K/128 PTQ1_0 blocks -> K/64 Q2_0 blocks.
#include "ds41_transcode.hpp"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

struct Ptq1Block {   // = strata_block_ptq1_0 (src/kernels/cuda/fork/fork_vecdot.cuh)
    uint8_t qs[24];
    uint8_t qh[2];
    __half d;
};
static_assert(sizeof(Ptq1Block) == 28, "PTQ1_0 block is 28 bytes");

struct Q20Block {    // = block_q2_0 (ggml-common.h)
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q20Block) == 18, "Q2_0 block is 18 bytes");

// trit digit {0,1,2} of element e (0..127) - the packer's inversion, as prom_ptq1_code (fork_prom.cuh) without the -1:
// element 16t + m in qs[m] trit t (e < 80), 80 + 8t + m in qs[16+m] trit t (e < 120), 120 + 2t + h in qh[h] trit t;
// trit t of byte b is ((uint8)(b * 3^t) * 3) >> 8
__device__ __forceinline__ uint32_t ptq1_digit(const Ptq1Block& b, int e) {
    uint8_t v;
    int t;
    if (e < 80) { v = b.qs[e & 15]; t = e >> 4; }
    else if (e < 120) { const int k = e - 80; v = b.qs[16 + (k & 7)]; t = k >> 3; }
    else { const int k = e - 120; v = b.qh[k & 1]; t = k >> 1; }
    uint32_t x = v;
    for (int i = 0; i < t; ++i) x = (x * 3u) & 0xFFu;
    return (x * 3u) >> 8;   // 0, 1, 2 = trit -1, 0, +1 = Q2_0 code 0, 1, 2
}

// one thread per Q2_0 code byte: 32 per PTQ1_0 block (2 halves x 16 bytes); byte k of half h holds elements
// 64h + 4k .. 64h + 4k + 3
__global__ void k_ptq1_to_q2_0(const Ptq1Block* __restrict__ src, Q20Block* __restrict__ dst, int64_t nb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * 32) return;
    const int64_t b = i >> 5;
    const int h = (int) ((i >> 4) & 1), k = (int) (i & 15);
    const Ptq1Block& s = src[b];
    const int e0 = 64 * h + 4 * k;
    uint32_t byte = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) byte |= ptq1_digit(s, e0 + j) << (2 * j);
    Q20Block& o = dst[2 * b + h];
    o.qs[k] = (uint8_t) byte;
    if (k == 0) o.d = s.d;
}

// the same over `count` matrices: matrix c reads src + c * src_stride and writes dst + c * dst_stride (bytes)
__global__ void k_ptq1_to_q2_0_strided(const uint8_t* __restrict__ src, size_t src_stride, uint8_t* __restrict__ dst,
                                       size_t dst_stride, int64_t nb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * 32) return;
    const size_t c = blockIdx.y;
    const Ptq1Block* sp = (const Ptq1Block*) (src + c * src_stride);
    Q20Block* dp = (Q20Block*) (dst + c * dst_stride);
    const int64_t b = i >> 5;
    const int h = (int) ((i >> 4) & 1), k = (int) (i & 15);
    const Ptq1Block& s = sp[b];
    const int e0 = 64 * h + 4 * k;
    uint32_t byte = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) byte |= ptq1_digit(s, e0 + j) << (2 * j);
    Q20Block& o = dp[2 * b + h];
    o.qs[k] = (uint8_t) byte;
    if (k == 0) o.d = s.d;
}

}  // namespace

void ds41_ptq1_to_q2_0_strided(const void* src, size_t src_stride, void* dst, size_t dst_stride, size_t ptq1_bytes,
                               int count, void* stream) {
    const int64_t nb = (int64_t) (ptq1_bytes / 28);
    if (nb <= 0 || count <= 0) return;
    const int threads = 256;
    const dim3 grid((unsigned) ((nb * 32 + threads - 1) / threads), (unsigned) count);
    k_ptq1_to_q2_0_strided<<<grid, threads, 0, (cudaStream_t) stream>>>((const uint8_t*) src, src_stride, (uint8_t*) dst,
                                                                         dst_stride, nb);
}

size_t ds41_q2_0_bytes_from_ptq1(size_t ptq1_bytes) { return ptq1_bytes / 28 * 36; }

void ds41_ptq1_to_q2_0(const void* src, void* dst, size_t ptq1_bytes, void* stream) {
    const int64_t nb = (int64_t) (ptq1_bytes / 28);
    if (nb <= 0) return;
    const int threads = 256;
    const int64_t grid = (nb * 32 + threads - 1) / threads;
    k_ptq1_to_q2_0<<<(unsigned) grid, threads, 0, (cudaStream_t) stream>>>((const Ptq1Block*) src, (Q20Block*) dst, nb);
}

namespace {
__global__ void k_add_f32(float* dst, const float* src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] += src[i];
}
}  // namespace

void ds41_add_f32(float* dst, const float* src, size_t n, void* stream) {
    if (n == 0) return;
    const int threads = 256;
    const int64_t grid = std::min<int64_t>(((int64_t) n + threads - 1) / threads, 4096);
    k_add_f32<<<(unsigned) grid, threads, 0, (cudaStream_t) stream>>>(dst, src, (int64_t) n);
}
