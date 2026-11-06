// src/prefill/moe_iq4_w4a8.cu - see include/strata/prefill/moe_iq4_w4a8.hpp.
//
// llama.cpp's MMQ runs these products at 1 CTA of 8 warps per SM with every load synchronous (load the 256-value
// tile, convert it to int8 in shared memory, sync, MMA, sync): ~28 % (gate/up) and ~19 % (down) of the int8 peak on a
// 3090.  This kernel computes the same sums with a different schedule:
//   - CTA = 128 weight rows x up to BN tokens of one expert; warp w owns rows 16w..16w+15 and all of the tile's
//     tokens, so its A fragments (the weights) are converted once per K step and used for every token tile, in
//     registers;
//   - 128-value K steps through a cp.async pipeline (raw IQ4 rows + the q8_1 activation rows), one barrier per step;
//     the weights are read where they lie (pointer table) and never copied to an int8 tile;
//   - the token tiles of the last, partial tile are skipped (one instantiation per count of 8-token tiles).
// The arithmetic is MMQ's, step for step: per 32-value block the exact int32 dot (mma.m16n8k32), then
// sum += (float) dot * d_w * d_y with d_w = f16 d * (ls - 32) (IQ4_XS) or f16 d (IQ4_NL), blocks in K order, one
// accumulator per output from K = 0 - as mul_mat_q_process_tile + vec_dot_q8_0_q8_1_mma (Turing+ path) do when the
// launch tiles (no stream-k split).  EPI 0 does it as MMQ's SASS does (I2FP, FMUL, FFMA).  EPI 1 needs two FFMAs: the
// MMA's C starts at the bits of M = 1.5 * 2^23, so the result read as a float is x = M + dot exactly (|dot| <=
// 32 * 127 * 127 < 2^22), and fma(x, d_w, -M * d_w) = round(x * d_w - M * d_w) = round(dot * d_w) - the FMUL's value -
// because -M * d_w is exact: d_w = f16 d * (ls - 32) has at most 11 + 6 significant bits (IQ4_NL: 11), so 3 * 2^22 *
// d_w fits a float's 24.  Then sum += t * d_y as before.  The int8 lookup is llama.cpp's get_int_from_table_16 (MIT).
#include "strata/prefill/moe_iq4_w4a8.hpp"

#if defined(GGML_USE_HIP)
// The gfx906 HIP build compiles every CUDA source of the tree as HIP: this kernel is NVIDIA PTX (mma.sync, cp.async),
// so there it declines every product and the MMQ path runs as before.
namespace strata::prefill::mmq {
int iq4_w4a8_mode() { return 0; }
bool run_iq4_w4a8(const Product&, void*) { return false; }
}  // namespace strata::prefill::mmq
#else

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>

namespace strata::prefill::mmq {
namespace {

constexpr int kThreads = 256;           // 8 warps x 16 weight rows
constexpr int kBM = 128;                // weight rows per CTA
constexpr int kYRow = 144;              // one block_q8_1_mmq row: 4 f32 scales + 128 int8

// WROW: shared bytes per weight row and K step (72 of data; 80 = conflict-free A reads, 72 = 2-way on 2 lanes but
// the cp.async writes 2-way instead of 4-way); LOOP 0: per token tile 4 MMAs then their epilogue, LOOP 1: one flat
// software-pipelined sequence - MMA s, then the epilogue of MMA s - 2, the next tile's B two MMAs ahead
template <int BN_, int STAGES_, int MINB_, int EPI_, int WROW_, int LOOP_>
struct Cfg {
    static constexpr int BN = BN_, STAGES = STAGES_, MINB = MINB_, EPI = EPI_, WROW = WROW_, LOOP = LOOP_;
    static constexpr int WStage = kBM * WROW;
    static constexpr int YStage = BN * kYRow;
    static constexpr int Stage = WStage + YStage;
    static constexpr int Smem = STAGES * Stage;
};

struct Tab16 {
    const char* p[kGatherGroupMax];
};

__device__ __forceinline__ void cp8(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
__device__ __forceinline__ void lds128(float4& v, uint32_t addr) {
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr));
}
template <uint32_t C0>
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(C0));
}

constexpr uint32_t kMagicBits = 0x4B400000u;   // 1.5 * 2^23
constexpr float kMagic = 12582912.0f;


// kvalues_iq4nl {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113} as 4 little-endian words
constexpr uint32_t kT0 = 0xBFAD9881u, kT1 = 0xF6EADDCFu, kT2 = 0x26190D01u, kT3 = 0x71594535u;

// 8 packed nibbles (byte i = value i low, value i + 16 high) -> lo: the 4 low nibbles' int8, hi: the 4 high ones'
// (llama.cpp get_int_from_table_16, MIT)
__device__ __forceinline__ void lut16(uint32_t q, uint32_t& lo, uint32_t& hi) {
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t l0 = __byte_perm(kT0, kT1, q), h0 = __byte_perm(kT2, kT3, q);
    const uint32_t l1 = __byte_perm(kT0, kT1, q >> 16), h1 = __byte_perm(kT2, kT3, q >> 16);
    const uint32_t m0 = __byte_perm(l0, h0, sel), m1 = __byte_perm(l1, h1, sel >> 16);
    lo = __byte_perm(m0, m1, 0x6420);
    hi = __byte_perm(m0, m1, 0x7531);
}

__device__ __forceinline__ float sel4(const float4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// one CTA tile: weight rows row0..row0+127 of expert matrix w, the tile's tokens = activation rows yrow0..yrow0+nv-1
template <bool XS, class C, int NT>
__device__ __forceinline__ void tile(const char* __restrict__ w, const char* __restrict__ y,
                                     const int32_t* __restrict__ ids, float* __restrict__ dst, int64_t ld_dst,
                                     int row0, int nk, int64_t row_bytes, int64_t ncols_y, int yrow0, int nv,
                                     unsigned char* smem) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const uint32_t s0 = (uint32_t) __cvta_generic_to_shared(smem);
    const int my_r = tid & (kBM - 1), my_p0 = tid >> 7;
    const char* my_row = w + (int64_t) (row0 + my_r) * row_bytes;

    auto issue = [&](int kb, int st) {
        const uint32_t sw = s0 + st * C::Stage, sy = sw + C::WStage;
        // weights: 128 rows x 9 pieces of 8 bytes (IQ4_XS: the super-block's 8-byte header + this half's 64 qs
        // bytes; IQ4_NL: 4 blocks of 18 bytes)
        // (thread tid copies pieces tid/128, +2, +4, ... of row tid % 128)
        const uint32_t dw = sw + my_r * C::WROW;
        const char* rk = XS ? my_row + (int64_t) (kb >> 1) * 136 : my_row + (int64_t) kb * 72;
#pragma unroll
        for (int p = my_p0; p < 9; p += 2) {
            const char* src;
            if constexpr (XS) src = p == 0 ? rk : rk + 8 + (kb & 1) * 64 + (p - 1) * 8;
            else src = rk + p * 8;
            cp8(dw + p * 8, src);
        }
        // activations: the tile's rows are consecutive in each 128-value block of the q8_1 array
        const char* yk = y + ((int64_t) kb * ncols_y + yrow0) * kYRow;
#pragma unroll
        for (int i = tid; i < NT * 8 * 9; i += kThreads) cp16(sy + i * 16, yk + (int64_t) i * 16);
    };

    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < C::STAGES - 1; ++s) {
        if (s < nk) issue(s, s);
        cp_commit();
    }

    for (int kb = 0; kb < nk; ++kb) {
        cp_wait<C::STAGES - 2>();
        __syncthreads();
        {
            const int kn = kb + C::STAGES - 1;
            if (kn < nk) issue(kn, kn % C::STAGES);
            cp_commit();
        }
        const int st = kb % C::STAGES;
        const unsigned char* W = smem + st * C::Stage + warp * 16 * C::WROW;
        const unsigned char* Wg = W + g * C::WROW;
        const unsigned char* Wg8 = W + (g + 8) * C::WROW;
        const uint32_t sy = s0 + st * C::Stage + C::WStage;

        // this warp's A fragments and weight scales for the step's 4 blocks
        uint32_t a[4][4];
        float da[4][2], nm[4][2];
        if constexpr (XS) {
            const int h = kb & 1;
            const float d0 = __half2float(*(const __half*) Wg), d1 = __half2float(*(const __half*) Wg8);
            const uint32_t sh0 = *(const uint16_t*) (Wg + 2), sh1 = *(const uint16_t*) (Wg8 + 2);
            const uint32_t sl0 = *(const uint32_t*) (Wg + 4), sl1 = *(const uint32_t*) (Wg8 + 4);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int ib = 4 * h + b;   // the block in its super-block
                const int ls0 = ((sl0 >> (4 * ib)) & 0x0F) | (((sh0 >> (2 * ib)) & 0x03) << 4);
                const int ls1 = ((sl1 >> (4 * ib)) & 0x0F) | (((sh1 >> (2 * ib)) & 0x03) << 4);
                da[b][0] = d0 * (ls0 - 32);
                da[b][1] = d1 * (ls1 - 32);
                const uint32_t q0 = *(const uint32_t*) (Wg + 8 + 16 * b + 4 * t);
                const uint32_t q1 = *(const uint32_t*) (Wg8 + 8 + 16 * b + 4 * t);
                lut16(q0, a[b][0], a[b][2]);
                lut16(q1, a[b][1], a[b][3]);
            }
        } else {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                da[b][0] = __half2float(*(const __half*) (Wg + 18 * b));
                da[b][1] = __half2float(*(const __half*) (Wg8 + 18 * b));
                const int o = 18 * b + 2 + 4 * t;   // 2-byte aligned for even b
                uint32_t q0, q1;
                if (b & 1) {
                    q0 = *(const uint32_t*) (Wg + o);
                    q1 = *(const uint32_t*) (Wg8 + o);
                } else {
                    q0 = (uint32_t) *(const uint16_t*) (Wg + o) | ((uint32_t) *(const uint16_t*) (Wg + o + 2) << 16);
                    q1 = (uint32_t) *(const uint16_t*) (Wg8 + o) | ((uint32_t) *(const uint16_t*) (Wg8 + o + 2) << 16);
                }
                lut16(q0, a[b][0], a[b][2]);
                lut16(q1, a[b][1], a[b][3]);
            }
        }
        if constexpr (C::EPI == 1) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                nm[b][0] = -kMagic * da[b][0];
                nm[b][1] = -kMagic * da[b][1];
            }
        }

        if constexpr (C::LOOP == 1) {
            uint32_t bq[2][4][2];
            float4 dq[2][2];
            auto load_b = [&](int n, int buf) {
                lds128(dq[buf][0], sy + (n * 8 + 2 * t) * kYRow);
                lds128(dq[buf][1], sy + (n * 8 + 2 * t + 1) * kYRow);
                const uint32_t ra = sy + (n * 8 + (lane & 7)) * kYRow + 16 + 32 * (lane >> 4) + 16 * ((lane >> 3) & 1);
                ldsm4(bq[buf][0][0], bq[buf][0][1], bq[buf][1][0], bq[buf][1][1], ra);
                ldsm4(bq[buf][2][0], bq[buf][2][1], bq[buf][3][0], bq[buf][3][1], ra + 64);
            };
            int cr[3][4];
            load_b(0, 0);
#pragma unroll
            for (int s = 0; s < NT * 4 + 2; ++s) {
                if (s < NT * 4) {
                    const int n = s >> 2, k = s & 3;
                    // tile n + 1's B into the buffer tile n - 1 used: its last MMA was s - 3, its last epilogue s - 1
                    if (k == 2 && n + 1 < NT) load_b(n + 1, (n + 1) & 1);
                    mma_s8<C::EPI == 1 ? kMagicBits : 0u>(cr[s % 3], a[k], bq[n & 1][k][0], bq[n & 1][k][1]);
                }
                if (s >= 2) {
                    const int e = s - 2, n = e >> 2, k = e & 3;
                    const int* c = cr[e % 3];
                    const float y0 = sel4(dq[n & 1][0], k), y1 = sel4(dq[n & 1][1], k);
                    if constexpr (C::EPI == 1) {
                        const float t0 = __fmaf_rn(__int_as_float(c[0]), da[k][0], nm[k][0]);
                        const float t1 = __fmaf_rn(__int_as_float(c[1]), da[k][0], nm[k][0]);
                        const float t2 = __fmaf_rn(__int_as_float(c[2]), da[k][1], nm[k][1]);
                        const float t3 = __fmaf_rn(__int_as_float(c[3]), da[k][1], nm[k][1]);
                        acc[n][0] += t0 * y0;
                        acc[n][1] += t1 * y1;
                        acc[n][2] += t2 * y0;
                        acc[n][3] += t3 * y1;
                    } else {
                        acc[n][0] += c[0] * da[k][0] * y0;
                        acc[n][1] += c[1] * da[k][0] * y1;
                        acc[n][2] += c[2] * da[k][1] * y0;
                        acc[n][3] += c[3] * da[k][1] * y1;
                    }
                }
            }
        } else {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            float4 db0, db1;
            lds128(db0, sy + (n * 8 + 2 * t) * kYRow);
            lds128(db1, sy + (n * 8 + 2 * t + 1) * kYRow);
            uint32_t b[4][2];
            const uint32_t ra = sy + (n * 8 + (lane & 7)) * kYRow + 16 + 32 * (lane >> 4) + 16 * ((lane >> 3) & 1);
            ldsm4(b[0][0], b[0][1], b[1][0], b[1][1], ra);
            ldsm4(b[2][0], b[2][1], b[3][0], b[3][1], ra + 64);
            int c[4][4];
#pragma unroll
            for (int k = 0; k < 4; ++k) mma_s8<C::EPI == 1 ? kMagicBits : 0u>(c[k], a[k], b[k][0], b[k][1]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float y0 = sel4(db0, k), y1 = sel4(db1, k);
                if constexpr (C::EPI == 1) {
                    const float t0 = __fmaf_rn(__int_as_float(c[k][0]), da[k][0], nm[k][0]);
                    const float t1 = __fmaf_rn(__int_as_float(c[k][1]), da[k][0], nm[k][0]);
                    const float t2 = __fmaf_rn(__int_as_float(c[k][2]), da[k][1], nm[k][1]);
                    const float t3 = __fmaf_rn(__int_as_float(c[k][3]), da[k][1], nm[k][1]);
                    acc[n][0] += t0 * y0;
                    acc[n][1] += t1 * y1;
                    acc[n][2] += t2 * y0;
                    acc[n][3] += t3 * y1;
                } else {
                    acc[n][0] += c[k][0] * da[k][0] * y0;
                    acc[n][1] += c[k][1] * da[k][0] * y1;
                    acc[n][2] += c[k][2] * da[k][1] * y0;
                    acc[n][3] += c[k][3] * da[k][1] * y1;
                }
            }
        }
        }
    }
    cp_wait<0>();

    const int r = row0 + warp * 16 + g;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int j = n * 8 + 2 * t;
        if (j < nv) {
            float* o = dst + (int64_t) ids[yrow0 + j] * ld_dst + r;
            o[0] = acc[n][0];
            o[8] = acc[n][2];
        }
        if (j + 1 < nv) {
            float* o = dst + (int64_t) ids[yrow0 + j + 1] * ld_dst + r;
            o[0] = acc[n][1];
            o[8] = acc[n][3];
        }
    }
}

// grid: (row tile, token tile, expert) with the row tile fastest; tiles past an expert's rows return at once
template <bool XS, class C>
__global__ void __launch_bounds__(kThreads, C::MINB)
    iq4_w4a8_kernel(Tab16 tab, const char* __restrict__ y, const int32_t* __restrict__ ids,
                    const int32_t* __restrict__ bounds, float* __restrict__ dst, int64_t ld_dst, int nk,
                    int64_t row_bytes, int64_t ncols_y, int nty, int ntx) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int it = blockIdx.x % nty, rest = blockIdx.x / nty;
    const int jt = rest % ntx, z = rest / ntx;
    const int lo = bounds[z], cnt = bounds[z + 1] - lo;
    const int j0 = jt * C::BN;
    if (j0 >= cnt) return;
    const int nv = min(C::BN, cnt - j0);
    const char* w = tab.p[0];
#pragma unroll
    for (int i = 1; i < kGatherGroupMax; ++i)
        if (i == z) w = tab.p[i];
    const int row0 = it * kBM, yrow0 = lo + j0;
    const int nt = (nv + 7) >> 3;
#define STRATA_W4A8_NT(n) \
    if constexpr (n * 8 <= C::BN) { \
        if (nt == n) { tile<XS, C, n>(w, y, ids, dst, ld_dst, row0, nk, row_bytes, ncols_y, yrow0, nv, smem); return; } \
    }
    STRATA_W4A8_NT(1) STRATA_W4A8_NT(2) STRATA_W4A8_NT(3) STRATA_W4A8_NT(4)
    STRATA_W4A8_NT(5) STRATA_W4A8_NT(6) STRATA_W4A8_NT(7) STRATA_W4A8_NT(8)
    STRATA_W4A8_NT(9) STRATA_W4A8_NT(10) STRATA_W4A8_NT(11) STRATA_W4A8_NT(12)
    STRATA_W4A8_NT(13) STRATA_W4A8_NT(14) STRATA_W4A8_NT(15) STRATA_W4A8_NT(16)
#undef STRATA_W4A8_NT
}

// configurations (STRATA_W4A8_CFG): 1 CTA/SM, 2 stages, two FFMAs (EPI 1), rows of 72 bytes, the pipelined loop;
// tokens per CTA 0 = 128, 1 = 104, 2 = 112, 3 = 96
using Cfg0 = Cfg<128, 2, 1, 1, 72, 1>;
using Cfg1 = Cfg<104, 2, 1, 1, 72, 1>;
using Cfg2 = Cfg<112, 2, 1, 1, 72, 1>;
using Cfg3 = Cfg<96, 2, 1, 1, 72, 1>;

int cfg_index() {
    static const int c = [] {
        const char* v = std::getenv("STRATA_W4A8_CFG");
        const int i = v != nullptr ? std::atoi(v) : 0;
        return i >= 0 && i <= 3 ? i : 0;
    }();
    return c;
}

template <class C>
void set_smem() {
    cudaFuncSetAttribute(iq4_w4a8_kernel<true, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::Smem);
    cudaFuncSetAttribute(iq4_w4a8_kernel<false, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::Smem);
}

// per device, once: SM count, compute capability, the configs' shared-memory opt-in (the engine drives both GPUs)
int device_info(int& nsm) {
    constexpr int kMaxDev = 16;
    static std::once_flag once[kMaxDev];
    static int nsm_d[kMaxDev], cc_d[kMaxDev];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= kMaxDev) return 0;
    std::call_once(once[dev], [dev] {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        nsm_d[dev] = prop.multiProcessorCount;
        cc_d[dev] = prop.major * 10 + prop.minor;
        set_smem<Cfg0>();
        set_smem<Cfg1>();
        set_smem<Cfg2>();
        set_smem<Cfg3>();
    });
    nsm = nsm_d[dev];
    return cc_d[dev];
}

template <class C>
void launch(bool xs, const Tab16& tab, const Product& p, int64_t row_bytes, cudaStream_t s) {
    const int nty = (int) (p.w_rows / kBM), ntx = (int) ((p.max_rows + C::BN - 1) / C::BN);
    const dim3 grid((unsigned) (nty * ntx * p.n));
    const int nk = (int) (p.w_cols / 128);
    if (xs)
        iq4_w4a8_kernel<true, C><<<grid, kThreads, C::Smem, s>>>(tab, (const char*) p.xq, p.ids, p.bounds, p.dst,
                                                                 p.ld_dst, nk, row_bytes, p.total_rows, nty, ntx);
    else
        iq4_w4a8_kernel<false, C><<<grid, kThreads, C::Smem, s>>>(tab, (const char*) p.xq, p.ids, p.bounds, p.dst,
                                                                  p.ld_dst, nk, row_bytes, p.total_rows, nty, ntx);
}

}  // namespace

// the compute capabilities the kernel's bits were checked against MMQ's on (prefill_iq4_w4a8_test): Ampere (8.0-8.8) and
// Blackwell 12.0; Ada (8.9) has no card to test on, so it keeps MMQ
static bool arch_ok(int cc) { return (cc >= 80 && cc < 89) || cc == 120; }

void report_counts();
int iq4_w4a8_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_IQ4_W4A8");
        const int r = v != nullptr ? std::atoi(v) : 0;
        if (r == 1 || r == 2)
            std::fprintf(stderr, "strata: prompt experts: the IQ4 W4A8 kernel runs %s (STRATA_IQ4_W4A8=%d, config %d; "
                                 "MMQ's bits)\n", r == 1 ? "the IQ4_XS and IQ4_NL products" : "the IQ4_NL (down) products",
                         r, cfg_index());
        if (r == 1 || r == 2) std::atexit(report_counts);
        return r == 1 || r == 2 ? r : 0;
    }();
    return m;
}

// dispatch counters (the path ran / why it declined), printed once at exit when the kernel is on
std::atomic<long long> g_ran[2], g_declined[6];   // ran: [IQ4_XS, IQ4_NL]; declined: shape, scales, arch, splits-K, alignment, type
const char* const kWhy[6] = {"shape", "NVFP4 scales", "architecture", "MMQ would split K", "alignment", "other type"};
void report_counts() {
    std::fprintf(stderr, "strata: IQ4 W4A8 kernel: ran %lld IQ4_XS and %lld IQ4_NL products; declined", g_ran[0].load(), g_ran[1].load());
    for (int i = 0; i < 6; ++i) std::fprintf(stderr, " %s %lld%s", kWhy[i], g_declined[i].load(), i < 5 ? "," : "");
    std::fprintf(stderr, " (MMQ ran those)\n");
}
bool decline(int why) { g_declined[why].fetch_add(1, std::memory_order_relaxed); return false; }

bool run_iq4_w4a8(const Product& p, void* stream) {
    const bool xs = p.type == GGML_TYPE_IQ4_XS;
    if (!xs && p.type != GGML_TYPE_IQ4_NL) return decline(5);
    if (p.n <= 0 || p.n > kGatherGroupMax || p.max_rows <= 0) return decline(0);
    if (p.w_rows % kBM != 0 || p.w_cols % (xs ? 256 : 128) != 0) return decline(0);
    if (p.y_scale != nullptr) return decline(1);   // (NVFP4 w4a4 scales: not an IQ4 product)
    int nsm = 0;
    const int cc = device_info(nsm);
    if (!arch_ok(cc)) return decline(2);
    if (launch_splits_k(p.type, p.w_rows, p.max_rows, p.n)) return decline(3);   // MMQ's stream-k fixup adds the sums in another order
    const int64_t row_bytes = p.w_cols / (xs ? 256 : 32) * (xs ? 136 : 18);
    Tab16 tab{};
    for (int z = 0; z < p.n; ++z) {
        tab.p[z] = p.tab ? (const char*) p.tab->p[z] : (const char*) p.w + (size_t) z * p.expert_bytes;
        if (((uintptr_t) tab.p[z] & 7) != 0) return decline(4);   // 8-byte cp.async pieces
    }
    for (int z = p.n; z < kGatherGroupMax; ++z) tab.p[z] = tab.p[0];
    if ((((uintptr_t) p.xq) & 15) != 0) return decline(4);
    const cudaStream_t s = (cudaStream_t) stream;
    switch (cfg_index()) {
        case 1: launch<Cfg1>(xs, tab, p, row_bytes, s); break;
        case 2: launch<Cfg2>(xs, tab, p, row_bytes, s); break;
        case 3: launch<Cfg3>(xs, tab, p, row_bytes, s); break;
        default: launch<Cfg0>(xs, tab, p, row_bytes, s); break;
    }
    g_ran[xs ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace strata::prefill::mmq

#endif  // GGML_USE_HIP
