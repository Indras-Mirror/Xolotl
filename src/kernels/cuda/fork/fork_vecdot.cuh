// src/kernels/cuda/fork/fork_vecdot.cuh - fork formats' q8_1 dot products: Q2_K (DeepSeek-V4 down), Q3_K and MXFP4 (MiMo GSQ-RCO).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// Q2_K (DeepSeek-V4's down projections): llama.cpp's vec_dot_q2_K_q8_1 (vecdotq.cuh), VDR_Q2_K_Q8_1_MMVQ = 1.
constexpr int VDR_Q2_K = 1;

__device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(int v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ scales,
                                                             const half2& dm2, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);   // the min times the sum of the QUANTIZED activations
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}
__device__ __forceinline__ float vec_dot_q2_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_K* bq2_K = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const uint8_t* scales = bq2_K->scales + scale_offset;
    const int v = get_int_b4(bq2_K->qs, iqs);
    int u[QR2_K];
    float d8[QR2_K];
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

// Q3_K (MiMo-V2.6-Flash GSQ-RCO's gate/up/down on some layers): llama.cpp's vec_dot_q3_K_q8_1 (vecdotq.cuh),
// VDR_Q3_K_Q8_1_MMVQ = 1.  The high-bit mask is inverted so a clear bit subtracts 4.
constexpr int VDR_Q3_K = 1;
__device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(const int& vl, const int& vh, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ scales, const int& scale_offset,
                                                             const float& d3, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}
__device__ __forceinline__ float vec_dot_q3_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q3_K* bq3_K = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const float d = __half2float(bq3_K->d);
    const int vl = get_int_b2(bq3_K->qs, iqs);
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
    int u[QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// MXFP4 (MiMo's source-format down projections on 3 layers): llama.cpp's vec_dot_mxfp4_q8_1 (vecdotq.cuh),
// VDR_MXFP4_Q8_1_MMVQ = 2.  kvalues_mxfp4 holds the e2m1 values doubled, hence the 0.5.
constexpr int VDR_MXFP4 = 2;
__device__ __forceinline__ float vec_dot_mxfp4_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_mxfp4* bq4 = (const block_mxfp4*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4; ++l) {
        const int2 v = get_int_from_table_16(get_int_b1(bq4->qs, iqs + l), kvalues_mxfp4);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    return e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds) * sumi;
}

// PTQ1_0 (type 143, llama-tq's group-128 ternary, 1.75 bpw; layout: include/strata/kernels/cpu/ptq1_0.hpp):
// {-1,0,+1}*d, 5 trits per byte in qs[0..23], 4 in qh[0..1], fp16 d.  Element order: qs[m] (m < 16) trit t is
// element 16t + m; qs[16+m] trit t is 80 + 8t + m; qh[h] trit t is 120 + 2t + h.  Trit t of byte b is
// ((uint8)(b * 3^t) * 3) >> 8, a digit {0,1,2}.
// The dot is llama-tq's vec_dot_ptq1_0_q8_1 (github.com/LL4nc33/llama-tq @52a37aec, ggml/src/ggml-cuda/vecdotq.cuh,
// MIT): the whole 128-weight block per call, digits {0,1,2} through dp4a, the -1 folded in as
// sum((q-1)*a) = sum(q*a) - sum(a) per q8_1 sub-block.
constexpr int QK_PTQ1_0 = 128;
struct strata_block_ptq1_0 {
    uint8_t qs[24];
    uint8_t qh[2];
    half d;
};
static_assert(sizeof(strata_block_ptq1_0) == 28, "PTQ1_0 block is 28 bytes");

__device__ __forceinline__ float vec_dot_ptq1_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const strata_block_ptq1_0* bq = (const strata_block_ptq1_0*) vbq + kbx;
    int sumi[4] = {0, 0, 0, 0};
    int sumu[4] = {0, 0, 0, 0};
    const uint32_t* qs32 = (const uint32_t*) bq->qs;   // 28-byte blocks of a 4-aligned row: 4-aligned
    // qs[0..15]: element t*16 + m for trit t of byte m
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t packed = qs32[w];
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);   // [b0, 0, b1, 0]
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);   // [b2, 0, b3, 0]
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu;
            v_hi = w_hi & 0x00FF00FFu;
            const int e = t * 16 + 4 * w;
            const int u = get_int_b4(bq8_1[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = __byte_perm(w_lo, w_hi, 0x7531);
            sumi[e >> 5] = ggml_cuda_dp4a(q, u, sumi[e >> 5]);
            sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
        }
    }
    // qs[16..23]: element 80 + t*8 + m
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t packed = qs32[4 + w];
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu;
            v_hi = w_hi & 0x00FF00FFu;
            const int e = 80 + t * 8 + 4 * w;
            const int u = get_int_b4(bq8_1[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = __byte_perm(w_lo, w_hi, 0x7531);
            sumi[e >> 5] = ggml_cuda_dp4a(q, u, sumi[e >> 5]);
            sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
        }
    }
    // qh[0..1]: element 120 + t*2 + h
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t v = bq->qh[h];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t w = v * 3;
            const int q = (int) (w >> 8);
            v = w & 0xFF;
            const int e = 120 + t * 2 + h;
            const int a = (int) bq8_1[iqs + (e >> 5)].qs[e & 31];
            sumi[e >> 5] += q * a;
            sumu[e >> 5] += a;
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) acc += __low2float(bq8_1[iqs + k].ds) * (float) (sumi[k] - sumu[k]);
    return __half2float(bq->d) * acc;
}

// Strata-DS4 fork: ONE QUARTER of a PTQ1_0 block per call (Fmt<143> ipb 4; q = iqs = 0..3; bq8_1 = the block's 4
// q8_1 sub-blocks).  The whole-block dot gave a K = 5120 row 40 calls for 32 lanes (two rounds, the second 8 lanes
// wide, each call a 128-weight chain): decode's native_gu_kernel<143> ran at ~136 GB/s (nsys 2026-10-12).  Quarter q:
//   qs[4q..4q+3]  trits t = 0..4 -> elements 16t + 4q .. +3   (contiguous per t: 5 dp4a, the same SIMD extraction)
//   qs[16+2q], qs[17+2q] trits t -> elements 80 + 8t + 2q, +1 (10 weights, scalar)
//   qh[0], qh[1] trit q          -> elements 120 + 2q, +1     (2 weights, scalar)
// = 32 weights; the -1 folded in per q8_1 sub-block as above.  Same values; the float sum order over a block changes.
__device__ __forceinline__ float vec_dot_ptq1_0_q8_1_quarter(const void* __restrict__ vbq,
                                                             const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                                             const int& q) {
    const strata_block_ptq1_0* bq = (const strata_block_ptq1_0*) vbq + kbx;
    int sumi[4] = {0, 0, 0, 0};
    int sumu[4] = {0, 0, 0, 0};
    {
        const uint32_t packed = ((const uint32_t*) bq->qs)[q];
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);   // [b0, 0, b1, 0]
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);   // [b2, 0, b3, 0]
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3u, w_hi = v_hi * 3u;
            v_lo = w_lo & 0x00FF00FFu;
            v_hi = w_hi & 0x00FF00FFu;
            const int e = t * 16 + 4 * q;
            const int u = get_int_b4(bq8_1[e >> 5].qs, (e & 31) >> 2);
            const int dq = __byte_perm(w_lo, w_hi, 0x7531);
            sumi[e >> 5] = ggml_cuda_dp4a(dq, u, sumi[e >> 5]);
            sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
        }
    }
    {
        uint32_t v = (uint32_t) bq->qs[16 + 2 * q] | ((uint32_t) bq->qs[17 + 2 * q] << 16);   // two 16-bit lanes
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w = v * 3u;
            v = w & 0x00FF00FFu;
            const int e = 80 + 8 * t + 2 * q;   // even: e and e + 1 share a q8_1 sub-block
            const int8_t* a = bq8_1[e >> 5].qs + (e & 31);
            const int a0 = a[0], a1 = a[1];
            sumi[e >> 5] += (int) ((w >> 8) & 0xFFu) * a0 + (int) (w >> 24) * a1;
            sumu[e >> 5] += a0 + a1;
        }
    }
    {
        const uint32_t p3 = q == 0 ? 1u : q == 1 ? 3u : q == 2 ? 9u : 27u;   // trit q of a byte b: ((b * 3^q) mod 256) * 3 >> 8
        const int8_t* a = bq8_1[3].qs + (120 + 2 * q - 96);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t v = ((uint32_t) bq->qh[h] * p3) & 0xFFu;
            const int d = (int) ((v * 3u) >> 8);
            sumi[3] += d * a[h];
            sumu[3] += a[h];
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) acc += __low2float(bq8_1[k].ds) * (float) (sumi[k] - sumu[k]);
    return __half2float(bq->d) * acc;
}

// IQ1_S (Unsloth's Dynamic V3.1 DeepSeek-V4.1 GGUFs: UD-IQ1_M / UD-IQ2_XXS keep whole ffn_gate/up_exps, and some
// ffn_down_exps, in IQ1_S): llama.cpp's vec_dot_iq1_s_q8_1 (vecdotq.cuh), VDR_IQ1_S_Q8_1_MMVQ = 1, one 32-value
// sub-block (iqs 0..7) per call.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
