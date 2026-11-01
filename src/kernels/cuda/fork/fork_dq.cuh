// src/kernels/cuda/fork/fork_dq.cuh - fork formats' dequantizers: dq_q2_k, dq_mxfp4.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// Q2_K (DeepSeek-V4's down projections): ggml's dequantize_row_q2_K.  Thread tid writes 8 of the block's 256
// values: 16-value groups, four 2-bit quants per uint8, the low scale nibble for the quant and the high one for
// the min.  QK_K/16 = 16 groups, QK_K/4 = 64 quant bytes per block.
template<typename dst_t>
__device__ void dq_q2_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q2_K* x = (const block_q2_K*) vx + ibs;
    const float d = (float) x->dm.x;
    const float dmin = (float) x->dm.y;
    const int v0 = 8 * tid;   // thread tid writes the 8 values [8*tid, 8*tid + 8) of the 256
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int v = v0 + i;
        const int n = v & 128;               // the block's two 128-value qs halves (qs base 32 per half)
        const int rem = v & 127;
        const int j = rem >> 5;              // 0..3, the 2-bit quant shift 2*j
        const int half = (rem >> 4) & 1;     // the low or high 16 of that 32-value q window
        const int l = v & 15;
        const int is = (n >> 7) * 8 + 2 * j + half;
        const uint8_t sc = x->scales[is];
        const int q = (x->qs[(n >> 7) * 32 + l + 16 * half] >> (2 * j)) & 3;
        yy[v] = cvt<dst_t>(d * (sc & 0xF) * q - dmin * (sc >> 4));
    }
}

// MXFP4 (MiMo's down projections on 3 layers): ggml's dequantize_row_mxfp4 - 32-value blocks, low nibbles are values
// 0..15, high nibbles 16..31, value = kvalues_mxfp4[q] * 2^(e-127) / 2.  Same thread layout as dq_iq4_nl.
template<typename dst_t>
__device__ void dq_mxfp4(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_mxfp4* x = (const block_mxfp4*) vx + ibs * (QK_K / QK_MXFP4);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = e8m0_to_fp32(x[ib].e) * 0.5f;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_mxfp4[q4[j] >> 4]);
    }
}

// PTQ1_0 (143): a QK_K "ibs" unit is two 128-weight blocks; thread tid writes the 8 values [8*tid, 8*tid + 8),
// which sit in one block.  Trit t of byte b = ((uint8)(b * 3^t) * 3) >> 8 (digit 0..2), value (digit - 1) * d.
template<typename dst_t>
__device__ void dq_ptq1_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const strata_block_ptq1_0* x = (const strata_block_ptq1_0*) vx + 2 * ibs + (tid >> 4);
    const float d = __half2float(x->d);
    const int e0 = (8 * tid) & 127;
    dst_t* y = yy + 8 * tid;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = e0 + i;
        uint8_t b;
        int t;
        if (e < 80) { b = x->qs[e & 15]; t = e >> 4; }
        else if (e < 120) { b = x->qs[16 + ((e - 80) & 7)]; t = (e - 80) >> 3; }
        else { b = x->qh[(e - 120) & 1]; t = (e - 120) >> 1; }
        const int p3 = t == 0 ? 1 : t == 1 ? 3 : t == 2 ? 9 : t == 3 ? 27 : 81;
        const int digit = ((int) (uint8_t) (b * p3) * 3) >> 8;
        y[i] = cvt<dst_t>(d * (float) (digit - 1));
    }
}

// IQ1_S (Unsloth's Dynamic V3.1 DeepSeek-V4.1 experts): ggml's dequantize_iq1_s - thread tid writes 8 values of
// 32-value sub-block tid % 8; qh[ib] holds the sub-block's 3 grid-index bits per 8-group, its scale (bits 12-14)
// and the delta sign (bit 15).
template<typename dst_t>
__device__ void dq_iq1_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float) x[ibs].d * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
