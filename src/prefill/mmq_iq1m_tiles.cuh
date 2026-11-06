// mmq_iq1m_tiles.cuh - llama.cpp MMQ tile loader for IQ1_M (the QFN45 gate/up experts of 27 of 48 layers), appended to
// the overlay's copy of ggml-cuda/mmq-load-tiles.cuh by cmake/strata_ggml_iq1m_overlay.cmake.
//
// Port of Basalt 86b1e97f4 (Jesus Gulfo, jesdga95/basalt: "prefill: IQ1_M takes the MMQ kernels"), rewritten against the
// pinned ggml-cuda's loader signature (x_gate, ggml_prec) and with its dp4a branch for cards without int8 MMA.
//
// IQ1_M: eight grid values per 11-bit index (qs + 3 bits of qh), a grid shift of -+1/8 per eight values (bit 3 of the
// same qh nibble) and a 3-bit scale per 16 values over one fp16 superblock scale spread across the scale words.
// dequantize_row_iq1_m: y = d * (2s + 1) * (g + delta), g in {-1, 0, 1}, delta = qh & 8 ? -1/8 : +1/8.  The GPU grid
// iq1s_grid_gpu stores g + 1 in {0, 1, 2} per nibble, so y = d * (2s + 1) / 8 * (8 * (g + 1) - 8 -+ 1): the int8 weight
// 8 * (g + 1) - 9 (bit set) or - 7 (clear) is EXACT, in [-9, 9], and the scale per 16 values is d * (2s + 1) / 8.  The
// tile then has IQ2_XS's layout (int8 values, one float per 16), so IQ1_M rides IQ2_XS's Q8_0_16 dot kernels unchanged.
#pragma once

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_m(
        const char * __restrict__ x, const char * __restrict__ x_gate, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    GGML_UNUSED(x_gate);

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // one thread per 32 values (as IQ2_XS): 4 grid indices, 2 qh bytes, 2 scales of 3 bits
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_m * bxi = (const block_iq1_m *) x + kbx0 + i*stride;

        const int       qs_packed = get_int_b4(bxi->qs, kqsx);   // block_iq1_m is 56 bytes: qs stays 4-byte aligned
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

    #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl  = bxi->qh[2*kqsx + l/2] >> (4*(l%2));
            const int grid = iq1s_grid_gpu[qs[l] | ((qhl & 0x07) << 8)];
            const int sub  = (qhl & 0x08) ? 0xF7F7F7F7 : 0xF9F9F9F9;   // -9 or -7 in every byte

            const int grid_l = __vadd4(((grid >> 0) & 0x0F0F0F0F) << 3, sub);
            const int grid_h = __vadd4(((grid >> 4) & 0x0F0F0F0F) << 3, sub);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint16_t * sc = (const uint16_t *) bxi->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d   = __half2float(scale.f16) / 8.0f;
        const int   tmp = sc[kqsx/2] >> (6*(kqsx%2));

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * (2*((tmp >> 0) & 0x07) + 1);
        x_df[i*sram_stride                             + 2*kqsx+1] = d * (2*((tmp >> 3) & 0x07) + 1);
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * (2*((tmp >> 0) & 0x07) + 1);
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * (2*((tmp >> 3) & 0x07) + 1);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}
