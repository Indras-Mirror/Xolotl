// Strata-DS4 fork code (docs/UPSTREAM_SYNC.md): decode-once Split<> traits for formats only the fork's Fmt<> list has.

// IQ1_S (19): vec_dot_iq1_s_q8_1 (fork_vecdot.cuh) as load + apply, so Unsloth's UD-IQ1_M / UD-IQ2_XXS IQ1_S
// gate/up and down experts take the multi kernels (native_gu_multi_kernel / native_down_multi_kernel) instead of the
// per-entry ones.  apply(load(...)) is the dot's dp4a chain and float expression in the same order: bitwise equal.
// RTX 5090, UD-IQ1_M DSpark decode (CUPTI): native_gu_kernel<19> 57.9 us per call vs IQ2_XXS's multi 47.5 us.
template<> inline constexpr bool kSplit<19> = true;
template<> struct Split<19> {   // IQ1_S
    struct W { int g[8]; float d1q, delta; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = grid_lut[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        r.d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        r.delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            sumi = ggml_cuda_dp4a(r.g[l0 + 0], get_int_b4(bq8_1[iqs].qs, l0 + 0), sumi);
            sumi = ggml_cuda_dp4a(r.g[l0 + 1], get_int_b4(bq8_1[iqs].qs, l0 + 1), sumi);
        }
        const float2 ds = __half22float2(bq8_1[iqs].ds);
        return r.d1q * (ds.x * sumi + ds.y * r.delta);
    }
};
