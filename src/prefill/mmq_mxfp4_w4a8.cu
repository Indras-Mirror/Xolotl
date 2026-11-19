// src/prefill/mmq_mxfp4_w4a8.cu - the tile width for the int8 (W4A8) MXFP4 MMQ instance in mmq_nvfp4_w4a8.cu.
//
// A 12xa build compiles mmq.cuh's MXFP4 instance as FP4 x FP4 (load_tiles_mxfp4_fp4 + the FP4 MMA), while moe_mmq.cu
// quantizes every product's activations as Q8_1, so the prompt path runs mmq_nvfp4_w4a8.cu's int8 MXFP4 instance
// (weights decoded to int8, Q8_1 activations; with the FP4 one, Unsloth's DeepSeek-V4.1-Flash-1017 UD-IQ3_XXS - 30
// of 40 layers MXFP4 downs - returned NaN on every MXFP4 layer).  moe_mmq.cu sees the FP4 MMA's config table, so the
// tile width it picks is from the wrong table for that instance; this TU hides Blackwell's FP4 MMA from mmq.cuh as
// mmq_nvfp4_w4a8.cu does (the host's tile config, blackwell_mma_available(cc)) and picks it from the int8 table.
#include "common.cuh"

#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_switch_J strata_mxfp4_w4a8_mul_mat_q_switch_J
#define mul_mat_q_case strata_mxfp4_w4a8_mul_mat_q_case
#include "mmq.cuh"

#include <cstdint>

namespace strata::prefill::mmq {
// moe_mmq.cu Context::run's tile choice (fewest tiles over J = 8..128 among the configs that fit shared memory),
// from the int8 config table mmq_nvfp4_w4a8.cu's MXFP4 instance launches with
int64_t mxfp4_w4a8_j_best(int64_t max_rows, int64_t w_rows) {
    const int dev = ggml_cuda_get_device(), cc = ggml_cuda_info().devices[dev].cc;
    const size_t smpbo = ggml_cuda_info().devices[dev].smpbo;
    const bool fallback = ggml_cuda_mmq_needs_fallback(w_rows);
    int64_t J_best = 0, ntiles_best = INT64_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(GGML_TYPE_MXFP4, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) continue;
        const int64_t nt = (max_rows + config.J - 1) / config.J;
        if (nt < ntiles_best) { J_best = J; ntiles_best = nt; }
    }
    return J_best;
}
}  // namespace strata::prefill::mmq
