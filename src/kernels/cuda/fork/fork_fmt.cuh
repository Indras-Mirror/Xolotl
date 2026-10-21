// src/kernels/cuda/fork/fork_fmt.cuh - fork formats in the grouped kernels: Fmt<10/11/39> and the STRATA_*_FMTS_FORK lists.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

template<> struct Fmt<10> { static constexpr int qk = 256, ipb = QI2_K / VDR_Q2_K, step = VDR_Q2_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<11> { static constexpr int qk = 256, ipb = QI3_K / VDR_Q3_K, step = VDR_Q3_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q3_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<39> { static constexpr int qk = 32, ipb = QI_MXFP4 / VDR_MXFP4, step = VDR_MXFP4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_mxfp4_q8_1(v, y, kbx, iqs); } };

// PTQ1_0 (143): the whole 128-weight block per call (4 q8_1 sub-blocks), so ipb 1 and iqs always 0.
#ifndef STRATA_PTQ1_QUARTER   // the whole-block dot (llama-tq's) - default until the quarter split is measured
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 1, step = 0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1(v, y, kbx, iqs); } };
#else                         // -DSTRATA_PTQ1_QUARTER: a quarter block per call (fork_vecdot.cuh), 4x the calls per row, 1/4 the chain (UNMEASURED)
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 4, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1_quarter(v, y, kbx, iqs); } };
#endif
// Fork formats, kept out of upstream's lists above so those stay verbatim (fewer merge conflicts): Q2_K (10,
// DeepSeek-V4 down), Q3_K (11, MiMo GSQ-RCO / V4.1 tier), MXFP4 (39, MiMo outliers), PTQ1_0 (143, V4.1 ternary tier).  Each use site expands the _FORK list too.
//
// UNION (xolotl): GLM's branch had this list INLINE in its own iq_kernels.cu, with three entries the fork's
// extraction never picked up - Q4_K (12, GLM-5.3-Flash layers 3-5 down_exps; upstream's list has none) and
// IQ2_S/IQ3_XXS (18/22, the Maya-S-v2 down_exps).  The merge reconciled GLM's inline copy to this header and
// dropped them, so d_qk(18) == 0 and native_expert_grouped rejected the model at tier init:
//   "tier init: native_expert_grouped has no kernel for layer 3's types 16/16/18" (16/16 = IQ2_XXS gate/up,
// 18 = IQ2_S down) - i.e. GLM-5.3 could not load at all, while the DS4/DSV4.1 types were fine.  Measured
// against the working tree: ~/AI/Strata-GLM/build-glm-gpu/glm_generate passes the same point and prints
// "moe lora: routed-expert deltas on 26 layer(s) (n_embd 4096, n_ff 2048)".  Fmt<12>/Fmt<18>/Fmt<22> are all
// defined in this build, so this is the whole fix (see HANDOFF.md §6).
// MXFP4 (39) on gate/up too: Qwen3.8-Flash-Next-MXFP4 (all three projections MXFP4); Fmt<39> and the plain
// per-column kernel were already there for the MiMo down outliers, only the gate/up list did not name it, so
// native_expert_supported refused the file ("layer 0's experts are MXFP4/MXFP4 ..., no GPU kernels for")
#define STRATA_GU_FMTS_FORK(X) X(10) X(11) X(39) X(143)
#define STRATA_D_FMTS_FORK(X) X(10) X(11) X(12) X(18) X(22) X(39) X(143)
#define STRATA_MMVQ_FMTS_FORK(X) X(10) X(143)
