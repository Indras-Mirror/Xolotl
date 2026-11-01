// CUDA-free grouped-expert format registry, shared by kernels and CPU admission.
#pragma once
#include <cstdint>

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#ifdef STRATA_Q6K_EXPERTS   // opt-in build (-DSTRATA_Q6K_EXPERTS=ON): one more instance per kernel, loaded at start
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(14) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(147) X(148) X(149)
#else
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(147) X(148) X(149)
#endif
// Q5_K / Q6_K downs: Qwen3.6-35B-A3B (FF 512, whole 256-blocks per row); Flash-Next's FF 640 never selects them
#ifdef STRATA_Q6K_EXPERTS
#define STRATA_D_FMTS(X) X(20) X(21) X(23) X(42) X(7) X(6) X(2) X(3) X(8) X(13) X(14) X(40) X(144) X(146) X(147) X(148) X(149)
#else
#define STRATA_D_FMTS(X) X(20) X(21) X(23) X(42) X(7) X(6) X(2) X(3) X(8) X(13) X(40) X(144) X(146) X(147) X(148) X(149)
#endif
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(146) X(147) X(148) X(149)
// IQ1_S (19) gate/up and down: Unsloth's Dynamic V3.1 DeepSeek-V4.1 GGUFs (UD-IQ1_M: IQ1_S gate/up on 20 of 40 layers,
// IQ1_S down on 5; UD-IQ2_XXS: IQ1_S gate/up on 10) - Fmt<19> / Split<19> in src/kernels/cuda/fork/
#define STRATA_GU_FMTS_FORK(X) X(10) X(11) X(19) X(39) X(143)
#define STRATA_D_FMTS_FORK(X) X(10) X(11) X(12) X(16) X(17) X(18) X(19) X(22) X(29) X(39) X(143)
#define STRATA_MMVQ_FMTS_FORK(X) X(10) X(143)

namespace strata::kernels::registry {
constexpr int block_size(int type) noexcept {
    switch (type) {
        case 2: case 3: case 6: case 7: case 8: case 20: return 32;
        case 39: return 32;
        case 40: case 42: return 64;
        case 143: return 128;
        case 144: case 145: case 146: case 147: case 148: case 149: return 128;   // APR / Gyro-L trellis (upstream)
        case 10: case 11: case 12: case 13: case 14: case 16: case 17:
        case 18: case 19: case 21: case 22: case 23: case 29: return 256;
        default: return 0;
    }
}
constexpr int gate_block(int type) noexcept {
    switch (type) {
#define STRATA_REG_CASE(T) case T: return block_size(T);
        STRATA_GU_FMTS(STRATA_REG_CASE)
        STRATA_GU_FMTS_FORK(STRATA_REG_CASE)
#undef STRATA_REG_CASE
        default: return 0;
    }
}
constexpr int down_block(int type) noexcept {
    switch (type) {
#define STRATA_REG_CASE(T) case T: return block_size(T);
        STRATA_D_FMTS(STRATA_REG_CASE)
        STRATA_D_FMTS_FORK(STRATA_REG_CASE)
#undef STRATA_REG_CASE
        default: return 0;
    }
}
constexpr bool supported_pair(int gate, int down, int64_t width, int64_t ff) noexcept {
    const int qg=gate_block(gate), qd=down_block(down);
    // width%256 also implies the native product%256 check, without overflow.
    return width>0 && ff>0 && qg>0 && qd>0 && width%qg==0 && ff%qd==0 && width%256==0;
}
constexpr bool supported_triple(int gate,int up,int down,int64_t width,int64_t ff) noexcept {
    if(gate==up)return supported_pair(gate,down,width,ff);
    const bool mixed=(gate==10 && up==11)||(gate==11 && up==10);
    return mixed && supported_pair(gate,down,width,ff) && supported_pair(up,down,width,ff) && ff%8==0;
}
} // namespace strata::kernels::registry
