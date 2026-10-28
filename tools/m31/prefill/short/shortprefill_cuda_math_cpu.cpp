// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "shortprefill_cuda_math_cpu.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace strata::m31::cuda_math_cpu {
namespace {
void check(bool value, const char * why) {
    if (!value) throw std::invalid_argument(why);
}
int32_t dot32_scalar(const int8_t * weight, const int8_t * activation) {
    int32_t sum = 0;
    for (size_t k = 0; k < 32; ++k) sum += int32_t(weight[k]) * int32_t(activation[k]);
    return sum;
}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
constexpr std::array<uint64_t, 256> make_signs8() {
    std::array<uint64_t, 256> result{};
    for (size_t mask = 0; mask < result.size(); ++mask)
        for (size_t lane = 0; lane < 8; ++lane)
            if (mask & (size_t(1) << lane)) result[mask] |= uint64_t(255) << (8 * lane);
    return result;
}
constexpr auto signs8 = make_signs8();

// Same integer values/scales as the scalar public loader; x86 little-endian
// SIMD only changes the lookup/sign expansion, not any FP operation.
__attribute__((target("avx2,f16c"), always_inline))
inline __m256i decode32_avx2(ggml_type type, const void * source, size_t group, float & scale) {
    const size_t block = group / 8, sub = group % 8;
    if (type == GGML_TYPE_IQ3_XXS) {
        const auto & weight = static_cast<const block_iq3_xxs *>(source)[block];
        uint32_t aux;
        std::memcpy(&aux, weight.qs + 64 + sub * 4, sizeof(aux));
        const auto * code = weight.qs + sub * 8;
        // Eight independent L1 loads outperform AVX2 gather on EPYC 9654.
        const __m256i positive = _mm256_setr_epi32(
            int32_t(iq3xxs_grid[code[0]]), int32_t(iq3xxs_grid[code[1]]),
            int32_t(iq3xxs_grid[code[2]]), int32_t(iq3xxs_grid[code[3]]),
            int32_t(iq3xxs_grid[code[4]]), int32_t(iq3xxs_grid[code[5]]),
            int32_t(iq3xxs_grid[code[6]]), int32_t(iq3xxs_grid[code[7]]));
        const __m256i negative = _mm256_set_epi64x(
            int64_t(signs8[ksigns_iq2xs[(aux >> 21) & 127]]),
            int64_t(signs8[ksigns_iq2xs[(aux >> 14) & 127]]),
            int64_t(signs8[ksigns_iq2xs[(aux >> 7) & 127]]),
            int64_t(signs8[ksigns_iq2xs[aux & 127]]));
        const __m256i decoded = _mm256_sub_epi8(_mm256_xor_si256(positive, negative), negative);
        const float d = _cvtsh_ss(weight.d);
        scale = (float(aux >> 28) * d + d / 2.f) / 2.f;
        return decoded;
    } else {
        const auto & weight = static_cast<const block_iq4_xs *>(source)[block];
        const __m128i code = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight.qs + sub * 16));
        const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
        const __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(code, _mm_set1_epi8(15)));
        const __m128i high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(code, 4), _mm_set1_epi8(15)));
        const int ls = ((weight.scales_l[sub / 2] >> (4 * (sub % 2))) & 15) |
                       (((weight.scales_h >> (2 * sub)) & 3) << 4);
        scale = _cvtsh_ss(weight.d) * float(ls - 32);
        return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    }
}

__attribute__((target("avx2,f16c"), always_inline))
inline void unpack32_avx2(ggml_type type, const void * source, size_t group, int8_t * values, float & scale) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(values), decode32_avx2(type, source, group, scale));
}

// Transpose eight rows of eight packed int32 values. Two such transposes
// replace 128 scalar stores when arranging a 16-row VNNI weight tile.
__attribute__((target("avx2"), always_inline))
inline void transpose8(const __m256i * rows, __m256i * output) {
    const __m256i t0=_mm256_unpacklo_epi32(rows[0],rows[1]),t1=_mm256_unpackhi_epi32(rows[0],rows[1]);
    const __m256i t2=_mm256_unpacklo_epi32(rows[2],rows[3]),t3=_mm256_unpackhi_epi32(rows[2],rows[3]);
    const __m256i t4=_mm256_unpacklo_epi32(rows[4],rows[5]),t5=_mm256_unpackhi_epi32(rows[4],rows[5]);
    const __m256i t6=_mm256_unpacklo_epi32(rows[6],rows[7]),t7=_mm256_unpackhi_epi32(rows[6],rows[7]);
    const __m256i s0=_mm256_unpacklo_epi64(t0,t2),s1=_mm256_unpackhi_epi64(t0,t2);
    const __m256i s2=_mm256_unpacklo_epi64(t1,t3),s3=_mm256_unpackhi_epi64(t1,t3);
    const __m256i s4=_mm256_unpacklo_epi64(t4,t6),s5=_mm256_unpackhi_epi64(t4,t6);
    const __m256i s6=_mm256_unpacklo_epi64(t5,t7),s7=_mm256_unpackhi_epi64(t5,t7);
    output[0]=_mm256_permute2x128_si256(s0,s4,0x20);output[1]=_mm256_permute2x128_si256(s1,s5,0x20);
    output[2]=_mm256_permute2x128_si256(s2,s6,0x20);output[3]=_mm256_permute2x128_si256(s3,s7,0x20);
    output[4]=_mm256_permute2x128_si256(s0,s4,0x31);output[5]=_mm256_permute2x128_si256(s1,s5,0x31);
    output[6]=_mm256_permute2x128_si256(s2,s6,0x31);output[7]=_mm256_permute2x128_si256(s3,s7,0x31);
}

__attribute__((target("avx2")))
int32_t dot32_avx2(const int8_t * weight, const int8_t * activation) {
    const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight));
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation));
    // IQ3/IQ4 weights are in [-127,127], as are the quantized activations.
    // Each adjacent pair is <=2*127*127, so maddubs cannot saturate.
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
    const __m256i groups = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(groups), _mm256_extracti128_si256(groups, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2")))
bool has_min32_avx2(const int8_t * activation) {
    const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation));
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(values, _mm256_set1_epi8(int8_t(-128)))) != 0;
}

__attribute__((target("avx2")))
bool tile_has_min_avx2(const Q8Mmq32 * const * activations, size_t tokens, size_t groups) {
    for (size_t token = 0; token < tokens; ++token)
        for (size_t group = 0; group < groups; ++group)
            if (has_min32_avx2(activations[token][group].qs)) return true;
    return false;
}

__attribute__((target("avx2,fma,f16c")))
void tile_avx2(ggml_type type, const void * weights, size_t weight_row_stride,
               size_t rows, size_t columns, const Q8Mmq32 * const * activations,
               size_t tokens, float * output, size_t output_token_stride) {
    for (size_t row = 0; row < rows; ++row) {
        const auto * source = static_cast<const uint8_t *>(weights) + row * weight_row_stride;
        float sums[16]{};
        for (size_t group = 0; group < columns / 32; ++group) {
            int8_t decoded[32]; float scale;
            unpack32_avx2(type, source, group, decoded, scale);
            for (size_t token = 0; token < tokens; ++token) {
                const auto & activation = activations[token][group];
                const int32_t integer_sum = dot32_avx2(decoded, activation.qs);
                const __m128 scaled_integer = _mm_set_ss(float(integer_sum) * scale);
                const __m128 result = _mm_fmadd_ss(scaled_integer,
                    _mm_set_ss(activation.d), _mm_set_ss(sums[token]));
                sums[token] = _mm_cvtss_f32(result);
            }
        }
        for (size_t token = 0; token < tokens; ++token)
            output[token * output_token_stride + row] = sums[token];
    }
}

// Sixteen independent output rows fill the sixteen VNNI int32 lanes. Four
// weight bytes per row are transposed into each dword; eight dpbusd operations
// produce all sixteen exact signed dot32 values without horizontal reductions.
// The +128 shift of the weights is removed before converting integers to F32.
template <size_t Tokens>
__attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
void tile_vnni(ggml_type type, const void * weights, size_t weight_row_stride,
               size_t rows, size_t columns, const Q8Mmq32 * const * activations,
               float * output, size_t output_token_stride) {
    for (size_t row_start = 0; row_start < rows; row_start += 16) {
        const size_t valid_rows = std::min(size_t(16), rows - row_start);
        const __mmask16 row_mask = __mmask16((uint32_t(1) << valid_rows) - 1);
        __m512 sums[Tokens];
#pragma GCC unroll 16
        for (size_t token = 0; token < Tokens; ++token) sums[token] = _mm512_setzero_ps();
        for (size_t group = 0; group < columns / 32; ++group) {
            alignas(64) uint32_t packed[8][16];
            alignas(64) float scales[16]{};
            for (size_t lane = 0; lane < valid_rows; ++lane) {
                int8_t decoded[32];
                const auto * source = static_cast<const uint8_t *>(weights) + (row_start + lane) * weight_row_stride;
                unpack32_avx2(type, source, group, decoded, scales[lane]);
                for (size_t quad = 0; quad < 8; ++quad) {
                    uint32_t word;
                    std::memcpy(&word, decoded + quad * 4, sizeof(word));
                    packed[quad][lane] = word ^ 0x80808080u;
                }
            }
            for (size_t lane = valid_rows; lane < 16; ++lane)
                for (size_t quad = 0; quad < 8; ++quad) packed[quad][lane] = 0x80808080u;
            __m512i weight[8];
#pragma GCC unroll 8
            for (size_t quad = 0; quad < 8; ++quad)
                weight[quad] = _mm512_load_si512(packed[quad]);
            const __m512 weight_scale = _mm512_load_ps(scales);
#pragma GCC unroll 16
            for (size_t token = 0; token < Tokens; ++token) {
                const auto & activation = activations[token][group];
                __m512i integer = _mm512_setzero_si512();
#pragma GCC unroll 8
                for (size_t quad = 0; quad < 8; ++quad) {
                    uint32_t word;
                    std::memcpy(&word, activation.qs + quad * 4, sizeof(word));
                    integer = _mm512_dpbusd_epi32(integer, weight[quad], _mm512_set1_epi32(int32_t(word)));
                }
                const __m256i aq = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation.qs));
                const __m256i pair_sums = _mm256_maddubs_epi16(_mm256_set1_epi8(1), aq);
                const __m256i quad_sums = _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1));
                __m128i act_sum = _mm_add_epi32(_mm256_castsi256_si128(quad_sums), _mm256_extracti128_si256(quad_sums, 1));
                act_sum = _mm_hadd_epi32(act_sum, act_sum);
                act_sum = _mm_hadd_epi32(act_sum, act_sum);
                integer = _mm512_sub_epi32(integer, _mm512_set1_epi32(128 * _mm_cvtsi128_si32(act_sum)));
                const __m512 partial = _mm512_mul_ps(_mm512_cvtepi32_ps(integer), weight_scale);
                sums[token] = _mm512_fmadd_ps(partial, _mm512_set1_ps(activation.d), sums[token]);
            }
        }
#pragma GCC unroll 16
        for (size_t token = 0; token < Tokens; ++token)
            _mm512_mask_storeu_ps(output + token * output_token_stride + row_start, row_mask, sums[token]);
    }
}

__attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
void tile_vnni_dispatch(ggml_type type, const void * weights, size_t stride, size_t rows, size_t columns,
                         const Q8Mmq32 * const * activations, size_t tokens, float * output, size_t output_stride) {
#define M31_VNNI_CASE(N) case N: tile_vnni<N>(type, weights, stride, rows, columns, activations, output, output_stride); break
    switch (tokens) {
        M31_VNNI_CASE(1); M31_VNNI_CASE(2); M31_VNNI_CASE(3); M31_VNNI_CASE(4);
        M31_VNNI_CASE(5); M31_VNNI_CASE(6); M31_VNNI_CASE(7); M31_VNNI_CASE(8);
        M31_VNNI_CASE(9); M31_VNNI_CASE(10); M31_VNNI_CASE(11); M31_VNNI_CASE(12);
        M31_VNNI_CASE(13); M31_VNNI_CASE(14); M31_VNNI_CASE(15); M31_VNNI_CASE(16);
    }
#undef M31_VNNI_CASE
}

// IQ3 benefits from decoding one K32 group at a time. IQ4's nibble layout
// benefits from decoding the complete K256 block row-first. Both variants
// consume the decoded groups in the same ascending K32 FMA order.
template <size_t Tokens, bool Prepared, size_t Groups>
__attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
void tile_vnni_transpose(ggml_type type, const void * weights, size_t stride,
                         size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                         const int32_t * const * corrections, float * output, size_t output_stride) {
    static_assert(Groups==1 || Groups==8);
    for(size_t row_start=0;row_start<rows;row_start+=16) {
        const size_t valid=std::min(size_t(16),rows-row_start);
        const __mmask16 mask=__mmask16((uint32_t(1)<<valid)-1);
        __m512 sums[Tokens];
#pragma GCC unroll 16
        for(size_t token=0;token<Tokens;++token)sums[token]=_mm512_setzero_ps();
        for(size_t first_group=0;first_group<columns/32;first_group+=Groups) {
            alignas(64) float block_scales[Groups][16]{};
            __m256i block_decoded[Groups][16];
            for(size_t row=0;row<valid;++row) {
                const auto *source=static_cast<const uint8_t *>(weights)+(row_start+row)*stride;
                for(size_t sub=0;sub<Groups;++sub)
                    block_decoded[sub][row]=decode32_avx2(type,source,first_group+sub,block_scales[sub][row]);
            }
            for(size_t row=valid;row<16;++row)
                for(size_t sub=0;sub<Groups;++sub)block_decoded[sub][row]=_mm256_setzero_si256();
            for(size_t sub=0;sub<Groups;++sub) {
                const size_t group=first_group+sub;
                __m256i low[8],high[8];
                transpose8(block_decoded[sub],low);transpose8(block_decoded[sub]+8,high);
                __m512i weight[8];
#pragma GCC unroll 8
                for(size_t quad=0;quad<8;++quad)
                    weight[quad]=_mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(low[quad]),high[quad],1),
                                                 _mm512_set1_epi32(int32_t(0x80808080u)));
                const __m512 weight_scale=_mm512_load_ps(block_scales[sub]);
#pragma GCC unroll 16
                for(size_t token=0;token<Tokens;++token) {
                    const auto &activation=activations[token][group];
                    __m512i integer=_mm512_setzero_si512();
#pragma GCC unroll 8
                    for(size_t quad=0;quad<8;++quad) {
                        uint32_t word;std::memcpy(&word,activation.qs+4*quad,sizeof(word));
                        integer=_mm512_dpbusd_epi32(integer,weight[quad],_mm512_set1_epi32(int32_t(word)));
                    }
                    int32_t correction;
                    if constexpr(Prepared)correction=corrections[token][group];
                    else {
                        const __m256i aq=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation.qs));
                        const __m256i pairs=_mm256_maddubs_epi16(_mm256_set1_epi8(1),aq);
                        const __m256i quads=_mm256_madd_epi16(pairs,_mm256_set1_epi16(1));
                        __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(quads),_mm256_extracti128_si256(quads,1));
                        sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);
                        correction=128*_mm_cvtsi128_si32(sum);
                    }
                    integer=_mm512_sub_epi32(integer,_mm512_set1_epi32(correction));
                    const __m512 partial=_mm512_mul_ps(_mm512_cvtepi32_ps(integer),weight_scale);
                    sums[token]=_mm512_fmadd_ps(partial,_mm512_set1_ps(activation.d),sums[token]);
                }
            }
        }
#pragma GCC unroll 16
        for(size_t token=0;token<Tokens;++token)
            _mm512_mask_storeu_ps(output+token*output_stride+row_start,mask,sums[token]);
    }
}

template <bool Prepared, size_t Groups>
__attribute__((target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))
void tile_transpose_dispatch(ggml_type type,const void *weights,size_t stride,size_t rows,size_t columns,
                              const Q8Mmq32 * const *activations,const int32_t * const *corrections,
                              size_t tokens,float *output,size_t output_stride) {
#define M31_TRANSPOSE_CASE(N) case N: tile_vnni_transpose<N,Prepared,Groups>(type,weights,stride,rows,columns,activations,corrections,output,output_stride); break
    switch(tokens) {
        M31_TRANSPOSE_CASE(1);M31_TRANSPOSE_CASE(2);M31_TRANSPOSE_CASE(3);M31_TRANSPOSE_CASE(4);
        M31_TRANSPOSE_CASE(5);M31_TRANSPOSE_CASE(6);M31_TRANSPOSE_CASE(7);M31_TRANSPOSE_CASE(8);
        M31_TRANSPOSE_CASE(9);M31_TRANSPOSE_CASE(10);M31_TRANSPOSE_CASE(11);M31_TRANSPOSE_CASE(12);
        M31_TRANSPOSE_CASE(13);M31_TRANSPOSE_CASE(14);M31_TRANSPOSE_CASE(15);M31_TRANSPOSE_CASE(16);
    }
#undef M31_TRANSPOSE_CASE
}
#endif
}

bool supported(ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS;
}

void quantize_mmq_d4(const float * input, size_t columns, std::vector<Q8Mmq32> & output) {
    check(input && columns && columns % 32 == 0, "M31 CUDA CPU math: invalid quantization shape");
    output.resize(columns / 32);
    quantize_mmq_d4_into(input, columns, output.data());
}

void quantize_mmq_d4_into(const float * input, size_t columns, Q8Mmq32 * output) {
    check(input && output && columns && columns % 32 == 0, "M31 CUDA CPU math: invalid quantization shape");
    for (size_t group = 0; group < columns / 32; ++group) {
        const float * row = input + group * 32;
        float amax = 0;
        for (size_t k = 0; k < 32; ++k) {
            check(std::isfinite(row[k]), "M31 CUDA CPU math: nonfinite activation");
            amax = std::max(amax, std::abs(row[k]));
        }
        auto & result = output[group];
        if (amax == 0) {
            result.d = 0;
            std::fill(std::begin(result.qs), std::end(result.qs), int8_t(0));
            continue;
        }
        // Preserve CUDA's two divisions and roundf (ties away from zero),
        // rather than CPU Q8_K's scale and nearest-even conversion.
        const float d_inv = 127.f / amax;
        result.d = 1.f / d_inv;
        for (size_t k = 0; k < 32; ++k)
            result.qs[k] = int8_t(std::round(row[k] * d_inv));
    }
}

int32_t integer_dot32(const int8_t * weight, const int8_t * activation) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static const bool avx2 = __builtin_cpu_supports("avx2");
    // sign_epi8(-128,-1) wraps. Public callers may provide the full int8
    // domain even though our activation quantizer produces only [-127,127].
    if (avx2 && !has_min32_avx2(activation)) return dot32_avx2(weight, activation);
#endif
    return dot32_scalar(weight, activation);
}

void unpack_weight32(ggml_type type, const void * weight_row, size_t group,
                     int8_t * values, float & scale) {
    check(supported(type) && weight_row && values, "M31 CUDA CPU math: unsupported weight type/pointer");
    const size_t block = group / 8, sub = group % 8;
    if (type == GGML_TYPE_IQ3_XXS) {
        const auto & weight = static_cast<const block_iq3_xxs *>(weight_row)[block];
        uint32_t aux;
        std::memcpy(&aux, weight.qs + 64 + sub * 4, sizeof(aux));
        const float d = ggml_fp16_to_fp32(weight.d);
        const int ls = int(aux >> 28);
        scale = (float(ls) * d + d / 2.f) / 2.f;
        for (size_t part = 0; part < 4; ++part) {
            const uint8_t signs = ksigns_iq2xs[(aux >> (7 * part)) & 127];
            for (size_t half = 0; half < 2; ++half) {
                const uint32_t grid = iq3xxs_grid[weight.qs[sub * 8 + part * 2 + half]];
                for (size_t lane = 0; lane < 4; ++lane) {
                    const int value = int((grid >> (8 * lane)) & 255);
                    const size_t position = half * 4 + lane;
                    values[part * 8 + position] = int8_t(signs & (1u << position) ? -value : value);
                }
            }
        }
    } else {
        const auto & weight = static_cast<const block_iq4_xs *>(weight_row)[block];
        const int ls = ((weight.scales_l[sub / 2] >> (4 * (sub % 2))) & 15) |
                       (((weight.scales_h >> (2 * sub)) & 3) << 4);
        scale = ggml_fp16_to_fp32(weight.d) * float(ls - 32);
        for (size_t k = 0; k < 16; ++k) {
            const uint8_t code = weight.qs[sub * 16 + k];
            values[k] = kvalues_iq4nl[code & 15];
            values[k + 16] = kvalues_iq4nl[code >> 4];
        }
    }
}

float dot_mmq(ggml_type type, const void * weight_row, size_t columns,
              const Q8Mmq32 * activation) {
    check(supported(type) && weight_row && activation && columns && columns % 256 == 0,
          "M31 CUDA CPU math: invalid dot shape/type");
    float sum = 0;
    for (size_t group = 0; group < columns / 32; ++group) {
        int8_t decoded[32];
        float scale;
        unpack_weight32(type, weight_row, group, decoded, scale);
        const int32_t integer_sum = integer_dot32(decoded, activation[group].qs);
        // Native NVIDIA mma.cuh expression is sum += C.x * dA * dB.
        // Keep the first product rounded before the final multiply-add.
        const float scaled_integer = float(integer_sum) * scale;
        sum = std::fma(scaled_integer, activation[group].d, sum);
    }
    return sum;
}

void mul_mat_quantized_tile(ggml_type type, const void * weights, size_t weight_row_stride,
                            size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                            size_t tokens, float * output, size_t output_token_stride,
                            const int32_t * const * corrections) {
    check(supported(type) && weights && activations && output && rows && columns &&
          columns % 256 == 0 && tokens > 0 && tokens <= 16 && output_token_stride >= rows &&
          weight_row_stride >= ggml_row_size(type, int64_t(columns)),
          "M31 CUDA CPU math: invalid quantized tile");
    for (size_t token = 0; token < tokens; ++token) {
        check(activations[token], "M31 CUDA CPU math: missing tile activation");
        if(corrections)check(corrections[token],"M31 CUDA CPU math: missing correction row");
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static const bool vnni = __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw") &&
                              __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
    if (vnni) {
        if(type==GGML_TYPE_IQ4_XS) {
            if(corrections)tile_transpose_dispatch<true,8>(type,weights,weight_row_stride,rows,columns,activations,corrections,tokens,output,output_token_stride);
            else tile_transpose_dispatch<false,8>(type,weights,weight_row_stride,rows,columns,activations,nullptr,tokens,output,output_token_stride);
        } else {
            if(corrections)tile_transpose_dispatch<true,1>(type,weights,weight_row_stride,rows,columns,activations,corrections,tokens,output,output_token_stride);
            else tile_transpose_dispatch<false,1>(type,weights,weight_row_stride,rows,columns,activations,nullptr,tokens,output,output_token_stride);
        }
        return;
    }
#endif
    mul_mat_quantized_tile_avx2_reference(type, weights, weight_row_stride, rows, columns, activations,
                                         tokens, output, output_token_stride);
}

void mul_mat_quantized_tile_vnni_reference(ggml_type type,const void *weights,size_t weight_row_stride,
                                          size_t rows,size_t columns,const Q8Mmq32 * const *activations,
                                          size_t tokens,float *output,size_t output_token_stride) {
    check(supported(type)&&weights&&activations&&output&&rows&&columns&&columns%256==0&&tokens>0&&tokens<=16&&
          output_token_stride>=rows&&weight_row_stride>=ggml_row_size(type,int64_t(columns)),
          "M31 CUDA CPU math: invalid legacy VNNI reference tile");
    for(size_t token=0;token<tokens;++token)check(activations[token],"M31 CUDA CPU math: missing legacy VNNI activation");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static const bool vnni=__builtin_cpu_supports("avx512vnni")&&__builtin_cpu_supports("avx512bw")&&
                            __builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("f16c")&&__builtin_cpu_supports("fma");
    if(vnni){tile_vnni_dispatch(type,weights,weight_row_stride,rows,columns,activations,tokens,output,output_token_stride);return;}
#endif
    mul_mat_quantized_tile_avx2_reference(type,weights,weight_row_stride,rows,columns,activations,tokens,output,output_token_stride);
}

void mul_mat_quantized_tile_avx2_reference(ggml_type type, const void * weights, size_t weight_row_stride,
                                          size_t rows, size_t columns, const Q8Mmq32 * const * activations,
                                          size_t tokens, float * output, size_t output_token_stride) {
    check(supported(type) && weights && activations && output && rows && columns && columns % 256 == 0 &&
          tokens > 0 && tokens <= 16 && output_token_stride >= rows &&
          weight_row_stride >= ggml_row_size(type, int64_t(columns)), "M31 CUDA CPU math: invalid AVX2 reference tile");
    for (size_t token = 0; token < tokens; ++token)
        check(activations[token], "M31 CUDA CPU math: missing reference tile activation");
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static const bool avx2_fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                                 __builtin_cpu_supports("f16c");
    if (avx2_fma && !tile_has_min_avx2(activations, tokens, columns / 32)) {
        tile_avx2(type, weights, weight_row_stride, rows, columns, activations, tokens, output, output_token_stride);
        return;
    }
#endif
    for (size_t row = 0; row < rows; ++row) for (size_t token = 0; token < tokens; ++token)
        output[token * output_token_stride + row] = dot_mmq(type,
            static_cast<const uint8_t *>(weights) + row * weight_row_stride, columns, activations[token]);
}

void mul_mat_rows(ggml_type type, const void * weights, size_t weight_row_stride,
                  size_t rows, size_t columns, const float * x, size_t tokens,
                  std::vector<float> & output) {
    check(supported(type) && weights && x && rows && tokens && columns && columns % 256 == 0 &&
          weight_row_stride >= ggml_row_size(type, int64_t(columns)) &&
          rows <= std::numeric_limits<size_t>::max() / tokens,
          "M31 CUDA CPU math: invalid matrix shape/type");
    output.resize(rows * tokens);
    std::vector<Q8Mmq32> activation;
    for (size_t token = 0; token < tokens; ++token) {
        quantize_mmq_d4(x + token * columns, columns, activation);
        for (size_t row = 0; row < rows; ++row)
            output[token * rows + row] = dot_mmq(type,
                static_cast<const uint8_t *>(weights) + row * weight_row_stride,
                columns, activation.data());
    }
}
} // namespace strata::m31::cuda_math_cpu
