// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
// CPU arithmetic matching the native Blackwell NVFP4 MMQ contract. This is a
// separate opt-in prefill implementation; CPU decode/Q8_K math is untouched.
// Source contract: ggml-cuda/quantize.cu::quantize_mmq_nvfp4 (CUDA >=12.8),
// mmq-vec-dot.cuh::ggml_cuda_mmq_vec_dot_fp4_fp4_mma. NVFP4 MMQ uses W4A4,
// irrespective of the BF16 accumulation preference used by the cuBLAS path.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
namespace strata::m31::cuda_nvfp4_cpu {
inline constexpr int8_t fp4_values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
inline const std::array<float,127> &fp8_half_table(){
    static const auto result=[](){std::array<float,127> t{};
        for(int c=1;c<127;++c){const int e=c>>3,m=c&7;t[size_t(c)]=e?std::ldexp(1.f+float(m)/8.f,e-8):std::ldexp(float(m),-10);}return t;}();return result;
}
inline float fp8_half(uint8_t code){return code<127?fp8_half_table()[code]:0.f;}
inline uint8_t fp8_code(float value){
    if(!(value>0))return 0;
    const auto &half=fp8_half_table();
    // __nv_fp8_e4m3 uses finite saturation and nearest-even. Scales here are
    // stored as raw E4M3; ggml's shared UE4M3 decoding table returns half of it.
    auto at=std::lower_bound(half.begin(),half.end(),value*.5f);
    if(at==half.end())return 126;if(at==half.begin())return 0;
    const int high=int(at-half.begin()),low=high-1;
    const float midpoint=half[size_t(low)]+half[size_t(high)];
    if(value<midpoint)return uint8_t(low);if(value>midpoint)return uint8_t(high);
    return uint8_t((low&1)?high:low);
}
inline uint8_t fp4_code(float value){
    static constexpr float boundary[7]={.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f};
    const float a=std::fabs(value);unsigned result=7;
    for(unsigned i=0;i<7;++i){if(a<boundary[i]){result=i;break;}
        if(a==boundary[i]){result=(i&1)?i+1:i;break;}}
    return uint8_t(result|(std::signbit(value)?8:0));
}
struct Activation {
    float row_scale=0;
    std::vector<uint8_t> scales; // one raw E4M3 code per16 values
    std::vector<int8_t> values;  // signed doubled E2M1 in ordinary column order
};
inline float scale_error(const float *values,float inv_row,float inv_sub,float sub_scale){
    float error=0;const float dequant=2.f*sub_scale;
    for(int i=0;i<16;++i){const float value=values[i]*inv_row;
        const float decoded=float(fp4_values[fp4_code(value*inv_sub)])*.5f;
        const float delta=std::fabs(value)-std::fabs(decoded)*dequant;
        error=std::fma(delta,delta,error);}
    return error;
}
inline void quantize(const float *input,size_t columns,Activation &output){
    if(!input||!columns||columns%64)throw std::invalid_argument("NVFP4 CPU input must have a nonzero multiple of64 columns");
    float maximum=0;for(size_t c=0;c<columns;++c){if(!std::isfinite(input[c]))throw std::invalid_argument("nonfinite NVFP4 CPU input");maximum=std::max(maximum,std::fabs(input[c]));}
    output.row_scale=maximum/(6.f*448.f);
    const float inv_row=output.row_scale>0?1.f/output.row_scale:0.f;
    output.scales.resize(columns/16);output.values.resize(columns);
    static constexpr int offsets[5]={0,-1,1,-2,2};
    for(size_t base=0;base<columns;base+=16){
        float amax=0;for(int i=0;i<16;++i)amax=std::max(amax,std::fabs(input[base+size_t(i)]*inv_row));
        const int first=int(fp8_code(amax/6.f));uint8_t best_code=uint8_t(first);
        float sub_scale=fp8_half(best_code);
        float best=scale_error(input+base,inv_row,sub_scale>0?.5f/sub_scale:0.f,sub_scale);
        for(int candidate=1;candidate<5;++candidate){const int code=first+offsets[candidate];if(code<0||code>126)continue;
            const float scale=fp8_half(uint8_t(code));
            const float error=scale_error(input+base,inv_row,scale>0?.5f/scale:0.f,scale);
            if(error<best){best=error;best_code=uint8_t(code);sub_scale=scale;}}
        output.scales[base/16]=best_code;
        // The final native constructor multiplies once by the combined scale;
        // candidate-error evaluation above uses two separate multiplications.
        const float combined=inv_row*(sub_scale>0?.5f/sub_scale:0.f);
        for(size_t i=0;i<16;++i)output.values[base+i]=fp4_values[fp4_code(input[base+i]*combined)];
    }
}
inline int integer_dot_scalar(const uint8_t *weight,const int8_t *activation){
    int result=0;for(size_t i=0;i<8;++i){result+=int(fp4_values[weight[i]&15])*int(activation[i]);result+=int(fp4_values[weight[i]>>4])*int(activation[i+8]);}return result;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("ssse3"))) inline int integer_dot_simd(const uint8_t *weight,const int8_t *activation){
    const __m128i packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(weight));
    const __m128i low=_mm_and_si128(packed,_mm_set1_epi8(15));
    const __m128i high=_mm_and_si128(_mm_srli_epi16(packed,4),_mm_set1_epi8(15));
    const __m128i indices=_mm_unpacklo_epi64(low,high);
    const __m128i lookup=_mm_loadu_si128(reinterpret_cast<const __m128i*>(fp4_values));
    const __m128i values=_mm_shuffle_epi8(lookup,indices);
    const __m128i a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(activation));
    const __m128i pairs=_mm_maddubs_epi16(_mm_abs_epi8(values),_mm_sign_epi8(a,values));
    __m128i sums=_mm_madd_epi16(pairs,_mm_set1_epi16(1));
    sums=_mm_hadd_epi32(sums,sums);sums=_mm_hadd_epi32(sums,sums);return _mm_cvtsi128_si32(sums);
}
#endif
inline float dot(const void *weight_row,const Activation &activation,size_t columns){
    if(!weight_row||!columns||columns%64||activation.values.size()!=columns||activation.scales.size()!=columns/16)
        throw std::invalid_argument("invalid NVFP4 CPU dot geometry");
    const auto *weight=static_cast<const uint8_t*>(weight_row);float result=0;
#if defined(__x86_64__) || defined(__i386__)
    static const bool simd=__builtin_cpu_supports("ssse3");
#endif
    for(size_t base=0;base<columns;base+=64,weight+=36){
        for(size_t sub=0;sub<4;++sub){const auto *q=weight+4+sub*8;const auto *a=activation.values.data()+base+sub*16;
            int integer;
#if defined(__x86_64__) || defined(__i386__)
            integer=simd?integer_dot_simd(q,a):integer_dot_scalar(q,a);
#else
            integer=integer_dot_scalar(q,a);
#endif
            const float scale=fp8_half(weight[sub])*fp8_half(activation.scales[base/16+sub]);
            result=std::fma(float(integer),scale,result);}}
    return result*activation.row_scale;
}
}
