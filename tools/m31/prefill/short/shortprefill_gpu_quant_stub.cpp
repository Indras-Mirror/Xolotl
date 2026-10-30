// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "shortprefill_gpu_quant.hpp"
#include <stdexcept>
namespace strata::m31 {
struct ShortPrefillGpuQuantizer::Impl {};
bool ShortPrefillGpuQuantizer::available() noexcept{return false;}
ShortPrefillGpuQuantizer::ShortPrefillGpuQuantizer(int){throw std::runtime_error("M31 GPU activation quantizer requires a CUDA build");}
ShortPrefillGpuQuantizer::~ShortPrefillGpuQuantizer()=default;
void ShortPrefillGpuQuantizer::reserve(size_t,size_t){throw std::runtime_error("CUDA quantizer unavailable");}
void ShortPrefillGpuQuantizer::quantize_q8(ggml_type,const float*,size_t,size_t,std::vector<cuda_math_cpu::Q8Mmq32>&){throw std::runtime_error("CUDA quantizer unavailable");}
void ShortPrefillGpuQuantizer::quantize_nvfp4(const float*,size_t,size_t,std::vector<cuda_nvfp4_cpu::Activation>&){throw std::runtime_error("CUDA quantizer unavailable");}
void ShortPrefillGpuQuantizer::oai_quantize_q8(ggml_type,const float*,const float*,size_t,size_t,std::vector<cuda_math_cpu::Q8Mmq32>&){throw std::runtime_error("CUDA quantizer unavailable");}
void ShortPrefillGpuQuantizer::oai_quantize_nvfp4(const float*,const float*,size_t,size_t,std::vector<cuda_nvfp4_cpu::Activation>&){throw std::runtime_error("CUDA quantizer unavailable");}
const ShortPrefillGpuQuantizer::Stats &ShortPrefillGpuQuantizer::stats() const noexcept{static const Stats empty;return empty;}
}
