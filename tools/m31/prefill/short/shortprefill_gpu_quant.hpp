// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
#include "shortprefill_cuda_math_cpu.hpp"
#include "shortprefill_cuda_nvfp4_cpu.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::m31 {
// Only activations visit CUDA, never expert weights. Calls use a private
// stream and reusable pinned/device buffers. They complete before returning
// and are serialized by the owning CPU executor. A negative device selects
// the calling thread's current CUDA device, captured before CPU submission.
class ShortPrefillGpuQuantizer {
public:
    struct Stats {
        uint64_t calls=0,h2d_bytes=0,d2h_bytes=0,reserve_calls=0,allocations=0,runtime_allocations=0;
        double wall_seconds=0,reserve_seconds=0,host_copy_seconds=0,submit_seconds=0,wait_seconds=0,unpack_seconds=0;
        size_t device_bytes=0,pinned_bytes=0,reserved_max_values=0,reserved_max_rows=0;
        int stream_priority=0;
    };
    static bool available() noexcept;
    explicit ShortPrefillGpuQuantizer(int device=-1);
    ~ShortPrefillGpuQuantizer();
    ShortPrefillGpuQuantizer(const ShortPrefillGpuQuantizer&)=delete;
    ShortPrefillGpuQuantizer& operator=(const ShortPrefillGpuQuantizer&)=delete;
    // Bound all future calls before launching expert-weight uploads: max_values
    // bounds columns*rows and max_rows bounds rows for norm/down/OAI inputs.
    // The padded packed-buffer bound covers every matrix satisfying both limits.
    // No device/pinned allocation occurs in calls within these reserved bounds.
    void reserve(size_t max_values,size_t max_rows);
    // Input [columns,rows], output flattened [columns/32,rows]. IQ3_XXS and
    // IQ4_XS select PR's F32 D4 scales. The native quantizer runs unmodified.
    void quantize_q8(ggml_type type,const float *input,size_t columns,size_t rows,
                     std::vector<cuda_math_cpu::Q8Mmq32> &output);
    void quantize_nvfp4(const float *input,size_t columns,size_t rows,
                       std::vector<cuda_nvfp4_cpu::Activation> &output);
    // gate/up have already had their external expert scales applied. Execute
    // the PR CUDA OAI expression before quantization, with identical fast-math
    // flags. This avoids a hidden F32 D2H/H2D round trip between the operations.
    void oai_quantize_q8(ggml_type type,const float *gate,const float *up,
                         size_t columns,size_t rows,std::vector<cuda_math_cpu::Q8Mmq32> &output);
    void oai_quantize_nvfp4(const float *gate,const float *up,size_t columns,size_t rows,
                           std::vector<cuda_nvfp4_cpu::Activation> &output);
    const Stats &stats() const noexcept;
private:
    struct Impl;std::unique_ptr<Impl> p_;
};
}
