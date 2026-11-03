// Compile-only regression for the shared int4 arrays in iq_kernels.cu's
// expdec gate/up and down kernels. Never launches a kernel or loads a model.
//
// HIP 7.0.2/gfx1100 (no GPU required):
//   /opt/rocm-7.0.2/llvm/bin/clang++ -x hip --offload-arch=gfx1100 -std=c++17 -O2 \
//     --cuda-device-only -S -emit-llvm tests/hip/shared_int4_alignment.cu -o shared-int4.ll
//   The same command with -DTEST_LEGACY_SHARED_ALIGN=1 must fail with the
//   original "alignas attribute cannot be applied to types" diagnostic.
// CUDA (no GPU required): compile both macro variants with nvcc --ptx -O2
// -arch=sm_120 -std=c++17 and compare the PTX, including .shared .align 16.
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

static_assert(sizeof(int4) == 16 && alignof(int4) == 16, "int4 layout changed");
#ifdef TEST_LEGACY_SHARED_ALIGN
#define TEST_SHARED_INT4 __shared__ alignas(16) int4
#else
#define TEST_SHARED_INT4 __shared__ __align__(16) int4
#endif

extern "C" __global__ void shared_int4_gu_probe(int4* out) {
    TEST_SHARED_INT4 tile[64];
    const unsigned t = threadIdx.x & 63;
    tile[t] = out[t];
    __syncthreads();
    out[t] = tile[t ^ 1];
}
extern "C" __global__ void shared_int4_down_probe(int4* out) {
    TEST_SHARED_INT4 tile[128];
    const unsigned t = threadIdx.x & 127;
    tile[t] = out[t];
    __syncthreads();
    out[t] = tile[t ^ 1];
}
