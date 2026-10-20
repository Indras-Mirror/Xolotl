// bench_ds41_conv_pool_spin.cu - bench_ds41_conv_pool's stand-in for compute: every SM spins for `cycles` clocks.
#include <cuda_runtime.h>

__global__ static void ds41_pool_spin(long long cycles, int * sink) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (sink && threadIdx.x == 0 && blockIdx.x == 0x7fffffff) sink[0] = 1;   // never true: keeps the loop
}

extern "C" cudaError_t ds41_pool_bench_spin(long long cycles, int blocks, cudaStream_t s) {
    ds41_pool_spin<<<blocks, 128, 0, s>>>(cycles, nullptr);
    return cudaGetLastError();
}
