#include "device_context.hpp"
#include <cuda_runtime.h>

namespace strata::m31 {
std::string release_cuda_device(int device) {
    auto failed = [](const char * step, cudaError_t status) {
        (void) cudaGetLastError();
        return std::string("M31 device release: ") + step + ": " + cudaGetErrorString(status);
    };
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return failed("select device", status);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return failed("synchronize", status);
    status = cudaDeviceReset();
    if (status != cudaSuccess) return failed("reset primary context", status);
    return {};
}
PinnedHostRange::PinnedHostRange(void * data, size_t bytes) noexcept : data_(data) {
    pinned_ = data && bytes && cudaHostRegister(data, bytes, cudaHostRegisterPortable) == cudaSuccess;
    if (!pinned_) (void) cudaGetLastError();
}
PinnedHostRange::~PinnedHostRange() {
    if (pinned_ && cudaHostUnregister(data_) != cudaSuccess) (void) cudaGetLastError();
}
} // namespace strata::m31
