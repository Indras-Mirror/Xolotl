#include "device_context.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace strata::m31 {
std::string cuda_prefix_runtime_identity() {
    int device = -1, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess || cudaDriverGetVersion(&driver) != cudaSuccess) return {};
    std::ostringstream out;
    out << "cuda-runtime=" << runtime << " driver=" << driver << " ordinal=" << device
        << " arch=" << properties.major << '.' << properties.minor << " name=" << properties.name
        << " pci=" << properties.pciDomainID << ':' << properties.pciBusID << ':' << properties.pciDeviceID;
    return out.str();
}
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
