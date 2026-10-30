#pragma once
#include "q36_kv_vmm.hpp"
#include <cuda.h>
#include <algorithm>

namespace strata { namespace q36 { namespace kv {
// Private, non-exportable allocations: no UVM and no IPC shareable handles.
// Runtime checkpoint/restore validation is required for every supported driver.
class CudaDriver final : public Driver {
    int device_;
    size_t page_ = 0;
public:
    explicit CudaDriver(int device): device_(device) {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        size_t minimum = 0;
        if (cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && minimum) {
            // Prefer 2 MiB; use an exact multiple of the device's requirement.
            align_up(size_t(2) << 20, minimum, page_);
        }
    }
    size_t granularity() const override { return page_; }
    bool reserve(uintptr_t& address, size_t bytes, size_t alignment) override {
        CUdeviceptr pointer = 0;
        if (cuMemAddressReserve(&pointer, bytes, alignment, 0, 0) != CUDA_SUCCESS) return false;
        address = static_cast<uintptr_t>(pointer); return true;
    }
    bool map(uintptr_t address, size_t bytes) override {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        CUmemGenericAllocationHandle handle;
        if (cuMemCreate(&handle, bytes, &prop, 0) != CUDA_SUCCESS) return false;
        if (cuMemMap(address, bytes, 0, handle, 0) != CUDA_SUCCESS) {
            cuMemRelease(handle); return false;
        }
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(address, bytes, &access, 1) != CUDA_SUCCESS) {
            // Failure to unmap means the arena cannot safely proceed.
            const CUresult result = cuMemUnmap(address, bytes);
            cuMemRelease(handle);
            if (result != CUDA_SUCCESS) throw std::runtime_error("Q36 KV access rollback failed");
            return false;
        }
        // Mapping retains the physical allocation until its final unmap.
        if (cuMemRelease(handle) != CUDA_SUCCESS) throw std::runtime_error("Q36 KV handle release failed");
        return true;
    }
    bool unmap(uintptr_t address, size_t bytes) override { return cuMemUnmap(address, bytes) == CUDA_SUCCESS; }
    bool free_address(uintptr_t address, size_t bytes) override { return cuMemAddressFree(address, bytes) == CUDA_SUCCESS; }
};
}}}
