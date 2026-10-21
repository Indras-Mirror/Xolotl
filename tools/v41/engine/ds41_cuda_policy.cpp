#include "ds41_cuda_policy.hpp"

#include "strata/platform/cuda_cc.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(STRATA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace strata::v41 {

bool ds41_min_visible_cuda_cc(int& min_cc_encoded, std::string& err) {
    min_cc_encoded = 100;
#if defined(STRATA_ENABLE_CUDA)
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) {
        err = "cudaGetDeviceCount failed or no devices";
        return false;
    }
    int min_cc = 100;
    for (int d = 0; d < nd; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) {
            err = "cudaGetDeviceProperties failed";
            return false;
        }
        min_cc = std::min(min_cc, strata::cuda_cc_encode(prop.major, prop.minor));
    }
    min_cc_encoded = min_cc;
#endif
    return true;
}

void ds41_apply_mmq_cc_policy(bool backend_is_cuda, bool& chunk_mmq) {
    if (!backend_is_cuda || !chunk_mmq) return;
    const char* force = std::getenv("STRATA_FORCE_CHUNK_MMQ");
    if (force && force[0] == '1') return;
    int min_cc = 100;
    std::string err;
    if (!ds41_min_visible_cuda_cc(min_cc, err)) {
        chunk_mmq = false;
        return;
    }
    if (!strata::cuda_cc_below_mmq_tensor_cores(min_cc)) return;
    std::fprintf(stderr,
                 "ds41_generate: compute capability %d.%d has no int8 tensor cores; "
                 "disabling chunk-MMQ (MMVQ/dp4a prefill). Set STRATA_FORCE_CHUNK_MMQ=1 to override at your own risk.\n",
                 min_cc / 10, min_cc % 10);
    chunk_mmq = false;
}

}  // namespace strata::v41
