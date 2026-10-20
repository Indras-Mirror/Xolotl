#pragma once

#include <string>

namespace strata::v41 {

/// Lowest compute capability among visible CUDA devices (10 * major + minor).  No devices: returns 100.
bool ds41_min_visible_cuda_cc(int& min_cc_encoded, std::string& err);

/// Apply Volta/Pascal prompt policy: turn off chunk-MMQ unless STRATA_FORCE_CHUNK_MMQ=1.
void ds41_apply_mmq_cc_policy(bool backend_is_cuda, bool& chunk_mmq);

}  // namespace strata::v41
