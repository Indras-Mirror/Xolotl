#pragma once

#include <algorithm>
#include <vector>

namespace strata {

/// One GPU's compute capability as 10 * major + minor (Volta sm_70 -> 70).
inline int cuda_cc_encode(int major, int minor) { return major * 10 + minor; }

/// True when int8 tensor-core MMQ is unavailable or unsafe (Pascal / Volta: use dp4a / MMVQ instead).
inline bool cuda_cc_below_mmq_tensor_cores(int cc) { return cc < 75; }

inline bool cuda_cc_below_mmq_tensor_cores(int major, int minor) {
    return cuda_cc_below_mmq_tensor_cores(cuda_cc_encode(major, minor));
}

/// Minimum encoded CC across ordinals (empty -> a value above 75 so MMQ stays allowed).
inline int cuda_min_cc_encoded(const std::vector<int>& encoded_cc) {
    if (encoded_cc.empty()) return 100;
    return *std::min_element(encoded_cc.begin(), encoded_cc.end());
}

}  // namespace strata
