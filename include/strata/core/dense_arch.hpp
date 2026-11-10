// include/strata/core/dense_arch.hpp - what xolotl-dense's reader will and will not load, as pure functions.
//
// xolotl-dense was written for the Qwen3.8-27B dense file (architecture `qwen35`: GDN + dense attention + a dense
// FFN, plain pre-norm residual).  The Qwen4 generation adds the Flash-Next features to that:
//
//   * hyper-connections - SUPPORTED (2026-11-10): `gr_read`/`gr_write` in kernels/gr.hpp are the reference
//     kernels, the same ones the MoE path runs.  A dense file carrying them loads.
//   * a QSA indexer - NOT supported here: the dense engine has plain dense attention, and `native_qsa.hpp` is
//     included for its types only.  Refused by name.
//   * PLE n-gram embeddings - NOT supported here.  Refused by name.
//
// Header-only and GPU-free: the loader (src/core/dense_model.cpp) and tests/core/dense_arch_test.cpp share it.
#pragma once

#include <string>

namespace strata::core {

/// The Flash-Next features found in the file's metadata and tensor names.
struct DenseArchProbe {
    bool hyper_connections = false;  ///< <arch>.hyper_connection.count > 0
    bool qsa_indexer = false;        ///< <arch>.attention.indexer.head_count > 0
    bool ple = false;                ///< <arch>.ple.layers non-empty, or a blk.N.ple_key.weight tensor
};

/// Empty when the reader may load the file, otherwise the reason it refuses.  `allow_experimental` is
/// STRATA_DENSE_EXPERIMENTAL_ARCH=1: an unrecognised dense architecture is opt-in, never the default.
///
/// Hyper-connections are not a refusal - they are served.  A file that has them is accepted here and the HC
/// tensor set is validated by `dense_hc_refusal` once the geometry is known.
inline std::string dense_arch_refusal(const std::string& arch, const DenseArchProbe& p, bool allow_experimental) {
    std::string features;
    auto add = [&features](const char* f) {
        if (!features.empty()) features += ", ";
        features += f;
    };
    if (p.qsa_indexer) add("a QSA indexer");
    if (p.ple) add("PLE n-gram embeddings");
    if (!features.empty())
        return "dense model: this file has " + features +
               "; this engine has none of those kernels (hyper-connections ARE supported)";
    if (arch.rfind("qwen3", 0) == 0) return {};
    if (!allow_experimental)
        return "dense model: architecture '" + arch +
               "' has not been validated with this engine; set STRATA_DENSE_EXPERIMENTAL_ARCH=1 to try it with the "
               "generic dense reader";
    return {};
}

/// The hyper-connection geometry's own guard, from the two metadata keys.  Empty when usable.
///
/// The kernels take `hc * n_embd` as one contiguous row and the native path reads pairs, so both the stacked
/// width and the bottleneck rank must be even; a zero rank with a non-zero count is a broken file, not a model.
inline std::string dense_hc_refusal(int hc, int hc_lr, int n_embd, bool with_mtp) {
    if (hc <= 0) return {};
    if (hc_lr <= 0)
        return "dense model: hyper_connection.count is " + std::to_string(hc) +
               " but hyper_connection.low_rank is missing or zero";
    if (n_embd <= 0) return "dense model: hyper-connections without an embedding width";
    const long long hc_dim = (long long) hc * n_embd;
    if (hc_dim % 2 != 0 || hc_lr % 2 != 0)
        return "dense model: hyper-connections need an even hc*n_embd (" + std::to_string(hc_dim) +
               ") and low rank (" + std::to_string(hc_lr) + ")";
    if (with_mtp)
        return "dense model: hyper-connections with the MTP block are not wired here yet (the MTP block has its own "
               "nextn_hc_head mixer); run without --mtp and let an external drafter (--dflash2) do the speculation";
    return {};
}

}  // namespace strata::core
