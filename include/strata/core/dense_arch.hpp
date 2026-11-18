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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

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

/// xolotl: the execution plan of a looped file: [0, start), then [start, end) `count` times, then [end, n_layer).
/// Empty when the loop keys describe no loop (count <= 1) or are out of range; `err` says which.
inline std::vector<int> dense_loop_plan(int n_layer, int start, int end, int count, std::string& err) {
    std::vector<int> plan;
    if (!(0 <= start && start < end && end <= n_layer) || count < 1) {
        err = "dense model: loop [" + std::to_string(start) + ", " + std::to_string(end) + ") x" + std::to_string(count) +
              " does not fit " + std::to_string(n_layer) + " layers";
        return plan;
    }
    if (count == 1) return plan;
    for (int l = 0; l < start; ++l) plan.push_back(l);
    for (int c = 0; c < count; ++c)
        for (int l = start; l < end; ++l) plan.push_back(l);
    for (int l = end; l < n_layer; ++l) plan.push_back(l);
    return plan;
}
/// xolotl: --layer-schedule "0-3,4-27,4-27,28-31" (comma-separated layers or inclusive ranges) -> the plan.  Empty
/// string = ok and the plan untouched.  Every layer has to be a trunk layer; the schedule must run at least one.
inline std::string dense_parse_schedule(const std::string& s, int n_layer, std::vector<int>& plan) {
    if (s.empty()) return {};
    std::vector<int> out;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t comma = std::min(s.find(',', at), s.size());
        const std::string item = s.substr(at, comma - at);
        at = comma + 1;
        if (item.empty()) return "dense model: --layer-schedule has an empty item in '" + s + "'";
        const size_t dash = item.find('-');
        int a = -1, b = -1;
        try {
            size_t used = 0;
            a = std::stoi(item.substr(0, dash), &used);
            if (used != (dash == std::string::npos ? item.size() : dash)) a = -1;
            b = a;
            if (dash != std::string::npos) {
                b = std::stoi(item.substr(dash + 1), &used);
                if (used != item.size() - dash - 1) b = -1;
            }
        } catch (const std::exception&) {
            a = b = -1;
        }
        if (a < 0 || b < 0 || a >= n_layer || b >= n_layer || b < a)
            return "dense model: --layer-schedule item '" + item + "' is not a layer or range inside 0.." +
                   std::to_string(n_layer - 1);
        for (int l = a; l <= b; ++l) out.push_back(l);
        if (comma == s.size()) break;
    }
    plan = std::move(out);
    return {};
}
/// xolotl: mHC streams exist only inside [mhc_start, mhc_end): the visit of mhc_start creates them and the visit of
/// mhc_end - 1 averages them away.  So every run of mHC layers in the plan has to be exactly mhc_start .. mhc_end - 1,
/// in order.  Empty when the plan keeps that.
inline std::string dense_mhc_plan_refusal(const std::vector<int>& plan, int n_layer, int mhc_start, int mhc_end) {
    std::vector<int> p = plan;
    if (p.empty())
        for (int l = 0; l < n_layer; ++l) p.push_back(l);
    for (size_t i = 0; i < p.size();) {
        if (p[i] < mhc_start || p[i] >= mhc_end) { ++i; continue; }
        for (int l = mhc_start; l < mhc_end; ++l, ++i)
            if (i >= p.size() || p[i] != l)
                return "dense model: the layer schedule enters the mHC block [" + std::to_string(mhc_start) + ", " +
                       std::to_string(mhc_end) + ") somewhere other than its start or leaves it early (slot " +
                       std::to_string(i) + "); the streams only exist across the whole block";
    }
    return {};
}
}  // namespace strata::core
