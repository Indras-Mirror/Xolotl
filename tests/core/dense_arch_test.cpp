// tests/core/dense_arch_test.cpp - the dense engine's admission rules (include/strata/core/dense_arch.hpp).
//
// Pure: no GGUF, no GPU.  Two rules are pinned here:
//   * what the reader refuses outright (a QSA indexer, PLE; an unvalidated architecture is opt-in) - and that
//     hyper-connections are NOT refused, because as of 2026-11-10 the dense engine serves them;
//   * the hyper-connection geometry's own guard, which is where a file with a half-wired HC set is caught.
#include "strata/core/dense_arch.hpp"

#include <cstdio>
#include <string>

using strata::core::DenseArchProbe;
using strata::core::dense_arch_refusal;
using strata::core::dense_hc_refusal;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

static bool names(const std::string& s, const char* word) { return s.find(word) != std::string::npos; }

int main() {
    const DenseArchProbe none;

    std::printf("dense_arch: the validated family\n");
    check(dense_arch_refusal("qwen35", none, false).empty(),
          "qwen35 dense, no Flash-Next features, accepted without the opt-in");

    std::printf("dense_arch: hyper-connections are served, not refused\n");
    {
        DenseArchProbe p;
        p.hyper_connections = true;
        check(dense_arch_refusal("qwen35", p, false).empty(), "a qwen3* file with hyper-connections is accepted");
        check(dense_arch_refusal("qwen4", p, false).empty() == false,
              "an unvalidated arch with hyper-connections still needs the opt-in");
        check(dense_arch_refusal("qwen4", p, true).empty(), "and is accepted with it");
    }

    std::printf("dense_arch: what is still refused, by name\n");
    {
        DenseArchProbe p;
        p.qsa_indexer = true;
        check(names(dense_arch_refusal("qwen4", p, false), "QSA indexer"), "a QSA indexer is refused and named");
    }
    {
        DenseArchProbe p;
        p.ple = true;
        check(names(dense_arch_refusal("qwen4", p, false), "PLE"), "PLE is refused and named");
        check(!dense_arch_refusal("qwen4", p, true).empty(), "the opt-in does not bypass PLE");
    }
    {
        DenseArchProbe p;
        p.hyper_connections = true;
        p.qsa_indexer = true;
        p.ple = true;
        const std::string r = dense_arch_refusal("qwen4", p, false);
        // The message's parenthetical SAYS hyper-connections are supported, so match the list itself.
        check(names(r, "QSA indexer") && names(r, "PLE") && !names(r, "has hyper-connections"),
              "HC is not in the refused-feature list even when present");
    }

    std::printf("dense_arch: an unvalidated architecture is opt-in\n");
    {
        const std::string r = dense_arch_refusal("qwen4", none, false);
        check(!r.empty() && names(r, "STRATA_DENSE_EXPERIMENTAL_ARCH=1"),
              "qwen4 without features is refused, and the opt-in is named");
        check(dense_arch_refusal("qwen4", none, true).empty(), "the same file is accepted with the opt-in");
        check(names(dense_arch_refusal("qwen4_27b", none, false), "qwen4_27b"),
              "the refusal echoes the actual architecture string");
    }

    std::printf("dense_hc: the geometry guard\n");
    check(dense_hc_refusal(0, 0, 5120, false).empty(), "no hyper-connections: nothing to check");
    check(dense_hc_refusal(4, 320, 5120, false).empty(), "4 streams of 5120, rank 320 (Flash-Next's shape): accepted");
    check(dense_hc_refusal(4, 320, 2560, false).empty(), "4 streams of 2560: accepted");
    {
        const std::string r = dense_hc_refusal(4, 0, 5120, false);
        check(!r.empty() && names(r, "low_rank"), "a count without a rank is refused, naming the key");
    }
    {
        const std::string r = dense_hc_refusal(3, 320, 5121, false);
        check(!r.empty() && names(r, "even"), "an odd hc*n_embd is refused (the kernels read pairs)");
    }
    {
        const std::string r = dense_hc_refusal(4, 0, 5120, true);
        check(!r.empty() && names(r, "low_rank"), "a broken geometry is reported before the MTP rule");
    }
    {
        const std::string r = dense_hc_refusal(4, 320, 5120, /*with_mtp=*/true);
        check(!r.empty() && names(r, "nextn_hc_head"), "hyper-connections + MTP is refused, naming what is missing");
    }

    if (failures == 0) {
        std::printf("\ndense_arch: all checks passed\n");
        return 0;
    }
    std::printf("\ndense_arch: %d check(s) FAILED\n", failures);
    return 1;
}
