#!/usr/bin/env python3
"""Compile/link current Metal fallback ABIs and verify that each fails closed.

Runs on a CPU without a Metal SDK on Linux, using CUDA headers only for the
shared host declarations (CUDA_INCLUDE_DIR, default /usr/local/cuda/include).
No GPU runtime is linked or invoked. On macOS it uses the Metal compat headers.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include "strata/kernels/qwen35_attention.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    using namespace strata::kernels;
    const NativeExpertLayout layout{};
    switch (std::atoi(argv[1])) {
    case 0: qwen35_gdn_out_norm(nullptr, nullptr, nullptr, nullptr, 0, 0, 0.f, nullptr); break;
    case 1: native_expert_grouped(layout, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 0, 0, nullptr, nullptr, nullptr, nullptr, 0, nullptr); break;
    case 2: cpu::iq256_gu_rows(0, nullptr, 0, 0, 0, nullptr, 0, nullptr, 0, 0, 1.f); break;
    case 3: cpu::iq512_gu_rows(0, nullptr, 0, 0, 0, nullptr, 0, nullptr, 0, 0, 1.f); break;
    case 4: cpu::q2_0_rows_q8_0(nullptr, 0, 0, nullptr, 0, nullptr, 0, 0); break;
    case 5: cpu::ptq1_0_rows_q8_0(nullptr, 0, 0, nullptr, 0, nullptr, 0, 0); break;
    case 6: cpu::ptq1_0_vec_dot_q8_0(0, nullptr, 0, nullptr, 0, nullptr, 0, 1); break;
    default: return 2;
    }
    return 0; // Every fallback must stop before reaching this line.
}
'''
NAMES = ["qwen35_gdn_out_norm", "native_expert_grouped", "cpu::iq256_gu_rows",
         "cpu::iq512_gu_rows", "cpu::q2_0_rows_q8_0", "cpu::ptq1_0_rows_q8_0",
         "cpu::ptq1_0_vec_dot_q8_0"]


def main():
    with tempfile.TemporaryDirectory(prefix="metal-stub-link-") as tmp:
        tmp = Path(tmp)
        source = tmp / "calls.cpp"
        source.write_text(SOURCE)
        binary = tmp / "stub_link"
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++20", "-O0", "-ffunction-sections", "-fdata-sections", "-pthread",
            "-I" + str(ROOT / "include"),
        ]
        if sys.platform == "darwin":
            command += ["-DSTRATA_GPU_METAL=1",
                        "-I" + str(ROOT / "metal/include/metal_compat"),
                        "-I" + str(ROOT / "metal/include"), "-Wl,-dead_strip"]
        else:
            cuda_headers = Path(os.environ.get("CUDA_INCLUDE_DIR", "/usr/local/cuda/include"))
            if not (cuda_headers / "cuda_runtime.h").exists():
                raise RuntimeError("Set CUDA_INCLUDE_DIR to the CUDA headers; no GPU or runtime library is required")
            command += ["-I" + str(cuda_headers), "-Wl,--gc-sections"]
        command += shlex.split(os.environ.get("CXXFLAGS", ""))
        command += [str(source), str(ROOT / "metal/src/engine/stubs.cpp"),
                    str(ROOT / "metal/src/engine/stubs_0145.cpp"),
                    str(ROOT / "metal/src/engine/stubs_144.cpp"),
                    str(ROOT / "metal/src/engine/missing.cpp"), "-o", str(binary)]
        subprocess.run(command, check=True)
        env = os.environ.copy()
        env.pop("STRATA_METAL_STUBS", None)
        for index, name in enumerate(NAMES):
            result = subprocess.run([str(binary), str(index)], env=env, capture_output=True, text=True)
            expected = f"strata metal: strata::kernels::{name} has no Metal implementation yet"
            if result.returncode != 86 or expected not in result.stderr:
                raise AssertionError((name, result.returncode, result.stdout, result.stderr))
            diagnostic = subprocess.run([str(binary), str(index)], env={**env, "STRATA_METAL_STUBS": "log"},
                                        capture_output=True, text=True)
            if diagnostic.returncode != 0 or f"strata metal: STUB strata::kernels::{name}" not in diagnostic.stderr:
                raise AssertionError((name, diagnostic.returncode, diagnostic.stdout, diagnostic.stderr))
        print("Metal fallback ABI link: 7 passed; default exit 86: 7 passed; diagnostic log mode: 7 passed")


if __name__ == "__main__":
    main()
