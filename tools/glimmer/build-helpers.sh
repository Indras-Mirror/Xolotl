#!/bin/bash
# Build xolotl-glimmer's helper programs against a llama.cpp checkout with Muse Glimmer support (#26841: text, the
# muse-glimmer mtmd projector), built with its default CMake so <build>/bin has libmtmd + libllama + ggml.  They run
# as separate processes so libmtmd's shared ggml never meets the engine's own vendored ggml.
#   tools/glimmer/build-helpers.sh ~/llama.cpp [build dir name, default build]
#   -> build-xolotl/xolotl-glimmer-vision (pictures in) and build-xolotl/glimmer_ref_dump (the parity reference)
set -euo pipefail
L=${1:?usage: build-helpers.sh <llama.cpp checkout> [build dir]}
B="$L/${2:-build}/bin"
cd "$(dirname "$0")/../.."
mkdir -p build-xolotl
F=(-O2 -std=c++17 ${STRATA_VISION_DEFS:--DSTRATA_MTMD_HELPER_NO_OPT} -Itools/vision -I"$L/include" -I"$L/ggml/include" -I"$L/tools/mtmd"
   -L"$B" -lmtmd -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$B")
g++ tools/vision/strata_vision.cpp "${F[@]}" -o build-xolotl/xolotl-glimmer-vision
g++ tools/glimmer/glimmer_ref_dump.cpp -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" -L"$B" -lllama \
    -Wl,-rpath,"$B" -o build-xolotl/glimmer_ref_dump
echo "built build-xolotl/xolotl-glimmer-vision, glimmer_ref_dump"
