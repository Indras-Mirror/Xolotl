#!/bin/bash
# Build xolotl-lfm3's two helper programs against a llama.cpp checkout with LFM3 support (upstream f3586e6 or later,
# built with its default CMake so build/bin has libmtmd + libllama + ggml).  They run as separate processes so
# libmtmd's shared ggml never meets the engine's own vendored ggml.
#   tools/lfm3/build-helpers.sh ~/llama.cpp
#   -> build-xolotl/xolotl-lfm3-vision (images, audio, video in) and build-xolotl/xolotl-lfm3-speech (speech out)
set -euo pipefail
L=${1:?usage: build-helpers.sh <llama.cpp checkout>}
cd "$(dirname "$0")/../.."
mkdir -p build-xolotl
F=(-O2 -std=c++17 -Itools/vision -I"$L/include" -I"$L/ggml/include" -I"$L/tools/mtmd"
   -L"$L/build/bin" -lmtmd -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin")
g++ tools/vision/strata_vision.cpp "${F[@]}" -o build-xolotl/xolotl-lfm3-vision
g++ tools/lfm3/lfm3_speech.cpp "${F[@]}" -o build-xolotl/xolotl-lfm3-speech
g++ tools/lfm3/lfm3_ref_dump.cpp -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" -L"$L/build/bin" -lllama \
    -Wl,-rpath,"$L/build/bin" -o build-xolotl/lfm3_ref_dump
echo "built build-xolotl/xolotl-lfm3-vision, xolotl-lfm3-speech, lfm3_ref_dump"
