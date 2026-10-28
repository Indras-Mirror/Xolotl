#!/usr/bin/env bash
# fetch-llama.sh - third_party/llama.cpp for Xolotl: llama.cpp at the pinned commit + tools/xolotl/ggml/xolotl-ggml.patch.
# The engines build against its ggml (cmake -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp, or leave it unset and CMake
# fetches + patches the same commit itself); tools/vision and tools/audio build against the whole tree (mtmd).
#   bash tools/xolotl/fetch-llama.sh            (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/../.."
PIN=6050b467
PATCH="$PWD/tools/xolotl/ggml/xolotl-ggml.patch"
DST=third_party/llama.cpp
if [ -d "$DST/.git" ]; then
    git -C "$DST" fetch -q origin "$PIN" 2>/dev/null || git -C "$DST" fetch -q origin
else
    [ -e "$DST" ] && { echo "fetch-llama: $DST exists and is not a git checkout - move it away first" >&2; exit 1; }
    git clone -q https://github.com/ggml-org/llama.cpp.git "$DST"
fi
if git -C "$DST" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "fetch-llama: $DST already carries xolotl-ggml.patch"
    exit 0
fi
git -C "$DST" checkout -q "$PIN"
git -C "$DST" apply "$PATCH"
echo "fetch-llama: $DST = llama.cpp $PIN + xolotl-ggml.patch"
