#!/usr/bin/env bash
# fetch-llama.sh - third_party/llama.cpp for Xolotl: llama.cpp at the pinned commit + tools/xolotl/ggml/xolotl-ggml.patch.
# The engines build against its ggml (cmake -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp, or leave it unset and CMake
# fetches + patches the same commit itself); tools/vision and tools/audio build against the whole tree (mtmd).
#   bash tools/xolotl/fetch-llama.sh [DIR]      (from the repo root; DIR defaults to third_party/llama.cpp)
# Re-run it after a pull that changed xolotl-ggml.patch: an older patch is reset away (the tree is vendored - local
# edits inside it are discarded) and the current one applied, so the tree always matches the checked-in patch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PIN=6050b467
PATCH="$PWD/tools/xolotl/ggml/xolotl-ggml.patch"
DST=${1:-third_party/llama.cpp}
if [ -d "$DST/.git" ]; then
    git -C "$DST" fetch -q origin "$PIN" 2>/dev/null || git -C "$DST" fetch -q origin
else
    [ -e "$DST" ] && { echo "fetch-llama: $DST exists and is not a git checkout (an old copied tree?) - move it away" \
                           "(mv $DST $DST.old) and run this again" >&2; exit 1; }
    git clone -q https://github.com/ggml-org/llama.cpp.git "$DST"
fi
if git -C "$DST" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "fetch-llama: $DST already carries xolotl-ggml.patch"
    exit 0
fi
# an older xolotl-ggml.patch (or any edit) in the tree: back to the clean pin, then the current patch
if ! git -C "$DST" diff --quiet || [ "$(git -C "$DST" rev-parse --short=8 HEAD)" != "$PIN" ]; then
    echo "fetch-llama: $DST does not carry the current xolotl-ggml.patch - resetting it to $PIN first"
fi
git -C "$DST" checkout -q -f "$PIN"
git -C "$DST" apply "$PATCH"
echo "fetch-llama: $DST = llama.cpp $PIN + xolotl-ggml.patch"
