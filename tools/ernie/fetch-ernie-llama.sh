#!/usr/bin/env bash
# fetch-ernie-llama.sh - the ERNIE 5.5 reference harness: llama.cpp at PR #31925's head, built standalone under
# third_party/llama.cpp-ernie55.  It exists to smoke-test the `ernie5_5` architecture (loader, tensor map,
# metadata read) WITHOUT touching the vendored pin at third_party/llama.cpp that the engine builds against.
#
# Why a separate tree: PR #31925 is unmerged and its branch does not rebase cleanly onto our pin (it descends
# from a later master), so it cannot be folded into xolotl-ggml.patch.  Keeping it side by side means the engine
# build is unaffected by anything here.
#
# The tree is gitignored (/third_party/llama.cpp-ernie55).  Re-running this script is idempotent: an existing
# checkout is reset to the same PR head.
#
#   bash tools/ernie/fetch-ernie-llama.sh            # fetch + configure + build llama-cli/server/quantize
#   bash tools/ernie/fetch-ernie-llama.sh --no-build # fetch + configure only
#   bash tools/ernie/fetch-ernie-llama.sh --shard    # also download shard 1 of the abliterated GGUF (~1 GB)
#
# Shard 1 is enough for every structural check: it carries all 819 small tensors and the whole metadata
# (1514 tensors declared, 44 blocks), and `models/ernie55/` is gitignored like the rest of /models/.
set -euo pipefail
cd "$(dirname "$0")/../.."

PR=31925
HEAD=7d9d5c35117dc67a70fb0ec6e0fb9e104633c158
DST=${ERNIE_LLAMA_DIR:-third_party/llama.cpp-ernie55}
SHARD_DIR=models/ernie55
REPO=orcarouter/ERNIE-5.5-Flash-Base-abliterated-GGUF
MIX=ERNIE-5.5-Flash-Base-abliterated-Mix-60G

BUILD=1
SHARD=0
for a in "$@"; do
    case "$a" in
        --no-build) BUILD=0 ;;
        --shard)    SHARD=1 ;;
        *) echo "fetch-ernie-llama: unknown argument $a" >&2; exit 2 ;;
    esac
done

if [ ! -d "$DST/.git" ]; then
    echo "fetch-ernie-llama: cloning llama.cpp into $DST"
    mkdir -p "$(dirname "$DST")"
    git init -q "$DST"
    git -C "$DST" remote add origin https://github.com/ggml-org/llama.cpp 2>/dev/null || true
fi
# the PR head, not master: the branch is the only thing that carries src/models/ernie5-5.cpp
git -C "$DST" fetch -q --depth 1 origin "pull/$PR/head"
git -C "$DST" checkout -q -f "$HEAD"
if [ ! -f "$DST/src/models/ernie5-5.cpp" ]; then
    echo "fetch-ernie-llama: $DST has no src/models/ernie5-5.cpp - PR $PR did not fetch" >&2
    exit 1
fi
echo "fetch-ernie-llama: $DST = llama.cpp $HEAD (PR $PR)"

# Configure + build.  -j3 and nice: this box may be running a quantisation job that owns most of the RAM.
if [ ! -f "$DST/build/build.ninja" ]; then
    cmake -S "$DST" -B "$DST/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF
fi
if [ "$BUILD" = 1 ]; then
    nice -n 15 cmake --build "$DST/build" -j 3 --target llama-cli llama-server llama-quantize
fi

if [ "$SHARD" = 1 ]; then
    mkdir -p "$SHARD_DIR"
    hf download "$REPO" --include "$MIX/$MIX-00001-of-00058.gguf" --local-dir "$PWD/$SHARD_DIR"
    echo "fetch-ernie-llama: shard 1 in $SHARD_DIR (ERNIE_GGUF=$(pwd)/$SHARD_DIR/$MIX/$MIX-00001-of-00058.gguf)"
fi

cat <<EOF

next:
  ERNIE_GGUF=$PWD/$SHARD_DIR/$MIX/$MIX-00001-of-00058.gguf cmake --build build-xolotl --target ernie_geometry_test
  ./build-xolotl/ernie_geometry_test          # 46 checks: geometry, tensor map, real shard-1 gate

EOF
