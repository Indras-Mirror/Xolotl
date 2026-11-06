#!/usr/bin/env bash
# p2p_variant_build.sh - build a ds41_generate VARIANT from this worktree's ds41_moe.cpp (branch p2p-direct) against
# the main tree's build-xolotl objects, into /media/Crucial1TB/xolotl-bin/p2p.  build-xolotl and the serving binary
# are never written.  Links libxolp2p (~/AI/Xolotl/private/xolp2p).
set -euo pipefail
W=$(cd "$(dirname "$0")/../../.." && pwd)            # this worktree
M=$HOME/AI/Xolotl                                     # the main tree (build-xolotl)
B=$M/build-xolotl
OUT=${OUT:-/media/Crucial1TB/xolotl-bin/p2p}
X=$M/private/xolp2p
mkdir -p "$OUT"
make -s -C "$X" libxolp2p.a
cd "$B"
cc=$(ninja -t commands ds41_generate | grep -F -- "-c /home/mal/AI/Xolotl/tools/v41/engine/ds41_moe.cpp" | head -1)
ld=$(ninja -t commands ds41_generate | grep -E -- "-o ds41_generate( |$)" | head -1)
[ -n "$cc" ] && [ -n "$ld" ] || { echo "could not find the build commands"; exit 1; }
cc=${cc//"-c $M/tools/v41/engine/ds41_moe.cpp"/"-c $W/tools/v41/engine/ds41_moe.cpp"}
cc=$(sed -E "s# -o [^ ]+ds41_moe\.cpp\.o# -o $OUT/ds41_moe.cpp.o#; s# -MF [^ ]+# -MF $OUT/ds41_moe.d#" <<<"$cc")
cc=${cc/"/usr/bin/c++ "/"/usr/bin/c++ -I$W/tools/v41/engine -I$X -DSTRATA_XOLP2P=1 "}
echo "[p2p] compile ds41_moe.cpp"; eval "$cc"
rm -f "$OUT/libds41_moe_cuda.a"; ar qc "$OUT/libds41_moe_cuda.a" "$OUT/ds41_moe.cpp.o"; ranlib "$OUT/libds41_moe_cuda.a"
ld=${ld//"libds41_moe_cuda.a"/"$OUT/libds41_moe_cuda.a"}
ld=$(sed -E "s# -o ds41_generate( |$)# -o $OUT/ds41_generate\1#" <<<"$ld")
ld="${ld% && :} $X/libxolp2p.a -lcuda && :"
echo "[p2p] link"; eval "$ld"
ls -la "$OUT/ds41_generate"
