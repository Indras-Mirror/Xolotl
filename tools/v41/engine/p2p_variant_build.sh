#!/usr/bin/env bash
# p2p_variant_build.sh - build a ds41_generate VARIANT from this worktree's ds41_moe.cpp, ds41_dense.cpp and
# ds41_generate.cpp (branch p2p-direct) against the main tree's build-xolotl objects, into $OUT (default
# /media/Crucial1TB/xolotl-bin/p2p).  build-xolotl and the serving binary are never written.  Links libxolp2p
# (~/AI/Xolotl/private/xolp2p).  The main tree must sit at this branch's base commit (no engine diff there), or the
# swapped objects would meet mismatched ones.  Never run it while an engine runs from $OUT or reads $OUT's drive.
set -euo pipefail
W=$(cd "$(dirname "$0")/../../.." && pwd)            # this worktree
M=$HOME/AI/Xolotl                                     # the main tree (build-xolotl)
B=$M/build-xolotl
OUT=${OUT:-/media/Crucial1TB/xolotl-bin/p2p}
X=$M/private/xolp2p
command -v ninja >/dev/null || PATH=$PATH:$HOME/anaconda3/bin   # systemd units do not get anaconda's ninja
mkdir -p "$OUT"
make -s -C "$X" libxolp2p.a
cd "$B"
cmds=$(ninja -t commands ds41_generate)
compile() {   # SRC (basename) OBJ (build-xolotl relative) -> $OUT/SRC.o
    local src=$1 obj=$2 cc
    cc=$(grep -F -- "-o $obj " <<<"$cmds" | grep -F -- "-c $M/tools/v41/engine/$src" | head -1)
    [ -n "$cc" ] || { echo "could not find the compile command for $src"; exit 1; }
    cc=${cc//"-c $M/tools/v41/engine/$src"/"-c $W/tools/v41/engine/$src"}
    cc=${cc//"-o $obj "/"-o $OUT/$src.o "}
    cc=$(sed -E "s# -MF [^ ]+# -MF $OUT/$src.d#" <<<"$cc")
    cc=${cc/"/usr/bin/c++ "/"/usr/bin/c++ -I$W/tools/v41/engine -I$X -DSTRATA_XOLP2P=1 "}
    echo "[p2p] compile $src"; eval "$cc"
}
compile ds41_moe.cpp      CMakeFiles/ds41_moe_cuda.dir/tools/v41/engine/ds41_moe.cpp.o
compile ds41_dense.cpp    CMakeFiles/ds41_dense.dir/tools/v41/engine/ds41_dense.cpp.o
compile ds41_generate.cpp CMakeFiles/ds41_generate.dir/tools/v41/engine/ds41_generate.cpp.o
rm -f "$OUT/libds41_moe_cuda.a" "$OUT/libds41_dense.a"
ar qc "$OUT/libds41_moe_cuda.a" "$OUT/ds41_moe.cpp.o"; ranlib "$OUT/libds41_moe_cuda.a"
ar qc "$OUT/libds41_dense.a" "$OUT/ds41_dense.cpp.o" "$B/CMakeFiles/ds41_dense.dir/tools/v41/engine/ds41_engram.cpp.o"; ranlib "$OUT/libds41_dense.a"
ld=$(grep -E -- "-o ds41_generate( |$)" <<<"$cmds" | head -1)
[ -n "$ld" ] || { echo "could not find the link command"; exit 1; }
ld=${ld//" libds41_moe_cuda.a"/" $OUT/libds41_moe_cuda.a"}
ld=${ld//" libds41_dense.a"/" $OUT/libds41_dense.a"}
ld=${ld//" CMakeFiles/ds41_generate.dir/tools/v41/engine/ds41_generate.cpp.o"/" $OUT/ds41_generate.cpp.o"}
ld=$(sed -E "s# -o ds41_generate( |$)# -o $OUT/ds41_generate\1#" <<<"$ld")
ld="${ld% && :} $X/libxolp2p.a -lcuda && :"
echo "[p2p] link"; eval "$ld"
ls -la "$OUT/ds41_generate"
