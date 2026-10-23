#!/bin/bash
# tools/v41/gpu_window_defer.sh OUTDIR - STRATA_FILE_DEFER A/B on the serving path (DSpark depth 4, V3, serve flags).
#   speed:   arms off / R0 (defer every file-tier miss) / R2 (only those ranked 3rd-6th), 3 runs each, interleaved,
#            256 tokens; decode tok/s + file ms from the decode line
#   quality: off and R0 with STRATA_DSPARK_DUMP, then an exact --dump-logits prefill of each run's own stream (no
#            defer) -> tools/v41/eval/dspark_tap_kl.py: KL / top-1 of the verify rows vs that prefill (off = the floor)
# Refuses while the V4 quant pass holds the card unless FORCE=1.
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"; X=/home/mal/AI/Xolotl; cd "$X" || exit 1
P=$(cat /home/mal/v41-work/quant-1017-v4/pass.pid 2>/dev/null)
if kill -0 "$P" 2>/dev/null && [ "${FORCE:-0}" != 1 ]; then echo "quant pass $P is running" >&2; exit 1; fi
MODEL="/media/mal/SSD NVME/AI/V41-GSQ-1017-v3.gguf"
ENGRAM="/media/Crucial1TB/V41-1017-Engram:/media/mal/SSD NVME/AI/V41-1017-Engram"
RANK=/media/mal/NVME1TB/Models/rank-1017-v3.bin
SIDECAR=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision-1017.Q2_K.gguf
IDS=/home/mal/AI/Strata-Backup/v41-dev/sweep-1127/p256.i32
TMP=/media/mal/NVME1TB/defer-ab; mkdir -p "$TMP"
COMMON=(-m "$MODEL" --backend cuda --experts gpu --engram "$ENGRAM" --rank "$RANK" --ctx 32768 --arena-adapt --kv-fp4
        --route-bias 0.5 --route-bias-file 0.3 --arena-gib 64 --prompt-batch 12288 --temp 0)
DSP=(STRATA_DSPARK="$SIDECAR" STRATA_DSPARK_KEEP=$PWD/tools/v41/serve/dspark_expert_counts.txt:64 STRATA_DSPARK_DEPTH=4)
until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 75 ]; do sleep 5; done
arm_env() { case $1 in off) echo "";; R0) echo "STRATA_FILE_DEFER=0";; R2) echo "STRATA_FILE_DEFER=2";; esac; }
run() {   # name arm [extra env...]
    local name=$1 arm=$2; shift 2
    env "${DSP[@]}" $(arm_env "$arm") "$@" bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${COMMON[@]}" \
        --ids-file "$IDS" -n 256 > "$OUT/$name.out" 2> "$OUT/$name.log"
    printf '%-10s rc=%s %s | %s\n' "$name" "$?" "$(grep -oE '^decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s' "$OUT/$name.log" | tail -1)" \
        "$(grep -oE 'file [0-9]+ ms|STRATA_FILE_DEFER=[0-9]+: .*' "$OUT/$name.log" | tail -2 | tr '\n' ' ')"
}
for rep in 1 2 3; do for arm in off R0 R2; do run "s_${arm}_$rep" "$arm"; done; done
for arm in off R0; do
    run "q_$arm" "$arm" STRATA_DSPARK_DUMP="$TMP/q_$arm.dump"
    python3 - "$IDS" "$OUT/q_$arm.out" "$OUT/q_$arm.i32" <<'PY'
import sys, numpy as np
p = np.fromfile(sys.argv[1], dtype=np.int32)
g = [int(l) for l in open(sys.argv[2]) if l.strip().lstrip('-').isdigit()]
np.concatenate([p, np.array(g, np.int32)]).tofile(sys.argv[3]); print('stream', len(p) + len(g))
PY
    N=$(( $(stat -c %s "$OUT/q_$arm.i32") / 4 ))
    bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${COMMON[@]}" --ids-file "$OUT/q_$arm.i32" -n 1 --ppl \
        --dump-logits "$TMP/q_$arm.pre" > "$OUT/qp_$arm.out" 2> "$OUT/qp_$arm.log"
    echo "q_$arm prefill rc=$? N=$N $(nice -n 19 python3 tools/v41/eval/dspark_tap_kl.py "$TMP/q_$arm.dump" "$TMP/q_$arm.pre" "$N")"
    [ "${KEEP_DUMPS:-0}" = 1 ] || rm -f "$TMP/q_$arm.dump" "$TMP/q_$arm.pre"
done
echo "done $(date +%T)"
