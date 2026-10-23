#!/bin/bash
# tools/v41/gpu_window1.sh - the V4-pause GPU window, rerunnable (2026-10-23 profile, corrected).
#
#   ./gpu_window1.sh [OUTDIR]        all runs (A stage profile, B nsys, C DSpark, D pf_b sweep,
#                                    E file-tier read-ahead, F dummy-DMA A/B)
#   ./gpu_window1.sh OUTDIR A B D    only the named runs
#
# Refuses to start while the V4 quant pass holds the card (pass.pid alive) unless FORCE=1.
# Numbers from the 2026-10-23 window: in the maintainer notes (not in this repo)
set -u
OUT=${1:-/home/mal/v41-work/gpuwin-$(date +%H%M)}; shift || true
RUNS=${*:-A B C D E F}
X=/home/mal/AI/Xolotl
cd "$X" || exit 1
mkdir -p "$OUT"

P=$(cat /home/mal/v41-work/quant-1017-v4/pass.pid 2>/dev/null)
if kill -0 "$P" 2>/dev/null && [ "${FORCE:-0}" != 1 ]; then
    echo "quant pass $P is running - 'v4 stop' first (FORCE=1 overrides)" >&2; exit 1
fi

MODEL="/media/mal/SSD NVME/AI/V41-GSQ-1017-v3.gguf"
ENGRAM="/media/Crucial1TB/V41-1017-Engram:/media/mal/SSD NVME/AI/V41-1017-Engram"
RANK=/media/mal/NVME1TB/Models/rank-1017-v3.bin
SIDECAR=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision-1017.Q2_K.gguf   # -1017 is part of the name
IDS=/home/mal/AI/Strata-Backup/v41-dev/sweep-1127/p256.i32
ARGS=(-m "$MODEL" --ids-file "$IDS" --backend cuda --experts gpu --engram "$ENGRAM" --rank "$RANK"
      --ctx 32768 --arena-adapt --kv-fp4 --route-bias 0.5 --route-bias-file 0.3 --arena-gib 64
      --prompt-batch 12288 --temp 0)

need() { [ -f "$1" ] || { echo "missing $1" >&2; exit 1; }; }
need "$MODEL"; need "$IDS"; need "$SIDECAR"
cp -f "$IDS" "$OUT/" 2>/dev/null
until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 75 ]; do
    echo "waiting for 75 GiB free ($(date +%T))"; sleep 5
done

run_a() {   # stage timings
    echo "A start $(date +%T)"
    STRATA_DEC_PROF=1 bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" -n 256 \
        > "$OUT/A.out" 2> "$OUT/A.log"
    echo "A rc=$? $(date +%T)"
    grep -E "^decode step ms/token|^decode:" "$OUT/A.log" | tail -4
}

run_b() {   # nsys: kernels, copies, sync
    echo "B start $(date +%T)"
    STRATA_PROFILE_DECODE=1 bash tools/ds4/memguard.sh 86 72 -- \
        /usr/local/cuda/bin/nsys profile -o "$OUT/B" --force-overwrite true \
        --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt \
        --sample=none --cpuctxsw=none build-xolotl/ds41_generate "${ARGS[@]}" -n 64 \
        > "$OUT/B.out" 2> "$OUT/B.log"
    echo "B rc=$? $(date +%T)"; grep -E "^decode:" "$OUT/B.log" | tail -1
    nice -n 19 /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
        --format csv -o "$OUT/Bstats" "$OUT/B.nsys-rep" > /dev/null 2>&1
}

run_c() {   # DSpark depth 4 (the serving baseline the other runs do not include)
    echo "C start $(date +%T)"
    STRATA_DSPARK="$SIDECAR" STRATA_DSPARK_KEEP=$PWD/tools/v41/serve/dspark_expert_counts.txt:64 STRATA_DSPARK_DEPTH=4 \
        bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" -n 256 \
        > "$OUT/C.out" 2> "$OUT/C.log"
    echo "C rc=$? $(date +%T)"; grep -E "^decode:" "$OUT/C.log" | tail -1
}

run_d() {   # prefetch budget sweep: the 2026-10-23 profile says 86% of PCIe traffic is prefetch,
            # 54% of it wasted.  Watch tok/s and the "prefetch N issued / M used" counter.
    for b in 0 0.7 1.43 2.0 2.9; do
        echo "D pf_b=$b start $(date +%T)"
        bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" --pf-b "$b" -n 128 \
            > "$OUT/D$b.out" 2> "$OUT/D$b.log"
        printf 'D pf_b=%-5s %s | %s\n' "$b" "$(grep -oE '^decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s' "$OUT/D$b.log" | tail -1)" \
            "$(grep -oE 'prefetch [0-9]+ issued / [0-9]+ outside the arena / [0-9]+ used' "$OUT/D$b.log" | tail -1)"
    done
}

run_e() {   # file-tier read-ahead.  advise_file posix_fadvise(WILLNEED)s the predictor's picks for layer l+ahead
            # into the PAGE CACHE, but decode misses read with O_DIRECT (STRATA_DECODE_DIRECT, default on), which
            # bypasses it - so advice alone only doubles the I/O.  The real test: buffered decode reads + advice,
            # against today's direct reads (= run D's pf_b 1.43 row, the default).
    for a in 0 1 2; do
        echo "E buffered advise_ahead=$a start $(date +%T)"
        STRATA_DECODE_DIRECT=0 STRATA_ADVISE_AHEAD=$a STRATA_ADVISE_TOP=8 \
            bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" -n 128 \
            > "$OUT/E$a.out" 2> "$OUT/E$a.log"
        printf 'E buffered advise_ahead=%-2s %s | %s\n' "$a" \
            "$(grep -oE '^decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s' "$OUT/E$a.log" | tail -1)" \
            "$(grep -oE 'file [0-9]+ ms' "$OUT/E$a.log" | tail -1)"
    done
}

run_f() {   # dummy prefetch DMA: the binary now skips it by default (246 of 2874 DMAs on 2026-10-23);
            # this is the A/B against the old replay-exact behaviour.
    for v in ${F_VALUES:-1 0}; do
        echo "F no_dummy=$v start $(date +%T)"
        STRATA_PF_NO_DUMMY=$v bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" -n 128 \
            > "$OUT/F$v.out" 2> "$OUT/F$v.log"
        printf 'F no_dummy=%-2s %s | %s\n' "$v" \
            "$(grep -oE '^decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s' "$OUT/F$v.log" | tail -1)" \
            "$(grep -oE 'prefetch [0-9]+ issued / [0-9]+ outside the arena / [0-9]+ used' "$OUT/F$v.log" | tail -1)"
    done
}

run_r() {   # repeat of the default (pf_b 1.43, direct, no-dummy) at the end: the run-to-run noise
    echo "R start $(date +%T)"
    bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate "${ARGS[@]}" -n 128 > "$OUT/R.out" 2> "$OUT/R.log"
    printf 'R default repeat %s\n' "$(grep -oE '^decode: [0-9]+ tokens in [0-9.]+ s = [0-9.]+ tok/s' "$OUT/R.log" | tail -1)"
}

for r in $RUNS; do
    "run_$(echo "$r" | tr 'A-Z' 'a-z')"
done
echo "results in $OUT"
