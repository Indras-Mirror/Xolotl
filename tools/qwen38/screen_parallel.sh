#!/usr/bin/env bash
# Screen the MoE dump across N layers at once, instead of the driver's one-at-a-time loop.
#
# WHY: one layer costs ~1 CPU-hour (measured: L0 burned ~3,600 s of CPU). The driver runs them
# sequentially on ~2.3 cores, so 48 layers is ~21 h. The pristine control screen did the same 48
# layers in 3 h because it ran them in parallel - its per-layer logs finish in three waves of
# ~16. The work is independent per layer (each writes its own L<NN>.json and .npz), so this is a
# scheduling change, not a numerical one: the same screen.py, the same tiers, the same inputs.
#
# Idempotent: a layer with BOTH its json and npz is skipped, so this can be re-run, and the
# chain driver skips those layers too when it resumes.
#
#   screen_parallel.sh [JOBS]     (default 14; one BLAS thread each so JOBS <= cores)
set -uo pipefail

# SINGLE INSTANCE. Two launchers would each pick up the same unfinished layer and both write
# L<NN>.json / .npz for it - a corrupt artifact that the skip check would then treat as done.
# The lock is held for the life of the process, so a second invocation exits immediately.
exec 9>"${STATE_LOCK:-/home/mal/.quetza-data/qwen38-screen.lock}"
if ! flock -n 9; then
    echo "another screen_parallel.sh holds the lock; not starting a second one" >&2
    exit 0
fi

XOLOTL=/home/mal/AI/Xolotl
T=$XOLOTL/tools/qwen38
NVME=/media/mal/NVME1TB/qwen38
MODEL=$NVME/qwen38-full-q8_0.gguf
DUMPS=$NVME/dumps-abl
STATS=$NVME/stats-abl
IMP=$NVME/imp-abl
EDIT_REPORT=/home/mal/.quetza-data/qwen38-q8-edit.json
STATE=/home/mal/.quetza-data/qwen38-abl-chain

JOBS=${1:-14}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$STATS" "$IMP" || exit 2

screen_one() {
    local L=$1
    local f p
    f=$(printf "%s/L%02d.json" "$STATS" "$L")
    p=$(printf "%s/L%02d.npz" "$IMP" "$L")
    if [ -f "$f" ] && [ -f "$p" ]; then
        echo "L$L skip (present)"
        return 0
    fi
    # ANOTHER PROCESS MAY ALREADY BE SCREENING THIS LAYER. Two workers on one layer write the
    # same L<NN>.json / .npz; whichever finishes second wins, and a reader can catch a torn file
    # in between. That happened on 2026-10-28 (a stray launch started layer 0 while an orphaned
    # worker was already on it). Absence of an artifact does NOT mean absence of a worker, so
    # check for the worker itself. The bracket keeps this pattern from matching its own command
    # line, which is the failure mode `pkill -f` has already caused twice in this run.
    if ps -eo args= | grep -q -- "python3 .*[s]creen\.py .*--layer $L "; then
        echo "L$L skip (a worker for this layer is already running)"
        return 0
    fi
    ionice -c3 nice -n 10 python3 "$T/screen.py" \
        --model-dir "$MODEL" --dump "$DUMPS" --layer "$L" \
        --tiers ptq1,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,q8_0 \
        --edit-report "$EDIT_REPORT" --dn-tiers iq4_nl,q8_0 \
        --out "$f" --importance-out "$p" \
        >"$(printf "%s/screen-L%02d.log" "$STATE" "$L")" 2>&1
    local rc=$?
    # a layer is done only when BOTH artifacts exist: screen.py writes json then npz, so a
    # non-zero exit can still leave one of them behind and fool the skip check next time
    if [ "$rc" -ne 0 ] || [ ! -f "$f" ] || [ ! -f "$p" ]; then
        echo "L$L FAILED rc=$rc json=$([ -f "$f" ] && echo yes || echo no) npz=$([ -f "$p" ] && echo yes || echo no)" >&2
        rm -f "$f" "$p"
        return 1
    fi
    echo "L$L done"
}
export -f screen_one
export T NVME MODEL DUMPS STATS IMP EDIT_REPORT STATE

todo=${LAYERS:-$(seq 0 47)}
echo "=== parallel screen: jobs=$JOBS, layers ${LAYERS:-0-47} (present layers skipped)"
printf '%s\n' $todo | xargs -P "$JOBS" -I{} bash -c 'screen_one {}'

# the exit status of xargs is 123 if any invocation failed; verify from the artifacts, which is
# the only thing the next stage actually depends on
n_j=$(ls "$STATS"/*.json 2>/dev/null | wc -l)
n_p=$(ls "$IMP"/*.npz 2>/dev/null | wc -l)
echo "=== artifacts: $n_j json, $n_p npz"
if [ "$n_j" -eq 48 ] && [ "$n_p" -eq 48 ]; then
    echo "SCREEN_COMPLETE 48/48"
    exit 0
fi
echo "SCREEN_INCOMPLETE - missing:"
for L in $(seq 0 47); do
    f=$(printf "%s/L%02d.json" "$STATS" "$L"); p=$(printf "%s/L%02d.npz" "$IMP" "$L")
    { [ -f "$f" ] && [ -f "$p" ]; } || echo "  L$L"
done
exit 1
