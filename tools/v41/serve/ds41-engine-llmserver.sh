#!/usr/bin/env bash
# ds41-engine-llmserver.sh - ds41-engine.sh for llm-server (RTX 3090 main + V100 remote expert card, 60 GiB RAM, LXC):
# no ComfyUI check / GPU lock / user scope; a 57 GiB RAM cap in a system scope instead of 84 GiB.  The engine settings
# (DSpark, lookup drafts, Q6_K dense, prestage) are ds41-engine.sh's.
set -u
cd "$(dirname "$0")/../../.."
export GGML_NO_BACKTRACE=1
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
DSPARK_SIDECAR=/root/models/V41-GSQ/V41-MTP-Vision.Q2_K.gguf
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-$DSPARK_SIDECAR}"
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-4096}"
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-1.5}"
export STRATA_ARENA_PIN_AFTER="${STRATA_ARENA_PIN_AFTER:-1}"
export STRATA_DSPARK_PREFAULT="${STRATA_DSPARK_PREFAULT:-1}"
export STRATA_DSPARK_STASH="${STRATA_DSPARK_STASH:-1}"
# ZFS: the Engram tables have no raw-device path (FIEMAP)
export STRATA_ENGRAM_RAWDEV="${STRATA_ENGRAM_RAWDEV:-0}"
[ -n "${STRATA_PROMPT_BATCH:-}" ] && set -- "$@" --prompt-batch "$STRATA_PROMPT_BATCH"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
BUILD_DIR="${STRATA_BUILD_DIR:-$PWD/build-xolotl}"
# DS41_GDB=1: the engine under gdb, its output (and a backtrace on a crash) in /tmp/ds41-gdb.log - stdout stays the protocol
GDB=()
[ "${DS41_GDB:-0}" = 1 ] && GDB=(gdb -q -batch -nx -ex "set pagination off" -ex "set logging file /tmp/ds41-gdb.log"
    -ex "set logging redirect on" -ex "set logging enabled on" -ex "set print thread-events off" -ex run -ex bt
    -ex "thread apply all bt 12" --args)
# DS41_NSYS=<report path>: the engine under nsys (CUDA API + kernels + copies, graphs as one range)
[ -n "${DS41_NSYS:-}" ] && GDB=(/usr/local/bin/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none
    --cuda-memory-usage=false --cuda-graph-trace=graph --kill=none -f true -o "$DS41_NSYS")
exec systemd-run --scope -q --unit="ds41-serve-$$" -p MemoryMax="${DS41_MEMORY_MAX:-57G}" -p MemorySwapMax=0 \
    "${GDB[@]}" "$BUILD_DIR/ds41_generate" "$@"
