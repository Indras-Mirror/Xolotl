#!/usr/bin/env bash
# buster-engine.sh - DeepSeek-V4.1 on Buster (Volta sm_70). Same contract as ds41-engine.sh for serve/server.py.
set -u
cd "$(dirname "$0")/../.." || exit 1
export GGML_NO_BACKTRACE=1
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# Volta kernels (see docs/NVIDIA_V100.md): opt-in decode tables, prompt attention on mma.m8n8k4, BF16 via FP16 TC.
export STRATA_SM70_TABLE="${STRATA_SM70_TABLE:-1}"
export STRATA_PROMPT_ATTN_OLD="${STRATA_PROMPT_ATTN_OLD:-1}"
export STRATA_BF16_TC="${STRATA_BF16_TC:-1}"
# sm_70: chunk-MMQ is disabled at runtime (ds41_cuda_policy); do not force it on.
export STRATA_FORCE_CHUNK_MMQ="${STRATA_FORCE_CHUNK_MMQ:-0}"
# Serve defaults tuned for dual-socket Broadwell + V100 memory bandwidth (not the 4090 desktop defaults).
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark stays off unless the caller sets STRATA_SERVE_DSPARK=<sidecar> (empty export still makes getenv non-null).
[ -n "${STRATA_SERVE_DSPARK:-}" ] || unset STRATA_SERVE_DSPARK
# Optional RTX 3090 head GPU (device 0 in PCI order): trunk on --stage-devices, head + DSpark on --head-device / STRATA_HEAD_DEVICE.
[ -n "${STRATA_HEAD_DEVICE:-}" ] || unset STRATA_HEAD_DEVICE
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT:-q6_K}"
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-1.5}"
BUILD_DIR="${STRATA_BUILD_DIR:-$PWD/build-buster}"
[ -x "$BUILD_DIR/ds41_generate" ] || BUILD_DIR="$PWD/build-xolotl"
LOCK="${BUSTER_GPU_LOCK:-$HOME/.cache/xolotl/buster-gpu.lock}"
mkdir -p "$(dirname "$LOCK")"
exec flock "$LOCK" systemd-run --user --scope -q --unit="buster-ds41-$$" -p MemoryMax=140G -p MemorySwapMax=0 \
  "$BUILD_DIR/ds41_generate" "$@"
