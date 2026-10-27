#!/usr/bin/env bash
# ds41-engine.sh - the DeepSeek-V4.1 engine as Strata's server starts it (`<exe> --serve <args>`), with memguard's
# protections in the FOREGROUND (memguard.sh backgrounds its command, which would break the stdin/stdout protocol):
# ComfyUI check, the shared GPU lock (held while the server runs), an 84 GiB RAM cap, no swap.  (= GLM's glm-engine.sh)
set -u
cd "$(dirname "$0")/../../.."
if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "ds41-engine: refusing - ComfyUI is running on :8188 (RAM)" >&2; exit 3
fi
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
export GGML_NO_BACKTRACE=1
# 4 KiB pages pin the 60 GiB arena at 1.2 GB/s (load ~50 s); THP pins at ~0.16 GB/s (~4 min)
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# prompt tokens route exactly (Mal 2026-10-12): turns <= 96 new tokens run as EXACT passes (no route bias / file
# penalty; ~6-7 tok/s, beats the ~18 s batch floor below ~110 tokens: 41 tokens 6.9 s vs 18.0 s, KL vs batch 0.012),
# longer turns as the exact batch.  The biased quick path (KL 0.10-0.15, ~40-57 tok/s):
# STRATA_SERVE_STEP_MAX=768 STRATA_SERVE_PASS_BIAS=1
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark speculative decode in --serve (Mal's go 2026-10-14; exact speculative sampling, the output distribution is the
# trunk's): the Q2_K sidecar, the draft's experts pruned to the 64 most used per layer (dspark_expert_counts.txt),
# depth 4 with the confidence-head trim at 0.6 (s11: +30% avg, prose 9.5 -> 17.3 tok/s).  STRATA_SERVE_DSPARK= (empty)
# turns it off.  The engine args (strata-ds41.json): route bias x0.5 + file 0.3, --arena-adapt, arena 64 GiB, --kv-fp4.
DSPARK_SIDECAR=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-$DSPARK_SIDECAR}"
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
# s6 (2026-10-17, verify/s6/improve2): prompt-lookup drafts of up to 14 tokens in the DSpark loop - a copy of the
# context (edits, rewrites) verified like any draft, so the output is the same (greedy edit text identical; 1040/1077
# drafts accepted, edit decode 32.0 -> 35.9 tok/s).  And DSpark's 2.9 GB of weights leave the card during a batch of
# >= 4096 new tokens, so the prompt batch is wider (16.8K fresh prompt: 263 -> 314 tok/s; re-load 2.6 s).  =0 = off.
export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-4096}"
# Dense half (attention + shared expert + head, the file's Q8_0) re-quantized to Q6_K at load (ggml, file untouched):
# 7.59 -> 5.86 GB, +1.6 GiB of VRAM expert slots -> decode +11 % after a 6.5K prompt, +15-35 % after 16.8K, edits
# +9 %; prefill on long prompts -9 % (Q6_K MMQ).  Quality on the 100 API fixtures (S6, same bf16 path): KL vs FP8
# 0.2281 -> 0.2306, API top-1 85.87 -> 85.76 % (the x0.5 route bias costs 4.6 pt).  STRATA_DENSE_REQUANT= (empty) = off.
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"
# prompt speed (s13 run4/run5, 2026-10-15): --prompt-batch 12288 fits only with the chunk prestage
# region at 1.5 GiB (was 3.0): a 20K prompt 2 sweeps instead of 3, 60.4 -> 53.4 s in serve at 512K (+13%; one-shot
# 328 -> 382 tok/s).  16384 OOMs (a 1.42 GiB alloc) even at prestage 0.75 - the next step.
# REVERTED 2026-10-15 19:35: the production harness check OOMed at batch 12288 (the same 1.42 GiB graph buffer) with
# another process holding 2.5 GB of VRAM - too fragile as a default on a desktop.
# FIXED 2026-10-15 20:25 (session 13c): the engine now sizes the layer-major batch from the VRAM ACTUALLY free
# (ds41_generate.cpp's pb_cap; calibrated need ~= 4.3 GiB + 0.95 MiB a column) and retries at half the width, the
# chunk stage is clamped from free VRAM, and the expert-cache seed skips/retries instead of failing the request -
# harness-verified with a foreign 4 GiB holder: 3/3 turns complete, 0 errors (logs ~/AI/Strata-Backup/v41-dev/s13/run15*).
# So the opt-in is now SAFE to leave on - a tight card simply gets a narrower batch (that re-reads the file tier once
# per batch: ~82 GB a sweep, so keep the width as high as the card allows).
# ADOPTED as the DEFAULT 2026-10-15 21:40 (session 13c): the quiet-box A/B (run17, 3 runs a side, spread < 0.5%):
# batch 8192 = 64.39 / 63.99 / 64.28 s (318.0 / 320.1 / 318.6 tok/s, 157.2 GB of file-tier reads in 3 sweeps);
# batch 12288 = 52.72 / 52.91 / 52.88 s (388.5 / 387.1 / 387.3 tok/s, 114.5 GB in 2 sweeps) -> **+21.5%**, the
# reader does the same 4.95-5.15 GB/s either way.  12288 is now in strata-ds41.json; STRATA_PRESTAGE_GIB=1.5 below.
# 16384 still OOMs without the cap engaged (the next step is the state tensors: NC=8192 wants 2.57 GiB, NC=12288
# 3.86 GiB, of which the engram is only 0.19/0.28 GiB each - the rest is the per-column KV/attention state).
# the chunk prestage region: 1.5 GiB is what the adopted 12288 batch was measured with (3.0 GiB starves the batch's
# own buffers on a 24 GB card).  0 = off.
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-1.5}"
# an explicit STRATA_PROMPT_BATCH still wins over the json's --prompt-batch (it is appended after those args)
[ -n "${STRATA_PROMPT_BATCH:-}" ] && set -- "$@" --prompt-batch "$STRATA_PROMPT_BATCH"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
# xolotl: the unified tree builds every engine into build-xolotl; STRATA_BUILD_DIR overrides it, and the
# pre-merge build dir (build-ds4-gpu) is still found so an old checkout of this script keeps working
BUILD_DIR="${STRATA_BUILD_DIR:-$PWD/build-xolotl}"
[ -x "$BUILD_DIR/ds41_generate" ] || [ ! -x "$PWD/build-ds4-gpu/ds41_generate" ] || BUILD_DIR="$PWD/build-ds4-gpu"
# systemd-run --user applies the MemoryMax=84G cap and the no-swap guard.  It does not exist in a container
# without a user manager (unprivileged LXC, some Docker setups; issue #60), so fall back to a plain exec there
# rather than failing to start.  STRATA_NO_SCOPE=1 forces the fallback on a box that does have systemd.
use_scope=1
[ "${STRATA_NO_SCOPE:-0}" = 1 ] && use_scope=0
if [ "$use_scope" = 1 ] && ! systemd-run --user --scope -q --unit="ds41-probe-$$" true >/dev/null 2>&1; then
    use_scope=0
    echo "ds41-engine: systemd-run --user is unavailable here - running without the memory scope" >&2
fi
if [ "$use_scope" = 1 ]; then
    exec flock "$LOCK" systemd-run --user --scope -q --unit="ds41-serve-$$" -p MemoryMax=84G -p MemorySwapMax=0 \
        "$BUILD_DIR/ds41_generate" "$@"
fi
exec flock "$LOCK" "$BUILD_DIR/ds41_generate" "$@"
