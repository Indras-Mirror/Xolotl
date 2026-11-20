#!/usr/bin/env bash
# Capture REAL QFN activations for training: the residual entering every layer >= SPLIT, plus the final residual.
#
#   qfn_capture.sh <prompt-tokens-file> <out-dir> [split=44] [extra strata args...]
#
# Uses the prefill taps that are already in the production engine - no engine change:
#   STRATA_KVA_CAPTURE=<dir>      -> <dir>/L<l>.bin for l >= STRATA_KVA_SPLIT: [T, hc*n_embd] rows entering layer l
#   STRATA_KVA_CAPTURE_FP32=1     -> fp32 records (default bf16)
#   STRATA_PREFILL_DUMP_R_ALL=<f> -> the final multi-stream residual (after layer 47) as bf16 [T, hc*n_embd], APPENDED
#   STRATA_PREFILL_CPU_SHARE=0    -> no CPU expert share, so the capture is the GPU path only (parity runs)
# Verified 21 Nov on the real pack (25-token prompt): L44..L47 written, model output unchanged (" Madrid").
# The prompt's LAST token is not prefilled on a native pack (it is the first verify window), so a capture of an
# n-token prompt holds n-1 positions: append one extra token to the window you want captured.
#
# This replaces qfn_dump_layers.sh (--dump-layers is unreachable on a native pack: the token loop hands the native
# pack straight to verify windows at generate.cpp:18390, before the dump write at 18486).
# Needs the 4090: ask on the relay first.
set -euo pipefail
TOK=${1:?usage: $0 <tokens-file> <out-dir> [split] [strata args...]}
OUT=${2:?usage: $0 <tokens-file> <out-dir> [split] [strata args...]}
SPLIT=${3:-44}
shift 3 2>/dev/null || shift $#
R=$(cd "$(dirname "$0")/../.." && pwd)
N="${QFN_NATIVE:?set QFN_NATIVE to the QFN native GGUF}"
mkdir -p "$OUT"
rm -f "$OUT"/L*.bin "$OUT"/final.bf16
STRATA_KVA_CAPTURE="$OUT" STRATA_KVA_CAPTURE_FP32=1 STRATA_KVA_SPLIT="$SPLIT" \
STRATA_PREFILL_DUMP_R_ALL="$OUT/final.bf16" STRATA_PREFILL_CPU_SHARE=${STRATA_PREFILL_CPU_SHARE:-0} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$R/build-xolotl/strata" \
    --pack "${QFN_PACK:?set QFN_PACK to the QFN pack directory}" --native "$N" --ple-gguf "$N" --ple-io direct \
    --expert-profile "$R/data/expert-profile.bin" --expert-cache auto \
    --prefill 256 --spec 2 --max-context 8192 --kv int8 \
    --tokens-file "$TOK" --max-new 1 --greedy "$@" > "$OUT/strata.log" 2>&1
grep -E 'KVA|prefill [0-9]+ tokens' "$OUT/strata.log" || true
ls -la "$OUT"
