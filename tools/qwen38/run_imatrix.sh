#!/bin/bash
# run_imatrix.sh - V4-methodology imatrix for Qwen3.8-Flash-Next XQ
#
# Corpus: /media/mal/NVME1TB/qwen38/corpus-v4  (prep_calib.py, V4 domain mix,
# DeepSeek template tokens stripped).  773 chunks x 512 = 395,776 real tokens.
# Output goes to NVME1TB, never Crucial1TB (98% full; the 162G source lives there).
#
#   bash tools/qwen38/run_imatrix.sh            # full run, detached log on NVME
#   CHUNKS=20 bash tools/qwen38/run_imatrix.sh  # short probe
set -uo pipefail

BIN=${IMATRIX_BIN:-/home/mal/AI/llama.cpp-qwen4exp/build/bin/llama-imatrix}
# Read the model from NVMe: measured 2.1 GB/s vs 394 MB/s on Crucial1TB (dd iflag=direct).
# The Crucial1TB copy stays in place - it is the llama-quantize source and must not move.
SRC=${SRC:-/media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf}
CORPUS=${CORPUS:-/media/mal/NVME1TB/qwen38/corpus-v4/corpus.txt}
OUT=${OUT:-/media/mal/NVME1TB/qwen38/imatrix-v4.gguf}
LOG=${LOG:-/media/mal/NVME1TB/qwen38/logs/imatrix-v4.log}
CHUNKS=${CHUNKS:-48}        # 396,199 tokens / 8192 = 48 (V4 used 8192-token sequences)
SAVE_EVERY=${SAVE_EVERY:-12}
THREADS=${THREADS:-12}
# Why ubatch 8192 and not 512: each 512-token batch routes to ~all 512 experts (top-10 of 512
# per token), so the expert working set per physical batch is ~128 GB no matter how few tokens
# it holds.  I/O is therefore per UBATCH, not per token.  512-token ubatches over 396k tokens
# read ~39 TB; 8192-token ubatches read ~6 TB.  That was the whole 33 h vs 2 h difference.
CTX=${CTX:-8192}
BATCH=${BATCH:-8192}
UBATCH=${UBATCH:-8192}

fail() { echo "run_imatrix: $*" >&2; exit 2; }

[ -x "$BIN" ] || fail "no llama-imatrix at $BIN"
[ -f "$SRC" ] || fail "no source model at $SRC"
[ -f "$CORPUS" ] || fail "no corpus at $CORPUS"
[ "$(stat -c%s "$CORPUS")" -gt 1000000 ] || fail "corpus looks truncated: $(stat -c%s "$CORPUS") bytes"
command -v mountpoint >/dev/null && { mountpoint -q /media/mal/NVME1TB || fail "NVME1TB not mounted"; }
[ -f "$CORPUS" ] || fail "corpus vanished after mount check"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" || fail "cannot create output/log dirs"
FREE_GB=$(df --output=avail -BG "$(dirname "$OUT")" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -gt 10 ] || fail "only ${FREE_GB} GB free on $(dirname "$OUT"); need >10"

if [ -e "$OUT" ]; then
  fail "$OUT already exists - move it aside rather than overwriting a possibly-good imatrix"
fi

echo "run_imatrix:"
echo "  source  $SRC"
echo "  corpus  $CORPUS ($(stat -c%s "$CORPUS") bytes)"
echo "  output  $OUT"
echo "  log     $LOG"
echo "  chunks  $CHUNKS   save every $SAVE_EVERY   threads $THREADS"

exec "$BIN" \
  -m "$SRC" -f "$CORPUS" -o "$OUT" \
  --chunks "$CHUNKS" --save-frequency "$SAVE_EVERY" \
  -t "$THREADS" -c "$CTX" -b "$BATCH" -ub "$UBATCH" -ngl 99 --cpu-moe
