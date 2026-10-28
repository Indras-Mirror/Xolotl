#!/bin/bash
# run_quant.sh - run llama-quantize for the Qwen3.8 XQ build, with the space gate enforced
# BEFORE any byte is written.
#
# Why a wrapper: llama-quantize writes the output as it goes. If the volume fills at 80%, you
# lose hours and are left with a partial file whose size may still look plausible. Every check
# here runs first and exits non-zero without touching the output path.
#
# The read/write split is deliberate: the 162.6 GB source stays on Crucial1TB (26 G free), the
# ~83 GB output goes to NVME1TB (~192 G free). Disjoint volumes, so no staging copy is needed.
#
#   bash tools/qwen38/run_quant.sh <tensor-types.txt> [out.gguf]
#
# Exit codes: 2 = a precondition failed (nothing written), otherwise llama-quantize's own code.
set -uo pipefail

SRC=${SRC:-/media/Crucial1TB/qwen38/qwen38-full-q8_0.gguf}
IMATRIX=${IMATRIX:-/media/Crucial1TB/qwen38/imatrix.gguf}
OUT=${2:-/media/mal/NVME1TB/qwen38/Qwen3.8-Flash-Next-XQ.gguf}
TT=${1:-}
BIN=${LLAMA_QUANTIZE:-/home/mal/AI/llama.cpp-qwen4exp/build/bin/llama-quantize}
MARGIN_GB=${MARGIN_GB:-6}          # headroom on top of the predicted size
GATE_GIB=${GATE_GIB:-77}           # the quality gate: must beat 94.1% top-1 at under 77 GiB

fail() { echo "run_quant: $*" >&2; exit 2; }

[ -n "$TT" ] || fail "usage: run_quant.sh <tensor-types.txt> [out.gguf]"
[ -x "$BIN" ] || fail "no llama-quantize at $BIN"
[ -f "$SRC" ] || fail "no source model at $SRC"
[ -f "$IMATRIX" ] || fail "no imatrix at $IMATRIX (imatrix still running?)"
[ -f "$TT" ] || fail "no tensor-types file at $TT"

SRC_GB=$(du -b --apparent-size "$SRC" | cut -f1)
[ "${SRC_GB:-0}" -gt 10000000000 ] || fail "source looks too small: $SRC_GB bytes"

OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR" || fail "cannot create $OUT_DIR"
[ -w "$OUT_DIR" ] || fail "$OUT_DIR is not writable"

# predicted output: 77 GiB is the ceiling we are allowed; budget for it plus margin
CEIL_GB=$(( GATE_GIB * 1073741824 / 1000000000 + 1 ))
NEED_GB=$(( CEIL_GB + MARGIN_GB ))
FREE_GB=$(df --output=avail -BG "$OUT_DIR" | tail -1 | tr -dc '0-9')

echo "run_quant:"
echo "  source   $SRC  ($(( SRC_GB / 1000000000 )) GB)   [read]"
echo "  imatrix  $IMATRIX"
echo "  types    $TT"
echo "  output   $OUT  [write]"
echo "  ceiling  ${CEIL_GB} GB (${GATE_GIB} GiB gate) + ${MARGIN_GB} GB margin = ${NEED_GB} GB needed"
echo "  free     ${FREE_GB} GB on $OUT_DIR"

if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  fail "NOT ENOUGH SPACE: ${FREE_GB} GB free, need ${NEED_GB} GB. Nothing was written."
fi

# the imatrix must actually be populated, not a zero-byte stub from an interrupted run
IM_BYTES=$(du -b "$IMATRIX" | cut -f1)
[ "${IM_BYTES:-0}" -gt 10000 ] || fail "imatrix looks empty: $IM_BYTES bytes"
echo "  imatrix  $(( IM_BYTES / 1024 )) KiB - ok"

# refuse to silently overwrite a previous output
if [ -e "$OUT" ]; then
  fail "$OUT already exists - move it aside rather than overwriting a possibly-good quant"
fi

echo "run_quant: gate passed, starting"
# --allow-requantize is REQUIRED: the source is Q8_0 for every tensor, and llama-quantize
# refuses to convert an already-quantized tensor without it (llama-quant.cpp:1253,
# "requantizing from type Q8_0 is disabled").  Without the flag the first expert tensor
# aborts the run.  The Q8_0 step's own error is 0.552% vs FP8's 2.55%, so requantising from
# it is within the precision this pipeline already assumes.
exec "$BIN" --imatrix "$IMATRIX" --tensor-type-file "$TT" --allow-requantize "$SRC" "$OUT" Q8_0
