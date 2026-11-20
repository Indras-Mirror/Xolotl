#!/usr/bin/env bash
# SUPERSEDED 21 Nov by qfn_capture.sh (prefill taps STRATA_KVA_CAPTURE / STRATA_PREFILL_DUMP_R_ALL, no engine change).
# Kept for the record of why --dump-layers cannot work on a native pack.
#
# One-shot QFN run that records the residual after EVERY layer (the training data path).
#
#   qfn_dump_layers.sh <prompt-tokens-file> <out.bin> [max-new] [extra strata args...]
#
# Reads a pretokenized prompt (comma-separated ids), runs `strata` once in generate mode with --dump-layers, and
# prints the expected file size so the output can be checked before it is parsed with qfn_layer_dump.py.
#
# FACTS THIS RELIES ON (read from the source, not assumed):
#   * --dump-layers is written by session_loop (src/core/session.cpp:662-670, 757-763), which only runs on the
#     CAPTURED path - so --no-capture must NOT be passed, and the expert pool must be live (no --no-pool).
#   * Setting --dump-layers turns fused GR OFF (src/program/generate.cpp:4579:
#     layer_set_fused_gr(... && o.dump_layers.empty() ...)), so the dump is from the non-fused path.  Do not compare
#     its numbers against a fused- GR run without accounting for that.
#   * Two flag families mean the same file: --dump-routing (peer-friendly name) and --dump-layers (this one).
#   * Size per position: (n_layers + 1) * hc * n_embd floats = 49 * 4 * 2560 * 4 B = 2.0 MB for QFN (48 layers,
#     hc 4, n_embd 2560).  A 1-token prompt is ~2 MB; 200 decoded tokens are ~400 MB.
#
# BLOCKED (verified 21 Nov, on the real pack): a native (IQ) pack REQUIRES `--spec T` with T >= 2
# (generate.cpp:4394-4400), and `--spec` needs the token graph, which the build does NOT construct when a layer or
# half dump is armed (`generate.cpp:8335` and `8400`: `... && layer_dump == nullptr && half_dump == nullptr`).
# So `--dump-layers` and `--spec` are mutually exclusive, and this pack needs `--spec`.  Same for `--dump-halves`.
# Attempts on the real pack and what each returned:
#   * --prefill auto            -> refused: the pack wants an explicit --prefill CHUNK
#   * --prefill 2048            -> "prefill: device buffers for a chunk of 2048 tokens do not fit" (361 MiB free)
#   * --prefill 256 --spec 2    -> prefill OK (12 tokens, 375 ms), then "--spec needs the device residency table
#                                  (--expert-profile, --expert-cache and the token graph)"
#   * --prefill 256 (no --spec) -> refused up front: a native pack needs --spec >= 2
# Getting real activations out therefore needs a code change in the engine (capture the ladder with the token graph
# up, or allow a dumped non-spec run on a native pack), not more flags.  Do not re-burn a GPU slot on this.
#
# The GPU: this needs the 4090 for the engine (~8-24 GB depending on the expert tier).  Ask on the relay first if a
# peer is using it - see the etiquette in the relay skill.
set -euo pipefail

TOKENS_FILE=${1:?usage: $0 <tokens-file> <out.bin> [max-new] [strata args...]}
OUT=${2:?usage: $0 <tokens-file> <out.bin> [max-new] [strata args...]}
MAXNEW=${3:-1}
shift 3 2>/dev/null || shift $#

REPO=/home/mal/AI/Xolotl
EXE=$REPO/build-xolotl/strata
PACK=/media/mal/NVME1TB/Xolotl/qfn45/pack
NATIVE="/media/mal/SSD NVME/Xolotl/qfn45/Swift1.5-QFN-XQ-45GB.gguf"
PROFILE=$REPO/data/expert-profile.bin

for f in "$EXE" "$TOKENS_FILE" "$PROFILE"; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }
done

# QFN geometry, so the caller knows what the file should weigh before spending a GPU slot on it.
read -r NLAYERS HC NEMBD <<<"$(~/anaconda3/bin/python3 - "$NATIVE" <<'PY'
import sys
sys.path.insert(0, "/media/Crucial1TB/xolotl-train/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
def kv(k):
    return r.fields[k].contents() if k in r.fields else None
nl = kv("qwen4exp.block_count")
hc = kv("qwen4exp.hyper_connection.count")
ne = kv("qwen4exp.embedding_length")
print(nl or 48, hc or 4, ne or 2560)
PY
)"
PER_POS=$(( (NLAYERS + 1) * HC * NEMBD ))
echo "geometry: ${NLAYERS} layers, hc ${HC}, n_embd ${NEMBD} -> ${PER_POS} floats (${PER_POS} x 4 B) per position"

set -x
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$EXE" \
    --pack "$PACK" \
    --native "$NATIVE" \
    --ple-gguf "$NATIVE" \
    --ple-io direct \
    --expert-profile "$PROFILE" \
    --expert-cache auto \
    --prefill 256 \
    --max-context 4096 \
    --kv int8 \
    --tokens-file "$TOKENS_FILE" \
    --max-new "$MAXNEW" \
    --greedy \
    --dump-layers "$OUT" \
    "$@"
set +x

if [[ -f $OUT ]]; then
    BYTES=$(stat -c %s "$OUT")
    echo "wrote $OUT: ${BYTES} bytes = $(( BYTES / 4 / PER_POS )) positions of ${PER_POS} floats"
    [[ $(( BYTES % (PER_POS * 4) )) -eq 0 ]] || echo "WARNING: size is not a whole number of positions" >&2
else
    echo "no output file was written" >&2
    exit 1
fi
