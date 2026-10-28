#!/bin/bash
# fetch_ple.sh DEST [WAIT_PID] - fetch the IQ4_NL PLE n-gram table for Qwen3.8-Flash-Next.
#
# What this is: `per_layer_token_embd.weight`, [160, 320001536] = 51.2B parameters.  In the FP8
# checkpoint it is F8_E4M3, split into 128 tensors of [2500012, 160] across 31 shards (49.6 GB).
# The engine only ever reads this table as IQ4_NL (ngram.cpp), and converting FP8 -> IQ4_NL would
# need ~205 GB of RAM, so we take the already-IQ4_NL table instead (same route nvfp4_convert.py
# documents).
#
# Source: ISTA-DASLab's GSQ-RCO GGUFs, shard 2 of -of-00002.  That shard is a single-tensor GGUF
# holding exactly this tensor (verified by reading its header: type 20 = IQ4_NL, dims 160 x
# 320001536), and it is byte-identical across that repo's IQ2_XS / IQ3_XXS / IQ3_S / Q2_0 builds
# (all 28.80 GB), i.e. it is the shared base table, not a per-quant artifact.
#
# Deliberately NOT from the Swift-1.5 or other finetune repos: those could carry a table that
# moved during finetuning, and we are quantising the official base.
#
#   bash tools/qwen38/fetch_ple.sh "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8/ple" [PID]
set -u
DEST=${1:?usage: fetch_ple.sh DEST [WAIT_PID]}
WAIT=${2:-}
REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
FILE=IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
OUT=Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf
BASE="https://huggingface.co/$REPO/resolve/main"
SIZE=${SIZE:-28800000000}     # replaced at run time by the real Content-Length
CONN=${CONN:-16}

if [ -n "$WAIT" ]; then
  echo "$(date '+%F %T') waiting for pid $WAIT to finish before starting the PLE fetch"
  while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
fi

mkdir -p "$DEST" || exit 1
cd "$DEST" || exit 1

if [ -f "$OUT.ok" ]; then echo "$(date '+%F %T') $OUT already present and verified"; exit 0; fi

# ask the server for the real size rather than trusting a hard-coded one
REAL=$(curl -sIL "$BASE/$FILE" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v); print v}')
[ -n "${REAL:-}" ] && SIZE=$REAL
echo "$(date '+%F %T') $OUT expected size $SIZE bytes ($(numfmt --to=iec "$SIZE"))"

for attempt in 1 2 3; do
  echo "$(date '+%F %T') attempt $attempt"
  aria2c -c -x"$CONN" -s"$CONN" -k1M --file-allocation=none \
         --max-tries=5 --retry-wait=10 --timeout=60 --console-log-level=warn \
         --summary-interval=60 --auto-file-renaming=false --allow-overwrite=true \
         -d "$DEST" -o "$OUT" "$BASE/$FILE" || true
  got=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$got" = "$SIZE" ]; then touch "$OUT.ok"; echo "$(date '+%F %T') $OUT size VERIFIED"; exit 0; fi
  echo "$(date '+%F %T') size $got / $SIZE - retrying"
done
echo "$(date '+%F %T') FAILED - re-run to resume"; exit 1
