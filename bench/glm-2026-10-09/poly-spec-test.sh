#!/usr/bin/env bash
# The PolyStrata-style speed test: run THEIR engine (strata-poly, CUDA) on OUR file, with and without the draft block.
#   nodraft : --no-draft            (their "26.7-27.7 t/s" arm)
#   draft   : --draft-model BLOCK   (their "47-51 t/s" arm)
# Waits for the ds4-gpu lock (our other test holds it), then runs. Never touches our engine.
set -u
cd $HOME/AI/PolyStrata
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
BLK=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/draft-block.gguf
D=$HOME/AI/Strata-GLM/bench/glm-2026-10-09
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
IDS=$(python3 -c "import numpy as np; a=np.fromfile('$D/prompts/eval_code.i32',dtype='<i4')[:2000]; print(','.join(map(str,a.tolist())))")
: > "$D/poly-spec.summary"
echo "=== poly-spec start $(date +%T)  prompt=${IDS:0:24}... ===" | tee "$D/poly-spec.summary"
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 30; done
echo "lock free at $(date +%T)" | tee -a "$D/poly-spec.summary"
for tag in nodraft draft; do
  R=(); [ "$tag" = draft ] && R=(--draft-model "$BLK")
  echo "--- $tag $(date +%T) ---" | tee -a "$D/poly-spec.summary"
  ./build/strata-poly --model "$M" --max-context 32768 --expert-profile glm.profile "${R[@]}" \
      --tokens "$IDS" --max-new 128 > "/tmp/poly-$tag.log" 2>&1
  echo "exit $?" | tee -a "$D/poly-spec.summary"
  grep -iE "token|t/s|/s\b|decode|writ|guess|draft|accept" "/tmp/poly-$tag.log" | tail -10 | tee -a "$D/poly-spec.summary"
done
echo "=== poly-spec done $(date +%T) ===" | tee -a "$D/poly-spec.summary"
