#!/usr/bin/env bash
# Maya-S-v2 with its own MTP/NextN block (blk.45): PolyStrata's engine (has GLM spec decode) no-draft vs draft, then
# our engine on the same file (no MTP; the s25e wiring).  eval_code 2000 + 128, like poly-spec-test.sh on RCO
# (RCO there: 5.05 no-draft / 5.17 draft).
set -u
D=$HOME/AI/Strata-GLM/bench/glm-2026-10-09
M=/media/Crucial1TB/models/GLM-5.3-Flash-Maya-S-v2/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S=$D/maya-mtp.summary; : > $S
IDS=$(python3 -c "import numpy as np; a=np.fromfile('$D/prompts/eval_code.i32',dtype='<i4')[:2000]; print(','.join(map(str,a.tolist())))")
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 5; done
cd $HOME/AI/PolyStrata
for tag in nodraft draft; do
  R=(); [ "$tag" = nodraft ] && R=(--no-draft)
  echo "--- poly $tag $(date +%T) ---" | tee -a $S
  flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=84G -p MemorySwapMax=0 \
    ./build/strata-poly --model "$M" --max-context 32768 --expert-profile $D/maya.profile "${R[@]}" \
    --tokens "$IDS" --max-new 128 > $D/maya-poly-$tag.log 2>&1
  echo "exit $?" | tee -a $S
  grep -iE "t/s|tok/s|decode|draft|accept|guess|error|unsupported|cannot" $D/maya-poly-$tag.log | tail -8 | tee -a $S
done
cd $HOME/AI/Strata-GLM
echo "--- ours (no MTP) $(date +%T) ---" | tee -a $S
bash tools/ds4/memguard.sh 84 70 -- build-glm-gpu/glm_generate -m "$M" --backend cuda --experts gpu --arena-gib 72 \
  --arena-skip-resident --ctx 32768 --prefill-chunk 1024 --chunk-mmq --vram-margin 5.5 --pcie 1.0 \
  --stop 154820,154827,154829 --ids-file $D/prompts/eval_code.i32 -n 128 > $D/maya-ours.log 2>&1
echo "exit $?" | tee -a $S
grep -hE "slots auto|^tier:|^prefill:|^decode :|experts/token|out of memory|error|no kernel|native" $D/maya-ours.log | tee -a $S
echo "=== maya-mtp done $(date +%T) ===" | tee -a $S
