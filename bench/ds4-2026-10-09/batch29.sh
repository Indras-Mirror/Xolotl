#!/bin/bash
# b29 (~5 min): REAP saliency calibration on the real model - all-domain and en+zh+code-only prompts (8.4K tokens each)
cd $HOME/AI/Strata-DS4 || exit 1
R=bench/ds4-2026-10-09/reap; export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.reap2
F="--vram-lru --pf-b 0.7 --f16-q8 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --comp-type iq4_nl --icomp-q8"
for c in cal cal_enzh; do
  Q_PROMPT=$R/$c.i32 Q_N=4 Q_PPL= bash bench/ds4-2026-10-08/q.sh $R/sal-$c.log $F --saliency $R/sal-$c.bin
  grep -E "saliency:|prefill|rc=" $R/sal-$c.log
done
