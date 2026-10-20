#!/bin/bash
# b32 (~9 min): soft pruning, bigger penalties + --arena-adapt (an NVMe-read expert replaces its layer's LRU arena
# expert, so a conversation's pruned-but-needed experts migrate into RAM): 25% at 0.5 / 1.5, 40% at 1.5
cd $HOME/AI/Strata-DS4 || exit 1
R=bench/ds4-2026-10-09/reap; export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.reap3
F="--vram-lru --pf-b 0.7 --f16-q8 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --comp-type iq4_nl --icomp-q8 --arena-adapt"
for x in "enzh25a05 enzh25 0.5" "enzh25a15 enzh25 1.5" "enzh40a15 enzh40 1.5"; do
  set -- $x
  DS4_PPL_DUMP=$R/nll-$1.f32 Q_PROMPT=$R/eval.i32 Q_N=64 bash bench/ds4-2026-10-08/q.sh $R/ev-$1.log $F --prune $R/prune-$2.txt --prune-penalty $3
  grep -E "prune:|^prefill:|^ppl|^decode :|experts/token|file reads|rc=" $R/ev-$1.log
done
python3 tools/ds4/reap_eval.py $R/eval.segments base=$R/nll-base.f32 $(for c in enzh25 enzh25s20 enzh25a05 enzh25a15 enzh40a15; do [ -s $R/nll-$c.f32 ] && printf '%s=%s ' $c $R/nll-$c.f32; done)
