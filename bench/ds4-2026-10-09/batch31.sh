#!/bin/bash
# b31 (~9 min): soft pruning with the pruned experts on NVMe only (file tier, read on demand) - Mal's "prune more,
# fetch as needed": en+zh 25% at penalty 0.05 / 0.2, en+zh 40% at 0.2.  Same eval prompt + per-domain ppl as b30.
cd $HOME/AI/Strata-DS4 || exit 1
R=bench/ds4-2026-10-09/reap; export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.reap3
F="--vram-lru --pf-b 0.7 --f16-q8 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --comp-type iq4_nl --icomp-q8"
for x in "enzh25s05 enzh25 0.05" "enzh25s20 enzh25 0.2" "enzh40s20 enzh40 0.2"; do
  set -- $x
  DS4_PPL_DUMP=$R/nll-$1.f32 Q_PROMPT=$R/eval.i32 Q_N=64 bash bench/ds4-2026-10-08/q.sh $R/ev-$1.log $F --prune $R/prune-$2.txt --prune-penalty $3
  grep -E "prune:|tier:|^ppl|^decode :|experts/token|file reads|rc=" $R/ev-$1.log
done
python3 tools/ds4/reap_eval.py $R/eval.segments base=$R/nll-base.f32 $(for c in enzh25 enzh25s05 enzh25s20 enzh40s20; do [ -s $R/nll-$c.f32 ] && printf '%s=%s ' $c $R/nll-$c.f32; done)
