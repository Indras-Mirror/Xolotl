#!/bin/bash
# b30 (~14 min, two blocks if GLM needs the GPU between): REAP evals on the held-out 9.6K prompt.  Mal cares about
# English + Chinese (+ code): the en+zh+code calibration is primary, at 15% and 25%; all-domain 25% and random 25%
# (control) for comparison; fr/ja/pt are never calibrated (canary for language-specific experts).  Per-domain ppl via DS4_PPL_DUMP + tools/ds4/reap_eval.py
cd $HOME/AI/Strata-DS4 || exit 1
R=bench/ds4-2026-10-09/reap; export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.reap2
F="--vram-lru --pf-b 0.7 --f16-q8 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --comp-type iq4_nl --icomp-q8"
for c in ${CFGS:-base enzh15 enzh25 all25 rand25}; do
  P=(); [ $c != base ] && P=(--prune $R/prune-$c.txt)
  DS4_PPL_DUMP=$R/nll-$c.f32 Q_PROMPT=$R/eval.i32 Q_N=64 bash bench/ds4-2026-10-08/q.sh $R/ev-$c.log $F "${P[@]}"
  grep -E "prune:|tier:|^ppl|^decode :|experts/token|rc=" $R/ev-$c.log
done
python3 tools/ds4/reap_eval.py $R/eval.segments base=$R/nll-base.f32 $(for c in enzh15 enzh25 all25 rand25; do [ -s $R/nll-$c.f32 ] && printf '%s=%s ' $c $R/nll-$c.f32; done)
