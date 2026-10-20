#!/bin/bash
# b33 (~10-12 min): long-context decode probes (--pos-offset, empty caches: attention cost real, hit rate optimistic)
# 256K / 512K x unpruned / soft-pruned 25% (enzh, penalty 0.5, --arena-adapt); rotated iq4_nl KV + icomp q8
cd $HOME/AI/Strata-DS4 || exit 1
O=bench/ds4-2026-10-09/longctx; R=bench/ds4-2026-10-09/reap; B=bench/ds4-2026-10-08
export DS4_BIN=$B/bin/ds4_generate.reap3 Q_PROMPT=$B/ctx/probe16.i32 Q_N=48 Q_PPL=
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58 --vram-margin 2.0 --comp-type iq4_nl --icomp-q8"
P="--prune $R/prune-enzh25.txt --prune-penalty 0.5 --arena-adapt"
for x in "p256k 262144 -" "p256k-prune 262144 P" "p512k 524288 -" "p512k-prune 524288 P"; do
  set -- $x; X=; [ "$3" = P ] && X=$P
  bash $B/q.sh $O/b33-$1.log $F --pos-offset $2 $X
  printf '%-14s %s | %s\n' $1 "$(grep -h '^decode :' $O/b33-$1.log | sed 's/.*= //')" "$(grep -h 'slots auto' $O/b33-$1.log | grep -o '[0-9]* slots')"
  grep -hE 'decode ms/token|experts/token|file reads|prune:|rc=' $O/b33-$1.log | sed 's/^/    /'
done
