#!/bin/bash
DS=$HOME/AI/Xolotl
G="/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf"; EN="/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"; R="/media/mal/SSD NVME/AI/V41-GSQ/rank.bin"
SC=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf
W=$HOME/AI/Strata-Backup/v41-dev/d2; mkdir -p $W; cd $DS
run() { tag=$1; shift
  until [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge 75 ]; do sleep 5; done
  env "$@" STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- build-ds4-dspark/ds41_generate -m "$G" --ids-file $HOME/AI/Strata-Backup/v41-dev/r1ab-1237/p256.i32 -n 256 --temp 0 --backend cuda --experts gpu --engram "$EN" --rank "$R" --quiet $XARGS > $W/$tag.tok 2> $W/$tag.log
  echo "$tag exit=$?"; }
run prof_mx STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4
run prof_q3 STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_EXP_TYPE=q3_K
for f in prof_mx prof_q3; do echo "== $f"; tr '\r' '\n' < $W/$f.log | grep -E "^decode:|^DSpark|^pass_v|VRAM after|killed|ds41_generate:" | cut -c1-230; done
