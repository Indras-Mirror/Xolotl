#!/bin/bash
DS=$HOME/AI/Xolotl
G="/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf"; EN="/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"; R="/media/mal/SSD NVME/AI/V41-GSQ/rank.bin"
SC=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf
W=$HOME/AI/Strata-Backup/v41-dev/d2; mkdir -p $W; cd $DS
run() { tag=$1; shift
  until [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge 75 ]; do sleep 5; done
  env "$@" STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- build-ds4-dspark/ds41_generate -m "$G" --ids-file $HOME/AI/Strata-Backup/v41-dev/r1ab-1237/p256.i32 -n 256 --temp 0 --backend cuda --experts gpu --engram "$EN" --rank "$R" --quiet $XARGS > $W/$tag.tok 2> $W/$tag.log
  echo "$tag exit=$?"; }
XARGS="--route-bias 0 --route-bias-file 0"
run plain_nb
run spec4_nb STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4
