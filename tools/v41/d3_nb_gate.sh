#!/bin/bash
# correctness gate, decode biases OFF (routing then matches the prefill's): D3 (nb_a) vs per-token (nb_b), -n 96,
# each dumps its verify rows; then an independent prefill of each stream (arb_nb_*).
DS=$HOME/AI/Xolotl
G="/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf"
EN="/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"
R="/media/mal/SSD NVME/AI/V41-GSQ/rank.bin"
SC=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf
W=$HOME/AI/Strata-Backup/v41-dev/d2; cd "$DS" || exit 1
run() { tag=$1; shift
  until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 75 ]; do sleep 5; done
  echo ">> $tag $(date +%T)"
  env "$@" STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- \
      build-ds4-dspark/ds41_generate -m "$G" --ids-file $HOME/AI/Strata-Backup/v41-dev/r1ab-1237/p256.i32 \
      -n 96 --temp 0 --backend cuda --experts gpu --engram "$EN" --rank "$R" --quiet --route-bias 0 --route-bias-file 0 \
      > "$W/$tag.tok" 2> "$W/$tag.log"
  echo "$tag exit=$? $(date +%T)"
}
run nb_a STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4 STRATA_TIER_BATCH=1 STRATA_DSPARK_DUMP=$W/nb_a.vdump
run nb_b STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_DUMP=$W/nb_b.vdump
python3 - <<'PY'
import struct
p=open('/home/mal/AI/Strata-Backup/v41-dev/r1ab-1237/p256.i32','rb').read()
for tag in ['nb_a','nb_b']:
    toks=[int(l) for l in open('/home/mal/AI/Strata-Backup/v41-dev/d2/'+tag+'.tok') if l.strip().lstrip('-').isdigit()]
    open('/home/mal/AI/Strata-Backup/v41-dev/d2/arb_'+tag+'.i32','wb').write(p+struct.pack('<%di'%len(toks),*toks))
PY
bash "$W/arb_run.sh" nb_a nb_b
echo "NBGATE DONE $(date +%T)"
