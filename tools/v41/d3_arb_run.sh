#!/bin/bash
# independent full-path prefill of prompt + a DSpark stream: every row's logits (row p predicts token p+1)
DS=$HOME/AI/Xolotl
G="/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf"
EN="/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"
R="/media/mal/SSD NVME/AI/V41-GSQ/rank.bin"
W=$HOME/AI/Strata-Backup/v41-dev/d2; cd "$DS" || exit 1
for tag in "$@"; do
  until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 75 ]; do sleep 5; done
  echo ">> arb_$tag $(date +%T)"
  STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- build-ds4-dspark/ds41_generate -m "$G" --ids-file "$W/arb_$tag.i32" \
      -n 1 --temp 0 --backend cuda --experts gpu --engram "$EN" --rank "$R" --quiet --dump-logits "$W/arb_$tag.logits" \
      > "$W/arb_$tag.tok" 2> "$W/arb_$tag.log"
  echo "arb_$tag exit=$? $(date +%T)"
done
echo "ARB DONE $(date +%T)"
[ -n "$NO_BCAST" ] || $HOME/.local/bin/relay broadcast "strata-dspark: prefill checks finished - 4090 free for now" --name strata-dspark >/dev/null 2>&1
