#!/bin/bash
# v41 DSpark phase-2 queue (session 7 -> 8).
#   1. bis_d0    STRATA_DSPARK_DEPTH=0   (single-token pass_v)   - does the pass_v machinery itself diverge?
#   2. bis_noacc STRATA_DSPARK_NOACC=1   (every draft rejected)  - do rejected positions leave residue in some state?
#   3. compare   both arms' token streams vs plain_nb.tok (the bias-off plain run).
#   4. prof_iq3  bias-ON depth-4 profile of the CHOSEN draft format (IQ3_XXS); compare vs prof_mx.log.
# Arms 1-2 run with decode biases OFF (--route-bias 0 --route-bias-file 0) to match plain_nb.tok.
# Every engine job runs under the shared ds4-gpu.lock via memguard and waits for >=75 GB free RAM.
# Cancel between runs: touch ~/AI/Strata-Backup/v41-dev/d2/queue.CANCEL
set -u
DS=$HOME/AI/Xolotl
G="/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf"
EN="/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"
R="/media/mal/SSD NVME/AI/V41-GSQ/rank.bin"
SC="/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf"
W="$HOME/AI/Strata-Backup/v41-dev/d2"; mkdir -p "$W"; cd "$DS" || exit 1
CANCEL="$W/queue.CANCEL"
rm -f "$CANCEL"

echo "== build (idempotent) $(date +%T)"
nice ninja -C "$DS/build-ds4-dspark" ds41_generate 2>&1 | tail -3

run() { tag=$1; shift
  if [ -e "$CANCEL" ]; then echo "$tag SKIPPED (cancel requested)"; return 0; fi
  until [ "$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)" -ge 75 ]; do sleep 5; done
  echo ">> $tag $(date +%T)"
  env "$@" STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- \
      build-ds4-dspark/ds41_generate -m "$G" --ids-file $HOME/AI/Strata-Backup/v41-dev/r1ab-1237/p256.i32 \
      -n 256 --temp 0 --backend cuda --experts gpu --engram "$EN" --rank "$R" --quiet $XARGS \
      > "$W/$tag.tok" 2> "$W/$tag.log"
  echo "$tag exit=$? $(date +%T)"
}
XARGS="--route-bias 0 --route-bias-file 0"
run bis_d0    STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=0
run bis_noacc STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_NOACC=1
XARGS=""
run prof_iq3  STRATA_DSPARK=$SC STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_EXP_TYPE=iq3_xxs

echo "===== compare (bisect arms vs plain_nb) ====="
W="$W" python3 - <<'PY'
import os
W = os.environ.get('W', '.')
def ids(p):
    try: return [int(l) for l in open(p) if l.strip().isdigit()]
    except FileNotFoundError: return []
a = ids(os.path.join(W, 'plain_nb.tok'))
print('plain_nb tokens:', len(a))
for f in ['bis_d0.tok', 'bis_noacc.tok']:
    b = ids(os.path.join(W, f)); n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    print(f, len(b), 'first divergence vs plain_nb:', d)
PY
echo "===== prof_iq3 ====="
tr '\r' '\n' < "$W/prof_iq3.log" | grep -E "^decode:|^DSpark:|^pass_v|VRAM after" | cut -c1-230
echo "QUEUE DONE $(date +%T)"
