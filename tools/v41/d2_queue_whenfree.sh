#!/bin/bash
# Arm d2_queue.sh to run once the GPU has been IDLE for D2_IDLE_SECS (default 180) seconds.
#   IDLE = no nvidia-smi compute apps AND memory.used < 2000 MiB (the desktop baseline is ~334 MiB).
# Cancel:  touch ~/AI/Strata-Backup/v41-dev/d2/queue.CANCEL      (also cancels the queue between its runs)
# or:      kill "$(cat ~/AI/Strata-Backup/v41-dev/d2/whenfree.pid)"
set -u
DS=$HOME/AI/Xolotl
W=$HOME/AI/Strata-Backup/v41-dev/d2; mkdir -p "$W"
IDLE_SECS=${D2_IDLE_SECS:-180}
STEP=15
LOG="$W/whenfree.log"
CANCEL="$W/queue.CANCEL"
PIDFILE="$W/whenfree.pid"
echo $$ > "$PIDFILE"
idle() {
  local apps used
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ "${apps:-9}" = "0" ] && [ "${used:-99999}" -lt 2000 ]
}
echo "whenfree armed $(date +%T): need ${IDLE_SECS}s sustained GPU idle; cancel via $CANCEL" >> "$LOG"
streak=0
while :; do
  if [ -e "$CANCEL" ]; then echo "cancel requested $(date +%T)" >> "$LOG"; rm -f "$PIDFILE"; exit 0; fi
  if idle; then streak=$((streak + STEP)); else streak=0; fi
  if [ "$streak" -ge "$IDLE_SECS" ]; then
    echo "GPU idle ${IDLE_SECS}s -> launching queue $(date +%T)" >> "$LOG"
    bash "$DS/tools/v41/d2_queue.sh" >> "$LOG" 2>&1
    echo "queue finished $(date +%T)" >> "$LOG"
    rm -f "$PIDFILE"; exit 0
  fi
  sleep "$STEP"
done
