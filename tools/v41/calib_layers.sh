#!/bin/bash
# calib_layers.sh FROM TO - run v41_calib.py layer FROM..TO in order, each as its own GPU block.  Normally started
# and stopped through calibctl.sh (alias v41calib), not directly.
#   Resumable: a layer is done once stats/LNN.json exists (written last, after h/L+1); done layers are skipped, and
#   the first undone layer L needs h/LL.bin (kept: --gc only deletes h/L-1 after layer L succeeds).  First start
#   (nothing done, no h/L00.bin) runs `init`.
#   Stoppable: $W/STOP is checked before each layer and while waiting -> exit 0 cleanly (calibctl stop).
#   Each layer waits for its shard (.ok; layer L = shard L+3, engram shards 47/48 for layers 1/14), then for the
#   shared GPU lock AND MemAvailable >= NEED_GIB, then runs under memguard (which holds the lock for that layer
#   only, so strata-glm can slot in between layers).  Stops on failure.
#   Also waits for <3 GB VRAM in use.  IDLE_MIN=N: the first layer starts only after the lock + GPU have stayed free
#   N minutes (calibctl start-when-idle: wait for strata-glm to finish).  A CUDA OOM retries the layer up to 4 more times, halving V41_ROWS each time.
#   Log: $V41_WORK/calib.log (+ one log per layer in $V41_WORK/logs/).   Env: NEED_GIB=38 MAX_GIB=44 IDLE_MIN=0 EXTRA=""
set -u
cd /home/mal/AI/Xolotl
W=${V41_WORK:-/media/Crucial1TB/v41-calib}; mkdir -p "$W/logs"
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
mkdir -p "${LOCK%/*}"   # flock creates the file, not its directory (fresh box without Quetza, #140)
NEED=${NEED_GIB:-38}; MAX=${MAX_GIB:-44}; IDLE=${IDLE_MIN:-0}
log() { echo "$(date '+%F %T') $*" | tee -a "$W/calib.log"; }
stopped() { [ -f "$W/STOP" ] && { log "stop requested - exiting before layer $1"; rm -f "$W/current"; exit 0; }; }
# V41_SRC="dir1:dir2:..." = another checkpoint's shard dirs (also read by v41_calib.py); unset = the dealignai FP8.
if [ -n "${V41_SRC:-}" ]; then IFS=: read -r -a SRC <<< "$V41_SRC"
else SRC=(/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
          "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"); fi
ok() { for d in "${SRC[@]}"; do [ -f "$d/$1.ok" ] && return 0; done; return 1; }
trap 'rm -f "$W/current"' EXIT
log "driver start: layers $1..$2 (pid $$)"
for L in $(seq "$1" "$2"); do
  N=$(printf %02d "$L")
  if [ -f "$W/stats/L$N.json" ]; then continue; fi
  stopped "$L"
  if [ ! -f "$W/h/L$N.bin" ]; then
    if [ "$L" = 0 ]; then
      log "init: embedding -> h/L00"
      nice -n 5 python3 tools/v41/v41_calib.py init > "$W/logs/init.log" 2>&1 || { log "init failed"; exit 1; }
    else
      log "layer $L: input h/L$N.bin missing - cannot resume here (rerun from an earlier layer)"; exit 1
    fi
  fi
  need="model-$(printf %05d $((L + 3)))-of-00048.safetensors"
  [ "$L" = 1 ] && need="$need model-00047-of-00048.safetensors"
  [ "$L" = 14 ] && need="$need model-00048-of-00048.safetensors"
  for f in $need; do
    if ! ok "$f"; then log "layer $L: waiting for $f"; echo "$L waiting-shard" > "$W/current"
      until ok "$f"; do stopped "$L"; sleep 60; done; fi
  done
  for try in 1 2 3 4 5; do
    rows=$((32768 >> (try - 1)))          # V41_ROWS per try: 32768, 16384, 8192, 4096, 2048
    echo "$L waiting-gpu" > "$W/current"
    if [ "$IDLE" -gt 0 ]; then log "layer $L: waiting for the GPU lock to stay free ${IDLE} min (strata-glm idle)"; fi
    free_since=""
    while :; do
      stopped "$L"
      if flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ] \
         && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 3000 ]; then
        free_since=${free_since:-$(date +%s)}
        [ $(( $(date +%s) - free_since )) -ge $((IDLE * 60)) ] && break
      else free_since=""; fi
      sleep 30
    done
    IDLE=0                   # only the first layer waits for a sustained-idle GPU; later layers interleave per the lock
    stopped "$L"
    log "layer $L: start (try $try, V41_ROWS=$rows)"
    echo "$L running" > "$W/current"
    V41_ROWS=$rows PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 python3 tools/v41/v41_calib.py layer "$L" --gc ${EXTRA:-} \
      > "$W/logs/L$N.log" 2>&1
    rc=$?
    log "layer $L: rc=$rc $(grep -E 'done in|Error|error' "$W/logs/L$N.log" | tail -1)"
    [ $rc -eq 0 ] && break
    # unattended runs: a CUDA OOM (e.g. another process took VRAM) gets 2 more tries after a pause; anything else stops
    grep -q "OutOfMemoryError\|CUDA out of memory" "$W/logs/L$N.log" || exit $rc
    [ "$try" = 5 ] && { log "layer $L: OOM 5 times - giving up"; exit $rc; }
    cp "$W/logs/L$N.log" "$W/logs/L$N.oom$try.log"; log "layer $L: OOM - retrying now with half the rows"; sleep 5
  done
done
log "layers $1..$2 finished"
if [ "$2" = 39 ] && [ -f "$W/stats/L39.json" ] && [ ! -f "$W/stats/head_L40.json" ]; then   # end-of-chain gate
  until flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ]; do
    stopped 40; sleep 30; done
  log "head: final norm + head on h/L40"
  MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 python3 tools/v41/v41_calib.py head \
    > "$W/logs/head.log" 2>&1
  log "head: rc=$? $(grep -E 'ALL/eval|Error' "$W/logs/head.log" | tail -1)"
fi
