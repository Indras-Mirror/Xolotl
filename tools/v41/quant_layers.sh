#!/bin/bash
# quant_layers.sh QWORK FROM TO -- QUANT_ARGS...  - the quantize pass (v41_quant.py layer) over layers FROM..TO in
# its own work dir, then (TO = 39) its head gate.  calib_layers.sh's rules: one memguard'd GPU block per layer behind
# the shared lock (strata-glm slots in between), resumable (a layer is done when q/LNN.json + stats/LNN.json exist;
# init on a fresh dir), stoppable ($QWORK/STOP checked before each layer), stops on failure.
#   e.g.  quant_layers.sh /media/Crucial1TB/v41-quant-mixed 0 39 -- --map /media/Crucial1TB/v41-calib/alloc_full.json
#         quant_layers.sh /media/Crucial1TB/v41-quant-q2 0 39 -- --tier q2_0                  (the all-Q2_0 control)
# Log: $QWORK/quant.log, per layer $QWORK/logs/LNN.log.   Env: NEED_GIB=38 MAX_GIB=44
set -u
cd /home/mal/AI/Xolotl
W=$1; FROM=$2; TO=$3; shift 3; [ "${1:-}" = "--" ] && shift
mkdir -p "$W/logs"
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
mkdir -p "${LOCK%/*}"   # flock creates the file, not its directory (fresh box without Quetza, #140)
NEED=${NEED_GIB:-38}; MAX=${MAX_GIB:-44}
Q="python3 tools/v41/v41_quant.py"
log() { echo "$(date '+%F %T') $*" | tee -a "$W/quant.log"; }
stopped() { [ -f "$W/STOP" ] && { log "stop requested - exiting before layer $1"; rm -f "$W/current"; exit 0; }; }
gpu_wait() { until flock -n "$LOCK" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge "$NEED" ]; do
  stopped "$1"; sleep 30; done; }
trap 'rm -f "$W/current"' EXIT
log "driver start: layers $FROM..$TO, args: $* (pid $$)"
for L in $(seq "$FROM" "$TO"); do
  N=$(printf %02d "$L")
  if [ -f "$W/q/L$N.json" ] && [ -f "$W/stats/L$N.json" ]; then continue; fi
  stopped "$L"
  if [ ! -f "$W/h/L$N.bin" ]; then
    if [ "$L" = 0 ]; then
      log "init: embedding -> h/L00"
      nice -n 5 $Q init --qwork "$W" > "$W/logs/init.log" 2>&1 || { log "init failed"; exit 1; }
    else
      log "layer $L: input h/L$N.bin missing - cannot resume here (rerun from an earlier layer)"; exit 1
    fi
  fi
  for try in 1 2 3 4 5; do
    rows=$((32768 >> (try - 1)))          # CUDA OOM: retry with half the rows per GPU call (V41_ROWS), up to 5 tries
    echo "$L waiting-gpu" > "$W/current"
    gpu_wait "$L"
    stopped "$L"
    log "layer $L: start (try $try, V41_ROWS=$rows)"
    echo "$L running" > "$W/current"
    V41_ROWS=$rows PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- \
      nice -n 5 $Q layer "$L" --qwork "$W" --gc "$@" > "$W/logs/L$N.log" 2>&1
    rc=$?
    log "layer $L: rc=$rc $(grep -E 'quant L|done in|Error|error' "$W/logs/L$N.log" | tail -2 | tr '\n' ' ')"
    [ $rc -eq 0 ] && break
    grep -q "OutOfMemoryError\|CUDA out of memory" "$W/logs/L$N.log" || exit $rc
    [ "$try" = 5 ] && { log "layer $L: OOM 5 times - giving up"; exit $rc; }
    cp "$W/logs/L$N.log" "$W/logs/L$N.oom$try.log"; log "layer $L: OOM - retrying now with half the rows"; sleep 5
  done
done
log "layers $FROM..$TO finished"
if [ "$TO" = 39 ] && [ -f "$W/stats/L39.json" ] && [ ! -f "$W/stats/head_L40.json" ]; then
  gpu_wait 40
  log "head: final norm + head on h/L40 (quantized chain)"
  MEMGUARD_ALLOW_COMFY=0 bash tools/ds4/memguard.sh "$MAX" "$NEED" -- nice -n 5 $Q head --qwork "$W" > "$W/logs/head.log" 2>&1
  log "head: rc=$? $(grep -E 'ALL/eval|Error' "$W/logs/head.log" | tail -1)"
fi
