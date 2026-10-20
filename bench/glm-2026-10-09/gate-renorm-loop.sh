#!/usr/bin/env bash
# Behavioral (loop) gate for --renorm-skip: start the ABLATED server with skip 0.15 + --renorm-skip and run the
# s20 keylogger prompt.  s20: ablated + skip (no renorm) degenerates into "import LPVOID, LPVOID, ..." x107,
# finish=length.  With renorm the answer should terminate cleanly (finish=stop).
set -u
cd $HOME/AI/Strata-GLM
PY=$HOME/AI/Strata/.venv/bin/python
SHIM=tools/glm/serve/serve_glm.py
CFG=tools/glm/serve/strata-glm-unc-ablated.json   # fully ablated + skip 0.15 + --renorm-skip
PORT=8140
D=bench/glm-2026-10-09
LOG=/tmp/renorm-abl-serve.log

[ -x "$PY" ] || { echo "missing $PY"; exit 1; }
setsid "$PY" "$SHIM" --engine strata --config "$CFG" --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "=== loop-gate start $(date +%T)  serve pid $PID ===" | tee "$D/renorm-loop.summary"
for i in $(seq 1 300); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready after ${i}0s" | tee -a "$D/renorm-loop.summary"; break; }
    kill -0 "$PID" 2>/dev/null || { echo "server died" | tee -a "$D/renorm-loop.summary"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
"$PY" "$D/probe_abl.py" "$PORT" renorm-abl '{"max_tokens":900,"reasoning_effort":"low"}' \
    > "$D/renorm-abl-probe.txt" 2>&1
echo "probe exit $?" | tee -a "$D/renorm-loop.summary"
grep -E "^=== \[|finish=|reason_chars|content_chars" "$D/renorm-abl-probe.txt" | tee -a "$D/renorm-loop.summary"
kill -- -"$PID" 2>/dev/null; sleep 2; kill -9 -- -"$PID" 2>/dev/null
echo "=== loop-gate done $(date +%T) ===" | tee -a "$D/renorm-loop.summary"
