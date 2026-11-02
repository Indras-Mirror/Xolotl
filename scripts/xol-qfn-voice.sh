#!/usr/bin/env bash
# xol-qfn-voice.sh - QFN (chat) + LFM3-Omni (ears + voice) on one card, then the Quetza CLI with voice on.
#
#   scripts/xol-qfn-voice.sh              start/attach both servers, then open QuetzaCodetl
#   scripts/xol-qfn-voice.sh --serve-only start/attach the servers and exit (they keep running)
#   scripts/xol-qfn-voice.sh [...]        anything else goes to xol-qfn -> xol-quetza
#
# One GPU, two engines: QFN's elastic expert cache takes all free VRAM at start (23.4 / 24.5 GB), and
# LFM3's vision + speech encoder then fails to load (`cudaMalloc failed: out of memory`, reproduced
# 2026-11-02).  POST /v1/vram asks QFN to hold RESERVE_MIB free; LFM3 loads in ~6 s (3.2 GB) and keeps
# ~1.7 GB headroom.  Measured with both up: ASR, TTS (24 kHz pcm) and /v1/decide answer concurrently,
# and QFN keeps its speed (a 96-token answer took 0.75 s while LFM3 streamed 3.5 s of speech).
#
# Quetza's turn-taking stays on QFN: /v1/decide is a route on the chat server (xol-quetza sets
# QUETZA_DECIDE_URL), independent of which server does the ears.
#
# Portable: the repo root comes from this script's location; only the two configs and the model
# assets are machine-specific.  Requires QuetzaCodetl's voice backends (src/services/voice/voiceBackends.ts,
# 2026-11-02) for QUETZA_VOICE_BACKEND=lfm3 to mean anything.
set -u

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${XOLOTL_PY:-$REPO_DIR/.venv/bin/python}
QFN_CONFIG=${XOLQFN_VOICE_QFN_CONFIG:-${QUETZA_SETTINGS_DIR:-$HOME/.quetza-data/xolotl}/xol-qfn45.json}
LFM3_CMD=${QUETZA_LFM3_SERVER:-$HOME/.local/bin/xol-lfm3-serve}
QUETZA_CMD=${XOLQFN_CMD:-$HOME/.local/bin/xol-qfn}
QFN_URL=${XOLQFN_VOICE_QFN_URL:-http://127.0.0.1:8131}
LFM3_URL=${QUETZA_LFM3_URL:-http://127.0.0.1:8132}
RESERVE_MIB=${XOLQFN_VOICE_RESERVE_MIB:-4608}
QFN_PORT=${QFN_URL##*:}
QFN_UNIT=xol-qfn-voice-qfn
LFM3_UNIT=xol-qfn-voice-lfm3

SERVE_ONLY=0
if [ "${1:-}" = "--serve-only" ]; then SERVE_ONLY=1; shift; fi

log_info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
log_ok()   { echo -e "\033[1;32m[+]\033[0m $*"; }
log_err()  { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

healthy() { curl -s --max-time 2 "$1/health" >/dev/null 2>&1; }
wait_healthy() {   # url seconds
    local i=0
    while [ "$i" -lt "$2" ]; do
        healthy "$1" && return 0
        sleep 1; i=$((i + 1))
    done
    return 1
}

STARTED_QFN=0
STARTED_LFM3=0
cleanup() {
    [ "$STARTED_LFM3" = 1 ] && systemctl --user stop "$LFM3_UNIT" >/dev/null 2>&1
    [ "$STARTED_QFN" = 1 ] && systemctl --user stop "$QFN_UNIT" >/dev/null 2>&1
    return 0
}
[ "$SERVE_ONLY" = 0 ] && trap cleanup EXIT INT TERM

# ---- 1. QFN (chat) ----------------------------------------------------------------------------
if healthy "$QFN_URL"; then
    log_ok "QFN already running at $QFN_URL - attaching"
else
    avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail" -lt 55 ]; then
        log_err "only ${avail} GiB RAM available - QFN needs ~50 GiB (arena + pinned KV); XOLQFN_FORCE=1 overrides"
        [ "${XOLQFN_FORCE:-0}" = 1 ] || exit 3
    fi
    log_info "starting QFN on $QFN_URL ($(basename "$QFN_CONFIG")) ..."
    systemctl --user is-active --quiet "$QFN_UNIT" && systemctl --user stop "$QFN_UNIT" >/dev/null 2>&1
    systemctl --user reset-failed "$QFN_UNIT" >/dev/null 2>&1
    systemd-run --user --unit="$QFN_UNIT" --nice=5 --working-directory="$REPO_DIR" \
        "$PY" -m serve.server --engine strata --config "$QFN_CONFIG" --host 127.0.0.1 --port "$QFN_PORT" >/dev/null || exit 4
    STARTED_QFN=1
    if ! wait_healthy "$QFN_URL" 300; then
        log_err "QFN not healthy in 300 s - see: journalctl --user -u $QFN_UNIT -n 40"
        cleanup; exit 5
    fi
    log_ok "QFN healthy (engine + expert cache resident)"
fi

# ---- 2. hand VRAM to the LFM3 encoder ---------------------------------------------------------
res=$(curl -s -X POST -H 'Content-Type: application/json' -d "{\"reserve_mib\":$RESERVE_MIB}" "$QFN_URL/v1/vram")
case "$res" in
    *'"status": "ok"'*) log_ok "QFN reserve ${RESERVE_MIB} MiB for LFM3: $(echo "$res" | grep -o '"vram_free_mib": [0-9]*')" ;;
    *) log_err "POST $QFN_URL/v1/vram failed: ${res:-no answer}"; exit 6 ;;
esac

# ---- 3. LFM3 (ears + voice) -------------------------------------------------------------------
if healthy "$LFM3_URL"; then
    log_ok "LFM3 already running at $LFM3_URL - reusing"
else
    log_info "starting LFM3-Omni on $LFM3_URL ($LFM3_CMD) ..."
    systemctl --user is-active --quiet "$LFM3_UNIT" && systemctl --user stop "$LFM3_UNIT" >/dev/null 2>&1
    systemctl --user reset-failed "$LFM3_UNIT" >/dev/null 2>&1
    systemd-run --user --unit="$LFM3_UNIT" --nice=5 "$LFM3_CMD" >/dev/null || exit 7
    STARTED_LFM3=1
    if ! wait_healthy "$LFM3_URL" 180; then
        log_err "LFM3 not healthy in 180 s - see: journalctl --user -u $LFM3_UNIT -n 40"
        cleanup; exit 8
    fi
    log_ok "LFM3 healthy (2B: ears, voice)"
fi

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[INFO] VRAM /'

# ---- 4. kill what the CLI leaves behind (Ctrl+C, /quit, a crash) --------------------------------
# The CLI kills its own VAD/TTS children on exit; this sweeps any that outlived it (a SIGKILL, a
# crash), but only ORPHANS - a listener whose parent is gone is reparented to init. A listener
# belonging to another window has a live parent and is left alone.
stop_orphan_voice() {
    local pids
    pids=$(ps -eo pid=,ppid=,args= 2>/dev/null | awk '
        ($2 == 1 || $2 == 0) && (
            /[q]uetza-vad-listener/ ||
            /[p]aplay --raw --rate=24000/ ||
            /[f]fmpeg .*-ar 24000.*-f s16le/
        ) { print $1 }')
    if [ -n "$pids" ]; then
        log_info "stopping voice leftovers: $(echo $pids | tr '\n' ' ')"
        kill $pids 2>/dev/null
    fi
    return 0
}

# ---- 5. voice on LFM3, then the CLI -----------------------------------------------------------
[ -z "${QUETZA_VOICE_STT:-}" ] && [ -z "${QUETZA_VOICE_TTS:-}" ] && export QUETZA_VOICE_BACKEND=${QUETZA_VOICE_BACKEND:-lfm3}
export QUETZA_LFM3_URL="$LFM3_URL"
export QUETZA_LFM3_SERVER="$LFM3_CMD"
export QUETZA_LFM3_VOICE=${QUETZA_LFM3_VOICE:-male}
# start the pipeline on REPL up (useChronosWake's QUETZA_VOICE_AUTOSTART path), so no /voice-auto needed
if [ "${XOLQFN_VOICE_NO_AUTOSTART:-0}" != 1 ]; then
    export QUETZA_VOICE_AUTOSTART=1
fi
log_info "voice backends: ${QUETZA_VOICE_BACKEND:+all=$QUETZA_VOICE_BACKEND }${QUETZA_VOICE_STT:+ears=$QUETZA_VOICE_STT }${QUETZA_VOICE_TTS:+voice=$QUETZA_VOICE_TTS}${QUETZA_VOICE_AUTOSTART:+ (autostart)}"

if [ "$SERVE_ONLY" = 1 ]; then
    log_ok "serve-only: QFN $QFN_URL, LFM3 $LFM3_URL left running"
    trap - EXIT INT TERM HUP
    exit 0
fi

# NOT exec: exec replaces this shell and drops the cleanup traps, which is why Ctrl+C used to
# leave both servers running (2026-11-02).
trap 'cleanup; stop_orphan_voice; exit 130' INT TERM HUP
"$QUETZA_CMD" "$@"
code=$?
cleanup
stop_orphan_voice
exit $code
