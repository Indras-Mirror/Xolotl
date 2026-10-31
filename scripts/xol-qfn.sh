#!/usr/bin/env bash
# xol-qfn.sh - serve a qwen4exp pack on THIS checkout's engine, then open a client.
#
# Portable by construction: the repo root is derived from this script's location, the engine
# defaults to ./build-xolotl/strata in THAT checkout, and the expert profile / MTP draft come from
# that checkout's data/. A fresh clone + build needs no machine-specific edits -- only the model
# assets, which are not source:
#
#   ./scripts/xol-qfn.sh --model /path/to/model.gguf --pack /path/to/pack
#
#   xol-qfn.sh                      serve + open the Quetza CLI
#   xol-qfn.sh --serve-only         serve in the foreground, no client
#   xol-qfn.sh --status             what is running on which port
#   xol-qfn.sh --stop               stop the engine
#   xol-qfn.sh --print-config       show the resolved config and exit
#
# Env (all optional):
#   XOLQFN_MODEL   XOLQFN_PACK   the model + its pack (required unless the template's defaults exist)
#   XOLQFN_PORT    8099          XOLQFN_CTX  262144        XOLQFN_KVRES 32768
#   XOLQFN_MAX_GIB 84            cgroup MemoryMax       XOLQFN_NEED_GIB 62  free-RAM floor to start
#   XOLQFN_GUARD   1             set 0 to skip the cgroup + preflight guard
#   XOLQFN_PYTHON  python3       interpreter for serve/server.py (stdlib only, no venv needed)
set -u

SELF=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "$SELF")/.." && pwd)

PORT="${XOLQFN_PORT:-8099}"
CTX="${XOLQFN_CTX:-262144}"
KVRES="${XOLQFN_KVRES:-32768}"
MAX_GIB="${XOLQFN_MAX_GIB:-84}"
NEED_GIB="${XOLQFN_NEED_GIB:-62}"
GUARD="${XOLQFN_GUARD:-1}"
PY="${XOLQFN_PYTHON:-$(command -v python3)}"

TEMPLATE="$REPO/configs/xol-qfn.json"
ENGINE="$REPO/build-xolotl/strata"
EFFECTIVE="${XOLQFN_EFFECTIVE_CONFIG:-/tmp/xol-qfn.effective.json}"
LOG="${XOLQFN_LOG:-/tmp/xol-qfn-serve.log}"
PIDFILE="${XOLQFN_PIDFILE:-/tmp/xol-qfn.pid}"
UNIT="xol-qfn-serve"
URL="http://127.0.0.1:${PORT}"
MAX_WAIT="${XOLQFN_MAX_WAIT:-900}"

c_ok=$'\033[32m'; c_wa=$'\033[33m'; c_er=$'\033[31m'; c_di=$'\033[2m'; c_off=$'\033[0m'
log_info() { printf '%s[xol-qfn]%s %s\n' "$c_di" "$c_off" "$*"; }
log_ok()   { printf '%s[xol-qfn]%s %s\n' "$c_ok" "$c_off" "$*"; }
log_warn() { printf '%s[xol-qfn]%s %s\n' "$c_wa" "$c_off" "$*" >&2; }
log_err()  { printf '%s[xol-qfn]%s %s\n' "$c_er" "$c_off" "$*" >&2; }

avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
have_pid()  { local p; p=$(cat "$PIDFILE" 2>/dev/null); [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && printf '%s' "$p"; }
is_up()     { curl -sf --max-time 2 "$URL/v1/models" >/dev/null 2>&1; }

serve_only=0; action=""
MODEL="${XOLQFN_MODEL:-}"; PACK="${XOLQFN_PACK:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --pack)  PACK="$2";  shift 2 ;;
        --port)  PORT="$2"; URL="http://127.0.0.1:${PORT}"; shift 2 ;;
        --serve-only) serve_only=1; shift ;;
        --status|--stop|--print-config) action="${1#--}"; shift ;;
        -h|--help) sed -n '2,26p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) log_err "unknown argument: $1 (see --help)"; exit 2 ;;
    esac
done

resolve_config() {
    [ -f "$TEMPLATE" ] || { log_err "missing template: $TEMPLATE"; exit 2; }
    # Model assets are not source, so the template carries placeholders. Anything not supplied on
    # the command line / env falls back to the same directory convention used on the build box,
    # and preflight reports clearly if that path does not exist.
    [ -z "$MODEL" ] && MODEL="/media/mal/NVME1TB/Models/xol-qfn/Qwen3.8-Flash-Next-XQ-abl.gguf"
    [ -z "$PACK" ]  && PACK="/media/mal/NVME1TB/Models/xol-qfn/pack-xq"
    python3 - "$TEMPLATE" "$EFFECTIVE" "$REPO" "$MODEL" "$PACK" "$CTX" "$KVRES" "$PORT" "$LOG" <<'PY'
import json, sys
tpl, out, repo, model, pack, ctx, kvres, port, log = sys.argv[1:10]
s = open(tpl).read()
s = s.replace("@REPO@", repo).replace("@MODEL@", model).replace("@PACK@", pack)
s = s.replace("@CTX@", str(ctx)).replace("@KVRES@", str(kvres))
s = s.replace("@LOG@", log).replace("@NAME@", "xol-qfn")
c = json.loads(s)
c.pop("_comment", None)
c["port"] = int(port)
json.dump(c, open(out, "w"), indent=1)
print(out)
PY
}

show_status() {
    if is_up; then
        log_ok "engine UP on :$PORT (pid $(have_pid || echo '?'))"
        curl -s --max-time 3 "$URL/v1/models" | head -c 300; echo
    else
        log_warn "engine not answering on :$PORT"
    fi
    printf '  repo  : %s\n  engine: %s\n  model : %s\n  pack  : %s\n' "$REPO" "$ENGINE" "${MODEL:-<unset>}" "${PACK:-<unset>}"
}

do_stop() {
    local p
    if p=$(have_pid); then
        log_info "stopping pid $p"
        kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
        for _ in $(seq 1 15); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$p" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    systemctl --user stop "${UNIT}.scope" 2>/dev/null || true
    log_ok "engine stopped"
}

preflight() {
    [ -x "$ENGINE" ]      || { log_err "no engine at $ENGINE -- build it first (see docs/INSTALL.md)"; exit 2; }
    [ -f "$REPO/data/expert-profile.bin" ] || { log_err "missing $REPO/data/expert-profile.bin"; exit 2; }
    [ -d "$REPO/data/mtp/rt" ]             || { log_warn "no MTP draft at $REPO/data/mtp/rt (speculation will be off)"; }
    [ -f "$MODEL" ]       || { log_err "no model at '$MODEL' -- pass --model PATH"; exit 2; }
    [ -d "$PACK" ]        || { log_err "no pack at '$PACK' -- pass --pack DIR"; exit 2; }
    [ -x "$PY" ]          || { log_err "no python3"; exit 2; }
    # A multi-GB download/copy inflating page cache while the engine maps tens of GiB is how this
    # box froze on 2026-10-29 17:52. Refuse instead of racing it.
    if systemctl --user is-active --quiet swift-dl.service 2>/dev/null \
       && [ "${XOLQFN_ALLOW_BUSY:-0}" != 1 ]; then
        log_err "refusing: swift-dl.service is running (large download + arena = the 17:52 freeze)."
        log_err "  wait for it, or accept the risk: XOLQFN_ALLOW_BUSY=1"
        exit 3
    fi
    local a; a=$(avail_gib)
    if [ "$GUARD" = 1 ] && [ "$a" -lt "$NEED_GIB" ]; then
        log_err "refusing: MemAvailable ${a} GiB < ${NEED_GIB} GiB. Set XOLQFN_GUARD=0 to override."
        exit 3
    fi
    log_info "MemAvailable ${a} GiB (need ${NEED_GIB}, cap ${MAX_GIB}G)"
}

start_engine() {
    local fg="$1"
    if is_up; then log_ok "engine already up on :$PORT -- reusing"; return 0; fi
    preflight
    local -a cmd=("$PY" -m serve.server --engine strata --config "$EFFECTIVE" --port "$PORT")
    : > "$LOG"
    if [ "$GUARD" = 1 ]; then
        cmd=(systemd-run --user --scope -q --unit="$UNIT" -p "MemoryMax=${MAX_GIB}G" -p MemorySwapMax=0 -- "${cmd[@]}")
        log_info "guarded: ${UNIT}.scope MemoryMax=${MAX_GIB}G swap=0"
    fi
    if [ "$fg" = 1 ]; then
        log_info "serving in foreground (ctrl-c stops); log: $LOG"
        ( cd "$REPO" && "${cmd[@]}" ) 2>&1 | tee -a "$LOG"
        return $?
    fi
    ( cd "$REPO" && exec setsid nohup "${cmd[@]}" ) >>"$LOG" 2>&1 < /dev/null &
    local p=$!; echo "$p" > "$PIDFILE"
    log_info "engine starting (pid $p); waiting for $URL/v1/models (max ${MAX_WAIT}s)"
    local w=0
    while [ "$w" -lt "$MAX_WAIT" ]; do
        if is_up; then log_ok "engine ready"; sleep 1; return 0; fi
        if ! kill -0 "$p" 2>/dev/null; then
            log_err "engine died during load; tail of $LOG:"; tail -20 "$LOG" >&2; rm -f "$PIDFILE"; return 1
        fi
        sleep 3; w=$((w + 3))
        [ $((w % 30)) -eq 0 ] && log_info "  ... ${w}s"
    done
    log_err "not ready after ${MAX_WAIT}s; check $LOG"; return 1
}

case "$action" in
    status) resolve_config >/dev/null; show_status; exit 0 ;;
    stop)   do_stop; exit 0 ;;
    print-config) resolve_config >/dev/null; echo "$EFFECTIVE"; cat "$EFFECTIVE"; exit 0 ;;
esac

resolve_config >/dev/null
[ -n "$MODEL" ] || { log_err "no model given: pass --model PATH or set XOLQFN_MODEL"; exit 2; }

W=64; bar=$(printf '+%*s+' "$W" '' | tr ' ' '-')
printf '%s\n| %-*s |\n| %-*s |\n| %-*s |\n%s\n' "$bar" $((W-2)) "XOL-QFN - Xolotl engine" \
    $((W-2)) "$(basename "$MODEL")" $((W-2)) "port $PORT  ctx $CTX  kv int8  spec 4" "$bar"

if [ "$serve_only" = 1 ]; then start_engine 1; exit $?; fi
start_engine 0 || { log_err "engine failed to start"; exit 1; }

CLI="$REPO/dist/cli.mjs"
if [ ! -f "$CLI" ]; then
    log_ok "engine serving at $URL (no Quetza CLI in this checkout: $CLI)"
    log_ok "OpenAI: $URL/v1/chat/completions   Anthropic: $URL/v1/messages   UI: $URL/"
    exit 0
fi
log_ok "endpoint $URL -- 'scripts/xol-qfn.sh --stop' frees it"
env QUETZA_LOCAL_MODE=1 QUETZA_CODE_REPL=1 QUETZA_ENDPOINT="$URL" QUETZA_CONTEXT_SIZE="$CTX" \
    QUETZA_MODEL_NAME="xol-qfn" ANTHROPIC_BASE_URL="$URL" ANTHROPIC_API_KEY="local-dummy-key" \
    COLORTERM=truecolor CLAUDE_CODE_SYNTAX_HIGHLIGHT=1 DISABLE_TELEMETRY=1 DISABLE_AUTO_UPDATE=1 \
    SKIP_HOOKS=1 bun "$CLI" "$@"
