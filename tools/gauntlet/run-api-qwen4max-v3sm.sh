#!/bin/bash
# Qwen4-Max (OpenRouter) on Gauntlet v3-small — the API baseline the local Qwen4 quant is measured against.
#
# Same task set and turns as the local v3-small runs (xq/tools/qx/qfnara-gsmall.sh):
#   01,03,04,06,09,10 · GAUNTLET_TURNS=10
#
# Three arms, because greedy and the thinking budget both move a model's result:
#   greedy  - temperature 0, thinking budget 32768 (the README's API-score convention; reruns repeat exactly)
#   t1      - temperature 1.0 (Qwen's recommended sampling), same 32K thinking budget
#   max     - temperature 1.0 AND reasoning_effort=max — the model at its own maximum depth.
#             NOTE: taskeval.chat() deliberately drops reasoning_effort whenever a thinking budget is set,
#             so the max arm MUST run without TASKEVAL_THINK_BUDGET (the effort knob is the budget then).
#             Measured on one prompt (temp 0): default 1,988 reasoning tokens, effort=high 1,970 (no effect),
#             effort=max 13,886 (~7x). GAUNTLET_MAX_TOKENS is raised so a long chain is not cut at 36,864,
#             and GAUNTLET_TASK_TOKENS bounds each task so one runaway task cannot drain the balance.
#
# Usage:  bash tools/gauntlet/run-api-qwen4max-v3sm.sh [greedy|t1|max|all]
# Key:    $OPENROUTER_API_KEY, else ~/.quetza-data/qwen4-quetza/settings.json (env.OPENROUTER_API_KEY)
# Output: out/gauntlet-v3sm-qwen4max-or[-t1|-max]/gauntlet.json + gauntlet.log
set -u
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
O=$X/out
MODEL=${QWEN4_OR_MODEL:-qwen/qwen4-max}
ONLY=${GAUNTLET_ONLY:-01,03,04,06,09,10}
PY=/home/mal/anaconda3/bin/python3
MAX_STEP=${QWEN4_MAX_STEP_TOKENS:-65536}
MAX_TASK=${QWEN4_MAX_TASK_TOKENS:-120000}

KEY="${OPENROUTER_API_KEY:-}"
if [ -z "$KEY" ]; then
  KEY=$($PY -c "import json;print(json.load(open('$HOME/.quetza-data/qwen4-quetza/settings.json'))['env'].get('OPENROUTER_API_KEY',''))" 2>/dev/null)
fi
[ -z "$KEY" ] && { echo "no OpenRouter key (OPENROUTER_API_KEY or ~/.quetza-data/qwen4-quetza/settings.json)" >&2; exit 1; }

go() { # outdir env-assignments...
  local out=$1; shift
  [ -f "$out/gauntlet.json" ] && { echo "== $(basename "$out"): already has gauntlet.json, skipping"; return; }
  echo "== $(basename "$out") $(date '+%F %T')  model=$MODEL tasks=$ONLY  $*"
  env TASKEVAL_KEY="$KEY" TASKEVAL_MODEL="$MODEL" GAUNTLET_TURNS=10 "$@" \
    timeout 28800 $PY "$X/tools/gauntlet/gauntlet.py" run https://openrouter.ai/api "$out" --only "$ONLY" \
      > "$out.log" 2>&1 || echo "  run exited non-zero"
  $PY "$X/tools/gauntlet/gauntlet.py" score "$out" 2>&1 | tail -1
}

cd "$X" || exit 1
case ${1:-greedy} in
  greedy) go $O/gauntlet-v3sm-qwen4max-or    TASKEVAL_NO_EFFORT=1 TASKEVAL_THINK_BUDGET=32768 TASKEVAL_TEMPERATURE=0 ;;
  t1)     go $O/gauntlet-v3sm-qwen4max-or-t1 TASKEVAL_NO_EFFORT=1 TASKEVAL_THINK_BUDGET=32768 TASKEVAL_TEMPERATURE=1.0 ;;
  max)    go $O/gauntlet-v3sm-qwen4max-or-max TASKEVAL_EFFORT=max TASKEVAL_TEMPERATURE=1.0 \
             GAUNTLET_MAX_TOKENS=$MAX_STEP GAUNTLET_TASK_TOKENS=$MAX_TASK ;;
  all)    go $O/gauntlet-v3sm-qwen4max-or    TASKEVAL_NO_EFFORT=1 TASKEVAL_THINK_BUDGET=32768 TASKEVAL_TEMPERATURE=0
          go $O/gauntlet-v3sm-qwen4max-or-t1 TASKEVAL_NO_EFFORT=1 TASKEVAL_THINK_BUDGET=32768 TASKEVAL_TEMPERATURE=1.0
          go $O/gauntlet-v3sm-qwen4max-or-max TASKEVAL_EFFORT=max TASKEVAL_TEMPERATURE=1.0 \
             GAUNTLET_MAX_TOKENS=$MAX_STEP GAUNTLET_TASK_TOKENS=$MAX_TASK ;;
  *)      echo "usage: $0 [greedy|t1|max|all]" >&2; exit 2 ;;
esac
