#!/usr/bin/env bash
# The --300k server thought in garbage (base64 / broken comments). A/B: the exact ablated serve args, greedy, a
# 33-token chat prompt with <think>, -n 600; only --ctx differs (524288 = unchanged by the CAPMAX edit; 307200 = new).
set -u; cd "$(dirname "$0")/../.."
D=bench/glm-2026-10-09; S=$D/garbage.summary; : > $S
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 5; done
ARGS=$(python3 -c "import json;a=json.load(open('tools/glm/serve/strata-glm-unc-ablated.json'))['args'];i=a.index('--ctx');del a[i:i+2];print(' '.join(a))")
for ctx in 307200 524288; do
  echo "--- ctx $ctx $(date +%T) ---" | tee -a $S
  bash tools/ds4/memguard.sh 84 70 -- build-glm-gpu/glm_generate $ARGS --ctx $ctx --temp 0 \
     --ids-file $D/prompts/chat_pal.i32 -n 600 > $D/garbage-$ctx.log 2>&1
  echo "exit $?" | tee -a $S
  grep -hE "^decode :|out of memory|error" $D/garbage-$ctx.log | tee -a $S
  $HOME/AI/Strata/.venv/bin/python - $D/garbage-$ctx.log <<'PY' | tee -a $S
import sys, re
sys.path.insert(0, "tools/glm/serve"); sys.path.insert(0, "/home/mal/AI/Strata/tools")
import glm_tok
ids = [int(l) for l in open(sys.argv[1], errors="replace") if re.fullmatch(r"\d+\n?", l)]
t = glm_tok.load("/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer").decode(ids)
print(f"[{len(ids)} ids] HEAD: {t[:700]!r}\n  TAIL: {t[-500:]!r}")
PY
done
echo "=== garbage done $(date +%T) ===" | tee -a $S
