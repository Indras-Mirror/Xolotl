#!/usr/bin/env bash
# s26b: the second wave on Maya-S-v2 (ablated, ctx 300K, census + vram-lru from wave 1), one lever at a time:
#  G pcieauto  : --pcie auto (the CPU pool now applies the LoRA delta too -> FreeToken-style CPU/PCIe balance)
#  H q4dense   : G + --dense-requant q4_k (Maya-S24's dense layout built at load: less dense read, more slots)
#  I bias02    : H + --route-bias 0.02 (cache-aware routing: resident experts win near-ties)
#  J ctx512k   : H at --ctx 524288
# Quality: chunked ppl on eval_chat / eval_code for the base and every math-changing arm (requant, route bias).
#  K elastic4k : H + 4096-token chunks with the per-request elastic cache (margin 7.5 for the prompt, --vram-grow
#                1.5 for decode: Ds4MoeTier::shrink_cache/grow_cache)    L elastic6k: the same at 6144
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M="/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
TOK=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer
D=bench/glm-2026-10-09
CEN0=$HOME/.quetza-data/strata-glm/maya-s-v2.census
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S=$D/maya-improve2.summary
: > $S
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --chunk-mmq
        --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt
        --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip
        --stop 154820,154827,154829 --lora "$A" --lora-exps)
text() {
  $HOME/AI/Strata/.venv/bin/python - "$1" "$TOK" <<'EOF' 2>&1
import sys, re
sys.path.insert(0, "tools/glm/serve"); sys.path.insert(0, "/home/mal/AI/Strata/tools")
import glm_tok
ids = [int(l) for l in open(sys.argv[1], errors="replace") if re.fullmatch(r"\d+\n?", l)]
print(f"  text [{len(ids)} ids]: {glm_tok.load(sys.argv[2]).decode(ids)[:500]!r}")
EOF
}
run() {   # name -- args
  local name=$1; shift; [ "${1:-}" = "--" ] && shift
  echo "--- $name ($*) $(date +%T) ---" | tee -a $S
  : > $D/mi2-$name.vram
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $D/mi2-$name.vram; sleep 0.2; done ) &
  local smi=$!
  cp $CEN0 $D/mi2-$name.census
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --census $D/mi2-$name.census "$@" > $D/mi2-$name.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  grep -hE "grow_cache|slots auto|^tier:|^prefill:|^decode :|ppl:|experts/token|tier ms/token|decode ms/token|out of memory|error" $D/mi2-$name.log | tee -a $S
  echo "  exit $rc, VRAM peak $(sort -n $D/mi2-$name.vram | tail -1) MiB" | tee -a $S
  grep -q "ppl:" $D/mi2-$name.log || text $D/mi2-$name.log | tee -a $S
}
T=(--temp 0 --ids-file $D/prompts/chat_code6k.i32 -n 256)
PPL=(--ctx 4096 --prefill-chunk 1024 --vram-margin 5.5 -n 1 --ppl)
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 10; done
run G-pcieauto -- --ctx 307200 --prefill-chunk 1024 --vram-margin 5.5 "${T[@]}" --pcie auto
run H-q4dense  -- --ctx 307200 --prefill-chunk 1024 --vram-margin 5.5 "${T[@]}" --pcie auto --dense-requant q4_k
run K-elastic4k -- --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 "${T[@]}" --pcie auto --dense-requant q4_k
run L-elastic6k -- --ctx 307200 --prefill-chunk 6144 --vram-margin 10 --vram-grow 1.5 "${T[@]}" --pcie auto --dense-requant q4_k
run I-bias02   -- --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 "${T[@]}" --pcie auto --dense-requant q4_k --route-bias 0.02
run J-ctx512k  -- --ctx 524288 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 "${T[@]}" --pcie auto --dense-requant q4_k
for p in chat code; do
  run Q-base-$p   -- "${PPL[@]}" --ids-file $D/prompts/eval_$p.i32 --pcie auto
  run Q-q4d-$p    -- "${PPL[@]}" --ids-file $D/prompts/eval_$p.i32 --pcie auto --dense-requant q4_k
  run Q-bias-$p   -- "${PPL[@]}" --ids-file $D/prompts/eval_$p.i32 --pcie auto --dense-requant q4_k --route-bias 0.02
done
echo "=== maya-improve2 done $(date +%T) ===" | tee -a $S
