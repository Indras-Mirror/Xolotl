#!/usr/bin/env bash
# Maya-S-v2 (on the NTFS NVMe), our LoRA ablation, ctx 300K: the improvements one at a time (s26).
#  A base       : today's --300k serving config (chunk 1024, MMQ forced off under --lora-exps, index-order seed)
#  B mmq        : + the LoRA down delta in the MMQ chunk path (native_expert_lora_down) -> --chunk-mmq kept
#  C census     : + --census (bootstrapped from the 7 calibration saliency dumps; rank = uses / blob bytes)
#  D lru        : + --vram-lru --vram-pin 0.6 (helios: pin the census core, LRU the rest)
#  E chunk4k    : D at --prefill-chunk 4096
#  F chunk6k    : D at --prefill-chunk 6144
# Prompt: chat_code6k.i32 (12736 tokens, a chat turn with thinking) -> 256 tokens; the decoded text is printed so
# coherence can be read.  VRAM peak sampled every 0.2 s.  Waits for the ds4-gpu lock.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M="/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
TOK=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer
D=bench/glm-2026-10-09
CEN0=$HOME/.quetza-data/strata-glm/maya-s-v2.census
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S=$D/maya-improve.summary
: > $S
BASE=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 307200 --chunk-mmq
      --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt
      --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip --pcie 1.0
      --stop 154820,154827,154829 --lora "$A" --lora-exps --temp 0
      --ids-file $D/prompts/chat_code6k.i32 -n 256)
text() {
  $HOME/AI/Strata/.venv/bin/python - "$1" "$TOK" <<'EOF' 2>&1
import sys, re
sys.path.insert(0, "tools/glm/serve"); sys.path.insert(0, "/home/mal/AI/Strata/tools")
import glm_tok
ids = [int(l) for l in open(sys.argv[1], errors="replace") if re.fullmatch(r"\d+\n?", l)]
t = glm_tok.load(sys.argv[2]).decode(ids)
print(f"  text [{len(ids)} ids]: {t[:600]!r}")
EOF
}
run() {   # name margin chunk env... -- extra args
  local name=$1 margin=$2 chunk=$3; shift 3
  local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ $# -gt 0 ] && shift
  echo "--- $name (margin $margin, chunk $chunk ${envs[*]} $*) $(date +%T) ---" | tee -a $S
  : > $D/mi-$name.vram
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $D/mi-$name.vram; sleep 0.2; done ) &
  local smi=$!
  env "${envs[@]}" bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${BASE[@]}" --vram-margin $margin \
      --prefill-chunk $chunk "$@" > $D/mi-$name.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  grep -hE "census:|slots auto|^tier:|^prefill:|^decode :|experts/token|tier moves|out of memory|error|no kernel" $D/mi-$name.log | tee -a $S
  echo "  exit $rc, VRAM peak $(sort -n $D/mi-$name.vram | tail -1) MiB" | tee -a $S
  text $D/mi-$name.log | tee -a $S
}
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 10; done
run A-base    5.5 1024 GLM_LORA_MMQ_OFF=1 --
run B-mmq     5.5 1024 X=1 --
cp $CEN0 $D/mi-C.census; run C-census 5.5 1024 X=1 -- --census $D/mi-C.census
cp $CEN0 $D/mi-D.census; run D-lru    5.5 1024 X=1 -- --census $D/mi-D.census --vram-lru --vram-pin 0.6
cp $CEN0 $D/mi-E.census; run E-chunk4k 9   4096 X=1 -- --census $D/mi-E.census --vram-lru --vram-pin 0.6
cp $CEN0 $D/mi-F.census; run F-chunk6k 12  6144 X=1 -- --census $D/mi-F.census --vram-lru --vram-pin 0.6
echo "=== maya-improve done $(date +%T) ===" | tee -a $S
