#!/usr/bin/env bash
# GLM-5.3-Flash Maya-S-v2 (peasantsmith, IQ2_XXS gate/up + IQ2_S/IQ3_XXS down, Q6_K dense/attn, 3 shards + blk.45
# NextN) on our engine, against the RCO s11 numbers.  Needs the s25e wiring: IQ2_S/IQ3_XXS down kernels
# (STRATA_D_FMTS_FORK), the split-aware tier geometry, block_count - nextn_predict_layers.
# Waits for the download to finish and for the ds4-gpu lock.
#  1. maya-base-chat / maya-base-code: s11's ppl runs verbatim (no prune, chunk 1024, margin 5.5) -> RCO 5.547 / 3.668
#  2. maya-abl300k: the strata-glm-unc-ablated-300k.json args (our LoRA, --lora-exps, prune, skips, renorm) -> decode
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
MD=/media/Crucial1TB/models/GLM-5.3-Flash-Maya-S-v2/Maya-S-v2-IQ2_XXS
M=$MD/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
TOK=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer
D=bench/glm-2026-10-09
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S="$D/maya.summary"
: > "$S"
echo "waiting for the download $(date +%T)" | tee -a "$S"
while pgrep -f "hf download peasantsmith/GLM-5.3-Flash-Maya" >/dev/null || \
      [ ! -f "$M" ] || [ ! -f "$MD/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf" ]; do sleep 30; done
ls -l "$MD" | tee -a "$S"
while pgrep -f "gate-ctx300k.s[h]" >/dev/null; do sleep 30; done   # do not interleave with the 300K gate
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 30; done

text() {   # the generated ids (one per line in the log) -> text
  $HOME/AI/Strata/.venv/bin/python - "$1" "$TOK" <<'EOF' 2>&1 | head -c 1500
import sys, re
sys.path.insert(0, "tools/glm/serve"); sys.path.insert(0, "/home/mal/AI/Strata/tools")
import glm_tok
ids = [int(l) for l in open(sys.argv[1], errors="replace") if re.fullmatch(r"\d+\n?", l)]
print(f"[{len(ids)} ids] " + glm_tok.load(sys.argv[2]).decode(ids))
EOF
}
run() {
  local name=$1; shift
  echo "### $name $(date +%T)" | tee -a "$S"
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1
  echo "exit $? $(date +%T)" | tee -a "$S"
  grep -hE "slots auto|tier:|ppl:|prefill:|decode :|experts/token|out of memory|error|fail|no kernel|native" "$D/$name.log" | tee -a "$S"
}
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
E=(-n 1 --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq --ppl)
run maya-base-chat --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}"
run maya-base-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}"
run maya-abl300k --backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 307200 --prefill-chunk 1024 \
    --chunk-mmq --vram-margin 5.5 --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt \
    --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip --pcie 1.0 \
    --stop 154820,154827,154829 --lora "$A" --lora-exps --ids-file $D/prompts/eval_chat.i32 -n 128
text "$D/maya-abl300k.log" | tee -a "$S"
echo | tee -a "$S"
echo "=== maya done $(date +%T) ===" | tee -a "$S"
