#!/usr/bin/env bash
# s26c: the elastic cache with its two leaks fixed - the per-token input upload sized by the largest chunk (16-24 MB
# of mask per decode token at chunk 4096/6144) and the dense chunk compute buffer kept after the prompt
# (GlmDense::release_big) - then 512K, then an nsys profile of the decode.  Maya-S-v2 + LoRA, census, pcie auto, q4 dense.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M="/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf"
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
TOK=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer
D=bench/glm-2026-10-09
CEN0=$HOME/.quetza-data/strata-glm/maya-s-v2.census
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S=$D/maya-improve3.summary
: > $S
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --chunk-mmq
        --prune $D/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt
        --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip
        --stop 154820,154827,154829 --lora "$A" --lora-exps --pcie auto)
T=(--temp 0 --ids-file $D/prompts/chat_code6k.i32 -n 256)
text() {
  $HOME/AI/Strata/.venv/bin/python - "$1" "$TOK" <<'EOF' 2>&1
import sys, re
sys.path.insert(0, "tools/glm/serve"); sys.path.insert(0, "/home/mal/AI/Strata/tools")
import glm_tok
ids = [int(l) for l in open(sys.argv[1], errors="replace") if re.fullmatch(r"\d+\n?", l)]
print(f"  text [{len(ids)} ids]: {glm_tok.load(sys.argv[2]).decode(ids)[:400]!r}")
EOF
}
run() {   # name [env...] -- args
  local name=$1; shift
  local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ $# -gt 0 ] && shift
  echo "--- $name (${envs[*]} $*) $(date +%T) ---" | tee -a $S
  : > $D/mi3-$name.vram
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $D/mi3-$name.vram; sleep 0.2; done ) &
  local smi=$!
  cp $CEN0 $D/mi3-$name.census
  env X=1 "${envs[@]}" bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --census $D/mi3-$name.census "$@" > $D/mi3-$name.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  grep -hE "ppl:|grow_cache|slots auto|^tier:|^prefill:|^decode :|experts/token|tier ms/token|decode ms/token|out of memory|error" $D/mi3-$name.log | tee -a $S
  echo "  exit $rc, VRAM peak $(sort -n $D/mi3-$name.vram | tail -1) MiB" | tee -a $S
  grep -q "ppl:" $D/mi3-$name.log || text $D/mi3-$name.log | tee -a $S
}
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 10; done
# gate 2: q4_k dense = +7.5% decode but +5.2% chat ppl (5.34 -> 5.61) -> q5_k (keep), q6_k (as shipped) arms
run M-el4k-q5    -- "${T[@]}" --dense-requant q5_k --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5
run M-el4k-q6    -- "${T[@]}" --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5
run N-el6k-q5    -- "${T[@]}" --dense-requant q5_k --ctx 307200 --prefill-chunk 6144 --vram-margin 10 --vram-grow 1.5
run O-el4k-512k  -- "${T[@]}" --dense-requant q5_k --ctx 524288 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5
PPL=(--ctx 4096 --prefill-chunk 1024 --vram-margin 5.5 -n 1 --ppl)
for p in chat code; do run Q-q5d-$p -- "${PPL[@]}" --ids-file $D/prompts/eval_$p.i32 --dense-requant q5_k; done
# the decode profile: nsys captures only the decode loop (GLM_PROFILE_DECODE -> cudaProfilerStart/Stop)
echo "--- P-nsys $(date +%T) ---" | tee -a $S
cp $CEN0 $D/mi3-P.census
GLM_PROFILE_DECODE=1 flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=84G -p MemorySwapMax=0 \
  /usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --sample=none \
  --cpuctxsw=none -o $D/mi3-decode -f true \
  "$B" -m "$M" "${COMMON[@]}" "${T[@]}" --dense-requant q5_k --census $D/mi3-P.census --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 \
  > $D/mi3-P.log 2>&1
echo "  nsys exit $?" | tee -a $S
/usr/local/cuda/bin/nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f csv -o $D/mi3-decode $D/mi3-decode.nsys-rep > /dev/null 2>&1
ls $D/mi3-decode*.csv 2>/dev/null | tee -a $S
echo "=== maya-improve3 done $(date +%T) ===" | tee -a $S
