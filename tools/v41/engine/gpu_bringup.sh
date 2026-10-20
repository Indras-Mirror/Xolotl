#!/usr/bin/env bash
# gpu_bringup.sh - first GPU runs of the V4.1 engine on the finished GSQ file (needs the GPU: run only when the chain
# is DONE and Mal's GPU rule allows).  Each step logs to $OUT; stops at the first failure.
#   1. CUDA activation rounding vs torch (bit-exact)          test_ds41_ops --cuda
#   2. expert tier on real mixed-tier experts, CUDA paths      test_ds41_moe_gpu --gpu (layers 0, 20, 39)
#   3. end to end: ppl + prompt/decode speed on held-out code  ds41_generate --ppl (2048 tokens, chunk 512)
set -u
cd $HOME/AI/Xolotl
GGUF="${GGUF:-/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf}"
OUT="${OUT:-$HOME/AI/Strata-Backup/v41-dev/gpu-$(date +%m%d-%H%M)}"
ENGRAM="/media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
mkdir -p "$OUT"
G=build-ds4-gpu
run() { echo "== $*" | tee -a "$OUT/steps.log"; "$@" >> "$OUT/steps.log" 2>&1; local rc=$?; echo "rc=$rc" | tee -a "$OUT/steps.log"; return $rc; }
run $G/test_ds41_ops $HOME/AI/Strata-Backup/v41-dev/fq --cuda || exit 1
run bash tools/ds4/memguard.sh 40 30 -- $G/test_ds41_moe_gpu "$GGUF" --layers 0,20,39 --tokens 6 --gpu || exit 1
python3 -c "import numpy as np; np.fromfile('/home/mal/AI/Strata-Backup/v41-dev/eval/seq20_code.i32', np.int32)[:2048].tofile('$OUT/code2k.i32')"
run bash tools/ds4/memguard.sh 84 70 -- $G/ds41_generate -m "$GGUF" --ids-file "$OUT/code2k.i32" -n 32 --ppl \
    --backend cuda --experts gpu --engram "$ENGRAM" --prefill-chunk 512 --arena-gib 60 --vram-lru \
    ${RANK:+--rank "$RANK"} || exit 1
grep -E "PPL|prompt:|decode:|tier:|VRAM:" "$OUT/steps.log"
