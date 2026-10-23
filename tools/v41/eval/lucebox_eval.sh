#!/bin/bash
# tools/v41/eval/lucebox_eval.sh - Lucebox's public 1017 eval (8 x 1024 teacher-forced) on a ds41 GGUF.
#   lucebox_eval.sh MODEL.gguf RANK.bin OUTDIR [SEQS]      e.g. SEQS="0 1" for a plumbing check
# Eval set: ~/v41-work/eval/lucebox-1017 (heldout_8x1024.npy, teacher_8x1024.npz from Lucebox's HF repo, eval/).
# One engine run per sequence (each needs a fresh KV); logits (~0.5 GB a sequence) are scored and deleted.
# Refuses while the V4 quant pass holds the card unless FORCE=1.
set -u
M=$1; R=$2; OUT=$3; SEQS=${4:-0 1 2 3 4 5 6 7}
EV=${LUCEBOX_EVAL:-/home/mal/v41-work/eval/lucebox-1017}
X=/home/mal/AI/Xolotl; cd "$X" || exit 1; mkdir -p "$OUT"
P=$(cat /home/mal/v41-work/quant-1017-v4/pass.pid 2>/dev/null)
if kill -0 "$P" 2>/dev/null && [ "${FORCE:-0}" != 1 ]; then echo "quant pass $P is running" >&2; exit 1; fi
ENGRAM=${ENGRAM:-"/media/Crucial1TB/V41-1017-Engram:/media/mal/SSD NVME/AI/V41-1017-Engram"}
TMP=${LOGIT_TMP:-/media/mal/NVME1TB/lucebox-logits}; mkdir -p "$TMP"
for s in $SEQS; do
    python3 -c "import numpy as np; np.load('$EV/heldout_8x1024.npy')[$s].astype(np.int32).tofile('$OUT/seq$s.i32')"
    bash tools/ds4/memguard.sh 86 72 -- build-xolotl/ds41_generate -m "$M" --ids-file "$OUT/seq$s.i32" -n 1 --ppl \
        --dump-logits "$TMP/seq$s.f32" --backend cuda --experts gpu --engram "$ENGRAM" --rank "$R" --ctx 4096 \
        --arena-adapt --kv-fp4 --arena-gib 64 --prompt-batch 12288 --temp 0 > "$OUT/seq$s.out" 2> "$OUT/seq$s.log"
    echo "seq $s rc=$? $(grep -E '^PPL' "$OUT/seq$s.out" "$OUT/seq$s.log" | head -1)"
    nice -n 19 python3 tools/v41/eval/lucebox_score.py "$EV" "$s" "$TMP/seq$s.f32" --acc "$OUT/acc.json" && rm -f "$TMP/seq$s.f32"
done
python3 tools/v41/eval/lucebox_score.py --report "$OUT/acc.json" | tee "$OUT/result.json"
