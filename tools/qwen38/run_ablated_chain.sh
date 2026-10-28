#!/usr/bin/env bash
# Ablated Qwen3.8-Flash-Next XQ chain, after q8_ablate.py has edited the Q8 model in place.
#
#   1. iq_pack on the EDITED model   -> $PACK
#   2. llama-imatrix on the EDITED model -> $IMATRIX
#   3. strata dump the EDITED model  -> $DUMPS
#   4. screen.py all layers          -> $STATS + $IMP
#   5. v41_alloc.py --layer-choice   -> $ALLOC
#
# Every stage is idempotent: a stage whose output exists and is complete is skipped, so a
# power cut or a kill costs only the stage in flight. Nothing here touches pristine data:
# the pre-edit model survives as a verified range backup in $ROLLBACK.
set -uo pipefail

XOLOTL=/home/mal/AI/Xolotl
T=/home/mal/AI/Xolotl/tools/qwen38
NVME=/media/mal/NVME1TB/qwen38
MODEL=$NVME/qwen38-full-q8_0.gguf
EDIT_REPORT=/home/mal/.quetza-data/qwen38-q8-edit.json
ROLLBACK=/home/mal/qwen38-pristine/edit-rollback
PACK=/home/mal/qwen38-pack-abl
IMATRIX=$NVME/imatrix-abl.gguf
IMATRIX_LOG=$NVME/logs/imatrix-abl.log
DUMPS=$NVME/dumps-abl
STATS=$NVME/stats-abl
IMP=$NVME/imp-abl
ALLOC=$NVME/alloc-abl.json
STATE=/home/mal/.quetza-data/qwen38-abl-chain
LOG=$STATE/chain.log
CORPUS=$NVME/corpus-v4/corpus.txt
IDS=$NVME/corpus-v4/ids-49153.txt
PROFILE=/home/mal/qwen38-profile.bin
IMATRIX_BIN=/home/mal/AI/llama.cpp-qwen4exp/build/bin/llama-imatrix
STRATA=$XOLOTL/build-xolotl/strata

mkdir -p "$STATE" "$(dirname "$IMATRIX_LOG")" "$(dirname "$IMATRIX")" || exit 2
exec >>"$LOG" 2>&1

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { say "FAILED: $*"; printf 'FAILED %s\n' "$*" > "$STATE/status"; exit 90; }
ok() { printf '%s\n' "$1" > "$STATE/status"; }

say "=== ablated chain start ==="

# ---------------------------------------------------------------- 0. preconditions
[ -f "$MODEL" ] || die "no edited model"
[ -f "$EDIT_REPORT" ] || die "no edit report"
python3 - "$EDIT_REPORT" <<'PY' || die "edit report is not a completed 145-target edit"
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("state") != "edited" or d.get("layers") != "all" or len(d.get("targets", [])) != 145:
    print("report:", d.get("state"), d.get("layers"), len(d.get("targets", [])))
    raise SystemExit(1)
print("edit: 145 targets, alpha range %.5f..%.5f" % (
    min(t["alpha_eff"] for t in d["targets"]), max(t["alpha_eff"] for t in d["targets"])))
PY
[ -d "$ROLLBACK" ] || die "no rollback directory"
[ -f "$CORPUS" ] || die "no corpus"
[ -s "$IDS" ] || die "no ids file"
[ -s "$PROFILE" ] || die "no expert profile"
[ -x "$IMATRIX_BIN" ] || die "no llama-imatrix"
[ -x "$STRATA" ] || die "no strata"

# ---------------------------------------------------------------- 1. pack
if [ -f "$PACK/index.txt" ] && [ -f "$STATE/stage1.done" ]; then
    say "STEP1 skip (pack present)"
else
    say "STEP1 iq_pack on the edited model"
    rm -rf "$PACK"
    ionice -c3 nice -n 19 python3 "$XOLOTL/tools/iq_pack.py" \
        --gguf "$MODEL" --out "$PACK" --compat-bf16 >"$STATE/step1.log" 2>&1 \
        || { tail -20 "$STATE/step1.log"; die "iq_pack failed"; }
    [ -f "$PACK/native_experts.txt" ] || die "pack has no native_experts.txt"
    touch "$STATE/stage1.done"
    ok "STEP1 pack done"
    say "STEP1 done: $(du -sh "$PACK" | cut -f1)"
fi

# ---------------------------------------------------------------- 2. imatrix
if [ -f "$STATE/stage2.done" ] && [ -f "$IMATRIX" ]; then
    say "STEP2 skip (imatrix present)"
else
    say "STEP2 llama-imatrix on the edited model"
    [ -e "$IMATRIX" ] && mv "$IMATRIX" "$IMATRIX.partial.$(date +%s)"
    IMATRIX_BIN="$IMATRIX_BIN" SRC="$MODEL" CORPUS="$CORPUS" OUT="$IMATRIX" LOG="$IMATRIX_LOG" \
        bash "$T/run_imatrix.sh" >>"$STATE/step2.log" 2>&1 \
        || { tail -20 "$STATE/step2.log"; die "imatrix failed"; }
    # Completion proof from the file itself: save_imatrix() writes chunk_count at every
    # save, so a truncated run leaves a smaller number. A log line can be stale; this cannot.
    python3 - "$IMATRIX" <<'PY' || die "imatrix did not finish all 48 chunks"
import sys
sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
cc = r.fields.get("imatrix.chunk_count")
cs = r.fields.get("imatrix.chunk_size")
if cc is None:
    print("no imatrix.chunk_count in the file")
    raise SystemExit(1)
n = int(cc.contents())
print("chunk_count", n, "chunk_size", int(cs.contents()) if cs else "(absent)")
raise SystemExit(0 if n == 48 else 1)
PY
    touch "$STATE/stage2.done"
    ok "STEP2 imatrix done"
    say "STEP2 done: $(stat -c%s "$IMATRIX") bytes"
fi

# ---------------------------------------------------------------- 3. dumps
if [ -f "$STATE/stage3.done" ] && [ "$(ls "$DUMPS"/*.bin 2>/dev/null | wc -l)" -eq 1152 ]; then
    say "STEP3 skip (dumps present)"
else
    say "STEP3 free space, then dump the edited model"
    # dumps-abl needs ~23G and NVME has ~16G. The pristine dumps are the control arm's only
    # remaining derived artifact and are not regenerable once the model is edited, so they are
    # MOVED to the USB disk (verified by file count) rather than deleted.
    USB=/media/mal/Team CX2/qwen38-control/dumps-pristine
    if [ -d "$NVME/dumps" ]; then
        if [ -d "$USB" ]; then
            # A previous attempt copied but did not get to delete. Only remove the NVMe copy
            # after re-proving the USB copy is complete - otherwise an interrupted move would
            # be treated as done and the control dumps would exist only as a partial copy.
            a=$(ls "$NVME/dumps"/*.bin 2>/dev/null | wc -l)
            b=$(ls "$USB"/*.bin 2>/dev/null | wc -l)
            [ "$a" -eq 1152 ] && [ "$b" -eq "$a" ] || die "interrupted dump move: NVMe $a vs USB $b files"
            [ "$(du -sb "$NVME/dumps" | cut -f1)" -eq "$(du -sb "$USB" | cut -f1)" ] \
                || die "interrupted dump move: sizes differ"
            say "  resuming an interrupted move: USB copy verified complete, removing NVMe copy"
            rm -rf "$NVME/dumps"
        else
            need=$(du -sb "$NVME/dumps" | cut -f1)
            free=$(df --output=avail -B1 "/media/mal/Team CX2" | tail -1 | tr -d ' ')
            [ "$free" -gt $((need + 2*1024*1024*1024)) ] || die "no room to move pristine dumps ($need bytes)"
            say "  moving pristine dumps ($(du -sh "$NVME/dumps" | cut -f1)) to Team CX2"
            cp -a "$NVME/dumps" "$USB" || die "copy of pristine dumps failed"
            a=$(ls "$NVME/dumps"/*.bin | wc -l)
            b=$(ls "$USB"/*.bin | wc -l)
            [ "$a" -eq "$b" ] || die "pristine dump copy has $b of $a files"
            [ "$(du -sb "$NVME/dumps" | cut -f1)" -eq "$(du -sb "$USB" | cut -f1)" ] \
                || die "pristine dump copy differs in size"
            rm -rf "$NVME/dumps"
            say "  pristine dumps moved and verified ($b files)"
        fi
    fi
    # hard gate: the dump needs ~23G and must not start on a volume that cannot hold it
    free_g=$(df --output=avail -BG "$NVME" | tail -1 | tr -dc '0-9')
    [ "${free_g:-0}" -ge 25 ] || die "dump needs ~25G on $NVME, only ${free_g}G free"
    say "  dump space ok: ${free_g}G free"
    mkdir -p "$DUMPS"
    cd "$XOLOTL" || die "no repo"
    STRATA_DUMP_MOE_ALL="$DUMPS" "$STRATA" \
        --native "$MODEL" --pack "$PACK" \
        --ple-gguf "$MODEL" --ple-io direct \
        --mmap-experts --spec 2 \
        --expert-profile "$PROFILE" --expert-cache 1536 \
        --prefill 2048 --max-context 49184 --max-new 1 \
        --tokens-file "$IDS" >"$STATE/step3.log" 2>&1 \
        || { grep -m5 -iE "error|abort|assert|out of memory|failed|cannot|refus" "$STATE/step3.log"; tail -20 "$STATE/step3.log"; die "dump failed"; }
    n=$(ls "$DUMPS"/*.bin 2>/dev/null | wc -l)
    [ "$n" -eq 1152 ] || die "dump produced $n files, expected 1152"
    touch "$STATE/stage3.done"
    ok "STEP3 dumps done"
    say "STEP3 done: 1152 files, $(du -sh "$DUMPS" | cut -f1)"
fi

# ---------------------------------------------------------------- 4. screen
mkdir -p "$STATS" "$IMP"
done_layers=0
for L in $(seq 0 47); do
    f=$(printf "%s/L%02d.json" "$STATS" "$L")
    if [ -f "$f" ] && [ -f "$(printf "%s/L%02d.npz" "$IMP" "$L")" ]; then
        done_layers=$((done_layers+1))
    fi
done
if [ "$done_layers" -eq 48 ] && [ -f "$STATE/stage4.done" ]; then
    say "STEP4 skip (48 screens present)"
else
    say "STEP4 screen layers ($done_layers/48 already present)"
    for L in $(seq 0 47); do
        f=$(printf "%s/L%02d.json" "$STATS" "$L")
        p=$(printf "%s/L%02d.npz" "$IMP" "$L")
        if [ -f "$f" ] && [ -f "$p" ]; then continue; fi
        say "  screen L$L"
        # --tiers must cover EVERY rung the allocator can pick for gate/up, or
        # items_for raises "missing measurement". q8_0 is in that ladder.
        ionice -c3 nice -n 10 python3 "$T/screen.py" \
            --model-dir "$MODEL" --dump "$DUMPS" --layer "$L" \
            --tiers ptq1,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,q8_0 \
            --edit-report "$EDIT_REPORT" --dn-tiers iq4_nl,q8_0 \
            --out "$f" --importance-out "$p" >"$(printf "%s/screen-L%02d.log" "$STATE" "$L")" 2>&1 \
            || { tail -15 "$(printf "%s/screen-L%02d.log" "$STATE" "$L")"; die "screen L$L failed"; }
    done
    touch "$STATE/stage4.done"
    ok "STEP4 screen done"
    say "STEP4 done: $(ls "$STATS"/*.json | wc -l) layers"
fi

# ---------------------------------------------------------------- 5. allocate
if [ -f "$ALLOC" ] && [ -f "$STATE/stage5.done" ]; then
    say "STEP5 skip (allocation present)"
else
    say "STEP5 allocate"
    # The cheapest legal build, from the measured expert count. GU is gate+up at 1.75 bpw;
    # edited down cannot go below IQ4_NL. BUDGET_GIB overrides; the default adds 10% to the
    # floor so the DP has slack to buy error down where it matters.
    FLOOR=$(python3 - "$STATS" <<'PY'
import glob, json, sys
GU, DN = 2 * 2560 * 640, 2560 * 640
GU_BPW, DN_BPW = 1.75, 4.5          # PTQ1_0 gate/up, IQ4_NL edited down
n = sum(len(json.load(open(p))["experts"]) for p in glob.glob(sys.argv[1] + "/L*.json"))
print(int(round(n * (GU * GU_BPW + DN * DN_BPW) / 8)))
PY
)
    say "  cheapest legal expert bytes: $(python3 -c "print(f'{$FLOOR/2**30:.2f} GiB')")"
    W=$STATE/allocwork; mkdir -p "$W"; ln -sfn "$STATS" "$W/stats"
    # Sweep the budget and report the measured knee before committing to one. The
    # allocator's objective is held-out expert-output MSE, which is a proxy - not
    # top-1 agreement - so the point of the sweep is to show where more bytes stop
    # buying error, not to pick a number in advance.
    say "  budget sweep (expert GiB -> proxy error):"
    : >"$STATE/sweep.tsv"
    for M in 1.00 1.10 1.20 1.35 1.50 1.75; do
        B=$(python3 -c "print(f'{$FLOOR * $M / 2**30:.2f}')")
        if python3 "$XOLOTL/tools/v41/v41_alloc.py" --work "$W" --arch qwen4exp \
            --per-proj --edit-downs --layer-choice --budget-gib "$B" \
            --unseen-floor iq2_xxs --out "$STATE/sweep-$B.json" >"$STATE/sweep-$B.log" 2>&1
        then
            E=$(python3 -c "
import json; d=json.load(open('$STATE/sweep-$B.json'))
print(f\"{d['used_bytes']/2**30:.3f}\t{d['proxy_err_mixed']:.6f}\")")
            printf '%s\t%s\n' "$B" "$E" >>"$STATE/sweep.tsv"
            say "    budget ${B} GiB -> $(echo "$E" | tr '\t' ' ') (used, err)"
        else
            say "    budget ${B} GiB -> infeasible"
        fi
    done
    # Default 1.20x the legal floor: experts ~45 GiB, total ~74 GiB with the 26.8 GiB
    # PLE table and ~2.5 GiB dense. That still fits the 90 GB box with mmap experts;
    # 3x would be ~142 GiB and would not run here at all. BUDGET_GIB overrides.
    # The real quality test is top-1 agreement on the built file, not this proxy.
    BUDGET=${BUDGET_GIB:-$(python3 -c "print(f'{$FLOOR * 1.20 / 2**30:.2f}')")}
    say "  budget: ${BUDGET} GiB"
    python3 "$XOLOTL/tools/v41/v41_alloc.py" --work "$W" --arch qwen4exp \
        --per-proj --edit-downs --layer-choice --budget-gib "$BUDGET" \
        --unseen-floor iq2_xxs --out "$ALLOC" >"$STATE/step5.log" 2>&1 \
        || { tail -20 "$STATE/step5.log"; die "alloc failed"; }
    touch "$STATE/stage5.done"
    ok "STEP5 allocation done"
    say "STEP5 done: $(grep -c '^L' "$STATE/step5.log") layer lines"
fi

# ---------------------------------------------------------------- 6. tensor types + quantize
TT=$NVME/tensor_types-abl.txt
QOUT="/media/mal/Team CX2/qwen38-abl/Qwen3.8-Flash-Next-XQ-abl.gguf"
if [ -f "$STATE/stage6.done" ] && [ -f "$QOUT" ]; then
    say "STEP6 skip (quantized model present)"
else
    say "STEP6 tensor types + llama-quantize"
    python3 "$T/emit_tensor_types.py" "$ALLOC" --per-proj --dense-type "${DENSE_TYPE:-Q8_0}" -o "$TT" \
        >"$STATE/step6-emit.log" 2>&1 || { tail -20 "$STATE/step6-emit.log"; die "emit failed"; }
    # the collapse must be a no-op: every expert in a layer carries the same tier
    python3 - "$ALLOC" <<'PY' || die "allocation still needs a lossy collapse"
import json, sys
d = json.load(open(sys.argv[1]))["map"]
bad = [(L, k) for L in d for k in ("gu", "dn")
       if len({e[k] for e in d[L].values() if k in e}) > 1]
print(f"{len(d)} layers, {len(bad)} groups with mixed tiers")
raise SystemExit(1 if bad else 0)
PY
    mkdir -p "$(dirname "$QOUT")"
    PRED=$(python3 "$T/predict_size.py" --types "$TT" --gguf "$MODEL" --ftype Q8_0 --json)
    say "  predicted: $(echo "$PRED" | python3 -c 'import json,sys; print(f"{json.load(sys.stdin)[\"gib\"]:.2f} GiB")')"
    python3 "$T/predict_size.py" --types "$TT" --gguf "$MODEL" --ftype Q8_0 \
        --ceiling-gib "${CEILING_GIB:-90}" >"$STATE/step6-predict.log" 2>&1 \
        || { cat "$STATE/step6-predict.log"; die "predicted size exceeds the ceiling"; }
    SRC="$MODEL" IMATRIX="$IMATRIX" GATE_GIB="${CEILING_GIB:-90}" MARGIN_GB=6 \
        bash "$T/run_quant.sh" "$TT" "$QOUT" >"$STATE/step6-quant.log" 2>&1 \
        || { tail -25 "$STATE/step6-quant.log"; die "llama-quantize failed"; }
    [ -s "$QOUT" ] || die "quantize produced no output"
    touch "$STATE/stage6.done"
    ok "STEP6 quantize done"
    say "STEP6 done: $(du -h "$QOUT" | cut -f1)"
fi

# ---------------------------------------------------------------- 7. ternary payloads
if [ -f "$STATE/stage7.done" ] && [ -f "$QOUT.ptq1" ]; then
    say "STEP7 skip (ternary output present)"
else
    say "STEP7 replace PTQ1_0 chosen gate/up payloads"
    python3 "$T/apply_ternary.py" --input "$QOUT" --source "$MODEL" --alloc "$ALLOC" \
        --edit-report "$EDIT_REPORT" --importance "$IMP" \
        --out "$STATE/apply-ternary.json" >"$STATE/step7.log" 2>&1 \
        || { tail -25 "$STATE/step7.log"; die "apply_ternary failed"; }
    touch "$STATE/stage7.done"
    ok "STEP7 ternary done"
    say "STEP7 done: $(du -h "$QOUT.ptq1" | cut -f1)"
fi

say "=== ablated chain complete: $QOUT.ptq1 ==="
ok "CHAIN_COMPLETE"
