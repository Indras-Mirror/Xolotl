#!/usr/bin/env bash
# run_v2_night.sh - the QFN V2 overnight chain, unattended and resumable.
#
#   Swift 1.5 Q8_0 shards -> our abliteration -> llama-quantize with the published GSQ-RCO
#   per-tensor allocation + their imatrix -> top-1/KL gate against the parent -> speed test.
#
# Every stage writes $STATE/<n>-<name>.done and is skipped if that marker exists, so a kill or a
# power cut costs only the stage in flight.  A failure writes $STATE/status and exits non-zero; it
# never continues into a later stage on a broken earlier one.
#
# WHY "--keep-split": the input is 5 shards and per_layer_token_embd.weight is the ONLY tensor in
# shard 2.  Keeping the split keeps the PLE table alone in shard 2, which is exactly what the
# engine wants (--native <shard1> --ple-gguf <shard2>).
#
# WHY the tensor-type file is generated and not used raw: llama-quantize's parser rejects '#' lines
# (tools/quantize/quantize.cpp:parse_tensor_type_file) and matches overrides with std::regex_search
# (src/llama-quant.cpp:693), so the produced file has no comments and escaped dots.
#
#   bash tools/qwen38/run_v2_night.sh            # full run
#   bash tools/qwen38/run_v2_night.sh --from-pack  # resume at the quantise stage
set -uo pipefail

XOLOTL=/home/mal/AI/Xolotl
T=$XOLOTL/tools/qwen38
STATE=/home/mal/.quetza-data/qfn-v2-night
LOG=$STATE/chain.log
SHARDS=/media/mal/NVME1TB/Models/Swift1.5-Q8_0/Q8_0
ADAPTER=/media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf
REF=/media/mal/NVME1TB/qwen38/gsq-ref
ALLOC=$REF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS.rco-allocation.txt
IMATRIX=$REF/imatrix-swiftfn-v1mix.gguf
OUTBASE=/media/mal/NVME1TB/qwen38/swift15-abl-v2
OUT1=$OUTBASE-00001-of-00005.gguf
OUT2=$OUTBASE-00002-of-00005.gguf
TYPES=$STATE/tensor-types.txt
EDIT_REPORT=$STATE/q8-edit.json
ROLLBACK="/media/mal/Team CX2/qfn-v2-rollback"
PACK_PARENT=/media/mal/NVME1TB/qwen38/pack-swift-abl
PACK_QUANT=/media/mal/NVME1TB/qwen38/pack-swift-abl-v2
IDS=/media/mal/NVME1TB/qwen38/corpus-v4/ids-512.txt
CORPUS_IDS=/media/mal/NVME1TB/qwen38/corpus-v4/ids-49153.txt
QUANTIZE=$HOME/AI/llama.cpp-qwen4exp/build/bin/llama-quantize
STRATA=$XOLOTL/build-xolotl/strata
# --spec 2 needs a non-empty residency profile (48 layers x 512 experts); without it strata exits 2
# "--spec needs the device residency table".  NB: strata has NO "generate" subcommand.
PROFILE=/home/mal/qwen38-profile.bin
MAX_GB=70
THREADS=16                       # peer's measured figure: ~705 MB peak per worker

mkdir -p "$STATE" || exit 2
exec >>"$LOG" 2>&1

say()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
done_() { touch "$STATE/$1.done"; say "DONE $1"; }
have() { [ -f "$STATE/$1.done" ]; }
die()  { say "FAILED: $*"; printf 'FAILED %s\n' "$*" > "$STATE/status"; exit 90; }
ok()   { printf '%s\n' "$1" > "$STATE/status"; }

# ---- interpreter: the systemd PATH has no anaconda and /usr/bin/python3 has no numpy.
# run_ablated_chain.sh already lost a night to exactly this (a missing module reported as
# "infeasible").  Pin it and prove it once.
PY=""
for c in /home/mal/anaconda3/bin/python3 "$(command -v python3 2>/dev/null)" /usr/bin/python3; do
    [ -n "${c:-}" ] && [ -x "$c" ] || continue
    "$c" -c 'import numpy' >/dev/null 2>&1 && { PY=$c; break; }
done
[ -n "$PY" ] || die "no python3 with numpy"
export PATH="$(dirname "$PY"):$PATH"
say "=== QFN V2 night chain starting (python $PY) ==="

free_gb() { df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'; }

# ---------------------------------------------------------------- 0. wait + verify the download
if ! have 0-verify; then
    TARGET=54400261312
    S2=$SHARDS/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00002-of-00005.gguf
    say "0 verify: waiting for shard 2 ($(stat -c %s "$S2" 2>/dev/null || echo 0)/$TARGET)"
    while systemctl --user is-active --quiet swift-dl2.service; do sleep 30; done
    got=$(stat -c %s "$S2" 2>/dev/null || echo 0)
    if [ "$got" -ne "$TARGET" ]; then
        # curl -C - is idempotent; resume rather than fail the night
        say "0 verify: shard 2 is $got, expected $TARGET - resuming"
        systemd-run --user --unit=swift-dl2-resume --collect /usr/bin/curl -4 -L --fail \
            --retry 20 --retry-delay 5 --retry-all-errors -C - -o "$S2" \
            "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/resolve/main/Q8_0/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00002-of-00005.gguf" \
            || die "could not resume shard 2"
        while systemctl --user is-active --quiet swift-dl2-resume.service; do sleep 30; done
        got=$(stat -c %s "$S2" 2>/dev/null || echo 0)
        [ "$got" -eq "$TARGET" ] || die "shard 2 is $got after resume, expected $TARGET"
    fi
    say "0 verify: hashing 5 shards against the published SHA-256"
    bash "$T/verify_q8_shards.sh" || die "shard verification failed"
    done_ 0-verify
fi

# ---------------------------------------------------------------- 1. ablate dry-run
# The first real test of q8_ablate's shard code.  MUST print targets=145 shards_used=5.
if ! have 1-dryrun; then
    [ -f "$ADAPTER" ] || die "no adapter at $ADAPTER"
    say "1 dry-run: applying the adapter to 5 shards in memory, writing nothing"
    "$PY" "$T/q8_ablate.py" --model-dir "$SHARDS" --adapter "$ADAPTER" --scale 1.0 \
        --dry-run --report "$STATE/dryrun.json" > "$STATE/1-dryrun.out" 2>&1 \
        || die "ablate dry-run exited non-zero (see $STATE/1-dryrun.out)"
    head -8 "$STATE/1-dryrun.out"
    grep -q 'targets=145 shards_used=5' "$STATE/1-dryrun.out" \
        || die "dry-run did not report 'targets=145 shards_used=5' - STOPPING, do not force it"
    done_ 1-dryrun
fi

# ---------------------------------------------------------------- 2. range backup
# ~41 GB, written to the SATA drive.  '/' has ~69 GB free and NVME1TB has no room for both this and
# the quant output, so the rollback deliberately lives on a third volume.
if ! have 2-rollback; then
    mountpoint -q "/media/mal/Team CX2" || die "Team CX2 is not mounted - refusing to edit without a rollback"
    need=$(free_gb "/media/mal/Team CX2")
    [ "$need" -ge 60 ] || die "Team CX2 has ${need} GB free, need 60"
    say "2 rollback: per-shard range backup -> $ROLLBACK"
    "$PY" "$T/q8_ablate.py" --model-dir "$SHARDS" --adapter "$ADAPTER" --scale 1.0 \
        --make-range-backup "$ROLLBACK" > "$STATE/2-rollback.out" 2>&1 \
        || die "range backup failed (see $STATE/2-rollback.out)"
    tail -4 "$STATE/2-rollback.out"
    done_ 2-rollback
fi

# ---------------------------------------------------------------- 3. ablate (the in-place edit)
if ! have 3-ablate; then
    say "3 ablate: prepare"
    "$PY" "$T/q8_ablate.py" --model-dir "$SHARDS" --adapter "$ADAPTER" --scale 1.0 \
        --prepare --range-backup "$ROLLBACK" --report "$EDIT_REPORT" > "$STATE/3-prepare.out" 2>&1 \
        || die "ablate --prepare failed"
    say "3 ablate: editing tensors in place"
    "$PY" "$T/q8_ablate.py" --model-dir "$SHARDS" --adapter "$ADAPTER" --scale 1.0 \
        --report "$EDIT_REPORT" > "$STATE/3-ablate.out" 2>&1 \
        || die "ablate edit failed (journal at $EDIT_REPORT.journal.json - rerunnable)"
    tail -3 "$STATE/3-ablate.out"
    "$PY" -c "
import json,sys
r=json.load(open('$EDIT_REPORT'))
print('edit report:', {k:v for k,v in r.items() if k in ('targets','complete','model','shards','rollback_kind')})" \
        > "$STATE/3-report.txt" 2>&1 || true
    tail -6 "$STATE/3-report.txt"
    done_ 3-ablate
fi

# ---------------------------------------------------------------- 4. tensor-type file + size gate
if ! have 4-types; then
    say "4 types: allocating from $ALLOC"
    "$PY" "$T/prep_alloc.py" "$ALLOC" --out "$TYPES" --shard-dir "$SHARDS" \
        --estimate --max-gb "$MAX_GB" > "$STATE/4-types.out" 2>&1
    rc=$?
    tail -6 "$STATE/4-types.out"
    [ "$rc" -eq 0 ] || die "allocation prep/estimate failed (rc=$rc) - see $STATE/4-types.out"
    done_ 4-types
fi

# ---------------------------------------------------------------- 5. pack the ablated parent
# The engine loads a pack, not a shard set.  --gguf is shard 1 (output.weight + the output_hc_*
# head); iq_pack walks the rest of the shards itself.
#
# --compat-bf16 is REQUIRED and is not a quality knob here.  The engine reads the hyper-connection
# and router projections as BF16 from the pack (iq_pack.py FORM), but this source stores hc_* as
# Q8_0 and ffn_gate_inp as F32 — 362 tensors that would otherwise be refused with
# "<name> is Q8_0, but the engine requires BF16".  It dequantises and rounds to nearest-even, which
# for Q8_0 input is the same values llama-quantize would have produced.  1.54 GB of dense.bin.
if ! have 5-pack-parent; then
    say "5 pack: building the parent pack -> $PACK_PARENT"
    "$PY" "$XOLOTL/tools/iq_pack.py" --gguf "$SHARDS/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf" \
        --compat-bf16 --out "$PACK_PARENT" > "$STATE/5-pack.out" 2>&1 \
        || die "iq_pack on the ablated parent failed (see $STATE/5-pack.out)"
    tail -5 "$STATE/5-pack.out"
    done_ 5-pack-parent
fi

# ---------------------------------------------------------------- 6. quantise
if ! have 6-quantize; then
    [ -f "$IMATRIX" ] || die "no imatrix at $IMATRIX"
    [ -f "$TYPES" ] || die "no tensor-type file"
    # free space: the output is ~68 GB; refuse to start on a volume that cannot hold it
    f=$(free_gb /media/mal/NVME1TB)
    [ "$f" -ge 80 ] || die "NVME1TB has ${f} GB free, need 80 for the quant output"
    say "6 quantize: $(wc -l < "$TYPES") overrides, imatrix $(basename "$IMATRIX"), ${THREADS} threads"
    "$QUANTIZE" --imatrix "$IMATRIX" --tensor-type-file "$TYPES" --allow-requantize \
        --keep-split "$SHARDS/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf" \
        "$OUTBASE" Q8_0 "$THREADS" > "$STATE/6-quantize.out" 2>&1 \
        || die "llama-quantize failed (see $STATE/6-quantize.out)"
    tail -6 "$STATE/6-quantize.out"
    ls -l "$OUTBASE"-0*.gguf > "$STATE/6-output.txt" 2>&1 || die "no output shards produced"
    cat "$STATE/6-output.txt"
    tot=$(du -cb "$OUTBASE"-0*.gguf | tail -1 | cut -f1)
    say "6 quantize: total $((tot/1000000)) MB"
    if [ "$tot" -gt $((MAX_GB*1000000000)) ]; then
        die "output is $((tot/1000000)) MB, over the ${MAX_GB} GB cap"
    fi

    # An imatrix that does not cover a quantised tensor silently degrades it to plain RTN
    # (llama-quant.cpp looks the name up, misses, logs "did not find weights" and passes nullptr).
    # Independently cross-referenced against the imatrix's 926 tensor names: of the 447 tensors this
    # allocation actually quantises, 444 have weights and exactly 3 do not - output.weight,
    # token_embd.weight, per_layer_token_embd.weight.  Those three are embeddings/PLE, they are the
    # same three the reference build is missing, and they are 29.48 of 68.14 GB.  So the expected
    # count is EXACTLY 3; any fourth name means a tensor we meant to have weighted silently is not.
    say "6 quantize: checking which tensors the imatrix did not cover"
    grep -o "did not find weights for [^ ]*" "$STATE/6-quantize.out" 2>/dev/null \
        | sed 's/.*for //' | sed 's/[^A-Za-z0-9._-]*$//' | sort -u > "$STATE/6-imatrix-missing.txt" || true
    uncovered=$(wc -l < "$STATE/6-imatrix-missing.txt")
    say "6 quantize: $uncovered tensor(s) without imatrix weights"
    sed 's/^/    /' "$STATE/6-imatrix-missing.txt"
    if grep -qE "ffn_(gate|up|down)_exps" "$STATE/6-imatrix-missing.txt"; then
        die "expert tensors got no imatrix weights - the run would be plain RTN in the graded tier"
    fi
    unexpected=$(grep -vE '^(output|token_embd|per_layer_token_embd)\.weight$' \
        "$STATE/6-imatrix-missing.txt" | head -5)
    if [ -n "$unexpected" ]; then
        die "unexpected tensors without imatrix weights: $unexpected"
    fi
    say "6 quantize: uncovered set is the expected embeddings/PLE trio"

    # Did llama-quantize actually apply the allocation, or silently fall back?
    say "6 quantize: verifying every tensor's type against the allocation"
    "$PY" "$T/verify_quant.py" "$OUTBASE"-0*.gguf --types "$TYPES" --json "$STATE/6-verify.json" \
        > "$STATE/6-verify.out" 2>&1
    rc=$?
    cat "$STATE/6-verify.out"
    [ "$rc" -eq 0 ] || die "quantised file does not match the allocation (rc=$rc)"
    done_ 6-quantize
fi

# ---------------------------------------------------------------- 7. pack the quantised file
if ! have 7-pack-quant; then
    say "7 pack: building the quantised pack -> $PACK_QUANT"
    "$PY" "$XOLOTL/tools/iq_pack.py" --gguf "$OUT1" --compat-bf16 --out "$PACK_QUANT" \
        > "$STATE/7-pack.out" 2>&1 \
        || die "iq_pack on the quantised file failed (see $STATE/7-pack.out)"
    tail -5 "$STATE/7-pack.out"
    done_ 7-pack-quant
fi

# ---------------------------------------------------------------- 8. top-1 / KL gate vs the parent
# logits_gate.py compares strata --dump-logits from the ablated Q8 parent and from the quantised
# file.  Both are driven through the same packs and the same token ids.
if ! have 8-gate; then
    # Two separate formats, and getting either wrong produces confident garbage:
    #   strata --tokens-file  wants whitespace/comma-separated TEXT
    #   logits_gate --ids     does np.fromfile(dtype=int32), i.e. BINARY
    # And the row count is (n_prompt - 1 + max_new) (src/program/generate.cpp:6087), so a 512-token
    # prompt writes 512 rows and the gate needs at least 513 ids - one more than the prompt, which is
    # why the prompt file and the gate file cannot be the same file.
    "$PY" -c "
import numpy as np
ids = [int(x) for x in open('$CORPUS_IDS').read().split()]
P = 512
assert len(ids) >= P + 8, f'need {P+8} token ids, corpus has {len(ids)}'
assert max(ids) < 300000, 'token ids look wrong'
with open('$STATE/8-prompt.txt', 'w') as f:
    f.write(' '.join(map(str, ids[:P])) + '\n')
np.array(ids[:P + 8], dtype=np.int32).tofile('$STATE/8-gate.i32')
print('prompt', P, 'tokens; gate ids', P + 8, '; max id', max(ids[:P+8]))" \
        > "$STATE/8-ids.out" 2>&1 || die "could not build the prompt/gate ids (see $STATE/8-ids.out)"
    cat "$STATE/8-ids.out"

    for kind in parent quant; do
        [ -f "$STATE/8-$kind.logits" ] && continue
        if [ "$kind" = parent ]; then
            N=$SHARDS/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf
            P=$PACK_PARENT
            PL=$SHARDS/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00002-of-00005.gguf
        else
            N=$OUT1; P=$PACK_QUANT; PL=$OUT2
        fi
        [ -f "$N" ] || die "8 gate: no $N"
        [ -f "$PL" ] || die "8 gate: no PLE shard $PL"
        say "8 gate: dumping logits for $kind (native $(basename "$N"), ple $(basename "$PL"))"
        "$STRATA" --native "$N" --pack "$P" --ple-gguf "$PL" --spec 2 --ple-io direct \
            --mmap-experts --expert-cache 1536 --expert-profile "$PROFILE" --prefill 512 --max-context 2048 \
            --tokens-file "$STATE/8-prompt.txt" --max-new 1 --dump-logits "$STATE/8-$kind.logits" \
            > "$STATE/8-$kind.out" 2>&1 || die "strata logit dump failed for $kind"
    done
    # Report the gate; a failure is a RESULT, not a crash.  Stage 9 still runs so the speed number
    # is not lost, and status records both.
    "$PY" "$T/logits_gate.py" "$STATE/8-parent.logits" "$STATE/8-quant.logits" --ids "$STATE/8-gate.i32" \
        --json "$STATE/8-gate.json" > "$STATE/8-gate.out" 2>&1
    rc=$?
    cat "$STATE/8-gate.out"
    printf '%s' "$rc" > "$STATE/8-gate.rc"
    if [ "$rc" -eq 0 ]; then
        done_ 8-gate
    else
        say "8 gate: logits_gate exited $rc - recording, continuing to the speed test"
    fi
fi

# ---------------------------------------------------------------- 9. speed test
if ! have 9-speed; then
    say "9 speed: prefill/decode of the quantised file"
    "$PY" "$T/load_test.py" --model "$OUT1" --pack "$PACK_QUANT" --tokens-file "$IDS" \
        --prompt-tokens 512 --max-new 32 --json "$STATE/9-speed.json" \
        > "$STATE/9-speed.out" 2>&1 || say "9 speed: load_test failed, see 9-speed.out"
    tail -14 "$STATE/9-speed.out" 2>/dev/null
    done_ 9-speed
fi

gate=$(cat "$STATE/8-gate.rc" 2>/dev/null || echo "not-run")
if [ "$gate" = "0" ]; then
    ok "COMPLETE - gate PASSED - artifact $OUTBASE-0000{1,2}-of-00005.gguf"
else
    ok "QUANT COMPLETE, GATE FAILED (rc=$gate) - artifact $OUTBASE-0000{1,2}-of-00005.gguf"
fi
say "=== QFN V2 night chain finished ==="
