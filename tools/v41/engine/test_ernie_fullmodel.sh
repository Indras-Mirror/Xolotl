#!/usr/bin/env bash
# tools/v41/engine/test_ernie_fullmodel.sh - the ERNIE-5.5-Flash full-model gate.
#
# Given one GGUF and one prompt, this script proves three things about that file:
#
#   1. TOKENIZER: our `ernie_generate --encode-only` ids equal the llama.cpp PR tokenizer's ids
#      (`llama-tokenize --ids`) for the prompt - the per-file tokenizer gate.
#   2. OURS: ernie_generate (dense graphs + routed-expert tier) generates N greedy tokens from
#      that prompt; the ids come from its `generated:` line, the text from --out-text.
#   3. REFERENCE: the PR build's llama-simple generates the same N greedy tokens, and its output
#      TEXT is tokenized back to ids. The two id lists are compared position by position.
#
# llama-simple in this PR build (examples/simple/simple.cpp) has a bespoke three-flag parser:
# `-m`, `-n` and `-ngl`, then the prompt as positional text. It has NO `-t` and NO `-ot`, so the
# packet's `-t T -ot exps=CPU` cannot be honoured literally. `-ngl 0` is the faithful equivalent:
# the whole graph (dense half AND all experts) runs on the CPU, so the 4090 never has to hold the
# expert tier. REF_NGL overrides it.
#
# The reference emits text, not ids, and llama-eval-callback (the only other PR binary) prints the
# INPUT ids of a single decode - no sampling loop - so it cannot yield generated ids. The id
# extraction is therefore the packet's fallback: tokenize llama-simple's generated text back with
# `llama-tokenize --ids --no-bos --no-escape`. Re-tokenizing joined pieces is canonical rather than
# the emitted sequence, so a subword boundary can show a false divergence; the TEXT verdict is
# printed beside it to tell the two apart.
#
# usage: ./test_ernie_fullmodel.sh <gguf> "<prompt>" <n_tokens> [arena_gib]
#   arena_gib defaults to 8; with ERNIE_EXPERTS=gpu it is the host arena (the VRAM cache is
#   sized by --slots 0 = auto).
#
# env:
#   ERNIE_GEN       our binary (default: the first of build-fg-cpu / build-fg-gpu / build-ernie-g3)
#   ERNIE_EXPERTS   cpu | gpu                                       (default cpu)
#   ERNIE_THREADS   dense-graph CPU threads                          (default 8)
#   ERNIE_CUDA      1 = also run the dense graphs on CUDA device 0   (default off)
#   ERNIE_SLOTS     VRAM expert slots, 0 = auto                      (default 0)
#   LLAMA_SIMPLE    reference binary (default: the PR build's)
#   LLAMA_TOKENIZE  PR tokenizer binary (default: the PR build's)
#   LLAMA_PERPLEXITY  the NLL arm's reference (default: the PR build's llama-perplexity)
#   REF_NGL         llama-simple -ngl (default 0: reference on the CPU)
#   NLL_ARM         1 = the llama-perplexity vs teacher-forced NLL corroboration (default 1)
#   NLL_CTX         its window in tokens (default 32; the text is repeated until 2*NLL_CTX-1 tokens exist)
#   TIMEOUT         hard per-arm limit in seconds (default 2700 = 45 min); a timed-out arm prints
#                   `TIMEOUT-ARM <name>` and the run continues to the next arm
#   KEEP            1 = keep the scratch dir and print its path
#   ALLOW_LOW_MEM   1 = run even if MemAvailable < 60 GiB (default: refuse with exit 2)
#
# ERNIE_EXPERTS=gpu never falls back to a CPU build: with no CUDA-flavour binary it exits 2, and a
# run whose tier line is not `tier gpu:` is refused.  Elapsed seconds, s/token and tok/s are printed
# for every arm.
#
# The bar is the packet's >=95%; at n=16 that is already 16/16 (15/16 = 93.8% fails), i.e. every
# token must match.
#
# exit: 0 PASS, 1 FAIL, 2 prerequisites missing/incomplete (or gpu requested without a GPU run, or
#       low memory), 3 the GPU is busy.
set -u
exec 3>&1   # fd 3 = the real stdout: arm progress must escape the arm's own redirection

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PR_BIN=${LLAMA_PR_BIN:-$ROOT/third_party/llama.cpp-ernie55/build-cuda/bin}   # llama.cpp PR #31925 build

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: $0 <gguf> \"<prompt>\" <n_tokens> [arena_gib]" >&2
    exit 2
fi

GG=$1
PROMPT=$2
N=$3
ARENA=${4:-8}

EXPERTS=${ERNIE_EXPERTS:-cpu}
THREADS=${ERNIE_THREADS:-8}
SLOTS=${ERNIE_SLOTS:-0}
NGL=${REF_NGL:-0}

if ! [[ $N =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "n_tokens must be a positive integer (got '$N')" >&2
    exit 2
fi
case $EXPERTS in
    cpu|gpu) ;;
    *) echo "ERNIE_EXPERTS must be cpu or gpu (got '$EXPERTS')" >&2; exit 2 ;;
esac

if [ -n "${ERNIE_GEN:-}" ]; then
    GEN=$ERNIE_GEN
else
    GEN=""
    if [ "$EXPERTS" = "gpu" ]; then   # never fall back to a CPU flavour: a silent CPU run would look like a pass
        cands=("$ROOT/build-fg-gpu/ernie_generate" "$ROOT/build-ernie-g3-cuda/ernie_generate")
    else
        cands=("$ROOT/build-fg-cpu/ernie_generate" "$ROOT/build-fg-gpu/ernie_generate" "$ROOT/build-ernie-g3/ernie_generate")
    fi
    for c in "${cands[@]}"; do
        [ -x "$c" ] && { GEN=$c; break; }
    done
    if [ -z "$GEN" ] && [ "$EXPERTS" = "gpu" ]; then
        echo "gpu requested, no CUDA binary at ${cands[*]}" >&2
        exit 2
    fi
fi
SIMPLE=${LLAMA_SIMPLE:-$PR_BIN/llama-simple}
TOK=${LLAMA_TOKENIZE:-$PR_BIN/llama-tokenize}

# ---- feature detect ------------------------------------------------------------------------
if [ ! -e "$GG" ]; then
    echo "MISSING: model file '$GG' does not exist" >&2
    exit 2
fi
case $GG in
    *.partial)
        echo "INCOMPLETE: '$GG' is a .partial file - wait for the conversion to finish" >&2
        exit 2 ;;
esac
if [ -e "$GG.partial" ]; then
    echo "INCOMPLETE: '$GG.partial' still exists - the GGUF is not whole yet" >&2
    exit 2
fi
for f in "$GEN" "$SIMPLE" "$TOK"; do
    if [ -z "$f" ] || [ ! -x "$f" ]; then
        echo "MISSING: '$f' is not an executable (set ERNIE_GEN / LLAMA_SIMPLE / LLAMA_TOKENIZE)" >&2
        exit 2
    fi
done
if [ "$EXPERTS" = "gpu" ] || [ "${ERNIE_CUDA:-0}" = "1" ]; then
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
    if [ -n "$busy" ]; then
        echo "GPU BUSY: compute apps present ($busy) - refusing the GPU run" >&2
        exit 3
    fi
fi

TIMEOUT=${TIMEOUT:-2700}
mem_avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_avail_gib" -lt 60 ] && [ "${ALLOW_LOW_MEM:-0}" != "1" ]; then
    echo "SKIP: MemAvailable ${mem_avail_gib} GiB < 60 GiB (arena $ARENA GiB would thrash; set ALLOW_LOW_MEM=1 to override)" >&2
    exit 2
fi

# one arm under the hard per-arm limit: ARMRC / ARMSECS are the results
run_arm() {
    local name=$1
    shift
    local t0 t1
    t0=$(date +%s)
    timeout --signal=TERM --kill-after=30 "$TIMEOUT" "$@"
    ARMRC=$?
    t1=$(date +%s)
    ARMSECS=$((t1 - t0))
    if [ "$ARMRC" -eq 124 ] || [ "$ARMRC" -eq 137 ]; then
        echo "TIMEOUT-ARM $name after ${ARMSECS}s (limit ${TIMEOUT}s)" >&3
    fi
}
rate() {   # $1 tokens, $2 seconds
    awk -v n="$1" -v s="$2" 'BEGIN{ if (s > 0) printf "%d tok, %.1fs, %.2fs/token, %.2f tok/s", n, s, s/n, n/s; else printf "%d tok", n }'
}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ernie-fullgate.XXXXXX")
if [ "${KEEP:-0}" = "1" ]; then
    echo "scratch: $tmp"
else
    trap 'rm -rf "$tmp"' EXIT
fi

fam=$("$GEN" --encode-only -m "$GG" --bos -p "$PROMPT" 2>"$tmp/enc.err")
if [ $? -ne 0 ]; then
    echo "FAIL: our encoder failed: $(cat "$tmp/enc.err")" >&2
    exit 2
fi
printf '%s\n' "$fam" | tr ',' '\n' >"$tmp/ids"
nprompt=$(wc -l <"$tmp/ids")

# ---- 1. the per-file tokenizer gate --------------------------------------------------------
ref_tok=$("$TOK" -m "$GG" --ids --no-escape -p "$PROMPT" 2>"$tmp/tok.err" |
    sed -n 's/^\[\(.*\)\]$/\1/p' | tail -1 | tr -d ' ')
ours_tok=$(tr '\n' ',' <"$tmp/ids" | sed -e 's/,$//')
tokfail=0
if [ "$ours_tok" = "$ref_tok" ]; then
    echo "TOKENIZER: IDENTICAL ($ours_tok)"
else
    echo "TOKENIZER: DIFFER"
    echo "  ours: $ours_tok"
    echo "  ref:  $ref_tok"
    tokfail=1   # the reference is then fed a different prompt, so its ids are not comparable
fi

# ---- 2. ours -------------------------------------------------------------------------------
extra=()
[ "$EXPERTS" = "gpu" ] && extra+=(--slots "$SLOTS")
[ "${ERNIE_CUDA:-0}" = "1" ] && extra+=(--cuda)
G=()
ours_ok=0
run_arm ours-greedy "$GEN" "$GG" --ids-file "$tmp/ids" -n "$N" --experts "$EXPERTS" --arena-gib "$ARENA" \
    --threads "$THREADS" ${extra[@]+"${extra[@]}"} --out-text "$tmp/ours.txt" \
    >"$tmp/ours.out" 2>"$tmp/ours.err"
if [ "$ARMRC" -ne 0 ]; then
    echo "FAIL: ernie_generate exited $ARMRC (ours-greedy, ${ARMSECS}s):" >&2
    tail -5 "$tmp/ours.err" >&2
elif ! grep -q '^generated:' "$tmp/ours.out"; then
    echo "FAIL: no 'generated:' line from ernie_generate (ours-greedy, ${ARMSECS}s)" >&2
    tail -5 "$tmp/ours.out" >&2
else
    ours_ok=1
    mapfile -t G < <(sed -n 's/^generated://p' "$tmp/ours.out" | tr ' ' '\n' | grep -v '^$')
    echo "OURS: $(rate "${#G[@]}" "$ARMSECS"), ids: ${G[*]}"
    sed -n 's/^prefill /  ours: prefill /p; s/^tier /  ours: tier /p' "$tmp/ours.out"
    grep -E '^WARNING' "$tmp/ours.err" | sed 's/^/  /' | head -2
    # --experts gpu must really be on the GPU: the tier's mode line is the evidence, not the flag
    if [ "$EXPERTS" = "gpu" ] && ! grep -q '^tier gpu:' "$tmp/ours.out"; then
        echo "GPU NOT USED: --experts gpu produced no 'tier gpu:' line (silent CPU fallback?) - refusing the result" >&2
        tail -3 "$tmp/ours.err" >&2
        exit 2
    fi
    echo "  (ours' greedy arm done; reference arm next - it is the slow, all-CPU side)"
fi

# ---- 3. reference --------------------------------------------------------------------------
echo "REFERENCE: llama-simple -m '$GG' -n $N -ngl $NGL \"$PROMPT\"  (all-CPU: no -ot/-t in this binary)"
run_arm reference-greedy "$SIMPLE" -m "$GG" -n "$N" -ngl "$NGL" "$PROMPT" >"$tmp/ref.out" 2>"$tmp/ref.err"
ref_secs=$ARMSECS
ref_timedout=0
if [ "$ARMRC" -eq 124 ] || [ "$ARMRC" -eq 137 ]; then ref_timedout=1; fi
[ "$ARMRC" -ne 0 ] && echo "  note: llama-simple exited $ARMRC after ${ARMSECS}s; keeping the partial output"
# the same binary with -n 0 renders the prompt alone; the generated text is the remainder
pn=0
[ "$nprompt" -le 1 ] && pn=1   # n_ctx = n_prompt + n_predict - 1 must stay > 0
run_arm reference-prompt "$SIMPLE" -m "$GG" -n "$pn" -ngl "$NGL" "$PROMPT" >"$tmp/refp.out" 2>"$tmp/refp.err"

strip_nl() {   # $1 in, $2 out: drop exactly one trailing newline (llama-simple's own)
    local sz
    sz=$(stat -c%s "$1" 2>/dev/null || echo 0)
    if [ "$sz" -gt 0 ] && [ "$(tail -c 1 "$1" | od -An -tu1 | tr -d ' ')" = "10" ]; then
        head -c $((sz - 1)) "$1" >"$2"
    else
        cp "$1" "$2" 2>/dev/null || : >"$2"
    fi
}
strip_nl "$tmp/ref.out" "$tmp/ref.txt"
strip_nl "$tmp/refp.out" "$tmp/refp.txt"

plen=$(stat -c%s "$tmp/refp.txt" 2>/dev/null || echo 0)
if [ "$plen" -gt 0 ] && head -c "$plen" "$tmp/ref.txt" 2>/dev/null | cmp -s - "$tmp/refp.txt"; then
    tail -c "+$((plen + 1))" "$tmp/ref.txt" >"$tmp/refgen.txt"
else
    echo "  note: reference prompt rendering does not prefix its output; using the whole output"
    cp "$tmp/ref.txt" "$tmp/refgen.txt"
fi

R=()
if [ -s "$tmp/refgen.txt" ]; then
    mapfile -t R < <("$TOK" -m "$GG" --ids --no-bos --no-escape -f "$tmp/refgen.txt" 2>>"$tmp/tok.err" |
        sed -n 's/^\[\(.*\)\]$/\1/p' | tail -1 | tr ',' '\n' | tr -d ' ' | grep -v '^$')
fi
echo "REFERENCE: $(rate "${#R[@]}" "$ref_secs"), ids: ${R[*]:-<empty>}"
sed -n 's/^llama_perf_context_print: \(prompt eval time\)/  ref: \1/p; s/^llama_perf_context_print: \( *eval time\)/  ref: \1/p' "$tmp/ref.err"
[ -s "$tmp/refgen.txt" ] && printf '  text: %s\n' "$(head -c 200 "$tmp/refgen.txt" | tr '\n' ' ')"

# ---- 4. corroboration: teacher-forced NLL over one window, vs llama-perplexity's ln(PPL) ----
# The greedy ids are the gate; this arm is the sensitive second opinion. llama-perplexity needs at
# least 2*n_ctx-1 tokens; one chunk of NLL_CTX is evaluated (stride 0, so chunk 1 starts at token 0,
# exactly our rows 0..NLL_CTX-1). Its NLL is ln(Final estimate PPL); ours is
# -mean(log_softmax(row i)[ids[i+1]]) from --dump-logits-rows.
PERPLEXITY=${LLAMA_PERPLEXITY:-$PR_BIN/llama-perplexity}
NLL_CTX=${NLL_CTX:-32}
if [ "${NLL_ARM:-1}" != "1" ]; then
    echo "NLL: skipped (NLL_ARM=0)"
elif [ ! -x "$PERPLEXITY" ]; then
    echo "NLL: skipped (no $PERPLEXITY)"
elif ! python3 -c 'import numpy' 2>/dev/null; then
    echo "NLL: skipped (python3 + numpy not available)"
else
    nlltxt=$tmp/nll.txt
    : >"$nlltxt"
    reps=0
    ntok=0
    while [ "$reps" -lt 64 ] && [ "$ntok" -lt "$((2 * NLL_CTX - 1))" ]; do
        reps=$((reps + 1))
        [ "$reps" -eq 1 ] && printf '%s' "$PROMPT" >>"$nlltxt" || printf ' %s' "$PROMPT" >>"$nlltxt"
        ntok=$("$TOK" -m "$GG" --ids --no-escape -f "$nlltxt" 2>/dev/null |
            sed -n 's/^\[\(.*\)\]$/\1/p' | tail -1 | tr ',' '\n' | grep -c '[0-9]')
    done
    mapfile -t IDA < <("$TOK" -m "$GG" --ids --no-escape -f "$nlltxt" 2>/dev/null |
        sed -n 's/^\[\(.*\)\]$/\1/p' | tail -1 | tr ',' '\n' | tr -d ' ' | grep -v '^$')
    printf '%s\n' "${IDA[@]:0:NLL_CTX}" >"$tmp/nll_ids"
    # OURS' arm first: it is fast, so there is a real number even if the (disk-bound) reference never finishes
    run_arm ours-nll "$GEN" "$GG" --ids-file "$tmp/nll_ids" -n 0 --prefill-chunk 0 --dump-logits-rows "$tmp/nll_rows.f32" \
        --experts "$EXPERTS" --arena-gib "$ARENA" --threads "$THREADS" ${extra[@]+"${extra[@]}"} \
        >"$tmp/nll.out" 2>"$tmp/nll.err"
    rc=$ARMRC
    nll_secs=$ARMSECS
    cat >"$tmp/nll.py" <<'PY'
import sys
import numpy as np
rows, idsf = sys.argv[1], sys.argv[2]
ids = [int(x) for x in open(idsf).read().split()]
n = len(ids)
raw = np.fromfile(rows, dtype=np.float32)
vocab = raw.size // n
if n < 2 or vocab < 1 or raw.size % n:
    sys.exit(1)
L = raw.reshape(n, vocab)
losses = np.empty(n - 1)
for i in range(n - 1):
    x = L[i].astype(np.float64)
    m = x.max()
    losses[i] = (m + np.log(np.exp(x - m).sum())) - x[ids[i + 1]]
# llama-perplexity scores only rows [n_ctx/2, n_ctx-1) (perplexity.cpp: first = n_ctx/2): the same rows here, or the cold
# first half (losses 5-11 nats on a repeated prompt) makes ours look 16x worse (measured: 1.31 vs 0.080; same rows 0.093)
print("%.6f" % float(losses[n // 2:].mean()))
PY
    ours_nll=""
    if [ $rc -eq 0 ] && [ -s "$tmp/nll_rows.f32" ]; then
        ours_nll=$(python3 "$tmp/nll.py" "$tmp/nll_rows.f32" "$tmp/nll_ids" 2>>"$tmp/nll.err")
        sed -n 's/^prefill /  ours nll pass: prefill /p' "$tmp/nll.out"
        echo "  ours NLL (teacher-forced, $((NLL_CTX - 1)) predictions, ${nll_secs}s): ${ours_nll:-<failed>} - reference arm next"
    fi
    # the reference arm: dense half on the GPU, experts on CPU (-ot exps=CPU) is the fast combo; if this build
    # refuses -ot, fall back to -ngl 0 (all CPU) and say which one produced the number.  A timeout ends the
    # arm (a fallback would cost another TIMEOUT seconds for the same verdict-less answer).
    ppl=""
    ppl_mode=""
    ppl_timedout=0
    try_ppl() {
        local label=$1
        shift
        run_arm reference-nll "$PERPLEXITY" -m "$GG" -f "$nlltxt" -c "$NLL_CTX" --chunks 1 --no-repack "$@" \
            >"$tmp/ppl.out" 2>"$tmp/ppl.err"
        ppl_secs=$ARMSECS
        if [ "$ARMRC" -eq 124 ] || [ "$ARMRC" -eq 137 ]; then ppl_timedout=1; return 0; fi
        ppl=$(sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p' "$tmp/ppl.out" "$tmp/ppl.err" | tail -1)
        [ -z "$ppl" ] && ppl=$(sed -n 's/^\[1\]\([0-9.]*\),.*/\1/p' "$tmp/ppl.out" | tail -1)
        [ -z "$ppl" ] && ppl=$(sed -n 's/.*\[1\]\([0-9.]*\),.*/\1/p' "$tmp/ppl.out" | tail -1)
        [ -n "$ppl" ] && ppl_mode="$label (${ARMSECS}s)"
        return 0
    }
    if [ "$EXPERTS" = "gpu" ] || [ "${ERNIE_CUDA:-0}" = "1" ]; then
        try_ppl "-ot exps=CPU (dense half on the GPU)" -ot 'exps=CPU'
        [ -z "$ppl" ] && [ "$ppl_timedout" -eq 0 ] && try_ppl "-ngl 0 -ot exps=CPU (all CPU)" -ngl 0 -ot 'exps=CPU'
    else   # a CPU-intent run keeps the 4090 out of it
        try_ppl "-ngl 0 -ot exps=CPU (all CPU)" -ngl 0 -ot 'exps=CPU'
    fi
    [ -z "$ppl" ] && [ "$ppl_timedout" -eq 0 ] && try_ppl "-ngl 0 (all CPU, no override)" -ngl 0
    if [ -n "$ours_nll" ] && [ -n "$ppl" ]; then
        ref_nll=$(awk -v p="$ppl" 'BEGIN{ if (p > 0) printf "%.6f", log(p) }')
        delta=$(awk -v a="$ours_nll" -v b="$ref_nll" 'BEGIN{ if (b != "") printf "%+.6f", a - b }')
        echo "NLL ($NLL_CTX-token window, $((NLL_CTX - 1)) predictions): ours=$ours_nll ref=$ref_nll (ref PPL=$ppl, delta=$delta; ref mode: $ppl_mode)"
    else
        echo "NLL: unavailable (ours='${ours_nll}' ref PPL='${ppl}'; see $tmp/nll.err $tmp/ppl.err)"
    fi
fi

# ---- 5. the verdict ------------------------------------------------------------------------
# ours' generated text comes from --decode of the generated ids, not from --out-text: the `--ids`
# path loads no tokenizer, so its out-text renders every prompt token as <id>/<|SPECIAL|> and the
# prompt/generated boundary inside it is not the reference's. --decode uses the real tokenizer.
nours=${#G[@]}
gen_csv=$(IFS=,; echo "${G[*]}")
if [ "$nours" -gt 0 ] && "$GEN" --decode "$gen_csv" -m "$GG" >"$tmp/oursgen.txt" 2>/dev/null; then :; else
    [ "$nours" -gt 0 ] && echo "  note: --decode of our generated ids failed; skipping the text comparison"
    : >"$tmp/oursgen.txt"
fi
[ -s "$tmp/oursgen.txt" ] && printf '  ours text: %s\n' "$(head -c 200 "$tmp/oursgen.txt" | tr '\n' ' ')"
[ -s "$tmp/refgen.txt" ] && printf '  ref  text: %s\n' "$(head -c 200 "$tmp/refgen.txt" | tr '\n' ' ')"
if cmp -s "$tmp/oursgen.txt" "$tmp/refgen.txt"; then
    echo "TEXT-AGREEMENT: identical"
else
    echo "TEXT-AGREEMENT: differs (ours $(wc -c <"$tmp/oursgen.txt") bytes, ref $(wc -c <"$tmp/refgen.txt") bytes)"
fi

if [ "$ours_ok" -eq 0 ]; then
    echo "TOP1-AGREEMENT: n/a - ours' greedy arm produced no ids (TIMEOUT-ARM / failure above)"
    echo "VERDICT-INCOMPLETE"
    exit 1
fi
k=0
firstdiv=-1
for ((i = 0; i < nours; i++)); do
    if [ "$i" -lt "${#R[@]}" ] && [ "${G[$i]}" = "${R[$i]}" ]; then
        k=$((k + 1))
    elif [ "$firstdiv" -lt 0 ]; then
        firstdiv=$i
    fi
done
pct=$((100 * k / nours))
echo "TOP1-AGREEMENT: $k/$nours ($pct%)"
if [ "$firstdiv" -ge 0 ]; then
    echo "  first divergence at $firstdiv: ours=${G[$firstdiv]:-<none>} ref=${R[$firstdiv]:-<none>}"
fi
if [ "${#R[@]}" -eq 0 ]; then
    echo "  (the reference arm produced no ids - TIMEOUT-ARM above?; the gate needs both sides)"
    echo "VERDICT-INCOMPLETE"
    exit 1
fi
if [ "$tokfail" -ne 0 ]; then
    echo "  (tokenizer gate failed: this id comparison used two different prompts)"
fi
if [ "$ref_timedout" -eq 1 ]; then
    echo "  (reference arm timed out: only the ${#R[@]} tokens it had emitted are compared)"
fi
if [ "$tokfail" -eq 0 ] && [ "$((100 * k))" -ge "$((95 * nours))" ]; then
    echo "PASS"
    exit 0
fi
echo "FAIL"
exit 1
