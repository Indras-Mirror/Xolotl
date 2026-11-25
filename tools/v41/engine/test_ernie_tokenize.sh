#!/usr/bin/env bash
# tools/v41/engine/test_ernie_tokenize.sh - the ernie5 tokenizer gate.
#
# The invariant: for every string below,
#   ernie_generate --encode-only -m GGUF --no-bos -p "$S"
# produces exactly the ids llama.cpp's own tokenizer produces:
#   llama-tokenize -m GGUF --ids --no-bos -p "$S"
# and decode(encode(S)) == S, byte for byte.
#
#   ERNIE_GEN=<binary>  LLAMA_TOKENIZE=<binary>  ERNIE_GGUF=<file>  ./test_ernie_tokenize.sh
set -u

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${ERNIE_GEN:-$ROOT/build-ernie-g3/ernie_generate}
GG=${ERNIE_GGUF:?set ERNIE_GGUF=<an ERNIE-5.5 GGUF (any quant; a small --layers 3 conversion is enough)>}
PR=${LLAMA_TOKENIZE:-$(cd "$(dirname "$0")/../../.." && pwd)/third_party/llama.cpp-ernie55/build-cuda/bin/llama-tokenize}

for f in "$BIN" "$GG" "$PR"; do
    if [ ! -e "$f" ]; then
        echo "missing: $f" >&2
        exit 2
    fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

n=0
same=0
rt=0
fail=0

check() {
    local s=$1
    n=$((n + 1))
    local ours pr
    ours=$("$BIN" --encode-only -m "$GG" --no-bos -p "$s" 2>"$tmp/err") || {
        echo "FAIL [$n] encoder exited non-zero: $(cat "$tmp/err")"
        fail=$((fail + 1))
        return
    }
    pr=$("$PR" -m "$GG" --ids --no-bos -p "$s" 2>/dev/null | grep -E '^\[[0-9, ]*\]$' | tail -1 |
        sed -e 's/[][]//g' -e 's/ //g')
    if [ "$ours" = "$pr" ]; then
        same=$((same + 1))
    else
        echo "FAIL [$n] ids differ for: $(printf '%s' "$s" | head -c 80)"
        echo "  ours: $ours"
        echo "  pr:   $pr"
        fail=$((fail + 1))
    fi
    printf '%s' "$s" >"$tmp/in"
    if "$BIN" --decode "$ours" -m "$GG" >"$tmp/dec" 2>"$tmp/err" && cmp -s "$tmp/in" "$tmp/dec"; then
        rt=$((rt + 1))
    else
        echo "FAIL [$n] round-trip differs for: $(printf '%s' "$s" | head -c 80) ($(cat "$tmp/err"))"
        fail=$((fail + 1))
    fi
}

a300=$(printf 'a%.0s' $(seq 1 300))

check 'The capital of France is'
check '1234567 items'
check '1'
check '12'
check '123'
check '1234'
check '1234567890123'
check 'Hello, world!  1234'
check $'def f(x): return x+1\n'
check $'  leading spaces and\ttabs\n\nnewlines'
check 'MiXeD CaSe WoRds and_underscores-and-dashes'
check 'emoji 🙂🚀 and accents café naïve'
check '日本語のテキストと中文混在'
check "$a300"
check $'\n\n\n=== section ===\n'
check $'print(\'hello\')  # comment\n'
check '42+58=100; 3.14159; 1,000,000'
check 'URL https://example.com/path?q=1&y=2#x'
check $'\u00a0nbsp\u2003emspace'
check 'trailing spaces   '

echo "ids identical to llama-tokenize: $same/$n"
echo "round-trip decode(encode(s)) == s: $rt/$n"

bos=$("$BIN" --encode-only -m "$GG" --bos -p 'The capital of France is')
want='2,2029,10301,1507,13573,1543'
if [ "$bos" = "$want" ]; then
    echo "BOS line: $bos (expected)"
else
    echo "FAIL BOS line: got '$bos', want '$want'"
    fail=$((fail + 1))
fi

if [ "$fail" -ne 0 ]; then
    echo "FAILED: $fail check(s)"
    exit 1
fi
echo "OK"
