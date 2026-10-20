#!/usr/bin/env bash
# reassemble.sh - join V41-GSQ-1017.gguf from its byte-split parts (V41-GSQ-1017.gguf.part-NN-of-MM) and verify it.
#
#   bash reassemble.sh [DIR] [--delete-parts]
#
# DIR: where the parts and SHA256SUMS are (default: this script's directory).  The joined file is written there.
# --delete-parts: remove each part right after it has been appended (needs ~152 GB free instead of ~304 GB);
#                 every part is checksummed BEFORE anything is joined or deleted.
# The parts are plain byte ranges of the original file (`cat` order = part number), not llama.cpp gguf-split
# shards: Strata-DSV4.1's GSQ file uses expert types (PTQ1_0 / Q2_0 maps) that gguf-split does not know.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
del=0
for a in "$@"; do
    case "$a" in
        --delete-parts) del=1 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) dir="$(cd "$a" && pwd)" ;;
    esac
done
cd "$dir"
out=V41-GSQ-1017.gguf
[ -f SHA256SUMS ] || { echo "SHA256SUMS not found in $dir"; exit 1; }
mapfile -t parts < <(ls -1 "$out".part-*-of-* 2>/dev/null | sort)
[ ${#parts[@]} -gt 0 ] || { echo "no $out.part-* files in $dir"; exit 1; }
total="${parts[0]##*-of-}"
[ ${#parts[@]} -eq "$((10#$total))" ] || { echo "found ${#parts[@]} parts, expected $total"; exit 1; }

echo "checking ${#parts[@]} parts..."
grep -E "  $out\.part-" SHA256SUMS | sha256sum -c --quiet - || { echo "a part is corrupt - re-download it"; exit 1; }

[ -e "$out" ] && { echo "$out already exists - move it away first"; exit 1; }
need=0
for p in "${parts[@]}"; do need=$((need + $(stat -c %s "$p"))); done
free=$(( $(df --output=avail -B1 . | tail -1) ))
if [ $del -eq 0 ] && [ $free -lt $need ]; then
    echo "need $((need / 1000000000)) GB free for the joined file (have $((free / 1000000000)) GB) - use --delete-parts"
    exit 1
fi

for p in "${parts[@]}"; do
    echo "appending $p"
    cat "$p" >> "$out"
    if [ $del -eq 1 ]; then rm -f "$p"; fi
done

echo "verifying $out (a few minutes)..."
grep -E "  $out\$" SHA256SUMS | sha256sum -c - && echo "OK: $out is complete"
