#!/bin/bash
# fetch_fp8.sh DEST - resumable, hash-verified fetch of Qwen/Qwen3.8-Flash-Next-FP8.
#
# Downloads the 112 files that carry the weights (135.0 GB) and SKIPS the 32 PLE-only shards
# (50.5 GB): the PLE n-gram table is fetched separately as IQ4_NL (fetch_ple.sh), because the
# engine only reads it as IQ4_NL and dequantizing the FP8 copy would need ~205 GB of RAM.
#
# Every file is checked against the hash HuggingFace publishes for it (sha256 for the LFS weight
# shards, git sha1 for the small text files).  This is NOT optional on this box: a Crucial 32 GB
# DIMM has a confirmed bad byte lane at 56.6 GB (amethyst: ram-fault-memtest-2026-09-18) that has
# silently corrupted model files before.  A file that fails its hash is deleted and re-fetched; if
# the same file fails twice at full size the script stops and points at the RAM rather than
# looping forever (a second identical mismatch is far more likely to be the DIMM than the network).
#
# Resumable: aria2c -c continues partial files, and a `.ok` marker means "on disk AND verified",
# so a dropped connection, a killed shell or a reboot just continues on the next run.
#
#   bash tools/qwen38/fetch_fp8.sh "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8"
set -u
DEST=${1:?usage: fetch_fp8.sh DEST}
HERE=$(dirname "$(readlink -f "$0")")
MAN=$HERE/fp8-manifest.json
REPO=Qwen/Qwen3.8-Flash-Next-FP8
BASE="https://huggingface.co/$REPO/resolve/main"
MIN_FREE_GB=${MIN_FREE_GB:-5}
CONN=${CONN:-16}         # connections per file

mkdir -p "$DEST" || exit 1
cd "$DEST" || exit 1

python3 "$HERE/manifest_list.py" "$MAN" > .fetch-list || { echo "manifest read failed"; exit 1; }

verified() {   # verified NAME SIZE ALGO HASH
  [ -f "$1.ok" ] || return 1
  [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" = "$2" ] || return 1
  return 0
}

skipped=0; need=0
while IFS=$'\t' read -r name size algo hash; do
  if verified "$name" "$size" "$algo" "$hash"; then skipped=$((skipped+1)); else need=$((need+1)); fi
done < .fetch-list
echo "$(date '+%F %T') files already verified: $skipped   to fetch: $need"

while IFS=$'\t' read -r name size algo hash; do
  verified "$name" "$size" "$algo" "$hash" && continue
  free_gb=$(df --output=avail -BG "$DEST" | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "$(date '+%F %T') STOP: only ${free_gb}G free - free space and re-run"; exit 2
  fi
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') [$name] $(numfmt --to=iec "$size") attempt $attempt"
    aria2c -c -x"$CONN" -s"$CONN" -k1M --file-allocation=none \
           --max-tries=5 --retry-wait=10 --timeout=60 --console-log-level=warn \
           --summary-interval=60 --auto-file-renaming=false --allow-overwrite=true \
           -d "$DEST" -o "$name" "$BASE/$name" || true
    got=$(stat -c %s "$name" 2>/dev/null || echo 0)
    if [ "$got" != "$size" ]; then
      echo "$(date '+%F %T') [$name] size $got / $size - will retry"; continue
    fi
    echo "$(date '+%F %T') [$name] verifying $algo ..."
    # HF's `oid` for a non-LFS file is the GIT BLOB hash: sha1("blob <len>\0" + content).
    # That is not sha1sum of the bytes; using sha1sum here produces a guaranteed false MISMATCH.
    if [ "$algo" = "sha256" ]; then actual=$(sha256sum "$name" | cut -d' ' -f1)
    else                             actual=$(git hash-object -- "$name"); fi
    if [ "$actual" = "$hash" ]; then
      touch "$name.ok"; echo "$(date '+%F %T') [$name] VERIFIED"; break
    fi
    echo "$(date '+%F %T') [$name] $algo MISMATCH"
    echo "  expected $hash"
    echo "  actual   $actual"
    rm -f "$name"
    if [ "$attempt" -ge 2 ]; then
      echo "$(date '+%F %T') STOP: $name failed its hash twice at full size."
      echo "  That is the signature of the known bad DIMM (bad byte lane ~56.6 GB)."
      echo "  Run: python3 ~/AI/jev/reflex/verify_models.py  and memtest before blaming the network."
      exit 3
    fi
  done
done < .fetch-list

bad=0
while IFS=$'\t' read -r name size algo hash; do
  verified "$name" "$size" "$algo" "$hash" || { echo "NOT COMPLETE: $name"; bad=$((bad+1)); }
done < .fetch-list
if [ "$bad" -eq 0 ]; then echo "$(date '+%F %T') ALL FILES VERIFIED"; exit 0; fi
echo "$(date '+%F %T') $bad files still incomplete - re-run this script"; exit 1
