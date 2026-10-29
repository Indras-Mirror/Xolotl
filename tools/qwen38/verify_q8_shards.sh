#!/bin/bash
# Verify the Swift 1.5 Q8_0 download against the SHA256 the HF repo publishes (LFS oids).
# ~188 GB of reads; expect 2-4 min on NVME1TB. Exit 0 = every shard matches.
set -u
D=/media/mal/NVME1TB/Models/Swift1.5-Q8_0/Q8_0
declare -A WANT=(
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf]=faaae3162c70d3970d18c2e74702adf090f4c71801f59ff7e7753fd207d29a2e
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00002-of-00005.gguf]=8aeb4c8e478809a91b2094f61ba276265d10425d036a3a9c5c9959ea650858d4
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00003-of-00005.gguf]=85f477cd99f6070315d4dcc80fd1956dfe94a0795ee9885bdccce27eac1d15ba
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00004-of-00005.gguf]=297389c30cdffe92a6970a59f767bf835b906fa68914ee361493e150bfdc9a2d
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00005-of-00005.gguf]=cad9c5a60edc7697b28da03b38d1f41aeeca9ae809d242919c540db7c5de80f8
)
declare -A SIZE=(
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf]=699524416
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00002-of-00005.gguf]=54400261312
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00003-of-00005.gguf]=44905193504
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00004-of-00005.gguf]=44150177536
  [Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00005-of-00005.gguf]=44105612928
)
bad=0
for f in $(ls "$D"/*.gguf 2>/dev/null | sort); do
  b=$(basename "$f")
  want=${WANT[$b]:-}
  wsz=${SIZE[$b]:-}
  [ -z "$want" ] && { echo "UNEXPECTED FILE: $b"; bad=1; continue; }
  got=$(stat -c %s "$f")
  if [ "$got" != "$wsz" ]; then echo "SIZE MISMATCH $b: $got != $wsz"; bad=1; continue; fi
  h=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$h" = "$want" ]; then echo "OK   $b"; else echo "FAIL $b  got=$h want=$want"; bad=1; fi
done
[ "$bad" = 0 ] && echo "=== ALL SHARDS VERIFIED" || echo "=== VERIFICATION FAILED"
exit $bad
