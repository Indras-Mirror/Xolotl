# Persistent exact-prefix cache

The native serving engines can keep reusable prompt state in a shared disk directory. The feature is off unless `STRATA_PREFIX_CACHE_DIR` is set. Model loading still happens after a process restart.

For latency-sensitive serving, `STRATA_PREFIX_CACHE_NONBLOCKING_LOOKUP=1` makes a busy shared-directory lease an optional cache miss instead of waiting behind a background writer. `STRATA_PREFIX_CACHE_MIN_RESTORE_CELLS=512` additionally skips disk records that improve on the available RAM prefix by fewer than 512 native key cells. Short requests are skipped before taking the lease. Cells are adapter-specific (DeepSeek uses two cells per token); this is not a token limit. Encoded-image auxiliary records are exempt from the minimum gain. Both policies default off, preserve exact-prefix validation and may trade a disk hit for fresh computation. They do not bound capture time or contention outside the disk lookup.

Flash also supports `STRATA_FLASH_PREFIX_MIN_CAPTURE_TOKENS=512`. It skips disk snapshot preparation for shorter token histories before joining checkpoint work, reserving buffers or copying native state. Those histories keep the existing RAM cache behavior. The default is zero (no additional threshold); malformed explicit values disable Flash disk capture but do not disable compatible record lookup. Longer captures can still copy substantial state synchronously.

Use a private local directory owned by the serving user (permissions `0700` on Linux). Each directory is one quota domain across model families and processes. Processes sharing it must use the same disk limit, free-space reserve and entry limit; a conflicting policy disables the tier for that process. A new directory records that policy. Use a different empty directory when changing the policy.

```sh
export STRATA_PREFIX_CACHE_DIR="$HOME/.cache/strata/prefix"
export STRATA_PREFIX_CACHE_BYTES=68719476736
export STRATA_PREFIX_CACHE_MIN_FREE_BYTES=1073741824
export STRATA_PREFIX_CACHE_QUEUE_BYTES=17179869184
```

The example allows 64 GiB on disk, retains 1 GiB free and allows up to 16 GiB of temporary immutable snapshots, queued writes and vector restore buffers in one process. A 32 GiB disk budget is `34359738368`. The default disk limit is 64 GiB, reserve 1 GiB and temporary budget 2 GiB. Explicit limits must be positive decimal bytes. An invalid value disables the tier. Large snapshots that cannot fit are skipped; they do not expand the limit. MiniMax state can exceed 5 GiB at 40K tokens, so the default temporary budget may be too small for that workload.

The temporary budget is separate from existing hot RAM caches. Existing RAM limits remain unchanged. Configure the family's existing RAM cache explicitly when choosing a 4–16 GiB hot budget. MiniMax also accepts `STRATA_PREFIX_CACHE_RAM_BYTES` for an additional bounded hot snapshot LRU; its default is zero. Disk persistence does not silently reduce a previously configured RAM cache.

## Reuse and storage

A lookup compares exact token and image identities and chooses the longest indexed restorable checkpoint in the compatible model namespace. On restart it reads bounded catalogue metadata, then reads only the selected payload. It does not preload the disk tier. The adapters retain full native state: Qwen recurrent and convolution state, attention/MTP KV, checkpoints and hidden seeds; DeepSeek compressed/ring/compressor state; MiniMax attention and index KV. This is exact native checkpoint reuse, with no approximate KV splicing or routing changes.

The namespace binds the complete resolved model file inventory, file size, device/inode and nanosecond modification/change timestamps, the running executable and loaded shared-library inventory, backend and computation settings. This is a conservative local identity, not a cryptographic digest of every weight byte. Inputs must remain immutable while the engine is running. Moving or replacing a file or executable can cause a safe miss even when its contents are unchanged.

Only successfully completed requests publish reusable snapshots. Background workers receive immutable host data, never a live device pointer. Temporary snapshots count against a bounded reservation before capture. Capture itself can copy gigabytes synchronously; measure full response and next-request admission latency as well as TTFT. Disk writing is asynchronous; host capture is not free. Qwen3.6 suppresses repeated captures of the same exact checkpoint while a write is queued or its committed files remain unchanged. This bounded process-local metadata check retains all checkpoint positions; evicted, changed or rejected entries can be captured again. It checks known file metadata before copying host state, without hashing payloads on the request thread. The first capture and newly encountered checkpoints still copy their complete state.

DeepSeek's `enc_skip` mode uses an additional explicit replay checkpoint format. It stores source KV and compressor state at a natural completed prefill chunk and requires at least the engine's full dependency-reach window of fresh suffix tokens. The unchanged native prefill reconstructs skipped late-layer state before any output. These records contain no generation logits and cannot be used for an exact-prompt regeneration. Full prompt snapshots retain their logits and DSpark state.

Payloads and metadata have corruption checks. Data is flushed before an atomic catalogue publication. Incomplete or corrupt entries are misses. A directory lease pins a selected file through validation and adapter reading; eviction cannot remove it. The global directory lock serializes file I/O across processes. LRU eviction enforces logical file bytes, available space and a bounded catalogue. It can discard uncommitted files left by a crashed writer. Use only a dedicated cache directory; do not put unrelated files there.

The native completion counters continue reporting reused and newly computed tokens. Family logs report actual restore/capture times and bytes; the common store exposes cumulative hit/miss, write/drop/error, byte and I/O-time counters. Prompt text is not logged by this feature.

For Qwen3.6 and DeepSeek capture diagnostics, set `STRATA_PREFIX_CAPTURE_TRACE=1`. The opt-in lifecycle records contain only engine/source, PID, slot, capture instance, monotonic timestamp, event and request epoch (when available; otherwise `-1`), with no prompts or payloads. Events identify capture begin, observed cancellation, abort/discard, and admission to the write queue or request-local pending list. They observe request-owned capture, not the asynchronous disk writer. In Qwen3.6 parallel serving, internal DONE can precede capture; the actual BDONE is deferred until capture and slot cleanup finish. Pair the capture instance with the native slot/request owner and slot-release event when qualifying cancellation; these records do not establish a strict latency bound or resource-release counters.

## Current limits and qualification

The initial adapters target native Linux single-stage serving, including the existing Qwen/DeepSeek batching paths. Layer-split, peer and multi-device paths retain their existing behavior and have no persistent tier where the adapter cannot represent complete native state. Storage is disabled outside Linux in this initial implementation. Builds and hardware execution for other backends must be reported separately; a host codec test does not establish GPU correctness.

Flash MTP batch configurations without a private slot draft retain a bounded complete main prompt checkpoint before admission and publish it only on successful completion. This reuses the exact prompt checkpoint; it does not claim to persist a generated suffix whose draft state the slot does not own.

For keyed image persistence, the frontend must start and identify an eager encoder before starting the native engine. Its selected helper, projector, model, effective options (including token limits and CPU/GPU selection), loaded implementation and frontend normalization inventory bind the native namespace. Both UUID and byte-based picture keys also include the actual selected pipeline before `HAVE`; a later fallback invalidates picture memos before checking native rows. Same-pipeline restarts can reuse disk state; changed pipelines miss safely.

A lazy or elastic encoder that has not started is unbound. Qwen3.6 does not persist or restore its image rows or image-containing KV; Flash and DeepSeek likewise skip image-containing disk checkpoints from an unbound keyed producer. Text-only disk caching and the existing RAM/image path remain available. MiniMax already verifies actual F32 image rows before restoring a checkpoint. Direct native callers that use keyed embeddings must provide a correct `STRATA_PREFIX_CACHE_VISION_ID` binding (`frontend-vision-pipeline-v1:` plus the selected pipeline's 64 hexadecimal identity); this is a producer contract, not authentication. Inputs must remain immutable while the processes are running. Default-off image behavior is unchanged.

A full frontend restart still forgets its public UUID-to-image registry. An unknown UUID receives the existing cache-miss response; resend image bytes before testing prefix reuse. Persisted native image state is not a persistent picture gallery. DeepSeek and MiniMax retain their existing bytes-based image protocol. Native Qwen3.6 image-row entries can be lazily restored when the frontend supplies a known identity.

For qualification, compare a clean computation, warm RAM and the first request after a real process restart. Use unchanged synthetic 20K/40K prefixes with different suffixes across fresh sessions, and retain actual prompt/cached/new token counts. Record native PID/start identity, active concurrency, restore/capture I/O times, TTFT, complete short-response latency and decode throughput. Also test A/B/A, cancellation during prefill/decode/I/O, image miss/refill, sleep/wake, model/config mismatch, truncation, interrupted writes, disk exhaustion and shared quota/LRU. Do not count an identical whole-prompt replay alone as evidence of changed-tail shared-prefix reuse.

### Optional real storage-failure host check

`persistent_prefix_storage_fault` is a CPU-only opt-in test of the real shared Store. Without `STRATA_PREFIX_FAULT_DIR` it exits 77 and CTest marks it skipped. It requires a dedicated, initially empty 8 MiB tmpfs owned by the unprivileged test user; it refuses ordinary filesystems and tmpfs volumes larger than 16 MiB. Never point it at a shared cache or fill a data filesystem.

From the repository root, compile outside the mounted directory (the mount is deliberately `noexec`):

```sh
HOST_BUILD=$(mktemp -d)
FAULT_MOUNT=$(mktemp -d)
c++ -std=c++17 -O1 -pthread -Iinclude tests/persistent_prefix_storage_fault.cpp \
  -o "$HOST_BUILD/persistent_prefix_storage_fault"
sudo mount -t tmpfs -o size=8M,uid=$(id -u),gid=$(id -g),mode=0700,nosuid,nodev,noexec \
  tmpfs "$FAULT_MOUNT"
STRATA_PREFIX_FAULT_DIR="$FAULT_MOUNT" "$HOST_BUILD/persistent_prefix_storage_fault"
# After the test and its child have exited:
sudo umount "$FAULT_MOUNT"
```

The same executable is registered with CTest when `STRATA_BUILD_CONVERSATION_TESTS=ON`. Retain the log/binary and temporary parent directories as needed; no cleanup of unrelated files is required.

The first case consumes free space with its own pressure file inside the admitted writer callback, then observes actual `write(2)` errno `ENOSPC` while writing the admitted payload. It checks no visible checkpoint, released queue budget and successful small-write and fresh-Store lookup after truncating only its pressure file. This exercises space exhaustion after the Store's preflight, rather than preflight refusal.

The second case kills only its own child writer at a pipe barrier after payload `fsync` but before its callback returns for catalogue publication. It checks that the interrupted entry is a miss, an earlier committed checkpoint survives, and a subsequent write plus fresh-Store lookup recovers. It does not exercise a crash after catalogue rename or every filesystem's durability semantics.
