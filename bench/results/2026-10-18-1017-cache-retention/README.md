# Uniform LRU cache retention for short batch continuations

RTX 3090 24 GiB, Ryzen 3900X, 32 GiB RAM, 970 EVO NVMe; CUDA 13 / sm86. Same 1017 Q2_K shards, F32 indexer sidecar, `--no-enc-skip`, 131072 context, 16 GiB host arena, STEP_MAX=96, temperature=0/seed=42. DSpark was disabled. Both arms use executable SHA256 a6d8c93dbea76f5c9512ec8635e0cdae46aa7d7d04263acb6463f4407a1ba9d1; only `STRATA_SERVE_KEEP_CACHE=1` and `STRATA_SERVE_KEEP_LRU=1` differ. The code is opt-in; defaults remain unchanged.

The batch path does not admit/reorder LRU experts. Retain its mapped prefix, evict only the tail needed by the calibrated batch budget, then refill the saved exact slot/layer/expert mapping. Mixed-size LRU caches remain refused. Round retained bytes down to an actual segment boundary; the old up-rounding left a 137-token continuation capped at 103 columns, forcing two complete expert sweeps. This patch gives it 164–171 columns, keeping one sweep. Core padded-tail/bounds fixes are a prerequisite.

Each arm starts with the same 4060-token coding request and returns the identical known 85-token solution, then appends three 137-token marker requests. Exact code and all three literal markers match. Each continuation exercises the intended 96 < new_tokens <= 256 batch branch.

| Request | Release/reseed | Retain LRU prefix | Physical engine reads, release / retain |
|---|---:|---:|---:|
| Initial coding | 105.505 s | 105.497 s | 171.070 / 169.661 GiB |
| Continuation A | 33.734 s | 25.910 s | 62.403 / 45.112 GiB |
| Continuation B | 30.879 s | 21.128 s | 56.856 / 36.028 GiB |
| Continuation C | 30.683 s | 21.259 s | 56.087 / 35.631 GiB |
| Continuation mean | 31.765 s | 22.766 s | |

Mean continuation latency falls 28.33%; initial latency is unchanged. Physical reads are `/proc/<engine>/io` deltas, not disk/partition sums. API cgroup swap=0 and all memory.events fields=0 in both arms. This is three continuations on one fixture, not a general 67K/128K or throughput claim.

Separate correctness validation enables DS41_CHECK_ARENA (excluded from timing). It returns identical code/markers, checks 384 refilled slots per continuation against the source file (1152 total, zero mismatches), and passes branch/resource gates. The first prototype failed with a native fill error; the padded physical-tail and live admission bugs were fixed before these results. Existing segmented/per-layer GPU tests pass; the new padded-tail regression fails before the core fix and passes after it.

CUDA build passed. HIP/SYCL builds and independent review are unavailable. Separate read-only self-review covered prefix/tail ownership, no batch admission, exact-slot refill, refusal/fallback and opt-in boundaries. Long-context repeated-steer and combined DSpark qualification remain separate gates; this feature is not enabled by this report alone.
