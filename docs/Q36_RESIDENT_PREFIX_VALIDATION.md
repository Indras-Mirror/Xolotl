# Resident KV prefix validation

`--resident-prefix-reuse` avoids restoring attention KV that an execution owner
already holds. It is opt-in. The selected host-page allocation cookies and
content versions must match the GPU certificate, in addition to the logical
prefix and mapping generation. A conflicting page ends the shortcut; the
remaining prefix is restored normally. DeltaNet state is always restored.

Token identity alone is insufficient. A previously recomputed interval can hold
a different numerical KV representation from the host record selected for
rollback. Page provenance preserves the existing restore behavior without
copying every unchanged page. Writes, recycling, COW, mapping trim and checkpoint
pause invalidate the appropriate certificates.

## Configuration and method

Measured on Linux with an RTX 5090 and the Qwen3.6-35B-A3B UD-Q4_K_XL model:
two native owners, sampled MTP3, a 262144-position per-request limit, 5632 MiB
aggregate attention KV, 128 snapshots and a 24576 MiB shared host pool. Neither
the cache budget nor the context limit changes with this option.

The same CUDA executable was restarted for OFF and ON runs. Inputs and sampling
parameters were paired. Measurements below exclude two warmup requests/cycles
and use eight measured samples. Medians average the middle pair; p90 uses nearest
rank. HTTP runs were local to the inference host, without client WAN/tunnel time.

The image scenario uses a neutral roughly 40K history, 50 previously seen frames,
one new frame and an illustrative replaceable state block. Its actual prompt
contains 40934 positions; 40621 are cached and 313 are read. Those positions include
visual cells. The tool scenario retains the offered schema while appending real
function calls/results and assistant replies. It uses a pure local test function;
its execution time is measured separately. Both use temperature 0.9, top_p 0.9,
seed 1030 and thinking disabled. Output limits are 700 for the image scenario and 128
for tool requests. These are full completions, not an 85% streaming-abort workload.

## Paired results

| Client-visible interval | OFF p50 / p90, ms | ON p50 / p90, ms |
|---|---:|---:|
| Image/state request to first content | 195.436 / 197.530 | 162.306 / 164.346 |
| Tool request to complete arguments | 236.560 / 255.911 | 202.130 / 220.795 |
| Tool result request to first content | 131.927 / 134.621 | 98.016 / 100.158 |

An early tool-name delta is not a native first-token measurement. In particular,
forced tool choices can emit a parser primer before engine generation. The
tool benchmark above uses `auto` for both requests and measures complete arguments.
Client result-to-next-send handoff was approximately 2 ms, separately from model
work. Percentiles of different stages must not be added.

In the image scenario, ON retained 39936 positions (858 MiB) and copied 685
positions (14.717 MiB), instead of restoring the full 872.717 MiB prefix. The
version conflict includes the entire previously replayed 173-position interval.
These are native restore-range counters, not a hardware PCIe trace. Recurrent
state transfers remain. Cached/read counts matched OFF; all ten image responses
and all twenty tool HTTP responses matched text, finish and full usage fields.

## Correctness checks

An independent debug run used `--resident-prefix-verify`: all 47 positive skip
decisions covered all 44 main/MTP layer, K/V and head planes, with zero different
bytes and zero fallback. All 54 requests were matched by native slot and epoch.
Serial 16/16, concurrent HTTP 8/8, image 10/10 and tool 20/20 responses matched their
OFF references. Verification copies add substantial overhead; debug timings are
not performance results and debug execution did not establish grouped GPU overlap.

A separate normal OFF/ON run checked semantic prefix edits and appends, shared
and distinct concurrent histories in reversed orders, native grouped execution,
decode and fresh long-prefill cancellation/recovery, a 250100-position five-needle
prompt, repeated reuse and same-process checkpoint sleep/wake. All regular
response pairs matched text, finish and core usage. Cancellation samples prove
the tested release boundary, not a latency percentile or universal deadline.

Nine portable CTest entries and sanitizer checks passed. Real host-pool tests
cover partial writes, pending/stale completion, page recycling, COW failure,
immutable leases and concurrent metadata access. CUDA qualification used two
owners and this model/quantization. Four-owner execution, 122B, HIP, SYCL, Windows
and other GPUs remain unqualified. Checkpoint HTTP integration is supplied
separately by the sleep support; the first request after checkpoint pause restores
normally, and a later request can establish a new resident certificate.
