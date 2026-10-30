# Qwen3.6 validation scope

The measurements below describe a frozen development candidate before its
public rebase. They are not a fresh measurement of the rebased PR, not
cross-hardware guarantees and not assertions embedded in the example tools.
Tests after the rebase are listed separately in the pull request.

## Hardware and model

One RTX 5090, CUDA 13.1, GCC 14 and an EPYC 9654 with 624 GB host RAM;
Qwen3.6-35B-A3B UD-Q4_K_XL, matching vision projector and MTP=3. The context was
configured as 262144. Five inserted-fact cases used 255998–256002 actual prompt
tokens, and five other cases used approximately 202K tokens. All ten retrieved
the requested fact. This does not claim that a request contained all 262144
prompt tokens or that every supported quant fits the same VRAM budget.

## Functional checks

- Native cancellation/state checks: 96 qualified generation/prefill interruptions
  at 40K/200K, with correct answers from the next caller. Eight cancelled-prefill
  replays returned the expected answer.
- Automatic and explicit rollback: six distinct stable prefixes of 30–50K,
  changed within one final user message. All 60 responses were correct;
  18 cached-versus-fresh output token-ID comparisons matched. Pressure cases
  evicted entries without corrupting the retained callers.
- A 200K prompt followed by 650 committed generated cells was interrupted.
  Fast parking discarded 657 cells, of which only seven were prompt cells.
  Original-prompt replay read those seven again; replay including the old
  generated output reread 653. An invalid intervening GEN preserved metadata.
- Image upload, UUID-only reference, exact missing-image 400, grammar aliases,
  JSON schemas and final streaming usage chunks were checked through HTTP.
- A synthetic one- and six-caller workload with roughly 40K history, 50 retained
  pictures, a new picture and a replaced suffix returned 252 valid answers.
  In 231 post-warmup requests at least 30302 tokens were reused. Reordering or
  deleting early pictures still requires reading the changed visual suffix;
  this can be much larger than the newly added text.
- A prior segmentation comparison kept 384 greedy output tokens identical at
  4K/40K/200K. Later cancellation diagnostics changed logging and queue
  publication, not CUDA arithmetic or synchronization.

## Observed cancellation latency

The final prequeued HTTP check used exactly 20 trials for each context/phase,
with caller B sent and observed in the FIFO before closing caller A. All 80
trials qualified and B answered correctly. No slow trials were discarded.

| Context / phase | STOP to DONE p95 | HTTP close to old-state park p95 | Maximum old-state park |
| --- | ---: | ---: | ---: |
| 40K generation | 1.150 ms | 2.591 ms | 11.839 ms |
| 40K prefill | 2.167 ms | 3.214 ms | 3.515 ms |
| 200K generation | 1.699 ms | 4.794 ms | 13.316 ms |
| 200K prefill | 1.050 ms | 3.803 ms | 3.959 ms |

These are different boundaries. DONE includes the old request's pending CUDA
writes; parking also includes saving its state before restoring B. B's prompt
start additionally includes B's restore and protocol work. Its 40K-generation
p95 was **12.681 ms**, with a maximum of 13.175 ms. The maximum observed next
prompt latency at 200K generation was 14.663 ms. Thus the p95 old-work-release
target passed, but a universal 10 ms deadline and the full 40K handoff target
**did not pass**. The next caller's first token also includes new inference.

A diagnostic run measured a 9.961 ms blocking stderr write after GPU work had
already stopped. That synchronous diagnostic was removed from the abort path.
Verbose handoff logs are off by default and GEN is published before optional
logging outside its queue mutex. Other OS, driver and file-I/O delays remain
possible; no hard real-time property is claimed.

## Grammar checks and cost

Historical checks covered 1300 valid responses across GBNF application formats,
a binary choice and a JSON object, with MTP=0/3. Twenty HTTP cases included
identical duplicate grammar fields, schema aliases, JPEG input and UUID reuse.
The CPU real-vocabulary checker covered independent clones, delayed activation,
EOS, full-row fallback, cached masks and candidate permutation.

Warm constrained decode overhead was 0.15–5.23% in matched ABBA requests. The
first sampled grammar request added **152–232 ms** to build full-vocabulary
masks. Warm cost does not bound cold requests or newly encountered states.
An initial 1000-per-mode distribution comparison had a marginal difference at
one position (Holm p=0.04864); one independent 5000-per-mode replication did not
reproduce it. This empirical check does not prove equality of every joint
output distribution. The implementation's deterministic-draft acceptance and
rejection use the target probabilities; no top-k approximation is accepted
when the exact cutoff cannot be established.

The public tools generate neutral example prompts and grammars. Their samples
are new measurements, not byte-for-byte replicas of the historical application
inputs. Keep all output rows, counts, seeds, warmups, cold requests, failures
and binary/grammar hashes when reporting a new run.

## Attention precision

The q36 attention path stores KV and prompt probabilities in FP16, uses grouped
query decode and a CUDA flash-prefill kernel when supported. The flash path can
be disabled with `STRATA_Q36_FA=0`. FP16 rounding and changed reduction order
can change logits and generated tokens; the functional and retrieval checks
above are not a comprehensive quality evaluation. The numerical checker
compares synthetic attention inputs, rather than downstream model quality.
A historical teacher-forced comparison against llama.cpp covered six text cases
and 384 continuation steps: mean KL was 0.00381458 and top-1 agreement was
97.3958%. The preceding engine measured 0.00413182 and the same aggregate
top-1 agreement on those cases. These small text checks do not establish general
quality equivalence or quantify an isolated FP16/GQA speed improvement.

## Not covered

The historical hardware runs did not validate 122B models, other quants, HIP,
SYCL or Windows. New GPU runs require a separate hardware allocation and should
not overlap a build. The serving feature does not include generic engine sleep;
that change is proposed separately. No production application or deployment
configuration is part of these tests.

## Native batching and shared host-cache validation (2026-10-30)

A later CUDA candidate adds native batching and a host prefix pool shared across
execution owners. Hardware evidence below uses the35B UD-Q4_K_XL model, MTP3,
context262144, two native owners and one RTX5090. Four/six HTTP callers queue
onto those two owners; these results do not claim four-owner grouped forwards.
The grouped-forward trace and per-owner progress establish actual two-owner
native overlap. Request grammar/code-update checks, repeated seeded grammar
requests, changed-tail versus fresh control, busy-sleep409 and SSE checks passed.
A separate unconstrained check passed48 observable-response comparisons across
greedy and sampled(.9/.9) profiles with64/128-token budgets: sequential repeats
and concurrent/reversed-order requests agreed in exact text, finish reason and
core token usage. Thinking was disabled and request bytes/seeds were reused.
This does not compare internal token IDs or logits, establish unseeded parity,
or prove internal RNG-state equivalence. Prefill-cancel timing was checked separately; decode timing is not used as
a substitute for it.

The synthetic robot workload used~39935 prompt tokens and12 requests per caller,
including image references, quiet replies, changed state and cancellation. All
156 requests across1/2/4/6 callers completed without HTTP errors. Warm reusable
prefixes reported37901 cached cells; native metadata also showed68 restores of
at least37000 cells whose destination execution owner differed from the
published prefix's owner. This tests global cache reuse rather than only slot
affinity. The observed HTTP timings include admission, scheduling, vision and
CPU grammar work:

| HTTP callers / native owners | Requests | TTFT p50 / p90 (ms) | Quiet completion p50 / p90 (ms) |
| --- | ---: | ---: | ---: |
|1 /2|12|504 /507|521 /564|
|2 /2|24|650 /976|657 /1086|
|4 /2|48|1942 /4802|2097 /4969|
|6 /2|72|3068 /6673|2940 /5807|

These are synthetic workload observations, not a general latency guarantee or
an isolated batching speedup. The earlier frozen single-request reference had a
lower one-caller median TTFT (~392ms); the new path does not establish a
single-caller speed improvement. Cold constrained-output masks can dominate CPU
time, while exact repeated grammar states are much cheaper. Compare equivalent
cache/grammar warmth when assessing throughput.

A separate20-sample decode-cancellation test used~39017-token victim/peer
prompts. Raw client socket close to the native `slot_released` event measured
p50=2.256ms, p95=2.692ms, max=2.791ms. Each valid sample matched the victim's
request epoch, cancel finish and release event, and required peer progress after
release. The endpoint follows parking/reset/fencing; the earlier finish event
was not used as GPU release. This is a measured decode workload, not a universal
10ms deadline.

A separate fresh-prefix prefill-cancellation test collected10 valid samples at
nominal40000 tokens and10 at nominal200000 tokens (small nonce/chat-template
overhead is recorded by the harness). Socket close to actualslot_released
measured p95/max=2.308/2.308ms and2.464/2.464ms respectively. The harness observed
native admission/prefill markers and no HTTP content; matching native cancel
records generated zero tokens. It does not identify the precise CUDA instruction
running at close. All valid slow observations were retained. The measured p95
below10ms goal passed these prefill and decode scenarios; this is not a universal
real-time guarantee. Raw protocol clients that queue BGEN behind another held
admission gate remain outside that latency claim; the Python frontend serializes
admission control and cancels waiting requests without reassigning their output.

The same shared-cache candidate answered five near-limit needle requests with
~256000 input tokens correctly, then retained256000 cached cells across
same-process checkpoint/wake. A separate checkpoint test restored four distinct
conversations and a known image UUID. Generic checkpoint support in these runs
comes from the combined integration with PR75; these results must not be read
as standalone PR76 implementing all sleep backends.

Portable tests cover shared snapshot publication/pins, record leases, blocked
copy-on-write and eviction, six logical callers migrating across execution
owners, and the physical KV budget/copy planner. These tests do not execute CUDA
math. The standalone PR76 extraction builds with CUDA grammar ON and OFF, including
flash-check targets. The real-vocabulary grammar checker passed20 exact and20
fallback cases. Six portable CTest entries passed (Hub, concurrent Hub, VMM,
offline HTTP checker, protocol adapter and36 helper tiling cases);49 standalone
serving Python tests also passed. The flash-check executables were compiled but
not run on a GPU in this extraction check. The unconstrained live-response
checker scope is described above; HIP, SYCL, Windows,122B and other quants remain
unqualified for native batching.
