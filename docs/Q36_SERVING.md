# Qwen3.6 serving, vision and reusable conversations

The qwen35moe engine supports text and image input, sampled MTP, constrained
output and continuation of a conversation whose final message changes. It
retains the longest unchanged prefix. Attention KV is truncated at that prefix;
DeltaNet recurrence is restored from a matching state snapshot. Arbitrary
recurrence states cannot be reconstructed by truncating attention KV alone.

These features are in `strata-q36`, also selected by `strata` for qwen35moe.
The tested model is Qwen3.6-35B-A3B UD-Q4_K_XL on one RTX 5090. Other quants,
122B models, HIP, SYCL and Windows have not received the same hardware checks.
The CUDA build uses the normal build instructions. For native GBNF, also set
`-DSTRATA_Q36_GRAMMAR=ON` with a complete pinned llama.cpp tree; see
[constrained output](Q36_GRAMMAR.md).

## Example configuration

Supply matching model, tokenizer and projector files. Paths here are placeholders.
The image encoder is a separate `strata-vision` executable, built as described in
the main vision documentation.

```json
{
  "exe": "/path/to/build/strata-q36",
  "args": [
    "--native", "/path/to/model.gguf",
    "--max-context", "262144", "--mtp", "3", "--vision",
    "--snapshots", "128", "--snapshot-every", "512",
    "--park-mib", "16384", "--image-cache-mib", "2048"
  ],
  "tokenizer": "/path/to/tokenizer",
  "model_name": "qwen3.6",
  "gpu": 0,
  "vision": {
    "exe": "/path/to/strata-vision",
    "model": "/path/to/model.gguf",
    "mmproj": "/path/to/mmproj.gguf",
    "gpu": true,
    "max_tokens": 1024
  }
}
```

```bash
python -m serve.server --engine strata --config q36.json \
  --host 127.0.0.1 --port 8080
```

Do not copy the large host-memory budgets blindly. On the tested 35B model a
DeltaNet snapshot is about 63 MiB: 32 snapshots use about 2 GiB of pinned RAM,
and 128 use about 8 GiB. `--park-mib` is a separate host KV budget. The shared
pool evicts an entire least-recently-used parked prefix together with its state
references. Smaller budgets can require more prefill. `--snapshots 0` disables
saved recurrence states. Both snapshots and host parking default to zero; the
large budgets above are explicit opt-in settings.

`--snapshot-every` controls periodic snapshots near the prompt's end; snapshots
also cover recent message boundaries and a learned divergence point. Prefix
identity includes image keys and their row positions. Automatic reuse therefore
works when an application keeps two messages and replaces only the suffix of
the final user message, without adding a new message boundary each turn.

Applications can request up to four explicit snapshots using
`prompt_cache_breakpoint: {"mode": "explicit"}` on a content part/message, or
Anthropic-style `cache_control: {"type": "ephemeral"}`. These are hints to retain
a state at that boundary. Actual reuse is reported by
`usage.prompt_tokens_details.cached_tokens`; a marker is not a promise that a
state survives a later memory-pressure eviction.

## Pictures by UUID

Upload bytes once with a UUID:

```json
{"type":"image_url","uuid":"frame-001","image_url":{"url":"data:image/png;base64,..."}}
```

A later request can use:

```json
{"type":"image_url","uuid":"frame-001","image_url":null}
```

An unknown or evicted UUID without data returns HTTP 400 with
`Cache miss for image at index 0 but data is not provided.` Resend the request
with full bytes in that case. Treat UUIDs as immutable image identities: do not
reuse the same UUID for different images. Known UUID data can be ignored as in
vLLM's multimodal UUID contract.

The server asks an `img_refs=1` engine which image encodings it holds (`HAVE`)
and sends header-only `SVR1` records for those pictures. This avoids re-encoding
and resending held rows. If rollback needs rows no longer held, `NEED_IMAGES`
causes one refill attempt. Both encoding and identity caches have bounded
storage; UUID reuse is not persistent storage across restarts or all evictions.
The application remains responsible for deciding when to compact its context.

## Sampling and cancellation

MTP remains available with temperature sampling and grammar constraints. A
speculative draft is deterministic; acceptance uses the target probability of
that draft, and rejection samples the target with that draft excluded. This
preserves the requested target distribution. Greedy is a separate exact-match
path. See the grammar document for penalties and full-vocabulary fallbacks.

Disconnecting an HTTP stream sends STOP. Prefill is transactional: interruption
inside a chunk restores a consistent recurrence state and leaves the public
position at the last committed boundary. Verify work is checked between graph
groups before committing recurrence. An interrupted request must not corrupt
another caller or lose the already committed prefix.

After cancellation, parking may use an existing owned snapshot if at most 512
committed prompt cells would need to be reread; generated answer cells do not consume that prompt-replay bound. This is cache trimming, not output truncation. A later
request that includes that generated text can prefill it again. A cold prefix
without an eligible snapshot uses the normal copy path instead.

Diagnostic timing is opt-in with `STRATA_Q36_PHASE_TRACE` or
`STRATA_Q36_CANCEL_TRACE`. Normal request summaries remain available. See the
[validation scope](Q36_GRAMMAR_VALIDATION.md) for observed latency tails; no
hard real-time deadline is promised.

## Checks

The generic HTTP check connects to an already reserved server; it never starts
an engine or a workload manager:

```bash
python tools/qwen35/check_serving.py --url http://127.0.0.1:8080 \
  --model qwen3.6 --require-cache --vision --cancel --out serving-check.json
```

It uses synthetic stable prefixes, a replaced suffix, generated red image data,
UUID references, an unknown-UUID retry and an optional stream disconnect followed
by another caller. Its HTTP timings include the next request's own work; they
are not measurements of old GPU-work release. The native grammar benchmark,
CPU vocabulary checker and flash-attention numerical checker are documented in
[Q36_GRAMMAR.md](Q36_GRAMMAR.md).

## Native continuous batching (CUDA)

`--batch 2` through `--batch 4` enables native continuous batching in serve
mode. A scheduler groups independently ready decode/verify rows for shared
projections, routing and expert work. Each request keeps its own DeltaNet
recurrence, attention KV, rotary positions, grammar, sampling RNG and MTP
proposal/verification state. HTTP concurrency alone is not evidence that a
particular forward grouped multiple owners. `--batch 1` is the existing
single-request path and does not use the shared host-cache implementation.

Set the Python service's `parallel` to the desired native slot count; the service
passes the corresponding `--batch` argument. For example, replace the cache budgets above with these native
arguments, add the batch settings, and select `"parallel": 2`:

```json
["--batch-kv-mib", "5632", "--batch-prefill-chunk", "256",
 "--snapshots", "128", "--snapshot-every", "512", "--park-mib", "24576"]
```

These are example opt-in budgets, not a requirement or a guarantee that another
model/quant will fit. Native batch currently requires CUDA virtual-memory
management. HIP explicitly rejects batch greater than one; other backends and
Windows have not been qualified for this path. Hybrid CPU expert execution is
not supported by the grouped forward. The startup physical-memory check refuses
READY when weights, vision, scratch and the requested KV budget cannot coexist.
MTP is preserved; memory admission does not disable it or shorten context.

| Option | Meaning |
| --- | --- |
| `--batch 1..4` | Maximum native execution owners; greater than one requires serve mode. |
| `--batch-kv-mib N` | Aggregate physical device KV budget, default6144MiB. Full context strides are reserved virtually; only admitted prefixes/output headroom are mapped. |
| `--batch-prefill-chunk N` | Worker prefill chunk:0 selects256 for two owners or64 for three/four. Explicit values8..2048 are accepted; larger scratch may fail memory preflight. |
| `--snapshots N` | Total shared recurrence snapshot slots in batch mode, not per execution owner. Default0. |
| `--park-mib N` | Total shared pinned host KV budget in batch mode, not per owner. Default0. |
| `--image-cache-mib N` | Shared host image-row cache budget; separate from KV and recurrence snapshots. |

`--max-context 262144` is a per-request limit. It does not promise four fully
resident262144-token contexts at once. On the tested35B geometry one full
attention KV context is about5.5GiB; four would require about22GiB before
weights, vision and scratch. Requests whose rounded physical KV reservations
would exceed the aggregate budget wait for admission. Idle mappings remain
charged and may be reclaimed only after that owner's work is fenced and its
prefix parked. The limit is never silently truncated to fit concurrency.

The shared host cache is independent of execution slots. A completed prefix is
published only after its host copies finish. A later request can fork that
immutable prefix on any free owner; concurrent forks hold leases and get private
copies of host pages before modifying them. Active leases prevent whole-record
LRU eviction. Recurrence snapshot reservations are unavailable until DMA
completion, and pinned snapshots cannot be overwritten. Metadata locks do not
cover GPU waits or full host-page copies. Image keys and embedding row positions
remain part of prefix identity. Twelve useful state anchors per retained prefix
and a bounded16-record LRU limit metadata retention; smaller configured pools
can evict prefixes and require prefill again.

For the tested model128 recurrence snapshots require about8GiB pinned RAM,
plus the separately selected24GiB host KV example above. One full262144-token
KV prefix needs about5.5GiB of that shared host pool. Budget for retained caller
histories and forks, not just the number of active GPU slots. Replacing a suffix
can reuse an earlier matching state; snapshots do not contain logits, so an
exact prompt may need a small final replay. Mutable GPU state is never shared
between callers.

The native advertises `batch_pinned=1`: each request stays on its BGEN owner
until BDONE so grammar and RNG do not restart during foreground/background
migration. Image HAVE queries serialize with admission control; waiting for a
long admission is separate from the native reply timeout. A missing native image
reference can be refilled once before any token is emitted. Refill releases the
control lock before entering the image-preparation FIFO and rejects process or
generation changes. Unknown UUIDs still return the existing400 cache-miss error.
Disconnect cancellation targets the registered owner directly, including before
BADM; slot reuse waits for actual GPU fencing and parking.
A raw protocol client can queue BGEN behind another held admission gate; a
cancelled queued command still waits for that gate before its final control
reply. This raw-protocol queue case has no bounded-release latency claim. The
Python frontend owns the admission lock and cancels its waiting request before
sending a new command.

`STRATA_Q36_BATCH_TRACE=1` emits metadata-only native JSONL events: parsed request
begin, completed grouped forward with owner epochs, token progress, finish/MTP
counters, cross-owner cache_restore and slot_released. The release event follows
parking and fencing; the earlier finish event is not proof of old-work release.
No prompt text is included. `STRATA_Q36_BATCH_WAIT_US` bounds the rendezvous
wait (0..5000us, default150us); measure latency and grouped-forward frequency
before increasing it. Cold grammar-mask construction can dominate CPU time even
when a completed GPU forward is fast. Compare cold grammar, warm grammar and
ordinary generation separately.

Generic process checkpoint/deep sleep is proposed separately in PR75. Combined
integration evidence must identify that dependency; the batching change does not
implement another engine's sleep protocol. No universal cancellation deadline or
four-owner throughput improvement is promised by configuration alone.

## Portable cache checks

The host-cache and VMM planners require neither a model nor a CUDA context:

```bash
cmake -S . -B build-cpu -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF
cmake --build build-cpu --target test_q36_cache_hub test_q36_cache_hub_concurrency test_q36_kv_vmm \
  q36_adapter_protocol_mock q36_batch_helpers_mock
ctest --test-dir build-cpu -R 'test_q36_|q36_batch_isolation_offline' --output-on-failure
python -m unittest serve.test_parallel serve.test_have_admission serve.test_batch_image_refill
```

These checks exercise ownership, publication, copy-on-write, eviction, admission
control and protocol routing. They do not establish CUDA numerical equivalence,
request-level token parity, GPU overlap, memory fit or checkpoint restoration.
Those require separate hardware validation with actual per-request seeds and
per-owner progress.

For a seeded observable-output comparison, reserve and start separate batch1
and batch2 services with identical model, tokenizer and sampling options, then
run (the script does not start engines):

```bash
python tools/qwen35/check_batch_isolation.py \
  --serial-url http://127.0.0.1:8080 --parallel-url http://127.0.0.1:8081 \
  --model qwen3.6 --callers 4 --tokens 64 128 --profiles greedy sampled \
  --out batch-isolation --run
```

Without `--run` it makes no requests or files. `--url URL` uses the same already
running service for both serial and concurrent arms. The checker disables
thinking, repeats each sequential request twice, then repeats concurrent rounds
with reversed order using the same request bytes and seeds. It checks strict SSE
termination, response text, finish reasons and core token usage, and reports the
first difference in `report.json`. Request/whole-run deadlines default to30/600s.
Exact observable agreement does not prove internal RNG/logit equivalence or
native GPU overlap. It does not compare internal token IDs, and unseeded output
parity is not established by this check.

## Optional resident prefix shortcut

`--resident-prefix-reuse` is an opt-in CUDA native-batch optimization (requires
serve mode and2..4 owners). The default restore path is unchanged. It retains a
CPU-only description of each Engine's attention KV prefix; it allocates no extra
GPU cache and changes neither context, MTP, sampling nor host-cache ownership.
After admission, an exact cell prefix match on the same Engine/mapping generation
can skip KV restore only while the selected immutable host pages also match the
certified allocation cookies and content-write versions. Matching tokens alone
is insufficient: recomputing a prefix with different prefill geometry can produce
different KV bytes. The shortcut stops at the first conflicting page and restores
the remainder normally. A different
owner still restores its missing KV normally. The selected DeltaNet snapshot is
always restored; resident attention KV does not reconstruct recurrence.

A certificate is created only after the selected RAM prefix has been restored
(or matched an existing byte-proven certificate). New main+MTP teacher-prefill
rows extend it only after their KV has been mirrored to RAM and the existing
copy completion fence has finished. Every host-page allocation/recycle gets a
new cookie; every partial D2H or COW write gets a new version before copying and
becomes readable only after successful completion. No extra pages are pinned.
The existing first-sampling completion fence also bounds final publication. The single-token
fast path that does not teacher-prefill MTP cannot certify that extra cell.
Every main/MTP attention write clamps the valid prefix before it is enqueued;
partial prefill, rollback or rejected speculation never promotes the bound.
Decode cancellation can keep an unchanged prompt prefix because its writes are
in the suffix. Mapping trim invalidates the generation before unmap, including
failed unmaps; suffix growth alone preserves the old prefix. Identity includes
image-key/row cell values, not just token IDs or the original publishing slot.

In this opt-in mode the idle checkpoint pause/resume protocol is advertised.
`CHECKPOINT_PAUSE` refuses busy owners and invalidates their CPU shortcuts before
checkpointing. Host records remain available, so the first request after resume
performs the normal KV restore. The batch path has no executable graphs; resume
reports elapsed paused time and zero graph executions. Generic checkpoint/API
integration still comes from PR75; no other engine's sleep code is included.

With `STRATA_Q36_BATCH_TRACE` enabled, a `resident_prefix` event reports selected
reuse, skipped/copied cells and bytes, mapping generation and certified end.
These counters describe KV restore only. A speedup and byte-identical default
behavior require separate same-day GPU validation; portable tests establish
metadata/lifecycle invariants, not model numerics or a latency deadline.

`--resident-prefix-verify` requires the reuse flag and is a diagnostic only. It
compares the proposed skipped GPU bytes with the selected RAM pages, logs counts
and hashes without raw data, and falls back to a full restore on any mismatch.
Its synchronous staging copies add substantial overhead; it is unsuitable for
latency comparisons. Successful qualification requires zero mismatches, not merely
correct output after diagnostic fallback.
