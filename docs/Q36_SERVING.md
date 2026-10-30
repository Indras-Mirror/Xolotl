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
