# MiniMax-M3.1-Flash

Opt-in MiniMax engine and shim to the common Strata HTTP server, with a separate
image/video opt-in, developed in [issue #62](https://github.com/Indras-Mirror/Xolotl/issues/62).
See [build, run, measured results and reproduction commands](../../docs/MINIMAX_M31.md).

```sh
cmake -S . -B build-m31 -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON
cmake --build build-m31 --target m31_generate m31_checks -j 8
ctest --test-dir build-m31 --output-on-failure -R m31
```

`STRATA_M31=OFF` is the default. It adds no targets or build settings. The entry
point remains `tools/m31/m31_generate.cpp`, as in upstream's landing scaffold.
`m31_engine` holds the engine; `m31_generate` provides offline logit dumps and
`--serve`. Python code in `serve/` selects the M31 tokenizer/template/parser and
runs `serve/server.py`; it does not implement a second HTTP server.

## Engine protocol

```text
in : GEN <max_new> [key=value ...] <ids csv>
     BGEN <slot> <max_new> [key=value ...] <ids csv>
     GENI <max_new> [key=value ...] <SVE1-file> <ids csv>
     BGENI <slot> <max_new> [key=value ...] <SVE1-file> <ids csv>
     STOP
     SLEEP
     WAKE
     QUIT
out: INFO key=value ...                  (startup: ... sleep=0|1)
     READY <ctx> stop
     RESUME <reused>
     PP <read> <total> <ms> <tok/s>
     T <id>
     DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> ... <reused> ...
     SLEPT seconds=<s> kv_tokens=<n> kv_bytes=<b> dense_bytes=<b> experts=<n> conversation=0|1 ...
     INFO woke=1 wake_seconds=<s> ...    (then READY, or before a woken request's lines)
     ERR <text>
```

`GENI` and `BGENI` require `-DSTRATA_M31_VISION=ON`, which defaults to OFF.
Without it they return `ERR vision unavailable`. The embeddings file precedes
the token list, matching the common server's serializer. Visual rows use token
ID `-1`; ordinary text IDs must remain within the vocabulary.

SVE1 contains consecutive little-endian records: five int32 fields
`0x31455653, rows, nx, ny, width`, then `rows * width` F32 values. The M31 shim
writes `nx=rows`, `ny=1`, `width=6144`; the engine accepts `nx * ny=rows` and uses
ordinary linear decoder positions. Records cover complete visual spans in
prompt order. The engine checks file size, dimension, row mapping and finite
values before changing KV state.

Prefix reuse reports restored engine state. Visual prefixes additionally match
the actual F32 embedding bytes, so identical placeholder IDs or file names
cannot reuse a different picture's KV. `cache_prompt=false` disables lookup
and store. `STOP` is accepted at prefill boundaries and decode steps, and the
process stays available for the next request. Stdout carries only protocol
lines; diagnostics go to stderr.

## Optional image and video input

There is no `mtmd` target in this tree. Build the separate helper in `vision/`
against unmodified llama.cpp PR #31192 at `6b95a990`; its ggml stays in that
process. The M31 shim starts it lazily when vision is configured. Images and all
video frames join the text in one mixed prefill. The full remaining mixed prompt
must fit the configured chunk and planner budget; otherwise it fails explicitly.

Missing helper/model/mmproj, helper death, malformed output and startup or reply
timeouts produce `vision unavailable` (HTTP 503), including a blocked helper
write or incomplete reply. The helper is reaped and a later request may restart
it. Text requests remain available. Build, configuration, content-part examples
and portable oracle commands are in the
[vision section](../../docs/MINIMAX_M31.md#vision-separate-opt-in).
Adaptive short-chunk prefill remains a separate contribution.

`SLEEP` (CUDA backend, between requests) is a deep sleep: the engine frees all of
its VRAM (dense weights, expert cache, KV/index rows, scratch), unregisters the
RAM arena and resets the device's CUDA context, so the process holds no GPU
memory and another program can use the whole card. The arena, a host copy of
the dense device buffers and the valid KV/index rows stay in RAM. `WAKE`
creates a new context, uploads dense weights and KV rows, seeds the expert cache
with the same experts in the same slots, and answers `READY`; the arena then
registers in the background. A `GEN` while asleep wakes first (an `INFO woke=1`
line precedes its output). The conversation continues without a new prompt
read: the same prefix is reused as without the sleep, and logits are bitwise
equal. `SLEEP` while asleep and `WAKE` while awake are idempotent. A failed
wake answers `ERR` and leaves the engine asleep; `conversation=0` in `SLEPT`
reports that the KV rows could not be kept (the next request reads its prompt).

## Layout

- `m31_engine.*`, `m31_geometry.hpp`, `m31_arena.hpp`: model graph and storage.
- `m31_config.*`: typed defaults, budgets and validation.
- `gpu/`, `prefill/`, `cpu/`: expert streaming, planning and fused decode.
- `serve/`: common-server shim, tokenizer export, tool parsing and bounded media/helper I/O.
- `vision/`: external mtmd helper and its separate pinned CMake build.
- `tests/`: small CPU tests; `tests/oracle/` is a separate, pinned llama.cpp build.
- `bench/`: portable quality, speed and logit comparison clients.

The optional oracle is test infrastructure. It is not needed to run the engine
or HTTP server. Development probes, machine launchers and experimental kernels
are excluded.

## Validation checklist

The numerical and performance items below record PR1's text validation and its
remaining limitations. Vision build and reproduction commands are documented
separately; these PR1 checks do not claim final image/video acceptance.

- [x] Opt-in target; default-build artifact and target comparisons are recorded
  with their exact coverage and Qwen link limitation in
  [the default-build section](../../docs/MINIMAX_M31.md#default-build-comparison).
- [x] Vision is a separate build opt-in with an external helper; adaptive short
  prefill remains a separate slice. The default text configuration needs neither.
- [x] CPU and CUDA build/check commands, GPU-offload oracle commands and the
  model revision are documented. HIP, SYCL, Windows and other GPUs remain untested.
- [x] Historical v8 numbers identify the hardware and distinguish the GPU oracle
  from the roughly 78% top-1 CPU/GPU agreement. They are not final PR1 quality results.
- [x] Self-contained gate20, public long and cache-partition inputs have portable
  reproduction commands; cache comparisons state the required numerical history.
- [x] Final worker numerical/cache/lifecycle checks and same-day quality/speed
  collection are complete; all outcomes are recorded in
  [the PR1 results](../../docs/MINIMAX_M31.md#pr1-candidate-checks) and
  [the receipt](bench/receipts/pr1-text.json).
- [ ] Every tools23 trial meets the unchanged 19/23 threshold: PR1's original
  unseeded trial scored 17/23, and the reference seed-1 trial scored 18/23.
  Both failures remain in the report; no best trial replaces them.
- [ ] HIP, SYCL and Windows build coverage remains unverified. The root Qwen
  executable also has the documented upstream link failure.

The final PR1 decode medians were about 14–15% below historical v8 on this
machine, with the cause unresolved. Collection completion is not an overall-PASS
claim; the measured results and remaining limits are explicit in the document.
