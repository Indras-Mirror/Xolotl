# MiniMax-M3.1-Flash

Opt-in text engine and shim to the common Strata HTTP server, developed in
[issue #62](https://github.com/Indras-Mirror/Xolotl/issues/62).
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
     GENI <max_new> [...] <ids csv> <embeddings>
     BGENI <slot> <max_new> [...] <ids csv> <embeddings>
     STOP
     QUIT
out: READY <ctx> stop
     RESUME <reused>
     PP <read> <total> <ms> <tok/s>
     T <id>
     DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> ... <reused> ...
     ERR <text>
```

The current slice is text only. `GENI` and `BGENI` fail explicitly with
`ERR vision unavailable`. Vision and adaptive short prefill are follow-up slices.
Prefix reuse must reflect restored engine state, rather than a promised cache
hit; `cache_prompt=false` disables lookup and store. `STOP` is accepted at prefill
boundaries and decode steps, and the process stays available for the next request.
Stdout carries only protocol lines; diagnostics go to stderr.

## Layout

- `m31_engine.*`, `m31_geometry.hpp`, `m31_arena.hpp`: model graph and storage.
- `m31_config.*`: typed defaults, budgets and validation.
- `gpu/`, `prefill/`, `cpu/`: expert streaming, planning and fused decode.
- `serve/`: common-server shim, tokenizer export and tool parsing.
- `tests/`: small CPU tests; `tests/oracle/` is a separate, pinned llama.cpp build.
- `bench/`: portable quality, speed and logit comparison clients.

The optional oracle is test infrastructure. It is not needed to run the engine
or HTTP server. Development probes, machine launchers and experimental kernels
are excluded.

## Validation checklist

- [x] Opt-in target; default-build artifact and target comparisons are recorded
  with their exact coverage and Qwen link limitation in
  [the default-build section](../../docs/MINIMAX_M31.md#default-build-comparison).
- [x] Text engine and common-server shim are isolated from vision and adaptive
  short prefill, which remain separate slices.
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
