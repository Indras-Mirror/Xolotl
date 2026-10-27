# tools/m31 — MiniMax-M3.1-Flash engine (scaffold)

This directory is the **landing shape** for the community port in
[issue #62](https://github.com/Indras-Mirror/Xolotl/issues/62), not the port itself. There is no engine code here yet: the
contributor's tree lives on their fork, and this scaffold exists so that branch drops in
without a CMake redesign.

Nothing here is built by default. `tools/m31/cmake/m31.cmake` is `include()`d OPTIONAL at the
end of the root `CMakeLists.txt` (after `glm.cmake`) and returns immediately unless
`-DSTRATA_M31=ON` is passed **and** `tools/m31/m31_generate.cpp` exists. A default configure is
unchanged.

## Building it

```sh
cmake -S . -B build-m31 -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=ON   # existing tree flags, plus this one
cmake --build build-m31 --target m31_generate -j
```

| option | default | target |
|---|---|---|
| `STRATA_M31` | `OFF` | `m31_engine` (every `tools/m31/*.cpp` except the entry point), `m31_generate` (the entry point) |
| `STRATA_M31_VISION` | `OFF` | `m31_vision` from `tools/m31/vision/*.cpp` — only when an `mtmd` target exists |

`STRATA_M31_VISION` stays off until llama.cpp PR #31192 (the mtmd encoder) is in the pinned
ggml. Until then the vision path is the contributor's helper process, opt-in and documented in
their `docs/MINIMAX_M31.md`; it is not a target in this tree, because linking an encoder the
tree does not have would be a build claim we cannot keep.

## What the port must provide

The engine is started by `serve/server.py` as `<exe> --serve <args>` and speaks the same line
protocol as `tools/glm/glm_generate.cpp` (`--serve` block, ~line 538) and the DS4.1 engine. The
server side is `StrataEngine` in `serve/server.py` (~line 807); the protocol is not M31-specific:

```
in : GEN <max_new> [key=value ...] <ids csv>      solo request
     GENI <max_new> [...] <ids csv> <embeddings>  solo request with image/video embeddings
     BGEN <slot> <max_new> [...] <ids csv>        batch slot
     BGENI <slot> <max_new> [...] <ids csv> <embeddings>
     STOP                                        cancel the request in flight
     QUIT
out: READY <ctx> stop
     RESUME <reused>                              prefix reuse actually taken
     PP <read> <total> <ms> <tok/s>               one line per prefill boundary
     T <id>                                      one per generated token
     DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> ... <reused> ...
     ERR <text>
```

So the "~150-line shim to `serve/server.py`" in the issue is the contributor's serve loop for
this protocol, not new server code. Three behaviours the server depends on and the port must
carry (their plan item 1):

1. **Prefix reuse** — `RESUME <reused>` for a request that extends or shares the last prompt,
   otherwise the whole prompt is re-read and the conversation cache is pointless.
2. **Cancellation** — a `STOP` is honoured at the next token or prefill boundary, and the process
   stays alive for the next `GEN`.
3. **`GENI` / `BGENI`** — image and video rows go through the prefill path with embeddings, the
   same way text does, instead of one small LLM batch per frame.

## Layout expected by `m31.cmake`

```
tools/m31/
  m31_generate.cpp        the serve entry point (arg parsing, --serve loop, sampling)
  *.cpp                   the rest of the engine -> m31_engine
  vision/*.cpp            optional, STRATA_M31_VISION only
  cmake/m31.cmake         this scaffold
```

Exactly one `main()` in `tools/m31/*.cpp`. The development probes and measurement launchers are
expected to be removed before merge (issue #62, plan item 2); a second `main()` collides at link
time and `CONFIGURE_DEPENDS` will pick a new file up automatically.

## Merge checklist (from AGENTS.md)

- [ ] Opt-in target, default build byte-identical; say how that was checked.
- [ ] One PR per slice: (a) text engine + serve shim, (b) vision, (c) adaptive short-chunk prefill.
- [ ] Every backend a file touches is built (CUDA; HIP/SYCL where applicable — say what was not).
- [ ] `docs/MINIMAX_M31.md` with the numbers and the hardware they were measured on.
- [ ] The claim that the PR's own GPU and CPU paths agree on only ~78% top-1 (KL ~0.78) explained
      in the docs, with why GSM8K/tools is the gate instead of KL to a CPU oracle.
- [ ] The reference commit and the exact commands behind the GSM8K, tool-call and byte-identical
      logit numbers, so they can be re-run on a single 4090.
