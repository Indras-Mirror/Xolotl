# DeepSeek-V4.1: routed experts on a second GPU (`STRATA_DS41_REMOTE`, opt-in)

A second NVIDIA card holds a set of routed experts and computes them for the engine, layer by layer, while the
engine's card computes its own. The engine is unchanged without the variable.

## Why a helper process

On Windows (WDDM) a CUDA context on a second card limits the host memory **one process** can pin: with a context on
the RTX 4080 in the process, pinning stopped at 56 GiB and the 4080's context broke past it, whatever the flags
(`cudaHostRegisterDefault`, `Mapped`, `Portable|Mapped`). With 48 GiB pinned or less, the 4080 could allocate and write
its full 15 GiB. The engine pins a 72 GiB arena, so the second card's work runs in its own process: a process that only
touches the RTX 5090 pinned 80 GiB while a second process held and rewrote 14 GiB on the 4080 (56 rounds, no failures).

Measured on one PC: RTX 5090 (PCIe 5.0 x16) + RTX 4080 over OCuLink (PCIe 4.0 x4), i7-13700K, 128 GB, Windows 11,
driver 617.42, CUDA 13.0.

## How it works

- `ds41_helper` (built with `ds41_generate`, CUDA builds only) reads the routing profile (`--rank`), skips the first
  `--skip-gib` of it (about what the engine's card keeps in its VRAM cache, default 22) and loads the next `--gib`
  (default 13.5) into its card. It then waits for requests in a named shared-memory segment (`ds41_remote.hpp`).
- The engine (`STRATA_DS41_REMOTE=<name>`) opens the segment at start-up and waits for the helper to be ready. In a
  verify/decode pass (`gpu_run_n_ex`), the experts the helper holds and the engine's card does not become a fifth
  class, REMOTE. Right after the classify, the engine writes the pass's rows and the REMOTE entries to the segment;
  the helper quantizes the rows to q8_1 and runs `native_expert_grouped` per format class - the engine's own kernels -
  and writes one row per entry back. The engine collects them just before its weighted sum and uploads them into
  their rows, as it does for rows the CPU pool computed.
- Prompt chunks (`run_chunk`) use a second, larger segment, `<name>_chunk` (the helper's `--chunk-tok`, default 12288
  rows; about 1 GB of shared memory). The engine sends the chunk's rows and the entries of the helper's experts that
  are not in its VRAM, computes the rest itself, and scatters the helper's rows into its parts before the weighted
  sum. The helper runs them on the engine's prompt kernels: MMQ over a stretch of its slab (its experts are laid out
  per layer and format class at run_chunk's MMQ stride, `ds41_mmq_plan.hpp`), PTQ1_0 matrices transcoded to Q2_0
  first. The file-tier staging skips the helper's experts.
- The RAM arena leaves out the helper's experts (a pass never reads them from it), so its 72 GiB hold the next
  experts of the ranking instead. `STRATA_DS41_REMOTE_DUP=1` keeps the old arena, which also holds them.

## Keeping the helper's card awake

The engine sends a request every few layer calls, often 5-20 ms apart. Between requests the 4080 dropped to its idle
clock, and a request then took 1-2 ms instead of ~0.14 ms - longer than the engine's own layer call (~1.2 ms), so the
engine waited. `remote_bench` (5 tokens, 2 experts per request) against the helper:

| idle gap between requests | median round trip |
| ---: | ---: |
| 0 | 135 us |
| 1 ms | 152 us |
| 5 ms | 953 us |
| 20 ms | 1931 us |

While idle, the helper runs one of its experts every `DS41_HELPER_WARM_US` (default 100; 0 = off) on its own stream.
After `DS41_HELPER_WARM_IDLE_S` seconds (default 10) with no request it stops, so the card idles (and its fan stops)
while no one is using the model; the next request starts it again.

| keep-warm interval | 4080 while idle | median at 0 / 5 / 20 ms gaps |
| ---: | --- | --- |
| none | 210 MHz | 135 / 953 / 1931 us |
| 150 us | 1.7 GHz, ~60 W | 165 / 177 / 249 us (p90 802 at 20 ms) |
| 100 us | 2.6 GHz, ~100 W | 137 / 142 / 151 us |

The helper's card draws ~100 W while it serves.

## Measured

Same PC as above, the serve configuration of `run-ds41-1017.bat` (DSpark depth 4, route bias 0.5 / 0.3, 72 GiB
arena, prompt batch 12288) plus `STRATA_D3_DMA_STREAM=1` in every run below. `decode_bench.py` (4 prompts, ~1,450
tokens) and `agent_bench.py` (a 29.5K-token first prompt, then 4 agent turns of 38-2,242 new tokens with 300 tokens
of decode each). Two runs of each, interleaved, same day:

| | decode | first prompt (29.5K) | agent turns' prefill (4,478 new tokens) | agent decode |
| --- | ---: | ---: | ---: | ---: |
| helper for decode only, arena with its experts | 35.5 / 33.7 tok/s | 65.0 / 65.3 s | 36.2 / 36.5 s | 31.1 / 33.0 tok/s |
| helper for decode and prompt chunks (MMQ), arena without them | 37.6 / 36.0 tok/s | 61.7 / 61.6 s | 31.3 / 31.3 s | 32.2 / 33.5 tok/s |

Without the helper at all (earlier the same day): decode 34.5 / 31.7 tok/s, agent decode 29.6 / 29.4 tok/s, first prompt
63-64 s, agent turns' prefill 36.3 s. Agent decode moves with what the model writes (the turns differ run to run).

The pieces, measured separately:

- Prompt chunks on the helper's MMVQ (the decode kernel) first: agent turns' prefill 36.2 -> 31.2 s, but the first
  prompt did not gain - the 12288-row chunks took the helper 25 us an entry (a 16,568-entry layer: 365 ms) and the
  engine waited 45 ms a layer for it. On MMQ the same layer takes 84 ms (plus ~40 ms to bring the rows up over
  PCIe 4.0 x4) and the engine waits 0.0 ms.
- A one-shot 3,000-token prompt (docs text): 16.17 s without the helper -> 15.43 s with the helper on chunks
  (file wait 7.1 -> 5.3 s, 5,092 -> 3,768 file-tier experts read).
- The arena without the helper's experts, decode only (chunks off): no change measured (34.4 / 32.2 vs 34.0 / 33.3 tok/s).
- `STRATA_DS41_REMOTE_BIAS=1`: agent decode 30.9 tok/s against 32.2 / 31.5 without it - no gain.

Exactness. The helper's rows are not bit-identical to the engine card's: an RTX 4080 (sm89) build and the RTX 5090's
(sm120a) round differently, so texts differ after some tokens even with `--route-bias 0` (1 of 4 prompts identical).
Perplexity over the same 3,000 tokens, one-shot, prompt batch 12288:

| | PPL | top-1 |
| --- | ---: | ---: |
| no helper | 5.8712 | 0.601 |
| helper, prompt chunks off (its experts read from the file tier, the engine's MMQ) | 5.9025 | 0.605 |
| helper, prompt chunks on MMQ | 5.9025 | 0.605 |
| helper, prompt chunks on MMVQ (before MMQ) | 5.8691 | 0.605 |

The helper on MMQ gives the engine's own result to four digits. The first two rows differ only in where the engine
found the same experts (the RAM arena or the file tier) - the engine computes both - so that spread comes from the
engine's two prompt paths, not from the helper.

## Against `--remote-device` (Windows)

The engine's own second card, `--remote-device N`, runs in the engine's process, so on Windows it meets the pinning
limit above. On this PC it started only with `--conversation-cache-mib 0` (else "pool: CUDA stream: out of memory" on
the RTX 5090 at load) and `STRATA_ARENA_PIN_GIB=48` (pinning the whole sliced arena left the 5090 refusing every later
allocation; `auto` came to 97 GiB and did not help, 56 failed, 40 and 48 loaded). The rest of the arena is then
locked, not pinned.

Same day, upstream main 03be8a60 with the Windows fixes, the settings of "Measured" above plus
`STRATA_DS41_PASS_GRAPHS=16` and `--conversation-cache-mib 0` in every run, two runs each where shown:

| | decode | agent turns' prefill | agent decode | agent total |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 alone | 35.8 / 36.8 / 37.0 tok/s | 36.3 / 37.5 / 37.0 s | 35.3 / 35.9 / 35.6 tok/s | 64.4 / 67.9 / 70.3 s |
| RTX 5090 alone, `STRATA_ARENA_PIN_GIB=48` | 29.1 tok/s | 38.3 s | 25.2 tok/s | 71.9 s |
| `--remote-device 1` (RTX 4080), `STRATA_ARENA_PIN_GIB=48` | 26.2 / 26.9 tok/s | 34.4 / 34.6 s | 23.2 / 23.3 tok/s | 74.9 / 78.1 s |
| `STRATA_DS41_REMOTE` (this helper on the RTX 4080) | 40.1 / 42.9 tok/s | 32.1 / 32.0 s | 37.4 / 40.3 tok/s | 62.5 / 60.3 s |

Most of `--remote-device`'s loss here is the pin cap: the RTX 5090 alone with the same cap loses about as much. With
the helper the engine's process sees one card and pins the whole arena. On Linux, where this limit does not apply,
`--remote-device` is as fast as the helper on decode (below).

## Linux

The same PC on Ubuntu 26.04 (kernel 7.0, driver 595.99.02, CUDA 13.2, `iommu=pt`), main e9d0826d built for sm_89;120a
with `-DSTRATA_GGML_CUDA=ON`. The settings of the Windows table without the three Windows-only ones
(`STRATA_BLOB_DIRECT=0`, `STRATA_ARENA_SLICE_GIB`, the expert mirror); the model on ext4. `--remote-device 1` at its
defaults (no pin cap, conversation pool on). Interleaved; fresh = the agent bench's first, 29K-token prompt:

| | decode | fresh prompt | agent turns' prefill | agent total |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 alone | 51.9 / 53.5 / 52.2 tok/s | 55.6 / 56.1 / 54.9 s | 40.5 / 40.8 / 40.3 s | 60.4 / 64.1 / 62.1 s |
| `--remote-device 1` (RTX 4080) | 58.8 / 59.3 tok/s | 60.1 / 56.9 s | 33.4 / 33.7 s | 52.7 / 51.0 s |
| `STRATA_DS41_REMOTE` (this helper on the RTX 4080) | 59.6 / 59.8 / 58.9 tok/s | 49.3 / 49.6 / 49.3 s | 31.7 / 32.1 / 31.9 s | 50.0 / 49.9 / 47.9 s |

The two tie on decode; the helper reads a fresh long prompt 7-10 s faster. The helper builds from main as is
(`ninja -C <build> ds41_generate ds41_helper`) and is started the same way as on Windows:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ds41_helper -m V41-XQ-1017.gguf --rank rank-1017-085.bin --device 1 --skip-gib 22 --gib 13.5 &
export STRATA_DS41_REMOTE=ds41_remote STRATA_D3_DMA_STREAM=1
# ... then the usual serve_ds41.py start (the config's "gpu": 0 keeps the engine on the first card)
```

## Settings

| variable / flag | where | meaning |
| --- | --- | --- |
| `STRATA_DS41_REMOTE=<name>` | engine | use the helper serving segment `<name>` (off when unset) |
| `STRATA_DS41_REMOTE_WAIT` | engine | seconds to wait for the helper at start-up (default 600) |
| `STRATA_DS41_REMOTE_PROF=1` | engine | per 2000 calls: entries per call, how long the engine waited for the helper |
| `STRATA_DS41_REMOTE_BIAS=1` | engine | the helper's experts take `--route-bias`'s bonus too, like the ones in VRAM (no gain measured) |
| `STRATA_DS41_REMOTE_CHUNK=0` | engine | prompt chunks without the helper |
| `STRATA_DS41_REMOTE_DUP=1` | engine | the RAM arena holds the helper's experts too (the old arena) |
| `--device`, `--skip-gib`, `--gib`, `--name` | helper | the card, where its set starts in the profile, its size, the segment name |
| `--chunk-tok` | helper | the widest prompt chunk it takes (default 12288; 0 = decode only) |
| `DS41_HELPER_MMQ=0` | helper | prompt chunks on native_expert_grouped instead of MMQ |
| `DS41_HELPER_WARM_US` | helper | keep-warm interval while idle (default 100; 0 = off) |
| `DS41_HELPER_WARM_IDLE_S` | helper | stop the keep-warm after this many seconds with no request, so the card can idle (default 10; 0 = never stop) |
| `DS41_HELPER_PROF=1` | helper | per 20000 requests: time per stage; per prompt chunk of 1024+ rows: rows up / compute (syncs, so slower) |
| `DS41_HELPER_CLASSES=1` | helper | prints its set's experts per format class |

Start the helper first, with only its card visible, then the engine:

```bat
set CUDA_VISIBLE_DEVICES=1
start "" ds41_helper -m V41-XQ-1017.gguf --rank rank-1017-085.bin --device 0 --skip-gib 22 --gib 13.5
set CUDA_VISIBLE_DEVICES=
set STRATA_DS41_REMOTE=ds41_remote
set STRATA_D3_DMA_STREAM=1
rem ... then the usual serve_ds41.py start (the config's "gpu": 0 keeps the engine on the first card)
```

One engine per helper. The helper keeps serving when the engine restarts.

## Not done

- The helper's set is fixed at start-up (the profile's order). The engine's own cache adapts; the helper's does not.
- A chunk wider than `--chunk-tok`, or with more of the helper's entries than half its entries, runs without it.
