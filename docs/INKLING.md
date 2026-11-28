# Inkling-2-Mini on Xolotl

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache 2.0) is a 26B
looped MoE with 3B active parameters per token (the official count; about 5.4B effective per token, because the
12-layer loop core runs 3 times). `xolotl-inkling` is Xolotl's engine for it, behind the usual
OpenAI- and Anthropic-compatible server, with text, images (`--mmproj`) and up to 512K context on one 24 GB card.
Our quant, **Inkling-2-Mini XQ Unbound** (13.93 GB, uncensored), is on Hugging Face:
[IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) (release notes:
[CHANGELOG](xolotl/CHANGELOG.md#inkling-2-mini-xq-re-released-as-unbound-22-nov)).

| | |
|---|---|
| parameters | 26B total, 3B active (official; about 3.1B distinct weights per token: 1.6B shared + 8 of 128 experts), about 5.4B effective per token because layers 6-17 run 3 times |
| layers | 24 physical, 48 applications per token: layers 6 to 17 run 3 passes, with low-rank loop controllers |
| experts | 128 routed, top-8, sigmoid router with a selection bias, SiTU activation, no shared expert |
| attention | sliding window 1024, a global layer every 6th; no RoPE (learned relative-position bias, log-scaled positions on the global layers) |
| extras | short causal convolutions on K, V and both layer outputs; block attention residuals that mix earlier depths |
| context | 1M trained (`--context` sets what you allocate) |

## Numbers (RTX 4090, 22 Nov 2026)

| | 14 GB XQ quant | NVFP4 parent |
|---|---:|---:|
| decode, 2.8K prompt | **113 tok/s** (127 at 7K served, 22 Nov late) | 89 tok/s |
| decode with `--dspark` | **136 tok/s** (~140 on served thinking requests) | 137 tok/s |
| prefill, 8K prompt (served) | ~3,500 tok/s | ~3,000 tok/s (16K context, 512-row chunks) |
| prefill, 32K prompt (served) | **4,190 tok/s** | - (the parent leaves little VRAM for big chunks) |
| prompt chunk (auto) | 2048 rows | 256-512 rows |
| decode during agent work (Gauntlet v3-small, 193K tokens, served sampling) | 116.5 tok/s | - |
| Gauntlet v3-small, 6 tasks (one run) / v3, 7 tasks | 4.42 of 6 (3 solved) / 4 of 7 solved | - |
| long context, `--kv-type k8v4` | 512K fits; decode 108 / 93 / 55 tok/s at 30K / 117K / 477K | - |
| images (`--mmproj`, F16 tower) | a 4K screenshot encodes in 1-2 s | same |

## First numbers (RTX 4090, exact NVFP4 parent, all weights in VRAM, 21 Nov 2026)

| | |
|---|---|
| decode | **57 tok/s** at a short context, 51 tok/s at 8K (llama.cpp reference build: 51) |
| prefill | 1.1K tok/s at 8K |
| parity vs the llama.cpp reference | logits KL 2e-4 and top-1 100 % on a short prompt; KL 3e-3, top-1 97.7 % teacher-forced over 1.6K tokens of text; 64 greedy tokens identical on 2 of 3 prompts, the third splits at a near tie after 215 of 248 characters |
| VRAM | 20.5 GiB weights + 0.75 GiB state at 16K context (token table in RAM) |

The 22 Nov numbers above come from the fused decode kernels and the tensor-core prompt attention
([CHANGELOG](xolotl/CHANGELOG.md)).

## Speculative decoding with the DSpark (MTP) drafter (opt-in)

Inkling ships a block drafter in `mtp.safetensors`: 3 dense layers that read the target's hidden states at layer
visits 5, 23, 41 and the final norm, and propose up to 7 tokens per step through a rank-256 Markov head.
`--dspark <drafter.gguf>` turns it on (the GGUF is the `dspark/` file of
[unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)); `xolotl-inkling-dspark` runs the
drafter alone against saved target taps.

- Greedy and sampled requests both decode speculatively (sampled speculation is exact: the target's own sample decides each
  token; `XOLOTL_SPEC_SAMPLED=0` limits it to greedy).
- The short convolutions make the model recurrent, so a rejected draft is undone exactly: every verify keeps its
  `[conv state ; chunk inputs]` and the state after the accepted prefix is copied back (no second forward). Sliding
  ring cells of rejected positions are marked empty; global K/V is positional.
- Draft length is dynamic: drafting stops when the cumulative top-1 probability falls below
  `XOLOTL_DSPARK_TAU` (default 0.35), at most `XOLOTL_DSPARK_MAXD` (7). Each verify length keeps its own graph.
- The drafter's Markov head, context projection, softcap and argmax over the 200K vocabulary run on the GPU
  (draft 25 ms -> 4 ms per step).
- Correctness: verify lengths 1, 5, 6 and 7 are bit-identical to plain decoding. Lengths 2 to 4 use a different
  reduction order in ggml's quantised mat-vec, so a greedy near-tie can flip after a few hundred tokens (rounding,
  not a logic error). `XOLOTL_SPEC_EXACT=1` pads those lengths for gates.
- Speed: 113 -> 136 tok/s on a 2.8K-token prompt (2.4 of up to 7 drafts accepted a step), ~140 on served thinking
  requests. It helps most on structured text (code, maths); on prose acceptance is lower. `XOLOTL_DSPARK_ADAPT=1`
  (the wrapper's default) switches to plain decoding while drafting is not paying.
- The drafter's global layer keeps a ring of the last `XOLOTL_DSPARK_WINDOW` positions (default 32768; 0 = a
  full-context cache), so it costs no extra memory at long context.

## K/V cache types

`--kv-type f16|q8_0|k8v4` sets the cache of the global (full-context) layers; the sliding-window rings stay f16.
The quantised caches run on the same fused kernels as f16: the tensor-core prompt attention stages the raw
q8_0 / q4_0 rows and dequantises them in the tile, decode attention reads them in place, and the decode prepare
kernel (conv + clip + norm + store) writes q8_0 / q4_0 rows itself. A 107,800-token prompt at k8v4 prefills in 29 s
(~3.7K tok/s). **512K context fits one 24 GB card** at k8v4 with the DSpark drafter and the image tower loaded
(6.8 GiB of K/V state, 23.4 GB of VRAM in use): a 477,603-token prompt prefills in ~3 min.

Decode attention over the global layers reads the whole cache every token, so it is the part that grows with
context. Its kernel scores one cell per lane (the K row in registers, no cross-lane reductions) and runs near the
card's memory bandwidth (~840 GB/s at 117K). Served decode at k8v4, no drafter, RTX 4090:

| context | 7K | 30K | 117K | 477K |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 127 | 118 | 100 | 56 |

Against f16 on the parent: q8_0 KL 3.6e-4 with identical greedy output, k8v4 (q8_0 keys, q4_0 values) KL 8e-4.

**Hadamard rotation of the 4-bit leg (default on).** A q4_0 leg goes through a randomised Hadamard rotation
before it is stored: on k8v4 the values are rotated and the attention output is rotated back once per head (keys
stay plain q8_0, which is near-lossless without it). The rotation spreads outlier channels across each 32-value
block. Per-token |dNLL| against f16 on 32K tokens of wikitext (f16 run-to-run floor 0.0395):

| k8v4 | per-token \|dNLL\| | excess over the floor |
| --- | ---: | ---: |
| no rotation | 0.0483 | 0.0088 |
| K and V rotated | 0.0459 | 0.0064 |
| **V rotated (default)** | **0.0454** | **0.0059** |

Mean perplexity is unchanged in every arm (within one standard error). `XOLOTL_INKLING_KV_ROT=0` turns it off,
`=2` rotates every leg (f16 and q8_0 too; with f16 it matches the unrotated cache within the floor, the
exactness check).

## Long context with little VRAM: `--kv-vram` and `--kv-sparse` (opt-in)

`--kv-vram N` keeps the first N cells of the global layers' K/V on the card and the rest in pinned host RAM, so the
context is no longer capped by VRAM: `--context 524288 --kv-vram 32768` runs 512K on a 12 GB card at the VRAM of a
32K context (k8v4: ~10.5 KB a token; the host part of 512K is 6.1 GiB). Prompts stream the host part through the fused
prompt kernel in 32K-cell chunks and merge the partial softmax sums with the VRAM part. Decode runs the host part on
the CPU (AVX2 k8v4 kernel, cell slices claimed by a persistent thread pool) while the GPU does the VRAM part, and the
two merge in the existing split-K reduce. Before the boundary the result is identical to the all-VRAM path; past it,
greedy output matched on our checks and per-token NLL stayed within the engine's own precision noise. Works with
`--dspark`: a speculative verify chunk (up to 16 tokens) reads the host part per token up to its own position, and with
`--kv-sparse` picks one page set for the whole chunk (the union of what each token wants). 34.5K prompt, 33.5K cells in
host RAM, RTX 4090: greedy 128 tokens with the drafter identical to without it, exact and sparse; no speed gain yet
(22.1 vs 23.0 tok/s exact, 44.0 vs 44.5 sparse) because the CPU op costs about the same per token in a verify chunk as
in a single step.

With exact attention, decode past the resident part is bound by host RAM bandwidth: every token reads the whole host
K/V once per global-layer visit (8 per token). Measured with a 10.98 GB XQT build (same recipe as the released 10.94 GB
file), `--context 524288 --kv-vram 32768`, before the current CPU kernel and slice scheduling (+12-17 % and +16 %
decode at 34K in our A/B runs):

| prompt | RTX 4090 prefill / decode tok/s | RTX 3060 12 GB prefill / decode tok/s |
| --- | ---: | ---: |
| 8.9K | 3,988 / 132 | 725 / 48.6 |
| 34.6K | 5,187 / 86 | 726 / 37.4 |
| 137.7K | 3,969 / 10.5 | 445 / 8.0 |
| 267.9K | 2,909 / 5.3 | 289 / 4.1 |
| 495.3K | 1,955 / 2.7 | |

VRAM stays at 13.6-14.5 GB on the 4090 and 11.0-11.6 GB on the 3060 at every depth. Needle found at 32K, 128K, 256K
and 500K (498,925 tokens). A follow-up turn on a cached 495K prompt answers in 1.7 s. (Ryzen 7 5700X, DDR4; the
3060 is on a PCIe 3.0 x4 slot.)

`--kv-sparse N` (needs `--kv-vram`; **not exact**) makes decode read only part of the host K/V: per K/V head the N
cells on the 64-cell pages that score highest, half by a landmark (q · the page's mean key, as in ShadowKV) and half
by an upper bound on q · k (the page's per-dimension key min / max, as in Quest), plus the newest 1,088 cells (their
position bias is not flat yet). Pages it skips still count, as 64 cells at their mean key and mean value. Attention
over the pages it reads is exact. The page summaries are built on first use and rebuilt from wherever a step writes.
Measured on a 34.5K prompt with 1K cells resident (33.5K in host RAM), RTX 3060, against exact attention over the
host part:

| `--kv-sparse` | host-output rel. error | greedy 128 tokens vs exact | decode tok/s |
| --- | ---: | --- | ---: |
| off (exact) | 0 | | 21.3 |
| 1024 | 0.157 | identical | |
| 2048 | 0.110 | identical | 30.5 |
| 4096 | 0.078 | identical | |

At long context on the same 3060 (`--context 524288 --kv-vram 32768 --kv-sparse 2048`, the released 10.94 GB file):
137.5K prompt 448 / **24.5** tok/s prefill / decode (exact: 8.0 decode), 267.2K prompt 291 / **18.8** (exact: 4.1).

Inkling's global heads spread their attention widely, so the error is higher than these methods report on other
models; treat it as an option for long documents where exact decode is too slow. **Not for agent loops:** two
coding-agent runs (QuetzaCodetl worker, 32.6K-token prompt, 32K cells resident, RTX 3060) derailed with
`--kv-sparse 2048` (a 33K-token thinking loop on a one-line bug; a reply that ran past 20K tokens without a tool call),
while the same task with exact attention finished and passed its tests in 4 minutes.

DSpark on a 12 GB card next to the 10.94 GB file: the BF16 drafter with its buffers (one compute buffer per verify
length) needs about 1 GB more than the RTX 3060 has. It fits with the Q8_0 drafter
(`tools/inkling/quant_dspark.py`, 402 MB; same acceptance and identical output as BF16 on our check),
`XOLOTL_DSPARK_WINDOW=8192` and `XOLOTL_DSPARK_MAXD=3` (11.6 GB), but it did not pay there: served greedy 700-token
answers ran 47.9 / 49.7 / 43.3 tok/s with it against 49.1 / 49.5 / 49.8 without (two code prompts, one prose). A
drafter step takes ~17 ms on the 3060 and a 3-4 token verify costs about two plain steps, so we leave it off on 12 GB
cards. `XOLOTL_INKLING_SPARSE_SEL=0|1|2`
picks Quest / landmark / both (default), `XOLOTL_INKLING_SPARSE_APPROX=0` drops the skipped-page estimate,
`XOLOTL_INKLING_SPARSE_CHECK=1` prints the error against exact attention (slow).

Spill switches (default on, `=0` off): `XOLOTL_INKLING_SPILL_FUSED`, `_SPILL_CPUOP`, `_SPILL_FAR`, `_SPILL_TP`,
`_SPILL_SIMD` (`=3` the previous CPU kernel); `XOLOTL_INKLING_SPILL_SLICES` (default 32) sets the CPU slices per
token; `XOLOTL_INKLING_SPILL_PROF=1` prints the CPU op's time and the gap between visits.

## Get a GGUF

- **Exact parent (what the numbers above use):** the official checkpoint keeps its experts in NVFP4 (layers 0-19)
  and MXFP8 (20-23). llama.cpp's converter from [PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694)
  dequantises them; `tools/inkling/convert-keep-nvfp4.patch` makes it keep the NVFP4 bits (+ a per-expert `.scale`)
  and writes MXFP8 as BF16 (exact). 22.9 GB. `tools/inkling/check_nvfp4_exact.py` checks the experts bit for bit.
- **Ours:** [IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored): `Inkling-2-Mini-XQ-Uncensored.gguf` (13.93 GB,
  the model), `mmproj-Inkling-2-Mini-XQ-Uncensored-F16.gguf` (0.64 GB, images) and
  `Inkling-2-Mini-XQ-Uncensored-DSpark-BF16.gguf` (0.64 GB, the drafter). Load them with `--native`, `--mmproj` and
  `--dspark`.
- **Upstream GGUFs and llama.cpp quants** of Inkling-2-Mini load too (same tensor names; the `.scale` tensors are
  optional). Not yet tested here; a dequantised BF16 file is ~52 GB, so on a 24 GB card use a quant or
  `--cpu-exps`.

## XQT edition: the 12 GB card file

[IndrasMirror/Inkling-2-Mini-XQT-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQT-Uncensored)
(28 Nov 2026): Unbound v3 in 10.94 GB, with part of the routed experts in Xolotl's trellis-coded types (TQP5 / TQP6 /
TQP7), so the whole model and 32K of context fit a 12 GB card. RTX 3060 12 GB, k8v4, 8K prompt: 32K context 10.9 GB
VRAM, 765 tok/s prefill, 48.3 tok/s decode; 64K context 11.3 GB, 516 / 48.4. MMLU-100 77 %, HumanEval-21 21/21,
tool 10/10, agent 3/3, MATH-24 91.7 / 91.7 / 83.3 %. Xolotl only (other engines refuse the TQP types).

## Build and run

```bash
ninja -C build-xolotl xolotl-inkling xolotl-inkling-dspark   # configured as in AGENTS.md
python tools/inkling/export_tokenizer.py <Inkling-2-Mini HF dir> <pack dir>   # once: the server's tokenizer pack
# edit configs/xol-inkling.json (gguf path, tokenizer pack, context), then:
.venv/bin/python -m serve.server --engine strata --config configs/xol-inkling.json --port 8134
```

Add `--mmproj <file>` for images, `--dspark <file>` for the drafter, and `--kv-type k8v4 --context 524288` for
512K. `--cpu-exps 20-23` keeps those layers' experts in host RAM (the CPU computes them) for more context on a 24 GB card.
Thinking effort follows `reasoning_effort` (the chat template's effort line); sampling defaults to the model's
`generation_config.json` (temperature 1.0).

## Notes

- The forward graph is ported from llama.cpp PR #32694 by Pedro Cuenca (MIT). One bug in that PR is fixed here:
  with NVFP4 experts kept, loop passes 2 and 3 ran their experts without the per-expert scale (upstream's
  dequantised GGUFs never hit it); `tools/inkling/ref-loop-expert-scale-fix.patch` is the fix for the reference.
- The conv state is recurrent, so a request resumes by appending to what the engine read, or from the checkpoint
  one token before the end of the previous prompt (sliding rings + conv states), else from token 0.
- Vision: `--mmproj <mmproj.gguf>` runs the 23-block image tower inside the engine (rebuilt from llama.cpp PR
  #32694's inkling2v). Its weights go to the GPU per picture and are freed after it, so a text-only session keeps
  no VRAM for it; big pictures (a 4K screenshot is 16K positions) use flash attention and encode in 1-2 s. The
  mmproj is a separate file (convert_hf_to_gguf.py --mmproj).
