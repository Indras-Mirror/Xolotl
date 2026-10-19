<h1 align="center">Strata-DSV4.1</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers × 384 experts) on one gaming PC - 152 GiB of experts across VRAM, pinned RAM and NVMe</b><br>
DeepSeek-V4.1 and V4-Flash engines built on <a href="https://github.com/Niko1221/Strata">Strata</a> · Linux · NVIDIA · MIT</p>

<p align="center"><b>RTX 4090 + 90 GB RAM:&nbsp; prefill ~250 tok/s (exact) &nbsp;·&nbsp; DSpark decode 17-30 tok/s at the coding-safe bias (x0.5) &nbsp;·&nbsp; 512K context</b></p>

> **Status: experimental, single-developer research code, measured on one PC.** The V4.1 engine generates correct
> text, its GPU prompt path is deterministic (byte-identical logits run to run), and it serves an OpenAI / Anthropic
> API with tool calls (one sequence at a time).  The GSQ model file is on Hugging Face:
> **[IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)**
> (upload complete 2026-10-13; all 16 parts, the vision/MTP sidecar and the small files are on the Hub).
> The earlier **DeepSeek-V4-Flash (0731)** engine is still here and documented in the second half of this README.
> (This repo was called `Strata-DS4`; old links redirect.)

## Latest (2026-10-15)

- **Coding quality vs speed, measured on real tasks** (21 HumanEval + 10 tool calls, greedy; `~/AI/v41-dev/taskeval`):
  the decode route bias trades correctness for speed, so the default is now **x0.5** (`--route-bias 0.5
  --route-bias-file 0.3 --arena-adapt`).
  | route bias | HumanEval (21) | tool calls | DSpark tok/s (docs / python) |
  | --- | ---: | ---: | ---: |
  | x0.25 | 20/21 | 8/8 | 13.7 / 16.1 |
  | **x0.5 (default)** | **20/21** | **10/10** | 22.3 / 22.9 -> **30.3 / 29.1** (below) |
  | x0.75 | 19/21 | 10/10 | 28.3 / 35.2 |
  | 1.0 (old default) | 16/21 | 10/10 | 37.7 / 39.1 |
- **DSpark in `--serve`** (`STRATA_SERVE_DSPARK`, on by default in `ds41-engine.sh`): exact speculative sampling (the
  output distribution is the model's at any temperature; unit-tested over 400k draws per case).
- **Confidence-gated verify** (`STRATA_DSPARK_CONF=0.6`): the draft head's confidence output decides how many drafts a
  verify pass checks - a rejected draft still costs its experts' PCIe/NVMe bytes (~25 ms each).  Same tokens up to
  rounding ties; 256-token decode, x0.5: prose 9.5 -> **17.3**, code 18.8 -> **25.5**, docs 29.9 -> 30.3, python 26.6 -> **29.1** tok/s.
- **Bit-exact 4-bit KV cache** (`--kv-fp4`): V4.1 already rounds its compressed KV to an NVFP4 grid and its indexer keys to
  an MXFP4 grid; they are now stored that way (exact packers in the ggml fork, `tools/v41/ggml/kv-fp4.patch`) instead of
  BF16 - logits bitwise identical, **+1.1 GiB of VRAM at 512K context**.
- **Serve turns ~2x faster**: DSpark's seed rows come off the prompt batch instead of re-running the last 128 tokens.
  and small turns read only the experts they route (whole-layer prefetch only from 640 rows).  512K context:
  +272-token turn 33.5 -> **~12 s**, +1000-token turn 40.2 -> **19.4 s**, fresh 8000 tokens 51 -> 37 s (a small turn's
  expert reads now run at ~5.3 GB/s, the NVMe's ceiling).
- End to end through the server (the defaults): HumanEval 20/21, tool calls 10/10, agentic bug-fix loops 3/3, and real
  QuetzaCodetl agent turns.
- **+2.4 GiB of expert slots while serving**: the prompt batch's 8192-column state is allocated only while a batch runs
  (512K: 1227 -> 1509 experts in VRAM; every 2.6 GiB of slots is 10-20% of a verify pass).
- NVMe: 32 concurrent reads per call (was 8): -17% per expert read, decode +4-5%.  Ternary (PTQ1_0) CPU expert kernel
  3.9-7.7x faster.

## Latest (2026-10-14)

- **DSpark speculative decoding works and beats plain decode** (one-shot `ds41_generate`, greedy): the checkpoint's
  own `mtp.0-2` draft stack drafts 4 tokens, one batched verify pass checks them.  Recommended:
  `STRATA_DSPARK=V41-MTP-Vision.Q2_K.gguf STRATA_DSPARK_DEPTH=4` (Q2_K sidecar on the Hub; frees 2.76 GB of VRAM for experts).
  | Prompt (256 tokens, default decode biases) | plain | DSpark | |
  | --- | ---: | ---: | ---: |
  | docs | 29.9 | **39.5** | 1.32x |
  | python | 20.6 | **31.7** | 1.53x |
  | code | 26.7 | **35.6** | 1.33x |
  The verify pass serves the union of the pass's experts once per layer (the batched "D3" tier: PCIe-all misses,
  file-tier experts read in parallel and run on the GPU, VRAM LRU, one-slot cross-layer prefetch) - verify 167 -> ~100 ms.
  Correctness: with biases off, every verify row matches an independent prefill except at near-ties.  PCIe is now the wall
  (nsys: expert DMAs = 1/3 of decode).  Not yet in `--serve` (temperature-1 sampling needs speculative sampling - designed).
- **Quality baseline on DwarfStar's 112 official DeepSeek-API fixtures** (`tools/v41/quality/`): exact routing 86.2% API
  top-1 on the short prompts; **the default decode route bias costs 14 points (72.1%)** and causes repetition loops on some
  prompts.  Bias x0.25 keeps 85.2% at 12.6 / 7.4 tok/s - pick for your use (curve in `docs/ds4/V41_START_HERE.md` §5c).
  Smarter biases that only bend near-ties (`STRATA_RB_PROTECT`, `STRATA_RB_MARGIN`) are being measured.
- **Routing per request, and a route meter** (`--serve`): the GEN keys `route_bias=` / `route_bias_file=` (from the
  server's `strata_tune`) override the startup biases for one request, so one server answers exact-routing requests
  beside the fast default; serve_ds41.py maps model names ending in `-exact` (no bias) and `-fast` (0.5 / 0.3).
  `STRATA_RB_POOL=M` keeps every pick inside the exact top-M (colibri's CACHE_ROUTE window, arXiv:2412.00099 Max-Rank);
  with `STRATA_RB_SMART=1` (or PROTECT / POOL set) the protect / pool ranks also change per request (`route_protect=`,
  `route_pool=`) and each request logs `routing vs exact: X% of expert picks swapped, Y% of router weight kept`.
  Measured on an RTX 3090 (Windows, 1017 dsx 2-bit, medians of 6 requests on one server): exact 9.7 tok/s; 0.5 / 0.3
  21.7 tok/s, 37.1% swapped; 0.25 / 0.15 19.1 tok/s, 26.1%; protect 2 + pool 12 at 1.0 / 0.6 17.5 tok/s, 18.6%.
- ggml upstream fixes applied (#30103 flash-attn race, #29953 MMQ OOB reads); prefill logits unchanged bit for bit.

## DeepSeek-V4.1-Flash

V4.1-Flash is a different shape from V4-Flash: **40 layers × 384 routed experts (top-6)**, an **Engram** n-gram
memory (hashed rows read from a ~300 GB table on disk at layers 1 and 14), and **shared compressed-KV / indexer
layers** (a few "KV source" layers own the compressed cache; the layers above read it and keep only their own
128-token window). Its experts do not fit in RAM even at 2 bits, so the engine (`tools/v41/engine`) keeps them in
three tiers:

| Tier | What lives there | How it is fed |
| --- | --- | --- |
| VRAM cache | the hottest ~1,300-1,600 experts (sized from measured free VRAM after the prompt) | `--vram-lru`, `--route-bias` |
| Pinned RAM arena | ~6,700 experts (60 GiB, `cudaHostRegister`) | DMA straight to the GPU |
| NVMe file tier | the other ~8,700 experts | O_DIRECT reads at ~4.5 GB/s, prefetched a layer ahead |

A bigger arena does not help on this box: 62 / 65 GiB cut NVMe reads 10% but decode stayed within run-to-run noise
(24.4-25.2 tok/s at 60, 24.5 at 62, 24.7 at 65), because the file penalty already keeps decode routing off the NVMe tier.

### The model file: GSQ mixed tiers (152 GiB)

Download: **[huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)**
(16 byte-split parts + `reassemble.sh`, which verifies and joins them; also the DSpark draft stack + vision tower sidecar).
The upload scripts are in `tools/v41/hf/`.

`tools/v41` holds a GPTQ-style quantisation pipeline (`v41_calib.py`, `v41_alloc.py`, `v41_quant.py`,
`v41_gptq_graph.py`) that gives every expert its own format from calibration statistics: **PTQ1_0** (ternary,
1.75 bpw), **Q2_0** (2.25 bpw) or **Q3_K**, early layers richer (~2.5 bpw) and late layers leaner (~2.0 bpw).
Quality on held-out data against the FP8 original ("teacher"):

| Domain | GSQ ppl | Teacher ppl |
| --- | ---: | ---: |
| All (head gate) | 5.645 | 4.915 |
| agent | 2.118 | 2.121 |
| reasoning | 6.19 | 5.83 |
| code | 1.98 | 1.74 |
| zh | 7.64 | 6.83 |

Top-1 agreement 0.674 vs 0.701. In the engine: 4096-token code prompt ppl 2.380, top-1 0.822.

### Results (RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe, 2026-10-12)

| | First working GPU run (2026-10-11 morning) | Now |
| --- | ---: | ---: |
| Prefill, 8192-token prompt | ~27 tok/s (chunk 512) | **244-248 tok/s** (one-shot) · 250-269 tok/s (server, before MMQ) |
| Prefill, 16K / 20K prompt | - | **288 / 266 tok/s** |
| Decode (256-token prompt, 256 tokens) | 3.4 tok/s | **25.5-26.3 tok/s** (27.7 with the opt-in ggml patch) |
| Decode in a real agent session (20-32K context, server) | - | 22.7-28.2 tok/s |
| Model load | 211 s | ~50 s |

- **The prompt is exact.** Every prefill speed-up keeps the last-token logits byte-identical (8192-token check md5
  `ee31507a0e88`), also through the server and its conversation reuse. The prompt's routed experts run on int8
  tensor cores (MMQ; the ternary experts through a lossless PTQ1_0 -> Q2_0 transcode on the card): +12% over the
  dp4a kernels (`--no-chunk-mmq`, md5 `4d229102872a`) at equal quality (4096-token code ppl 2.3790 vs 2.3799).
- **Decode routes with two selection-only biases** (`--route-bias 1.0`, `--route-bias-file 0.6`, both defaults):
  a bonus for experts already in VRAM and a penalty for experts that would be read from NVMe. Mixing weights stay
  exact and every expert stays reachable. Cost on 4096-token perplexity: code 2.38 -> 2.87, agentic text 15.11 -> 6.39
  (better: the GSQ quant gave its cheapest formats to rarely-used experts). `--route-bias 0 --route-bias-file 0` =
  exact routing at ~3.5 tok/s.
- Where the time goes - prefill (8192 tokens, ~37 s): dense half ~11 s, experts ~26 s (the grouped expert kernels and
  ~10 GB/s of expert uploads). Decode (~38 ms/token): dense half ~19 ms (its Q8_0 weights read at ~880 GB/s - the
  card's bandwidth), expert tier ~16 ms.

How each number was reached, step by step: [docs/ds4/V41_PROGRESS.md](ds4/V41_PROGRESS.md).

### Running the V4.1 engine

Windows (native, MSVC + CUDA): see [WINDOWS.md](WINDOWS.md) for the build, downloads, configs and measured numbers.

```bash
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds41_generate test_ds41_moe_gpu
# once: copy the Engram tables (202.8 GB) off slow storage, one file per drive if needed
python3 tools/v41/extract_engram.py /path/to/DeepSeek-V4.1-Flash-FP8 --to layers.1=/fast/a --to layers.14=/fast/b
# defaults are the tuned config (layer-major prompt, prefetch, enc-skip, prestage, route bias 1.0 + file penalty 0.6)
STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 84 70 -- build-ds4-gpu/ds41_generate -m V41-GSQ.gguf \
     --ids-file prompt.i32 -n 128 --backend cuda --experts gpu --engram "/fast/a:/fast/b" --rank rank.bin
```

`STRATA_NO_LARGEPAGES=1` pins the 60 GiB arena at 1.2 GB/s (load ~50 s) instead of ~0.16 GB/s. Optional +5% decode:
`patch -p1 -d third_party/llama.cpp < tools/v41/ggml/dsv41-fake-quant.patch`, configure with `-DSTRATA_DSV41_FQ_OP=ON`,
and check `build-ds4-gpu/test_ds41_fq --cuda` says bit for bit (it adds one op to ggml; the stock build does not need it). `--ppl` /
`--dump-logits F` / `--dump-last F` for quality checks; `test_ds41_moe_gpu GGUF --layers 0,20,39 --tokens 2 --gpu
--chunk 64` checks the expert tier against a dequantised F32 reference. Full engineering log, every measurement and
the next steps: [docs/ds4/V41_RESUME_PROMPT.md](ds4/V41_RESUME_PROMPT.md).

### Serving it (OpenAI + Anthropic API, tool calls)

```bash
# Strata's server (~/AI/Strata/serve) with V4.1's tokenizer, DeepSeek's chat encoder, DSML tool calls, reasoning effort
~/AI/Strata/.venv/bin/python tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150
# or with QuetzaCodetl in front (starts the server, attaches if it is already up)
tools/v41/serve/strata-ds41-quetza.wrapper
```

`ds41_generate --serve` runs every request the way the one-shot path does; measured at 512K context on a 20K-token
conversation:

| Request | Path | Time |
| --- | --- | --- |
| fresh 20,000-token prompt | layer-major batch, exact | 75 s (266 tok/s) |
| new turn of <= 96 tokens (tool result, short message) | exact passes from the VRAM cache | 41 tokens: 6.9 s |
| new turn of 97+ tokens | layer-major batch, exact (a short batch still streams almost every expert) | 300 / 641 tokens: 18.5 / 18.7 s |
| 11,260-token tool result on 20K reused (a live agent session) | layer-major batch, exact | 49 s (230 tok/s) |
| the same prompt again (regenerate) | snapshot + saved logits | no prefill |

Every prompt token routes exactly by default. Exact passes and the batch agree to KL 0.012 on the next-token
distribution (batch-shape float noise). An opt-in **fast quick path** runs turns of up to 768 tokens with the decode
biases at 40-57 tok/s (300 tokens in 6.4 s), but measurably shifts the output (KL 0.10-0.15):
`STRATA_SERVE_STEP_MAX=768 STRATA_SERVE_PASS_BIAS=1`. A conversation that extends the last request continues from it;
one whose re-rendered history differs (DeepSeek's encoder drops earlier turns' thinking) restores the snapshot taken at
the end of the last prompt. The server writes every request's thinking, answer and tool calls to
`/tmp/strata-ds41-text.log` (`STRATA_TEXT_LOG`) and keeps the last 100 requests at `/api-monitor`.

---

## DeepSeek-V4-Flash (0731) engine

### What this is

[Strata](https://github.com/Niko1221/Strata) is an inference engine specialised for one MoE model
(Qwen3.8-Flash-Next): the hot experts live in a VRAM cache, the rest in pinned RAM, and each layer's misses are
split between the CPU and PCIe at the same time. This fork applies the same idea to **DeepSeek-V4-Flash-0731**, an
architecture llama.cpp supports but which is unusually hard to run fast on a consumer PC:

- 43 layers × 256 experts (top-6), ~73 GiB of experts at 2-bit - far more than VRAM, most of RAM;
- **hyper-connections** (4 residual streams mixed by a 20-iteration Sinkhorn normalisation every layer);
- **compressed sparse attention** (raw 128-token window + 4× compressed blocks chosen by a "lightning indexer" + 128×
  compressed blocks), attention sinks, a grouped output LoRA, hash routing on the first 3 layers, a SwiGLU clamp.

### How fast is it?

Measured on: **RTX 4090 24 GB · Ryzen 7 5700X (8 cores, AVX2) · 90 GB DDR4-3200 · NVMe**, model
`DeepSeek-V4-Flash-Q2-0731.gguf` (80.8 GiB: experts IQ2_XXS gate/up + Q2_K down, dense Q8_0). Decode = greedy
generation, 128 tokens after a 600-token prompt; quality = perplexity over that prompt (lower is better).

**Decode**

| Engine / config | Decode | Perplexity |
| --- | ---: | ---: |
| llama.cpp, best config (`-ngl 99 -cmoe --moe-expert-cache 40 -fa on`) | 16.3 tok/s | - |
| Strata-DS4 2026-10-07 (`--dense-requant q6_k --slots auto`) | 16.8-17.3 tok/s | 10.47-10.64 |
| **Strata-DS4 now** (`--vram-lru --pf-b 0.7 --arena-skip-resident --arena-gib 58`, recommended) | **23.3-24.0 tok/s** | 10.37-10.55 |
| + `--route-bias 0.05` (lossy, see below; measured on the older config: +9% / +1% perplexity) | - | - |

The 2026-10-08 numbers were measured with a video player and other background processes using about a core of the
CPU (the CPU computes part of the expert misses, so a quiet machine should do a little better). Perplexity moves by
up to ~0.2 between identical runs: which experts the CPU and which the GPU computes depends on timing, and the two
paths round differently (q8_1 vs Q8_K activations).

**Prompt processing (prefill)**

| Prompt | decode-loop prefill (before) | `--prefill-chunk 4096 --chunk-mmq --chunk-prestage` |
| --- | ---: | ---: |
| 3,000 tokens | ~18-21 tok/s | 184-259 tok/s |
| 6,000 tokens | ~18-21 tok/s | **359-392 tok/s** |

Chunked prefill is longer-prompt-faster: each chunk streams every expert of every layer over PCIe once (~74 GB), so
the cost per token falls as the chunk fills. It uses MMQ (int8 tensor cores) for the expert products, so perplexity
moves by ~0.3% (6.38 -> 6.39-6.41 on the 3K prompt); without `--chunk-mmq` it is the decode math.

**Long context** (2026-10-08/09)

Measured up to a 128K-token real document (chunked prefill 4096); 256K loads and decodes. DeepSeek-V4's
compressed sparse attention makes long context cheap in principle - each CSA layer attends to a fixed 512 compressed
blocks chosen by its lightning indexer - but the indexer itself scores every block, and that was the cost that grew:

| position | prefill | attention+router per token, before -> now | decode before | decode now |
| --- | ---: | ---: | ---: | ---: |
| 600 | (decode loop) | 15.5 ms -> 15.9 ms | 25.0 tok/s | 23.7-25.3 tok/s |
| 16,384 | 466 tok/s | 28.7 ms -> - | 12.7 tok/s | not re-measured yet |
| 131,072 | 248 tok/s | 48.1 / 38.2 ms -> **17.1-18.7 ms** | 10.4 tok/s | **19.0-22.4 tok/s** (probe, upper bound) |
| 262,144 | - | -> 18.5 ms | - | **21.8 tok/s** (probe, upper bound) |

- "Now" = the fused lightning-indexer kernel (one CUDA kernel for score + ReLU + head weights + mask instead of a
  ~9-op chain with two `[blocks x 64]` copies, on by default off-CPU) and an O(512) instead of O(blocks) sparse row
  selection. The attention column is real; the "now" decode totals come from a `--pos-offset` probe (decoding at a deep
  position over empty caches), whose expert hit rate is a short prompt's (~80%) rather than a long document's
  (53-63% measured), so a real 128K document will land between the two columns. A real-document re-run is next.
- Prefill at 128K was 73% dense-half time; query-blocked raw-window attention (`DS4_RAW_BLOCK`) is in but its long
  prefill timing is not measured yet.

**Compressed KV cache options** (the CSA/HCA compressed-key caches; quality on the 600-token prompt):

| `--comp-type` | bytes vs F32 | perplexity |
| --- | ---: | ---: |
| `f32` (default) | 1x | 10.4468 |
| `q8_0` + `--icomp-q8` | ~0.27x | 10.5031 (+0.54%) |
| **`iq4_nl` + `--icomp-q8`** (recommended for long context) | ~0.14x | **10.5090 (+0.60%)** |
| `q4_0` + `--icomp-q8` | ~0.14x | 10.5358 (+0.85%) |

Below 8 bits the rows are Walsh-Hadamard rotated (128-blocks) before rounding and after the gather (`--comp-rot`
forces it either way); that is why 4-bit costs barely more than 8-bit here. At 256K the 4-bit caches give back a few
GB of VRAM, i.e. a few hundred more expert slots. `--comp-host` keeps the cache in pinned RAM (zero-copy reads); it
works, but at 256K it costs ~3 ms/token for +62 slots - only worth it beyond ~512K.

**Adaptive expert pruning (2026-10-09, experimental)**

`--prune LIST --prune-penalty 0.5 --arena-adapt`: a REAP-scored list of the least useful 25% of experts (calibrated on
English + Chinese + code) is kept out of VRAM and RAM and left on NVMe; the router can still pick one when it wants it
by more than the penalty, and a picked expert is promoted into the RAM arena (least-recently-used out), so the
experts a conversation actually needs move in. On held-out text (9.6K tokens, per-domain perplexity vs unpruned):

| | code / English / Chinese | German | French / Japanese / Portuguese (never calibrated) | all | decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard prune 25% | -1.0 to +1.2% | +72% | +7 to +115% | +25.6% | 19.3 tok/s |
| **soft 25% + adapt** | -1.4 to +0.9% | +3.9% | -0.3 to +0.8% | **+0.3%** | **17.25 tok/s** |
| random 25% (control) | +17 to +29% | +40% | +6 to +35% | +20.7% | 18.9 tok/s |
| unpruned | - | - | - | - | 15.71 tok/s |

Calibration: `ds4_generate --prefill-chunk 4096 --chunk-mmq --saliency sal.bin` on a mixed prompt
(`tools/ds4/make_reap_prompts.py`), then `tools/ds4/reap_prune.py sal.bin --frac 0.25 -o prune.txt`. The saliency
accumulator and `reap_prune.py` come from the Strata-GLM work. Details: [FINDINGS s37](ds4/FINDINGS.md).

The first token's logits match llama.cpp's (same top-1; KL 0.03-0.10, which is the difference between the GPU's and
the CPU's 8-bit activation rounding).

### How it works

Per token, per layer (`tools/ds4/ds4_generate.cpp`):

1. **Predict** (`Ds4Dense::predict`): the layer's router applied to an earlier hidden state guesses the experts
   (~62-70% recall) and the predicted misses start DMA-ing into VRAM on a second CUDA stream **under** the attention.
2. **Attention + router** (`Ds4Dense`, ggml on CUDA, CUDA graphs, fused hyper-connection ops, one input upload per
   token, one readback per layer).
3. **Experts** (`Ds4MoeTier`): hits from the VRAM slot cache (seeded from a routing profile), prefetched experts, a
   share of the misses DMA'd over PCIe and computed on the GPU, the rest computed by the CPU pool straight from a
   pinned RAM arena - all three at once.
4. **Finish**: shared expert + routed sum, hyper-connection post-mix.

What moved the needle on the real model (each step measured, in order): CUDA graphs 5.6 → 9.7 tok/s; llama.cpp's
fused DeepSeek-V4 hyper-connection kernels 11.2; enough VRAM back for 2150 expert slots 12.8; a 70 GiB RAM arena
(no NVMe reads mid-token) 16.3; PCIe share + one upload per token 17.0; then (2026-10-08) reusing the captured CUDA
graphs without ggml's per-call node compare (attention 22 → 17 ms/token) 18.8; an exclusive VRAM LRU (a miss takes the
least-recently-used slot of its layer; hit rate 70 → 84%) 22.3; one shared compute arena for all ~500 attention graph
variants (~850 MiB of VRAM → +120 expert slots) and skipping the compressed-row pooling on tokens that complete no
block (3 of 4 tokens in CSA layers, 127 of 128 in HCA) → 23-24.

**Prefill** (`--prefill-chunk N`): the same per-layer graphs for N tokens at once, built per chunk on a shared
allocator; the expert side streams each layer's experts into VRAM once per chunk and runs them through MMQ, the next
layer's experts DMA in while the current one computes, and the VRAM expert cache is seeded after the prompt.

### Running it

Requirements: Linux, CUDA 13.x toolkit, an NVIDIA GPU with 24 GB, **~80 GB of free RAM**, the GGUF above.

```bash
git clone https://github.com/Indras-Mirror/Strata-DSV4.1 && cd Strata-DSV4.1
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds4_generate          # ggml-cuda takes ~10 minutes the first time

# token ids in, token ids out (tools/ds4/ds4_chat.py does text <-> ids with the GGUF's tokenizer)
# short prompts, fastest decode
bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m DeepSeek-V4-Flash-Q2-0731.gguf \
     --ids 0,1,2 -n 128 --slots auto --pcie 0.55 --dense-requant q6_k --vram-lru --pf-b 0.7 \
     --arena-skip-resident --arena-gib 58 --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
# long prompts: chunked prefill (needs the full arena - do NOT combine with --arena-skip-resident)
bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m DeepSeek-V4-Flash-Q2-0731.gguf \
     --ids-file prompt.i32 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --vram-lru --pf-b 0.7 \
     --prefill-chunk 4096 --chunk-mmq --chunk-prestage --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
python3 tools/ds4/ds4_chat.py "Explain RoPE in two sentences." -n 128 --model DeepSeek-V4-Flash-Q2-0731.gguf -- \
     --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
```

`memguard.sh` runs the engine in a cgroup with a hard RAM cap and swap off, so a misconfiguration kills the process
instead of freezing the desktop - use it. Startup takes ~1.5-2 minutes (the 70 GiB arena is read and pinned).

Useful flags: `--slots N|auto` (VRAM expert cache), `--arena-gib` (pinned RAM arena; the rest is read from the file),
`--pcie F` (share of misses sent over PCIe - **single-token decode only**), `--tier-batch-pcie F` (the same share
for batched passes: prompt chunks and **every serve / DSpark decode and verify call** - see below), `--pf-b` (predicted prefetch per layer), `--dense-requant q4_k|q5_k|q6_k`,
`--vram-lru` (misses take the layer's least-recently-used VRAM slot), `--arena-skip-resident` (the VRAM-seeded
experts stay out of the RAM arena), `--prefill-chunk N` / `--chunk-mmq` / `--chunk-prestage` (prefill),
`--prune F` / `--prune-penalty X` / `--saliency F` / `--arena-adapt` (adaptive expert pruning, see above),
`--comp-type f32|q8_0|q5_0|q4_0|iq4_nl` / `--icomp-q8` / `--comp-rot 0|1` / `--comp-host` (compressed KV cache,
see above), `--pos-offset N` (decode-timing probe at a deep position, no prefill), `--route-bias D` and
`--skip-miss T` (lossy, see below), `--mtp FILE [--verify]` (DeepSeek-V4's MTP draft head:
63% acceptance measured; `--verify` is CPU-tested, not yet tuned on CUDA, and chunked prefill does not fill the MTP
window yet), `--ppl` (perplexity of the prompt), `--dump-logits`, `--temp`, `--ctx`.

**PCIe vs CPU pool for RAM-tier misses.**  An expert that is in the RAM arena but not in VRAM is either copied to the
card over PCIe or computed in place by the CPU pool.  Two knobs set the share that goes over PCIe:

| Flag | Env | Default | Governs |
|---|---|---|---|
| `--pcie F` | - | 0.75 | the single-token decode path (`gpu_run`); **not used by `--serve` with DSpark** |
| `--tier-batch-pcie F` | `STRATA_TIER_BATCH_PCIE` | 1.0 | batched passes (`gpu_run_n_ex`): prompt chunks and every serve / DSpark decode + verify |

So with the served config (DSpark on), `--tier-batch-pcie` is the one that matters, and at its default the CPU pool
never runs.  The default fits an 8-core dual-channel DDR4 desktop (Ryzen 7 5700X: the pool does ~24 GB/s, about the
same as PCIe, so 0.5-0.85 measured -5 % to -8 % or within noise).  On a many-core, many-channel host set it low:
chimpera measured a 24-core Threadripper 5965WX (~150 GB/s) + RTX 5090 on gen4 going from 30 / 44 to 43 / 64 tok/s
(prose / code, DSpark serve) at `--tier-batch-pcie 0`, monotone from 1.0 down, threads plateau at 16+
([#3](https://github.com/Indras-Mirror/Xolotl/issues/3)).  FreeToken's rule of thumb: the pool pays when its expert
bandwidth is well above PCIe's (about 2x).  Sweep 1.0 / 0.5 / 0.0 on your box.

### Cache-aware routing (`--route-bias`, lossy, off by default)
Adds `D` to the router's *selection* score of experts already in VRAM (the mixing weights stay exact), so near-ties
resolve toward the cache. It trades quality for speed - measured: 0.05 → +9% decode / +1.1% perplexity, 0.2 → +29% /
+9.6%. Leave it at 0 unless you have measured it on your own workload.
`--skip-miss T` drops a routed expert that is a VRAM miss and weighs less than T of the token's routing weight
(0.05: ~0.4% of lookups, within run-to-run noise; 0.10: 23.8 tok/s vs 22.3-23.4 for the reference runs, +5% perplexity
- not recommended).

### Verifying it

- `test_ds4_dense` - the dense half vs a CPU reference forward (`ds4_ref`) on small synthetic models: bit-exact on the
  CPU backend; `--cuda` agrees to ~1e-4 (CUDA vs CPU arithmetic).
- `test_ds4_moe` / `test_ds4_moe_gpu --gpu` - the expert tier on real expert slices vs a dequantised F32 reference,
  every source (VRAM slot, prefetch, PCIe, CPU, file), the SwiGLU clamp forced on, and multi-token runs.
- `tools/ds4/compare_logits.py` - the engine's logits vs llama.cpp goldens.
- `DS4_CHECK_GPU=1` recomputes every GPU-computed expert on the CPU and logs the difference; `DS4_CHECK_LRU=1`
  byte-checks every VRAM-LRU swap against the file.
- Chunked prefill is bit-identical to the one-token decode loop on the CPU backend (logits of every prompt position,
  chunks of 1-70 tokens; `DS4_CPU_FA_REF=1` for >= 64-token chunks, whose flash-attention otherwise tiles), and every
  speed change above was checked the same way and mutation-tested (break it on purpose, the gate must fail).

### Roadmap

- **MTP verification on CUDA**: the head drafts at 63% acceptance here (74-93% reported elsewhere); at that rate
  depth 1 is worth ~+10%, deeper drafts lose (each costs ~9 ms and more expert reads).
- Long-context decode on a real document (the probe numbers above are an upper bound): the expert hit rate falls to
  53-63% on long documents; a cross-layer expert cache is the candidate fix.
- The dense half's ~4,500 small kernels per token (attention+router is ~15.5 ms against a ~6 ms weight-read floor).
  In progress: 2026-10-09 cut a CSA layer from 121 to ~107 kernels per token (161 -> 130 when a block completes;
  attention+router 16.5 -> 16.0 ms in the first A/B), all bit-exact on the CPU gates; an nsys profile decides what
  is kernel time and what is launch gaps.
- DSpark (0731's native 5-token drafter) and n-gram drafting on the same verification path.
- A server (OpenAI-compatible) and a merge back into a single multi-model Strata; a MiMo-V2.6-Flash engine is next.

### Credits

Built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT). Uses
[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), including its DeepSeek-V4 hyper-connection kernels,
as the reference implementation. DeepSeek-V4-Flash by [DeepSeek](https://huggingface.co/deepseek-ai). The engine's
design notes and every measurement behind the numbers above are in [docs/ds4/](ds4/) (`FINDINGS.md`).

License: MIT (see [LICENSE](LICENSE)).
