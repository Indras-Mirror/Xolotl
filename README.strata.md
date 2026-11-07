<h1 align="center">Strata</h1>

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Apple Silicon Macs (M1 to M5)**, built from source by `./setup.sh` (the Metal engine): [Mac](docs/MAC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the Strata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md).

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `BENCH.bat` (`./bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):**
  4- to 8-bit, closer to the full model (93-95% the same next token as Q8_0). Fast where every expert fits in RAM
  (~88 GB for UD-Q4_K_XL up to ~136 GB for Q8_0); with less RAM Strata reads part of them from the SSD while it
  answers, which is much slower (7-8.5 tokens/s for UD-Q4_K_XL on a 64 GB PC).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (Strata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

---

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

How each number was reached, step by step: [docs/ds4/V41_PROGRESS.md](docs/ds4/V41_PROGRESS.md).

### Running the V4.1 engine

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
the next steps: [docs/ds4/V41_RESUME_PROMPT.md](docs/ds4/V41_RESUME_PROMPT.md).

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

