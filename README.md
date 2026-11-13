<p align="center"><img src="docs/brand/assets/readme-header-1200.png" alt="Xolotl - frontier MoE models on one gaming PC" width="100%"></p>

<h1 align="center">Xolotl</h1>

<p align="center"><b>DeepSeek-V4.1-Flash, all 552 billion parameters, on one gaming PC at 33 tok/s on real agent tasks (41 in fast mode)</b>, alongside GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B, LFM3-Omni and Muse Glimmer 30B, all behind one OpenAI- and Anthropic-compatible server. Built on <a href="https://github.com/Niko1221/Strata">Strata</a> and heavily extended. Linux · NVIDIA · MIT.</p>

> **Built from [Strata](https://github.com/Niko1221/Strata), and grown a long way past it.** Niko1221's engine,
> server and core kernels are still at the heart of it (merged at 0.1.44.1), but Xolotl adds more than 70,000 lines
> on top: its own DeepSeek-V4.1, V4-Flash and GLM-5.3-Flash engines, the Qwen3.8-27B dense engine (grown from Mirko
> Perrone's port), the LFM3-Omni engine, fast mode, Xolotl Direct, and the quantisation, expert-tiering and serving
> work behind every model here.

## Latest

- **14 Nov** - Qwen3.8-27B with DFlash2: the 8-token verify pass got cheaper, **171 -> 204 tok/s** on a best-case 2K
  prompt (our 13.7 GiB XQ quant, one 4090). The dense engine now reads one position of all 8 drafted tokens per load
  and uses the multi-column layout the other engines already had. Same text, bit for bit; default on. Real agent-task
  numbers next. [CHANGELOG](docs/xolotl/CHANGELOG.md#qwen38-27b-with-dflash2-the-8-token-verify-gets-cheaper-171---204-toks-14-nov).
- **13 Nov** - [Muse Glimmer 30B](docs/GLIMMER.md) (Meta, dense) natively at its full 512K context on one 24 GB
  card: 47 tok/s decode, 1.3-8x faster with Meta's DFlash drafter (up to 379 tok/s), 3,100 tok/s prefill, tool calls, reasoning and pictures through the usual server.
  [CHANGELOG](docs/xolotl/CHANGELOG.md#muse-glimmer-30b-natively-at-512k-13-nov).
- **13 Nov** - Keirolabs joins Exa, Perplexity and DeepSeek as a keyed web search choice in Settings > Tools (your own
  key; nothing changes without one). By Manasbh (#136).
  [CHANGELOG](docs/xolotl/CHANGELOG.md#keirolabs-as-a-keyed-web-search-service-13-nov).
- **11 Nov** - An NVIDIA L20 (Ada, 92 SMs) now takes row tables measured on itself instead of an RTX 4070/4080's (`891`):
  15 multi-column cells and two one-column cells it wins with, and a card that stops running 22 splits it loses with. Same
  text, bit for bit; default on, `STRATA_MMVQ_IL_891=0` turns it off.
  [CHANGELOG](docs/xolotl/CHANGELOG.md#an-nvidia-l20-gets-row-tables-it-was-measured-on-11-nov).
- **11 Nov** - Qwen3.8-27B reads prompts 19-29 % faster with its draft head on (30K prompt: 1,067 -> 1,394 tok/s):
  a prompt token now leaves only its K/V rows in the draft block instead of running the whole block. Same text, bit for
  bit; default on. [CHANGELOG](docs/xolotl/CHANGELOG.md#qwen38-27b-reads-prompts-19-29--faster-with-its-draft-head-on-11-nov).
- **10-11 Nov** - Eleven fixes: a Pillow JPEG 2000 security hole closed for picture requests, two prompt-path faults
  (`Xid 31` on long prompts, an expert read before its upload landed), NaN routers, misspelt tool-call openers, a hung
  streaming client no longer blocking the queue, the sm_70 build.
  [CHANGELOG](docs/xolotl/CHANGELOG.md#fixes-ported-10-11-nov-correctness-security-tool-calls).
- **11 Nov** - Opt-in `"reasoning_loop_recovery": "wrap"`: thinking that repeats whole passages is closed and the
  model answers, on any model (the older modes did nothing on DeepSeek-V4.1 or ended with no answer). Not yet measured
  on a model.
- **10 Nov** - `xolotl-dense` runs the gated residual (hyper-connections), the Qwen4 generation's replacement for every
  layer norm - the piece the dense engine was missing for a Qwen4-27B dense file, built and checked before that file
  exists. Same kernels the MoE path uses, plain-residual models untouched, and a file carrying a feature the engine
  lacks (a QSA indexer, PLE) is refused by name. Bit-identical across batching on a synthetic model; parity against a
  reference forward pass needs the weights, which do not exist yet.
  [CHANGELOG](docs/xolotl/CHANGELOG.md#xolotl-dense-runs-the-gated-residual-hyper-connections-10-nov).
- **8 Nov** - Short agent turns (a tool result on a long conversation) are faster on DeepSeek-V4.1 (4.40 -> 3.83 s for
  161 tokens) and GLM-5.3-Flash (5.47 -> 4.63 s for 246 tokens): the CPU computes some of the experts while the card
  runs the rest, and the CPU's i-quant kernels now decode an expert once for all its tokens (-32 % at 4-8 tokens).
  GLM-5.3-Flash decode is 28.5-29 tok/s since the RAM upgrade. Default on; numbers in the
  [CHANGELOG](docs/xolotl/CHANGELOG.md#cpu-i-quant-down-rows-one-decode-for-every-token-8-nov).
- **8 Nov** - Qwen3.8-27B decode: MTP now stays on with thinking off (52 -> 111 tok/s), one-pass verify attention
  (+7-10 %) and an n-gram draft chain (`--ngram-chain`: 137 -> 173 tok/s when rewriting a file). Same text, bit for bit.
- **8 Nov** - [Gauntlet v3](#what-one-pc-can-do) is the main table: every model on the same 6 hard agentic tasks. Our DeepSeek-V4.1 and Qwen3.8-Flash-Next builds score at or above their own full-precision APIs on it.
- **8 Nov** - The Qwen3.8-27B dense engine is `xolotl-dense` now (built on Mirko Perrone's dense port); `strata-dense`
  stays as a symlink, so nothing breaks.
- **7 Nov** - [DeepSeek-V4.1 fast mode](#fast-mode-deepseek-v41) (opt-in): 41 tok/s on real agent tasks against 32.6,
  the same Gauntlet score.
- **7 Nov** - [Xolotl Direct](docs/XOLOTL_DIRECT.md) (Linux, experimental): the NVMe drive writes experts straight into
  a GeForce card's VRAM, 3.1 ms per expert instead of 11.
- **7 Nov** - DeepSeek-V4.1 loads in 44 s instead of 96, same output.
- **6 Nov** - +8 % decode on RTX 40 cards, +13-17 % prefill on IQ1_M layers, +11 % DeepSeek-V4.1 long-prompt prefill
  (ideas from Basalt). All on by default and bit-exact.
- **2-3 Nov** - [LFM3-Omni-2B](docs/LFM3.md) natively (text, pictures, audio and video in, speech out), and as the
  ears and voice of Qwen3.8-Flash-Next on one 4090.

Everything else, with PR numbers and who did it: [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md).

## Fast mode (DeepSeek-V4.1)

DeepSeek-V4.1's decode speed is set by how many expert bytes cross the RAM bus and the NVMe drive per token. Fast mode
keeps a 2-bit (Q2_K) copy of the two biggest expert formats (about half of those bytes) in a ~40 GB sidecar file and
reads that instead. The model file is not changed; turning it off is unsetting one variable.

| DeepSeek-V4.1 V4, RTX 4090 | Decode, agent tasks (tok/s) | Decode, 8K prompt (tok/s) | Gauntlet score | Next-token agreement |
| -------------------------- | --------------------------: | ------------------------: | :------------: | :------------------: |
| normal                     |                        32.6 |                      26.1 |      4.99      |        87.0 %        |
| **fast mode**              |            **41.1 (+26 %)** |          **33.0 (+27 %)** |    **4.99**    |      **86.2 %**      |

```bash
build-xolotl/ds41_generate -m V41-XQ-1017-v4.gguf --build-shadow /nvme/V4.fast.x41s   # once, ~30 min CPU (or download v4-fast/ from the model's HF page)
XOLOTL_SHADOW=$(cat /nvme/V4.fast.x41s.list)  <your usual serve command>
```

It is lossy, so it is opt-in: on the agent tasks it cost nothing measurable, but it changes about one token in a
hundred, so keep normal mode where every token counts. Details: [docs/XOLOTL_FAST.md](docs/XOLOTL_FAST.md). The idea
is compress-then-correct from [PROMETHEVS](https://github.com/Magnvs7/Promethevs) (Ryan Flavell).

## What one PC can do

**One RTX 4090 (24 GB) · Ryzen 7 5700X · 110 GB DDR4 · one NVMe drive.** Every number is measured on that machine,
through the real server, while the model works through the [Gauntlet](tools/gauntlet/README.md) v3-small: 6 hard
agentic coding tasks graded by hidden tests, 10 turns each, thinking on (32K-token budget), each model as it is
served. Decode is the generation speed during those tasks.

| Model                                                                                                             | Size      | Score (of 6) | Decode (tok/s) | Prefill (tok/s) | Context | API score |
| ----------------------------------------------------------------------------------------------------------------- | --------- | :----------: | :------------: | :-------------: | :-----: | :-------: |
| **[DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)**     | 552B MoE  |     4.99     |      32.6      |     289-409     |   512K  |    4.49   |
| **DeepSeek-V4.1 V4, [fast mode](#fast-mode-deepseek-v41)**                                                        | 552B MoE  |     4.99     |      41.1      |     258-439     |   512K  |     -     |
| **GLM-5.3-Flash**                                                                                                 | 320B MoE  |   to come    |     28.5-29    |     312-450     |   512K  |    5.87   |
| **[Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)** | 125B MoE  |     4.49     |      157       |   3,800-4,070   |   200K  |    3.90   |
| **Qwen3.8-27B**                                                                                                   | 27B dense |     5.50     |      105       |   1,420-1,630   |   200K  |    4.47   |

<sub>Score (of 6) = each task's progress summed (1 per solve, partial credit from the hidden tests), averaged over the runs.
API score = each model's own full-precision API through OpenRouter (DeepSeek-V4.1-Flash-1017, GLM-5.3-Flash, Qwen3.8-Flash,
Qwen3.8-27B), the same tasks and turns, reasoning on (8 Nov).
Qwen3.8-27B: one run on 8 Nov after that day's decode fixes (MTP with thinking off, one-pass verify attention,
`--ngram-chain`); before them it scored 4.47 twice at 89 tok/s. GLM-5.3-Flash: no Gauntlet run yet (score to come); its decode is from an agent replay on 8 Nov (a 25K-token
conversation with tool-output turns, Maya 512K as served; since the RAM upgrade every expert sits in RAM, 23 tok/s before it),
prefill 454 tok/s on a fresh 25K prompt.
Prefill = a fresh 6K-17K-token prompt (`tools/throughput.py`), DeepSeek-V4.1's with its dense half at Q8_0. In the
Gauntlet DeepSeek-V4.1 runs with its dense half at Q6_K, as it is served. More numbers, the Q8_0 references, the speed log and the older HumanEval table (every model 19-21/21
there, so it no longer separates them): [docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md).</sub>

**Smaller cards.** The engine fits the work to the card, and what doesn't fit lives in RAM and on the SSD, so RAM
matters as much as VRAM. Community runs on 12 GB cards: DeepSeek-V4.1 at 8-10 tok/s (RTX 3060, 46 GB RAM),
Qwen3.8-Flash-Next at 39-48 tok/s (RTX 3060 / 4070). The runs, and what your machine needs:
[docs/WILL_IT_RUN.md](docs/WILL_IT_RUN.md).

## Quick start

Linux (or Windows 11 for DeepSeek-V4.1, [docs/WINDOWS.md](docs/WINDOWS.md)), an NVIDIA GPU, CUDA 12.x/13.x, and as much RAM as you can.

```bash
git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
bash tools/xolotl/fetch-llama.sh   # llama.cpp at the pinned commit + tools/xolotl/ggml/xolotl-ggml.patch -> third_party/llama.cpp
cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_DSV41_FQ_OP=ON \
  -DSTRATA_DSV41_HC_MIX=ON
ninja -C build-xolotl      # strata, xolotl-dense, glm_generate, ds41_generate, ds4_generate
```

Get a model:

- **DeepSeek-V4.1-Flash:** the **[1017 builds](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)**
  of the October checkpoint (V4 recommended, with `v4/rank-1017-v4.bin`), or the earlier
  [GSQ build](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored) (152 GiB, mixed ternary /
  2-bit / 3-bit). Both cards have the download and Engram steps. Optional for V4: the
  [fast-mode](#fast-mode-deepseek-v41) sidecar (~40 GB), built once with `ds41_generate --build-shadow`.
- **Qwen3.8-Flash-Next:** **[XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)**
  (the two `.gguf` shards plus `pack-qfn45/`).
- **Qwen3.8-27B:** any dense Qwen3.8-27B GGUF. `python tools/dense_setup.py --download UD-Q4_K_M --build --arch 89` fetches
  one and writes its run config; add `--mtp --ngram-chain` to its engine args for the fast (still exact) decode.

Then serve it (edit the model paths in the DeepSeek and GLM configs first):

```bash
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150           # DeepSeek-V4.1-Flash
XOLOTL_SHADOW=$(cat /nvme/V4.fast.x41s.list) python3 tools/v41/serve/serve_ds41.py ...                                # the same in fast mode (V4)
python3 tools/glm/serve/serve_glm.py  --engine strata --config tools/glm/serve/strata-glm-maya-abl-512k.json --port 8144  # GLM-5.3-Flash
./scripts/xol-qfn.sh --serve-only --model <shard 1 .gguf> --pack <pack-qfn45>                                       # Qwen3.8-Flash-Next (port 8099)
python3 -m serve.server --engine strata --config <your-27b.json> --port 8090                                          # Qwen3.8-27B (exe: build-xolotl/xolotl-dense)
```

Then point any OpenAI- or Anthropic-compatible client at `http://127.0.0.1:<port>/v1`.

## How it works, in one paragraph

A mixture-of-experts model only uses a few experts per token. Xolotl keeps every expert somewhere - the hottest in
VRAM, tens of GiB in a pinned RAM arena the GPU reads by DMA, the rest on NVMe read at the drive's ceiling - and
moves the ones each token's router asks for, a layer ahead. Each model adds its own tricks: DeepSeek-V4.1 gets a
calibrated mixed-precision quantisation, speculative decoding with exact sampling, a bit-exact 4-bit KV cache for 512K
context, an arena that adapts to the conversation, an opt-in fast mode that reads a 2-bit copy of its biggest experts,
and Xolotl Direct, which lets the NVMe drive write experts straight into VRAM; GLM-5.3-Flash gets its abliteration LoRA on every path and soft
pruning; the Qwen models get MTP speculative decoding. **[Read the full write-up](docs/xolotl/HOW_IT_WORKS.md).**

## Read more

| Doc                                                | What's in it                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| [How Xolotl works](docs/xolotl/HOW_IT_WORKS.md)    | the three memory tiers, per-model engines, the server, the agent-loop work        |
| [Will it run on my machine?](docs/WILL_IT_RUN.md)  | GPU, RAM and disk needs for DeepSeek-V4.1                                         |
| [Benchmarks](docs/xolotl/BENCHMARKS.md)            | every measured number, exactness checks, how to measure your own PC               |
| [Changelog](docs/xolotl/CHANGELOG.md)              | everything that landed, newest first, with the PR numbers and who did it          |
| [DeepSeek-V4.1-Flash engine](docs/DEEPSEEK_V41.md) | GSQ quantisation, DSpark, the fp4 KV cache, Engram                                |
| [DeepSeek-V4.1 fast mode](docs/XOLOTL_FAST.md)     | the 2-bit expert sidecar: build, serve, switches, what it costs and what we tried |
| [DeepSeek-V4.1 on Windows](docs/WINDOWS.md)        | native MSVC + CUDA build, configs, RTX 50 notes                                   |
| [GLM-5.3-Flash engine](docs/glm)                   | KDA + MLA attention, the LoRA on routed experts, soft pruning                     |
| [LFM3-Omni engine](docs/LFM3.md)                   | pictures, audio, video and speech on a 2B model, park/unpark beside a big one     |
| [Muse Glimmer engine](docs/GLIMMER.md)             | Meta's dense 30B at 512K on 24 GB, its channel / ATEM tool-call format, pictures  |
| [Qwen3.8-27B dense howto](docs/dense/HOWTO.md)     | dense vs MoE paths, binaries, configs, MTP (by Mirko Perrone, also in Italian)    |
| [Setting up with an agent](AGENTS.md)              | the build-and-serve steps for an AI agent (or a person) setting Xolotl up         |
| [Qwen3.8 / upstream Strata](README.strata.md)      | the Strata engine this grew from                                                  |
| [History](docs/history/README.md)                  | earlier session notes, kept as written                                            |
| [Brand kit](docs/brand/XOLOTL_BRAND.md)            | logo, banner, palette                                                             |

## Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

Built on **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp),
with the Qwen3.8-27B dense port by [perronemirko](https://github.com/perronemirko/Strata). Model weights by DeepSeek,
Zhipu AI, Qwen, Liquid AI and Meta, with quants and fine-tunes by peasantsmith, ukisai, d0xin and gcsa-abliterix. Thanks to
every contributor: NektoDron, Xiaomi2008, AlexeyMatskevich, paul123637-lab, kelheor, exye137, penink-WH, cpacsm75,
jesdga95 (Basalt) and Ryan Flavell (PROMETHEVS). Who did what: **[CREDITS.md](CREDITS.md)**.

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe. Research code from one developer, measured on one PC; reports from other hardware welcome. MIT.

## Support

If Xolotl's been useful to you and you want to help support what I do, you can shout me a coffee on
[Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)

Write-ups, benchmarks and news about Xolotl: **[indrasmirror.au](https://indrasmirror.au/blog-xolotl-frontier-models-one-gaming-pc)**.
