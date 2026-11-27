<p align="center"><img src="docs/brand/assets/readme-header-1200.png" alt="Xolotl - frontier MoE models on one gaming PC" width="100%"></p>

<h1 align="center">Xolotl</h1>

<p align="center"><b>DeepSeek-V4.1-Flash, all 552 billion parameters, on one gaming PC at 33 tok/s on real agent tasks (41 in fast mode)</b>, alongside GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B, LFM3-Omni, Muse Glimmer 30B and <b>Inkling-2-Mini (127 tok/s, 512K context on one 24 GB card)</b>, all behind one OpenAI- and Anthropic-compatible server. Built on <a href="https://github.com/Niko1221/Strata">Strata</a> and heavily extended. Linux · NVIDIA · MIT.</p>

> **Built from [Strata](https://github.com/Niko1221/Strata), and grown a long way past it.** Niko1221's engine,
> server and core kernels are still at the heart of it (merged at 0.1.44.1), but Xolotl adds more than 70,000 lines
> on top: its own DeepSeek-V4.1, V4-Flash and GLM-5.3-Flash engines, the Qwen3.8-27B dense engine (grown from Mirko
> Perrone's port), the LFM3-Omni engine, the Inkling-2-Mini engine (looped MoE, 512K on 24 GB), fast mode, Xolotl Direct, and the quantisation, expert-tiering and serving
> work behind every model here.

## Latest

One line each; the detail, numbers and credits are in [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md).

- **27 Nov** - Second card as an expert tier, faster: DS4.1 on a 4090 + 3060 prefill 231 -> 253 tok/s, decode 25 -> 28-30; Engram tables in IQ4_NL (202 -> 110 GB, decode +7 %). [more](docs/xolotl/CHANGELOG.md#second-card-expert-tier-faster-engram-tables-in-iq4_nl-27-nov)
- **27 Nov** - [ERNIE-5.5-Flash](docs/ERNIE.md) preview, now served: OpenAI- and Anthropic-compatible server with tool calls and thinking on or off (`configs/xol-ernie.json`), ~14 tok/s decode and 115-120 tok/s prefill on one RTX 4090 with our [XQ file](https://huggingface.co/IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored) (99.9 GB). Speed work (DSpark, GPU prefill) is next.
- **27 Nov** - Windows: the rest of #148 (POSIX shims for MSVC, positional reads in `ds4_moe`, the LFM3 speech helper, the GLM census), thanks mehzhure (#149-#154). [more](docs/xolotl/CHANGELOG.md#windows-the-rest-of-148-lands-27-nov)
- **27 Nov** - Trellis-coded tensor types TQP5/6/7 in Xolotl's ggml (decode kernels, opt-in `XOLOTL_TQP_MMQ=1` prompt kernel). [more](docs/xolotl/CHANGELOG.md#trellis-coded-tensor-types-tqp567-27-nov)
- **26 Nov** - [ERNIE-5.5-Flash](docs/ERNIE.md) preview: Baidu's 280B MoE on its own engine, `ernie_generate`, reading llama.cpp PR #31925 GGUFs (checked against the PR: KL 0.004).
- **24 Nov** - Windows fixes for the V4.1 engine + picture reader (thanks kelheor, #144-#146) and a tool-call loop guard (from Delta / Project Maya). [more](docs/xolotl/CHANGELOG.md#windows-fixes-for-the-deepseek-v41-engine-and-its-picture-reader-a-tool-call-loop-guard-in-the-server-24-nov)
- **24 Nov** - Inkling-2-Mini LoRA training behind the real 23K-token harness prompt, bit for bit with the engine: held-out loss -15.6 %, LiveCodeBench level. [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-lora-training-behind-the-real-harness-prompt-prompt-carry-bit-for-bit-with-the-engine-24-nov)
- **23 Nov** - [LoRA training](docs/LORA_TRAINING.md#inkling-2-mini-the-whole-model-on-one-4090) for the whole Inkling-2-Mini on one RTX 4090 (-16.7 % held-out loss in 100 steps), served with `--lora`. [more](docs/xolotl/CHANGELOG.md#lora-training-for-inkling-2-mini-the-whole-model-on-one-rtx-4090-and---lora-at-serve-time-23-nov)
- **23 Nov** - Inkling-2-Mini: `--vram-budget-mib` for 12-16 GB cards works again, 96 fewer decode launches a token, numeric `reasoning_effort`.
- **22 Nov** - Inkling-2-Mini decode 118 -> **127 tok/s** (fused loop controllers, folded expert scales). [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-decode-118---127-toks-fused-loop-controllers-folded-expert-scales-bf16-router-22-nov-late)
- **22 Nov** - Inkling-2-Mini at **512K context** on one 4090: `k8v4` on the fast kernels, 477K-token decode 30 -> 55 tok/s. [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-512k-context-on-one-rtx-4090-with-k8v4-on-the-fast-kernels-4-bit-v-leg-hadamard-rotated-22-nov-evening)
- **22 Nov** - Inkling-2-Mini XQ re-released as **"Unbound"** ([HF](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored)): KL 0.028, MATH-24 88.9 %. [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-xq-re-released-as-unbound-22-nov)
- **22 Nov** - Inkling-2-Mini prefill 1.2K -> **4.2K tok/s** at 32K, decode 93 -> 113 (136 with DSpark). [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-prefill-12k---42k-toks-at-32k-decode-93---113-toks-dspark-136-toks-22-nov)
- **21 Nov** - [Inkling-2-Mini](docs/INKLING.md) on its own engine, `xolotl-inkling`: 93 tok/s decode on one 4090, plus our 13.93 GB XQ quant. [more](docs/xolotl/CHANGELOG.md#inkling-2-mini-decode-67---93-toks-fused-conv-situ-and-stream-stack-kernels-21-nov-evening)
- **21 Nov** - [LoRA training](docs/LORA_TRAINING.md) on the served MoE models, proven end to end on QFN. [more](docs/xolotl/CHANGELOG.md#lora-training-proven-end-to-end-on-qfn-a-harness-adapter-trained-from-the-served-engines-activations-21-nov-later)
- **19 Nov** - DeepSeek-V4.1 decode switches on by default: agent tasks ~40 -> **42.7 tok/s**. [more](docs/xolotl/CHANGELOG.md#deepseek-v41-decode-switches-on-by-default-agent-tasks-427-toks-19-nov)
- **18 Nov** - Looped transformers in the dense engine (SenseNova-Loop-9B). [more](docs/xolotl/CHANGELOG.md#looped-transformers-in-the-dense-engine-sensenova-loop-9b-18-nov)
- **14 Nov** - Qwen3.8-27B with DFlash2: 171 -> **204 tok/s**, same text bit for bit. [more](docs/xolotl/CHANGELOG.md#qwen38-27b-with-dflash2-the-8-token-verify-gets-cheaper-171---204-toks-14-nov)
- **13 Nov** - [Muse Glimmer 30B](docs/GLIMMER.md) at 512K on one 24 GB card, up to 379 tok/s. [more](docs/xolotl/CHANGELOG.md#muse-glimmer-30b-natively-at-512k-13-nov)
- **8 Nov** - [Gauntlet v3](#what-one-pc-can-do) is the main table. **7 Nov** - [V4.1 fast mode](#fast-mode-deepseek-v41) (41 tok/s on agent tasks) and [Xolotl Direct](docs/XOLOTL_DIRECT.md).

Everything else, in full, with PR numbers and who did it: [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md)
(web search choices, the L20 tables, the 10-11 Nov fixes, CPU expert splits, LFM3-Omni and more).

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
| **Qwen3.8-27B** (Swift-1.5 GSQ-RCO, 13.2 GB)                                                                      | 27B dense |     5.50     |      105       |   1,420-1,630   |   200K  |    4.47   |
| **[Qwen3.8-27B Swift-1.5 XQ 14 GB](https://huggingface.co/IndrasMirror/Qwen3.8-27B-Swift-1.5-XQ-Uncensored)** (ours, uncensored) | 27B dense |     6.00     |      120       |   1,840-2,120   |   200K  |     -     |
| **[Muse Glimmer 30B](docs/GLIMMER.md)** (heretic IQ4_NL)                                                          | 30B dense |     4.89     |      148       |   3,330-3,410   |   512K  |     -     |
| **[Inkling-2-Mini XQ Unbound 13.9 GB](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored)** (ours, uncensored, [engine](docs/INKLING.md)) | 26B looped MoE, 3B active | 4.42 | 116.5 | 3,500-4,190 | 512K | - |

<sub>Inkling-2-Mini: one run on 22 Nov (solved 3/6, 09-dst-scheduler at 0.92), k8v4 K/V, plain decode, decode averaged over
193K generated tokens; on the full 7-task Gauntlet v3 it solved 4/7. It is not on OpenRouter.
Score (of 6) = each task's progress summed (1 per solve, partial credit from the hidden tests), averaged over the runs.
API score = each model's own full-precision API through OpenRouter (DeepSeek-V4.1-Flash-1017, GLM-5.3-Flash, Qwen3.8-Flash,
Qwen3.8-27B), the same tasks and turns, reasoning on (8 Nov).
Qwen3.8-27B: one run on 8 Nov after that day's decode fixes (MTP with thinking off, one-pass verify attention,
`--ngram-chain`); before them it scored 4.47 twice at 89 tok/s. Qwen3.8-27B Swift-1.5 XQ 14 GB: our XQ quant (3.8 bpw) of UkisAI's Swift 1.5 after our own ARA abliteration, MTP 3 + n-gram, 15 Nov: 6/6 solved in 73K tokens and 10.4 min, the first 27B run to solve task 04; our 13.4 GB XQ of the same model scored 4.44 (3/6, 127K tokens), so 3.8 is the served default. The Gauntlet decodes greedily (temperature 0), so a rerun repeats the same tokens (three runs: 73,190 tokens each, 118-123 tok/s): every Gauntlet row here is one greedy run per model. `TASKEVAL_TEMPERATURE=served TASKEVAL_SEED=N` gives sampled runs to average. It fits 200K context, MTP and the vision encoder on the 24 GB card at once (peak 23.7 GB). Muse Glimmer 30B: one run on 14 Nov with `--ngram 8 --dflash`, as served. GLM-5.3-Flash: no Gauntlet run yet (score to come); its decode is from an agent replay on 8 Nov (a 25K-token
conversation with tool-output turns, Maya 512K as served; since the RAM upgrade every expert sits in RAM, 23 tok/s before it),
prefill 454 tok/s on a fresh 25K prompt.
Prefill = a fresh 6K-17K-token prompt (`tools/throughput.py`), DeepSeek-V4.1's with its dense half at Q8_0. In the
Gauntlet DeepSeek-V4.1 runs with its dense half at Q6_K, as it is served. More numbers, the Q8_0 references, the speed log and the older HumanEval table (every model 19-21/21
there, so it no longer separates them): [docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md).
Qwen4-Max on OpenRouter (16 Nov, the same 6 tasks / 10 turns as below): **5.50/6** greedy and **5.50/6** at
temperature 1.0, 47K completion tokens, 56.8 tok/s e2e, ~$0.34 a run — the baseline the local Qwen4 quant will be
scored against. Reaching for `reasoning_effort=max` made it *worse* (4.50/6, 145K tokens, 3.4x the wall time: it
reasoned over `04-log-needle` for all ten turns and never wrote the answer file):
[arms, raw JSON and throughput](docs/benchmarks/2026-11-16-qwen4-max-openrouter-gauntlet-v3-small.md).</sub>

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
| [Prune masking](docs/EXPERT_KEEP.md)               | `STRATA_EXPERT_KEEP` task profiles: keep lists, how to build and gate one, results |
| [LFM3-Omni engine](docs/LFM3.md)                   | pictures, audio, video and speech on a 2B model, park/unpark beside a big one     |
| [Muse Glimmer engine](docs/GLIMMER.md)             | Meta's dense 30B at 512K on 24 GB, its channel / ATEM tool-call format, pictures  |
| [Inkling-2-Mini engine](docs/INKLING.md)           | Thinking Machines' looped MoE: 512K on 24 GB, DSpark drafter, pictures, our quant |
| [ERNIE-5.5-Flash engine (preview)](docs/ERNIE.md)  | Baidu's 280B latent MoE: our [XQ file on HF](https://huggingface.co/IndrasMirror/ERNIE-5.5-Flash-XQ-Uncensored), which GGUFs it reads, flags |
| [Qwen3.8-27B dense howto](docs/dense/HOWTO.md)     | dense vs MoE paths, binaries, configs, MTP (by Mirko Perrone, also in Italian)    |
| [Setting up with an agent](AGENTS.md)              | the build-and-serve steps for an AI agent (or a person) setting Xolotl up         |
| [Qwen3.8 / upstream Strata](README.strata.md)      | the Strata engine this grew from                                                  |
| [History](docs/history/README.md)                  | earlier session notes, kept as written                                            |
| [Brand kit](docs/brand/XOLOTL_BRAND.md)            | logo, banner, palette                                                             |

## Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

Built on **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp),
with the Qwen3.8-27B dense port by [perronemirko](https://github.com/perronemirko/Strata). Model weights by DeepSeek,
Zhipu AI, Baidu, Qwen, Liquid AI, Meta and Thinking Machines, with quants, fine-tunes and refusal edits by peasantsmith, ukisai,
d0xin, gcsa-abliterix, nokeep and unsloth. Thanks to
every contributor: NektoDron, Xiaomi2008, AlexeyMatskevich, paul123637-lab, kelheor, exye137, penink-WH, cpacsm75,
jesdga95 (Basalt) and Ryan Flavell (PROMETHEVS). Who did what: **[CREDITS.md](CREDITS.md)**.

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe. Research code from one developer, measured on one PC; reports from other hardware welcome. MIT.

## Support

If Xolotl's been useful to you and you want to help support what I do, you can shout me a coffee on
[Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)

Write-ups, benchmarks and news about Xolotl: **[indrasmirror.au](https://indrasmirror.au/blog-xolotl-frontier-models-one-gaming-pc)**.
