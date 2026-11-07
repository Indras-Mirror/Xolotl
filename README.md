<p align="center"><img src="docs/brand/assets/readme-header-1200.png" alt="Xolotl - frontier MoE models on one gaming PC" width="100%"></p>

<h1 align="center">Xolotl</h1>

<p align="center"><b>DeepSeek-V4.1-Flash - all 552 billion parameters - on one gaming PC, at 33 tok/s on real agent tasks (41 in fast mode).</b><br>
Plus GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B and LFM3-Omni: one engine, one OpenAI / Anthropic-compatible server.<br>
Built on <a href="https://github.com/Niko1221/Strata">Strata</a> and heavily extended · Linux · NVIDIA · MIT</p>

> **A Strata fork.** Xolotl keeps [Strata](https://github.com/Niko1221/Strata)'s engine, server and kernels at its
> core (merged at 0.1.44.1) and adds ~71,000 lines on top: the DeepSeek-V4.1 and V4-Flash engines, the GLM-5.3-Flash
> engine and the Qwen3.8-27B dense port, each with its own quantisation and expert-tiering work.

## Latest

Full notes for every item: [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md).

- **7 Nov - DeepSeek-V4.1 fast mode (opt-in): 41 tok/s decode on real agent tasks, against 32.6 normally** (+26 %, the
  Gauntlet on the 4090), solving the same tasks (8/12 in both modes); short34 top-1 86.2 % vs 87.0 %. A 2-bit (Q2_K) copy of the biggest experts in a sidecar file.
  [Fast mode](#fast-mode-deepseek-v41)
- **7 Nov - Xolotl Direct (Linux, experimental, opt-in): NVMe straight into VRAM on a GeForce card.** Experts read
  from the drive skip system RAM: one expert 11 -> 3.1 ms, zero RAM-bus traffic, the same output. With the
  layer-ahead VRAM stash: +3 % decode on DeepSeek-V4.1. Needs Resizable BAR and two small kernel modules.
  [Setup](docs/XOLOTL_DIRECT.md)
- **7 Nov - DeepSeek-V4.1 loads in 44 s instead of 96 s**, same output. The converted dense weights are cached on disk
  after the first load, and the RAM arena is pinned while it fills. On by default. Every switch now also takes an
  `XOLOTL_` name (`XOLOTL_ARENA_PIN_AFTER=0` = `STRATA_ARENA_PIN_AFTER=0`; the old names keep working).
  [Details](docs/xolotl/CHANGELOG.md#deepseek-v41-loads-in-44-s-and-xolotl_-names-for-every-switch-7-nov)
- **6 Nov - +8 % decode on every RTX 40 card**: QFN 45 GB 139 -> 150 tok/s on the 4090, same output. A tuned set of
  bit-exact speed switches for sm_89, on by default (`STRATA_SM89_PERF=0` turns it off).
  [Details](docs/xolotl/CHANGELOG.md#rtx-40-sm_89-speed-switches-on-by-default-6-nov)
- **6 Nov - DeepSeek-V4.1: +11 % long-prompt prefill, +2.7 % decode**, same output bit for bit. The next prompt
  chunk's Engram reads start a chunk early, and the shared expert runs while the routed experts load. Both on by
  default, ideas from Basalt. [Details](docs/xolotl/CHANGELOG.md#deepseek-v41-engram-read-ahead-and-the-late-shared-expert-6-nov)
- **6 Nov - +13-17 % prefill on IQ1_M layers** (QFN 45 GB: up to 3,890 tok/s at 16K on the 4090), and closer to
  the Q8_0 parent than before. On by default (`STRATA_MMQ_IQ1M=0` turns it off), port of Basalt.
  [Details](docs/xolotl/CHANGELOG.md#iq1_m-prompts-on-mmq-6-nov)
- **4 Nov - faster short requests with the disk prefix cache** (#115): 9.0 s -> 0.32 s for a short request after a
  47K-token prompt, opt-in. Also the MiniMax replay fix (#116), the Basalt ports and RTX 5090 `--kv-q8` numbers (#114).
  [Details](docs/xolotl/CHANGELOG.md#prefix-cache-latency-minimax-replay-and-the-basalt-ports-4-nov)
- **3 Nov - LFM3 as the ears and voice of QFN** on one 4090 (`scripts/xol-qfn-voice.sh`).
  [Details](docs/xolotl/CHANGELOG.md#lfm3-as-the-ears-and-voice-of-a-bigger-model-3-nov)
- **2 Nov - LFM3-Omni-2B, natively**: text, pictures, audio and video in, text and speech out, ~300 tok/s.
  [docs/LFM3.md](docs/LFM3.md)
- **31 Oct - [Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)**:
  Swift 1.5, uncensored, 199 tok/s decode and 3,204 tok/s prefill at 16K on the 4090, Gauntlet v2 6.68.
  [Details](docs/xolotl/CHANGELOG.md#qwen38-flash-next-xq-45-gb-31-oct)
- **30 Oct - Qwen 4, day-one prep**: what's known and the plan. [docs/QWEN4.md](docs/QWEN4.md)
- **25 Oct - [DeepSeek-V4.1 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)**, the best
  DeepSeek build: 86.9% top-1 agreement with the full model, 162/164 HumanEval. Run it with `--rank v4/rank-1017-v4.bin`.
  [Details](docs/xolotl/CHANGELOG.md#deepseek-v41-v4-25-oct)
- **October - community work**:
  - [grammar and a disk prefix cache](docs/xolotl/CHANGELOG.md#grammar-a-disk-prefix-cache-and-serving-fixes-99-105-113) (#99, #105-#113)
  - [serving fixes](docs/xolotl/CHANGELOG.md#more-community-fixes-96-98-100-104) (#96-#104)
  - [nine PRs](docs/xolotl/CHANGELOG.md#nine-community-pull-requests-86-95) (#86-#95)
  - [ISTA-DASLab](docs/xolotl/CHANGELOG.md#ista-daslabs-ggufs-work-too) and [Unsloth](docs/xolotl/CHANGELOG.md#upstream-llamacpp-and-unsloth-deepseek-v41-ggufs-82-84-85) GGUFs
  - [ZFS ARC](docs/xolotl/CHANGELOG.md#openzfs-arc-counted-as-available-ram-83) (#83)
  - [deep sleep](docs/xolotl/CHANGELOG.md#deep-sleep-for-minimax-m31-and-deepseek-v41-75) (#75)
  - [Qwen3.6 serving](docs/xolotl/CHANGELOG.md#qwen36-serving-76-79-80) (#76-#80)
  - [GLM long-conversation fix](docs/xolotl/CHANGELOG.md#glm-53-flash-long-conversations-no-longer-crash-46) (#46)
  - [batch slots, the conversation pool, multi-GPU and more](docs/xolotl/CHANGELOG.md#earlier-in-october)

## Fast mode (DeepSeek-V4.1)

DeepSeek-V4.1's decode speed is set by how many expert bytes cross the RAM bus and the NVMe drive per token. Fast mode
keeps a 2-bit (Q2_K) copy of the two biggest expert formats (the MXFP4 and Q3_K experts, about half of those bytes)
in a sidecar file and reads that instead. The model file is not changed, and turning it off is unsetting one variable.

**On real agent work it decodes at 41.1 tok/s against 32.6 in normal mode (+26 %), and solves the same tasks.** Both
numbers are decode during the Gauntlet's coding tasks below (speculative drafts land more often on code); on a fixed
8K-prompt probe it is 33.0 vs 26.1 tok/s.

| RTX 4090, V4, 8K prompt | decode | short34 top-1 | short34 NLL |
|---|---|---|---|
| normal | 26.1 tok/s | 87.01 % | 0.512 |
| **fast** | **33.0 tok/s (+27 %)** | **86.16 %** | **0.540** |

<sub>Decode here = the 8K-prompt probe; short34 = agreement with the full model's next token over our 34-text set.</sub>

```bash
build-xolotl/ds41_generate -m V41-XQ-1017-v4.gguf --build-shadow /nvme/V4.fast.x41s   # once, ~30 min CPU, ~40 GB
XOLOTL_SHADOW=$(cat /nvme/V4.fast.x41s.list)  <your usual serve command>
```

Agentic coding, the [Gauntlet](tools/gauntlet/README.md) v3-small (6 hard tasks graded by hidden tests, 10 turns,
thinking on, served settings, 2 runs each):

| | solved | score | decode during the tasks |
|---|---|---|---|
| normal | 8 / 12 | 4.99 | 32.6 tok/s |
| **fast** | **8 / 12** | **4.99** | **41.1 tok/s (+26 %)** |

The same tasks solved and missed in both modes (each missed 04 once; 03 stays half-done in every run).

It is lossy, so it is opt-in: on the Gauntlet it cost nothing measurable, but short34 shows it moves a few tokens in a
hundred, so keep the normal mode where every token counts. Everything else
(switches, the importance matrix, what was tried): [docs/XOLOTL_FAST.md](docs/XOLOTL_FAST.md). The idea is
compress-then-correct from [PROMETHEVS](https://github.com/Magnvs7/Promethevs) (Ryan Flavell).

## What one PC can do

**One RTX 4090 (24 GB) · Ryzen 7 5700X · 110 GB DDR4 · one NVMe drive.** Every number is measured on that machine,
through the real server, while the model works through the [Gauntlet](tools/gauntlet/README.md) v3-small: 6 hard
agentic coding tasks graded by hidden tests, 10 turns each, thinking on (32K-token budget), each model as it is
served, 2 runs per model. Decode is the generation speed during those tasks.

| Model | Size | Solved | Score (of 6) | Decode | Prefill | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)** (XQ) | 552B MoE | 8/12 | 4.99 | **32.6 tok/s** | **289-409 tok/s** | 512K |
| the same, [fast mode](#fast-mode-deepseek-v41) | 552B MoE | 8/12 | 4.99 | **41.1 tok/s** | | 512K |
| **GLM-5.3-Flash** | 320B MoE | (running 8 Nov) | | | **290-375 tok/s** | 512K |
| **[Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)** | 125B MoE | 7/12 | 4.49 | **157 tok/s** | **3,800-4,070 tok/s** | 200K |
| **Qwen3.8-27B** (dense) | 27B | (running 8 Nov) | | | **1,240-1,390 tok/s** | 200K |

<sub>Score = the sum of each task's progress (1 per solved task, partial credit from the hidden tests otherwise),
averaged over the 2 runs. Prefill = a fresh 6K-17K-token prompt through the server (`tools/throughput.py`;
Qwen3.8-Flash-Next at 16K since the 6 Nov speed-ups). DeepSeek-V4.1's Gauntlet runs have its dense half at Q6_K, as
it is served; its prefill column and the engine's Q8_0 reference numbers: [docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md).
The older table (21 HumanEval problems, 10 tool calls, 3 agentic loops) is there too: every model scores 19-21/21,
10/10, 3/3 on it, so it no longer tells them apart. LFM3-Omni-2B (a 2B voice model, not a coding agent):
[docs/LFM3.md](docs/LFM3.md).</sub>

**Speed-ups** (same 4090 box; before/after on the same prompt and settings; full log in
[docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md#speed-log)):

| Date | Change | Model | Prefill before | Prefill after | Decode |
| --- | --- | --- | ---: | ---: | --- |
| 7 Nov | [Fast mode](#fast-mode-deepseek-v41): 2-bit (Q2_K) copy of the biggest experts (**opt-in, lossy**: short34 top-1 87.0 -> 86.2 %) | [DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored), agent tasks / 8K prompt | | | **32.6 -> 41.1 tok/s on agent tasks (+26 %)**; 26.1 -> 33.0 on the 8K probe |
| 6 Nov | RTX 40 (sm_89) speed switches (default on) | [Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored), 16K prompt | 4,054 tok/s | 4,069 tok/s (same) | **139 -> 150 tok/s (+8 %)** |
| 6 Nov | Engram read-ahead + late shared expert (default on) | [DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored), 26.6K prompt / 8K decode | 456 tok/s | **506 tok/s (+11 %)** | **25.7 -> 26.5 tok/s (+2.7 %)** |
| 6 Nov | IQ1_M experts on MMQ for prompts (default on) | [Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored), 16K prompt | 3,239-3,430 tok/s | **3,796-3,890 tok/s (+13-17 %)** | within noise |

**Smaller cards (12-16 GB).** The engine fits the work to the card, and the experts that don't fit live in RAM and on
the SSD, so system RAM matters as much as VRAM. Community runs:

| Card | RAM | Model | Prefill | Decode | Run |
| --- | --- | --- | ---: | ---: | --- |
| RTX 3060 12 GB | 64 GB | Qwen3.8-Flash-Next Q2_0 | 882 tok/s (4K), 981 (32K), 934 (128K) | 39-45 tok/s | [2026-10-09](bench/results/2026-10-09-community-rtx-3060-0141/README.md) |
| RTX 4070 12 GB | 32 GB | Qwen3.8-Flash-Next IQ2_XS | | 48 tok/s | [2026-10-08](bench/results/2026-10-08-community-rtx4070-iq2-xs/README.md) (Hcl192088) |
| RTX 3060 12 GB | 46 GB | DeepSeek-V4.1-Flash | 149 tok/s (8K) | 8-10 tok/s | [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) |
| RTX 5070 12 GB | 64 GB | DeepSeek-V4.1-Flash | 121-231 tok/s (4-8K) | 4.7 tok/s | [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) |

More cards, AMD and Intel included: [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md).

**What that means in practice:**

- **A 552B frontier model as a local coding agent.** Point Claude Code, Codex, opencode or Open WebUI at it - tool
  calls, streaming, prefix reuse across agent turns, a thinking budget.
- **No server farm.** The weights (152 GiB for DeepSeek-V4.1) live across VRAM, pinned RAM and NVMe; the engine
  carries up the experts each token needs.
- **Fast where it counts.** The Qwen models run at 100-200 tok/s; an agent's small follow-up turn on DeepSeek-V4.1
  costs ~8 s on a 6K-16K context, and the engine learns the conversation's experts between turns. When speed matters
  more than the last bit of quality, DeepSeek-V4.1's [fast mode](#fast-mode-deepseek-v41) decodes at 41 tok/s on agent tasks
  (32.6 normally).

## Quick start

Linux, an NVIDIA GPU with 24 GB, CUDA 12.x/13.x, and RAM for the arena (90 GB was used above).

```bash
git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
bash tools/xolotl/fetch-llama.sh   # llama.cpp at the pinned commit + tools/xolotl/ggml/xolotl-ggml.patch -> third_party/llama.cpp
cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_DSV41_FQ_OP=ON \
  -DSTRATA_DSV41_HC_MIX=ON
ninja -C build-xolotl      # strata, strata-dense, glm_generate, ds41_generate, ds4_generate
```

Get a model:

- **DeepSeek-V4.1-Flash:** the **[1017 builds](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)**
  of the October checkpoint (V4 recommended, with `v4/rank-1017-v4.bin`), or the earlier
  [GSQ build](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored) (152 GiB, mixed ternary /
  2-bit / 3-bit). Both cards have the download and Engram steps. Optional for V4: the
  [fast-mode](#fast-mode-deepseek-v41) sidecar (~40 GB), built once with `ds41_generate --build-shadow`.
- **Qwen3.8-Flash-Next:** **[XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)**
  (the two `.gguf` shards plus `pack-qfn45/`).

Then serve it (edit the model paths in the DeepSeek and GLM configs first):

```bash
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150           # DeepSeek-V4.1-Flash
XOLOTL_SHADOW=$(cat /nvme/V4.fast.x41s.list) python3 tools/v41/serve/serve_ds41.py ...                                # the same in fast mode (V4)
python3 tools/glm/serve/serve_glm.py  --engine strata --config tools/glm/serve/strata-glm-maya-abl-512k.json --port 8144  # GLM-5.3-Flash
./scripts/xol-qfn.sh --serve-only --model <shard 1 .gguf> --pack <pack-qfn45>                                       # Qwen3.8-Flash-Next (port 8099)
python3 -m serve.server --engine strata --config <your-27b.json> --port 8090                                          # Qwen3.8-27B (exe: build-xolotl/strata-dense)
```

Then point any OpenAI- or Anthropic-compatible client at `http://127.0.0.1:<port>/v1`.

**Will it run on my machine?** NVIDIA RTX 20 or newer, the more RAM the better, a fast NVMe on ext4 or NTFS (not
exFAT): the details for DeepSeek-V4.1 are in [docs/WILL_IT_RUN.md](docs/WILL_IT_RUN.md).

## How it works, in one paragraph

A mixture-of-experts model only uses a few experts per token. Xolotl keeps every expert somewhere - the hottest in
VRAM, tens of GiB in a pinned RAM arena the GPU reads by DMA, the rest on NVMe read at the drive's ceiling - and
moves the ones each token's router asks for, a layer ahead. Each model adds its own tricks: DeepSeek-V4.1 gets a
calibrated mixed-precision quantisation, speculative decoding with exact sampling, a bit-exact 4-bit KV cache for 512K
context, an arena that adapts to the conversation, an opt-in fast mode that reads a 2-bit copy of its biggest experts,
and Xolotl Direct, which lets the NVMe drive write experts straight into VRAM; GLM-5.3-Flash gets its abliteration LoRA on every path and soft
pruning; the Qwen models get MTP speculative decoding. **[Read the full write-up](docs/xolotl/HOW_IT_WORKS.md).**

## Read more

| | |
| --- | --- |
| [How Xolotl works](docs/xolotl/HOW_IT_WORKS.md) | the three memory tiers, per-model engines, the server, the agent-loop work |
| [Benchmarks](docs/xolotl/BENCHMARKS.md) | every measured number, exactness checks, how to measure your own PC |
| [Changelog](docs/xolotl/CHANGELOG.md) | everything that landed, newest first, with the PR numbers and who did it |
| [DeepSeek-V4.1-Flash engine](docs/DEEPSEEK_V41.md) | GSQ quantisation, DSpark, the fp4 KV cache, Engram |
| [DeepSeek-V4.1 fast mode](docs/XOLOTL_FAST.md) | the 2-bit expert sidecar: build, serve, switches, what it costs and what we tried |
| [DeepSeek-V4.1 on Windows](docs/WINDOWS.md) | native MSVC + CUDA build, configs, RTX 50 notes |
| [GLM-5.3-Flash engine](docs/glm) | KDA + MLA attention, the LoRA on routed experts, soft pruning |
| [LFM3-Omni engine](docs/LFM3.md) | pictures, audio, video and speech on a 2B model, park/unpark beside a big one |
| [Qwen3.8 / upstream Strata](README.strata.md) | the Strata engine this grew from |
| [Brand kit](docs/brand/XOLOTL_BRAND.md) | logo, banner, palette |

## Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

Built on **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp),
with the Qwen3.8-27B dense port by [perronemirko](https://github.com/perronemirko/Strata). Model weights by DeepSeek,
Zhipu AI, Qwen and Liquid AI, with quants and fine-tunes by peasantsmith, ukisai, d0xin and gcsa-abliterix. Thanks to
every contributor: NektoDron, Xiaomi2008, AlexeyMatskevich, paul123637-lab, kelheor, exye137, penink-WH, cpacsm75,
jesdga95 (Basalt) and Ryan Flavell (PROMETHEVS). Who did what: **[CREDITS.md](CREDITS.md)**.

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe. Research code from one developer, measured on one PC; reports from other hardware welcome. MIT.

## Support

If Xolotl's been useful to you and you want to help support what I do, you can shout me a coffee on
[Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)
