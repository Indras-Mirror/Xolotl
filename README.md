<p align="center"><img src="docs/brand/assets/readme-header-1200.png" alt="Xolotl - frontier MoE models on one gaming PC" width="100%"></p>

<h1 align="center">Xolotl</h1>

<p align="center"><b>DeepSeek-V4.1-Flash - all 552 billion parameters - on one gaming PC, at 33 tok/s on real agent tasks (41 in fast mode).</b><br>
Plus GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B and LFM3-Omni: one engine, one OpenAI / Anthropic-compatible server.<br>
Built on <a href="https://github.com/Niko1221/Strata">Strata</a> and heavily extended · Linux · NVIDIA · MIT</p>

> **A Strata fork.** Xolotl keeps [Strata](https://github.com/Niko1221/Strata)'s engine, server and kernels at its
> core (merged at 0.1.44.1) and adds ~71,000 lines on top: the DeepSeek-V4.1 and V4-Flash engines, the GLM-5.3-Flash
> engine and the Qwen3.8-27B dense port, each with its own quantisation and expert-tiering work.

## Latest

- **8 Nov** - [Gauntlet v3](#what-one-pc-can-do) is the main table: every model on the same 6 hard agentic tasks.
- **7 Nov** - [DeepSeek-V4.1 fast mode](#fast-mode-deepseek-v41) (opt-in): 41 tok/s on real agent tasks against 32.6,
  the same tasks solved.
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

| DeepSeek-V4.1 V4, RTX 4090 |    Decode, agent tasks |      Decode, 8K prompt | Gauntlet v3-small | Next-token agreement |
| -------------------------- | ---------------------: | ---------------------: | :---------------: | :------------------: |
| normal                     |             32.6 tok/s |             26.1 tok/s |       8 / 12      |        87.0 %        |
| **fast mode**              | **41.1 tok/s (+26 %)** | **33.0 tok/s (+27 %)** |     **8 / 12**    |      **86.2 %**      |

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
served, 2 runs per model. Decode is the generation speed during those tasks.

| Model                                                                                                             | Size      |  Solved | Score |     Decode |           Prefill | Context | Full-precision API, same tasks |
| ----------------------------------------------------------------------------------------------------------------- | --------- | :-----: | :---: | ---------: | ----------------: | :-----: | :----------------------------: |
| **[DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)**     | 552B MoE  |  8 / 12 |  4.99 | 32.6 tok/s |     289-409 tok/s |   512K  |            running             |
| **DeepSeek-V4.1 V4, [fast mode](#fast-mode-deepseek-v41)**                                                        | 552B MoE  |  8 / 12 |  4.99 | 41.1 tok/s |         measuring |   512K  |               -                |
| **GLM-5.3-Flash**                                                                                                 | 320B MoE  | running |   -   |          - |     290-375 tok/s |   512K  |            running             |
| **[Qwen3.8-Flash-Next XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)** | 125B MoE  |  7 / 12 |  4.49 |  157 tok/s | 3,800-4,070 tok/s |   200K  |            running             |
| **Qwen3.8-27B**                                                                                                   | 27B dense | running |   -   |          - | 1,240-1,390 tok/s |   200K  |            running             |

<sub>Score (of 6) = each task's progress summed (1 per solve, partial credit from the hidden tests), averaged over the 2 runs.
Prefill = a fresh 6K-17K-token prompt (`tools/throughput.py`). DeepSeek-V4.1 runs here with its dense half at Q6_K, as
it is served. More numbers, the Q8_0 references, the speed log and the older HumanEval table (every model 19-21/21
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
| [Qwen3.8 / upstream Strata](README.strata.md)      | the Strata engine this grew from                                                  |
| [Brand kit](docs/brand/XOLOTL_BRAND.md)            | logo, banner, palette                                                             |

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
