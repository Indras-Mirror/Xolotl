<p align="center"><img src="docs/brand/assets/readme-header-1200.png" alt="Xolotl - frontier MoE models on one gaming PC" width="100%"></p>

<h1 align="center">Xolotl</h1>

<p align="center"><b>DeepSeek-V4.1-Flash - all 552 billion parameters - on one gaming PC, at 27 tok/s.</b><br>
Plus GLM-5.3-Flash, Qwen3.8-Flash-Next and Qwen3.8-27B: one engine, one OpenAI / Anthropic-compatible server.<br>
Built on <a href="https://github.com/Niko1221/Strata">Strata</a> and heavily extended · Linux · NVIDIA · MIT</p>

> **A Strata fork.** Xolotl keeps [Strata](https://github.com/Niko1221/Strata)'s engine, server and kernels at its
> core (merged at 0.1.44.1) and adds ~71,000 lines on top: the DeepSeek-V4.1 and V4-Flash engines, the GLM-5.3-Flash
> engine and the Qwen3.8-27B dense port, each with its own quantisation and expert-tiering work.
>
> **Starred this as Strata-DSV4.1?** You are in the right place. Strata-DSV4.1 grew into **Xolotl**: the same
> DeepSeek-V4.1 engine (faster now), plus GLM-5.3-Flash and the Qwen3.8 models, in one tree. Old links and clones
> redirect here; nothing DeepSeek-V4.1 was removed.

## Latest

Full notes for every item: [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md).

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

## What one PC can do

**One RTX 4090 (24 GB) · Ryzen 7 5700X · 90 GB DDR4 · one NVMe drive** (110 GB RAM since 1 Nov; the numbers below
are from the 90 GB runs unless marked). Every number is measured on that machine,
through the real server, while it solved real tasks (21 HumanEval problems, 10 tool-call tasks, 3 agentic bug-fix
loops), greedy, thinking on with a 32K-token thinking budget.

| Model | Size | Prefill | Decode | HumanEval | Tool calls | Agentic | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **[DeepSeek-V4.1-Flash](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)** | 552B MoE | **180-320 tok/s** | **26.8 tok/s** | 21/21 | 10/10 | 3/3 | 512K |
| **[DeepSeek-V4.1-Flash-1017 V4](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)** (XQ) | 552B MoE | **289-409 tok/s** | **25.7 tok/s** | 21/21 | 10/10 | 3/3 | 512K |
| **GLM-5.3-Flash** | 320B MoE | **290-375 tok/s** | **25.3 tok/s** | 19/21 | 10/10 | 3/3 | 512K |
| **Qwen3.8-Flash-Next** | 125B MoE | **3,000-3,430 tok/s** | **127.8 tok/s** | 21/21 | 10/10 | 3/3 | 200K |
| **Qwen3.8-27B** (dense) | 27B | **1,240-1,390 tok/s** | **98.5 tok/s** | 21/21 | 10/10 | 3/3 | 200K |
| **[LFM3-Omni-2B](https://huggingface.co/IndrasMirror/LFM3-Omni-2B-XQ-Uncensored)** (XQ-6.0, omni) | 2B | **26,000-32,800 tok/s** | **~300 tok/s** | 13/21 | 9/10 | 0/3 | 32K |

<sub>Decode = generation speed during those tasks. Prefill = a fresh 6K-17K-token prompt through the server
(`tools/throughput.py`; Qwen3.8-Flash-Next: 3,430 / 3,005 tok/s at 6.5K / 16.7K, Qwen3.8-27B: 1,385 / 1,244 tok/s,
DeepSeek-V4.1: 178 / 268 tok/s at 6.1K / 16.5K - its prompt batch is sized from free VRAM, ~6-9K tokens per sweep
on a 24 GB card with the speculative-decode draft loaded).
DeepSeek-V4.1 with the speculative-decode draft off the card during long prompts (the default since 2026-10-17):
187 / 317 tok/s at 6.4K / 16.7K. GLM-5.3-Flash: soft prune + `--route-bias 0.25`
(`tools/glm/serve/strata-glm-maya-abl-512k-bias.json`), prefill 301 / 374 tok/s at 6.1K / 15.6K and 368-376 on fresh
30K / 44K prompts. DeepSeek-V4.1-Flash-1017 V4 (25 Oct): dense half Q8_0, `rank-1017-v4.bin` + `--tier-batch-pcie 0.25`,
prefill 289-297 / 395-409 tok/s at 6.1K / 16.5K; full HumanEval (164) 162/164. How close the DeepSeek quant gets to DeepSeek's own API on the same tasks:
[docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md#deepseek-v41-quant-vs-the-deepseek-api).
LFM3-Omni-2B (2 Nov, `xolotl-lfm3`, alone on the card): a non-reasoning model, so thinking off; prefill 32,796 / 26,049
tok/s at 5.3K / 14K, decode 292-310 tok/s; it also hears, sees and speaks (speech out at 12x real time) - see
[docs/LFM3.md](docs/LFM3.md). Its agentic 0/3 is the 2B model calling a tool the agent does not have.</sub>

**Speed-ups since those runs** (same 4090 box; before/after on the same prompt and settings; full log in
[docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md#speed-log)):

| Date | Change | Model | Prefill before | Prefill after | Decode |
| --- | --- | --- | ---: | ---: | --- |
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
  costs ~8 s on a 6K-16K context, and the engine learns the conversation's experts between turns.

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
  2-bit / 3-bit). Both cards have the download and Engram steps.
- **Qwen3.8-Flash-Next:** **[XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)**
  (the two `.gguf` shards plus `pack-qfn45/`).

Then serve it (edit the model paths in the DeepSeek and GLM configs first):

```bash
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150           # DeepSeek-V4.1-Flash
python3 tools/glm/serve/serve_glm.py  --engine strata --config tools/glm/serve/strata-glm-maya-abl-512k.json --port 8144  # GLM-5.3-Flash
./scripts/xol-qfn.sh --serve-only --model <shard 1 .gguf> --pack <pack-qfn45>                                       # Qwen3.8-Flash-Next (port 8099)
python3 -m serve.server --engine strata --config <your-27b.json> --port 8090                                          # Qwen3.8-27B (exe: build-xolotl/strata-dense)
```

Then point any OpenAI- or Anthropic-compatible client at `http://127.0.0.1:<port>/v1`.

### Will it run on my machine? (DeepSeek-V4.1)

What I've measured is one setup: **RTX 4090 (24 GB), 90 GB DDR4, Linux, NVMe**. Beyond that:

- **OS:** Linux, or Windows 11 natively (MSVC + CUDA; build and serve steps in [docs/WINDOWS.md](docs/WINDOWS.md)).
  Everything measured here is on Linux.
- **GPU:** NVIDIA, compute capability 7.5 or newer (RTX 20/30/40/50). Build with your card's architecture:
  `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 30, `89` for RTX 40, `120` for RTX 50 (use CUDA 13.x for RTX 50; 12.8 builds
  had prompt-path crashes). **Tesla V100 (sm_70):** experimental DS41-on-Buster path only — see [BUSTER.md](BUSTER.md)
  (`STRATA_EXPERIMENTAL_SM60=ON`, CUDA 12.x, no chunk-MMQ on Volta).
- **VRAM:** 24 GB is what I run. 16 GB should load but is untested, and prompts get processed in smaller batches, so
  prefill will be slower. Below 16 GB won't work.
- **RAM:** 90 GB here, with a 64 GB expert cache in RAM. Less RAM means a smaller cache, so more experts come off the
  NVMe each token and decode is slower. More RAM helps: around 256 GB holds every expert.
- **Disks:** about 152 GiB for the quant plus ~200 GB for the Engram tables. Both want a fast NVMe on ext4 or NTFS.
  **Not exFAT** (Linux's exFAT driver serialises parallel reads; expert reads were 5x slower here) and not a hard
  drive (the Engram lookups are ~48 small random reads per token).

## How it works, in one paragraph

A mixture-of-experts model only uses a few experts per token. Xolotl keeps every expert somewhere - the hottest in
VRAM, tens of GiB in a pinned RAM arena the GPU reads by DMA, the rest on NVMe read at the drive's ceiling - and
moves the ones each token's router asks for, a layer ahead. Each model adds its own tricks: DeepSeek-V4.1 gets a
calibrated mixed-precision quantisation, speculative decoding with exact sampling, a bit-exact 4-bit KV cache for 512K
context and an arena that adapts to the conversation; GLM-5.3-Flash gets its abliteration LoRA on every path and soft
pruning; the Qwen models get MTP speculative decoding. **[Read the full write-up](docs/xolotl/HOW_IT_WORKS.md).**

## Read more

| | |
| --- | --- |
| [How Xolotl works](docs/xolotl/HOW_IT_WORKS.md) | the three memory tiers, per-model engines, the server, the agent-loop work |
| [Benchmarks](docs/xolotl/BENCHMARKS.md) | every measured number, exactness checks, how to measure your own PC |
| [Changelog](docs/xolotl/CHANGELOG.md) | everything that landed, newest first, with the PR numbers and who did it |
| [DeepSeek-V4.1-Flash engine](docs/DEEPSEEK_V41.md) | GSQ quantisation, DSpark, the fp4 KV cache, Engram |
| [DeepSeek-V4.1 on Windows](docs/WINDOWS.md) | native MSVC + CUDA build, configs, RTX 50 notes |
| [GLM-5.3-Flash engine](docs/glm) | KDA + MLA attention, the LoRA on routed experts, soft pruning |
| [LFM3-Omni engine](docs/LFM3.md) | pictures, audio, video and speech on a 2B model, park/unpark beside a big one |
| [Qwen3.8 / upstream Strata](README.strata.md) | the Strata engine this grew from |
| [Brand kit](docs/brand/XOLOTL_BRAND.md) | logo, banner, palette |

## Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

Built on **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 (its engine, server and kernels, merged at
0.1.44.1) and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp); the Qwen3.8-27B dense port is
[perronemirko](https://github.com/perronemirko/Strata)'s. The DeepSeek-V4.1 and GLM-5.3-Flash engines, the GSQ quant
and the unification are Indras Mirror's. Model weights: DeepSeek, Zhipu AI, Qwen, with quants and fine-tunes by
peasantsmith, ukisai, d0xin and gcsa-abliterix; LFM3-Omni by [Liquid AI](https://huggingface.co/LiquidAI), its engine
following llama.cpp's LFM3 and libmtmd work.

Contributors, with thanks: [NektoDron](https://github.com/NektoDron) (MMQ id slack, forced tool calls, the arena pipe
and indexer WMMA opt-ins, #48-#52; checked #55; the Q3_K + GPTQ Q2_0 lite calibration and allocation tooling, #58;
the ggml 6050b467 pool-hook link fix, #72; the control-queue wakeup fix, #79; Qwen3.6 vision, prefix cache and
batched MTP, #76; deep sleep, #75; resident prefix reuse, #80; complete Unsloth GGUF and DSpark support, #84; cancelled-request slot wakeups, #88; shared host checkpoint payloads, #89; learned suffix-edit checkpoints, #90; the bounded tokenizer span cache, #91; Qwen3.6 snapshot ownership, #92; the Metal fallback signature sync, #93; request-owned Flash batch cancellation, #95; MiniMax M3.1 serving, Qwen picture numbering and image minimum fixes, #96, #101, #102, #104; HIP shared int4 alignment, #105; completed-response and EOS-drain cancellation fixes, #107, #108; the persistent prefix cache, #99; native GBNF for Qwen Flash and DeepSeek V4.1, #112, #113; nonblocking prefix lookups and the minimum restore gain, #115; the MiniMax sampling replay keyword, #116),
[Xiaomi2008](https://github.com/Xiaomi2008) (IQ1_S routed experts for Unsloth's DeepSeek-V4.1 UD-IQ1_M, #86; ds41 fused router ops and batched state copies, #87; DSML tool-call fixes and the opt-in keep_cut_tool_calls, #106, #109, #110),
[AlexeyMatskevich](https://github.com/AlexeyMatskevich) (the upstream deepseek41 GGUF layout, #82; OpenZFS ARC
accounting, #83; IQ3_S expert downs for Unsloth's UD-Q2_K_XL, #85),
[paul123637-lab](https://github.com/paul123637-lab) (the DSpark seed fixes, #54 and #56; RTX 5080 and 5060 Ti helper
numbers), [kelheor](https://github.com/kelheor) (the vision `pread` fix, #51; the Windows ds41 build fix, #94), [exye137](https://github.com/exye137)
(the thread-local CUDA stream leak, #57), [penink-WH](https://github.com/penink-WH) (AC922 rig), [cpacsm75](https://github.com/cpacsm75) (192 GB
arena tuning on Windows, #77; the RTX 5090 arena and `--kv-q8` numbers at 256K/512K, #114) and
[jesdga95](https://github.com/jesdga95/basalt) (Basalt: the priced stream-all prefill chunk, quoted tool calls in string
values, union/nullable tool parameters, IQ code bytes by shift, IQ1_M on MMQ, and the ideas behind the
DeepSeek-V4.1 Engram read-ahead and late shared expert).

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe. Research code from one developer, measured on one PC; reports from other hardware welcome. MIT.

## Support

If Xolotl's been useful to you and you want to help support what I do, you can shout me a coffee on
[Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)
