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

## What one PC can do

**One RTX 4090 (24 GB) · Ryzen 7 5700X · 90 GB DDR4 · one NVMe drive.** Every number is measured on that machine,
through the real server, while it solved real tasks (21 HumanEval problems, 10 tool-call tasks, 3 agentic bug-fix
loops), greedy, thinking on with a 32K-token thinking budget.

| Model | Size | Prefill | Decode | HumanEval | Tool calls | Agentic | Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **[DeepSeek-V4.1-Flash](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)** | 552B MoE | **180-320 tok/s** | **26.8 tok/s** | 21/21 | 10/10 | 3/3 | 512K |
| **GLM-5.3-Flash** | 320B MoE | **290-375 tok/s** | **25.3 tok/s** | 19/21 | 10/10 | 3/3 | 512K |
| **Qwen3.8-Flash-Next** | 125B MoE | **3,000-3,430 tok/s** | **127.8 tok/s** | 21/21 | 10/10 | 3/3 | 200K |
| **Qwen3.8-27B** (dense) | 27B | **1,240-1,390 tok/s** | **98.5 tok/s** | 21/21 | 10/10 | 3/3 | 200K |

<sub>Decode = generation speed during those tasks. Prefill = a fresh 6K-17K-token prompt through the server
(`tools/throughput.py`; Qwen3.8-Flash-Next: 3,430 / 3,005 tok/s at 6.5K / 16.7K, Qwen3.8-27B: 1,385 / 1,244 tok/s,
DeepSeek-V4.1: 178 / 268 tok/s at 6.1K / 16.5K - its prompt batch is sized from free VRAM, ~6-9K tokens per sweep
on a 24 GB card with the speculative-decode draft loaded).
DeepSeek-V4.1 with the speculative-decode draft off the card during long prompts (the default since 2026-10-17):
187 / 317 tok/s at 6.4K / 16.7K. GLM-5.3-Flash: soft prune + `--route-bias 0.25`
(`tools/glm/serve/strata-glm-maya-abl-512k-bias.json`), prefill 301 / 374 tok/s at 6.1K / 15.6K and 368-376 on fresh
30K / 44K prompts. How close the DeepSeek quant gets to DeepSeek's own API on the same tasks:
[docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md#deepseek-v41-quant-vs-the-deepseek-api).</sub>

**What that means in practice:**

- **A 552B frontier model as a local coding agent.** Point Claude Code, Codex, opencode or Open WebUI at it - tool
  calls, streaming, prefix reuse across agent turns, a thinking budget.
- **No server farm.** The weights (152 GiB for DeepSeek-V4.1) live across VRAM, pinned RAM and NVMe; the engine
  carries up the experts each token needs.
- **Fast where it counts.** The Qwen models run at 100-130 tok/s; an agent's small follow-up turn on DeepSeek-V4.1
  costs ~8 s on a 6K-16K context, and the engine learns the conversation's experts between turns.

## Recently added

- **Batch slots** for the DeepSeek-V4.1 engine (#22, NektoDron): `"parallel": N` in a ds41 config runs several
  requests in one pass over the model, one token each, on the server's existing slot scheduler. How it works, what a
  slot costs and the limits: [docs/BATCHING.md](docs/BATCHING.md).
- **Conversation pool** (#19, #20, NektoDron): every conversation is kept at its turn boundary in pinned host RAM
  (`--conversation-cache-mib`, default 8192) and a request resumes from the longest matching prefix, pictures
  included. Agents taking turns no longer re-read their prompts (measured on a 5090: three agents at 30-60K tokens,
  47.5 -> 3.96 s a turn).
- **Long prompts** past 512K tokens no longer run out of memory or crash on the pass graph (#21, NektoDron): 768K and
  1M prompts complete.
- **Exact resumes** (#18, NektoDron): a partial unpark resumes only inside the window ring, and the Engram history is
  written back on every resume (bit-identical to a fresh prefill). The **image encoder runs on the GPU** with fast
  SDPA (#17, NektoDron).
- **Multi-GPU layer split** (AtticusG3 / buster, Niko1221, bsvinay): `--layer-split` across cards with a PCIe hand-off,
  an optional head device for the output head + DSpark, a VRAM expert cache per GPU and sm_70 (V100) support:
  [docs/MULTI_GPU.md](docs/MULTI_GPU.md), [BUSTER.md](BUSTER.md). Under a split the engine serves one request at a
  time with its own split-aware conversation cache; batch slots and the pool are single-card.
- **Remote expert card** (`--remote-device`, masahiro hibi): a second GPU holds extra experts for the first, with
  `--route-bias-remote`.
- Serve: STOP cancels a long prefill (clawdia-saka), `--kv-q8` / `--kv-type` for the compressed KV + indexer keys,
  a park window floor for agent side calls, cached DSpark verify-width graphs, forced tool calls in V4.1's DSML shape,
  and the expert-cache fixes from clawdia-saka.
- **DeepSeek-V4.1-Flash-1017 builds** of the October checkpoint:
  [IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored)
  (V2 142 GiB; V3 152 GiB, recalibrated on a code + agentic heavy mix). **V4** (same size as V3, with our XQ
  refinement on every expert) is quantising now. Early per-layer error is 29-48% below V3's; release expected around 26-27 Oct.
- **Native Windows** for the DeepSeek-V4.1 engine (MSVC + CUDA, RTX 50 notes from #14): [docs/WINDOWS.md](docs/WINDOWS.md).
- **Vision** input for DeepSeek-V4.1 (#8), per-request routing options (#11), a 16-row tier call (#9) and a uniform
  VRAM expert cache under the auto slot budget (#15).
- Engine: a VRAM breakdown in the decode trace (dense weights, DSpark draft, free memory) and per-GPU one-time setup
  in the dense kernels.

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

Get a model: DeepSeek-V4.1-Flash is **[IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)**
(152 GiB, our GSQ mixed ternary / 2-bit / 3-bit quant - the card has the download and Engram steps), or the newer
**[1017 builds](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored)** of the October checkpoint. Then serve it
(edit the model paths in its config first):

```bash
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150           # DeepSeek-V4.1-Flash
python3 tools/glm/serve/serve_glm.py  --engine strata --config tools/glm/serve/strata-glm-maya-abl-512k.json --port 8144  # GLM-5.3-Flash
python3 -m serve.server --engine strata --config <your-qwen-flash.json> --port 8130                                    # Qwen3.8-Flash-Next
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
| [DeepSeek-V4.1-Flash engine](docs/DEEPSEEK_V41.md) | GSQ quantisation, DSpark, the fp4 KV cache, Engram |
| [DeepSeek-V4.1 on Windows](docs/WINDOWS.md) | native MSVC + CUDA build, configs, RTX 50 notes |
| [GLM-5.3-Flash engine](docs/glm) | KDA + MLA attention, the LoRA on routed experts, soft pruning |
| [Qwen3.8 / upstream Strata](README.strata.md) | the Strata engine this grew from |
| [Brand kit](docs/brand/XOLOTL_BRAND.md) | logo, banner, palette |

## Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

Built on **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 (its engine, server and kernels, merged at
0.1.44.1) and [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp); the Qwen3.8-27B dense port is
[perronemirko](https://github.com/perronemirko/Strata)'s. The DeepSeek-V4.1 and GLM-5.3-Flash engines, the GSQ quant
and the unification are Indras Mirror's. Model weights: DeepSeek, Zhipu AI, Qwen, with quants and fine-tunes by
peasantsmith, ukisai, d0xin and gcsa-abliterix.

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe. Research code from one developer, measured on one PC; reports from other hardware welcome. MIT.

## Support

If Xolotl's been useful to you and you want to help support what I do, you can shout me a coffee on
[Ko-fi](https://ko-fi.com/indrasmirror). Every bit helps keep this going :)
