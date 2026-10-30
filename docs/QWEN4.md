# Qwen 4: getting ready for day one

Qwen 4 isn't out yet (no weights, no model cards, no official config as of 30 Oct 2026). This page is what we know
from public code and how Xolotl plans to run it, so support can land as close to release day as possible. Everything
below the engine plan is unconfirmed until the real `config.json` and weights appear, and it will be corrected then.

## What's been seen in public code

| source | what it shows |
|---|---|
| Intel XPU kernels PR for vLLM (analytics-zoo `vllm-xpu-kernels` #22, 19-22 Oct; the repo has since gone private, a Wayback copy exists) | kernels for **Qwen4-27B dense** next to Qwen3.8-Flash-Next: "H5120 / dense HC", QSA with 24 Q / 4 KV heads. It also says the Qwen4 QSA weights weren't provided, so those kernels were tested on the config and the math only |
| NInfer design notes (`Neroued/ninfer`, `docs/designs/qwen4-27b-*.md`, 27 Oct) | a full Qwen4-27B config table derived from that PR (below) |
| Qwen Code (28 Oct) | every `qwen4*` model id gets a 1M-token context default (a placeholder until the real catalog) |
| model names in various PRs | qwen4-max, qwen4-plus, qwen4-coder-plus, qwen4-omni-120b-think, Qwen4.0-Max |

## Qwen4-27B (dense), as reported

| | |
|---|---|
| layers | 64, repeating GDN, GDN, GDN, QSA; plus one MTP layer |
| hidden / FFN | 5120 / 14336 SwiGLU, **dense (no MoE)** |
| hyper-connections | 4 residual streams (width 20480), low rank 640, two per layer plus a final mixer |
| GDN | 16 QK heads, 48 V heads, head dim 128, conv kernel 4 |
| QSA | 24 Q / 4 KV heads, head dim 256, sigmoid output gate; indexer 4 heads x 128, 4-token blocks, 2048-token budget |
| RoPE | theta 1e7, 64 rotary dims, interleaved MRoPE |
| vocab / context | 248320 / 262144 |
| PLE (n-gram table) | none |

It's the same family as Qwen3.8-Flash-Next (`qwen4exp`: hyper-connections, GDN, QSA), just dense and wider, without
the MoE and the PLE table.

## How Xolotl will run it

**Qwen 4 Flash / Next-style MoE.** If it keeps the `qwen4exp` architecture (which is what Qwen3.8-Flash-Next is, and
it's a fair bet), the `strata` engine runs it today: hyper-connections, GDN, QSA, the routed experts and PLE are all
there. Day one should mostly be checking the converter output and the metadata, then quants.

**Qwen4-27B dense** goes into `strata-dense` (the engine behind the Qwen3.8-27B dense port). What it needs:

1. **The hyper-connection residual** around every GDN / QSA / FFN block and the final mixer. The generic HC kernels
   (`gr_read` / `gr_write`) already take the geometry at runtime, so hidden 5120 / rank 640 works on them as is. The
   tuned fused HC kernel (`src/kernels/cuda/fused_gr.cu`) is compiled for 2560 / 320 only; templating it on the
   geometry is a speed-up that can follow.
2. **QSA attention** in place of full attention, reusing the `strata` QSA kernels (their shapes are runtime too; the
   27B changes only the KV head count, 2 to 4).
3. **MTP** with the HC stem, plus the loader / metadata keys once the real GGUF naming is known.
4. A test at the exact geometry on random weights before the real weights land, then the real thing.

Quants for both will go up on Hugging Face as soon as they pass our quality gates.

*Last updated 30 Oct 2026.*
