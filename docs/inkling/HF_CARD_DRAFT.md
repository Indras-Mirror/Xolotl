---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model:
- thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
tags:
- gguf
- moe
- inkling
- inkling-2-mini
- looped-transformer
- quantized
- mixed-precision
- uncensored
- abliterated
- xolotl
- consumer-gpu
language:
- en
pipeline_tag: text-generation
---

> **Draft, not released.** Sizes, speeds and quality numbers are still being measured. Every TBD below gets a
> real number before this card is published.

# Inkling-2-Mini · XQ, abliterated

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) with the refusal
direction edited out of the weights, then requantised with Indra's Mirror's mixed-precision **XQ** pipeline.
A 26B-parameter looped MoE with about 3B active parameters, 128 routed experts at top-8, and a 1M-token context.

Runs on [Xolotl](https://github.com/Indras-Mirror/Xolotl), an engine built for this class of model and machine,
on top of [Strata](https://github.com/Niko1221/Strata). The forward graph is ported from llama.cpp
([PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694) by [Pedro Cuenca](https://github.com/pcuenca),
MIT); the engine around it is Xolotl's own.

| | |
|---|---|
| file size | TBD |
| routed experts | TBD |
| dense backbone | TBD |
| arch / tensors | TBD |

---

## The model

Inkling 2 Mini is a 24-layer decoder-only transformer with sparse MoE feed-forward blocks. Its middle 12 layers
(layers 6 to 17) run three times with shared weights, so each token gets 48 layer applications out of 24 layers of
weights, and the cache carries 48 slots.

| | |
|---|---|
| total / active | 26B total, about 3B active |
| loop | layers 6 to 17 run 3 passes, 48 cache slots; passes 2 and 3 read the previous pass through per-layer low-rank loop controllers |
| MoE | 128 routed experts, top-8, no shared expert, sigmoid router with a selection bias, SiTU activation |
| attention | sliding window 1024 with a global layer every 6th (5, 11, 17, 23); no RoPE: a learned relative-position bias per head, with log-scaled positions on the global layers |
| extras | short causal convolutions on K, V, the attention output and the FFN output; block attention residuals that mix earlier depths; final logit softcap 30 |
| vocab | 200,058 |
| context | 1M tokens (the trained key says 1048576) |
| official precision | routed experts NVFP4 on layers 0 to 19, MXFP8 on 20 to 23, everything else BF16 |

The official checkpoint ships in that mixed precision. This build starts from it rather than from a dequantised
copy, so the native FP4 experts can be carried through at their own precision.

## How it was made

1. **Abliteration** on the full-precision parts of the official checkpoint. Two candidates are being measured
   against each other (Heretic-style ARA and the classic refusal-direction projection we used for the Qwen3.8
   Flash-Next release); the one that keeps agentic coding quality best on the Gauntlet ships. Refusal gate numbers TBD.
2. **Quantisation** with XQ: every tensor gets its own format picked by measured sensitivity for a given
   bits-per-weight budget, with the official NVFP4 and MXFP8 experts used as the source of truth for how much
   precision they can afford to lose. The XQ pipeline itself is private.

Xolotl found and fixed a bug in the llama.cpp loop path along the way: passes 2 and 3 of the loop ran their
experts without the per-expert NVFP4 scale when the GGUF keeps NVFP4 experts. The upstream converter dequantises,
so upstream never meets that case. It is fixed in Xolotl's build.

## Quality

Measured against the parent, which is the official checkpoint converted exactly (NVFP4 and MXFP8 kept) and run on
the same engine, hardware and settings.

| | parent | this (XQ) |
|---|---|---|
| Gauntlet v3-small, solved | TBD | TBD |
| Gauntlet v3-small, score (of 6) | TBD | TBD |
| held-out top-1 agreement | n/a | TBD |
| held-out KL | n/a | TBD |
| refusal gate, harmful prompts refused | TBD | TBD |
| refusal gate, harmless prompts refused | TBD | TBD |

The Gauntlet is Xolotl's agentic coding set: hard tasks run over multiple turns with tools in a sandbox, hidden
tests graded after every turn, thinking on, run at the model's recommended sampling over several seeds.

## Speed

TBD. The engine is still being written. Numbers will be measured on an RTX 4090 through the real server and
posted here.

## Files

| file | size | what |
|---|---|---|
| `Inkling-2-Mini-XQ-Uncensored.gguf` | TBD | the model |
| `mmproj-Inkling-2-Mini-XQ.gguf` | TBD | planned: the official vision projector, unchanged, for images (text first) |

## Quick start

Xolotl's `xolotl-inkling` engine runs any Inkling-2-Mini GGUF: the upstream converter's, llama.cpp's quants, or
this file.

```bash
git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
# build per docs/INSTALL.md, then serve with the Inkling config
```

## Credits

The model is [Thinking Machines Lab](https://thinkingmachines.ai)'s, released with open weights under Apache 2.0.
The Inkling 2 forward graph and the GGUF converter come from llama.cpp PR #32694 by Pedro Cuenca (MIT), which
extends the earlier Inkling 2 work in llama.cpp and builds on ggml and llama.cpp by Georgi Gerganov and the
llama.cpp contributors. Xolotl itself is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221.

## License

Apache 2.0, the same as the official model. The [official card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
says, in its General Information section: "**License:** Apache 2.0", with the link to
[LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
