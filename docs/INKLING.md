# Inkling-2-Mini on Xolotl

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache 2.0) is a 26B
looped MoE with about 3B active parameters. `xolotl-inkling` is Xolotl's engine for it, behind the usual
OpenAI- and Anthropic-compatible server.

| | |
|---|---|
| parameters | 26B total, about 3B active |
| layers | 24 physical, 48 applications per token: layers 6 to 17 run 3 passes, with low-rank loop controllers |
| experts | 128 routed, top-8, sigmoid router with a selection bias, SiTU activation, no shared expert |
| attention | sliding window 1024, a global layer every 6th; no RoPE (learned relative-position bias, log-scaled positions on the global layers) |
| extras | short causal convolutions on K, V and both layer outputs; block attention residuals that mix earlier depths |
| context | 1M trained (`--context` sets what you allocate) |

## Numbers (RTX 4090, exact NVFP4 parent, all weights in VRAM, 21 Nov 2026)

| | |
|---|---|
| decode | **57 tok/s** at a short context, 51 tok/s at 8K (llama.cpp reference build: 51) |
| prefill | 1.1K tok/s at 8K |
| parity vs the llama.cpp reference | logits KL 2e-4 and top-1 100 % on a short prompt; KL 3e-3, top-1 97.7 % teacher-forced over 1.6K tokens of text; 64 greedy tokens identical on 2 of 3 prompts, the third splits at a near tie after 215 of 248 characters |
| VRAM | 20.5 GiB weights + 0.75 GiB state at 16K context (token table in RAM) |

Speed work is ongoing (fused kernels, flash attention with the relative bias, MTP drafting); these are the first
numbers, not the ceiling.

## Get a GGUF

- **Exact parent (what the numbers above use):** the official checkpoint keeps its experts in NVFP4 (layers 0-19)
  and MXFP8 (20-23). llama.cpp's converter from [PR #32694](https://github.com/ggml-org/llama.cpp/pull/32694)
  dequantises them; `tools/inkling/convert-keep-nvfp4.patch` makes it keep the NVFP4 bits (+ a per-expert `.scale`)
  and writes MXFP8 as BF16 (exact). 22.9 GB. `tools/inkling/check_nvfp4_exact.py` checks the experts bit for bit.
- **Upstream GGUFs and llama.cpp quants** of Inkling-2-Mini load too (same tensor names; the `.scale` tensors are
  optional). Not yet tested here; a dequantised BF16 file is ~52 GB, so on a 24 GB card use a quant or
  `--cpu-exps`.

## Build and run

```bash
ninja -C build-xolotl xolotl-inkling            # configured as in AGENTS.md
python tools/inkling/export_tokenizer.py <Inkling-2-Mini HF dir> <pack dir>   # once: the server's tokenizer pack
# edit configs/xol-inkling.json (gguf path, tokenizer pack, context), then:
.venv/bin/python -m serve.server --engine strata --config configs/xol-inkling.json --port 8134
```

`--cpu-exps 20-23` keeps those layers' experts in host RAM (the CPU computes them) for more context on a 24 GB card.
Thinking effort follows `reasoning_effort` (the chat template's effort line); sampling defaults to the model's
`generation_config.json` (temperature 1.0).

## Notes

- The forward graph is ported from llama.cpp PR #32694 by Pedro Cuenca (MIT). One bug in that PR is fixed here:
  with NVFP4 experts kept, loop passes 2 and 3 ran their experts without the per-expert scale (upstream's
  dequantised GGUFs never hit it); `tools/inkling/ref-loop-expert-scale-fix.patch` is the fix for the reference.
- The conv state is recurrent, so a request resumes by appending to what the engine read, or from the checkpoint
  one token before the end of the previous prompt (sliding rings + conv states), else from token 0.
- Not yet: vision (the model has a vision tower), flash attention, MTP drafting.
