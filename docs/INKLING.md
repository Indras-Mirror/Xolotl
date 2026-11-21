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

Speed work is ongoing (fused kernels, flash attention with the relative bias, faster speculative decoding); these
are the first numbers, not the ceiling.

## Speculative decoding with the DSpark (MTP) drafter (experimental, opt-in)

Inkling ships a block drafter in `mtp.safetensors`: 3 dense layers that read the target's hidden states at layer
visits 5, 23, 41 and the final norm, and propose up to 7 tokens per step through a rank-256 Markov head.
`--dspark <drafter.gguf>` turns it on (the GGUF is the `dspark/` file of
[unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)); `xolotl-inkling-dspark` runs the
drafter alone against saved target taps.

- Greedy requests decode speculatively; sampled requests decode normally and keep the drafter fed.
- The short convolutions make the model recurrent, so a rejected draft is undone exactly: every verify keeps its
  `[conv state ; chunk inputs]` and the state after the accepted prefix is copied back (no second forward). Sliding
  ring cells of rejected positions are marked empty; global K/V is positional.
- Draft length is dynamic: drafting stops when the cumulative top-1 probability falls below
  `XOLOTL_DSPARK_TAU` (default 0.35), at most `XOLOTL_DSPARK_MAXD` (7). Each verify length keeps its own graph.
- The drafter's Markov head, context projection, softcap and argmax over the 200K vocabulary run on the GPU
  (draft 25 ms -> 4 ms per step).
- Correctness: verify lengths 1, 5, 6 and 7 are bit-identical to plain decoding. Lengths 2 to 4 use a different
  reduction order in ggml's quantised mat-vec, so a greedy near-tie can flip after a few hundred tokens (rounding,
  not a logic error). `XOLOTL_SPEC_EXACT=1` pads those lengths for gates.
- Speed is still being tuned: on a test quant, a fixed 3-token verify gives 76.5 tok/s vs 65.4 plain. Each token
  picks 8 of 128 experts, so a verify chunk reads the union of its tokens' experts; a grouped MoE kernel for
  verify is the next lever.

## K/V cache types

`--kv-type f16|q8_0|k8v4` sets the cache of the global (full-context) layers; the sliding-window rings stay f16.
Attention over a quantised cache dequantises per chunk with an online softmax, so memory stays bounded at long
context. Against f16 on the parent: q8_0 KL 3.6e-4 with identical greedy output, k8v4 (q8_0 keys, q4_0 values)
KL 8e-4. f16 is bit-identical to before.

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
ninja -C build-xolotl xolotl-inkling xolotl-inkling-dspark   # configured as in AGENTS.md
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
- Not yet: vision (the model has a 23-layer vision tower; in progress), flash attention (the pinned ggml has no
  flash-attention kernel for per-head masks, which the relative bias needs).
