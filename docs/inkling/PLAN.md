# Xolotl-Inkling: Inkling 2 Mini in Xolotl - engine plan (21 Nov 2026)

Goal (Mal, 21 Nov): Xolotl runs thinkingmachines' Inkling-2-Mini fully - its own engine (`xolotl-inkling`), served by
`serve/server.py` like every other Xolotl model - and the model becomes the next ablation + XQ target. The forward graph is
borrowed from llama.cpp (MIT, credited); the engine around it is ours.

## The model (config.json, rev 6e7c8248f7)

| | |
| --- | --- |
| weights | 26.2B total, 3.4B active (5.4B effective with the loop); official release = mixed precision: routed experts NVFP4 (layers 0-19) and MXFP8 (20-23), everything else BF16, attention-residual queries F32 |
| layers | 24 physical, hidden 2304, vocab 201024 padded / 200058 real, logit softcap 30, 1M context |
| loop | layers 6-17 (`core_start` 6, `core_end` 18) run 3 times with shared weights = 48 layer applications; passes 2-3 read the previous pass ("bank") through per-layer low-rank controllers (msg_query [n_embd, 4], ctrl_down [2*n_embd, 64], ctrl_gain / ctrl_forget [64, n_embd], for Q, K, V and FFN) |
| attention | GQA 24 q / 8 kv heads of 128, NO RoPE: a learned relative-position bias (r = W_rel x, 16 dims/head, clipped; table [16, extent] gathered by distance), extent 1024 on global layers / the window on SWA layers; SWA window 1024 on the pattern's layers; per-query sink logit (attn_sink_proj); Q and K L2-clipped to 22.6 (Q per-channel scaled first), V RMS-normed without gamma; global layers multiply q and r by tau = 1 + 0.1*log(max(1, (pos+1)/8192)); sigmoid output gate |
| short convs | causal k=4 depthwise conv with residual on K, V (global layers add input-dependent taps per head), the attention output and the FFN output - 4 conv states per layer |
| attention residuals | every 6 layers a new residual stream starts; each layer's attention input (1 query on SWA layers, 3 = Q/K/V on global ones) and FFN input are softmax mixes over the earlier block outputs + the current stream (mix()) |
| MoE | 128 experts, top-8, sigmoid gating + selection bias, normalized weights, route_scale 8, fused gate_up [n_embd, 2*1152, 128] + down; activation SITU = 4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25) |
| extras | vision tower (23 layers, patch 20, merge 2) -> mmproj; MTP drafter (3 layers, block 8, markov head) in mtp.safetensors |

## Where the pieces come from

| piece | reference | Xolotl home |
| --- | --- | --- |
| forward graph (base Inkling 2) | llama.cpp `src/models/inkling2.cpp` (#32217 merged + #32694 refactor) | `src/program/inkling_main.cpp` |
| loop | llama.cpp PR #32694 (pcuenca) `src/models/inkling2-loop.cpp` | same file |
| relative-position inputs, iSWA masks | llama.cpp `llama-graph.cpp` build_inp_mem_hybrid_iswa_rel, `set_input_relative_pos` | engine: we build rel_pos / masks on the host per ubatch |
| MoE with SITU, sigmoid gating | llama.cpp `build_moe_ffn` (LLM_FFN_SITU) | engine: ggml_mul_mat_id + plain ops |
| converter (HF -> GGUF, keeps NVFP4 / MXFP8) | PR #32694 `conversion/inkling2.py` (on top of upstream) | used as is for the GGUF; a Xolotl copy only if it needs changes |
| tokenizer + chat format | tiktoken o200k-style (tokenizer.json), upstream chat parser #32381 | `tools/strata_tokenizer.py`, a serve-side turn format (like serve/atem.py) |
| engine skeleton: serve line protocol, prefix reuse, park, sampling | `src/program/glimmer_main.cpp` / `lfm3_main.cpp` | inkling_main.cpp |

Every op the graph uses exists in Xolotl's pinned ggml (6050b467); #32217 touched only CUDA flash-attention dispatch
(checked when attention is wired). Reference build: `/media/Crucial1TB/llama.cpp-inkling` (PR #32694 head, CUDA sm_89).

## State per sequence

48 layer applications = 48 cache slots (24 physical + 2 x 12 looped), each with its own K/V and its own 4 conv states.
Global layers: K/V over the whole context (positional, append-only). SWA layers: a ring of window + ubatch cells.
Conv states: (k-1) = 3 inputs x (2*1024 + 2*2304) channels per slot, ~80 KB. Checkpoint for prefix reuse = SWA rings +
conv states taken one token before the end of the prompt (the Glimmer scheme).

## Stages and gates

| # | stage | gate |
| --- | --- | --- |
| S0 | reference: build PR #32694, convert the official checkpoint to GGUF keeping NVFP4 / MXFP8, golden dumps (per-layer l_out, logits) on fixed prompts | reference generates coherent text; dumps saved |
| S1 | engine skeleton from glimmer_main: GGUF load (strata reader + ggml backend), weights on CUDA, serve protocol, sampler | READY, tokens out (graph stubbed) |
| S2 | graph: one layer (SWA, then global) incl. convs, rel-pos bias, sinks, clips, attention residual mix, MoE SITU | per-layer l_out vs reference < 1e-3 rel on the golden prompt |
| S3 | full model incl. the loop and the output mix | logits KL vs reference ~0; greedy text identical for 128 tokens |
| S4 | decode state: KV slots, SWA rings, conv states, append-only resume, checkpoint, park | prompt+decode == one-shot prefill logits; multi-turn reuse test |
| S5 | serve: family `inkling2`, chat template + tool-call parser, tokenizer, example config | serve tests; a real agent turn with tool calls |
| S6 | vision (mmproj), MTP drafter, speed (NVFP4/MXFP8 on sm_89, CUDA graphs) | image Q&A works; tok/s logged in BENCHMARKS |
| S7 | docs/INKLING.md, README, CHANGELOG, CREDITS (thinkingmachines, llama.cpp #32217 / #32694 / #32381) | review, then push on Mal's word |

After S3: ablation (refusal direction on the reference, applied to the BF16 parts) and the XQ pipeline (private - nothing of
it lands in this repo). The official NVFP4/MXFP8 experts are the source precision for both.

Evaluation (Mal, 21 Nov): the PARENT model - the official checkpoint converted exactly (NVFP4/MXFP8 kept) and run on
xolotl-inkling - is the baseline every derivative is compared against, on the same box, same harness, same settings:
parent vs ablated vs XQ vs ablated+XQ on the Gauntlet (greedy) and on held-out logit KL / top-1 against the parent.
Mini is not on OpenRouter (21 Nov), so there is no API run; the comparison is local only.
