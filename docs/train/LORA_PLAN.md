# LoRA fine-tuning in Xolotl - plan (20 Nov 2026)

Goal (Mal, 20 Nov 18:50): Xolotl trains LoRA adapters on the models it already runs - Hy4-flash, Qwen3.8-Flash-Next
(QFN), DeepSeek-V4.1 - on one PC (RTX 4090 24 GB, 110 GB RAM, NVMe), with the base weights frozen and quantized
exactly as served (Q8_0, XQ mixes), streamed through the same VRAM / RAM / NVMe expert tiers as inference.

Branch `lora-train` (worktree `~/AI/Xolotl-train`, from `hy4-port`). Private ggml tree for the new kernels:
`/media/Crucial1TB/xolotl-train/llama.cpp` (pin 6050b467 + tools/xolotl/ggml/xolotl-ggml.patch); the production tree
`~/AI/Xolotl/third_party/llama.cpp` is never edited in place - new ggml work lands as additions to xolotl-ggml.patch.

## How training differs from inference

- Frozen base, small trainable LoRA (A [r, in], B [out, r]) on chosen matrices. Gradients and AdamW state are MB.
- The backward pass needs (1) the transposed product with every frozen weight on the path (W^T dy), quantized as served,
  (2) a backward rule for every op on the path, (3) the activations the rules need - kept per layer boundary on the host
  and recomputed per layer during backward (checkpointing), so memory is O(layers x tokens x hidden).
- Experts: a step's batch routes to most experts of every layer, so a step streams the expert set twice (forward +
  backward), the way a long prompt does. Throughput is bounded by expert bytes / tier bandwidth, like prefill.

## What exists (pinned ggml)

Backward rules: add/sub/mul/div, sqr/sqrt/log, sum/sum_rows/mean, repeat, rms_norm, mul_mat, scale, set/cpy/cont,
reshape/view/permute/transpose, get_rows, diag_mask, soft_max, rope, silu/relu/exp/neg/step/abs/sgn/softplus, glu
SWIGLU, cross_entropy_loss.  CUDA: cross_entropy_loss(+back), opt_step_adamw, opt_step_sgd, out_prod (F32 only).

## Gaps, in build order

| # | Piece | Why | Gate |
|---|---|---|---|
| K1 | OUT_PROD with a quantized src0 on CUDA (dequant tile -> cuBLAS), all types the engines serve (Q8_0, Q4_K..Q6_K, IQ*, MXFP4, Q2_0...) | mul_mat's backward w.r.t. the activation is out_prod(W, dy^T): today CUDA asserts F32 | grad vs CPU and vs F32-dequant reference, rel err < 1e-5 (F32 accum) |
| K2 | Backward rules: sigmoid, swiglu_clamp, soft_max_ext with sinks (+ sinks grad = none, frozen), concat, scale_bias, clamp, top_k/argsort (no grad: constant) | Hy4/DS4 attention + router + Engram paths | finite differences on random inputs (ggml test-backend-ops grad mode) |
| K3 | Experts inside the per-layer TRAINING graph (decided 20 Nov): the layer's used experts' quantized gate/up/down are uploaded from the tier (arena / file) into a VRAM staging buffer; per expert x_e = get_rows(x, idx_e) -> mul_mat / swiglu_clamp / mul_mat -> * w_e (gathered routing weights, so the router gets its gradient) -> get_rows_back (scatter-add) into the layer output. ggml autodiff does the rest; W^T products go through K1. Needs the GET_ROWS_BACK backward rule (added). A dedicated grouped expert-backward kernel only if this is too slow | the routed experts run outside ggml graphs at inference | grads vs CPU and finite differences on a mini MoE layer (Q8_0 experts), then a real Hy4 layer |
| K4 | Model-specific ops: Hy4/DS4 dsv4_hc_pre/post (the engine's fused path; Hy4Dense already builds them from plain ops), lightning indexer selection (constant mask), QFN gated delta net (chunked scan backward), DS4.1 fq op | gradients must reach every layer below the deepest LoRA | finite differences + per-layer grad check vs PyTorch on a 2-layer slice |
| T1 | Trainer: LoRA params, layer-major forward with host checkpoints, reverse layer loop (recompute + backward), loss = CE on target tokens, AdamW (opt_step_adamw), grad clip, LR schedule, adapter save as a llama.cpp-format LoRA GGUF (what ARA already produces / llama.cpp and Xolotl load) | | loss falls on a tiny overfit set; the saved adapter loaded at inference reproduces the trained logits |
| T2 | Per-model wiring: Hy4 first (its dense half is plain ggml ops = K1+K2 cover it), then DS4.1 (shares hc/MLA/indexer), then QFN (needs the GDN backward) | | per-model overfit test + one real run |

Out of scope for v1: LoRA on the routed experts (per-expert adapters), full fine-tuning, multi-GPU.

## Status

- 20 Nov: plan; private ggml tree set up (identical to production's ggml: diff -rq clean).
- 20 Nov 18:45: K1 written (ggml-cuda out-prod.cu: quantized/F16/BF16 src0, chunked dequant + cuBLAS; supports_op), with
  weight-shaped OUT_PROD cases in test-backend-ops (9 types x 2 shapes x trans_b). K2 written in ggml.c: SIGMOID,
  SWIGLU_CLAMP, CONCAT, CLAMP, GET_ROWS_BACK rules; SOFT_MAX mask/sinks, TOP_K_MASK, ARGSORT, TOP_K marked non-
  differentiable. Build: /media/Crucial1TB/xolotl-train/build.sh (unit xtrain-build, log build.log).
- 20 Nov 19:20: **K1 PASS** - test-backend-ops OUT_PROD on CUDA0 vs CPU: 264 OK / 0 FAIL, incl. the new weight-shaped
  cases for Q8_0, Q4_K, Q6_K, IQ2_XXS, IQ4_XS, MXFP4, Q2_K, F16 (both trans_b; the m4096 x k20000 case = 2 dequant
  chunks). BF16 has no CPU reference (CPU out_prod lacks BF16) - not needed (served weights are quantized).
  **K2 PASS** - grad mode (finite differences) on CPU + CUDA: SWIGLU_CLAMP 18/18 (after a cont() fix for view inputs),
  CLAMP 10/10, SIGMOID 10/10, CONCAT 42/42, GET_ROWS 472/472, SOFT_MAX inplace=0 459 OK. The 97 SOFT_MAX "FAIL"s are
  all sinks=1 rows on BOTH backends and predate this work (the rule did not change); a direct check of softmax+sinks
  (scratch smx.cpp) gives analytic = numeric to 4-5 digits at scale 1 and 0.1, so that is a test-harness artifact.
  SOFT_MAX inplace=1 aborts in ggml's own in-place rule (upstream; filtered with -p inplace=0). Also fixed: DIV's
  backward did not reduce a broadcast divisor (repeat_back added, as ADD/MUL have).
  **T0 PASS** (tools/train/test_xtrain_grad.cpp, mini Hy4-shaped MoE layer, Q8_0 weights, LoRA on a Q8_0 projection):
  CPU finite differences on an F32 copy of the weights: dx 3.2e-3, dA 1.3e-4, dB 1.5e-4; Q8_0 vs F32-copy grads dA/dB
  identical, dx 3.0e-2 (the CPU Q8_0 dot rounds activations to 8 bits: the as-served forward is a step function of x,
  so finite differences must use the smooth copy); CUDA vs CPU dA 4.1e-4, dB 4.6e-4, dx 3.3e-4.
  Lessons for the trainer: route with host-known ids as a FLAT 2-D gather (t * E + id) - GET_ROWS' backward is 2-D only.
  xolotl-ggml.patch regenerated from the private tree (same 22 files + out-prod.cu; applies clean on a fresh 6050b467).
