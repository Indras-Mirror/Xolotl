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
- 20 Nov 19:40: **T1 Hy4 trainer converges** - 64 tokens, rank-8 adapters on layers 38-41 attention (16 adapters,
  1.30 M params), loss 6.774 -> 0.215 in 30 steps, backward 1.2 s/step (4 layers), forward ~4 s (reference CPU experts);
  adapters saved to /media/Crucial1TB/xolotl-train/lora-t1.bin. OPEN: whole-model FD on blk.41.attn_output B gives
  analytic -0.0484 vs numeric -0.0922 (1.9x) - HY4_TRAIN_CHECK=1 (train-graph output vs forward per layer) is built and
  next. Adapter-at-inference test (HY4_LORA_LOAD, t1prompt/) built, not run. Handover: ~/.quetza-data/RESUME-2026-11-20-LORA-QFN.md.
  Reference docs from Basalt/Adamant (MIT) saved in docs/train/ref/.
- 20 Nov 20:10: **FD question RESOLVED - the gradient is correct.** HY4_TRAIN_CHECK=1 (`~/v41-work/hy4/check1.log`): the
  training graph's layer output vs the inference forward's, layers 41..38, rel err 1.3e-3 / 1.1e-3 / 8e-4 / 5.6e-4 (both
  steps) - the train graph IS the served forward. HY4_FD_DIR (new; directional derivative along the unit gradient over
  all 1.3 M adapter params, `fd2.log`): numeric/analytic = 2.130 (eps 0.003), 1.369 (0.01), 1.092 (0.03), **0.996 (0.1)**;
  the excess is a constant ~0.007 loss noise floor (activation rounding in quantized products + top-k expert flips),
  so it shrinks as 1/eps. The single-element FD flips SIGN across eps (+0.051, +0.072, -0.069, -0.054 vs analytic
  -0.048): pure noise - the earlier "1.9x" was noise. Rule: on the as-served forward, check gradients directionally
  with eps ~0.1, never per element.
  Adapter GGUF (5c, writer + loader): `tools/train/lora_gguf.hpp` (llama.cpp format: general.type=adapter,
  general.architecture=<base>, adapter.type=lora, adapter.lora.alpha; `<W>.lora_a` [n_in, r], `<W>.lora_b` [r, n_out],
  F32; reader takes F32/F16). hy4_train HY4_TRAIN_SAVE=*.gguf writes it; hy4_generate HY4_LORA_LOAD takes .gguf or the
  raw .bin (alpha from the file). `tools/train/lora_bin_to_gguf` converted lora-t1.bin -> lora-t1.gguf, round trip
  EXACT, gguf-py reads keys + 32 tensors (arch hy_v4_flash).
- 20 Nov 20:50: **5b PASS - the trained adapter works at inference.** `hy4_generate_lora` with
  HY4_LORA_LOAD=lora-t1.gguf (via the new GGUF reader) on the first 24 training tokens: the 40 generated ids equal
  `t1prompt/expected.txt` exactly (40/40); the base model produces different text (`~/v41-work/hy4/lorainf.log`). So
  T1 is complete on Hy4: train -> save GGUF -> load -> reproduces the trained continuation.
  **QFN (5e) scoping:** the pack is `qwen4exp` (48 layers: 36 gated-delta-net + 12 full-attention with the QSA
  lightning indexer and compress ratio 4, hyper-connections hc=4 with low-rank 320 mix, PLE per-layer embeddings,
  512 experts top-10, latent 640; weights Q8_0/BF16/F32 dense, experts Q2_0/IQ1_M/IQ4_XS/IQ4_NL). The Strata QFN engine
  is native CUDA (no ggml graphs), so the trainer will be a per-layer ggml graph ported from the private llama.cpp's
  `src/models/qwen4exp.cpp` (its non-fused paths: build_hc_mix/combine from plain ops; delta net via
  delta-net-base.cpp build_delta_net_chunking) - the Hy4 trainer structure, forward = the same graphs.
  **K4 GDN backward rules (ggml.c) + SOLVE_TRI upper:** SOLVE_TRI (dB = A^-T G via a new upper-triangular solve: CPU
  back substitution, CUDA via the cuBLAS path with the fill mode flipped; op param 0 = upper, 0 keeps old graphs),
  dA = -tril(dB X^T); TRI (mask the grad); CUMSUM (sum - cumsum + g); SSM_CONV (dsx = sum_k pad_k(g^T * c_k), dc =
  per-k reductions); PAD (crop, non-circular). Gates: test-backend-ops grad mode CPU + CUDA OK for TRI, CUMSUM,
  SSM_CONV, PAD; SOLVE_TRI forward incl. 3 new upper cases CUDA vs CPU 27/27 OK; SOLVE_TRI backward checked by
  `tools/train/test_xtrain_solve_tri.cpp` (max abs vs central FD, n 10/17/30, 3-D + 4-D batches, 1 + 16 threads:
  4e-7..4e-5). test-backend-ops' own SOLVE_TRI grad check stays red: its relative per-element metric divides by
  (gn + ga), which is 0 on the untouched upper triangle - not a gradient error (same gradients score 1e-4 when the
  0/0 entries are skipped). Test-side changes live in `tools/train/test-backend-ops-train.patch` (production patch
  stays ggml/ only; both apply clean on 6050b467).
- 20 Nov 21:05: **5d GDN as a differentiable ggml graph - PASS.** `tools/train/gdn_train.hpp` (`gdn_chunked`) is a
  line-for-line port of llama.cpp `build_delta_net_chunking` (non-KDA path, which is what QFN uses: its gate is
  [1, H, T]) built only from ops that have backward rules. Two changes were needed for autodiff, both semantics-neutral:
  the identity matrix is an input (ggml_fill/ggml_diag have no backward) and the per-chunk outputs are CONCATenated
  instead of ggml_set_inplace into v (in-place writes break backward graphs). Forward gate: python reference
  `tools/train/gdn_ref.py` (same algorithm, ggml op conventions verified numerically) agrees to 1.2e-6 (rel 2.7e-7) at
  S16/H2/T100 (2 chunks incl. 28 padded), and at T=4/64/128/200 shapes. Gradient gate: `tools/train/test_gdn_chunked.cpp`
  central FD on every element of q,k,v,g,b (10k elements, eps 1e-2, stride 1): max abs err dq 6.9e-5, dk 1.3e-4,
  dv 6.4e-5, dg 1.5e-4, db 1.2e-4 against gradient magnitudes 1.2..10.5 (mean relative 1e-5..1e-4 = FD noise).
  Autodiff gaps this exposed and fixed in ggml.c (all backward-only): (1) ADD/SUB/MUL backward assume equal shapes, so
  gdn_chunked expands every grad-carrying broadcast operand with ggml_repeat_4d (the reference gets away with implicit
  broadcast because it never trains); (2) PERMUTE/TRANSPOSE backward returned bare strided views, which the binary-op
  CPU kernel rejects as src0 (nb00 != 4) - both now ggml_cont; (3) SCALE backward hit the same thing via a permuted
  grad - it conts when !ggml_is_padded_1d.
