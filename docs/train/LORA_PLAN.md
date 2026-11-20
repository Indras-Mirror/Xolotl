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
  Gate build: `L=/media/Crucial1TB/xolotl-train; g++ -std=c++20 -O2 -I tools/train -I $L/llama.cpp/ggml/include
  tools/train/test_gdn_chunked.cpp -L$L/build-cuda/bin -lggml-cpu -lggml-base -lggml -Wl,-rpath,$L/build-cuda/bin
  -lpthread -o $L/bin/test_gdn_chunked` then `test_gdn_chunked S H T 1e-2 <stride> /tmp/g.bin` + `python3
  tools/train/gdn_ref.py /tmp/g.bin` (needs the rebuilt libggml-base/libggml-cpu: the backward fixes are in them).
  Baseline note (so nobody re-litigates it): `test-backend-ops grad -b CPU -o '^SCALE'` aborts at ggml.c's
  `!node->view_src || ...` assert and prints one `SUM ... MAA` FAIL - BOTH reproduce with the three Xolotl hunks
  reverted (verified 20 Nov 21:4x), so they predate this work and are harness/in-place artifacts. MUL_MAT_HADAMARD
  grad fails with MAA = 1.0 because it has no backward rule (also pre-existing).
- 20 Nov 21:35: **QFN layer map written** (`docs/train/QFN_LAYER_MAP.md`). Op-gap list verified by grep of
  ggml_compute_backward: MUL_MAT_ID has NO backward and no decomposed fallback (rebuild the expert loop Hy4-style);
  SET_ROWS has no rule (QSA mask scatter + KV writes); LIGHTNING_INDEXER is fused with no fallback but its weights are
  a constant fill(1/sqrt(idx_dim)) and it only feeds zero-grad selection - needs a test, not an assumption. The private
  tree already carries `cparams.training` (llama-cparams.h:58, set by llama_opt_init) and a dense-attention training
  bypass (llama-graph.cpp:3258), so llama.cpp's own finetune path is a second possible route (time-boxed probe).
- 20 Nov 22:05: GDN gate re-run at the PACK's own geometry (`ssm.state_size` 128, 16 k heads, 48 v heads): T=64
  (1 chunk) forward 1.9e-9 abs / 6.8e-7 rel, FD rel 4e-6..1.3e-5; T=128 (2 chunks, state path) forward 1.6e-9 /
  5.5e-7 rel, FD rel 7e-6..1.9e-5 (dg sampled only idx 0, both sides 0). Head expansion rule verified: llama.cpp
  `ggml_repeat_4d` TILES, so v head j reads k head j % n_k_heads (16) - recorded in the header and QFN_LAYER_MAP.md.
  `test_gdn_chunked` gained an input-scale arg so the loss stays O(1) at production dims (unscaled it reaches 1e12 and
  every metric is relative).
- 20 Nov 22:35: **QFN layer CORE as one training graph - PASS.** `tools/train/test_gdn_layer.cpp` builds what
  `qwen4exp.cpp build_layer_attn_linear` does between the qkv projection and the output projection: conv input
  (zero history = position 0) -> `ggml_ssm_conv(4, channels)` -> SILU -> channel split q|k|v -> `build_gdn_l2_norm`
  (= rms_norm(x, eps/S) * 1/sqrt(S)) on q,k -> `ggml_repeat_4d` head tile (48 v heads from 16 k heads, j % 16) ->
  `gdn_chunked`. Forward vs `tools/train/gdn_layer_ref.py` (numpy, same composition): 1.9e-8 abs / 3.8e-7 rel at
  S8/nk2/nv6/T96 (2 chunks). FD (stride 4, 1980 elements each): dx 1.1e-4 (|g| 0.19), dc 9.8e-5 (|g| 0.30),
  dg 1.8e-6, db 9.8e-7; mean relative 1.6e-5..6.7e-5. The FD alone would only prove the graph is self-consistent,
  hence the reference - together they prove the graph IS the model's layer core.
  `gdn_ref.py` now exposes `gdn_forward(...)` (the chunk loop, reused by the layer reference).
  Exhaustive re-run (stride 1, S8/nk2/nv6/T96): all 7,920 x elements + all c/g/b: dx 2.6e-4 (|g| 0.23), dc 2.0e-4
  (|g| 0.31), dg 2.2e-6, db 1.5e-6; mean relative 1.5e-5..6.0e-5; forward 1.9e-8 abs / 3.8e-7 rel.
- 20 Nov 23:20: **MoE expert loop as a differentiable graph - PASS** (`tools/train/moe_train.hpp`): the replacement
  for MUL_MAT_ID, mirroring build_moe_ffn as qwen4exp calls it - softmax router, top-k selection host-side (constant,
  as TOP_K/ARGSORT are zero-grad anyway), weights gathered from probs over the flat index t*E + expert, normalised
  over the k selected (F16-min clamp), then one pass per used expert: get_rows -> gate/up mul_mat -> silu(gate)*up ->
  down mul_mat -> * weight -> get_rows_back scatter-add. Gates: forward vs `tools/train/moe_ref.py` 1.1e-7 rel
  (D16/F12/E6/K2/T10) and 2.3e-7 at production shape D64/F32/E512/K10/T8; FD on every element of cur, the router and
  the expert matrices: dx 6.4e-6 (|g| 0.46), drouter 3.9e-6, dup 2.4e-6, dgate 2.5e-6, ddown 9.3e-7; mean rel
  7e-6..3e-5. At 512 experts the sampled mean rel rises to 1.3e-4..3.9e-4 purely because each expert now serves
  0-1 tokens (small |g|), max abs err stays 5e-6.
  **Two bugs the gates caught (worth remembering):** (1) the first version dropped the up branch, so `up_exps` got a
  null gradient - caught by a segfault on grab(), not by FD; (2) the weight gather needs the flat index t*E + expert,
  not the expert id - the FD check passed on that wrong graph (analytic and numeric were wrong together), only the
  numpy reference caught it. Hence: every differentiable component gets BOTH a FD gate and a reference forward.
- 20 Nov 23:55: **one COMPLETE QFN gated-delta-net layer as a training graph - PASS**
  (`tools/train/qfn_layer_train.hpp`, `qfn_gdn_layer`): wqkv_z projection + qkv/z split, conv with zero history
  (tokens on dim0, as ssm_conv wants), SILU, q|k|v split, l2 norm, repeat-tile to 48 v heads, beta/alpha from
  w_ba (beta = rows [0,n_v), alpha = [n_v,2n_v) - the alpha view's offset in qwen4exp.cpp:1133 pins the order),
  gate = softplus(alpha + dt) * ssm_a, gdn_chunked, gated RMS norm (sigmoid gate, not silu), ssm_out, residual.
  Forward vs `tools/train/qfn_layer_ref.py`: 4.5e-7 rel at T=20 (1 chunk) and 6.3e-7 at T=70 (2 chunks, state path).
  FD on every element of cur and all 7 weight tensors: the downstream pieces are 1e-6..1e-4 relative; the pieces
  upstream of the delta net (cur, wqkv, conv) show a larger gap that is **pure central-difference truncation** -
  dconv's worst element goes 1.50 (eps 1e-2) -> 3.84 (3e-2) -> 0.23 (3e-3) -> 2.6e-3 (3e-4), i.e. linear in eps
  and still falling. A real gradient error would be a constant offset (that is how the MoE index bug presented).
  Rule for this layer: FD at eps ~3e-4 (still above the F32 noise floor) or judge by convergence in eps.
- 20 Nov 21:58: **FIRST QFN TRAINING RUN on real pack weights - PASS** (`tools/train/qfn_train.cpp` +
  `tools/train/qfn_export.py`). Real `blk.0` gated-delta-net tensors dequantized from the 45 GB XQ pack
  (`/media/Crucial1TB/xolotl-train/qfn_l0.f32`, D 2560, hd 128, 16 k / 48 v heads, channels 10240), two real
  token-embedding windows (32 tokens each, token_embd rows 100-163). Frozen base, rank-8 alpha-16 adapters on
  `blk.0.attn_qkv.weight`, `blk.0.attn_gate.weight`, `blk.0.ssm_out.weight`, AdamW lr 3e-3, loss = ||layer(x_a) -
  target||^2 with target = the same layer's base output on the other window (a memorisation gate).
  **400 steps, 2m08s on CPU (16 threads, ~0.3 s/step): loss 9.667 -> 0.008 (the last-third scatter 0.002-0.018 is
  AdamW noise at lr 3e-3), saved `qfn_l0_lora.gguf`; reloading it into a fresh graph gives relative L2 vs target
  base 1.410 -> adapted 0.041 (34.6x).** GGUF round trip bit-exact, `general.type=adapter`, arch `qwen4exp`,
  alpha 16. Adapter/base tensor pairing checked against the pack itself: all three pairs OK.
  **Bug the pairing check caught:** the first version put one adapter on the CONCATENATED qkv+z (16384 rows) - no
  such base tensor exists in the pack, so llama.cpp could never apply it. The layer now keeps `attn_qkv` and
  `attn_gate` separate (an adapter can only attach to a real base tensor); the layer gates still pass after the split.
  Not yet done: loading the adapter in llama.cpp itself (no llama-cli in this build tree) - that is the external
  check, and it needs the full model resident. Also not a fine-tune: the target is synthetic (memorisation), because
  real gradients need real activations from the served model (next step).
- 21 Nov 04:20: **2-LAYER STACK TRAINING PASS** (`tools/train/qfn_train_stack.cpp`). Real blk.0 + blk.1 of the XQ pack
  chained (each layer's residual output feeding the next), rank-8 adapters on attn_qkv / attn_gate / ssm_out of BOTH
  layers (6 adapters), T=16, 100 steps, lr 3e-3: **loss 2.409 -> 0.0004 in ~2 min (CPU, 16 threads)**; saved
  `qfn_stack2_lora.gguf`, reloaded bit-exact, **relative L2 vs target 1.4318 -> 0.0130 (109.9x)**. All 6 adapter
  pairs checked against the pack's real tensors: PASS. This is the first run where each layer's gradient travels
  back through the layers above it - the depth path is exercised, not just a single layer's internals.
- 21 Nov 04:35: **Llama.cpp application convention verified against the source** (`src/llama-graph.cpp:1652-1671` +
  `src/llama-adapter.h:53-57`): `scale = adapter_scale * alpha / rank` with `rank = b->ne[0]`, and the delta is
  `scale * B (A x)` added to `W^T x`. Our writer emits `a` ne = [n_in, rank], `b` ne = [rank, n_out] and the trainer
  applies exactly `(alpha/rank) * B (A x)`, so a saved adapter behaves identically inside llama.cpp at the default
  `adapter_scale = 1.0`. (The structural check - `w->ne[0] == a->ne[0]`, `w->ne[1] == b->ne[1]` - is what the moe/loader
  path validates; both hold for all 6 stack adapters.)
- 21 Nov 04:45: **loader-equivalence check for adapters** - `tools/train/lora_loader_check.py` replicates, from
  `src/llama-adapter.cpp:195-390`, every condition that makes llama.cpp accept or throw (general.type/adapter.type,
  arch equality, tensor existence in the base model, a/b pairing, `W.ne0 == a.ne0`, `W.ne1 == b.ne1`,
  `a.ne1 == b.ne0`, and the resulting `scale = alpha / rank`). Both adapters pass on all 9 checks:
  `qfn_l0_lora.gguf` (3 weights) and `qfn_stack2_lora.gguf` (6 weights), scale 2.000 each.
- 21 Nov 04:55: **adapter loaded by llama.cpp ITSELF - PASS** (the external check that was missing). Built
  `llama-completion` (this tree names the CLI that, not llama-cli) against the private ggml and ran the real
  `qwen4exp` pack with `--lora`:
  * positive: `-m Swift1.5-QFN-XQ-45GB.gguf --lora qfn_l0_lora.gguf` loads with NO error and generates; greedy
    output on "The capital of France is" changes from `Paris. The capital of Germany is Berlin` (base) to
    `</think>![` (adapted) - the adapter is genuinely applied at inference.
  * negative control: the same flag pointed at the model file itself fails with
    `failed to apply lora adapter: expect general.type to be 'adapter', but got: model` - proving the loader's
    checks really execute (so the positive run is meaningful, not a silently ignored flag).
  So the whole chain is proven with real artefacts: train -> save GGUF -> llama.cpp accepts it -> inference changes.
  Note: CPU-only in this run (the 4090 was busy with chronos-qx's arm), `-ngl 0`, ctx 64.
- 21 Nov 05:10: **real-activation path mapped** - `tools/train/qfn_layer_dump.py` reads a strata `--dump-layers`
  file into per-layer (input, target) pairs. Layout read from the source (`src/core/session.cpp:662-670`,
  `757-763`): per position one record of `(n_layers + 1) * hc * n_embd` f32, slot 0 = the input residual (the
  embedding broadcast to the hc streams), slot l+1 = the residual after layer l, each slot stream-major
  `hc x n_embd`. Self-tested against a synthetic file with known slot values: position 2 / layer 1 extracts
  slot 1 as input and slot 2 as output. `--summary` prints per-slot RMS so a real dump can be sanity-checked
  (slot 0 should look like embeddings, the last slot much larger).
  What it buys: matching slot l+1 from slot l is RECONSTRUCTION on the model's own data, not a preference
  fine-tune - the honest first target on real activations.
- 21 Nov 05:25: **real-activation capture prepared** - `tools/train/qfn_dump_layers.sh <tokens-file> <out.bin>` runs
  the engine once in generate mode with `--dump-layers` and checks the output size against the pack's own geometry.
  Two source facts it encodes: (1) `--dump-layers` is written by `session_loop`, so `--no-capture` / `--no-pool` must
  not be set; (2) ARMING `--dump-layers` TURNS FUSED GR OFF (`generate.cpp:4579`), so its numbers come from the
  non-fused path - never compare them against a fused-GR run without accounting for that.
  Geometry from the pack: `block_count` 48, `hyper_connection.count` 4, `embedding_length` 2560 ->
  **2.01 MB per position** (501,760 floats); 200 decoded tokens would be ~400 MB.
  Not yet run: the 4090 is in use by chronos-qx's arm series (verified: their strata --serve holds ~8.5 GiB).
- 21 Nov 05:45: **real-activation capture is BLOCKED in this build - verified in source and on the GPU.**
  `strata --dump-layers` cannot coexist with the native (IQ) pack this model is served from:
  (a) a native pack requires `--spec T` with T >= 2 (`generate.cpp:4394-4400`);
  (b) `--spec` needs the token graph, and `generate.cpp:8335`/`8400` build that graph only when
  `layer_dump == nullptr && half_dump == nullptr` - i.e. arming either dump removes exactly what `--spec` needs.
  Four runs on the real pack (now with the 4090 released by chronos-qx) each stopped at a different link in that chain:
  `--prefill auto` refused (needs a chunk), `--prefill 2048` OOM'd in device buffers (361 MiB free),
  `--prefill 256 --spec 2` prefilled fine then died on "--spec needs the device residency table ... and the token
  graph", and `--prefill 256` without `--spec` was refused up front. `--dump-halves` has the identical condition.
  Recorded in `tools/train/qfn_dump_layers.sh` so nobody re-burns a GPU slot: the next step is an ENGINE change
  (keep the token graph while dumping, or allow a dumped non-spec run on a native pack), not a flag combination.
- 21 Nov 06:10: **hyper-connections: PORTED BUT NOT VERIFIED - the FD gate does not complete.** `qfn_hc_mix` /
  `qfn_hc_combine` (non-fused paths of qwen4exp.cpp 347-414 / 416-444) are in `tools/train/qfn_layer_train.hpp` and
  `tools/train/test_qfn_hc.cpp` builds against them, but the test hangs and I stopped rather than keep burning time.
  What is actually known:
  * the SMALL configuration completes: D=4 hc=2 hd=2, with `ggml_graph_compute` at **nth=1** - forward graph, backward
    graph and compute all finish and the loss prints;
  * at **nth=8** the same case deadlocks - 8 threads spinning, ~140 min CPU over 17 min wall, no progress
    (NOTE: part of that repeated burn was my own: a file restore silently put `nth=8` back, so I re-ran the case I had
    already diagnosed. Check the thread argument before re-running a "hang".);
  * at D=8 hc=3 hd=4 it spins single-threaded at 100% CPU for 10 min with no output - an infinite loop the small case
    does not hit. Not yet localised (candidates: the hand-rolled `ggml_view_2d` strides in `qfn_hc_mix`, the
    `ggml_repeat_4d` broadcast in `qfn_hc_combine`, or the rms_norm+MUL chain in backward).
  **Nothing here should claim hyper-connections are validated.** The validated pieces (conv -> l2 norm -> head tile ->
  GDN -> gated norm -> out proj, and the MoE) do not include them.
  Lesson: run every new gate at a SMALL shape with nth=1 first, then scale - two of these failures were threading,
  not maths.
  **Bisected since (bisect_hc.cpp): both halves are individually FINE.** `qfn_hc_mix` alone (D8 hc3 lr6 T8, nth=1):
  forward graph, backward graph and compute all complete, loss -0.008787. `qfn_hc_combine` alone, same shape,
  completes, loss -0.032751. So neither op is broken on its own; the hang appears only in the CHAINED sequence
  (mix -> block -> combine -> mix -> MLP -> combine), at nth=1, at both hc=2 and hc=3. Next step is mechanical: add
  one stage at a time to bisect_hc.cpp rather than guessing which line is slow.
