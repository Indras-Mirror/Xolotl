# QFN (qwen4exp) layer map for LoRA training (20 Nov 2026)

Scope: what a QFN layer's forward is built from in the private llama.cpp
(`/media/Crucial1TB/xolotl-train/llama.cpp`, `src/models/qwen4exp.cpp`), and which of those ops still lack a
backward rule. This is the work list for porting QFN into the trainer (plan: `docs/train/LORA_PLAN.md` section 5e).

Verified in this session: the op-gap counts below (grep of `ggml_compute_backward`), the `cparams.training` mode
(`src/llama-cparams.h:58`, set by `llama_opt_init`), the dense-attention training bypass
(`src/llama-graph.cpp:3258`), and the constant lightning-indexer weights (`qwen4exp.cpp:877`). The per-function line
numbers come from a read-only mapping pass and were spot-checked, not all read line by line.

## Layer inventory

48 trunk layers: 36 gated-delta-net (linear attention) layers + 12 full-attention QSA layers at `(il+1) % 4 == 0`
(il = 3, 7, ..., 47). il = 48 is the MTP block (a 13th QSA layer).

| Sub-graph | Where | Ops | Backward status |
|---|---|---|---|
| HC pre/mix | `build_hc_mix` 347-414 | RMS_NORM MUL RESHAPE MUL_MAT SCALE VIEW SILU SIGMOID ADD CONT | fused `ggml_dsv4_hc_pre_gated` (386) has **no backward**; **non-fused fallback exists at 390-405** |
| HC post/combine | `build_hc_combine` 416-444 | SIGMOID SCALE RESHAPE REPEAT MUL ADD | fused `ggml_dsv4_hc_post` (431) no backward; **fallback 433-439** |
| QKVZ projections | `build_qkvz` 671-702 | MUL_MAT VIEW_3D/2D RESHAPE | OK |
| beta/alpha/gate | 1124-1155 | MUL_MAT VIEW SIGMOID ADD SOFTPLUS MUL RESHAPE | OK (gate reshaped to ne[0]==1 -> non-KDA) |
| causal conv | `build_conv_state_at` 1393-1445, 1166-1177 | RESHAPE VIEW SCALE GET_ROWS CPY CONCAT TRANSPOSE CONT SSM_CONV SILU | OK (SSM_CONV rule written this session) |
| GDN recur | `build_delta_net_chunking` (delta-net-base.cpp:16) | SCALE PERMUTE PAD MUL RESHAPE TRANSPOSE CONT CUMSUM SUB TRI EXP REPEAT MUL_MAT NEG VIEW FILL DIAG ADD SOLVE_TRI SET | OK in `tools/train/gdn_train.hpp` (FILL/DIAG replaced by an identity **input**) |
| gated norm + out proj | 1225-1235 | VIEW RMS_NORM SIGMOID MUL RESHAPE MUL_MAT | OK |
| MoE | `build_layer_ffn` 1240 -> `build_moe_ffn` (llama-graph.cpp:2194); **differentiable replacement DONE: `tools/train/moe_train.hpp`** | MUL_MAT SOFT_MAX ARGSORT GET_ROWS RESHAPE SUM_ROWS CLAMP DIV SCALE **MUL_MAT_ID** VIEW GLU MUL ADD CONT | **MUL_MAT_ID has no backward rule and no decomposed fallback** -> rebuild the expert loop as the Hy4 trainer does (per used expert get_rows -> mul_mat -> swiglu -> mul_mat -> routing weight -> get_rows_back) |
| shared expert | 1261-1282 | MUL_MAT GLU/SILU SIGMOID MUL ADD | OK |
| PLE (per-layer embeddings) | `build_inp_ple` 1447, `build_ple` 1467 | GET_ROWS RESHAPE MUL_MAT RMS_NORM MUL SUM_ROWS SCALE CLAMP UNARY REPEAT VIEW TRANSPOSE CONT CPY CONCAT ADD + conv | OK |
| full attention (dense path) | `build_layer_attn` 1014 | MUL_MAT VIEW CONT RMS_NORM ROPE SOFT_MAX/FLASH_ATTN_EXT SIGMOID MUL PERMUTE | OK - `cparams.training` makes the graph use k_cur/v_cur directly (llama-graph.cpp:3258), so no SET_ROWS KV writes |
| QSA (lightning indexer) | `build_qsa_sel` 811, `build_attn_qsa` 955 | **LIGHTNING_INDEXER**, TOP_K, TOP_K_MASK (fused) / SET_ROWS, FILL, FLASH_ATTN_EXT | see below |

## The QSA question (12 of 48 layers)

- `ggml_lightning_indexer` (`qwen4exp.cpp:878`) is fused with **no decomposed fallback in this file**, and is called
  unconditionally. Its weights are a constant `fill(1/sqrt(idx_dim))` (877), so the only thing that depends on it is
  the block **selection**: `ggml_top_k` -> mask -> attention. TOP_K and ARGSORT are already marked piecewise-constant
  (zero-grad) in `ggml_compute_backward`; if nothing downstream of the indexer needs a gradient, the fused op never has
  to be differentiated. **This must be tested, not assumed** - build the backward for one QSA layer and see.
- `build_attn_qsa` has no `cparams.training` bypass for its KV writes (unlike dense attention), and its mask path uses
  SET_ROWS (no backward rule). Either add such a bypass (use k_cur/v_cur + a plain mask in training) or write the
  SET_ROWS rule for the mask use.

## QFN geometry and the GDN head expansion (verified against the pack)

`qwen4exp.ssm.state_size = 128` (head dim, k and v), `ssm.group_count = 16` (k heads), `ssm.time_step_rank = 48`
(v heads), `ssm.inner_size = 6144 = 48 x 128`, `ssm.conv_kernel = 4`; attention 24 heads / 2 kv, indexer 4 heads x
128 keys, indexer top_k 2048, compress ratio 4 every 4th layer, 512 experts top-10, latent/ffn 640, PLE ngram 3 with
16 heads.

The GDN graph (`tools/train/gdn_train.hpp`) requires q/k already at the v-head count. `qwen4exp.cpp:1212-1216` does
`ggml_repeat_4d(q_conv, head_dim, num_v_heads, T, S)` when the counts differ - ggml's repeat TILES, so **v head j uses
k head j % 16** (verified numerically: repeat(3->9) = 10 20 30 10 20 30 10 20 30). Any trainer port must use the same
mapping or gradients will be assigned to the wrong k heads.

Gate at production geometry (S=128, H=48, T=64, inputs scaled 0.125): forward vs the numpy reference 1.9e-9 abs
(6.8e-7 rel), FD gradients dq 3.9e-8, dk 1.6e-7, dv 2.8e-8, dg 7.2e-8, db 5.9e-8 against |g| ~2-8e-3, i.e. 4e-6..1.3e-5
relative - FD noise.

## Two possible routes

1. **Our own per-layer trainer** (Hy4 style, what `hy4_train.cpp` does): full control, expert loop instead of
   MUL_MAT_ID, no KV-cache or selection ops in the graph at all (k/v come from the layer input). Already have: GDN,
   SSM_CONV, PAD/TRI/CUMSUM/SOLVE_TRI, the adapter/GGUF plumbing. Remaining: HC fallbacks (exists), PLE, MoE loop,
   dense attention block, and a decision on how to score attention in the QSA layers (full attention over the sequence
   at training lengths is the natural choice - the indexer exists to *save* compute on long contexts).
2. **llama.cpp's built-in training** (`llama_opt_init`/`llama_opt_epoch`, `examples/training/finetune.cpp`): the
   private tree already has `cparams.training` and a dense-attention bypass, so this may be closer than expected -
   but it needs the whole graph differentiable, i.e. MUL_MAT_ID, SET_ROWS and the QSA path solved, plus the fused
   ops disabled (`fused_gdn_ch`/`fused_gdn_ar` off, HC non-fused, TOP_K_MASK decomposed).

Route 1 is the one the Hy4 work already validates; route 2 is worth a time-boxed probe because it would give us
upstream-maintained optimizers and a public-path training story.
