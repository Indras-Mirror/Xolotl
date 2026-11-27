# DFly2 drafter for DeepSeek-V4.1 — port spec (27 Nov 2026)

Source of truth: Tencent AngelSpec (`github.com/Tencent/AngelSpec`, e02ea36, Apache-2.0) — `models/draft/dfly2.py`
(= `DSparkDraftModel` with no Markov head) on `models/draft/dflash.py` (`DFlashDraftModel`, `MTPFusion`,
`HiddenStatesCorrection`), `models/draft/dsv41_mla.py`, `models/draft/dsv41_attentive_correction.py`,
`models/target/dsv41_target.py`, and the wrapper `models/dflash.py` (`_create_dflash_sparse_kv_indices`,
`_target_index_candidates`). Weights: `AngelSlim/DS-V4.1-Flash-DFly2-B6` (8.9 GB, FP8 32x32-block dense (ue8m0 scales) +
FP4 experts, 2392 tensors). Paper: arXiv 2611.34776. Also published: `AngelSlim/GLM-5.3-Flash-DFly2-B6` (BF16 20.7 GB),
`AngelSlim/Hy4-flash-DFly2-B6`.

## What it is
A block-parallel drafter: one forward over a block of 6 slots `[anchor, MASK x5]` (mask id 128799) proposes 5 tokens.
`dfly_config.causal_block = false` (slots see each other inside the block). No KV cache of its own, no indexer of
its own: draft layer i reads target layer `target_kv_layers[i]` (28, 32, 36) KV latents at the positions that
layer's indexer picked for the last accepted token (anchor − 1).

## Inputs per draft step (anchor at position a)
- `E(x)`: the target's token embedding (frozen, shared).
- `h_{a-1}`: the target's final hidden at a−1 **after the final norm** (what feeds its LM head / MTP layer).
- For each draft layer i (target layer L = 28/32/36): L's cached latents, and L's indexer top-512 positions for
  query a−1, plus the local window.
- Positions: draft slots at a .. a+5; RoPE per `dfly_config.rope` (theta 160000, YaRN factor 16, orig 65536,
  beta 32/1) = the target's compressed-layer RoPE (`compress_rope_theta`), so latents read from the target match.

## Forward
1. `x = E([anchor, mask*5])`  ([6, 5120]).
2. MTP fusion, mode `slot0`: `f = eh_proj(enorm(E(anchor)) :: hnorm(h_{a-1}))`; `x[0] += gate ⊙ f`
   (gate is a per-channel [5120] vector; other slots get nothing).
3. hc_mult = 4: `x` copied into 4 hc streams (equal copies); three DS4.1 blocks (`model.layers.{0,1,2}`), each:
   hc mixing (`hc_attn_*`, `hc_ffn_*`, sinkhorn 20 iters as the target's mHC), `attn_norm`, attention, `ffn_norm`,
   MoE (128 routed experts, gate weight + bias + bias_vl, sqrtsoftplus scoring like the target, top-k per drafter
   config; + shared expert w1/w2/w3; swiglu_limit 10). The last block's FFN mix collapses the streams.
4. Attention of block i (`DSV41DraftAttention`), same weights layout as our DSpark blocks:
   `q = rope(wq_b(q_norm(wq_a(x))))` [64 heads x 512], draft latents `rope(kv_norm(wkv(x)))` [512],
   keys/values = `[ctx_latent(L) | draft_latents]` (latent MQA: the 512-wide latent is key and value),
   each draft row reads the index list: target top-512 candidates (drop any inside the local window), the local
   window positions `a-W .. a-1` (W = `context_local_window`; ≤ the target's 128 sliding window), then the 6
   draft keys (non-causal). Per-head `attn_sink`, scaling 512^-0.5. Output: inverse-RoPE on the last 64 channels,
   grouped `wo_a` (8 groups, o_lora 1024) then `wo_b`.
5. `final_norm`.
6. Hidden correction: `h' = h + down(silu(gate(cat)) * up(cat))`, `cat = hidden_norm(h) :: embed_norm(E(prev))`
   (prev for slot j = the token at a+j-1; slot j predicts a+j. RESOLVED 27 Nov from `_served_draft_chain`
   (models/dflash.py): slots 0 and 1 take real tokens (x_{a-1}, then the anchor x_a); slots 2..5 take the previous
   slot's own proposal, so serving is ONE backbone pass then a 5-step sequential unroll of correction + LM head
   (slot 0's proposal is unused). The gated block read is causal (`back < 0` masked), so slot j's correction only
   needs the block keys of slots 0..j: the unroll is exact, no re-run of earlier slots), intermediate 5120. Plus the attentive correction (gated block mode, `block_kv_in = hidden_embed`,
   `source_layer = -1` → reads the LAST draft layer's context = target layer 36 latents + its index list):
   `q = rope(wq_b(q_norm(q_in(cat))))`, context read over `latent/indices` with `attn_sink`, plus a gated read over
   the block's own keys `rope(kv_norm(kv_in(cat)))` (causal inside the block, `block_rel_bias[h, back]`,
   `block_sink`, gate = `tanh(block_gate(cat) + block_gate_bias[:, slot])`), summed, `grouped_output` → added to delta.
7. Logits: the target's `output.weight` on `h'` (no Markov head). Slots 1..5 give the 5 draft tokens.

## Target side (our engine)
- Layers 28/32/36 compress 1:1 (`compress_ratios[L] == 1`): their compressed cache has one latent per token
  (compressor wkv + norm, roped with compress_rope_theta) = the ctx latent the drafter reads; the local window
  reads the raw sliding-window latents. AngelSpec reproduces vLLM's fp8_ds_mla record (channels 0-447 FP8 per
  64-group, 448-511 bf16 rope). Our --kv-fp4 cache is coarser: expect a small acceptance cost; test with Q8 KV too.
- Keep, per step, the top-512 indices of layers 28/32/36 for the last accepted token (the engine already computes
  them in the target forward; DS4.1's candidate blocks come from layer 20).
- `h_{a-1}` after the final norm: already available where DSpark takes its inputs.
- Verify: target forward over the 5 drafts (+ bonus), accept the longest matching prefix (greedy) — same
  verification path as DSpark.

## Plan
1. Offline acceptance check before any C++: dump from the engine, for N agent-style prompts, the inputs above per
   position + the target's greedy continuation; run AngelSpec's own modules (torch, CPU or the 4090) on them;
   tokens accepted per step vs our DSpark on the same positions.
2. If it beats DSpark: port into tools/v41/engine/ds41_dspark.{cpp,hpp} as a DFly2 mode (loader for the AngelSlim
   names; weights → GGUF sidecar, FP4 experts → MXFP4), validate step outputs against the torch run, then A/B
   decode tok/s on agent prompts, `--kv-fp4` vs Q8 KV.

## Offline acceptance dump (pinned 27 Nov from AngelSpec)
- Draft config `angelspec/config/dfly2_dsv41_flash_draft_config.json`: target hidden 5120, 40 layers (= our DS4.1),
  `target_layer_ids [19, 27, 31, 35]`, kv/index/attn-init layers 28/32/36, `target_index_topk 512`,
  `context_local_window 128`, `enable_confidence_head` true (adaptive draft length, like our DSpark).
- Aux id L = the ATTENTION INPUT of target layer L+1, i.e. layer L+1's `attn_norm` output (after its mHC pre-gate
  collapses the 4 streams), NOT vLLM's mean of the mHC copies (`apply_dsv41_attention_input_capture`; train_entry sets
  `dsv41_attention_input_capture` whenever the drafter reuses target kv/index layers). So dump `attn_norm` outputs of
  layers 20, 28, 32, 36.
- Our dump (`STRATA_DUMP_DFLY2=F`, ds41_generate): per decode step {tok, pos, HC, 5 x (HC x 5120 state, HC pre mix)}
  at layers 19/27/31/35/39 via Ds41Dense::read_state. Attention input of layer L+1 = attn_norm_{L+1}(hc_pre(x_L, pre_L))
  (model.py Block.forward: layer L's FFN hc_mixes makes the NEXT attention's pre mix, so the raw state alone is not
  enough). Final pre-norm hidden = hc_pre(x_39, pre_39) (Transformer.forward: no separate head collapse).
- Final hidden = the input of the final `norm`, pre-final-norm (`apply_dsv41_final_hidden_capture`;
  offline loader refuses a post-norm dump).
- The drafter recomputes the target's layer 28/32/36 latents and indexer picks from those inputs with a frozen bundle
  (those layers' attention + indexer weights, plus embed / head / norm): export it from our GGUF dequantised to bf16.
- Self-check (`dsv41_target.py`): RMS(hidden / attn_norm.weight) must be ~1 on >= 90 % of sampled rows, else the tap
  point is wrong. Run it on our dump before measuring anything.
