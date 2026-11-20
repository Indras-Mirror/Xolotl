# Hy4-flash native port: plan and resume point

Started 20 Nov 2026. Branch `hy4-port` (worktree `~/AI/Xolotl-hy4`), cut from main 888d6e720.

## Where things are

| What | Where |
|---|---|
| FP8 parent (tencent/Hy4-flash-FP8 rev 003107fa24, 148 shards, 268 GiB) | `/media/mal/Team CX2/Xolotl/hy4-flash/Hy4-flash-FP8` (shards 24-148 symlinked from Crucible) |
| Reference llama.cpp (PR #32517, `Little0o0/hy4-flash` cf70454a) | `/media/Crucial1TB/llama.cpp-hy4`, worktree of `~/AI/llama.cpp-qwen4exp` |
| Converter patch for the public release | branch `hy4-release-convert` in that worktree (local only) |
| Q8_0 GGUF, split: shard 1 = 166.6 GB of Q8_0, shard 2 = Engram F8_E4M3, 63.4 GB | `/media/mal/NVME1TB/Xolotl/hy4-flash/Hy4-flash-Q8_0-0000{1,2}-of-00002.gguf` (`~/v41-work/hy4/convert-q8.sh`, unit `hy4-convert-q8`, log `~/v41-work/hy4/convert-q8.log`) |
| Tencent's GGUF headers (metadata reference) | `~/v41-work/hy4/angelslim/q4km-0000{1,2}-head.bin`, patches + READMEs alongside |
| Header parser | `~/v41-work/hy4/tools/gguf_head.py` |

## The converter, and why it needed a patch

PR #32517 was written against Tencent's internal checkpoint, so it fails on the public FP8 release (also reported on the PR).
The patch changes names only, never the maths, so our GGUF is the one Tencent's GGUFs were made with:

- the architecture is `HYV4FlashFor{CausalLM,ConditionalGeneration}`, not `HYV4Next*`
- the sparse layers are `deepseek_sparse_attention`, in `layer_types` and as the `rope_parameters` key
- there's no `engram_enable` key: Engram is on whenever `engram_layer_ids` is set
- there's no `engram_vocab_map.npz`: the token map is a `lookup_table` int32 buffer per Engram layer (120,818 entries, all four equal), and the compressed vocab is `max + 1 = 92620`
- the 5-layer `model.dfly_layers.*` draft head is dropped (Tencent's GGUF has a 1-layer MTP block from their internal checkpoint instead)
- the Engram tables ship as BF16 (660,006,416 rows x 96, 127 GB). Concatenating them inside the converter ran past 85 GB of RSS, so I stopped it. They're now written as a separate second shard by `tools/hy4/hy4_engram_gguf.py`, streamed in row chunks (`HY4_ENGRAM_SPLIT=1` keeps them out of shard 1, which is also Tencent's layout).
- **Engram encoding:** Tencent's table is F8_E4M3 with **each row scaled so its max |x| hits 448, computed in BF16** (`x / (rowmax/448)`). It's 100% byte-equal on the 19.2M-byte reference chunk; a plain cast matched about 0.001%. This is lossless for the model because the graph RMS-norms every looked-up row. The BF16 values are not E4M3-exact (about 91% change under a plain cast), so BF16 is the master.

Checked against Tencent's Q4_K_M header (range-fetched, no full download):

- **identical:** the Engram token map, the hash multipliers, the head offsets and the bucket sizes, plus all 1,380 tensor names and shapes (theirs also carry `blk.42`, the MTP block)
- **different, expected:** the special tokens are spelled `:opensource` where Tencent's internal build has `:6124c78e` (same ids), and the public chat template drops `reasoning_effort=low`

## Architecture against DS4.1 (what carries over and what is new)

| Piece | DS4.1 (`ds41_*`) | Hy4-flash | Port |
|---|---|---|---|
| Residual | mHC, hc=4, Sinkhorn comb | hc=4, **identity comb**, `post = 2*sigmoid(...)+eps` (`hc_magnitude 2`) | reuse hc_pre, simplify hc_post |
| Engram | 4 layers, hashed n-gram tables | 4 layers (0-based 1/11/21/31), 3-gram x 8 heads x 96, **per-row RMS norm x lookup_norm**, kv bias, **additive per-hc scale**, gate `sigmoid(sign*sqrt|dot|)`, fp32-epsilon norms | adapt ds41_engram (hash identical) |
| Attention | MQA 512, compressed KV (ratio 2/1), window 128 | **MLA** (q_lora 1024, kv_lora 512 + rope 64, absorbed: K 576 / V 512 then wv_b 512->256 per head), **sigmoid output gate**, **post-attn RMS norm**, sinks | new |
| Layer mix | per-layer ratio | 30 sliding layers (64 heads, window 512) + 12 DSA layers (32 heads) at 0,1,5,9,...,41 | new |
| Sparse select | indexer over compressed rows | **k-pool indexer**: 16 heads x 128, gated softmax pool of 4 raw tokens (ape), RMS-normed pooled key roped at its first member, top 511 pools + the current group = 2048 cells | new (reuse ds41 indexer kernels where shapes fit) |
| RoPE | — | **interleaved (adjacent pairs)**, theta 1e7 sparse / 1e4 sliding (transformers #50589 found split-half wrong) | check against the reference dumps |
| MoE | 256 x MXFP4, hash routing on early layers | **latent MoE**: router on 3072, `routed_down` 3072->1536, 512 experts top-16 in 1536 (ff 1536), `routed_up` 1536->3072; sigmoid + bias, norm, x3.998; swiglu clamp 10 on routed only; 1 shared expert at full width | adapt ds41_moe tiering |
| Norms | pre-norms | sandwich: attn_norm/post_attention_norm, ffn_norm/post_ffw_norm | trivial |
| Dense lead | — | layer 0 dense FFN 13824 | trivial |
| Draft | DSpark | `dfly` 5-layer DFlash-style head (mtp_fusion, hidden_correction, confidence head) | later; needs a vLLM/SGLang reference |
| Vision | — | 27-layer ViT, Tencent ships `mmproj-Hy4-flash-bf16.gguf` | later |

Sizes at Q8_0: experts 3.85 GB per layer x 41 = 158 GB; one token touches 16 x 7.08M x 41 = 4.6B expert params, about 4.9 GB. Dense (no experts, no Engram) is about 7 GB, which fits on the 4090. Engram is 63 GB of F8 but only 4 x 16 rows x 96 B per token, so it can be read lazily from NVMe. With Q4_K experts (~84 GB), the whole model fits in the 110 GB of RAM.

## Phases (each gated on the real model)

0. **Reference**: build the PR; convert Q8_0; run llama.cpp on short34 + a few chats (coherence, tok/s). Dump per-layer tensors (`llama-eval-callback`) on 2-3 prompts as the golden data.
1. **Geometry + loader**: `hy4_geometry.hpp` reads the `hy_v4_flash.*` keys and resolves the full tensor map from the GGUF; unit test against the Q8 file.
2. **Dense graph** `hy4_dense.cpp`: HC, Engram, MLA sliding + DSA k-pool layers, gate, sandwich norms, sinks. Per-layer compare against the golden dumps.
3. **Latent MoE** on the ds41 expert tier (RAM + NVMe tiers, routed_down/up around the expert compute). First-token logits within KL tolerance of the reference, then greedy token match.
4. **Serve**: `hy4-engine.sh`, chat template, reasoning_effort, Gauntlet v3-small + short34 against DS4.1.
5. **Speed**: dfly drafter (DSpark path), expert hot-set, Q4_K experts for a RAM-only build.

## Open items

- A llama.cpp speed baseline at Q8 experts won't fit in RAM (158 GB > 110 GB). A Q4_K_M requant needs ~157 GiB more disk (NVMe has ~69 GB left after the Q8). Either download Tencent's Q4_K_M later in place of something else, or test speed only in the native engine.
- Read the licence (`LICENSE`, tencent-hy-license) before publishing any quant.
