# data/ds41: small files for DeepSeek-V4.1-Flash-1017

These files are used by the Windows RTX 5090 setup in [docs/WINDOWS_RTX5090.md](../../docs/WINDOWS_RTX5090.md) and by its
three `tools/v41/serve/strata-ds41-1017-*-5090-win.json` configs.

| File | Bytes | sha256 |
| --- | ---: | --- |
| `rank-route26k.bin` | 122,880 | `4d670ce37419ecf7e2fc6e1c14817a25dcc0a5d86db032f64645b3e1df44da22` |
| `bias_vl-1017.gguf` | 64,000 | `c953619637331684e57445146234f766abd35d903d599e0bf0c339e6e3eceacb` |
| `deepseek-v4.1-flash-1017-atomic-lora-rv.f32` | 40,960 | `e6ad39d63adf2600c73d88b7f191d69f6100cc1ea99da302e7ae2b48c9e1c6ce` |

## rank-route26k.bin

The `--rank` file.
- **Content:** all 15,360 routed experts (40 layers x 384), hottest first, as int32 `(layer, expert)` pairs.
- **How it was made:** from the model's own routing on 26K tokens of Japanese, English and code text, using `STRATA_ROUTE_LOG`, `--no-enc-skip` and `tools/v41/engine/v41_rank_route_log.py`.
- **What it does:**
  - The engine seeds its expert cache in this order.
  - `--build-mirror` writes the coldest experts first.
  - The route bias prefers the experts that are already cached.
- **Which quants:** every quant here keeps the checkpoint's own router (GSQ's router values and `exp_probs_b` are bitwise the original's), so one file serves both GSQ IQ2_XS and UD-IQ2_XXS.

## bias_vl-1017.gguf

The checkpoint's 40 image-position router biases, `blk.N.exp_probs_b_vl.bias` (F32, 384 values each), for `tools/v41/vision/make_vl_gguf.py`.

- **Why it is needed:** ISTA-DASLab's GSQ-RCO GGUF leaves these tensors out, so the engine refuses `--vision` with it. Its 40 plain `exp_probs_b.bias` tensors are bitwise equal to the checkpoint's, so the checkpoint's `bias_vl` belongs to the same router.
- **Where the values come from:** they were read from Unsloth's UD-IQ2_XXS GGUF (after `v41_from_llamacpp.py`). They are byte-for-byte equal to the copy inside the GSQ `-vl` file this setup has run since 2026-10-26.
- **License:** the values are DeepSeek's (DeepSeek-V4.1-Flash-1017, MIT).

## deepseek-v4.1-flash-1017-atomic-lora-rv.f32

An `--ablate` pair (`r`, then `v = r`; 2 x 5120 float32, unit length) taken from AtomicChat's `DeepSeek-V4.1-Flash-1017-Uncensored-LoRA.gguf` (AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GGUF, `lora/`, MIT).

- **What the LoRA is:**
  - It has rank 1, with 122 sites: `attn_output_b`, `ffn_down_shexp`, `ffn_down_exps` (all 384 experts) and the Engram value rows 20480-25599. It does not touch the embeddings.
  - Every `lora_b` is this same unit vector `r` (minimum cosine 1.0000000).
  - Every `lora_a` is `-(r^T W)` on the original weights.
- **What follows:**
  - Its model card's `--lora-scaled 2.0` is `W - 2 r r^T W`, which is a reflection.
  - `--ablate <this file> --ablate-scale 2 --ablate-sites engram,attn,shexp,routed` applies the same map, exactly, to the quantized weights the engine runs.
- **Checks:**
  - `r` agrees with the direction AtomicChat's own GSQ build reflects its 82 dense writers along: |cos| 0.99995.
  - On Unsloth's UD-IQ2_XXS, the run-time edit is KL 0.115 (EAR 88.4 %) from the edited original. The edited original is Unsloth's UD-Q8_K_XL with the same map.
