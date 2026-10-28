# Qwen4Exp (Qwen3.8-Flash-Next) — what the architecture demands of a quantiser

Sources of truth, in order: the model's own `config.json`, the tensor inventory of the real
GGUF, and arXiv 2608.30320 ("On the Design of Qwen3.8-Next Architecture"). Everything below is
either read from those or measured on this model.

This exists so a Qwen4-architecture release can be quantised on day one instead of rediscovering
the same constraints. The V4/DeepSeek pipeline is **not** transferable as-is: its wins came from
per-expert tiering of separately-stored experts, which this format cannot express.

## The model

| property | value |
|---|---|
| backbone params | 125B (120.8B of that is routed experts) |
| n-gram tables | +51.2B params, held **off the accelerator** |
| layers | 48: 36 Gated DeltaNet (linear attention) + 12 full attention (every 4th) |
| experts | 512 routed, **top-10** per token, + 1 shared expert |
| expert width | `moe_intermediate_size` 640, `hidden_size` 2560 |
| residual | **Gated Residual**: 4 branches (`hc_count`), low-rank 320, elementwise gate |
| long context | QSA — sparse attention with a compressed lightweight indexer |
| attention | 24 heads x 256, 2 KV heads (GQA), partial RoPE (factor 0.25) |

## Size reality (measured)

The 51.2B-param n-gram table is 26.8 GiB at IQ4_NL — **42% of the output file**, and it is not
where any compute happens. The routed experts are 122.4 GiB at Q8.

Everything else — attention, GDN, gated residual, indexer, shared expert, embeddings — totals
about **5.1 GiB**. That number is the whole argument below.

## Tier table, measured on this model's real weights

ggml's own quantiser (the one `llama-quantize` uses), verified identical across two ggml builds
and, for Q8_0, against gguf-py.

| type | bpw | gate/up (row 2560) | down (row 640) |
|---|---|---|---|
| q2_0 | 2.25 | 72.2% | 71.9% |
| iq2_xxs | 2.06 | 34.5% | **not representable** |
| iq2_xs | 2.31 | 30.1% | not representable |
| iq2_s | 2.56 | 25.1% | not representable |
| iq3_xxs | 3.06 | 18.2% | not representable |
| iq3_s | 3.44 | 13.8% | not representable |
| iq4_nl | 4.50 | 7.5% | 7.5% |
| q6_k | 6.56 | 1.7% | not representable |
| q8_0 | 8.50 | 0% | 0% |
| nvfp4 | 4.50 | 11.5% | 11.5% |
| mxfp4 | 4.25 | 11.9% | 11.8% |

Two hard consequences:

1. **q2_0 is unusable.** 72% relative weight error at 2.25 bpw, while iq2_xs is 30% at 2.31.
   It is strictly dominated and has been removed from the gate/up ladder.
2. **`n_ff = 640` rules out every 256-block type on the down projection** (IQ2_*, IQ3_*, Q3_K,
   Q5_K, Q6_K). The down ladder is therefore limited to types whose block size divides 640:
   **ptq1_0 (128), q2_0 (64), iq4_nl (32), nvfp4 (32), mxfp4 (32), q8_0 (32)**.
   This is a FORMAT limit — do not write "ternary cannot go on down": it can, 640/128 = 5
   exactly, and our encoder emits it at exactly 1.75 bpw.
3. **PTQ1_0 on down is excluded for OUTPUT PRECISION, not for erasing the ablation.**
   An earlier reading of this document said IQ4_NL "keeps only ~15% of the edit". That was
   WRONG — it used a whole-tensor norm ratio, which counts quantisation noise spread across all
   2560 output columns. The ablation acts on ONE direction only (see below), and along that
   direction IQ4_NL preserves 96-101% of the edit. Corrected measurements follow.

### The ablation direction survives fine; it is total precision that suffers

The adapter's down edit has a **shared output direction**: for each layer `B` is `[2560,1]` and
identical across all 512 experts; only the input vector `A` (`[1,640]`) differs per expert. Every
routed expert therefore writes its edit along the same `u`, and the ablation works by moving the
residual along `u` — not by moving it in all 2560 directions.

So the metric that decides edit survival is the component along `u`:

    keep = <(Y_quantised - Y_pristine), u> / <(Y_edited - Y_pristine), u>

Measured with `measure_direction_survival.py` (L2/L16/L32, real routed tokens, down tier varied,
gate/up held at the edited values):

| down tier | keep (L2 / L16 / L32) | reading |
|---|---|---|
| q8_0 | 1.000 / 1.000 / 1.000 | exact |
| iq4_nl | 0.962 / 0.983 / 1.008 | **edit direction preserved within 4%** |
| nvfp4 | 1.009 / 1.019 / 0.945 | comparable |
| mxfp4 | 1.007 / 1.001 / 0.936 | comparable |
| ptq1 | 1.161 / 1.113 / 1.060 | direction kept, but ~10-16% of the u-component is noise |
| q2_0 | 1.013 / 1.461 / 1.797 | u-component dominated by noise; unstable |

Values slightly above 1 mean noise is projecting onto `u`; the direction is not *lost*, it is
*contaminated*. q2_0 at 1.8 is unreliable. IQ4_NL sits within ±4% of unity.

Independently, the whole-tensor error on down at iq4_nl is only **0.29%** output error
(`edn_iq4_nl` in the real screen), so iq4_nl down is also cheap in general-quality terms.

**Conclusion: IQ4_NL on the edited down projection preserves the abliteration and costs
21.09 GiB instead of 39.84 GiB.** The +18.75 GiB for Q8_0 down is NOT required to keep the
ablation. Use IQ4_NL; keep Q8_0 on the ladder only as an optional top rung.

Do not use PTQ1_0/q2_0 on down: not because they erase the edit direction, but because their
noise contaminates it and their general precision is poor.

## The allocation this implies

Compression belongs where redundancy actually is: the routed experts (top-10 of 512, so a given
expert sees ~2% of tokens). Every other family is dense — every token touches it — and is small.

| family | tier | why |
|---|---|---|
| routed gate/up | measured ladder, per layer | sparse; per-expert error is measured per expert |
| routed down | **iq4_nl** (q8_0 optional) | 640-block constraint; IQ4_NL preserves 96-101% of the ablation direction at 0.29% output error |
| shared expert | q8_0 | always active; nothing dilutes its error |
| GDN (`ssm_*`) | q8_0 | **recurrent state accumulates error over the 262k context** — V4 had no equivalent risk |
| gated residual (`hc_*`) | q8_0 | gates the residual stream every layer; 0.6 GiB total |
| indexer | q8_0 | picks context micro-blocks; wrong blocks = long-context failure |
| full attention (`attn_*`) | q8_0 | 12 of 48 layers, 2 GiB total |
| router (`ffn_gate_inp`) | f32 | top-10 of 512; a small error flips selection discontinuously. llama-quantize leaves it F32 |
| n-gram table | iq4_nl | 42% of the file; lookup, off-accelerator, so precision costs RAM not speed |

Keeping the **entire** backbone (everything except experts and the n-gram table) at q8_0 costs
about **+2.4 GiB** over IQ4_NL and makes it idempotent — the source is already Q8_0, so those
tensors are copied rather than requantised, i.e. **bit-identical to the parent**. For 2.4 GiB
there is no argument to touch them.

That is the V4 lesson translated: protect what is dense and sensitive, squash what is sparse and
redundant. On DeepSeek the redundancy was per-expert and the format allowed per-expert tiers.
Here the redundancy is still per-expert, but the format stores all 512 experts in one tensor with
one type — so the choice is per **layer**, and only the measurement inside a layer is per expert.

## What still has to be verified, not assumed

- The n-gram table at IQ4_NL is the biggest single byte block after the experts. Lowering it to
  IQ2_S/IQ3 would save 10-13 GiB but each row is used whole, so errors do not average out. Not
  measured yet; do not lower it on faith.
- The allocator's objective is held-out expert-output MSE. That is a **proxy** for quality; the
  real gate is top-1 agreement against the parent on held-out text
  (`strata --dump-logits` + `tools/qwen38/logits_gate.py`). The tooling exists and is tested; the
  runs and the pass marks are not. **The parent is the edited Q8 on disk** — the abliteration was
  applied in place, so there is no pristine whole file to compare against, and quantisation damage
  is the only thing the allocator can be wrong about anyway.
- Do **not** feed a strata dump to `tools/v41/logits_kl.py`. That tool assumes ds41's headerless
  format; strata writes a 2 x int32 `{n_vocab, n_rows}` header first
  (`src/program/generate.cpp:13542-13548`), so every row would be shifted 8 bytes and every number
  wrong-but-plausible. `logits_gate.py` parses the header and refuses a file whose size disagrees
  with it.
