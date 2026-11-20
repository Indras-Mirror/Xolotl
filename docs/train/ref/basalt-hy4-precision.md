# Hy4-flash precision

Where Hy4-flash rounds and how its output compares. Method: [quality](quality.md). Guide: [Hy4-flash](../models/hy4-flash.md).

## Summary

- **Against BF16:** the published pack scores mean KLD 0.201 and top-1 86.09% ([below](#against-the-bf16-reference)).
- **Placement decides rounding.** A GPU expert and a CPU expert round their activations differently, so a different tier size
  can change the text. With the tiers pinned, a fixed prompt repeats exactly.
- **Drafting does not change output.** With the tiers pinned, drafted decoding gives plain decoding's tokens.
- **The output head is Q8_0 on the card.** The pack stores it in BF16 and the engine rounds it to Q8_0 as it loads. That is
  lossy against BF16, but within placement noise ([below](#the-output-head)).

## Weights

The published pack, from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)). Every rounding is
deterministic and runs once in the converter; `convert.py --verify` repeats it and compares the bytes.

| Tensors | Stored as | From the BF16 checkpoint |
| --- | --- | --- |
| Routed experts (layers 1 to 41) | NVFP4 (E2M1, a UE4M3 scale per 16 values, an FP32 scale per matrix) | tensor scale amax / (6 x 448); per 16 values the UE4M3 scale with the least squared error among half to four times amax / 6 (the 27B pack's rounding) |
| Attention, latent, shared-expert, dense-FFN, Engram and indexer projections; token embedding | Q8_0 | ggml's reference rounding, byte for byte AngelSlim's Q8_0 tensors |
| Output head | BF16 | as stored; the engine rounds it to Q8_0 as it loads (ggml's reference rounding) |
| Norms, hyper-connections, router, biases | F32 | as stored |
| Engram table | E4M3 codes, no scale | each BF16 value to the nearest code, ties to even; few of the values are E4M3 already (9% exact, 2.7% RMS error) |
| Vision encoder and projector | BF16 (convolutions and position table F32) | as stored |
| DFly2 drafter | BF16 in the file; projections rounded to Q8_0 as they load (`kv_b` stays BF16), vectors widened to FP32; the drafts read the target's Q8_0 head | drafting only: cannot change output |

The routed experts' weight RMS error is 0.081 in NVFP4 (the Q4_K_M test pack: 0.071 gate and up, 0.077 Q4_K down, 0.018 Q6_K
down on 19 layers).

## Where each stage rounds

| Stage | Format | Arithmetic |
| --- | --- | --- |
| Routed experts | NVFP4 | q8_1 activations on the GPU, Q8_0 on the CPU; prompt chunks MMQ (int8 mma) with each matrix's FP32 scale applied after |
| Q8_0 projections | Q8_0 | q8_1 activations (decode); MMQ (prompt) |
| Output head | Q8_0 | q8_1 activations, fp32 logits |
| Engram rows | E4M3 | decoded to fp32; the per-head norms and everything after in fp32 |
| Norms, router, hyper-connections | F32 | fp32 |
| Sparse layers' rows | fp16 (`--kv fp8`: FP8 row, [below](#attention-caches)) | decode: fp32 scores, softmax and accumulation; prompt: fp16 tensor cores, fp32 accumulation |
| Window layers' rings | fp16 | as the sparse layers' rows |
| Indexer pooled keys | fp16 | a prompt scores them on tensor cores with the query in two fp16 halves |

A decode row's arithmetic does not depend on how many rows share its window.

## Attention caches

- **`--kv fp16` (default):** the rows as fp16, as Basalt's other models keep theirs and as llama.cpp's BF16 reference run kept
  its cache. The values fit fp16 with room: the latent is at most sqrt(512) x 0.652 = 14.8 and the rope values at most 89 over
  all 42 layers' weights, and fp16 has 3 more mantissa bits than BF16.
- **`--kv fp8`:** vLLM's `fp8_ds_mla` row of 656 bytes for the sparse layers. Bytes 0 to 511 are E4M3 codes of the 512 latent
  values, rounded to BF16 first; bytes 512 to 527 four fp32 scales, one per 128 values, each
  `2^ceil(log2(max(amax / 448, 1e-4)))`; bytes 528 to 655 the 64 roped values in BF16. A code is `E4M3(v / scale)`, to nearest
  even, saturating. A code times its power-of-two scale is exact in BF16 and in fp16, so every reader decodes the row exactly:
  decode attention in fp32, prompt attention's tensor cores in fp16, the drafter's gather into its fp16 scratch. The window
  rings and the pooled keys stay fp16.
- **`--kv fp8` against BF16:** mean KLD 0.2046, top-1 86.04%, PPL 17.4371 over the 100 chunks (`--expert-cache 4800`, T 1,
  25,500 rows). The default's fp16 rows score mean KLD 0.2003, top-1 85.87% and PPL 17.4934 on the same tier size, so the
  format costs 0.0043 KLD and a heavier tail (p99 3.21 against 3.01, max 23.3 against 20.8) while top-1 and PPL do not move
  beyond their spread. It halves the cache: 8.6 KB a position against 14.6, and 8.4 GiB against 14.3 at 256K. The default stays
  fp16, the format the BF16 reference kept; `--kv fp8` is the memory choice. The summary's 0.201 mean KLD for the pack was
  measured before this comparison and without a pinned tier; the fp16 rows here land within 0.001 of it.
- Needles in a 253,410-token prompt are found with both formats.
- Pin the tier when comparing formats: this model declares no `quality_flags` for the gate, so pass `--expert-cache N` (and the
  same count) to both runs. Without it the freer VRAM of fp8 moves experts onto the card and the two runs are not comparable: an
  unpinned pair reads 0.2085 against 0.2030, the opposite sign to the pinned one.

## Experts and placement

- **Tiers round differently.** An expert on the GPU and the same expert on the CPU quantize their activations differently,
  so which experts are resident can change a token. Placement follows routing, and the main card's copies refill after a long
  prompt or an image.
- **Pinned for checks.** Identity checks pin the tiers (`--diag-tier-frozen`).
- **Clamped SwiGLU** on the routed experts (limit 10), not on the shared expert.

## Prompt path

Prompt chunks run the dense and expert products as MMQ (q8_1 rows, int8 mma, fp32 accumulation) and the attention on fp16
tensor cores, so their rows round differently from decoded ones. The chunk size moves the rounding too: a 4,500-row prompt read
in one chunk against 1,024-row chunks gives last-row KL 0.0032.

## Drafting

`--spec` verifies up to 6 rows at once (the default). Each verify row equals plain decode's, including the indexer's pooled keys,
which are built from the same raw rows whichever window completes a group. The drafter reads the target's caches without writing
them and keeps its block's rows in scratch, so a rejected block leaves nothing behind. With the tiers pinned, drafted output is
plain decoding's token for token. The drafter's own rounding (Q8_0 projections, FP8 block rows with `--kv fp8`) changes only
which drafts are accepted.

## Against the BF16 reference

Reference: [quality](quality.md#the-bf16-reference), `hy4flash-bf16-base.kld` (base PPL 18.572). 100 chunks, 25,500 rows:

| Pack | KLD mean | Top-1 |
|---|---:|---:|
| NVFP4 (the published pack; the 0.5.0 engine, BF16 head) | 0.201 | 86.09% |

### The output head

The head reads 742 MB as BF16 and 394 MB as Q8_0; the room goes to the expert tier (on an RTX 5090, 137 instead of 133 copies
a layer). Measured through the gate above (T 1, tiers frozen), against the same build with the BF16 head:

| Head | Placement | Chunks | PPL | KLD mean | Top-1 |
|---|---|---:|---:|---:|---:|
| BF16 | pinned, 130 copies a layer | 50 | 14.5744 | 0.21232 | 85.99% |
| Q8_0 | pinned, 130 copies a layer | 50 | 14.5743 | 0.21233 | 85.96% |
| BF16 | auto (133) | 100 | 17.5916 | 0.20600 | 86.03% |
| Q8_0 | auto (137) | 100 | 17.4626 | 0.20485 | 86.14% |

At one placement the head moves mean KLD by +0.00001 and top-1 by 0.03 points (the largest change of one row's NLL 0.077). Placement
alone moves more: the same 50 chunks with the BF16 head score 0.21549 at 133 copies against 0.21232 at 130.

## Lossy routing flags

What the flags do: [quality](quality.md#lossy-routing-flags). Here k is 16, so `--route-tail-skip` takes R 1 to 15 (bare 15), and
`--route-prior` adds LAMBDA to the sigmoid score plus its bias, which the top 16 is taken from (bare 0.01).

Two cards (400 W + 180 W), the default placement, the 0.5.0 engine's BF16 head. Speed: `basalt-bench -t conv,struct,prose
--seed 1`, one cold run each (off twice, before and after the rest); ms a window is the step time. CPU and bus picks: the run's `TIER final` line. Quality: the gate at 50
chunks against the BF16 base (12,750 rows; this subset's off row reads 0.2069, the 100-chunk one 0.201).

| Flags | CPU / bus picks | conv tok/s, ms/w | struct tok/s, ms/w | prose tok/s, ms/w | KLD | Same top-1 |
|---|---:|---:|---:|---:|---:|---:|
| off | 6.9% / 2.7% | 86.2, 24.79 | 173.8, 33.48 | 85.4, 20.22 | 0.2069 | 86.10% |
| `--route-tail-skip 15` | 6.0% / 2.2% | 97.8, 23.77 | 180.7, 32.04 | 86.9, 19.36 | 0.2056 | 86.07% |
| `--route-tail-skip 12` | 4.2% / 1.2% | 110.8, 20.50 | 208.4, 28.37 | 102.8, 17.40 | 0.2373 | 85.57% |
| `--route-tail-skip 8` | 2.2% / 0.4% | 126.1, 17.64 | 228.6, 25.86 | 111.2, 15.93 | 0.2826 | 83.94% |
| `--route-prior 0.01` | 4.4% / 1.4% | 108.1, 21.84 | 200.5, 29.27 | 98.9, 18.33 | 0.2212 | 85.70% |
| `--route-prior 0.03` | 2.6% / 0.7% | 116.5, 19.43 | 219.1, 26.76 | 108.4, 16.60 | 0.2465 | 85.33% |
| `--route-prior 0.1` | 1.0% / 0.2% | 129.9, 18.65 | 224.2, 26.42 | 115.6, 15.85 | 0.2941 | 83.82% |
| both (0.1, 12) | 1.0% / 0.2% | 130.6, 18.50 | 224.6, 26.37 | 110.0, 16.08 | 0.2965 | 83.83% |

Decode here waits on the CPU's experts and the bus copies, so dropping them pays directly: `--route-tail-skip 15` (12% of the misses
dropped) costs nothing the gate sees for 4% of conv's window, 12 (44% dropped) cuts it by 17% for +0.030 KLD, 8 (76%) by 29% for
+0.076. `--route-prior 0.01` cuts 12% for +0.014; 0.1 empties the CPU's share but costs +0.087 and 2.3 points of top-1. With the prior
on, few misses are left for the tail skip. The gate's PPL falls as the error grows (14.85 off, 13.77 at R 8), so it is no measure
of closeness here; KLD and top-1 are.

With the Q8_0 head the same steps hold: off 24.0 ms a conv window (two runs, 22.5 and 25.6) and KLD 0.2060; R 15 23.1 ms and
0.2091; R 12 20.6 ms and 0.2402; LAMBDA 0.01 21.5 ms and 0.2295. A KLD step under about 0.005 is placement noise.
