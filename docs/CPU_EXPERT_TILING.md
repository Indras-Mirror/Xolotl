# CPU expert tier: row-interleaved ("tiled") kernels - to check per engine and build

Status: **open, to measure** (2026-10-31). The CPU expert tier is where DeepSeek-V4.1's decode speed comes from when
the experts don't all fit on the card, so every engine and build should be checked for this.

## The idea

A quantised matrix-vector product walks the weight matrix row by row. Today the CPU tier does that in two ways:

- **one token** (most decode calls; a DSpark window's expert groups hold ~1.4 tokens): ggml-cpu's per-token,
  per-row `vec_dot` (`native_gu_mt_min`, #152);
- **two or more tokens**: our multi-token kernels (`iq256/iq512_gu_rows`, `kq256`, `ptq1_0_rows_q8_0` rows3,
  `q2_avx2`) decode each weight block once for every token. Measured 2.0-2.4x ggml-cpu at three tokens.

Neither **interleaves rows**. A row-interleaved ("tiled") layout repacks the weights at load so that 4 (or 8) rows sit
side by side in each block, in small strips. One SIMD instruction then produces partial sums for several rows at once
in its lanes, so:

- the per-block scale work (convert to float, multiply by the block scale) is done once per 4 rows, not per row;
- there is no horizontal add at the end of each row;
- the activation block is loaded once for 4 rows.

ggml's Q4_0 repack (4x4 / 4x8 / 8x8), KleidiAI's dotprod/i8mm microkernels and ik_llama.cpp's `-rtr` all do this.

## What it bought on ARM (Xolotl-Mobile, 2026-10-31)

Galaxy A53 (2x Cortex-A78), LFM3-2B shapes, Q4_0 x Q8_0, hand-written dotprod kernels, both checked against a scalar
reference:

| tokens (columns) | row kernel | 4-row tiled | |
| ---: | ---: | ---: | --- |
| 1 | 118.7 ms/step | **79.2 ms/step** | 1.5x; tiled reads weights at ~96% of the phone's 10.7 GB/s |
| 2 | 138.8 | 87.7 | a 2nd token costs +11% |
| 4 | 230.9 | 128.8 | |
| 8 | 408.9 | 232.7 | |

The 1-token gain is the part that matters for the CPU expert tier, because that is the common decode case here.

## Why it may or may not carry over

- **x86 AVX2 / AVX-512**: the IQ codebook formats are documented as lookup-bound (~5 GB/s per core). Interleaving
  removes the per-row scale and reduction overhead but not the lookups, so the gain is smaller there, maybe none.
  K-quants, Q2_0, PTQ1_0 and MXFP4/NVFP4 have less lookup work and more to gain. AVX-512 VNNI (`vpdpbusd`) and
  AVX-VNNI give the same lane trick as ARM's `vdotq_laneq`.
- **ARM (Mac, any Linux ARM box)**: our own CPU expert kernels are x86-only, so the CPU tier there is entirely ggml's
  NEON per-token dots. The phone result applies directly.
- It only pays where the CPU tier is **compute-bound**. Where it already reads at the RAM ceiling, it won't help.

## How to check (per engine, per build)

1. With the engine's real expert formats, measure the 1-token CPU expert kernel per core in GB/s of weights
   (`native_expert_parity` times the multi-token kernels; `kq_cpu_bench` the K-quants), and compare it with that machine's per-core share of RAM
   read bandwidth. Well under it = compute-bound = a candidate.
2. For a candidate format, prototype a 4-row interleaved repack at load (the blob layout is ours: `expert_layout`)
   and an interleaved 1-token kernel, and check it bit-exact against ggml's per-token dot or against the scalar
   reference. Rounding changes the tokens, so gate it the way #152 does.
3. Measure end to end on the served model (decode tok/s and the pass split's tier time), Q8 dense per
   [the test rules](TEST_REQUESTS.md). Drop it if it's flat.

| engine / build | CPU-tier expert formats seen in its docs | what to check first |
| --- | --- | --- |
| DeepSeek-V4.1 (`ds41_generate`) | PTQ1_0, Q2_K / Q2_0, Q4_0 / Q6_K / Q8_0 classes | 1-token path for each class; PTQ1 rows3 already groups tokens, not rows |
| MiniMax-M3.1 (`m31_generate`) | IQ3_S, IQ3_XXS, IQ4_XS, Q8_0 (NVFP4 stays on the GPU) | IQ4_XS and Q8_0 (least lookup-bound) |
| GLM-5.3-Flash | Q4_K, Q5_K, Q2_K, IQ3_XXS | Q4_K / Q5_K 1-token (kq256 is opt-in and multi-token only) |
| Qwen3.8-Flash-Next / qwen4exp (`strata`) | PTQ1_0, Q2_0 (RCO) | rows3 at 1 token vs a row-interleaved variant |
| Qwen3.6 35B (`q36`) | UD-Q4_K_XL (Q4_K / Q5_K / Q6_K) | hybrid CPU experts at 1 token |
| Mac (Apple Silicon) | Q2_0, IQ3_XXS, IQ3_S, UD-Q4_K_XL | everything is ggml NEON: the phone kernel's direct case |
| Strix Halo | IQ4_XS, UD-Q4_K_XL, IQ3_* | unified memory, high bandwidth: likely compute-bound |
| Windows / AMD HIP / Intel Arc / older GPUs | same formats as above, same CPU code | AVX1 builds (older CPUs) have the most per-row overhead |
| Qwen3.8-27B dense (`strata-dense`) | none (CUDA only, no CPU tier) | n/a |

Source of the ARM numbers: `~/AI/Xolotl-Mobile` (bench/xm_bench.cpp, docs/PLAN.md).
