# Quantisation methods for DeepSeek-V4.1-Flash routed experts

**Research note, 2026-10-19.** Scope: a method study for the 15,360 routed experts of DeepSeek-V4.1-Flash (and its 1017
checkpoint), starting from the pipeline in `tools/v41/`. No code was changed and no GPU job was run; the only new
measurements below are CPU/read-only checks of checkpoint tensors and calibration statistics. Every number is tagged
`[our measurement]`, `[paper]`, `[community]`, or `[inferred]`; a claim without a tag is a summary of the cited source.

---

## 1. Executive summary

Ranked by expected quality gain per unit of work on one RTX 4090 + 90 GB RAM:

1. **GSQ refinement on top of GPTQ, at the existing three grids.** This is the only direction with a directly comparable,
   large published result in the *same* deployment format: on Qwen3-8B GGUF, GSQ's Q2_K scored 56.28 on their average
   versus Unsloth's Q2_K at 50.03, and Q3_K_M 61.61 versus 60.52 `[paper]`. Our pipeline already has the GPTQ
   initialiser and the fit/held-out split that GSQ expects; the missing piece is joint optimisation of grid assignment
   and group scale (`v41_quant.py`'s `REFINE` hook). We tried a simplified version and it regressed (section 3.2), so
   the experiment is explicitly a *fix and re-test*, not a re-run. One-line experiment: refine 3 layers x 384 experts
   for 300 steps from GPTQ, compare held-out output error at the same tier and bytes, gate = 5% lower error.
2. **Allocation: replace the per-layer normalised Lagrangian with a global, domain-weighted assignment.** Our
   "RCO-lite" allocation already beats uniform Q2_0 by 20.9% proxy error, but the objective treats every layer and
   every domain equally. GEMQ's global linear program plus router-aware allocation reports 7.93 vs 10.67 WikiText-2
   perplexity against uniform at 1.5 bits on Mixtral, and 6.03 vs 6.29 at 2.0 bits `[paper]`. A global LP over our
   existing stats costs seconds; per-domain costs would cost one calibration rerun. One-line experiment: re-solve the
   existing 1017 stats with a global LP and a 40/30/30 code/agentic/rest weighting, gate = 2% lower proxy at equal
   bytes with the head gate not worse on layers 5/20/32.
3. **Per-expert bias-corrected affine healing.** MoE aggregation amplifies output bias, and the correction is tiny:
   one D-vector plus one per-channel scale per expert (~0.1% of expert bytes). KBVQ-MoE uses exactly this
   ("bias-corrected output stabilization") and gets 3-bit Qwen1.5-MoE to 67.99 vs 68.07 FP16 `[paper]`. It does not
   change the quantiser, only adds a cheap post-pass and a per-expert epilogue in the kernel. One-line experiment: fit
   bias and scale on fit tokens for 3 layers, evaluate held-out output error, gate = 2% lower error for <=0.1% bytes.

**What not to prioritise.** Our own CPU measurements say FP4-grid scale snapping and power-of-two (E8M0) scales are
*worse* than the free fp16 scales we already use (section 3.1). A Hadamard rotation was already tried against the
engine's own ternary GPTQ and lost at equal bits (gate+up 1.36x worse, down a tie; section 3.3), so rotation is a
low-probability side bet. Vector/codebook quantisation is the strongest quality idea in the literature but needs new
CPU and CUDA decoders; it is a project, not an experiment (section 3.5).

**The largest structural finding** is that the experts are exactly MXFP4 (E2M1 values, one E8M0 power-of-two scale per
32 weights), confirmed in the checkpoint headers and by dequantising a real expert `[our measurement]`. The model was
trained with FP4 in the forward pass throughout post-training, so its weight distribution is already bounded and
block-scaled. That makes classic outlier rescue less relevant, but it also means a re-quantiser's optimal group scale
is generally *not* one of the native power-of-two scales: we measured that restricting to them costs 17-51% more
weight-space MSE (section 3.1).

---

## 2. Our current method, as-is

The pipeline is documented in `docs/ds4/V41_1017_QUANT_PLAN.md` and `tools/v41/hf-1017/README.md`. In one page:

**Formats.** Every routed expert's gate/up pair and down projection get their own tier: PTQ1_0 ternary
({-1,0,+1} x fp16 scale per 128, 1.75 bpw), Q2_0 ({-1,0,+1,+2} x fp16 scale per 64, 2.25 bpw), or Q3_K (3-bit with
6-bit sub-scales per 16 and an fp16 super-scale per 256, 3.4375 bpw). The 1017 build is 62.4% ternary, 24.7% Q2_0 and
13.0% Q3_K by weight share: 132.4 GiB of experts at 2.093 bpw. The dense half (attention, shared expert, head) is Q8_0
in the file and Q6_K at load; embedding/compressor/indexer stay F16; norms, router and mHC stay F32.

**Fit.** Calibration is 32 sequences x 8,192 tokens (262,144 tokens), roughly 40% code, 30% agentic/tool use, 30%
everything else, with about a quarter of each domain held out. A layer-streaming teacher forward runs DeepSeek's own
`model.py` from the HF shards, keeping the hidden state on disk between layers, and the MoE is processed expert-major
so each FP4 expert is dequantised once. For every expert, static scales are chosen first by a diag(H)-weighted search
over 11 clip ratios and both signs, rounded to the stored format, and GPTQ then chooses the integers with error
feedback. Gate+up are stacked on the input Hessian; down uses the Hessian of the clamped SwiGLU x routing weight. Act
order is on.

**Allocation and sequential pass.** Per-expert held-out output error (routing weight included) is normalised by the
layer's routed output energy, gate+up and down are allocated separately, edge layers count double, at most 50% of a
layer's expert projections may be ternary outside the gentler band and 85% inside layers 20-31 and 33-35, and experts
with fewer than 8 held-out tokens are floored at Q2_0. A Lagrangian bisection plus a greedy pass solves the budget. A
second layer-by-layer chain then quantises with `--propagate quant`, so later layers are fitted on the outputs the
quantised model will actually see.

**Measured results.** GPTQ beats RTN in every group: error ratios 0.58-0.66 ternary, 0.70-0.73 Q2_0 and 0.74 Q3_K;
spearman(RTN error, GPTQ error) is 0.992-0.996 over 15,096 experts, and an allocation picked from RTN errors scaled by
the median ratio has -0.77% true GPTQ cost and 88.6% identical tier choices `[our measurement]`. At equal bytes, less
ternary is worse: ternary funds Q3_K for sensitive experts. More budget helps: 132.4 GiB gives 1.766 proxy, 140 GiB
1.506, 144 GiB 1.408. The 1017 build's head gate is 6.934 ppl / 0.670 top-1 against a 6.572 / 0.687 edited teacher
(+5.5%); per domain, code is +9.1% (worst), chat +6.2%, agent -5.4% `[our measurement]`. The short agentic Gauntlet scored 3/5 versus 4/5
for the older V4.1 v2 build. GPTQ screening costs about 1.48 s per expert at three tiers (189 s per 128 experts,
extrapolated to 10.7 min/layer); the RTN screen costs about 0.24 s per expert (2.8 min/layer); the full sequential
quantise pass is 3.8 min/layer, about 2.5 h for 40 layers `[our measurement]`.

**Calibration coverage is not the bottleneck.** On the 1017-v3 calibration stats, the median expert has 2,120 fit and
494 held-out routed tokens; only 276 of 15,360 experts (1.8%) have fewer than 8 held-out tokens, and the 1,572,864
routed assignments per layer are spread over all 384 experts `[our measurement]`. MoEQuant's expert-balanced sampling
exists to fix a problem we already mostly do not have `[paper]`; the remaining allocation question is not coverage but
the objective and its granularity.

**Engine economics.** Decode is 37.5 ms/token plain (23-26 tok/s served with DSpark). Of that, Q8_0 dense mat-vecs are
8.9 ms at ~850 GB/s, expert misses are ~6.0 ms, the CPU pool is ~1.1-1.4 ms per expert, and the on-device expert
kernels are ~3.7 ms. The ternary kernel is ALU-bound at ~136 GB/s, not bandwidth-bound `[our measurement]`
(`docs/ds4/V41_START_HERE.md`). Any proposal that adds bytes per expert is competing with that 37.5 ms budget, and any
proposal that adds arithmetic to the ternary path is competing with an already ALU-bound kernel.

---

## 3. The eight directions

### 3.1 FP4-native quantisation

**What it is.** Quantise from the values the model actually stores rather than treating the dequantised weights as a
continuous tensor: choose ternary or 2-bit levels as subsets or projections of the 16 E2M1 levels, and tie the group
scale to the native E8M0 block scale.

**Evidence.** DeepSeek's V4 report says FP4 quantisation-aware training is applied to MoE expert weights and to the
indexer QK path: FP32 master weights are quantised to MXFP4 and dequantised back to FP8 for the forward pass, with a
straight-through estimator through the quantiser `[paper]` (<https://arxiv.org/abs/2606.19348>). The report argues the
FP4-to-FP8 dequantisation is lossless because E4M3 has more exponent range, provided the ratio of FP4 sub-block scales
inside an FP8 block stays below a threshold. Unsloth confirms the released checkpoint stores routed experts natively
in MXFP4: all 1,328 tensors are bit-identical to the official weights, with KL ~0 and 100% top-token agreement `[community]`
(<https://unsloth.ai/docs/models/deepseek-v4>). NVIDIA's format description gives the exact grid:
E2M1 has fifteen distinct magnitudes {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}, and MXFP4 gives every 32 values an
E8M0 power-of-two scale `[paper/community]`
(<https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/>). The MXFP4 error-reduction work shows
that MXFP4's coarse power-of-two block scale is the main source of its gap to NVFP4 (about 10% end-to-end), which
overflow-aware scaling and macro-block scaling close to below 1% with 6.2% GEMM overhead `[paper]`
(<https://arxiv.org/abs/2603.08713>).

**What we verified on this model.** The 1017 expert tensors are packed int8 with shape `[2304, 2560]` for a logical
`[2304, 5120]` weight, plus an F8_E8M0 scale of shape `[2304, 160]`: one scale per 32 weights, exact E2M1 codes
`[our measurement]`, one expert read from `model-00008-of-00048.safetensors`. So the "FP4-native" premise is true,
not an assumption.

**What we measured that argues against the obvious use of it.** Using layer 5 expert 0's gate matrix (11.8M weights),
we quantised every 32-weight block in weight space under three constraints:

| Grid | Ternary MSE (block units) | Q2 MSE (block units) |
| --- | ---: | ---: |
| Best scale restricted to E2M1 magnitudes {0.5,1,1.5,2,3,4,6} | 0.9716 | 0.7342 |
| Free continuous scale (our current search family) | **0.9223** | **0.6965** |

`[our measurement]`, weight-space MSE, no diag(H) weighting and no GPTQ. Restricting the scale to the native E2M1
magnitudes is 5.3-5.4% *worse* than a free scale. The reason is simple: the target values are E2M1 x 2^e, but the
optimal scale for a 3- or 4-level re-quantiser is not itself an E2M1 value; it sits between levels and balances the
errors.

We then measured group-size and power-of-two scaling:

| Scale type | Format / group | G=32 | G=64 | G=128 | G=256 |
| --- | --- | ---: | ---: | ---: | ---: |
| Free fp16 scale | ternary | 9.104e-5 | 9.431e-5 | 9.582e-5 | 9.648e-5 |
| Free fp16 scale | Q2 | 3.617e-5 | 3.939e-5 | 4.118e-5 | 4.216e-5 |
| E8M0 (power of two) | ternary | 1.070e-4 | 1.094e-4 | 1.102e-4 | 1.103e-4 |
| E8M0 (power of two) | Q2 | 5.470e-5 | 5.647e-5 | 5.713e-5 | 5.725e-5 |

`[our measurement]`, same expert, weight-space MSE per weight. Two conclusions. Finer groups help: for Q2, per-32 is
8.2% better than per-64 with free scales, and for ternary per-64 is 1.6% better than per-128. But forcing E8M0 scales
costs far more than that gain: Q2 per-32 with E8M0 (5.47e-5) is 39% worse than Q2 per-64 with a free fp16 scale
(3.94e-5). A per-32 E8M0 format therefore does *not* buy quality at equal bytes. The native FP4 scale tells you where
the training blocks are, not what the re-quantiser's optimal re-quantisation blocks are.

**Fit to this model.** The premise fits perfectly and the evidence for *not* snapping scales is strong. A two-level
scale scheme (one fp16 base per 64 plus a 2-bit offset per 32) is the only variant we would still consider: it would
cost +0.0625 bpw and could recover part of the finer-group gain, but we have not measured it and it needs a new block
layout and decoder. `[speculative]`

**Expected gain.** Zero to slightly negative for scale snapping and E8M0/32 at equal bytes. A two-level scale is
plausible for 2-4% weight-space MSE at +2.8% bytes, which is probably not worth a new format. The honest verdict is
that FP4-native *grid projection* is a dead end for our scalar tiers; the FP4 structure is already captured by the
free fp16 scale search.

**Cost.** The offline study is minutes on CPU. A two-level format would be 3-5 days of encoder + CPU/CUDA decoder work
for a marginal gain.

**Experiment and gate.** E0 (already done here): reproduce the table above on three experts from layers 5, 20 and 32,
with a diag(H)-weighted objective. Gate for any follow-up: >=2% lower held-out output error at equal bytes. The
measured weight-space result predicts failure, so this is a no-go unless the Hessian-weighted result reverses it.

### 3.2 GSQ refinement

**What it is.** GSQ (Gumbel-Softmax Quantization) keeps the scalar grid and the deployment format but jointly learns
the per-coordinate grid assignment and the per-group scale `[paper]` (<https://arxiv.org/abs/2604.18556>). For ternary
it uses separate mask and sign logits; for b > 2 bits it uses 2^b logits per coordinate, or five local shift logits
around the GPTQ grid point for b > 3. The objective is block output reconstruction, the relaxation is Gumbel-softmax
with an annealed temperature and a rectified-sigmoid binary regulariser, and the optimiser is Lion rather than AdamW
because saturated Gumbel relaxations kill gradients. Initialisation is a GPTQ warm start: logits are set from GPTQ's
zero/nonzero mask and sign, with noise and a scale, and the scales start at GPTQ's values. The code is public
(<https://github.com/IST-DASLab/GSQ>).

**Evidence.** At 2.13 bpp, GSQ scores 68.55 zero-shot average on Llama-3.1-8B-Instruct versus 37.53 for GPTQ at
2.25 bpp, 63.79 for EfficientQAT, 69.88 for QTIP at 2.00 and 69.83 for PV-Tuning at 2.27; on 70B, 75.57 versus 57.38
for GPTQ, 77.25 for QTIP and 76.27 for PV-Tuning. At 3.13 bpp, 72.32 versus 59.18 for GPTQ and 72.81 for QTIP. Ternary
GSQ on 8B reaches 63.44 at 1.71 bpp `[paper]`. The directly relevant result is the GGUF projection: on Qwen3-8B,
GSQ-Q2_K averages 56.28 versus Unsloth-Q2_K 50.03, and GSQ-Q3_K_M 61.61 versus 60.52 `[paper]`. For a trillion-scale
MoE, the paper quantises only non-shared experts in Kimi-K2.5 to 2 bits, leaving shared experts, router and vision
alone, and reports 24 hours of optimisation; at 2.13 bpp it loses 2.33 points on AIME25 and 12.72 on GPQA-D but gains
8.00 on LiveCodeBench and is neutral on MATH500 `[paper]`. That is the closest published analogue to our setting.

**Fit to this model.** Excellent on three counts. Our output format is fixed and unsupported by GSQ's deployment
formats, but the *algorithm* is format-agnostic; we already have a GPTQ initialiser, per-expert fit/held-out splits,
and per-expert Hessians. The 24-hour Kimi-K2.5 run is evidence that a trillion-parameter MoE is tractable with
block-wise, expert-local optimisation. The 1017 has 15,360 experts, but our per-expert objective uses only that
expert's routed tokens (median 2,120 fit tokens), not a whole-model calibration set, so the per-expert cost is much
smaller than GSQ's dense-model cost.

**Why our prototype failed.** `tools/v41/v41_repair.py` has a "GSQ/AdaRound-style" refinement: rectified sigmoid,
choice between the GPTQ code and the next code toward the original value, quadratic loss `tr(dW H dW^T)`, 200 steps,
scales fixed. It made layer 13 ternary error worse, 24.9% to 34.1% `[our measurement]`
(`docs/ds4/V41_1017_QUANT_PLAN.md`). That is not a refutation of GSQ; it is a refutation of a stripped-down variant
that (a) does not learn scales, (b) only moves one code step, (c) has no annealed Gumbel noise or Lion, and (d) uses
200 steps where GSQ uses 20 epochs over 4,096 x 4,096 tokens. The re-test must implement the actual objective or a
careful ablation of it, and must evaluate on held-out tokens, not on the fit set.

**Expected gain.** A 5-10% reduction in held-out expert output error at fixed tier and bytes is the planning number,
with the Qwen3-8B GGUF result as the upper anchor (+6.25 average points at Q2_K) `[paper]`. The gain should be larger
on the ternary tier, where the assignment matters most and where our GPTQ ternary already beats a Hadamard ternary
baseline.

**Cost.** A per-expert objective with ~2,100 fit tokens, 200-500 steps, one Hessian already computed: 0.1-0.5 s per
expert is a reasonable first estimate, or 0.4-2 h for all 15,360 experts; 3-12 h is the conservative planning range.
This is the same order as the existing GPTQ screen (1.48 s/expert for three tiers). `[inferred]`

**Minimum experiment.** Three layers (5, 20, 32), all 384 experts, three arms: GPTQ control, GSQ-mask+sign with fixed
scales, GSQ with learned scales. 300 steps, Lion, GPTQ warm start, fit/eval split unchanged. Command sketch: wire the
refinement into `v41_quant.py`'s `REFINE` hook and run
`V41_REFINE=gsq V41_REFINE_STEPS=300 python3 tools/v41/v41_quant.py layer 5 --qwork ... --map ... --no-write`.
Gate: >=5% lower held-out output error than the GPTQ control at identical tier and byte count on all three layers;
if it fails on layer 5 but passes on 20/32, keep it for the layers where it wins and re-run allocation.

### 3.3 Rotations

**What it is.** Multiply the weights and the activations by an orthogonal matrix so that outliers are spread across
channels before quantisation, then fold the transform away in adjacent weights or apply it at runtime. QuaRot uses
fixed Hadamard rotations on the residual stream, attention and KV cache; SpinQuant learns rotations; QuIP# uses a
randomised Hadamard transform (RHT) plus E8 lattice codebooks; TWLA uses a Kronecker-structured orthogonal rotation
chosen to make weights "ternary-friendly tri-modal" `[paper]`
(<https://arxiv.org/abs/2404.00456>, <https://arxiv.org/abs/2405.16406>, <https://arxiv.org/abs/2402.04396>,
<https://arxiv.org/abs/2606.13054>). Bonsai's ternary 27B stores every language-model weight ternary in a fixed
rotated basis, with blockwise Hadamard rotation folded into the stored weights, and retains 98.2% of its Qwen3.8-27B
parent (84.78 vs 86.32 on a 14-benchmark thinking suite) at 1.75 bpw `[community/paper]`
(<https://docs.prismml.com/models/bonsai-2-27b>). HIGGS shows the same idea data-free: Hadamard rotations plus
MSE-optimal grids outperform NF4 `[paper]` (<https://arxiv.org/abs/2411.17525>).

**Evidence at 4-6 bits is strong.** QuaRot quantises Llama-2-70B to 4-bit weights/activations/KV with at most 0.29
WikiText-2 perplexity loss and 99% of zero-shot performance, and its microbenchmark shows the Hadamard adds about
0.09 ms to a 5120x5120 batch-1 INT4 matvec (0.464 -> 0.547 ms) `[paper]`. SpinQuant shows learned rotations beat
random Hadamard at 4-4-4: 69.3 vs 65.1 average on LLaMA-3 70B `[paper]`. At 2 bits, QuIP#'s RHT plus lattice
codebooks reaches 3.91 WikiText-2 on Llama-2-70B versus 5.12 FP16 `[paper]`; QTIP improves that to 3.70 with trellis
codes `[paper]`.

**Why the evidence does not transfer cleanly to 1.75-bit scalar ternary.** For vector quantisation, rotating to an
approximately Gaussian ball is close to a requirement: the codebook is designed for that distribution. For a scalar
grid with 3-4 levels, what matters is the *shape* of each group's weight distribution, not whether it is
sub-Gaussian. With FP4-trained weights, the per-32 block distribution is already bounded and quantised; there are no
classic 50x outliers to disperse. Our own A/B is the most relevant evidence: a mesh Hadamard ternary implementation
at the same 1.75 bpw was 1.36x worse than our GPTQ ternary on gate+up (it won 19% of experts), 0.985x on down (a
tie), and our own earlier Hadamard test bought less than 1 GiB at equal quality and needed a fused down-projection
kernel `[our measurement]` (`docs/ds4/V41_START_HERE.md`, `verify/s6/hadamard-engine-map.md`). Bonsai and TWLA fold
the rotation into *training/fine-tuning*, not pure PTQ; Bonsai explicitly says the discretisation and rotation were
incorporated into the training loop `[community]`. That is not available to us at 552B.

**Where the inverse lives.** For the routed-expert branch only, no per-expert inverse is needed if the same
block-diagonal Hadamard H is used for every expert and both operands are rotated:
`x' = xH`, `W1' = W1H` gives the same gate/up output; for down, `a' = aH` and `W2' = W2H` gives the same output. The
router and the shared expert must keep seeing the unrotated `x`, so the rotation is applied to the tier's own copy of
the MoE input (one transform per token per layer) and to each expert's SwiGLU activation (one transform per entry).
The summed expert output does not need an inverse because the down projection is already rotated on both sides.
`verify/s6/hadamard-engine-map.md` traces the exact hand-off points in `ds41_moe.cpp`; the cheap half is a block
rotate before the activation quantise, while the down rotation is invasive because the default CUDA path fuses SwiGLU
straight into q8_1. A blockwise 128 Hadamard is the right block size for our formats: it is a power of two, it can be
fused with the group quantiser, and it does not mix across quantisation groups. The runtime cost is one D=5,120
transform per token per layer plus six FF=2,304 transforms (one per routed expert), all ~O(n log n) and a few hundred
KB of memory traffic per layer, so it should be well under 0.5 ms/token if fused. `[inferred]`

**Expected gain.** Low. The literature's rotation wins are at 2-bit *vector* quantisation or 4-bit activation
quantisation; our equal-bits A/B lost. A blockwise variant matched to the group boundaries and applied only to the
ternary tier is the one untested corner, and even that is a 0-3% bet.

**Cost and experiment.** Medium kernel work (a rotate kernel, a fused SwiGLU-rotate-quantise path, graph-capture
safeguards). Experiment: three layers, 128 experts, ternary only, blockwise H=128 on x and on the expert's SwiGLU
activation, GPTQ at both arms. Gate: >=2% lower held-out output error at equal bytes *and* measured engine cost
<=0.5 ms/token. If the error gate fails, stop; if it passes but the engine cost gate fails, the format is not
deployable on our decode path.

### 3.4 MoE-aware calibration and allocation

**What it is.** Use routing statistics, gate weights and expert load to decide calibration coverage and bit
allocation. `v41_alloc.py` already does the essentials: the cost of a tier is held-out output error with the routing
weight included, per projection, normalised per layer, with edge weighting and a floor for under-observed experts.

**Evidence.** MoEQuant identifies inter-expert sample imbalance and intra-expert affinity imbalance, and its
expert-balanced self-sampling plus affinity-guided quantisation gains more than 10 HumanEval points on DeepSeekMoE-16B
at 4-bit `[paper]` (<https://arxiv.org/abs/2505.03804>). EAQuant adds smoothing aggregation, routing-consistency
alignment and calibration-data balance, with 1.15-13.81% average accuracy gains across three MoE architectures at
W4A4/W3A4/W3A3/W2A4 `[paper]` (<https://arxiv.org/abs/2506.13329>). MxMoE's two observations are that linear blocks
have very different sensitivity and that expert activation frequencies are heterogeneous; it reports 2.4 lower
WikiText-2 perplexity than GPTQ at 2.25-bit `[paper]` (<https://arxiv.org/abs/2505.05799>). QuantMoE-Bench finds
structure-aware mixed precision at 65.35% versus GPTQ's 64.30%, and specifically suggests shared experts may need more
bits than routed ones `[paper]` (<https://arxiv.org/abs/2406.08155>). GEMQ replaces per-layer importance with a
global linear program and adds router fine-tuning; at 1.5 bits on Mixtral, GEMQ is 7.93 WikiText-2 versus 10.67 for
uniform, and router fine-tuning alone moves 9.66 to 7.69 `[paper]` (<https://arxiv.org/abs/2605.23078>). KBVQ-MoE
notes that expert redundancy wastes codebook capacity and that aggregation amplifies output bias, and shares a
KLT-SVD component across experts `[paper]` (<https://arxiv.org/abs/2602.11184>). RCO formalises budget-exact
assignment as optimisation on a Riemannian manifold, which is the principled version of our Lagrangian
`[paper]` (<https://arxiv.org/abs/2605.00649>).

**Fit to this model.** Coverage is already adequate (median 494 held-out tokens per expert; 1.8% under-observed), so
EBSS-style resampling is not the lever. The levers that fit are (a) the global vs per-layer objective, (b) domain
weighting, and (c) per-domain error stats. Our current objective normalises each layer by its routed energy so every
layer counts equally; that is a deliberate robust choice, but it ignores the fact that a 1% error in layer 32 costs
more than a 1% error in layer 5. GEMQ's global LP and QuantMoE-Bench's learned layer importance both attack exactly
that. Domain weighting is also absent: we care about code and agentic behaviour, and our held-out set has per-domain
labels, but the screening stats store only a single aggregate energy and error per expert, so a per-domain allocation
would require re-running the calibration half with per-domain accumulators.

**Expected gain.** 2-5% proxy error at equal bytes from a global LP plus domain weights; no quality guarantee until
the head gate is run. Router fine-tuning is not applicable in the usual form because our router stays F32 and routing
is computed on unquantised hidden states; what drifts is the hidden state itself, and the sequential quantise pass
already adapts later layers to that drift `[inferred]`.

**Cost.** Allocation is CPU-only (seconds). Per-domain stats need a calibration rerun (~11 min/layer including GPTQ
screening, or ~3 min/layer with RTN screening), i.e., 2-7 h for 40 layers. Router fine-tuning is not recommended.

**Experiment and gate.** Re-solve the existing 1017 stats with a global LP, per-projection, with domain weights
code 0.4 / agentic 0.3 / rest 0.3 approximated through the existing per-layer energies. Gate: >=2% lower proxy at
equal bytes; then a 3-layer head-gate check (layers 5, 20, 32 chained to the head) must not be worse than the current
map. If the proxy improves but the head gate does not, do not adopt it.

### 3.5 Vector and codebook quantisation

**What it is.** Quantise groups of weights to indices into a learned codebook instead of scalar grids: AQLM uses
additive multi-codebook quantisation, QuIP# uses E8 lattice codebooks after a randomised Hadamard transform, QTIP uses
trellis-coded quantisation with computed codes, VPTQ uses second-order optimisation and channel-independent
refinement, GPTVQ uses vector groups with Hessian updates, and HIGGS uses Hadamard plus MSE-optimal grids
`[paper]` (<https://arxiv.org/abs/2401.06118>, <https://arxiv.org/abs/2402.04396>,
<https://arxiv.org/abs/2406.11235>, <https://arxiv.org/abs/2409.17066>, <https://arxiv.org/abs/2402.15319>,
<https://arxiv.org/abs/2411.17525>).

**Evidence.** At 2 bits on Llama-2-70B, QTIP reaches 3.70 WikiText-2 versus 3.91 for QuIP#, 3.83 for AQLM and 3.12
FP16; its batch-1 throughput on an RTX 6000 Ada is 23.5 tok/s versus 22.2 for QuIP# and 8.78 for AQLM, and the authors
report <=4 instructions per weight and over 80% of peak memory bandwidth during decoding `[paper]`. VPTQ reports
0.01-0.34 perplexity reduction on LLaMA-2, 0.38-0.68 on Mistral-7B and 4.41-7.34 on LLaMA-3 at 2 bits versus prior
state of the art, with 10.4-18.6% of the quantisation time and 1.6-1.8x throughput `[paper]`. GPTVQ takes 3-11 hours
on one H100 for Llama-2-70B `[paper]`. AQLM says a 7B model takes about one day on one A100 `[paper]`. KBVQ-MoE
applies VQ to MoE experts with shared KLT-SVD components and bias-corrected output, reaching 67.99 vs 68.07 FP16 at
3 bits on Qwen1.5-MoE `[paper]`. PTQTP is a different point on the same curve: two ternary trit-planes instead of one,
which rivals 1.58-bit QAT, needs a single hour of quantisation versus 10-14 GPU days for training-based methods, and
runs 4.63x faster than FP16 `[paper]` (<https://arxiv.org/abs/2509.16989>).

**Fit to this model.** Quality-wise this is the most promising family: a 2.0-2.25 bpw vector format could plausibly
beat our 2.09 bpw scalar mix by a meaningful margin, and PTQTP's dual trit-planes are attractive for the Q3_K tier
because they use only adds and a second ternary dot. The blocker is the engine, not the algorithm. Our decode reads
whole expert blobs at random from three media tiers; the CPU pool already spends 1.1-1.4 ms per expert on base-3
decode, and the ternary CUDA kernel is ALU-bound at 136 GB/s. A codebook format needs (a) a per-layer codebook small
enough to stay resident, (b) a decoder that beats or matches the current base-3 unpack on both CPU and CUDA, and (c)
random access at expert granularity. QTIP's computed-code trellis is attractive because it needs no codebook lookup
and only a few instructions per weight, which is the closest match to our existing kernel style. KBVQ-MoE's shared
KLT-SVD component is attractive for MoE because our experts are known to share per-layer distributions.

**Expected gain.** Medium-high quality, high cost. If QTIP-style 2-bit codes reach the published 0.58 ppl gap at 70B
versus FP16, that is far better than our current +5.5% head-gate gap, but the comparison is across models and formats
and cannot be transferred as a number `[inferred]`.

**Cost.** The honest estimate is 2-6 weeks for a production CPU + CUDA decoder plus a GGUF type, plus quantisation
time. A PTQTP-style dual ternary tier is smaller (a second ternary dot and a new packing) but still a new type on both
paths.

**Experiment and gate.** Do the *simulation* first, with no kernels: on three layers, fit a QTIP-style 2-bit trellis
and a PTQTP dual-plane at the same bytes as our mixed scalar map, measure held-out output error. Gate: >=10% lower
error at equal bytes (the bar for justifying kernel work). If the simulation wins by less than 10%, the engine risk
dominates and it should not be built.

### 3.6 Fine-tuning and healing after quantisation

**What it is.** Recover quality after the integer assignments are fixed: tune scales, add low-rank corrections, fit
per-channel affine compensation, or distil from the full-precision teacher.

**Evidence.** PV-Tuning shows that straight-through estimation is sub-optimal for extreme compression and proposes a
representation-agnostic fine-tuning framework, reaching the first Pareto-optimal 2-bit Llama-2 models `[paper]`
(<https://arxiv.org/abs/2405.14852>). EfficientQAT trains all block parameters in a block-wise phase (Block-AP) then
tunes only quantisation parameters end-to-end (E2E-QP); it produces a 2-bit Llama-2-70B on one A100-80GB in 41 hours
with less than 3 points degradation (69.48 vs 72.41) `[paper]` (<https://arxiv.org/abs/2407.11062>). LLM-QAT distils
from the model's own generations without training data `[paper]` (<https://arxiv.org/abs/2305.17888>). Unsloth reports
that 4-bit QAT recovers up to 70% of the lost accuracy and adds 1-3% on GPQA/MMLU-Pro `[community]`
(<https://unsloth.ai/docs/blog/quantization-aware-training-qat>). KBVQ-MoE's bias-corrected output stabilisation adds a
channel-wise affine compensation after VQ and reports near-lossless 3-bit MoE `[paper]`. ScaleQ-1.58's calibration
insight is orthogonal but important: reason about the model's own thinking traces during ternarisation (AYOT), which
with 4M tokens reaches over 90.52% of a BitNet b1.58 2B4T model on four math/coding tasks for Qwen3-1.7B, +8.97
absolute on Qwen3-4B, and generalises up to 235B and to MoE `[paper]` (<https://arxiv.org/abs/2608.01078>).

**Fit to this model.** Full QAT is out of reach on one 4090 for 552B, but three pieces are feasible with the
layer-streaming teacher we already have. First, *scale-only tuning*: with the integer codes fixed, optimise the
per-group scales on the fit split, which is EfficientQAT's E2E-QP restricted to our scale count. Second, *per-expert
affine healing*: a D-vector bias and a per-channel scale, fitted on fit tokens, which is the KBVQ-MoE recipe and a
direct answer to MoE aggregation bias. Third, *teacher-generated reasoning traces in calibration*: the AYOT result
suggests that plain-text calibration under-represents the thinking distribution that the Gauntlet and code tasks
exercise. Our current mix is already agentic-heavy, but it does not contain the model's own CoT traces. Our earlier
LQER-style low-rank corrections gave 1-5% error reduction for up to +14% size and were rejected; the affine variant
is far cheaper (a D-vector and a channel scale, ~0.1% of expert bytes) and should be tested before any adapter.

**Expected gain.** Per-expert affine: 1-3% output error for ~0.1% bytes, with no new quantisation pass. Scale-only
tuning: 1-4%, with a cost similar to the existing GPTQ screen. AYOT calibration: 1-3% on reasoning/code domains,
mostly through better assignment and allocation rather than a new format. LoRA/adapters: poor value at our size.

**Cost.** Affine healing: minutes to ~1 h on GPU, or CPU if activations are cached. Scale-only tuning: 0.1-1 s per
expert, 0.4-4 h. AYOT: one calibration forward (~11 min/layer), plus re-screening.

**Experiment and gate.** Start with affine healing on three layers. Fit on the fit split, evaluate on held-out
output error. Gate: >=2% lower error for <=0.1% extra expert bytes and <=0.1 ms/token in the engine. Scale-only tuning
is the next arm of the same experiment. If both fail, the remaining healing avenue (distillation) is not worth the
compute on this box.

### 3.7 The dense half and non-expert parts

**What is there.** Attention (including the CSA compressor and indexer), the shared expert, the output head, the
embedding, norms, router and mHC. The dense half is roughly 4% of the model's parameters but a much larger share of
decode time: Q8_0 mat-vecs are 8.9 ms of a 37.5 ms token, at ~850 GB/s `[our measurement]`.

**What is worth bits.** Our current choice is Q8_0 in the file and Q6_K at load, measured at +11-35% decode and +1%
KL with identical edit text `[our measurement]`. That is a good trade. Going below Q6_K is risky: the community
DeepSeek-V4 Q2 analysis identifies three paths that break at 2 bits, and all three are dense-side: the attention
output projection `Wo` (noise stacks with sequence length), the DSA indexer's top-k selection (a flattened score
distribution selects the wrong tokens), and the RoPE-dependent positional logits (aliasing past a few thousand
tokens) `[community]` (<https://github.com/paean-ai/deeptide/blob/main/docs/deepseek-v4-flash-q2.md>). DeepSeek
themselves keep everything except the expert weights and the indexer QK in FP8/BF16, and they even keep the indexer's
top-k scores in BF16 to preserve 99.7% KV recall `[paper]`. The indexer QK path is already FP4 in the checkpoint, so
there is no extra saving available there without retraining. The norms, router and mHC are F32 and negligible in
bytes; the router in particular decides expert selection and should stay high precision.

**Fit, expected gain and cost.** The dense half is ~4% of parameters but roughly a quarter of decode time, so every
GiB moved from experts to dense is a worse trade than the reverse. Expected gain from more dense bits is negligible;
expected saving from fewer dense bits is ~4-5 GiB at a risk concentrated on exactly the paths the community
identifies as fragile. A dense re-quant is minutes; a KV-compression experiment is ~1-2 h plus a serve test. The
byte budget is better spent on the experts.

**KV and RAM.** The one dense-side lever worth measuring is KV compression if it frees pinned RAM for the 60 GiB
expert arena. The attention state is already CSA-compressed and the RoPE dimensions are kept higher precision; our
engine docs put attention-level levers at <=0.7 ms total, so this is a RAM-capacity question, not a decode-time one
`[our measurement]`. Measure arena hit rate before and after, not perplexity alone.

**Experiment and gate.** No new experiment is needed for dense quality; the existing Q6_K-at-load measurement is the
baseline. If a KV compression experiment is run, gate it on expert-arena hit rate and decode tok/s with KL not worse
than +1%, and abandon it if the arena does not grow.

### 3.8 DeepSeek-specific and community results

**What DeepSeek does.** They do not serve a 2-bit model. They train the experts in MXFP4 during post-training, with
the FP32 master weights and an STE through the quantiser, and they quantise the indexer QK path to FP4 and its scores
to BF16 `[paper]`. The report notes that FP4 x FP8 currently runs at the same speed as FP8 x FP8 on existing
hardware, with a theoretical 1/3 improvement on future hardware `[paper]`. This matters for us twice: the model is
already adapted to a coarse 4-bit grid, and the vendor's own deployment is 4-bit, not 1-2 bit. Our 2.09 bpw build is
therefore a genuinely more aggressive regime than the one the model was trained for.

**Unsloth.** Their lossless route is a bit-exact MXFP4 repack of the experts plus FP8/BF16 for everything else; the
Q8 file is 162 GB and their Q4 file keeps the same bit-exact experts and only quantises the 4% non-expert tensors
`[community]` (<https://unsloth.ai/docs/models/deepseek-v4>). This is the strongest available evidence that the
expert FP4 path should be preserved exactly when the goal is lossless, and that the non-expert 4% is where their
low-bit files spend their risk.

**DwarfStar (antirez/ds4).** Asymmetric 2-bit: routed experts at IQ2_XXS/Q2_K, shared experts, projections and other
decision-critical tensors kept high precision; imatrix-tuned and validated `[community]`
(<https://dwarfstar.sh/docs/architecture/>, <https://github.com/antirez/ds4>). A quality-focused community benchmark
puts the ~81 GB Q2 build at about 76% SWE-bench Verified (estimated from a 100-case subset, 95% lower bound 68.8%)
versus 79.0% for the vanilla model, 70.3% on an Aider Polyglot Python/Rust subset, and 78.1 MMLU-Pro versus 83.0
(temperature-1 caveat) `[community]` (<https://gist.github.com/Farenheith/c7c6c75b2918d7074af96ec06672ff07>). The
deeptide note is a useful counterweight: the Q2 build is a technology-feasibility preview, not a production coding
model, and the 153 GiB Q4_K mixed build is the recommended in-architecture upgrade `[community]`. Our 1017 build sits
between these at 132.4 GiB of experts / 152 GiB total and 2.09 bpw, with a +5.5% head-gate gap and a real Gauntlet
regression (3/5 vs 4/5). That regression is the single most important number to attack; it is consistent with the
community finding that 2-bit damage shows up first on agentic and long-context tasks.

**ik_llama.cpp and trellis.** The fork ships IQ2_KT (about 2.125 bpw) and other trellis types derived from QTIP, with
CPU and CUDA kernels `[community]` (<https://github.com/ikawrakow/ik_llama.cpp>). This is the best evidence that a
trellis format can be made deployable in a ggml-style engine, but it is also evidence of the scale of the work: a new
type on both CPU and CUDA plus all the conversion and validation plumbing. Our engine would need the same.

**What this means for us.** The community consensus for this model family is: keep the dense/critical path high
precision, spend the aggressive quantisation on routed experts, calibrate on agentic/code data, and expect 2-bit to
cost agentic quality first. Our pipeline already follows that recipe.

**Fit, expected gain and cost.** The expert-only strategy is already ours; the value of this direction is validation
and an external control, not a new algorithm. Expected gain is zero directly, but DwarfStar's IQ2_XXS-gate/up plus
Q2_K-down layout is a useful independent point at roughly our byte count, and its quality numbers are a warning that
our Gauntlet regression is the expected failure mode rather than an anomaly. Cost is a literature/benchmark review
(hours), plus one offline comparison arm. **Experiment and gate:** build an IQ2_XXS-style map (gate/up pushed one
tier cheaper, down one tier richer) on the existing stats and compare it against our RCO-lite map on the same three
layers; gate = our map must be >=2% better on held-out output error at equal bytes, otherwise the community
asymmetry is telling us something our proxy misses and should be investigated before the next full build.

---

## 4. A proposed DSV41-native method

The combined recipe, in pipeline order, with the changes marked:

1. **Calibration.** Keep the V3 agentic-first mix and its held-out split. Add 4-8 sequences of the model's own
   reasoning traces (teacher-generated, or from the served 1017 build) to the fit split only. Do not use them for
   evaluation; the point is to expose the quantiser to the thinking distribution, following ScaleQ-1.58's AYOT
   result. Cost: one calibration forward, no extra GPU for generation if traces are captured from the existing
   probe.
2. **Per-expert statistics.** Keep the layer-streaming teacher and expert-major MoE. For each expert, accumulate the
   gate+up input Hessian and the down input Hessian, the fit/held-out split, the routed token count, and the gate
   weight sum. Store per-domain energy and error accumulators in addition to the aggregate, so allocation can weight
   code/agentic/rest. This is the only calibration-side code change.
3. **Initial assignment.** Keep GPTQ at the three exact grids with static diag(H)-weighted scales, act order on, and
   the RTN screen with the measured GPTQ/RTN ratios (0.992-0.996 spearman; -0.77% true cost). Do not adopt FP4-grid
   scale snapping or E8M0 scales; the measurements in section 3.1 say they lose at equal bytes.
4. **Refinement.** Run a real GSQ-style refinement per expert, initialised from GPTQ, with mask and sign logits and
   learned group scales, on the fit split, evaluated on the held-out split. Use Lion, an annealed temperature, and a
   short schedule matched to our token counts (200-500 steps), and refine only experts whose GPTQ error is above a
   threshold (the tail where the allocation would otherwise have to spend Q3 bits). This is the main new algorithm.
   It writes the same formats, so no engine change.
5. **Allocation.** Replace the per-layer normalised Lagrangian with a global LP (or RCO's manifold-constrained
   version) over per-expert, per-projection costs, with domain weights and the existing edge/tier caps. Re-run the
   allocation after refinement, because refinement changes the cost ordering.
6. **Sequential pass and healing.** Run the sequential quantise pass with `--propagate quant` as today. For the
   worst-error experts, fit a per-expert output bias and per-channel scale on the fit split and store them as a small
   side table; add the affine epilogue in the engine. This is the KBVQ-MoE bias-corrected output stabilisation,
   restricted to the experts that need it.
7. **Optional rotation, only if the experiment wins.** A blockwise H=128 on the routed-expert input and on each
   expert's SwiGLU activation, folded into the new quantiser and the engine's existing hand-offs. This is the only
   change that touches the CUDA graph, so it goes last and only with a measured engine-cost win.
8. **Export and validate.** Bit-exact GGUF encoders unchanged; head gate, FP8 fixtures, end-to-end KL, Gauntlet,
   overthinking pairs, then serve.

**What is new versus off-the-shelf.** GSQ operates block-wise on dense models and projects back into GGUF K-quants;
our variant operates per routed expert on the token distribution the router actually gives that expert, with the
held-out split used as a gate, and writes three engine-native formats. RCO's exact-budget optimisation is applied at
expert-projection granularity with per-domain weights, not only per block. The FP4-native part of the design is a
negative result (do not snap scales), which is itself useful: it removes a tempting but losing branch. The affine
healing is a direct MoE-specific addition that the dense GSQ pipeline does not need.

**What is deliberately not in the recipe.** Vector/codebook quantisation (too much kernel work before a simulation
gate), full QAT (out of reach on one 4090), router fine-tuning (our router is unquantised and the hidden-state drift
is already handled by sequential propagation), and any dense-side reduction below Q6_K at load.

---

## 5. Experiment plan

All experiments assume the frozen V3 calibration work dir and the current map; no experiment touches the serving
process or the peer's quant pass. GPU time is an estimate on this 4090; the first three items fit the "<=2 GPU-hours
each" constraint.

| # | Experiment | Inputs | Command sketch / change | GPU time | Pass gate | Go/no-go |
| --- | --- | --- | --- | ---: | --- | --- |
| E1 | **GSQ refinement, fixed scales** | layers 5/20/32, 384 experts, existing h/pre/stats | Add `V41_REFINE=gsq` (mask+sign logits, GPTQ warm start, 300 steps, Lion) to `v41_quant.py`; run `v41_quant.py layer L --no-write --map alloc` in a scratch qwork | ~1-2 h | Held-out output error >=5% below the GPTQ control at identical tiers/bytes on all three layers | If pass, E1b (learned scales); if fail on all, stop |
| E1b | **GSQ refinement, learned scales** | as E1 | Same, plus per-group scale logits | ~2 h | >=5% below E1's fixed-scale arm | If pass, adopt; if E1 passes and E1b does not, ship fixed-scale |
| E2 | **Affine healing** | layers 5/20/32, fit tokens | Fit D-vector bias + per-channel scale on fit split; evaluate held-out output error; no format change | ~0.5 h | >=2% lower error for <=0.1% bytes | If pass, prototype the epilogue; if fail, drop |
| E3 | **Global LP + domain weights** | existing 1017 stats | Extend `v41_alloc.py` with `--global-lp --domain-weights code=0.4,agent=0.3,rest=0.3`; CPU only | 0 (CPU minutes) | >=2% lower proxy at equal bytes; 3-layer head gate not worse | If pass, full-map alloc and head gate |
| E4 | **Blockwise Hadamard ternary** | layers 5/20/32, 128 experts | Rotate x and SwiGLU activation with H=128; GPTQ at both arms | ~1 h | >=2% lower held-out error at equal bytes; engine cost <=0.5 ms/token | If error gate fails, stop; if engine gate fails, do not deploy |
| E5 | **FP4-native grid ablation** | 3 experts, CPU only | Reproduce the section 3.1 table with diag(H) weighting | 0 (CPU minutes) | >=2% lower error at equal bytes | Already predicted negative; only run if a two-level-scale variant is proposed |
| E6 | **VQ/PTQTP simulation** | layers 5/20/32 | Offline QTIP-style trellis and PTQTP dual-plane at equal bytes; no kernels | ~1 h CPU | >=10% lower error at equal bytes | If pass, scope the kernel work; if fail, close |
| E7 | **Full rebuild with winners** | full 40 layers | Recalibrate (if AYOT adopted), re-screen, refine, alloc, sequential pass, export | ~10 h + calibration | Head gate ppl <=6.85, top-1 >=0.670, Gauntlet >=3/5 | If head gate passes but Gauntlet regresses, investigate domain-specific error before shipping |

Ordering and dependencies: E1 and E2 are independent and can run back to back on the same layers. E3 can run
immediately on the existing stats and should be done in parallel with E1 because it changes the map E1 refines
against. E4 is independent but low-probability and should not block E1/E3. E6 is a scoping study and can run on CPU
while the GPU is busy. E7 only after at least one of E1/E2/E3 passes its gate.

What "held-out output error" means throughout: the same metric `v41_calib.py` and `v41_quant.py` already report,
`sum ||f(x) - f(x; Q)||^2 / sum ||f(x)||^2` on the held-out sequences, with the routing weight included. It is a
proxy, and the head gate is the real gate. The Gauntlet and the API fixtures are the final arbiters for agentic and
code behaviour.

---

## 6. Engine and kernel impact

| Change | Bytes / bpw | Decode cost | Kernel work | Risk | Verdict |
| --- | --- | --- | --- | --- | --- |
| GSQ refinement (fixed/learned scales) | unchanged (same formats) | none | none | quantisation time; overfit if the held-out split is not respected | **Primary** |
| Per-expert affine healing | +2 x D fp16 per expert = ~20 KB x 15,360 ~= 0.3 GB (~0.2%) | one FMA per output element; <0.1 ms/token | epilogue in the expert output path, CPU and CUDA | extra state; must be stored in the sidecar/rank data | **Secondary** |
| Global LP / domain-weighted allocation | unchanged | none | none | proxy/head-gate mismatch | **Secondary** |
| Blockwise Hadamard (H=128) | unchanged | ~2 D-transforms + 6 FF-transforms per token/layer; <0.5 ms if fused | rotate kernel, fused SwiGLU-rotate-quantise, graph-capture care | down rotation is invasive; per-expert R2; transposed-convention bug risk | **Conditional** |
| Q2 with two-level scale (fp16/64 + 2-bit offset/32) | 2.25 -> 2.3125 bpw (+2.8%) | one extra scale decode | new block layout, CPU/CUDA decoder | marginal gain, more bytes | **Defer** |
| PTQTP dual trit-planes (for Q3 tier) | ~3.2 bpw vs 3.44 (inferred) | two ternary dots; adds instead of multiplies | new packing, CPU/CUDA dot path | slower CPU pool; format churn | **Defer** |
| QTIP-style trellis VQ | ~2.0-2.25 bpw | LUT/computed-code decode; published 23.5 tok/s on 70B single GPU | new type, CPU + CUDA, codebook handling | 2-6 weeks; decode could lose to base-3 unpack | **Defer behind simulation** |
| Dense Q6_K at load (current) | ~6.56 bpw dense | measured +11-35% decode | none | +1% KL | **Keep** |
| Dense below Q6_K / attention / indexer / RoPE | saves ~4-5 GiB | unknown | none | community evidence of long-context collapse; vendor keeps these high precision | **Do not** |

The decode budget is the tie-breaker. Expert bytes are already the dominant traffic, and the ternary kernel is
ALU-bound rather than bandwidth-bound `[our measurement]`. A format that adds arithmetic to the ternary path (dual
planes, trellis decode) can lose even if it is smaller; a format that only changes the scale interpretation is cheap.
That is why GSQ, allocation and affine healing rank above every new format: they buy quality with zero decode cost.

---

## 7. Risks and open questions

**The GSQ re-test is the highest-variance item.** Our prototype regressed and we do not yet know which ingredient was
missing. The three candidates are the fixed scales, the one-step move, and the absence of Gumbel noise/Lion. The
experiment isolates them; if all three variants regress, our per-expert quadratic objective may be a poor match to
GSQ's block-output objective, and the direction should be closed rather than patched indefinitely.

**The held-out metric is a proxy.** Our screening error is measured on held-out sequences with the routing weight
included, but the final head gate and the Gauntlet are the only numbers that matter. The 1017 build's Gauntlet
regression (3/5 vs 4/5) shows that a small aggregate ppl change can hide an agentic regression. Every experiment
that changes the expert map should end with a head-gate and, for the final candidate, a Gauntlet run.

**FP4-native analysis is weight-space.** The section 3.1 tables use one expert and plain MSE, without diag(H)
weighting and without GPTQ. The conclusion (free scales beat snapped/E8M0 scales) is robust in sign but the
percentage could move under the real objective. E5 is the confirmation, and it is cheap.

**Rotation and the FP4 grid.** A rotation mixes the native FP4 blocks, converting exactly representable weights into
a continuous matrix. This does not violate anything, but it means the "FP4-native" exactness argument and the
rotation argument pull in opposite directions. We should not pursue both at once.

**The two-level scale idea is unmeasured.** It is the only FP4-native variant that could still win, and it costs
+0.0625 bpw. It needs a new block layout and a decoder. Do not build it unless E5 shows a >=2% Hessian-weighted gain.

**Calibration drift.** `v41_calib.py prep` reads live local text, so re-running it does not reproduce an old set; the
saved tokens/meta are the record. Any AYOT addition must be versioned with the calibration work dir, and the old set
kept for comparability.

**Blast radius.** Everything above is offline quantisation work except the affine epilogue and the Hadamard kernel.
Both of those touch the decode path and must be gated on measured engine cost, not only on error. The serving build
and the peer's V3 quant pass must not be touched during the experiments; scratch qworks and `--no-write` arms are the
safe way to run E1/E2/E4.

**Open questions.** Does the 1017's agentic regression come from the chat-weighted calibration mix, the gentler tier
map, the edit, or the checkpoint itself? The card lists all four as candidates and V3 goes after the first two. A
domain-weighted allocation and AYOT calibration are the method-side contributions to that question; they do not
settle it. Is the 24-hour Kimi-K2.5 GSQ result on hardware far beyond one 4090? Almost certainly, but the per-expert
schedule here is much smaller, and the E1 arm will give the real number for our box.

---

## 8. Sources

**Quantisation methods.**
- GSQ: <https://arxiv.org/abs/2604.18556>; code <https://github.com/IST-DASLab/GSQ>
- RCO: <https://arxiv.org/abs/2605.00649>
- QuaRot: <https://arxiv.org/abs/2404.00456>
- SpinQuant: <https://arxiv.org/abs/2405.16406>
- QuIP#: <https://arxiv.org/abs/2402.04396>
- QTIP: <https://arxiv.org/abs/2406.11235>
- AQLM: <https://arxiv.org/abs/2401.06118>
- VPTQ: <https://arxiv.org/abs/2409.17066>
- GPTVQ: <https://arxiv.org/abs/2402.15319>
- HIGGS: <https://arxiv.org/abs/2411.17525>
- PV-Tuning: <https://arxiv.org/abs/2405.14852>
- EfficientQAT: <https://arxiv.org/abs/2407.11062>
- LLM-QAT: <https://arxiv.org/abs/2305.17888>
- TWLA: <https://arxiv.org/abs/2606.13054>
- ScaleQ-1.58: <https://arxiv.org/abs/2608.01078>
- PTQTP: <https://arxiv.org/abs/2509.16989>
- AWQ: <https://arxiv.org/abs/2306.00978>
- SVDQuant: <https://arxiv.org/abs/2411.05007>

**MoE-specific.**
- MxMoE: <https://arxiv.org/abs/2505.05799>
- MoEQuant: <https://arxiv.org/abs/2505.03804>
- EAQuant: <https://arxiv.org/abs/2506.13329>
- QuantMoE-Bench: <https://arxiv.org/abs/2406.08155>
- GEMQ: <https://arxiv.org/abs/2605.23078>
- KBVQ-MoE: <https://arxiv.org/abs/2602.11184>

**Model and FP4 formats.**
- DeepSeek-V4 technical report (FP4 QAT, section 5.2.1): <https://arxiv.org/abs/2606.19348>
- NVIDIA, Pushing Intelligence to 4-bit (E2M1 levels, MXFP4/NVFP4): <https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/>
- MXFP4 error reduction (OAS/MBS): <https://arxiv.org/abs/2603.08713>
- FP4 All the Way: <https://arxiv.org/abs/2505.19115>

**Community / deployment.**
- Unsloth DeepSeek-V4 guide (bit-exact MXFP4 experts): <https://unsloth.ai/docs/models/deepseek-v4>
- Unsloth QAT: <https://unsloth.ai/docs/blog/quantization-aware-training-qat>
- DwarfStar architecture: <https://dwarfstar.sh/docs/architecture/>; code <https://github.com/antirez/ds4>
- DwarfStar Q2 quality benchmark: <https://gist.github.com/Farenheith/c7c6c75b2918d7074af96ec06672ff07>
- Deeptide Q2 capability limits: <https://github.com/paean-ai/deeptide/blob/main/docs/deepseek-v4-flash-q2.md>
- Bonsai ternary 27B: <https://docs.prismml.com/models/bonsai-2-27b>
- ik_llama.cpp (trellis types): <https://github.com/ikawrakow/ik_llama.cpp>

**Our own records.** `docs/ds4/V41_1017_QUANT_PLAN.md`, `docs/ds4/V41_TERNARY_PROFILE_PLAN.md`,
`docs/ds4/V41_START_HERE.md`, `verify/s6/hadamard-engine-map.md`, `tools/v41/hf-1017/README.md`, and the
calibration statistics under `/home/mal/v41-work/calib-1017-v3/stats`.
