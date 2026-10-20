# XQ — Xolotl Quant: the DeepSeek-V4.1-specific quantisation pipeline (living doc, RESUME FROM HERE)

Owner: `xolotl-engine` session (relay name), Mal's call 2026-10-19 17:05: **"XQ" replaces "GSQ" in every quant name**,
and the DeepSeek-specific GSQ-style refinement becomes the XQ pipeline proper.  Keep this file current at every step:
it is the resume point if a session runs out of context.  Newest status first in section 0.

---

## 0. Status log (newest first)

- **2026-10-19 17:10** - Doc started.  State:
  - The released builds were *not* GSQ: they are GPTQ + our "RCO-lite" Lagrangian tier allocator on GSQ-family grids
    (ternary PTQ1_0 / Q2_0 / Q3_K).  Hence the rename to XQ (Mal).  Stage names from now on:
    **XQ-A** = GPTQ + allocator (what V1/v2/1017/V3 are), **XQ-R** = XQ-A + the DeepSeek-specific GSQ-style refinement
    (this doc, being built).
  - Research done: `docs/ds4/research/DSV41_QUANT_METHOD_RESEARCH.md` (worker `dsv41-quant-research`, 667 lines,
    ranked: 1 GSQ refinement, 2 global-LP domain-weighted allocation, 3 per-expert bias/channel-scale healing,
    4 Hadamard on ternary (low odds), 5 VQ (needs kernels).  Negative: FP4-grid scale snapping +5.3% MSE, E8M0 scales
    +39%).
  - In flight: worker `gsq-spec` (flash) -> `docs/ds4/research/GSQ_ALGORITHM_SPEC.md` (exact GSQ from the IST-DASLab
    code, file:line cited, mapped onto our tensors).  Design (section 3) waits on it.
  - V3 (= XQ-A, V1 recipe on the 40% code / 30% agentic / 30% rest calibration) quant pass running (peer
    `xolotl-quant`), head gate ETA ~19:20.  It is the A/B baseline for XQ-R.
  - HF rename pending (needs the 1017 upload to finish, ~18:30): `tools/v41/hf-1017/restructure_v2.py` (dry run OK)
    moves the 1017 build into `v2/` as `V41-XQ-1017.gguf.part-*` and renames the repo to
    `IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored`.  V4.1 repo `...-GSQ-Uncensored` -> `...-XQ-Uncensored` too.

## 1. Facts (measured on this machine; do not re-derive)

- Experts are MXFP4: int8-packed E2M1 + one F8_E8M0 scale per 32 weights (checkpoint headers).  15,360 routed experts,
  543.6B weights; SwiGLU (w1 gate / w3 up [2304 x 5120], w2 down [5120 x 2304]), clamped (`lim`), output x router gate.
- Calibration (V3 mix, 32 x 8192 tokens, 1/4 held out): median 2,120 fit / 494 held-out routed tokens per expert;
  1.8% of experts below 8 held-out tokens.
- GPTQ/RTN error ratio by tier 0.66 / 0.73 / 0.74 (ternary / Q2_0 / Q3_K); spearman(RTN, GPTQ) 0.992-0.996; an
  allocation picked from RTN errors is as good as from GPTQ errors (-0.77% true cost) -> `V41_SCREEN_RTN=1` (commit
  d64bd5ce) makes calibration ~2.4 h instead of ~7.5 h.
- Allocation: at equal bytes less ternary is WORSE (ternary funds Q3_K); budget helps.  V3 map = V1 recipe
  (`--per-proj --edge-weight 2.0 --max-ternary-frac 0.5 --min-eval 8 --unseen-floor q2_0 --ternary-cap-at
  20-31,33-35=0.5 --budget-gib 142.38`), proxy 2.6502 vs 2.6589 for 140 GiB uncapped, on the V3 stats.
- 1017 build (XQ-A, 132.4 GiB experts): head gate ppl 6.934 vs fp teacher 6.572 (+5.5%), top-1 0.670 vs 0.687; code
  +9.1% worst.  V3 mix fp teacher: 6.360 / 0.706 (57,337 tokens).
- Quant pass per layer ~205 s: experts 132 s (GPU ~97% busy), load 19 s, attention 18 s, wrap-up 36 s (10.7 GB
  hidden-state hand-off to the next layer).
- Old prototype `v41_repair.refine` (fixed scales, one-step moves, rectified sigmoid, Adam, 200 steps, Hessian
  quadratic) made layer-13 ternary error WORSE: 24.9% -> 34.1% (docs/ds4/V41_1017_QUANT_PLAN.md).

## 2. Where XQ-R plugs in

`tools/v41/v41_quant.py`: `quant(Ws, H, tier, damp, do_gptq)` does scale search (`v41_gptq.search_scales`) -> GPTQ
(`v41_gptq_graph.gptq`, CUDA graph) -> returns `[(Q, S)]` per weight in the stored format.  Existing hook:
`V41_REFINE=1` calls `v41_repair.refine(Wst, Q, Sst, lo, hi, H, steps)` after GPTQ.  `Quant.expert(...)` has the
expert's fit/held-out tokens (`xe`, `ge`, `fit`), so an expert-output objective (not just per-matrix) is possible
there.  Output bytes/format are unchanged by refinement -> no engine or kernel change.

## 3. XQ-R design (to be written after GSQ_ALGORITHM_SPEC.md lands)

Candidate DeepSeek-specific elements (from the research paper section 3.2/4; each to be A/B'd):
1. Real GSQ parameterisation (categorical/mask+sign logits, annealed Gumbel-softmax, Lion, GPTQ warm start).
2. Joint scale learning under each format's storage rule (fp16 per 128 / per 64; Q3_K: decide).
3. Expert-output objective through the clamped SwiGLU with gate/up/down together, on the expert's routed fit tokens
   weighted by the router gate weight (vs the per-matrix Hessian quadratic).
4. Held-out early stopping per expert (keep GPTQ if refinement does not beat it on held-out: never worse by design).
5. Domain token weights (code / agentic up-weighted) - Mal's target use.
6. Cost control: low-rank Hessian or activation sub-batches so 15,360 experts fit in one night on a 4090.

## 4. Experiment plan (gates)

- E1 (3 layers 5/20/32 of the V3 map, all 384 experts): GPTQ control vs XQ-R variants; gate = >=5% lower held-out
  expert output error at identical tier + bytes on all three layers, and never worse per expert (fallback).
- E2: head gate on the V3 mix with XQ-R on all layers vs V3 (target ppl closer to 6.360 than V3's).
- E3: fixtures (`/home/mal/v41-work/quality-1017/cases.jsonl`, 112 cases recorded from the 1017 API 2026-10-19) +
  short Gauntlet + HumanEval subset vs V3.

## 5. How to resume (commands)

```bash
cd /home/mal/AI/Xolotl
relay up xolotl-engine; relay peers; relay inbox xolotl-engine --new
cat docs/ds4/XQ_PIPELINE.md            # this file, section 0 first
cat docs/ds4/research/GSQ_ALGORITHM_SPEC.md docs/ds4/research/DSV41_QUANT_METHOD_RESEARCH.md
ls /home/mal/v41-work/quant-v3* /home/mal/v41-work/calib-1017-v3/stats | head
```
Rules: commit named files only; never push `xolotl`; publish via `xolotl-publish` and read its diff; no deletions
under /home/mal/v41-work or the model drives without asking (path + size + why); GPU via the relay with
`xolotl-quant`; Mal OKs outward text.
