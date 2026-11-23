# Inkling-2-Mini LoRA training on one 24 GB GPU (plan + status)

Branch `inkling-train`, worktree `~/AI/Xolotl/worktrees/inkling-train`. Scope report (op inventory, geometry, VRAM
arithmetic): `~/.quetza-data/conductor/reports/scope-ink-train.md` (22 Nov). This file is the living plan.

## Goal

Train adapters for `inkling2_loop` (Inkling-2-Mini, 24 physical layers, 48 slot visits, 128 experts top-8) on the
served XQ GGUF: frozen weights stay quantised exactly as served (Q8_0 attention, Q2_0/NVFP4/IQ3/IQ4 experts), F32
adapter masters, one RTX 4090. Targets after the first proof: a confidence fine-tune and a QuetzaCodetl
coding/harness adapter.

## References (read-only)

- `~/AI/refs/automodel` - NVIDIA NeMo Automodel, `nemo_automodel/components/models/inkling2/` - a full PyTorch
  Inkling-2 implementation. **Use: maths oracle** for per-component gradients on small shapes (fp32, CPU).
- `~/AI/refs/nokeep-unbound` - nokeep's v2 recipe (LoRA r16 / alpha 32 on q, k, v, o, gate; lr 1e-4 cosine; 2 epochs;
  max-len 8192; completion-only loss; self-distilled harmful rows + original-model anchors). **Use: first real job and
  pass/fail target** (their numbers: 80.8 / 83.3 % useful at Q8_0).

## Design (from the scope report, section 3)

- Training runs the reference graph (all fused `inkling_*` custom ops off; `XOLOTL_INKLING_FUSED=0`,
  `ATTN_CHUNKS=1`, `CTRL_REF=1`, see the report's env table).
- A `train` variant of `G`/`layer()`: conv without state (zero history + ssm_conv over the chunk), attention over the
  chunk's own K/V (no cache writes, host masks), controllers on base tensors, LoRA on base tensors.
- Per-slot VJP: forward all 48 slots saving each slot's inputs; head CE in 256-position chunks; reverse loop with one
  graph per slot (surrogate loss `sum(out * g_out)`), explicit gradients for every depth-bank block tensor read, and
  adapter gradients ADDED over the 3 loop visits. A single 48-slot graph cannot fit (~56 GB of attention at T=2048).
- Reuse from the QFN/Hy4 trainers (`tools/train/`, `tools/hy4/engine/`): G1 graph wrapper + shared gallocr, surrogate
  VJP, AdamW/clip/schedule, metrics, `lora_gguf.hpp`, `lora_loader_check.py`.

## Work items and status

| # | item | status |
|---|---|---|
| W2 | TANH / L2_NORM backward, FILL ignore-src | **DONE** e54d371d2 (grad gates CPU + CUDA, negative control) |
| W1 | `GGML_OP_MUL_MAT_ID_BACK` (dX through frozen experts), CPU + CUDA | **DONE** 23 Nov (gates below) |
| W3 | ggml patch hygiene (train tree == patch) | done with W2/W1 (patch regenerated from the train tree) |
| W4 | `train` variant of `G`/`layer()` | **DONE (WIP commit)** - `--check 64` bit-exact vs inference with f16 K/V |
| W5 | per-slot VJP trainer (`src/program/inkling_train.cpp`, `xolotl-inkling --train`) | written, compiles; `--fd` gate in progress (needed batched GET_ROWS_BACK + softmax mask grad) |
| W6 | gates: train graph == inference graph (T=64, rel err < 1e-3); per-component FD + NeMo oracle; directional FD over all adapters (eps 0.1); 20-step overfit loss drop; every adapter gets a non-zero grad | |
| W7 | adapter loading in `xolotl-inkling` | |
| W8 | throughput / memory tuning | |
| M1 | **milestone: reproduce nokeep v2 on our XQ** (data prep slice `ink-train-data`), compare useful % with their card | |
| W9 | prefix carry (windows starting at p0 > 0) | v2 |

## W1 - MUL_MAT_ID_BACK

`ggml_mul_mat_id_back(as, grad, ids, ne11)`: `dst[:, e % ne11, t] = sum_e as[ids[e,t]]^T grad[:, e, t]`, shape of b.
Experts are frozen (the backward asserts no gradient is asked for `as`); ids outside `[0, E)` are skipped.

- CPU: each thread owns a block-aligned K slice, dequantises its slice of every routed expert row once (`to_float`)
  and adds g-weighted copies into the routed columns - no barrier, no reduction.
- CUDA (`ggml-cuda/mmid-back.cu`): routing to the host (one sync, so the node disables CUDA graphs), per expert with
  routed columns: dequantise to F32, gather its gradient columns, one cuBLAS SGEMM, scatter-add (no atomics: one
  expert never sees a dst column twice).
- Gates: `test-backend-ops grad -o MUL_MAT_ID` (FD through the forward, F32 experts: quantised experts quantise b
  to Q8 on the CPU, which makes FD noise) and `test-backend-ops -o MUL_MAT_ID_BACK` (CUDA vs CPU for every expert
  type in the XQ file: Q2_0, NVFP4, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL + F32/F16/BF16/Q8_0/Q4_K/Q6_K/MXFP4; broadcast and
  per-slot b, one-hot routing, skipped ids, Inkling-sized: 128 experts, gate_up 2304 -> 2304, down 1152 -> 2304).

### W1 gate results (23 Nov)

- CPU grad (FD through the forward, F32 experts): 72/72 checks OK, 0 FAIL. Negative control (gradient scaled by
  0.5): MAA 0.333 on every case -> the gate discriminates. (Count the `SUM(...)` verdict lines, not the op lines.)
- CUDA vs CPU (`-o MUL_MAT_ID_BACK`, NMSE <= 1e-6): 60/60 OK - 13 types x {per-slot b, broadcast b, one-hot,
  skipped ids} + Inkling-sized gate_up (Q2_0, NVFP4, IQ3_XXS, IQ3_S, IQ4_XS) and down (Q2_0, NVFP4, IQ4_NL).
- CUDA grad mode fails FD at MAA 6e-4..1.2e-3 - and so does plain upstream `MUL_MAT` on CUDA (2.4e-4..4e-4): the
  CUDA forward is not exact enough for FD at 1e-4. The chain that proves the CUDA backward is CPU-backward == FD
  plus CUDA-backward == CPU-backward.
- Residual: no timing yet (`perf` mode only runs its own case list); measure inside a real step (W8).

### W4/W5 status (23 Nov evening) - see ~/.quetza-data/RESUME-2026-11-23-INKLING-TRAIN-W5.md

- `--check 64`: per-slot streams + logits **bit-exact** (0.000 rel err, top-1 64/64) vs the engine's forward when
  `XOLOTL_INKLING_TRAIN_KV16=1` (f16 K/V operands as the cache path); f32 K/V (the training default) differs by the
  engine's f16 attention rounding only (logits mean 4.3e-2, top-1 61/64).
- New ggml rules needed by the full graph: batched GET_ROWS_BACK (relative-bias gather, router weights; CPU loop +
  CUDA atomics kernel; grad cases pass CPU + CUDA) and the softmax MASK gradient (the relative bias is the mask:
  dm = softmax_back(dy) at scale 1, repeat_back over broadcast dims). New test `test_soft_max_grad` (output op MUL ->
  run with `-o MUL`): result pending at handoff.

## Rules

ggml changes go into the train tree (`/media/Crucial1TB/xolotl-train/llama.cpp`) first, then into
`tools/xolotl/ggml/xolotl-ggml.patch` (+ `tools/train/test-backend-ops-train.patch` for tests); never edit the
production third_party tree in place. New files must be `git add -N`-ed in the pin checkout before the patch is
regenerated (a plain `git diff` drops untracked files).
