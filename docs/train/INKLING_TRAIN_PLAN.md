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
| W5 | per-slot VJP trainer (`src/program/inkling_train.cpp`, `xolotl-inkling --train`) | **DONE 23 Nov** - `--fd` (frozen routing) + `--overfit` pass (below) |
| W6 | gates: train graph == inference graph (T=64, rel err < 1e-3); per-component FD + NeMo oracle; directional FD over all adapters (eps 0.1); 20-step overfit loss drop; every adapter gets a non-zero grad | |
| W7 | adapter loading in `xolotl-inkling` | **DONE 23 Nov** - `--lora` / `--lora-scale`; gate below |
| W8 | throughput / memory tuning | |
| M1 | **milestone: reproduce nokeep v2 on our XQ** (data prep slice `ink-train-data`), compare useful % with their card | |
| W9 | prefix carry (windows starting at p0 > 0) | **done 24 Nov**: bit for bit to a 23,061-token prefix, snapshot, r2 trained behind it |

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

### W5 gates (23 Nov night)

- Softmax mask rule: the test-backend-ops FD reference is too noisy (f32 SUM loss); the rule is proven against a
  double-precision analytic reference, `tools/train/smgrad_ref.cpp` (16/16 at < 1e-7, gradient ~0.5).
  `test_soft_max_grad` now runs at the measured floor (max_maa_err 1e-2): 16/16 CPU + CUDA. SOFT_MAX_BACK / REPEAT_BACK
  CUDA == CPU.
- New rule: broadcast SUB backward (repeat_back of the subtrahend's gradient); `test_sub_bcast_grad` 3/3.
- `--fd` redesigned: steps along the gradient (d = g_S/|g_S|, per target set) instead of a random direction (a random
  unit direction over 9.78 M params has <g,d> ~ |g|/sqrt(N), below the forward's noise). Routing is **frozen** for the
  gate (base forward's top-k ids replayed; `XOLOTL_INKLING_FD_FREE_ROUTING=1` to keep it live): with live routing, tiny
  steps flip near-tie expert choices and the loss jumps. Result at eps 0.1: all 0.36 %, q 1.7 %, k 2.1 %, v 3.5 %
  (0.27 % at 0.03), o 2.4 %, gate 2.0 %, rel 1.1 %; 0/144 zero-grad adapters. Smaller eps = q8/f16 rounding steps.
- Fused CUDA top-k MoE router vs unfused (`GGML_CUDA_DISABLE_FUSION=1` or `XOLOTL_INKLING_ROUTER_REF=1`): loss 2.2912
  vs 2.2719 on w64. Slot 0 (identical inputs) agrees to 1.6e-7; the gap grows over the 48 visits (median 3e-4 at slot 1,
  2.6 % at slot 47) = ulp differences amplified by routing flips + rounding, same size as the f16-vs-f32 K/V gap.
  The kernel is correct (new TOPK_MOE cases at Inkling's shape, 128 experts / top-8 / T 64 + 512: 16/16).
  `XOLOTL_INKLING_DUMP_STREAMS=<file>` dumps every slot's output stream for such comparisons.
- `--overfit 20 --lr 1e-3` on a real harness window (`gate/h512.jsonl`): loss 1.227 -> 0.0002 (w512 is repeated text,
  base loss 0.009 - useless for this gate). Step time / VRAM peak (frozen weights 13.9 GB): T=512 3.0 s / 15.8 GB,
  T=1024 5.7 s / 15.9 GB, T=2048 14.0 s / 17.4 GB (~150-180 tok/s).

### First real run + W7 (23 Nov night)

- r1 (`--data train-t2048.jsonl`, 15,227 windows, no prefix; eval = 32 windows from the 14 held-out conversations,
  `harness_build_inkling.py --heldout-only`): held-out 1.492 -> 1.243 (step 100) -> 1.223 -> 1.215 -> 1.212 (step 400,
  -18.8 %); stopped at 400 (proof in; Mal: shorter proofs).
- W7 `--lora`: the loader attaches each `.lora_a/.lora_b` pair to its model tensor through G::mm (the trainer's hook);
  the T=1 stacked q/rel/gate(/k/v) mat-vec steps aside with an adapter, the fused-attention gate projections go through
  mm(). Gate (`w7_gate.sh`, step-400 adapter): `--check 64 --init-from` (adapter in both trainer and engine, f16 K/V)
  0.000 / 64 of 64; `--dump --lora` (fused serve paths) vs trainer logits 4.1e-2 mean (base: 4.3e-2 = the fused-path
  gap), adapter effect 1.12 (top-1 changes 38 / 64).
- Checkpoints (Mal): `--save P --save-every N` -> `P-stepNNNNNN.gguf` kept + `P` latest + `P.state` (AdamW moments,
  step); `--resume P`; SIGTERM/SIGINT checkpoint then exit 0.

### W9 prefix carry + r2 (24 Nov night)
- Gate `--check 64 --check-prefix P` (KV16): first version drifted 2.9e-2 (P=1500) / 6.1e-2 (P=23061). Bisection: exact for
  P <= 16, then exact only when prefix + T was a multiple of 4 (< 256 cells) / 64 (more) and P < 1024. Cause: the engine
  attends over a 256-multiple of cells and its V x probs mul_mat has no F32 precision flag (f16 accumulation in cuBLAS), so
  results depend on the cell count and on where masked cells sit (one masked head cell: 2.9e-3 at slot 0). Not state.
- Fix: G::tpad pads the train attention to the 256-multiple; the gate's sliding slots read every prefix cell in the
  engine's order (swa_full; ring sized for prefix + window, 1024-token prefill chunks, one-softmax budget). Training keeps
  the 1023 tail (tail vs full differs at the same level as a pure precision change, 2.7e-2 vs 2.9e-2, under the engine's
  own fused/unfused 4.3e-2). Exact at P = 0, 16, 17, 20, 300, 1023, 1024, 1500, 23061. Snapshot A-B-A: identical loss,
  0 tokens re-prefilled.
- Limits: REPEAT_BACK grid-y (65,535 cells) - now slabbed, untested past 65K; memory (55K prefix x 72 tokens = 22.9 GB).
  --fd / --overfit with a 55K prefix at T = 145 run out of memory; at T = 72 an overfit converges (0.386 -> 0.0000, 2.1 s/step).
- r2: 300 steps, held-out 1.3821 -> 1.1802 -> 1.1688 -> 1.1659 (-15.6 %), 87 min. LCB-20: base 15/20, r2 13/20 (4 vs 2 flips).
- Task adapter (private benchmark, 1,461 windows, 1460 steps): held-out 0.3987 -> 0.2853, benchmark worse (see LORA_TRAINING.md).

## Rules

ggml changes go into the train tree (`/media/Crucial1TB/xolotl-train/llama.cpp`) first, then into
`tools/xolotl/ggml/xolotl-ggml.patch` (+ `tools/train/test-backend-ops-train.patch` for tests); never edit the
production third_party tree in place. New files must be `git add -N`-ed in the pin checkout before the patch is
regenerated (a plain `git diff` drops untracked files).
