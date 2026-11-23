# LoRA training in Xolotl (EXPERIMENTAL)

> **Status, 21 Nov 2026: experimental, and now proven end to end on one model.** A LoRA for Qwen3.8-Flash-Next
> (QFN, Swift 1.5 XQ 45 GB) trained on the RTX 4090 from the **served engine's own activations**, with the real
> next-token loss, lowers the loss on held-out windows it never saw: **1.4116 -> 1.3197 (-6.5 %)** in 36 steps (~6 min).
> It is research code: no one-command "fine-tune my model" yet, and the adapter is not served by Xolotl yet (llama.cpp
> loads it). Nothing here changes inference: the served engines, their defaults and their outputs are untouched.
>
> **23 Nov 2026: Inkling-2-Mini, the whole model.** A LoRA on all 24 layers of Inkling-2-Mini (XQ, 13.9 GB), trained
> through the full looped forward (48 slot visits) on the served quantized weights, on one RTX 4090: held-out loss
> **1.492 -> 1.243 after 100 steps (-16.7 %), 1.212 after 400 (-18.8 %)**. Xolotl serves the adapter itself
> (`xolotl-inkling --serve ... --lora adapter.gguf`). See [Inkling-2-Mini](#inkling-2-mini-the-whole-model-on-one-4090).

Goal: train LoRA adapters on the MoE models Xolotl already serves (Hy4-flash, Qwen3.8-Flash-Next / QFN, later
DeepSeek-V4.1) on one PC - an RTX 4090 24 GB, ~110 GB RAM, NVMe - with the base weights **frozen and quantized exactly
as served** (Q8_0, the XQ mixes), streamed through the same VRAM / RAM / NVMe expert tiers inference uses.

The full plan, every gate's numbers and the dated status log: [docs/train/LORA_PLAN.md](train/LORA_PLAN.md).
QFN's layer-by-layer op map for the port: [docs/train/QFN_LAYER_MAP.md](train/QFN_LAYER_MAP.md).

## How it works: a LoRA for a 125B MoE on one gaming PC

The model is far too big to train whole on one card. Xolotl already runs it, so the trick is to let the **engine** do
everything that does not need a gradient, and train only the top of the network.

1. **Data.** Your own agent transcripts (here: QuetzaCodetl / Claude Code sessions), rendered with the serve's own chat
   template and tokenizer, cut into ~2K-token windows. The loss covers only the assistant's turns (thinking, tool calls,
   the end-of-turn token) and only from strong producers. Our build: 15,917 windows, 15.2M loss tokens, behind the same
   20,135-token system prompt + tool schemas the served model really sees.
2. **Capture.** The served engine prefills prefix + window once per window and writes, from taps already in the
   production binary, the residual entering layer 44 (`STRATA_KVA_CAPTURE`), layer 47's sparse-attention selection
   (`STRATA_QSA_DUMP`) and the final residual (`STRATA_PREFILL_DUMP_R_ALL`). So the trainer starts from exactly the
   activations the served model computes, quantized experts, YaRN rope and all. The shared prefix is stored once.
3. **Train the top 4 layers on the GPU.** Rank-16 adapters on layers 44-47 (GDN: `attn_qkv`, `attn_gate`, `ssm_out`;
   attention: `attn_q/k/v/output`), base weights frozen and quantized as served (Q8_0 dense, IQ4/Q8 experts, ~6 GB
   in VRAM). Per step: the prefix's layer state is computed once and carried in (GDN state + conv tail, K/V); a forward
   pass 44 -> 47 records each layer's input and MoE routing; the head gives the loss gradient; small per-layer backward
   graphs walk down 47 -> 44; AdamW runs on the host. Layer 47 replays the engine's own sparse selection, so its
   attention is the served one. Peak ~23 GB on a 24 GB card at 3K tokens, ~7-20 s per window.
4. **Out.** An adapter in llama.cpp's LoRA GGUF format, a metrics log, and a self-contained loss-graph page.

Every piece was checked against the served model before training: the C++ forward reproduces the engine on a real
21,686-position window (loss 1.7798 vs the engine's 1.7795), the GPU forward agrees (1.7780), the prefix carry is exact.

### Run it (QFN)

```bash
export QFN_NATIVE=<Swift1.5-QFN-XQ-45GB.gguf> QFN_PACK=<pack-qfn45>
# 1. the request your harness really sends (once): a fake /v1/messages endpoint, run your agent against it
python3 tools/train/harness_capture_server.py req.json 8399
# 2. windows from your transcripts (chats.jsonl: one conversation per line, messages + tool calls)
.venv/bin/python tools/train/harness_build.py --chats chats.jsonl --req req.json --out data/v1
# 3. capture (needs the GPU; ~70-140 s per window on the slow path)
.venv/bin/python tools/train/harness_capture.py data/v1 data/cap --n 40
# 4. train (build: see LORA_PLAN.md; QFN_EVAL_N windows held out)
QFN_EVAL_N=2 QFN_EVAL_EVERY=12 qfn_harness_train "$QFN_NATIVE" data/cap my-run 300
python3 tools/train/lossplot.py runs --png runs/loss.png      # runs/index.html: live loss graphs
python3 tools/train/lora_loader_check.py "$QFN_NATIVE" runs/my-run/adapter.gguf
```

### Results so far (21 Nov 2026, RTX 4090)

| run | train windows | steps | held-out loss (base -> end) | time |
| --- | --- | --- | --- | --- |
| smoke-r16 (1 window, overfit check) | 1 | 20 | 1.778 -> 1.633 (best) -> 1.664 | ~7 min |
| proof-r16 | 12 (12,917 loss tokens) | 36 | **1.4116 -> 1.3197 (-6.5 %)** | ~6 min |

Both adapters pass llama.cpp's LoRA loader rules. Not yet measured: the adapter on real agent tasks, a long run on
the full set.

### Option: fast capture through `--serve` (experimental, does NOT match yet)

The slow capture starts one engine per window and re-reads the 20K prefix each time (~35 s load + ~25 s prefill for
~1.5K new tokens). `tools/train/harness_capture_serve.py` loads one `strata --serve` engine instead; the serve's root
checkpoint sits exactly at the prefix end, so each window reads only its own tokens: **2.6-3.6 s per window instead
of 66-137 s** (measured, 4 windows). It needs the opt-in engine flag `STRATA_KVA_CAPTURE_PER_REQ=1` (one capture
directory per read, from that read's first position; off = the engine is unchanged).

**Do not train on it yet.** Compared with the slow capture (`tools/train/harness_capture_compare.py`), the prefix rows
are bit-identical but every window row differs: median relative error 0.12-0.13 per row (worst 0.22) at layers 44-47
and the final residual, and layer 47 selects different cells for every query. That holds even for a request read cold
from token 0, so it is not the checkpoint restore: the serve path reads the window differently (it splits the read at
turn boundaries). Which of the two the served model really computes is the open question; the slow path is the one
the gates above verified.

## Inkling-2-Mini: the whole model on one 4090

Inkling-2-Mini (24 physical layers run as 48 slot visits through a 3-pass loop core, 128 experts / top-8) fits in
VRAM (XQ 13.9 GB), so here the trainer runs **the entire model** every step, with no capture pass.

1. **Data.** `tools/train/harness_build_inkling.py` renders your agent transcripts with Inkling's own chat template
   and tokenizer (reasoning kept in history, as the serve renders it) into windows of up to 3,200 tokens. The loss
   covers the assistant's turns only. `--heldout-only` renders the held-out conversations as an eval set;
   `check_harness_inkling.py` gates both (prefix-free ids, decode round-trip, mask = assistant turns, no leak).
2. **Train.** `xolotl-inkling --train <model.gguf> --data windows.jsonl --eval eval.jsonl ...`
   (`src/program/inkling_train.cpp`). Rank-16 adapters on `attn_q / k / v / output / gate / rel` of every layer
   (144 adapters, 9.78M parameters); the base weights stay frozen and quantized exactly as served (XQ experts, Q6_K
   head). Per step: the slot visits run forward one by one, the head gives the cross-entropy gradient (chunked over the
   real vocabulary, softcap included), then each visit's graph runs backward in reverse order (a vector-Jacobian
   product per visit; the loop's message bank and depth mixing are carried as a stream DAG); AdamW on the host.
   The backward through the frozen 4-bit experts is a new ggml op, `GGML_OP_MUL_MAT_ID_BACK` (CPU + CUDA).
3. **Checkpoints.** `--save P --save-every N` keeps `P-stepNNNNNN.gguf` at every checkpoint (check any of them with
   `--eval`, or serve it), `P` is always the latest adapter and `P.state` its AdamW state. `--resume P` continues the
   run from that step with the same data order; stopping the unit (SIGTERM) or Ctrl-C writes a checkpoint first.
4. **Serve.** `xolotl-inkling --serve --native <model.gguf> --lora adapter.gguf [--lora-scale S]` applies the adapter
   in the same graph hook the trainer uses (`G::mm`).
5. **Prompt carry.** A harness window really sits behind a long system prompt + tool schemas (23,061 tokens for ours),
   and an agent turn behind its whole conversation. `--prefix prompt.json` (one context for every window) or a
   per-window `"prefix"` / `"context"` runs that context through the inference engine once (no gradient, base
   weights); each slot visit of the window then attends to the slot's cached K/V (global slots: every prefix position;
   sliding slots: the last 1,023) and starts its convolutions from the slot's conv state. The engine state right after
   the shared prompt is snapshotted (conv states + sliding rings, ~0.3-0.5 GB), so every window restores it instead of
   re-reading 23K tokens, and windows whose context extends the previous one prefill only the new tokens.
   `tools/train/harness_subwindows.py` cuts long windows into pieces that fit behind the prompt
   (T = budget / (prompt + context)), each carrying its earlier tokens as `"context"`.

Gates, all before the first real run (numbers in [docs/train/INKLING_TRAIN_PLAN.md](train/INKLING_TRAIN_PLAN.md)):
the training graph reproduces the engine's forward **bit for bit** (every slot and all logits, with f16 K/V like the
cache); every new backward rule passes finite differences on CPU and CUDA vs CPU (the softmax-mask rule against a
double-precision reference, `tools/train/smgrad_ref.cpp`); a directional finite difference along the gradient agrees
within 3.5 % for every adapter type (0.36 % overall) with MoE routing held fixed, as the gradient assumes; a 20-step
overfit on one real window takes the loss from 1.227 to 0.0002. With a carried prefix (`--check T --check-prefix P`) the
training graph matches the engine bit for bit at every prefix length tried (0, 17, 300, 1,023, 1,024, 1,500 and the full
23,061-token harness prompt; top-1 64/64), and a window evaluated twice around another one behind the prompt gives the
identical loss with 0 tokens re-prefilled. Getting there taught one thing worth knowing: Inkling's attention product
accumulates in f16, so the result depends on the cell layout (one extra masked cell moved the first layer by 2.9e-3 and
the logits by ~3e-2 after 48 slot visits). The trainer therefore pads its attention cells to the engine's 256-multiple,
and the check gate reads the sliding slots in the engine's own order.

```bash
# windows from your transcripts + the request your harness sends (see "Run it (QFN)" step 1 for the capture server)
INKLING_PACK=<pack dir> python3 tools/train/harness_build_inkling.py --chats chats.jsonl --req req.json --out data/
INKLING_PACK=<pack dir> python3 tools/train/harness_build_inkling.py --chats chats.jsonl --req req.json --out eval/ --heldout-only
# train (T <= 2048 fits 24 GB; eval every 100 steps)
xolotl-inkling --train Inkling-2-Mini-XQ.gguf --data data/windows.jsonl --eval eval/windows.jsonl --eval-n 32 \
    --eval-every 100 --steps 400 --warmup 30 --lr 1e-4 --max-t 2048 --save runs/r1/adapter.gguf --save-every 100
# behind the harness prompt: sub-windows that fit (prompt + context) x T, then train with the prompt carried
python3 tools/train/harness_subwindows.py --windows data/windows.jsonl --prefix data/prefix.json --out data-r2/
xolotl-inkling --train Inkling-2-Mini-XQ.gguf --data data-r2/windows.jsonl --eval eval-r2/windows.jsonl --eval-n 91 \
    --eval-every 100 --steps 300 --max-t 1024 --prefix data/prefix.json --max-prefix 30000 --save runs/r2/adapter.gguf --save-every 100
# continue it later (weights + optimizer state + step)
xolotl-inkling --train Inkling-2-Mini-XQ.gguf --data data/windows.jsonl ... --steps 800 --save runs/r1/adapter.gguf --resume runs/r1/adapter.gguf
# serve it
xolotl-inkling --serve --native Inkling-2-Mini-XQ.gguf --lora runs/r1/adapter.gguf
```

### Results (23 Nov 2026, RTX 4090)

| run | data | steps | held-out loss (base -> end) | time |
| --- | --- | --- | --- | --- |
| overfit gate | 1 window (T 512) | 20 | 1.227 -> 0.0002 (train window) | 1 min |
| harness r1 | 15,227 windows (13.8M loss tokens), T <= 2048, no tool-schema prefix | 100 | **1.492 -> 1.243 (-16.7 %)** | ~12 min |
| harness r1 (same run) | | 400 | **1.492 -> 1.212 (-18.8 %)** | ~50 min |
| harness r2 (24 Nov) | 22,921 coding-weighted sub-windows (5.2M loss tokens) **behind the real 23K-token tool-schema prompt** | 300 | **1.382 -> 1.166 (-15.6 %)** | 87 min |

r1 held-out = 32 windows from 14 conversations the builder kept out of training; r2 held-out = 91 pieces of 24 windows
from the same conversations, each scored behind the full prompt as served (its base loss is lower because the prompt
helps). Step time 3.0 / 5.7 / 14 s at T = 512 / 1024 / 2048 without a prefix (VRAM 15.8 / 15.9 / 17.4 GB); behind the
23K prompt ~5-14 s (T ~ 300, 21.5 GB peak).

**Behaviour, honestly:** lower loss on our transcripts has not yet shown up as better coding. LiveCodeBench, 20 problems
(4 easy + 16 medium), greedy, 16K thinking: base 15/20, with the r2 adapter 13/20 (4 base-only passes, 2 adapter-only;
inside the noise of 20 problems), and the adapter costs ~20 % decode speed through the per-weight LoRA path. The pipeline
(data, prompt carry, training, serving) is proven; which data actually makes the model a better agent is the open part.

### What we learned about the data (24 Nov)

Two adapters, two honest negatives on behaviour, both with clear loss gains:

| adapter | data | held-out loss | behaviour |
| --- | --- | --- | --- |
| harness r2 | 22.9K sub-windows of our own agent transcripts, coding-weighted, behind the 23K prompt | -15.6 % | LiveCodeBench-20: 13/20 vs base 15/20 (inside the noise) |
| task adapter | 6 successful runs of a stronger local model (a 45 GB MoE teacher) on two tasks of a private multi-step agent benchmark, 1,461 windows behind each run's real conversation (up to 65K tokens) | **-28.4 %** on the held-out runs | worse on both tasks it was trained on (2/12 and 0/12 vs base 4/12 and 4/12); one run went 4x longer before stopping |

What that says, and what we would do next:

- **The trainer is not the bottleneck; the data is.** Teacher-forced imitation lowers loss on the teacher's text fast,
  but a few trajectories teach the surface of those runs, and small deviations at inference compound into loops.
  Six runs is far too few: plan on tens to hundreds of successful trajectories per task family.
- **Train on whole trajectories.** The task adapter only saw the first <=65K tokens of each run and collapsed exactly
  where runs go past that. Long-context training (above) matters for agent data.
- **Select for success and for the behaviour you want.** Our transcripts mix planning, mistakes and recoveries;
  `harness_build_inkling.py --mask-errors` drops turns whose tool call failed (3 % of tool results in our corpus);
  coding gains probably need curated, verified coding trajectories rather than everything we have.
- **Adapter strength.** Rank 16, alpha 32 is a strong default; try `--lora-scale 0.5` at serve time and earlier
  checkpoints (`P-stepNNNNNN.gguf`) before concluding a dataset is bad.
- **Evaluate behaviour with several samples.** Single runs at temperature 1 swing by several points on a 12-check
  task; use 3+ samples per arm, or greedy where the task allows.

## What is in

| piece | where | gate |
| --- | --- | --- |
| OUT_PROD with a **quantized** weight on CUDA (dequant tile + cuBLAS): the `W^T dy` product of every frozen matrix | `tools/xolotl/ggml/xolotl-ggml.patch` (`ggml-cuda/out-prod.cu`) | test-backend-ops CUDA vs CPU 264 / 0, Q8_0, Q4_K, Q6_K, IQ2_XXS, IQ4_XS, MXFP4, Q2_K, F16 |
| Backward rules ggml lacked: SIGMOID, SWIGLU_CLAMP, CONCAT, CLAMP, GET_ROWS_BACK, broadcast DIV, CONT of a permuted grad, SOLVE_TRI (incl. upper / back substitution), TRI, CUMSUM, SSM_CONV, PAD | same patch (`ggml.c`) | finite differences on CPU and CUDA |
| Chunked gated delta net as a differentiable graph (QFN's linear attention) | `tools/train/gdn_train.hpp` | FD on every element + numpy reference, 1 and 2 chunks, pack geometry |
| MoE expert loop as a differentiable graph (replaces MUL_MAT_ID; the router gets its gradient) | `tools/train/moe_train.hpp` | FD + numpy reference at production shape (E512 / K10) |
| One complete QFN gated-delta-net layer, then hyper-connections, as training graphs | `tools/train/qfn_layer_train.hpp`, `test_qfn_*` | FD + reference forward; whole layer vs real captured data |
| Hy4-flash trainer: adapters in every dense graph, the layer as one graph (experts inside), CE head, AdamW, clip, save | `tools/hy4/engine/hy4_train.cpp` | train graph == served forward per layer; trained adapter reproduces the trained text at inference (`hy4_generate`, `HY4_LORA_LOAD`) |
| QFN trainers on real pack weights: one layer, then a 2-layer stack | `tools/train/qfn_train.cpp`, `qfn_train_stack.cpp` | 1 layer: loss 9.667 -> 0.008 in 400 steps; 2 layers / 6 adapters: 2.409 -> 0.0004 in 100 steps (CPU, ~2 min) |
| Harness data: real request capture, window builder (serve rendering, byte-offset loss mask) | `tools/train/harness_capture_server.py`, `harness_build.py` | rendered ids == llama-tokenize; prefix identical across windows |
| Capture from the served engine (no engine change) | `tools/train/harness_capture.py` | prefix rows bit-identical across runs |
| QFN top-4 stack forward + prefix carry + YaRN + layer-47 selection replay | `tools/train/qfn_topk.hpp`, `qfn_topk_gate.cpp`, `qfn_topk_long.cpp` | 25-token smoke: top-1 25/25; real 21,686-position window: loss 1.7798 vs engine 1.7795 |
| **GPU trainer** (per-layer backward graphs, shared allocator, host AdamW, metrics, adapter) | `tools/train/qfn_harness_train.cpp`, `metrics.hpp`, `lossplot.py` | step-0 loss = base (1.7780); held-out loss falls (table above) |
| CUDA fixes for training shapes: OUT_PROD with a cuBLAS-invalid view stride, REPEAT_BACK past the grid-z and grid-y limits | `tools/xolotl/ggml/xolotl-ggml.patch` | backward-only ops; the trainer runs where it crashed before |
| Inkling backward rules: `GGML_OP_MUL_MAT_ID_BACK` (activation gradient through frozen quantized experts), TANH, L2_NORM, FILL, batched GET_ROWS_BACK, the softmax **mask** gradient (Inkling's learned relative bias is the mask), broadcast SUB | same patch | FD on CPU (negative controls), CUDA vs CPU for every XQ expert type; softmax mask vs a double-precision reference |
| **Inkling-2-Mini trainer** (whole model, per-slot VJP, stream DAG, chunked CE head, AdamW, adapter GGUF, `--check` / `--fd` / `--overfit` gates) and **`--lora` at serve time** | `src/program/inkling_train.cpp`, `inkling_main.cpp` | train graph == engine forward bit for bit; held-out loss falls (table above) |
| **Prompt carry** for Inkling (engine prefill of the context, carried K/V + conv state, prompt snapshot, per-window `prefix` / `context`, engine-layout cell padding) and `harness_subwindows.py` | `src/program/inkling_train.cpp`, `inkling_main.cpp`, `tools/train/harness_subwindows.py` | `--check-prefix`: bit for bit up to a 23,061-token prefix; snapshot A-B-A identical; held-out loss falls behind the prompt (r2) |
| Adapter files in **llama.cpp's LoRA GGUF format** | `tools/train/lora_gguf.hpp`, `lora_bin_to_gguf.cpp` | llama.cpp itself loads the QFN adapter and generates (with a negative control); scale convention checked against llama.cpp's apply path |

The Hy4-flash engine (`tools/hy4/`, `-DHY4` targets, off unless you build them) ships in the same change: its
converter fixups, the dense half on CUDA with the experts on the DeepSeek-V4.1 tier, and its own plan
([docs/hy4/](hy4/)). `ds41_moe.cpp`'s per-token expert cap became `DS41_MOE_MAX_K` (default 8, so DeepSeek-V4.1 builds are
unchanged; Hy4 builds use 16).

## What is not done

- **A long run and a behavioural check.** The pipeline is proven on held-out loss; next: the full window set, then the
  adapter on real agent tasks. QFN: serving it in Xolotl needs a merged clone pack (the engine has no LoRA path for
  these tensors yet); llama.cpp can load it today. Inkling-2-Mini: served by Xolotl (`--lora`).
- **Inkling-2-Mini contexts past 65K cells**: the backward's REPEAT_BACK used to stop at 65,535 cells (CUDA grid-y);
  the patch now runs it in slabs (identical below 65K, where every run above used it), but no run past 65K has been
  made yet, and memory is the next wall (a 72-token window behind a 55K prefix peaks at 22.9 GB; the attention needs
  cell chunks or a smaller cache type there). Speed: a step is still slow behind a long prefix, and serving with
  `--lora` costs ~20 % decode speed.
- **Fast capture parity** (above): find why the serve path's window rows differ from the one-shot path.
- DeepSeek-V4.1 wiring (it shares the hyper-connection / MLA / indexer ops, so it follows QFN).
- LoRA on the routed experts themselves, full fine-tuning, more than one GPU: out of scope for v1.
- The tools under `tools/train/` are research drivers and reference scripts. The harness tools take the model from
  `QFN_NATIVE` / `QFN_PACK`; some older gate drivers still default to their author's paths: pass your own.

## Build

The trainers build against a ggml tree that carries the current `tools/xolotl/ggml/xolotl-ggml.patch`
(`bash tools/xolotl/fetch-llama.sh [DIR]`); the gate build recipes are in the status log of
[LORA_PLAN.md](train/LORA_PLAN.md).
