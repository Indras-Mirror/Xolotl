# LoRA training in Xolotl (EXPERIMENTAL)

> **Status, 21 Nov 2026: experimental, and now proven end to end on one model.** A LoRA for Qwen3.8-Flash-Next
> (QFN, Swift 1.5 XQ 45 GB) trained on the RTX 4090 from the **served engine's own activations**, with the real
> next-token loss, lowers the loss on held-out windows it never saw: **1.4116 -> 1.3197 (-6.5 %)** in 36 steps (~6 min).
> It is research code: no one-command "fine-tune my model" yet, and the adapter is not served by Xolotl yet (llama.cpp
> loads it). Nothing here changes inference: the served engines, their defaults and their outputs are untouched.

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
| CUDA fixes for training shapes: OUT_PROD with a cuBLAS-invalid view stride, REPEAT_BACK past the grid-z limit | `tools/xolotl/ggml/xolotl-ggml.patch` | backward-only ops; the trainer runs where it crashed before |
| Adapter files in **llama.cpp's LoRA GGUF format** | `tools/train/lora_gguf.hpp`, `lora_bin_to_gguf.cpp` | llama.cpp itself loads the QFN adapter and generates (with a negative control); scale convention checked against llama.cpp's apply path |

The Hy4-flash engine (`tools/hy4/`, `-DHY4` targets, off unless you build them) ships in the same change: its
converter fixups, the dense half on CUDA with the experts on the DeepSeek-V4.1 tier, and its own plan
([docs/hy4/](hy4/)). `ds41_moe.cpp`'s per-token expert cap became `DS41_MOE_MAX_K` (default 8, so DeepSeek-V4.1 builds are
unchanged; Hy4 builds use 16).

## What is not done

- **A long run and a behavioural check.** The pipeline is proven on held-out loss; next: the full window set, then the
  adapter on real agent tasks. Serving it in Xolotl needs a merged clone pack (the engine has no LoRA path for these
  tensors yet); llama.cpp can load it today.
- **Fast capture parity** (above): find why the serve path's window rows differ from the one-shot path.
- DeepSeek-V4.1 wiring (it shares the hyper-connection / MLA / indexer ops, so it follows QFN).
- LoRA on the routed experts themselves, full fine-tuning, more than one GPU: out of scope for v1.
- The tools under `tools/train/` are research drivers and reference scripts. The harness tools take the model from
  `QFN_NATIVE` / `QFN_PACK`; some older gate drivers still default to their author's paths: pass your own.

## Build

The trainers build against a ggml tree that carries the current `tools/xolotl/ggml/xolotl-ggml.patch`
(`bash tools/xolotl/fetch-llama.sh [DIR]`); the gate build recipes are in the status log of
[LORA_PLAN.md](train/LORA_PLAN.md).
