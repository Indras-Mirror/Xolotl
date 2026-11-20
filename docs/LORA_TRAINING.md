# LoRA training in Xolotl (EXPERIMENTAL)

> **Status, 21 Nov 2026: experimental, research code.** The backward kernels, the per-layer training graphs and the
> adapter format are built and gated. **Real fine-tuning is not proven yet**: every training run so far used a
> memorisation target (fit a layer's output to a fixed target), not real activations with a real next-token loss. Use it
> to read, build on and test; do not expect a working "fine-tune my model" command. Nothing here changes inference:
> the served engines, their defaults and their outputs are untouched.

Goal: train LoRA adapters on the MoE models Xolotl already serves (Hy4-flash, Qwen3.8-Flash-Next / QFN, later
DeepSeek-V4.1) on one PC - an RTX 4090 24 GB, ~110 GB RAM, NVMe - with the base weights **frozen and quantized exactly
as served** (Q8_0, the XQ mixes), streamed through the same VRAM / RAM / NVMe expert tiers inference uses.

The full plan, every gate's numbers and the dated status log: [docs/train/LORA_PLAN.md](train/LORA_PLAN.md).
QFN's layer-by-layer op map for the port: [docs/train/QFN_LAYER_MAP.md](train/QFN_LAYER_MAP.md).

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
| Adapter files in **llama.cpp's LoRA GGUF format** | `tools/train/lora_gguf.hpp`, `lora_bin_to_gguf.cpp` | llama.cpp itself loads the QFN adapter and generates (with a negative control); scale convention checked against llama.cpp's apply path |

The Hy4-flash engine (`tools/hy4/`, `-DHY4` targets, off unless you build them) ships in the same change: its
converter fixups, the dense half on CUDA with the experts on the DeepSeek-V4.1 tier, and its own plan
([docs/hy4/](hy4/)). `ds41_moe.cpp`'s per-token expert cap became `DS41_MOE_MAX_K` (default 8, so DeepSeek-V4.1 builds are
unchanged; Hy4 builds use 16).

## What is not done

- **A real fine-tune.** Next: train on real per-layer activations captured from the served model, with the real loss,
  and show the adapter changes the model's answers the intended way at inference.
- DeepSeek-V4.1 wiring (it shares the hyper-connection / MLA / indexer ops, so it follows QFN).
- LoRA on the routed experts themselves, full fine-tuning, more than one GPU: out of scope for v1.
- The tools under `tools/train/` are test drivers and reference scripts. Several take the paths of their author's
  machine as defaults; pass your own.

## Build

The trainers build against a ggml tree that carries the current `tools/xolotl/ggml/xolotl-ggml.patch`
(`bash tools/xolotl/fetch-llama.sh [DIR]`); the gate build recipes are in the status log of
[LORA_PLAN.md](train/LORA_PLAN.md).
