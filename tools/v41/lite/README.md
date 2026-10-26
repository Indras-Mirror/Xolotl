# HXQ3K-lite: Q3_K + GPTQ Q2_0 routed experts

CPU calibration, size allocation and GGUF writing for DeepSeek-V4.1-Flash-1017. The published
[heretic lite190u model](https://huggingface.co/NektoDron/DeepSeek-V4.1-Flash-1017-heretic-HXQ3K-lite-GGUF)
uses 190 GiB of experts (214.1 GB including the dense tensors), against 217.5 GiB of experts in HXQ3K.
The dense tensors and retained Q3_K experts are byte-identical to HXQ3K. The output uses Xolotl's mixed expert
layout and private Q2_0 grid; it needs Xolotl, rather than a general GGUF runner.

## Method

`lite_prep.py` prepares 96 sequences of 2048 tokens from public antirez/ds4 prompts (fc80bd69),
IlyaGusev/ru_turbo_saiga, NeelNanda/pile-10k and Xolotl sources. Each domain holds out approximately one quarter of its sequences.
The corpus directory contains `prompts.jsonl`, `ru_turbo_saiga.jsonl`, and `pile10k.jsonl` in those sources' schemas.
No local conversation transcripts are read.

`lite_calib.py` runs the existing `v41_calib` teacher forward layer by layer on CPU. Each worker fits GPTQ Q2_0
on fit rows, reads the corresponding llama.cpp imatrix Q3_K bytes, and measures the four gate/up + down
combinations on held-out rows. Gate and up share a type; down can use another. The teacher stays in the native
FP4 reference path. `defer_teacher` moves its expert computation into the worker, then adds outputs in expert
order before the shared expert and hyper-connection update. Ordinary `v41_calib` callers keep their old path.

`lite_alloc.py --cost use` minimizes the expert's relative output error weighted by its share of routed tokens,
using a Lagrange multiplier search and a greedy pass over the remaining byte budget. Experts with too few held-out
rows use the layer median relative cost weighted by their own usage. Experts with energy per held-out row above
100 times the layer median are forced to Q3_K (96 in this release). Frequency alone does not choose the type.
The older `rel` and `energy` objectives remain available for comparison.

`lite_write.py` applies the dense-tensor recipe of `v41_gguf`, copies retained Q3_K bytes without requantizing,
and takes Q2_0 bytes from calibration. `lite_cmp.py` compares it with an HXQ3K reference through both type maps.
`lite_calib2.py` is a compatibility entry point for `lite_calib.py`.

## Inputs and commands

Use Python with NumPy, PyTorch (native FP4/FP8 dtypes), safetensors and transformers. Supply DeepSeek's reference
inference directory (`model.py`, `config.json`) and a verified heretic checkpoint. The existing source loader
requires a `<shard>.ok` marker after each shard has been checked. Do not create markers for unchecked files.

```bash
export V41_REF="$PWD/reference/inference"
export V41_SRC="$PWD/heretic-fp8"       # config, tokenizer, index and verified checkpoint shards
export V41_WORK="$PWD/lite-work"
export LITE_Q3K_GGUF="$PWD/heretic-Q3_K-exps.gguf"  # llama.cpp imatrix Q3_K stacked expert tensors

python tools/v41/lite/lite_prep.py --src corpus-src --n 96 --seq 2048
python tools/v41/lite/lite_calib.py init
python tools/v41/lite/lite_calib.py run --out lite-out --workers 24 --worker-threads 6 --threads 64 --moe-threads 16
python tools/v41/lite/lite_alloc.py --stats lite-out/stats --gib 190 --cost use --out lite190.json
python tools/v41/lite/lite_write.py --map lite190.json --q2 lite-out/q2 --out lite190.gguf
python tools/v41/lite/lite_cmp.py HXQ3K.gguf lite190.gguf
```

Attention and MoE phases have separate thread counts. Size/worker defaults target a large EPYC CPU; lower them
for other machines. Calibration is CPU-only. Stats plus the following layer's input (or stats) allow resume.
Do not reuse a work directory after changing the checkpoint, corpus or calibration settings.

Full checkpoint shards work without any Engram overrides. To use extracted tables and small copies of shards 47/48
instead, set all three variables below. The small files retain every tensor except `engram.embed.*`, under the
original tensor names; the extracted tables use those names too. For writing, the verified source loader still
expects the small shard copies under the original checkpoint filenames in `V41_SRC`, with their verification
markers; the table-header fallback uses `LITE_ENGRAM_L1/L14`. Extraction is documented in
[the XQ model card](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored#quick-start-4-steps).

```bash
export LITE_ENGRAM_SMALL_DIR="$PWD/engram-small"
export LITE_ENGRAM_L1="$PWD/engram-l1/engram-L1.safetensors"
export LITE_ENGRAM_L14="$PWD/engram-l14/engram-L14.safetensors"
```

## Checks

```bash
# CPU; the real-expert test requires the inputs above and retains its scratch output.
python tools/v41/lite/lite_selftest.py
python tools/v41/lite/lite_layer_selftest.py --tokens "$V41_WORK/tokens.npy" --out-dir layer-selftest
python tools/v41/lite/test_lite_alloc.py
python tools/v41/tests/test_v41_gptq.py
python tools/v41/tests/test_v41_formats.py
python tools/v41/tests/test_torch_kernels.py
```

The self-test compares a spawned worker's teacher output for a real expert against DeepSeek `Expert.forward`,
bitwise in bf16, and checks all four held-out errors are finite. A real layer-0 integration gate also requires identical residual and pre-mix states for normal/deferred forwards.
The allocation tests cover byte accounting,
protected experts, usage weighting, unseen experts and an impossible budget (which must fail, rather than loop).
The format test builds the existing C reference and needs the vendored ggml headers and a built ggml-base library.

On 112 ds4q cases with DeepSeek API top-20 reference probabilities: HXQ3K top-1 91.90%, NLL 0.363, KL 0.110;
lite190u 91.40%, 0.386, 0.132; XQ V4 88.95%, 0.451, 0.197. KL uses the API top-20 distribution renormalized to
its observed mass against the model's full log-softmax. Lite190u also scored GSM8K 99/100, tools 21/23 and
JBB harmful refusals 0/100 in these small local evaluations. These are observations on fixed sets, not general
benchmark claims. See the model card for memory and throughput measurements.
