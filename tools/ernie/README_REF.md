# ERNIE 5.5 Flash reference cross-check

Two independent implementations of the ERNIE 5.5 trunk, compared tensor by tensor:

* `ernie_ref.py` — a numpy float32 forward over the **FP8 safetensors** (128x128 block
  dequant), following the vLLM-ERNIE model code (which subclasses upstream vLLM 0.32
  `DeepseekV4Attention` / `DeepseekV4ScalingRotaryEmbedding`).
* `ernie_dump.cpp` — a tiny `libllama` driver that runs one `llama_decode()` with the
  llama.cpp ERNIE 5.5 PR (#31925) CPU build and writes every named graph tensor the
  PR's `cb()` emits (`l_last-N`, `q-N`, `kv-N`, `ffn_moe_topk-N`, ...).
* `ernie_compare.py` — matches the two dumps, prints relative L2 per tensor, and the
  router top-10 id/weight agreement.

## Running it

```bash
# 1. llama.cpp PR CPU build (CPU only; GGML_CUDA=OFF)
cmake -S ~/AI/Xolotl/third_party/llama.cpp-ernie55 -B ~/.cache/llama-pr-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.cache/llama-pr-cpu -j$(nproc)
g++ -O2 -std=c++17 tools/ernie/ernie_dump.cpp -o ~/.cache/llama-pr-cpu/bin/ernie_dump \
    -I ~/AI/Xolotl/third_party/llama.cpp-ernie55/include \
    -I ~/AI/Xolotl/third_party/llama.cpp-ernie55/ggml/include \
    -L ~/.cache/llama-pr-cpu/bin -Wl,-rpath,$HOME/.cache/llama-pr-cpu/bin \
    -lllama -lggml -lggml-base

# 2. the GGUF under test (layers 0-2; Q8_0 backbone, Q8_0 / F16 experts)
gguf=.../gguf-dev/ernie-L3-q8.gguf
python3 tools/ernie/ernie_gguf.py "$FP8_DIR" --layers 3 --experts q8_0 \
    --metadata-from "$BASE_GGUF" -o "$gguf"

# 3. dump + reference + compare (32 tokens)
TOK=$(seq -s, 30000 30031)
ernie_dump -m "$gguf" --tokens "$TOK" --dump /tmp/g32 --names \
  hc_init,l_last,hc_attn_pre,hc_ffn_pre,attn_norm,ffn_norm,qr,qr_norm,q_norm,q,kv,\
attn_hca,attn_derope,attn_wo_a,attn_gate,attn_out,ffn_out,ffn_shexp,ffn_routed_up,\
ffn_moe_probs,ffn_moe_probs_split,ffn_moe_topk,ffn_moe_weights
python3 tools/ernie/ernie_ref.py "$FP8_DIR" --layers 3 --tokens "$TOK" --out /tmp/ref32.npz
python3 tools/ernie/ernie_compare.py /tmp/ref32.npz /tmp/g32 --layers 3

# 300 tokens: the HCA compressor runs (two complete 128-token windows)
TOK=$(seq -s, 30000 30299)
ernie_dump ... --tokens "$TOK" --dump /tmp/g300 --names <same list>
python3 tools/ernie/ernie_ref.py "$FP8_DIR" --layers 3 --tokens "$TOK" --out /tmp/ref300.npz --vha llamacpp
python3 tools/ernie/ernie_compare.py /tmp/ref300.npz /tmp/g300 --layers 3
```

Token ids: the Preview vocab's low ids (specials / byte tokens) carry near-zero BF16
embeddings (`~1e-7`), and Q8_0 rounds those rows to exactly zero — the GGUF forward then
degenerates to all-zero q/kv and nothing is comparable. Use ids >= ~1500, which have
normal `~0.02` embeddings; the scripts above use 30000..30031 / 30000..30299.

`ernie_dump --no-flash-attn` switches the attention path (FlashAttention keeps the KV
cache in F16; the non-FA path accumulates KQ in BF16). FlashAttention is the closer
match to vLLM's kernels and is the default.

## What is compared

Per layer: `l_last-N` (the 4-stream mHC residual), `hc_attn_pre-N` / `hc_ffn_pre-N`
(the pre-mixed sublayer input), `attn_norm` / `ffn_norm`, `qr` (q-latent pre-norm),
`qr_norm`, `q` (post per-head QK-norm + rope), `kv` (post-norm + rope), `attn_hca`
(attention out, before the inverse rope), `attn_derope`, `attn_wo_a` (grouped output
before the gate), `attn_gate`, `attn_out`, `ffn_out`, the shared expert, the routed up
projection, the dense routing probs and the router top-10.

The router ids are reconstructed from `ffn_moe_probs` + `ffn_moe_weights`
(`w_j == probs[id_j]` exactly) because the small I32 top-k node's buffer is recycled by
the scheduler before the callback can read it reliably for every token; the directly
dumped ids are reported separately as a diagnostic.

## The VHA head mix: the PR and vLLM agree (corrected)

vLLM: `m = I + V U^T`, `out[h] = sum_g m[h, g] o[g]` (`VHAPostMix.forward` is `torch.matmul(m, o)` with
`o` = [T, H, D]; the TP path is `o + V (U^T o)`, the same map). The PR's converter writes the same `m`
(byte-identical), and `build_head_mix()`'s `ggml_mul_mat(mix, x)` on that row-major [h][g] tensor computes
`sum_g m[h, g] x[g]` too. An earlier version of this reference applied the mix with
`einsum("hg,thd->tgd")`, which sums over `h` (= `m^T o`); that made it look as if the PR were transposed.
Fixed to `einsum("hg,tgd->thd")`; `--vha llamacpp` is now an alias. The F16-oracle numbers in the
cross-check table were taken with `--vha llamacpp`, which was the correct `m o` all along.

The HCA rotary is *not* a difference: both implement the compress YaRN (base 160000,
factor 16, beta 32/1, original ctx 65536, mscale 1) with the truncated correction dims
15/25. Applying that rope to the PR's own `kv_norm-N` reproduces its `kv-N` to 1.6e-6,
which is what pinned the truncation down (`floor`/`ceil` of the correction dims).

Everything else (mHC with the ERNIE `1/(sqrt(mean)+eps)` pre-scale, Sinkhorn-20, the
per-head unweighted QK norm, the 128-wide gated compressor with APE, K=V, the attention
sink, the sigmoid output gate, the grouped `wo_a`/`wo_b`, the split-feature sigmoid
router with bias-for-selection-only + renormalise + per-expert scale, the 2048-d latent
MoE, the shared expert and SiTU) matches to F32 accumulator order.

## Unquantised oracle

`ernie_gguf.py --ftype f16 --experts f16` writes every matrix as F16. An E4M3 value
times a power-of-two `scale_inv` is exactly representable in F16, so that file carries
the FP8 weights without loss and lets the cross-check isolate the *math* from the Q8_0
weight rounding. Use it for the exactness gate; use the Q8_0 file for the realistic
"two quantised paths agree" numbers.
