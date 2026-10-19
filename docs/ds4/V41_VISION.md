# V4.1 image input

`ds41_generate --serve --vision` accepts image embeddings through `GENI`.
`tools/v41/serve/serve_ds41.py` connects OpenAI and Anthropic image content blocks
to the resident encoder in `tools/v41/vision/ds41_vision.py`.

## Configuration

Add `--vision` to the engine's arguments and a `vision` section to the server
configuration. Paths below are examples; the Python interpreter needs PyTorch,
NumPy and Pillow. The reference directory must contain DeepSeek's V4.1
`vision.py`, `image_processor.py`, `model.py` and their imports.

```json
"vision": {
  "exe": "/path/to/xolotl/tools/v41/vision/ds41_vision.py",
  "python": "/path/to/venv/bin/python",
  "mmproj": "/path/to/V41-MTP-Vision.gguf",
  "model": "/path/to/V41-GSQ.gguf",
  "reference_dir": "/path/to/dsv41-inference-ref/inference",
  "threads": 8,
  "gpu": false
}
```

The shim starts the encoder script with `python`; `exe` retains the generic
server's configuration shape. This configuration encodes images on the CPU;
the engine can still run on CUDA. The encoder reads vision, aligner and span
delimiter weights from the sidecar. Its `model` argument is accepted for
protocol compatibility.

Without `--vision`, the dense engine builds the existing text graphs, without
the image mask or VL-routing operations. `GENI` then returns an error. The
server's existing default image expansion is kept behind an overridable
`expand_image_tokens` method; the DS41 shim supplies the V4.1 expansion.

## Engine processing

The request is:

```text
GENI <max_new> [temperature=... top_p=... top_k=... min_p=... seed=...] <embedding_file> <ids_csv>
```

The file contains one or more SVE1 records: five little-endian int32 values
`[0x31455653, n, nx, ny, n_embd]`, followed by `n * n_embd` float32 values.
Records follow prompt order. The reader checks the header, width, grid, row
count and payload length. Row counts split adjacent images inside a single
image-token run. `--image-token` defaults to 129264.

Rows replace token embeddings at their absolute prompt positions, including
across prefill chunks. All span positions use sequential 1-D RoPE positions.
They select `exp_probs_b_vl.bias`, suppress the Engram gate and stop Engram's
n-gram lookback. Image requests prefill fresh and do not create reusable
token-only snapshots. The following text request also starts fresh, because
two different pictures can have exactly the same prompt token IDs.

## Checks on 2026-10-18

Hardware: NVIDIA RTX 3090 (PCIe 4.0 x16), 125 GiB system RAM.
Engine: `build-xolotl/ds41_generate`, V41-GSQ GGUF, Engram layers 1/14 and the
deployed Q2_K DSpark sidecar. Encoder: CPU float32, 8 threads, PyTorch
2.13.0+cu130, Q8_0/F32 V41-MTP-Vision sidecar. API: `127.0.0.1:8150`,
262144 context, temperature 0.8, routing bias 0.8, thinking off for these checks.

| Request | Answer | Prompt tokens | Reused tokens | End-to-end time |
| --- | --- | ---: | ---: | ---: |
| Red square | Red square | 230 | 0 | 19.94 s |
| Blue circle, same prompt and image dimensions | Blue circle | 230 | 0 | 14.83 s |
| Both images, in order | Red square, then blue circle. | 450 | 0 | 8.68 s |
| Text after images: 7 + 8 | 15 | 18 | 0 | 3.18 s |

The two single-image requests had identical token sequences and different
image contents. The multi-image request used the cached encoder files.
These are correctness checks, not a throughput benchmark. `/v1/models`
reported `input_modalities: ["text", "image"]`.

A separate streamed request with two fresh 560x560 images returned
`1. Green triangle` and `2. Yellow circle` in order. HTTP headers arrived after
14.72 s, the first answer text after 23.73 s, and the request finished in
24.02 s on the same hardware and settings.

Other checks passed:

- CUDA build: `ninja -C build-xolotl ds41_generate`.
- Server template, tokenizer, image expansion, literal markers, combined
  record order, mismatch handling, OpenAI tools and Anthropic tools:
  `python tools/v41/serve/test_serve_ds41.py /path/to/tokenizer`.
- Generic encoder lifecycle: `python -m unittest serve.test_vision_busy serve.test_vision_elastic`
  (11 tests).
- Generic image markers and temporary-file cleanup:
  `python -m unittest serve.test_server.ImageMarkers serve.test_server.VisionTempFiles`
  (9 tests).
- A temporary CPU protocol harness exercised truncated headers/payloads,
  wrong widths/grids, excessive row counts and separated/adjacent image runs.
- Earlier encoder validation against the reference loaded with the same
  dequantized sidecar weights: three synthetic images had zero relative
  error, correct span counts and byte-identical ENC protocol output.

The original FP8 vision checkpoint shards were unavailable, so that independent
weight comparison was not run. Long image conversations, OCR quality, a GPU
image encoder and a CUDA byte-identity comparison were not tested.

Repeat the image-content check against a running server with:

```sh
python tools/v41/vision/check_api.py --url http://127.0.0.1:8150
```

## Vision-disabled text regression

Separate Release builds of upstream `f7c561b` and the PR's engine implementation
`4e843f3` were compared on the CPU using the real 40-layer V41-GSQ model and
Engram layers 1/14. Both used GCC 13.3.0, the same patched ggml checkout,
2 CPU threads, 4096 context, 8 prompt tokens, temperature 0.8 and seed 17.
`--vision` was absent. The repository had no published release/tag, so upstream
`main` at the PR base was used as the baseline.

The comparison passed with the fused fake-quant and HC-mix build options both
ON, and again with both OFF (the stock options). These are correctness checks;
the CPU runs do not measure CUDA byte identity or performance.

| Path | Compared data | Result |
| --- | --- | --- |
| Sequential, chunk size 1 | All prompt logits, last logits, 3 sampled token IDs | Byte-identical |
| Chunked, chunk size 4 | All prompt logits, last logits, 3 sampled token IDs | Byte-identical |
| Layer-major, batch 8 / chunk 4 | All prompt logits, last logits, 3 sampled token IDs | Byte-identical |
| Serve, batch 16 / chunk 4 | First prompt, regeneration and continuation logits, tokens and reuse protocol | Byte-identical |

Each one-shot prompt-logit dump contains 4,136,960 bytes; its SHA-256 in all
three paths is `87c76ccaa0993876d34d8522fb085aa02ed6a9bcc96299351ad63679ae0d3b36`.
Each last-logit dump contains 517,120 bytes, with SHA-256
`0f98cf9d18af6412694a371031a0a530e9b5e606a19704baca5881d3b4b8c5af`.
The serving check confirms `RESUME 8` on regeneration and continuation.

The repeatable check uses CPU inference only and saves a `results.json` with
the compared sizes and hashes:

```sh
python tools/v41/vision/check_text_regression.py \
  --baseline /path/to/base-cpu-build/ds41_generate \
  --candidate /path/to/pr-cpu-build/ds41_generate \
  --model /path/to/V41-GSQ.gguf \
  --engram /path/to/layers.1:/path/to/layers.14 \
  --out /path/to/results
```

### Backend build scope

CPU-only Release builds and the CUDA engine build passed. DS41's current
documented GPU target is Linux/NVIDIA (`docs/DEEPSEEK_V41.md`).
`tools/v41/engine/v41_engine.cmake` defines a CUDA tier and a CPU fallback, not
native HIP or SYCL DS41 GPU targets. The separate SYCL project returns before
the fork's DS41 targets are included. HIP/SYCL toolchains were unavailable;
no HIP/SYCL build or runtime support is claimed by this change.

## Current limits

Uncached images encode one at a time on the CPU. Repeated images use the
server's image-byte cache, but image requests still prefill the full prompt.
The generic server's per-image encoder timeout defaults to 300 seconds.

Image preparation completes before the HTTP response stream opens. There
are no per-image progress messages or streaming heartbeats during that stage;
the normal prompt-progress logs start afterward and exclude image encoding
time. Large or numerous images can therefore look idle before inference starts.
