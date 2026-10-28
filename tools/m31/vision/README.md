# Optional MiniMax vision helper

Xolotl has no `mtmd` target. `STRATA_M31_VISION=ON` enables mixed input in the
M31 engine; the encoder is a separate executable with its own llama.cpp/ggml ABI.
It never creates an LLM context. The HTTP shim starts it lazily, so an absent or
broken encoder leaves text requests available.

Build it in a separate directory from an **unmodified** llama.cpp PR #31192 at
`6b95a9908d2c7b242817a447708aec38e937e2c0`:

```sh
git clone https://github.com/ggml-org/llama.cpp.git llama-m31
git -C llama-m31 fetch origin refs/pull/31192/head
git -C llama-m31 checkout --detach 6b95a9908d2c7b242817a447708aec38e937e2c0
cmake -S tools/m31/vision -B build-m31-vision \
  -DM31_LLAMA_SOURCE_DIR="$PWD/llama-m31" -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-m31-vision --target m31_vision_adapter -j 8
```

Use `-DGGML_CUDA=OFF` for a CPU encoder. CUDA and CPU embeddings differ; use the
same backend and GGML environment on both sides of a numerical comparison.
Video decoding also needs `ffmpeg` in `PATH`. The helper supports both images
and video through mtmd's own media decoding, token boundaries and timestamps.
`video_fps` defaults to 1; llama.cpp's default is different, so pass it explicitly
to the reference server too.

The helper checks the model's `minimax-m3` architecture and the projector's
`clip` architecture, `minimax_m3` projector type and 6144 output dimension. These
checks catch a different model family; they cannot establish that two weight
files are from the same model revision. Use the matching mmproj supplied with
your GGUF revision, and retain both files' hashes in measurement receipts.

For a standalone encoder run, create `encode.json` with paths on your machine:

```json
{
  "model": "/path/to/model.gguf",
  "mmproj": "/path/to/mmproj.gguf",
  "backend": "cuda",
  "threads": 16,
  "prompt": "<__media__>Describe this picture.",
  "media": ["/path/to/picture.png"],
  "encode": true,
  "encoder_mode": "single",
  "add_special": false,
  "video_fps": 1,
  "timestamp_interval_ms": 5000
}
```

`prompt` must be the complete rendered model prompt, with mtmd's default media
marker `<__media__>` for each media item. Wrap the example in the model's chat
template for inference; the HTTP shim does this for API requests.

```sh
build-m31-vision/m31_vision_adapter encode.json new-output-directory
```

The result is `m31-vision-ipc-v1`: ordered text/image chunks, linear decoder
positions, serialized mtmd metadata and per-image float32 arrays. It contains no
LLM logits. The shim assembles all text, image and video-frame rows into one
GENI/BGENI request. The encoder may run once per video frame; the LLM receives
one mixed prefill rather than a separate prefill per frame.

Persistent mode is `m31_vision_adapter --worker config.json`. It emits one JSON
`{"ready":true,...}` line, then accepts JSON request lines with `prompt`, `media`
and a new `out` directory. Each request returns `{"ok":true,"manifest":...}` or
`{"ok":false,"error":...}`. Model, projector and backend stay fixed for the
worker lifetime. The embedding cache is bounded (256 MiB by default).

`serve/vision_encoder.py` applies separate finite startup and request deadlines.
The request deadline covers stdin backpressure and incomplete reply lines. EOF,
helper death, malformed JSON, missing executable/model/mmproj, startup failure,
timeout and cancellation close and reap the helper and raise
`vision unavailable: ...`. Stderr goes to `encoder.log`, avoiding unread-pipe
deadlocks. A later vision request may start a fresh helper; text does not wait
for it. Cancellation is polled at most every 100 ms while waiting on a pipe.
The Python transport is validated on Linux; Windows pipe behavior, HIP and SYCL
are unverified.

Run the model-free transport fault checks with:

```sh
python3 tools/m31/tests/test_vision_helper.py -v
ctest --test-dir build-m31 --output-on-failure -R m31_vision
```

The tests cover missing files, startup/reply timeout, partial lines, invalid
JSON, death during a request, a full stdin pipe, heavy stderr, cancellation and
successful recovery. Inference, logit parity and TTFT require the model and are
documented in `docs/MINIMAX_M31.md` with measurement receipts.
