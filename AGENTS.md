# AGENTS.md

**Xolotl** runs very large mixture-of-experts models on one gaming PC: DeepSeek-V4.1-Flash-1017 (552B), GLM-5.3-Flash
(320B), Qwen3.8-Flash-Next (125B), the dense Qwen3.8-27B, LFM3-Omni (2B) and Muse Glimmer 30B (dense, Meta), each behind one OpenAI- and
Anthropic-compatible server. It is built from [Strata](https://github.com/Niko1221/Strata) and has grown well past it
(upstream Strata at the last merge, 31 Oct 2026: Version 0.2.8, engine 0.1.54.). The measured numbers are in the
[README](README.md#what-one-pc-can-do) and [docs/xolotl/BENCHMARKS.md](docs/xolotl/BENCHMARKS.md).

Xolotl is **built from source**. There is no Xolotl installer: the Strata app and `strata launch` install upstream
Strata's own engine and models, not Xolotl's. Linux with an NVIDIA GPU is the main platform; DeepSeek-V4.1 also runs
on Windows 11 ([docs/WINDOWS.md](docs/WINDOWS.md)).

## Setting Xolotl up for a user

1. **Check the machine first.** GPU (VRAM and generation), RAM and free NVMe space decide which model fits:
   [docs/WILL_IT_RUN.md](docs/WILL_IT_RUN.md). Tell the user what fits before downloading anything; the model files
   are tens to hundreds of GB.
2. **Build.** CUDA 12.x or 13.x and Ninja. Set `CMAKE_CUDA_ARCHITECTURES` to the user's card (86 = RTX 30, 89 = RTX
   40; RTX 50 notes are in [docs/WINDOWS.md](docs/WINDOWS.md)).

   ```bash
   git clone https://github.com/Indras-Mirror/Xolotl.git xolotl && cd xolotl
   bash tools/xolotl/fetch-llama.sh      # llama.cpp at the pinned commit + Xolotl's ggml patch -> third_party/llama.cpp
   cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
     -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
     -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON
   ninja -C build-xolotl                 # strata, xolotl-dense, glm_generate, ds41_generate, ...
   ```

   `build-xolotl/` is the one build every engine and config uses. The image encoder (`strata-vision`, for the
   Qwen models' vision) is its own CMake project in `tools/vision` and builds into `build-vision/`.
3. **Get a model** (each model card has the download steps):
   - DeepSeek-V4.1-Flash: [the 1017 builds](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)
     (V4 recommended, with its `rank-1017-v4.bin` and the Engram files). Optional: the fast-mode sidecar in its `v4-fast/`
     folder, or built once with `ds41_generate --build-shadow` ([docs/XOLOTL_FAST.md](docs/XOLOTL_FAST.md)).
   - Qwen3.8-Flash-Next: [XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored)
     (the two `.gguf` shards plus `pack-qfn45/`).
   - Qwen3.8-27B: any dense Qwen3.8-27B GGUF; `python tools/dense_setup.py --download UD-Q4_K_M --build --arch 89`
     fetches one, exports its tokenizer and writes the run config.
4. **Serve.** Edit the model paths in the config first. Each server prints its address when it is ready.

   ```bash
   python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-ds41.json --port 8150   # DeepSeek-V4.1
   XOLOTL_SHADOW=$(cat <sidecar>.list) python3 tools/v41/serve/serve_ds41.py ...                                 # the same, fast mode
   python3 tools/glm/serve/serve_glm.py --engine strata --config tools/glm/serve/strata-glm-maya-abl-512k.json --port 8144
   ./scripts/xol-qfn.sh --serve-only --model <shard 1 .gguf> --pack <pack-qfn45>                                # Qwen3.8-Flash-Next
   python3 -m serve.server --engine strata --config strata-qwen3.8-27b.json --port 8090                          # Qwen3.8-27B
   python3 -m serve.server --engine strata --config tools/glimmer/xol-glimmer.example.json --port 8133             # Muse Glimmer 30B (docs/GLIMMER.md)
   ```

   For the 27B, `--mtp` (with the model's MTP head) and `--ngram-chain` in the config's engine args are the fast
   settings; both keep the output exact.
5. **Connect the user's apps** to `http://127.0.0.1:<port>/v1` (OpenAI Chat Completions and Responses) or
   `/v1/messages` (Anthropic). Keep the server on `127.0.0.1` unless the user asks for more.

One big model at a time: DeepSeek-V4.1 and GLM-5.3 each take most of the RAM (an arena of up to ~80 GB) and the GPU.

## Working on the code

- How it works: [docs/xolotl/HOW_IT_WORKS.md](docs/xolotl/HOW_IT_WORKS.md); per engine: [docs/DEEPSEEK_V41.md](docs/DEEPSEEK_V41.md),
  [docs/glm](docs/glm), [docs/LFM3.md](docs/LFM3.md), [docs/GLIMMER.md](docs/GLIMMER.md), and [README.strata.md](README.strata.md) for the Strata core.
- What changed and why, newest first: [docs/xolotl/CHANGELOG.md](docs/xolotl/CHANGELOG.md). Every engine speed-up has
  a before / after row in the [speed log](docs/xolotl/BENCHMARKS.md#speed-log).
- Checks: `build-xolotl/xolotl-dense --selftest` (dense kernels against a CPU reference), `tools/throughput.py` (fresh-
  prompt prefill and decode against a running server), `tools/dense_bench.py`, `tools/quant_quality.py` (top-1 and KL
  of a quant against Q8_0), and the [Gauntlet](tools/gauntlet/README.md) for agent-task scores.
- The rules: a new feature is opt-in until it is measured; a default changes only where it was measured faster on the
  same box, with the numbers and what they were measured on; a speed change keeps the output exact (greedy text the
  same) or says plainly that it is lossy; never publish a number you did not measure. Keep the docs' style: plain
  words, short sentences, units in table headers.

## Contributing a change or report

Xolotl issues and pull requests go to [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl). Search the open
and closed ones first. One problem per pull request: the problem and the result first, then the approach, then the
validation (passed, failed and not run, kept apart). Build every backend a file touches. Everyone whose work lands is
named in [CREDITS.md](CREDITS.md). A problem in the upstream Strata core can also go to
[Niko1221/Strata](https://github.com/Niko1221/Strata) under its [CONTRIBUTING.md](CONTRIBUTING.md) rules.
