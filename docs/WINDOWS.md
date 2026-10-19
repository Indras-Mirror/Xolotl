# DeepSeek-V4.1-Flash on Windows (native, no WSL)

The DeepSeek-V4.1 engine (`tools/v41/engine`) builds and serves natively on Windows 11 with MSVC + CUDA.

Tested on one machine:
- RTX 3090 24 GB, Ryzen 7 5800X, 128 GB DDR4-3200, Samsung 990 EVO Plus NVMe
- Windows 11 Pro, driver 616.92, CUDA 12.8, Visual Studio 2022 Build Tools

Other cards should work with `CUDA_ARCH` set (89 = RTX 40, 120 = RTX 50) but are untested. Reports are welcome.

## What you need

| | |
| --- | --- |
| GPU | NVIDIA, 24 GB, driver new enough for CUDA 12.8 |
| RAM | 128 GB recommended for the 72 GiB arena in the example configs; lower `--arena-gib` on smaller machines |
| Disk | ~370 GB on an NVMe (NTFS): the model (148-163 GB) + the official Engram shards (203 GB), plus room for the parts while joining |
| Tools | Visual Studio 2022 (Desktop development with C++), CUDA Toolkit 12.8+, Git for Windows (Git Bash), Python 3.12 |

## 1. Code and llama.cpp

```bat
git clone https://github.com/Indras-Mirror/Xolotl
cd Xolotl
```

Then, in **Git Bash**, from the repo root:

```bash
bash tools/xolotl/fetch-llama.sh      # llama.cpp at the pinned commit + the Xolotl ggml patch
```

## 2. Build

```bat
python -m venv .venv-build
.venv-build\Scripts\pip install cmake ninja
build-win.bat configure
build-win.bat
```

- `build-win.bat` finds Visual Studio through `vswhere` and CUDA through `CUDA_PATH`.
- Set `CUDA_ARCH` for your card (default 86).
- It runs 6 parallel jobs by default; set `JOBS` to change that. Full parallel nvcc can exhaust RAM.
- Output: `build-win\ds41_generate.exe`.

## 3. Model files

`tools/xolotl/win_fetch_files.py` downloads named files from a Hugging Face repo with curl. It resumes, checks every file against the LFS sha256, and needs no hf CLI. It puts files flat in the target folder, so the tokenizer files can sit next to the model.

Example: the official 1017 Engram shards plus the 1017 GSQ parts, then the join:

```bat
python tools\xolotl\win_fetch_files.py deepseek-ai/DeepSeek-V4.1-Flash-1017 main D:\models\dsv41-1017-engram ^
  config.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors
python tools\xolotl\win_fetch_files.py IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored main D:\models\dsv41-1017-gsq ^
  SHA256SUMS rank-1017-085.bin V41-GSQ-1017.gguf.experts.json V41-MTP-Vision-1017.Q2_K.gguf ^
  tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja ^
  V41-GSQ-1017.gguf.part-01-of-15 ... V41-GSQ-1017.gguf.part-15-of-15
python tools\xolotl\win_join_parts.py D:\models\dsv41-1017-gsq V41-GSQ-1017.gguf
```

- `win_join_parts.py` is `reassemble.sh` without bash: it hashes while it joins, then checks the result against SHA256SUMS.
- **The Engram shards are read in place on NTFS**, so `extract_engram.py` is not needed.
- **GGUFs published as llama.cpp split shards** (`-00001-of-0000N.gguf`) must be merged first, because the expert tier reads a single file. `tools/xolotl/gguf_merge_shards.py` does it byte for byte without decoding Strata's tensor types (which llama.cpp's gguf-split cannot size):

  ```bat
  python tools\xolotl\gguf_merge_shards.py <model>-00001-of-0000N.gguf <model>.gguf
  ```

  Expert ranks from such builds (e.g. `expert-rank.json`, a list of `[layer, expert, count]`) convert to `--rank` files as int32 `(layer, expert)` pairs, hottest first.

## 4. Serve

```bat
python -m venv .venv-serve
.venv-serve\Scripts\pip install jinja2 regex
.venv-serve\Scripts\python -I tools\v41\serve\serve_ds41.py --engine strata --config tools\v41\serve\strata-ds41-1017-win.json --port 8150
```

The Windows configs are `strata-ds41*-win.json`. They carry `ds41-engine.sh`'s settings as `"env"`, because there is no flock or systemd-run on Windows. Edit their paths (`exe`, `-m`, `--engram`, `--rank`, `STRATA_SERVE_DSPARK`, `tokenizer`, `log`) for your folders.
- `--engram` takes `;` between folders on Windows.
- Browser chat and monitor: http://127.0.0.1:8150
- OpenAI API: `/v1`; Anthropic API: `/v1/messages`

## Measured on the test machine (served, greedy, 400 tokens, median of 3)

| Model | Prose | Code | Agent JSON | Prefill (13.3K prompt) |
| --- | ---: | ---: | ---: | ---: |
| V4.1 September GSQ, `--arena-gib 64`, PCIe share 1.0 | 13.0 | 17.4 | 15.7 tok/s | 209.5 tok/s |
| same, `--arena-gib 72`, `STRATA_TIER_BATCH_PCIE=0.75` | 14.7 | 22.1 | 18.8 tok/s | - |
| 1017, lxe's dsx 2-bit (merged), September DSpark draft | 12.3 | 18.6 | 15.4 tok/s | 131.9 tok/s |

Notes from this machine:
- **Large pages are not needed for CUDA copies on Windows.** 4 KB pinned pages copy to the card at 26 GB/s here (PCIe 4.0 x16).
- **Decode is bound by kernel-launch overhead.** CUDA graphs halve the dense half's time (`GGML_CUDA_DISABLE_GRAPHS=1` doubles it); the expert tier is not graphed yet.

## What the port changed

- **`tools/v41/engine/ds41_sys.hpp`:** the engine's file and memory calls on Win32. The Linux paths forward to the same POSIX calls as before.
  - Positional reads through `CreateFileW` + overlapped I/O; O_DIRECT maps to `FILE_FLAG_NO_BUFFERING`.
  - 64-bit offsets, because MSVC's `off_t` is 32-bit.
  - `_aligned_malloc` / `_aligned_free` pairs; `GlobalMemoryStatusEx` for MemAvailable.
- **Engram:** the raw block-device read (FIEMAP) is Linux-only; Windows reads the safetensors file.
- **MSVC:**
  - `ptq1_0.cpp` gets intrinsics without GCC target attributes.
  - `ds41_generate` links with an 8 MiB stack reserve, Linux's default, because of its alloca scratch.
  - `strcasecmp` maps to `_stricmp`.
  - `bench_ds41_engram` stays Linux-only.
- **A Windows-only heap corruption, fixed in CMakeLists.txt.** Every prompt batch over 96 tokens crashed with 0xC0000374.
  - `ds41_generate` links ggml-cuda (the dense half) and `strata_mmq`, which are two copies of llama.cpp's MMQ host code.
  - MSVC's linker takes a template instance (`mul_mat_q_case<T>`) from the first library on the line, `ggml-cuda.lib`, which is compiled with `GGML_CUDA_USE_GRAPHS`. That makes `ggml_backend_cuda_context` larger than the one `strata_mmq` allocates.
  - GNU ld searches forward only and keeps `strata_mmq`'s copy, so Linux never hits it.
  - The fix defines the same `GGML_CUDA_USE_GRAPHS` for `strata_mmq`, giving the same layout whichever copy the linker picks.
