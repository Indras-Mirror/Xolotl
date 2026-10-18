# Running DeepSeek-V4.1-Flash on Windows with cenote

Tested on one RTX 3090 24 GB + Ryzen 7 5800X + 128 GB DDR4 + Windows 11 Pro (2026-10-18). Other cards should
work with a different `CMAKE_CUDA_ARCHITECTURES` (89 = RTX 40, 120 = RTX 50), but they are untested here. Reports
are welcome.

## What you need

| | |
| --- | --- |
| GPU | NVIDIA, 24 GB, with a driver new enough for CUDA 12.8 |
| RAM | 128 GB recommended. The config pins a 72 GiB arena; lower `--arena-gib` on smaller machines (Xolotl's own setup uses 60 GiB on a 90 GB box) |
| Disk | ~370 GB on a fast NVMe (NTFS): the model (152-163 GB) + the official Engram shards (203 GB), plus room for the parts while joining |
| Build tools | Visual Studio 2022 Build Tools (Desktop development with C++), CUDA Toolkit 12.8, Git for Windows (for Git Bash), Python 3.12 |

## 1. Get the code and llama.cpp

```bat
git clone https://github.com/vrami-dev/cenote
cd cenote
```

Then, in **Git Bash** from the repo root:

```bash
bash tools/xolotl/fetch-llama.sh      # llama.cpp at Xolotl's pinned commit + its ggml patch -> third_party/llama.cpp
```

## 2. Build

```bat
python -m venv .venv-build
.venv-build\Scripts\pip install cmake ninja
build-win.bat configure
build-win.bat
```

- `build-win.bat` expects VS 2022 Build Tools and CUDA 12.8 in their default folders, and builds for sm_86. Edit the paths and `CMAKE_CUDA_ARCHITECTURES` for your machine.
- It builds `build-win\ds41_generate.exe` with 6 parallel jobs. Set `JOBS` to change that, but full parallel nvcc can use a lot of RAM.

## 3. Download the model

DeepSeek-V4.1-Flash-1017 (IndrasMirror's GSQ, 15 parts) plus the official 1017 Engram shards, resumable and checked
against Hugging Face's sha256:

```bat
python tools\xolotl\win_fetch_files.py deepseek-ai/DeepSeek-V4.1-Flash-1017 main D:\models\dsv41-1017-engram ^
  config.json model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors

python tools\xolotl\win_fetch_files.py IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored main D:\models\dsv41-1017-gsq ^
  SHA256SUMS rank-1017-085.bin V41-GSQ-1017.gguf.experts.json V41-MTP-Vision-1017.Q2_K.gguf ^
  tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja ^
  V41-GSQ-1017.gguf.part-01-of-15 V41-GSQ-1017.gguf.part-02-of-15 ... V41-GSQ-1017.gguf.part-15-of-15

python tools\xolotl\win_join_parts.py D:\models\dsv41-1017-gsq V41-GSQ-1017.gguf
```

- `win_fetch_files.py` puts every file flat in the folder you give it, so the tokenizer files sit next to the model and that folder doubles as the tokenizer folder.
- The Engram shards are read in place: on NTFS there is no need for `extract_engram.py`.
- For the September build, `python tools\xolotl\win_fetch_v41.py <dir>` downloads and joins everything in one go.

## 4. Serve

```bat
python -m venv .venv-serve
.venv-serve\Scripts\pip install jinja2 regex
.venv-serve\Scripts\python -I tools\v41\serve\serve_ds41.py --engine strata --config tools\v41\serve\strata-ds41-1017-win.json --port 8150
```

Edit the paths in the config (`exe`, `-m`, `--engram`, `--rank`, `STRATA_SERVE_DSPARK`, `tokenizer`, `log`) to match
your folders. Loading takes about 60-70 s. Then:

- Browser chat: http://127.0.0.1:8150
- OpenAI API: `http://127.0.0.1:8150/v1`, model `deepseek-v4.1-flash-xolotl`
- Anthropic API: `/v1/messages`
- Reasoning effort: `none` / `low` / `medium` / `high` / `max`

Close other GPU-heavy apps first. The engine fills most of the card with cached experts once it answers, and pins the
RAM arena for as long as it runs.

## If something goes wrong

- **Every prompt over 96 tokens crashes (0xC0000374).** Your build is missing the `GGML_CUDA_USE_GRAPHS` define on `strata_mmq`. Re-run `build-win.bat configure` on this branch. As a stopgap, `--no-chunk-mmq` in the config's `args` avoids it at some prompt speed.
- **Slow decode.** Check that the model and the Engram shards are on an NVMe drive, not exFAT or a USB disk, and that nothing else holds VRAM.
