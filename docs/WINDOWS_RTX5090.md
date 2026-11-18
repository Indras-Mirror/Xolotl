# DeepSeek-V4.1-Flash-1017 on Windows: one RTX 5090, 64 GB RAM

This fork runs DeepSeek-V4.1-Flash-1017 natively on Windows 11 (no WSL) with an OpenAI / Anthropic-compatible server, a
1M-token context and image input, from either of two public 2-bit quants:

| Quant | Files to download | Short-chat decode, tok/s (fast / light / exact) | Agreement with the original (EAR) |
| --- | --- | ---: | ---: |
| [ISTA-DASLab GSQ-RCO IQ2_XS](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | 155 GB + 111 GB Engram | 81 / 74 / 39 | 86.1 % |
| [Unsloth UD-IQ2_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 301 GB | 71 / 64 / 36 | 88.5 % |

GSQ IQ2_XS is the faster one; UD-IQ2_XXS is closer to the original model. Both run with the same engine and almost
the same config.

It is upstream Xolotl at `03a138f8` (2026-10-26) plus 55 commits (`git log 03a138f8..`):
- **Windows build:** MSVC + CUDA 13 for `sm_120a`, the MXFP4 MMQ fix for Blackwell builds, and `build-win.bat` under
  any console code page.
- **More model formats:**
  - `tools/v41/v41_from_llamacpp.py` for llama.cpp `deepseek41` GGUFs such as Unsloth's.
  - IQ1_S routed experts.
  - Q5_K / Q6_K Engram tables, and a hot set for IQ4_NL Engram tables.
  - K-quant dense matrices.
- **Opt-in decode switches**, each measured on its own (bit-identical or not, and its speed). The configs here
  turn on the ones that won; [tools/v41/engine/DECODE_OPTINS.md](../tools/v41/engine/DECODE_OPTINS.md) and the
  `local rbN` commit messages describe them.
- **Serve:**
  - Disk KV sessions.
  - A saved boundary at the end of the system prompt, so a shared system prompt is not read again.
  - The image encoder uses the GPU only during a turn with a new picture.
- **`--ablate`**: rank-1 edits (Orca's or AtomicChat's direction) applied at run time to any quant.
- **13 upstream fixes** taken by cherry-pick ([UPSTREAM_SYNC.md](../UPSTREAM_SYNC.md)).

Everything below was measured on one PC. Reports from other machines are welcome.

## The test machine

| Part | |
| --- | --- |
| GPU | RTX 5090 32 GB, PCIe 5.0 x16, driver 621.01. MSI Afterburner applies +210 MHz core / +2000 MHz memory at logon (power limit 100 %). |
| CPU | AMD Ryzen 9 9950X3D (16 cores, 32 threads) |
| RAM | 64 GB DDR5 (2 x 32 GB at 6400 MT/s); the page file brings the commit limit to about 95 GB |
| Drives | Samsung 990 EVO Plus 4 TB (model files), Samsung 9100 PRO 8 TB (PCIe 5.0; expert copy 1), an 800 GB enterprise NVMe (EO000800KYDLA; expert copy 2) |
| Software | Windows 11 Pro (build 26300), CUDA 13.4, MSVC 14.51, Python 3.14; torch 2.15.0+cu134 for the image encoder |

## Speed on this machine

### Decode

These are server numbers for short chats. Each run sends 3 prompts (Japanese, math, code), up to 320 new tokens,
greedy, no thinking, and reads the server's own `predicted_per_second`. Each cell is the mean of 6 answers (2 rounds
of the 3 prompts).

| | `-fast` (the configs' default) | `-light` | `-exact` |
| --- | ---: | ---: | ---: |
| GSQ IQ2_XS | 80.9 | 73.8 | 38.8 |
| UD-IQ2_XXS | 71.3 | 64.4 | 36.0 |
| UD-IQ2_XXS + AtomicChat's edit | 68.9 | 62.4 | 36.4 |

- **When:** GSQ was measured on 2026-11-03, before the last three switches in its config were added (they gave about
  +1 % on the server and +2 % forced). UD was measured on 2026-11-04 with exactly the configs below.
- **The routing names:**
  - `-exact` is the model's own routing.
  - `-light` and `-fast` nudge the router towards experts already in VRAM (route bias 0.25 / 0.15 and 0.5 / 0.3).
  - The nudge is faster but changes the output a little. Decode agreement with the original:
    - GSQ IQ2_XS: 87.5 % (exact), 86.6 % (light), 84.6 % (fast).
    - UD-IQ2_XXS: 89.5 %, 88.0 %, 85.4 %.
  - The model name in a request picks the routing; see step 5.

### Prefill

GSQ IQ2_XS, fresh prompts (this repo's docs as text, nothing cached). Measured on 2026-10-28 on an earlier build of
this tree (rb15); the later builds' checks showed the same prompt times.

| Prompt tokens | Time | tok/s |
| ---: | ---: | ---: |
| 872 | 4.7 s | 185 |
| 3,362 | 7.5 s | 450 |
| 15,112 | 12.6 s | 1,195 |
| 32,402 | 17.5 s | 1,856 |
| 65,509 | 31.4 s | 2,090 |
| 131,601 | 60.9 s | 2,161 |
| 274,114 | 128.1 s | 2,140 |

- Short prompts are dominated by fixed per-request costs.
- In a conversation, a new turn processes only its new tokens; the earlier ones are cached.
- UD-IQ2_XXS's prefill was not measured.

### Quality against the original

The reference is Unsloth's UD-Q8_K_XL: its routed experts are the checkpoint's own MXFP4, and its BF16 dense tensors
run as Q8_0. The test text is 16 chunks of 2,047 positions (English, code, Japanese, classical Japanese), run on the
prompt path with exact routing.

| | EAR | KLD | Top-1 | PPL ratio |
| --- | ---: | ---: | ---: | ---: |
| GSQ IQ2_XS | 86.09 % | 0.163 | 84.44 % | 1.053 |
| UD-IQ2_XXS | 88.46 % | 0.113 | 86.62 % | 1.040 |

EAR is the expected acceptance rate: per position, the sum of min(p, q), which is the chance that a token drawn from
the quant would be accepted as the original's, as in speculative decoding.

## Setup

### What you need

- **GPU:** an RTX 50 card with 32 GB. Only a 5090 was tested. Upstream's [WINDOWS.md](WINDOWS.md) covers 24 GB cards.
- **RAM:** 64 GB, plus a page file of 32 GB or more (see the notes at the end).
- **Disk space:**
  - GSQ IQ2_XS: about 605 GB = 266 GB of downloads + 40 GB for the image-ready shard 1 + 2 x 149 GB of expert copies.
  - UD-IQ2_XXS: about 615 GB after setup = the 135 GB Engram shard + the 166 GB converted model + 2 x 157 GB of expert
    copies. The other five downloaded shards (165 GB) can go once the conversion is done.
  - The expert copies want two fast NVMe drives; see step 4.
- **Tools:**
  - Visual Studio Build Tools with "Desktop development with C++" (MSVC 14.51 here).
  - CUDA 13.1 or newer (13.4 here). RTX 50 needs CUDA 13 and the `120a` architecture.
  - Git for Windows (for Git Bash).
  - Python 3.12 or newer.

The examples use these example folders; change them to yours:

| | |
| --- | --- |
| `D:\src\Xolotl` | this repo |
| `D:\models\...` | model files |
| `E:\models` | expert copy 1 (on a second NVMe drive) |
| `F:\models` | expert copy 2 (on a third NVMe drive) |

### 1. Code and llama.cpp

```bat
git clone -c core.autocrlf=false https://github.com/Keisawaakira/Xolotl D:\src\Xolotl
cd /d D:\src\Xolotl
```

Then, in **Git Bash**, from the repo root:

```bash
bash tools/xolotl/fetch-llama.sh      # llama.cpp at the pinned commit + the Xolotl ggml patch
```

`core.autocrlf=false` keeps the ggml patch in LF; a CRLF copy does not apply.

### 2. Build

```bat
python -m venv .venv
.venv\Scripts\pip install cmake ninja jinja2 regex numpy pillow
set PATH=%CD%\.venv\Scripts;%PATH%
set CUDA_ARCH=120a
build-win.bat configure
build-win.bat
```

- **Output:** `build-win\ds41_generate.exe`.
- **Finding the compiler:** `build-win.bat` finds Visual Studio through `vswhere` and CUDA through `CUDA_PATH`. Set
  `VCVARS` to a `vcvars64.bat` if `vswhere` picks a toolset that nvcc rejects.
- **Jobs:** it runs 6 parallel jobs (`JOBS`). More can run out of RAM.
- **The `.venv`:** it also serves steps 3-5. For image input, add a CUDA build of torch from
  [pytorch.org](https://pytorch.org/get-started/locally/). One built for CUDA 12.8 or newer runs on RTX 50; this PC
  uses 2.15.0+cu134.

### 3. Model files

`tools\xolotl\win_fetch_files.py <repo> <revision> <folder> <files...>` downloads with curl.
- It resumes interrupted downloads.
- It checks every large (LFS) file against the Hub's sha256.
- It puts each file flat in the target folder.

#### A. GSQ IQ2_XS

```bat
python tools\xolotl\win_fetch_files.py ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF cb02422c900df489cea5833e8a92171224059b3c ^
  D:\models\dsv41-1017-gsq-iq2xs ^
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf ^
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf ^
  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
```

**Why that revision:**
- On 2026-10-27, ISTA-DASLab moved the repo's `main` to llama.cpp's DeepSeek-V4.1 layout (6 shards).
- The engine reads the earlier 4-shard layout directly, with no conversion, and that is what was measured here.
- The newer layout should go through `v41_from_llamacpp.py` like Unsloth's files, but that has not been tried.

| File (revision `cb02422c`) | Bytes | sha256 |
| --- | ---: | --- |
| `...-GSQ-RCO-IQ2_XS-00001-of-00004.gguf` | 39,535,776,992 | `4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798` |
| `...-GSQ-RCO-IQ2_XS-00002-of-00004.gguf` | 39,494,617,536 | `933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e` |
| `...-GSQ-RCO-IQ2_XS-00003-of-00004.gguf` | 39,423,838,496 | `bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240` |
| `...-GSQ-RCO-IQ2_XS-00004-of-00004.gguf` | 36,771,591,232 | `e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110,595,290,816 | `eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a` |

**For image input**, give shard 1 the 40 image-position router biases that ISTA's file leaves out. Without them the
engine refuses `--vision`. The biases ship in this repo as `data\ds41\bias_vl-1017.gguf`:

```bat
.venv\Scripts\python tools\v41\vision\make_vl_gguf.py ^
  --src-shard1 D:\models\dsv41-1017-gsq-iq2xs\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf ^
  --bias-from data\ds41\bias_vl-1017.gguf ^
  --out-dir D:\models\dsv41-1017-gsq-iq2xs-vl --out-prefix DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-vl
```

- **What it writes:** shard 1 is rewritten (40 GB; its old data is checked byte for byte), and shards 2-4 become hard
  links, so the output folder must be on the same drive.
- **Text only:** point the config's `-m` at the original shard 1, delete the config's `"vision"` section, and skip this
  step.

#### B. Unsloth UD-IQ2_XXS

```bat
python tools\xolotl\win_fetch_files.py unsloth/DeepSeek-V4.1-Flash-1017-GGUF main D:\models\dsv41-1017-ud-iq2xxs\UD-IQ2_XXS ^
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf ^
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00004-of-00006.gguf ^
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf
python tools\xolotl\win_fetch_files.py deepseek-ai/DeepSeek-V4.1-Flash-1017 main D:\models\dsv41-1017-ud-iq2xxs config.json
.venv\Scripts\python -I tools\v41\v41_from_llamacpp.py ^
  D:\models\dsv41-1017-ud-iq2xxs\UD-IQ2_XXS\DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf ^
  --hf-config D:\models\dsv41-1017-ud-iq2xxs\config.json ^
  --out D:\models\dsv41-1017-ud-iq2xxs\DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-xolotl.gguf ^
  --tokenizer-dir D:\models\dsv41-1017-tokenizer
```

- **The files tested here:** the shards as on `main` on 2026-11-03 (5.8 MB, 49.96 / 49.95 / 48.97 / 16.42 / 135.17
  GB).
- **What the converter writes:**
  - The engine's single-file GGUF (166 GB, under 2 minutes here). The routed experts are copied byte for byte.
  - The tokenizer folder.
  - It does not copy the Engram tables: the engine reads them in place from shard 6, so keep that shard.
  - Shards 1-5 are no longer read after the conversion.

#### Both: tokenizer and image files

The UD converter already wrote the tokenizer. For GSQ, fetch it (the same four files):

```bat
python tools\xolotl\win_fetch_files.py IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored main D:\models\dsv41-1017-tokenizer ^
  tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja
```

For image input:
- **The BF16 mmproj:** the encoder's weights.
- **The V41-MTP-Vision sidecar:** read for the vision config only.
- **The official checkpoint's reference code:** `vision.py` and `image_processor.py`.

```bat
python tools\xolotl\win_fetch_files.py ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF main D:\models\dsv41-1017-vision ^
  mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf
python tools\xolotl\win_fetch_files.py IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored main D:\models\dsv41-1017-vision ^
  V41-MTP-Vision-1017.Q2_K.gguf
python tools\xolotl\win_fetch_files.py deepseek-ai/DeepSeek-V4.1-Flash-1017 main D:\models\dsv41-1017-vision\inference ^
  inference/vision.py inference/image_processor.py inference/config.json
```

### 4. Expert copies

The engine reads the routed experts that are not in VRAM or RAM from disk. Two aligned copies on two NVMe drives,
read half from each, make decode faster than reading the model file:
- `STRATA_BLOB_PRIMARY` names copy 1.
- `STRATA_BLOB_MIRROR` names copy 2.
- `STRATA_BLOB_MIRROR_SHARE` sets each drive's share.

Build each copy from the same file that the config's `-m` names: the `-vl` shard 1 for GSQ with images, or the
converted file for UD. Then check every blob of the copy against it:

```bat
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%
set M=D:\models\dsv41-1017-gsq-iq2xs-vl\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-vl-00001-of-00004.gguf
build-win\ds41_generate.exe -m %M% --rank data\ds41\rank-route26k.bin --build-mirror E:\models\dsv41-1017-gsq-iq2xs-vl.experts.primary --mirror-gib 200
build-win\ds41_generate.exe -m %M% --check-mirror E:\models\dsv41-1017-gsq-iq2xs-vl.experts.primary
build-win\ds41_generate.exe -m %M% --rank data\ds41\rank-route26k.bin --build-mirror F:\models\dsv41-1017-gsq-iq2xs-vl.experts.mirror --mirror-gib 200
build-win\ds41_generate.exe -m %M% --check-mirror F:\models\dsv41-1017-gsq-iq2xs-vl.experts.mirror
```

- **Size and time:** each copy is 149 GB for GSQ (157 GB for UD, built from the converted file). Building one took
  about a minute here; checking it took about two.
- **`--mirror-gib 200`:** it is above the total, so every expert is copied.
- **With one fast drive:** delete both `STRATA_BLOB_*` lines from the config. The experts are then read from the model
  file, and decode is slower.

### 5. Configs and the server

| Config | Model |
| --- | --- |
| [`tools/v41/serve/strata-ds41-1017-gsq-iq2xs-5090-win.json`](../tools/v41/serve/strata-ds41-1017-gsq-iq2xs-5090-win.json) | GSQ IQ2_XS |
| [`tools/v41/serve/strata-ds41-1017-ud-iq2xxs-5090-win.json`](../tools/v41/serve/strata-ds41-1017-ud-iq2xxs-5090-win.json) | UD-IQ2_XXS |
| [`tools/v41/serve/strata-ds41-1017-ud-iq2xxs-atomic-5090-win.json`](../tools/v41/serve/strata-ds41-1017-ud-iq2xxs-atomic-5090-win.json) | UD-IQ2_XXS + AtomicChat's edit (below) |

These are the configs this PC runs, with its paths replaced by the example folders.

**Edit before the first start:**
- **Every path:** `exe`, `cwd`, `-m`, `--engram`, `--rank`, `STRATA_BLOB_PRIMARY` / `STRATA_BLOB_MIRROR`,
  `tokenizer`, `log`, `slot_save_path`, `lib_dirs` (the CUDA `bin\x64` folder), and the `"vision"` section's paths.
- **`"python"` in `"vision"`:** a Python with CUDA torch, numpy and pillow, such as `.venv\Scripts\python.exe` after
  step 2.
- **Memory:**
  - `--arena-gib 30` (the pinned RAM cache of experts) fits 64 GB of RAM. With 128 GB, a larger arena is faster.
  - `--vram-margin 2` leaves 2 GiB of the card free.
- **Threads:** `STRATA_WAVE_THREADS 16`, `STRATA_DEC_READERS 8` and `STRATA_DEC_PARTS 8` fit this 16-core CPU.

**Start the server:**

```bat
.venv\Scripts\python -X utf8 tools\v41\serve\serve_ds41.py --engine strata ^
  --config tools\v41\serve\strata-ds41-1017-gsq-iq2xs-5090-win.json --host 127.0.0.1 --port 8080
```

- **Where to connect:**
  - Browser chat and monitor: http://127.0.0.1:8080
  - OpenAI API: `/v1`
  - Anthropic API: `/v1/messages`
- **The model name picks the routing for a request:**
  - `deepseek-v4.1-flash-1017-gsq-iq2xs` uses the config's own routing (`--route-bias 0.5 --route-bias-file 0.3`,
    the same as `-fast`).
  - `deepseek-v4.1-flash-1017-gsq-iq2xs-light` and `deepseek-v4.1-flash-1017-gsq-iq2xs-exact` are the slower, more
    faithful choices.
- **Context:** `--ctx 1048576` with `--kv-fp4`. The FP4 KV packing is exact, because the model itself rounds its
  compressed KV to FP4.
- **Conversations:** `slot_save_path` keeps them on disk.
- **Load time:** the UD config loaded in 13 s in the last test here.

### Optional: AtomicChat's edit at run time

AtomicChat's [DeepSeek-V4.1-Flash-1017-Uncensored-LoRA](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GGUF)
is one rank-1 reflection:
- Every `lora_b` is the same unit vector `r`, and every `lora_a` is `-(r^T W)`.
- So its recommended `--lora-scaled 2.0` is `W - 2 r r^T W` on the attention output, the shared and routed experts'
  down projections, and the Engram values.

The engine applies the same map to those writes as `x - 2 r (r.x)`. That is exact on the quantized weights, with no
re-rounding:

```
--ablate D:\src\Xolotl\data\ds41\deepseek-v4.1-flash-1017-atomic-lora-rv.f32 --ablate-scale 2 --ablate-sites engram,attn,shexp,routed
```

- **The atomic config:** `strata-ds41-1017-ud-iq2xxs-atomic-5090-win.json` is the UD config plus these arguments. They
  work on the GSQ config too.
- **Measured on UD-IQ2_XXS:** KL 0.115 (EAR 88.4 %) from the edited original (UD-Q8_K_XL with the same map). Decode
  speed is the table above.
- **The direction file:** where it comes from is in [data/ds41/README.md](../data/ds41/README.md).
- **AtomicChat's own GSQ build:** it reflects only the 82 dense writers and keeps GSQ's routed experts byte for byte. The
  routed experts write about 30 % of the per-token part along `r`, so that build leaves that part unflipped; the
  run-time flags cover it.

## Notes for Windows

- **Page file:** on this 64 GB PC the server commits about 56 GB: the pinned arena plus Windows' backing for the VRAM it
  holds. With a small page file, `cudaMallocHost` can fail at start. A commit limit of about 95 GB (RAM + page file)
  works here.
- **Other GPU programs:** close them before starting. The engine sizes its VRAM cache at start.
- **UTF-8 mode:** on Windows set to a non-English language, run Python with `-X utf8`.
- **Large pages:** they are not needed (`STRATA_NO_LARGEPAGES=1`).
- **Upstream's notes:** the general Windows notes (port details, a 72 GiB arena that comes up unpinned) are in
  [WINDOWS.md](WINDOWS.md).

## Credits

- **Models:**
  - DeepSeek: DeepSeek-V4.1-Flash-1017 (MIT).
  - ISTA-DASLab: the GSQ-RCO GGUFs and the BF16 mmproj (MIT).
  - Unsloth: the UD GGUFs (MIT).
  - AtomicChat: the Uncensored LoRA (MIT).
- **Code:**
  - Indras Mirror: Xolotl, plus the XQ repository's tokenizer files and vision sidecar.
  - Niko1221: [Strata](https://github.com/Niko1221/Strata).
