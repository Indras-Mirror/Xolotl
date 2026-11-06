# Fast mode: a 2-bit shadow of DeepSeek-V4.1's biggest experts

> **Opt-in and lossy (7 Nov 2026).** Faster decode for a measurable quality cost. Off unless you turn it on; the model
> file is never changed.

DeepSeek-V4.1 V4 keeps its experts in four formats. The two biggest, the 1,056 hottest experts in MXFP4 (18.8 MB each)
and the 1,778 Q3_K experts (15.2 MB), are about half of the bytes that cross the slow lanes (the RAM bus and the
NVMe drive) on every decode pass. Fast mode reads a 2-bit copy (Q2_0, ~10 MB) of each of those experts instead,
from a sidecar file next to the model. Smaller experts mean ~450 more fit in VRAM, more fit in the RAM arena, and the
file tier shrinks.

## What it does

V4, RTX 4090, Q6_K dense, 80 GiB RAM arena, 8K prompt, 512 tokens:

| | ms per decode pass | decode tok/s | short34 top-1 | short34 NLL |
|---|---|---|---|---|
| normal | 111.6 - 117.2 | 25.7 - 26.5 | 87.01 % | 0.512 |
| **fast** | **83.8 - 90.4** | **34.1 - 36.7 (+29 to +43 %)** | 84.46 % | 0.566 |

- Two runs each so far (the decoded text differs between runs, which moves how many drafted tokens are accepted, so
  single runs swing by ~5 %); more prompts and rounds are being run.
- The quality cost is real: about 2.5 points of top-1 agreement with the full model on our short34 set, and +10 %
  NLL. For comparison, V4's gain over V3 came largely from keeping exactly these hot experts at full precision.
  Use it when speed matters more than the last bit of quality (long agent runs, drafts, chat); keep the normal mode
  for work where every token counts.
- Work in progress to cut the loss: a shadow of the Q3_K experts only (the hot MXFP4 experts stay exact) and a
  calibrated encoder.

## Use it

1. **Build the sidecar once** (~30 min on the CPU, ~35 GB on disk; the model is only read):

   ```bash
   CUDA_VISIBLE_DEVICES= build/ds41_generate -m /path/to/V41-XQ-1017-v4.gguf --build-shadow /fast/drive/V4.fast.x41s
   ```

   It writes `V4.fast.x41s` plus two small companion files and `V4.fast.x41s.list`, and prints the relative error per
   expert type. Put it on an NVMe drive: it is read during decode like the model itself.

2. **Serve with it:**

   ```bash
   XOLOTL_SHADOW=$(cat /fast/drive/V4.fast.x41s.list)  <your usual Xolotl serve command>
   ```

   At startup the log says `shadow: N experts from ... as q2_0 / q2_0`. The engine checks the sidecar was built from
   this exact model file; if it was not (or a file is damaged), it says `shadow: NOT used (...)` and serves the normal
   experts.

| switch | what |
|---|---|
| `XOLOTL_SHADOW=<file>[:<file>...]` | the sidecar files (the `.list`) |
| `XOLOTL_SHADOW_FROM=39,11` | when building: which expert types get a 2-bit copy (39 = MXFP4, 11 = Q3_K) |
| `XOLOTL_SHADOW_THREADS=16` | when building: CPU threads |

**Turn it off:** unset `XOLOTL_SHADOW`.
