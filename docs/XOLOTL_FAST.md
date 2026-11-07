# Fast mode: a 2-bit shadow of DeepSeek-V4.1's biggest experts

> **Opt-in and lossy (7 Nov 2026).** Faster decode for a measurable quality cost. Off unless you turn it on; the model
> file is never changed.

DeepSeek-V4.1 V4 keeps its experts in four formats. The two biggest, the 1,056 hottest experts in MXFP4 (18.8 MB each)
and the 1,778 Q3_K experts (15.2 MB), are about half of the bytes that cross the slow lanes (the RAM bus and the
NVMe drive) on every decode pass. Fast mode reads a 2-bit copy (Q2_K, ~12 MB) of each of those experts instead,
from a sidecar file next to the model. Smaller experts mean ~320 more fit in VRAM, more fit in the RAM arena, and the
file tier shrinks.

## What it does

V4, RTX 4090, Q6_K dense, 80 GiB RAM arena, 8K prompt, 512 tokens; short34 = agreement with the full model's next
token over our 34-text quality set:

| | decode tok/s | short34 top-1 | short34 NLL |
|---|---|---|---|
| normal | 26.1 | 87.01 % | 0.512 |
| **fast (Q2_K shadow, default)** | **33.0 (+27 %)** | **86.16 %** | **0.540** |
| fast, first version (Q2_0 shadow) | 34.1 - 36.7 (+29 to +43 %) | 84.46 % | 0.566 |

- The quality cost is under a point of top-1 agreement with the full model (the first Q2_0 version lost 2.5). Use it
  when speed matters more than the last bit of quality (long agent runs, drafts, chat); keep the normal mode for work
  where every token counts.
- Q2_K is llama.cpp's 2-bit k-quant: a scale and a minimum for every 16 weights, so its weight error is 0.27 of
  the expert's (Q2_0 with one scale per 64: 0.35, even with a searched scale). It is ~15 % bigger than Q2_0, hence a
  little less speed for much less loss.
- Also measured, and dropped: a shadow of only the MXFP4 experts (85.73 %, +24 %), the 1,000 best-ranked experts
  kept exact (85.40 %, +15 %), an imatrix-weighted Q2_0 (84.04 %, +42 %).
- Speed is one 8K run per arm, same window (the decoded text moves how many drafted tokens are accepted, so single
  runs swing by ~5 %).

Agentic coding, the [Gauntlet](../tools/gauntlet/README.md) v3-small (6 hard tasks graded by hidden tests, 10 turns,
thinking on, served settings, 2 runs each):

| | solved | score | decode during the tasks |
|---|---|---|---|
| normal | 8 / 12 | 4.99 | 32.6 tok/s |
| **fast** | **8 / 12** | **4.99** | **41.1 tok/s (+26 %)** |

The same tasks solved and missed in both modes (each missed 04 once; 03 stays half-done in every run).

## Use it

1. **Build the sidecar once** (~30 min on the CPU, ~40 GB on disk; the model is only read):

   ```bash
   CUDA_VISIBLE_DEVICES= build/ds41_generate -m /path/to/V41-XQ-1017-v4.gguf --build-shadow /fast/drive/V4.fast.x41s
   ```

   It writes `V4.fast.x41s` plus a few companion files and `V4.fast.x41s.list`, and prints the relative error per
   expert type. Put it on an NVMe drive: it is read during decode like the model itself.

   *Optional, what our numbers used:* an importance matrix, so the 2-bit copy is most exact where the inputs are
   large. Serve a set of your typical prompts once with `XOLOTL_IMATRIX_OUT=/fast/drive/V4.x41i` (it is written when
   the engine exits), then build with `XOLOTL_SHADOW_IMATRIX=/fast/drive/V4.x41i`. We used 120 prompts of
   ~500 tokens, mostly code and agent work (code 58, long agent sessions 30, reasoning 14, the rest tool calls, web,
   chat).

2. **Serve with it:**

   ```bash
   XOLOTL_SHADOW=$(cat /fast/drive/V4.fast.x41s.list)  <your usual Xolotl serve command>
   ```

   At startup the log says `shadow: N experts from ... as q2_K / q2_K`. The engine checks the sidecar was built from
   this exact model file; if it was not (or a file is damaged), it says `shadow: NOT used (...)` and serves the normal
   experts.

| switch | what |
|---|---|
| `XOLOTL_SHADOW=<file>[:<file>...]` | the sidecar files (the `.list`) |
| `XOLOTL_SHADOW_FROM=39,11` | when building: which expert types get a 2-bit copy (39 = MXFP4, 11 = Q3_K) |
| `XOLOTL_SHADOW_TYPE=10` | when building: the 2-bit format (10 = Q2_K, the default; 42 = Q2_0, the first version) |
| `XOLOTL_SHADOW_IMATRIX=<file>` | when building: weight the 2-bit fit by an importance matrix |
| `XOLOTL_IMATRIX_OUT=<file>` | when serving: record that importance matrix (measurement only, output unchanged) |
| `XOLOTL_SHADOW_THREADS=16` | when building: CPU threads |

**Turn it off:** unset `XOLOTL_SHADOW`.
