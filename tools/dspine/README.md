# Drafter acceptance on our own 27B (DSpine vs DFlash2)

Research tooling, not part of the app. It measures how many tokens per round a block drafter would get accepted on
Xolotl's quantized Qwen3.8-27B, offline, from hidden states the engine dumps. Paths are those of the 9 Nov 2026 run
(the author's drives): edit them before reuse.

1. `mk_prompts.py` writes 6 prompts (code, code edit, agent, explain, math, chat).
2. `run_dump.py [max_new]` (GPU): greedy answers from `xolotl-dense` (thinking off), then each prompt + answer is read
   again as one prompt with `XOLOTL_TAP_DUMP` / `XOLOTL_TAP_LAYERS` set, so the tap rows of every position land in
   `taps.bin` (layers 1,5,16,19,31,33,46,47,61 = both drafters' taps).
3. `eval_accept.py dspine|dflash2 [16,8]` (GPU): walks each sequence as the speculative loop would (anchor, draft a
   block, accept the leading matches against the engine's greedy tokens). `dspine_ref.py` is a torch port of
   Tencent's DSpine drafter (github.com/Tencent/DSpine, `dspine_mlx`); DFlash2 uses `tools/dflash2/dflash2_ref.py`.
   `TOP1=1` skips DSpine's refinement tail; `SHIFT=-1` with a matching dump tests the tap convention.

Result (9 Nov 2026, d0xin uncensored 27B IQ3_S, 6 prompts x 640 tokens), tokens per round, mean over prompts:

| | block 16 | block 8 |
| --- | ---: | ---: |
| tencent/Qwen3.8-27B-DSpine-b16 | 5.41 | 4.60 |
| z-lab/Qwen3.8-27B-DFlash2 | **5.79** | **4.92** |

DSpine's refinement tail adds 8 % (5.01 without it); it leads only on the agent prompt (4.54 vs 4.30). On this
fine-tuned target DFlash2 (already in `xolotl-dense --dflash2`) accepts as much or more, so DSpine is not ported.
