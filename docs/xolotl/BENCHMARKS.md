# Xolotl benchmarks

[Back to the README](../../README.md) · [How it works](HOW_IT_WORKS.md)

## HumanEval / tool calls / agentic (`tools/taskeval`)

Every number below is measured on **one RTX 4090 (24 GB) + Ryzen 7 5700X + 90 GB DDR4 + NVMe**, through the real
server, on real tasks: 21 HumanEval problems, 10 tool-call tasks and 3 agentic bug-fix loops (mock repo, hidden
tests), greedy, **thinking on with a 32K-token thinking budget** (the way every model is served). The speeds are the
server's own per-request timings **during those tasks**, not a synthetic benchmark. The harness is in
[`tools/taskeval`](../../tools/taskeval) - run it yourself.

| Model | Size | File (quant) | Context | HumanEval (21) | Tool calls (10) | Agentic (3) | Prompt (short, see below) | Writes answer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DeepSeek-V4.1-Flash** | 552B MoE | [GSQ mixed 1.75-3.4 bpw, uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored) (152 GiB) | 512K | 21/21 | 10/10 | 3/3 | 26 tok/s | 26.4 tok/s |
| **GLM-5.3-Flash** | 320B MoE | Maya-S-v2 IQ2_XXS + abliteration LoRA (96 GB), soft prune + route bias 0.25 | 512K | 19/21 | 10/10 | 3/3 | | 25.3 tok/s |
| **Qwen3.8-Flash-Next** | 125B MoE | Swift 1.5 SC117 GSQ-RCO IQ3_XXS, abliterated | 200K | 21/21 | 10/10 | 3/3 | 475 tok/s | 127.8 tok/s |
| **Qwen3.8-27B** (dense) | 27B | Swift 1.5 uncensored IQ3_S + MTP head (13 GB) | 200K | 21/21 | 10/10 | 3/3 | 1,097 tok/s | 98.5 tok/s |

DeepSeek-V4.1-Flash is complete: 21/21 HumanEval, 10/10 tool calls, 3/3 agentic, 63,402 answer tokens in 2,973 s,
prefill 26 tok/s (short HumanEval prompts - not a representative prefill figure), decode **26.4 tok/s**. GLM-5.3-Flash
(soft prune + `--route-bias 0.25`, `tools/glm/serve/strata-glm-maya-abl-512k-bias.json`): 19/21, 10/10, 3/3, 101,269
tokens in 4,263 s, decode **25.3 tok/s**; the two misses are HE/32 (a wrong answer after 25.8K tokens of thinking) and
HE/160 (it reached its 33,792-token ceiling). Qwen3.8-Flash-Next: 21/21, 10/10, 3/3, 30,809 answer tokens in 275 s, decode
**127.8 tok/s**; Qwen3.8-27B: 21/21, 10/10, 3/3, 30,722 answer tokens in 319 s, decode **98.5 tok/s** (2026-10-17;
the prefill column is again short HumanEval prompts).

Also built and verified: the earlier **DeepSeek-V4-Flash (0731)** engine (`ds4_generate`, 284B MoE, Q2 GGUF):
19.4 tok/s decode, 600-token perplexity 10.39 (the previous engine's runs: 10.41-10.51).

**Exactness.** Unifying four engines into one tree was checked against each engine's own previous build, not just
"it still answers": DeepSeek-V4.1's logits are bit-identical, GLM-5.3 and Qwen3.8-Flash-Next give the same greedy
tokens (within each engine's own run-to-run variation), the 27B's teacher-forced log-probabilities differ by 0.015
nats on average, and the server builds byte-identical engine prompts for every model.


## Prefill on a real prompt (`tools/throughput.py`)

The prompt column above is HumanEval's short prompts (a few hundred new tokens each, where the fixed per-request
cost dominates). The probe below sends a **fresh** long prompt (no prefix reuse), then a 256-token decode on top of
it, then a ~160-token agent turn on the cached context; the speeds are the server's own timings. Quiet box (desktop
only, <1 GB VRAM used).

| Model | Prefill, ~6K fresh | Prefill, ~16.5K fresh | Decode after 6K / 16.5K | +160-token turn on 6K / 16.5K |
| --- | ---: | ---: | ---: | ---: |
| **DeepSeek-V4.1-Flash** (2026-10-17) | 178 tok/s | 268 tok/s | 23.5 / 21.2 tok/s | 7.9 s / 8.9 s |
| **DeepSeek-V4.1-Flash**, draft off the card during long prompts (default, 2026-10-17 eve) | 187 tok/s | 317 tok/s | 22.6-26.0 / 21.6-24.4 tok/s | 8.4-8.7 s / 8.6-9.3 s |
| **GLM-5.3-Flash** (route bias 0.25) | 301 tok/s | 374 tok/s | 22.7 / 21.7 tok/s | |
| **Qwen3.8-Flash-Next** | 3,430 tok/s | 3,005 tok/s | | 0.7-0.8 s (+300) |
| **Qwen3.8-27B** | 1,385 tok/s | 1,244 tok/s | | 0.4-0.6 s (+300) |

DeepSeek-V4.1's prompt batch is sized from free VRAM: ~9,000 tokens per sweep on the first prompt, ~5,900 once the
speculative-decode draft model (2.9 GB) is on the card (since 2026-10-17 the draft is dropped for prompts of 4,096+
tokens and reloaded after them, 2.6-3.8 s: `STRATA_SERVE_DSPARK_EVICT`, 0 = keep it) - a desktop app holding VRAM (a browser, a video call)
lowers it further (10 GiB free: 190 tok/s at 16.5K). Its decode here follows a 6-16K context, so it is below the
26.8 tok/s of the short HumanEval contexts. The Qwen3.8-Flash-Next row is a freshly started server; after an hour of agent work
(the expert cache warm) the same probe gives 3,139 tok/s at 6K and 3,903 tok/s at 16.4K.

## Speed log

Engine speed-ups, newest first, each measured before/after on the same box (RTX 4090, Ryzen 7 5700X, 110 GB RAM), the
same prompt and the same settings, one change at a time. "Runs" = repeats per arm, with a second run of the old
build as the noise control.

| Date | Change | Model | Test | Before | After | Control (old build again) | Quality |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 6 Nov | IQ1_M experts on MMQ for prompts (`STRATA_MMQ_IQ1M`, default on) | Qwen3.8-Flash-Next XQ 45 GB | 16,383-token prompt + 1,024 tokens, 3 runs | prefill 3,239 tok/s | **prefill 3,796 tok/s (+17.2 %)** | 3,281 tok/s (+1.3 %) | closer to Q8_0: prompt KL 0.224 -> 0.184 (20 prompts), top-1 90 % both |
| 6 Nov | the same, recheck on the public build (default on vs `STRATA_MMQ_IQ1M=0`) | Qwen3.8-Flash-Next XQ 45 GB | the same, 2 runs | prefill 3,430 tok/s | **prefill 3,890 tok/s (+13.4 %)** | | |

Decode for the IQ1_M rows: 130.8 / 124.5 / 127.1 tok/s (before / after / control), then 135.9 / 132.9 (before /
after). The time per verify window was 3-4 % higher with it in both A/Bs, while a rerun of the old build alone moved 3 %:
inside the noise of these runs, and the decode kernels are not the ones this change touches. Details:
[CHANGELOG](CHANGELOG.md#iq1_m-prompts-on-mmq-6-nov).

## DeepSeek-V4.1 quant vs the DeepSeek API

The same tasks, the same prompts, temperature 0. API = DeepSeek's `deepseek-flash` endpoint (full precision, default
reasoning effort, `max_tokens` 36,864 covering thinking + answer); local = the GSQ quant (uncensored) through the Xolotl
server, default x0.5 routing bias.

**Gauntlet v2** ([`tools/gauntlet`](../../tools/gauntlet): 7 multi-turn agentic coding tasks, tools in a sandbox,
hidden tests graded every turn; SCORE 0-10 weighs solved, progress and turns):

| Run | Solved | SCORE | Tokens | Decode | Per task (progress on unsolved ones, solved@turn) |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek API | 5/7 | 6.41 | 106,792 | - | 01@4 02@3 03 0.50 06 0.91 07@9 10@4 14@4 |
| **DeepSeek-V4.1 GSQ (local)** | 4/7 | 6.20 | 104,648 | 23.4 tok/s | 01 0.92 02@3 03 0.50 06@3 07 0.77 10@7 14@3 |
| Qwen3.8-27B (local) | 4/7 | 5.41 | 118,145 | 56.6 tok/s | 01@6 02@3 03 0.50 06@3 07 0.91 10@4 14 0.00 |
| Qwen3.8-Flash-Next (local) | 5/7 | 5.50 | 144,246 | 151.2 tok/s | 01@8 02@8 03 0.50 06@3 07@8 10 0.00 14@3 |

**LiveCodeBench v6** (`verify/bench/codebench.py`: LCB's generic prompt, extractor and grader, one shot, pass@1). The
API scored 54/60 = **90.0%** on the fixed 60-problem stratified slice (`lcb60`; 91.7% in the agentic variant with
test runs). Locally, 15 of those problems (7 hard, 7 medium, 1 easy; all ones the API answered in <= 8K tokens,
including its two cheap misses) run one at a time, paired with the API's answers:

| Problem | API | Local |
| --- | --- | --- |
| abc398_f (hard) | fail (stopped on its own after 2,426 tokens) | **pass**, 27,844 tokens (thinking capped at 27.6K) |
| 3751 (medium) | pass, 2,420 tokens | pass, 27,825 tokens (capped at 27.6K) |
| abc395_e (hard) | pass, 2,130 tokens | pass, 6,457 tokens |
| the other 12 | 11/12 pass | running with a 93K thinking allowance |

The quant reasons 3-12x longer than the API on one-shot problems, while on the Gauntlet both sides spend about the
same tokens. Not yet separated: how much of that comes from the quantization, the abliteration or the routing bias.

## DeepSeek-V4.1-Flash-1017 V4 (25 Oct)

**V4 speed update (25 Oct, late): it's caught up, and prefill is way up.** On the coding tasks at the top of this page (21 HumanEval, 10
tool calls, 3 agentic loops) V4 solved **21/21, 10/10, 3/3** and decoded at **25.7 tok/s** (29.0 with the newer
conversation-seeded VRAM cache). On the **full HumanEval (all 164)** it got **162/164** (98.8%, greedy, 32K thinking budget,
decoding at 40 tok/s across the run). Prefill on a fresh prompt is **289-297 tok/s at 8K and 395-409 at 20K** (dense half at
Q8_0), up from 192 / 271 that morning, and decode is 21.7-21.8 / 22.6-24.5 tok/s after 8K / 20K prompts. Short follow-up messages got a lot quicker too: a 32-token
message now takes ~3.9 s instead of ~7.4. With the default Q6_K re-quant prefill is 298-300 / 385-388 and decode
23.5-23.9 / 25.3-25.6. It's all engine work, none of it costs quality: the route bias refreshing as you go, zero-copy NVMe reads,
read-ahead per layer, MMQ for the experts staged from RAM, DSpark staying loaded when the prompt fits anyway, and short
turns going through the prompt batch, and the VRAM cache reloading with this conversation's experts after each
prompt. Pull the latest Xolotl (25 Oct or newer) and use
`--rank v4/rank-1017-v4.bin --tier-batch-pcie 0.25`.

**Coding speed depends on the kind of coding (2 Nov).** A user on a 3090 Ti + 128 GB asked why they see ~35 tok/s
while coding when our tables say ~24. Same machine, same served settings (V4, Q6_K dense, arena 80 GiB, DSpark depth 4
with prompt-lookup drafts), an 8K code prompt, 256 tokens out:

| context | writing new code | repeating the code with comments added |
|---|---|---|
| 524K (as served) | 23.6 tok/s | **35.3 tok/s** |
| 131K | 24.2 tok/s | 34.4 tok/s |
| 131K + `STRATA_PRESTAGE_GIB=1.5` | 24.6 tok/s | 33.6 tok/s |

Context size and prestaging barely move it. What moves it is the work: when the answer repeats or edits code that is
already in the conversation (refactors, adding comments, patching a file), the prompt-lookup drafts get accepted far
more often and decode goes up by about half. Writing new code from scratch stays around 24 tok/s. One probe per cell.

## Measuring it yourself

```bash
python3 tools/taskeval/taskeval.py run http://127.0.0.1:<port> out && python3 tools/taskeval/taskeval.py score out
python3 tools/taskeval/taskeval.py speed out          # prefill / decode tok/s during those tasks
python3 tools/throughput.py http://127.0.0.1:<port> --prompt 20000 --decode 256   # fresh prefill, decode, a small turn
python3 tools/turnbench.py http://127.0.0.1:<port>    # the cost of a small agent turn on a long context
```
