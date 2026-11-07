# Xolotl — HANDOFF (2026-10-17, session 6 — CURRENT; sessions 5 and earlier below are history)

## S7 Session 7 (2026-10-18 08:00-) - running log, newest last
- **Gauntlet v2 on the Q6_K-dense default (`verify/e2e/gauntlet-v2-q6k/score.txt`): ds41 5/7, SCORE 5.50, decode
  36.7 tok/s** (v1 Q8_0: 4/7, 6.20, 23.4).  01 + 07 now PASS (were 0.92 / 0.77); **14 = FAIL by Mal's call**: v1 solved
  it at turn 3, this run over-verified - turns 3-7 each ~28K thinking tokens (the budget), prompt growing 29K -> 113K;
  stopped at turn 7 by killing the server (the task records a chat error and is graded; the other 6 results kept).
  Decode on the long thinking stretches was **42-56 tok/s** (DSpark accepts well on repetitive reasoning) - a
  candidate line for the HF card / README (not posted; Mal to OK the text).  One run - one task flipping is noise-level.
- **Prefix-cache BUG (to fix):** task 14 turn 7 re-read all 113,736 prompt tokens ("reused 0", ~300 s at ~370 tok/s
  in 10,225-token batches); turns 4-6 reused 29K / 57K / 85K fine.  Lead: turn 6 was the only turn whose thinking ran
  PAST the 27,648 cap (server.log ~5578: "thinking: 28414 of max 36864" - the "of budget" -> "of max" switch, done at
  28,433 tokens), so the engine's own token sequence (with the forced end of thinking) likely no longer matches the
  history the client sent back -> prefix match fails from the first differing token (here: apparently from 0, which
  is a second question - partial reuse should still have covered the 85K before turn 6).  Repro: a turn whose thinking
  hits the cap, then any follow-up.  Files: `tools/v41/serve/serve_ds41.py` (budget / forced `</think>`), the engine's
  prefix matcher (ds41_generate serve path).
- CPU prep (commit beb27b33): `V41_SRC` override; v2's own rank.bin (+ `strata-ds41-v2r.json`); stale-rank effect = small;
  **v2 was served from exFAT (NVME1TB) vs v1 ntfs3 (SSD NVME)** -> `docs/ds4/V41_TERNARY_PROFILE_PLAN.md`.
- Speed re-test `verify/s6/v1v2r-speed.sh` running 08:47 (log `verify/s6/v1v2r-speed.log`); the morning diag was
  STOPPED (not run) to free the GPU for it - re-run `bash verify/s6/morning-diag.sh` after (it waits on score.txt,
  which exists, so it starts at once).
- Disk: deleted `/media/Crucial1TB/MCP` (stale), all Odin DB copies on Crucible (Mar 2026) + `Odin/experiments/data/odin_test.db`
  (kept: live odin.db + `Team CX2/odin-backups/odin_20261014_203542.db.zst`); 2 unremovable files left on Crucible
  (NTFS I/O error - needs chkdsk/ntfsfix).  1017 rest-of-shards download -> Crucial1TB waits for the Engram one (Mal:
  one download at a time).

- **09:10 v2 re-test with its own rank** (`verify/s6/v1v2r-speed.log`): still slower off disk, equal/faster on the edit
  probe.  **Disk microbench: exFAT NVME1TB 0.96 GB/s O_DIRECT expert reads (no thread scaling) vs ntfs3 SSD NVME
  5.1-5.3; raw NVME1TB 3.31** -> v2's slowdown is the filesystem, not ternary.  Buffered reads on exFAT 0.58 GB/s;
  FIEMAP unsupported on exFAT.  Engram random 4K reads: NVME1TB 10.3K/s, Team CX2 5.6K/s, Crucible 4.4K/s.
- **09:23 `verify/s7/swap-v2-test.sh` armed** (v2 onto SSD NVME for a fair test; v1 to a verified temp copy on NVME1TB
  and back; duplicate dealignai shards 47/48 on NVME1TB deleted first).  09:30: `MIMO_CHUNK_PROF` transcode timer
  (0d980857), built into build-s6 only; `verify/s7/v2s-speed.sh` adds prof / tc3 / mmvq arms.
- **09:50 Mal: skip the x0.3 bias test** (the 1017 may change the overthinking picture) - think diag allowed to finish,
  a watcher stops morning-diag.sh before the bias test.  **1017 download split** (Crucial too small by ~23 GB):
  `tools/v41/fetch_1017.sh` (armed after the Engram loop): small files + shards 1-39 -> Crucial1TB (MAIN), 40-46 ->
  Crucible, 47-48 already on Team CX2.  1017 calib work dir `/home/mal/v41-work/calib-1017` (tokens/hashes/meta copied).
- **GIT DISCIPLINE (shared worktree)**: several sessions use `~/AI/Xolotl` at once.  Commit **named files only**;
  never `git add -A`, `git add .` or `git commit -a`.

- **10:39 swap test DONE** (v1 restored, verified): v2 on SSD NVME >= v1 (prefill +5 %, edits +12 %, decode ~= / +10 % at
  16.5K); transcode 52 / 137 ms per 6K / 16K prompt (~0.2 %); MMVQ-for-ternary worse; TC_GIB=3 maybe +3-5 % (1 run).
- **10:45 Mal: v2-style map for the 1017** - **10:50: but gentler in 20-35** (sacrifice some size for chat quality;
  agentic/tools/code first, balanced).  `v41_calib.py prep --mix` added; the 1017 calibration set rebuilt with chat 4 /
  multi 2 / agentic_long 8 (45420fbf).  Note: prep reads live ~/AI text, so it never reproduces an old set (the V4.1
  default re-run differed in 30/32 sequences) - the saved tokens/hashes/meta are the record.
- **11:00-11:10 Mal: v2 replaces v1 everywhere until the 1017**: v1 GGUF deleted, v2 copied to `SSD NVME/AI/V41-GSQ-v2/`
  (sha256 verified), NVME1TB copy deleted, configs repointed (dfa0ea32), smoke test OK.  No v2 upload unless asked.
- **HF card update note - DRAFT, waiting for Mal's OK** (the Q6_K line needs the Xolotl commits pushed first):
  > **Update, 18 Oct 2026** - A few things since this went up:
  > - **Faster decode, same file.** Xolotl now re-quantises the dense half to Q6_K at load by default: +11-35% decode
  >   on long contexts and +9% on edits, for about 1% more KL vs FP8. Just pull the latest Xolotl, no re-download.
  > - **I built a v2 of this quant**: 140 GiB instead of 152, with act-order GPTQ. Quality about the same (slightly better
  >   held-out perplexity, 0.36 pt lower agreement with the API on chat-style prompts). Served from the same drive it's
  >   as fast or faster: +5% prefill, +12% on edits. It's what I'm running now.
  > - **Working on DeepSeek-V4.1-Flash-1017 now.** Same architecture, so it should run at the same speed. Downloading
  >   at the moment (slow internet), quant up in a few days.
  > - **If you'd like v2 before then**, say so in the Community tab and I'll upload it after the 1017.
  > - **Tip:** don't put the GGUF or the Engram tables on an exFAT drive - Linux's exFAT driver serialises parallel
  >   reads; expert reads were 5x slower for me (0.96 vs 5.1 GB/s).
- 11:20 RESUME_PROMPT START HERE rewritten as one block for both session-7 windows.

## S6 Session 6 (2026-10-17 09:40-) - running log, newest last
- **GPU ownership**: no peer on the GPU at start (bfd97150 idle since 08:06; the 09:20 ds41 start was the outreach
  session's throughput probe for the Phison call, during Teams = 10 GiB free VRAM -> batch 5192, not a quiet number).
  Broadcast on relay that this session owns the GPU. **Lane script now uses port 8161** (`GAUNTLET_PORT`), and
  `GLM_CFG=` picks the GLM config (`verify/e2e/gauntlet-local-all.sh`, backup `.bak-s6`).
- **GLM chat-text cross-check DONE - it overturns session 5's provisional B1.** `verify/s5/glmq/sweep-chat6.log` +
  `sweep-q6.log` (per-token path, `GLM_BIAS_PROMPT=1`, 2,000-token `eval_chat` / 2,586-token `agentic03`):

  | config | chat ppl | agentic ppl | decode tok/s | VRAM hit |
  | --- | ---: | ---: | ---: | ---: |
  | exact (per-token / chunked) | 5.53 / 5.56 | 8.00 | ~11 | 28 % |
  | soft prune 0.05 | 5.43 | 7.89 | 11 | 29 % |
  | prune + bias 0.1 / file 0.1 | 6.56 | 7.51 | 18-20 | 69 % |
  | prune + bias 0.25 / file 0.15 | 7.22 | 5.49 | 26-28 | 86 % |
  | **prune + bias 0.25 (no file bias)** | **6.89** | **5.44** | **25-28** | 86 % |
  | prune + bias 0.5 / file 0.3 (B1) | 9.05 | 4.32 | 35-36 | 98 % |
  | prune + bias 1.0 / file 0.6 | 57.5 | 7.20 | 40 | 100 % |

  The agentic text is the Gauntlet 03 task, and the census (`~/.quetza-data/strata-glm/maya-s-v2.census`, ~240K tokens
  of decode history, updated by EVERY glm_generate run incl. these sweeps, ~1 % per run) was built partly on Gauntlet
  traffic, so "bias lowers agentic ppl" is probably the bias replaying routes already used for that text. Chat text
  is the honest measure. File bias adds cost without speed. **Candidate = soft prune + `--route-bias 0.25`** (chat
  +25 %, cf. ds41's accepted +21 % code ppl), `tools/glm/serve/strata-glm-maya-abl-512k-bias.json`; downstream test =
  `verify/s6/glm-full.sh` (replay 13/14, smoke, HumanEval e2e, Gauntlet v2, throughput on one server).
  Census snapshot at 09:48: `verify/s6/maya-s-v2.census.20261017-0948`.
- **ds41 throughput, quiet box (344 MiB VRAM used), `verify/e2e/s6-throughput-ds41{,-noarena}/`:**

  | ds41 | arena ON (s5 default) | arena OFF |
  | --- | ---: | ---: |
  | prefill 6.1K fresh | 178.5 | 179.4 |
  | prefill 16.5K fresh | 226.2 | 266.6 |
  | decode 256 after 6K / 16.5K | 18.4 / 14.5 | 22.9 / 20.6 |
  | +160-token turn on 16.8K | 12.65 s | 8.95 s |
  | 37-token decode seed after 16.5K | 15.3 s | 3.2 s |

  Prompt batch: 9,069 on the first prompt (13.7 GiB free), ~5,900 after the DSpark sidecar (2.9 GB) loads - never
  12,288, so README's "~350-390 tok/s on 20K" is wrong for the served setup (to correct).
  **Cause (arena ON slower back-to-back):** request end retires up to 2,048 arena slots for the reader, and the next
  prompt pauses the reader at once -> those slots stay EMPTY through the whole next prompt (neither victim nor
  candidate resident): 16.5K prompt file-tier reads 24,926 vs 17,527. turnbench (s5's evidence) had 20 s gaps.
  **Fix (uncommitted until the A/B):** `rebal_cancel` in `tools/v41/engine/ds41_moe.cpp` - at prompt start, queued
  jobs the reader has not started give the slot back to their victim (bytes untouched), the one in-flight read is
  waited for; `STRATA_ARENA_REBAL_DECODE=0` = reads only at request end; `DS41_CHECK_ARENA=1` also byte-checks restored
  victims. A/B = `verify/s6/ds41-arena-ab.sh` (NEW NEWND OFF OLD CHK), pre-fix binary `verify/s6/build-pre-cancel/`.
- **ds41 arena fix VERIFIED + committed + pushed** (8d43fe1b): A/B (`verify/s6/ds41-arena-ab.log`) old 221-226
  prefill @16.5K / decode 14.5-14.9 after 16.5K / +160 turn 11.2-12.7 s; new default (cancel + no decode reads) 268 /
  21.2 / 8.9 s; arena off 267 / 20.6-20.7 / 8.0-9.0 s; the +169 turn on 6.4K 7.9 s (off 9.0-10.2). DS41_CHECK_ARENA:
  3,000+ restored victims + 2,500+ published swaps, 0 bad. Default now = reads only between requests
  (`STRATA_ARENA_REBAL_DECODE=1` = also during decode, -5..-19 % decode). README/BENCHMARKS ds41 prefill corrected
  (9fba44f1, pushed): 178 / 268 tok/s at 6.1K / 16.5K.
- **Gauntlet v2 qflash calibration** (`verify/e2e/gauntlet-v2/qflash`): **5/7, SCORE 5.50** (API 5/7, 6.41): 01@8 02@8
  06@3 07@8 14@3, 03 0.50 (same scene-B wall as the API), 10 0.00; decode 151 tok/s; 2 well-formed unknown-tool calls
  (`edit_file`), 0 corrupt. **Decision: the v2 set stays frozen** (no all-zero problem; changing it = a new API
  baseline).
- **GLM candidate (soft prune + `--route-bias 0.25`), `verify/s6/glm-bias025/`:** replay 13/14 x2 clean (valid names,
  same calls as exact routing in s5); **e2e HumanEval 19/21, tool 10/10, agentic 3/3, decode 25.3 tok/s** (101,269
  tokens, 4,263 s). Fails: HE/32 (find_zero, wrong answer after 25.8K tokens), HE/160 (hit the 33,792 ceiling).
  Exact routing was not run on those two (~1 h each at 10 tok/s).
- **Arena, gapped schedule** (`verify/s6/rebal/tb6.log`, turnbench 4K + 8 turns, 20 s gaps): small-turn prefill mean
  off 9.06 s, new default 8.13 s (-10 %, = s5's -10 %), with decode reads 7.81 s; file-tier reads per small turn
  -21..-29 %. Default kept (decode reads cost back-to-back decode 5-19 %).
- **GLM throughput (bias 0.25)**: prefill 301 / 374 tok/s at 6.1K / 15.6K fresh, decode 22.7 / 21.7 tok/s after them.
- **GLM BUG: CUDA OOM on a fresh ~30K-token re-prefill** (`verify/s6/glm-bias025/engine.log` lines ~300-357, twice):
  after a ~28K-token thinking turn the next request loses prefix reuse (hist=0, lcp ~4.5K) and re-prefills 30-33K
  tokens; `shrink_cache` frees the same ~650 slots as for a 17-token prompt, `safe_chunk` keeps 4096 at cap 32768
  (product 134M < its 268M limit), and `cudaGraphInstantiate` fails (exit 134; the server restarts the engine, the
  Gauntlet task is lost). A 15.6K fresh prompt works. Suspect: per-shape CUDA graphs accumulating over a long
  conversation, or the s28 limit calibrated at a different VRAM state. **The GLM Gauntlet run was stopped (invalid:
  01/02 lost to the crash)**. Repro: serve the bias config, run a long conversation, then a fresh 30K prompt
  (`tools/throughput.py --prompt 38000`). Not fixed yet.
- **Mal asked (13:30) for a full DeepSeek quant vs API benchmark on STANDARD coding benchmarks** (not only ours):
  `verify/bench/codebench.py` (venv `verify/bench/.venv`: evalplus 0.3.1, datasets; LiveCodeBench repo clone
  `verify/bench/lcb`, data `verify/bench/lcb_data/test6.jsonl`): **EvalPlus HumanEval+ (164)** with evalplus's
  instruction prompt + `evalplus.evaluate` (canonical solutions score 163/164 - HE/32 find_zero's reference fails a
  plus input under evalplus's tolerance, the same for every run), **LiveCodeBench v6 (175, 2025-01..04)** with LCB's
  generic prompt (copied verbatim - its prompts package needs an old anthropic SDK) + LCB's own extractor/grader
  (negative-tested: wrong / empty answers score 0), and **lcb60** = fixed stratified 20/20/20 subset
  (`lcb_subset60.json`, random.Random(0)) for the local engine (the full 175 at 25 tok/s would be ~20 h). Greedy,
  pass@1; thinking total 32K + 4096 answer on both sides (local `reasoning_budget_tokens`, API max_tokens). API runs
  `verify/bench/runs/api-{heplus,lcb}` (started 13:33); lcb60 for the API = the full run filtered to the subset.
  Also `out/taskeval-v2-api` (taskeval: HE all 164 + tool + agentic, reasoning passed back, `TASKEVAL_API_THINK`).
  Local runs pending: ds41 `codebench.py gen http://127.0.0.1:8161 heplus|lcb60 verify/bench/runs/ds41-...`.
- **Research (Mal 14:10: "check Xolotl forks and other repos ... see if anyone has something we don't")**:
  forks of Indras-Mirror/Xolotl: `vincalkr/Strata-DSV4.1` (AMD gfx1100 port; measured: ggml fused dense ops
  dsv41-fake-quant + dsv41-fuse +13 % decode / +23 % with DSpark, GGML_CUDA_DISABLE_GRAPHS=1 +6.6 % decode on HIP,
  arena 60->75 GiB +40 % decode on 128 GB RAM, prefill 363 tok/s on a 63K fresh prompt), `mab776/Strata-DS4` (V4-Flash
  0731 engine, 2 GPUs; prompt-lookup drafts +42-45 % decode on edits, async hand-offs, fused hc gates, device-side
  expert mix; a 1,733-fork survey of upstream Strata in its README.mab776.md), `cgtaotao-ctrl` (nothing new).
  `Jackwwg83/Strata-DS` (V4.1, RTX 5090 Laptop + 64 GB: ~30 tok/s short / 24-25 at 4K-256K; SAGE 1.59 bpw quant;
  adaptive RAM tier, 2 MiB-part reads with persistent threads, copy-engine next-layer prefetch, radix top-512 for long
  context, and LOSSY tail skips (HumanEval+ -1.2) we do not take). Upstream Niko1221/Strata: nothing after 0.1.44.1
  (merged). **Our build had STRATA_DSV41_HC_MIX=OFF** (= vincalkr's dsv41-fuse, bit-exact, -546 kernels/token): build-s6
  turns it on. NVMe: we already have O_DIRECT + read_part + 32 persistent readers; the second-drive mirror was built and
  measured dead in s12 (Strata-Backup/Strata-DS4/docs/ds4/V41_START_HERE.md: file tier not bandwidth-bound). The
  Engram is split over the SATA Crucial SSD + NVMe (`--engram /media/Crucial1TB/...:/media/mal/SSD NVME/...`).
- **Implemented (60f2c020, opt-in, not yet measured):** `STRATA_SERVE_KEEP_CACHE=1` (small batch turns shrink the
  VRAM cache only by what the batch needs and refill exactly the evicted slots - the s12 notes' A2;
  ExpertCache::admit_at) and `STRATA_SERVE_LOOKUP=L` (prompt-lookup drafts beside DSpark, mab776's design, <= 6 per
  pass = kMaxMulti 8). GLM: no speculative decode at all (KDA recurrent state cannot rewind) - lookup drafts there
  need a KDA snapshot/restore per verify pass: separate project.
- **GPU window (verify/s6/window.sh, after the ds41/q27 Gauntlet):** build-s6 (HC-mix ON) during q27 ->
  `verify/s6/glm-oom.sh` (BASE / NOGRAPH / CAPN arms: warm 8K, fresh ~30K, ~45K) -> `verify/s6/ds41-improve.sh`
  (H0 HC off, H1 HC on, G1 graphs off, K1 keep-cache, KC keep-cache byte checks, L1 lookup; throughput, editprobe =
  greedy rewrite-a-file turn whose text must be identical across arms, turnbench). Then start
  `verify/s6/ds41-bench.sh` (LCB60-agent, LCB60, taskeval tool+agent) with the winning build/env.
- `verify/s6/badcalls.py <run dirs>` counts malformed tool calls in Gauntlet runs (reproduces s5: GLM served 3/27 bad,
  exact 0/39).
- **Gauntlet v2 final** (`verify/e2e/gauntlet-v2/score.txt`): API 5/7 6.41 | ds41 4/7 6.20 (23.4 tok/s) | q27 4/7 5.41 |
  qflash 5/7 5.50. ds41 and the API trade tasks (API solved 01/07, ds41 06). `gauntlet.py` (d71d95bd): a failed model
  request now ends the task where it got to (`chat_error`) - q27 had lost 14 to a 200K-context overflow.
- **GLM OOM FIXED** (96f93962): CUDA graphs off for big batches only - `GGML_CUDA_GRAPH_MAX_TOKENS` (ggml-cuda patch
  `tools/v41/ggml/cuda-graph-max-tokens.patch`, also in the public `tools/xolotl/ggml/xolotl-ggml.patch`), 64 in
  `glm-engine.sh`. `verify/s6/glm-oom.sh`: BASE / CAPN OOM on 30K / 44K fresh; NOGRAPH ok but decode 16.9-18.4;
  FIX ok, prefill 292 / 368 / 376, decode 19.5-21.7.
- **ds41 improvements** (dee2af6c; `verify/s6/improve*/`, editprobe text identical to H1 in every arm):
  HC-mix ON +3-5 % decode; graphs off slower; **DSpark re-seed on a repeated prompt** (delta 0 used to leave sp_t=-1,
  no DSpark at all) 21.9 -> 32.0; **lookup 14** (kVerify 16) 35.9 edit decode, 1040/1077 accepted; **DSPARK_EVICT
  4096** 16.8K prefill 263 -> 314. Both lookup 14 + evict are defaults in `ds41-engine.sh`. Keep-cache: never engages
  under the default `--vram-lru` (`shrink_cache_for` now logs why) - making it work with vram_lru is the next job.
  The P1 profile printed nothing (the profilers print at engine close / one-shot only).
  **build-xolotl rebuilt with HC_MIX=ON** (19:10); served-default check S1 (`verify/s6/improve3.log`): edit 36.6 tok/s,
  text identical; thr prefill 188 / 318, decode 22.6 / 21.6 (below D1's 26.0 / 24.4 an hour earlier - noise or heat,
  watch it).
- **LiveCodeBench quant vs API** (`verify/s6/ds41-lcb15.sh`, 15 lcb60 problems, paired with the paid `runs/api-lcb60`,
  no new API spend). One hour, 32K budget (27.6K thinking cap): only 3 done - local 3/3 (abc398_f hard which the API
  failed, 3751, abc395_e) vs API 2/3, but **local thinks 3-12x longer** (6.5-27.8K vs 2.1-2.4K tokens; 3 of the 4
  started hit the cap). Gauntlet tokens are equal (105K vs 107K), so it is one-shot reasoning only. The API's two misses
  in the 15 stopped on their own (not cut off); its 3 length-capped lcb60 problems are outside the 15. Mal (19:15):
  run the rest with full thinking -> `verify/s6/lcb15-full.log` (build-xolotl, CB_THINK=98304, no deadline, resumable:
  done ids are skipped). HF card updated (commit 6992cf5e, "Against the DeepSeek API").

- **Night (19:10-23:00)**: LCB local full-thinking run paused at 5/15 (all 5 PASS; abc398_f/3751/abc395_e/abc400_c/
  abc395_f; resumable `verify/s6/ds41-lcb15.sh`, CB_THINK=98304, cap 76.8K); API re-run of its 6 lcb60 misses at max
  effort (`runs/api-lcb60-max`, ~200K tokens ~$0.12): abc400_d / abc400_g / 3762 PASS, abc398_f / 3763 / arc190_a still
  fail.  API facts (docs): deepseek-flash = DeepSeek-V4.1-Flash-1017; temperature ignored in thinking mode.
  HF card updated (6992cf5e); GitHub pushed to 7375e305.  Then the quant investigation + v2 requant: see the START HERE
  block of RESUME_PROMPT.md (2026-10-17 23:00).

- **Night + morning (2026-10-17 23:00 - 10-18 07:50): the quant investigation + GSQ v2** - full results in the
  START HERE block of RESUME_PROMPT.md (2026-10-18 07:50).  In one line each: FP8 93.65 % vs v1 86.13 % API top-1 (the
  quant is the gap); error-fixing adapters 1-5 %; Hadamard skipped (< 1 GiB, needs the fused down kernel); act-order
  adopted; v2 built (139.8 GiB, ppl 5.603 vs 5.645, fixtures 85.76 vs 86.12 %, but slower - not default); dense Q6_K
  at load = the new served default (0ecf9111, not pushed); GitHub issue #2 answered (kept open); 1017 Engram shards
  downloading to Team CX2 (resumable).

# Xolotl — HANDOFF (2026-10-17, session 5 — history)
**RESUME = the "START HERE" block at the top of `RESUME_PROMPT.md`** (written at the end of session 5, ~08:20). It
supersedes S5.2/S5.3 below where they disagree (e.g. the adaptive arena is now ON by default, GitHub is pushed and the
repo renamed to Indras-Mirror/Xolotl).

## S5.0 Session 5, second half (2026-10-17 05:40-08:20) in one list
- Arena ON by default (cff4bb24); `build-xolotl` fully rebuilt with the README configure line (pinned ggml in
  third_party/llama.cpp; the old cache pointed at the moved Strata-DS4); a CPU-target bug fixed on the way.
- Wrappers `strata-ds41-quetza` / `strata-glm-quetza` repointed to `build-xolotl/` (backups `*.bak-20261017`).
- GLM `--route-bias-file` + decode-only `--route-bias` (7d7775c1, opt-in); the ppl sweep (RESUME 6.4) says the ds41
  recipe (bias 0.5 / file 0.3) is 3.5x faster than exact with LOWER agentic-text ppl (4.32 vs 8.0) - the chat-text
  cross-check is still to run; no served config changed yet.
- Gauntlet v2 (f37a229d): 10 turns, no token budget, per-turn hidden grading, normalised progress; core set 01,02,03,06,
  07,10,14; API 5/7 (SCORE 6.41) in `out/gauntlet-v2-api`. Out of the public docs until refined + all models re-run.
- README rebuilt headline-first + docs/xolotl/{HOW_IT_WORKS,BENCHMARKS}.md; brand (Assets/ = Mal's picks); HF card
  updated; GitHub renamed + pushed (fast-forward). Qwen prefill measured properly (Flash 3,005-3,430, 27B 1,244-1,385).
- A second Claude session (bfd97150) ran ds41 turnbench on the same GPU/port: the s5 ds41 throughput probes are
  contaminated (except 263.8 tok/s @16.7K on a fresh server). Re-measure on a quiet box.

**State in one line:** every Xolotl model has now run the Gauntlet core set and the HumanEval e2e; README's two Qwen
rows are filled; GLM's served config was found to corrupt tool calls under its skip/prune approximations (the merge is
clean); the small-turn prefill fix (716cf901) is in `build-pf` only. **Nothing pushed.**

## S5.1 What session 5 did (overnight 2026-10-16/17)
1. **Fixed a runner bug that graded every LOCAL Gauntlet task as a runner error** (5ed4adf8): with
   `TASKEVAL_THINK_BUDGET` set, the per-step answer cap overwrote the exact-answer pattern `answer`, so grading called
   `re.search(<int>)`. API runs never set it, so every earlier API number stands. `selftest.py` gains case **f** (the
   local path; the pre-fix runner dies on it) (751a4ab0).
2. **`gauntlet.py score` reports throughput** (918f7b55): decode, prefill over NEW prompt tokens, new/reused, e2e.
3. **Gauntlet core set (03,12,13,14,15,16; 12K tokens/task) on all models + the API** - table in
   `tools/gauntlet/README.md` (d54ef113): API 1/6, qflash 1/6, q27 0/6, ds41 0/6, GLM served 0/6, GLM exact 1/6.
   Results: `verify/e2e/gauntlet-local/<model>/`, `out/gauntlet-core-api/`, `verify/e2e/gauntlet-local/score.txt`.
4. **HumanEval e2e remainder**: qflash and q27 both 21/21 he, 10/10 tool, 3/3 agentic; decode 127.8 / 98.5 tok/s
   -> README rows (b9c605a3). `taskeval.py score/speed` over `r_glm` crashes (the 5-problem partial has no
   results.json) - score the other dirs explicitly.
5. **GLM A/B** (scratch, `/tmp/claude-1000/.../scratchpad/glmab/`): Gauntlet 13/14 step-1 requests replayed greedy on
   A Xolotl served config, B pre-merge `~/AI/Strata-Backup/Strata-GLM` build same flags, C/D the two trees with exact
   routing. C == D (identical on 14; same tool calls on 13) -> **no merge bug**. Malformed calls only under
   `--skip-miss/--skip-file/--skip-file-prefill 0.15 --renorm-skip --prune prune-ezct-0.25 --prune-penalty 0.05`.
   Gauntlet with exact routing (`gauntlet-local-all.sh glm-exact`): 0 malformed of 39 calls (served: 3 of 27), the
   table's only 03 pass - but decode 10.1 vs 30.6 tok/s.
6. ds41 Gauntlet ran on **`build-pf`** (has the prefill fix); `build-xolotl/ds41_generate` is from 14:52 (pre-fix) and
   its CMakeCache still points STRATA_GGML_DIR at the moved `~/AI/Strata-DS4` - rebuild it before relying on it.

## S5.2 Open (needs Mal)
- **GLM's served config**: keep the fast approximations (30 tok/s, corrupt calls) or serve exact routing (10 tok/s,
  clean)? Or something between (e.g. skip thresholds only on prefill). Mal's rule: soft prune only - `--skip-*` drops
  experts outright.
- **The core set barely separates models** (13/15/16 zero for all; 12 only free credit). Hardening rule says replace
  tasks nobody passes 3x - candidates for the next calibration.
- **Wrappers** `strata-ds41-quetza` / `strata-glm-quetza` check `build-ds4-gpu/` / `build-glm-gpu/`, which do not exist
  in Xolotl - launching through them likely fails (untested this session).
- **Adaptive arena** (`STRATA_ARENA_REBAL=1`, off by default, uncommitted in ds41_moe.cpp/.hpp, ds41_generate.cpp,
  turnbench.py): see S5.3.

## S5.3 Adaptive arena (in progress)
Committed opt-in (5429b5ad, `STRATA_ARENA_REBAL=1`, off by default). The size-aware planner fixed the "slot small"
rejections (s4: 3,178 of 3,178): now 2,561 swaps, all published, 0 bad under `DS41_CHECK_ARENA=1`. On the turnbench
agent schedule (4K base, 8 turns, `--gap 20`): file-tier experts per small turn -21..-30%, small-turn wall time
8.48 -> 7.60 s (-10%); the simulated ceiling was -35..-45%. Logits are not bit-identical when on (MMVQ vs MMQ by
residency), so the gate is byte checks + an output-quality run: `verify/e2e/final-think/r_ds41_rebal` (HumanEval/tool/
agentic with it on, session 5): **21/21, 10/10, 3/3 - same as `r_ds41`**, decode 26.8 vs 26.4 tok/s, 16,170 swaps
published over the run, 0 bad. (`r_ds41` ran on the older `build-xolotl` binary, so that is a quality match, not a
clean speed A/B.) Making it default is Mal's call.
Next levers: more swaps per gap (`end_batch`, `per_tick`), and the prompt's own reads overlapping the dense phase
(s4 item 1, predicted router) - not started.

---
# Xolotl — HANDOFF (2026-10-16, session 4 — CURRENT; sessions 1-3 below are history)
**State in one line:** the engine work is unchanged from session 3 (one tree, four models, verified); session 4 built
and calibrated **The Gauntlet** — a new 10-task agentic benchmark in `tools/gauntlet/` where the full-precision API
scores **3/10** — and stopped the HumanEval e2e chain mid-GLM on Mal's instruction. **Nothing pushed.**

## S4.0 Resume

```bash
cd ~/AI/Xolotl && git status --short && git log --oneline -12
python3 tools/gauntlet/validate.py            # must be 10x OK (exit 0) - the benchmark's own gate
python3 tools/gauntlet/selftest.py            # must be a-e PASS (exit 0) - the runner's gate
cat tools/gauntlet/README.md tools/gauntlet/HARDENING.md   # what the benchmark is, and how to harden it
cat out/gauntlet-api-final.log               # the calibration run (3/10)
```
Then read `RESUME_PROMPT.md` section 14 for the continuation brief.

## S4.1 What session 4 did

1. **Built the Gauntlet** (10 tasks, one directory each) and calibrated it: the API scored 8/10 - too easy.
2. **Hardened it in three waves** with `flash` workers (one slice per task, every slice gated by `validate.py` on its
   own task + a derivability table): 01 -> three modules; 02 SemVer -> an invented release-stamp spec with `merge` and
   a `select` constraint language; 04 -> 12 ids, a cross-footing count table, a one-clause exemption pair; 05 -> its
   semantics un-puzzled from Python's `re`; 06 -> the purge-order contract + an 8-thread stress; 07 -> header-signature
   dialects + per-row rounding + a running balance; 08 -> `C/Q/A` curves and arcs; 09 -> `agenda()` + `blackouts.txt` +
   minimum notice; 10 -> a third module with cross-module faults.
3. **Calibrated again: 3/10** (PASS 01, 02, 06 - all at the 12-step ceiling; FAIL 03, 04, 05, 07, 08, 09, 10), 246K
   tokens, 1,453 s.
4. **Fairness audit** (independent worker, read-only): all **246 hidden assertions** mapped to the sentences that
   document them -> **FAIR AS IS**; four findings, all resolved (R1 07's unstated trailing newline, R2 05's substring
   `import re` scan, R3 09's tzdb dependency, R4 03's wall-clock assertion).
5. **Closed two grading-integrity holes** in the runner: the workspace could grade itself (`conftest.py` planted via
   `run_python` measurably flipped 01 from "13 failed, 20 passed" to "33 passed") and model code inherited this
   shell's environment (`DEEPSEEK_API_KEY` visible). Both fixed and probed.
6. **Stopped the e2e chain** on Mal's instruction, mid-GLM, after its current problem: GLM has 5 of 21 HumanEval
   attempted (PASS 0/8/16/24, FAIL 32 at its own 33,792-token ceiling); qflash and q27 never ran.

## S4.2 Where everything is

| what | where |
| --- | --- |
| the benchmark | `tools/gauntlet/` - `README.md` (full docs), `DESIGN.md` (rules + task table + verification record), `HARDENING.md` (evidence-backed levers + sources), `gauntlet.py` (runner), `validate.py`, `selftest.py`, `tasks/NN-*/` |
| the calibration run | `out/gauntlet-api-final/` (+ `.log`); earlier waves: `out/gauntlet-api-rest/` (04/06/09/10), `out/gauntlet-api-0207/`, `out/gauntlet-api-01/` |
| lane scripts | `verify/e2e/gauntlet-local-ds41.sh` (serve a local engine on 8151 + run + score; local-only, `verify/` is in `.git/info/exclude`), `verify/e2e/final-think-resume.sh` (the e2e remainder: qflash + q27 only, then score/speed) |
| e2e results so far | `verify/e2e/final-think/r_ds41.out` (21/21 he, 10/10 tool, 3/3 agentic, 26.4 tok/s decode), `r_glm.out` (5 of 21 he, partial), `server_*.log` |
| worker reports | `~/.quetza-data/conductor/reports/ga-*.md` (one per slice, each with its gate output and derivability table) |
| the audit | `~/.quetza-data/conductor/reports/ga-audit-final.md` |

## S4.3 Measured results (the numbers to quote)

- **Gauntlet, `deepseek-flash`: 3/10** (was 8/10 pre-hardening). Passes 01, 02, 06 - each at the 12-step ceiling and
  never calling `submit`. Misses are narrow: 03 one scene, 07/10 two cases, 04 two assertions; 05 (122) and 09 (36)
  collapse.
- **e2e**, thinking on with a 32K budget: DSV4.1 21/21 he, 10/10 tool, 3/3 agentic, decode 26.4 tok/s; GLM partial (5
  problems, ~28 tok/s, one 893 s outlier and one failure at its token ceiling).

## S4.4 Verification and integrity (all re-run by the conductor, not taken on a worker's word)

- `validate.py` **10/10 OK** on the settled tree; `selftest.py` **a-e PASS** (a/b/c/d answer-grading + loud failure + a
  real task solved; e = the workspace cannot grade itself).
- **~100 repeat task-validations, zero flakes** (whole gate 3x; 06 10x; 03/08 10x each; 01/02/07/10 5x each).
- **A fresh worktree at HEAD validates 10/10** - every task file ships (this is what caught 04's `service.log` being
  swallowed by the repo-wide `*.log` ignore rule; fixed with a negation in `.gitignore`).
- **Grading cannot be subverted**: deny-list + pre-grading strip + a six-variable child env (no credentials), each
  verified with the conductor's own probe scripts.

## S4.5 Parked (needs Mal's word)

1. **The local quant contrast** - `bash verify/e2e/gauntlet-local-ds41.sh`. Expect hours; GPU must be free. Mal is
   using the 4090 for a game, so: **do not start an engine without him saying so.**
2. **The e2e remainder** - `bash verify/e2e/final-think-resume.sh` (qflash + q27, then score/speed over all four).
3. **The push** to Strata-DSV4.1 - still gated on Mal's OK and the README table being complete (the V4.1 row is filled
   from the measured run; the other three rows say `pending`).

## S4.6 Traps that cost time this session (do not repeat)

- **`pkill -f "<pattern>"` matches your own shell's command line** - it killed my own command mid-sequence twice, and
  the teardown silently didn't happen. Use exact PIDs from `ps`, or a bracket pattern.
- **`| tee out/x.log` fails if `out/` doesn't exist**, and the harness's python still starts; check the log file
  exists before believing a run started.
- **A relay `ask` sits in the worker's inbox until the worker polls it** - a long tool loop can miss it entirely; if
  the report comes back without the addition, re-ask or dispatch a follow-up slice.
- **Never run the full `validate.py` while a slice is editing its task dir** - you will read a half-written tree
  (this produced a false "reference: hidden PASS NO" during the integrity slice).
- **A worker's "verified" is a claim**: re-run the gate on the settled tree (that is how the mid-write false negative
  was caught, and how the `service.log` ignore trap was found).

---
**State in one line:** all four models (DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B dense) +
the legacy DeepSeek-V4-Flash-0731 engine run from THIS tree, verified against each one's previous production build;
the live wrappers point here; the old trees are in `~/AI/Strata-Backup`.  **Nothing pushed.**  Mal: push only when
DSV4.1 is definitely up to scratch + the README carries every model's measured numbers.

## S3.0 Resume

```bash
cd ~/AI/Xolotl && git status --short && git log --oneline -8
cat verify/e2e/final-think.log; cat verify/e2e/final-think/{score,speed}.txt   # the README table's source run
```
Then: fill README.md's {{...}} placeholders from score.txt / speed.txt (taskeval.py score / speed), fresh-clone build
test (S3.5), rebuild build-xolotl + build-vision against third_party/llama.cpp (S3.5), then ask Mal for the push.

## S3.1 Where everything is now

| what | where |
| --- | --- |
| the engine | `~/AI/Xolotl` branch `xolotl` (remotes: `github` = Indras-Mirror/Strata-DSV4.1, `github-glm`, `niko` = upstream, `perrone`, `origin` = the backed-up fork, `up` = ~/AI/Strata) |
| patched llama.cpp | `third_party/llama.cpp` (untracked, gitignored) = llama.cpp `3cf03257` + `tools/xolotl/ggml/xolotl-ggml.patch` (verified identical, diff -r). Fresh checkout: `bash tools/xolotl/fetch-llama.sh` |
| builds | `build-xolotl/` (all engines; CMakeCache still says STRATA_GGML_DIR=~/AI/Strata-DS4/... -> RECONFIGURE to third_party/llama.cpp before the next build), `build-vision/` (strata-vision; same) |
| OG upstream | `~/AI/Strata` (kept: its `.venv` runs every server; `data/expert-profile*.bin` used by the Qwen Flash config) |
| model data | `~/AI/Strata-data` (Qwen Flash packs + MTP) - keep |
| backup | `~/AI/Strata-Backup/`: Strata-DS4 (+ worktrees Strata-GLM/-um/-rebase/MiMo, repaired), Strata-qwen, strata-dense-build/-runtime, Strata-DS4-private, models-dense, v41-dev. Rollback binaries kept (build-ds4-gpu, build-glm-gpu, strata-dense-build). 89 GB of big intermediates deleted with Mal's OK - manifest `DELETED-2026-10-16.txt` |
| wrappers | strata-quetza (configs `~/.quetza-data/xolotl/`), strata-dense-quetza (draft 3), strata-glm-quetza, strata-ds41-quetza -> Xolotl; backups `*.bak-prexolotl-20261016`. qwen3.8-swift-quetza-ninfer: model path fixed (moved to /media/Crucial1TB/models/ninfer). v41allday/v41calib symlinks -> Xolotl tools |
| e2e harness | `tools/taskeval` (smoke + HumanEval-21 + tools-10 + agentic-3; `speed` = prefill/decode during the tasks; `TASKEVAL_THINK_BUDGET`) |

## S3.2 Bugs found this session (all fixed + committed)

| # | what | mechanism |
| --- | --- | --- |
| 12 | every Qwen request failed: `'Tokenizer' object has no attribute '_re'` | fork renamed `_re`->`_pre_res`, dense lineage still read `_re`; clean textual merge |
| - | strata-dense (27B) not built at all | its CMake target only ever existed uncommitted in /tmp/strata-dense-eval; reconstructed (+ Q5_0/Q6_K MMQ) |
| - | `/v1/decide` missing on the dense engine | the score= edit was also only in /tmp; re-implemented |
| - | voice sidecar: cold start killed under the first request; children outlived SIGTERM | last_used=0 until after ensure(); no closer |
| 13 | V4-0731 lost layer 42 (`profile: per token 252`) | GLM's unconditional `block_count - nextn_predict_layers`; now only when blk.N.nextn.* exists |
| - | GLM serve OOM on prompts <= 256 tokens (production too) | shrink only for >256 but every prompt took the chunk path |
| - | GLM/DSV4.1 shims ran ~/AI/Strata's 0.1.39 server, not this tree's | moved; then 3 drifts vs 0.1.44: run(force=), finish(reason=), template.caps; and the call-opener follower guard (`<function=`) that turned every DSML/GLM tool call into text -> `F.FUNC_START = ""` in both shims |
| - | a fresh clone could not build | ggml patches only in an untracked tree -> one patch on the pin |

## S3.3 Merges this session
* upstream **Strata 0.1.44.1** (Niko1221 main 296b59d2, 282 commits): 11 conflicts, unions.  Deliberate deviation:
  K-quant MMQ stays static + always on (upstream made it a run-time opt-in DSO -> would switch off the dense and GLM
  prompt paths).  Gave Qwen Flash decode 116.5 -> 151 tok/s.
* the Strata-qwen voice work (uncommitted there) - voice.py, /v1/audio/*, --audio-pad/SVA1, tools/audio.
* NOT merged: mimo (Mal: drop it; weights deleted).  perrone/qwen3.8-27b = same dense code as ours.

## S3.4 Verified (logs in verify/, untracked)
* DSV4.1: logits bit-identical; e2e (non-thinking) Xolotl 20/21 / 10/10 / 3/3, 5391 tok, 596 s == production 598 s;
  decode during tasks 27.5 vs 27.8 tok/s.
* GLM: engine within production's own noise; serve GEN lines byte-identical; tool calls identical at every split.
* Qwen Flash: identical greedy chat + image answers; decode 151 vs 116.5, prefill 3577 vs 3509 (speedbench).
* 27B: teacher-forced mean |dlogprob| 0.015 (newer ggml MMQ in the prompt path); decode 116 (draft 3) vs 105;
  prefill 1240 vs 1331 (-7%, OPEN: newer MMQ kernels; nsys both to find it).  NInfer on the same bench: 4402 / 176.
* V4-0731: ppl 10.39 vs production 10.41-10.51, decode 19.4 vs 19.3.
* serve suite 733 tests; test_serve_ds41.py all passed.

## S3.5 Still to do before a push (in order)
1. finish the thinking run -> README table (placeholders {{...}}).
2. fresh clone of this repo into scratch, configure WITHOUT STRATA_GGML_DIR, build (proves the patch path).
3. reconfigure build-xolotl / build-vision: `-DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp`, `-DLLAMA_DIR=...` (full
   rebuild ~8 min; do not run while benchmarking - it skews CPU-pool speeds).
4. Mal's go, then: push `xolotl` -> github `main` (it fast-forwards over 9d6529fc; the graft replace-ref does not push and
   is not needed for that), set repo description/topics (draft in docs/brand/XOLOTL_BRAND.md), social preview image.

## S3.6 Open (not blockers)
* 27B vs NInfer: prefill 3.5x / decode 1.5x behind; the gap is int8 GEMM kernels + spec depth. Options: port kernels,
  or serve the 27B through NInfer. Mal to decide.
* the 27B -7% prefill vs production; GLM <=16-token prompts now use the decode loop (fine, but slower than MMQ for 17-256).
* llama.cpp b11390 (MoE MMQ temp-buffer padding fix) - check whether xolotl-ggml.patch already covers it.
* NVIDIA's Qwen3.8-Flash-Next NVFP4 QAD: readable now (NVFP4 W4A8); waiting for an abliterated one; needs ~102 GB disk.


---

# Xolotl — MERGE HANDOFF (2026-10-16, session 2)

**Goal (unchanged):** ONE engine, ONE tree, serving **GLM-5.3-Flash + Qwen 3.8 27b dense + Qwen
Flash-Next MoE + DeepSeek-V4.1-Flash**, eventually pushed over the existing
**`Indras-Mirror/Strata-DSV4.1`** repo (preserves the stars — do NOT create a new repo).

**The build is now green and the test suite runs.** The tree builds all four engines and passes
53/54 ctest. One RUNTIME failure is open (§6) and one ctest failure is environmental (§7).

Everything in this document is measured, not assumed. Where something is a hypothesis it says so.

---

## 0. TL;DR for the next session

```bash
cd ~/AI/Xolotl && git status --short && git log --oneline -3    # see §6: an UNVERIFIED one-line fix may be uncommitted
ninja -C build-xolotl                                           # REBUILD FIRST (it was in flight at handoff)
ninja -C build-xolotl && ctest -j2 --output-on-failure          # expect: no work to do / 53 of 54
cat glm-tierinit-ab.log                                         # the A/B smoke test launched at handoff (see §6.4)
```

Then: re-run the GLM smoke test (§6.4) and confirm the tier-init error is gone. That is the last known
blocker for serving GLM-5.3 from this tree.

---

## 1. Where things stand

| | |
|---|---|
| repo | `~/AI/Xolotl` — **separate sparse clone** (not a worktree), branch **`xolotl`**, tree **clean** |
| origin | `~/AI/Strata-DS4` (the fork) → so `glm`, `mimo`, `deepseek4` are `origin/*` |
| up | `~/AI/Strata` (upstream) → `up/dense-eval`, `up/main`, `up/qwen` |
| graft | applied as a **local replace ref**: `git replace -l` → `01529aaa82b9f5c58d2bb87264b7b3f7b823678f` (needed for the upstream merge; replace refs do NOT push — see §11.1) |
| base | `deepseek4` (= DSV4.1), then `up/dense-eval`, then `origin/glm` merged in |
| commits on `xolotl` | **12** on top of the merged base (log in §9) |
| build dir | `~/AI/Xolotl/build-xolotl` — configure PASSES; **build COMPLETE** (64/64, 0 errors, last rebuild 06:02:43 after the §6 fix) |
| engines built | `strata` (Qwen), `glm_generate`, `ds41_generate`, `ds4_generate` — all four run (§7.3) |
| tests | **53/54 pass**; `ple_parity` fails on a missing model fixture (§7.2) |
| open | §6: bug 11 (GLM tier init) **fixed + compiled green, runtime verification still owed** — one command |
| pushed | **NOTHING**. Nothing pushed, no repo created, no wrapper repointed. |
| disk | `/` at **98 %**, **~19 G free** (was 26 G before the §5 bug-8 MMQ instances + the bug-11 rebuild) — the real constraint (why the clone is sparse, excludes `bench/`) |
| GPU | free. 778 MiB used, **no compute apps**. The only "GPU" process is Odin's embedding server, CPU-only (`-ngl 0`). |

---

## 2. Every repo, worktree and tree on disk — do not lose track

There are **exactly TWO real git repos** in the Strata family. Everything else is a **worktree** of
one of them, a **build dir**, or a **runtime/scratch** copy. `git worktree list` is the truth:

### 2.1 `~/AI/Strata` — UPSTREAM (remote `up` from Xolotl)

| path | branch | head |
|---|---|---|
| `~/AI/Strata` | `main` | `9bb510c8` |
| `~/AI/Strata-qwen` | `qwen` | `e405daf1` (**has UNCOMMITTED work** — see §10.2) |
| `/tmp/strata-dense-eval` | `dense-eval` | `eb339324` — **prunable, directory already gone** |

Other branches: `dense-eval` `eb339324` (the lineage merged into Xolotl), `qwen-dense` `e405daf1`,
`qwen-wip-frontend` `3ed86dc7`, `qwen-0.1.39-backup` `9bb510c8`.

### 2.2 `~/AI/Strata-DS4` — THE FORK (**origin** from Xolotl)

| path | branch | head | notes |
|---|---|---|---|
| `~/AI/Strata-DS4` | `deepseek4` | `9d6529fc` | the base; **this is what the DSV4.1 wrapper runs** |
| `~/AI/Strata-GLM` | `glm` | `94ac0227` | **4 files uncommitted**; the GLM wrapper runs its `build-glm-gpu/` |
| `~/AI/Strata-GLM-rebase` | `glm-rebase` | `fb58e0db` | **382 files STAGED** |
| `~/AI/Strata-GLM-um` | `glm-upstream` | `87cd462f` | **1 file uncommitted** |
| `~/AI/Strata-MiMo` | `mimo` | `bf38193f` | the MiMo branch — **NOT merged** into Xolotl (deliberate) |

Other branches: `deepseek4-preclean` `0282dc74`, `ds4-engine-wip` `225da367`, `ds4-on-0141` `77237127`,
`ds4-upstream` `be3a4ba5`, `main` `6f32ec07`.

### 2.3 Non-repo trees that matter

| path | what it is |
|---|---|
| `~/AI/Xolotl` | the merge work (this doc). Sparse clone of Strata-DS4; `third_party/llama.cpp` is an **empty submodule** → the build points `STRATA_GGML_DIR` at the fork's patched copy (§4) |
| `~/AI/Strata-DS4-private` | present, **not inspected this session** — check before assuming it is disposable |
| `~/AI/Strata-data` | present, **not inspected this session** |
| `~/AI/strata-dense-build` | `strata-dense` — the engine binary `strata-dense-quetza` runs (source tree was `/tmp/strata-dense-eval`, now gone) |
| `~/AI/strata-dense-runtime` | `serve/` + tokenizer for the dense wrapper — **byte-identical copy of the repo's `serve/`** (only `__pycache__` differs), i.e. nothing to merge |
| `~/AI/ds4-ref` | scratch reference |

---

## 3. Merge topology — what was merged, in what order, and the trap

Order of work on `xolotl` (each is a commit, §9):

1. **`487cd576`** scaffold (merge scope + driver).
2. **`4a39e65b`** the graft point: Xolotl's history is joined to the fork's so the two lineages share
   a common ancestor.
3. **`86bd1ec1`** merge **`up/dense-eval`** (upstream): Qwen Flash + Qwen 27b dense + `/v1/decide`
   voice turns + the whole `serve/` frontend layer; 345 upstream commits, **48 conflicts**.
4. **`08597293`** merge **`origin/glm`**: GLM-5.3 + abliteration LoRA + elastic cache; **13 conflicts**.
5. `55a1ac38`, `0ae26656` — README restore, the criss-cross note.
6. Build-gate commits: `4b9bdf87`, `5587b517`, `8c69e9b2`, `84a1e42d`, then this session's
   **`b108d171`** and **`7012b147`**.

### 3.1 ⚠ THE CRISS-CROSS (cost the previous session hours)

`git merge-base --all xolotl glm` returns **TWO** bases, so git synthesises a **virtual base that
itself contains conflict markers** — verified: the stage-1 blob of `serve/server.py` held
`<<<<<<<<< Temporary merge branch 1`. A plain merge of glm therefore reported **51** conflicts
including all of `serve/`, which glm never touched.

* **Always use `git merge -s resolve`** here → the real count (13 for glm).
* `core.useReplaceRefs=false` does NOT avoid it.

---

## 4. Build

### 4.1 Configure (the line that works)

```bash
cd ~/AI/Xolotl
cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_GGML_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_GGML_DIR=/home/mal/AI/Strata-DS4/third_party/llama.cpp -DSTRATA_DSV41_FQ_OP=ON
ninja -C build-xolotl          # log: /tmp/xolotl-build.log
```

* `STRATA_GGML_DIR` **must** point at `~/AI/Strata-DS4/third_party/llama.cpp` (175 M, present, and it
  **has** the DSV41 fake-quant op so `-DSTRATA_DSV41_FQ_OP=ON` is valid). The clone's own
  `third_party/llama.cpp` is an **uninitialised submodule** (empty).
* Handy: `ninja -C build-xolotl -k 0` to collect ALL errors in one pass instead of stopping at the
  first. That is how bug 10 was found (it was silent, not a compile error).
* Effective options in this build: `STRATA_MMQ_KQUANTS=ON`, `STRATA_ENABLE_CUDA=ON`,
  `STRATA_DSV41_FQ_OP=ON`; **`STRATA_MMQ_DENSE_TYPES` is never set anywhere** (so those `#ifdef`s are
  inert); `STRATA_ORCA_Q4KS_MMQ` OFF, `STRATA_Q6K_EXPERTS` OFF, `STRATA_HIP_GFX906` OFF.

### 4.2 Result

`ninja`: **no work to do**. All targets build; all four engines link. Warnings only.

---

## 5. The build gate — bugs 1–10, all FIXED

The build/run gate did its job: every one of these was a real merge defect. "Mechanism" = why the
merge produced it.

| # | Where | Symptom | Mechanism / fix | Commit |
|---|---|---|---|---|
| 1–3 | `CMakeLists.txt` / `tools/ds4/cmake/fork.cmake` | configure FATAL: duplicate targets `ds4_generate`, `ds4_mtp_bench`, `ds4_expert_parity`, `ds4_moe_replay` | GLM's branch had the engine-slice includes **inline**; the fork moved them into `tools/ds4/cmake/fork.cmake` (which also aggregates `v41_engine.cmake`). Only `tools/glm/cmake/glm.cmake` is not in fork.cmake. Dropped GLM's copies. | `4b9bdf87` |
| 4 | `src/kernels/cuda/shared_expert.cu` | `error: identifier "f16_from_f32" is undefined` | It includes `bf16_bits.hpp` (which only *mentions* `f16_from_f32` in a comment) but not `f16_bits.hpp`. Added the include. `dequant_bf16.cu` has the same gap but reaches it transitively — **left alone; re-check if the build complains**. | `5587b517` |
| 5 | `CMakeLists.txt`, `src/prefill/moe_mmq.cu` | `fatal error: strata/prefill/mmq_types.inc: No such file or directory` | `cmake/mmq_types.cmake` provides the generator `strata_gen_mmq_types`, but **no branch ever included or called it**, and `mmq_types.inc` has never existed in git — while the merged `moe_mmq.cu` (from the dense lineage) `#include`s it. Wired it in. | `8c69e9b2` |
| 6 | `src/prefill/moe_mmq.cu` | 20 × `case label value has already appeared in this switch` | The dense lineage's refactor left a hand-written case block that is a **superset** of the generated `STRATA_MMQ_TYPES` expansion, and it also duplicated ITSELF (Q2_K/Q3_K once in "the UD packs" and again in the MiMo line; Q4_K/Q5_K/Q5_1 in two blocks; Q6_K in three). **The dense lineage's own file has the same defect → that branch is a refactor in progress, not a working state.** Reduced the block; then (§5.1) removed it entirely. | `8c69e9b2`, `b108d171` |
| 7 | `src/kernels/cuda/iq_kernels.cu` | LINK: `undefined reference to native_expert_grouped(..., const NativeExpertLora*)` | The merge's **"take ours"** on this file dropped GLM's routed-expert LoRA. `iq_kernels.hpp` declares a **14-arg** `native_expert_grouped` (with `lora = nullptr`); the merged `.cu` defined a **13-arg** one — a *different function* — so the declared one was undefined. GLM's block is **not** a superset of ours: it predates the fork's PROM kernels (`prom_gu`/`prom_down`/`prom_large`), `exp_mode_type` and the PTQ1_0/Q3_K types. Fixed as a **real union** (4 hunks into OUR body): `lora_down_kernel` + `native_expert_lora_down`, the `lora` parameter, the down-correction tail after `check("native_expert_grouped/down")` (opt-out `GLM_LORA_NO_DOWN`), and `sw_v1 = v1 \|\| (lora && lora->a_d)` in the non-gfx906 branch (the down LoRA needs the **float `h`**, which only the unfused SwiGLU pass writes). | `b108d171` |
| 8 | `src/prefill/moe_mmq.cu` + `CMakeLists.txt` | LINK: `undefined reference to mul_mat_q_case<(ggml_type)2/3/6/19>` (Q4_0, Q4_1, Q5_0, IQ1_S) | The dense lineage's hand-written list named Q4_0/Q4_1/IQ1_S, written against a build that **GLOBbed every template instance**; the fork's CUDA branch lists its instances **explicitly** and its file never mentioned them. The merge took the union of the *lists* but only the fork's instance set. Fixed by adding `q4_0 q4_1 iq1_s` to `_strata_mmq_cuda_types` and deleting the hand-written block — which is the generator's own contract: **advertised == instantiated**. | `b108d171` |
| 9 | `tools/v41/engine/ds41_moe.cpp:1590` | `error: 'struct ...::Gpu' has no member named 'cache'` (CPU flavour only) | `set_cache_enabled` uses `im_->gpu->cache` unguarded; the **CPU flavour's stub `Gpu` has only `close()`**. Dormant in the fork because `v41_engine.cmake` creates `ds41_moe_cpu` only `if(TARGET strata_kernels_cpu AND TARGET ggml-cpu)` and **the fork's configure has no `ggml-cpu`**; the merged dense lineage enables it, waking the CPU tier. Guarded with `#if defined(DS4_MOE_CUDA)`. | `b108d171` |
| 10 | `CMakeLists.txt:1704 vs 1707` | **No GLM engine at all** — silently. `glm_dense.a` built; **no `glm_generate` binary existed** | `glm.cmake` creates `glm_generate` only `if(TARGET ds4_moe_cuda)`, and `ds4_moe_cuda` is created by `fork.cmake`. The merge left `glm.cmake` **before** `fork.cmake`, so the gate was always false and the configure succeeded silently. Swapped the includes (glm.cmake's own header says it must come "AFTER the tools/ds4 slices it builds on"). Now `glm_generate` builds (681 MB). | `7012b147` |
| **11** | `src/kernels/cuda/fork/fork_fmt.cuh:24` | **Runtime, not compile:** `glm_generate: tier init: native_expert_grouped has no kernel for layer 3's types 16/16/18` — GLM-5.3 **could not load at all** | The **fork-format list lost three types**. GLM's branch kept its copy INLINE in its own `iq_kernels.cu` (line 736) with `X(12) X(18) X(22)` — Q4_K down_exps for GLM-5.3 layers 3-5, and the IQ2_S/IQ3_XXS Maya-S-v2 down_exps; the fork had meanwhile extracted this list into `fork_fmt.cuh` with only `X(10) X(11) X(39) X(143)`. The merge reconciled GLM's inline copy to the extracted header and dropped 12/18/22 → `d_qk(18) == 0` → `native_expert_supported(16, 18, 4096, 2048)` false. Nothing referenced 18 before, so it still compiled and linked. **Fixed by the union** (§6). | *uncommitted at handoff* |

### 5.1 The one-file outcome of 5/6/8

`src/prefill/moe_mmq.cu` `supported()` and the dispatch switch are now **one generated list**
(`STRATA_MMQ_TYPES`) with only a `default:` remaining. The former hand-written types became
CMake list entries, so `supported()` and `strata_mmq`'s compiled instances cannot drift — which is
exactly the invariant the file's own comment says costs **100x on prefill** when it breaks
(`docs/UNSLOTH_Q4.md`, `#296`).

Note also: the generated file now lands in **`<build>/generated/strata/prefill/mmq_types.inc`**, not
`<build>/strata/…` — at `<build>/strata` it created a **DIRECTORY** named `strata`, and
`add_executable(strata ...)` then failed to link with `cannot open output file strata: Is a directory`.

---

## 6. Bug 11 — GLM tier init rejected the model. DIAGNOSED, FIX APPLIED, **REBUILD NOT DONE**

### 6.0 Verdict: it IS a merge regression, and the cause is a single line

**Measured, not inferred.** Run A (below) — the *working* `~/AI/Strata-GLM/build-glm-gpu/glm_generate`
(2026-10-10) with the same args — passes the point where the Xolotl binary dies, and prints a line the
Xolotl binary never reaches:

```
moe lora: routed-expert deltas on 26 layer(s) (n_experts 288, n_embd 4096, n_ff 2048)
```

**Cause — `src/kernels/cuda/fork/fork_fmt.cuh:24`, `STRATA_D_FMTS_FORK`:**

```
GLM  (inline, its iq_kernels.cu:736):  X(10) X(11) X(12) X(39) X(18) X(22)
ours (the extracted header)         :  X(10) X(11) X(39) X(143)
```

Per GLM's own comment on that line: **12 = Q4_K down_exps (GLM-5.3-Flash layers 3-5; upstream's list
has none)**, and **18/22 = the Maya-S-v2 down_exps (IQ2_S / IQ3_XXS)**, 39 = MXFP4 (MiMo), 143 =
PTQ1_0 (the V4.1 ternary tier). The fork had moved this list into `fork_fmt.cuh` (to keep upstream's
lists verbatim across merges); GLM's branch still carried it **inline**. The merge reconciled GLM's
inline copy to the extracted header and kept only the fork's entries, so `d_qk(18) == 0` and
`native_expert_supported(16, 18, 4096, 2048)` returns false for every Maya-S-v2 layer (16/16 =
IQ2_XXS gate/up, 18 = IQ2_S down). **GLM-5.3 therefore could not load at all**, while the DS4/DSV4.1
types were untouched.

Why the build gate could not catch it: the list only gates the kernel `switch` cases and that
predicate. Nothing in the compiler references type 18, so it compiled, linked and passed ctest.
**This is the class of defect only running the model finds.**

### 6.0.1 The fix (applied in the working tree, one line)

```c
#define STRATA_D_FMTS_FORK(X) X(10) X(11) X(12) X(18) X(22) X(39) X(143)
```

`Fmt<12>`, `Fmt<18>` and `Fmt<22>` are **all already defined** in this tree (verified: 1 definition
each), so no port from GLM is needed — this is the whole fix. `STRATA_GU_FMTS_FORK` and
`STRATA_MMVQ_FMTS_FORK` need nothing: the fork's versions are already supersets of GLM's.

**STATUS AT HANDOFF:**

* ✅ **Rebuild DONE and GREEN** — `ninja -C build-xolotl` finished **64/64, 0 errors** at **06:02:43**
  (`iq_kernels.cu.o` written 06:02:34, `glm_generate` relinked 06:02:43). The fix **compiles and links**.
* ❌ **Runtime NOT verified.** The A/B job's command B ran **06:02:27 → 06:02:32** — *before* both of
  those timestamps (proven by `stat`), so it exercised the **OLD binary**: its `types 16/16/18` failure
  is the *pre-fix* behaviour, **not** a counter-example to the fix. Do not read it as one.
* **Next session: re-run §6.4 command B** (and use `--vram-margin 7.5`, see §6.5) — the tier-init error
  should be gone and tokens should come out. That is the last known blocker for serving GLM-5.3.

### 6.0.2 The general lesson (worth writing down)

**The `*_FMTS*` X-macro lists are the most merge-sensitive surface in this kernel file.** The fork
extracted its formats into `fork_fmt.cuh` with "kept out of upstream's lists above so those stay
verbatim (fewer merge conflicts)" — the *opposite* trade-off from GLM, which kept its 3 extra entries
inline. A `-s resolve` merge silently prefers one side. **Whenever a new model is added to the merge,
diff every `STRATA_*_FMTS*` definition against every branch that touched it** — a lost type is a
runtime tier-init failure, not a compile error:

```bash
cd ~/AI/Xolotl && for b in origin/glm origin/deepseek4 up/dense-eval; do echo "== $b"; git grep -h "define STRATA_.*_FMTS" $b; done
grep -rn "define STRATA_.*_FMTS" src/ | sed 's/^/ours /'
```

### 6.1 The failure, verbatim

Real run of the **Xolotl** `glm_generate` (GLM-5.3 Maya-S-v2 + the abliteration LoRA):

```bash
cd ~/AI/Xolotl/build-xolotl && ./glm_generate \
  -m "/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf" \
  --backend cuda --experts gpu --arena-gib 12 --ctx 8192 \
  --census /home/mal/.quetza-data/strata-glm/maya-s-v2.census \
  --lora "/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf" --lora-exps \
  --ids 100,200,300,400,500,600,700,800 -n 8 --temp 0
```

progresses correctly — CUDA init, `dense backend: CUDA0`, LoRA parsed ("routed-expert exps are NOT
applied yet (CPU-path slice pending)" for the *dense* half), **dense half loaded in 4.5 s**, 45 layers
(34 KDA, 11 MLA), 288 experts top-8, vocab 154880, "lora: routed-expert deltas for **26 layer(s)**",
slots auto → 15.44 GiB free → 14.44 GiB of slots, census 12060 counts, "lora-exps: pcie_frac 0.25" —
then, at **tier init**:

```
glm_generate: tier init: native_expert_grouped has no kernel for layer 3's types 16/16/18
```

`16 = IQ2_XXS` (gate), `16 = IQ2_XXS` (up), `18 = IQ2_S` (down). The message is emitted from
`tools/ds4/ds4_moe.cpp:772` (the same string also in `tools/v41/engine/ds41_moe.cpp:1326`).

### 6.2 What is already ruled out

* The check goes through `native_expert_supported3(gate, up, d, n_embd, n_ff)` →
  `native_expert_supported(16, 18, …)` (`src/kernels/cuda/iq_kernels.cu:2812`). **Its body is
  byte-identical to `origin/glm`'s** (`2990`), and so are its helpers `is_iq` (`:1926`; ours only
  *adds* `t == 143` PTQ1_0), `gu_qk` (`:1936`) and `d_qk` (`:1945`). For 16/18 both return true.
  Ours and GLM's `is_iq` both return true for 16 and 18. **So the loss is not in these functions.**
* The engine reached tier init, so the LoRA plumbing I restored (§5 bug 7) is at least constructed
  and the routed-expert deltas were attached for 26 layers. **The failure is a support/type-table
  question, not the LoRA overload.**

### 6.3 How the verdict was reached (keep this method)

Two hypotheses were on the table: **(A)** a lost entry in the `STRATA_*_FMTS*` X-macro lists, or
**(B)** my argument set differing from the wrapper's (which adds `--prefill-chunk 4096 --chunk-mmq
--arena-skip-resident --vram-margin 7.5 --vram-grow 1.5 --prune … --arena-adapt --skip-*`). Rule out
(B) by running the **same command against the working binary**; it passed, which is (A). Then the
macro diff in §6.0.2 pinpointed the line. Note for the record: `is_iq`, `gu_qk`, `d_qk` **bodies** are
byte-identical to `origin/glm`'s — the loss was in the macro *they expand*, which is why reading only
the functions (the earlier session's instinct) would have missed it.

### 6.4 Commands — re-measure and verify

```bash
M=/media/mal/SSD\ NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
L=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
C=/home/mal/.quetza-data/strata-glm/maya-s-v2.census

# A) the WORKING reference binary (control)
cd /home/mal/AI/Strata-GLM && ./build-glm-gpu/glm_generate -m "$M" --backend cuda --experts gpu \
  --arena-gib 12 --ctx 8192 --census "$C" --lora "$L" --lora-exps \
  --ids 100,200,300,400,500,600,700,800 -n 8 --temp 0
  # passes tier init; prints "moe lora: routed-expert deltas on 26 layer(s) (n_embd 4096, n_ff 2048)"

# B) the XOLOTL binary - after `ninja -C build-xolotl`. Add the wrapper's flags to be faithful:
cd /home/mal/AI/Xolotl/build-xolotl && ./glm_generate -m "$M" --backend cuda --experts gpu \
  --arena-gib 24 --arena-skip-resident --ctx 8192 --prefill-chunk 4096 --chunk-mmq \
  --vram-margin 7.5 --vram-grow 1.5 --arena-adapt --skip-miss 0.15 --skip-file 0.15 \
  --skip-file-prefill 0.15 --renorm-skip --pf-b 1.0 --pcie auto --census "$C" \
  --lora "$L" --lora-exps --ids 100,200,300,400,500,600,700,800 -n 8 --temp 0
```

Take `~/AI/.quetza-data/conductor/ds4-gpu.lock` first (`flock -n <lock> true` to fail fast instead of
blocking) — the GLM/DSV4.1 wrappers use it. Kill a stray run with **`pkill -x glm_generate`**
(never `pkill -f`, see §10.3).

### 6.5 What the A/B job actually produced (measured — read `~/AI/Xolotl/glm-tierinit-ab.log`)

**A — the working `~/AI/Strata-GLM/build-glm-gpu/glm_generate` (2026-10-10), same minimal args:**
it **passes tier init** —

```
moe lora: routed-expert deltas on 26 layer(s) (n_experts 288, n_embd 4096, n_ff 2048)
tier: gpu, 2176 resident slots, arena 12.0 GiB (1813 experts), file tier 10283; load 132.0 s
```

then dies on **a VRAM OOM of my own making**:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4.22 MiB on device 0: cudaMalloc failed: out of memory
glm_generate: prefill failed at 0: gallocr(attn) failed
```

**That OOM is a HARNESS artifact, not a bug:** `--arena-gib 12` leaves the Dense/attn graph no room
(`slots auto: … margin 1.00 GiB`). The GLM wrapper runs this model with **`--arena-gib 64
--vram-margin 7.5`**. So when re-running, either raise the arena or pass `--vram-margin 7.5` — B's log
shows the engine itself reserving the margin correctly (`slots auto: 14.69 GiB free → 7.19 GiB of
slots (margin 7.50 GiB)`).

**B — the Xolotl binary:** `tier init: native_expert_grouped has no kernel for layer 3's types 16/16/18`
— the **pre-fix** binary (§6.0.1). It also shows the wrapper's real flag effects: `lora-exps:
pcie_frac -1.00`, `prune: 3024 (layer, expert) pairs masked out of routing`.

The script is at `/tmp/glm-tierinit-ab.sh` (**/tmp — may not survive a reboot**; §6.4's commands are the
self-contained copy). It is finished; nothing is left running. The GPU is free.

---

## 7. Tests

### 7.1 How to run

```bash
cd ~/AI/Xolotl/build-xolotl && ctest -j2 --output-on-failure     # 54 tests, ~55 s
```

### 7.2 Result: 53/54 pass

`98% tests passed, 1 tests failed out of 54`. The failure is **environmental, not a regression**:

```
43 - ple_parity ***Failed 0.01 sec
ple_parity: required PLE table is missing or incompatible:
  ../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf (cannot open …)
```

That GGUF is a **model fixture**, downloaded by `tools/test_setup_modelscope.py` (paths in
`tools/test_setup_golden.json`). It is absent from this machine (`find` over `/media/mal` and
`/media/Crucial1TB` found nothing). The test's wiring — `--selftest --in/--out bench/micro` with
`WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}` — is **byte-identical in ours, in `origin/deepseek4`
and in `up/dense-eval`**, so the merge did not touch it. To make it pass: fetch the fixture, or run
with `-DSTRATA_PLE_FIXTURE_DIR=<dir>` / set the `STRATA_PLE_GGUF` path. **Nothing to fix in code.**

All GPU tests pass, including `cuda_device_selftest`, the `kv_*`, `gr_*`, `iq_*`, `native_*`,
`qsa_*`, `s2_*`, `verify_*`, `rope_parity`, `sampler_parity`.

### 7.3 Engines built and smoke-tested (start-up)

| binary | model line | `--help` |
|---|---|---|
| `build-xolotl/strata` | Qwen Flash-Next MoE **and** Qwen 3.8 27b dense (`src/program/generate.cpp` + `serve/`) | `strata generate --pack DIR --tokens …` |
| `build-xolotl/glm_generate` | GLM-5.3-Flash (681 MB) | `glm_generate -m model.gguf (--ids …)` |
| `build-xolotl/ds41_generate` | DeepSeek-V4.1-Flash | "unknown argument --help" + pointer |
| `build-xolotl/ds4_generate` | DeepSeek (DS4 engine line) | full usage |

All four start and print usage (exit 0). The GLM end-to-end run is the one in §6.

---

## 8. Wrappers and model artifacts — what runs what TODAY

None of these wrappers point at Xolotl yet. They are the **reference implementations** to compare
against (§11.4).

| wrapper (`~/.local/bin/`) | tree it uses | engine | config | model weights | port / ctx |
|---|---|---|---|---|---|
| `strata-quetza` | `~/AI/Strata` | `build/strata` + `serve/server.py` | `strata-swift-iq3_xxs{,-sc117}.json` | Qwen3.8 Flash-Next (Swift, IQ3_XXS) | 8130 unc / 8128 stock, ctx 262144, VISION=lazy |
| `strata-quetza-refusal`, `strata-quetza.bak-20261005` | — | variants of the above | — | — | — |
| `strata-dense-quetza` | `~/AI/strata-dense-build` | `strata-dense` + runtime `~/AI/strata-dense-runtime` | `~/.quetza-data/strata-dense/strata-dense-quetza.json` | `/media/mal/NVME1TB/Models/Swift-1.5-27B-GSQ-RCO/d0xin-unc-27b-IQ3_S-mtp-q6embd.gguf` | 8090, ctx 200000, KV `k8v4`, MTP draft 2 |
| `strata-ds41-quetza` | `~/AI/Strata-DS4` | `build-ds4-gpu/ds41_generate --serve` via `tools/v41/serve/serve_ds41.py` | `tools/v41/serve/strata-ds41.json` | V41 GSQ (see §12) | 8150, ctx 524288 |
| `strata-glm-quetza` | `~/AI/Strata-GLM` | `build-glm-gpu/glm_generate` via `tools/glm/serve/glm-engine.sh` | `tools/glm/serve/strata-glm-*.json` | Maya-S-v2 IQ2_XXS 3-shard (default `= --maya`) + abliteration LoRA | 8140 unc / 8141 stock / 8142 max / 8143 maya300 / 8144 maya512 |

### 8.1 GLM-5.3 artifacts (used in §6)

```
model   /media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf
        (47.8 GB + 47.8 GB + 0.84 GB = 96 GB, 3 shards)
LoRA    /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf  (277 MB)
prune   ~/AI/Strata-GLM/bench/glm-2026-10-09/prune-ezct-0.25.txt   (penalty 0.05)
census  ~/.quetza-data/strata-glm/maya-s-v2.census                 (12060 counts)
tokenizer /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer
```

### 8.2 Resources / locks

* **GPU lock:** `~/.quetza-data/conductor/ds4-gpu.lock` — `flock`; the GLM/DSV4.1 wrappers hold it
  while serving. `flock -n <lock> true` to test, i.e. fail fast instead of blocking.
* GLM/Dense wrappers refuse while **ComfyUI** listens on `:8188`, and cap RAM (84 GiB, swap 0) via
  `glm-engine.sh`.

---

## 9. `xolotl` commit log (12 commits on the merged base)

```
7012b147 xolotl: include fork.cmake BEFORE glm.cmake - the merge built no GLM engine (build gate, bug 10)
b108d171 xolotl: complete the merge - LoRA overload, one MMQ list, the CPU tier (build gate, bugs 7-9)
84a1e42d xolotl: HANDOFF - state, the remaining LoRA-overload bug with its diagnosis, traps, resume commands
8c69e9b2 xolotl: generate mmq_types.inc and de-duplicate its case labels (build gate, bugs 5-6)
5587b517 xolotl: shared_expert.cu needs f16_bits.hpp (build gate, 4th merge bug)
4b9bdf87 xolotl: fix the 3 configure failures the build gate found (duplicate targets)
08597293 xolotl: merge glm (GLM-5.3 + abliteration LoRA + elastic cache) - 13 files, all two-sided
0ae26656 xolotl docs: the graft creates a CRISS-CROSS - use -s resolve (13 glm conflicts, not 51)
55a1ac38 xolotl: restore the DS4 section the README merge dropped
86bd1ec1 xolotl: merge upstream dense-eval (Qwen Flash + 27b dense + /v1/decide voice) into DSV4.1
4a39e65b xolotl: the graft point, the real cross-lineage cost, upstream's 345 new commits, prior art
487cd576 xolotl: scaffold the Strata-family unification (scope + merge driver)
```

`MERGE_SCOPE.md` in this repo holds the pre-merge scope, topology, both conflict tables and prior art.

---

## 10. Traps — each of these already cost time once

1. **The graft creates a CRISS-CROSS** (§3.1). Use `git merge -s resolve`. A plain merge lies about
   the conflict count and invents conflicts in files the other branch never touched.
2. **DO NOT DELETE THE OLD WORKTREES.** `Strata-qwen` holds the **voice sidecar UNTRACKED**
   (`serve/voice.py`, `tools/audio/`) plus uncommitted `serve/server.py` / `src/program/generate.cpp`
   — it exists **nowhere else**. `Strata-GLM` (4 files), `Strata-GLM-rebase` (382 staged),
   `Strata-GLM-um` (1) also have uncommitted work. Mal's call: leave them until Xolotl is proven.
   **Worktrees share `.git`, so commits survive — uncommitted work does not.**
   Only `git worktree prune` has been run (removed one dead `/tmp/strata-dense-eval` registration).
3. **`pkill -f "<pattern>"` kills your own shell** when the pattern is in your command line (cost the
   previous session exit 144). Use `pkill -x ninja`.
4. **Disk is at 98 % (~19 G free, down from 26 G).** The clone is sparse (excludes `bench/`) for this
   reason. The bug-8 MMQ instances and the bug-11 rebuild cost ~7 G. Adding more instances, or a second
   build dir, will eat it — watch `df -h /` before big builds.
5. **`ninja` stops at the first error by default** — the HEADER of the error may be a *consequence*.
   Use `-k 0` to see everything.
6. **A silent CMake failure is possible** (§5 bug 10): the configure can "succeed" while a whole
   engine binary is never created. **Check for the artifact, not the exit code.**
7. **Two things provide MMQ instances**: llama.cpp's own `ggml-cuda` target compiles the full
   template-instance set, and `strata_mmq` compiles the CMake subset (the build tree really does
   contain e.g. two `mmq-instance-q4_k.o`). Only `strata_mmq`'s set is what `moe_mmq.cu` must match.
8. **`up/dense-eval`'s `moe_mmq.cu` does not compile as committed** (its hand-written case block
   duplicates itself and the generated list). Do not treat that branch's file as a working reference.
9. **`glm.cmake` include order matters** (§5 bug 10) — after `fork.cmake`, always.
10. **Enabling `ggml-cpu` wakes the fork's CPU flavour**, which had a latent compile error (§5 bug 9).
    Expect more of that class: code paths the fork never compiled are now live.
11. **Path with a space**: `/media/mal/SSD NVME/…` — quote it.
12. **The `STRATA_*_FMTS*` X-macro lists are the most merge-sensitive surface in `iq_kernels.cu`** (§6.0.2).
    The fork extracted its formats into `fork_fmt.cuh` ("kept out of upstream's lists above so those stay
    verbatim"), GLM kept its three extra entries **inline** — a `-s resolve` merge silently prefers one
    side, and a lost type is a **runtime tier-init failure, not a compile error**. Diff them against every
    merged branch whenever a new model joins the tree.
13. **`pkill -f` self-kills even in a `;`-chained command** — it happened *again* in this session writing this
    very document (exit 144), even after the trap was written down. `pkill -x <name>`; for the GLM smoke
    test `pkill -x glm_generate`.
14. **Loading GLM-5.3 at a small `--arena-gib` produces a confusing VRAM OOM** (§6.5) that looks like a
    merge bug. The model's real configuration is `--arena-gib 64 --vram-margin 7.5`.

---

## 11. Open decisions for Mal

1. **Push `xolotl` over `Indras-Mirror/Strata-DSV4.1`** (as `main`) once it builds and serves — he
   wants this. The graft must be made **permanent** first: `git replace` refs do **not** push. Either
   `git filter-repo --replace-refs update-no-add`, or redo the merge as `--allow-unrelated-histories`.
2. **Drop the legacy `tools/ds4/ds4_*` engine?** It is still built, and it is what made the MiMo merge
   cost 71 hunks.
3. **MiMo: in or out** (or a `--legacy` build target)? Resume with:
   `git merge --no-commit --no-ff -s resolve origin/mimo`.
4. **Then, per model: serve/test legs against the current wrapper**, and only then clean up the
   worktrees (§10.2).
5. **The §6 runtime failure** — decide whether to chase it now or after the push.

---

## 12. Already-known facts — do NOT re-derive

* **FQ_OP is genuinely live** in the DS4 build: `-DSTRATA_DSV41_FQ_OP=ON`, the llama.cpp tree patched,
  and `--kv-fp4` requires `act_quant` (`ds41_dense.cpp:1050`) which the serve config sets.
* The DSV4.1 prefill work is **committed and pushed** on `deepseek4` (`14e34b19`, `9d6529fc`): batch
  12288 default, the shared-card VRAM cap, partial unpark. **The 16384 blocker is NAMED**: the batch
  holds ~8.0 GiB of fixed buffers, the OOM was a transient **1424.3 MiB graph buffer**, and the
  ~1.05 GiB deficit is recoverable from one **speed** knob (`STRATA_PRESTAGE_GIB` /
  `MIMO_CHUNK_STAGE_MIB`) with **no quality gate**.
* **DSV4.1 artifacts**: weights `/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf`; engram
  `/media/Crucial1TB/V41-Engram : /media/mal/SSD NVME/AI/V41-Engram`; rank
  `/media/mal/SSD NVME/AI/V41-GSQ/rank.bin`; DSpark sidecar
  `/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf`.
* **`~/AI/v41-dev` is 82 G of bench/logs** — but `v41-dev/s13/run3/p20k.i32` is the **20 480-token
  prompt the harnesses use; keep it.**
* `strata-dense-runtime` is a **byte-identical copy of the repo's `serve/`** — nothing to merge.
* **`strata-dense-build`'s source tree was `/tmp/strata-dense-eval`, which is gone.**

---

## 13. Session-2 summary, in one paragraph

The merge is no longer "half-built": the tree configures, builds all four engines, and passes 53/54
ctest (the one failure is a missing model fixture). **Eleven** real merge defects were found — ten
build-gate ones (the LoRA overload, the MMQ case lists, the never-generated header, a latent
CPU-flavour compile error the merged config woke up, and a *silent* one that removed the GLM engine
entirely), and an eleventh that only *running the model* could find: the fork-format list lost GLM's
`Q4_K / IQ2_S / IQ3_XXS` down types, so GLM-5.3 died at tier init with `types 16/16/18`. The fix is one
line, it compiles green, and the A/B control run **proved it is a regression** (the working binary
passes that exact point). What remains is one command: re-run §6.4's B against the rebuilt binary and
watch the tier init pass. Nothing is pushed; nothing outside `~/AI/Xolotl` was modified; the GPU is
free.

**Claims I would most expect to be wrong, stated plainly:** (1) the bug-11 fix has only been
**compiled**, not run — the runtime check is the single remaining step; (2) `ple_parity` fails for a
missing model fixture, which I verified is absent from this machine but did **not** fix; (3)
`Strata-DS4-private` and `Strata-data` were never inspected this session.
