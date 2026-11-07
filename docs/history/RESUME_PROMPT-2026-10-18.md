# RESUME PROMPT — Xolotl — START HERE (2026-10-18 ~11:20, session 7, unified across both session-7 windows)

You are resuming a multi-session project for **Mal** (Malich Coory, Indra's Mirror): **Xolotl** (`~/AI/Xolotl`, branch
`xolotl`, public as github.com/Indras-Mirror/Xolotl) - one local inference engine (a heavily extended Strata fork) that
runs DeepSeek-V4.1-Flash (552B MoE, our GSQ quant), GLM-5.3-Flash and the Qwen3.8 models on ONE RTX 4090 (24 GB) +
Ryzen 7 5700X + 90 GB DDR4.  Measured = measured on this computer; don't re-derive it.  Read this block, then the S7
log at the top of `HANDOFF.md`, then `docs/ds4/V41_1017_QUANT_PLAN.md`, `docs/ds4/V41_TERNARY_PROFILE_PLAN.md`.
Everything below the session-7 block is history.

> **GIT (shared worktree):** several sessions share `~/AI/Xolotl`.  Commit named files only; never `git add -A`,
> `git add .` or `git commit -a`.  If you inherit a commit whose message does not match its contents, do not rewrite
> history - record the remaining files in a named follow-up commit.
>

## 0. What is running (check `ps` + `nvidia-smi` before using the GPU; nothing else of ours runs)
```bash
ps -eo pid,etime,args | grep -E "[f]etch_1017|[f]etch_v41|[c]alib_layers|[q]uant_layers|[w]eight_edit|[d]s41_generate" | cut -c1-150
tail -3 /home/mal/v41-work/fetch_1017.log "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017/fetch-engram.log"
```
- **1017 download, ONE at a time (Mal's rule):** Engram shards -> Team CX2 (47 verified 09:29; 48 at 94.8/101.5 GB at
  11:12), then **`tools/v41/fetch_1017.sh`** (armed, waits for the Engram loop pid 1783870; log
  `/home/mal/v41-work/fetch_1017.log`): small files + shards 1-39 -> `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017`
  (MAIN), shards 40-46 -> `/media/mal/Crucible/models/DeepSeek-V4.1-Flash-1017`.  ~307 GB at ~14-20 MB/s = ~5-6 h.
  All tools: `V41_SRC="/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017:/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017:/media/mal/Crucible/models/DeepSeek-V4.1-Flash-1017"`.
  After a reboot: the command in `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017/RESUME-DOWNLOAD.txt`.
- The GPU is FREE (no engine, no calibration).  Finished today: Gauntlet (Q6_K), think diag (see 1), swap test (see 1).

## 1. What session 7 established (measured)
- **THE SERVED DEEPSEEK IS GSQ v2 (since 11:05, Mal)**: `/media/mal/SSD NVME/AI/V41-GSQ-v2/V41-GSQ-v2.gguf` + its own
  `rank.bin` (+ experts.json); `strata-ds41.json` (+ bias03/bias025) repointed (dfa0ea32) -> the `strata-ds41-quetza`
  wrapper, Gauntlet lane and scripts all use v2; smoke-tested 11:10 (loads ~90 s, correct answer).  **v1 GGUF deleted**
  (still on HF: IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored); no v2 copy on NVME1TB any more.  Tokenizer stays in
  `SSD NVME/AI/V41-GSQ/tokenizer`.  v2 stays until the 1017 quant replaces it (serve that from SSD NVME too).
- **v2 vs v1 on the SAME drive (swap test 10:10-10:33, `verify/s7/v2s-speed.log` vs `verify/s6/v1v2r-speed.log`):**
  prefill 184.5-187.6 / 292-296 vs 177-178 / 280-281 (+5 %), decode after 6K ~= , after 16.5K 23.4-23.8 vs 20.3-22.5,
  edits 42.3-45.0 vs 37.6-39.8 (+12 %).  The earlier "v2 slower" was the exFAT drive: O_DIRECT expert reads exFAT
  NVME1TB 0.96 GB/s vs ntfs3 SSD NVME 5.1-5.3 (raw NVME1TB 3.31).  **Never serve/bench a GGUF from exFAT.**
- **Ternary needs no kernel work for now:** the prompt-path PTQ1_0 -> Q2_0 transcode = 52 ms per 6K prompt, 137 ms per 16K
  (~0.2 %, `MIMO_CHUNK_PROF`, build-s6); ternary-on-MMVQ (`STRATA_MMQ_PTQ1=0`) is worse (16K 248 tok/s);
  `STRATA_MMQ_TC_GIB=3` 16K 306 tok/s in ONE run (maybe +3-5 %, confirm x2 before adopting).
- **v2 quality vs v1** (s6): calib held-out ppl 5.603 vs 5.645 (better), top-1 0.6753 vs 0.6745; 100 chat-style API
  fixtures 85.76 vs 86.12 % (-0.36 pt), KL vs FP8 0.239 vs 0.229 (+4 %) - about half from the near-all-ternary band
  20-35.  No task benchmark (Gauntlet / LCB) was run on v2.
- **Q6_K-dense default on the Gauntlet v2** (`verify/e2e/gauntlet-v2-q6k/score.txt`, on v1): 5/7, SCORE 5.50, decode
  36.7 tok/s (v1 Q8_0 dense: 4/7, 6.20, 23.4).  14 = fail (Mal's call, over-verification).  42-56 tok/s on long thinking.
- **Prefix-cache bug** (not urgent): Gauntlet 14 turn 7 re-read 113,736 tokens ("reused 0") after the only turn whose
  thinking ran past the 27,648 cap.  Lead + files in HANDOFF S7.
- **Think diag**: both 40-layer chains ran (FP8 + GSQ on the quant's own 8K abc398_f thinking; FP8 ppl 1.674 / top-1
  0.840), but `think_diag.py head` crashed (`TypeError: generator + tuple`) -> no comparison.  Data kept in
  `/media/Crucial1TB/v41-thinkdiag/` (a small fix + head/compare re-run, no chain re-run).  The x0.3 bias test was
  cancelled (Mal).  Overthinking work is parked until the 1017 quant exists.
- Engram random 4K reads (32 thr): NVME1TB 10.3K/s, Team CX2 5.6K/s, Crucible 4.4K/s, ntfs3 SSD NVME ~290K/s.  The engine
  reads the EXTRACTED `engram-L1/L14.safetensors`, never the shards.
- Code (session 7): `V41_SRC`/`_URL`/`_NAME` source override; `v41_calib.py prep --mix`; `MIMO_CHUNK_PROF` transcode
  timer (build-s6 only); `tools/v41/fetch_1017.sh`.

## 2. Next, in order
1. **Tier map for the 1017: v2-style, CONFIRMED by Mal (10:45), but GENTLER in layers 20-35 (Mal 10:50: "happy to
   sacrifice a bit of size" for chat quality; agentic/tools/code first, but balanced).**  Sweep on the 1017 stats:
   `--ternary-cap-at "20-31,33-35=X"` for X = 1.0 / 0.85 / 0.75 against budgets -12 / -10 / -8 GiB (130.38 / ~132.4 /
   ~134.4 GiB); pick the smallest whose proxy <= v1-recipe AND whose held-out chat domain is not worse; then the API
   fixtures decide.
2. **1017 calibration set is BUILT** (10:50, `/home/mal/v41-work/calib-1017`): rebalanced mix agentic_long 8, agent 5,
   code 6, reason 3, web 2, zh 2, chat 4, multi 2 (V4.1 set kept in `calib-1017/v41-mix/`); same 32 x 8192 tokens.
   Calibration streams behind the download (layer L needs shard L+3; init needs shard 2 and is not shard-gated).
   Mal's call when to start.
3. Uncensoring is Mal's side and is kept outside this repo.
4. Then the rest of the 1017 plan (tier map, GGUF, act-order quant, Engram extraction to ntfs3/ext4, sidecar, rank,
   serve config) + validation (fixtures vs the 1017 FP8, Gauntlet, overthinking pairs).  **Serve the 1017 GGUF from SSD
   NVME** (it replaces v2 there), never NVME1TB/exFAT.
5. **HF card update note** - drafted 11:00 (text in HANDOFF S7), waits for Mal's OK.  Its Q6_K line needs the Xolotl
   commits pushed first (push = Mal's OK).  No v2 upload unless people ask (Mal).
6. Prefix-cache bug; think_diag head fix (optional); `STRATA_MMQ_TC_GIB=3` x2 confirm.
7. Later: mixed-precision dense (shared expert Q4_K + attention Q6_K), overthinking options (after 1017), vision slice 2.
8. **After the 1017 is served and validated: reformat NVME1TB exFAT -> ext4** (Mal's choice, not NTFS).  Unused 64 GB
   `swap64.img` on it (active swap = /swap.img, /swap2.img, zram).  Crucible is ntfs-3g (fuseblk, slow random reads).

## 3. Repo / disk
- github `main` = the `public` branch (worktree `~/AI/Xolotl-public`), published 2026-10-18 as one sync commit on top
  of 7375e305.  Day-to-day work happens on the private `xolotl` branch in `~/AI/Xolotl`; publish by syncing files into
  the public worktree and pushing from there (a pre-push hook blocks pushing anything else to github).
- Disk (11:12): system 106 GB, Crucial1TB 274 GiB (-> ~18 after its 1017 share), Crucible 84 (-> ~52), Team CX2 55
  (shard 48 finishing), NVME1TB 378 (exFAT: storage only), SSD NVME 20 (v2 + Engram L1 + GLM).  Deleted in s7: v1 GGUF,
  NVME1TB dealignai shard 47/48 duplicates + v2 copy, Crucial `MCP/` + `.pnpm-store`, stale Odin DB copies on Crucible
  (kept: live `~/AI/Odin/odin.db` + Team CX2 `odin-backups/odin_20261014_203542.db.zst`).

## 4. Standing rules (Mal) - as session 6 (section 4 below) plus
- **Public vs private (2026-10-18, Mal):** work in `~/AI/Xolotl` (branch `xolotl`, PRIVATE - never pushed; a pre-push
  hook refuses it).  GitHub gets the `public` branch from the `~/AI/Xolotl-public` worktree.  Publish with
  `xolotl-publish` (commits there; `--push` also pushes): it carries over changes since `git config xolotl.lastsync`,
  skips the private files and stops if a new line mentions private material.
- **One download at a time**; chain the next only with Mal's OK.  **Never serve/bench a GGUF from exFAT.**
- **Commit named files only** (box at the top).
- Outward text (Reddit/HF/GitHub/email) = show it, Mal posts/OKs.  Mal's r/LocalLLaMA thread (2026-10-18): quote measured
  numbers (quant thinks 3-12x the API's tokens on one-shot LCB problems, ~equal on the agentic Gauntlet; cause not
  separated; Engram cost = random IOPS, ~1 ms/token on NTFS NVMe vs ~10 ms on exFAT).

# Session-6 end block (2026-10-18 07:50) - history, superseded by the session-7 block above

You are resuming a multi-session project for **Mal** (Malich Coory, Indra's Mirror): **Xolotl** (`~/AI/Xolotl`, branch
`xolotl`, public as github.com/Indras-Mirror/Xolotl) - one local inference engine (a heavily extended Strata fork) that
runs DeepSeek-V4.1-Flash (552B MoE, our GSQ quant), GLM-5.3-Flash and the Qwen3.8 models on ONE RTX 4090 (24 GB) +
Ryzen 7 5700X + 90 GB DDR4 + NVMe.  Everything marked measured was measured on this computer - don't re-derive it.
Read this block, then the session-6 log at the top of `HANDOFF.md`.  Older blocks below are history.

## 0. First: who owns the GPU, and what is still running (started by session 6, all `setsid nohup`)
```bash
ps -eo pid,etime,args | grep -E "[g]auntlet-local-all|[m]orning-diag|[b]ias-quick|[f]etch_v41_loop|[f]etch_v41.sh|ds41_generate|[v]41_" | cut -c1-150
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader     # desktop alone ~0.4-0.9 GB
ss -ltnp | grep -E ":81(4[0-9]|5[0-9]|6[0-9]) "
```
1. **Gauntlet v2 on the NEW served default (v1 experts + dense Q6_K)** - `verify/e2e/gauntlet-v2-q6k.log`, result
   `verify/e2e/gauntlet-v2-q6k/score.txt` (+ `ds41/throughput-*.json`).  Compare with v1 at Q8_0 dense
   (`verify/e2e/gauntlet-v2/score.txt`): **ds41 4/7, SCORE 6.20, decode 23.4 tok/s** (API 5/7 6.41).  Started 06:25,
   was on the last task (14) at 07:44.  If Q6_K holds 4/7-ish and decode is up, it stays the default.
2. **`verify/s6/morning-diag.sh`** (log `verify/s6/morning-diag.log`), starts when the Gauntlet's score.txt appears:
   a. **FP8 vs GSQ v1 on the quant's own thinking**: the first 8,192 tokens (prompt 316 + reasoning) of the abc398_f
      effort-50 trace (`verify/bench/runs/ds41-cause-E50/traces/abc398_f.json`), through the FP8 chain
      (`/media/Crucial1TB/v41-thinkdiag/fp8`) and the GSQ chain (`.../gsq`); `tools/v41/quality/think_diag.py compare`
      -> `/media/Crucial1TB/v41-thinkdiag/think_diag.json`: per-position p(`</think>`, id 128822), entropy, p(top),
      first greedy stop, the top FP8 stop points with the text before them.  Question: does the original want to stop
      thinking where the quant keeps hand-checking?  (Both are the dealignai-abliterated weights, so this isolates the
      QUANT; abliteration itself can lengthen reasoning - pqhaz's 1017 card: median completion 357 -> 1517 tokens.)
   b. **abc400_c at a lighter route bias x0.3** (`tools/v41/serve/strata-ds41-bias03.json`, + the Q6_K default),
      `verify/s6/bias-quick.sh`, stopped at 40K thinking tokens; at x0.5 it thought 76.9K (cap) and PASSED.  Output
      `verify/bench/runs/ds41-bias-strata-ds41-bias03-abc400_c/`.  A finish well under 40K = the bias drives the length.
3. **1017 download (resumable)**: `deepseek-ai/DeepSeek-V4.1-Flash-1017` Engram shards 47 + 48 (101.5 GB each) ->
   **`/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017/`** (Team CX2 = SATA SSD, 221 GB free at start), ~20 MB/s
   (~3 h for both).  `tools/v41/fetch_v41_loop.sh` re-runs `fetch_v41.sh` until all verified (curl -C - resume TESTED
   by killing the transfer at 994 MB; sha256 per file, `.ok` markers).  After a reboot: the command in
   `RESUME-DOWNLOAD.txt` in that folder.  ONE fetch per folder at a time.  The other 46 shards (~307 GB) need space:
   Mal's plan = put the FP8 on another drive, and delete the v2 GGUF (NVME1TB) once 1017 is down to make room for the
   1017 quant.  Every 1017 shard differs from V4.1 (hashes + sampled Engram/embedding bytes) - nothing reusable.

## 1. What session 6 established (measured)
- **Served default now** (`tools/v41/serve/ds41-engine.sh`, `build-xolotl` rebuilt with HC_MIX=ON): HC-mix, DSpark
  re-seed on repeated prompts, prompt-lookup drafts <= 14 (`STRATA_SERVE_LOOKUP`), DSpark off the card for >= 4096-token
  prompts (`STRATA_SERVE_DSPARK_EVICT`), **dense Q6_K at load (`STRATA_DENSE_REQUANT=q6_K`; empty = off; file untouched)**.
  v1 + Q6_K vs v1: VRAM expert slots 13.01 -> 14.62 GiB, decode after 6.5K 22.4 -> 24.8 tok/s, after 16.8K ~20-24 ->
  28.1, edits 36.8 -> 40.0, prefill 6.5K = (~187), 16.8K 314-321 -> 291.  Quality (same bf16 path, 100 API fixtures):
  KL vs FP8 0.2281 -> 0.2306, API top-1 85.87 -> 85.76 %; the 1200-token greedy edit text is identical.
- **The quant is the API gap**: DwarfStar's 100 short API fixtures, top-1 vs the API: FP8 original 93.65 %, GSQ v1
  exact routing 86.13 %, x0.25 85.08, **served x0.5 81.49**, full bias 71.76.  Entropy: quant 0.343 vs FP8 0.288 (+19 %:
  the "unsureness" in numbers).  Tools: `tools/v41/quality/fp8_teacher_score.py`, `tools/v41/v41_e2e.py` (end-to-end
  variants: `--experts fp4|gsq`, `--dense fp8|q8|q6k`, `V41_GSQ_GGUF`, `V41_GSQ_HYBRID="20-35=/path"`).
- **Overthinking = over-verification, not loops** (abc398_f effort 50: 52.5K tokens, PASS; algorithm early, "So
  algorithm works" at 43 %, then character-by-character hand checks).  Effort 50 does not shorten it.
- **Error-fixing adapters don't pay** (`tools/v41/v41_repair.py`): LQER rank 8/32 -1.6/-5 % error, dense-side linear
  ~0 (L32: 3x worse).  My AdaRound-style refinement made it worse (not used).  Hadamard (`tools/v41/v41_hsweep.py`, all
  40 layers x 16 experts): medians ternary 0.904 / q2 0.974 / q3 0.944, gate/up-only ~1.0 -> < 1 GiB at equal quality
  and needs the fused down-projection kernel -> SKIPPED.  **Act-order GPTQ** 0.978 / 0.988 / 0.987, free -> ADOPTED
  (`V41_ACT_ORDER=1` in v41_quant).  Engine map for a future rotation: `verify/s6/hadamard-engine-map.md`.
- **The allocation proxy is reliable** (v1 realized/predicted per layer 0.90-1.20).  `v41_alloc.py` gained
  `--ternary-cap-at "20-31,33-35=1.0"` and `--budget-gib`.
- **GSQ v2** = `/media/mal/NVME1TB/Models/V41-GSQ-v2/V41-GSQ-v2.gguf` (139.8 GiB vs 151.8; map
  `/media/Crucial1TB/v41-calib/alloc_v2.json`, -12 GiB experts, act-order; serve config `strata-ds41-v2.json`).
  Calib held-out ppl 5.603 vs v1 5.645 (teacher 4.915); fixtures 85.76 % vs 86.12 %, KL 0.239 vs 0.229 (the all-ternary
  band 20-35 = ~half of that); **SLOWER**: prefill 170-176 / 277-286 vs 186 / 314-321, decode after 6.5K 18-19 vs 22-23.
  **Why is NOT settled - two candidate causes, not yet separated:** (a) ternary: the prompt MMQ path transcodes every
  PTQ1_0 matrix to Q2_0 on the card before the matmul (`ds41_moe.cpp` ~3582, `ds41_transcode.cu`) - confirmed to exist,
  the likely main prefill cost; ternary decode (5 trits/byte unpack) being slower is a hypothesis, unmeasured;
  (b) **the test served v2 with v1's `rank.bin`** (VRAM/arena seed order = per-byte order from v1's map) - would mainly
  hurt decode and may explain why the RAM arena held only 7,336 v2 experts vs 7,119 v1 (+3 % for an 8 % smaller file).
  Separate them with the rank re-test (section 1b), then profile.  **NOT the default; kept for the ternary work.**
- **All experts at Q2_0 instead of ternary (Mal's question)**: no size gain - v1 IS exactly the all-Q2_0 byte count
  (budget-neutral mix: 42.8 % ternary paid for 18 % Q3_K); all-Q2_0 would drop the transcode but has ~22 % more
  predicted error (proxy 2.27 vs 1.86) at the same 142 GiB.  Only if the ternary paths can't be fixed.
- **HauhauCS "K_P" quants** = importance-matrix mixed precision with a per-model profile (claims "1-2 quant levels" for
  5-15 % bigger files) - the same idea as our allocation; nothing new to borrow for the experts.  For the DENSE part a
  per-tensor mix (shared expert Q4_K, attention Q6_K/Q8_0) is the open experiment (priority 3).
- 1017 = same size as V4.1 (763.2B params, 510.3 GB, same layout).  Earlier (s5-era) dense requant test: q4_K +12 %
  forced-token speed, code ppl +1.7 %, KL 2.8x floor (V41_START_HERE.md) - not adopted.
- Vision: only the Python encoder exists (`tools/v41/vision/`, cosine >= 0.999); engine/serve integration (slice 2)
  was never built - V4.1 cannot take images yet.  "MTP-Vision" sidecar = the checkpoint's `mtp.0-2` blocks (= the DSpark
  draft) + the vision tower.

## 1b. Two things to know before anything else
- **The 1017 quantization plan is written down in full: `docs/ds4/V41_1017_QUANT_PLAN.md`** (facts checked, recipe from
  the session-6 lessons, all 10 steps with commands / times / disk, the code changes needed first (source paths are
  hard-coded to the dealignai FP8 in v41_calib / calib_layers / v41_gguf / v41_sidecar), the tier-map decision, the
  validation order, the disk plan).  1017's config + tokenizer are IDENTICAL to V4.1 (calibration tokens reusable);
  every shard's bytes differ.
- **Session 7, 09:23: `verify/s7/swap-v2-test.sh` (log `verify/s7/swap-v2-test.log`)** waits for the morning diag, deletes
  the duplicate dealignai shards 47/48 on NVME1TB, swaps v1 OUT of SSD NVME (verified temp copy on NVME1TB), v2 IN,
  runs `verify/s7/v2s-speed.sh` (-> `verify/s7/v2s-speed.log`, compare with v1/v1b in `verify/s6/v1v2r-speed.log`),
  then ALWAYS swaps v1 back (sha256-verified).  If it stopped midway: read the log - v1 is either in place or at
  `/media/mal/NVME1TB/Models/V41-GSQ-v1-tmp/V41-GSQ.gguf`.  DeepSeek can't be served while it runs.
- **Session 7 RESULT (09:10): v2's slowdown = the exFAT drive, NOT ternary.**  v2 own-rank re-test still slower off
  disk but EQUAL/FASTER on the resident-expert edit probe; O_DIRECT expert reads exFAT NVME1TB 0.96 GB/s vs ntfs3
  SSD NVME 5.1-5.3 (raw NVME1TB 3.31).  Never serve a GGUF from exFAT.  Details: `docs/ds4/V41_TERNARY_PROFILE_PLAN.md` s0.
- **Session 7 (08:00-) CPU prep - read `docs/ds4/V41_TERNARY_PROFILE_PLAN.md` before any v2/ternary work.**  Found:
  the stale rank.bin is SMALL (v2 coverage 39.3 -> 39.8 % VRAM, 86.7 -> 87.2 % VRAM+arena; v1 40.1 / 85.0); **v2 was
  served from exFAT on NVME1TB (970 EVO Plus) while v1 is on ntfs3 SSD NVME** - every file-tier miss and every
  prefill sweep goes through that drive, a new and likely big confound; the ternary decode kernel can only explain
  ~1-2 ms of v2's +9 ms/token (quarter-block dot already a dead end).  Ready: `/media/mal/NVME1TB/Models/V41-GSQ-v2/rank.bin`
  (v2's own), `tools/v41/serve/strata-ds41-v2r.json`, `verify/s6/v1v2r-speed.sh` (not run).  `V41_SRC` override
  added (see the 1017 plan).  Crucial1TB 1017 download STOPPED on Mal's word (one download at a time): start it after
  the Engram one finishes - command in `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017/RESUME-DOWNLOAD.txt`.
- **The v2 speed test was confounded**: v2 was served with v1's `rank.bin` (`v41_rank.py` orders experts by routing
  count PER BYTE from the tier map, so the VRAM/arena seed order was v1's).  Regenerate:
  `python3 tools/v41/engine/v41_rank.py --stats /media/Crucial1TB/v41-calib/stats --map /media/Crucial1TB/v41-calib/alloc_v2.json --out /media/mal/NVME1TB/Models/V41-GSQ-v2/rank.bin`,
  point `strata-ds41-v2.json`'s `--rank` at it, and re-run the v1/v2 speed test (`verify/s6/v1v2-speed.sh` pattern,
  ~25 min) BEFORE concluding anything about ternary.

## 2. Mal's priorities, in order
1. Read the Gauntlet + diag results (section 0) and report them plainly.
2. **Ternary kernels** (makes v2-style quants fast): profile v1 vs v2 (decode: time per expert type; prefill: the
   PTQ1_0 transcode cost); a native PTQ1_0 MMQ prompt path; a faster trit unpack for decode; check the arena/VRAM slot
   sizing by format class.  Then re-test v2's speed.  Without this, more ternary is not useful.
3. **Mixed-precision dense** (every GB counts): e.g. shared expert Q4_K + attention Q6_K; measure exactly like Q6_K
   (engine speed back to back + `v41_e2e.py` same-path quality).  Needs a per-tensor type option in the requant switch.
4. **The 1017 quant**: calibration pass + v41_alloc + v41_quant (act-order) on the base 1017 once downloaded, dense
   Q6_K at load.  Mal handles the uncensoring side himself.
5. Overthinking, by what the diag says (cheapest first): (a) a lighter route bias - the served x0.5 is FURTHER from the
   original than the quant itself (81.5 vs 86.1 % API top-1), so spend some of the Q6_K speed on e.g. x0.3; (b) a small
   logit nudge on `</think>` once an answer is written, tuned on the LCB set so hard problems aren't cut; (c) a
   behaviour adapter on the last layers only (36-39 shape confidence; FP8 targets computable locally with the streaming
   pass; backprop through 4 layers) - real engineering, a maybe; (d) a full end-to-end behaviour adapter on short
   traces (rented multi-GPU, hundreds of $).  The error-fixing kind is ruled out (section 1).  Note: if the diag shows
   the FP8 keeps going too, the cause is the model/abliteration, not the quant (pqhaz's 1017 card: abliteration took
   median completion 357 -> 1517 tokens).
6. Optional: vision slice 2; LCB resume (`verify/s6/ds41-lcb15.sh`, 5/15 done, all PASS); Hadamard if kernels allow.

## 3. Repo / git / public state
- github `main` = 7375e305 (session-6 docs push).  **Local, NOT pushed:** 90d0c47a, e4cc257d, 0ecf9111 (+ this
  session's docs commit).  Push only with Mal's OK, fast-forward only.
- GitHub issue #2 (sgmihai: multi-GPU + 1017): answered 2026-10-18 (one GPU so no multi-GPU from us, SSH dev would be
  harder, mab776's fork has 2-GPU work, 1017 quant planned) and kept OPEN as a tracker.  Issue #1 closed earlier.
- HF card `IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored` has the "Against the DeepSeek API" section (6992cf5e);
  the Q6_K default / v2 findings are NOT on it yet.
- Disk (07:50): NVME1TB 30 GB free (v2 GGUF 150 GB there, the 0731 V4 GGUF deleted s6 - its wrapper
  `deepseek-local-quetza` is dead); system 103 GB; Crucial1TB 87 GB (`V41-Engram` 95 GB = LIVE engine data, keep;
  `v41-calib`, `v41-fp8-fixtures`, `v41-thinkdiag` in place; everything else consolidated in `v41-results/`); Team CX2 =
  the 1017 download.  The v1 GSQ (`/media/mal/SSD NVME/AI/V41-GSQ/`) and the dealignai FP8 (Crucial `models/` + shards
  47/48 on NVME1TB) are untouched and still needed.

## 4. Standing rules (Mal)
- NSFW headless-only; DeepSeek API costs money (API only on the same subset as local, state tokens, ask first);
  no HumanEval for ds41; soft prune only; small tests first; ONE requant after all expert-changing options are measured;
  quality must be "pretty much negligible" for any speed/size trade; fast-forward pushes only, destructive ops need a
  go; show outward text (GitHub/HF/email) and wait for the go; sound like Mal, not AI ("computer", not "box").
- Use exact PIDs - `pgrep -f`/`kill $(pgrep -f ...)` killed my own shell in s6; old `rc=1` lines in reused logs fool
  grep waiters (match the specific layer/marker).  Never edit a running bash script.
- `/conductor` workers: `qc-dispatch <slice> --packet <file> --model flash`, then `qc-await <slice>` in the background.

# Session-6 night log (2026-10-17 23:00 - 10-18 06:30) — history, superseded by the START HERE block above

Mal went to bed 22:55 and left the overnight run to Claude: **"best quality, at smallest size"**. Read this block, then
the S6 log at the top of HANDOFF.md. Older blocks below are history.

## What was established tonight (measured)
- **The quant, not the abliteration, is what costs API agreement**: DwarfStar's 100 short API fixtures (6,159
  positions), top-1 vs the API: FP8 original (uncensored) **93.65 %** (NLL 0.330) vs GSQ v1 exact routing 86.13 %, x0.25
  85.08 %, served x0.5 81.49 %, full bias 71.76 %.  Tool: `tools/v41/quality/fp8_teacher_score.py` (work dir
  `/media/Crucial1TB/v41-fp8-fixtures`, the FP8 streaming pass takes ~17 min: ~25 s/layer).
- **Overthinking = over-verification, not loops**: abc398_f at effort 50 = 52.5K thinking tokens, PASS; the right
  algorithm in the first few %, "So algorithm works" at 43 %, then hand-checking ever longer strings
  (`verify/bench/runs/ds41-cause-E50/traces/abc398_f.json`; codebench now saves full traces in `<run>/traces/`).
  Effort 50 does not shorten it.  The x0.25-bias cause test was stopped (bias cost known from the fixtures).
- **Error-fixing adapters do not pay** (`tools/v41/v41_repair.py`, GSQ student chain on 10 calib seqs, held-out):
  per-expert LQER rank 8 / 32 = -1.6 / -5.3 % error (L4), -4.8 % (L13) for ~+14 % size at r32; a dense-side linear
  correction ~ -0.5 %.  The error is high-rank noise - spend bytes on bits.  (Behaviour adapter on short API traces =
  needs rented multi-GPU end-to-end training; only if v2 still overthinks.)
- **My GSQ/AdaRound refinement makes it WORSE** (L13: ternary 24.9 -> 34.1 %) - not used (V41_REFINE stays off).
- **Hadamard rotation (block-128, random signs) before GPTQ**: L13 ternary -8.7 %, q2 -2.2 %, q3 -5 %.  Needs an
  engine change + re-screen (~7 h) to adopt; decision rule: skip unless L25/L32 show a much bigger gain.
- **The allocation proxy is reliable**: v1's realized sequential per-layer error / predicted = 0.90-1.02 in the
  sensitive layers, ~1.0-1.25 in 20-35 (tiny absolute), 1.08-1.20 in 36-39; total 1.72 vs 1.80 predicted.
- **The ternary cap was the binding guard**: v41_alloc `--ternary-cap-at` (new) lets 20-31,33-35 go 100 % ternary;
  `--budget-gib` (new).  Proxy (sum of per-layer rel. error; v1 = 1.865): -10 GiB 1.61, **-12 GiB 1.71 (chosen, Mal
  OK'd; 8 % margin)**, -14 GiB 1.84 (break-even), -16 GiB 1.98 (36-39 at 0.75: 1.87).

## Decisions taken overnight (00:45, measured - `verify/s6/hsweep-chain.log`, `/media/Crucial1TB/v41-results/hsweep/stats/`)
- **All-layer sweep** (`tools/v41/v41_hsweep.py`, 16 experts x 40 layers, held-out output error vs plain GPTQ; medians):
  Hadamard r128 (gate/up AND down rotated) ternary 0.904 / q2 0.974 / q3 0.944; r1024 the same; **gu128 (gate/up only,
  the cheap engine change) ~1.0 (no gain)**; **act-order GPTQ 0.978 / 0.988 / 0.987** (37 of 40 layers better).
  In bytes (`tools/v41/hsweep_alloc.py`): r128 -2.8 % proxy at 130.38 GiB (< 1 GiB at equal quality); act-order -1.3 %.
- **Hadamard SKIPPED** (the gain needs the down-projection rotation = the fused swiglu_q8_1 kernel + CPU paths, see
  `verify/s6/hadamard-engine-map.md` - a deepseek-flash worker's file:line map - for < 1 GiB).  **Act-order ADOPTED**
  (`V41_ACT_ORDER=1`, v41_quant.quant: same scales, same format, no engine change; verified on-grid).
- Repair measurements L4/13/25/32 (`/media/Crucial1TB/v41-results/repair/stats/repair_LNN.json`): LQER r32 -4.5..-5.3 %;
  dense-side correction useless (L32 made it 3x worse); refinement worse; L25's tiny error IS low-rank (r8 -66 %) - a
  note for a later "ternary band + cheap correction" idea, not tonight.
- **Build started 00:43 with V41_ACT_ORDER=1** (env verified in the quant process).  Layer 0: v2 3.41 GiB, held-out rel
  err 0.0297 vs v1 3.70 GiB / 0.0352; 199 s per layer.

## RESULTS by 05:45 (measured)
- **GSQ v2 built** (`/media/mal/NVME1TB/Models/V41-GSQ-v2/V41-GSQ-v2.gguf`, 139.8 GiB vs v1 151.8; work dir
  `/home/mal/v41-work/quant-v2`; act-order on; 00:43-03:16, ~3.8 min/layer).  Every layer's held-out error below v1's
  (-6..-16 %).  Head gate (calib held-out, 49K tok): **v2 ppl 5.603 / top-1 0.6753** vs v1 5.645 / 0.6745 (teacher
  4.915 / 0.7011).
- **End to end on the 100 API fixtures** (`tools/v41/v41_e2e.py`, `verify/s6/e2e-*.log`; v1 reproduces its engine score:
  86.12 % vs 86.13 %): v1 API top-1 86.12 %, KL-vs-FP8 0.2293, entropy 0.343 | **v2 85.76 %, KL 0.2394 (+4 %), 0.346** |
  hybrid (v2 + v1 layers 20-35) 85.94 %, 0.2336 (the near-all-ternary band = ~half of v2's chat-text loss).  FP8 entropy
  0.288: the quant's "unsureness" is +19 % entropy and v2 does NOT change it.
- **Speed (served defaults, two rounds each, `verify/s6/v1v2-speed.log`): v2 is SLOWER** - prefill 170-176 / 277-286 vs
  v1 186 / 314-321 tok/s, decode after 6.5K 18.1-19.2 vs 21.7-23.1; edits +3 %.  Cause: the RAM arena holds 7,336 v2
  experts vs 7,119 v1 (+3 % for an 8 % smaller file - ternary does not shrink the slots), and v2 moved bytes into the hot
  sensitive layers (more Q3_K).  **v2 is NOT the new default** (kept on disk for Mal).
- **Dense Q6_K = the night's real win** (engine switch `STRATA_DENSE_REQUANT=q6_K`: ggml re-quantizes the 321 Q8_0 dense
  matrices at load, 7.59 -> 5.86 GB, file untouched): quality on FP8 experts KL 0.0152 -> 0.0155, API top-1 93.60 -> 93.64
  (negligible); **v1 + Q6_K: VRAM expert slots 13.01 -> 14.62 GiB, decode after 6.5K 24.8 (v1 22.4, +11 %), after 16.8K
  28.1 (v1 20.4-24, +15-35 %), edits 40.0 (36.8, +9 %); prefill 6.5K =, 16.8K 291 (-9 %)** (`verify/s6/v1q6k-speed.log`).
  **Confirmed on the real quant, same bf16 path** (`verify/s6/e2e-v1q8.log` vs `e2e-v1q6k.log`): KL vs FP8 0.2281 ->
  0.2306 (+1 %), API top-1 85.87 -> 85.76 %, NLL 0.5218 -> 0.5226; the 1200-token greedy edit text is IDENTICAL to the
  Q8_0 reference.  **=> made the served default** in `tools/v41/serve/ds41-engine.sh`
  (`STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"`; empty = off).  Not pushed - Mal to OK.
- Not run: the v2 overthinking pairs (v2 is not deployed; v1's experts are unchanged, so its thinking is unchanged).

Disk (2026-10-18 07:40): night results consolidated in `/media/Crucial1TB/v41-results/{e2e/<run>,hsweep,repair,quant-mixed}`
(activations deleted, stats/scores kept); `v41-calib`, `v41-fp8-fixtures`, `V41-Engram` (LIVE engine data) stay put.

## The overnight pipeline (all background; check each log)
1. `verify/s6/repair-chain.log` - the measurement chain (layers 0-32, measures at 4/13/25/32: LQER / dense / refine /
   Hadamard), results in `/media/Crucial1TB/v41-results/repair/stats/repair_LNN.json`.
2. `verify/s6/build-v2.sh` > `verify/s6/build-v2.log` - **the v2 requant** (started by Claude after the L25/L32
   Hadamard decision): map `/media/Crucial1TB/v41-calib/alloc_v2.json` (-12 GiB, experts 130.38 GiB), file
   **`/media/mal/NVME1TB/Models/V41-GSQ-v2/V41-GSQ-v2.gguf`** (layout already written), work dir
   `/home/mal/v41-work/quant-v2` (quant_layers.sh, resumable: done layers skip; ~8.5 h), then the head gate vs v1
   (held-out ppl 5.645 / top-1 0.674; teacher 4.915 / 0.701).
3. `verify/s6/e2e-runs.sh` > `verify/s6/e2e-runs.log` - after BUILD V2 DONE: end-to-end on the 100 fixtures vs API and
   vs FP8 (KL, top-1, entropy, p(top)): gsq-v2, gsq-v1, fp4-q8, fp4-q6k (dense Q8_0 vs Q6_K on perfect experts).
   `tools/v41/v41_e2e.py`.
4. Morning: serve v2 (same config, `-m` the v2 file; rank.bin from v1) - smoke, fixtures through the engine, thinking
   length on abc398_f / abc400_c vs v1 (27.8K capped / 76.9K capped at x0.5).  Dense Q6_K only if E3 ~= E2
   (negligible, Mal's rule) - a repack, no requant.  Nothing pushed to GitHub/HF without Mal.
Disk: the old 0731 V4-Flash GGUF was deleted (Mal's pick) for room; v1 GSQ untouched at `/media/mal/SSD NVME/AI/V41-GSQ`.
Pitfall hit tonight: `pgrep -f` / `kill $(pgrep -f PATTERN)` matched my own shell - use exact PIDs.

# RESUME PROMPT — Xolotl — START HERE (written 2026-10-17 ~08:20, end of session 5; sessions 1-4 below are history)

You are resuming a multi-session project for **Mal** (Malich Coory, Indra's Mirror). This block is self-contained: read
it, then `sed -n '1,80p' HANDOFF.md` (the session-5 block). Everything here marked **measured** was measured on this box
(RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe) - do not re-derive it. The sections below this block are older
sessions' history; trust this block where they disagree.

## 6.0 FIRST THING: who owns the GPU (a collision cost session 5 an hour)

At the end of session 5 **another Claude Code session (`bfd97150-47cf-45df-ab05-f58bc1967b3d`, the continuation of the
session-4 ds41-prefill conversation) was running ds41 turnbench experiments in this same tree, on the same GPU and the
same port 8151.** Both sessions' scripts start a server on :8151 and `pkill -x strata/ds41_generate` on teardown, so they
killed each other's engines (exit 143), interleaved requests on one server (prefix reuse lost, prefill halved), and the
ds41 throughput numbers of session 5's last two probes are INVALID. Before any GPU work:

```bash
ps -eo pid,lstart,args | grep -E "[c]laude --|[t]urnbench|[s]erve_ds41|[g]auntlet.py run|ds41_generate|glm_generate" | cut -c1-160
ss -ltnp | grep -E ":81(4[0-9]|5[0-9]) "      # anything on 8140-8159?
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # the desktop alone is ~0.7-0.9 GB
grep -n "engine started" /tmp/strata-ds41.log | tail -3                     # a ds41 start you did not make = someone else
```

If another session is using the GPU, **ask Mal which one owns it** (session 5 recommended: one owner; the other does
CPU-only work). Consider using a different port than 8151 for your own servers (e.g. 8161) so a peer's teardown
(`ss sport = :8151` + kill) cannot hit you - `verify/e2e/gauntlet-local-all.sh` hard-codes 8151 (URL + down()).

## 6.1 What is DONE and PUBLIC (session 5)

- **GitHub: `Indras-Mirror/Strata-DSV4.1` was renamed to `Indras-Mirror/Xolotl`** (Mal's go). 24 stars kept; the old
  URL 301-redirects; fast-forward push of branch `xolotl` -> `main` (never force). Local remote `github` already points
  at the new URL. Description + topics set (strata, deepseek, deepseek-v4, glm, qwen, moe, ...). **Mal OK'd pushing
  updates to main this session** (README/numbers/docs) - keep pushes fast-forward and announce them.
  Still manual for Mal: Settings -> Social preview = `docs/brand/assets/github-social-preview-1280x640.png`.
- **README rebuilt headline-first** (`README.md`): DeepSeek-V4.1 is the headline; "a Strata fork, heavily extended"
  (Strata's core kept, ~71K lines added: 1,368 commits, +71,049/-259 in src/include/tools/serve vs Niko1221/Strata main);
  a note for Strata-DSV4.1 stargazers; the model table (columns **Prefill | Decode**, Mal's naming); quick start; one
  paragraph of how-it-works; links to the deep pages **`docs/xolotl/HOW_IT_WORKS.md`** and **`docs/xolotl/BENCHMARKS.md`**;
  short credits. The **Gauntlet is deliberately OUT of the public docs** until it is refined and every model re-run
  (Mal). `README.*.md` translations are still the OLD content (not updated).
- **Hugging Face card** `IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored` updated (README only, commit abd78744):
  engine = Xolotl (formerly Strata-DSV4.1), a 2026-10-17 update note (HumanEval 21/21, tool 10/10, agentic 3/3, 26.8 tok/s;
  small-turn prefill, adaptive arena), `build-xolotl` build/run steps, tag `xolotl`. Local copy:
  `verify/s5/hf-card-README.md` (upload with `huggingface_hub.HfApi().upload_file`, path_in_repo README.md).
- **Brand**: Mal's final picks are in `Assets/` (committed): `circle.1.png` = THE logo, `wordmark-banner...2.png` = the
  banner, `hero...2.png`. Composed locally (PIL, no image credits) into `docs/brand/assets/`: `logo-circle-400.png`
  (README credits), `avatar-512/256.png`, `favicon.ico`, `favicon-64.png`; `readme-header-1200.png` = the banner.
  `docs/brand/XOLOTL_BRAND.md` has a FINAL section. Do not generate new images (OpenRouter credits).

## 6.2 Engine state (all committed, branch `xolotl`, pushed)

- **`build-xolotl` is the one build** - fully rebuilt this session with the README's exact configure line
  (`-DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp`, the pinned llama.cpp 3cf03257 + `tools/xolotl/ggml/xolotl-ggml.patch`;
  `STRATA_Q6K_EXPERTS` = default ON). Its old cache pointed at the moved `~/AI/Strata-DS4`; fixed. A fresh `git clone` +
  `tools/xolotl/fetch-llama.sh` + that cmake line configures cleanly (checked; the full compile of a fresh clone was NOT
  run - only the in-tree full build). `build-pf` (session 4's ds41-only dir) is obsolete.
- **ds41: the conversation-adaptive arena is ON by default** (cff4bb24; `STRATA_ARENA_REBAL=0` = off). Measured
  (turnbench 4K base + 8 turns, 20 s gaps): file-tier reads per small turn -21..-30 %, small-turn wall 8.48 -> 7.60 s;
  2,561 swaps all published, `DS41_CHECK_ARENA=1` byte check 0 bad; e2e with it on (`verify/e2e/final-think/r_ds41_rebal`):
  21/21 he, 10/10 tool, 3/3 agentic, decode 26.8 tok/s. The full build caught a CPU-target bug (the `Rebal` forward
  declaration was inside the CUDA guard) - fixed in the same commit.
- **GLM: `--route-bias-file X` added and `--route-bias` made DECODE-only** (7d7775c1, opt-in; NO served config uses them
  yet). File-tier = neither VRAM cache nor RAM arena. The table resets after each request so the prompt routes exactly.
  `GLM_BIAS_PROMPT=1` (one-shot, `--prefill-chunk 0`) applies the decode biases to the prompt so `--ppl` measures them.
- **Wrappers repointed** (`~/.local/bin`, backups `*.bak-20261017`): `strata-ds41-quetza` and `strata-glm-quetza`
  checked for `build-ds4-gpu/` / `build-glm-gpu/` (absent in Xolotl) -> now `build-xolotl/`. `strata-quetza` and
  `strata-dense-quetza` already used build-xolotl. Wrappers were NOT launched end to end this session.
- **`tools/throughput.py`** (new): one headline probe per model - a fresh ~N-token prompt (prefill), a decode request,
  a +200-token agent turn; server timings. Wired into the lane script (`GAUNTLET_THROUGHPUT_ONLY=1`).

## 6.3 Measured numbers (the ones to quote)

| model | prefill (fresh prompt, server) | decode (HumanEval e2e, thinking 32K) | HumanEval / tool / agentic |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | ~210 tok/s @7.8K (session 4); **263.8 tok/s @16.7K** (s5, fresh server, valid); ~350-390 @20K (V4.1 notes) | 26.8 (arena on) | 21/21, 10/10, 3/3 |
| GLM-5.3-Flash | not measured with a sane config yet | ~28 (partial run, skip config) | partial (5 of 21) |
| Qwen3.8-Flash-Next | **3,430 @6.5K / 3,005 @16.7K** (upstream Strata 10-05 build on the same prompts: 3,378 / 2,933) | 127.8 | 21/21, 10/10, 3/3 |
| Qwen3.8-27B dense | **1,385 @6.5K / 1,244 @16.7K** | 98.5 | 21/21, 10/10, 3/3 |

The old README prefill numbers (Qwen Flash 475, 27B 1,097) were HumanEval's short prompts (a few hundred new tokens:
fixed per-request cost dominates) - replaced. README now shows the probe numbers. A +300-token agent turn on Qwen
Flash costs 0.7-0.8 s; on 27B 0.4-0.6 s.

**ds41 throughput is the open item:** the s5 probes on 8K gave prefill 129-177 tok/s and decode 12-17 tok/s, but those
runs were contaminated by the other session's interleaved turnbench (see 6.0). Re-measure on a quiet box:
`GAUNTLET_THROUGHPUT_ONLY=1 GAUNTLET_OUT=throughput-ds41 THROUGHPUT_PROMPTS="8000 20000" bash verify/e2e/gauntlet-local-all.sh ds41`
then A/B decode with `STRATA_ARENA_REBAL=0` (the arena's reader thread reads NVMe while decode also reads the file tier -
rule out that it costs decode; the HumanEval e2e says 26.8 with it on, so likely fine). Note the prompt batch is sized
from free VRAM (`serve: VRAM X GiB free -> prompt batch N (of 12288)`): a desktop app holding VRAM (Teams, a browser)
shrinks it and lowers prefill.

## 6.4 GLM: the investigation and the decision that is pending

Session 4's served config (`tools/glm/serve/strata-glm-maya-abl-512k.json`: `--skip-miss/--skip-file/--skip-file-prefill
0.15 --renorm-skip --prune prune-ezct-0.25.txt --prune-penalty 0.05`) **corrupts tool calls** (byte-salad names, mojibake
after `<tool_call>`, "I'm sorry, I cannot..." loops; Gauntlet core 0/6). The engine's own comment
(`tools/glm/glm_generate.cpp`, `--lora-exps` block) says why: a skipped expert loses its abliteration-LoRA delta, the
ablation stops being all-or-nothing, "the model loops"; `--renorm-skip` is the "test" escape hatch the config used.
Greedy A/B (`verify/s5/glmab/`): the pre-merge build and Xolotl give identical tool calls with exact routing -> **the
merge is clean**; corruption appears only with the skip/prune approximations.

The fix = the ds41 recipe (soft, every expert reachable, full LoRA delta): no `--skip-*`, keep the soft prune, add
`--route-bias 0.5 --route-bias-file 0.3` (ds41's served values; decode only). Perplexity sweep (`--ppl`, teacher-forced
2,586-token agentic transcript = the 03 raytracer task + its reference solution rendered in GLM's chat template with
tool calls, `verify/s5/glmq/agentic03.ids`; biases applied to the prompt via GLM_BIAS_PROMPT; 200-token decode):

| config | ppl | decode tok/s | experts from VRAM cache |
| --- | ---: | ---: | ---: |
| E exact (no prune) | 8.00 (repeat 8.14) | ~11 | 32-34 % |
| P soft prune 0.05 | 7.89 | 11.3 | 32 % |
| **B1 prune + bias 0.5 / file 0.3** | **4.32** | **35.0** | **98.7 %** |
| B2 prune + bias 1.0 / file 0.6 | 7.20 | 39.5 | 100 % |
| SK the served skip config | 4.99 | 28.2 | 38 % (176 entries/token skipped) |
| EG exact, `--pcie 1.0` (no CPU pool) | 8.21 | 9.9 | 30 % |
| EP exact, `--pf-b 0` (no prefetch) | 8.17 | 9.7 | 24 % |
| EN exact, no LoRA | 8.50 | 11.5 | - |
| B1N bias 0.5/0.3, no LoRA | 4.67 | 38.2 | 98.9 % |
| ED exact, LoRA dense-only | 8.41 | 11.7 | 36 % |
| R exact, PRE-MERGE binary (per-token) | 7.84 | 10.4 | 34 % |
| EC exact, chunked MMQ prompt path | 8.11 | - | - |
| RC exact, chunked, PRE-MERGE binary | 8.06 | - | - |

Reading it: **no engine bug** - exact routing gives the same ppl on two independent code paths (per-token and chunked
MMQ) and on both builds, with/without the CPU pool, prefetch and LoRA. Biasing toward the VRAM-cached (census-hot)
experts LOWERS agentic-text ppl (8.0 -> 4.3). The same thing is documented for DSV4.1 (README.strata.md, the V4.1
section: "agentic text 15.11 -> 6.39 (better: the GSQ quant gave its cheapest formats to rarely-used experts)") - the
Maya-S-v2 quant evidently degrades its cold experts the same way. B2 (stronger bias) is worse than B1: too much
distortion. **Provisional winner: B1** - 3.5x faster decode than exact, ppl better than exact on agentic text.

**Still to do before switching GLM's served config (in order):**
1. The chat-text cross-check (the s5 attempt was invalid - the IDS override was ignored): exact is documented at ppl
   ~5.27-5.34 on the old gate's `eval_chat` (Maya+LoRA). Run `bash verify/s5/glmq/glm_sweep.sh XE XB1 XB2` (~10 min,
   GPU must be free). B1 should not be much worse than XE on chat; if it is, consider bias 0.25/0.15.
2. New served config: copy `tools/glm/serve/strata-glm-maya-abl-512k.json` -> drop `--skip-miss 0.15 --skip-file 0.15
   --skip-file-prefill 0.15 --renorm-skip`, add `--route-bias 0.5 --route-bias-file 0.3` (keep `--prune ... --prune-penalty
   0.05`, soft). Serve it, then the malformed-call replay (`verify/s5/glmab/ab.sh` pattern: Gauntlet 13/14 step-1
   requests `verify/s5/glmab/13.json` `14.json`, greedy) and the HumanEval e2e
   (`TASKEVAL_THINK_BUDGET=32768 python3 tools/taskeval/taskeval.py run http://127.0.0.1:<port> verify/e2e/final-think/r_glm2`).
   Also the throughput probe. Then fill the README GLM row (it says "numbers coming") and push.
3. Mal's rule (memory `feedback-soft-prune-only`): every expert must stay reachable (penalty, never mask) and report the
   ppl cost - B1 satisfies it; the `--skip-*` flags do NOT (they drop experts).

## 6.5 The Gauntlet (benchmark) - v2 built, needs refining + an overnight run of every model

`tools/gauntlet/` (README.md documents v1; v2 = f37a229d). **v2 rules** (Mal): `GAUNTLET_TURNS` turns per task
(default **10**), **no token budget** (each turn: the served 32K thinking budget + 4096 answer), the hidden tests grade a
COPY of the workspace after every turn that wrote a file or ran code (the model never sees it) -> a progress curve and
the turn the task was first solved (the task ends there); progress is normalised from the untouched workspace's own
score (01 starts at 0.61, 10 at 0.60, 12 at 0.07 - no free credit); `score` = solved count, SCORE = sum of progress,
mean solve turn, throughput (decode / prefill-of-new-tokens / new+reused / e2e). `selftest.py` = 7 cases a-g, all PASS
(f = a local engine's think budget - the bug that graded every local task as a runner error until 5ed4adf8; g = per-turn
grading). `validate.py <task dir>` must say OK for every task you change.

**v2 core set: 01, 02, 03, 06, 07, 10, 14** (replaced 12/13/15/16 that nobody passed - even API at max effort fails
15/16). **API baseline (done, do NOT re-run - credit): `out/gauntlet-v2-api`: 5/7, SCORE 6.41**, solved 01@4 02@3 07@9
10@4 14@4, 03 stuck at 0.50 (scene B), 06 at 0.91, 106,792 tokens, 546 s.
v1 core results (old set, 12K budget, for history): API 1/6, qflash 1/6, q27 0/6, ds41 0/6, GLM served 0/6, GLM exact 1/6.

Mal: "leave the gauntlet out until it's refined for better pass rate, and we redo all our models overnight tonight."
Next:
1. Calibrate on the fastest local model first (small test): `GAUNTLET_OUT=gauntlet-v2 bash verify/e2e/gauntlet-local-all.sh qflash`
   (~10-20 min). Look at the spread: a set where every local model scores 0 solved is useless; target each model
   solving some and partially progressing on others.
2. Refine for pass rate if needed: add 1-3 medium tasks (same format: `tasks/NN-name/{task.json,workspace,hidden,
   reference,trap}`; rules in tools/gauntlet/DESIGN.md + HARDENING.md; `validate.py` must pass; derivability: every
   hidden assertion maps to a documented sentence). Keep the API from being re-run more than once per set version.
3. Overnight: `GAUNTLET_OUT=gauntlet-v2 bash verify/e2e/gauntlet-local-all.sh ds41 qflash q27 glm` - but first change the
   `glm)` line in that script to the NEW GLM config (6.4 step 2). The lane runs the throughput probe after each model.
   ds41 will be the slow one (~21 tok/s decode; budget ~1.5-2 h). Score: `python3 tools/gauntlet/gauntlet.py score
   out/gauntlet-v2-api verify/e2e/gauntlet-v2/*/`.
4. Then put the v2 results back into README/`docs/xolotl/BENCHMARKS.md` (removed in 29bb3c1a) and push.

## 6.6 Where everything is

| what | where |
| --- | --- |
| lane script (serve -> gauntlet -> throughput -> teardown per model) | `verify/e2e/gauntlet-local-all.sh` (local-only: `verify/` is in `.git/info/exclude`). Env: GAUNTLET_OUT, GAUNTLET_ONLY, GAUNTLET_TURNS, GAUNTLET_THROUGHPUT_ONLY, THROUGHPUT_PROMPTS, DS41_BUILD; models ds41 qflash q27 glm glm-exact |
| e2e (HumanEval/tool/agentic) | `tools/taskeval/taskeval.py run|score|speed`; results `verify/e2e/final-think/r_{ds41,ds41_rebal,qflash,q27,glm(partial)}`; `taskeval.py score` crashes on `r_glm` (no results.json) - list dirs explicitly |
| session-5 experiments (scripts, inputs, logs) | `verify/s5/{glmq,glmab,qpf,rebal}/`; the clean GLM sweep = `verify/s5/glmq/glm_sweep.sh` |
| throughput probes | `verify/e2e/throughput/{q27,ds41}/`, `verify/e2e/throughput-ds41b/` (ds41 ones contaminated except the 16.7K 263.8) |
| Gauntlet runs | `out/gauntlet-v2-api` (v2 API), `out/gauntlet-core-api` + `verify/e2e/gauntlet-local/*` (v1 core) |
| configs | ds41 `tools/v41/serve/strata-ds41.json` + `ds41-engine.sh`; GLM `tools/glm/serve/*.json` + `glm-engine.sh`; Qwen Flash `~/.quetza-data/xolotl/strata-swift-iq3_xxs-sc117-lazyvision.json`; 27B written by `strata-dense-quetza` (lane copy `verify/e2e/gauntlet-local/cfg_q27.json`) |
| old trees (reference builds) | `~/AI/Strata-Backup/{Strata-DS4,Strata-GLM,...}`; pre-merge GLM binary `~/AI/Strata-Backup/Strata-GLM/build-glm-gpu/glm_generate`; upstream Strata `~/AI/Strata/build/strata` (10-05) |
| engine logs | ds41 `/tmp/strata-ds41.log` (SHARED by every ds41 server - grep "engine started" for timestamps); GLM per-config `log` key |

## 6.7 Rules (Mal's standing constraints)

- Measure before concluding; A/B against the previous production build; small tests before full runs; report the
  numbers with what they were measured on.
- Soft prune only (penalty, never a mask/skip); report the ppl cost of any speed trick.
- **API credit**: one DeepSeek API baseline per benchmark version (memory `feedback-api-credit`); calibrate on locals.
- Never `pkill -f` a pattern that matches your own command line; kill exact PIDs.
- **Never edit a bash script that is currently running** - bash reads scripts incrementally; session 5 broke a running
  sweep this way (`syntax error near ;;`). Copy to a new file and run the copy.
- Build per-run args INSIDE the run function when env overrides are used (the IDS bug).
- Commit as you land; `/tmp` does not survive a reboot (copy scratch results into `verify/`).
- Public pushes: fast-forward only; Mal OK'd README/numbers/docs updates to `Indras-Mirror/Xolotl` main in session 5.
  Anything destructive (force push, history rewrite, deleting releases/model files) still needs his explicit go.
- NSFW hard rule (global CLAUDE.md) applies to any image work.

## 6.8 Done looks like

- GLM on the B1-style config with clean tool calls, its README row filled from a real e2e + throughput run.
- ds41 throughput re-measured on a quiet box (8K and 20K prefill, decode) and the README row confirmed or corrected.
- Gauntlet v2 refined, every model run overnight, results in README/BENCHMARKS, pushed.
- Optionally: translations updated or removed from the README links; social preview uploaded (Mal).

---

# HISTORY (sessions 1-4) — RESUME PROMPT — Xolotl (the unified Strata engine) — written 2026-10-16 ~17:00, end of session 3

You are resuming a multi-session project for **Mal** (Malich Coory, Indra's Mirror). Read this whole file, then the top
block ("session 3") of `~/AI/Xolotl/HANDOFF.md`. Do not re-derive anything this file states as measured. Mal's rules
for this project: measure before concluding, A/B against the previous production build, small tests first, never push
or delete without his explicit go, commit locally as you go (a reboot already wiped uncommitted /tmp work once).

---

## 1. What Xolotl is

`~/AI/Xolotl` (git branch `xolotl`) is ONE engine tree that runs every Strata-family model Mal uses:

| model | engine binary | serve shim / config | wrapper (live) |
| --- | --- | --- | --- |
| **DeepSeek-V4.1-Flash** (552B MoE, GSQ 152 GiB) | `build-xolotl/ds41_generate` | `tools/v41/serve/serve_ds41.py` + `strata-ds41.json` (512K, DSpark, `--kv-fp4`, route bias x0.5) via `ds41-engine.sh` | `strata-ds41-quetza` (port 8150) |
| **GLM-5.3-Flash** (320B MoE, Maya-S-v2 IQ2_XXS + abliteration LoRA) | `build-xolotl/glm_generate` | `tools/glm/serve/serve_glm.py` + `strata-glm-maya-abl-512k.json` via `glm-engine.sh` | `strata-glm-quetza` (8144) |
| **Qwen3.8-Flash-Next** (125B MoE, SC117 abliterated IQ3_XXS, lazy vision) | `build-xolotl/strata` + `build-vision/bin/strata-vision` | `python -m serve.server` + `~/.quetza-data/xolotl/strata-swift-iq3_xxs-sc117-lazyvision.json` | `strata-quetza` (8130) |
| **Qwen3.8-27B dense** (d0xin uncensored IQ3_S + MTP) | `build-xolotl/strata-dense` | `python -m serve.server`, config written by the wrapper (`--draft-max 3`, k8v4, 200K) | `strata-dense-quetza` (8090) |
| DeepSeek-V4-Flash-0731 (legacy, 284B) | `build-xolotl/ds4_generate` | no serve shim (Mal serves 0731 via a llama.cpp fork) | — |

Every server is Xolotl's own `serve/server.py` (Strata 0.1.44.1 + local features: lazy vision, `/v1/decide`, voice
sidecar `serve/voice.py`). Python: `~/AI/Strata/.venv/bin/python` (OG Strata's venv) or system `python3` (dense wrapper).
Every config defaults to `reasoning_budget_tokens: 32768` (Mal: all models get a 32K thinking budget).

Lineage: Strata (Niko1221, MIT) + our forks Strata-DSV4.1 (DeepSeek) and Strata-GLM + perronemirko's 27B dense port +
the Strata-qwen voice work. Upstream merged at 0.1.44.1 (`niko/main` 296b59d2).

## 2. Where everything is

- `~/AI/Xolotl` — the engine. Remotes: `github` = Indras-Mirror/Strata-DSV4.1 (the push target), `github-glm`,
  `niko` (upstream), `perrone`, `origin` = `~/AI/Strata-Backup/Strata-DS4`, `up` = `~/AI/Strata`.
- `third_party/llama.cpp` — untracked + gitignored: llama.cpp **3cf03257** + `tools/xolotl/ggml/xolotl-ggml.patch`
  (19 ggml files: V4.1 ops, bit-exact fp4 KV, deterministic MMQ, upstream #30103/#29953, GLM no-graph). Proven
  identical (diff -r) to the tree everything was measured on. Fresh: `bash tools/xolotl/fetch-llama.sh`. CMake applies
  the patch itself when STRATA_GGML_DIR is unset.
- `build-xolotl/` + `build-vision/` — **their CMakeCache still points STRATA_GGML_DIR / LLAMA_DIR at
  `~/AI/Strata-DS4/third_party/llama.cpp`, which moved.** Reconfigure before the next build (§5 step 3).
- `~/AI/Strata` — OG upstream checkout (keep: venv + `data/expert-profile*.bin` used by the Qwen Flash config).
- `~/AI/Strata-data` — Qwen Flash packs + MTP (keep).
- `~/AI/Strata-Backup/` — every old tree (Strata-DS4 + worktrees, Strata-qwen, strata-dense-build/-runtime,
  Strata-DS4-private, models-dense, v41-dev). Rollback binaries kept. 89 GB deleted with Mal's OK —
  `DELETED-2026-10-16.txt`. Wrapper rollbacks: `~/.local/bin/*.bak-prexolotl-20261016`.
- Model files: GLM `/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/` (the `SSD NVME` partition does NOT
  auto-mount after a reboot: `udisksctl mount -b /dev/nvme0n1p3`), V4.1 `/media/mal/SSD NVME/AI/V41-GSQ/`, engram
  `/media/Crucial1TB/V41-Engram` + `SSD NVME/AI/V41-Engram`, Qwen Flash + 27B + DSpark sidecar + GLM LoRA on
  `/media/mal/NVME1TB/Models/`, NInfer artifact `/media/Crucial1TB/models/ninfer/`.
- Verification artefacts (untracked): `~/AI/Xolotl/verify/` — every A/B script, log and stub-engine test of session 3.

## 3. What is VERIFIED (do not redo)

| | result |
| --- | --- |
| DSV4.1 | logits bit-identical to production; e2e non-thinking 20/21 HumanEval, 10/10 tools, 3/3 agentic, 5391 tok, 596 s == production 598 s; decode during tasks 27.5 vs 27.8 tok/s |
| GLM-5.3 | engine output within production's own run-to-run noise; serve GEN lines byte-identical; tool calls identical at every stream split |
| Qwen Flash | identical greedy chat + image answers; decode 151 vs 116.5 tok/s (the 0.1.44 merge), prefill 3577 vs 3509 |
| 27B dense | teacher-forced mean abs dlogprob 0.015 vs production; decode 116 (draft 3) vs 105; prefill 1240 vs 1331 (-7%, open) |
| V4-0731 | ppl 10.39 vs production 10.41-10.51; decode 19.4 vs 19.3 |
| server | 733 unit tests; `tools/v41/serve/test_serve_ds41.py` all passed |
| NInfer (27B, for reference) | prefill 4402 / decode 176 tok/s on the same speedbench (20K prompt, 1024-token code) |

Bugs fixed in session 3 (details + mechanisms in HANDOFF S3.2): tokenizer `_re` (bug 12), the missing strata-dense
target + `/v1/decide` (lost with /tmp), voice sidecar cold-start kill + orphaned children, V4-0731 nextn layer drop
(bug 13), GLM short-prompt OOM, the GLM/DSV4.1 shims running the old 0.1.39 server + four 0.1.44 API drifts (incl.
**every DSML/GLM tool call came out as text** — `F.FUNC_START = ""`), and fresh-clone build failure (ggml patch).

## 4. What was running when this was written

`~/AI/Xolotl/verify/e2e/final-think.sh` — THE README TABLE RUN: DSV4.1 -> GLM -> Qwen Flash -> 27B, each through
`tools/taskeval` (smoke + HumanEval-21 + tools-10 + agentic-3), greedy, thinking on, `TASKEVAL_THINK_BUDGET=32768`,
the wrappers' configs. Results land in `verify/e2e/final-think/`:
`score.txt` (pass counts), `speed.txt` (prefill / decode tok/s DURING the tasks — Mal wants these, NOT speedbench),
`r_<model>/results.json`, `server_<model>.log`. Log: `verify/e2e/final-think.log` ("ALL DONE" at the end).
At ~16:55 DSV4.1 had 20 PASS lines, 0 fails. Expect ~3-4 h total. If it died: rerun `bash verify/e2e/final-think.sh`
(it is idempotent per model; delete `final-think/r_<model>` to redo one).

## 5. Next steps, in order

1. **Fill the README table.** `README.md` has `{{DS41_HE}}`-style placeholders (HE / TOOL / AGENT / PF / DEC per model:
   DS41, GLM, QF, Q27). Source: `python3 tools/taskeval/taskeval.py score verify/e2e/final-think/r_*` and `... speed ...`.
   Format: `20/21`, `10/10`, `3/3`, prefill as an integer, decode one decimal. Also add one line under the table: the
   date, "thinking on, 32K budget, greedy". Commit.
2. **Fresh-clone build test** (proves `xolotl-ggml.patch` + CMake): `git clone ~/AI/Xolotl /tmp/xolotl-fresh` (or the
   scratchpad), configure WITHOUT `STRATA_GGML_DIR` (CMake fetches llama.cpp 3cf03257 and applies the patch), build
   `strata strata-dense glm_generate ds41_generate`. Must be 0 errors. ~10 min CPU — not while a GPU benchmark runs.
   Disk: ~113 GB free on `/` now.
3. **Reconfigure the local builds:** `cmake -S . -B build-xolotl -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp` (all
   other cache options stay) + `ninja -C build-xolotl`; `cmake -S tools/vision -B build-vision
   -DLLAMA_DIR=$PWD/third_party/llama.cpp` + `ninja -C build-vision strata-vision`. Then a 1-request smoke per model.
4. **Push — ONLY after Mal says go** (he said: not until DSV4.1 is definitely up to scratch + the README lists every
   model's HumanEval + throughput). `git push github xolotl:main` — it fast-forwards over GitHub main (9d6529fc is an
   ancestor of xolotl); the local `git replace` graft is not pushed and not needed. Before pushing check: no secrets
   (`git grep -nE "sk-|hf_[A-Za-z0-9]{20}|or-v1-"`), no file > 50 MB in the pushed history, `verify/` untracked.
   Then (Mal's call): GitHub repo description + topics (draft in `docs/brand/XOLOTL_BRAND.md`: "Xolotl -
   DeepSeek-V4.1-Flash (552B MoE) on one gaming PC: a Strata engine..."; topics deepseek, deepseek-v4, glm, qwen, moe,
   llm-inference, offloading, speculative-decoding, cuda, local-llm, strata), social preview =
   `docs/brand/assets/github-social-preview-1280x640.png`. Possibly rename the repo to `Xolotl` IN PLACE (Settings ->
   Rename keeps stars + redirects; NEVER create a new repo with the old name). README header image already says
   "DeepSeek-V4.1-Flash on one gaming PC".
5. **The Gauntlet** (`tools/gauntlet/`, Mal's idea: a small benchmark so hard most models fail, to expose what quants
   cost): DESIGN.md has the rules and 10 task specs; tasks 01 (ledger root-cause) and 02 (semver precedence) are written
   and validated (`python3 tools/gauntlet/validate.py` -> both OK); the runner is mechanics-tested. **Write tasks 03-10**
   per DESIGN.md (each: workspace/, hidden/test_hidden_*.py, reference/, trap/, task.json; validate.py must print OK),
   keeping each <= ~400 lines and the whole run <= ~40 min at ~25 tok/s. Then calibrate: run it on DSV4.1 (Xolotl),
   and — when Mal OKs API spend — the DeepSeek API (`deepseek-flash` IS V4.1-Flash; port the env knobs from
   `verify/e2e/api/taskeval_api.py`: TASKEVAL_MODEL / TASKEVAL_KEY, no `reasoning_effort: none` on the API). Replace
   any task every model passes. Keep `hidden/` out of public pushes if it becomes a leaderboard.

## 6. Open questions for Mal (do not decide them yourself)

- **27B vs NInfer:** strata-dense is 3.5x behind on prefill (IQ3_S MMQ ~67 effective TOPS vs NInfer's int8 A8 GEMMs
  ~240) and 1.5x on decode (spec depth / acceptance). Port NInfer-class kernels into strata-dense (a project) or serve
  the 27B through NInfer and add `/v1/decide` there (`~/AI/quetza-workspace/voice-duplex/research/NINFER_DECIDE_MAP.md`).
  The open-jet fork (github cutlerbenjamin1-cmd/open-jet qwen38-perf: 151 tok/s on a 3090, bit-exact 7-draft MTP,
  K6/V5 KV) needs Q4_K_M / UD-Q4_K_XL files, not our IQ3_S, and its prefill is slower than NInfer's.
- **Qwen Flash quality upgrade:** NVIDIA's QAD NVFP4 Qwen3.8-Flash-Next (102 GB) — Xolotl reads NVFP4 since 0.1.44
  (W4A8 on the 4090); Mal is waiting for an abliterated version; needs ~102 GB free disk.
- MiMo: dropped (Mal). Its branch `origin/mimo` was never merged; weights deleted.

## 7. Known open items (not blockers)

- 27B prefill -7% vs production (newer ggml MMQ kernels in the prompt path; nsys both binaries to locate).
- GLM prompts of 17-256 tokens now shrink the elastic cache per request (correct; slightly slower than before the fix,
  which crashed instead). <= 16 tokens go through the decode loop.
- llama.cpp b11390 (CUDA MoE MMQ temp-buffer padding fix, 2026-10-04): check whether xolotl-ggml.patch covers it.
- `ple_parity` ctest fails for a missing model fixture (environmental, 55/56).

## 8. Traps that already cost time (read before touching anything)

1. **Semantic merge conflicts compile fine.** Renamed attributes (`_re`), X-macro type lists (`STRATA_*_FMTS*`), and
   monkeypatching shims vs a newer server's signatures all built and passed ctest while a model was broken. After ANY
   merge, run every model end to end (engine A/B + serve-level stub-engine GEN-line diff + a real tool call).
2. **`pkill -f <pattern>` kills your own shell** when the pattern is in your command (exit 144) — kill by PID or `pkill -x`.
3. **Production configs carry timing-dependent flags** (GLM `--skip-*`, `--arena-adapt`; DSV4.1 `--arena-adapt`):
   compare engines with them OFF, and measure the reference against ITSELF first (noise floor).
4. A short `run_tests`/serve prompt path can differ from the long one (GLM's c_stage OOM); test both.
5. Don't benchmark while compiling (CPU-pool experts share the CPU).
6. `git apply` inside a non-repo dir under another repo resolves against the parent repo — use
   `GIT_CEILING_DIRECTORIES` (the CMake check does).
7. `/tmp` does not survive reboots — keep artefacts under `~/AI/Xolotl/verify/` or the scratchpad and COMMIT code.
8. Background waits: use until-loops on a condition, not chained sleeps.

## 9. Key commands

```bash
cd ~/AI/Xolotl
git log --oneline -12                                   # session 3: e818dd6a .. c2c36a9e (+ README fill)
python3 tools/taskeval/taskeval.py score DIR...         # pass counts       (run: taskeval.py run URL DIR)
python3 tools/taskeval/taskeval.py speed DIR...         # tok/s during the tasks
python3 tools/gauntlet/validate.py                      # Gauntlet tasks valid?
python3 tools/gauntlet/gauntlet.py run URL DIR [--only 01,02] ; gauntlet.py score DIR...
bash verify/stub-ab.sh <tag> <shim> <config> [cwd]      # serve-level: GEN lines from a stub engine (STUB_STOP_ID=<eos>)
strata-ds41-quetza / strata-glm-quetza / strata-quetza / strata-dense-quetza   # Mal's wrappers (Xolotl)
```

## 10. UPDATE 17:15 (end of session 3) — read this first

- **README run, DSV4.1 done (thinking, 32K budget):** HumanEval **21/21**, tools **10/10**, agentic **3/3**; during the
  tasks: prefill 26 tok/s (short prompts - not representative), decode **26.4 tok/s**; 63,402 tokens, 2,973 s.
  GLM started 17:12, then Qwen Flash, then 27B (`verify/e2e/final-think.log`).
- **Gauntlet:** tasks 01-09 written AND validated (`python3 tools/gauntlet/validate.py` -> all OK):
  01 ledger root-cause, 02 semver, 03 ray tracer (pixel-compared, graphics), 04 log needle (20K tok, long context),
  05 regex engine (fuzz vs re + no catastrophic backtracking), 06 TTL+LRU cache (threads), 07 CSV money (3 dialects),
  08 SVG raster (nonzero/evenodd, graphics), 09 DST scheduler. Runner has a `run_python` tool now.
  **Task 10 (flaky-suite) is WIP:** only `workspace/` exists, and its visible tests currently fail even in file order
  (2 of 4) - they must PASS in file order and fail only in some orders. Fix the test design (e.g. make
  `test_tags_are_per_item` / `test_tagged_only_lists_stock` order-dependent only), then write `reference/inventory.py`
  (default `tags=None` + copy, `reset()` clears `_by_tag`, remove drops the name from `_by_tag`), `trap/` (fix only the
  mutable default), `hidden/test_hidden_order.py` (import the visible test functions and run them in 30 seeded
  shuffled orders), and `task.json`. Then validate.
- **Then (Mal asked):** run the Gauntlet on the DeepSeek API (`deepseek-flash` = V4.1-Flash; Mal OKed API use for this):
  give `tools/taskeval/taskeval.py`'s chat() the env knobs from `verify/e2e/api/taskeval_api.py` (TASKEVAL_MODEL,
  TASKEVAL_KEY -> Authorization header; drop `reasoning_effort: none`; URL https://api.deepseek.com), run
  `python3 tools/gauntlet/gauntlet.py run https://api.deepseek.com out/gauntlet-api`, then the same on local DSV4.1
  (strata-ds41-quetza, port 8150). Goal: the API should fail several tasks; replace any task everything passes.


---

## 11. UPDATE — session 4 (2026-10-16 ~17:40), read this INSTEAD of section 10

**Ended:** the Gauntlet is complete (10/10 written + validated) and has been calibrated against the API; 7 hardening
workers are in flight to raise its difficulty; the 4-model e2e chain is still running on the GPU.

- **Gauntlet is DONE as a benchmark**: all 10 tasks validate (`python3 tools/gauntlet/validate.py` -> 10x OK).
  Task 10 (flaky-suite) is written (visible suite passes in file order, fails ~1 run in 3 shuffled; trap = fix only
  the shared default list). The documented **exact-answer task format is now implemented** (`re.search` on the final
  assistant message) and `selftest.py` drives the real runner for 4 checks - commit `82d1aa64`. A task with neither
  hidden tests nor an `answer` now fails loudly instead of scoring PASS silently.
- **Measured API baseline (SUPERSEDED - see section 13 for the post-hardening number, 3/10): `deepseek-flash` scored 8/10** - it failed only 03-raytracer and
  05-regex-engine; 114,210 completion tokens, 700 s for all ten. Too easy => 7 tasks are being hardened right now
  (01, 02 -> invented release-ordering spec, 04, 06, 07, 08, 09), 10 queued. Target: the API fails ~7/10.
- **Conductor session 4** (relay name `conductor-1344159.1195555`): workers `ga-harden-{01,02,04,06,07,08,09}`,
  one slice per task, each gate = `validate.py <that task>` OK + a derivability table (every hidden assertion ->
  the documented sentence that states it). Reports land in `~/.quetza-data/conductor/reports/ga-harden-NN.md`.
  **Never run the full `validate.py` while a hardener is editing its task dir** - it will read a half-written tree.
- **Sequence when they land**: verify each diff myself (not the worker's summary), run
  `validate.py` over the frozen set + `selftest.py` (its check d re-verifies task 01), commit, then re-run the API
  (`TASKEVAL_MODEL=deepseek-flash TASKEVAL_KEY="$DEEPSEEK_API_KEY" TASKEVAL_NO_EFFORT=1 python3
  tools/gauntlet/gauntlet.py run https://api.deepseek.com out/gauntlet-api2`) to confirm the difficulty moved.
  Then dispatch the flaky-suite hardening (10) and the fairness audit over the FINAL set (the first audit worker was
  killed before it wrote anything - its subject was being rewritten underneath it).
- **GPU**: the e2e chain is self-driving (`verify/e2e/final-think.sh`, scores itself, ends with `ALL DONE`). GLM is
  slow because one HumanEval problem can consume the whole 32K thinking budget: measured 25,446 tok at 28.7 tok/s =
  **893 s for one problem** (others 35-75 s), 4/21 done in 22 min => GLM ~19:00, then qflash, then q27. Mal may stop
  the chain; the moment the GPU frees, run the local Gauntlet:
  `TASKEVAL_THINK_BUDGET=32768 python3 tools/gauntlet/gauntlet.py run http://127.0.0.1:8151 out/gauntlet-ds41`
  (serve ds41 on 8151 the way `final-think.sh` does). That API-vs-quant contrast on the hardened set is the
  deliverable; the e2e HumanEval numbers are saturated (21/21) and show little.
- **Nothing pushed.** Commits this session: `02acb010` (task 10), `ee0610ea` (API knobs), `80290ee2` (DESIGN table),
  `0ef1e51a` (runner robustness: a bad tool path or one task's exception can no longer kill a run), `82d1aa64`
  (answer grading + selftest). `README.md` is still dirty on purpose - its table waits on the e2e chain.


---

## 12. UPDATE — session 4 continued (2026-10-16 ~18:00): e2e chain STOPPED mid-GLM, Gauntlet frozen + hardened

**Mal's call at 17:54: kill the HumanEval e2e run; the rest resumes later.** Done as asked - it finished the
problem it was on, then stopped:

- **ds41 e2e: COMPLETE** - `verify/e2e/final-think/r_ds41.out`, 21/21 HumanEval, 10/10 tools, 3/3 agentic, 63,402
  tok, 2,973 s, prefill 26 tok/s, **decode 26.4 tok/s** (`taskeval.py score/speed`).
- **GLM e2e: STOPPED, 5 of 21 HumanEval attempted** - `verify/e2e/final-think/r_glm.out`:
  `HumanEval/0 PASS 75s` · `/8 PASS 37s` · `/16 PASS 35s` · `/24 PASS 893s (25,446 tok)` ·
  `/32 **fail** 1203s (33,792 tok = its own max_tokens ceiling)`. The tool cases and agentic loops never ran.
- **qflash and q27 never started.** The GPU is free again (744 MiB).
- **Resume the chain later with:** `bash verify/e2e/final-think-resume.sh` - it runs ONLY qflash then q27 on the
  same port/config/paths as the original driver and then writes `score.txt`/`speed.txt` over all four dirs (the
  GLM row stays the 5-problem partial; re-running GLM means re-running its block in `final-think.sh`).
  Both that script and `verify/e2e/gauntlet-local-ds41.sh` live under `verify/`, which `.git/info/exclude` deliberately
  keeps out of the repo - local files on this machine, NOT part of a fresh clone (which is why the gitignore trap on
  `04-log-needle/workspace/service.log` had to be fixed with a repo-level negation instead).
- **Note on the thinking budget (Mal is right, and it matters):** `TASKEVAL_THINK_BUDGET=32768` sets
  `reasoning_budget_tokens` - an ALLOWANCE a thinking model may spend, not a forced spend. GLM is the one that
  spends it (25,446 tok on HumanEval/24), and HumanEval/32 failed because its 33,792 max_tokens ceiling cut the
  answer off. A model that stops thinking earlier is never penalised.

**The Gauntlet is FROZEN and hardened** (all commits on `xolotl`, nothing pushed): 8 tasks strengthened in two
waves, each slice worker-verified AND re-verified by the conductor (`validate.py` per task + a derivability table
mapping every hidden assertion to the documented sentence). `validate.py` over all 10: **10x OK, exit 0**
(`/tmp/ga-frozen-gate.log`), and `selftest.py` a/b/c/d all PASS after check (d) was re-pointed at the three-module
task 01.

**Measured difficulty against `deepseek-flash`** (the control; a task it passes shows no quant gap):

| task | before | after hardening |
| --- | --- | --- |
| 01 ledger | PASS 9 steps | **FAIL** 12 steps / 22.4K tok |
| 02 release-order | PASS 7 steps | PASS 9 steps -> wave 2: 3 surfaces, 57 hidden tests |
| 03 raytracer | FAIL | FAIL (untouched) |
| 04 log-needle | PASS 12 steps | measured (`out/gauntlet-api-rest`) |
| 05 regex-engine | FAIL | FAIL (untouched) |
| 06 ttl-cache | PASS 10 steps | measured (`out/gauntlet-api-rest`) |
| 07 csv-money | PASS 10 steps | PASS at the 12-step ceiling, never submitted -> wave 2: running balance |
| 08 svg-raster | PASS 12 steps | **FAIL** 14 steps / **45.6K tok** |
| 09 dst-scheduler | PASS 12 steps | measured (`out/gauntlet-api-rest`) |
| 10 flaky-suite | PASS 12 steps | measured (`out/gauntlet-api-rest`) |

**Running right now:** (a) `verify/e2e/gauntlet-local/ds41/` - the LOCAL ds41 through the same frozen Gauntlet
(`bash verify/e2e/gauntlet-local-ds41.sh`; think budget 32768) - this is the API-vs-quant contrast; (b) the
fairness/determinism audit `ga-audit-final` -> `~/.quetza-data/conductor/reports/ga-audit-final.md`.

**Next:** finish the two measurements, then a single clean full-set API run for the official /10, then the README
table from `score.txt`/`speed.txt` (ds41 done, GLM partial, qflash/q27 to run), and the fresh-clone build test.


---

## 13. UPDATE - session 4, benchmark-hardening phase COMPLETE (2026-10-16 ~18:30)

**"Get the benchmark perfect before running models through it" is done. Nothing is running; the GPU is free; no
model has been through the hardened set yet.**

- **All 10 tasks hardened** (commits on `xolotl`, nothing pushed): 01 three-module ledger (33 hidden cases), 02
  `release-order` with three consistent surfaces (57), 03 raytracer (plus the audit's wall-clock assert removed),
  04 scaled to 12 ids in a 948-line log with a cross-footing count table and a one-clause exemption pair, 05
  un-puzzled regex semantics - possessive/longest-once, line-wise anchors, `.` matching newlines, an invented `~`
  (123), 06 cache with the purge-order contract and an 8-thread stress (58 asserts), 07 dialects + per-row rounding
  + a running balance (22), 08 curves and elliptical arcs with a documented N=32 flattening rule (38 cases),
  09 an exact `agenda()` + `blackouts.txt` + minimum notice (37), 10 three-module flaky hunt (60).
- **Verification, all re-run by the conductor on the settled tree:** `validate.py` **10/10 OK**; `selftest.py` a-e
  all PASS (e = the workspace cannot grade itself); ~100 repeat task-validations with **zero flakes** (whole gate 3x,
  06 10x, 03/08 10x each, 01/02/07/10 5x each); a **fresh worktree at HEAD validates 10/10**; an independent
  fairness audit returned **FAIR AS IS** over 246 assertions with R1-R4 resolved.
- **Not gameable from the workspace:** `write_file` refuses - and the pre-grading strip removes - `conftest.py`,
  `pytest.ini`, `sitecustomize.py` and every stdlib shadow the tasks import (`.py`, sourceless `.pyc`, package dirs,
  any path component). Measured: a planted `conftest.py` flipped 01's honest "13 failed, 20 passed" to a green
  "33 passed" before the fix. Every model-touching subprocess now runs with six variables (no credentials), proven
  by a probe that dumps `os.environ` from inside `run_python`.
- **CALIBRATED (2026-10-16, `out/gauntlet-api-final`): `deepseek-flash` scores 3/10** on the frozen hardened set -
  PASS 01, 02, 06 (all three at the 12-step ceiling, never submitting), FAIL 03, 04, 05, 07, 08, 09, 10; 246,337
  completion tokens, 1,453 s. It scored 8/10 before the hardening. The misses are narrow (03 one scene, 07/10 two
  cases, 04 two assertions; 05 collapses with 122 failures), i.e. the frontier sits on the set. Recorded in
  `tools/gauntlet/DESIGN.md` + `HARDENING.md`.
- **GPU: free. Mal is using it (a game) - do NOT start an engine without him saying so.** The local contrast is
  queued, not run: `bash verify/e2e/gauntlet-local-ds41.sh` (expect hours: ~20 min was measured for a single hard
  request at a 32K thinking budget). Then the pair to quote is API 3/10 vs quant N/10.
- **Also still pending Mal's call:** the e2e resume (`verify/e2e/final-think-resume.sh`) for qflash + q27, and the
  push to Strata-DSV4.1.


---

# 14. (HISTORY) the session-4 continuation brief (2026-10-16 ~19:00) - superseded by the START HERE block at the top

**Sections 11-13 below are history of how we got here. Read this section, then `TOOLS/GAUNTLET`'s docs, then work the
list. Nothing is running; the GPU is Mal's (he is gaming) — do not start an engine without him saying so.**

## 14.1 The two projects in this tree

1. **Xolotl** — one engine tree (DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.8-27B dense + the
   legacy V4-0731) serving every model, verified against each previous production build. Session 3 finished this; the
   wrappers point here; **nothing is pushed**.
2. **The Gauntlet** (`tools/gauntlet/`) — a new 10-task agentic coding benchmark, built and calibrated this session:
   the full-precision API scores **3/10** on it (8/10 before hardening). This is the session's real deliverable and
   where the remaining work is.

## 14.2 Read these first (in this order)

```bash
cat tools/gauntlet/README.md        # what the benchmark is, the 10 tasks, how to run/grade/add one, integrity, limits
cat tools/gauntlet/DESIGN.md        # the design rules, the task table, the calibration, the verification record
cat tools/gauntlet/HARDENING.md     # the evidence-backed levers for making tasks hard (with sources) - read before hardening anything
sed -n '1,80p' HANDOFF.md           # the session-4 block: where everything is, what was verified, the traps
```

## 14.3 State in one screen

- **Gauntlet: complete, gated, calibrated.** 10 tasks; `validate.py` -> 10x OK; `selftest.py` -> a-e PASS; ~100 repeat
  validations with **zero flakes**; a fresh checkout validates 10/10; an independent fairness audit returned
  **FAIR AS IS** over 246 assertions with its four findings all resolved; grading cannot be subverted from the
  workspace and no credential reaches model code.
- **Calibration (`out/gauntlet-api-final`, `deepseek-flash`, greedy): 3/10.** PASS **01, 02, 06** — each at the 12-step
  ceiling and never submitting; FAIL **03, 04, 05, 07, 08, 09, 10**; 246,337 completion tokens, 1,453 s. Recorded in
  `DESIGN.md`, `HARDENING.md`, `tools/gauntlet/README.md` and this file.
- **e2e chain: stopped mid-GLM on Mal's instruction.** `r_ds41` complete (21/21 he, 10/10 tool, 3/3 agentic, 26.4 tok/s
  decode); `r_glm` partial (5 of 21: PASS 0/8/16/24, FAIL 32 at its own 33,792-token ceiling, ~28 tok/s); qflash and
  q27 never started.
- **Nothing pushed; `README.md`'s model table has the V4.1 row filled from the measured run and the other three rows
  explicitly `pending`.**

## 14.4 Next steps, in order

1. **The local quant contrast (the number Mal actually wants).** Needs the GPU free and his go-ahead:
   ```bash
   bash verify/e2e/gauntlet-local-ds41.sh        # serves ds41 on 8151, waits, runs the 10 tasks, scores
   ```
   Expect **hours** (a 32K thinking budget cost ~20 min for one hard request). Overnight is the right window. Then
   the quoted pair is **API 3/10 vs quant N/10** — if the quant also fails everything the set needs a task it *can*
   pass, so check the spread, not just the total.
2. **Then optionally the e2e remainder** for the README table:
   ```bash
   bash verify/e2e/final-think-resume.sh         # qflash, then q27, then score.txt/speed.txt over all four dirs
   ```
   Then fill the three `pending` rows in `README.md` from `verify/e2e/final-think/score.txt` + `speed.txt`.
3. **If a task proves too easy in either run, harden it** — one `flash` worker per task via `qc-dispatch` with a
   packet, gate = `validate.py <that task>` + a derivability table (every hidden assertion -> the documented sentence
   that states it). Follow `HARDENING.md`; do **not** tighten a step budget to force failures.
4. **The push** to `Indras-Mirror/Strata-DSV4.1` — only on Mal's explicit OK, and decide first whether
   `RESUME_PROMPT.md` / `HANDOFF.md` (they carry local paths) ship or stay out, and keep `tasks/*/hidden/` out of any
   public leaderboard.

## 14.5 Rules for this project (Mal's standing constraints)

- **Measure before concluding**: a worker's "verified" is a claim — re-run the gate on a settled tree. Report
  `file:line` / command output, not impressions.
- **Do not run models without being asked**, and **never start an engine while the GPU is in use** — check
  `nvidia-smi` and ask.
- **Never run the full `validate.py` while a slice is editing its task dir** (it reads half-written trees).
- **Never `pkill -f` with a pattern that matches your own command line** (it has killed my own shell twice); kill by
  exact PID, or `tmux kill-session -t qc-<slice>` for workers.
- **Commit as you land each slice** — a reboot has already wiped uncommitted `/tmp` work once; verify the diff and the
  gate yourself before committing.
- **Nothing destructive or public without an explicit OK** (push, delete, model files, the README's public shape).

## 14.6 Acceptance for "this project is done"

- One clean full-set Gauntlet run on the API and one on a local quant, whose numbers are quoted in `README.md` and
  `DESIGN.md` (the API row is already there: 3/10).
- The README's four model rows all filled from real runs (V4.1 done, GLM/qflash/q27 pending).
- The fresh-clone build test (session 3's S3.5) passed — the last release blocker for the engine side.
- Mal's explicit go for the push.
