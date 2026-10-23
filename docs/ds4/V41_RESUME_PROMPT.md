# DeepSeek-V4.1-Flash: GSQ-RCO quant + Strata-DS4 port — HANDOFF / RESUME

> **NEW SESSION? Read [V41_START_HERE.md](V41_START_HERE.md) ONLY** - it is the complete, current briefing (~4K tokens).
> This file is the engineering LOG (history, every A/B, ~40K tokens): open it only for a specific question.

## RESUME PROMPT 2026-10-16 ~00:10 (session 13d) - paste the block below into a new session
WHERE WE GOT TO (all committed and pushed on `deepseek4`, CODE at `ad1b0f4c` (the park/unpark fix; only doc commits after it); logs `~/AI/v41-dev/s13/run9..run19*`):
- **PREFILL, 20,480-token prompt, quiet box, one-shot, prestage 1.5, no DSpark: batch 12288 = 387.6 tok/s (52.8 s)
  vs batch 8192 = 318.9 (64.2 s) -> +21.5%, and 12288 is now the serve default** (`strata-ds41.json`).  The whole win
  is one fewer whole-file-tier sweep: 114.5 GB of reads in 2 sweeps vs 157.2 GB in 3; the reader does the same
  4.95-5.15 GB/s either way.  In `--serve` at 512K with DSpark the same change was 66.85 -> 58.44 s (306 -> 350 tok/s)
  (run6).  At 12288 the split is dense 15.3 s / experts 35.8 s (file wait 20.4 s).
- **A shared card no longer fails a turn.**  `--serve` sizes the layer-major batch from the VRAM actually free
  (calibrated `need ~= 4.3 GiB + 0.95 MiB a column`), retries at HALF the width, clamps the chunk stage (two 1 GiB
  halves -> from free VRAM, 16 MiB floor), and floors/skips the expert-cache seed (a slot budget of 0 used to mean
  "the whole model": 142.38 GiB asked at 0.53 GiB free).  Measured: with a 2.5 GB neighbour the 12288 config ran at
  batch 6428 (233.9 tok/s) and at 9026 (251.1 tok/s) with 0 errors - the old binary answered HTTP 400.
- **Continuations reuse their prefix now**: `STRATA_SERVE_PARK_PARTIAL` (default on) unparks at the longest common
  prefix instead of demanding an exact one.  The continuation of a 21K-token conversation matched 21,165 of 21,166
  parked tokens (the chat template re-renders ONE token), so a 23,531-token prompt became 2,326 tokens of prefill
  (`reused=0` before, `reused=21165` after - that is the proof).  The wall-clock delta (265.9 -> 24.7 s) is CONFOUNDED:
  run15's arm ran at a capped batch 1114 on a loaded box and run19's at 5837 on a quiet one, so do not quote 10.8x as
  this fix's measure; a like-for-like pair is still owed.
- **Read path: CLOSED, null.**  `STRATA_WAVE_PARTS` 4/16 = 0% at prompt size (s11's 32x8 stands); the offset-sorted
  experiment SEGFAULTED (exit 139) and is REVERTED - do not re-attempt blind (the whole scattered-vs-dd gap is ~9%).
- Decode is untouched this session (~17-30 tok/s with DSpark; the read path is not its bottleneck).

THE GAP TO 400-500 tok/s PROMPT: 350 (serve) / 388 (one-shot) today.  The measured route is ONE SWEEP PER PROMPT
(batch 20480: ~82 GB of reads instead of 114.5 -> est. ~45-48 s, ~430-450 tok/s), plus a faster dense half (it already
drops 20.5 -> 15.3 s from 8192 to 12288).  Its blocker is now NAMED (see item 2).

READ FIRST: docs/ds4/V41_START_HERE.md block "Session 13c" (the full write-up), then
docs/ds4/V41_NEXT_PREFILL_TERNARY.md.  The wrapper's own config is tools/v41/serve/{strata-ds41.json,ds41-engine.sh}.

ORDER FOR THE NEXT SESSION:
1. **Check that the restarted wrapper took the new defaults.**  Mal must restart `strata-ds41-quetza`; then
   `/tmp/strata-ds41.log` on a long turn must show `serve: VRAM <F> GiB free -> prompt batch N (of 12288)` and, after a
   side call, `unparked the N-token conversation at lcp M (partial: re-feeds from there)`.  If it still shows 8192,
   the json/launcher edit did not reach the running engine.
2. **16384 / 20480 (one sweep per prompt) - the STATE TENSORS are the blocker, and they are now named.**  Measure with
   `STRATA_VRAM_TRACE=1`; it prints `vram trace: state NC=N wants X GiB (i_eng 24 x 256 = Y GiB each, ...)`.  The
   numbers: NC=8192 -> state 2.57 GiB (i_eng 2 x 0.19 GiB); NC=12288 -> 3.86 GiB (i_eng 2 x 0.28 GiB).  So ~0.32 MiB
   a column, and almost none of it is the engram - it is the per-column tensors created in `Ds41Dense::Impl::alloc_state`
   (ds41_dense.cpp ~lines 920-950: i_emb, routed, shexp, o_fn, o_ids/o_wts, cand, sel/selm, i_widx/i_wmask,
   i_nvis/i_cslot/i_cpos, i_sidx/i_sslot, i_lastblk).  Cheapest first: (a) `cand`/`sel`/`selm`/`o_ids` exist only for
   layers that run an indexer - allocate them per those layers, or drop them when no indexer runs; (b) fp16/bf16 for
   i_emb / routed / shexp / o_fn (F32 today) halves ~1 GiB at NC=8192; (c) only then consider structural changes to the
   elastic-column path.  A 16384 batch needs ~19.5 GiB of batch VRAM (4.3 + 0.95x16384/1024), so ~2.1 GiB less state
   is what makes it fit beside DSpark on a clean card.
3. **Re-run the batch A/B with 16384 in the set** (`~/AI/v41-dev/s13/run17.sh` - it already gates on a CLEAN card,
   >= 23 GiB free, yields to `audiocpp_cli`, and stops at the cutoff), 3 runs a side, alternating, adopt only on
   > 3% with spread < 4.3%.
4. **Check `STRATA_DSV41_FQ_OP`** (the fused activation-rounding op): bit-identical on CUDA and CPU, ~3000 fewer tiny
   kernels per decode token, measured 26.3 -> 27.7 tok/s decode - but it needs the llama.cpp tree patched with
   `tools/v41/ggml/dsv41-fake-quant.patch`.  Verify whether this checkout's tree is patched; if not, applying it is
   the cheapest remaining win, and measure it at PROMPT width too (the dense half is 15.3 s of 52.8 s at 12288).
5. Decode levers, all previously MEASURED and unadopted: q4_K/q5_K dense requant (attn_router 17.11 -> 13.52 ms;
   decode 26.6 -> 29.75 tok/s) - Mal's quality gate first (code + agent ppl/KL against a same-arena q8 control);
   the elementwise "zoo" (5.45 ms/token, 3288 kernels < 5 us).
6. The ONE-SHOT path has NO VRAM cap by design (a benchmark path must fail loudly, not silently narrow).  If a
   friendlier failure is wanted, add a pre-flight check that prints needed vs free GiB before the alloc dies.
7. DO NOT re-attempt blind (all measured dead ends): the offset-sorted wave reader (segfault), a second-drive blob
   mirror, a smarter VRAM policy than LRU (+-1%), `STRATA_CHUNK_PF_MIN` whole-layer prefetch (neutral/negative),
   a larger arena (62/65 GiB neutral), zstd on the expert region (0.92).

RULES IN FORCE (learned the hard way tonight - each one cost a wasted leg):
- A one-shot measurement leg needs a CLEAN card: >= 23 GiB free VRAM at start (its dense half takes ~12 GiB and the
  un-capped batch ~12 GiB more; at 9.5 GiB free the allocation fails - run13).  `--serve` needs no such care now.
- Gate on VRAM (`memory.free` / `memory.used`), never on "any compute app": a foreign Chrome holds a ~51 MiB CUDA
  context that never clears (this project's own note).  Use `pgrep -x <name>` (process NAME): `pgrep -f` matches your
  own command line and will park you forever (it did, for 2 minutes tonight).
- Yield the card to a peer's announced hold; the peers coordinate over relay.  This session: `mal` (CPU crawl, an
  embed phase at ~00:50 that wants the card, and he restores `quetza-stt-whisper` - 2.5 GB - when the card LOOKS idle),
  `xolotl-video` (Drama-3 TTS, 22 GiB, takes the card in announced holds; that stray whisper was his).  Tell mal to
  gate his restore on VRAM (< 2 GB for 3 consecutive minutes), not on "no compute apps".
- Never rebuild a build dir a run is using: `build-ds4-gpu` is the wrapper's engine, `build-ds4-dspark` is the
  one-shot profiling harness.
- The harnesses: run15 (shared card + cap), run17 (batch A/B), run18 (serve smoke of the default), run19 (park/unpark).
  For an order-only change the gate is a byte-identical `.tok` - strip memguard's banner first, it embeds a PID.

```text
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4
deepseek4:main - pushing is OK'd).  READ FIRST: docs/ds4/V41_START_HERE.md block "Session 13c", then
docs/ds4/V41_RESUME_PROMPT.md's top section (session 13d: what tonight achieved and the ordered plan).

State: prefill on a 20K prompt is 387.6 tok/s one-shot / ~350 in serve at the adopted --prompt-batch 12288 (was
318.9 / 306 at 8192 - the win is one fewer whole-file-tier sweep: 114.5 GB in 2 vs 157.2 GB in 3).  A shared card no
longer fails a turn (the batch is sized from free VRAM and retried at half width; the chunk stage and the cache seed
were fixed too), and a continuation after a side call now reuses its 21K-token prefix (265.9 -> 24.7 s).
Target is still 400-500 tok/s: the route is ONE SWEEP PER PROMPT (batch 16384/20480), whose blocker is the
NC-scaled state tensors (2.57 GiB at NC=8192, 0.32 MiB a column; the engram is only 0.19 GiB of it) in
ds41_dense.cpp's alloc_state.  Shrink those, then A/B 16384 with ~/AI/v41-dev/s13/run17.sh.

Rules: one-shot legs need a CLEAN card (>= 23 GiB free); gate on VRAM used, never "any compute app"; `pgrep -x`, not
`pgrep -f`; never rebuild a build dir a run uses; yield the 4090 to the peers' announced holds (relay).  Mal must
restart strata-ds41-quetza for the new defaults to take effect - check /tmp/strata-ds41.log for "prompt batch ... (of
12288)" and "(partial: re-feeds from there)".
```


UPDATE ~20:25 (READ FIRST): the shared-card item is CLOSED and verified in the harness.  A foreign 4 GiB VRAM holder
(Whisper's role) made the OLD binary answer HTTP 400 on turn 1; the patched engine completes 3/3 turns + the s11 smoke
with 0 errors (correct, not fast: turn 1 105.2 s at a capped B=4281, turn 2 265.9 s at B=1114).  In: (1) `--serve`
sizes the layer-major batch from free VRAM (calibrated need ~= 4.3 GiB + 0.95 MiB/col) and retries at half width;
(2) the chunk stage is clamped from free VRAM (16 MiB floor) - it was a hard 2x1 GiB demand, the 503's cause;
(3) the cache seed no longer asks 142 GiB when the card is tight (free <= margin made slot_gib 0 = "no limit"): it
skips the cache, or retries with the cache off.  NEXT, in order: (a) READ PATH - MEASURED and exhausted at prompt size
(run16, quiet box: WAVE_PARTS 4/16 = 0%; 5.16 GB/s of reader time; tokens identical once memguard's PID banner is
excluded from the .tok compare).  STRATA_READ_SORTED was tried and SEGFAULTED (exit 139) -> REVERTED, do not
re-attempt blind (the whole scattered-vs-dd gap is ~9%, near the noise).  So the PREFILL lever is the batch WIDTH
(~82 GB re-read per layer-major sweep), i.e. the VRAM work above.  (b) name the fixed ~2.19 GiB buffer that forces
the 4.3 GiB fixed term - `STRATA_VRAM_TRACE=1` now prints the state tensors' sizes, so any traced leg names it;
(c) DONE, ADOPTED: the 8192-vs-12288 prompt-batch A/B (run17, 3 runs a side, spread < 0.5%) gives **12288 = +21.5%**
on a 20K prompt (387.6 vs 318.9 tok/s; 114.5 GB of file-tier reads in 2 sweeps vs 157.2 GB in 3) - **12288 is now the
serve default** (strata-ds41.json + STRATA_PRESTAGE_GIB=1.5).  Next for prompt speed: the batch's VRAM is dominated by
the NC-scaled state tensors (2.57 GiB at NC=8192, 3.86 at 12288; the engram is only 0.19/0.28 each) - shrink those to
fit a wider batch, then 16384.  Next, also: a serve-stack smoke on the new default (the A/B was the one-shot path).
Logs: ~/AI/v41-dev/s13/run9..run17*.  Studio/video work is xolotl-video's now.
UPDATE ~19:40: production harness check found: Quetza's Whisper STT (2.5 GB VRAM) makes batch 12288
OOM -> REVERTED to 8192 (12288 opt-in via STRATA_PROMPT_BATCH); even 8192 OOMs on a 2nd long prompt with Whisper
up -> free-VRAM-aware batch sizing + OOM recovery: DONE in 13c (see the ~20:25 entry).  Park did not unpark in the
harness (template re-render, see START_HERE) -> resume at the longest common prefix below the snapshot.
DONE from 13b: the park/unpark continuation fix (commit ad1b0f4c).  The continuation matched 21,165 of 21,166 parked
tokens and the old code demanded an EXACT prefix, so it missed by one token and re-read everything: turn 2 went
reused=21,165 tokens (cached_tokens 21,165 in the harness's own usage) where the old code reused 0.  Do NOT quote the
265.9 -> 24.7 s delta as the fix's measure - those two arms ran at different batch widths on differently loaded boxes
(the mechanism, reused=0 -> 21,165, is the evidence; a like-for-like timing pair is still owed).
STRATA_SERVE_PARK_PARTIAL=0 restores the old behaviour.
UPDATE ~18:55: production config (DSpark on) confirmed: 20K prompt 66.85 -> 58.44 s (306 -> 350 tok/s).  Next:
park DSpark weights in host RAM during batch prompts (~5 s of 58), then the 16384 OOM (1.42 GiB alloc), then the read
pipeline.  Video: ~/AI/xolotl-video (Remotion, 3:20 visual cut, out/xolotl-v1.mp4; `npm run studio` to edit; text
in src/facts.json).  Logos: docs/brand/out/logo-circle/ (Mal's favourite design made circular).
UPDATE ~18:35: prompt batch 12288 + prestage 1.5 ADOPTED in serve (20K prompt 60.4 -> 53.4 s, 382 tok/s); mirror
dead for good.  NEXT for 400-500: find the 1.42 GiB alloc that OOMs batch 16384 (likely the dense compute buffer at
width 16384) and shrink it; then the read pipeline (file wait 20 s of 54).  Skill video-explainer added (Claude +
Quetza).  Mal's target = prompt 400-500 tok/s.  ~/AI/v41-dev/s13/run4.sh -> run4.out: 20K prompt at batch
8192/12288/16384/20480 with STRATA_PRESTAGE_GIB 3.0/1.5/0.75 (does a bigger batch fit when prestage shrinks?), then
the second-drive mirror rebuilt on NVME1TB and A/B'd on full sweeps (s12 found nothing on fresh 8000 - if again
nothing, the limit is the engine's read pipeline, not the drive: chunk prof reader wall vs bytes says which).
A Phison drive may arrive.  Brand/video (docs/brand/, uncommitted on purpose): VIDEO_SCRIPT.md (Mal reads it
himself), VIDEO_PLAN.md (coded Remotion explainer), geometric logo takes from klein in
/media/mal/Crucible/AI-ART/ComfyUI/output/Xolotl/geo*.png.

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4
deepseek4:main - pushing is OK'd).  READ FIRST: docs/ds4/V41_START_HERE.md block "Session 13b", then
docs/ds4/V41_NEXT_PREFILL_TERNARY.md.

1. Check the unattended batches: cat ~/AI/v41-dev/s13/run3.out (prompt-batch sweep + ternary screen) and
   ~/AI/v41-dev/s13/run3b.out (the PARK GATE) and run3c.out (mesh screen L5).  KNOWN: prompt path NVMe-bound (file wait
   27 of 40 s experts on a 20K prompt, 328 tok/s); --prompt-batch 16384/32768 OOM (needs buffer sizing).  (park gate: - run3's own gate failed on a wrong driver path; logs in ~/AI/v41-dev/s13/run3/).  It built park/unpark
   into build-ds4-dspark, ran the PARK GATE (must print "continuation tokens IDENTICAL"; the B:1 log must show
   "unparked"), the --prompt-batch 8192/16384/32768 sweep on a 20K prompt, and the meshapplied ternary screen
   (layers 0, 5).
2. If the park gate is IDENTICAL: rebuild production (ninja -C build-ds4-gpu ds41_generate, no engine running), commit,
   tell Mal to restart strata-ds41-quetza.  It removes the 70 s re-read after every Quetza side call.
3. Prompt speed (Mal's goal: ~800 tok/s like sybil-solutions/dsv41-flash-offload): adopt the best --prompt-batch in
   tools/v41/serve/strata-ds41.json if it fits VRAM and wins; then A2 (keep the VRAM expert cache through small batch
   turns: "experts 0 hit" + "cache open" in every batch turn) and A0's chunk-prof reading (NVMe vs PCIe bound).
4. Ternary: read mesh_screen summary - mesh better than ours where ours is PTQ1_0 -> plan B2 (engine class with
   rotated activations); otherwise park it.
5. Brand: Mal reviews docs/brand/out/ (Xolotl) - do not rename / push brand files without his go.

RULES: one-shot runs through bash tools/ds4/memguard.sh 86 72 -- ...; --serve through ds41-engine.sh /
taskeval/engine.sh (flock + scope).  >= 75 GB MemAvailable.  Kill by exact PID.  Never rebuild a build dir a run
launches from.  Gate on VRAM used (>= 2 GB) not "any compute app" (a foreign Chrome holds a CUDA context).  A/Bs
need >= 3 runs a side (spread 1.5-4.3%).  After a reboot: udisksctl mount -b /dev/nvme0n1p3 -t ntfs3.
```

## RESUME PROMPT 2026-10-15 ~17:00 (end of session 13) - paste the block below into a new session

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4
deepseek4:main - pushing is OK'd; GitHub = Indras-Mirror/Strata-DSV4.1).  READ FIRST: docs/ds4/V41_START_HERE.md
section 5c block "Session 13" (newest), then only what a question needs.  Traces / logs: ~/AI/v41-dev/s13/.

STATE (RTX 4090 + 90 GB RAM), all committed + pushed, production build-ds4-gpu rebuilt.  Serve defaults UNCHANGED
since session 11.  Decode today (serve flags, one-shot 256): DSpark python 27.96 / prose 16.33 / code 26.68 tok/s.

SESSION 13 CLOSED ALL THREE PLANNED LEVERS by recording the verify passes' routing (STRATA_ROUTE_TRACE) and replaying
it offline (tools/v41/sim/route_sim.py, calibrated to the engine's own counts within 0.2%):
- smarter VRAM policy than LRU: Belady ceiling only 4-8%; LRU windows / LRU-2 / LFU-decay / TinyLFU all within +-1%
  of today's LRU (window 4 is already best).  Do not retry without a new signal of NEXT-PASS routing.
- in-pass draft pruning: oracle -4.8%, a real rule -0.7% (inside the noise).
- VMM refactor: already done by session 11's elastic columns.
- also measured: VRAM swap demotions cost ~0; plain decode is slower than DSpark on every prompt (prose too).
Verify-pass cost model: ms = 25.7 + 5.7 x rows + 0.49 x PCIe misses + 2.93 x file misses (R2 0.977).

NEXT candidates (unmeasured, Mal to pick): lower-precision DSpark draft experts (2.1 GiB of Q2_K on the card; K < 64
costs acceptance, lower bits might not) -> more trunk slots; file-tier misses (2.9 ms each, ~17% of a pass); the
dense half's fixed ~26 ms a pass.  The expert tier is capacity-bound.

GOTCHA: after a reboot "SSD NVME" (/dev/nvme0n1p3, the GGUF) may not be mounted:
udisksctl mount -b /dev/nvme0n1p3 -t ntfs3.

RULES: one-shot runs through bash tools/ds4/memguard.sh 86 72 -- ... (it takes the GPU lock; never wrap it in
flock); --serve runs NOT through memguard.  Need >= 75 GB MemAvailable.  Kill by exact PID only.  Never rebuild a
build dir while a run launches from it.  A foreign headless Chrome holds a 51 MiB CUDA context: gate on VRAM used
(>= 2 GB) or pgrep -x ds41_generate, not "any compute app".  Any A/B needs >= 3 runs a side (spread 1.5-4.3%).
```

## RESUME PROMPT 2026-10-15 ~06:00 (end of session 12) - paste the block below into a new session

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4
deepseek4:main - pushing is OK'd; GitHub = Indras-Mirror/Strata-DSV4.1).  READ FIRST: docs/ds4/V41_START_HERE.md
section 5c block "Session 12" (newest), then only what a question needs.  Logs / scripts of that session:
~/AI/v41-dev/s12/ (run1 mirror A/B, run2 serve turns + controls, run3/run6 the 64x16 false positive, run4 the
arena-32 mirror control, run5 the chunk prof).

STATE (RTX 4090 + 90 GB RAM), all committed + pushed, production build-ds4-gpu rebuilt:
- Serve defaults UNCHANGED from session 11: route bias x0.5 + file 0.3 + --arena-adapt + arena 64 GiB, DSpark
  (exact speculative sampling) + confidence trim 0.6, --kv-fp4, elastic state columns, batch-seeded DSpark tail,
  chunk prefetch OFF, wave reader 32 threads x 8 parts, D3 file reads 32 x 32.
- MEASURED AND CLOSED - do not redo: (a) the second-drive blob mirror (`--build-mirror` / `--check-mirror`,
  STRATA_BLOB_MIRROR, 84 GiB on NVME1TB, share 0.28): the pattern is +33% in the rb2 microbench (6.22 -> 8.29 GB/s)
  and does NOTHING in the engine, at arena 64 or in the arena-32 positive control - the file tier is not
  bandwidth-bound; (b) warm_reseed (STRATA_WARM_RESEED, default OFF): inside the bench's spread and it seeds more
  file-tier blobs; (c) wave reader 64 x 16: a FALSE POSITIVE (29.56 s in one run, 31.52 s repeated, vs 31.51/31.98
  baselines).
- The file tier is not the bottleneck in a batch turn: chunk prof (MIMO_CHUNK_PROF=1) gives the wave reader
  5.2-5.3 GB/s over 22.7 GB with the caller waiting on reads for only 35-159 ms of a 30 s prompt; the "experts"
  phase is the arena experts' PCIe DMA / prestage / device sum.  Levers that pay: VRAM slots (2.6 GiB = 10-20% of a
  python verify pass) and the dense half.
- **The 512K turn bench's run-to-run spread is 1.5-4.3%** (same config back to back; the later run is slower).  Any
  A/B needs >= 3 runs a side - a single pair produced the 64 x 16 false positive.
- Done 2026-10-15 (Mal's go): the 84 GiB mirror file is DELETED and session 11's hourly chronos nudge cron is
  removed (pre-removal crontab at ~/.quetza-data/crontab-backup-2026-10-15.txt; re-arm before an overnight run).
- A FOREIGN headless Chrome (agent-browser, not ours) holds a 51 MiB CUDA context, so the harnesses' gpu_busy()
  ("any compute app") never clears: gate on VRAM used instead (>= 2 GB = busy).
- Relay: --name fix live (untracked, ~/AI/QuetzaCodetl/wrappers/tools/relay).

NEXT - Mal's pick for this session, in docs/ds4/V41_START_HERE.md "### NEXT (session 13)": (1) in-pass draft
pruning (a verify pass runs 1+depth rows and its expert bytes scale with the width - 72/91/118/135 ms for 1-4 rows;
drop TRAILING draft rows mid-pass once their unique misses are known to be expensive: they are just not verified
this pass, so the distribution stays exact - measure time per token, NOT tokens/pass); (2) a VRAM replacement
policy smarter than LRU (score victims by the predicted next use the prefetch path already computes, add a
frequency term, or a prediction-based warm set across a batch turn - the released-set warm_reseed of session 12
showed nothing); (3) the VMM refactor for the prompt state's ~2.6 GiB (cuMemAddressReserve/Map with stable virtual
addresses so cached graphs keep their pointers, giving the dense half's NC columns back to the expert cache: +2.6
GiB = 10-20% of a verify pass; only worth it with a decode-side measurement, gate on bit-exactness with
STRATA_PRESTAGE_GIB pinned).  EVERY one of these needs >= 3 runs a side - the bench's identical-config spread is
1.5-4.3% and a single pair already produced the 64 x 16 false positive.

RULES: one-shot runs through bash tools/ds4/memguard.sh 86 72 -- ... (it takes the GPU lock; never wrap it in
flock); --serve runs NOT through memguard (engine.sh / ds41-engine.sh: flock + systemd-run scope).  Need >= 75 GB
MemAvailable.  Kill by exact PID only; never pkill -f a pattern in your own command line.  Never rebuild a build
dir while a run launches from it ("Permission denied").  ninja with an unknown target builds NOTHING.
```

## RESUME PROMPT 2026-10-15 ~03:30 (end of session 11) - paste the block below into a new session

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4 deepseek4:main
- pushing is OK'd; GitHub = Indras-Mirror/Strata-DSV4.1).  READ FIRST: docs/ds4/V41_START_HERE.md section 5c block
"Session 11" (newest; the complete record of the overnight session), then only what a question needs.

STATE (RTX 4090 + 90 GB RAM), all committed + pushed, production build-ds4-gpu rebuilt:
- Serve defaults (tools/v41/serve/ds41-engine.sh + strata-ds41.json, used by the strata-ds41-quetza wrapper): route
  bias x0.5 + file 0.3 + --arena-adapt + arena 64 GiB, DSpark speculative decode (exact sampling) + confidence trim
  0.6, --kv-fp4 (bit-exact), elastic state columns, batch-seeded DSpark tail, chunk prefetch OFF (on-demand reads, 32 threads x 8 parts),
  D3 file reads 32 x 32.  Production e2e: HumanEval 20/21, tools 10/10, agentic 3/3, real Quetza turns work.
- 512K serve: 14.00 GiB free after the dense half (was 10.46), turns: fresh 8000 ~37 s, +272 ~15 s, +1000 ~20 s
  (were 51 / 33.5 / 40; +272 now ~11-12 s with the wave-reader parts), decode ~21-34 tok/s on the 4-turn bench.
- third_party/llama.cpp is UNTRACKED: ggml changes ship as tools/v41/ggml/*.patch (kv-fp4.patch is applied).
- When A/B-ing bitwise: pin STRATA_PRESTAGE_GIB (the prompt prestage region follows free VRAM and moves numerics).

OPEN / IDEAS (ask Mal before big ones):
1. Decode is PCIe-bound (~70 of ~130 ms a python verify pass is expert DMA).  Measured, no clear gain: q* CPU share
   (fast PTQ1_0 kernel now; Q3_K / Q2_0 CPU paths unchanged), compression (7.5%), striping (no free NVMe).  Untried:
   in-pass draft pruning (drop trailing draft tokens mid-pass when their unique misses are expensive - lossless,
   they are just not verified); a smarter VRAM replacement than LRU.
2. Small turns: < 128 new tokens run as exact 16-token passes (~7-12 tok/s; +60 = 6.6 s), >= 128 as a batch (+150 =
   ~10 s, was 17.6).  Decode right after a batch turn starts colder (the VRAM cache is re-seeded from the rank list,
   losing the LRU warmth) - keeping the cache's residency across a batch would help both.
3. Prefill floor for batch turns: +272 tokens = dense ~3 s + experts ~7.7 s, the experts = NVMe reads of the routed
   file tier now at ~5.3 GB/s (the drive's ceiling ~5.4) - next: read fewer bytes (bigger RAM arena) or overlap the
   reads with the dense half of the same layer.
4. Merge question: ~/AI/research/MERGE_REPORT.md (glm <-> deepseek4 closest; monorepo ~3-5 days).
5. Relay: --name fix is live in ~/AI/QuetzaCodetl/wrappers/tools/relay (untracked); Quetza's bundled relay.ts was
   regenerated but needs a Quetza rebuild (not committed - that tree has others' uncommitted work).

RULES: one-shot runs through bash tools/ds4/memguard.sh 86 72 -- ... (it takes the GPU lock; never wrap it in flock);
--serve runs NOT through memguard (engine.sh / ds41-engine.sh: flock + systemd-run scope).  Need >= 75 GB MemAvailable.
Kill by exact PID only; never pkill -f a pattern in your own command line.  Never rebuild a build dir while a run
launches from it (the binary swap fails the launch: "Permission denied").  ninja with an unknown target builds NOTHING.
```

## RESUME PROMPT 2026-10-14 ~21:10 (end of session 10) - paste the block below into a new session

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4 deepseek4:main
- pushing is OK'd; GitHub = Indras-Mirror/Strata-DSV4.1).  HEAD at handoff: a5075937 (+ a docs commit).  Overnight run:
Mal is asleep - work autonomously, measure before concluding, keep tests small, commit + push verified work.

READ FIRST: docs/ds4/V41_START_HERE.md section 5c blocks "Session 10b" then "Session 10" (newest first), then
~/AI/research/REPORT.md (FreeToken + llama.cpp PR #29887 survey, ranked ideas at the end).

STATE (RTX 4090 + 90 GB RAM):
- Quality (real tasks, 21 HumanEval + 10 tool calls, greedy): route bias x0.25 20/21, **x0.5 20/21 (Mal's pick)**,
  x0.75 19/21, 1.0 (old default) 16/21; tool calling 10/10 everywhere.  Harness ~/AI/v41-dev/taskeval (taskeval.py,
  arm.sh / sarm.sh, smoke.py, engine.sh + cfg.json run the build-ds4-dspark binary).
- x0.5 lossless speed ladder (one-shot DSpark, 128 tok, docs/python tok/s): 16.7/14.6 -> arena-adapt in D3 22.3/22.9
  -> draft pruned (STRATA_DSPARK_KEEP=tools/v41/serve/dspark_expert_counts.txt:64) ~24.6/26.2 -> --arena-gib 68
  python 27-28.4 (72 GiB hits the 4 GiB RAM floor: unusable; serve = 64).  FreeToken's published number: 22-25 tok/s
  DS-V4-Flash on an RTX 5090.
- DSpark in --serve: CODED (opt-in STRATA_SERVE_DSPARK=<sidecar>), exact speculative sampling (unit test
  ~/AI/v41-dev/taskeval/spec_accept_test.cpp), line protocol temp 0 / temp 1 ~25.5-25.9 tok/s.  A CRASH on 2nd
  requests (CUDA invalid argument in Ds41DSpark::step) has a fix BUILT but NOT TESTED (stale gallocr addresses).

DO, IN ORDER (Mal's go for all of it, 2026-10-14):
0. Check the machine first: Mal ran a big zstd DB copy to Team CX2 (CPU ~8 cores) - wait until `uptime` load is low
   and no zstd runs before any GPU/CPU-heavy test.  GPU-sharing: relay peer voice-duplex/mal asks before GPU use.
1. Verify the DSpark gallocr fix: ~/AI/v41-dev/taskeval: `DRIVE_ENV="STRATA_SERVE_DSPARK=/media/mal/NVME1TB/Models/
   V41-Sidecar/V41-MTP-Vision.Q2_K.gguf STRATA_DSPARK_DEPTH=4 STRATA_DSPARK_KEEP=/home/mal/AI/Strata-DS4/tools/v41/
   serve/dspark_expert_counts.txt:64" python3 drive_dsp.py crash2.json crash2.log 262144 crash2_reqs.json` (arm_args
   holds "--arena-adapt --route-bias 0.5 --route-bias-file 0.3 --arena-gib 64") - must run all 3 requests.  Then the
   one-shot DSpark tokens/pass re-check (did stale drafts cost acceptance?), then HTTP: `bash sarm.sh` with HE=1
   (smoke.py + HumanEval through DSpark serve; expect ~20/21) - see spipe2.sh.
2. 512K context (Mal wants it): serve VRAM after the dense half is 10.36 GiB at 512K vs 11.83 at 32K and 14.28 one-shot,
   and HTTP DSpark serve ran only 15.9-17.6 tok/s (77% hits).  Find what the KV / snapshot / prompt-batch buffers cost
   and shrink them (Mal: Strata-dense uses int8 K / int4 V - see ~/AI/Strata* dense docs; check DS4's cache dtypes),
   so 512K serve keeps expert slots.  Re-test the crash sequence at 524288.
3. q* (FreeToken, REPORT.md #1): STRATA_TIER_BATCH_PCIE=0.85 / 0.75 at x0.5 + arena-adapt + K=64 (docs/python, 128
   tok, compare verify ms/pass); if it helps, pick the PCIe share by recency (FreeToken offload_kernels.py:261-345).
4. Serve defaults + Quetza: tools/v41/serve/ds41-engine.sh -> build-ds4-dspark binary (or rebuild build-ds4-gpu),
   args --arena-adapt --route-bias 0.5 --route-bias-file 0.3 --arena-gib 64, env STRATA_SERVE_DSPARK + KEEP + DEPTH 4;
   strata-ds41.json ctx (512K if item 2 lands, else 256K); test the strata-ds41-quetza wrapper end to end (a real
   Quetza turn with tool calls).
5. GitHub README + HF card: session 10 results (quality table, speed ladder, DSpark serve), push.
6. Mal asked: how mergeable are Strata GLM / Strata Qwen (~/AI/Strata) / Strata dense / Strata-DS4 - a short
   divergence assessment (shared files, diff size, what a common core would be).  Read-only.
7. Relay/chronos: `relay ask --name X` re-registers the calling window as X (found tonight) - fix or document in the
   relay CLI (~/.local/bin/relay, source ~/AI/QuetzaCodetl/wrappers/tools) and the chronos skill; hourly wake for this
   session = crontab `17 * * * * /home/mal/.local/bin/chronos/ds4-overnight-nudge.sh` (line removed at handoff; the
   script targets relay name strata-dspark - `relay up strata-dspark` first) + `relay wait --timeout 7200` as a
   background task, re-armed after every wake.

RULES: one-shot runs through bash tools/ds4/memguard.sh 86 72 -- ... (it takes the GPU lock; never wrap it in flock);
--serve runs NOT through memguard (engine.sh / ds41-engine.sh: flock + systemd-run scope).  Need >= 75 GB MemAvailable.
Kill by exact PID only; never pkill/pgrep -f a pattern in your own command line.  bench scripts: source the gen()
helper from ~/AI/v41-dev/s10/run3.sh - gen() clobbers the globals tag/p/n (never use them as loop variables).
Bias-on text depends on residency: compare verify ms/pass and tokens/pass, +-5% noise.  Build: nice -n19 ninja -C
build-ds4-dspark ds41_generate.  anaconda's `patch` is corrupt: /usr/bin/patch.
```

## RESUME PROMPT 2026-10-14 ~06:10 (end of session 9) - paste the block below into a new session

```
Continue Mal's DeepSeek-V4.1-Flash engine work in ~/AI/Strata-DS4 (branch deepseek4; push: git push ds4 deepseek4:main
- pushing is OK'd; GitHub = Indras-Mirror/Strata-DSV4.1).  HEAD at handoff: 834728b0 (plus a docs commit).

READ FIRST, in order, and nothing else until a question needs it: docs/ds4/V41_START_HERE.md section 5c blocks
"Session 9d" -> "Session 9c" -> "NEXT DESIGN ITEM" -> "Session 9b" -> "Session 9" (newest first).

STATE (all measured, RTX 4090 + 90 GB RAM, bias ON unless noted):
- DSpark speculative decoding WORKS (one-shot ds41_generate, greedy): Q2_K sidecar + D3 batched verify (default) +
  1-slot verify prefetch (default) -> docs 39.5 / python 31.7 / code 35.6 tok/s vs plain 29.9 / 20.6 / 26.7
  (1.32-1.53x).  Config: STRATA_DSPARK=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf STRATA_DSPARK_DEPTH=4.
  Verify core correct (bias-off vs independent prefill: near-ties only).  PCIe is the wall (nsys).
- Quality baseline (DwarfStar 112 official API fixtures, tools/v41/quality/): exact routing 86.21% API top-1 (short
  group); DEFAULT decode route bias (1.0 / 0.6) = 72.13% (-14 pt) + repetition loops; bias x0.25 = 85.18% at
  12.6 / 7.4 tok/s; x0.5 = 81.66%.  Our GSQ weights are 3.7 pt below DwarfStar's IQ2_XXS-Q2_K on short prompts.
- Smart bias (opt-in env STRATA_RB_PROTECT=P / STRATA_RB_MARGIN=d, ds41_dense.cpp router): PROTECT=3 84.56% but only
  10.6 / 6.6 tok/s (flat x0.25 dominates it); PROTECT=2 83.14%; MARGIN variants NOT yet measured.
- --arena-adapt (adaptive RAM arena, Mal's idea) FIXED (slot-capacity check + stream sync; DS41_CHECK_ARENA: 3600
  swaps, 0 bad; zero routing change): exact routing docs 5.48 -> 6.90, python 3.51 -> 5.61 tok/s (+26-60%).
- ggml upstream fixes #30103/#29953 applied (vendored third_party is gitignored: re-apply tools/v41/ggml/
  upstream-fixes.patch with /usr/bin/patch on a fresh checkout; companion fix in src/prefill/moe_mmq.cu J_best).
- Production serve (tools/v41/serve/ds41-engine.sh, binary build-ds4-gpu, temp 1.0) is UNTOUCHED: all work is in
  build-ds4-dspark.  DSpark is NOT in --serve (needs speculative SAMPLING - design in START_HERE 5c, not started).

FIRST: check ~/AI/v41-dev/hf/publish.log - "CARD PUBLISHED" = the HF Q2_K sidecar upload + model card update
(IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored) finished; otherwise finish them (card staged at
~/AI/v41-dev/hf/model-card-README.md; upload: hf upload <repo> <file> <path>).

NEXT (Mal's call where marked):
1. MARGIN variants on the 34-case subset (~/AI/v41-dev/quality/short34.jsonl, ~7 min each) + 128-token speed on
   docs/python: STRATA_RB_MARGIN=0.1 / 0.3 (score: tools/v41/quality/score_official.py run|score; quality runs use
   STRATA_SERVE_PASS_BIAS=1 so the bias applies).  Compare against the curve; full 102 only for the winner.
2. Re-measure arena-adapt speed after the sync fix; test arena-adapt + bias x0.25 (quality-first speed) and with
   DSpark (D3 uses its FILEG path, not arena-adapt - wire it in if it helps); decide default-on.
3. MAL DECIDES the route-bias level (speed-first default vs quality-first x0.25 / smart variant) - present the table.
4. Then (Mal's go) speculative sampling for DSpark in --serve (opt-in STRATA_SERVE_DSPARK).

TOOLS: ~/AI/v41-dev/bench4/ (prompts prose/python/docs/code .i32, runs.py <tags> = tok/s, tokens/pass, verify ms,
hits, prefetch, rep8 loop score; bench4.sh/q2pf.sh templates); ~/AI/v41-dev/quality/ (cases.jsonl, short.jsonl,
short34.jsonl, q_* results); ~/AI/v41-dev/smartbias/ (run2.sh, arena_verify.sh, logs); tools/v41/vdump_cmp.py +
d3_nb_gate.sh (verify-vs-prefill gate); ~/AI/v41-dev/d2/det_pair.sh (prefill determinism md5).  Engine taps:
DS41_PROF_X (D3 phases), DS41_CHECK_LRU, DS41_CHECK_ARENA, STRATA_DSPARK_DUMP, STRATA_PROFILE_DECODE (+ nsys
--capture-range=cudaProfilerApi), STRATA_DEC_MARGIN + STRATA_FORCE_IDS (forced-token equivalence check).

RULES: one-shot model runs through bash tools/ds4/memguard.sh 86 72 -- ... (it takes ds4-gpu.lock itself - never wrap
it in flock); --serve runs must NOT go through memguard (it backgrounds the command -> stdin /dev/null): launch like
ds41-engine.sh / score_official.py (flock + systemd-run scope, exec'd).  Need >= 75 GB MemAvailable.  Kill by exact
PID only; NEVER pkill/pgrep -f a pattern that also appears in your own command line (it kills your shell - happened
twice).  Keep tests small (34-case subset, 128 tokens) unless deciding.  Bias-on text depends on VRAM residency:
compare verify ms / tokens-per-pass on identical text, filter rep8 > 0.3, timing noise +-5%.  Build:
nice -n19 ninja -C build-ds4-dspark ds41_generate.  anaconda's `patch` binary is corrupt: use /usr/bin/patch.
GPU SHARING: relay peer `voice-duplex` (Quetza, strata-dense 27B voice work) relay-asks and WAITS for an explicit 'go'
before any GPU use (DS4 has priority, Mal).  Claim your relay name with `relay up strata-dspark`; wake on messages
with `relay wait --timeout 14400` as a background task, re-armed after every wake; reply with the FULL ask id.
Relay system notes: memory quetza-relay-unified.md; QuetzaCodetl relay/REPL changes are built but UNCOMMITTED
(main.tsx holds another session's voice work).
```

## UPDATE 2026-10-13 ~20:00 (session 8, conductor) - READ FIRST: D3 batched verify tier (opt-in, BROKEN)

DSpark spec-decode is coded and runs (3.3-3.6 tokens/pass) but is SLOWER than plain (20.5 vs 27.4 tok/s): the verify
pass (`pass_v`) is tier-dominated (~10x the dense half) because it calls `tier.run_to_dev` once PER pass token, so
nothing is shared across the pass.  Mal's bar: verify <= ~100 ms/pass -> 35-40 tok/s (currently 168-212 ms).

**D3 = the batched per-expert-class tier (builds, CRASHES).**  Full detail + leads in `V41_START_HERE.md` §5c
"Session 8".  Short form:
- New `gpu_run_n_ex` (`ds41_moe.cpp`, after `gpu_run_n`): union sharing + one `native_expert_grouped` per
  (9 format class, site); new `Ds41MoeTier::run_multi_to_dev`; `pass_v` calls it once per layer (4 tokens at a time).
  New activation buffer `im.nact_multi_n` + `nact_slot_n()`.
- **OPT-IN: `STRATA_TIER_BATCH=1`** (default = the working per-token path).  Debug: `DS41_DBG_X=1`.
- **Crash:** SIGSEGV in the CPU pool (`native_gu_rows` -> `ggml_vec_dot_q3_K_q8_K`) at layer 0, nt=4, on the prompt
  tail; a gdb run also gave `ds4_moe: c_stage: out of memory` + SIGABRT - likely one host OOB.  `[dbg_x] ENTER
  gpu_run_n_ex` prints, the `nj`/per-job dump does NOT -> a bad `nact` pointer between ENTER and the CPU job loop.
  **First checks:** `im.nact_multi_n` sized + non-empty; `nact_slot_n`'s index (`t*nfmt + nact_act_idx`) vs its
  stride (`kNativeActBytes`); the PCIe `si` bound.
- A/B harness `~/AI/v41-dev/d2/d3_test.sh` (bias off, depth 4, -n 64): `d3_a` (batched) exit 139, `tok_b`
  (per-token) exit 0 / 64 clean tokens.  Correctness target `d3_a.tok == tok_b.tok`; perf target verify <= ~100 ms.
  Update the script's env (`STRATA_NO_TIER_BATCH` -> `STRATA_TIER_BATCH`).

**Correctness bisect DONE** (prior session's `d2_queue.sh`): both arms diverge from plain (`bis_d0` token 8,
`bis_noacc` token 11) -> the divergence is in `pass_v`'s core at m=1; drafting/rollback/rejection-residue cleared.

**MTP (Mal asked):** explored (FINDINGS §13-22).  MTP's draft is cheap (6-9 ms; 6.1 ms `--mtp-keep 2`) but it LOSES
(18.9-20.6 vs 24.9 plain) because the multi-token verify pass is the wall - the SAME wall as DSpark.  So D3 is the
lever for both MTP and DSpark.

**GPU sharing:** live relay peer `strata-dense` (Strata dense 27B, wants the 4090 for short MTP benches); this
session's relay name is `strata-dspark`.  Runs do not fit together -> SERIALIZE (`relay ask strata-dense`, yield,
broadcast `4090 free` when done).


**Last updated:** 2026-10-13 ~03:20 by Claude (session 4, conductor). Decode bottleneck MEASURED (dense Q8_0 weight bandwidth); the earlier GPU-idle attribution was a TOOLING BUG; slice C / C' / spec-reads refuted; HF upload complete.
**READ §UPDATE 2026-10-13 ~05:10 FIRST, then §UPDATE ~03:20, then §UPDATE 2026-10-12 ~20:45, then §UPDATE ~19:50, then §HANDOFF ~19:30**, then §UPDATE ~10:15, then ~08:00, then MORNING, then NIGHT-2, then 2026-10-11 NIGHT / EVENING / 13:10.

## UPDATE 2026-10-13 ~05:10 (session 5, conductor, ONE flash worker max - Mal) - READ FIRST

Mal's goal now: **DSpark speculative decode + vision, target 50 tok/s.**  Status: neither vision nor DSpark runs in the
engine yet (sidecar weights only); decode is plain one-token 26.4-27.5 tok/s (forced) / 28-29.5 (luckier token paths).
- **DSpark decisive teacher test RUNNING** (`~/AI/v41-dev/dspark/teacher.sh` -> `teacher.log`, then `followup.sh`):
  FP8 teacher states for the same 639 tokens in `/media/Crucial1TB/v41-teacher`, then `dspark_accept.py --teacher`
  (-> `accept_teacher.log`), then teacher layer 39 + `v41_calib.py head --save-argmax` and both state sources re-scored
  with the new `--teacher-top1` report (-> `accept_exact_t1.log`, `accept_teacher_t1.log`).  Those two tool flags are
  new and uncommitted until that run proves them.  50 tok/s = 20 ms/token: needs ~2+ accepted drafts per verify pass
  (measured so far 0.73 biased / 0.41 exact = ceiling ~30 tok/s).
- **Vision slice 1 in flight** (worker `v41-vision-enc`, packet `~/.quetza-data/conductor/packets/v41-vision-enc-task.md`):
  a torch encoder process (`tools/v41/vision/`) that is a drop-in for Strata server's `strata-vision` protocol, sidecar
  weights, gated vs model.py on the original weights.  Next: engine GENI + image-token rules (bias_vl, Engram mask),
  then the serve shim (which hard-codes `~/AI/Strata` - make it portable for outside users).
- Corrections to the 03:20 notes (details in `V41_DECODE_OVERLAP_PLAN.md` §6 / §8): the `spec4` result means the NVMe
  wall IS on the critical path (the implementation's overhead ate its gain); the nrows=1024 mmvq is `attn_output_a`
  (dense mat-vecs at ~84% of peak bandwidth).
- HF repo housekeeping done (card 4-step quick start, Engram placeholders `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt`).

## UPDATE 2026-10-13 ~03:20 (session 4, conductor; handoff 20261012-220915-805454) - READ FIRST: decode is BANDWIDTH-BOUND on the dense Q8_0 weights, and the old attribution was a TOOLING BUG

**The number that matters.** Per layer ~393 us of *device* time = **223 us (57%) in `mul_mat_vec_q<Q8_0>`** (the
dense projections), ~86 us in a 49-name elementwise zoo (**33% of the token**), 46 us other quant matvecs, 24 us
`rms_norm` x4, and **13.2 us (3.4%) in `flash_attn_ext_f16`** - the actual attention. Per token the dense Q8_0
matvecs move **6.17 GB at ~690 GB/s = 68% of the 4090's 1008 GB/s peak = 8.94 ms/token**. **Decode is
bandwidth-bound on dense weight BYTES** - not host-bound, not KV-bound, not attention-bound.

**TOOLING BUG (now fixed).** `calls.py` / `idle_attrib.py` unioned only KERNEL+MEMCPY+MEMSET. The dense half runs as
**CUDA graphs** (ggml-cuda capture) whose activities live ONLY in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, and with the
default `--cuda-graph-trace=graph` the graph *nodes* never enter the KERNEL table. Verified in `dec7.sqlite`:
`KERNEL` 36406 rows, **0 with a non-NULL `graphId`**; `GRAPH_TRACE` 7742 rows / 1563.16 ms in the same 4.5 s window.
So every earlier reading of this engine was wrong: **GPU busy 18.42 -> 27.66 ms/token; idle 60.7% -> 41.0%**, and
the old "117.7 of 121.7 ggml syncs have nothing to wait for / the 17.83 ms is mostly host" is REFUTED (with graphs
included: 81.9 of 121.7 syncs find device work in flight, median post-drain lag 2.7 us). Both scripts now sweep
GRAPH_TRACE and report graph launches/token; `--kernels-only` reproduces the old numbers byte-for-byte.
**Treat §1/§1b/§2 of `V41_DECODE_OVERLAP_PLAN.md` as SUSPECT; its §7/§8 carry the corrections.**

**REFUTED - do not re-attempt:**
1. **Slice C** (move expert selection on-device): the k-ordered `decode_rows_sum` (`iq_kernels.hpp:72-74`,
   `ds41_moe.cpp:2378`) needs ALL k rows and feeds `attn_router(l+1)`, so the host must hold the ids AND finish the
   miss service every layer. No cross-layer pipelining.
2. **C'** (device-servable file tier): the 6.0 ms `file_ms` IS the NVMe transfer (25 MB/token at the measured
   4.65 GB/s); GPUDirect Storage is not operational (`nvidia-fs` module absent, and the gguf is on `ntfs3`, which
   libcufile does not accept), and resident is capacity-impossible (83 GB file tier vs 13.5 GiB free VRAM).
3. **`STRATA_SPEC_TOP=4`** cut 29% of the file wall for **0.0 tok/s** - the file wall is NOT the critical path.

**§3 slice B - CODED, flag `STRATA_TIER_GRAPH`, DEFAULT OFF.** Fixed per-class metadata layout + one CUDA graph per
(site, class) site. Verified **bit-identical** (forced tokens 0-diff on the rebuilt binary) and **neutral**
(26.90 off vs 26.40/26.55 on). It *does* cut `cudaLaunchKernel` 239.4 -> 52.3/token - and gains nothing, because the
device is busy under those launches. Host-CPU hygiene only: keep default-OFF or revert.

**THE LEVERS - only two have money:**
- **A) requant the dense projections Q8_0 -> Q4_0/Q4_K** - the node profile ESTIMATES −4.0 to −4.5 ms/token, **but
  the tree already contradicts it**: `STRATA_DENSE_REQUANT` was measured in `~/AI/v41-dev/dq/results.txt` - q6_K
  (7.59 -> 5.86 GB) and q5_K (7.59 -> 4.91 GB) both hold decode at **~29 tok/s** (q5_K 29.17/28.85 vs the q8 baseline
  28.3-29.5), and the ~19:30 handoff already concluded "frees VRAM, ~+100 experts, does NOT speed decode". Cause: **ggml's
  K-quant `mul_mat_vec` gets lower effective bandwidth per byte than Q8_0's**, so fewer bytes != less time. Treat A as a
  **VRAM-recovery** change, not a speed lever; a real test is a kernel achieved-GB/s comparison, not a byte argument.
- B) fuse the elementwise zoo (5.45 ms/token, 33%, 3288 ops/token, mean < 5 us): −1 to −3 ms, MEDIUM risk (must stay
  bit-identical). **This removes WORK rather than bytes, so it is the lever most likely to convert - prefer it.**
- Every attention-level lever (KV quant / smaller window / MLA absorb / split-K) shares a **~0.7 ms** ceiling - skip.
- Residual: an unmatched `nrows=1024` mmvq bucket (1/layer, 1.60 ms/token, 18% of the mmvq).

**HF UPLOAD: COMPLETE** - 16/16 parts + `V41-MTP-Vision.gguf` + tokenizer/small files, "ALL DONE" 2026-10-13
03:02:50, on `IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored`. The xet `upload_xorb` retry deadlock recurred once
(part-12 stalled 64 min); fixed exactly as documented by killing the uploader **BY PID** (the until-loop respawns it
in 60 s); parts 13-16 then completed.

### RESUME PROMPT (paste this)

> Maximize DeepSeek-V4.1-Flash **decode** tok/s on one RTX 4090; repo `~/AI/Strata-DS4`, branch `deepseek4`, HEAD
> `5e5f4c16`. Decode is 26.4-27.5 tok/s = 37.5 ms/token. **The bottleneck is MEASURED: the dense half is
> bandwidth-bound on Q8_0 weight bytes - `mul_mat_vec_q<Q8_0>` 8.94 ms/token = 57% of a layer at 68% of the 4090's
> peak; attention is only 3.4%.** Read `docs/ds4/V41_DECODE_OVERLAP_PLAN.md` §7 + §8 FIRST (they correct §1/§1b/§2,
> which were computed with a tool set that never swept `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`), then
> `~/.quetza-data/conductor/reports/attn-node-profile.md` and `attn-router-decomp.md`.
> **Dead ends - do not retry:** slice C (on-device selection), C' (device-servable file tier), removing ggml-stream
> syncs, slice B's tier graph capture - all bit-identical and/or neutral. **The open decision is Mal's:** lever A
> (requant dense Q8_0 -> Q4_0/Q4_K, ~−4.0/−4.5 ms/token, a quality trade) or lever B (fuse the elementwise zoo,
> −1..−3 ms, bit-identity risk). Do not start either without his call.
> Measurement rules that still hold: forced-token identity
> (`STRATA_FORCE_IDS=~/AI/v41-dev/dec7/force256.tok`, diff = 0 differing lines, filter the memguard banner line) is
> the correctness bar; measure with nsys **call counts + MAX CONCURRENT ACTIVITIES including GRAPH_TRACE**, never
> tok/s (the noise floor lies); A/B with `~/AI/v41-dev/k143/ab.sh TAG [ENV=VAL ...]`; GPU runs via
> `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- <bin>` after `MemAvailable >= 75 GiB`; one GPU job at a
> time; kill by PID only, never `tmux kill-server`.
> Uncommitted work in the tree: `V41_DECODE_OVERLAP_PLAN.md` §5-§8 (+134) and `ds41_moe.cpp` +156/-21 (slice B,
> default OFF, fully gated).


## UPDATE 2026-10-12 ~20:45 (session 3) - READ FIRST: the forced A/B verdicts, and the S4 sync removal

**The forced-token A/B is the load-bearing tool now: every kernel number below was re-run with
`STRATA_FORCE_IDS=<file>` (both variants decode the SAME 256 tokens), because greedy text divergence
confounded the earlier numbers.**  Read §19:50's "+11% quarter-block" as RETRACTED.

- **Quarter-block dot, FORCED A/B (dec3):** whole 26.92 / 26.68 tok/s (tier 16.9 / 16.7 ms), quarter
  26.81 / 26.74 (tier 16.8 / 16.9), old path 26.15 / 26.03 (tier 17.0 / 17.2); **all 6 runs produced the
  same 256 tokens** (0 lines differ from `k143/whole_1.tok`).  So the quarter dot is **~0.4 ms of the tier
  (~1%), not 11%** - the unforced 29.6-30.3 runs came from a token path with ~45% fewer misses.
  `STRATA_EXP_MODE_T143=2 / T11=2`: no gain.  **Misses drive decode speed.**
- **Rank argsort + warp top-k (ggml patch):** unit test 140/140 ok; **warp top-k k=6 2.99 us vs bitonic
  8.70 us** (rank kernel alone 6.38).  Forced decode A/B: within noise.  Kept (it is a clean win on the
  kernel's own cost, just not visible at 80 sorts/token in a 37.5 ms token).
- **x-skip: bit-identical on prefill** - `pf_xhost_rank0 [STRATA_X_HOST=1]` and `pf_rank0_b` gave the SAME
  8K md5 `5c2b0e1ed697`.  Decode: within noise.  Costs nothing, so keep.
- **Prefetch dummy-DMA skip (`STRATA_PF_NO_DUMMY=1`):** tokens identical, no measurable gain (the DMA is on
  the idle-ish copy engine while the host waits anyway).  Prefetch counters now print: ~8065 issued /
  708 outside the arena / 2499 used per 256 tokens.
- **PREFILL md5 REPRODUCIBILITY: NOT FIXED by the MoE-only deterministic MMQ patch (seq5).**  8K `--dump-last`
  is now `9b20b5335929` (cap 1 GiB), `b1ad8e47781f` (64), `3de2ed258753` (cap 1 GiB again!), `3084d9c87087`
  (4 GiB) - **four different md5s, and the two identical-cap runs differ too.**  The `chunk prestage` region
  size varies run to run (1.30 / 1.05 / 1.59 / 1.38 GiB) = free VRAM at load varies -> the expert *grouping*
  inside the prefill chunk varies.  So the residual nondeterminism is NOT in the MoE MMQ launcher (that one
  is now tile-only) - look next at the DENSE MMQ path (`mmq.cu`, `mul_mat_q_case` stream-k by free VRAM) and
  at `src/prefill/moe_mmq.cu` (the engine's own MMQ context - it may not go through ggml's launcher at all).
  **Consequence: there is no stable prefill exactness reference right now.  Decode IS deterministic**
  (per-expert dot + `decode_rows_sum` in k order), which is why the forced A/B is trustworthy.
- **S4 layer sync REMOVED on the decode (run_to_dev) path** (`ds41_moe.cpp gpu_run`): the per-layer
  `cudaStreamSynchronize` there was a *second* host round trip on the same stream - the `sum sync` at the
  end of the same call already waits for the identical work.  Kept on the host-readback path (there it IS
  the wait) and under `DS4_CHECK_LRU` (it compares slots against the file right there) and
  `STRATA_LAYER_SYNC=1` (old behaviour).  hit_ms/pcie_ms (diagnostics, never printed) are now only read when
  that sync ran, because `cudaEventElapsedTime` on an incomplete event returns `cudaErrorNotReady`.
  **STATUS: measured by dec6/seq6 (build gen.v6, gate, forced A/B off vs on x2, tier profile, then the
  predictive-file-read instruments).  Reported saving into the record only after seq6 lands.**
  Tier profile said 0.123 ms/call x ~40 = ~4.9 ms/token was going into that bucket.
- **Host-sync budget (graph-level nsys `nsys/dec3.sqlite` + `DS4_TIER_PROF`, 3800 layer calls, 0.510 ms/call):**
  `cpu pool 0.216 · gpu launch 0.132 · layer sync 0.123 · activation 0.021 · sum 0.018`.  Decode under the
  profiler 21.2 tok/s (slowed); live per-token: `begin 0.9 predict 0.25 attn_router 17.2 tier 16.5 finish 2.0`.
  **GPU busy ~20 ms of 37.5 ms** -> the host is on the critical path; that is the decode target.
- **HOST-CALL ATTRIBUTION (dec3.sqlite, graph-level, 4.5 s decode window, 96 tokens) - the decisive decode numbers:**
  GPU busy union 18.27 ms/token, **idle 28.60 ms/token (61%)**.  161,295 host runtime calls = **1680/token**, of
  which `cudaLaunchKernel 417.7` + `cudaMemcpyAsync 387.1` + `cudaStreamSynchronize 257.7` + `cudaGraphLaunch 80.7`
  (2.0/layer - attn_router + finish_predict).  Host time: `cudaStreamSynchronize 26.21 ms/token` (101.7 us each!),
  graphLaunch 3.46, memcpyAsync 3.03, launchKernel 2.12.  GPU gaps>5us: 357/token, **28.13 ms/token**, owned by
  `cudaStreamSynchronize 11.28` + **14.68 ms with no single owner** (= the host doing ~884 short calls; 14.68/884
  ~= 16 us of idle per call).  **By stream:** `stream 14 = gp.s, the tier` (322 kernels/tok: swiglu_q8_1_entries
  81.0 = 2/layer, native_gu 53.6, native_down_multi 44.5, quantize_q8_1 39.6, decode_rows_sum 39.6 = 1/layer) with
  89.2 syncs/token = 5.03 ms; `stream 16 = ggml's dense stream` (k_bin_bcast 8.5, mul_mat_vec_q 5.2, rms_norm 4.7)
  with **121.7 syncs/token = 17.02 ms/token** (3/layer, 139.9 us each); stream 7 = the one-shot load sync (383 ms).
  So the two attack surfaces, measured: **(i) ~14.7 ms/token of GPU idle from ~800 tier host calls/token** -
  `launch_group` does 1 `cudaMemcpyAsync` + 1 kernel launch PER FORMAT CLASS (9 classes, 3 groups: hits/pcie/file)
  = ~20 calls/layer, and the metadata pointers are already passed into device memory, so a fixed-slot metadata
  layout + one graph-captured launch set per layer is the fix (884 -> ~80 calls/token);
  **(ii) 17.0 ms/token of host blocking on ggml's stream**, 3 syncs/layer from `ggml_backend_synchronize`
  (ds41_dense.cpp:1266/1326/1459/1471) for the attn_router + finish_predict readbacks (the finish_predict one
  exists only to feed prefetch - `pf_b=0` should drop it).
- **S4 REMOVAL: NOT DEMONSTRATED - and the first read of it was WRONG.**  I first called this a ~5 ms/token win
  off `ls_off` 25.50/25.74 (tier 18.36/18.05) vs `ls_on_1` 21.96 (tier 23.06).  Then `ls_on_2` came in at
  **26.35 tok/s, tier 17.63 - the FASTEST run of the four.**  Full set (identical tokens, identical work: 59511
  VRAM hits / 255 cpu in every run):
  `off 25.50 (tier 18.36) / on 21.96 (23.06) / off 25.74 (18.05) / on 26.35 (17.63)`.
  `ls_on_1` was globally slow in EVERY phase (begin 1.52, predict 0.39, attn_router 18.43, finish 2.27) = machine
  noise, not the flag.  **Verdict: the per-layer sync is subsumed by the sum sync that follows it - the host waits
  for the same GPU work either way, so removing it is harmless (tokens identical, gate PASS worst rel 1.440e-02)
  but unproven as a win.  Kept for now; revert freely.**
  **And the tier profile now PROVES the mechanism (dec6 tp_on/tp_off, 10200 layer calls each):** with the sync,
  `layer sync 0.160 + sum 0.011 = 0.171 ms/call`; without it, `layer sync 0.000 + sum 0.169 = 0.169 ms/call`.
  The wait is not removed, it MOVES into the sum sync that follows - total unchanged (0.171 vs 0.169), which is
  what the tok/s A/B could not resolve.  So: S4 is a no-op for speed.  The lesson generalises: **a sync only
  costs time if it is not subsumed by a later wait in the same layer** - consolidate syncs, don't just move them.
- **METHODOLOGY - the noise floor is now the problem, and tok/s cannot resolve these slices.**  With IDENTICAL
  work the tier phase varied 17.63 - 23.06 ms (5.4 ms, 30%) between runs while the box was busy (HF upload +
  local analysis).  The earlier dec3 A/B had tier 16.5-17.2 (0.5 ms) on a quiet box.  So: (a) for launch-overhead
  work the decisive metric is the **host call/sync COUNT from nsys** (each `cudaLaunchKernel`/`cudaMemcpyAsync` is
  ~5-8 us of host, so cutting 280 calls/token = ~2.2 ms, below the loud-box noise but plain in the counts), not
  tok/s; (b) run A/Bs on a quiet box or with many reps; (c) never let the CPU-heavy local work run beside a timed
  decode run.
- **SCAN VERDICT (`reports/decode-overlap-scan.md`, and it corroborates the attribution above independently:
  GPU busy 18.10 vs 18.27, sync-gap owner 11.28 same, no-single-owner 14.68 same).  THREE THINGS IT ADDS:**
  (i) **max concurrent kernels = 1** across the whole decode window - streams 14 and 16 NEVER overlap, every layer
  is serialised; (ii) **328 MB/token of expert H2D copies run strictly between the tier's syncs**, not overlapped
  with compute (stream 15: 28.5 copies, 271 MB); (iii) **12.21 ms/token is a pure pipeline bubble** - neither the
  GPU nor the host is doing work (host blocked in a serialising call, device queue empty), and only 47.6 syncs/token
  exceed 100 us yet those account for 22.86 ms/token.  **Its ranking (bigger ceiling than the call-count work):**
  1. **Overlap the per-layer expert H2D with compute - ceiling ~11-14 ms/token, output EXACT** (the weights are a
     copy): double-buffer the selected experts' device staging (copy layer l+1's experts while layer l's kernels
     run, on a copy stream with EVENTS instead of the stream sync); the chunk prefetch/prestage machinery is the
     thing to build on.  **This is now the top slice.**
  2. Collapse the 258 syncs/token (the 47.6 long ones are the whole stall) - and with S1/S4 gone make the two
     streams actually concurrent instead of serialised.
  3. Attack the 12.21 ms bubble: it is neither copy nor compute, so it is HOST DECISION latency - the tier's
     `skip_miss` / PCIe-budget / resident host loops between graph launches - and moving those policy loops
     on-device removes the gate.
  4. Kernel micro-opt is NOT the lever (argsort's 0.72 ms/token is ~2% of the token); overlap is 10-20x larger.
- **ROOT CAUSE / STRATEGIC (why everything is serial, and what would actually fix it):** the host must have the
  router's top-k ids BEFORE it can launch the tier (that is S1, `attn_router`'s one sync/layer - the last
  unavoidable host wait, and the reason max concurrency is 1).  Everything downstream (which tier each expert is
  in, the PCIe budget, the arena slot -> device pointer) is decided by HOST loops over host-side tables
  (`slot_of`, the cache, `skip_miss`), which is also the scan's 12.21 ms bubble.  So the structural fix is to
  **move the expert selection + descriptor build on-device**: a small kernel that reads the router ids from
  `im.o_ids`, looks up a device-side (tier, device-pointer) table, and writes the metadata block for
  `native_expert_grouped` in place.  Then the host never needs the ids, every layer's work is merely ENQUEUED on
  the stream, the host runs 40 layers ahead, the H2D copies overlap the compute by construction, and S1 + the
  bubble both disappear.  The metadata packing + fixed slots (slices 2 below) are the prerequisites for that,
  which is why they come first.
- **SLICE QUEUE (decode, in order; each is env-gated so it can be A/B'd in one binary):**
  1. **Overlap the expert H2D with compute** (scan's #1; ceiling 11-14 ms/token; EXACT) - see above.
  2. **tier metadata packing - CODED, awaiting seq7.**  `launch_group` packed per format class (1 `cudaMemcpyAsync`
     + 1 launch per class, 9 classes x 3 groups = ~20 host calls/layer); now every class of a group is packed into
     the h_meta block first and copied up in ONE `cudaMemcpyAsync` (STRATA_GRP_ONE_COPY=0 = old).  Bit-identical by
     construction (same kernels/bytes/order).  Expect ~7 fewer host calls/layer of the measured ~884/token.
     MEASURE IT BY CALL COUNT IN NSYS, not tok/s (see the methodology note above - a ~2 ms host effect hides under
     a 5 ms noise floor).
  2. **CUDA-graph the tier's class launches** (the big one): `native_expert_grouped` already receives ptr/st/ng/
     dst/tok as DEVICE pointers into the metadata block, so a graph's node params are FIXED - only the metadata
     contents change per token.  Pack into fixed per-class slots, launch all 9 classes every layer (n=0 for
     absent ones), capture once -> per layer: 1 memcpy + 1 `cudaGraphLaunch` instead of ~20 calls (884 -> ~80
     calls/token).  Needs the kernel to accept ng=0.
  3. **Defer the `finish_predict` readback** (the third ggml sync, 139.9 us each): it exists only to give
     `tier.prefetch` the next layer's top-16, and it uses a blocking `ggml_backend_tensor_get`
     (ds41_dense.cpp:1391).  `finish(l)` and `attn_router(l+1)` are on the SAME stream, so no host sync is
     needed between them - issue the readback async, run `attn_router(l+1)`, then sync ONCE for both.  Trade-off
     to measure: the prefetch would then overlap tier(l+1) instead of attn(l+1).
  4. `pf_b=0` (plain `finish()`, no predict graph, no readback) - the cheap control for (3): the resume doc says
     speculative prefetch measured NET NEGATIVE (finish graph +1.5 ms, issuing +1.3 ms vs -1.8 ms tier).
- **SEQ6/SEQ7 VERDICTS (2026-10-12 ~21:05) - this stretch produced NO SPEED WIN; it closed three questions:**
  - **Predictor recall curve MEASURED (the scout's open question #1, new evidence - no paper publishes 2-3-layer
    DeepSeek numbers).**  `STRATA_FILE_PRED=1`, 672 file-tier routes per 256 tokens:
    k=1 top4 50.2% / top8 59.7% / **top16 66.9%**; k=2 44.7 / 53.1 / **62.4**; k=3 40.0 / 46.8 / **54.1**;
    candidates k=1 475/626/853, k=2 437/584/795, k=3 409/524/715, k=4 361/498/671.
    So one layer ahead catches 2/3 of file misses in the top-16, and a 3-layer lead still gets 54% - **a 2-3
    layer lead IS viable**, the blocker is purely the issuing cost.
  - **Speculative file reads: DEAD AGAIN (third measurement).**  `STRATA_SPEC_TOP=4`: 26.48 tok/s, 589 reads issued
    / 383 used (65%), file 1098 ms; `=8`: 25.21 tok/s, 781 issued / 446 used (57%), file 1180 ms.  Against
    no-spec 26.73 / 26.64.  More depth = more overhead, no catch payoff.  Do not retry without first fixing the
    issuing path (dedicated I/O pool, not the decode thread).
  - **Tier metadata packing: CORRECT but SPEED-NEUTRAL, and that is the finding.**  `g1` (one copy/group)
    27.30 / 26.16 (tier 16.22 / 16.84) vs `g1n` (`STRATA_GRP_ONE_COPY=0`, per class) 26.71 / 26.56; tier profile
    0.410 vs 0.395 ms/call.  All runs bit-identical tokens.  **So the ~10 metadata copies a layer are NOT the
    bottleneck** - the theoretical ~2 ms/token is unmeasurable, and the "gpu launch" bucket is dominated by the
    grouping loops + kernel launches, not the copies.  Keep it (bit-identical, ~280 fewer host calls/token, and
    it is the fixed-layout prerequisite for graph-capturing the launches) but do NOT count it as a win.
  - **Therefore: the levers are the OVERLAP (slice A) and the ON-DEVICE SELECTION (slice C), not call-count
    trimming.**  See `docs/ds4/V41_DECODE_OVERLAP_PLAN.md` - it now carries the full measured case, the three
    slices with exact code sites, and the measurement rules (forced-token identity as the correctness bar; nsys
    call counts + max-concurrent-kernels as the speed metric; never time a run with CPU-heavy work beside it).
- **HANDOFF:** if this session hits its token limit the successor reads this section first; the running queues are
  `~AI/v41-dev/dec{6,7}/seq{6,7}.sh` -> `results.txt` + `k143/results.txt` (g_/h_/ls_/tp_/g1_ prefixes), the HF
  upload loop is `~/AI/v41-dev/hf/loop.pid` (kill the `python3 hf_upload.py` BY PID if a part stalls >45 min - it
  wedged once at ~20:29 on an xet `cas::upload_xorb` retry and had to be restarted), and the hourly chronos wake
  `cw10122040-4729` watches both.
- **Scout reports (read-only workers):** `~/.quetza-data/conductor/reports/moe-miss-research.md` (36 KB - miss
  arithmetic: ~10.2 ms of the 37.5 ms token is miss latency, zero-miss ~+36%; ranked top-5: free-to-issue +
  deeper speculative reads, utility/frequency admission vs pure LRU, miss-only low-precision tier,
  residency-conditional routing bias (+10-15%, ppl-gated), Fate percentile widening / HOBBIT stacking) and
  `decode-overlap-scan.md` (upstream GitHub scan, in flight).
- **HF upload:** 16 x 10.7 GB parts, 3 done at ~20:00, ~7-8 MB/s (part 1 spiked to 127 MB/s), loop alive
  (`~/AI/v41-dev/hf/loop.pid`, `upload.log`).

## UPDATE 2026-10-12 ~19:50 (session 2 after the handoff) - READ FIRST, then HANDOFF ~19:30 below

- **Dense requant: DROPPED.**  q5_K decode 29.17 / 28.85 tok/s = q8 (28.31 / 29.46) although the dense bytes fell 7.59 ->
  4.91 GB; q5_K costs code ppl 2.3790 -> 2.4150 (+1.5%), agent 14.47 -> 15.61 (+7.9%, KL 0.35), reason 6.60 -> 6.74.
  q6_K: same speed as q8, small loss.  q4_K not run (stopped).  Decode is not dense-bandwidth-bound.
- **Ternary quarter-block dot (-DSTRATA_PTQ1_QUARTER) gate PASS** (test_ds41_moe_gpu worst rel 1.440e-02 = whole).
  Unforced A/B: quarter 29.58 tok/s vs whole 26.61 (tier 13.0 vs 17.0 ms) BUT the texts diverge at ~token 6 (float
  sum order + greedy near-ties) and the quarter run had fewer misses (98.5% vs 97.3% VRAM hits) -> CONFOUNDED.
  Added **STRATA_FORCE_IDS=<file>** (ds41_generate one-shot decode: decode a fixed token list, e.g. a previous run's
  stdout) so variants do the same work.  Forced A/B queue: ~/AI/v41-dev/dec2/seq2.sh -> dec2/results.txt + k143/results.txt (f_*).
- **Rank argsort (ggml-cuda argsort.cu, rows <= 1024; GGML_CUDA_RANK_ARGSORT=0 = bitonic):** the router top-6 + the
  prefetch top-16 = ~80 bitonic sorts of 384 a token at 9 us.  Unit test ~/AI/v41-dev/argsort/test_rank_argsort.cu
  (vs bitonic + std::stable_sort).  Ties now come out in ascending column order (bitonic: network order) -> the 8K md5
  can move only through exact float ties.  Ships as tools/v41/ggml/rank-argsort.patch (third_party is gitignored).
- **x-skip (ds41_moe.cpp gpu_run):** the activation D2H + stream sync + H2D only when the CPU pool runs (nk > 0) or a
  diagnostic needs h_x; else quantize x_dev in place (bit-identical).  STRATA_X_HOST=1 = old path.
- **Decode time budget (nsys dec.sqlite, 96 tokens):** GPU kernels busy ~20 ms/token (union); expert H2D 314 MB/token
  (27.6 prefetch copies on stream 15 + 5.3 demand on stream 14, ~9.5 MB each); GPU-idle gaps 32.6 ms/token under
  nsys (host in cudaGraphLaunch 10.9 - node-level tracing inflates it, re-measure with ~/AI/v41-dev/nsys/run2.sh
  --cuda-graph-trace=graph; syncs 4.1; blocking memcpy 2.6; no CUDA call ~11).  CUDA graphs ARE used (352k of 404k
  kernels).  The 87 graph instantiates are warm-up (clustered), not per token.
- **Forced A/B (first attempt) was NOT forced** (the .tok files start with a memguard banner line; the reader stopped
  at it) - fixed: non-id lines skipped, log prints "STRATA_FORCE_IDS: N forced tokens".  What it still showed, at
  similar miss counts (~1400 pcie / ~640 file per 256): quarter 26.73 / 26.65 vs whole 26.92 / 26.68 tok/s, tier
  16.3-16.5 vs 16.7-16.9 ms -> **the quarter dot is worth ~0.4 ms of tier (~1%), not 11%**; quarter_1/2's 29.6-30.3
  came from a token path with far fewer misses (802 pcie / 481 file).  **Misses drive decode speed** (file reads cost
  ~2.7 ms each, 1.5-1.8 s per 256 tokens).  STRATA_EXP_MODE_T143=2 / T11=2: no gain.
- **Prefill exactness (8K --dump-last):** reference ee31507a0e88 is NOT reproducible any more: uncapped m_tc
  (STRATA_MMQ_TC_GIB=64) gives f4bc02e65a97 - the transcode scratch took ALL free VRAM, so how the experts were split
  into MMQ launches (= the sum order) depended on free VRAM.  The 1 GiB cap makes it environment-independent
  (bitonic: 5694d4c88b26) but costs ~1.1 s of the 8K experts half (241.7 vs 250.5 tok/s) and does NOT return the dense
  half (12.0-12.1 s either way).  Testing 2 / 4 GiB caps (dec4).  Rank argsort moves bits through exact ties
  (0371cfb1d96b), the quarter dot through sum order (4e64b88329d3) -> quality check (ppl + KL vs dq/*_q8.logits)
  before any becomes default.
- **Prefetch dummies:** prefetch() DMAs a 9.5 MB ZERO blob for each predicted expert outside the arena ("the replay's
  rule": timing faithfulness to moe_replay) - wasted H2D time that shares the copy engine with demand misses.
  STRATA_PF_NO_DUMMY=1 skips the DMA (slot stays -1: tokens unchanged).  Decode summary now prints "prefetch N issued /
  D outside the arena / U used".
- **Warp top-k:** ggml_argsort_top_k records k in op_params[1]; CUDA k <= 32 & ncols <= 1024 -> one warp per row
  selects the first k (same order as the rank kernel / stable sort).  Rank kernel alone: 6.43 vs 8.74 us (only -26%,
  one block does all 384^2 compares).  Patch: tools/v41/ggml/rank-argsort.patch (ggml.c + ggml-cuda/argsort.cu,
  relative to the fq-patched tree; apply after dsv41-fake-quant.patch).  NOTE ~/anaconda3/bin/patch is broken (RAM
  damage) - use /usr/bin/patch.
- Queues: ~/AI/v41-dev/dec3/seq3.sh (argsort+topk unit test, builds gen.v3 / gen.v3q, 8K md5, forced A/B g_*, nsys
  graph-level dec3 with DS4_TIER_PROF) -> ~/AI/v41-dev/dec4/seq4.sh (x-skip prefill check, TC caps 2/4, PF_NO_DUMMY A/B h_*).
- Scout report (read-only, flash worker): ~/.quetza-data/conductor/reports/dec-scout.md - argsort sites, router-twice
  (not removable: different layer + input; a Q8_0/F16 copy for the PREDICTION only is exact), F16 indexer/compressor
  -> Q8_0 (V4 --f16-q8 precedent, lossy), sync table S1-S9.

## HANDOFF 2026-10-12 ~19:30 - READ THIS FIRST (supersedes everything below where they differ)

Session ended for a fresh-context restart.  All code is committed + pushed (repo Indras-Mirror/Strata-DSV4.1, branch
deepseek4 -> `git push ds4 deepseek4:main`).  Below: state, numbers, in-flight jobs, the exact next steps.

### Current numbers (RTX 4090 24 GB, Ryzen 7 5700X 8C/16T, 90 GB DDR4, NVMe)
- Prefill 8K: **244-248 tok/s** (MMQ default; was 219-220).  Exactness ref (default path): 8K `--dump-last` md5
  **ee31507a0e88** (`--no-chunk-mmq`: 4d229102872a).  4096 code ppl 2.3790 (MMVQ 2.3799).
- Decode p256 x 256 (fq patch applied in third_party): **28.3-29.5 tok/s** (dq q8 baseline runs), 26.6-27.4 earlier.
- Per decode token (nsys, decode-only): dense Q8_0 mat-vec 8.84 ms (bandwidth floor); expert kernels ~7 ms
  (native_gu<143> 2.72, down<143> 1.31, gu_multi<42> 0.93, down_multi<42> 0.75, Q3_K ~1.1); argsort 0.72; F32
  router+predict 0.77; F16 indexer/compressor 0.7; flash-attn 0.69; NVMe reads ~5.7 ms; ~270 host syncs/token.

### Done this session (commits 5f1a768c .. this one)
- HF release **public**: https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored - card =
  tools/v41/hf/README.md (full GSQ process from the real data: 42.8% ternary / 39.2% Q2_0 / 18.0% Q3_K = 2.250 bpw;
  per-layer held-out error from /media/Crucial1TB/v41-quant-mixed/quant.log), reassemble.sh (tested), Engram
  placeholders, base_model_relation quantized.  Re-upload the card after edits:
  `python3 -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='tools/v41/hf/README.md', path_in_repo='README.md', repo_id='IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored')"`
- Prefill: PTQ1_0 -> Q2_0 lossless device transcode (tools/v41/engine/ds41_transcode.cu, test_ds41_transcode ALL
  PASS) + MMQ for every class (mmq_plan in ds41_moe.cpp), default ON (`--no-chunk-mmq` = old).  +12% 8K.
- STRATA_DENSE_REQUANT=<q6_K|q5_K|q4_K> load-time dense requant (file untouched; ds41_dense.cpp).  Works: 321 Q8_0
  matrices, q6_K 7.59 -> 5.86 GB.
- DSpark offline harness (tools/v41/dspark_accept.py; `--teacher WORKDIR` mode added, UNRUN), STRATA_DUMP_MAIN dump
  fixed (states ENTERING layers 37-39), v41_calib.py `prep-ids` (teacher forward over given token files), logits_kl.py.
- Upload hardening: xet fixed concurrency 3, read/connect timeouts 300s/30s, supervisor loop (upload-start.sh /
  upload-stop.sh).  A restart does NOT lose a part: xet dedups chunks already on the server.

### Committed but UNBUILT / UNMEASURED (opt-in, default path unchanged)
- `src/kernels/cuda/fork/fork_vecdot.cuh` vec_dot_ptq1_0_q8_1_quarter + `fork_fmt.cuh` Fmt<143> ipb 4 behind
  **-DSTRATA_PTQ1_QUARTER** (default = llama-tq whole-block dot).  Build: CMAKE_CUDA_FLAGS is empty in
  build-ds4-gpu's cache -> `cmake -S . -B build-ds4-gpu -DCMAKE_CUDA_FLAGS=-DSTRATA_PTQ1_QUARTER` (rebuilds
  strata_kernels, ~5 min all cores; never while a GPU run uses the binary).
- `iq_kernels.cu`: STRATA_EXP_MODE_T<type>=<1|2|4> per-type layout override + PTQ1_0 (143) / Q3_K (11) added to the
  multi-row "AMD" layouts (mode 2 = two rows per warp, 4 = four rows, 1 = 64-lane row).  Runtime env, no rebuild flag.
- ds41_moe.cpp: m_tc transcode scratch capped at STRATA_MMQ_TC_GIB (1) - should restore the 3 GiB prestage region
  and the ~1.2 s dense-half loss on 8K (verify: 8K prompt split dense ~10.8 s, prestage 3.00 GiB in the log).

### In flight when the session ended
1. **HF upload** (supervisor loop, survives terminal/Claude restarts; NOT a reboot): part 1 of 16 at ~5 MB/s
   (~9 h total).  Check: `tail ~/AI/v41-dev/hf/upload.log`; chunk progress in the newest
   ~/.cache/huggingface/xet/logs/*.log ("Completed upload_xorb").  After a reboot: mount SSD NVME
   (`udisksctl mount -b /dev/nvme0n1p3`) then `bash tools/v41/hf/upload-start.sh`.  When ALL DONE: remove the
   "Upload in progress" notes from tools/v41/hf/README.md + README.md, re-upload the card.
2. **Dense requant test** ~/AI/v41-dev/dq/run.sh -> results.txt (q5_K / q4_K rows were still running).  So far:
   q6_K code ppl 2.3818 (q8 2.3790, KL 0.010, top-1 97.6%), reason 6.6496 (6.5974, KL 0.015), agent 14.16 (14.47 but
   KL 0.28, top-1 82.9% - high-entropy text, needs a noise baseline), **decode NOT faster** (26.8/28.6 vs 28.3/29.5:
   ggml's Q6_K mat-vec costs more per byte than Q8_0's).  Mal: if a type is promising (faster AND close quality), do a
   full coding-quality run with it; else drop the idea (it frees VRAM, ~+100 experts, but does not speed decode).

### NEXT (Mal's priorities: decode kernels first; he is happy with prefill for now)
1. **Ternary decode kernels.**  Gate: `build-ds4-gpu/test_ds41_moe_gpu GGUF --layers 0,20,32,39 --gpu --tokens 8`
   (decode path vs F32; today worst rel ~1.2e-2).  Speed: `~/AI/v41-dev/k143/ab.sh TAG [ENV=VAL ...]` (p256 x 256,
   --slots 1500 pinned, STRATA_DEC_PROF; appends to ~/AI/v41-dev/k143/results.txt).  Order: baseline (current
   binary) x2; STRATA_EXP_MODE_T143=2, =4; T11=2/4; rebuild with -DSTRATA_PTQ1_QUARTER, gate, x2 (+ with T143 modes).
   Then the Prism-fork idea (PrismML-Eng/llama.cpp ggml-cuda/vecdotq.cuh): exact int16 activation sums in q8_1 ds.y
   (quantize_q8_1<exact_isum>) -> drop the sumu dp4a (half the ternary dp4a work).  Safe here: our Q2_0 / Q3_K dots
   do not read ds.y (checked: iq_kernels.cu vec_dot_q2_0_q8_1 maps codes with a byte-perm table).  Re-profile with
   nsys (~/AI/v41-dev/nsys/run.sh pattern; decode-only window via nsys export sqlite).
2. Other decode items: argsort of 384 (k_argsort 9 us x 80/token) -> top_k; F16 indexer/compressor matrices -> Q8_0
   (V4 engine's --f16-q8: ~0.4 ms + VRAM); router computed twice per layer (attn_router + finish_predict F32 7.9 MB
   each); host syncs (~270/token: attn_router readback, tier layer sync, predict readback).
3. **DSpark decisive test** (it looked weak: 0.73 accepted/pass with decode biases, 0.41 EXACT routing; V4 in
   llama.cpp 3.4).  Teacher states for the same 639 tokens:
   `V41_WORK=/media/Crucial1TB/v41-teacher python3 tools/v41/v41_calib.py prep-ids ~/AI/v41-dev/r1ab-1237/p256.i32
   ~/AI/v41-dev/dspark/gen_exact.i32` -> `init` -> `layer L --no-gptq` for L = 0..39 (GPU ~20 GB VRAM + ~20 GiB RAM;
   NOT beside the engine; shards on Crucial1TB, ~15-30 min) -> `python3 tools/v41/dspark_accept.py
   ~/AI/v41-dev/dspark/main_exact.bin ~/AI/v41-dev/dspark/gen_exact.txt --teacher /media/Crucial1TB/v41-teacher`
   (prints engine-vs-teacher cosine per target layer too).  Teacher ~3 accepted -> our 2-bit trunk drift is the
   cause; teacher also ~0.5 -> harness bug or DSpark is weak, stop.
4. Vision (exported in the sidecar /media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf, no engine code).

### Rules (keep)
- Kill by PID only - never `pkill -f`/`pgrep -f` + kill with a pattern that appears in your own command line (killed
  my own shell again this session).  GPU runs via `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- ...`;
  wait for MemAvailable >= 75 GiB between runs; one GPU job at a time; never rebuild build-ds4-gpu while a run uses
  it; CPU-heavy work (builds, torch harnesses) skews decode timing - serialise.
- Measure before claiming; report ppl/KL for anything lossy; Mal decides quality trades; soft routing only.
- README = current results; history in docs/ds4/V41_PROGRESS.md; commit messages end with the Co-Authored-By line.
- Mal OK'd "a couple of workers for extra hands" (/conductor, flash): good fits = the A/B run batches, nsys
  analysis, docs; keep kernel edits + verification with the conductor.

### Paste-able resume prompt
Continue the DeepSeek-V4.1-Flash Strata engine work for Mal in ~/AI/Strata-DS4 (branch deepseek4; push with
`git push ds4 deepseek4:main`).  FIRST read docs/ds4/V41_RESUME_PROMPT.md section "HANDOFF 2026-10-12 ~19:30"
fully (copy: ~/AI/DS4_V41_RESUME_2026-10-12.md).  Check the two in-flight jobs (HF upload log, dense requant
results), then work the NEXT list in order: ternary decode kernels (gate, then A/B), other decode items, the DSpark
teacher test, vision.  Mal may use /conductor with a couple of flash workers for extra hands; verify their work
yourself.

## UPDATE 2026-10-12 ~19:10 - READ FIRST (then ~18:00)

- **Prefill MMQ DEFAULT (5f1a768c):** prompt experts on int8 tensor cores, PTQ1_0 via the lossless device transcode.
  8K: 219-220 -> 244-248 tok/s, 4096 code ppl 2.3799 -> 2.3790, KL 7.6e-3, top-1 agree 97.9%, deterministic; 8K
  --dump-last md5 is now **ee31507a0e88** (--no-chunk-mmq: 4d229102872a).  Uncommitted: m_tc capped at
  STRATA_MMQ_TC_GIB (1) - taking all free VRAM shrank the prestage region 3.00 -> 1.37 GiB (dense half +1.2 s).
- **DSpark verdict so far: NOT worth building on this trunk.**  Acceptance 0.73/pass with decode biases, 0.41 with
  EXACT routing (worse!), confidence head uninformative.  Both generations are coherent code (exact one better).
  Pending decisive test: teacher (FP8) hidden states for the SAME tokens via `v41_calib.py prep-ids` (added) on
  p256.i32 + ~/AI/v41-dev/dspark/gen_exact.i32 (383 ids) -> init -> layer 0..39 --no-gptq -> h/L37-39 -> run
  dspark_accept on those (needs a small loader for h/LNN.bin [1,S,4,5120] bf16-as-int16) + `head` for teacher top-1.
  If teacher states also give ~0.4-0.7, DSpark is just weak here; if ~3, our 2-bit trunk drift kills it.
- **Decode profile (nsys, decode-only window):** dense Q8_0 mat-vec 8.84 ms/token (bandwidth floor), expert kernels
  ~7 ms: native_gu<143> 2.72 ms (76 us/launch, ~136 GB/s), down<143> 1.31, gu_multi<42> 0.93, down_multi<42> 0.75,
  Q3_K ~1.1; argsort 0.72, F32 router+predict 0.77, F16 indexer/compressor 0.7, flash-attn 0.69.
- **Ternary decode kernel work (UNBUILT, src/kernels/cuda):** fork_vecdot.cuh vec_dot_ptq1_0_q8_1_quarter + Fmt<143>
  ipb 4 (fork_fmt.cuh; -DSTRATA_PTQ1_WHOLE = old); iq_kernels.cu STRATA_EXP_MODE_T<type>=<1|2|4> per-type layout
  override + PTQ1_0/Q3_K in the AMD (multi-row) layouts.  A/B script ~/AI/v41-dev/k143/ab.sh TAG [ENV=..]
  (--slots 1500, p256 x 256, DEC_PROF).  Gate first: build-ds4-gpu/test_ds41_moe_gpu GGUF --layers 0,20,32,39 --gpu
  --tokens 8 (decode path vs F32).  Prism fork (PrismML-Eng/llama.cpp, vecdotq.cuh) ideas: exact int sums in q8_1
  ds.y (drops the sumu dp4a - safe here: Q2_0/Q3_K dots do not read ds.y), qh trits paired into dp4a, _multi
  (decode once, many columns: for verify windows), MMQ tile loader expanding PTQ1_0 to int8.
- Dense requant test RUNNING: ~/AI/v41-dev/dq/run.sh -> results.txt (q8 baselines: code 2.3790 / agent 14.4735 /
  reason 6.5974, decode 28.31 tok/s).
- HF upload: running (part 1), xet concurrency capped at 3 (upload-start.sh) after ~1/3 of bandwidth was lost to
  timeouts; resumes on rerun.

## UPDATE 2026-10-12 ~18:00 (evening session) - READ FIRST

**HF release LIVE (public): https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored** - card =
tools/v41/hf/README.md (full GSQ process, measured tier mix 42.8% ternary / 39.2% Q2_0 / 18.0% Q3_K = 2.250 bpw,
per-layer held-out error from /media/Crucial1TB/v41-quant-mixed/quant.log), reassemble.sh (tested: join, --delete-parts,
corrupt part refused), Engram placeholders model-0004{7,8}-...NEED-TO-DOWNLOAD.txt.  Upload = tools/v41/hf/hf_upload.py
(resumable, 16 x 10 GiB byte parts staged to NVME1TB/hf-staging, source GGUF O_RDONLY only, deletes only its staged parts);
start/pause: tools/v41/hf/upload-start.sh / upload-stop.sh, log ~/AI/v41-dev/hf/upload.log, state (sha256 of every
part + whole file) ~/AI/v41-dev/hf/state.json.  ~1.6 h per part at ~1.85 MB/s; after a reboot just run upload-start.sh.
README links HF; HF card links GitHub.

**DSpark (V4.1 native drafter) - offline acceptance on OUR trunk: 0.73 accepted / pass (1.73 tokens per verify pass;
first draft token 47%; dist 0:53% 1:29% 2:11% 3:5% 4:1.1% 5:0.3%).**  V4 DSpark in llama.cpp: 3.4.  Harness:
tools/v41/dspark_accept.py (DeepSeek model.py on our torch kernels + bf16 weight cache, ~2 s/draft) on
STRATA_DUMP_MAIN dumps.  BUG FIXED on the way: main_hidden = the state ENTERING target layers 37-39 (= output of
36-38; model.py "reads the attention input of its target layers") - the first dump was one layer late (0.15).
IN FLIGHT: exact-routing dump (~/AI/v41-dev/dspark/dump_exact.sh -> main_exact.bin + gen_exact.txt, --route-bias 0
--route-bias-file 0).  Run dspark_accept.py on it: if acceptance jumps, the decode biases are the cause (verify with
less bias); if not, DSpark is a poor match for the 2-bit trunk -> spend the effort on dense/prefill instead.

**Prefill finding:** the prompt path's expert kernels are dp4a MMVQ (CUDA cores), ~10 s GPU per 4K tokens.  Strata's
MMQ (int8 tensor cores) covers Q2_0 + Q3_K (+ MXFP4) but NOT PTQ1_0, and --chunk-mmq gates per class -> every class
with a ternary matrix stayed on MMVQ.  BUILT (uncommitted, unbuilt into ds41_generate yet): tools/v41/engine/
ds41_transcode.cu (PTQ1_0 -> Q2_0 on the device, same d, code = trit+1: LOSSLESS; test_ds41_transcode ALL PASS, 11.8M
values incl. a full gate matrix + edge d) + ds41_moe.cpp mmq_plan()/mmq_launch transcode path (STRATA_MMQ_PTQ1=0 off;
copies in gp.m_tc capped by free VRAM - 2 GiB; release_chunk now frees MMQ buffers).  NEXT: build, test_ds41_moe --gpu
--chunk 512 --chunk-mmq on layers 0,20,39 (vs F32 ref), then 8K prompt --chunk-mmq A/B (speed, md5 WILL differ:
different kernels; gate on 4096 code ppl 2.3799 +- tiny and KL).  H2D: cudaHostAlloc 22.5 GB/s, 4K-page registered
arena 18.8-19.7 GB/s (scratchpad h2d.py), engine achieves ~13 -> the copies are fed badly, not page-size-limited.

**Dense requant experiment (Mal: side project):** STRATA_DENSE_REQUANT=q6_K|q5_K|q4_K (load-time, file untouched,
every Q8_0 dense matrix; STRATA_DENSE_REQUANT_HEAD=0 keeps output.weight Q8_0) added to ds41_dense.cpp (uncommitted).
Measure 4096 code + agentic ppl + KL vs Q8_0 + decode tok/s + slots freed; Mal decides.  Dense = ~7 GB of 163.

nsys profiles queued/running: ~/AI/v41-dev/nsys/run.sh -> pf8k.nsys-rep (8K prefill) + dec.nsys-rep (decode).

## UPDATE 2026-10-12 ~10:15 (Mal shut the PC down mid-run) - READ FIRST

HEAD = 3f8c4f67 + this doc.  Done this session:
- **fq patch APPLIED** to third_party/llama.cpp (untracked tree is now patched; build-ds4-gpu configured
  -DSTRATA_DSV41_FQ_OP=ON).  test_ds41_fq --cuda bit for bit; 8K md5 4d229102872a; decode p256 x 256 = 27.37 / 26.62
  tok/s (was 24.4-25.2 today), 28.9 tok/s on a 384-token run.  Two runs' tokens differ at token 86 ONLY because the
  VRAM cache size differed (1547 vs 1540 slots -> cache-aware route bias): use --slots N for reproducible A/Bs.
- Arena verdict: keep 60 GiB (see below).  128 GB RAM test deferred (note above).
- **Speculative file-tier reads** (3f8c4f67, OPT-IN STRATA_SPEC_TOP=4/8, default 0): predicting 2 layers ahead catches
  60-81% of NVMe misses (tokens bit-identical, --slots 1500), tier -1.8 ms/token, but finish graph +1.5 and issuing
  +1.3 ms -> break-even (24.97-25.10 vs 25.56).  Gotchas fixed: argsort_top_k is a VIEW (row stride NE), 8 readers +
  notify_all preempted the decode thread (now 4 + notify_one, that variant measured only under export contention -
  re-measure on a quiet box).  Wrong-guess reads slow layer-1 Engram (same NVMe).
- **DSpark = V4.1's native drafter** (the mtp.* weights).  From DeepSeek model.py (~/AI/gsq-rco/dsv41-inference):
  3 blocks run as ONE stack, block_size 5 (1 real token + 4 noise id 128799) drafted in one pass, input main_hidden =
  HC-mean of trunk layers 37/38/39 concatenated (15360) -> main_proj -> main_norm; each block = window-128 attention
  (DSparkAttention, keeps its own window KV of main_kv) + MoE 128 experts top-3 + shared expert + mHC; stage 2 has
  norm + markov_head (embed/head rank 256, sequential logit bias) + confidence_head; output head = trunk head.weight.
  Mal: wants draft ~6 or less (had good results with 6); native block is 5.
- **Sidecar exported + verified**: /media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf (8.42 GB, 347 tensors):
  mtp.0-2 experts MXFP4 = lossless byte repack of the FP4 (36 sampled experts bit-exact), other 2-D weights Q8_0
  (worst rel RMS 6e-3), norms/router/mHC F32, vision.* + aligner.* (Q8_0) + image_* .  tools/v41/v41_sidecar.py.
  Resident cost for DSpark in VRAM ~2.7 GB per block if all experts resident (decide: all 3 blocks ~8 GB is too much ->
  experts via tier or keep only hot ones; measure).
- **Offline acceptance test** (tools/v41/dspark_accept.py): engine dump STRATA_DUMP_MAIN=F (last 128 prompt tokens run
  as steps, then decode) -> ~/AI/v41-dev/dspark/main_p256.bin (511 records) + gen_p256.txt.  The script was LOADING
  when Mal shut down - NO RESULT YET.  Rerun (CPU, ~10 GB RAM, engine must NOT be running):
  `python3 tools/v41/dspark_accept.py ~/AI/v41-dev/dspark/main_p256.bin ~/AI/v41-dev/dspark/gen_p256.txt --max 64`
  (unverified: first real run of the loader - expect fixups; prints mean accepted length, per-depth tokens/pass,
  confidence calibration).  Reference: V4 DSpark in llama.cpp = 3.4 accepted / block of 5 (FINDINGS s7).
- Vision: exported only; no engine code.  Plan (Mal): Q8_0 (or less if viable), lives in RAM/NVMe, loaded on demand.

NEXT (Mal's priorities): (1) DSpark acceptance number -> if good, engine: DSpark stack as ggml graph (resident dense,
experts resident or tiered), main_hidden capture at layers 37-39 (read_state per step is the slow measurement path -
do it in-graph), window KV, then verify pass of 1+5 tokens (pass_m / run_multi + rollback via snapshot), draft depth
configurable 1-5, use confidence to cut.  (2) Dense half host overhead (~5 ms/token syncs + tiny kernels) - Mal wants
this.  (3) vision engine.  (4) stashed span-upload change, fast exact quick path, text-log poem check (from ~08:00).
HF upload still awaiting Mal's answers (public/private, repo name, model card) - ~25 h at 1.85 MB/s, 4 x ~41 GB parts.

## UPDATE 2026-10-12 ~08:00 (supersedes MORNING where they differ) - READ FIRST

Done since MORNING (all pushed, HEAD = 76d69d95 + this doc):
- 9356d8e6 server text log (/tmp/strata-ds41-text.log, STRATA_TEXT_LOG) + api_monitor (/api-monitor) - NOT yet seen on
  a live wrapper run.  tools/v41/serve/drive_engine.py = serve-protocol driver (survives /tmp wipes now).
- 3e8a7479 **fresh 20K prompt VRAM OOM FIXED** (was hitting all modes: prestage region left only 1.5 GiB for the dense
  pass buffer = 2.3 GB at 20K).  Prestage reserve 3 GiB (STRATA_PRESTAGE_KEEP_GIB) + Ds41Dense::set_oom_hook ->
  Ds41MoeTier::release_prestage + retry.  Fresh 20K: 266-269 tok/s.
- **Exact quick path verdict** (~/AI/v41-dev/xq, 20K ctx; turns 41 / 300 / 641 tokens): exact passes 6.9 / 48.1 / 86.2 s
  (KL vs batch 0.013 / 0.012 / 0.0002), biased passes 1.1 / 6.4 / 11.3 s (KL 0.15 / 0.10 / 0.009), batch 18.0 / 18.5 /
  18.7 s.  Exact passes ~6-7 tok/s (exact routing misses the VRAM cache) -> win only < ~110 tokens.
  **Wrapper default now (ds41-engine.sh): STRATA_SERVE_STEP_MAX=96 + STRATA_SERVE_PASS_BIAS=0** = every prompt token
  exact; biased quick path opt-in `STRATA_SERVE_STEP_MAX=768 STRATA_SERVE_PASS_BIAS=1`.  Mal still wants a FAST quick
  path that does not skew results - open problem (ideas: exact routing but smarter miss handling for passes - the
  misses are what cost; run misses via run_chunk-style grouped streaming per pass; or larger passes).
- 76d69d95 README rewritten: "Results" first-run vs now + serving table; step-by-step history moved to
  docs/ds4/V41_PROGRESS.md (Mal: README = what we achieved, docs = progression).
- Duplicate MTP shards on NVME1TB deleted (cmp-identical to Crucial's; Mal OK'd).  Crucial1TB/models/...-FP8 has 46/48
  shards incl. 1 (vision) and 44-46 (MTP).  Engram runtime tables: L1 on SSD NVME/AI/V41-Engram, L14 on
  Crucial1TB/V41-Engram (extracted, fast); shards 47/48 on NVME1TB are only the source.

**ARENA RESULT (done ~08:20): keep --arena-gib 60.**  p256 x 256 decode: arena 60 = 25.21 / 24.41 tok/s (75 GB peak
system RAM), 62 = 24.49 (77 GB), 65 = 24.70 (80 GB); NVMe reads 701/619 -> 620 -> 584 (-10%) but tok/s within noise
(the file penalty already keeps decode off the file tier).  Implication for Mal's 128 GB idea: the speed slope is flat
here; the realistic payoff is lowering the file penalty (quality) at similar speed - measure that before buying.
(Original note:) **arena size test** ~/AI/v41-dev/arena/run.sh -> results.txt (one line per run:
tier line, decode tok/s + file reads, peak system RAM GB) for --arena-gib 60, 62, 65, 60 (p256 x 256 one-shot).  Mal:
total RAM ~77 or ~80 GB is OK only if measurably better (wants ~10 GB left for the desktop/Firefox; pinned memory
cannot be reclaimed).  If 62/65 help clearly: set "--arena-gib" in tools/v41/serve/strata-ds41.json (engine default 60)
and the memguard cap (ds41-engine.sh MemoryMax=84G may need 86-88G).  Mal is considering 128 GB RAM (replace the 16 GB
stick + add 64 GB); my estimate: decode ~28-30 tok/s at today's biases (NVMe misses -> RAM) or same speed with a lower
file penalty; prefill ~unchanged.  Caution: known bad byte lane @56.6 GB (memory note ram-fault-2026-09-18), 4x mixed
DIMMs may downclock.

**LATER (Mal 2026-10-12, not now): the 128 GB RAM question as a quality test.**  Decode p256 x 256 with
--route-bias-file 0 and 0.3 at --arena-gib 60 and 65.  If the bigger arena closes much of the speed gap at a low file
penalty, 128 GB buys near-exact routing (code ppl 2.87 -> ~2.67) at close to today's speed; if not, skip the RAM.

**Decode budget + small-win ideas (~08:45, from the arena logs + NIGHT-2 profile, ~39 ms/token):**
dense half ~19 ms (GPU ~14, rest host syncs/launches), tier ~16 ms (layer sync ~6, NVMe reads ~5.7 = 2.4 reads/token at
~2.3 ms each ON the critical path - read only after routing is known, cpu-tier experts ~1.8).  Ideas, smallest first:
1. fq patch (+5%, applying now).
2. Speculative file-tier reads: Ds41MoeTier::prefetch turns a predicted file-tier expert into a DUMMY DMA (wasted).
   Instead start its O_DIRECT read k=2-4 layers ahead (dense.predict(l+k), like STRATA_ADVISE_AHEAD - which is useless
   now: it hints the page cache, but the reads are O_DIRECT) into a small pinned stash; the miss path checks the stash
   first.  The SSD is idle during decode, so wrong guesses only cost bandwidth.  FIRST measure: how many of a token's
   actual file misses are in predict(l+k)'s top-16, for k = 1..4.
3. Second NVMe: mirror the hottest file-tier experts (by rank.bin) onto NVME1TB (970 EVO Plus, 48 GB free, exFAT - needs
   O_DIRECT; reformat ext4 per the NInfer note) and split each read_part across both drives (~2.3 -> ~1.3 ms per read).
4. Dense half host overhead (~5 ms/token of syncs / tiny kernels) - biggest remaining lever, more work (fusion).
Swap/zram is not a lever: the arena is pinned and the engine scope has MemorySwapMax=0.

Queue now: (a) read arena results, decide arena; (b) MTP head export Q8_0 + engine draft/verify; (c) vision export +
engine/shim; (d) fq patch apply (Mal OK'd) + verify; (e) stashed span-upload change validate; (f) a fast exact quick
path idea; (g) watch the text log on Mal's next session (poem request thought 6,590 tokens - loop or long thinking?).

## HANDOFF 2026-10-12 MORNING (~06:30) - wrapper tested by Mal, queue: exact quick path, MTP + vision export, fq patch

### State in one paragraph
Repo ~/AI/Strata-DS4, branch deepseek4, GitHub Indras-Mirror/Strata-DSV4.1 (remote `ds4`, `git push ds4 deepseek4:main`,
pushing OK'd by Mal).  HEAD = 9356d8e6 + this doc.  The serve wrapper `~/.local/bin/strata-ds41-quetza` works:
Mal ran a real Quetza session 2026-10-12 05:38-06:20 (no errors).  build-ds4-gpu/ds41_generate = HEAD (built 06:2x).
The box rebooted overnight (~05:17): **"SSD NVME" (/dev/nvme0n1p3) does NOT auto-mount** - it holds V41-GSQ.gguf,
rank.bin, tokenizer and engram-L1.  Fix: `udisksctl mount -b /dev/nvme0n1p3` (gives ntfs3, which the raw Engram
reads need; check `mount | grep "SSD NVME"` says ntfs3).  /tmp was wiped by the reboot (old scratch logs gone).

### Mal's live session numbers (512K ctx, quick path ON because that engine started before dd6a185d)
fresh 20,000-token prompt: 77.6 s = 258 tok/s (batch, exact) · 11,260-token tool result on 20K reused: 49 s = 230 tok/s
· quick-path turns: 27 tok in 0.8 s, 500 tok 10.9 s (46 tok/s), 668 tok 15.7 s · decode 22.7-28.2 tok/s, 96-98% VRAM hits
· a "write a poem" request thought for 6,590 tokens (cancelled by Mal; Ctrl-C/STOP worked).  Unknown whether it LOOPED
(no text was logged then - text log added since, see below).  Reasoning effort default = 75 (high) in the shim.
Note: each turn re-reads the previous assistant reply (DeepSeek's template drops earlier thinking -> lcp falls back to
the snapshot at the end of the last prompt) - expected, costs ~reply length.

### Mal's decisions today
1. **No quick path by default** (dd6a185d: ds41-engine.sh exports STRATA_SERVE_STEP_MAX=0) because agent sessions do
   many tool calls and the quick path routes those prompt tokens with the decode biases.  BUT every turn now pays the
   batch floor (~16.8 s + n/250).  He wants the **EXACT quick path** (passes with exact routing) if it is fast enough.
   `STRATA_SERVE_STEP_MAX=768 strata-ds41-quetza` = old quick path (env passes through to the engine).
2. **Apply the +5% fused fake-quant patch** ("if it's good") - tools/v41/ggml/dsv41-fake-quant.patch.
3. **Export the MTP head (3 layers, mtp.0-2) - keep it ~Q8_0 - and the vision tower; we want both.**
4. Full text in the server logs (done: 9356d8e6).

### Files / sources for the exports (VERIFIED 06:25)
- Full FP8 checkpoint (46 of 48 shards) on **/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/** incl.
  shard 1 (vision.* 259 tensors) and shards 44-46 (mtp.* 2401 tensors, ~7.9 GB, `.ok` markers).  index json there too.
- I mistakenly re-downloaded shards 44-46 into /media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/ (duplicates,
  deleted 2026-10-12 with Mal's OK).  Engram: the engine reads the EXTRACTED tables - layer 1 on
  /media/mal/SSD NVME/AI/V41-Engram, layer 14 on /media/Crucial1TB/V41-Engram (fast ntfs3 drives, raw block reads);
  the original checkpoint shards 47/48 on NVME1TB (exFAT, slow) are only the source, unused at runtime.
- mtp.N.* names: attn, attn_norm, ffn (experts!), ffn_norm, hc_attn_*, hc_ffn_*, main_norm, main_proj, ... (DSpark /
  V4.1 MTP = 3 layers).  The GSQ GGUF writer = tools/v41/ (v41_gguf.py / quant pipeline, see §Export pipeline below);
  mtp.* and vision.* were simply never in the export plan ("mtp.* not exported yet").
- Engine side for MTP: run_multi (tier, up to 4 tokens; per-expert geometry falls back to token-by-token) and the serve
  passes (pass_m) are the verify building blocks; the V4-0731 engine (tools/ds4) has a full MTP draft/verify
  implementation to port from (`--mtp`, `--verify`, FINDINGS s19: depth 1 wins at ~60% acceptance).
- Vision: no engine code at all yet.  Needs: export vision tower, ViT forward (ggml), image-token injection, shim image
  input (the server already has a Vision class for Qwen - serve/server.py `class Vision`).

### Work queue (Mal's order)
1. **Exact quick path** - code in HEAD (STRATA_SERVE_PASS_BIAS=0, built, UNMEASURED).  Measure with
   tools: serve driver (rewrite of the lost scratch serve_drive.py: drive ds41-engine.sh over stdin with
   `GEN n temperature=0 ids`; DRIVE_ENV passes env).  Compare per turn size (41 / 300 / 641 tokens on a 20K context):
   exact passes vs biased passes vs batch (STEP_MAX=0).  Exactness: STRATA_SERVE_DUMP last-row logits vs the batch
   path (expect cos ~0.999 = batch-shape noise).  If exact passes beat the batch floor for small turns, set the wrapper
   to STEP_MAX=<crossover> + PASS_BIAS=0 and tell Mal.
2. **MTP head export at Q8_0** (+ engine draft/verify) and **vision tower export** (then engine + shim support).
3. **Fused fake-quant patch**: `patch -p1 -d third_party/llama.cpp < tools/v41/ggml/dsv41-fake-quant.patch`
   (use /usr/bin/patch - anaconda's patch is broken), `cmake -S . -B build-ds4-gpu -DSTRATA_DSV41_FQ_OP=ON`, rebuild
   (~5 min, all cores - never while Mal is using the wrapper), run `build-ds4-gpu/test_ds41_fq --cuda` (must say bit for
   bit) and the 8K exactness check (md5 4d229102872a), then decode p256 x 256 (expect ~27.7 tok/s).  Record in docs
   that the local third_party tree is now patched.
4. **Stashed** (`git stash list`: "span-upload precompute"): uploads all layer-major sub-batch index maps once in
   begin_batch instead of 4 tiny synchronous tensor_sets per span (measured upload_span = 2.97 s of a 37 s 8K prompt,
   probably queued behind prestage DMAs) + STRATA_SPAN_PROF profiler.  NOT VALIDATED (my "md5 OK" this morning ran
   on the committed code - the stash had already happened).  `git stash pop`, build, check md5 4d229102872a + prefill
   tok/s warm (2 runs each A/B).
5. Check the text log on the next live run: `tail -f /tmp/strata-ds41-text.log`; if the poem case loops, investigate
   (route bias / quick path?) - otherwise maybe lower the default reasoning effort (shim EFFORT default 75).

### Server observability (9356d8e6)
- `/tmp/strata-ds41-text.log`: every request's header (prompt tokens, max, thinking, tools), the thinking / answer /
  tool-call text as generated, end line (finish, tokens, seconds).  STRATA_TEXT_LOG=<path>|0.
- `/api-monitor` (config "api_monitor": true): last 100 prompts + answers.  `/metrics` (JSON or Prometheus),
  `/slots` (live reasoning_tail / output_tail), engine log /tmp/strata-ds41.log (`serve:` lines per request),
  server log /tmp/strata-ds41-quetza-8150.log (progress lines).

### Rules (keep)
- third_party/llama.cpp is gitignored/untracked: ops go in patch files under tools/v41/ggml/ (opt-in), never committed
  there.  Mal OK'd applying the fq patch locally (2026-10-12).
- Don't rebuild build-ds4-gpu while a script or the wrapper uses it.  GPU runs:
  `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 84 70 -- env VARS build-ds4-gpu/ds41_generate ...`.
  After killing an engine wait for MemAvailable >= 70 GiB.  Kill by PID; never `pgrep -f` a pattern in your own cmdline.
- Exactness references: 8192-token code prompt (eval/seq20_code.i32[:8192]) --dump-last md5 4d229102872a;
  p256 = /home/mal/AI/v41-dev/r1ab-1237/p256.i32.  Standard flags: --backend cuda --experts gpu --engram
  "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram" --rank "/media/mal/SSD NVME/AI/V41-GSQ/rank.bin".

### Paste-able resume prompt
    Continue the DeepSeek-V4.1-Flash Strata engine work for Mal in ~/AI/Strata-DS4 (branch deepseek4, GitHub remote
    `ds4` = Indras-Mirror/Strata-DSV4.1, pushing OK'd). Read docs/ds4/V41_RESUME_PROMPT.md §HANDOFF 2026-10-12 MORNING
    fully first (then NIGHT-2). First: make sure "SSD NVME" is mounted (udisksctl mount -b /dev/nvme0n1p3, ntfs3).
    Queue in Mal's order: (1) measure the exact quick path (STRATA_SERVE_PASS_BIAS=0, already in HEAD) vs biased passes
    vs batch on 41/300/641-token turns over a 20K context + exactness via STRATA_SERVE_DUMP, and set the wrapper default
    from the result; (2) export the MTP head (mtp.0-2, ~Q8_0) and the vision tower from
    /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 (shards 44-46 and 1), then engine MTP draft/verify
    (port from tools/ds4) and vision; (3) apply tools/v41/ggml/dsv41-fake-quant.patch, -DSTRATA_DSV41_FQ_OP=ON, verify
    test_ds41_fq + md5 4d229102872a; (4) validate the stashed span-upload change. Rules: memguard for every GPU run,
    never rebuild while the wrapper/scripts use build-ds4-gpu, measure before claiming, report ppl/KL of anything lossy,
    soft routing only. Commit with "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>".

## HANDOFF 2026-10-12 NIGHT-2 (~00:10) - serve wrapper live, decode 25.5-26.3 tok/s (27.7 opt-in), short turns ~40 tok/s

Mal (2026-10-11 ~20:10): "set the wrapper up for our current thing, update github ..., keep working on the decode
stuff, dont make mistakes".  All below is committed and pushed to ds4/main (HEAD after this doc).

### Wrapper / --serve (DONE, installed)
- `~/.local/bin/strata-ds41-quetza` (= tools/v41/serve/strata-ds41-quetza.wrapper; port 8150, 512K ctx, QuetzaCodetl)
  -> serve_ds41.py (Strata's server: OpenAI + Anthropic + DSML tools; 40 shim tests pass) -> ds41-engine.sh
  (GPU lock, 84G cap, now exports STRATA_NO_LARGEPAGES=1) -> `ds41_generate --serve`; strata-ds41.json = tuned defaults
  + the new Engram dirs.  Checked end to end over HTTP: OpenAI chat, Anthropic tool_use + tool_result follow-up
  (reused 302 cached tokens through the snapshot).
- ea04b97f: per request the VRAM expert cache is closed for a long prompt (Ds41MoeTier::release_cache) and re-seeded
  for the decode (~0.55 s); enc-skip on in serve; regenerate (same prompt) = snapshot + saved logits, no prefill;
  serve exits with _Exit after QUIT (the detached stdin reader held cin's lock -> exit hung until the pipe closed);
  STRATA_SERVE_DUMP=<dir> writes each request's last-row logits.
- Exactness (512K ctx): 8192-token request md5 4d229102872a = the one-shot exact path; 256-token request byte-identical
  to one-shot; 512 reused + 7680 new = 4d229102872a.  Misaligned reuse (256 + 7936) = cos 0.9985 (= the sub-batch
  boundary shift; S=256 vs 512 one-shot gives 0.9991).
- 1cf5d37d / e66471f3: turns of <= 768 new tokens (STRATA_SERVE_STEP_MAX) run on the decode side in dense passes of
  16 (STRATA_SERVE_PASS_M; dense weights read once per pass, experts per token from the VRAM cache): 41-token turn
  18.0 -> 1.5 s, 300 tokens 11.4 -> 7.5 s (passes of 8), 641 tokens 15.5 s vs 19.4 warm / 25.0 cold on the batch.
  QUALITY NOTE: those prompt tokens route with the decode biases (as decode).  Last-row logits of a 256-token prompt
  vs exact: KL 0.27 (passes) / 0.30 (steps), top-1 shifted.  STRATA_SERVE_STEP_MAX=0 = every prompt token exact.
  Mal may want a lower threshold - his call (it trades turn latency for prompt exactness).

### Decode (one-shot p256 x 256 tokens, default flags; run-to-run +-3%)
23.82 (start) -> a966904c O_DIRECT decode miss reads (STRATA_DECODE_DIRECT=0 off) + one pinned metadata copy per
grouped launch + one sync for attn_router's 3 readbacks: 24.46 -> 59920d80 routed sum on the card into the dense
routed tensor (run_to_dev, decode_rows_sum in double = host loop bit for bit; STRATA_DEC_HOSTSUM=1 off): 24.73 ->
5ed8126a NVMe-read experts DMA'd + run on the card instead of the CPU pool (STRATA_FILE_GPU=0 off): 25.5-26.3.
All bit-identical tokens except 5ed8126a (GPU vs CPU arithmetic for those experts; DS4_CHECK_GPU covers them).
- 3ac307db OPT-IN fused activation rounding: tools/v41/ggml/dsv41-fake-quant.patch adds GGML_OP_DSV41_FAKE_QUANT to a
  llama.cpp tree, -DSTRATA_DSV41_FQ_OP=ON uses it (refuses an unpatched tree).  test_ds41_fq: bit-identical to the
  composite on CUDA + CPU (6.3M values incl. edge cases); with it 8K md5 4d229102872a, tokens identical, 27.66 tok/s.
  NOT applied to third_party/llama.cpp (untracked - see the rule below).  Mal: apply it or not?
- Re-sweep --pcie {0.75,1.0} x --pf-b {1.43,2.0}: all within noise (24.3-25.3) -> defaults kept.
- Tried + reverted (no gain): persistent IO pool for the file reads (spawn cost moved, total +0.016 ms/layer).
- Profile now (~38 ms/token): dense half ~19 ms (Q8_0 mat-vecs ~8.6 ms at ~880 GB/s = bandwidth; ~14 ms GPU total),
  tier ~16 ms (layer sync ~6 ms = expert kernels + DMAs, NVMe reads ~5 ms), finish ~2 ms, begin ~0.8 ms.
- Prefill unchanged: 8192 tokens 221 tok/s x2 runs (md5 4d229102872a); serve at 512K: 250 tok/s.

### Open questions for Mal
1. Apply tools/v41/ggml/dsv41-fake-quant.patch to third_party/llama.cpp (+5% decode, exact)?
2. Short-turn threshold (768) vs prompt exactness (see QUALITY NOTE).
3. MTP drafting needs mtp.* (checkpoint shards 44-46) exported - those shards are no longer on NVME1TB (only 47/48).
4. Bigger decode lever: dense Q8_0 -> Q6_K would cut ~1.9 GB/token (~2 ms) at some quality cost - not done (lossy).

### Rules (add to the paste-able prompt)
- third_party/llama.cpp is gitignored / untracked: never add ops there (patch files under tools/v41/ggml/, opt-in).
- Don't rebuild build-ds4-gpu while a script is launching it (I broke a sweep that way tonight; reran it).
- After killing an engine, wait for MemAvailable to recover before the next memguard launch (70 GiB floor).
- GPU runs: `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 84 70 -- env ... build-ds4-gpu/ds41_generate ...`;
  serve tests: drive ds41-engine.sh over stdin (memguard backgrounds its command); kill by PID, never pkill -f.


## HANDOFF 2026-10-11 NIGHT (~20:15) - prefill 219-227 tok/s (8K) / 288 (16K), decode 21.1 tok/s - READ THIS FIRST

Session: inline conductor (workers all stopped; Mal said do the work directly). Repo GitHub **Indras-Mirror/Strata-DSV4.1**
(local remote `ds4`; `git push ds4 deepseek4:main` - Mal OK'd pushing progress). Branch deepseek4, HEAD after this doc.

### Current measured state (RTX 4090, 90 GB RAM, default flags)
| | value |
|---|---|
| Prefill 8192-token code prompt | 218.9-226.5 tok/s (split dense ~11 s / experts ~25 s; file wait 0.4 s) |
| Prefill 16384 tokens | 288.0 tok/s |
| Decode, 256-tok prompt, 128 tokens | **21.1 tok/s** (route-bias 1.0 + file penalty 0.6, the defaults) |
| Decode after 8K / 16K prompt | 21.0 / 25.4 tok/s |
| Prompt exactness | default 8192 last row md5 4d229102872a = the exact full path (enc-skip, prestage, single piece, Engram async/raw: all byte-identical); GPU prompt deterministic run to run |
| Quality (4096 tok, --ppl, deterministic) | code 2.3799 exact prompt. Decode-routing proxy (--bias-prompt 1550): code none 2.380 / rb1.0 2.674 / rb1.0+f0.6 2.873 / rb1.0+f1.0 2.937 / rb2.0 4.518; agentic_long 15.11 / 6.88 / 6.39 / 5.13 / 5.05 |

### Defaults now in ds41_generate (each measured; old behaviour flag in brackets)
--prompt-batch 8192 [0 = chunk path], --enc-skip on for one-shot [--no-enc-skip], prefetch slabs [--no-chunk-prefetch],
layer prestage 3 GiB [STRATA_PRESTAGE_GIB=0], one run_chunk piece (3 GB budget) [STRATA_CHUNK_BUDGET_GB],
routed sums device-direct [STRATA_ROUTED_HOST=1], --pcie 0.75, --pf-b 1.43, --vram-lru [--no-vram-lru],
--route-bias 1.0 and --route-bias-file 0.6 (Mal's calls; decode-only - the prompt is exact) [--route-bias 0 --route-bias-file 0],
fused finish+predict [STRATA_NO_FUSE_PREDICT=1], Engram 32 threads + 2M-row cache [STRATA_ENGRAM_CACHE_ROWS=0],
Engram raw block-device reads for files with >= 1024 extents [STRATA_ENGRAM_RAWDEV=0/1]. STRATA_PROM=1 stays OFF (+0.6% ppl).

### Engram tables (moved 2026-10-11 night)
`--engram "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"` (layer 14 on Crucial MX500 SATA ntfs3,
layer 1 on SSD NVME ntfs3; each dir has the full index). Built by `tools/v41/extract_engram.py` (verified 2000 random
rows/tensor). The SSD NVME file has 62,112 extents (drive 99% full) -> raw /dev/nvme0n1p3 reads via FIEMAP (Mal is in
the `disk` group; 256 spans checked at open). Decode Engram 10 -> 1 ms/token. The old FP8 checkpoint on NVME1TB (exFAT,
~13K random reads/s) still works with the old --engram path. Disk space now: SSD NVME ~7 GB free, Crucial1TB ~17 GB.
Mal deleted SSD NVME's pagefile.sys (Windows recreates it).

### Commits tonight (after ee9dbec8)
1f7d8a4d decode profiler STRATA_DEC_PROF + --bias-prompt quality proxy + file readahead hook (off: no gain) ·
3532f148 tuned defaults + extract_engram.py · 3d4328cd Engram raw reads + fused finish_predict + file penalty default ·
18a01111 biases decode-only (prompt exact).

### Decode profile (STRATA_DEC_PROF=1, per token, file penalty 0.6): begin (Engram) 1.4 ms, predict 0.4, attn_router
~20.5, tier ~23.6 (first 64 tokens, includes cold file reads), finish 2.7.  nsys (--cuda-graph-trace=node): GPU kernel
time ~20 ms/token (~5,350 kernels; dense mat-vec ~8 ms; k_bin_bcast/rms_norm/scale/unary ~1,300+ tiny kernels), host
does ~380 cudaStreamSynchronize + ~110 graph launches per token -> GPU idle ~60%.

### NEXT (in order)
1. Decode dense half (attn_router ~20 ms/token): per-layer syncs + readbacks (ids/wts/fn = 3 tensor_get), merge into one
   readback; look at the ~1,300 k_bin_bcast/scale/unary kernels per token (fusion, as the V4-0731 engine did);
   GPU-side expert dispatch for VRAM hits without a host round trip.
2. Decode tier (~24 ms/token cold): remaining file/CPU misses; R1 merged (16b05583).
3. Prefill > 300 at 8K: expert H2D runs ~10 GB/s effective (copy engine ~40% busy; pinned copies median 13 GB/s); prestage
   the next layer's file-tier experts from the slab too (VRAM permitting); the dense half 11 s.
4. ~~Wrapper~~ DONE 2026-10-11 ~22:20 (ea04b97f): --serve defers / releases the VRAM cache per request, enc-skip on,
   short turns (<= 384 new tokens) as decode steps, regenerate = snapshot + saved logits; wrapper INSTALLED as
   ~/.local/bin/strata-ds41-quetza (port 8150, 512K ctx); HTTP OpenAI + Anthropic tool calls checked end to end.

### Paste-able resume prompt
    Continue the DeepSeek-V4.1-Flash Strata engine work for Mal in ~/AI/Strata-DS4 (branch deepseek4; GitHub remote `ds4`
    = Indras-Mirror/Strata-DSV4.1, pushing progress is OK'd). Read docs/ds4/V41_RESUME_PROMPT.md section "HANDOFF
    2026-10-11 NIGHT" fully first. State: prefill 219-227 tok/s at 8K / 288 at 16K (exact), decode 21.1 tok/s with
    the default route-bias 1.0 + file penalty 0.6 (Mal's calls; decode-only). Mal's targets: prefill 300-400, decode
    20-30 "without losing too much quality"; soft routing only (every expert reachable), report ppl cost of anything
    lossy. Work inline (no workers unless Mal asks). Rules: GPU runs via `STRATA_NO_LARGEPAGES=1 bash
    tools/ds4/memguard.sh 84 70 -- [env VARS=..] build-ds4-gpu/ds41_generate ...` (env vars INSIDE the scope);
    --engram "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"; verify exactness with --dump-last /
    --dump-logits md5 (8192 code prompt default = 4d229102872a; decode logits vs a saved dump); never pkill -f a
    pattern that matches your own command line (use PIDs); don't rebuild build-ds4-gpu while a script is launching
    it. Next: decode dense half (syncs/readbacks/tiny kernels), then prefill H2D, then the serve wrapper.
    Commit with "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>".

## HANDOFF 2026-10-11 EVENING (inline conductor, workers stopped) - prefill 176/203 tok/s, decode 15.9 (18.8 with file penalty)

Repo renamed: GitHub **Indras-Mirror/Strata-DSV4.1** (local remote `ds4`; push `git push ds4 deepseek4:main`, Mal OK'd).
README is V4.1-first. All commits below are on deepseek4 and pushed (52030f58+).

### Prefill (8192-token code prompt, default flags, every step byte-identical last-row logits md5 4d229102872a)
54 (chunk 2048) -> 102.6 layer-major (474f4ac1) -> 105.9 file-tier prefetch (3122a4c6) -> 133-140 enc-skip (72078b60;
now DEFAULT for one-shot, 598c4937) -> 142.9 one run_chunk piece per layer (3 GB budget, STRATA_CHUNK_BUDGET_GB) ->
149.1 layer prestage (992580ef: next layer's arena experts -> 3 GiB device region during its dense half,
STRATA_PRESTAGE_GIB) -> 153.9 routed sums via run_chunk_dev + pinned host_fn (8a790ec1) -> **176.3** Engram async
(3bd85015). 16K prompt: 203 tok/s (B=8192; B=16384 same speed, needs --prefill-chunk 256 to fit).
Determinism: 8bd712bf (indexer top-k ids sorted) - GPU prompt path byte-identical run to run; S (dense sub-batch) changes
the attention summation order (different but valid md5), B does not.
Split now (8192): dense ~19 s, experts ~27 s. nsys (4096): pinned expert H2D median 13 GB/s (p90 23.5), copy engine busy
only ~40% of the expert phase; pageable per-prompt H2D ~1.2 s/4K (the constant-size pageable copies are the one-time
dense weight load). Expert kernels ~10 s GPU time per 4096 tokens (chunk prof undercounts them).
STRATA_PROM=1: +6% speed but ppl 2.3799 -> 2.3946 (+0.6%, deterministic) -> stays OFF.
Ideas not done: prestage file-tier experts too (slab -> device) when VRAM allows; pinned/async for remaining pageable
uploads; dense half profile (19 s / 8192).

### Decode (256-tok prompt p256.i32, 128 tokens, --pf-b 1.43 --pcie 0.75 --route-bias 1.0)
13.2 at start of evening (regression found: Engram decode reads 11 ms/token: reader kept <= 2*nthreads groups on the
caller -> serial with 32 threads; Engram table on exFAT NVME1TB = ~13K random reads/s vs ~290K/s on ntfs3 SSD NVME)
-> b126e279 parallel small batches + 2M-row raw cache (STRATA_ENGRAM_CACHE_ROWS) -> **15.9**, logits identical.
16b05583 r1 CPU-pool batch merged (bitwise, CPU expert ms 2.7 s -> 0.84 s per 128 tok).
bee81052 `--route-bias-file X`: selection-only penalty on file-tier experts: 0.3 -> 16.47, **0.6 -> 18.81 tok/s**.
QUALITY RUN in flight at hand-off time: quality_rb (decode-mode ppl, code + agentic_long 1024 tok) for base / rb1.0 /
rb1.0+file0.6 / rb1.0+file1.0 / rb2.0 / prune65 -> logs /home/mal/AI/v41-dev/qual-HHMM.  Pick the default from it
(Mal: soft only, report ppl cost).
Dead end: --route-bias-arena (bonus for RAM-arena experts) made decode slower (pulls selection off VRAM) - removed.
Next decode levers: Engram table to a fast FS (~200 GB needed; ask Mal - would take Engram 4.8 ms -> <1 ms/token);
file-tier predicted prefetch (E3) for the remaining ~8 ms/token of file waits; dense half ~28 ms/token vs ~15 on the
V4-0731 engine (nsys the decode dense half: per-layer syncs, small kernels); MTP drafting.

### Process
Workers (lm-prefill, engram-io, xk-kernels, pf-prefetch, enc-skip, gpu-determinism, pcie-h2d, engram-wire, h2d-2) are
all done/killed; their branches stay local (v41-*). Mal asked the conductor to do the work inline (2026-10-11 ~16:40).
GPU runs: always `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 84 70 --`; env vars must go INSIDE (`-- env X=1 cmd`).

## HANDOFF 2026-10-11 13:10 (ENGINE SPEED) - GPU engine works; decode 3.4 -> 15.5 tok/s; prefill work in flight

### 0. One-paragraph state
The GSQ file is DONE (`/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf`, 152 GiB, 40 layers; head gate eval ppl 5.645 vs teacher
4.915 = +14.9%, top-1 0.674 vs 0.701; agent 2.118 vs 2.121, reason 6.19 vs 5.83, zh 7.64 vs 6.83, code 1.98 vs 1.74, agentic_long
12.78 vs 9.79). The engine (tools/v41/engine) now runs correctly on the 4090: 2048-token code ppl 2.51 / top-1 0.816 (GPU),
GPU-experts vs CPU-experts logits cos 0.996 mean, 62/64 top-1 equal. Decode went 3.4 -> **15.5 tok/s steady (128 tokens)** with
`--route-bias 1.0` (selection-only bonus for VRAM-resident experts; nothing pruned) and 16.7 tok/s (64 tok) with soft prune 65%
pen 1.0 + route-bias 0.5. Prefill is **~27 tok/s at chunk 512, 54.5 tok/s at chunk 2048** and is the current focus (Mal: target
300-400 tok/s prefill, then decode 20+). Mal is conducting via /conductor with flash workers; keep at most 2-3 workers alive at
once (Mal, 13:05). Nothing pushed.

### 1. Commits this session (branch deepseek4, all local)
- `baa59315` v41_quant 4x faster GSQ pass (CUDA-graph GPTQ `v41_gptq_graph.py` + threaded encode), bit-identical (L17 re-run
  vs GGUF: 1152 matrices identical, h/L18 identical; `compare_layer_bytes.py`).
- `02ef47de` v41_calib --gc also deletes shared/L(L-1).pt (3 GB/layer from L21 filled Crucial1TB);
  `92d9dfdb` chain retries immediately (5 s, not 2 min / 10 min - Mal).
- `2ff530bb` **ds41_moe chunk path race fix** (the NaN): non-last class passes returned without syncing s_cp, the next pass's
  reader threads overwrote pinned bounce slots still being DMA'd -> garbage/NaN prompt rows whenever >1 class pass AND
  file-tier experts. `test_ds41_moe --chunk N` added (run_chunk rows vs reference; FAIL before, PASS after).
- `8e21392a` deferred VRAM cache (one-shot: arena at load, prompt runs, `Ds41Dense::release_pass` + release_chunk, then the VRAM
  cache is sized from measured free memory and seeded: 892 -> 1563 experts at margin 1 GiB); `Ds41Dense::set_route_bias`
  (per-layer F32 rbias added to the top-k SELECTION score only; weights from unbiased scores; zero = model routing);
  `ds41_generate --prune F --prune-penalty X --route-bias X`; `v41_prune_list.py RANK FRAC OUT`.
- `c3a8c9fe` ds41_generate prints "prompt split: dense / experts (file wait) / logits".
- Uncommitted elsewhere: none in the main tree (bench/ probe dirs untracked as before).

### 2. Measurements (all on this box; logs under /home/mal/AI/v41-dev/)
Load: 49 s with `STRATA_NO_LARGEPAGES=1` (4 KiB pages pin at 1.18 GB/s vs THP madvise 0.16 GB/s; was 211 s). ALWAYS set it.
Decode (256-tok code prompt, 64 decoded, `--pf-b 1.43`, arena 60, vram-lru, margin 1; sweep logs sweep*-HHMM):
| config | tok/s | VRAM hits | prompt ppl* |
|---|---|---|---|
| base | 3.38 | 48% | 3.49 |
| threads 16 | 2.16 (worse: SMT) | | |
| pf-b 1.43 | 3.59 | 60% | |
| + pcie 0.75 | 3.81 | 61% | |
| prune 55% pen 0.5 | 8.54 | 74% | 3.48 |
| prune 55% hard (mask) | 11.43 | | 4.11 (+18%: REJECTED, Mal wants soft only) |
| prune 65% pen 1.0 | 9.84 | 73% | 3.62 |
| p65 pen1.0 + route-bias 0.5 | **16.67** | 95% | 3.60 |
| route-bias 1.0 only (no prune) | 14.72 (64 tok), **15.50 (128 tok, steady)** | 96% | 3.50 |
| p65 pen1.0 + rb 1.0 | 14.37 | 98% | |
*IMPORTANT: the "prompt ppl" column is computed on the PROMPT, where route bias is NOT active (it is applied after the deferred
VRAM seed) - so it measures only the prune penalty. The true decode-mode quality of route-bias needs `--prefill-chunk 1 --ppl`
(token-by-token, bias active). That check (`/home/mal/AI/v41-dev/quality_rb.sh`) was started and STOPPED after one result
(base code 1024 tok: ppl 3.4026 top-1 0.748, 2.7 tok/s) to give the GPU to prefill. RE-RUN IT (~25 min) before choosing a default.
Decode tier profile at rb 1.0 (128 tok): 0.73 ms/layer call (cpu pool 0.52 incl. file waits, layer sync 0.10); ~29 ms/token tier,
~35 ms dense; file waits ~14 ms/token.
R1 (one CPU-pool batch per layer, branch `v41-r1-pool-batch` @ `5253ec22`, worktree ~/AI/Strata-DS4-r1): bitwise identical
(DS41_CHECK_POOL_BATCH=1: 0 mismatches; test_ds41_moe_gpu --chunk 64 PASS), CPU expert ms halved (1925 -> 892) but decode
15.36 vs 15.50 (neutral: it drops the arena-vs-file-read overlap and file waits dominate at 96% hits). PARKED, not merged;
re-test after file-tier prefetch hides the SSD waits.
Prefill (4096-token code prompt, no decode; logs pf-1217): chunk 512 = 27.0 tok/s (8 chunks x 19 s), 1024 = 40.3, 2048 = 54.5,
2048+chunk-mmq = 43.8 (MMQ slower). Split at 2048 (MIMO_CHUNK_PROF=1): 85 s = dense 25.1 s + experts 59.2 s of which **file-tier
SSD wait 52.9 s (137.5 GB at 2.58 GB/s)**; GPU grouped expert kernels 0.26 s. 1024 tok chunk 1024: dense 5.5 s WITH --engram,
**1.4 s without** -> Engram row reads ~4.1 s (~12K reads/s; 49K reads). nsys (nsys-pf/pf1k.nsys-rep, arena 8): expert kernels
native_gu<143> 1.09 s, native_gu<11> 0.61 s, native_down<143> 0.38 s, <11> 0.23 s, Q2_0 multi 0.36 s => ~2.3 s per 1024 tokens
(decode-style row kernels on 1024-row groups; caps prefill ~450 tok/s later); cudaMemcpyAsync 5.1 s/16K calls (pageable small
uploads block), cudaMallocHost 3.4 s per prompt (pinned bounce halves re-allocated every prompt).
SSD: nvme0 "SSD NVME" does 4.25-4.5 GB/s with 8-16 parallel 2.6 MiB preads, 1.4 GB/s single-thread, 1.67 GB/s sequential 16 MiB.

### 3. Workers (conductor session 636553.13852289; relay inbox `conductor-636553.13852289`; packets/reports in ~/.quetza-data/conductor)
IN FLIGHT at 13:10 (adopt; re-arm `qc-await <slice>` each as its own background Bash):
- `lm-prefill` (flash, ~32 min in): layer-major prompt batches (`--prompt-batch B`, default 4096-8192; experts over the whole batch,
  dense in sub-batches of `--prefill-chunk`); worktree ~/AI/Strata-DS4-lm branch `v41-lm-prefill` (dense/generate modified,
  build-lm configured). Packet `packets/lm-prefill-task.md`. It relays "GPU?" before GPU runs - answer "go" only when the GPU is free.
  Told (relay): file reads dominate, size the batch from free VRAM (~8 GB idle during prefill).
- `engram-io` (flash, ~6 min in): fast batched/deduped/high-QD + async Engram reads, module-only (ds41_engram + a bench tool);
  worktree ~/AI/Strata-DS4-eng branch `v41-engram-io`. Packet `packets/engram-io-task.md`. CPU+disk only.
DONE (reports read, verified): `moe-offload-survey`, `ds4-src` (antirez ds4 + Chida82 fork: see reports - top items decayed
route-hotness eviction, prefill next-layer prefetch, file-tier prefetch from the GPU predictor, fewer readback syncs),
`glm-vs-ds41` (per-class pool/launch overhead + residency explain GLM 25 vs us; recs R1-R5), `r1-pool-batch` (parked, above).
`ds4-study` was a broken-packet run (ignore its report).
QUEUED (packets written, NOT dispatched - dispatch one at a time as slots free; max 2-3 alive):
- `enc-skip` (packets/enc-skip-task.md): exact encoder-only prefill - decoder layers 20-39 only on the last 2560 prompt tokens
  (proof in the packet: kv_source 20 / index-key ownership; index sources 24-36 only make queries; decoder layers interact only
  through the 128 window). Roughly halves dense + SSD reads for long prompts. Must compose with lm-prefill (small standalone
  dense methods). Best dispatched AFTER lm-prefill lands (same files).
- `pf-prefetch` (packets/pf-prefetch-task.md): prompt path reads layer l+1's file-tier experts while layer l computes (ds4 recipe)
  + keep pinned prompt buffers across prompts. ds41_moe run_chunk only - can run beside lm-prefill.
- `xk-kernels` (packets/xk-kernels-task.md): prompt-batch CUDA kernels for PTQ1_0/Q2_0 (and why Q3_K MMQ was slower).
Dispatch with: `qc-dispatch <slice> --model flash --prompt "Read the full task packet at ~/.quetza-data/conductor/packets/<slice>-task.md and follow it exactly (...report/done/relay...)."`
then `qc-await <slice>` in background. **FOOTGUN: `qc-dispatch --prompt` OVERWRITES packets/<slice>.md with the prompt text** -
keep packet bodies in `packets/<slice>-task.md` (a different name) as done here. Workers from this session show
"(prev: 636553.13852289)" in qc-status; `qc-status --close-done` skips them -> use `--close-done --all-sessions` (closes only DONE).
qc-await FROZEN-TOKENS fires during long builds - check `tmux capture-pane -pt qc-<slice>` before acting; re-arm.

### 4. Next steps (in order)
1. Land lm-prefill: verify diff, run its GPU gate (old chunk path vs --prompt-batch: last-row logits cos >= 0.9999, all-row top-1),
   then measure 4096/8192-token prompts at batch 4096/8192 with the prompt split. Expected: file reads once per batch -> ~150-400 tok/s.
2. Land engram-io, then wire its async API into Ds41Dense::begin (start at begin, wait at layers 1/14) - dense 5.5 -> ~1.4 s/1K tok.
3. Dispatch pf-prefetch (beside lm-prefill), then enc-skip (after lm-prefill), then xk-kernels.
4. Re-run quality_rb.sh (decode-mode ppl for route bias 1.0 / 2.0 / prune+rb) -> pick the decode default (Mal: soft only, every
   expert reachable). Then decode work: fewer per-layer syncs (11/layer; GPU-side weighted sum; fuse predict into finish), decayed
   route-hotness eviction (ds4-src E1), file-tier prefetch from dense.predict (E3), re-test R1.
5. Serving: `strata-ds41-quetza` wrapper (tools/v41/serve) once prefill/decode settle; the serve path must also get deferred cache /
   route bias / prompt batches (currently one-shot only: `defer_vram = !serve`).

### 5. Commands
Bench prompt (4096 code tokens, prompt split): `bash /home/mal/AI/v41-dev/prefill_sweep.sh` (edit configs). Decode sweeps:
`/home/mal/AI/v41-dev/decode_sweep*.sh` (256-tok prompt + 64 decode, one config per line). Quality (decode-mode, bias active):
`/home/mal/AI/v41-dev/quality_rb.sh`. Best current decode command:
`STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 84 70 -- build-ds4-gpu/ds41_generate -m "/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf" --ids-file F -n 128 --backend cuda --experts gpu --engram "/media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8" --prefill-chunk 2048 --arena-gib 60 --vram-lru --vram-margin 1 --rank "/media/mal/SSD NVME/AI/V41-GSQ/rank.bin" --pf-b 1.43 --pcie 0.75 --route-bias 1.0`
GPU tier test incl. prompt path: `build-ds4-gpu/test_ds41_moe_gpu GGUF --layers 0,20,39 --tokens 2 --gpu --chunk 64`.

### 6. Drives / housekeeping done today
Both Engram shards on NVME1TB (verified). Crucial1TB + Team CX2 now kernel ntfs3 (fstab, backup /etc/fstab.bak-ntfs3-20261011-104202);
Crucible stays ntfs-3g (volume dirty; ntfs3 refused rw; options: leave / ntfsfix -d / Windows chkdsk). Temporary sudo rule
/etc/sudoers.d/zz-claude-ntfs3 + /usr/local/sbin/ntfs3-remount: Mal to remove (`sudo rm` both) if not already.
v41-calib/h + shared (89 GB) deleted (not needed for any re-quant: stats/route/tokens/alloc maps kept). syv-qwen + vLLM wrappers deleted.
maptism-hub.service was stopped for the remount and restarted.
Rules learned: never `pkill -f` a pattern matching your own command line (killed my shell again today - use PIDs);
unquoted heredocs execute backticks in packet text (use <<'EOF').


**Last updated:** 2026-10-11 ~07:20 by `strata-ds4-gpu` (Claude).  **READ §HANDOFF 2026-10-11 FIRST** - everything
below it is history (still accurate unless this section says otherwise).

## HANDOFF 2026-10-11 07:20

### 1. State right now
- **GSQ quant pass running** (step 4/4 of `v41allday`, chain pid in /media/Crucial1TB/v41-calib/allday.pid): layers 0-14
  written into `/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf` (152 GiB, single file, kernel ntfs3); layer 15 started
  07:07; ~12.5 min/layer -> **ETA ~12:20**, then its own head gate on the quantized chain
  (`/media/Crucial1TB/v41-quant-mixed/stats/head_L40.json`).  Check: `v41allday status`, `tail
  /media/Crucial1TB/v41-quant-mixed/quant.log`.  Pause: `v41allday stop` (after the current layer); resume
  `v41allday start` (resumes at the first unfinished layer).  The model is unusable until all 40 layers are written
  (unwritten layers are zero experts).
- **Calibration DONE** 01:29.  Teacher head gate (`/media/Crucial1TB/v41-calib/stats/head_L40.json`, held-out
  49,146 tokens): **ALL/eval ppl 4.915, top-1 0.701**; per domain eval: code 1.739, agent 2.121, reason 5.832,
  zh 6.831, agentic_long 9.793.  This is the bar for the GSQ file.
- **Tier map** `/media/Crucial1TB/v41-calib/alloc_full.json` (v41_alloc --per-proj): budget = all-Q2_0 bytes (142.4 GiB
  experts), proxy error 1.882 vs all-Q2_0 2.269 (-17%).  Early layers get Q3_K (L4-8 ~2.45-2.52 bpw), L33-39 ~2.0.
  9 (gu,down) format classes model-wide, up to 7-8 per layer.
- **Night fixes (all committed, all in the running chain):** L7 OOM (1.88 GiB fp32 torch.cat of the propagated output
  -> chunked bf16, `bf23f296`); L13 crash in `q3k_split` (all-zero super-block whose fp16 d did not round-trip ->
  search/re-round + log, `2989e535`); L14 slow (Engram table on Team CX2: slow drive + ntfs-3g FUSE, 1h40).
- **Engram shards copied to NVMe** (Mal OK'd 2026-10-11): `/media/mal/NVME1TB/Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/`
  shard 48 (L14) DONE + verified (size + 302 sampled MiB) + `.ok` + `model.safetensors.index.json`; **shard 47 (L1) DONE 07:18 +
  verified the same way (302 sampled MiB, 0 differ) + `.ok`** (copy log `/home/mal/AI/v41-dev/copy47.log`).  NVME1TB now 58 GB free.  Engine config
  (`tools/v41/serve/strata-ds41.json`), `gpu_bringup.sh` and `v41_calib.DIRS` list NVME1TB FIRST.
- **PLAN for the running chain (agreed with Mal 07:15): do NOT stop/restart it for the Engram move.**  Only layers 1 and
  14 read Engram and both are already written; layers 15-39 and the final head gate never read it, so a restart would
  only redo a layer.  The NVMe Engram copies are for the ENGINE (decode reads 48 rows/token).  Let the chain finish
  (~12:20), then: (a) finish/verify the shard 47 copy, (b) Mal remounts the ntfs-3g drives with ntfs3 (commands below),
  (c) GPU bring-up.  Only if a layer must be RE-quantized later (e.g. a re-export) does the order matter: the calib/quant
  code already reads Engram from NVME1TB first (v41_calib.DIRS).
- **Drives:** Crucial1TB, Crucible, Team CX2 are `fuseblk` (ntfs-3g, slow random reads).  Suggested to Mal (not done,
  needs sudo): Crucible/Team CX2 -> `ntfs3` read-only (zero risk), Crucial1TB -> ntfs3 rw after the chain (or keep).
  Never `force`-mount a dirty volume.  With both Engram shards on NVME1TB this matters less for the engine.
- **Housekeeping done 07:05:** deleted `/media/mal/NVME1TB/syv-qwen` (vLLM Qwen3.8 repo + 23 GB models) and the wrappers
  `qwen3.8-quetza-vllm`, `qwen3.8-swift-quetza-vllm` (+ .bak) - Mal: moved to NInfer.  Gateway profiles untouched.

- **08:14 SPEED-UP (Mal asked; commit `baa59315`):** chain stopped after L17 (L18 1 min in, discarded), `v41_quant.py`
  now uses `v41_gptq_graph.gptq` (CUDA-graph inner loop, bit-identical, 8x) + threaded encode/pwrite.  Verified: L17
  re-run into a qdir == the GGUF bytes of the old code (all 1152 matrices) + identical h/L18 (`compare_layer_bytes.py`;
  NB C.D = 5120, not 4096).  Layer ~830 s -> ~203 s.  Restarted 08:14 at L18 -> **ETA ~09:30-09:45** incl. head gate.
  Old file: /home/mal/AI/v41-dev/v41_quant.py.pre-graph.  Test dir to delete: /media/mal/SSD NVME/AI/v41-test-L17.
- **GSQ PASS DONE 10:20** (all 40 layers rc=0; L22 + L26 needed the OOM retry at V41_ROWS=8192).  Head gate,
  quantized chain vs teacher (held-out eval): **ALL 4.915 -> 5.645 (+14.9%), top-1 0.701 -> 0.674**; agent 2.121 ->
  2.118 (=), reason 5.832 -> 6.190 (+6%), zh 6.831 -> 7.639 (+12%), code 1.739 -> 1.978 (+14%), agentic_long 9.793 ->
  12.781 (+31%).  Fit split only +4.1% (fit/eval gap: few eval sequences, agentic_long/code eval most hurt).  No
  all-Q2 control was run, so the mixed map's gain over plain Q2 is not measured.  Test dir deleted.  Crucial1TB:
  v41-calib/h (20G) + shared (69G) are leftovers, deletable on Mal's OK.  NEXT: engine GPU bring-up (§4 step 2).
- **Reference found (r/LocalLLaMA, Chida82, github.com/Chida82/sf-ds4-1flash):** stock antirez ds4 already runs V4.1
  Flash Q2 (152 GB + 189 GB Engram) with SSD streaming -> usable as a TOKEN-PARITY reference for our engine.  Their
  M5 Max numbers: decode 12 -> 25 tok/s, prefill 404 -> 636 tok/s; per-token profile ~50 ms, SSD expert reads ~4 ms.
  Their wins (each bit-exact): no per-layer GPU sync in decode, expert reads over a thread pool, a few kernel fusions,
  prefill prefetches layer L+1's experts while L computes, prefill reads split across two SSDs holding byte-identical
  copies (verified at start), KV cache on the other drive.  Their "rejected ideas" table with numbers: copy the habit.

### 2. The engine (tools/v41/engine/, all committed; GPU NEVER RUN YET)
Build trees: `build-ds4` (CPU, ggml from FetchContent pin) and `build-ds4-gpu` (CUDA, STRATA_GGML_DIR=third_party/llama.cpp
- gitignored, NOT tracked: never add ops there; ds41_ops.hpp builds the rounding from stock ops for that reason).
CMake: `tools/v41/engine/v41_engine.cmake` (included at the end of tools/ds4/cmake/fork.cmake).
- `ds41_geometry.hpp` - keys + full tensor map + per-expert tier maps; full-length GGUF arrays (strata::GgufFile keeps
  64 items).  `test_ds41_geometry FILE` -> PASS on the real file.
- `ds41_dense.{hpp,cpp}` - dense half.  Pass = n tokens at pos0 (decode n=1 cached graphs w/ CUDA-graph reuse; prefill
  n=chunk built per pass).  Single-pass HC (pre_state carries the next attention's pre-mix); window ring [ring|pass];
  compressed KV on kv sources (2,8,14,20: ratio 2 pooled / 1 plain) read by later same-ratio layers; indexer (fused
  ggml_lightning_indexer) on index sources (2,8,14,20,24,28,32,36), top-k reused until the next; L20 candidate blocks
  (pool_2d max, newest block pinned) masking L24+; attention = flash-attn with the QUERY AS BATCH (each query its own
  128 window + top-k rows; pad keys to 256 off-CPU), sinks; Engram rows via Ds41Engram at L1/L14; act-quant rounding;
  KV BF16 (lossless with act-quant); snapshot/restore (rings, ~11 MB) for multi-turn; predict() router hint; read_sel tap.
- `ds41_ops.hpp` - FP8/pow2, FP4/E4M3, FP4/pow2 fake-quant from stock ggml ops (Veltkamp + magic-number RNE).
- `ds41_engram.{hpp,cpp}` - hash from GGUF arrays, parallel preads (fork-join pool) of 256-B rows + 8-B scales.
- `ds41_moe.{hpp,cpp}` - FORK of tools/ds4/ds4_moe with per-expert format CLASSES (Ds41MoeGeom::eclass, cls_gu, cls_d):
  per-class CPU pool runs (Q2_0 classes need ActQ: im.actq), per-class grouped GPU launches, VRAM LRU swaps within
  (layer, class), chunk path = one pass per class + one weighted sum, run_multi token-by-token for per-expert geoms,
  `skip_file` option.  `Ds41FileBlobs` = an expert's blob from the 3 per-type tensors (preads).  `ds41_read_ranked`.
- `ds41_generate.cpp` - CLI (`--ids/--ids-file -n --ppl --dump-logits`), `--serve` (Strata engine line protocol: GEN/
  STOP/QUIT -> READY/RESUME/PP/T/DONE/ERR; snapshot reuse), flags `--backend cuda|cpu --experts gpu|cpu --engram D:D
  --rank r.bin --slots auto|N --vram-margin --arena-gib --vram-lru --arena-adapt --pcie --pf-b --prefill-chunk N
  --chunk-mmq --skip-miss --skip-file --ctx --temp --stop --no-actquant`.
- Tests: `test_ds41_dense` (vs golden), `test_ds41_ops [--cuda]`, `test_ds41_moe[_gpu] FILE --layers .. [--gpu]`,
  `ds41_golden.py` (model.py on the GGUF's weights, `--no-actquant --zero-experts`, env DS41_TEST_TOPK /
  DS41_TEST_CAND_BLOCKS on both sides), `compare_ds41.py`.
- Serving: `tools/v41/serve/serve_ds41.py` (+40 tests), `ds41-engine.sh`, `strata-ds41.json` (port 8150, ctx 524288,
  chunk 1024, arena 64, vram-lru, rank file), `strata-ds41-quetza.wrapper` (NOT installed in ~/.local/bin until the
  engine runs).  Tokenizer: `/media/mal/SSD NVME/AI/V41-GSQ/tokenizer`.  Rank: `/media/mal/SSD NVME/AI/V41-GSQ/rank.bin`
  (v41_rank.py: calibration count per byte).
- Shared-file fork edits (one-liners, documented): gguf_reader.hpp (PTQ1_0 143), native_expert.hpp (kNativeActBytes
  5840, kNativeHBytes 2628, kNativeFFMax 2304), expert.hpp (ActQ kActMax 5120).

### 3. Gates passed (CPU)
- 3-layer file: block outputs cos >= 0.99996 vs model.py (21 tok), >= 0.99983 (1688 tok, top-512 of 844); decode ==
  chunked prefill; act-quant rounding bit-identical to torch (2M values); BF16 KV bit-identical to F32.
- 40-layer dense-only file (`/home/mal/AI/v41-dev/v41-all-h.gguf`, zero experts, 300 tok): real top-k: cos >= 0.996
  every layer, logits mean 0.99995, top-1 295/300; forced top-32/8 blocks: selected sets identical 272/300 at L2, mean
  Jaccard 0.98-0.994 at all index layers (boundary ties from ggml F16 activations); decode == prefill 300/300.
- Tier on the REAL L0 (all 7 classes mixed per token): rel 1.2-1.5e-2 vs dequant+F32 (the Q8 activation rounding) PASS.
- Dropping act-quant moves model.py's own outputs cos 1.6e-3 / 16% routing sets -> kept on (measure ppl on/off on GPU).

### 4. NEXT (in order) - when the GSQ pass is DONE (~12:20) and Mal OKs the GPU
1. Finish/verify the shard 47 copy (above).  Compare the GSQ head gate (v41-quant-mixed/stats/head_L40.json) with the
   teacher's 4.915 - report per domain to Mal plainly.
2. `bash tools/v41/engine/gpu_bringup.sh` (RANK=/media/mal/SSD\ NVME/AI/V41-GSQ/rank.bin optional): CUDA rounding
   bit-exact -> tier on real L0/20/39 with --gpu -> ds41_generate --ppl on 2048 held-out code tokens (chunk 512).
   Expect issues on first GPU contact (graph reuse, FA with ne3 batch, top_k sizes, lightning-indexer shapes): fix with
   the CPU tree as oracle (`--backend cpu`).
3. Engine ppl on the 6 held-out eval sequences (`/home/mal/AI/v41-dev/eval/seq*.i32`, 8192 tok each) vs the GSQ head
   gate: the engine must reproduce it (same weights; engine has Q8_0 dense vs FP8 + Q8 activations - small diff).
   Also act-quant on/off ppl.
4. Speed (Mal's target: beat GLM's 25 tok/s decode / 400 tok/s prefill, 512K ctx).  Routing is flat: rank shows 16 GiB
   of experts = 42% of lookups, 64 GiB = 78%, 80 GiB = 86% -> expect ~8-10 tok/s at first.  Levers, measure each:
   arena size (84 GiB memguard cap), vram-lru, arena-adapt, pf-b prefetch (Ds41Dense::predict), per-class launch
   count (up to ~4 per token-layer: fuse into one launch), merge the 3 readbacks per layer, Engram read time/token,
   prefill chunk 2048-8192 + chunk-mmq (Q3_K only), **soft-prune** = route-bias penalty on the coldest experts
   (GLM's recipe, not yet in ds41: port set_route_bias / --prune-penalty from tools/ds4), skip-file.
5. Size (Mal asked: Maya-size ~90 GiB?).  `/home/mal/AI/v41-dev/size_sweep.py`: prune+mixed proxy error 0.54 at
   89 GiB vs current mix 1.88 / all-Q2 2.17 - BUT pruning cost is only measured on the calibration corpus (OOD
   risk).  Recommendation given: keep the 152 file, soft-prune at runtime, decide on a smaller re-export after real
   ppl/tok/s (new tier map = minutes on CPU; a new GSQ pass = ~8 h GPU).
6. Serving: install `strata-ds41-quetza` (copy wrapper to ~/.local/bin) once `--serve` works; test via serve_ds41
   (OpenAI + Anthropic, tool calls), coherence, agentic tasks, loop test, DwarfStar's official V4.1 fixtures.
7. Later: DSpark/MTP (mtp.* not exported yet), 512K long-context run, upstream of the Engram/indexer work.

### 5. Rules (Mal)
Mal decides GPU use (strata-glm shares it: lock ~/.quetza-data/conductor/ds4-gpu.lock).  Never push without Mal's OK
(nothing pushed).  Commit each verified step with the Co-Authored-By trailer.  Never `pkill -f` a pattern that can match
your own command line (it killed this session's shell once): use PIDs.  Never edit a running script in place
(.new + mv).  CPU work nice 15; big RAM runs via tools/ds4/memguard.sh (it takes the GPU lock - not while the chain
runs).  Keep HF shards 47/48 on Crucible/Team CX2 as backups.  NSFW is headless-only (CLAUDE.md).  Explain plainly.


**Last updated:** 2026-10-10 ~09:05 by `strata-ds4-gpu`.

**2026-10-10 NIGHT (engine port, by strata-ds4-gpu; Mal asleep, chain running): READ §Engine (night of 10-10) FIRST.**
Chain: calibration resumed 22:28 at layer 25 (`v41allday start`), then tier map -> layout -> GSQ pass (~07:00 est).
Engine (tools/v41/engine/, all committed): geometry/loader, Ds41Dense (CPU-gated vs model.py), act-quant ops,
Ds41MoeTier (per-expert format classes), Ds41FileBlobs, ds41_generate (CLI/--ppl/--serve).  GPU runs NOT done yet
(GPU owned by the chain).  Next: §Engine next steps.

**09:04 UPDATE (supersedes 09:01):** the WHOLE chain runs unattended: `v41allday` (tools/v41/all_day.sh + alldayctl.sh,
start|status|stop|log): calibration 7-39 + head -> v41_alloc tier map -> GGUF layout on SSD NVME/AI/V41-GSQ -> GSQ quant pass
(mixed, in place) + its head gate.  All-Q2_0 control SKIPPED (Mal).  Resumable: rerun `v41allday start` after a failure/stop.
Log /media/Crucial1TB/v41-calib/allday.log.

(old) **09:01:** calibration is ARMED, not stopped: `v41calib start-when-idle` (driver waits until the GPU lock is free +
VRAM <3 GB for 15 min straight, i.e. strata-glm finished, then runs layers 7-39 + head unattended; CUDA OOM retries a layer
2x). Check `v41calib status` / calib.log before doing anything; do NOT `start` again while it shows RUNNING.
**Read order:** this file → [V41_PLAN.md](V41_PLAN.md) (design, V4→V4.1 deltas, all research) →
[V41_FEASIBILITY.md](V41_FEASIBILITY.md) (sizes, speed estimates) → [RESEARCH_2026-10-09.md](RESEARCH_2026-10-09.md)
(community ideas) → `../../V41_PLAN`-adjacent [../../RESEARCH](../RESEARCH_2026-10-09.md).
**Repo:** `~/AI/Strata-DS4`, branch `deepseek4`. Local only — ahead of the public `ds4/main` at `924516a`.
**Since 2026-10-10 `deepseek4` contains upstream Strata (merge `cc55d4c9`, upstream `fb58e0db`)**; pulling upstream is a
plain `git merge upstream/main` — read [../UPSTREAM_SYNC.md](../UPSTREAM_SYNC.md) (process, gates, fork-isolation rules)
before touching any file under `src/` or `include/`.
**Nothing from this work is pushed;** push only when Mal says.

---

## Paste-able resume prompt

> You are continuing the **DeepSeek-V4.1-Flash GSQ-RCO mixed-tier quant + Strata-DS4 port** for Mal, in
> `~/AI/Strata-DS4` (branch `deepseek4`; second worktree `~/AI/Strata-DS4-up`, branch `ds4-upstream`).
> Read, in order: `docs/ds4/V41_RESUME_PROMPT.md` (this file, ALL of it - especially §In flight and §Session log
> 2026-10-10), `docs/UPSTREAM_SYNC.md` (before touching `src/` or `include/`), then `docs/ds4/V41_PLAN.md` as needed.
>
> **0. Ground rules (Mal).** Work solo (no conductor workers). Measure before concluding; commit each verified step
> with the Co-Authored-By trailer; **never push without Mal's explicit OK** (`deepseek4` is public as `ds4`).
> **Mal decides when the GPU is used**: don't start GPU work he hasn't OK'd, and give strata-glm GPU windows when
> asked. Never kill processes by a `pgrep -f` pattern that can match your own command line, and never wait on one
> (`until ! pgrep -f X` never ends - it matches itself): use PIDs, pid files, or `grep "[x]..."`. Never edit a running
> bash script in place (write a copy, `mv` over it). CPU work at `nice -n 15`, 4 threads. Big RAM jobs only under
> `tools/ds4/memguard.sh`. Never a CPU-only full-model load. Explain results plainly to Mal (he reads the summary,
> not the tool output).
>
> **1. Relay.** `relay_rename strata-ds4-gpu`, then `relay_peers`. `strata-glm` (~/AI/Strata-GLM) shares the RTX 4090
> and the shared Strata core; replies to its asks often fail with `peer_gone` - answer with a fresh `relay_ask` instead.
>
> **2. State check (2 min).**
> - Calibration: `tools/v41/calibctl.sh status` (alias `v41calib`, in ~/.bashrc). Resume with `v41calib start`
>   only if Mal OKs the GPU. When all 40 layers are done, run `v41calib start` once more: it skips every layer and runs
>   the `head` gate (the driver running on 2026-10-10 was started before the head step existed).
> - Download: count `*.ok` (`ls /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/*.ok | grep -c model-`,
>   plus 47 on Crucible and 48 on Team CX2 = 48 total). `tail .../sequence2.log`. Re-run
>   `setsid nohup bash tools/v41/fetch_sequence2.sh &` if no `fetch_v41`/`curl` is running and shards are missing.
> - Worktree `~/AI/Strata-DS4-up`: `git status` - the PTQ1_0 GPU work may still be uncommitted there (§In flight).
>
> **2b. FIRST: the calibration is STOPPED (7/40 layers) - the GPU was handed to strata-glm at Mal's request.**
> Restart with `v41calib start` ONLY when strata-glm relays "GPU free" or Mal says so (it resumes at layer 7;
> layer 7's first try OOM'd on a hot expert, fixed in this commit by chunking the expert forward - watch layer 7
> finish: `v41calib status`, `tail /media/Crucial1TB/v41-calib/logs/L07.log`).
>
> **3. Continue** with §Next steps.  Built 2026-10-10: PTQ1_0 GPU (all 9 tier pairs in the grouped kernel), GGUF
> writer, quantize pass + driver (§Export pipeline).  When calibration layers 0-39 + head are done: `v41_alloc.py
> --layers 0-39 --per-proj --out /media/Crucial1TB/v41-calib/alloc_full.json`, report the tier map to Mal in plain
> terms, then run the export (§Export pipeline, with Mal's GPU OK).  Meanwhile: the V4.1 engine port (§Next steps 5).

---

## Status snapshot (2026-10-10 ~09:00)

| item | state |
|---|---|
| Download | **complete**: 48/48 verified (1-46 on Crucial1TB, 47 Crucible, 48 Team CX2) |
| Calibration | **STOPPED at 7/40 for strata-glm** (Mal 09:00); layer 7 OOM'd 08:55 (hot expert, 88K tokens at L6) - fixed (chunked expert forward); ~6.5 h of GPU left. `v41calib start` when the GPU is handed back |
| Calibration data | agentic-first corpus, 32 x 8192 tokens (§Calibration pipeline); old Pile run archived in `v41-calib/old-pile2048/` (biased stats: damp 0.01) |
| GPTQ damping | **1.0** (layer-8 sweep; 0.01 overfit) |
| Layer-0 tier map (new corpus) | per-proj: 383 ptq1 / 140 q2_0 / 245 q3_k slots @ 2.25 bpw, proxy -39% vs all-Q2_0; per-expert 190/114/80, -24%; down proj = 68% of an expert's Q2 error; 3 rarely-routed experts floored at q2_0 |
| Engine | `deepseek4` = upstream Strata `fb58e0db` merged (`cc55d4c9`) + fork isolation (`20618210`): p600 ppl 10.47, decode 19.55 tok/s (pre-merge 16.8-17.3) |
| `head` gate | built (`5d99832d`), wired into `calib_layers.sh` after layer 39; CPU smoke passed (`e567a476`) |
| GGUF writer | **built + tested** (`70f81b03`, `d020b910`): `v41_gguf.py`, in-place expert mode, independent checker |
| Quantize pass | **built, CPU-smoked** (`d020b910`): `v41_quant.py` + `quant_layers.sh`; not run on the GPU yet |
| PTQ1_0 ternary | CPU done (`50e9428d`); GPU grouped-expert kernels done + tested (`4a403b15`); all 9 tier pairs through `native_expert_grouped` PASS (`cc26cb5d`) |
| GSQ refinement / V4.1 engine port | not started (GSQ: optional, plugs in where v41_quant calls GPTQ) |
| GitHub | nothing pushed today; `Indras-Mirror/Strata-MiMo` **deleted** at Mal's request (local `~/AI/Strata-MiMo` kept, 10 unpushed commits) |

---

## In flight (2026-10-10 ~09:00) - check each of these first

1. **Calibration driver** `calib_layers.sh 0 39` (pid in `/media/Crucial1TB/v41-calib/driver.pid`), detached via
   `calibctl`. Logs: `v41-calib/calib.log`, `v41-calib/logs/LNN.log`. It was started from the copy of
   `calib_layers.sh` BEFORE the head step was added, so after layer 39 run `v41calib start` again (head only).
2. **PTQ1_0 GPU: DONE + PASSED (`4a403b15`, on `deepseek4`).** `ptq1_0_cuda_test`: dequant f16 bit-exact
   (16384/16384); `iq_mmvq(143)` worst 9.9e-8 relative vs the exact reference (the first run's ~2e-4 "fail" was the
   test's fp32 activation scale; the kernel reads fp16).  Pieces: `strata_block_ptq1_0` + `vec_dot_ptq1_0_q8_1`
   (fork_vecdot.cuh, llama-tq @52a37aec MIT), `Fmt<143>` in the FORK lists (fork_fmt.cuh), `dq_ptq1_0` (fork_dq.cuh),
   3 hooks in iq_kernels.cu.  Rerun: `flock ~/.quetza-data/conductor/ds4-gpu.lock ~/AI/Strata-DS4-up/build-gpu/ptq1_0_cuda_test`.
   **Still open:** `native_expert_grouped` with a 143 layout (compiles, untested); a 4-part (ipb 4) dot for the
   2304-wide down rows (ipb 1 leaves lanes idle: 18 blocks per row vs 32 lanes).
   - Reference sources kept in `docs/ds4/calib-refs/llama-tq/` (`mmvq-ptq1_0.cuh` planar kernel, `vecdotq.cuh`).
3. **`head` CPU smoke: PASSED 2026-10-10 08:08** (4 x 256 Pile workdir, layer 0 on CPU, `head --layer 1`; ppl ~1e7 is
   expected on layer 1's output - code path only).  The first try died on `assert N % B == 0` (default batch 16 vs
   N=4); fixed in `e567a476` (batch default = largest divisor of N <= 4096/S; the real 32 x 8192 run is unchanged).
   Redo: `V41_WORK=<tmp> HF_DATASETS_OFFLINE=1 python3 tools/v41/v41_calib.py prep --n 4 --seq 256 && ... init &&
   OMP_NUM_THREADS=4 ... layer 0 --device cpu --gptq-experts 0 && ... head --layer 1 --device cpu`.
4. **Download: complete** (48/48 verified 2026-10-10 ~08:20).  **Quant-pass CPU smoke** (scratch, gone after a
   reboot): `v41_quant.py` layers 0 (q2_0) and 1 (ptq1, Engram layer) + head on a 4x256 workdir; layer 0 passed
   (in-place fill + checker PASS).  Redo recipe = the commit message of `d020b910` + §Export pipeline.
5. **strata-glm** is merging upstream into GLM (branch `glm-upstream`, worktree `~/AI/Strata-GLM-um`), taking DS4's
   shared-file resolution (`cc55d4c9`/`20618210`) and moving its LoRA kernels into `src/kernels/cuda/fork/`.  GLM keeps
   its own `native_expert.cpp` (no PTQ1_0 - Mal's call), so that one shared file will differ.

---

## Session log 2026-10-10 (what changed and why)

- **Afternoon session (08:05-09:05), all committed:** PTQ1_0 GPU test passed (the earlier "fail" was the test's fp32
  scale); grouped-kernel test over all 9 tier pairs; `v41_formats.py` (bit-exact encoders); GGUF writer (DwarfStar
  names, per-expert tiers, split Engram shard); quantize pass + driver; writer `--experts-in-place` (no QDIR copy:
  peak disk = the final file); independent checker.  Mal's decisions: **no Engram copy** (engine reads it from HF
  shards 47/48 - keep them), the control pass writes no weights, final file on SSD NVME (163 GiB free; 152 GiB
  needed), serve V4.1 exactly like `~/.local/bin/strata-glm-quetza` (§Serving).  **Bug found + fixed: tokenizer
  types** - 54 added-but-not-special tokens (<｜System｜>, <｜User｜>, <｜Assistant｜>, ...) must be type 4
  (USER_DEFINED), not 1, or they tokenize as text (389/394 texts wrong -> 0/394 after; also fixed in the GGUF meta).
  Mal asked (answered): RCO/GSQ from the native-FP4 experts IS viable (FP4 is the trained model; 4.25 -> 2.25 bpw;
  GPTQ beats RTN 0.58/0.70/0.74; the head gate on the quantized chain vs the all-Q2_0 control decides).

- **GPTQ overfit found and fixed.**  Layers 0-7 on the old Pile corpus: from layer 1 on, experts with fewer fit tokens
  than the median were WORSE than RTN on held-out data (q3 ratio up to 1.43): most experts see ~2.5K fit tokens but
  the gate/up Hessian is 5120 wide (rank-deficient), and damp 0.01 was too weak.  Layer-8 sweep (48 experts, damps
  0.01-3): **damp 1.0** beats RTN in every group (GPTQ/RTN 0.58 ptq1 / 0.70 q2 / 0.74 q3), 3.0 is worse.  New flags:
  `--damp a,b,c` (sweep; extra damps stored as `e_<tier>@<damp>`), `--expert-stride K`, `--screen-only TAG`.
- **Calibration corpus rebuilt agentic-first** (Mal: calibrate for what we keep - agentic tool use, long-form,
  coding, English + Chinese).  Researched other V4.1 quants: nobody published a calibrated V4.1 tier map
  (pfeifferj's RCO "kept the initial allocation"; apetersson / pyrodog / INC AutoRound are uncalibrated).  Borrowed:
  antirez/ds4's 4690-prompt imatrix set (V4 DSML -> re-rendered for V4.1) and DeepSeek's official V4.1 encoder.  Mal's
  own session transcripts are the long-agentic slice (Mal OK'd; scrubbed, never leaves the box).  Mal chose 256K
  tokens (not 512K).  smalinin publishes a 262K-token V4.1 imatrix (`imatrix-dsv41-262144.gguf`, 789 MB, not fetched):
  plan to cross-check per-expert coverage against it before the final map ("borrow other calibrations and match").
- **Overnight stall (lesson):** the "wait for layer 8" watcher used `pgrep -f` on a pattern in its own command line
  and never fired; the GPU sat idle 01:51-05:37.  Same bug recurred once for a build wait (caught).  Use log/pid-file
  watches only.
- **8K sequences OOM'd the 24 GB card** at 2 x 8192 per attention batch: `--batch` now defaults to ~4K tokens
  (1 x 8192) and the driver sets `expandable_segments`.
- **Stoppable/resumable run** (Mal will work over several sessions): `calibctl.sh` + STOP file + done = stats exist.
- **Upstream Strata merged** (Mal: keep tracking upstream, isolate fork edits).  Histories were unrelated (upstream
  rewrote its history; our main had the identical tree) -> one-time graft merge; future syncs are `git merge
  upstream/main`.  Then the isolation refactor (fork code into `src/kernels/cuda/fork/*.cuh`, `tools/ds4/cmake/
  {fork,ggml_cuda}.cmake`, `include/strata/artifact/ds4_guard.hpp`): footprint in upstream files +727 -> +327 lines.
  Upstream's Foresight swap / router look-ahead prefetch / prefill CPU share / MMVQ_IL rows table /
  --expert-cache-per-layer live in the Qwen engine path - port them to the DS4 path by hand later (they did NOT come
  in through the merge).
- **SwiGLU clamp stays:** V4.1's own config has `swiglu_limit: 10.0` and DeepSeek's `model.py:845` applies it.
- **FP8/FP4 question (Mal):** the routed experts we quantize are native FP4 (the trained values - no better source
  exists); "FP8" is the dense weights, which stay >= Q8.  Errors are measured against the FP4 teacher.
- **Engine name:** undecided (Mal is thinking: Terra / Firma / ...; suggestions given: Tezca, Xibalba, Fathom, Mantle).
  Strata is MIT - renaming is allowed; keep Niko1221's copyright + license, credit "built on Strata".

---

## Calibration + GPTQ screening pipeline (built 2026-10-09 23:00-23:40)

- `tools/v41/v41_calib.py` — DeepSeek's `model.py` run **one layer at a time** straight from the HF shards.
  **Use DeepSeek's inference `config.json` (`~/AI/gsq-rco/dsv41-inference/config.json`)**: the HF one in the model dir
  is the transformers schema and silently leaves ModelArgs at its TINY defaults (caught; now cross-checked by assert).
  - `tools/v41/build_corpus.py` (CPU, ~1 min) -> `v41-calib/corpus/*.jsonl`: session transcripts (Claude Code /
    Quetza / GLM jsonl; longest prefix <= 8191 tokens; secrets scrubbed value-by-value; tool results clipped) and
    antirez/ds4's 4690 imatrix prompts (V4 DSML parsed back into calls), all rendered by `dsv41_encoding.py`
    (DeepSeek's V4.1 encoder, vendored: DSML tags have a LEADING SPACE in V4.1).  References + their provenance:
    `docs/ds4/calib-refs/README.md`.
  - `prep --n 32 --seq 8192 --corpus v41-calib/corpus` (CPU): packs the agentic-first mix (old Pile mix without
    `--corpus`), BOS per sequence, stratified ~1/4 held out per domain; engram hash ids -> `hashes.npy`.
  - `init` (CPU): embedding (shard 2) -> `h/L00.bin` (10 GiB bf16, [128,2048,4,5120]).
  - `layer L` (GPU, ~20 GB VRAM, ~20-25 GiB RAM): attention half per batch of ~4K tokens (1 x 8192; 2 x 8192 ran
    out of the 24 GB card), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -> gate -> **expert-major**
    MoE (each FP4 expert's reference `Expert.forward` once over all its tokens) -> **per-expert GPTQ** at
    ptq1/q2_0/q3_k (+RTN control) fitted on the 98 fit sequences, **error measured on the 30 held-out ones**:
    `e_<tier>` whole expert, `egu_<tier>` gate+up only, `edn_<tier>` down only, `rtn_<tier>`; plus REAP saliency
    (`reap` = mean ||g f(x)||), counts, `y_energy`.  -> `stats/LNN.json`, `route/LNN.npz`, `h/L(N+1).bin`,
    `pre/`, `shared/` (SharedAttentionRuntime per batch: source-layer compressed KV / index keys / top-k / candidates).
  - Engram: `MemmapEngramEmbedding` replaces the 98 GB table with memmap row reads (MADV_RANDOM).
- `tools/v41/v41_gptq.py` — exact-format grids: PTQ1_0 `{-1,0,1}·d`/128, Q2_0 `{-1,0,1,2}·d`/64 (d fp16, sign
  free), Q3_K `(q-4)·d·(sc-32)`/16 (fp16 d per 256, 6-bit sc).  Static diag(H)-weighted scale search (11 clip ratios x
  both signs), then stacked multi-tier GPTQ (no act-order).  Everything it outputs is exactly representable.
- `tools/v41/calibctl.sh` (alias `v41calib`) drives `calib_layers.sh`: start/stop/stop-now/status/log; a layer is
  done when `stats/LNN.json` exists; `STOP` file checked between layers; first start runs `init`.  `--damp` takes a
  comma list for sweeps (`--screen-only TAG --expert-stride K` writes `stats/LNN_TAG.json` without touching the chain).
- `tools/v41/calib_layers.sh FROM TO` — one memguard'd GPU block per layer behind `ds4-gpu.lock` (strata-glm slots in
  between), shard-gated (layer L = shard L+3; 47 for L=1, 48 for L=14), resumable, `--gc` keeps only 2 h files.
  Log `/media/Crucial1TB/v41-calib/calib.log`, per layer `logs/LNN.log`.  Restart = re-run the same command.
- `tools/v41/v41_alloc.py [--layers 0-7] [--per-proj] [--out map.json]` — tier map: cost = held-out error / layer's
  held-out routed energy (layers 0-1 and 37-39 x2), budget = all-Q2_0 bytes, Lagrangian bisection + greedy fill;
  ternary <= 50%/layer; experts with < 8 held-out tokens floored at q2_0.  Prints per-layer tier counts, bpw,
  GPTQ/RTN error ratios, and the proxy error of the mix vs all-Q2_0.
- Tests: `tests/test_v41_gptq.py` (17/17), `tests/test_calib_equiv.py L` (split forward == reference; needs a smoke
  workdir: `V41_WORK=<dir> v41_calib.py prep --n 4 --seq 256 && init && layer 0 --device cpu --gptq-experts 3`).
- Known limits: the screening uses GPTQ init only (GSQ refinement is the next step, at the chosen tier); the
  allocation proxy treats layers as equally important (validate end-to-end: ppl/KL needs shard 43 = head).

---

## Export pipeline (built 2026-10-10; run after the full tier map)

```
48 HF shards (read only) --quantize pass, layer by layer (GPU)--> experts written IN PLACE into shard 1 of the GGUF
                                                                   (holes laid out first by the writer)
```
1. Tier map: `python3 tools/v41/v41_alloc.py --layers 0-39 --per-proj --out /media/Crucial1TB/v41-calib/alloc_full.json`.
2. Lay out the final file (CPU, ~10 min, ~10 GB written; the rest is holes until filled):
   `python3 tools/v41/v41_gguf.py --out "/media/mal/SSD NVME/<dir>/V41-GSQ.gguf" --rtn-map .../alloc_full.json
   --experts-in-place --no-engram` -> `V41-GSQ.gguf` + `V41-GSQ.gguf.experts.json` (with --no-engram it is one plain
   file, no split).  Size at the all-Q2_0 budget: 151.8 GiB; SSD NVME had 163 GiB free (2026-10-10) - nothing else
   goes there first.
3. Quantize pass (GPU, ~4-5 h estimated, measure): `tools/v41/quant_layers.sh /media/Crucial1TB/v41-quant-mixed 0 39
   -- --map .../alloc_full.json --gguf "<the file>"`.  Per-layer lock (strata-glm slots in), resumable, STOP file
   (`touch <qwork>/STOP`), ends with the head gate on the QUANTIZED chain (stats/head_L40.json in the qwork).
4. All-Q2_0 control: `quant_layers.sh /media/Crucial1TB/v41-quant-q2 0 39 -- --tier q2_0 --no-write` (head
   numbers only, writes no weights).  Mal's rule: the mixed file ships only if its held-out per-domain ppl beats
   this control's (and compare both with the teacher's head_L40.json from the calibration run).  If it loses:
   lay out an all-Q2_0 file (`--rtn q2_0 --experts-in-place`) and rerun step 3 with `--tier q2_0 --gguf`.
5. Check: `python3 tools/v41/tests/check_v41_gguf.py <file> --fp4 8 --layers 0-39` (layout, dense vs source,
   every layer's sampled experts decoded vs FP4 - a hole decodes to rel ~1) and `v41_gguf.py --verify <file>
   --rtn-map ... --experts-in-place`.
6. Engram: **NOT copied** (Mal 2026-10-10: no 203 GB copy).  The engine port reads the Engram rows straight from
   the HF shards 47 (layer 1, on Crucible = USB SATA SSD) and 48 (layer 14, Team CX2): row = 256 B at
   `embed.weight` + r*256, scale = 8 B at `embed.scale` + r*8 (2 reads per row, 96 per token, prefetched at token
   start).  KEEP shards 47/48.  Measure Crucible's latency at layer 1 (early in the token: little time to hide
   it); if slow, move that one table (~94 GiB) to a faster SSD.  `v41_gguf.py` without --no-engram still writes a
   264-B-row Engram shard 2 (DwarfStar's format) if one is ever wanted.

Formats (`v41_formats.py`, bit-exact vs ggml / ptq1_0.cpp): Q8_0 34 B/32, Q2_0 18 B/64, Q3_K 110 B/256, PTQ1_0
28 B/128.  Expert layout in the GGUF: a role whose 384 experts share a type is the usual `blk.L.ffn_<role>_exps.weight`;
mixed -> `blk.L.ffn_<role>_exps.<q2_0|q3_k|ptq1_0>.weight` (experts of that type, id order) + I32
`blk.L.ffn_exps_{gu,down}_type[384]` (slot = rank among the experts of the same type).  Gate+up share a type.
Dense: DwarfStar's names/types (`deepseek41_quantize.py` @fc80bd6): Q8_0 attention / shared / head (rel 5.4e-3 vs
FP8), F16 embedding / compressor / indexer / Engram wkv, F32 norms / router / mHC.  mtp.* (DSpark) not exported yet.

## Serving (Mal 2026-10-10: "we'll be running deepseek like `~/.local/bin/strata-glm-quetza`")

Copy GLM's shape exactly (`~/AI/Strata-GLM/tools/glm/serve/`, its RESUME_PROMPT.md "Serving plumbing" + "Metrics"):
engine `ds41_generate --serve` (Strata's engine line protocol, docs/BATCHING.md §Engine protocol + GEN/GENI) ->
launcher `tools/v41/serve/ds41-engine.sh` (copy glm-engine.sh: ComfyUI refusal, `flock` the shared GPU lock for the
server's life, `systemd-run --scope MemoryMax=84G MemorySwapMax=0`) -> shim `tools/v41/serve/serve_ds41.py` that
imports **`~/AI/Strata/serve`** (that checkout has the llama.cpp-style metrics: `/metrics`, `/slots`,
`?format=prometheus` -> `llamacpp:*` + `strata:*`; commit 9bb510c8 is LOCAL - its remote is upstream Niko1221,
NEVER push it) and runs under `/home/mal/AI/Strata/.venv/bin/python` -> configs `tools/v41/serve/*.json` (NO
`--slots` arg: server.py dies on it) -> wrapper `~/.local/bin/strata-ds41-quetza` (copy strata-glm-quetza: port,
model name, context, QuetzaCodetl env).
**Done:** `tools/v41/serve/export_tokenizer.py` (HF tokenizer.json -> vocab.json / merges.txt / token_type.json):
Strata's BPE with pre `joyai-llm` == HF tokenizers on 394 corpus texts / 465,645 tokens, 0 mismatches.
**Next (designed, not written):** `serve_ds41.py`:
- tokenizer: patch `ST.Tokenizer` to pre="joyai-llm" (as serve_glm.py does for glm4);
- end of turn: `S.IM_END = "<｜end▁of▁sentence｜>"`;
- template: replace `F.ChatTemplate` / `S.ChatTemplate` with a class whose `render(messages, tools,
  add_generation_prompt, **kw)` calls `dsv41_encoding.encode_messages` (pure Python; tools go on the first system
  message as OpenAI-format `{"type":"function","function":...}`, tool results via role "tool" + merge; thinking_mode
  "chat" when `enable_thinking is False` else "thinking"; reasoning_effort low/high/max (map medium->high); the
  generation suffix is `<｜Assistant｜>` + `<think>` (thinking) or `</think>` (chat) - strip it for
  add_generation_prompt=False); set `.caps` by hand (tools/thinking true);
- tool calls: a streaming rewriter in front of `F.OutputParser.feed` that turns DSML (`<｜DSML｜ calls>`,
  `<｜DSML｜ invoke name="X">`, `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`; note the
  LEADING SPACE in the tag names) into Qwen text (`<tool_call>\n<function=X>\n<parameter=K>\nV\n</parameter>\n
  </function>\n</tool_call>`, one per invoke; drop the calls wrapper), holding back a partial `<｜DSML｜` / `</｜DSML｜`
  tag; check how parse_tool_call trims the value newlines and string="false" -> JSON;
- tests (`tools/v41/serve/test_serve_ds41.py`): tokenizer == HF; template == encode_messages on chats with tools /
  tool results / thinking; DSML split at every byte boundary -> the same calls as `parse_message_from_completion_text`;
  end to end with the server's `--engine mock` + a DSML script over HTTP (OpenAI + Anthropic tool calls).
Tokenizer dir for the real config: next to the final GGUF (SSD NVME).

## Goal (Mal's decisions — all made)

1. **Our own quant of DeepSeek-V4.1-Flash, uncensored**, from `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`
   (510.3 GB). The abliteration (MPOA) touches **dense weights only** — routed experts, engram, router, norms are
   byte-identical to the base. Reported: MMLU 86.96 → 82.74, HarmBench 100% at both effort-off and max.
2. **GSQ (IST-DASLab) quantization of the routed experts, mixed tiers, byte-budget-neutral at all-Q2 size**
   (~153 GB of experts, 2.25 bpw average):
   - sensitive experts → **Q3_K** (3.44 bpw)
   - default → **Q2_0** (2.25 bpw = GSQ's native 2-bit grid)
   - weakest → **PTQ1_0 ternary** (1.75 bpw)
   - Every 1% of experts moved to ternary pays for 0.42% moved to Q3_K.
   - Mal: *"don't lobotomize it"* — **all-Q2 is the fallback**; the mixed tier only ships if it beats an all-Q2 control
     on held-out **per-domain** ppl, **including languages never used in calibration** (FINDINGS s37 pattern).
   - Dense / shared expert / router / engram stay **≥ Q8_0**.
3. **Hadamard rotation for the ternary tier** (and measure it on Q2 too): rotate the weight's input dim by a
   block-128 Hadamard with random signs before GSQ (calibration activations rotated the same way); runtime FWHT on the
   activation. Per-tensor flag in the GGUF.
4. **Engram (203 GB = 189 GiB) never in RAM.** Rows are read from SSD — 48 rows / ~12 KB per token — prefetched at
   token start (the hash depends only on token ids, so it can start before the forward pass). Mal will move the engram
   to NVMe after GSQ (needs ~145 GiB freed on NVME1TB; SSD NVME's 163 GiB is reserved for the expert pack).
5. **Run it in Strata-DS4** (port V4.1 into our engine; reuse soft-prune + arena-adapt + VRAM LRU + RAM arena).
   Expected **6–10 tok/s** at first (NVMe-bound: ~55–60% of the pack fits in ~85–90 GB fast memory), then DSpark.

---

## Where things are on disk (free space as of 2026-10-09 22:51)

| what | path | free |
|---|---|---|
| source shards (all but engram) | `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 368 G |
| engram layer 1 (shard 47, 101.5 GB) ✓ | `/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 57 G |
| engram layer 14 (shard 48, 101.5 GB) | `/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` | 152 G |
| planned GSQ output (~151 GiB) | `/media/mal/SSD NVME/…` (NTFS, PCIe4; mount `udisksctl mount -b /dev/nvme0n1p3`) | 163 G |
| engram final home (later) | `/media/mal/NVME1TB/…` (needs ~145 GiB freed) | 49 G |

**Repos / references**
| what | path |
|---|---|
| GSQ + RCO repos | `~/AI/gsq-rco/GSQ`, `~/AI/gsq-rco/RCO` (cloned, not installed; reuse the quantizer modules) |
| DeepSeek V4.1 reference code (MIT) | `~/AI/gsq-rco/dsv41-inference/` (`model.py`, `engram.py`, `kernel.py`=tilelang, `convert.py`, `vision.py`, `config.json`) |
| DwarfStar = antirez/ds4 V4.1 C reference | `~/AI/ds4-ref` @ `fc80bd6`: `ds4_engram.c`, `ds4_deepseek41_cuda.cuh`, `gguf-tools/deepseek41_quantize.py` |
| msuiche GLP control vector (alt uncensor, base only) | `~/AI/Strata-DS4-private/v41/glp.gguf` (39 rank-1 dirs, alpha 0.5) |
| video transcript (Local Explained, V4.1 HW) | `~/AI/Strata-DS4-private/v41/video-UMrKQWelWp0-transcript.txt` |
| pfeifferj V4.1 GSQ-RCO GGUF (3.0 bpw) = 2nd oracle | HF `pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF`, its `runtime/` has a patched llama.cpp |

**Checkpoint layout** (one layer per shard): 1 vision · 2 embed · **3..42 = layers 0..39** (dense + 384 experts each)
· 43 head+norm · 44–46 DSpark/MTP · **47 = engram layer 1** · **48 = engram layer 14**.
**Mapping:** `shard = layer + 3` (layer L → shard L+3). GSQ's calibration forward adds the engram at layers **1 and 14**
(`config.json` `engram_layer_ids: [1,14]`, applied at model.py:1262). So layers **0–13 need only shard 47**; layer 14+
needs **shard 48**. Layer 0 needs no engram at all.

---

## Download (detached, survives the session)

- **Driver:** `tools/v41/fetch_sequence2.sh` (launched with `setsid nohup … &`), log
  `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/sequence2.log`; each file's own progress in the
  `fetch.log` of its directory. (`fetch_sequence.sh` / `sequence.log` are **superseded**.)
- **Order (Engram-matched to GSQ, set 2026-10-09 20:00):**
  `47 → backbone layers 2–13 (shards 5–16) → 48 → the rest of the backbone (17–46)`.
  Rationale: shard 47 is the only Engram shard GSQ needs for layers 0–13, so those layers' backbone shards are fetched
  first so GSQ can run **while** the rest downloads; 48 is packed in well before layer 14.
  The script is **self-healing**: if it starts and no 47 fetch is running (e.g. after a reboot), it fetches 47 itself.
- **Fetcher:** `tools/v41/fetch_v41.sh DEST [REPO]` — resumable (`curl -C -`), one `.ok` marker per verified file,
  sha256 via **O_DIRECT** read (the page cache lies on this box). Env: `JOBS=1 FIT=1 MARGIN_GB=10 ONLY=regex SKIP=regex
  ORDER=name|size`. Re-run the same command any time: verified files are skipped, partials resume.
- **Manual restart after a reboot** (or just `setsid nohup bash tools/v41/fetch_sequence2.sh &`, which does all of this):
  ```
  cd /media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-00047-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 ONLY='model-000(0[5-9]|1[0-6])-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd "/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8" && JOBS=1 FIT=1 ONLY='model-00048-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  cd /media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 && JOBS=1 FIT=1 SKIP='model-0004[78]-of' ~/AI/Strata-DS4/tools/v41/fetch_v41.sh "$PWD"
  ```
- **Speed:** ~8–14 MB/s (occasional 1–3 MB/s dips): ~9 min per 7.4 GB shard, ~2 h per engram shard, ~11–12 h total.
- **Cautions:** never edit a running bash script in place (bash reads incrementally) — write a copy and `mv` over it.
  The HF token is visible in `ps` (`curl -H`) — don't paste process listings into anything shared.

---

## GPU coordination (single RTX 4090, shared with GLM)

- Shared with **`strata-glm`** over quetza-relay. **Lock:** `~/.quetza-data/conductor/ds4-gpu.lock`.
  Take it for **≤10–12 min blocks**, `relay_ask strata-glm` before you do.
- **Every full-model / big-RAM run** goes through `bench/ds4-2026-10-08/q.sh <log> [extra args]`:
  it waits for the lock **and** MemAvailable ≥ 70 GiB, refuses if ComfyUI (:8188) is up, then runs under
  `tools/ds4/memguard.sh 80 78 -- …`.
- `tools/ds4/memguard.sh <max_gib> <need_avail_gib> -- <cmd>` runs the process in a systemd **user scope** with
  `MemoryMax=<max_gib>G` and `MemorySwapMax=0` (kernel OOM-kills *this* process instead of swapping the desktop),
  plus a watchdog that kills the scope if MemAvailable drops below `MEMGUARD_FLOOR_GIB` (default 4), and holds the
  GPU lock for the whole run. **Why:** on 2026-10-06 a CPU-only full-model oracle (~81 GB anon RAM) + ComfyUI
  start-up thrashed swap into a **hard freeze** (no OOM kill). So: never a CPU-only full-model load.
- While GLM benchmarks: run CPU work at `nice -n 15`, 4 threads; **don't** rebuild `build-ds4-gpu` while a batch is queued.
- The calibration takes the lock **per layer** (memguard holds it for that layer only), so other jobs slot in between
  layers.  A short GPU test: `flock ~/.quetza-data/conductor/ds4-gpu.lock <cmd>` (waits for the next gap).  A longer
  GPU job (real-model p600 gate etc.): `v41calib stop` -> wait for the driver to exit -> run -> `v41calib start`
  (template: `~/AI/Strata-DS4-up/bench/upstream-merge-2026-10-10/gate.sh`; `gate-isolated-pause.sh` = the variant that
  leaves the calibration paused for a strata-glm window).
- The merged-tree GPU build lives in the worktree: `~/AI/Strata-DS4-up/build-gpu` (Ninja, sm_89, CUDA, tests ON,
  `STRATA_GGML_DIR=/home/mal/AI/Strata-DS4/third_party/llama.cpp`); the main checkout's `build-ds4-gpu` predates the
  merge.  Configure line: `docs/UPSTREAM_SYNC.md` / commit `cc55d4c9` message.

---

## Built and verified (commits on `deepseek4`)

| commit | what | verification |
|---|---|---|
| `28dd12dc`,`6ce93ed3`,`626dbac0` | `docs/ds4/V41_PLAN.md` — plan, V4→V4.1 deltas, mixed tiers, research rounds 1–2 | — |
| `28dd12dc`,`c67b47ef`,`383fd8b7` | `tools/v41/fetch_v41.sh`, `fetch_sequence.sh`, `fetch_sequence2.sh` | tiny-repo test incl. corrupted-file recovery |
| `f3bc404d` | `tools/v41/torch_kernels/kernel.py` — torch stand-in for DeepSeek's tilelang `kernel.py` (`act_quant` / `fp4_act_quant` / `fp8_gemm` / `fp4_gemm` / `sparse_attn` w/ sink / `hc_split_sinkhorn`; rounding copied from the kernels) | `tools/v41/tests/test_torch_kernels.py` 15/15; `tests/selftest_model_cpu.py` runs DeepSeek's `model.py` (prefill + decode + DSpark) on it |
| `e1076fc3` | `tools/v41/shard_check.py` — per-shard validation + per-expert weight-space tier errors → `bench/v41/shard-check/` | layer 0: 2,334 tensors, 0 issues |
| `50e9428d` | **PTQ1_0 ternary on CPU**: `include/strata/kernels/cpu/ptq1_0.hpp`, `src/kernels/cpu/ptq1_0.cpp` (pack / unpack / quantize_ref / dequant; Q8_0 dot scalar + AVX2), wired into `native_expert.cpp` via a type-trait wrapper | `tests/cpu/ptq1_0_test.cpp` **16/16** (target `ptq1_0_test` in `build-ds4`): bytes identical to llama-tq's reference quantizer at n = 5120 / 2304 / 128; AVX2 == scalar bitwise; == float reference; mutation caught |
| `42c77429` | ds4_dense: stale-graph unbind compares buffer pointers (GLM's gallocr audit) | ASan CPU gate `swa16`/`tame` (shared + per-graph allocator): PASS |
| `c3ca0410` | `docs/ds4/V41_FEASIBILITY.md` | — |
| `9221882a` | `docs/ds4/RESEARCH_2026-10-09.md` (Infernix, Strata 0.1.41, community ideas) | — |

| `2e442b20`,`317ef99a` | v41_gptq.py (exact-format multi-tier GPTQ), v41_calib.py (layer-streaming calibration), v41_alloc.py, calib_layers.sh | test_v41_gptq 17/17; test_calib_equiv: split forward == DeepSeek Block.forward bit-exact (layers 0/1) |
| `80d7cccd` | GPTQ damp sweep options; default damp 1.0 | layer-8 sweep (§Session log) |
| `d2890cc2` | `build_corpus.py`, vendored `dsv41_encoding.py`, `prep --corpus`, `docs/ds4/calib-refs/` | 0 render failures (4690 + 40 sessions); tags balanced; secrets scan clean; per-sequence BOS/role-token check |
| `baa0b23b`,`0099974a`,`7effced9`,`122c7c3e` | `calibctl.sh` (v41calib), resumable driver + STOP, ~4K-token attention batches, ETA | dry-run start/stop/resume/stop-now in a scratch dir |
| `cc55d4c9`,`497dbd47` | **upstream Strata merged** (graft merge, 8 conflicts) | CPU parity (ptq1_0_test, pool_tasks_test 168 bitwise, q8k identical, iq_avx2_parity 0 fail, ds4_geometry, test_ds4_moe 15/15); dense gate bit-identical 63/63; p600 ppl 10.41 / 19.29 tok/s |
| `20618210` | fork isolation (fork/*.cuh, tools/ds4/cmake/fork.cmake, ds4_guard.hpp) | same CPU gates + dense gate bit-identical; p600 ppl 10.47 / 19.55 tok/s |
| `2a4254d8`,`d564238a` | `docs/UPSTREAM_SYNC.md` (process, gates, isolation rules) | — |
| `5d99832d`,`e567a476` | `v41_calib.py head` + end-of-chain head gate in `calib_layers.sh`; batch default divides N | CPU smoke (4x256, head on L1) runs end to end |
| `4a403b15`,`cc26cb5d` | PTQ1_0 in the grouped-expert CUDA kernels (dot, Fmt<143>, dequant, row bytes) | `ptq1_0_cuda_test`: dequant bit-exact, mmvq 9.9e-8, `native_expert_grouped` all 9 Q2_0/Q3_K/PTQ1_0 pairs rel <= 1.6e-5 |
| `d52f57ca` | `v41_formats.py` block encoders/decoders | `tests/test_v41_formats.py` 6/6 bit-exact vs ggml + ptq1_0.cpp (builds `tests/fmt_ref.cpp`) |
| `70f81b03` | `v41_gguf.py` writer + `v41_gguf_meta.py` (DwarfStar metadata port) | full dry run claims every source tensor; 2-layer real file, synthetic mixed tiers: --verify 68/68 + `check_v41_gguf.py` PASS |
| `d020b910` | `v41_quant.py`, `quant_layers.sh`, writer `--experts-in-place`, cmd_layer hooks | CPU smoke quant L0 (q2_0); in-place file + fill: check PASS, Q2_0 rel 0.413 vs FP4 |
### Facts established (measured, or read from source)
- **V4.1 has NO per-head q RMS norm** (V4 does). JigSawPT's port erroneously kept it → fluent text but layer-39
  correlation 0.78 vs reference. Do **not** add it.
- FP8 dense: **32×32** E8M0 block scales. FP4 experts: int8-packed, **low nibble = even index**; E2M1 codes
  {0, .5, 1, 1.5, 2, 3, 4, 6 + negatives}; one E8M0 scale per 32 along K.
- Tensor names: `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}`,
  `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}` (I8 `[2304,2560]` / `[5120,1152]`), `ffn.shared_experts.*`,
  `ffn.gate.{weight,bias,bias_vl}`, `hc_{attn,ffn}_{fn,base,scale}`, `attn.attn_sink`.
- **Weight-only priors cannot rank experts.** All 384 experts of layer 0 have the same RTN error to 3 decimals
  (ternary 0.443 / Q2 0.378 / Q3 0.183 relative); FP4 training gives every expert the same per-32 distribution.
  ⇒ Allocation must come from **calibration**: activation-weighted (GPTQ/GSQ) error × REAP saliency.
- `transformers` has no `deepseek_v41` (only `v4`) → the reference is DeepSeek's own `model.py` + our torch kernels.
- Strata's native expert kernels run Q2_0, Q2_K, Q3_K, IQ2_XXS, IQ1_M, and (2026-10-10) PTQ1_0 on CPU and GPU.
- **Engine port must teach `include/strata/artifact/gguf_reader.hpp` type 143** (`ggml_type_name` / `block_geometry`:
  128 elems, 28 B, "PTQ1_0") - today it would size PTQ1_0 tensors as 0 bytes.  Shared upstream file: keep the edit
  isolated per docs/UPSTREAM_SYNC.md.  Its tensor-name check (`check_ds4_model`) is V4's; V4.1 needs its own.
- A mixed-tier layer = up to 9 `native_expert_grouped` calls (one per gu/down type pair present), each with its
  own `NativeExpertLayout`; the engine's per-layer single expert stride (`Ds4GgufBlobs`) must become per-expert.
- The vendored ggml (`third_party/llama.cpp`) has `GGML_TYPE_COUNT = 43`: **type 143 must never reach ggml**
  (it would read past the traits table). CPU path wraps it via a local traits object in `native_expert.cpp`.
- PTQ1_0 = id 143 in llama-tq and ik_llama.cpp (#2550). Block = 28 B: `qs[0..15]` 5 trits/byte (values m+16n),
  `qs[16..23]` 5 trits/byte (80+m+8n), `qh[0..1]` 4 trits/byte (120+h+2n, shifted to the top trit), fp16 `d`.
- DwarfStar's CUDA SSD expert cache is plain global LRU (we are ahead); **take** its engram prefetch-at-token-start.
- Speed references: sybil-solutions (3090 + 503 GiB 8-ch DDR4) ~20 tok/s to 261K (experts all in RAM + AVX2 CPU tier);
  JigSawPT (5090 + 126 GiB, NVMe streaming, MXFP4) 5.12 tok/s on new content.

---

## Next steps (in order)

1. **Calibration runner + GPTQ screening — DONE; full run in progress (2026-10-10).**  `head` step BUILT
   (`v41_calib.py head [--layer 40]`, writes `stats/head_L40.json`: ppl / top-1 per domain, fit vs held-out); expect
   held-out ppl in the single digits on web/code and a sane top-1 (~0.5-0.7) - a broken layer anywhere shows up as a
   ppl far off.  Then `v41_alloc.py --layers 0-39 --per-proj` for the full map (cross-check rarely-routed experts
   against smalinin's V4.1 imatrix).
2. **Quantize pass — BUILT** (§Export pipeline; GPTQ at the chosen tier).  **GSQ refinement** (optional, plugs in
   where `v41_quant.quant()` calls GPTQ) (screening already gives the GPTQ init and the tier per expert). Quantizer modules from
   `~/AI/gsq-rco/GSQ/src/quantization` (2-bit grid = Q2_0 group 64; ternary group 128 = PTQ1_0; 3-bit → project onto
   Q3_K). Record each expert's activation-weighted error at all three widths (GPTQ init is cheap; run GSQ refinement
   only at the chosen width) → RCO-lite knapsack on `error × saliency` under the exact byte budget → second pass at the
   chosen widths. Optional Hadamard rotation per tensor (measure both).
3. **PTQ1_0 on the GPU** — IN PROGRESS (§In flight 2): done in the grouped-expert kernels (`iq_kernels.cu` FORK path,
   which is what DS4's experts use); the plan below for `native_mmvq.cu` is only needed for dense use.  Original note:
   add to `src/kernels/cuda/native_mmvq.cu` (+ header `native_mmvq.hpp`: `supported(143)`,
   `weight_bytes`, dispatch) with q8_1 activations: per 128-block, 4 q8_1 sub-blocks, dp4a of `(digit-1)` with q8.
   Reference: llama-tq `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` (MIT). Then the FWHT op for rotated tensors, then prefill
   (dequant → fp16 GEMM is exact for ternary). Later: R8 row-interleaved CPU repack (ik #2564: 3.4×).
4. **GGUF writer — DONE** (§Export pipeline).  Open: rotation flags (if Hadamard is used), mtp.* (DSpark) export,
   a load test in DwarfStar for the dense half (their loader will refuse our expert types).
5. **Strata-DS4 V4.1 port** (V41_PLAN.md "Plan B"): loader; dense half (single-pass HC, ratio 0/2/1 compressors,
   kv/index source sharing, two-level indexer, fp4 compressed KV, **no q norm**); engram op with prefetch;
   384-expert tier (Q2_0 / Q3_K / PTQ1_0); then DSpark. Golden numbers from step 1.
6. **Validation**: per-domain held-out ppl of mixed vs all-Q2_0 control vs the FP4 original; then speed with soft-prune.

---

## Other open items (DS4 V4, from earlier today)

- **b33:** DS4 V4 decodes 21.2 tok/s at both 256K and 512K (`--pos-offset` probe, `iq4_nl` KV); soft-prune neutral in
  the short probe — **needs a real 64K-document run** (prefill + decode, unpruned vs pruned).
- **Strata upstream merge: DONE 2026-10-10** (Mal: keep tracking upstream).  Upstream main `fb58e0db` merged
  (`cc55d4c9`), fork code isolated into `src/kernels/cuda/fork/` + `tools/ds4/cmake/` (`20618210`); gates in
  `bench/upstream-merge-2026-10-10/`.  The old `~/AI/Strata-DS4-0141` worktree is obsolete.  Upstream features that
  live in the Qwen engine path (Foresight swap / router look-ahead prefetch, prefill CPU share, MMVQ_IL rows table,
  --expert-cache-per-layer) are worth porting into the DS4 path as ideas.  GLM is doing the same merge (branch
  `glm-upstream`) and takes DS4's shared-file resolution.
- **Ideas queued for DS4** (`RESEARCH_2026-10-09.md`): rank-gated `--arena-adapt` admission, persisted expert state,
  elastic KV / 32K VRAM KV window for 512K, adaptive MTP draft length, batched NVMe reads.
- **MiMo-Audio 7B** was deleted at Mal's request (sources: `~/AI/Strata-MiMo/docs/mimo/MODEL_FILES.md`). He wants it
  back later: `hf download XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ `-Tokenizer`) — **download to ext4 first** (`hf`
  hangs on exFAT file locks), then move to `NVME1TB/Models/Mimo-Audio/`.

---

## Working rules (this box / Mal)

- GPU is shared with `strata-glm` over quetza-relay; lock `~/.quetza-data/conductor/ds4-gpu.lock`; full-model runs via
  `bench/ds4-2026-10-08/q.sh` + `tools/ds4/memguard.sh` (never a CPU-only full-model load — it froze the box once).
- CPU work at `nice -n 15`, 4 threads while GLM benchmarks. Don't rebuild `build-ds4-gpu` while a batch is queued.
- Never `pgrep -f`+kill a pattern that appears in your own command line (killed own shell once) — use PIDs/PGIDs.
- Verify downloads with **O_DIRECT** reads; RAM has a known bad lane (soft-offlined).
- Commits end with the `Co-Authored-By` trailer; **no pushes without Mal's OK.**
- Measure before concluding; reproduce before diagnosing; commit each verified step.

---

## Engine (night of 10-10) - tools/v41/engine/

| file | what | gate |
|---|---|---|
| `ds41_geometry.hpp` | keys, full tensor map (conditional sets), per-expert tier maps, full-length GGUF arrays (the reader samples 64) | `test_ds41_geometry` PASS + mutations |
| `ds41_dense.{hpp,cpp}` | V4.1 dense half: one pass = n tokens at pos0 (decode n=1, prefill chunk); single-pass HC; window ring [ring \| pass rows]; compressed KV on kv sources (ratio 2 pooled, 1 plain) read by later layers; indexer (fused ggml_lightning_indexer) on index sources, top-k reused; layer-20 candidate blocks (pool max + pinned newest block); flash-attn with the QUERY AS BATCH (each query its own window + top-k rows), sinks; Engram rows from HF shards 47/48; snapshot/restore (rings only) | CPU vs `ds41_golden.py`: 3-layer file, 21 tok cos>=0.99996 top-1 21/21; 1688 tok (top-512 of 844) cos>=0.99983; decode == prefill |
| `ds41_ops.hpp` | model.py's KV/indexer rounding (FP8/pow2, FP4/E4M3, FP4/pow2) as exact stock-ggml float tricks, `act_quant` default on | bit-identical to torch on 2M values (NaN-overflow case saturates) |
| `ds41_engram.{hpp,cpp}` | n-gram hash (GGUF arrays) + parallel preads of rows + scales | via the dense gate (layer 1) |
| `ds41_golden.py` / `compare_ds41.py` | model.py on the GGUF's own weights (CPU, layer by layer) / per-layer cos, routing sets, logits | - |
| `ds41_moe.{hpp,cpp}` | fork of tools/ds4/ds4_moe: per-expert format CLASSES (gu,d pair); per-class launches / pool runs / LRU; chunk = pass per class | CPU path via ds41_generate on the 3-layer file; GPU path compiled, NOT RUN |
| `ds41_generate.cpp` | `-m F --ids ... [--ppl] [--prefill-chunk N] [--serve] [--engram D:D:D] [--rank r.bin] [--vram-lru]` | CPU smoke on the 3-layer file |

Facts found tonight:
- model.py's KV rounding matters: without it 3-layer outputs move cos 1.6e-3 and 16% of routing sets (DwarfStar
  emulates it too).  Engine with it vs golden with it: cos >= 0.9992 (step-function amplification of Q8 noise).
- The CPU native-expert buffers were V4-sized: raised for V4.1 (kNativeActBytes 5840, kNativeHBytes 2628,
  kNativeFFMax 2304, ActQ 5120 wide).  Q2_0 classes need ActQ activations (Strata's Q2_0 kernels), not nact.
- The writer's `--experts-in-place` file is shorter than its last expert region until the quant pass writes it
  (the partial test file was truncate-extended); the real file is complete after step 4.
- NEVER `pkill -f` with a pattern in your own command line: it killed this session's shell once tonight.
- 3-layer test file: /home/mal/AI/v41-dev/v41-L0-2h.gguf (holes = zero experts); goldens in v41-dev/g1, g2, g2q.

40-layer CPU gate (2026-10-11 00:30, dense-only test file /home/mal/AI/v41-dev/v41-all-h.gguf, zero experts, 300 tok):
- real top-k (512 >= visible rows), no act-quant: block outputs cos >= 0.996 every layer (most >= 0.9993), logits
  mean cos 0.99995, top-1 295/300 vs model.py
- top-32 / 8 candidate blocks (forced selection): selected-row sets identical for 272/300 queries at L2, mean Jaccard
  0.98-0.994 at every index layer (L24+ candidate layers incl.); differences = boundary ties (ggml F16 activations)
- engine decode == chunked prefill over 40 layers: logits cos 1.000000, top-1 300/300 (act-quant on, top-32)
- zero experts make |x| ~3e4 by L39 (unnatural), amplifying noise; the real gate is ppl on the filled file.

Real file (2026-10-11 01:55): calibration done 01:29 (teacher head gate ALL/eval ppl 4.915, top-1 0.701; code 1.74,
agent 2.12, reason 5.83, zh 6.83, agentic_long 9.79); tier map = all-Q2_0 bytes, proxy error -17%; layout passes
test_ds41_geometry (9 classes model-wide, 7 in L0); GSQ pass started 01:41 at ~13 min/layer -> ETA ~10:30.
test_ds41_moe (CPU) on the real L0, all 7 classes mixed per token: rel 1.2-1.5e-2 vs dequant+F32 = PASS (the Q8
activation rounding, as DS4's tier).  Held-out eval token files for engine ppl: /home/mal/AI/v41-dev/eval/*.i32
(the head gate's eval sequences: compare engine ppl with stats/head_L40.json and the GSQ pass's QMIX head).

GSQ pass night notes (2026-10-11): L7 OOM'd 3x on a 1.88 GiB fp32 torch.cat of the propagated output (fixed:
chunked into bf16, 4d0..; passed try 4); L13 died on q3k_split (an all-zero super-block whose fp16 d did not round-trip;
fixed: search/re-round, logged; passed on the chain's retry); L14 (Engram table on Team CX2) is I/O-bound like in
calibration (8294 s there) -> ETA ~13:00.  Engram row reads from Team CX2 will matter for engine speed too: measure.

DRIVES (found 2026-10-11 06:45): Crucial1TB, Crucible (Engram L1) and Team CX2 (Engram L14) are mounted fuseblk =
ntfs-3g (FUSE, single-threaded small random reads: why calib/quant L14 took 2.3 h - wchan request_wait_answer); SSD NVME
is kernel ntfs3.  After the chain: remount Crucible + Team CX2 as ntfs3 (udisksctl / mount -t ntfs3) and measure the
engine's Engram rows/token before/after - decode reads 48 rows/token from them.
Size options (Mal asked for a Maya-size ~90 GiB): /home/mal/AI/v41-dev/size_sweep.py - calibration proxy with
prune-or-tier per expert: 152 GiB current mix 1.88 (all-Q2 2.17); prune+mixed 120 GiB 0.37, 99 GiB 0.45, 89 GiB 0.54
(57% pruned), 79 GiB 0.68.  Pruning cost only measured on the calibration corpus (OOD risk) -> recommended: keep the
152 file, soft-prune at runtime (route-bias penalty on the coldest experts, GLM's recipe), decide on a smaller file
after real ppl + tok/s.

### Engine next steps (in order)
1. When the layout exists (step 3 of the chain): `test_ds41_geometry` on the real file; full 40-layer dense gate on
   the CPU with zero experts (golden `--zero-experts` is NOT written yet: experts are being filled during step 4 -
   race), small S (~200) so ratio-1 + all index layers run; DS41 candidate logic at small S needs an env override of
   candidate_topk_blocks on both sides (not written).
2. GPU (after step 4 finishes, ~07:00, Mal's GPU): `test_ds41_ops --cuda` (bit-exactness of the rounding on CUDA),
   then `build-ds4-gpu/ds41_generate --backend cuda --experts gpu --prefill-chunk 512 --ppl` through memguard on a
   held-out text; compare ppl/top-1 with the calibration head gate (`stats/head_L40.json`); act-quant on/off ppl.
3. Speed: router predictor (prefetch), rank file from calibration counts (`stats/LNN.json` experts[e].n ->
   int32 pairs), vram-lru, arena budget, chunk MMQ, decode graph reuse; target > GLM (25 tok/s decode, 400 tok/s
   prefill); 512K context (caches: ~1.5 GB at 512K with BF16 storage - set kv_type BF16).
4. Serving: ds41-engine.sh / configs / strata-ds41-quetza wrapper (copy GLM's); tool-call + coherence checks.

## GPU window 1 (2026-10-23) - decode is PCIe-bound; prefetch precision is the lever

Full numbers: the 2026-10-23 GPU profile (maintainer notes, not in this repo); rerun with
`tools/v41/gpu_window1.sh` (refuses while the V4 pass holds the card). V3 + serve flags,
256-token decode = 22.2 tok/s; GPU busy only 46% of decode.

- `attn_router 17.3` / `tier 23.5` ms/token. The tier stage is almost entirely host->device DMA:
  32.5 GiB per 64 tokens (508 MiB/token, 22.6 GB/s = the PCIe x16 ceiling).
- 3,328 large copies = 454 demand misses + 2,874 prefetch DMA; only **1,309 prefetches were used
  (45.6%)**. The waste is ~15 GiB / ~10 ms per token on a saturated bus.
- Device->host (the per-layer routing readback lich-77 removes) is **0.31 ms/token** - that idea is
  dead for us; the prefetcher is the thing to fix.
- Cheap predictors cannot replace `Ds41Dense::predict` (temporal 8%, spatial 0.3%, frequency 11% vs
  the deployed 45.6%): consecutive-token routing overlap is only 42-48%, so an unranked pick is
  chance. Keep the head; add a confidence margin.
- Kernels are not the problem (268 ms per 64 tokens; `native_*` already do 2-4 rows per warp).
- Next: `--pf-b` sweep 0 / 0.7 / 1.43 / 2.0 / 2.9, `STRATA_PF_NO_DUMMY=1`, then a top-1-vs-top-2
  score margin in `build_predict`. Rerun C now that the sidecar name is fixed
  (`V41-MTP-Vision-1017.Q2_K.gguf`) for the DSpark depth-4 baseline.
