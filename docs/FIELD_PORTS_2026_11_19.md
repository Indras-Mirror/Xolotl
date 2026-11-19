# Field ports, 19 Nov 2026

What came in from other forks and engines on branch `field-ports-20261119`, what each switch does, and the A/B runs
that decide each one.  The scan behind it: `private/notes/ENGINE_SCAN_2026-11-19.html`.  Upstream Strata was **not**
merged (Mal: later versions are reported to have slowdowns); only parts that stand alone were looked at.

Everything below is **off by default** unless the results section says otherwise.  A switch becomes a default only
after an A/B on this box (RTX 4090, Ryzen 7 5700X, 110 GB) shows a win outside the noise (Xolotl's rule).

## Results on this box (19 Nov)

DeepSeek-V4.1 V4 XQ, Mal's served config (`private/serve/strata-ds41-1017-v4-serve-q6.json`: Q6_K dense, 80 GiB arena,
DSpark depth 4, route bias 1.0 / 1.0, `STRATA_RB_PROTECT=2`, `STRATA_CHUNK_CPU=1`) through this branch's
`ds41-engine.sh`, fixed probe 8K prompt / 512 tokens.  Results under `~/v41-work/ports-20261119/`.

**Judge by milliseconds per verify pass, not tok/s.**  tok/s moves 5-8 % between identical runs with how many drafted
tokens DSpark gets accepted (2.98-3.71 tokens per pass here); the pass time is steadier (89.0 / 89.1 ms for the same
arm in two runs), but the box drifts too (three baselines in one afternoon: 96.1, 98.5, 103.8 ms).  Single-run
differences under ~5 % need interleaved rounds.

| Check | Result |
| --- | --- |
| Identity: this branch, nothing set, vs production `build-xolotl` (short34, 2 runs each) | NLL 0.4614 / 0.4653 vs 0.4630 / 0.4634, top-1 88.42-88.65 %; mean KL to production 0.0038-0.0042 vs 0.0023 production-to-itself (VRAM seed 690-702 differs run to run, so no bit identity) |
| Normal mode decode | 28.5 tok/s, 110 ms/pass (7 Nov table: 26.1) |
| Fast mode (`XOLOTL_SHADOW`) decode | 34.3 tok/s, 98.8 ms/pass (7 Nov table: 33.0) |
| Keisawaakira step 1 (ROUTE_FUSE X_DEV META_PULL PF_HOSTWAIT HIT_FAST=1), fast | 90.6 ms/pass |
| + step 2 (ATTN_TAIL=3 RB_HOOK HIT_FAST=3 DENSE_ASYNC PIN_RB) | 92.5 ms/pass |
| **+ step 3 (GRAPH_CONC=7 QKV_FUSE) = "s3"** | **89.1 ms/pass (-10 %)**, 37.4 tok/s; short34 0.4651 / 88.61 % (inside the noise band) |
| + step 4 (DEMOTE_ASYNC) | 98.6 ms/pass: lost, not taken |
| DSpark keep 48 / 32 experts (`STRATA_DSPARK_KEEP`) instead of 64 | 92.9 / 89.3 ms/pass but 3.16 / 2.98 tokens per pass vs 3.39: slower overall, keep 64 |

**s3 is the served default now**, set in `tools/v41/serve/ds41-engine.sh`; `XOLOTL_KEIS_DECODE=0` turns the set off,
any single switch with its own `=0`.  Not bit-identical (sums move); quality inside the noise.

**On agent tasks** (Gauntlet v3-small tasks 01 and 03, fast mode, served settings, one run against the two 7 Nov fast
runs): decode **42.7 tok/s** against 39.6 / 40.8, **+5-8 %**.  01 solved at turn 4, 03 at 0.50 as in every earlier run.
Today's 03 ran longer (16,356 tokens against ~9,600), which slows decode, so the gain is not flattered.

**Which s3 switches engage on the served config.**  The engine log says only seven do: X_DEV, META_PULL, PF_HOSTWAIT,
HIT_FAST=3, DENSE_ASYNC, PIN_RB, QKV_FUSE.  `STRATA_RB_PROTECT=2` turns on the smart router, whose route meter
(`o_xids`) the attention tail does not support (`tail_on` requires `!o_xids`), so **ATTN_TAIL is off, and RB_HOOK and
GRAPH_CONC with it** ("OFF (needs STRATA_ATTN_TAIL ...)").  ROUTE_FUSE is skipped by the smart router too (`route_op`
requires `!smart_router()`), and a `--vision` engine needs `STRATA_ROUTE_FUSE=2` for its router anyway.  So the -10 %
is the seven switches; that is why steps 2 and 3 measured flat.  Open job: carry the route meter's readback in the
attention tail, so ATTN_TAIL / RB_HOOK / GRAPH_CONC work with RB_PROTECT, then measure (gain unknown until then).

Note for scripted runs: `score_v4.py`-style harnesses that start `ds41_generate` directly skip the launcher, so they
also skip `STRATA_DSPARK_KEEP` (DSpark then takes 5.14 GB instead of 2.91 GB and the VRAM seed drops ~130 experts)
and the s3 defaults.  Export them, or go through `ds41-engine.sh`.

### Two follow-ups, 19 Nov night: both flat

- **Attention tail under `STRATA_RB_PROTECT`** (`STRATA_ROUTE_METER=0`, opt-in): the tail, RB_HOOK and GRAPH_CONC
  engage, but 3 rotated rounds give 98.7 ms per pass against 94.8 with the meter on.  Dense stays ~30 ms either way; the
  expert tier (57-65 ms) is the pass.  Default unchanged.
- **`STRATA_LA2` re-measured** (6 clean rounds each, alternating; 3 more overlapped another job's CPU renders and were
  dropped): baseline 94.0 ms per pass (89.5-98.2), LA2 W64 95.1 (90.8-102.2); it won the first 2 rounds and lost the last
  4 by 3-5 ms.  NVMe blocked 14.9 -> 9.6 ms, but the guess adds ~2 ms of dense and the early reads queue in front of real
  misses on the drive.  Stays opt-in; not pursued further.

### Where decode time goes now: NVMe

The engine's own pass split says it: **file-tier reads block 15 % of each pass in fast mode (15 ms of 99) and 18 % in
normal mode (20 of 110)**, about 2.8 ms per read, almost none overlapped.  That is the most any prefetching can give.

- `STRATA_D3_SPEC=8` (October's verify-pass speculative reads, `predict_multi` per layer): 40 of 44 reads used (91 %),
  but the extra prediction graph made a pass 117 ms instead of 89.  `STRATA_SPEC_TOP` only exists in the one-token
  decode loop (the served path never calls it); `STRATA_LOOKAHEAD=2` (one layer ahead, in-graph) issues almost
  nothing in time.
- New, **`STRATA_LA2=1` + `STRATA_D3_SPEC_SRC=la2` + `STRATA_D3_SPEC=N`** (TokenWing's two-ahead guess, below): layer l's
  attention graph also applies layer l+2's router to l's FFN input (top `STRATA_LA2_W`, default 64: the decode route
  bias pushes file-tier experts deep into the guess, a top 16 held ~8 % of the file misses), read back with l's
  routing; the best-scored NVMe-tier experts (N per layer) go to the stash readers.  File blocked 15.9 -> 10.4 ms/pass
  (width 64, N 2) / 11.7 (width 32); 34-62 % of the reads are used.  Pass time: 3 interleaved rounds (fast + s3, N 2): baseline 113.8 / 88.9 / 101.4 ms (mean 101.4,
  blocked 16.4), width 32 100.6 / 93.5 / 96.9 (97.0, blocked 11.7), width 64 96.7 / 91.0 / 96.1 (94.6, blocked 9.6); wall
  15.57 / 15.14 / 15.39 s.  **Blocked -41 % reproducibly; the pass gain (~3-7 %) is inside the baseline's spread** (its
  three runs span 25 ms).  The guess still costs dense time (wider argsort + readback per layer), so it stays opt-in;
  next: rank only file-tier experts on the card (a per-layer residency mask) so a narrow top-k does, then re-measure.

## Ported

### Keisawaakira's DeepSeek-V4.1 work (their fork of this tree, 26 Oct to 4 Nov)

37 of their commits, replayed with conflicts resolved toward Xolotl's behaviour.  Their ggml ops are merged into
`tools/xolotl/ggml/xolotl-ggml.patch` (three-way merge at pin 6050b467).  Their measurements are on an RTX 5090 with
92 % of routed experts as VRAM hits (GSQ IQ2_XS, route bias 0.25 / 0.15); gains here should be smaller, because more
of a token's time is file and RAM reads.  Details and their numbers: `tools/v41/engine/DECODE_OPTINS.md`.

| Switch | What it does | Theirs (5090) |
| --- | --- | --- |
| `STRATA_ROUTE_FUSE=1` (`=2` with vision) | router post-processing and the routing prediction as one op | +2.2 % |
| `STRATA_X_DEV=1` | the MoE input quantized on the card | +1.1 % |
| `STRATA_META_PULL=1` | the tier's launch metadata goes up by a kernel, not behind the prefetch DMAs | +1.9 % |
| `STRATA_PF_HOSTWAIT=1` | the host waits on the prefetch, not the copy engine | with the above, +2.1 % |
| `STRATA_HIT_FAST=1|2|3` | the hits launch without a full wait (3: inside the graph, needs RB_HOOK) | +2.4 % alone; =3 with RB_HOOK +5.4 % |
| `STRATA_RB_HOOK=1` | the graphs write the decode readbacks and a flag the host spins on | +1.6 to +2.3 % |
| `STRATA_ATTN_TAIL=1|2|3` | the FFN hc mixes and the shared expert run after the routing readback | +0.55 % (=3) |
| `STRATA_GRAPH_CONC=2..7` | branches of the attention graph on a second stream | up to +3.4 % (=7) |
| `STRATA_QKV_FUSE=1` | q / kv projections fused | +0.4 % |
| `STRATA_DEMOTE_ASYNC=1` | a vram_lru demotion stages its victim off the sum's path | +0.9 % |
| `STRATA_HIT_PF=1|2|3` | the attention graph's hits also take the layer's prefetched experts | see DECODE_OPTINS |
| `STRATA_ROUTER_BF16`, `STRATA_INPUT_PACK`, `GGML_CUDA_MMVF_SPLIT` | rb28 | each bit-identical |
| `STRATA_EMBD_RAM`, `STRATA_LRU_LATE` | rb30 | each bit-identical |
| build54 set | `AA_EVENT`, `LAUNCH_FLUSH`, `ENG_ASYNC`, `MIRROR_POLICY/BW`, `ARENA_ALIGN`, `BLOB_PRIMARY`, `DEC_SPLIT/PARTS/READERS`, `CPU_OVERLAP`, `DENSE_ASYNC`, `DROP_MMAP`, `READ_FIRST`, `SUM_ZC`, `PCIE_SPLIT`, `HOST_PIN`, `PIN_RB`, `ATTN_AHEAD` | see their commit 63f17ae6d |
| `STRATA_SPEC_CANCEL=1` | wrong speculative file reads stop once routing is known | on in their build; opt-in here |

Served together (their rb14 candidate, 11 scenarios): +12.6 % to +23 %.

Also in: disk sessions (`SAVE <path>` / `RESTORE <path>` serve commands), `--ablate FILE` / `--ablate-scale` /
`--ablate-sites` (a refusal direction removed at run time, on any quant), `STRATA_WIDE_PROMPT`, `STRATA_KV_GATHER`,
`STRATA_SHORT_MMVQ`, `STRATA_SERVE_PREFIX_SAVE` (a pool boundary at the end of the system prompt), the image
encoder's `LEND`, MXFP8 / Q6_K Engram tables, Q6_K / Q5_K dense matrices, the Engram hot set for GGUF tables, the
`STRATA_MOE_TRACE` diagnostics, a random seed per request without one, and the Windows RTX 5090 guide.

Kept as Xolotl's, not theirs: the decode routing defaults (`--route-bias 1.0`, `--route-bias-file 0.6`; their rb25 made
both 0), our SHEXP_LATE default (when `STRATA_ATTN_TAIL` is set it takes over the shared expert, and SHEXP_LATE stands
down for that pass), our Q5_K Engram reader, our IQ1_S experts (#86), our DSML cut fix (#106), our control-queue fix
(#79), and our CUDA top-k with upstream's deterministic ties (#133); their batched top-k for 16+ rows (rb8e) is not taken.

### Juud_engine (bnkaiteam), for the Qwen3.8-Flash-Next engine

| Switch | What it does | Theirs (RTX 4090) |
| --- | --- | --- |
| `STRATA_POOL_ADAPTIVE_SPIN=1` | a pool worker parks after 100 us at the end of a layer's CPU work, not 20 ms | together: +3.1 to +9.1 % decode |
| `STRATA_SKIP_UNUSED_ACTQ=1` | one GPU: no activation quantization for a token whose experts are all on the card | (and +9.6 to +15.6 % in a second set) |

`JUUD_POOL_ADAPTIVE_SPIN` / `JUUD_SKIP_UNUSED_ACTQ` work as aliases.

## A/B plan (when the GPU is free)

All runs: close browsers first (`~/v41-work/close-browsers.sh`), interleaved arms, a fresh engine per arm, same seed.
DeepSeek: short34 logit dumps for identity (`xolotl-test-method`), then served decode on the agent set.

1. **DS4.1, identity**: build-ports `ds41_generate` with nothing set against the production binary: short34 logits
   bit-identical (the replay must not change the default path).
2. **DS4.1, the stack**: `STRATA_ROUTE_FUSE=1 STRATA_X_DEV=1 STRATA_META_PULL=1 STRATA_PF_HOSTWAIT=1 STRATA_HIT_FAST=1`,
   then `+ STRATA_ATTN_TAIL=3 STRATA_RB_HOOK=1 STRATA_HIT_FAST=3 STRATA_DENSE_ASYNC=1 STRATA_PIN_RB=1`, then
   `+ STRATA_GRAPH_CONC=7 STRATA_QKV_FUSE=1`, then `+ STRATA_DEMOTE_ASYNC=1`.  Drop any step inside the noise.
3. **QFN45**: `STRATA_POOL_ADAPTIVE_SPIN=1 STRATA_SKIP_UNUSED_ACTQ=1`, decode on the agent set; then each alone.
4. **QFN45 prompts** (existing switches, upstream's measured wins): `--prefill auto:32768` against the default 8192
   at 20K and 32K tokens; `STRATA_PREFILL_EQUAL=1` at 8.3K, 11K, 20K and 32K tokens.

## Looked at, not ported

- **Upstream Strata 0.2.9-0.2.15** as a merge (Mal's call).  Its RTX 4090 prompt win (0.2.14, Strata's own prompt
  kernels: 20K tokens 6.11 -> 5.66 s) needs that whole rework.  The 32K chunk ceiling and the equal cut already exist
  here as opt-ins (item 4 above).
- **StrataGP's 16 patches**: written against Strata 0.1.x (30 Sept); the 0.2.8 merge overtook them.
- **ik_llama #2915** (prompt batches reuse cached experts): both engines here already run prompt experts from the VRAM
  cache.
- **Exact speculative sampling** (Basalt 0.4.0): for drafters that propose their top token, sampling the target and
  accepting on a match is already the same distribution; a gain needs a drafter that samples (DFlash2's probabilities).
- **Basalt's DS4.1 overlaps** (drafter / Engram / copy-engine ordering, sort-free indexer top-k): a separate engine;
  the same family as the switches above, so they wait for the A/B results.
- **Ostinato** (arXiv 2611.11533): the short prefill after a tool output routed onto resident experts; no code, a
  design job for the DS4.1 prompt path.

## Second card: GLM (`glm_generate --remote-device N --remote-gib G`)

Verified on one card before the RTX 3060 arrives: `STRATA_REMOTE_SAME_DEVICE=1` (tests only) lets the "second card" be
the tier's own card, and `STRATA_REMOTE_CHECK=1` (debug) runs each prompt chunk once with its original experts.  Maya-S
v2 IQ2_XXS, 2,312-token prompt + 64 tokens: the remote rows match their true share to <= 5.2e-7, and remote on vs
off moves the first-step logits by KL 0.164 - exactly what changing only `--prefill-chunk` 1024 -> 512 does on the
plain tier (0.164; this quant amplifies summation order).  Plain runs are bit-identical run to run.  Prompt and decode
calls both go through the remote.  Without `--census` GLM ranks experts in index order, so a test without it puts the
remote's experts in one or two layers; the served config passes `--census` (use-ranked across layers).  On one card
the remote's memory comes out of the VRAM margin (`--vram-margin` 9 with `--remote-gib 2` fits; 6 + 3 runs the prompt
staging out of memory) - not an issue on a real second card.

## Looked at on 19 Nov (afternoon)

- **TokenWing** (runmyai/tokenwing, MIT, llama.cpp fork; streams experts from NVMe): the same design as ours (router
  lookahead, RAM + VRAM caches that follow use, layer-major prompts).  What it has that we lacked: the two-ahead guess
  computed in the launch the layer already makes (taken above as `STRATA_LA2`), the ids written into pinned host
  memory inside that launch with a helper thread queuing the reads, demand reads ahead of guesses with guesses read in
  512 KiB chunks and cancelled when the layer starts, and an expert's gate/up computed while its down part is still
  being read.  Their "oracle mode" (the real engine run with perfect guesses) is the right way to measure the ceiling;
  our pass split already prints the blocked share.  Their docs say tok/s swings by double digits between identical
  runs; their headline speeds are user reports.
- **Halocline** (T-Crypt/halocline, MIT; Basalt 0.4.1 + basalt_ada, sm_89 and Qwen4 family only): Qwen3.8-Flash-Next
  UD-IQ4_XS (93 GB) on one 4090 + 64 GB: decode 148-182 tok/s (ours, QFN45: ~150), prompt **5,322 tok/s at 16K**
  (ours: 3,000-3,430).  The prompt speed comes from Basalt / basalt_ada's Ada FP8 / F16 prompt products (Infernix tile
  layouts) and a CPU share for streamed prompt experts - a port from another engine, not a cherry-pick.  Its dense
  27B notes: the MTP drafter's full 248K-row head costs 1.1 of every 1.6 ms draft step; a frequency-ranked draft head
  (FR-Spec, arXiv 2502.14856) is their next step - applies to our dense drafters too.
- **KV precision tail** (Anbeeld/beellama.cpp `--kv-tail-tokens`): most of the KV cache quantized, the newest 1,024
  tokens in F16, one softmax over both inside FlashAttention.  Their Qwen 27B / 64K / RTX 3090 numbers: q4_0 KV median
  KLD 0.00185 -> 0.00106 (-43 %) for 96 MiB more; q8_0 barely moves.  Not for DeepSeek-V4.1 (`--kv-fp4` packs to the
  model's own FP4 grids, exact).  It fits the dense engine's `q4_0` / `k8v4` KV profiles (Qwen3.8-27B now, Qwen4-27B
  next); needs the attention kernels to merge two segments by log-sum-exp.  Also there: KVarN (variance-normalised
  KV quant, better than q4_0 at the same size).
- **Hy4-flash** (Tencent, 16 Nov): 220B (155.5B network + 64B Engram), 512 routed / 16 active, 12B active, hybrid
  sparse + sliding-window attention, hyper-connections, built-in DFly2 drafter - DeepSeek-V4.1's family.  llama.cpp
  support is an unmerged PR (#32517); early testers like its token efficiency (20-24 tok/s Q3 on 2x3090).  An XQ run
  waits for the PR to merge and for independent evals; the engine would build on the DS4.1 port.
- **Nace** quants: their 72 GB DeepSeek-V4.1 coding quant keeps 200 of 384 experts per layer (a hard prune - against
  our soft-prune rule); their Hy4-flash Q3_K / Q4_K files report KL 0.233 / 0.133.
- **PolyStrata** (a Strata wrapper with one-click install), **BigMoeOnEdge** (flash streaming on stock llama.cpp,
  8-16 GB cards): nothing for this box.

