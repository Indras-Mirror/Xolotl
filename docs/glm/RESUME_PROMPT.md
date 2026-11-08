# Resume prompt: Strata for GLM-5.3-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-GLM`.

---

We're building Strata-GLM: the Strata inference engine for **GLM-5.3-Flash** (320B total / 18B active MoE, MIT) on my
RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4, decoding faster than llama.cpp, with a runtime
abliteration LoRA. **GLM is the current priority** (Mal, 2026-10-08). MiMo is parked (kept for its image/audio
input); DeepSeek-V4 has its own session.
Worktree `~/AI/Strata-GLM`, branch **`glm`**, cut from `deepseek4` `a693545`. Same git repo as the other two
worktrees; the three stay separate for now and get joined later. **Work solo** (no /conductor, workers or
subagents). Don't push anywhere until Mal says.

Read first, in this order: this file; `docs/glm/PLAN.md` (target files, port order); `docs/glm/ARCHITECTURE.md`
(deep dive: formulas from the reference code, reuse map, memory/decode budget, decisions);
`docs/glm/FINDINGS.md`; then the DS4 docs we build on: `docs/ds4/RESUME_PROMPT.md` (rules, gates, lessons),
`docs/ds4/ENGINE_DENSE.md`, `docs/ds4/ENGINE_MOE.md`; `git log --oneline -15`.

## 2026-11-09: the next lever is MTP speculative decode

Since the RAM upgrade (1 Nov) every Maya-S-v2 expert sits in the 80 GiB RAM arena (file tier 0): decode is 28.5-29
tok/s and PCIe-bound, and short prompts use the CPU split (`STRATA_CHUNK_CPU`, 0.4). The NextN draft block was
rejected in October (SPEC_DRAFT.md: a k-token verify touches up to 5.2x more distinct experts, and they came off the
SSD then). With no file tier, and the CPU pool able to compute RAM experts beside the card, re-measure it: the block
is already in our GGUF (`blk.45`), Project Maya runs it (v1.1.56, incl. multi-request MTP). Port the GLM NextN
forward + a verify window into `glm_generate`, try k=1..3, keep it opt-in until it wins at equal quality.

## State (2026-10-10 NIGHT-2, s28) - **CURRENT** (supersedes LATE below; read FINDINGS s28 first)
Committed on `glm` (local; push with `git push glm glm:main` when Mal says).
**Shipped serve config** (tools/glm/serve/strata-glm-maya-abl-512k.json; the wrapper default): Maya-S-v2 + LoRA,
512K, chunk 4096, margin 7.5, vram-grow 1.5, **--pf-b 1.0, --arena-gib 64, native q6_k dense (no --dense-requant: Mal's call - quality over +2 tok/s)**.
Measured decode **25.0 tok/s** (q4_k would be 27.2), prefill ~400; exact ppl chat 5.27 / code 3.60 (the old q5_k
default was 5.60 / 3.64).  Long prompts fixed (the 19K-token OOM; 100K verified) - see FINDINGS s28 item 1.
**Decode is PCIe-bound**, not host-latency-bound (s27 corrected): H2D 26 ms/token at ~22 GB/s; kernels 16 ms.
**Settled:** arena 72 GiB with q4 is only +1.5% (27.64) at ~5 GiB free -> 64 stays.  **Decided (Mal):** native q6, not q4_k
(+1.5% ppl for +8.8% speed is not worth it).  **Next levers:** see FINDINGS s28 "Remaining levers"; the q5_k chat anomaly bisect.
ggml patch: third_party/llama.cpp is untracked - apply tools/glm/patches/*.patch to a fresh copy (GNU /usr/bin/patch;
anaconda's `patch` binary is corrupt).

## State (2026-10-10 LATE) - **CURRENT: Maya-S-v2 is the model; s26 stack built; gate 4 (final) may be running** (supersedes NIGHT)
**PUSHED to Indras-Mirror/Strata-GLM main** (2026-10-10 late, `a864dfad`; earlier local-only notes below are stale).
Read FINDINGS s25d + s26 + **s27** first.

**DECODE IS HOST-LATENCY-BOUND (FINDINGS s27, incl. its ERRATA) - read both before touching decode speed.**
nsys: 70% of all CUDA API time is `cudaStreamSynchronize`, **356 calls/token = 8 per layer x 45 layers** (predict,
attn, the `o_span` readback, the two `gpu_run` syncs, the `routed_sum` upload, the finish graph - each a synchronous
ggml call).  The per-layer expert wall is the SUM of card+pcie+pool, so the 9.1 ms/token CPU pool does not overlap
anything.  The fix is de-synchronisation (GPU-side MoE sum, keep the residual on-device, fold `predict` into the
previous finish graph), NOT the copy engine and NOT tier tuning.  **Do NOT repeat the retracted claim that the GPU is
~2% busy**: that kernel table holds only direct launches (the MoE tier) - the attention/finish/predict work runs
inside 34036 `cudaGraphLaunch` calls whose nodes were not expanded, so real occupancy is UNMEASURED.  Re-capture with
`--cuda-graph-trace=node` first.  Prefetch stays (`--pf-b 0` is a net loss, gate 4).

**MODEL:** Maya-S-v2 (peasantsmith, FP8-derived i-quants, 3 shards + NextN blk.45) at
`/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/` (NTFS NVMe, verified copy; the SATA original on
/media/Crucial1TB is slow FUSE - can be deleted). Wired: IQ2_S/IQ3_XXS down kernels, split-aware tier geometry,
block_count - nextn_predict_layers. Our gcsa LoRA works on it (coherent).

**MEASURED (Maya+LoRA, 300K, 12.7K-token chat prompt -> 256 tok, all coherent):**
- base (RCO's --300k flags): prefill 205 / decode 21.7 tok/s (RCO: 158 / 17.8)
- best so far (elastic + chunk 6144 + q5_k dense + census + pcie auto): **prefill 433 / decode 24.9**; 512K: 389 / 24.8
- ppl Maya+LoRA chat 5.34 / code 3.72 (RCO unablated 5.55 / 3.67). Requant ppl with skips ON is noise-confounded
  (q4 chat 5.61, q5 5.73 but q5 code 3.66) -> gate 4 measures it with exact math (X-* arms).

**NEW FLAGS (s26):** `--census FILE` (live per-expert counts; bootstrap tools/glm/census_from_saliency.py ->
~/.quetza-data/strata-glm/maya-s-v2.census), `--pcie auto`, `--dense-requant q4_k|q5_k`, `--vram-pin F` (with
--vram-lru; quality option, -4% speed), `--vram-grow K` now works in --serve (per-request shrink before prompts >256
tok, grow after; grown at startup), chunk cap 8192, MMQ + CPU pool apply the LoRA delta (no forced pcie 1.0 / mmq off),
`GLM_PROFILE_DECODE=1` (nsys --capture-range=cudaProfilerApi). Fixed: per-token full input-span upload, chunk compute
buffer kept after prompt (GlmDense::release_big), indexer pools sized by chunk in decode (NW).

**DROPPED (measured/evidence):** MTP on one GPU (project-maya disables it; helios -26..-57%; PolyStrata +2%),
--route-bias (hit fell, ppl up), vram-lru as default.

**NEXT (in order):**
1. Read `bench/glm-2026-10-09/maya-final.summary` (gate-maya-final.sh: R300/R512 x 4k/6k, R300-4k-nopf = --pf-b 0,
   X-q6/q5/q4 exact ppl, nsys). If it did not run: `bash bench/glm-2026-10-09/gate-maya-final.sh` (build first:
   `cmake --build build-glm-gpu --target glm_generate`; the last 3 commits were not in a GPU run yet).
2. Pick flags: q5_k vs none (by X-* ppl), chunk 4096 vs 6144, pf-b. Edit tools/glm/serve/strata-glm-maya-abl-{300k,512k}.json.
3. Install wrapper: `cp tools/glm/serve/strata-glm-quetza.wrapper ~/.local/bin/strata-glm-quetza` (adds --maya
   port 8143, --maya512 port 8144). Smoke-test via Quetza.
4. **Decode bottleneck - see FINDINGS s27 (this REVERSES the old reading).** Gate 4 answered the prefetch question:
   `--pf-b 0` is a net LOSS (23.81 vs 24.49 tok/s; experts 20.61 -> 24.55 ms/token against predict+prefetch 2.15).
   Do NOT tune the copy engine or the tier split: the GPU is busy 2.1% of the token and the per-layer expert wall is
   card+pcie+pool summed serially. Attack the **8 syncs/layer** (356/token): (c) fold `predict` into the previous
   finish graph, (b) keep the activation on-device / skip the redundant `d_x` H2D when nk == 0, then (a) move the MoE
   weighted sum onto the GPU (bitwise-equal double accumulation) so `finish` consumes it on-device. Then (d) pipeline.
5. Harness bench: ~/AI/quetza-workspace/model-gauntlet/harness-bench (run_bench.py --model glm-maya added; start the
   server with the wrapper's config on port 8143 first).
6. When done with the GPU: relay to peer `strata-ds4-gpu` (mcp quetza-relay relay_ask) that the GPU is free for its
   V4.1 calibration.
GPU is shared via ~/.quetza-data/conductor/ds4-gpu.lock (tools/ds4/memguard.sh). Mal wants: 30 tok/s ablated at 512K,
high prefill, then coding/agentic/coherency benchmarks.

## State (2026-10-10 NIGHT) - **CURRENT: ablated server 3x; the residency lever found (margin must leave scratch room); spec block fetched but does NOT pay** (supersedes the PM block below)
**Everything committed on `glm`, NOT pushed** (12 commits; latest `9e423502`).

**MEASURED TODAY (all real runs):**
- **Ablated decode 3x**: skip 0 = 5.44 t/s -> skip 0.15 + `--renorm-skip` = **16.27 t/s**. That IS
  `strata-glm-quetza --full` / `tools/glm/serve/strata-glm-unc-ablated.json`. (FINDINGS s25.)
- **Residency lever**: `--ctx 524288 --vram-margin 5.5` -> 5.67 GiB expert slots / **731 experts**; `--ctx 32768
  --vram-margin 1.5` -> 15.07 GiB / **1959 experts (2.7x)**. The 512K KV (~6 GiB) + the 5.5 GiB margin are ~11.5 GiB
  that is not experts. **BUT margin 1.5 is TOO LOW -> `ds4_moe: c_parts: out of memory` at decode** (the prefill
  scratch needs VRAM). Use ~3-4 GiB. `--ctx 300000 --vram-margin 1.5` -> 8.69 GiB slots. Sweep:
  `bench/glm-2026-10-09/vram-margin.summary`.
- **The PolyStrata draft block does NOT pay on our box**: built their engine with CUDA (`~/AI/PolyStrata`) and ran
  OUR file + the fetched block -> 5.05 t/s no-draft, **5.17 with the block (+2%)**, 85% accepted (accurate, starved).
  Our own engine does 16-20 t/s on the same file. **Spec pays IFF experts are resident** -> the residency lever.
- **Load slowness is RAM/page-cache thrash**, NOT the disk (dd = 2.5 GB/s) or heat (55 C): the concurrent Maya
  download evicts the model's warm pages (MemAvailable 80 -> 19 GiB). `--arena-lazy` sidesteps it.

**ARTIFACTS:**
- `draft-block.gguf` (2.79 GB, 29 tensors, SHA256 verified) in `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/` - GLM's
  NextN block, model-independent ("one serves every size"); fetched with PolyStrata's `glm_draft_fetch.py` (copy in
  `/tmp/polystrata-mtp/`).
- `~/AI/PolyStrata` built (`build/strata-poly`, CUDA), tokenizer (`glm-tokenizer/`), `glm.profile`.
- **Maya-S-v2 downloading** to `/media/Crucial1TB/models/GLM-5.3-Flash-Maya-S-v2` (96.5 GB, ~60 GB at NIGHT).
- New docs: `docs/glm/{PEERS,SPEC_DRAFT,FAST_START}.md`; `tools/glm/mtp_probe.py`, `--arena-lazy`.

**NEXT (in order):**
1. **Maya-S-v2** (Mal wants it): needs `IQ2_S(22)` + `IQ3_XXS(18)` added to `STRATA_D_FMTS_FORK` (its gate/up IQ2_XXS
   and Q6_K dense are already supported) + sharded loading + ignore `blk.45`. Then apply our LoRA. **CAVEAT: i-quant
   experts decode SLOWER on the CPU** (PolyStrata's own note) - it may be a QUALITY play, not a speed one.
2. **Serving config**: set the ablated config to `--ctx 32768` (or 300K) + `--vram-margin 3.5` -> ~2.5x resident;
   A/B the decode. (Fix the margin-1.5 OOM.)
3. **Test `--arena-lazy`** (armed; `bench/glm-2026-10-09/lazy-arena.log`) -> if it answers in ~10 s, put it in the
   serve configs.
4. **Then** re-test the draft block at high residency; port the GLM NextN forward + the `Controller` only if it pays
   (`SPEC_DRAFT.md` steps B-C).

## State (2026-10-10 PM) - **abliteration-fast is close; BAKE REJECTED; the X(12) gap found** (superseded by the block above)
**UNCOMMITTED in `~/AI/Strata-GLM` (branch `glm`) - nothing pushed.** Three changes live in the working tree:
1. `src/kernels/cuda/iq_kernels.cu:736` - added `X(12)` to `STRATA_D_FMTS_FORK` (Q4_K down_exps, layers 3-5). WITHOUT
   this the merged build cannot load the model (`tier init: native_expert_grouped has no kernel for layer 3's 12/12/12`).
   s22 documented it but it was NEVER committed - it lived only in the `~/AI/Strata-GLM-um` dirty tree. **Required for
   glm to serve at all.**
2. `tools/ds4/ds4_moe.hpp` + `tools/ds4/ds4_moe.cpp` + `tools/glm/glm_generate.cpp` - **`--renorm-skip`** (default OFF):
   rescale the MoE sum by Sum(all w)/Sum(kept w) after a skip drop (decode `gpu_run`:1660-1679; chunks `gpu_run_chunk`
   `c_w` pre-scale :1969-1996, :2424-2432). `--lora-exps` no longer forces skip->0 when `--renorm-skip` is set.
3. `tools/glm/lora_bake.cpp` (new tool) + the bake measurement - FINDINGS s23. **Verdict: DO NOT bake** - the rank-1
   delta is below the q2_k/q4_k resolution and a same-type requant even drifts a ZERO-delta tensor 2% (GSQ grid != ggml).

**Gates (FINDINGS s23):** renorm does NOT regress - stock skip0.15 ppl 3.6206 -> **3.6090** (renorm ON); ablated b-off
3.5943 -> b-on (**skip 0.15 KEPT**, renorm ON) **3.5899**. Build: `build-glm-gpu/glm_generate` 679 MB / 3117 cubins.
**Loop gate PASSED (09:24)** - ablated + skip 0.15 + renorm gave a complete coherent keylogger answer (2542 chars
content, **0** "LPVOID" repeats; `finish=length` only because it hit the 900-token budget). **ablated + skip 0.15 works**
(`bench/glm-2026-10-09/renorm-loop.summary`, `renorm-abl-probe.txt`) - the s20 loop is fixed.

**NEXT (in order):**
1. The ablated server config is now `strata-glm-unc-ablated.json` (skip 0.15 + `--renorm-skip`, fully ablated; the
   separate `-ablated-renorm.json` was folded in). **Run it: `strata-glm-quetza --full`** (new wrapper mode, port 8140).
   A/B its decode tok/s vs the old skip-0 path (~15-18 t/s). If it ever loops again, the levers are (a) unify the
   prefill/decode skip rules, (b) teach the MMQ chunk path the deltas (`ds4_moe.cpp:2145-2180` builds no `plo`).
2. **COMMITTED, not pushed** (`1b05fa5e` code + `431a2e4f` docs): the X(12) fix, `--renorm-skip`, `lora_bake.cpp`, the
   findings/resume docs, and the ablated-turned-fast serve config.
3. Community peers surveyed in **`docs/glm/PEERS.md`** - `project-maya` (same model + upstream, MIT) matters most: a
   **from-FP8 quant toolchain** (GPTQ error-feedback, per-expert FP8 stats; 97.9% of FP8 zero-shot at ~90-96 GB), MTP
   that pays **only head/tail-pipelined on 2 GPUs** (reconciles our s21 single-GPU rejection), an SSD tier, and a
   reusable eval harness (`kl_eval`/`zs_*`/`loop_test`/`ctx_fill`). Also `bodhi37/strata` (NVMe read-amp fix, O_DIRECT,
   pressure governor) and maya's finding that **cache policy is not the lever (LRU=LFU~Belady)**. Abliterated-model
   research is in FINDINGS s24. Note a ppl pass is now ~7 min (arena load 236-411 s on this box).

## State (2026-10-10) - **UPSTREAM MERGED** (superseded by the block above)
**glm is now a real descendant of upstream Strata.** `glm` == `87cd462f`, a MERGE whose second parent is
`upstream/main` (fb58e0db), so from here a plain `git merge upstream/main` syncs - no graft, no rebase.
**Local only, NOT pushed.** (DS4 did the same: deepseek4 = 497dbd47 contains merge cc55d4c9.)

- HOW the graft worked: `git replace --graft 6f32ec07 a1641e9f && git merge --no-ff --no-commit upstream/main;
  git replace -d 6f32ec07`. Our old origin/main 6f32ec07 is TREE-IDENTICAL to upstream a1641e9f (filter-repo rewrote
  hashes); the replace ref is REPO-GLOBAL, delete it immediately. Same 8 conflicts as DS4's: CMakeLists.txt, README.md,
  src/kernels/cpu/{iq_avx2,native_expert,pool}.cpp, src/kernels/cuda/iq_kernels.cu, src/prefill/moe_mmq.cu,
  tools/strata_tokenizer.py. README: ours + upstream's moved to README.strata.md.
- SHARED CORE = DS4's merge commit **cc55d4c9** (in the shared object store): `git checkout cc55d4c9 --   src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/iq_avx2_rows.inl include/strata/kernels/cpu/iq_avx2.hpp   src/kernels/cpu/pool.cpp src/prefill/moe_mmq.cu tools/strata_tokenizer.py tools/ds4/cmake/ds4_dense.cmake`.
  pool.cpp and moe_mmq.cu came out byte-identical to DS4's. **KEEP OUR native_expert.cpp** (native_fmt3; there is NO
  PTQ1_0 in the GLM fork - Mal). iq_kernels.cu: start from cc55d4c9, re-add ONLY our abliteration LoRA
  (lora_down_kernel + the `lora` param + forcing sw_v1 so the float `h` exists).
- ISOLATION (Mal's ask, now structural): upstream's `STRATA_GU/D/MMVQ_FMTS` stay VERBATIM and fork formats live in
  `STRATA_*_FMTS_FORK` macros expanded at each use site. GLM needs **X(12) added to STRATA_D_FMTS_FORK** (Q4_K
  down_exps, layers 3-5) - upstream's STRATA_D_FMTS has no Q4_K down, so without it tier init dies with
  "native_expert_grouped has no kernel for layer 3's types 12/12/12".
- BUILD (a worktree has no third_party/llama.cpp): `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
  -DGGML_CUDA_GRAPHS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_NATIVE_EXPERTS=ON
  -DSTRATA_GGML_DIR=<main worktree>/third_party/llama.cpp`. glm.cmake and ds4_dense.cmake now honour STRATA_GGML_DIR.
  **Without STRATA_GGML_CUDA/GGML_CUDA_FA/GGML_CUDA_GRAPHS you get a 64 MB binary with NO ggml-cuda; the correct one
  is ~679 MB / 3117 cubins (`cuobjdump --list-elf | wc -l`).**
- GATES (2026-10-10, all green): build 679 MB / 3117 cubins; pool_tasks_test 168 bitwise; q8k_quant_parity identical;
  iq_avx2_parity 0 failures; code ppl 3.6478 + chat 5.7046 (pre-merge code 3.6066 = +1.1%, inside the residency
  noise - NOT A/B'd against the old binary); **abliteration gate: code ppl WITH the routed-expert LoRA 3.5943 vs
  pre-merge 3.5967 -> the spliced LoRA is correct.** Gate script: bench/glm-2026-10-09/gate-um.sh.
- NEXT: (1) adopt DS4's **20618210** - pure MOVES, every signature unchanged: fork code into
  src/kernels/cuda/fork/*.cuh behind one #include, our LoRA kernels there too, so our shared files match DS4's;
  (2) DONE - decode A/B pre-merge vs merged, identical flags, both at file tier 146: 15.69 -> **16.25 tok/s (+3.6%)**,
  so the merge does NOT regress GLM decode. NOTE the ~20.65 tok/s in s18 was a FILE-TIER-0 run; always compare at the
  same file tier (the tier line prints it) or the residency dominates the number. DS4's merge gave +15% - re-check
  ours at file tier 0 when a window allows; (3) scratch worktrees
  ~/AI/Strata-GLM-rebase and ~/AI/Strata-GLM-um are droppable.

## State (2026-10-09 LATE) - superseded by the block above
**Serving is DONE and the model is fast. The abliteration is half-applied (mechanically working now) and its
QUALITY is the open problem.** Public repo github.com/Indras-Mirror/Strata-GLM (remote `glm`, branch `glm`);
last pushes `62808ddf`, `52bd0ad3`. Commit as you go; push to `glm` (Mal asks for it).

**Run it now:** `~/.local/bin/strata-glm-quetza` (uncensored LoRA, port 8140, 512K ctx) / `--fast` (skip-miss 0.15)
/ `--stock` (8141). It starts the server (~55-70 s load) then QuetzaCodetl. Verified end to end: plain chat,
`/v1/messages` tool call, streaming + streaming tool call, 2-turn reuse, and a real harness round trip (`-p`).

**Speed (FINDINGS s18).** Serving flags at 512K ctx: **decode 20.65 tok/s, prefill ~165-175 tok/s**.
`--skip-miss 0.15` is the lever (from 9.72 at 0.05): code ppl +0.21%, chat ppl -0.97% (both inside the 1-2%
residency noise of s14) -> it is the DEFAULT in `strata-glm-unc.json` now. Context barely matters (32K = +7%).

**Serving plumbing (FINDINGS s16-s18).** `glm_generate --serve` speaks Strata's engine line protocol;
GlmDense `snapshot()/restore()` give multi-turn reuse; `tools/glm/serve/` = tokenizer export (verified exact),
`serve_glm.py` (Strata's server + GLM's tokenizer/template/tool calls), `glm-engine.sh` (memguard protections in
the foreground), configs. Two traps that cost hours:
1. **A config must NOT pass `--slots`** - Strata's server.py:490 does `int(args[index("--slots")+1])`, so a
   `--slots auto` config dies in the SERVER (traceback in the SERVER log, engine log looks silent); the engine
   default `slots=0` IS auto.
2. **Multi-turn refill**: `dense.snapshot()` can FAIL (device copy) and the serve loop then left `snap_len=-1`, so
   the "shares the last prompt" restore never fired and EVERY follow-up refilled from 0 (real traffic: 19160 then
   19317 tokens, 0 reused, ~117 s each). It now logs every decision (`serve: reuse lcp=.. hist=.. snap=.. ->
   reused=..`) and any snapshot failure. Verified: lcp 423 >= snap 421 -> 423 reused, 6.5 s -> 3.3 s. Any request
   from a DIFFERENT conversation still resets the one state, so benchmarking against a live session thrashes it.

**Metrics (FINDINGS s18).** `/metrics` (JSON, Monitor tab) carries live `thinking`/`output`; `/slots` is a full
view; `?format=prometheus` (or `Accept: text/plain`) emits llama.cpp's names (`llamacpp:prompt_tokens_seconds`,
`llamacpp:predicted_tokens_seconds`, `llamacpp:kv_cache_tokens`, `n_decode_total`, `requests_processing/deferred`)
plus `strata:*` (prefill/decode rate, phase, expert tiers, VRAM/RAM/GPU). They live in `~/AI/Strata/serve/server.py`
- that checkout's remote is the UPSTREAM **Niko1221/Strata**, so **do not push there**; the GLM shim imports it.
To put them in Mal's repo, port into `Strata-GLM/serve/server.py` (a diverged copy) and repoint the shim.

**Abliteration (FINDINGS s17, s19, s20) - WORKING.** The adapter's 135 A/B pairs = 57 "solo" (attn_output 30,
ffn_down_shexp 27, applied in the ggml graph) + 78 routed-expert (`ffn_{gate,up,down}_exps`, layers 3-28 x 288; the
README confirms the gate/up factors are structural ZEROS - only routed/shared down_proj + o_proj matter, and
alpha=1/r=1 so there is no scale to apply). `--lora --lora-exps` applies the deltas on BOTH decode (`gpu_run`) and the
non-MMQ chunk path. FOUR bugs now fixed: (i) the down correction read the float `h` the default SwiGLU path never
writes (`sw_v1 = v1 || (lora && lora->a_d)`); (ii) the chunk path silently did nothing (`kMaxEnt`=32 vs 8192 entries ->
dedicated `gp.c_exp`); (iii) `--chunk-mmq` sent prefill through the UNINSTRUMENTED MMQ path (un-ablated prefill +
ablated decode = broken model); (iv) **THE LOOP - `--skip-miss`/`--skip-file` drop a low-weight expert's read on the
miss path, and a dropped expert loses its delta, so the ablation stops being all-or-nothing and the model degenerates
into a repetition loop.** Matched A/B (effort=low, temp 0.6, top_k 20, rep 1.05): skip 0.15 -> "from ctypes import
LPVOID, LPVOID, ..." x107, finish=length, no answer; skip 0 -> finish=stop, 6740 chars, a COMPLETE coherent keylogger.
`--lora-exps` now forces `chunk_mmq=false`, `pcie_frac=1.0` AND `skip_miss=skip_file=skip_file_chunk=0`.
Verified: code ppl 3.5628 -> 3.5967 (+0.95%); benign prompts normal; **the keylogger prompt now COMPLIES with a
complete, coherent answer and terminates** (the stock model refuses it with a clean refusal trajectory; the ablated
one does not).
**Caveat:** the `skip -> 0` guard is a SOURCE edit that is compile-verified but NOT yet in the built binary - the
ds4 session held the GPU lock. The CONFIG `tools/glm/serve/strata-glm-unc-ablated.json` already has the skip flags at
0, which fixes it with the CURRENT binary; rebuild `glm_generate` (`cmake --build build-glm-gpu --target glm_generate`)
in the next GPU-free window. **Cost:** the ablated config cannot use skip-miss, so it decodes at the pre-skip rate
(~15-18 tok/s in these probes) - correctness over speed. **Still untested:** the other refusal categories, adapter v1,
and the reference cross-check (`~/AI/llama.cpp-glm53` glm5next LoRA on the same GGUF+adapter - the way to prove our
application matches upstream).

**30+ tok/s = MTP - and the prior evidence says be careful.** `FINDINGS s7`: "a 2-token verify touches 15.2 distinct
experts -> MTP ~0.9x here (memory-bound)". Plan: `ARCHITECTURE.md:78` (Strata-fied MTP: the block's dense part
resident, its 288 experts through the tier, `--mtp-resident`); head at
`/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/mtp/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.58 GB, depth-1 acceptance 70-75%).
Strata already has `include/strata/core/{mtp,verify,coupled_draft,native_head}.hpp` + the tier's multi-token
`gpu_run_n`. **The skip-miss sweep is DONE** (decode, 64 tok, ctx 524288, serving flags): 0.05 -> 9.72,
0.15 -> 20.65, 0.25 -> 24.44, **0.35 -> 28.70 tok/s** - so **30+ is reachable by skip-miss alone** (~0.4-0.5), at a
growing quality cost that MUST be gated (skipped weight 248.9 -> 260.1/token; ppl at 0.15 was +0.21% code /
-0.97% chat - only 0.15 has been gated so far). `bench/glm-2026-10-09/dec-0.25.log` / `dec-0.35.log`.
**The tier profile changed with the lever**: at 20 tok/s a token is COMPUTE-bound (wall ~19 ms = gpu hits 9.17 +
cpu pool 5.23) with the CPU/PCIe misses nearly gone (cpu 2.1%, pcie 1.1%, file tier 19) - unlike s7's
memory-bound 8.5 tok/s, which is the case the "0.9x MTP" verdict was measured in. Re-derive the MTP economics
from a CURRENT profile before investing. It is still a large build: load the 4.58 GB head, its own DSA layer +
MoE + eh_proj, then draft/verify plumbing (Strata's `verify.hpp`/`coupled_draft.hpp` and the tier's `gpu_run_n`
are the starting points).

**Traps that cost real time**
- **Never `pgrep`/`pkill -f` a pattern that also appears in your OWN command line** - it matched this shell several
  times (`glm_generate --serve`, `serve_glm.py`, and a bracketed pattern whose literal text sat elsewhere on the
  same line). Use `fuser -k 8140/tcp`, or a /proc scan that skips `$$` and `$PPID`.
- `--ppl` slows prefill a lot: measure speed WITHOUT it. Chunked ppl is only comparable at the SAME margin
  (residency moves it 1-2%, s14).
- GPU shared with `strata-ds4-gpu` (relay); one full-model process box-wide; `memguard.sh` takes the lock itself -
  never wrap a chain in `flock` on it (deadlock).

## State (2026-10-09 morning) - superseded by the block above
- **First action: `relay_rename` -> `strata-glm`** (the DS4 session `strata-ds4-gpu` addresses us by that name).
- Public repo: github.com/Indras-Mirror/Strata-GLM (remote `glm`, branch glm -> main). Mal: English + Chinese (+code)
  are what matter. GPU is SHARED with `strata-ds4-gpu` over the relay MCP: blocks <= 10 min, ask/announce, kill exact
  PIDs only (pkill/pgrep -f matched this shell twice), CPU tests niced with --threads 2 while DS4 benchmarks.
- FINDINGS s7-s8: routing is flat -> residency is what counts. Done: exclusive byte-sized VRAM/arena residency,
  chunked prefill (GlmDense passes <= 4096), REAP saliency + `--prune` / `--prune-penalty`, `--skip-miss`, lightning
  indexer (long context; 524K allocates), shared-allocator stale-pointer fix, FA mask-stride fix.
- Numbers (REAP 25%, fully resident): prefill 156-175 tok/s (chunk 1024), decode 8.5-10.4 tok/s (16.4 with
  --skip-miss 0.10, +4% ppl). Hard prune hurts uncalibrated languages (German -18..-29%).
- **Long context WORKS (FINDINGS s9):** `--ctx 524288` allocates (~6 GB latent cache, 734 VRAM slots left at margin
  5.5); 50K-token prefill **229.6 tok/s**; decode at 50K depth **8.23 tok/s** (vs ~8.5 at 2K).
- **DONE (FINDINGS s16): serving + Quetza wrapper `~/.local/bin/strata-glm-quetza`.** Engine `glm_generate --serve`
  + GlmDense snapshot/restore + tools/glm/serve/ (tokenizer verified exact, Strata server shim, configs, foreground
  memguard launcher). The "silent" first start was Strata's server doing `int(args[args.index("--slots")+1])` on our
  `--slots auto`: **drop `--slots` from the configs** (engine default `slots=0` = auto). Verified end to end at ctx
  524288: plain chat, `/v1/messages` tool call, 2-turn reuse (`31 reused`), streaming + streaming tool call; decode
  ~8-9 tok/s. Wrapper starts the server (uncensored LoRA 8140 / stock 8141) then Quetza. Not yet run as a full
  Quetza session. Open: serve-mode `--vram-grow` shrink-before-prefill.
- **NEXT: the same for DS4** (`ds4_generate --serve` in `~/AI/Strata-DS4`, the strata-ds4-gpu session's tree - do not
  edit it) + a wrapper; coordinate with strata-ds4-gpu over the relay.
- **Best real-use config (FINDINGS s15):** `--prune bench/glm-2026-10-09/prune-ezct-0.25.txt --prune-penalty 0.05
  --arena-adapt --skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35 --prefill-chunk 2048
  --chunk-mmq --vram-margin 8 --vram-grow 1 --arena-skip-resident --arena-gib 72 --slots auto` -> prefill ~220-230,
  decode ~9.8 tok/s in one process; German ~= unpruned. Measure real use WITHOUT --ppl (it slows prefill).
- **Soft prune DONE (FINDINGS s10-s12):** GLM's sigmoid router needs penalty ~0.05 (not DS4's 0.5). Recommended:
  `--prune bench/glm-2026-10-09/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35` -> decode ~10.8 tok/s (hard prune 9.5), prefill 130-140, German +2% vs unpruned (hard +61%),
  code -1.7%. `--arena-admit` (heat gate) is a loss on GLM: leave off. Clean base: chat 5.547, code 3.668.
- **Next (in order):** (1) more speed: chunk 2048 = 160 tok/s prefill but needs margin ~8 -> VRAM re-seed / elastic ExpertCache
  (shrink/grow exists, --vram-elastic) so decode keeps margin-1 slots; prestage = no gain (s14); gate quality with the 2000-token evals (chunked) + the decode-loop chat
  gate (chain10; loop vs loop only). Prompt cache across turns (DwarfStar idea, FINDINGS s13) once GLM is served.
  (2) **Numerics (s14): resident (MMVQ) vs streamed (MMQ) experts move chunked ppl 1-2%** - compare at equal
  margin; find the off path with the fork oracle: rebuild ~/AI/llama.cpp-glm53 CPU (patched, `GLM_NO_FUSED_LID=1`) and compare indexer scoring on the
  mini fixture past 11 tokens; then the GPU chunk-vs-loop ppl gap (18.47 vs 20.46 on neutral, CPU fixture exact).
  (3) Long context deeper: prefill ~25% of 200K/500K (Mal: prove it, don't fill it) and decode at that depth; a
  gather path for decode if masked FA over the whole latent cache starts to cost (it didn't at 50K). (4) Re-seed the VRAM cache after chunked prefill (margin
  5.5 costs ~500 slots / ~1 tok/s). (5) Abliteration: transplant drowzeys' 30 o_proj tensors (L15-43 + MTP) -> Q4_K
  attn_output, instead of the expert LoRA. (6) Bigger chunks (4096) for 300-500+ prefill.

## State (2026-10-08 night) - **superseded by 2026-10-09**
- **LATE UPDATE (FINDINGS s6)**: `--lora` is wired for the DENSE half and **verified active** (adapter changes
  generation; `logits[0]` max|d| 2.14); the routed-expert LoRA is still NOT wired (not abliterated). A real-prompt
  warm-cache A/B measured **cold 1.75 -> warm 1.88 tok/s decode (+7% only)** - the arena caps at ~55% of the 12096
  (layer,expert) pairs, so the profile barely helps, and file reads stayed 472 ms/pass. Speed levers, measured:
  bigger arena (~72 GiB holds every expert), higher `--pcie`, lower `--pf-b`, and chunked MMQ prefill (prefill is
  still the 1-token loop at ~1.9 tok/s). Prompts + run scripts live in `bench/glm-2026-10-08/`.
- **The model is downloaded + verified**: `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`
  (113585695168 B, sha256 `5f03b74f...38d9f6d`, verified + renamed by the fetcher) and the MTP head
  `mtp/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.6 GB). `bash ~/AI/glm_fetch_all.sh` is the resumable fetcher (now done).
- **FIRST real-model run DONE** (`glm_generate`, CUDA, memguard 84 70, `--slots auto --arena-gib 60 --vram-lru --ctx 2048
  -n 16`): 45 layers (34 KDA, 11 MLA), dense half on CUDA in 1.4 s, 1269 VRAM slots, arena 60 GiB = 6917 experts + file
  tier 5179, tier load 55.9 s. Decode **1.75 tok/s COLD** (expert hit 18.6%, file reads 460 ms/pass) = **no routing
  profile yet**. Details in FINDINGS s4.
- **Blocker fixed and BUILT**: `STRATA_D_FMTS` (src/kernels/cuda/iq_kernels.cu) was missing Q4_K (12), so GLM's Q4_K
  expert layers 3-5 were refused at tier init. Added `X(12)`; `build-glm-gpu` rebuilt; the run succeeded.
- **Abliteration LoRA: loader DONE + verified, NOT wired** (`tools/glm/glm_lora.hpp`, FINDINGS s5). Remaining: a
  `--lora <adapter.gguf>` flag, apply `solo()` deltas in the `GlmDense` graph (`attn_output`, `ffn_down_shexp`) and
  `exps()` deltas in `Ds4MoeTier` (gate/up BEFORE the SwiGLU, down after) in both CPU and CUDA paths.
- Both trees build clean: `build-glm` (CPU), `build-glm-gpu` (CUDA). **No builds while the `strata-ds4` session runs.**

## State (2026-10-08 evening) - superseded by the night block above
- **Code written; the CPU tree builds clean and the mini fixture runs end to end - NOT run on the real model yet**
  (commits `e889e75` + this session):
  - `tools/glm/glm_dense.{hpp,cpp}` - GlmDense, phase 1: mHC (fused ggml_dsv4_hc_* ops), KDA via
    `ggml_gated_delta_net` (KDA path, K=1) + causal conv state, nope-MLA over an F16 latent cache with flash-attn
    (no indexer: exact up to 2051 tokens, refuses beyond unless allow_long_ctx), dense FFN layers 0-2 and shared
    expert with `ggml_swiglu_clamp` (limit = swiglu_clamp_shexp, 10), sigmoid router + exp_probs_b selection bias +
    norm + x2.5, head = mean of 4 streams -> output_norm -> output.  One input upload per pass (slot + causal mask +
    route bias), one readback per layer ([fn|ids|wts] block).  Graph per (layer, n, MLA cap); per-graph allocators
    with uid reuse.  KDA state cannot rewind: positions must arrive in order (no verify/MTP rollback yet).
  - `tools/glm/glm_generate.cpp` - driver (GlmDense + Ds4MoeTier, decode-loop prefill, --ppl, --route-bias,
    --vram-lru, --dump-routes FILE writes a GLM routing profile in seed_from_routes(file, 512) format; without
    --profile the arena/VRAM cache are seeded in (layer, expert) index order).
  - `tools/glm/cmake/glm.cmake` (included from root CMakeLists after ds4_engine.cmake): `glm_generate` (CUDA tree,
    needs ds4_moe_cuda) / `glm_generate_cpu` (CPU tree).
  - `third_party/llama.cpp` copied from Strata-DS4 (untracked, gitignored; 175 MB): the ggml with every GLM op.
- Verified while writing (no build yet): ggml ops gated_delta_net (scales q by 1/sqrt(S) itself; identical to
  the fork's), swiglu_clamp semantics == Ds4MoeTier's CPU clamp (gate min lim, up clamp +-lim), fork's MoE routing
  order, GGUF tensor shapes (ARCHITECTURE.md; header dump in
  /tmp/claude-1000/.../scratchpad/glm_hdr.json is gone after reboot - re-read with the range-read trick if needed).
  Expert types: layers 3-5 Q4_K, 6-8 Q3_K, 9-44 Q2_K; the tier handles per-layer types + dense layers already.
- **Build (2026-10-08 evening):** `build-glm` (CPU) builds clean on the first try and `glm_generate_cpu` runs the new
  `tools/glm/make_mini_glm.py` fixture end to end (4 layers, finite logits, exit 0) - no GPU, no real model.
  `build-glm-gpu` (CUDA, arch 89, MMQ_KQUANTS) was building as this was written -> `glm_generate`.
- Model download: `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/` (~34/105.8 GiB at 18:5x, ~12 MiB/s; resumable:
  `bash ~/AI/glm_fetch_all.sh` - curl `-C -` appends to `$D/GLM-...q4kattn.gguf.part`, then sha256 + rename + the MTP
  head into `mtp/`; launch detached with `setsid nohup`). The old `download_glm53_flash.sh` `FORCE_EXFAT` command is
  stale (restarts a partial from zero) - do NOT use it. LoRAs in `lora/`.
- Reference fork at `~/AI/llama.cpp-glm53` (neurall/llama.cpp 2e0435a). A **CPU-only** build now exists there (`build/`:
  `llama-cli`, `llama-perplexity`, `llama-tokenize`). The CUDA fork build (for the real-model tok/s baseline) is still
  TODO - do it only with DS4 idle. Fixture oracle is pending: the fork accepts the fixture's tokenizer now, but its
  DSA/k-pool graph asserts (`ggml_set_rows` in `glm5-next.cpp`) on the tiny dims - the fixture's `attention.indexer.*` /
  MLA dims must match the fork before it can serve as a gate. Likely primary gate instead: real-model `llama-perplexity`
  vs our `--ppl` on the same ids.
- Relay: the `strata-glm` session's DS4 ask (`e85816a6`) was still open; the 2026-10-08 evening session had no relay
  tools, so it only checked `ps`/`nvidia-smi` - DS4 was idle (no `ds4_generate`, GPU ~9%), which is why the builds ran.
  Before any GPU block > ~15 min, still coordinate with `strata-ds4` (compiles skew its timings).

## Next (in order)
1. **Wire the abliteration LoRA** (loader DONE + verified, `tools/glm/glm_lora.hpp`). Split in two:
   - **(a) DENSE HALF - IN THE TREE (2026-10-08 late)**: `--lora <adapter.gguf>` on `glm_generate`; `GlmDense` applies
     the `solo()` deltas in `build_attn` as `y += mul_mat(B, mul_mat(A, x))` for `attn_output` (KDA + MLA inputs) and
     `ffn_down_shexp` (the SwiGLU output `z`); the A `[in,1]` / B `[1,out]` F32 payloads go in their own backend
     buffer (`Impl::lctx`/`lbuf`, uploaded in `init`). No-op when `--lora` is absent (all deltas NULL).  NOT yet built
     or run - verify with `bench/glm-2026-10-08/run_lora.sh` (logits with vs without the adapter).
   - **(b) ROUTED EXPERTS - NOT WIRED (the abliteration the adapter mostly carries)**. The adapter's
     `ffn_{gate,up,down}_exps` (layers 3-28, all 288 experts) are Ds4MoeTier's, and the expert math does NOT have a
     single hook: the CPU path is `pool->run_split_multi_native` (ds4_moe.cpp:1123, from `cpu_run` at :1103), and
     `gpu_run`/`gpu_run_n`/`gpu_run_chunk` each have their own CPU-pool fallback + the CUDA grouped/MMQ kernels. Plan:
     (i) add `native_gu_rows_lora` / `native_down_rows_lora` to `src/kernels/cpu/native_expert.{hpp,cpp}` (rank-1:
     `g += b_g[r]*(a_g.x)`, `u += b_u[r]*(a_u.x)` before the clamp, `out += b_d[r]*(a_d.h)` after) - additive, DS4
     passes nothing; (ii) a `Ds4ExpertLora` interface on the tier + three `NativeLora1` handles plumbed into the CPU
     pool's expert jobs; (iii) for the CUDA paths, force LoRA-active (layer, expert) to the CPU tier (their kernel
     would otherwise silently skip the adapter). Start with `--experts cpu` correctness, then wire the CPU-fallback.
2. **Warm the expert cache** (independent, cheap, no code): `--dump-routes routes.bin` on a few prompts ->
   `--profile routes.bin --vram-lru` -> rerun; the 1.75 tok/s is a COLD number (18.6% hit) and should climb.
   `bench/glm-2026-10-08/run_warm.sh` does both on a 690-token real prompt.
3. **Real prompt**: ids via the HF `tokenizers` lib on `bench/glm-2026-10-08/upstream/tokenizer.json` (the fork's
   `llama-tokenize` refuses - its GLM path needs `ctx_other`); prompts in `bench/glm-2026-10-08/prompts/`
   (`neutral*.i32`, `chat.i32`). Then compare first-token logits vs the fork.
4. **Fork CUDA oracle build** (`~/AI/llama.cpp-glm53`, sm_89) still TODO (needed for real-model tok/s + the `--lora`
   reference); its CPU tools are done. `make_mini_glm.py` is fork-loadable EXCEPT the fork's DSA/k-pool path asserts on
   tiny dims - gate on the real model instead (FINDINGS s3).
5. **SPEED (Mal's goal: 20+ decode, 300+ prefill; Qwen-Flash-Next level = 50-70 decode / 1500 prefill)**: the two
   levers not yet switched on are (a) the routing profile (task 2) and (b) **chunked MMQ prefill** - `GlmDense` already
   supports `max_tokens` up to 64 and the tier has `run_chunk`/`chunk_mmq`, but `glm_generate` still prefils with the
   one-token decode loop (1.6 tok/s). Wiring `dc.max_tokens = chunk` + `attn_router_n(l, pos0, n)` + `tier.run_chunk`
   is the prefill path (DS4 measured 392 tok/s chunked). Then ARCHITECTURE.md Decisions 6-7: CUDA graphs, Strata-fied
   MTP (upstream llama.cpp PR #29928 added GLM5Next MTP support 2026-10-08), the lightning indexer.

## Rules (non-negotiable; same as DS4)
- **Every full-model load through `tools/ds4/memguard.sh <cap> <need> -- <cmd>`** (cgroup RAM cap, swap off, shared
  GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`). One full-model process box-wide. memguard checks MemAvailable
  BEFORE the lock, so to queue behind another engine wait first:
  `until flock -n ~/.quetza-data/conductor/ds4-gpu.lock true && [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 70 ]; do sleep 30; done`
- Before any GPU run: `ss -ltn | grep 8188` (ComfyUI up -> stop and tell Mal) and `nvidia-smi`.
- **GPU sharing with the DS4 session** over Quetza Relay (use the CLI `~/.local/bin/relay` - the relay MCP tools were
  NOT available in this session): it is `strata-ds4`; claim this session with `relay rename strata-glm`. Before a GPU
  block > ~15 min, `relay ask strata-ds4 "need/ for/ eta:"`, announce done. NB: the CLI acts as an ephemeral
  `relay-cli-<pid>`, and `strata-ds4` connects then disconnects, so a `peer_not_found` just means retry until it is in
  its connected window; `relay reply` to an older ask can return `unknown_ask` - send a fresh `relay ask` instead.
- **No builds while the DS4 session is running** - a compile (CMake/ninja/nvcc) skews its benchmark timings. Check
  `pgrep -af ds4_generate` and `nvidia-smi` first; only let finish a build that is already almost done.
- Never edit `~/AI/Strata-DS4` or `~/AI/Strata-MiMo` (other sessions' worktrees); read them and `git show` freely.
  Port shared code as your own commits on `glm`. Shared kernels (`src/kernels/`, `include/strata/kernels/`):
  backward-compatible additions only.
- If `/media/mal/SSD NVME` or `NVME1TB` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3` / `nvme1n1p2`.
- Never `pkill -f` a pattern in your own command line; kill only your own PIDs.
- Test the real model end to end early (quality + tok/s), fix after; keep the gates mutation-tested.
- Keep results in `bench/glm-YYYY-MM-DD/`, update `docs/glm/FINDINGS.md` + this file at milestones, say what was NOT
  tested.

## UPSTREAM RE-BASE (2026-10-10) - the way to get the upstream features into our engine
The table items (Foresight prefetch, MMVQ_IL rows table + sm_89, --expert-cache-per-layer, prefill CPU share,
page-lock #1250, Gumbel drafts) are all in upstream/main, but our fork CANNOT `git merge` it: upstream rewrote
history on 2026-10-06 (docs/TROUBLESHOOTING.md #1276), so the histories are unrelated. The working path is a
3-WAY merge with our fork's ORIGINAL upstream as the base:
    BM=$(git merge-base HEAD origin/main)          # = 6f32ec07 (2026-10-04), our pre-fork upstream
    git merge-tree --write-tree --merge-base=$BM glm upstream/main
That produces exactly **8 conflicts** (verified 2026-10-10): CMakeLists.txt, README.md,
src/kernels/cpu/{iq_avx2,native_expert,pool}.cpp, src/kernels/cuda/iq_kernels.cu, src/prefill/moe_mmq.cu,
tools/strata_tokenizer.py - i.e. only where OUR kernel additions and upstream's changes overlap. Everything else
auto-merges, and the merged tree carries upstream's features.

Re-base recipe (SAFE - on a scratch worktree, `glm` stays green):
  1. `git worktree add -b glm-rebase ~/AI/Strata-GLM-rebase upstream/main` (upstream/main was fetched from
     ~/AI/Strata; the `upstream` remote = Niko1221/Strata already exists in our repo).
  2. `git checkout glm -- tools/ds4 tools/glm`  (engines are DISJOINT from upstream: upstream has no tools/ds4|glm)
  3. Append to the root CMakeLists the 2 include() lines (tools/ds4/cmake/ds4_moe.cmake, tools/glm/cmake/glm.cmake).
  4. Symlink third_party/llama.cpp -> the main worktree's copy (glm.cmake hardcodes that path for ggml-impl.h).
  5. Configure: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
     -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<third_party/llama.cpp>`. VERIFIED: all targets present
     (strata_kernels, strata_mmq, ds4_moe_cuda, glm_generate) and glm_dense.cpp compiles CLEAN against
     upstream/main's headers.
  6. Then do the real 3-way merge (step above) into that worktree, resolve the 8 conflicts (keep BOTH sides), build,
     and run our gates (code/chat ppl, the decode-loop gate, the abliteration gate). Rollback: `git worktree remove
     ~/AI/Strata-GLM-rebase && git branch -D glm-rebase`.
Overlap, measured: the shared surface is include/ + src/ (188 files; our fork added ~3391 lines on an older upstream
snapshot). Engines (tools/ds4, tools/glm) are 100% ours. Features are env-gated upstream (STRATA_FS_SLOTS /
FS_AHEAD / PREFILL_CPU_SHARE / MMVQ_IL_ROWS / STAGE_PIN) so nothing changes until switched on. Foresight + prefill
CPU share need ENGINE-SIDE wiring (their wiring lives in upstream's src/program/generate.cpp, which we do not build).
COORDINATE with strata-ds4-gpu: we share include/+src/, so this re-base should be done ONCE for both engines.

UPDATE (2026-10-10, later): **the re-base BUILDS.** After the 8 conflicts were resolved by hand, the build failed with
42 errors (40 in iq_avx2.cpp, 1 iq_kernels.cu:2521, 1 prefetch_ahead). CAUSE (strata-ds4-gpu's diagnosis, correct):
our fork's iq_avx2.cpp "IQ2_XS block" is not ours - it is UPSTREAM'S OLD CODE, which upstream moved into
src/kernels/cpu/iq_avx2_rows.inl. Our merge kept the old copy, so it lost its helpers (row_dot/prefetch_ahead).
FIX: take upstream's iq_avx2.cpp + thread our SwiGLU `lim` clamp - do NOT try to restore the old block.
Concretely, the ds4 session's re-base worktree **~/AI/Strata-DS4-up (branch ds4-upstream)** already had it resolved:
  - src/kernels/cpu/iq_avx2.cpp      (upstream's + our lim; 15 lines off upstream)   -> copied verbatim
  - src/kernels/cpu/iq_avx2_rows.inl (upstream's + our lim)                          -> copied verbatim
  - src/kernels/cpu/pool.cpp         (our resolution matched theirs EXACTLY)          -> no change
  - src/kernels/cpu/native_expert.cpp: theirs adds DS4-only PTQ1_0 (ternary id 143); KEEP OURS (no PTQ1_0 needed)
  - src/kernels/cuda/iq_kernels.cu:   KEEP OURS (the abliteration LoRA kernels live here; theirs has none) - the one
    fix is line ~2521 `swiglu_entries_kernel<<<...>>>(gate, up, h, nh)` -> add `, L.swiglu_limit`.
RESULT: `cmake --build build-rebase --target glm_generate` -> **0 errors, links, and `glm_generate` runs (prints usage)**.
Rebase binary build-rebase/glm_generate is 64 MB vs our main build's 663 MB - FLAG: check the kernel/format coverage
(the gates will catch a missing expert type).
STILL TODO: (b) isolation refactor (move the fork kernels to a fork include + additive macros so upstream merges are
cheap), and (c) run the gates (code/chat ppl + decode-loop gate + abliteration gate) - needs the GPU.
