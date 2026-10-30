# Strata-GLM findings

> **To check (2026-10-31): row-interleaved CPU expert kernels.** Check the CPU experts (Q4_K, Q5_K, Q2_K, IQ3_XXS) for a row-interleaved 1-token kernel; kq256 is opt-in and multi-token only. See [CPU_EXPERT_TILING.md](../CPU_EXPERT_TILING.md).

## s1 (2026-10-08): why GLM-5.3-Flash, what to run, what fits

**Choice (Mal, 2026-10-08):** order is now GLM-5.3-Flash -> (DS4 in its own session) -> MiMo later. MiMo is kept for
its omnimodality (audio in, video, speech out), not coding.

**Quality research** (benchmarks + Reddit, 2026-10-08):
- Artificial Analysis Intelligence Index: GLM-5.3-Flash 42, DeepSeek V4.1 Flash 39, MiMo-V2.6-Flash 38,
  DeepSeek V4 Flash 0731 34.
- Vendor-reported, GLM vs DS 0731: Terminal Bench 2.1 84.3 vs 82.7, DeepSWE 63.4 vs 54.4, Toolathlon 78.4 vs 70.3.
- User reports: GLM writes the cleanest code and fixes root causes, but is slow and loops on long unattended runs.
  DeepSeek is fast and needs little babysitting. MiMo-V2.6-Flash is widely panned for coding agents (thinking and
  tool loops).

**DeepSeek V4.1-Flash is not viable here:** it's a new architecture (encoder-decoder + engram, 552B + 196B). Read
from the Q2_K GGUF headers (vcruz305, needs a llama.cpp fork): 264 GB = 197 GB experts + 65 GB engram (can stay
on NVMe) + 3 GB other. That's about 2x RAM+VRAM even with the engram on disk.

**GLM files:** target is neuralll's 3.0-bit Q4_K-attention GSQ-RCO (113.6 GB). Uncensoring is a rank-1 LoRA
applied at run time (GCSA Abliterix v2 by default); details in PLAN.md.

**Architecture** (PLAN.md): it overlaps with DS V4 (mHC with 4 streams, MLA layers, clamped SwiGLU, sigmoid router
with bias) and adds KDA linear attention in 34 of 45 layers, a relative of Strata's Qwen-Next gated delta net.

NOT measured yet: anything on the real model.

## s2 (2026-10-08): reference read, ops verified, phase-1 code written (not built)
- neurall/llama.cpp glm5-next.cpp read in full; its graph is the spec (ARCHITECTURE.md). mHC identical to DS V4
  (same fused ops); head = plain mean of the 4 streams (no hc_head). Dense FFN layers 0-2 ALSO use the SwiGLU clamp
  (swiglu_clamp_shexp[il] = 10 on every layer).
- Strata's ggml (DS4's third_party/llama.cpp) already has gated_delta_net (KDA), lightning_indexer, ssm_conv,
  dsv4_hc_*, swiglu_clamp: no new kernels for phase 1.
- The expert tier needs no changes (per-layer Q4_K/Q3_K/Q2_K + non-routed dense layers work since MiMo).
- Online (2026-10-08): no GLM-5.3-Flash-specific kernel projects worth reusing - neurall's fork (expert cache from
  llama.cpp PR #27861, MTP-only GGUF loading), LayerStoRm (expert streaming, multi-GPU SM120, PCIe-bound, 24.5 tok/s on
  2x5090+2x5080), DGPP (GB10-only engine). The KDA/MLA kernels everyone uses are llama.cpp's.
- MTP head GGUF (neuralll, 4.6 GB): depth-1 acceptance 70-75% but slower in the stock fork with 30% CPU experts; ours
  is planned Strata-fied (ARCHITECTURE.md). NOT measured here.

## s3 (2026-10-08 evening): CPU build clean; the mini fixture runs the whole dense path
- `build-glm` (CPU tree) configures and builds clean on the first try: `tools/glm/glm_dense.cpp` (881 lines) +
  `glm_generate.cpp` compile and link with no errors, `glm_generate_cpu` runs. First compile of the phase-1 code.
- New `tools/glm/make_mini_glm.py`: a 4-layer `glm5-next` GGUF (tiny dims, same structure as the real model) -
  blk.0 KDA+dense, blk.1 MLA+MoE, blk.2 KDA+MoE, blk.3 MLA+MoE - so all four attention x FFN combinations are covered.
  `glm_generate_cpu -m /tmp/mini-glm.gguf --backend cpu --experts cpu --ids 1,2,3 -n 3` loads and runs end to end (mHC,
  KDA conv + recurrence, dense MLA, dense FFN, sigmoid router + shared expert + `Ds4MoeTier`, mean head): finite logits,
  valid token ids, exit 0. No GPU, no real model needed.
- `--experts cpu` note: the tier reports `arena 0.0 GiB (0 experts)` and reads experts from the GGUF per token (file
  tier) - output is correct, but the CPU pool/arena is not populated; look at the tier's cpu-only path if CPU speed
  ever matters.
- `build-glm-gpu` (CUDA, arch 89, MMQ_KQUANTS) also builds clean, and `glm_generate --backend cuda --experts cpu` runs
  the same fixture with the same token stream as the CPU backend (44 5 9 5 44) - so on this fixture the CUDA path for
  mHC / `gated_delta_net` (KDA) / `flash_attn_ext` (MLA) / head agrees with CPU. GPU left at 8% / 0.6 GB; ComfyUI down.
- Reference fork: a **CPU-only** build exists (`~/AI/llama.cpp-glm53/build/bin/{llama-cli,llama-perplexity,
  llama-tokenize}`); no CUDA fork build yet (DS4 was running). Making the fixture fork-loadable surfaced two findings:
  (1) **FIXED** - MLA layers must have `glm5-next.attention.head_count_kv` = 1, not n_head: the DSA layer writes the
  shared latent (kv_lora), so a nonzero kv-head count sizes the kv cache `n_embd_k_gqa = head_k*head_kv` too wide
  (llama-kv-cache.cpp:209/233) -> `ggml_set_rows` `a->ne[0]==b->ne[0]` assert via `build_dsa_layer`. The fixture now
  writes `HEAD_COUNT_KV = [0, 1, 0, 1]` and that assert is gone. (Our engine is unaffected: `is_kda` is `head_count_kv==0`.)
  (2) **REMAINING** - the CPU-only fork then hits `binary_op: dst f32, src0 f32, src1 f16` in the fused lightning-indexer
  path (`cparams.fused_lid`, llama-context.cpp:249) - the fork's GLM indexer is CUDA-oriented, so a CUDA fork build (or
  the real model) is needed. Gate plan: fork `llama-perplexity` vs our `--ppl` on the same ids.
- Build rule learned this session: do NOT build while the `strata-ds4` session is running (compiles skew its timings).

## s4 (2026-10-08 night): FIRST real-model run (GLM-5.3-Flash 3.0-bit Q4K-attn) - runs, slow cold
- Blocker fixed: `native_expert_supported` (src/kernels/cuda/iq_kernels.cu) requires the down-proj type to be in
  `STRATA_D_FMTS`; Q4_K (12) was missing, so GLM's Q4_K expert layers (3-5) were refused at tier init
  ("native_expert_grouped has no kernel for layer 3's types 12/12/12"). Added `X(12)` - `Fmt<12>`, `vec_dot_q4_K_q8_1`
  and the row-bytes `case 12` already existed, so it is a one-token, backward-compatible fix.
- `glm_generate --backend cuda --experts gpu --slots auto --arena-gib 60 --vram-lru --ctx 2048 -n 16` (memguard 84 70):
  45 layers (34 KDA, 11 MLA), 288 experts top-8, vocab 154880; dense half on CUDA in **1.4 s**; 1269 resident slots;
  arena 60 GiB = 6917 experts + file tier 5179; tier load 55.9 s. Prefill (5 tok, decode-loop) 1.6 tok/s;
  **decode 1.75 tok/s** (15 passes / 8.55 s). experts **549 ms/token** - hit **18.6%** (gpu 6%, cpu 71.8%, pcie 9.6%,
  file-tier 2160), file reads 460 ms/pass: COLD cache, seeded in index order with **no routing profile**.
  attention+router 14.7 ms, finish 2.7 ms, head 0.75 ms. VRAM free at the end 1.67 GiB (tight).
- NOT a quality result: output was a degenerate repeat (token 500 x16) on arbitrary prompt ids and with **no
  abliteration LoRA** - runtime LoRA (ARCHITECTURE Decision 5) is **not implemented**. Next: (a) routing profile
  (`--dump-routes` -> `--profile`) to warm the expert cache, (b) implement the runtime rank-1 LoRA (Abliterix v2 as the
  baseline), (c) a real tokenized prompt + compare vs the fork.

## s5 (2026-10-08 night): abliteration adapter LOADER done + verified (NOT wired yet)
- `tools/glm/glm_lora.hpp` (header-only) parses a rank-1 GGUF LoRA into host floats: `exps(layer,mod,expert)` for the
  routed `ffn_{gate,up,down}_exps`, `solo(layer,mod)` for `ffn_down_shexp` / `attn_output`; `Lora1{a,b}` is one
  `y += b*(a.x)`. Verified against `lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf`:
  gate 26 layers a=4096 b=2048, up 26 (4096/2048), down 26 (2048/4096), down_shexp 27 (2048/4096),
  attn_output 30 (in per layer 8192 KDA / 16384 MLA -> 4096); spot values sane; shapes match the GGUF.
- Adapter truth (from the GGUF, corrects PLAN's older note): `alpha 1.0`, F16, 135 pairs, and it touches gate/up as
  well as down.
- NOT wired: no `--lora` flag; neither the `GlmDense` graph (`attn_output`, `ffn_down_shexp`) nor `Ds4MoeTier`
  (routed exps) applies it yet. That is the next slice (see RESUME_PROMPT.md Next).

## s6 (2026-10-08 late): abliteration LoRA wired into the DENSE half; cold baseline reproduced on a real prompt
- **`--lora <adapter.gguf>` is in** (`glm_generate.cpp`); `GlmDense` applies the adapter's whole-module rank-1 pairs in
  `build_attn`: `attn_output` on both the KDA and the MLA attention output, `ffn_down_shexp` on the shared expert's
  SwiGLU output, each as `y += mul_mat(B, mul_mat(A, x))` (A `[in,1]`, B `[1,out]`, F32, in their own backend buffer
  `Impl::lctx`/`lbuf`, uploaded in `init`). Without `--lora` every delta is NULL and the graph is bit-identical. Both
  trees built clean; `glm_dense.cpp` + `glm_generate.cpp` pass `-fsyntax-only`.
- **NOT wired: the routed experts** (`ffn_{gate,up,down}_exps`, layers 3-28) - where the adapter's abliteration mostly
  sits. The tier's expert math has no single hook: CPU = `pool->run_split_multi_native` (ds4_moe.cpp:1123, from
  `cpu_run` :1103), duplicated across `gpu_run`/`gpu_run_n`/`gpu_run_chunk` + the CUDA grouped/MMQ kernels. Needs new
  kernels + a tier interface - design in RESUME_PROMPT.md. **So the model is not yet abliterated.**
- **Real prompt, not arbitrary ids**: `bench/glm-2026-10-08/prompts/` (690-token `neutral6x`) tokenized with the HF
  `tokenizers` lib on `bench/glm-2026-10-08/upstream/tokenizer.json`. The fork's `llama-tokenize` REFUSES on
  `glm5-next` (`llama_init_from_model: glm5-next requires ctx_other`) - the HF lib is the working tokenizer.
- **Cold baseline reproduced** (690-token prompt, `--dump-routes routes.bin`, 512x42x8 u16 = 344064 B): prefill
  **1.74 tok/s**, decode **1.75 tok/s**, hit 19.0%, cpu 71.4%, pcie 9.6%, file tier 2152, file reads 461 ms/pass;
  experts 552 ms/token of which **cpu pool 525 ms** - and 461 of that 525 is FILE READS. The tier is I/O-bound, not
  compute-bound: the warm arena (no file reads) should cut the CPU pool toward ~65 ms and lift decode several-fold.
- **LoRA verified ACTIVE** (bench/glm-2026-10-08/run_lora.sh, real 21-token chat prompt, same ids both runs): the
  adapter changes generation (token 3 `29` -> `16343`) and `logits[0]` max|d| = **2.14** (mean 0.318), same argmax.
  So the loader -> graph path works end to end. (It only moves the dense-half modules; not an abliterated model yet.)
- **Warm cache: the routing profile BARELY helped** (`--profile routes.bin --vram-lru`): prefill 1.74 -> **1.91 tok/s**,
  decode 1.75 -> **1.88 tok/s** (+7%), hit 19.0 -> **29.7%** - but **file reads stayed 472 ms/pass** (cold 461) and the
  cpu pool stayed 476 ms/token. Why: the arena holds only ~55-57% of the 12096 (layer,expert) pairs (the 60 GiB
  budget), so ordering barely matters; worse, the 512-token profile block saw only **6551 distinct** experts, so the
  ranked arena stopped at 54 GiB instead of the 60 GiB budget. The `--dump-routes` sample (512 tokens = 1 block) is too
  small to rank the whole table.
- **Next speed levers (measured, not guessed)**: (a) a BIGGER arena - all ~12096 experts is ~72 GiB and would put file
  tier at 0 (MemAvailable was 85 GiB); (b) raise `--pcie` (0.25 default -> 0.55, DS4's value) so more misses compute on
  the GPU instead of the CPU pool; (c) lower `--pf-b` (1.43 -> ~0.7): 60 prefetch issued/token vs 22.9 useful = wasted
  DMAs; (d) **chunked MMQ prefill** - prefill is still the 1-token decode loop (1.9 tok/s), the single biggest number
  (DS4 392 tok/s chunked).


## s7 (2026-10-09): routing is FLAT -> chunked prefill, exclusive byte-sized cache, REAP pruning (bench/glm-2026-10-09/)
- **Why the cache could not help:** GLM's aux-free routing is near-uniform. Over 512 tokens the top 10% of experts per
  layer take 18% of routes; a 1265-slot VRAM cache ranked by frequency hits 10.6% on held-out tokens (= its share of
  experts). Locality is weak too: consecutive tokens share 0.83 of 8 experts/layer (random 0.22), a 64-token window
  is no better than random, and a 2-token verify touches 15.2 distinct experts -> MTP ~0.9x here (memory-bound).
  Decode speed = which fraction of the 12096 experts (107.2 GB) is resident at all, not which ones.
- **Lossless fixes (measured):** `--arena-skip-resident` (VRAM slots no longer duplicate arena experts) + `--arena-gib
  72`: decode 1.75 -> 3.95 tok/s. VRAM cache sized in BYTES (`slot_gib`, smallest blobs first; was a Q4_K-sized
  count): 951 -> 1851 slots at margin 3.5 GiB.
- **Chunked prefill** (`--prefill-chunk N [--chunk-mmq]`; GlmDense passes up to 4096 tokens: per-layer hand-off
  tensors and router blocks are now ONE shared set, chunk graphs share one re-planning allocator, `logits_rows` reads
  any 16 rows): 2000 tokens 1.7 -> **80-90 tok/s** (1024-token chunks, file tier present). Needs `--vram-margin 5.5`
  at chunk 1024 + MMQ (3.5 OOMs in `m_xg`).
- **Open gate:** chunked vs decode-loop ppl on `neutral` (115 tok): loop 20.4601 (bit-identical over 2 runs), chunked
  19.3327 (-0.057 NLL); first-token logits max|d| 1.51, same argmax. Needs the fork oracle (or MMQ off) to say which
  path drifts.
- **REAP pruning** (`--saliency F` in chunked prefill: tier reads back every entry's expert output, accumulates
  w*||f(x)||; `tools/glm/reap_prune.py`; `--prune F` masks experts out of routing AND out of arena/VRAM). Calibration
  4 x 2000 tokens (code, prose, zh/ja, chat). Held-out 2000-token evals, chunk 1024, margin 5.5:

  | config | eval_code ppl | eval_chat ppl | prefill tok/s | decode tok/s | file tier |
  |---|---|---|---|---|---|
  | base (no prune) | 3.676 | 5.633 | 86-90 | 3.72 | 1335 |
  | REAP 14.2% (fit92) | **3.576** | **6.672 (+18%)** | 150 | 7.2 | 145 |
  | random 14.2% (control) | 3.909 (+6.3%) | - | 151 | 7.0 | 145 |
  | REAP 25% | 3.597 | - | **175** | **9.3** | 0 |

  REAP's choice matters (random is 9% worse than REAP on code), and code survives even 25%. **But the chat eval
  (mostly German README text, a language NOT in calibration) lost 18% at 14%**: a static prune list holds only for
  calibrated domains. Next: soft pruning (penalty, not mask) + broader calibration (more languages/domains).
- Process lesson: `pkill/pgrep -f <pattern>` matched my own shell twice (exit 144). Kill exact PIDs from a pid file.
- **Chain 3 (2026-10-09 05:25-05:33):** broader calibration (+romance READMEs, python, json/tool-call; still NO
  German) made the German-heavy chat eval WORSE: 7.28 (4-domain list 6.67, base 5.63). German relies on its own
  experts that a broad non-German calibration ranks low -> pruning is domain-specific; build the prune list from the
  mix actually used (Mal's idea: English + Chinese + code + tool calls) and keep several lists.
- **skip_miss (decode, REAP 25% prune, fully resident, `--vram-margin 1`, decode-loop ppl over 300 eval_code
  tokens):**

  | --skip-miss | decode tok/s | ppl | skipped/token |
  |---|---|---|---|
  | 0 | 10.38 | 11.159 | 0 |
  | 0.05 | 11.37 | 11.465 (+2.7%) | 31 |
  | 0.10 | **16.38** | 11.604 (+4.0%) | 115 |

  Decode 1.75 (yesterday) -> 10.4 lossless-ish (prune 25%) -> 16.4 with skip 0.10. The margin-1 VRAM cache (hit
  35.8% vs 29% at margin 5.5) is worth ~1 tok/s: re-seeding the cache bigger after a chunked prefill is a TODO.

## s8 (2026-10-09 morning): lightning indexer IN (long context), shared-allocator bug fixed
- **Lightning indexer** (`GlmDense`, MLA layers; glm5-next.cpp build_kpool_select semantics): pooled-key cache
  `ipool` F16 [128, ctx/4+1] (pool p = tokens 4p..4p+3: LayerNorm(indexer.attn_k x) mixed by a per-channel softmax over
  the 4 of compressor_gate x + ape), carry `itail` [256, 3] across passes; score = ggml_lightning_indexer (128 x 32 heads,
  wmma), pool visible iff 4p+3 <= t (step mask on GPU), top-512 pools (CUB top_k) + incomplete tail, dead slots to
  per-slot dump rows, F16 additive mask via set_rows -> flash_attn. Below idx_top_k positions the dense causal path
  runs (exact and cheaper); the pool cache is updated on every pass either way. `--allow-long-ctx` = force dense.
  Context is no longer capped at 2051: `--ctx N` sizes the latent cache (N=524288 -> ~6 GB VRAM).
- **CPU gate** (`bench/glm-2026-10-09/gate_indexer.sh`, mini fixture, idx_top_k 8): G1 indexer == dense at 11 tokens
  (max|d| 0), G2 differs at 30, G3/G4 chunk 7 / chunk 5 == one-token loop (0), G5 dense chunk == dense loop (0); clean
  under AddressSanitizer (`build-glm-asan`). Mutation (tail off by one) -> G1 4.49 + G4 0.354 FAIL. Indexer SCORING is
  not gated against the fork yet (`~/AI/llama.cpp-glm53` patched: `GLM_NO_FUSED_LID=1` disables its fused indexer so
  its CPU build can run the fixture - rebuild + compare = TODO).
- **Real model, 2300-token prompt crossing 2048** (REAP 25%, chunk 1024): indexer ppl 4.3780 vs forced dense 4.3765
  (+0.03%; ~250 of 2300 positions dropped per token at the end), prefill 156 tok/s, decode 8.5 tok/s (margin 5.5).
- **BUG FIXED - shared gallocr stale pointers:** gallocr treats a tensor with ->data set as externally allocated, so
  after the shared chunk allocator's buffer grew, earlier chunk graphs kept pointers into freed memory (CPU: heap
  corruption with chunks > 4 tokens; GPU: silent). `Impl::alloc` now nulls data/buffer of every graph tensor not in a
  persistent buffer before re-planning (pointer compare only - the stale buffer must not be dereferenced). The
  chunked-prefill numbers of s7 ran with this bug: neutral ppl moved 19.33 -> 18.47 after the fix (loop 20.46) - the
  remaining chunk-vs-loop gap is GPU-only (CPU fixture is bit-exact) -> MMQ vs CPU-pool expert numerics or batched FA;
  **open, needs the fork oracle**. Prune-sweep conclusions (relative) stand; absolute chunked ppl should be re-taken.
- **BUG FIXED - flash_attn mask stride:** CUDA FA with 512-wide heads needs every mask stride % 16 bytes; the indexer
  mask rows are cap + nsel halfs -> padded to a multiple of 8.
- **524288-token context allocates** (11.15 GiB free after the dense half -> 734 expert slots at margin 5.5, REAP 25%
  fully resident, arena 69.2 GiB). 50K-token prefill + decode at that depth: see s9 / bench/glm-2026-10-09/long50k.log.
- From strata-ds4-gpu (DS4 measurements, to port): soft prune `--prune-penalty 0.5` + `--arena-adapt` (pruned experts
  read from NVMe on demand into an LRU arena slot) keeps languages: ppl +0.3% overall, German +3.9% (hard +72%), decode
  +10%. GLM has both flags; next GPU block tests them with an en+zh+code+tools calibration list.

## s9 (2026-10-09 06:08): 524288-token context, 50K-token prefill + decode at 50K depth
- REAP 25% prune, chunk 1024, margin 5.5 (734 VRAM slots): **prefill 50000 tokens in 217.8 s = 229.6 tok/s**
  (attention+router 43 s, experts 169 s); **decode at 50K depth 8.23 tok/s** (attention+router 20.6 ms/token vs
  ~22 at 2K) - long context costs almost nothing in decode. Log: bench/glm-2026-10-09/long50k.log.

## s10 (2026-10-09 16:54-17:03): soft prune on GLM - penalty must be scaled to the router (bench/glm-2026-10-09/chain5.sh)
- **Penalties don't transfer from DS4.** GLM selects on sigmoid(logit) + exp_probs_b: scores in [0,1], and the bias
  spread across experts is only p10-p90 0.03-0.13 (blk.3 0.035 ... blk.44 0.131). DS4's 0.5 would be ~a hard prune
  here, so the sweep is 0.02 / 0.05 / 0.10.
- List: REAP 25% (72/layer) from en+zh+code+tools calibration (`sal-cal_{code,prose,multi,chat,python,json}`,
  12000 tokens; romance + German held out) -> `prune-ezct-0.25.txt`; control `prune-rand-0.25.txt` (72/layer, seed 41).
  Soft runs add `--arena-adapt`. Held-out 2000-token evals, chunk 1024, margin 5.5:

  | config | eval_chat ppl (German, uncalibrated) | eval_code ppl | prefill tok/s |
  |---|---|---|---|
  | base, no prune (s7, ran WITH the gallocr bug) | 5.633 | 3.676 | 86-90 |
  | hard REAP 25% | 8.949 (+59%) | **3.564** | 161 (code) |
  | hard random 25% (control) | - | 4.289 (+20% vs REAP) | 170 |
  | soft 0.02 + adapt | **5.457** | - | 89 |
  | soft 0.05 + adapt | 5.516 | 3.583 | 97-102 |
  | soft 0.10 + adapt | 5.528 | - | 114 |

  Soft prune gets back all of the uncalibrated-language loss (8.95 -> 5.46-5.53) while code stays within 0.5% of
  hard. The REAP choice matters (random costs +20% on code). **Caveat:** the base row predates the allocator fix
  (s8 moved chunked ppl by ~4%), so "soft < base" isn't established. Re-take base on the next block.
- **Decode** (eval_code300, -n 32, margin 1, soft 0.05 + adapt): **7.57 tok/s** vs 10.38 for the hard 4-domain 25%
  list (sm0). Experts 114 vs 78 ms/token: 282 file-tier reads, 211 arena swaps, 43 ms/pass file reads, cpu-pool wall
  98 vs 73 ms. Over 32 tokens that is adapt warming up / thrashing. Prefill also drops 161 -> ~100 tok/s.
- Next: re-take the no-prune base; longer decode (256+ tokens) to see whether adapt settles; an admission gate (DS4
  research, Infernix: promote a file-read expert only if its decayed hit count beats the coldest arena expert, not
  on first read); try soft 0.02 on code + decode.
- Process: two copies of chain5 overlapped for ~1 min (a queued job still in its `sleep 120` was missed by the
  process check). memguard.sh already takes ds4-gpu.lock for each run, so an outer `flock` on the same lock
  deadlocks: never wrap memguard runs in it. c5-hard-chat.log was truncated by the duplicate's start; its ppl
  (8.9493) came from the monitor (`c5-hard-chat.note`).

## s11 (2026-10-09 17:12-17:21): clean base; winning back soft-prune decode (bench/glm-2026-10-09/chain6.sh)
- **Base measured again after the allocator fix** (no prune, chunk 1024, margin 5.5): eval_chat **5.547** (s7 said 5.633),
  eval_code **3.668** (3.676). Against it: soft 0.02 chat 5.457 (-1.6%), soft 0.05 chat 5.516 (-0.6%) / code 3.583
  (-2.3%), hard 25% code 3.564 (-2.8%) but chat 8.95 (+61%). **Soft prune 25% costs no quality vs unpruned** on
  either the calibrated domain or the held-out language, and is slightly better on these evals.
- New tier options (tools/ds4/ds4_moe.{hpp,cpp}, glm_generate flags; both default off):
  - `--skip-file T`: like skip_miss, but only for experts the arena does not hold (NVMe file tier = the soft-pruned
    ones); dropped if their weight is < T x the token's weight sum.
  - `--arena-admit H`: arena_adapt admission gate. Per-(layer, expert) heat with a half-life of H tokens; a file
    read is promoted only if its heat is >= 2 and beats the coldest arena expert (which becomes the victim).
- Decode (eval_code300: 300-token decode-loop prompt = ppl, then 64 tokens timed; margin 1; ezct 25% list):

  | config | decode tok/s | ppl (299 tok) | file reads | file ms/pass | experts ms/token |
  |---|---|---|---|---|---|
  | hard 25% | **9.52** | 11.974 | 0 | 0 | 86.2 |
  | soft 0.05 + adapt (s10, -n 32) | 7.57 | 11.764 | 282 | 43.4 | 114.1 |
  | soft 0.05 + adapt + admit 64 | 7.20 | 11.860 | 709 | 51.8 | 121.0 |
  | **soft 0.05 + adapt + skip-file 0.10** | **8.46** | **11.554** | 249 (494 skipped) | 21.3 | 98.8 |
  | soft 0.05 + adapt + admit 64 + skip-file 0.10 | 8.23 | 11.676 | 304 (580 skipped) | 26.3 | 102.4 |

  The gate is a loss: GLM's routing is flat, so the soft-pruned experts that do win come back often enough that
  refusing to cache them means re-reading them from NVMe (709 reads vs 282). Promote-on-first-read stays the default.
  skip-file 0.10 halves the NVMe time and gets back 0.9 of the 1.95 tok/s soft prune costs, with no ppl loss (the
  299-token decode-loop ppl spread here is ~3%, i.e. noise level; it is not a gain).
- Remaining gap to hard: ~21 ms/pass of file reads + a slightly busier CPU pool. Next: skip-file 0.15/0.20, penalty
  0.10 (fewer soft-pruned wins), and skip-file + skip-miss 0.05 to buy back speed elsewhere; chunked prefill
  (161 -> ~100 tok/s under soft prune) has no skip yet; run_chunk is the place to add it.

## s12 (2026-10-09 17:25-17:37): soft prune at hard-prune speed - skip-file sweep (chain7) + prefill skip (chain8)
- New: `--skip-file-prefill T` (Ds4MoeConfig::skip_file_chunk): in a prompt chunk, a file-tier entry below T x its
  token's weight sum is redirected to the token's heaviest expert with weight 0 (contributes nothing; an expert left
  without entries is never read). Kernels untouched.
- **Decode** (chain7; eval_code300, 64 tokens timed, margin 1, ezct 25% list, soft = penalty 0.05 + adapt):

  | config | decode tok/s | file ms/pass | file-tier skipped |
  |---|---|---|---|
  | hard 25% (s11) | 9.52 | 0 | - |
  | hard 25% + skip-miss 0.05 | 10.77 | 0 | - |
  | soft + skip-file 0.10 (s11) | 8.46 | 21.3 | 494 |
  | **soft + skip-file 0.15** | **9.82** | 7.9 | 1023 |
  | soft + skip-file 0.20 | 10.10 | 4.1 | 1087 |
  | soft penalty 0.10 + skip-file 0.10 | 8.89 | 17.1 | 224 |
  | soft + skip-file 0.10 + skip-miss 0.05 | 8.91 | 21.6 | 568 |

  The 299-token decode-loop ppls (10.9-11.8) swing ~7% run to run: too small to rank quality. Quality is judged
  at full size below.
- **Quality at full size + prefill** (chain8; 2000-token held-out evals, chunk 1024, soft 0.05 + adapt; the prefill
  skip applies the same rule, so it is also the quality proxy for decode skip-file):

  | config | eval_chat (German) | eval_code | prefill tok/s (chat / code) |
  |---|---|---|---|
  | base, no prune (s11) | 5.547 | 3.668 | ~88 |
  | hard 25% | 8.949 (+61%) | 3.564 (-2.8%) | 161 (code) |
  | soft, no skip (s10) | 5.516 (-0.6%) | 3.583 (-2.3%) | 97 / 102 |
  | **soft + skip-file-prefill 0.15** | **5.665 (+2.1%)** | **3.605 (-1.7%)** | **130 / 140** |
  | soft + skip-file-prefill 0.20 | 5.860 (+5.6%) | 3.625 (-1.2%) | 143 / 157 |

- **Recommended config:** `--prune prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15` -> decode ~9.8 tok/s (= hard prune), prefill 130-140 (hard 161), German +2% vs
  unpruned instead of +61%, code still better than unpruned. 0.20 buys ~10% more prefill speed for another
  +3.5% on German. To win back the rest: skip-miss on VRAM misses (hard + 0.05 -> 10.77), bigger chunks, VRAM
  re-seed after prefill.

## s13 (2026-10-09 17:39-17:55): PCIe share, skip-miss on top, decode-path quality gate, DwarfStar (chains 9-10)
- **--pcie sweep** (recommended soft config, eval_code300, 64 tokens timed): 0.25 9.82 | 0.35 **9.97** | 0.45 9.12 |
  0.55 9.86 | 0.65 9.14 tok/s. Flat within noise from 0.25 to 0.55. The PCIe DMA reads the same host RAM the CPU pool
  reads, so moving work to PCIe slows the pool (0.45: cpu pool 77 ms for fewer experts). At 0.65 PCIe is the long pole
  (65 ms). Keep 0.25-0.35.
- **+ --skip-miss 0.05: 10.82 tok/s** (+10%; tier wall 74.8 vs 81.5 ms).
- **Decode-path quality gate** (new): eval_chat (German, 2000 tokens) as a decode-loop ppl, so every token goes through
  run() and skip-file/skip-miss act exactly as in decode (~3.5 min/run at ~9 tok/s):
  recommended (`--skip-file 0.15 --pcie 0.35`) **5.940**, + skip-miss 0.05 **5.807**. Skip-miss costs no measurable
  quality. Decode-loop ppl is not comparable with chunked ppl (the open chunk-vs-loop gap, s8); compare loop to loop
  only. The 300-token loop ppls also do NOT track the CPU/PCIe split (0.55 lowest, 0.65 highest) -> few-% run noise.
- **DwarfStar** (antirez/ds4, ~/AI/ds4-ref fc80bd6) runs GLM-5.3-Flash on Metal/CUDA/ROCm: M5 Max 128 GB, Q4_K SSD
  streaming: prefill 121 tok/s, decode 11.9-14.9. It **refuses our GGUF** (expects 46 blocks with the MTP layer
  inside; ours keeps MTP in a separate file; its CUDA path also has no Q3_K). Head-to-head would need its own
  `glm53-q2` (90 GiB). Ideas taken: (1) prompt/KV cache across turns (`--kv-disk-dir`; full GLM 5.3: 16-token
  append 30.8 -> 2.9 s). GLM's carried state is all in GlmDense's sbuf (KDA conv+S per layer, MLA kvc rows < pos,
  ipool rows < pos/4, itail) -> a snapshot/restore is straightforward; worth it once GLM is served (glm_generate is
  one prompt per process). (2) Prefill streaming = read layer L+1's experts while L computes: our chunk_prestage
  does that for arena experts but is OFF with arena_adapt (soft prune) - a lever for prefill. (3) Its VRAM cache
  "protects every hit before evicting for a miss" - n/a, ours is static with flat routing.
- **Best config now:** `--prune prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35` -> decode ~10.8 tok/s (hard 25% 9.52; unpruned ~3.7),
  prefill 130-140 tok/s, German chunked ppl +2% vs unpruned.

## s14 (2026-10-09 17:56-18:12): prefill - prestage, bigger chunks; chunked ppl depends on residency (chains 11-12)
- `chunk_prestage` (DMA layer l+1's arena experts while layer l computes - DwarfStar's streamed-prefill idea) is now
  allowed with arena_adapt (prompt chunks never swap arena slots; decode's direct arena reads sync s_cp first). Its
  device half was sized to a whole layer at the largest blob (288 x 15.2 MB = 2 x 4.4 GB -> c_stage OOM even at
  margin 8.5); now capped by `STRATA_PRESTAGE_MIB` (default 1536) and a layer that does not fit is prestaged in part.
  **Result: no gain** (margin 6.5, soft+sfp0.15, code: 140.1 without, 139.3 with). GLM prefill is not waiting on
  the arena DMA. Leave it off.
- **Chunk 2048** (margin 8, + prestage): **160 tok/s** vs 140 at chunk 1024 (+14%), code ppl 3.626. Needs margin ~8
  (chunk 2048 buffers), which costs decode VRAM slots in the same process - only worth it with a VRAM re-seed /
  elastic cache after prefill, or for long prompts.
- **Chunked ppl is repeatable to ~0.1%** (chain12, chain8's soft+sfp0.15 code config x3: 3.6003 / 3.6034 / 3.6034;
  chain8: 3.6047), **but it moves 1-2% with the VRAM margin**: code 3.587 (margin 8.5) / 3.603 (5.5) / 3.620 (6.5);
  chat 5.665 (5.5) / 5.769 (6.5, prestage); hard code 3.564 (5.5) / 3.606 (6.5). The margin only changes which
  experts are VRAM-resident (grouped MMVQ, q8_1 activations) vs streamed (MMQ) -> **the two expert paths differ
  numerically by ~1-2% ppl**. Same family as the open chunk-vs-loop gap (s8). Consequences: (1) compare chunked
  ppls only at the same margin; (2) "soft prune +2% vs base" (s12) is within this path spread; the robust result
  is soft ~= base and hard = +61% on German; (3) the fork oracle (RESUME next step) should say which path is right.

## s15 (2026-10-09 18:13-18:24): real-use numbers + elastic VRAM cache (chains 13-14)
- **Real use = chunked prefill then decode in ONE process, no --ppl** (the --ppl head over every position was
  costing prefill a lot): soft best config, eval_code 2000-token prompt, 64 decoded tokens:

  | margin / chunk | --vram-grow | slots prefill -> decode | prefill tok/s | decode tok/s |
  |---|---|---|---|---|
  | 5.5 / 1024 | - | 1493 | 177.9 | 8.83 |
  | 8 / 2048 | - | 1143 | 229.8 | 8.74 |
  | 5.5 / 1024 | 1 | 1495 -> 1974 (+479 in 151 ms) | 177.7 | **9.80** |
  | **8 / 2048** | **1** | 1145 -> 1909 (+764 in 241 ms) | **217.9** | **9.82** |

  The decode numbers of s11-s13 (10.82) were at margin 1 after a 300-token decode-loop prompt (which also warms
  arena_adapt); in real use the prompt's margin shrank the VRAM cache and cost ~2 tok/s.
- **Elastic cache** (`--vram-grow KEEP_GIB`; Ds4MoeConfig::slot_gib_max + Ds4MoeTier::grow_cache): the cache is
  opened at the decode size (free - KEEP) as CUDA VMM segments (64 MiB) in the ranked order, mapped/seeded only up
  to the prompt's budget (free - margin), and after the prompt (release_chunk) the rest is mapped and filled with
  the next ranked experts from their arena copies (the arena still holds them: arena_skip_resident skips only the
  seeded part). Chunk 2048 now costs decode nothing.
- Left on the table: GlmDense keeps the chunk allocator's compute buffer (allo_big) and NT-sized hand-off state
  after the prompt (1909 slots vs 2091 at margin 1). Freeing/re-creating allo_big needs care: chunk Vars cache
  the allocator pointer.
- **Best real-use config now:** `--prune bench/glm-2026-10-09/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt
  --skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35 --prefill-chunk 2048 --chunk-mmq
  --vram-margin 8 --vram-grow 1 --arena-skip-resident --arena-gib 72 --slots auto` -> prefill ~220-230 tok/s,
  decode ~9.8 tok/s (session start: 2.1 tok/s prefill-loop / 3.7 decode unpruned; this morning hard prune 175 / 9.3
  with German +61%).

## s16 (2026-10-09 evening): serving GLM behind Strata's API server + Quetza wrapper - BUILT, NOT YET WORKING END TO END
Why: Strata (~/AI/Strata) serves only Qwen3.8-Flash-Next; GLM/DS4 have their own engines, which were one-shot CLIs.
Strata's server talks to its engine over a line protocol, so GLM now speaks it.
- **Engine:** `glm_generate --serve` (tools/glm/glm_generate.cpp, serve block before the prefill): stdin reader
  thread; `GEN <max_new> [temperature= top_p= top_k= min_p= seed=] <ids csv>` -> `RESUME n`, `PP read total ms tok/s`,
  `T id`, `DONE gen prompt prompt_ms decode_ms finish 0 0 reused hits lookups 0 file 0 read`; `STOP` mid-decode
  (finish=cancel); `QUIT`. Sampler `sample_p` (temp/top-k/min-p/top-p). Stops on `--stop` ids.
- **Multi-turn reuse:** `GlmDense::snapshot()/restore()/snapshot_pos()` copy only the recurrent state (KDA S + conv,
  MLA itail) + position; MLA latent rows and complete indexer pools below the position are never rewritten, so the
  rewind is exact. Snapshot at the end of every prompt; a request that extends what was fed continues, one that
  shares the last prompt restores the snapshot, else reset. Untested on the real model.
- `--serve` forces `--vram-grow` off (the elastic grow is one-shot; every request's prefill needs the chunk VRAM).
  TODO: shrink the cache back before each chunked prefill to keep the grow in serve mode.
- **Frontend shim** tools/glm/serve/: `export_tokenizer.py` (GGUF -> vocab.json/merges.txt/token_type.json/
  chat_template.jinja; exported to /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/tokenizer), `glm_tok.py` (Strata BPE +
  llama.cpp CHATGLM4 pre-split; **verified exact** on bench/glm-2026-10-08/prompts neutral + chat .ids),
  `serve_glm.py` (runs Strata's serve.server with GLM's tokenizer, stop ids <|user|>/<|observation|>/<|endoftext|>,
  GLM tool calls `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`, tool-arg streaming off),
  `glm-engine.sh` (memguard's protections in the FOREGROUND - memguard.sh backgrounds its command, which gives it
  /dev/null as stdin; holds ds4-gpu.lock while the server runs), configs `strata-glm-unc.json` (abliteration LoRA,
  dense half only; port 8140) / `strata-glm.json` (stock, 8141): soft prune best config, ctx 524288, chunk 1024,
  margin 5.5, sampling temperature 1.0 / top_p 0.95.
- **Wrapper:** `~/.local/bin/strata-glm-quetza [--stock] [--port=N]` (same shape as strata-quetza).
- **Status: WORKING end to end (verified 2026-10-09 ~18:5x).** The "first start died silently" was NOT the engine:
  the engine reached `READY` in 62 s (738 resident slots at ctx 524288). Strata's `serve/server.py:490` does
  `asked = next((int(args[args.index(k) + 1]) for k in ("--batch", "--slots") if k in args), 0)` and our config
  passed `--slots auto` -> `ValueError: invalid literal for int() with base 10: 'auto'` in the SERVER. Its traceback
  is in the SERVER log, not the engine log - which is why it looked silent. **Fix: drop `--slots` from the configs**
  (`glm_generate`'s default `slots=0` IS auto - tools/glm/glm_generate.cpp:47,82). No engine change needed.
- **Verified through the API** (server on 8140; `/health` ok, max_context 524288): plain `/v1/chat/completions`
  (finish_reason stop, content "4"); `/v1/messages` tool call -> clean `tool_use` `{"city":"Melbourne"}` with
  stop_reason tool_use; 2-turn reuse -> engine log `46 tokens (31 reused)`, API `cache_n 31`; streaming SSE (text)
  and a streaming tool call (`input_json_delta` accumulates the JSON, stop_reason tool_use). Decode **~8-9 tok/s** at
  512K ctx (738 slots - the 512K latent cache costs slots vs the 10.8 of s11-s13).
- **Still open:** `--serve` forces `--vram-grow` off (every request's chunked prefill needs the chunk VRAM); TODO
  shrink-before-prefill to keep the grow. Wrapper `~/.local/bin/strata-glm-quetza` (uncensored LoRA on 8140 / stock
  on 8141) is written and its endpoint verified by curl, but NOT yet launched as a full Quetza session.
- **DS4:** `ds4_generate --serve` does NOT exist yet in `~/AI/Strata-DS4` (grepped 2026-10-09 18:5x); that tree is
  the `strata-ds4-gpu` session's - do not edit it. Relay ask pending (no peer connected at 18:5x).

## s17 (2026-10-09 19:10-22:25): dense abliteration verified; routed-expert LoRA (WIP, NOT working)
- **Dense-only abliteration is active and looks sufficient behaviorally.** The adapter (`gcsa-abliterix v2`) has 135
  A/B pairs; GlmDense applies the 57 "solo" ones (attn_output 30, ffn_down_shexp 27). Same prompt +- adapter moves the
  first-token logits **max|d| 2.52, mean|d| 0.40** (same argmax) - so the dense half is live on the real model.
- **Refusal battery** (bench/glm-2026-10-09/ablate_probe.py): uncensored server **6/6 complied, 0 refused**
  (profanity / roast / violence / crime / drugs / controversial-idea). Caveat: no stock baseline was run (time), so
  this measures "the unc model does not refuse" - it does not yet prove the base refuses. Pass `reasoning_effort: low`
  (chat_template_kwargs): the default `Reasoning Effort: max` spent >1024 tokens thinking on the roast prompt and never
  reached an answer (1 earlier probe came back UNRESOLVED for that reason).
- **Metrics parity: already a superset of llama.cpp.** `/metrics` = engine facts (experts_vram, arena_experts,
  max_context), live request state, cumulative totals, GPU util/temp/power/PCIe gen+width, a history ring,
  conversation-cache counters; `/props` = chat template + defaults. Nothing to add for "same sort of metrics".
- **Routed-expert LoRA (NEW: `--lora --lora-exps`): BUILT, NOT WORKING.** Interface `NativeExpertLora`
  (include/strata/kernels/iq_kernels.hpp) + `lora_gu_kernel`/`lora_down_kernel` (src/kernels/cuda/iq_kernels.cu),
  called from `native_expert_grouped` after the GU launch (before SwiGLU) and after the down launch; `Ds4MoeLoraHost`
  (tools/ds4/ds4_moe.hpp) + device upload in the tier + per-entry expert ids at the `gpu_run` and `gpu_run_chunk`
  sites; glm_generate flattens the adapter's `ffn_*_exps` into it. Two blockers:
  1. **NaN.** With `--lora-exps` (pcie 1.0, chat.i32) the run finished rc 0 but dumped **all-NaN logits** -> the
     per-entry expert id is wrong somewhere. Suspect: in the PCIe launch group `dst[i]` may already be an expert id,
     not the routing index the decode site assumes (`exp[i] = ids6[dst[i]]`); an out-of-range id reads past the A/B
     arrays. Check the PCIe `dst` build in gpu_run before anything else.
  2. **Coverage is partial even when correct.** Only the GPU grouped path is instrumented; the CPU pool
     (`native_gu_rows`/`native_down_rows`) and the MMQ chunk path are not, and the tier sends a large share of experts
     there (the A/B showed cpu pool ~390 ms/token). So within one token some experts would be ablated and some not -
     the all-or-nothing problem. Cheapest fix: when `--lora-exps`, force `pcie_frac = 1.0` AND require the whole routed
     set in the arena (no file tier), so every expert goes through `native_expert_grouped`; else wire the CPU/MMQ paths.
  Also the fp32 upload is ~550 MB VRAM (26 layers x 288 experts x 6 arrays) - store fp16 to halve it, and the configs
  never enable `--lora-exps` (the default path is bit-identical: `lora == nullptr`).

## s18 (2026-10-09 20:2x-20:4x): decode 2.1x for ~0.2% ppl (--skip-miss 0.15); reuse diagnostic; metrics
- **`--skip-miss` doubles decode at ~no cost on code.** Serving flags, prefill-chunk 1024, ctx 524288, arena 72 GiB:
  | `--skip-miss` | decode | code ppl (eval_code, 1999 pos) |
  | 0.05 (shipped) | 9.72 tok/s | 3.5989 |
  | **0.15** | **20.65 tok/s** | 3.6066 (**+0.21%**) |
  | 0.15, ctx 32768, margin 1 | 22.19 tok/s | - |
  So skip-miss is the lever, not context (32K buys only +7%).  `--fast` wrapper mode ->
  tools/glm/serve/strata-glm-unc-fast.json (skip-miss 0.15).  Chat/prose ppl gate still running.
- **Multi-turn reuse was silently broken - now diagnosed.** `dense.snapshot()` can FAIL (its device copy), and the
  serve loop then left snap_len = -1: the "shares the last prompt" restore never fired, so EVERY follow-up refilled
  from 0 (real traffic: 19160 then 19317 tokens, both "(0 reused)", ~117 s each - the "taking a while per prompt").
  The engine now logs the decision (`serve: reuse lcp=.. hist=.. snap=.. -> reused=..`) and a snapshot failure; with
  a working snapshot a follow-up reuses (verified: lcp 423 >= snap 421 -> reused 423; turn 6.5 s -> 3.3 s).  Note:
  any request from a DIFFERENT conversation resets the one state, so interleaved clients thrash it (as here).
- **Metrics**: `/metrics` now carries the live `thinking`/`output` text; `/slots` is a full view (phase,
  prompt/generated, tok/s, reasoning+content); `/metrics?format=prometheus` (or `Accept: text/plain`) emits
  llama.cpp's names (`llamacpp:*`) plus `strata:*` (prefill/decode rate, phase, expert tiers, VRAM).  Committed in
  ~/AI/Strata/serve/server.py (9bb510c8); that checkout is the UPSTREAM (Niko1221) - not pushed.  Strata-GLM's own
  serve/server.py is a diverged copy the shim does not import.
- **Ablation status**: the dense half (57/135 pairs) does NOT stop hard refusals - the keylogger prompt refuses
  ("I can't help with that ... spyware ... consent") at the default effort AND at effort=low.  The routed-expert
  LoRA (`--lora-exps`) is required.  Its NaN is NOT the per-entry expert-id mapping (verified: both dst sites pass a
  routing index, so `ids6[dst[i]]` is right) nor the A/B shapes (gate/up 4096->2048, down 2048->4096, matching the
  tier's n_embd 4096 / n_ff 2048) - re-test with the current binary.  The CPU pool and the MMQ chunk path are still
  uninstrumented (all-or-nothing: a large share of experts are computed there).

## s19 (2026-10-09 late): routed-expert LoRA WORKS (all paths, consistent); the refusal prompt loops
- **Three bugs found and fixed in `--lora-exps`, in order:**
  1. **NaN**: the down correction read the float `h` buffer, but the default SwiGLU path (`sw_v1=false`) writes
     only `hq` (quantized) - `h` was garbage.  Fix: `sw_v1 = v1 || (lora && lora->a_d)` materialises `h`.
     (Bisected with GLM_LORA_NO_GU / GLM_LORA_NO_DOWN: GU-off still NaN, DOWN-off clean.)
  2. **The chunk path silently did nothing**: `kMaxEnt` is only 32 (kMaxTok 4), so a 1024-token chunk's 8192
     entries failed my `NE <= kMaxEnt` guard.  Fix: a dedicated `gp.c_exp` sized to the chunk's `ne`.
  3. **Prefill/decode mismatch (the model-breaking one)**: the config's `--chunk-mmq` sends prefill chunks through
     the MMQ path, which is NOT instrumented -> un-ablated prefill + ablated decode = a broken model (benign
     prompts looped: "write one sentence about the ocean" produced nothing, "2+2" still worked).  Fix: `--lora-exps`
     forces `chunk_mmq = false` (and `pcie_frac = 1.0` so every miss takes the one instrumented path).
- **Now mechanically correct**, verified: code ppl dense 3.5628 -> dense+exps **3.5967 (+0.95%)** (chunk and decode
  consistent); benign prompts answer normally ("2+2" -> "4"; ocean -> a real sentence); the expert deltas move the
  first-token logits (max|d| 1.61 vs dense-only).
- **But the hard-refusal prompt does not comply - it loops.** The keylogger prompt (effort=low) runs ~8-10k chars of
  reasoning and never answers, at every budget tried (500/1800/2200) and through the real harness (QuetzaCodetl -p
  reports "exceeded the 1500 output token maximum" after 14m21s).  Different from the dense-only model, which
  refused crisply.  So the routed deltas DO change the refusal behaviour, but the result is a deliberation spiral,
  not compliance - an ablation-quality question, not a plumbing one.  Unverified: whether other refusal prompts
  comply, and whether a bigger budget eventually terminates.
- **Cost of the ablated mode**: `--chunk-mmq` off drops prefill (220 -> ~130 tok/s); `pcie 1.0` + the whole set in
  the arena are required (a file-tier expert is computed by the CPU pool and would not be ablated).

## s20 (2026-10-09 late): THE ABLITERATION LOOP - CAUSE FOUND AND FIXED (the skip heuristics drop experts whose delta is then never applied)
Context: s19 left the routed-expert LoRA mechanically working (both paths, +0.95% code ppl) but the hard-refusal prompt
"looped" (8-10k chars of reasoning, never answered, through the harness too). Mal: treat the loop as a BUG, not a weak
adapter. This session found and fixed the cause on the GPU (a window the ds4 session granted).

**What was actually happening** (probe of the ablated server's /v1/chat/completions, capturing `reasoning_content` +
`content`): the keylogger prompt at the config default (temp 1.0, top_p 0.95) does NOT just "think long" - the CONTENT
DEGENERATES into a repetition loop ("... two related fictional universes:" and, with sane sampling, "from ctypes import
LPVOID, LPVOID, LPVOID, ..." x107) and never terminates. So it is model degeneration, not a budget problem.
- Control, same prompt+sampling, STOCK model: coherent refusal trajectory ("This is a request for malware/spyware code
  ... framed as an attack tool"), top-6-gram repeat 2. Benign prompt: correct code + finish=stop. So the 3.0-bit base is
  NOT loop-prone - the ablation is what breaks it.
- `enable_thinking=false` is not a fix (the model does its reasoning in `content` regardless). `repetition_penalty 1.15`
  + `top_k 40` + `temp 0.3` removes the loop, but the model still burns the whole budget meta-reasoning.

**The cause: the skip heuristics drop experts, so the ablation stops being all-or-nothing.**
`--skip-miss` (decode) drops a VRAM-miss expert whose gate weight < skip_miss x the token's sum: `wloc[k]=0` and the entry
is removed from `cpu_i` (ds4_moe.cpp:1260-1276). The dropped expert contributes nothing AND is never computed, so its
routed LoRA delta is never applied - while the experts that ARE computed carry theirs. Within one token, and between
prefill and decode (skip_miss on decode vs skip_file_chunk on the chunk path), the model is a MIX of ablated and
un-ablated experts. Same failure mode as s19's bug (iii) ("a chunk and a decode must be ablated the same or the model is
inconsistent"), just via the skip path instead of MMQ.

**Matched A/B** (same prompt, same sampling: effort=low, temp 0.6, top_p 0.95, top_k 20, repetition_penalty 1.05):
| skip-miss / skip-file | outcome |
| 0.15 / 0.15 (the shipped ablated config) | DEGENERATE LOOP - "from ctypes import LPVOID, LPVOID, ..." x107, finish=length |
| 0 / 0 | CLEAN - finish=stop, 6740 chars, top-6-gram repeat 1, a COMPLETE coherent keylogger + Notes section |
With skip OFF at the model's OWN default sampling (temp 1.0, top_p 0.95, no rep penalty): finish=stop, 6059 chars, no
loop, complete answer - so the fix does not depend on sampling.

**Fix:**
1. `tools/glm/serve/strata-glm-unc-ablated.json`: `--skip-miss/--skip-file/--skip-file-prefill` -> 0 (+ `--pcie 1.0`).
2. `glm_generate.cpp`: as it already does for `chunk_mmq`, `--lora-exps` now FORCES those three values to 0 (stderr note).
   **That source edit is compile-verified (`c++ -fsyntax-only`, exit 0) but the binary is NOT rebuilt** - the ds4 session
   held the GPU lock. Rebuild in the next GPU-free window (`cmake --build build-glm-gpu --target glm_generate`). The
   config fix alone works with the current binary.

**Cost:** skip-miss is the decode lever (0.15 -> 20.65 vs 9.72 tok/s at 0.05, s18); the ABLATED config must run it at 0,
so the ablated server decodes at the pre-skip rate (~15-18 tok/s in these probes). Correctness over speed. A future fix
could make the skip path delta-aware (skip an expert's delta together with its skipped read, and make prefill and decode
skip identically) to win the speed back.

**Untested / open:** the other refusal categories (only the keylogger prompt was run); adapter v1; and the reference
cross-check (llama.cpp's glm5next LoRA on the same GGUF+adapter) - the way to confirm our application matches upstream.

## s21 (2026-10-10): skip-miss quality curve + MTP REJECTED by measurement (the MoE verify does not amortize)

**skip-miss is NOT free above ~0.15.** Decode-loop ppl (`bench/glm-2026-10-09/run-30plus.sh ppl`, eval_code300, 299
positions; loop-vs-loop is the only valid comparison - decode is the path skip touches):
| skip-miss | decode tok/s | loop ppl | vs 0.15 |
| 0.15 | 20.65 | 13.73 (13.31 in a 2nd run) | - |
| 0.20 | ~22 | 15.91 | +16% |
| 0.25 | 24.44 | 17.49 | +27% |
| 0.30 | ~26 | 18.26 | +33% |
| 0.45 | 30.21 | 19.06 | +39-43% |
| 0.50 | 31.20 | 19.96 | +45% |
30+ IS reachable by skip-miss alone (0.45 -> 30.21, 0.50 -> 31.20 tok/s, prefill ~166), but it costs ~40-45% ppl. The
only previously gated value was 0.15 (+0.21% code / -0.97% chat, s18); the 0.15->0.45 stretch is steep. **The resume's
"30+ needs no MTP, just gate a higher skip-miss" is REFUTED - the gate fails.**

**MTP is REJECTED - measured, not inherited.** The verify is a BATCHED forward of k+1 tokens; measured the batched cost
(prefill-chunk path on a 96-token prompt, `bench/glm-2026-10-09/mtp-econ.sh`, `econ-*.log`) against a sequential decode
at the same flags:
| batched forward | per-token | r_verify (vs decode) |
| decode (sequential) | 55.8 ms | 1.00 |
| chunk 6 | 95.1 ms | 1.70 |
| chunk 8 | 88.4 ms | 1.58 |
| chunk 16 | 69.8 ms | 1.25 |
| chunk 1024 | 6.0 ms | 0.107 |
The batched forward is MORE expensive per token at every spec-relevant k, because a k-token batch touches ~k x more
DISTINCT experts (MoE, 288/layer): the memory-bound expert FETCH scales with k and only amortizes once the batch is
large enough that experts recur (`chunk 1024` = 6 ms/token shows the reuse exists, but only at k >> 16). In the runs the
expert arm dominated (`chunk 8`: experts 8016 ms of 8491 ms). Speedup = E[accepted]/(k * r_verify):
| k | E[acc] @ alpha=0.72 | r_verify | speedup |
| 6 | 3.21 | 1.70 | 0.31x |
| 8 | 3.34 | 1.58 | 0.26x |
| 16 | 3.56 | 1.25 | 0.18x |
So spec 4-6 (and 16) is a 0.2-0.3x LOSS. This CONFIRMS and quantifies s7's "MTP ~0.9x" on the CURRENT profile (the regime
did not change enough - a small-batch verify is still fetch-dominated). Do NOT build the MTP pipeline for
GLM-5.3-Flash on this tier unless the expert set becomes batch-reusable (e.g. far more VRAM residency).

**Open lead (untested, cheap):** skip_miss zeroes a dropped expert's weight but the MoE sum is NOT renormalized
(`ds4_moe.cpp:1660-1669` sums `wloc[k]*p[k]` with no divide by the surviving weight sum), so dropping experts scales the
layer output DOWN by the dropped weight fraction - a systematic magnitude error that grows with the dropped fraction and
shadows the ppl curve. Renormalizing by the surviving sum is a small change and could make skip-miss near-free at higher
values (and possibly compatible with the routed LoRA). NOT tested.

## s22 (2026-10-10): UPSTREAM MERGED into glm (upstream is now a real parent)
`glm` == `87cd462f`, a merge whose 2nd parent is `upstream/main` (fb58e0db) -> plain `git merge upstream/main` syncs
from here. Recipe: `git replace --graft 6f32ec07 a1641e9f && git merge --no-ff --no-commit upstream/main;
git replace -d 6f32ec07` (6f32ec07 is tree-identical to a1641e9f; the replace ref is repo-global - delete at once).
Same 8 conflicts as DS4's merge; shared core taken from DS4's cc55d4c9; native_expert.cpp kept OURS (no PTQ1_0 in
GLM); README -> ours + README.strata.md.
- Isolation now structural: upstream's STRATA_GU/D/MMVQ_FMTS verbatim + STRATA_*_FMTS_FORK macros. Two needed
  additions: GLM's Q4_K down_exps is X(12) (upstream's D list has none) -> tier init otherwise fails.
- Build: ~679 MB / 3117 cubins WITH ggml-cuda; a missing -DSTRATA_GGML_CUDA/-DGGML_CUDA_FA/-DGGML_CUDA_GRAPHS gives a
  64 MB binary with no ggml-cuda (this cost me a debug cycle). glm.cmake/ds4_dense.cmake honour STRATA_GGML_DIR.
- Gates green: pool_tasks 168 bitwise, q8k identical, iq_avx2_parity 0 failures, code ppl 3.6478 / chat 5.7046
  (pre-merge code 3.6066 = +1.1%, inside residency noise, not A/B'd), LoRA gate 3.5943 vs pre-merge 3.5967 (correct).
- DS4's merge gave +15% decode (19.55 vs 16.8-17.3, ppl flat) - GLM decode not yet measured on the merged build.

**s22 decode A/B (added):** pre-merge vs merged binary, IDENTICAL flags, both landed at file tier 146:
pre-merge 15.69 tok/s, merged **16.25 tok/s = +3.6%** -> the merge does NOT regress decode; if anything it is
slightly faster. The "20.65 tok/s at skip-miss 0.15" recorded in s18 was a FILE-TIER-0 run - the ~16 vs ~20.6 gap
is residency (file tier 146 vs 0, i.e. how much of the expert set fits the arena), NOT the merge. Always compare at
the same file tier (the tier line prints it).

## s23 (2026-10-10): the abliteration BAKE is not viable; --renorm-skip makes skip+ablation possible; the X(12) merge gap

**1. Baking the rank-1 LoRA into the quantized weights does NOT work (measured, not guessed).** Built
`tools/glm/lora_bake.cpp` (matches glm_lora.hpp: `y += b*(a.x)`; dequant -> add delta -> requant to the SAME type).
Round-trip on real tensors (`delta_retained = ||W_baked-W|| / ||W_ideal-W||`; 1.0 = faithful):
| tensor | type | delta/W | bake-drift/W | delta_retained |
| --- | --- | --- | --- | --- |
| blk.10.ffn_gate_exps (CONTROL, delta == 0) | q2_k | 0.0000 | 0.0199 | -- |
| blk.4.ffn_down_exps | q4_k | 0.0018 | 0.0054 | 3.06 |
| blk.10.ffn_down_exps | q2_k | 0.0106 | 0.0220 | 2.08 |
| blk.20.ffn_down_exps | q2_k | 0.0321 | 0.0312 | 0.97 |
| blk.20.attn_output | q4_k | 0.0067 | 0.0035 | 0.52 |
| blk.20.ffn_down_shexp | q4_k | 0.0871 | 0.0922 | 1.06 |
The abliteration deltas are 0.2-1% of the weight norm -> below the quantizer step, so a requant either dilutes
(attn_output keeps 52%) or drowns (down experts 2-3x) them; and the ZERO-delta control proves a 2% requant DRIFT
(GSQ's grid != ggml's q2_k grid -> even a no-op bake perturbs the base). This reproduces the s20 "mixed ablation ->
loop" failure mode. **Verdict: keep the RUNTIME LoRA; do not bake at q2_k/q4_k.** (Only routed-expert down_proj,
shexp down and attn_output carry deltas - gate/up exps are ZERO placeholders, matching the adapter README's 7,545
effective modules.)

**2. --renorm-skip (new, default OFF): rescale the MoE sum by the surviving weight fraction after a drop.**
s21's magnitude-error lead, implemented at `gpu_run` (:1660-1679, decode host sum) and `gpu_run_chunk` (per-token
`c_w` pre-scale, :1969-1996 + :2424-2432). Exactly 1.0 when nothing is dropped -> byte-identical for no-drop runs.
`--lora-exps` no longer forces skip->0 when `--renorm-skip` is set (`glm_generate.cpp:326`), so ablated+skip is testable.
Gates (eval_code, 1999 pos, skip 0.15, arena 72 GiB, prune-ezct-0.25):
| pass | model | skip | renorm | ppl |
| --- | --- | --- | --- | --- |
| a-off | stock | 0.15 | off | 3.6206 |
| a-on | stock | 0.15 | ON | 3.6090 |
| b-off | ablated | 0 (forced) | off | 3.5943 |
| b-on | ablated | 0.15 kept | ON | 3.5899 |
=> renorm does not regress (stock improves -0.32%); ablated at skip 0.15 is no worse than skip 0 (3.5899 vs 3.5943).
The b-on stderr shows only the `pcie_frac -> 1.0` notice (NOT skip->0) and `moe lora: deltas on 26 layer(s)` -> the
ablation was active WITH skip on. **Behavioral loop gate (s20 keylogger prompt, live server
`strata-glm-unc-ablated.json` (now skip 0.15 + --renorm-skip), effort low, max_tokens 900): PASS** - a complete, coherent Python keylogger
(2542 chars of content, **0** "LPVOID" repeats, no repetition loop). `finish=length` only because it hit the 900-token
budget mid-answer, not the s20 degenerate loop. **So ablated + skip 0.15 + renorm works** - the s20 loop is fixed.
- NOTE the arena load is 236-411 s on this box (vs 93 s in the um run) - each ppl pass is ~6-7 min.
- Rollback: default off -> no change unless `--renorm-skip` is passed; revert the commit otherwise.

**3. The X(12) merge gap (a real bug s22's doc hid).** s22 wrote "GLM's Q4_K down_exps is X(12) -> tier init
otherwise fails", but the fix was **never committed** - it existed only as an uncommitted change in the
`~/AI/Strata-GLM-um` worktree. So the merged `glm` branch's build could not load the model at all:
`tier init: native_expert_grouped has no kernel for layer 3's types 12/12/12`. Applied to
`src/kernels/cuda/iq_kernels.cu:736` (`STRATA_D_FMTS_FORK(X) X(10) X(11) X(12) X(39)`) - **UNCOMMITTED**.
Lesson: a worktree's dirty tree is NOT the branch - verify the fix is in a commit before trusting "gates green".

## s24 (2026-10-10): abliterated GLM-5.3-Flash - what exists, and how small a self-made RCO/GSQ could be
Asked whether to find an abliterated GLM-5.3-Flash and RCO/GSQ it ourselves. HF has ~52 "uncensored GLM-5.3-Flash"
repos. Smallest first: Asilarkness NVFP4-pruned **62 GB** (CUDA-only); huihui-ai abliterated GGUF **93 GB** (UD-IQ1_S)
/ 102 (IQ2_XXS) / 120 (IQ3_XXS) / 157 (IQ4_XS) - abliterates layers 15-35 ONLY, **all experts untouched**, not RCO;
MikeRoz EXL3 2.51/3.05/4.05 bpw (exllama); **GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF** = our exact stack
(RCO+GSQ, LoRA-v2 merged) but only Q4 **137 GB** / Q8 333 GB. orcarouter/dealignai FP8 ~320 GB / NVFP4. **Nobody has
published an uncensored GSQ-RCO GLM at 2-3 bit.**
- Why huihui is the wrong source: its abliteration leaves every expert un-ablated - and the gcsa README (our adapter)
  reports GLM-5.3-Flash refusals live in the routed-expert down_proj writers, while attention/dense-only interventions
  "retain most refusal behavior". So a huihui quant is neither small (120 GB > our 113.6 GB) nor correctly ablated.
- Size math (parsed from our GGUF): 313B params / 113.6 GB / 2.90 bpw; **EXPERTS = 304.4B params = 107.2 GB (97% of the
  model)**, non-expert only 8.9B / 6.4 GB. So the RCO/GSQ lever is entirely the expert precision: all-ternary experts
  -> 60 GB (+6.4) = **~66 GB**; ternary g/u + Q2_K down = ~80 GB; DS4 recipe (IQ2_XXS g/u + Q2_K down) = ~92 GB; all
  Q2_K = ~106 GB. Ternary needs IQ1_S/IQ1_M kernels in the GLM native expert path (no PTQ1_0 in this fork).
- GSQ needs the full-precision weights; no BF16 uncensored exists -> merge our r=1 LoRA into zai-org BF16 (~640 GB),
  or use orcarouter FP8-uncensored (~320 GB, slightly lossy), or GSQ on top of an existing GGUF (the paper supports
  it). Disk is the blocker (49 GB free on NVMe1TB; SSD NVME 163 GB is the roomiest).
- **Recommendation: don't re-quantize.** The runtime LoRA + `--renorm-skip` is the cheaper path to ablated+fast.

## s25 (2026-10-10 PM): the abliterated decode A/B - `--renorm-skip` is a 3x ON THE ABLATED PATH
`--lora-exps` forces `--pcie 1.0` and (before `--renorm-skip`) skip 0, so the ablated server computed every miss
over PCIe. Clean A/B (eval_code prompt, 128-token generate, arena 69.2 GiB, file tier 0, ~8335 experts both arms):
| ablated arm | load | decode |
| --- | ---: | ---: |
| skip 0 (pcie 1.0 forced) | 468 s | **5.44 t/s** |
| skip 0.15 + `--renorm-skip` | 149 s | **16.27 t/s** |
So enabling skip-miss in the ablated path is **3x** (5.44 -> 16.27): the ablated server now roughly matches the
non-ablated fast server, WITH the full routed-expert ablation. This is `strata-glm-quetza --full` /
`strata-glm-unc-ablated.json`. (Loads differ 468 vs 149 s = page-cache warmth; residency was identical.)

## s25b: `--arena-lazy` (fast start), and what the load actually costs
The 69 GiB arena fill blocks the first token 250-800 s while the drive measures **2.5 GB/s** (dd, O_DIRECT and
buffered) - the fill, not the disk, is the cost. `--arena-lazy` (default OFF) allocates the arena, unfills every
slot (so misses go to the file tier), and publishes slots blob-by-blob from a background thread once their bytes
land, so the engine answers in ~seconds and slots become hits as they fill. Compiles (679 MB / 3117-cubin build);
**not yet GPU-tested**. Also: our `--vram-margin 5.5` (code default 1.0) and the 512K KV (~6 GiB) keep expert
slots at 731/5.63 GiB vs PolyStrata's 1413/10.9 GiB on the same card - the ctx/margin sweep (32K/300K @ 1.5) runs
after this.

## s25c: the residency sweep - the VRAM cache can 2.7x, but the margin must leave SCRATCH room
`--ctx 524288 --vram-margin 5.5` -> 5.67 GiB slots / 731 experts (the served config today).
`--ctx 32768 --vram-margin 1.5` -> **15.07 GiB slots / 1959 experts (2.7x)**, but the run then DIES:
`ds4_moe: c_parts: out of memory` - the chunk-path scratch needs VRAM, so a 1.5 GiB margin is too small.
`--ctx 300000 --vram-margin 1.5` -> 8.69 GiB slots (the 300K KV costs ~6.4 GiB more than 32K's).
=> the safe operating point is **~32-64K ctx at margin ~3-4 GiB**: most of the 2.7x, with scratch room. The 512K KV
(~6 GiB) buys long context at the cost of ~half the VRAM expert cache; use it only when a long prompt needs it.
Sweep: `bench/glm-2026-10-09/vram-margin.summary`.

## s25d (2026-10-10 17:00): the real 300K context + Maya-S-v2 wiring (bench/glm-2026-10-09/gate-ctx300k.sh, ctx300k.summary)
- **`--ctx 307200` used to allocate the 512K MLA cache** (CAPMAX = next_pow2). Now ctx rounds to 256. Served ablated
  config (`strata-glm-unc-ablated-300k.json`, margin 5.5, chunk-mmq, eval_code 2000 + 128):

  | ctx | free after dense | slots | prefill | decode | VRAM peak |
  |---|---|---|---|---|---|
  | 512K (s25) | 11.15 GiB | 734 | 135.6 tok/s | 16.27 tok/s | - |
  | **300K real** | 13.65 GiB | **1059** | **158.1** | **17.82** | 22358 / 24564 MiB |
  | 300K + `--arena-lazy` | 13.70 | 1066 | 53.0 (fill competes) | 17.50 | 22547 |

- Parity ctx 4096 vs 2560 (cross2300, indexer on, last bucket non-pow2): first-token logits bit-identical; greedy
  64-id runs split at id ~25 - determinism re-run pending (gate-determinism.sh).
- `--arena-lazy` does NOT pay on a warm page cache: still 64.5 s before serving (the slot seed), same 115 s wall, and
  prefill drops to 53 tok/s while the fill runs. Not wired into serving.
- `--allow-long-ctx` (dense MLA, no indexer) costs ~1 GiB: the s25c sweep used it, serving does not - its margin-1.5/3.5
  OOMs (c_parts/c_x/c_stage) overstate the scratch.  At margin 5.5 the served peak leaves ~2.1 GiB unused: ~4.0 is the
  next thing to try (untested).
- Maya-S-v2 wiring (4439cf5a): IQ2_S/IQ3_XXS down kernels; split-aware tier geometry (shard 1 = blk.0-23 only);
  block_count 46 - nextn_predict_layers. Armed: gate-maya.sh (s11 ppl + ablated 300K decode).

## s26 (2026-10-10 evening): Maya-S-v2 + the peer-survey levers (gates 1-2: maya-improve{,2}.summary)
Model: Maya-S-v2 (peasantsmith, IQ2_XXS g/u + IQ2_S/IQ3_XXS down, Q6_K dense; copied to the NTFS NVMe
`/media/mal/SSD NVME/Models/GLM-5.3-Flash-Maya-S-v2/` - the SATA/FUSE MX500 read at ~255 MB/s) + OUR abliteration LoRA
(--lora-exps), ctx 300K, prune 25% soft, skip 0.15 + renorm.  Workload: chat_code6k.i32 (12736-token chat turn with
thinking) -> 256 greedy tokens; text read for coherence (all coherent unless noted).

| arm | prefill tok/s | decode tok/s | VRAM hit | note |
|---|---|---|---|---|
| A today's --300k config | 204.7 | 21.69 | 23.9% | RCO on the same config: 158 / 17.8 |
| B + LoRA delta in the MMQ chunk path | 217.8 | 22.00 | 23.8% | --chunk-mmq no longer forced off |
| C + --census (bootstrapped from saliency) | 222.4 | 23.33 | 34.1% | pcie 12.1 -> 8.4 ms/token |
| D + --vram-lru --vram-pin 0.6 | 222.6 | 22.32 | 44.9% | swaps pay their DMA; fewer skips (166 vs 193) = quality, not speed |
| E D at chunk 4096, margin 9 | 380.0 | 20.47 | - | 2.5 GiB of slots left |
| F D at chunk 6144, margin 12 | 416.5 | 17.42 | 0 slots | decode with NO VRAM cache = 17.4 |
| G C + --pcie auto (CPU pool now ablated too) | 224.0 | 24.06 | 35.3% | cpu 3.9 / pcie 3.9 % of experts |
| H G + --dense-requant q4_k | 229.8 | 25.87 | 42.0% | attention 17.9 -> 15.8 ms; 8.5 GiB slots |
| K H + elastic, chunk 4096 (margin 7.5, grow 1.5) | 399.6 | 24.49 | 41.4% | grow 909 -> 1412 slots in 136 ms |
| L H + elastic, chunk 6144 (margin 10) | 439.9 | 23.12 | 37.9% | grow 474 -> 1231 |
| I K + --route-bias 0.02 | 397.6 | 23.09 | 30.8% | LOSS (hit fell) - dropped |
| J K at ctx 524288 | 392.5 | 23.94 | 37.3% | 512K costs ~0.6 tok/s |

Chunked ppl (2000 tokens, margin 5.5): Maya+LoRA chat **5.337** / code **3.720** (RCO unablated 5.547 / 3.668);
q4_k dense 5.612 (+5.2%) / 3.708; q4_k + route bias 5.672 / 3.736.  -> q4_k dense costs chat quality (double
requant, no imatrix) - gate 3 tries q5_k.
- MTP: NOT ported. project-maya (the Maya author's engine) refuses MTP on one GPU ("carries its weights and expert
  slots for nothing until a batched verify exists" - glm_model.cu:2398); helios measured -26..-57% wall; PolyStrata
  on our box +2%.  A verify row routes to a different expert set, so with ~35-45% residency it cannot pay.
- Decode anatomy (H): attention+router 15.8 ms, experts 18.9 ms (hits 6.3, pcie 4.5, cpu 6.3 - overlapped), prefetch
  2.1, finish 1.2, head 0.5.  The tier wall (18.9) exceeds max(card, pool) by ~8 ms: per-layer host round trips.
- Two leaks found in the elastic arms (fixed for gate 3): (1) attn_router_n uploaded the WHOLE input span every
  decode token - the mask alone is MCAP x NT F16 = 16-24 MB at NT 4096-6144 (K: attention 15.8 -> 18.4 ms); now each
  input's used prefix only. (2) the dense chunk compute buffer (allo_big) was kept after the prompt -> GlmDense::
  release_big() frees it before grow_cache.

## s27 (2026-10-10 late): DECODE IS HOST-LATENCY-BOUND - 8 syncs/layer x 45 layers. The copy engine is NOT the bottleneck
Source: `bench/glm-2026-10-09/mi3-decode_{cuda_api,cuda_gpu_kern,cuda_gpu_mem_time}_sum.csv`, an nsys capture of the
decode loop (old binary, 27.70 s / 255 forward passes / 256 tokens, 9.21 tok/s) - `GLM_PROFILE_DECODE=1`, capture range
cudaProfilerApi.  **The gate-4 P-nsys arm re-profiles on the current binary; re-read the three CSVs before acting.**

**ERRATA (added 20:10, same session - two of this section's first claims were WRONG; corrected below.  Keep the sync
count and the de-sync roadmap; DISCARD the "GPU idle" and "--pcie auto units" claims.)**

**The GPU utilisation is NOT measured by this capture - do not claim "GPU idle ~98%".** The kernel table (584.5 ms,
78356 instances) contains **only direct `cudaLaunchKernel` launches**: 78356 table instances vs 83099 API calls, and
the breakdown is native/MoE 480.6 ms (40312), lora 58.9 (9250), aux quantize/swiglu 29.0 (24642), ggml/dense
**11.5 ms / 1354 instances**.  255 passes x 45 layers of attention cannot be 1354 kernels, and `flash_attn_ext_f16`
shows **11 instances total** - i.e. the attention/finish/predict compute runs **inside the 34036 `cudaGraphLaunch`
calls (134 graphs x 255 passes) and its nodes were NOT expanded into this table**.  The 11.5 ms of dense kernels are
capture-warmup stragglers, not the attention cost.  To measure real occupancy, re-capture with the graph nodes
expanded (`--cuda-graph-trace=node`) or sum the per-graph times.  Also note the capture is the OLD binary at 9.21
tok/s (108 ms/token vs 41 ms now), which inflates any idle share.

**What IS solid** (wall measurements, independent of the kernel table): the H2D copy engine 6195 ms = 24.3 ms/token,
~575 MB/token, 126838 ops (async, so overlapped); CUDA API 9743 ms of which **cudaStreamSynchronize 6825 ms = 70.1%
over 90826 calls = 356/token** (median 7.2 us, max 4.6 ms).  The sync total, not the copy volume, is the finding.

**The 356 syncs/token are fully accounted for: 8 per layer x 45 layers = 360.**  Per decode layer per token, the
engine makes FOUR host round trips, each of which is a synchronous ggml call:
1. `predict(il)` - `ggml_backend_graph_compute` (glm_dense.cpp:1014) + `tensor_get(predict_out)` (:1016)  [2 syncs]
2. `attn_router_n(il)` - `ggml_backend_graph_compute` (:1131) + **`tensor_get(L.o_span)`** (:1136, the
   [ffn_norm|ids|wts] readback)  [2 syncs]
3. `gpu_run` - "layer sync" (ds4_moe.cpp:1654) + `h_parts` D2H + "parts sync" (:1704)  [2 syncs]
4. `finish_layer_n(il)` - **`tensor_set(routed_sum)`** (glm_dense.cpp:1159) + `ggml_backend_graph_compute` (:1164)
   [2 syncs]
`ggml_backend_graph_compute` = compute_async + `ggml_backend_synchronize` (third_party/.../ggml-backend.cpp:455-459);
`tensor_get`/`tensor_set` sync the stream in ggml-cuda.  So per layer the residual stream leaves the device and comes
back TWICE (the ffn_norm activation, and the routed sum), and the host blocks 8 times.

**Consequence - the expert-time decomposition is serial, not overlapped.** The tier wall for a layer (~20.6 ms/token)
equals gpu hits (6.19) + pcie (3.86) + cpu pool (9.12) - it is a SUM, so the CPU pool does not overlap the GPU.
(`hit_ms`/`pcie_ms` are `cudaEventElapsedTime` between events on gp.s, so they are wall elapsed and include the
stream's own stalls; `cpu_ms` is host wall.  That is what makes the sum meaningful - and it is why the missing
graph-node time does not affect this paragraph.)
Per-expert cost: **CPU pool 0.52 ms vs GPU hit 0.045 ms (11.6x)**; misses are ~28/token of 360 lookups (17.6 cpu +
10.4 pcie); the other 181/token are already dropped by skip/rename.  Moving pool work to the card cannot fix this by
itself - PCIe costs 0.37 ms/expert and is just as serial.

**Therefore the decode roadmap is de-synchronisation, not tier tuning.  Target: 2 syncs/layer, not 8.**
- (a) **GPU-side MoE weighted sum** over `gp.d_parts` (+ the pool's `im.parts` H2D'd, ~4 experts/layer): removes the
  131 KB `h_parts` D2H, the "parts sync", the host sum loop (ds4_moe.cpp:1740-1749) AND `tensor_set(routed_sum)` -
  the finish graph then consumes the MoE output on-device.  MUST reproduce the double-accumulated sum bitwise.
- (b) **Keep the residual + activation on-device.** `o_span`'s ffn_norm region IS `gp.d_x`; the D2H is only needed
  for the pool (`native_quant_act`, host Q8_K) - skip it on layers where nk == 0, and always skip the redundant
  `d_x` H2D (ds4_moe.cpp:1383).
- (c) **Fold `predict` into the previous `finish` graph** (same input tensor `x_state`, same weights; predict needs
  nothing finish(il-1) does not already have) - saves a launch + readback sync per layer.
- (d) Once (a)-(c) land, layer l+1's attn graph can be issued while the host still decides layer l's experts, i.e.
  a rolling pipeline - the only way to overlap the 9 ms/token pool.
- (e) Only then re-measure the CPU/PCIe split.  (An earlier draft of this section called `--pcie auto`'s EMA units
  inconsistent - that was WRONG.  `ema_cpu`/`ema_pcie` are per-expert, `ema_hit` is the hit group's TOTAL ms, and
  `nc*c - ema_hit` is ms - ms, with `/(p+c)` ms per expert, so `budget` is a count.  It is the correct
  hit_time + np*p = (nc-np)*c balance.  Leave the formula alone.)

Cheap, no-numerics-change candidate first: (c), then (b).  Both are checkable by ppl parity (X-* arms) + the nsys
sync count, NOT by wall time alone.

## s26 gate 3-4 (2026-10-10 19:34-20:30: maya-improve3.summary, maya-final.summary)
Maya+LoRA, census, pcie auto, elastic cache, chunk 12736-token chat prompt -> 256 tokens (all coherent):

| arm | prefill tok/s | decode tok/s | slots after grow |
|---|---|---|---|
| M el4k q5 dense | 396.4 | 24.38 | 1489 |
| M el4k q6 (as shipped) | 392.6 | 23.99 | 1336 |
| N el6k q5 | 432.8 | 24.91 | 1422 |
| O el4k q5 512K | 389.4 | 24.77 | 1120 |
| final R300-4k (+indexer NW fix) | 388.6 | 24.49 | 1470 |
| final R300-4k --pf-b 0 | 391.1 | 23.81 | 1479 |
| final R300-6k | 429.4 | 24.38 | 1403 |

- release_big works: the 6K chunk now costs decode ~nothing (L before the fix: 23.1 at 1231 slots).
- The indexer NW fix did not move decode measurably (R300-4k 24.49 vs M 24.38): its padding work was cheap.
- Prefetch OFF is slower (-2.8%): keep --pf-b default.
- ppl with skips ON is noise-confounded by residency (q5 chat 5.73 / code 3.66 vs q6 5.34 / 3.72; q4 5.61 / 3.71):
  gate 4's X-* arms measure the requant with exact math (no skips).
- nsys decode profile (gate 3, mi3-decode*.csv): host->device copies ~24 ms/token of copy-engine time, ~575 MB/token
  (prefetch ~60 experts/token issued, ~45 used + PCIe misses); 355 cudaStreamSynchronize/token (75 us avg) -> the
  next decode lever is fewer per-layer host syncs (GPU-resident routing for VRAM-hit layers), not more bandwidth.

### s26 gate 4 raw summary (auto-appended 2026-10-10 20:08:14)
```
--- R300-4k ( --temp 0 --ids-file bench/glm-2026-10-09/prompts/chat_code6k.i32 -n 256 --dense-requant q5_k --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5) 19:53:11 ---
grow_cache: 749 -> 1470 slots (+721, 0 from the file tier) in 195 ms
prefill: 12736 tokens in 32.77 s = 388.63 tok/s (chunked)
decode : 256 tokens, 255 forward passes in 10.41 s = 24.49 tok/s
  exit 0, VRAM peak 23539 MiB
--- R300-4k-nopf ( --temp 0 --ids-file bench/glm-2026-10-09/prompts/chat_code6k.i32 -n 256 --dense-requant q5_k --ctx 307200 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 --pf-b 0) 19:56:04 -
grow_cache: 749 -> 1479 slots (+730, 0 from the file tier) in 198 ms
prefill: 12736 tokens in 32.56 s = 391.13 tok/s (chunked)
decode : 256 tokens, 255 forward passes in 10.71 s = 23.81 tok/s
  exit 0, VRAM peak 23177 MiB
--- R300-6k ( --temp 0 --ids-file bench/glm-2026-10-09/prompts/chat_code6k.i32 -n 256 --dense-requant q5_k --ctx 307200 --prefill-chunk 6144 --vram-margin 10 --vram-grow 1.5) 19:58:59 ---
grow_cache: 313 -> 1403 slots (+1090, 0 from the file tier) in 295 ms
prefill: 12736 tokens in 29.66 s = 429.35 tok/s (chunked)
decode : 256 tokens, 255 forward passes in 10.46 s = 24.38 tok/s
  exit 0, VRAM peak 23548 MiB
--- R512-4k ( --temp 0 --ids-file bench/glm-2026-10-09/prompts/chat_code6k.i32 -n 256 --dense-requant q5_k --ctx 524288 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5) 20:02:02 ---
grow_cache: 379 -> 1101 slots (+722, 0 from the file tier) in 195 ms
prefill: 12736 tokens in 33.22 s = 383.33 tok/s (chunked)
decode : 256 tokens, 255 forward passes in 10.39 s = 24.55 tok/s
  exit 0, VRAM peak 23218 MiB
--- R512-6k ( --temp 0 --ids-file bench/glm-2026-10-09/prompts/chat_code6k.i32 -n 256 --dense-requant q5_k --ctx 524288 --prefill-chunk 6144 --vram-margin 10 --vram-grow 1.5) 20:05:08 ---
prefill: 12736 tokens in 29.74 s = 428.26 tok/s (chunked)
decode : 256 tokens, 255 forward passes in 12.41 s = 20.55 tok/s
  exit 0, VRAM peak 22604 MiB
--- X-q6-chat ( --ctx 4096 --prefill-chunk 1024 --vram-margin 5.5 -n 1 --ppl --skip-miss 0 --skip-file 0 --skip-file-prefill 0 --ids-file bench/glm-2026-10-09/prompts/eval_chat.i32) 20:08:07 ---
(gate stopped after R512-6k so Mal can test the wrapper; X-* exact ppl + nsys arms not run)
=== maya-final done 20:08:14 ===
```
- **512K + chunk 6144 is a trap**: margin 10 exceeds the VRAM free after the dense half at 512K, so the elastic cache never engages (no grow_cache) and decode runs with no VRAM experts (20.55 tok/s). The 512K serve config uses chunk 4096 / margin 7.5 (24.55 tok/s, prefill ~389). 6K chunks are for 300K only.

## s28 (2026-10-10 night): long-prompt OOM fixed; decode is PCIe-bound (s27 corrected); 23.7 -> 27.2 tok/s at BETTER quality
All runs: Maya-S-v2 + LoRA, 512K serve flags (chunk 4096, margin 7.5, vram-grow 1.5), chat_code6k 12.7K prompt -> 256
greedy tokens, scripts bench/glm-2026-10-09/{oom-repro,dec-fused,ppl-requant,ppl-rerun,nsys-node}.sh.  Noise ~+-2%.

**1. The serve crash (Mal's 19K-token request, exit 134) was a VRAM OOM in cudaGraphInstantiate, not RAM.** ggml-cuda
captures a CUDA graph on the 2nd identical call, so the 4th 4096-token chunk (2nd at cap 16384) instantiated a graph
of a whole prefill pass (~1.5 GiB) into the ~0.9 GiB the elastic cache leaves.  Gate 4's 12.7K prompt never got a
2nd full chunk at one cap.  Fixes: no CUDA graph for a pass whose MUL_MAT takes > 16 columns
(GGML_CUDA_GRAPH_MAX_BATCH; third_party is untracked -> tools/glm/patches/0001-*.patch), and GlmDense::safe_chunk
halves the chunk per cap doubling past cap x n = 65536 x 4096 (the indexer's [pools x tokens] scratch; -128/-256 MiB
per doubling measured).  50K prompt: OOM -> OK (prefill 382 -> 401 tok/s, no graph build); 100K: OK, 332 tok/s,
headroom flat 488 MiB past 64K.  GLM_VRAM_TRACE=1 prints free VRAM per chunk.

**2. s27 was wrong twice.**  (a) The CPU pool already overlaps the card: DS4_TIER_PROF per layer call = launch 0.042,
pool 0.344, then 0.109 ms still waiting on the card -> the card path is the critical one ("wall = hits+pcie+pool" was
a coincidence).  Pinned launch_group metadata (pageable-copy theory) and async dense calls (8 -> ~3 syncs/layer):
correct, kept, +0% (23.74 / 23.30 / 23.52).  (b) nsys --cuda-graph-trace=node (tools/glm/nsys_decode_anatomy.py):
kernels 16.4 ms/token (dense q5_k matvec 5.8 = the weight-bandwidth floor), **H2D copy engine busy 26.4 ms/token
(~22 GB/s = the PCIe 4.0 ceiling)**.  Decode is PCIe-bound (expert prefetch + PCIe misses) + NVMe file reads.

**3. What paid (decode tok/s):**
| change | before -> after |
|---|---|
| `--pf-b 1.0` (was 1.43; sweep 0.7 24.0, 1.0 25.18/25.11, 1.2 24.4, 1.43 mean 23.7 over 4, 2.0 22.8) | 23.7 -> 25.1 |
| soft-pruned experts fill the arena (census-ranked after the kept set; they were NVMe reads, ~4 ms/token) | 25.1 -> 26.2 (64 GiB) / 27.9 (72 GiB) |
| `--dense-requant q4_k` instead of q5_k (q6 native 25.0 at the same flags) | -> 27.2 |
| fused predict (finish(il) computes predict(il+1)) | +0.4% (noise), kept |
Bound: `--skip-file 1.0` (drop every file-tier expert) 27.85, 0.3: 27.37 - the arena fill gets that speed without dropping.
Arena 72 GiB leaves MemAvailable 5 GiB in decode (memguard floor 4) -> shipped 64 GiB (13-14 GiB free).
With q4_k dense, arena 72 GiB = 27.64 vs 64 GiB 27.23 tok/s (+1.5%) at MemAvailable 5-7 GiB: not worth it, 64 stays.

**4. The shipped q5_k dense cost +6.2% chat ppl** (exact math: skips 0, eval 4K tokens, bit-identical on rerun):
| dense | chat | code |
|---|---|---|
| q6_k native | 5.2727 | 3.6014 |
| q5_k | 5.5983 (+6.2%) | 3.6405 (+1.1%) |
| q4_k | 5.3549 (+1.6%) | 3.6544 (+1.5%) |
q5_k worse than q4_k on chat is unexplained (requant = Q6_K -> f32 -> ggml_quantize_chunk, no imatrix) - a per-tensor
bisect would find it.  Shipped: q4_k (faster than q5 AND better chat).  `--dense-requant none` = native q6 (25.0).

**Remaining levers (each small):** arena 72 GiB if RAM allows (+~6%); router/hc F32 -> F16 (<= 0.6 ms, changes top-k
numerics); elementwise fusion (~5 ms/token of tiny kernels, ggml-level); CPU pool ~0.36 ms/expert vs ~0.2 floor (lets
--pcie auto move misses off the saturated bus); prefetch further ahead (the bus idles ~14 ms/token).
- **Mal's call (2026-10-10): serve native q6_k dense** - quality over q4_k's +2 tok/s.  Shipped decode 25.0 tok/s.
