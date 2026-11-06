# Xolotl changelog

What landed in Xolotl, newest first. The README keeps a one-line summary of each item and links here.

- [November 2026](#november-2026)
- [Model releases](#model-releases)
- [October 2026](#october-2026)

## November 2026

### IQ1_M prompts on MMQ (6 Nov)

- **IQ1_M experts take llama.cpp's MMQ kernels for prompts** (on by default, `STRATA_MMQ_IQ1M=0` restores the old path; port of
  Basalt 86b1e97f4 by jesdga95). Upstream MMQ has IQ1_S but not IQ1_M, so the 27 IQ1_M layers of
  [QFN 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored) ran their prompt experts
  on the FP16 path, while its other 21 layers already took MMQ. The MMQ tile holds an IQ1_M weight exactly as int8 with
  one scale per 16 values (IQ2_XS's tile layout), with its own loader (`src/prefill/mmq_iq1m_tiles.cuh`). The type is
  added by a configure-time overlay of the pinned ggml-cuda headers in the build tree
  (`cmake/strata_ggml_iq1m_overlay.cmake`); the shared llama.cpp tree is not touched, and any anchor mismatch turns the
  overlay off. CMake option `STRATA_MMQ_IQ1M` (ON) builds it. Test: `tests/cuda/prefill_mmq_iq1m_test.cpp` (random
  IQ1_M blocks against a double product, 0.23 % relative error).
- **Measured on the RTX 4090**, QFN 45 GB, 16,383-token prompt + 1,024 tokens, 3 runs per arm, every arm emitting the
  same text: prefill **3,239 -> 3,796 tok/s (+17.2 %)**; a second run of the old path gave +1.3 %. A recheck on the public build
  (default on vs `STRATA_MMQ_IQ1M=0`, 2 runs): 3,430 -> 3,890 tok/s (+13.4 %). Decode stays within the run-to-run noise
  (the time per verify window 3-4 % higher in both A/Bs; a rerun of the old build moved 3 %).
- **Quality: closer to Q8_0.** Prompt-path logits on 20 prompts against the Q8_0 parent model: the old path KL mean
  0.224 / median 0.0096 / max 1.60, IQ1_M on MMQ 0.184 / 0.0077 / 1.08, top-1 90 % on both. MMQ rounds the activations
  to q8_1, as QFN's other 21 layers already did.

### Prefix-cache latency, MiniMax replay and the Basalt ports (4 Nov)

- **Faster short requests with the disk prefix cache on** (#115, NektoDron). Opt-ins:
  `STRATA_PREFIX_CACHE_NONBLOCKING_LOOKUP=1`, `STRATA_PREFIX_CACHE_MIN_RESTORE_CELLS=512` and
  `STRATA_FLASH_PREFIX_MIN_CAPTURE_TOKENS=512`. Measured on the 4090 with the Swift IQ3_XXS Flash engine and a 47K-token
  prompt, cache on. A short request right after a long one took 9.0 / 2.1 / 2.0 s stock and 0.32 / 0.26 / 0.24 s with
  the opt-ins; after a restart, 9.9 / 1.9 / 1.8 s against 0.34 / 0.28 / 0.24 s. Snapshots written dropped from 13 (4.9
  GB) to 2 (2.5 GB). Answers were the same in both runs. All three are off by default, and so is the cache itself.
- **MiniMax M3.1 grammar replay** forwards the common sampling keyword (#116, NektoDron).
- **Basalt ports** (jesdga95): the stream-all prefill chunk priced from the link and the non-resident bytes, quoted
  tool calls inside string values, union and nullable tool parameters decoded by their schema, and IQ3_S / IQ2_S /
  IQ3_XXS / IQ1_M code bytes taken by shift instead of through a byte pointer. The IQ change gives the same decode
  speed on the 4090 (104-107 tok/s on both builds), and the kernel parity tests pass.
- **RTX 5090 numbers for a bigger box** (#114, cpacsm75): `--arena-gib` and `--kv-q8` at 256K and 512K, in
  [docs/DEEPSEEK_V41.md](../DEEPSEEK_V41.md). The 512K prompt went 1,056 -> 1,498 tok/s with `--kv-q8`.
- DS4.1 vision: the `--reserve-mib 900` note in [docs/DEEPSEEK_V41.md](../DEEPSEEK_V41.md).

### LFM3-Omni-2B, natively (2 Nov)

`xolotl-lfm3`, [docs/LFM3.md](../LFM3.md): Liquid AI's omni model (text, pictures, audio and video in; text and speech
out) in its own engine behind the usual server. About 300 tok/s decode, 26-33K tok/s prefill, and `/v1/audio/speech`
and `/v1/audio/transcriptions` from the model itself. Park/unpark takes ~50 ms, so it can sit beside a big model as its
ears, eyes and voice. Its logits match llama.cpp's (KL 1.3e-7 on an exact-layout run).

### LFM3 as the ears and voice of a bigger model (3 Nov)

`scripts/xol-qfn-voice.sh`: QFN 45 GB and LFM3-Omni on one 4090, then QuetzaCodetl in voice mode. QFN's elastic expert
cache has to hand ~4.5 GB back (`POST /v1/vram`) or LFM3's encoder cannot load. After that both are resident at 22.8 GB,
and LFM3 transcribes and speaks while QFN keeps its speed: 0.75 s for a 96-token answer while 3.5 s of speech streamed.

## Model releases

### Qwen3.8-Flash-Next XQ 45 GB (31 Oct)

On [Hugging Face](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored). It's Swift 1.5
(UkisAI's reasoning-efficient fine-tune) with the refusal direction removed, quantised with our XQ pipeline: 44.9 GB of
model plus the 28.8 GB n-gram (PLE) table, which is read from disk. On the 4090 it does **199 tok/s decode and 3,204
tok/s prefill at a 16K prompt** (160 / 2,819 at 8K). On Gauntlet v2 it solved 5/7, score 6.68 (our previous 55.8 GB
build 6/7, 6.50; Qwen's own API 5/7, 5.50; one run each), and it refused 0 of 100 held-out harmful prompts and 0 of 30
safe ones. Serve it with `./scripts/xol-qfn.sh --model <shard 1 .gguf> --pack <pack-qfn45>`.

### Qwen 4, day one (prep, 30 Oct)

Qwen 4 isn't out yet, but we've been getting ready so Xolotl can run it as close to release day as possible. If Qwen 4
Flash keeps Qwen3.8-Flash-Next's architecture, the engine already runs it. The 27B looks dense, so it's going into
`strata-dense` with the hyper-connections and QSA attention added. What's known so far and the plan:
[docs/QWEN4.md](../QWEN4.md).

### DeepSeek-V4.1 V4 (25 Oct)

On [Hugging Face](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored) (`v4/`, 152 GiB), the best
DeepSeek build so far. Through the real engine it agrees with the full-precision model on the top-1 token 86.9% of the
time against V3's 82.9% (KL 0.306 vs 0.405; 85.7% vs 81.5% on the Python reference path). Its perplexity is within
0.4% of the full model's, and the 1,056 hottest experts stay in DeepSeek's native FP4. It solved 21/21, 10/10 and 3/3 on
the README's coding tasks, 162/164 on the full HumanEval, and 6/7 (6.50) on Gauntlet v2 against the DeepSeek API's 5/7
(6.41). Run it with V4's own rank file, `--rank v4/rank-1017-v4.bin`, and `--tier-batch-pcie 0.25`. If you downloaded V4
before 25 Oct, grab the rank file (123 KB):
`hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored v4/rank-1017-v4.bin --local-dir V41-XQ-1017`.
Speeds in detail: [BENCHMARKS.md](BENCHMARKS.md#deepseek-v41-flash-1017-v4-25-oct).

## October 2026

### Grammar, a disk prefix cache and serving fixes (#99, #105-#113)

- Native GBNF (`grammar` / `structured_outputs.grammar`) for Qwen Flash and DeepSeek V4.1 behind the build options
  `-DSTRATA_FLASH_GRAMMAR=ON` / `-DSTRATA_DS41_GRAMMAR=ON`, with MTP/DSpark drafts kept (#112, #113, NektoDron).
- An opt-in Linux disk tier that keeps exact prefix checkpoints across restarts
  ([docs/PERSISTENT_PREFIX_CACHE.md](../PERSISTENT_PREFIX_CACHE.md), #99, NektoDron).
- Completed JSON responses out of disconnect accounting, and batch cancellation kept through the EOS drain (#107, #108,
  NektoDron). HIP shared int4 alignment (#105, NektoDron).
- DSML tool calls no longer reach the content as text, a turn cut inside a DSML tag sends none, and opt-in
  `keep_cut_tool_calls` keeps a call cut at the token limit (#106, #109, #110, Xiaomi2008).

Every new option is off by default.

### More community fixes (#96, #98, #100-#104)

- MiniMax M3.1 serving works with the current frontend again (#96, NektoDron).
- Qwen picture numbering (`chat_template_kwargs: {"add_vision_id": true}`) reaches the template (#101) and can be a
  server default (`"add_vision_id": true` in the run config, #104). New Qwen3.6 setups keep the official 64-token image
  minimum (#102). All NektoDron.
- Layer split no longer aborts on the first request at boundaries other than 20 (#98, diagnosed by ghino18, confirmed by
  tragicwinds). A request beyond the batch slots waits for one again instead of a 503 (#100, reported by tragicwinds).

### Nine community pull requests (#86-#95)

- IQ1_S routed experts for Unsloth's DeepSeek-V4.1 UD-IQ1_M (#86), and opt-in fused router ops and batched state copies
  in the DeepSeek-V4.1 engine (#87), both Xiaomi2008.
- Opt-in request-owned batch cancellation (#95), learned checkpoints for edited prompt suffixes (#90), shared immutable
  checkpoint payloads (#89), a bounded tokenizer span cache (#91) and Qwen3.6 snapshot ownership (#92). Cancelled requests
  no longer wait for a slot (#88), and the Metal fallback signatures are synced (#93). All NektoDron.
- The Windows DeepSeek-V4.1 build compiles again (#94, kelheor).

Every new option is off by default.

### ISTA-DASLab's GGUFs work too

Xolotl loads [ISTA-DASLab](https://huggingface.co/ISTA-DASLab)'s
[GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) of DeepSeek-V4.1-Flash-1017
(the unedited model, IQ2_XXS and IQ2_XS), thanks heaps to ISTA for putting those out. IQ2_XS on the 4090: perplexity 3.93
on a 256-token test (our V3 3.60, same flags) and sane chat answers. Set `-m` to their first shard
(`...-00001-of-00004.gguf`), keep `--engram` on our Engram tables (identical hash tables), and leave the `vision` section
out of the config (their files don't carry the router's image bias).

### Upstream llama.cpp and Unsloth DeepSeek-V4.1 GGUFs (#82, #84, #85)

`ds41_generate` reads the `deepseek41` layout upstream's converter writes (ISTA-DASLab's GSQ-RCO files since Oct 27,
Unsloth's), the combined Engram table straight from its GGUF shard, the upstream DSpark sidecar, and IQ3_S routed-expert
downs (Unsloth's UD-Q2_K_XL). AlexeyMatskevich (#82, #85) and NektoDron (#84). Running them, measured against ISTA
IQ2_XS: [docs/DEEPSEEK_V41.md](../DEEPSEEK_V41.md).

### OpenZFS ARC counted as available RAM (#83)

AlexeyMatskevich, opt-in `STRATA_ZFS_ARC=1`: on a ZFS box the reclaimable ARC no longer shrinks or refuses the ds41
arena, and `zfs_arc_max` no longer has to be capped by hand.

### Deep sleep for MiniMax-M3.1 and DeepSeek-V4.1 (#75)

NektoDron: opt-in `"deep_sleep": "auto" | "native" | "checkpoint"` plus authenticated `POST /sleep` and `/wake` gives the
GPU back while the loaded model and conversations stay in host RAM. M3.1 sleeps natively (wake ~1.4-1.9 s); DeepSeek-V4.1
uses the CUDA driver checkpoint (restore ~19-21 s). Off by default: [docs/DEEP_SLEEP.md](../DEEP_SLEEP.md). **Draft:**
merged for testing; the GPU numbers predate the rebase, and the DeepSeek checkpoint/restore hasn't been re-run on current
main yet (#74 stays open until it is).

### Qwen3.6 serving (#76, #79, #80)

- Vision, a shared prefix cache and batched sampled MTP (#76, NektoDron): text and image prefixes are reused across
  callers and owners, native continuous batches keep each request's DeltaNet state, KV, grammar and MTP separate, and
  GBNF / JSON-schema output is native: [docs/Q36_SERVING.md](../Q36_SERVING.md).
- Resident prefix reuse (#80, NektoDron, opt-in `--resident-prefix-reuse`): when the immutable host pages still
  describe exactly the KV already on the card, it is kept instead of copied back. An image follow-up copied 14.7 MiB
  instead of 872.7 MiB, and first content went from 195 to 162 ms. `--resident-prefix-verify` byte-checks it.
- No more lost control-queue wakeups (#79, NektoDron): a fairness waiter could miss its notification and sit out the
  500 ms timeout; it now rechecks under the lock.

### GLM-5.3-Flash long conversations no longer crash (#46)

vthomasino: when the graph pool fills and gets rebuilt, every cached graph (predict and rows included) and the CUDA
backend's captured graphs are dropped too, so nothing replays against freed memory. Regression tool:
`tools/xolotl-glm-manylengths.py`.

### Earlier in October

- **Batch slots** for the DeepSeek-V4.1 engine (#22, NektoDron): `"parallel": N` in a ds41 config runs several requests
  in one pass over the model, one token each, on the server's existing slot scheduler. How it works, what a slot costs
  and the limits: [docs/BATCHING.md](../BATCHING.md).
- **Conversation pool** (#19, #20, NektoDron): every conversation is kept at its turn boundary in pinned host RAM
  (`--conversation-cache-mib`, default 8192), and a request resumes from the longest matching prefix, pictures included.
  Agents taking turns no longer re-read their prompts (measured on a 5090: three agents at 30-60K tokens, 47.5 -> 3.96 s
  a turn).
- **Long prompts** past 512K tokens no longer run out of memory or crash on the pass graph (#21, NektoDron): 768K and 1M
  prompts complete.
- **Exact resumes** (#18, NektoDron): a partial unpark resumes only inside the window ring, and the Engram history is
  written back on every resume (bit-identical to a fresh prefill). The **image encoder runs on the GPU** with fast SDPA
  (#17, NektoDron).
- **Multi-GPU layer split** (AtticusG3 / buster, Niko1221, bsvinay): `--layer-split` across cards with a PCIe hand-off,
  an optional head device for the output head + DSpark, a VRAM expert cache per GPU and sm_70 (V100) support:
  [docs/MULTI_GPU.md](../MULTI_GPU.md), [BUSTER.md](../../BUSTER.md). Under a split the engine serves one request at a
  time with its own split-aware conversation cache; batch slots and the pool are single-card.
- **Remote expert card** (`--remote-device`, masahiro hibi): a second GPU holds extra experts for the first, with
  `--route-bias-remote`.
- Serve: STOP cancels a long prefill (clawdia-saka), `--kv-q8` / `--kv-type` for the compressed KV + indexer keys, a park
  window floor for agent side calls, cached DSpark verify-width graphs, forced tool calls in V4.1's DSML shape, and the
  expert-cache fixes from clawdia-saka.
- **DeepSeek-V4.1-Flash-1017 builds** of the October checkpoint:
  [IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)
  (V2 142 GiB; V3 152 GiB, recalibrated on a code + agentic heavy mix; **V4** 152 GiB, out 25 Oct, with our XQ refinement
  on every expert and the hottest experts kept in native FP4: 85.7% top-1 agreement with the full model vs V3's 81.5% on
  the Python reference path).
- **Native Windows** for the DeepSeek-V4.1 engine (MSVC + CUDA, RTX 50 notes from #14): [docs/WINDOWS.md](../WINDOWS.md).
- **Vision** input for DeepSeek-V4.1 (#8), per-request routing options (#11), a 16-row tier call (#9) and a uniform VRAM
  expert cache under the auto slot budget (#15).
- Engine: a VRAM breakdown in the decode trace (dense weights, DSpark draft, free memory) and per-GPU one-time setup in
  the dense kernels.
