# Xolotl changelog

What landed in Xolotl, newest first. The README keeps a one-line summary of each item and links here.

- [November 2026](#november-2026)
- [Model releases](#model-releases)
- [October 2026](#october-2026)

## November 2026

### Second-card expert tier, faster; Engram tables in IQ4_NL (27 Nov)

- **Second card (`--remote-device N`) for DeepSeek-V4.1 and ERNIE.** The card that holds the next-ranked experts now
  takes prompt work too: a compact wire (only the tokens routed to its experts go over PCIe), MMQ for prompt calls of
  32+ tokens, one metadata copy per block, keep-warm between requests, and a free-VRAM floor before the serve loop grows
  its buffers (fixes an out-of-memory exit on `--fast`). Measured on an RTX 4090 + RTX 3060 12 GB, DS4.1-1017, 4.1K-token
  prompt:

  | setup | prefill tok/s | decode tok/s |
  |---|---:|---:|
  | 4090 alone | 231 | 25.3 |
  | 4090 + 3060, before | 139.5 | 28.2 |
  | 4090 + 3060, `STRATA_REMOTE_HEAD=1` | 247 | 28.5 |
  | + `STRATA_REMOTE_PIPE=1` (blocks double-buffered) | 252.9 | 27.5 |
  | `--fast` + 3060 | 249 | 29.9 |

  ERNIE-5.5-Flash with the 3060: 12.5 tok/s decode, 136 prefill (4090 alone 9.8-10.5 / 111-115).
  `STRATA_REMOTE_HEAD=1` puts the hottest experts on the second card; it and `STRATA_REMOTE_PIPE=1` are opt-in for now.
  Known: on our short34 logit check two cards score a little below one (top-1 75.2 % vs 76.9 %, the same with the old
  and the new wire, so it predates this change); we are looking into it.
- **Engram tables in IQ4_NL, one GGUF per layer** (`--engram L1.gguf:L14.gguf`). 202 GB of FP8 tables become 110 GB.
  With a 2 GiB IQ4 hot set (`ds41_engram_hot pack --dir L1.gguf:L14.gguf`, about twice the rows of the FP8 one) and both
  files on one NVMe: decode 29.11 vs 27.25 tok/s, prefill 244.9 vs 248.3, short34 logits level with FP8 (top-1
  75.24 vs 74.72 %). Keep the tables on a filesystem that reads fast at random; on exFAT decode fell to ~25 tok/s.
  Converter: `tools/v41/engram/engram_iq4nl_cuda.cu` (byte-identical to ggml's `quantize_iq4_nl`, one layer per GPU,
  ~7 min a layer). IQ4_NL Engram idea from ISTA-DASLab's GSQ GGUFs; the GGUF table reader is NektoDron's.
- **CPU: TQP5/6/7 AVX2 row kernel 3.3x** (bit-identical to the scalar reference, `tqp_avx2_exact_test`), and
  `xqt_decode_bench` for the CPU expert lane.
- **ERNIE:** DSpark speculative decoding behind `--dspark` (work in progress: acceptance is still below the reference,
  so it stays off), the DSA indexer scores in column blocks (fixes a graph allocation failure on 25K-token prompts), and
  `STRATA_DENSE_REQUANT=<type>` for the dense matrices.

### Windows: the rest of #148 lands (27 Nov)

Six pull requests from [mehzhure](https://github.com/mehzhure), each one fix from the #148 report:
`strata/platform/win_compat.hpp` for the POSIX pieces MSVC lacks (#149), the Inkling image encoder reads its mmproj
through the fd API MSVC has (#150), `ds4_moe` reads the shards through `tools/ds4/ds4_fd.hpp` with a positional
`ReadFile` and a pread test from 8 threads (#151), `xolotl-lfm3` starts its speech helper with `CreateProcess` (#152),
`tools/lfm3/build-helpers-win.bat` (#153) and the GLM census write-back that now replaces the file on Windows (#154).
Linux behaviour is unchanged (the Linux build of every touched program and the pread test pass).

### Trellis-coded tensor types (TQP5/6/7) (27 Nov)

- **New ggml types** `TQP5`, `TQP6`, `TQP7` (ids 147-149; 2.625, 3.125 and 3.625 bits a weight: 128 weights in 2 + 8K bytes), in
  `tools/xolotl/ggml/xolotl-ggml.patch`: CPU `to_float` / `vec_dot`, CUDA matrix-vector (decode) and dequant, and
  gguf-py ids. The ids sit well above upstream ggml's range and the file carries a GGUF metadata flag, so other
  engines reject a TQP file cleanly instead of misreading it. Re-run `tools/xolotl/fetch-llama.sh` after pulling.
- **Prompt path.** By default a TQP tensor is dequantised and multiplied with cuBLAS. `XOLOTL_TQP_MMQ=1` switches
  to a tiled MMQ kernel for the TQP types only (no other type is affected even when it is set). It is faster on
  prompts and very slightly less accurate, so it stays opt-in.
- **Status: experimental.** On a 13.9 GB Inkling-2-Mini test file the trellis tensors came out measurably closer to the
  parent model than our XQ file (agent-KL cross-entropy 0.19618 vs 0.19939) with benchmarks at parity, but prompts were
  14-26 % slower. We are not releasing that file; a smaller Inkling edition at XQ quality, and an ERNIE-5.5-Flash one,
  are being tested.
- Thanks: trellis-coded quantisation follows [QTIP](https://arxiv.org/abs/2406.11235) (Tseng, Sun, Hou, De Sa) and
  turboderp's [EXL3](https://github.com/turboderp-org/exllamav3); the MMQ tile loader follows the pattern of
  [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)'s IQ*_KT types and the TQ loader in the agentionai
  llama.cpp fork by Laurent Zuijdwijk (MIT); recipe ideas from [nari-labs](https://huggingface.co/nari-labs). The model
  is Thinking Machines Lab's (Apache 2.0), the refusal edit nokeep's; Xolotl is built on
  [Strata](https://github.com/Niko1221/Strata).

### Monitor: the queued count includes requests waiting beyond the engine's batch slots (24 Nov, #147)

- In parallel mode a request enters the server's live set at once and its `queued` flag is cleared, so the Monitor
  showed 0 waiting while the engine had requests queued behind its batch slots. The count now includes live requests
  beyond the engine's slots (one slot when the engine runs requests one at a time).
- Thanks: **lodos2005** (Hakan Yavuz) for the fix (#147, first contribution).

### Windows fixes for the DeepSeek V4.1 engine and its picture reader; a tool-call loop guard in the server (24 Nov)

- **#144** (fast mode on Windows): every `XOLOTL_*` name now works on Windows too (it was copied onto `STRATA_*` on Linux
  only), and the `XOLOTL_SHADOW` / `STRATA_SHADOW` list - and the `.list` that `--build-shadow` writes - uses `;` on
  Windows, as `--engram` does, so a drive letter's `:` no longer splits it. A Windows `.list` built before this is
  `:`-joined: replace the `:` between files with `;` ([XOLOTL_FAST.md](../XOLOTL_FAST.md)).
- **#145** (dense requant cache on Windows): the model path is split on `/` and `\`, so the cache lands in
  `xolotl-cache` next to the model under the model's own file name, and later starts read it back (~25 s of a load).
- **#146** (`lend_mib` / `per_image`): the server's keyed picture path now takes the turn's VRAM lend before the first
  picture it has to encode and stops a `per_image` encoder (standby respawn included) before the prompt reaches the
  engine; a picture already encoded takes neither. New-picture turns were 3.5-4x slower without it on Windows.
- **Loop guard** (`serve/loop_guard.py`, on by default, config `"loop_guard"`): an agent that makes the same tool call 4
  times in a row, or one of its last turn's calls 3 times in its last 12 with the same result, gets one short note
  before its next turn (that prompt only; nothing is blocked).
- Thanks: **kelheor** for #144-#146, each with a reproduction on Windows and a proposed fix; the loop guard and its 32
  tests are from **musicurgy**'s [Delta Inference Engine](https://github.com/musicurgy/delta-inference-engine) 0.95
  (MIT), whose design follows [Project Maya](https://github.com/mw00/project-maya)'s (MIT).

### Inkling-2-Mini LoRA training behind the real harness prompt: prompt carry, bit for bit with the engine (24 Nov)

- **Prompt carry** (`--prefix prompt.json`, or a per-window `"prefix"` / `"context"`): the context before a window runs
  through the inference engine once; each slot visit attends to that slot's cached K/V and starts from its conv state.
  The state after the shared prompt is snapshotted, so windows restore it instead of re-reading it, and a growing
  conversation prefills only its new tokens. `tools/train/harness_subwindows.py` cuts windows to fit behind the prompt.
- **Gate:** `--check T --check-prefix P` matches the engine bit for bit at every prefix tried, up to the full
  23,061-token harness prompt (top-1 64/64). The first version drifted by 3-6e-2: not wrong state, but Inkling's
  f16-accumulated attention product depends on the cell layout, so the trainer now pads its cells to the engine's
  256-multiple and the gate reads sliding slots in the engine's order.
- **Harness r2:** 22,921 coding-weighted sub-windows behind the 23K prompt, held-out loss **1.382 -> 1.166 (-15.6 %)**
  in 300 steps (87 min). LiveCodeBench-20 with the adapter: 13/20 vs base 15/20, inside the noise: lower transcript loss
  is not yet better coding. How-to and numbers: [docs/LORA_TRAINING.md](../LORA_TRAINING.md#inkling-2-mini-the-whole-model-on-one-4090).
- `harness_build_inkling.py --mask-errors`: leave out of the loss an assistant turn whose tool call came back as an error.
- A task adapter trained on 6 teacher runs of a private agent benchmark: held-out loss -28.4 %, but worse on the
  benchmark itself. Both results point at the data (size, whole trajectories, success selection), written up in
  [What we learned about the data](../LORA_TRAINING.md#what-we-learned-about-the-data-24-nov).
- REPEAT_BACK (CUDA) runs in grid-y slabs too, so the backward no longer stops at 65,535 attention cells (untested past
  65K; memory is the next limit there).

### LoRA training for Inkling-2-Mini, the whole model on one RTX 4090, and `--lora` at serve time (23 Nov)

- **`xolotl-inkling --train`** trains rank-16 LoRA adapters on every layer of Inkling-2-Mini (attention q / k / v /
  output / gate / rel, 144 adapters, 9.78M parameters) through the full looped forward (48 slot visits), with the base
  weights frozen and quantized exactly as served. On the QuetzaCodetl harness data (15,227 windows, T <= 2048):
  held-out loss **1.492 -> 1.243 in 100 steps (-16.7 %)**, 1.212 at 400 (-18.8 %). 3.0 / 5.7 / 14 s a step at
  T = 512 / 1024 / 2048, 15.8-17.4 GB VRAM. How-to and gates: [docs/LORA_TRAINING.md](../LORA_TRAINING.md#inkling-2-mini-the-whole-model-on-one-4090).
- **Checkpoints and resume.** `--save P --save-every N` keeps `P-stepNNNNNN.gguf` per checkpoint, `P` = the latest
  adapter, `P.state` = the AdamW state; `--resume P` continues the run exactly; `systemctl stop` / Ctrl-C writes a
  checkpoint before exiting.
- **`xolotl-inkling --serve ... --lora adapter.gguf [--lora-scale S]`** serves an adapter (llama.cpp LoRA GGUF format,
  what `--train` writes). Gate: with the adapter on both sides the training graph and the engine's forward agree bit
  for bit; the fused serve path stays within the same 4 % of the trainer with or without the adapter, while the
  adapter itself moves the logits by ~110 %. Without `--lora` nothing changes.
- **New ggml backward** (in `tools/xolotl/ggml/xolotl-ggml.patch`, inference untouched): `GGML_OP_MUL_MAT_ID_BACK`, the
  activation gradient through frozen quantized experts (CPU + CUDA, every XQ expert type), TANH, L2_NORM, FILL, batched
  GET_ROWS_BACK, the softmax mask gradient, broadcast SUB. Each has a finite-difference test in test-backend-ops
  (`tools/train/test-backend-ops-train.patch`); the softmax mask rule is also checked against a double-precision
  reference (`tools/train/smgrad_ref.cpp`).
- **Harness data for Inkling**: `tools/train/harness_build_inkling.py` (+ `--heldout-only` eval sets) and
  `check_harness_inkling.py`.
- Found on the way: the fused CUDA top-k MoE router and the unfused graph give losses 0.8 % apart on Inkling. The
  kernel is right (they agree to 1.6e-7 at the first slot; new test-backend-ops cases at Inkling's 128 experts / top-8);
  the gap is rounding amplified through the 48 looped visits, the same size as f16 vs f32 attention.

### Inkling-2-Mini: `--vram-budget-mib` fixed, 96 fewer decode launches, numeric reasoning effort (23 Nov)

- **`--vram-budget-mib` works again** (the mode that splits experts between VRAM and pinned RAM for 12-16 GB cards).
  Two bugs, both on prompt chunks: (1) every expert absent from an arena mapped to one shared zero row, so a token
  could name that row several times, and llama.cpp's MMQ id helper assumes a token uses an expert at most once: the
  unwritten slots sent `quantize_mmq_q8_1` out of bounds (illegal memory access). Each top-k position now has its own
  zero row. (2) The cold experts were staged into a new graph *leaf*, which ggml-alloc never frees, so every layer's
  staging stayed resident (13-24 GB of graph buffer at 128K) and MMQ asserted on the padded view; now a DUP node.
  Greedy tokens match the all-in-VRAM run. Decode with cold experts is PCIe-bound (XQ at a 10 GB budget: 13.5 tok/s).
- **Decode, -96 launches a token, bit-identical** (teacher-forced NLL through the decode path unchanged to the bit):
  the K and V attention-prepare kernels run as one launch (`XOLOTL_INKLING_PREP_PAIR=0` = two), the q and rel clips as
  one op writing one buffer (`XOLOTL_INKLING_CLIP_PAIR=0`), and the loop controller's down projection reads its
  activations once per 4 rows (7.06 -> 6.13 us; `XOLOTL_INKLING_CTRL_RB`). ~130 -> ~132 tok/s in-process.
- **Numeric `reasoning_effort`** for Inkling (0 = off .. 0.99), as llama.cpp and SGLang clients send it in
  `chat_template_kwargs`; the names keep their meaning.
- **Loop count.** `XOLOTL_INKLING_PASSES=N` (research only) runs the loop core N times. WikiText 8K perplexity: 2 passes
  12.06, **3 (trained) 6.71**, 4 11.23, 6 7.07 - the three-pass cycle that
  [drhb's looped-transformer study](https://drhb.github.io/blog/posts/looped-transformers/) found; 3 stays.
- **The exact NVFP4 parent** on one 4090: 97 tok/s decode with everything in VRAM up to 32K context (k8v4); above
  that its BF16 layer-20-23 experts spill and decode is PCIe-bound (`--cpu-exps 20-23`: 36 tok/s at 128K).

### Inkling-2-Mini decode 118 -> 127 tok/s: fused loop controllers, folded expert scales, BF16 router (22 Nov, late)

Peer engines for this model (satellitedown/skyne98's ingine, 140 tok/s on an RTX 3090; ManyGlue0's InklingEngine) run
about 450 kernels a token; we ran 2,111. A decode token reads ~3.8 GB of weights (1.8 GB dense with the 3x loop,
1.6 GB of experts, 0.5 GB head) and our weight mat-vecs already ran at ~950 GB/s, so the gap was the small kernels.
- **Loop controllers** (24 visits a token, ~25 kernels each for the q/k/v and FFN controllers): one fused op, three
  kernels, reading BF16 copies of the controller weights. The checkpoint stores them as BF16 and the GGUF's F32
  tensors hold exactly those values, so the copies are exact and half the bytes.
- **Per-expert NVFP4 scales**: the gate/up scale is applied inside the SiTU kernel and the down scale inside the
  fused MoE finish (the same fp32 multiply, bit-identical), removing a gather and a multiply each a visit.
- **Router** read as its exact BF16 copy for decode.

2,111 -> 1,482 kernels a token, GPU time 7.48 -> 6.80 ms. Served decode, k8v4, no drafter:

| context | 7.3K | 30K | 117K | 477K |
| --- | ---: | ---: | ---: | ---: |
| before | 118 | 108 | 93 | 55 |
| after | **127** | **118** | **100** | **56** |

With the DSpark drafter, code requests run ~150 tok/s. Quality: 8K NLL in 16-token steps at the f16 run-to-run floor
(controllers vs the chain 0.0205 per-token, mean -0.0004; scale folds bit-identical). Switches back:
`XOLOTL_INKLING_CTRL_FUSED=0`, `XOLOTL_INKLING_SCALE_FOLD=0`, `XOLOTL_INKLING_ROUTER_F32=1`.

### Inkling-2-Mini long-context decode near the memory bandwidth: 477K tokens 30 -> 55 tok/s (22 Nov, night)

Profiled with Nsight Systems at
117K: the global layers' decode attention took 571 us a visit (~340 GB/s of K/V), the merge of its splits ~20 us a
call, 48 calls a token, and the host filled and uploaded a `[cells, T]` mask and distance array every token.
- A new decode attention kernel scores **one cell per lane**: each lane holds its cell's whole K row in registers
  and dots it with q from shared memory, the relative-position bias folded in, so a score needs no cross-lane
  reduction; then p v with lanes over dims, 16 V loads in flight. 571 -> 232 us a visit at 117K (~840 GB/s).
- The split merge runs parallel over the splits (8 warps, coalesced) instead of every thread walking all of them.
- The global layers' mask and distance come from the token's position inside the kernel (cell j = position j):
  no host arrays (3.8 MB a token at 477K). Bit-identical to the arrays.

Served decode, k8v4, no drafter, 400 tokens after the prompt, RTX 4090:

| context | 7.3K | 30K | 117K | 477K |
| --- | ---: | ---: | ---: | ---: |
| before | 108 | 93 | 65 | 30 |
| after | **118** | **108** | **93** | **55** |

Quality: 8K wikitext NLL in 16-token steps (the decode kernel on every position) stays at the f16 run-to-run floor
for f16 and k8v4, 32K mean NLL change -0.00005 (se 0.0005); the 477K needle and the drafter's verify path checked.
Each piece has a switch back (`XOLOTL_INKLING_DECATTN_LANE=0`, `_RED2=0`, `_POS=0`, `_STREAM=0`).

### Inkling-2-Mini: 512K context on one RTX 4090 with k8v4 on the fast kernels, 4-bit V leg Hadamard-rotated (22 Nov, evening)

**Quantised K/V on the fast kernels.** `--kv-type k8v4` (q8_0 keys, q4_0 values on the global layers) and `q8_0` used
to fall back to the reference op chain, about 130 tok/s of prefill at 100K. The tensor-core prompt attention now stages
the raw q8_0 / q4_0 rows with cp.async and dequantises them into its f16 tile, decode attention reads them in place,
and the decode prepare kernel (K/V conv, key clip, value norm, cache store) writes q8_0 / q4_0 rows itself. A
107,800-token prompt at k8v4 prefills in **29 s (~3.7K tok/s)**; decode at 32K context is **90 tok/s at k8v4**
(89 at f16, 86.6 before the prepare kernel took quantised rows). Gate: fast vs chain KL 3.1e-3 at k8v4 (f16 run-to-run
floor 2.0e-3); a 107K-token summary is correct.

**512K context on one RTX 4090** (`--context 524288 --kv-type k8v4`, with the DSpark drafter and the image tower
loaded: 12.5 GiB of weights + 6.8 GiB of K/V state, 23.4 GB of VRAM in use). A 477,603-token prompt (wikitext + the
four Vedas, one planted sentence at 10 % depth) prefills in ~218 s (~2.2K tok/s) and the model quotes the planted
sentence exactly; decode at that depth ran ~27 tok/s with the drafter on (55 tok/s plain after the kernel work
below). Two things stood in the way:
- Decode at long context fell off the fast kernel: the decode window was split into chunks sized for the old
  chain's score tensors (`--attn-budget`), and past ~100K cells that meant several chunks, which the split-K kernel
  does not take. Decode then fell back to the chain, which asked for a 17 GB graph buffer at 477K. Decode and verify
  steps now plan one chunk at any length (the kernel reads K/V in place and builds no score tensor).
- The kernel kept one partial result per 64 cells: ~93 MB per token at 477K, and ~3 GB across the drafter's verify
  lengths, which each keep their own graph buffer. Splits are now capped at 512; past 32K cells a split walks
  several 64-cell chunks and joins them with an online softmax. Up to 32K context nothing changes (bit-identical
  output); with the cap forced low the result stays at the f16 noise floor.

**Hadamard rotation of the 4-bit leg, default on** (`XOLOTL_INKLING_KV_ROT`, 0 = off, 2 = every leg). A q4_0 leg is
rotated by a randomised Hadamard transform before it is stored, which spreads outlier channels across each 32-value
block; on k8v4 that is the values, with the attention output rotated back once per head. The q8_0 keys stay plain:
they are near-lossless without it (q8_0 noise = the f16 floor on Inkling; int8 KL 3.4e-4, top-1 99.5 % on the dense
27B). The decode prepare kernel applies the rotation itself (a shared-memory FWHT), so it costs no speed. Per-token
|dNLL| against f16, wikitext:

| k8v4 | 32K prompt path | 8K in 16-token steps (decode kernels) |
| --- | ---: | ---: |
| f16 run-to-run floor | 0.0395 | 0.0359 |
| no rotation | 0.0483 | 0.0438 |
| K and V rotated | 0.0459 | 0.0403 |
| **V rotated (default)** | **0.0454** | **0.0403** |

Mean perplexity is unchanged in every arm (within one standard error).

**Fixes, all in the released engine too:**
- Graph reuse: the prompt attention kernel bakes the chunk's start position into its op params, but the graph key had
  no start position, so a prompt chunk with the same length in the same 256-cell bucket reran at the previous chunk's
  position (wrong attention on multi-turn follow-ups that hit it). Dump gate: KL to the chain 0.258 -> 0.002.
- The prompt kernel's dequantise step read cp.async-staged rows without a block barrier (cp.async.wait covers only a
  thread's own copies): garbage logits on long quantised prompts.
- The DSpark drafter kept a full-context f16 cache on its global layer (2 GiB at 512K, the allocation that refused to
  load, and an O(context) cast per draft step). It is now a ring of the last `XOLOTL_DSPARK_WINDOW` positions
  (default 32768; 0 = the old cache).

### Inkling-2-Mini XQ re-released as "Unbound" (22 Nov)

[IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) now holds the **Unbound** build, under the same file name and
size (13.93 GB), so existing links and configs keep working; the first (ARA) file stays in the repo's history.
[nokeep](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) found that a plain refusal-direction projection does
not survive on this model: the short causal convolution after each attention output mixes neighbouring positions, so
19-88 % of what a projection removes leaks back. Their edit projects through a convolution-aware subspace and keeps
row norms, on the 20 attention-output projections of layers 4-23; no expert is touched. This build applies it at
strength 4 (they ship 3): on a sweep, 3 / 4 / 5 gave greedy MATH-24 87.5 / 87.5 / 83.3 % and 34 / 24 / 21 refused of
100 harmful prompts. The XQ experts are unchanged from the first release.

| RTX 4090, Xolotl's engine | first release (ARA) | **Unbound** |
| --- | ---: | ---: |
| MATH-24 sampled, 3 seeds (thinking tokens) | 83.3 % (1,024) | **88.9 % (911)** |
| MMLU-100 greedy / MMLU-30 thinking | 76 % / 87 % | 71 % / 86.7 % |
| HumanEval (21) / tool choice (10) / agent loops (3) | 19 / 8 / 3 | **21 / 9 / 3** |
| WikiText-2 perplexity (same engine) | 4.554 | **4.496** |
| KL to the official model, 8,192 held-out tokens (top-1) | - (its bf16 parent: 0.0336, 92.1 %) | **0.0282 (92.8 %)** |
| harmful / harmless refused of 100 (BERT gate, thinking off) | 4 / 1 | 24 / 0 |
| Gauntlet v3, 7 tasks, thinking on | not run | 4/7 solved |
| Gauntlet v3-small, 6 tasks (the README table's score) | not run | 4.42 of 6 (3 solved) |

MMLU-100 moves ~4.5 points on a 100-question sample. Most of the 24 counted refusals are partial or vague answers; if
you need zero, nokeep's own release at strength 3 measures 0/277 on their sealed set with a judge (a different gate).

### Inkling-2-Mini prefill 1.2K -> 4.2K tok/s at 32K, decode 93 -> 113 tok/s, DSpark 136 tok/s (22 Nov)

**Prompt attention on the tensor cores** (a port of Basalt's `attn_prompt.cu`, by Jesus Gulfo, MIT): a prompt chunk
attends flash style over the f16 K/V cache in place, with the relative-position bias and the attention sink computed
inside the kernel (the bias is one extra small tensor-core product over the band of distances a key tile spans). The
old path built a `[cells, rows, heads]` score tensor and host masks, so its cost grew with context and it had to keep
chunks at 512 rows. Without it, prompt chunks grow to **2048 rows**, which reads each expert once for four times as
many tokens. Measured on the RTX 4090, 14 GB XQ quant, served:

| prompt | before | after |
| --- | ---: | ---: |
| 8K tokens | ~1,670 tok/s | ~3,500 tok/s |
| 32K tokens | ~1,210 tok/s (27 s) | **4,190 tok/s (7.8 s)** |

Also for prompt chunks: the MoE weighted sum over the 8 experts is one bandwidth-bound kernel instead of a batched
mat-vec over a transposed copy (-24 % of prefill GPU time at 2048-row chunks). The chunk size is picked at load from
the VRAM left after the weights (2048 / 1024 / 512 / 256 rows: the quant at 36K-131K context gets 2048, the NVFP4
parent at 32K gets 256); `--ubatch N` still overrides it. `XOLOTL_INKLING_PROMPT_ATTN=0` restores the old path.

**Decode** (all default on, each with an env switch back to the old path): split-K decode attention that reads the f16
K/V window in place with the relative bias in-kernel (`XOLOTL_INKLING_DECATTN=0`), the attention prepare (K/V
convolutions, key clip, value norm, cache writes) in two kernels, the MoE finish (weighted sum, post-conv, residual) in
one, and the verify step's conv history fused. Plain decode **93 -> 113 tok/s**; with the DSpark drafter (`--dspark`)
**136 tok/s** on a 2.8K-token prompt (2.4 of up to 7 drafts accepted per step) and ~140 tok/s on served thinking
requests. The NVFP4 parent on the same engine: 89 tok/s plain, 137 tok/s with DSpark.

**Quality:** this model's logits move at the rounding level with any kernel change (near-tied expert picks flip), so
each change was judged by paired NLL and KL against the NVFP4 parent on 2 x 1024 held-out tokens, prefill and decode
paths: prefill NLL 1.1967 -> 1.1987, KL to the parent 0.0364 -> 0.0369; decode KL to the parent 0.0377, unchanged.
Both inside the noise between runs of the old path.

Also: Inkling-2-Mini is **3B active** per token (the official count), about 5.4B effective because its 12-layer loop
core runs 3 times; the 21 Nov notes said "5.4B active". The example config's sampling now matches the model card
(temperature 1.0, top_p 1.0, top_k 64 = the engine's maximum).

### Inkling-2-Mini decode 67 -> 93 tok/s: fused conv, SiTU and stream-stack kernels (21 Nov, evening)

**Inkling-2-Mini decode 67 -> 93 tok/s** on the RTX 4090 (the 14 GB XQ quant, 256-token
greedy; 93-96 tok/s in served evals): the depthwise conv step, SiTU and the residual-stream stack are one CUDA
kernel each (bit-identical logits to the op chains they replace), q/rel/gate are one stacked mat-vec per token, and
the loop controllers run as one batched chain. Needs the updated ggml patch (one new hook, no effect on other
engines): `bash tools/xolotl/fetch-llama.sh <your ggml dir>`, then rebuild. Quick quality screen of the released
quant against its parents (greedy, thinking off): MMLU-100 76 % (original 73 %, abliterated parent 74 %),
HumanEval-21 19-21/21 for all three, tool choice 8/10 and agent loops 3/3 for all three. Worth knowing for anyone
measuring this model: its logits are chaotic at the rounding level (near-tied expert picks flip), so any engine
change - even a different prefill chunk size - moves held-out KL by ~0.005 over 2K tokens.

### Inkling-2-Mini XQ, abliterated, released (21 Nov)

**Inkling-2-Mini XQ, abliterated**: 13.93 GB, passed its quality gates against the
abliterated parent (held-out KL 0.013, top-1 96.3 %) and refuses 4/100 harmful prompts (1/100 harmless). On one RTX
4090 through Xolotl's server: **1,162 tok/s prefill at 8K, 59.7 tok/s decode**, the same as the 22.9 GB parent.
On [IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored)
with the model's DSpark (MTP) drafter and vision projector as optional files; the agentic Gauntlet is running.
Engine: speculative decoding with the DSpark drafter (`--dspark`, experimental)
and quantised K/V caches (`--kv-type q8_0|k8v4`). Decode is kernel-launch bound (16.7 ms a token against a ~3.4 ms
memory floor), so kernel fusion toward 100+ tok/s is next, then vision (first step: 93 tok/s, above).

### Inkling-2-Mini on its own engine, `xolotl-inkling` (21 Nov)

[Inkling-2-Mini](docs/INKLING.md) (Thinking Machines, 26B looped MoE, 5.4B active per token) on
its own engine, `xolotl-inkling`: **57 tok/s decode, 1.1K tok/s prefill** on the RTX 4090 from the exact NVFP4
checkpoint, greedy output matching the llama.cpp reference. The graph is ported from llama.cpp PR #32694 by Pedro
Cuenca (MIT); a loop-pass expert-scale bug in it is fixed here. First numbers: fused kernels, flash attention
and MTP drafting are next.

Correction (21 Nov, evening): Inkling-2-Mini is 5.4B active per token, not "about 3B". About 3.1B distinct weights
are read per token (1.6B shared + 8 of 128 experts), but layers 6-17 run three times, so the per-token compute is 5.4B
(Thinking Machines' figure).

### LoRA training proven end to end on QFN: a harness adapter trained from the served engine's activations (21 Nov, later)

**A real fine-tune works** - [docs/LORA_TRAINING.md](../LORA_TRAINING.md#how-it-works-a-lora-for-a-125b-moe-on-one-gaming-pc).
A rank-16 LoRA on Qwen3.8-Flash-Next's top four layers, trained on the RTX 4090 from activations the **served engine**
captured (prefix + window prefill, quantized base, YaRN rope, layer 47's own sparse selection replayed), with the real
next-token loss on agent transcripts (QuetzaCodetl / Claude Code sessions, assistant turns only). On 12 windows for 36
steps (~6 min) the loss on two held-out windows fell **1.4116 -> 1.3197 (-6.5 %)**. Before training, the trainer's
forward reproduced the engine on a real 21,686-position window (loss 1.7798 CPU / 1.7780 GPU vs the engine's 1.7795).
New: the harness data builder and request capture, the capture driver, the QFN top-4 stack with the prefix carry, the
GPU trainer (`tools/train/qfn_harness_train.cpp`) with live loss graphs (`lossplot.py`). Two CUDA fixes for training
shapes in `xolotl-ggml.patch` (OUT_PROD with a cuBLAS-invalid view stride, REPEAT_BACK beyond the grid-z limit) - both
backward-only ops, so inference is unchanged; rebuild after `fetch-llama.sh`.

**Option, experimental: fast capture through `--serve`** - one engine, the root checkpoint at the prefix end: 2.6-3.6 s
per window instead of 66-137 s. Opt-in engine flag `STRATA_KVA_CAPTURE_PER_REQ=1` (off = unchanged). **Its window rows
do not match the slow capture yet** (median relative error 0.12-0.13 per row, prefix rows bit-identical), so do not
train on it until that is explained - details in the doc.

### LoRA training (experimental), the Hy4-flash engine, FP8 Qwen4-Exp n-gram tables, Windows build fix (21 Nov)

**LoRA training, experimental** - [docs/LORA_TRAINING.md](../LORA_TRAINING.md). Backward kernels for training adapters
on the MoE models Xolotl serves with the base frozen and quantized as served: OUT_PROD with a quantized weight on CUDA,
the backward rules ggml lacked (SIGMOID, SWIGLU_CLAMP, CONCAT, CLAMP, GET_ROWS_BACK, SOLVE_TRI upper, TRI, CUMSUM,
SSM_CONV, PAD), the chunked gated delta net and the MoE expert loop as differentiable graphs, a Hy4-flash trainer and
QFN trainers on real pack weights, adapters written in llama.cpp's LoRA GGUF format (llama.cpp loads them). Every piece
has finite-difference and reference gates. **Real fine-tuning is not proven yet** - the runs so far fit a fixed target.
Nothing changes for inference: every ggml change is in the backward pass or in SOLVE_TRI's new upper variant, and
DeepSeek-V4.1 greedy output (96 tokens, served settings) is identical to main's when both runs get the same number of
VRAM expert slots - the slot count follows the card's free VRAM, and one slot more or less moves main's own output too.

**Hy4-flash engine** (`tools/hy4/`, built only when asked): geometry + tensor map, the Engram shard writer (F8_E4M3,
byte-exact vs Tencent), the dense half on CUDA passing the golden gate, experts on the DeepSeek-V4.1 tier, k-pool
selection. `ds41_moe.cpp`'s expert cap is now `DS41_MOE_MAX_K` (default 8: DeepSeek-V4.1 builds unchanged).

**FP8 Qwen4-Exp checkpoints from upstream llama.cpp's converter** (#32476) load: the n-gram table arrives as GGML type
43 = F8_E4M3 with its scale in `per_layer_token_embd.scale`. 43 is also MXFP8 in older DeepSeek-V4.1 GGUFs (llama.cpp
#30817, closed unmerged), so the reader tells the two apart by the tensor's size; MXFP8 files read exactly as before.

**Windows:** `src/kernels/cuda/expert_keep.cu` builds with MSVC again (#143, reported and verified by
[kelheor](https://github.com/kelheor)). Also: the Q2_0 AVX2 row kernel unpacks codes in 7 ops instead of ~8.5
(llama.cpp #30447, homeofe; bit-identical, one token 1.28-1.58x faster a row), opt-in `STRATA_TRACE_DIR` live
thinking traces in serve, a serve test that no longer breaks other tests under `unittest discover`, and ERNIE 5.5
Flash (`ernie5_5`) is recognised - geometry and tensor map - and refused with a clear message (no execution path yet).

### DeepSeek-V4.1 decode switches on by default, agent tasks 42.7 tok/s (19 Nov)

`tools/v41/serve/ds41-engine.sh` now turns on [Keisawaakira](https://github.com/Keisawaakira)'s decode set
(`tools/v41/engine/DECODE_OPTINS.md`): routing copies fused and read on the card, the metadata pulled once, the
prefetch waiting on the host, the hit fast path, async dense, pinned route buffers, fused QKV.
`XOLOTL_KEIS_DECODE=0` turns it off; any single switch takes its own `=0`.

Measured on an RTX 4090 + Ryzen 7 5700X, DeepSeek-V4.1-Flash-1017 V4 XQ, dense Q6_K, fast mode:

| test | before | after |
| --- | ---: | ---: |
| 8K prompt + 512 tokens, verify pass | 98.8 ms (34.3 tok/s) | **89.1 ms (37.4 tok/s), -10 %** |
| Gauntlet v3-small tasks 01 + 03, decode during the tasks | 39.6 / 40.8 tok/s (two runs, 7 Nov) | **42.7 tok/s (+5-8 %)** |
| short34 NLL / top-1 | 0.4614-0.4653 / 88.42-88.65 % (noise band) | 0.4651 / 88.61 % |

Not bit-identical (the sums move); quality is inside the noise. With `STRATA_RB_PROTECT` set, the attention tail and
the two switches behind it (`STRATA_RB_HOOK`, `STRATA_GRAPH_CONC`) stay off and `STRATA_ROUTE_FUSE` is skipped, so the
numbers above come from the other seven; making the tail work with the route meter is the next job.

Also on this branch, opt-in:
- `STRATA_LA2=1 STRATA_D3_SPEC_SRC=la2 STRATA_D3_SPEC=2`: layer l's attention graph also guesses layer l+2's experts,
  and the NVMe-tier ones are read early (the design of [TokenWing](https://github.com/runmyai/tokenwing)). Time blocked
  on NVMe reads per pass falls 16.4 -> 9.6 ms (-41 %, three interleaved rounds); the pass itself gains 3-7 %, inside the
  run-to-run spread, so it stays off.
- GLM-5.3-Flash expert tier on a second card: `glm_generate --remote-device N --remote-gib G`, checked exact to rounding
  on one card (`STRATA_REMOTE_SAME_DEVICE=1 STRATA_REMOTE_CHECK=1`).

Full record: [docs/FIELD_PORTS_2026_11_19.md](../FIELD_PORTS_2026_11_19.md).

### Looped transformers in the dense engine: SenseNova-Loop-9B (18 Nov)

`xolotl-dense` runs models that repeat a block of layers per token. SenseNova-Loop-9B (a Qwen3.5-9B body) runs layers
4-27 twice, so 32 layers make 56 executions, and inside that block it carries two residual streams mixed by Sinkhorn
routers (mHC). The engine keeps one execution slot per run of a layer: a repeat shares the weights (no second copy in
VRAM) and owns its own KV cache or GDN state, as the reference does. The file's loop and mHC settings come from the
GGUF (arch `qwen35loop`, converter `tools/sensenova/convert_sensenova_loop.py`). `--layer-schedule 0-3,4-27,4-27,28-31`
loops any dense file the same way, without copying weights. The file's YaRN is applied only with
`STRATA_DENSE_FILE_ROPE=1`.

Measured on an RTX 4090, Q8_0, 768 tokens of code, prose and agent text, against the Hugging Face model in FP32 (no
YaRN on either side; `tools/sensenova/hf_ref_logits.py`):

| run | top-1 | KL mean |
| --- | ---: | ---: |
| xolotl-dense, Q8_0 | 95.96 % | 0.0289 |
| xolotl-dense, Q8_0, `STRATA_DENSE_ACT_F32=1` | 97.14 % | 0.0146 |
| Hugging Face with Q8_0 weights (rounded in FP32), for reference | 97.27 % | 0.0168 |

- **The prompt goes through the batched MMQ prefill** with the mHC block in it (same kernels as decode, a chunk of
  tokens at a time): 318 -> 3,027 tok/s on a 2K prompt (decode 41.5 -> 45.4 tok/s, run-to-run noise). `--check` passes
  (MMQ prefill vs one token at a time: 1.28 %, the usual MMQ figure; the 27B is unchanged at 1.12 %).
  `STRATA_DENSE_MHC_PREFILL=0` returns to the old path.
- **`STRATA_DENSE_ACT_F32=1` (opt-in):** Q8_0 projections multiply the FP32 activation instead of its q8_1 copy. It halves
  the KL above and matches the Hugging Face Q8_0-weights run (KL 0.002 between them). It costs decode speed (45.4 ->
  37.7 tok/s) and sends the prompt through the one-token path (275 tok/s), so it stays off by default.
- **`STRATA_DENSE_PROF=1`** has an `mhc` bucket: on this model the mHC steps take 4.9 of 24.5 ms per token.
- **`xolotl-dense --logits`** writes teacher-forced log-probabilities for parity checks; `XOLOTL_LOGITS_PREFILL=N` takes
  every N-th position through the batched prefill.

The reference implementation is SenseTime's (the model's `modeling_sensenova_loop.py`).

### `--conversation-cache-exchange`: two big conversations stop evicting each other (18 Nov, opt-in, not yet measured)

When the server switches between two conversations that each fit the conversation cache but not together, the
incoming one was counted against the budget while the outgoing one was parked, so the park was refused and the bigger
conversation re-read its whole prompt on every turn (seen on Qwen3.8-Flash-Next: ~76 s per turn on a 300K
conversation). With `--conversation-cache-exchange` the park is retried without counting the incoming image, which is
freed moments later; the physical-RAM check still applies. A unit test replays the measured sequence on the real cache
class. It has not yet been measured with two live clients, so it is off by default.

### Keenable as a web search choice, with or without a key (17 Nov)

Settings > Tools > Web search has a fifth choice beside the keyed services: Keenable (no key needed). Chosen, the agent's
searches go to Keenable (`createKeenableSearch` in `apps/host/server/web-search-keyed.mjs`, the same shape as Exa's: titles,
links and snippets, search only - the fetch provider does not change). Unlike Exa, Perplexity and Keirolabs it works with no
key: the public endpoint (`POST /v1/search/public`, `X-Keenable-Title: Xolotl`) is rate limited per address. An optional saved
key switches to the keyed endpoint (`X-API-Key`) and lifts those limits; the key travels only as `STRATA_SEARCH_KEY`, as
before. The no-key default stays DuckDuckGo - Keenable is used only when the person selects it. Proposed by KGKallasmaa
(#139), who works with Keenable and said so in the issue.

### Qwen3.8-27B with DFlash2: the 8-token verify gets cheaper, 171 -> 204 tok/s (14 Nov)

DFlash2 drafts 7 tokens and checks all 8 in one pass, so every GEMV of that pass multiplies 8 columns. Those 8-column
GEMVs were the slow part: a Q5_K matrix ran at ~600 GB/s with 8 columns against ~900 with 1-4, the IQ4_XS down
projection at 246. Nsight Compute on the Q5_K kernel at 8 columns shows why: the load/store pipe is the bottleneck
(78 %, DRAM only 72 %), because each lane issues one 4-byte activation load per column per int. Two changes, both
bit-exact and default on:

- **B6 layout in the dense engine.** `xolotl-dense` never applied the card's default switch table, so the B6 multi-column
  layout (each column's activations loaded once per weight block and reused across rows) that the other engines get on
  sm_60/86/89 was off. It now takes `STRATA_B6_MMVQ_ROWS` from that table (`=0` turns it off): 171.3 -> 184.8 tok/s.
- **The interleaved activation copy at 2-8 columns.** `native_mmvq_il` (one 16-byte load reads a position of every
  column) already ran the MoE verify, but the dense engine never used it, and an RTX 4090 had no table for 5-8 columns,
  so those widths fell back. `xolotl-dense` now writes the interleaved copy once per input and takes it for every
  GEMV that supports it, and the 4090 (sm_89 with 100+ SMs) has its own 5-8 column table (`STRATA_MMVQ_IL=0` /
  `STRATA_MMVQ_IL_4090=0` turn them off). At 8 columns: Q5_K 571-598 -> 728-798 GB/s, IQ4_XS 246-625 -> 811, Q6_K head
  636 -> 825. Decode 186.0 -> 204.1 tok/s.

Measured on Qwen3.8-27B XQ (13.7 GiB of weights, not released yet), DFlash2 k7, a 2K synthetic prompt + 256 tokens,
interleaved rounds in both orders. That prompt is the best case for a drafter (almost every draft is accepted); on real
agent work DFlash2 accepts 40-53 % of drafts, so expect less there - the Gauntlet run with DFlash2 is next. Every step was
checked bit for bit: the kernel bench against the old kernels and single-column calls, `--selftest`, and 512 greedy tokens
identical with the change on and off, DFlash2 against plain decoding, and against the previous build. DFlash2 stays
opt-in (`--dflash2 <pack>`) until the Gauntlet says it beats MTP on real work. `dense27_mmvq_bench` (27B shapes, GB/s per
format and width) gained the interleaved path as a column.

### Muse Glimmer 30B natively at 512K (13 Nov)

`xolotl-glimmer` (`src/program/glimmer_main.cpp`) serves Meta's dense Muse Glimmer 30B: llama.cpp's muse-glimmer graph on
ggml-backend CUDA with the xolotl-lfm3 engine around it (serve protocol, prefix reuse, a checkpoint of the sliding rings,
park/unpark). At 512K the 13 global layers (no position encoding) hold the context's K/V (q8, 3.5 GiB) and the 39 sliding
layers a 2,560-cell ring: 14.4 GiB of weights + 3.5 GiB of K/V on a 24 GB card, the token table in RAM. YaRN is not needed
(llama.cpp keeps the sliding layers at their trained rope scale), so a `--rope-scaling yarn` setting changes nothing for
this model. `serve/atem.py` reads its turn format (`to=self` reasoning, `to=user` answer, ATEM `<atem:invoke>` tool calls)
as the Qwen form the server knows; `arch: muse-glimmer` sets its stop tokens, sampling (0.6 / 0.9 / 20) and effort words.
`tools/strata_tokenizer.py` gained the `llama4` pre-tokenizer (ids equal llama.cpp's on 77 test lines); pictures go through
`xolotl-glimmer-vision` (`tools/glimmer/build-helpers.sh`). Served chat, tool call, tool round trip, thinking off and a
picture all correct. Prefill 3,110-3,130 tok/s and decode 46-48 tok/s at 512K (llama.cpp wrapper on the same
GGUF: ~763 / 39-42). Parity with llama.cpp over 4,096 tokens at equal batch size: mean KL 1.0e-3, 99.0 % the same top
token, closer than llama.cpp at ubatch 512 is to itself at ubatch 256 (1.5e-3). `--ngram 8` (prompt-lookup drafts
checked in one pass, kept only when they equal the sample plain decode draws) decodes copy-heavy replies 2.4-5.7x faster
(116-279 tok/s) and free writing at 0.98x. `--dflash` runs Meta's DFlash block drafter natively (taps of layers 2/14/26/38/50
fused into the drafter's ring inside the model's own graph, one non-causal draft pass, the same exact check): 186-379 tok/s
on copy-heavy replies (3.9-8.0x), 1.3-2.4x on prose, output equal up to rounding. [docs/GLIMMER.md](../GLIMMER.md).

### Keirolabs as a keyed web search service (13 Nov)

Settings > Tools > Web search has a fourth keyed choice, Keirolabs (your own key), beside Exa, Perplexity and DeepSeek. With
its key saved, the agent's searches go to Keirolabs' lite search (`createKeirolabsSearch` in `apps/host/server/web-search-keyed.mjs`,
the same shape as Exa's: titles, links and snippets); the key reaches the agent only as `STRATA_SEARCH_KEY`, as before. Without a
key nothing changes, and the rate-limit sentence still names only Exa, Perplexity and DeepSeek. Contributed by Manasbh (#136),
who works at Keirolabs and said so in the PR.

### The ggml patch applies again on a Windows checkout (12 Nov)

`tools/xolotl/ggml/xolotl-ggml.patch` stopped applying on Windows with `core.autocrlf=true` after #133: the `top-k.cu` part has
11 empty context lines, and a CRLF copy of a bare empty line is no longer a valid context line (`corrupt patch at line 1857`).
`.gitattributes` now keeps every `*.patch` byte for byte (`-text`), and those 11 lines carry their leading space. On Linux the
patch applies at 6050b467, passes the reverse check CMake runs, and gives the upstream `top-k.cu` blob 19cb45d1. A plain pull
fixes an existing Windows clone: the patch file itself changed, so git writes it out again under the new rule. Reported, diagnosed and worked around by trojy13 (#134).

### An NVIDIA L20 gets row tables it was measured on (11 Nov)

An L20 is compute capability 8.9, so it ran `kIlRows_89` / `kIl1Rows_89` - tables read off an RTX 4070 (46 SMs) and an RTX
4080 (48). An L20 has 92 SMs and 864 GB/s of GDDR6 on a datacenter part, and nothing in the tree said so. `il_cc()` now
answers `891` for an Ada card with 80-99 SMs, and that band takes `kIlRows_891` and `kIl1Rows_891`, read off three
`mmvq_il_parity --bench --emit-table` runs on one. A cell is kept only where the runs agree:

| The three runs on one L20 | cells | kept a split | agreed to stay plain | did not agree twice |
| --- | ---: | ---: | ---: | ---: |
| 2-4 columns (what a verify window asks) | 72 | 15 | 16 | 41 |
| one column (what a decode of one token asks) | 24 | 2 | 8 | 14 |

30 of the multi-column cells differ from `kIlRows_89`: 22 splits the consumer table has and these runs did not confirm, 4
splits it lacks, 4 where the count itself differs. The one-column band is only Q3_K's two widest classes (2560x10240
27.4/44.7/44.9 -> 18.6/17.8/18.2 us, 2560x12288 30.1/48.1/48.4 -> 23.2/32.2/21.3 us). `STRATA_MMVQ_IL_891=0` puts the card
back on `89`; `STRATA_DEFAULTS=026` does the same to everything since 0.2.6.

Exactness: the interleaved kernel is bitwise `native_mmvq`, and `mmvq_il_parity` passes in all four states (default,
`STRATA_MMVQ_IL_891=0`, `STRATA_DEFAULTS=026`, `STRATA_MMVQ_IL_89=0`). The greedy answers were identical in every arm below -
the same 256 `T` lines, the same sha256.

| Same binary, on against off, 4,000-token prompt, 256 tokens, 3 rounds (tok/s) | tables on | tables off |
| --- | ---: | ---: |
| 27B Q4_K_M, no draft head: prompt read / decode | 1,483.7 / 42.27 | 1,487.9 / 42.29 |
| 27B Q4LynnStyle + MTP `--draft-max 2`, prompt read / decode | 1,403.2 / 86.61 | 1,406.4 / 86.64 |

Both pairs are inside the noise, and that is the honest reading: a dense decode reads one token at a time and a prompt goes
through llama.cpp's MMQ kernels, so a 2-4 column cell is only reached by a verify window. What the tables do is stop this
card from running a split it loses with.

A third pack was tried for the multi-column half and it says the same thing from the other side: `ds41_generate` never looks
a table up at all. The interleaved kernels' only call site is `src/core/verify.cpp:1507`, which that engine does not
include, and its dense half links ggml. Six rounds an arm, with the box's other three cards idle, put both settings inside a
0.1 % prompt-read spread and a decode sign that follows which arm started - and that pack's greedy answer at one offset
changes with what ran before it, so the switch never gets a reading there at all. Numbers in the card guide.

**The sm_89 switch stack, measured here.** `xolotl-dense` never calls `apply_nv_defaults`, so the nine exact switches of
`nv_default_env(8, 9)` - +8.1 % decode on an RTX 4090's MoE model - are simply unset on the dense engine. Set by hand on this
card against leaving them unset: decode 85.95 against 86.65 tok/s (-0.8 %), prompt read 1,405.3 against 1,406.1 (-0.06 %),
same text. So they stay unset, and nothing changed for them to change. Card guide: [docs/NVIDIA_L20.md](../NVIDIA_L20.md).

### Qwen3.8-27B reads prompts 19-29 % faster with its draft head on (11 Nov)

With `--mtp`, every prompt token also went through the whole draft block (its attention over its cache, its output
projection, its feed-forward), eight tokens at a time. Nothing reads that output: the block is one layer, and a
prompt position leaves behind only its K/V rows, which come from the block's input alone. `xolotl-dense` now writes
just those rows, with the same launches in the same order as before, so the cache holds the same bits.
`STRATA_DENSE_MTP_KV_ONLY=0` restores the old path.

| Swift-1.5 27B IQ3_S + MTP 3, k8v4, RTX 4090 (same binary, switch off / on, 2 rounds interleaved) | Before (tok/s) | After (tok/s) |
| --- | ---: | ---: |
| 2K prompt read-in | 1,646 / 1,716 | **1,972 / 2,032 (+19 %)** |
| 30K prompt read-in | 1,067 / 1,106 | **1,394 / 1,402 (+29 %)** |
| decode after the 2K / 30K prompt | 143.7 / 96.0 | 144.3 / 100.0 |

Exactness: four greedy answers through the server (code, prose, a script, a summary of a 24K-character document),
switch off against on: byte-identical text and the same draft cycles (106 / 124 / 42 / 139 cycles at 3.76 / 2.45 /
3.10 / 2.86 tokens each), so the draft head saw the same cache. `xolotl-dense --selftest` passes.

### The fused router ops and batched copies (#87) stay opt-in: flat on a 4090 (11 Nov)

Xiaomi2008's #87 (`-DSTRATA_DSV41_ROUTE_OP=ON -DSTRATA_DSV41_COPIES_OP=ON`) measured +2.8 % on an RTX 5090. On this
RTX 4090 it is inside the noise, so the default build keeps both off. The options stay in the tree for larger cards.

| DeepSeek-V4.1 served Q6 config, 8K prompt, 512 tokens, RTX 4090, 3 rounds interleaved | Off | On |
| --- | ---: | ---: |
| decode, tok/s (three rounds) | 27.88 / 25.82 / 27.62 | 26.84 / 27.65 / 27.31 |
| mean | 27.11 | 27.27 |
| the one pair that produced the same text (same passes, same tokens): decode wall | 18.33 s | 18.48 s |
| same pair: ms a pass | 115.27 | 116.22 |

The rounds differ by up to 7 % on their own because the sampled text differs (155 to 188 passes for 512 tokens); the
same-text pair is the fair one, and it shows no gain.

### Fixes ported 10-11 Nov (correctness, security, tool calls)

No speed claims here; these stop faults or wrong behaviour. Each has its own tests in the pull request.

| Fix | What it stops | PR |
| --- | --- | --- |
| Picture requests cannot reach Pillow's JPEG 2000 decoder (GHSA-78g4-xqwv-q3g8) | a heap overflow in the libopenjp2 bundled with Pillow 11.1.0-12.3.0, reachable from a crafted picture | #125 |
| A failed engine start survives a concurrent close (upstream #4792) | a crash when a start fails while the server shuts the engine | #125 |
| The MMQ row table's over-read margin comes from the tile bound (upstream #5108) | `Xid 31` on long prompts | #126 |
| PCIe staging: flag B is raised behind the copies queued before it (upstream #5096, #4844) | a kernel reading an expert before its upload landed | #126 |
| A NaN router input picks valid experts, says so, and ends the request (upstream #4825) | a fault or a silently shifted answer | #124 |
| An assistant turn sent back without its thinking gets this server's copy back (Strata #5643) | a follow-up turn re-reading the whole previous answer | #124 |
| Route-prior NaN guard (Strata #5655) | a NaN prior spreading into routing | #124 |
| A misspelt opener (`<toolcall>`, `<tool-call>`) before a declared tool's `<function=NAME>` is the call (upstream #5276) | Qwen3.8-Flash-Next's call coming back as answer text, the agent waiting for a result | #128 |
| A streaming client that stops reading gives the engine's turn back (upstream #4271) | one hung client holding every other request in the queue | #129 |
| `moe_iq4_w4a8` compiles for sm_60-75 targets | the sm_70 build failing (#119) | #121 |
| `requirements.txt` lists `jsonschema` and `referencing` | a fresh venv failing on `response_format` (#119) | #123 |

Also new and opt-in: `STRATA_DSPARK_COST=1`, a cost-aware DSpark / lookup window for DeepSeek-V4.1 (#122, asked for
in #120; off by default, not yet measured here).

### Thinking that repeats whole passages can be closed instead of cut off (11 Nov, opt-in)

`"reasoning_loop_recovery": "wrap"` in a serve config: when the thinking repeats the same passages (the existing
detector, checked every 512 output tokens), the server closes the thinking the way the thinking budget closes it and
the model answers. Once a reply; off by default. The two older modes do not cover every model: `"recover"` rewrites
Qwen's effort sentence, so it does nothing on a prompt without one (DeepSeek-V4.1), and `"stop"` ends the reply with no
answer.

Why: one saved DeepSeek-V4.1 trace (a LiveCodeBench hard problem, 52,475 tokens, it passed) repeats the same
six-line check 76 times over 60 % of its thinking; the detector fires on it at about 7,200 tokens.

Checked on the CPU only (`serve.test_reasoning_loop_recovery`: two new cases, 20 pass; `serve.test_server` 360,
`serve.test_dsv41` 70 pass). **Not measured on a model yet**: it changes the reply when it fires, and the detector
also fired on 2 of 86 saved Qwen3.8-27B traces that were step-by-step work, not loops. So it stays opt-in until a
pass-rate A/B.

### `xolotl-dense` runs the gated residual (hyper-connections) (10 Nov)

`xolotl-dense` served Qwen3.8-27B and its fine-tunes - a plain pre-norm residual. The Qwen4 generation, whose 27B
dense model is announced but unreleased, carries the Flash-Next additions on top of that skeleton; the largest of
them is **hyper-connections**: the residual is not one vector but a stack of streams (Flash-Next: 4 of 2560 wide,
rank 320), and every layer norm is replaced by a mixer that reads all of them and writes back per-stream.

This is the missing piece for a Qwen4-27B dense file. Whether that file also carries the QSA indexer or PLE tables
is unknown until it ships - if it does, the engine says so and refuses rather than half-runs it.

The dense engine now runs them, on the same kernels the MoE path already uses (`gr_read`/`gr_write` in
`kernels/gr.hpp` - no new CUDA):

- **Config:** `hyper_connection.count` and `hyper_connection.low_rank` are read from the file; absent keys mean the
  plain residual, so every existing model is untouched.
- **Per layer:** `hc_{attn,ffn}_{down,up,inject,norm}` replace `attn_norm`/`post_attention_norm`; the matrices are
  uploaded in the BF16 engine form the mixer takes, one host rounding from whatever the file stores.
- **Output:** `output_hc_{norm,down,up}` collapse the stack where `output_norm` used to sit.
- **Refused, each with its own message, rather than half-run:** hyper-connections with the MTP block (that block has
  its own `nextn_hc_head` mixer), with drafter taps (a tap is one `n_embd` row; an HC layer holds `hc`), and on the
  MMQ prompt path (the prompt falls back to `run()`, `NC` tokens per call - correct, just not the fast path).
- **Also refused ahead of a load:** a QSA indexer and PLE n-gram embeddings, which the dense engine has no kernels
  for. `xolotl-dense --check <file>` names whichever the file carries instead of failing on a shape much later. An
  architecture string this engine was never validated on is opt-in: `STRATA_DENSE_EXPERIMENTAL_ARCH=1`.

Verification, without a single Qwen4 weight (none exist):

| check | result |
| --- | --- |
| `dense_arch_test` (18 checks: what is refused, the opt-in, the HC geometry) | all pass |
| synthetic HC model (`tools/qwen4_hc_fixture.py`), `--check` | loads; `hyper-connections: 2 streams of 256, rank 8` |
| the same model, 40 tokens one at a time vs chunks of 3,2,5,1,8,4 | max logit difference **0** (bit-identical) |
| a real Qwen3.8-class dense file (VISTA-4B-Q8_0) on the same build | passes: bit-identical singletons vs chunks, MMQ prefill 1.48 % (no "before" number was taken, so this is a pass, not a comparison) |

The bit-identical line is the meaningful one: the multi-column read (`gr_read_multi`) and the single-token read
(`gr_read` + its workspace) agree exactly, which is what an implementer gets wrong first.

**Not yet proven: numerical parity of a whole forward pass.** The kernels are the ones Flash-Next validates against
the reference and the wiring is self-consistent, but nothing here has compared the output against a reference model.
When Qwen4-27B lands, the first task is that comparison, not a benchmark.

### Long thinking runs: a loop no longer loses the answer, a retry no longer re-reads the conversation (9 Nov)

Seen on GLM-5.3-Flash in a coding agent: a reply thought for 18,361 tokens, then wrote one token ('Dot') 256 times;
the server's loop guard (#606) ended it as "length" with no answer after ten minutes. The client's next request
(the cut turn dropped, a new message) then read the whole 26,357-token conversation again (52 s).

- **Server (every model):** a one-token run while the model is still *thinking* is dropped from what the engine
  continues from, and the thinking is closed the way the thinking budget closes it, so the model answers. Once a
  reply; a run in the answer still ends it. `"repeat_wrap_up": false` in the config restores the old ending.
- **GLM engine:** the conversation snapshot now sits before the prompt's last `<|assistant|>` instead of after the
  generation prompt, so a request that re-renders the end of the prompt resumes there (`STRATA_GLM_SNAP_TOKEN=0` =
  the old place). A follow-up reads those few tokens again.

| GLM-5.3-Flash Maya 512K, 7.4K-token prompt | Before | After |
| --- | ---: | ---: |
| the retry shape (last turn dropped, new message) | 7,412 tokens read again, ~17 s | **7,402 reused, 1.2 s** |
| a normal follow-up | reused | reused (7,410 of 7,429) |

### CPU IQ1_M gate/up rows: one decode for every token (9 Nov, for smaller cards)

The same gap as the down rows below, on the gate/up side: IQ1_M experts (Qwen3.8-Flash-Next XQ 45 GB has them in 27 of its
48 layers) ran ggml-cpu's single-token dot, decoding every block again for each token. They now take the AVX-2 rows
DeepSeek-V4.1's CPU lane already used (`iq256x_rows`, ggml's arithmetic lane for lane below `STRATA_IQ_WIDE_MIN` tokens),
under the gate/up rule (no AVX-512, from `STRATA_IQ_MT_MIN` tokens, default arithmetic). `STRATA_NO_IQ1M_ROWS=1` restores
ggml's dot.

| `kq_cpu_bench --dims 2560,640`, IQ1_M gate/up + Q2_0 down, Ryzen 7 5700X, 8 threads | Before (ms an expert) | After (ms an expert) |
| --- | ---: | ---: |
| 1 token | 0.38 | 0.36 |
| 4 tokens | 1.22 | **1.04** |
| 8 tokens | 2.42 | **1.56** |

On a 24 GB RTX 4090 it changes nothing measurable: ~95 % of Flash-Next's decode experts are in the card's cache and
route-resident moves most of the rest there, so the CPU computes very few (same binary, on/off interleaved, two runs
each: 2K prefill 1,430 / 1,435 vs 1,342 / 1,416 tok/s, 8K 3,397 / 3,386 vs 3,430 / 3,380, decode inside the run-to-run
spread). It is for cards with less VRAM (12 GB), where more experts run on the CPU; a 12 GB measurement is to come.

### CPU i-quant down rows: one decode for every token (8 Nov)

The CPU pool's multi-token kernels decoded an i-quant block once for all the tokens routed to an expert, but only for
gate/up: the **down** rows (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) ran ggml-cpu's single-token dot per token, so each
extra token cost almost as much as the first. They now take the same AVX-2 row kernels, under the gate/up rule (a CPU
without AVX-512, from `STRATA_IQ_MT_MIN` tokens, default arithmetic; `STRATA_NO_IQ_DOWN_MT=1` restores ggml's dot).
One token is unchanged, so decode's CPU tier is untouched; a group of 2+ tokens rounds like the gate/up kernels already
did (any model with i-quant down experts on the CPU, e.g. a Flash-Next verify window, gets it on its next build).

| `kq_cpu_bench --dims 4096,2048`, IQ2_XXS gate/up + IQ2_S down, Ryzen 7 5700X, 8 threads | Before (ms an expert) | After (ms an expert) |
| --- | ---: | ---: |
| 1 token | 1.93 | 1.93 |
| 4 tokens | 4.47 | **3.02** |
| 8 tokens | 8.23 | **5.61** |

GLM's short-prompt CPU split can now take a 0.4 share (served Maya configs). Same agent replay as below:

| GLM-5.3-Flash Maya 512K, tool-output turns on a 25K conversation | No split (s) | Split 0.3, old kernel (s) | **Split 0.4, new kernel (s)** |
| --- | ---: | ---: | ---: |
| +246 tokens | 5.47 | 4.79 | **4.63** |
| +300 tokens | 5.56 | 4.97 | **4.82** |
| +590 tokens | 6.21 | 5.45 | **5.36** |
| +642 tokens | 6.30 | 5.89 | **5.49** |
| +1,011 tokens | 6.86 | 6.13 | **6.12** |
| +1,877 tokens | 8.02 | 7.54 | **7.46** |

Exact-math perplexity (every 256-token chunk through the split): chat 5.516 (no split) / 5.305, code 3.624 / 3.617.
`kq_cpu_bench --x / --xgu / --xdown` runs the rows through the EX entry point for comparison; `STRATA_IQ_WIDE_MIN` sets
from how many tokens the EX lane's wide rows start (default 7, unchanged; lower measured slower here).

### GLM-5.3-Flash: the same CPU split for short prompts (8 Nov)

GLM's expert tier (`ds4_moe`) gets the `STRATA_CHUNK_CPU` split too (same knobs). Since the RAM upgrade every Maya expert
sits in the 80 GiB arena, so a small agent turn's floor is all PCIe: a 250-token batch touches nearly all 288 experts in
all 42 layers, and +246 tokens on a 25K conversation took 5.47 s, as long as the 150-token reply after it. The CPU
unpacks GLM's i-quants at ~11 GB/s against PCIe's ~22, so its share is 0.3 here (0.5 overloads it; a 4-token cap was no
better). The pool applies the routed-expert LoRA as it does in decode.

| GLM-5.3-Flash Maya 512K (served flags), 25K-token conversation, tool-output turns | Before (s) | After (s) |
| --- | ---: | ---: |
| +246 tokens | 5.47 | **4.79** |
| +300 tokens | 5.56 | **4.97** |
| +590 tokens | 6.21 | **5.45** |
| +642 tokens | 6.30 | **5.89** |
| +1,011 tokens | 6.86 | **6.13** |
| +1,877 tokens | 8.02 | **7.54** |

Exact-math perplexity (skips off, every 256-token chunk through the split): chat 5.516 -> 5.362, code 3.624 -> 3.600.
The CPU's rows keep more activation precision than the card's q8_1 MMQ path (GLM's chunked ppl moves 1-2 % with the
expert path), so the drift is toward the exact model. Decode is unchanged (28.5-31 tok/s, PCIe-bound).

### DeepSeek-V4.1: the CPU takes the quiet experts of a short prompt (8 Nov)

**`STRATA_CHUNK_CPU=1`** (opt-in; on in the served config). An agent turn's tool output (a few hundred tokens) runs
as a batch, and a batch that small touches about 185 experts a layer for about 5 tokens each. Profiled on a 4090
(PCIe 4.0, measured 15.8 GB/s host to card), a 161-token turn copied 59 GB of experts to the card in 3.7 s while its
kernels took 0.47 s: the link was the floor. Now the CPU pool computes the layer's lowest-count experts that are in
the RAM arena (read once, from RAM, where they already are) while the card runs the rest, and their rows join the
card's before the weighted sum. Routing is unchanged; the CPU's rows come from the decode CPU tier's kernels.
`STRATA_CHUNK_CPU_FRAC` (default 0.5) is the CPU's share of the layer's streamed bytes (0.7 measured no faster),
`STRATA_CHUNK_CPU_MAXTOK` (default 8) keeps busier experts on the card, `STRATA_CHUNK_CPU_MAXN` (default 2048) skips
wide batches.

| DeepSeek-V4.1 `--fast`, 30K-token conversation, tool-output turns | Before (s) | After (s) |
| --- | ---: | ---: |
| +161 tokens | 4.40 | **3.83** |
| +514 tokens | 6.00 | **5.47** |
| +906 tokens | 7.46 | **7.01** |
| +1,866 tokens | 10.32 | **9.80** |
| short34 (34 fresh ~85-token prompts, batch path) | 151 | **129** |

short34 against the API fixtures: NLL 0.4889 -> 0.4924, top-1 87.15 % -> 87.29 %, overlap 0.8794 -> 0.8792 (the base
rerun was bit-identical; VRAM placement alone moves NLL ~0.03). What is left of a short turn's floor is the NVMe
reads of experts that are only on the SSD (~2 s), which can't start before the layer's routing is known.

### DeepSeek-V4.1: an interrupted reply no longer re-reads the conversation (8 Nov)

After a cancelled reply a client re-renders the end of its prompt. The conversation pool then held a boundary at the
end of that prompt and an older one below it, and the resume took the older one: in a real 64K-token agent session
it re-read 10,593 tokens (25 s) when 64,366 of 64,369 were reusable. The resume now takes the further of the two (a
partial resume up to 16 tokens below the newer boundary is exact). `check_conv_reuse.py --scenario interrupt` covers
it (bit-exact on the CPU), and `--gpu` runs the reuse checks on the card in ~2 minutes instead of ~25.

### Qwen3.8-27B: a shortlist head for the MTP drafts (8 Nov)

**`--draft-vocab <file>`** (after NInfer's `--lm-head-draft`): each MTP draft step read the whole 248,320-row output
head (1.1 ms, three times a cycle). Drafts now score a copy of the rows of the 131,072 most frequent tokens (NInfer's
frequency ranking) plus the prompt's own tokens that the list lacks (so a file rewrite keeps its identifiers).
Verification still uses the full head, so the output does not change. Drafting takes 4.0 ms a cycle instead of 6.0,
and costs 591 MiB of VRAM. If that room isn't there, the shortlist is skipped and MTP stays on. Served by
`xol-dense-quetza` and `xol-dense-voice` (27B + LFM3 at 200K: 23.9 of 24.6 GB).

| Qwen3.8-27B, MTP 3 + `--ngram-chain`, greedy, thinking off (mean of 2 runs) | Before (tok/s) | After (tok/s) |
| --- | ---: | ---: |
| Rewrite a 140-line file with one change | 174.7 | **182.8** |
| Write new code | 141.8 | **144.7** |
| Prose | 91.9 | **97.4** |

### Qwen3.8-27B: one-pass verify attention and an n-gram draft chain (8 Nov)

Two ideas from [NInfer](https://github.com/JGamboa/ninfer-4090-windows), the fastest 27B engine we have measured on this
card (170-200 tok/s), ported into `xolotl-dense`:

- **Verify attention in one launch.** The decode attention ran once per column of a verify window, so a 4-token window
  read the KV cache 4 times. It now takes every column in one launch: each K/V cell is loaded once and updates every
  column that sees it. Per column it is the same split-K arithmetic, with explicit fma / mul, so a column gives the same
  bits alone as inside a window (`--selftest`: 0 bits off over an 8-column window crossing a split boundary). With it,
  RMS norm uses 1,024 threads holding the row in registers (it was one 256-thread block per row, 8 us a call).
- **`--ngram-chain`** (in the style of llama.cpp's ngram-mod): a pool maps a hash of the last 8 tokens to the token
  that last followed them, over the whole conversation. After MTP's drafts the proposal continues from the pool, up to
  7 drafts, and a round only uses the wider window when the pool adds at least 3. Verification is unchanged.

| Qwen3.8-27B, MTP 3, greedy, thinking off | Before (tok/s) | After (tok/s) |
| --- | ---: | ---: |
| Synthetic code prompt, 2K context (2 rounds) | 134.0 / 133.1 | **144.0 / 142.3** |
| The same at 30K context (2 rounds) | 89.4 / 89.2 | **98.2 / 98.2** |
| Rewrite a 140-line file with one change, `--ngram-chain` (7.0 tokens a cycle) | 137.3 | **173.0** |
| Write new code, `--ngram-chain` | 127.0 | **140.4** |
| Prose, `--ngram-chain` | 88.9 | 90.4 |

On real agent work (Gauntlet v3-small, one run, as served with all of the above): **score 5.50, 5 of 6 solved, decode
104.6 tok/s** (before: 4.47 twice at 89 tok/s). Fresh-prompt prefill 1,634 / 1,419 tok/s at 6K / 16K.

Greedy answers with and without MTP, and with and without the chain, are word for word the same. The voice setup (27B at
200K next to LFM3) still fits with the chain's 7 rollback snapshots: MTP stays on, 22.8 of 24.5 GB.

**DFlash2 for the 27B** (`--dflash2 <pack>`, z-lab's Qwen3.8-27B-DFlash2 drafter, opt-in) runs and gives the same greedy
text, but no speed-up yet: it drafts 7 and gets 40-53 % accepted (about 3.8 tokens a cycle, MTP 3's figure), while an
8-column check costs more. It stays opt-in until the verify pass gets cheaper.

### Qwen3.8-27B: MTP stays on with thinking off, decode 52 -> 111 tok/s (8 Nov)

`xolotl-dense` switched speculative decoding off whenever a repetition or presence penalty was set. Every thinking-off
request carries Qwen's model-card sampling (presence penalty 1.5), so voice mode and every no-thinking chat decoded one
token at a time. A penalty only changes the sampling, and each verified column is drawn after the tokens before it are in
the penalty window, so speculation now stays on and the text is exactly what single steps give (greedy answer checked
word for word). Fresh 2K / 6K / 16K prompts, thinking off: 52.3 / 52.0 / 47.8 -> 105.3 / 110.7 / 102.7 tok/s.
Thinking-on requests (the Gauntlet, 89 tok/s) were never affected.

### Gauntlet v3 is the main table, with each model's own API beside it (8 Nov)

The README now leads with the [Gauntlet](../../tools/gauntlet/README.md) v3-small (6 hard agentic coding tasks graded by
hidden tests, 10 turns, thinking on, each model as served) instead of the HumanEval / tool-call / agentic screen, on
which every model scored 19-21/21, 10/10, 3/3. The same tasks ran on each model's full-precision API through OpenRouter
(reasoning on): DeepSeek-V4.1-Flash-1017 4.49 against our V4's 4.99, Qwen3.8-Flash 3.90 against our 45 GB build's 4.49,
Qwen3.8-27B 4.47 against 4.47, GLM-5.3-Flash 5.87 (our local GLM run is still to come). Fresh-prompt prefill on
GLM-5.3-Flash re-measured: 312 tok/s at 6K, 450 at 16K. The old table is in [BENCHMARKS.md](BENCHMARKS.md).

### The dense engine is `xolotl-dense` now (8 Nov)

The Qwen3.8-27B dense engine builds as `build-xolotl/xolotl-dense` (log lines `xolotl-dense: ...`). The port it
grew from is Mirko Perrone's ([perronemirko/Strata](https://github.com/perronemirko/Strata)); the Xolotl tree unified
and extended it. `strata-dense` stays next to it (a symlink, a copy on Windows), so existing configs, wrappers and
scripts keep working unchanged. Internals keep their names (`STRATA_DENSE`, `dense_model.cpp`, the `STRATA_*`
switches). The same weights give the same output on both names.

### DeepSeek-V4.1 fast mode now uses Q2_K: under a point of quality for +27 % decode (7 Nov)

The fast-mode sidecar is now built as **Q2_K** (llama.cpp's 2-bit k-quant, a scale and a minimum per 16 weights)
instead of Q2_0, weighted by an importance matrix recorded while serving (`XOLOTL_IMATRIX_OUT`, then
`XOLOTL_SHADOW_IMATRIX` at build time). Weight error 0.35 -> 0.27. short34 top-1 **86.16 %** (normal 87.01 %, the Q2_0
version 84.46 %), NLL 0.540 (0.512 / 0.566); decode **33.0 vs 26.1 tok/s (+27 %)** on the 4090, ~320 more experts in
VRAM. The sidecar is ~15 % bigger (~40 GB). `--build-shadow` defaults to Q2_K now (`XOLOTL_SHADOW_TYPE=42` = the old
Q2_0). Also measured and dropped: a shadow of only the MXFP4 experts (85.73 %, +24 %), the 1,000 best-ranked experts
kept exact (85.40 %, +15 %), an imatrix-weighted Q2_0 (84.04 %, +42 %): the 2-bit format, not the encoder, was the
limit. Agentic coding (Gauntlet v3-small, 6 tasks, 2 runs each, served settings): **8/12 solved in both modes, score
4.99 vs 4.99, decode 41.1 vs 32.6 tok/s (+26 %)** during the tasks. Details: [docs/XOLOTL_FAST.md](../XOLOTL_FAST.md).

### DeepSeek-V4.1 fast mode: a 2-bit shadow of the biggest experts (7 Nov, opt-in, lossy)

`ds41_generate --build-shadow F` writes a Q2_0 copy of every MXFP4 and Q3_K expert weight (3,543 experts, 54.7 ->
35.1 GB; about half of the slow-lane bytes) to a sidecar file; `XOLOTL_SHADOW=$(cat F.list)` serves from it. The model file is only read,
and the engine falls back to the normal experts if the sidecar does not match the model. On the 4090 (8K prompt,
512 tokens): **83.8-90.4 vs 111.6-117.2 ms per decode pass, 34.1-36.7 vs 25.7-26.5 tok/s (+29 to +43 %, two runs each)**; ~450 more experts fit in VRAM and
file-tier waits halve. Quality: short34 top-1 **84.46 % vs 87.01 %**, NLL 0.566 vs 0.512 - a real cost, hence
opt-in. ggml's reference Q2_0 quantizer sets the scale to the block's max (relative error 0.72); the shadow builder
searches each block's scale and sign (0.35). Next: a Q3_K-only shadow (the hot MXFP4 experts stay exact) and a
calibrated encoder. Details: [docs/XOLOTL_FAST.md](../XOLOTL_FAST.md).

### Xolotl Direct: NVMe straight into VRAM (7 Nov, experimental)

When DeepSeek-V4.1 decodes, the experts that are in neither VRAM nor the RAM arena are read from the NVMe drive. Until
now those bytes went NVMe -> RAM bounce -> pinned staging -> PCIe -> VRAM: four trips over the RAM bus, which is the
wall on a DDR4 box. Xolotl Direct has the drive write them straight into the card's memory, on the normal NVIDIA
driver and a GeForce card (no driver patch, no datacenter card). It is opt-in at build time (`-DXOLOTL_DIRECT=ON`) and
at run time (`XOLOTL_P2P=1`), Linux only, and needs Resizable BAR plus two small GPL kernel modules
(`kernel/xolotl-direct/`). Setup: [docs/XOLOTL_DIRECT.md](../XOLOTL_DIRECT.md).

- One 18 MiB expert NVMe -> VRAM: **3.1 ms** against ~11 ms on the normal path; 64 MiB at **6.31 GB/s** (the drive's
  limit) against 4.61; no system-RAM traffic. Every direct read checked against a normal read: 0 differences.
- Decode, V4, 4090, 8K prompt / 512 tokens, two rounds with identical VRAM: normal 116.5 ms/pass 25.36 tok/s |
  direct 116.0 / 25.48 | direct + file lookahead (`XOLOTL_LOOKAHEAD=2 XOLOTL_LA_SPEC=2`) **113.0 / 26.14 (+3.1 %)**.
  On its own the direct path saves ~1.7 ms of file waiting per pass but barely moves the pass (those reads already
  overlap other work); the gain comes from the VRAM stash it makes possible, filled a layer early.
- The same output: P2P on and off decode the same text.
- Tested on one machine so far (RTX 4090, Linux 7.0, driver 595.91, Samsung PM9A1). Reports from other cards and
  drives welcome.

### DeepSeek-V4.1 loads in 44 s, and XOLOTL_ names for every switch (7 Nov)

Loading DeepSeek-V4.1 V4 with an 80 GiB RAM arena took 96 s on the 4090 box. Two things cost most of it, and both are
fixed by default now:

- **The dense weights are converted once, not on every load.** With a Q6_K dense half, the conversion took ~25 s a
  load on the CPU. Xolotl now writes the result to `xolotl-cache/` next to the model (~5.5 GB for V4 at Q6_K) and
  reads it back on later loads. The cache is keyed on the model file's size and date and the dense type, so a new
  model or setting writes a new one. It is only written when the drive has its size + 2 GiB free.
  `XOLOTL_DENSE_REQUANT_CACHE=<dir>` puts it somewhere else, `XOLOTL_DENSE_REQUANT_CACHE=0` turns it off.
- **The RAM arena is pinned while it fills.** Pinning an untouched 80 GiB arena makes the kernel fault and zero every
  page on one thread. Filling first and pinning in four pieces alongside the fill spreads that work over the readers.
  `XOLOTL_ARENA_PIN_AFTER=0` goes back to the old order, `XOLOTL_ARENA_PIPE=<n>` sets the pieces (default 4, 0 = one
  registration after the fill).

Measured on the V4 serve config (4090, 80 GiB arena, second load): **96 s -> 44 s**. The output is unchanged; every
decode run on 7 Nov used these settings.

**XOLOTL_ names.** Every engine switch documented as `STRATA_<X>` can now also be set as `XOLOTL_<X>`. The engine and
its launch scripts copy each `XOLOTL_` variable onto the `STRATA_` one at start, and the Xolotl name wins if both are
set. Nothing that uses the `STRATA_` names changes.

### RTX 40 (sm_89) speed switches on by default (6 Nov)

The Qwen engine has had a per-card table of speed switches for RTX 30 cards (sm_86) since 0.2.0: switches that give the
same tokens bit for bit and were faster on that card. RTX 40 cards (sm_89) had none, so they ran the plain defaults.
Measured on the 4090 tonight, the sm_86 set as one stack makes decode faster here too, so sm_89 now gets the same
table: `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_SIDE_MIXER=2`,
`STRATA_ATTN_DAILY=1`, `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`, `STRATA_QSA_ROWS=1`, `STRATA_B6_MMVQ_ROWS=1`.
`STRATA_SM89_PERF=0` turns it off, and a switch you set yourself is kept.

- QFN 45 GB, 16,383-token prompt + 1,024 tokens, every arm emitting the same text: decode **138.96 -> 150.21 tok/s
  (+8.1 %)**, 3 interleaved runs (149-151 against 138-140), base against base -0.2 %; time per verify window -6.4 %;
  prefill unchanged. Recheck on the public build, defaults against `STRATA_SM89_PERF=0`, 2 runs: 149.5 against 138.9.
- Output the same in every arm. Measured as a stack on one card (no drop-one per switch on sm_89 yet).
- Also screened tonight and left out (inside the run noise on the 4090): the prompt-side GDN and dedupe switches, the
  adapt overlap, fused waits, PLE / verify dedupe, Q/L fusion, the down-projection group cap, embeddings in VRAM, the
  cost-aware MTP window, and the 31 Oct prompt picks (`STRATA_PF_GROUP_SCRATCH`, `STRATA_PREFILL_ROUTED_STREAM`).

### DeepSeek-V4.1: Engram read-ahead and the late shared expert (6 Nov)

Two changes to the DeepSeek-V4.1 engine (`tools/v41/engine/ds41_dense.cpp`), both on by default and both computing
exactly what the old path did. The ideas come from Basalt (jesdga95: 9e95f399f, 6220768cf), re-derived for this engine.

- **Engram read-ahead** (`STRATA_DS41_ENGRAM_AHEAD`, `=0` off). A long prompt goes through the model in chunks, and each
  chunk needs its rows from the Engram tables on disk. Before, the next chunk's reads only overlapped the current chunk's
  first layer. Now they start as soon as the current chunk reaches its first Engram layer, so the disk works beside
  almost the whole chunk. Prompts of one chunk (up to ~6-9K tokens on a 24 GB card) are unchanged.
- **Late shared expert** (`STRATA_DS41_SHEXP_LATE`, `=0` off). Each decode step's router and shared expert used to be
  one GPU job, and the host waited for both before it could start loading the routed experts. Now the router's answer
  comes back first, the host starts the expert reads, and the card computes the shared expert at the same time.
- **Measured** on the RTX 4090 box (110 GB RAM), DeepSeek-V4.1-Flash-1017 V4 XQ with the dense half at Q8_0, through the
  server, both on vs both off, 3 interleaved rounds:

  | Test | Off | On | Change |
  | --- | ---: | ---: | ---: |
  | Prefill, 26,622-token prompt | 469 / 448 / 449 tok/s | 507 / 507 / 504 tok/s | **+11.1 %** |
  | Decode, 8K prompt, 512 tokens | 25.51 / 26.02 / 25.70 tok/s | 26.10 / 26.61 / 26.63 tok/s | **+2.7 %** |
  | Prefill, 6,382-token prompt (one chunk) | 318 / 317 / 320 tok/s | 313 / 320 / 317 tok/s | same |

- **Quality: identical.** Scored against DeepSeek's own API outputs (34 cases): NLL 0.5152, top-1 86.11 % with both
  on and with both off, and the logits are bit-identical at the same VRAM seed. The first 8K decode round gave the same
  text, the same expert counts and the same hash on both.
- Tried and dropped: a speculative-decode window of 3 instead of 4 (Basalt 1a01349f1) was 5 % slower here, and a 25 % /
  40 % soft-prune list on top of the route bias cost 3-5 % NLL with no decode gain.

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

### Inkling-2-Mini XQ Unbound 13.9 GB (22 Nov; first release 21 Nov)

On [Hugging Face](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored): Thinking Machines' 26B looped MoE (3B active), our XQ quant with nokeep's
convolution-aware refusal edit ("Unbound"), plus the model's own DSpark drafter and vision projector as optional
files. On the 4090 it does **113 tok/s decode (136-140 with the drafter), 4,190 tok/s prefill at 32K, and 512K
context** (55 tok/s decode at 477K tokens). Held-out KL to the official model 0.028 (top-1 92.8 %), HumanEval 21/21,
Gauntlet v3 4/7. Runs on its own engine, `xolotl-inkling`: [docs/INKLING.md](../INKLING.md).
[Release notes](#inkling-2-mini-xq-re-released-as-unbound-22-nov).

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
  [docs/MULTI_GPU.md](../MULTI_GPU.md), [BUSTER.md](../BUSTER.md). Under a split the engine serves one request at a
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
