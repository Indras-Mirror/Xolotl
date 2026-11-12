# Xolotl changelog

What landed in Xolotl, newest first. The README keeps a one-line summary of each item and links here.

- [November 2026](#november-2026)
- [Model releases](#model-releases)
- [October 2026](#october-2026)

## November 2026

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
