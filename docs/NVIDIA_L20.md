# NVIDIA L20 (Ada, sm_89)

This page is for people running Xolotl on an L20: 92 SMs, 47.4 GiB of GDDR6 at 864 GB/s, compute capability 8.9. An L20 is Ada,
so it ran the tables and default switches that were measured on RTX 40 cards. Those cards are 46 and 48 SMs; this card is 92,
and it is a datacenter part with a different memory system. Every number below was measured on the machine in the history
section, and none of it is a guarantee for another L20 setup.

> **On this page:** [Quick start](#quick-start) · [What runs on an L20](#what-runs-on-an-l20) · [Opt-in switches](#opt-in-switches) ·
> [Checking the kernels](#checking-and-benchmarking-the-l20-kernels) · [Limits](#limits) · [History](#history--measurements)

## Quick start

The L20 needs no special build flags; it is the architecture the CUDA engine already builds for.

```sh
cmake -S . -B build-xolotl -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-xolotl --target xolotl-dense -j
```

Nothing has to be exported. An L20 answers `891` where other Ada cards answer `89` and picks up its own row tables by itself.

## What runs on an L20

The interleaved matvec kernels (`native_mmvq_il`) are picked by a table: per (weight type, columns, output width) the rows a
warp splits a row into. The values are speed only - the kernel's output is bitwise the plain `native_mmvq` one.

| Table | What an L20 gets |
| --- | --- |
| 2-4 columns (`kIlRows_891`) | the cells where three runs on one L20 picked the same rows count twice: 15 cells kept a split, 16 agreed to stay plain, 41 did not agree twice and keep the plain kernel |
| One column (`kIl1Rows_891`) | what a decode of one token at a time asks. Only Q3_K's two widest classes cleared the 3 % rule here; everything else, Q4_K included, stays the plain single-column kernel |
| 5-8 columns | nothing. Those cells stay off behind `STRATA_MMVQ_IL_WIDE` until a card measures them |

30 of the 75 multi-column cells differ from `kIlRows_89`, which is what an L20 used until now: 22 splits the 4070/4080 table
has and these runs did not confirm, 4 splits it lacks, 4 where the count itself differs (a `1` becomes a `2` or a `4`).

## Opt-in switches

| Switch | What it does |
| --- | --- |
| `STRATA_MMVQ_IL_891=0` | Puts the card back on `89`: the 4070/4080 tables, which is what an L20 ran before its own. `STRATA_DEFAULTS=026` does the same to everything since 0.2.6; `STRATA_MMVQ_IL_891=1` wins over both |
| `STRATA_MMVQ_IL=0` | Turns the interleaved kernels off everywhere and runs `native_mmvq`'s own multi-column kernels |
| `STRATA_SM89_PERF` | The nine exact speed switches of the sm_86/sm_89 table. `strata` (the MoE engine) applies them at start; `xolotl-dense` does not, so on the dense engine they are off unless you set them yourself. Both are in the history section: on the dense engine they are not a loss here, and on the MoE engine, which does apply them, decode measured +3.6-7.1 % with them on |

## Checking and benchmarking the L20 kernels

```sh
cmake --build build-xolotl --target mmvq_il_parity -j
CUDA_VISIBLE_DEVICES=1 ./build-xolotl/mmvq_il_parity                        # every case, T 1-8, rows 1/2/4 and the table, bitwise
CUDA_VISIBLE_DEVICES=1 ./build-xolotl/mmvq_il_parity --bench --emit-table   # the same cases timed, printed as tables
```

The table checks run on the host, so they pass against the CPU reference too (`CUDA_VISIBLE_DEVICES=`).

## Limits

- End to end the tables are invisible on the two dense packs we ran: a decode reads one token at a time (one column), and a
  prompt is read through llama.cpp's MMQ kernels, so a 2-4 column cell is only reached by a verify window. That is what the
  2026-11-11 numbers below are measuring, and why they are flat.
- On the MoE pack of the last section the tables touch nothing at all. Its dense half is Q8_0/F32/BF16, its routed experts go
  through the expert kernels, and the one tensor that is on a weight format the tables know - the draft head, Q5_K at 248,320
  rows - sits in the widest class, which both tables answer 0. `891` and `89` choose the same kernels there, so turning one of
  them off moves nothing but the noise.
- A third pack, DeepSeek-V4.1-Flash, never looks a table up at all. The interleaved kernels have one call site,
  `src/core/verify.cpp:1507`, and that is included by `generate.cpp` and `q36_engine.cu` - not by the DS41 engine, whose dense
  half links ggml. So on that model `891` and `89` are not two answers to a lookup; there is no lookup, and the
  `STRATA_DENSE_REQUANT=q6_K` half is read by llama.cpp's own GEMV. What is card-aware there is a different rule:
  `ds41_apply_mmq_cc_policy` drops chunk-MMQ only below cc 7.5, so an L20 keeps it.
- 41 of 72 multi-column cells did not pick the same rows count in two of three runs. The card is not unstable, the cold-run
  timing of a single GEMV is; a cell is kept only where the runs agree, so a real gain can be dropped here and found later
  with more rounds.

## History / measurements

**Rig.** one NVIDIA L20 (92 SMs, 47.4 GiB), Xeon Gold 6530, driver 595.99.02, CUDA 13.2, revision `d878f642`. Every arm ran on
`CUDA_VISIBLE_DEVICES=1` with the same binary, three rounds, a 4,000-token prompt and 256 generated tokens, greedy.

**The new tables, on vs off** (`STRATA_MMVQ_IL_891=0` = the 4070/4080 tables). The generated text was identical in every arm:
the same 256 `T` lines, the same sha256, in all four runs.

| Pack | Prompt read, tok/s (on / off) | Decode, tok/s (on / off) |
| --- | ---: | ---: |
| Qwen3.8-27B Q4_K_M, no draft head (one token at a time) | 1,483.7 / 1,487.9 (-0.3 %) | 42.27 / 42.29 (-0.05 %) |
| Qwen3.8-27B Q4LynnStyle + MTP `--draft-max 2` (verify window of 3) | 1,403.2 / 1,406.4 (-0.2 %) | 86.61 / 86.64 (-0.03 %) |

**The sm_89 switch stack on the dense engine.** The nine exact switches of `nv_default_env(8, 9)`, set by hand against leaving
them unset on the MTP pack above: decode 85.95 against 86.65 tok/s (-0.8 %), prompt read 1,405.3 against 1,406.1 tok/s
(-0.06 %), the greedy text identical. So the stack the 4090 gave +8.1 % on is not taken here, and `xolotl-dense` does not
apply it at start anyway.

**The same stack on the engine that does apply it** - `strata` serves it at start, so on a MoE card these nine switches are on
unless someone turns them off. Qwen3.8-Flash-Next Swift-1.5 XQ 45 GB, 32K context, one ~13.9K-token prompt and 1,024
generated tokens, greedy, six rounds a side in both start orders. A MoE's decode is bound by the RAM bus, so this was run
after the other three cards had sat at 0 % for half an hour; the probe writes the neighbours' util into every round and all
24 rounds read 0.

| arm | started | prompt read, tok/s | decode, tok/s |
| --- | --- | ---: | ---: |
| default (stack on) | first | 3,587.2 | 228.45 |
| `STRATA_SM89_PERF=0` | second | 3,568.1 | 213.35 |
| `STRATA_SM89_PERF=0` | first | 3,578.6 | 219.75 |
| default (stack on) | second | 3,592.2 | 227.70 |

Medians to medians that is +7.1 % and +3.6 %. Round against round - every arm saw the same six texts, and each arm's six
hashes matched the other arms' - the stack was ahead in 10 of the 12 pairings: +5.9 % to +8.0 % where it ran first, -1.9 %
to +6.5 % where it ran second. The two negatives are that pair's first two rounds and nothing in the record says why. The
prompt read stayed inside 0.6 % either way, the same answer the dense packs gave.

**The third pack, which has no reading either way** - DeepSeek-V4.1-Flash `V41-XQ-1017-v4` (40 layers, 163 GB, ctx 65536,
arena 18 GiB, DSpark depth 4), four arms of six rounds on card 1, one ~13.9K-token prompt and 512 generated tokens, greedy,
the other three cards at 0 %.

| arm | started | prompt read, tok/s | decode, tok/s |
| --- | --- | ---: | ---: |
| default (tables on) | first | 464.6 | 31.65 |
| `STRATA_MMVQ_IL_891=0` | second | 464.1 | 31.90 |
| `STRATA_MMVQ_IL_891=0` | first | 464.2 | 32.00 |
| default (tables on) | second | 464.2 | 32.35 |

The prompt read spread 0.1 % across all four arms, and the decode sign follows which arm started (-0.8 % with the stack on
first, +1.1 % with it on second), not which way the switch sat. It cannot be pinned down by text either: the arm that starts
first reads a cold box and answers differently, so on both runs the first arm's text parted from the other three's at rounds
2, 3 and 4, and all four agreed at round 1 alone - its routing is lossy on purpose (route bias 0.5 with a 0.3 file penalty,
a conversation-adaptive arena), so a run carries the runs before it. The gate every section above rests on, the same sha in
every arm, has nothing to hold on here. That is what the Limits line is the whole of: this engine does not look one up.
