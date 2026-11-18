# Field ports, 19 Nov 2026

What came in from other forks and engines on branch `field-ports-20261119`, what each switch does, and the A/B runs
that decide each one.  The scan behind it: `private/notes/ENGINE_SCAN_2026-11-19.html`.  Upstream Strata was **not**
merged (Mal: later versions are reported to have slowdowns); only parts that stand alone were looked at.

Everything below is **off by default**.  A switch becomes a default only after an A/B on this box (RTX 4090, Ryzen 7
5700X, 110 GB) shows a win outside the noise and the same tokens (Xolotl's rule).

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
