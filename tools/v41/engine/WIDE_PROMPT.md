# Wide prompt batches (`STRATA_WIDE_PROMPT=1`)

A long prompt runs layer-major, `--prompt-batch` tokens at a time: each batch
streams every routed expert of layers 0-19 once, and the batch that holds the
enc-skip replay window (the last 2560 tokens) streams layers 20-39 as well. On
this PC those sweeps are drive-bound (about 4.4 s per 20 layers with the RAM
arena at 30 GiB), so a long prompt's time is mostly its number of batches.

The batch width was set by VRAM. A batch column cost about 0.72 MiB: the dense
half's state columns 0.32 MiB, and the tier's chunk buffers 0.40 MiB (the MMVQ
scratch 181 KB, the parts 120 KB, the input and the summed rows 20 KB each, the
q8_1 copy 6 KB, plus 41 KB of MMQ scratch past 16K rows). An RTX 5090 therefore
held 12-20K columns. Past about 16K tokens, ds41_generate also split the batch
into two tier calls (`STRATA_CHUNK_BUDGET_GB`), and each call swept the layer
again.

`STRATA_WIDE_PROMPT=1` (opt-in) brings a column down to about 0.36 MiB (state
0.24, parts 0.12):

- The dense half keeps the FFN's hc state (`hap`) in the `x_state` columns it
  replaces. Every reader of a column's `x_state` (the Engram add, the hc mixes,
  `hc_pre`, `hc_post`) feeds `h`, so the write of `h` comes after them in the
  graph order. The finish reads it and writes the new state over it. Expert
  prediction reads `x_state` before the attention, so it is unchanged.
- The MoE input stays on the card: the tier reads the dense half's `o_fn` rows
  (`Ds41MoeTier::run_chunk_xdev`). There is no host copy and no `c_x`.
- The tier allocates the MMVQ scratch and the q8_1 rows only for a call that
  has an MMVQ launch. That means VRAM-resident experts (a serve turn with the
  cache kept) or a layer MMQ does not cover. The MMQ path never needs them.
- The weighted sums go straight into the dense half's `routed` rows (no
  `c_out`, no pinned landing pad).
- The MMQ scratch is sized by the call's largest expert run, not its token
  count.

In ds41_generate, every batch is one tier call per layer. A batch that is not
the last ends on a sub-batch boundary: otherwise its last columns fall past the
enc-skip cut, which rounds down to one, and run layers 20-39, which is another
sweep. It also ends before the replay window, so only the last batch runs
layers 20-39. With the cache released, `--serve` sizes the width with 0.44 /
0.38 MiB a column instead of 0.95 / 0.78 (and 4.3 GiB fixed in the graph term).
A batch with the cache kept keeps the old constants.

The same kernels run on the same rows, so every token's sums keep their bits.
The switch needs one GPU stage with a GPU tier, no remote helper card and no
`STRATA_ROUTED_HOST`. Otherwise ds41_generate refuses to start: the dense half
no longer copies the MoE input to the host.

## Measured

RTX 5090, the UD-IQ3_XXS model with the production settings plus
`STRATA_INDEXER_NVIS=1`, `GGML_CUDA_MXFP4_WMMA=1` and `STRATA_KV_GATHER=1`,
one-shot runs, last-row logits dumped:

| prompt | off, `--prompt-batch 12288` | on | batches | logits |
|---|---|---|---|---|
| 13,000 tokens | 20.75 s (file wait 14.4 s) | 13.61 s at 13312 (file wait 8.6 s) | 2 -> 1 | identical |
| 60,001 tokens | 48.28 s (file wait 26.2 s) | 32.14 s at 30720 (file wait 14.0 s) | 5 -> 2 | identical |

The VRAM trace at 30720 columns: state 7.30 GiB, the tier's buffers 7.09 GiB
(parts 3.52, the two stage halves 2.0, MMQ 0.65), and the 1.5 GiB lazy prestage
region in the batch that runs layers 20-39. 4.6-5.6 GiB was still free at the
end (two runs).

When the width is capped below the prompt length, the replay-window rule
applies. With the switch on, `--prompt-batch 12288` and the 13,000-token
prompt, the batches are 10240 + 2760 instead of 12288 + 712. That takes
17.38 s instead of 20.75 s (3 sweeps of 20 layers instead of 4), and the
logits are identical.

The serve probe used `--serve` with the production config and the rb8g
candidate env (`STRATA_SERVE_STEP_MAX=16` plus the three switches above). It
sends a first prompt, then five tool-result turns:

| first prompt | rb8g, `--prompt-batch 12288` | rb9 with the switch, `--prompt-batch 32768` |
|---|---|---|
| 58,233 tokens | 52.1 s (47.5 s in an earlier run) | 31.5 s |
| 10,251 tokens | 13.1 s | 12.5 s |

- The first request's logits were byte-identical in both pairs.
- The tool turns (34-1421 new tokens, with the cache kept, so on the lazy
  MMVQ path) took 21.7 s against 21.1 s on the 58K probe, and 20.8 s against
  20.4 s on the 10K probe.
- From the second turn on, the replies differ, as they do between two runs
  of one engine: the VRAM seed holds 3 more experts.

Without the switch, a wider batch was slower: at 20480 the tier split each
layer into two calls (52.2 s), and at 20345 the batches did not end on a
sub-batch, so every batch ran layers 20-39 for its last 377 columns (62.8 s).
The logits stayed identical at 20480 too: the MoE sums do not depend on which
tokens share a batch.

Decode: the forced 1093-token routes are identical with and without the switch
(f30f0845) when the VRAM slot count is the same (`--slots 1451`). With the
automatic seed, the switch leaves 40 MB more VRAM after a short prompt (the
512 state columns without `hap`), so the seed holds 3 more experts. The routes
then differ, as they do for any change of placement.
