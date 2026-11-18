# Decode opt-ins from the DeepInfer review (rb10, rb11)

A decode token runs 40 layers, and each layer makes a round trip:

1. The attention graph runs on the card.
2. The routing is read back.
3. The tier (host) places the six experts and launches their kernels.
4. The finish graph runs after the routed sum.

On the GSQ IQ2_XS test configuration (route bias 0.25 / 0.15), 92 % of the routed experts are VRAM hits, so the card
and these round trips matter more than they do on the UD model, where the drives dominate.

## Where a GSQ decode token goes

Forced 1093-token trajectory, rb9 + the three prompt switches, `--slots 2378`, ~27 ms a token (37 tok/s).

`STRATA_DEC_PROF` stages per token:

| stage | ms |
|---|---|
| begin | 0.75 |
| predict | 0.28 |
| attn_router | 7.15 |
| tier | 15.3 (of which file reads ~8, the sum wait ~4) |
| finish | 2.4 |

Nsight Systems shows:

- GPU busy ~17.8 of 32.4 ms a token (CUPTI inflates host time).
- The main attention graph: 46 kernels, 133 us a layer, 58 us of them in mul_mat_vec_q.
- The router's post-processing alone is 10 kernels, ~9 us a layer.
- 18 of 40 layers a token have all six experts VRAM-resident; such runs are mostly 1-3 layers long.

## The switches (all opt-in, the same bits)

### `STRATA_ROUTE_FUSE=1`

`ggml_dsv41_route` (vendored ggml, CUDA only) replaces two kernel chains with one kernel each:

- the router's softplus, sqrt, two bias adds, top-6, get_rows, sum_rows, scale_bias, div and scale;
- each routing prediction's softplus, sqrt, two adds and top-16.

It is DeepInfer's mega_gate idea, minus its gate GEMM; the F32 gate matvec stays as it was.

The kernel reproduces the stock chain's values bit for bit:

- Each column is scored by its own thread, with `op_softplus` / `op_sqrt` written the same way.
- The selection keys are those of `k_argsort_topk_warp_f32_i32`: value descending, equal values by ascending column.
- Warp 0 takes the first k with `__reduce_min_sync`, which gives the same min as the shuffle tree.
- The weights are summed as `sum_rows` does (`warp_reduce_sum` with weight j in lane j).
- They are then divided and scaled as `ggml_div` / `scale_f32` do.

It is off under `GGML_CUDA_RANK_ARGSORT=0`, because the bitonic kernel orders equal scores by its wiring. It falls
back to the stock chain for vision rows and the smart / margin route bias, and when the prediction graph also
predicts further layers.

Effect: the attention graph is 133 -> 125 us and the finish + prediction graph 19 -> 15.5 us a layer.

### `STRATA_X_DEV=1`

`Ds41MoeTier::run_to_dev` takes the MoE input's device copy (the dense half's `o_fn`) as well as the routing
readback's host copy:

- The card quantizes the input where the attention graph left it.
- The pool takes the host copy.

This removes the per-layer memcpy and host-to-device copy of the activation. It needs one GPU stage, so that `o_fn`
is on the tier's card.

## Two waits that cost more on this PC (rb11)

Windows 11, WDDM, driver 621.01, RTX 5090. nsys on the GSQ decode showed two delays that a GPU profile on Linux
would not suggest:

- **Small copies queue behind DMAs.** The copy engine runs copies in submission order. The prefetch stream's expert
  DMAs are about 10 MB each, about 0.2 ms each. The tier's per-layer launch metadata (about 160 B, `cudaMemcpyAsync`
  from pinned memory) waited behind them: p50 44 us, p90 189 us from the call to the copy, about 1.9 ms a token.
- **A wait on another stream's event is released late.** A `cudaStreamWaitEvent` was released 0.1-0.3 ms after its
  event fired. A host wait (`cudaEventSynchronize`) on the same event returns in about 0.01 ms.

### `STRATA_META_PULL=1`

A small kernel on the tier's stream copies each launch's metadata (and the sum weights when `STRATA_SUM_ZC` is off)
from the pinned block, which is mapped into the card's address space. It does not use the copy engine, so it does not
wait for the DMAs. The same bytes in the same stream order give the same bits.

### `STRATA_PF_HOSTWAIT=1`

A layer that computes a prefetched expert must wait for the prefetch DMAs. With this switch the host waits for their
event before it launches the hits, instead of the tier's stream waiting for it.

### `STRATA_HIT_FAST=1` / `=2`

DeepInfer keeps the expert map readable by the card, so a layer whose experts are all resident needs no trip through
the CPU. This switch needs `STRATA_ATTN_AHEAD`: the attention graph is queued before the host has the routing.

`Ds41MoeTier::spec_hits` queues four things on the dense stream, behind the attention graph and its routing
readbacks:

1. **The dispatch.** A one-thread kernel (`decode_hit_dispatch`) reads the routing on the card. It looks each expert
   up in a table of VRAM slot addresses: the cache's residency for that layer, rebuilt by the host before each launch.
   The table is pinned host memory read over the bus, so no copy waits behind the DMAs. If all six experts are
   resident, it writes their addresses and a group count of 6. Otherwise it writes a count of 0.
2. **The quantize.**
3. **The grouped expert launch.**
4. **The weighted sum** (`decode_rows_sum_gated`), which runs only when the count is not 0.

Steps 2-4 are the kernels, parameters and values that `gpu_run` uses for an all-hit layer.

When the routing comes back, `gpu_run` keeps the launch if all six routed experts are resident in the slots the
table had. It then skips the activation, the hits' launch and the sum sync. The finish, queued behind on the same
stream, reads the sum in order.

Every other layer runs as before. On those layers the launch does nothing, except that its quantize writes to
`d_xq`, and the layer writes the same bytes there.

`=2` queues the launch only when the prefetch's prediction for the layer has its top 6 all resident.

On the 1093-token run, the launch was kept on 19048 of 42321 layers (45 %; `=2`: 11799 of 13747).

The launch has to share the dense stream because of the second delay above. Because of that, the shared expert (the
tail graph, about 36 us) now runs after the hits instead of beside them. That takes back most of what the fast path
saves.

## Measured

GSQ, forced 1093, `--slots 2378`, interleaved pairs (tok/s):

| arms | pair 1 | pair 2 | pair 3 | mean |
|---|---|---|---|---|
| off / ROUTE_FUSE | 36.53 / 37.14 | 36.10 / 36.81 | 35.55 / 36.63 | 36.06 / 36.86 (+2.2 %) |
| ROUTE_FUSE / + X_DEV | 37.00 / 37.46 | 36.95 / 37.38 | 36.71 / 37.00 | 36.89 / 37.28 (+1.1 %) |

The production UD-IQ3_XXS model (route bias 0, forced 1093, `--slots 1451`), off / both switches: 16.76 / 17.20 and
17.06 / 17.18 tok/s (16.91 / 17.19, +1.7 %), routes f30f0845 in both arms, as rb8g / rb9.

The decode routes and logits (`STRATA_DEC_TOPK`) were byte-identical in every arm and to rb9 (50c29791). The 13K
prompt's last-row logits with `STRATA_ROUTE_FUSE` were also byte-identical (72207939); the batch path runs the same
op.

### rb11

GSQ, the same run. Every arm has the two rb10 switches on. Arms were run interleaved, in tok/s.

| arms | pair 1 | pair 2 | pair 3 | mean |
|---|---|---|---|---|
| off | 34.13 | 34.24 | 34.35 | 34.24 |
| `STRATA_META_PULL` | 34.96 | 34.79 | 34.92 | 34.89 (+1.9 %) |
| + `STRATA_PF_HOSTWAIT` | 34.97 | 35.00 | 34.88 | 34.95 (+2.1 %) |
| + `STRATA_HIT_FAST=1` | 35.14 | 35.09 | 35.07 | 35.10 (+2.5 %) |

A second set, 4 pairs, all on top of `STRATA_META_PULL` + `STRATA_PF_HOSTWAIT` (34.95 mean):

- `STRATA_HIT_FAST=1`: +0.2 % (35.02).
- `STRATA_HIT_FAST=2`: +0.3 to +1.3 % in three pairs. Its fourth run, 38.97, is left out: the whole PC was faster
  for one run.

Routes and logits were 50c29791 in every arm.

`STRATA_HIT_FAST` alone, without the other two, gave 34.05 -> 34.86 (+2.4 %, 3 pairs) and 38.5 -> 38.8 (+0.7 %,
2 pairs). Most of that is the metadata copy it does not make.

The PC itself moved between about 34 and 38.5 tok/s for the same arm during these runs. Only the interleaved pairs
are comparable. NetEase GameViewerServer was running; that cause is not confirmed.

## Tried and removed

`STRATA_INPUT_ASYNC`: a decode step's ~14 small inputs (embedding row, positions, window rows / mask, compressed-row
slots) went up from a pinned block with async copies, instead of ~14 blocking `tensor_set`s.

- Byte-identical, but no gain: 37.27 vs 37.22 tok/s over 3 pairs on top of the two switches above.
- The step's begin (~0.75 ms) is not those copies.

The first `STRATA_HIT_FAST` ran its launch on the tier's stream:

- The tier's stream waited for the attention graph's event, and the dense stream waited for the launch's sum.
- Both waits were released 0.2-0.3 ms late (nsys). It was slower than off (33.74 vs 34.24 tok/s, bit-identical).
