# Decode opt-ins from the DeepInfer review (rb10)

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

## Tried and removed

`STRATA_INPUT_ASYNC`: a decode step's ~14 small inputs (embedding row, positions, window rows / mask, compressed-row
slots) went up from a pinned block with async copies, instead of ~14 blocking `tensor_set`s.

- Byte-identical, but no gain: 37.27 vs 37.22 tok/s over 3 pairs on top of the two switches above.
- The step's begin (~0.75 ms) is not those copies.
