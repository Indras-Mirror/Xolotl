# Decode opt-ins from the DeepInfer review (rb10 - rb18)

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
predicts further layers. `=2` (rb16, below) takes the vision rows as well.

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

## Work inside the graphs (rb12)

nsys on the rb11 decode (GSQ, 10 s, about 170 tokens under the profiler) showed a third cost:

- An op queued on its own on the dense stream starts 3-10 us after the op before it ends, even when it was queued long
  before. Inside a CUDA graph, a node starts 0.1-0.5 us after the one before it.
- A layer had about 10 such ops: the three routing readbacks (fn, ids, wts), the prediction readback and the six
  kernels of `STRATA_HIT_FAST`.
- While a prefetch DMA is in flight, each one costs more. A readback took 3.7-5.7 us instead of 1.1-1.3, and the gap
  before it was 6-10 us instead of 4-8.
- Such gaps added up to 0.42 s of the 10 s on the dense stream.

### `GGML_OP_DSV41_HOOK` (vendored ggml)

`ggml_dsv41_hook(ctx, a, b, c, fn, ud)`: at this node the CUDA backend calls `fn(stream, node, ud)`, and `fn` queues
its own kernels on the graph's stream.

- Under CUDA graphs, `fn` runs when the graph is captured (the warm-up runs call it directly). The kernels it queued
  are replayed with the graph.
- The node runs after the ops that write `a`, `b` and `c`. Its result is a view of `a`.

### `STRATA_RB_HOOK=1`

Needs `STRATA_PIN_RB`, `STRATA_DENSE_ASYNC` and `STRATA_ATTN_TAIL`.

- The single-token attention graph ends with a hook node behind the router. Its kernel (`strata::kernels::push_to_host`,
  one block) writes fn, ids and wts into the pinned packet over the bus, then sets a flag word in the packet after a
  system-scope fence.
- The finish + prediction graph writes the 16 predicted ids and a second flag the same way.
- The host clears a flag before it launches the graph and spins on it. This replaces three D2H copies and an event for
  the routing, and a D2H copy and an event for the prediction.
- A flag that is still clear after 2 s gets a full sync, and an error if it is clear after that.

The packet gets the same bytes, so the routes and logits are byte-identical.

### `STRATA_HIT_FAST=3`

Needs `STRATA_RB_HOOK`. The launch of `=1` (dispatch, quantize, grouped launch, sum) becomes a second hook node in the
attention graph, behind the readback node. The host has the routing as soon as the readback kernel is done, while the
hits run on in the same graph.

- A node runs on every launch of its graph, so each launch must be armed. Before the graph goes out,
  `Ds41MoeTier::hit_arm(layer)` rebuilds the layer's table row and sets the layer's armed word in pinned memory. The
  dispatch reads the word and clears it.
- An unarmed launch does nothing, except that its quantize writes `d_xq` (the bytes the layer writes there anyway).
- The decode loop arms a layer before `finish_predict_attn` queues its graph, or before `attn_router` runs it.
- When a launch is not kept but found all six experts resident, the tier waits for the dense stream (the graph) before
  it writes the same buffers.
- The hooks are set up at the first decode step, because the tier opens its VRAM cache after the prompt. Setting them
  drops the cached graphs once.

## Fewer launches, branches inside the graphs (rb13)

nsys on the rb11 decode, per graph launch (medians, under the profiler):

- `cudaGraphLaunch` took 31 us for the attention graph (about 59 kernels) and 15 us for a small graph (about 13).
- The attention graph's first kernel started 5 us after its launch call returned: the card waited for the launch. A
  small graph's started 36 us after: it was queued behind earlier work.
- Inside the attention graph, the hc mixes of the block input (rms norm 4.9 us, the 24-row product 8.5 us, mixes 0.8,
  comb 1.8) ran between the output projection and `hc_post`, although only the block input feeds them.

### `STRATA_ATTN_TAIL=3`

Needs `STRATA_RB_HOOK` (without it, it runs as `=2`). The tail's ops (the FFN hc mixes and the shared expert) are the
end of the attention graph itself: after the readback and hit hooks and the ring writes, the order they ran in as a
graph of their own, without that graph's launch.

### `GGML_OP_DSV41_STREAM` (vendored ggml)

`ggml_dsv41_stream(ctx, a, mode, k, keep, n)` runs part of a graph on another stream of the CUDA backend:

- mode 0 (fork): the nodes after it run on stream k, after the work queued before it.
- mode 1 (end): the nodes after it run on the main stream again.
- mode 2 (join): the main stream waits for stream k's work up to its last end node.

Inside a CUDA graph a fork and a join are edges, the mechanism of ggml-cuda's own Q/K/V concurrency. The result is a
view of `a`. The `keep` tensors stay allocated up to the node, so a join keeps a branch's graph tensors alive and the
allocator gives none of their memory to a node that runs beside the branch.

### `STRATA_GRAPH_CONC=7`

Needs `STRATA_ATTN_TAIL` and the fused hc mix op. A sum of:

- 1: the attention's hc mixes of the block input on stream 1, beside the attention. Forked behind the attention's
  input (`xin`), joined before `hc_post`.
- 2: `STRATA_HIT_FAST=3`'s hits on stream 1, behind the routing readback, joined at the graph's end.
- 4: with `STRATA_ATTN_TAIL=3`, the FFN hc mixes and their writes on stream 2, beside the shared expert, joined at the
  end.

The same kernels run on the same inputs. In nsys, `=6` took the attention graph from 224.5 to 202.6 us.

A fork as the graph's first operation (two root nodes) was slower: `=1` lost 4.9 %, and nsys's node tracing crashed
on that graph. Behind `xin` the graph has one root.

### `STRATA_QKV_FUSE=1`

A layer's `attn_q_a` (5120 -> 1280) and `attn_kv` (5120 -> 512) both multiply `xn`, and both are Q8_0 here. With
this switch they are two views of one [5120, 1792] tensor in VRAM. The upload writes each through its view, so it
takes no extra memory.

A one-token graph multiplies them as one matrix and splits the result. mmvq's launch shape depends on the type, K and
the token count, not on the row count, so each row's dot product is the same. The prompt keeps two products (MMQ's
stream-k split can depend on the row count).

## Two copies the sum waited for (rb14)

### `STRATA_DEMOTE_ASYNC=1`

When vram_lru gives an expert from the arena a VRAM slot, the slot's old expert (the victim) moves into the arena slot
the new one leaves: a demotion. Its D2H (about 9 MB) ran on the tier's stream ahead of the layer's sum, so the sum's
sync waited for it. In the profile below, a layer call without PCIe or file-tier experts spent 0.100 ms in the sum's
sync, and 0.348 ms when it demoted (2170 of 43680 calls, about 2 a token).

With this switch the victim is first copied inside VRAM, into one of two spare d_stage slots, and the D2H from there
runs on a stream of its own. The sum waits only for the copy inside VRAM.

- The spare slots are d_stage 6 and 7. A call fills d_stage from the bottom (the PCIe share, then file-tier blobs);
  with top-k 6 that never reaches slot 6. A call whose staging would reach them waits for their D2H first.
- The arena slot gets its bytes later than before, so what reads or writes it waits for the D2H: the next call for
  the same layer and the prefetches for that layer wait for that layer's demotions (long done by then). None is left
  in flight after the token's last routed layer, or when a prompt or multi-token call starts.
- The same bytes reach the same arena slot. `DS4_CHECK_LRU=1` (every swap's VRAM slot and demoted arena copy against
  the file, byte for byte) over the forced run: 5000+ swaps, 0 bad.

`DS4_TIER_PROF`, one run each (rb13 set + `GGML_CUDA_FP4_PACK`), the sum's sync in ms:

| layer calls | count | without | with |
|---|---|---|---|
| no PCIe expert, no file read, a demotion | 2170 | 0.348 | 0.104 |
| one PCIe expert, no file read, a demotion | 283 | 0.389 | 0.219 |
| no demotion (for comparison) | 11631 | 0.100 | 0.100 |

That is about 0.55 ms a token; the token time shrank by about half of it (below). Where the rest goes was not traced.

### `GGML_CUDA_FP4_PACK=1` (vendored ggml)

With `--kv-fp4`, `set_rows` packs the compressed KV rows into NVFP4 and the indexer keys into MXFP4. The packers look,
per 16 values (NVFP4) or 32 (MXFP4), for the scale that reproduces the fake-quantized values exactly. They ran one
thread per block, so a 512-value decode row was 8 threads, each trying up to 7 scales on 4 sub-blocks in turn.

With this switch every value has its own thread. The lanes of a (sub-)block try each candidate scale together and keep
the first one all of them reproduce (a warp vote): the same candidates in the same order, the same bytes. CUDA only;
HIP and MUSA keep the per-block kernel.

- A set_rows test against the per-block kernel was byte-identical. It wrote 1 to 4096 rows, 11 seeds, of: values on
  the NVFP4 and MXFP4 grids at every E4M3 scale, rows that several scales reproduce, values off the grid (some with
  amax = 3 * 2^k), zeros, huge and tiny values, inf and NaN.
- nsys, one 512-value row alone: NVFP4 18.1 -> 2.8 us, MXFP4 6.3 -> 1.4 us.
- nsys in the decode: the packers 14.7 -> 2.1 us a call. The attention graphs that write a compressed row (4 a token)
  took 283 -> 265 us. That is about 0.07 ms a token, below what the decode runs below can resolve.

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

### rb12

GSQ, the same run, interleaved pairs, in tok/s. Every arm has the rb10 switches, `STRATA_META_PULL` and
`STRATA_PF_HOSTWAIT`.

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| `HIT_FAST=1` / + `STRATA_RB_HOOK` | 38.79 / 39.93 | 39.26 / 39.31 | 38.17 / 38.84 | 40.96 / 41.62 | 39.30 / 39.92 (+1.6 %) |
| `RB_HOOK` + `HIT_FAST=1` / `=3` | 38.93 / 40.10 | 40.30 / 41.30 | 39.33 / 39.94 | 40.19 / 41.12 | 39.69 / 40.61 (+2.3 %) |

A third set, three arms with the same switches:

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| `HIT_FAST=1` | 37.82 | 39.67 | 38.44 | 39.68 | 38.90 |
| + `STRATA_RB_HOOK` | 38.84 | 40.14 | 40.21 | 39.01 | 39.55 (+1.7 %) |
| + `STRATA_RB_HOOK` + `HIT_FAST=3` | 41.29 | 40.47 | 41.24 | 41.01 | 41.00 (+5.4 %) |

Routes and logits were 50c29791 in every arm. `=3` kept 19655 of 43680 armed launches (45 %).

### rb13

GSQ, the same run, interleaved, in tok/s. Every arm has the rb12 best set: the rb10 switches, `STRATA_META_PULL`,
`STRATA_PF_HOSTWAIT`, `STRATA_RB_HOOK` and `STRATA_HIT_FAST=3` (`STRATA_ATTN_TAIL=2` from the configuration).

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| rb12 best | 41.22 | 41.24 | 41.26 | 41.25 | 41.24 |
| + `ATTN_TAIL=3` | 41.48 | 41.45 | 41.48 | 41.47 | 41.47 (+0.55 %) |

Each set below starts from rb12 best + `ATTN_TAIL=3`. The PC was in its slower state (about 39 tok/s for it).

| arms | pair 1 | pair 2 | pair 3 | mean |
|---|---|---|---|---|
| + nothing | 39.39 | 39.29 | 39.26 | 39.31 |
| + `GRAPH_CONC=2` | 40.02 | 39.94 | 39.99 | 39.98 (+1.70 %) |
| + `GRAPH_CONC=4` | 39.67 | 39.70 | 39.74 | 39.70 (+0.99 %) |
| + `GRAPH_CONC=6` | 40.19 | 40.20 | 40.19 | 40.19 (+2.24 %) |

| arms | pair 1 | pair 2 | pair 3 | mean |
|---|---|---|---|---|
| + nothing | 39.18 | 39.24 | 39.32 | 39.25 |
| + `GRAPH_CONC=6` | 40.22 | 40.13 | 40.20 | 40.18 (+2.39 %) |
| + `GRAPH_CONC=7` | 40.59 | 40.62 | 40.56 | 40.59 (+3.42 %) |

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| + `GRAPH_CONC=7` | 40.32 | 40.61 | 40.62 | 40.50 | 40.51 |
| + `GRAPH_CONC=7` + `QKV_FUSE` | 40.63 | 40.74 | 40.72 | 40.62 | 40.68 (+0.41 %) |

The whole rb13 set against rb12 best, on the same build:

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| rb12 best | 38.32 | 39.10 | 38.99 | 42.57 | 39.75 |
| + `ATTN_TAIL=3` + `GRAPH_CONC=7` + `QKV_FUSE` | 39.47 | 40.74 | 40.64 | 44.32 | 41.29 (+3.9 %) |

The PC moved to its fast state for pair 4. Routes and logits were 50c29791 in every arm.

### rb14

GSQ, the same run, interleaved, in tok/s. Every arm has the rb13 set (rb12 best + `ATTN_TAIL=3` + `GRAPH_CONC=7` +
`QKV_FUSE`).

| arms | pair 1 | pair 2 | pair 3 | pair 4 | mean |
|---|---|---|---|---|---|
| rb13 set | 39.99 | 39.96 | 39.35 | 39.50 | 39.70 |
| + `FP4_PACK` | 39.76 | 39.84 | 39.57 | 39.49 | 39.67 (-0.1 %) |
| + `FP4_PACK` + `DEMOTE_ASYNC` | 40.13 | 40.12 | 40.05 | 39.97 | 40.07 (+0.9 %) |

Routes and logits were 50c29791 in every arm, and with `DS4_CHECK_LRU=1` too. `FP4_PACK`'s expected +0.3 % is within
the pairs' spread (+-0.3 %).

## Serve acceptance of the rb14 candidate

The candidate: the rb14 exe, the exact switches since rb8g (`STRATA_INDEXER_NVIS`, `GGML_CUDA_MXFP4_WMMA`,
`STRATA_KV_GATHER`, `STRATA_WIDE_PROMPT`, the rb10 - rb13 sets, `STRATA_DEMOTE_ASYNC`; not `GGML_CUDA_FP4_PACK`),
`STRATA_SERVE_STEP_MAX=16` and `--prompt-batch 32768`. The reference: the GSQ test configuration in use (rb5 exe).
Eleven greedy serve scenarios each (srvtest: reuse, a conversation coming back, a 15.7K prompt, thinking, a cancelled
stream continued last).

| pair | all-decode tok/s | answers |
|---|---|---|
| rb5 config / candidate, as configured | 38.68 / 43.55 (+12.6 %) | all sensible, worded differently |
| rb14 exe: config / + the switches (1) | 39.28 / 48.32 (+23 %) | 8 of 11 identical (2) |
| the same with `STRATA_SERVE_KEEP_CACHE=0` (1) | 45.78 / 54.04 (+18 %) | 11 of 11 identical |

(1) Both arms: `--slots 2350`, `STRATA_ARENA_REBAL=0`, `STRATA_PRESTAGE_GIB=0`, the configuration's STEP_MAX 96 and
batch 12288. The auto seed gave rb5 2396 slots and rb14 2401: another placement, other rounding.

(2) The configuration itself is not reproducible from the first short batch turn on: the same arm run twice differed
from that turn (631 tokens) on. keep-cache (default on since the 10-26 upstream merge) lends the cache's tail to such
a turn, and how many slots it takes depends on the free VRAM it measures at that moment: 587 to 649 slots over five
runs of the same turn. The cache after the turn differs, and so does the placement. rb5 has no keep-cache.

Prompt turns, serve wall as configured (rb5 / candidate): 631 new tokens 18.2 / 6.6 s, the 15.7K first prompt 49.9 /
23.2 s, 323 new tokens on the 15.7K conversation 18.6 / 7.8 s.

One-shot, the 15.3K prompt + 300 forced tokens (`--slots 1900`):

| | rb5 | rb14, the configuration's settings | the candidate |
|---|---|---|---|
| prompt | 33.9 s | 14.9 s | 9.8 s |
| decode | 32.56 tok/s | 32.39 | 36.90 |
| routes + logits | 04abb074 | 62281d0f | 62281d0f |

rb5 differs from rb14 by the base changes since rb5 (for example rb8e's tie rule in the prompt top-k). The pairs ran
one after the other, and the PC moved to a faster state during the runs: compare within a pair only.

## Keep-cache's fixed borrow (rb15)

`STRATA_SERVE_KEEP_FIXED=1` (opt-in): for a short batch turn, keep-cache gives back a fixed part of the VRAM cache,
worked out from the batch alone. Before, it gave back what a free-VRAM reading said was missing.

- The part: what the batch needs by the width check below it (both terms, plus one 64 MiB segment), plus
  `STRATA_SERVE_KEEP_PAD_GIB` (default 1.5). The pad stands for what the decode's own buffers took of the 2 GiB margin
  since the seed: 0.5 to 1.3 GiB on GSQ.
- The old target stays as a floor. If the free VRAM after the fixed part is still short of it, the rest comes from
  the reading as before, and the log says that the turn is not reproducible.
- Cost: 30 to 140 more slots out than the reading asked for. Refilling 656 to 728 slots took 205 to 319 ms, so about
  10 to 40 ms more per batch turn.

The new GSQ configuration (the rb14 candidate + this), srvtest, 11 scenarios:

| runs | slots out per batch turn | answers |
|---|---|---|
| twice, back to back | the same in both: 665 (21-80 tokens), 728 (631), 656-684 after the 15.7K prompt | differ from the 2nd request on |
| twice, `STRATA_ARENA_REBAL=0` | the same | 11 of 11 identical, the cancelled stream too |
| `STRATA_ARENA_REBAL=0`, 8 s idle before each request | the same | identical up to the 15.7K prompt, then not |

- With the rebalance on, the arena's background swaps finish at times that depend on the gaps between requests, and
  an expert's tier changes CPU vs GPU rounding (known since 2026-10-22).
- With the 8 s gaps, the cache's re-seed after the 15.7K prompt took 2367 slots instead of 2364. The re-seed is sized
  from the free VRAM at that moment (23.31 to 23.34 GiB over six runs). rb16's `STRATA_SERVE_RESEED_FIXED` (below)
  sizes it from the first seed instead.
- The rebalance did not change the speed here. All-decode, back to back: 48.03 / 47.58 tok/s on, 49.05 / 49.73 off.
  With the 8 s gaps, when all 3,385 planned swaps were done: 48.05 on, 47.73 off. Prompt phases matched to 0.05 s.
- The K:\model\test configuration runs with `STRATA_SERVE_KEEP_FIXED=1` and `STRATA_ARENA_REBAL=0` since 2026-10-28.

## The --vision router, the hit lookups and the re-seed (rb16)

A review of the rb15 setup (2026-10-28) found two places that the one-shot runs above did not cover. The server adds
`--vision` to the engine whenever its config has a "vision" section, and the K:\model\test configuration has one.

### `STRATA_ROUTE_FUSE=2`

With `--vision`, the router's selection bias is the image-position mix
`(exp_probs_b_vl - exp_probs_b) * mask + exp_probs_b`, a computed tensor. So `=1` fell back to the stock chain on
every layer of every token: the mix's 4 ops and the 10 kernels instead of one.

`=2` hands `exp_probs_b_vl` and the mask to `ggml_dsv41_route_vl`, which forms the mix in the kernel:

- It uses `__fsub_rn` / `__fmul_rn` / `__fadd_rn`, the stock sub, mul and add.
- The mask is 0 or 1, so the product is exact.
- It covers every row: decode tokens and image prompts.

### `STRATA_HIT_PAR=1`

`decode_hit_dispatch` (`STRATA_HIT_FAST`) looked up the six experts' VRAM addresses one after the other. The table and
the armed word live in mapped pinned memory, so that was 7 bus round trips in a row (p50 9.9 us a layer in the GH26R
trace).

With this switch, lane k looks up routing k while lane 0 reads the armed word: one round trip.

- `grp_ptr` is written only when all six are resident.
- With `n_groups` 0, nothing reads it.

Forced 1093, `--slots 2378`, the rb13 set + `STRATA_DEMOTE_ASYNC`, route bias 0.25 / 0.15, 3 rounds interleaved
(tok/s):

| arm | runs | mean |
|---|---|---|
| no `--vision` | 38.10 / 38.05 / 37.41 | 37.85 |
| `--vision`, `ROUTE_FUSE=1` (as in rb15 production) | 37.99 / 37.14 / 37.14 | 37.42 |
| `--vision`, `ROUTE_FUSE=2` | 37.70 / 37.46 / 37.77 | 37.64 |
| `--vision`, `ROUTE_FUSE=2` + `HIT_PAR=1` | 38.15 / 37.72 / 37.67 | 37.85 |

- Routes (50c29791) and token ids (558cec61) were the same in all arms, with `--vision` or without. The final build
  gave the same.
- `--vision` cost about 1 %, less than the extra kernels suggested.
- Each switch alone is within the noise. The two together are +1.1 %, faster in all three rounds.

### `STRATA_SERVE_RESEED_FIXED=1`

A prompt of more than 4096 new tokens releases the VRAM cache, and the decode after it re-seeds the cache. The
re-seed was sized from the free VRAM at that moment, so it took 2364 to 2367 slots, depending on how long the card had
been idle before.

With this switch, every seed after the first asks for the first seed's GiB minus `STRATA_SERVE_RESEED_PAD_GIB`
(default 0.5), rounded down to 1/16 GiB.

- The pad covers the ~0.32 GiB the process grows at its first long prompt (7.05 -> 7.37 GiB, the same for 16K to
  274K prompts).
- The free VRAM minus the margin still caps the seed. A seed under the target is logged as not reproducible.
- On GSQ, the first seed is 21.63 GiB (2401 experts), and the re-seed 21.12 GiB (2344) instead of 21.31 GiB
  (2364-2365). That is 21 fewer experts, after long prompts only.

### Serve, the live K:\model\test configuration

Route bias 0.5 / 0.3, srvtest with the image probe first, 12 scenarios. rb16 = the rb15 configuration + the three
switches above.

| run | answers | decode, the first 6 scenarios | decode, all |
|---|---|---|---|
| rb15 (production) | reference | 50.99 tok/s | 51.23 |
| rb16 | the first 6 the same as rb15, image answer included; then not (the re-seed) | 52.67 | 51.57 |
| rb16, 8 s idle before each request | all 12 the same as the run above, the cancelled stream too | 51.11 | 51.14 |
| rb16, second start | all 12 the same | 53.30 | 53.42 |
| rb15, second start | the first 6 the same | 52.02 | 52.37 |

- Over the first 6 scenarios (the same answers in all runs), rb16 was +3.3 % and +2.5 % in the two pairs.
- The image answer was the same in all runs, and its 232-token prompt took 3.42 -> 3.22 s.
- The runs with 8 s gaps decode slower, rb15's too (47.73 against 49.05 / 49.73). The cause was not traced.

Checked, no change: `STRATA_SERVE_STEP_MAX=48` against 16, on the rb16 configuration.

- A 21 / 34 / 45-token turn as passes took 1.07 / 1.67 / 2.15 s, and as one batch 0.90 / 1.31 / 1.53 s.
- A 31-token turn took 0.97 s both ways.
- 16 stays.

## Prefetched experts and partial layers in the graph (rb17)

The review also suggested running a layer's resident experts in the attention graph when one of its six is missing.
`DS4_TIER_PROF` now also splits the tier calls by kind. On the K:\model\test configuration (route bias 0.5 / 0.3,
forced 1093, `--slots 2378`, `--vision`), rb16:

| kind of layer call | calls | ms a call |
|---|---|---|
| all six resident, run in the graph (kept) | 26725 | 0.002 |
| hits + pool (no PCIe share, no file read) | 10733 | 0.303 |
| a PCIe share | 422 | 0.550 |
| a file read | 5800 | 1.041 |

The hits + pool calls averaged 5.70 hits and 0.30 pool experts, and 10778 prefetched experts were used. So about
7500 of them were six hits with one prefetched:

- The prefetch for layer L goes out after L's graph. `STRATA_ATTN_AHEAD` queues that graph with the previous layer's
  finish, so the graph's dispatch only had the table of resident slots.
- The tier then waited for the DMA, launched all six and synced their sum.

### `STRATA_HIT_PF=1` (with `STRATA_HIT_FAST=3`)

The graph's dispatch also takes the layer's prefetched experts.

- prefetch() publishes its list (expert ids and staging addresses) in mapped pinned memory, stamped with the layer's
  arming. A copy queued behind the DMAs writes the stamp into device memory when they are home.
- The dispatch matches its routed misses against the list and publishes its decision: all six or not, and which.
  Then it waits for the stamp, so the grouped launch behind it reads the DMA'd bytes.
- After 50 ms that wait sets an error word, and the host aborts on it. That never happened.
- The host reads the decision only when a routed expert is in the prefetch. Otherwise the table decides, as before.
- Odd layers stage in a second buffer (6 blobs, ~55 MB, allocated at the first decode, after the seed), so a
  prefetch never writes what the previous layer's graph may still read.
- vram_lru's copy of a prefetched expert into a slot waits for the prefetch on the host first.
- Where a layer's graph was not queued ahead, its arming moves before its prefetch, so the list carries the stamp.

### `STRATA_HIT_PART=1` (with `STRATA_HIT_PF`)

A layer with a miss runs its resident and prefetched experts in the graph: packed, each with its routing index as its
row. The in-graph sum is gated off, and the call computes only the rest.

- When the rest is the pool's alone (arena experts, no PCIe share, no file read), the call launches nothing on the
  card. Its sum goes on the dense stream behind the graph, unsynced; the finish behind it reads the output in order.
- Otherwise the call's own card work waits for the graph's grouped launch: the graph's sum kernel writes a stamp
  into mapped memory once it has run. The layer's file reads go out before that wait.
- A graph event-record node for that wait was refused in the capture ("the operation cannot be performed in the
  present state").

Both switches run the same kernels on the same inputs, and the sums add in the same order (`decode_rows_sum_gated` /
`_sel`).

Forced 1093, `--slots 2378`, the K:\model\test configuration (rb16), 3 rounds interleaved (tok/s):

| arm | runs | mean |
|---|---|---|
| rb16 | 46.13 / 45.89 / 46.01 | 46.01 |
| + `STRATA_HIT_PF` | 47.09 / 46.63 / 46.83 | 46.85 (+1.8 %) |
| + `STRATA_HIT_PF` + `STRATA_HIT_PART` | 47.09 / 47.43 / 47.68 | 47.40 (+3.0 %) |

- Routes (4c5a4ba7) were the same in all arms.
- With both, by kind (one profiled run):
  - kept: 34262 calls, 7537 of them with a prefetched expert, 0.015 ms;
  - hits + pool: 3196 calls, 0.216 ms (0.182 of it the pool);
  - a file read: 5800 calls, 1.050 ms.
- Why not more: a prefetched expert's DMA lands 0.1 to 0.2 ms after the routing. That wait moved from the host to the
  card.
- Caught on the way by the route hash (46edbe66 instead of 4c5a4ba7): once the decision came before the DMA's end,
  vram_lru copied a prefetched expert into its slot before its bytes were home. The host wait above fixed it.

Serve, the K:\model\test configuration (rb16), srvtest with the image probe first, 12 scenarios, rb16 / rb17
alternated twice:

| run | answers | decode, all |
|---|---|---|
| rb16 | reference | 52.27 tok/s |
| rb17 + both switches | all 12 the same, the image and the cancelled stream too | 54.99 |
| rb17, 8 s idle before each request | all 12 the same | 54.80 |
| rb16, second start | all 12 the same | 53.13 |
| rb17, second start | all 12 the same | 55.09 |

- +5.2 % and +3.7 % in the two pairs, every scenario faster.
- Prompt times did not change: 29.0 to 29.4 s over the 12 scenarios.
- The re-seed after the 15.7K prompt was 2344 experts in all five runs.

## Waiting only for the prefetched experts a layer takes (rb18)

The rb17 review pointed at two waits:

- The dispatch waited for the stamp written behind ALL of the layer's prefetch DMAs, even when it took only the first.
- The resident experts' grouped launch waited behind that wait too.

`STRATA_HIT_PF_STAT=1` (a diagnostic) counts the dispatches' waits on the card. It also queues a copy behind each
list entry's DMA, so it sees when the taken ones were home. Forced 1093, the K:\model\test configuration (rb17
settings):

| list (real DMAs) | last taken | waits | to the taken | to the end of the wait |
|---|---|---|---|---|
| 1 | first | 7741 | 55.3 us | 56.4 us |
| 2 | first | 1729 | 54.6 us | 219.5 us |
| 2 | second | 753 | 212.2 us | 213.2 us |

- 10223 waits: 682 ms until the taken experts were home, 976 ms until the waits ended.
- A list's second expert lands ~160 us after its first (about 58 GB/s for a ~9.5 MB blob).
- In the old trace a full in-graph launch of six experts took ~54 us (gate/up 30, down 23).

### `STRATA_HIT_PF=2`

`=1`, and each list entry gets its own ready word, written by a 4-byte copy right behind that entry's DMA. The dispatch
waits for the words of the entries it takes. The batch's word is no longer written. The waits drop to 725 ms; the
"list 2, first taken" ones to 54.7 us.

### `STRATA_HIT_PF=3`

`=2`, and while the graph is captured the dispatch splits its run into two sets:

- The residents run at once on the dense stream.
- The prefetched experts run on a second stream, a branch of the graph: a wait kernel, then their own grouped launch
  with their own scratch.
- The sum runs after both, in routing order.

Outside a capture it is `=2` (a stream wait there costs 0.1-0.3 ms on WDDM).

### The timeout fails closed

- After the 50 ms timeout, the dispatch (or the wait kernel) also zeroes the groups and the sum gate. Nothing behind it
  reads the stage.
- The host stops at the next layer's call, as before. It now also stops before a logits read that came after the
  timeout (ds41_generate's head_logits), so a timeout in the last layer cannot reach a sampled token.
- `STRATA_HIT_PF_FAULT=<layer>[:<n>]` (a test): from the layer's n-th prefetch on, no ready word is written.
- Forced runs with n = 200, at layer 20 and at layer 39, with `=1` and with `=3`: all four stopped. Every top32
  record before the stop was rb17's, byte for byte (203 and 208 records), and the stopped step wrote none.

### Measured

Forced 1093, `--slots 2378`, `--vision`, the K:\model\test configuration, interleaved (tok/s):

| arm | runs | mean |
|---|---|---|
| rb17 | 51.40 / 51.34 / 51.28 | 51.34 |
| rb18a `=1` | 51.26 / 51.32 / 51.20 | 51.26 |
| rb18a `=2` | 51.75 / 51.67 / 51.78 | 51.73 (+0.8 %) |

| arm | runs | mean |
|---|---|---|
| rb17 | 51.17 / 51.26 / 51.28 | 51.24 |
| rb18b `=2` | 51.63 / 51.74 / 51.64 | 51.67 (+0.8 %) |
| rb18b `=3` | 51.79 / 52.00 / 51.76 | 51.85 (+1.2 %) |

- Routes 4c5a4ba7 in every run.
- `=3` gains less than the waits suggested. In nsys (8 s of each), GPU idle in gaps of 20 us or more is ~7.7 ms a
  token. ~5 ms of it is the file-read layers (quantize -> the sum: 832 us, 4.9 layers a token); the rest is many
  small host gaps.
- The same rb17 exe ran 47.45 tok/s at 18:20 and 51.3 tok/s from 19:40, with the same file and pool times. Compare
  runs of one session only.

Serve, the K:\model\test configuration (rb17 = production), srvtest with the image probe first, 12 scenarios, rb17
/ rb18c `=3` alternated twice:

| run | answers | decode, all |
|---|---|---|
| rb17 | reference | 58.03 tok/s |
| rb18c `=3` | all 12 the same, the image and the cancelled stream too | 61.66 |
| rb17, second start | all 12 the same | 62.23 |
| rb18c `=3`, second start | all 12 the same | 62.96 |
| rb18c `=3`, 8 s idle before each request | all 12 the same | 62.81 |

- The first run of a series was the slowest here as in the rb17 check, so the first pair's +6.3 % is mostly that.
  The second pair: +1.2 %.
- The re-seed after the 15.7K prompt was 2344 experts in all five runs.

### `STRATA_ARENA_DEDUP=1` (opt-in, NOT the same bits)

The forced run still had 1392 experts both in VRAM and in the RAM arena after 1075 tokens. `gpu_run` refreshes
`last_use` for every routed expert, so an arena copy of a VRAM expert stays "recent" though the pool never reads it.

With the switch, a file read's arena victim is the least recently used expert that the VRAM cache also holds, if
there is one. A vram_lru eviction of such an expert demotes it back into the arena (as before).

Forced 1093, rb18c `=3`, 3 pairs:

| arm | tok/s | file reads | file ms | both, token 1075 |
|---|---|---|---|---|
| off | 52.94 / 51.97 / 52.12 (52.34) | 6706 | 4762 | 1392 |
| on | 53.66 / 53.78 / 53.83 (53.76, +2.7 %) | 5182 | 3750 | 322 |

- The bits change: more misses come from the arena, so the PCIe share and the prefetches change (CPU vs card
  rounding).
- Its drift against rb17 on the forced trajectory: top-1 1067 / 1093, KL (top-32 lower bound) mean 7.2e-3.
- For scale: `STRATA_PCIE_SPLIT=0` against `=1` (production has `=1`) gives top-1 1075 / 1093 and KL 7.0e-3. The route
  bias change 0.25 / 0.15 -> 0.5 / 0.3 gives top-1 1023 / 1093 and KL 8.7e-2.
- Two runs with the switch were byte-identical to each other.
- Not checked: serve. Keep-cache's regrow and the re-seed after a long prompt fill VRAM from the arena, and with the
  switch most VRAM experts have no arena copy. Those fills would read the SSD instead.
- Serve (2026-10-29, the K:\model\test config, 12 scenarios, order rb18 / C / rb18 / C / C with 8 s gaps): all decode
  61.48 / 64.32 -> 62.98 / 65.49 / 65.19 tok/s; C's three runs gave the same answers (the gaps too); the re-seed after
  the 15.7K prompt 0.93 -> 1.07 s.  The sequence's first run was slow in everything, so the serve gain is 0 to +1.8 %.

## Engram: a hot set for the IQ4_NL GGUF, and cold reads on demand (rb20)

The Engram table here is ISTA's IQ4_NL GGUF (110.6 GB, 144-byte rows), read through the OS file cache.  A text the
engine has not read since the cache lost its rows waits for them: forced 1093 on two new texts, `STRATA_ENG_PROF=1`,
0.37 / 0.41 ms per pass at layer 1 on the first run, 0 on the second, decode +1.3 / +1.6 % on the second.

### `STRATA_ENGRAM_NOCACHE=1` (a test switch)

The tables are opened with `O_DIRECT` (`FILE_FLAG_NO_BUFFERING`), and each read goes through a 4 KiB-aligned bounce
buffer.  Every row read then comes from the drive, so the same text can be measured cold again and again.

### Hot set for a GGUF table

`load_hot` refused the GGUF tables (the packs held FP8 rows).  A version-2 pack holds the GGUF table's own raw rows
(header + table type and row size; `ds41_engram.cpp`, `load_hot`), and loads only into the same tables (the 64-row
check against the tables stays).  `ds41_engram_hot` builds on Windows now, sizes its budgets with `--row-bytes 144`,
and eval reads a -2 marker in a document as where its answer starts.

Ranked on the strata-kld corpus (ja / en / code / koten, 385K tokens, with Strata's DETAILS.md cut out because the serve's
long prompt reads it): 1.66M rows used by two or more 2048-token windows = a 0.22 GiB pack.  Simulated on the serve
scenarios' text (cold cache, the 2M-row cache): decode 37.8 -> 34.6 disk reads a token, prefill 33.9K -> 26.4K a 1K
tokens.

Serve, the K:\model\test config with `STRATA_ENGRAM_NOCACHE=1`, order no hot / hot / hot / no hot:

| | no hot | hot set |
|---|---|---|
| answers vs the production config | 12 / 12 same | 12 / 12 same |
| Engram wait at layer 1, ms a pass | 0.95 / 1.01 | 0.85 / 0.85 |
| the 15.7K prompt, ms | 11010 / 10802 | 10460 / 10459 (warm cache: 10483) |
| all prompts, s | 28.7 / 28.4 | 28.0 / 28.0 (warm: 28.2) |
| all decode, tok/s | 63.80 / 63.99 | 64.04 / 64.05 |

- With every Engram read from the drive, the long prompt is only 3-5 % slower than warm, and decode ~1.5 %: the reads
  overlap other work, and this SSD is fast.  So a hot set can win at most that, on text the cache does not hold.
- The 0.22 GiB set takes back the long prompt's cold cost and +0.2 % decode.

## Tried and removed

`STRATA_INPUT_ASYNC`: a decode step's ~14 small inputs (embedding row, positions, window rows / mask, compressed-row
slots) went up from a pinned block with async copies, instead of ~14 blocking `tensor_set`s.

- Byte-identical, but no gain: 37.27 vs 37.22 tok/s over 3 pairs on top of the two switches above.
- The step's begin (~0.75 ms) is not those copies.

The first `STRATA_HIT_FAST` ran its launch on the tier's stream:

- The tier's stream waited for the attention graph's event, and the dense stream waited for the launch's sum.
- Both waits were released 0.2-0.3 ms late (nsys). It was slower than off (33.74 vs 34.24 tok/s, bit-identical).

`STRATA_ATTN_AHEAD=2`: `finish_predict_attn`'s two graphs, built once more as one graph (the same nodes in the same
order, its own tensors).

- It saves a launch, but -0.45 % against `ATTN_TAIL=3` (41.47 -> 41.28 tok/s, 4 pairs, bit-identical).
- The one launch call has to go out before any of it runs. The small finish + prediction part then waits for the
  attention graph's launch call (the 31 us above), and the prediction lands later.

The first `STRATA_GRAPH_CONC=1` forked at the very start of the attention graph (two root nodes): -4.9 % (39.27 ->
37.36 tok/s, 4 pairs, bit-identical). Forked behind `xin` it gains (above).

`STRATA_GRAPH_CONC` + 8 (rb16): the hits' branch forked right after the router, beside the readback kernel instead of
behind it. The hits read only the routing and the FFN input on the card, and the arming happens before the graph goes
out. Bit-identical, but 37.85 -> 37.73 tok/s on top of the two rb16 switches (3 rounds).
