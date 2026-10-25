# V4.1 lightning indexer: visibility and checked MXFP4 WMMA

The normal path is unchanged. Set `STRATA_INDEXER_NVIS=1` to pass visible-key counts
straight to the indexer instead of materialising an additive mask. Set
`GGML_CUDA_MXFP4_WMMA=1` as well to use the existing NVIDIA WMMA kernel for MXFP4
keys. The switches affect the indexer only; candidate-block masks, candidate
selection, canonical top-k ordering, and `STRATA_INDEXER_TILE` stay in the graph.
`STRATA_INDEXER_NVIS=0` and `GGML_CUDA_INDEXER_VEC=1` are reference controls.

## Source and build

`tools/xolotl/fetch-llama.sh` and the CMake FetchContent path apply the complete
`tools/xolotl/ggml/xolotl-ggml.patch` to llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. That complete patch is the source of
these ggml changes. `tools/v41/ggml/kv-fp4.patch` is the historical packing patch;
it is already included in the complete patch and must not be applied again.
With `STRATA_GGML_DIR`, use a private checkout carrying the complete patch; CMake
checks for it but does not modify that external checkout.

Build `ds41_generate` and `test_ds41_indexer` with the usual engine flags.
`test_ds41_indexer` runs the CPU visibility gate; `test_ds41_indexer --cuda --large`
checks visibility and WMMA at ID=128, IH=32, top-512 and capacities up to 262144.
All GPU tests must run under the host's GPU arbiter.

## Visibility

`ggml_lightning_indexer` still accepts an F16 additive mask. CPU and CUDA also
accept F32 `[1, queries, 1, mask_streams]` visible-key counts, broadcast across
streams the same way as the additive mask. Key i is visible exactly when
`i < count`. An invisible block writes `-inf` and returns before loading Q or K;
a partial block avoids loading invisible K rows. The visible vec arithmetic is
unchanged, including its final addition of zero. Every output cell is written.

## MXFP4 precision and fallback

An MXFP4 block is e2m1 times `2^(e-127)`. All its values are exactly representable
in fp16 for E8M0 e=104..140. WMMA uses the more conservative e=114..140 range:
nonzero values are then normal fp16 (2^-14 through 49152). PTX leaves MMA
subnormal-input handling unspecified. Before using WMMA, each block checks all
Q values for finite, exact, normal-or-zero fp16 conversion and checks the scales
of its visible K blocks. A failed check runs the same vec implementation in that
block, on the same CUDA stream, without a host synchronisation or an allocation.

WMMA uses fp16 operands and fp32 accumulators. The dot-product reduction order
can differ from vec; the ReLU, head weights and sequential fp32 head accumulation
(including FMA) follow vec. This is not a general promise of bitwise WMMA parity.
The opt-in switch must be validated on each target GPU and model.

NVIDIA's precision contract:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-wmma-mma

## Real-input replay

Set `STRATA_INDEXER_DUMP=/existing/directory` for one diagnostic prefill, with
`STRATA_INDEXER_NVIS=1`. It retains and writes the first eligible input quartet
per layer as `indexer.lN.idx` (Q, packed K, weights, nvis). Existing files are
preserved. `STRATA_INDEXER_DUMP_MIN_KV` defaults to 65536;
`STRATA_INDEXER_DUMP_MIN_FILL=0.94` waits until the last query sees at least 94%
of the padded capacity. For a 250K prompt, MIN_KV=131072 captures the ratio-2
131072-key and ratio-1 262144-key forms near the last prefill batch.

Run `test_ds41_indexer --cuda --bench --replay /directory/*.idx`. For every captured query
it compares the legacy-mask vec output with the nvis vec output bit for bit,
then compares WMMA scores and actual GPU top-512 sets to vec. It reports changed
rows/ids, score errors, exact-fp16 Q conversion, K scale range and fallback
conditions. `--bench` also times 10 repetitions of just the indexer graph for
legacy-mask vec, nvis vec and nvis WMMA (launch and synchronisation included).
The synthetic cases also cover non-monotonic/empty visibility,
partial blocks, strided/broadcast nvis, multiple streams, nonrepresentable Q,
fp16 subnormals, out-of-range K scales, and zero blocks.
