# Smaller DSpark expert weights, measured by acceptance (v41-dspark-shrink)

The DSpark sidecar `/media/mal/NVME1TB/Xolotl/deepseek-v4.1/sidecar/V41-MTP-Vision.gguf` is 8.42 GB, of which the nine routed-expert
tensors `mtp.{0,1,2}.ffn.experts.{w1,w2,w3}.weight` ([in,ff,128] each, 802 MB, MXFP4) are 7.22 GB.  A drafter's precision
cannot change output quality - every draft is verified by the trunk - so it can be compressed hard, and only the
acceptance rate moves.  This measures that trade.

## How

`tools/v41/v41_sidecar_requant.cpp` (built as the cmake target `v41_sidecar_requant`, appended to `v41_engine.cmake`)
reads the sidecar through ggml's gguf API, dequantizes the nine expert tensors MXFP4 -> F32 with ggml's own
`to_float` trait, requantizes them with `ggml_quantize_chunk`, and writes a fresh GGUF that leaves every other tensor
byte-identical.  The engine also grew an equivalent on-load path (`STRATA_DSPARK_EXP_TYPE=<ggml type name>`,
`ds41_dspark.cpp`), which avoids writing files; both were used and agree (below).

The gate is the offline harness the D1 slice shipped:
`flock ~/.quetza-data/conductor/ds4-gpu.lock build-ds4-dspark/test_ds41_dspark --sidecar <f> -m "<V41-GSQ>" --main
~/AI/v41-dev/dspark/main_exact.bin --gen ~/AI/v41-dev/dspark/gen_exact.txt --max 378`.  All 378 scored steps fit the
records (511).  The gate is deterministic: two baseline runs printed the identical 3.280.

## Results (378 steps, RTX 4090)

| arm | bpw (experts) | file | card weights | accepted/pass | tokens/pass | 1st draft | ms/draft |
|---|---|---|---|---|---|---|---|
| MXFP4 (baseline) | 4.25 | 8.42 GB | 8.61 GB | **3.280** | 4.280 | 88.6% | 3.60 |
| Q3_K (env) | 3.4375 | - | 7.23 GB | 3.246 (-0.034) | 4.246 | 88.6% | 3.56 |
| Q3_K (file) | 3.4375 | 7.04 GB | 7.23 GB | 3.246 (-0.034) | 4.246 | 88.6% | 3.65 |
| Q2_K (file) | 2.625 | 5.66 GB | 5.85 GB | 3.233 (-0.047) | 4.233 | 88.1% | 3.21 |
| IQ3_XXS (env) | 3.0625 | - | 6.59 GB | 3.280 (-0.000) | 4.280 | 89.2% | 3.19 |

"card weights" is the stack's `weight_bytes()` (its 8.61 GB baseline includes the 0.7 GB duplicated trunk head; a
shared-head path would trim all rows by ~0.7 GB).  File = on-disk sidecar.  The Q3_K row was produced both ways and
matched exactly (same mean, same accepted-length distribution: 0:11.4 1:12.7 2:12.4 3:9.8 4:10.8 5:42.9), which
validates both quantizers and the tool's byte layout.

Numeric check of the files (rel RMS of the requantized experts vs the source MXFP4, 3 experts x 9): Q3_K 0.155,
Q2_K 0.449 with element correlation 0.989 against MXFP4 - i.e. the 15%/45% is the difference between two lossy 4-bit /
2-bit formats, not the error of the new format (Q3_K reconstructs the underlying weights more faithfully than MXFP4).
IQ2_XXS / IQ2_XS were not attempted: `ggml_quantize_requires_imatrix` rejects them (no imatrix exists for these
tensors), so IQ3_XXS (3.06 bpw) is the smallest IQ-class type ggml will quantize here.  IQ4_XS is the same 4.25 bpw as
MXFP4 and frees nothing.

## Recommendation

**All three shrink for ≤ 0.05 accepted/pass (well inside the ~0.1 bar), so the choice is VRAM, not quality.**
The best value is **IQ3_XXS**: acceptance is *identical* to MXFP4 (3.280, -0.000) while the on-card stack drops
8.61 -> 6.59 GB, **freeing 2.02 GB**.  If the absolute smallest is wanted, **Q2_K** costs only 0.047 accepted/pass
(-1.4%) and frees **2.76 GB** (5.85 GB); Q3_K sits between (1.38 GB freed) and is the weakest of the three, so no
reason to pick it.  Recommended default: `STRATA_DSPARK_EXP_TYPE=iq3_xxs` (or the Q2_K sidecar where disk/VRAM is
tighter).  All freed space goes back to the expert tier, which is the decode-critical path at 50 tok/s.

## Residuals

- IQ3_XXS and Q3_K(-env) were measured via `STRATA_DSPARK_EXP_TYPE`, not from files; only Q3_K and Q2_K sidecars were
  written (disk is at 98%).  `V41-MTP-Vision.IQ3_XXS.gguf` was deliberately not produced - use the env switch.
- `mtp.2.confidence_head.proj.weight` is re-emitted with its declared rank collapsed (1,5376) -> (5376,); the data is
  byte-identical and ggml treats the shapes as equal, but it is the one metadata difference from the source sidecar.
- Optional Q4_K sanity point not run (4.5 bpw > MXFP4's 4.25 - it would grow the stack, no upside).
