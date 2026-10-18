# cenote benchmarks

One machine, measured 2026-10-18:
- RTX 3090 24 GB (PCIe 4.0 x16)
- Ryzen 7 5800X (8 cores / 16 threads)
- 128 GB DDR4-3200 (4 x 32 GB, dual channel)
- Samsung 990 EVO Plus NVMe
- Windows 11 Pro, NVIDIA driver 616.92, CUDA 12.8 build

Model: [IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
(September checkpoint, 163 GB), plus the official Engram shards 47/48 of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), read in place.

## Method

1. Start a fresh server for each variant (`tools/v41/serve/serve_ds41.py` with `strata-ds41-win.json` plus that variant's change).
2. Warm up once with the same 200-token prose answer.
3. Time three prompts: a 350-word insurance explainer (prose), an LRU cache class with asserts (code), and 12 rows of CRM JSON (agent). Each runs 3 times: greedy, thinking off, 400 tokens.
4. The number is `timings.predicted_per_second` from the API; the table shows the median of the 3 runs.

All runs were on a quiet disk. An earlier round ran during a 200 GB download to the same SSD and was discarded as noise. Every variant keeps the upstream decode settings: `--route-bias 0.5 --route-bias-file 0.3 --arena-adapt --kv-fp4`, DSpark Q2_K draft at depth 4, dense weights re-quantized to q6_K. Every server start begins with an empty VRAM expert cache.

## Results (median tok/s)

| Variant | Prose | Code | Agent | Average |
| --- | ---: | ---: | ---: | ---: |
| 64 GiB arena (Xolotl's served config), run first | 13.0 | 17.4 | 15.7 | 15.4 |
| 64 GiB arena, run last (drift check) | 13.0 | 17.8 | 16.0 | 15.6 |
| 72 GiB arena | 14.3 | 20.7 | 18.4 | 17.8 |
| 80 GiB arena | 14.4 | 20.2 | 17.7 | 17.4 |
| 72 GiB, DSpark off | 16.5 | 19.1 | 18.0 | 17.9 |
| 72 GiB + `STRATA_TIER_BATCH_PCIE=0.75` | 14.4 | 22.0 | 18.0 | 18.1 |
| same, repeated (round 3) | 14.7 | 22.1 | 18.8 | **18.5** |
| same + `STRATA_DSPARK_CONF=0.75` | 14.5 | 21.3 | 18.2 | 18.0 |
| same + `STRATA_DSPARK_CONF=0.85` | 14.4 | 21.4 | 18.7 | 18.2 |

Raw results: [data/round2.jsonl](data/round2.jsonl) and [data/round3.jsonl](data/round3.jsonl). Each line has every run, the DSpark tokens per pass and the expert-tier counts.

- **Adopted:** the 72 GiB arena and PCIe share 0.75. That is +19% on average and code goes 17.4 -> 22.1. About 25 GB of RAM stays free.
- **DSpark:** it helps code (3.5-3.9 tokens per pass) and costs prose (about 1.7 tokens per pass). Raising its confidence threshold did not recover prose, because the draft itself costs time every step.

## Other measurements

- **Long prompt.** A fresh 13,294-token prompt took 63.45 s (209.5 tok/s). The next turn of 21 tokens on the 13,572 cached took 7.7 s total. Decode right after the long prompt ran at 15.1 tok/s.
- **Host-to-GPU copies.**
  - Pinned 4 KB-page memory copies at 25.9-26.0 GB/s, `cudaMallocHost` at 26.0-26.1 GB/s, and pageable memory at 11.8-11.9 GB/s (1 GiB buffer, 64 MiB and 9 MiB copies).
  - Large pages do not matter for these copies on Windows. On Linux, [dsx](https://github.com/lxe/dsx) measured 13 vs 24.7 GB/s (IOMMU).
- **Where a decode token goes.**
  - Measured with `STRATA_DEC_PROF=1` on the one-shot engine path, on repetitive synthetic output at 98% VRAM hits (so faster than real text).
  - Attention + router: 23.4-24.4 ms. Expert tier: 19.4-21.4 ms. Finish: 3.9-4.3 ms.
  - With `GGML_CUDA_DISABLE_GRAPHS=1`, attention + router is 47.4 ms. CUDA graphs halve that step on Windows, so kernel-launch overhead is the next thing to attack in the expert tier.

## Caveats

- One machine, greedy short-context tests. Speeds depend on the text (DSpark acceptance) and on how warm the expert cache is.
- `--route-bias` trades routing exactness for speed in decode. Xolotl's notes put it at about 4.6 points of API top-1. Prompts route exactly.
- Not a quality benchmark. Quality numbers for the quant are on its model card.
