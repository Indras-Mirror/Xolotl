# Will it run on my machine? (DeepSeek-V4.1)

[Back to the README](../README.md)

What I've measured is one setup: **RTX 4090 (24 GB), 110 GB DDR4 (90 GB until 1 Nov), Linux, NVMe**. Beyond that:

- **OS:** Linux, or Windows 11 natively (MSVC + CUDA; build and serve steps in [WINDOWS.md](WINDOWS.md)).
  Everything measured here is on Linux.
- **GPU:** NVIDIA, compute capability 7.5 or newer (RTX 20/30/40/50). Build with your card's architecture:
  `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 30, `89` for RTX 40, `120` for RTX 50 (use CUDA 13.x for RTX 50; 12.8 builds
  had prompt-path crashes). **Tesla V100 (sm_70):** experimental DS41-on-Buster path only — see [BUSTER.md](../BUSTER.md)
  (`STRATA_EXPERIMENTAL_SM60=ON`, CUDA 12.x, no chunk-MMQ on Volta).
- **VRAM:** 24 GB is what I run. Smaller cards work with smaller prompt batches (slower prefill) and fewer experts on
  the card: community runs on 12 GB cards (RTX 3060 with 46 GB RAM: 149 tok/s prefill at 8K, 8-10 tok/s decode; RTX
  5070 with 64 GB: 4.7 tok/s) are in the README's smaller-cards table and [DEEPSEEK41.md](DEEPSEEK41.md).
- **RAM:** 110 GB here, with an 80 GB expert cache in RAM (64 GB on the 90 GB box). Less RAM means a smaller cache, so more experts come off the
  NVMe each token and decode is slower. More RAM helps: around 256 GB holds every expert.
- **Disks:** about 152 GiB for the quant plus ~200 GB for the Engram tables, and ~40 GB more for the optional fast
  mode. All of it wants a fast NVMe on ext4 or NTFS.
  **Not exFAT** (Linux's exFAT driver serialises parallel reads; expert reads were 5x slower here) and not a hard
  drive (the Engram lookups are ~48 small random reads per token).

## Community runs on smaller cards

The engine fits the work to the card, and the experts that don't fit live in RAM and on the SSD, so system RAM
matters as much as VRAM.

| Card           | RAM   | Model                     |                               Prefill |      Decode | Run                                                                                      |
| -------------- | ----- | ------------------------- | ------------------------------------: | ----------: | ---------------------------------------------------------------------------------------- |
| RTX 3060 12 GB | 64 GB | Qwen3.8-Flash-Next Q2_0   | 882 tok/s (4K), 981 (32K), 934 (128K) | 39-45 tok/s | [2026-10-09](../bench/results/2026-10-09-community-rtx-3060-0141/README.md)              |
| RTX 4070 12 GB | 32 GB | Qwen3.8-Flash-Next IQ2_XS |                                     - |    48 tok/s | [2026-10-08](../bench/results/2026-10-08-community-rtx4070-iq2-xs/README.md) (Hcl192088) |
| RTX 3060 12 GB | 46 GB | DeepSeek-V4.1-Flash       |                        149 tok/s (8K) |  8-10 tok/s | [DEEPSEEK41.md](DEEPSEEK41.md)                                                           |
| RTX 5070 12 GB | 64 GB | DeepSeek-V4.1-Flash       |                  121-231 tok/s (4-8K) |   4.7 tok/s | [DEEPSEEK41.md](DEEPSEEK41.md)                                                           |

More cards, AMD and Intel included: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
