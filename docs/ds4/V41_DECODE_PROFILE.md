# V4.1 decode profile (Windows, RTX 5090)

Where a decode token's time goes, and the diagnostics used to find out. Measured on 2026-10-21: RTX 5090 (and the
RTX 4080 helper, `STRATA_DS41_REMOTE`, where stated), 128 GB DDR5-4000, Windows 11, the 1017 XQ v2 GGUF, CUDA 13.0
(sm_89 + sm_120a), dense re-quantized to Q6_K, route bias 0.5 / 0.3, 72 GiB arena. The engine was a build from that
day, before main cached the verify-width pass graphs by default (`DS41_PASS_GRAPHS`); the DSpark trace below ran with
an equivalent cache on.

## One token, no DSpark

One-shot, 200-token prompt, 192 tokens decoded, 97% of expert lookups in VRAM. `STRATA_DEC_PROF=1` stamps the host
clock; `STRATA_DEC_PROF=2` synchronizes the device before each stamp, so each stage holds its own GPU work.

| ms per token | `STRATA_DEC_PROF=1` | `=2` (synced) |
| --- | ---: | ---: |
| begin (Engram read ~1.1) | 1.4 | 1.4 |
| predict + the prefetch copies | 0.5 | 11.9 |
| attention + router (40 graphs, then the ids to the host) | 18.9 | 12.7 |
| expert tier | 14.3 | 13.8 |
| finish (40 graphs) | 3.1 | 2.9 |
| decode | 26.8 tok/s | 23.5 tok/s |

The prefetch copies normally run under the attention; synced, they show up on their own line.

The dense half reads about 6.3 GiB of weights per token (sum of its matrix products, `DS41_GRAPH_OPS=1`), about
4.4 ms at the 5090's bandwidth, but takes 15.6 ms of GPU time. It runs about 3,550 kernels per token: 40 attention
graphs of ~77 kernels (CPY 11, RMS_NORM 6, CONT 5, CONCAT 5, MUL 4.5, GET_ROWS 4, 8 Q6_K matrix products, ...) and
40 finish graphs of ~12. Almost all one-token graphs already replay as CUDA graphs (a temporary counter in ggml:
3,859-4,082 of 4,096 computes replayed), so the remaining ~11 ms is the kernels themselves, about 3 us each.

## Kernel trace of DSpark decode

Nsight Systems 2024.5 produced no report on this Windows install (CUDA built without the profiler API, so
`STRATA_PROFILE_DECODE`'s capture range never opens). `tools/v41/prof/cupti_trace.cpp` is a small
`CUDA_INJECTION64_PATH` tracer instead: it writes every kernel and copy with its GPU start and end to a CSV, and
`tools/v41/prof/trace_summary.py <trace.csv> <seconds>` sums the last N seconds (GPU busy vs idle, kernels by name,
copies by size, gaps on the dense stream).

One-shot, the 5090 alone, depth 4, 61 verify passes (tracing slows a pass from 56.7 to 63.4 ms):

| decode window, 3.8 s | |
| --- | --- |
| GPU busy with kernels | 710 ms (19%) |
| sum of kernel times | 758 ms, ~12.4 ms per pass; kernels under 3 us: 145 ms in total |
| copies | 40,055, 36.7 GB; expert blobs (~9.8 MB) to the card: 3,394, 33 GB, at ~27.8 GB/s |
| gaps over 50 us on the dense stream | 3,061 ms: between a layer's attention and finish graphs (the tier) 2,296 ms, between layers 455 ms |

So the card mostly waits: on expert copies (the misses, plus prefetches of which 1,046 of 1,675 were not used) and
on the host between layers. Pinned host-to-device copies measure 34.2 GB/s on this RTX 5090 (PCIe 5.0 x16, DDR5-4000
dual channel); the RTX 4080 over OCuLink 6.6 GB/s.

Prefetch off (`STRATA_D3_PF=0`): one-shot (cold cache) 49.3 / 52.8 -> 55.1 / 58.0 tok/s, but served with the helper
no change (decode 42.1 / 43.3 vs 43.5 / 42.6 tok/s, agent decode 42.4 / 39.4 vs 40.2 / 40.0), so the default stays.

Tried, no gain, not kept: a verify pass's tier call ending on a CUDA event the next dense graph waits on (ggml's
`ggml_backend_event_wait`) instead of a host sync of the tier's stream. Same text; one-shot 52.8 / 53.0 vs 52.3 / 55.6
tok/s; served together with prefetch off, decode 44.0 / 45.0 vs 44.2 / 43.9 tok/s, agent decode 40.3 / 38.8 vs
41.4 / 39.1. Each layer still waits on the host for the routed ids, so the card idles once per layer either way.

## Using the tracer (Windows)

CUPTI comes from the `nvidia-cuda-cupti` wheel (unzip it; it has headers and the DLL but no import library, so the
tracer loads the DLL at run time). From a VS x64 developer prompt:

```
cl /nologo /O2 /EHsc /LD /I<wheel>\nvidia\cu13\include /I<CUDA>\include tools\v41\prof\cupti_trace.cpp /Fe:cupti_trace.dll
set CUDA_INJECTION64_PATH=<full path>\cupti_trace.dll
set CUPTI_DLL=<wheel>\nvidia\cu13\bin\x86_64\cupti64_<version>.dll
set CUPTI_TRACE_OUT=<full path>\trace.csv
```

Then start the engine or the server as usual, and run `python tools\v41\prof\trace_summary.py trace.csv 4` after it.

## Settings

| Variable | What it does |
| --- | --- |
| `STRATA_DEC_PROF=2` | the decode and verify phase timers with a device sync before each stamp (diagnostic, slower) |
| `DS41_GRAPH_OPS=1` | the op mix and matrix-product weight MiB of the first 200 cached graphs (diagnostic) |

Unset, neither changes anything: `STRATA_DEC_PROF=1` keeps its host-clock stamps, and the one-shot `pass_v split`
reads the same clock as before.

## Not done

- The dense half's ~3,550 small kernels per token: fusing them (hc mixes, norms, copies, router) is the next step.
- The tier's two host round trips per layer (the routed ids come back to the host; the call ends with a stream sync).
