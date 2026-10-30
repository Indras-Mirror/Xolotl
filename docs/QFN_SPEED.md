# Qwen3.8-Flash-Next (xol-qfn): what actually makes it fast

> **To check (2026-10-31): row-interleaved CPU expert kernels.** Check the CPU experts (PTQ1_0, Q2_0) for a row-interleaved 1-token kernel next to rows3, which groups tokens but not rows. See [CPU_EXPERT_TILING.md](CPU_EXPERT_TILING.md).

Measured 2026-10-29 on Linux, RTX 4090 24 GB, Ryzen 7 5700X, 90 GB DDR4-3200, model on a
3.5 GB/s NVMe. Model: the abliterated XQ mixed-tier build, **78.83 GiB**, arch `qwen4exp`.

This note exists because the same mistake was made four times in one afternoon: adding a flag that
looks like it should help, measuring it, and finding it cost more than it gave. The numbers below
are the evidence, so the next person can skip the search.

## The result

16,383-token prompt, 1,024 generated tokens, greedy:

| metric | spec 4 (winner) | spec 6 |
|---|---|---|
| **decode** | **152.58 tok/s** | 120.99 tok/s |
| **prefill** | **4,284.77 tok/s** | 4,167.19 tok/s |
| speculative acceptance | **0.876** | 0.660 |
| tokens per round | **3.30** | 2.38 |
| CPU experts distinct per layer | **1.03** | 0.69 |
| pool per round | 5.48 ms | 3.76 ms |

Spec 6 accepts *fewer* drafts per window and lands net slower, even though it leaves less CPU
work per round. **`--spec 4` is the right value for this model.** Do not raise it.

## The config

```sh
STRATA_ROUTE_RESIDENT=0.8 STRATA_ROUTE_RESIDENT_RANKS=5-9 \
  strata --native xol-qfn.gguf --pack pack-xol-qfn --ple-gguf xol-qfn.gguf \
    --ple-io direct \
    --expert-profile data/expert-profile.bin \
    --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 \
    --mtp <mtp-dir> \
    --max-context 32768 --kv int8 --kv-resident 32768 \
    --max-new 1024 --tokens-file ids.txt
```

A ready serve config lives at `strata-xol-qfn.json` (port 8099, model name `xol-qfn`).
`serve/server.py`'s `child_env()` exports a config's `env` block to the engine, so the
`STRATA_ROUTE_RESIDENT*` variables travel with the config.

## What each choice bought, and what it cost

| change | measured effect |
|---|---|
| **drop `--resident-experts`** | TTFT **51.4 s → 5.14 s**. It page-locks the expert complement at load (34 GiB copied); the default arena is faster to start and no slower to run. |
| **drop `--mmap-experts`** | That flag opts *out* of the resident arena back to `MapViewOfFile`; with it, decode reads ~1.84 GB of experts from disk per round. |
| **`--kv-resident 32768`** | Frees VRAM for expert slots: CPU experts/layer **2.33 → 1.03**. |
| **`--expert-cache auto`** (not a fixed count) | Lets the card decide; an explicit 12000 was not reached and a small explicit value caps the cache. |
| **`--spec 4 --spec-min-p 0.5`** | Acceptance **0.876** vs 0.359 at spec 6 with a different draft vocabulary. |
| **drop `--mtp-q4` and `--mtp-draft-vocab en`** | The full-precision, full-vocab drafter accepted *better* than the 4-bit/English-only ones. Shrinking the drafter to buy expert slots is a losing trade here. |
| **drop `--lookup-chain`** | Prompt-lookup drafting did not help on this workload. |
| **generate 1,024 tokens, not 64** | Short runs understate steady-state decode badly (64-token runs read ~65 tok/s where 1,024-token runs read 152). |

**`STRATA_ROUTE_RESIDENT` is the one lever that mattered**, and it was already in the engine
(`src/core/verify.cpp`). At margin 0.8, ranks 5-9 it swapped 99.0% of tail routing decisions back
onto resident experts (`nonres_entries before=89686 after=25744`), taking CPU experts per layer
from 2.33 to 1.03. It only re-orders ranks 5-9, so the top-4 routing — and therefore the output —
is untouched.

## Things that are NOT the bottleneck

* **The PLE n-gram table (26.82 GiB, 34% of the file).** Read with `--ple-io direct`; the engine
  logs `PLE 0.1 ms` per prefill. It never occupies VRAM. Lowering its precision is not a speed lever.
* **Model size.** The previous best on this box was 151.0 tok/s on a 71.59 GiB model; this build is
  78.83 GiB and reaches 152.58. A 10% larger file did not cost speed.
* **Disk.** In the winning arm the expert tiers report **0 blob reads from the file**.

## Related

* `docs/DETAILS.md` — the engine's own knobs.
* `tools/qwen38/ARCHITECTURE.md` — the `qwen4exp` architecture and the tier ladder.
* Basalt (`jesdga95/basalt`) requires Blackwell (sm_120) + CUDA 13 and cannot run on Ada (sm_89);
  its packs are a single `.basalt` file this engine does not read. Not a route on a 4090.

## Max context

`--max-context 262144` loads (the model's trained length) with `--kv int8`:

| config | ctx | expert slots | decode @16K | prefill |
|---|---|---|---|---|
| 32768, full KV in VRAM | 32K | 7779 | 152.58 | 4,285 |
| 262144, `--kv-resident 262144` | 256K | **6152** | 135.4 | 3,233 |
| **262144, `--kv-resident 32768`** | **256K** | **7566** | — | — |

Max context alone costs ~11% decode and ~25% prefill because the int8 K/V takes VRAM from the
expert cache. Keeping `--kv-resident` at 32K restores almost all the slots: only the first 32K
cells sit in VRAM and the rest stream from pinned RAM, which costs nothing when a conversation is
shorter than that. **That is the config to run.**

## Real-work throughput

On the Gauntlet (7 agentic coding tasks, 43 tool-calling steps, thinking at xhigh) decode measured
a median of **117 tok/s**, mean 114, max 151. The 152.58 above is a pure 1,024-token generation;
thinking and tool calls land in the same range.
