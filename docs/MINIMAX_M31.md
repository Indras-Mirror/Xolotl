# MiniMax-M3.1-Flash

This page documents the opt-in MiniMax-M3.1 engine in `tools/m31` and its shim
to Strata's common HTTP server, developed in [issue #62](https://github.com/Indras-Mirror/Xolotl/issues/62).
Feature-specific build options, numerical contracts and measured results are described below.
DSpark is outside this contribution series.

The tested model is the unmodified Unsloth UD-IQ3_S, revision
[`02c81e41305bf8b718ad2e288f4e52a1cd49f55f`](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF/tree/02c81e41305bf8b718ad2e288f4e52a1cd49f55f).
Keep all five shards together and pass the first shard to `--model`.
It has 428 billion parameters, about 23 billion active parameters, 60 layers,
and MSA in every layer. The files occupy 163.4 GiB; routed experts occupy
152.5 GiB in IQ3_XXS, IQ4_XS and NVFP4. A later model revision may have different weights.
Other quants have not passed this port's acceptance checks.

## Build

From the repository root, with a C++20 compiler and CMake 3.24 or newer.
Install the root `requirements.txt` into your Python environment before running
the CPU tests or HTTP server.

```sh
cmake -S . -B build-m31-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=OFF
cmake --build build-m31-cpu --target m31_generate m31_checks -j 8
ctest --test-dir build-m31-cpu --output-on-failure -R m31

cmake -S . -B build-m31 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build build-m31 --target m31_generate m31_checks -j 8
ctest --test-dir build-m31 --output-on-failure -R m31
```

`120a` is the tested RTX 5090 target. Choose your GPU's architecture when building
elsewhere. Both CUDA options are required: `STRATA_ENABLE_CUDA` configures
Strata's CUDA targets, and `STRATA_GGML_CUDA` enables the backend M31 uses.
The port uses the tree's ggml dependency; `STRATA_GGML_DIR` is supported
for an existing dependency checkout. It does not link the llama.cpp test oracle
into the engine. Python server dependencies are the common server's dependencies
in the root `requirements.txt`.

`STRATA_M31` defaults to `OFF`. Its CMake include returns before adding targets,
flags or dependencies when disabled. This keeps M31 out of the normal install.
The measured default-build comparison against base `41c46c2f` is described below.
Its coverage is limited to the named configurations and artifacts.

The measurement host needed local CUDA/glibc compiler compatibility flags
(`-U_GNU_SOURCE -D_DEFAULT_SOURCE` and an external declaration include). They
were applied equally to comparison builds and were not added to this port.
The commands above assume a compatible compiler, toolkit and system C library;
they do not claim that every recent glibc/toolkit combination works unchanged.

The historical v8 build used GCC 14 and CUDA 13.1 on Linux. HIP, SYCL, Windows,
other GPUs and two-GPU placement have not been validated for this port. A CPU
build is a useful correctness path; its outputs are not the numeric oracle for
the accepted CUDA arithmetic. No claim of HIP or SYCL build coverage is made here.

## Default-build comparison

The vision and short-prefill follow-ups independently repeated the OFF comparison
against upstream `4499fcbc` on 2026-10-28. CPU and CUDA target lists, normalized
compile commands and the named raw artifact hashes matched on each branch.
The feature sections identify their receipts. The deterministic NVCC metadata
controls and Qwen executable limitation below still apply. This is build
isolation evidence; it does not claim other-model inference coverage.

The following comparison is the original PR1 evidence:

With GCC 14.3, CUDA 13.1.115 and `sm_120a`, base `41c46c2f` with M31 OFF,
the candidate with M31 OFF, and a fresh candidate build with M31 ON produced
identical **full raw SHA-256** values for five existing artifacts:
`glm_generate`, `ds41_generate`, `libstrata_engine.a`, `libstrata_mmq.a`, and
`CMakeFiles/strata.dir/src/program/generate.cpp.o`. All 417 existing compile
commands matched after substituting source/build paths. The OFF target list was
identical; enabling M31 removed no existing target. A separate CPU OFF comparison
matched `libstrata_kernels_cpu.a`, `ds4_ref` and `glm_generate_cpu`, with the same
target list. Artifact hashes and exact feature selections are in
[`default-build-isolation.json`](../tools/m31/bench/receipts/default-build-isolation.json).

The CUDA comparison used stable private source/build paths and compiler working
directory, a distinct 31-bit decimal `--frandom-seed` per CUDA source,
`--objdir-as-tempdir` and `CCACHE_DISABLE=1`. These controls were applied only to
comparison caches. Binary payloads were not normalized or rewritten, and the
measured M31 worker and vendor sources were unchanged. Ordinary unseeded NVCC
builds are not claimed to produce deterministic whole-file hashes.

The root Qwen `strata` executable has the same pre-existing
`new_pool_for_device(ctx,int,int)` link failure in this base/vendor combination.
Its entry-point object was compared; a successful executable link or Qwen model
run is not claimed. The comparison disabled `STRATA_DENSE`, `STRATA_DSV41_FQ_OP`,
`STRATA_DSV41_HC_MIX` and upstream tests, as recorded in the receipt. It does not
cover every optional feature, other-model inference, HIP, SYCL or Windows.
Those limits remain part of the review coverage.

## Run through the common server

Choose paths for your machine:

```sh
MODEL=/path/to/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf
TOKENIZER_DIR="$PWD/m31-tokenizer"
python tools/m31/serve/export_tokenizer.py "$MODEL" "$TOKENIZER_DIR"
```

Create an engine configuration with absolute paths. `exe` names the CMake-built
`m31_generate`. `args` supplies the model and resource budgets; the common server
adds `--serve` itself.

```json
{
  "exe": "/path/to/Xolotl/build-m31/m31_generate",
  "args": ["--model", "/path/to/model-00001-of-00005.gguf", "--backend", "cuda",
           "--ctx", "40000", "--chunk", "32768", "--threads", "80",
           "--arena-gib", "170", "--gpu-experts-gib", "7",
           "--register-arena", "1", "--register-limit-gib", "176",
           "--gpu-pack-threads", "8"],
  "tokenizer": "/path/to/Xolotl/m31-tokenizer",
  "model_name": "minimax-m31"
}
```

The example budgets describe the measured EPYC/5090 machine. They are not suitable
for every PC. A 163 GiB model needs substantial free system RAM. Reduce threads
and resource budgets for your hardware. Zero arena/cache budgets choose the
CUDA memory policy after model geometry and available memory are known; explicit
budgets reproduce the measured profile. CPU zero arena keeps mapped weights.
Registered host memory is a separate explicit budget, rather than an implicit
request to pin the whole model. See `m31_generate --help` for the current resource
options. The accepted defaults need no `M31_*` variables.
Unsupported combinations fail at startup. CUDA fusion is disabled inside the
engine subprocess before backend initialization; explicitly enabling fusion is
refused because that arithmetic has not passed the accepted gates.
`GGML_CUDA_MMQ_PREC` defaults to `auto`; known overrides `q4` and `q8` are
recorded in the effective configuration, and unknown values fail at startup.
Changing that precision requires a new numerical gate. The measurements here
use the default. A failed CPU graph allocation releases partial ownership so a
later request can rebuild it after reset.

```sh
env -u CUDA_DEVICE_MAX_CONNECTIONS -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/m31/serve/serve_m31.py --engine strata --config m31.json \
  --tokenizer "$TOKENIZER_DIR" --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m31","messages":[{"role":"user","content":"Explain sparse attention."}],"max_tokens":256}'
```

OpenAI and Anthropic endpoints, SSE, the request queue and health checks come from
`serve/server.py`. M31 supplies its tokenizer, GGUF chat template and tool parser.
The exported model template uses adaptive thinking. `reasoning_effort=none`
hides reasoning from the response; it does not guarantee that the model skips
internal reasoning. Streaming with visible reasoning can therefore produce its
first delta earlier than streaming with reasoning hidden.
Generation is serialized to one active sequence. The engine uses the common
`GEN`/`BGEN`/`STOP`/`QUIT` line protocol. Stdout is reserved for that protocol.
Prefix reuse reports the number actually reused with `RESUME`; cancellation is
checked at prefill boundaries and decode steps. The controller keeps stdin open
while serving. Closing stdin cancels the current request at the next boundary
and exits; a one-line `echo GEN ... | engine --serve` closes that connection
immediately, so use the common server or an interactive controller for serving.
`cache_prompt=false` disables both reuse and storage for the request. M31 fits the requested output length into
the remaining context, including the common token-limit aliases.
For chat requests, an explicit `0` in `n_predict`, `max_completion_tokens`, or
`max_tokens` returns zero completion tokens without running inference. Omitted
limits and `-1` use the context left; positive limits are clipped to that space.
`n_predict` takes precedence, then `max_completion_tokens`, then `max_tokens`.
Boolean, fractional, and less-than-`-1` limits are rejected. The Responses API
keeps the common server's positive-or-omitted `max_output_tokens` contract.

## Numerical contract

The reference is llama.cpp PR [#31192](https://github.com/ggml-org/llama.cpp/pull/31192),
commit `6b95a9908d2c7b242817a447708aec38e937e2c0`, with GPU op offload,
F16 K/V and CUDA fusion disabled. Long CUDA batch prefill uses stock grouped MMQ; short chunks use the policy
described below. Decode uses
fused CPU experts with GPU dense operations. Rank and reduction order are part of
the measured recipe.

The gate is aggregate top-1 agreement at least 99%, mean KL at most 0.001 and
maximum position KL at most 0.05. Byte identity is also reported separately.
The historical GPU-offload oracle and CPU oracle agreed on only about 78% top-1,
with mean KL about 0.78. MSA block selection appears sensitive to rounding near
selection boundaries. That observation is why the CUDA path is compared with the
GPU-offload oracle and also checked on GSM8K and tool calls. It is not evidence
that a CPU oracle is interchangeable with that placement.

Gate20 covers every row of 20 RU/EN/code/math prompts: 28,723 positions, lengths
64–3,000. Long tests at 16K and 32K cover the final 256 rows per prompt, including
32 teacher-forced decode steps. They do not prove equality for every long-context
row. The bounded head must run after the full final-layer FFN, including rows
whose logits are discarded. Both engines must use the same actual chunk geometry.

The oracle can require a smaller physical KV allocation than the candidate to
leave compute space. This is allowed only with an explicit capacity-equivalence
trace: contiguous positions starting at zero, no shift or wrap, identical active
256-key padding, trained RoPE context 1,048,576, frequency scale 1 and extension
factor 0. The replay script validates those conditions.

## Adaptive short-chunk prefill

Inside the already opt-in M31 engine, CUDA now selects the accepted path from
the number of **new rows in each actual chunk**. No environment variables or
additional mode switches are needed:

| New rows | Routed experts | Dense/attention |
| --- | --- | --- |
| 1 | CPU fused decode with bounded row scheduling | Existing GPU decode |
| 2–31 | Existing CPU path | GPU |
| 32–1024 | Complete experts split between CPU and GPU | GPU |
| Above 1024 | Existing native GPU batch | GPU |

CPU-only builds retain their previous prefill routing. Fused single-token decode
uses bounded row scheduling in both CPU-only and CUDA builds. Hybrid layers currently
support IQ3_XXS and IQ4_XS. A layer containing NVFP4 uses the existing native
GPU path; it does not approximate NVFP4 on the CPU. This covers the tested
UD-IQ3_S model's heterogeneous expert formats.

The split is calculated per layer from expert weight bytes, route assignments
and actual GPU-cache residency. A resident expert stays on the GPU. Cold experts
move as complete units to balance estimated CPU and GPU costs. The fixed rates
(220 GB/s CPU memory, 6000 GOP/s CPU compute, 57 GB/s PCIe and 60000 GOP/s GPU
compute) are calibration parameters from the EPYC 9654 / RTX 5090 experiments;
they are not measured peak bandwidth claims or a hardware autotuner.

Hybrid CPU expert arithmetic consumes the native CUDA activation quantization
and OAI results. It preserves integer dot32/FMA and original-rank reduction order.
Prepared corrections and CPU vector dispatch retain that arithmetic. Compact
GPU subsets of 1–8 routes are padded to nine private rows so ggml keeps MMQ;
padding outputs never enter the original rank reduction. A persistent CPU
submission thread overlaps CPU work with bounded GPU weight uploads. Existing
slot leases stay live until the real GPU consumer has finished.

Activation helper buffers are reserved at initialization for at most 1024 rows,
before the planner samples free VRAM. The long-context planner therefore sees
those retained allocations. Single-token dot arithmetic and rank-reduction order
are unchanged; fused CPU matrix-row scheduling now uses bounded dynamic batches,
as described below. The long batch algorithm is unchanged, although the
available-memory budget can alter long chunk sizes.
The same-chunk numerical contract still applies. Report actual chunks when
comparing two builds; a different partition is a different numerical history.

Prefix reuse, rewind and cancellation retain the PR1 protocol. For example,
a cached 1024-row prefix followed by 100 new rows selects the 100-row hybrid
path, independently of the full 1124-row conversation length. A 300-row suffix
likewise selects hybrid. The prefix's earlier batch history remains in KV;
these requests must be compared with the corresponding live-history reference,
not an unrelated cold 1124/1324-row forward.

### Original short-prefill numerical checks

The original PR3 CUDA worker measured on October 28 had SHA-256
`5a7d75db17e0e7fecff61a7d3a488a5eaf13872cb529eb0fc5fe20e36dfd3f2a`;
the separately built `4499fcbc` baseline is
`a2f47f967e01ac8c1ed3d92a241bed1cf527ab781e32787fb88ba6405a97e707`.
On 2026-10-28, both used one RTX 5090 32 GB, EPYC 9654 (96 cores / 192 threads),
624 GiB RAM, PCIe 5.0 x16, GCC 14.3 and CUDA 13.1.115 / sm_120a. GPU1's
vLLM-omni, ffmpeg and the neighbouring VM remained running.
The host ran Ubuntu 26.04.1 LTS, Linux 7.0.0-34-generic and NVIDIA driver 595.91.07.
The speed-window hardware capture reported a 440 W GPU0 power limit (575 W stock).

All seven one-chunk boundaries (1, 2, 31, 32, 128, 1024, 1025 rows; **2,243 rows**
total) matched both the baseline and the pinned PR #31192 CUDA GPU-offload
oracle bitwise across all 200,064 vocabulary entries per row. Top-1 was 100%,
KL and maximum absolute difference were zero. The predeclared strict reference
for 1/2/31 was the unchanged baseline path; GPU equality there is an additional
observation on these inputs, not a general CPU/GPU equivalence guarantee.
The full text gate20 also remained GPU-oracle bitwise: **28,723 rows**. It used
sealed fresh same-day oracle results from the vision window with matching IDs,
positions, actual chunks, F16 KV, placement and fusion profile rechecked.

A same-worker check first performed a 515-row warmup, retaining the quant/OAI
helpers, then 16K and 32K text with fixed 4096-row chunks. The final 256 rows
at each depth (**512 selected rows**) matched `4499fcbc` bitwise; this is not
an all-row long-context GPU-oracle comparison. The 515-row warmup's selected
256 heads matched too. A separate fixed-chunk-256 prefix/cancellation suite
passed all **34 requests**, including reuse, edits, rewind, cancellation,
cache opt-out and retry.

Boundary profiles positively exercised 54 hybrid layers at each of 32/128/1024.
They reported 167,521,792 retained quant-helper device bytes and zero runtime
quant-helper allocations. Other engine allocations are not included in that
counter. None of these 162 layer records used the 1–8-route padding branch
or a cached expert; model-level positive coverage of those cases is not claimed.
Component tests cover their rules.

CPU and CUDA each passed **15/15** model-free ctest groups. After publishing
the portable control tracer, chunk parser and cache-head comparator, that
benchmark suite passed **39/39**, and both ctest configurations again passed 15/15.
These additions changed no C++ or measured worker binary. The published trace
classes/parser match the measured versions by AST; the cache comparator is
byte-identical to the executed comparator. Only local ownership guards and CLI
plumbing differ. Their model-free provenance is recorded in
[`pr3-prefill-tools-checks.json`](../tools/m31/bench/receipts/pr3-prefill-tools-checks.json).
Default-build CPU/CUDA isolation against `4499fcbc` is recorded in
[`pr3-default-build-isolation.json`](../tools/m31/bench/receipts/pr3-default-build-isolation.json).
The detailed numerical recipe, input reconstruction and counters are in
[`pr3-prefill-numeric.json`](../tools/m31/bench/receipts/pr3-prefill-numeric.json).
HIP, SYCL, Windows, other GPUs and two-GPU placement were not tested.

### Original full-vocabulary cache-state checks

The original PR3 and base `m31_cache_model_check` executables then
passed four actual-model runs: A1124/B1124/A1324/B1324, reuse 1024, decode 7,
context 40000, maxchunk 32768 and checkpoint 512 MiB. The arena/cache/registration
and CPU 80 settings were the same as the throughput profile. GPU packing used
the checker's unchanged default 8, confirmed in every startup log.

All nine heads at each length matched base bitwise across all 200064 logits:
whole, repeated whole, whole-rewind, whole-decode-rewind, split, split-rewind,
split-decode-rewind, checkpoint restore and workspace-release/replay. This is
**18 paired full-vocabulary heads / 3,601,152 F32 pairs** with exact recipes,
finite values, full byte equality and raw top-1 metadata independently checked.
Each checker also passed its own same-history invariants. Actual cold chunks
were 1124/1324; split chunks were 1024+100/300. Decode-rewind records contain
seven real single-token forwards before replay.

Whole versus split, and whole versus whole-rewind, were not byte-identical in
either branch. Those are different histories, retained explicitly; they are
not mixed into a false equality requirement. Source, checker/library/native
binary and input seals remained unchanged. This numeric collection preceded
the separately declared speed-only confirmation in its owned GPU window.
The full evidence is in
[`pr3-prefill-cache-numeric.json`](../tools/m31/bench/receipts/pr3-prefill-cache-numeric.json).

### Initial live-prefix and long-throughput ABBA window

On 2026-10-28, one complete A B B A B A A B window compared the original PR3
worker (A) with `4499fcbc` (B), using the hardware and resource profile above.
Every source/binary seal stayed unchanged, request bodies matched, and the
background container identities stayed unchanged. All samples are retained.
The historical speed corpus is 930,275 bytes, SHA-256
`b3dd2af20842931b69a67cc3f80ae6cfe30c36c7df20a270895d9b6915d62067`.
Its 79 source spans include a locally modified llama-arch.cpp; a clean pinned
llama.cpp checkout does not reproduce the complete file. Repeating these exact
HTTP bodies requires that archived corpus. `--corpus` remains an explicit
portable input; another corpus defines a new matched measurement. The public
cache/boundary/long numerical generators in this document are separate, exact recipes.
**The initial speed window failed the predeclared 3% gate: 4K decode was
10.93% slower by median.** PP at all depths and decode at 16K/32K passed.
Collection completion is not an overall performance PASS. A separate
confirmation, if run, does not replace this failed result.

Live-prefix timing primes an actual 1024-row KV prefix, then measures +100 or
+300 rows with GEN max_new=1, temperature=0, seed=123 and ignore_eos=1. Each
arm has one excluded timing warmup and two measured repetitions per suffix.
All 48 append observations, including warmups, satisfied RESUME1024/read and
matching-history greedy-ID checks: +100 produced 84384 and +300 produced 32.
The prime was `[1024]`, each suffix was `[100]` or `[300]`; prime/reset work
is retained and excluded from append timing. Reported prompt time includes
encoding the new rows and the first output head/sample.

| Live append | A, ms median [min–max] | B, ms median [min–max] | B/A |
| --- | ---: | ---: | ---: |
| +100 | 829.554 [826.554–904.397] | 1904.745 [1903.110–1906.040] | 2.296× |
| +300 | 1175.985 [1174.520–1244.720] | 2939.020 [2936.540–2942.960] | 2.499× |

Each arm then loaded a fresh common server for the portable om_bench recipe:
one warmup, then 4K/16K/32K, greedy low reasoning, cache_prompt=false,
ignore_eos=true, 256 generated tokens. Both branches used the same PR3 client
and nonce seed. PP is prompt_n/prompt_seconds; TG is 255/decode_seconds.
Both include the first head/sample in prompt time. The table gives tokens/s.

| Depth | A PP [range] | B PP [range] | A TG [range] | B TG [range] | TG delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 1377.89 [1376.82–1380.58] | 1377.54 [1373.86–1380.49] | 23.40 [21.16–26.61] | 26.27 [26.00–26.57] | -10.93% |
| 16K | 1535.86 [1533.53–1537.17] | 1531.74 [1530.03–1531.92] | 25.58 [21.10–26.10] | 25.73 [24.95–25.95] | -0.58% |
| 32K | 1260.54 [1257.43–1261.65] | 1249.91 [1249.28–1250.60] | 24.21 [20.49–25.11] | 24.64 [22.59–25.09] | -1.75% |

All measured rows, with one retained warmup per arm excluded from this table:

| Arm | PP4K | TG4K | PP16K | TG16K | PP32K | TG32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-A | 1380.58 | 24.79 | 1533.53 | 25.57 | 1257.43 | 23.59 |
| 02-B | 1379.75 | 26.02 | 1531.76 | 25.63 | 1249.97 | 24.77 |
| 03-B | 1380.49 | 26.57 | 1531.72 | 25.95 | 1250.60 | 24.52 |
| 04-A | 1377.52 | 21.16 | 1536.32 | 26.10 | 1261.65 | 24.83 |
| 05-B | 1373.86 | 26.51 | 1531.92 | 24.95 | 1249.84 | 25.09 |
| 06-A | 1376.82 | 26.61 | 1537.17 | 25.58 | 1260.85 | 25.11 |
| 07-A | 1378.26 | 22.00 | 1535.41 | 21.10 | 1260.24 | 20.49 |
| 08-B | 1375.34 | 26.00 | 1530.03 | 25.82 | 1249.28 | 22.59 |

All adjacent declared pair deltas (A/B−1), in percent:

| A / B | PP4K | TG4K | PP16K | TG16K | PP32K | TG32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-A / 02-B | +0.06 | -4.72 | +0.12 | -0.25 | +0.60 | -4.77 |
| 04-A / 03-B | -0.22 | -20.39 | +0.30 | +0.60 | +0.88 | +1.29 |
| 06-A / 05-B | +0.22 | +0.37 | +0.34 | +2.52 | +0.88 | +0.06 |
| 07-A / 08-B | +0.21 | -15.38 | +0.35 | -18.30 | +0.88 | -9.32 |

The actual planner recipes were stable within each branch:

| Request | Prompt rows | A chunks | B chunks |
| --- | ---: | --- | --- |
| Warmup |1277 |1277 |1277 |
| 4K |4348 |4348 |4348 |
| 16K |16638 |9984+6654 |10240+6398 |
| 32K |33022 |9984+7808+6400+5504+3326 |10240+7936+6528+5632+2686 |

The quant-helper reservation changes available memory and therefore long
partitions. All four paired 4K response messages matched, while 16K/32K messages
differed between branches with different chunk histories. These throughput
rows do not establish cross-recipe logit identity or isolate a kernel cause.
The separate fixed 4096 selected-row checks above provide their own numerical
evidence. Sampled clocks and code review did not establish the cause of slow
decode arms; no row is discarded or excused by that uncertainty.

The complete raw summaries, timing warmups, cache observations, source/input
hashes and unchanged acceptance verdict are recorded in
[`pr3-prefill-abba.json`](../tools/m31/bench/receipts/pr3-prefill-abba.json).
### Separate speed-only confirmation

A separately declared A B B A B A A B window repeated only the speed recipe
after the full-vocabulary cache gate passed. It used the same original binaries,
model, corpus SHA, resource budgets, current client, fixed nonce seed and
control-only trace wrapper. A fresh server was loaded for every arm; there were
no interleaved prefix-latency requests. All eight arms and timing warmups remain.
Source/binary seals, request bodies and background container identities stayed
unchanged. The same 3% rule was declared before the run.

**This confirmation also failed:** 4K TG passed, but 16K and 32K TG medians were
3.51% and 3.12% slower. All three PP medians passed. The first full window remains
a separate failure; results are not pooled to hide either failed criterion.

| Depth | A PP [range] | B PP [range] | A TG [range] | B TG [range] | TG delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 1378.33 [1370.44–1379.00] | 1379.27 [1376.08–1379.62] | 25.78 [22.16–26.18] | 25.89 [24.22–26.30] | -0.44% |
| 16K | 1534.11 [1533.12–1535.56] | 1529.98 [1525.05–1530.92] | 24.52 [21.77–25.75] | 25.41 [23.83–25.73] | -3.51% |
| 32K | 1259.25 [1258.33–1260.54] | 1248.56 [1247.08–1249.63] | 24.28 [21.27–25.00] | 25.06 [23.12–25.26] | -3.12% |

| Arm | PP4K | TG4K | PP16K | TG16K | PP32K | TG32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-A | 1377.78 | 26.18 | 1533.12 | 25.21 | 1258.33 | 25.00 |
| 02-B | 1379.62 | 25.77 | 1529.89 | 25.73 | 1247.08 | 25.04 |
| 03-B | 1376.08 | 26.02 | 1525.05 | 25.28 | 1248.39 | 25.26 |
| 04-A | 1378.87 | 22.16 | 1534.42 | 21.77 | 1258.77 | 21.27 |
| 05-B | 1379.18 | 24.22 | 1530.07 | 25.54 | 1248.73 | 23.12 |
| 06-A | 1370.44 | 25.79 | 1535.56 | 23.83 | 1259.72 | 24.73 |
| 07-A | 1379.00 | 25.77 | 1533.80 | 25.75 | 1260.54 | 23.83 |
| 08-B | 1379.35 | 26.30 | 1530.92 | 23.83 | 1249.63 | 25.08 |

All declared pair deltas (A/B−1), percent:

| A / B | PP4K | TG4K | PP16K | TG16K | PP32K | TG32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-A / 02-B | -0.13 | +1.61 | +0.21 | -2.02 | +0.90 | -0.16 |
| 04-A / 03-B | +0.20 | -14.86 | +0.61 | -13.89 | +0.83 | -15.79 |
| 06-A / 05-B | -0.63 | +6.46 | +0.36 | -6.69 | +0.88 | +6.94 |
| 07-A / 08-B | -0.03 | -2.01 | +0.19 | +8.09 | +0.87 | -4.99 |

Actual chunks remained the same branch-specific recipes as the first window.
4K response messages matched across every declared pair; long responses used
different chunk histories. Neither code inspection nor sampled CPU/GPU
counters isolated the cause of decode differences. Changing the corpus, affinity,
sampling, thresholds or removing an outlier is not used to turn a failed
window into a pass. The short-append latency improvement is measured; the full
long-throughput no-regression gate for the original worker was **not met**.

Complete confirmation data and its unchanged failure are recorded in
[`pr3-prefill-speed-confirmation.json`](../tools/m31/bench/receipts/pr3-prefill-speed-confirmation.json).

### Original sampled quality

The original `5a7d75db…` worker completed all 123 requests in one unseeded trial:

| Suite | Original PR3 | Required | Result |
| --- | ---: | ---: | --- |
| GSM8K, first 100 ordered rows |97/100 |95/100 |PASS |
| Tools, 20 prompts plus 3 continuations |18/23 |19/23 |**FAIL** |

Wire requests explicitly set temperature 1, top_p 0.95, top_k 40, min_p 0.05
and max_tokens 6000. They omitted seed, reasoning_effort, chat_template_kwargs
and cache_prompt, using model-default thinking and the server's normal cache
policy. All 123 rows had zero length stops, marker leaks and invalid tool IDs.
GSM8K requests reported no reused tokens; tools reported 0 on the first request
and 768 on each of the remaining 22. Tool calls were never executed.

The five tool failures were cases 4, 8, 11, 12, 18. The returned JSON was valid;
these cases selected preparatory shell/calendar/status probes as their first
action. Case 8 had the expected service action second, but the unchanged scorer
checks the first call and it remains a failure. The three wrong GSM8K rows
(12, 16, 93) contained wrong final numeric answers; extraction matched those
visible answers. Parser, common-server shim, grammar, sampler, quality client
and fixtures were unchanged from 4499fcbc. These facts do not establish a
stochastic or numerical cause, and no successful replacement trial is used.
The original quality trial remains **FAIL** because 18<19; it is retained
separately from the corrected-worker trial below.

All outcomes, sampling fields, cache counts, hashes and source/binary stability
are in [`pr3-prefill-quality.json`](../tools/m31/bench/receipts/pr3-prefill-quality.json).

### Decode investigation and bounded row scheduling (2026-10-29)

Commit `13af6b16` replaces static per-worker partitions of the fused CPU
gate/up and down matrices with **64-row dynamic batches**. Two separately
64-byte-aligned atomic counters are reset by the submitting thread before ggml
publishes each graph. Workers claim independent output rows; the dot kernels,
scale/route multiplications, hidden activation arithmetic and original-rank
reduction order are unchanged. All five barriers still use the actual callback
participant count. This limits matrix work stranded behind a slower worker;
every participant must still reach the barriers.

The change has no scheduling flag, affinity policy or lazy-allocation option.
The hybrid CPU lane and eager quant-helper reservation remain as before.
A model-free fixture now also requests nine workers for eight down-matrix
batches, exercising participants that receive no down work, repeated graph reuse,
and the existing requested-four/pool-two mismatch. Repeating identical inputs
does not independently prove reset correctness if stale outputs match; the
changing-input gate20 decode sequences below provide stronger reset coverage.

**Corrected-worker numerical, decode and cache-speed gates passed. Overall acceptance is FAIL: tools23 scored 18/23 against 19/23. The PR remains draft.** Historical
and diagnostic measurements below belong to the original worker or explicitly
named diagnostic binaries. Completed corrected-worker numerical checks are
identified separately in the validation subsection; no diagnostic throughput
is promoted to final acceptance.

#### Historical controls retained

The base/base controls both passed the unchanged per-depth decode gate
(median candidate/base >=0.97). Default deltas at 4K/16K/32K were
+2.543% / -0.127% / +0.802%; fixed-3072 deltas were
+0.651% / +0.841% / -1.799%. These are two observed control windows, not a
precise estimate of the population false-failure rate.

The original PR3 family failed six of seven completed windows. All failures
remain recorded; the two always_inline rows are a previous unsuccessful
candidate, not the bounded-row change:

| Historical window | A worker | 4K TG delta | 16K TG delta | 32K TG delta | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Initial full | Original PR3 | -10.931% | -0.585% | -1.754% | FAIL |
| Separate speed-only | Original PR3 | -0.442% | -3.513% | -3.118% | FAIL |
| Clean default | Original PR3 | -0.099% | +0.980% | -0.001% | PASS |
| Clean fixed-3072 | Original PR3 | -3.359% | -0.220% | -1.106% | FAIL |
| Earlier candidate, fixed-3072 | always_inline | -3.792% | +0.269% | +0.969% | FAIL |
| Earlier candidate, default | always_inline | +0.595% | +0.211% | -3.561% | FAIL |
| Original PR3, subsequent default control window | Original PR3 | -8.576% | -5.950% | -7.963% | FAIL |

Original PR3 identifies worker `5a7d75db…`; base identifies `a2f47f96…`
at `4499fcbc`. The always_inline candidate was `2f2a1b3e…`.
The decode numerator was 255 transitions for 256 outputs. The original
default planner can choose different long chunk histories between branches;
fixed-3072 removes that recipe difference. No slow arm was discarded.

#### Warmup and allocation controls

The proposed hybrid-warmup explanation had an incorrect premise:
`speed.py` defines warmup as 1,024 selected input tokens and **32 generated
tokens**. Formatting produces an actual 1,277-token prompt, chunk `[1277]`,
outside the hybrid range. All 40 archived warmup groups in the audited scope
confirmed this. Historical failures therefore do not require a hybrid pass.

A separate native diagnostic tested exact prompt warmups of 32 and 1,025
tokens: A=original PR3/32, B=original PR3/1025, C=base/32, D=base/1025.
Fresh processes ran in ABCDDCBA order, two per condition; each filtered pair
forms ABBA. Long raw prompts were 4096/16384/32768, fixed chunks were 4096,
and generation used 128 greedy outputs. All long recipes and generated IDs
matched; this does not claim full-vocabulary logit identity.

| Condition | 4K median TG | 16K median TG | 32K median TG |
| --- | ---: | ---: | ---: |
| A, original PR3 / warm32 | 23.555577 | 23.961574 | 23.196042 |
| B, original PR3 / warm1025 | 26.326595 | 24.682994 | 25.075013 |
| C, base / warm32 | 24.660416 | 25.767398 | 25.212078 |
| D, base / warm1025 | 26.079787 | 25.971172 | 24.380025 |

The retained slow 08-A process delivered 20.824 / 22.107 / 21.211 tok/s;
the other A process delivered 26.287 / 25.816 / 25.181. Even without a hybrid
warmup, B/D at 16K was **-4.960%**. Two observations per condition do not prove
warmup poisoning or its absence.

After each actual hybrid warmup, one CPU-lane thread was added. It then slept
in `futex_do_wait`, with zero additional CPU ticks and scheduler runtime
through all three long requests. Masks remained 0–191, nice 0 and policy 0.
The slow process did not show additional lane CPU usage or greater pool
runqueue wait than its fast counterpart. These snapshots do not distinguish
useful work from userspace barrier spin.

Another diagnostic compared eager and lazy helper initialization in the
**same ELF**, ABBA with two fresh processes per side, the unchanged HTTP speed
client/corpus, and fixed-3072 chunks. Lazy/eager decode deltas were
**-2.371% / -1.421% / +3.945%**. Bodies, recipes and generated responses
matched. The rounded GPU memory counter was exactly 160 MiB lower with lazy
initialization at each depth, consistent with the 167,521,792-byte helper
counter. This did not produce a consistent benefit. A supplementary observer
was introduced during the first eager 32K decode and covered later processes
fully, so this window also has a partial-start instrumentation confound.
It is not a clean causal estimate or an accepted fix.

#### Worker progress, SMT and frequency

A temporary same-treatment fused-thread probe retained all 128 complete
sampled tokens (7,296 layers). Fast/slow quartile median routed times were
22.220/25.296 ms at 4K, 22.319/31.242 ms at 16K and 22.219/32.105 ms at 32K.
Slow layers had larger critical-worker matrix times and corresponding barrier
waits. For gate/up, the median critical/median-worker work ratio increased
from about 1.18 in fast long-context samples to 1.82–1.92 in slow ones.
These are descriptive quartiles of one treatment, not independent trials or
an outlier filter applied to acceptance.

Own-model SMT pairs were associated with some slow samples: stationary
entry/exit CPUs, different logical siblings of the same core, and overlapping
callback intervals. They appeared in 25.0% / 37.5% of slow 16K/32K layers and
0% of their fast layers. In one repeated slow 32K case, a critical worker on
CPU104 shared a core with another model worker on CPU8 for 114 layers.
Entry/exit observations do not exclude migration away and back; overlapping
wall intervals do not prove continuous execution on those CPUs. External
background matches were less consistent, and the 1 Hz observer cannot resolve
sub-millisecond matrix stages.

A sustained whole-pool frequency collapse was not supported. In archived
normal versus pinned replays, user-cycles/task-clock stayed near 3.43–3.47 GHz
while routed work rose from about 23 ms to 32–35 ms; IPC fell from about 2.0
to 1.58–1.69. Uniform probe APERF/MPERF ratios in the four 4K counter windows
were 1.46270–1.46864. Those counters were system-wide on mapped logical CPUs,
and the reference frequency was not independently established; the ratios
are not converted to GHz. A short frequency change on one critical worker
is not excluded.

#### Same-process scheduling/affinity control

A four-condition diagnostic held one process, pool and heap throughout:
A=static/all initially allowed CPUs, B=dynamic/all, C=static/one allowed
logical sibling per physical core, D=dynamic/physical. The physical mask
covered all 96 physical cores and all 12 L3 domains. Only model tasks changed
affinity, at completed-request boundaries; background jobs retained their
settings. Uniform observers started before model load.

Each depth used ABCDDCBA, two observations per condition, canonical raw prompts
4348/16638/33022, fixed-3072 chunks and 256 greedy outputs. All 24 observations
are retained. One native PID, 18 consumed scheduling transitions, identical
per-depth prompt/chunk/output histories and before/after affinity masks were
verified.

| Contrast, median TG | 4K | 16K | 32K |
| --- | ---: | ---: | ---: |
| Dynamic/all versus static/all, B/A | +2.9596% | +6.9191% | +8.6244% |
| Dynamic/physical versus static/physical, D/C | +7.614% | +7.012% | +3.114% |
| Static physical versus static all, C/A | -6.116% | -1.099% | +2.979% |

Dynamic row scheduling helped under both affinity settings; physical-only
placement alone was not a consistent improvement. The selected source change
therefore changes row scheduling and leaves affinity and eager allocation
alone. This supports a matrix-work imbalance/barrier amplification mechanism.
It does **not** establish why the original PR3 processes encountered slow
placement/progress more often than base in the historical windows: their
fused source was already identical, and base also exhibited slow tails.

This is a single-process crossover with n=2 per condition/depth. Expanding a
mask does not restore its previous placement, so carryover remains possible.
It is neither the required fresh-process HTTP acceptance nor a broad hardware
calibration. The first process state, other machines and quants may behave
differently.

#### Corrected-source validation status

The corrected source is `13af6b16`. CPU and CUDA builds completed, and each
passed all 15 model-free ctest groups. The final CUDA worker SHA-256 is
`8580e92a4b7011ed523c00d13b4fc9d68e827a7ed0041a527e31fd005645b481`.

All seven boundaries (1/2/31/32/128/1024/1025; 2,243 rows) matched the sealed
saved PR3, base and GPU-oracle references bitwise across all 200,064 vocabulary
entries per row. The corrected-worker gate20 matched the sealed GPU oracle
and saved PR3 in all 20 cases, 28,723 rows. This gate20 comparison does not
claim a newly generated base4499 gate20 dump. Prefix/cancellation passed all
34 requests; the 32 completed deterministic token sequences matched saved PR3.
Cancelled tails are timing-dependent, so their contract is checked without
claiming arbitrary byte equality. These are new corrected-worker runs against
saved references, not a fresh oracle collection.

| Corrected-worker check | Status |
| --- | --- |
| CPU and CUDA build/ctest | PASS, 15/15 each |
| Boundaries 1/2/31/32/128/1024/1025, bitwise contract | PASS, all 2,243 rows |
| gate20 | PASS, 20 cases / 28,723 rows |
| Prefix/cancel | PASS, 34/34; 32 deterministic sequences matched |
| Live-cache +100/+300 speed retained | PASS, 2.233x / 2.396x versus base |
| Fresh-process ABBABAAB versus base, fixed-3072, each depth >=97% | PASS, all three depths; PP gates also PASS |
| Fresh-process ABBABAAB versus base, default, each depth >=97% | PASS, all three decode depths |
| GSM8K100, reference sampling | PASS, 98/100 against 95/100 |
| tools23, reference sampling | FAIL, 18/23 against 19/23 |
| Final resource cleanup | PASS, all recorded checks |
| Overall acceptance | FAIL, tools23 threshold only |

The completed corrected-worker fixed-3072 window used eight fresh processes
in ABBABAAB order (four per arm), the same HTTP bodies and actual chunk recipes,
and 256 greedy outputs. All responses matched across arms. All eight worker
seals and protocol traces validated, and all three PP gates passed.

| Fixed-3072 depth | Corrected median TG | Base median TG | Corrected/base delta | TG gate |
| --- | ---: | ---: | ---: | --- |
| 4K | 27.62988284 | 25.93835484 | +6.5213387916% | PASS |
| 16K | 27.00913608 | 25.14828611 | +7.3995100801% | PASS |
| 32K | 25.98482006 | 25.11387616 | +3.4679787920% | PASS |

The separate default window also completed ABBABAAB, four fresh processes per
arm:

| Default depth | Corrected median TG | Base median TG | Corrected/base delta | TG gate |
| --- | ---: | ---: | ---: | --- |
| 4K | 27.32844088 | 26.16726554 | +4.4375112128% | PASS |
| 16K | 26.71576819 | 25.85912935 | +3.3127133837% | PASS |
| 32K | 25.88018169 | 24.23378420 | +6.7938109701% | PASS |

**All six final decode gates passed.** There was one final ABBABAAB window per
mode, with no retries to obtain PASS. These results do not replace any
historical failure. Unlike fixed-3072, default long chunk histories may differ
between corrected PR3 and base because the unchanged helper reservation
affects planner headroom. Matching all default recipes, response messages or
full logits is not claimed; default measures the matched HTTP workload with
each branch's planner.

The final live-cache speed check also passed:

| Append to live 1024-token prefix | Corrected median, ms | Base median, ms | Base/corrected |
| --- | ---: | ---: | ---: |
| +100 | 853.0165 | 1904.775 | 2.2329872869x |
| +300 | 1226.975 | 2939.955 | 2.3961001650x |

All 48 observations are retained. The 16 timing warmups were excluded from
the medians; prime/reset work was recorded separately. Actual reuse was 1024,
new rows read were 100 or 300, and same-history greedy IDs passed. The original
append measurements above remain separate historical results.

The corrected worker completed **one** sampled quality trial: GSM8K 98/100
passed the 95/100 threshold; tools23 scored 18/23 and failed the 19/23 threshold.
All 123 records completed with zero length stops, marker leaks or invalid tool
IDs. Requests retained the reference sampling: temperature 1, top_p 0.95,
top_k 40, min_p 0.05, max_tokens 6000, model-default thinking and no added seed.
No replacement trial was used to obtain a passing result. The tools failure
is not excused by the successful numerical/performance checks.

The final pipeline has complete=True and unchanged=True. Stage return codes
were numeric 0, performance 0 and quality 1. Controller exit 1 therefore records
a completed quality-gate failure, not an incomplete run or cleanup failure.
All cleanup operations returned 0; the final resource audit passed every
recorded check: miner ownership restored, no hold or temporary preset, config
restored, leases free, CPU chat healthy and background identities retained.

Two earlier technical failures are retained separately: the reserve session's
first resource audit ran before the asynchronous miner process appeared and
failed; its later readiness audit passed. The first fused-profile derivation
also returned 2 / valid=False because it expected nth in worker records. The
corrected v2 reads nth from layer summaries and validates 57 layers / 80 workers;
it is valid=True on the same 128 sampled tokens. The initially failed audit, parser output and raw measurements remain
available; neither failure was silently removed.

The original cache speeds and original GSM8K 97/100, tools 18/23 FAIL remain
separate historical evidence. Overall acceptance of the corrected worker is
**FAIL solely because 18<19 on tools23**, so the PR stays draft. HIP, SYCL,
Windows, other GPUs/quants, multi-GPU placement and a new OFF-build isolation
comparison were not tested in this follow-up.

The complete evidence, including all historical failures, both final decode
windows, numerical references, quality and resource receipts, is in
[`pr3-decode-root.json`](../tools/m31/bench/receipts/pr3-decode-root.json).
It identifies measured source commit 13af6b1664a1aa1e89e759bd58fbd0f014baeb2e
separately from any later documentation-only commit.

The local root investigation archive retains every diagnostic observation and
failure under `/mnt/ai/logs/m31-pr2/root/`: `historical-warmup-audit.json`,
`warmup-session/warmup-summary.json`, `warmup-thread-analysis.json`,
`reserve-session/diagnostic-analysis.json`, `instrumentation-scope.json`
inside that reserve directory, `thread-analysis-v2/summary.json`,
`own-smt-analysis/summary.json`, `critical-background-analysis/summary.json`,
`thread-profile-session/thread-frequency-analysis.json`, and
`dynamic-affinity-session/dynamic-affinity-analysis.json`. These temporary
instrumentation probes are local diagnostic artifacts, not shipped benchmark
interfaces. The linked final receipt identifies the corrected ELF and its
completed validation results.

For a portable corrected/base comparison, use the public boundary, prefix,
cache-head and HTTP recipes below with separate output directories and a
separately built `4499fcbc` baseline. Run each arm inside the same GPU resource
reservation, retain CPU/GPU background workloads, and use fresh processes in
ABBABAAB order. Keep all results, including failures. The HTTP command below
uses the same public client for both servers; it does not recreate the
temporary same-process switches:

```sh
# MODEL, TOKENIZER_DIR and CORPUS are explicit local inputs.
# Set the server's native --chunk to 32768 for default, then 3072 for fixed.
python tools/m31/bench/speed.py --url http://127.0.0.1:8080 \
  --tokenizer "$TOKENIZER_DIR" --corpus "$CORPUS" --repetitions 1 \
  --nonce-seed m31-pr3-abba-fixed --out speed-corrected-A1
```

Repeat for each fresh arm and both chunk settings with unique output paths.
The standard hardware/resource profile is the RTX 5090 32 GB at 440 W,
EPYC 9654, 624 GiB RAM, 80 CPU threads, a 170 GiB arena, a 7 GiB expert cache,
a 176 GiB registered-memory budget, context 40000 and 8 packing threads.
Keep F16 KV, fusion off, and OMP/OPENBLAS at one thread. A different corpus or chunk history
defines a new comparison and must be reported.

### Reproduce short-chunk checks and timings

Build and run the normal CUDA target above; there is no short-prefill option
to enable. Compare against a separate checkout at upstream `4499fcbc`. Keep
model, resource budgets, F16 KV, fusion/MMQ settings, token IDs and chunk recipe
fixed. For a direct 128-row hybrid numerical check:

```sh
export GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_MMQ_PREC=auto
python tools/m31/bench/make_prompts.py gate20-text
build-m31-oracle/m31_oracle tokenize "$MODEL" gate20-text gate20-ids
build-m31/m31_generate --model "$MODEL" --backend cuda --ctx 4096 \
  --chunk 128 --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
  --register-arena 1 --register-limit-gib 176 --gpu-pack-threads 8 \
  --ids-file gate20-ids/code_02_128.i32 --rows all --out short-native
python tools/m31/bench/replay_oracle.py --candidate short-native \
  --ids gate20-ids --oracle build-m31-oracle/m31_oracle \
  --model "$MODEL" --threads 80 --out short-oracle
python tools/m31/bench/compare_logits.py short-oracle/reference short-native \
  --require-bitwise --oracle-contract gpu-offload --out short-comparison.json
```

`gate20-ids/code_02_128.i32` is a 128-token little-endian int32 input generated
by the portable prompt/tokenization recipe above. Keep complete actual chunks and all 200064 logits per selected row.
The policy boundaries are 1/2/31/32/1024/1025; short tests must not accidentally
fall back to a long batch and then claim to cover hybrid.

The 1/2/31-row arithmetic uses `4499fcbc` at the same IDs/chunks as its
strict byte oracle. The 2–31-row path keeps PR1's CPU activation arithmetic;
its GPU-oracle differences are reported as diagnostics. New hybrid 32/128/1024
and preserved batch 1025 are checked strictly against the GPU-offload oracle.
This distinction was declared before the GPU boundary run; it does not relax
the complete gate20's GPU byte requirement.

For the exact seven-boundary and retained-helper long recipe, generate the
public cache text, then take prefixes of its **complete** token stream (ignore
its `.length` truncation for this step). This Python-only step uses exported
vocabulary/merges, NFC and parse_special=False, without BOS or a chat template:

```sh
python tools/m31/bench/make_prompts.py cache-text --suite cache-observatory
python - "$TOKENIZER_DIR" <<'PYCODE'
import json, struct, sys
from pathlib import Path
sys.path[:0] = ["tools", "tools/m31/serve"]
from m31_tok import M31Tokenizer
export = Path(sys.argv[1])
vocab = json.loads((export / "vocab.json").read_text())
tokens = [None] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
metadata = json.loads((export / "tokenizer.json").read_text())
tokenizer = M31Tokenizer(tokens, (export / "merges.txt").read_text().splitlines(),
    json.loads((export / "token_type.json").read_text()),
    special_ids=list(metadata["special_ids"].values()))
ids = tokenizer.encode(Path("cache-text/cache_observatory_1324.txt").read_text(),
                       parse_special=False)
for directory, cases in (
    ("boundary-ids", [(f"n{n:04d}", n) for n in (1, 2, 31, 32, 128, 1024, 1025)]),
    ("fixed-long-ids", [("00-retain515", 515), ("10-long16384", 16384),
                        ("20-long32768", 32768)]),
    ("cache-numeric-ids", [("corpus", 33000)])):
    out = Path(directory)
    out.mkdir()
    for name, count in cases:
        assert len(ids) >= count
        (out / (name + ".i32")).write_bytes(struct.pack(f"<{count}i", *ids[:count]))
PYCODE
```

Run `--cases boundary-ids --rows all --ctx 4096 --chunk 2048` on each executable;
each boundary must remain one actual chunk. Replay the independent GPU oracle
using the candidate's manifests as above. Use the baseline's same-path heads
for the strict 1/2/31 checks. For long-context allocation/history coverage:

```sh
# BASE_ENGINE names a separately built 4499fcbc m31_generate.
for ARM in base candidate; do
  if [ "$ARM" = base ]; then ENGINE="$BASE_ENGINE"; else ENGINE=build-m31/m31_generate; fi
  "$ENGINE" --model "$MODEL" --backend cuda --threads 80 \
    --ctx 40000 --chunk 4096 --arena-gib 170 --gpu-experts-gib 7 \
    --register-arena 1 --register-limit-gib 176 --gpu-pack-threads 8 \
    --rows last:256 --head-tail 256 --cases fixed-long-ids --out "fixed-long-$ARM"
done
python - <<'PYCODE'
import json, sys
from pathlib import Path
sys.path.insert(0, "tools/m31/bench")
import compare_logits as C
for name, count in (("00-retain515", 515), ("10-long16384", 16384), ("20-long32768", 32768)):
    reference, candidate = (Path("fixed-long-" + arm) / name for arm in ("base", "candidate"))
    expected = [(pos, min(4096, count - pos)) for pos in range(0, count, 4096)]
    for case in (reference, candidate):
        manifest, _ = C.load_case(case)
        assert C.execution_chunks(manifest) == expected
        assert C.full_last_layer_before_crop(manifest)
    result = C.compare_case(reference, candidate)
    print(json.dumps(result))
    assert result["bitwise_identical"]
PYCODE
```

The sorted case names keep 515-row warmup, 16K, then 32K in one Engine process.
Reset clears KV between cases and retains auxiliary allocations. Compare raw
input and executable hashes with the receipt; another partition is a new recipe.

For live-prefix latency, use `tools/m31/bench/prefix_speed.py` with at least 1324
natural-text token IDs. It primes an actual 1024-row KV prefix, checks `RESUME
1024`, then measures a 100- or 300-row append. Prime/reset work is recorded and
excluded from append timing. The measured window used one warmup and two measured
repetitions per suffix per arm. Keep the warmup and every measured repetition:

```sh
python tools/m31/bench/make_prompts.py prefix-cache-text --suite cache-observatory
build-m31-oracle/m31_oracle tokenize "$MODEL" prefix-cache-text prefix-cache-ids
python tools/m31/bench/prefix_speed.py --engine build-m31/m31_generate \
  --model "$MODEL" --ids-file prefix-cache-ids/cache_observatory_1324.i32 --out prefix-candidate \
  --prefix 1024 --suffixes 100,300 --repetitions 2 \
  --threads 80 --context 40000 --chunk 32768 --arena-gib 170 \
  --gpu-experts-gib 7 --register-arena 1 --register-limit-gib 176 --pack-threads 8
```

Repeat with the base executable and new output directories in the order
candidate/base/base/candidate/base/candidate/candidate/base. Use the same current
PR3 `bench/speed.py` client for both branches' servers. The optional nonce seed
fixes every HTTP body, including the warmup, across arms:

```sh
python tools/m31/bench/speed.py --url http://127.0.0.1:8080 \
  --tokenizer "$TOKENIZER_DIR" --corpus "$CORPUS" --repetitions 1 \
  --nonce-seed m31-pr3-abba-fixed --out speed-A1
```

Repeat 4K/16K/32K in the same order with a freshly loaded server on every arm,
and new output directories. Omitting the flag retains the old random nonces.
GPU and CPU background jobs must remain running in both arms. Report median, range, all pairs, actual prompt sizes/chunks and executable
hashes. The accepted performance gate rejects a median PP or TG slowdown above
3% at any of those depths; a noisy failed window is retained as a failure.

### Reproduce full-vocabulary cache heads

The complete-ID generator above also writes `cache-numeric-ids/corpus.i32`.
Set `IDS_FILE="$PWD/cache-numeric-ids/corpus.i32"`, `PR3_BUILD` to the candidate
build and `BASE_BUILD` to an independent `4499fcbc` build. Build the checker with
`cmake --build BUILD --target m31_cache_model_check -j 8` in each build tree.
The following Bash commands use the measured resource profile and new outputs:

```bash
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_MMQ_PREC=auto
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_DEVICE_MAX_CONNECTIONS CUDA_DEVICE_MAX_COPY_CONNECTIONS
common=(--model "$MODEL" --ids-file "$IDS_FILE" --backend cuda --device 0
        --ctx 40000 --chunk 32768 --reuse 1024 --decode-steps 7
        --threads 80 --arena-gib 170 --gpu-experts-gib 7
        --register-arena 1 --register-limit-gib 176 --checkpoint-mib 512
        --cpu-fused 1)
"$PR3_BUILD/m31_cache_model_check" "${common[@]}" --length 1124 --out cache-A-1124
"$BASE_BUILD/m31_cache_model_check" "${common[@]}" --length 1124 --out cache-B-1124
"$PR3_BUILD/m31_cache_model_check" "${common[@]}" --length 1324 --out cache-A-1324
"$BASE_BUILD/m31_cache_model_check" "${common[@]}" --length 1324 --out cache-B-1324
python3 tools/m31/bench/compare_prefix_heads.py cache-A-1124 cache-B-1124 \
  --length 1124 --out cache-1124-comparison.json
python3 tools/m31/bench/compare_prefix_heads.py cache-A-1324 cache-B-1324 \
  --length 1324 --out cache-1324-comparison.json
```

The unchanged checker uses eight GPU packing threads; there is no checker CLI option for this setting. Confirm `pack_threads=8` in its startup log. Keep each command's exit status, complete log, `summary.json`, and all nine `.f32` files. A checker failure remains a failure even if it wrote some heads.

The comparator checks every one of the 200064 F32 logits in all nine heads, exact chunk recipes, finite values, raw top-1 metadata, rewind/decode-rewind, repeated cold prefill, checkpoint restore, and workspace release. Whole prefills must use one 1124/1324-token chunk; cached prefills must use exactly 1024+100/300. Every same-history baseline/PR3 head must have identical bytes. Different whole-versus-segmented histories are reported separately and do not have an equality requirement. The comparator returns nonzero on any mismatch; it has no tolerance option.

Model-free regression tests, after measurement windows have finished:

```bash
python3 tools/m31/tests/test_prefix_heads.py
```


Run GSM8K100 and tools23 with the quality recipe below on the final executable:
temperature 1, top_p 0.95, max_tokens 6000 and model-default thinking. The existing
thresholds are 95/100 and 19/23. Keep unsuccessful sampled trials too.

Model-free CPU and CUDA ctest include policy/cost, native activation arithmetic,
CPU lane failure/recovery, requested-four/pool-two worker mismatch, nine workers
for eight down-matrix batches, and allocation-failure retry tests. They do not replace model-level GPU checks.

## Cache state and numerical history

Prefix reuse retains K/V and MSA index rows produced by the earlier forward
passes. That history includes their batch partitions. It does not rebuild those
rows as one larger cold batch. On the tested RTX 5090, changing the partition
changed the pinned llama.cpp GPU-offload reference itself:

| Prompt tokens | Whole / split batches | Last-head top-1, whole / split | KL(whole || split) |
| --- | --- | --- | ---: |
| 257 | `257` / `256 + 1` | 12156 / 8364 | 4.1723231901 |
| 1324 | `1324` / `1024 + 300` | 32 / 32 | 0.0240769726 |

These are observations on the deterministic observatory fixture below, with
F16 KV, fusion off, context 40,000 and PR #31192 at the pinned commit. Each of the
four native last-position heads was bitwise identical to the corresponding
reference head with the **same** partition: all 200,064 F32 logits, KL 0 and
maximum absolute difference 0. Registered and staged transport also agreed for
those heads. The machine was the EPYC 9654 / RTX 5090 described below; this is
not a measurement on other GPUs. The portable receipt is
[`cache-partitions.json`](../tools/m31/bench/receipts/cache-partitions.json).

A cached conversation can therefore choose a different next token from a fresh
full prefill when their batch histories differ. Even an unchanged top-1, as in
the 1324-token example, does not imply identical logits or subsequent decoding.
No byte-identity promise is made between those different histories.

The diagnostic first found small layer-0 F16 K/V/index differences when forming
the prefix with different batch sizes; later layers amplified them. A plausible
explanation is shape-dependent floating-point reduction order in dense CUDA
projections, together with different one-token and batch kernels. The separate
CPU expert decode path can also change the one-token recipe. The 1324-token
example has a 300-token GPU suffix, so a CPU-only explanation is insufficient.
The specific kernel cause has not been isolated. The upstream counterexample
is the demonstrated observation; the causal explanation remains an inference.

Cache correctness still requires strict checks with the same history: rewind,
decode-then-rewind, checkpoint restore and workspace release must preserve the
retained state and reproduce its suffix heads. The public `m31_cache_model_check`
checks these invariants. The separate `m31_cache_oracle` builds the old prompt,
optionally decodes, removes the suffix through llama.cpp's public KV API and
replays the new suffix. That is the appropriate reference for a live-history
cache result. A new cold full prefill is a different numerical experiment.
`cache_prompt=false` requests fresh processing and disables cache lookup and store.
The short/long numerical gates continue to require matching actual chunks;
the partition observation does not relax their thresholds.

## PR1 candidate checks

On 2026-10-27, the final text runtime sources in local commits `b7121972` and
`45b54299` built with GCC 14.3 / CUDA 13.1.115 / `sm_120a`. The CUDA
`m31_generate` SHA-256 was
`8c3710d7a3bda60e1a3556cc05b96fa3c95777d2e70ed06c484d5d89ce57527e`.
CPU and CUDA each passed all 10 model-free ctest groups, including forced CPU
allocator failures followed by reset/retry and byte-equal output.

The same binary passed the full gate20 against independent GPU-offload oracle
outputs: 20 prompts, 28,723 positions (every prompt row), top-1 100%, mean/max KL 0 and
bitwise equality. The historical 16K/32K input pair also passed on all 512
selected positions with the same exact results. Token IDs, actual chunk plans,
placement, F16 KV and final-layer policy were checked before comparing logits.
The candidate used the context and resource settings shown in the run example;
gate20 used context 4,096 and chunk 2,048. The long oracle used the verified
bounded-capacity replay described below.

Cache-state checks passed for 257/R256 and 1324/R1024, including rewind,
decode-before-rewind, checkpoint restore and released compute buffers. The four
corresponding full-vocabulary heads matched the independently executed cache
oracle bitwise. A 34-request protocol test with fixed chunk 256 also passed:
boundary reuse, appended and edited prompts, cancellation during prefill/decode,
retry, and cache opt-out. Keeping the chunk recipe fixed is necessary for these
strict cold/resumed comparisons, as explained above.

The four public Russian/code cases at 16K and 32K also matched the GPU-offload
oracle bitwise on all 1,024 selected positions: top-1 100%, mean/max KL 0.
These long checks cover selected tail rows, rather than every position.

The common HTTP server passed live checks for streaming and non-streaming
client disconnects, queued work after cancellation, all three zero-token aliases,
invalid boolean limits and recovery after deliberately ending its own worker.
The request interrupted by worker death returned an error; a later request
restarted it and completed. First visible SSE output arrived after 3.04 seconds
in the cancellation probe. Seven HTTP cache requests also passed actual reuse,
opt-out and same-recipe response checks.

Quality and speed collection completed on the same day with that unchanged
worker. All tools23 trials are retained below; the tools threshold stayed at
19/23 and the GSM8K100 threshold stayed at 95/100.

| Tools23 trial | PR1 common server | llama.cpp tuned Q8_0 reference |
| --- | ---: | ---: |
| Original unseeded trial | **17/23 FAIL** | 20/23 PASS |
| Predeclared seed 0 | 19/23 PASS | 20/23 PASS |
| Predeclared seed 1 | 20/23 PASS | **18/23 FAIL** |

All six trials had zero length stops, control-marker leaks and invalid tool IDs.
The first PR1 trial's six failures include four also seen in historical v8 and
two that chose preparatory shell/search calls before the expected first action.
All six tools were present in the model prompt and grammar. Seeds 0 and 1 were
declared for both engines before those replications ran. No best trial replaced
the initial 17/23 failure, and the reference seed-1 failure is also retained.

The client omitted `cache_prompt` on quality requests, leaving each server's
cache policy active. Native tools responses usually reported 768 reused tokens;
the reference usually reported 838, with 841 on the final continuation. The
receipt records these API counts per request. Fixed seeds do not make different
cache histories, KV types or kernels equivalent, so these scores alone do not
identify the cause of a quality difference.

GSM8K100 scored **97/100 for PR1 and 96/100 for the tuned-Q8 reference**. Both
passed with zero length stops, marker leaks and invalid IDs. All 100 test rows
were used in their original order, with the same client, dataset and explicit
sampling fields. One stochastic 100-row run per engine does not establish a
statistically significant quality difference.

Speed used the same corpus and HTTP request recipe on all three arms: warmup,
256 generated tokens, greedy sampling, low reasoning, `ignore_eos=true` and
`cache_prompt=false`. The table gives median [minimum–maximum] tokens/s over
all retained rows. PR1 and the F16 reference have three repeats at each depth;
the tuned-Q8 reference has two. Prefill includes the first output head and
sample; decode uses the remaining 255 transitions, consistently for both servers.

| Engine/configuration | Depth | Prefill tokens/s | Decode tokens/s |
| --- | --- | ---: | ---: |
| PR1 F16 (3 runs) | 4K | 1410.50 [1385.82–1413.44] | 23.36 [23.30–24.58] |
| PR1 F16 (3 runs) | 16K | 1528.63 [1518.99–1529.11] | 23.22 [23.08–24.03] |
| PR1 F16 (3 runs) | 32K | 1241.56 [1239.93–1242.55] | 21.50 [20.92–22.74] |
| llama.cpp F16 (3 runs) | 4K | 425.29 [425.12–425.34] | 21.43 [21.31–21.54] |
| llama.cpp F16 (3 runs) | 16K | 474.23 [465.99–474.70] | 19.75 [16.96–20.52] |
| llama.cpp F16 (3 runs) | 32K | 445.21 [430.06–446.30] | 19.59 [18.77–19.70] |
| llama.cpp Q8_0 (2 runs) | 4K | 421.50 [421.45–421.56] | 21.45 [21.35–21.54] |
| llama.cpp Q8_0 (2 runs) | 16K | 470.34 [470.25–470.44] | 20.97 [20.69–21.25] |
| llama.cpp Q8_0 (2 runs) | 32K | 435.76 [429.67–441.86] | 20.63 [20.58–20.68] |

PR1 used context 40,000, F16 main/index KV, 80 CPU threads, 170 GiB arena,
7 GiB expert cache and explicit registered-memory budget 176 GiB. The F16
reference matched context 40,000 and thread counts 80/80; it used ubatch 2,048
and expert layers 10–12 on the GPU. The tuned-Q8 reference used context 65,536,
threads 64/96, Q8_0 KV, ubatch 2,048 and the same three GPU expert layers.
CUDA fusion was explicitly off in all compared processes. The Q8 configuration
is the historical tuning recipe under this measured environment, not a claim
that it is the fastest possible reference configuration.

Hardware was one RTX 5090 with a 440 W power limit, EPYC 9654 (96 cores,
192 logical CPUs) and about 624 GiB host RAM, on Ubuntu 26.04.1 / Linux
7.0.0-34, driver 595.91.07. A load snapshot confirmed PCIe generation 5, width
16. GPU1 retained a miner and two VLLM workers. Their presence is recorded;
the snapshot does not measure their effect on these results. Thread affinity
was not explicitly pinned. No compilation or conversion ran in the speed window.

That October 27 worker's decode medians were **14.9%, 14.0% and 15.0% below
historical v8** at 4K, 16K and 32K. The cause was unresolved in that window;
those runs did not establish a whole-model explanation or a measured fix.
The slower rows remain here. The October 28 comparison below tests the later
barrier fix against protected v8 in the same window. Prefill remained near
the historical values. All metric rows, model shard identities, environment and exact
worker hash are in
[`pr1-text.json`](../tools/m31/bench/receipts/pr1-text.json).

Collection is complete, but the two failed tools trials remain failed. Numerical,
cache, lifecycle and GSM checks passed within their stated coverage; this is not
an overall-PASS claim. HIP/SYCL/Windows builds and the Qwen executable limitation
also remain open review coverage.

## Generation follow-up: CPU barrier (2026-10-28)

The later worker keeps the callback's actual `nth` participant count and uses
CPU relax while waiting, matching v8: `pause` on x86, without an operating-system
scheduler yield or a spin-count threshold. Its arrival and generation counters
have v8's 64-byte spacing on x86. Quantized arithmetic and the acquire/release
synchronization contract are unchanged.

An initial pause-only build passed correctness but failed the predeclared
generation gate: its median difference from v8 was -3.77% at 4K and +0.83%
at 32K. That complete eight-run trial is retained. The accepted build adds
counter separation and passed the same gate: the absolute median difference
is below 3% at both depths, and neighboring A/B pairs have both signs.

| Depth | PR1 + barrier fix decode median [min–max] | Protected v8 decode median [min–max] | PR1/v8 |
| --- | ---: | ---: | ---: |
| 4K | 25.64 [24.87–26.05] | 25.28 [24.58–25.94] | +1.45% |
| 32K | 24.34 [23.62–24.64] | 24.51 [24.26–24.86] | -0.66% |

Units are tokens/s. The run order was A B B A B A A B, four repeats per
engine, with a fresh server and an excluded 1K/32-output warmup on each run.
Requests used the om_bench recipe: greedy, low reasoning, `ignore_eos=true`,
`cache_prompt=false` and 256 outputs; decode counts 255 transitions. Corpus
slices and nonces were identical across arms. Both used context 40,000, F16
main/index KV, 80 CPU threads, chunk 32,768, arena 170 GiB, expert cache 7 GiB,
registration budget 176 GiB, PACK8, upload depth8/stream1, poll50 and no affinity.
CUDA fusion was off. Hardware and toolchain were the EPYC 9654 / single RTX 5090
system described above. Neighboring services and miners on GPU1 stayed running;
CPU/process/memory and both-GPU telemetry was retained. This did not isolate
their causal effect or measure RAM bandwidth.

PR1 used the common server and native line protocol; protected v8 retained
its accepted standalone JSON/interactive-ACK server and stage profiler.
The result establishes the stated same-window gate, rather than universal
performance equivalence. The earlier unfixed same-window comparison was
-10.62%/-4.30%/-1.22% at 4K/16K/32K; those results are also retained.

The new CUDA worker SHA-256 is `938bfae60ef125f47834293fa8cfe55bda42af84b1c40e421e2a9fbea4321587`.
CPU and CUDA each passed all 10 model-free ctest groups. A fresh GPU-offload
oracle matched all 128 code128 logit rows bitwise, including 32 teacher-forced
decode transitions: top-1 100%, mean/max KL0. The numerical test is a short
diagnostic, not a repeat of the full gate20 or the long-context gates.
GSM8K100, tools23, 16K speed and long-context logits were not rerun on this
new worker. The October 27 quality failures and coverage limits remain.

All per-run rows, both candidates, the gate and hashes are in
[`pr1-barrier-followup.json`](../tools/m31/bench/receipts/pr1-barrier-followup.json).

## Historical results, not final PR1 acceptance

The following numbers belong to the protected port6 v8 executables and their
standalone adapter. They do **not** certify the cleaned engine or common-server
shim in this branch. The PR1 measurements above used its own CMake-built
executable and are reported separately.

Hardware: one RTX 5090, 32 GiB VRAM, PCIe 5.0 x16; AMD EPYC 9654, 96 cores;
624 GiB host RAM. GPU1 was not used by the candidate; the reference report records
a Pearl miner on GPU1. Candidate: 80 CPU workers, 32 host workers, eight packing
workers (the historical host-worker knob is inactive in stock MMQ), 170 GiB arena, 7 GiB expert cache, no cache seed, context 40,000,
F16 main/index KV, maximum chunk 32,768 and fusion off.

| Test | Historical Xolotl v8 | Historical llama.cpp PR #31192 |
| --- | ---: | ---: |
| Prefill 4K / 16K / 32K, tokens/s | 1390.85 / 1535.00 / 1246.65 | 425 / not measured / 433 |
| Decode 4K / 16K / 32K, tokens/s | 27.445 / 27.015 / 25.305 | 23.33 / 22.63 / 21.92 |
| GSM8K, first 100 test questions | 97/100 | 96/100 |
| Tool prompts plus continuations | 19/23 | 19/23 |

The v8 speed values are medians of two requests at each depth after warmup,
256 generated tokens, greedy sampling, `ignore_eos=true`, `cache_prompt=false`
and `reasoning_effort=low`. Raw v8 32K decode rates were 24.61 and 26.00 tokens/s.
The six retained speed rows were:

| Round | Actual prompt tokens | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: |
| 1, 4K | 4347 | 1378.4 | 26.87 |
| 1, 16K | 16635 | 1533.3 | 26.85 |
| 1, 32K | 33022 | 1249.2 | 24.61 |
| 2, 4K | 4352 | 1403.3 | 28.02 |
| 2, 16K | 16636 | 1536.7 | 27.18 |
| 2, 32K | 33019 | 1244.1 | 26.00 |

Two repetitions do not establish a statistical advantage. Actual 32K prefill
chunks were `10240/7936/6528/5632/2686` and
`10240/7936/6528/5632/2683`, rather than one 32K pass.

The reference was separately tuned: 64 decode threads, 96 batch threads,
context 65,536, Q8_0 KV, ubatch 2,048, expert layers 10–12 on the GPU and the rest
in RAM. These are comparisons of tested configurations, not matched KV/thread
settings. Issue #62 described a same-day comparison; the preserved baseline
report is dated 2026-10-26 and the v8 acceptance is dated 2026-10-27. The new
same-day PR1/reference measurements above provide the current comparison.

Historical gate20 was bitwise equal on all 28,723 positions; the selected long
rows were also bitwise equal. Canonical v8 SHA-256 values were
`2ab8b3c466becb4a8535b886e131d16a70ec5e07c995ac278599ff5de217859a`
for `m31_generate` and
`39a86da6bd1a601adbfe02ee8556278be5c7c6f87a2ca3b0cc3ea5d47a778472`
for its standalone worker. Those are historical identities, not this branch's
build hashes. Vision and short-prefill measurements are deliberately excluded.

## Reproduce quality and speed

The quality client uses Python's standard library. The speed client also uses
the server's tokenizer dependencies when loading a local M31 export; the logit
comparator needs NumPy. Use the same ordered GSM8K main test set for both
servers. The first 100 rows are used without shuffling. The preserved test file's
SHA-256 is `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
It matches the public dataset blob `e4c2ff4942b9a78bd74f04141224c11e28d12dc9`
at commit `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`.
The historical quality client SHA-256 was
`bca423fc5b57609337aff1cff5d931a328d728fa8fe0a199f4fa7f0e1222d286`;
the portable client's fixture AST and scoring predicates are preserved. For example, fetch the
public data separately and record its SHA-256:

```sh
curl -L https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl \
  -o gsm8k-test.jsonl
sha256sum gsm8k-test.jsonl
python tools/m31/bench/quality.py --url http://127.0.0.1:8080 \
  --dataset gsm8k-test.jsonl --suite all --out results-m31-quality
# The identical client and dataset against an already running reference server:
python tools/m31/bench/quality.py --url http://127.0.0.1:8081 \
  --dataset gsm8k-test.jsonl --suite all --out results-ref-quality
# Tools alone, including the three tool-result continuations:
python tools/m31/bench/quality.py --url http://127.0.0.1:8080 \
  --suite tools23 --out results-m31-tools23
# Controlled replications, retaining both seeds on each server:
python tools/m31/bench/quality.py --url http://127.0.0.1:8080 \
  --suite tools23 --seed 0 --out results-m31-tools23-seed0
python tools/m31/bench/quality.py --url http://127.0.0.1:8080 \
  --suite tools23 --seed 1 --out results-m31-tools23-seed1
python tools/m31/bench/quality.py --url http://127.0.0.1:8081 \
  --suite tools23 --seed 0 --out results-ref-tools23-seed0
python tools/m31/bench/quality.py --url http://127.0.0.1:8081 \
  --suite tools23 --seed 1 --out results-ref-tools23-seed1
```

Quality sends temperature 1, top-p 0.95, top-k 40, min-p 0.05 and maximum
6,000 tokens, with model-default thinking and no seed. The old client omitted
top-k/min-p and inherited 40/0.05 from both the v8 adapter and pinned llama.cpp.
The portable client sends them explicitly so the common server's defaults cannot
change this recipe. It preserves the historical questions,
tool definitions and scoring predicates. The 20 tool prompts and three
continuations are frozen in `bench/tool_fixtures.py`; local-looking paths there
are literal prompt data and are never read or executed. `raw.jsonl` retains every
request and full response. `summary.json` records client/data hashes, counts,
length stops, marker leaks and invalid tool IDs. Leak detection also covers
MiniMax's turn markers, namespaced tool prefix and `<mm:think>` tags; these are
additional API cleanup checks, rather than changed answer-scoring predicates. The thresholds are at least
95/100 GSM8K and 19/23 tools, with no length stops, leaked control markers or
invalid tool IDs. Quality sampling is stochastic; retain all outcomes. `--seed UINT64` is optional,
including 0, and records an explicit seed in the request and summary. Omitting it
keeps the historical no-seed request. A fixed seed controls sampling within an
engine; it does not make different kernels or batch histories numerically equal.

For speed, provide the same sufficiently long UTF-8 corpus to both servers.
The historical corpus SHA-256 was
`b3dd2af20842931b69a67cc3f80ae6cfe30c36c7df20a270895d9b6915d62067`.
That corpus is not distributed here; identify your own corpus for a fresh
comparison, or supply the retained corpus to reproduce that exact input:

```sh
python tools/m31/bench/speed.py --url http://127.0.0.1:8080 \
  --tokenizer "$TOKENIZER_DIR" --corpus corpus.txt --repetitions 3 --out results-m31-speed
python tools/m31/bench/speed.py --url http://127.0.0.1:8081 \
  --corpus corpus.txt --repetitions 3 --out results-ref-f16-speed
# Repeat against the separately launched tuned-Q8 configuration:
python tools/m31/bench/speed.py --url http://127.0.0.1:8081 \
  --corpus corpus.txt --repetitions 2 --out results-ref-q8-speed
```

This is the portable `om_bench` request recipe, with the corpus path supplied
explicitly. The common server has no `/tokenize` or `/detokenize` endpoints, so
its run uses the same exported `M31Tokenizer` locally. The reference run uses its
HTTP tokenizer endpoints. Both then send the same chat request recipe over HTTP.
The tokenizer-oracle check above must pass before treating their inputs as
comparable. Actual prompt counts
include template overhead. Store the corpus with your receipts; a different
corpus is a new measurement.

The speed client computes prefill as `1000 * prompt_n / prompt_ms`. It computes
decode as `1000 * (generated - 1) / predicted_ms` for **both** servers. The first
output head and first sampled token belong to prompt time; decode time covers
the remaining transitions. For 256 output tokens, the decode numerator is 255.
The common server's generic reported rate uses 256 here, while the reference
uses 255, so trusting those fields would introduce about 0.4% bias between the
servers. `results.tsv` keeps `raw_prefill_tps` and `raw_gen_tps` beside the normalized
rates; `raw.jsonl` retains the unmodified API response. `decode_steps` and
`decode_s` record the denominator. The historical v8 worker already reported
the `n-1` rate; its table above retains the original measured values.

Separate startup and registration from steady-state speed. Run reference and
candidate in isolated windows, without compilation or other CPU-heavy work.
Retain all repetitions and report their range.

## Reproduce GPU-offload logit checks

Build the reference in a **separate** checkout and process. Do not link its ggml
into Xolotl. The pinned reference and model revision are required to reproduce
the historical arithmetic.

```sh
git clone https://github.com/ggml-org/llama.cpp.git llama-m31-reference
git -C llama-m31-reference fetch origin refs/pull/31192/head
git -C llama-m31-reference checkout 6b95a9908d2c7b242817a447708aec38e937e2c0
git -C llama-m31-reference diff --exit-code
cmake -S llama-m31-reference -B llama-m31-reference/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DLLAMA_CURL=OFF
cmake --build llama-m31-reference/build --target llama-server -j 8
cmake -S tools/m31/tests/oracle -B build-m31-oracle -DCMAKE_CXX_COMPILER=g++-14 \
  -DM31_LLAMA_SOURCE="$PWD/llama-m31-reference" \
  -DM31_LLAMA_BUILD="$PWD/llama-m31-reference/build"
cmake --build build-m31-oracle --target m31_oracle m31_tokenizer m31_cache_oracle -j 8
python tools/m31/tests/check_tokenizer_oracle.py --model "$MODEL" \
  --oracle build-m31-oracle/m31_tokenizer
```

Set `MODEL` as above. The following commands write new output directories and
refuse to overwrite previous results. Gate20 requires about 22 GiB per set of
F32 logits for this vocabulary; reserve space for both sets.

```sh
python tools/m31/bench/make_prompts.py gate20-text
build-m31-oracle/m31_oracle tokenize "$MODEL" gate20-text gate20-ids
build-m31/m31_generate --model "$MODEL" --backend cuda --ctx 4096 \
  --chunk 2048 --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
  --register-arena 1 --register-limit-gib 176 --gpu-pack-threads 8 \
  --decode-tail 32 --rows all --cases gate20-ids --out gate20-native
python tools/m31/bench/replay_oracle.py --candidate gate20-native \
  --ids gate20-ids --oracle build-m31-oracle/m31_oracle --model "$MODEL" \
  --threads 80 --out gate20-oracle
python tools/m31/bench/compare_logits.py gate20-oracle/reference gate20-native \
  --require-suite --oracle-contract gpu-offload --require-bitwise --out gate20-comparison.json

python tools/m31/bench/make_prompts.py long-text --suite long --corpus corpus.txt
build-m31-oracle/m31_oracle tokenize "$MODEL" long-text long-ids
build-m31/m31_generate --model "$MODEL" --backend cuda --ctx 40000 \
  --chunk 32768 --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
  --register-arena 1 --register-limit-gib 176 --gpu-pack-threads 8 \
  --decode-tail 32 --head-tail 256 --rows last:256 --cases long-ids --out long-native
python tools/m31/bench/replay_oracle.py --candidate long-native --ids long-ids \
  --oracle build-m31-oracle/m31_oracle --model "$MODEL" --threads 80 \
  --capacity input-length --out long-oracle
python tools/m31/bench/compare_logits.py long-oracle/reference long-native \
  --oracle-contract gpu-offload --require-bitwise --out long-comparison.json
```

For a long test with no external corpus, `long-public` generates four deterministic
fixtures: Russian and code tasks, each at 16,384 and 32,768 tokens. Every source
text has 2,048 varied passages from the same task definitions as gate20. The
vocabulary-only oracle truncates it to the exact requested length. This adds
coverage beyond the historical corpus cases; it is not a reproduction of their
private source text.

```sh
python tools/m31/bench/make_prompts.py long-public-text --suite long-public
build-m31-oracle/m31_oracle tokenize "$MODEL" long-public-text long-public-ids
build-m31/m31_generate --model "$MODEL" --backend cuda --ctx 40000 \
  --chunk 32768 --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
  --register-arena 1 --register-limit-gib 176 --gpu-pack-threads 8 \
  --decode-tail 32 --head-tail 256 --rows last:256 \
  --cases long-public-ids --out long-public-native
python tools/m31/bench/replay_oracle.py --candidate long-public-native \
  --ids long-public-ids --oracle build-m31-oracle/m31_oracle --model "$MODEL" \
  --threads 80 --capacity input-length --out long-public-oracle
python tools/m31/bench/compare_logits.py long-public-oracle/reference long-public-native \
  --oracle-contract gpu-offload --require-bitwise --out long-public-comparison.json
```

The replay helper runs cases sequentially and copies the native chunk plan into
the oracle environment. It sets fusion off for the oracle process, then checks
capacity equivalence when requested. `--require-bitwise` asks a stricter question
than the numerical acceptance thresholds; keep both outcomes if identity fails.
The 5090 memory budgets above are examples. On a 4090, lower cache and chunk
budgets may be necessary; replay the resulting actual chunks and report the
changed recipe instead of claiming an identical hardware run.

### Reproduce the cache partition observation

The observatory input is public and deterministic. For each record `i`, its text is:

```text
Record {i}: The observatory measured clear skies over the northern river. Observer number {i % 71} entered the temperature and checked the previous report.
```

Records 0–6999 are joined with a newline. The tokenizer then keeps the first
257 or 1324 IDs. The following runs the two partitions separately and verifies
that each native head agrees with its matching oracle. All output directories
must be new.

```sh
python tools/m31/bench/make_prompts.py cache-text --suite cache-observatory
build-m31-oracle/m31_oracle tokenize "$MODEL" cache-text cache-ids
for case_spec in 257:256 1324:1024; do
  case_length=${case_spec%:*}
  reuse_tokens=${case_spec#*:}
  for recipe in whole split; do
    if [ "$recipe" = whole ]; then chunk_limit=32768; else chunk_limit=$reuse_tokens; fi
    build-m31/m31_generate --model "$MODEL" --backend cuda --ctx 40000 \
      --chunk "$chunk_limit" --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
      --register-arena 1 --register-limit-gib 176 --rows last --head-tail 1 \
      --ids-file "cache-ids/cache_observatory_$case_length.i32" \
      --out "cache-native-$case_length-$recipe"
    python tools/m31/bench/replay_oracle.py \
      --candidate "cache-native-$case_length-$recipe" --ids cache-ids \
      --oracle build-m31-oracle/m31_oracle --model "$MODEL" --threads 32 \
      --out "cache-oracle-$case_length-$recipe"
    python tools/m31/bench/compare_logits.py \
      "cache-oracle-$case_length-$recipe/reference" "cache-native-$case_length-$recipe" \
      --require-bitwise --out "cache-matched-$case_length-$recipe.json"
  done
  build-m31/m31_cache_model_check --model "$MODEL" --backend cuda \
    --ids-file "cache-ids/cache_observatory_$case_length.i32" \
    --length "$case_length" --reuse "$reuse_tokens" --ctx 40000 --chunk 32768 \
    --threads 80 --arena-gib 170 --gpu-experts-gib 7 \
    --register-arena 1 --register-limit-gib 176 --checkpoint-mib 512 \
    --decode-steps 7 --out "cache-state-$case_length"
  for decode_steps in 0 7; do
    build-m31-oracle/m31_cache_oracle --model "$MODEL" \
      --ids-file "cache-ids/cache_observatory_$case_length.i32" \
      --reuse "$reuse_tokens" --ctx 40000 --chunk 32768 --threads 32 \
      --decode-before-rewind "$decode_steps" \
      --out "cache-history-$case_length-$decode_steps"
  done
done
```

Compare live-history oracle heads with the corresponding native public-API
cache check. The strict assertions below apply to the same history. The
whole-versus-split comparison is printed as a separate diagnostic.

```sh
python - <<'PY'
import json
from pathlib import Path
import numpy as np

for length in (257, 1324):
    state = Path(f"cache-state-{length}")
    summary = json.loads((state / "summary.json").read_text())
    assert summary["same_history_bitwise"]
    for steps, native_head in ((0, "whole_rewind"), (7, "whole_decode_rewind")):
        native = np.fromfile(state / (native_head + ".f32"), dtype="<f4")
        oracle = np.fromfile(f"cache-history-{length}-{steps}/logits.f32", dtype="<f4")
        assert native.size == oracle.size == 200064
        assert np.isfinite(native).all() and np.isfinite(oracle).all()
        assert np.array_equal(native.view("u4"), oracle.view("u4"))
    print(length, "same-history: bitwise", "whole-vs-split:", summary["whole_vs_split_bitwise"])
PY
```

A whole-versus-split difference is not an assertion failure in this diagnostic:
those are the two intentionally different computations being observed. A
same-history difference fails the cache check. If VRAM pressure changes the
planner's recorded chunks, choose a smaller explicit chunk for both sides and
record the changed experiment. These commands exercise model-sized GPU work;
they are separate from the small default CPU tests.

For the measured F16 reference configuration, launch the reference in its own
window, then use the three-repeat reference speed command above:

```sh
env -u CUDA_DEVICE_MAX_CONNECTIONS -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
  GGML_CUDA_DISABLE_FUSION=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  llama-m31-reference/build/bin/llama-server --model "$MODEL" \
  --host 127.0.0.1 --port 8081 --alias minimax-m31 --parallel 1 \
  --ctx-size 40000 --batch-size 2048 --ubatch-size 2048 \
  --threads 80 --threads-batch 80 --n-gpu-layers 999 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 --fit off \
  --temp 1 --top-p 0.95 --top-k 40 --min-p 0.05 \
  --load-mode none --override-tensor 'blk\.(10|11|12)\.ffn_.*_exps\.=CUDA0,ffn_.*_exps\.=CPU'
```

For the tuned-Q8 reference, use a separate window with the following settings.
This is the arm used for the reference quality trials and two-repeat speed run:

```sh
env -u CUDA_DEVICE_MAX_CONNECTIONS -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
  GGML_CUDA_DISABLE_FUSION=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  llama-m31-reference/build/bin/llama-server --model "$MODEL" \
  --host 127.0.0.1 --port 8081 --alias minimax-m31 --parallel 1 \
  --ctx-size 65536 --batch-size 2048 \
  --ubatch-size 2048 --threads 64 --threads-batch 96 --n-gpu-layers 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --fit off \
  --temp 1 --top-p 0.95 --top-k 40 --min-p 0.05 \
  --load-mode none --override-tensor 'blk\.(10|11|12)\.ffn_.*_exps\.=CUDA0,ffn_.*_exps\.=CPU'
```

The same-day comparison explicitly disables CUDA fusion in both engine
processes. The historical best preset did not retain a complete environment
receipt, so its actual fusion setting cannot be established. The Q8 row uses
the historically tuned context/thread/KV/placement settings under this explicit
new environment; it is not a claim of the fastest possible reference setup.
Tiled MM remains enabled by default, with its force override unset.

The F16 and tuned-Q8 results above remain separate configurations. For new
comparisons, record source and dependency
commits, executable hashes, all model shard hashes, compiler/toolkit, effective
configuration, GPU clocks/power, CPU placement and other machine load with the
results. The recorded outcomes for these recipes are in the PR1 candidate
checks and receipt above; new hardware needs its own measurements.

## Limits

Numerical acceptance covers the 20 short prompts and selected rows at 16K and
32K, not every possible input or every row above 32K. The model's advertised
context capacity is not a tested port capacity. F16 main and index KV consume
substantial VRAM; memory planning can split a prompt into several expert passes.
Arena population and optional host registration add startup work and require
free host memory. A single active sequence is supported. Feature-specific build
requirements and validation are stated in the vision and short-prefill sections
when those follow-ups are present. Speculative decode and the Q2 experiments
are not included. HIP and SYCL toolchains were unavailable, so those backends were
not built. Windows, other GPU models, two-GPU placement and other-model inference
were not tested. The two tools-trial failures and Qwen link failure remain explicit limitations.
The October 28 generation gate covers only the stated hardware and two depths;
it does not replace the original slower measurements or the untested coverage.
