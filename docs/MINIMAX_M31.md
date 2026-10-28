# MiniMax-M3.1-Flash text engine

This is the text slice of the community port discussed in [issue #62](https://github.com/Indras-Mirror/Xolotl/issues/62).
It adds an opt-in engine in `tools/m31` and a shim to Strata's common HTTP server.
Images, video, DSpark and adaptive short prefill are outside this slice.
`GENI` and `BGENI` return `ERR vision unavailable`.

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
options. Algorithm choices follow accepted v8 without setting `M31_*` variables.
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
F16 K/V and CUDA fusion disabled. CUDA prefill uses stock grouped MMQ; decode uses
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
free host memory. A single active sequence is supported. Image/video input,
adaptive CPU/GPU short prefill, speculative decode and the Q2 experiments are
not included. HIP and SYCL toolchains were unavailable, so those backends were
not built. Windows, other GPU models, two-GPU placement and other-model inference
were not tested. The two tools-trial failures and Qwen link failure remain explicit limitations.
The October 28 generation gate covers only the stated hardware and two depths;
it does not replace the original slower measurements or the untested coverage.
