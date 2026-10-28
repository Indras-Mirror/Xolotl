# Porting Basalt engine work into Xolotl (survey, 2026-10-29)

[Basalt](https://github.com/jesdga95/basalt) (MIT, 276 stars) is a local inference engine for
Qwen3.8-Flash-Next on NVIDIA Blackwell. It is **not a separate codebase that happens to solve the
same problem** — it is the same engine lineage as this repo's `src/` tree, forked and renamed
(`strata::` → `basalt::`, `include/strata/…` → `include/basalt/…`). Several files it patches are
the same file here, line for line:

| Basalt file | Xolotl counterpart | Status |
|---|---|---|
| `src/kernels/cpu/iq_avx512.cpp` | `src/kernels/cpu/iq_avx512.cpp` | same file, 298 vs 273 lines |
| `src/prefill/prefill.cpp` | `src/prefill/prefill.cpp` | same file |
| `src/core/verify.cpp` | `src/core/verify.cpp` | same file |
| `src/kernels/cpu/pool.cpp` | `src/kernels/cpu/pool.cpp` | same file |
| `src/kernels/cuda/qsa_select.cu` | `src/kernels/cuda/qsa_select.cu` | same file |
| `src/platform/memory.cpp` | `src/platform/memory.cpp` | same file |
| `src/core/expert_source.cpp` | `src/core/expert_source.cpp` | same file |
| `src/kernels/cpu/nvfp4_avx512.cpp` | `src/kernels/cpu/nvfp4_avx512.cpp` | same file |
| `src/kernels/cuda/dec_moe.cuh` | `src/kernels/cuda/iq_kernels.cu` | different layout |

**This distinction decides the port strategy.** For the Qwen3.8-Flash-Next / GLM engines built from
`src/`, a port is mostly a namespace/header rewrite and can be reviewed hunk by hunk. For the
DeepSeek-V4.1 engine (`tools/v41/engine/`, a separate implementation) nothing copies: those items
have to be re-derived against that engine's own structures.

Two axes matter more than size:

- **Bit-identity.** A change that produces identical output on a same-path A/B is cheap to accept:
  it cannot regress quality, only speed. Most kernel items below are in this class, and each says so.
- **Where it pays.** Decode on this box is dominated by expert-tier file/RAM waits, not arithmetic
  (measured elsewhere in this tree: 79 % of a decode step is file-read wait on a 152 GB model). A
  kernel that speeds up arithmetic moves a small fraction of the step. Prefill and the CPU pool are
  where the remaining headroom actually is.

## Survey

| # | Change | Class | Portability, verified by reading the target |
|---|---|---|---|
| 98 | pool: a batch ends with its last job; a parked worker yields after a short spin | output unchanged | **PORTED** — same file, same epoch-tagged protocol |
| 107 | prefill chunk size priced from the link and non-resident bytes | same output on the measured placement | **PORTED** (adapted: the pricing, not the constant) |
| 111 | AVX-512 sign masks via `pdep`, IQ3_S/IQ2_S grid gathers | bit-identical | Code-portable, **unverifiable here** — see the CPU note |
| 74 | NVFP4 CPU expert rows at 1–2 tokens | bit-identical (documented) | Code-portable, **unverifiable here** — same reason |
| 112 | Q2_0 expert down in 64-row blocks, 4 lanes a row | bit-identical | **Concept only**: our `native_down_multi_kernel` is a different kernel |
| 114 | elastic KV tail keeps no RAM copies of its experts | output unchanged | Concept is present (`lend_from` region) — a real port, needs work |
| 105 | QSA window of 5–8 queries as two rows of four | bit-identical scores | **Not portable**: our QSA scorer is a different design |
| 101 | CPU warm-up on mixed-format packs, 3 likeliest experts | output unchanged (reads only) | **Not portable**: the feature does not exist here |
| 113 | RAM guard sheds and waits instead of stopping the request | behaviour change (serve) | **Not portable**: no RAM guard exists here |
| 13 | APR trellis types (TQ2_T…TQP7) + Hadamard-folded experts | new formats | Large (+2757), a format decision |
| 100 | lazy L2 prefetch also takes the next layer's FFN read matrices | output unchanged | **Skip** — below the author's own noise floor |

**Read the last four rows with care.** "Not portable" does not mean "not useful" — it means the
change targets something this tree does not have, so it is a new feature to design, not a patch to
apply. That is a much larger and riskier job than a port, and each one needs its own decision
(does the Qwen/GLM engine even want a CPU-tier warm-up? does it want a RAM guard?), so none of them
should be smuggled in under "porting Basalt work".

**The CPU note that decides two items.** This box is a **Ryzen 7 5700X (Zen 3): it has no AVX-512
at all.** So `iq_avx512.cpp` and `nvfp4_avx512.cpp` never execute here, `cpu_avx512_ok()` is false,
and the AVX2 path runs instead. Items 111 and 74 can be written but cannot be built into a tested
state, run, or timed on this machine — a port there would be an unverifiable claim. They also carry
a second warning: our own `iq_avx512.cpp` records that AVX-512 gathers measured **3–5 % slower**
than scalar lookups on a Zen 4 part, which is why ours are behind `STRATA_IQ_GATHER`. Basalt's
unconditional gathers are tuned for their hardware, not this one.

## What landed

- **98** (`8a644809`) — pool: the batch ends with its last job; a parked worker yields its core after
  50 µs. The park barrier was redundant once claims became epoch-tagged, which this header's own
  comment already argued. `wait_parked()` is deleted. `pool.cpp` syntax-checks clean.
- **107** (`511b4d45`) — prefill: `stream_all_min_for(bytes, gbps)` in a new
  `include/strata/prefill/stream_all.hpp`, wired opt-in (`STRATA_PREFILL_STREAM_PRICED=1`) through
  the existing `probe_pcie_h2d_gbps()`. It only ever *raises* the threshold; 1024 stays the measured
  floor, and an armed CPU share still wins. Its test (`tests/prefill/stream_all_test.cpp`) **passes
  under ctest in the full build**.
- **112** (`57474e19`) — Q2_0 expert down at 64 rows a block, 4 lanes a row. Its gate
  `q2_down_wide_parity` calls `native_expert_grouped` twice on identical inputs and compares bytes:
  **10,240/10,240 bit-identical across 5 seeds**, and it **passes under ctest in the full build**.
- Full CUDA build of `ds41_generate` with all three: **313/313, exit 0**.

`kRoutedExtraMs` (0.24) is Basalt's constant, not ours, which is why 107 ships opt-in.

### What #98 could NOT be verified here (2026-10-29)

The gate for #98 is `pool_stress` - it drives hundreds of thousands of 0-4 job batches
through `run()` and `run_split()` with gaps long enough for the workers to sleep, which is
exactly the late-wake race that removing the park barriers could reintroduce. **It cannot
run on this box.** `src/kernels/cpu/expert.cpp` is compiled with `-mavx512f -mavx512bw
-mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi`, this CPU is a Ryzen 7 5700X (Zen 3, no
AVX-512), and the test therefore skips at `cpu_features().usable()`.

Tried and rejected: enabling the scalar oracle path (`expert_set_oracle_q8_0(true)`) so the
protocol could run without AVX-512 kernels. The oracle functions are plain C++, but the
translation unit is compiled with AVX-512 enabled, so the compiler is free to vectorise
them - the binary died with **SIGILL (exit 132)** rather than running. That change was
reverted; a crash is worse than an honest skip.

So the status of #98 is: **compiles in a full CUDA build, and its protocol is unchanged by
reasoning, but the stress gate has not been run.** It needs a machine with AVX-512 (any
Zen 4/5, Skylake-X or newer) or a build of `kernels_cpu` with the ISA flags relaxed. Until
then the timing question (is dropping 3-4 barriers per layer worth anything?) is also open.

`pool_tasks_test` DID run (168 bitwise comparisons passed) and `pool_test` also skips for the
same AVX-512 reason - so the ctest line "4/4 passed" is misleading: two of the four passed
vacuously. Exit 0 from a skipping test is not evidence.

**114 was attempted and deferred, on evidence.** It is the one remaining item with a real target
(`choose_resident_keep_from` and `resident_lent_slots` both exist here), but it needs three
prerequisites this tree does not have: `exp_flag()` (Basalt's experimental-flag system),
`set_spares()`, and `home_given_up` — the last being the mechanism the PR is actually about (the
copy-less experts taking RAM homes at admission). Writing that blind, into the expert-residency hot
path, with no way to build or test until the box is free, is how an engine gets a defect nobody
finds for a week. It wants its own session with a working build.

Details that change how each item should be taken:

- **107 must be reconciled, not copied.** Its pricing formula replaces a constant default. Our
  `stream_all_min()` also serves `STRATA_PREFILL_CPU_SHARE`/`arm_cpu_share`, which Basalt does not
  have: below the threshold we hand some non-resident experts to the idle decode CPU pool instead of
  streaming them. Port the *pricing* (link rate × non-resident bytes, between floor and default) and
  keep our share logic on top. A straight copy silently deletes the share.
- **100 is below the noise floor.** Its own author measures +0.32 % against a 0.39 % box noise floor
  and filed it as a draft asking whether to keep it. Not worth the review.
- **111 needs `-mbmi2`** for `_pdep_u64`/`_pext_u64` on GCC/Clang. Check the build flag lands on the
  right target only; a global `-mbmi2` would raise the binary's CPU floor.
- **112 is not a file-level copy.** Basalt's `dec_moe.cuh` glue layer does not exist here; the
  equivalent launch site is `iq_kernels.cu:2883`. The kernel body ports; the call site has to be
  re-derived against `native_down_multi_kernel`.
- **13 is a format decision, not a speed patch.** It adds six ggml types (144–149) and a Hadamard
  fold. Worth attention only if a Gyro-L/APR pack is actually wanted; it is not a drop-in.
- **Basalt #68** ("an image is a data URL unless a host or a directory is allowed") is a
  security hardening unrelated to speed and should be judged on its own, separately from this list.

Repo state at survey time: `Indras-Mirror/Strata`, `Indras-Mirror/Strata-GLM`, `mab776/Strata-DS4`,
`musicurgy/delta-inference-engine` and `NektoDron/Xolotl` had **no open pull requests**. Basalt had
26 open. This survey covers them.

## Method

Ports land one at a time on `basalt-ports`, each with the bar its class demands:

- **bit-identical items** are accepted only on a same-path A/B producing identical output — for the
  kernels that means the existing parity harnesses (`native_expert_parity`, `iq_avx512_test`,
  `qsa_scores_parity`), not a tolerance.
- **output-unchanged items** (pool, warm-up, prefetch) are accepted on identical token streams over a
  real model *and* a timing delta that clears this box's noise floor. A delta inside the floor is not
  evidence of a win, and will be reported as "no measurable change", not as an improvement.

Two bars stay separate: bit-exact for same-path arms, the ~1e-2 relative class only for arms that
genuinely take different paths (cpu-vs-gpu expert compute). Collapsing them into one number hides
which question was asked.
