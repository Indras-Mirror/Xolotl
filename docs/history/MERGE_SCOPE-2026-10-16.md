# Xolotl — scoping the unification of the Strata engine family

**Date:** 2026-10-16 · **Measured with:** `git merge-tree` (read-only) + `git diff --numstat` on the fork's own refs
**Verdict: feasible.** It is **two merges** — within the fork (13 files) and across lineages to the newest Qwen
(48 files) — about **60 files total, ~50 of them code**, four of them conflicting in both. A reconciliation, not a
rewrite. The cross-lineage half is only computable after a graft, which §3 shows exists.

---

## 1. Inventory — what each folder actually is

The "so many folders" is mostly illusion: **five are clones of ONE fork (different branches), two are clones of
ONE upstream, and four are not repos at all.**

| folder | size | what it is |
|---|---|---|
| `~/AI/Strata-DS4` | 29 G | clone of **fork** `Indras-Mirror/Strata-DSV4.1`, branch **`deepseek4`** ← the DSV4.1 work |
| `~/AI/Strata-GLM` | 2.2 G | clone of **fork**, branch **`glm`** |
| `~/AI/Strata-GLM-rebase` | 327 M | clone of **fork**, branch **`glm-rebase`** (= the merge base, see §3) |
| `~/AI/Strata-GLM-um` | 1.1 G | clone of **fork**, branch **`glm-upstream`** (the attempted graft — see §3) |
| `~/AI/Strata-MiMo` | 4.0 G | clone of **fork**, branch **`mimo`** |
| `~/AI/Strata` | 1.1 G | clone of **upstream** `Niko1221/Strata`, branch `main` |
| `~/AI/Strata-qwen` | 902 M | clone of **upstream**, branch **`qwen`** ← the newer Qwen lineage |
| `~/AI/strata-dense-runtime` | 14 M | **not a repo** — a copy of the `serve/` Python runtime (same files as the repo's `serve/`) |
| `~/AI/strata-dense-build` | 891 M | **not a repo** — a build dir; `CMakeCache.txt` says its source was `/tmp/strata-dense-eval` (ephemeral, gone) |
| `~/AI/v41-dev` | 82 G | **not a repo** — DSV4.1 scratch: bench runs, logs, decode sweeps |
| `~/AI/Strata-DS4-private` | 1.1 M | **not a repo** — `v41/` notes + a route-probe corpus |

So there are exactly **two** source repos: the fork (5 branches) and upstream (2 branches).

## 2. The shared core is already shared

All engines sit on one core in the same repo:

```
src/{artifact,core,kernels,ngram,plan,platform,prefill,program,spec}   +   include/strata/
```

**Engines on top of that core (4 of them):**

| engine | path | LOC | branch | status |
|---|---|---|---|---|
| DSV4.1 (v41) | `tools/v41/engine/ds41_*` | 12,597 | `deepseek4` | **current** (the launcher runs it) |
| GLM | `tools/glm/glm_*` | 2,636 | `glm` | current |
| old DS4 "slice" | `tools/ds4/ds4_*` | — | all | **legacy** — see §6 |
| Qwen / base runtime | `serve/` (Python) + `src/` core | — | `main` | current |

## 3. Topology — measured, not assumed

- `main` (Qwen/base) **is an ancestor of `deepseek4`, `glm` AND `mimo`.** → the unified branch is simply
  `merge(deepseek4, glm, mimo)`; Qwen rides along free.
- `git merge-base deepseek4 glm == glm-rebase HEAD == fb58e0db`. The repo was **staged for this merge**.
- `git merge-base deepseek4 mimo == 2b959ed5`.
- **Upstream `Niko1221/Strata` is a SEPARATE history.** The fork's root is `01529aaa`; Niko1221's is `a7908053`;
  `git merge-base --is-ancestor upstream/main main` → **NO**. A normal merge would need
  `--allow-unrelated-histories`.
- **The existing graft did NOT work.** `Strata-GLM-um`'s HEAD claims *"glm: merge upstream/main (Niko1221) into
  glm — graft recipe connects the histories"*, but `git diff --stat glm glm-upstream` = 57 files, **+258/-3217**,
  and **0 new paths** — it brought in no upstream content and net-deleted. Treat `glm-upstream` as **not** a
  successful upstream merge; do not build the plan on it.
- **THE GRAFT WORKS — the two histories CAN be joined (found 2026-10-16).** The fork's root commit `01529aaa`
  has tree `4591e4cafcc8`, which is **exactly** the tree of upstream commit **`f2a08d4b`** (2026-09-24, *"Strata:
  Qwen3.8-Flash-Next on a 12-24 GB NVIDIA GPU…"*). The fork is a re-rooted copy of upstream at that point, so:
  ```bash
  git replace --graft 01529aaa f2a08d4b     # join the histories (local replace ref; `git replace -d 01529aaa` undoes)
  ```
  After the graft, `merge-base` resolves for the first time — **`up/qwen ^ deepseek4` = `up/dense-eval ^
  deepseek4` = `fb58e0db`** (2026-10-08). See §4b for the resulting conflict surface.
- **`strata-dense-build`'s source was `/tmp/strata-dense-eval` (gone)** — the dense engine is the upstream `src/`
  core + `serve/` runtime, not a second engine. `~/AI/Strata-qwen/engine/strata` is an **untracked built binary**,
  which is why it looked like one.
- **Correction to an earlier draft of this doc:** using `git diff up/main main` I first read the Qwen merge as
  free. That was **wrong** — `main` is not a local ref in this clone, so the command printed `fatal:` and I had
  swallowed stderr. The newest Qwen is a real 47-48 file merge (§4b).

## 4. Conflict scope — the real work

`git merge-tree --write-tree --name-only` (read-only, no checkout):

### `xolotl = merge(deepseek4, glm)` → 10 files, 33 hunks

| file | hunks |
|---|---|
| `src/kernels/cuda/iq_kernels.cu` | **15** |
| `CMakeLists.txt` | 5 |
| `README.md` | 5 |
| `src/kernels/cuda/native_mmvq.cu` | 5 |
| `src/kernels/cpu/pool.cpp` | 2 |
| `tools/ds4/ds4_moe.hpp` | 2 |
| `include/strata/kernels/iq_kernels.hpp` | 1 |
| `include/strata/kernels/cpu/native_expert.hpp` | 1 |
| `include/strata/kernels/cpu/pool.hpp` | 1 |
| `include/strata/artifact/gguf_reader.hpp` | 1 |

### `+ merge(mimo)` → 7 more files, 46 hunks

| file | hunks |
|---|---|
| `tools/ds4/ds4_moe.cpp` | **25** |
| `src/kernels/cuda/iq_kernels.cu` | 9 |
| `tools/ds4/ds4_moe.hpp` | 5 |
| `src/kernels/cpu/native_expert.cpp` | 3 |
| `CMakeLists.txt` | 2 |
| `include/strata/kernels/iq_kernels.hpp` | 1 |
| `src/prefill/moe_mmq.cu` | 1 |

**Raw diff size is misleading** — `glm`'s +1.6 M lines is one file
(`bench/glm-2026-10-08/upstream/tokenizer.json`, 1.44 M lines of benchmark output). The tracked **code is only
34 MB / 1,039 files**; 19 GB of the repo is `bench/` artifacts.

## 4b. The cross-lineage merge (newest Qwen / dense) — 47-48 files

*Only computable after the graft (§3).* **Before** the graft merge-base is empty, so `git merge-tree` reports
every file as added on both sides. **After** it:

| merge | conflicted files |
|---|---|
| `up/qwen` (newest Qwen, 2026-10-13) vs `deepseek4` | **47** |
| `up/dense-eval` (newest dense port, 2026-10-13) vs `deepseek4` | **48** |

The 48 (dense-eval), minus docs: the real work is **~38 code files** —

```
cmake/hip_backend.cmake          CMakeLists.txt                 setup.py
include/strata/core/expert_source.hpp      include/strata/kernels/verify_kernels.hpp
include/strata/prefill/gemm.hpp            include/strata/prefill/prefill.hpp
src/core/{expert_source.cpp,layer.cpp,verify.cpp}
src/kernels/cpu/{iq_avx2.cpp,native_expert.cpp}
src/kernels/cuda/{iq_kernels.cu,qsa_decode_attn.cu,verify_kernels.cu}
src/prefill/{moe_mmq.cu,prefill.cpp}
serve/{frontend.py,server.py,test_parallel.py,test_server.py}
docs/{AMD_HIP,BATCHING,COMMUNITY_BENCHMARKS,DETAILS,INTEL,MULTI_GPU,UNSLOTH_Q4}.md + README{,.ja}.md
```

**Note the overlap with §4:** `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cpu/native_expert.cpp`,
`src/prefill/moe_mmq.cu` and `CMakeLists.txt` conflict in **both** merges. Resolve each once, together.

**Graft durability (open):** `git replace` refs are local and do **not** push. A pushed merge would still carry
real parents, but a later `merge-base` would see unrelated histories again unless the graft is made permanent
(`git filter-repo --replace-refs` / a re-rooted import) or the merge is done as an explicit
`--allow-unrelated-histories` content merge. Decide before pushing.

### GOTCHA, found the hard way (2026-10-16): the graft creates a CRISS-CROSS

Once the graft is in place, `git merge-base --all xolotl glm` returns **TWO** bases (`fb58e0db` **and**
`a6935457`). Git then synthesises a *virtual* merge base by merging the two, and **that virtual base can itself
contain conflict markers** — verified: the stage-1 blob of `serve/server.py` literally contained
`<<<<<<<<< Temporary merge branch 1`. The result is **spurious conflicts on files glm never touched** (the plain
merge of glm reported **51** conflicts, including all of `serve/` where `git diff fb58e0db origin/glm -- serve/`
is *empty*).

**Fix:** the intra-fork merges (`glm`, `mimo`) do NOT need the graft — their single base is `fb58e0db`. Merge them
with a single-base strategy:

```bash
git merge --no-commit --no-ff -s resolve origin/glm   # 13 conflicts, not 51 - the real set
```

**Better, and the recommended order (supersedes §9):** do the intra-fork merges **first**, on plain `deepseek4`
(no graft in play), and apply the graft + the cross-lineage merge **last**. Stacking the upstream merge on top of
the graft first, as this branch currently does, makes every later merge pay the criss-cross tax.

`git -c core.useReplaceRefs=false` does **not** avoid it — the second base comes from the merged upstream parent,
not from the replace ref alone.

## 5. The three hard parts

1. **`src/kernels/*` — the quantisation kernels.** Both branches optimised the same IQ/MMVQ kernels for their
   model. This is the genuine, knowledge-heavy conflict. Strategy: reconcile **per kernel function**, not per
   file; gate each on the existing parity tests (`src/kernels/*_parity.cpp`) — the harness is already there.
2. **`tools/ds4/ds4_moe.*` — legacy engine, 30 of the 46 mimo hunks.** See §6.
3. **`CMakeLists.txt`** — must enable all engines at once. No branch currently references more than one; this is
   the single integration point (`.cmake` slices already exist per engine).

## 6. Dropping the legacy engine removes most of the conflict

`tools/ds4/ds4_*` is the **old** "slice" dense engine (still built by `tools/ds4/cmake/ds4_dense.cmake`), while
`deepseek4` moved to `tools/v41/engine/ds41_*`. Mal: *"any old DS4 stuff isn't necessarily needed, now that we've
got DSV4.1 working well."* If `tools/ds4/ds4_*` is dropped from the unified tree (take `deepseek4`'s side /
delete), **~30 of the 46 mimo hunks and both glm `ds4_moe.hpp` hunks disappear** — leaving ~13 files of real work,
almost all in the kernels. **Decision needed from Mal**; I have not deleted anything.

## 7. Feasibility, disk, and what I did

- **Disk is the only hard blocker:** `/` is at **97%, 26 G free**, and `Strata-DS4` is 29 G. A normal clone is
  impossible. Because the *code* is only 34 MB, this repo was created as a **sparse clone excluding `bench/`**:
  `~/AI/Xolotl` is **122 MB**, and disk free was unchanged after creating it.
- `~/AI/Xolotl` is a clone **with history** (can push later), branch **`xolotl`** at `deepseek4`'s HEAD,
  `origin` = `~/AI/Strata-DS4`, `glm`/`mimo` reachable as `origin/glm` / `origin/mimo`.
- **Nothing was merged and nothing was deleted** — this file plus `tools/xolotl/merge-xolotl.sh` are the whole
  scaffold. The tree is clean at `xolotl` = `deepseek4`.

## 8. Risks / unknowns

- **Unverified:** whether GLM's and mimo's kernel changes are *compatible* (both may have changed the same
  IQ3/IQ2 decode numerics for different models). Only resolving the hunks and running the parity tests answers it.
- **Unverified:** that the merged tree actually *builds* all engines — the CMakeLists integration is untried.
- **Unknown:** whether "Qwen (dense)" also means *newer* `Strata-qwen` work (2026-10-13) beyond the fork's `main`
  (2026-10-04). If so that is an extra "sync upstream" job with unrelated-history cost, kept out of scope here.
- MiMo is lowest priority and contributes the largest single conflict file; it can be merged last or skipped.

## 9. Recommended sequence

1. In `~/AI/Xolotl`, branch `xolotl` (already at `deepseek4`): **`git replace --graft 01529aaa f2a08d4b`**, then
   **merge `up/dense-eval`** (or `up/qwen`) → resolve 48 files. Biggest single step; do it first so later merges
   sit on the newest base. Resolve `iq_kernels.cu` / `native_expert.cpp` / `moe_mmq.cu` / `CMakeLists.txt` here.
2. **Merge `origin/glm`** → 10 files / 33 hunks (kernels first, CMakeLists last). Only the leftover hunks remain.
3. **Merge `origin/mimo`** → 7 files / 46 hunks — optional/lowest priority; drop `tools/ds4/ds4_*` first if Mal
   approves (§6), which removes ~30 of the 46.
4. Build all engines; run `src/kernels/*_parity` + each engine's golden tests. Then a serve/throughput leg per
   model (GLM flash, Qwen 27b dense, DSV4.1, Qwen Flash) — the acceptance test for "one engine, four models".
5. **Decide graft durability before pushing** (§4b): `git replace` does not push. Make it permanent or redo the
   merge as `--allow-unrelated-histories`.

## 11. Prior art — this is a solved design problem (searched 2026-10-16)

The established pattern for "one engine, N architectures" is a shared tensor/kernel runtime plus a
**per-architecture model implementation registered behind one interface**, with the architecture selected from the
model file's own metadata. Not one hand-merged binary.

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** — the reference. ggml tensor lib → `libllama`
  runtime → one graph builder per architecture (`src/models/*.cpp`) → one server. This project already **vendors
  llama.cpp** (`third_party/llama.cpp`), and `src/` + a per-model engine is the same shape, half-applied.
- **[Inferflow](https://inferflow.github.io/)** — serves most common transformer models by *editing a config*,
  no code written. The config-driven end of the spectrum.
- **[Candle](https://www.local-llm.net/tools/)** (HuggingFace, Rust) — **47 architecture families**, format-agnostic
  loading (SafeTensors/GGUF/PyTorch). Proof the registry scales.
- **[vLLM](https://inference.net/content/vllm-advanced-custom-inference-pipelines/)** — "registering novel model
  architectures **without forking** vLLM" — the same registry idea.
- **[TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/cpp-runtime-overview.html)** —
  NVIDIA's C++ runtime, same layering.

**Implication for Xolotl:** keep `src/` (kernels/prefill/core) as the runtime and make each engine a **registered
architecture** behind one interface, chosen by the model's metadata (GGUF `general.architecture`) — then the four
models share one serve path *by construction* and a fifth costs one file, not a fork. The merge in §9 is still the
prerequisite: the engines must first coexist in one tree.

## 10. Reproduce every number here

```bash
cd ~/AI/Strata-DS4
git merge-tree --write-tree --name-only deepseek4 glm   | grep '^CONFLICT'
git merge-tree --write-tree --name-only deepseek4 mimo  | grep '^CONFLICT'
for b in deepseek4 glm mimo; do git merge-base --is-ancestor main $b && echo "main<=$b"; done
git diff --stat glm glm-upstream            # the graft that brought nothing in
```
