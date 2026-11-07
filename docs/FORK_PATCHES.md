# Where Xolotl patches upstream Strata, and how each conflict was settled

The routine is in [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md): `tools/sync-upstream.sh` merges `niko/main` into a fresh
branch in its own worktree, `git rerere` replays every resolution recorded below, and `tools/lost_lines.py`
checks that every line Xolotl added since the merge base is still in the merged tree.  `public` moves only after
the build and the gates pass.

## Merge of 2026-10-31 (upstream 0.2.8, 2,586 commits since 14 Oct; 45 files conflicted)

Rule used everywhere: keep Xolotl's behaviour, take upstream's on top.  Where our side was an older backport of
the same upstream code, upstream's newer form won (nothing of ours is lost that way).

| file | Xolotl's side (kept) | upstream's side (taken) |
|---|---|---|
| `src/kernels/cpu/iq_avx2_rows.inl`, `iq_avx512.cpp` | DeepSeek-V4 SwiGLU clamp (`lim`) on every gate/up row path | `EX` / `Singleton` template paths (wide-token prompt rows, window-invariant arithmetic); the reference rows get `INFINITY` (no clamp) |
| `src/kernels/cpu/native_expert.cpp` | PTQ1_0 + Q2_0 rows, MiMo mixed gate/up (`up_type`), `blck()`/`row_size()` for PTQ1_0 | trellis types (TQ*), singleton IQ path, KQ arithmetic guard; its new 1-token shortcuts (`exact_one`, `x512_one`) are taken only with one type and no clamp |
| `src/kernels/cpu/pool.cpp/.hpp` | per-job `fmt` (DS4.1 mixed tiers), GLM LoRA fields | the same `fmt` field for 149-cold, NUMA slices, `NativeArithmetic` |
| `src/kernels/cuda/iq_kernels.cu` | mixed gate/up kernel, PROM prompt kernels, PTQ1_0/Q3_K AMD cases, `STRATA_EXP_MODE_T<type>`, GLM LoRA down, clamp in `swiglu_entries_kernel` | nibble slots, EXPDEC chains, Hadamard-folded down, `sm75_down_em` |
| `src/prefill/moe_mmq.cu`, `CMakeLists.txt` | the generated MMQ type list (`cmake/mmq_types.cmake`, single source of truth), `swiglu_clamp` | trellis types reach the generated list through cmake; `swiglu_quant`; `type_blck` |
| `src/qwen35/q36_engine.cu`, `tools/qwen35/q36_mtmd_dump.cpp` | **Xolotl's whole file** (NektoDron: vision, batching, shared-prefix cache, FP16 KV, flash prefill) | not merged: upstream rebuilt Qwen3.6 batching/vision another way (`RowSel`, per-session graph sets, FP32 KV). **To port later:** upstream's q36 speed commits (prompt delta rule, chunk sizing, router+shared-gate fusion) |
| `src/core/verify.cpp`, `src/program/generate.cpp` | `kv_target` chain branch (generate.cpp) | the newer split-overlap (ours was an older backport), `STRATA_SELECT_LONG`, narrow verify windows |
| `src/core/conversation_copy.hpp` | - | upstream's handle pool (the same leak our #57 fixed, fixed more completely) |
| `serve/server.py` | deep sleep, `/v1/decide`, lazy vision + `on_demand_vision`, image-id cache + refill, grammar, cache markers, `done_log`, voice sidecar closers, `structured_compact`, `log_size`/`since` for the encoder log, cancel-while-queued in the request log | queue deadlines + exceptions, logprobs, prompt cache (#567), DeepSeek pictures, launch attestation, split serving, `parse_config_text` |
| `setup.py` | `Xolotl-data`, own settings file, `is_xolotl` | `home()`, trash checks, app settings file, unusable-folder checks |
| `tools/strata_tokenizer.py` | per-`pre` pattern lists (`_pre_res`, joyai-llm) | DeepSeek-V3 sequence (now one of the lists) |
| docs | the "To check" notes, MULTI_GPU sections, README (Xolotl's) | upstream's rewritten pages |

Known follow-ups: the q36 speed port above; upstream's own DeepSeek-V4.1 engine (`src/dsv41`) now sits beside
ours (`tools/v41`) - nothing shares a target, but compare them; upstream's KVA / DFlash2 / layer-major prefill are
in the tree and opt-in, worth an A/B on our models.

## Accepting a merge

1. `tools/lost_lines.py <merge-base> public HEAD` - every listed line must be a rewrite that keeps the behaviour.
2. Build `strata`, `ds41_generate`, `glm_generate`, `xolotl-dense` with build-xolotl's options.
3. The gates in UPSTREAM_SYNC.md, plus the Flash-Next check: `tools/quant_quality.py run` on the served quant must
   match the previous build's TSV (same top-1 / KL to the old build) and `tools/throughput.py` must not regress.
4. Fast-forward `public`, rebuild `build-xolotl`, keep the previous binaries until the next day's use is clean.
