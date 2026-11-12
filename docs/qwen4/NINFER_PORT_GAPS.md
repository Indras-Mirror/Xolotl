# NInfer Qwen4-27B → Xolotl: op and geometry gap analysis

Written 2026-11-12 against `public` @ de1a9ff26. Sources read:

* NInfer's Qwen4-27B design note and op checklist (`ninfer-qwen4-27b-rtx5090.md`,
  `ninfer-qwen4-27b-op-checklist.md`, dated 2026-10-27) — architecture, merged-row
  layouts, per-op geometry.
* `github.com/Neroued/ninfer` at the time of writing: `include/ninfer/ops/{gated_residual,qsa,
  linear,linear_split,mhc}.h` and the `src/ops/{gated_residual,softmax_attention/qsa,mhc}`
  trees, read through `gh api`.
* Our plan `xq/notes/QWEN4_27B_DENSE_PLAN.md` (Route B, slices S1-S5) and
  `xq/notes/QWEN4_ENGINE_READINESS.md`.

**The spec is unconfirmed.** Every Qwen4-27B shape below is NInfer's reading (itself a
reading of Intel's kernel inventory and `transformers`' `qwen4_exp`). Nothing here is a
measured property of a released model. Rows marked UNVERIFIED are claims no one on this
machine could check without weights, a GPU run, or a full build.

## The target, in the terms this engine uses

Dense, 64 layers, `H=5120`, FFN 14336 SwiGLU, layer pattern GDN,GDN,GDN,QSA (48/16),
4 residual streams (HC) with bottleneck rank **640**, QSA 24 Q / **4 KV** heads × d256 with
a 4 Q / 1 K × d128 indexer (block 4, budget 2048 cells), no PLE / n-gram, one MTP layer
with its own HC + QSA + dense FFN, MRoPE sections `[11,11,10]` theta 1e7, RMS eps 1e-6.

Our notes originally carried HC rank **320** (Flash-Next's). NInfer's Qwen4 note says
**640**. The two are not the same kernel instantiation; see the HC section.

## 1. Hyper-connections (HC / "gated residual")

NInfer today implements the same 4-stream, H=2560, rank-320 geometry Flash-Next uses
(`include/ninfer/ops/gated_residual.h`: "Four-stream expansion and offset RMSNorm, with
H=2560 and S=4"; `HcMixWeights::down_injection` is BF16 `[324,10240]`, injection folded in
as rows 320..323). The Qwen4 note raises H to 5120 (residual width 20480) and the rank to
640, and separates injection from the FP8 down/up weights (checklist H1-H5).

| NInfer requirement | Xolotl today | Gap |
|---|---|---|
| `hc_expand_norm`, `hc_grouped_rmsnorm`, `hc_combine`, `hc_combine_norm` at H=5120 | `gr_read` / `gr_write` and the norm/postops kernels take `(n_embd, hc, hc_lr)` at run time: `include/strata/kernels/gr.hpp:43-47`, `src/kernels/cuda/gr.cu:378-426`, `src/kernels/cuda/native_gr_norm.cu`, `native_gr_postops.cu`. The dense engine already drives them with the file's own geometry: `src/core/dense_model.cpp:663-668` | none for H; the kernels are runtime-shaped. UNVERIFIED numerically at H=5120 (no weights). |
| HC down/up `[640,20480]`, `[20480,640]`, FP8 / AllowA8 | Xolotl stores HC down/up **BF16** (`layer_tensors`: `hc_attn_down.weight {hc_dim, hc_lr}`, `include/strata/core/family_layout.hpp:172-180`; `NativeDense::eligible` does not upload them; `gr_read` reads bf16 pairs, `src/kernels/cuda/gr.cu:388`) | **format gap**: no FP8 HC projection kernels. rank 640 itself is fine on the generic path. |
| HC injection separate, BF16 `[4,20480]`, output FP32 `[4,M]` | already separate tensors (`hc_*_inject.weight`) and FP32 injection buffer (`gr.hpp`) | none |
| fast fused HC read at rank 640 | `src/kernels/cuda/fused_gr.cu:34` `LR=320`, compile-time; `fused_gr_supported()` returns false for any other rank (`:2531-2532`), and both the layer loop and dense model fall back to `gr.cu` (`src/core/layer.cpp:1475-1476`, `:1615`) | **not a load blocker**; the fast path is 320-only by design. A 640 model runs the generic kernels. |
| final mixer without injection | `lm_head_mix` / `output_hc_down/up/norm` (`family_layout.hpp:228-232`, `layer.cpp:1375`) | none; the dense engine already has it (`dense_model.cpp:1209-1215`) |
| HC + MTP | dense engine **refuses** MTP with HC: `dense_hc_refusal()` (`include/strata/core/dense_arch.hpp:58-68`) | MTP block has its own mixer (`nextn_hc_head` in the Qwen4 note); not wired anywhere in Xolotl. UNVERIFIED layout. |

The per-layer tensor set differs from NInfer's merged parent: Xolotl has six separate HC
matrices per mixer + a norm vector (`family_layout.hpp:172-180`), NInfer merges
down+injection. A converter is needed either way; no engine change for the merge itself.

## 2. QSA (sparse full attention) at 24 Q / 4 KV

The indexer geometry NInfer needs (**4 Q / 1 K, d128, block 4**) is exactly what Xolotl
has: `idx_q_heads=4`, `idx_key_dim=128`, `idx_block=4` (`include/strata/core/geometry.hpp:43-46`)
and the compiled kernels assert those numbers
(`src/kernels/cuda/qsa_select.cu:21-23`, `native_qsa_score.cu:32-34`,
`native_qsa_indexer.cu:36-38`). No indexer work.

The **main attention** geometry differs: Xolotl is 24 Q / **2 KV** (ratio 12), Qwen4 is
24 Q / **4 KV** (ratio 6). Three kernel families pin ratio 12 at compile time and are
written around it:

* decode: `src/kernels/cuda/qsa_decode_attn.cu:23-25` (`HD=256`, `G=12`), and the block
  layouts assume a 12-head, 16-row group (`:177`, `:227`, `:431` builds `pr[12]` from
  three `float4`s). The host entry points **`std::exit(1)`** when
  `s.n_head != G * s.n_head_kv` (`:1057`, `:1076`, `:1189`).
* prompt: `src/kernels/cuda/qsa_prompt_attn.cu:29-31`, same 16-row MMA padding
  (`:1830`); `qsa_prompt_attn_batch` returns false for a non-24/2/256 geometry, which the
  prefill caller already treats as "use the old kernel"
  (`src/prefill/prefill.cpp:4114`, `:6990`).
* the sm_6x prompt variant: `src/kernels/cuda/qsa_pattn.cu:56-59`.

The **generic** path is ratio-agnostic: `kv_gather_*` + `qsa_attend_step`
(`src/kernels/cuda/qsa.cu:89-99` validates `n_head % n_head_kv` at run time; the gather
and attend kernels take `n_head`/`n_head_kv` as arguments). The QSA layer selects the
fast path with `g_fast_attn` (`src/core/layer.cpp:1265-1289`). A 24/4 model must take the
generic path; it must not reach the fast kernels' `exit(1)`.

Other QSA pieces match the checklist:

| NInfer | Xolotl |
|---|---|
| full main K/V history, 4-token compressed index keys, raw tail | `kv_append_step` / `indexer_key_append` / `qsa_tail_extract` (`layer.cpp:1204-1234`, `src/kernels/cuda/qsa.cu`) |
| FP8 E4M3 row-256 KV (Qwen4 note) | Xolotl offers fp16 / int8 / q4_0 / K8V4 (`kv_q8.hpp`, `kv_q4.hpp`); **no FP8 KV path** — UNVERIFIED whether NInfer's FP8 cache is required for day 0 or a size optimisation |
| sigmoid output gate per query feature | `qsa_gate_apply` (`layer.cpp:1296-1299`) |
| block top-k, budget 2048 cells | `qsa_block_topk` / `topk_512_step` (`layer.cpp:1255-1260`) |

## 3. Dense FFN on the strata (Flash-Next) path

Today the strata host graph has exactly one FFN, the MoE block:

* `moe_layer` (`src/core/layer.cpp:559-564`) = router + caller-owned expert outputs + combine;
  `moe_shared` (`:470-515`) is the SwiGLU shared expert with a **scalar** sigmoid gate.
* `moe_route` hard-requires `ffn_gate_inp.weight` (`src/core/layer.cpp:407-408`).
* the layout table demands the whole MoE tensor set on every layer:
  `include/strata/core/family_layout.hpp:161-171`; the runtime shape check does the same:
  `src/core/layout.cpp:50-80`.
* native upload covers only attention/GDN/shared-expert tensors:
  `src/core/native_dense.cpp:154-165`.
* geometry defaults `n_expert` to 512 and requires it positive:
  `include/strata/core/geometry.hpp:53`, `include/strata/artifact/gguf_reader.hpp:729-750`,
  `include/strata/core/family_layout.hpp:85-88`.
* `compiled_geometry_mismatch` pins `expert_feed_forward_length` /
  `expert_shared_feed_forward_length` to 640 (`family_layout.hpp:110-127`), and
  `geometry_unsupported` pins the same (`gguf_reader.hpp:863`).

So a dense file with no `expert_count` is refused before any tensor is read. Route B
slices S1-S5 (geometry, layout, upload, a `dense_ffn()` beside `moe_shared`, the
`n_expert == 0` branch including the MTP block) are the work; they are landed separately
from this document and their status is in the commit message, not here.

The arithmetic already exists twice and can be reused rather than rewritten:
`shared_expert` computes `silu(x·gate)·(x·up)·down` with the per-tensor activation
contract (`src/kernels/cuda/shared_expert.cu:279-483`); the dense engine's `ffn_matmul`
(`src/core/dense_model.cpp:642-656`) is the same FFN for `xolotl-dense`.

## 4. MTP + HC

NInfer: one MTP layer with its own HC, QSA and dense FFN; conditioning is the 20480-wide
residual before the final mixer; `fc_hidden` is `[5120,5120]` over 4M columns; a Q4
proposal head is separate from the shared FP8 head.

Xolotl today has two unrelated MTP stories:

* the stratum `MtpDrafter` (`src/core/mtp.cpp`) is the **MoE** draft block: it requires
  `mlp.shared_expert.{gate,up,down}_proj.weight` and the router (`mtp.cpp:349-352`),
  loads 512 resident experts (`:330-347`), and its forward runs router → experts →
  shared expert → combine (`:1106-1200`). It refuses a vram-less expert set; a dense MTP
  file has none of these tensors.
* the dense engine refuses MTP together with HC outright (`dense_arch.hpp:58-68`), naming
  the `nextn_hc_head` mixer.

Consequence: **no engine can run a Qwen4-27B MTP layer today.** The names and layout for a
dense MTP block are UNVERIFIED (the note gives shapes, not GGUF names). The strata path's
`n_expert == 0` branch therefore covers the MTP block's *tensors* in `layer_tensors`
(`LayerKind::Mtp`, `family_layout.hpp:212-218`) but the drafter itself needs its own dense
branch; see the slice report.

## 5. GDN, linear, embedding, vision

These match; no port work is implied.

| NInfer requirement | Xolotl |
|---|---|
| GDN 16 K / 48 V × d128, conv 10240 × 4, FP32 state | same compiled shape (`geometry.hpp:30-36`), `gdn_layer` (`layer.cpp:252`), native GDN asserts state 128 (`src/kernels/cuda/native_gdn.cu:39`) |
| QKVZ `[16384,5120]`, A/B `[96,5120]` | generic quantized GEMVs at runtime widths (`layer.cpp:252-385`) |
| six-way `linear_split` parent `[14976,5120]` | Xolotl has separate `attn_q/k/v/o` and `indexer.*` projections and no merged parent; the merged row order (K,V,index-K,Q,gate,index-Q) and the `[2176,5120]` prefix view are **not implemented** — a converter/pack change, not a kernel gap (checklist L3/M1) |
| SwiGLU gate/up `[28672,5120]`, down `[5120,14336]` | `linear_swiglu` / shared-expert path take widths at run time; NVFP4 is not an Xolotl format |
| embedding / LM head `[248320,5120]`, unbound | `embed_row` / `lm_head` with the file's shapes (`layer.cpp:1320`, `:1396`); the vocab is free (`geometry_unsupported` comment) |
| Qwen vision (27 layers, d1152, head 72, merger) | `src/models/common/qwen/vision/`; reused as-is |

Weight formats are the widest gap that is *not* engine code: NInfer's parents are
NVFP4 / FP8 E4M3 row-BF16 / Q4 G64. Xolotl's strata path serves GGUF quant types through
`iq_pack` plus native `Q8_0/Q8_1` MMVQ, and its QSA/attention projections are S-family /
Q2_0 / BF16. **A Qwen4 GGUF can load; an NVFP4 parent cannot without a converter.** The
Q4 proposal head (`q4_g64_fp16`) has no Xolotl equivalent.

## 6. Compile-time geometry pins that block the target

| Pin | Where | Blocks | Status |
|---|---|---|---|
| `n_embd = 2560` | `geometry.hpp:24`, `compiled_geometry_mismatch:110-127`, `geometry_unsupported:845` | a 5120-wide model | **open** — the runtime kernels are width-generic, but this is a declared kernel contract; relaxing it needs a decision (see §8) |
| `n_layers = 48` | same | a 64-layer model | **open** — same class as `n_embd` |
| `n_ff = 640` | same | dense FFN 14336 | relaxed for `n_expert == 0` in the Route B slice |
| `n_head_kv = 2` | `geometry.hpp:40`, pins above | 24/4 QSA | relaxed for the compiled 24-Q ratio in the widening slice; the fast kernels stay 12-only |
| `hc_lr = 320` | `geometry.hpp:50`, `fused_gr.cu:41` | rank 640 | **not a blocker**: `fused_gr_supported()` falls back to the generic kernels; the fast path stays 320-only |
| `expert_count > 0` | `gguf_reader.hpp:732-733`, `family_layout.hpp:85-88` | any dense file | relaxed for a file with no/false `expert_count` |
| expert CPU kernel `H/FF` | `include/strata/kernels/cpu/expert.hpp:39` | nothing for dense (the expert kernel is never called) | — |

The last one is the reason the plan says a 5120-wide dense model does not imply new
kernels: the compiled asserts are on *feature* dims (GDN state 128, indexer 4/128/4/64,
expert FF), not hidden size. But `n_embd`/`n_layers` are still declared contracts in the
geometry struct; whether they are load-bearing is a per-kernel question that needs a full
build and a run to settle.

## 7. What is UNVERIFIED in this document

* Every Qwen4-27B number (the spec is one person's reading).
* Numerical behaviour at H=5120, rank 640, 24/4: no GPU run, and a 15-hour pipeline
  owns the card. All kernel-support claims above are read, not measured.
* The dense-FFN arithmetic against a reference: no dense Qwen4 reference exists upstream
  (`num_experts <= 0` is refused by the converter).
* FP8 HC projections and FP8 KV: Xolotl has neither; the NInfer note's precision
  choices are not ported.
* The MTP dense layout and GGUF names.
* That `n_embd`/`n_layers` are truly runtime for every kernel the dense path touches.

## 8. Open questions for the owner

1. **HC rank 320 vs 640.** Our notes say 320 (Flash-Next); NInfer says 640. The kernels
   take it at run time, so both load; only the fused fast read is 320-only. Confirm from
   the real header/config on day 0 before tuning.
2. **Relax `n_embd`/`n_layers` in the geometry contract?** A 5120/64 Qwen4 is refused
   today by `compiled_geometry_mismatch` even with a dense FFN. The kernels look
   width-generic, but "looks generic" is not "compiled and run at 5120". Either relax
   with a documented risk or keep the refusal and let `xolotl-dense` own the first load.
3. **Which engine owns day 0** — Route B (strata + dense FFN) only pays off if QSA/PLE are
   present; otherwise `xolotl-dense` (which already serves HC at any rank) is shorter.
4. **Weight format**: is a GGUF/Q8-style quant acceptable for day 0, or is the NVFP4/FP8
   parent a requirement? The latter is a converter and kernel project, not a port.
