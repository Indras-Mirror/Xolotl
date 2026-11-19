// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding), and Q2_K for DeepSeek-V4-Flash's expert
// down projections (with Q3_K, the Q2_0 file's embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
// xolotl: IQ3_S, 2..8 columns, reading native_q8_1_interleave's position-major ints (pm) and scales (dl) for `ncols`:
// bitwise iq_mmvq's output.  iq_mmvq_il_supported: when it runs (else call iq_mmvq).
bool iq_mmvq_il_supported(int ggml_type, int ncols);
void iq_mmvq_il(int ggml_type, const void* w, const int* pm, const float* dl, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32; fp16 times `*scale` when given (device:
/// an NVFP4 expert's s_down, read by the kernel itself so the source may be released right after it).
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale = nullptr);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.  `tail` (device, NVFP4 only): the blob's {s_gate, s_up,
/// s_down, 0}; gate rows are scaled by s_gate and up rows by s_up (s_down goes on the down matrix: folded into up,
/// the up weights would sit near 1e-8, deep in FP16's subnormals).
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream, const float* tail = nullptr);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
    // MiMo GSQ-RCO: the up projection's own type when it differs from the gate's (-1 = gu_type, every other model);
    // set only by native_expert_layout3.  Gate and up rows are computed by different warps, so a block never mixes.
    int up_type = -1;
    size_t up_row = 0;
    // DeepSeek-V4's SwiGLU clamp (swiglu_clamp_exp): gate -> min(gate, lim), up -> clamp(up, -lim, lim) before
    // silu(gate) * up.  +inf (native_expert_layout's default, every other model) = no clamp, the same bits as before.
    float swiglu_limit = std::numeric_limits<float>::infinity();
    size_t tail_off = 0;                // NVFP4: {s_gate, s_up, s_down, 0} at the blob's end; 0 = none
    // APR (prism.hadamard): the down weights are Hadamard-folded, so native_expert_grouped rotates h before
    // quantizing it (hadamard_rows).  had_block 0 = none; d_signs is a DEVICE vector of n_ff signs or nullptr.
    int had_block = 0;
    bool gu_nib = false;                // STRATA_EXPERT_NIBBLE: gate/up as 4-bit codes (strata/core/expert_nibble.hpp's
                                        // slot); set only by native_expert_layout_nib
    const float* d_signs = nullptr;
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// STRATA_EXPERT_NIBBLE: a converted VRAM slot - gate/up as 4-bit codes, scales and d (expert_nibble.hpp's layout:
/// up_off = the up rows' codes, down_off = kGuBytes, bytes = the slot), the GGUF down rows after them.  IQ2_S /
/// IQ3_XXS / IQ3_S at n_embd 2560, n_ff 640 only, and only in a build with the kernels (`native_expert_nib_supported`).
NativeExpertLayout native_expert_layout_nib(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
bool native_expert_nib_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Gate and up of different types (MiMo GSQ-RCO): equal types give exactly `native_expert_layout`'s layout.
NativeExpertLayout native_expert_layout3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff);
/// `native_expert_supported` for a gate/up/down triple: equal gate/up = the pair check; otherwise only the mixed
/// pairs that have a kernel (Q2_K/Q3_K either way round).
bool native_expert_supported3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);
/// out[t] = sum_k w[t*K + k] * parts[t*K + k] over rows of H floats, t < n (device pointers; the prompt chunk's mix).
void weighted_rows_sum(const float* parts, const float* w, int K, int64_t H, int64_t n, float* out, void* stream);
/// One decode token's top-k mix on the card: out[i] = sum_k w[k] * parts[k * H + i] in double, k in order (== the
/// tier's host loop bit for bit); w[k] == 0 skips row k.
void decode_rows_sum(const float* parts, const float* w, int K, int64_t H, float* out, void* stream);
/// decode_rows_sum with the pool's rows read where the host wrote them: row k comes from `host_rows` (pinned host
/// memory under UVA, read by the kernel over PCIe - no H2D copy per row) when sel[k] != 0, else from `parts`; the
/// weights and selectors travel with the launch.  The same values in the same k-order double sum: the same bits.
constexpr int kRowsSumSelMax = 16;
struct RowsSumSel {
    float w[kRowsSumSelMax];
    int32_t sel[kRowsSumSelMax];
};
void decode_rows_sum_sel(const float* parts, const float* host_rows, const RowsSumSel& ws, int K, int64_t H, float* out,
                         void* stream);
/// STRATA_META_PULL (ds41 decode): copy `bytes` (a multiple of 4) from mapped pinned host memory `src` (its device
/// address) to device memory `dst` with a kernel, on `stream` - not through the copy engine, where a small copy waits
/// behind every bulk DMA queued before it.
void pull_from_host(void* dst, const void* src, size_t bytes, void* stream);
/// STRATA_HIT_FAST (ds41 decode): the routed hits' grouped-launch addresses from the routing on the card.  row[e]: expert
/// e's VRAM slot address, 0 = not resident.  All K of ids resident -> grp_ptr[k] = row[ids[k]], *n_groups = K; else
/// *n_groups = 0 (then the grouped kernels and decode_rows_sum_gated do nothing).  All pointers are device memory.
/// armed (STRATA_HIT_FAST=3; null = always): a word in mapped pinned memory; 0 -> *n_groups = 0; the dispatch clears it.
/// par (STRATA_HIT_PAR): one lane per lookup, all reads in flight at once - one bus round trip for a mapped row table and
/// armed word instead of K + 1; the same *n_groups and (when it is K) grp_ptr.
void decode_hit_dispatch(const int32_t* ids, const unsigned long long* row, int K, int n_experts,
                         unsigned long long* grp_ptr, int32_t* n_groups, void* stream, int32_t* armed = nullptr,
                         bool par = false);
/// STRATA_HIT_PF: the layer's prefetched experts as the host published them (mapped pinned memory): valid when `stamp`
/// is the arming's stamp (the armed word's value), `n` entries of expert id and staging address.
constexpr int kHitPfMax = 8;
struct HitPfList {
    int32_t stamp;
    int32_t n;
    int32_t e[kHitPfMax];
    unsigned long long ptr[kHitPfMax];
};
/// decode_hit_dispatch (par) that also takes a routed expert from `pf` when it is not resident.  All K found: grp_ptr /
/// ent_dst for all of them, *n_groups = *sum_gate = K.  Otherwise, with `part` (STRATA_HIT_PART), the found ones packed
/// in routing order (ent_dst = their routing index), *n_groups = their count and *sum_gate = 0; without it *n_groups =
/// *sum_gate = 0.  dec (mapped pinned memory): [1] = K or 0, [2] = the mask of routing indices the launch runs, then
/// [0] = the stamp - the decision, published at once (the host plans the layer on it).  Then, if a prefetched expert is
/// taken, the dispatch waits until *ready == the stamp (device memory, written by a copy queued behind the prefetch's
/// DMAs), so the grouped launch behind it reads their bytes; after wait_ns it sets *err = the stamp (mapped; the host
/// aborts on it) and *n_groups = *sum_gate = 0, so nothing behind it reads the stage.  *stamp_dev = the stamp.  An
/// unarmed launch (armed word 0) only zeroes *n_groups and *sum_gate.
/// ready_item (null = none): kHitPfMax words, word j written with the stamp by a copy queued right behind list entry j's
/// DMA.  item_wait (STRATA_HIT_PF=2): wait for the taken entries' words only, not for *ready (the whole batch).
/// stat (STRATA_HIT_PF_STAT, device memory, kHitPfStatWords; null = off): counters of the waits, see kHitPfStatWords.
/// Layout of the stat counters: [((n - 1) * 8 + last) * 4 + f] for a wait on a list of n entries whose last taken entry
/// is `last`, then [256 + r * 4 + f] by r = the taken resident experts; f: 0 waits, 1 ns until the taken entries were
/// home, 2 ns until the wait ended (the batch without item_wait), 3 waits whose taken entries were home at once.
constexpr int kHitPfStatWords = 8 * 8 * 4 + 8 * 4;
/// STRATA_HIT_PF=3 (device memory; grp_ptr null = no split): the run's prefetched experts as a second set for their
/// own grouped launch (grp_ptr / ent_dst / *n_groups; the first set then holds the residents alone) and wrec[3] =
/// {mask of the list entries taken, list entries, residents taken} for decode_hit_pf_wait.  The dispatch does not wait.
struct HitPfSplit {
    unsigned long long* grp_ptr = nullptr;
    int32_t* n_groups = nullptr;
    int32_t* ent_dst = nullptr;
    int32_t* wrec = nullptr;
};
void decode_hit_dispatch_pf(const int32_t* ids, const unsigned long long* row, int K, int n_experts,
                            unsigned long long* grp_ptr, int32_t* n_groups, void* stream, int32_t* armed,
                            const HitPfList* pf, const int32_t* ready, int32_t* dec, long long wait_ns,
                            int32_t* ent_dst, int32_t* sum_gate, bool part, int32_t* stamp_dev, int32_t* err,
                            const int32_t* ready_item = nullptr, bool item_wait = false,
                            unsigned long long* stat = nullptr, HitPfSplit split = {});
/// STRATA_HIT_PF=3: a split dispatch's wait for its taken entries' words (ready_item == *stamp_dev), queued on the
/// stream of the prefetched set's grouped launch.  After wait_ns: *err = the stamp, *n_groups2 = *sum_gate = 0.
void decode_hit_pf_wait(const int32_t* wrec, const int32_t* stamp_dev, const int32_t* ready_item, long long wait_ns,
                        int32_t* n_groups2, int32_t* sum_gate, int32_t* err, unsigned long long* stat, void* stream);
/// decode_rows_sum_sel without pool rows, the weights in device memory, and only when *gate != 0: the same bits.
/// done (STRATA_HIT_PART, mapped pinned memory; null = none): *done = *stamp first, whatever the gate - the launches
/// queued before this one (the grouped kernel) are finished then.
void decode_rows_sum_gated(const float* parts, const float* w, const int32_t* gate, int K, int64_t H, float* out,
                           void* stream, const int32_t* stamp = nullptr, int32_t* done = nullptr);
/// STRATA_RB_HOOK (ds41 decode): a readback queued inside a captured graph.  Up to kHostPushMax arrays of 4-byte words
/// go from device memory to mapped pinned host memory (its device address) by one block, not through the copy engine;
/// then *flag = 1 (mapped pinned memory; null = no flag) once the words are visible to the host.
constexpr int kHostPushMax = 4;
struct HostPush {
    const void* src[kHostPushMax];
    void* dst[kHostPushMax];
    int words[kHostPushMax];
    int n;
    int32_t* flag;
};
void push_to_host(const HostPush& p, void* stream);
/// The device address of pinned host memory (cudaHostGetDevicePointer), or nullptr.
void* host_dev_ptr(void* host);

/// A770 port (STRATA_PF_XMX=2): up to kXmxGroupMax experts' products in one XMX launch, the weights decoded from the
/// quantized rows into local memory.  Expert i: Y_i[cnt[i]][N] = X_i . W_i^T (X FP16, Y FP32, rows back to back).
/// gate/up: b != nullptr (a = gate, b = up rows of n_embd = K; the output's virtual rows interleave gate/up);
/// down: b == nullptr, a = the [N][K] rows.  N a multiple of 128, K of 32 (and a whole number of rows per 256 block).
constexpr int kXmxGroupMax = 16;
void iq_xmx_grouped(int ty, const uint8_t* const* a, const uint8_t* const* b, const int32_t* cnt, int n,
                    const uint16_t* X, float* Y, int N, int K, void* stream);

/// Routed-expert rank-1 LoRA (the GLM abliteration adapter): per-layer deltas the grouped path adds to the experts'
/// gate/up pre-activations and to their down outputs, WITHOUT touching the quantized blobs.  For expert e, entry
/// token x: s_g = <a_g[e], x>, gate_r += b_g[e][r] * s_g (and the same for up); then s_d = <a_d[e], h>, and the
/// down output += b_d[e] * s_d (h = the SwiGLU output already in scratch).  Every pointer is DEVICE memory; all
/// null (the default) = off, bit-identical to the unmodified path.  The arrays are indexed by expert id.
/// UNION (xolotl): independent of the fork's decode_rows_sum/iq_xmx_grouped above - both declarations kept.
struct NativeExpertLora {
    const float* a_g = nullptr;      ///< [n_experts][n_embd]
    const float* b_g = nullptr;      ///< [n_experts][n_ff]
    const float* a_u = nullptr;
    const float* b_u = nullptr;
    const float* a_d = nullptr;      ///< [n_experts][n_ff]
    const float* b_d = nullptr;      ///< [n_experts][n_embd]
    const int32_t* ent_exp = nullptr;   ///< [cap_entries] expert id of each entry (device, as ent_tok/ent_dst)
    const float* x = nullptr;        ///< fp32 activations, `x_stride` floats per token (indexed by ent_tok[e])
    int64_t x_stride = 0;
    int64_t n_experts = 0;
    int64_t ent_lo = 0, ent_hi = 0;  ///< correct entries [lo, hi) (the launch's active entry range); lo == hi = none
};

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  The results do not depend on it.
/// `lora` (optional): the routed-expert rank-1 deltas above; null = the unmodified path.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0, const NativeExpertLora* lora = nullptr);
/// The routed-expert LoRA's down correction on its own (for the MMQ chunk path): entries [0, n), expert ent_exp[i],
/// float SwiGLU row h + i * n_ff, adds b_d[e] * (a_d[e] . h) into row ent_dst[i] of `out` (n_embd floats).
void native_expert_lora_down(const NativeExpertLora& lora, const int32_t* ent_exp, const int32_t* ent_dst,
                             const float* h, int64_t n_ff, int64_t n_embd, int32_t n, float* out, void* stream);
/// true: `native_expert_grouped`'s launches before the group stride (STRATA_GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries.  Bitwise the same results
/// (native_grouped_parity checks it); kept for A/B timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);
/// The bench / parity only: the 4-bit gate/up kernel of every format (-1 = expert_nibble.hpp's gu_skeleton, the
/// default; 0 = the S skeleton; 1 = the P skeleton).  Bitwise the same results; set before graph capture.
void native_expert_nib_set_variant(int v);
/// gu-proto (nib_gu_proto.cuh): variant v's registers, local bytes and resident 256-thread blocks per SM for gu_type;
/// false when v is not a gu-proto id (>= 10) or the type has none.
bool native_expert_nib_variant_attr(int v, int gu_type, int* regs, int* local_bytes, int* blocks_per_sm);

/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down, 3 gate/up only).
void native_expert_set_mode(int mode, int phase);

/// APR (prism.hadamard): y = H_block (signs * x) per block of `block` values of each of `rows` rows of `width`
/// (x == y allowed).  `signs`: a device vector of `width` signs, or nullptr (identity).  The same floats as the CPU's
/// hadamard_rotate.  The f16 form rotates in place, computing in float.
void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream);
void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream);
/// A device copy (on the current device) of the host sign vector `host` of `width` signs, made on first use and
/// kept for the process.  Call it at setup, not inside a CUDA graph capture.  nullptr for a nullptr `host`.
const float* hadamard_device_signs(const float* host, int width);

/// EXPDEC (STRATA_EXPDEC=1, docs/EXPERTDEC.md): `native_expert_grouped` stages the group's q8_1 activations in shared
/// memory and computes several rows per lane, the same sums per (row, entry) as the default kernels (bitwise), for
/// gate/up IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S at n_embd 2560 and down IQ4_NL / Q2_0 at n_ff 640; other layouts
/// keep the default kernels.  Set before graph capture; captured graphs keep the kernels they captured.
void native_expert_set_expdec(int on);
int native_expert_expdec();
/// Its knobs, read from the environment at startup (STRATA_EXPDEC_R 1 | 2, _DR 1 | 2 | 4, _MINB 2 | 4, _D20 0 | 1,
/// _RB, _DRB, _MINBLK): gate/up and down rows per lane, the gate/up kernel's __launch_bounds__ minimum blocks per SM
/// (2 = up to 128 registers, 4 = up to 64), EXPDEC for the IQ4_NL down too, gate/up and down rows per block (0 = auto
/// from the launch's groups), the auto rule's minimum blocks.  In the setter a field <= 0 (< 0 for the two rows per
/// block and down_iq4_nl) keeps the current value.
/// `down` (STRATA_EXPDEC_DOWN, default 0): EXPDEC for the down too; by default only the gate/up (in the engine the
/// Q2_0 down was ~10% slower with EXPDEC, docs/EXPERTDEC.md par. 7).
struct NativeExpertExpdecParams {
    int gu_rows_per_lane = 0, down_rows_per_lane = 0, gu_min_blocks_per_sm = 0, down_iq4_nl = -1, down = -1;
    int gu_rows_per_block = -1, down_rows_per_block = -1, min_blocks = 0;
};
void native_expert_set_expdec_params(const NativeExpertExpdecParams& p);
NativeExpertExpdecParams native_expert_expdec_params();
/// Whether EXPDEC has kernels for this layout's gate/up and down (the bench's check).
bool native_expert_expdec_takes(const NativeExpertLayout& L, bool& gu, bool& down);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
/// Volta prompt experts (STRATA_PF_WMMA): Y[rows of expert e] = X[rows] . W_e^T for a group of experts, W_e at
/// W + e * expert_bytes (n_out rows of K values of type ty, GGUF blocks), X FP16 (ldx), rows bounds[e]..bounds[e+1],
/// Y FP32 (ldy).  FP16 tensor cores, FP32 sums.  false: not launched (shape or type not covered).  CUDA only.
bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream);
void iq_set_old_kernels(bool old);
bool iq_old_kernels();

/// Strata-DS4 fork: the prompt-batch ("PROM") grouped kernels for the PTQ1_0 (143) / Q2_0 (42) expert tiers - one
/// weight-tile decode per (row, group) in shared memory, applied to every entry of the group with int8 dp4a
/// (STRATA_PROM=1 at first use selects it; default off).  Used by `native_expert_grouped` for those formats only.
void native_expert_prom_set_enabled(bool enabled);
bool native_expert_prom_enabled();
/// Test only: ignore the measured type/size gate (PTQ1_0 and >= 64 entries per group), so a unit test can run the
/// kernels on both tiers at every row count.  Never set by the engine.
void native_expert_prom_set_force(bool force);

}  // namespace strata::kernels
