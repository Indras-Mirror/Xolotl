// include/strata/kernels/dense_kernels.hpp - the kernels a DENSE qwen35 model (Qwen3.8-27B) needs and the MoE
// engine does not have.
//
// The GDN recurrence, the quantized GEMVs (native_mmvq), RoPE, the per-head RMSNorm and the attention output gate
// already exist and are reused by src/core/dense_model.cpp.  What was missing, because the MoE model routes its
// attention through the QSA indexer and has no plain FFN:
//
//   * an RMSNorm that writes to a second buffer          (dense_rms_norm)
//   * SwiGLU                                             (dense_swiglu)
//   * an F32 GEMV for the two tiny GDN projections       (dense_gemv_f32)   ssm_alpha / ssm_beta, n_out = 48
//   * a K/V cache and full causal attention over it      (dense_kv_append, dense_attn_decode)
//
// The attention is a split-K flash decode: one block per (head, chunk of 256 cells) keeps an online softmax
// (m, l, acc) per warp, the warps are merged inside the block, and a second kernel merges the chunks.  Every
// query head reads the K/V head `h / (n_head / n_kv)` (GQA: contiguous groups, the same map qsa.hpp pins).
//
// KV layout: [n_kv][max_ctx][head_dim] IEEE half, so one head's cells are contiguous.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w        (w NOT shifted: the GGUF already holds 1 + w)
void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream);

/// The GDN output norm of a DENSE qwen35 model: y[h, :] = rms_norm(o[h, :]) * w * SiLU(z[h, :]) for each of `heads`
/// value heads of `cols` (= state_size) values.  llama.cpp's build_norm_gated for qwen35 uses SiLU; the MoE
/// engine's native_gdn_out_norm gates with a sigmoid, which is wrong here (it was the cause of the garbage text).
void dense_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int heads, int cols, float eps,
                        void* stream);

/// Greedy pick over one row of `n` logits: *out_id = the index of the largest (the lowest on a tie; NaN ignored) and
/// *out_p = its softmax probability.  Both are DEVICE pointers.  This is what an MTP draft needs (the token and how
/// sure the head is of it, for the --draft-p-min cut-off).
void dense_argmax_prob(const float* logits, int n, int* out_id, float* out_p, void* stream);

/// out[i] = silu(gate[i]) * up[i]
void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream);

/// y[r] = sum_i W[r * n_in + i] * x[i]           W is a GGUF matrix stored as F32: ne0 = n_in is the contiguous axis
void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);

/// dst[0..n) = value   (the RoPE position of every row)
void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream);

/// Writes this token's K and V (n_kv, head_dim) f32 into cell `pos` of every head's cache as half.
void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream);

/// How the KV cache stores one head's values (the --kv option; the same names as strata's).
///   F16 : head_dim halves.
///   Q8  : head_dim/32 blocks of {half d; int8 q[32]}   (ggml Q8_0, "int8")
///   Q4  : head_dim/32 blocks of {half d; uint8 q[16]}  (ggml Q4_0, "q4_0"), stored after an orthonormal
///         Walsh-Hadamard rotation of the head's 256 values: the attention rotates q the same way and rotates its
///         output back, so the cache needs head_dim == 256 for this format.
/// K and V have a format each: `--kv k8v4` is Q8 keys (exact scores) with Q4 values.
enum DenseKvFormat : int { DENSE_KV_F16 = 0, DENSE_KV_Q8 = 1, DENSE_KV_Q4 = 2 };

/// Bytes of ONE head's ONE cell in `fmt` (a cache is [n_kv][max_ctx][this]).
uint64_t dense_kv_cell_bytes(int fmt, int head_dim);

/// dense_kv_append for any format: k_cache / v_cache are [n_kv][max_ctx][dense_kv_cell_bytes(fmt)] bytes.
void dense_kv_append_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v, int pos,
                         int n_kv, int head_dim, int max_ctx, void* stream);

/// dense_attn_decode for any format.  q is NOT rotated by the caller and out comes back in the original space.
void dense_attn_decode_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                           float* scratch, int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale,
                           void* stream);

/// xolotl: dense_attn_decode_fmt for the n (1..8) columns of a verify window in one launch: q / out are
/// [n][n_head][head_dim], column c attends over cells [0, n_ctx0 + c).  The scratch must hold n times
/// dense_attn_scratch_bytes(...).  A column's result does not depend on n (single steps match window columns).
void dense_attn_decode_multi_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt,
                                 float* out, float* scratch, int n_head, int n_kv, int head_dim, int n_ctx0, int n,
                                 int max_ctx, float scale, void* stream);

/// Bytes of scratch dense_attn_decode needs for a cache of `max_ctx` cells.
uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx);

/// out (n_head, head_dim) = softmax(q . K^T * scale) V over cells [0, n_ctx), n_ctx >= 1 (the current token's
/// cell is already appended).  q is (n_head, head_dim) f32, already normalised and rotated.
void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream);

// ============================ the batched prompt path (T consecutive tokens at a time) ============================
//
// Same arithmetic as the per-token kernels above, with the token as the slow index: every buffer is [T, ...] and the
// order-dependent steps (the GDN conv and delta-rule state, the KV append, the causal mask) walk the tokens inside
// one launch.  Feeding a chunk therefore has to land on the same logits as feeding those tokens one at a time.

/// q[r, :] = q_full[r, 0:head_dim] for r in [0, rows): the query half of every head's [q | gate] block.  q_full is
/// [rows, 2*head_dim] and q is [rows, head_dim] (rows = T * n_head).
void dense_split_q(const float* q_full, float* q, int rows, int head_dim, void* stream);

/// out[i] = attn[i] * sigmoid(gate) with the gate the second half of every head's q_full block (rows = T * n_head).
void dense_gate_apply(const float* attn, const float* q_full, float* out, int rows, int n_head, int head_dim,
                      void* stream);

/// dst[i] = pos0 + i / heads   (the position of every row of a [T, heads, head_dim] tensor)
void dense_positions_i32(int32_t* dst, int rows, int heads, int32_t pos0, void* stream);

/// K and V of T consecutive cells (k/v are [T, n_kv, head_dim] f32) into cells pos0..pos0+T-1 of every cache.
void dense_kv_append_rows(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int T, int pos0,
                          int n_kv, int head_dim, int max_ctx, void* stream);

/// dense_kv_append_rows for any KV format: the caches are [n_kv][max_ctx][dense_kv_cell_bytes(fmt)] bytes.
void dense_kv_append_rows_fmt(void* k_cache, void* v_cache, int k_fmt, int v_fmt, const float* k, const float* v,
                              int T, int pos0, int n_kv, int head_dim, int max_ctx, void* stream);

/// Causal attention for T consecutive queries at positions pos0..pos0+T-1: one block per (head, token), the same
/// warp-per-cell online softmax as dense_attn_decode, restricted to cells [0, pos0 + t].  q is [T, n_head, head_dim]
/// f32 (normalised and rotated), out is the same shape.
void dense_attn_chunk(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int T, int pos0,
                      int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream);

/// dense_attn_chunk for any KV format.  q is NOT rotated by the caller and out comes back in the original space.
void dense_attn_chunk_fmt(const float* q, const void* k_cache, const void* v_cache, int k_fmt, int v_fmt, float* out,
                          int T, int pos0, int n_head, int n_kv, int head_dim, int max_ctx, float scale, void* stream);

/// y[r, :] = W x[r, :] for r in [0, rows): the batched form of dense_gemv_f32 (one warp per output row per input row).
void dense_gemv_f32_rows(const float* W, const float* X, float* y, int n_in, int n_out, int rows, void* stream);

/// The 4-tap causal conv + SiLU over a chunk: history is [channels, 3] (oldest first, updated in place to the
/// chunk's last three inputs), qkv and h are [T, channels].  Only the SiLU output is written (the decode path keeps
/// the raw conv output in its own buffer; the prompt path does not need it).
///
/// A speculative pass feeds several columns at once and has to be able to undo the columns after any of them, which
/// means the recurrent state has to be captured after every column.  The chunk kernels walk the columns inside one
/// launch with the state already in registers, so they can write those snapshots as they go: `snap.slot[j]` receives
/// the state as it was right after column j, for j < snap.n.  Doing it here costs one extra store per column instead
/// of a separate copy per column, and it costs nothing at all when snap.n is 0.
struct DenseGdnSnap {
    static constexpr int kMaxSlots = 8;
    float* slot[kMaxSlots] = {};    ///< slot[j] = the state right after column j (state) / history after column j (conv)
    int n = 0;                      ///< how many slots are live; 0 asks for no snapshots
};

void dense_gdn_conv_chunk(float* history, const float* qkv, const float* conv_w, float* h, int channels, int T,
                          const DenseGdnSnap& snap = {}, void* stream = nullptr);

/// In-place L2 norm of the q and k heads of every token of a chunk: row (t, r) of the norm lives at
/// h + t*channels + base + r*cols, cols = 128 (the heads of one token are contiguous inside the token's C channels).
/// The same scaling as native_gdn_l2_norm.
void dense_gdn_l2_norm(float* h, int T, int channels, int base, int rows_per_token, int cols, float eps, void* stream);

/// gate[t, h] = softplus(alpha[t, h] + dt[h]) * ssm_a[h];  beta[t, h] = sigmoid(beta[t, h])   (heads = 48)
void dense_gdn_gates(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int heads,
                     int rows, void* stream);

/// The delta-rule recurrence over a chunk, state in registers, block per (value head, 32 state columns), walking the
/// T tokens in order, then the output norm y = rms_norm(o) * gamma * SiLU(z) - SiLU, not the MoE path's sigmoid.
/// h is [T, 2*S*k_heads + S*v_heads], gate/beta [T, v_heads], z and y [T, v_heads * S], state [S, v_heads, S].
/// `snap` works as in dense_gdn_conv_chunk: snap.slot[j] receives the delta-rule state right after column j.  The
/// state is read from and written to memory once per CHUNK instead of once per token, which is what makes a
/// multi-column decode pass cheap: the state is 3 MiB per layer.
void dense_gdn_rec_chunk(float* state, const float* h, const float* gate, const float* beta, const float* z,
                         const float* gamma, float eps, float* y, int k_heads, int v_heads, int T,
                         const DenseGdnSnap& snap = {}, void* stream = nullptr);

}  // namespace strata::kernels
