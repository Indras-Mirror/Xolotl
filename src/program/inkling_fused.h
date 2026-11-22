// Fused decode ops for xolotl-inkling: GGML_OP_CUSTOM nodes with a CUDA implementation (ggml_cuda_custom_op, the
// xolotl ggml patch) and a CPU fallback.  Each replaces a chain of tiny ggml kernels with one, doing the same float
// operations in the same order, so the results match the op chains they replace.
#pragma once
#include "ggml.h"

// out = scale * (x + conv(state ++ x, w) [+ sum_lag win(lag) * dyn(lag, head)]), and the state takes the last K-1
// inputs (written in place).  x [C, T] contiguous, w [K, C], state = a [K-1, C] view of the slot's conv state,
// dyn = nullptr or [K * nh, T] (per-head input-dependent taps).  The chain it replaces: concat + cpy + ssm_conv
// (+ K x (cont transpose + mul + add)) + add + scale.
// hist (optional, --dspark verify) = a [K-1+T, C] view (row per channel) that receives [state ; inputs] for the exact
// rollback of a rejected draft tail, the chain's concat + cpy.
ggml_tensor* inkling_conv_step(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* state, ggml_tensor* dyn,
                               int K, int nh, float scale, ggml_tensor* hist = nullptr);

// MoE finish: res + conv_step(sum_i d[:, i, t] * wk[i, t]) in one kernel - the routing-weighted sum of the k expert
// outputs d [C, k, T] (wk [k, T]), the ffn post-convolution (state/hist as inkling_conv_step, no dynamic taps), and the
// residual add.  Replaces cont + permute + the weighted-sum mat-vec + conv + add.
ggml_tensor* inkling_moe_finish(ggml_context* c, ggml_tensor* d, ggml_tensor* wk, ggml_tensor* w, ggml_tensor* state,
                                ggml_tensor* res, int K, float scale, ggml_tensor* hist = nullptr, ggml_tensor* ds = nullptr,
                                ggml_tensor* ids = nullptr);   // ds [NE] + ids [k, T]: the per-expert down scale, w * ds[id]

// SiTU on a gate|up tensor gu [2F, k, T]: beta*tanh(g/beta)*sigmoid(g) * (lb > 0 ? lb*tanh(u/lb) : u) -> [F, k, T].
// The chain it replaces: 2 conts + scale + tanh + scale + sigmoid + mul (+ scale + tanh + scale) + mul.
// sc [NE] + ids [k, T] (optional): the per-expert gate/up scale applied first (gu * sc[id], as the mul it replaces).
ggml_tensor* inkling_situ(ggml_context* c, ggml_tensor* gu, float beta, float lb, ggml_tensor* sc = nullptr, ggml_tensor* ids = nullptr);

// [E, n, T] from n tensors of [E, T] (contiguous F32, n <= GGML_MAX_SRC): row i = srcs[i]; nullptr if not applicable
ggml_tensor* inkling_stack(ggml_context* c, ggml_tensor* const* srcs, int n);

// thr * l2_norm(x [* scale], eps = thr) [* tau] over rows of x [hd, nh, T] (hd < 1024); scale [hd, nh], tau [T]; the
// mul + l2_norm + scale (+ mul) chain of clip() as one kernel; nullptr if not applicable
ggml_tensor* inkling_clip(ggml_context* c, ggml_tensor* x, ggml_tensor* scale, float thr, ggml_tensor* tau);
// two clips in one launch (q and rel): one buffer [a | b], each part as inkling_clip; same T, shared tau
ggml_tensor* inkling_clip2(ggml_context* c, ggml_tensor* xa, ggml_tensor* sa, float thra, ggml_tensor* xb, ggml_tensor* sb,
                           float thrb, ggml_tensor* tau);

// mix(): rms-normed streams srcs[i] [E, T] (n <= 9), softmax over <stream, query_j>/sqrt(E), weighted sum -> [E, nq, T]
// (query [E, nq], nq <= 4) in one kernel; fp32, a different summation order than the 6-kernel chain (rounding level)
ggml_tensor* inkling_mix(ggml_context* c, ggml_tensor* const* srcs, int n, ggml_tensor* query);

// the relative-position bias mask for attention: M[j, t, h] = (table[rp[j, t], h, t] - sinks[h, t]) + mask[j, t] as
// [cells, T, nh] F32 (or F16 for flash attention) in one kernel; table [Ext+1, nh, T] (contiguous), sinks [nh, T],
// rp/mask [cells, T] (row views ok).  Bit-identical to sub + permute + get_rows + permute + add (+ cast).
ggml_tensor* inkling_relbias(ggml_context* c, ggml_tensor* table, ggml_tensor* sinks, ggml_tensor* rp, ggml_tensor* mask,
                             ggml_type out_type);

// decode / verify attention (T <= 64) over an f16 K/V window read in place: q [HD, nh, T] (contiguous F32), k/v
// [HD, nkv, cells] F16 views (nh % nkv == 0, nh / nkv <= 4, HD 64/128/256), mask = the inkling_relbias F32 output
// [cells, T, nh] (bias + causal/window mask); softmax(q.k * scale + mask) with a zero sink logit, times V -> [HD, nh, T].
// Two kernels (split-K partials + reduce) replace the V transpose copy + K.Q + softmax + V.P chain; fp32 math.
// gate (optional) = [HD * nh, T] F32 pre-activations: the output is multiplied by sigmoid(gate) (the attention output gate).
ggml_tensor* inkling_attn_dec(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale,
                              ggml_tensor* gate = nullptr);

// the MoE weighted sum x [C, T] = sum_i d[:, i, t] * wk[i, t] (d [C, k, T], wk k*T floats), for prompt chunks
ggml_tensor* inkling_moe_wsum(ggml_context* c, ggml_tensor* d, ggml_tensor* wk);

// prompt attention (T > 16), flash style on the tensor cores (port of Basalt attn_prompt.cu, MIT): q [128, nh, T] (nh = 3 nkv),
// k/v = the whole f16 cache [128, nkv, cap] (position p in cell p % cap), R [16, nh, T], sinks [nh, T], relb [16, Ext+1];
// rows at positions pos0..pos0+T-1, window > 0 = sliding.  kst/vst = the cache stores (ordering only).  nullptr = unsupported.
ggml_tensor* inkling_attn_prompt(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* R,
                                 ggml_tensor* sinks, ggml_tensor* relb, ggml_tensor* kst, ggml_tensor* vst, int64_t pos0,
                                 int window, float scale);

// inkling_attn_dec with the relative bias computed in-kernel instead of a [cells, T, nh] mask: bias(j, h) =
// (relb[:, rp[j]] . R[:, h, t] - sinks[h, t]) + mask[j, t]; relb = the padded table [n_rel, Ext+1] (n_rel <= 32),
// R [n_rel, nh, T] (the clipped relative state), sinks [nh, T], rp/mask [cells, T] row views (the host plan).
// Replaces the table mat-mul + inkling_relbias + the mask read.
ggml_tensor* inkling_attn_dec_rel(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* relb,
                                  ggml_tensor* Rt, ggml_tensor* sinks, ggml_tensor* rp, ggml_tensor* mask, float scale,
                                  ggml_tensor* gate = nullptr);
// the same for a positional slot (cell j = position j, the global layers): the mask (j <= pos[t]) and the distance
// (min(pos[t] - j, ext)) come from pos [T] (I64, the step's positions) inside the kernel - no host [cells, T] arrays.
// Only where inkling_dec_pos_ok() holds (the kernel that takes it); else nullptr.
bool inkling_dec_pos_ok(int hd, int nh, int nkv, ggml_type kt, ggml_type vt, int nrel, size_t knb1, size_t knb2);
ggml_tensor* inkling_attn_dec_rel_pos(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* relb,
                                      ggml_tensor* Rt, ggml_tensor* sinks, ggml_tensor* pos, int ext, float scale,
                                      ggml_tensor* gate = nullptr);

// attention prepare (key/value side) in one kernel: K and V convolutions (xk/xv [C, T] rows, C = hd*nkv; state = the
// slot's whole conv state, K at row 0 and V at row off_v; dynk/dynv = the global layers' dynamic taps or nullptr), the
// key clip and value RMS norm per head row, the writes of both into kcache/vcache [hd, nkv, cells] at cells idx [T]
// (I64; f16, q8_0 or q4_0 rows, quantised as ggml's set_rows does), and the optional verify history (hist = the slot's
// conv_hist).  rotk / rotv = the [hd, hd] cache rotation M = H diag(s)/sqrt(hd) or nullptr, per leg: that leg's rows
// are rotated before the write (the kernel reads row 0 = s/sqrt(hd) and runs the FWHT).  Returns a 1-element tensor to expand into the graph
// BEFORE the attention reads the caches.  Replaces 2 conv + clip + rms_norm (+ 2 rotations) + 2 set_rows.
ggml_tensor* inkling_attn_prep(ggml_context* c, ggml_tensor* xk, ggml_tensor* xv, ggml_tensor* wk, ggml_tensor* wv,
                               ggml_tensor* state, int off_v, ggml_tensor* dynk, ggml_tensor* dynv, int K, float scale,
                               ggml_tensor* idx, ggml_tensor* kcache, ggml_tensor* vcache, float thr, float eps,
                               ggml_tensor* hist, ggml_tensor* rotk = nullptr, ggml_tensor* rotv = nullptr);

// the loop controller for K controllers and T <= 16 tokens: rms_norm(cur - f * cur + g * msg) * w with
// [g; f] = tanh(U (D [rms(cur); rms(msg)])).  cur [E, T, K] or [E, T] (shared by every k), msg [E, T, K] (strided
// views are fine), D [2E, r, K] BF16, U [r, 2E, K] BF16 (gain rows then forget rows), w [E] or [E, 1, K].
ggml_tensor* inkling_ctrl(ggml_context* c, ggml_tensor* cur, ggml_tensor* msg, ggml_tensor* down, ggml_tensor* up,
                          ggml_tensor* w, float eps);

// true when the ggml build runs these on CUDA (the custom-op hook is present); else the engine keeps the op chains
bool inkling_fused_available();
