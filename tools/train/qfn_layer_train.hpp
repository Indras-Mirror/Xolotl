// One QFN (qwen4exp) gated-delta-net layer as a differentiable training graph.
//
// Follows qwen4exp.cpp build_layer_attn_linear (1100-1240) between the layer input and the residual add:
//   qkvz = wqkv_z^T cur, split into qkv_mixed (2*hd*nk + hd*nv rows) and z (hd*nv rows)
//   conv_in = qkv_mixed with d_conv-1 ZERO rows prepended (a training sequence starts at position 0)
//   y = silu(ssm_conv(conv_in, conv_w)); q | k | v split by channel, l2-normed q,k, tiled to n_v heads
//   beta  = sigmoid(rows [0, n_v)   of w_ba^T cur)      alpha = rows [n_v, 2*n_v)  (the alpha view's offset is
//   exactly n_v elements in qwen4exp.cpp:1133, which pins the order)
//   gate = softplus(alpha + dt) * ssm_a        (ssm_a is stored as -A_log.exp(), i.e. negative -> log decay <= 0)
//   o = gdn_chunked(q, k, v, gate, beta, I);  o = rms_norm(o, ssm_norm) * sigmoid(z)   (sigmoid, not silu)
//   out = ssm_out^T o;  layer_out = cur + out
//
// Every op here has a backward rule (see ggml.c) - the gates are tools/train/test_qfn_layer.cpp (FD) and
// tools/train/qfn_layer_ref.py (forward reference).
#pragma once

#include "ggml.h"
#include "gdn_train.hpp"

#include <cmath>

namespace xolotl::train {

struct QfnLayer {
    // weights (all F32 in the trainer)
    // the pack stores qkv and z (the gate) as SEPARATE tensors - an adapter can only attach to a real base
    // tensor, so they stay separate here too (a lora on the concatenation would not pair with the model).
    ggml_tensor * wqkv   = nullptr;   // [D, 2*hd*nk + hd*nv]   (blk.<l>.attn_qkv.weight)
    ggml_tensor * wgate  = nullptr;   // [D, hd*nv]             (blk.<l>.attn_gate.weight)
    ggml_tensor * conv_w = nullptr;   // [d_conv, channels]
    ggml_tensor * w_ba   = nullptr;   // [D, 2*n_v]
    ggml_tensor * ssm_a  = nullptr;   // [n_v]  (negative: -A_log.exp())
    ggml_tensor * dt     = nullptr;   // [n_v]  (sigmoid-free bias added to alpha)
    ggml_tensor * norm_w = nullptr;   // [hd]
    ggml_tensor * w_out  = nullptr;   // [hd*n_v, D]
    ggml_tensor * ident  = nullptr;   // [CS, CS] identity input
    // optional LoRA adapters (A [n_in, r], B [r, n_out]; delta = (alpha/r) * B A)
    ggml_tensor * lora_a_qkv = nullptr;  ggml_tensor * lora_b_qkv = nullptr;
    ggml_tensor * lora_a_gate = nullptr; ggml_tensor * lora_b_gate = nullptr;
    ggml_tensor * lora_a_out = nullptr;  ggml_tensor * lora_b_out = nullptr;
    float lora_alpha = 16.0f;
    int64_t lora_rank = 8;
};

// y = W^T x (+ adapter): the adapter is applied to a frozen base, exactly as llama.cpp's --lora does.
inline ggml_tensor * qfn_linear(ggml_context * c, ggml_tensor * W, ggml_tensor * x,
                                ggml_tensor * A, ggml_tensor * B, float alpha, int64_t rank) {
    ggml_tensor * y = ggml_mul_mat(c, W, x);
    if (A && B) y = ggml_add(c, y, ggml_scale(c, ggml_mul_mat(c, B, ggml_mul_mat(c, A, x)), alpha / (float) rank));
    return y;
}

// What a GDN layer carries across a split (a shared prefix -> a window): the recurrent state and the conv history.
// Inputs are constants (no gradient into the prefix); outputs are filled when the block runs the prefix itself.
struct GdnCarry {
    ggml_tensor * conv_in = nullptr;    // [d_conv-1, channels] the prefix's last qkv rows (time on dim0), or nullptr = zeros
    ggml_tensor * s_in = nullptr;       // [hd, hd, 1, n_v] recurrent state, or nullptr = zeros
    ggml_tensor * conv_out = nullptr;   // filled: [d_conv-1, channels] this block's last qkv rows
    ggml_tensor * s_out = nullptr;      // filled: the state after the last token
};

// The layer's own contribution to the residual (no residual add), [D, T].
inline ggml_tensor * qfn_gdn_block(ggml_context * c, ggml_tensor * cur /* [D, T] */, const QfnLayer & w,
                                   int64_t hd, int64_t nk, int64_t nv, int64_t d_conv, float rms_eps,
                                   GdnCarry * carry = nullptr) {
    const int64_t D = cur->ne[0], T = cur->ne[1];
    const int64_t n_qkv = 2 * hd * nk + hd * nv, channels = n_qkv, n_z = hd * nv;

    // ---- projection + split (build_qkvz) ----
    ggml_tensor * qkv = qfn_linear(c, w.wqkv, cur, w.lora_a_qkv, w.lora_b_qkv, w.lora_alpha, w.lora_rank);   // [n_qkv, T]
    ggml_tensor * zt  = qfn_linear(c, w.wgate, cur, w.lora_a_gate, w.lora_b_gate, w.lora_alpha, w.lora_rank); // [n_z, T]

    // ---- causal conv, zero history ----
    // ssm_conv takes the conv window on dim0 and channels on dim1: [d_conv-1+T, channels]
    ggml_tensor * qkv_t = ggml_cont(c, ggml_transpose(c, qkv));                     // [T, channels]
    ggml_tensor * conv_in = carry && carry->conv_in
        ? ggml_concat(c, carry->conv_in, qkv_t, 0)                                  // prefix history, then the window
        : ggml_cont(c, ggml_pad_ext(c, qkv_t, (int) (d_conv - 1), 0, 0, 0, 0, 0, 0, 0));
    if (carry && T >= d_conv - 1)
        carry->conv_out = ggml_cont(c, ggml_view_2d(c, qkv_t, d_conv - 1, channels, qkv_t->nb[1],
                                                    (size_t) (T - (d_conv - 1)) * ggml_element_size(qkv_t)));
    ggml_tensor * y = ggml_silu(c, ggml_ssm_conv(c, conv_in, w.conv_w));      // [channels, T]

    const size_t es = ggml_element_size(y), row = ggml_row_size(y->type, channels);
    ggml_tensor * q = ggml_view_4d(c, y, hd, nk, T, 1, hd * es, row, row * T, 0);
    ggml_tensor * k = ggml_view_4d(c, y, hd, nk, T, 1, hd * es, row, row * T, (size_t) hd * nk * es);
    ggml_tensor * v = ggml_view_4d(c, y, hd, nv, T, 1, hd * es, row, row * T, (size_t) 2 * hd * nk * es);

    // l2 norm (models.h build_gdn_l2_norm): rms_norm(x, eps/n) * 1/sqrt(n)
    auto l2 = [&](ggml_tensor * t) {
        return ggml_scale(c, ggml_rms_norm(c, t, rms_eps / (float) hd), 1.0f / std::sqrt((float) hd));
    };
    q = l2(ggml_cont(c, q)); k = l2(ggml_cont(c, k));
    q = ggml_repeat_4d(c, q, hd, nv, T, 1);      // model: repeat tiles -> v head j reads k head j % nk
    k = ggml_repeat_4d(c, k, hd, nv, T, 1);

    // ---- beta / alpha / gate ----
    ggml_tensor * ba = ggml_mul_mat(c, w.w_ba, cur);                          // [2*n_v, T]
    ggml_tensor * beta_l = ggml_view_2d(c, ba, nv, T, ba->nb[1], 0);
    ggml_tensor * alpha  = ggml_view_2d(c, ba, nv, T, ba->nb[1], (size_t) nv * ggml_element_size(ba));
    ggml_tensor * beta = ggml_sigmoid(c, ggml_reshape_4d(c, ggml_cont(c, beta_l), 1, nv, T, 1));
    ggml_tensor * gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, ggml_cont(c, alpha),
                                                               ggml_repeat_4d(c, w.dt, nv, T, 1, 1))),
                                  ggml_repeat_4d(c, w.ssm_a, nv, T, 1, 1));
    gate = ggml_reshape_4d(c, gate, 1, nv, T, 1);                             // [1, n_v, T]

    // ---- the delta-net core ----
    ggml_tensor * o = gdn_chunked(c, q, k, ggml_cont(c, v), gate, beta, w.ident,
                                  carry ? carry->s_in : nullptr, carry ? &carry->s_out : nullptr);   // [hd, n_v, T]
    o = ggml_reshape_3d(c, o, hd, nv, T);

    // ---- gated norm + output projection ----
    ggml_tensor * z = ggml_reshape_4d(c, ggml_cont(c, zt), hd, nv, T, 1);
    ggml_tensor * on = ggml_mul(c, ggml_rms_norm(c, o, rms_eps), ggml_repeat_4d(c, w.norm_w, hd, nv, T, 1));
    on = ggml_mul(c, on, ggml_sigmoid(c, z));
    ggml_tensor * flat = ggml_reshape_2d(c, on, hd * nv, T);
    return qfn_linear(c, w.w_out, flat, w.lora_a_out, w.lora_b_out, w.lora_alpha, w.lora_rank);  // [D, T]
}

// returns [D, T] - the block plus a plain residual (the single-stream form, no hyper-connections)
inline ggml_tensor * qfn_gdn_layer(ggml_context * c, ggml_tensor * cur /* [D, T] */, const QfnLayer & w,
                                   int64_t hd, int64_t nk, int64_t nv, int64_t d_conv, float rms_eps) {
    return ggml_add(c, cur, qfn_gdn_block(c, cur, w, hd, nk, nv, d_conv, rms_eps));
}

}  // namespace xolotl::train

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connections (qwen4exp.cpp build_hc_mix 347-414 / build_hc_combine 416-444, NON-fused paths).
// The model keeps hc parallel residual streams [n_embd, hc, T]; every layer mixes them to one stream, runs its
// block, and scatters the result back into the streams. This is the form the served engine dumps per layer
// (--dump-layers hardware slot = hc x n_embd), so the trainer needs it to consume real activations.
// ---------------------------------------------------------------------------------------------------------------
struct QfnHc {
    ggml_tensor * norm   = nullptr;   // [D, hc]   gamma, per stream (stored [D*hc], created ne [D, hc])
    ggml_tensor * down   = nullptr;   // [D*hc, hc_lr]
    ggml_tensor * up     = nullptr;   // [hc_lr, D*hc]
    ggml_tensor * inject = nullptr;   // [D*hc, hc]
};

// x [D, hc, T] -> mixed [D, T]; *inject_out gets the [hc, T] scatter weights (nullptr to skip)
inline ggml_tensor * qfn_hc_mix(ggml_context * c, ggml_tensor * x, const QfnHc & w, int64_t hc, float rms_eps,
                                ggml_tensor ** inject_out) {
    const int64_t D = x->ne[0], T = x->ne[2];
    const int64_t hc_dim = D * hc;

    // grouped RMSNorm: rms over ONE stream (dim0 = D), then the [D, hc] gamma
    ggml_tensor * xn = ggml_mul(c, ggml_rms_norm(c, x, rms_eps), w.norm);          // [D, hc, T]
    ggml_tensor * xn2 = ggml_reshape_2d(c, ggml_cont(c, xn), hc_dim, T);           // [D*hc, T]

    ggml_tensor * lo = ggml_scale(c, ggml_mul_mat(c, w.down, xn2), 1.0f / (float) hc);
    lo = ggml_silu(c, lo);                                                          // [hc_lr, T]
    ggml_tensor * gate = ggml_mul_mat(c, w.up, lo);                                 // [D*hc, T]

    // sigmoid gate (qwen4exp.cpp build_hc_mix: ggml_mul(xn, ggml_sigmoid(gate))). It was MISSING until 21 Nov 09:30:
    // the FD gate passed without it (analytic and numeric were wrong together); the real-data gate
    // (qfn_topk_gate vs captures) caught it - lesson 3 again: every piece needs a forward reference, not only FD.
    ggml_tensor * gated = ggml_mul(c, xn, ggml_sigmoid(c, ggml_reshape_3d(c, ggml_cont(c, gate), D, hc, T)));
    // collapse the streams by their mean (same loop as the model's hc_stream_mean: views over dim1)
    ggml_tensor * mixed = ggml_cont(c, ggml_view_2d(c, gated, D, T, (size_t) D * hc * sizeof(float), 0));
    for (int64_t s = 1; s < hc; ++s)
        mixed = ggml_add(c, mixed, ggml_view_2d(c, gated, D, T, (size_t) D * hc * sizeof(float), (size_t) s * D * sizeof(float)));
    mixed = ggml_scale(c, mixed, 1.0f / (float) hc);                                // [D, T]

    if (inject_out)
        *inject_out = ggml_scale(c, ggml_mul_mat(c, w.inject, xn2), 1.0f / (float) hc);   // [hc, T]
    return mixed;
}

// residual [D, hc, T], block_out [D, T], inject [hc, T] -> [D, hc, T]
// 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add.
inline ggml_tensor * qfn_hc_combine(ggml_context * c, ggml_tensor * residual, ggml_tensor * block_out,
                                    ggml_tensor * inject, int64_t hc) {
    const int64_t D = residual->ne[0], T = residual->ne[2];
    ggml_tensor * wt = ggml_scale(c, ggml_sigmoid(c, inject), 2.0f);                 // [hc, T]
    wt = ggml_reshape_3d(c, ggml_cont(c, wt), 1, hc, T);
    ggml_tensor * b = ggml_reshape_3d(c, block_out, D, 1, T);
    b = ggml_repeat_4d(c, b, D, hc, T, 1);
    return ggml_add(c, residual, ggml_mul(c, b, wt));
}
