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
    ggml_tensor * wqkv_z = nullptr;   // [D, 2*hd*nk + hd*nv + hd*nv]
    ggml_tensor * conv_w = nullptr;   // [d_conv, channels]
    ggml_tensor * w_ba   = nullptr;   // [D, 2*n_v]
    ggml_tensor * ssm_a  = nullptr;   // [n_v]  (negative: -A_log.exp())
    ggml_tensor * dt     = nullptr;   // [n_v]  (sigmoid-free bias added to alpha)
    ggml_tensor * norm_w = nullptr;   // [hd]
    ggml_tensor * w_out  = nullptr;   // [hd*n_v, D]
    ggml_tensor * ident  = nullptr;   // [CS, CS] identity input
};

// returns [D, T]
inline ggml_tensor * qfn_gdn_layer(ggml_context * c, ggml_tensor * cur /* [D, T] */, const QfnLayer & w,
                                   int64_t hd, int64_t nk, int64_t nv, int64_t d_conv, float rms_eps) {
    const int64_t D = cur->ne[0], T = cur->ne[1];
    const int64_t n_qkv = 2 * hd * nk + hd * nv, channels = n_qkv, n_z = hd * nv;

    // ---- projection + split (build_qkvz) ----
    ggml_tensor * qkvz = ggml_mul_mat(c, w.wqkv_z, cur);                       // [n_qkv + n_z, T]
    ggml_tensor * qkv = ggml_view_2d(c, qkvz, n_qkv, T, qkvz->nb[1], 0);
    ggml_tensor * zt  = ggml_view_2d(c, qkvz, n_z, T, qkvz->nb[1], (size_t) n_qkv * ggml_element_size(qkvz));

    // ---- causal conv, zero history ----
    // ssm_conv takes the conv window on dim0 and channels on dim1: [d_conv-1+T, channels]
    ggml_tensor * conv_in = ggml_cont(c, ggml_pad_ext(c, ggml_cont(c, ggml_transpose(c, qkv)),
                                                      (int) (d_conv - 1), 0, 0, 0, 0, 0, 0, 0));
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
    ggml_tensor * o = gdn_chunked(c, q, k, ggml_cont(c, v), gate, beta, w.ident);   // [hd, n_v, T]
    o = ggml_reshape_3d(c, o, hd, nv, T);

    // ---- gated norm + output projection ----
    ggml_tensor * z = ggml_reshape_4d(c, ggml_cont(c, zt), hd, nv, T, 1);
    ggml_tensor * on = ggml_mul(c, ggml_rms_norm(c, o, rms_eps), ggml_repeat_4d(c, w.norm_w, hd, nv, T, 1));
    on = ggml_mul(c, on, ggml_sigmoid(c, z));
    ggml_tensor * flat = ggml_reshape_2d(c, on, hd * nv, T);
    ggml_tensor * out = ggml_mul_mat(c, w.w_out, flat);                        // [D, T]
    return ggml_add(c, cur, out);
}

}  // namespace xolotl::train
