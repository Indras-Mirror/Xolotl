// Differentiable chunked gated delta net for training graphs (ggml ops that all have backward rules).
// Port of llama.cpp src/models/delta-net-base.cpp build_delta_net_chunking (non-KDA path: g is one log-decay per head
// and token), with two changes so ggml autodiff can run through it:
//   - the identity matrix is an INPUT (caller fills it with I_CS) instead of ggml_fill + ggml_diag (no backward),
//   - the per-chunk outputs are concatenated instead of ggml_set_inplace into v (in-place ops break backward graphs).
// The initial recurrent state is zero (training sequences start at position 0), so the first chunk has no state terms.
// Every broadcast operand that carries a gradient (b, g) is expanded with ggml_repeat_4d first: ggml's ADD/SUB/MUL
// backward assumes equal shapes.  Needs the Xolotl backward rules for SOLVE_TRI, TRI, CUMSUM, PAD and the cont fixes
// in PERMUTE/TRANSPOSE/SCALE backward (ggml.c).
// Gate: tools/train/test_gdn_chunked.cpp (forward vs tools/train/gdn_ref.py, central-FD gradients on q,k,v,g,b).
//
//   q, k : [S_k, H, n_tokens, 1]       v : [S_v, H, n_tokens, 1]  (S_k == S_v; q/k already expanded to the v heads)
//   Expansion rule (QFN: 16 k heads, 48 v heads): llama.cpp does ggml_repeat_4d(q_conv, head_dim, n_v_heads, T, S), which
//   TILES - v head j uses k head j % n_k_heads (verified: repeat(3 -> 9) gives 10 20 30 10 20 30 10 20 30). Do the same
//   here and let REPEAT's backward sum the copies.
//   g    : [1, H_v, n_tokens, 1]  (log decay, <= 0)    b : [1, H_v, n_tokens, 1] (beta, after sigmoid)
//   ident: [CS, CS] F32 identity (CS = 64)
// returns o : [S_v, H_v, n_tokens, 1]  (q is scaled by 1/sqrt(S_k) inside, as llama.cpp)
#pragma once

#include "ggml.h"

#include <cmath>

namespace xolotl::train {

constexpr int kGdnChunk = 64;

inline ggml_tensor * gdn_slice(ggml_context * c, ggml_tensor * t, int64_t chunk) {   // llama.cpp get_slice_2d
    return ggml_view_4d(c, t, t->ne[0], t->ne[1], 1, t->ne[3], t->nb[1], t->nb[2], t->nb[3], (size_t) chunk * t->nb[2]);
}

// s0: optional initial recurrent state [S_v, S_v, 1, H_v] (the state a shared prefix leaves; a constant input - no
// gradient flows into the prefix). Same layout as the running state `s` below. nullptr = zero (a sequence from 0).
// s_out: optional, receives the state after the last chunk (for computing a prefix state itself).
inline ggml_tensor * gdn_chunked(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * g,
                                 ggml_tensor * b, ggml_tensor * ident, ggml_tensor * s0 = nullptr,
                                 ggml_tensor ** s_out = nullptr) {
    const int64_t S_k = q->ne[0], H_k = q->ne[1], n_tokens = q->ne[2];
    const int64_t S_v = v->ne[0], H_v = v->ne[1];
    GGML_ASSERT(S_k == S_v && H_v == H_k && q->ne[3] == 1);   // caller expands q/k to the v heads (model-specific mapping)
    const int64_t CS = kGdnChunk;
    q = ggml_scale(c, q, 1.0f / std::sqrt((float) S_k));
    q = ggml_permute(c, q, 0, 2, 1, 3);   // [S_k, n_tokens, H_k, 1]
    k = ggml_permute(c, k, 0, 2, 1, 3);
    v = ggml_permute(c, v, 0, 2, 1, 3);   // [S_v, n_tokens, H_v, 1]
    g = ggml_permute(c, g, 0, 2, 1, 3);   // [1, n_tokens, H_v, 1]
    b = ggml_permute(c, b, 0, 2, 1, 3);
    const int pad = (int) ((CS - n_tokens % CS) % CS);
    const int64_t n_chunks = (n_tokens + pad) / CS;
    q = ggml_pad(c, ggml_cont(c, q), 0, pad, 0, 0);
    k = ggml_pad(c, ggml_cont(c, k), 0, pad, 0, 0);
    v = ggml_pad(c, ggml_cont(c, v), 0, pad, 0, 0);
    g = ggml_pad(c, ggml_cont(c, g), 0, pad, 0, 0);
    b = ggml_pad(c, ggml_cont(c, b), 0, pad, 0, 0);
    // b and g carry gradients, so expand them to full shape with REPEAT before multiplying: ggml's MUL
    // backward asserts equal shapes, and REPEAT's backward (repeat_back) sums the copies back.
    ggml_tensor * b_full = ggml_repeat_4d(c, b, S_v, n_tokens + pad, H_v, 1);
    ggml_tensor * v_b = ggml_mul(c, v, b_full);
    ggml_tensor * k_b = ggml_mul(c, k, b_full);
    q   = ggml_reshape_4d(c, q,   S_k, CS, n_chunks, H_k);
    k   = ggml_reshape_4d(c, k,   S_k, CS, n_chunks, H_k);
    k_b = ggml_reshape_4d(c, k_b, S_k, CS, n_chunks, H_v);
    v   = ggml_reshape_4d(c, v,   S_v, CS, n_chunks, H_v);
    v_b = ggml_reshape_4d(c, v_b, S_v, CS, n_chunks, H_v);
    g   = ggml_reshape_4d(c, g,   1,   CS, n_chunks, H_v);

    ggml_tensor * g_cs = ggml_cumsum(c, ggml_cont(c, ggml_transpose(c, g)));                 // [CS, 1, n_chunks, H_v]
    // both operands are repeated to the full shape: ggml's SUB backward assumes equal shapes (it would
    // hand the [CS,CS] grad straight to a [CS,1] operand), so the broadcast must be explicit.
    ggml_tensor * g_cs_j = ggml_repeat_4d(c, ggml_reshape_4d(c, g_cs, 1, CS, n_chunks, H_v), CS, CS, n_chunks, H_v);
    ggml_tensor * g_cs_i = ggml_repeat_4d(c, g_cs, CS, CS, n_chunks, H_v);
    ggml_tensor * decay = ggml_exp(c, ggml_tri(c, ggml_sub(c, g_cs_j, g_cs_i), GGML_TRI_TYPE_LOWER_DIAG));
    ggml_tensor * kb = ggml_mul(c, ggml_mul_mat(c, k, k_b), decay);                            // [CS, CS, n_chunks, H_v]
    ggml_tensor * kq = ggml_tri(c, ggml_mul(c, ggml_mul_mat(c, k, q), decay), GGML_TRI_TYPE_LOWER_DIAG);
    ggml_tensor * attn = ggml_tri(c, kb, GGML_TRI_TYPE_LOWER);
    ggml_tensor * lhs = ggml_add(c, attn, ident);
    ggml_tensor * lin = ggml_solve_tri(c, lhs, ggml_neg(c, attn), true, true, false);
    attn = ggml_add(c, lin, ident);
    v = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, v_b)), attn);                        // [S_v, CS, n_chunks, H_v]
    ggml_tensor * g_exp = ggml_exp(c, g_cs);                                                  // [CS, 1, n_chunks, H_v]
    ggml_tensor * kbg = ggml_mul(c, ggml_cont(c, ggml_transpose(c, k_b)),
                                 ggml_repeat_4d(c, g_exp, CS, S_k, n_chunks, H_v));
    ggml_tensor * k_cd = ggml_mul_mat(c, kbg, attn);                                          // [S_k, CS, n_chunks, H_v]
    ggml_tensor * q_g_exp = ggml_mul(c, q,
                                     ggml_repeat_4d(c, ggml_cont(c, ggml_transpose(c, g_exp)), S_k, CS, n_chunks, H_k));
    ggml_tensor * g_last = ggml_view_4d(c, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3], g_cs->nb[1], g_cs->nb[2], g_cs->nb[3],
                                        ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    ggml_tensor * g_last_exp_t = ggml_cont(c, ggml_transpose(c, ggml_exp(c, g_last)));       // [1, 1, n_chunks, H_v]
    ggml_tensor * gl_full = ggml_repeat_4d(c, g_last, CS, 1, n_chunks, H_v);                  // SUB backward needs equal shapes
    ggml_tensor * g_diff_exp_t = ggml_cont(c, ggml_transpose(c, ggml_exp(c, ggml_neg(c, ggml_sub(c, g_cs, gl_full)))));
    ggml_tensor * kg = ggml_mul(c, k, ggml_repeat_4d(c, g_diff_exp_t, S_k, CS, n_chunks, H_k));
    ggml_tensor * kg_t = ggml_cont(c, ggml_transpose(c, kg));                               // [CS, S_k, n_chunks, H_v]
    ggml_tensor * v_t = ggml_cont(c, ggml_transpose(c, v));                                   // [CS, S_v, n_chunks, H_v]

    ggml_tensor * s = s0;        // [S_v, S_v, 1, H_v]; zero (nullptr) before the first chunk unless a prefix state is given
    ggml_tensor * out = nullptr;
    for (int64_t ch = 0; ch < n_chunks; ++ch) {
        ggml_tensor * c_v_t = gdn_slice(c, v_t, ch);
        ggml_tensor * c_kq = gdn_slice(c, kq, ch);
        ggml_tensor * v_new = c_v_t;
        ggml_tensor * o = nullptr;
        if (s) {
            v_new = ggml_sub(c, c_v_t, ggml_mul_mat(c, gdn_slice(c, k_cd, ch), s));
            o = ggml_add(c, ggml_mul_mat(c, s, gdn_slice(c, q_g_exp, ch)), ggml_mul_mat(c, v_new, c_kq));
        } else {
            o = ggml_mul_mat(c, ggml_cont(c, v_new), c_kq);
        }
        out = out ? ggml_concat(c, out, o, 2) : o;                                            // [S_v, CS, ch+1, H_v]
        if (ch + 1 < n_chunks || s_out) {
            ggml_tensor * kgv = ggml_mul_mat(c, gdn_slice(c, kg_t, ch), ggml_cont(c, v_new));  // [S_k, S_v, 1, H_v]
            ggml_tensor * gl = gdn_slice(c, g_last_exp_t, ch);
            s = s ? ggml_add(c, ggml_mul(c, s, ggml_repeat_4d(c, gl, S_v, S_v, 1, H_v)), kgv) : kgv;
        }
    }
    // padded rows have k = 0 and g = 0, so they add nothing to the state and do not decay it: s_out is the state after
    // the last REAL token for any n_tokens (gate: test_gdn_state.cpp, split at a chunk boundary and inside a chunk)
    if (s_out) *s_out = s;
    // [S_v, CS*n_chunks, H_v] -> drop the padding -> [S_v, H_v, n_tokens]
    out = ggml_reshape_3d(c, ggml_cont(c, out), S_v, CS * n_chunks, H_v);
    out = ggml_view_3d(c, out, S_v, n_tokens, H_v, out->nb[1], out->nb[2], 0);
    return ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3));
}

}  // namespace xolotl::train
