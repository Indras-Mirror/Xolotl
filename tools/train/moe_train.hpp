// Differentiable MoE FFN for the QFN trainer (replacement for MUL_MAT_ID, which has no backward rule).
//
// Mirrors llama-graph.cpp build_moe_ffn as qwen4exp calls it: logits = gate_inp^T cur, softmax, top-k
// (selection is host-side and constant - it is zero-grad for the model too, TOP_K/ARGSORT are marked so), weights
// gathered from probs, normalised over the k selected (norm_w = true for QFN, with the model's F16-min clamp) and
// optionally scaled, then one pass per used expert: get_rows the expert's token rows -> up/gate mul_mat -> SILU ->
// down mul_mat -> * routing weight -> get_rows_back scatter-add. QFN has no swiglu clamp and expert_weights_scale=0.
//
// The expert matrices are ordinary tensors (F32 in the trainer, the pack's quantized slices in a serve check);
// gradients w.r.t. cur, the router and the expert weights all flow. Standardize with llama-graph.cpp:2450-2510 and
// the Hy4 trainer's identical loop (tools/hy4/engine/hy4_dense.cpp:948-972).
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xolotl::train {

// Host-side routing: softmax over the router logits, then the k best experts per token.
// logits: [E, T] (token is dim 1), ids: [K*T] as token*K + slot, weights: [K*T] the softmax probs of the selected.
inline void moe_route(const float * logits, int64_t E, int64_t T, int64_t K,
                      std::vector<int32_t> & ids, std::vector<float> & weights) {
    ids.assign((size_t) (K * T), 0);
    weights.assign((size_t) (K * T), 0.0f);
    std::vector<float> p((size_t) E);
    for (int64_t t = 0; t < T; ++t) {
        float mx = -INFINITY;
        for (int64_t e = 0; e < E; ++e) mx = std::fmax(mx, logits[(size_t) (e * T + t)]);
        double sum = 0;
        for (int64_t e = 0; e < E; ++e) { p[(size_t) e] = std::exp(logits[(size_t) (e * T + t)] - mx); sum += p[(size_t) e]; }
        for (int64_t e = 0; e < E; ++e) p[(size_t) e] = (float) (p[(size_t) e] / sum);
        for (int64_t k = 0; k < K; ++k) {
            int64_t best = -1;
            float bv = -1.0f;
            for (int64_t e = 0; e < E; ++e) {
                if (p[(size_t) e] <= bv) continue;
                bool used = false;
                for (int64_t q = 0; q < k; ++q) used = used || ids[(size_t) (t * K + q)] == e;
                if (!used) { bv = p[(size_t) e]; best = e; }
            }
            ids[(size_t) (t * K + k)] = (int32_t) best;
            weights[(size_t) (t * K + k)] = bv;
        }
    }
}

// Per-expert index lists for the graph: for each expert, the token rows it serves and the flat (t*K+k) weight ids.
inline void moe_expert_lists(const std::vector<int32_t> & ids, int64_t K, int64_t T, int64_t E,
                             std::vector<std::vector<int32_t>> & tok_of_expert,
                             std::vector<std::vector<int32_t>> & pair_of_expert) {
    tok_of_expert.assign((size_t) E, {});
    pair_of_expert.assign((size_t) E, {});
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            const int32_t e = ids[(size_t) (t * K + k)];
            tok_of_expert[(size_t) e].push_back((int32_t) t);
            pair_of_expert[(size_t) e].push_back((int32_t) (t * K + k));
        }
}

// Index tensors the graph reads: all constant (selection is not differentiated).
struct MoeIndex {
    ggml_tensor * wsel = nullptr;                 // I32 [K*T]: flat index t*E + expert, for get_rows on probs
    std::vector<ggml_tensor *> tok, pair;         // per expert: I32 [ne]; empty experts have nullptr
    // the same lists concatenated in expert order (one gather / one scatter for the whole layer)
    ggml_tensor * tok_all = nullptr, * pair_all = nullptr;   // I32 [K*T]
    std::vector<int64_t> off, cnt;                            // per expert: first row in tok_all, row count
};

// ids: expert per (t, k) as t*K + k -> the flat index into probs is t*E + e, built here so the caller cannot
// confuse the two (the FD gate cannot catch that mistake: the analytic and numeric graphs would both be wrong).
inline MoeIndex moe_make_index(ggml_context * c, const std::vector<int32_t> & ids, int64_t K, int64_t T, int64_t E,
                               const std::vector<std::vector<int32_t>> & tok_of_expert,
                               const std::vector<std::vector<int32_t>> & pair_of_expert) {
    MoeIndex ix;
    ix.wsel = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) ids.size());
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k)
            ((int32_t *) ix.wsel->data)[(size_t) (t * K + k)] = (int32_t) (t * E + ids[(size_t) (t * K + k)]);
    ix.tok.resize(tok_of_expert.size());
    ix.pair.resize(pair_of_expert.size());
    {
        std::vector<int32_t> ta, pa;
        ix.off.assign(tok_of_expert.size(), 0);
        ix.cnt.assign(tok_of_expert.size(), 0);
        for (size_t e = 0; e < tok_of_expert.size(); ++e) {
            ix.off[e] = (int64_t) ta.size();
            ix.cnt[e] = (int64_t) tok_of_expert[e].size();
            ta.insert(ta.end(), tok_of_expert[e].begin(), tok_of_expert[e].end());
            pa.insert(pa.end(), pair_of_expert[e].begin(), pair_of_expert[e].end());
        }
        ix.tok_all = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) ta.size());
        ix.pair_all = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) pa.size());
        std::memcpy(ix.tok_all->data, ta.data(), ta.size() * sizeof(int32_t));
        std::memcpy(ix.pair_all->data, pa.data(), pa.size() * sizeof(int32_t));
    }
    for (size_t e = 0; e < tok_of_expert.size(); ++e) {
        if (tok_of_expert[e].empty()) continue;
        const int64_t ne = (int64_t) tok_of_expert[e].size();
        ix.tok[e] = ggml_new_tensor_1d(c, GGML_TYPE_I32, ne);
        std::memcpy(ix.tok[e]->data, tok_of_expert[e].data(), (size_t) ne * sizeof(int32_t));
        ix.pair[e] = ggml_new_tensor_1d(c, GGML_TYPE_I32, ne);
        std::memcpy(ix.pair[e]->data, pair_of_expert[e].data(), (size_t) ne * sizeof(int32_t));
    }
    return ix;
}

// cur [n_embd, T]  gate_inp [n_embd, E]  up/gate_exps [n_embd, n_ff, E]  down_exps [n_ff, n_embd, E]
// -> [n_embd, T]
inline ggml_tensor * moe_ffn(ggml_context * c, ggml_tensor * cur, ggml_tensor * gate_inp,
                             ggml_tensor * up_exps, ggml_tensor * gate_exps, ggml_tensor * down_exps,
                             const MoeIndex & ix, bool norm_w, float w_scale) {
    const int64_t n_embd = cur->ne[0], T = cur->ne[1];
    const int64_t E = gate_inp->ne[1], K = ix.wsel->ne[0] / T;
    const int64_t n_ff = up_exps->ne[1];

    ggml_tensor * probs = ggml_soft_max(c, ggml_mul_mat(c, gate_inp, cur));      // [E, T]
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_2d(c, probs, 1, E * T), ix.wsel);
    w = ggml_reshape_2d(c, w, K, T);
    if (norm_w) w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    if (w_scale != 0.0f && w_scale != 1.0f) w = ggml_scale(c, w, w_scale);
    ggml_tensor * wf = ggml_reshape_2d(c, ggml_cont(c, w), 1, K * T);

    // One gather of every (token, expert) row in expert order, the per-expert matmuls on VIEWS of it, a balanced
    // concat back to [n_embd, K*T], one routing-weight multiply and ONE scatter (get_rows_back sums the K rows of each
    // token). The earlier form scattered every expert into its own [n_embd, T] tensor: O(experts * T * n_embd) memory,
    // 10+ GB at T = 2048 on QFN (512 experts) - this is O(K * T * n_embd * log E).
    if (ix.tok_all && !ix.off.empty()) {
        ggml_tensor * xs = ggml_get_rows(c, cur, ix.tok_all);                                 // [n_embd, K*T]
        std::vector<ggml_tensor *> parts;
        for (int64_t ex = 0; ex < E; ++ex) {
            const int64_t ne = ix.cnt[(size_t) ex];
            if (ne == 0) continue;
            ggml_tensor * gW = ggml_view_3d(c, gate_exps, n_embd, n_ff, 1, gate_exps->nb[1], gate_exps->nb[2], ex * gate_exps->nb[2]);
            ggml_tensor * uW = ggml_view_3d(c, up_exps,   n_embd, n_ff, 1, up_exps->nb[1],   up_exps->nb[2],   ex * up_exps->nb[2]);
            ggml_tensor * dW = ggml_view_3d(c, down_exps, n_ff, n_embd, 1, down_exps->nb[1], down_exps->nb[2], ex * down_exps->nb[2]);
            ggml_tensor * xe = ggml_view_2d(c, xs, n_embd, ne, xs->nb[1], (size_t) ix.off[(size_t) ex] * xs->nb[1]);
            ggml_tensor * a = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, gW, xe)), ggml_mul_mat(c, uW, xe));
            parts.push_back(ggml_mul_mat(c, dW, a));                                          // [n_embd, ne]
        }
        while (parts.size() > 1) {                                                            // balanced concat
            std::vector<ggml_tensor *> nx;
            for (size_t i = 0; i + 1 < parts.size(); i += 2) nx.push_back(ggml_concat(c, parts[i], parts[i + 1], 1));
            if (parts.size() % 2) nx.push_back(parts.back());
            parts.swap(nx);
        }
        ggml_tensor * ys = parts[0];                                                          // [n_embd, K*T]
        ggml_tensor * ws = ggml_reshape_2d(c, ggml_get_rows(c, wf, ix.pair_all), 1, ys->ne[1]);
        ys = ggml_mul(c, ys, ggml_repeat_4d(c, ws, n_embd, ys->ne[1], 1, 1));                // MUL backward: equal shapes
        return ggml_get_rows_back(c, ys, ix.tok_all, cur);                                    // [n_embd, T]
    }
    ggml_tensor * routed = nullptr;
    for (int64_t ex = 0; ex < E; ++ex) {
        if (!ix.tok[(size_t) ex]) continue;
        ggml_tensor * gW = ggml_view_3d(c, gate_exps, n_embd, n_ff, 1, gate_exps->nb[1], gate_exps->nb[2], ex * gate_exps->nb[2]);
        ggml_tensor * uW = ggml_view_3d(c, up_exps,   n_embd, n_ff, 1, up_exps->nb[1],   up_exps->nb[2],   ex * up_exps->nb[2]);
        ggml_tensor * dW = ggml_view_3d(c, down_exps, n_ff, n_embd, 1, down_exps->nb[1], down_exps->nb[2], ex * down_exps->nb[2]);
        ggml_tensor * xe = ggml_get_rows(c, cur, ix.tok[(size_t) ex]);                        // [n_embd, ne]
        ggml_tensor * gt = ggml_mul_mat(c, gW, xe);                                           // [n_ff, ne]
        ggml_tensor * up = ggml_mul_mat(c, uW, xe);                                           // [n_ff, ne]
        ggml_tensor * a = ggml_mul(c, ggml_silu(c, gt), up);                                  // silu(gate) * up
        ggml_tensor * ye = ggml_mul(c, ggml_mul_mat(c, dW, a), ggml_get_rows(c, wf, ix.pair[(size_t) ex]));
        ggml_tensor * sc = ggml_get_rows_back(c, ye, ix.tok[(size_t) ex], cur);               // [n_embd, T]
        routed = routed ? ggml_add(c, routed, sc) : sc;
    }
    return routed;
}

}  // namespace xolotl::train
