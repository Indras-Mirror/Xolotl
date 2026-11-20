// The TOP of the QFN trunk as a differentiable graph: layers [L0, 48) + the head, on the pack's weights AS SERVED
// (dense Q8_0, experts IQ4_XS/IQ4_NL/Q8_0 - nothing dequantized except the few BF16 tensors, which ggml's OUT_PROD
// cannot read and which are small), fed by real captured residuals (STRATA_KVA_CAPTURE L<L0>.bin).
//
// Mirrors src/models/qwen4exp.cpp (non-fused paths) and the numpy reference tools/train/qfn_full_layer_ref.py, which
// matches the served model on real data (21 Nov: layers 44-47 cos >= 0.99993, logits top-1 25/25).
//
//   GDN layer : hc_mix -> qfn_gdn_block -> hc_combine -> hc_mix -> moe + sigmoid-gated shared expert -> hc_combine
//   QSA layer : same, with qfn_qsa_dense_block (exact while T <= indexer top_k: every pool is kept)
//   head      : hc_mix(output_hc_*, no inject) -> output.weight -> logits [n_vocab, T]
//
// Routing is host-side and constant (top-k is zero-grad in the model too): a forward pass gives each layer's router
// input, moe_route picks the experts, and the graph is built with those lists.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "moe_train.hpp"
#include "qfn_layer_train.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace xolotl::train {

// ---- pack loader: selected tensors into one ggml context, keeping their stored types (BF16 -> F32) ----------------
struct PackTensors {
    gguf_context * g = nullptr;
    ggml_context * meta = nullptr;   // shapes/types only
    ggml_context * w = nullptr;      // the loaded weights
    std::map<std::string, ggml_tensor *> t;
    std::string path;

    int64_t kv_i(const char * key) const {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0) { std::fprintf(stderr, "missing kv %s\n", key); std::exit(1); }
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
            case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, id);
            default: std::fprintf(stderr, "kv %s: unexpected type\n", key); std::exit(1);
        }
    }
    float kv_f(const char * key) const {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0) { std::fprintf(stderr, "missing kv %s\n", key); std::exit(1); }
        return gguf_get_val_f32(g, id);
    }

    bool open(const std::string & p) {
        path = p;
        gguf_init_params gp = { true, &meta };
        g = gguf_init_from_file(p.c_str(), gp);
        return g != nullptr;
    }

    // load `names` (tensors that do not exist are skipped silently; callers check what they need)
    bool load(const std::vector<std::string> & names) {
        size_t bytes = 0;
        std::vector<std::string> have;
        for (const auto & n : names) {
            ggml_tensor * m = ggml_get_tensor(meta, n.c_str());
            if (!m) continue;
            have.push_back(n);
            const ggml_type ty = m->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : m->type;
            bytes += ggml_row_size(ty, m->ne[0]) * (size_t) (ggml_nelements(m) / m->ne[0]) + 256;
        }
        ggml_init_params ip = { bytes + have.size() * ggml_tensor_overhead() + (1u << 20), nullptr, false };
        w = ggml_init(ip);
        if (!w) return false;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        const size_t data0 = gguf_get_data_offset(g);
        std::vector<uint16_t> bf;
        for (const auto & n : have) {
            ggml_tensor * m = ggml_get_tensor(meta, n.c_str());
            const bool is_bf = m->type == GGML_TYPE_BF16;
            ggml_tensor * x = ggml_new_tensor(w, is_bf ? GGML_TYPE_F32 : m->type, ggml_n_dims(m), m->ne);
            ggml_set_name(x, n.c_str());
            const int64_t id = gguf_find_tensor(g, n.c_str());
            const size_t off = data0 + gguf_get_tensor_offset(g, id);
            const size_t nb = ggml_nbytes(m);
            if (fseeko(f, (off_t) off, SEEK_SET) != 0) { std::fclose(f); return false; }
            if (is_bf) {
                bf.resize((size_t) ggml_nelements(m));
                if (std::fread(bf.data(), 2, bf.size(), f) != bf.size()) { std::fclose(f); return false; }
                float * d = (float *) x->data;
                for (size_t i = 0; i < bf.size(); ++i) {
                    const uint32_t u = (uint32_t) bf[i] << 16;
                    std::memcpy(&d[i], &u, 4);
                }
            } else if (std::fread(x->data, 1, nb, f) != nb) {
                std::fclose(f);
                return false;
            }
            t[n] = x;
        }
        std::fclose(f);
        return true;
    }
    // the same tensors into a BACKEND buffer (the trainer: weights on the 4090); BF16 still becomes F32
    ggml_backend_buffer_t wbuf = nullptr;
    bool load_backend(const std::vector<std::string> & names, ggml_backend_t be) {
        std::vector<std::string> have;
        for (const auto & n : names) if (ggml_get_tensor(meta, n.c_str())) have.push_back(n);
        ggml_init_params ip = { have.size() * ggml_tensor_overhead() + (1u << 20), nullptr, true };
        w = ggml_init(ip);
        for (const auto & n : have) {
            ggml_tensor * m = ggml_get_tensor(meta, n.c_str());
            ggml_tensor * x = ggml_new_tensor(w, m->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : m->type, ggml_n_dims(m), m->ne);
            ggml_set_name(x, n.c_str());
            t[n] = x;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(w, be);
        if (!wbuf) return false;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        const size_t data0 = gguf_get_data_offset(g);
        std::vector<uint8_t> stage;
        std::vector<float> conv;
        for (const auto & n : have) {
            ggml_tensor * m = ggml_get_tensor(meta, n.c_str());
            const size_t nb = ggml_nbytes(m);
            stage.resize(nb);
            if (fseeko(f, (off_t) (data0 + gguf_get_tensor_offset(g, gguf_find_tensor(g, n.c_str()))), SEEK_SET) != 0 ||
                std::fread(stage.data(), 1, nb, f) != nb) { std::fclose(f); return false; }
            if (m->type == GGML_TYPE_BF16) {
                conv.resize((size_t) ggml_nelements(m));
                const uint16_t * b = (const uint16_t *) stage.data();
                for (size_t i = 0; i < conv.size(); ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&conv[i], &u, 4); }
                ggml_backend_tensor_set(t[n], conv.data(), 0, conv.size() * 4);
            } else {
                ggml_backend_tensor_set(t[n], stage.data(), 0, nb);
            }
        }
        std::fclose(f);
        return true;
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "tensor %s not loaded\n", n.c_str()); std::exit(1); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

struct QfnGeom {
    int64_t D, hc, hd, nk, nv, d_conv, E, K, NL, fai, n_head, n_kv, ahd, n_rot, top_k, n_vocab;
    float eps, rope_base;
    // the SERVED rope: xol-qfn runs --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 (ext_factor resolves to
    // 1 for yarn, generate.cpp:4343). Strata's YaRN (kernels/rope_scaling.hpp) is ggml's rope_yarn exactly, so these
    // go straight into ggml_rope_ext. QFN_ROPE_SCALE=2 turns it on (0/unset = plain rope, the smoke captures).
    float rope_freq_scale = 1.0f, rope_ext = 0.0f;
    int rope_orig_ctx = 0;
    bool is_qsa(int64_t l) const { return (l + 1) % fai == 0; }
};

inline QfnGeom qfn_geom(const PackTensors & p) {
    QfnGeom q{};
    q.D = p.kv_i("qwen4exp.embedding_length");
    q.hc = p.kv_i("qwen4exp.hyper_connection.count");
    q.hd = p.kv_i("qwen4exp.ssm.state_size");
    q.nk = p.kv_i("qwen4exp.ssm.group_count");
    q.nv = p.kv_i("qwen4exp.ssm.time_step_rank");
    q.d_conv = p.kv_i("qwen4exp.ssm.conv_kernel");
    q.E = p.kv_i("qwen4exp.expert_count");
    q.K = p.kv_i("qwen4exp.expert_used_count");
    q.NL = p.kv_i("qwen4exp.block_count");
    q.fai = p.kv_i("qwen4exp.full_attention_interval");
    q.n_head = p.kv_i("qwen4exp.attention.head_count");
    q.n_kv = p.kv_i("qwen4exp.attention.head_count_kv");
    q.ahd = p.kv_i("qwen4exp.attention.key_length");
    q.n_rot = p.kv_i("qwen4exp.rope.dimension_count");
    q.top_k = p.kv_i("qwen4exp.attention.indexer.top_k");
    q.eps = p.kv_f("qwen4exp.attention.layer_norm_rms_epsilon");
    q.rope_base = p.kv_f("qwen4exp.rope.freq_base");
    ggml_tensor * o = ggml_get_tensor(p.meta, "output.weight");
    q.n_vocab = o ? o->ne[1] : 0;
    if (const char * rs = std::getenv("QFN_ROPE_SCALE"); rs && std::atof(rs) > 1.0) {
        q.rope_freq_scale = 1.0f / (float) std::atof(rs);
        q.rope_ext = 1.0f;
        q.rope_orig_ctx = 262144;
    }
    return q;
}

inline std::vector<std::string> qfn_layer_names(int64_t l) {
    const std::string b = "blk." + std::to_string((long long) l) + ".";
    std::vector<std::string> v;
    for (const char * s : { "attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_beta.weight",
                            "ssm_alpha.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight",
                            "attn_q.weight", "attn_q_norm.weight", "attn_k.weight", "attn_k_norm.weight",
                            "attn_v.weight", "attn_output.weight",
                            "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                            "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                            "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                            "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                            "ffn_down_shexp.weight" })
        v.push_back(b + s);
    return v;
}

// ---- a LoRA pair on one base tensor -----------------------------------------------------------------------------
struct Lora {
    ggml_tensor * a = nullptr;   // [n_in, r]
    ggml_tensor * b = nullptr;   // [r, n_out]
};

inline ggml_tensor * lin(ggml_context * c, ggml_tensor * W, ggml_tensor * x, const Lora * l, float scale) {
    ggml_tensor * y = ggml_mul_mat(c, W, x);
    if (l && l->a && l->b) y = ggml_add(c, y, ggml_scale(c, ggml_mul_mat(c, l->b, ggml_mul_mat(c, l->a, x)), scale));
    return y;
}

// ---- the full-attention block, dense path -------------------------------------------------------------------------
// x [D, T] -> [D, T].  Per head the Q projection holds [q (ahd) | gate (ahd)]; q/k RMS-normed with their weights;
// IMRoPE == NEOX on the first n_rot dims for text (every section carries the same position); GQA maps q head h to kv
// head h / (n_head / n_kv) (contiguous groups - NOT the GDN tiling); scale 1/sqrt(ahd); the attention output is gated
// by sigmoid(gate) before attn_output.
struct QsaLoras { Lora q, k, v, o; };

// What the attention layer carries across a split: the prefix's keys (normed + rotated) and values, [hd, n_kv, T1].
// Inputs are constants; outputs are filled when the block runs the prefix. The window's `pos` must continue the
// prefix's positions (T1, T1+1, ...).
struct QsaCarry {
    ggml_tensor * k_in = nullptr, * v_in = nullptr;
    ggml_tensor * k_out = nullptr, * v_out = nullptr;
    // the engine's QSA selection replayed (STRATA_QSA_DUMP): additive mask [n_past + T, T], 0 on each query's selected
    // cells and -inf elsewhere (constant: the selection is zero-grad in the model too). nullptr = dense causal.
    ggml_tensor * sel_mask = nullptr;
};

// q (normed, rotated), gate, k (normed, rotated), v of the full-attention layer: x [D, T] -> q,gate [hd, H, T], k,v [hd, HK, T]
inline void qfn_qsa_qkv(ggml_context * c, ggml_tensor * x, const PackTensors & p, int64_t l, const QfnGeom & G,
                        ggml_tensor * pos, const QsaLoras * lo, float ls, ggml_tensor ** q_out, ggml_tensor ** gate_out,
                        ggml_tensor ** k_out, ggml_tensor ** v_out) {
    const std::string b = "blk." + std::to_string((long long) l) + ".";
    const int64_t T = x->ne[1], H = G.n_head, HK = G.n_kv, hd = G.ahd;
    const size_t es = sizeof(float);
    if (q_out || gate_out) {
        ggml_tensor * qf = lin(c, p.get(b + "attn_q.weight"), x, lo ? &lo->q : nullptr, ls);     // [2*hd*H, T]
        ggml_tensor * q = ggml_view_3d(c, qf, hd, H, T, 2 * hd * es, 2 * hd * H * es, 0);
        if (gate_out) *gate_out = ggml_view_3d(c, qf, hd, H, T, 2 * hd * es, 2 * hd * H * es, hd * es);
        q = ggml_mul(c, ggml_rms_norm(c, ggml_cont(c, q), G.eps), ggml_repeat_4d(c, p.get(b + "attn_q_norm.weight"), hd, H, T, 1));
        if (q_out) *q_out = ggml_rope_ext(c, q, pos, nullptr, (int) G.n_rot, GGML_ROPE_TYPE_NEOX, G.rope_orig_ctx, G.rope_base,
                                          G.rope_freq_scale, G.rope_ext, 1.0f, 32.0f, 1.0f);
    }
    ggml_tensor * k = ggml_reshape_3d(c, lin(c, p.get(b + "attn_k.weight"), x, lo ? &lo->k : nullptr, ls), hd, HK, T);
    k = ggml_mul(c, ggml_rms_norm(c, k, G.eps), ggml_repeat_4d(c, p.get(b + "attn_k_norm.weight"), hd, HK, T, 1));
    *k_out = ggml_rope_ext(c, k, pos, nullptr, (int) G.n_rot, GGML_ROPE_TYPE_NEOX, G.rope_orig_ctx, G.rope_base,
                           G.rope_freq_scale, G.rope_ext, 1.0f, 32.0f, 1.0f);
    *v_out = ggml_reshape_3d(c, lin(c, p.get(b + "attn_v.weight"), x, lo ? &lo->v : nullptr, ls), hd, HK, T);
}

inline ggml_tensor * qfn_qsa_dense_block(ggml_context * c, ggml_tensor * x, const PackTensors & p, int64_t l,
                                         const QfnGeom & G, ggml_tensor * pos, const QsaLoras * lo, float ls,
                                         QsaCarry * carry = nullptr) {
    const std::string b = "blk." + std::to_string((long long) l) + ".";
    const int64_t T = x->ne[1], H = G.n_head, HK = G.n_kv, hd = G.ahd;
    ggml_tensor *q, *gate, *k, *v;
    qfn_qsa_qkv(c, x, p, l, G, pos, lo, ls, &q, &gate, &k, &v);
    if (carry) { carry->k_out = k; carry->v_out = v; }
    int64_t n_past = 0;
    if (carry && carry->k_in) {                                            // prefix cells first, then the window's
        n_past = carry->k_in->ne[2];
        k = ggml_concat(c, carry->k_in, k, 2);
        v = ggml_concat(c, carry->v_in, v, 2);
    }
    const int64_t TK = T + n_past;
    // kv heads expanded to contiguous groups: [hd, 1, HK, TK] -> [hd, H/HK, HK, TK] -> [hd, H, TK]
    auto expand = [&](ggml_tensor * t) {
        t = ggml_reshape_4d(c, ggml_cont(c, t), hd, 1, HK, TK);
        t = ggml_repeat_4d(c, t, hd, H / HK, HK, TK);
        return ggml_reshape_3d(c, t, hd, H, TK);
    };
    k = expand(k);
    v = expand(v);

    ggml_tensor * qp = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));      // [hd, T, H]
    ggml_tensor * kp = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));      // [hd, TK, H]
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));      // [TK, hd, H]
    ggml_tensor * kq = ggml_mul_mat(c, kp, qp);                            // [TK, T_q, H]
    if (carry && carry->sel_mask)
        kq = ggml_soft_max_ext(c, kq, carry->sel_mask, 1.0f / std::sqrt((float) hd), 0.0f);   // the replayed selection
    else
        kq = ggml_soft_max(c, ggml_diag_mask_inf(c, ggml_scale(c, kq, 1.0f / std::sqrt((float) hd)), (int) n_past));
    ggml_tensor * o = ggml_mul_mat(c, vt, kq);                             // [hd, T_q, H]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                      // [hd, H, T]
    o = ggml_mul(c, o, ggml_sigmoid(c, ggml_cont(c, gate)));
    return lin(c, p.get(b + "attn_output.weight"), ggml_reshape_2d(c, o, hd * H, T), lo ? &lo->o : nullptr, ls);
}

// ---- per-layer weights wired into the existing GDN/HC structs -----------------------------------------------------
struct TopLayer {
    int64_t l = 0;
    bool qsa = false;
    QfnLayer gdn;          // GDN weights (+ its 3 LoRA slots)
    QsaLoras qsa_lora;     // QSA LoRA slots
    QfnHc hc_attn, hc_ffn;
    GdnCarry gdn_carry;    // used when `carry` is set on the half below
    QsaCarry qsa_carry;
};

inline TopLayer qfn_top_layer(ggml_context * c, const PackTensors & p, int64_t l, const QfnGeom & G, ggml_tensor * ident) {
    const std::string b = "blk." + std::to_string((long long) l) + ".";
    TopLayer L;
    L.l = l;
    L.qsa = G.is_qsa(l);
    auto hc = [&](const char * pre) {
        QfnHc h;
        h.norm = ggml_reshape_2d(c, p.get(b + pre + "_norm.weight"), G.D, G.hc);
        h.down = p.get(b + pre + "_down.weight");
        h.up = p.get(b + pre + "_up.weight");
        h.inject = p.get(b + pre + "_inject.weight");
        return h;
    };
    L.hc_attn = hc("hc_attn");
    L.hc_ffn = hc("hc_ffn");
    if (!L.qsa) {
        QfnLayer & w = L.gdn;
        w.wqkv = p.get(b + "attn_qkv.weight");
        w.wgate = p.get(b + "attn_gate.weight");
        w.conv_w = p.get(b + "ssm_conv1d.weight");
        // w_ba = [beta | alpha] rows (qwen4exp.cpp:1133 order); both F32 and frozen, so a concat is exact
        w.w_ba = ggml_concat(c, p.get(b + "ssm_beta.weight"), p.get(b + "ssm_alpha.weight"), 1);
        w.ssm_a = p.get(b + "ssm_a");
        w.dt = p.get(b + "ssm_dt.bias");
        w.norm_w = p.get(b + "ssm_norm.weight");
        w.w_out = p.get(b + "ssm_out.weight");
        w.ident = ident;
    }
    return L;
}

// mixed [D, T] -> routed MoE + sigmoid-gated shared expert, [D, T]
inline ggml_tensor * qfn_ffn(ggml_context * c, ggml_tensor * x, const PackTensors & p, int64_t l, const MoeIndex & ix) {
    const std::string b = "blk." + std::to_string((long long) l) + ".";
    ggml_tensor * moe = moe_ffn(c, x, p.get(b + "ffn_gate_inp.weight"), p.get(b + "ffn_up_exps.weight"),
                                p.get(b + "ffn_gate_exps.weight"), p.get(b + "ffn_down_exps.weight"), ix, true, 0.0f);
    ggml_tensor * sh = ggml_mul_mat(c, p.get(b + "ffn_down_shexp.weight"),
                                    ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, p.get(b + "ffn_gate_shexp.weight"), x)),
                                             ggml_mul_mat(c, p.get(b + "ffn_up_shexp.weight"), x)));
    ggml_tensor * gin = p.get(b + "ffn_gate_inp_shexp.weight");                          // [D]
    ggml_tensor * sg = ggml_sigmoid(c, ggml_mul_mat(c, ggml_reshape_2d(c, gin, gin->ne[0], 1), x));   // [1, T]
    sh = ggml_mul(c, sh, ggml_repeat_4d(c, sg, sh->ne[0], sh->ne[1], 1, 1));
    return moe ? ggml_add(c, moe, sh) : sh;
}

// The attention half of a layer: res [D, hc, T] -> res after hc_combine, and the ffn input (mixed2) + its inject.
inline ggml_tensor * qfn_attn_half(ggml_context * c, ggml_tensor * res, TopLayer & L, const PackTensors & p,
                                   const QfnGeom & G, ggml_tensor * pos, float ls, ggml_tensor ** mixed2,
                                   ggml_tensor ** inj2, bool carry = false) {
    ggml_tensor * inj = nullptr;
    ggml_tensor * mixed = qfn_hc_mix(c, res, L.hc_attn, G.hc, G.eps, &inj);
    ggml_tensor * att = L.qsa ? qfn_qsa_dense_block(c, mixed, p, L.l, G, pos, &L.qsa_lora, ls,
                                                    carry ? &L.qsa_carry : nullptr)
                              : qfn_gdn_block(c, mixed, L.gdn, G.hd, G.nk, G.nv, G.d_conv, G.eps,
                                              carry ? &L.gdn_carry : nullptr);
    ggml_tensor * res1 = qfn_hc_combine(c, res, att, inj, G.hc);
    *mixed2 = qfn_hc_mix(c, res1, L.hc_ffn, G.hc, G.eps, inj2);
    return res1;
}

inline ggml_tensor * qfn_head(ggml_context * c, ggml_tensor * res, const PackTensors & p, const QfnGeom & G) {
    QfnHc h;
    h.norm = ggml_reshape_2d(c, p.get("output_hc_norm.weight"), G.D, G.hc);
    h.down = p.get("output_hc_down.weight");
    h.up = p.get("output_hc_up.weight");
    ggml_tensor * mixed = qfn_hc_mix(c, res, h, G.hc, G.eps, nullptr);
    return ggml_mul_mat(c, p.get("output.weight"), mixed);                                 // [n_vocab, T]
}

}  // namespace xolotl::train
