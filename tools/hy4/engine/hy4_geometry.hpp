// tools/hy4/engine/hy4_geometry.hpp - Hy4-flash (`hy_v4_flash`) geometry and tensor map.
//
// The GGUF is the llama.cpp PR #32517 convention (the converter Tencent's AngelSlim/Hy4-flash-GGUF was made with; our
// own Q8_0 comes from it plus the public-release name patch, see docs/hy4/PORT_PLAN.md).  Every number comes from the
// file's `hy_v4_flash.*` keys; a missing key is an error, never a default.  `hy4_check_tensors` resolves the full
// tensor map both ways: every predicted name must be present and every name in the file must be predicted.
//
// Layer kinds: each trunk layer is either a sliding-window layer (swa_heads heads, window n_swa) or a sparse (DSA)
// layer (n_head heads) that owns a k-pool indexer.  Both share one MLA geometry: q_lora -> n_head x (nope + rope),
// kv_lora + rope latent (K row = kv_lora + rope, V row = kv_lora), wk_b / wv_b per head.  Layers < n_dense_lead run a
// dense FFN, the rest the latent MoE (router at n_embd, routed_down n_embd -> latent, experts in latent, routed_up).
// Engram modules sit on their own layers and write the hc streams before the attention pre-mix.
//
// Header-only; reads the header region only, no tensor data.
#pragma once

#include "ds41_geometry.hpp"   // detail::u64 / f64 / arr, ds41_read_int_array (shared GGUF key readers)

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::hy4 {

using strata::GgufModel;
using strata::GgufFile;

struct Hy4Geometry {
    // trunk
    int64_t n_embd = 0, n_layer = 0, n_layer_all = 0, n_nextn = 0, vocab = 0, ctx_len = 0;
    double rms_eps = 0;
    // attention (MLA, absorbed)
    std::vector<int64_t> n_head;          ///< [n_layer_all] heads per layer (sliding layers carry more)
    std::vector<bool> is_swa;             ///< [n_layer_all]
    int64_t n_swa = 0;                    ///< sliding window (tokens)
    int64_t q_lora = 0, kv_lora = 0;
    int64_t d_qk = 0, d_v = 0, d_rope = 0; ///< per-head key (nope + rope) / value length, rope part
    int64_t d_nope() const { return d_qk - d_rope; }
    double rope_base = 0, rope_base_swa = 0;
    // indexer (sparse layers)
    std::vector<bool> has_indexer;        ///< [n_layer] (indexer.types)
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0, idx_kpool = 0;
    bool idx_select_tail = false;
    // FFN / MoE
    int64_t n_dense_lead = 0, n_ff = 0;   ///< dense lead layers and their FFN width
    int64_t n_expert = 0, n_used = 0, n_ff_exp = 0, n_shared = 0, n_latent = 0;
    double weights_scale = 0;
    bool weights_norm = false;
    int64_t gating_func = 0;              ///< llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE (2 = sigmoid)
    std::vector<double> swiglu_clamp;     ///< [n_layer_all] routed experts only
    // hyper-connections
    int64_t hc = 0;
    double hc_eps = 0, hc_magnitude = 0;
    // Engram
    std::vector<int64_t> engram_layers;   ///< 0-based trunk layers, ascending
    int64_t engram_ngram = 0, engram_heads_per_ngram = 0, engram_dim = 0, engram_pad = 0;
    std::vector<int64_t> engram_mult;     ///< [n_engram * ngram]
    std::vector<int64_t> engram_offsets;  ///< [n_engram * (ngram-1) * heads_per_ngram] row offset in the shared table
    std::vector<int64_t> engram_buckets;  ///< same shape: bucket count (prime)
    std::vector<int64_t> engram_token_map;///< [vocab] token -> compressed id
    int64_t engram_cols() const { return (engram_ngram - 1) * engram_heads_per_ngram; }
    int64_t engram_rows() const {
        int64_t r = 0;
        for (size_t i = 0; i < engram_offsets.size(); ++i) r = std::max(r, engram_offsets[i] + engram_buckets[i]);
        return r;
    }
    int64_t engram_slot(int64_t l) const {
        for (size_t i = 0; i < engram_layers.size(); ++i) if (engram_layers[i] == l) return (int64_t) i;
        return -1;
    }
    int64_t bos = -1, eos = -1;

    int64_t heads(int64_t l) const { return n_head[(size_t) l]; }
    bool swa(int64_t l) const { return is_swa[(size_t) l]; }
    bool indexed(int64_t l) const { return l < n_layer && has_indexer[(size_t) l]; }
    bool moe(int64_t l) const { return l >= n_dense_lead; }
};

namespace detail {
using strata::v41::detail::u64;
using strata::v41::detail::f64;
using strata::v41::detail::arr;

inline std::string bools(const GgufFile & g, const std::string & k, int64_t n, std::vector<bool> & out) {
    std::vector<int64_t> v;
    std::string e = arr(g, k, v);
    if (!e.empty()) return e;
    if ((int64_t) v.size() != n) return k + ": " + std::to_string(v.size()) + " entries, expected " + std::to_string(n);
    out.assign(v.begin(), v.end());
    return {};
}
}  // namespace detail

/// Reads the geometry from a `hy_v4_flash` GGUF (`path` = the metadata shard, for the long token-map array).
inline std::string hy4_read_geometry(const GgufModel & model, const std::string & path, Hy4Geometry & G) {
    using namespace detail;
    const GgufFile & g = model.meta();
    const MetaValue * arch = g.get("general.architecture");
    if (!arch || arch->s != "hy_v4_flash") return "general.architecture is not hy_v4_flash";
    const std::string P = "hy_v4_flash.";
    std::string e;
#define HY4_REQ(x) do { e = (x); if (!e.empty()) return e; } while (0)
    int64_t blocks = 0;
    HY4_REQ(u64(g, P + "block_count", blocks));
    if (const MetaValue * v = g.get(P + "nextn_predict_layers")) G.n_nextn = (int64_t) v->u;
    G.n_layer_all = blocks;
    G.n_layer = blocks - G.n_nextn;
    HY4_REQ(u64(g, P + "embedding_length", G.n_embd));
    HY4_REQ(u64(g, P + "vocab_size", G.vocab));
    HY4_REQ(u64(g, P + "context_length", G.ctx_len));
    HY4_REQ(f64(g, P + "attention.layer_norm_rms_epsilon", G.rms_eps));

    HY4_REQ(arr(g, P + "attention.head_count", G.n_head));
    if ((int64_t) G.n_head.size() != G.n_layer_all) return "attention.head_count: one entry per block expected";
    HY4_REQ(bools(g, P + "attention.sliding_window_pattern", G.n_layer_all, G.is_swa));
    HY4_REQ(u64(g, P + "attention.sliding_window", G.n_swa));
    int64_t kv_heads = 0;
    HY4_REQ(u64(g, P + "attention.head_count_kv", kv_heads));
    if (kv_heads != 1) return "attention.head_count_kv must be 1 (MLA runs as MQA)";
    HY4_REQ(u64(g, P + "attention.q_lora_rank", G.q_lora));
    HY4_REQ(u64(g, P + "attention.kv_lora_rank", G.kv_lora));
    HY4_REQ(u64(g, P + "attention.key_length_mla", G.d_qk));
    HY4_REQ(u64(g, P + "attention.value_length_mla", G.d_v));
    HY4_REQ(u64(g, P + "rope.dimension_count", G.d_rope));
    HY4_REQ(f64(g, P + "rope.freq_base", G.rope_base));
    HY4_REQ(f64(g, P + "rope.freq_base_swa", G.rope_base_swa));
    int64_t k_len = 0, v_len = 0;
    HY4_REQ(u64(g, P + "attention.key_length", k_len));
    HY4_REQ(u64(g, P + "attention.value_length", v_len));
    if (k_len != G.kv_lora + G.d_rope || v_len != G.kv_lora)
        return "attention.key_length/value_length disagree with kv_lora_rank + rope (absorbed MLA)";
    if (G.d_rope <= 0 || G.d_rope >= G.d_qk) return "rope.dimension_count must be inside key_length_mla";

    HY4_REQ(u64(g, P + "attention.indexer.top_k", G.idx_topk));
    HY4_REQ(u64(g, P + "attention.indexer.head_count", G.idx_heads));
    HY4_REQ(u64(g, P + "attention.indexer.key_length", G.idx_dim));
    HY4_REQ(u64(g, P + "attention.indexer.kpool", G.idx_kpool));
    if (const MetaValue * v = g.get(P + "attention.indexer.kpool_select_tail")) G.idx_select_tail = v->u != 0;
    HY4_REQ(bools(g, P + "attention.indexer.types", G.n_layer, G.has_indexer));
    if (G.idx_kpool < 2 || G.idx_topk % G.idx_kpool) return "indexer top_k must be a multiple of kpool >= 2";
    for (int64_t l = 0; l < G.n_layer; ++l)
        if (G.has_indexer[(size_t) l] && G.is_swa[(size_t) l])
            return "layer " + std::to_string(l) + " is sliding-window and cannot own an indexer";

    HY4_REQ(u64(g, P + "leading_dense_block_count", G.n_dense_lead));
    HY4_REQ(u64(g, P + "feed_forward_length", G.n_ff));
    HY4_REQ(u64(g, P + "expert_count", G.n_expert));
    HY4_REQ(u64(g, P + "expert_used_count", G.n_used));
    HY4_REQ(u64(g, P + "expert_feed_forward_length", G.n_ff_exp));
    HY4_REQ(u64(g, P + "expert_shared_count", G.n_shared));
    HY4_REQ(u64(g, P + "expert_latent_length", G.n_latent));
    HY4_REQ(f64(g, P + "expert_weights_scale", G.weights_scale));
    HY4_REQ(u64(g, P + "expert_gating_func", G.gating_func));
    {
        const MetaValue * v = g.get(P + "expert_weights_norm");
        if (!v) return "missing " + P + "expert_weights_norm";
        G.weights_norm = v->u != 0;
    }
    {
        const MetaValue * v = g.get(P + "swiglu_clamp_exp");
        if (!v || v->type != MetaType::ARRAY || (int64_t) v->count != G.n_layer_all || v->items.size() != v->count)
            return P + "swiglu_clamp_exp: one entry per block expected";
        for (const auto & it : v->items) G.swiglu_clamp.push_back(it.num());
    }

    HY4_REQ(u64(g, P + "hyper_connection.count", G.hc));
    HY4_REQ(f64(g, P + "hyper_connection.epsilon", G.hc_eps));
    HY4_REQ(f64(g, P + "hyper_connection.magnitude", G.hc_magnitude));

    if (g.get(P + "ple.layers")) {
        HY4_REQ(arr(g, P + "ple.layers", G.engram_layers));
        HY4_REQ(u64(g, P + "ple.ngram_size", G.engram_ngram));
        HY4_REQ(u64(g, P + "ple.heads_per_ngram", G.engram_heads_per_ngram));
        HY4_REQ(u64(g, P + "embedding_length_per_layer_input", G.engram_dim));
        HY4_REQ(u64(g, P + "engram.pad_token_id", G.engram_pad));
        HY4_REQ(arr(g, P + "ple.layer_multipliers", G.engram_mult));
        HY4_REQ(strata::v41::ds41_read_int_array(path, P + "ple.head_offsets", G.engram_offsets));
        HY4_REQ(strata::v41::ds41_read_int_array(path, P + "ple.head_vocab_sizes", G.engram_buckets));
        HY4_REQ(strata::v41::ds41_read_int_array(path, P + "engram.token_map", G.engram_token_map));
        const size_t n_e = G.engram_layers.size();
        const size_t cols = (size_t) G.engram_cols();
        if (G.engram_ngram < 2 || cols == 0 || G.engram_mult.size() != n_e * (size_t) G.engram_ngram ||
            G.engram_offsets.size() != n_e * cols || G.engram_buckets.size() != n_e * cols ||
            (int64_t) G.engram_token_map.size() != G.vocab || G.engram_pad >= G.vocab)
            return "Engram metadata is inconsistent";
        for (size_t i = 0; i < n_e; ++i)
            if (G.engram_layers[i] < 0 || G.engram_layers[i] >= G.n_layer || (i && G.engram_layers[i] <= G.engram_layers[i - 1]))
                return "ple.layers out of range or order";
        for (size_t c = 0; c < G.engram_buckets.size(); ++c)
            if (G.engram_buckets[c] < 2 || G.engram_offsets[c] < 0 || G.engram_offsets[c] + G.engram_buckets[c] > INT32_MAX)
                return "Engram bucket range out of range";
    }
    if (const MetaValue * v = g.get("tokenizer.ggml.bos_token_id")) G.bos = (int64_t) v->u;
    if (const MetaValue * v = g.get("tokenizer.ggml.eos_token_id")) G.eos = (int64_t) v->u;
#undef HY4_REQ
    return {};
}

struct Hy4TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;   ///< GGUF order (dim 0 fastest)
};

/// Every tensor the trunk needs, with its expected shape (the nextn block's tensors are listed separately by prefix).
inline std::vector<Hy4TensorSpec> hy4_expected_tensors(const Hy4Geometry & G) {
    std::vector<Hy4TensorSpec> out;
    const uint64_t E = (uint64_t) G.n_embd, H = (uint64_t) G.hc;
    auto add = [&](std::string n, std::vector<uint64_t> s) { out.push_back({ std::move(n), std::move(s) }); };
    add("token_embd.weight", { E, (uint64_t) G.vocab });
    add("output.weight", { E, (uint64_t) G.vocab });
    add("output_norm.weight", { E });
    add("output_hc_fn.weight", { H * E, H });
    add("output_hc_base.weight", { H });
    add("output_hc_scale.weight", { 1 });
    if (!G.engram_layers.empty()) add("per_layer_token_embd.weight", { (uint64_t) G.engram_dim, (uint64_t) G.engram_rows() });
    const uint64_t ql = (uint64_t) G.q_lora, kvl = (uint64_t) G.kv_lora, rope = (uint64_t) G.d_rope;
    const uint64_t dqk = (uint64_t) G.d_qk, dv = (uint64_t) G.d_v, nope = (uint64_t) G.d_nope();
    const uint64_t lat = (uint64_t) G.n_latent, ffx = (uint64_t) G.n_ff_exp, nx = (uint64_t) G.n_expert;
    for (int64_t l = 0; l < G.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const uint64_t nh = (uint64_t) G.heads(l);
        add(b + "attn_norm.weight", { E });
        add(b + "attn_sinks.weight", { nh });
        add(b + "attn_q_a.weight", { E, ql });
        add(b + "attn_q_a_norm.weight", { ql });
        add(b + "attn_q_b.weight", { ql, nh * dqk });
        add(b + "attn_kv_a_mqa.weight", { E, kvl + rope });
        add(b + "attn_kv_a_norm.weight", { kvl });
        add(b + "attn_k_b.weight", { nope, kvl, nh });
        add(b + "attn_v_b.weight", { kvl, dv, nh });
        add(b + "attn_output.weight", { nh * dv, E });
        add(b + "attn_gate.weight", { E, nh * dv });
        add(b + "post_attention_norm.weight", { E });
        add(b + "hc_attn_fn.weight", { H * E, 2 * H });
        add(b + "hc_attn_base.weight", { 2 * H });
        add(b + "hc_attn_scale.weight", { 2 });
        add(b + "hc_ffn_fn.weight", { H * E, 2 * H });
        add(b + "hc_ffn_base.weight", { 2 * H });
        add(b + "hc_ffn_scale.weight", { 2 });
        add(b + "ffn_norm.weight", { E });
        add(b + "post_ffw_norm.weight", { E });
        if (G.indexed(l)) {
            const uint64_t d = (uint64_t) G.idx_dim, ih = (uint64_t) G.idx_heads;
            add(b + "indexer.attn_k.weight", { E, d });
            add(b + "indexer.k_norm.weight", { d });
            add(b + "indexer_compressor_value.weight", { d, d });
            add(b + "indexer_compressor_gate.weight", { d, d });
            add(b + "indexer_compressor_ape.weight", { d, (uint64_t) G.idx_kpool });
            add(b + "indexer_compressor_norm.weight", { d });
            add(b + "indexer.attn_q_b.weight", { ql, d * ih });
            add(b + "indexer.proj.weight", { E, ih });
        }
        if (G.engram_slot(l) >= 0) {
            const uint64_t cols = (uint64_t) G.engram_cols(), ed = (uint64_t) G.engram_dim;
            add(b + "engram_kv.weight", { cols * ed, E * (H + 1) });
            add(b + "engram_kv_b.weight", { E, H + 1 });
            add(b + "engram_gate.weight", { E, H });
            add(b + "engram_lookup_norm.weight", { ed, cols });
            add(b + "engram_pre_gamma.weight", { E });
            add(b + "engram_hc_scale.weight", { E, H });
        }
        if (!G.moe(l)) {
            add(b + "ffn_gate.weight", { E, (uint64_t) G.n_ff });
            add(b + "ffn_up.weight", { E, (uint64_t) G.n_ff });
            add(b + "ffn_down.weight", { (uint64_t) G.n_ff, E });
        } else {
            add(b + "ffn_gate_inp.weight", { E, nx });
            add(b + "exp_probs_b.bias", { nx });
            add(b + "ffn_routed_down.weight", { E, lat });
            add(b + "ffn_routed_up.weight", { lat, E });
            add(b + "ffn_gate_exps.weight", { lat, ffx, nx });
            add(b + "ffn_up_exps.weight", { lat, ffx, nx });
            add(b + "ffn_down_exps.weight", { ffx, lat, nx });
            const uint64_t sh = ffx * (uint64_t) G.n_shared;
            add(b + "ffn_gate_shexp.weight", { E, sh });
            add(b + "ffn_up_shexp.weight", { E, sh });
            add(b + "ffn_down_shexp.weight", { sh, E });
        }
    }
    return out;
}

/// Checks the file's tensors against the prediction: every predicted name present with its shape, and every tensor in
/// the file predicted (the nextn block's `blk.<n_layer..>.*` tensors are accepted when nextn_predict_layers > 0).
inline std::string hy4_check_tensors(const GgufModel & model, const Hy4Geometry & G) {
    const std::vector<Hy4TensorSpec> want = hy4_expected_tensors(G);
    std::set<std::string> predicted;
    for (const auto & t : want) {
        predicted.insert(t.name);
        const strata::TensorInfo * ti = model.find(t.name);
        if (!ti) return "missing tensor " + t.name;
        if (ti->shape != t.shape) {
            std::string got, exp;
            for (auto d : ti->shape) got += std::to_string(d) + ",";
            for (auto d : t.shape) exp += std::to_string(d) + ",";
            return t.name + ": shape {" + got + "} expected {" + exp + "}";
        }
    }
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & t : model.shard(s).tensors()) {
            if (predicted.count(t.name)) continue;
            bool nextn = false;
            for (int64_t l = G.n_layer; l < G.n_layer_all && !nextn; ++l)
                nextn = t.name.rfind("blk." + std::to_string(l) + ".", 0) == 0;
            if (!nextn) return "unexpected tensor " + t.name;
        }
    return {};
}

}  // namespace strata::hy4
