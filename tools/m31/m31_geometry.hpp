// MiniMax-M3.1 GGUF contract. Header/directory inspection only: never scans weight payloads.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31 {

struct Geometry {
    int64_t layers = 0, embd = 0, heads = 0, kv_heads = 0, head_dim = 0, rope_dim = 0;
    int64_t dense_ff = 0, expert_ff = 0, dense_lead = 0, experts = 0, used = 0, shared = 0;
    int64_t index_heads = 0, index_dim = 0, block = 0, top_blocks = 0, local_blocks = 0;
    int64_t vocab = 0, context = 0;
    float rms_eps = 0, rope_base = 0, route_scale = 0;
    bool norm_routes = false, nfc = false;
    uint64_t expert_bytes = 0, other_bytes = 0, embedding_bytes = 0, expert_scale_bytes = 0;
    std::vector<int> gate_types, up_types, down_types;
    bool routed(int64_t l) const { return l >= dense_lead && l < layers; }
    uint64_t kv_bytes(int64_t tokens) const {
        // M1: F16 main K/V plus F16 index K, including the leading dense FFN layers.
        return uint64_t(layers) * uint64_t(tokens) *
            uint64_t(2 * kv_heads * head_dim + index_dim) * 2;
    }
};

inline bool ends_with(const std::string & s, const std::string & tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}
inline bool expert_tensor(const std::string & name) {
    return ends_with(name, "_exps.weight") || ends_with(name, "_exps.scale");
}
inline std::string layer_name(int64_t l, const std::string & suffix) {
    return "blk." + std::to_string(l) + "." + suffix;
}

inline Geometry read_geometry(const GgufModel & model) {
    const GgufFile & f = model.meta();
    const MetaValue * arch = f.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING || arch->s != "minimax-m3")
        throw std::runtime_error("M31: general.architecture must be minimax-m3");
    const std::string p = "minimax-m3.";
    auto number = [&](const std::string & key) {
        const MetaValue * v = f.get(p + key);
        if (!v || !v->is_num() || !std::isfinite(v->num()))
            throw std::runtime_error("M31: missing/invalid metadata " + p + key);
        return v->num();
    };
    auto integer = [&](const std::string & key, bool zero = false) -> int64_t {
        const double x = number(key);
        // Tight cap also prevents overflow in subsequent dimension products for malformed files.
        if (std::floor(x) != x || x < (zero ? 0 : 1) || x > 16777216)
            throw std::runtime_error("M31: out-of-range integer " + p + key);
        return int64_t(x);
    };
    auto boolean = [&](const std::string & key, bool required) {
        const MetaValue * v = f.get(key);
        if (!v && !required) return false;
        if (!v || v->type != MetaType::BOOL)
            throw std::runtime_error("M31: missing/invalid boolean " + key);
        return v->u != 0;
    };
    Geometry g;
    g.layers = integer("block_count");
    g.embd = integer("embedding_length");
    g.heads = integer("attention.head_count");
    g.kv_heads = integer("attention.head_count_kv");
    g.head_dim = integer("attention.key_length");
    if (integer("attention.value_length") != g.head_dim)
        throw std::runtime_error("M31: key/value head dimensions differ");
    g.rope_dim = integer("rope.dimension_count");
    g.context = integer("context_length");
    g.dense_ff = integer("feed_forward_length");
    g.expert_ff = integer("expert_feed_forward_length");
    g.dense_lead = integer("leading_dense_block_count", true);
    g.experts = integer("expert_count");
    g.used = integer("expert_used_count");
    g.shared = integer("expert_shared_count");
    g.index_heads = integer("attention.indexer.head_count");
    g.index_dim = integer("attention.indexer.key_length");
    g.block = integer("attention.indexer.block_size");
    g.top_blocks = integer("attention.indexer.top_k");
    g.local_blocks = integer("attention.indexer.local_blocks", true);
    g.rms_eps = float(number("attention.layer_norm_rms_epsilon"));
    g.rope_base = float(number("rope.freq_base"));
    g.route_scale = float(number("expert_weights_scale"));
    g.norm_routes = boolean(p + "expert_weights_norm", true);
    g.nfc = boolean("tokenizer.ggml.normalizer.nfc", false);
    if (integer("expert_gating_func") != 2)
        throw std::runtime_error("M31: only sigmoid expert gating (2) is supported");
    if (g.layers > 1024 || g.experts > 4096 || g.heads > 4096 || g.head_dim > 4096 ||
        g.embd > 65536 || g.dense_ff > 262144 || g.expert_ff > 65536 || g.index_dim > 4096 ||
        g.block > 256 || 256 % g.block != 0 || g.shared > 16 || g.dense_lead >= g.layers || g.used > g.experts ||
        g.heads % g.kv_heads || g.index_heads != g.kv_heads || g.rope_dim % 2 ||
        g.rope_dim > g.head_dim || g.rope_dim > g.index_dim || g.local_blocks > g.top_blocks ||
        !std::isfinite(g.rms_eps) || !std::isfinite(g.rope_base) || !std::isfinite(g.route_scale) ||
        !(g.rms_eps > 0) || !(g.rope_base > 0) || !(g.route_scale > 0))
        throw std::runtime_error("M31: unsupported/inconsistent geometry");

    std::set<std::string> expected;
    auto tensor = [&](const std::string & name, std::vector<uint64_t> shape,
                      bool f32 = false) -> const TensorInfo & {
        size_t sh = 0;
        const TensorInfo * t = model.find(name, &sh);
        if (!t) throw std::runtime_error("M31: missing tensor " + name);
        if (t->shape != shape || (f32 && t->type != 0))
            throw std::runtime_error("M31: wrong shape/type for " + name);
        int block = 0, bytes = 0;
        if (!block_geometry(t->type, block, bytes) || shape[0] % uint64_t(block) || !model.in_bounds(*t, sh))
            throw std::runtime_error("M31: unsupported type/alignment or truncated tensor " + name);
        expected.insert(name);
        return *t;
    };
    auto u = [](int64_t x) { return uint64_t(x); };
    const TensorInfo * te = model.find("token_embd.weight");
    if (!te || te->shape.size() != 2 || te->shape[1] < 1 || te->shape[1] > 16777216)
        throw std::runtime_error("M31: invalid token_embd.weight");
    g.vocab = int64_t(te->shape[1]);
    tensor("token_embd.weight", {u(g.embd), u(g.vocab)});
    tensor("output.weight", {u(g.embd), u(g.vocab)});
    tensor("output_norm.weight", {u(g.embd)}, true);
    for (int64_t l = 0; l < g.layers; ++l) {
        auto t = [&](const std::string & s, std::vector<uint64_t> shape, bool f32 = false) -> const TensorInfo & {
            return tensor(layer_name(l, s), std::move(shape), f32);
        };
        t("attn_norm.weight", {u(g.embd)}, true);
        t("ffn_norm.weight", {u(g.embd)}, true);
        t("attn_q.weight", {u(g.embd), u(g.head_dim * g.heads)});
        for (const char * role : {"k", "v"})
            t(std::string("attn_") + role + ".weight", {u(g.embd), u(g.head_dim * g.kv_heads)});
        t("attn_output.weight", {u(g.head_dim * g.heads), u(g.embd)});
        t("attn_q_norm.weight", {u(g.head_dim)}, true);
        t("attn_k_norm.weight", {u(g.head_dim)}, true);
        // M3.1 requires all FOUR index tensors even on layers with a dense FFN.
        t("indexer.q_proj.weight", {u(g.embd), u(g.index_dim * g.index_heads)});
        t("indexer.k_proj.weight", {u(g.embd), u(g.index_dim)});
        t("indexer.q_norm.weight", {u(g.index_dim)}, true);
        t("indexer.k_norm.weight", {u(g.index_dim)}, true);
        if (!g.routed(l)) {
            t("ffn_gate.weight", {u(g.embd), u(g.dense_ff)});
            t("ffn_up.weight", {u(g.embd), u(g.dense_ff)});
            t("ffn_down.weight", {u(g.dense_ff), u(g.embd)});
            g.gate_types.push_back(-1); g.up_types.push_back(-1); g.down_types.push_back(-1);
        } else {
            t("ffn_gate_inp.weight", {u(g.embd), u(g.experts)});
            t("exp_probs_b.bias", {u(g.experts)}, true);
            int types[3] = {};
            int r = 0;
            for (const char * role : {"gate", "up", "down"}) {
                const bool down = r == 2;
                const std::string stem = std::string("ffn_") + role;
                const uint64_t n0 = u(down ? g.expert_ff : g.embd);
                const uint64_t n1 = u(down ? g.embd : g.expert_ff);
                types[r++] = int(t(stem + "_exps.weight", {n0, n1, u(g.experts)}).type);
                const std::string scale = stem + "_exps.scale";
                if (model.find(layer_name(l, scale))) t(scale, {u(g.experts)}, true);
                else if (types[r - 1] == 40)
                    throw std::runtime_error("M31: NVFP4 tensor requires " + layer_name(l, scale));
                t(stem + "_shexp.weight", down ?
                    std::vector<uint64_t>{u(g.expert_ff * g.shared), u(g.embd)} :
                    std::vector<uint64_t>{u(g.embd), u(g.expert_ff * g.shared)});
            }
            g.gate_types.push_back(types[0]); g.up_types.push_back(types[1]); g.down_types.push_back(types[2]);
        }
    }
    for (size_t sh = 0; sh < model.size(); ++sh) {
        for (const auto & t : model.shard(sh).tensors()) {
            if (!expected.count(t.name)) throw std::runtime_error("M31: unexpected tensor " + t.name);
            const uint64_t bytes = tensor_payload_bytes(t);
            if (ends_with(t.name, "_exps.weight")) g.expert_bytes += bytes;
            else g.other_bytes += bytes;
            if (ends_with(t.name, "_exps.scale")) g.expert_scale_bytes += bytes;
            if (t.name == "token_embd.weight") g.embedding_bytes = bytes;
        }
    }
    return g;
}

} // namespace strata::m31
