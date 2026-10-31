// Checked name/shape adapter for the llama.cpp DeepSeek-V4.1 DSpark GGUF convention.
// This is the three-stage, single-pass DSpark draft (not arbitrary DFlash).
#pragma once

#include "ds41_geometry.hpp"

#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::v41 {

struct Ds41DSparkGgufBinding {
    std::string source;
    std::string destination;
    bool force_f32 = false;  // norms/biases/scales used by elementwise graph operations
};

inline bool ds41_is_dflash_dspark(const GgufFile & file) {
    const MetaValue * arch = file.get("general.architecture");
    return arch && arch->type == MetaType::STRING && arch->s == "dflash";
}

// The geometry still belongs to the target. Fail before allocating backend weights if the draft does not describe
// the exact architecture the existing DSpark graph executes. Payload bounds are checked separately by the loader.
inline std::string ds41_dflash_dspark_bindings(const std::map<std::string, MetaValue> & metadata,
                                              const std::vector<TensorInfo> & tensors,
                                              const Ds41Geometry & g, int block, int used, int noise,
                                              std::vector<Ds41DSparkGgufBinding> & bindings) {
    bindings.clear();
    auto get = [&](const std::string & key) -> const MetaValue * {
        const auto i = metadata.find(key);
        return i == metadata.end() ? nullptr : &i->second;
    };
    auto number = [&](const std::string & key, double expected, bool runtime_float = false) {
        const MetaValue * value = get(key);
        return value && value->is_num() && std::isfinite(value->num()) &&
               (runtime_float ? (float) value->num() == (float) expected : value->num() == expected);
    };
    auto exact_array = [&](const std::string & key, const std::vector<int64_t> & expected) {
        const MetaValue * value = get(key);
        if (!value || value->type != MetaType::ARRAY || value->count != expected.size() ||
            value->items.size() != expected.size()) return false;
        for (size_t i = 0; i < expected.size(); ++i)
            if (!value->items[i].is_num() || value->items[i].num() != (double) expected[i]) return false;
        return true;
    };
    const MetaValue * arch = get("general.architecture");
    if (!arch || arch->s != "dflash") return "DSpark GGUF: architecture is not dflash";
    if (g.n_layer != 40 || g.n_swa != 128 || block != 5 || used != 3)
        return "DSpark GGUF: this graph requires the V4.1 40-layer target, window 128, block 5 and top-3 draft";
    for (int64_t x : {g.n_embd, g.n_ff, g.n_head, g.d_head, g.d_rope, g.r_q, g.o_groups, g.o_lora, g.hc})
        if (x <= 0 || x > (1 << 20)) return "DSpark GGUF: invalid target geometry";
    if (g.vocab <= 0 || g.vocab > (1 << 24) || g.ctx_len <= 0 ||
        g.n_head * g.d_head % g.o_groups != 0)
        return "DSpark GGUF: invalid target geometry";
    const std::string p = "dflash.";
    const std::vector<std::pair<std::string, double>> required = {
        {"block_count", 3}, {"context_length", (double) g.ctx_len}, {"embedding_length", (double) g.n_embd},
        {"feed_forward_length", (double) g.n_ff}, {"attention.head_count", (double) g.n_head},
        {"attention.head_count_kv", 1}, {"attention.key_length", (double) g.d_head},
        {"attention.value_length", (double) g.d_head}, {"rope.dimension_count", (double) g.d_rope},
        {"rope.freq_base", g.rope_base}, {"attention.layer_norm_rms_epsilon", g.rms_eps},
        {"attention.q_lora_rank", (double) g.r_q}, {"attention.sliding_window", 128},
        {"expert_count", 128}, {"expert_used_count", 3},
        {"expert_feed_forward_length", (double) g.n_ff}, {"expert_shared_count", 1},
        {"expert_weights_scale", g.weights_scale}, {"expert_weights_norm", g.norm_topk ? 1.0 : 0.0},
        {"expert_gating_func", 4}, {"attention.output_group_count", (double) g.o_groups},
        {"attention.output_lora_rank", (double) g.o_lora}, {"hyper_connection.count", (double) g.hc},
        {"hyper_connection.sinkhorn_iterations", (double) g.hc_iters}, {"hyper_connection.epsilon", g.hc_eps},
        {"hyper_connection.single_pass", 1}, {"block_size", 5},
        {"sample_from_anchor", 1}, {"has_confidence_head", 1}
    };
    const std::set<std::string> runtime_floats = {"rope.freq_base", "attention.layer_norm_rms_epsilon",
                                                 "expert_weights_scale", "hyper_connection.epsilon"};
    for (const auto & kv : required)
        if (!number(p + kv.first, kv.second, runtime_floats.count(kv.first) != 0))
            return "DSpark GGUF: missing or incompatible " + p + kv.first;
    if (!number("tokenizer.ggml.mask_token_id", noise)) return "DSpark GGUF: noise token differs from the runtime";
    if (!exact_array(p + "target_layers", {37, 38, 39}) ||
        !exact_array(p + "attention.compress_ratios", {0, 0, 0}))
        return "DSpark GGUF: incompatible target layers or attention compression";
    for (const char * key : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
        const MetaValue * value = get(p + key);
        if (!value || value->type != MetaType::ARRAY || value->count != 3 || value->items.size() != 3)
            return "DSpark GGUF: missing " + p + key;
        for (const MetaValue & x : value->items)
            if (!x.is_num() || (float) x.num() != (float) g.swiglu_limit)
                return "DSpark GGUF: incompatible " + p + key;
    }

    struct Spec { std::string destination; std::vector<uint64_t> shape; bool elementwise; bool expert; };
    std::map<std::string, Spec> expected;
    auto add = [&](const std::string & source, const std::string & destination, std::vector<uint64_t> shape,
                   bool elementwise = false, bool expert = false) {
        expected.emplace(source, Spec{destination, std::move(shape), elementwise, expert});
    };
    const uint64_t d = (uint64_t) g.n_embd, hc = (uint64_t) g.hc, mix = (uint64_t) g.hc_mix();
    const uint64_t dh = (uint64_t) g.d_head, nh = (uint64_t) g.n_head, rq = (uint64_t) g.r_q;
    const uint64_t og = (uint64_t) g.o_groups, ol = (uint64_t) g.o_lora, ff = (uint64_t) g.n_ff;
    const uint64_t vocab = (uint64_t) g.vocab;
    if (!d || !hc || !dh || !nh || !rq || !og || !ol || !ff || !vocab || nh * dh % og)
        return "DSpark GGUF: invalid target geometry";
    add("fc.weight", "mtp.0.main_proj.weight", {3 * d, d});
    add("enc.output_norm.weight", "mtp.0.main_norm.weight", {d}, true);
    add("output_norm.weight", "mtp.2.norm.weight", {d}, true);
    add("markov_w1.weight", "mtp.2.markov_head.embed.weight", {256, vocab});
    add("markov_w2.weight", "mtp.2.markov_head.head.weight", {256, vocab});
    add("conf_proj.weight", "mtp.2.confidence_head.proj.weight", {d + 256, 1});
    for (int stage = 0; stage < 3; ++stage) {
        const std::string s = "blk." + std::to_string(stage) + ".";
        const std::string t = "mtp." + std::to_string(stage) + ".";
        add(s + "attn_sinks.weight", t + "attn.attn_sink", {nh}, true);
        add(s + "attn_kv_a_norm.weight", t + "attn.kv_norm.weight", {dh}, true);
        add(s + "attn_q_a_norm.weight", t + "attn.q_norm.weight", {rq}, true);
        add(s + "attn_kv.weight", t + "attn.wkv.weight", {d, dh});
        add(s + "attn_output_a.weight", t + "attn.wo_a.weight", {nh * dh / og, og * ol});
        add(s + "attn_output_b.weight", t + "attn.wo_b.weight", {og * ol, d});
        add(s + "attn_q_a.weight", t + "attn.wq_a.weight", {d, rq});
        add(s + "attn_q_b.weight", t + "attn.wq_b.weight", {rq, nh * dh});
        add(s + "attn_norm.weight", t + "attn_norm.weight", {d}, true);
        add(s + "ffn_norm.weight", t + "ffn_norm.weight", {d}, true);
        add(s + "exp_probs_b.bias", t + "ffn.gate.bias", {128}, true);
        add(s + "ffn_gate_inp.weight", t + "ffn.gate.weight", {d, 128}, true);
        add(s + "ffn_gate_exps.weight", t + "ffn.experts.w1.weight", {d, ff, 128}, false, true);
        add(s + "ffn_down_exps.weight", t + "ffn.experts.w2.weight", {ff, d, 128}, false, true);
        add(s + "ffn_up_exps.weight", t + "ffn.experts.w3.weight", {d, ff, 128}, false, true);
        add(s + "ffn_gate_shexp.weight", t + "ffn.shared_experts.w1.weight", {d, ff});
        add(s + "ffn_down_shexp.weight", t + "ffn.shared_experts.w2.weight", {ff, d});
        add(s + "ffn_up_shexp.weight", t + "ffn.shared_experts.w3.weight", {d, ff});
        for (const char * site : {"attn", "ffn"}) {
            const std::string n = std::string("hc_") + site;
            add(s + n + "_fn.weight", t + n + "_fn", {hc * d, mix});
            add(s + n + "_base.weight", t + n + "_base", {mix}, true);
            add(s + n + "_scale.weight", t + n + "_scale", {3}, true);
        }
    }
    std::set<std::string> seen;
    std::vector<Ds41DSparkGgufBinding> result;
    for (const TensorInfo & tensor : tensors) {
        const auto i = expected.find(tensor.name);
        if (i == expected.end()) return "DSpark GGUF: unexpected tensor " + tensor.name;
        if (!seen.insert(tensor.name).second) return "DSpark GGUF: duplicate tensor " + tensor.name;
        const Spec & spec = i->second;
        bool shape_ok = tensor.shape == spec.shape;
        // llama.cpp may preserve grouped output-A as a 3-D tensor; the graph already reshapes this same byte order.
        if (!shape_ok && tensor.name.find(".attn_output_a.weight") != std::string::npos)
            shape_ok = tensor.shape == std::vector<uint64_t>{nh * dh / og, ol, og};
        if (!shape_ok) return "DSpark GGUF: incompatible shape for " + tensor.name;
        const bool float_type = tensor.type == 0 || tensor.type == 1 || tensor.type == 30;
        const bool type_ok = spec.expert ? tensor.type == 39 : spec.elementwise ? float_type : float_type || tensor.type == 8;
        if (!type_ok) return "DSpark GGUF: unsupported type for " + tensor.name;
        result.push_back({tensor.name, spec.destination, spec.elementwise});
    }
    if (seen.size() != expected.size()) {
        for (const auto & item : expected)
            if (!seen.count(item.first)) return "DSpark GGUF: missing tensor " + item.first;
    }
    bindings = std::move(result);
    return {};
}

}  // namespace strata::v41
