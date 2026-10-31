// Pure CPU validation of the modern DeepSeek-V4.1 DSpark header adapter. No tensor payload or backend needed.
#include "ds41_dspark_gguf.hpp"
#include <cstdio>
#include <stdexcept>
using namespace strata;
using namespace strata::v41;

namespace {
void number(std::map<std::string, MetaValue> & m, const std::string & k, double n) {
    MetaValue v; v.type = MetaType::F64; v.f = n; m[k] = v;
}
void array(std::map<std::string, MetaValue> & m, const std::string & k, std::initializer_list<int64_t> xs) {
    MetaValue v; v.type = MetaType::ARRAY; v.elem = MetaType::I64; v.count = xs.size();
    for (int64_t n : xs) { MetaValue x; x.type = MetaType::I64; x.u = (uint64_t) n; v.items.push_back(x); }
    m[k] = v;
}
Ds41Geometry target() {
    Ds41Geometry g;
    g.n_layer = 40; g.n_embd = 5120; g.n_ff = 2304; g.n_head = 64; g.d_head = 512; g.d_rope = 64;
    g.r_q = 1280; g.o_groups = 8; g.o_lora = 1024; g.hc = 4; g.hc_iters = 20; g.n_swa = 128;
    g.vocab = 129280; g.ctx_len = 1048576; g.rope_base = 10000; g.rms_eps = 1e-20;
    g.hc_eps = 1e-6; g.weights_scale = 1.5; g.norm_topk = true; g.swiglu_limit = 10;
    return g;
}
std::map<std::string, MetaValue> metadata() {
    std::map<std::string, MetaValue> m;
    MetaValue arch; arch.type = MetaType::STRING; arch.s = "dflash"; m["general.architecture"] = arch;
    number(m, "dflash.block_count", 3);
    number(m, "dflash.context_length", 1048576);
    number(m, "dflash.embedding_length", 5120);
    number(m, "dflash.feed_forward_length", 2304);
    number(m, "dflash.attention.head_count", 64);
    number(m, "dflash.attention.head_count_kv", 1);
    number(m, "dflash.attention.key_length", 512);
    number(m, "dflash.attention.value_length", 512);
    number(m, "dflash.rope.dimension_count", 64);
    number(m, "dflash.rope.freq_base", 10000.0);
    number(m, "dflash.attention.layer_norm_rms_epsilon", 9.999999682655225e-21);
    number(m, "dflash.attention.q_lora_rank", 1280);
    number(m, "dflash.attention.sliding_window", 128);
    number(m, "dflash.expert_count", 128);
    number(m, "dflash.expert_used_count", 3);
    number(m, "dflash.expert_feed_forward_length", 2304);
    number(m, "dflash.expert_shared_count", 1);
    number(m, "dflash.expert_weights_scale", 1.5);
    number(m, "dflash.expert_weights_norm", 1);
    number(m, "dflash.expert_gating_func", 4);
    array(m, "dflash.swiglu_clamp_exp", {10, 10, 10});
    array(m, "dflash.swiglu_clamp_shexp", {10, 10, 10});
    number(m, "dflash.attention.output_group_count", 8);
    number(m, "dflash.attention.output_lora_rank", 1024);
    array(m, "dflash.attention.compress_ratios", {0, 0, 0});
    number(m, "dflash.hyper_connection.count", 4);
    number(m, "dflash.hyper_connection.sinkhorn_iterations", 20);
    number(m, "dflash.hyper_connection.epsilon", 9.999999974752427e-07);
    number(m, "dflash.hyper_connection.single_pass", 1);
    number(m, "dflash.block_size", 5);
    array(m, "dflash.target_layers", {37, 38, 39});
    number(m, "dflash.sample_from_anchor", 1);
    number(m, "dflash.has_confidence_head", 1);
    number(m, "tokenizer.ggml.mask_token_id", 128799);
    return m;
}
std::vector<TensorInfo> tensors() {
    std::vector<TensorInfo> ts;
    { TensorInfo t; t.name = "blk.0.ffn_gate_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_down_exps.weight"; t.shape = {2304, 5120, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_up_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_gate_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_down_exps.weight"; t.shape = {2304, 5120, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_up_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_gate_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_down_exps.weight"; t.shape = {2304, 5120, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_up_exps.weight"; t.shape = {5120, 2304, 128}; t.type = 39; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_sinks.weight"; t.shape = {64}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_kv_a_norm.weight"; t.shape = {512}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_q_a_norm.weight"; t.shape = {1280}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_kv.weight"; t.shape = {5120, 512}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_output_a.weight"; t.shape = {4096, 8192}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_output_b.weight"; t.shape = {8192, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_q_a.weight"; t.shape = {5120, 1280}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_q_b.weight"; t.shape = {1280, 32768}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.attn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.exp_probs_b.bias"; t.shape = {128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_gate_inp.weight"; t.shape = {5120, 128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_gate_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_down_shexp.weight"; t.shape = {2304, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_up_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.ffn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_attn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_attn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_attn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_ffn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_ffn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.0.hc_ffn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "enc.output_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "fc.weight"; t.shape = {15360, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_sinks.weight"; t.shape = {64}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_kv_a_norm.weight"; t.shape = {512}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_q_a_norm.weight"; t.shape = {1280}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_kv.weight"; t.shape = {5120, 512}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_output_a.weight"; t.shape = {4096, 8192}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_output_b.weight"; t.shape = {8192, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_q_a.weight"; t.shape = {5120, 1280}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_q_b.weight"; t.shape = {1280, 32768}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.attn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.exp_probs_b.bias"; t.shape = {128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_gate_inp.weight"; t.shape = {5120, 128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_gate_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_down_shexp.weight"; t.shape = {2304, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_up_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.ffn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_attn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_attn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_attn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_ffn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_ffn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.1.hc_ffn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_sinks.weight"; t.shape = {64}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_kv_a_norm.weight"; t.shape = {512}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_q_a_norm.weight"; t.shape = {1280}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_kv.weight"; t.shape = {5120, 512}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_output_a.weight"; t.shape = {4096, 8192}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_output_b.weight"; t.shape = {8192, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_q_a.weight"; t.shape = {5120, 1280}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_q_b.weight"; t.shape = {1280, 32768}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.attn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "conf_proj.weight"; t.shape = {5376, 1}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.exp_probs_b.bias"; t.shape = {128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_gate_inp.weight"; t.shape = {5120, 128}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_gate_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_down_shexp.weight"; t.shape = {2304, 5120}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_up_shexp.weight"; t.shape = {5120, 2304}; t.type = 8; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.ffn_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_attn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_attn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_attn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_ffn_base.weight"; t.shape = {24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_ffn_fn.weight"; t.shape = {20480, 24}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "blk.2.hc_ffn_scale.weight"; t.shape = {3}; t.type = 0; ts.push_back(t); }
    { TensorInfo t; t.name = "markov_w1.weight"; t.shape = {256, 129280}; t.type = 30; ts.push_back(t); }
    { TensorInfo t; t.name = "markov_w2.weight"; t.shape = {256, 129280}; t.type = 30; ts.push_back(t); }
    { TensorInfo t; t.name = "output_norm.weight"; t.shape = {5120}; t.type = 0; ts.push_back(t); }
    return ts;
}
}  // namespace

int main(int argc, char ** argv) {
    int fails = 0, checks = 0;
    auto check = [&](const char * name, bool ok) {
        ++checks; if (!ok) { ++fails; std::fprintf(stderr, "FAIL %s\n", name); }
    };
    auto m = metadata(); auto ts = tensors(); auto g = target();
    std::vector<Ds41DSparkGgufBinding> bs;
    auto bind = [&](const std::map<std::string, MetaValue> & meta, const std::vector<TensorInfo> & tensors,
                    const Ds41Geometry & geom) {
        return ds41_dflash_dspark_bindings(meta, tensors, geom, 5, 3, 128799, bs);
    };
    check("all 78 real-layout tensors", bind(m, ts, g).empty() && bs.size() == 78);
    std::map<std::string, std::string> names;
    std::set<std::string> destinations;
    for (const auto & b : bs) { names[b.source] = b.destination; destinations.insert(b.destination); }
    check("no destination collision", destinations.size() == 78);
    check("anchor projection", names["fc.weight"] == "mtp.0.main_proj.weight");
    check("anchor norm", names["enc.output_norm.weight"] == "mtp.0.main_norm.weight");
    check("final norm", names["output_norm.weight"] == "mtp.2.norm.weight");
    check("markov embed", names["markov_w1.weight"] == "mtp.2.markov_head.embed.weight");
    check("markov head", names["markov_w2.weight"] == "mtp.2.markov_head.head.weight");
    check("confidence", names["conf_proj.weight"] == "mtp.2.confidence_head.proj.weight");
    check("router bias", names["blk.1.exp_probs_b.bias"] == "mtp.1.ffn.gate.bias");
    check("gate/up/down order", names["blk.0.ffn_gate_exps.weight"] == "mtp.0.ffn.experts.w1.weight" &&
          names["blk.0.ffn_down_exps.weight"] == "mtp.0.ffn.experts.w2.weight" &&
          names["blk.0.ffn_up_exps.weight"] == "mtp.0.ffn.experts.w3.weight");
    auto rejects_meta = [&](const char * label, const std::string & key, double value) {
        auto changed = m; number(changed, key, value);
        check(label, !bind(changed, ts, g).empty() && bs.empty());
    };
    rejects_meta("reject two-pass HC", "dflash.hyper_connection.single_pass", 0);
    rejects_meta("reject non-anchor draft", "dflash.sample_from_anchor", 0);
    rejects_meta("reject different hidden width", "dflash.embedding_length", 4096);
    rejects_meta("reject different expert count", "dflash.expert_count", 64);
    rejects_meta("reject fractional expert count", "dflash.expert_count", 128.000001);
    rejects_meta("reject NaN router metadata", "dflash.expert_gating_func", NAN);
    rejects_meta("reject different router function", "dflash.expert_gating_func", 1);
    rejects_meta("reject different window", "dflash.attention.sliding_window", 256);
    rejects_meta("reject different noise token", "tokenizer.ggml.mask_token_id", 128798);
    rejects_meta("reject different rope", "dflash.rope.freq_base", 160000);
    rejects_meta("reject missing confidence", "dflash.has_confidence_head", 0);
    rejects_meta("reject different context support", "dflash.context_length", 65536);
    {
        auto changed = m; changed.erase("dflash.hyper_connection.epsilon");
        check("reject missing required key", !bind(changed, ts, g).empty());
        changed = m; array(changed, "dflash.target_layers", {36, 38, 39});
        check("reject incompatible feature taps", !bind(changed, ts, g).empty());
        changed = m; array(changed, "dflash.attention.compress_ratios", {0, 1, 0});
        check("reject draft compressed attention", !bind(changed, ts, g).empty());
        changed = m; array(changed, "dflash.swiglu_clamp_shexp", {10, 9, 10});
        check("reject differing shared clamp", !bind(changed, ts, g).empty());
    }
    {
        auto changed = ts; changed.pop_back();
        check("reject missing head", !bind(m, changed, g).empty());
        changed = ts; changed.push_back(changed.back());
        check("reject duplicate name", !bind(m, changed, g).empty());
        changed = ts; changed[0].shape[2] = 127;
        check("reject expert shape mismatch", !bind(m, changed, g).empty());
        changed = ts; changed[0].type = 26;
        check("reject unsupported expert type", !bind(m, changed, g).empty());
        changed = ts; changed.back().name = "markov_w2.scale";
        check("reject unsupported scaled markov", !bind(m, changed, g).empty());
        changed = ts;
        for (auto & t : changed) if (t.name == "blk.0.attn_kv_a_norm.weight") t.type = 30;
        check("BF16 norm accepted", bind(m, changed, g).empty());
        bool norm_cast = false;
        for (auto & b : bs) if (b.source == "blk.0.attn_kv_a_norm.weight") norm_cast = b.force_f32;
        check("BF16 norm converted for binary kernels", norm_cast);
        for (auto & t : changed) if (t.name == "blk.0.attn_kv_a_norm.weight") t.type = 8;
        check("reject quantized norm", !bind(m, changed, g).empty());
        changed = ts;
        for (auto & t : changed) if (t.name == "blk.0.attn_output_a.weight") t.shape = {4096, 1024, 8};
        check("grouped 3D output-A same byte order", bind(m, changed, g).empty());
    }
    {
        auto bad = g; bad.n_embd = -1;
        check("reject invalid target geometry", !bind(m, ts, bad).empty());
        check("reject unsupported draft block", !ds41_dflash_dspark_bindings(m, ts, g, 4, 3, 128799, bs).empty());
    }
    if (argc == 2) {
        try {
            GgufFile file(argv[1]);
            const std::string e = bind(file.metadata(), file.tensors(), g);
            check("external actual header", e.empty() && bs.size() == 78);
            if (!e.empty()) std::fprintf(stderr, "%s\n", e.c_str());
            std::puts("External mode checks metadata and tensor directory only, not payload bytes or GPU execution.");
        } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); ++fails; }
    }
    std::printf("DSpark GGUF adapter: %d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
