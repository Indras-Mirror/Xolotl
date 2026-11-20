// include/strata/artifact/ernie_geometry.hpp - the ERNIE 5.5 Flash (`ernie5_5`) geometry and tensor map.
//
// Phase 1 of the port, modelled on `ds4_geometry.hpp`: ERNIE 5.5 is a DeepSeek-V4 hybrid (llama.cpp's
// `llama_model_ernie5_5` inherits `llama_model_deepseek4`), so where the two share a mechanism this file names
// it the same way and only the differences are new.
//
//   1. `ErnieGeometry` is every number a Phase-2/3 kernel depends on, read from the GGUF's own `ernie5_5.*`
//      keys - D=4096, 44 blocks (43 trunk + 1 MTP), 64 q / 1 kv heads, HCA head 512, MLA head 256 / kv_lora
//      512 / q_lora 1024, o_groups 8, 512 experts top-10, expert dim 2048 in a 2048-d latent, dense lead FFN
//      14336, hc 4 (Sinkhorn 20), window 128, the per-layer `compress_ratios` (0 = MLA, 128 = HCA), the
//      indexer 64/128/2048, situ betas, YaRN factor 16 / orig 65536, compress rope base 160000, MultiMax.
//      A key that is absent is an error, not a default.
//
//   2. `check_ernie_model` resolves the FULL tensor map: every name in the file must be one this model
//      predicts, and every predicted tensor must be present.  The per-layer set is conditional exactly as the
//      architecture is: HCA tensors (compressor, grouped output) where ratio != 0, the MLA + DSA indexer set
//      where ratio == 0, a dense FFN on the leading dense block, latent-MoE tensors on every other block, and
//      the `nextn.*` head on the MTP block.
//
// Types are NOT pinned to one set: ERNIE ships three mixed-precision GPTQ builds of the same architecture, so
// a spec declares a TYPE CLASS instead of an exact id.  `Float` (norms, routers, scalars, the indexer's small
// projections) must stay F32/F16/BF16; `Weight` (everything large) must not be F32.  That is still a real check -
// it catches a norm quantised by accident or a projection left at full precision - without refusing the
// 60G/122G/245G builds, whose per-tensor types differ.
//
// It does NOT load tensor data: `GgufModel` mmaps the header, so this runs without touching the weights.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace strata {

namespace ernie_type {
constexpr uint32_t F32 = 0;
constexpr uint32_t F16 = 1;
constexpr uint32_t BF16 = 30;  // ggml_type BF16, as gguf_reader.hpp's name table ids it
}  // namespace ernie_type

/// The ERNIE 5.5 geometry.  Field names track `llama-hparams`' ERNIE fields (src/models/ernie5-5.cpp) so a
/// value can be traced both ways; every one is read from a key by `ernie_read_geometry`.
struct ErnieGeometry {
    int64_t n_embd = 0;                   ///< embedding_length
    int64_t n_layer_all = 0;              ///< block_count (trunk + nextn blocks)
    int64_t n_layer_nextn_requested = 0;  ///< nextn_predict_layers as declared
    int64_t n_layer_nextn = 0;            ///< effective: zeroed when the nextn tensors are absent
    int64_t n_layer_dense_lead = 0;       ///< leading_dense_block_count (blocks with a plain FFN)

    // attention, shared
    int64_t n_head = 0;              ///< attention.head_count (64)
    int64_t n_head_kv = 0;           ///< attention.head_count_kv (1: MQA)
    int64_t d_rope = 0;              ///< rope.dimension_count
    int64_t r_q = 0;                 ///< attention.q_lora_rank
    int64_t o_groups = 0;            ///< attention.output_group_count
    int64_t o_lora = 0;              ///< attention.output_lora_rank
    int64_t n_swa = 0;               ///< attention.sliding_window (the HCA window)

    // attention, HCA (compress ratio != 0)
    int64_t d_head_hca = 0;          ///< attention.key_length_swa: the wide HCA head, K == V (512)
    int64_t d_head_hca_v = 0;        ///< attention.value_length_swa (512)

    // attention, MLA + DSA (compress ratio == 0)
    int64_t d_head_mla_k = 0;        ///< attention.key_length_mla (256), includes the rope slice
    int64_t d_head_mla_v = 0;        ///< attention.value_length_mla (256)
    int64_t kv_lora = 0;             ///< attention.kv_lora_rank (512)
    int64_t d_kv_row = 0;            ///< attention.key_length: the MLA cache row's K half, kv_lora + rope (576).
                                     ///< This is NOT the HCA head - `key_length_swa` is - and confusing the two
                                     ///< sizes every compressor tensor.  `ernie_read_geometry` asserts the sum.

    // latent MoE
    int64_t n_expert = 0;            ///< expert_count (512)
    int64_t n_expert_used = 0;       ///< expert_used_count (top-10)
    int64_t n_ff = 0;                ///< feed_forward_length: the DENSE LEAD block's FFN width (14336)
    int64_t n_ff_exp = 0;            ///< expert_feed_forward_length: a routed/shared expert's width (2048)
    int64_t n_expert_shared = 0;     ///< expert_shared_count
    int64_t n_embd_lat = 0;          ///< expert_latent_length: the experts' working width (2048)
    int64_t gating_func = 0;         ///< expert_gating_func
    bool weights_norm = false;       ///< expert_weights_norm (norm_topk_prob)

    // mHC
    int64_t hc = 0;                  ///< hyper_connection.count (multiplicity)
    int64_t hc_sinkhorn_iters = 0;   ///< hyper_connection.sinkhorn_iterations
    double hc_eps = 0;               ///< hyper_connection.epsilon

    // lightning indexer (MLA layers only)
    int64_t indexer_n_head = 0;      ///< attention.indexer.head_count
    int64_t indexer_key_dim = 0;     ///< attention.indexer.key_length
    int64_t indexer_top_k = 0;       ///< attention.indexer.top_k

    // activations and logits
    double situ_beta = 0;            ///< activation.situ_beta
    double situ_linear_beta = 0;     ///< activation.situ_linear_beta
    std::vector<double> logit_multimax;  ///< logit_multimax, the MultiMax head's learned parameters

    double rms_eps = 0;              ///< attention.layer_norm_rms_epsilon
    std::string rope_scaling_type;   ///< rope.scaling.type ("yarn")
    double rope_freq_base = 0;       ///< rope.freq_base
    double rope_factor = 0;          ///< rope.scaling.factor
    double rope_orig_ctx = 0;        ///< rope.scaling.original_context_length
    double yarn_beta_fast = 0;       ///< rope.scaling.yarn_beta_fast
    double yarn_beta_slow = 0;       ///< rope.scaling.yarn_beta_slow
    double compress_rope_base = 0;   ///< attention.compress_rope_freq_base

    int64_t context_length = 0;      ///< context_length
    int64_t vocab_size = 0;          ///< the tokenizer.ggml.tokens array length (there is no ernie5_5.vocab_size)
    std::string tokenizer_pre;       ///< tokenizer.ggml.pre ("ernie5")

    std::vector<int64_t> compress_ratios;  ///< attention.compress_ratios, per block (0 = MLA, 128 = HCA)

    int64_t n_layer() const { return n_layer_all - n_layer_nextn; }  ///< trunk layers
    int64_t hc_dim() const { return hc * n_embd; }                   ///< hc*D = 16384
    int64_t hc_mix_dim() const { return (2 + hc) * hc; }             ///< (2+hc)*hc = 24
    int64_t hca_heads_dim() const { return n_head * d_head_hca; }    ///< n_heads*d_h_hca = 32768
    int64_t o_group_dim() const { return hca_heads_dim() / o_groups; }  ///< = 4096
    int64_t mla_qk_nope() const { return d_head_mla_k - d_rope; }    ///< 256 - 64 = 192
    int64_t mla_heads_dim() const { return n_head * d_head_mla_v; }  ///< = 16384

    /// True where the block compresses its window (HCA).  `compress_ratios[i] == 0` is the MLA + DSA layer.
    bool is_hca(int64_t i) const {
        return i < (int64_t)compress_ratios.size() && compress_ratios[(size_t)i] != 0;
    }
    bool is_mla(int64_t i) const { return !is_hca(i); }
};

// ------------------------------------------------------------------ key readers (missing key == error)
inline std::string ernie_want_u64(const GgufFile& g, const char* key, uint64_t& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (!v->is_num()) return std::string(key) + " is not a number";
    out = (uint64_t)v->num();
    return {};
}
inline std::string ernie_want_f64(const GgufFile& g, const char* key, double& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (!v->is_num()) return std::string(key) + " is not a number";
    out = v->num();
    return {};
}
inline std::string ernie_want_bool(const GgufFile& g, const char* key, bool& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    out = v->u != 0;
    return {};
}
inline std::string ernie_key_f64_arr(const GgufFile& g, const char* key, std::vector<double>& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (v->type != MetaType::ARRAY) return std::string(key) + " is not an array";
    if (v->count > v->items.size())  // the reader samples arrays at 64 (see read_value)
        return std::string(key) + " has " + std::to_string(v->count) + " items, beyond the reader's 64-item cap";
    out.clear();
    out.reserve(v->items.size());
    for (const auto& e : v->items) out.push_back(e.num());
    return {};
}
inline std::string ernie_key_i64_arr(const GgufFile& g, const char* key, std::vector<int64_t>& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (v->type != MetaType::ARRAY) return std::string(key) + " is not an array";
    if (v->count > v->items.size())
        return std::string(key) + " has " + std::to_string(v->count) + " items, beyond the reader's 64-item cap";
    out.clear();
    out.reserve(v->items.size());
    for (const auto& e : v->items) out.push_back((int64_t)e.num());
    return {};
}

/// The cheap architecture check: `general.architecture == "ernie5_5"` and the values a kernel would branch on.
inline std::string check_architecture_ernie(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "ernie5_5") return "architecture is '" + arch->s + "', this engine requires 'ernie5_5'";
    struct Req {
        const char* key;
        uint64_t want;
    };
    const Req reqs[] = {
        {"ernie5_5.block_count", 44},
        {"ernie5_5.embedding_length", 4096},
        {"ernie5_5.attention.head_count", 64},
        {"ernie5_5.attention.head_count_kv", 1},
        {"ernie5_5.expert_count", 512},
        {"ernie5_5.expert_used_count", 10},
        {"ernie5_5.expert_latent_length", 2048},
        {"ernie5_5.hyper_connection.count", 4},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    return {};  // empty == ok
}

/// Read every `ernie5_5.*` key into `out`, from the METADATA shard alone (`find` answers "is this tensor in the
/// model", which is all `ernie_read_geometry` needs of the other shards).  The single-shard form exists because
/// ERNIE ships as 58 shards: a harness that has opened only shard 1 can still read the whole geometry and check
/// every tensor that shard holds.  On the first missing/mistyped key returns the reason; the geometry is then
/// unreliable and must not be used.
inline std::string ernie_read_geometry_meta(
        const GgufFile& g, ErnieGeometry& out,
        const std::function<const TensorInfo*(const std::string&)>& find) {
    std::string err;
    uint64_t u = 0;

    if (!(err = ernie_want_u64(g, "ernie5_5.embedding_length", u)).empty()) return err;
    out.n_embd = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.block_count", u)).empty()) return err;
    out.n_layer_all = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.leading_dense_block_count", u)).empty()) return err;
    out.n_layer_dense_lead = (int64_t)u;

    if (!(err = ernie_want_u64(g, "ernie5_5.attention.head_count", u)).empty()) return err;
    out.n_head = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.head_count_kv", u)).empty()) return err;
    out.n_head_kv = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.rope.dimension_count", u)).empty()) return err;
    out.d_rope = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.q_lora_rank", u)).empty()) return err;
    out.r_q = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.output_group_count", u)).empty()) return err;
    out.o_groups = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.output_lora_rank", u)).empty()) return err;
    out.o_lora = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.sliding_window", u)).empty()) return err;
    out.n_swa = (int64_t)u;

    if (!(err = ernie_want_u64(g, "ernie5_5.attention.key_length_swa", u)).empty()) return err;
    out.d_head_hca = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.value_length_swa", u)).empty()) return err;
    out.d_head_hca_v = (int64_t)u;

    if (!(err = ernie_want_u64(g, "ernie5_5.attention.key_length_mla", u)).empty()) return err;
    out.d_head_mla_k = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.value_length_mla", u)).empty()) return err;
    out.d_head_mla_v = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.kv_lora_rank", u)).empty()) return err;
    out.kv_lora = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.key_length", u)).empty()) return err;
    out.d_kv_row = (int64_t)u;
    if (out.d_kv_row != out.kv_lora + out.d_rope)
        return "attention.key_length (" + std::to_string(out.d_kv_row) + ") is not kv_lora_rank + rope (" +
               std::to_string(out.kv_lora) + " + " + std::to_string(out.d_rope) + "); it is not the HCA head - that "
               "is attention.key_length_swa";

    if (!(err = ernie_want_u64(g, "ernie5_5.expert_count", u)).empty()) return err;
    out.n_expert = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.expert_used_count", u)).empty()) return err;
    out.n_expert_used = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.feed_forward_length", u)).empty()) return err;
    out.n_ff = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.expert_feed_forward_length", u)).empty()) return err;
    out.n_ff_exp = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.expert_shared_count", u)).empty()) return err;
    out.n_expert_shared = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.expert_latent_length", u)).empty()) return err;
    out.n_embd_lat = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.expert_gating_func", u)).empty()) return err;
    out.gating_func = (int64_t)u;
    if (!(err = ernie_want_bool(g, "ernie5_5.expert_weights_norm", out.weights_norm)).empty()) return err;

    if (!(err = ernie_want_u64(g, "ernie5_5.hyper_connection.count", u)).empty()) return err;
    out.hc = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.hyper_connection.sinkhorn_iterations", u)).empty()) return err;
    out.hc_sinkhorn_iters = (int64_t)u;
    if (!(err = ernie_want_f64(g, "ernie5_5.hyper_connection.epsilon", out.hc_eps)).empty()) return err;

    if (!(err = ernie_want_u64(g, "ernie5_5.attention.indexer.head_count", u)).empty()) return err;
    out.indexer_n_head = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.indexer.key_length", u)).empty()) return err;
    out.indexer_key_dim = (int64_t)u;
    if (!(err = ernie_want_u64(g, "ernie5_5.attention.indexer.top_k", u)).empty()) return err;
    out.indexer_top_k = (int64_t)u;

    if (!(err = ernie_want_f64(g, "ernie5_5.activation.situ_beta", out.situ_beta)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.activation.situ_linear_beta", out.situ_linear_beta)).empty()) return err;
    if (!(err = ernie_key_f64_arr(g, "ernie5_5.logit_multimax", out.logit_multimax)).empty()) return err;

    if (!(err = ernie_want_f64(g, "ernie5_5.attention.layer_norm_rms_epsilon", out.rms_eps)).empty()) return err;
    {
        const MetaValue* v = g.get("ernie5_5.rope.scaling.type");
        if (!v) return "missing ernie5_5.rope.scaling.type";
        out.rope_scaling_type = v->s;
    }
    if (!(err = ernie_want_f64(g, "ernie5_5.rope.freq_base", out.rope_freq_base)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.rope.scaling.factor", out.rope_factor)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.rope.scaling.original_context_length", out.rope_orig_ctx)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.rope.scaling.yarn_beta_fast", out.yarn_beta_fast)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.rope.scaling.yarn_beta_slow", out.yarn_beta_slow)).empty()) return err;
    if (!(err = ernie_want_f64(g, "ernie5_5.attention.compress_rope_freq_base", out.compress_rope_base)).empty()) return err;

    if (!(err = ernie_want_u64(g, "ernie5_5.context_length", u)).empty()) return err;
    out.context_length = (int64_t)u;
    {
        const MetaValue* v = g.get("tokenizer.ggml.pre");
        if (!v) return "missing tokenizer.ggml.pre";
        out.tokenizer_pre = v->s;
    }
    {
        // There is no ernie5_5.vocab_size key: the vocabulary length is the tokenizer array's count (the reader
        // samples the items at 64, so `count` is the only honest source).
        const MetaValue* v = g.get("tokenizer.ggml.tokens");
        if (!v) return "missing tokenizer.ggml.tokens";
        if (v->type != MetaType::ARRAY) return "tokenizer.ggml.tokens is not an array";
        out.vocab_size = (int64_t)v->count;
    }

    if (!(err = ernie_key_i64_arr(g, "ernie5_5.attention.compress_ratios", out.compress_ratios)).empty()) return err;
    if ((int64_t)out.compress_ratios.size() < out.n_layer_all)
        return "ernie5_5.attention.compress_ratios has " + std::to_string(out.compress_ratios.size()) +
               " entries, need at least block_count (" + std::to_string(out.n_layer_all) + ")";
    for (int64_t r : out.compress_ratios)
        if (r != 0 && r != 128) return "compress_ratios contains " + std::to_string(r) + " (only 0 = MLA, 128 = HCA)";

    if (!(err = ernie_want_u64(g, "ernie5_5.nextn_predict_layers", u)).empty()) return err;
    out.n_layer_nextn_requested = (int64_t)u;
    out.n_layer_nextn = 0;
    if (out.n_layer_nextn_requested > 0 && out.n_layer_nextn_requested < out.n_layer_all) {
        // llama.cpp zeroes it when the MTP head's projection is absent (the MTP block carries `nextn.*`).
        const std::string probe = "blk." + std::to_string(out.n_layer_all - out.n_layer_nextn_requested) +
                                  ".nextn.enorm.weight";
        out.n_layer_nextn = find(probe) == nullptr ? 0 : out.n_layer_nextn_requested;
    }
    return {};
}

/// The whole-model form: the metadata shard plus a lookup across every shard.
inline std::string ernie_read_geometry(const GgufModel& model, ErnieGeometry& out) {
    return ernie_read_geometry_meta(model.meta(), out,
                                    [&model](const std::string& n) { return model.find(n); });
}

// ------------------------------------------------------------------ the tensor map
/// Whether a spec pins its tensors' storage class.  ERNIE ships three mixed-precision builds, so only the class is
/// checked: `Float` tensors must stay F32/F16/BF16 (a quantised norm or router is a bug), and `Weight` tensors must
/// not be F32 (a 280B model cannot carry F32 projections; F16/BF16 pass, as the real file's indexer does).  See the
/// header comment.
enum class ErnieTypeClass { Float, Weight };

struct ErnieTensorSpec {
    std::string name;
    std::vector<uint64_t> shape;  // GGUF order: dim 0 varies fastest
    ErnieTypeClass cls = ErnieTypeClass::Weight;
};

namespace ernie_detail {
inline ErnieTensorSpec spec(const std::string& n, std::initializer_list<uint64_t> ne, ErnieTypeClass c) {
    ErnieTensorSpec s;
    s.name = n;
    s.shape.assign(ne.begin(), ne.end());
    s.cls = c;
    return s;
}
inline bool type_matches(uint32_t t, ErnieTypeClass c) {
    const bool is_float = t == ernie_type::F32 || t == ernie_type::F16 || t == ernie_type::BF16;
    return c == ErnieTypeClass::Float ? is_float : t != ernie_type::F32;
}
}  // namespace ernie_detail

/// Every tensor the architecture predicts, by name, with the shape a kernel will assume.  The per-layer
/// conditionals mirror `llama_model_ernie5_5::load_arch_tensors` (src/models/ernie5-5.cpp).
///
/// On the real 122G/245G builds this resolves to exactly 1514 tensors (the file's own `split.tensors.count`).
inline std::vector<ErnieTensorSpec> ernie_expected_tensors(const ErnieGeometry& g) {
    using ernie_detail::spec;
    using TC = ErnieTypeClass;
    std::vector<ErnieTensorSpec> v;

    const uint64_t D = (uint64_t)g.n_embd;
    const uint64_t hc = (uint64_t)g.hc;
    const uint64_t hcm = (uint64_t)g.hc_mix_dim();
    const uint64_t hcd = (uint64_t)g.hc_dim();
    const uint64_t V = (uint64_t)g.vocab_size;
    const uint64_t rq = (uint64_t)g.r_q;
    const uint64_t nh = (uint64_t)g.n_head;

    // root
    v.push_back(spec("token_embd.weight", {D, V}, TC::Weight));
    v.push_back(spec("output_norm.weight", {D}, TC::Float));
    v.push_back(spec("output.weight", {D, V}, TC::Weight));
    v.push_back(spec("output_hc_fn.weight", {hcd, hc}, TC::Float));
    v.push_back(spec("output_hc_base.weight", {hc}, TC::Float));
    v.push_back(spec("output_hc_scale.weight", {1}, TC::Float));

    for (int64_t i = 0; i < g.n_layer_all; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_mtp = i >= g.n_layer();

        // ---- common to every block
        v.push_back(spec(p + "attn_norm.weight", {D}, TC::Float));
        v.push_back(spec(p + "ffn_norm.weight", {D}, TC::Float));
        v.push_back(spec(p + "hc_attn_fn.weight", {hcd, hcm}, TC::Float));
        v.push_back(spec(p + "hc_attn_base.weight", {hcm}, TC::Float));
        v.push_back(spec(p + "hc_attn_scale.weight", {3}, TC::Float));
        v.push_back(spec(p + "hc_ffn_fn.weight", {hcd, hcm}, TC::Float));
        v.push_back(spec(p + "hc_ffn_base.weight", {hcm}, TC::Float));
        v.push_back(spec(p + "hc_ffn_scale.weight", {3}, TC::Float));
        v.push_back(spec(p + "attn_sinks.weight", {nh}, TC::Float));
        v.push_back(spec(p + "attn_vha.weight", {nh, nh}, TC::Float));
        v.push_back(spec(p + "attn_q_a.weight", {D, rq}, TC::Weight));
        v.push_back(spec(p + "attn_q_a_norm.weight", {rq}, TC::Float));

        if (g.is_hca(i)) {
            // ---- HCA: grouped low-rank output, a compressor, and no indexer
            const uint64_t dh = (uint64_t)g.d_head_hca;
            const uint64_t og = (uint64_t)g.o_groups, ol = (uint64_t)g.o_lora;
            const uint64_t ratio = (uint64_t)g.compress_ratios[(size_t)i];
            v.push_back(spec(p + "attn_q_b.weight", {rq, nh * dh}, TC::Weight));
            v.push_back(spec(p + "attn_kv.weight", {D, dh}, TC::Weight));
            v.push_back(spec(p + "attn_kv_a_norm.weight", {dh}, TC::Float));
            v.push_back(spec(p + "attn_output_a.weight", {(uint64_t)g.o_group_dim(), ol, og}, TC::Weight));
            v.push_back(spec(p + "attn_output_b.weight", {og * ol, D}, TC::Weight));
            v.push_back(spec(p + "attn_gate.weight", {rq, og * ol}, TC::Weight));
            v.push_back(spec(p + "attn_compressor_wkv.weight", {D, dh}, TC::Weight));
            v.push_back(spec(p + "attn_compressor_wgate.weight", {D, dh}, TC::Weight));
            v.push_back(spec(p + "attn_compressor_ape.weight", {dh, ratio}, TC::Float));
            v.push_back(spec(p + "attn_compressor_norm.weight", {dh}, TC::Float));
        } else {
            // ---- MLA + DSA: absorbed latent, per-head K/V, and the lightning indexer
            const uint64_t qk_nope = (uint64_t)g.mla_qk_nope();
            const uint64_t dk = (uint64_t)g.d_head_mla_k;
            const uint64_t dv = (uint64_t)g.d_head_mla_v;
            const uint64_t kl = (uint64_t)g.kv_lora;
            const uint64_t id = (uint64_t)g.indexer_key_dim;
            const uint64_t ih = (uint64_t)g.indexer_n_head;
            v.push_back(spec(p + "attn_q_b.weight", {rq, nh * dk}, TC::Weight));
            v.push_back(spec(p + "attn_kv_a_mqa.weight", {D, kl + (uint64_t)g.d_rope}, TC::Weight));
            v.push_back(spec(p + "attn_kv_a_norm.weight", {kl}, TC::Float));
            v.push_back(spec(p + "attn_k_b.weight", {qk_nope, kl, nh}, TC::Weight));
            v.push_back(spec(p + "attn_v_b.weight", {kl, dv, nh}, TC::Weight));
            v.push_back(spec(p + "attn_output.weight", {nh * dv, D}, TC::Weight));
            v.push_back(spec(p + "attn_gate.weight", {rq, nh * dv}, TC::Weight));
            v.push_back(spec(p + "indexer.attn_k.weight", {D, id}, TC::Weight));
            v.push_back(spec(p + "indexer.k_norm.weight", {id}, TC::Float));
            v.push_back(spec(p + "indexer.k_norm.bias", {id}, TC::Float));
            v.push_back(spec(p + "indexer.attn_q_b.weight", {rq, ih * id}, TC::Weight));
            v.push_back(spec(p + "indexer.proj.weight", {D, ih}, TC::Float));
        }

        if (i < g.n_layer_dense_lead) {
            // ---- the dense lead block: a plain SwiGLU/situ FFN at the full feed_forward_length
            const uint64_t nff = (uint64_t)g.n_ff;
            v.push_back(spec(p + "ffn_gate.weight", {D, nff}, TC::Weight));
            v.push_back(spec(p + "ffn_up.weight", {D, nff}, TC::Weight));
            v.push_back(spec(p + "ffn_down.weight", {nff, D}, TC::Weight));
            continue;
        }

        // ---- latent MoE: two routers stacked, a down/norm/up around the experts, experts at half width
        const uint64_t ne = (uint64_t)g.n_expert;
        const uint64_t lat = (uint64_t)g.n_embd_lat;
        const uint64_t fe = (uint64_t)g.n_ff_exp;
        const uint64_t se = (uint64_t)g.n_expert_shared;
        v.push_back(spec(p + "ffn_gate_inp.weight", {D, 2 * ne}, TC::Float));
        v.push_back(spec(p + "exp_probs_b.bias", {ne}, TC::Float));
        v.push_back(spec(p + "exp_scale.weight", {ne}, TC::Float));
        v.push_back(spec(p + "ffn_routed_down.weight", {D, lat}, TC::Weight));
        v.push_back(spec(p + "ffn_routed_up.weight", {lat, D}, TC::Weight));
        v.push_back(spec(p + "ffn_routed_norm.weight", {lat}, TC::Float));
        v.push_back(spec(p + "ffn_gate_exps.weight", {lat, fe, ne}, TC::Weight));
        v.push_back(spec(p + "ffn_down_exps.weight", {fe, lat, ne}, TC::Weight));
        v.push_back(spec(p + "ffn_up_exps.weight", {lat, fe, ne}, TC::Weight));
        v.push_back(spec(p + "ffn_gate_shexp.weight", {D, fe * se}, TC::Weight));
        v.push_back(spec(p + "ffn_down_shexp.weight", {fe * se, D}, TC::Weight));
        v.push_back(spec(p + "ffn_up_shexp.weight", {D, fe * se}, TC::Weight));

        if (is_mtp) {
            // ---- the MTP (ECHO) block: its own mHC head, the same layer reused as a drafter
            v.push_back(spec(p + "nextn.eh_proj.weight", {2 * D, D}, TC::Weight));
            v.push_back(spec(p + "nextn.enorm.weight", {D}, TC::Float));
            v.push_back(spec(p + "nextn.hnorm.weight", {D}, TC::Float));
            v.push_back(spec(p + "nextn.shared_head_norm.weight", {D}, TC::Float));
            v.push_back(spec(p + "nextn.hc_head_fn.weight", {hcd, hc}, TC::Float));
            v.push_back(spec(p + "nextn.hc_head_base.weight", {hc}, TC::Float));
            v.push_back(spec(p + "nextn.hc_head_scale.weight", {1}, TC::Float));
        }
    }
    return v;
}

/// The name -> spec lookup the checks use.
inline std::map<std::string, const ErnieTensorSpec*> ernie_spec_index(const std::vector<ErnieTensorSpec>& want) {
    std::map<std::string, const ErnieTensorSpec*> by_name;
    for (const auto& s : want) by_name[s.name] = &s;
    return by_name;
}

/// ONE tensor against its spec.  `bounded` is the caller's `in_bounds` answer: a single shard opened on its own
/// has no `GgufModel` to ask.
inline std::string ernie_check_one(const ErnieTensorSpec& w, const TensorInfo& t, bool bounded) {
    if (t.shape.size() != w.shape.size()) {
        std::string got;
        for (auto d : t.shape) got += (got.empty() ? "" : ",") + std::to_string(d);
        return "tensor '" + t.name + "' has " + std::to_string(t.shape.size()) + " dims (" + got + "), expected " +
               std::to_string(w.shape.size());
    }
    for (size_t d = 0; d < w.shape.size(); ++d)
        if (t.shape[d] != w.shape[d])
            return "tensor '" + t.name + "' dim " + std::to_string(d) + " = " + std::to_string(t.shape[d]) +
                   ", expected " + std::to_string(w.shape[d]);
    if (!ernie_detail::type_matches(t.type, w.cls))
        return std::string("tensor '") + t.name + "' is " + ggml_type_name(t.type) +
               (w.cls == ErnieTypeClass::Float ? ", expected F32/F16/BF16 (this tensor stays unquantized)"
                                               : ", expected a quantized or F16/BF16 matrix (not F32)");
    if (!bounded) return "tensor '" + t.name + "' payload is outside its shard";
    return {};
}

/// Resolve the full tensor map against a model.  Returns "" when every tensor in the file is expected AND every
/// expected tensor is present with the shape and type class the kernels assume; otherwise the first failure,
/// naming the tensor (and, for a shape/type mismatch, what it has vs what is required).
inline std::string check_ernie_model(const GgufModel& model, const ErnieGeometry& g) {
    const std::vector<ErnieTensorSpec> want = ernie_expected_tensors(g);
    const auto by_name = ernie_spec_index(want);

    for (size_t sh = 0; sh < model.size(); ++sh) {
        for (const auto& t : model.shard(sh).tensors()) {
            const auto it = by_name.find(t.name);
            if (it == by_name.end())
                return "unknown tensor '" + t.name + "' in " + model.shard(sh).path() +
                       " (not in the ernie5_5 tensor map)";
            const std::string e = ernie_check_one(*it->second, t, model.in_bounds(t, sh));
            if (!e.empty()) return e;
        }
    }
    // In a split model the later shards' tensors are all checked above; the presence pass is what catches a
    // tensor the file omits entirely.
    for (const auto& s : want)
        if (model.find(s.name) == nullptr) return "missing tensor '" + s.name + "'";
    return {};
}

/// The expected tensors a model does not hold.  The lookup is injected so a harness that has opened ONE shard of
/// a 58-shard file can check the rest of the map instead of having the whole load refused.
inline std::vector<std::string> ernie_missing_tensors(
        const std::vector<ErnieTensorSpec>& want,
        const std::function<const TensorInfo*(const std::string&)>& find) {
    std::vector<std::string> missing;
    for (const auto& s : want)
        if (find(s.name) == nullptr) missing.push_back(s.name);
    return missing;
}
inline std::vector<std::string> ernie_missing_tensors(const GgufModel& model, const ErnieGeometry& g) {
    return ernie_missing_tensors(ernie_expected_tensors(g),
                                 [&model](const std::string& n) { return model.find(n); });
}

}  // namespace strata
