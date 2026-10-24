// tools/v41/engine/ds41_geometry.hpp - DeepSeek-V4.1-Flash (`deepseek41`) geometry, tensor map and expert tiers.
//
// The GGUF is the one tools/v41/v41_gguf.py writes (DwarfStar's dense names, per-expert routed tiers).  Every number
// comes from the file's `deepseek41.*` keys (or the `deepseek41.config` JSON for the Engram shape the flat keys lack);
// a missing key is an error, never a default.  `ds41_check_tensors` resolves the full tensor map: every name in the
// file must be predicted and every predicted name present, with the conditional sets exactly as model.py builds them:
// compressor tensors on kv source layers (gate only where ratio > 1), indexer query/proj on index source layers,
// indexer key on kv sources, Engram on its layers, routed experts either one stacked tensor per role or one per type
// + the I32 type maps.
//
// Header-only; reads the header region only (GgufModel mmaps it), no tensor data except the two I32 type maps.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "ds41_sys.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::v41 {

namespace ty {
constexpr uint32_t F32 = 0, F16 = 1, Q8_0 = 8, Q3_K = 11, I8 = 24, I32 = 26, MXFP4 = 39, Q2_0 = 42, PTQ1_0 = 143;
}

/// One layer's routed experts: the type of each expert's gate+up and of its down, and where each expert's rows are
/// (tensor name + slot inside it).  Uniform roles have a single tensor (slot == expert id).
struct Ds41ExpertMap {
    std::vector<int> gu_type, d_type;              ///< [n_expert] ggml type ids
    std::vector<int> gu_slot, d_slot;              ///< [n_expert] index inside the type's stacked tensor
    std::map<int, std::string> gate_t, up_t, down_t;   ///< type -> tensor name
};

struct Ds41Geometry {
    int64_t n_embd = 0, n_layer = 0, n_head = 0, d_head = 0, d_rope = 0, r_q = 0, o_lora = 0, o_groups = 0;
    int64_t n_swa = 0, n_expert = 0, n_used = 0, n_ff = 0, n_shared = 0;
    double weights_scale = 0;
    bool norm_topk = false;
    int64_t hc = 0, hc_iters = 0;
    double hc_eps = 0, rms_eps = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0;
    int64_t cand_layer = -1, cand_blocks = 0, cand_bsize = 0;
    double rope_base = 0, compress_rope_base = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0, orig_ctx = 0;
    int64_t vocab = 0, ctx_len = 0;
    double swiglu_limit = 0;
    std::vector<int64_t> ratio;                      ///< [n_layer] 0 / 2 / 1
    std::vector<int64_t> kv_sources, index_sources;
    // Engram
    std::vector<int64_t> engram_layers, engram_rows; ///< layer ids, table rows per layer
    int64_t engram_heads = 0, engram_dim = 0, engram_ngram = 0, engram_pad = 0, engram_cvocab = 0;
    std::vector<uint32_t> engram_token_map;          ///< [vocab] compressed id
    std::vector<uint64_t> engram_primes;             ///< [n_engram_layers * (ngram-1) * heads]
    std::vector<int64_t> engram_mult;                ///< [n_engram_layers * ngram]
    int64_t bos = -1, eos = -1;
    std::vector<Ds41ExpertMap> experts;              ///< [n_layer]

    bool is_kv_source(int64_t l) const { return contains(kv_sources, l); }
    bool is_index_source(int64_t l) const { return contains(index_sources, l); }
    int64_t engram_slot(int64_t l) const {
        for (size_t i = 0; i < engram_layers.size(); ++i) if (engram_layers[i] == l) return (int64_t) i;
        return -1;
    }
    /// The kv source whose compressed KV layer l reads (model.py shared_attn.compress_kv: the latest source <= l), or -1.
    int64_t kv_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : kv_sources) if (k <= l) s = k;
        return s;
    }
    /// The index source whose top-k layer l uses (the latest index source <= l), or -1.
    int64_t idx_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : index_sources) if (k <= l) s = k;
        return s;
    }
    int64_t hc_mix() const { return (2 + hc) * hc; }

    /// True if a pipeline may start a new stage at layer `split`.  Layers on the next stage that read indexer or
    /// compressed KV from below `split` are carried as shadow layers (`stage_import_layers`); handoff copies their
    /// live cache at each column boundary, so a split does not have to fall on an index-source layer.
    bool split_boundary_ok(int64_t split) const {
        if (split <= 0 || split >= n_layer) return false;
        const std::vector<int64_t> imp = stage_import_layers(split, n_layer);
        auto imported = [&](int64_t x) {
            if (x < 0) return true;
            for (int64_t y : imp) if (y == x) return true;
            return false;
        };
        for (int64_t l = split; l < n_layer; ++l) {
            const int64_t ix = idx_src(l);
            const int64_t kv = kv_src(l);
            if (ix >= 0 && ix < split && !imported(ix)) return false;
            if (kv >= 0 && kv < split && !imported(kv)) return false;
            if (cand_layer >= 0 && l > cand_layer && is_index_source(l) && cand_layer < split && !imported(cand_layer))
                return false;
        }
        return true;
    }
    std::vector<int> valid_split_boundaries() const {
        std::vector<int> out;
        for (int64_t b = 1; b < n_layer; ++b)
            if (split_boundary_ok(b)) out.push_back((int) b);
        return out;
    }
    /// Layers on the previous stage that must be copied when entering [stage_lo, stage_hi).
    std::vector<int64_t> stage_import_layers(int64_t stage_lo, int64_t stage_hi) const {
        std::vector<int64_t> out;
        auto add = [&](int64_t x) {
            if (x < 0 || x >= stage_lo) return;
            for (int64_t y : out) if (y == x) return;
            out.push_back(x);
        };
        for (int64_t l = stage_lo; l < stage_hi; ++l) {
            add(idx_src(l));
            add(kv_src(l));
        }
        if (cand_layer >= 0 && cand_layer < stage_lo) {
            for (int64_t l = stage_lo; l < stage_hi; ++l)
                if (l > cand_layer && is_index_source(l)) add(cand_layer);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

private:
    static bool contains(const std::vector<int64_t> & v, int64_t x) {
        for (int64_t y : v) if (y == x) return true;
        return false;
    }
};

// ------------------------------------------------------------------ full metadata arrays
// strata::GgufFile keeps 64 items of an array (a sample); the Engram token map has 129,280.  This walks the KV
// section of the file itself and returns one key's whole array (integers only), or an error.
namespace detail {
struct Rd {
    int fd = -1;
    uint64_t pos = 0;
    std::vector<uint8_t> buf;
    uint64_t buf_at = 0;
    bool fill(uint64_t at, size_t n) {
        if (at >= buf_at && at + n <= buf_at + buf.size()) return true;
        const size_t want = std::max<size_t>(n, 1 << 20);
        buf.resize(want);
        const ssize_t r = ds41_sys::pread(fd, buf.data(), want, (int64_t) at);
        if (r < (ssize_t) n) return false;
        buf.resize((size_t) r);
        buf_at = at;
        return true;
    }
    bool get(void * dst, size_t n) {
        if (!fill(pos, n)) return false;
        std::memcpy(dst, buf.data() + (pos - buf_at), n);
        pos += n;
        return true;
    }
    template <class T> bool get(T & v) { return get(&v, sizeof(T)); }
};
inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;   // U8 I8 BOOL
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}
}  // namespace detail

inline std::string ds41_read_int_array(const std::string & path, const std::string & key, std::vector<int64_t> & out) {
    detail::Rd r;
    r.fd = ds41_sys::open(path.c_str(), O_RDONLY);
    if (r.fd < 0) return "cannot open " + path;
    struct Closer { int fd; ~Closer() { ds41_sys::close(fd); } } closer { r.fd };
    uint32_t magic = 0, version = 0;
    uint64_t n_t = 0, n_kv = 0;
    if (!r.get(magic) || magic != 0x46554747u || !r.get(version) || !r.get(n_t) || !r.get(n_kv)) return "bad GGUF header";
    auto skip_str = [&]() { uint64_t n; if (!r.get(n)) return false; r.pos += n; return true; };
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t kl = 0;
        if (!r.get(kl) || kl > 4096) return "bad key";
        std::string k(kl, '\0');
        if (!r.get(k.data(), kl)) return "bad key";
        uint32_t t = 0;
        if (!r.get(t)) return "bad type";
        if (t == 8) { if (!skip_str()) return "bad string"; continue; }
        if (t != 9) { const size_t s = detail::scalar_size(t); if (!s) return "bad scalar type"; r.pos += s; continue; }
        uint32_t et = 0;
        uint64_t cnt = 0;
        if (!r.get(et) || !r.get(cnt)) return "bad array";
        if (k != key) {
            if (et == 8) { for (uint64_t j = 0; j < cnt; ++j) if (!skip_str()) return "bad string array"; }
            else { const size_t s = detail::scalar_size(et); if (!s) return "bad array type"; r.pos += s * cnt; }
            continue;
        }
        const size_t s = detail::scalar_size(et);
        if (!s || et == 7 || et == 6 || et == 12) return key + ": not an integer array";
        out.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) {
            uint8_t b[8] = {};
            if (!r.get(b, s)) return key + ": truncated";
            int64_t v = 0;
            switch (et) {
                case 0: v = b[0]; break;
                case 1: v = (int8_t) b[0]; break;
                case 2: { uint16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 3: { int16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 4: { uint32_t x; std::memcpy(&x, b, 4); v = x; break; }
                case 5: { int32_t x; std::memcpy(&x, b, 4); v = x; break; }
                default: std::memcpy(&v, b, 8); break;
            }
            out[j] = v;
        }
        return {};
    }
    return "missing " + key;
}

// ------------------------------------------------------------------ key readers (missing key == error)
namespace detail {
inline std::string u64(const GgufFile & g, const std::string & k, int64_t & out) {
    const MetaValue * v = g.get(k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = (int64_t) v->num();
    return {};
}
inline std::string f64(const GgufFile & g, const std::string & k, double & out) {
    const MetaValue * v = g.get(k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = v->num();
    return {};
}
inline std::string arr(const GgufFile & g, const std::string & k, std::vector<int64_t> & out) {
    const MetaValue * v = g.get(k);
    if (!v) return "missing " + k;
    if (v->type != MetaType::ARRAY) return k + " is not an array";
    if (v->count > v->items.size()) return k + ": longer than the reader's sample (use ds41_read_int_array)";
    out.clear();
    for (const auto & e : v->items) out.push_back((int64_t) e.num());
    return {};
}
/// `"key":<number>` out of the deepseek41.config JSON (flat search; the keys are unique in DeepSeek's config)
inline std::string json_num(const std::string & js, const std::string & key, double & out) {
    const std::string pat = "\"" + key + "\":";
    const size_t p = js.find(pat);
    if (p == std::string::npos) return "deepseek41.config has no " + key;
    out = std::strtod(js.c_str() + p + pat.size(), nullptr);
    return {};
}
}  // namespace detail

/// A `deepseek41` GGUF in the llama.cpp convention (llama.cpp PR #30817's converter; ISTA-DASLab's GSQ-RCO GGUFs):
/// llama.cpp key names, the Engram hash arrays as I32/I64 tensors, its tables in a separate file.  Ours has
/// `deepseek41.config`.
inline bool ds41_is_llamacpp_gguf(const GgufModel & model) {
    return model.meta().get("deepseek41.config") == nullptr && model.meta().get("deepseek41.embedding_length") != nullptr;
}

namespace detail {
/// an I32 / I64 tensor of the file, whole (the llama.cpp convention's Engram arrays)
inline std::string int_tensor(const GgufModel & m, const std::string & n, std::vector<int64_t> & out) {
    size_t si = 0;
    const TensorInfo * t = m.find(n, &si);
    if (!t) return "missing tensor " + n;
    if (t->type != 26 && t->type != 27) return n + ": not I32/I64";
    uint64_t cnt = 1;
    for (uint64_t d : t->shape) cnt *= d;
    if (!m.in_bounds(*t, si)) return n + ": out of bounds";
    const uint8_t * p = m.shard(si).tensor_data(*t);
    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) {
        if (t->type == 26) { int32_t x; std::memcpy(&x, p + 4 * i, 4); out[i] = x; }
        else std::memcpy(&out[i], p + 8 * i, 8);
    }
    return {};
}
}  // namespace detail

/// ds41_read_geometry for the llama.cpp convention: the same Ds41Geometry from its key names (every key still
/// required).  The routing, attention and Engram hash constants are DeepSeek's either way - only where they live
/// differs (checked on ISTA's IQ2_XXS: token map, primes and multipliers byte-equal to ours).
inline std::string ds41_read_geometry_llamacpp(const GgufModel & model, Ds41Geometry & G) {
    const GgufFile & g = model.meta();
    std::string e;
    const std::string P = "deepseek41.";
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "embedding_length", &G.n_embd }, { "block_count", &G.n_layer }, { "attention.head_count", &G.n_head },
             { "attention.key_length", &G.d_head }, { "rope.dimension_count", &G.d_rope },
             { "attention.q_lora_rank", &G.r_q }, { "attention.output_lora_rank", &G.o_lora },
             { "attention.output_group_count", &G.o_groups }, { "attention.sliding_window", &G.n_swa },
             { "expert_count", &G.n_expert }, { "expert_used_count", &G.n_used },
             { "expert_feed_forward_length", &G.n_ff }, { "expert_shared_count", &G.n_shared },
             { "hyper_connection.count", &G.hc }, { "hyper_connection.sinkhorn_iterations", &G.hc_iters },
             { "attention.indexer.head_count", &G.idx_heads }, { "attention.indexer.key_length", &G.idx_dim },
             { "attention.indexer.top_k", &G.idx_topk }, { "attention.candidate_source_layer_id", &G.cand_layer },
             { "attention.candidate_topk_blocks", &G.cand_blocks }, { "attention.candidate_block_size", &G.cand_bsize },
             { "context_length", &G.ctx_len }, { "engram_n_heads", &G.engram_heads },
             { "engram_head_dim", &G.engram_dim }, { "engram_max_ngram_size", &G.engram_ngram },
             { "engram_pad_token_id", &G.engram_pad }, { "engram_compressed_vocab_size", &G.engram_cvocab } })
        if (!(e = detail::u64(g, P + u.k, *u.v)).empty()) return e;
    struct F { const char * k; double * v; };
    for (const F & f : std::initializer_list<F>{
             { "expert_weights_scale", &G.weights_scale }, { "hyper_connection.epsilon", &G.hc_eps },
             { "attention.layer_norm_rms_epsilon", &G.rms_eps }, { "rope.freq_base", &G.rope_base },
             { "attention.compress_rope_freq_base", &G.compress_rope_base }, { "rope.scaling.factor", &G.rope_factor },
             { "rope.scaling.yarn_beta_fast", &G.beta_fast }, { "rope.scaling.yarn_beta_slow", &G.beta_slow },
             { "rope.scaling.original_context_length", &G.orig_ctx } })
        if (!(e = detail::f64(g, P + f.k, *f.v)).empty()) return e;
    {
        const MetaValue * v = g.get(P + "expert_weights_norm");
        if (!v) return "missing deepseek41.expert_weights_norm";
        G.norm_topk = v->u != 0;
        const MetaValue * sf = g.get(P + "expert_scoring_func");
        if (!sf || sf->s != "sqrtsoftplus") return "expert_scoring_func is not sqrtsoftplus";
        // one SwiGLU clamp for every layer (DeepSeek's swiglu_limit); a per-layer array of equal values here
        const MetaValue * sl = g.get(P + "swiglu_clamp_exp");
        if (!sl || sl->type != MetaType::ARRAY || sl->items.empty()) return "missing deepseek41.swiglu_clamp_exp";
        G.swiglu_limit = sl->items[0].num();
        for (const auto & it : sl->items)
            if (it.num() != G.swiglu_limit) return "swiglu_clamp_exp differs between layers (one limit supported)";
    }
    if (!(e = detail::arr(g, P + "attention.compress_ratios", G.ratio)).empty()) return e;
    if ((int64_t) G.ratio.size() > G.n_layer) G.ratio.resize((size_t) G.n_layer);   // (+ the 3 MTP blocks' entries)
    if (!(e = detail::arr(g, P + "kv_source_layer_ids", G.kv_sources)).empty()) return e;
    if (!(e = detail::arr(g, P + "index_source_layer_ids", G.index_sources)).empty()) return e;
    if (!(e = detail::arr(g, P + "engram.layer_ids", G.engram_layers)).empty()) return e;
    if (!(e = detail::arr(g, P + "engram.num_embeddings", G.engram_rows)).empty()) return e;
    const TensorInfo * te = model.find("token_embd.weight");
    if (!te || te->shape.size() != 2) return "missing token_embd.weight";
    G.vocab = (int64_t) te->shape[1];
    std::vector<int64_t> tm, pr;
    if (!(e = detail::int_tensor(model, "engram_token_map.weight", tm)).empty()) return e;
    if (!(e = detail::int_tensor(model, "engram_primes.weight", pr)).empty()) return e;
    if (!(e = detail::int_tensor(model, "engram_multipliers.weight", G.engram_mult)).empty()) return e;
    if ((int64_t) tm.size() != G.vocab) return "engram_token_map size";
    G.engram_token_map.assign(tm.begin(), tm.end());
    G.engram_primes.assign(pr.begin(), pr.end());
    detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
    detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
    return {};
}

/// the checks both conventions share (after the keys are read)
inline std::string ds41_check_geometry(Ds41Geometry & G) {
    if ((int64_t) G.ratio.size() != G.n_layer) return "compress_ratios has " + std::to_string(G.ratio.size()) + " entries";
    for (int64_t r : G.ratio) if (r != 0 && r != 1 && r != 2) return "compress ratio " + std::to_string(r) + " (V4.1 has 0/1/2)";
    for (int64_t l : G.kv_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad kv source";
    for (int64_t l : G.index_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad index source";
    for (int64_t l = 0; l < G.n_layer; ++l)
        if (G.ratio[(size_t) l] && (G.kv_src(l) < 0 || G.idx_src(l) < 0 || G.ratio[(size_t) G.kv_src(l)] != G.ratio[(size_t) l]))
            return "layer " + std::to_string(l) + " has no kv/index source of its ratio";
    if (G.n_shared != 1 || G.hc < 1 || G.d_head <= G.d_rope || (G.n_head * G.d_head) % G.o_groups) return "unsupported shape";
    const int64_t nel = (int64_t) G.engram_layers.size();
    if ((int64_t) G.engram_primes.size() != nel * (G.engram_ngram - 1) * G.engram_heads ||
        (int64_t) G.engram_mult.size() != nel * G.engram_ngram || (int64_t) G.engram_rows.size() != nel)
        return "engram arrays do not match its shape";
    for (int64_t i = 0; i < nel; ++i) {   // the bucket ranges tile the table exactly (v41_gguf_meta checks the same)
        uint64_t s = 0;
        for (int64_t j = 0; j < (G.engram_ngram - 1) * G.engram_heads; ++j)
            s += G.engram_primes[(size_t) (i * (G.engram_ngram - 1) * G.engram_heads + j)];
        if ((int64_t) s != G.engram_rows[(size_t) i]) return "engram primes do not tile the table";
    }
    return {};
}

inline std::string ds41_read_geometry(const GgufModel & model, Ds41Geometry & G) {
    const GgufFile & g = model.meta();
    const MetaValue * arch = g.get("general.architecture");
    if (!arch || arch->s != "deepseek41") return "not a deepseek41 GGUF";
    if (ds41_is_llamacpp_gguf(model)) {
        const std::string e = ds41_read_geometry_llamacpp(model, G);
        return e.empty() ? ds41_check_geometry(G) : e;
    }
    std::string e;
    const std::string P = "deepseek41.";
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "hidden_size", &G.n_embd }, { "num_hidden_layers", &G.n_layer }, { "num_attention_heads", &G.n_head },
             { "head_dim", &G.d_head }, { "qk_rope_head_dim", &G.d_rope }, { "q_lora_rank", &G.r_q },
             { "o_lora_rank", &G.o_lora }, { "o_groups", &G.o_groups }, { "sliding_window", &G.n_swa },
             { "n_routed_experts", &G.n_expert }, { "num_experts_per_tok", &G.n_used },
             { "moe_intermediate_size", &G.n_ff }, { "n_shared_experts", &G.n_shared }, { "hc_mult", &G.hc },
             { "hc_sinkhorn_iters", &G.hc_iters }, { "index_n_heads", &G.idx_heads }, { "index_head_dim", &G.idx_dim },
             { "index_topk", &G.idx_topk }, { "candidate_source_layer_id", &G.cand_layer },
             { "candidate_topk_blocks", &G.cand_blocks }, { "candidate_block_size", &G.cand_bsize },
             { "vocab_size", &G.vocab }, { "max_position_embeddings", &G.ctx_len } })
        if (!(e = detail::u64(g, P + u.k, *u.v)).empty()) return e;
    struct F { const char * k; double * v; };
    for (const F & f : std::initializer_list<F>{
             { "routed_scaling_factor", &G.weights_scale }, { "hc_eps", &G.hc_eps }, { "rms_norm_eps", &G.rms_eps },
             { "rope_theta", &G.rope_base }, { "compress_rope_theta", &G.compress_rope_base },
             { "rope_scaling.factor", &G.rope_factor }, { "rope_scaling.beta_fast", &G.beta_fast },
             { "rope_scaling.beta_slow", &G.beta_slow }, { "rope_scaling.original_max_position_embeddings", &G.orig_ctx },
             { "swiglu_limit", &G.swiglu_limit } })
        if (!(e = detail::f64(g, P + f.k, *f.v)).empty()) return e;
    {
        const MetaValue * v = g.get(P + "norm_topk_prob");
        if (!v) return "missing deepseek41.norm_topk_prob";
        G.norm_topk = v->u != 0;
        const MetaValue * sf = g.get(P + "scoring_func");
        if (!sf || sf->s != "sqrtsoftplus") return "scoring_func is not sqrtsoftplus";
    }
    if (!(e = detail::arr(g, P + "compress_ratios", G.ratio)).empty()) return e;
    if (!(e = detail::arr(g, P + "kv_source_layer_ids", G.kv_sources)).empty()) return e;
    if (!(e = detail::arr(g, P + "index_source_layer_ids", G.index_sources)).empty()) return e;
    if ((int64_t) G.ratio.size() != G.n_layer) return "compress_ratios has " + std::to_string(G.ratio.size()) + " entries";
    for (int64_t r : G.ratio) if (r != 0 && r != 1 && r != 2) return "compress ratio " + std::to_string(r) + " (V4.1 has 0/1/2)";
    for (int64_t l : G.kv_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad kv source";
    for (int64_t l : G.index_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad index source";
    for (int64_t l = 0; l < G.n_layer; ++l)
        if (G.ratio[(size_t) l] && (G.kv_src(l) < 0 || G.idx_src(l) < 0 || G.ratio[(size_t) G.kv_src(l)] != G.ratio[(size_t) l]))
            return "layer " + std::to_string(l) + " has no kv/index source of its ratio";
    if (G.n_shared != 1 || G.hc < 1 || G.d_head <= G.d_rope || (G.n_head * G.d_head) % G.o_groups) return "unsupported shape";
    // Engram: arrays (full, not the reader's sample) + the shape from the config JSON
    const MetaValue * cfg = g.get(P + "config");
    if (!cfg) return "missing deepseek41.config";
    double d = 0;
    if (!(e = detail::json_num(cfg->s, "engram_n_heads", d)).empty()) return e;
    G.engram_heads = (int64_t) d;
    if (!(e = detail::json_num(cfg->s, "engram_head_dim", d)).empty()) return e;
    G.engram_dim = (int64_t) d;
    if (!(e = detail::json_num(cfg->s, "engram_max_ngram_size", d)).empty()) return e;
    G.engram_ngram = (int64_t) d;
    if (!(e = detail::u64(g, P + "engram.pad_id", G.engram_pad)).empty()) return e;
    if (!(e = detail::u64(g, P + "engram.compressed_vocab_size", G.engram_cvocab)).empty()) return e;
    if (!(e = detail::arr(g, P + "engram.layer_ids", G.engram_layers)).empty()) return e;
    if (!(e = detail::arr(g, P + "engram.rows", G.engram_rows)).empty()) return e;
    const std::string & path = g.path();
    std::vector<int64_t> tm, pr;
    if (!(e = ds41_read_int_array(path, P + "engram.token_map", tm)).empty()) return e;
    if (!(e = ds41_read_int_array(path, P + "engram.primes", pr)).empty()) return e;
    if (!(e = ds41_read_int_array(path, P + "engram.multipliers", G.engram_mult)).empty()) return e;
    if ((int64_t) tm.size() != G.vocab) return "engram.token_map size";
    G.engram_token_map.assign(tm.begin(), tm.end());
    G.engram_primes.assign(pr.begin(), pr.end());
    const int64_t nel = (int64_t) G.engram_layers.size();
    if ((int64_t) G.engram_primes.size() != nel * (G.engram_ngram - 1) * G.engram_heads ||
        (int64_t) G.engram_mult.size() != nel * G.engram_ngram || (int64_t) G.engram_rows.size() != nel)
        return "engram arrays do not match its shape";
    for (int64_t i = 0; i < nel; ++i) {   // the bucket ranges tile the table exactly (v41_gguf_meta checks the same)
        uint64_t s = 0;
        for (int64_t j = 0; j < (G.engram_ngram - 1) * G.engram_heads; ++j)
            s += G.engram_primes[(size_t) (i * (G.engram_ngram - 1) * G.engram_heads + j)];
        if ((int64_t) s != G.engram_rows[(size_t) i]) return "engram primes do not tile the table";
    }
    detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
    detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
    return {};
}

/// The full tensor map + per-layer expert tiers.  `layers`: the blocks the file must hold (a partial test file
/// holds a subset - pass them; the real file holds 0..n_layer-1).
inline std::string ds41_check_tensors(const GgufModel & m, Ds41Geometry & G, const std::vector<int64_t> & layers) {
    std::map<std::string, std::pair<std::vector<uint64_t>, uint32_t>> want;   // name -> (shape GGUF order, type; ~0 = any)
    const uint32_t ANY = ~0u;
    // the llama.cpp convention (ISTA-DASLab's GGUFs): the same tensors and shapes, but dense types of the quantizer's
    // choice (Q5_K / Q8_0 / BF16 / F32 - the loader converts what the graph needs), attn_output_a stored 3-D, the
    // Engram projections under llama.cpp's names, no vision router bias, and the Engram hash arrays as tensors
    const bool lc = ds41_is_llamacpp_gguf(m);
    auto add = [&](const std::string & n, std::vector<uint64_t> sh, uint32_t t) { want[n] = { std::move(sh), lc ? ANY : t }; };
    const uint64_t D = (uint64_t) G.n_embd, HC = (uint64_t) G.hc, MIX = (uint64_t) G.hc_mix();
    const uint64_t DH = (uint64_t) G.d_head, NH = (uint64_t) G.n_head, RQ = (uint64_t) G.r_q;
    const uint64_t OG = (uint64_t) G.o_groups, OL = (uint64_t) G.o_lora, NE = (uint64_t) G.n_expert, FF = (uint64_t) G.n_ff;
    const uint64_t IH = (uint64_t) G.idx_heads, ID = (uint64_t) G.idx_dim;
    add("token_embd.weight", { D, (uint64_t) G.vocab }, ty::F16);
    add("output_norm.weight", { D }, ty::F32);
    add("output.weight", { D, (uint64_t) G.vocab }, ty::Q8_0);
    G.experts.assign((size_t) G.n_layer, {});
    for (int64_t l : layers) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * site : { "attn", "ffn" }) {
            add(b + "hc_" + site + "_fn.weight", { HC * D, MIX }, ty::F16);
            add(b + "hc_" + site + "_base.weight", { MIX }, ty::F32);
            add(b + "hc_" + site + "_scale.weight", { 3 }, ty::F32);
            add(b + site + "_norm.weight", { D }, ty::F32);
        }
        add(b + "attn_sinks.weight", { NH }, ty::F32);
        add(b + "attn_q_a.weight", { D, RQ }, ty::Q8_0);
        add(b + "attn_q_b.weight", { RQ, NH * DH }, ty::Q8_0);
        add(b + "attn_q_a_norm.weight", { RQ }, ty::F32);
        add(b + "attn_kv.weight", { D, DH }, ty::Q8_0);
        add(b + "attn_kv_a_norm.weight", { DH }, ty::F32);
        if (lc) add(b + "attn_output_a.weight", { NH * DH / OG, OL, OG }, ANY);   // the same bytes, 3-D
        else add(b + "attn_output_a.weight", { NH * DH / OG, OG * OL }, ty::Q8_0);
        add(b + "attn_output_b.weight", { OG * OL, D }, ty::Q8_0);
        if (G.is_kv_source(l)) {
            add(b + "attn_compressor_kv.weight", { D, DH }, ty::F16);
            add(b + "attn_compressor_norm.weight", { DH }, ty::F32);
            add(b + "indexer.attn_k.weight", { DH, ID }, ty::F16);
            add(b + "indexer.k_norm.weight", { ID }, ty::F32);
            if (G.ratio[(size_t) l] > 1) add(b + "attn_compressor_gate.weight", { D, DH }, ty::F16);
        }
        if (G.is_index_source(l)) {
            add(b + "indexer.attn_q_b.weight", { RQ, IH * ID }, ty::F16);
            add(b + "indexer.proj.weight", { D, IH }, ty::F16);
        }
        add(b + "ffn_gate_inp.weight", { D, NE }, ty::F32);
        add(b + "exp_probs_b.bias", { NE }, ty::F32);
        if (!lc) add(b + "exp_probs_b_vl.bias", { NE }, ty::F32);   // (vision only; absent in the llama.cpp convention)
        add(b + "ffn_gate_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_up_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_down_shexp.weight", { FF, D }, ty::Q8_0);
        if (G.engram_slot(l) >= 0) {   // (llama.cpp names: engram_q / engram_k / engram_wkv - ds41_canonical_name)
            add(b + (lc ? "engram_q.weight" : "engram_q_norm.weight"), { D, HC }, ty::F32);
            add(b + (lc ? "engram_k.weight" : "engram_k_norm.weight"), { D, HC }, ty::F32);
            add(b + (lc ? "engram_wkv.weight" : "engram_kv.weight"),
                { (uint64_t) ((G.engram_ngram - 1) * G.engram_heads * G.engram_dim), (HC + 1) * D }, ty::F16);
        }
        // routed experts: uniform role tensors or per-type tensors + the I32 maps
        Ds41ExpertMap & X = G.experts[(size_t) l];
        X.gu_type.assign(NE, -1); X.d_type.assign(NE, -1); X.gu_slot.assign(NE, -1); X.d_slot.assign(NE, -1);
        for (int k = 0; k < 2; ++k) {   // 0 = gate+up, 1 = down
            const std::string mapn = b + (k ? "ffn_exps_down_type" : "ffn_exps_gu_type");
            std::vector<int> & tv = k ? X.d_type : X.gu_type;
            std::vector<int> & sv = k ? X.d_slot : X.gu_slot;
            const char * roles[2] = { k ? "down" : "gate", k ? nullptr : "up" };
            size_t si = 0;
            const TensorInfo * tm = m.find(mapn, &si);
            if (!tm) {   // uniform: one stacked tensor per role
                const TensorInfo * t0 = m.find(b + "ffn_" + roles[0] + "_exps.weight");
                if (!t0) return "missing " + b + "ffn_" + roles[0] + "_exps.weight (and no " + mapn + ")";
                for (uint64_t e = 0; e < NE; ++e) { tv[e] = (int) t0->type; sv[e] = (int) e; }
                if (!k) {   // gate and up share one type and one byte stride per expert, even where the dense types are free (#38)
                    const TensorInfo * tu = m.find(b + "ffn_up_exps.weight");
                    if (tu && tu->type != t0->type)
                        return b + "ffn_up_exps.weight is " + std::string(ggml_type_name(tu->type)) + ", its gate is " +
                               ggml_type_name(t0->type) + " (gate and up must share a type)";
                }
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string n = b + "ffn_" + r + "_exps.weight";
                    add(n, k ? std::vector<uint64_t>{ FF, D, NE } : std::vector<uint64_t>{ D, FF, NE }, t0->type);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[(int) t0->type] = n;
                }
                continue;
            }
            add(mapn, { NE }, ty::I32);
            if (tm->type != ty::I32 || tm->shape.size() != 1 || tm->shape[0] != NE) return mapn + ": bad shape/type";
            const int32_t * mv = reinterpret_cast<const int32_t *>(m.shard(si).tensor_data(*tm));
            std::map<int, int> count;
            for (uint64_t e = 0; e < NE; ++e) { tv[e] = mv[e]; sv[e] = count[mv[e]]++; }
            for (const auto & [t, n] : count) {
                // MXFP4 = an expert kept exactly as trained (the checkpoint's own FP4 bytes)
                if (t != (int) ty::Q2_0 && t != (int) ty::Q3_K && t != (int) ty::PTQ1_0 && t != (int) ty::MXFP4)
                    return mapn + ": type " + std::to_string(t) + " is not an expert tier";
                std::string tn = ggml_type_name((uint32_t) t);
                for (char & c : tn) c = (char) std::tolower((unsigned char) c);
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string nm = b + "ffn_" + r + "_exps." + tn + ".weight";
                    add(nm, k ? std::vector<uint64_t>{ FF, D, (uint64_t) n } : std::vector<uint64_t>{ D, FF, (uint64_t) n }, (uint32_t) t);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[t] = nm;
                }
            }
        }
    }
    if (lc) {   // the Engram hash arrays (read by ds41_read_geometry_llamacpp)
        const uint64_t NEL = (uint64_t) G.engram_layers.size(), NG = (uint64_t) G.engram_ngram;
        const uint64_t NB = (NG - 1) * (uint64_t) G.engram_heads;
        add("engram_token_map.weight", { (uint64_t) G.vocab }, ANY);
        add("engram_primes.weight", { NB, NEL }, ANY);
        add("engram_offsets.weight", { NB, NEL }, ANY);
        add("engram_multipliers.weight", { NG, NEL }, ANY);
    }
    // every predicted tensor present with its shape/type; every file tensor predicted
    for (const auto & [n, st] : want) {
        size_t s = 0;
        const TensorInfo * t = m.find(n, &s);
        if (!t) return "missing tensor " + n;
        if (t->shape != st.first) {
            std::string a, w;
            for (auto x : t->shape) a += std::to_string(x) + " ";
            for (auto x : st.first) w += std::to_string(x) + " ";
            return n + ": shape [" + a + "] expected [" + w + "]";
        }
        if (st.second != ANY && t->type != st.second)
            return n + ": type " + std::string(ggml_type_name(t->type)) + " expected " + ggml_type_name(st.second);
        if (!m.in_bounds(*t, s)) return n + ": out of bounds";
    }
    for (size_t s = 0; s < m.size(); ++s)
        for (const auto & t : m.shard(s).tensors())
            if (!want.count(t.name)) return "unexpected tensor " + t.name;
    return {};
}

/// The name the engine uses for a file tensor: ours unchanged; the llama.cpp convention's Engram projections mapped to
/// ours (engram_q -> engram_q_norm, engram_k -> engram_k_norm, engram_wkv -> engram_kv; same shapes, model.py's
/// q_norm / k_norm / wkv).
inline std::string ds41_canonical_name(const std::string & n) {
    static const char * from[3] = { ".engram_q.weight", ".engram_k.weight", ".engram_wkv.weight" };
    static const char * to[3] = { ".engram_q_norm.weight", ".engram_k_norm.weight", ".engram_kv.weight" };
    for (int i = 0; i < 3; ++i) {
        const size_t L = std::strlen(from[i]);
        if (n.size() > L && n.compare(n.size() - L, L, from[i]) == 0) return n.substr(0, n.size() - L) + to[i];
    }
    return n;
}

/// The blocks a file holds: `strata.v41.partial_layers` when the writer made a partial test file, else all.
inline std::vector<int64_t> ds41_file_layers(const GgufModel & m, const Ds41Geometry & G) {
    std::vector<int64_t> v;
    if (detail::arr(m.meta(), "strata.v41.partial_layers", v).empty()) return v;
    for (int64_t l = 0; l < G.n_layer; ++l) v.push_back(l);
    return v;
}

}  // namespace strata::v41
