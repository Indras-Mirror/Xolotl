// tools/v41/engine/ernie_geometry.hpp - ERNIE-5.5-Flash (`ernie5_5`) geometry for the CPU dense path.
//
// Every number comes from the GGUF's `ernie5_5.*` keys (tools/ernie/ernie_gguf.py writes them).  A missing key is an
// error, never a default.  The tensor names are llama.cpp PR #31925's (`src/models/ernie5-5.cpp`,
// `conversion/ernie5.py`); the ones this slice reads are listed by ErnieGeometry::tensor_name checks below.
//
// HCA layers (compress ratio 128) are the DeepSeek-V4 layers plus the sigmoid output gate and the VHA head mix;
// ratio-0 layers are MLA+DSA and are NOT part of this slice (they fail cleanly at load).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::v41 {

struct ErnieGeometry {
    static constexpr const char * kArch = "ernie5_5";

    int64_t n_embd = 0, n_layer = 0, n_head = 0, kv_heads = 0;
    int64_t d_head = 0, d_rope = 0, q_lora = 0, kv_lora = 0;
    int64_t o_groups = 0, o_lora = 0;
    int64_t n_swa = 0;
    int64_t n_ff = 0;                 ///< dense FFN width (leading dense block)
    int64_t n_expert = 0, n_used = 0, n_ff_exp = 0, n_latent = 0, n_shared = 0;
    int64_t hc = 0, hc_iters = 0;
    double hc_eps = 0, rms_eps = 0;
    double rope_base = 0;             ///< 8e6 (MLA / indexer; unused by HCA)
    double compress_rope_base = 0;    ///< 160000, the HCA (and compressed row) rotary base
    double rope_factor = 0, beta_fast = 0, beta_slow = 0, orig_ctx = 0;
    double situ_beta = 0, situ_lb = 0;
    int64_t dense_lead = 0;           ///< leading dense FFN blocks (1)
    int64_t vocab = 0;
    std::vector<int64_t> ratio;       ///< [n_layer] 0 = MLA+DSA, 128 = HCA
    double multimax[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // MLA + DSA (compress ratio 0) - read when present; required when a block has ratio 0
    int64_t d_head_mla_k = 0, d_head_mla_v = 0;   ///< 256 (incl. the 64 rope dims) / 256
    int64_t idx_n_head = 0, idx_key_dim = 0, idx_top_k = 0;   ///< 64 / 128 / 2048
    bool has_mla = false;
    int64_t mla_nope() const { return d_head_mla_k - d_rope; }   ///< 192

    bool is_hca(int64_t l) const { return ratio[(size_t) l] != 0; }
    int64_t n_heads_group() const { return n_head / o_groups; }
    int64_t o_group_dim() const { return n_heads_group() * d_head; }

    /// general.architecture of the file (empty on error) - the load-time arch key.
    static std::string arch_of(const GgufModel & m) {
        const MetaValue * a = m.meta().get("general.architecture");
        return a && a->type == MetaType::STRING ? a->s : std::string();
    }

    /// Reads + validates the `ernie5_5.*` block.  False + `err` on a non-ernie5_5 file or a missing/bad key.
    bool load(const GgufModel & m, std::string & err) {
        const GgufFile & g = m.meta();
        const std::string arch = arch_of(m);
        if (arch != kArch) {
            err = "not an " + std::string(kArch) + " GGUF (general.architecture = \"" + arch + "\")";
            return false;
        }
        auto num = [&](const std::string & k, double & v) {
            const MetaValue * p = g.get(k);
            if (!p || !p->is_num()) { err = "missing or non-numeric GGUF key " + k; return false; }
            v = p->num();
            return true;
        };
        auto i64 = [&](const std::string & k, int64_t & v) {
            double d = 0;
            if (!num(k, d)) return false;
            v = (int64_t) llround(d);
            return true;
        };
        auto arr = [&](const std::string & k, std::vector<double> & out) {
            const MetaValue * p = g.get(k);
            if (!p || p->type != MetaType::ARRAY) { err = "missing GGUF array " + k; return false; }
            if (p->count > p->items.size()) { err = "GGUF array too long: " + k; return false; }
            out.clear();
            for (const MetaValue & e : p->items) out.push_back(e.num());
            return true;
        };

        const std::string warm = "ernie5_5.";
        auto K = [&](const char * k) { return warm + k; };

        if (!i64(K("embedding_length"), n_embd) || !i64(K("block_count"), n_layer) ||
            !i64(K("feed_forward_length"), n_ff) || !i64(K("attention.head_count"), n_head) ||
            !i64(K("attention.head_count_kv"), kv_heads) || !i64(K("attention.key_length_swa"), d_head) ||
            !i64(K("rope.dimension_count"), d_rope) || !i64(K("attention.q_lora_rank"), q_lora) ||
            !i64(K("attention.kv_lora_rank"), kv_lora) || !i64(K("attention.sliding_window"), n_swa) ||
            !i64(K("attention.output_group_count"), o_groups) || !i64(K("attention.output_lora_rank"), o_lora) ||
            !i64(K("expert_count"), n_expert) || !i64(K("expert_used_count"), n_used) ||
            !i64(K("expert_feed_forward_length"), n_ff_exp) || !i64(K("expert_latent_length"), n_latent) ||
            !i64(K("expert_shared_count"), n_shared) || !i64(K("hyper_connection.count"), hc) ||
            !i64(K("hyper_connection.sinkhorn_iterations"), hc_iters) || !i64(K("leading_dense_block_count"), dense_lead))
            return false;
        if (!num(K("attention.layer_norm_rms_epsilon"), rms_eps) || !num(K("hyper_connection.epsilon"), hc_eps) ||
            !num(K("attention.compress_rope_freq_base"), compress_rope_base) || !num(K("rope.freq_base"), rope_base) ||
            !num(K("rope.scaling.factor"), rope_factor) ||
            !num(K("rope.scaling.original_context_length"), orig_ctx) ||
            !num(K("rope.scaling.yarn_beta_fast"), beta_fast) ||
            !num(K("rope.scaling.yarn_beta_slow"), beta_slow) ||
            !num(K("activation.situ_beta"), situ_beta) || !num(K("activation.situ_linear_beta"), situ_lb))
            return false;

        std::vector<double> ratios, mm;
        if (!arr(K("attention.compress_ratios"), ratios)) return false;
        if (!arr(K("logit_multimax"), mm)) return false;
        if ((int64_t) ratios.size() < n_layer) { err = "compress_ratios is shorter than block_count"; return false; }
        ratio.clear();
        for (int64_t l = 0; l < n_layer; ++l) {
            const int64_t r = (int64_t) llround(ratios[(size_t) l]);
            if (r != 0 && r != 128) { err = "compress ratio " + std::to_string(r) + " is neither 0 (MLA) nor 128 (HCA)"; return false; }
            ratio.push_back(r);
        }
        for (int64_t r : ratio) has_mla = has_mla || r == 0;
        if (has_mla) {
            if (!i64(K("attention.key_length_mla"), d_head_mla_k) || !i64(K("attention.value_length_mla"), d_head_mla_v) ||
                !i64(K("attention.indexer.head_count"), idx_n_head) ||
                !i64(K("attention.indexer.key_length"), idx_key_dim) || !i64(K("attention.indexer.top_k"), idx_top_k))
                return false;
            if (d_head_mla_k <= d_rope || idx_key_dim < d_rope) { err = "MLA/indexer head smaller than the rope slice"; return false; }
        }
        if (mm.size() != 8) { err = "logit_multimax must hold 8 numbers"; return false; }
        for (int i = 0; i < 8; ++i) multimax[i] = mm[(size_t) i];

        const TensorInfo * emb = m.find("token_embd.weight");
        if (!emb || emb->shape.size() != 2 || (int64_t) emb->shape[0] != n_embd) { err = "token_embd.weight missing or not [n_embd, vocab]"; return false; }
        vocab = (int64_t) emb->shape[1];

        if (hc != 4) { err = "mHC with hyper_connection.count != 4 is not supported"; return false; }
        if (n_head % o_groups != 0) { err = "attention.head_count is not a multiple of output_group_count"; return false; }
        if (kv_heads != 1) { err = "HCA expects one K/V head (attention.head_count_kv = 1)"; return false; }
        if (kv_lora != d_head) { err = "kv_lora_rank != key_length_swa"; return false; }
        if (d_rope > d_head || d_rope % 2) { err = "rope.dimension_count must be even and <= the head"; return false; }
        return true;
    }
};

}  // namespace strata::v41
