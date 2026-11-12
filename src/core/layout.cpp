// src/core/layout.cpp - the shape checks.  See the header for why this exists.
#include "strata/core/layout.hpp"

#include <cstdio>

namespace strata::core {
namespace {

/// A required 2-D shape is `ne0` (the CONTIGUOUS axis, matching the manifest and `s_gemv`'s convention
/// `y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0) times `ne1`.
///
/// A 1-D tensor's ENGINE FORM is not decoration.  The engine form is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena held
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
bool fail(std::string& err, const LayerView& v, const char* suffix, const char* what, int64_t got,
          int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld",
                  (long long) v.layer(), v.name(suffix).c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

bool check_one(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    const LayerView v(t, layer);
    const bool qsa = is_qsa_layer(g, layer);

    // ---- the 2-D tensor set, per layer family
    auto need2 = [&](const char* suffix, int64_t ne0, int64_t ne1) -> bool {
        const WeightRef* r = v.get(suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(suffix);
            return false;
        }
        if (r->ne0 != ne0) return fail(err, v, suffix, "ne0", r->ne0, ne0);
        if (r->ne1 != ne1) return fail(err, v, suffix, "ne1", r->ne1, ne1);
        return true;
    };
    // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim)
    // after its own transpose, so the pack's orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim].
    if (g.has_hyper_connections() &&
        (!need2("hc_attn_down.weight", g.hc_dim(), g.hc_lr) ||
         !need2("hc_attn_up.weight", g.hc_lr, g.hc_dim()) ||
         !need2("hc_attn_inject.weight", g.hc_dim(), g.hc) ||
         !need2("hc_ffn_down.weight", g.hc_dim(), g.hc_lr) ||
         !need2("hc_ffn_up.weight", g.hc_lr, g.hc_dim()) ||
         !need2("hc_ffn_inject.weight", g.hc_dim(), g.hc)))
        return false;
    if (g.has_experts()) {
        // MoE, EVERY layer
        if (!need2("ffn_gate_inp.weight", g.n_embd, g.n_expert) ||
            !need2("ffn_gate_shexp.weight", g.n_embd, g.n_ff) ||
            !need2("ffn_up_shexp.weight", g.n_embd, g.n_ff) ||
            !need2("ffn_down_shexp.weight", g.n_ff, g.n_embd))
            return false;
    } else {
        // Route B: the dense FFN, the same orientation as the shared expert's gate/up/down
        if (!need2("ffn_gate.weight", g.n_embd, g.n_ff) ||
            !need2("ffn_up.weight", g.n_embd, g.n_ff) ||
            !need2("ffn_down.weight", g.n_ff, g.n_embd))
            return false;
    }
    // GDN only
    if (!qsa && (!need2("attn_qkv.weight", g.n_embd, g.ssm_conv_channels) ||
                 !need2("attn_gate.weight", g.n_embd, g.ssm_value_dim) ||
                 !need2("ssm_out.weight", g.ssm_value_dim, g.n_embd) ||
                 !need2("ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels) ||
                 !need2("ssm_alpha.weight", g.n_embd, g.ssm_v_heads) ||
                 !need2("ssm_beta.weight", g.n_embd, g.ssm_v_heads)))
        return false;
    // QSA only
    if (qsa && (!need2("attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim) ||
                !need2("attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim) ||
                !need2("attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim) ||
                !need2("attn_output.weight", g.n_head * g.head_dim, g.n_embd)))
        return false;
    // THE INDEXER.  `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are different
    // numbers - conflating them is what made the planner's indexer term 4x too big in round 194.
    if (qsa && g.has_indexer() &&
        (!need2("indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim) ||
         !need2("indexer.k_proj.weight", g.n_embd, g.idx_key_dim)))
        return false;

    // ---- the 1-D set.  EVERY ONE OF THESE IS `F32` EXCEPT the shared expert's scalar gate, which is BF16 -
    // and that single exception is the one a reader would not guess, so it is written down rather than
    // inferred from the tensor count.
    auto need1 = [&](const char* suffix, int64_t elements, WeightKind kind) -> bool {
        const WeightRef* r = v.get(suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(suffix);
            return false;
        }
        if (r->elements != elements) return fail(err, v, suffix, "elements", r->elements, elements);
        if (r->kind != kind) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) elements * (kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(suffix).c_str(), (int) r->kind,
                          (unsigned long long) r->bytes, (int) kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
        return true;
    };
    if (g.has_hyper_connections()) {
        if (!need1("hc_attn_norm.weight", g.hc_dim(), WeightKind::F32) ||
            !need1("hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32))
            return false;
    } else if (!need1("attn_norm.weight", g.n_embd, WeightKind::F32) ||
               !need1("post_attention_norm.weight", g.n_embd, WeightKind::F32)) {
        return false;
    }
    if (g.has_experts() && !need1("ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32)) return false;
    if (!qsa && (!need1("ssm_a", g.ssm_v_heads, WeightKind::F32) ||
                 !need1("ssm_dt.bias", g.ssm_v_heads, WeightKind::F32) ||
                 !need1("ssm_norm.weight", g.ssm_state_size, WeightKind::F32)))
        return false;
    if (qsa && (!need1("attn_q_norm.weight", g.head_dim, WeightKind::F32) ||
                !need1("attn_k_norm.weight", g.head_dim, WeightKind::F32)))
        return false;
    if (qsa && g.has_indexer() &&
        (!need1("indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32) ||
         !need1("indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32)))
        return false;
    return true;
}

}  // namespace

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    return check_one(table, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_one(table, g, l, err)) return false;
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }
    // The split is derived twice over in `docs/semantics.md` - the interval and an explicit ratio array -
    // and both give 36 GDN and 12 QSA.  Counting them here is the check that the LAYER TYPE PREDICATE and
    // the pack agree, which no per-tensor shape check can see.
    if (n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "layer split is %lld QSA / %lld GDN, the geometry says %lld / %lld",
                      (long long) n_qsa, (long long) n_gdn, (long long) g.n_qsa_layers(),
                      (long long) g.n_gdn_layers());
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
