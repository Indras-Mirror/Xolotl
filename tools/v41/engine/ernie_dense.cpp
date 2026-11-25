// tools/v41/engine/ernie_dense.cpp - ERNIE-5.5-Flash CPU dense path (HCA and MLA + DSA layers).  See ernie_dense.hpp.
//
// Numerics follow the llama.cpp ERNIE 5.5 PR (#31925, src/models/ernie5-5.cpp + deepseek4.cpp: build_hc_pre_ernie,
// build_hca_compressed_kv_from_state, build_hca_attention, build_mla_qk / build_mla_top_k / build_mla_out,
// build_head_mix, build_latent_moe, the sigmoid gate and the grouped low-rank output) and vLLM-ERNIE's attention.py /
// moe.py.  The FFN is the dense SiTU block on the leading dense layer and the latent MoE everywhere else; a layer's
// attention is the HCA path (compress ratio != 0) or the MLA + token-level DSA indexer (ratio 0).

#include "ernie_dense.hpp"

#include "ggml.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::v41 {
namespace {

constexpr int64_t kHcaRatio = 128;   // compress_ratios value of an HCA layer (and its window/pool width)

inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Round-trip through F16: the reference's KV cache element type.
inline float f16_(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

// beta*tanh(g/beta)*sigmoid(g) * beta_l*tanh(u/beta_l)   (llama.cpp LLM_FFN_SITU)
inline float situ(float g, float u, float beta, float beta_l) {
    return beta * std::tanh(g / beta) * sigmoidf_(g) * beta_l * std::tanh(u / beta_l);
}

struct LayerW {
    std::vector<float> attn_norm, ffn_norm;
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    std::vector<float> sinks, vha;
    std::vector<float> q_a, q_a_norm, q_b, kv_w, kv_norm;
    std::vector<float> wo_a, wo_b, o_gate;
    std::vector<float> c_kv, c_gate, c_ape, c_norm;
    // MLA + DSA (compress ratio 0).  The latent K row is [kv_c | k_pe | indexer k]; K absorption and the per-head
    // V expansion use mk_b / mv_b, which the converter splits out of the HF kv_b_proj.
    std::vector<float> mkv_a;      // [kv_lora + d_rope][n_embd]  attn_kv_a_mqa
    std::vector<float> mkv_norm;   // [kv_lora]                  attn_kv_a_norm
    std::vector<float> mk_b;       // [n_head][kv_lora][mla_qk_nope]
    std::vector<float> mv_b;       // [n_head][d_head_mla_v][kv_lora]
    std::vector<float> mwo;        // [n_embd][n_head * d_head_mla_v]
    std::vector<float> idx_k;      // [indexer_key_dim][n_embd]
    std::vector<float> idx_kn, idx_kb;  // [indexer_key_dim]
    std::vector<float> idx_qb;     // [indexer_n_head * indexer_key_dim][r_q]
    std::vector<float> idx_w;      // [indexer_n_head][n_embd]
    std::vector<float> ffn_gate, ffn_up, ffn_down;   // leading dense block only
    // latent MoE (every block past the leading dense one).  The 512 stacked experts are NOT dequantised here: only
    // the top-10 of each token are read from the GGUF, per pass (see load_expert).
    std::vector<float> r_gate;        // [2*n_expert][n_embd], the two sigmoid routers stacked
    std::vector<float> r_bias;        // [n_expert]  exp_probs_b: selection only
    std::vector<float> r_scale;       // [n_expert]  exp_scale: the learnable routed scaling factor
    std::vector<float> r_down, r_up;  // [n_latent][n_embd] / [n_embd][n_latent]
    std::vector<float> r_lat_norm;    // [n_latent]
    std::vector<float> s_gate, s_up, s_down;   // shared expert, on the hidden state
    bool hca = true, mla = false, dense_ffn = false;
    /// indexer.proj is F32 in the converter's output (it is a small tensor); if a file stores it Q8_0 the
    /// activation takes ggml's int8 rounding instead.
    bool idx_w_q8 = false;
    /// Every matrix of the layer is Q8_0, so the matmuls take ggml's Q8_0 activation path (see quantize_q8): the
    /// reference's mul_mat quantises each activation row to int8 blocks, and mimicking that removes the ~1e-3
    /// activation-rounding gap between an exact F32 GEMM and tensors that were built for it.
    bool q8 = true;
    /// The same question for the MoE matmuls (routed down/up, shared expert) and for the stacked routed experts:
    /// Q8_0 takes ggml's int8 activation rounding, anything else runs exact F32 here.
    bool q8_moe = false, q8_exp = false;
};

/// Per-layer persistent attention state: every raw K/V row by position, the compressed rows of completed blocks, and
/// the pending (not yet 128-token) block of compressor projections.
struct LayerCache {
    std::vector<float> raw_k;      ///< [pos][d_head], each row RoPE'd at its position
    std::vector<float> comp_k;     ///< [nblk][d_head], normed + RoPE'd at 128*b
    std::vector<float> pend_kv;    ///< [128][d_head]  (attn_compressor_kv output)
    std::vector<float> pend_gate;  ///< [128][d_head]  (attn_compressor_gate output)
    int64_t pend = 0, pend_base = 0;
    /// MLA layers: one row per position, [kv_lora | d_rope | indexer_key_dim] - the reference's full-context K row
    /// (the attention K/V read its first kv_lora + d_rope entries, the indexer the tail).
    std::vector<float> mla_row;
};

}  // namespace

struct ErnieDense::Impl {
    ErnieDenseConfig cfg;
    ErnieGeometry G;
    std::unique_ptr<GgufModel> model;
    const TensorInfo * emb = nullptr;
    size_t emb_shard = 0;
    std::vector<LayerW> W;
    std::vector<LayerCache> C;
    std::vector<double> inv_freq;   // [d_rope/2], YaRN at the compress base
    std::vector<double> inv_freq_plain;  // [d_rope/2], plain base - MLA + its indexer use rope_theta, not YaRN

    std::vector<float> x;           // [n][hc][n_embd]  (ggml order [n_embd, hc, n])
    int n = 0, pos0 = 0;
    int64_t next_pos = 0;
    int prep_lo = -1, prep_hi = -1;
    std::string err;

    // ---- helpers -------------------------------------------------------
    int n_threads() const { return std::max(1, cfg.n_threads); }

    template <class F>
    void par(int64_t nwork, F fn) const {
        const int nt = (int) std::min<int64_t>(n_threads(), std::max<int64_t>(1, nwork));
        if (nt <= 1) {
            for (int64_t i = 0; i < nwork; ++i) fn(i);
            return;
        }
        std::vector<std::thread> th;
        th.reserve((size_t) nt);
        for (int k = 0; k < nt; ++k) {
            const int64_t a = nwork * k / nt, b = nwork * (k + 1) / nt;
            th.emplace_back([&fn, a, b] { for (int64_t i = a; i < b; ++i) fn(i); });
        }
        for (auto & t : th) t.join();
    }

    /// Any GGML type -> F32 (F32 has no to_float in the type traits table, so it is copied directly).
    static void to_f32(ggml_type ty, const void * src, float * dst, int64_t n) {
        if (ty == GGML_TYPE_F32) { std::memcpy(dst, src, (size_t) n * sizeof(float)); return; }
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        tr->to_float(src, dst, n);
    }

    int tensor_type(const std::string & name) {
        size_t s = 0;
        const TensorInfo * t = model->find(name, &s);
        return t ? (int) t->type : -1;
    }

    bool load_full(const std::string & name, int64_t expect, std::vector<float> & out) {
        size_t s = 0;
        const TensorInfo * t = model->find(name, &s);
        if (!t) { err = "missing tensor " + name; return false; }
        const int64_t ne = (int64_t) t->elements();
        if (expect > 0 && ne != expect) {
            err = "tensor " + name + " has " + std::to_string(ne) + " elements, expected " + std::to_string(expect);
            return false;
        }
        const ggml_type ty = (ggml_type) t->type;
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr || (ty != GGML_TYPE_F32 && !tr->to_float)) {
            err = "tensor " + name + ": unsupported type " + std::string(ggml_type_name(ty));
            return false;
        }
        out.resize((size_t) ne);
        to_f32(ty, model->shard(s).tensor_data(*t), out.data(), ne);
        return true;
    }

    /// Dequantise ONE expert of a stacked `[n_in, n_out, n_expert]` tensor into `out` ([n_out][n_in], the layout
    /// `linear()` takes).  Experts are read in their stored type - the file's Q8_0 stays 32-element blocks on disk,
    /// only the expert a token selected is converted.  moe_ffn drops the touched pages afterwards (POSIX).
    bool load_expert(const std::string & name, int64_t e, int64_t n_in, int64_t n_out, std::vector<float> & out) {
        size_t s = 0;
        const TensorInfo * t = model->find(name, &s);
        if (!t) { err = "missing tensor " + name; return false; }
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != n_out ||
            e < 0 || e >= (int64_t) t->shape[2]) {
            err = "tensor " + name + " is not [" + std::to_string(n_in) + ", " + std::to_string(n_out) + ", n_expert]";
            return false;
        }
        const ggml_type ty = (ggml_type) t->type;
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr || (ty != GGML_TYPE_F32 && !tr->to_float)) {
            err = "tensor " + name + ": unsupported type " + std::string(ggml_type_name(ty));
            return false;
        }
        const size_t blck = ggml_blck_size(ty);
        if ((size_t) n_in % blck) { err = "tensor " + name + ": a row is not a whole number of blocks"; return false; }
        const size_t row_bytes = (size_t) n_in / blck * ggml_type_size(ty);
        const uint8_t * base = model->shard(s).tensor_data(*t) + (size_t) e * (size_t) n_out * row_bytes;
        out.resize((size_t) (n_out * n_in));
        par(n_out, [&](int64_t o) { to_f32(ty, base + (size_t) o * row_bytes, out.data() + (size_t) o * n_in, n_in); });
        return true;
    }

    /// One token-embedding row (dequantised).
    bool emb_row(int64_t tok, float * dst) {
        if (tok < 0 || tok >= G.vocab_size) { err = "token id " + std::to_string(tok) + " outside the vocabulary"; return false; }
        const ggml_type ty = (ggml_type) emb->type;
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr || (ty != GGML_TYPE_F32 && !tr->to_float)) {
            err = "token_embd.weight: unsupported type " + std::string(ggml_type_name(ty));
            return false;
        }
        const size_t blck = ggml_blck_size(ty);
        if ((size_t) G.n_embd % blck) { err = "token_embd row is not a whole number of blocks"; return false; }
        const size_t row_bytes = (size_t) G.n_embd / blck * ggml_type_size(ty);
        const uint8_t * src = model->shard(emb_shard).tensor_data(*emb) + (size_t) tok * row_bytes;
        to_f32(ty, src, dst, G.n_embd);
        return true;
    }

    /// ggml's Q8_0 activation rounding (arch/x86 quantize_row_q8_0): per 32-element block, d = max|x|/127 stored as
    /// F16 and q = round-to-nearest-even(x * (127/max|x|)).  The reciprocal is 127/amax, NOT 1/(amax/127): the two
    /// differ in the last bit and that alone is a 2e-4 difference on a 4096-wide dot.  The reference's mul_mat runs
    /// this on every activation row of a Q8_0 weight.
    static void quantize_q8(const float * x, int64_t n, int8_t * qs, float * d) {
        for (int64_t b = 0; b < n / 32; ++b) {
            const float * xb = x + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
            const float dd = amax / 127.0f;
            const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
            d[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(dd));
            for (int j = 0; j < 32; ++j) qs[b * 32 + j] = (int8_t) nearbyintf(xb[j] * id);
        }
    }

    /// y[t][o] = sum_i x[t][i] * w[o][i]   (w is [out][in] row-major, the GGUF's ne0 = in).  `q8`: round the
    /// activations to Q8_0 first (ggml's path for a Q8_0 weight); the dequantised weight already carries its fp16
    /// block scale, so the result is d_w*d_x*sum(q_w*q_x) - the same product the reference forms.
    void linear(const float * xp, int64_t nt, int64_t in, const float * w, int64_t out, float * y, bool q8) const {
        if (!q8) {
            par(nt * out, [&](int64_t idx) {
                const int64_t t = idx / out, o = idx % out;
                const float * xr = xp + t * in;
                const float * wr = w + o * in;
                double s = 0.0;
                for (int64_t i = 0; i < in; ++i) s += (double) xr[i] * (double) wr[i];
                y[idx] = (float) s;
            });
            return;
        }
        const int64_t nb = in / 32;
        std::vector<int8_t> qs((size_t) (nt * in));
        std::vector<float> dx((size_t) (nt * nb));
        par(nt, [&](int64_t t) { quantize_q8(xp + t * in, in, qs.data() + t * in, dx.data() + t * nb); });
        par(nt * out, [&](int64_t idx) {
            const int64_t t = idx / out, o = idx % out;
            const float * wr = w + o * in;
            const int8_t * qr = qs.data() + t * in;
            const float * dr = dx.data() + t * nb;
            double s = 0.0;
            for (int64_t b = 0; b < nb; ++b) {
                double sb = 0.0;
                for (int64_t j = 0; j < 32; ++j) sb += (double) wr[b * 32 + j] * (double) qr[b * 32 + j];
                s += sb * (double) dr[b];
            }
            y[idx] = (float) s;
        });
    }

    /// y[t][:] = x[t][:] * w / sqrt(mean(x^2)+eps)   (w may be null: the unweighted per-head q norm)
    void rmsnorm(const float * xp, const float * w, int64_t nt, int64_t d, float * y) const {
        const float eps = (float) G.rms_eps;
        par(nt, [&](int64_t t) {
            const float * xr = xp + t * d;
            float * yr = y + t * d;
            double s = 0.0;
            for (int64_t i = 0; i < d; ++i) s += (double) xr[i] * (double) xr[i];
            const float inv = 1.0f / std::sqrt((float) (s / (double) d) + eps);
            if (w) for (int64_t i = 0; i < d; ++i) yr[i] = xr[i] * inv * w[i];
            else   for (int64_t i = 0; i < d; ++i) yr[i] = xr[i] * inv;
        });
    }

    /// YaRN inv_freq for the HCA rotary (base 160000, factor 16, orig ctx 65536, beta 32/1) - ggml's rope_yarn
    /// with attn_factor cancelling the mscale, so cos/sin are unscaled and only the theta mix differs from plain RoPE.
    void build_inv_freq() {
        const int64_t rd = G.d_rope, half = rd / 2;
        const double base = G.compress_rope_base, factor = G.rope_factor, orig = G.rope_orig_ctx;
        const double bf = G.yarn_beta_fast, bs = G.yarn_beta_slow;
        constexpr double kPi = 3.14159265358979323846;
        auto corr_dim = [&](double rotations) {
            return (double) rd * std::log(orig / (rotations * 2.0 * kPi)) / (2.0 * std::log(base));
        };
        const int64_t low = (int64_t) std::max(0.0, std::floor(corr_dim(bf)));
        const int64_t high = (int64_t) std::min((double) rd - 1.0, std::ceil(corr_dim(bs)));
        const double span = std::max(0.001, (double) (high - low));
        inv_freq.assign((size_t) half, 0.0);
        for (int64_t i = 0; i < half; ++i) {
            const double pf = std::pow(base, (double) (2 * i) / (double) rd);
            const double extrap = 1.0 / pf;
            const double interp = 1.0 / (factor * pf);
            const double lin = std::min(1.0, std::max(0.0, ((double) i - (double) low) / span));
            const double mask = 1.0 - lin;   // 1 = extrapolation (high frequency), 0 = interpolation
            inv_freq[(size_t) i] = interp * (1.0 - mask) + extrap * mask;
        }
    }

    /// Plain (non-YaRN) inv_freq at rope_theta: the DSA layer's QK rope and its indexer's NEOX rope both use it.
    void build_inv_freq_plain() {
        const int64_t half = G.d_rope / 2;
        inv_freq_plain.assign((size_t) half, 0.0);
        for (int64_t i = 0; i < half; ++i)
            inv_freq_plain[(size_t) i] = 1.0 / std::pow(G.rope_freq_base, (double) (2 * i) / (double) G.d_rope);
    }

    /// GPT-J / interleaved partial RoPE on the last d_rope channels of a d_head row (ggml ROPE_TYPE_NORMAL with
    /// n_offs = d_head - d_rope).
    void rope(float * v, int64_t d_head, int64_t pos, bool inverse) const {
        const int64_t rd = G.d_rope, off = d_head - rd;
        for (int64_t i = 0; i < rd / 2; ++i) {
            const double th = (double) pos * inv_freq[(size_t) i];
            const float c = (float) std::cos(th);
            const float s = (float) (inverse ? -std::sin(th) : std::sin(th));
            const float a = v[off + 2 * i], b = v[off + 2 * i + 1];
            v[off + 2 * i] = a * c - b * s;
            v[off + 2 * i + 1] = a * s + b * c;
        }
    }

    /// ggml ROPE_TYPE_NORMAL over the first d_rope values of `v` at the plain base (the MLA q_pe / k_pe slices).
    void rope_norm_slice(float * v, int64_t pos, bool inverse) const {
        const int64_t rd = G.d_rope;
        for (int64_t i = 0; i < rd / 2; ++i) {
            const double th = (double) pos * inv_freq_plain[(size_t) i];
            const float c = (float) std::cos(th);
            const float s = (float) (inverse ? -std::sin(th) : std::sin(th));
            const float a = v[2 * i], b = v[2 * i + 1];
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = a * s + b * c;
        }
    }

    /// ggml ROPE_TYPE_NEOX over the first d_rope values of a row of `d` (the indexer's q/k: half-split rotation).
    void rope_neox(float * v, int64_t pos, bool inverse) const {
        const int64_t half = G.d_rope / 2;
        for (int64_t i = 0; i < half; ++i) {
            const double th = (double) pos * inv_freq_plain[(size_t) i];
            const float c = (float) std::cos(th);
            const float s = (float) (inverse ? -std::sin(th) : std::sin(th));
            const float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    /// LayerNorm (mean/variance over `d`, weight and bias) - the indexer's k_norm.  torch semantics: the biased
    /// variance and eps inside the square root.
    void layernorm(const float * xp, const float * w, const float * b, int64_t nt, int64_t d, float eps,
                   float * y) const {
        par(nt, [&](int64_t t) {
            const float * xr = xp + t * d;
            float * yr = y + t * d;
            double mean = 0.0;
            for (int64_t i = 0; i < d; ++i) mean += (double) xr[i];
            mean /= (double) d;
            double var = 0.0;
            for (int64_t i = 0; i < d; ++i) { const double c = (double) xr[i] - mean; var += c * c; }
            var /= (double) d;
            const float inv = 1.0f / std::sqrt((float) var + eps);
            for (int64_t i = 0; i < d; ++i) yr[i] = (float) (((double) xr[i] - mean) * (double) inv) * w[i] + b[i];
        });
    }

    /// mHC pre on the 4 streams.  pre/post are [nt, hc], comb is [nt][src][dst].  ERNIE's pre-RMS is
    /// 1/(sqrt(mean(x^2)) + eps); everything else is the DS4.1 build_hc_split_mixes structure.
    void hc_pre(const float * xp, int64_t nt, const std::vector<float> & hc_fn, const std::vector<float> & hc_base,
                const std::vector<float> & hc_scale, float * collapsed, float * pre, float * post, float * comb) const {
        const int64_t D = G.n_embd, HC = G.hc, hc_dim = HC * D, mix_dim = (2 + HC) * HC;
        const double hc_eps = G.hc_eps;
        par(nt, [&](int64_t t) {
            const float * flat = xp + t * hc_dim;
            double ss = 0.0;
            for (int64_t i = 0; i < hc_dim; ++i) ss += (double) flat[i] * (double) flat[i];
            const float rms = (float) std::sqrt(ss / (double) hc_dim) + (float) hc_eps;
            float mixes[24];
            for (int64_t j = 0; j < mix_dim; ++j) {
                const float * wr = hc_fn.data() + j * hc_dim;
                double s = 0.0;
                for (int64_t i = 0; i < hc_dim; ++i) s += (double) flat[i] * (double) wr[i];
                mixes[j] = (float) (s / (double) rms);
            }
            for (int64_t s = 0; s < HC; ++s) {
                pre[t * HC + s] = sigmoidf_(mixes[s] * hc_scale[0] + hc_base[s]) + (float) hc_eps;
                post[t * HC + s] = sigmoidf_(mixes[HC + s] * hc_scale[1] + hc_base[HC + s]) * 2.0f;
            }
            // comb[src][dst]: a row softmax over dst, one column normalisation, then 19 (row, column) passes
            float * cb = comb + t * HC * HC;
            for (int64_t src = 0; src < HC; ++src) {
                float * r = cb + src * HC;
                float mx = -INFINITY;
                for (int64_t dst = 0; dst < HC; ++dst) {
                    r[dst] = mixes[2 * HC + src * HC + dst] * hc_scale[2] + hc_base[2 * HC + src * HC + dst];
                    mx = std::max(mx, r[dst]);
                }
                float den = 0.0f;
                for (int64_t dst = 0; dst < HC; ++dst) { r[dst] = std::exp(r[dst] - mx); den += r[dst]; }
                for (int64_t dst = 0; dst < HC; ++dst) r[dst] = r[dst] / den + (float) hc_eps;
            }
            auto norm_cols = [&]() {
                for (int64_t dst = 0; dst < HC; ++dst) {
                    float s = (float) hc_eps;
                    for (int64_t src = 0; src < HC; ++src) s += cb[src * HC + dst];
                    for (int64_t src = 0; src < HC; ++src) cb[src * HC + dst] /= s;
                }
            };
            auto norm_rows = [&]() {
                for (int64_t src = 0; src < HC; ++src) {
                    float s = (float) hc_eps;
                    for (int64_t dst = 0; dst < HC; ++dst) s += cb[src * HC + dst];
                    for (int64_t dst = 0; dst < HC; ++dst) cb[src * HC + dst] /= s;
                }
            };
            norm_cols();
            for (int64_t it = 1; it < G.hc_sinkhorn_iters; ++it) { norm_rows(); norm_cols(); }

            float * cr = collapsed + t * D;
            for (int64_t d = 0; d < D; ++d) cr[d] = 0.0f;
            for (int64_t s = 0; s < HC; ++s) {
                const float p = pre[t * HC + s];
                const float * xs = xp + t * hc_dim + s * D;
                for (int64_t d = 0; d < D; ++d) cr[d] += p * xs[d];
            }
        });
    }

    /// x[dst] = post[dst]*cur + sum_src comb[src][dst]*residual[src]
    void hc_post(const float * cur, const float * residual, const float * post, const float * comb, int64_t nt,
                 float * out) const {
        const int64_t D = G.n_embd, HC = G.hc, hc_dim = HC * D;
        par(nt, [&](int64_t t) {
            const float * rt = residual + t * hc_dim;
            const float * ct = cur + t * D;
            const float * pm = post + t * HC;
            const float * cb = comb + t * HC * HC;
            float * ot = out + t * hc_dim;
            for (int64_t dst = 0; dst < HC; ++dst) {
                float * od = ot + dst * D;
                for (int64_t d = 0; d < D; ++d) od[d] = pm[dst] * ct[d];
                for (int64_t src = 0; src < HC; ++src) {
                    const float c = cb[src * HC + dst];
                    const float * rs = rt + src * D;
                    for (int64_t d = 0; d < D; ++d) od[d] += c * rs[d];
                }
            }
        });
    }

    bool run_layer(int il, std::vector<ErnieTap> * taps);
    bool attn_hca(int il, const float * cur, int64_t nt, std::vector<ErnieTap> * taps, float * out);
    bool attn_mla(int il, const float * cur, int64_t nt, std::vector<ErnieTap> * taps, float * out);
    bool moe_ffn(int il, const float * fcur, int64_t nt, std::vector<ErnieTap> * taps, float * out);
    bool check_ready(int lo, int hi);
    void tap_push(std::vector<ErnieTap> * taps, const std::string & name, std::vector<int64_t> ne,
                  const float * data, int64_t count) const;
};

void ErnieDense::Impl::tap_push(std::vector<ErnieTap> * taps, const std::string & name, std::vector<int64_t> ne,
                                const float * data, int64_t count) const {
    if (!taps) return;
    while (ne.size() < 4) ne.push_back(1);
    ErnieTap t;
    t.name = name;
    t.ne = ne;
    t.data.assign(data, data + count);
    taps->push_back(std::move(t));
}

bool ErnieDense::Impl::check_ready(int lo, int hi) {
    if (prep_lo > lo || prep_hi < hi) {
        err = "prepare(layers " + std::to_string(lo) + ".." + std::to_string(hi) + ") was not called";
        return false;
    }
    if (n <= 0) { err = "no pass: call begin() first"; return false; }
    return true;
}

/// The latent MoE of one layer, from the FFN-normed hidden `fcur` ([nt][n_embd]) to `out` ([nt][n_embd]).
///
/// Mirrors llama.cpp PR #31925's `build_latent_moe` and vLLM-ERNIE's `Ernie5_5MoE.forward`:
///   scores  = sigmoid(router0 . x) + sigmoid(router1 . x)              (split-feature routing)
///   select  = top-10 of scores + exp_probs_b                            (the bias never enters the weights)
///   weight  = scores[e] * exp_scale[e] / clamp(sum of the 10 selected scores, 1e-20)
///   routed  = fc1_latent_proj(x); the 10 experts run in the latent, SiTU-GLU, weighted sum
///   out     = fc2_latent_proj(latent_norm(routed)) + shared_expert(x)
/// The experts read the GGUF in their stored type, one selected expert at a time.
bool ErnieDense::Impl::moe_ffn(int il, const float * fcur, int64_t nt, std::vector<ErnieTap> * taps, float * out) {
    const int64_t D = G.n_embd, LAT = G.n_embd_lat, FE = G.n_ff_exp, NE = G.n_expert, TK = G.n_expert_used;
    const int64_t SE = FE * G.n_expert_shared;   // the shared expert's (single, wider) FFN
    const LayerW & w = W[(size_t) il];
    const std::string b = "blk." + std::to_string(il) + ".";

    // ---- split-feature router: the two [NE, D] matrices are stacked in ffn_gate_inp
    std::vector<float> lg((size_t) (nt * 2 * NE));
    linear(fcur, nt, D, w.r_gate.data(), 2 * NE, lg.data(), false);   // F32 router: no activation rounding
    std::vector<float> probs((size_t) (nt * NE));
    par(nt, [&](int64_t t) {
        const float * l0 = lg.data() + t * 2 * NE;
        const float * l1 = l0 + NE;
        float * p = probs.data() + t * NE;
        for (int64_t e = 0; e < NE; ++e) p[e] = sigmoidf_(l0[e]) + sigmoidf_(l1[e]);
    });
    tap_push(taps, "ffn_moe_probs_split-" + std::to_string(il), {NE, nt}, probs.data(), nt * NE);

    // ---- selection (bias-corrected) and weights (plain scores, renormalised, per-expert scale)
    std::vector<int32_t> sel((size_t) (nt * TK));
    std::vector<float> wts((size_t) (nt * TK));
    std::vector<float> pfinal((size_t) (taps ? nt * NE : 0));
    par(nt, [&](int64_t t) {
        const float * p = probs.data() + t * NE;
        std::vector<int32_t> ord((size_t) NE);
        for (int64_t e = 0; e < NE; ++e) ord[(size_t) e] = (int32_t) e;
        auto better = [&](int32_t a, int32_t c) {
            const float sa = p[a] + w.r_bias[(size_t) a], sc = p[c] + w.r_bias[(size_t) c];
            return sa > sc || (sa == sc && a < c);   // ggml's argsort is descending; ties are unspecified there
        };
        std::partial_sort(ord.begin(), ord.begin() + TK, ord.end(), better);
        float sum = 0.0f;
        for (int64_t k = 0; k < TK; ++k) sum += p[ord[(size_t) k]];
        if (sum < 1e-20f) sum = 1e-20f;   // ggml_clamp(w_sum, 1e-20, INF)
        for (int64_t k = 0; k < TK; ++k) {
            const int32_t e = ord[(size_t) k];
            sel[(size_t) (t * TK + k)] = e;
            wts[(size_t) (t * TK + k)] = p[e] * w.r_scale[(size_t) e] / sum;
        }
        if (!pfinal.empty())
            for (int64_t e = 0; e < NE; ++e) pfinal[(size_t) (t * NE + e)] = p[e] * w.r_scale[(size_t) e] / sum;
    });
    if (taps) {
        // The PR names these twice - "ffn_moe_topk_ernie"/"ffn_moe_probs_final" in build_latent_moe, then
        // "ffn_moe_topk"/"ffn_moe_probs" by build_moe_ffn for the same tensors - and the later name is the one its
        // dumper writes, so these taps carry that name.
        const std::string n = std::to_string(il);
        std::vector<float> ids((size_t) (nt * TK));   // I32 in the reference, dumped as floats
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (float) sel[i];
        tap_push(taps, "ffn_moe_topk-" + n, {TK, nt}, ids.data(), nt * TK);
        tap_push(taps, "ffn_moe_probs-" + n, {NE, nt}, pfinal.data(), nt * NE);
        tap_push(taps, "ffn_moe_weights-" + n, {1, TK, nt}, wts.data(), nt * TK);
    }

    // ---- fc1_latent_proj: the experts work in the latent space
    std::vector<float> routed((size_t) (nt * LAT));
    linear(fcur, nt, D, w.r_down.data(), LAT, routed.data(), w.q8_moe);
    tap_push(taps, "ffn_routed_down-" + std::to_string(il), {LAT, nt}, routed.data(), nt * LAT);

    // ---- the routed experts, grouped by expert so each selected expert is read once per pass
    struct Slot {
        int32_t t, slot;
        float w;
    };
    std::vector<std::vector<Slot>> by_e((size_t) NE);
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t k = 0; k < TK; ++k)
            by_e[(size_t) sel[(size_t) (t * TK + k)]].push_back(
                    {(int32_t) t, (int32_t) k, wts[(size_t) (t * TK + k)]});
    const bool want_ex = taps != nullptr;
    std::vector<float> tg, tu, ts, td;   // the expert-level taps: [FE, TK, nt] (gate/up/situ), [LAT, TK, nt] (down)
    if (want_ex) {
        tg.assign((size_t) (FE * TK * nt), 0.0f);
        tu.assign((size_t) (FE * TK * nt), 0.0f);
        ts.assign((size_t) (FE * TK * nt), 0.0f);
        td.assign((size_t) (LAT * TK * nt), 0.0f);
    }
    std::vector<float> wg, wu, wd, xg, yg, yu, ys, yd;
    std::vector<float> acc((size_t) (nt * LAT), 0.0f);
    for (int64_t e = 0; e < NE; ++e) {
        const std::vector<Slot> & items = by_e[(size_t) e];
        if (items.empty()) continue;
        const int64_t ng = (int64_t) items.size();
        xg.resize((size_t) (ng * LAT));
        for (int64_t i = 0; i < ng; ++i)
            std::memcpy(xg.data() + i * LAT, routed.data() + (size_t) items[(size_t) i].t * LAT, (size_t) LAT * sizeof(float));
        if (!load_expert(b + "ffn_gate_exps.weight", e, LAT, FE, wg) ||
            !load_expert(b + "ffn_up_exps.weight", e, LAT, FE, wu))
            return false;
        yg.resize((size_t) (ng * FE));
        yu.resize((size_t) (ng * FE));
        ys.resize((size_t) (ng * FE));
        linear(xg.data(), ng, LAT, wg.data(), FE, yg.data(), w.q8_exp);
        linear(xg.data(), ng, LAT, wu.data(), FE, yu.data(), w.q8_exp);
        for (int64_t i = 0; i < ng * FE; ++i)
            ys[(size_t) i] = situ(yg[(size_t) i], yu[(size_t) i], (float) G.situ_beta, (float) G.situ_linear_beta);
        if (!load_expert(b + "ffn_down_exps.weight", e, FE, LAT, wd)) return false;
        yd.resize((size_t) (ng * LAT));
        linear(ys.data(), ng, FE, wd.data(), LAT, yd.data(), w.q8_exp);
        for (int64_t i = 0; i < ng; ++i) {
            const Slot & it = items[(size_t) i];
            float * a = acc.data() + (size_t) it.t * LAT;
            const float * y = yd.data() + i * LAT;
            for (int64_t j = 0; j < LAT; ++j) a[j] += it.w * y[j];
            if (want_ex) {
                // taps in the reference's ggml order: [fe | latent, top_k, n_tokens]
                const size_t o = (size_t) (it.slot + TK * it.t);
                std::memcpy(tg.data() + o * FE, yg.data() + (size_t) i * FE, (size_t) FE * sizeof(float));
                std::memcpy(tu.data() + o * FE, yu.data() + (size_t) i * FE, (size_t) FE * sizeof(float));
                std::memcpy(ts.data() + o * FE, ys.data() + (size_t) i * FE, (size_t) FE * sizeof(float));
                std::memcpy(td.data() + o * LAT, y, (size_t) LAT * sizeof(float));
            }
        }
    }
    // A prefill-sized pass touches most of the layer's 6.6 GiB of experts.  They are clean, file-backed pages of a
    // MAP_PRIVATE view: dropping them from this process's page tables keeps the runner's resident set down (the
    // kernel keeps them in the page cache, so a later pass re-faults them without disk I/O).
#ifndef _WIN32
    for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
        size_t s = 0;
        const TensorInfo * t = model->find(b + n, &s);
        if (!t) continue;
        const uint64_t bytes = tensor_payload_bytes(*t);
        if (!bytes) continue;
        const uintptr_t p = (uintptr_t) model->shard(s).tensor_data(*t);
        const uintptr_t page = 4096;
        ::madvise((void *) (p & ~(page - 1)), (size_t) (bytes + (p & (page - 1))), MADV_DONTNEED);
    }
#endif
    tap_push(taps, "ffn_moe_out-" + std::to_string(il), {LAT, nt}, acc.data(), nt * LAT);
    if (want_ex) {
        const std::string n = std::to_string(il);
        tap_push(taps, "ffn_moe_gate-" + n, {FE, TK, nt}, tg.data(), FE * TK * nt);
        tap_push(taps, "ffn_moe_up-" + n, {FE, TK, nt}, tu.data(), FE * TK * nt);
        tap_push(taps, "ffn_moe_situ-" + n, {FE, TK, nt}, ts.data(), FE * TK * nt);
        tap_push(taps, "ffn_moe_down-" + n, {LAT, TK, nt}, td.data(), LAT * TK * nt);
    }

    // ---- latent_norm + fc2_latent_proj
    std::vector<float> ln((size_t) (nt * LAT));
    rmsnorm(acc.data(), w.r_lat_norm.data(), nt, LAT, ln.data());
    std::vector<float> rup((size_t) (nt * D));
    linear(ln.data(), nt, LAT, w.r_up.data(), D, rup.data(), w.q8_moe);
    tap_push(taps, "ffn_routed_up-" + std::to_string(il), {D, nt}, rup.data(), nt * D);

    // ---- the shared expert reads the original hidden state
    std::vector<float> sg((size_t) (nt * SE)), su((size_t) (nt * SE)), sx((size_t) (nt * SE));
    linear(fcur, nt, D, w.s_gate.data(), SE, sg.data(), w.q8_moe);
    linear(fcur, nt, D, w.s_up.data(), SE, su.data(), w.q8_moe);
    for (int64_t i = 0; i < nt * SE; ++i)
        sx[(size_t) i] = situ(sg[(size_t) i], su[(size_t) i], (float) G.situ_beta, (float) G.situ_linear_beta);
    std::vector<float> sh((size_t) (nt * D));
    linear(sx.data(), nt, SE, w.s_down.data(), D, sh.data(), w.q8_moe);
    tap_push(taps, "ffn_shexp-" + std::to_string(il), {D, nt}, sh.data(), nt * D);

    for (int64_t i = 0; i < nt * D; ++i) out[i] = rup[(size_t) i] + sh[(size_t) i];
    return true;
}

/// The MLA + DSA attention of one layer (`compress_ratios[i] == 0`), from the attention-normed `cur` ([nt][n_embd])
/// to `out` ([nt][n_embd]).
///
/// Mirrors llama.cpp PR #31925's `build_mla_qk` / `build_mla_top_k` / `build_mla_out` (ernie5-5.cpp) and vLLM-ERNIE's
/// `Ernie5_5DSAAttention` + `Ernie5_5Indexer`:
///
///   q_c   = q_a x, RMS(q_a_norm);  gate = sigmoid(o_gate q_c)
///   q     = q_b q_c = [q_nope 192 | q_pe 64] per head, q_pe RoPE'd at rope_theta (plain, no YaRN)
///   kv    = kv_a_mqa x = [kv_c 512 | k_pe 64];  kv_c RMS(kv_a_norm), k_pe RoPE'd
///   cache row = [kv_c | k_pe | indexer k 128]   (F16, one row per position - the reference's full-context half)
///   Q     = [k_b[h] q_nope | q_pe]  (K absorption; attention score scale 1/sqrt(192 + 64))
///   V     = kv_c, expanded per head by v_b[h] after attention
///   index indexer score(t, j) = sum_h relu(q_idx[t,h] . k_idx[j]) * w[t,h]   (NEOX RoPE, 1/sqrt(128*64) on w)
///   top   = min(n_kv, 2048 + 128) of score, with the 128-cell window forced to +inf (vLLM's merge_swa_topk set)
///   attn  softmax over the selected cells + the learned per-head sink, then VHA head mix, sigmoid gate, wo
///
/// The K rows are stored once and read by both the attention and the indexer, so chunked prefill and 1-token decode
/// go through the same arithmetic.
bool ErnieDense::Impl::attn_mla(int il, const float * cur, int64_t nt, std::vector<ErnieTap> * taps, float * out) {
    const int64_t D = G.n_embd, NH = G.n_head, QL = G.r_q;
    const int64_t KL = G.kv_lora, DR = G.d_rope, QKH = G.d_head_mla_k, NOPE = G.mla_qk_nope(), DV = G.d_head_mla_v;
    const int64_t ID = G.indexer_key_dim, IH = G.indexer_n_head, TOPK = G.indexer_top_k;
    const int64_t KROW = KL + DR, ROW = KROW + ID;
    const int64_t OJ = NH * DV;
    const int64_t nkv = (int64_t) pos0 + nt;
    const int64_t n_top = std::min<int64_t>(nkv, TOPK + G.n_swa);
    const float att_scale = 1.0f / std::sqrt((float) QKH);
    const LayerW & w = W[(size_t) il];
    LayerCache & c = C[(size_t) il];
    const std::string n = std::to_string(il);

    // ---- q latent, gate, q -------------------------------------------------
    std::vector<float> qc((size_t) (nt * QL));
    linear(cur, nt, D, w.q_a.data(), QL, qc.data(), w.q8);
    tap_push(taps, "q_c-" + n, {QL, nt}, qc.data(), nt * QL);
    std::vector<float> qcn((size_t) (nt * QL));
    rmsnorm(qc.data(), w.q_a_norm.data(), nt, QL, qcn.data());

    std::vector<float> gate((size_t) (nt * OJ));
    linear(qcn.data(), nt, QL, w.o_gate.data(), OJ, gate.data(), w.q8);
    for (float & g : gate) g = sigmoidf_(g);
    tap_push(taps, "attn_gate-" + n, {OJ, nt}, gate.data(), nt * OJ);

    std::vector<float> q((size_t) (nt * NH * QKH));
    linear(qcn.data(), nt, QL, w.q_b.data(), NH * QKH, q.data(), w.q8);
    par(nt * NH, [&](int64_t i) { rope_norm_slice(&q[(size_t) (i * QKH + NOPE)], pos0 + i / NH, false); });
    {
        std::vector<float> qpe((size_t) (nt * NH * DR));
        for (int64_t t = 0; t < nt; ++t)
            for (int64_t h = 0; h < NH; ++h)
                std::memcpy(&qpe[(size_t) ((t * NH + h) * DR)], &q[(size_t) ((t * NH + h) * QKH + NOPE)],
                            (size_t) DR * sizeof(float));
        tap_push(taps, "q_pe-" + n, {DR, NH, nt}, qpe.data(), nt * NH * DR);
    }

    // ---- kv latent ---------------------------------------------------------
    std::vector<float> kvpe((size_t) (nt * KROW));
    linear(cur, nt, D, w.mkv_a.data(), KROW, kvpe.data(), w.q8);
    tap_push(taps, "kv_cmpr_pe-" + n, {KROW, nt}, kvpe.data(), nt * KROW);
    std::vector<float> kvc((size_t) (nt * KL));
    for (int64_t t = 0; t < nt; ++t)
        std::memcpy(&kvc[(size_t) (t * KL)], &kvpe[(size_t) (t * KROW)], (size_t) KL * sizeof(float));
    rmsnorm(kvc.data(), w.mkv_norm.data(), nt, KL, kvc.data());
    tap_push(taps, "kv_cmpr-" + n, {KL, nt}, kvc.data(), nt * KL);
    std::vector<float> kpe((size_t) (nt * DR));
    for (int64_t t = 0; t < nt; ++t)
        std::memcpy(&kpe[(size_t) (t * DR)], &kvpe[(size_t) (t * KROW + KL)], (size_t) DR * sizeof(float));
    par(nt, [&](int64_t t) { rope_norm_slice(&kpe[(size_t) (t * DR)], pos0 + t, false); });
    tap_push(taps, "k_pe-" + n, {DR, 1, nt}, kpe.data(), nt * DR);

    // ---- K absorption: q_abs[h][l] = sum_nope k_b[h][l][nope] * q_nope[h][nope] ----
    std::vector<float> Qcur((size_t) (nt * NH * KROW));
    par(nt * NH, [&](int64_t i) {
        const int64_t h = i % NH;
        const float * qn = q.data() + (size_t) (i * QKH);
        const float * wb = w.mk_b.data() + (size_t) (h * KL * NOPE);
        float * dst = Qcur.data() + (size_t) (i * KROW);
        // k_b is Q8_0, so the reference's mul_mat rounds the q_nope activation to int8 blocks (like linear())
        std::vector<int8_t> qs((size_t) NOPE);
        std::vector<float> dx((size_t) (NOPE / 32));
        if (w.q8) quantize_q8(qn, NOPE, qs.data(), dx.data());
        for (int64_t l = 0; l < KL; ++l) {
            const float * wr = wb + (size_t) (l * NOPE);
            double s = 0.0;
            if (w.q8) {
                for (int64_t b = 0; b < NOPE / 32; ++b) {
                    double sb = 0.0;
                    for (int64_t j = 0; j < 32; ++j) sb += (double) wr[b * 32 + j] * (double) qs[(size_t) (b * 32 + j)];
                    s += sb * (double) dx[(size_t) b];
                }
            } else {
                for (int64_t p = 0; p < NOPE; ++p) s += (double) wr[p] * (double) qn[p];
            }
            dst[l] = (float) s;
        }
        std::memcpy(dst + KL, qn + NOPE, (size_t) DR * sizeof(float));
    });
    {
        std::vector<float> qabs((size_t) (nt * NH * KL));
        for (int64_t i = 0; i < nt * NH; ++i)
            std::memcpy(&qabs[(size_t) (i * KL)], &Qcur[(size_t) (i * KROW)], (size_t) KL * sizeof(float));
        tap_push(taps, "q_nope_absorbed-" + n, {KL, NH, nt}, qabs.data(), nt * NH * KL);
    }
    tap_push(taps, "Qcur-" + n, {KROW, NH, nt}, Qcur.data(), nt * NH * KROW);

    // ---- indexer key: LayerNorm (weight + bias, eps 1e-6), NEOX RoPE on the leading d_rope ----
    std::vector<float> idxk((size_t) (nt * ID));
    linear(cur, nt, D, w.idx_k.data(), ID, idxk.data(), w.q8);
    layernorm(idxk.data(), w.idx_kn.data(), w.idx_kb.data(), nt, ID, 1e-6f, idxk.data());
    par(nt, [&](int64_t t) { rope_neox(&idxk[(size_t) (t * ID)], pos0 + t, false); });
    tap_push(taps, "indexer_k-" + n, {ID, 1, nt}, idxk.data(), nt * ID);

    // ---- store the cache row [kv_c | k_pe | indexer k] at each position -----
    if ((int64_t) c.mla_row.size() < nkv * ROW) c.mla_row.resize((size_t) (nkv * ROW));
    std::vector<float> kcur((size_t) (nt * ROW));
    for (int64_t t = 0; t < nt; ++t) {
        float * row = &kcur[(size_t) (t * ROW)];
        std::memcpy(row, &kvc[(size_t) (t * KL)], (size_t) KL * sizeof(float));
        std::memcpy(row + KL, &kpe[(size_t) (t * DR)], (size_t) DR * sizeof(float));
        std::memcpy(row + KROW, &idxk[(size_t) (t * ID)], (size_t) ID * sizeof(float));
        float * cache = c.mla_row.data() + (size_t) (((int64_t) pos0 + t) * ROW);
        std::memcpy(cache, row, (size_t) ROW * sizeof(float));
        if (cfg.kv_fp16) for (int64_t d = 0; d < ROW; ++d) cache[d] = f16_(cache[d]);
    }
    tap_push(taps, "Kcur-" + n, {ROW, 1, nt}, kcur.data(), nt * ROW);
    {
        std::vector<float> ikc((size_t) (nkv * ID));   // the cache tail, as the indexer reads it
        for (int64_t j = 0; j < nkv; ++j)
            std::memcpy(&ikc[(size_t) (j * ID)], c.mla_row.data() + (size_t) (j * ROW + KROW),
                        (size_t) ID * sizeof(float));
        tap_push(taps, "indexer_k_cache-" + n, {ID, 1, nkv}, ikc.data(), nkv * ID);
    }

    // ---- indexer query and head weights ------------------------------------
    std::vector<float> idxq((size_t) (nt * IH * ID));
    linear(qcn.data(), nt, QL, w.idx_qb.data(), IH * ID, idxq.data(), w.q8);
    par(nt * IH, [&](int64_t i) { rope_neox(&idxq[(size_t) (i * ID)], pos0 + i / IH, false); });
    tap_push(taps, "indexer_q-" + n, {ID, IH, nt}, idxq.data(), nt * IH * ID);

    std::vector<float> idxw((size_t) (nt * IH));
    linear(cur, nt, D, w.idx_w.data(), IH, idxw.data(), w.idx_w_q8);
    {
        const float ws = 1.0f / std::sqrt((float) (ID * IH));
        for (float & v : idxw) v *= ws;
    }
    tap_push(taps, "indexer_weights-" + n, {IH, nt}, idxw.data(), nt * IH);

    // ---- scores, top-k over the causal history + the window, sparse attention ----
    std::vector<float> score((size_t) (nt * nkv));
    std::vector<int32_t> sel((size_t) (nt * n_top), -1);   // -1 pads tokens with fewer causal cells than n_top
    std::vector<int64_t> nsel((size_t) nt, 0);
    par(nt, [&](int64_t t) {
        const int64_t p = pos0 + t;
        const int64_t lo = std::max<int64_t>(0, p - (G.n_swa - 1));
        float * sc = score.data() + (size_t) (t * nkv);
        for (int64_t j = 0; j < nkv; ++j) {
            if (j > p) { sc[j] = -INFINITY; continue; }
            if (j >= lo) { sc[j] = INFINITY; continue; }   // the window always survives top-k (idx_mask = +inf)
            const float * kj = c.mla_row.data() + (size_t) (j * ROW + KROW);
            double s = 0.0;
            for (int64_t h = 0; h < IH; ++h) {
                const float * qh = idxq.data() + (size_t) ((t * IH + h) * ID);
                double d = 0.0;
                for (int64_t i = 0; i < ID; ++i) d += (double) qh[i] * (double) kj[i];
                s += (d > 0.0 ? d : 0.0) * (double) idxw[(size_t) (t * IH + h)];
            }
            sc[j] = (float) s;
        }
        std::vector<int32_t> cand;
        cand.reserve((size_t) (p + 1));
        for (int64_t j = 0; j <= p; ++j) cand.push_back((int32_t) j);
        if ((int64_t) cand.size() > n_top)
            std::partial_sort(cand.begin(), cand.begin() + n_top, cand.end(), [&](int32_t a, int32_t b) {
                if (sc[a] != sc[b]) return sc[a] > sc[b];
                return a < b;   // ggml's top_k leaves ties unspecified; index order is the only deterministic rule
            });
        nsel[(size_t) t] = std::min<int64_t>((int64_t) cand.size(), n_top);
        std::memcpy(sel.data() + (size_t) (t * n_top), cand.data(), (size_t) (nsel[(size_t) t] * 4));
    });
    tap_push(taps, "indexer_score-" + n, {nkv, nt}, score.data(), nt * nkv);
    {
        std::vector<float> ids((size_t) (nt * n_top));
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (float) sel[i];
        tap_push(taps, "indexer_top_k-" + n, {n_top, nt}, ids.data(), nt * n_top);
    }

    std::vector<float> lat_scratch((size_t) (nt * NH * KL));   // the per-head attention output, in the latent
    par(nt, [&](int64_t t) {
        const int64_t ns = nsel[(size_t) t];
        const int32_t * s = sel.data() + (size_t) (t * n_top);
        std::vector<float> logit((size_t) ns);
        std::vector<float> lat((size_t) KL);
        for (int64_t h = 0; h < NH; ++h) {
            const float * qv = Qcur.data() + (size_t) ((t * NH + h) * KROW);
            float mx = w.sinks[(size_t) h];
            for (int64_t k = 0; k < ns; ++k) {
                const float * kr = c.mla_row.data() + (size_t) ((int64_t) s[k] * ROW);
                double d = 0.0;
                for (int64_t i = 0; i < KROW; ++i) d += (double) qv[i] * (double) kr[i];
                logit[(size_t) k] = (float) d * att_scale;
                mx = std::max(mx, logit[(size_t) k]);
            }
            double den = std::exp((double) (w.sinks[(size_t) h] - mx));
            for (int64_t k = 0; k < ns; ++k) {
                logit[(size_t) k] = (float) std::exp((double) (logit[(size_t) k] - mx));
                den += (double) logit[(size_t) k];
            }
            for (int64_t d = 0; d < KL; ++d) lat[(size_t) d] = 0.0f;
            for (int64_t k = 0; k < ns; ++k) {
                const float pj = (float) ((double) logit[(size_t) k] / den);
                const float * kr = c.mla_row.data() + (size_t) ((int64_t) s[k] * ROW);
                for (int64_t d = 0; d < KL; ++d) lat[(size_t) d] += pj * kr[d];
            }
            std::memcpy(&lat_scratch[(size_t) ((t * NH + h) * KL)], lat.data(), (size_t) KL * sizeof(float));
        }
    });

    // ---- per-head V expansion (v_b[h]) then the VHA head mix --------------
    std::vector<float> attm((size_t) (nt * OJ));
    {
        std::vector<float> xh((size_t) (nt * KL)), yh((size_t) (nt * DV));
        for (int64_t h = 0; h < NH; ++h) {
            for (int64_t t = 0; t < nt; ++t)
                std::memcpy(&xh[(size_t) (t * KL)], &lat_scratch[(size_t) ((t * NH + h) * KL)],
                            (size_t) KL * sizeof(float));
            linear(xh.data(), nt, KL, w.mv_b.data() + (size_t) (h * DV * KL), DV, yh.data(), w.q8);
            for (int64_t t = 0; t < nt; ++t)
                std::memcpy(&attm[(size_t) (t * OJ + h * DV)], &yh[(size_t) (t * DV)], (size_t) DV * sizeof(float));
        }
    }
    tap_push(taps, "attn_mla-" + n, {OJ, nt}, attm.data(), nt * OJ);

    std::vector<float> mixed((size_t) (nt * OJ));
    par(nt, [&](int64_t t) {
        for (int64_t g = 0; g < NH; ++g) {
            float * og = mixed.data() + (size_t) (t * OJ + g * DV);
            for (int64_t d = 0; d < DV; ++d) og[d] = 0.0f;
            for (int64_t h = 0; h < NH; ++h) {
                const float m = w.vha[(size_t) (g * NH + h)];
                if (m == 0.0f) continue;
                const float * oh = attm.data() + (size_t) (t * OJ + h * DV);
                for (int64_t d = 0; d < DV; ++d) og[d] += m * oh[d];
            }
        }
    });
    {   // tap the head mix in its ggml shape: [v_head_dim, n_head, n_tokens]
        std::vector<float> hm((size_t) (nt * OJ));
        for (int64_t t = 0; t < nt; ++t)
            for (int64_t h = 0; h < NH; ++h)
                std::memcpy(&hm[(size_t) ((t * NH + h) * DV)], &mixed[(size_t) (t * OJ + h * DV)],
                            (size_t) DV * sizeof(float));
        tap_push(taps, "attn_head_mix-" + n, {DV, NH, nt}, hm.data(), nt * OJ);
    }
    for (int64_t i = 0; i < nt * OJ; ++i) mixed[(size_t) i] *= gate[(size_t) i];
    linear(mixed.data(), nt, OJ, w.mwo.data(), D, out, w.q8);
    tap_push(taps, "attn_out-" + n, {D, nt}, out, nt * D);
    return true;
}

bool ErnieDense::Impl::run_layer(int il, std::vector<ErnieTap> * taps) {
    const int64_t D = G.n_embd, HC = G.hc, NH = G.n_head, DH = G.d_head_hca, QL = G.r_q;
    const int64_t OG = G.o_groups, OL = G.o_lora, OD = G.o_group_dim();
    const int64_t nt = n;
    const LayerW & w = W[(size_t) il];
    LayerCache & c = C[(size_t) il];

    const int64_t hc_dim = HC * D;
    std::vector<float> collapsed((size_t) (nt * D)), pre((size_t) (nt * HC)), post((size_t) (nt * HC)),
        comb((size_t) (nt * HC * HC));
    hc_pre(x.data(), nt, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, collapsed.data(), pre.data(), post.data(),
           comb.data());
    tap_push(taps, "hc_pre-" + std::to_string(il), {HC, nt}, pre.data(), nt * HC);
    tap_push(taps, "hc_post-" + std::to_string(il), {HC, nt}, post.data(), nt * HC);
    tap_push(taps, "hc_attn_pre-" + std::to_string(il), {D, nt}, collapsed.data(), nt * D);

    std::vector<float> cur((size_t) (nt * D));
    rmsnorm(collapsed.data(), w.attn_norm.data(), nt, D, cur.data());
    tap_push(taps, "attn_norm-" + std::to_string(il), {D, nt}, cur.data(), nt * D);

    // ---- attention: HCA (window + compressed rows) or MLA + DSA (indexer top-k) ----
    std::vector<float> attn_out((size_t) (nt * D), 0.0f);
    if (w.mla && !attn_mla(il, cur.data(), nt, taps, attn_out.data())) return false;
    if (w.hca) {
        std::vector<float> qc((size_t) (nt * QL));
        linear(cur.data(), nt, D, w.q_a.data(), QL, qc.data(), w.q8);
        tap_push(taps, "qr-" + std::to_string(il), {QL, nt}, qc.data(), nt * QL);
        std::vector<float> qcn((size_t) (nt * QL));
        rmsnorm(qc.data(), w.q_a_norm.data(), nt, QL, qcn.data());

        const int64_t OJ = OG * OL;   // 8192
        std::vector<float> gate((size_t) (nt * OJ));
        linear(qcn.data(), nt, QL, w.o_gate.data(), OJ, gate.data(), w.q8);
        for (float & g : gate) g = sigmoidf_(g);
        tap_push(taps, "attn_gate-" + std::to_string(il), {OJ, nt}, gate.data(), nt * OJ);

        std::vector<float> q((size_t) (nt * NH * DH));
        linear(qcn.data(), nt, QL, w.q_b.data(), NH * DH, q.data(), w.q8);
        rmsnorm(q.data(), nullptr, nt * NH, DH, q.data());   // per-head, unweighted
        tap_push(taps, "q_norm-" + std::to_string(il), {DH, NH, nt}, q.data(), nt * NH * DH);
        par(nt * NH, [&](int64_t i) { rope(&q[(size_t) (i * DH)], DH, pos0 + i / NH, false); });
        tap_push(taps, "q-" + std::to_string(il), {DH, NH, nt}, q.data(), nt * NH * DH);

        std::vector<float> kv((size_t) (nt * DH));
        linear(cur.data(), nt, D, w.kv_w.data(), DH, kv.data(), w.q8);
        rmsnorm(kv.data(), w.kv_norm.data(), nt, DH, kv.data());
        par(nt, [&](int64_t t) { rope(&kv[(size_t) (t * DH)], DH, pos0 + t, false); });
        tap_push(taps, "kv-" + std::to_string(il), {DH, 1, nt}, kv.data(), nt * DH);
        if (cfg.kv_fp16) for (float & v : kv) v = f16_(v);   // the cache is F16 in the reference

        // ---- raw window / compressor state ---------------------------------
        if ((int64_t) c.raw_k.size() < (int64_t) ((pos0 + nt) * DH)) c.raw_k.resize((size_t) ((pos0 + nt) * DH));
        std::memcpy(c.raw_k.data() + (size_t) pos0 * DH, kv.data(), (size_t) (nt * DH) * sizeof(float));

        std::vector<float> ckv((size_t) (nt * DH)), cg((size_t) (nt * DH));
        linear(cur.data(), nt, D, w.c_kv.data(), DH, ckv.data(), w.q8);
        linear(cur.data(), nt, D, w.c_gate.data(), DH, cg.data(), w.q8);
        if (c.pend_kv.empty()) { c.pend_kv.assign((size_t) (kHcaRatio * DH), 0.0f); c.pend_gate.assign((size_t) (kHcaRatio * DH), 0.0f); }
        for (int64_t t = 0; t < nt; ++t) {
            const int64_t p = pos0 + t;
            if (c.pend == 0) c.pend_base = p - (p % kHcaRatio);
            else if (c.pend_base + c.pend != p) { err = "compressor state is not contiguous at position " + std::to_string(p); return false; }
            std::memcpy(c.pend_kv.data() + (size_t) (c.pend * DH), ckv.data() + (size_t) (t * DH), (size_t) DH * sizeof(float));
            std::memcpy(c.pend_gate.data() + (size_t) (c.pend * DH), cg.data() + (size_t) (t * DH), (size_t) DH * sizeof(float));
            if (++c.pend == kHcaRatio) {
                c.comp_k.resize(c.comp_k.size() + (size_t) DH);
                float * row = c.comp_k.data() + c.comp_k.size() - (size_t) DH;
                for (int64_t d = 0; d < DH; ++d) {
                    float mx = -INFINITY;
                    for (int64_t p2 = 0; p2 < kHcaRatio; ++p2) {
                        const float s = c.pend_gate[(size_t) (p2 * DH + d)] + w.c_ape[(size_t) (p2 * DH + d)];
                        mx = std::max(mx, s);
                    }
                    double den = 0.0, acc = 0.0;
                    for (int64_t p2 = 0; p2 < kHcaRatio; ++p2) {
                        const float s = c.pend_gate[(size_t) (p2 * DH + d)] + w.c_ape[(size_t) (p2 * DH + d)];
                        const double e = std::exp((double) (s - mx));
                        den += e;
                        acc += e * (double) c.pend_kv[(size_t) (p2 * DH + d)];
                    }
                    row[d] = (float) (acc / den);
                }
                rmsnorm(row, w.c_norm.data(), 1, DH, row);
                rope(row, DH, c.pend_base, false);
                if (cfg.kv_fp16) for (int64_t d = 0; d < DH; ++d) row[d] = f16_(row[d]);
                c.pend = 0;
            }
        }

        // ---- attention: raw window + compressed rows + the learned sink ------
        const float scale = 1.0f / std::sqrt((float) DH);
        std::vector<float> att((size_t) (nt * NH * DH));
        par(nt, [&](int64_t t) {
            const int64_t p = pos0 + t;
            const int64_t lo = std::max<int64_t>(0, p - (G.n_swa - 1));
            const int64_t nraw = p - lo + 1;
            const int64_t ncomp = (p + 1) / kHcaRatio;
            std::vector<float> logit((size_t) (nraw + ncomp));
            for (int64_t h = 0; h < NH; ++h) {
                const float * qv = q.data() + (size_t) ((t * NH + h) * DH);
                float mx = w.sinks[(size_t) h];
                for (int64_t j = 0; j < nraw; ++j) {
                    const float * kk = c.raw_k.data() + (size_t) ((lo + j) * DH);
                    double s = 0.0;
                    for (int64_t d = 0; d < DH; ++d) s += (double) qv[d] * (double) kk[d];
                    logit[(size_t) j] = (float) s * scale;
                    mx = std::max(mx, logit[(size_t) j]);
                }
                for (int64_t b = 0; b < ncomp; ++b) {
                    const float * kk = c.comp_k.data() + (size_t) (b * DH);
                    double s = 0.0;
                    for (int64_t d = 0; d < DH; ++d) s += (double) qv[d] * (double) kk[d];
                    logit[(size_t) (nraw + b)] = (float) s * scale;
                    mx = std::max(mx, logit[(size_t) (nraw + b)]);
                }
                double den = std::exp((double) (w.sinks[(size_t) h] - mx));
                for (int64_t j = 0; j < nraw + ncomp; ++j) {
                    logit[(size_t) j] = (float) std::exp((double) (logit[(size_t) j] - mx));
                    den += (double) logit[(size_t) j];
                }
                float * ov = att.data() + (size_t) ((t * NH + h) * DH);
                for (int64_t d = 0; d < DH; ++d) ov[d] = 0.0f;
                for (int64_t j = 0; j < nraw; ++j) {
                    const float wj = (float) ((double) logit[(size_t) j] / den);
                    const float * kk = c.raw_k.data() + (size_t) ((lo + j) * DH);
                    for (int64_t d = 0; d < DH; ++d) ov[d] += wj * kk[d];
                }
                for (int64_t b = 0; b < ncomp; ++b) {
                    const float wj = (float) ((double) logit[(size_t) (nraw + b)] / den);
                    const float * kk = c.comp_k.data() + (size_t) (b * DH);
                    for (int64_t d = 0; d < DH; ++d) ov[d] += wj * kk[d];
                }
            }
        });
        tap_push(taps, "attn_hca-" + std::to_string(il), {DH * NH, nt}, att.data(), nt * NH * DH);

        par(nt * NH, [&](int64_t i) { rope(&att[(size_t) (i * DH)], DH, pos0 + i / NH, true); });
        tap_push(taps, "attn_derope-" + std::to_string(il), {DH, NH, nt}, att.data(), nt * NH * DH);

        // ---- VHA head mix: out[g] = sum_h (I + V U^T)[g][h] * x[h] ----------
        std::vector<float> mixed((size_t) (nt * NH * DH));
        par(nt, [&](int64_t t) {
            for (int64_t g = 0; g < NH; ++g) {
                float * og = mixed.data() + (size_t) ((t * NH + g) * DH);
                for (int64_t d = 0; d < DH; ++d) og[d] = 0.0f;
                for (int64_t h = 0; h < NH; ++h) {
                    const float m = w.vha[(size_t) (g * NH + h)];
                    if (m == 0.0f) continue;
                    const float * oh = att.data() + (size_t) ((t * NH + h) * DH);
                    for (int64_t d = 0; d < DH; ++d) og[d] += m * oh[d];
                }
            }
        });

        // ---- grouped low-rank output, sigmoid gate, wo_b --------------------
        const int64_t HPG = G.n_heads_group();   // heads per group
        std::vector<float> oa((size_t) (nt * OJ), 0.0f);
        par(nt, [&](int64_t t) {
            std::vector<int8_t> qs((size_t) OD);
            std::vector<float> dx((size_t) (OD / 32));
            for (int64_t g = 0; g < OG; ++g) {
                const float * gin = mixed.data() + (size_t) ((t * NH + g * HPG) * DH);
                const float * wg = w.wo_a.data() + (size_t) (g * OL * OD);
                float * outg = oa.data() + (size_t) (t * OJ + g * OL);
                if (w.q8) quantize_q8(gin, OD, qs.data(), dx.data());
                for (int64_t o = 0; o < OL; ++o) {
                    const float * wr = wg + o * OD;
                    double s = 0.0;
                    if (w.q8) {
                        for (int64_t b = 0; b < OD / 32; ++b) {
                            double sb = 0.0;
                            for (int64_t j = 0; j < 32; ++j)
                                sb += (double) wr[b * 32 + j] * (double) qs[(size_t) (b * 32 + j)];
                            s += sb * (double) dx[(size_t) b];
                        }
                    } else {
                        for (int64_t i = 0; i < OD; ++i) s += (double) gin[i] * (double) wr[i];
                    }
                    outg[o] = (float) s;
                }
            }
        });
        {   // the PR's attn_wo_a is [o_lora, nt, groups]; our buffer is [t][g*OL+o]
            std::vector<float> perm((size_t) (nt * OJ));
            for (int64_t g = 0; g < OG; ++g)
                for (int64_t t = 0; t < nt; ++t)
                    for (int64_t o = 0; o < OL; ++o) perm[(size_t) ((g * nt + t) * OL + o)] = oa[(size_t) (t * OJ + g * OL + o)];
            tap_push(taps, "attn_wo_a-" + std::to_string(il), {OL, nt, OG}, perm.data(), nt * OJ);
        }
        for (int64_t i = 0; i < nt * OJ; ++i) oa[(size_t) i] *= gate[(size_t) i];
        linear(oa.data(), nt, OJ, w.wo_b.data(), D, attn_out.data(), w.q8);
        tap_push(taps, "attn_out-" + std::to_string(il), {D, nt}, attn_out.data(), nt * D);
    }

    std::vector<float> x_next((size_t) (nt * hc_dim));
    hc_post(attn_out.data(), x.data(), post.data(), comb.data(), nt, x_next.data());
    tap_push(taps, "hc_attn_post-" + std::to_string(il), {D, HC, nt}, x_next.data(), nt * hc_dim);
    x.swap(x_next);

    // ---- FFN branch -----------------------------------------------------
    std::vector<float> fcollapsed((size_t) (nt * D)), fpre((size_t) (nt * HC)), fpost((size_t) (nt * HC)),
        fcomb((size_t) (nt * HC * HC));
    hc_pre(x.data(), nt, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, fcollapsed.data(), fpre.data(), fpost.data(),
           fcomb.data());
    tap_push(taps, "hc_ffn_pre-" + std::to_string(il), {D, nt}, fcollapsed.data(), nt * D);

    std::vector<float> fcur((size_t) (nt * D));
    rmsnorm(fcollapsed.data(), w.ffn_norm.data(), nt, D, fcur.data());
    tap_push(taps, "ffn_norm-" + std::to_string(il), {D, nt}, fcur.data(), nt * D);

    std::vector<float> ffn_out((size_t) (nt * D), 0.0f);
    if (w.dense_ffn) {
        const int64_t FF = G.n_ff;
        std::vector<float> fg((size_t) (nt * FF)), fu((size_t) (nt * FF));
        linear(fcur.data(), nt, D, w.ffn_gate.data(), FF, fg.data(), w.q8);
        linear(fcur.data(), nt, D, w.ffn_up.data(), FF, fu.data(), w.q8);
        for (int64_t i = 0; i < nt * FF; ++i) fg[(size_t) i] = situ(fg[(size_t) i], fu[(size_t) i],
                                                                  (float) G.situ_beta, (float) G.situ_linear_beta);
        linear(fg.data(), nt, FF, w.ffn_down.data(), D, ffn_out.data(), w.q8);
    } else if (!cfg.stub_moe_ffn) {
        if (!moe_ffn(il, fcur.data(), nt, taps, ffn_out.data())) return false;
    }
    tap_push(taps, "ffn_out-" + std::to_string(il), {D, nt}, ffn_out.data(), nt * D);

    std::vector<float> x_final((size_t) (nt * hc_dim));
    hc_post(ffn_out.data(), x.data(), fpost.data(), fcomb.data(), nt, x_final.data());
    tap_push(taps, "l_last-" + std::to_string(il), {D, HC, nt}, x_final.data(), nt * hc_dim);
    x.swap(x_final);
    return true;
}

// ------------------------------------------------------------------------ public API
ErnieDense::ErnieDense() : p_(new Impl()) {}
ErnieDense::~ErnieDense() = default;
const ErnieGeometry & ErnieDense::geom() const { return p_->G; }
const std::string & ErnieDense::last_error() const { return p_->err; }
int ErnieDense::pass_size() const { return p_->n; }
int ErnieDense::pass_pos0() const { return p_->pos0; }
int64_t ErnieDense::layer_lo() const { return p_->prep_lo; }
int64_t ErnieDense::layer_hi() const { return p_->prep_hi; }
const std::vector<float> & ErnieDense::residual() const { return p_->x; }

bool ErnieDense::init(const std::string & gguf, const ErnieDenseConfig & cfg, std::string & err) {
    p_->cfg = cfg;
    try {
        p_->model.reset(new GgufModel(GgufModel::open(gguf)));
    } catch (const std::exception & e) {
        err = std::string("open ") + gguf + ": " + e.what();
        return false;
    }
    // One ERNIE geometry in the tree: the GGUF gate's arch check and key reader (include/strata/artifact/
    // ernie_geometry.hpp), then the structural invariants this runner's kernels depend on.
    const std::string aerr = check_architecture_ernie(p_->model->meta());
    if (!aerr.empty()) { err = aerr; return false; }
    err = ernie_read_geometry(*p_->model, p_->G);
    if (err.empty()) {
        const ErnieGeometry & G = p_->G;
        if (G.n_head % G.o_groups != 0) err = "attention.head_count is not a multiple of output_group_count";
        else if (G.kv_lora != G.d_head_hca) err = "kv_lora_rank != key_length_swa";
        else if (G.d_rope > G.d_head_hca || G.d_rope % 2) err = "rope.dimension_count must be even and <= the head";
        else if (G.d_rope > G.d_head_mla_k || G.d_kv_row != G.kv_lora + G.d_rope)
            err = "the MLA geometry is inconsistent (key_length_mla, key_length, kv_lora_rank, rope.dimension_count)";
        else if (G.indexer_n_head <= 0 || G.indexer_key_dim <= 0 || G.indexer_top_k <= 0)
            err = "attention.indexer.{head_count,key_length,top_k} must be positive";
        else if (G.n_embd % 32 || G.r_q % 32 || G.d_head_hca % 32 || G.n_embd_lat % 32 || G.n_ff_exp % 32 ||
                 G.kv_lora % 32 || G.indexer_key_dim % 32 || G.d_head_mla_v % 32)
            err = "a dense-path width is not a Q8_0 block multiple";
    }
    if (!err.empty()) return false;
    p_->emb = p_->model->find("token_embd.weight", &p_->emb_shard);
    if (!p_->emb) { err = "token_embd.weight is missing"; return false; }
    if ((int64_t) p_->emb->shape[1] != p_->G.vocab_size) {
        err = "token_embd has " + std::to_string(p_->emb->shape[1]) + " rows but the tokenizer arrays hold " +
              std::to_string(p_->G.vocab_size) + " ids";
        return false;
    }
    p_->build_inv_freq();
    p_->build_inv_freq_plain();
    p_->W.assign((size_t) p_->G.n_layer(), LayerW());
    p_->C.assign((size_t) p_->G.n_layer(), LayerCache());
    return true;
}

bool ErnieDense::prepare(int lo, int hi, std::string & err) {
    Impl & I = *p_;
    const ErnieGeometry & G = I.G;
    if (lo < 0 || hi > G.n_layer() || lo >= hi) { err = "prepare: bad layer range"; return false; }
    const int64_t D = G.n_embd, HC = G.hc, NH = G.n_head, DH = G.d_head_hca, QL = G.r_q;
    const int64_t OG = G.o_groups, OL = G.o_lora, OD = G.o_group_dim();
    for (int il = lo; il < hi; ++il) {
        LayerW & w = I.W[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        w.hca = G.is_hca(il);
        w.mla = !w.hca;
        auto need = [&](const std::string & n, int64_t e, std::vector<float> & out) { return I.load_full(n, e, out); };
        // the tensors both layer kinds have (the converter maps the two HF naming schemes onto one GGUF name)
        if (!need(b + "attn_norm.weight", D, w.attn_norm) || !need(b + "ffn_norm.weight", D, w.ffn_norm) ||
            !need(b + "hc_attn_fn.weight", HC * D * (2 + HC) * HC, w.hc_attn_fn) ||
            !need(b + "hc_attn_base.weight", (2 + HC) * HC, w.hc_attn_base) ||
            !need(b + "hc_attn_scale.weight", 3, w.hc_attn_scale) ||
            !need(b + "hc_ffn_fn.weight", HC * D * (2 + HC) * HC, w.hc_ffn_fn) ||
            !need(b + "hc_ffn_base.weight", (2 + HC) * HC, w.hc_ffn_base) ||
            !need(b + "hc_ffn_scale.weight", 3, w.hc_ffn_scale) ||
            !need(b + "attn_sinks.weight", NH, w.sinks) || !need(b + "attn_vha.weight", NH * NH, w.vha) ||
            !need(b + "attn_q_a.weight", D * QL, w.q_a) || !need(b + "attn_q_a_norm.weight", QL, w.q_a_norm))
            { err = I.err; return false; }
        if (w.hca) {
            if (!need(b + "attn_q_b.weight", QL * NH * DH, w.q_b) || !need(b + "attn_kv.weight", D * DH, w.kv_w) ||
                !need(b + "attn_kv_a_norm.weight", DH, w.kv_norm) ||
                !need(b + "attn_output_a.weight", OD * OL * OG, w.wo_a) ||
                !need(b + "attn_output_b.weight", OG * OL * D, w.wo_b) ||
                !need(b + "attn_gate.weight", QL * OG * OL, w.o_gate) ||
                !need(b + "attn_compressor_kv.weight", D * DH, w.c_kv) ||
                !need(b + "attn_compressor_gate.weight", D * DH, w.c_gate) ||
                !need(b + "attn_compressor_ape.weight", DH * kHcaRatio, w.c_ape) ||
                !need(b + "attn_compressor_norm.weight", DH, w.c_norm))
                { err = I.err; return false; }
        } else {
            // the stacked per-head K/V splits and the indexer; their element counts are exact, the dims are checked
            // so a transposed or mis-split tensor cannot silently feed the kernels
            auto need_dims = [&](const std::string & n, std::vector<int64_t> dims, std::vector<float> & out) {
                size_t s = 0;
                const TensorInfo * t = I.model->find(n, &s);
                if (!t) { I.err = "missing tensor " + n; return false; }
                for (size_t i = 0; i < dims.size(); ++i)
                    if ((int64_t) t->shape[i] != dims[i]) {
                        I.err = n + " has dim " + std::to_string(i) + " = " + std::to_string((int64_t) t->shape[i]) +
                                ", expected " + std::to_string(dims[i]);
                        return false;
                    }
                int64_t e = 1;
                for (int64_t d : dims) e *= d;
                return I.load_full(n, e, out);
            };
            const int64_t QKH = G.d_head_mla_k, NOPE = G.mla_qk_nope(), DV = G.d_head_mla_v;
            const int64_t KL = G.kv_lora, DR = G.d_rope, ID = G.indexer_key_dim, IH = G.indexer_n_head;
            if (!need_dims(b + "attn_q_b.weight", {QL, NH * QKH}, w.q_b) ||
                !need_dims(b + "attn_kv_a_mqa.weight", {D, KL + DR}, w.mkv_a) ||
                !need_dims(b + "attn_kv_a_norm.weight", {KL}, w.mkv_norm) ||
                !need_dims(b + "attn_k_b.weight", {NOPE, KL, NH}, w.mk_b) ||
                !need_dims(b + "attn_v_b.weight", {KL, DV, NH}, w.mv_b) ||
                !need_dims(b + "attn_output.weight", {NH * DV, D}, w.mwo) ||
                !need_dims(b + "attn_gate.weight", {QL, NH * DV}, w.o_gate) ||
                !need_dims(b + "indexer.attn_k.weight", {D, ID}, w.idx_k) ||
                !need_dims(b + "indexer.k_norm.weight", {ID}, w.idx_kn) ||
                !need_dims(b + "indexer.k_norm.bias", {ID}, w.idx_kb) ||
                !need_dims(b + "indexer.attn_q_b.weight", {QL, IH * ID}, w.idx_qb) ||
                !need_dims(b + "indexer.proj.weight", {D, IH}, w.idx_w)) { err = I.err; return false; }
        }
        w.dense_ffn = il < G.n_layer_dense_lead;
        if (w.dense_ffn) {
            if (!need(b + "ffn_gate.weight", D * G.n_ff, w.ffn_gate) ||
                !need(b + "ffn_up.weight", D * G.n_ff, w.ffn_up) ||
                !need(b + "ffn_down.weight", G.n_ff * D, w.ffn_down)) { err = I.err; return false; }
        } else {
            const int64_t LAT = G.n_embd_lat, NE = G.n_expert, SE = G.n_ff_exp * G.n_expert_shared;
            if (!need(b + "ffn_gate_inp.weight", D * 2 * NE, w.r_gate) ||
                !need(b + "exp_probs_b.bias", NE, w.r_bias) || !need(b + "exp_scale.weight", NE, w.r_scale) ||
                !need(b + "ffn_routed_down.weight", D * LAT, w.r_down) ||
                !need(b + "ffn_routed_up.weight", LAT * D, w.r_up) ||
                !need(b + "ffn_routed_norm.weight", LAT, w.r_lat_norm) ||
                !need(b + "ffn_gate_shexp.weight", D * SE, w.s_gate) ||
                !need(b + "ffn_up_shexp.weight", D * SE, w.s_up) ||
                !need(b + "ffn_down_shexp.weight", SE * D, w.s_down)) { err = I.err; return false; }
            // the stacked experts are read per used expert at run time; only their declared shape is checked here
            auto stack_ok = [&](const char * n, int64_t n_in, int64_t n_out) {
                const TensorInfo * t = I.model->find(b + n);
                return t && t->shape.size() == 3 && (int64_t) t->shape[0] == n_in &&
                       (int64_t) t->shape[1] == n_out && (int64_t) t->shape[2] == NE;
            };
            if (!stack_ok("ffn_gate_exps.weight", LAT, G.n_ff_exp) ||
                !stack_ok("ffn_up_exps.weight", LAT, G.n_ff_exp) ||
                !stack_ok("ffn_down_exps.weight", G.n_ff_exp, LAT)) {
                err = b + "ffn_{gate,up,down}_exps.weight is not the expected [in, out, n_expert] stack";
                return false;
            }
            // the routed experts use the same Q8_0 activation rounding when the file stores them as Q8_0
            const int q8t = (int) GGML_TYPE_Q8_0;
            w.q8_moe = I.tensor_type(b + "ffn_routed_down.weight") == q8t &&
                       I.tensor_type(b + "ffn_routed_up.weight") == q8t &&
                       I.tensor_type(b + "ffn_gate_shexp.weight") == q8t &&
                       I.tensor_type(b + "ffn_up_shexp.weight") == q8t &&
                       I.tensor_type(b + "ffn_down_shexp.weight") == q8t;
            w.q8_exp = I.tensor_type(b + "ffn_gate_exps.weight") == q8t &&
                       I.tensor_type(b + "ffn_up_exps.weight") == q8t &&
                       I.tensor_type(b + "ffn_down_exps.weight") == q8t;
        }
        // every matrix this layer multiplies with is Q8_0 in the converter's output; then the matmuls take ggml's
        // Q8_0 activation rounding (the same arithmetic the reference runs).  A different type falls back to exact F32.
        const int q8t = (int) GGML_TYPE_Q8_0;
        if (w.hca) {
            w.q8 = I.tensor_type(b + "attn_q_a.weight") == q8t && I.tensor_type(b + "attn_q_b.weight") == q8t &&
                   I.tensor_type(b + "attn_kv.weight") == q8t && I.tensor_type(b + "attn_output_a.weight") == q8t &&
                   I.tensor_type(b + "attn_output_b.weight") == q8t && I.tensor_type(b + "attn_gate.weight") == q8t &&
                   I.tensor_type(b + "attn_compressor_kv.weight") == q8t &&
                   I.tensor_type(b + "attn_compressor_gate.weight") == q8t;
        } else {
            w.q8 = I.tensor_type(b + "attn_q_a.weight") == q8t && I.tensor_type(b + "attn_q_b.weight") == q8t &&
                   I.tensor_type(b + "attn_kv_a_mqa.weight") == q8t && I.tensor_type(b + "attn_k_b.weight") == q8t &&
                   I.tensor_type(b + "attn_v_b.weight") == q8t && I.tensor_type(b + "attn_output.weight") == q8t &&
                   I.tensor_type(b + "attn_gate.weight") == q8t &&
                   I.tensor_type(b + "indexer.attn_k.weight") == q8t &&
                   I.tensor_type(b + "indexer.attn_q_b.weight") == q8t;
            w.idx_w_q8 = I.tensor_type(b + "indexer.proj.weight") == q8t;
        }
        if (w.dense_ffn)
            w.q8 = w.q8 && I.tensor_type(b + "ffn_gate.weight") == q8t &&
                   I.tensor_type(b + "ffn_up.weight") == q8t && I.tensor_type(b + "ffn_down.weight") == q8t;
    }
    I.prep_lo = lo;
    I.prep_hi = hi;
    return true;
}

bool ErnieDense::begin(const int * tokens, int n, int pos0, std::string & err) {
    Impl & I = *p_;
    if (n <= 0) { err = "begin: no tokens"; return false; }
    if (pos0 < 0) { err = "begin: negative position"; return false; }
    if ((int64_t) pos0 != I.next_pos) {
        err = "begin: position " + std::to_string(pos0) + " but the state continues at " + std::to_string(I.next_pos);
        return false;
    }
    const int64_t D = I.G.n_embd, HC = I.G.hc;
    I.x.assign((size_t) (n * HC * D), 0.0f);
    std::vector<float> row((size_t) D);
    for (int t = 0; t < n; ++t) {
        if (!I.emb_row(tokens[t], row.data())) { err = I.err; return false; }
        for (int64_t s = 0; s < HC; ++s) std::memcpy(&I.x[(size_t) ((t * HC + s) * D)], row.data(), (size_t) D * sizeof(float));
    }
    I.n = n;
    I.pos0 = pos0;
    return true;
}

bool ErnieDense::begin_seeded(const float * x, int n, int pos0, std::string & err) {
    Impl & I = *p_;
    if (n <= 0) { err = "begin_seeded: no tokens"; return false; }
    if ((int64_t) pos0 != I.next_pos) {
        err = "begin_seeded: position " + std::to_string(pos0) + " but the state continues at " + std::to_string(I.next_pos);
        return false;
    }
    I.x.assign(x, x + (size_t) n * I.G.hc * I.G.n_embd);
    I.n = n;
    I.pos0 = pos0;
    return true;
}

bool ErnieDense::run(int lo, int hi, std::vector<ErnieTap> * taps, std::string & err) {
    Impl & I = *p_;
    if (!I.check_ready(lo, hi)) { err = I.err; return false; }
    for (int il = lo; il < hi; ++il) {
        if (!I.run_layer(il, taps)) { err = I.err; return false; }
    }
    I.next_pos = (int64_t) I.pos0 + I.n;
    return true;
}

}  // namespace strata::v41
