// qfn_topk_long <pack.gguf> <capture-root> <window-dir> [threads]
//
// The LONG-CONTEXT gate on a real harness window (harness_capture.py output): the shared 20K-token Quetza prefix +
// the window, the served rope (set QFN_ROPE_SCALE=2), layer 47's engine selection replayed.
//   layers 44-46 (GDN): every position in chunks of QFN_LONG_CHUNK (2048) carried by GdnCarry; the window rows are
//                       compared with the captured x45/x46/x47.bf16
//   layer 47 (QSA)    : the prefix only yields K/V (accumulated); the window's queries run in chunks of 512 over every
//                       earlier cell with the STRATA_QSA_DUMP selection as an additive mask; compared with final.bf16
//   head              : CE over the window's loss tokens, ours vs the head applied to the captured final residual
#include "ggml.h"
#include "ggml-cpu.h"
#include "qfn_topk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace xolotl::train;

static std::vector<float> read_bf16(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> b((size_t) n / 2);
    if (std::fread(b.data(), 2, b.size(), f) != b.size()) std::exit(1);
    std::fclose(f);
    std::vector<float> v(b.size());
    for (size_t i = 0; i < b.size(); ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&v[i], &u, 4); }
    return v;
}

static void report(const char * tag, const float * in, const float * out, const float * ref, size_t n) {
    double dd = 0, rr = 0, dot = 0, oo = 0;
    for (size_t i = 0; i < n; ++i) {
        const double dr = (double) ref[i] - in[i], dout = (double) out[i] - in[i];
        dd += (dout - dr) * (dout - dr); rr += dr * dr; dot += dr * dout; oo += dout * dout;
    }
    std::printf("[long] %s: update rel err %.3e cos %.6f\n", tag, std::sqrt(dd / rr), dot / std::sqrt(rr * oo));
    std::fflush(stdout);
}

// minimal JSON int-array reader for tokens.json {"tokens": [...], "mask": [...]}
static std::vector<int> json_ints(const std::string & s, const std::string & key) {
    std::vector<int> v;
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return v;
    p = s.find('[', p);
    const size_t e = s.find(']', p);
    std::stringstream ss(s.substr(p + 1, e - p - 1));
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <pack.gguf> <capture-root> <window-dir> [threads]\n", argv[0]); return 2; }
    const std::string root = argv[2], wdir = argv[3];
    const int nth = argc > 4 ? atoi(argv[4]) : 16;
    const int64_t CH = std::getenv("QFN_LONG_CHUNK") ? atoll(std::getenv("QFN_LONG_CHUNK")) : 2048;
    const int64_t CQ = 512;
    const int64_t L0 = 44;

    PackTensors p;
    if (!p.open(argv[1])) return 1;
    const QfnGeom G = qfn_geom(p);
    std::vector<std::string> names = { "output.weight", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight" };
    for (int64_t l = L0; l < G.NL; ++l) for (auto & n : qfn_layer_names(l)) names.push_back(n);
    if (!p.load(names)) return 1;
    const int64_t row = G.hc * G.D;

    std::vector<float> pre = read_bf16(root + "/prefix_x44.bf16"), win = read_bf16(wdir + "/x44.bf16");
    const int64_t P = (int64_t) (pre.size() / row), W = (int64_t) (win.size() / row), N = P + W;
    std::vector<float> cur(pre);
    cur.insert(cur.end(), win.begin(), win.end());
    pre.clear(); pre.shrink_to_fit();
    std::ifstream tf(wdir + "/tokens.json");
    std::string tjs((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const std::vector<int> wtok = json_ints(tjs, "tokens"), wmask = json_ints(tjs, "mask");
    std::printf("[long] prefix %lld + window %lld = %lld positions | chunk %lld | rope freq_scale %.3f ext %.1f orig %d | "
                "%zu window tokens, %d loss\n", (long long) P, (long long) W, (long long) N, (long long) CH,
                G.rope_freq_scale, G.rope_ext, G.rope_orig_ctx, wtok.size(), (int) std::count(wmask.begin(), wmask.end(), 1));

    // layer 47 selection: pos -> cells
    std::map<int64_t, std::vector<int32_t>> sel;
    {
        FILE * f = std::fopen((wdir + "/qsa.bin").c_str(), "rb");
        int32_t h[4];
        while (f && std::fread(h, 4, 4, f) == 4) {
            std::vector<int32_t> ids((size_t) h[2] * h[3]);
            if (std::fread(ids.data(), 4, ids.size(), f) != ids.size()) break;
            for (int32_t t = 0; t < h[2]; ++t)
                sel[h[1] + t] = std::vector<int32_t>(ids.begin() + (size_t) t * h[3], ids.begin() + (size_t) (t + 1) * h[3]);
        }
        if (f) std::fclose(f);
    }
    for (int64_t t = P; t < N; ++t) if (!sel.count(t)) { std::fprintf(stderr, "no selection for position %lld\n", (long long) t); return 1; }

    auto make_ctx = [](size_t mb) { ggml_init_params ip = { mb << 20, nullptr, false }; return ggml_init(ip); };
    auto new_f32 = [](ggml_context * c, const float * d, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        ggml_tensor * t = ggml_new_tensor_4d(c, GGML_TYPE_F32, n0, n1, n2, n3);
        std::memcpy(t->data, d, (size_t) ggml_nelements(t) * 4);
        return t;
    };
    auto ident_t = [](ggml_context * c) {
        ggml_tensor * id = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
        for (int i = 0; i < kGdnChunk * kGdnChunk; ++i) ((float *) id->data)[i] = (i % (kGdnChunk + 1)) == 0 ? 1.0f : 0.0f;
        return id;
    };
    auto pos_t = [](ggml_context * c, int64_t p0, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        for (int64_t i = 0; i < n; ++i) ((int32_t *) t->data)[i] = (int32_t) (p0 + i);
        return t;
    };
    struct HostT { std::vector<float> d; int64_t ne[4] = { 1, 1, 1, 1 }; };
    auto to_host = [](ggml_tensor * t, HostT & h) {
        h.d.resize((size_t) ggml_nelements(t));
        std::memcpy(h.d.data(), t->data, h.d.size() * 4);
        for (int i = 0; i < 4; ++i) h.ne[i] = t->ne[i];
    };

    // one layer over rows [p0, p0+n) of `cur`, with optional carries / mask; returns the output rows
    struct Carry { HostT conv, s; std::vector<float> k, v; int64_t nkv = 0; };
    auto layer_chunk = [&](int64_t l, int64_t p0, int64_t n, Carry & cy, const std::vector<float> * mask,
                           std::vector<float> & out) {
        const bool qsa = G.is_qsa(l);
        std::vector<int32_t> ids;
        std::vector<float> wts;
        std::vector<std::vector<int32_t>> tok_of, pair_of;
        for (int phase = 0; phase < 2; ++phase) {
            ggml_context * c = make_ctx(phase == 0 ? 12288 : 16384);
            ggml_tensor * x = new_f32(c, &cur[(size_t) (p0 * row)], G.D, G.hc, n, 1);
            TopLayer L = qfn_top_layer(c, p, l, G, ident_t(c));
            if (qsa) {
                if (cy.nkv) {
                    L.qsa_carry.k_in = new_f32(c, cy.k.data(), G.ahd, G.n_kv, cy.nkv, 1);
                    L.qsa_carry.v_in = new_f32(c, cy.v.data(), G.ahd, G.n_kv, cy.nkv, 1);
                    L.qsa_carry.k_in = ggml_reshape_3d(c, L.qsa_carry.k_in, G.ahd, G.n_kv, cy.nkv);
                    L.qsa_carry.v_in = ggml_reshape_3d(c, L.qsa_carry.v_in, G.ahd, G.n_kv, cy.nkv);
                }
                if (mask) L.qsa_carry.sel_mask = new_f32(c, mask->data(), cy.nkv + n, n, 1, 1);
                if (mask) L.qsa_carry.sel_mask = ggml_reshape_2d(c, L.qsa_carry.sel_mask, cy.nkv + n, n);
            } else if (!cy.s.d.empty()) {
                L.gdn_carry.conv_in = ggml_reshape_2d(c, new_f32(c, cy.conv.d.data(), cy.conv.ne[0], cy.conv.ne[1], 1, 1),
                                                      cy.conv.ne[0], cy.conv.ne[1]);
                L.gdn_carry.s_in = new_f32(c, cy.s.d.data(), cy.s.ne[0], cy.s.ne[1], cy.s.ne[2], cy.s.ne[3]);
            }
            ggml_tensor *m2, *inj2;
            ggml_tensor * res1 = qfn_attn_half(c, x, L, p, G, pos_t(c, p0, n), 1.0f, &m2, &inj2, true);
            ggml_cgraph * g = ggml_new_graph_custom(c, 1 << 18, false);
            if (phase == 0) {
                ggml_tensor * logits = ggml_mul_mat(c, p.get("blk." + std::to_string((long long) l) + ".ffn_gate_inp.weight"), m2);
                ggml_build_forward_expand(g, logits);
                ggml_graph_compute_with_ctx(c, g, nth);
                std::vector<float> lt((size_t) (G.E * n));
                const float * lp = (const float *) logits->data;
                for (int64_t t = 0; t < n; ++t)
                    for (int64_t e = 0; e < G.E; ++e) lt[(size_t) (e * n + t)] = lp[(size_t) (t * G.E + e)];
                moe_route(lt.data(), G.E, n, G.K, ids, wts);
                moe_expert_lists(ids, G.K, n, G.E, tok_of, pair_of);
            } else {
                MoeIndex ix = moe_make_index(c, ids, G.K, n, G.E, tok_of, pair_of);
                ggml_tensor * y = qfn_hc_combine(c, res1, qfn_ffn(c, m2, p, l, ix), inj2, G.hc);
                ggml_build_forward_expand(g, y);
                ggml_tensor * outs[4] = { L.gdn_carry.conv_out, L.gdn_carry.s_out, L.qsa_carry.k_out, L.qsa_carry.v_out };
                for (ggml_tensor * t : outs) if (t) ggml_build_forward_expand(g, t);
                ggml_graph_compute_with_ctx(c, g, nth);
                out.resize((size_t) (n * row));
                std::memcpy(out.data(), y->data, out.size() * 4);
                if (!qsa) { to_host(L.gdn_carry.conv_out, cy.conv); to_host(L.gdn_carry.s_out, cy.s); }
                else {
                    const size_t kn = (size_t) ggml_nelements(L.qsa_carry.k_out);
                    const size_t o0 = cy.k.size();
                    cy.k.resize(o0 + kn); cy.v.resize(o0 + kn);
                    std::memcpy(&cy.k[o0], L.qsa_carry.k_out->data, kn * 4);
                    std::memcpy(&cy.v[o0], L.qsa_carry.v_out->data, kn * 4);
                    cy.nkv += n;
                }
            }
            ggml_free(c);
        }
    };
    // layer 47 prefix: K/V only (no attention, no FFN: nothing after this layer reads the prefix)
    auto qsa_kv_chunk = [&](int64_t l, int64_t p0, int64_t n, Carry & cy) {
        ggml_context * c = make_ctx(8192);
        ggml_tensor * x = new_f32(c, &cur[(size_t) (p0 * row)], G.D, G.hc, n, 1);
        TopLayer L = qfn_top_layer(c, p, l, G, ident_t(c));
        ggml_tensor * mixed = qfn_hc_mix(c, x, L.hc_attn, G.hc, G.eps, nullptr);
        ggml_tensor *k, *v;
        qfn_qsa_qkv(c, mixed, p, l, G, pos_t(c, p0, n), nullptr, 1.0f, nullptr, nullptr, &k, &v);
        ggml_cgraph * g = ggml_new_graph_custom(c, 8192, false);
        ggml_build_forward_expand(g, k);
        ggml_build_forward_expand(g, v);
        ggml_graph_compute_with_ctx(c, g, nth);
        const size_t kn = (size_t) ggml_nelements(k), o0 = cy.k.size();
        cy.k.resize(o0 + kn); cy.v.resize(o0 + kn);
        std::memcpy(&cy.k[o0], k->data, kn * 4);
        std::memcpy(&cy.v[o0], v->data, kn * 4);
        cy.nkv += n;
        ggml_free(c);
    };

    const auto T0 = std::chrono::steady_clock::now();
    std::vector<float> win_in;
    for (int64_t l = L0; l < G.NL; ++l) {
        const auto tl = std::chrono::steady_clock::now();
        win_in.assign(cur.begin() + P * row, cur.end());
        std::vector<float> next((size_t) (N * row), 0.0f), out;
        Carry cy;
        if (!G.is_qsa(l)) {
            for (int64_t p0 = 0; p0 < N; p0 += CH) {
                const int64_t n = std::min(CH, N - p0);
                layer_chunk(l, p0, n, cy, nullptr, out);
                std::memcpy(&next[(size_t) (p0 * row)], out.data(), out.size() * 4);
            }
        } else {
            for (int64_t p0 = 0; p0 < P; p0 += CH) qsa_kv_chunk(l, p0, std::min(CH, P - p0), cy);
            for (int64_t p0 = P; p0 < N; p0 += CQ) {
                const int64_t n = std::min(CQ, N - p0), tk = cy.nkv + n;
                std::vector<float> mask((size_t) (tk * n), -INFINITY);
                for (int64_t t = 0; t < n; ++t)
                    for (int32_t cell : sel[p0 + t])
                        if (cell >= 0 && cell < tk) mask[(size_t) (t * tk + cell)] = 0.0f;
                layer_chunk(l, p0, n, cy, &mask, out);
                std::memcpy(&next[(size_t) (p0 * row)], out.data(), out.size() * 4);
            }
        }
        cur.swap(next);
        const std::string ref = l + 1 < G.NL ? wdir + "/x" + std::to_string((long long) l + 1) + ".bf16" : wdir + "/final.bf16";
        const std::vector<float> rv = read_bf16(ref);
        char tag[200];
        std::snprintf(tag, sizeof tag, "layer %lld %s (%.0f s) window vs %s", (long long) l, G.is_qsa(l) ? "QSA" : "GDN",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count(), ref.substr(ref.rfind('/') + 1).c_str());
        report(tag, win_in.data(), &cur[(size_t) (P * row)], rv.data(), rv.size());
    }

    // head + CE over the window's loss tokens (target = next window token)
    const std::vector<float> fin = read_bf16(wdir + "/final.bf16");
    double ce_o = 0, ce_c = 0;
    int n_ce = 0, agree = 0, hit_o = 0, hit_c = 0;
    for (int64_t p0 = 0; p0 < W; p0 += 256) {
        const int64_t n = std::min<int64_t>(256, W - p0);
        for (int which = 0; which < 2; ++which) {
            ggml_context * c = make_ctx(4096);
            const float * src = which == 0 ? &cur[(size_t) ((P + p0) * row)] : &fin[(size_t) (p0 * row)];
            ggml_tensor * lg = qfn_head(c, new_f32(c, src, G.D, G.hc, n, 1), p, G);
            ggml_cgraph * g = ggml_new_graph_custom(c, 4096, false);
            ggml_build_forward_expand(g, lg);
            ggml_graph_compute_with_ctx(c, g, nth);
            for (int64_t t = 0; t < n; ++t) {
                const int64_t wt = p0 + t;
                if (wt + 1 >= (int64_t) wtok.size() || !wmask[(size_t) wt + 1]) continue;   // predict a LOSS token
                const float * z = (const float *) lg->data + (size_t) t * G.n_vocab;
                const int y = wtok[(size_t) wt + 1];
                const double m = *std::max_element(z, z + G.n_vocab);
                double s = 0;
                for (int64_t i = 0; i < G.n_vocab; ++i) s += std::exp((double) z[i] - m);
                const double ce = m + std::log(s) - z[y];
                const int am = (int) (std::max_element(z, z + G.n_vocab) - z);
                if (which == 0) { ce_o += ce; hit_o += am == y; ++n_ce; } else { ce_c += ce; hit_c += am == y; }
            }
            ggml_free(c);
        }
    }
    (void) agree;
    std::printf("[long] CE over %d loss tokens: ours %.4f (top-1 %d) vs head(captured final) %.4f (top-1 %d) | %.0f s total\n",
                n_ce, ce_o / std::max(1, n_ce), hit_o, ce_c / std::max(1, n_ce), hit_c,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count());
    return 0;
}
