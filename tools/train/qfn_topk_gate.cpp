// qfn_topk_gate <pack.gguf> <capture-dir> <L0> [tokens-file] [threads]
//
// The C++ training graph's FORWARD on real data: layers [L0, 48) + head built by qfn_topk.hpp on the pack's weights
// as stored (quantized), fed with the captured residual L<L0>.bin, chaining its OWN outputs and comparing each layer
// against the capture of the next (L<l+1>.bin, final.bf16 after the last layer). With a tokens file it also reports
// next-token CE from our logits and from the head applied to the captured final residual.
// Pass bar = the numpy reference's numbers (qfn_full_layer_ref.py --to-final): per-layer update cos >= 0.9999,
// logits top-1 agreement with the captured final on every position.
#include "ggml.h"
#include "ggml-cpu.h"
#include "qfn_topk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace xolotl::train;

static std::vector<float> read_res(const std::string & path, int64_t row) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v;
    if (path.size() > 5 && path.substr(path.size() - 5) == ".bf16") {
        std::vector<uint16_t> b((size_t) n / 2);
        if (std::fread(b.data(), 2, b.size(), f) != b.size()) std::exit(1);
        v.resize(b.size());
        for (size_t i = 0; i < b.size(); ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&v[i], &u, 4); }
    } else {
        v.resize((size_t) n / 4);
        if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1);
    }
    std::fclose(f);
    if (v.size() % (size_t) row) { std::fprintf(stderr, "%s: not a whole number of rows\n", path.c_str()); std::exit(1); }
    return v;
}

static void report(const char * tag, const std::vector<float> & in, const std::vector<float> & out,
                   const std::vector<float> & ref) {
    double dd = 0, rr = 0, dot = 0, oo = 0, ii = 0, eo = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double dr = (double) ref[i] - in[i], dout = (double) out[i] - in[i];
        dd += (dout - dr) * (dout - dr); rr += dr * dr; dot += dr * dout; oo += dout * dout;
        eo += ((double) out[i] - ref[i]) * ((double) out[i] - ref[i]); ii += (double) ref[i] * ref[i];
    }
    std::printf("[gate] %s: update rel err %.3e cos %.6f | out rel %.3e\n", tag, std::sqrt(dd / rr),
                dot / std::sqrt(rr * oo), std::sqrt(eo / ii));
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <pack.gguf> <capture-dir> <L0> [tokens-file] [threads]\n", argv[0]); return 2; }
    const std::string cap = argv[2];
    const int64_t L0 = atoll(argv[3]);
    const char * tokpath = argc > 4 ? argv[4] : nullptr;
    const int nth = argc > 5 ? atoi(argv[5]) : 16;

    PackTensors p;
    if (!p.open(argv[1])) { std::fprintf(stderr, "cannot open pack\n"); return 1; }
    const QfnGeom G = qfn_geom(p);
    std::vector<std::string> names = { "output.weight", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight" };
    for (int64_t l = L0; l < G.NL; ++l) for (auto & n : qfn_layer_names(l)) names.push_back(n);
    const auto t0 = std::chrono::steady_clock::now();
    if (!p.load(names)) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("[gate] loaded %zu tensors for layers %lld..%lld + head in %.1f s (threads %d)\n", p.t.size(),
                (long long) L0, (long long) G.NL - 1,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), nth);

    const int64_t row = G.hc * G.D;
    std::vector<float> cur = read_res(cap + "/L" + std::to_string((long long) L0) + ".bin", row);
    const int64_t T = (int64_t) (cur.size() / (size_t) row);
    std::printf("[gate] D %lld hc %lld T %lld | E %lld K %lld | attn %lld/%lld x %lld rot %lld | vocab %lld\n",
                (long long) G.D, (long long) G.hc, (long long) T, (long long) G.E, (long long) G.K,
                (long long) G.n_head, (long long) G.n_kv, (long long) G.ahd, (long long) G.n_rot, (long long) G.n_vocab);
    if (T > G.top_k) { std::fprintf(stderr, "T > indexer top_k: QSA is not dense\n"); return 2; }

    auto make_ctx = [](size_t mb) { ggml_init_params ip = { mb << 20, nullptr, false }; return ggml_init(ip); };
    auto inputs = [&](ggml_context * c, const std::vector<float> & r, ggml_tensor ** pos, ggml_tensor ** ident) {
        ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, G.D, G.hc, T);
        std::memcpy(x->data, r.data(), r.size() * 4);
        *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        for (int64_t t = 0; t < T; ++t) ((int32_t *) (*pos)->data)[t] = (int32_t) t;
        *ident = ggml_new_tensor_2d(c, GGML_TYPE_F32, kGdnChunk, kGdnChunk);
        float * id = (float *) (*ident)->data;
        for (int i = 0; i < kGdnChunk * kGdnChunk; ++i) id[i] = (i % (kGdnChunk + 1)) == 0 ? 1.0f : 0.0f;
        return x;
    };

    for (int64_t l = L0; l < G.NL; ++l) {
        const auto tl = std::chrono::steady_clock::now();
        // phase A: the router input from the current weights -> host routing
        std::vector<int32_t> ids;
        std::vector<float> wts;
        {
            ggml_context * c = make_ctx(3072);
            ggml_tensor *pos, *ident;
            ggml_tensor * x = inputs(c, cur, &pos, &ident);
            TopLayer L = qfn_top_layer(c, p, l, G, ident);
            ggml_tensor *m2, *inj2;
            qfn_attn_half(c, x, L, p, G, pos, 1.0f, &m2, &inj2);
            ggml_tensor * logits = ggml_mul_mat(c, p.get("blk." + std::to_string((long long) l) + ".ffn_gate_inp.weight"), m2);
            ggml_cgraph * g = ggml_new_graph_custom(c, 65536, false);
            ggml_build_forward_expand(g, logits);
            ggml_graph_compute_with_ctx(c, g, nth);
            // ggml's [E, T] result is t*E + e; moe_route wants e*T + t (test_moe_train.cpp:91) - transpose here
            std::vector<float> lt((size_t) (G.E * T));
            const float * lp = (const float *) logits->data;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t e = 0; e < G.E; ++e) lt[(size_t) (e * T + t)] = lp[(size_t) (t * G.E + e)];
            moe_route(lt.data(), G.E, T, G.K, ids, wts);
            ggml_free(c);
        }
        std::vector<std::vector<int32_t>> tok_of, pair_of;
        moe_expert_lists(ids, G.K, T, G.E, tok_of, pair_of);
        int64_t used = 0;
        for (auto & v : tok_of) used += !v.empty();
        // phase B: the whole layer with that routing
        std::vector<float> out((size_t) (row * T));
        {
            ggml_context * c = make_ctx(4096);
            ggml_tensor *pos, *ident;
            ggml_tensor * x = inputs(c, cur, &pos, &ident);
            TopLayer L = qfn_top_layer(c, p, l, G, ident);
            ggml_tensor *m2, *inj2;
            ggml_tensor * res1 = qfn_attn_half(c, x, L, p, G, pos, 1.0f, &m2, &inj2);
            MoeIndex ix = moe_make_index(c, ids, G.K, T, G.E, tok_of, pair_of);
            ggml_tensor * y = qfn_hc_combine(c, res1, qfn_ffn(c, m2, p, l, ix), inj2, G.hc);
            ggml_cgraph * g = ggml_new_graph_custom(c, 65536, false);
            ggml_build_forward_expand(g, y);
            ggml_graph_compute_with_ctx(c, g, nth);
            std::memcpy(out.data(), y->data, out.size() * 4);
            ggml_free(c);
        }
        const std::string nxt = l + 1 < G.NL ? cap + "/L" + std::to_string((long long) l + 1) + ".bin" : cap + "/final.bf16";
        char tag[160];
        std::snprintf(tag, sizeof tag, "layer %lld %s (%lld experts, %.1f s) vs %s", (long long) l, G.is_qsa(l) ? "QSA" : "GDN",
                      (long long) used, std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count(),
                      nxt.substr(nxt.rfind('/') + 1).c_str());
        report(tag, cur, out, read_res(nxt, row));
        cur.swap(out);
    }

    // head on ours and on the captured final
    auto head = [&](const std::vector<float> & r) {
        ggml_context * c = make_ctx(2048);
        ggml_tensor *pos, *ident;
        ggml_tensor * x = inputs(c, r, &pos, &ident);
        ggml_tensor * lg = qfn_head(c, x, p, G);
        ggml_cgraph * g = ggml_new_graph_custom(c, 4096, false);
        ggml_build_forward_expand(g, lg);
        ggml_graph_compute_with_ctx(c, g, nth);
        std::vector<float> v((size_t) ggml_nelements(lg));
        std::memcpy(v.data(), lg->data, v.size() * 4);
        ggml_free(c);
        return v;
    };
    const std::vector<float> lo = head(cur), lc = head(read_res(cap + "/final.bf16", row));
    const int64_t V = G.n_vocab;
    std::vector<int32_t> toks;
    if (tokpath) {
        FILE * f = std::fopen(tokpath, "r");
        int v;
        while (f && std::fscanf(f, "%d%*[, \n]", &v) == 1) toks.push_back(v);
        if (f) std::fclose(f);
    }
    int agree = 0, hit = 0, n_ce = 0;
    double ce_o = 0, ce_c = 0, dd = 0, cc = 0;
    for (int64_t t = 0; t < T; ++t) {
        const float * a = &lo[(size_t) (t * V)], * b = &lc[(size_t) (t * V)];
        const int64_t ia = std::max_element(a, a + V) - a, ib = std::max_element(b, b + V) - b;
        agree += ia == ib;
        for (int64_t i = 0; i < V; ++i) { dd += (double) (a[i] - b[i]) * (a[i] - b[i]); cc += (double) b[i] * b[i]; }
        if (t + 1 < (int64_t) toks.size()) {
            const int32_t y = toks[(size_t) t + 1];
            auto ce = [&](const float * z) {
                const double m = *std::max_element(z, z + V);
                double s = 0;
                for (int64_t i = 0; i < V; ++i) s += std::exp((double) z[i] - m);
                return m + std::log(s) - z[y];
            };
            ce_o += ce(a); ce_c += ce(b); hit += ib == y; ++n_ce;
        }
    }
    std::printf("[gate] logits ours vs head(captured final): rel %.3e, top-1 agree %d/%lld\n", std::sqrt(dd / cc), agree, (long long) T);
    if (n_ce)
        std::printf("[gate] next-token CE over %d positions: ours %.4f, head(captured final) %.4f; top-1 = next %d/%d\n",
                    n_ce, ce_o / n_ce, ce_c / n_ce, hit, n_ce);
    return 0;
}
