// hy4_generate <model-00001-of-N.gguf> <golden dir> [n_gen] - the Hy4 P2b gate: chunked prefill + greedy decode with
// the native dense half (CUDA) and the reference CPU experts.
//
// The prompt is <dir>/tokens.i32 (a hy4_golden dump).  Prefill runs in passes of HY4_CHUNK tokens (default 256), so
// every pass after the first attends over K rows written by earlier passes (and the sliding window crosses chunk
// edges).  After the prefill: the last row's logits against <dir>/logits_last.f32 and the last chunk's final-layer
// state against <dir>/l_out-<last>.f32 when present.  Then n_gen greedy tokens, one pass each, printed detokenized.
#include "hy4_dense.hpp"
#include "hy4_moe.hpp"
#include "../../train/lora_gguf.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::hy4;

template <class T> static bool load(const std::string & p, std::vector<T> & v) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    v.resize((size_t) f.tellg() / sizeof(T));
    f.seekg(0);
    f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    return true;
}

// GPT-2 byte-level BPE: token text -> bytes
static std::string detok(const std::string & t) {
    static int inv[512];
    static bool init = false;
    if (!init) {
        for (int & x : inv) x = -1;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const int cp = keep ? b : 256 + n++;
            inv[cp] = b;
        }
        init = true;
    }
    std::string out;
    for (size_t i = 0; i < t.size();) {
        unsigned c = (unsigned char) t[i];
        int cp, len;
        if (c < 0x80) { cp = (int) c; len = 1; }
        else if ((c >> 5) == 6 && i + 1 < t.size()) { cp = (int) (((c & 31) << 6) | (t[i + 1] & 63)); len = 2; }
        else { out += t.substr(i); break; }   // a special token (wide chars): keep as is
        if (cp < 512 && inv[cp] >= 0) out += (char) inv[cp];
        else out += t.substr(i, (size_t) len);
        i += (size_t) len;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hy4_generate <model.gguf> <golden dir> [n_gen]\n");
        return 2;
    }
    const std::string gguf = argv[1], gd = argv[2];
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 32;
    const int chunk = std::getenv("HY4_CHUNK") ? std::atoi(std::getenv("HY4_CHUNK")) : 256;
    const int nthr = std::getenv("HY4_THREADS") ? std::atoi(std::getenv("HY4_THREADS")) : 16;
    std::vector<int32_t> toks;
    if (!load(gd + "/tokens.i32", toks) || toks.empty()) { std::fprintf(stderr, "no tokens.i32\n"); return 2; }
    const int np = (int) toks.size();

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "no CUDA\n"); return 1; }
    Hy4DenseConfig cfg;
    cfg.backend = be;
    cfg.ctx_max = std::max(2048, (np + n_gen + 64 + 3) / 4 * 4);
    cfg.max_pass = std::max(chunk, 1);
    Hy4Dense dense;
    std::string err;
    if (!dense.init(gguf, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const Hy4Geometry & G = dense.geom();
    // HY4_LORA_LOAD=<file>: adapters saved by hy4_train (HY4_TRAIN_SAVE): a LoRA GGUF (.gguf, llama.cpp format) or the raw
    // .bin (name, n_in, n_out, rank, A, B per adapter)
    if (const char * lf = std::getenv("HY4_LORA_LOAD")) {
        std::vector<std::string> names;
        std::vector<std::vector<float>> As, Bs;
        int64_t rank = 0;
        float file_alpha = 0;
        const std::string lfs = lf;
        if (lfs.size() > 5 && lfs.compare(lfs.size() - 5, 5, ".gguf") == 0) {
            std::vector<xolotl::train::LoraTensorPair> ad;
            std::string arch;
            if (!xolotl::train::read_lora_gguf(lfs, ad, file_alpha, arch, err)) { std::fprintf(stderr, "lora load: %s\n", err.c_str()); return 1; }
            for (auto & p : ad) {
                if (rank && p.rank != rank) { std::fprintf(stderr, "lora load: mixed ranks are not supported\n"); return 1; }
                rank = p.rank;
                names.push_back(p.name); As.push_back(std::move(p.A)); Bs.push_back(std::move(p.B));
            }
        }
        std::ifstream in(lfs.size() > 5 && lfs.compare(lfs.size() - 5, 5, ".gguf") == 0 ? std::string() : lfs, std::ios::binary);
        for (uint32_t len; in && in.read((char *) &len, 4);) {
            std::string nm(len, '\0');
            in.read(nm.data(), len);
            int64_t a, b, r;
            in.read((char *) &a, 8); in.read((char *) &b, 8); in.read((char *) &r, 8);
            As.emplace_back((size_t) (a * r)); Bs.emplace_back((size_t) (r * b));
            in.read((char *) As.back().data(), (std::streamsize) (As.back().size() * 4));
            in.read((char *) Bs.back().data(), (std::streamsize) (Bs.back().size() * 4));
            names.push_back(nm);
            rank = r;
        }
        const float alpha = std::getenv("HY4_LORA_ALPHA") ? (float) std::atof(std::getenv("HY4_LORA_ALPHA")) : file_alpha > 0 ? file_alpha : 16.0f;
        if (names.empty() || !dense.lora_add(names, (int) rank, alpha, 1, err)) { std::fprintf(stderr, "lora load: %s\n", err.c_str()); return 1; }
        for (int i = 0; i < dense.lora_count(); ++i) dense.lora_set(i, As[(size_t) i].data(), Bs[(size_t) i].data());
        std::printf("loaded %d adapters (rank %lld, alpha %.0f) from %s\n", dense.lora_count(), (long long) rank, alpha, lf);
    }
    Hy4Moe moe;
    if (!moe.init(dense.model(), gguf, G, nthr, err)) { std::fprintf(stderr, "moe: %s\n", err.c_str()); return 1; }
    const MetaValue * tv = dense.model().meta().get("tokenizer.ggml.tokens");
    auto piece = [&](int id) { return tv && id >= 0 && (size_t) id < tv->items.size() ? detok(tv->items[(size_t) id].s) : std::string("?"); };

    std::vector<int> ids((size_t) (cfg.max_pass * G.n_used));
    std::vector<float> wts(ids.size()), routed((size_t) (cfg.max_pass * G.n_latent));
    // HY4_PROF=1: wall time per stage (begin / attn+router / experts / finish), summed over the decode steps
    const bool prof = std::getenv("HY4_PROF") && std::getenv("HY4_PROF")[0] == '1';
    double t_begin = 0, t_attn = 0, t_moe = 0, t_fin = 0;
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    auto pass = [&](const int * t, int n, int pos0) -> bool {
        auto a = clk::now();
        if (!dense.begin(t, n, pos0)) return false;
        if (n == 1) t_begin += since(a);
        for (int il = 0; il < G.n_layer; ++il) {
            const float * fn = nullptr;
            bool dl = false;
            a = clk::now();
            if (!dense.attn_router(il, ids.data(), wts.data(), &fn, &dl)) return false;
            if (n == 1) t_attn += since(a);
            if (dl) continue;
            a = clk::now();
            if (!moe.run(il, fn, ids.data(), wts.data(), n, routed.data())) return false;
            if (n == 1) t_moe += since(a);
            a = clk::now();
            if (!dense.finish(il, routed.data())) return false;
            if (n == 1) t_fin += since(a);
        }
        return true;
    };

    auto t0 = std::chrono::steady_clock::now();
    int pos = 0, last_n = 0;
    while (pos < np) {
        const int n = std::min(chunk, np - pos);
        if (!pass(toks.data() + pos, n, pos)) return 1;
        last_n = n;
        pos += n;
        std::printf("prefill %d/%d\n", pos, np);
        std::fflush(stdout);
    }
    const double t_pre = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> lg((size_t) G.vocab), lref, st, sref;
    if (!dense.logits(last_n - 1, 1, lg.data())) return 1;
    int next = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    if (load(gd + "/logits_last.f32", lref) && lref.size() == lg.size()) {
        const int ar = (int) (std::max_element(lref.begin(), lref.end()) - lref.begin());
        double mx = 0, zr = 0, zo = 0, kl = 0;
        const double mr = lref[(size_t) ar], mo = lg[(size_t) next];
        for (size_t i = 0; i < lg.size(); ++i) {
            mx = std::max(mx, (double) std::fabs(lg[i] - lref[i]));
            zr += std::exp(lref[i] - mr);
            zo += std::exp(lg[i] - mo);
        }
        for (size_t i = 0; i < lg.size(); ++i) {
            const double pr = std::exp(lref[i] - mr) / zr;
            kl += pr * (((lref[i] - mr) - std::log(zr)) - ((lg[i] - mo) - std::log(zo)));
        }
        std::printf("prefill logits vs ref: argmax ours %d '%s' ref %d '%s' %s | max|diff| %.4f | KL %.3e\n", next,
                    piece(next).c_str(), ar, piece(ar).c_str(), next == ar ? "MATCH" : "DIFFER", mx, kl);
    }
    // the final-layer reference: the whole prompt in one file, or (HY4_LOUT_REF=<file>) only the last ubatch's rows
    const std::string lref_f = std::getenv("HY4_LOUT_REF") ? std::string(std::getenv("HY4_LOUT_REF"))
                                                           : gd + "/l_out-" + std::to_string(G.n_layer - 1) + ".f32";
    if (load(lref_f, sref) && dense.read_state(st)) {
        const size_t row = (size_t) (G.hc * G.n_embd);
        const size_t off = sref.size() == (size_t) np * row ? (size_t) (np - last_n) * row : 0;
        if (sref.size() == (size_t) np * row || sref.size() == (size_t) last_n * row) {
            double num = 0, den = 0;
            for (size_t i = 0; i < st.size(); ++i) {
                const double d = (double) st[i] - sref[off + i];
                num += d * d;
                den += (double) sref[off + i] * sref[off + i];
            }
            std::printf("last chunk l_out-%lld rel err %.3e (rows %d..%d)\n", (long long) (G.n_layer - 1), std::sqrt(num / den),
                        np - last_n, np - 1);
        }
    }
    std::printf("prefill %d tokens in %.1f s (chunk %d)\n", np, t_pre, chunk);

    std::string text = piece(next);
    std::vector<int> gen = { next };
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 1; i < n_gen && next != (int) G.eos && pos < cfg.ctx_max - 1; ++i) {
        if (!pass(&next, 1, pos)) return 1;
        ++pos;
        if (!dense.logits(0, 1, lg.data())) return 1;
        next = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        gen.push_back(next);
        text += piece(next);
    }
    const double t_dec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("generated %zu tokens (%.2f tok/s reference-expert path):", gen.size(), (gen.size() - 1) / std::max(t_dec, 1e-9));
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n---\n%s\n---\n", text.c_str());
    if (prof && gen.size() > 1) {
        const double k = 1e3 / (double) (gen.size() - 1);
        std::printf("decode ms/token: begin %.1f | attn+router %.1f | experts %.1f | finish %.1f | rest (logits) %.1f\n",
                    t_begin * k, t_attn * k, t_moe * k, t_fin * k, (t_dec - t_begin - t_attn - t_moe - t_fin) * k);
    }
    moe.report();
    return 0;
}
