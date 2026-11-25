// test_ernie_graph - run the ERNIE-5.5 dense ggml graphs on the CPU backend and dump named intermediates.
//
// usage: test_ernie_graph MODEL --tokens 1,2,3 --out DIR [--layers lo:hi] [--threads N] [--chunk N]
//                               [--tokens-file FILE] [--checks] [--no-dump] [--names a,b]
//
//   MODEL            ernie5_5 GGUF (ernie-L3-q8.gguf: layers 0-2, Q8_0 backbone + experts)
//   --chunk N        walk the prompt in passes of N tokens (N = 1 is decode; default: one pass).  A chunked run
//                    dumps "residual-all" [n_embd, hc, nt] and, with --checks, compares it with a single pass.
//   --checks         decode-vs-prefill and chunked-vs-single residual checks (in-process, fresh state each run)
//   --no-dump        run only (no output files)
//
// The routed experts are computed by a host loop over the GGUF's stacked expert tensors (the top-10 the graph selected,
// Q8_0 activation rounding, SiTU-GLU, weighted sum in the 2048-wide latent space) - the dense graph runs end to end
// without the tier.  The dump layout is the CPU runner's (ernie_dense) and the llama.cpp PR dumper's, so
// compare_ernie_graph.py diffs the three directly.
#include "ernie_graph.hpp"
#ifdef ERNIE_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Dump {
    std::vector<ErnieGraphTap> taps;
    std::vector<float> resid;   // nt * hc * n_embd
    std::vector<float> logits;  // n_vocab (last token)
    int nt = 0;
};

std::string sanitize(const std::string & s) {
    std::string r = s;
    for (auto & c : r)
        if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) c = '_';
    return r;
}

bool wanted(const std::vector<std::string> & names, const std::string & n) {
    if (names.empty()) return true;
    for (const auto & p : names) {
        if (n == p) return true;
        if (n.size() > p.size() + 1 && n.compare(0, p.size(), p) == 0 && n[p.size()] == '-') {
            bool digits = true;
            for (size_t i = p.size() + 1; i < n.size(); ++i)
                if (!std::isdigit((unsigned char) n[i])) { digits = false; break; }
            if (digits) return true;
        }
    }
    return false;
}

std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t c = s.find(',', pos);
        if (c == std::string::npos) c = s.size();
        if (c > pos) v.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
        pos = c + 1;
    }
    return v;
}

void usage(const char * a) {
    std::fprintf(stderr, "usage: %s MODEL --tokens 1,2,3 --out DIR [--layers lo:hi] [--threads N] [--chunk N] "
                         "[--tokens-file FILE] [--checks] [--no-dump] [--names a,b] [--cuda]\n", a);
}

// ---- the host tier: the top-k experts from the GGUF, Q8_0 activation rounding, SiTU-GLU, the weighted latent sum -----

inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float situ(float g, float u, float beta, float beta_l) {
    return beta * std::tanh(g / beta) * sigmoidf_(g) * beta_l * std::tanh(u / beta_l);
}

// ggml's Q8_0 activation rounding (per 32-element block, d = max|x|/127 as F16, q = rint(x*127/max))
// small join-all worker split (the host tier is the slow part of the test; the box has 16 cores)
template <class F>
void tpar(int64_t nwork, F fn) {
    const int nth = (int) std::min<int64_t>(std::max(1u, std::thread::hardware_concurrency()),
                                            std::max<int64_t>(1, nwork));
    if (nth <= 1) {
        for (int64_t i = 0; i < nwork; ++i) fn(i);
        return;
    }
    std::vector<std::thread> th;
    th.reserve((size_t) nth);
    for (int k = 0; k < nth; ++k) {
        const int64_t a = nwork * k / nth, b = nwork * (k + 1) / nth;
        th.emplace_back([&fn, a, b] { for (int64_t i = a; i < b; ++i) fn(i); });
    }
    for (auto & t : th) t.join();
}

void quantize_q8(const float * x, int64_t n, int8_t * qs, float * d) {
    for (int64_t b = 0; b < n / 32; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        d[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax / 127.0f));
        for (int j = 0; j < 32; ++j) qs[b * 32 + j] = (int8_t) nearbyintf(xb[j] * id);
    }
}

struct ExpertTier {
    std::unique_ptr<GgufModel> model;
    int64_t LAT = 0, FE = 0, NE = 0, NU = 0, D = 0;
    double situ_beta = 4.0, situ_lb = 25.0;

    bool open(const std::string & path, const ErnieGeometry & g, std::string & err) {
        try {
            model.reset(new GgufModel(GgufModel::open(path)));
        } catch (const std::exception & e) { err = e.what(); return false; }
        LAT = g.n_latent; FE = g.n_ff_exp; NE = g.n_expert; NU = g.n_used; D = g.n_embd;
        situ_beta = g.situ_beta; situ_lb = g.situ_lb;
        return true;
    }

    // one stacked [n_in, n_out, NE] tensor's expert e as F32 rows [n_out][n_in]
    bool load_expert(const std::string & name, int64_t e, int64_t n_in, int64_t n_out, std::vector<float> & out,
                     bool & q8, std::string & err) {
        size_t s = 0;
        const TensorInfo * t = model->find(name, &s);
        if (!t) { err = "missing tensor " + name; return false; }
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != n_out ||
            e < 0 || e >= (int64_t) t->shape[2]) {
            err = "tensor " + name + " is not [in, out, n_expert]";
            return false;
        }
        const ggml_type ty = (ggml_type) t->type;
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr || (ty != GGML_TYPE_F32 && !tr->to_float)) { err = name + ": unsupported type"; return false; }
        q8 = ty == GGML_TYPE_Q8_0;
        const size_t blck = ggml_blck_size(ty);
        if ((size_t) n_in % blck) { err = name + ": a row is not whole blocks"; return false; }
        const size_t row_bytes = (size_t) n_in / blck * ggml_type_size(ty);
        const uint8_t * base = model->shard(s).tensor_data(*t) + (size_t) e * (size_t) n_out * row_bytes;
        out.resize((size_t) (n_out * n_in));
        tpar(n_out, [&](int64_t o) {
            if (ty == GGML_TYPE_F32) std::memcpy(out.data() + o * n_in, base + (size_t) o * row_bytes, (size_t) n_in * 4);
            else tr->to_float(base + (size_t) o * row_bytes, out.data() + o * n_in, n_in);
        });
        return true;
    }

    // y[t][o] = sum_i x[t][i] w[o][i]; q8: round every activation row to Q8_0 first (ggml's Q8_0 weight path)
    static void linear(const float * x, int64_t nt, int64_t in, const float * w, int64_t out_n, float * y, bool q8) {
        const int64_t nb = in / 32;
        std::vector<int8_t> qs(q8 ? (size_t) (nt * in) : 0);
        std::vector<float> dx(q8 ? (size_t) (nt * nb) : 0);
        if (q8) tpar(nt, [&](int64_t t) { quantize_q8(x + t * in, in, qs.data() + t * in, dx.data() + t * nb); });
        tpar(nt * out_n, [&](int64_t idx) {
            const int64_t t = idx / out_n, o = idx % out_n;
            const float * wr = w + o * in;
            double s = 0.0;
            if (!q8) {
                const float * xr = x + t * in;
                for (int64_t i = 0; i < in; ++i) s += (double) xr[i] * (double) wr[i];
            } else {
                const int8_t * qr = qs.data() + t * in;
                const float * dr = dx.data() + t * nb;
                for (int64_t b = 0; b < nb; ++b) {
                    double sb = 0.0;
                    for (int j = 0; j < 32; ++j) sb += (double) wr[b * 32 + j] * (double) qr[b * 32 + j];
                    s += sb * (double) dr[b];
                }
            }
            y[idx] = (float) s;
        });
    }

    // routed_sum [nt][LAT] from the graph's fn [nt][LAT] and its ids / wts [nt][NU]
    bool routed(int il, const int * ids, const float * wts, const float * fn, int64_t nt, std::vector<float> & acc,
                std::string & err) {
        const std::string b = "blk." + std::to_string(il) + ".";
        struct Slot { int t, k; float w; };
        std::vector<std::vector<Slot>> by_e((size_t) NE);
        for (int64_t t = 0; t < nt; ++t)
            for (int64_t k = 0; k < NU; ++k) {
                const int e = ids[t * NU + k];
                if (e < 0 || e >= NE) { err = "selected expert outside the range"; return false; }
                by_e[(size_t) e].push_back(Slot { (int) t, (int) k, wts[t * NU + k] });
            }
        acc.assign((size_t) (nt * LAT), 0.0f);
        std::vector<float> wg, wu, wd, xg, yg, yu, ys, yd;
        bool q8g = false, q8u = false, q8d = false;
        for (int64_t e = 0; e < NE; ++e) {
            const auto & items = by_e[(size_t) e];
            if (items.empty()) continue;
            const int64_t ng = (int64_t) items.size();
            xg.resize((size_t) (ng * LAT));
            for (int64_t i = 0; i < ng; ++i)
                std::memcpy(xg.data() + i * LAT, fn + (size_t) items[(size_t) i].t * LAT, (size_t) LAT * 4);
            if (!load_expert(b + "ffn_gate_exps.weight", e, LAT, FE, wg, q8g, err) ||
                !load_expert(b + "ffn_up_exps.weight", e, LAT, FE, wu, q8u, err) ||
                !load_expert(b + "ffn_down_exps.weight", e, FE, LAT, wd, q8d, err))
                return false;
            yg.resize((size_t) (ng * FE)); yu.resize((size_t) (ng * FE)); ys.resize((size_t) (ng * FE));
            linear(xg.data(), ng, LAT, wg.data(), FE, yg.data(), q8g);
            linear(xg.data(), ng, LAT, wu.data(), FE, yu.data(), q8u);
            for (int64_t i = 0; i < ng * FE; ++i)
                ys[(size_t) i] = situ(yg[(size_t) i], yu[(size_t) i], (float) situ_beta, (float) situ_lb);
            yd.resize((size_t) (ng * LAT));
            linear(ys.data(), ng, FE, wd.data(), LAT, yd.data(), q8d);
            for (int64_t i = 0; i < ng; ++i) {
                const Slot & it = items[(size_t) i];
                float * a = acc.data() + (size_t) it.t * LAT;
                const float * y = yd.data() + i * LAT;
                for (int64_t j = 0; j < LAT; ++j) a[j] += it.w * y[j];
            }
        }
        return true;
    }
};

// one run of layers [lo, hi): returns the taps (of the last pass only), the whole prompt's residual and the last
// token's logits.
bool run(ErnieGraph & g, ExpertTier & tier, const std::vector<int> & toks, int lo, int hi, int chunk, bool want_taps,
         Dump & d, std::string & err) {
    const ErnieGeometry & G = g.geom();
    const int nt = (int) toks.size();
    const int64_t col = G.hc * G.n_embd;
    if (chunk <= 0 || chunk > nt) chunk = nt;
    d.nt = nt;
    d.resid.assign((size_t) (nt * col), 0.0f);
    std::vector<int> ids((size_t) (G.n_used * chunk));
    std::vector<float> wts((size_t) (G.n_used * chunk));
    std::vector<float> acc;
    for (int p0 = 0; p0 < nt; p0 += chunk) {
        const int n = std::min(chunk, nt - p0);
        if (!g.begin(toks.data() + p0, n, p0)) { err = g.last_error(); return false; }
        const bool last = p0 + n == nt;
        if (want_taps && last && p0 == 0) {   // the residual after begin (hc_init: the embedding over the 4 streams)
            std::vector<float> x0;
            if (!g.read_state(0, n, x0)) { err = g.last_error(); return false; }
            ErnieGraphTap t;
            t.name = "hc_init";
            t.ne = { G.n_embd, G.hc, (int64_t) n, 1 };
            t.data = std::move(x0);
            d.taps.push_back(std::move(t));
        }
        std::vector<ErnieGraphTap> tp;
        for (int il = lo; il < hi; ++il) {
            const float * fn = nullptr;
            tp.clear();
            if (!g.attn_router(il, ids.data(), wts.data(), &fn, (want_taps && last) ? &tp : nullptr)) {
                err = g.last_error();
                return false;
            }
            if (il >= (int) G.dense_lead) {
                if (!tier.routed(il, ids.data(), wts.data(), fn, n, acc, err)) return false;
                if (want_taps && last) {
                    ErnieGraphTap t;
                    t.name = "ffn_moe_out-" + std::to_string(il);
                    t.ne = { G.n_latent, n, 1, 1 };
                    t.data = acc;
                    tp.push_back(std::move(t));
                }
                std::vector<ErnieGraphTap> ft;
                if (!g.finish(il, acc.data(), (want_taps && last) ? &ft : nullptr)) { err = g.last_error(); return false; }
                for (auto & t : ft) tp.push_back(std::move(t));
            } else {
                std::vector<ErnieGraphTap> ft;
                if (!g.finish(il, nullptr, (want_taps && last) ? &ft : nullptr)) { err = g.last_error(); return false; }
                for (auto & t : ft) tp.push_back(std::move(t));
            }
            if (want_taps && last) for (auto & t : tp) d.taps.push_back(t);
        }
        std::vector<float> x;
        if (!g.read_state(0, n, x)) { err = g.last_error(); return false; }
        std::memcpy(d.resid.data() + (size_t) p0 * col, x.data(), (size_t) (n * col) * sizeof(float));
    }
    // the head on the last token (the last pass holds columns [0, nt - p0l))
    {
        std::vector<ErnieGraphTap> ht;
        const int p0l = ((nt - 1) / chunk) * chunk;
        const int lastn = nt - p0l;
        d.logits.assign((size_t) G.vocab, 0.0f);
        if (!g.logits(lastn - 1, 1, d.logits.data(), want_taps ? &ht : nullptr)) { err = g.last_error(); return false; }
        for (auto & t : ht) d.taps.push_back(std::move(t));
    }
    return true;
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b, double * mx) {
    double num = 0.0, den = 0.0, m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double dd = (double) a[i] - (double) b[i];
        num += dd * dd;
        den += (double) b[i] * (double) b[i];
        m = std::max(m, std::fabs(dd));
    }
    if (mx) *mx = m;
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    std::string model = argv[1], out, tok_arg, tok_file, layers_arg, names_arg;
    std::vector<std::string> names;
    int threads = 8, chunk = 0;
    bool checks = false, no_dump = false, use_cuda = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); exit(2); }
            return argv[++i];
        };
        if (a == "--tokens") tok_arg = next();
        else if (a == "--tokens-file") tok_file = next();
        else if (a == "--out") out = next();
        else if (a == "--layers") layers_arg = next();
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--names") names_arg = next();
        else if (a == "--checks") checks = true;
        else if (a == "--no-dump") no_dump = true;
        else if (a == "--cuda") use_cuda = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model.empty() || (tok_arg.empty() && tok_file.empty()) || (out.empty() && !no_dump)) { usage(argv[0]); return 2; }
    {
        size_t pos = 0;
        while (pos < names_arg.size()) {
            size_t c = names_arg.find(',', pos);
            if (c == std::string::npos) c = names_arg.size();
            std::string s = names_arg.substr(pos, c - pos);
            if (!s.empty()) names.push_back(s);
            pos = c + 1;
        }
    }
    std::vector<int> toks;
    if (!tok_arg.empty()) toks = parse_ints(tok_arg);
    if (!tok_file.empty()) {
        std::ifstream f(tok_file);
        if (!f) { std::fprintf(stderr, "cannot read %s\n", tok_file.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            std::vector<int> v = parse_ints(line);
            toks.insert(toks.end(), v.begin(), v.end());
        }
    }
    if (toks.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
    const int nt = (int) toks.size();

    ErnieGraphConfig cfg;
    cfg.n_threads = threads;
    cfg.max_pass = std::max(nt, 8);
    if (chunk > cfg.max_pass) cfg.max_pass = chunk;
    const char * ce = std::getenv("ERNIE_CTX_MAX");
    if (ce) cfg.ctx_max = std::atoll(ce);
    if (use_cuda) {
#ifdef ERNIE_HAVE_CUDA
        cfg.backend = ggml_backend_cuda_init(0);
        if (!cfg.backend) { std::fprintf(stderr, "--cuda: ggml_backend_cuda_init(0) failed\n"); return 1; }
#else
        std::fprintf(stderr, "--cuda: this build has no ggml-cuda\n"); return 2;
#endif
    }
    ErnieGraph g;
    std::string err;
    if (!g.init(model, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const ErnieGeometry & G = g.geom();
    int lo = 0, hi = (int) std::min<int64_t>(3, G.n_layer);
    if (!layers_arg.empty()) {
        const size_t c = layers_arg.find(':');
        if (c == std::string::npos) { lo = std::atoi(layers_arg.c_str()); hi = lo + 1; }
        else { lo = std::atoi(layers_arg.substr(0, c).c_str()); hi = std::atoi(layers_arg.substr(c + 1).c_str()); }
    }
    if (lo < 0 || hi > (int) G.n_layer || lo >= hi) { std::fprintf(stderr, "bad layer range %d:%d\n", lo, hi); return 2; }
    ExpertTier tier;
    if (!tier.open(model, G, err)) { std::fprintf(stderr, "experts: %s\n", err.c_str()); return 1; }
    if (!g.snapshot(0)) { std::fprintf(stderr, "snapshot: %s\n", g.last_error().c_str()); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    Dump d;
    // --checks compare against a one-pass run, so the dump run is always one pass then
    const int dump_chunk = checks ? nt : (chunk <= 0 ? nt : chunk);
    if (!run(g, tier, toks, lo, hi, dump_chunk, true, d, err)) {
        std::fprintf(stderr, "run: %s\n", err.c_str());
        return 1;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!no_dump) {
        std::ofstream man(out + "/manifest.json");
        if (!man) { std::fprintf(stderr, "cannot write %s/manifest.json\n", out.c_str()); return 1; }
        man << "{\"tensors\": [\n";
        int written = 0;
        for (const ErnieGraphTap & t : d.taps) {
            if (!wanted(names, t.name)) continue;
            const std::string file = sanitize(t.name) + ".f32";
            std::ofstream f(out + "/" + file, std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot write %s/%s\n", out.c_str(), file.c_str()); return 1; }
            f.write((const char *) t.data.data(), (std::streamsize) (t.data.size() * 4));
            f.close();
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (size_t i = 0; i < t.ne.size() && i < 4; ++i) ne[i] = t.ne[i] > 0 ? t.ne[i] : 1;
            man << (written ? ",\n" : "") << "    {\"name\": \"" << t.name << "\", \"file\": \"" << file
                << "\", \"seq\": 1, \"ne\": [" << ne[0] << ", " << ne[1] << ", " << ne[2] << ", " << ne[3]
                << "], \"ggml_type\": \"F32\", \"n_elements\": " << t.data.size() << "}";
            ++written;
        }
        auto add = [&](const std::string & name, std::vector<int64_t> ne, const std::vector<float> & v) {
            if (!wanted(names, name)) return;
            const std::string file = sanitize(name) + ".f32";
            std::ofstream f(out + "/" + file, std::ios::binary);
            if (!f) return;
            f.write((const char *) v.data(), (std::streamsize) (v.size() * 4));
            f.close();
            while (ne.size() < 4) ne.push_back(1);
            man << (written ? ",\n" : "") << "    {\"name\": \"" << name << "\", \"file\": \"" << file
                << "\", \"seq\": 1, \"ne\": [" << ne[0] << ", " << ne[1] << ", " << ne[2] << ", " << ne[3]
                << "], \"ggml_type\": \"F32\", \"n_elements\": " << v.size() << "}";
            ++written;
        };
        add("residual-all", { G.n_embd, G.hc, (int64_t) nt }, d.resid);
        add("result_output", { G.vocab, 1 }, d.logits);
        man << "\n], \"tokens\": [";
        for (int i = 0; i < nt; ++i) man << (i ? ", " : "") << toks[(size_t) i];
        man << "], \"layers\": [" << lo << ", " << hi << "], \"threads\": " << threads << ", \"chunk\": "
            << (chunk <= 0 ? nt : chunk) << ", \"engine\": \"ernie_graph\"}\n";
        man.close();
        std::printf("test_ernie_graph: layers %d..%d, %d tokens, %.2f s -> %d tensors in %s\n", lo, hi, nt, sec, written,
                    out.c_str());
    } else {
        std::printf("test_ernie_graph: layers %d..%d, %d tokens, %.2f s\n", lo, hi, nt, sec);
    }

    int fails = 0;
    if (checks) {
        // 1) chunked prefill vs one pass
        if ((chunk <= 0 ? nt : chunk) < nt) {
            Dump c;
            if (!g.restore() || !run(g, tier, toks, lo, hi, chunk, false, c, err)) {
                std::fprintf(stderr, "chunk run: %s\n", err.c_str());
                return 1;
            }
            double mx = 0;
            const double r = rel_l2(c.resid, d.resid, &mx);
            std::printf("CHECK chunked(%d)-vs-single  rel L2 %.3e  max abs %.3e  %s\n", chunk, r, mx, r < 1e-6 ? "PASS" : "FAIL");
            fails += r < 1e-6 ? 0 : 1;
        }
        // 2) decode 1 token after a prompt == the same position in one pass
        if (nt > 1) {
            Dump e;
            if (!g.restore()) { std::fprintf(stderr, "restore: %s\n", g.last_error().c_str()); return 1; }
            // [0, nt-1) in one pass (or chunked if chunk was given), then the last token on its own
            const int pre = nt - 1;
            const int ch = (chunk <= 0 || chunk >= pre) ? pre : chunk;
            std::vector<float> resid_pre;
            for (int p0 = 0; p0 < pre; p0 += ch) {
                const int n = std::min(ch, pre - p0);
                if (!g.begin(toks.data() + p0, n, p0)) { std::fprintf(stderr, "begin: %s\n", g.last_error().c_str()); return 1; }
                for (int il = lo; il < hi; ++il) {
                    const float * fn = nullptr;
                    std::vector<int> ids((size_t) (G.n_used * n));
                    std::vector<float> wts((size_t) (G.n_used * n)), acc;
                    if (!g.attn_router(il, ids.data(), wts.data(), &fn)) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                    if (il >= (int) G.dense_lead) {
                        if (!tier.routed(il, ids.data(), wts.data(), fn, n, acc, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
                        if (!g.finish(il, acc.data())) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                    } else if (!g.finish(il, nullptr)) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                }
            }
            // the last token alone
            {
                const int n = 1;
                if (!g.begin(toks.data() + pre, n, pre)) { std::fprintf(stderr, "begin: %s\n", g.last_error().c_str()); return 1; }
                for (int il = lo; il < hi; ++il) {
                    const float * fn = nullptr;
                    std::vector<int> ids((size_t) (G.n_used * n));
                    std::vector<float> wts((size_t) (G.n_used * n)), acc;
                    if (!g.attn_router(il, ids.data(), wts.data(), &fn)) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                    if (il >= (int) G.dense_lead) {
                        if (!tier.routed(il, ids.data(), wts.data(), fn, n, acc, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
                        if (!g.finish(il, acc.data())) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                    } else if (!g.finish(il, nullptr)) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                }
                std::vector<float> x1;
                if (!g.read_state(0, 1, x1)) { std::fprintf(stderr, "%s\n", g.last_error().c_str()); return 1; }
                const int64_t col = G.hc * G.n_embd;
                std::vector<float> ref(d.resid.begin() + (size_t) pre * col, d.resid.begin() + (size_t) (pre + 1) * col);
                double mx = 0;
                const double r = rel_l2(x1, ref, &mx);
                std::printf("CHECK decode-vs-prefill    rel L2 %.3e  max abs %.3e  %s\n", r, mx, r < 1e-6 ? "PASS" : "FAIL");
                fails += r < 1e-6 ? 0 : 1;
            }
        }
    }
    return fails == 0 ? 0 : 3;
}
