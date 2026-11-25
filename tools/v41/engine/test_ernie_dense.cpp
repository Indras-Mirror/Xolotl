// test_ernie_dense - run the ERNIE-5.5 HCA dense path on the CPU and dump named intermediates.
//
// usage: test_ernie_dense MODEL --tokens 1,2,3 --out DIR [--layers lo:hi] [--threads N]
//                        [--seed-x FILE] [--names a,b,c] [--stub-moe]
//
//   --layers lo:hi  run layers [lo, hi) (default 0 : min(3, block_count)); a seed is required for lo > 0
//   --seed-x FILE   residual to start from, [n_embd, hc, nt] float32 in ggml order (ne0 = the embedding dim) -
//                   the reference's `l_last-{lo-1}` file, so a mid-stack layer is gated on identical input
//   --stub-moe      zero the FFN of the MoE layers instead of running the latent MoE (the S3 gate)
//   --names         comma separated tap names or prefixes (exact, or "<prefix>-<layer>"); empty = all
//   --chunk N       walk the prompt in passes of N tokens (N = 1 is decode; default: one pass).  A chunked run also
//                   dumps "residual-all" [n_embd, hc, nt] - it must equal the single-pass residual bit for bit.
//
// Output: DIR/<name>.f32 (little-endian float32, ggml order: ne[0] fastest) and DIR/manifest.json in the layout
// tools/ernie/ernie_dump.cpp writes, so compare_ernie.py can diff the two directly.
#include "ernie_dense.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::v41;

namespace {

std::string sanitize(const std::string & s) {
    std::string r = s;
    for (auto & c : r)
        if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) c = '_';
    return r;
}

bool want(const std::vector<std::string> & names, const std::string & n) {
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

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s MODEL --tokens 1,2,3 --out DIR [--layers lo:hi] [--threads N] [--seed-x FILE] [--names a,b]\n"
                 "       [--stub-moe] [--chunk N]\n",
                 argv0);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    std::string model = argv[1], out, tok_arg, seed_file, names_arg, layers_arg;
    std::vector<std::string> names;
    int threads = 8, chunk = 0;
    bool stub_moe = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); exit(2); }
            return argv[++i];
        };
        if (a == "--tokens") tok_arg = next();
        else if (a == "--out") out = next();
        else if (a == "--layers") layers_arg = next();
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--seed-x") seed_file = next();
        else if (a == "--names") names_arg = next();
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--stub-moe") stub_moe = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model.empty() || tok_arg.empty() || out.empty()) { usage(argv[0]); return 2; }
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
    const std::vector<int> toks = parse_ints(tok_arg);
    if (toks.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
    const int nt = (int) toks.size();

    ErnieDenseConfig cfg;
    cfg.n_threads = threads;
    cfg.stub_moe_ffn = stub_moe;
    ErnieDense dn;
    std::string err;
    if (!dn.init(model, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    const strata::ErnieGeometry & G = dn.geom();
    int lo = 0, hi = (int) std::min<int64_t>(3, G.n_layer());
    if (!layers_arg.empty()) {
        const size_t c = layers_arg.find(':');
        if (c == std::string::npos) { lo = std::atoi(layers_arg.c_str()); hi = lo + 1; }
        else { lo = std::atoi(layers_arg.substr(0, c).c_str()); hi = std::atoi(layers_arg.substr(c + 1).c_str()); }
    }
    if (lo < 0 || hi > (int) G.n_layer() || lo >= hi) { std::fprintf(stderr, "bad layer range %d:%d\n", lo, hi); return 2; }
    if (lo > 0 && seed_file.empty()) {
        std::fprintf(stderr, "layer %d needs --seed-x (the reference's l_last-%d)\n", lo, lo - 1);
        return 2;
    }
    if (!dn.prepare(lo, hi, err)) { std::fprintf(stderr, "prepare: %s\n", err.c_str()); return 1; }

    // the residual to start from, when seeded (whole prompt)
    std::vector<float> seed;
    if (!seed_file.empty()) {
        std::ifstream f(seed_file, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot read %s\n", seed_file.c_str()); return 1; }
        seed.resize((size_t) nt * G.hc * G.n_embd);
        f.read((char *) seed.data(), (std::streamsize) (seed.size() * sizeof(float)));
        if (!f || (size_t) f.gcount() != seed.size() * sizeof(float)) {
            std::fprintf(stderr, "%s: expected %zu floats ([%lld, %lld, %d])\n", seed_file.c_str(), seed.size(),
                         (long long) G.n_embd, (long long) G.hc, nt);
            return 1;
        }
    }
    if (chunk <= 0 || chunk > nt) chunk = nt;
    const int64_t col = (int64_t) G.hc * G.n_embd;

    std::vector<ErnieTap> taps;
    std::vector<float> all((size_t) (nt * col));
    const auto t0 = std::chrono::steady_clock::now();
    for (int p0 = 0; p0 < nt; p0 += chunk) {
        const int n = std::min(chunk, nt - p0);
        if (!seed.empty()) {
            if (!dn.begin_seeded(seed.data() + (size_t) p0 * col, n, p0, err)) { std::fprintf(stderr, "begin_seeded: %s\n", err.c_str()); return 1; }
        } else {
            if (!dn.begin(toks.data() + p0, n, p0, err)) { std::fprintf(stderr, "begin: %s\n", err.c_str()); return 1; }
        }
        const bool last = p0 + n == nt;
        std::vector<ErnieTap> * tp = last ? &taps : nullptr;
        if (p0 == 0 && !seed.empty() && last) {
            // a seeded single pass has no hc_init: the seed is the layer input
        } else if (p0 == 0 && lo == 0 && seed.empty() && last) {
            ErnieTap t;
            t.name = "hc_init";
            t.ne = {G.n_embd, G.hc, (int64_t) nt, 1};
            t.data = dn.residual();
            taps.push_back(std::move(t));
        }
        if (!dn.run(lo, hi, tp, err)) { std::fprintf(stderr, "run: %s\n", err.c_str()); return 1; }
        std::memcpy(all.data() + (size_t) p0 * col, dn.residual().data(), (size_t) n * col * sizeof(float));
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (chunk < nt) {   // a chunked run reports the whole prompt's residual, whatever the chunk size
        ErnieTap t;
        t.name = "residual-all";
        t.ne = {G.n_embd, G.hc, (int64_t) nt, 1};
        t.data = all;
        taps.push_back(std::move(t));
    }

    std::ofstream man(out + "/manifest.json");
    if (!man) { std::fprintf(stderr, "cannot write %s/manifest.json\n", out.c_str()); return 1; }
    man << "{\"tensors\": [\n";
    int written = 0;
    size_t bytes = 0;
    for (const ErnieTap & t : taps) {
        if (!want(names, t.name)) continue;
        const std::string file = sanitize(t.name) + ".f32";
        std::ofstream f(out + "/" + file, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot write %s/%s\n", out.c_str(), file.c_str()); return 1; }
        f.write((const char *) t.data.data(), (std::streamsize) (t.data.size() * sizeof(float)));
        f.close();
        int64_t ne[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < t.ne.size() && i < 4; ++i) ne[i] = t.ne[i] != 0 ? t.ne[i] : 1;
        man << (written ? ",\n" : "") << "    {\"name\": \"" << t.name << "\", \"file\": \"" << file
            << "\", \"seq\": 1, \"ne\": [" << ne[0] << ", " << ne[1] << ", " << ne[2] << ", " << ne[3]
            << "], \"ggml_type\": \"F32\", \"n_elements\": " << t.data.size() << "}";
        ++written;
        bytes += t.data.size() * sizeof(float);
    }
    man << "\n], \"tokens\": [";
    for (int i = 0; i < nt; ++i) man << (i ? ", " : "") << toks[(size_t) i];
    man << "], \"layers\": [" << lo << ", " << hi << "], \"seed_x\": \"" << seed_file << "\", \"threads\": " << threads
        << ", \"chunk\": " << chunk << "}\n";
    man.close();

    std::printf("ernie_dense: layers %d..%d, %d tokens (chunk %d) in %.2f s -> %d tensors (%.1f MiB) in %s\n", lo, hi, nt,
                chunk, sec, written, (double) bytes / (1024.0 * 1024.0), out.c_str());
    return written == 0 ? 2 : 0;
}
