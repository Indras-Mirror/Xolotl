// tools/v41/engine/ds41_engram_hot.cpp - the Engram hot set: rank the table rows a corpus uses, simulate what a hot set
// of a given size saves on held-out text, and write the pack the engine pins in RAM (--engram-hot / STRATA_ENGRAM_HOT).
// CPU only (the hash depends on token ids alone, so ranking and simulation never touch the GPU; pack reads the tables).
//
// Corpus files (.i32): int32 token ids, -1 between documents (tools/v41/engram_corpus.py writes them).
//
//   ds41_engram_hot rank --gguf G --corpus train.i32 [--window 2048] --out rank.bin
//       Counts, for every (slot, row), how many corpus windows use it: a row repeated inside one window costs one read
//       anyway (the batch reader dedupes, the row cache keeps it), so cross-window reuse is what a pinned set buys.
//   ds41_engram_hot eval --gguf G --rank rank.bin --corpus heldout.i32 [--gib 0.5,1,2,4,8,16] [--cache-rows 2000000]
//                        [--decode-frac 0.25] [--chunk 4096]
//       Replays held-out documents as one server process would (row cache persists across requests, fill-until-full):
//       each document's first (1 - decode-frac) is prefilled in --chunk batches, the rest decoded token by token.
//       Prints disk row reads per decode token and per 1K prefill tokens for no hot set and each size.
//   ds41_engram_hot pack --gguf G --dir D1:D2 --rank rank.bin --gib 4 --out hot.bin [--threads 32]
//       Reads the top rows from the tables and writes the pack (format in ds41_engram.cpp, load_hot).
#include "ds41_engram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

std::vector<std::vector<int32_t>> load_docs(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "hot: cannot open %s\n", p.c_str()); std::exit(2); }
    std::vector<std::vector<int32_t>> docs(1);
    int32_t x;
    while (f.read(reinterpret_cast<char *>(&x), 4)) {
        if (x < 0) { if (!docs.back().empty()) docs.emplace_back(); }
        else docs.back().push_back(x);
    }
    if (docs.back().empty()) docs.pop_back();
    return docs;
}

std::vector<double> parse_list(const std::string & s) {
    std::vector<double> v;
    for (size_t a = 0, b; a <= s.size(); a = b + 1) {
        b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(std::atof(s.substr(a, b - a).c_str()));
    }
    return v;
}

// all (slot << 32 | row) keys of positions [p0, p1) of the document currently recorded in `eng`, appended to `out`
void keys_of(const Ds41Engram & eng, int nel, int64_t p0, int64_t p1, std::vector<uint64_t> & out) {
    const int64_t cols = eng.n_cols();
    std::vector<int64_t> ids((size_t) cols);
    for (int64_t p = p0; p < p1; ++p)
        for (int e = 0; e < nel; ++e) {
            eng.hash(p, e, ids.data());
            for (int64_t c = 0; c < cols; ++c) out.push_back(((uint64_t) e << 32) | (uint32_t) ids[(size_t) c]);
        }
}

void sort_unique(std::vector<uint64_t> & v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

struct Counted { uint64_t key; uint32_t count; };

// Rank file: "XENGRNK1" | u64 geometry fingerprint | u64 n | n x (u64 key, u32 count), count descending
void write_rank(const std::string & path, uint64_t fp, const std::vector<Counted> & v) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "hot: cannot write %s\n", path.c_str()); std::exit(1); }
    const uint64_t n = v.size();
    std::fwrite("XENGRNK1", 1, 8, f); std::fwrite(&fp, 8, 1, f); std::fwrite(&n, 8, 1, f);
    for (const Counted & c : v) { std::fwrite(&c.key, 8, 1, f); std::fwrite(&c.count, 4, 1, f); }
    if (std::fclose(f) != 0) { std::fprintf(stderr, "hot: write %s failed\n", path.c_str()); std::exit(1); }
}

std::vector<Counted> read_rank(const std::string & path, uint64_t fp) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "hot: cannot open %s\n", path.c_str()); std::exit(2); }
    char magic[8]; uint64_t rfp = 0, n = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "XENGRNK1", 8) != 0 || std::fread(&rfp, 8, 1, f) != 1 ||
        std::fread(&n, 8, 1, f) != 1) { std::fprintf(stderr, "hot: %s is not a rank file\n", path.c_str()); std::exit(2); }
    if (rfp != fp) { std::fprintf(stderr, "hot: %s was ranked for another geometry\n", path.c_str()); std::exit(2); }
    std::vector<Counted> v((size_t) n);
    for (Counted & c : v)
        if (std::fread(&c.key, 8, 1, f) != 1 || std::fread(&c.count, 4, 1, f) != 1) { std::fprintf(stderr, "hot: %s truncated\n", path.c_str()); std::exit(2); }
    std::fclose(f);
    return v;
}

// fold a sorted, unique-per-window key buffer into the running sorted (key, count) table
void fold(std::vector<uint64_t> & buf, std::vector<Counted> & acc) {
    std::sort(buf.begin(), buf.end());
    std::vector<Counted> run;
    for (size_t i = 0; i < buf.size();) {
        size_t j = i + 1;
        while (j < buf.size() && buf[j] == buf[i]) ++j;
        run.push_back({ buf[i], (uint32_t) (j - i) });
        i = j;
    }
    buf.clear();
    std::vector<Counted> m;
    m.reserve(acc.size() + run.size());
    size_t a = 0, b = 0;
    while (a < acc.size() || b < run.size()) {
        if (b == run.size() || (a < acc.size() && acc[a].key < run[b].key)) m.push_back(acc[a++]);
        else if (a == acc.size() || run[b].key < acc[a].key) m.push_back(run[b++]);
        else { m.push_back({ acc[a].key, acc[a].count + run[b].count }); ++a; ++b; }
    }
    acc.swap(m);
}

const int64_t kRowBytes = 256 + 8;   // dim + dim/32 (checked against the geometry at start)

size_t rows_for_gib(double gib) { return (size_t) (gib * 1024.0 * 1024 * 1024 / (double) (kRowBytes + 8)); }

int cmd_rank(Ds41Engram & eng, int nel, const std::string & corpus, int64_t window, const std::string & out) {
    const auto docs = load_docs(corpus);
    const double t0 = now_s();
    std::vector<Counted> acc;
    std::vector<uint64_t> buf, win;
    uint64_t ntok = 0, nwin = 0;
    for (const auto & d : docs) {
        for (int64_t p = 0; p < (int64_t) d.size(); ++p) eng.set_token(p, d[(size_t) p]);
        for (int64_t w0 = 0; w0 < (int64_t) d.size(); w0 += window) {
            win.clear();
            keys_of(eng, nel, w0, std::min<int64_t>((int64_t) d.size(), w0 + window), win);
            sort_unique(win);
            buf.insert(buf.end(), win.begin(), win.end());
            ++nwin;
            if (buf.size() > (size_t) 200000000) fold(buf, acc);
        }
        ntok += d.size();
    }
    fold(buf, acc);
    uint64_t once = 0;
    for (const Counted & c : acc) once += c.count == 1;
    std::vector<Counted> keep;
    keep.reserve(acc.size() - once);
    for (const Counted & c : acc) if (c.count >= 2) keep.push_back(c);   // used by one window only: no cross-window reuse
    std::stable_sort(keep.begin(), keep.end(), [](const Counted & x, const Counted & y) { return x.count > y.count; });
    write_rank(out, eng.geometry_fingerprint(), keep);
    std::printf("rank: %zu docs, %" PRIu64 " tokens, %" PRIu64 " windows of %" PRId64 " -> %zu distinct rows, %" PRIu64
                " used by one window only, %zu ranked (%.2f GiB as a pack) in %.1f s -> %s\n",
                docs.size(), ntok, nwin, window, acc.size(), once, keep.size(),
                (double) keep.size() * (kRowBytes + 8) / (1024.0 * 1024 * 1024), now_s() - t0, out.c_str());
    for (double g : { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 }) {
        const size_t k = std::min(rows_for_gib(g), keep.size());
        if (k) std::printf("  %5.2f GiB = top %zu rows: last kept row used by %u windows\n", g, k, keep[k - 1].count);
    }
    return 0;
}

int cmd_eval(Ds41Engram & eng, int nel, const std::string & rank_path, const std::string & corpus, std::vector<double> gibs,
             size_t cache_rows, double decode_frac, int64_t chunk) {
    const auto rank = read_rank(rank_path, eng.geometry_fingerprint());
    const auto docs = load_docs(corpus);
    // every held-out position's keys, per document, computed once and replayed for each configuration
    struct Doc { std::vector<uint64_t> keys; int64_t n = 0, split = 0; };
    std::vector<Doc> dv;
    const int64_t per = (int64_t) nel * eng.n_cols();
    uint64_t pre_tok = 0, dec_tok = 0;
    for (const auto & d : docs) {
        Doc x;
        x.n = (int64_t) d.size();
        x.split = std::max<int64_t>(1, (int64_t) ((double) x.n * (1.0 - decode_frac)));
        for (int64_t p = 0; p < x.n; ++p) eng.set_token(p, d[(size_t) p]);
        keys_of(eng, nel, 0, x.n, x.keys);
        pre_tok += (uint64_t) x.split; dec_tok += (uint64_t) (x.n - x.split);
        dv.push_back(std::move(x));
    }
    std::printf("eval: %zu held-out docs, %" PRIu64 " prefill + %" PRIu64 " decode tokens, %" PRId64 " lookups/token, "
                "row cache %zu rows (persistent, fill-until-full), prefill chunk %" PRId64 "\n",
                dv.size(), pre_tok, dec_tok, per, cache_rows, chunk);
    std::printf("%9s %10s %16s %18s %14s %14s\n", "hot GiB", "hot rows", "decode reads/tok", "prefill reads/1K", "decode hot %", "decode cache %");
    gibs.insert(gibs.begin(), 0.0);
    for (double g : gibs) {
        const size_t k = std::min(rows_for_gib(g), rank.size());
        std::vector<uint64_t> hot((size_t) k);
        for (size_t i = 0; i < k; ++i) hot[i] = rank[i].key;
        std::sort(hot.begin(), hot.end());
        auto in_hot = [&](uint64_t key) { return std::binary_search(hot.begin(), hot.end(), key); };
        std::unordered_set<uint64_t> cache;
        cache.reserve(cache_rows);
        uint64_t pre_reads = 0, dec_reads = 0, dec_hot = 0, dec_cache = 0, dec_unique = 0;
        std::vector<uint64_t> batch;
        // one batch: unique rows -> hot set, then row cache, else a disk read (which then fills the cache)
        auto run = [&](const uint64_t * kb, size_t nk, uint64_t & reads, bool dec) {
            batch.assign(kb, kb + nk);
            sort_unique(batch);
            for (uint64_t key : batch) {
                if (dec) ++dec_unique;
                if (in_hot(key)) { if (dec) ++dec_hot; continue; }
                if (cache.count(key)) { if (dec) ++dec_cache; continue; }
                ++reads;
                if (cache.size() < cache_rows) cache.insert(key);
            }
        };
        for (const Doc & d : dv) {
            for (int64_t p0 = 0; p0 < d.split; p0 += chunk) {
                const int64_t p1 = std::min(d.split, p0 + chunk);
                run(&d.keys[(size_t) (p0 * per)], (size_t) ((p1 - p0) * per), pre_reads, false);
            }
            for (int64_t p = d.split; p < d.n; ++p) run(&d.keys[(size_t) (p * per)], (size_t) per, dec_reads, true);
        }
        std::printf("%9.2f %10zu %16.2f %18.1f %13.1f%% %13.1f%%\n", g, k, dec_tok ? (double) dec_reads / dec_tok : 0.0,
                    pre_tok ? 1000.0 * pre_reads / pre_tok : 0.0, dec_unique ? 100.0 * dec_hot / dec_unique : 0.0,
                    dec_unique ? 100.0 * dec_cache / dec_unique : 0.0);
    }
    return 0;
}

int cmd_pack(Ds41Engram & eng, const std::string & rank_path, double gib, const std::string & out, int threads) {
    const auto rank = read_rank(rank_path, eng.geometry_fingerprint());
    const size_t k = std::min(rows_for_gib(gib), rank.size());
    std::vector<uint64_t> keys((size_t) k);
    for (size_t i = 0; i < k; ++i) keys[i] = rank[i].key;
    std::sort(keys.begin(), keys.end());   // ascending (slot, row) = ascending file offsets per table
    const int64_t rsz = eng.row_dim() + eng.row_dim() / 32;
    std::vector<uint8_t> pool((size_t) k * (size_t) rsz);
    std::atomic<size_t> next { 0 }, done { 0 };
    std::atomic<int> bad { 0 };
    const double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < std::max(1, threads); ++t)
        th.emplace_back([&] {
            for (;;) {
                const size_t a = next.fetch_add(4096);
                if (a >= k) break;
                const size_t b = std::min(k, a + 4096);
                for (size_t i = a; i < b; ++i)
                    if (!eng.read_raw((int) (keys[i] >> 32), (uint32_t) keys[i], &pool[i * (size_t) rsz])) bad.fetch_add(1);
                done.fetch_add(b - a);
            }
        });
    std::thread prog([&] {
        while (done.load() < k && !bad.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            const double dt = now_s() - t0;
            std::fprintf(stderr, "pack: %zu / %zu rows, %.0f rows/s\n", done.load(), k, done.load() / std::max(dt, 1e-9));
        }
    });
    for (auto & x : th) x.join();
    prog.join();
    if (bad.load()) { std::fprintf(stderr, "pack: %d row reads failed\n", bad.load()); return 1; }
    const std::string tmp = out + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "pack: cannot write %s\n", tmp.c_str()); return 1; }
    struct Hdr { char magic[8]; uint32_t version, slots, dim, srsz; uint64_t fp, count; } h {};
    std::memcpy(h.magic, "XENGHOT1", 8);
    h.version = 1; h.slots = 2; h.dim = (uint32_t) eng.row_dim(); h.srsz = (uint32_t) (eng.row_dim() / 32);
    h.fp = eng.geometry_fingerprint(); h.count = k;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(keys.data(), 8, k, f) == k &&
              std::fwrite(pool.data(), 1, pool.size(), f) == pool.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), out.c_str()) != 0) { std::fprintf(stderr, "pack: write failed\n"); return 1; }
    std::printf("pack: %zu rows (%.2f GiB) in %.1f s -> %s\n", k, (double) pool.size() / (1024.0 * 1024 * 1024), now_s() - t0, out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds41_engram_hot rank|eval|pack --gguf G ... (see the header of ds41_engram_hot.cpp)\n");
        return 2;
    }
    const std::string cmd = argv[1];
    std::string gguf, dir, corpus, rank, out, gibs = "0.5,1,2,4,8,16";
    int64_t window = 2048, chunk = 4096;
    size_t cache_rows = 2000000;
    double gib = 4.0, decode_frac = 0.25;
    int threads = 32;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto nxt = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (k == "--gguf") gguf = nxt();
        else if (k == "--dir") dir = nxt();
        else if (k == "--corpus") corpus = nxt();
        else if (k == "--rank") rank = nxt();
        else if (k == "--out") out = nxt();
        else if (k == "--window") window = std::atoll(nxt().c_str());
        else if (k == "--chunk") chunk = std::atoll(nxt().c_str());
        else if (k == "--cache-rows") cache_rows = (size_t) std::atoll(nxt().c_str());
        else if (k == "--decode-frac") decode_frac = std::atof(nxt().c_str());
        else if (k == "--gib") { gibs = nxt(); gib = std::atof(gibs.c_str()); }
        else if (k == "--threads") threads = std::atoi(nxt().c_str());
        else { std::fprintf(stderr, "hot: unknown arg %s\n", k.c_str()); return 2; }
    }
    if (gguf.empty()) { std::fprintf(stderr, "hot: --gguf is required (any V4.1 GGUF: only the Engram metadata is read)\n"); return 2; }
    GgufModel m = GgufModel::open(gguf);
    Ds41Geometry G;
    const std::string ge = ds41_read_geometry(m, G);
    if (!ge.empty()) { std::fprintf(stderr, "hot: geometry: %s\n", ge.c_str()); return 1; }
    if (G.engram_dim + G.engram_dim / 32 != kRowBytes || G.engram_layers.size() != 2) {
        std::fprintf(stderr, "hot: unexpected Engram shape (dim %" PRId64 ", %zu slots)\n", G.engram_dim, G.engram_layers.size());
        return 1;
    }
    const int nel = (int) G.engram_layers.size();
    Ds41Engram eng;
    if (cmd == "pack") {
        if (dir.empty() || rank.empty() || out.empty()) { std::fprintf(stderr, "hot: pack needs --dir --rank --out\n"); return 2; }
        std::vector<std::string> dirs;
        for (size_t a = 0, b; a <= dir.size(); a = b + 1) { b = dir.find(':', a); if (b == std::string::npos) b = dir.size(); if (b > a) dirs.push_back(dir.substr(a, b - a)); }
        std::string e;
        ::setenv("STRATA_ENGRAM_CACHE_ROWS", "0", 1);   // no row cache / hot set in the reader we pack from
        ::setenv("STRATA_ENGRAM_HOT", "0", 1);
        if (!eng.open(G, dirs, 1, e)) { std::fprintf(stderr, "hot: engram open: %s\n", e.c_str()); return 1; }
        return cmd_pack(eng, rank, gib, out, threads);
    }
    eng.setup_hash(G);
    if (cmd == "rank") {
        if (corpus.empty() || out.empty()) { std::fprintf(stderr, "hot: rank needs --corpus --out\n"); return 2; }
        return cmd_rank(eng, nel, corpus, window, out);
    }
    if (cmd == "eval") {
        if (corpus.empty() || rank.empty()) { std::fprintf(stderr, "hot: eval needs --corpus --rank\n"); return 2; }
        return cmd_eval(eng, nel, rank, corpus, parse_list(gibs), cache_rows, decode_frac, chunk);
    }
    std::fprintf(stderr, "hot: unknown command %s\n", cmd.c_str());
    return 2;
}
