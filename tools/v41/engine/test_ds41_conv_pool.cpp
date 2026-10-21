// test_ds41_conv_pool - the conversation pool's storage on the CPU (no CUDA call: the module is compiled without
// DS41_POOL_CUDA and "device" buffers are host memory): byte-exact save / restore, partial restore (from > 0), shared
// prefix chunks, incremental saves, LRU under a budget, pinning, boundaries, identities, error paths, accounting.
#include "ds41_conv_pool.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::v41;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d %s  ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// the real 1017 --kv-fp4 shape, scaled down in positions: comp NVFP4 288 B / idxk MXFP4 68 B rows of the 4 kv sources
// (L2, L8, L14 ratio 2 + L20 ratio 1 is one plausible split; the pool does not care), 40 ring tensors, 6 states
static Ds41PoolLayout make_layout(int64_t ctx) {
    Ds41PoolLayout L;
    const int ratio[4] = { 2, 2, 2, 1 };
    for (int s = 0; s < 4; ++s) {
        L.streams.push_back({ "comp" + std::to_string(s), 288, ratio[s], ctx / ratio[s] });
        L.streams.push_back({ "idxk" + std::to_string(s), 68, ratio[s], ctx / ratio[s] });
    }
    for (int l = 0; l < 40; ++l) L.regions.push_back({ "raw" + std::to_string(l), 512 * 144 * 2 });
    for (int l = 0; l < 6; ++l) L.regions.push_back({ "st" + std::to_string(l), 512 * 2 * 4 });
    L.max_positions = ctx;
    return L;
}

struct Dev {   // one "device" state: a buffer per stream (max_rows + 1 spare) and per region
    std::vector<std::vector<uint8_t>> s, r;
    Ds41PoolTargets t;
    Dev(const Ds41PoolLayout & L, uint32_t seed) {
        std::mt19937 g(seed);
        for (const auto & x : L.streams) { s.emplace_back((size_t) (x.max_rows + 1) * x.row_bytes); for (auto & b : s.back()) b = (uint8_t) g(); }
        for (const auto & x : L.regions) { r.emplace_back(x.bytes); for (auto & b : r.back()) b = (uint8_t) g(); }
        bind(L.fingerprint());
    }
    Dev(const Dev & o) : s(o.s), r(o.r) { bind(o.t.layout_fp); }
    Dev & operator=(const Dev &) = delete;
    void bind(uint64_t fp) {
        t = {};
        for (auto & v : s) t.stream.push_back(v.data());
        for (auto & v : r) t.region.push_back(v.data());
        t.layout_fp = fp;
    }
    void scramble(uint32_t seed) {
        std::mt19937 g(seed);
        for (auto & v : s) for (auto & b : v) b = (uint8_t) g();
        for (auto & v : r) for (auto & b : v) b = (uint8_t) g();
    }
    // copy the stream rows of positions [0, n) of o into this (a conversation sharing o's prefix)
    void share_prefix(const Dev & o, const Ds41PoolLayout & L, int64_t n) {
        for (size_t i = 0; i < s.size(); ++i)
            std::memcpy(s[i].data(), o.s[i].data(), (size_t) (n / L.streams[i].ratio) * L.streams[i].row_bytes);
    }
};

// rows of positions [from, n) and the regions of `got` equal `want`; rows outside [from/r, n/r) equal `outside`
static bool same(const Dev & got, const Dev & want, const Dev * outside, const Ds41PoolLayout & L, int64_t from, int64_t n,
                 bool regions = true) {
    bool ok = true;
    for (size_t i = 0; i < L.streams.size(); ++i) {
        const size_t rb = L.streams[i].row_bytes;
        const int64_t r = L.streams[i].ratio, a = from / r, e = n / r;
        if (std::memcmp(got.s[i].data() + a * rb, want.s[i].data() + a * rb, (size_t) (e - a) * rb)) { ok = false; std::printf("  stream %zu rows [%lld,%lld) differ\n", i, (long long) a, (long long) e); }
        if (outside) {
            if (a && std::memcmp(got.s[i].data(), outside->s[i].data(), (size_t) a * rb)) { ok = false; std::printf("  stream %zu wrote below row %lld\n", i, (long long) a); }
            const size_t tail = got.s[i].size() - (size_t) e * rb;
            if (std::memcmp(got.s[i].data() + e * rb, outside->s[i].data() + e * rb, tail)) { ok = false; std::printf("  stream %zu wrote past row %lld\n", i, (long long) e); }
        }
    }
    if (regions)
        for (size_t i = 0; i < L.regions.size(); ++i)
            if (got.r[i] != want.r[i]) { ok = false; std::printf("  region %zu differs\n", i); }
    return ok;
}

static std::vector<int32_t> toks(int64_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<int32_t> t((size_t) n);
    for (auto & x : t) x = (int32_t) (g() % 129280);
    return t;
}

int main() {
    const int64_t ctx = 32768;
    const Ds41PoolLayout L = make_layout(ctx);
    std::string err;
    std::printf("layout: %zu streams, %zu regions, %zu B/position, %zu B regions, fp %016llx\n", L.streams.size(),
                L.regions.size(), L.bytes_per_position(), L.region_bytes(), (unsigned long long) L.fingerprint());
    CHECK(L.bytes_per_position() == 890, "bpp %zu", L.bytes_per_position());

    Ds41PoolConfig cfg;
    cfg.cuda = false;
    cfg.slab_bytes = 256 << 10;          // small slabs: many chunks at test sizes (C = 294 positions)
    cfg.extent_bytes = 4 << 20;
    cfg.budget_bytes = (size_t) 64 << 20;
    cfg.verify = true;
    cfg.max_boundaries = 3;

    // ---- 1. byte-exact save / restore, no writes outside the restored rows
    {
        Ds41ConvPool pool;
        CHECK(pool.init(L, cfg, err), "%s", err.c_str());
        const int64_t C = (int64_t) pool.chunk_positions();
        std::printf("chunk %lld positions (%zu B used of %zu)\n", (long long) C, (size_t) C * 890, cfg.slab_bytes);
        CHECK(C % 2 == 0 && C * 890 <= (int64_t) cfg.slab_bytes, "C %lld", (long long) C);
        for (int64_t n : { (int64_t) 1, (int64_t) 2, (int64_t) 3, C - 1, C, C + 1, 2 * C + 7, (int64_t) 5001 }) {
            Dev d(L, 1000 + (uint32_t) n), keep = d;
            const auto t = toks(n, (uint32_t) n);
            std::vector<float> lg(129280);
            for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float) i * 0.5f + (float) n;
            const Ds41PoolId id = pool.save(d.t, t.data(), nullptr, n, lg.data(), lg.size(), err);
            CHECK(id, "save %lld: %s", (long long) n, err.c_str());
            d.scramble(77);
            const Dev scr = d;
            Ds41PoolRestore rr;
            const auto m = pool.best_prefix(t.data(), nullptr, n);
            CHECK(m.rec == id && m.pos == n && m.lcp == n && m.has_logits, "match rec %llu pos %lld", (unsigned long long) m.rec, (long long) m.pos);
            CHECK(pool.restore(id, m.boundary, 0, d.t, rr, err), "restore %lld: %s", (long long) n, err.c_str());
            CHECK(pool.wait(rr), "wait");
            CHECK(same(d, keep, &scr, L, 0, n), "n=%lld byte-exact", (long long) n);
            CHECK(rr.logits == lg, "logits n=%lld", (long long) n);
            CHECK(rr.pos == n, "pos");
        }
        const auto s = pool.stats();
        CHECK(s.used_bytes <= s.budget_bytes, "budget");
        CHECK(s.restores == 8 && s.restores_failed == 0, "restores %llu", (unsigned long long) s.restores);
        std::printf("1. %s\n", pool.stats_line().c_str());
    }

    // ---- 2. partial restore (from > 0): only rows of [from, pos) are written
    {
        Ds41ConvPool pool;
        CHECK(pool.init(L, cfg, err), "%s", err.c_str());
        const int64_t n = 3001;
        Dev d(L, 5), keep = d;
        const auto t = toks(n, 9);
        const Ds41PoolId id = pool.save(d.t, t.data(), nullptr, n, nullptr, 0, err);
        CHECK(id, "%s", err.c_str());
        for (int64_t from : { (int64_t) 0, (int64_t) 1, (int64_t) 293, (int64_t) 294, (int64_t) 295, (int64_t) 2000, (int64_t) 3000, n }) {
            d.scramble(100 + (uint32_t) from);
            const Dev scr = d;
            Ds41PoolRestore rr;
            CHECK(pool.restore(id, 0, from, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
            CHECK(same(d, keep, &scr, L, from, n), "from=%lld", (long long) from);
            CHECK(rr.logits.empty(), "no logits");
        }
    }

    // ---- 3. shared prefix chunks + incremental saves + inherited boundaries + identities
    {
        Ds41ConvPool pool;
        CHECK(pool.init(L, cfg, err), "%s", err.c_str());
        const int64_t C = (int64_t) pool.chunk_positions();
        const int64_t nA = 5000, nB = 6000, shareAB = 3000;
        auto tA = toks(nA, 1), tB = toks(nB, 2);
        std::copy(tA.begin(), tA.begin() + shareAB, tB.begin());
        Dev dA(L, 11), dB(L, 12);
        dB.share_prefix(dA, L, shareAB);
        const Dev kA = dA, kB = dB;
        const Ds41PoolId a = pool.save(dA.t, tA.data(), nullptr, nA, nullptr, 0, err);
        const auto s0 = pool.stats();
        const Ds41PoolId b = pool.save(dB.t, tB.data(), nullptr, nB, nullptr, 0, err);
        const auto s1 = pool.stats();
        CHECK(a && b, "%s", err.c_str());
        CHECK(s1.shared_chunks - s0.shared_chunks == (uint64_t) (shareAB / C), "shared %llu want %lld",
              (unsigned long long) (s1.shared_chunks - s0.shared_chunks), (long long) (shareAB / C));
        const size_t region_slabs = (L.region_bytes() + cfg.slab_bytes - 1) / cfg.slab_bytes;
        CHECK(s1.slabs_used - s0.slabs_used == (size_t) ((nB + C - 1) / C - shareAB / C) + region_slabs, "B took %zu slabs",
              s1.slabs_used - s0.slabs_used);
        Dev d(L, 13);
        Ds41PoolRestore rr;
        CHECK(pool.restore(b, 0, 0, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
        CHECK(same(d, kB, nullptr, L, 0, nB), "B from shared chunks");
        CHECK(pool.restore(a, 0, 0, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
        CHECK(same(d, kA, nullptr, L, 0, nA), "A still intact");

        // the same conversation one turn on: A + 1500 tokens, the device rows past nA rewritten, new rings
        const int64_t nA2 = 6500;
        auto tA2 = tA; auto t2 = toks(nA2 - nA, 3); tA2.insert(tA2.end(), t2.begin(), t2.end());
        Dev dA2 = kA;
        { Dev x(L, 21); for (size_t i = 0; i < L.streams.size(); ++i) {
              const size_t rb = L.streams[i].row_bytes; const int64_t r = L.streams[i].ratio;
              std::memcpy(dA2.s[i].data() + (nA / r) * rb, x.s[i].data() + (nA / r) * rb, (size_t) ((nA2 / r) - (nA / r)) * rb); }
          for (size_t i = 0; i < x.r.size(); ++i) dA2.r[i] = x.r[i]; }
        const Dev kA2 = dA2;
        const auto s2 = pool.stats();
        const Ds41PoolId a2 = pool.save(dA2.t, tA2.data(), nullptr, nA2, nullptr, 0, err);
        const auto s3 = pool.stats();
        CHECK(a2 && a2 != a, "%s", err.c_str());
        CHECK(s3.superseded == s2.superseded + 1 && s3.records == s2.records, "A superseded (records %zu -> %zu)", s2.records, s3.records);
        CHECK(s3.bytes_d2h - s2.bytes_d2h < (uint64_t) (2 * C + (nA2 - nA)) * 890 + L.region_bytes(), "incremental bytes %llu",
              (unsigned long long) (s3.bytes_d2h - s2.bytes_d2h));
        const auto recs = pool.records();
        for (const auto & r : recs) if (r.id == a2) CHECK(r.boundaries.size() == 2 && r.boundaries[0].pos == nA && r.boundaries[1].pos == nA2, "boundaries");
        // a prompt that continues A's first turn but not the second: boundary nA of a2, with next = nA2
        auto tq = tA; tq.push_back(5); tq.push_back(6);
        auto m = pool.best_prefix(tq.data(), nullptr, (int64_t) tq.size());
        CHECK(m.rec == a2 && m.pos == nA && m.lcp == nA && m.next_pos == nA2, "turn-1 match rec %llu pos %lld lcp %lld next %lld",
              (unsigned long long) m.rec, (long long) m.pos, (long long) m.lcp, (long long) m.next_pos);
        auto tq2 = tA2; tq2.resize(nA + 100);  // the second turn partly (lcp inside turn 2): boundary nA, next nA2
        m = pool.best_prefix(tq2.data(), nullptr, (int64_t) tq2.size());
        CHECK(m.rec == a2 && m.pos == nA && m.lcp == nA + 100 && m.next_pos == nA2, "partial match next %lld", (long long) m.next_pos);
        CHECK(pool.restore(a2, 0, 0, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
        CHECK(same(d, kA, nullptr, L, 0, nA), "turn-1 boundary of a2 = A's state");
        CHECK(pool.restore(a2, 1, 0, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
        CHECK(same(d, kA2, nullptr, L, 0, nA2), "turn-2 boundary");
        // identities: same tokens, an image identity at position 100 -> the match stops there (no boundary <= 100)
        std::vector<uint64_t> ids(tA.size(), 0); ids[100] = 0xabcdef01;
        m = pool.best_prefix(tA.data(), ids.data(), (int64_t) tA.size());
        CHECK(m.pos == 0, "identity mismatch must not match (pos %lld)", (long long) m.pos);
        const Ds41PoolId ai = pool.save(kA.t, tA.data(), ids.data(), nA, nullptr, 0, err);
        CHECK(ai && ai != a2, "save with identities");
        m = pool.best_prefix(tA.data(), ids.data(), (int64_t) tA.size());
        CHECK(m.rec == ai && m.pos == nA, "identity match");
        m = pool.best_prefix(tA.data(), nullptr, (int64_t) tA.size());
        CHECK(m.rec == a2 && m.pos == nA, "plain tokens still match the plain record");
        // exact re-save: touched, no copy
        const auto s4 = pool.stats();
        CHECK(pool.save(kA2.t, tA2.data(), nullptr, nA2, nullptr, 0, err) == a2, "re-save touched");
        const auto s5 = pool.stats();
        CHECK(s5.saves_touched == s4.saves_touched + 1 && s5.bytes_d2h == s4.bytes_d2h, "touch copies nothing");
        // a prefix of a stored record with a boundary there: no copy at all
        const Ds41PoolId ap = pool.save(kA.t, tA.data(), nullptr, nA, nullptr, 0, err);
        const auto s6 = pool.stats();
        CHECK(ap && s6.bytes_d2h == s5.bytes_d2h, "prefix save is copy-free (%llu B)", (unsigned long long) (s6.bytes_d2h - s5.bytes_d2h));
        pool.clear(false);
        const auto s7 = pool.stats();
        CHECK(s7.slabs_used == 0 && s7.host_bytes == 0 && s7.records == 0, "clear frees everything (slabs %zu host %zu)", s7.slabs_used, s7.host_bytes);
    }

    // ---- 4. LRU under the budget, pinning, boundary cap
    {
        Ds41PoolConfig c2 = cfg;
        c2.budget_bytes = (size_t) 48 << 20;   // 192 slabs of 256 KiB: 5 records of 37 slabs
        c2.verify = false;
        Ds41ConvPool pool;
        CHECK(pool.init(L, c2, err), "%s", err.c_str());
        const int64_t n = 4000;               // ~14 chunk slabs + 23 region slabs + 16 KB host = ~37 slabs a record
        Dev d(L, 31);
        std::vector<Ds41PoolId> ids;
        std::vector<std::vector<int32_t>> ts;
        for (int i = 0; i < 6; ++i) {
            ts.push_back(toks(n, 400 + (uint32_t) i));
            ids.push_back(pool.save(d.t, ts.back().data(), nullptr, n, nullptr, 0, err));
            CHECK(ids.back(), "save %d: %s", i, err.c_str());
            const auto s = pool.stats();
            CHECK(s.used_bytes <= s.budget_bytes, "budget %zu > %zu", s.used_bytes, s.budget_bytes);
            if (i == 0) {   // pin the first conversation's boundary as the shared root
                const Ds41PoolId root = pool.pin_root(ids[0], 0, err);
                CHECK(root, "pin_root: %s", err.c_str());
                ids[0] = root;
            }
            if (i == 2) {   // touch record 1: record 2... order of LRU changes
                Ds41PoolRestore rr;
                CHECK(pool.restore(ids[1], 0, 0, d.t, rr, err) && pool.wait(rr), "%s", err.c_str());
            }
        }
        const auto recs = pool.records();
        auto alive = [&](Ds41PoolId id) { for (const auto & r : recs) if (r.id == id) return true; return false; };
        CHECK(alive(ids[0]), "pinned root survives");
        CHECK(alive(ids[5]), "newest survives");
        CHECK(!alive(ids[2]) || alive(ids[1]), "LRU: record 1 (touched) outlives record 2");
        const auto s = pool.stats();
        CHECK(s.evictions >= 2, "evictions %llu", (unsigned long long) s.evictions);
        std::printf("4. %s\n", pool.stats_line().c_str());
        // the pinned share (half the budget): the root + one more record fit, a third pin is refused
        CHECK(pool.pin(ids[5], true), "second pin fits");
        bool refused = false;
        for (const auto & r : pool.records()) if (!r.pinned) refused |= !pool.pin(r.id, true);
        CHECK(refused, "pinned share cap");
        // boundary cap: one conversation, 5 turns -> first + newest 2 (max_boundaries 3)
        pool.clear(false);
        std::vector<int32_t> t = toks(500, 77);
        Ds41PoolId last = 0;
        for (int turn = 0; turn < 5; ++turn) {
            auto more = toks(300, 900 + (uint32_t) turn);
            t.insert(t.end(), more.begin(), more.end());
            last = pool.save(d.t, t.data(), nullptr, (int64_t) t.size(), nullptr, 0, err);
            CHECK(last, "%s", err.c_str());
        }
        for (const auto & r : pool.records())
            if (r.id == last) {
                CHECK(r.boundaries.size() == 3 && r.boundaries[0].pos == 800 && r.boundaries[2].pos == 2000,
                      "cap: %zu boundaries (first %lld last %lld)", r.boundaries.size(),
                      (long long) (r.boundaries.empty() ? -1 : r.boundaries[0].pos),
                      (long long) (r.boundaries.empty() ? -1 : r.boundaries.back().pos));
            }
        CHECK(pool.records().size() == 1, "the turns superseded each other (%zu records)", pool.records().size());
    }

    // ---- 5. error paths leave the device and the pool untouched
    {
        Ds41ConvPool pool;
        CHECK(pool.init(L, cfg, err), "%s", err.c_str());
        Dev d(L, 41);
        const auto t = toks(1000, 5);
        const Ds41PoolId id = pool.save(d.t, t.data(), nullptr, 1000, nullptr, 0, err);
        CHECK(id, "%s", err.c_str());
        d.scramble(3);
        const Dev before = d;
        Ds41PoolRestore rr;
        Ds41PoolTargets bad = d.t; bad.layout_fp ^= 1;
        CHECK(!pool.restore(id, 0, 0, bad, rr, err) && !rr.live_clobbered, "wrong fingerprint refused");
        std::printf("   refused: %s\n", err.c_str());
        bad = d.t; bad.region.pop_back();
        CHECK(!pool.restore(id, 0, 0, bad, rr, err) && !rr.live_clobbered, "wrong shape refused");
        bad = d.t; bad.stream[3] = nullptr;
        CHECK(!pool.restore(id, 0, 0, bad, rr, err), "null pointer refused");
        CHECK(!pool.restore(id, 1, 0, d.t, rr, err), "no such boundary");
        CHECK(!pool.restore(id, 0, 1001, d.t, rr, err), "from past the boundary");
        CHECK(!pool.restore(id + 99, 0, 0, d.t, rr, err), "no such record");
        CHECK(same(d, before, &before, L, 0, 0) && d.r == before.r, "device untouched by refused restores");
        const auto s0 = pool.stats();
        CHECK(!pool.save(d.t, t.data(), nullptr, ctx + 1, nullptr, 0, err), "save past max_positions refused");
        CHECK(!pool.save(bad, t.data(), nullptr, 1000, nullptr, 0, err), "save with a null pointer refused");
        const auto s1 = pool.stats();
        CHECK(s1.slabs_used == s0.slabs_used && s1.records == s0.records && s1.host_bytes == s0.host_bytes, "failed saves change nothing");
        // a save larger than the whole budget fails cleanly
        Ds41PoolConfig c3 = cfg; c3.budget_bytes = 4 << 20;
        Ds41ConvPool small;
        CHECK(small.init(L, c3, err), "%s", err.c_str());
        const auto tl = toks(20000, 6);
        Dev dl(L, 7);
        CHECK(!small.save(dl.t, tl.data(), nullptr, 20000, nullptr, 0, err), "over-budget save refused");
        std::printf("   refused: %s\n", err.c_str());
        CHECK(small.stats().slabs_used == 0, "over-budget save left no slabs");
        // verify mode catches a corrupted slab (host memory scribbled behind the pool's back is not possible through
        // the API; the checksum path is exercised by every verified restore above)
    }

    std::printf(fails ? "test_ds41_conv_pool: %d FAILED\n" : "test_ds41_conv_pool: all passed\n", fails);
    return fails ? 1 : 0;
}
