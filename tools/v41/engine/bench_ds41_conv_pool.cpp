// bench_ds41_conv_pool <model.gguf> [--ctx 262144] [--budget-mib 8192] [--sizes 2048,8192,32768,65536,131072]
//                      [--reps 5] [--no-fp4] [--fill 32768 (0 = skip E)] [--sys 8192] [--slab-kib 2048] [--prealloc]
//                      (--sizes 0 skips A/B/C)
//
// The conversation pool on the card, with the model's real state shapes (geometry from the GGUF: kv source layers,
// ratios, d_head / idx_dim, the window ring; --kv-fp4 row sizes NVFP4 288 B / MXFP4 68 B).  No weights are loaded: the
// device caches are plain cudaMalloc buffers of the engine's sizes filled with a pattern.  Measures:
//   A. park/unpark as main does it (pageable host vectors, synchronous copies) - the baseline
//   B. pool save (D2H into pinned slabs, async) / restore (H2D + wait) of n positions: host enqueue, device time
//      (CUDA events), wall; byte-exact check of every restore against the pattern
//   C. incremental save (+1024 positions on the same conversation) and partial restore (from = n - 4096)
//   D. overlap: a save while a 100 ms kernel runs on another stream (does the copy wait for compute? does the kernel slow?)
//   E. fill: how many conversations of --fill positions the budget holds, without and with a shared --sys prefix;
//      best_prefix time over all of them; a 4-agent round-robin switch (save current + restore next)
// Run ONLY inside an aictl window on the reserved card (CUDA_VISIBLE_DEVICES=0): the miner lives there otherwise.
#include "ds41_conv_pool.hpp"
#include "ds41_geometry.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); std::exit(3); } } while (0)

// bench_ds41_conv_pool_spin.cu: a kernel that keeps every SM busy for `cycles` clocks
extern "C" cudaError_t ds41_pool_bench_spin(long long cycles, int blocks, cudaStream_t s);

struct DevState {
    std::vector<void *> s, r;              // device buffers (stream rows incl. the spare, regions)
    std::vector<size_t> sbytes, rbytes;
    Ds41PoolTargets t;
};

static uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }

// a deterministic pattern per (buffer, byte offset, generation)
static void pattern(uint8_t * p, size_t n, uint64_t buf, uint64_t gen) {
    for (size_t i = 0; i < n; i += 8) {
        const uint64_t v = mix((buf << 40) ^ (gen << 56) ^ (i >> 3));
        std::memcpy(p + i, &v, std::min<size_t>(8, n - i));
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_ds41_conv_pool <model.gguf> [options]\n"); return 2; }
    int64_t ctx = 262144, fill_n = 32768, sys_n = 8192;
    size_t budget_mib = 8192, slab_kib = 2048;
    int reps = 5;
    bool fp4 = true, prealloc = false;
    std::vector<int64_t> sizes = { 2048, 8192, 32768, 65536, 131072 };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--ctx") ctx = std::atoll(nx().c_str());
        else if (a == "--budget-mib") budget_mib = (size_t) std::atoll(nx().c_str());
        else if (a == "--reps") reps = std::max(2, std::atoi(nx().c_str()));
        else if (a == "--slab-kib") slab_kib = (size_t) std::atoll(nx().c_str());
        else if (a == "--no-fp4") fp4 = false;
        else if (a == "--prealloc") prealloc = true;
        else if (a == "--fill") fill_n = std::atoll(nx().c_str());
        else if (a == "--sys") sys_n = std::atoll(nx().c_str());
        else if (a == "--sizes") {
            sizes.clear();
            std::string v = nx();
            for (size_t p = 0; p < v.size();) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); sizes.push_back(std::atoll(v.substr(p, q - p).c_str())); p = q + 1; }
        } else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    // ---- geometry -> the engine's state shapes (ds41_dense.cpp alloc: comp [DH, cmax+1], idxk [ID, cmax+1],
    // raw / raw_snap [DH, SWA+16] BF16 (act_quant), st_kv/st_sc(_snap) [DH, 2] F32 on ratio-2 kv sources)
    GgufModel m = GgufModel::open(argv[1]);
    Ds41Geometry G;
    std::string err = ds41_read_geometry(m, G);
    if (!err.empty()) { std::fprintf(stderr, "geometry: %s\n", err.c_str()); return 1; }
    const int64_t DH = G.d_head, ID = G.idx_dim, RAW = G.n_swa + 16;
    const size_t comp_row = fp4 ? (size_t) DH / 64 * 36 : (size_t) DH * 2, idx_row = fp4 ? (size_t) ID / 32 * 17 : (size_t) ID * 2;
    Ds41PoolLayout lay;
    std::vector<size_t> s_alloc;
    for (int64_t l = 0; l < G.n_layer; ++l) {
        if (!G.is_kv_source(l)) continue;
        const int r = (int) G.ratio[(size_t) l];
        const int64_t cmax = ctx / r;
        lay.streams.push_back({ "comp" + std::to_string(l), comp_row, r, cmax });
        lay.streams.push_back({ "idxk" + std::to_string(l), idx_row, r, cmax });
        s_alloc.push_back((size_t) (cmax + 1) * comp_row);
        s_alloc.push_back((size_t) (cmax + 1) * idx_row);
    }
    for (int64_t l = 0; l < G.n_layer; ++l) lay.regions.push_back({ "raw" + std::to_string(l), (size_t) (DH * RAW * 2) });
    for (int64_t l = 0; l < G.n_layer; ++l)
        if (G.is_kv_source(l) && G.ratio[(size_t) l] == 2) {
            lay.regions.push_back({ "st_kv" + std::to_string(l), (size_t) (DH * 2 * 4) });
            lay.regions.push_back({ "st_sc" + std::to_string(l), (size_t) (DH * 2 * 4) });
        }
    lay.max_positions = ctx;
    std::printf("# geometry: %lld layers, kv sources:", (long long) G.n_layer);
    for (int64_t l : G.kv_sources) std::printf(" L%lld(r%lld)", (long long) l, (long long) G.ratio[(size_t) l]);
    std::printf("; rows comp %zu B idxk %zu B; %zu B/position; regions %zu (%.2f MiB); ctx %lld\n", comp_row, idx_row,
                lay.bytes_per_position(), lay.regions.size(), lay.region_bytes() / 1048576.0, (long long) ctx);

    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("# device %s, PCIe %02x:%02x, %d copy engines\n", prop.name, prop.pciBusID, prop.pciDeviceID, prop.asyncEngineCount);

    // ---- device state + its reference pattern (generation 1) in pinned host memory
    DevState D;
    std::vector<std::vector<uint8_t>> ref_s, ref_r;
    for (size_t i = 0; i < lay.streams.size(); ++i) {
        void * p; CK(cudaMalloc(&p, s_alloc[i])); D.s.push_back(p); D.sbytes.push_back(s_alloc[i]);
        ref_s.emplace_back(s_alloc[i]); pattern(ref_s.back().data(), s_alloc[i], i, 1);
        CK(cudaMemcpy(p, ref_s.back().data(), s_alloc[i], cudaMemcpyHostToDevice));
    }
    for (size_t i = 0; i < lay.regions.size(); ++i) {
        void * p; CK(cudaMalloc(&p, lay.regions[i].bytes)); D.r.push_back(p); D.rbytes.push_back(lay.regions[i].bytes);
        ref_r.emplace_back(lay.regions[i].bytes); pattern(ref_r.back().data(), lay.regions[i].bytes, 1000 + i, 1);
        CK(cudaMemcpy(p, ref_r.back().data(), lay.regions[i].bytes, cudaMemcpyHostToDevice));
    }
    D.t.stream = D.s; D.t.region = D.r; D.t.layout_fp = lay.fingerprint();
    size_t dev_bytes = 0; for (size_t b : D.sbytes) dev_bytes += b; for (size_t b : D.rbytes) dev_bytes += b;
    std::printf("# device state %.1f MiB\n", dev_bytes / 1048576.0);

    auto scramble = [&](int64_t from_pos, int64_t to_pos) {   // overwrite rows of [from, to) + regions with 0x5A
        for (size_t i = 0; i < lay.streams.size(); ++i) {
            const auto & sp = lay.streams[i];
            const int64_t a = from_pos / sp.ratio, e = std::min<int64_t>(to_pos / sp.ratio + 1, sp.max_rows + 1);
            CK(cudaMemset((uint8_t *) D.s[i] + a * sp.row_bytes, 0x5A, (size_t) (e - a) * sp.row_bytes));
        }
        for (size_t i = 0; i < lay.regions.size(); ++i) CK(cudaMemset(D.r[i], 0x5A, D.rbytes[i]));
        CK(cudaDeviceSynchronize());
    };
    // device rows of [0, n) + regions vs the reference (generation of each part given by refs)
    std::vector<uint8_t> chk;
    auto verify = [&](int64_t n) -> bool {
        bool ok = true;
        for (size_t i = 0; i < lay.streams.size(); ++i) {
            const auto & sp = lay.streams[i];
            const size_t b = (size_t) (n / sp.ratio) * sp.row_bytes;
            chk.resize(b);
            CK(cudaMemcpy(chk.data(), D.s[i], b, cudaMemcpyDeviceToHost));
            if (std::memcmp(chk.data(), ref_s[i].data(), b)) { ok = false; std::printf("#   MISMATCH stream %s\n", sp.name.c_str()); }
        }
        for (size_t i = 0; i < lay.regions.size(); ++i) {
            chk.resize(D.rbytes[i]);
            CK(cudaMemcpy(chk.data(), D.r[i], D.rbytes[i], cudaMemcpyDeviceToHost));
            if (std::memcmp(chk.data(), ref_r[i].data(), D.rbytes[i])) { ok = false; std::printf("#   MISMATCH region %s\n", lay.regions[i].name.c_str()); }
        }
        return ok;
    };
    auto pos_bytes = [&](int64_t n) { size_t b = 0; for (const auto & sp : lay.streams) b += (size_t) (n / sp.ratio) * sp.row_bytes; return b; };

    // ---- pool
    Ds41PoolConfig cfg;
    cfg.budget_bytes = budget_mib << 20;
    cfg.slab_bytes = slab_kib << 10;
    cfg.prealloc = prealloc;
    cfg.device = dev;
    Ds41ConvPool pool;
    auto t0 = clk::now();
    if (!pool.init(lay, cfg, err)) { std::fprintf(stderr, "pool init: %s\n", err.c_str()); return 1; }
    std::printf("# pool: slab %zu KiB, chunk %zu positions, init %.1f ms (%s extents of %zu MiB)\n", cfg.slab_bytes >> 10,
                pool.chunk_positions(), ms_since(t0), prealloc ? "the whole budget in" : "lazy", cfg.extent_bytes >> 20);
    {   // pinned allocation speed: one 256 MiB extent
        void * p = nullptr;
        t0 = clk::now();
        CK(cudaHostAlloc(&p, (size_t) 256 << 20, cudaHostAllocPortable));
        const double a = ms_since(t0);
        t0 = clk::now();
        CK(cudaFreeHost(p));
        std::printf("# cudaHostAlloc 256 MiB: %.1f ms (free %.1f ms) -> %.2f s for the full %zu MiB budget\n", a, ms_since(t0),
                    a * (double) budget_mib / 256.0 / 1000.0, budget_mib);
    }

    std::vector<int32_t> tok((size_t) ctx);
    { std::mt19937 g(1); for (auto & x : tok) x = (int32_t) (g() % 129280); }
    std::vector<float> logits((size_t) G.vocab, 0.25f);
    bool all_ok = true;

    std::printf("\n## A/B/C: n\tparkKV_MiB\tpark_pageable_ms\tunpark_pageable_ms\tsave_enqueue_ms\tsave_dev_ms\tsave_wall_ms\tsave_GBps\t"
                "restore_enqueue_ms\trestore_dev_ms\trestore_wall_ms\trestore_GBps\tincr_save_MiB\tincr_save_ms\t"
                "partial_restore_ms\texact\n");
    for (int64_t n : sizes) {
        if (n < 1 || n > ctx) continue;
        const size_t kv = pos_bytes(n), tot = kv + lay.region_bytes();
        // A. baseline: main's park (ggml_backend_tensor_get = synchronous cudaMemcpy into pageable vectors) + unpark
        double park = 1e30, unpark = 1e30;
        for (int r = 0; r < reps; ++r) {
            std::vector<std::vector<uint8_t>> pv;
            t0 = clk::now();
            for (size_t i = 0; i < lay.streams.size(); ++i) {
                const size_t b = (size_t) (n / lay.streams[i].ratio + 1) * lay.streams[i].row_bytes;
                pv.emplace_back(b);
                CK(cudaMemcpy(pv.back().data(), D.s[i], b, cudaMemcpyDeviceToHost));
            }
            for (size_t i = 0; i < lay.regions.size(); ++i) { pv.emplace_back(D.rbytes[i]); CK(cudaMemcpy(pv.back().data(), D.r[i], D.rbytes[i], cudaMemcpyDeviceToHost)); }
            park = std::min(park, ms_since(t0));
            t0 = clk::now();
            size_t k = 0;
            for (size_t i = 0; i < lay.streams.size(); ++i, ++k) CK(cudaMemcpy(D.s[i], pv[k].data(), pv[k].size(), cudaMemcpyHostToDevice));
            for (size_t i = 0; i < lay.regions.size(); ++i, ++k) CK(cudaMemcpy(D.r[i], pv[k].data(), pv[k].size(), cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());
            unpark = std::min(unpark, ms_since(t0));
        }
        // B. pool save / restore (fresh record each rep: different first token so nothing is shared)
        double se = 1e30, sd = 1e30, sw = 1e30, re = 1e30, rd = 1e30, rw = 1e30;
        bool exact = true;
        Ds41PoolId id = 0;
        for (int r = 0; r < reps; ++r) {
            pool.clear(false);
            tok[0] = 1000 + r;
            t0 = clk::now();
            id = pool.save(D.t, tok.data(), nullptr, n, logits.data(), logits.size(), err);
            const double e1 = ms_since(t0);
            if (!id) { std::fprintf(stderr, "save %lld: %s\n", (long long) n, err.c_str()); return 1; }
            pool.fence();
            const double w1 = ms_since(t0);
            if (r) { se = std::min(se, e1); sw = std::min(sw, w1); sd = std::min(sd, pool.stats().last_save_ms); }   // rep 0 pays the extent allocations
            scramble(0, n);
            Ds41PoolRestore rr;
            t0 = clk::now();
            if (!pool.restore(id, 0, 0, D.t, rr, err)) { std::fprintf(stderr, "restore: %s\n", err.c_str()); return 1; }
            const double e2 = ms_since(t0);
            if (!pool.wait(rr, &err)) { std::fprintf(stderr, "wait: %s\n", err.c_str()); return 1; }
            const double w2 = ms_since(t0);
            re = std::min(re, e2); rw = std::min(rw, w2); rd = std::min(rd, pool.stats().last_restore_ms);
            exact &= verify(n) && rr.logits == logits;
        }
        // C. incremental save: the same conversation +1024 positions (device rows of [n, n+1024) and the rings change)
        double inc_ms = 0, inc_mib = 0, part_ms = 0;
        if (n + 1024 <= ctx) {
            for (size_t i = 0; i < lay.streams.size(); ++i) {
                const auto & sp = lay.streams[i];
                const size_t a = (size_t) (n / sp.ratio) * sp.row_bytes, b = (size_t) ((n + 1024) / sp.ratio) * sp.row_bytes;
                pattern(ref_s[i].data() + a, b - a, i, 2);
                CK(cudaMemcpy((uint8_t *) D.s[i] + a, ref_s[i].data() + a, b - a, cudaMemcpyHostToDevice));
            }
            for (size_t i = 0; i < lay.regions.size(); ++i) {
                pattern(ref_r[i].data(), D.rbytes[i], 1000 + i, 2 + (uint64_t) n);
                CK(cudaMemcpy(D.r[i], ref_r[i].data(), D.rbytes[i], cudaMemcpyHostToDevice));
            }
            const auto s0 = pool.stats();
            t0 = clk::now();
            const Ds41PoolId id2 = pool.save(D.t, tok.data(), nullptr, n + 1024, logits.data(), logits.size(), err);
            pool.fence();
            inc_ms = ms_since(t0);
            const auto s1 = pool.stats();
            inc_mib = (double) (s1.bytes_d2h - s0.bytes_d2h) / 1048576.0;
            if (!id2) { std::fprintf(stderr, "incremental save: %s\n", err.c_str()); return 1; }
            // partial restore: the live state shares [0, n + 1024 - 4096)
            const int64_t from = std::max<int64_t>(0, n + 1024 - 4096);
            scramble(from, n + 1024);
            const auto rec = pool.best_prefix(tok.data(), nullptr, n + 1024);
            Ds41PoolRestore rr;
            t0 = clk::now();
            if (!pool.restore(rec.rec, rec.boundary, from, D.t, rr, err) || !pool.wait(rr, &err)) { std::fprintf(stderr, "partial: %s\n", err.c_str()); return 1; }
            part_ms = ms_since(t0);
            exact &= rec.pos == n + 1024 && verify(n + 1024);
            // back to generation 1 for the next size
            for (size_t i = 0; i < lay.streams.size(); ++i) { pattern(ref_s[i].data(), D.sbytes[i], i, 1); CK(cudaMemcpy(D.s[i], ref_s[i].data(), D.sbytes[i], cudaMemcpyHostToDevice)); }
            for (size_t i = 0; i < lay.regions.size(); ++i) { pattern(ref_r[i].data(), D.rbytes[i], 1000 + i, 1); CK(cudaMemcpy(D.r[i], ref_r[i].data(), D.rbytes[i], cudaMemcpyHostToDevice)); }
        }
        all_ok &= exact;
        std::printf("%lld\t%.1f\t%.2f\t%.2f\t%.3f\t%.2f\t%.2f\t%.1f\t%.3f\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f\t%.2f\t%s\n", (long long) n,
                    kv / 1048576.0, park, unpark, se, sd, sw, tot / sd / 1e6, re, rd, rw, tot / rd / 1e6, inc_mib, inc_ms, part_ms,
                    exact ? "yes" : "NO");
        std::fflush(stdout);
    }

    // ---- D. overlap with compute: a ~100 ms kernel on a non-blocking stream, a 128K save + restore during it
    {
        const int64_t n = std::min<int64_t>(131072, ctx);
        cudaStream_t ks;
        CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
        int khz = 0;
        CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
        const long long cyc = (long long) khz * 100;   // kHz -> 100 ms
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        auto kernel_ms = [&]() { float ms = 0; CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b)); return ms; };
        CK(cudaEventRecord(a, ks)); CK(ds41_pool_bench_spin(cyc, prop.multiProcessorCount * 2, ks)); CK(cudaEventRecord(b, ks));
        const float alone = kernel_ms();
        pool.clear(false);
        tok[0] = 7;
        CK(cudaEventRecord(a, ks)); CK(ds41_pool_bench_spin(cyc, prop.multiProcessorCount * 2, ks)); CK(cudaEventRecord(b, ks));
        t0 = clk::now();
        const Ds41PoolId id = pool.save(D.t, tok.data(), nullptr, n, nullptr, 0, err);
        pool.fence();
        const double save_wall = ms_since(t0);
        Ds41PoolRestore rr;
        t0 = clk::now();
        const bool rok = id && pool.restore(id, 0, 0, D.t, rr, err) && pool.wait(rr);
        const double rest_wall = ms_since(t0);
        const float with = kernel_ms();
        std::printf("\n## D. overlap: kernel alone %.1f ms, with a %lld-position save+restore %.1f ms; save wall %.2f ms, "
                    "restore wall %.2f ms (both inside the kernel = the copies do not wait for compute) %s\n", alone,
                    (long long) n, with, save_wall, rest_wall, rok && save_wall + rest_wall < with ? "OK" : "CHECK");
        all_ok &= rok && verify(n);
        CK(cudaStreamDestroy(ks));
    }

    // ---- E. fill the budget: distinct conversations of fill_n, then the same with a shared sys_n prefix (--fill 0: skip)
    if (fill_n > 0) {
        fill_n = std::min(fill_n, ctx);
        for (int shared = 0; shared < 2; ++shared) {
            pool.clear(false);
            std::mt19937 g(99);
            std::vector<int32_t> t((size_t) fill_n);
            std::vector<Ds41PoolId> ids;
            double save_ms = 0;
            int k = 0;
            const uint64_t ev0 = pool.stats().evictions;   // cumulative: count this run's only
            for (;; ++k) {
                for (auto & x : t) x = (int32_t) (g() % 129280);
                if (shared) for (int64_t i = 0; i < std::min(sys_n, fill_n); ++i) t[(size_t) i] = (int32_t) (i * 7 + 3);
                t0 = clk::now();
                const Ds41PoolId id = pool.save(D.t, t.data(), nullptr, fill_n, logits.data(), logits.size(), err);
                save_ms += ms_since(t0);
                if (!id) { std::fprintf(stderr, "fill save: %s\n", err.c_str()); break; }
                if (pool.stats().evictions != ev0) break;
                ids.push_back(id);
                if (k > 20000) break;
            }
            pool.fence();
            const auto s = pool.stats();
            t0 = clk::now();
            const auto mt = pool.best_prefix(t.data(), nullptr, fill_n);
            const double bp = ms_since(t0);
            std::printf("\n## E%d. %s: %zu conversations of %lld positions fit in %zu MiB (%.1f MiB each, slabs %zu/%zu, host %.1f MiB, "
                        "shared %.0f MiB); save enqueue mean %.3f ms; best_prefix over %zu records %.2f ms (match pos %lld)\n",
                        shared, shared ? "shared prefix" : "distinct", ids.size(), (long long) fill_n, budget_mib,
                        s.used_bytes / 1048576.0 / std::max<size_t>(1, s.records), s.slabs_used, s.slabs_total,
                        s.host_bytes / 1048576.0, s.bytes_shared / 1048576.0, save_ms / std::max(1, k), s.records, bp,
                        (long long) mt.pos);
            if (shared) std::printf("##    (shared: the first %lld positions of each conversation are the same system prompt)\n", (long long) sys_n);
        }
        // the 4-agent round robin: four 32K conversations with a shared sys prefix, each turn +512 positions;
        // a switch = save the current (incremental) + restore the next from the shared prefix
        pool.clear(false);
        const int A = 4;
        const int64_t base = std::min<int64_t>(32768, ctx - 4096);
        std::vector<std::vector<int32_t>> conv(A);
        std::vector<Ds41PoolId> rec(A, 0);
        std::mt19937 g(5);
        for (int i = 0; i < A; ++i) {
            conv[i].resize((size_t) base);
            for (auto & x : conv[i]) x = (int32_t) (g() % 129280);
            for (int64_t j = 0; j < std::min(sys_n, base); ++j) conv[i][(size_t) j] = (int32_t) (j * 7 + 3);
            rec[i] = pool.save(D.t, conv[i].data(), nullptr, base, logits.data(), logits.size(), err);
        }
        pool.fence();
        double sw_sum = 0, sw_max = 0;
        int nsw = 0;
        for (int turn = 0; turn < 8; ++turn) {
            for (int i = 0; i < A; ++i) {
                // the live one is agent i: it decoded +512 positions; save it, switch to agent i+1
                for (int j = 0; j < 512; ++j) conv[i].push_back((int32_t) (g() % 129280));
                const int nx = (i + 1) % A;
                t0 = clk::now();
                rec[i] = pool.save(D.t, conv[i].data(), nullptr, (int64_t) conv[i].size(), logits.data(), logits.size(), err);
                const auto m2 = pool.best_prefix(conv[nx].data(), nullptr, (int64_t) conv[nx].size());
                Ds41PoolRestore rr;
                const bool ok = rec[i] && m2.rec && pool.restore(m2.rec, m2.boundary, std::min<int64_t>(sys_n, m2.pos), D.t, rr, err) && pool.wait(rr);
                const double w = ms_since(t0);
                if (!ok) { std::fprintf(stderr, "switch: %s\n", err.c_str()); all_ok = false; break; }
                sw_sum += w; sw_max = std::max(sw_max, w); ++nsw;
            }
        }
        std::printf("\n## E2. 4 agents round robin (%lld -> %lld positions, shared %lld): %d switches, mean %.2f ms, max %.2f ms "
                    "(save current + best_prefix + restore next from the shared prefix + wait); %s\n", (long long) base,
                    (long long) conv[0].size(), (long long) sys_n, nsw, sw_sum / std::max(1, nsw), sw_max, pool.stats_line().c_str());
    }
    std::printf("\n# %s\n", all_ok ? "BENCH OK: every restore byte-exact" : "BENCH: MISMATCH");
    pool.clear(false);
    for (void * p : D.s) cudaFree(p);
    for (void * p : D.r) cudaFree(p);
    return all_ok ? 0 : 1;
}
