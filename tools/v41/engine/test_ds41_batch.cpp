// test_ds41_batch - the batch-slot gates on the CPU backend with the real model (tools/v41/engine/ds41_batch.hpp).
//
//   test_ds41_batch -m M.gguf --engram A:B --prompts p0.i32,p1.i32,p2.i32 [--steps 8] [--feed 24] [--threads 48]
//                   [--ctx 8192] [--json out.json] [--backend cpu|cuda] [--kv-fp4]
// --backend cuda: the dense half on the card (the batched graphs' CUDA kernels), the routed experts still on the CPU tier
// (row by row, the same bits whatever the window) - the card's batch-vs-solo numbers without the VRAM expert tier.
//
// K conversations (prompt files of int32 ids).  Slots 0..K-1 hold the SOLO references, slots K..2K-1 the batch runs.
//   ref    every conversation alone: its prompt as solo passes of <= 16 rows (Ds41RowsStep, the per-slot reference
//          path = bind_slot + begin + attn_router / tier / finish), then --steps greedy decode steps, logits kept.
//   (b)    batch: the same prompts read solo into slots K.., then --steps windows with ONE row per slot (begin_rows),
//          teacher-forced with the reference tokens: per row top-1 / KL / max |diff| against the solo logits.
//          The same windows through the reference path (STRATA_ROWS_REF semantics) must be bit-exact to solo.
//   (c)    feed: the last conversation's last --feed prompt tokens are read as rows INSIDE the windows of the others'
//          decode (ds41_plan_rows, <= 16 rows a window), then it decodes with them: its last prompt row and its decode
//          rows against its solo reference, the others' rows as in (b).
//   bind   the solo path on another slot: slot 0's conversation read again in slot 2K-1 must be bit-exact; half way
//          through its decode the slot is swap_slots()'d with slot K and decodes on there (bit-exact again).
//   pool   the conversation pool across slots: conversation 1's turn boundary saved from slot 2K-2, restored into slot
//          K, decoded there - bit-exact with its solo reference (the pool's default store, host memory on the CPU).
// DS41_TEST_LAYERS=n runs only the first n layers (a faster gate of the same code; not the model's logits).
#include "ds41_batch.hpp"
#include "ds41_conv_reuse.hpp"
#include "ds41_dense.hpp"
#include "ds41_moe.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int> read_ids(const std::string & f) {
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    std::vector<int> v;
    if (!in) return v;
    v.resize((size_t) in.tellg() / 4);
    in.seekg(0);
    in.read((char *) v.data(), (std::streamsize) (v.size() * 4));
    return v;
}

int argmax(const float * x, int n) { return (int) (std::max_element(x, x + n) - x); }

struct Cmp { double maxdiff = 0, kl = 0; bool top1 = true, exact = true; };
Cmp cmp(const float * ref, const float * t, int n) {
    Cmp c;
    float ma = -INFINITY, mb = -INFINITY;
    for (int i = 0; i < n; ++i) { ma = std::max(ma, ref[i]); mb = std::max(mb, t[i]); }
    double za = 0, zb = 0;
    for (int i = 0; i < n; ++i) { za += std::exp((double) ref[i] - ma); zb += std::exp((double) t[i] - mb); }
    const double lza = std::log(za), lzb = std::log(zb);
    for (int i = 0; i < n; ++i) {
        c.maxdiff = std::max(c.maxdiff, (double) std::fabs(ref[i] - t[i]));
        if (ref[i] != t[i]) c.exact = false;
        const double la = (double) ref[i] - ma - lza, lb = (double) t[i] - mb - lzb;
        c.kl += std::exp(la) * (la - lb);
    }
    c.top1 = argmax(ref, n) == argmax(t, n);
    return c;
}

struct Agg {
    long n = 0, top1 = 0, exact = 0;
    double kl_max = 0, kl_sum = 0, diff_max = 0;
    void add(const Cmp & c) {
        ++n; top1 += c.top1; exact += c.exact; kl_max = std::max(kl_max, c.kl); kl_sum += c.kl; diff_max = std::max(diff_max, c.maxdiff);
    }
    std::string json() const {
        char b[256];
        std::snprintf(b, sizeof b, "{\"rows\": %ld, \"top1\": %ld, \"bitexact\": %ld, \"kl_max\": %.3e, \"kl_mean\": %.3e, \"maxdiff\": %.5f}",
                      n, top1, exact, kl_max, n ? kl_sum / n : 0.0, diff_max);
        return b;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string model, engram, prompts_csv, json_out, backend = "cpu";
    bool kv_fp4 = false;
    int steps = 8, feed = 24, threads = 48;
    int64_t ctx = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (k == "-m") model = next();
        else if (k == "--engram") engram = next();
        else if (k == "--prompts") prompts_csv = next();
        else if (k == "--steps") steps = std::atoi(next().c_str());
        else if (k == "--feed") feed = std::atoi(next().c_str());
        else if (k == "--threads") threads = std::atoi(next().c_str());
        else if (k == "--ctx") ctx = std::atoll(next().c_str());
        else if (k == "--json") json_out = next();
        else if (k == "--backend") backend = next();
        else if (k == "--kv-fp4") kv_fp4 = true;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    std::vector<std::vector<int>> P;
    { std::stringstream ss(prompts_csv); std::string f; while (std::getline(ss, f, ',')) if (!f.empty()) P.push_back(read_ids(f)); }
    const int K = (int) P.size();
    if (model.empty() || K < 2) { std::fprintf(stderr, "usage: test_ds41_batch -m M --engram A:B --prompts a,b[,c..]\n"); return 2; }
    for (auto & p : P) if (p.size() < (size_t) feed + 4) { std::fprintf(stderr, "a prompt is shorter than --feed + 4\n"); return 2; }
    if (K > Ds41Dense::kMaxRows) { std::fprintf(stderr, "at most %d prompts\n", Ds41Dense::kMaxRows); return 2; }

    ggml_backend_t be = backend == "cuda" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                                          : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "no %s backend\n", backend.c_str()); return 1; }
    if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, threads);
    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = threads;
    dc.ctx_max = ctx;
    dc.max_pass = 64;
    dc.prompt_batch = 0;
    { std::stringstream ss(engram); std::string d; while (std::getline(ss, d, ':')) if (!d.empty()) dc.engram_dirs.push_back(d); }
    dc.n_slots = 2 * K;
    dc.kv_fp4 = kv_fp4;
    Ds41Dense dense;
    std::string err;
    const double t_load = now_s();
    if (!dense.init(model, dc, err)) { std::fprintf(stderr, "dense init: %s\n", err.c_str()); return 1; }
    Ds41FileBlobs src(model);
    if (!src.open(err)) { std::fprintf(stderr, "experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    Ds41MoeTier tier;
    Ds41MoeConfig mc;
    mc.cpu_only = true;
    mc.slots = TG.n_layers * TG.n_experts;
    mc.threads = threads;
    mc.arena_gib = 0;
    mc.max_arena_gib = 0;
    if (!tier.init(TG, &src, mc, err)) { std::fprintf(stderr, "tier init: %s\n", err.c_str()); return 1; }
    const int nv = (int) dense.geom().vocab;
    std::fprintf(stderr, "loaded in %.1f s: %lld layers, %d slots, %d conversations, steps %d, feed %d\n", now_s() - t_load,
                 (long long) dense.n_layer(), dense.n_slots(), K, steps, feed);

    Ds41RowsStep rs(dense, tier);
    std::vector<float> lg;
    // a slot's prompt [from, to) as solo passes of <= 16 rows; the last row's logits into last (if given)
    auto solo_prompt = [&](int slot, const std::vector<int> & ids, int from, int to, std::vector<float> * last) -> bool {
        rs.set_reference(true);
        for (int c = from; c < to; c += 16) {
            const int m = std::min(16, to - c);
            Ds41Dense::Row rows[16];
            for (int i = 0; i < m; ++i) rows[i] = Ds41Dense::Row { slot, c + i, ids[(size_t) (c + i)] };
            const int w = m - 1;
            if (!rs.run(rows, m, &w, 1, nullptr, lg, err)) return false;
        }
        if (last) *last = lg;
        return true;
    };

    // ---- ref: every conversation alone (slots 0..K-1)
    std::vector<std::vector<float>> ref_last(K);              // the last prompt row
    std::vector<std::vector<std::vector<float>>> ref_dec(K);  // decode step s: the row fed at P + s
    std::vector<std::vector<int>> ref_tok(K);                 // greedy tokens: tok[0] from the last prompt row
    double t0 = now_s();
    for (int k = 0; k < K; ++k) {
        const int L = (int) P[k].size();
        if (!solo_prompt(k, P[k], 0, L, &ref_last[k])) { std::fprintf(stderr, "ref prompt %d: %s\n", k, err.c_str()); return 1; }
        int x = argmax(ref_last[k].data(), nv);
        ref_tok[k].push_back(x);
        for (int s = 0; s < steps; ++s) {
            Ds41Dense::Row r { k, L + s, x };
            const int w = 0;
            if (!rs.run(&r, 1, &w, 1, nullptr, lg, err)) { std::fprintf(stderr, "ref decode %d: %s\n", k, err.c_str()); return 1; }
            ref_dec[k].push_back(lg);
            x = argmax(lg.data(), nv);
            ref_tok[k].push_back(x);
        }
        std::fprintf(stderr, "ref %d: prompt %d, tokens", k, L);
        for (int t : ref_tok[k]) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, " (%.0f s so far)\n", now_s() - t0);
    }

    // ---- (b) batch decode windows: slots K..2K-1 read the same prompts solo, then one row per slot a window
    auto batch_decode = [&](bool reference, Agg & agg, std::vector<std::vector<int>> & got) -> bool {
        got.assign(K, {});
        for (int k = 0; k < K; ++k)
            if (!solo_prompt(K + k, P[k], 0, (int) P[k].size(), nullptr)) { std::fprintf(stderr, "batch prompt %d: %s\n", k, err.c_str()); return false; }
        rs.set_reference(reference);
        for (int s = 0; s < steps; ++s) {
            Ds41Dense::Row rows[16];
            int want[16];
            for (int k = 0; k < K; ++k) {
                rows[k] = Ds41Dense::Row { K + k, (int) P[k].size() + s, ref_tok[k][(size_t) s] };   // teacher-forced
                want[k] = k;
            }
            if (!rs.run(rows, K, want, K, nullptr, lg, err)) { std::fprintf(stderr, "batch window %d: %s\n", s, err.c_str()); return false; }
            for (int k = 0; k < K; ++k) {
                const float * t = lg.data() + (size_t) k * nv;
                agg.add(cmp(ref_dec[k][(size_t) s].data(), t, nv));
                got[k].push_back(argmax(t, nv));
            }
        }
        return true;
    };
    Agg b_ref, b_bat;
    std::vector<std::vector<int>> g_ref, g_bat;
    t0 = now_s();
    if (!batch_decode(true, b_ref, g_ref)) return 1;
    std::fprintf(stderr, "(b) reference windows vs solo: %s (%.0f s)\n", b_ref.json().c_str(), now_s() - t0);
    t0 = now_s();
    if (!batch_decode(false, b_bat, g_bat)) return 1;
    std::fprintf(stderr, "(b) batched windows vs solo: %s (%.0f s; %s)\n", b_bat.json().c_str(), now_s() - t0, rs.stats_line(true).c_str());

    // ---- (c) the last conversation's last `feed` prompt tokens read inside the others' decode windows
    Agg c_dec, c_feed_last, c_feed_dec;
    t0 = now_s();
    {
        const int f = K - 1, Lf = (int) P[f].size();
        for (int k = 0; k < K; ++k)   // the others' prompts whole, the fed one without its tail
            if (!solo_prompt(K + k, P[k], 0, k == f ? Lf - feed : (int) P[k].size(), nullptr)) { std::fprintf(stderr, "feed prompt: %s\n", err.c_str()); return 1; }
        rs.set_reference(false);
        Ds41RowsFeed F { K + f, P[f].data(), Lf, Lf - feed, 0 };
        F.ids = P[f].data();
        F.n = Lf;
        F.done = Lf - feed;
        F.pos0 = 0;
        std::vector<int> step(K, 0);   // decode steps done per slot
        int fx = -1, fstep = 0;         // the fed slot's decode, once its prompt is read
        std::vector<int> want;
        std::vector<char> fdone;
        int windows = 0;
        const int feed_max = std::getenv("TEST_FEED_ROWS") ? std::atoi(std::getenv("TEST_FEED_ROWS")) : 0;
        while (true) {
            int ds[16], dp[16], dt[16], nd = 0;
            for (int k = 0; k < K; ++k) {
                if (k == f) continue;
                if (step[k] >= steps) continue;
                ds[nd] = K + k; dp[nd] = (int) P[k].size() + step[k]; dt[nd] = ref_tok[k][(size_t) step[k]]; ++nd;
            }
            const bool fdec = fx >= 0 && fstep < steps;
            if (fdec) { ds[nd] = K + f; dp[nd] = Lf + fstep; dt[nd] = ref_tok[f][(size_t) fstep]; ++nd; }
            const bool feeding = F.done < F.n;
            if (nd == 0 && !feeding) break;
            Ds41Dense::Row rows[16];
            const int n = ds41_plan_rows(ds, dp, dt, nd, &F, feeding ? 1 : 0, 16, feed_max, rows, want, fdone);
            if (!rs.run(rows, n, want.data(), (int) want.size(), nullptr, lg, err)) { std::fprintf(stderr, "feed window: %s\n", err.c_str()); return 1; }
            ++windows;
            for (size_t i = 0; i < want.size(); ++i) {
                const Ds41Dense::Row & r = rows[want[i]];
                const int k = r.slot - K;
                const float * t = lg.data() + i * (size_t) nv;
                if (k == f && r.pos == Lf - 1) { c_feed_last.add(cmp(ref_last[f].data(), t, nv)); fx = 0; }
                else if (k == f) { c_feed_dec.add(cmp(ref_dec[f][(size_t) fstep].data(), t, nv)); ++fstep; }
                else { c_dec.add(cmp(ref_dec[k][(size_t) step[k]].data(), t, nv)); ++step[k]; }
            }
        }
        std::fprintf(stderr, "(c) %d windows: others' decode %s; fed prompt's last row %s; its decode %s (%.0f s)\n", windows,
                     c_dec.json().c_str(), c_feed_last.json().c_str(), c_feed_dec.json().c_str(), now_s() - t0);
    }

    // ---- bind: conversation 0 solo again on the last slot (its state is (c)'s, overwritten from position 0)
    Agg a_bind;
    {
        int s2 = 2 * K - 1;
        std::vector<float> last;
        if (!solo_prompt(s2, P[0], 0, (int) P[0].size(), &last)) { std::fprintf(stderr, "bind prompt: %s\n", err.c_str()); return 1; }
        a_bind.add(cmp(ref_last[0].data(), last.data(), nv));
        rs.set_reference(true);
        int x = ref_tok[0][0];
        for (int s = 0; s < steps; ++s) {
            if (s == steps / 2) {   // the conversation moves to slot K (whose own state goes to s2)
                if (!dense.swap_slots(s2, K)) { std::fprintf(stderr, "swap: %s\n", dense.last_error().c_str()); return 1; }
                s2 = K;
            }
            Ds41Dense::Row r { s2, (int) P[0].size() + s, x };
            const int w = 0;
            if (!rs.run(&r, 1, &w, 1, nullptr, lg, err)) { std::fprintf(stderr, "bind decode: %s\n", err.c_str()); return 1; }
            a_bind.add(cmp(ref_dec[0][(size_t) s].data(), lg.data(), nv));
            x = ref_tok[0][(size_t) s + 1];
        }
        std::fprintf(stderr, "bind: conversation 0 on slot %d (swapped to slot %d half way) vs slot 0: %s\n", 2 * K - 1, s2, a_bind.json().c_str());
    }
    // ---- pool: conversation 1 saved at its prompt's end from slot 2K-2, restored into another slot (K)
    Agg a_pool;
    {
        const int sa = 2 * K - 2, sb = sa == K ? 2 * K - 1 : K;   // two different slots (bind is done with both)
        const int L = (int) P[1].size();
        Ds41ConvStore pool;
        if (!pool.init(dense, ctx, (size_t) 512 << 20, 8, !ggml_backend_is_cpu(be), err)) { std::fprintf(stderr, "pool init: %s\n", err.c_str()); return 1; }
        std::vector<float> last;
        if (!solo_prompt(sa, P[1], 0, L, &last)) { std::fprintf(stderr, "pool prompt: %s\n", err.c_str()); return 1; }
        a_pool.add(cmp(ref_last[1].data(), last.data(), nv));
        if (!dense.bind_slot(sa) || !dense.snapshot(L)) { std::fprintf(stderr, "pool snapshot: %s\n", dense.last_error().c_str()); return 1; }
        if (!pool.save(dense, P[1].data(), nullptr, 0, L, last.data(), last.size(), err)) { std::fprintf(stderr, "pool save: %s\n", err.c_str()); return 1; }
        if (!pool.fence(&err)) { std::fprintf(stderr, "pool fence: %s\n", err.c_str()); return 1; }
        // slot sa goes on with something else (its rings / rows overwritten from position 0) before the restore
        if (!solo_prompt(sa, P[2], 0, 40, nullptr)) { std::fprintf(stderr, "pool other: %s\n", err.c_str()); return 1; }
        if (!dense.bind_slot(sb)) { std::fprintf(stderr, "pool bind: %s\n", dense.last_error().c_str()); return 1; }
        const ConvMatch m = pool.best_prefix(P[1].data(), nullptr, 0, L);
        ConvRestore rr;
        if (m.pos != L || !pool.restore(dense, m, 0, rr, err)) { std::fprintf(stderr, "pool restore (match %lld): %s\n", (long long) m.pos, err.c_str()); return 1; }
        dense.rearm((int) rr.pos);
        if (!dense.restore()) { std::fprintf(stderr, "pool rearm: %s\n", dense.last_error().c_str()); return 1; }
        dense.set_engram_history(L - 3, 3, P[1].data() + L - 3, nullptr);   // as the serve loop does on a resume
        a_pool.add(cmp(ref_last[1].data(), rr.logits.data(), nv));
        rs.set_reference(true);
        int x = argmax(rr.logits.data(), nv);
        for (int s = 0; s < steps; ++s) {
            Ds41Dense::Row r { sb, L + s, x };
            const int w = 0;
            if (!rs.run(&r, 1, &w, 1, nullptr, lg, err)) { std::fprintf(stderr, "pool decode: %s\n", err.c_str()); return 1; }
            a_pool.add(cmp(ref_dec[1][(size_t) s].data(), lg.data(), nv));
            x = ref_tok[1][(size_t) s + 1];
        }
        std::fprintf(stderr, "pool: conversation 1 saved from slot %d, restored into slot %d (%s): %s\n", sa, sb, pool.kind(), a_pool.json().c_str());
    }
    // ---- noise floor: the same decode positions of every conversation as ONE solo pass of `steps` rows (what a solo
    // DSpark verify pass does) against the one-row steps - the batch-shape noise the solo path itself has
    Agg a_noise;
    {
        const int s2 = 2 * K - 2;
        for (int k = 0; k < K; ++k) {
            const int L = (int) P[k].size();
            if (!solo_prompt(s2, P[k], 0, L, nullptr)) { std::fprintf(stderr, "noise prompt: %s\n", err.c_str()); return 1; }
            rs.set_reference(true);
            const int m = std::min(steps, 16);
            Ds41Dense::Row rows[16];
            int want[16];
            for (int i = 0; i < m; ++i) { rows[i] = Ds41Dense::Row { s2, L + i, ref_tok[k][(size_t) i] }; want[i] = i; }
            if (!rs.run(rows, m, want, m, nullptr, lg, err)) { std::fprintf(stderr, "noise pass: %s\n", err.c_str()); return 1; }
            for (int i = 0; i < m; ++i) a_noise.add(cmp(ref_dec[k][(size_t) i].data(), lg.data() + (size_t) i * nv, nv));
        }
        std::fprintf(stderr, "noise: the decode rows as one %d-row solo pass vs one-row steps: %s\n", std::min(steps, 16), a_noise.json().c_str());
    }
    int free_top1 = 0, free_n = 0;
    for (int k = 0; k < K; ++k)
        for (int s = 0; s < steps; ++s) { ++free_n; free_top1 += g_bat[k][(size_t) s] == ref_tok[k][(size_t) s + 1]; }
    std::string js = "{\"layers\": " + std::to_string(dense.n_layer()) + ", \"conversations\": " + std::to_string(K) +
                     ", \"steps\": " + std::to_string(steps) + ", \"feed\": " + std::to_string(feed) +
                     ", \"b_reference\": " + b_ref.json() + ", \"b_batched\": " + b_bat.json() +
                     ", \"b_greedy_same\": [" + std::to_string(free_top1) + ", " + std::to_string(free_n) + "]" +
                     ", \"c_others\": " + c_dec.json() + ", \"c_fed_last_row\": " + c_feed_last.json() +
                     ", \"c_fed_decode\": " + c_feed_dec.json() + ", \"bind\": " + a_bind.json() + ", \"pool\": " + a_pool.json() + ", \"noise\": " + a_noise.json() + "}";
    std::printf("RESULT %s\n", js.c_str());
    if (!json_out.empty()) { std::ofstream o(json_out); o << js << "\n"; }
    // the CPU backend: a window's rows are the solo rows bit for bit; a GPU's dense kernels round by column count, so
    // there the batched rows must stay within the solo path's own noise (its multi-row pass against one-row steps)
    const bool cpu_be = ggml_backend_is_cpu(be);
    const bool batch_ok = cpu_be ? b_bat.exact == b_bat.n && c_dec.exact == c_dec.n && c_feed_last.exact == c_feed_last.n
                                 : b_bat.kl_sum / std::max(1L, b_bat.n) <= 2.0 * a_noise.kl_sum / std::max(1L, a_noise.n) &&
                                   b_bat.top1 >= a_noise.top1 * b_bat.n / std::max(1L, a_noise.n) - 1;
    const bool ok = b_ref.exact == b_ref.n && a_bind.exact == a_bind.n && a_pool.exact == a_pool.n && batch_ok;
    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL (see RESULT)");
    tier.close();
    std::fflush(stdout);
    std::_Exit(ok ? 0 : 3);
}
