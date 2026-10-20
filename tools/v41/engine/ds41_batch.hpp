// tools/v41/engine/ds41_batch.hpp - batch slots for the DeepSeek-V4.1 engine: one batch WINDOW = rows of several
// conversations (slots) through every layer in one pass.  The dense half runs the rows batched (Ds41Dense::begin_rows:
// the projections / router / shared expert over every row, the attention + indexer per slot on its own state), the
// routed experts run as ONE tier call per layer over the union of the rows (run_multi_to_dev, kMaxMulti = 16), and the
// logits of the wanted rows come back.  The serve loop's slot protocol (BGEN / BT / BDONE / BSTOP / BYIELD) drives it;
// docs/BATCHING.md describes the interface between the two.
//
//   Ds41RowsStep st(dense, tier);
//   Ds41Dense::Row rows[] = { {0, p0, t0}, {2, p2, t2}, {3, q, a}, {3, q + 1, b} };   // grouped by slot
//   int want[] = { 0, 1, 3 };                                                         // rows whose logits are needed
//   st.run(rows, 4, want, 3, rdev, logits, err);                                      // logits: 3 * n_vocab
//
// STRATA_ROWS_REF=1: the per-slot reference instead - every slot's rows as its own solo pass (bind_slot + begin + the
// same tier calls), the batch gates' baseline (and the protocol's stub before begin_rows existed).
#pragma once

#include "ds41_dense.hpp"
#include "ds41_moe.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::v41 {

/// Check a window's rows: 1..kMaxRows, every slot valid, a slot's rows contiguous at consecutive ascending positions,
/// no slot in two groups.  The groups (first row, row count) are returned in `groups`.
inline bool ds41_rows_groups(const Ds41Dense::Row * rows, int n, int n_slots, std::vector<std::pair<int, int>> & groups,
                             std::string & err) {
    groups.clear();
    if (n < 1 || n > Ds41Dense::kMaxRows) { err = "rows: 1.." + std::to_string(Ds41Dense::kMaxRows) + " rows a window"; return false; }
    for (int i = 0; i < n; ++i) {
        if (rows[i].slot < 0 || rows[i].slot >= n_slots || rows[i].pos < 0) { err = "rows: bad slot / position"; return false; }
        if (i > 0 && rows[i].slot == rows[i - 1].slot) {
            if (rows[i].pos != rows[i - 1].pos + 1) { err = "rows: a slot's rows must be consecutive positions"; return false; }
            ++groups.back().second;
            continue;
        }
        for (const auto & g : groups)
            if (rows[g.first].slot == rows[i].slot) { err = "rows: a slot's rows must be contiguous"; return false; }
        groups.push_back({ i, 1 });
    }
    return true;
}

/// One short prompt being read inside the decode windows (chunked prefill): `ids[done..n)` are still to feed, at
/// positions pos0 + done ...
struct Ds41RowsFeed {
    int slot = 0;
    const int * ids = nullptr;
    int n = 0, pos0 = 0, done = 0;
};

/// Plan one window: a row for every decoding slot (dec_slot[i] feeds dec_tok[i] at dec_pos[i]), then prompt rows of
/// `feeds` in order (a slot's rows consecutive), up to max_rows (<= kMaxRows) in all and at most `feed_max` prompt rows
/// (STRATA_BATCH_FEED_ROWS; 0 = as many as fit).  Advances feeds[i].done by what it planned.  Returns the row count;
/// want gets the rows whose logits a sampler needs: every decode row, and a feed's LAST prompt row when this window
/// finishes it (feed_done[i] = 1 then).
inline int ds41_plan_rows(const int * dec_slot, const int * dec_pos, const int * dec_tok, int n_dec, Ds41RowsFeed * feeds,
                          int n_feeds, int max_rows, int feed_max, Ds41Dense::Row * out, std::vector<int> & want,
                          std::vector<char> & feed_done) {
    max_rows = std::min(max_rows, Ds41Dense::kMaxRows);
    want.clear();
    feed_done.assign((size_t) std::max(0, n_feeds), 0);
    int n = 0;
    for (int i = 0; i < n_dec && n < max_rows; ++i) {
        out[n] = Ds41Dense::Row { dec_slot[i], dec_pos[i], dec_tok[i] };
        want.push_back(n++);
    }
    int fed = 0;
    for (int f = 0; f < n_feeds && n < max_rows; ++f) {
        Ds41RowsFeed & F = feeds[f];
        while (F.done < F.n && n < max_rows && (feed_max <= 0 || fed < feed_max)) {
            out[n++] = Ds41Dense::Row { F.slot, F.pos0 + F.done, F.ids[F.done] };
            ++F.done;
            ++fed;
        }
        if (F.done == F.n && F.n > 0 && n > 0 && out[n - 1].slot == F.slot) { want.push_back(n - 1); feed_done[(size_t) f] = 1; }
    }
    return n;
}

class Ds41RowsStep {
public:
    Ds41RowsStep(Ds41Dense & dense, Ds41MoeTier & tier) : d_(dense), t_(tier) {}

    /// One window: rows[0..n) (grouped by slot, see ds41_rows_groups) through every layer; logits of rows want[0..n_want)
    /// (ascending) into `logits` (n_want * n_vocab, in want order).  rdev: dense.routed_dev() on the GPU tier (re-read
    /// it after a set_columns), nullptr = the host path (the CPU tier).  The bound slot is the same afterwards.
    bool run(const Ds41Dense::Row * rows, int n, const int * want, int n_want, float * rdev, std::vector<float> & logits,
             std::string & err) {
        if (!ds41_rows_groups(rows, n, d_.n_slots(), groups_, err)) return false;
        for (int i = 0; i < n_want; ++i)
            if (want[i] < 0 || want[i] >= n || (i > 0 && want[i] <= want[i - 1])) { err = "rows: bad want list"; return false; }
        static const bool ref = std::getenv("STRATA_ROWS_REF") && std::getenv("STRATA_ROWS_REF")[0] == '1';
        const int nv = (int) d_.geom().vocab;
        logits.resize((size_t) n_want * (size_t) nv);
        const double t0 = now();
        ++windows_;
        rows_ += n;
        if (ref || force_ref_) {
            const int bound = d_.slot();
            int wi = 0;
            for (const auto & g : groups_) {
                const int s = rows[g.first].slot, k = g.second;
                tids_.resize((size_t) k);
                for (int i = 0; i < k; ++i) tids_[(size_t) i] = rows[g.first + i].tid;
                if (!d_.bind_slot(s) || !d_.begin(tids_.data(), k, rows[g.first].pos)) { err = d_.last_error(); d_.bind_slot(bound); return false; }
                if (!layers(k, rdev, err)) { d_.bind_slot(bound); return false; }
                const double tl = now();
                for (; wi < n_want && want[wi] < g.first + k; ++wi)
                    if (!d_.logits(want[wi] - g.first, 1, logits.data() + (size_t) wi * nv)) { err = d_.last_error(); d_.bind_slot(bound); return false; }
                ms_logits_ += now() - tl;
            }
            if (!d_.bind_slot(bound)) { err = d_.last_error(); return false; }
        } else {
            if (!d_.begin_rows(rows, n)) { err = d_.last_error(); return false; }
            if (!layers(n, rdev, err)) return false;
            const double tl = now();
            for (int i = 0; i < n_want; ) {   // contiguous runs of wanted rows: one head graph each
                int j = i + 1;
                while (j < n_want && want[j] == want[j - 1] + 1) ++j;
                if (!d_.logits(want[i], j - i, logits.data() + (size_t) i * nv)) { err = d_.last_error(); return false; }
                i = j;
            }
            ms_logits_ += now() - tl;
        }
        ms_all_ += now() - t0;
        return true;
    }
    /// Force the per-slot reference path (tests); STRATA_ROWS_REF=1 does the same for a whole process.
    void set_reference(bool on) { force_ref_ = on; }
    /// Since the last reset: windows, rows, and the wall split (ms) - dense (begin / attention / finish), tier, logits.
    std::string stats_line(bool reset) {
        char b[256];
        const double w = windows_ ? (double) windows_ : 1.0;
        std::snprintf(b, sizeof b, "%ld windows, %.2f rows each, %.1f ms a window (dense %.1f = prefetch %.1f + attention %.1f + "
                      "finish %.1f, experts %.1f, logits %.1f)", windows_, windows_ ? (double) rows_ / windows_ : 0.0, ms_all_ / w,
                      ms_dense_ / w, ms_pred_ / w, ms_attn_ / w, ms_fin_ / w, ms_tier_ / w, ms_logits_ / w);
        if (reset) { windows_ = rows_ = 0; ms_all_ = ms_dense_ = ms_tier_ = ms_logits_ = ms_pred_ = ms_attn_ = ms_fin_ = 0; }
        return b;
    }

private:
    static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    // every layer of the current pass (m columns): pass_v's D3 path - the predicted union prefetched under the
    // attention (m <= 8), the routed experts as one run_multi_to_dev per kMaxMulti rows (GPU) or row by row (CPU tier)
    bool layers(int m, float * rdev, std::string & err) {
        const Ds41Geometry & G = d_.geom();
        const int nl = (int) d_.n_layer(), K = (int) G.n_used, D = (int) G.n_embd;
        ids_.resize((size_t) m * K);
        w_.resize((size_t) m * K);
        if (!rdev) routed_.resize((size_t) m * D);
        static const int d3_pf = std::getenv("STRATA_D3_PF") ? std::atoi(std::getenv("STRATA_D3_PF")) : 1;
        const bool pf = rdev && d3_pf > 0 && m <= 8;
        double tq = now();
        for (int l = 0; l < nl; ++l) {
            const double tp0 = now();
            if (pf) {
                pred_.resize((size_t) m * 16);
                if (d_.predict_multi(l, pred_.data())) {
                    score_.clear();
                    for (int t = 0; t < m; ++t)
                        for (int i = 0; i < 16; ++i) {
                            const int e = pred_[(size_t) t * 16 + i];
                            if (e < 0) continue;
                            auto it = std::find_if(score_.begin(), score_.end(), [&](const std::pair<int, int> & p) { return p.first == e; });
                            if (it == score_.end()) score_.push_back({ e, 16 - i }); else it->second += 16 - i;
                        }
                    std::stable_sort(score_.begin(), score_.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
                    rank_.clear();
                    for (const auto & p : score_) rank_.push_back(p.first);
                    t_.prefetch_multi(l, rank_.data(), (int) rank_.size(), d3_pf);
                }
            }
            const double ta0 = now();
            ms_pred_ += ta0 - tp0;
            const float * fn = nullptr;
            if (!d_.attn_router(l, ids_.data(), w_.data(), &fn)) { err = d_.last_error(); return false; }
            double t = now(); ms_dense_ += t - tq; ms_attn_ += t - ta0; tq = t;
            if (rdev) {
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int c = std::min(kMM, m - off);
                    if (!t_.run_multi_to_dev(l, c, ids_.data() + (size_t) off * K, w_.data() + (size_t) off * K,
                                             fn + (size_t) off * D, rdev + (size_t) off * D)) { err = "tier.run_multi_to_dev failed"; return false; }
                }
            } else {
                for (int r = 0; r < m; ++r)
                    if (!t_.run(l, ids_.data() + (size_t) r * K, w_.data() + (size_t) r * K, fn + (size_t) r * D,
                                routed_.data() + (size_t) r * D)) { err = "tier.run failed"; return false; }
            }
            t = now(); ms_tier_ += t - tq; tq = t;
            if (!d_.finish(l, rdev ? nullptr : routed_.data())) { err = d_.last_error(); return false; }
            t = now(); ms_dense_ += t - tq; ms_fin_ += t - tq; tq = t;
        }
        return true;
    }
    Ds41Dense & d_;
    Ds41MoeTier & t_;
    bool force_ref_ = false;
    std::vector<std::pair<int, int>> groups_;
    std::vector<int> tids_, ids_, pred_, rank_;
    std::vector<float> w_, routed_;
    std::vector<std::pair<int, int>> score_;
    long windows_ = 0, rows_ = 0;
    double ms_all_ = 0, ms_dense_ = 0, ms_tier_ = 0, ms_logits_ = 0, ms_pred_ = 0, ms_attn_ = 0, ms_fin_ = 0;
};

}  // namespace strata::v41
