// tools/v41/engine/ds41_batch_slots.hpp - --serve's batch slots (--batch-slots N / --batch N): the pieces of the slot
// protocol that do not need the serve loop's locals.
//
//   - ds41_k_*: the dense half's per-slot API (Ds41Dense::bind_slot / swap_slots / begin_rows / n_slots, the "batch
//     slots" core - docs/BATCHING.md), called only when this build's Ds41Dense has it (detected at
//     compile time, so this file builds on a tree without the core too);
//   - Ds41SlotsStub: the same slot ids on a ONE-state dense half - the bound slot's state lives in the tensors, the
//     others in host copies; a window runs its slots one after another through the old single-sequence path (step /
//     pass_m).  Slower than the core, but byte for byte the solo engine: what the protocol tests run against;
//   - ds41_plan_window: which rows the next batch window takes (one routing class per window, every decoding slot of
//     it, then up to the row limit of the prompt being read) - a pure function, unit-tested.
#pragma once

#include "ds41_dense.hpp"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::v41 {

// ---- the core's per-slot API, when this Ds41Dense has it
template <class D, class = void> struct ds41_has_slots : std::false_type {};
template <class D>
struct ds41_has_slots<D, std::void_t<decltype(std::declval<D &>().bind_slot(0)), decltype(std::declval<D &>().swap_slots(0, 1)),
                                     decltype(std::declval<const D &>().n_slots())>> : std::true_type {};
template <class C, class = void> struct ds41_cfg_has_slots : std::false_type {};
template <class C> struct ds41_cfg_has_slots<C, std::void_t<decltype(std::declval<C &>().n_slots)>> : std::true_type {};

constexpr bool ds41_core_slots() { return ds41_has_slots<Ds41Dense>::value && ds41_cfg_has_slots<Ds41DenseConfig>::value; }

template <class C> void ds41_k_config(C & c, int n) {
    if constexpr (ds41_cfg_has_slots<C>::value) c.n_slots = std::max(0, n);   // (0 = auto, where the core has it)
    else (void) c, (void) n;
}
template <class D> int ds41_k_n(const D & d) {
    if constexpr (ds41_has_slots<D>::value) return d.n_slots();
    else return (void) d, 1;
}
template <class D> bool ds41_k_bind(D & d, int s) {
    if constexpr (ds41_has_slots<D>::value) return d.bind_slot(s);
    else return (void) d, s == 0;
}
template <class D> bool ds41_k_swap(D & d, int a, int b) {
    if constexpr (ds41_has_slots<D>::value) return d.swap_slots(a, b);
    else return (void) d, a == b;
}
/// One batch pass over rows of several slots (slot[i], pos[i], tok[i]); false when the core has no begin_rows.
template <class D> bool ds41_k_begin_rows(D & d, const int * slot, const int * pos, const int * tok, int n) {
    if constexpr (ds41_has_slots<D>::value) {
        std::vector<typename D::Row> rows((size_t) n);
        for (int i = 0; i < n; ++i) { rows[(size_t) i].slot = slot[i]; rows[(size_t) i].pos = pos[i]; rows[(size_t) i].tid = tok[i]; }
        return d.begin_rows(rows.data(), n);
    } else {
        (void) d, (void) slot, (void) pos, (void) tok, (void) n;
        return false;
    }
}

// ---- the stub: N slot ids over the one state of a single-slot dense half
class Ds41SlotsStub {
public:
    /// `positions(s)`: how many positions slot s's conversation has in the state (its history / snapshot); asked when its
    /// state leaves the tensors.  `before_write`: called before the tensors are overwritten (the pool's fence: a save in
    /// flight may still read them).
    void init(Ds41Dense & d, int n, std::function<int64_t(int)> positions, std::function<void()> before_write) {
        d_ = &d;
        host_.assign((size_t) std::max(1, n), Copy());
        res_ = 0;
        positions_ = std::move(positions);
        before_write_ = std::move(before_write);
    }
    int n_slots() const { return (int) host_.size(); }
    int slot() const { return res_; }
    /// The tensors hold slot s's state from here on (the resident slot's state goes to its host copy first).
    bool bind_slot(int s, std::string & err) {
        if (s < 0 || s >= n_slots()) { err = "stub: no slot " + std::to_string(s); return false; }
        if (s == res_) return true;
        Copy out;
        if (!save(positions_ ? positions_(res_) : 0, out, err)) return false;
        host_[(size_t) res_] = std::move(out);
        if (!load(host_[(size_t) s], err)) return false;
        host_[(size_t) s] = Copy();
        res_ = s;
        ++binds_;
        return true;
    }
    /// Exchange slots a and b's states; the bound slot id stays bound (it now holds the other state).  Call BEFORE the
    /// caller exchanges its own per-slot records (positions() is asked for the resident slot as it is now).
    bool swap_slots(int a, int b, std::string & err) {
        if (a < 0 || b < 0 || a >= n_slots() || b >= n_slots()) { err = "stub: bad slot"; return false; }
        if (a == b) return true;
        if (a != res_ && b != res_) { std::swap(host_[(size_t) a], host_[(size_t) b]); return true; }
        const int other = a == res_ ? b : a;
        Copy out;
        if (!save(positions_ ? positions_(res_) : 0, out, err)) return false;
        Copy in = std::move(host_[(size_t) other]);
        host_[(size_t) other] = std::move(out);
        if (!load(in, err)) return false;
        return true;
    }
    int64_t binds() const { return binds_; }

private:
    struct Region { ggml_tensor * t = nullptr; size_t off = 0; std::vector<uint8_t> b; };
    struct Copy {
        bool empty = true;
        int snap_pos = -1;
        std::vector<Region> live, snap;   // live: the live rings (+ positional rows), snap: the snapshot rings (+ rows)
    };
    static bool get(const std::vector<Ds41Dense::StateRegion> & rs, std::vector<Region> & out) {
        out.clear();
        for (const auto & r : rs) {
            Region g{ r.t, r.off, std::vector<uint8_t>(r.bytes) };
            ggml_backend_tensor_get(r.t, g.b.data(), r.off, r.bytes);
            out.push_back(std::move(g));
        }
        return true;
    }
    static void put(const std::vector<Region> & rs) {
        for (const auto & g : rs) ggml_backend_tensor_set(g.t, g.b.data(), g.off, g.b.size());
    }
    // state_regions(n) = the SNAPSHOT rings + rows [0, n] of the positional caches.  The live rings are read by
    // snapshotting them first (snapshot() copies live -> snap); the slot's own snapshot rings are read before that.
    bool save(int64_t positions, Copy & out, std::string & err) {
        out = Copy();
        if (positions < 1) return true;
        if (before_write_) before_write_();
        ggml_backend_synchronize(d_->backend());
        out.snap_pos = d_->snapshot_pos();
        if (!get(d_->state_regions(positions), out.snap)) { err = "stub: snapshot copy failed"; return false; }
        if (!d_->snapshot(0)) { err = "stub: snapshot failed"; return false; }
        if (!get(d_->state_regions(positions), out.live)) { err = "stub: state copy failed"; return false; }
        d_->rearm(out.snap_pos);
        out.empty = false;
        return true;
    }
    bool load(const Copy & in, std::string & err) {
        if (before_write_) before_write_();
        ggml_backend_synchronize(d_->backend());
        if (in.empty) { d_->rearm(-1); return true; }   // a fresh slot: what the tensors hold is never read (positions from 0)
        put(in.live);                                    // live rings into the snapshot tensors ...
        d_->rearm(0);
        if (!d_->restore()) { err = "stub: restore failed"; return false; }   // ... snap -> live
        put(in.snap);                                    // then the slot's own snapshot rings (and the rows again)
        ggml_backend_synchronize(d_->backend());
        d_->rearm(in.snap_pos);
        return true;
    }
    Ds41Dense * d_ = nullptr;
    std::vector<Copy> host_;
    int res_ = 0;
    int64_t binds_ = 0;
    std::function<int64_t(int)> positions_;
    std::function<void()> before_write_;
};

// ---- the window planner
/// A slot as the planner sees it: decoding (one row a window) or reading (rows of its prompt), and its routing class
/// (the decode biases are global in the dense half: rows of different classes never share a window).
struct Ds41PlanSlot {
    bool decoding = false;
    int cls = 0;
};
struct Ds41Plan {
    int cls = -1;                 ///< the window's routing class (-1: nothing to run)
    std::vector<int> dec;         ///< the decoding slots it takes (one row each), ascending
    int read_rows = 0;            ///< rows of the prompt being read it takes (after the decode rows)
};
/// The next window: the first class at or after round-robin position `rr` (a slot index; the reading slot counts at its
/// own index) that has work; every decoding slot of that class (at most max_rows), then min(read_left, max_rows - decode
/// rows) rows of the prompt being read if it is of that class.  `rr` moves past the slot that chose the class, so the
/// classes take turns.  A read always gets at least one row when its class runs (a slot reads only when it is not
/// decoding, so the decode rows of its class are at most max_rows - 1 with max_rows >= the slot count).
inline Ds41Plan ds41_plan_window(const std::vector<Ds41PlanSlot> & slots, int read_slot, int read_cls, int read_left,
                                 int max_rows, int & rr) {
    Ds41Plan p;
    const int n = (int) slots.size();
    if (n == 0 && read_slot < 0) return p;
    const int span = std::max(n, read_slot + 1);
    int pick = -1;
    for (int o = 0; o < span && pick < 0; ++o) {
        const int s = (std::max(0, rr) + o) % span;
        if (s == read_slot && read_left > 0) pick = s;
        else if (s < n && slots[(size_t) s].decoding) pick = s;
    }
    if (pick < 0) return p;
    p.cls = pick == read_slot && read_left > 0 ? read_cls : slots[(size_t) pick].cls;
    rr = (pick + 1) % span;
    for (int s = 0; s < n && (int) p.dec.size() < max_rows; ++s)
        if (slots[(size_t) s].decoding && slots[(size_t) s].cls == p.cls && s != read_slot) p.dec.push_back(s);
    if (read_slot >= 0 && read_left > 0 && read_cls == p.cls)
        p.read_rows = std::min(read_left, std::max(0, max_rows - (int) p.dec.size()));
    return p;
}

}  // namespace strata::v41
