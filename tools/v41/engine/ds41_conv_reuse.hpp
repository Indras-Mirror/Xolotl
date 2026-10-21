// tools/v41/engine/ds41_conv_reuse.hpp - conversation reuse for ds41_generate --serve: position identities (token +
// picture), prefix matching, and the conversation store the serve loop parks conversations in.
//
// A position's identity is its token AND, for an image position, which picture row it is: every image span is one run
// of the same <|image|> id, so a token-only match would "continue" a different picture.  ident[p] = 0 for a text
// position (prompt or generated), else img_ident(key, row) (never 0) with key = the server's img_keys= entry for that
// picture (the hash of its normalized bytes; serve/server.py #2228) or, without keys, an FNV-1a of its grid + rows.
//
// The store keeps conversations at a TURN BOUNDARY: the state Ds41Dense::snapshot() took at the end of a prompt (the
// window ring and ratio-2 compressor state there + every position-indexed compressed-KV / indexer-key row before it),
// the boundary's last-row logits, its tokens and identities.  Ds41ConvStoreHost is the synchronous host-memory store
// (copies of Ds41Dense::state_regions, LRU under a byte budget, supersede-on-extend); the pinned / async store
// (ds41_conv_pool) has the same calls.
#pragma once

#include "ds41_dense.hpp"

#include "ggml-backend.h"
#if defined(DS41_POOL_CUDA)
#include "ds41_conv_pool.hpp"
#include "ds41_conv_pool_dense.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

inline uint64_t conv_mix64(uint64_t x) {   // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

inline uint64_t conv_fnv1a(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/// Row `row` of the picture with key `key`: never 0 (0 = a text position).
inline uint64_t conv_img_ident(uint64_t key, int64_t row) { return conv_mix64(key ^ conv_mix64((uint64_t) row)) | 1ull; }

/// The longest common prefix of two (tokens, identities) sequences; a null identity array reads as all text (0),
/// an identity array shorter than its tokens reads as 0 past its end.
inline size_t conv_common_prefix(const int * a, const uint64_t * ai, size_t ain, size_t na,
                                 const int * b, const uint64_t * bi, size_t bin, size_t nb) {
    const size_t n = std::min(na, nb);
    size_t i = 0;
    for (; i < n; ++i) {
        if (a[i] != b[i]) break;
        const uint64_t x = ai && i < ain ? ai[i] : 0, y = bi && i < bin ? bi[i] : 0;
        if (x != y) break;
    }
    return i;
}

/// A resume at s below a boundary b reads the window [s - SWA + 1, s) from b's ring (positions [b - RAW, b), RAW =
/// SWA + 16) and, at an odd s, the ratio-2 state of s - 1 (b's holds b - 2, b - 1): exact for an even s, b - s <= 16.
constexpr int64_t kConvPartialMax = 16;
inline int64_t conv_partial_resume(int64_t lcp, int64_t boundary) {
    const int64_t s = lcp & ~(int64_t) 1;
    return (s < boundary && boundary - s <= kConvPartialMax) ? s : 0;
}

struct ConvMatch {
    int64_t rec = 0;            ///< 0 = no match
    int boundary = -1, next_boundary_idx = -1;   ///< the store's boundary indices for pos / next_boundary
    int64_t pos = 0;            ///< the boundary when it is a prefix of the prompt (<= lcp, <= n); 0 = none
    int64_t lcp = 0;            ///< common prefix of the record and the prompt
    int64_t next_boundary = -1; ///< the record's boundary above lcp (for conv_partial_resume), -1 = none
    bool has_logits = false;
    int64_t resume() const { return pos > 0 ? pos : next_boundary > 0 ? conv_partial_resume(lcp, next_boundary) : 0; }
};

struct ConvRestore {
    int64_t pos = 0;            ///< the boundary the device state now holds (the snapshot tensors: rearm + restore)
    std::vector<float> logits;  ///< the boundary's last-row logits (empty: none saved)
    std::vector<int> tokens;    ///< the record's tokens [0, pos)
    std::vector<uint64_t> ident;///< ... and identities (empty = all text)
    bool live_clobbered = false;///< a failure after the device was written: the live state is garbage
};

/// The synchronous host store (std::vector copies of Ds41Dense::state_regions; ggml_backend_tensor_get / set).
class Ds41ConvStoreHost {
public:
    Ds41ConvStoreHost(size_t budget_bytes, size_t slots) : budget_(budget_bytes), slots_(slots) {}
    bool enabled() const { return budget_ > 0 && slots_ > 0; }

    /// Save the boundary n (the snapshot tensors hold its state). 0 = not saved (too big for the budget even empty).
    /// The same tokens + identities again only refresh the LRU; a record whose tokens are a prefix of these goes.
    int64_t save(Ds41Dense & dense, const int * tokens, const uint64_t * ident, size_t ident_n, int64_t n,
                 const float * logits, size_t n_logits, std::string & err) {
        if (!enabled() || n < 1) { err = "disabled"; return 0; }
        for (size_t i = 0; i < recs_.size(); ++i) {
            Rec & r = recs_[i];
            if ((int64_t) r.tokens.size() == n &&
                conv_common_prefix(r.tokens.data(), r.ident.data(), r.ident.size(), r.tokens.size(), tokens, ident,
                                   ident_n, (size_t) n) == (size_t) n) {
                Rec keep = std::move(r);
                recs_.erase(recs_.begin() + (std::ptrdiff_t) i);
                recs_.push_back(std::move(keep));   // most recently used last
                return recs_.back().id;
            }
        }
        Rec r;
        r.id = ++next_id_;
        r.tokens.assign(tokens, tokens + n);
        bool any = false;
        for (size_t i = 0; i < ident_n && i < (size_t) n; ++i) any |= ident[i] != 0;
        if (any) { r.ident.assign(ident, ident + std::min<size_t>(ident_n, (size_t) n)); r.ident.resize((size_t) n, 0); }
        if (logits && n_logits) r.logits.assign(logits, logits + n_logits);
        size_t need = r.tokens.size() * 4 + r.ident.size() * 8 + r.logits.size() * 4;
        const auto regs = dense.state_regions(n);
        for (const auto & g : regs) need += g.bytes;
        if (need > budget_) { err = "a " + std::to_string(need >> 20) + " MiB record exceeds the budget"; return 0; }
        // supersede: a record this one extends holds nothing it does not (positions [0, its n) are the same)
        for (size_t i = 0; i < recs_.size();) {
            const Rec & o = recs_[i];
            if (!o.pinned && o.tokens.size() < (size_t) n &&
                conv_common_prefix(o.tokens.data(), o.ident.data(), o.ident.size(), o.tokens.size(), tokens, ident,
                                   ident_n, (size_t) n) == o.tokens.size()) {
                bytes_ -= o.bytes; recs_.erase(recs_.begin() + (std::ptrdiff_t) i); ++superseded_; continue;
            }
            ++i;
        }
        while (!recs_.empty() && (recs_.size() >= slots_ || bytes_ + need > budget_)) {
            auto v = std::find_if(recs_.begin(), recs_.end(), [](const Rec & x) { return !x.pinned; });
            if (v == recs_.end()) { err = "only pinned records left"; return 0; }
            bytes_ -= v->bytes; recs_.erase(v); ++evictions_;
        }
        dense_sync(dense);
        for (const auto & g : regs) {
            Rec::Region rg{ g.t, g.off, std::vector<uint8_t>(g.bytes) };
            ggml_backend_tensor_get(g.t, rg.b.data(), g.off, g.bytes);
            r.regions.push_back(std::move(rg));
        }
        r.bytes = need;
        bytes_ += need;
        recs_.push_back(std::move(r));
        return recs_.back().id;
    }

    /// The record that lets this prompt resume furthest: a boundary that is a prefix (pos) or, failing that, a
    /// boundary within kConvPartialMax above the common prefix (next_boundary).  Ties: the most recent.
    ConvMatch best_prefix(const int * tokens, const uint64_t * ident, size_t ident_n, int64_t n) const {
        ConvMatch best;
        for (size_t i = recs_.size(); i-- > 0;) {
            const Rec & r = recs_[i];
            ConvMatch m;
            m.rec = r.id;
            m.lcp = (int64_t) conv_common_prefix(r.tokens.data(), r.ident.data(), r.ident.size(), r.tokens.size(),
                                                 tokens, ident, ident_n, (size_t) n);
            const int64_t b = (int64_t) r.tokens.size();
            if (m.lcp == b && b <= n) m.pos = b; else m.next_boundary = b;
            m.has_logits = !r.logits.empty();
            if (m.resume() > best.resume()) best = m;
        }
        return best;
    }

    /// Write record `rec`'s boundary state into the device (the snapshot tensors + the position caches).  `from`:
    /// positions [0, from) already hold the same rows (skipped).  The caller then does rearm(pos) + restore().
    bool restore(Ds41Dense & dense, const ConvMatch & m, int64_t from, ConvRestore & out, std::string & err) {
        out = ConvRestore();
        const int64_t rec = m.rec;
        auto it = std::find_if(recs_.begin(), recs_.end(), [&](const Rec & r) { return r.id == rec; });
        if (it == recs_.end()) { err = "no such record"; return false; }
        (void) from;   // the host store copies whole regions (a region is one tensor's rows [0, n])
        // batch slots: the record goes into the BOUND slot's tensors, whichever slot it was saved from (every slot
        // has the same regions, in the same order, for the same n)
        const auto regs = dense.state_regions((int64_t) it->tokens.size());
        if (regs.size() != it->regions.size()) { err = "pool: the record's regions do not match the slot"; return false; }
        for (size_t i = 0; i < regs.size(); ++i)
            if (regs[i].bytes != it->regions[i].b.size() || regs[i].off != it->regions[i].off) { err = "pool: region size mismatch"; return false; }
        dense_sync(dense);
        out.live_clobbered = true;
        for (size_t i = 0; i < regs.size(); ++i) ggml_backend_tensor_set(regs[i].t, it->regions[i].b.data(), regs[i].off, regs[i].bytes);
        dense_sync(dense);
        out.live_clobbered = false;
        out.pos = (int64_t) it->tokens.size();
        out.logits = it->logits;
        out.tokens = it->tokens;
        out.ident = it->ident;
        ++restores_;
        Rec keep = std::move(*it);   // most recently used last
        recs_.erase(it);
        recs_.push_back(std::move(keep));
        return true;
    }

    size_t size() const { return recs_.size(); }
    size_t bytes() const { return bytes_; }
    size_t evictions() const { return evictions_; }
    size_t superseded() const { return superseded_; }
    size_t clear() { const size_t n = recs_.size(); recs_.clear(); bytes_ = 0; return n; }
    bool fence(std::string * = nullptr) { return true; }
    std::string stats_line() const {
        return "host store: " + std::to_string(recs_.size()) + " records, " + std::to_string(bytes_ >> 20) + " MiB, " +
               std::to_string(evictions_) + " evicted, " + std::to_string(superseded_) + " superseded";
    }

private:
    struct Rec {
        int64_t id = 0;
        std::vector<int> tokens;
        std::vector<uint64_t> ident;
        std::vector<float> logits;
        struct Region { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
        std::vector<Region> regions;
        size_t bytes = 0;
        bool pinned = false;
    };
    static void dense_sync(Ds41Dense & d) { ggml_backend_synchronize(d.backend()); }
    std::deque<Rec> recs_;   // least recently used first
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, restores_ = 0;
    int64_t next_id_ = 0;
};

#if defined(DS41_POOL_CUDA)
/// The pinned pool (ds41_conv_pool: pinned slabs, shared prefix chunks, up to 4 boundaries a record,
/// async copies on its own stream) behind the same calls.  cuda = false on the CPU backend (the tensors are host memory).
class Ds41ConvStorePinned {
public:
    bool init(const Ds41Dense & d, int64_t ctx, size_t budget, bool cuda, std::string & err) {
        Ds41PoolLayout lay;
        if (!ds41_pool_from_dense(d, ctx, lay, snap_, live_)) { err = "pool layout from the dense half failed"; return false; }
        ctx_ = ctx;
        slot_ = d.slot();
        Ds41PoolConfig c;
        c.budget_bytes = budget;
        c.cuda = cuda;
        return pool_.init(lay, c, err);
    }
    int64_t save(Ds41Dense & dense, const int * tokens, const uint64_t * ident, size_t ident_n, int64_t n,
                 const float * logits, size_t n_logits, std::string & err) {
        const uint64_t * ids = pad(ident, ident_n, n);
        ggml_backend_synchronize(dense.backend());   // the device data is final (the save reads it on the pool's stream)
        if (!retarget(dense, err)) return 0;
        static_assert(sizeof(int) == sizeof(int32_t), "token ids are int32");
        return (int64_t) pool_.save(snap_, reinterpret_cast<const int32_t *>(tokens), ids, n, logits, n_logits, err);
    }
    ConvMatch best_prefix(const int * tokens, const uint64_t * ident, size_t ident_n, int64_t n) {
        const uint64_t * ids = pad(ident, ident_n, n);
        const Ds41PoolMatch pm = pool_.best_prefix(reinterpret_cast<const int32_t *>(tokens), ids, n);
        ConvMatch m;
        m.rec = (int64_t) pm.rec;
        m.pos = pm.pos;
        m.lcp = pm.lcp;
        m.boundary = pm.boundary;
        m.has_logits = pm.has_logits;
        m.next_boundary = pm.next_pos;
        m.next_boundary_idx = pm.next_boundary;
        if (m.pos > 0 && m.next_boundary > 0 && conv_partial_resume(m.lcp, m.next_boundary) <= m.pos) m.next_boundary = -1;
        if (m.pos == 0 && !ids) {
            // a record with no boundary under the common prefix: best_prefix skips it, but a boundary within 16 above
            // the prefix resumes too (conv_partial_resume).  Text prompts only (records() carries no identities).
            for (const auto & r : pool_.records()) {
                const std::vector<int32_t> t = pool_.tokens(r.id);
                const int64_t l = (int64_t) conv_common_prefix(t.data(), nullptr, 0, t.size(), tokens, nullptr, 0, (size_t) n);
                for (size_t b = 0; b < r.boundaries.size(); ++b) {
                    const int64_t s = r.boundaries[b].pos > l ? conv_partial_resume(l, r.boundaries[b].pos) : 0;
                    if (s > m.resume()) {
                        m = ConvMatch();
                        m.rec = (int64_t) r.id; m.lcp = l; m.next_boundary = r.boundaries[b].pos; m.next_boundary_idx = (int) b;
                    }
                }
            }
        }
        return m;
    }
    bool restore(Ds41Dense & dense, const ConvMatch & m, int64_t from, ConvRestore & out, std::string & err) {
        out = ConvRestore();
        if (!retarget(dense, err)) return false;
        const bool at_pos = m.pos > 0 && m.resume() == m.pos;
        const int b = at_pos ? m.boundary : m.next_boundary_idx;
        const int64_t bpos = at_pos ? m.pos : m.next_boundary;
        Ds41PoolRestore rr;
        if (!pool_.restore((Ds41PoolId) m.rec, b, std::min(from, bpos), snap_, rr, err)) { out.live_clobbered = rr.live_clobbered; return false; }
        if (!pool_.wait(rr, &err)) { out.live_clobbered = true; return false; }
        out.pos = rr.pos;
        out.logits = std::move(rr.logits);
        return true;
    }
    bool fence(std::string * err = nullptr) { return pool_.fence(err); }
    size_t size() const { return pool_.stats().records; }
    size_t bytes() const { return pool_.stats().used_bytes; }
    size_t evictions() const { return pool_.stats().evictions; }
    std::string stats_line() const { return pool_.stats_line(); }

private:
    const uint64_t * pad(const uint64_t * ident, size_t ident_n, int64_t n) {
        bool any = false;
        for (size_t i = 0; i < ident_n && (int64_t) i < n; ++i) any |= ident[i] != 0;
        if (!any) return nullptr;
        idbuf_.assign(ident, ident + std::min<size_t>(ident_n, (size_t) n));
        idbuf_.resize((size_t) n, 0);
        return idbuf_.data();
    }
    // batch slots: a record is saved from / restored into the BOUND slot's tensors (every slot has the same layout);
    // the pointers are taken again whenever the bound slot or its state (swap_slots) may have changed - cheap
    bool retarget(const Ds41Dense & d, std::string & err) {
        if (d.n_slots() <= 1) return true;   // one slot: the pointers taken at init, as always
        Ds41PoolLayout lay;
        Ds41PoolTargets s, l;
        if (!ds41_pool_from_dense(d, ctx_, lay, s, l) || s.layout_fp != snap_.layout_fp) {
            err = "pool: slot " + std::to_string(d.slot()) + " does not have the pool's layout";
            return false;
        }
        snap_ = std::move(s);
        live_ = std::move(l);
        slot_ = d.slot();
        return true;
    }
    Ds41ConvPool pool_;
    Ds41PoolTargets snap_, live_;
    int64_t ctx_ = 0;
    int slot_ = 0;
    std::vector<uint64_t> idbuf_;
};
#endif

/// The serve loop's store: the pinned pool when built with it (STRATA_CONV_POOL=host: the synchronous host store).
class Ds41ConvStore {
public:
    bool init(const Ds41Dense & d, int64_t ctx, size_t budget, size_t slots, bool cuda, std::string & err) {
        if (budget == 0) return true;
        const char * e = std::getenv("STRATA_CONV_POOL");
#if defined(DS41_POOL_CUDA)
        if (!(e && std::strcmp(e, "host") == 0)) {
            pin_ = std::make_unique<Ds41ConvStorePinned>();
            if (pin_->init(d, ctx, budget, cuda, err)) { kind_ = "pinned"; return true; }
            pin_.reset();
            return false;
        }
#endif
        (void) d; (void) ctx; (void) cuda; (void) e;
        host_ = std::make_unique<Ds41ConvStoreHost>(budget, slots);
        kind_ = "host";
        return true;
    }
    const char * kind() const { return kind_; }
#if defined(DS41_POOL_CUDA)
#define DS41_STORE_CALL(x) (pin_ ? pin_->x : host_->x)
#else
#define DS41_STORE_CALL(x) (host_->x)
#endif
    int64_t save(Ds41Dense & d, const int * t, const uint64_t * id, size_t idn, int64_t n, const float * lg, size_t nl, std::string & err) {
        return DS41_STORE_CALL(save(d, t, id, idn, n, lg, nl, err));
    }
    ConvMatch best_prefix(const int * t, const uint64_t * id, size_t idn, int64_t n) { return DS41_STORE_CALL(best_prefix(t, id, idn, n)); }
    bool restore(Ds41Dense & d, const ConvMatch & m, int64_t from, ConvRestore & out, std::string & err) {
        return DS41_STORE_CALL(restore(d, m, from, out, err));
    }
    bool fence(std::string * err = nullptr) { return DS41_STORE_CALL(fence(err)); }
    size_t size() const { return DS41_STORE_CALL(size()); }
    size_t bytes() const { return DS41_STORE_CALL(bytes()); }
    size_t evictions() const { return DS41_STORE_CALL(evictions()); }
    std::string stats_line() const { return DS41_STORE_CALL(stats_line()); }
#undef DS41_STORE_CALL
private:
    std::unique_ptr<Ds41ConvStoreHost> host_;
#if defined(DS41_POOL_CUDA)
    std::unique_ptr<Ds41ConvStorePinned> pin_;
#endif
    const char * kind_ = "off";
};

}  // namespace strata::v41
