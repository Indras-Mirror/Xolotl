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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
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
    bool restore(Ds41Dense & dense, int64_t rec, int64_t from, ConvRestore & out, std::string & err) {
        out = ConvRestore();
        auto it = std::find_if(recs_.begin(), recs_.end(), [&](const Rec & r) { return r.id == rec; });
        if (it == recs_.end()) { err = "no such record"; return false; }
        (void) from;   // the host store copies whole regions (a region is one tensor's rows [0, n])
        dense_sync(dense);
        out.live_clobbered = true;
        for (const auto & g : it->regions) ggml_backend_tensor_set(g.t, g.b.data(), g.off, g.b.size());
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

}  // namespace strata::v41
