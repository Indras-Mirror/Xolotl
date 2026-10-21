// tools/v41/engine/ds41_conv_pool.cpp - see ds41_conv_pool.hpp.  Built with DS41_POOL_CUDA for the engine and the card
// bench; without it (test_ds41_conv_pool) it is plain host code and makes no CUDA call at all.
#include "ds41_conv_pool.hpp"
#include "ds41_sys.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <new>

#ifdef DS41_POOL_CUDA
#include <cuda_runtime.h>
#endif

namespace strata::v41 {

// ------------------------------------------------------------------ layout
namespace {
uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
template <class T> uint64_t fnv_v(uint64_t h, T v) { return fnv(h, &v, sizeof v); }

// 64-bit checksum of a slab (cfg.verify): 8 bytes at a time, cheap enough for a debug mode
uint64_t sum64(const uint8_t * p, size_t n) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 0xff51afd7ed558ccdull;
        h ^= h >> 29;
    }
    for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
}  // namespace

uint64_t Ds41PoolLayout::fingerprint() const {
    uint64_t h = 0xcbf29ce484222325ull;
    h = fnv_v(h, kDs41PoolVersion);
    h = fnv_v(h, (uint64_t) streams.size());
    for (const auto & s : streams) {
        h = fnv_v(h, (uint64_t) s.row_bytes);
        h = fnv_v(h, (int64_t) s.ratio);
        h = fnv_v(h, s.max_rows);
    }
    h = fnv_v(h, (uint64_t) regions.size());
    for (const auto & r : regions) h = fnv_v(h, (uint64_t) r.bytes);
    h = fnv_v(h, max_positions);
    return h ? h : 1;
}

size_t Ds41PoolLayout::bytes_per_position() const {
    size_t b = 0;
    for (const auto & s : streams) b += s.row_bytes / (size_t) std::max(1, s.ratio);
    return b;
}

size_t Ds41PoolLayout::region_bytes() const {
    size_t b = 0;
    for (const auto & r : regions) b += r.bytes;
    return b;
}

bool Ds41PoolLayout::valid(std::string & err) const {
    if (max_positions < 1) { err = "pool layout: max_positions < 1"; return false; }
    for (const auto & s : streams) {
        if (!s.row_bytes || s.ratio < 1) { err = "pool layout: stream " + s.name + " has no rows"; return false; }
        if (s.max_rows * s.ratio < max_positions) {
            err = "pool layout: stream " + s.name + " holds fewer than max_positions positions";
            return false;
        }
    }
    for (const auto & r : regions)
        if (!r.bytes) { err = "pool layout: region " + r.name + " is empty"; return false; }
    return true;
}

// ------------------------------------------------------------------ implementation
struct Ds41ConvPool::Impl {
    Ds41PoolLayout L;
    Ds41PoolConfig cfg;
    uint64_t fp = 0;
    bool ok = false;
    mutable std::mutex mu;

    // chunk geometry: C positions a slab; a stream's rows of chunk k start at soff[s] inside the slab
    int64_t C = 0;
    std::vector<size_t> soff;
    size_t chunk_used = 0;        // bytes of a full chunk
    size_t region_slabs = 0;      // slabs a boundary's regions take

    // slabs
    std::vector<uint8_t *> extents;
    size_t allocated = 0;         // pinned bytes taken (extents)
    std::vector<uint8_t *> slab_ptr;
    std::vector<uint32_t> slab_ref;
    std::vector<uint64_t> slab_sum;   // cfg.verify
    std::vector<uint32_t> slab_free;
    size_t slabs_used = 0;
    size_t host_bytes = 0;

    struct Bnd {
        int64_t pos = 0;
        std::vector<uint32_t> slabs;
        std::vector<float> logits;
        std::vector<std::vector<uint8_t>> blobs;
        size_t host() const {
            size_t n = logits.size() * sizeof(float);
            for (const auto & b : blobs) n += b.size();
            return n;
        }
    };
    struct Rec {
        Ds41PoolId id = 0;
        std::vector<int32_t> tok;
        std::vector<uint64_t> ids;              // empty = all zero
        std::vector<uint32_t> chunks;           // slab of chunk k (positions [kC, min((k+1)C, n)))
        std::vector<std::shared_ptr<Bnd>> bnd;  // ascending pos
        uint64_t stamp = 0;
        bool pinned = false;
        int64_t n() const { return (int64_t) tok.size(); }
        size_t meta() const { return tok.size() * sizeof(int32_t) + ids.size() * sizeof(uint64_t); }
    };
    std::map<Ds41PoolId, Rec> recs;
    Ds41PoolId next_id = 1;
    uint64_t clock = 0;

    // copies
    uint64_t seq = 0;             // batches queued
#ifdef DS41_POOL_CUDA
    cudaStream_t st = nullptr;
    struct Op { uint64_t seq; bool save; size_t bytes; cudaEvent_t a, b; };
    std::deque<Op> ops;           // queued batches (timing events), oldest first
    std::vector<cudaEvent_t> ev_free;
#endif
    Ds41PoolStats S;

    // ---------------------------------------------------------------- slab allocator
    bool grow(std::string & err) {
        if (allocated >= cfg.budget_bytes) return false;
        const size_t want = std::min(cfg.extent_bytes, cfg.budget_bytes - allocated);
        const size_t ns = want / cfg.slab_bytes;
        if (ns == 0) return false;
        uint8_t * p = nullptr;
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            cudaSetDevice(cfg.device);
            const cudaError_t e = cudaHostAlloc((void **) &p, ns * cfg.slab_bytes, cudaHostAllocPortable);
            if (e != cudaSuccess) { err = std::string("pool: cudaHostAlloc: ") + cudaGetErrorString(e); cudaGetLastError(); return false; }
        } else
#endif
        {
            p = (uint8_t *) ds41_sys::aligned_alloc(4096, ns * cfg.slab_bytes);   // MSVC has no std::aligned_alloc
            if (!p) { err = "pool: out of host memory"; return false; }
        }
        extents.push_back(p);
        allocated += ns * cfg.slab_bytes;
        for (size_t i = 0; i < ns; ++i) {
            slab_ptr.push_back(p + i * cfg.slab_bytes);
            slab_ref.push_back(0);
            slab_sum.push_back(0);
            slab_free.push_back((uint32_t) (slab_ptr.size() - 1));
        }
        // pop_back takes the lowest index: keep the free list descending so slabs fill in address order
        std::sort(slab_free.begin(), slab_free.end(), std::greater<uint32_t>());
        return true;
    }
    size_t used_bytes() const { return slabs_used * cfg.slab_bytes + host_bytes; }
    // can `slabs` more slabs and `hb` more host bytes be held without evicting?
    bool fits(size_t slabs, size_t hb) const {
        return used_bytes() + slabs * cfg.slab_bytes + hb <= cfg.budget_bytes;
    }
    uint32_t take_slab() {
        const uint32_t s = slab_free.back();
        slab_free.pop_back();
        slab_ref[s] = 1;
        ++slabs_used;
        return s;
    }
    void unref(uint32_t s) {
        if (--slab_ref[s] == 0) { slab_free.push_back(s); --slabs_used; }
    }
    void unref_bnd(std::shared_ptr<Bnd> & b) {
        if (b.use_count() == 1) {
            for (uint32_t s : b->slabs) unref(s);
            host_bytes -= b->host();
        }
        b.reset();
    }
    void drop(std::map<Ds41PoolId, Rec>::iterator it) {
        Rec & r = it->second;
        for (uint32_t s : r.chunks) unref(s);
        for (auto & b : r.bnd) unref_bnd(b);
        host_bytes -= r.meta();
        recs.erase(it);
    }
    // the least recently used unpinned record other than `keep`
    bool evict_one(Ds41PoolId keep) {
        auto v = recs.end();
        for (auto it = recs.begin(); it != recs.end(); ++it)
            if (!it->second.pinned && it->first != keep && (v == recs.end() || it->second.stamp < v->second.stamp)) v = it;
        if (v == recs.end()) return false;
        drop(v);
        ++S.evictions;
        return true;
    }
    // room for `slabs` free slabs and `hb` host bytes: grow extents first, then evict
    bool make_room(size_t slabs, size_t hb, Ds41PoolId keep, std::string & err) {
        std::string grow_err;
        for (;;) {
            const bool f = fits(slabs, hb);
            if (f && slab_free.size() >= slabs) return true;
            if (f && grow_err.empty() && grow(grow_err)) continue;
            if (!evict_one(keep)) {
                err = !grow_err.empty() ? grow_err
                                        : "pool: no room (" + std::to_string(slabs) + " slabs + " + std::to_string(hb) +
                                              " host bytes: over the budget, or the rest is pinned)";
                return false;
            }
        }
    }

    // ---------------------------------------------------------------- copies
    bool begin_batch(std::string & err) {
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            cudaSetDevice(cfg.device);
            Op op{ ++seq, false, 0, nullptr, nullptr };
            for (cudaEvent_t * e : { &op.a, &op.b }) {
                if (!ev_free.empty()) { *e = ev_free.back(); ev_free.pop_back(); }
                else if (cudaEventCreate(e) != cudaSuccess) { err = "pool: cudaEventCreate"; cudaGetLastError(); return false; }
            }
            cudaEventRecord(op.a, st);
            ops.push_back(op);
            return true;
        }
#endif
        (void) err;
        ++seq;
        return true;
    }
    // one copy of the current batch; false = the CUDA call failed (err set)
    bool copy(void * dst, const void * src, size_t n, bool d2h, std::string & err) {
        if (!n) return true;
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            const cudaError_t e = cudaMemcpyAsync(dst, src, n, d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, st);
            if (e != cudaSuccess) { err = std::string("pool: cudaMemcpyAsync: ") + cudaGetErrorString(e); cudaGetLastError(); return false; }
            return true;
        }
#endif
        (void) d2h; (void) err;
        std::memcpy(dst, src, n);
        return true;
    }
    void end_batch(bool save, size_t bytes) {
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            Op & op = ops.back();
            op.save = save;
            op.bytes = bytes;
            cudaEventRecord(op.b, st);
            return;
        }
#endif
        (void) save; (void) bytes;
    }
    // account finished batches (timing); `block` = wait for all of them
    bool reap(bool block, std::string * err) {
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            bool good = true;
            while (!ops.empty()) {
                Op & op = ops.front();
                cudaError_t e = block ? cudaEventSynchronize(op.b) : cudaEventQuery(op.b);
                if (e == cudaErrorNotReady) { cudaGetLastError(); break; }
                if (e != cudaSuccess) {
                    good = false;
                    if (err) *err = std::string("pool: copy failed: ") + cudaGetErrorString(e);
                    cudaGetLastError();
                } else {
                    float ms = 0;
                    cudaEventElapsedTime(&ms, op.a, op.b);
                    if (op.save) { S.last_save_ms = ms; S.total_save_ms += ms; }
                    else { S.last_restore_ms = ms; S.total_restore_ms += ms; }
                }
                ev_free.push_back(op.a);
                ev_free.push_back(op.b);
                ops.pop_front();
            }
            return good;
        }
#endif
        (void) block; (void) err;
        return true;
    }
    bool wait_seq(uint64_t s, std::string * err) {
#ifdef DS41_POOL_CUDA
        if (cfg.cuda) {
            bool good = true;
            while (!ops.empty() && ops.front().seq <= s) {
                Op & op = ops.front();
                const cudaError_t e = cudaEventSynchronize(op.b);
                if (e != cudaSuccess) {
                    good = false;
                    if (err) *err = std::string("pool: copy failed: ") + cudaGetErrorString(e);
                    cudaGetLastError();
                } else {
                    float ms = 0;
                    cudaEventElapsedTime(&ms, op.a, op.b);
                    if (op.save) { S.last_save_ms = ms; S.total_save_ms += ms; }
                    else { S.last_restore_ms = ms; S.total_restore_ms += ms; }
                }
                ev_free.push_back(op.a);
                ev_free.push_back(op.b);
                ops.pop_front();
            }
            return good;
        }
#endif
        (void) s; (void) err;
        return true;
    }

    // the stream rows of positions [a, e) inside chunk k: one copy per stream (d2h: device -> slab)
    bool copy_chunk(uint32_t slab, int64_t k, int64_t a, int64_t e, const Ds41PoolTargets & dev, bool d2h,
                    size_t & bytes, std::string & err) {
        uint8_t * base = slab_ptr[slab];
        for (size_t s = 0; s < L.streams.size(); ++s) {
            const auto & sp = L.streams[s];
            const int64_t r = sp.ratio;
            const int64_t r0 = std::max<int64_t>(k * C / r, a / r), r1 = std::min<int64_t>((k + 1) * C / r, e / r);
            if (r1 <= r0) continue;
            uint8_t * host = base + soff[s] + (size_t) (r0 - k * C / r) * sp.row_bytes;
            uint8_t * devp = (uint8_t *) dev.stream[s] + (size_t) r0 * sp.row_bytes;
            const size_t n = (size_t) (r1 - r0) * sp.row_bytes;
            if (!(d2h ? copy(host, devp, n, true, err) : copy(devp, host, n, false, err))) return false;
            bytes += n;
        }
        return true;
    }
    // the regions, packed back to back over the boundary's slabs
    bool copy_regions(const std::vector<uint32_t> & slabs, const Ds41PoolTargets & dev, bool d2h, size_t & bytes,
                      std::string & err) {
        size_t off = 0;   // byte offset in the packed stream
        for (size_t i = 0; i < L.regions.size(); ++i) {
            size_t left = L.regions[i].bytes, roff = 0;
            while (left) {
                const size_t si = off / cfg.slab_bytes, so = off % cfg.slab_bytes;
                const size_t n = std::min(left, cfg.slab_bytes - so);
                uint8_t * host = slab_ptr[slabs[si]] + so;
                uint8_t * devp = (uint8_t *) dev.region[i] + roff;
                if (!(d2h ? copy(host, devp, n, true, err) : copy(devp, host, n, false, err))) return false;
                bytes += n;
                off += n; roff += n; left -= n;
            }
        }
        return true;
    }
    void checksum_chunks(const Rec & r, size_t from_chunk) {
        for (size_t k = from_chunk; k < r.chunks.size(); ++k)
            slab_sum[r.chunks[k]] = sum64(slab_ptr[r.chunks[k]], cfg.slab_bytes);
    }

    bool targets_ok(const Ds41PoolTargets & dev, std::string & err) const {
        if (dev.layout_fp && dev.layout_fp != fp) { err = "pool: layout fingerprint differs (version or shape)"; return false; }
        if (dev.stream.size() != L.streams.size() || dev.region.size() != L.regions.size()) {
            err = "pool: targets do not match the layout"; return false;
        }
        for (void * p : dev.stream) if (!p) { err = "pool: null stream pointer"; return false; }
        for (void * p : dev.region) if (!p) { err = "pool: null region pointer"; return false; }
        return true;
    }
    static int64_t lcp_of(const Rec & r, const int32_t * tok, const uint64_t * ids, int64_t n) {
        const int64_t m = std::min<int64_t>(r.n(), n);
        int64_t i = std::mismatch(r.tok.begin(), r.tok.begin() + m, tok).first - r.tok.begin();
        // identities: compare up to i (both "all zero" = equal)
        if (r.ids.empty() && !ids) return i;
        for (int64_t j = 0; j < i; ++j) {
            const uint64_t a = r.ids.empty() ? 0 : r.ids[(size_t) j];
            const uint64_t b = ids ? ids[j] : 0;
            if (a != b) return j;
        }
        return i;
    }
    size_t pinned_bytes() const {
        size_t b = 0;
        for (const auto & [id, r] : recs)
            if (r.pinned) b += (r.chunks.size() + r.bnd.size() * region_slabs) * cfg.slab_bytes + r.meta();
        return b;
    }
};

Ds41ConvPool::Ds41ConvPool() : p_(std::make_unique<Impl>()) {}

Ds41ConvPool::~Ds41ConvPool() {
    Impl & im = *p_;
#ifdef DS41_POOL_CUDA
    if (im.cfg.cuda && im.ok) {
        cudaSetDevice(im.cfg.device);
        if (im.st) cudaStreamSynchronize(im.st);
        im.reap(true, nullptr);
        for (cudaEvent_t e : im.ev_free) cudaEventDestroy(e);
        if (im.st) cudaStreamDestroy(im.st);
        for (uint8_t * e : im.extents) cudaFreeHost(e);
        return;
    }
#endif
    for (uint8_t * e : im.extents) ds41_sys::aligned_free(e);   // _aligned_free on Windows, free elsewhere
}

bool Ds41ConvPool::init(const Ds41PoolLayout & layout, const Ds41PoolConfig & cfg, std::string & err) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (im.ok) { err = "pool: init twice"; return false; }
    if (!layout.valid(err)) return false;
#ifndef DS41_POOL_CUDA
    if (cfg.cuda) { err = "pool: built without CUDA (cfg.cuda must be false)"; return false; }
#endif
    im.L = layout;
    im.cfg = cfg;
    if (im.cfg.slab_bytes < 4096) { err = "pool: slab_bytes < 4096"; return false; }
    im.cfg.slab_bytes = (im.cfg.slab_bytes + 4095) / 4096 * 4096;
    im.cfg.extent_bytes = std::max(im.cfg.slab_bytes, im.cfg.extent_bytes / im.cfg.slab_bytes * im.cfg.slab_bytes);
    if (im.cfg.max_boundaries < 1) im.cfg.max_boundaries = 1;
    im.fp = layout.fingerprint();
    // C: the most positions whose rows fit one slab, a multiple of every ratio
    int64_t lc = 1;
    for (const auto & s : layout.streams) {
        int64_t a = lc, b = s.ratio;
        while (b) { const int64_t t = a % b; a = b; b = t; }
        lc = lc / a * s.ratio;
    }
    auto chunk_size = [&](int64_t c) {
        size_t b = 0;
        for (const auto & s : layout.streams) b += (size_t) (c / s.ratio) * s.row_bytes;
        return b;
    };
    if (layout.streams.empty()) im.C = layout.max_positions;
    else {
        const size_t bpp = std::max<size_t>(1, layout.bytes_per_position());
        int64_t c = (int64_t) (im.cfg.slab_bytes / bpp) / lc * lc;
        while (c > 0 && chunk_size(c) > im.cfg.slab_bytes) c -= lc;
        if (c < lc) { err = "pool: a slab cannot hold one chunk (slab_bytes too small)"; return false; }
        im.C = c;
    }
    im.soff.clear();
    size_t off = 0;
    for (const auto & s : layout.streams) { im.soff.push_back(off); off += (size_t) (im.C / s.ratio) * s.row_bytes; }
    im.chunk_used = off;
    im.region_slabs = (layout.region_bytes() + im.cfg.slab_bytes - 1) / im.cfg.slab_bytes;
#ifdef DS41_POOL_CUDA
    if (im.cfg.cuda) {
        cudaError_t e = cudaSetDevice(im.cfg.device);
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&im.st, cudaStreamNonBlocking);
        if (e != cudaSuccess) { err = std::string("pool: CUDA stream: ") + cudaGetErrorString(e); cudaGetLastError(); return false; }
    }
#endif
    if (im.cfg.prealloc) {
        while (im.allocated < im.cfg.budget_bytes)
            if (!im.grow(err)) {
                if (!err.empty()) return false;
                break;
            }
    }
    im.S.budget_bytes = im.cfg.budget_bytes;
    im.S.slab_bytes = im.cfg.slab_bytes;
    im.S.chunk_positions = (size_t) im.C;
    im.ok = true;
    return true;
}

bool Ds41ConvPool::ready() const { return p_->ok; }
const Ds41PoolLayout & Ds41ConvPool::layout() const { return p_->L; }
size_t Ds41ConvPool::chunk_positions() const { return (size_t) p_->C; }

Ds41PoolId Ds41ConvPool::save(const Ds41PoolTargets & dev, const int32_t * tokens, const uint64_t * ids, int64_t n,
                              const float * logits, size_t n_logits, std::string & err, const Ds41PoolSaveOpts & opts) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto fail = [&](const std::string & e) { err = e; ++im.S.saves_failed; return (Ds41PoolId) 0; };
    if (!im.ok) return fail("pool: not initialized");
    if (n < 1 || n > im.L.max_positions) return fail("pool: save of " + std::to_string(n) + " positions (max " +
                                                     std::to_string(im.L.max_positions) + ")");
    if (!tokens) return fail("pool: no tokens");
    if (!im.targets_ok(dev, err)) { ++im.S.saves_failed; return 0; }
    im.reap(false, nullptr);
    bool any_id = false;
    if (ids) for (int64_t i = 0; i < n && !any_id; ++i) any_id = ids[i] != 0;

    // the record sharing the longest prefix; an exact one with a boundary at n is only touched
    Ds41PoolId base = 0;
    int64_t blcp = 0;
    for (auto & [id, r] : im.recs) {
        const int64_t l = Impl::lcp_of(r, tokens, any_id ? ids : nullptr, n);
        if (l > blcp || (l == blcp && base && r.stamp > im.recs.at(base).stamp)) { blcp = l; base = id; }
    }
    if (base && blcp == n) {
        Impl::Rec & r = im.recs.at(base);
        if (r.n() == n && !r.bnd.empty() && r.bnd.back()->pos == n) {
            r.stamp = ++im.clock;
            if (opts.pin && !r.pinned) {
                if (im.pinned_bytes() + (r.chunks.size() + r.bnd.size() * im.region_slabs) * im.cfg.slab_bytes >
                    (size_t) (im.cfg.pinned_max_frac * (double) im.cfg.budget_bytes)) return fail("pool: pinned share full");
                r.pinned = true;
            }
            ++im.S.saves_touched;
            return base;
        }
    }
    const int64_t nchunks = (n + im.C - 1) / im.C;
    // shared chunks: chunk k is shared iff every position the new record needs from it lies in the common prefix
    std::vector<uint32_t> chunks;
    std::vector<std::shared_ptr<Impl::Bnd>> bnd;
    int64_t shared = 0;
    if (base) {
        const Impl::Rec & b = im.recs.at(base);
        while (shared < nchunks && shared < (int64_t) b.chunks.size() &&
               std::min<int64_t>((shared + 1) * im.C, n) <= blcp) ++shared;
        for (int64_t k = 0; k < shared; ++k) { chunks.push_back(b.chunks[(size_t) k]); ++im.slab_ref[b.chunks[(size_t) k]]; }
        for (const auto & x : b.bnd) if (x->pos <= blcp && x->pos <= n) bnd.push_back(x);
    }
    const bool new_bnd = bnd.empty() || bnd.back()->pos != n;
    // the boundary cap, before making room: keep the first (the deepest shared root) and the newest
    while ((int) bnd.size() + (new_bnd ? 1 : 0) > im.cfg.max_boundaries && !bnd.empty()) {
        const size_t v = bnd.size() > 1 ? 1 : 0;
        im.unref_bnd(bnd[v]);
        bnd.erase(bnd.begin() + (std::ptrdiff_t) v);
    }
    const size_t need = (size_t) (nchunks - shared) + (new_bnd ? im.region_slabs : 0);
    size_t hb = (size_t) n * sizeof(int32_t) + (any_id ? (size_t) n * sizeof(uint64_t) : 0);
    if (new_bnd) {
        hb += (logits ? n_logits : 0) * sizeof(float);
        for (const auto & bl : opts.blobs) hb += bl.bytes;
    }
    auto release = [&]() {
        for (uint32_t s : chunks) im.unref(s);
        for (auto & x : bnd) im.unref_bnd(x);
    };
    if (opts.pin) {
        const size_t mine = (size_t) nchunks + bnd.size() * im.region_slabs + (new_bnd ? im.region_slabs : 0);
        if (im.pinned_bytes() + mine * im.cfg.slab_bytes > (size_t) (im.cfg.pinned_max_frac * (double) im.cfg.budget_bytes)) {
            release();
            return fail("pool: pinned share full");
        }
    }
    // make room: base may leave too (what we share from it is referenced already)
    if (!im.make_room(need, hb, 0, err)) { release(); ++im.S.saves_failed; return 0; }

    // queue the copies
    if (!im.begin_batch(err)) { release(); ++im.S.saves_failed; return 0; }
    size_t bytes = 0;
    bool good = true;
    for (int64_t k = shared; k < nchunks && good; ++k) {
        const uint32_t s = im.take_slab();
        chunks.push_back(s);
        good = im.copy_chunk(s, k, k * im.C, std::min<int64_t>((k + 1) * im.C, n), dev, true, bytes, err);
    }
    std::shared_ptr<Impl::Bnd> nb;
    if (good && new_bnd) {
        nb = std::make_shared<Impl::Bnd>();
        nb->pos = n;
        for (size_t i = 0; i < im.region_slabs; ++i) nb->slabs.push_back(im.take_slab());
        good = im.copy_regions(nb->slabs, dev, true, bytes, err);
        if (logits && n_logits) nb->logits.assign(logits, logits + n_logits);
        for (const auto & bl : opts.blobs)
            nb->blobs.emplace_back((const uint8_t *) bl.data, (const uint8_t *) bl.data + bl.bytes);
    }
    im.end_batch(true, bytes);
    if (!good) {
        // a failed D2H changed no device data; give the slabs back (stream order keeps their reuse safe)
        release();
        if (nb) for (uint32_t s : nb->slabs) im.unref(s);
        ++im.S.saves_failed;
        return 0;
    }
    if (nb) { im.host_bytes += nb->host(); bnd.push_back(std::move(nb)); }

    Impl::Rec r;
    r.id = im.next_id++;
    r.tok.assign(tokens, tokens + n);
    if (any_id) r.ids.assign(ids, ids + n);
    r.chunks = std::move(chunks);
    r.bnd = std::move(bnd);
    r.stamp = ++im.clock;
    r.pinned = opts.pin;
    im.host_bytes += r.meta();
    const Ds41PoolId id = r.id;
    // the base this record continues (its tokens a prefix of ours) is superseded unless pinned
    if (base) {
        auto it = im.recs.find(base);
        if (it != im.recs.end() && !it->second.pinned && blcp == it->second.n() && it->second.n() <= n) {
            im.drop(it);
            ++im.S.superseded;
        }
    }
    if (im.cfg.verify) {
        std::string e2;
        im.wait_seq(im.seq, &e2);
        im.checksum_chunks(r, 0);
        for (const auto & x : r.bnd) for (uint32_t s : x->slabs) im.slab_sum[s] = sum64(im.slab_ptr[s], im.cfg.slab_bytes);
    }
    im.recs.emplace(id, std::move(r));
    ++im.S.saves;
    im.S.bytes_d2h += bytes;
    im.S.shared_chunks += (uint64_t) shared;
    im.S.bytes_shared += (uint64_t) shared * im.chunk_used;
    return id;
}

Ds41PoolMatch Ds41ConvPool::best_prefix(const int32_t * tokens, const uint64_t * ids, int64_t n) const {
    const Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    Ds41PoolMatch best;
    uint64_t best_stamp = 0;
    if (!im.ok || !tokens || n < 1) return best;
    bool any_id = false;
    if (ids) for (int64_t i = 0; i < n && !any_id; ++i) any_id = ids[i] != 0;
    for (const auto & [id, r] : im.recs) {
        const int64_t l = Impl::lcp_of(r, tokens, any_id ? ids : nullptr, n);
        int b = -1;
        for (int i = 0; i < (int) r.bnd.size(); ++i) if (r.bnd[(size_t) i]->pos <= l) b = i;
        if (b < 0) continue;
        const int64_t pos = r.bnd[(size_t) b]->pos;
        if (pos > best.pos || (pos == best.pos && r.stamp > best_stamp)) {
            best = {};
            best.rec = id; best.boundary = b; best.pos = pos; best.lcp = l;
            best.has_logits = !r.bnd[(size_t) b]->logits.empty();
            if (b + 1 < (int) r.bnd.size()) { best.next_boundary = b + 1; best.next_pos = r.bnd[(size_t) b + 1]->pos; }
            best_stamp = r.stamp;
        }
    }
    return best;
}

bool Ds41ConvPool::restore(Ds41PoolId rec, int boundary, int64_t from, const Ds41PoolTargets & dev,
                           Ds41PoolRestore & out, std::string & err) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    out = {};
    auto fail = [&](const std::string & e) { err = e; ++im.S.restores_failed; return false; };
    if (!im.ok) return fail("pool: not initialized");
    auto it = im.recs.find(rec);
    if (it == im.recs.end()) return fail("pool: no record " + std::to_string(rec));
    Impl::Rec & r = it->second;
    if (boundary < 0 || boundary >= (int) r.bnd.size()) return fail("pool: no boundary " + std::to_string(boundary));
    const Impl::Bnd & b = *r.bnd[(size_t) boundary];
    if (from < 0 || from > b.pos) return fail("pool: restore from " + std::to_string(from) + " past the boundary");
    if (b.pos > im.L.max_positions) return fail("pool: boundary past max_positions");
    if (!im.targets_ok(dev, err)) { ++im.S.restores_failed; return false; }
    if ((int64_t) r.chunks.size() * im.C < b.pos) return fail("pool: record is missing chunks");
    if (im.cfg.verify) {   // the host bytes must be what the save left there
        std::string e2;
        if (!im.wait_seq(im.seq, &e2)) return fail(e2);
        for (int64_t k = from / im.C; k * im.C < b.pos; ++k)
            if (sum64(im.slab_ptr[r.chunks[(size_t) k]], im.cfg.slab_bytes) != im.slab_sum[r.chunks[(size_t) k]])
                return fail("pool: checksum mismatch in chunk " + std::to_string(k));
        for (uint32_t s : b.slabs)
            if (sum64(im.slab_ptr[s], im.cfg.slab_bytes) != im.slab_sum[s]) return fail("pool: checksum mismatch in regions");
    }
    // every check passed: from here on the device is written
    if (!im.begin_batch(err)) { ++im.S.restores_failed; return false; }
    size_t bytes = 0;
    bool good = true, queued = false;
    for (int64_t k = from / im.C; k * im.C < b.pos && good; ++k) {
        const size_t before = bytes;
        good = im.copy_chunk(r.chunks[(size_t) k], k, std::max<int64_t>(from, k * im.C),
                             std::min<int64_t>((k + 1) * im.C, b.pos), dev, false, bytes, err);
        queued |= bytes > before;
    }
    if (good) { const size_t before = bytes; good = im.copy_regions(b.slabs, dev, false, bytes, err); queued |= bytes > before; }
    im.end_batch(false, bytes);
    out.rec = rec; out.pos = b.pos; out.from = from; out.bytes = bytes; out.seq = im.seq;
    if (!good) {
        out.live_clobbered = queued || bytes > 0;
        ++im.S.restores_failed;
        return false;
    }
    out.logits = b.logits;
    out.blobs = b.blobs;
    r.stamp = ++im.clock;
    ++im.S.restores;
    im.S.bytes_h2d += bytes;
    return true;
}

bool Ds41ConvPool::wait(const Ds41PoolRestore & r, std::string * err) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    const auto t0 = std::chrono::steady_clock::now();
    const bool good = im.wait_seq(r.seq, err);
    im.S.fence_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return good;
}

bool Ds41ConvPool::fence(std::string * err) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    const auto t0 = std::chrono::steady_clock::now();
    const bool good = im.wait_seq(im.seq, err);
    im.S.fence_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return good;
}

bool Ds41ConvPool::busy() const {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    im.reap(false, nullptr);
#ifdef DS41_POOL_CUDA
    return !im.ops.empty();
#else
    return false;
#endif
}

Ds41PoolId Ds41ConvPool::pin_root(Ds41PoolId rec, int boundary, std::string & err) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto it = im.recs.find(rec);
    if (it == im.recs.end()) { err = "pool: no record"; return 0; }
    const Impl::Rec & src = it->second;
    if (boundary < 0 || boundary >= (int) src.bnd.size()) { err = "pool: no boundary"; return 0; }
    const int64_t pos = src.bnd[(size_t) boundary]->pos;
    // an existing pinned record of exactly this prefix
    for (auto & [id, r] : im.recs)
        if (r.pinned && r.n() == pos && !r.bnd.empty() && r.bnd.back()->pos == pos &&
            Impl::lcp_of(r, src.tok.data(), src.ids.empty() ? nullptr : src.ids.data(), pos) == pos) return id;
    const size_t nch = (size_t) ((pos + im.C - 1) / im.C);
    size_t mine = nch + im.region_slabs * (size_t) (boundary + 1);
    if (im.pinned_bytes() + mine * im.cfg.slab_bytes > (size_t) (im.cfg.pinned_max_frac * (double) im.cfg.budget_bytes)) {
        err = "pool: pinned share full";
        return 0;
    }
    const size_t hb = (size_t) pos * sizeof(int32_t) + (src.ids.empty() ? 0 : (size_t) pos * sizeof(uint64_t));
    if (!im.fits(0, hb) && !im.make_room(0, hb, rec, err)) return 0;
    it = im.recs.find(rec);   // make_room may have rearranged nothing of rec (kept), but be safe
    const Impl::Rec & s2 = it->second;
    Impl::Rec r;
    r.id = im.next_id++;
    r.tok.assign(s2.tok.begin(), s2.tok.begin() + pos);
    if (!s2.ids.empty()) r.ids.assign(s2.ids.begin(), s2.ids.begin() + pos);
    for (size_t k = 0; k < nch; ++k) { r.chunks.push_back(s2.chunks[k]); ++im.slab_ref[s2.chunks[k]]; }
    for (int i = 0; i <= boundary; ++i) r.bnd.push_back(s2.bnd[(size_t) i]);
    r.stamp = ++im.clock;
    r.pinned = true;
    im.host_bytes += r.meta();
    const Ds41PoolId id = r.id;
    im.recs.emplace(id, std::move(r));
    return id;
}

bool Ds41ConvPool::pin(Ds41PoolId rec, bool on) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto it = im.recs.find(rec);
    if (it == im.recs.end()) return false;
    if (on && !it->second.pinned) {
        const Impl::Rec & r = it->second;
        if (im.pinned_bytes() + (r.chunks.size() + r.bnd.size() * im.region_slabs) * im.cfg.slab_bytes + r.meta() >
            (size_t) (im.cfg.pinned_max_frac * (double) im.cfg.budget_bytes)) return false;
    }
    it->second.pinned = on;
    return true;
}

bool Ds41ConvPool::evict(Ds41PoolId rec) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto it = im.recs.find(rec);
    if (it == im.recs.end()) return false;
    im.drop(it);
    ++im.S.evictions;
    return true;
}

size_t Ds41ConvPool::clear(bool keep_pinned) {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    size_t n = 0;
    for (auto it = im.recs.begin(); it != im.recs.end();) {
        auto nx = std::next(it);
        if (!keep_pinned || !it->second.pinned) { im.drop(it); ++n; }
        it = nx;
    }
    return n;
}

Ds41PoolStats Ds41ConvPool::stats() const {
    Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    im.reap(false, nullptr);
    Ds41PoolStats s = im.S;
    s.records = im.recs.size();
    for (const auto & [id, r] : im.recs) { s.pinned += r.pinned; s.boundaries += r.bnd.size(); }
    s.slabs_used = im.slabs_used;
    s.slabs_total = im.slab_ptr.size();
    s.pinned_allocated = im.slab_ptr.size() * im.cfg.slab_bytes;
    s.host_bytes = im.host_bytes;
    s.used_bytes = im.used_bytes();
    return s;
}

std::string Ds41ConvPool::stats_line() const {
    const Ds41PoolStats s = stats();
    char b[512];
    std::snprintf(b, sizeof b,
                  "pool: %zu records (%zu pinned, %zu boundaries), %.1f / %.0f MiB used (slabs %zu/%zu, host %.1f MiB), "
                  "saves %llu (+%llu touched, %llu failed), restores %llu (%llu failed), evicted %llu, superseded %llu, "
                  "shared %.1f MiB, last save %.2f ms, last restore %.2f ms, fence wait %.1f ms",
                  s.records, s.pinned, s.boundaries, s.used_bytes / 1048576.0, s.budget_bytes / 1048576.0, s.slabs_used,
                  s.slabs_total, s.host_bytes / 1048576.0, (unsigned long long) s.saves, (unsigned long long) s.saves_touched,
                  (unsigned long long) s.saves_failed, (unsigned long long) s.restores, (unsigned long long) s.restores_failed,
                  (unsigned long long) s.evictions, (unsigned long long) s.superseded, s.bytes_shared / 1048576.0,
                  s.last_save_ms, s.last_restore_ms, s.fence_wait_ms);
    return b;
}

std::vector<Ds41PoolRecordInfo> Ds41ConvPool::records() const {
    const Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    std::vector<Ds41PoolRecordInfo> v;
    for (const auto & [id, r] : im.recs) {
        Ds41PoolRecordInfo x;
        x.id = id; x.n = r.n(); x.pinned = r.pinned; x.stamp = r.stamp; x.chunks = r.chunks.size();
        for (uint32_t s : r.chunks) x.own_slabs += im.slab_ref[s] == 1;
        for (const auto & b : r.bnd) {
            x.boundaries.push_back({ b->pos, !b->logits.empty() });
            if (b.use_count() == 1) x.own_slabs += b->slabs.size();
        }
        v.push_back(std::move(x));
    }
    return v;
}

std::vector<int32_t> Ds41ConvPool::tokens(Ds41PoolId rec) const {
    const Impl & im = *p_;
    std::lock_guard<std::mutex> lk(im.mu);
    auto it = im.recs.find(rec);
    return it == im.recs.end() ? std::vector<int32_t>{} : it->second.tok;
}

}  // namespace strata::v41
