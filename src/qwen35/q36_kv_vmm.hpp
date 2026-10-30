#pragma once
// CPU-testable private VMM planner. Call only between synchronized forwards.
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>
#include <stdexcept>

namespace strata { namespace q36 { namespace kv {
inline bool mul(size_t a, size_t b, size_t& out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b; return true;
}
inline bool align_up(size_t n, size_t alignment, size_t& out) {
    if (!alignment || n > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
    out = (n + alignment - 1) / alignment * alignment; return true;
}
class Budget {
    size_t cap_, used_ = 0;
    mutable std::mutex mutex_;
public:
    explicit Budget(size_t cap): cap_(cap) {}
    bool acquire(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (n > cap_ - used_) return false;
        used_ += n; return true;
    }
    void release(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (n > used_) std::terminate();
        used_ -= n;
    }
    size_t used() const { std::lock_guard<std::mutex> lock(mutex_); return used_; }
    size_t cap() const { return cap_; }
};
struct Driver {
    virtual ~Driver() = default;
    virtual size_t granularity() const = 0;
    virtual bool reserve(uintptr_t& address, size_t bytes, size_t alignment) = 0;
    // map must clean up its own partial create/map/access failure.
    virtual bool map(uintptr_t address, size_t bytes) = 0;
    virtual bool unmap(uintptr_t address, size_t bytes) = 0;
    virtual bool free_address(uintptr_t address, size_t bytes) = 0;
};
// One head row per host page. Unlike a pitched 2D copy, a row never spans
// the unmapped virtual gap between two heads. Offsets are relative to this
// layer's K/V device base and this host page's K/V base respectively.
template<class F>
bool for_each_copy_row(size_t ctx, size_t cell, size_t heads, size_t host_page_tokens,
                       size_t mapped_tokens, size_t a, size_t b, F row) {
    size_t stride = 0, host_stride = 0, extent = 0;
    if (!ctx || !cell || !heads || !host_page_tokens || a > b || b > ctx || b > mapped_tokens ||
        !mul(ctx, cell, stride) || !mul(host_page_tokens, cell, host_stride) ||
        !mul(stride, heads, extent) || !mul(host_stride, heads, extent)) return false;
    for (size_t c = a; c < b;) {
        const size_t page = c / host_page_tokens, offset = c % host_page_tokens;
        const size_t count = std::min(b - c, host_page_tokens - offset);
        for (size_t head = 0; head < heads; ++head)
            row(page, head * stride + c * cell, head * host_stride + offset * cell, count * cell);
        c += count;
    }
    return true;
}
class Arena {
    Budget& budget_;
    Driver& driver_;
    size_t heads_, ctx_, cell_, page_, stride_ = 0, reserved_ = 0, pages_ = 0;
    uintptr_t address_ = 0;
    uint64_t mapping_generation_ = 1;
    bool healthy_ = true;
    bool page_count(size_t end, size_t& count) const {
        size_t bytes = 0, rounded = 0;
        if (end > ctx_ || !mul(end, cell_, bytes) || !align_up(bytes, page_, rounded)) return false;
        count = rounded / page_; return true;
    }
public:
    Arena(Budget& budget, Driver& driver, size_t heads, size_t ctx, size_t cell)
        : budget_(budget), driver_(driver), heads_(heads), ctx_(ctx), cell_(cell), page_(driver.granularity()) {}
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    bool init() {
        if (address_) return true;
        if (!heads_ || !ctx_ || !cell_ || !page_ || !mul(ctx_, cell_, stride_) || stride_ % page_ ||
            !mul(heads_, stride_, reserved_)) return false;
        return driver_.reserve(address_, reserved_, page_) && address_ != 0;
    }
    void* get() const { return reinterpret_cast<void*>(address_); }
    size_t stride() const { return stride_; }
    uint64_t mapping_generation() const { return mapping_generation_; }
    size_t committed_tokens() const { return pages_ * page_ / cell_; }
    size_t committed_bytes() const { return pages_ * page_ * heads_; }
    bool healthy() const { return healthy_; }
    // SIZE_MAX means invalid request/overflow. Count includes all heads, with page rounding.
    size_t bytes_for(size_t end) const {
        size_t p = 0, bytes = 0;
        if (!page_count(end, p) || !mul(p, page_, bytes) || !mul(bytes, heads_, bytes))
            return std::numeric_limits<size_t>::max();
        return bytes;
    }
    bool ensure(size_t end) {
        size_t target = 0, delta = 0;
        if (!healthy_ || !address_ || !page_count(end, target)) return false;
        if (target <= pages_) return true;
        if (!mul(target - pages_, page_, delta) || !mul(delta, heads_, delta) || !budget_.acquire(delta)) return false;
        size_t mapped = 0;
        const size_t add = target - pages_;
        for (size_t head = 0; head < heads_; ++head)
            for (size_t p = pages_; p < target; ++p) {
                if (!driver_.map(address_ + head * stride_ + p * page_, page_)) {
                    // Reverse the exact successful prefix, retaining accounting on rollback failure.
                    size_t unreclaimed = 0;
                    while (mapped) {
                        --mapped;
                        if (!driver_.unmap(address_ + (mapped / add) * stride_ + (pages_ + mapped % add) * page_, page_))
                            ++unreclaimed;
                    }
                    budget_.release(delta - unreclaimed * page_);
                    if (unreclaimed) { healthy_ = false; throw std::runtime_error("Q36 KV VMM rollback unmap failed"); }
                    return false;
                }
                ++mapped;
            }
        pages_ = target; return true;
    }
    // Caller must synchronize the stream and release snapshot owners above end first.
    bool trim(size_t end) {
        size_t target = 0;
        if (!healthy_ || !page_count(end, target)) return false;
        if (target >= pages_) return true;
        // Invalidate residency BEFORE even a partially failing unmap. Growth
        // preserves the prefix, while remapping must never resurrect old hints.
        if(mapping_generation_==std::numeric_limits<uint64_t>::max()) {healthy_=false;return false;}
        ++mapping_generation_;
        for (size_t p = pages_; p > target; --p) {
            for (size_t head = 0; head < heads_; ++head) {
                if (!driver_.unmap(address_ + head * stride_ + (p - 1) * page_, page_)) {
                    healthy_ = false; return false;
                }
                budget_.release(page_);
            }
            --pages_;
        }
        return true;
    }
    ~Arena() {
        if (address_ && healthy_) {
            if (!trim(0)) return;
            driver_.free_address(address_, reserved_);
        }
    }
};
}}}
