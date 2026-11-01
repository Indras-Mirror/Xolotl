#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// Contiguous checkpoint bytes. Ordinary copies are DEEP, as with std::vector.
// Only share() opts into immutable payload sharing. Every mutable access first
// detaches, while metadata (token IDs, LRU stamps, pins) remains caller-owned.
// Like vector, mutating the SAME object concurrently is not supported. Distinct
// shared aliases may be read or detached independently: once published, a block
// stays immutable even when all other aliases disappear. Refcounts are NOT a
// write-permission test. A mutable pointer or
// iterator must no longer be in use when share() publishes an alias: in the
// checkpoint path all device copies have completed before that point.
class SharedBytes {
public:
    using value_type = uint8_t;
    using size_type = size_t;
    using iterator = uint8_t*;
    using const_iterator = const uint8_t*;
    // Conservatively charged per alias, even though the shared control/vector
    // allocation physically exists once. Payload capacity is charged separately.
    static constexpr size_t allocation_overhead = 96;

    SharedBytes() = default;
    SharedBytes(std::initializer_list<uint8_t> v) { assign(v.begin(), v.end()); }
    explicit SharedBytes(size_t n, uint8_t value = 0) { resize(n, value); }
    SharedBytes(const std::vector<uint8_t>& v) { assign(v.begin(), v.end()); }
    SharedBytes(std::vector<uint8_t>&& v) {
        if (!v.empty() || v.capacity()) storage_ = std::make_shared<Block>(std::move(v));
    }
    SharedBytes(const SharedBytes& other) {
        if (other.storage_) storage_ = std::make_shared<Block>(*other.storage_);
    }
    SharedBytes& operator=(const SharedBytes& other) {
        if (this != &other) { SharedBytes copy(other); swap(copy); }
        return *this;
    }
    SharedBytes(SharedBytes&&) noexcept = default;
    SharedBytes& operator=(SharedBytes&& other) noexcept {
        if (this != &other) storage_ = std::move(other.storage_);
        return *this;
    }
    SharedBytes& operator=(const std::vector<uint8_t>& v) { assign(v.begin(), v.end()); return *this; }
    SharedBytes& operator=(std::vector<uint8_t>&& v) { SharedBytes copy(std::move(v)); swap(copy); return *this; }
    SharedBytes& operator=(std::initializer_list<uint8_t> v) { assign(v.begin(), v.end()); return *this; }

    SharedBytes share() const noexcept {
        // Publish immutability BEFORE publishing another owner. It is never
        // cleared on this block, including when the last alias is released.
        if (storage_) storage_->published.store(true, std::memory_order_release);
        return SharedBytes(storage_);
    }
    // Diagnostic only; no mutation decision may depend on a relaxed refcount.
    bool shared() const noexcept { return storage_ && storage_.use_count() > 1; }
    bool published() const noexcept { return storage_ && storage_->published.load(std::memory_order_acquire); }
    void swap(SharedBytes& other) noexcept { storage_.swap(other.storage_); }
    size_t size() const noexcept { return storage_ ? storage_->size() : 0; }
    size_t capacity() const noexcept { return storage_ ? storage_->capacity() : 0; }
    bool empty() const noexcept { return size() == 0; }
    size_t bytes() const noexcept {
        if (!storage_) return 0;
        return capacity() > SIZE_MAX - allocation_overhead ? SIZE_MAX : capacity() + allocation_overhead;
    }
    const uint8_t* data() const noexcept { return storage_ ? storage_->data() : nullptr; }
    uint8_t* data() { return storage_ ? writable().data() : nullptr; }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end() const noexcept { const auto p = data(); return p ? p + size() : p; }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    iterator begin() { return data(); }
    iterator end() { auto p = data(); return p ? p + size() : p; }
    const uint8_t& operator[](size_t i) const { return (*storage_)[i]; }
    uint8_t& operator[](size_t i) { return writable()[i]; }
    const uint8_t& front() const { return storage_->front(); }
    uint8_t& front() { return writable().front(); }
    const uint8_t& back() const { return storage_->back(); }
    uint8_t& back() { return writable().back(); }

    void reserve(size_t n) { if (n > capacity()) mutate([&](Storage& v) { v.reserve(n); }); }
    void resize(size_t n, uint8_t value = 0) { if (n != size()) mutate([&](Storage& v) { v.resize(n, value); }); }
    void clear() { if (!empty()) mutate([](Storage& v) { v.clear(); }); }
    void pop_back() { mutate([](Storage& v) { v.pop_back(); }); }
    void push_back(uint8_t value) { mutate([&](Storage& v) { v.push_back(value); }); }
    void assign(size_t n, uint8_t value) {
        auto copy = std::make_shared<Block>(n, value);
        storage_.swap(copy);
    }
    void assign(std::initializer_list<uint8_t> v) { assign(v.begin(), v.end()); }
    template<class InputIt, std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
    void assign(InputIt first, InputIt last) {
        auto copy = first == last ? std::make_shared<Block>() : std::make_shared<Block>(first, last);
        storage_.swap(copy);
    }
    template<class InputIt, std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        const size_t offset = position(pos);
        size_t result = offset;
        mutate([&](Storage& v) {
            const auto it = v.insert(v.begin() + (std::ptrdiff_t) offset, first, last);
            result = size_t(it - v.begin());
        });
        return storage_->data() ? storage_->data() + result : nullptr;
    }
    iterator insert(const_iterator pos, size_t n, uint8_t value) {
        const size_t offset = position(pos);
        size_t result = offset;
        mutate([&](Storage& v) {
            const auto it = v.insert(v.begin() + (std::ptrdiff_t) offset, n, value);
            result = size_t(it - v.begin());
        });
        return storage_->data() ? storage_->data() + result : nullptr;
    }
    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }
    iterator erase(const_iterator first, const_iterator last) {
        const size_t a = position(first), b = position(last);
        size_t result = a;
        mutate([&](Storage& v) {
            const auto it = v.erase(v.begin() + (std::ptrdiff_t) a, v.begin() + (std::ptrdiff_t) b);
            result = size_t(it - v.begin());
        });
        return storage_->data() ? storage_->data() + result : nullptr;
    }
    bool operator==(const SharedBytes& other) const {
        return size() == other.size() && (empty() || std::equal(begin(), end(), other.begin()));
    }
    bool operator==(const std::vector<uint8_t>& other) const {
        return size() == other.size() && (empty() || std::equal(begin(), end(), other.begin()));
    }
    friend bool operator==(const std::vector<uint8_t>& a, const SharedBytes& b) { return b == a; }

private:
    using Storage = std::vector<uint8_t>;
    struct Block final : Storage {
        using Storage::Storage;
        Block() = default;
        explicit Block(const Storage& v) : Storage(v) {}
        explicit Block(Storage&& v) : Storage(std::move(v)) {}
        // Ordinary copies are new, private blocks, never copies of the marker.
        Block(const Block& other) : Storage(static_cast<const Storage&>(other)) {}
        std::atomic<bool> published{false};
    };
    explicit SharedBytes(const std::shared_ptr<Block>& storage) noexcept : storage_(storage) {}
    size_t position(const_iterator pos) const { return empty() ? 0 : size_t(pos - data()); }
    std::shared_ptr<Block> clone() const {
        auto copy = std::make_shared<Block>();
        copy->reserve(storage_->capacity());
        constexpr size_t step = 1u << 20;
        for (size_t at = 0; at < storage_->size(); at += step) {
            copy->insert(copy->end(), storage_->begin() + (std::ptrdiff_t) at,
                         storage_->begin() + (std::ptrdiff_t) std::min(storage_->size(), at + step));
            progress_beat();
        }
        return copy;
    }
    template<class Fn> void mutate(Fn fn) {
        if (!storage_ || published()) {
            auto copy = storage_ ? clone() : std::make_shared<Block>();
            fn(*copy);
            storage_.swap(copy); // the complete mutation succeeds before publication
        } else fn(*storage_);
    }
    Storage& writable() {
        if (!storage_) storage_ = std::make_shared<Block>();
        else if (published()) {
            // Every published block remains immutable forever; another alias
            // must also clone, regardless of refcount or release ordering.
            auto copy = clone();
            storage_.swap(copy); // no publication or mutation if allocation/copy threw
        }
        return *storage_;
    }
    std::shared_ptr<Block> storage_;
};

} // namespace strata::core
