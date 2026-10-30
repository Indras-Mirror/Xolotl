#pragma once
#include "q36_page_provenance.hpp"
// Shared host-cache metadata only. No CUDA calls. The owner allocates buffers
// once, joins every worker and destroys every lease before destroying the Hub.
// Native code accessing the public compatibility fields must hold mutex.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata { namespace q36 { namespace cache {
struct Snapshot {
    int64_t pos = -1;
    uint64_t hash = 0, used = 0;
    int prio = 0;
    uint8_t* host = nullptr;
    size_t refs = 0, pins = 0;
    bool writing = false;
};
struct Conversation {
    std::vector<int32_t> tok;
    std::vector<uint64_t> val;
    std::vector<uint8_t*> pages;
    std::vector<Snapshot*> states;
    int64_t mirrored = 0;
    uint64_t used = 0, id = 0;
    size_t leases = 0;
    int origin_slot = -1;
};
class Hub {
public:
    std::recursive_mutex mutex;
    // FIXED after construction: all Snapshot* addresses remain stable.
    std::vector<Snapshot> snaps;
    std::vector<uint8_t*> free_pages;
    std::map<uint8_t*, int> page_refs;
    std::vector<Conversation> parked;
    size_t n_pages = 0, snapshot_bytes = 0, page_bytes = 0;
    bool snaps_initialized = false, pages_initialized = false;
    uint64_t snap_clock = 0, conv_clock = 0, next_id = 0;
    bool track_page_versions=false;
    struct PageVersion { uint64_t cookie=0,version=0;bool ready=false; };
    std::map<uint8_t*,PageVersion> page_versions;
    uint64_t page_clock=0;
    uint64_t next_page_version() {
        if(page_clock==UINT64_MAX)throw std::overflow_error("cache page version exhausted");
        return ++page_clock;
    }
    strata::q36::resident::PageStamp begin_page_write(uint8_t* page) {
        if(!track_page_versions)return {};
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if(page_refs.count(page)==0 || page_refs.at(page)!=1 || !page_versions.count(page))
            throw std::logic_error("cache write needs an exclusively owned page");
        auto& v=page_versions.at(page);v.ready=false;v.version=next_page_version();
        return {uint64_t(reinterpret_cast<uintptr_t>(page)),v.cookie,v.version};
    }
    bool finish_page_write(strata::q36::resident::PageStamp ticket) {
        if(!track_page_versions)return true;
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto* page=reinterpret_cast<uint8_t*>(uintptr_t(ticket.page));
        auto found=page_versions.find(page);
        if(!ticket.valid() || !page_refs.count(page) || page_refs.at(page)!=1 || found==page_versions.end() ||
           found->second.ready || found->second.cookie!=ticket.cookie || found->second.version!=ticket.version)return false;
        found->second.ready=true;return true;
    }
    strata::q36::resident::PageStamp page_stamp(uint8_t* page) {
        if(!track_page_versions)return {};
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto found=page_versions.find(page);
        if(!page_refs.count(page) || found==page_versions.end() || !found->second.ready)return {};
        return {uint64_t(reinterpret_cast<uintptr_t>(page)),found->second.cookie,found->second.version};
    }
    std::vector<strata::q36::resident::PageStamp> page_stamps(const std::vector<uint8_t*>& pages) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        std::vector<strata::q36::resident::PageStamp> stamps;stamps.reserve(pages.size());
        for(auto page:pages)stamps.push_back(page_stamp(page));
        return stamps;
    }
    explicit Hub(size_t snapshots = 128): snaps(snapshots) {}
    Hub(const Hub&) = delete;
    Hub& operator=(const Hub&) = delete;
    bool owns(const Snapshot* snapshot) const {
        // Avoid ordering comparisons between unrelated pointers.
        for (const auto& item : snaps) if (&item == snapshot) return true;
        return false;
    }
    bool ready(const Snapshot& snapshot) const { return !snapshot.writing && snapshot.pos >= 0; }
    bool reusable(const Snapshot& snapshot) const { return snapshot.refs == 0 && snapshot.pins == 0 && !snapshot.writing; }
    void retain_pages(const std::vector<uint8_t*>& pages) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (auto* page : pages) {
            auto found = page_refs.find(page);
            if (found == page_refs.end() || found->second <= 0) throw std::logic_error("retain unknown cache page");
        }
        for (auto* page : pages) {
            auto found = page_refs.find(page);
            ++found->second;
        }
    }
    void release_pages(std::vector<uint8_t*>& pages) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (auto* page : pages) {
            auto found = page_refs.find(page);
            if (found == page_refs.end() || found->second <= 0) throw std::logic_error("release unknown cache page");
            if (--found->second == 0) { page_refs.erase(found); free_pages.push_back(page); }
        }
        pages.clear();
    }
    void retain_states(const std::vector<Snapshot*>& states) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (auto* state : states) {
            if (!owns(state) || !ready(*state)) throw std::logic_error("retain unavailable cache snapshot");
        }
        for (auto* state : states) ++state->refs;
    }
    void release_states(std::vector<Snapshot*>& states) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (auto* state : states) {
            if (!owns(state) || state->refs == 0) throw std::logic_error("release unknown cache snapshot");
            --state->refs;
        }
        states.clear();
    }
    void release_conv(Conversation& conv) { release_pages(conv.pages); release_states(conv.states); }
    uint8_t* take_page() {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (free_pages.empty()) return nullptr;
        const uint64_t cookie=track_page_versions?next_page_version():0;
        auto* page = free_pages.back(); free_pages.pop_back();
        if (!page || page_refs.count(page)) throw std::logic_error("cache free page is still owned");
        page_refs.emplace(page, 1);
        if(track_page_versions)page_versions[page]={cookie,0,false};
        return page;
    }
    // Copy takes place outside metadata locks. The extra source reference keeps
    // the original immutable and alive until the callback has COMPLETED its copy.
    template<class Copy> uint8_t* cow_page(uint8_t* source, Copy copy) {
        uint8_t* destination = nullptr;
        strata::q36::resident::PageStamp write_ticket;
        {
            std::lock_guard<std::recursive_mutex> lock(mutex);
            auto found = page_refs.find(source);
            if (found == page_refs.end() || found->second < 1) throw std::logic_error("COW unknown page");
            destination = take_page();
            if (!destination) return nullptr;
            ++found->second;
            write_ticket=begin_page_write(destination);
        }
        bool complete = false;
        try { complete = copy(destination, source, page_bytes); }
        catch (...) {
            std::vector<uint8_t*> source_pin{source}, failed{destination};
            release_pages(source_pin); release_pages(failed); throw;
        }
        std::vector<uint8_t*> source_pin{source}; release_pages(source_pin);
        if (!complete) { std::vector<uint8_t*> failed{destination}; release_pages(failed); return nullptr; }
        if(!finish_page_write(write_ticket))throw std::logic_error("cache COW completion is stale");
        return destination; // one owned reference, not yet published
    }
    // All page writes must already be complete. This transfers, rather than
    // duplicates, the conversation's existing page/state ownership references.
    uint64_t publish(Conversation&& conv) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (conv.mirrored < 0 || uint64_t(conv.mirrored) > conv.val.size() ||
            uint64_t(conv.mirrored) > conv.tok.size()) throw std::logic_error("cache prefix bounds invalid");
        for (auto* page : conv.pages) {
            auto found = page_refs.find(page);
            if (found == page_refs.end() || found->second <= 0) throw std::logic_error("publish unknown cache page");
            if(track_page_versions && !page_stamp(page).valid())throw std::logic_error("publish unfinished cache page");
        }
        for (auto* state : conv.states)
            if (!owns(state) || !ready(*state) || !state->refs || state->pos > conv.mirrored)
                throw std::logic_error("publish unavailable cache snapshot");
        if (next_id == UINT64_MAX) throw std::overflow_error("cache record ID exhausted");
        conv.id = ++next_id; conv.used = ++conv_clock; conv.leases = 0;
        parked.push_back(std::move(conv)); return next_id;
    }
    // Eviction only removes a whole published prefix whose active lease count
    // is zero. Buffer refs shared with other records are independently retained.
    bool evict_lru() {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        auto victim = parked.end();
        for (auto it = parked.begin(); it != parked.end(); ++it)
            if (it->leases == 0 && (victim == parked.end() || it->used < victim->used)) victim = it;
        if (victim == parked.end()) return false;
        release_conv(*victim); parked.erase(victim); return true;
    }
private:
    Conversation* find_id(uint64_t id) {
        for (auto& conv : parked) if (conv.id == id) return &conv;
        return nullptr;
    }
public:
    class RecordLease {
        Hub* hub_ = nullptr;
        uint64_t id_ = 0;
        friend class Hub;
        RecordLease(Hub& hub, uint64_t id): hub_(&hub), id_(id) {
            std::lock_guard<std::recursive_mutex> lock(hub_->mutex);
            auto* conv = hub_->find_id(id_);
            if (!conv) throw std::logic_error("lease missing cache record");
            ++conv->leases;
        }
    public:
        RecordLease() = default;
        RecordLease(const RecordLease&) = delete;
        RecordLease& operator=(const RecordLease&) = delete;
        RecordLease(RecordLease&& other) noexcept: hub_(other.hub_), id_(other.id_) { other.hub_ = nullptr; }
        RecordLease& operator=(RecordLease&& other) noexcept {
            if (this != &other) { reset(); hub_ = other.hub_; id_ = other.id_; other.hub_ = nullptr; }
            return *this;
        }
        void reset() noexcept {
            if (!hub_) return;
            std::lock_guard<std::recursive_mutex> lock(hub_->mutex);
            auto* conv = hub_->find_id(id_);
            // Native code must also skip leased records in custom eviction.
            if (!conv || !conv->leases) std::terminate();
            --conv->leases; hub_ = nullptr;
        }
        explicit operator bool() const { return hub_ != nullptr; }
        uint64_t id() const { return id_; }
        ~RecordLease() { reset(); }
    };
    class Checkout {
        Hub* hub_ = nullptr;
        friend class Hub;
        Checkout(Hub& hub, Conversation value, RecordLease lease_): hub_(&hub), conv(std::move(value)), lease(std::move(lease_)) {}
    public:
        Conversation conv;
        RecordLease lease;
        Checkout() = default;
        Checkout(const Checkout&) = delete;
        Checkout& operator=(const Checkout&) = delete;
        Checkout(Checkout&& other) noexcept: hub_(other.hub_), conv(std::move(other.conv)), lease(std::move(other.lease)) { other.hub_ = nullptr; }
        Checkout& operator=(Checkout&& other) noexcept {
            if (this != &other) {
                reset(); hub_ = other.hub_; conv = std::move(other.conv); lease = std::move(other.lease); other.hub_ = nullptr;
            }
            return *this;
        }
        Conversation release_conv() { Conversation result = std::move(conv); conv = Conversation{}; return result; }
        void reset() noexcept {
            if (hub_) { hub_->release_conv(conv); hub_ = nullptr; }
            lease.reset();
        }
        explicit operator bool() const { return hub_ != nullptr; }
        ~Checkout() { reset(); }
    };
    Checkout checkout(size_t index) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (index >= parked.size()) return {};
        auto& original = parked[index];
        RecordLease lease(*this, original.id);
        Conversation copy = original; copy.leases = 0;
        retain_pages(copy.pages); retain_states(copy.states);
        original.used = ++conv_clock;
        return Checkout(*this, std::move(copy), std::move(lease));
    }
    class SnapPin {
        Hub* hub_ = nullptr;
        Snapshot* snapshot_ = nullptr;
    public:
        SnapPin() = default;
        SnapPin(Hub& hub, Snapshot& snapshot): hub_(&hub), snapshot_(&snapshot) {
            std::lock_guard<std::recursive_mutex> lock(hub.mutex);
            if (!hub.owns(&snapshot) || !hub.ready(snapshot)) throw std::logic_error("pin unavailable cache snapshot");
            ++snapshot.pins;
        }
        SnapPin(const SnapPin&) = delete;
        SnapPin& operator=(const SnapPin&) = delete;
        SnapPin(SnapPin&& other) noexcept: hub_(other.hub_), snapshot_(other.snapshot_) { other.hub_ = nullptr; }
        SnapPin& operator=(SnapPin&& other) noexcept {
            if (this != &other) { reset(); hub_ = other.hub_; snapshot_ = other.snapshot_; other.hub_ = nullptr; } return *this;
        }
        void reset() noexcept {
            if (!hub_) return;
            std::lock_guard<std::recursive_mutex> lock(hub_->mutex);
            if (!snapshot_->pins) std::terminate();
            --snapshot_->pins; hub_ = nullptr;
        }
        explicit operator bool() const { return hub_ != nullptr; }
        ~SnapPin() { reset(); }
    };
    class SnapshotWrite {
        Hub* hub_ = nullptr;
        Snapshot* snapshot_ = nullptr;
        bool committed_ = false;
        friend class Hub;
        SnapshotWrite(Hub& hub, Snapshot& snapshot): hub_(&hub), snapshot_(&snapshot) {
            snapshot.writing = true; ++snapshot.pins; snapshot.pos = -1;
        }
    public:
        SnapshotWrite() = default;
        SnapshotWrite(const SnapshotWrite&) = delete;
        SnapshotWrite& operator=(const SnapshotWrite&) = delete;
        SnapshotWrite(SnapshotWrite&& other) noexcept: hub_(other.hub_), snapshot_(other.snapshot_), committed_(other.committed_) { other.hub_ = nullptr; }
        SnapshotWrite& operator=(SnapshotWrite&& other) noexcept {
            if (this != &other) { reset(); hub_ = other.hub_; snapshot_ = other.snapshot_; committed_ = other.committed_; other.hub_ = nullptr; } return *this;
        }
        uint8_t* data() const { return snapshot_ ? snapshot_->host : nullptr; }
        Snapshot* get() const { return snapshot_; }
        explicit operator bool() const { return hub_ != nullptr; }
        // completed() must fence all asynchronous writes and run WITHOUT mutex.
        // A false result leaves this reservation writing and unavailable.
        template<class Completed> bool commit(int64_t pos, uint64_t hash, int prio, Completed completed) {
            if (!hub_ || committed_ || pos < 0 || !completed()) return false;
            std::lock_guard<std::recursive_mutex> lock(hub_->mutex);
            snapshot_->pos = pos; snapshot_->hash = hash; snapshot_->prio = prio;
            snapshot_->used = ++hub_->snap_clock; snapshot_->writing = false;
            // Keep the reservation's pin through the caller's retain callback.
            // reset/destruction releases it after ownership has been transferred.
            committed_ = true; return true;
        }
        // Caller must finish/cancel pending writes BEFORE abandoning a reservation.
        void reset() noexcept {
            if (!hub_) return;
            std::lock_guard<std::recursive_mutex> lock(hub_->mutex);
            if (!committed_) { snapshot_->pos = -1; snapshot_->writing = false; }
            if (!snapshot_->pins) std::terminate();
            --snapshot_->pins; hub_ = nullptr;
        }
        ~SnapshotWrite() { reset(); }
    };
    SnapshotWrite reserve_snapshot(Snapshot& snapshot) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (!owns(&snapshot) || !reusable(snapshot)) return {};
        return SnapshotWrite(*this, snapshot);
    }
};
}}}
