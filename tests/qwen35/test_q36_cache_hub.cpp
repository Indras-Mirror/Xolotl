#include "../../src/qwen35/q36_cache_hub.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>
using namespace strata::q36::cache;
int main() {
    Hub hub(128);
    std::array<std::array<uint8_t, 64>, 128> snapshot_memory{};
    std::array<std::array<uint8_t, 256>, 20> page_memory{};
    hub.snapshot_bytes = 64; hub.page_bytes = 256;
    for (size_t i = 0; i < hub.snaps.size(); ++i) hub.snaps[i].host = snapshot_memory[i].data();
    for (auto& page : page_memory) hub.free_pages.push_back(page.data());
    hub.n_pages = page_memory.size();
    auto* stable = &hub.snaps[0];
    Conversation invalid_bounds; invalid_bounds.mirrored = 1;
    bool refused = false;
    try { hub.publish(std::move(invalid_bounds)); } catch (const std::logic_error&) { refused = true; }
    assert(refused && hub.parked.empty());
    // Reserve -> writing pin -> copy -> completion fence -> published metadata.
    auto write = hub.reserve_snapshot(*stable);
    assert(write && stable->writing && stable->pins == 1 && !hub.ready(*stable));
    Conversation unfinished; unfinished.tok.resize(1023); unfinished.val.resize(1023);
    unfinished.mirrored = 1023; unfinished.states.push_back(stable);
    refused = false;
    try { hub.publish(std::move(unfinished)); } catch (const std::logic_error&) { refused = true; }
    assert(refused && unfinished.states.size() == 1 && hub.parked.empty());
    assert(!hub.reserve_snapshot(*stable));
    bool copy_complete = false;
    assert(!write.commit(1023, 42, 3, [&] { return copy_complete; }));
    assert(!hub.ready(*stable));
    std::memset(write.data(), 7, hub.snapshot_bytes); copy_complete = true;
    assert(write.commit(1023, 42, 3, [&] { return copy_complete; }));
    assert(hub.ready(*stable) && stable->pins == 1);
    std::vector<Snapshot*> keep{stable}; hub.retain_states(keep); write.reset();
    assert(stable->pins == 0 && stable->refs == 1 && !hub.reserve_snapshot(*stable));
    hub.release_states(keep);
    {
        Hub::SnapPin pin(hub, *stable);
        assert(!hub.reserve_snapshot(*stable));
        Hub::SnapPin moved(std::move(pin)); assert(moved && !pin);
    }
    assert(stable->pins == 0);
    // Six callers live in one shared cache; any of four execution owners can
    // fork any published prefix without deleting it for another owner.
    std::vector<uint64_t> ids;
    for (size_t caller = 0; caller < 6; ++caller) {
        Conversation conv;
        conv.tok.assign(16, int32_t(caller)); conv.val.assign(16, caller); conv.mirrored = 16;
        conv.pages.push_back(hub.take_page());
        std::memset(conv.pages[0], int(caller), hub.page_bytes);
        for (size_t anchor = 0; anchor < 12; ++anchor) {
            auto& snapshot = hub.snaps[caller * 12 + anchor];
            auto pending = hub.reserve_snapshot(snapshot); assert(pending);
            assert(pending.commit(int64_t(anchor + 1), caller * 100 + anchor, 0, [] { return true; }));
            conv.states.push_back(&snapshot); ++snapshot.refs;
        }
        ids.push_back(hub.publish(std::move(conv)));
    }
    for (size_t round = 0; round < 6; ++round) {
        std::array<Hub::Checkout, 4> execution_owners;
        for (size_t owner = 0; owner < execution_owners.size(); ++owner) {
            const size_t caller = (round + owner) % 6;
            execution_owners[owner] = hub.checkout(caller);
            assert(execution_owners[owner].conv.val.front() == caller);
            assert(execution_owners[owner].conv.pages.front()[0] == caller);
            assert(execution_owners[owner].conv.states.size() == 12);
        }
        assert(hub.parked.size() == 6);
    }
    std::array<std::thread, 4> concurrent_owners;
    for (size_t owner = 0; owner < concurrent_owners.size(); ++owner)
        concurrent_owners[owner] = std::thread([&, owner] {
            for (size_t round = 0; round < 24; ++round) {
                const size_t caller = (owner + round) % 6;
                auto checkout = hub.checkout(caller);
                assert(checkout && checkout.conv.pages[0][0] == caller);
            }
        });
    for (auto& thread : concurrent_owners) thread.join();
    for (const auto& record : hub.parked) {
        assert(record.leases == 0 && hub.page_refs.at(record.pages[0]) == 1);
        for (const auto* state : record.states) assert(state->refs == 1);
    }
    { auto older = hub.checkout(4); auto newer = hub.checkout(5); }
    std::array<Hub::Checkout, 4> active;
    for (size_t owner = 0; owner < active.size(); ++owner) active[owner] = hub.checkout(owner);
    assert(hub.parked.size() == 6);
    auto fork = hub.checkout(0);
    assert(fork.conv.pages == active[0].conv.pages);
    assert(hub.parked[0].leases == 2);
    assert(hub.page_refs.at(fork.conv.pages[0]) == 3);
    assert(hub.snaps[0].refs == 3);
    const uint64_t protected_id = active[0].lease.id();
    assert(hub.evict_lru());
    for (const auto& record : hub.parked) assert(record.id != ids[4]);
    assert(hub.evict_lru()); // only the remaining unleased caller can be evicted
    assert(!hub.evict_lru());
    // Published/active aliases remain immutable while COW writes a new page.
    auto* source = fork.conv.pages[0];
    const auto initial_refs = hub.page_refs.at(source);
    auto* own = hub.cow_page(source, [&](uint8_t* dst, const uint8_t* src, size_t bytes) {
        assert(bytes == hub.page_bytes && hub.page_refs.at(source) == initial_refs + 1);
        std::atomic<bool> unlocked{false};
        std::thread probe([&] { if (hub.mutex.try_lock()) { unlocked = true; hub.mutex.unlock(); } });
        probe.join(); assert(unlocked);
        std::memcpy(dst, src, bytes); return true;
    });
    assert(own && own != source && hub.page_refs.at(source) == initial_refs);
    own[0] = 99; assert(source[0] == 0);
    std::vector<uint8_t*> own_ref{own}; hub.release_pages(own_ref);
    const size_t free_before = hub.free_pages.size();
    assert(!hub.cow_page(source, [](uint8_t*, const uint8_t*, size_t) { return false; }));
    assert(hub.free_pages.size() == free_before && hub.page_refs.at(source) == initial_refs);
    // Checkout transfer leaves exactly one active ref owner; record lease is
    // stable despite vector erases/reallocation and IDs are never reused.
    Conversation transferred = fork.release_conv();
    fork.reset();
    assert(hub.page_refs.at(source) == initial_refs);
    hub.release_conv(transferred);
    assert(hub.page_refs.at(source) == initial_refs - 1);
    for (auto& owner : active) owner.reset();
    while (hub.evict_lru()) {}
    assert(hub.parked.empty() && hub.page_refs.empty());
    assert(hub.free_pages.size() == hub.n_pages);
    Conversation newer; const uint64_t new_id = hub.publish(std::move(newer));
    assert(new_id > protected_id);
    assert(hub.evict_lru());
    // Capacity is fixed at 128. All active pins make optional snapshot writes
    // fail immediately; releasing one pin permits exactly that slot's reuse.
    std::vector<Hub::SnapPin> pins;
    for (auto& snapshot : hub.snaps) {
        auto pending = hub.reserve_snapshot(snapshot); assert(pending);
        assert(pending.commit(1, 17, 0, [] { return true; })); pending.reset();
        pins.emplace_back(hub, snapshot);
    }
    for (auto& snapshot : hub.snaps) assert(!hub.reserve_snapshot(snapshot));
    pins[17].reset();
    { auto pending = hub.reserve_snapshot(hub.snaps[17]); assert(pending); }
    assert(hub.snaps[17].pos == -1 && !hub.snaps[17].writing && hub.snaps[17].pins == 0);
    pins.clear();
    assert(&hub.snaps[0] == stable && hub.snaps.size() == 128);
    std::cout << "q36 shared cache hub: PASS\n";
}
