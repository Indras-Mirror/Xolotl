#include "../../src/qwen35/q36_snapshot_owner.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <thread>

using namespace strata::q36::cache;
namespace Owner = strata::q36::snapshot_owner;

struct Fixture {
    Hub hub{16};
    std::array<std::array<uint8_t, 16>, 16> snapshots{};
    std::array<std::array<uint8_t, 64>, 16> pages{};
    std::vector<uint64_t> hashes;

    Fixture() : hashes(33) {
        hub.track_page_versions = true;
        hub.snapshot_bytes = 16;
        hub.page_bytes = 64;
        hub.n_pages = pages.size();
        for (size_t i = 0; i < snapshots.size(); ++i) hub.snaps[i].host = snapshots[i].data();
        for (auto& page : pages) hub.free_pages.push_back(page.data());
        for (size_t i = 0; i < hashes.size(); ++i) hashes[i] = 1000 + i;
    }

    Snapshot* capture(std::vector<Snapshot*>& owned, int64_t pos, uint8_t marker) {
        auto lock = std::lock_guard<std::recursive_mutex>(hub.mutex);
        for (auto& state : hub.snaps) {
            auto pending = hub.reserve_snapshot(state);
            if (!pending) continue;
            std::memset(pending.data(), marker, hub.snapshot_bytes);
            pending.data()[1] = uint8_t(pos);
            assert(pending.commit(pos, hashes[(size_t) pos], 0, [] { return true; }));
            std::vector<Snapshot*> added{&state};
            hub.retain_states(added);
            owned.push_back(&state);
            return &state;
        }
        assert(false && "fixture has enough snapshots");
        return nullptr;
    }

    uint64_t record(uint8_t marker) {
        Conversation c;
        c.tok.assign(16, 7);
        c.val.assign(16, 7); // identical semantic cells for deliberately different numerical payloads
        c.mirrored = 16;
        auto* page = hub.take_page();
        assert(page);
        auto write = hub.begin_page_write(page);
        std::memset(page, marker, hub.page_bytes);
        assert(hub.finish_page_write(write));
        c.pages.push_back(page);
        capture(c.states, 8, marker);
        capture(c.states, 16, marker);
        return hub.publish(std::move(c));
    }

    void prune(std::vector<Snapshot*>& owned, int64_t boundary) {
        auto lock = std::lock_guard<std::recursive_mutex>(hub.mutex);
        auto removed = Owner::prune(owned, boundary, hashes);
        hub.release_states(removed);
    }

    void cow(Conversation& c, size_t first, size_t last, uint8_t marker) {
        auto* source = c.pages.at(0);
        auto* destination = hub.cow_page(source, [](uint8_t* dst, const uint8_t* src, size_t bytes) {
            std::memcpy(dst, src, bytes);
            return true;
        });
        assert(destination && destination != source);
        c.pages[0] = destination;
        std::vector<uint8_t*> previous{source};
        hub.release_pages(previous);
        auto ticket = hub.begin_page_write(destination);
        std::memset(destination + first, marker, last - first);
        assert(hub.finish_page_write(ticket));
    }

    void empty() {
        while (hub.evict_lru()) {}
        assert(hub.parked.empty() && hub.page_refs.empty() && hub.free_pages.size() == pages.size());
        for (auto& state : hub.snaps) assert(state.refs == 0 && state.pins == 0 && !state.writing);
    }
};

int main() {
    Fixture f;
    f.record(11);
    f.record(29);
    auto* a = Owner::best(f.hub.parked[0].states, 16, f.hashes);
    auto* b = Owner::best(f.hub.parked[1].states, 16, f.hashes);
    assert(a && b && a != b && a->pos == b->pos && a->hash == b->hash);
    assert(a->host[0] == 11 && b->host[0] == 29);
    assert(Owner::at(f.hub.parked[0].states, 16, a->hash) == a);
    assert(Owner::at({}, 16, a->hash) == nullptr); // foreign semantic twin is not a dedup certificate
    assert(Owner::best(f.hub.parked[1].states, 16, f.hashes) == b);
    assert(Owner::best(f.hub.parked[1].states, 7, f.hashes) == nullptr);
    auto short_prompt = f.hashes;
    short_prompt.resize(17);
    assert(Owner::best(f.hub.parked[1].states, 16, short_prompt)->pos == 8); // final row still needs logits

    // Four execution owners repeatedly checkout two identical semantic prefixes.
    // Their real Hub leases/refcounts keep both numerical payloads isolated.
    std::array<std::thread, 4> threads;
    for (size_t owner = 0; owner < threads.size(); ++owner) {
        threads[owner] = std::thread([&, owner] {
            for (size_t round = 0; round < 64; ++round) {
                const size_t index = (owner + round) % 2;
                auto checkout = f.hub.checkout(index);
                std::lock_guard<std::recursive_mutex> lock(f.hub.mutex);
                auto* selected = Owner::best(checkout.conv.states, 16, f.hashes);
                assert(selected == (index ? b : a));
                assert(selected->host[0] == checkout.conv.pages[0][0]);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    assert(a->refs == 1 && b->refs == 1);

    // Replaying unchanged semantic cells from8 invalidates the child's old16
    // anchor. The parent's anchor and KV page remain owned and unchanged.
    auto fork = f.hub.checkout(0);
    assert(a->refs == 2);
    f.prune(fork.conv.states, 8);
    assert(a->refs == 1 && Owner::at(fork.conv.states, 16, f.hashes[16]) == nullptr);
    assert(Owner::best(fork.conv.states, 16, f.hashes)->pos == 8);
    f.cow(fork.conv, 8, 16, 47);
    auto* replayed = f.capture(fork.conv.states, 16, 47);
    assert(replayed != a && replayed != b && replayed->host[0] == 47);
    assert(Owner::at(fork.conv.states, 16, f.hashes[16]) == replayed);
    assert(f.hub.parked[0].pages[0][8] == 11 && fork.conv.pages[0][8] == 47);
    const uint64_t child_id = f.hub.publish(fork.release_conv());
    fork.reset();
    assert(f.hub.parked.back().id == child_id);

    // A partial HOSTpage append gets a new COW pointer/version, but does not
    // change the numerical prefix<=16. Its owned checkpoint MUST stay reusable.
    auto append = f.hub.checkout(2);
    const auto old_stamp = f.hub.page_stamp(append.conv.pages[0]);
    std::array<uint8_t, 16> prefix{};
    std::memcpy(prefix.data(), append.conv.pages[0], prefix.size());
    f.cow(append.conv, 16, 20, 71);
    assert(!(f.hub.page_stamp(append.conv.pages[0]) == old_stamp));
    assert(std::memcmp(prefix.data(), append.conv.pages[0], prefix.size()) == 0);
    f.prune(append.conv.states, 16);
    assert(Owner::best(append.conv.states, 16, f.hashes) == replayed);
    assert(Owner::at(append.conv.states, 16, f.hashes[16]) == replayed);
    auto* appended = f.capture(append.conv.states, 20, 71);
    assert(Owner::best(append.conv.states, 20, f.hashes) == appended);

    // A cancelled/rolled-back chunk cannot retain an anchor beyond its
    // committed boundary. Existing earlier state remains valid for parking.
    f.prune(append.conv.states, 16);
    assert(appended->refs == 0 && Owner::best(append.conv.states, 20, f.hashes) == replayed);
    assert(Owner::at(append.conv.states, 20, f.hashes[20]) == nullptr);
    append.reset();

    // Incomplete snapshot publication and foreign unowned snapshots never
    // become a candidate even if their position/hash metadata looks useful.
    auto pending = f.hub.reserve_snapshot(*appended);
    assert(pending);
    std::vector<Snapshot*> not_published{pending.get()};
    assert(Owner::best(not_published, 20, f.hashes) == nullptr);
    assert(!pending.commit(20, f.hashes[20], 0, [] { return false; }));
    assert(Owner::at(not_published, 20, f.hashes[20]) == nullptr);
    pending.reset();

    // A leased source record cannot be evicted. A selected payload stays
    // pinned even after its last record is evicted and its refs reach zero.
    {
        auto left = f.hub.checkout(0);
        auto right = f.hub.checkout(1);
        auto child = f.hub.checkout(2);
        assert(!f.hub.evict_lru());
    }
    {
        Hub::SnapPin pin(f.hub, *replayed);
        while (f.hub.evict_lru()) {}
        assert(replayed->refs == 0 && replayed->pins == 1);
        assert(!f.hub.reserve_snapshot(*replayed));
        assert(Owner::at({replayed}, 16, f.hashes[16]) == nullptr);
    }
    assert(f.hub.reserve_snapshot(*replayed));
    f.empty();
    std::cout << "Snapshot-owner lineage PASS: semantic twins, real Hub forks/COW, replay/cancel bounds, partial-page append, writer/pin/lease lifetime and four-owner isolation\n";
}
