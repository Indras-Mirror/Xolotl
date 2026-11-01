#include "strata/core/conversation_cache.hpp"

#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

namespace {
std::atomic<long> fail_after{-1};
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::abort(); }
}
using strata::core::SharedBytes;
const uint8_t* address(const SharedBytes& b) { return b.data(); }
}

#if defined(__GNUC__)
#define TEST_NOINLINE __attribute__((noinline))
#else
#define TEST_NOINLINE
#endif
// Keep the intentional malloc/free replacement hooks opaque to the optimizer;
// their pairing is deliberate, and failure injection must remain observable.
TEST_NOINLINE void* operator new(std::size_t n) {
    if (fail_after.load() >= 0 && fail_after.fetch_sub(1) == 0) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
TEST_NOINLINE void* operator new[](std::size_t n) { return ::operator new(n); }
TEST_NOINLINE void operator delete(void* p) noexcept { std::free(p); }
TEST_NOINLINE void operator delete(void* p, std::size_t) noexcept { std::free(p); }
TEST_NOINLINE void operator delete[](void* p) noexcept { std::free(p); }
TEST_NOINLINE void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#undef TEST_NOINLINE

int main() {
    using namespace strata::core;
    const std::vector<uint8_t> expected = {1,2,3,4};
    SharedBytes original{1,2,3,4};
    SharedBytes deep(original), assigned;
    assigned = original;
    check(original == expected && deep == original && assigned == original, "default copy values");
    check(address(deep) != address(original) && address(assigned) != address(original), "ordinary copies are deep by default");
    check(!original.published() && !deep.published() && !assigned.published(), "ordinary deep copies remain private");
    auto alias = original.share();
    check(address(alias) == address(original) && alias.shared() && original.shared(), "only explicit share aliases payload");
    check(original.published() && alias.published(), "sharing permanently marks the block before publication");
    const auto* stable = address(original);
    alias.data()[0] = 9;
    check(original == expected && alias[0] == 9 && address(alias) != stable, "mutable data detaches before raw writes");
    check(address(original) == stable, "old reader storage does not move");
    auto resize = original.share();
    resize.resize(10, 7);
    check(original == expected && resize.size() == 10 && resize[9] == 7, "resize growth detaches");
    auto shrink = original.share();
    shrink.resize(1);
    check(original.size() == 4 && shrink.size() == 1, "shrink does not change older readers");
    auto clear = original.share();
    clear.clear();
    check(clear.empty() && clear.capacity() >= 4 && original == expected, "clear preserves vector capacity and old readers");
    auto edit = original.share();
    edit.begin()[1] = 8;
    check(original == expected && edit[1] == 8, "mutable iterators detach");
    edit = original.share();
    edit[2] = 8;
    check(original == expected && edit[2] == 8, "indexed writes detach");
    edit = original.share();
    const uint8_t extra[] = {8,9};
    edit.insert(std::as_const(edit).begin() + 2, std::begin(extra), std::end(extra));
    check(original == expected && edit == std::vector<uint8_t>({1,2,8,9,3,4}), "range insert detaches");
    edit = original.share();
    edit.insert(std::as_const(edit).begin(), 2, 7);
    check(original == expected && edit == std::vector<uint8_t>({7,7,1,2,3,4}), "count insert detaches");
    edit = original.share();
    edit.erase(std::as_const(edit).begin() + 1, std::as_const(edit).begin() + 3);
    check(original == expected && edit == std::vector<uint8_t>({1,4}), "erase detaches");
    edit = original.share();
    edit.assign(3, 3);
    check(edit == std::vector<uint8_t>({3,3,3}) && original == expected, "count assign is not an iterator overload");
    edit.assign({4,5});
    check(edit == std::vector<uint8_t>({4,5}), "initializer assign");
    SharedBytes reserved{1,2,3,4};
    reserved.reserve(100);
    auto reserved_alias = reserved.share();
    reserved_alias.resize(2);
    check(reserved_alias.capacity() == reserved.capacity() && reserved_alias.size() == 2 && reserved.size() == 4,
          "detach preserves reserved capacity");
    const auto charged = original.bytes();
    auto more = original.share();
    check(charged >= original.capacity() + SharedBytes::allocation_overhead &&
          more.bytes() == charged, "each alias conservatively charges full capacity and storage header");
    SharedBytes moved(std::move(more));
    check(more.empty() && moved == expected && address(moved) == address(original), "move keeps alias identity and empties source");
    auto* self = &moved;
    moved = std::move(*self);
    check(moved == expected, "self move does not lose bytes");
    SharedBytes empty;
    auto empty_alias = empty.share();
    check(empty.empty() && empty_alias.empty() && empty == empty_alias && empty.bytes() == 0, "empty sharing has no allocation");

    // Simulate the existing completed D2H fence: publish sharing only after the
    // producer has stopped using its mutable pointer. No deferred raw writer is
    // allowed to survive share(), just as with publishing an immutable vector.
    SharedBytes captured(1024, 0);
    auto* producer = captured.data();
    std::thread writer([producer] { std::memset(producer, 0x5a, 1024); });
    writer.join();
    auto parked = captured.share();
    captured.data()[0] = 0x66;
    check(parked[0] == 0x5a && parked[1023] == 0x5a, "fenced capture remains immutable after later writes");

    ConversationCheckpoint checkpoint;
    checkpoint.ids = {1,2,3}; checkpoint.imgs = {{1,99}}; checkpoint.used = 7; checkpoint.pinned = true;
    checkpoint.gdn = {1}; checkpoint.ple = {2}; checkpoint.tails = {3}; checkpoint.dead = {4}; checkpoint.block_pos = {5};
    auto shared_checkpoint = checkpoint; // metadata must be independently owned
    shared_checkpoint.gdn = checkpoint.gdn.share(); shared_checkpoint.ple = checkpoint.ple.share();
    shared_checkpoint.tails = checkpoint.tails.share(); shared_checkpoint.dead = checkpoint.dead.share();
    shared_checkpoint.block_pos = checkpoint.block_pos.share();
    check(checkpoint.bytes() == shared_checkpoint.bytes(), "checkpoint budget charges shared payloads fully");
    shared_checkpoint.ids[0] = 8; shared_checkpoint.imgs[0].hash = 100;
    shared_checkpoint.used = 9; shared_checkpoint.pinned = false;
    shared_checkpoint.gdn[0] = 11; shared_checkpoint.ple[0] = 12; shared_checkpoint.tails[0] = 13;
    shared_checkpoint.dead[0] = 14; shared_checkpoint.block_pos[0] = 15;
    check(checkpoint.ids[0] == 1 && checkpoint.imgs[0].hash == 99 && checkpoint.used == 7 && checkpoint.pinned,
          "token/image/LRU/pin metadata never shares");
    check(checkpoint.gdn[0] == 1 && checkpoint.ple[0] == 2 && checkpoint.tails[0] == 3 &&
          checkpoint.dead[0] == 4 && checkpoint.block_pos[0] == 5, "all five payload kinds detach");

    int allocation_failures = 0;
    for (int operation = 0; operation < 5; ++operation) {
        bool completed = false;
        for (int fail = 0; fail < 8 && !completed; ++fail) {
            SharedBytes source{1,2,3,4};
            auto target = source.share();
            const auto* before = address(target);
            const auto old_size = target.size(), old_capacity = target.capacity(), old_charge = target.bytes();
            fail_after = fail;
            try {
                if (operation == 0) target.resize(1024, 9);
                if (operation == 1) target.reserve(1024);
                if (operation == 2) target.data()[0] = 9;
                if (operation == 3) target.insert(std::as_const(target).begin() + 1, 128, 9);
                if (operation == 4) target = source; // default deep-copy assignment
                fail_after = -1;
                completed = true;
            } catch (const std::bad_alloc&) {
                fail_after = -1;
                ++allocation_failures;
                check(target == expected && address(target) == before && target.size() == old_size &&
                      target.capacity() == old_capacity && target.bytes() == old_charge,
                      "allocation failure has no observable partial mutation");
            }
            check(source == expected, "allocation or mutation never damages an older alias");
        }
        check(completed, "fault injection eventually reaches successful operation");
    }
    check(allocation_failures >= 10, "fault injection exercised control/payload/growth allocations");
    fail_after = 0;
    auto no_allocate = original.share();
    fail_after = -1;
    check(address(no_allocate) == address(original), "sharing itself does not allocate");

    const SharedBytes thread_source(1024, 0x44);
    std::atomic<bool> valid{true};
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) readers.emplace_back([&,i] {
        for (int n = 0; n < 50; ++n) {
            auto copy = thread_source.share();
            copy.data()[i] = uint8_t(i);
            if (thread_source[i] != 0x44 || copy[i] != i || thread_source[1023] != 0x44) valid = false;
        }
    });
    for (auto& reader : readers) reader.join();
    check(valid, "distinct aliases can concurrently detach while immutable readers survive");
    SharedBytes last_alias = [] {
        SharedBytes local{1,2,3,4};
        return local.share();
    }();
    check(!last_alias.shared() && last_alias.published(), "last remaining alias still remembers publication");
    const auto* old_last = address(last_alias);
    last_alias.data()[0] = 9;
    check(address(last_alias) != old_last && !last_alias.published(),
          "last alias clones rather than using refcount as write permission");
    for (int repeat = 0; repeat < 24; ++repeat) {
        SharedBytes left(16384, 0x44);
        auto right = left.share(); // only two owners: no extra immutable base
        std::barrier gate(2);
        std::thread a([&] {
            gate.arrive_and_wait();
            left.data()[0] = 0x99;
        });
        std::thread b([&] {
            gate.arrive_and_wait();
            right.erase(std::as_const(right).begin() + 3, std::as_const(right).begin() + 4);
        });
        a.join(); b.join();
        check(left.size() == 16384 && std::as_const(left)[0] == 0x99 && std::as_const(left)[3] == 0x44 &&
              right.size() == 16383 && std::as_const(right)[0] == 0x44 && std::as_const(right)[3] == 0x44,
              "last two aliases can detach and erase concurrently without a base reader");
        auto frozen_tail = right.share();
        right = {}; // release the source before the surviving alias writes
        const auto* tail_before = address(frozen_tail);
        frozen_tail.pop_back();
        check(address(frozen_tail) != tail_before && frozen_tail.size() == 16382,
              "source release does not unfreeze the surviving alias");
    }
    std::printf("shared_bytes_test: %d checks, %d allocation failures preserved\n", checks, allocation_failures);
}
