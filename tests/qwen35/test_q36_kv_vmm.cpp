#include "../../src/qwen35/q36_kv_vmm.hpp"
#include <cassert>
#include <set>
#include <iostream>
using namespace strata::q36::kv;
struct Mock : Driver {
    std::set<uintptr_t> maps;
    size_t fail_at = SIZE_MAX, calls = 0, reserves = 0, frees = 0;
    size_t granularity() const override { return size_t(2) << 20; }
    bool reserve(uintptr_t& p, size_t, size_t) override { p = 1ULL << 40; ++reserves; return true; }
    bool map(uintptr_t p, size_t) override { if (calls++ == fail_at) return false; return maps.insert(p).second; }
    bool unmap(uintptr_t p, size_t) override { return maps.erase(p) == 1; }
    bool free_address(uintptr_t, size_t) override { ++frees; return true; }
};
int main() {
    constexpr size_t MiB = size_t(1) << 20;
    // 11 attention blocks incl MTP, K/V, two heads, 256 FP16 components.
    constexpr size_t planes = 44, ctx = 262144, cell = 512;
    Mock driver;
    Budget budget(5632 * MiB);
    {
        Arena arena(budget, driver, planes, ctx, cell);
        assert(arena.init());
        assert(arena.stride() == 128 * MiB);
        assert(arena.bytes_for(40000) == 880 * MiB);
        assert(arena.bytes_for(ctx) == 5632 * MiB);
        assert(arena.ensure(40000));
        assert(arena.committed_tokens() == 40960);
        assert(budget.used() == 880 * MiB);
        const auto original = driver.maps;
        // Transactionally roll back on a map failure, preserving previous prefix.
        driver.fail_at = driver.calls + 23;
        assert(!arena.ensure(80000));
        assert(driver.maps == original);
        assert(budget.used() == 880 * MiB);
        driver.fail_at = SIZE_MAX;
        assert(arena.ensure(ctx));
        assert(!arena.ensure(ctx + 1));
        assert(budget.used() == 5632 * MiB);
        assert(arena.trim(40000));
        assert(driver.maps == original);
        assert(budget.used() == 880 * MiB);
    }
    assert(budget.used() == 0 && driver.maps.empty());
    assert(driver.reserves == driver.frees);
    // Four active 40K requests fit; a further full-context reservation queues.
    Mock d1,d2,d3,d4,d5;
    Arena a(budget,d1,planes,ctx,cell), b(budget,d2,planes,ctx,cell), c(budget,d3,planes,ctx,cell), d(budget,d4,planes,ctx,cell), e(budget,d5,planes,ctx,cell);
    assert(a.init() && b.init() && c.init() && d.init() && e.init());
    assert(a.ensure(40000) && b.ensure(40000) && c.ensure(40000) && d.ensure(40000));
    assert(budget.used() == 3520 * MiB);
    assert(!e.ensure(ctx));
    assert(d5.maps.empty());
    // Output reservation is rounded and bounded; 256K remains legal after reclaim.
    assert(a.trim(0) && b.trim(0) && c.trim(0) && d.trim(0));
    assert(e.ensure(ctx));
    Mock bad;
    Arena overflow(budget,bad,SIZE_MAX,ctx,cell);
    assert(!overflow.init());
    Arena misaligned(budget,bad,planes,ctx-1,cell);
    assert(!misaligned.init());
    assert(e.bytes_for(SIZE_MAX) == SIZE_MAX);
    size_t out;
    assert(!mul(SIZE_MAX,2,out));
    assert(!align_up(SIZE_MAX,2,out));
    // Sparse VMM park/restore: each copied row lies inside a mapped physical
    // page, including partial final host pages and non-page-aligned starts.
    for (size_t end : {size_t(1), size_t(1023), size_t(1024), size_t(1025),
                       size_t(4095), size_t(4096), size_t(4097), size_t(40001), ctx}) {
        const size_t mapped = (end + 4095) / 4096 * 4096;
        for (size_t start : {size_t(0), end / 2, end - 1, end}) {
            size_t total = 0;
            assert(for_each_copy_row(ctx, cell, 2, 1024, mapped, start, end,
                [&](size_t page, size_t dev, size_t host, size_t bytes) {
                    const size_t head = dev / (ctx * cell);
                    const size_t position = dev % (ctx * cell);
                    assert(head < 2);
                    assert(position + bytes <= mapped * cell);
                    assert(position / (2 * MiB) == (position + bytes - 1) / (2 * MiB));
                    assert(host / (1024 * cell) == head);
                    assert(host % (1024 * cell) + bytes <= 1024 * cell);
                    assert(page * 1024 * cell + host % (1024 * cell) == position);
                    total += bytes;
                }));
            assert(total == (end - start) * cell * 2);
        }
    }
    bool called = false;
    auto invalid = [&](size_t, size_t, size_t, size_t) { called = true; };
    assert(!for_each_copy_row(ctx, cell, 2, 1024, 4096, 0, 4097, invalid));
    assert(!for_each_copy_row(ctx, cell, 2, 1024, ctx, 20, 19, invalid));
    assert(!for_each_copy_row(SIZE_MAX, cell, 2, 1024, ctx, 0, 1, invalid));
    assert(!called);
    std::cout << "q36 KV VMM CPU planner: PASS\n";
}
