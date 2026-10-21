// tools/v41/engine/test_ds41_batch_plan.cpp - the batch-slot window planner (ds41_batch_slots.hpp ds41_plan_window):
// one routing class per window, every decoding slot of it, the prompt being read fills the rest, classes take turns.
// No model, no GPU.
#include "ds41_batch_slots.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace strata::v41;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: %s - ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::vector<Ds41PlanSlot> mk(std::initializer_list<int> cls) {   // -1 = not decoding
    std::vector<Ds41PlanSlot> v;
    for (int c : cls) v.push_back({ c >= 0, c >= 0 ? c : 0 });
    return v;
}

int main() {
    {   // nothing to run
        int rr = 0;
        const Ds41Plan p = ds41_plan_window(mk({ -1, -1, -1 }), -1, -1, 0, 16, rr);
        CHECK(p.cls < 0 && p.dec.empty() && p.read_rows == 0, "cls %d", p.cls);
    }
    {   // one class: every decoding slot, every window
        int rr = 0;
        for (int w = 0; w < 5; ++w) {
            const Ds41Plan p = ds41_plan_window(mk({ 0, -1, 0, 0 }), -1, -1, 0, 16, rr);
            CHECK(p.cls == 0 && p.dec == std::vector<int>({ 0, 2, 3 }), "window %d: %zu rows", w, p.dec.size());
        }
    }
    {   // two classes (exact / fast): never in one window, they alternate, every slot keeps moving
        int rr = 0;
        const auto s = mk({ 0, 1, 0, 1, 1 });
        std::vector<int> moved(5, 0);
        int last = -1, switches = 0;
        for (int w = 0; w < 40; ++w) {
            const Ds41Plan p = ds41_plan_window(s, -1, -1, 0, 16, rr);
            CHECK(p.cls == 0 || p.cls == 1, "cls %d", p.cls);
            for (int b : p.dec) {
                CHECK(s[(size_t) b].cls == p.cls, "slot %d (class %d) in a class %d window", b, s[(size_t) b].cls, p.cls);
                ++moved[(size_t) b];
            }
            switches += last >= 0 && p.cls != last;
            last = p.cls;
        }
        for (int b = 0; b < 5; ++b) CHECK(moved[(size_t) b] >= 10, "slot %d moved %d times in 40 windows", b, moved[(size_t) b]);
        CHECK(switches >= 10, "classes switched only %d times", switches);
    }
    {   // a prompt read beside the decoders of its class: the rest of the 16 rows
        int rr = 0;
        const Ds41Plan p = ds41_plan_window(mk({ 0, -1, 0 }), 1, 0, 100, 16, rr);
        CHECK(p.cls == 0 && p.dec.size() == 2 && p.read_rows == 14, "dec %zu read %d", p.dec.size(), p.read_rows);
        int rr2 = 0;
        const Ds41Plan q = ds41_plan_window(mk({ 0, -1, 0 }), 1, 0, 5, 16, rr2);
        CHECK(q.read_rows == 5, "a short tail: %d rows", q.read_rows);
    }
    {   // a read of another class gets windows of its own (16 rows), the decoders theirs
        int rr = 0;
        int read = 0, dec = 0;
        for (int w = 0; w < 6; ++w) {
            const Ds41Plan p = ds41_plan_window(mk({ 0, -1, 0 }), 1, 1, 1000, 16, rr);
            if (p.cls == 1) { CHECK(p.dec.empty() && p.read_rows == 16, "read window: dec %zu read %d", p.dec.size(), p.read_rows); ++read; }
            else { CHECK(p.cls == 0 && p.read_rows == 0 && p.dec.size() == 2, "decode window"); ++dec; }
        }
        CHECK(read >= 2 && dec >= 2, "read %d / decode %d windows of 6", read, dec);
    }
    {   // 16 slots, 15 decoding and one reading of the same class: the read still gets a row
        std::vector<Ds41PlanSlot> s(16, { true, 0 });
        s[7].decoding = false;
        int rr = 0;
        const Ds41Plan p = ds41_plan_window(s, 7, 0, 50, 16, rr);
        CHECK(p.dec.size() == 15 && p.read_rows == 1, "dec %zu read %d", p.dec.size(), p.read_rows);
        std::set<int> got(p.dec.begin(), p.dec.end());
        CHECK(!got.count(7), "the reading slot decodes");
    }
    {   // only a read (no decoder): all 16 rows are its
        int rr = 3;
        const Ds41Plan p = ds41_plan_window(mk({ -1, -1, -1, -1 }), 2, 0, 40, 16, rr);
        CHECK(p.cls == 0 && p.dec.empty() && p.read_rows == 16, "read %d", p.read_rows);
    }
    std::printf(fails ? "test_ds41_batch_plan: %d FAILED\n" : "test_ds41_batch_plan: all passed\n", fails);
    return fails ? 1 : 0;
}
