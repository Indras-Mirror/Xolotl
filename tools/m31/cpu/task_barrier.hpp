#pragma once
#include <atomic>
#include <thread>

namespace strata::m31 {
// A reusable barrier for the participants actually assigned to a custom CPU op.
// ggml may assign fewer workers than a caller requested. Never wait for the
// configured pool size: only the callback's nth participants enter this barrier.
class CpuTaskBarrier {
    std::atomic<unsigned> arrived_{0};
    std::atomic<unsigned> generation_{0};
public:
    void wait(unsigned participants) noexcept {
        if (participants <= 1) return;
        const auto generation = generation_.load(std::memory_order_relaxed);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
        } else {
            unsigned spins = 0;
            while (generation_.load(std::memory_order_acquire) == generation)
                if (++spins >= 128) std::this_thread::yield();
        }
    }
};
} // namespace strata::m31
