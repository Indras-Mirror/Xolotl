#pragma once
#include <atomic>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#endif

namespace strata::m31 {
// A reusable barrier for the participants actually assigned to a custom CPU op.
// ggml may assign fewer workers than a caller requested. Never wait for the
// configured pool size: only the callback's nth participants enter this barrier.
class CpuTaskBarrier {
    // Match v8's 64-byte spacing on x86: arrival writes must not invalidate
    // the cache line that the other participants poll for completion.
    alignas(64) std::atomic<unsigned> arrived_{0};
    alignas(64) std::atomic<unsigned> generation_{0};
    static void relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        _mm_pause();
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
        __asm__ volatile("yield" ::: "memory");
#elif defined(_MSC_VER) && defined(_M_ARM64)
        __yield();
#elif defined(__riscv)
#if defined(__riscv_zihintpause)
        __asm__ volatile("pause");
#else
        __asm__ volatile(".4byte 0x100000F");
#endif
#endif
    }
public:
    void wait(unsigned participants) noexcept {
        if (participants <= 1) return;
        const auto generation = generation_.load(std::memory_order_relaxed);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
        } else {
            // Match the accepted v8 ggml barrier: CPU relax until completion,
            // without a spin limit or an operating-system scheduler yield.
            while (generation_.load(std::memory_order_acquire) == generation)
                relax();
        }
    }
};
} // namespace strata::m31
