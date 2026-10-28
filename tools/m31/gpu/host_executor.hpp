#pragma once
// Bounded, reusable CPU work for host-side activation quantization and OAI.
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace strata::m31 {
class GpuHostExecutor {
public:
    struct Options {
        unsigned threads = 32;          // includes the submitting thread; 1..64
        size_t parallel_threshold = 32; // smaller calls never wake workers
        size_t minimum_rows_per_worker = 4;
        // Optional exact worker placement; affinity[0] is the caller during jobs.
        // Empty preserves the affinity inherited when the executor was created.
        std::vector<unsigned> affinity;
    };
    struct Stats {
        size_t serial_jobs = 0, parallel_jobs = 0, worker_ranges = 0;
    };
    using Rows = std::function<void(size_t begin, size_t end)>;
    explicit GpuHostExecutor(Options options);
    ~GpuHostExecutor();
    GpuHostExecutor(const GpuHostExecutor &) = delete;
    GpuHostExecutor & operator=(const GpuHostExecutor &) = delete;

    // Submissions are serialized. Callback ranges are disjoint and cover [0,count).
    // Each row must be independent; callbacks must not modify FP control settings.
    // Workers use the caller's fenv/MXCSR; their FP exception flags are merged back.
    // Callback exceptions drain all ranges before rethrow; the next job remains usable.
    // Reentrant use of this same executor from its callback is rejected.
    bool parallel_rows(size_t count, const Rows & callback);
    bool parallel_count(size_t count, const Rows & callback) { return parallel_rows(count, callback); }
    void close(); // waits for the current submission; idempotent; joins all workers
    Stats stats() const;
    Options options() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::m31
