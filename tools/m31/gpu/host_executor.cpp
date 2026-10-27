#include "host_executor.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace strata::m31 {
namespace {
thread_local const void * active_executor = nullptr;
struct ActiveScope {
    const void * previous = active_executor;
    explicit ActiveScope(const void * executor) { active_executor = executor; }
    ~ActiveScope() { active_executor = previous; }
};
struct Environment {
    fenv_t value;
#if defined(__SSE__)
    unsigned mxcsr;
#endif
    Environment() {
        if (fegetenv(&value)) throw std::runtime_error("cannot read host FP environment");
#if defined(__SSE__)
        mxcsr = _mm_getcsr();
#endif
    }
    void apply() const noexcept {
        fesetenv(&value);
#if defined(__SSE__)
        _mm_setcsr(mxcsr);
#endif
    }
};
std::pair<size_t, size_t> partition(size_t count, unsigned ranges, unsigned index) {
    const size_t base = count / ranges, extra = count % ranges;
    const size_t begin = base * index + std::min<size_t>(index, extra);
    return {begin, begin + base + (index < extra)};
}
} // namespace

struct GpuHostExecutor::Impl {
    Options cfg;
    mutable std::mutex submission;
    std::mutex state;
    std::condition_variable work, done;
    std::vector<std::thread> workers;
    bool stopping = false, closed = false;
    uint64_t generation = 0;
    const Rows * callback = nullptr;
    size_t count = 0;
    unsigned participants = 0, pending = 0;
    std::exception_ptr failure;
    Environment caller_environment;
    std::atomic<int> exceptions{0};
    std::atomic<unsigned> mxcsr_flags{0};
    std::atomic<size_t> serial_jobs{0}, parallel_jobs{0}, worker_ranges{0};

    explicit Impl(Options options) : cfg(options) {
        if (cfg.threads < 1 || cfg.threads > 64 || !cfg.parallel_threshold || !cfg.minimum_rows_per_worker)
            throw std::invalid_argument("GPU host executor: invalid threads/threshold/grain");
        if (!cfg.affinity.empty()) {
            if (cfg.affinity.size() != cfg.threads)
                throw std::invalid_argument("GPU host executor: affinity must match thread count");
#if defined(__linux__)
            for (size_t i = 0; i < cfg.affinity.size(); ++i) {
                if (cfg.affinity[i] >= CPU_SETSIZE ||
                    std::find(cfg.affinity.begin(), cfg.affinity.begin() + i, cfg.affinity[i]) != cfg.affinity.begin() + i)
                    throw std::invalid_argument("GPU host executor: invalid/duplicate CPU affinity");
            }
#else
            throw std::invalid_argument("GPU host executor: explicit affinity requires Linux");
#endif
        }
        workers.reserve(cfg.threads - 1);
        try {
            for (unsigned index = 1; index < cfg.threads; ++index) {
                workers.emplace_back([this, index] { worker(index); });
#if defined(__linux__)
                if (!cfg.affinity.empty()) {
                    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cfg.affinity[index], &mask);
                    if (pthread_setaffinity_np(workers.back().native_handle(), sizeof(mask), &mask))
                        throw std::runtime_error("GPU host executor: cannot set worker CPU affinity");
                }
#endif
            }
        } catch (...) {
            { std::lock_guard lock(state); stopping = true; }
            work.notify_all();
            for (auto & thread : workers) thread.join();
            throw;
        }
    }
    ~Impl() { shutdown(); }

    void run(const Rows & job, size_t total, unsigned ranges, unsigned index,
             const Environment & incoming) noexcept {
        try {
            const Environment previous;
            incoming.apply();
            {
                // Restore the worker even when user work throws.
                struct Restore {
                    const Environment & prior;
                    Impl * self;
                    ~Restore() {
                        self->exceptions.fetch_or(fetestexcept(FE_ALL_EXCEPT), std::memory_order_relaxed);
#if defined(__SSE__)
                        self->mxcsr_flags.fetch_or(_mm_getcsr() & 0x3fU, std::memory_order_relaxed);
#endif
                        prior.apply();
                    }
                } restore{previous, this};
                ActiveScope scope(this);
                const auto [begin, end] = partition(total, ranges, index);
                job(begin, end);
            }
        } catch (...) {
            std::lock_guard lock(state);
            if (!failure) failure = std::current_exception();
        }
    }
    void worker(unsigned index) {
        uint64_t seen = 0;
        std::unique_lock lock(state);
        while (true) {
            work.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) return;
            seen = generation;
            if (index >= participants) continue;
            const auto * job = callback;
            const size_t total = count;
            const unsigned ranges = participants;
            const Environment incoming = caller_environment;
            lock.unlock();
            run(*job, total, ranges, index, incoming);
            worker_ranges.fetch_add(1, std::memory_order_relaxed);
            lock.lock();
            if (--pending == 0) done.notify_one();
        }
    }
    bool submit(size_t total, const Rows & job) {
        if (active_executor == this) throw std::logic_error("GPU host executor: reentrant submission");
        std::unique_lock serial(submission);
        if (closed) throw std::logic_error("GPU host executor: already closed");
        if (!job) throw std::invalid_argument("GPU host executor: empty callback");
        if (!total) return false;
#if defined(__linux__)
        struct CallerAffinity {
            cpu_set_t prior{};
            bool changed = false;
            explicit CallerAffinity(const std::vector<unsigned> & cpus) {
                if (cpus.empty()) return;
                if (pthread_getaffinity_np(pthread_self(), sizeof(prior), &prior))
                    throw std::runtime_error("GPU host executor: cannot read caller CPU affinity");
                cpu_set_t next; CPU_ZERO(&next); CPU_SET(cpus[0], &next);
                if (pthread_setaffinity_np(pthread_self(), sizeof(next), &next))
                    throw std::runtime_error("GPU host executor: cannot set caller CPU affinity");
                changed = true;
            }
            ~CallerAffinity() { if (changed) pthread_setaffinity_np(pthread_self(), sizeof(prior), &prior); }
        } caller_affinity(cfg.affinity);
#endif
        const unsigned ranges = unsigned(std::min<size_t>(cfg.threads, total / cfg.minimum_rows_per_worker));
        if (total < cfg.parallel_threshold || ranges < 2) {
            serial_jobs.fetch_add(1, std::memory_order_relaxed);
            ActiveScope scope(this);
            job(0, total);
            return false;
        }
        const Environment incoming;
        {
            std::lock_guard lock(state);
            caller_environment = incoming;
            callback = &job;
            count = total;
            participants = ranges;
            pending = ranges - 1;
            failure = nullptr;
            exceptions.store(0, std::memory_order_relaxed);
            mxcsr_flags.store(0, std::memory_order_relaxed);
            ++generation;
        }
        parallel_jobs.fetch_add(1, std::memory_order_relaxed);
        work.notify_all();
        run(job, total, ranges, 0, incoming);
        std::exception_ptr error;
        {
            std::unique_lock lock(state);
            done.wait(lock, [&] { return pending == 0; });
            callback = nullptr;
            error = failure;
            failure = nullptr;
        }
        // run() restored caller control bits. Merge sticky status from all ranges,
        // matching independent rows evaluated serially under the same environment.
        feraiseexcept(exceptions.load(std::memory_order_relaxed));
#if defined(__SSE__)
        _mm_setcsr(_mm_getcsr() | mxcsr_flags.load(std::memory_order_relaxed));
#endif
        if (error) std::rethrow_exception(error);
        return true;
    }
    void shutdown() {
        if (active_executor == this) throw std::logic_error("GPU host executor: reentrant close");
        std::lock_guard serial(submission);
        if (closed) return;
        { std::lock_guard lock(state); stopping = true; }
        work.notify_all();
        for (auto & thread : workers) thread.join();
        closed = true;
    }
};
GpuHostExecutor::GpuHostExecutor(Options options) : p_(std::make_unique<Impl>(options)) {
    int inherited_cpus = -1;
#if defined(__linux__)
    cpu_set_t allowed;
    if (!pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed)) inherited_cpus = CPU_COUNT(&allowed);
#endif
    std::fprintf(stderr, "M31 persistent host executor: threads=%u workers=%u min_rows=%zu grain=%zu explicit_cpus=%zu inherited_cpus=%d\n",
        options.threads, options.threads - 1, options.parallel_threshold, options.minimum_rows_per_worker,
        options.affinity.size(), inherited_cpus);
}
GpuHostExecutor::~GpuHostExecutor() = default;
bool GpuHostExecutor::parallel_rows(size_t count, const Rows & callback) { return p_->submit(count, callback); }
void GpuHostExecutor::close() { p_->shutdown(); }
GpuHostExecutor::Stats GpuHostExecutor::stats() const {
    return {p_->serial_jobs.load(), p_->parallel_jobs.load(), p_->worker_ranges.load()};
}
GpuHostExecutor::Options GpuHostExecutor::options() const { return p_->cfg; }
} // namespace strata::m31
