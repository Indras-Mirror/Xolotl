// tools/v41/engine/ds41_engram.hpp - DeepSeek-V4.1 Engram: the n-gram hash (model.py / engram.py NgramHashState)
// and the row reader.  The tables are NOT copied anywhere: rows are read straight from the HF safetensors shards
// (layers.{1,14}.engram.embed.weight F8_E4M3 [rows, 256] + .scale F8_E8M0 [rows, 8]), two small preads per row,
// in parallel.  A row dequantizes to 256 floats = fp8 * 2^(e8m0 - 127) per 32 (model.py ParallelEngramEmbedding;
// exactly representable in bf16, so model.py's bf16 cast is a no-op on them).
//
// rows()/start() are the fast batch reader: hashes are deduped and sorted by row id, so per row the two preads come out
// ascending and the unique rows whose bytes share a 4 KiB page are one pread - which on the exFAT volume the tables live
// on is the difference between a random walk the driver serializes (~23K reads/s) and a sorted one it does not (~135K/s).
// rows_reference() keeps the original per-row random reader so the bench can diff the two byte for byte.
#pragma once

#include "ds41_geometry.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::v41 {

class Ds41Engram {
public:
    struct RawMap;   ///< a fragmented table file read straight from its block device (see ds41_engram.cpp)
    ~Ds41Engram();
    /// dirs: where the shards are (the index json is looked up in each, the shard files too).
    bool open(const Ds41Geometry & g, const std::vector<std::string> & dirs, int threads, std::string & err);
    bool enabled() const { return !tables_.empty(); }
    int64_t n_cols() const { return cols_; }                    ///< hash columns per token per layer (24)
    int64_t row_dim() const { return dim_; }                    ///< 256
    /// Record token `tid` at position `pos` (history for the hashes; positions may be rewritten).
    void set_token(int64_t pos, int tid);
    /// Record position `pos` as a DEAD token (an image span): it takes no part in any n-gram and stops the look-back,
    /// so a text n-gram never spans an image (model.py NgramHashState.DEAD).
    void set_dead(int64_t pos);
    /// Hash ids of the n-grams ending at `pos` for Engram slot `e` (0 = layer 1, 1 = layer 14): out[cols].
    void hash(int64_t pos, int e, int64_t * out) const;
    /// Rows for tokens at positions pos0..pos0+n-1, every Engram slot: out[e] = n * cols * dim floats (token-major).
    bool rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    /// The pre-batch reference reader: one pread pair per row, no dedup / sort / coalescing.  Kept so the byte-identity
    /// gate can diff it against the fast path over real tokens; not used by the engine.
    bool rows_reference(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    /// Async batch.  start() hashes, sorts and kicks the reads into `out` (the caller's storage, re-sized in place so a
    /// reused buffer costs no allocation) and returns at once without waiting on disk; wait() blocks until the reads
    /// finish and reports errors.  `out` must stay alive and untouched until wait() returns - it is the destination, not
    /// a copy.  Lets a caller overlap a pass's reads with unrelated work (the dense half knows all hashes at begin()).
    struct Batch;
    std::shared_ptr<Batch> start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err);
    bool wait(const std::shared_ptr<Batch> & b, std::string & err);
    /// Drop this module's tables from the page cache (posix_fadvise DONTNEED on our own fds; not a global drop).
    void drop_cache();
    /// bytes read so far (stats)
    uint64_t bytes_read() const { return bytes_; }
    /// row reads (pread syscalls) issued by the fast path (stats)
    uint64_t row_reads() const { return reads_; }
    uint64_t cache_hits() const { return cache_hits_; }   ///< unique rows served from the row cache
    uint64_t cache_misses() const { return cache_miss_; }

private:
    struct Table { int fd_w = -1, fd_s = -1; uint64_t off_w = 0, off_s = 0; int64_t rows = 0;
                   std::shared_ptr<RawMap> rm_w, rm_s; };
    struct ReadGroup;               // one coalesced pread (defined in the .cpp)
    const Ds41Geometry * g_ = nullptr;
    std::vector<Table> tables_;
    // Row cache: raw rows (dim weight bytes + dim/32 scale bytes) already read, keyed (layer slot << 32 | row).  Filled
    // by every read until `cache_cap_` rows (STRATA_ENGRAM_CACHE_ROWS, default 2M ~ 0.6 GB; 0 = off), never evicted:
    // a prompt's rows are all cached as it is read, so decode n-grams that repeat the context skip the disk (the Engram
    // table sits on exFAT here: ~13K random reads/s - a decode token's ~48 reads cost ~10 ms cold).
    std::unordered_map<uint64_t, uint32_t> cache_idx_;
    std::vector<uint8_t> cache_pool_;
    size_t cache_cap_ = 0, cache_used_ = 0;
    uint64_t cache_hits_ = 0, cache_miss_ = 0;
    std::mutex cache_mu_;
    std::vector<int64_t> hist_;      // compressed ids by position
    std::vector<int64_t> offsets_;   // [slot * cols + col] bucket range start
    int64_t cols_ = 0, dim_ = 0;
    uint64_t bytes_ = 0, reads_ = 0;

    // A fixed I/O pool with a FIFO job queue.  The row reads are latency-bound, so the pool is the disk's queue depth:
    // a batch fires one job per worker and the jobs steal chunks of the batch's read list.  Backs both rows()/start()
    // and the reference reader's fork-join `parallel`.
    struct Task {
        std::function<void(int64_t, int64_t)> fn;
        int64_t n = 0, parts = 1;                 // [0, n) split into `parts` chunks
        std::atomic<int64_t> next { 0 };
        std::atomic<int> pending { 0 };
        std::mutex mu;
        std::condition_variable cv;
        void wait() { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return pending.load() == 0; }); }
        void done() { std::lock_guard<std::mutex> lk(mu); if (--pending == 0) cv.notify_all(); }
    };
    void pump();
    void task_run(const std::shared_ptr<Task> & t);
    void fire(const std::shared_ptr<Task> & t, int64_t parts);
    void run_sync(const std::shared_ptr<Task> & t, int64_t parts);
    void parallel(int64_t n, const std::function<void(int64_t, int64_t)> & fn);
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    int64_t nthreads_ = 1;
    bool stop_ = false;
};

}  // namespace strata::v41
