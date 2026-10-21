// tools/v41/engine/ds41_conv_pool.hpp - DeepSeek-V4.1 --serve: the conversation pool's STORAGE (many parked
// conversations in pinned host RAM, swapped onto the card's caches in milliseconds instead of a re-prefill).
//
// The engine keeps ONE live history on the card.  Every cache is position-indexed except the window ring and the ratio-2
// compressor state ring (ds41_dense.hpp: snapshot / park), so a conversation's whole device state at a turn boundary
// pos is:
//   - STREAMS: the compressed KV / indexer-key rows of the kv source layers (row r holds positions [r*ratio, r*ratio +
//     ratio)): rows [0, pos / ratio) of each stream.  ~890 B a position with --kv-fp4 (4 sources, 1x ratio 1 + 3x 2).
//   - REGIONS: the snapshot rings (window K of every layer, the ratio-2 compressor states) - ~5.65 MiB a boundary.
//   - host data: the boundary's logits (+ optional caller blobs), the tokens and the per-position identities.
//
// This class stores such states and copies them back.  It knows nothing about ggml: the layout (sizes) is given once,
// the device pointers on every save / restore (Ds41Dense::pool_regions + ds41_conv_pool_dense.hpp build both).
//
// Storage: pinned host memory (cudaHostAlloc) taken lazily in extents up to the budget and cut into equal SLABS
// (2 MiB), so there is no fragmentation.  A record's positions are CHUNKS of C positions (one slab: every stream's rows
// of those positions, stream after stream); a boundary's regions take ceil(region bytes / slab) slabs.  Slabs are
// reference-counted: a new record shares every FULL chunk that lies inside its common prefix (tokens AND identities)
// with an existing record - the shared system prompt, or the same conversation one turn back (then a save copies only
// the last partial chunk and the new tail).  Least-recently-used records leave first; pinned records (the root: a shared
// system prompt) never.
//
// All copies run in order on the pool's own CUDA stream (non-blocking w.r.t. the engine's ggml streams): save()
// returns after queueing its device-to-host copies; restore() queues host-to-device copies behind every unfinished save
// and wait() blocks the host until they landed (call it before the first pass).  Only that stream's DMA touches slab
// memory, so a slab freed by an eviction can be reused at once.
//
// Guarantees: a failed check never touches the device (restore validates the record, boundary, ranges, layout
// fingerprint and pointers before the first copy); a CUDA error after copies were queued reports live_clobbered (the
// live state must be treated as garbage: full re-prefill).  A failed save leaves the pool unchanged.  Bytes come back
// bit for bit (test_ds41_conv_pool on the CPU, bench_ds41_conv_pool on the card; cfg.verify adds a checksum per slab).
//
// Threads: one mutex guards the object (stats() from a metrics thread is fine); save / restore from the engine thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::v41 {

constexpr uint32_t kDs41PoolVersion = 1;

using Ds41PoolId = uint64_t;   ///< a record; 0 = none

/// A position-indexed device cache: `max_rows` rows of `row_bytes`, row r holds positions [r*ratio, (r+1)*ratio).
struct Ds41PoolStream {
    std::string name;
    size_t row_bytes = 0;
    int ratio = 1;          ///< 1 or 2 (any value >= 1 that divides the chunk works)
    int64_t max_rows = 0;   ///< rows usable for positions (the spare row excluded)
};
/// A per-boundary device blob (one snapshot ring tensor, one compressor state tensor), copied whole.
struct Ds41PoolRegion {
    std::string name;
    size_t bytes = 0;
};
struct Ds41PoolLayout {
    std::vector<Ds41PoolStream> streams;
    std::vector<Ds41PoolRegion> regions;
    int64_t max_positions = 0;   ///< positions the device caches hold (ctx_max)
    /// version + every size: a record is only restored into a layout with the same fingerprint
    uint64_t fingerprint() const;
    size_t bytes_per_position() const;   ///< sum of row_bytes / ratio (fractional rows rounded per chunk)
    size_t region_bytes() const;
    bool valid(std::string & err) const;
};
/// Device pointers of one save / restore: row 0 of every stream and the start of every region, in layout order.
struct Ds41PoolTargets {
    std::vector<void *> stream;
    std::vector<void *> region;
    uint64_t layout_fp = 0;      ///< the fingerprint of the layout the pointers were taken with (0 = not checked)
};

struct Ds41PoolConfig {
    size_t budget_bytes = (size_t) 8192 << 20;   ///< --conversation-cache-mib: slabs in use + host bytes
    size_t slab_bytes = (size_t) 2 << 20;        ///< one chunk / one piece of a boundary's regions
    size_t extent_bytes = (size_t) 256 << 20;    ///< pinned allocation unit (taken lazily, kept until destruction)
    int max_boundaries = 4;                      ///< per record: the first (deepest shared root) + the newest ones
    double pinned_max_frac = 0.5;                ///< pinned records may hold at most this share of the budget
    bool cuda = true;                            ///< false: plain host memory + memcpy (tests; "device" = host pointers)
    int device = 0;                              ///< CUDA device of the targets
    bool verify = false;                         ///< checksum every slab after save, check before restore (debug)
    bool prealloc = false;                       ///< take the whole budget at init (else extents as saves need them)
};

struct Ds41PoolBlob {
    const void * data = nullptr;
    size_t bytes = 0;
};
struct Ds41PoolSaveOpts {
    bool pin = false;                       ///< pin the new record (a shared root); refused past pinned_max_frac
    std::vector<Ds41PoolBlob> blobs;        ///< host blobs kept with the new boundary (e.g. the DSpark seed rows)
};

struct Ds41PoolMatch {
    Ds41PoolId rec = 0;
    int boundary = -1;          ///< index into the record's boundaries (ascending pos)
    int64_t pos = 0;            ///< the boundary: resume decoding at this position (0 = no match)
    int64_t lcp = 0;            ///< common prefix of the record's tokens + identities with the prompt (>= pos)
    bool has_logits = false;
    /// The same record's nearest boundary PAST lcp (index / position, -1 = none): the logic's fixed PARK_PARTIAL
    /// resumes at s < next_pos from it when next_pos - s <= RAW - SWA (the ring still holds the window).
    int next_boundary = -1;
    int64_t next_pos = -1;
};

struct Ds41PoolRestore {
    Ds41PoolId rec = 0;
    int64_t pos = 0, from = 0;              ///< positions [from, pos) were copied (+ the regions)
    std::vector<float> logits;              ///< the boundary's logits (empty if none were saved)
    std::vector<std::vector<uint8_t>> blobs;
    size_t bytes = 0;                       ///< host-to-device bytes queued
    uint64_t seq = 0;                       ///< the copy batch wait() waits for
    bool live_clobbered = false;            ///< copies reached the device before the failure: live state is garbage
};

struct Ds41PoolBoundaryInfo {
    int64_t pos = 0;
    bool logits = false;
};
struct Ds41PoolRecordInfo {
    Ds41PoolId id = 0;
    int64_t n = 0;
    bool pinned = false;
    uint64_t stamp = 0;
    size_t chunks = 0, own_slabs = 0;       ///< own = slabs only this record holds (what evicting it frees)
    std::vector<Ds41PoolBoundaryInfo> boundaries;
};

struct Ds41PoolStats {
    size_t records = 0, pinned = 0, boundaries = 0;
    size_t slabs_used = 0, slabs_total = 0, slab_bytes = 0, chunk_positions = 0;
    size_t pinned_allocated = 0, host_bytes = 0, used_bytes = 0, budget_bytes = 0;
    uint64_t saves = 0, saves_touched = 0, saves_failed = 0, restores = 0, restores_failed = 0;
    uint64_t evictions = 0, superseded = 0, shared_chunks = 0;
    uint64_t bytes_d2h = 0, bytes_h2d = 0, bytes_shared = 0;   ///< bytes_shared: chunk bytes a save did not copy
    double last_save_ms = 0, last_restore_ms = 0;              ///< device time of the last finished save / restore
    double total_save_ms = 0, total_restore_ms = 0, fence_wait_ms = 0;
};

class Ds41ConvPool {
public:
    struct Impl;
    Ds41ConvPool();
    ~Ds41ConvPool();
    Ds41ConvPool(const Ds41ConvPool &) = delete;
    Ds41ConvPool & operator=(const Ds41ConvPool &) = delete;

    bool init(const Ds41PoolLayout & layout, const Ds41PoolConfig & cfg, std::string & err);
    bool ready() const;
    const Ds41PoolLayout & layout() const;
    size_t chunk_positions() const;

    /// Save the device state of positions [0, n) (tokens / ids = the conversation, ids nullable = all 0) with a boundary
    /// at n.  Precondition: the device data is final (the engine synchronized after the snapshot).  Returns the record,
    /// or 0 (err set, pool unchanged).  An existing record with the same tokens + ids and a boundary at n is only
    /// touched (no copy).  An unpinned record whose tokens are a prefix of these is replaced by the new one (its chunks
    /// and boundaries carry over by reference).  Makes room by evicting least-recently-used unpinned records.
    Ds41PoolId save(const Ds41PoolTargets & dev, const int32_t * tokens, const uint64_t * ids, int64_t n,
                    const float * logits, size_t n_logits, std::string & err, const Ds41PoolSaveOpts & opts = {});
    /// The record + boundary sharing the longest prefix with the prompt: the deepest boundary <= min(lcp, n).
    /// Ties: the most recently used.  pos == 0: no match.
    Ds41PoolMatch best_prefix(const int32_t * tokens, const uint64_t * ids, int64_t n) const;
    /// Queue the copies of boundary `boundary` of `rec` onto the device: stream rows of positions [from, pos) (the
    /// caller vouches that [0, from) on the device are already this record's - a live state sharing that prefix) and
    /// the regions.  Validates everything before the first copy.  wait(out) before the first pass.
    bool restore(Ds41PoolId rec, int boundary, int64_t from, const Ds41PoolTargets & dev, Ds41PoolRestore & out,
                 std::string & err);
    /// Host-wait for a restore's copies; false = CUDA error (live state garbage).
    bool wait(const Ds41PoolRestore & r, std::string * err = nullptr);
    /// Host-wait for every queued copy (before snapshot() and before a pass that writes positions a pending save reads).
    bool fence(std::string * err = nullptr);
    /// True while copies are queued and not finished.
    bool busy() const;

    /// A pinned record holding the prefix [0, pos_b) of rec (by reference; no copy).  0 = refused.
    Ds41PoolId pin_root(Ds41PoolId rec, int boundary, std::string & err);
    bool pin(Ds41PoolId rec, bool on);
    bool evict(Ds41PoolId rec);
    size_t clear(bool keep_pinned);
    /// Return free extents to the OS? Extents are kept (cudaFreeHost is a device-wide sync); this only reports.

    Ds41PoolStats stats() const;
    std::string stats_line() const;
    std::vector<Ds41PoolRecordInfo> records() const;
    /// Tokens of a record (empty if gone).
    std::vector<int32_t> tokens(Ds41PoolId rec) const;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::v41
