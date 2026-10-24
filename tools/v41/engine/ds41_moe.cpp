// tools/ds4/ds4_moe.cpp - the DeepSeek-V4 routed-expert tier (slice ds4-moe).
//
// The body of `Ds41MoeTier`, extracted from `tools/ds4/moe_replay.cpp`'s `Engine` (the measured harness, FINDINGS
// s10-s12) so the real engine can use it once per layer.  Read the header for the contract, the invariant and the
// SwiGLU-clamp note; this file is the implementation and keeps the replay's per-layer shape, minus the replay's
// synthetic dense gap and its recorded-route table.
//
// TWO TU FLAVOURS OF ONE SOURCE.  Compiled without `DS4_MOE_CUDA` it contains no CUDA call at all and references
// no symbol of `strata_kernels` / `strata_engine`, so a binary that links it (this slice's gate) cannot
// initialise the driver - accidental or otherwise.  Compiled with `DS4_MOE_CUDA` it is the full three-tier
// implementation.  `tools/ds4/cmake/ds4_moe.cmake` builds both.
#include "ds41_moe.hpp"
#include "ds41_transcode.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <array>
#include <functional>
#include <thread>
#include <utility>

#include "ds41_sys.hpp"

#include "strata/artifact/ds4_geometry.hpp"
#include "ds41_geometry.hpp"

#if defined(DS4_MOE_CUDA)
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include "ds41_remote.hpp"
#if __has_include(<cuda_profiler_api.h>)   // a CUDA install without the profiler API (some Windows toolkits)
#include <cuda_profiler_api.h>
#define DS41_HAVE_PROFILER_API 1
#endif
#if defined(DS4_MOE_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#include "ds41_mmq_plan.hpp"
#endif
#endif

namespace cpu = strata::kernels::cpu;

namespace strata::v41 {

namespace {
// Windows' WDDM driver model: native Windows, or WSL2 (its GPU goes through /dev/dxg to the Windows driver) - where a
// large pin mapped into two CUDA contexts leaves WDDM refusing later allocations (#32; strata generate's #253).
[[maybe_unused]] bool ds41_under_wddm() {
#ifdef _WIN32
    return true;
#else
    static const bool dxg = std::filesystem::exists("/dev/dxg");
    return dxg;
#endif
}

constexpr int kMaxParts = 8;   ///< the grouped kernel's group/entry cap; top-k 6 needs no more
constexpr int kMaxTok = Ds41MoeTier::kMaxMulti;   ///< tokens one run_multi call takes (a verify pass: 5-6 at depth 4)
/// A pool job carries at most cpu::MAXT tokens (the CPU kernels' register tiles); an expert that more of a call's
/// tokens route to becomes up to this many jobs, each its own slice of the tokens (every row is computed on its own,
/// so the outputs are the same as one job's).
constexpr int kJobSplit = (kMaxTok + strata::kernels::cpu::MAXT - 1) / strata::kernels::cpu::MAXT;
constexpr int kMaxEnt = kMaxTok * 8;   ///< (expert, token) entries and distinct experts of one run_multi call
/// File-tier misses one gpu_run_n_ex call runs on the card (FILEG) - kMaxParts per cpu::MAXT tokens, as when a 16-row
/// pass was two calls - and the blobs h_fstage holds (gpu_run uses the first kMaxParts)
constexpr int kMaxFg = kMaxParts * kJobSplit;
constexpr int kMaxDefer = kMaxParts;   ///< STRATA_FILE_DEFER: file-tier experts one layer may defer (d_stage slots)
constexpr size_t kMetaBytes = 65536;   ///< one call's grouped-launch metadata (gpu_run: <= 12 launches x ~200 B; gpu_run_n_ex: <= ~20 x ~1 KB)
constexpr int kMetaSites = 3;   ///< gpu_run's grouped-launch sites: the VRAM hits, the PCIe share, the file misses
constexpr int kMetaClsMax = 9;  ///< DS4 is per_expert = 9 format classes, each getting a fixed metadata slot (slice B)
constexpr int kBatch = 512;    ///< a `ds4routes.bin` ubatch, in tokens (what route_probe tokenises)

[[maybe_unused]] double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[maybe_unused]] int64_t mem_available_gib() {
    const uint64_t b = ds41_sys::mem_available_bytes();   // Linux: /proc/meminfo MemAvailable; Windows: ullAvailPhys
    return b == 0 ? -1 : (int64_t) (b >> 30);
}

/// One metadata integer.  The `deepseek4.*` geometry keys are one-element arrays in the converter's output, but a
/// scalar is accepted too rather than assumed away.
bool meta_i64(const GgufFile& f, const std::string& key, int64_t& out) {
    const MetaValue* v = f.get(key);
    if (!v) return false;
    if (v->type == MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].num();
        return true;
    }
    if (!v->is_num()) return false;
    out = (int64_t) v->num();
    return true;
}

#if defined(DS4_MOE_CUDA)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4_moe: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

/// The five metadata arrays of ONE grouped launch sit in a region of the pinned h_meta block at `b`:
/// [ptr n x u64][st n+1 x i32][ng 1 x i32][dst n x i32][tok n x i32], each part 16-byte aligned.
struct MetaOff { size_t ptr, st, ng, dst, tok, end; };
inline MetaOff meta_off(size_t b, int n) {
    auto al = [](size_t x) { return (x + 15) & ~(size_t) 15; };
    MetaOff m;
    m.ptr = b;
    m.st = al(m.ptr + 8 * (size_t) n);
    m.ng = al(m.st + 4 * (size_t) (n + 1));
    m.dst = al(m.ng + 4);
    m.tok = al(m.dst + 4 * (size_t) n);
    m.end = al(m.tok + 4 * (size_t) n);
    return m;
}

/// The host arena: every expert the budget allows, in the order it was ranked, pinned when asked.  Whatever the
/// budget cannot hold is the FILE TIER and is read from the blob source on demand - the same fallback the replay
/// has, kept because "the code only ever runs with the whole model in RAM" is not a property to rely on.
struct Arena {
    void* base = nullptr;
    uint64_t bytes = 0;
    bool pinned = false;
    bool built = false;
    std::string backing;
    std::unique_ptr<strata::core::PinnedArena> owner;
    void* own_map = nullptr;        ///< STRATA_ARENA_PIN_AFTER: the mapping build_arena made (munmap in close)
    uint64_t own_bytes = 0;
    bool own_reg = false;           ///< ... and it is cudaHostRegister'ed
    std::vector<int32_t> slot_of;   ///< (layer, expert) -> arena index, or -1
    std::vector<uint64_t> off;      ///< arena index -> byte offset (blobs differ in size per layer; used + 1 entries)
    int64_t used = 0, blob = 0, file_tier = 0;
    int64_t n_layers = 0, n_experts = 0;
    bool from_ranked = false;       ///< built from a routing profile (not just index order)
    double seconds = 0.0;           ///< how long the fill took (the startup report wants this number)

    const uint8_t* ptr(int64_t layer, int64_t expert) const {
        if (!base || layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) return nullptr;
        const int32_t s = slot_of[(size_t) (layer * n_experts + expert)];
        return s < 0 ? nullptr : (const uint8_t*) base + (size_t) off[(size_t) s];
    }
    void close() {
        owner.reset();
#ifndef _WIN32
        if (own_map) {
            if (own_reg) cudaHostUnregister(own_map);
            munmap(own_map, (size_t) own_bytes);
        }
#endif
        own_map = nullptr;
        own_bytes = 0;
        own_reg = false;
        base = nullptr;
        bytes = 0;
        built = false;
        from_ranked = false;
        slot_of.clear();
        off.clear();
        used = file_tier = 0;
    }
};

/// A persistent pool for a call's file-tier reads: D3 spawned (and joined) up to 8 std::threads per call - nsys at
/// x0.5 + arena-adapt: pthread_create 1.84 ms/token on the decode thread.  start() hands out task indices 0..n-1 to
/// the workers; wait() returns when every task is done.  One batch at a time: every start() is followed by a wait()
/// before the next start().
struct ReadPool {
    // STRATA_D3_READERS: worker count (default 32).  NVMe queue depth = readers x (each pread's 128 KiB commands):
    // s11 (2026-10-14, PM9A1 Gen4 x4): 8 x 8 parts -> 32 x 32 cut file-tier ms per blob ~17% (code 1925 -> 1606 ms
    // for the same 647 blobs), decode +4-5%; 64 was slower in the O_DIRECT microbench
    static int n_threads() {
        static const int n = std::getenv("STRATA_D3_READERS") ? std::max(1, std::min(64, std::atoi(std::getenv("STRATA_D3_READERS")))) : 32;
        return n;
    }
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv, cv_done;
    std::function<void(size_t)> fn;
    size_t tasks = 0;
    std::atomic<size_t> nx{0};
    int busy = 0;
    uint64_t gen = 0;
    bool stop = false;
    ReadPool() {
        for (int i = 0; i < n_threads(); ++i)
            th.emplace_back([this]() {
                uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return stop || gen != seen; });
                        if (stop) return;
                        seen = gen;
                        ++busy;
                    }
                    for (size_t r; (r = nx.fetch_add(1)) < tasks;) fn(r);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        if (--busy == 0) cv_done.notify_all();
                    }
                }
            });
    }
    void start(size_t n, std::function<void(size_t)> f) {
        std::lock_guard<std::mutex> lk(mu);
        fn = std::move(f);
        tasks = n;
        nx = 0;
        ++gen;
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return busy == 0 && nx.load() >= tasks; });
    }
    ~ReadPool() {
        { std::lock_guard<std::mutex> lk(mu); stop = true; }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
};

/// Decode speculative file-tier reads (Ds41MoeTier::spec_read): the experts the dense half predicts for a layer a few
/// layers ahead that live only on NVMe are read (O_DIRECT, read_part split) into pinned slots by a persistent reader
/// pool while the layers in between compute; gpu_run's file-miss path takes a matching slot instead of reading (waiting
/// for it if the read is still running).  The SSD is otherwise idle in decode, so a wrong guess costs only bandwidth.
/// A slot is tagged with the gpu_run call (`seq`) that will run its layer; it is reusable once that call has passed
/// (its DMA is behind that call's layer sync) and its read is done.
struct SpecStash {
    static constexpr int kSlots = 24, kParts = 4, kThreads = 4;   // 8 readers preempted the decode thread
    struct Slot { int64_t layer = -1, seq = -1; int32_t e = -1; int left = 0; };
    Ds41BlobSource* src = nullptr;
    int64_t stride = 0;
    uint8_t* buf = nullptr;   // kSlots * stride, pinned
    Slot slot[kSlots];
    int next = 0;
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    std::deque<std::pair<int, int>> q;   // (slot, part)
    std::vector<std::thread> th;
    bool stop = false;

    bool open(Ds41BlobSource* s, int64_t b) {
        src = s;
        stride = b;
        if (cudaMallocHost((void**) &buf, (size_t) kSlots * (size_t) stride) != cudaSuccess) {
            (void) cudaGetLastError();
            buf = nullptr;
            return false;
        }
        for (int i = 0; i < kThreads; ++i)
            th.emplace_back([this]() {
                for (;;) {
                    int si, part;
                    int64_t layer;
                    int32_t e;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv_job.wait(lk, [&] { return stop || !q.empty(); });
                        if (stop && q.empty()) return;
                        si = q.front().first;
                        part = q.front().second;
                        q.pop_front();
                        layer = slot[si].layer;
                        e = slot[si].e;
                    }
                    src->read_part(layer, e, part, kParts, buf + (size_t) si * (size_t) stride);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        if (--slot[si].left == 0) cv_done.notify_all();
                    }
                }
            });
        return true;
    }
    /// Queue a read of (layer, e) for gpu_run call `seq`; false when no slot is free (or it is already stashed).
    bool issue(int64_t layer, int32_t e, int64_t seq, int64_t now_seq) {
        std::lock_guard<std::mutex> lk(mu);
        for (const Slot& x : slot)
            if (x.layer == layer && x.e == e && x.seq == seq) return false;
        for (int t = 0; t < kSlots; ++t) {
            const int si = (next + t) % kSlots;
            Slot& x = slot[si];
            if (x.left != 0 || x.seq > now_seq) continue;   // reading, or waiting for its layer
            x.layer = layer;
            x.e = e;
            x.seq = seq;
            x.left = kParts;
            for (int p = 0; p < kParts; ++p) {
                q.emplace_back(si, p);
                cv_job.notify_one();   // one reader per part (notify_all woke every reader for each part)
            }
            next = (si + 1) % kSlots;
            return true;
        }
        return false;
    }
    /// The stashed bytes of (layer, e) for call `seq` (waits for a running read), or null.
    const uint8_t* take(int64_t layer, int32_t e, int64_t seq) {
        std::unique_lock<std::mutex> lk(mu);
        for (int si = 0; si < kSlots; ++si) {
            Slot& x = slot[si];
            if (x.layer != layer || x.e != e || x.seq != seq) continue;
            cv_done.wait(lk, [&] { return x.left == 0; });
            return buf + (size_t) si * (size_t) stride;
        }
        return nullptr;
    }
    ~SpecStash() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            q.clear();
        }
        cv_job.notify_all();
        for (auto& t : th) t.join();
        if (buf) cudaFreeHost(buf);
    }
};

/// STRATA_DS41_REMOTE=<name> (opt-in): the experts a helper process holds on a second GPU (ds41_remote.hpp).
/// gpu_run_n_ex posts the layer's entries for them right after its classify and collects the rows before its sum.
struct Remote {
    Ds41RemoteMap map;
    Ds41RemoteShm* s = nullptr;
    int64_t n = 0;                 ///< the last request number posted
    int64_t calls = 0, entries = 0;
    double wait_ms = 0;            ///< host time spent waiting for the helper after the engine's own work
    double wait_max = 0;           ///< the longest of those waits (both since the last STRATA_DS41_REMOTE_PROF print)
};

/// The prompt-chunk side of STRATA_DS41_REMOTE (segment `<name>_chunk`; STRATA_DS41_REMOTE_CHUNK=0 = off): run_chunk
/// sends the entries of the helper's experts that are not in VRAM, computes the rest itself, and scatters the
/// helper's rows into its parts before the weighted sum.  The file-tier staging skips those experts.
struct RemoteChunk {
    Ds41RemoteMap map;
    Ds41RemoteChunk* c = nullptr;
    const float* out_dev = nullptr;   ///< c->out() as the card sees it (mapped pinned memory)
    int64_t n = 0;                    ///< the last request number posted
    bool pending = false;             ///< a request of the current run_chunk is out
    int nr = 0;                       ///< its entries
    std::vector<char> mask_e;         ///< the current layer: expert -> the helper computes it in this call
    std::vector<int32_t> rrow;        ///< entry i of the request -> parts row
    int32_t* d_rrow = nullptr;
    int64_t d_rrow_cap = 0;
    int64_t calls = 0, entries = 0;
    double wait_ms = 0;
};

/// All the device state, so the CPU-only flavour has none of it to leak.
struct Gpu {
    Arena arena;
    std::unique_ptr<Remote> remote;    ///< STRATA_DS41_REMOTE (null = off)
    std::unique_ptr<RemoteChunk> rchunk;   ///< its prompt-chunk segment (null = off)
    std::unique_ptr<SpecStash> spec;   ///< spec_read's stash (opened on first use)
    std::unique_ptr<ReadPool> rpool;   ///< D3's file-tier readers (persistent; created on first use)
    // STRATA_FILE_DEFER (gpu_run_n_ex): a layer's deferred file-tier reads and q8_1 rows, by layer parity (layer l's
    // are consumed by layer l+1's call while layer l+1 defers into the other half); created on first use
    std::unique_ptr<ReadPool> rpool_def[2];
    void* d_xq_def[2] = {nullptr, nullptr};
    uint8_t* h_def[2] = {nullptr, nullptr};   ///< pinned staging for a deferred blob without an arena_adapt victim
    void* d_def_parts = nullptr;              ///< per token: K deferred rows (by routing rank) + the layer's own sum
    float* d_def_w = nullptr;
    float* h_def_w = nullptr;
    std::unique_ptr<strata::core::ExpertCache> cache;
    strata::kernels::NativeExpertLayout gl;
    std::vector<strata::kernels::NativeExpertLayout> gll;   ///< per model layer (all == gl for a uniform geometry)
    cudaStream_t s = nullptr;
    void *d_meta = nullptr;            ///< run(): grouped-launch metadata (device side of h_meta)
    uint8_t* h_meta = nullptr;         ///< pinned staging for it
    uint8_t* h_fstage = nullptr;       ///< run(): file-tier misses read here (pinned), then DMA'd (kMaxFg blobs)
    void *d_x = nullptr, *d_xq = nullptr, *d_parts = nullptr, *d_grp_ptr = nullptr, *d_grp_start = nullptr,
         *d_ngroups = nullptr, *d_ent_dst = nullptr, *d_ent_tok = nullptr, *d_scratch = nullptr, *d_stage = nullptr;
    cudaEvent_t ev[4] = {};
    void* d_pf = nullptr;         ///< kMaxPf staging slots, one expert blob each
    cudaStream_t s_pf = nullptr;  ///< the prefetch stream (never the main one: the DMAs run under the dense work)
    cudaEvent_t ev_pf = nullptr;
    cudaStream_t s_dc = nullptr;  ///< STRATA_DMA_OVERLAP: the demand-copy stream.  A layer's PCIe/file blob DMAs run
                                  ///< here, so they overlap that layer's resident-expert kernels on `s` instead of
                                  ///< queueing behind them; `s` waits on ev_dc before the miss kernels are launched.
    cudaEvent_t ev_dc = nullptr;
    /// STRATA_D3_DMA_STREAM=1: gpu_run_n_ex's own copy stream for its PCIe site (the arena-miss / FILEG H2D into the
    /// d_stage slots or an LRU victim slot, the LRU demote D2H and the stage->victim D2D), so those copies overlap the
    /// call's resident-expert kernels and each wave's kernel instead of queueing behind them on `s`.  The slots are
    /// split into two banks of `d3_bank` slots; waves alternate banks.  Events: dn_ev[0] = the cross-call barrier (`s`
    /// -> s_dn, once per call), dn_ev[1] = "this wave's copies are home" (s_dn -> `s`, before each wave's launch),
    /// dn_ev[2 + b] = "the kernels that read bank b are done" (`s` -> s_dn, before the bank is written again).
    /// Off (s_dn null) = the single-stream path, unchanged.  Not s_dc: that is gpu_run's opt-in overlap.
    cudaStream_t s_dn = nullptr;
    cudaEvent_t dn_ev[4] = {};
    int n_stage = 0;    ///< blobs d_stage holds: max(kMaxParts, STRATA_D3_STAGE_SLOTS); gpu_run uses the first kMaxParts
    int d3_bank = 0;    ///< s_dn on: slots per bank, min(kMaxParts, STRATA_D3_STAGE_SLOTS / 2)
    uint8_t* h_dummy = nullptr;   ///< pinned zero blob: a guess outside the arena still costs its DMA
    float* h_x = nullptr;         ///< pinned host copy of x (the H2D source and the CPU tier's input)
    void* h_parts = nullptr;      ///< pinned D2H landing pad for the expert outputs
    /// STRATA_TIER_GRAPH (slice B): each of gpu_run's kMetaSites sites gets ONE fixed metadata block - kMetaClsMax
    /// slots, one per format class, sized for kMaxParts entries - and every (site, class) launch is captured once
    /// into a CUDA graph of its own.  native_expert_grouped reads the group/entry counts from DEVICE memory and its
    /// grid is pinned by cap_groups/cap_entries with grid_groups=0, so the nodes' parameters are constant and only
    /// the block's CONTENTS change per token: per site ONE cudaMemcpyAsync + one cudaGraphLaunch per class the
    /// routing actually produced, in place of that class's three kernel launches and their cudaGetLastError.
    /// One exec per class (not one per site) is deliberate: the class set varies per layer, and a graph is a fixed
    /// set of kernels - a single 9-class graph would dispatch nine classes' gate/up+SwiGLU+down every layer where
    /// the routing needs ~2.7, which measured 3-4 ms/token of pure dispatch (see the report).
    cudaGraphExec_t tier_g[kMetaSites * kMetaClsMax] = {};
    bool tier_graph_ready = false;   ///< the captures succeeded: the fixed layout is in use
    bool tier_graph_off = false;     ///< capture refused (or does not apply): the direct per-class launches stay
    int tier_graph_cls = 0;          ///< classes = gp.gll.size() at capture (DS4 per_expert: 9)
    size_t tier_site_span = 0;       ///< bytes of one site's fixed metadata block
    size_t tier_w_off = 0;           ///< the out_dev sum weights, after the sites (never inside a captured block)

    /// run_chunk (prompt chunks, MiMo prefill): allocated on first use, for up to `c_tok` tokens.  The streamed
    /// experts go through two device halves of `c_half` bytes (copy stream `s_cp` fills one while `s` computes
    /// the other); file-tier blobs are read into the matching pinned bounce half first.
    int64_t c_tok = 0, c_half = 0;
    int64_t c_bhalf = 0;          ///< bytes of one pinned bounce half (file-tier reads); c_half may be larger (prestage)
    /// The cudaMallocHost'd landing pads (c_hparts, c_bounce) cost 3.4 s of cudaMallocHost per prompt (nsys) and are
    /// kept across prompts: `c_pin_tok` is the token capacity they hold, and release_chunk leaves them alone.  The
    /// DEVICE buffers (c_tok = their token capacity, c_half = the device halves) still go back, so the deferred VRAM
    /// seed has the memory.  `c_ready` marks the scaffolding (streams, events, device halves, bounce) as set up.
    int64_t c_pin_tok = 0;
    bool c_ready = false;
    /// Ds41MoeConfig::chunk_prestage: layer `c_pre_layer`'s arena experts `c_pre_e` (ascending ids, not VRAM-resident)
    /// are DMA'd / being DMA'd into c_stage[c_pre_half] at stride `c_pre_sl`, event c_copied[c_pre_half]
    int c_pre_layer = -1, c_pre_half = -1;
    int64_t c_pre_sl = 0;
    std::vector<int32_t> c_pre_e;
    /// Layer prestage (any geometry, per-expert formats too): while the caller runs layer `p_layer`'s dense half, its
    /// arena experts (not VRAM-resident) are DMA'd on s_cp into one device region `p_buf` (each at its own offset, its
    /// class's blob size); run_chunk launches them from there and streams only the rest.  STRATA_PRESTAGE_GIB (default
    /// 3, 0 = off) caps the region; freed by release_chunk (the deferred VRAM seed wants the memory).
    void* p_buf = nullptr;
    size_t p_cap = 0;
    int p_layer = -1;
    /// The layer an eager prestage WOULD have filled, when a narrow call skipped it (STRATA_PRESTAGE_MIN): exactly that
    /// layer - no other - gets the lazy fill, so the set of region-served layers (and so every kernel) is production's.
    /// Cleared wherever p_layer is reset by a release.
    int p_lazy_layer = -1;
    bool p_off = false, p_used_rec = false;
    std::vector<unsigned long long> p_ptr;
    cudaEvent_t p_copied = nullptr, p_used = nullptr;
    void *c_w = nullptr, *c_out = nullptr;   ///< run_chunk: the top-k weights and the summed rows on the card
    void *c_x = nullptr, *c_xq = nullptr, *c_parts = nullptr, *c_scratch = nullptr, *c_ptr = nullptr,
         *c_start = nullptr, *c_ng = nullptr, *c_dst = nullptr, *c_tokv = nullptr, *c_stage[2] = {};
    uint8_t* c_bounce[2] = {};
    float* c_hparts = nullptr;
    cudaStream_t s_cp = nullptr;
    cudaEvent_t c_copied[2] = {}, c_used[2] = {};
    /// run_chunk's MMQ products (Ds41MoeConfig::chunk_mmq): up to `m_rows` sorted entries per launch.  `m_ok[l]`:
    /// layer l's types are covered (-1 = not checked yet).
    int64_t m_rows = 0;
    void *m_gu = nullptr, *m_h = nullptr, *m_xg = nullptr, *m_xu = nullptr, *m_xd = nullptr, *m_bnd = nullptr,
         *m_iota = nullptr;
    std::vector<int8_t> m_ok;
    /// run_chunk's MMQ on PTQ1_0: the run's ternary matrices transcoded to Q2_0 here (ds41_transcode, lossless)
    void* m_tc = nullptr;
    size_t m_tc_cap = 0;
#if defined(DS4_MOE_MMQ)
    /// Created once and never destroyed: a binary that also links ggml-cuda (mimo_generate's dense backend) resolves
    /// ~ggml_backend_cuda_context to ggml-cuda's real one, which segfaults on the context strata_mmq built.
    strata::prefill::mmq::Context* m_ctx = nullptr;
#endif
    void free_mmq() {
        for (void* p : {m_gu, m_h, m_xg, m_xu, m_xd, m_bnd, m_iota, m_tc})
            if (p) cudaFree(p);
        m_gu = m_h = m_xg = m_xu = m_xd = m_bnd = m_iota = m_tc = nullptr;
        m_rows = 0;
        m_tc_cap = 0;
    }

    void close() {
        for (auto& e : ev)
            if (e) cudaEventDestroy(e);
        for (cudaGraphExec_t g : tier_g)
            if (g) cudaGraphExecDestroy(g);
        void* bufs[] = {d_x, d_xq, d_parts, d_grp_ptr, d_grp_start, d_ngroups, d_ent_dst, d_ent_tok, d_scratch,
                        d_stage, d_pf, d_meta};
        if (h_meta) cudaFreeHost(h_meta);
        if (h_fstage) cudaFreeHost(h_fstage);
        for (void* p : bufs)
            if (p) cudaFree(p);
        if (ev_pf) cudaEventDestroy(ev_pf);
        if (s_pf) cudaStreamDestroy(s_pf);
        if (ev_dc) cudaEventDestroy(ev_dc);
        if (s_dc) cudaStreamDestroy(s_dc);
        for (cudaEvent_t& e : dn_ev)
            if (e) cudaEventDestroy(e);
        if (s_dn) cudaStreamDestroy(s_dn);
        if (s) cudaStreamDestroy(s);
        if (h_dummy) cudaFreeHost(h_dummy);
        if (h_x) cudaFreeHost(h_x);
        if (h_parts) cudaFreeHost(h_parts);
        for (void* p : {c_x, c_xq, c_parts, c_scratch, c_ptr, c_start, c_ng, c_dst, c_tokv, c_stage[0], c_stage[1], c_w, c_out})
            if (p) cudaFree(p);
        for (uint8_t* p : c_bounce)
            if (p) cudaFreeHost(p);
        if (c_hparts) cudaFreeHost(c_hparts);
        for (int i = 0; i < 2; ++i) {
            if (c_copied[i]) cudaEventDestroy(c_copied[i]);
            if (c_used[i]) cudaEventDestroy(c_used[i]);
        }
        if (s_cp) cudaStreamDestroy(s_cp);
        free_mmq();
        spec.reset();
        cache.reset();
        arena.close();
        *this = Gpu();
    }
};

/// Ds41MoeConfig::remote_device: a second card that holds a STATIC set of experts and computes them itself (llm-server:
/// a V100 32 GB beside the 3090, behind a gen3 x4 link with no P2P - too slow to feed the main card blobs, fast enough
/// for a layer's activations in and its summed rows out).  The ranking right after the main card's VRAM head goes
/// here and stays out of the arena, so the host RAM holds the next experts instead.  Every tier call takes the remote
/// entries out (remote_begin: they point at a non-remote expert with weight 0), a worker thread bound to the card
/// computes them and sums them per token, and remote_end adds those rows to the call's output.
struct RemoteTier {
    static constexpr int kRB = 2048;     ///< tokens per block on the card (a prompt batch runs in blocks)
    static constexpr size_t kRMeta = 1 << 20;
    int dev = -1;
    uint8_t* base = nullptr;             ///< the experts, on `dev`
    size_t cap = 0, used = 0;
    std::vector<int64_t> off;            ///< (layer * n_experts + e) -> byte offset in base; -1 = not held
    static constexpr int kProxy = 8;
    std::vector<int32_t> proxy;          ///< per layer, kProxy: the best-ranked non-remote experts (substitutes of last resort)
    int64_t n = 0;
    cudaStream_t s = nullptr;
    void *d_x = nullptr, *d_xq = nullptr, *d_parts = nullptr, *d_scratch = nullptr, *d_meta = nullptr, *d_w = nullptr,
         *d_out = nullptr;
    uint8_t* h_meta = nullptr;
    float* h_w = nullptr;
    float* h_out = nullptr;              ///< pinned: the call's remote rows, summed per token (n x H)
    int64_t h_out_tok = 0;
    float* h_xd = nullptr;               ///< pinned: run_dev's input copied off the main card
    void* g_add = nullptr;               ///< on the main card: h_out's rows for an out_dev call
    int64_t g_add_tok = 0;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false, pending = false, ok = true;
    int64_t j_layer = 0;
    int j_n = 0;
    std::vector<int32_t> j_ids;
    std::vector<float> j_w;
    const float* j_x = nullptr;
    int64_t calls = 0, entries = 0;
    double wait_ms = 0;
    /// profile window (MIMO_CHUNK_PROF / DS41_PROF_X prints and resets it): the card's busy time per job (worker side,
    /// H2D + kernels + D2H + sync) and the caller's wait for it
    int64_t p_calls = 0, p_entries = 0;
    double p_busy = 0, p_wait = 0;
    bool held(int64_t layer, int64_t e, int64_t nx) const {
        return !off.empty() && off[(size_t) (layer * nx + e)] >= 0;
    }
    void close() {
        if (th.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            th.join();
        }
        if (dev >= 0) {
            int prev = 0;
            cudaGetDevice(&prev);
            cudaSetDevice(dev);
            for (void* p : {(void*) base, d_x, d_xq, d_parts, d_scratch, d_meta, d_w, d_out})
                if (p) cudaFree(p);
            if (s) cudaStreamDestroy(s);
            cudaSetDevice(prev);
        }
        if (g_add) cudaFree(g_add);
        for (void* p : {(void*) h_meta, (void*) h_w, (void*) h_out, (void*) h_xd})
            if (p) cudaFreeHost(p);
    }
};

#else

/// No CUDA in this flavour; the type exists only so `Ds41MoeImpl` compiles unchanged.
struct Gpu {
    void close() {}
};
struct RemoteTier {
    bool held(int64_t, int64_t, int64_t) const { return false; }
    void close() {}
};

#endif  // DS4_MOE_CUDA

/// Every (layer, expert) by descending routing frequency over the TRAIN half of a `ds4routes.bin` (alternating
/// 512-token blocks - the split `tools/ds4/route_skew.py` scores with).  The file's layout, verified on the real
/// one (3,170,304 records = 12,288 tokens x 43 x 6): for each 512-token ubatch, layer 0's whole batch (512 tokens
/// x 6 experts, token-major), then layer 1's, ...
std::vector<std::pair<int32_t, int32_t>> rank_routes(const std::string& path, int64_t n_layers, int64_t top_k,
                                                     std::string& err, int64_t batch = kBatch) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "cannot open " + path;
        return {};
    }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    std::vector<uint16_t> all((size_t) (bytes / 2));
    in.read((char*) all.data(), bytes);
    if (!in) {
        err = "short read on " + path;
        return {};
    }
    const int64_t records = (int64_t) all.size() / 2;
    const int64_t per_token = n_layers * top_k;
    if (records == 0 || records % per_token != 0) {
        err = "not a whole number of tokens (records " + std::to_string(records) + ", per token " +
              std::to_string(per_token) + ")";
        return {};
    }
    // `n_layers` here = layers IN THE FILE (the routed ones); `batch` = its ubatch in tokens.  A final short ubatch
    // (a file's tail) is laid out the same way with fewer tokens; it only shifts which half its tokens count in.
    const int64_t per_batch = batch * per_token;    // records in one ubatch
    std::map<std::pair<int32_t, int32_t>, int64_t> cnt;
    for (int64_t i = 0; i + 1 < (int64_t) all.size(); i += 2) {
        const int64_t rec = i / 2;
        const int64_t token = (rec / per_batch) * batch + (rec % (batch * top_k)) / top_k;
        if ((token / kBatch) % 2 != 0) continue;    // the held-out half (alternate 512-token blocks, as route_skew)
        cnt[{all[(size_t) i], all[(size_t) i + 1]}]++;
    }
    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve(cnt.size());
    for (auto& le : cnt) ranked.push_back(le.first);
    std::sort(ranked.begin(), ranked.end(), [&](const auto& x, const auto& y) {
        const int64_t cx = cnt[x], cy = cnt[y];
        return cx != cy ? cx > cy : x < y;
    });
    return ranked;
}

}  // namespace

// ================================ Ds41MoeImpl ================================

struct Rebal;        ///< STRATA_ARENA_REBAL: the conversation-adaptive arena (defined with rebalance_*; null in a CPU build)
#if defined(DS4_MOE_CUDA)
struct ChunkStage;   ///< chunk_prefetch's two pinned slabs (defined below, after `acquire`)
#endif

/// STRATA_FILE_DEFER: the file-tier experts one gpu_run_n_ex call deferred, for the next call (layer + 1) to add
struct FileDefer {
    bool on = false;
    int64_t layer = -1;
    int nt = 0, par = 0, n = 0, nent = 0;
    int32_t e[kMaxDefer], c[kMaxDefer], vic[kMaxDefer];
    const uint8_t* src[kMaxDefer];
    int32_t ent_j[kMaxEnt], ent_d[kMaxEnt];   ///< the deferred entries: routing index (t * K + k) and expert
    float w[kMaxEnt];
};

struct Ds41MoeImpl {
    FileDefer fdef;                   ///< STRATA_FILE_DEFER: the last call's deferred experts (gpu_run_n_ex)
    int64_t fdef_layers = 0, fdef_experts = 0, fdef_dropped = 0;
    Ds41MoeGeom g;
    Ds41MoeConfig cfg;
    Ds41BlobSource* blobs = nullptr;
    std::unique_ptr<Ds41BlobSource> owned_blobs;
    cpu::NativeFmt f;                 ///< the largest routed layer's (buffer sizing; the uniform geometry's only one)
    int64_t blob = 0;                 ///< the largest blob (buffer strides)
    std::vector<cpu::NativeFmt> fl;   ///< per MODEL layer; fl[l].bytes == 0 for a layer without routed experts
    std::vector<int64_t> bl;          ///< per layer blob bytes
    bool mixed_sizes = false;         ///< blob size differs between routed layers (MiMo): cache opened at seed
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<uint8_t> nact;        ///< the Q8_K activation the CPU kernel consumes
    cpu::ActQ actq;                   ///< the same for a Q2_0 class on Strata's own Q2_0 kernels (q2_native_kernels)
    std::vector<cpu::ActQ> actq_n;    ///< gpu_run_n_ex: one per pass token, for its Q2_0-class CPU jobs
    std::vector<uint8_t> nact_n;      ///< run_multi: kMaxTok of them, f.act_bytes apart
    /// One input-activation buffer per DISTINCT gate/up activation format among the classes (kNativeActBytes
    /// each, `nact_multi_act[b]` its `gu_act`), so the one-dispatch-per-layer path quantizes the layer's
    /// activation once per format instead of once per class.  The Q2_0-native classes share `actq`.
    std::vector<uint8_t> nact_multi;
    std::vector<int> nact_multi_act;
    std::vector<uint8_t> nact_multi_n;   ///< run_multi per-expert: kMaxTok x nact_multi_act x kNativeActBytes
    std::vector<float> parts_n;       ///< run_multi: the pool's outputs, one row per (token, k) entry
    std::vector<float> parts;         ///< top_k * n_embd, one expert output per routing index
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<uint8_t> ftmp;        ///< the file tier's read buffer, top_k blobs
    Ds41MoeStats st;
    std::vector<double> sal_sum;      ///< REAP: sum over routed tokens of w * ||expert output|| (cfg.saliency)
    std::vector<int64_t> sal_cnt;
    std::vector<float> sal_parts;
    bool inited = false;
    int64_t admitted = 0;
    double pcie_carry = 0.0, pf_carry = 0.0;
    int32_t pf_e[Ds41MoeConfig::kMaxPf] = {};
    int pf_n = 0;                     ///< staging slots the last prefetch() filled
    std::unique_ptr<Gpu> gpu;         ///< null in a CPU-only tier
    std::unique_ptr<RemoteTier> remote;   ///< Ds41MoeConfig::remote_device (null = off)
    std::vector<int64_t> last_use;    ///< arena_adapt: (layer, expert) -> the run() call that last used it
    int64_t tick = 0;
    int64_t run_seq = 0;              ///< gpu_run calls (spec_read tags its slots with the call that will use them)
    int64_t run_layer = -1;           ///< the layer of the last gpu_run call
    std::vector<int64_t> vlast;       ///< vram_lru: (layer, expert) -> the run() call that last used its VRAM copy
    std::vector<int64_t> vmiss;       ///< vram_lru: (layer, expert) -> the layer's call count at its last miss
    std::vector<int64_t> lcalls;      ///< vram_lru: run() calls per layer (= decode tokens)
    FILE* trace = nullptr;            ///< trace_open: the session-13 routing trace (measurement only)
    /// warm_reseed: (layer, expert) that held a VRAM slot when the cache was last released, most recently used
    /// first.  A seed consumes it: those experts fill the first STRATA_WARM_SHARE (default 500) per mille of the
    /// slots, the global rank the rest.  Empty = off.
    std::vector<std::pair<int32_t, int32_t>> warm;
#if defined(DS4_MOE_CUDA)
    std::unique_ptr<ChunkStage> cstage;   ///< chunk_prefetch: the two pinned slabs
#endif
    std::unique_ptr<Rebal> rebal;         ///< STRATA_ARENA_REBAL (null = off)
    std::vector<std::array<int32_t, 3>> shrunk;   ///< shrink_cache_for: (slot, layer, expert) evicted, for regrow_cache
};

void rebal_count(Ds41MoeImpl& im, int64_t layer, const int32_t* ids, int64_t n);   // defined with rebalance_*

// ---- the remote card (Ds41MoeConfig::remote_device) ---------------------------------------------------------------

/// `ranked` minus the experts the remote card holds (they take no VRAM slot and no arena space)
std::vector<std::pair<int32_t, int32_t>> drop_remote(const Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    if (!im.remote) return ranked;
    std::vector<std::pair<int32_t, int32_t>> kept;
    kept.reserve(ranked.size());
    for (const auto& le : ranked)
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts ||
            !im.remote->held(le.first, le.second, im.g.n_experts))
            kept.push_back(le);
    return kept;
}

#if defined(DS4_MOE_CUDA)
/// The head of `ranked` the VRAM cache takes (arena_order's walk): ranked order, up to `slots` experts or `slot_gib`
/// bytes.  The remote card takes the ranking right after it.
std::vector<char> vram_head(const Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    std::vector<char> res((size_t) (im.g.n_layers * im.g.n_experts), 0);
    if (im.cfg.no_cache || im.cfg.slots <= 0) return res;
    const double budget = im.cfg.slot_gib > 0 ? im.cfg.slot_gib * 1073741824.0 : (double) im.cfg.slots * (double) im.blob;
    double used = 0;
    int64_t n = 0;
    for (const auto& le : ranked) {
        if (n >= im.cfg.slots) break;
        if (!im.g.routed(le.first) || le.second < 0 || le.second >= im.g.n_experts) continue;
        char& r = res[(size_t) (le.first * im.g.n_experts + le.second)];
        if (r) continue;
        const int64_t b = im.bl[(size_t) im.g.cls(le.first, le.second)];
        if (used + (double) b > budget) break;
        r = 1;
        used += (double) b;
        ++n;
    }
    return res;
}

/// One job on the remote card (its worker thread, bound to the card): the job's remote entries, in blocks of kRB
/// tokens - activations up, one grouped launch per format class, the weighted sum per token (the other entries weigh
/// 0 and their rows are zero), the summed rows back to h_out.
bool remote_compute(Ds41MoeImpl& im, RemoteTier& r) {
    const int64_t H = im.g.n_embd, K = im.g.top_k, NX = im.g.n_experts, layer = r.j_layer;
    const int n = r.j_n;
    const size_t col = strata::kernels::native_q8_1_bytes((int) H, 1);
    auto al16 = [](size_t v) { return (v + 15) & ~(size_t) 15; };
    std::vector<int32_t> ents;
    for (int t0 = 0; t0 < n; t0 += RemoteTier::kRB) {
        const int nb = std::min(RemoteTier::kRB, n - t0);
        const int64_t NEb = (int64_t) nb * K;
        const int32_t* ids = r.j_ids.data() + (size_t) t0 * (size_t) K;
        const float* w = r.j_w.data() + (size_t) t0 * (size_t) K;
        float* hout = r.h_out + (size_t) t0 * (size_t) H;
        ents.clear();
        for (int64_t j = 0; j < NEb; ++j)
            if (r.held(layer, ids[j], NX)) ents.push_back((int32_t) j);
        if (ents.empty()) {
            std::memset(hout, 0, (size_t) nb * (size_t) H * 4);
            continue;
        }
        std::sort(ents.begin(), ents.end(), [&](int32_t a, int32_t b) {
            const int ca = im.g.cls(layer, ids[a]), cb = im.g.cls(layer, ids[b]);
            if (ca != cb) return ca < cb;
            if (ids[a] != ids[b]) return ids[a] < ids[b];
            return a < b;
        });
        for (int64_t j = 0; j < NEb; ++j) r.h_w[j] = 0.0f;
        for (int32_t j : ents) r.h_w[j] = w[j];
        if (cudaMemcpyAsync(r.d_x, r.j_x + (size_t) t0 * (size_t) H, (size_t) nb * (size_t) H * 4, cudaMemcpyHostToDevice, r.s) != cudaSuccess) return false;
        for (int c0 = 0; c0 < nb; c0 += 8)
            strata::kernels::native_quantize_q8_1((const float*) r.d_x + (size_t) c0 * (size_t) H, (uint8_t*) r.d_xq + (size_t) c0 * col,
                                                  (int) H, std::min(8, nb - c0), r.s);
        if (cudaMemsetAsync(r.d_parts, 0, (size_t) NEb * (size_t) H * 4, r.s) != cudaSuccess) return false;
        if (cudaMemcpyAsync(r.d_w, r.h_w, (size_t) NEb * 4, cudaMemcpyHostToDevice, r.s) != cudaSuccess) return false;
        size_t at = 0;
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start, dst, tok;
        for (size_t i = 0; i < ents.size();) {
            const int c = im.g.cls(layer, ids[ents[i]]);
            ptr.clear(); start.clear(); dst.clear(); tok.clear();
            size_t i1 = i;
            for (; i1 < ents.size() && im.g.cls(layer, ids[ents[i1]]) == c; ++i1) {
                const int32_t e = ids[ents[i1]];
                if (i1 == i || ids[ents[i1 - 1]] != e) {
                    ptr.push_back((unsigned long long) (r.base + r.off[(size_t) (layer * NX + e)]));
                    start.push_back((int32_t) dst.size());
                }
                dst.push_back(ents[i1]);
                tok.push_back(ents[i1] / (int32_t) K);
            }
            start.push_back((int32_t) dst.size());
            const int ng = (int) ptr.size();
            const size_t ne = dst.size();
            const size_t o_ptr = at, o_st = al16(o_ptr + 8 * (size_t) ng), o_ng = al16(o_st + 4 * (size_t) (ng + 1)),
                         o_dst = al16(o_ng + 4), o_tok = al16(o_dst + 4 * ne), o_end = al16(o_tok + 4 * ne);
            if (o_end > RemoteTier::kRMeta) {
                std::fprintf(stderr, "ds41_moe: remote: %zu bytes of launch metadata (> %zu)\n", o_end, RemoteTier::kRMeta);
                return false;
            }
            const int32_t ngv = ng;
            std::memcpy(r.h_meta + o_ptr, ptr.data(), 8 * (size_t) ng);
            std::memcpy(r.h_meta + o_st, start.data(), 4 * (size_t) (ng + 1));
            std::memcpy(r.h_meta + o_ng, &ngv, 4);
            std::memcpy(r.h_meta + o_dst, dst.data(), 4 * ne);
            std::memcpy(r.h_meta + o_tok, tok.data(), 4 * ne);
            if (cudaMemcpyAsync((uint8_t*) r.d_meta + o_ptr, r.h_meta + o_ptr, o_end - o_ptr, cudaMemcpyHostToDevice, r.s) != cudaSuccess)
                return false;
            uint8_t* dm = (uint8_t*) r.d_meta;
            strata::kernels::native_expert_grouped(im.gpu->gll[(size_t) c], (const unsigned long long*) (dm + o_ptr),
                                                  (const int32_t*) (dm + o_st), (const int32_t*) (dm + o_ng),
                                                  (const int32_t*) (dm + o_dst), (const int32_t*) (dm + o_tok), ng,
                                                  (int64_t) ne, r.d_xq, r.d_scratch, (float*) r.d_parts, r.s, ng);
            at = o_end;
            i = i1;
        }
        strata::kernels::weighted_rows_sum((const float*) r.d_parts, (const float*) r.d_w, (int) K, H, nb, (float*) r.d_out, r.s);
        if (cudaMemcpyAsync(hout, r.d_out, (size_t) nb * (size_t) H * 4, cudaMemcpyDeviceToHost, r.s) != cudaSuccess) return false;
        // h_meta / h_w are reused by the next block: its copies must have run
        const cudaError_t e = cudaStreamSynchronize(r.s);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "ds41_moe: remote: %s\n", cudaGetErrorString(e));
            return false;
        }
    }
    return true;
}

void remote_worker(Ds41MoeImpl& im, RemoteTier& r) {
    cudaSetDevice(r.dev);
    for (;;) {
        std::unique_lock<std::mutex> lk(r.mu);
        r.cv.wait(lk, [&] { return r.stop || r.pending; });
        if (r.stop) return;
        lk.unlock();
        const double t0 = now_ms();
        const bool ok = remote_compute(im, r);
        const double busy = now_ms() - t0;
        lk.lock();
        r.p_busy += busy;
        r.ok = ok;
        r.pending = false;
        lk.unlock();
        r.cv.notify_all();
    }
}

/// Opens the remote card once (the first seed): its share of `ranked` (after the VRAM head, up to its budget) is
/// read from the blob source and uploaded, and its worker starts.  The arena and the cache seeds drop those experts.
bool remote_build(Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (im.cfg.remote_device < 0 || im.remote || !im.gpu || im.cfg.cpu_only) return true;
    const double t0 = now_ms();
    std::unique_ptr<RemoteTier> rp(new RemoteTier());
    RemoteTier& r = *rp;
    int prev = 0, ndev = 0;
    ck(cudaGetDevice(&prev), "remote: current device");
    ck(cudaGetDeviceCount(&ndev), "remote: device count");
    if (im.cfg.remote_device == prev || im.cfg.remote_device >= ndev) {
        err = "remote device " + std::to_string(im.cfg.remote_device) + " is not a second card (" + std::to_string(ndev) +
              " visible, the tier runs on " + std::to_string(prev) + ")";
        return false;
    }
    r.dev = im.cfg.remote_device;
    ck(cudaSetDevice(r.dev), "remote: set device");
    struct Back { int d; ~Back() { cudaSetDevice(d); } } back{prev};
    const int64_t H = im.g.n_embd, K = im.g.top_k, NL = im.g.n_layers, NX = im.g.n_experts;
    const int64_t NEB = (int64_t) RemoteTier::kRB * K;
    ck(cudaStreamCreateWithFlags(&r.s, cudaStreamNonBlocking), "remote: stream");
    ck(cudaMalloc(&r.d_x, (size_t) RemoteTier::kRB * (size_t) H * 4), "remote: d_x");
    ck(cudaMalloc(&r.d_xq, (size_t) RemoteTier::kRB * strata::kernels::native_q8_1_bytes((int) H, 1)), "remote: d_xq");
    ck(cudaMalloc(&r.d_parts, (size_t) NEB * (size_t) H * 4), "remote: d_parts");
    ck(cudaMalloc(&r.d_scratch, strata::kernels::native_expert_scratch_bytes(NEB, im.g.n_ff)), "remote: d_scratch");
    ck(cudaMalloc(&r.d_meta, RemoteTier::kRMeta), "remote: d_meta");
    ck(cudaMalloc(&r.d_w, (size_t) NEB * 4), "remote: d_w");
    ck(cudaMalloc(&r.d_out, (size_t) RemoteTier::kRB * (size_t) H * 4), "remote: d_out");
    ck(cudaHostAlloc((void**) &r.h_meta, RemoteTier::kRMeta, cudaHostAllocPortable), "remote: h_meta");
    ck(cudaHostAlloc((void**) &r.h_w, (size_t) NEB * 4, cudaHostAllocPortable), "remote: h_w");
    size_t fr = 0, tot = 0;
    ck(cudaMemGetInfo(&fr, &tot), "remote: mem info");
    double want = im.cfg.remote_gib > 0 ? im.cfg.remote_gib * 1073741824.0 : (double) fr - im.cfg.remote_reserve_gib * 1073741824.0;
    want = std::min(want, (double) fr - 256.0 * 1048576.0);
    if (want < (double) im.blob) {
        err = "remote device has " + std::to_string(fr / 1048576) + " MiB free - no room for experts";
        return false;
    }
    r.cap = (size_t) want & ~(size_t) 255;
    while (cudaMalloc((void**) &r.base, r.cap) != cudaSuccess) {   // fragmentation: give back 256 MiB a try
        (void) cudaGetLastError();
        if (r.cap < (size_t) im.blob + (256u << 20)) {
            err = "remote device: cannot allocate the expert region";
            return false;
        }
        r.cap -= 256u << 20;
    }
    // the ranking after the VRAM head, in order, while it fits
    const std::vector<char> head = vram_head(im, ranked);
    r.off.assign((size_t) (NL * NX), -1);
    std::vector<std::array<int64_t, 3>> jobs;   // layer, expert, offset
    for (const auto& le : ranked) {
        if (le.first < 0 || le.first >= NL || le.second < 0 || le.second >= NX || !im.g.routed(le.first)) continue;
        const size_t at = (size_t) (le.first * NX + le.second);
        if (head[at] || r.off[at] >= 0) continue;
        const size_t b = ((size_t) im.bl[(size_t) im.g.cls(le.first, le.second)] + 255) & ~(size_t) 255;
        if (r.used + b > r.cap) continue;   // a smaller class may still fit
        r.off[at] = (int64_t) r.used;
        r.used += b;
        jobs.push_back({le.first, le.second, r.off[at]});
    }
    r.n = (int64_t) jobs.size();
    // substitutes of last resort: per layer the kProxy best-ranked experts the card does not hold (then any)
    const int KP = RemoteTier::kProxy;
    r.proxy.assign((size_t) (NL * KP), -1);
    std::vector<int> np((size_t) NL, 0);
    auto add_proxy = [&](int64_t l, int64_t e) {
        if (np[(size_t) l] >= KP || r.off[(size_t) (l * NX + e)] >= 0) return;
        for (int i = 0; i < np[(size_t) l]; ++i)
            if (r.proxy[(size_t) (l * KP + i)] == e) return;
        r.proxy[(size_t) (l * KP + np[(size_t) l]++)] = (int32_t) e;
    };
    for (const auto& le : ranked)
        if (le.first >= 0 && le.first < NL && le.second >= 0 && le.second < NX) add_proxy(le.first, le.second);
    for (int64_t l = 0; l < NL; ++l)
        for (int64_t e = 0; e < NX; ++e) add_proxy(l, e);
    // upload: 8 readers, each with its own pinned bounce
    std::atomic<size_t> nx{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> th;
    for (int k = 0; k < 8; ++k)
        th.emplace_back([&]() {
            cudaSetDevice(r.dev);
            uint8_t* buf = nullptr;
            if (cudaHostAlloc((void**) &buf, (size_t) im.blob, cudaHostAllocPortable) != cudaSuccess) { bad = true; return; }
            for (size_t q; !bad && (q = nx.fetch_add(1)) < jobs.size();) {
                const auto& j = jobs[q];
                im.blobs->read(j[0], j[1], buf);
                if (cudaMemcpy(r.base + j[2], buf, (size_t) im.bl[(size_t) im.g.cls(j[0], j[1])], cudaMemcpyHostToDevice) != cudaSuccess)
                    bad = true;
            }
            cudaFreeHost(buf);
        });
    for (auto& t : th) t.join();
    if (bad) {
        err = "remote device: expert upload failed";
        r.close();
        return false;
    }
    std::fprintf(stderr, "remote tier: device %d holds %lld experts (%.2f GiB of %.2f), uploaded in %.1f s\n", r.dev,
                 (long long) r.n, (double) r.used / 1073741824.0, (double) r.cap / 1073741824.0, (now_ms() - t0) / 1000.0);
    r.th = std::thread([&im, &r]() { remote_worker(im, r); });
    im.remote = std::move(rp);
    return true;
}

/// The remote card's profile window: calls, entries, its busy time and the caller's wait (then reset)
void remote_prof_print(RemoteTier& r, const char* what) {
    if (r.p_calls == 0) return;
    std::fprintf(stderr, "[prof_remote] %s: %lld calls, %lld entries; card busy %.0f ms (%.3f ms/call), caller waited %.0f ms "
                         "(%.3f ms/call)\n", what, (long long) r.p_calls, (long long) r.p_entries, r.p_busy,
                 r.p_busy / (double) r.p_calls, r.p_wait, r.p_wait / (double) r.p_calls);
    r.p_calls = r.p_entries = 0;
    r.p_busy = r.p_wait = 0;
}

/// A tier call's remote share: the remote entries leave `ids`/`w` (pointed, with weight 0, at a non-remote expert the
/// TOKEN does not route to - one the call already uses, else a layer proxy - so the call computes a row nobody adds
/// and no (token, expert) pair repeats: the CPU pool's jobs hold at most MAXT tokens) and run on the card meanwhile; the
/// destructor waits and adds their summed rows to `out` (host) or `out_dev` (the main card).
struct RemoteScope {
    Ds41MoeImpl& im;
    bool on = false;
    int n = 0;
    float* out = nullptr;
    void* out_dev = nullptr;
    std::vector<int32_t> ids;
    std::vector<float> w;
    RemoteScope(Ds41MoeImpl& im_, int64_t layer, int n_, const int32_t*& ids_io, const float*& w_io, const float* x,
                const void* x_dev, float* out_, void* out_dev_)
        : im(im_), n(n_), out(out_), out_dev(out_dev_) {
        if (!im.remote || !ids_io || !w_io || n <= 0 || layer < 0 || layer >= im.g.n_layers || !im.g.routed(layer)) return;
        RemoteTier& r = *im.remote;
        const int64_t K = im.g.top_k, NX = im.g.n_experts, H = im.g.n_embd, NE = (int64_t) n * K;
        int64_t cnt = 0;
        for (int64_t j = 0; j < NE; ++j)
            if (ids_io[j] >= 0 && ids_io[j] < NX && r.held(layer, ids_io[j], NX)) ++cnt;
        if (cnt == 0) return;
        ids.assign(ids_io, ids_io + NE);
        w.assign(w_io, w_io + NE);
        // candidates: the call's own non-remote experts (no extra blob), first seen first, then the layer's proxies
        std::vector<int32_t> cand;
        std::vector<char> seen((size_t) NX, 0);
        for (int64_t j = 0; j < NE; ++j) {
            const int32_t e = ids_io[j];
            if (e >= 0 && e < NX && !seen[(size_t) e] && !r.held(layer, e, NX)) { seen[(size_t) e] = 1; cand.push_back(e); }
        }
        for (int i = 0; i < RemoteTier::kProxy; ++i) {
            const int32_t e = r.proxy[(size_t) (layer * RemoteTier::kProxy + i)];
            if (e >= 0 && !seen[(size_t) e]) { seen[(size_t) e] = 1; cand.push_back(e); }
        }
        for (int64_t t = 0; t < n; ++t)
            for (int64_t k = 0; k < K; ++k) {
                const int64_t j = t * K + k;
                if (ids[(size_t) j] < 0 || ids[(size_t) j] >= NX || !r.held(layer, ids[(size_t) j], NX)) continue;
                int32_t sub = -1;
                for (int32_t c : cand) {   // the first candidate this token does not route to (K <= kProxy: one exists)
                    bool used = false;
                    for (int64_t k2 = 0; k2 < K && !used; ++k2) used = ids[(size_t) (t * K + k2)] == c;
                    if (!used) { sub = c; break; }
                }
                if (sub < 0) {
                    std::fprintf(stderr, "ds41_moe: remote: no substitute for layer %lld\n", (long long) layer);
                    std::abort();
                }
                ids[(size_t) j] = sub;
                w[(size_t) j] = 0.0f;
            }
        if (n > r.h_out_tok) {
            if (r.h_out) cudaFreeHost(r.h_out);
            ck(cudaHostAlloc((void**) &r.h_out, (size_t) n * (size_t) H * 4, cudaHostAllocPortable), "remote: h_out");
            r.h_out_tok = n;
        }
        if (!x && x_dev) {   // run_dev: the input is on the main card
            if (!r.h_xd) ck(cudaHostAlloc((void**) &r.h_xd, (size_t) H * 4, cudaHostAllocPortable), "remote: h_xd");
            ck(cudaMemcpy(r.h_xd, x_dev, (size_t) H * 4, cudaMemcpyDeviceToHost), "remote: x d2h");
            x = r.h_xd;
        }
        {
            std::lock_guard<std::mutex> lk(r.mu);
            r.j_layer = layer;
            r.j_n = n;
            r.j_ids.assign(ids_io, ids_io + NE);
            r.j_w.assign(w_io, w_io + NE);
            r.j_x = x;
            r.pending = true;
        }
        r.cv.notify_all();
        ++r.calls;
        r.entries += cnt;
        ++r.p_calls;
        r.p_entries += cnt;
        ids_io = ids.data();
        w_io = w.data();
        on = true;
    }
    ~RemoteScope() {
        if (!on) return;
        RemoteTier& r = *im.remote;
        const double t0 = now_ms();
        {
            std::unique_lock<std::mutex> lk(r.mu);
            r.cv.wait(lk, [&] { return !r.pending; });
        }
        r.wait_ms += now_ms() - t0;
        r.p_wait += now_ms() - t0;
        static const bool xprof = std::getenv("DS41_PROF_X") != nullptr;
        if (xprof && r.p_calls >= 640) remote_prof_print(r, "decode");
        if (!r.ok) {
            std::fprintf(stderr, "ds41_moe: the remote card failed a call\n");
            std::abort();
        }
        const size_t cnt = (size_t) n * (size_t) im.g.n_embd;
        if (out)
            for (size_t i = 0; i < cnt; ++i) out[i] += r.h_out[i];
        if (out_dev) {
            Gpu& gp = *im.gpu;
            if (n > r.g_add_tok) {
                if (r.g_add) cudaFree(r.g_add);
                ck(cudaMalloc(&r.g_add, cnt * 4), "remote: g_add");
                r.g_add_tok = n;
            }
            ck(cudaMemcpyAsync(r.g_add, r.h_out, cnt * 4, cudaMemcpyHostToDevice, gp.s), "remote: rows h2d");
            ds41_add_f32((float*) out_dev, (const float*) r.g_add, cnt, gp.s);
            ck(cudaStreamSynchronize(gp.s), "remote: add sync");
        }
    }
};
#else
bool remote_build(Ds41MoeImpl&, const std::vector<std::pair<int32_t, int32_t>>&, std::string&) { return true; }
struct RemoteScope {
    RemoteScope(Ds41MoeImpl&, int64_t, int, const int32_t*&, const float*&, const float*, const void*, float*, void*) {}
};
#endif


// ---- the one place a blob's address is decided: the arena first, the blob source second ----------------

/// Returns `(layer, expert)`'s blob and sets `*from_file` when it had to be read into slot `scratch` of `ftmp`.
/// The pointer is valid until the next call with the same `scratch` slot, which is all `run` needs.
const uint8_t* acquire(Ds41MoeImpl& im, int64_t layer, int64_t expert, int scratch, bool* from_file) {
#if defined(DS4_MOE_CUDA)
    if (im.gpu) {
        if (const uint8_t* p = im.gpu->arena.ptr(layer, expert)) {
            *from_file = false;
            return p;
        }
    }
#endif
    *from_file = true;
    ++im.st.file_tier;
    if (im.ftmp.size() < (size_t) (scratch + 1) * (size_t) im.blob)
        im.ftmp.resize((size_t) std::max<int64_t>(im.g.top_k, scratch + 1) * (size_t) im.blob);
    uint8_t* dst = im.ftmp.data() + (size_t) scratch * (size_t) im.blob;   // max-blob stride, layer's bytes
    const double t0 = now_ms();
    im.blobs->read(layer, expert, dst);
    im.st.file_ms += now_ms() - t0;
    return dst;
}

/// Sizes `ftmp` for `n` file-tier slots BEFORE a batch's first acquire().  acquire() grows `ftmp` on demand, and a
/// grow reallocates - so in a batch whose slot index passes top_k (a multi-token pass: up to kMaxEnt distinct
/// CPU misses) every earlier job's `blob` pointer into `ftmp` dangled, and the pool then read freed memory (the D3
/// SIGSEGV in ggml_vec_dot_q3_K_q8_K).  Call it with the batch's CPU-miss count; acquire() then never grows.
void reserve_ftmp(Ds41MoeImpl& im, int n) {
    const size_t need = (size_t) std::max<int64_t>(im.g.top_k, n) * (size_t) im.blob;
    if (im.ftmp.size() < need) im.ftmp.resize(need);
}

// ================================ prompt-prefill expert staging ================================
//
// The measured prompt (4096 tokens, chunk 2048) spends 52.9 s of its 85 s waiting on its own file-tier reads:
// run_chunk(l) reads layer l's file-tier experts on its critical path, so the NVMe reads never overlap the dense
// half or the previous layer.  antirez's ds4 does the opposite during prefill - one background thread reads EVERY
// routed expert of layer l+1 with its own fds while layer l computes (ds4.c:42471-42484, ds4_cuda.cu:29290-29393,
// report §A4), which is what turns the latency-bound reads into a bandwidth-bound pipeline.  This is that
// mechanism for our tier: two pinned host slabs ping-pong.  While layer l runs, a reader pool fills the other slab
// with layer l+1's file-tier experts (every expert of l+1 that is neither VRAM-resident nor in the RAM arena - at a
// prompt batch of >= 256 rows nearly all 384 are routed); run_chunk(l+1) then DMAs straight out of the slab.  A blob
// is published only when its read is complete (the job is joined before the slab is consumed), the reads are issued
// in file order by expert id (= the order run_chunk wants them), and a layer whose file tier does not fit the slabs
// is left to run_chunk's own wave reader (no behaviour change).

#if defined(DS4_MOE_CUDA)

/// MIMO_CHUNK_PROF: the staging's own numbers (the layer's file wait is no longer inside run_chunk, so the chunk
/// profile alone would not show that the reads happened at all).
struct ChunkPfProf {
    int64_t layers = 0;    ///< whole layers prefetched (read to completion)
    int64_t blobs = 0;     ///< blobs read into the slabs
    int64_t used = 0;      ///< file-tier DMAs served out of a slab by run_chunk
    int64_t dmas = 0;      ///< file-tier DMAs run_chunk issued (covered + not covered)
    int64_t off = 0;       ///< passes whose file tier the slab did NOT cover (they fell back to the wave reader)
    int64_t checked = 0;   ///< DS41_PF_CHECK: staged blobs re-read with read() and compared
    int64_t bad = 0;       ///< ...of which the staged bytes differed
    double gb = 0;         ///< bytes read into the slabs
    double ms = 0;         ///< summed reader-job wall time (across 8-16 threads)
};
ChunkPfProf g_pf;

struct ChunkStage {
    uint8_t* slab[2] = {nullptr, nullptr};
    int64_t cap = 0;                 ///< bytes of EACH slab (sized to one layer's file tier)
    int cur = 0;                     ///< the slab holding the layer being run
    int64_t layer[2] = {-1, -1};     ///< the layer each slab holds (-1 = none)
    std::vector<int32_t> e[2];       ///< staged expert ids (ascending)
    std::vector<int64_t> off[2];     ///< the slab offset of each staged expert
    std::vector<uint32_t> len[2];    ///< its blob bytes
    std::vector<int32_t> idx[2];     ///< expert id -> index into e[], or -1 (size n_experts)
    std::atomic<int64_t> done[2];    ///< blobs finished (published as they complete)
    std::atomic<int> cancel[2];      ///< the readers stop between blobs when this is set
    std::thread th[2];
    bool running[2] = {false, false};
    double ms[2] = {0, 0};           ///< the reader job's own wall time
    int64_t bytes[2] = {0, 0};       ///< what it read
    std::atomic<int64_t> total[2];   ///< blobs the job was asked for
};

int chunk_pf_threads() {
    static const int n = [] {
        const char* e = std::getenv("DS41_PF_THREADS");
        const int v = e ? std::atoi(e) : 16;
        return v < 1 ? 1 : (v > 32 ? 32 : v);
    }();
    return n;
}

/// The next routed layer after `layer` (chunk_prestage's own rule), or -1 when `layer` is the last one.
int64_t chunk_next_routed(const Ds41MoeImpl& im, int64_t layer) {
    for (int64_t k = 1; k < im.g.n_layers; ++k) {
        const int64_t nl = (layer + k) % im.g.n_layers;
        if (im.g.routed(nl)) return nl;
    }
    return -1;
}

/// The bytes `layer`'s file-tier experts occupy (the slab a prefetch of it would need).
int64_t chunk_stage_want(Ds41MoeImpl& im, int64_t layer) {
    Gpu& gp = *im.gpu;
    int64_t at = 0;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
        if (gp.arena.base && gp.arena.ptr(layer, e)) continue;
        if (gp.rchunk && gp.remote->s->holds[(size_t) (layer * im.g.n_experts + e)]) continue;   // the helper computes it
        at += (int64_t) im.bl[(size_t) im.g.cls(layer, e)];
    }
    return at;
}

/// The staged expert list for slab `s`: `layer`'s file-tier experts, ascending id = the order run_chunk's own
/// `order[2]` uses (so a lookup is an index).  -1 when a layer does not fit the slab (the caller falls back).
int64_t chunk_stage_build(Ds41MoeImpl& im, int s, int64_t layer) {
    ChunkStage& cs = *im.cstage;
    cs.layer[s] = -1;   // a partial build must never look like a staged layer
    cs.e[s].clear();
    cs.off[s].clear();
    cs.len[s].clear();
    if (cs.idx[s].size() != (size_t) im.g.n_experts) cs.idx[s].assign((size_t) im.g.n_experts, -1);
    else std::fill(cs.idx[s].begin(), cs.idx[s].end(), -1);
    Gpu& gp = *im.gpu;
    int64_t at = 0;
    for (int64_t e = 0; e < im.g.n_experts; ++e) {
        if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
        if (gp.arena.base && gp.arena.ptr(layer, e)) continue;
        if (gp.rchunk && gp.remote->s->holds[(size_t) (layer * im.g.n_experts + e)]) continue;   // the helper computes it
        const uint32_t bl = (uint32_t) im.bl[(size_t) im.g.cls(layer, e)];
        if (at + (int64_t) bl > cs.cap) return -1;
        cs.idx[s][(size_t) e] = (int32_t) cs.e[s].size();
        cs.e[s].push_back((int32_t) e);
        cs.off[s].push_back(at);
        cs.len[s].push_back(bl);
        at += (int64_t) bl;
    }
    cs.done[s].store(0);
    cs.cancel[s].store(0);
    cs.total[s].store((int64_t) cs.e[s].size());
    cs.layer[s] = layer;
    return at;
}

/// Start the reader job for slab `s` on its own thread (which fans out to `chunk_pf_threads()` preads).  The caller
/// must have joined the slab first.
void chunk_stage_run(Ds41MoeImpl& im, int s) {
    ChunkStage& cs = *im.cstage;
    if (cs.running[s]) return;
    if (cs.e[s].empty()) {   // nothing file-tier in this layer: staged-empty is a valid, complete answer
        cs.bytes[s] = 0;
        cs.ms[s] = 0;
        return;
    }
    cs.running[s] = true;
    cs.th[s] = std::thread([&im, s] {
        ChunkStage& c = *im.cstage;
        const int64_t n = (int64_t) c.e[s].size();
        const int64_t L = c.layer[s];
        const double t0 = now_ms();
        std::atomic<int64_t> next{0}, got{0};
        const int nth = (int) std::min<int64_t>(n, (int64_t) chunk_pf_threads());
        const size_t sc = std::max<size_t>(im.blobs->read_staged_scratch(), 4096);
        std::vector<std::thread> rd;
        for (int k = 0; k < nth; ++k)
            rd.emplace_back([&, k] {
                void* q = nullptr;
                uint8_t* scratch = ds41_sys::posix_memalign(&q, 4096, sc) == 0 ? (uint8_t*) q : nullptr;
                for (int64_t i; (i = next.fetch_add(1)) < n;) {
                    if (c.cancel[s].load(std::memory_order_relaxed)) break;
                    uint8_t* dst = c.slab[s] + c.off[s][(size_t) i];
                    if (scratch) im.blobs->read_staged(L, c.e[s][(size_t) i], dst, scratch, sc);
                    else im.blobs->read(L, c.e[s][(size_t) i], dst);
                    got.fetch_add((int64_t) c.len[s][(size_t) i], std::memory_order_relaxed);
                    c.done[s].fetch_add(1, std::memory_order_release);
                }
                ds41_sys::aligned_free(scratch);
            });
        for (auto& t : rd) t.join();
        c.bytes[s] = got.load();
        c.ms[s] = now_ms() - t0;
        // DS41_PF_CHECK=1: the staged bytes against the buffered read() they replace, byte for byte.  The read_staged
        // path slices an O_DIRECT span, and a wrong alignment would look exactly like the GPU prefill's own run-to-run
        // drift in the logits - so the identity is checked here instead of inferred from the outputs.
        static const bool check = std::getenv("DS41_PF_CHECK") != nullptr;
        if (check) {
            const int64_t nd = std::min<int64_t>(c.done[s].load(), n);
            std::vector<uint8_t> ref((size_t) im.blob), gotb((size_t) im.blob);
            int64_t bad = 0;
            for (int64_t i = 0; i < nd; ++i) {
                const uint32_t len = c.len[s][(size_t) i];
                im.blobs->read(L, c.e[s][(size_t) i], ref.data());
                std::memcpy(gotb.data(), c.slab[s] + c.off[s][(size_t) i], len);
                if (std::memcmp(ref.data(), gotb.data(), len) != 0) ++bad;
            }
            g_pf.bad += bad;
            g_pf.checked += nd;
            std::fprintf(stderr, "chunk prefetch CHECK: layer %lld - %lld of %lld staged blobs differ from read()\n",
                         (long long) L, (long long) bad, (long long) nd);
        }
        g_pf.blobs += c.done[s].load();
        g_pf.gb += (double) c.bytes[s] / 1e9;
        g_pf.ms += c.ms[s];
        if (c.done[s].load() >= n) ++g_pf.layers;
    });
}

void chunk_stage_join(Ds41MoeImpl& im, int s) {
    ChunkStage& cs = *im.cstage;
    if (!cs.running[s]) return;
    cs.th[s].join();
    cs.running[s] = false;
}

/// Allocate the two slabs once, sized to the largest routed layer's file tier and capped by cfg.chunk_prefetch_gib
/// (else by what host RAM can spare above cfg.mem_floor_gib).  False = prefetch off (a layer would not fit, or the
/// pinned allocation failed) - the caller then leaves run_chunk's on-demand reader in charge.
bool chunk_stage_alloc(Ds41MoeImpl& im) {
    if (im.cstage) return true;
    if (im.cfg.cpu_only || !im.gpu) return false;
    im.cstage.reset(new ChunkStage());
    int64_t want = 0;
    for (int64_t l = 0; l < im.g.n_layers; ++l)
        if (im.g.routed(l)) want = std::max(want, chunk_stage_want(im, l));
    int64_t cap = want;
    if (im.cfg.chunk_prefetch_gib > 0) cap = std::min(cap, (int64_t) (im.cfg.chunk_prefetch_gib * 1073741824.0 / 2.0));
    const int64_t avail = mem_available_gib();
    if (avail >= 0) cap = std::min(cap, std::max<int64_t>(0, (avail - (int64_t) im.cfg.mem_floor_gib - 2) * 1073741824LL / 2));
    if (want <= 0 || cap < want) {
        std::fprintf(stderr, "chunk prefetch: off - a layer's file tier is %.2f GiB, the slab budget is %.2f GiB\n",
                     want / 1073741824.0, (want <= 0 ? 0.0 : cap / 1073741824.0));
        im.cstage.reset();
        return false;
    }
    ChunkStage& cs = *im.cstage;
    cs.cap = cap;
    for (int i = 0; i < 2; ++i) {
        cs.done[i].store(0);
        cs.cancel[i].store(0);
        cs.total[i].store(0);
        if (cudaMallocHost((void**) &cs.slab[i], (size_t) cap) != cudaSuccess) {
            std::fprintf(stderr, "chunk prefetch: off - a pinned slab of %.2f GiB failed\n", cap / 1073741824.0);
            for (int j = 0; j < i; ++j) cudaFreeHost(cs.slab[j]);
            im.cstage.reset();
            return false;
        }
    }
    std::fprintf(stderr, "chunk prefetch: two %.2f GiB pinned slabs, %d reader threads (layer file tier up to %.2f GiB)\n",
                 cap / 1073741824.0, chunk_pf_threads(), want / 1073741824.0);
    return true;
}

/// Makes `layer`'s file-tier experts available in a slab (a synchronous fill when they were not prefetched), then
/// starts the NEXT routed layer's fill into the other slab.  Returns the slab holding `layer`, or -1 (use the
/// on-demand reader).  Called once per run_chunk(layer), before its class passes consume the slab.
int chunk_pf_prepare(Ds41MoeImpl& im, int64_t layer, int n) {
    if (!im.cfg.chunk_prefetch || !im.gpu || im.cfg.cpu_only) return -1;
    // STRATA_CHUNK_PF_MIN (rows; default: never - see below): the slabs stage a layer's WHOLE file tier before its routing is known -
    // a win only when the call routes most of it.  s11 (512K serve turns, batch s, prefetch on / off): fresh 8000
    // 33.7 / 38.8, +1000 tokens 19.2 / 22.0, +272 tokens 18.3 / 14.5 (it routes ~39% of the file tier) -> break-even
    // ~690 rows.  Below the cut the class passes read only the routed file-tier experts (the on-demand reader).
    // After the wave reader went to 32 threads x 8 parts (5.3 GB/s for the ROUTED experts only), reading on demand
    // beats the whole-layer prefetch at every size measured (512K turns, gate 640 / never: fresh 8000 33.6 / 32.4 s,
    // +1000 19.1 / 16.6 s, +272 10.5 / 10.5 s) -> default never; the slabs (2 x 2.71 GiB pinned) are then never
    // allocated either.  STRATA_CHUNK_PF_MIN=<rows> brings it back.
    static const int pf_min = std::getenv("STRATA_CHUNK_PF_MIN") ? std::atoi(std::getenv("STRATA_CHUNK_PF_MIN")) : (1 << 30);
    if (n < pf_min) return -1;
    if (!chunk_stage_alloc(im)) return -1;
    ChunkStage& cs = *im.cstage;
    int s = -1;
    for (int i = 0; i < 2; ++i)
        if (cs.layer[i] == layer) s = i;
    if (s < 0) {   // the first chunk's layer 0, a non-consecutive call, or a previous miss: fill synchronously
        s = 0;
        chunk_stage_join(im, s);
        if (chunk_stage_build(im, s, layer) < 0) {
            cs.layer[s] = -1;
            return -1;
        }
        chunk_stage_run(im, s);
    }
    chunk_stage_join(im, s);   // the whole layer must be in the slab before its DMAs start (antirez's publish rule)
    cs.cur = s;                // the slab gpu_run_chunk consumes (and the covers check looks at)
    const int64_t nx = chunk_next_routed(im, layer);
    if (nx >= 0 && nx != layer) {
        const int o = 1 - s;
        if (cs.layer[o] != nx) {
            cs.cancel[o].store(1);
            chunk_stage_join(im, o);
            cs.layer[o] = -1;
            if (chunk_stage_build(im, o, nx) >= 0) chunk_stage_run(im, o);
        }
    }
    return cs.layer[s] == layer ? s : -1;
}

/// Join the readers and free the slabs (pinned host memory; safe to call at any time, called only from close()).
void chunk_stage_close(Ds41MoeImpl& im) {
    if (!im.cstage) return;
    for (int i = 0; i < 2; ++i) {
        im.cstage->cancel[i].store(1);
        chunk_stage_join(im, i);
        if (im.cstage->slab[i]) cudaFreeHost(im.cstage->slab[i]);
        im.cstage->slab[i] = nullptr;
    }
    im.cstage.reset();
}

#endif  // DS4_MOE_CUDA


// ---- one input-activation buffer per distinct `gu_act` (the one-dispatch-per-layer path) ----------------------
/// The buffer for the gate/up activation format `gu_act` (built in `init`), or the single `nact` when the
/// geometry has no such class.  Two classes with the same `gu_act` share it: the quantized activation is a pure
/// function of the format and the hidden state, so quantizing once per format is byte-identical to per class.
uint8_t* nact_slot(Ds41MoeImpl& im, int gu_act) {
    for (size_t b = 0; b < im.nact_multi_act.size(); ++b)
        if (im.nact_multi_act[b] == gu_act) return im.nact_multi.data() + b * cpu::kNativeActBytes;
    return im.nact.data();
}
/// run_multi with per-expert classes: one slot per (token, distinct gate/up format).  `nact_slot` is shared
/// across tokens because a single-token gpu_run quantizes one activation; a batched pass must hold every
/// token's activation for every class its CPU experts use, so they cannot share.
int nact_act_idx(const Ds41MoeImpl& im, int gu_act) {
    for (size_t b = 0; b < im.nact_multi_act.size(); ++b) if (im.nact_multi_act[b] == gu_act) return (int) b;
    return -1;
}
uint8_t* nact_slot_n(Ds41MoeImpl& im, int t, int gu_act) {
    const int b = nact_act_idx(im, gu_act);
    if (b < 0) return im.nact.data();
    return im.nact_multi_n.data() + ((size_t) t * im.nact_multi_act.size() + (size_t) b) * cpu::kNativeActBytes;
}

// ================================ geometry ================================

bool ds41_moe_blob_layout(const Ds41MoeGeom& g, cpu::NativeFmt& f, std::string& err) {
    if (g.n_embd <= 0 || g.n_ff <= 0 || g.n_layers <= 0 || g.n_experts <= 0) {
        err = "geometry has a non-positive dimension";
        return false;
    }
    if (g.top_k < 1 || g.top_k > cpu::MAXT) {
        err = "top_k " + std::to_string(g.top_k) + " is outside 1.." + std::to_string((int) cpu::MAXT);
        return false;
    }
    if (g.per_expert()) {   // the largest class's layout: what every per-call buffer is sized by
        bool any = false;
        for (int64_t c = 0; c < g.n_classes(); ++c) {
            cpu::NativeFmt fc;
            if (!ds41_moe_class_layout(g, c, fc, err)) return false;
            if (!any || fc.bytes > f.bytes) f = fc;
            any = true;
        }
        if (!any) err = "per-expert geometry with no class";
        return any;
    }
    if (g.per_layer()) {   // the largest routed layer's layout: what every per-call buffer is sized by
        bool any = false;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.routed(l)) continue;
            cpu::NativeFmt fl;
            if (!ds41_moe_layer_layout(g, l, fl, err)) return false;
            if (!any || fl.bytes > f.bytes) f = fl;
            any = true;
        }
        if (!any) err = "per-layer geometry with no routed layer";
        return any;
    }
    std::string e;
    if (!cpu::native_fmt(g.gu_type, g.d_type, g.n_embd, g.n_ff, f, e)) {
        err = "native_fmt(" + std::to_string(g.gu_type) + "/" + std::to_string(g.d_type) + ") at " +
              std::to_string(g.n_embd) + "/" + std::to_string(g.n_ff) + ": " + e;
        return false;
    }
    f.swiglu_limit = g.swiglu_limit;
    return true;
}

bool ds41_moe_class_layout(const Ds41MoeGeom& g, int64_t c, cpu::NativeFmt& f, std::string& err) {
    if (!g.per_expert()) return ds41_moe_layer_layout(g, c, f, err);
    std::string e;
    const int gu = g.cls_gu[(size_t) c], dt = g.cls_d[(size_t) c];
    if (!cpu::native_fmt3(gu, gu, dt, g.n_embd, g.n_ff, f, e)) {
        err = "class " + std::to_string(c) + ": native_fmt3(" + std::to_string(gu) + "/" + std::to_string(dt) + "): " + e;
        return false;
    }
    f.swiglu_limit = g.swiglu_limit;
    return true;
}

bool ds41_moe_layer_layout(const Ds41MoeGeom& g, int64_t l, cpu::NativeFmt& f, std::string& err) {
    if (!g.per_layer()) return ds41_moe_blob_layout(g, f, err);
    if (!g.routed(l)) {
        err = "layer " + std::to_string(l) + " has no routed experts";
        return false;
    }
    if (g.top_k < 1 || g.top_k > cpu::MAXT) {
        err = "top_k " + std::to_string(g.top_k) + " is outside 1.." + std::to_string((int) cpu::MAXT);
        return false;
    }
    std::string e;
    if (!cpu::native_fmt3(g.gate_type(l), g.up_type(l), g.down_type(l), g.n_embd, g.n_ff, f, e)) {
        err = "layer " + std::to_string(l) + ": native_fmt3(" + std::to_string(g.gate_type(l)) + "/" +
              std::to_string(g.up_type(l)) + "/" + std::to_string(g.down_type(l)) + "): " + e;
        return false;
    }
    f.swiglu_limit = g.swiglu_limit;
    return true;
}

bool ds41_moe_geom_from_gguf(const std::string& gguf, Ds41MoeGeom& g, std::string& err) {
    try {
        GgufFile f(gguf);
        // the architecture's own key prefix: deepseek4.*, mimo2.*, ... (the key names are llama.cpp's, shared)
        std::string arch = "deepseek4";
        if (const MetaValue* a = f.get("general.architecture"); a && a->type == MetaType::STRING) arch = a->s;
        int64_t v = 0;
        const std::pair<std::string, int64_t*> keys[] = {
            {arch + ".block_count", &g.n_layers},
            {arch + ".embedding_length", &g.n_embd},
            {arch + ".expert_count", &g.n_experts},
            {arch + ".expert_used_count", &g.top_k},
            {arch + ".expert_feed_forward_length", &g.n_ff},
        };
        for (const auto& k : keys) {
            if (!meta_i64(f, k.first, v)) {
                err = "no metadata " + k.first + " (is this an MoE GGUF?)";
                return false;
            }
            *k.second = v;
        }
        // every layer's three expert types; a layer with none of the tensors has no routed experts (MiMo's layer 0)
        const char* roles[3] = {"gate", "up", "down"};
        std::vector<int> ty[3];
        for (int64_t l = 0; l < g.n_layers; ++l) {
            int have = 0;
            int t3[3] = {-1, -1, -1};
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long) l, roles[r]);
                const TensorInfo* ti = f.find(name);
                if (!ti) continue;
                ++have;
                // ne0 is the input width: gate/up are [n_embd, n_ff, n_experts], down is [n_ff, n_embd, n_experts].
                if ((int64_t) ti->shape.at(0) != (r == 2 ? g.n_ff : g.n_embd)) {
                    err = std::string(name) + " has ne0 " + std::to_string(ti->shape.at(0));
                    return false;
                }
                t3[r] = (int) ti->type;
            }
            if (have != 0 && have != 3) {
                err = "layer " + std::to_string(l) + " has " + std::to_string(have) + " of the 3 expert tensors";
                return false;
            }
            for (int r = 0; r < 3; ++r) ty[r].push_back(t3[r]);
        }
        int first = -1;
        bool uniform = true;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (ty[0][(size_t) l] < 0) { uniform = false; continue; }
            if (first < 0) first = (int) l;
            uniform = uniform && ty[0][(size_t) l] == ty[0][(size_t) first] &&
                      ty[1][(size_t) l] == ty[0][(size_t) first] && ty[2][(size_t) l] == ty[2][(size_t) first];
        }
        if (first < 0) {
            err = "no layer has routed expert tensors";
            return false;
        }
        g.gu_type = ty[0][(size_t) first];
        g.d_type = ty[2][(size_t) first];
        g.gate_types.clear();
        g.up_types.clear();
        g.down_types.clear();
        if (!uniform) {   // per-layer formats (or layers without experts): the vectors carry them
            g.gate_types = ty[0];
            g.up_types = ty[1];
            g.down_types = ty[2];
        }
        // the SwiGLU clamp: one limit for the whole tier (it has one NativeFmt / layout), so every layer must agree
        std::vector<double> lim;
        const std::string le = strata::ds4_key_f64_arr(f, (arch + ".swiglu_clamp_exp").c_str(), lim);
        if (le.empty() && !lim.empty()) {
            for (double x : lim)
                if (x != lim[0]) {
                    err = arch + ".swiglu_clamp_exp differs between layers; the tier supports one limit";
                    return false;
                }
            if (lim[0] > 0.0) g.swiglu_limit = (float) lim[0];   // <= 0 means "no clamp" in llama.cpp
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

// ================================ blob sources ================================

struct Ds41GgufBlobs::Impl {
    std::unique_ptr<GgufModel> model;
    std::vector<const uint8_t*> src[3];   // per role, per layer (null: no routed experts)
    std::vector<size_t> stride[3];        // per role, per layer: bytes of one expert's slice
    // ds41_sys::pread() path: per role, per layer, the shard's fd and the slice array's file offset.  A cold mmap memcpy
    // faults page by page (MiMo on ntfs3: ~1.1 GB/s even with 8 readers, FINDINGS s10); pread asks for each slice
    // in one request.  Same bytes.  STRATA_BLOB_MMAP=1 keeps the memcpy path.
    std::vector<int> fd[3];
    std::vector<uint64_t> foff[3];
    std::vector<int> fds;
    // O_DIRECT twins of `fds` (-1 where the filesystem refused it): 4 KiB-aligned spans into a per-thread bounce,
    // bypassing the page cache (MiMo's NVMe: 5.5 GB/s direct vs 2.6 buffered sequential).  STRATA_BLOB_DIRECT=0 off.
    std::vector<int> dfds;
    std::vector<int> dfd[3];
    ~Impl() {
        for (int f : fds) if (f >= 0) ds41_sys::close(f);
        for (int f : dfds) if (f >= 0) ds41_sys::close(f);
    }
};

Ds41GgufBlobs::Ds41GgufBlobs(Ds41MoeGeom g, std::string gguf) : im_(new Impl), g_(g), path_(std::move(gguf)) {}
Ds41GgufBlobs::~Ds41GgufBlobs() = default;

bool Ds41GgufBlobs::open(std::string& err) {
    cpu::NativeFmt f;
    if (!ds41_moe_blob_layout(g_, f, err)) return false;
    blob_ = f.bytes;   // the largest layer's
    try {
        im_->model.reset(new GgufModel(gguf_split_paths(path_)));
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            im_->src[r].assign((size_t) g_.n_layers, nullptr);
            im_->stride[r].assign((size_t) g_.n_layers, 0);
        }
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            if (!g_.routed(l)) continue;
            cpu::NativeFmt fl;
            if (!ds41_moe_layer_layout(g_, l, fl, err)) return false;
            im_->stride[0][(size_t) l] = fl.up_off;
            im_->stride[1][(size_t) l] = fl.down_off - fl.up_off;
            im_->stride[2][(size_t) l] = fl.bytes - fl.down_off;
            const int want[3] = {g_.gate_type(l), g_.up_type(l), g_.down_type(l)};
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long) l, roles[r]);
                size_t at = 0;
                const TensorInfo* ti = im_->model->find(name, &at);
                if (!ti) {
                    err = std::string("no tensor ") + name;
                    return false;
                }
                if (ti->type != (uint32_t) want[r]) {
                    err = std::string(name) + " has type " + ti->type_name();
                    return false;
                }
                im_->src[r][(size_t) l] = im_->model->shard(at).tensor_data(*ti);
                if (!std::getenv("STRATA_BLOB_MMAP")) {
                    if (im_->fds.size() < im_->model->size()) im_->fds.assign(im_->model->size(), -1);
                    int& fdv = im_->fds[at];
                    if (fdv < 0) fdv = ds41_sys::open(im_->model->shard(at).path().c_str(), O_RDONLY | O_CLOEXEC);
                    if (im_->fd[r].empty()) { im_->fd[r].assign((size_t) g_.n_layers, -1); im_->foff[r].assign((size_t) g_.n_layers, 0); }
                    im_->fd[r][(size_t) l] = fdv;
                    const char* dv = std::getenv("STRATA_BLOB_DIRECT");
                    if (!(dv && dv[0] == '0')) {
                        if (im_->dfds.size() < im_->model->size()) im_->dfds.assign(im_->model->size(), -2);
                        int& dfv = im_->dfds[at];
                        if (dfv == -2) dfv = ds41_sys::open(im_->model->shard(at).path().c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                        if (im_->dfd[r].empty()) im_->dfd[r].assign((size_t) g_.n_layers, -1);
                        im_->dfd[r][(size_t) l] = dfv;
                    }
                    im_->foff[r][(size_t) l] = im_->model->shard(at).data_start() + ti->offset;
                }
            }
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

size_t Ds41GgufBlobs::blob_bytes(int64_t l) const {
    if (l < 0 || l >= g_.n_layers) return 0;
    return im_->stride[0][(size_t) l] + im_->stride[1][(size_t) l] + im_->stride[2][(size_t) l];
}

void Ds41GgufBlobs::read_span(int r, int64_t layer, int64_t expert, size_t lo, size_t hi, uint8_t* dst) const {
    const size_t sz = im_->stride[r][(size_t) layer];
    if (hi > sz) hi = sz;
    if (lo >= hi) return;
    const size_t n = hi - lo;
    const int fdv = im_->fd[r].empty() ? -1 : im_->fd[r][(size_t) layer];
    const int dfv = im_->dfd[r].empty() ? -1 : im_->dfd[r][(size_t) layer];
    const uint64_t off = (fdv >= 0 || dfv >= 0) ? im_->foff[r][(size_t) layer] + (uint64_t) expert * sz + lo : 0;
    if (dfv >= 0) {   // O_DIRECT: the aligned span around the range, then the range out of it
        constexpr uint64_t A = 4096;
        const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1);
        struct Bounce { uint8_t* p = nullptr; size_t n = 0; ~Bounce() { ds41_sys::aligned_free(p); } };
        thread_local Bounce bb;
        if (bb.n < a1 - a0) {
            ds41_sys::aligned_free(bb.p);
            bb.p = nullptr;
            bb.n = 0;
            void* q = nullptr;
            if (ds41_sys::posix_memalign(&q, A, (size_t) (a1 - a0)) == 0) { bb.p = (uint8_t*) q; bb.n = (size_t) (a1 - a0); }
        }
        if (bb.p) {
            size_t got = 0;
            while (got < a1 - a0) {
                const ssize_t k = ds41_sys::pread(dfv, bb.p + got, (size_t) (a1 - a0 - got), (int64_t) (a0 + got));
                if (k <= 0) break;
                got += (size_t) k;
            }
            if (got >= off + n - a0) {   // the file's last span may be short of a1; the range is all we need
                std::memcpy(dst, bb.p + (off - a0), n);
                return;
            }
        }
    }
    if (fdv >= 0) {   // pread the range; fall back to the mmap on any short read / error
        size_t got = 0;
        while (got < n) {
            const ssize_t k = ds41_sys::pread(fdv, dst + got, n - got, (int64_t) (off + got));
            if (k <= 0) break;
            got += (size_t) k;
        }
        if (got == n) return;
    }
    std::memcpy(dst, im_->src[r][(size_t) layer] + (size_t) expert * sz + lo, n);
}

void Ds41GgufBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    // The three role slices back to back, [gate rows | up rows | down rows] (native_expert.hpp's layout).
    // Per-layer strides (gate and up may differ: MiMo).
    size_t at = 0;
    for (int r = 0; r < 3; ++r) {
        const size_t sz = im_->stride[r][(size_t) layer];
        read_span(r, layer, expert, 0, sz, dst + at);
        at += sz;
    }
}

void Ds41GgufBlobs::read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const {
    if (nparts <= 1) {
        if (part == 0) read(layer, expert, dst);
        return;
    }
    const size_t bl = blob_bytes(layer);
    auto cut = [&](int q) { return q >= nparts ? bl : std::min(bl, (bl * (size_t) q / (size_t) nparts) & ~(size_t) 4095); };
    const size_t p0 = cut(part), p1 = cut(part + 1);
    size_t at = 0;
    for (int r = 0; r < 3 && at < p1; ++r) {
        const size_t sz = im_->stride[r][(size_t) layer];
        const size_t lo = std::max(p0, at), hi = std::min(p1, at + sz);
        if (lo < hi) read_span(r, layer, expert, lo - at, hi - at, dst + lo);
        at += sz;
    }
}

bool Ds41MemoryBlobs::open(std::string& err) {
    cpu::NativeFmt f;
    if (!ds41_moe_blob_layout(g_, f, err)) return false;
    if (f.bytes != blob_) {
        err = "blob size " + std::to_string(blob_) + " != the geometry's " + std::to_string(f.bytes);
        return false;
    }
    return true;
}

void Ds41MemoryBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    const auto it = blobs_.find({layer, expert});
    if (it != blobs_.end()) std::memcpy(dst, it->second.data(), blob_);
    else std::memset(dst, 0, blob_);   // see the header: `has()` is the gate's assertion, this is the fallback
}

void Ds41MemoryBlobs::set(int64_t layer, int64_t expert, const uint8_t* blob) {
    if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_experts) return;
    blobs_[{layer, expert}].assign(blob, blob + blob_);
}

bool Ds41MemoryBlobs::has(int64_t layer, int64_t expert) const {
    return blobs_.find({layer, expert}) != blobs_.end();
}

// ================================ the tier ================================

Ds41MoeTier::Ds41MoeTier() : im_(new Ds41MoeImpl) {}
Ds41MoeTier::~Ds41MoeTier() { close(); }

bool Ds41MoeTier::init(const std::string& gguf, const Ds41MoeConfig& cfg, std::string& err) {
    Ds41MoeGeom g;
    if (!ds41_moe_geom_from_gguf(gguf, g, err)) return false;
    std::unique_ptr<Ds41BlobSource> src(new Ds41GgufBlobs(g, gguf));
    if (!src->open(err)) return false;
    if (!init(g, src.get(), cfg, err)) return false;
    im_->owned_blobs = std::move(src);
    return true;
}

bool Ds41MoeTier::init(const Ds41MoeGeom& geom, Ds41BlobSource* blobs, const Ds41MoeConfig& cfg, std::string& err) {
    close();
    im_->g = geom;
    im_->cfg = cfg;
    im_->blobs = blobs;
    // hand-port of sakamoto-sann b1e0db08: a NaN or negative auto budget would size the uniform cache nonsensically
    if (!std::isfinite(cfg.slot_gib) || cfg.slot_gib < 0) {
        err = "VRAM slot budget must be finite and nonnegative"; return false;
    }
    if (!ds41_moe_blob_layout(geom, im_->f, err)) return false;
    im_->blob = (int64_t) im_->f.bytes;
    // per-CLASS formats and blob sizes (class = layer unless the geometry is per-expert; all equal to f / blob for a
    // uniform geometry)
    const int64_t NCL = geom.n_classes();
    im_->fl.assign((size_t) NCL, cpu::NativeFmt());
    im_->bl.assign((size_t) NCL, 0);
    im_->mixed_sizes = false;
    for (int64_t c = 0; c < NCL; ++c) {
        if (!geom.per_expert() && !geom.routed(c)) continue;
        if (!geom.per_expert() && !geom.per_layer()) {
            im_->fl[(size_t) c] = im_->f;
        } else if (!ds41_moe_class_layout(geom, c, im_->fl[(size_t) c], err)) {
            return false;
        }
        im_->bl[(size_t) c] = (int64_t) im_->fl[(size_t) c].bytes;
        im_->mixed_sizes = im_->mixed_sizes || im_->bl[(size_t) c] != im_->blob;
    }
    if (!blobs->open(err)) return false;
    if ((int64_t) blobs->blob_bytes() != im_->blob) {
        err = "blob source is " + std::to_string(blobs->blob_bytes()) + " bytes, the geometry's is " +
              std::to_string(im_->blob);
        return false;
    }
    im_->pool.reset(new cpu::ExpertPool(cfg.threads > 0 ? cfg.threads : 0));
    im_->jobs.resize((size_t) geom.top_k);
    im_->parts.assign((size_t) geom.top_k * (size_t) geom.n_embd, 0.0f);
    im_->nact.assign(cpu::kNativeActBytes, 0);
    im_->nact_n.assign((size_t) kMaxTok * cpu::kNativeActBytes, 0);
    im_->actq_n.resize((size_t) kMaxTok);
    // the input-activation formats the classes of this model need (a Q2_0 gate/up uses `actq`, not one of these)
    im_->nact_multi_act.clear();
    for (int64_t c = 0; c < NCL; ++c) {
        if (!geom.per_expert() && !geom.routed(c)) continue;
        const int a = im_->fl[(size_t) c].gu_act;
        if (std::find(im_->nact_multi_act.begin(), im_->nact_multi_act.end(), a) == im_->nact_multi_act.end())
            im_->nact_multi_act.push_back(a);
    }
    im_->nact_multi.assign(im_->nact_multi_act.size() * cpu::kNativeActBytes, 0);
    im_->nact_multi_n.assign((size_t) kMaxTok * im_->nact_multi_act.size() * cpu::kNativeActBytes, 0);
    im_->parts_n.assign((size_t) kMaxEnt * (size_t) geom.n_embd, 0.0f);

    if (cfg.cpu_only) {
        im_->inited = true;
        return true;
    }

#if !defined(DS4_MOE_CUDA)
    err = "this build has no CUDA tier (compiled without DS4_MOE_CUDA); set cfg.cpu_only";
    return false;
#else
    if (cfg.cuda_device >= 0 && cudaSetDevice(cfg.cuda_device) != cudaSuccess) {
        err = "cudaSetDevice(" + std::to_string(cfg.cuda_device) + ") failed";
        return false;
    }
    Gpu& gp = *(im_->gpu = std::unique_ptr<Gpu>(new Gpu()));
    gp.arena.blob = im_->blob;
    gp.arena.n_layers = geom.n_layers;
    gp.arena.n_experts = geom.n_experts;
    gp.gl = strata::kernels::native_expert_layout(geom.gu_type, geom.d_type, geom.n_embd, geom.n_ff);
    gp.gl.swiglu_limit = geom.swiglu_limit;
    gp.gll.assign((size_t) NCL, gp.gl);
    for (int64_t l = 0; l < NCL; ++l) {   // l = a class (= the layer unless per-expert)
        if (!geom.per_expert() && !geom.routed(l)) continue;
        const int gt = geom.per_expert() ? geom.cls_gu[(size_t) l] : geom.gate_type(l);
        const int ut = geom.per_expert() ? gt : geom.up_type(l);
        const int dt = geom.per_expert() ? geom.cls_d[(size_t) l] : geom.down_type(l);
        if (!strata::kernels::native_expert_supported3(gt, ut, dt, geom.n_embd, geom.n_ff)) {
            err = "native_expert_grouped has no kernel for layer " + std::to_string(l) + "'s types " +
                  std::to_string(gt) + "/" + std::to_string(ut) + "/" + std::to_string(dt);
            return false;
        }
        strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout3(gt, ut, dt, geom.n_embd, geom.n_ff);
        L.swiglu_limit = geom.swiglu_limit;
        if ((int64_t) L.bytes != im_->bl[(size_t) l]) {
            err = "layer " + std::to_string(l) + ": GPU blob " + std::to_string(L.bytes) + " != CPU blob " +
                  std::to_string(im_->bl[(size_t) l]);
            return false;
        }
        gp.gll[(size_t) l] = L;
    }
    ck(cudaStreamCreate(&gp.s), "stream");
    const size_t H4 = (size_t) geom.n_embd * 4;
    ck(cudaMalloc(&gp.d_x, (size_t) kMaxTok * H4), "d_x");
    // d_xq: kMaxTok q8_1 rows.  native_q8_1_bytes (like the quantizer) takes at most 8 columns, so size one column
    // and check the layout is per-column contiguous - quantize_rows_q8_1 steps through > 8 rows on that
    const size_t xq_col = strata::kernels::native_q8_1_bytes((int) geom.n_embd, 1);
    if (strata::kernels::native_q8_1_bytes((int) geom.n_embd, 8) != 8 * xq_col) {
        err = "q8_1 layout is not per-column contiguous";
        return false;
    }
    ck(cudaMalloc(&gp.d_xq, (size_t) kMaxTok * xq_col), "d_xq");
    ck(cudaMalloc(&gp.d_parts, (size_t) kMaxEnt * H4), "d_parts");
    ck(cudaMalloc(&gp.d_grp_ptr, sizeof(unsigned long long) * kMaxEnt), "d_grp_ptr");
    ck(cudaMalloc(&gp.d_grp_start, sizeof(int32_t) * (kMaxEnt + 1)), "d_grp_start");
    ck(cudaMalloc(&gp.d_ngroups, sizeof(int32_t)), "d_ngroups");
    ck(cudaMalloc(&gp.d_ent_dst, sizeof(int32_t) * kMaxEnt), "d_ent_dst");
    ck(cudaMalloc(&gp.d_ent_tok, sizeof(int32_t) * kMaxEnt), "d_ent_tok");
    ck(cudaMalloc(&gp.d_scratch, strata::kernels::native_expert_scratch_bytes(kMaxEnt, geom.n_ff)), "d_scratch");
    // STRATA_D3_DMA_STREAM=1 (default off): gpu_run_n_ex's PCIe site copies on a stream of their own (Gpu::s_dn), into
    // two banks of staging slots.  STRATA_D3_STAGE_SLOTS=N: the d_stage slots (default kMaxParts, 2 * kMaxParts with
    // the stream on); d_stage holds max(kMaxParts, N) blobs, and gpu_run (and the stream-off path) use the first
    // kMaxParts as before.
    {
        const char* dn = std::getenv("STRATA_D3_DMA_STREAM");
        const bool dn_on = dn && dn[0] == '1';
        const char* ns = std::getenv("STRATA_D3_STAGE_SLOTS");
        const int want = ns && *ns ? std::max(1, std::atoi(ns)) : (dn_on ? 2 * kMaxParts : kMaxParts);
        gp.n_stage = std::max(kMaxParts, want);
        if (dn_on) {
            gp.d3_bank = std::max(1, std::min(kMaxParts, want / 2));
            ck(cudaStreamCreateWithFlags(&gp.s_dn, cudaStreamNonBlocking), "d3 dma stream");
            for (cudaEvent_t& e : gp.dn_ev) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "d3 dma event");
            std::fprintf(stderr, "ds41_moe: STRATA_D3_DMA_STREAM on: %d staging slots, 2 banks x %d\n", gp.n_stage, gp.d3_bank);
        }
    }
    ck(cudaMalloc(&gp.d_stage, (size_t) gp.n_stage * (size_t) im_->blob), "d_stage");
    for (int i = 0; i < 4; ++i) ck(cudaEventCreate(&gp.ev[i]), "event");
    ck(cudaMallocHost((void**) &gp.h_x, (size_t) kMaxTok * H4), "h_x");
    ck(cudaMallocHost(&gp.h_parts, (size_t) kMaxEnt * H4), "h_parts");
    ck(cudaMalloc(&gp.d_meta, kMetaBytes), "d_meta");
    ck(cudaMallocHost((void**) &gp.h_meta, kMetaBytes), "h_meta");
    if (const char* rn = std::getenv("STRATA_DS41_REMOTE")) {
        std::unique_ptr<Remote> r(new Remote());
        std::string e2;
        if (!r->map.open(rn, false, e2)) {
            err = "STRATA_DS41_REMOTE=" + std::string(rn) + ": " + e2 + " (start ds41_helper first)";
            return false;
        }
        r->s = r->map.shm();
        // the helper loads its experts from the model file: wait for it (up to STRATA_DS41_REMOTE_WAIT s, default 600)
        const double wait_s = std::getenv("STRATA_DS41_REMOTE_WAIT") ? std::atof(std::getenv("STRATA_DS41_REMOTE_WAIT")) : 600.0;
        const double tw = now_ms();
        while (r->s->ready.load(std::memory_order_acquire) == 0 && now_ms() - tw < wait_s * 1000.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (r->s->ready.load(std::memory_order_acquire) != 1) {
            err = "STRATA_DS41_REMOTE=" + std::string(rn) + ": the helper is not serving (failed, or still loading after " +
                  std::to_string((int) wait_s) + " s)";
            return false;
        }
        if (r->s->magic != Ds41RemoteShm::kMagic || r->s->version != Ds41RemoteShm::kVersion ||
            r->s->n_layers != geom.n_layers || r->s->n_experts != geom.n_experts || r->s->n_embd != geom.n_embd) {
            err = "STRATA_DS41_REMOTE=" + std::string(rn) + ": the helper serves another model or protocol";
            return false;
        }
        // pinned in this process too: the rows are DMA'd straight out of the segment
        ck(cudaHostRegister(r->s, sizeof(Ds41RemoteShm), cudaHostRegisterDefault), "register the remote segment");
        r->n = r->s->done.load(std::memory_order_acquire);
        std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE=%s: the helper holds %lld experts (%.2f GiB)\n", rn,
                     (long long) r->s->held, r->s->held_bytes / 1073741824.0);
        gp.remote = std::move(r);
        const char* rce = std::getenv("STRATA_DS41_REMOTE_CHUNK");
        if (!(rce && rce[0] == '0')) {
            std::unique_ptr<RemoteChunk> c(new RemoteChunk());
            std::string e3;
            void* dp = nullptr;
            if (!c->map.open(std::string(rn) + "_chunk", false, e3, 0)) {
                std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE=%s: no prompt-chunk segment (%s) - prompts run without the helper\n", rn, e3.c_str());
            } else if ((c->c = (Ds41RemoteChunk*) c->map.data())->magic != Ds41RemoteChunk::kMagic || c->c->n_embd != geom.n_embd) {
                std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE=%s: the prompt-chunk segment does not match - prompts run without the helper\n", rn);
            } else if (cudaHostRegister(c->c, (size_t) c->c->bytes, cudaHostRegisterMapped) != cudaSuccess ||
                       cudaHostGetDevicePointer(&dp, c->c->out(), 0) != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE=%s: cannot map the prompt-chunk segment - prompts run without the helper\n", rn);
            } else {
                c->out_dev = (const float*) dp;
                c->n = c->c->done.load(std::memory_order_acquire);
                c->mask_e.assign((size_t) geom.n_experts, 0);
                std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE=%s: prompt chunks too (up to %lld rows, %lld entries)\n", rn,
                             (long long) c->c->cap_tok, (long long) c->c->cap_ent);
                gp.rchunk = std::move(c);
            }
        }
    }
    // STRATA_DMA_OVERLAP=1 (slice A, docs/ds4/V41_DECODE_OVERLAP_PLAN.md): a separate stream for the demand copies
    // so a layer's PCIe/file DMA overlaps that layer's resident-expert kernels.  Off by default: unproven until the
    // nsys overlap metric (max concurrent kernels > 1) and repeated forced-token identity both hold.
    if (std::getenv("STRATA_DMA_OVERLAP")) {
        ck(cudaStreamCreateWithFlags(&gp.s_dc, cudaStreamNonBlocking), "dc stream");
        ck(cudaEventCreateWithFlags(&gp.ev_dc, cudaEventDisableTiming), "ev_dc");
    }
    if (cfg.pf_b > 0) {
        ck(cudaStreamCreateWithFlags(&gp.s_pf, cudaStreamNonBlocking), "pf stream");
        ck(cudaMalloc(&gp.d_pf, (size_t) Ds41MoeConfig::kMaxPf * (size_t) im_->blob), "d_pf");
        ck(cudaEventCreateWithFlags(&gp.ev_pf, cudaEventDisableTiming), "ev_pf");
        ck(cudaMallocHost((void**) &gp.h_dummy, (size_t) im_->blob), "h_dummy");
        std::memset(gp.h_dummy, 0, (size_t) im_->blob);
    }
    // a geometry whose blob size varies per layer opens its cache at the seed, sized to the ranked experts
    // A uniform geometry with an AUTO slot budget (slot_gib > 0, no --slots) opens at the seed too, sized by that
    // budget: there cfg.slots is the whole model (ds41_generate: n_layers * n_experts), so opening here asked for every
    // expert (all-Q3_K V4.1: 217.5 GiB) and failed tier init, and serve's release_cache() + seed never reopened it
    if (!cfg.no_cache && cfg.slots > 0 && !im_->mixed_sizes && !cfg.defer_cache && !(cfg.slot_gib > 0)) {   // defer_cache: opened at the seed
        std::vector<int64_t> sizes((size_t) cfg.slots, im_->blob);
        gp.cache.reset(new strata::core::ExpertCache());
        if (!gp.cache->open_sized(sizes, geom.n_layers, geom.n_experts, err)) return false;
    }
    // NO ARENA HERE.  The arena is 52-73 GiB of the real model and it is ORDERED (build_arena): building it twice -
    // once in index order here, once from the profile when the caller seeds - would read the file twice and page
    // the second read through the first's cache.  The first build wins (build_arena), and the first build is the
    // seed's; a caller that never seeds gets the file tier for every expert, which is slower but not wrong.
    im_->inited = true;
    return true;
#endif
}


/// The arena's ranking: with `arena_skip_resident`, minus the head the VRAM seed takes
/// (the same walk as the cache seed: ranked order, up to `slots` experts or the `slot_gib` bytes).
std::vector<std::pair<int32_t, int32_t>> arena_order(const Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    if (!(im.cfg.arena_skip_resident && !im.cfg.no_cache && im.cfg.slots > 0)) return ranked;
    const double budget = im.cfg.slot_gib > 0 ? im.cfg.slot_gib * 1073741824.0 : (double) im.cfg.slots * (double) im.blob;
    std::vector<char> res((size_t) (im.g.n_layers * im.g.n_experts), 0);
    double used = 0;
    int64_t n = 0;
    for (const auto& le : ranked) {
        if (n >= im.cfg.slots) break;
        if (!im.g.routed(le.first) || le.second < 0 || le.second >= im.g.n_experts) continue;
        char& r = res[(size_t) (le.first * im.g.n_experts + le.second)];
        if (r) continue;
        const int64_t b = im.bl[(size_t) im.g.cls(le.first, le.second)];
        if (used + (double) b > budget) break;
        r = 1;
        used += (double) b;
        ++n;
    }
    std::vector<std::pair<int32_t, int32_t>> rest;
    rest.reserve(ranked.size());
    for (const auto& le : ranked)
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts ||
            !res[(size_t) (le.first * im.g.n_experts + le.second)])
            rest.push_back(le);
    return rest;
}

/// STRATA_DS41_REMOTE: the experts the helper holds stay out of the arena (a verify/decode pass never reads them
/// from it, and their bytes hold the next experts of the ranking instead).  STRATA_DS41_REMOTE_DUP=1: the old
/// arena, which also holds them.
bool remote_skips_arena(const Ds41MoeImpl& im) {
#if defined(DS4_MOE_CUDA)
    static const bool dup = std::getenv("STRATA_DS41_REMOTE_DUP") && std::getenv("STRATA_DS41_REMOTE_DUP")[0] == '1';
    return !dup && im.gpu && im.gpu->remote;
#else
    (void) im;
    return false;
#endif
}

std::vector<std::pair<int32_t, int32_t>> arena_order_remote(const Ds41MoeImpl& im, std::vector<std::pair<int32_t, int32_t>> ranked) {
#if defined(DS4_MOE_CUDA)
    if (!remote_skips_arena(im)) return ranked;
    const uint8_t* holds = im.gpu->remote->s->holds;
    size_t n = 0;
    for (const auto& le : ranked)
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts ||
            !holds[(size_t) (le.first * im.g.n_experts + le.second)])
            ranked[n++] = le;
    ranked.resize(n);
#endif
    return ranked;
}

bool Ds41MoeTier::build_arena(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
#if !defined(DS4_MOE_CUDA)
    (void) ranked;
    (void) err;
    return true;   // no arena in a CPU-only tier: every blob comes from the source
#else
    if (!im_->gpu || im_->cfg.cpu_only) return true;
    Gpu& gp = *im_->gpu;
    const int64_t total = im_->g.n_layers * im_->g.n_experts;
    // An arena that already holds every expert has nothing to gain from another ordering, and a rebuild would
    // re-read up to 72 GiB of the model for no change in behaviour.  Neither has a re-seed with a ranking when the
    // arena was already built from one: the first ranking wins, exactly as it does for the cache, where `admit`
    // hands out the free slots to the first seed and never evicts.
    if (gp.arena.built && (gp.arena.used >= total || (gp.arena.from_ranked && !ranked.empty()))) return true;
    gp.arena.close();

    const double budget_gib = std::min(im_->cfg.arena_gib, im_->cfg.max_arena_gib);
    if (budget_gib <= 0) {
        gp.arena.built = true;
        return true;
    }
    const uint64_t cap_bytes = (uint64_t) (budget_gib * 1073741824.0);
    std::vector<std::pair<int32_t, int32_t>> order = ranked;
    if (order.empty()) {
        order.reserve((size_t) total);
        for (int64_t l = 0; l < im_->g.n_layers; ++l)
            for (int64_t e = 0; e < im_->g.n_experts; ++e)
                if (im_->g.routed(l)) order.push_back({(int32_t) l, (int32_t) e});
    }
    gp.arena.slot_of.assign((size_t) total, -1);
    gp.arena.off.assign(1, 0);
    std::vector<std::pair<int32_t, int32_t>> first;
    first.reserve(order.size());
    for (const auto& le : order) {
        const int64_t l = le.first, e = le.second;
        if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) continue;
        if (!im_->g.routed(l)) continue;
        if (gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] >= 0) continue;
        const uint64_t b = (uint64_t) im_->bl[(size_t) im_->g.cls(l, e)];
        if (gp.arena.off.back() + b > cap_bytes) {   // the budget in BYTES: blobs differ per layer (MiMo)
            ++gp.arena.file_tier;
            continue;
        }
        gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] = (int32_t) first.size();
        first.push_back({(int32_t) l, (int32_t) e});
        gp.arena.off.push_back(gp.arena.off.back() + b);
    }
    gp.arena.used = (int64_t) first.size();
    gp.arena.bytes = gp.arena.off.back();
    gp.arena.built = true;
    gp.arena.from_ranked = !ranked.empty();
    if (gp.arena.bytes == 0) {
        gp.arena.backing = "empty";
        return true;
    }
    const int64_t avail = mem_available_gib();
    const int64_t gib = (int64_t) (gp.arena.bytes / 1073741824ull);
    if (avail >= 0 && gib + (int64_t) im_->cfg.mem_floor_gib > avail) {
        err = "an arena of " + std::to_string(gib) + " GiB would leave less than " +
              std::to_string((int64_t) im_->cfg.mem_floor_gib) + " GiB free (MemAvailable " +
              std::to_string(avail) + " GiB)";
        gp.arena.close();
        return false;
    }
    // STRATA_ARENA_PIN_AFTER=1 (llm-server): fill first, register after.  PinnedArena registers the untouched mapping
    // and the kernel then faults and zeroes every page inside cudaHostRegister on one thread (72 GiB: 23.8 s at load,
    // ~3 GB/s); filled first, the faults are spread over the readers below and overlap the disk, and the registration
    // of resident pages is ~20 GB/s (pinbench, 16 GiB: register-then-fill 5.73 s, fill-then-register 2.58 s)
    bool pin_after = false;
#ifndef _WIN32
    if (im_->cfg.pin && std::getenv("STRATA_ARENA_PIN_AFTER") && std::getenv("STRATA_ARENA_PIN_AFTER")[0] == '1') {
        void* m = mmap(nullptr, (size_t) gp.arena.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            madvise(m, (size_t) gp.arena.bytes, MADV_NOHUGEPAGE);   // THP pins at ~0.16 GB/s (ds41-engine.sh's note)
            gp.arena.own_map = m;
            gp.arena.own_bytes = gp.arena.bytes;
            gp.arena.base = m;
            pin_after = true;
        }
    }
#endif
    if (pin_after) {
        // registered after the fill below
    } else if (im_->cfg.pin) {
        // STRATA_ARENA_SLICE_GIB=N (opt-in): when CUDA refuses to register the whole arena in one call (Windows: a
        // single block over ~50 GiB can fail while the same memory in 4 GiB pieces registers), pin it in N-GiB pieces
        // instead of leaving it unpinned.  The cuts fall on blob boundaries (a copy of one blob must not span two
        // registered regions).  Unset or 0: one registration, as before.
        const char* sl = std::getenv("STRATA_ARENA_SLICE_GIB");
        // #32: --remote-device under WDDM (Windows, WSL2) maps every pinned page into BOTH cards' contexts, and pinned
        // past ~48-56 GiB WDDM refuses every later allocation (5090 + 4080, 128 GB, driver 617.42: a 56 GiB pin left
        // cudaMemGetInfo at 0/0 and the conversation pool's stream "out of memory"; 40 and 48 GiB loaded with the pool
        // off).  The auto cap (#243) reads one card's shared budget (101.5 GiB there) and pins past it.  So, as
        // strata generate does for a multi-GPU run (#253, 8 GiB), the pin starts capped: 32 GiB leaves the default
        // 8 GiB conversation pool room under the 48 that loaded.  The rest of the arena stays locked in RAM (slower
        // streaming, not wrong).  STRATA_ARENA_PIN_GIB=N / auto / 0 (whole arena) overrides it.  On Windows the
        // separate-process helper (STRATA_DS41_REMOTE, #31) has no such limit and measured faster than both.
        uint64_t cap = 0;
        if (im_->cfg.remote_device >= 0 && strata::core::arena_pin_cap_gib() == -1 && ds41_under_wddm()) {
            cap = 32ull << 30;
            if (cap < gp.arena.bytes)
                std::fprintf(stderr, "arena: --remote-device under WDDM: at most %llu GiB of the arena is pinned (#32; "
                                     "STRATA_ARENA_PIN_GIB changes it; STRATA_DS41_REMOTE avoids the limit)\n",
                             (unsigned long long) (cap >> 30));
        }
        uint64_t slice = sl ? (uint64_t) std::max(0, std::atoi(sl)) << 30 : 0;
        if (cap && slice == 0) slice = 4ull << 30;   // a capped pin registers in pieces (PinnedArena needs the cuts)
        std::vector<uint64_t> cuts;
        if (slice > 0) {
            cuts.push_back(0);
            for (const uint64_t o : gp.arena.off)
                if (o >= cuts.back() + slice && o < gp.arena.bytes) cuts.push_back(o);
            cuts.push_back(gp.arena.bytes);
        }
        if (cuts.size() >= 3) gp.arena.owner.reset(new strata::core::PinnedArena(gp.arena.bytes, cuts, cap));
        else gp.arena.owner.reset(new strata::core::PinnedArena(gp.arena.bytes));
        if (!gp.arena.owner->valid()) {
            err = "pinned arena allocation of " + std::to_string(gib) + " GiB failed";
            gp.arena.close();
            return false;
        }
        gp.arena.base = gp.arena.owner->base;
        gp.arena.pinned = true;
        gp.arena.backing = "pinned (" + gp.arena.owner->note + ")";
        std::fprintf(stderr, "arena: %s\n", gp.arena.backing.c_str());
    } else {
        void* p = nullptr;
        if (ds41_sys::posix_memalign(&p, 4096, gp.arena.bytes) != 0 || !p) {
            err = "anonymous arena allocation failed";
            gp.arena.close();
            return false;
        }
        gp.arena.base = p;
        gp.arena.backing = "anonymous (pageable: the PCIe DMAs stage through the driver)";
    }
    int nt = im_->cfg.threads > 0 ? im_->cfg.threads : 8;
    nt = std::min<int>(nt, 16);
    const double t0 = now_ms();
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int c = 0; c < nt; ++c)
        th.emplace_back([&]() {
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= gp.arena.used) break;
                im_->blobs->read(first[(size_t) i].first, first[(size_t) i].second,
                                 (uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) i]);
            }
        });
    for (auto& t : th) t.join();
    if (pin_after) {
        const double tr = now_ms();
        if (cudaHostRegister(gp.arena.base, (size_t) gp.arena.bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) ==
            cudaSuccess) {
            gp.arena.own_reg = true;
            gp.arena.pinned = true;
            gp.arena.backing = "pinned after the fill (STRATA_ARENA_PIN_AFTER, 4 KB pages)";
        } else {
            (void) cudaGetLastError();
            gp.arena.backing = "anonymous (cudaHostRegister after the fill FAILED: the PCIe DMAs stage through the driver)";
        }
        std::fprintf(stderr, "arena: %s, filled in %.1f s, registered in %.1f s\n", gp.arena.backing.c_str(),
                     (tr - t0) / 1000.0, (now_ms() - tr) / 1000.0);
    }
    gp.arena.seconds = (now_ms() - t0) / 1000.0;
    return true;
#endif
}

bool Ds41MoeTier::seed_from_routes(const std::string& routes_bin, std::string& err) {
    return seed_from_routes(routes_bin, kBatch, err);
}

bool Ds41MoeTier::seed_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err) {
    std::string e;
    const std::vector<std::pair<int32_t, int32_t>> ranked =
        rank_routes(routes_bin, im_->g.n_routed(), im_->g.top_k, e, batch_tokens);
    if (ranked.empty()) {
        err = e.empty() ? "empty routing profile" : e;
        return false;
    }
    return seed_from_ranked(ranked, err);
}

/// `ranked` minus the pruned experts (Ds41MoeConfig::exclude)
std::vector<std::pair<int32_t, int32_t>> drop_excluded(const Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    const std::vector<char>& ex = im.cfg.exclude;
    if (ex.empty()) return ranked;
    std::vector<std::pair<int32_t, int32_t>> kept;
    kept.reserve(ranked.size());
    for (const auto& le : ranked) {
        const int64_t at = (int64_t) le.first * im.g.n_experts + le.second;
        if (at < 0 || at >= (int64_t) ex.size() || !ex[(size_t) at]) kept.push_back(le);
    }
    return kept;
}

/// warm_reseed: the order a seed fills the VRAM cache in, after a released cache.  The first `share` per mille of
/// `want` slots go to the experts released with a slot, most recently used first - those are the ones this
/// conversation routes to.  The rest, and everything when there is nothing warm, come from the global rank.
/// Consumes `im.warm` (a release always writes a fresh list).
std::vector<std::pair<int32_t, int32_t>> warm_first(Ds41MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t want) {
    if (im.warm.empty() || want <= 0) return ranked;
    const char* ws = std::getenv("STRATA_WARM_SHARE");
    const int64_t share = ws ? std::max(0, std::min(1000, std::atoi(ws))) : 500;
    if (share <= 0) { im.warm.clear(); return ranked; }
    const int64_t take = std::max<int64_t>(1, (want * share) / 1000);
    std::vector<char> seen((size_t) (im.g.n_layers * im.g.n_experts), 0);
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(ranked.size() + im.warm.size());
    int64_t n = 0;
    for (const auto& le : im.warm) {
        if (n >= take) break;
        if (!im.g.routed(le.first) || le.second < 0 || le.second >= im.g.n_experts) continue;
        char& sn = seen[(size_t) (le.first * im.g.n_experts + le.second)];
        if (sn) continue;
        sn = 1;
        out.push_back(le);
        ++n;
    }
    im.warm.clear();
    for (const auto& le : ranked) {
        if (!im.g.routed(le.first) || le.second < 0 || le.second >= im.g.n_experts) continue;
        char& sn = seen[(size_t) (le.first * im.g.n_experts + le.second)];
        if (sn) continue;
        sn = 1;
        out.push_back(le);
    }
    return out;
}

bool Ds41MoeTier::build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked_all, std::string& err) {
    if (im_->cfg.cpu_only) return true;
    const std::vector<std::pair<int32_t, int32_t>> ranked = drop_excluded(*im_, ranked_all);
    if (!remote_build(*im_, ranked, err)) return false;
    return build_arena(arena_order_remote(*im_, arena_order(*im_, drop_remote(*im_, ranked))), err);
}

void Ds41MoeTier::set_slot_gib(double gib) { im_->cfg.slot_gib = gib; }

void Ds41MoeTier::set_cache_enabled(bool on) {
    if (!im_) return;
#if defined(DS4_MOE_CUDA)
    // an open cache is release_cache()'s to drop, not a config flip's.  UNION (xolotl): the CPU flavour's stub Gpu
    // has no cache member, so this has to be guarded - the fork never hit it because its configure has no ggml-cpu
    // target and v41_engine.cmake only creates ds41_moe_cpu under `if(TARGET strata_kernels_cpu AND TARGET ggml-cpu)`;
    // the merged dense lineage enables ggml-cpu, which wakes the CPU flavour up.
    if (!on && im_->gpu && im_->gpu->cache) return;
#endif
    im_->cfg.no_cache = !on;
}

bool Ds41MoeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked_all, std::string& err) {
    if (im_->cfg.cpu_only) return true;   // a CPU-only tier has neither a cache nor an arena to seed
    if (!remote_build(*im_, drop_excluded(*im_, ranked_all), err)) return false;
    const std::vector<std::pair<int32_t, int32_t>> ranked = drop_remote(*im_, drop_excluded(*im_, ranked_all));
    if (!build_arena(arena_order_remote(*im_, arena_order(*im_, ranked)), err)) return false;
#if !defined(DS4_MOE_CUDA)
    (void) ranked;
    err = "the VRAM cache needs the CUDA tier (compiled without DS4_MOE_CUDA)";
    return false;
#else
    // A per-layer-sized geometry (MiMo) opens its cache now: one slot per ranked expert, each the size of ITS
    // layer's blob, until `slots` experts or `slot_gib` bytes.  The seed below admits in the same order, so slot i
    // is ranked[i]'s and fits it exactly; the cache never evicts, so a slot never meets another layer's blob.
    // slots=0 + slot_gib>0 (buster dual-V100): size the VRAM cache from free GiB, not a fake slot count.
    const bool want_cache = !im_->cfg.no_cache && (im_->cfg.slots > 0 || im_->cfg.slot_gib > 0);
    if (im_->mixed_sizes && im_->gpu && !im_->gpu->cache && want_cache) {
        const double budget = im_->cfg.slot_gib > 0 ? im_->cfg.slot_gib * 1073741824.0
                                                     : (double) im_->cfg.slots * (double) im_->blob;
        const int64_t max_slots = im_->cfg.slots > 0 ? im_->cfg.slots
                                                     : (int64_t) (budget / std::max<int64_t>(1, im_->blob)) + 4096;
        std::vector<int64_t> sizes;
        std::vector<char> seen((size_t) (im_->g.n_layers * im_->g.n_experts), 0);
        double used = 0;
        for (const auto& le : ranked) {
            if ((int64_t) sizes.size() >= max_slots) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            char& sn = seen[(size_t) (le.first * im_->g.n_experts + le.second)];
            if (sn) continue;
            const int64_t b = im_->bl[(size_t) im_->g.cls(le.first, le.second)];
            if (used + (double) b > budget) break;
            sn = 1;
            sizes.push_back(b);
            used += (double) b;
        }
        if (!sizes.empty()) {
            im_->gpu->cache.reset(new strata::core::ExpertCache());
            if (keep_cache_mode()) im_->gpu->cache->set_segment_bytes((int64_t) 64 << 20);   // shrink_cache_for
            if (!im_->gpu->cache->open_sized(sizes, im_->g.n_layers, im_->g.n_experts, err)) return false;
        }
    }
    // uniform blobs with defer_cache, or with an auto slot budget (#15), or slots=0 + slot_gib>0 (buster layer split):
    // the cache opens now (after the prompt chunks gave their VRAM back), at most `slots` experts (none = no cap) and
    // slot_gib bytes of whole 256-aligned slots (open_sized's padding)
    int64_t n_uni = im_->cfg.slots > 0 ? im_->cfg.slots : 0;
    if (im_->cfg.slot_gib > 0) {
        const int64_t by_gib = (int64_t) (im_->cfg.slot_gib * 1073741824.0 / (double) ((im_->blob + 255) / 256 * 256));
        n_uni = n_uni > 0 ? std::min<int64_t>(n_uni, by_gib) : by_gib;
    }
    if (!im_->mixed_sizes && (im_->cfg.defer_cache || im_->cfg.slot_gib > 0) && im_->gpu && !im_->gpu->cache &&
        want_cache && n_uni > 0) {
        std::vector<int64_t> sizes((size_t) n_uni, im_->blob);
        im_->gpu->cache.reset(new strata::core::ExpertCache());
        if (keep_cache_mode()) im_->gpu->cache->set_segment_bytes((int64_t) 64 << 20);
        if (!im_->gpu->cache->open_sized(sizes, im_->g.n_layers, im_->g.n_experts, err)) return false;
    }
    if (!im_->gpu || !im_->gpu->cache) return true;   // no cache: nothing to seed
    const int64_t want = im_->cfg.slots > 0
        ? std::min<int64_t>(im_->cfg.slots, im_->gpu->cache->full_slots())
        : im_->gpu->cache->full_slots();
    // the order the slots fill in.  A per-layer-sized cache (MiMo) keeps the rank: its sizes list above pairs slot
    // i with ranked[i].  Everywhere else a preceding release_cache() supplies the warm set.
    const std::vector<std::pair<int32_t, int32_t>> fill_order = im_->mixed_sizes ? ranked : warm_first(*im_, ranked, want);
    int64_t filled = 0;
    // vram_lru: the LRU's starting order = the fill order: the last-seeded expert goes out first
    auto lru_seed_order = [&](int64_t n_seeded) {
        if (!im_->cfg.vram_lru) return;
        const int64_t NX = im_->g.n_experts;
        im_->vlast.assign((size_t) (im_->g.n_layers * NX), -1);
        int64_t r = 0;
        for (const auto& le : fill_order) {
            if (r >= n_seeded) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= NX) continue;
            int64_t& v = im_->vlast[(size_t) (le.first * NX + le.second)];
            if (v != -1 || im_->gpu->cache->slot_of(le.first, le.second) < 0) continue;
            v = -2 - r++;   // rank 0 -> -2 (most recent of the seeds), the last seeded -> the oldest
        }
    };
    if (im_->mixed_sizes || !std::getenv("DS4_SERIAL_SEED")) {
        // MiMo (and DS4 since 2026-10-08: the serial seed below took 8 s for 2226 slots after chunked prefill): the
        // same admissions in the same order, but the copies overlap - arena blobs queued straight from the
        // pinned arena, file-tier blobs read by 8 threads into pinned buffers (the serial seed read one blob at a time:
        // ~7 s for ~1850 experts after a prompt).  Same slots, same bytes.
        strata::core::ExpertCache& cache = *im_->gpu->cache;
        if (im_->cfg.cuda_device >= 0 && cudaSetDevice(im_->cfg.cuda_device) != cudaSuccess) {
            err = "seed: cudaSetDevice failed";
            return false;
        }
        std::vector<std::array<int64_t, 3>> fjobs;   // slot, layer, expert
        for (const auto& le : fill_order) {
            if (filled >= want) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            if (cache.slot_of(le.first, le.second) >= 0) continue;
            const int32_t s = cache.admit(le.first, le.second);
            if (s < 0) break;
            if (const uint8_t* p = im_->gpu->arena.ptr(le.first, le.second)) {
                if (!cache.fill_slot_queued(s, p, err, im_->bl[(size_t) im_->g.cls(le.first, le.second)])) return false;
            } else {
                fjobs.push_back({s, le.first, le.second});
            }
            ++filled;
        }
        std::atomic<size_t> nx{0};
        std::atomic<bool> bad{false};
        std::mutex mu;
        std::string terr;
        std::vector<std::thread> th;
        const double t0 = now_ms();
        const int nth = (int) std::min<size_t>(8, fjobs.size());
        const int seed_dev = im_->cfg.cuda_device;
        for (int k = 0; k < nth; ++k)
            th.emplace_back([&, seed_dev]() {
                if (seed_dev >= 0 && cudaSetDevice(seed_dev) != cudaSuccess) {
                    bad = true;
                    std::lock_guard<std::mutex> lk(mu);
                    terr = "seed worker: cudaSetDevice failed";
                    return;
                }
                uint8_t* buf = nullptr;
                if (cudaMallocHost((void**) &buf, (size_t) im_->blob) != cudaSuccess) {
                    bad = true;
                    return;
                }
                for (size_t q; !bad && (q = nx.fetch_add(1)) < fjobs.size();) {
                    const auto& j = fjobs[q];
                    im_->blobs->read(j[1], j[2], buf);
                    std::lock_guard<std::mutex> lk(mu);   // the cache's counters; the copy is ~0.4 ms of a ~2 ms read
                    std::string e;
                    if (!cache.fill_slot_blocking((int32_t) j[0], buf, e, im_->bl[(size_t) im_->g.cls(j[1], j[2])])) {
                        terr = e;
                        bad = true;
                    }
                }
                cudaFreeHost(buf);
            });
        for (auto& t : th) t.join();
        if (cudaStreamSynchronize((cudaStream_t) 0) != cudaSuccess && !bad) {
            terr = "seed: queued fills failed";
            bad = true;
        }
        im_->st.file_tier += (int64_t) fjobs.size();
        im_->st.file_ms += now_ms() - t0;
        lru_seed_order(filled);
        if (bad) {
            err = terr.empty() ? "seed: pinned buffer allocation failed" : terr;
            return false;
        }
        im_->admitted = filled;
        return true;
    }
    for (const auto& le : fill_order) {
        if (filled >= want) break;
        if (im_->mixed_sizes) {   // keep slot i == the i-th distinct routed entry (the sizes list above)
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            if (im_->gpu->cache->slot_of(le.first, le.second) >= 0) continue;
        }
        const int32_t s = im_->gpu->cache->admit(le.first, le.second);
        if (s < 0) break;
        bool from_file = false;
        const uint8_t* p = acquire(*im_, le.first, le.second, 0, &from_file);
        if (!im_->gpu->cache->fill_slot_blocking(s, p, err, im_->bl[(size_t) im_->g.cls(le.first, le.second)])) return false;
        ++filled;
    }
    lru_seed_order(filled);
    im_->admitted = filled;
    return true;
#endif
}

void Ds41MoeTier::advise_file(int64_t layer, const int32_t* ids, int n) {
    if (!im_->inited || !im_->blobs || !ids || n <= 0 || layer < 0 || layer >= im_->g.n_layers || !im_->g.routed(layer)) return;
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= im_->g.n_experts || resident(layer, e) || in_arena(layer, e)) continue;
        im_->blobs->advise(layer, e);
        ++im_->st.advised;
    }
}

void Ds41MoeTier::trace_open(const std::string& path) {
    if (im_->trace) std::fclose(im_->trace);
    im_->trace = std::fopen(path.c_str(), "wb");
    if (!im_->trace) return;
    const int64_t L = im_->g.n_layers, X = im_->g.n_experts;
    const int32_t hdr[3] = { 'H', (int32_t) L, (int32_t) X };
    std::fwrite(hdr, 4, 3, im_->trace);
    std::vector<int32_t> v((size_t) (L * X * 3));
    for (int64_t l = 0; l < L; ++l)
        for (int64_t e = 0; e < X; ++e) {
            int32_t* r = v.data() + (size_t) (l * X + e) * 3;
            r[0] = im_->g.cls(l, e); r[1] = resident(l, e) ? 1 : 0; r[2] = in_arena(l, e) ? 1 : 0;
        }
    std::fwrite(v.data(), 4, v.size(), im_->trace);
}

void Ds41MoeTier::trace_mark(const float* v, int n) {
    if (!im_->trace) return;
    const int32_t hdr[2] = { 'P', n };
    std::fwrite(hdr, 4, 2, im_->trace);
    std::fwrite(v, 4, (size_t) n, im_->trace);
    std::fflush(im_->trace);
}

void Ds41MoeTier::spec_read(int64_t layer, const int32_t* ids, int n, bool d3) {
#if defined(DS4_MOE_CUDA)
    Ds41MoeImpl& im = *im_;
    // d3: the caller is the verify pass, whose FILEG path takes stashed blobs under arena_adapt too (gpu_run's
    // arena_adapt path reads into arena slots and never looks at the stash)
    if (!im.inited || !im.gpu || !im.blobs || im.cfg.cpu_only || (im.cfg.arena_adapt && !d3) || !ids || n <= 0) return;
    if (layer <= im.run_layer || layer >= im.g.n_layers || !im.g.routed(layer)) return;
    Gpu& gp = *im.gpu;
    if (!gp.spec) {
        gp.spec = std::make_unique<SpecStash>();
        if (!gp.spec->open(im.blobs, im.blob)) { gp.spec.reset(); return; }
    }
    const int64_t seq = im.run_seq + (layer - im.run_layer);   // decode: one gpu_run per layer
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= im.g.n_experts || resident(layer, e) || in_arena(layer, e)) continue;
        if (gp.spec->issue(layer, e, seq, im.run_seq)) ++im.st.spec_issued;
    }
#else
    (void) layer; (void) ids; (void) n;
#endif
}

void Ds41MoeTier::prefetch_multi(int64_t layer, const int32_t* ids, int n, int max_slots) {
    im_->pf_n = 0;
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || im_->cfg.cpu_only || im_->cfg.no_cache || !im_->gpu->cache) return;
    if (!ids || n <= 0 || max_slots <= 0 || !im_->g.routed(layer)) return;
    Gpu& gp = *im_->gpu;
    if (!gp.s_pf || !gp.d_pf) return;   // allocated with pf_b > 0 (the decode default)
    const int nb = std::min(max_slots, (int) Ds41MoeConfig::kMaxPf);
    for (int i = 0; i < n && im_->pf_n < nb; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= im_->g.n_experts) continue;
        if (gp.cache->slot_of(layer, e) >= 0) continue;
        bool dup = false;
        for (int j = 0; j < im_->pf_n; ++j) dup = dup || im_->pf_e[j] == e;
        if (dup) continue;
        const uint8_t* src = gp.arena.ptr(layer, e);
        if (!src) continue;   // file tier: no DMA source; the pass reads it on demand
        ck(cudaMemcpyAsync((uint8_t*) gp.d_pf + (size_t) im_->pf_n * (size_t) im_->blob, src,
                           (size_t) im_->bl[(size_t) im_->g.cls(layer, e)], cudaMemcpyHostToDevice, gp.s_pf), "prefetch dma");
        im_->pf_e[im_->pf_n++] = e;
    }
    if (im_->pf_n > 0) ck(cudaEventRecord(gp.ev_pf, gp.s_pf), "record prefetch");
    im_->st.prefetch_issued += im_->pf_n;
#else
    (void) layer; (void) ids; (void) n; (void) max_slots;
#endif
}

void Ds41MoeTier::prefetch(int64_t layer, const int32_t* top16, int n) {
    im_->pf_n = 0;
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || im_->cfg.cpu_only || im_->cfg.pf_b <= 0 || im_->cfg.no_cache || !im_->gpu->cache) return;
    if (!top16 || n <= 0 || !im_->g.routed(layer)) return;
    Gpu& gp = *im_->gpu;
    n = std::min(n, (int) Ds41MoeConfig::kPredW);
    int nb;
    if (im_->cfg.dither) {
        const double want = im_->cfg.pf_b + im_->pf_carry;
        nb = (int) std::floor(want);
        im_->pf_carry = want - std::floor(want);
    } else {
        nb = (int) std::lround(im_->cfg.pf_b);
    }
    nb = std::min(nb, (int) Ds41MoeConfig::kMaxPf);
    for (int i = 0; i < n && im_->pf_n < nb; ++i) {
        const int32_t e = top16[i];
        if (e < 0 || e >= im_->g.n_experts) continue;
        if (gp.cache->slot_of(layer, e) >= 0) continue;
        bool dup = false;
        for (int j = 0; j < im_->pf_n; ++j) dup = dup || im_->pf_e[j] == e;
        if (dup) continue;
        const uint8_t* src = gp.arena.ptr(layer, e);
        bool dummy = false;
        if (!src) {
            src = gp.h_dummy;   // a guess outside the arena still costs its DMA (the replay's own rule)
            ++im_->st.prefetch_dummy;
            dummy = true;
        }
        // No DMA for a guess outside the arena (its slot is -1 below either way, so routing and tokens are
        // unchanged) - the zero blob only kept the replay's timing, and it shares the H2D engine with the demand
        // misses.  The 2026-10-23 GPU profile (maintainer notes) showed the
        // H2D bus saturated by prefetch, so the skip is now the default; STRATA_PF_NO_DUMMY=0 restores the old
        // replay-exact timing (246 of 2,874 prefetches were dummies on that run).
        static const bool no_dummy = !(std::getenv("STRATA_PF_NO_DUMMY") && std::getenv("STRATA_PF_NO_DUMMY")[0] == '0');
        if (!(dummy && no_dummy))
            ck(cudaMemcpyAsync((uint8_t*) gp.d_pf + (size_t) im_->pf_n * (size_t) im_->blob, src,
                               (size_t) im_->bl[(size_t) im_->g.cls(layer, e)], cudaMemcpyHostToDevice, gp.s_pf), "prefetch dma");
        // A dummy slot holds ZEROS, not the expert: it must never be matched by run() (found by
        // tools/mimo/test_mimo_moe: prefetched experts outside the arena were computed from the zero blob, rel 0.2-0.5).
        // -1 matches no expert id; the DMA above still happens, so the replay-measured timing is unchanged.
        im_->pf_e[im_->pf_n++] = dummy ? -1 : e;
    }
    if (im_->pf_n > 0) ck(cudaEventRecord(gp.ev_pf, gp.s_pf), "record prefetch");
    im_->st.prefetch_issued += im_->pf_n;
#else
    (void) layer;
    (void) top16;
    (void) n;
#endif
}

// ---- the CPU path: the whole expert half, and the fallback for every miss the GPU does not take ----------

/// DS41_CHECK_POOL_BATCH=1, the bitwise determinism gate: recompute a layer's `nk` CPU experts (routing indices
/// `idx`) the pre-batching way - one `run_split_multi_native` per format class, `fmt` unset - into `im.parts`,
/// and assert every row is bitwise identical to the single-batch result already written there.  Same blobs,
/// same activation buffers (the per-`gu_act` ones, quantized again here), same kernels; only the pool's
/// park/drain scheduling differs, so a difference is a batching bug, not arithmetic.  False (and a printed row)
/// on the first mismatch.
bool check_pool_batch(Ds41MoeImpl& im, int64_t layer, const int32_t* ids6, const int32_t* idx, int nk,
                      const std::vector<const uint8_t*>& held, const float* x) {
    static bool announced = false;
    if (!announced) { announced = true; std::fprintf(stderr, "[pool-batch] per-class vs one-batch bitwise check active\n"); }
    const int64_t H = im.g.n_embd;
    std::vector<float> ref((size_t) nk * (size_t) H);
    for (int i = 0; i < nk; ++i)
        std::memcpy(ref.data() + (size_t) i * H, im.parts.data() + (size_t) idx[i] * H, (size_t) H * 4);
    std::vector<char> done((size_t) nk, 0);
    for (int a = 0; a < nk; ++a) {
        if (done[(size_t) a]) continue;
        const int c = im.g.cls(layer, ids6[idx[a]]);
        const cpu::NativeFmt& fc = im.fl[(size_t) c];
        if (cpu::q2_native_kernels(fc.gu_type)) cpu::act_quant_any(x, (int) fc.n_embd, im.actq);
        else cpu::native_quant_act(fc, x, nact_slot(im, fc.gu_act));
        int nj = 0;
        for (int b = a; b < nk; ++b) {
            if (done[(size_t) b] || im.g.cls(layer, ids6[idx[b]]) != c) continue;
            done[(size_t) b] = 1;
            cpu::ExpertJobMulti& j = im.jobs[(size_t) nj++];
            j.blob = held[(size_t) b];
            j.fmt = nullptr;   // the single-format path this check is against
            j.nt = 1;
            for (int t = 0; t < cpu::MAXT; ++t) { j.nact[t] = nullptr; j.out[t] = nullptr; }
            j.nact[0] = nact_slot(im, fc.gu_act);
            j.act[0] = &im.actq;
            j.out[0] = im.parts.data() + (size_t) idx[b] * H;
        }
        im.pool->run_split_multi_native(fc, im.jobs.data(), nj);
    }
    for (int i = 0; i < nk; ++i) {
        const float* g = im.parts.data() + (size_t) idx[i] * H;
        const float* r = ref.data() + (size_t) i * H;
        for (int64_t h = 0; h < H; ++h)
            if (std::memcmp(&g[h], &r[h], sizeof(float)) != 0) {
                std::fprintf(stderr, "[pool-batch] L%lld e%d differs at row %lld: batched %.9g, per-class %.9g\n",
                             (long long) layer, ids6[idx[i]], (long long) h, (double) g[h], (double) r[h]);
                return false;
            }
    }
    return true;
}

bool cpu_run(Ds41MoeImpl& im, int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out) {
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const double t0 = now_ms();
    std::vector<const uint8_t*> held((size_t) K, nullptr);
    for (int64_t k = 0; k < K; ++k) {
        bool from_file = false;
        held[(size_t) k] = acquire(im, layer, ids6[k], (int) k, &from_file);
    }
    // ONE pool dispatch for the whole layer: every class present goes into the same batch (each job carries its
    // own format), and the activation is quantized once per DISTINCT format - the Q2_0-native classes share
    // `actq`, every native class quantizes into the buffer of its `gu_act`.  Was one dispatch per class.
    const double c0 = now_ms();
    bool q2_done = false;
    std::vector<char> act_done(im.nact_multi_act.size(), 0);
    int nj = 0;
    for (int64_t k = 0; k < K; ++k) {
        const int c = im.g.cls(layer, ids6[k]);
        const cpu::NativeFmt& f = im.fl[(size_t) c];
        if (cpu::q2_native_kernels(f.gu_type)) {
            if (!q2_done) { cpu::act_quant_any(x, (int) f.n_embd, im.actq); q2_done = true; }
        } else {
            for (size_t b = 0; b < im.nact_multi_act.size(); ++b)
                if (im.nact_multi_act[b] == f.gu_act && !act_done[b]) {
                    cpu::native_quant_act(f, x, im.nact_multi.data() + b * cpu::kNativeActBytes);
                    act_done[b] = 1;
                }
        }
        cpu::ExpertJobMulti& jb = im.jobs[(size_t) nj++];
        jb.blob = held[(size_t) k];
        jb.fmt = &f;
        jb.nt = 1;
        for (int t = 0; t < cpu::MAXT; ++t) {
            jb.nact[t] = nullptr;
            jb.out[t] = nullptr;
        }
        jb.nact[0] = nact_slot(im, f.gu_act);
        jb.act[0] = &im.actq;
        jb.out[0] = im.parts.data() + (size_t) k * H;
    }
    im.pool->run_split_multi_native(im.fl[(size_t) im.g.cls(layer, ids6[0])], im.jobs.data(), nj);
    if (std::getenv("DS41_CHECK_POOL_BATCH")) {   // the bitwise gate: the pool's per-class path must agree
        std::vector<int32_t> ident((size_t) K);
        for (int64_t k = 0; k < K; ++k) ident[(size_t) k] = (int32_t) k;
        if (!check_pool_batch(im, layer, ids6, ident.data(), (int) K, held, x)) return false;
    }
    im.st.cpu_ms += now_ms() - c0;
    for (int64_t i = 0; i < H; ++i) {
        double s = 0;
        for (int64_t k = 0; k < K; ++k) s += (double) w6[k] * (double) im.parts[(size_t) (k * H + i)];
        out[i] = (float) s;
    }
    im.st.cpu += K;
    im.st.wall_ms += now_ms() - t0;
    return true;
}

#if defined(DS4_MOE_CUDA)

/// STRATA_TIER_GRAPH (slice B, docs/ds4/V41_DECODE_OVERLAP_PLAN.md §3): capture each (site, class) grouped launch
/// ONCE, over a fixed per-class metadata layout.  native_expert_grouped takes ptr/st/ng/dst/tok as device pointers,
/// reads the group/entry counts from device memory and launches a grid pinned by its host args, so captured nodes'
/// parameters never change - only the CONTENTS of the metadata block do, once a token.  The launcher gives every
/// ENTRY its own group (grp_start = 0..n), so ng = the class's entry count and the group loop strides by gridDim.y:
/// one group row still covers all of them and grid.x spreads the rows over the card, while the direct path's `n`
/// rows dispatched n-1 that iterated zero times.  Measured, that fixed one-row grid is within 0.6 ms/token of the
/// direct launches, against 3-4 ms/token for the single 9-class graph this first tried (see the report): a graph is
/// a FIXED kernel set, so "cheap when absent" cannot be a property of a node - only a per-class graph may be
/// skipped.  A class launch is three kernels (gate/up, SwiGLU+quantize, down) plus `ck`'s cudaGetLastError, so a
/// replay replaces six host calls with one.  Returns false, leaving the direct per-class path in place, when the
/// geometry has no fixed class set or CUDA refuses any part of the capture.
constexpr int kTierGraphGroups = 1;   ///< the fixed group rows per class (the kernels stride the real ng themselves)
static bool tier_graph_capture(Gpu& gp) {
    const int ncls = (int) gp.gll.size();
    const size_t slot = meta_off(0, kMaxParts).end;
    const size_t span = (size_t) ncls * slot;
    if (ncls <= 0 || ncls > kMetaClsMax || (size_t) kMetaSites * span + sizeof(float) * 64 > kMetaBytes) return false;
    cudaStream_t cs = nullptr;
    if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) return false;
    bool ok = true;
    for (int st = 0; st < kMetaSites && ok; ++st) {
        for (int c = 0; c < ncls && ok; ++c) {
            const uint8_t* dm = (const uint8_t*) gp.d_meta + (size_t) st * span + (size_t) c * slot;
            const MetaOff o = meta_off(0, kMaxParts);
            if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { ok = false; break; }
            strata::kernels::native_expert_grouped(
                gp.gll[(size_t) c], (const unsigned long long*) (dm + o.ptr), (const int32_t*) (dm + o.st),
                (const int32_t*) (dm + o.ng), (const int32_t*) (dm + o.dst), (const int32_t*) (dm + o.tok),
                kTierGraphGroups, kMaxParts, gp.d_xq, gp.d_scratch, (float*) gp.d_parts, cs, 0);
            cudaGraph_t g = nullptr;
            if (cudaStreamEndCapture(cs, &g) != cudaSuccess) { ok = false; break; }
            ok = cudaGraphInstantiate(&gp.tier_g[(size_t) st * ncls + (size_t) c], g, nullptr, nullptr, 0) == cudaSuccess;
            cudaGraphDestroy(g);
        }
    }
    cudaStreamDestroy(cs);
    if (!ok) {
        for (cudaGraphExec_t g : gp.tier_g)
            if (g) cudaGraphExecDestroy(g);
        for (cudaGraphExec_t& g : gp.tier_g) g = nullptr;
        return false;
    }
    gp.tier_graph_cls = ncls;
    gp.tier_site_span = span;
    gp.tier_w_off = (size_t) kMetaSites * span;
    return true;
}

/// One layer of the three-tier path.  The shape is `moe_replay.cpp`'s `Engine::layer`, with the replay's
/// synthetic dense gap and `--serial` wait removed: `run` is called AFTER the layer's dense work by contract, so
/// there is nothing to overlap the CPU tier against except the GPU tier, which is what runs beside it.
// DS4_TIER_PROF=1: where a layer's wall time goes inside gpu_run (summed, printed by close())
double g_prof[9] = {};
int64_t g_prof_n = 0;
const char* const g_prof_name[9] = {"split", "activation", "gpu launch", "cpu pool", "layer sync", "d2h parts",
                                    "sum", "admission", "pcie budget"};
/// arena_adapt: expert `e`'s bytes now sit in `victim`'s arena slot - hand the slot over (returns the slot).
/// arena_check_swap, DS41_CHECK_ARENA=1: the swapped-in expert's arena bytes AND the next slot's owner (what an overrun
/// would corrupt) against the file, byte for byte.  Shared by gpu_run (pool path) and gpu_run_n_ex (D3 FILEG path).
/// Publish ALL of a call's swaps before checking any (arena_check_swap): two victims in adjacent slots would otherwise
/// make the neighbour check read the second one's NEW bytes against its old owner - a false BAD.
int32_t arena_publish_swap(Ds41MoeImpl& im, int64_t layer, int32_t e, int32_t victim) {
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    const int32_t slot = gp.arena.slot_of[(size_t) (layer * NX + victim)];
    gp.arena.slot_of[(size_t) (layer * NX + victim)] = -1;
    gp.arena.slot_of[(size_t) (layer * NX + e)] = slot;
    ++im.st.arena_swaps;
    return slot;
}

void arena_check_swap(Ds41MoeImpl& im, int64_t layer, int32_t e, int32_t slot) {
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    static const bool check_arena = std::getenv("DS41_CHECK_ARENA") != nullptr;
    if (!check_arena) return;
    static int64_t n_chk = 0, n_bad = 0;
    std::vector<uint8_t> ref((size_t) im.blob);
    im.blobs->read(layer, e, ref.data());
    bool ok = std::memcmp((const uint8_t*) gp.arena.base + gp.arena.off[(size_t) slot], ref.data(),
                          (size_t) im.bl[(size_t) im.g.cls(layer, e)]) == 0;
    const int64_t total = (int64_t) gp.arena.slot_of.size();
    for (int64_t k = 0; k < total; ++k) {
        if (gp.arena.slot_of[(size_t) k] != slot + 1) continue;
        const int64_t nl = k / NX, ne = k % NX;
        im.blobs->read(nl, ne, ref.data());
        const bool okn = std::memcmp((const uint8_t*) gp.arena.base + gp.arena.off[(size_t) slot + 1], ref.data(),
                                     (size_t) im.bl[(size_t) im.g.cls(nl, ne)]) == 0;
        if (!okn) std::fprintf(stderr, "[arena-chk] L%lld e%d -> slot %d: NEIGHBOUR L%lld e%lld CORRUPT\n", (long long) layer, e, slot,
                               (long long) nl, (long long) ne);
        ok = ok && okn;
        break;
    }
    ++n_chk; n_bad += !ok;
    if (!ok) std::fprintf(stderr, "[arena-chk] L%lld e%d slot %d BAD\n", (long long) layer, e, slot);
    if (n_chk % 200 == 0) std::fprintf(stderr, "[arena-chk] %lld swaps checked, %lld bad\n", (long long) n_chk, (long long) n_bad);
}

/// arena_adapt: the layer's least-recently-used arena expert whose slot holds `need` bytes, not routed by this call
/// (last_use < tick) and not in `taken` - or -1.
int32_t arena_victim(Ds41MoeImpl& im, int64_t layer, uint64_t need, const int32_t* taken, int ntaken) {
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    int32_t victim = -1;
    int64_t best = INT64_MAX;
    for (int64_t v = 0; v < NX; ++v) {
        const int32_t vs = gp.arena.slot_of[(size_t) (layer * NX + v)];
        if (vs < 0) continue;
        if (gp.arena.off[(size_t) vs + 1] - gp.arena.off[(size_t) vs] < need) continue;
        const int64_t u = im.last_use[(size_t) (layer * NX + v)];
        if (u >= im.tick) continue;   // routed now
        bool t = false;
        for (int r = 0; r < ntaken && !t; ++r) t = taken[r] == (int32_t) v;
        if (t) continue;
        if (u < best) { best = u; victim = (int32_t) v; }
    }
    return victim;
}

bool gpu_run(Ds41MoeImpl& im, int64_t layer, const int32_t* ids6, const float* w6, const float* x_host,
             const void* x_dev, float* out, void* out_dev = nullptr) {
    static const bool prof = std::getenv("DS4_TIER_PROF") != nullptr;
    double tp = prof ? now_ms() : 0;
    auto mark = [&](int i) { if (prof) { const double t = now_ms(); g_prof[i] += t - tp; tp = t; } };
    Gpu& gp = *im.gpu;
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const int64_t B = im.blob;                        // staging STRIDE (the largest blob)
    auto BLX = [&](int32_t e) { return im.bl[(size_t) im.g.cls(layer, e)]; };   // an expert's blob (the bytes a DMA moves)
    const bool lru = im.cfg.vram_lru && gp.cache && !im.cfg.no_cache;   // uniform (DS4) too: every slot fits any blob
    ++im.run_seq;
    im.run_layer = layer;
    if (lru && !im.cfg.arena_adapt) ++im.tick;   // (arena_adapt bumps it below)
    if (im.cfg.arena_adapt) {   // every lookup of this layer is a use (the LRU the arena swaps below evict by)
        const int64_t NX = im.g.n_experts;
        if (im.last_use.empty()) im.last_use.assign((size_t) (im.g.n_layers * NX), -1);
        ++im.tick;
        for (int64_t k = 0; k < K; ++k) im.last_use[(size_t) (layer * NX + ids6[k])] = im.tick;
    }
    const double t0 = now_ms();
    Ds41MoeStats st;

    // ---- split the routed experts: resident, prefetched (also computed on the GPU), CPU, PCIe ----
    int32_t hit_i[kMaxParts], cpu_i[kMaxParts], pcie_i[kMaxParts], slot_i[kMaxParts] = {};
    unsigned long long pf_ptr[kMaxParts] = {};
    int nh = 0, nc = 0, np = 0, npf_hit = 0;
    for (int64_t k = 0; k < K; ++k) {
        const int32_t e = ids6[k];
        const int32_t s = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        slot_i[k] = s;
        int pf_at = -1;
        for (int j = 0; j < im.pf_n && s < 0; ++j)
            if (im.pf_e[j] == e) pf_at = j;
        if (s >= 0) {
            hit_i[nh++] = (int32_t) k;
        } else if (pf_at >= 0) {
            pf_ptr[k] = (unsigned long long) gp.d_pf + (size_t) pf_at * (size_t) B;
            hit_i[nh++] = (int32_t) k;
            ++npf_hit;
        } else {
            cpu_i[nc++] = (int32_t) k;
        }
    }
    st.prefetched_useful = npf_hit;
    // skip_miss: a miss with a small gate weight is dropped - its 6.75 MB never moves (wloc = the weights the sum uses)
    float wloc[kMaxParts];
    for (int64_t k = 0; k < K; ++k) wloc[k] = w6[k];
    if (im.cfg.skip_file > 0.0f && nc > 0) {   // file-tier misses only (not in the arena)
        double ws = 0;
        for (int64_t k = 0; k < K; ++k) ws += (double) w6[k];
        int keep = 0;
        for (int i = 0; i < nc; ++i) {
            const int32_t k = cpu_i[i];
            if (!gp.arena.ptr(layer, ids6[k]) && (double) w6[k] < (double) im.cfg.skip_file * ws) { wloc[k] = 0.0f; ++st.skipped; }
            else cpu_i[keep++] = k;
        }
        nc = keep;
    }
    if (im.cfg.skip_miss > 0.0f && nc > 0) {
        double ws = 0;
        for (int64_t k = 0; k < K; ++k) ws += (double) w6[k];
        int keep = 0;
        for (int i = 0; i < nc; ++i) {
            const int32_t k = cpu_i[i];
            if ((double) w6[k] < (double) im.cfg.skip_miss * ws) { wloc[k] = 0.0f; ++st.skipped; }
            else cpu_i[keep++] = k;
        }
        nc = keep;
    }
    mark(0);

    // The PCIe share of the misses, with the replay's error-diffused budget AND its eligibility rule: the DMA
    // reads straight out of the pinned host arena, so a blob the arena does not hold (the file tier) stays on the
    // CPU rather than being staged through the driver - which is what the replay measured, and the reason its
    // PCIe percentage is a percentage of what the arena held.
    {
        int eligible = 0;
        for (int i = 0; i < nc; ++i)
            if (gp.arena.ptr(layer, ids6[cpu_i[i]])) ++eligible;
        int budget;
        if (im.cfg.dither) {
            const double want = (double) eligible * im.cfg.pcie_frac + im.pcie_carry;
            budget = (int) std::floor(want);
            im.pcie_carry = want - budget;
        } else {
            budget = (int) std::lround((double) eligible * im.cfg.pcie_frac);
        }
        for (int i = 0; i < nc && np < budget; ++i) {
            const int k = cpu_i[nc - 1 - i];
            if (gp.arena.ptr(layer, ids6[k])) pcie_i[np++] = (int32_t) k;
        }
    }
    mark(8);
    int32_t cpu_keep[kMaxParts], nk = 0;
    bool cpu_k[kMaxParts] = {};   ///< routing index -> the pool computed it (the sum's source selector)
    for (int i = 0; i < nc; ++i) {
        bool in_pcie = false;
        for (int j = 0; j < np; ++j) in_pcie = in_pcie || pcie_i[j] == cpu_i[i];
        if (!in_pcie) {
            cpu_keep[nk++] = cpu_i[i];
            cpu_k[cpu_i[i]] = true;
        }
    }

    // ---- the activation, on both sides: Q8_K for the pool, q8_1 for the card ----
    // The host copy feeds only the CPU pool (nk > 0) and the two diagnostics.  Without them the card quantizes x_dev
    // where it lies (attn_router's output, already synchronized; this call's layer sync ends before the next graph can
    // rewrite it): no D2H, stream sync and H2D on ~39 of 40 layers a token.  The same bytes -> the same q8_1, bit for
    // bit.  STRATA_X_HOST=1 = always through the host (the old path).
    static const bool x_host_always = std::getenv("STRATA_X_HOST") || std::getenv("DS41_CHECK_POOL_BATCH") ||
                                      std::getenv("DS4_CHECK_GPU");
    if (x_dev && nk == 0 && !x_host_always) {
        strata::kernels::native_quantize_q8_1((const float*) x_dev, gp.d_xq, (int) H, 1, gp.s);
    } else {
        if (x_dev) {
            ck(cudaMemcpyAsync(gp.h_x, x_dev, (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "x d2h");
            ck(cudaStreamSynchronize(gp.s), "x sync");   // the pool needs it on the host, now
        } else {
            std::memcpy(gp.h_x, x_host, (size_t) H * 4);
        }
        ck(cudaMemcpyAsync(gp.d_x, gp.h_x, (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "x h2d");
        strata::kernels::native_quantize_q8_1((const float*) gp.d_x, gp.d_xq, (int) H, 1, gp.s);
    }
    ck(cudaEventRecord(gp.ev[0], gp.s), "record activation");
    mark(1);

    // STRATA_TIER_GRAPH: capture the three sites' class launches once, on the first layer that gets here - gp.gll
    // and the env-selected kernels are final by now.  Declared before the launch lambdas because they capture it by
    // reference.  Default OFF (nothing set), like STRATA_DMA_OVERLAP; the direct per-class path stays when CUDA
    // refuses the capture (a geometry without a fixed class set, or a non-capturable call under it).
    static const bool tier_graph_on = [] {
        const char* v = std::getenv("STRATA_TIER_GRAPH");
        return v != nullptr && v[0] != '0';
    }();
    if (tier_graph_on && !gp.tier_graph_ready && !gp.tier_graph_off) {
        if (tier_graph_capture(gp)) gp.tier_graph_ready = true;
        else {
            gp.tier_graph_off = true;
            std::fprintf(stderr, "ds4_moe: STRATA_TIER_GRAPH: CUDA refused the capture; keeping the per-class launches\n");
        }
    }
    const bool grp_graph = gp.tier_graph_ready;

    // ---- one group's metadata, then the grouped kernel: site 0 = the GPU's hits, 1 = the PCIe share, 2 = the file
    // misses.  One launch per format class among the entries (dst = routing index -> the expert's class); the metadata
    // buffers are reused between launches on the same stream, so they are uploaded per launch, in order
    // the five metadata arrays of a launch go into one region of the pinned h_meta block and up in ONE async copy
    // (they were five pageable copies per launch, ~20 per layer); regions are never reused within this call, and the
    // next call starts after this one's layer sync, so no copy can read a region that is being rewritten
    size_t meta_at = 0;
    // The five metadata arrays of ONE class sit in a region of the pinned h_meta block, laid out at `b` (meta_off) -
    // and under STRATA_TIER_GRAPH every class has its own FIXED region at that same layout, capacity kMaxParts.
    auto moff = [](size_t b, int n) { return meta_off(b, n); };
    auto pack_group1 = [&](int n, const std::vector<unsigned long long>& ptr, const std::vector<int32_t>& start,
                           const std::vector<int32_t>& dst, const std::vector<int32_t>& tok) -> size_t {
        const MetaOff o = moff(meta_at, n);
        const int32_t one = n;
        std::memcpy(gp.h_meta + o.ptr, ptr.data(), sizeof(unsigned long long) * (size_t) n);
        std::memcpy(gp.h_meta + o.st, start.data(), sizeof(int32_t) * (size_t) (n + 1));
        std::memcpy(gp.h_meta + o.ng, &one, sizeof(int32_t));
        std::memcpy(gp.h_meta + o.dst, dst.data(), sizeof(int32_t) * (size_t) n);
        std::memcpy(gp.h_meta + o.tok, tok.data(), sizeof(int32_t) * (size_t) n);
        meta_at = o.end;
        return o.ptr;
    };
    auto launch_group1 = [&](const strata::kernels::NativeExpertLayout& GL, int n, size_t off) {
        const MetaOff o = moff(off, n);
        uint8_t* dm = (uint8_t*) gp.d_meta;
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) (dm + o.ptr),
                                              (const int32_t*) (dm + o.st), (const int32_t*) (dm + o.ng),
                                              (const int32_t*) (dm + o.dst), (const int32_t*) (dm + o.tok),
                                              kMaxParts, n, gp.d_xq, gp.d_scratch, (float*) gp.d_parts, gp.s, n);
    };
    // Group the entries by format class (one kernel launch per class), but pack EVERY class's metadata into the
    // block first and copy the whole span up in ONE cudaMemcpyAsync - the old shape copied per class, ~7 extra
    // host calls a layer.  The classes read disjoint regions and gp.s orders them after the copy, so this is
    // bit-identical: same kernels, same bytes, same order.  STRATA_GRP_ONE_COPY=0 = the per-class copy.
    static const bool grp_one_copy = !std::getenv("STRATA_GRP_ONE_COPY") || std::atoi(std::getenv("STRATA_GRP_ONE_COPY")) != 0;
    auto launch_group = [&](int site, int n, const std::vector<unsigned long long>& ptr,
                            const std::vector<int32_t>& start, const std::vector<int32_t>& dst,
                            const std::vector<int32_t>& tok) {
        (void) start;
        if (n <= 0) return;   // a site with nothing to run is skipped: all nine of its classes are idle anyway
        struct Cls { int c = -1; int n = 0; size_t off = 0; std::vector<unsigned long long> p2;
                     std::vector<int32_t> d2, t2, s2; };
        std::vector<Cls> gs;
        int done = 0;
        bool used[kMaxParts] = {};
        while (done < n) {
            Cls g;
            for (int i = 0; i < n; ++i) {
                if (used[i]) continue;
                const int ci = im.g.cls(layer, ids6[dst[(size_t) i]]);
                if (g.c < 0) g.c = ci;
                if (ci != g.c) continue;
                used[i] = true;
                g.s2.push_back((int32_t) g.p2.size());
                g.p2.push_back(ptr[(size_t) i]);
                g.d2.push_back(dst[(size_t) i]);
                g.t2.push_back(tok[(size_t) i]);
            }
            g.s2.push_back((int32_t) g.p2.size());
            g.n = (int) g.p2.size();
            done += g.n;
            gs.push_back(std::move(g));
        }
        if (grp_graph) {
            // Slice B: this site's block is FIXED - one slot per class (kMetaClsMax), each laid out for kMaxParts
            // entries, which is exactly what the capture used, so a class's slot is a stable address for the graph.
            // Only the classes the routing produced are written AND launched: an unrouted class costs nothing here
            // (a graph is a fixed kernel set, so cheap-or-absent cannot be a property of a node - see the note on
            // tier_g).  One cudaMemcpyAsync of the site's whole block, then one cudaGraphLaunch per class.
            const size_t span = gp.tier_site_span;
            const size_t base = (size_t) site * span;
            const size_t slot = moff(0, kMaxParts).end;
            for (const Cls& g : gs) {
                const MetaOff o = moff(base + (size_t) g.c * slot, kMaxParts);
                std::memcpy(gp.h_meta + o.ptr, g.p2.data(), sizeof(unsigned long long) * (size_t) g.n);
                std::memcpy(gp.h_meta + o.st, g.s2.data(), sizeof(int32_t) * (size_t) (g.n + 1));
                std::memcpy(gp.h_meta + o.ng, &g.n, sizeof(g.n));
                std::memcpy(gp.h_meta + o.dst, g.d2.data(), sizeof(int32_t) * (size_t) g.n);
                std::memcpy(gp.h_meta + o.tok, g.t2.data(), sizeof(int32_t) * (size_t) g.n);
            }
            ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + base, gp.h_meta + base, span, cudaMemcpyHostToDevice, gp.s),
               "tier graph meta");
            for (const Cls& g : gs)
                ck(cudaGraphLaunch(gp.tier_g[(size_t) site * (size_t) gp.tier_graph_cls + (size_t) g.c], gp.s),
                   "tier graph launch");
            return;
        }
        bool one_copy = grp_one_copy;
        if (one_copy) {
            size_t end = meta_at;   // top-k 6 -> <= 12 launches: the block always fits
            for (const Cls& g : gs) end = moff(end, g.n).end;
            if (end > kMetaBytes) {   // cannot happen with top-k 6; wrap once, and if it STILL does not fit
                ck(cudaStreamSynchronize(gp.s), "meta wrap");   // fall back to the per-class path below -
                meta_at = 0;                                    // never drop a group
                end = 0;
                for (const Cls& g : gs) end = moff(end, g.n).end;
                one_copy = end <= kMetaBytes;
            }
        }
        if (one_copy) {
            const size_t lo = meta_at;
            for (Cls& g : gs) g.off = pack_group1(g.n, g.p2, g.s2, g.d2, g.t2);
            const size_t hi = meta_at;
            ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + lo, gp.h_meta + lo, hi - lo, cudaMemcpyHostToDevice, gp.s),
               "grp meta");
            for (const Cls& g : gs) launch_group1(gp.gll[(size_t) g.c], g.n, g.off);
        } else {
            for (Cls& g : gs) {
                if (moff(meta_at, g.n).end > kMetaBytes) {   // one class at a time: wrap like the old code did
                    ck(cudaStreamSynchronize(gp.s), "meta wrap");
                    meta_at = 0;
                }
                g.off = pack_group1(g.n, g.p2, g.s2, g.d2, g.t2);
                ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + g.off, gp.h_meta + g.off, moff(g.off, g.n).end - g.off,
                                   cudaMemcpyHostToDevice, gp.s), "grp meta");
                launch_group1(gp.gll[(size_t) g.c], g.n, g.off);
            }
        }
    };

    int ev_hit = -1, ev_pcie = -1;
    if (nh > 0) {
        std::vector<unsigned long long> ptr((size_t) nh);
        std::vector<int32_t> start((size_t) nh + 1), dst((size_t) nh, 0), tok((size_t) nh, 0);
        for (int i = 0; i < nh; ++i) {
            ptr[(size_t) i] = pf_ptr[hit_i[i]] ? pf_ptr[hit_i[i]]
                                               : (unsigned long long) gp.cache->device_slot(slot_i[hit_i[i]]);
            start[(size_t) i] = i;
            dst[(size_t) i] = hit_i[i];
        }
        start[(size_t) nh] = nh;
        if (npf_hit > 0) ck(cudaStreamWaitEvent(gp.s, gp.ev_pf, 0), "wait prefetch");
        launch_group(0, nh, ptr, start, dst, tok);
        ck(cudaEventRecord(gp.ev[2], gp.s), "record hits");
        ev_hit = 2;
    }
    // ---- vram_lru: the layer's least-recently-used resident expert not routed to now gives up its slot ----
    const int64_t NXL = im.g.n_experts;
    int32_t lru_new[2 * kMaxParts], lru_old[2 * kMaxParts], lru_dem[2 * kMaxParts];   // dem: arena slot or -1
    int n_lru = 0;
    // Exclusive tiers: a victim with no arena copy (the seed leaves VRAM experts out of the arena) is DEMOTED into the
    // arena slot the incoming expert vacates - else its next use would be a file read (FINDINGS s16: 5.9 -> 16.6
    // file reads/token without this).  Both copies run on gp.s: the D2H after the incoming blob left the arena slot,
    // the slot write after the D2H.
    auto arena_slot = [&](int32_t e) { return gp.arena.base ? gp.arena.slot_of[(size_t) (layer * NXL + e)] : -1; };
    auto demote = [&](int32_t victim_slot, int32_t into_arena, int64_t bytes, cudaStream_t st) {
        // reads the victim cache slot: it MUST be on the same stream as whoever overwrites that slot afterwards (the
        // lru pf copy, or the pcie group's stage copy under STRATA_DMA_OVERLAP), or the demoted blob is corrupted.
        ck(cudaMemcpyAsync((uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) into_arena],
                           gp.cache->device_slot(victim_slot), (size_t) bytes, cudaMemcpyDeviceToHost, st), "lru demote");
    };
    auto lru_victim = [&](int32_t incoming) -> int32_t {
        int32_t v = -1;
        int64_t best = INT64_MAX;
        const int ci = im.g.cls(layer, incoming);
        for (int64_t e = 0; e < NXL; ++e) {
            if (gp.cache->slot_of(layer, e) < 0) continue;
            if (im.g.cls(layer, e) != ci) continue;   // same class = same slot size
            const int64_t u = im.vlast[(size_t) (layer * NXL + e)];
            if (u >= im.tick) continue;   // used by this call (a hit) or already taken by it
            if (u < best) { best = u; v = (int32_t) e; }
        }
        return v;
    };
    // admission filter (cache_sim: admit-all 61.6% hit at 144 swaps/token; "second miss within 4 tokens" 63.0% at
    // 24.6): an expert takes a slot only if it also missed in one of the layer's previous kLruWin calls
    static const int64_t kLruWin = std::getenv("STRATA_LRU_WIN") ? std::atoll(std::getenv("STRATA_LRU_WIN")) : 4;   // admission window (calls)
    int64_t lc = 0;
    auto admit_ok = [&](int32_t e) {
        const int64_t m = im.vmiss[(size_t) (layer * NXL + e)];
        return m >= 0 && lc - m <= kLruWin;
    };
    if (lru) {
        if (im.vlast.empty()) im.vlast.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.vmiss.empty()) im.vmiss.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.lcalls.empty()) im.lcalls.assign((size_t) im.g.n_layers, 0);
        lc = ++im.lcalls[(size_t) layer];
        for (int64_t k = 0; k < K; ++k) im.vlast[(size_t) (layer * NXL + ids6[k])] = im.tick;
        // prefetched experts the routing used: device-to-device into a victim slot (the hits kernel above read the
        // staging copy; this copy only reads it too, and the next prefetch waits for the layer sync)
        for (int i = 0; i < nh; ++i) {
            const int k = hit_i[i];
            if (!pf_ptr[k] || !admit_ok(ids6[k])) continue;
            const int32_t v = lru_victim(ids6[k]);
            if (v < 0) continue;
            const int32_t sl = gp.cache->slot_of(layer, v);
            int32_t dem = -1;
            if (arena_slot(v) < 0) {
                dem = arena_slot(ids6[k]);   // prefetches come from the arena (never the dummy: pf_e = -1 then)
                if (dem < 0) continue;
                demote(sl, dem, BLX(v), gp.s);
            }
            ck(cudaMemcpyAsync(gp.cache->device_slot(sl), (const void*) pf_ptr[k], (size_t) BLX(ids6[k]), cudaMemcpyDeviceToDevice,
                               gp.s), "lru pf copy");
            im.vlast[(size_t) (layer * NXL + v)] = im.tick;   // not again in this call
            lru_old[n_lru] = v;
            lru_dem[n_lru] = dem;
            lru_new[n_lru++] = ids6[k];
        }
    }
    if (np > 0) {
        // STRATA_DMA_OVERLAP (see the Gpu::s_dc comment): the copies go on the demand-copy stream so they overlap the
        // resident-expert kernels already launched on gp.s, then ONE event orders gp.s before this group's kernels.
        // Sources are this call's arena experts and destinations are d_stage (or their own victim slot); the pool's
        // readers only write victims, and a victim is never routed this call, so the two sets are disjoint.
        const cudaStream_t cs = gp.s_dc ? gp.s_dc : gp.s;
        std::vector<unsigned long long> ptr((size_t) np);
        std::vector<int32_t> start((size_t) np + 1), dst((size_t) np, 0), tok((size_t) np, 0);
        for (int i = 0; i < np; ++i) {
            const uint8_t* p = gp.arena.ptr(layer, ids6[pcie_i[i]]);
            uint8_t* to = (uint8_t*) gp.d_stage + (size_t) i * (size_t) B;
            int32_t v = lru && admit_ok(ids6[pcie_i[i]]) ? lru_victim(ids6[pcie_i[i]]) : -1;
            const int64_t BL = BLX(ids6[pcie_i[i]]);
            if (v >= 0 && arena_slot(v) < 0 && arena_slot(ids6[pcie_i[i]]) < 0) v = -1;   // nowhere to demote to
            if (v >= 0) {
                uint8_t* vs = gp.cache->device_slot(gp.cache->slot_of(layer, v));
                im.vlast[(size_t) (layer * NXL + v)] = im.tick;
                int32_t dem = -1;
                if (arena_slot(v) < 0) {   // demote: arena -> stage, victim -> arena slot, stage -> victim slot
                    dem = arena_slot(ids6[pcie_i[i]]);
                    ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");
                    demote(gp.cache->slot_of(layer, v), dem, BL, cs);
                    ck(cudaMemcpyAsync(vs, to, (size_t) BL, cudaMemcpyDeviceToDevice, cs), "lru stage copy");
                } else {   // the DMA lands in the victim slot: the kernel below reads it there, and it stays
                    to = vs;
                    ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");
                }
                lru_old[n_lru] = v;
                lru_dem[n_lru] = dem;
                lru_new[n_lru++] = ids6[pcie_i[i]];
            } else {
                ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");
            }
            ptr[(size_t) i] = (unsigned long long) to;
            start[(size_t) i] = i;
            dst[(size_t) i] = pcie_i[i];
        }
        start[(size_t) np] = np;
        if (gp.s_dc) {   // this group's bytes must be home before its kernels run - one wait covers the whole group
            ck(cudaEventRecord(gp.ev_dc, cs), "record demand copies");
            ck(cudaStreamWaitEvent(gp.s, gp.ev_dc, 0), "wait demand copies");
        }
        launch_group(1, np, ptr, start, dst, tok);
        ck(cudaEventRecord(gp.ev[3], gp.s), "record pcie");
        ev_pcie = 3;
    }

    // ---- file-tier misses (in neither VRAM nor the RAM arena) on the card: read with O_DIRECT into a pinned staging
    // block on reader threads WHILE the pool computes the arena misses below, then DMA'd into free d_stage slots and
    // run by the grouped kernel - the pool took ~1.4 ms per such expert after its ~2 ms read, the DMA ~0.35 ms.
    // STRATA_FILE_GPU=0: the pool computes them (the old path); off with arena_adapt (it reads into arena slots).
    static const bool file_gpu = !(std::getenv("STRATA_FILE_GPU") && std::getenv("STRATA_FILE_GPU")[0] == '0');
    int32_t fg_k[kMaxParts];
    int nfg = 0;
    if (file_gpu && !im.cfg.arena_adapt && nk > 0) {
        int keep = 0;
        for (int i = 0; i < nk; ++i) {
            const int32_t k = cpu_keep[i];
            if (!gp.arena.ptr(layer, ids6[k]) && np + nfg < kMaxParts) {
                fg_k[nfg++] = k;
                cpu_k[k] = false;   // the sum reads its row from the card's parts
            } else {
                cpu_keep[keep++] = k;
            }
        }
        nk = keep;
    }
    std::vector<std::thread> frd;
    std::atomic<size_t> fnx{0};
    double tfg = 0;
    const uint8_t* fsrc[kMaxParts] = {};   // where each file-tier miss's bytes are: a spec_read slot or h_fstage
    int fread_b[kMaxParts], nfr = 0;       // the ones read now
    if (nfg > 0) {
        if (!gp.h_fstage) ck(cudaMallocHost((void**) &gp.h_fstage, (size_t) kMaxFg * (size_t) B), "h_fstage");
        tfg = now_ms();
        for (int b = 0; b < nfg; ++b) {
            fsrc[b] = gp.spec ? gp.spec->take(layer, ids6[fg_k[b]], im.run_seq) : nullptr;
            if (fsrc[b]) ++im.st.spec_hits;
            else { fsrc[b] = gp.h_fstage + (size_t) b * (size_t) B; fread_b[nfr++] = b; }
        }
        const int parts = nfr > 0 ? std::max(1, std::min(8, 8 / nfr)) : 1;
        const size_t tasks = (size_t) nfr * (size_t) parts;
        const int nth = (int) std::min<size_t>(8, tasks);
        for (int q = 0; q < nth; ++q)
            frd.emplace_back([&, parts, tasks]() {
                for (size_t r; (r = fnx.fetch_add(1)) < tasks;) {
                    const int b = fread_b[r / (size_t) parts];
                    im.blobs->read_part(layer, ids6[fg_k[b]], (int) (r % (size_t) parts), parts,
                                        gp.h_fstage + (size_t) b * (size_t) B);
                }
            });
    }

    mark(2);
    // ---- the CPU tier on what is left, WHILE the card works ----
    if (nk > 0) {
        std::vector<const uint8_t*> held((size_t) nk, nullptr);
        // arena blobs by address; file-tier blobs read IN PARALLEL (an NVMe needs several reads in flight - serial
        // 9 MB reads measured 1.6-5.3 ms each on MiMo, FINDINGS s9/s10) into their own ftmp slots - same bytes as
        // acquire() reads one by one
        std::vector<int> fi;
        for (int i = 0; i < nk; ++i) {
            held[(size_t) i] = gp.arena.ptr(layer, ids6[cpu_keep[i]]);
            if (!held[(size_t) i]) fi.push_back(i);
        }
        // ONE pool dispatch for the whole layer (see the join below): the batch used to be issued once per format
        // class, twice (arena-held, then file-tier) with the file reads overlapping the first pass.  The card
        // kernels launched above overlap this entire section either way, and the read wall (1-5 ms, latency-bound,
        // on the critical path) dwarfs the pool (sub-ms), so the trade is the collapse of 2 x classes park/drain
        // cycles into one for the small arena-vs-read overlap - and one activation quant per distinct format
        // instead of per class.
        std::vector<std::thread> rd;
        std::atomic<size_t> nx{0};
        double tf = 0;
        // File-tier blobs: with arena_adapt each is read STRAIGHT into the arena slot it will take (the layer's
        // least-recently-used arena expert not routed now) - no scratch copy - and one blob is split over up to 8
        // threads (read_part), since one 9 MB read alone ran ~3.3 ms (FINDINGS s16).  Same bytes as acquire().
        const int64_t NX = im.g.n_experts;
        const bool direct = im.cfg.arena_adapt && gp.arena.base && !im.last_use.empty();
        std::vector<uint8_t*> fdst(fi.size(), nullptr);
        std::vector<int32_t> fvic(fi.size(), -1);
        if (!fi.empty()) {
            if (im.ftmp.size() < (size_t) nk * (size_t) im.blob) im.ftmp.resize((size_t) nk * (size_t) im.blob);
            if (direct && gp.s_pf) ck(cudaStreamSynchronize(gp.s_pf), "pf sync");   // no prefetch DMA reads a victim
            // ... and no async copy from an EARLIER call touches a victim slot: the decode path skips its per-layer sync
            // (layer_synced), so a vram_lru demotion (D2H into an arena slot) or a PCIe DMA (reading one) can still be in
            // flight - and the reads below write arena memory from CPU threads, outside stream order.  Found by
            // DS41_CHECK_ARENA: 1 bad swap in 2200 (the demoted expert's late D2H landing on the new one).
            if (direct) {
                ck(cudaStreamSynchronize(gp.s), "arena-adapt sync");
                if (gp.s_dc) ck(cudaStreamSynchronize(gp.s_dc), "arena-adapt dc sync");
            }
            for (size_t q = 0; q < fi.size(); ++q) {
                fdst[q] = im.ftmp.data() + (size_t) fi[q] * (size_t) im.blob;
                if (!direct) continue;
                // The victim's SLOT must hold the incoming blob: arena slots are packed at each seeded expert's own blob
                // size, and DS4's 9 format classes differ in size - without this a larger blob read into a smaller slot
                // ran over into the next slot and corrupted that expert (garbage output under --arena-adapt).
                const int32_t victim = arena_victim(im, layer, (uint64_t) BLX(ids6[cpu_keep[fi[q]]]), fvic.data(), (int) q);
                if (victim < 0) continue;
                fvic[q] = (int32_t) victim;
                fdst[q] = (uint8_t*) gp.arena.base +
                          (size_t) gp.arena.off[(size_t) gp.arena.slot_of[(size_t) (layer * NX + victim)]];
            }
            tf = now_ms();
            const int parts = (int) std::max<size_t>(1, std::min<size_t>(8, 8 / fi.size()));
            const size_t tasks = fi.size() * (size_t) parts;
            const int nth = (int) std::min<size_t>(8, tasks);
            for (int q = 0; q < nth; ++q)
                rd.emplace_back([&, parts, tasks]() {
                    for (size_t r; (r = nx.fetch_add(1)) < tasks;) {
                        const size_t b = r / (size_t) parts;
                        im.blobs->read_part(layer, ids6[cpu_keep[fi[b]]], (int) (r % (size_t) parts), parts, fdst[b]);
                    }
                });
        }
        for (auto& t : rd) t.join();   // every CPU expert's bytes are home before the one batch goes out
        if (!fi.empty()) {
            im.st.file_tier += (int64_t) fi.size();
            im.st.file_ms += now_ms() - tf;   // the read wall, no longer overlapped with a first per-class batch
            static const bool check_read = std::getenv("DS4_CHECK_READ") != nullptr;
            for (size_t q = 0; q < fi.size(); ++q) {
                const int i = fi[q];
                held[(size_t) i] = fdst[q];
                if (check_read) {   // DS4_CHECK_READ=1: the split read against one whole read
                    std::vector<uint8_t> ref((size_t) im.blob);
                    im.blobs->read(layer, ids6[cpu_keep[i]], ref.data());
                    if (std::memcmp(ref.data(), fdst[q], (size_t) BLX(ids6[cpu_keep[i]])) != 0)
                        std::fprintf(stderr, "[read-chk] L%lld e%d: split read DIFFERS\n", (long long) layer,
                                     ids6[cpu_keep[i]]);
                    static int64_t n_chk = 0;
                    if (++n_chk % 200 == 0) std::fprintf(stderr, "[read-chk] %lld blobs checked\n", (long long) n_chk);
                }
            }
        }
        // ---- the single batch: all `nk` CPU experts, every class, each job with its own format; the activation
        // quantized once per DISTINCT gate/up format present (the Q2_0-native classes share `actq`) ----
        const double c0 = now_ms();
        bool q2_done = false;
        std::vector<char> act_done(im.nact_multi_act.size(), 0);
        for (int i = 0; i < nk; ++i) {
            const int c = im.g.cls(layer, ids6[cpu_keep[i]]);
            const cpu::NativeFmt& fc = im.fl[(size_t) c];
            if (cpu::q2_native_kernels(fc.gu_type)) {
                if (!q2_done) { cpu::act_quant_any(gp.h_x, (int) fc.n_embd, im.actq); q2_done = true; }
            } else {
                for (size_t b = 0; b < im.nact_multi_act.size(); ++b)
                    if (im.nact_multi_act[b] == fc.gu_act && !act_done[b]) {
                        cpu::native_quant_act(fc, gp.h_x, im.nact_multi.data() + b * cpu::kNativeActBytes);
                        act_done[b] = 1;
                    }
            }
            cpu::ExpertJobMulti& j = im.jobs[(size_t) i];
            j.blob = held[(size_t) i];
            j.fmt = &fc;
            j.nt = 1;
            for (int t = 0; t < cpu::MAXT; ++t) {
                j.nact[t] = nullptr;
                j.out[t] = nullptr;
            }
            j.nact[0] = nact_slot(im, fc.gu_act);
            j.act[0] = &im.actq;
            j.out[0] = im.parts.data() + (size_t) cpu_keep[i] * H;
        }
        im.pool->run_split_multi_native(im.fl[(size_t) im.g.cls(layer, ids6[cpu_keep[0]])], im.jobs.data(), nk);
        if (std::getenv("DS41_CHECK_POOL_BATCH") && !check_pool_batch(im, layer, ids6, cpu_keep, nk, held, gp.h_x))
            return false;
        st.cpu_ms = now_ms() - c0;
        // arena_adapt: the blobs read above already sit in their victims' slots - publish the swaps
        std::vector<int32_t> fslot(fi.size(), -1);
        for (size_t q = 0; q < fi.size(); ++q)
            if (fvic[q] >= 0) fslot[q] = arena_publish_swap(im, layer, ids6[cpu_keep[fi[q]]], fvic[q]);
        for (size_t q = 0; q < fi.size(); ++q)
            if (fvic[q] >= 0) arena_check_swap(im, layer, ids6[cpu_keep[fi[q]]], fslot[q]);
    }

    if (nfg > 0) {   // the file-tier blobs are home: up to the card, one grouped launch behind the hits / PCIe share
        for (auto& t : frd) t.join();
        im.st.file_tier += nfg;
        im.st.file_ms += now_ms() - tfg;
        std::vector<unsigned long long> ptr((size_t) nfg);
        std::vector<int32_t> start((size_t) nfg + 1), dst((size_t) nfg, 0), tok((size_t) nfg, 0);
        for (int j = 0; j < nfg; ++j) {
            uint8_t* to = (uint8_t*) gp.d_stage + (size_t) (np + j) * (size_t) B;
            ck(cudaMemcpyAsync(to, fsrc[j], (size_t) BLX(ids6[fg_k[j]]), cudaMemcpyHostToDevice,
                               gp.s), "file dma");
            ptr[(size_t) j] = (unsigned long long) to;
            start[(size_t) j] = j;
            dst[(size_t) j] = fg_k[j];
        }
        start[(size_t) nfg] = nfg;
        launch_group(2, nfg, ptr, start, dst, tok);
    }
    mark(3);
    // S4, the per-layer stream sync: it makes this layer's expert DMAs and kernels home before the bookkeeping
    // below publishes the slot swaps.  No GPU work inside this layer reads that bookkeeping, and the next
    // gpu_run (next token - a layer is visited once per token) reads it only after THIS call's own wait: the
    // sum on the run_to_dev path, the parts readback on the host path.  So on the decode path that later wait
    // already covers it and this sync is a second host round trip per layer (~40/token, measured 0.12 ms).
    // Kept where the caller reads parts on the host (there it IS that wait) and under DS4_CHECK_LRU, which
    // compares the slots against the file right here.  STRATA_LAYER_SYNC=1 = always sync (the old path).
    static const bool layer_sync_always = std::getenv("STRATA_LAYER_SYNC") != nullptr;
    const bool layer_synced = !out_dev || layer_sync_always || std::getenv("DS4_CHECK_LRU") != nullptr;
    if (layer_synced) ck(cudaStreamSynchronize(gp.s), "layer sync");
    if (lru)   // every lookup that was not a VRAM hit is a miss for the admission filter
        for (int64_t k = 0; k < K; ++k)
            if (slot_i[k] < 0) im.vmiss[(size_t) (layer * NXL + ids6[k])] = lc;
    static const bool check_lru = std::getenv("DS4_CHECK_LRU") != nullptr;
    for (int i = 0; i < n_lru; ++i) {   // vram_lru: the copies are done - the slots now hold the new experts
        gp.cache->replace(layer, lru_old[i], lru_new[i]);
        if (lru_dem[i] >= 0) {   // the arena slot changed hands: the demoted expert in, the promoted one out
            gp.arena.slot_of[(size_t) (layer * NXL + lru_new[i])] = -1;
            gp.arena.slot_of[(size_t) (layer * NXL + lru_old[i])] = lru_dem[i];
            if (!im.last_use.empty()) im.last_use[(size_t) (layer * NXL + lru_old[i])] = im.tick;   // warm, keep it
            ++st.vram_demotes;
        }
        ++st.vram_swaps;
        if (check_lru) {   // DS4_CHECK_LRU=1: the slot and the demoted arena copy against the file, byte for byte
            std::vector<uint8_t> ref((size_t) im.blob);
            std::string e;
            im.blobs->read(layer, lru_new[i], ref.data());
            const int64_t BL = BLX(lru_new[i]);
            const bool ok_v = gp.cache->verify_slot(gp.cache->slot_of(layer, lru_new[i]), ref.data(), e, BL);
            bool ok_a = true;
            if (lru_dem[i] >= 0) {
                im.blobs->read(layer, lru_old[i], ref.data());
                ok_a = std::memcmp(gp.arena.ptr(layer, lru_old[i]), ref.data(), (size_t) BL) == 0;
            }
            static int64_t n_ok = 0, n_bad = 0;
            (ok_v && ok_a ? n_ok : n_bad) += 1;
            if (!(ok_v && ok_a) || (n_ok + n_bad) % 500 == 0)
                std::fprintf(stderr, "[lru-chk] L%lld new e%d slot %s, demoted e%d arena %s (%lld ok, %lld bad)\n",
                             (long long) layer, lru_new[i], ok_v ? "ok" : "BAD", lru_old[i],
                             lru_dem[i] < 0 ? "-" : ok_a ? "ok" : "BAD", (long long) n_ok, (long long) n_bad);
        }
    }
    mark(4);
    float ms = 0;
    // cudaEventElapsedTime on an event that has not yet completed returns cudaErrorNotReady, so these need the
    // launches finished: layer_synced says the sync above ran.  hit_ms/pcie_ms are diagnostics only (never printed).
    if (layer_synced && ev_hit > 0) {
        ck(cudaEventElapsedTime(&ms, gp.ev[0], gp.ev[ev_hit]), "hits elapsed");
        st.hit_ms = ms;
    }
    if (layer_synced && ev_pcie > 0) {
        ck(cudaEventElapsedTime(&ms, gp.ev[ev_hit > 0 ? ev_hit : 0], gp.ev[ev_pcie]), "pcie elapsed");
        st.pcie_ms = ms;
    }
    if (out_dev) {
        // ---- the weighted sum on the card, straight into the caller's device row (run_to_dev): the pool's rows go
        // up into their routing-index rows of d_parts (beside the card's own), the weights through the metadata
        // block, and decode_rows_sum adds them in k order in double - the host loop below, bit for bit - with no
        // parts readback and no host -> device trip for the sum
        for (int i = 0; i < nk; ++i) {
            const int k = cpu_keep[i];
            if (wloc[k] == 0.0f) continue;
            float* hrow = (float*) gp.h_parts + (size_t) k * (size_t) H;
            std::memcpy(hrow, im.parts.data() + (size_t) k * (size_t) H, (size_t) H * 4);
            ck(cudaMemcpyAsync((float*) gp.d_parts + (size_t) k * (size_t) H, hrow, (size_t) H * 4, cudaMemcpyHostToDevice,
                               gp.s), "cpu row h2d");
        }
        // the sum weights live AFTER the captured blocks under STRATA_TIER_GRAPH (the graph's nodes must never see
        // them rewritten), and in the running region otherwise
        size_t o_w = grp_graph ? gp.tier_w_off : ((meta_at + 15) & ~(size_t) 15);
        if (o_w + sizeof(float) * (size_t) K > kMetaBytes) {
            ck(cudaStreamSynchronize(gp.s), "meta wrap");
            o_w = 0;
        }
        std::memcpy(gp.h_meta + o_w, wloc, sizeof(float) * (size_t) K);
        ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + o_w, gp.h_meta + o_w, sizeof(float) * (size_t) K, cudaMemcpyHostToDevice,
                           gp.s), "sum weights");
        strata::kernels::decode_rows_sum((const float*) gp.d_parts, (const float*) ((uint8_t*) gp.d_meta + o_w), (int) K, H,
                                         (float*) out_dev, gp.s);
        ck(cudaStreamSynchronize(gp.s), "sum sync");
    } else {
    ck(cudaMemcpyAsync(gp.h_parts, gp.d_parts, (size_t) std::max<int64_t>(kMaxParts, K) * (size_t) H * 4,
                       cudaMemcpyDeviceToHost, gp.s), "d2h parts");
    ck(cudaStreamSynchronize(gp.s), "parts sync");
    const float* gpu_parts = (const float*) gp.h_parts;
    mark(5);
    // DS4_CHECK_GPU=1: every card-computed expert recomputed on the CPU from the arena's bytes (diagnostic, slow)
    static const bool check_gpu = std::getenv("DS4_CHECK_GPU") != nullptr;
    if (check_gpu) {
        std::vector<float> ref((size_t) H);
        for (int64_t k = 0; k < K; ++k) {
            if (cpu_k[k]) continue;
            const uint8_t* p = gp.arena.ptr(layer, ids6[k]);
            for (int j = 0; j < nfg && !p; ++j)   // a file-tier miss run on the card: its staged bytes
                if (fg_k[j] == k) p = fsrc[j];
            if (!p) continue;
            ds41_moe_cpu_expert(im.fl[(size_t) im.g.cls(layer, ids6[k])], H, im.g.n_ff, p, gp.h_x, ref.data(), *im.pool);
            double num = 0, den = 0;
            const float* g = gpu_parts + (size_t) k * (size_t) H;
            for (int64_t i = 0; i < H; ++i) { const double d = g[i] - ref[(size_t) i]; num += d * d; den += (double) ref[(size_t) i] * ref[(size_t) i]; }
            bool is_pcie = false, is_file = false;
            for (int j = 0; j < np; ++j) is_pcie = is_pcie || pcie_i[j] == k;
            for (int j = 0; j < nfg; ++j) is_file = is_file || fg_k[j] == k;
            std::fprintf(stderr, "[chk] L%lld e%d %s rel %.3e\n", (long long) layer, ids6[k], is_file ? "file" : is_pcie ? "pcie" : "hit ",
                         std::sqrt(num / std::max(den, 1e-30)));
        }
    }

    // ---- the weighted sum.  Routing-indexed, so all three tiers write into one layout without a scatter ----
    // The pool's results are `im.parts` (host) and the card's are the readback above: the D2H is one bulk copy, so
    // it cannot be told to leave the pool's indices alone - the index's own source selects which buffer to read.
    // Reading `gpu_parts` for a CPU-computed index would sum uninitialized device memory, which is why this is not
    // a detail (the gate's file-tier arm caught it: the replay is speed-only and never sums a layer).
    for (int64_t i = 0; i < H; ++i) {
        double s = 0;
        for (int64_t k = 0; k < K; ++k) {
            if (wloc[k] == 0.0f) continue;   // a dropped miss: its parts row was never computed (0 * NaN is NaN)
            const float* p = cpu_k[k] ? im.parts.data() + (size_t) k * (size_t) H
                                      : gpu_parts + (size_t) k * (size_t) H;
            s += (double) wloc[k] * (double) p[(size_t) i];
        }
        out[i] = (float) s;
    }
    }   // host sum

    mark(6);
    // ---- admission: a compulsory miss takes a free slot and is filled for the NEXT token ----
    if (gp.cache && !im.cfg.no_cache && !im.mixed_sizes) {   // mixed: the seed sized every slot to its expert
        for (int i = 0; i < nc; ++i) {
            const int32_t e = ids6[cpu_i[i]];
            if (gp.cache->slot_of(layer, e) >= 0) continue;   // vram_lru already gave it a slot this call
            const uint8_t* p = gp.arena.ptr(layer, e);
            if (!p) continue;
            const int32_t s = gp.cache->admit(layer, e);
            if (s < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(s, p, e2, BLX(e))) {
                std::fprintf(stderr, "ds4_moe: fill_slot: %s\n", e2.c_str());
                std::abort();
            }
            ++im.admitted;
        }
    }

    mark(7);
    if (prof) ++g_prof_n;
    st.hits = nh;
    st.pcie = np + nfg;   // file-tier misses run on the card count with the PCIe share (DMA'd too)
    st.cpu = nk;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    im.pf_n = 0;   // the staging is per layer; the caller must run(l) before prefetch(l+1)
    return true;
}


/// q8_1 rows for `nt` tokens (nt <= kMaxTok) of `H` floats each: native_quantize_q8_1 takes at most 8 columns
/// per call (MAX_NCOLS), so a wider call steps through them 8 at a time.  The columns are contiguous (init checks the
/// layout) and every 32-element block lies inside one column, so the bytes are those of one call; nt <= 8 is the
/// single call it always was.
void quantize_rows_q8_1(const float* x, void* xq, int64_t H, int nt, cudaStream_t s) {
    if (nt <= 8) { strata::kernels::native_quantize_q8_1(x, xq, (int) H, nt, s); return; }
    const size_t col = strata::kernels::native_q8_1_bytes((int) H, 1);
    for (int c0 = 0; c0 < nt; c0 += 8)
        strata::kernels::native_quantize_q8_1(x + (size_t) c0 * (size_t) H, (uint8_t*) xq + (size_t) c0 * col, (int) H,
                                              std::min(8, nt - c0), s);
}

/// `nt` tokens of one layer in one pass (draft verification): the distinct experts across the tokens are each
/// fetched/read ONCE - a group with one entry per token that routed to it - so the second token's experts cost only
/// what the first did not already pay.  Entry j = t*K + k writes output row j; token t's sum is taken over its own k
/// in order, exactly as gpu_run does for one token.
bool gpu_run_n(Ds41MoeImpl& im, int64_t layer, int nt, const int32_t* ids, const float* w, const float* x_host,
               float* out) {
    Gpu& gp = *im.gpu;
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const int64_t B = im.blob;                        // staging stride
    const int64_t BL = im.bl[(size_t) layer];         // this layer's blob
    const cpu::NativeFmt& f = im.fl[(size_t) layer];
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) layer];
    const int NE = nt * (int) K;
    const double t0 = now_ms();
    Ds41MoeStats st;

    // ---- distinct experts, first-seen order ----
    int32_t ue[kMaxEnt], ent_u[kMaxEnt];
    int nu = 0;
    for (int j = 0; j < NE; ++j) {
        int u = -1;
        for (int i = 0; i < nu && u < 0; ++i)
            if (ue[i] == ids[j]) u = i;
        if (u < 0) { u = nu; ue[nu++] = ids[j]; }
        ent_u[j] = u;
    }
    // ---- classify: resident slot / prefetch staging / miss ----
    enum { HIT = 0, PCIE = 1, CPU = 2 };
    int cls[kMaxEnt];
    unsigned long long dptr[kMaxEnt] = {};
    int32_t miss[kMaxEnt];
    int nmiss = 0, npf = 0;
    for (int u = 0; u < nu; ++u) {
        const int32_t e = ue[u];
        const int32_t sl = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        int pf_at = -1;
        for (int j = 0; j < im.pf_n && sl < 0; ++j)
            if (im.pf_e[j] == e) pf_at = j;
        if (sl >= 0) {
            cls[u] = HIT;
            dptr[u] = (unsigned long long) gp.cache->device_slot(sl);
        } else if (pf_at >= 0) {
            cls[u] = HIT;
            dptr[u] = (unsigned long long) gp.d_pf + (size_t) pf_at * (size_t) B;
            ++npf;
        } else {
            cls[u] = CPU;
            miss[nmiss++] = u;
        }
    }
    // ---- the PCIe share of the distinct misses (same budget rule as gpu_run), at most kMaxParts staged ----
    int np = 0;
    {
        int eligible = 0;
        for (int i = 0; i < nmiss; ++i)
            if (gp.arena.ptr(layer, ue[miss[i]])) ++eligible;
        int budget;
        if (im.cfg.dither) {
            const double want = (double) eligible * im.cfg.pcie_frac + im.pcie_carry;
            budget = (int) std::floor(want);
            im.pcie_carry = want - budget;
        } else {
            budget = (int) std::lround((double) eligible * im.cfg.pcie_frac);
        }
        budget = std::min(budget, kMaxParts);
        for (int i = 0; i < nmiss && np < budget; ++i) {
            const int u = miss[nmiss - 1 - i];
            if (gp.arena.ptr(layer, ue[u])) { cls[u] = PCIE; ++np; }
        }
    }

    // ---- activations: Q8_K per token for the pool, q8_1 rows for the card ----
    std::memcpy(gp.h_x, x_host, (size_t) nt * (size_t) H * 4);
    for (int t = 0; t < nt; ++t)
        cpu::native_quant_act(f, gp.h_x + (size_t) t * H, im.nact_n.data() + (size_t) t * im.f.act_bytes);
    ck(cudaMemcpyAsync(gp.d_x, gp.h_x, (size_t) nt * (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "x h2d");
    quantize_rows_q8_1((const float*) gp.d_x, gp.d_xq, H, nt, gp.s);   // > 8 rows: 8 per quantizer call

    // ---- one grouped launch per GPU class (hits, PCIe), groups = distinct experts, entries = their tokens ----
    auto launch = [&](int want) {
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start, dst, tok;
        int si = 0;
        for (int u = 0; u < nu; ++u) {
            if (cls[u] != want) continue;
            if (want == PCIE) {
                ck(cudaMemcpyAsync((uint8_t*) gp.d_stage + (size_t) si * (size_t) B, gp.arena.ptr(layer, ue[u]),
                                   (size_t) BL, cudaMemcpyHostToDevice, gp.s), "pcie dma");
                dptr[u] = (unsigned long long) gp.d_stage + (size_t) si * (size_t) B;
                ++si;
            }
            ptr.push_back(dptr[u]);
            start.push_back((int32_t) dst.size());
            for (int j = 0; j < NE; ++j)
                if (ent_u[j] == u) { dst.push_back(j); tok.push_back(j / (int) K); }
        }
        const int ng = (int) ptr.size();
        if (ng == 0) return;
        start.push_back((int32_t) dst.size());
        const int32_t ngv = ng;
        ck(cudaMemcpyAsync(gp.d_grp_ptr, ptr.data(), sizeof(unsigned long long) * ng, cudaMemcpyHostToDevice, gp.s), "grp_ptr");
        ck(cudaMemcpyAsync(gp.d_grp_start, start.data(), sizeof(int32_t) * (ng + 1), cudaMemcpyHostToDevice, gp.s), "grp_start");
        ck(cudaMemcpyAsync(gp.d_ngroups, &ngv, sizeof(int32_t), cudaMemcpyHostToDevice, gp.s), "ngroups");
        ck(cudaMemcpyAsync(gp.d_ent_dst, dst.data(), sizeof(int32_t) * dst.size(), cudaMemcpyHostToDevice, gp.s), "ent_dst");
        ck(cudaMemcpyAsync(gp.d_ent_tok, tok.data(), sizeof(int32_t) * tok.size(), cudaMemcpyHostToDevice, gp.s), "ent_tok");
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) gp.d_grp_ptr,
                                              (const int32_t*) gp.d_grp_start, (const int32_t*) gp.d_ngroups,
                                              (const int32_t*) gp.d_ent_dst, (const int32_t*) gp.d_ent_tok, ng,
                                              (int64_t) dst.size(), gp.d_xq, gp.d_scratch, (float*) gp.d_parts, gp.s, ng);
    };
    if (npf > 0) ck(cudaStreamWaitEvent(gp.s, gp.ev_pf, 0), "wait prefetch");
    launch(HIT);
    launch(PCIE);

    // ---- the pool on the CPU misses, while the card works: one job per distinct expert, one row per token ----
    int nj = 0, ns = 0;   // ns: the distinct CPU experts so far (acquire's staging slot; a split job shares its blob)
    { int nc = 0; for (int u = 0; u < nu; ++u) nc += cls[u] == CPU; reserve_ftmp(im, nc); }
    if ((int) im.jobs.size() < nu * kJobSplit) im.jobs.resize((size_t) nu * kJobSplit);
    for (int u = 0; u < nu; ++u) {
        if (cls[u] != CPU) continue;
        bool from_file = false;
        const uint8_t* blob = acquire(im, layer, ue[u], ns++, &from_file);
        cpu::ExpertJobMulti* jb = nullptr;
        for (int j = 0; j < NE; ++j)
            if (ent_u[j] == u) {
                if (!jb || jb->nt == cpu::MAXT) {   // a new job: the first slice, or the previous one is full
                    jb = &im.jobs[(size_t) nj++];
                    jb->blob = blob;
                    jb->nt = 0;
                    jb->fmt = nullptr;   // this batch is one format (`f`); a stale per-job fmt from a prior call must not leak in
                    for (int t = 0; t < cpu::MAXT; ++t) { jb->nact[t] = nullptr; jb->out[t] = nullptr; }
                }
                jb->nact[jb->nt] = im.nact_n.data() + (size_t) (j / (int) K) * im.f.act_bytes;
                jb->out[jb->nt] = im.parts_n.data() + (size_t) j * (size_t) H;
                ++jb->nt;
            }
    }
    if (nj > 0) {
        const double c0 = now_ms();
        im.pool->run_split_multi_native(f, im.jobs.data(), nj);
        st.cpu_ms = now_ms() - c0;
    }
    ck(cudaMemcpyAsync(gp.h_parts, gp.d_parts, (size_t) NE * (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "d2h parts");
    ck(cudaStreamSynchronize(gp.s), "layer sync");
    const float* gpu_parts = (const float*) gp.h_parts;

    // ---- per-token weighted sums, k in order ----
    for (int t = 0; t < nt; ++t)
        for (int64_t i = 0; i < H; ++i) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                const int j = t * (int) K + (int) k;
                const float* p = cls[ent_u[j]] == CPU ? im.parts_n.data() + (size_t) j * H : gpu_parts + (size_t) j * H;
                acc += (double) w[j] * (double) p[(size_t) i];
            }
            out[(size_t) t * H + i] = (float) acc;
        }

    // ---- admission of the distinct misses (as gpu_run) ----
    if (gp.cache && !im.cfg.no_cache && !im.mixed_sizes) {
        for (int u = 0; u < nu; ++u) {
            if (cls[u] == HIT) continue;
            const uint8_t* p = gp.arena.ptr(layer, ue[u]);
            if (!p) continue;
            const int32_t sl = gp.cache->admit(layer, ue[u]);
            if (sl < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(sl, p, e2, BL)) {
                std::fprintf(stderr, "ds4_moe: fill_slot: %s\n", e2.c_str());
                std::abort();
            }
            ++im.admitted;
        }
    }
    // stats per (token, expert) lookup, so hit rates stay comparable with single-token runs
    for (int j = 0; j < NE; ++j) {
        const int c = cls[ent_u[j]];
        if (c == HIT) ++st.hits; else if (c == PCIE) ++st.pcie; else ++st.cpu;
    }
    st.prefetched_useful = npf;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    im.pf_n = 0;
    return true;
}

/// D3: gpu_run_n for a PER-EXPERT model (DS4 = 9 gate/up format classes).  Same union sharing as gpu_run_n -
/// each distinct expert is fetched/read ONCE, a group with one entry per token that routed to it - but the
/// grouped kernel takes ONE layout per launch, so the groups are split by format class and one
/// native_expert_grouped is launched per (class, site).  blob / fmt / layout are per expert's class.  A verify
/// pass of nt tokens is ONE call per layer instead of nt run_to_dev calls, so the union's experts, the PCIe
/// staging and the CPU pool are shared across the pass's tokens.
/// `out_dev != nullptr` = the weighted sum on the card, per token (bit-identical to nt run_to_dev calls);
/// `out_dev == nullptr` = the host sum into `out`.  Q2_0-native classes batch too: their CPU jobs read a per-token
/// ActQ (act_quant_any), the rest a native act slot.
bool gpu_run_n_ex(Ds41MoeImpl& im, int64_t layer, int nt, const int32_t* ids, const float* w,
                  const float* x_host, float* out, void* out_dev) {
    Gpu& gp = *im.gpu;
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const int64_t B = im.blob;                        // staging stride
    const int   NE = nt * (int) K;
    const double t0 = now_ms();
    Ds41MoeStats st;
    ++im.run_seq;   // spec_read tags its stash slots with the call that will use them (one call per layer per pass)
    im.run_layer = layer;
    if (std::getenv("DS41_DBG_X")) std::fprintf(stderr, "[dbg_x] ENTER gpu_run_n_ex layer=%lld nt=%d nact_multi_n=%zu nfmt=%zu kNativeActBytes=%d\n",
                                                  (long long) layer, nt, im.nact_multi_n.size(), im.nact_multi_act.size(), (int) cpu::kNativeActBytes);
    auto BLX = [&](int32_t e) { return im.bl[(size_t) im.g.cls(layer, e)]; };   // an expert's blob (the bytes a DMA moves)
    // DS41_PROF_X=1: wall ms per phase, summed over calls, printed every 640 calls (16 passes x 40 layers)
    // DS41_PROF_X=2: count only verify/decode-sized calls (nt <= 8) - a prompt's wide calls swamp the averages
    static const int xprof_mode = std::getenv("DS41_PROF_X") ? std::atoi(std::getenv("DS41_PROF_X")) : 0;
    const bool xprof = xprof_mode == 1 || (xprof_mode >= 2 && nt <= 8);
    static double xp[8] = {};
    static double xpool0[3] = {};
    static int64_t xpool_b0 = 0;
    static long xp_n = 0, xp_nu = 0, xp_cpu = 0, xp_file = 0;
    double xt = xprof ? now_ms() : 0;
    auto xmark = [&](int i) { if (xprof) { const double t = now_ms(); xp[i] += t - xt; xt = t; } };

    // ---- distinct experts, first-seen order (entries keep their routing index j = t*K + k) ----
    int32_t ue[kMaxEnt], ent_u[kMaxEnt];
    int nu = 0;
    for (int j = 0; j < NE; ++j) {
        int u = -1;
        for (int i = 0; i < nu && u < 0; ++i)
            if (ue[i] == ids[j]) u = i;
        if (u < 0) { u = nu; ue[nu++] = ids[j]; }
        ent_u[j] = u;
    }
    // ---- classify: resident slot / prefetch staging / miss ----
    // FILEG: a file-tier miss read to pinned staging, run on the card; REMOTE: held by the STRATA_DS41_REMOTE helper
    // DEFER: a file-tier miss STRATA_FILE_DEFER leaves to the next layer's call
    enum { HIT = 0, PCIE = 1, CPU = 2, FILEG = 3, REMOTE = 4, DEFER = 5 };
    int cls[kMaxEnt];
    bool from_pf[kMaxEnt] = {};   // a hit served out of a prefetch slot (prefetch_multi) - LRU-admitted below
    unsigned long long dptr[kMaxEnt] = {};
    int32_t miss[kMaxEnt];
    int nmiss = 0, npf = 0;
    for (int u = 0; u < nu; ++u) {
        const int32_t e = ue[u];
        const int32_t sl = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        int pf_at = -1;
        for (int j = 0; j < im.pf_n && sl < 0; ++j)
            if (im.pf_e[j] == e) pf_at = j;
        if (sl >= 0) { cls[u] = HIT; dptr[u] = (unsigned long long) gp.cache->device_slot(sl); }
        else if (pf_at >= 0) { cls[u] = HIT; from_pf[u] = true; dptr[u] = (unsigned long long) gp.d_pf + (size_t) pf_at * (size_t) B; ++npf; }
        else if (gp.remote && gp.remote->s->holds[(size_t) (layer * im.g.n_experts + e)]) cls[u] = REMOTE;
        else { cls[u] = CPU; miss[nmiss++] = u; }
    }
    // ---- STRATA_DS41_REMOTE: the helper's entries go out now and come back before the sum, so its round trip
    // (~0.1 ms) runs under this call's own launches.  Entry i of the request is routing index rrow[i]. ----
    int32_t rrow[kMaxEnt];
    int nr = 0;
    if (gp.remote) {
        Ds41RemoteShm& R = *gp.remote->s;
        for (int j = 0; j < NE; ++j)
            if (cls[ent_u[j]] == REMOTE && w[j] != 0.0f) {
                R.ent_expert[nr] = ids[j];
                R.ent_tok[nr] = j / (int) K;
                rrow[nr++] = j;
            }
        if (nr > 0) {
            std::memcpy(R.x, x_host, (size_t) nt * (size_t) H * 4);
            R.layer = (int32_t) layer;
            R.nt = nt;
            R.n_ent = nr;
            R.req.store(++gp.remote->n, std::memory_order_release);
        }
    }
    // ---- the PCIe share of the distinct misses (same budget rule as gpu_run) ----
    int np = 0;
    {
        int eligible = 0;
        for (int i = 0; i < nmiss; ++i)
            if (gp.arena.ptr(layer, ue[miss[i]])) ++eligible;
        // D3's OWN PCIe share (STRATA_TIER_BATCH_PCIE, default 1 = every arena miss over PCIe): the pool's cost per
        // expert grows with the pass's token count (it computes every routed token) while a DMA does not, and the
        // pool runs serially after the card's launches - measured (DS41_PROF_X): pool ~2 ms per expert, the critical
        // path of the whole pass.  --pcie (cfg.pcie_frac) stays the single-token decode's knob; this one is
        // --tier-batch-pcie (cfg.tier_batch_pcie), which also governs every DSpark/serve decode and verify call.
        static const double d3_env = std::getenv("STRATA_TIER_BATCH_PCIE") ? std::atof(std::getenv("STRATA_TIER_BATCH_PCIE")) : 1.0;
        const double d3_frac = im.cfg.tier_batch_pcie >= 0.0 ? std::min(1.0, im.cfg.tier_batch_pcie) : d3_env;
        int budget;
        if (im.cfg.dither) {
            const double want = (double) eligible * d3_frac + im.pcie_carry;
            budget = (int) std::floor(want);
            im.pcie_carry = want - budget;
        } else {
            budget = (int) std::lround((double) eligible * d3_frac);
        }
        budget = std::min(budget, kMaxEnt);   // the PCIe site streams through the d_stage slots in waves
        // q* (budget short of the eligible misses): the RECURRING misses cross PCIe first - missed again inside the
        // vram_lru admission window, so the DMA also admits them to VRAM - and the one-offs go to the CPU pool
        // (FreeToken's BY_RECENCY rule; the admission filter is our recency signal).  STRATA_TIER_QSTAR_RECENCY=0: the
        // old order (the last-seen misses first).
        static const bool qs_recency = !(std::getenv("STRATA_TIER_QSTAR_RECENCY") && std::getenv("STRATA_TIER_QSTAR_RECENCY")[0] == '0');
        if (qs_recency && budget < eligible && !im.vmiss.empty() && !im.lcalls.empty()) {
            static const int64_t win = std::getenv("STRATA_LRU_WIN") ? std::atoll(std::getenv("STRATA_LRU_WIN")) : 4;
            const int64_t lc_next = im.lcalls[(size_t) layer] + 1;
            for (int pass = 0; pass < 2 && np < budget; ++pass)
                for (int i = 0; i < nmiss && np < budget; ++i) {
                    const int u = miss[nmiss - 1 - i];
                    if (cls[u] == PCIE || !gp.arena.ptr(layer, ue[u])) continue;
                    const int64_t m = im.vmiss[(size_t) (layer * im.g.n_experts + ue[u])];
                    const bool recurring = m >= 0 && lc_next - m <= win;
                    if (recurring == (pass == 0)) { cls[u] = PCIE; ++np; }
                }
        } else {
            for (int i = 0; i < nmiss && np < budget; ++i) {
                const int u = miss[nmiss - 1 - i];
                if (gp.arena.ptr(layer, ue[u])) { cls[u] = PCIE; ++np; }
            }
        }
    }
    // ---- file-tier misses (in neither VRAM nor the arena) ON THE CARD: read in parallel (8 threads, split
    // blobs) into pinned h_fstage WHILE the hit kernels are launched and run, then DMA'd behind the PCIe share and
    // run in its grouped launch.  They used to be acquire()'d one at a time on the critical path before the pool
    // (DS41_PROF_X: 1.6 ms of a 2.9 ms call).  STRATA_FILE_GPU=0: the pool computes them (the old path).
    static const bool file_gpu = !(std::getenv("STRATA_FILE_GPU") && std::getenv("STRATA_FILE_GPU")[0] == '0');
    int fg_u[kMaxFg], nfg = 0;
    if (file_gpu) {
        for (int u = 0; u < nu; ++u)
            if (cls[u] == CPU && !gp.arena.ptr(layer, ue[u]) && nfg < kMaxFg) { cls[u] = FILEG; fg_u[nfg++] = u; }   // h_fstage: kMaxFg blobs
    }
    // arena_adapt (as gpu_run): every lookup is a use; each FILEG blob is read STRAIGHT into the arena slot of the
    // layer's least-recently-used arena expert not routed now (its slot big enough) - the DMA then sources it there,
    // and the slot changes hands at the end of the call.  The arena is pinned, so the DMA is the same as from
    // h_fstage.  No copy of this call writes a victim slot: the vram_lru demotions D2H into the slot of an expert
    // routed NOW (never a victim), and FILEG experts are not LRU-admitted.
    const bool adapt = im.cfg.arena_adapt && gp.arena.base;
    const uint8_t* fg_src[kMaxFg];
    int32_t fg_vic[kMaxFg];
    if (im.cfg.arena_adapt) {
        const int64_t NXA = im.g.n_experts;
        if (im.last_use.empty()) im.last_use.assign((size_t) (im.g.n_layers * NXA), -1);
        ++im.tick;
        for (int u = 0; u < nu; ++u) im.last_use[(size_t) (layer * NXA + ue[u])] = im.tick;
    }
    if (nfg > 0 && !gp.h_fstage) ck(cudaMallocHost((void**) &gp.h_fstage, (size_t) kMaxFg * (size_t) B), "h_fstage");
    bool any_vic = false;
    // a blob a spec_read (STRATA_D3_SPEC) already brought in sits in the stash: the DMA sources it there, and under
    // arena_adapt it is copied into its victim slot after the DMAs are queued (fg_copy); the rest are read now
    uint8_t* fg_copy[kMaxFg] = {};
    int fread_b[kMaxFg], nfr = 0;
    for (int b = 0; b < nfg; ++b) {
        fg_vic[b] = adapt ? arena_victim(im, layer, (uint64_t) BLX(ue[fg_u[b]]), fg_vic, b) : -1;
        uint8_t* vslot = fg_vic[b] >= 0
            ? (uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) gp.arena.slot_of[(size_t) (layer * im.g.n_experts + fg_vic[b])]]
            : nullptr;
        const uint8_t* stashed = gp.spec ? gp.spec->take(layer, ue[fg_u[b]], im.run_seq) : nullptr;
        if (stashed) {
            ++im.st.spec_hits;
            fg_src[b] = stashed;
            fg_copy[b] = vslot;
        } else {
            fg_src[b] = vslot ? vslot : gp.h_fstage + (size_t) b * (size_t) B;
            fread_b[nfr++] = b;
        }
        any_vic = any_vic || fg_vic[b] >= 0;
    }
    // ---- STRATA_FILE_DEFER=R (opt-in; CHANGES OUTPUTS): a file-tier miss whose every entry has routing rank >= R
    // (0 = each token's top expert) is not waited for.  Its read runs on a second reader pool while the NEXT layer's
    // call works, and that call adds the expert's weighted rows into its own routed sum (`prev` below): the layer's
    // output misses it, the next layer's routed sum carries it - one layer late, never dropped.  The DSpark / decode
    // device-sum path only (layer + 1 is this tier's next call there), never the last layer, not with the remote helper.
    static const int defer_r = std::getenv("STRATA_FILE_DEFER") ? std::atoi(std::getenv("STRATA_FILE_DEFER")) : -1;
    const FileDefer prev = im.fdef;   // the previous call's deferred experts, added at this call's sum
    im.fdef.on = false;
    bool fg_def[kMaxFg] = {};
    if (defer_r >= 0 && out_dev && !gp.remote && layer + 1 < im.g.n_layers && im.g.routed(layer + 1) && nfr > 0) {
        FileDefer& nd = im.fdef;
        nd.layer = layer; nd.nt = nt; nd.par = (int) (layer & 1); nd.n = 0; nd.nent = 0;
        int keep = 0;
        for (int i = 0; i < nfr; ++i) {
            const int b = fread_b[i], u = fg_u[b];
            bool low = nd.n < kMaxDefer;
            for (int j = 0; j < NE && low; ++j)
                if (ent_u[j] == u && (j % (int) K) < defer_r) low = false;
            if (!low) { fread_b[keep++] = b; continue; }
            cls[u] = DEFER;
            fg_def[b] = true;
            const int d = nd.n++;
            nd.e[d] = ue[u]; nd.c[d] = im.g.cls(layer, ue[u]); nd.vic[d] = fg_vic[b];
            if (fg_vic[b] < 0) {   // no arena_adapt victim slot to read into: a pinned staging slot of its own
                if (!gp.h_def[nd.par]) ck(cudaMallocHost((void**) &gp.h_def[nd.par], (size_t) kMaxDefer * (size_t) B), "h_def");
                fg_src[b] = gp.h_def[nd.par] + (size_t) d * (size_t) B;
            }
            nd.src[d] = fg_src[b];
            for (int j = 0; j < NE; ++j)
                if (ent_u[j] == u) { nd.ent_j[nd.nent] = j; nd.ent_d[nd.nent] = d; nd.w[nd.nent] = w[j]; ++nd.nent; }
        }
        nfr = keep;
        nd.on = nd.n > 0;
    }
    if (im.trace) {   // trace_open: 'C' layer nt nu, then per distinct expert {expert, first row, class, from_pf}
        int32_t rec[4 + 4 * kMaxEnt];
        int n = 0;
        rec[n++] = 'C'; rec[n++] = (int32_t) layer; rec[n++] = nt; rec[n++] = nu;
        for (int u = 0; u < nu; ++u) {
            int first = 0;
            while (first < NE && ent_u[first] != u) ++first;
            rec[n++] = ue[u]; rec[n++] = first / (int) K; rec[n++] = cls[u]; rec[n++] = from_pf[u] ? 1 : 0;
        }
        std::fwrite(rec, 4, (size_t) n, im.trace);
    }
    if (any_vic) {   // nothing still in flight from an EARLIER call (a gpu_run decode skips its layer sync) reads or
                     // writes a victim slot while the reader threads below fill it from the CPU
        if (gp.s_pf) ck(cudaStreamSynchronize(gp.s_pf), "d3 arena-adapt pf sync");
        ck(cudaStreamSynchronize(gp.s), "d3 arena-adapt sync");
        if (gp.s_dc) ck(cudaStreamSynchronize(gp.s_dc), "d3 arena-adapt dc sync");
        if (gp.s_dn) ck(cudaStreamSynchronize(gp.s_dn), "d3 arena-adapt dn sync");   // idle between calls; cheap
    }
    const double tfg = nfg > 0 ? now_ms() : 0;
    if (nfr > 0) {
        if (!gp.rpool) gp.rpool = std::make_unique<ReadPool>();
        // STRATA_D3_PARTS: a call's read budget in parts (default 32 = the readers, split over its blobs)
        static const int d3_parts = std::getenv("STRATA_D3_PARTS") ? std::max(1, std::atoi(std::getenv("STRATA_D3_PARTS"))) : 32;
        const int parts = std::max(1, std::min(d3_parts, d3_parts / nfr));
        gp.rpool->start((size_t) nfr * (size_t) parts, [&, parts](size_t r) {
            const int b = fread_b[r / (size_t) parts];
            im.blobs->read_part(layer, ue[fg_u[b]], (int) (r % (size_t) parts), parts, (uint8_t*) fg_src[b]);
        });
    }
    if (im.fdef.on) {   // the deferred reads: their own pool, captured by value (they outlive this call)
        const FileDefer& nd = im.fdef;
        auto& rp = gp.rpool_def[nd.par];
        if (!rp) rp = std::make_unique<ReadPool>();
        const int parts = std::max(1, 32 / nd.n);
        std::array<int32_t, kMaxDefer> de{};
        std::array<uint8_t*, kMaxDefer> dd{};
        for (int d = 0; d < nd.n; ++d) { de[d] = nd.e[d]; dd[d] = (uint8_t*) nd.src[d]; }
        Ds41BlobSource* bs = im.blobs;
        const int64_t dl = layer;
        rp->start((size_t) nd.n * (size_t) parts, [bs, dl, parts, de, dd](size_t r) {
            const size_t d = r / (size_t) parts;
            bs->read_part(dl, de[d], (int) (r % (size_t) parts), parts, dd[d]);
        });
    }
    xmark(0);
    // STRATA_D3_DMA_STREAM: the cross-call barrier.  Older work still queued on `s` (a gpu_run decode that skipped its
    // layer sync, its FILEG staging, ...) may read d_stage or the victim / arena slots this call's copies write: the
    // copy stream waits for all of it ONCE here, so this call's copies overlap only this call's own work below.
    const bool dn = gp.s_dn != nullptr;
    const cudaStream_t cs = dn ? gp.s_dn : gp.s;   // the PCIe site's copies
    if (dn) {
        ck(cudaEventRecord(gp.dn_ev[0], gp.s), "d3 dn barrier record");
        ck(cudaStreamWaitEvent(gp.s_dn, gp.dn_ev[0], 0), "d3 dn barrier wait");
    }
    // ---- activations: q8_1 rows for the card (one per token); the pool's per-class quant runs with its jobs ----
    std::memcpy(gp.h_x, x_host, (size_t) nt * (size_t) H * 4);
    ck(cudaMemcpyAsync(gp.d_x, gp.h_x, (size_t) nt * (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "x h2d");
    quantize_rows_q8_1((const float*) gp.d_x, gp.d_xq, H, nt, gp.s);   // > 8 rows: 8 per quantizer call
    if (im.fdef.on) {   // STRATA_FILE_DEFER: this layer's q8_1 rows, for the next call's run of the deferred experts
        const size_t xq_col = strata::kernels::native_q8_1_bytes((int) H, 1);
        void*& dq = gp.d_xq_def[im.fdef.par];
        if (!dq) ck(cudaMalloc(&dq, (size_t) kMaxTok * xq_col), "d_xq_def");
        ck(cudaMemcpyAsync(dq, gp.d_xq, (size_t) nt * xq_col, cudaMemcpyDeviceToDevice, gp.s), "xq def copy");
    }
    if (npf > 0) ck(cudaStreamWaitEvent(gp.s, gp.ev_pf, 0), "wait prefetch");
    xmark(1);
    // ---- one grouped launch per (site, class); groups = distinct experts, entries = the tokens routing to them ----
    // ---- vram_lru, as gpu_run: a PCIe miss that passes the admission filter (it also missed in one of the layer's
    // last kLruWin calls) is DMA'd straight into the slot of the layer's least-recently-used same-class resident
    // expert not routed by this pass - the kernel reads it there and it stays resident.  A victim with no arena copy
    // is demoted into the arena slot the incoming expert vacates.  Without this the pass never admitted anything
    // (the old admission was off for mixed blob sizes - every DS4 model) and the hit rate decayed over a run
    // (DS41_PROF_X: sum+sync 1.44 -> 1.67 ms/call over 256 tokens). ----
    const bool lru = im.cfg.vram_lru && gp.cache && !im.cfg.no_cache;
    const int64_t NXL = im.g.n_experts;
    static const int64_t kLruWin = std::getenv("STRATA_LRU_WIN") ? std::atoll(std::getenv("STRATA_LRU_WIN")) : 4;   // admission window (calls)
    int64_t lc = 0;
    int32_t lru_new[kMaxEnt], lru_old[kMaxEnt], lru_dem[kMaxEnt];
    int n_lru = 0;
    auto arena_slot = [&](int32_t e) { return gp.arena.base ? gp.arena.slot_of[(size_t) (layer * NXL + e)] : -1; };
    auto admit_ok = [&](int32_t e) { const int64_t m = im.vmiss[(size_t) (layer * NXL + e)]; return m >= 0 && lc - m <= kLruWin; };
    auto lru_victim = [&](int32_t incoming) -> int32_t {
        int32_t v = -1;
        int64_t best = INT64_MAX;
        const int ci = im.g.cls(layer, incoming);
        for (int64_t e = 0; e < NXL; ++e) {
            if (gp.cache->slot_of(layer, e) < 0 || im.g.cls(layer, e) != ci) continue;   // same class = same slot size
            const int64_t u = im.vlast[(size_t) (layer * NXL + e)];
            if (u >= im.tick) continue;   // routed by this pass, or already taken by it
            if (u < best) { best = u; v = (int32_t) e; }
        }
        return v;
    };
    if (lru) {
        if (!im.cfg.arena_adapt) ++im.tick;
        if (im.vlast.empty()) im.vlast.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.vmiss.empty()) im.vmiss.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.lcalls.empty()) im.lcalls.assign((size_t) im.g.n_layers, 0);
        lc = ++im.lcalls[(size_t) layer];
        for (int u = 0; u < nu; ++u) im.vlast[(size_t) (layer * NXL + ue[u])] = im.tick;
    }
    // The launches' metadata goes into the pinned h_meta block at monotonic offsets (never reused within this call;
    // the call ends with a stream sync, so the next call can start over) and up in ONE async copy per launch - it was
    // five pageable cudaMemcpyAsync per launch, each staged synchronously by the driver.  Offsets [0, kMetaW) hold the
    // pass's routing weights (the out_dev sum below).
    constexpr size_t kMetaW = 1024;
    size_t meta_at = kMetaW;
    auto al16 = [](size_t x) { return (x + 15) & ~(size_t) 15; };
    auto launch_group = [&](int c, const std::vector<unsigned long long>& ptr, std::vector<int32_t>& start,
                            const std::vector<int32_t>& dst, const std::vector<int32_t>& tok,
                            const void* xq = nullptr, float* parts_out = nullptr) {   // (STRATA_FILE_DEFER: other rows)
        const int ng = (int) ptr.size();
        if (ng == 0) return;
        start.push_back((int32_t) dst.size());
        const size_t ne = dst.size();
        auto span = [&](size_t at) { return al16(al16(al16(al16(al16(at + 8 * (size_t) ng) + 4 * (size_t) (ng + 1)) + 4) + 4 * ne) + 4 * ne); };
        if (span(meta_at) > kMetaBytes) {   // cannot happen for top-k 6 at kMaxTok 16 (a few KB); wrap behind a sync
            ck(cudaStreamSynchronize(gp.s), "meta wrap");   // every earlier copy out of h_meta is done
            meta_at = kMetaW;
        }
        const size_t o_ptr = meta_at, o_st = al16(o_ptr + 8 * (size_t) ng), o_ng = al16(o_st + 4 * (size_t) (ng + 1)),
                     o_dst = al16(o_ng + 4), o_tok = al16(o_dst + 4 * ne), o_end = al16(o_tok + 4 * ne);
        const int32_t ngv = ng;
        std::memcpy(gp.h_meta + o_ptr, ptr.data(), 8 * (size_t) ng);
        std::memcpy(gp.h_meta + o_st, start.data(), 4 * (size_t) (ng + 1));
        std::memcpy(gp.h_meta + o_ng, &ngv, 4);
        std::memcpy(gp.h_meta + o_dst, dst.data(), 4 * ne);
        std::memcpy(gp.h_meta + o_tok, tok.data(), 4 * ne);
        ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + o_ptr, gp.h_meta + o_ptr, o_end - o_ptr, cudaMemcpyHostToDevice, gp.s),
           "grp meta");
        meta_at = o_end;
        uint8_t* dm = (uint8_t*) gp.d_meta;
        strata::kernels::native_expert_grouped(gp.gll[(size_t) c], (const unsigned long long*) (dm + o_ptr),
                                              (const int32_t*) (dm + o_st), (const int32_t*) (dm + o_ng),
                                              (const int32_t*) (dm + o_dst), (const int32_t*) (dm + o_tok), ng,
                                              (int64_t) ne, xq ? xq : gp.d_xq, gp.d_scratch,
                                              parts_out ? parts_out : (float*) gp.d_parts, gp.s, ng);
    };
    // ---- one grouped launch per (site, class); groups = distinct experts, entries = the tokens routing to them.
    // The PCIe site (arena misses + FILEG) DMAs into the kMaxParts d_stage slots IN WAVES: when the slots run out
    // mid-class, the group so far is launched and the slots are reused - every copy and kernel is on gp.s, so a
    // copy into a slot queues behind the kernel that reads it.
    // STRATA_D3_DMA_STREAM: the same copies, in the same order, on gp.s_dn into two banks of gp.d3_bank slots.  A wave
    // fills one bank (as above: `si` runs on across classes, and a full bank launches the group so far), then the
    // next wave fills the other bank while this one's kernel runs.  Before each launch `s` waits for the copies
    // issued so far (dn_ev[1]); after it, dn_ev[2 + bank] marks the bank's readers, and the copy stream waits on that
    // before it writes the bank again.  At d3_bank == kMaxParts the waves (and so every launch) are the stream-off
    // ones; only the slot addresses differ. ----
    const int bank_w = dn ? gp.d3_bank : kMaxParts;
    int bank = 0;                           // the bank being filled (dn only; bank 0 = the stream-off slots)
    bool bank_read[2] = {false, false};     // dn_ev[2 + b] is recorded after a kernel of this call that reads bank b
    auto pcie_launch = [&](int c, const std::vector<unsigned long long>& ptr, std::vector<int32_t>& start,
                           const std::vector<int32_t>& dst, const std::vector<int32_t>& tok) {
        if (ptr.empty()) return;
        if (dn) {   // this wave's bytes (and every earlier PCIe-site copy) are home before its kernels run
            ck(cudaEventRecord(gp.dn_ev[1], gp.s_dn), "d3 dn copies record");
            ck(cudaStreamWaitEvent(gp.s, gp.dn_ev[1], 0), "d3 dn copies wait");
        }
        launch_group(c, ptr, start, dst, tok);
        if (dn) {
            ck(cudaEventRecord(gp.dn_ev[2 + bank], gp.s), "d3 dn bank record");
            bank_read[bank] = true;
        }
    };
    auto launch_site = [&](int want) {
        bool used[kMaxEnt] = {};
        auto in_site = [&](int u) { return cls[u] == want || (want == PCIE && cls[u] == FILEG); };
        int left = 0;
        for (int u = 0; u < nu; ++u) if (in_site(u)) ++left;
        int si = 0;   // the next d_stage slot (of the current bank)
        while (left > 0) {
            int c = -1;
            for (int u = 0; u < nu && c < 0; ++u)
                if (!used[u] && in_site(u)) c = im.g.cls(layer, ue[u]);
            if (c < 0) break;
            std::vector<unsigned long long> ptr;
            std::vector<int32_t> start, dst, tok;
            for (int u = 0; u < nu; ++u) {
                if (used[u] || !in_site(u) || im.g.cls(layer, ue[u]) != c) continue;
                used[u] = true; --left;
                if (want == PCIE) {
                    const int32_t e = ue[u];
                    const int64_t BL = BLX(e);
                    const uint8_t* src = gp.arena.ptr(layer, e);
                    if (cls[u] == FILEG) {   // its staging (or arena_adapt victim) slot = its index in fg_u
                        int b = 0;
                        while (fg_u[b] != u) ++b;
                        src = fg_src[b];
                    }
                    // LRU admission (arena-held PCIe misses only, as gpu_run)
                    int32_t v = (lru && cls[u] == PCIE && admit_ok(e)) ? lru_victim(e) : -1;
                    if (v >= 0 && arena_slot(v) < 0 && arena_slot(e) < 0) v = -1;   // nowhere to demote to
                    const bool demote = v >= 0 && arena_slot(v) < 0;
                    if (v < 0 || demote) {   // through a d_stage slot (the kernel reads it there)
                        if (si == bank_w) {   // the slots are all in this wave: launch it, then reuse them
                            pcie_launch(c, ptr, start, dst, tok);
                            ptr.clear(); start.clear(); dst.clear(); tok.clear();
                            si = 0;
                            if (dn) {   // the other bank; its last readers (an earlier wave) must be done first
                                bank ^= 1;
                                if (bank_read[bank]) ck(cudaStreamWaitEvent(gp.s_dn, gp.dn_ev[2 + bank], 0), "d3 dn bank wait");
                            }
                        }
                        uint8_t* to = (uint8_t*) gp.d_stage + (size_t) (bank * bank_w + si) * (size_t) B;
                        ck(cudaMemcpyAsync(to, src, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");
                        dptr[u] = (unsigned long long) to;
                        ++si;
                    }
                    if (v >= 0) {
                        uint8_t* vs = gp.cache->device_slot(gp.cache->slot_of(layer, v));
                        im.vlast[(size_t) (layer * NXL + v)] = im.tick;   // not again in this pass
                        int32_t dem = -1;
                        if (demote) {   // stage holds e; victim -> e's arena slot (D2H after e's H2D read it), stage -> victim slot
                            dem = arena_slot(e);
                            ck(cudaMemcpyAsync((uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) dem], vs, (size_t) BLX(v),
                                               cudaMemcpyDeviceToHost, cs), "lru demote");
                            ck(cudaMemcpyAsync(vs, (const void*) dptr[u], (size_t) BL, cudaMemcpyDeviceToDevice, cs), "lru stage copy");
                        } else {        // the DMA lands in the victim slot: the kernel reads it there, and it stays
                            ck(cudaMemcpyAsync(vs, src, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");
                            dptr[u] = (unsigned long long) vs;
                        }
                        lru_old[n_lru] = v;
                        lru_dem[n_lru] = dem;
                        lru_new[n_lru++] = e;
                    }
                }
                ptr.push_back(dptr[u]);
                start.push_back((int32_t) dst.size());
                for (int j = 0; j < NE; ++j)
                    if (ent_u[j] == u) { dst.push_back(j); tok.push_back(j / (int) K); }
            }
            if (want == PCIE) pcie_launch(c, ptr, start, dst, tok);
            else launch_group(c, ptr, start, dst, tok);
        }
    };
    launch_site(HIT);
    // vram_lru, as gpu_run: a prefetched expert the pass used is copied device-to-device into a victim slot (the
    // hits kernel above read the staging copy; this copy only reads it too, and the next prefetch waits for the sync)
    if (lru && npf > 0) {
        for (int u = 0; u < nu; ++u) {
            if (!from_pf[u] || !admit_ok(ue[u])) continue;
            const int32_t e = ue[u], v = lru_victim(e);
            if (v < 0) continue;
            const int32_t sl = gp.cache->slot_of(layer, v);
            int32_t dem = -1;
            if (arena_slot(v) < 0) {   // demote the victim into the arena slot the prefetched expert came from
                dem = arena_slot(e);
                if (dem < 0) continue;
                ck(cudaMemcpyAsync((uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) dem], gp.cache->device_slot(sl),
                                   (size_t) BLX(v), cudaMemcpyDeviceToHost, gp.s), "lru demote");
            }
            ck(cudaMemcpyAsync(gp.cache->device_slot(sl), (const void*) dptr[u], (size_t) BLX(e), cudaMemcpyDeviceToDevice, gp.s),
               "lru pf copy");
            im.vlast[(size_t) (layer * NXL + v)] = im.tick;
            lru_old[n_lru] = v; lru_dem[n_lru] = dem; lru_new[n_lru++] = e;
        }
    }
    if (nfg > 0) {   // the file-tier blobs must be home before their DMAs are queued
        if (nfr > 0) gp.rpool->wait();
        im.st.file_tier += nfg;
        im.st.file_ms += now_ms() - tfg;
    }
    launch_site(PCIE);
    if (dn) {   // belt and braces: `s` has waited on every PCIe-site copy, so the sync that ends this call covers s_dn
        ck(cudaEventRecord(gp.dn_ev[1], gp.s_dn), "d3 dn tail record");
        ck(cudaStreamWaitEvent(gp.s, gp.dn_ev[1], 0), "d3 dn tail wait");
    }
    // stashed FILEG blobs -> their arena_adapt victim slots, on the CPU while the card runs the DMAs (both only read
    // the stash slot; the slot is not reissued before this call has returned)
    for (int b = 0; b < nfg; ++b)
        if (fg_copy[b]) std::memcpy(fg_copy[b], fg_src[b], (size_t) BLX(ue[fg_u[b]]));
    xmark(2);
    // ---- the pool on the CPU misses: one job per distinct expert (split into jobs of <= cpu::MAXT tokens, kJobSplit),
    // its class's fmt + per-(token,class) quant ----
    int nj = 0, ns = 0;   // ns: the distinct CPU experts so far (acquire's staging slot; a split job shares its blob)
    { int nc = 0; for (int u = 0; u < nu; ++u) nc += cls[u] == CPU; reserve_ftmp(im, nc); }
    if ((int) im.jobs.size() < nu * kJobSplit) im.jobs.resize((size_t) nu * kJobSplit);
    std::vector<char> act_done(im.nact_multi_act.size() * (size_t) kMaxTok, 0);
    bool q2_done[kMaxTok] = {};   // a Q2_0-class job reads the token's ActQ (act_quant_any), not a native act slot
    for (int u = 0; u < nu; ++u) {
        if (cls[u] != CPU) continue;
        const cpu::NativeFmt& fc = im.fl[(size_t) im.g.cls(layer, ue[u])];
        bool from_file = false;
        const uint8_t* blob = acquire(im, layer, ue[u], ns++, &from_file);
        cpu::ExpertJobMulti* jb = nullptr;
        for (int j = 0; j < NE; ++j) {
            if (ent_u[j] != u) continue;
            if (!jb || jb->nt == cpu::MAXT) {   // a new job: the first slice, or the previous one is full
                jb = &im.jobs[(size_t) nj++];
                jb->blob = blob;
                jb->nt = 0;
                jb->fmt = &fc;
                for (int t = 0; t < cpu::MAXT; ++t) { jb->nact[t] = nullptr; jb->out[t] = nullptr; jb->act[t] = nullptr; }
            }
            const int tt = j / (int) K;
            if (cpu::q2_native_kernels(fc.gu_type)) {
                if (!q2_done[tt]) { cpu::act_quant_any(gp.h_x + (size_t) tt * H, (int) fc.n_embd, im.actq_n[(size_t) tt]); q2_done[tt] = true; }
                jb->act[jb->nt] = &im.actq_n[(size_t) tt];
            } else {
                uint8_t* slot = nact_slot_n(im, tt, fc.gu_act);
                const size_t ai = (size_t) tt * im.nact_multi_act.size() + (size_t) std::max(0, nact_act_idx(im, fc.gu_act));
                if (!act_done[ai]) {
                    cpu::native_quant_act(fc, gp.h_x + (size_t) tt * H, slot);
                    act_done[ai] = 1;
                }
                jb->nact[jb->nt] = slot;
            }
            jb->out[jb->nt] = im.parts_n.data() + (size_t) j * (size_t) H;
            ++jb->nt;
        }
    }
    if (nj > 0 && std::getenv("DS41_DBG_X")) {
        std::fprintf(stderr, "[dbg_x] nt=%d nu=%d nj=%d nact_multi_n=%zu nfmt=%zu\n", nt, nu, nj,
                     im.nact_multi_n.size(), im.nact_multi_act.size());
        for (int i = 0; i < nj; ++i) {
            const cpu::ExpertJobMulti& j = im.jobs[(size_t) i];
            std::fprintf(stderr, "[dbg_x] job%d blob=%p fmt=%p gu_act=%p gu_type=%d act_bytes=%d nt=%d nact0=%p out0=%p\n",
                         i, (const void*) j.blob, (const void*) j.fmt, j.fmt ? (const void*) j.fmt->gu_act : nullptr,
                         j.fmt ? j.fmt->gu_type : -9, j.fmt ? j.fmt->act_bytes : -9, j.nt, (void*) j.nact[0],
                         (void*) j.out[0]);
        }
    }
    xmark(3);
    if (nj > 0) {
        const double c0 = now_ms();
        im.pool->run_split_multi_native(im.fl[(size_t) im.g.cls(layer, ue[0])], im.jobs.data(), nj);
        st.cpu_ms = now_ms() - c0;
    }
    xmark(4);
    if (nr > 0) {
        Remote& r = *gp.remote;
        const double tw = now_ms();
        while (r.s->done.load(std::memory_order_acquire) != r.n) {
            ds41_remote_pause();
            if (now_ms() - tw > 10000.0) {
                std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE: no answer from the helper in 10 s (layer %lld)\n", (long long) layer);
                std::abort();
            }
        }
        if (r.s->status.load(std::memory_order_relaxed) != 0) {
            std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE: the helper failed a request (status %d, layer %lld)\n",
                         (int) r.s->status.load(), (long long) layer);
            std::abort();
        }
        const double wt = now_ms() - tw;
        r.wait_ms += wt;
        r.wait_max = std::max(r.wait_max, wt);
        ++r.calls;
        r.entries += nr;
        // STRATA_DS41_REMOTE_PROF=1: how long the engine waited for the helper after its own work, per call, over the
        // last 2000 calls (a run average would carry the first request's kernel loading)
        static const bool rprof = std::getenv("STRATA_DS41_REMOTE_PROF") != nullptr;
        if (rprof && r.calls % 2000 == 0) {
            std::fprintf(stderr, "ds41_moe: remote %lld calls: last 2000 %.2f entries/call, %.1f us waited/call (max %.1f)\n",
                         (long long) r.calls, (double) r.entries / 2000.0, 1000.0 * r.wait_ms / 2000.0, 1000.0 * r.wait_max);
            r.entries = 0;
            r.wait_ms = 0;
            r.wait_max = 0;
        }
    }
    const float* rout = gp.remote ? gp.remote->s->out : nullptr;
    if (out_dev) {
        // the helper's rows into their routing-index rows of d_parts, straight from the (registered) segment; the
        // stream sync below ends this call before the next request can overwrite them
        for (int i = 0; i < nr; ++i)
            ck(cudaMemcpyAsync((float*) gp.d_parts + (size_t) rrow[i] * (size_t) H, rout + (size_t) i * (size_t) H,
                               (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "remote row h2d");
        // ---- the weighted sum on the card, per token (as run_to_dev): the pool's rows go up into their
        // routing-index rows of d_parts, the pass's weights through the metadata block, and decode_rows_sum
        // adds one token's K rows in double - the host loop, bit for bit. ----
        for (int j = 0; j < NE; ++j) {
            if (w[j] == 0.0f) continue;
            if (cls[ent_u[j]] != CPU) continue;
            float* hrow = (float*) gp.h_parts + (size_t) j * (size_t) H;
            std::memcpy(hrow, im.parts_n.data() + (size_t) j * (size_t) H, (size_t) H * 4);
            ck(cudaMemcpyAsync((float*) gp.d_parts + (size_t) j * (size_t) H, hrow, (size_t) H * 4, cudaMemcpyHostToDevice,
                               gp.s), "cpu row h2d");
        }
        const size_t o_w = 0;   // [0, kMetaW): the pass's weights (NE <= kMaxEnt floats); the launches' metadata sits above
        static_assert(sizeof(float) * kMaxEnt <= 1024, "the weights must fit below kMetaW");
        std::memcpy(gp.h_meta + o_w, w, sizeof(float) * (size_t) NE);
        if (im.fdef.on)   // the deferred entries are left out here (weight 0: their rows are never read)
            for (int i = 0; i < im.fdef.nent; ++i) ((float*) (gp.h_meta + o_w))[im.fdef.ent_j[i]] = 0.0f;
        ck(cudaMemcpyAsync((uint8_t*) gp.d_meta + o_w, gp.h_meta + o_w, sizeof(float) * (size_t) NE, cudaMemcpyHostToDevice,
                           gp.s), "sum weights");
        for (int t = 0; t < nt; ++t)
            strata::kernels::decode_rows_sum((const float*) gp.d_parts + (size_t) t * (size_t) K * (size_t) H,
                                             (const float*) ((uint8_t*) gp.d_meta + o_w) + (size_t) t * (size_t) K,
                                             (int) K, H, (float*) out_dev + (size_t) t * (size_t) H, gp.s);
        if (prev.on && prev.layer + 1 == layer && prev.nt == nt) {
            // ---- STRATA_FILE_DEFER: the previous layer's deferred experts join this layer's routed sum.  Their reads
            // ran while this call worked; the blobs go up into d_stage slots 0..n-1 (every kernel of this call that
            // reads d_stage is queued before these copies on `s`, and `s` waited for every PCIe-site copy), run on
            // the previous layer's q8_1 rows into d_def_parts (per token: K rows by routing rank, then this layer's
            // sum as row K with weight 1 - exact in double), and decode_rows_sum writes the total back. ----
            gp.rpool_def[prev.par]->wait();
            const int KR = (int) K + 1;
            if (!gp.d_def_parts) ck(cudaMalloc(&gp.d_def_parts, (size_t) kMaxTok * (size_t) KR * (size_t) H * 4), "d_def_parts");
            if (!gp.d_def_w) ck(cudaMalloc((void**) &gp.d_def_w, sizeof(float) * (size_t) kMaxTok * (size_t) KR), "d_def_w");
            if (!gp.h_def_w) ck(cudaMallocHost((void**) &gp.h_def_w, sizeof(float) * (size_t) kMaxTok * (size_t) KR), "h_def_w");
            unsigned long long dp[kMaxDefer];
            for (int d = 0; d < prev.n; ++d) {
                uint8_t* to = (uint8_t*) gp.d_stage + (size_t) d * (size_t) B;
                ck(cudaMemcpyAsync(to, prev.src[d], (size_t) im.bl[(size_t) prev.c[d]], cudaMemcpyHostToDevice, gp.s), "def dma");
                dp[d] = (unsigned long long) to;
            }
            std::fill(gp.h_def_w, gp.h_def_w + (size_t) nt * (size_t) KR, 0.0f);
            for (int t = 0; t < nt; ++t) gp.h_def_w[(size_t) t * KR + K] = 1.0f;
            bool used[kMaxDefer] = {};
            for (int a = 0; a < prev.n; ++a) {   // one grouped launch per format class
                if (used[a]) continue;
                std::vector<unsigned long long> ptr;
                std::vector<int32_t> start, dst, tok;
                for (int d = a; d < prev.n; ++d) {
                    if (used[d] || prev.c[d] != prev.c[a]) continue;
                    used[d] = true;
                    ptr.push_back(dp[d]);
                    start.push_back((int32_t) dst.size());
                    for (int i = 0; i < prev.nent; ++i) {
                        if (prev.ent_d[i] != d) continue;
                        const int j = prev.ent_j[i], t = j / (int) K, k = j % (int) K;
                        dst.push_back(t * KR + k);
                        tok.push_back(t);
                        gp.h_def_w[(size_t) t * KR + k] = prev.w[i];
                    }
                }
                launch_group(prev.c[a], ptr, start, dst, tok, gp.d_xq_def[prev.par], (float*) gp.d_def_parts);
            }
            for (int t = 0; t < nt; ++t)
                ck(cudaMemcpyAsync((float*) gp.d_def_parts + ((size_t) t * KR + K) * (size_t) H, (float*) out_dev + (size_t) t * (size_t) H,
                                   (size_t) H * 4, cudaMemcpyDeviceToDevice, gp.s), "def own row");
            ck(cudaMemcpyAsync(gp.d_def_w, gp.h_def_w, sizeof(float) * (size_t) nt * (size_t) KR, cudaMemcpyHostToDevice, gp.s),
               "def weights");
            for (int t = 0; t < nt; ++t)
                strata::kernels::decode_rows_sum((const float*) gp.d_def_parts + (size_t) t * KR * (size_t) H,
                                                 gp.d_def_w + (size_t) t * KR, KR, H, (float*) out_dev + (size_t) t * (size_t) H, gp.s);
            ++im.fdef_layers;
            im.fdef_experts += prev.n;
            if (im.fdef_layers % 2000 == 0)
                std::fprintf(stderr, "ds41_moe: STRATA_FILE_DEFER=%d: %lld layer calls, %lld experts deferred, %lld dropped\n",
                             defer_r, (long long) im.fdef_layers, (long long) im.fdef_experts, (long long) im.fdef_dropped);
        }
        ck(cudaStreamSynchronize(gp.s), "sum sync");
    } else {
        ck(cudaMemcpyAsync(gp.h_parts, gp.d_parts, (size_t) NE * (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "d2h parts");
        ck(cudaStreamSynchronize(gp.s), "layer sync");
        const float* gpu_parts = (const float*) gp.h_parts;
        int32_t rrow_of[kMaxEnt];   // routing index -> the helper's row (-1: not sent, weight 0)
        for (int j = 0; j < NE; ++j) rrow_of[j] = -1;
        for (int i = 0; i < nr; ++i) rrow_of[rrow[i]] = i;
        for (int t = 0; t < nt; ++t)
            for (int64_t i = 0; i < H; ++i) {
                double acc = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const int j = t * (int) K + (int) k;
                    if (cls[ent_u[j]] == REMOTE && rrow_of[j] < 0) continue;   // weight 0: not sent
                    const float* p = cls[ent_u[j]] == CPU ? im.parts_n.data() + (size_t) j * H
                                   : cls[ent_u[j]] == REMOTE ? rout + (size_t) rrow_of[j] * (size_t) H
                                                             : gpu_parts + (size_t) j * H;
                    acc += (double) w[j] * (double) p[(size_t) i];
                }
                out[(size_t) t * H + i] = (float) acc;
            }
    }
    xmark(5);
    // ---- vram_lru bookkeeping (the stream is synced above: every copy is home) ----
    if (lru) {
        for (int u = 0; u < nu; ++u)   // every lookup that was not a VRAM hit is a miss for the admission filter
            if (cls[u] != HIT || from_pf[u]) im.vmiss[(size_t) (layer * NXL + ue[u])] = lc;
        for (int i = 0; i < n_lru; ++i) {   // the slots now hold the new experts
            gp.cache->replace(layer, lru_old[i], lru_new[i]);
            if (lru_dem[i] >= 0) {   // the arena slot changed hands: the demoted expert in, the promoted one out
                gp.arena.slot_of[(size_t) (layer * NXL + lru_new[i])] = -1;
                gp.arena.slot_of[(size_t) (layer * NXL + lru_old[i])] = lru_dem[i];
                if (!im.last_use.empty()) im.last_use[(size_t) (layer * NXL + lru_old[i])] = im.tick;
                ++st.vram_demotes;
            }
            ++st.vram_swaps;
            static const bool check_lru = std::getenv("DS41_CHECK_LRU") != nullptr;
            if (check_lru) {   // the slot and the demoted arena copy against the file, byte for byte
                static int64_t n_chk = 0, n_bad = 0;
                std::vector<uint8_t> ref((size_t) im.blob);
                std::string e;
                im.blobs->read(layer, lru_new[i], ref.data());
                const bool ok_v = gp.cache->verify_slot(gp.cache->slot_of(layer, lru_new[i]), ref.data(), e, BLX(lru_new[i]));
                bool ok_a = true;
                if (lru_dem[i] >= 0) {
                    im.blobs->read(layer, lru_old[i], ref.data());
                    ok_a = std::memcmp(gp.arena.ptr(layer, lru_old[i]), ref.data(), (size_t) BLX(lru_old[i])) == 0;
                }
                ++n_chk; n_bad += !(ok_v && ok_a);
                if (!(ok_v && ok_a)) std::fprintf(stderr, "[d3-lru-chk] L%lld new e%d slot %s, demoted e%d arena %s\n", (long long) layer,
                                                  lru_new[i], ok_v ? "ok" : "BAD", lru_old[i], ok_a ? "ok" : "BAD");
                if (n_chk % 200 == 0) std::fprintf(stderr, "[d3-lru-chk] %lld swaps checked, %lld bad\n", (long long) n_chk, (long long) n_bad);
            }
        }
    }
    // ---- admission of the distinct misses (the pre-LRU rule; uniform blob sizes only) ----
    if (!lru && gp.cache && !im.cfg.no_cache && !im.mixed_sizes) {
        for (int u = 0; u < nu; ++u) {
            if (cls[u] == HIT) continue;
            const uint8_t* p = gp.arena.ptr(layer, ue[u]);
            if (!p) continue;
            const int32_t sl = gp.cache->admit(layer, ue[u]);
            if (sl < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(sl, p, e2, BLX(ue[u]))) {
                std::fprintf(stderr, "ds4_moe: fill_slot: %s\n", e2.c_str());
                std::abort();
            }
            ++im.admitted;
        }
    }
    // arena_adapt: the stream is synced above (every DMA out of a victim slot is done) - the slots change hands
    int32_t fg_slot[kMaxFg];
    for (int b = 0; b < nfg; ++b)   // (a deferred blob's slot changes hands in the call that adds it, below)
        if (fg_vic[b] >= 0 && !fg_def[b]) fg_slot[b] = arena_publish_swap(im, layer, ue[fg_u[b]], fg_vic[b]);
    for (int b = 0; b < nfg; ++b)
        if (fg_vic[b] >= 0 && !fg_def[b]) arena_check_swap(im, layer, ue[fg_u[b]], fg_slot[b]);
    if (prev.on) {   // STRATA_FILE_DEFER: the previous call's deferred reads are home (waited above, or now) - their
                     // victim slots change hands; a deferral this call could not add (not layer + 1) is counted
        gp.rpool_def[prev.par]->wait();
        if (!(out_dev && prev.layer + 1 == layer && prev.nt == nt)) ++im.fdef_dropped;
        for (int d = 0; d < prev.n; ++d)
            if (prev.vic[d] >= 0) arena_check_swap(im, prev.layer, prev.e[d], arena_publish_swap(im, prev.layer, prev.e[d], prev.vic[d]));
    }
    for (int j = 0; j < NE; ++j) {
        const int c = cls[ent_u[j]];
        if (c == HIT) ++st.hits; else if (c == PCIE || c == FILEG || c == DEFER) ++st.pcie; else if (c == REMOTE) ++st.remote; else ++st.cpu;
    }
    st.prefetched_useful = npf;
    st.wall_ms = now_ms() - t0;
    if (xprof) {
        xmark(6);
        ++xp_n; xp_nu += nu;
        for (int u = 0; u < nu; ++u) if (cls[u] == CPU) { ++xp_cpu; if (!gp.arena.ptr(layer, ue[u])) ++xp_file; }
        if (xp_n % 640 == 0) {
            std::fprintf(stderr, "[prof_x] %ld calls, ms/call: classify %.3f xq %.3f launch %.3f jobs+reads %.3f pool %.3f "
                         "sum+sync %.3f admit %.3f | distinct %.2f/call, cpu %.2f, file %.2f\n", xp_n, xp[0] / xp_n,
                         xp[1] / xp_n, xp[2] / xp_n, xp[3] / xp_n, xp[4] / xp_n, xp[5] / xp_n, xp[6] / xp_n,
                         (double) xp_nu / xp_n, (double) xp_cpu / xp_n, (double) xp_file / xp_n);
            // the pool's own phases since the last print: gate/up, the ff quant, down, and its weight-read rate
            const cpu::ExpertPool& pl = *im.pool;
            const double g = pl.ms_multi_gu - xpool0[0], q = pl.ms_multi_q - xpool0[1], d = pl.ms_multi_down - xpool0[2];
            const int64_t by = pl.multi_bytes - xpool_b0;
            std::fprintf(stderr, "[prof_x] pool since last: gu %.1f ms, q %.1f ms, down %.1f ms, %.1f MB -> %.2f GB/s\n",
                         g, q, d, (double) by / 1e6, (double) by / 1e6 / std::max(1e-9, g + q + d));
            xpool0[0] = pl.ms_multi_gu; xpool0[1] = pl.ms_multi_q; xpool0[2] = pl.ms_multi_down; xpool_b0 = pl.multi_bytes;
        }
    }
    im.st.add(st);
    im.pf_n = 0;
    return true;
}
#endif  // DS4_MOE_CUDA


#if defined(DS4_MOE_CUDA)
/// Prompt chunk: `n` tokens of one layer, every expert on the GPU.  Distinct experts are groups; a resident one runs
/// from its slot (one launch for all of them), the rest are streamed - arena blobs by DMA straight from the pinned
/// arena, file-tier blobs read (8 threads) into a pinned bounce half and DMA'd from there - into two device halves,
/// batch k+1 copying while batch k computes.  Every entry (token, k) writes its own row of c_parts; the weighted sum
/// over k is done on the host.
// MIMO_CHUNK_PROF=1: run_chunk's time split (host waits on the copy events / final sync, GPU time of the grouped
// launches by event pairs), printed by release_chunk.  Timing only - the same work runs.
struct ChunkProf { double file_gb = 0, rd_busy = 0; double wall = 0, file = 0, file_rd = 0, wait_copied = 0, sync = 0, kern = 0, resident_kern = 0; int64_t calls = 0, launches = 0;
                   double tc = 0, tc_gb = 0; int64_t tc_launches = 0, mmq_subruns = 0; };   // + the PTQ1_0 -> Q2_0 transcode inside MMQ
ChunkProf g_cprof;
bool chunk_prof() { static const bool on = [] { const char* e = std::getenv("MIMO_CHUNK_PROF"); return e && *e == '1'; }(); return on; }

#if defined(DS4_MOE_MMQ)
// MmqPlan / mmq_plan / mmq_eff: ds41_mmq_plan.hpp (shared with ds41_helper)
#endif

// The stride run_chunk gives a streamed blob in a device half (BL, or BL rounded to whole blocks of every type when
// the layer runs on MMQ) - the same rule as in gpu_run_chunk (mmq_plan), for chunk_prestage
int64_t chunk_stride(Ds41MoeImpl& im, int64_t layer) {
    const int64_t BL = im.bl[(size_t) layer];
#if defined(DS4_MOE_MMQ)
    Gpu& gp = *im.gpu;
    namespace mmq = strata::prefill::mmq;
    if (!im.cfg.chunk_mmq || !mmq::built()) return BL;
    const MmqPlan P = mmq_plan(gp.gll[(size_t) layer], BL);
    return P.ok ? P.SL : BL;
#else
    return BL;
#endif
}

// chunk_prestage: DMA `layer`'s arena experts that are not VRAM-resident, ascending ids, into device half `h` at the
// run_chunk stride; run_chunk(layer) then only waits for c_copied[h].  The half must hold them all (else: skip).
void chunk_prestage(Ds41MoeImpl& im, int64_t layer, int h) {
    Gpu& gp = *im.gpu;
    gp.c_pre_layer = -1;
    if (!im.g.routed(layer) || !gp.s_cp) return;
    const int64_t BL = im.bl[(size_t) layer], SL = chunk_stride(im, layer), NX = im.g.n_experts;
    std::vector<int32_t> es;
    for (int64_t e = 0; e < NX; ++e) {
        if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
        if (gp.arena.ptr(layer, e)) es.push_back((int32_t) e);
    }
    if (es.empty() || (int64_t) es.size() * SL > gp.c_half) return;
    ck(cudaStreamWaitEvent(gp.s_cp, gp.c_used[h], 0), "prestage wait used");
    for (size_t i = 0; i < es.size(); ++i)
        ck(cudaMemcpyAsync((uint8_t*) gp.c_stage[h] + i * (size_t) SL, gp.arena.ptr(layer, es[i]), (size_t) BL,
                           cudaMemcpyHostToDevice, gp.s_cp), "prestage dma");
    ck(cudaEventRecord(gp.c_copied[h], gp.s_cp), "prestage copied");
    gp.c_pre_layer = (int) layer;
    gp.c_pre_half = h;
    gp.c_pre_sl = SL;
    gp.c_pre_e = std::move(es);
}

// Layer prestage: start DMAing `layer`'s arena experts into the prestage region (see Gpu::p_buf).  Called right after
// the previous layer's expert passes are queued, so the copies run on s_cp while the caller computes `layer`'s dense
// half (a different stream).  The region is reused only after the previous layer's launches from it (p_used).
/// STRATA_PRESTAGE_MIN (rows, default 4096): calls narrower than this get the LAZY prestage (see gpu_run_chunk).
int chunk_pstage_min() {
    static const int pmin = std::getenv("STRATA_PRESTAGE_MIN") ? std::atoi(std::getenv("STRATA_PRESTAGE_MIN")) : 4096;
    return pmin;
}

/// The prestage region, allocated once (false = prestage off for good: STRATA_PRESTAGE_GIB=0 or no room).
bool chunk_pstage_alloc(Ds41MoeImpl& im) {
    Gpu& gp = *im.gpu;
    if (gp.p_off) return false;
    if (!gp.p_buf) {
        const char* env = std::getenv("STRATA_PRESTAGE_GIB");
        const double gib = env ? std::atof(env) : 3.0;
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        // keep: VRAM left for the dense half's pass buffer, which grows with the context (2.3 GB at 20K); 1.5 GiB was
        // too little once the desktop held a few hundred MB more (fresh 20K prompt OOM, 2026-10-12)
        const char* kenv = std::getenv("STRATA_PRESTAGE_KEEP_GIB");
        const size_t want = (size_t) (gib * 1073741824.0), keep = (size_t) ((kenv ? std::atof(kenv) : 3.0) * 1073741824.0);
        gp.p_cap = fr > want + keep ? want : (fr > keep ? fr - keep : 0);
        if (gib <= 0 || gp.p_cap < ((size_t) 256 << 20) || cudaMalloc(&gp.p_buf, gp.p_cap) != cudaSuccess) {
            (void) cudaGetLastError();
            gp.p_buf = nullptr;
            gp.p_off = true;
            return false;
        }
        ck(cudaEventCreateWithFlags(&gp.p_copied, cudaEventDisableTiming), "ev");
        ck(cudaEventCreateWithFlags(&gp.p_used, cudaEventDisableTiming), "ev");
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "chunk prestage: %.2f GiB device region\n", gp.p_cap / 1073741824.0); }
    }
    return true;
}

/// The prestage layout of `layer` (arena experts not VRAM-resident, ascending id, packed at 256 B, up to p_cap) into
/// p_ptr - DMAing every one (`routed` null: the eager prestage) or only those `routed` marks (the lazy one, below).
/// One loop for both, so an expert's region address - and with it the launch grouping and the kernel each run takes
/// (MMVQ from the region, MMQ when streamed) - never depends on which of the two filled it.
int chunk_pstage_fill(Ds41MoeImpl& im, int64_t layer, const int32_t* routed) {
    Gpu& gp = *im.gpu;
    if (gp.p_used_rec) ck(cudaStreamWaitEvent(gp.s_cp, gp.p_used, 0), "pstage wait used");
    const int64_t NX = im.g.n_experts;
    gp.p_ptr.assign((size_t) NX, 0ull);
    size_t off = 0;
    int nput = 0;
    for (int64_t e = 0; e < NX; ++e) {
        if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
        const uint8_t* src = gp.arena.ptr(layer, e);
        if (!src) continue;
        const size_t bl = (size_t) im.bl[(size_t) (im.g.per_expert() ? im.g.cls(layer, (int32_t) e) : layer)];
        if (off + bl > gp.p_cap) break;
        if (!routed || routed[e]) {
            ck(cudaMemcpyAsync((uint8_t*) gp.p_buf + off, src, bl, cudaMemcpyHostToDevice, gp.s_cp), "pstage dma");
            gp.p_ptr[(size_t) e] = (unsigned long long) gp.p_buf + off;
            ++nput;
        }
        off = (off + bl + 255) & ~(size_t) 255;
    }
    if (std::getenv("STRATA_PSTAGE_TRACE")) {   // diagnostic: which layers fill, how many, and the region's packing
        static int said = 0;
        if (said++ < 4) {
            const int64_t c = im.g.per_expert() ? -1 : layer;
            std::fprintf(stderr, "pstage fill L%lld %s: %d experts, blob %llu, packed %llu, MMQ stride %lld (cls %lld)\n",
                         (long long) layer, routed ? "lazy" : "eager", nput, (unsigned long long) im.bl[(size_t) (c >= 0 ? c : 0)],
                         (unsigned long long) ((im.bl[(size_t) (c >= 0 ? c : 0)] + 255) & ~(uint64_t) 255),
                         (long long) chunk_stride(im, c >= 0 ? c : 0), (long long) c);
        }
    }
    if (!nput) return 0;
    ck(cudaEventRecord(gp.p_copied, gp.s_cp), "pstage copied");
    gp.p_layer = (int) layer;
    return nput;
}

void chunk_pstage_issue(Ds41MoeImpl& im, int64_t layer) {
    Gpu& gp = *im.gpu;
    gp.p_layer = -1;
    gp.p_lazy_layer = -1;
    if (gp.p_off || !gp.s_cp || !gp.arena.base || !im.g.routed(layer)) return;
    if (!chunk_pstage_alloc(im)) return;
    chunk_pstage_fill(im, layer, nullptr);
}

// cf >= 0 (per-expert geometry): only the entries whose expert is of class cf; `first` uploads and quantizes the
// activations, `last` takes the weighted sum (each entry's parts row is written by exactly one class pass)
/// STRATA_DS41_REMOTE prompt chunks: post the entries of the helper's experts that are not in VRAM (sorted by class,
/// then expert, so the helper can cut them into contiguous pieces) and mark those experts for this call's passes.
/// False = nothing posted (no entries, too many, or a chunk wider than the segment): the call runs as without it.
bool remote_chunk_post(Ds41MoeImpl& im, int64_t layer, int n, const int32_t* ids, const float* x_host) {
    Gpu& gp = *im.gpu;
    RemoteChunk& rc = *gp.rchunk;
    rc.pending = false;
    Ds41RemoteChunk& C = *rc.c;
    const int64_t NX = im.g.n_experts, K = im.g.top_k, NE = (int64_t) n * K, H = im.g.n_embd;
    if (n > C.cap_tok) return false;
    const uint8_t* holds = gp.remote->s->holds + (size_t) (layer * NX);
    std::fill(rc.mask_e.begin(), rc.mask_e.end(), 0);
    for (int64_t e = 0; e < NX; ++e)
        if (holds[e] && !(gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0)) rc.mask_e[(size_t) e] = 1;
    std::vector<int32_t> cnt((size_t) NX, 0);
    int64_t nr = 0, rest = 0;
    for (int64_t j = 0; j < NE; ++j) {
        if (ids[j] < 0 || ids[j] >= NX) return false;
        if (rc.mask_e[(size_t) ids[j]]) { ++cnt[(size_t) ids[j]]; ++nr; } else ++rest;
    }
    if (nr == 0 || rest == 0 || nr > C.cap_ent) return false;   // rest == 0: the passes need an entry of their own
    // counting sort by (class, expert)
    std::vector<int32_t> eo;
    for (int64_t e = 0; e < NX; ++e) if (cnt[(size_t) e]) eo.push_back((int32_t) e);
    std::stable_sort(eo.begin(), eo.end(), [&](int32_t a, int32_t b) { return im.g.cls(layer, a) < im.g.cls(layer, b); });
    std::vector<int32_t> first((size_t) NX, 0);
    int32_t at = 0;
    for (int32_t e : eo) { first[(size_t) e] = at; at += cnt[(size_t) e]; }
    rc.rrow.resize((size_t) nr);
    int32_t* ee = C.ent_expert();
    int32_t* et = C.ent_tok();
    for (int64_t j = 0; j < NE; ++j) {
        const int32_t e = ids[j];
        if (!rc.mask_e[(size_t) e]) continue;
        const int32_t p = first[(size_t) e]++;
        ee[p] = e;
        et[p] = (int32_t) (j / K);
        rc.rrow[(size_t) p] = (int32_t) j;
    }
    // the rows: a few threads (a 12288-row chunk is 200 MB)
    {
        const size_t total = (size_t) n * (size_t) H * 4;
        const int nth = total > ((size_t) 16 << 20) ? 4 : 1;
        std::vector<std::thread> th;
        for (int q = 1; q < nth; ++q)
            th.emplace_back([&, q]() {
                const size_t a = total * q / nth, b = total * (q + 1) / nth;
                std::memcpy((uint8_t*) C.x() + a, (const uint8_t*) x_host + a, b - a);
            });
        std::memcpy(C.x(), x_host, total / nth);
        for (auto& t : th) t.join();
    }
    C.layer = (int32_t) layer;
    C.n = n;
    C.n_ent = (int32_t) nr;
    C.req.store(++rc.n, std::memory_order_release);
    rc.pending = true;
    rc.nr = (int) nr;
    return true;
}

/// Wait for the helper's rows and scatter them into c_parts (on gp.s, before the weighted sum, which syncs gp.s
/// before the next post can rewrite the segment).
bool remote_chunk_collect(Ds41MoeImpl& im, int64_t layer) {
    Gpu& gp = *im.gpu;
    RemoteChunk& rc = *gp.rchunk;
    rc.pending = false;
    Ds41RemoteChunk& C = *rc.c;
    const double tw = now_ms();
    while (C.done.load(std::memory_order_acquire) != rc.n) {
        if (now_ms() - tw > 60000.0) {
            std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE: no answer from the helper for a prompt chunk in 60 s (layer %lld)\n", (long long) layer);
            return false;
        }
        ds41_remote_pause();
    }
    if (C.status.load(std::memory_order_relaxed) != 0) {
        std::fprintf(stderr, "ds41_moe: STRATA_DS41_REMOTE: the helper failed a prompt chunk (status %d, layer %lld)\n",
                     C.status.load(std::memory_order_relaxed), (long long) layer);
        return false;
    }
    rc.wait_ms += now_ms() - tw;
    if (rc.nr > rc.d_rrow_cap) {
        if (rc.d_rrow) cudaFree(rc.d_rrow);
        rc.d_rrow_cap = std::max<int64_t>(rc.nr, 4096);
        ck(cudaMalloc(&rc.d_rrow, sizeof(int32_t) * (size_t) rc.d_rrow_cap), "remote rows");
    }
    ck(cudaMemcpyAsync(rc.d_rrow, rc.rrow.data(), sizeof(int32_t) * (size_t) rc.nr, cudaMemcpyHostToDevice, gp.s), "remote rows");
    strata::kernels::scatter_rows_f32(rc.out_dev, (float*) gp.c_parts, rc.d_rrow, rc.nr, im.g.n_embd, gp.s);
    ++rc.calls;
    rc.entries += rc.nr;
    static const bool rprof = std::getenv("STRATA_DS41_REMOTE_PROF") != nullptr;
    if (rprof && rc.calls % 40 == 0)
        std::fprintf(stderr, "ds41_moe: remote prompt chunks: %lld calls, %.0f entries/call, %.1f ms waited/call\n",
                     (long long) rc.calls, (double) rc.entries / (double) rc.calls, rc.wait_ms / (double) rc.calls);
    Ds41MoeStats s2;
    s2.remote = rc.nr;
    im.st.add(s2);
    return true;
}

bool gpu_run_chunk(Ds41MoeImpl& im, int64_t layer, int n, const int32_t* ids, const float* w, const float* x_host,
                   float* out, void* out_dev = nullptr, int cf = -1, bool first_pass = true, bool last_pass = true) {
    Gpu& gp = *im.gpu;
    const bool prof = chunk_prof();
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> kev;   // prof: one pair per grouped launch
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> tcev;  // prof: one pair per MMQ sub-run's PTQ1_0 transcode (inside kev)
    bool in_resident = true;
    std::vector<char> kev_res;
    const int64_t H = im.g.n_embd, K = im.g.top_k, NE = (int64_t) n * K;
    const int64_t CI = cf >= 0 ? cf : layer;          // the format class this pass computes
    const int64_t BL = im.bl[(size_t) CI];
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) CI];
    const char* rmask = gp.rchunk && gp.rchunk->pending ? gp.rchunk->mask_e.data() : nullptr;   // STRATA_DS41_REMOTE
    auto in_pass = [&](int64_t j) { return (cf < 0 || im.g.cls(layer, ids[j]) == cf) && !(rmask && rmask[ids[j]]); };
    const double t0 = now_ms();
    Ds41MoeStats st;

    if (n > gp.c_tok) {   // (re)allocate the DEVICE buffers for n tokens (release_chunk gives them back)
        for (void* p : {gp.c_x, gp.c_xq, gp.c_parts, gp.c_scratch, gp.c_dst, gp.c_tokv, gp.c_w, gp.c_out}) if (p) cudaFree(p);
        gp.c_tok = n;
        const int64_t ne = (int64_t) n * K;
        ck(cudaMalloc(&gp.c_x, (size_t) n * (size_t) H * 4), "c_x");
        ck(cudaMalloc(&gp.c_xq, (size_t) n * strata::kernels::native_q8_1_bytes((int) H, 1)), "c_xq");   // its ncols <= 8
        ck(cudaMalloc(&gp.c_parts, (size_t) ne * (size_t) H * 4), "c_parts");
        ck(cudaMalloc(&gp.c_scratch, strata::kernels::native_expert_scratch_bytes(ne, im.g.n_ff)), "c_scratch");
        ck(cudaMalloc(&gp.c_dst, sizeof(int32_t) * (size_t) ne), "c_dst");
        ck(cudaMalloc(&gp.c_tokv, sizeof(int32_t) * (size_t) ne), "c_tok");
        ck(cudaMalloc(&gp.c_w, sizeof(float) * (size_t) ne), "c_w");
        ck(cudaMalloc(&gp.c_out, (size_t) n * (size_t) H * 4), "c_out");
    }
    if (n > gp.c_pin_tok) {   // the PINNED landing pad persists across prompts (see Gpu::c_pin_tok)
        if (gp.c_hparts) cudaFreeHost(gp.c_hparts);
        gp.c_hparts = nullptr;
        gp.c_pin_tok = n;
        ck(cudaMallocHost((void**) &gp.c_hparts, (size_t) n * (size_t) H * 4), "c_hparts");   // the summed rows
    }
    if (!gp.c_ready) {
        const char* mib = std::getenv("MIMO_CHUNK_STAGE_MIB");
        gp.c_half = (int64_t) (mib ? std::atoll(mib) : 1024) * 1048576;
        gp.c_half = std::max<int64_t>(gp.c_half, im.blob);
        // The stage is a SPEED knob (how many file-tier blobs a chunk streams per wave), not a requirement: size it
        // from the VRAM that is actually free, so a card shared with other GPU users runs smaller waves instead of
        // failing the prompt ("ds4_moe: c_stage: out of memory" - s13 run10, 12.5 GiB held by another process, the
        // prompt answered HTTP 503).  768 MiB stays free for the batch's own columns / graph / part buffers (the
        // caller caps the batch width from what is left: ds41_generate.cpp's pb_cap).
        {
            size_t cfr = 0, ctot = 0;
            if (cudaMemGetInfo(&cfr, &ctot) == cudaSuccess) {
                const int64_t room = ((int64_t) cfr - (int64_t) (768LL << 20)) / 2 - (int64_t) (1 << 20);
                // a hard floor too: a card with less than the reserve left still gets 16 MiB a half (one blob a wave),
                // never a failed allocation
                const int64_t floor_b = std::max<int64_t>(im.blob, (int64_t) (16LL << 20));
                const int64_t half = std::min<int64_t>(gp.c_half, std::max<int64_t>(room, floor_b));
                if (half < gp.c_half) {
                    std::fprintf(stderr, "ds4_moe: chunk stage %.0f -> %.0f MiB (%.2f GiB of VRAM free)\n",
                                 (double) gp.c_half / 1048576.0, (double) half / 1048576.0, (double) cfr / 1073741824.0);
                    gp.c_half = half;
                }
            }
        }
        if (gp.c_bhalf <= 0) gp.c_bhalf = gp.c_half;   // the pinned bounce halves persist; keep their size
        else gp.c_bhalf = std::min<int64_t>(gp.c_bhalf, gp.c_half);   // a stage smaller than the bounce would overflow it
        // prestage: one device half holds a whole layer (every expert at the MMQ stride, <= blob + 1 MiB each)
        if (im.cfg.chunk_prestage)
            gp.c_half = std::max<int64_t>(gp.c_half, im.g.n_experts * (im.blob + 1048576));
        const int64_t ng = im.g.n_experts;
        ck(cudaMalloc(&gp.c_ptr, sizeof(unsigned long long) * (size_t) ng), "c_ptr");
        ck(cudaMalloc(&gp.c_start, sizeof(int32_t) * (size_t) (ng + 1)), "c_start");
        ck(cudaMalloc(&gp.c_ng, sizeof(int32_t)), "c_ng");
        for (int h = 0; h < 2; ++h) {
            // + 1 MiB: the MMQ slot stride rounds a blob up (<= ~0.3 MiB) and MMQ reads up to 4 KiB past a matrix;
            // zeroed once, so whatever MMQ reads past a blob decodes to finite numbers (llama.cpp zero-pads too)
            ck(cudaMalloc(&gp.c_stage[h], (size_t) gp.c_half + 1048576), "c_stage");
            ck(cudaMemset(gp.c_stage[h], 0, (size_t) gp.c_half + 1048576), "c_stage zero");
            if (!gp.c_bounce[h]) ck(cudaMallocHost((void**) &gp.c_bounce[h], (size_t) gp.c_bhalf), "c_bounce");
            ck(cudaEventCreateWithFlags(&gp.c_copied[h], cudaEventDisableTiming), "ev");
            ck(cudaEventCreateWithFlags(&gp.c_used[h], cudaEventDisableTiming), "ev");
            ck(cudaEventRecord(gp.c_used[h], gp.s), "ev");
        }
        ck(cudaStreamCreateWithFlags(&gp.s_cp, cudaStreamNonBlocking), "s_cp");
        gp.c_ready = true;
    }
    im.pf_n = 0;   // a chunk does not use the decode prefetch staging

    // ---- MMQ (Ds41MoeConfig::chunk_mmq): this layer's types covered -> streamed blobs sit `SL` apart in a half (BL
    // rounded up to whole blocks of every type, so each matrix is one MMQ "expert stride" away from the next)
    bool use_mmq = false;
    int64_t SL = BL;
#if defined(DS4_MOE_MMQ)
    namespace mmq = strata::prefill::mmq;
    MmqPlan MP;
    if (im.cfg.chunk_mmq && mmq::built()) {
        if (gp.m_ok.size() != gp.gll.size()) gp.m_ok.assign(gp.gll.size(), -1);
        int8_t& ok = gp.m_ok[(size_t) CI];
        MP = mmq_plan(GL, BL);
        if (ok < 0) {
            ok = MP.ok;
            if (!ok) std::fprintf(stderr, "ds4_moe: run_chunk: layer %lld stays on MMVQ (types %d/%d/%d)\n",
                                  (long long) layer, GL.gu_type, GL.up_type >= 0 ? GL.up_type : GL.gu_type, GL.d_type);
        }
        use_mmq = ok > 0 && MP.ok;
    }
    const int gt = MP.gt, ut = MP.ut, dt = MP.dt;   // the types MMQ multiplies (PTQ1_0 -> its Q2_0 copy)
    if (use_mmq) {
        SL = MP.SL;
        const int64_t cap = std::max<int64_t>(16384, n);
        if (use_mmq && gp.m_rows < cap) {
            gp.free_mmq();
            gp.m_rows = cap;
            const int64_t FF = GL.n_ff;
            ck(cudaMalloc(&gp.m_gu, (size_t) cap * (size_t) (2 * FF) * 4), "m_gu");
            ck(cudaMalloc(&gp.m_h, (size_t) cap * (size_t) FF * 4), "m_h");
            ck(cudaMalloc(&gp.m_xg, mmq::q8_bytes(cap, H)), "m_xg");
            ck(cudaMalloc(&gp.m_xu, mmq::q8_bytes(cap, H)), "m_xu");
            ck(cudaMalloc(&gp.m_xd, mmq::q8_bytes(cap, FF)), "m_xd");
            ck(cudaMalloc(&gp.m_bnd, sizeof(int32_t) * (size_t) (im.g.n_experts + 1)), "m_bnd");
            ck(cudaMalloc(&gp.m_iota, sizeof(int32_t) * (size_t) cap), "m_iota");
            mmq::iota((int32_t*) gp.m_iota, cap, gp.s);
            if (!gp.m_ctx) gp.m_ctx = new mmq::Context();   // never deleted (see Gpu::m_ctx)
        }
    }
#endif

    // ---- groups: distinct experts with their entries (token-major entry j = t*K + k writes parts row j) ----
    const int64_t NX = im.g.n_experts;
    std::vector<int32_t> cnt((size_t) NX, 0);
    int64_t NEP = 0;   // entries of this pass
    for (int64_t j = 0; j < NE; ++j) {
        if (ids[j] < 0 || ids[j] >= NX) return false;
        if (!in_pass(j)) continue;
        ++cnt[(size_t) ids[j]];
        ++NEP;
    }
    // processing order: VRAM-resident experts, then arena (DMA straight away), then file tier (read in the background
    // from the start of the call, so the arena DMAs and the reads overlap); entries are laid out in this order, so
    // consecutive experts of one class are adjacent in the sorted entry arrays (one launch per run)
    std::vector<int32_t> order[3];
    std::vector<int32_t> slot_e((size_t) NX, -1);
    for (int64_t e = 0; e < NX; ++e) {
        if (!cnt[(size_t) e]) continue;
        const int32_t sl = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        slot_e[(size_t) e] = sl;
        order[sl >= 0 ? 0 : gp.arena.ptr(layer, e) ? 1 : 2].push_back((int32_t) e);
    }
    if (first_pass) rebal_count(im, layer, ids, NE);   // the conversation-adaptive arena's usage counts (all classes)
    // STRATA_ROUTE_LOG=<file> (diagnostic, off by default): every routed expert of this batch pass with its tier
    // (0 VRAM cache, 1 RAM arena, 2 file) and token count - turnbench's cross-turn overlap analysis reads it
    if (static FILE* rlog = std::getenv("STRATA_ROUTE_LOG") ? std::fopen(std::getenv("STRATA_ROUTE_LOG"), "a") : nullptr; rlog) {
        for (int t = 0; t < 3; ++t)
            for (int32_t e : order[t]) std::fprintf(rlog, "%lld %d %d %d\n", (long long) layer, e, t, cnt[(size_t) e]);
        std::fflush(rlog);
    }
    std::vector<int32_t> first((size_t) NX + 1, 0);
    {
        int32_t at = 0;
        for (const auto& o : order)
            for (int32_t e : o) {
                first[(size_t) e] = at;
                at += cnt[(size_t) e];
            }
    }
    std::vector<int32_t> dst((size_t) std::max<int64_t>(NEP, 1)), tok((size_t) std::max<int64_t>(NEP, 1)),
        fill(first.begin(), first.end() - 1);
    for (int64_t j = 0; j < NE; ++j) {
        if (!in_pass(j)) continue;
        const int32_t p = fill[(size_t) ids[j]]++;
        dst[(size_t) p] = (int32_t) j;
        tok[(size_t) p] = (int32_t) (j / K);
    }
    // entry arrays sorted by expert: uploaded once; each launch passes its groups' slices via grp_start offsets
    // (a later class pass reuses the buffers: same stream, so the previous pass's kernels read them first)
    ck(cudaMemcpyAsync(gp.c_dst, dst.data(), sizeof(int32_t) * (size_t) NEP, cudaMemcpyHostToDevice, gp.s), "c_dst");
    ck(cudaMemcpyAsync(gp.c_tokv, tok.data(), sizeof(int32_t) * (size_t) NEP, cudaMemcpyHostToDevice, gp.s), "c_tok");

    // ---- activations (the first class pass only) ----
    if (first_pass) ck(cudaMemcpyAsync(gp.c_x, x_host, (size_t) n * (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "c_x h2d");
    if (first_pass) {   // the MMVQ quantizer takes <= 8 columns per call; columns are contiguous per token, so step through them
        const size_t col = strata::kernels::native_q8_1_bytes((int) H, 1);
        if (strata::kernels::native_q8_1_bytes((int) H, 8) != 8 * col) {
            std::fprintf(stderr, "ds4_moe: run_chunk: q8_1 layout is not per-column contiguous\n");
            return false;
        }
        for (int c0 = 0; c0 < n; c0 += 8)
            strata::kernels::native_quantize_q8_1((const float*) gp.c_x + (size_t) c0 * (size_t) H,
                                                  (uint8_t*) gp.c_xq + (size_t) c0 * col, (int) H, std::min(8, n - c0), gp.s);
    }

    // a launch over groups `es` (expert ids) whose blobs are at `ptrs`: grp_start are offsets into the sorted entries
    std::vector<int32_t> lstart;
#if defined(DS4_MOE_MMQ)
    // MMQ over experts es[q0, q1) at ptrs[q0] + (q - q0) * SL: three products (gate, up -> SwiGLU -> down) over their
    // sorted entries [r0, r1); the down product writes each entry's parts row through c_dst
    std::vector<int32_t> mbnd;
    auto mmq_launch = [&](const std::vector<int32_t>& es, const std::vector<unsigned long long>& ptrs, size_t q0,
                          size_t q1) {
        const int64_t r0 = first[(size_t) es[q0]], r1 = first[(size_t) es[q1 - 1]] + cnt[(size_t) es[q1 - 1]];
        const int64_t R = r1 - r0, FF = GL.n_ff;
        int64_t maxr = 0;
        mbnd.clear();
        for (size_t q = q0; q < q1; ++q) {
            mbnd.push_back(first[(size_t) es[q]] - (int32_t) r0);
            maxr = std::max<int64_t>(maxr, cnt[(size_t) es[q]]);
        }
        mbnd.push_back((int32_t) R);
        ck(cudaMemcpyAsync(gp.m_bnd, mbnd.data(), sizeof(int32_t) * mbnd.size(), cudaMemcpyHostToDevice, gp.s), "m_bnd");
        const int32_t* tokr = (const int32_t*) gp.c_tokv + r0;
        mmq::quantize((const float*) gp.c_x, tokr, gp.m_xg, gt, H, H, R, gp.s);
        if (ut != gt) mmq::quantize((const float*) gp.c_x, tokr, gp.m_xu, ut, H, H, R, gp.s);
        const uint8_t* base = (const uint8_t*) ptrs[q0];
        // PTQ1_0 matrices of the run -> their Q2_0 copies in m_tc (one launch per transcoded role; the staged blobs
        // are SL apart, the copies SL2 apart; same stream, so the copies are done before the products read them)
        const int nq = (int) (q1 - q0);
        uint8_t* tc = (uint8_t*) gp.m_tc;
        cudaEvent_t tc0 = nullptr, tc1 = nullptr;
        const bool tcp = prof && (MP.tg || MP.tu || MP.td);
        if (tcp) {
            cudaEventCreate(&tc0);
            cudaEventCreate(&tc1);
            cudaEventRecord(tc0, gp.s);
        }
        if (MP.tg) ds41_ptq1_to_q2_0_strided(base, (size_t) SL, tc + MP.g2, MP.SL2, MP.g_bytes, nq, gp.s);
        if (MP.tu) ds41_ptq1_to_q2_0_strided(base + GL.up_off, (size_t) SL, tc + MP.u2, MP.SL2, MP.u_bytes, nq, gp.s);
        if (MP.td) ds41_ptq1_to_q2_0_strided(base + GL.down_off, (size_t) SL, tc + MP.d2, MP.SL2, MP.d_bytes, nq, gp.s);
        if (tcp) {
            cudaEventRecord(tc1, gp.s);
            tcev.emplace_back(tc0, tc1);
            g_cprof.tc_gb += (double) nq * (double) (MP.g_bytes + MP.u_bytes + MP.d_bytes) / 1e9;
        }
        if (prof) ++g_cprof.mmq_subruns;
        mmq::Product p;
        p.n = nq;
        p.bounds = (const int32_t*) gp.m_bnd;
        p.total_rows = R;
        p.max_rows = maxr;
        p.w = MP.tg ? tc + MP.g2 : base; p.expert_bytes = MP.tg ? MP.SL2 : (size_t) SL;
        p.type = gt; p.w_rows = FF; p.w_cols = H; p.xq = gp.m_xg;
        p.ids = (const int32_t*) gp.m_iota; p.dst = (float*) gp.m_gu; p.ld_dst = 2 * FF;
        gp.m_ctx->run(p, gp.s);
        p.w = MP.tu ? tc + MP.u2 : base + GL.up_off; p.expert_bytes = MP.tu ? MP.SL2 : (size_t) SL;
        p.type = ut; p.xq = ut != gt ? gp.m_xu : gp.m_xg; p.dst = (float*) gp.m_gu + FF;
        gp.m_ctx->run(p, gp.s);
        if (std::isfinite(GL.swiglu_limit))   // DeepSeek-V4 (every layer clamps at 10)
            mmq::swiglu_clamp((const float*) gp.m_gu, (float*) gp.m_h, R, FF, false, GL.swiglu_limit, gp.s);
        else
            mmq::swiglu((const float*) gp.m_gu, (float*) gp.m_h, R, FF, false, gp.s);
        mmq::quantize((const float*) gp.m_h, nullptr, gp.m_xd, dt, FF, FF, R, gp.s);
        p.w = MP.td ? tc + MP.d2 : base + GL.down_off; p.expert_bytes = MP.td ? MP.SL2 : (size_t) SL;
        p.type = dt; p.w_rows = H; p.w_cols = FF; p.xq = gp.m_xd;
        p.ids = (const int32_t*) gp.c_dst + r0; p.dst = (float*) gp.c_parts; p.ld_dst = H;
        gp.m_ctx->run(p, gp.s);
    };
#endif
    auto launch = [&](const std::vector<int32_t>& es, const std::vector<unsigned long long>& ptrs) {
        if (es.empty()) return;
#if defined(DS4_MOE_MMQ)
        bool strided = use_mmq;
        for (size_t q = 1; strided && q < ptrs.size(); ++q) strided = ptrs[q] == ptrs[0] + q * (unsigned long long) SL;
        // the run's PTQ1_0 copies need m_tc: sized for one device half's experts, capped by free VRAM (a sub-run
        // then holds at most tc_max experts); none -> this run stays on MMVQ
        size_t tc_max = SIZE_MAX;
        if (strided && MP.SL2 > 0) {
            const size_t want = (size_t) std::max<int64_t>(1, gp.c_half / SL) * MP.SL2;
            if (gp.m_tc_cap < want) {
                if (gp.m_tc) cudaFree(gp.m_tc);
                gp.m_tc = nullptr;
                gp.m_tc_cap = 0;
                size_t fr = 0, tot = 0;
                cudaMemGetInfo(&fr, &tot);
                const size_t keep = (size_t) 2 << 30;   // room for the dense pass buffer (grows with the prompt)
                // at most STRATA_MMQ_TC_GIB (1): taking all the free VRAM squeezed the next-layer prestage region
                // (3.00 -> 1.37 GiB) and cost the dense half ~1.2 s on an 8K prompt; 1 GiB = ~80 transcoded experts
                // per MMQ launch
                const char* tce = std::getenv("STRATA_MMQ_TC_GIB");
                const size_t tc_cap = (size_t) ((tce ? std::atof(tce) : 1.0) * 1073741824.0);
                size_t got = fr > keep ? std::min(std::min(want, tc_cap), fr - keep) : 0;
                got = got / MP.SL2 * MP.SL2;
                if (got >= MP.SL2 && cudaMalloc(&gp.m_tc, got) == cudaSuccess) gp.m_tc_cap = got;
                else { (void) cudaGetLastError(); gp.m_tc = nullptr; }
            }
            tc_max = gp.m_tc_cap / MP.SL2;
            if (tc_max == 0) strided = false;
        }
        if (strided) {   // sub-runs of at most m_rows entries (one expert has <= n <= m_rows) and tc_max experts
            cudaEvent_t e0 = nullptr, e1 = nullptr;
            if (prof) {
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, gp.s);
            }
            size_t q0 = 0;
            int64_t rows = 0;
            for (size_t q = 0; q < es.size(); ++q) {
                if (rows + cnt[(size_t) es[q]] > gp.m_rows || q - q0 >= tc_max) {
                    mmq_launch(es, ptrs, q0, q);
                    q0 = q;
                    rows = 0;
                }
                rows += cnt[(size_t) es[q]];
            }
            mmq_launch(es, ptrs, q0, es.size());
            if (prof) {
                cudaEventRecord(e1, gp.s);
                kev.emplace_back(e0, e1);
                kev_res.push_back(in_resident ? 1 : 0);
            }
            return;
        }
#endif
        lstart.clear();
        // cap_entries = NE, not this launch's count: the kernel's scratch regions are laid out by it and entries are
        // indexed absolutely (grp_start); its SwiGLU pass visits only [grp_start[0], grp_start[ng]).
        for (int32_t e : es) lstart.push_back(first[(size_t) e]);
        // the kernel walks entries [grp_start[g], grp_start[g+1]); groups here are not adjacent in the sorted
        // arrays, so each launch covers ONE contiguous run: callers pass experts in ascending id order and runs are
        // split where they are not adjacent (see below)
        lstart.push_back(first[(size_t) es.back()] + cnt[(size_t) es.back()]);
        const int32_t ng = (int32_t) es.size();
        ck(cudaMemcpyAsync(gp.c_ptr, ptrs.data(), sizeof(unsigned long long) * es.size(), cudaMemcpyHostToDevice, gp.s), "c_ptr");
        ck(cudaMemcpyAsync(gp.c_start, lstart.data(), sizeof(int32_t) * lstart.size(), cudaMemcpyHostToDevice, gp.s), "c_start");
        ck(cudaMemcpyAsync(gp.c_ng, &ng, sizeof(int32_t), cudaMemcpyHostToDevice, gp.s), "c_ng");
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        if (prof) {
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, gp.s);
        }
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) gp.c_ptr, (const int32_t*) gp.c_start,
                                              (const int32_t*) gp.c_ng, (const int32_t*) gp.c_dst,
                                              (const int32_t*) gp.c_tokv, ng, NE, gp.c_xq, gp.c_scratch,
                                              (float*) gp.c_parts, gp.s, ng);
        if (prof) {
            cudaEventRecord(e1, gp.s);
            kev.emplace_back(e0, e1);
            kev_res.push_back(in_resident ? 1 : 0);
        }
        // the host vectors above are pageable: cudaMemcpyAsync has staged them before returning
    };
    // ---- resident experts: adjacent in the entry order -> one launch
    std::vector<int32_t> run_e;
    std::vector<unsigned long long> run_p;
    int64_t ent_hit = 0, ent_str = 0;
    for (int32_t e : order[0]) {
        run_e.push_back(e);
        run_p.push_back((unsigned long long) gp.cache->device_slot(slot_e[(size_t) e]));
        ent_hit += cnt[(size_t) e];
    }
    launch(run_e, run_p);
    in_resident = false;

    // ---- streamed experts: arena ones, then the file tier, in batches that fit one device half.  File-tier blobs
    // are read by a background pool into pinned slots (both bounce halves, `cap` blobs; more = waves) from here on;
    // a batch waits only for the blobs it DMAs.
    // chunk_prestage: this layer's arena experts may already sit in a device half (issued by the previous call)
    std::vector<int32_t> arena_e(order[1]);
    // lazy prestage (a call narrower than STRATA_PRESTAGE_MIN, which skipped the eager one): the same region layout,
    // filled now that the routing is known with only the arena experts this call routes - production's kernels and
    // numerics, without DMAing the ~40% of a small turn's layer it never uses
    // (first_pass: once per layer, marking the routed arena experts of EVERY format class - the eager fill covers all
    // classes too, and a later class pass must find its experts in the region exactly as it would have)
    if (gp.p_lazy_layer == (int) layer && first_pass && gp.s_cp && gp.arena.base && !gp.p_off && chunk_pstage_alloc(im)) {
        gp.p_lazy_layer = -1;
        std::vector<int32_t> routed((size_t) NX, 0);
        int nr = 0;
        for (int64_t j = 0; j < NE; ++j) {
            const int32_t e = ids[j];
            if (routed[(size_t) e] || !gp.arena.ptr(layer, e) || (rmask && rmask[e])) continue;
            if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
            routed[(size_t) e] = 1;
            ++nr;
        }
        gp.p_layer = -1;
        if (nr) chunk_pstage_fill(im, layer, routed.data());
    }
    if (gp.p_layer == (int) layer && !arena_e.empty()) {   // layer prestage: launch the arena experts already on the card
        std::vector<int32_t> es, rest;
        std::vector<unsigned long long> ps;
        for (int32_t e : arena_e) {
            const unsigned long long pp = gp.p_ptr[(size_t) e];
            if (!pp) { rest.push_back(e); continue; }
            es.push_back(e);
            ps.push_back(pp);
            ent_str += cnt[(size_t) e];
        }
        if (!es.empty()) {
            ck(cudaStreamWaitEvent(gp.s, gp.p_copied, 0), "wait pstage");
            std::vector<int32_t> re;
            std::vector<unsigned long long> rp;
            for (size_t q = 0; q < es.size(); ++q) {
                if (!re.empty() && first[(size_t) re.back()] + cnt[(size_t) re.back()] != first[(size_t) es[q]]) {
                    launch(re, rp);
                    re.clear();
                    rp.clear();
                }
                re.push_back(es[q]);
                rp.push_back(ps[q]);
            }
            launch(re, rp);
            ck(cudaEventRecord(gp.p_used, gp.s), "pstage used");
            gp.p_used_rec = true;
        }
        arena_e = std::move(rest);
    }
    int h0 = 0;   // the first half the batch loop below uses
    bool used_pre = false;
    if (gp.c_pre_layer == (int) layer && gp.c_pre_sl == SL) {
        std::vector<int32_t> pidx((size_t) NX, -1);
        for (size_t i = 0; i < gp.c_pre_e.size(); ++i) pidx[(size_t) gp.c_pre_e[i]] = (int32_t) i;
        std::vector<int32_t> es, rest;
        std::vector<unsigned long long> ps;
        for (int32_t e : order[1]) {
            if (pidx[(size_t) e] < 0) { rest.push_back(e); continue; }
            es.push_back(e);
            ps.push_back((unsigned long long) ((uint8_t*) gp.c_stage[gp.c_pre_half] + (size_t) pidx[(size_t) e] * (size_t) SL));
            ent_str += cnt[(size_t) e];
        }
        if (!es.empty()) {
            ck(cudaStreamWaitEvent(gp.s, gp.c_copied[gp.c_pre_half], 0), "wait prestaged");
            // runs adjacent in the entry order AND in the half (strided: one MMQ launch per run)
            std::vector<int32_t> re;
            std::vector<unsigned long long> rp;
            for (size_t q = 0; q < es.size(); ++q) {
                if (!re.empty() && (first[(size_t) re.back()] + cnt[(size_t) re.back()] != first[(size_t) es[q]] ||
                                    rp.back() + (unsigned long long) SL != ps[q])) {
                    launch(re, rp);
                    re.clear();
                    rp.clear();
                }
                re.push_back(es[q]);
                rp.push_back(ps[q]);
            }
            launch(re, rp);
            ck(cudaEventRecord(gp.c_used[gp.c_pre_half], gp.s), "used prestaged");
            used_pre = true;
        }
        arena_e = std::move(rest);
        h0 = 1 - gp.c_pre_half;
    }
    gp.c_pre_layer = -1;
    std::vector<int32_t> stream_e(arena_e);
    stream_e.insert(stream_e.end(), order[2].begin(), order[2].end());
    for (int32_t e : stream_e) ent_str += cnt[(size_t) e];
    const size_t nA = arena_e.size(), nF = order[2].size();
    // ---- chunk_prefetch: layer l's file-tier experts may already sit in a pinned slab (read while layer l-1 ran).
    // The slab holds EVERY file-tier expert of the layer, so this pass's own file-tier set is a subset of it; if it
    // is not (a stale slab, or no slab), the on-demand wave reader below runs exactly as before.
    int64_t pf_slab = -1;
    if (im.cstage && nF && im.cstage->layer[im.cstage->cur] == layer) {
        const std::vector<int32_t>& si = im.cstage->idx[im.cstage->cur];
        bool covers = true;
        for (size_t k = 0; k < nF && covers; ++k) covers = si[(size_t) order[2][k]] >= 0;
        if (covers) pf_slab = im.cstage->cur;
    }
    if (im.cstage && nF) {
        g_pf.dmas += (int64_t) nF;
        if (pf_slab < 0) ++g_pf.off;
    }
    const int64_t per_half = std::max<int64_t>(1, gp.c_half / SL);
    const size_t slots_half = (size_t) std::max<int64_t>(1, gp.c_bhalf / BL), cap = 2 * slots_half;
    auto slot_ptr = [&](size_t k) { return gp.c_bounce[(k % cap) / slots_half] + ((k % cap) % slots_half) * (size_t) BL; };
    std::vector<std::atomic<int>> ready(nF), pdone(nF);   // pdone: parts of a blob read (STRATA_WAVE_PARTS)
    for (auto& r : ready) r.store(0, std::memory_order_relaxed);
    for (auto& r : pdone) r.store(0, std::memory_order_relaxed);
    std::atomic<size_t> rnext{0}, rdone{0};
    std::atomic<double> t_rd_last{0};
    size_t wave_end = 0;   // reads issued for file blobs [0, wave_end)
    std::vector<std::thread> readers;
    double t_file = 0, t_rd0 = 0;
    auto start_wave = [&]() {   // the next `cap` file blobs (their slots must be free: the caller made sure)
        for (auto& t : readers) t.join();
        readers.clear();
        const size_t w0 = wave_end, w1 = std::min(nF, wave_end + cap);
        wave_end = w1;
        // STRATA_WAVE_THREADS (default 32, was 8): the on-demand reader's threads - small batch turns read only their
        // routed file-tier experts here, so this is their NVMe queue depth.  s11, +272-token turn at 512K: 8 / 16 / 32
        // threads -> batch 14.89 / 13.78 / 13.55 s (3.17 / 3.61 / 3.79 GB/s)
        static const int wave_threads = std::getenv("STRATA_WAVE_THREADS") ? std::max(1, std::min(64, std::atoi(std::getenv("STRATA_WAVE_THREADS")))) : 32;
        // STRATA_WAVE_PARTS (default 8): each blob split into parts read by different threads (read_part, as D3) -
        // a blob is 3 tensor spans, and one thread reading them in turn left the queue shallow.  1 = whole blobs.
        // s11, +272-token turn (32 threads): parts 1 / 4 / 8 -> 3.60 / 5.02 / 5.29 GB/s, batch 14.25 / 11.17 / 10.68 s,
        // tokens identical (exact routing)
        static const size_t P = (size_t) (std::getenv("STRATA_WAVE_PARTS") ? std::max(1, std::min(16, std::atoi(std::getenv("STRATA_WAVE_PARTS")))) : 8);
        // REVERTED 2026-10-15 21:10 (s13c): STRATA_READ_SORTED (issue a wave's parts in FILE-OFFSET order) was tried
        // and SEGFAULTED in the batch (exit 139) - see V41_START_HERE's 13c block.  Not worth re-attempting blind:
        // the source's own note puts the whole scattered-vs-dd gap at ~9% (4.65 vs 5.1 GB/s), and at PROMPT size the
        // two orderable knobs (STRATA_WAVE_PARTS 4 / 16) measured 0% (320.9 / 316.5 / 320.9 tok/s), i.e. inside the
        // 1.5-4.3% noise.  The prefill lever is the batch width (fewer whole-file-tier sweeps), not read order.
        rnext.store(w0 * P);
        const int nth = (int) std::min<size_t>((size_t) wave_threads, (w1 - w0) * P);
        for (int k = 0; k < nth; ++k)
            readers.emplace_back([&, w1]() {
                for (size_t x; (x = rnext.fetch_add(1)) < w1 * P;) {
                    const size_t q = x / P;
                    if (P == 1) im.blobs->read(layer, order[2][q], slot_ptr(q));
                    else im.blobs->read_part(layer, order[2][q], (int) (x % P), (int) P, slot_ptr(q));
                    if (P == 1 || pdone[q].fetch_add(1, std::memory_order_acq_rel) + 1 == (int) P) {   // the blob's last part
                        ready[q].store(1, std::memory_order_release);
                        ready[q].notify_one();
                        if (prof && rdone.fetch_add(1) + 1 == nF) t_rd_last.store(now_ms());
                    }
                }
            });
    };
    if (nF && pf_slab < 0) {
        // the bounce slots are shared by every class pass: the previous pass returned (non-last passes do not
        // sync) with its DMAs out of them possibly still queued on s_cp - the readers must not overwrite them yet
        ck(cudaStreamSynchronize(gp.s_cp), "bounce free");
        t_rd0 = now_ms();
        start_wave();
    }
    int h_last = used_pre ? 1 - h0 : -1;
    for (size_t b0 = 0, bi = 0; b0 < stream_e.size(); b0 += (size_t) per_half, ++bi) {
        const size_t b1 = std::min(stream_e.size(), b0 + (size_t) per_half);
        const int h = (int) ((h0 + bi) & 1);
        h_last = h;
        // the half is free once the kernel that last read it is done
        ck(cudaStreamWaitEvent(gp.s_cp, gp.c_used[h], 0), "wait used");
        std::vector<int32_t> es;
        std::vector<unsigned long long> ps;
        for (size_t i = b0; i < b1; ++i) {
            const uint8_t* src;
            if (i < nA) {
                src = gp.arena.ptr(layer, stream_e[i]);   // (arena_e: not prestaged)
            } else if (pf_slab >= 0) {
                const size_t k = i - nA;
                const int32_t q = im.cstage->idx[pf_slab][(size_t) order[2][k]];
                src = im.cstage->slab[pf_slab] + im.cstage->off[pf_slab][(size_t) q];
                ++g_pf.used;
            } else {
                const size_t k = i - nA;
                if (k >= wave_end) {   // a new wave reuses the slots: every DMA out of them must be done first
                    const double tw = now_ms();
                    ck(cudaStreamSynchronize(gp.s_cp), "wave");
                    if (prof) g_cprof.wait_copied += now_ms() - tw;
                    start_wave();
                }
                if (!ready[k].load(std::memory_order_acquire)) {
                    const double tw = now_ms();
                    ready[k].wait(0, std::memory_order_acquire);
                    t_file += now_ms() - tw;
                }
                src = slot_ptr(k);
            }
            uint8_t* d = (uint8_t*) gp.c_stage[h] + (i - b0) * (size_t) SL;
            ck(cudaMemcpyAsync(d, src, (size_t) BL, cudaMemcpyHostToDevice, gp.s_cp), "stream dma");
            es.push_back(stream_e[i]);
            ps.push_back((unsigned long long) d);
        }
        ck(cudaEventRecord(gp.c_copied[h], gp.s_cp), "copied");
        ck(cudaStreamWaitEvent(gp.s, gp.c_copied[h], 0), "wait copied");
        // one launch per adjacent run inside the batch (the arena / file boundary can split one)
        std::vector<int32_t> re;
        std::vector<unsigned long long> rp;
        for (size_t q = 0; q < es.size(); ++q) {
            if (!re.empty() && first[(size_t) re.back()] + cnt[(size_t) re.back()] != first[(size_t) es[q]]) {
                launch(re, rp);
                re.clear();
                rp.clear();
            }
            re.push_back(es[q]);
            rp.push_back(ps[q]);
        }
        launch(re, rp);
        ck(cudaEventRecord(gp.c_used[h], gp.s), "used");
    }
    for (auto& t : readers) t.join();
    const int64_t n_file = (int64_t) nF;
    if (prof && nF && pf_slab < 0) {
        g_cprof.file_rd += now_ms() - t_rd0;
        g_cprof.rd_busy += t_rd_last.load() - t_rd0;
        g_cprof.file_gb += (double) nF * (double) BL / 1e9;
    }

    // ---- chunk_prestage: the next routed layer's arena experts stream in while this layer computes and the caller
    // runs the next dense half (the half this call did not use last; its kernels finish before the DMA starts) ----
    if (im.cfg.chunk_prestage && !im.cfg.arena_adapt && gp.arena.base && !im.g.per_expert()) {
        int64_t nl = layer;
        for (int64_t k = 0; k < im.g.n_layers; ++k) {
            nl = (nl + 1) % im.g.n_layers;
            if (im.g.routed(nl)) break;
        }
        chunk_prestage(im, nl, h_last >= 0 ? 1 - h_last : 0);
    }

    if (!last_pass) {   // the weighted sum waits for the last class pass
        st.hits = ent_hit;
        st.pcie = ent_str;
        st.file_tier = n_file;
        st.file_ms = t_file;
        st.wall_ms = now_ms() - t0;
        im.st.add(st);
        return true;
    }
    if (gp.rchunk && gp.rchunk->pending && !remote_chunk_collect(im, layer)) return false;
    // ---- outputs: the top-k mix on the card, n rows back ----
    ck(cudaMemcpyAsync(gp.c_w, w, sizeof(float) * (size_t) NE, cudaMemcpyHostToDevice, gp.s), "c_w");
    strata::kernels::weighted_rows_sum((const float*) gp.c_parts, (const float*) gp.c_w, (int) K, H, n,
                                       (float*) gp.c_out, gp.s);
    if (out_dev)   // straight into the caller's device tensor (no host round trip)
        ck(cudaMemcpyAsync(out_dev, gp.c_out, (size_t) n * (size_t) H * 4, cudaMemcpyDeviceToDevice, gp.s), "out d2d");
    else
        ck(cudaMemcpyAsync(gp.c_hparts, gp.c_out, (size_t) n * (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "out d2h");
    const double ts = now_ms();
    ck(cudaStreamSynchronize(gp.s), "chunk sync");
    if (!out_dev) std::memcpy(out, gp.c_hparts, (size_t) n * (size_t) H * 4);
    if (im.cfg.saliency) {   // REAP calibration: every entry's unweighted output row, back to the host
        im.sal_parts.resize((size_t) NE * (size_t) H);
        ck(cudaMemcpy(im.sal_parts.data(), gp.c_parts, sizeof(float) * im.sal_parts.size(), cudaMemcpyDeviceToHost), "sal d2h");
        const int64_t NX = im.g.n_experts;
        if (im.sal_sum.empty()) {
            im.sal_sum.assign((size_t) (im.g.n_layers * NX), 0.0);
            im.sal_cnt.assign((size_t) (im.g.n_layers * NX), 0);
        }
        std::vector<double> nrm((size_t) NE);
        std::vector<std::thread> th;
        for (int q = 0; q < 8; ++q)
            th.emplace_back([&, q]() {
                for (int64_t j = q; j < NE; j += 8) {
                    const float* r = im.sal_parts.data() + (size_t) j * (size_t) H;
                    double a2 = 0;
                    for (int64_t d = 0; d < H; ++d) a2 += (double) r[d] * r[d];
                    nrm[(size_t) j] = std::sqrt(a2);
                }
            });
        for (auto& t : th) t.join();
        for (int64_t j = 0; j < NE; ++j) {
            const size_t at = (size_t) (layer * NX + ids[j]);
            im.sal_sum[at] += (double) w[j] * nrm[(size_t) j];
            ++im.sal_cnt[at];
        }
    }
    if (prof) {
        g_cprof.sync += now_ms() - ts;
        for (size_t i = 0; i < kev.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, kev[i].first, kev[i].second);
            g_cprof.kern += ms;
            if (kev_res[i]) g_cprof.resident_kern += ms;
            cudaEventDestroy(kev[i].first);
            cudaEventDestroy(kev[i].second);
        }
        for (auto& e : tcev) {
            float ms = 0;
            cudaEventElapsedTime(&ms, e.first, e.second);
            g_cprof.tc += ms;
            cudaEventDestroy(e.first);
            cudaEventDestroy(e.second);
        }
        g_cprof.tc_launches += (int64_t) tcev.size();
        g_cprof.launches += (int64_t) kev.size();
        g_cprof.calls += 1;
        g_cprof.file += t_file;
        g_cprof.wall += now_ms() - t0;
    }
    st.hits = ent_hit;
    st.pcie = ent_str;
    st.file_tier = n_file;
    st.file_ms = t_file;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    return true;
}
#endif

#if defined(DS4_MOE_CUDA)
// a chunk over every format class present (one pass each; the sum after the last)
// the next routed layer after `layer` (wrapping: the next chunk's layer 0) gets the layer prestage
void chunk_pstage_next(Ds41MoeImpl& im, int64_t layer, int n) {
    // STRATA_PRESTAGE_MIN (rows, default 4096): a narrower call skips the next layer's prestage.  The prestage DMAs the
    // next layer's WHOLE arena share (~1.5 GiB) before its routing is known; a small serve turn routes ~60% of it and
    // its dense half queued behind those copies (s14 turnbench, +189 tokens: 9.83 -> 8.39 s with no prestage).  The
    // region is invalidated, never left holding an older layer the arena may have swapped since.
    int64_t nl = layer;
    for (int64_t k = 0; k < im.g.n_layers; ++k) {
        nl = (nl + 1) % im.g.n_layers;
        if (im.g.routed(nl)) break;
    }
    if (n < chunk_pstage_min()) {
        if (im.gpu) {
            im.gpu->p_layer = -1;
            // what chunk_pstage_issue would have filled (it returns early on these conditions, so then nothing)
            const Gpu& g = *im.gpu;
            im.gpu->p_lazy_layer = (!g.p_off && g.s_cp && g.arena.base && im.g.routed(nl)) ? (int) nl : -1;
        }
        return;
    }
    chunk_pstage_issue(im, nl);
}

bool chunk_classes(Ds41MoeImpl& im, int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out,
                   void* out_dev) {
    // chunk_prefetch: make this layer's file-tier experts available in a slab (and start the next layer's read), so
    // the passes below DMA out of pinned memory instead of reading NVMe on their critical path (a no-op when off).
    chunk_pf_prepare(im, layer, n);
    if (!im.g.per_expert()) {
        if (!gpu_run_chunk(im, layer, n, ids, w, x, out, out_dev)) return false;
        chunk_pstage_next(im, layer, n);
        return true;
    }
    const int64_t NE = (int64_t) n * im.g.top_k;
    const bool rc = im.gpu->rchunk && remote_chunk_post(im, layer, n, ids, x);   // STRATA_DS41_REMOTE
    std::vector<int> cl;
    for (int64_t j = 0; j < NE; ++j) {
        if (rc && im.gpu->rchunk->mask_e[(size_t) ids[j]]) continue;   // the helper's
        const int c = im.g.cls(layer, ids[j]);
        if (std::find(cl.begin(), cl.end(), c) == cl.end()) cl.push_back(c);
    }
    std::sort(cl.begin(), cl.end());
    for (size_t i = 0; i < cl.size(); ++i)
        if (!gpu_run_chunk(im, layer, n, ids, w, x, out, out_dev, cl[i], i == 0, i + 1 == cl.size())) return false;
    chunk_pstage_next(im, layer, n);
    return true;
}
#endif

bool Ds41MoeTier::run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids6 || !w6 || !x || !out) return false;
    RemoteScope rs(*im_, layer, 1, ids6, w6, x, nullptr, out, nullptr);
    rebal_count(*im_, layer, ids6, im_->g.top_k);
    if (!im_->g.routed(layer)) return false;
    if (im_->cfg.cpu_only) return cpu_run(*im_, layer, ids6, w6, x, out);
#if defined(DS4_MOE_CUDA)
    return gpu_run(*im_, layer, ids6, w6, x, nullptr, out);
#else
    return false;
#endif
}

bool Ds41MoeTier::run_to_dev(int64_t layer, const int32_t* ids6, const float* w6, const float* x, void* out_dev) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids6 || !w6 || !x || !out_dev) return false;
    RemoteScope rs(*im_, layer, 1, ids6, w6, x, nullptr, nullptr, out_dev);
    rebal_count(*im_, layer, ids6, im_->g.top_k);
    if (!im_->g.routed(layer) || im_->cfg.cpu_only) return false;
#if defined(DS4_MOE_CUDA)
    return gpu_run(*im_, layer, ids6, w6, x, nullptr, nullptr, out_dev);
#else
    return false;
#endif
}

bool Ds41MoeTier::run_multi(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids || !w || !x || !out || nt < 1 || nt > kMaxTok) return false;
    RemoteScope rs(*im_, layer, nt, ids, w, x, nullptr, out, nullptr);
    rebal_count(*im_, layer, ids, (int64_t) nt * im_->g.top_k);
    if (!im_->g.routed(layer)) return false;
    if (im_->cfg.cpu_only) {   // the CPU tier: token by token (no sharing; it is not the verify path)
        const int64_t K = im_->g.top_k, H = im_->g.n_embd;
        for (int t = 0; t < nt; ++t)
            if (!cpu_run(*im_, layer, ids + t * K, w + t * K, x + (size_t) t * H, out + (size_t) t * H)) return false;
        return true;
    }
#if defined(DS4_MOE_CUDA)
    if (im_->g.per_expert()) {   // D3: the batched per-class path; per-token only if it declines (a q2 class)
        if (gpu_run_n_ex(*im_, layer, nt, ids, w, x, out, nullptr)) return true;
        const int64_t K = im_->g.top_k, H = im_->g.n_embd;
        for (int t = 0; t < nt; ++t)
            if (!gpu_run(*im_, layer, ids + t * K, w + t * K, x + (size_t) t * H, nullptr, out + (size_t) t * H)) return false;
        return true;
    }
    return gpu_run_n(*im_, layer, nt, ids, w, x, out);
#else
    return false;
#endif
}

void Ds41MoeTier::profiler(bool on) {
#if defined(DS4_MOE_CUDA) && defined(DS41_HAVE_PROFILER_API)
    if (on) cudaProfilerStart(); else cudaProfilerStop();
#else
    (void) on;
#endif
}

bool Ds41MoeTier::run_multi_to_dev(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids || !w || !x || !out_dev || nt < 1 || nt > kMaxTok) return false;
    RemoteScope rs(*im_, layer, nt, ids, w, x, nullptr, nullptr, out_dev);
    rebal_count(*im_, layer, ids, (int64_t) nt * im_->g.top_k);
    if (!im_->g.routed(layer)) return false;
    if (std::getenv("DS41_DBG_X")) std::fprintf(stderr, "[dbg_x] ENTER run_multi_to_dev layer=%lld nt=%d\n", (long long) layer, nt);
#if defined(DS4_MOE_CUDA)
    if (!im_->cfg.cpu_only) {
        const int64_t K = im_->g.top_k, H = im_->g.n_embd;
        if (im_->g.per_expert()) {
            if (gpu_run_n_ex(*im_, layer, nt, ids, w, x, nullptr, out_dev)) return true;
            for (int t = 0; t < nt; ++t)   // per-expert fallback (a q2 class): per-token run_to_dev, NOT gpu_run_n
                if (!run_to_dev(layer, ids + t * K, w + t * K, x + (size_t) t * H,
                                (uint8_t*) out_dev + (size_t) t * (size_t) H * 4)) return false;
            return true;
        }
        std::vector<float> tmp((size_t) nt * H);
        if (!gpu_run_n(*im_, layer, nt, ids, w, x, tmp.data())) return false;
        ck(cudaMemcpyAsync(out_dev, tmp.data(), (size_t) nt * (size_t) H * 4, cudaMemcpyHostToDevice, im_->gpu->s), "multi h2d");
        ck(cudaStreamSynchronize(im_->gpu->s), "multi sync");
        return true;
    }
#else
    (void) layer; (void) nt; (void) ids; (void) w; (void) x; (void) out_dev;
#endif
    return false;
}

bool Ds41MoeTier::build_arena_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err) {
    if (im_->cfg.cpu_only) return true;
    std::string e;
    const std::vector<std::pair<int32_t, int32_t>> ranked =
        rank_routes(routes_bin, im_->g.n_routed(), im_->g.top_k, e, batch_tokens);
    if (ranked.empty()) {
        err = e.empty() ? "empty routing profile" : e;
        return false;
    }
    return build_arena(arena_order_remote(*im_, arena_order(*im_, drop_excluded(*im_, ranked))), err);
}

const std::vector<double>& Ds41MoeTier::saliency_sum() const { return im_->sal_sum; }
const std::vector<int64_t>& Ds41MoeTier::saliency_count() const { return im_->sal_cnt; }

void Ds41MoeTier::release_cache() {
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || !im_->gpu->cache) return;
    cudaDeviceSynchronize();   // no copy or kernel may still read a slot
    // warm_reseed (STRATA_WARM_RESEED=1 = on; OFF by default until it shows an effect): before the slots go, record
    // what held them and when it was last used.  The seed after this release puts those experts back first, so a
    // decode that follows a prompt turn starts from the set this conversation routes to instead of the global rank
    // (the batch path needs the VRAM, so the slots themselves cannot stay).  Measured 512K, 4 turns: inside the
    // bench's 1.5-4.3% run-to-run spread, and the seed reads MORE file-tier blobs (144 -> 192) - the LRU set at
    // release is largely the experts that just missed.
    im_->warm.clear();
    if (std::getenv("STRATA_WARM_RESEED") ? std::atoi(std::getenv("STRATA_WARM_RESEED")) != 0 : false) {
        const int64_t NX = im_->g.n_experts;
        std::vector<std::pair<int64_t, int64_t>> ord;   // (last use, layer * NX + expert)
        ord.reserve(2048);
        for (int64_t l = 0; l < im_->g.n_layers; ++l) {
            for (int64_t e = 0; e < NX; ++e) {
                if (im_->gpu->cache->slot_of(l, e) < 0) continue;
                const int64_t u = im_->vlast.size() ? im_->vlast[(size_t) (l * NX + e)] : -1;
                ord.emplace_back(u, l * NX + e);
            }
        }
        std::sort(ord.begin(), ord.end(), [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
            return a.first > b.first;   // most recently used first (vlast = the tick of its last use)
        });
        im_->warm.reserve(ord.size());
        for (const auto& p : ord) im_->warm.emplace_back((int32_t) (p.second / NX), (int32_t) (p.second % NX));
    }
    im_->gpu->cache.reset();
    im_->admitted = 0;
    im_->pf_n = 0;             // staged prefetches are keyed by expert, not slot - but they belong to the old decode
    im_->vlast.clear();        // the LRU order restarts from the next seed (lru_seed_order)
#endif
}

bool Ds41MoeTier::keep_cache_mode() {
    static const bool on = std::getenv("STRATA_SERVE_KEEP_CACHE") && std::getenv("STRATA_SERVE_KEEP_CACHE")[0] == '1';
    return on;
}

int64_t Ds41MoeTier::shrink_cache_for(int64_t need_bytes, std::string& err) {
#if defined(DS4_MOE_CUDA)
    // every refusal says why (s6: the first A/B never kept the cache and logged nothing)
    if (!im_->gpu || !im_->gpu->cache) { err = "no VRAM cache"; return -1; }
    if (!im_->gpu->cache->segmented()) { err = "the cache is not segmented"; return -1; }
    // The layer-major path does not perform LRU admission. Keep its live slot
    // mapping, remove the tail, and restore those exact (slot, layer, expert)
    // records after the batch. Opt-in until real-model byte/quality gates pass.
    static const bool keep_lru = [] {
        const char * value = std::getenv("STRATA_SERVE_KEEP_LRU");
        return value && value[0] == '1';
    }();
    if (im_->cfg.vram_lru && (!keep_lru || im_->mixed_sizes)) {
        err = "--vram-lru (uniform cache needs STRATA_SERVE_KEEP_LRU=1)";
        return -1;
    }
    if (!im_->shrunk.empty()) { err = "still shrunk from the last batch (regrow_cache did not run)"; return -1; }
    if (im_->cfg.cuda_device >= 0 && cudaSetDevice(im_->cfg.cuda_device) != cudaSuccess) {
        err = "shrink_cache_for: cudaSetDevice failed"; return -1;
    }
    strata::core::ExpertCache& c = *im_->gpu->cache;
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "shrink_cache_for: device failed"; return -1; }
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { err = "shrink_cache_for: cudaMemGetInfo failed"; return -1; }
    const int64_t deficit = need_bytes - (int64_t) fr;
    if (deficit <= 0) return 0;
    const int64_t keep_bytes = c.bytes() - deficit;
    if (keep_bytes <= 0) {   // the whole cache would go: release_cache is the same thing, done properly
        err = "the batch needs the whole cache (deficit " + std::to_string(deficit >> 20) + " MiB, cache " +
              std::to_string(c.bytes() >> 20) + " MiB)";
        return -1;
    }
    // shrink rounds its byte argument UP. Choose a segment boundary DOWN so
    // the batch really receives need_bytes; otherwise pb_cap splits a small
    // continuation into two full expert sweeps.
    const int64_t keep_boundary = keep_bytes / c.segment_bytes() * c.segment_bytes();
    const int64_t keep = c.slots_within(keep_boundary);
    if (keep <= 0) { err = "the batch needs every cache segment"; return -1; }
    const int64_t NX = im_->g.n_experts;
    std::vector<std::array<int32_t, 3>> gone;
    for (int64_t l = 0; l < im_->g.n_layers; ++l)
        for (int64_t e = 0; e < NX; ++e) {
            const int32_t sl = c.slot_of(l, e);
            if (sl >= keep) gone.push_back({sl, (int32_t) l, (int32_t) e});
        }
    std::sort(gone.begin(), gone.end());
    const int64_t n = c.evict_from(keep);
    if (!c.shrink(keep_boundary, err)) return -1;
    im_->admitted = im_->admitted > n ? im_->admitted - n : 0;
    im_->pf_n = 0;   // staged prefetches may point at evicted experts' slots
    im_->shrunk = std::move(gone);
    return n;
#else
    (void) need_bytes; (void) err;
    return -1;
#endif
}

int64_t Ds41MoeTier::regrow_cache(std::string& err) {
#if defined(DS4_MOE_CUDA)
    if (im_->shrunk.empty() || !im_->gpu || !im_->gpu->cache) { im_->shrunk.clear(); return 0; }
    if (im_->cfg.cuda_device >= 0 && cudaSetDevice(im_->cfg.cuda_device) != cudaSuccess) {
        err = "regrow_cache: cudaSetDevice failed"; im_->shrunk.clear(); return -1;
    }
    strata::core::ExpertCache& c = *im_->gpu->cache;
    const double t0 = now_ms();
    std::string ge;
    // After a layer-major batch the driver often cannot map another VMM segment (fragmentation).
    // Prefer keeping the hot shrunk cache over wiping both GPUs and PCIe-reseeding.
    if (!c.grow(c.full_bytes(), ge)) {
        (void) cudaDeviceSynchronize();
        if (!c.grow(c.full_bytes(), ge)) {
            size_t fr = 0, tot = 0;
            (void) cudaMemGetInfo(&fr, &tot);
            // Keeping a maxed-out shrunk cache leaves no room for the next pass graph (seen: 38 MiB OOM
            // on CUDA1 with ~275 MiB "free" but fragmented). Only soft-keep when there is real headroom.
            if (fr >= (size_t) (1.5 * 1073741824.0)) {
                std::fprintf(stderr, "serve: keep-cache regrow soft-fail (%s) - keeping %lld resident (%.2f GiB free)\n",
                             ge.c_str(), (long long) im_->admitted, fr / 1073741824.0);
                im_->shrunk.clear();
                err.clear();
                return 0;
            }
            std::fprintf(stderr, "serve: keep-cache regrow fail (%s, %.2f GiB free) - re-seed needed\n",
                         ge.c_str(), fr / 1073741824.0);
            im_->shrunk.clear();
            err = "regrow_cache: grow: " + ge;
            return -1;
        }
    }
    std::vector<std::array<int64_t, 3>> fjobs;   // slot, layer, expert (not in the arena: read from the file)
    int64_t added = 0;
    for (const auto& j : im_->shrunk) {
        if (j[0] >= c.slots()) break;   // a short grow (VRAM taken meanwhile): the rest stay out
        if (c.admit_at(j[1], j[2], j[0]) < 0) continue;   // re-admitted elsewhere meanwhile - leave the slot empty
        const int64_t BL = im_->bl[(size_t) im_->g.cls(j[1], j[2])];
        if (const uint8_t* p = im_->gpu->arena.ptr(j[1], j[2])) {
            if (!c.fill_slot_queued(j[0], p, err, BL)) { im_->shrunk.clear(); return -1; }
        } else {
            fjobs.push_back({j[0], j[1], j[2]});
        }
        ++added;
    }
    uint8_t* buf = nullptr;
    if (!fjobs.empty() && cudaMallocHost((void**) &buf, (size_t) im_->blob) != cudaSuccess) {
        err = "regrow_cache: pinned buffer"; im_->shrunk.clear(); return -1;
    }
    for (const auto& j : fjobs) {   // few: the evicted tail is mostly arena-resident (arena_skip_resident aside)
        im_->blobs->read(j[1], j[2], buf);
        if (!c.fill_slot_blocking((int32_t) j[0], buf, err, im_->bl[(size_t) im_->g.cls(j[1], j[2])])) {
            cudaFreeHost(buf); im_->shrunk.clear(); return -1;
        }
    }
    if (buf) cudaFreeHost(buf);
    if (cudaStreamSynchronize((cudaStream_t) 0) != cudaSuccess) { err = "regrow_cache: queued fills failed"; im_->shrunk.clear(); return -1; }
    static const bool check = std::getenv("DS41_CHECK_ARENA") != nullptr;
    if (check) {   // every refilled slot against the file, byte for byte
        std::vector<uint8_t> ref((size_t) im_->blob), got((size_t) im_->blob);
        int64_t bad = 0, n = 0;
        for (const auto& j : im_->shrunk) {
            if (c.slot_of(j[1], j[2]) != j[0]) continue;
            const size_t len = (size_t) im_->bl[(size_t) im_->g.cls(j[1], j[2])];
            im_->blobs->read(j[1], j[2], ref.data());
            cudaMemcpy(got.data(), c.device_slot(j[0]), len, cudaMemcpyDeviceToHost);
            ++n; bad += std::memcmp(got.data(), ref.data(), len) != 0;
        }
        std::fprintf(stderr, "[keep-chk] %lld refilled slots checked, %lld bad\n", (long long) n, (long long) bad);
    }
    im_->admitted += added;
    std::fprintf(stderr, "serve: cache regrown: %lld experts back (%zu from the file) in %.0f ms\n", (long long) added,
                 fjobs.size(), now_ms() - t0);
    im_->shrunk.clear();
    return added;
#else
    (void) err;
    return 0;
#endif
}

bool Ds41MoeTier::release_prestage() {
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || !im_->gpu->p_buf) return false;
    Gpu& gp = *im_->gpu;
    cudaDeviceSynchronize();   // no DMA into it and no expert kernel reading from it is still in flight
    cudaFree(gp.p_buf);
    if (gp.p_copied) cudaEventDestroy(gp.p_copied);
    if (gp.p_used) cudaEventDestroy(gp.p_used);
    gp.p_buf = nullptr;
    gp.p_copied = gp.p_used = nullptr;
    gp.p_cap = 0;
    gp.p_layer = -1;
    gp.p_lazy_layer = -1;
    gp.p_used_rec = false;
    gp.p_off = true;           // no prestage for the rest of this prompt (release_chunk re-arms it)
    return true;
#else
    return false;
#endif
}

void Ds41MoeTier::release_chunk() {
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu) return;
    if (chunk_prof() && g_cprof.calls) {
        const ChunkProf& c = g_cprof;
        std::fprintf(stderr, "chunk prof: %lld calls, %lld launches: wall %.0f ms = waits on reads %.0f + wave/copy waits "
                             "%.0f + final sync %.0f + other host %.0f; reader wall %.0f ms; GPU grouped kernels %.0f ms "
                             "(resident %.0f); file tier %.1f GB read in %.0f ms of reader time = %.2f GB/s\n",
                     (long long) c.calls, (long long) c.launches, c.wall, c.file, c.wait_copied, c.sync,
                     c.wall - c.file - c.wait_copied - c.sync, c.file_rd, c.kern, c.resident_kern, c.file_gb, c.rd_busy,
                     c.rd_busy > 0 ? c.file_gb / (c.rd_busy / 1000.0) : 0.0);
        std::fprintf(stderr, "chunk prof MMQ: %lld sub-runs; PTQ1_0 -> Q2_0 transcode %.0f ms GPU (inside the grouped "
                             "kernels) over %lld sub-runs, %.1f GB of ternary read = %.0f GB/s\n",
                     (long long) c.mmq_subruns, c.tc, (long long) c.tc_launches, c.tc_gb, c.tc > 0 ? c.tc_gb / (c.tc / 1000.0) : 0.0);
        g_cprof = ChunkProf{};
        if (im_->remote) remote_prof_print(*im_->remote, "prompt");
    }
    if (chunk_prof() && (g_pf.blobs || g_pf.used)) {
        const ChunkPfProf& p = g_pf;
        std::fprintf(stderr, "chunk prefetch: %lld layers, %lld blobs, %.1f GB read in %.0f ms of reader time = %.2f GB/s; "
                             "%lld of %lld file-tier DMAs served from a slab (%lld passes not covered)\n",
                     (long long) p.layers, (long long) p.blobs, p.gb, p.ms, p.ms > 0 ? p.gb / (p.ms / 1000.0) : 0.0,
                     (long long) p.used, (long long) p.dmas, (long long) p.off);
        if (p.checked)
            std::fprintf(stderr, "chunk prefetch CHECK: %lld staged blobs re-read with read(), %lld differ\n",
                         (long long) p.checked, (long long) p.bad);
        g_pf = ChunkPfProf{};
    }
    Gpu& gp = *im_->gpu;
    if (gp.s) cudaStreamSynchronize(gp.s);
    if (gp.s_cp) cudaStreamSynchronize(gp.s_cp);
    // The DEVICE buffers go back (the deferred VRAM seed wants the memory).  The PINNED landing pads (c_bounce,
    // c_hparts) and the chunk_prefetch slabs stay: cudaMallocHost was 13 calls / 3.4 s per prompt (nsys), and a
    // pinned slab's pages must not be churned.  Only close() (or a manual release) frees them.
    for (void* p : {gp.c_x, gp.c_xq, gp.c_parts, gp.c_scratch, gp.c_ptr, gp.c_start, gp.c_ng, gp.c_dst, gp.c_tokv,
                    gp.c_stage[0], gp.c_stage[1], gp.c_w, gp.c_out})
        if (p) cudaFree(p);
    gp.free_mmq();   // MMQ scratch + the PTQ1_0 transcode copies (GBs): the deferred VRAM seed wants them back
    for (int i = 0; i < 2; ++i) {
        if (gp.c_copied[i]) cudaEventDestroy(gp.c_copied[i]);
        if (gp.c_used[i]) cudaEventDestroy(gp.c_used[i]);
    }
    if (gp.s_cp) cudaStreamDestroy(gp.s_cp);
    gp.c_x = gp.c_xq = gp.c_parts = gp.c_scratch = gp.c_ptr = gp.c_start = gp.c_ng = gp.c_dst = gp.c_tokv = nullptr;
    gp.c_w = gp.c_out = nullptr;
    gp.c_stage[0] = gp.c_stage[1] = nullptr;
    for (int i = 0; i < 2; ++i) gp.c_copied[i] = gp.c_used[i] = nullptr;
    gp.s_cp = nullptr;
    gp.c_tok = gp.c_half = 0;
    gp.c_ready = false;
    gp.c_pre_layer = gp.c_pre_half = -1;
    gp.c_pre_e.clear();
    if (gp.p_buf) cudaFree(gp.p_buf);
    if (gp.p_copied) cudaEventDestroy(gp.p_copied);
    if (gp.p_used) cudaEventDestroy(gp.p_used);
    gp.p_buf = nullptr;
    gp.p_copied = gp.p_used = nullptr;
    gp.p_cap = 0;
    gp.p_layer = -1;
    gp.p_lazy_layer = -1;
    gp.p_off = gp.p_used_rec = false;
    gp.free_mmq();
#endif
}

bool Ds41MoeTier::run_chunk(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    if (!im_->inited || !ids || !w || !x || !out || n < 1) return false;
    if (!im_->g.routed(layer)) return false;
    RemoteScope rs(*im_, layer, n, ids, w, x, nullptr, out, nullptr);
#if defined(DS4_MOE_CUDA)
    if (im_->gpu && !im_->cfg.cpu_only) return chunk_classes(*im_, layer, n, ids, w, x, out, nullptr);
#endif
    // CPU tier: token by token
    const int64_t K = im_->g.top_k, H = im_->g.n_embd;
    for (int t = 0; t < n; ++t)
        if (!run(layer, ids + t * K, w + t * K, x + (size_t) t * H, out + (size_t) t * H)) return false;
    return true;
}

bool Ds41MoeTier::run_chunk_dev(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    if (!im_->inited || !ids || !w || !x || !out_dev || n < 1) return false;
    if (!im_->g.routed(layer)) return false;
    RemoteScope rs(*im_, layer, n, ids, w, x, nullptr, nullptr, out_dev);
#if defined(DS4_MOE_CUDA)
    if (im_->gpu && !im_->cfg.cpu_only) return chunk_classes(*im_, layer, n, ids, w, x, nullptr, out_dev);
#endif
    return false;
}

bool Ds41MoeTier::run_dev(int64_t layer, const int32_t* ids6, const float* w6, const void* x_dev, float* out) {
    if (!im_->inited) return false;
    if (!ids6 || !w6 || !x_dev || !out) return false;
    RemoteScope rs(*im_, layer, 1, ids6, w6, nullptr, x_dev, out, nullptr);
    rebal_count(*im_, layer, ids6, im_->g.top_k);
    if (!im_->g.routed(layer)) return false;
    if (im_->cfg.cpu_only) return false;   // there is no device input without a device tier
#if defined(DS4_MOE_CUDA)
    return gpu_run(*im_, layer, ids6, w6, nullptr, x_dev, out);
#else
    return false;
#endif
}

bool ds41_moe_cpu_expert(const cpu::NativeFmt& f, int64_t n_embd, int64_t n_ff, const uint8_t* blob,
                        const float* x, float* out, cpu::ExpertPool& pool) {
    (void) n_ff;
    std::vector<uint8_t> nact(cpu::kNativeActBytes);
    std::vector<float> parts((size_t) n_embd);
    cpu::ExpertJobMulti job;
    job.blob = blob;
    job.nt = 1;
    for (int t = 0; t < cpu::MAXT; ++t) {
        job.nact[t] = nullptr;
        job.out[t] = nullptr;
    }
    std::unique_ptr<cpu::ActQ> aq(new cpu::ActQ());
    if (cpu::q2_native_kernels(f.gu_type)) cpu::act_quant_any(x, (int) n_embd, *aq);
    else cpu::native_quant_act(f, x, nact.data());
    job.nact[0] = nact.data();
    job.act[0] = aq.get();
    job.out[0] = parts.data();
    pool.run_split_multi_native(f, &job, 1);
    std::memcpy(out, parts.data(), (size_t) n_embd * sizeof(float));
    return true;
}

// ================================ the conversation-adaptive arena (STRATA_ARENA_REBAL) ================================
//
// A serve turn's exact prompt routes most of the model: a +189-token turn on top of 4K routed ~7,000 distinct experts,
// ~2,600 of them only on NVMe (24 GB at 5.2 GB/s - the turn's floor).  Turns of one conversation reuse 97-99% of the
// file-tier experts earlier turns read, but the conversation's union is ~67 GB, so no cache holds it; what DOES help is
// FREQUENCY - replaying the s14 route trace, an arena holding the conversation's most-used experts (decayed routed-token
// counts) cuts a small turn's file-tier experts by ~35-45%.  LRU admission made it worse (the long first turn evicts
// the ranked arena).  So: count, plan after each prompt, and swap in the idle NVMe time between requests (s6: reads
// during decode too cost decode 5-19 %, STRATA_ARENA_REBAL_DECODE=1).
//
// Safety - only the ENGINE thread ever reads or writes arena.slot_of:
//   retire (engine, after cudaDeviceSynchronize: no copy or kernel in flight): the victim's slot_of = -1, so nothing
//     maps to the slot and no later DMA / demotion / arena_adapt swap can touch it;
//   read (the background thread): the candidate's blob from the file straight into the retired slot;
//   publish (engine, once the read is done): the candidate's slot_of = the slot.  If decode's own arena_adapt brought
//     the candidate in meanwhile, the slot goes to `free_slots` and the next swap uses it without retiring anyone.
// Reads pause during prompt turns (they would compete with the prompt's own NVMe reads).  The prompt's routing is
// exact, so its logits do not depend on where a blob is served from; decode's selection bias (route bias / file
// penalty) does depend on residency, as it already does under arena_adapt.
struct Rebal {
    std::vector<float> score;   ///< (layer, expert) routed tokens (prompt + decode), decayed by `decay` per plan
    struct Pair { int32_t layer, cand, victim; float gain; };
    std::vector<Pair> queue;
    size_t qpos = 0;
    struct Job { int32_t layer, e, slot, victim = -1; std::atomic<int> done{0}; };   ///< victim -1 = a free slot
    std::vector<std::unique_ptr<Job>> inflight;   ///< engine thread's list (retired, not yet published)
    std::deque<Job*> todo;                        ///< the reader's queue (under mu)
    std::vector<int32_t> free_slots;              ///< slots nobody maps (see publish)
    std::mutex mu;
    std::condition_variable cv;
    bool pause = false, stop = false;
    std::thread th;
    int64_t passes = 0, retired = 0, published = 0, wasted = 0, cancelled = 0;
    int64_t skip[5] = {};   ///< retire rejections: cand already in, victim gone, victim VRAM-resident, victim recently used, slot too small
    float decay = 0.9f, margin = 1.0f, min_score = 2.0f;
    int per_tick = 64, every = 4, max_plan = 6000, max_inflight = 4096, end_batch = 2048;
    bool decode_reads = false;  ///< reads only between requests (s6: reads during decode cost it 5-19 %); STRATA_ARENA_REBAL_DECODE=1 = also during decode
};

void rebal_count(Ds41MoeImpl& im, int64_t layer, const int32_t* ids, int64_t n) {
    if (!im.rebal || layer < 0 || layer >= im.g.n_layers) return;
    const int64_t NX = im.g.n_experts;
    float* sc = im.rebal->score.data() + (size_t) (layer * NX);
    for (int64_t i = 0; i < n; ++i)
        if (ids[i] >= 0 && ids[i] < NX) sc[ids[i]] += 1.0f;
}

#if defined(DS4_MOE_CUDA)
static void rebal_publish(Ds41MoeImpl& im) {
    Rebal& r = *im.rebal;
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    size_t k = 0;
    for (size_t i = 0; i < r.inflight.size(); ++i) {
        Rebal::Job& j = *r.inflight[i];
        if (!j.done.load(std::memory_order_acquire)) { r.inflight[k++] = std::move(r.inflight[i]); continue; }
        int32_t& so = gp.arena.slot_of[(size_t) (j.layer * NX + j.e)];
        static const bool check = std::getenv("DS41_CHECK_ARENA") != nullptr;
        if (check) {   // the slot's bytes against the file, byte for byte, before anything can use them
            static int64_t n_chk = 0, n_bad = 0;
            const size_t len = (size_t) im.bl[(size_t) im.g.cls(j.layer, j.e)];
            std::vector<uint8_t> ref((size_t) im.blob);
            im.blobs->read(j.layer, j.e, ref.data());
            const bool good = std::memcmp((const uint8_t*) gp.arena.base + gp.arena.off[(size_t) j.slot], ref.data(), len) == 0;
            ++n_chk; n_bad += !good;
            if (!good) std::fprintf(stderr, "[rebal-chk] L%d e%d slot %d BAD\n", j.layer, j.e, j.slot);
            if (n_chk % 500 == 0) std::fprintf(stderr, "[rebal-chk] %lld published swaps checked, %lld bad\n", (long long) n_chk, (long long) n_bad);
        }
        if (so < 0) { so = j.slot; ++r.published; }
        else { r.free_slots.push_back(j.slot); ++r.wasted; }   // arena_adapt brought it in first: keep the slot for later
    }
    r.inflight.resize(k);
}

/// Retire up to `max_n` planned swaps (the caller just synchronized the device) and hand them to the reader.
static void rebal_retire(Ds41MoeImpl& im, int max_n) {
    Rebal& r = *im.rebal;
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    const bool cache = gp.cache && !im.cfg.no_cache;
    int n = 0;
    std::vector<Rebal::Job*> add;
    while (n < max_n && r.qpos < r.queue.size() && (int) r.inflight.size() < r.max_inflight) {
        const Rebal::Pair p = r.queue[r.qpos++];
        int32_t& sc = gp.arena.slot_of[(size_t) (p.layer * NX + p.cand)];
        if (sc >= 0) { ++r.skip[0]; continue; }                           // already in (arena_adapt)
        const uint64_t need = (uint64_t) im.bl[(size_t) im.g.cls(p.layer, p.cand)];
        int32_t slot = -1, victim = -1;
        for (size_t f = 0; f < r.free_slots.size(); ++f)                  // a slot nobody maps first
            if (gp.arena.off[(size_t) r.free_slots[f] + 1] - gp.arena.off[(size_t) r.free_slots[f]] >= need) {
                slot = r.free_slots[f];
                r.free_slots.erase(r.free_slots.begin() + (long) f);
                break;
            }
        if (slot < 0) {
            int32_t& sv = gp.arena.slot_of[(size_t) (p.layer * NX + p.victim)];
            if (sv < 0) { ++r.skip[1]; continue; }                        // the victim left already
            if (cache && gp.cache->slot_of(p.layer, p.victim) >= 0) { ++r.skip[2]; continue; }   // VRAM-resident now
            if (!im.last_use.empty() && im.last_use[(size_t) (p.layer * NX + p.victim)] > im.tick - 4 * im.g.n_layers)
                { ++r.skip[3]; continue; }                                // decode used it in the last few tokens
            if (gp.arena.off[(size_t) sv + 1] - gp.arena.off[(size_t) sv] < need) { ++r.skip[4]; continue; }
            slot = sv;
            victim = p.victim;
            sv = -1;                                                      // retired: nothing maps to `slot` now
        }
        auto j = std::make_unique<Rebal::Job>();
        j->layer = p.layer; j->e = p.cand; j->slot = slot; j->victim = victim;
        add.push_back(j.get());
        r.inflight.push_back(std::move(j));
        ++r.retired;
        ++n;
    }
    if (add.empty()) return;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        for (Rebal::Job* j : add) r.todo.push_back(j);
    }
    r.cv.notify_all();
}

/// A prompt is starting: take back every retired swap the reader has not started, so the prompt finds no holes in the
/// arena (s6: with back-to-back requests, request_end's batch of retired-but-unread slots stayed empty through the whole
/// next prompt - neither the victim nor the candidate in the arena - and a 16.5K prompt read 42 % more blobs from NVMe,
/// decode after it ran 30 % slower).  A queued job's slot still holds its victim's bytes (nothing writes a retired slot
/// but the reader), so the victim simply gets its slot back; a free-slot job returns its slot to free_slots.  The one
/// read that may be in progress is waited for, then published.  Called with r.pause already set (no new pops).
static void rebal_cancel(Ds41MoeImpl& im) {
    Rebal& r = *im.rebal;
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    std::deque<Rebal::Job*> undo;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        undo.swap(r.todo);
    }
    if (!undo.empty()) {
        for (Rebal::Job* j : undo) {
            if (j->victim >= 0) {
                int32_t& sv = gp.arena.slot_of[(size_t) (j->layer * NX + j->victim)];
                if (sv < 0) sv = j->slot;                      // the victim is back, bytes untouched
                else r.free_slots.push_back(j->slot);          // arena_adapt re-admitted it elsewhere meanwhile
                static const bool check = std::getenv("DS41_CHECK_ARENA") != nullptr;
                if (check && sv == j->slot) {                  // the restored slot against the victim's file blob
                    static int64_t n_chk = 0, n_bad = 0;
                    const size_t len = (size_t) im.bl[(size_t) im.g.cls(j->layer, j->victim)];
                    std::vector<uint8_t> ref((size_t) im.blob);
                    im.blobs->read(j->layer, j->victim, ref.data());
                    const bool good = std::memcmp((const uint8_t*) gp.arena.base + gp.arena.off[(size_t) j->slot], ref.data(), len) == 0;
                    ++n_chk; n_bad += !good;
                    if (!good) std::fprintf(stderr, "[rebal-chk] restore L%d e%d slot %d BAD\n", j->layer, j->victim, j->slot);
                    if (n_chk % 500 == 0) std::fprintf(stderr, "[rebal-chk] %lld restored victims checked, %lld bad\n", (long long) n_chk, (long long) n_bad);
                }
            } else {
                r.free_slots.push_back(j->slot);
            }
            j->done.store(2, std::memory_order_relaxed);       // 2 = cancelled: dropped from inflight below
        }
        size_t k = 0;
        for (size_t i = 0; i < r.inflight.size(); ++i)
            if (r.inflight[i]->done.load(std::memory_order_relaxed) != 2) r.inflight[k++] = std::move(r.inflight[i]);
        r.inflight.resize(k);
        r.retired -= (int64_t) undo.size();
        r.cancelled += (int64_t) undo.size();
    }
    for (int spin = 0; spin < 20000; ++spin) {   // the reader holds at most one job: wait for it (one blob read)
        bool busy = false;
        for (const auto& j : r.inflight) busy |= j->done.load(std::memory_order_acquire) == 0;
        if (!busy) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
#endif

void Ds41MoeTier::rebalance_begin_prompt() {
#if defined(DS4_MOE_CUDA)
    if (!im_ || !im_->gpu || !im_->gpu->arena.base || im_->cfg.cpu_only) return;
    if (!im_->rebal) {   // on by default (s5: -10% small-turn prefill, e2e 21/21 10/10 3/3); STRATA_ARENA_REBAL=0 turns it off
        static const bool on = !(std::getenv("STRATA_ARENA_REBAL") && std::getenv("STRATA_ARENA_REBAL")[0] == '0');
        if (!on) return;
        im_->rebal = std::make_unique<Rebal>();
        Rebal& r0 = *im_->rebal;
        r0.score.assign((size_t) (im_->g.n_layers * im_->g.n_experts), 0.0f);
        if (const char* d = std::getenv("STRATA_ARENA_REBAL_DECODE")) r0.decode_reads = d[0] == '1';
        Ds41MoeImpl* imp = im_.get();
        r0.th = std::thread([imp] {   // the reader: retired slots <- the candidate's blob, outside stream order
            Rebal& r = *imp->rebal;
            Gpu& gp = *imp->gpu;
            for (;;) {
                Rebal::Job* j = nullptr;
                {
                    std::unique_lock<std::mutex> lk(r.mu);
                    r.cv.wait(lk, [&] { return r.stop || (!r.pause && !r.todo.empty()); });
                    if (r.stop) return;
                    j = r.todo.front();
                    r.todo.pop_front();
                }
                imp->blobs->read(j->layer, j->e, (uint8_t*) gp.arena.base + gp.arena.off[(size_t) j->slot]);
                j->done.store(1, std::memory_order_release);
            }
        });
        std::fprintf(stderr, "arena rebalance: on (conversation-adaptive arena, reads in the idle NVMe time)\n");
    }
    Rebal& r = *im_->rebal;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        r.pause = true;
    }
    rebal_cancel(*im_);
    rebal_publish(*im_);
#endif
}

void Ds41MoeTier::rebalance_plan() {
#if defined(DS4_MOE_CUDA)
    if (!im_ || !im_->rebal || !im_->gpu || !im_->gpu->arena.base) return;
    Ds41MoeImpl& im = *im_;
    Rebal& r = *im.rebal;
    Gpu& gp = *im.gpu;
    const int64_t NX = im.g.n_experts;
    const bool cache = gp.cache && !im.cfg.no_cache;
    const bool skip_held = remote_skips_arena(im);
    r.queue.clear();
    r.qpos = 0;
    std::vector<int32_t> cand, vict;
    for (int64_t l = 0; l < im.g.n_layers; ++l) {
        if (!im.g.routed(l)) continue;
        const float* sc = r.score.data() + (size_t) (l * NX);
        cand.clear();
        vict.clear();
        for (int64_t e = 0; e < NX; ++e) {
            if (!im.cfg.exclude.empty() && im.cfg.exclude[(size_t) (l * NX + e)]) continue;
            if (skip_held && gp.remote->s->holds[(size_t) (l * NX + e)]) continue;   // the helper computes it
            if (gp.arena.slot_of[(size_t) (l * NX + e)] < 0) { if (sc[e] >= r.min_score) cand.push_back((int32_t) e); }
            else if (!cache || gp.cache->slot_of(l, e) < 0) vict.push_back((int32_t) e);
        }
        std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return sc[a] > sc[b]; });
        std::sort(vict.begin(), vict.end(), [&](int32_t a, int32_t b) { return sc[a] < sc[b]; });
        // each candidate takes the least-used victim whose SLOT holds its blob: the arena packs every slot at its
        // seeded expert's own blob size and DS4's 9 format classes differ, so index-order pairing was rejected at
        // retire time for nearly every pair ("slot small", s14: 3,178 of 3,178 rejections)
        std::vector<char> taken(vict.size(), 0);
        size_t lo = 0;   // victims before lo are all taken
        for (int32_t c : cand) {
            const uint64_t need = (uint64_t) im.bl[(size_t) im.g.cls(l, c)];
            while (lo < vict.size() && taken[lo]) ++lo;
            if (lo >= vict.size() || sc[c] - sc[vict[lo]] <= r.margin) break;   // no victim left worth evicting
            for (size_t j = lo; j < vict.size(); ++j) {
                if (taken[j]) continue;
                if (sc[c] - sc[vict[j]] <= r.margin) break;   // victims are sorted: the rest gain even less
                const int32_t sv = gp.arena.slot_of[(size_t) (l * NX + vict[j])];
                if (gp.arena.off[(size_t) sv + 1] - gp.arena.off[(size_t) sv] < need) continue;
                taken[j] = 1;
                r.queue.push_back({ (int32_t) l, c, vict[j], sc[c] - sc[vict[j]] });
                break;
            }
        }
    }
    std::sort(r.queue.begin(), r.queue.end(), [](const Rebal::Pair& a, const Rebal::Pair& b) { return a.gain > b.gain; });
    if ((int) r.queue.size() > r.max_plan) r.queue.resize((size_t) r.max_plan);
    for (float& v : r.score) v *= r.decay;
    std::fprintf(stderr, "arena rebalance: %zu swaps planned (so far %lld retired, %lld published, %lld slots re-used, "
                 "%lld cancelled; skipped: cand in %lld, victim gone %lld, victim in VRAM %lld, victim recent %lld, slot small %lld)\n",
                 r.queue.size(), (long long) r.retired, (long long) r.published, (long long) r.wasted, (long long) r.cancelled,
                 (long long) r.skip[0],
                 (long long) r.skip[1], (long long) r.skip[2], (long long) r.skip[3], (long long) r.skip[4]);
    {
        std::lock_guard<std::mutex> lk(r.mu);
        r.pause = false;
    }
    r.cv.notify_all();
#endif
}

void Ds41MoeTier::rebalance_tick(bool request_end) {
#if defined(DS4_MOE_CUDA)
    if (!im_ || !im_->rebal || !im_->gpu) return;
    Rebal& r = *im_->rebal;
    rebal_publish(*im_);
    if (r.qpos >= r.queue.size()) return;
    if (!request_end && (!r.decode_reads || (++r.passes % r.every) != 0)) return;
    if ((int) r.inflight.size() >= r.max_inflight) return;
    cudaDeviceSynchronize();   // no copy or kernel may still touch a slot we retire
    rebal_retire(*im_, request_end ? r.end_batch : r.per_tick);
#else
    (void) request_end;
#endif
}

void Ds41MoeTier::close() {
    if (im_ && im_->rebal) {   // the reader first: it writes arena memory
        Rebal& r = *im_->rebal;
        {
            std::lock_guard<std::mutex> lk(r.mu);
            r.stop = true;
        }
        r.cv.notify_all();
        if (r.th.joinable()) r.th.join();
        im_->rebal.reset();
    }
    if (im_ && im_->trace) { std::fclose(im_->trace); im_->trace = nullptr; }
#if defined(DS4_MOE_CUDA)
    if (g_prof_n > 0) {
        double tot = 0;
        for (double v : g_prof) tot += v;
        std::fprintf(stderr, "tier profile (%lld layer calls, %.3f ms/call):", (long long) g_prof_n, tot / g_prof_n);
        for (int i = 0; i < 9; ++i) std::fprintf(stderr, " %s %.3f", g_prof_name[i], g_prof[i] / g_prof_n);
        std::fprintf(stderr, "\n");
    }
#endif
    if (!im_) return;
#if defined(DS4_MOE_CUDA)
    chunk_stage_close(*im_);   // join the prefetch readers, free the pinned slabs
#endif
    if (im_->remote) {
#if defined(DS4_MOE_CUDA)
        const RemoteTier& r = *im_->remote;
        if (r.calls > 0)
            std::fprintf(stderr, "remote tier: %lld calls, %lld entries, %.3f ms/call waiting for the card\n",
                         (long long) r.calls, (long long) r.entries, r.wait_ms / (double) r.calls);
#endif
        im_->remote->close();
        im_->remote.reset();
    }
    if (im_->gpu) im_->gpu->close();
    im_->gpu.reset();
    im_->pool.reset();
    im_->owned_blobs.reset();
    im_->blobs = nullptr;
    im_->inited = false;
    im_->pf_n = 0;
}

const Ds41MoeStats& Ds41MoeTier::stats() const { return im_->st; }
void Ds41MoeTier::reset_stats() { im_->st = Ds41MoeStats(); }
void Ds41MoeTier::add_stats(const Ds41MoeStats& s) { im_->st.add(s); }
const Ds41MoeGeom& Ds41MoeTier::geom() const { return im_->g; }
const cpu::NativeFmt& Ds41MoeTier::fmt() const { return im_->f; }
const Ds41MoeConfig& Ds41MoeTier::config() const { return im_->cfg; }
cpu::ExpertPool& Ds41MoeTier::pool() { return *im_->pool; }

const char* Ds41MoeTier::mode() const {
    if (!im_->inited) return "uninitialised";
    if (im_->cfg.cpu_only) return "cpu-only";
    return "gpu";
}

int64_t Ds41MoeTier::resident() const { return im_->admitted; }
int64_t Ds41MoeTier::blob_bytes(int64_t layer) const {
    if (layer < 0 || layer >= (int64_t) im_->bl.size()) return 0;
    return im_->bl[(size_t) layer];
}

int64_t Ds41MoeTier::file_tier() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? im_->gpu->arena.file_tier : 0;
#else
    return 0;
#endif
}

int64_t Ds41MoeTier::arena_experts() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? im_->gpu->arena.used : 0;
#else
    return 0;
#endif
}

bool Ds41MoeTier::in_remote(int64_t layer, int64_t expert) const {
    return im_ && im_->remote && im_->remote->held(layer, expert, im_->g.n_experts);
}

int64_t Ds41MoeTier::remote_experts() const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->remote) return im_->remote->n;
#endif
    return 0;
}

double Ds41MoeTier::remote_gib() const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->remote) return (double) im_->remote->used / 1073741824.0;
#endif
    return 0.0;
}

bool Ds41MoeTier::in_arena(int64_t layer, int64_t expert) const {
    if (in_remote(layer, expert)) return true;   // not the file tier: the route-bias penalty and the readers skip it
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->gpu) return im_->gpu->arena.ptr(layer, expert) != nullptr;
#endif
    (void) layer;
    (void) expert;
    return false;
}

bool Ds41MoeTier::remote_holds(int64_t layer, int64_t expert) const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->gpu && im_->gpu->remote && layer >= 0 && layer < im_->g.n_layers && expert >= 0 && expert < im_->g.n_experts)
        return im_->gpu->remote->s->holds[(size_t) (layer * im_->g.n_experts + expert)] != 0;
#endif
    (void) layer;
    (void) expert;
    return false;
}

bool Ds41MoeTier::resident(int64_t layer, int64_t expert) const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->gpu && im_->gpu->cache) return im_->gpu->cache->slot_of(layer, expert) >= 0;
#endif
    (void) layer;
    (void) expert;
    return false;
}

double Ds41MoeTier::arena_gib() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? (double) im_->gpu->arena.bytes / 1073741824.0 : 0.0;
#else
    return 0.0;
#endif
}

void Ds41MoeStats::add(const Ds41MoeStats& o) {
    hits += o.hits;
    prefetched_useful += o.prefetched_useful;
    prefetch_issued += o.prefetch_issued;
    prefetch_dummy += o.prefetch_dummy;
    advised += o.advised;
    cpu += o.cpu;
    pcie += o.pcie;
    file_tier += o.file_tier;
    file_ms += o.file_ms;
    arena_swaps += o.arena_swaps;
    vram_swaps += o.vram_swaps;
    vram_demotes += o.vram_demotes;
    skipped += o.skipped;
    remote += o.remote;
    gap_ms += o.gap_ms;
    hit_ms += o.hit_ms;
    pcie_ms += o.pcie_ms;
    cpu_ms += o.cpu_ms;
    wall_ms += o.wall_ms;
}

// ================================ DeepSeek-V4.1 blob source ================================

Ds41FileBlobs::Ds41FileBlobs(std::string gguf) : path_(std::move(gguf)) {}
Ds41FileBlobs::~Ds41FileBlobs() {
    for (int f : sfd_) if (f >= 0) ds41_sys::close(f);   // (fd_ / dfd_ are shard 0's)
    for (int f : sdfd_) if (f >= 0) ds41_sys::close(f);
    if (mfd_ >= 0) ds41_sys::close(mfd_);
    for (uint8_t* p : ball_) ds41_sys::aligned_free(p);
}

uint8_t* Ds41FileBlobs::take_bounce() const {
    std::lock_guard<std::mutex> lk(bmu_);
    if (!bfree_.empty()) {
        uint8_t* p = bfree_.back();
        bfree_.pop_back();
        return p;
    }
    uint8_t* p = (uint8_t*) ds41_sys::aligned_alloc(4096, (scratch_ + 4095) & ~(size_t) 4095);
    if (p) ball_.push_back(p);
    return p;
}

void Ds41FileBlobs::give_bounce(uint8_t* p) const {
    std::lock_guard<std::mutex> lk(bmu_);
    bfree_.push_back(p);
}

bool Ds41FileBlobs::open(std::string& err) {
    if (fd_ >= 0) return true;
    std::unique_ptr<GgufModel> m;
    try { m = std::make_unique<GgufModel>(GgufModel::open(path_)); }
    catch (const std::exception& e) { err = e.what(); return false; }
    if (m->size() > ((size_t) 1 << (64 - kShardShift))) { err = "Ds41FileBlobs: too many shards"; return false; }
    Ds41Geometry G;
    if (!(err = ds41_read_geometry(*m, G)).empty()) return false;
    const std::vector<int64_t> layers = ds41_file_layers(*m, G);
    if (!(err = ds41_check_tensors(*m, G, layers)).empty()) return false;
    const int64_t NL = (int64_t) layers.size(), NE = G.n_expert, D = G.n_embd, FF = G.n_ff;
    g_ = Ds41MoeGeom();
    g_.n_embd = D;
    g_.n_ff = FF;
    g_.n_layers = NL;
    g_.n_experts = NE;
    g_.top_k = G.n_used;
    g_.swiglu_limit = G.swiglu_limit > 0 ? (float) G.swiglu_limit : std::numeric_limits<float>::infinity();
    g_.eclass.assign((size_t) (NL * NE), 0);
    std::map<std::pair<int, int>, int> cid;
    for (int64_t l = 0; l < NL; ++l)
        for (int64_t e = 0; e < NE; ++e) {
            const std::pair<int, int> k{G.experts[(size_t) l].gu_type[(size_t) e], G.experts[(size_t) l].d_type[(size_t) e]};
            auto it = cid.find(k);
            if (it == cid.end()) {
                it = cid.emplace(k, (int) g_.cls_gu.size()).first;
                g_.cls_gu.push_back(k.first);
                g_.cls_d.push_back(k.second);
            }
            g_.eclass[(size_t) (l * NE + e)] = (int16_t) it->second;
        }
    g_.gu_type = g_.cls_gu[0];
    g_.d_type = g_.cls_d[0];
    // tensor name -> its shard's file + the tag its offsets carry (a split GGUF; one shard = tag 0)
    auto locate = [&](const std::string& n, const TensorInfo*& t) -> uint64_t {
        size_t si = 0;
        t = m->find(n, &si);
        return t ? ((uint64_t) si << kShardShift) + m->shard(si).data_start() : 0;
    };
    auto rb = [](int t, int64_t n) {
        int be = 0, bb = 0;
        block_geometry((uint32_t) t, be, bb);
        return (uint64_t) (n / be) * (uint64_t) bb;
    };
    for (size_t c = 0; c < g_.cls_gu.size(); ++c) {
        gu_rb_.push_back((uint32_t) rb(g_.cls_gu[c], D));
        d_rb_.push_back((uint32_t) rb(g_.cls_d[c], FF));
    }
    gate_off_.assign((size_t) (NL * NE), 0);
    up_off_.assign((size_t) (NL * NE), 0);
    down_off_.assign((size_t) (NL * NE), 0);
    for (int64_t l = 0; l < NL; ++l) {
        const Ds41ExpertMap& X = G.experts[(size_t) l];
        for (int64_t e = 0; e < NE; ++e) {
            const int gt = X.gu_type[(size_t) e], dt = X.d_type[(size_t) e];
            const TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
            const uint64_t bg = locate(X.gate_t.at(gt), tg), bu = locate(X.up_t.at(gt), tu), bd = locate(X.down_t.at(dt), td);
            if (!tg || !tu || !td) { err = "Ds41FileBlobs: an expert tensor of layer " + std::to_string(l) + " is missing"; return false; }
            const uint64_t gb = rb(gt, D) * (uint64_t) FF, db = rb(dt, FF) * (uint64_t) D;
            const size_t i = (size_t) (l * NE + e);
            gate_off_[i] = bg + tg->offset + (uint64_t) X.gu_slot[(size_t) e] * gb;
            up_off_[i] = bu + tu->offset + (uint64_t) X.gu_slot[(size_t) e] * gb;
            down_off_[i] = bd + td->offset + (uint64_t) X.d_slot[(size_t) e] * db;
        }
    }
    for (size_t c = 0; c < g_.cls_gu.size(); ++c)
        max_ = std::max<size_t>(max_, (size_t) (2 * (uint64_t) gu_rb_[c] * (uint64_t) FF + (uint64_t) d_rb_[c] * (uint64_t) D));
    // The chunk-prefetch reader pool's O_DIRECT twin (measured 2.25 -> 4.65 GB/s on the scattered expert pattern).
    // STRATA_BLOB_DIRECT=0 keeps the buffered path; a filesystem that refuses O_DIRECT just leaves dfd_ = -1.
    const char* dv = std::getenv("STRATA_BLOB_DIRECT");
    direct_ok_ = !(dv && dv[0] == '0');
    for (size_t s = 0; s < m->size(); ++s) {   // every shard (a single-file GGUF: just the file)
        const std::string sp = s == 0 ? path_ : m->shard(s).path();
        const int f = ds41_sys::open(sp.c_str(), O_RDONLY);
        if (f < 0) { err = "cannot open " + sp; return false; }
        ds41_sys::posix_fadvise(f, 0, 0, POSIX_FADV_RANDOM);
        sfd_.push_back(f);
        sdfd_.push_back(direct_ok_ ? ds41_sys::open(sp.c_str(), O_RDONLY | O_DIRECT) : -1);
        direct_ok_ = direct_ok_ && sdfd_.back() >= 0;
    }
    fd_ = sfd_[0];
    dfd_ = direct_ok_ ? sdfd_[0] : -1;
    // decode misses (read_part) through it too: random 6.75 MB blobs 1.43 vs 1.99 ms cold (rdbench, SSD NVME), and no
    // page-cache churn inside the 84 GiB cgroup that also holds the pinned arena
    const char* pd = std::getenv("STRATA_DECODE_DIRECT");
    part_direct_ = !(pd && pd[0] == '0');
    uint64_t smax = 0;
    for (size_t c = 0; c < gu_rb_.size(); ++c) {
        smax = std::max<uint64_t>(smax, (uint64_t) gu_rb_[c] * (uint64_t) g_.n_ff);
        smax = std::max<uint64_t>(smax, (uint64_t) d_rb_[c] * (uint64_t) g_.n_embd);
    }
    scratch_ = (size_t) smax + 8192;   // the largest role span + the 4096-rounding a pread adds on both ends
    // The mirror is read O_DIRECT whatever the model file uses: on Windows the model file is best read buffered
    // (STRATA_BLOB_DIRECT=0: it is also memory-mapped, and NO_BUFFERING reads of a mapped file serialize), while the
    // mirror is a second, unmapped file.
#ifdef _WIN32
    const bool mirror_ok = true;
#else
    const bool mirror_ok = dfd_ >= 0;
#endif
    const char* mp = std::getenv("STRATA_BLOB_MIRROR");
    if (mp && mp[0] && sfd_.size() > 1) {
        std::fprintf(stderr, "blob mirror: NOT used (a split GGUF; the mirror is single-file only)\n");
    } else if (mp && mp[0] && part_direct_ && mirror_ok) {
        std::string merr;
        if (!open_mirror(mp, merr)) std::fprintf(stderr, "blob mirror: NOT used (%s)\n", merr.c_str());
    }
    return true;
}

// The mirror file: a 4096-byte-aligned header, then the blobs, each [gate | up | down] padded to 4096.
//   header: "S41MIRR1", u64 source file size, u64 n, u64 data start; then n x {i32 layer, i32 expert, u64 offset}
namespace {
struct MirrorHead { char magic[8]; uint64_t src_size, n, data; };
struct MirrorEnt { int32_t l, e; uint64_t off; };
}

bool Ds41FileBlobs::open_mirror(const char* path, std::string& err) {
    const int fd = ds41_sys::open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { err = std::string("cannot open ") + path + " O_DIRECT"; return false; }
    auto fail = [&](const std::string& m) { ds41_sys::close(fd); err = m; return false; };
    ds41_sys::Stat st{}, ss{};
    if (ds41_sys::fstat(fd, &st) != 0 || ds41_sys::fstat(fd_, &ss) != 0) return fail("fstat");
    uint8_t* h = (uint8_t*) ds41_sys::aligned_alloc(4096, 4096);
    std::unique_ptr<uint8_t, void (*)(void*)> hg(h, ds41_sys::aligned_free);
    if (ds41_sys::pread(fd, h, 4096, 0) != 4096) return fail("short header");
    MirrorHead H;
    std::memcpy(&H, h, sizeof H);
    if (std::memcmp(H.magic, "S41MIRR1", 8) != 0) return fail("bad magic");
    if (H.src_size != (uint64_t) ss.st_size) return fail("built from a different model file (size)");
    const uint64_t NE = (uint64_t) (g_.n_layers * g_.n_experts);
    if (H.n == 0 || H.n > NE || H.data < sizeof H + H.n * sizeof(MirrorEnt) || (H.data & 4095)) return fail("bad header");
    const size_t tb = (size_t) H.data;
    uint8_t* t = (uint8_t*) ds41_sys::aligned_alloc(4096, tb);
    std::unique_ptr<uint8_t, void (*)(void*)> tg(t, ds41_sys::aligned_free);
    for (size_t got = 0; got < tb;) {
        const ssize_t k = ds41_sys::pread(fd, t + got, tb - got, (int64_t) got);
        if (k <= 0) return fail("short index");
        got += (size_t) k;
    }
    std::vector<uint64_t> mo(NE, ~0ull);
    uint64_t end = H.data;
    for (uint64_t i = 0; i < H.n; ++i) {
        MirrorEnt x;
        std::memcpy(&x, t + sizeof H + i * sizeof x, sizeof x);
        if (x.l < 0 || x.l >= g_.n_layers || x.e < 0 || x.e >= g_.n_experts || (x.off & 4095) || x.off < H.data)
            return fail("bad index entry");
        mo[(size_t) (x.l * g_.n_experts + x.e)] = x.off;
        end = std::max<uint64_t>(end, x.off + ((expert_bytes(x.l, x.e) + 4095) & ~(uint64_t) 4095));
    }
    if (end > (uint64_t) st.st_size) return fail("truncated (" + std::to_string(st.st_size) + " of " + std::to_string(end) + " bytes)");
    // spot check: 8 blobs spread over the index (STRATA_BLOB_MIRROR_CHECK=N: N, 0 = all), whole, against the model file
    const char* ck = std::getenv("STRATA_BLOB_MIRROR_CHECK");
    uint64_t nchk = ck ? (uint64_t) std::max(0, std::atoi(ck)) : 8;
    if (nchk == 0 || nchk > H.n) nchk = H.n;
    std::vector<uint8_t> a(max_);
    for (uint64_t k = 0; k < nchk; ++k) {
        MirrorEnt x;
        std::memcpy(&x, t + sizeof H + (nchk > 1 ? k * (H.n - 1) / (nchk - 1) : 0) * sizeof x, sizeof x);
        const size_t n = expert_bytes(x.l, x.e);
        read(x.l, x.e, a.data());
        const size_t span = (n + 4095) & ~(size_t) 4095;
        uint8_t* m = (uint8_t*) ds41_sys::aligned_alloc(4096, span);
        std::unique_ptr<uint8_t, void (*)(void*)> mg(m, ds41_sys::aligned_free);
        size_t got = 0;
        while (got < span) {
            const ssize_t r = ds41_sys::pread(fd, m + got, span - got, (int64_t) (x.off + got));
            if (r <= 0) break;
            got += (size_t) r;
        }
        if (got < n || std::memcmp(a.data(), m, n) != 0)
            return fail("blob (" + std::to_string(x.l) + ", " + std::to_string(x.e) + ") differs from the model file");
    }
    const char* sh = std::getenv("STRATA_BLOB_MIRROR_SHARE");
    if (sh && sh[0]) mshare_ = std::max(0, std::min(1000, (int) std::lround(std::atof(sh) * 1000.0)));
    // STRATA_BLOB_MIRROR_PART_SHARE: read_part's own share (decode misses: one blob split over both drives waits on the
    // slower one); default = STRATA_BLOB_MIRROR_SHARE.  The prompt prefetch (read_staged, whole blobs) uses the latter.
    const char* ps = std::getenv("STRATA_BLOB_MIRROR_PART_SHARE");
    mpshare_ = ps && ps[0] ? std::max(0, std::min(1000, (int) std::lround(std::atof(ps) * 1000.0))) : mshare_;
    moff_.swap(mo);
    mirrored_ = (int64_t) H.n;
    mfd_ = fd;
    std::fprintf(stderr, "blob mirror: %s - %lld experts (%.1f GiB), share %.3f of each mirrored blob's parts (%.3f for "
                 "decode reads)\n", path, (long long) H.n, (double) (end - H.data) / 1073741824.0, mshare_ / 1000.0,
                 mpshare_ / 1000.0);
    return true;
}

bool Ds41FileBlobs::build_mirror(const std::string& out, const std::vector<std::pair<int32_t, int32_t>>& order,
                                 uint64_t budget, std::string& err) const {
    if (fd_ < 0) { err = "source not open"; return false; }
    if (sfd_.size() > 1) { err = "the mirror is single-file only (this GGUF is split)"; return false; }
    ds41_sys::Stat ss{};
    ds41_sys::fstat(fd_, &ss);
    std::vector<MirrorEnt> ent;
    std::vector<uint8_t> seen((size_t) (g_.n_layers * g_.n_experts), 0);
    uint64_t bytes = 0;
    for (const auto& le : order) {
        if (le.first < 0 || le.first >= g_.n_layers || le.second < 0 || le.second >= g_.n_experts) continue;
        uint8_t& sn = seen[(size_t) (le.first * g_.n_experts + le.second)];
        if (sn) continue;
        sn = 1;
        const uint64_t b = (expert_bytes(le.first, le.second) + 4095) & ~(uint64_t) 4095;
        if (bytes + b > budget) break;
        ent.push_back({le.first, le.second, 0});
        bytes += b;
    }
    const uint64_t data = (sizeof(MirrorHead) + ent.size() * sizeof(MirrorEnt) + 4095) & ~(uint64_t) 4095;
    uint64_t at = data;
    for (MirrorEnt& x : ent) {
        x.off = at;
        at += (expert_bytes(x.l, x.e) + 4095) & ~(uint64_t) 4095;
    }
    const int fd = ds41_sys::open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { err = "cannot create " + out + " (O_DIRECT)"; return false; }
    auto wr = [&](const uint8_t* p, uint64_t n, uint64_t off) {
        while (n) {
            const ssize_t k = ds41_sys::pwrite(fd, p, (size_t) std::min<uint64_t>(n, 1u << 30), (int64_t) off);
            if (k <= 0) return false;
            p += k;
            off += (uint64_t) k;
            n -= (uint64_t) k;
        }
        return true;
    };
    {   // header + index
        uint8_t* h = (uint8_t*) ds41_sys::aligned_alloc(4096, (size_t) data);
        std::memset(h, 0, (size_t) data);
        MirrorHead H;
        std::memcpy(H.magic, "S41MIRR1", 8);
        H.src_size = (uint64_t) ss.st_size;
        H.n = ent.size();
        H.data = data;
        std::memcpy(h, &H, sizeof H);
        std::memcpy(h + sizeof H, ent.data(), ent.size() * sizeof(MirrorEnt));
        const bool ok = wr(h, data, 0);
        ds41_sys::aligned_free(h);
        if (!ok) { ds41_sys::close(fd); err = "header write failed"; return false; }
    }
    // batches of kB blobs: 16 threads read the next batch (read_part, O_DIRECT) while this thread writes the last
    constexpr size_t kB = 48;
    const size_t stride = (max_ + 4095) & ~(size_t) 4095;
    uint8_t* buf[2] = {(uint8_t*) ds41_sys::aligned_alloc(4096, kB * stride), (uint8_t*) ds41_sys::aligned_alloc(4096, kB * stride)};
    auto fill = [&](size_t b0, uint8_t* dst) {
        const size_t n = std::min(kB, ent.size() - b0);
        constexpr int P = 8;
        std::atomic<size_t> nx{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 16; ++t)
            th.emplace_back([&]() {
                for (size_t q; (q = nx.fetch_add(1)) < n * P;) {
                    const MirrorEnt& x = ent[b0 + q / P];
                    uint8_t* d = dst + (q / P) * stride;
                    if (q % P == 0) std::memset(d + expert_bytes(x.l, x.e), 0, stride - expert_bytes(x.l, x.e));
                    read_part(x.l, x.e, (int) (q % P), P, d);
                }
            });
        for (auto& t : th) t.join();
    };
    const double t0 = now_ms();
    bool ok = true;
    if (!ent.empty()) fill(0, buf[0]);
    for (size_t b0 = 0, k = 0; b0 < ent.size() && ok; b0 += kB, ++k) {
        std::thread next;
        if (b0 + kB < ent.size()) next = std::thread(fill, b0 + kB, buf[(k + 1) & 1]);
        const size_t n = std::min(kB, ent.size() - b0);
        for (size_t i = 0; i < n && ok; ++i) {   // blobs are consecutive in the file: one write per run
            const MirrorEnt& x = ent[b0 + i];
            ok = wr(buf[k & 1] + i * stride, (expert_bytes(x.l, x.e) + 4095) & ~(uint64_t) 4095, x.off);
        }
        if (next.joinable()) next.join();
        if ((k % 40) == 0)
            std::fprintf(stderr, "\rbuild-mirror: %zu / %zu experts, %.1f GB/s   ", b0 + n, ent.size(),
                         (double) (ent[b0 + n - 1].off + expert_bytes(ent[b0 + n - 1].l, ent[b0 + n - 1].e) - data) /
                             ((now_ms() - t0) * 1e6));
    }
    ds41_sys::aligned_free(buf[0]);
    ds41_sys::aligned_free(buf[1]);
    if (ok) ok = ds41_sys::fsync(fd) == 0;
    ds41_sys::close(fd);
    if (!ok) { err = "write failed (disk full?)"; return false; }
    std::fprintf(stderr, "\nbuild-mirror: %s - %zu experts, %.2f GiB in %.1f s\n", out.c_str(), ent.size(),
                 (double) (at - data) / 1073741824.0, (now_ms() - t0) / 1000.0);
    return true;
}

bool Ds41FileBlobs::read_direct(uint64_t off, size_t n, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes, int fd) const {
    if (fd < 0) {   // the model file: the shard the tag names (an explicit fd = the mirror, plain offsets)
        if (!direct_ok_) return false;
        fd = sdfd(off);
        off = sloc(off);
    }
    if (fd < 0 || !scratch || !n) return false;
    constexpr uint64_t A = 4096;
    const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1);
    const size_t span = (size_t) (a1 - a0);
    if (span > scratch_bytes) return false;
    size_t got = 0;
    while (got < span) {
        const ssize_t k = ds41_sys::pread(fd, scratch + got, span - got, (int64_t) (a0 + got));
        if (k <= 0) return false;
        got += (size_t) k;
    }
    std::memcpy(dst, scratch + (off - a0), n);
    return true;
}

bool Ds41FileBlobs::mirror_pick(int64_t layer, int64_t expert) const {
    // whole-blob reads: a fixed hash of (layer, expert) sends mshare_ per mille of the mirrored blobs to the mirror
    const uint32_t h = (uint32_t) (((uint64_t) (layer * g_.n_experts + expert) * 2654435761ull) >> 7);
    return (int) (h % 1000u) < mshare_;
}

void Ds41FileBlobs::read_staged(int64_t layer, int64_t expert, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes) const {
    if (mfd_ >= 0 && scratch && scratch_bytes >= scratch_ && mirror_pick(layer, expert)) {
        const uint64_t mo = moff_[(size_t) (layer * g_.n_experts + expert)];
        if (mo != ~0ull) {   // the mirror blob is contiguous: [0, n) in steps the bounce holds
            const uint64_t n = expert_bytes(layer, expert);
            const uint64_t step = (uint64_t) (scratch_bytes - 8192) & ~(uint64_t) 4095;
            bool ok = true;
            for (uint64_t a = 0; a < n && ok; a += step)
                ok = read_direct(mo + a, (size_t) std::min<uint64_t>(step, n - a), dst + a, scratch, scratch_bytes, mfd_);
            if (ok) return;
        }
    }
    if (!direct_ok_ || scratch_bytes < scratch_) {
        read(layer, expert, dst);
        return;
    }
    Span s[3];
    spans(layer, expert, s);
    uint64_t at = 0;
    for (const Span& x : s) {
        if (!read_direct(x.off, (size_t) x.bytes, dst + at, scratch, scratch_bytes)) {
            read(layer, expert, dst);   // one short/refused span: fall back to the buffered whole-blob read
            return;
        }
        at += x.bytes;
    }
}

void Ds41FileBlobs::advise(int64_t layer, int64_t expert) const {
    if (fd_ < 0) return;
    Span s[3];
    spans(layer, expert, s);
    for (const Span& x : s)
        if (x.bytes) ds41_sys::posix_fadvise(sfd(x.off), (int64_t) sloc(x.off), (int64_t) x.bytes, POSIX_FADV_WILLNEED);
}

size_t Ds41FileBlobs::expert_bytes(int64_t l, int64_t e) const {
    const int c = g_.cls(l, e);
    return (size_t) (2 * (uint64_t) gu_rb_[(size_t) c] * (uint64_t) g_.n_ff + (uint64_t) d_rb_[(size_t) c] * (uint64_t) g_.n_embd);
}

size_t Ds41FileBlobs::blob_bytes(int64_t l) const {
    size_t m = 0;
    for (int64_t e = 0; e < g_.n_experts; ++e) m = std::max(m, expert_bytes(l, e));
    return m;
}

void Ds41FileBlobs::spans(int64_t l, int64_t e, Span s[3]) const {
    const size_t i = (size_t) (l * g_.n_experts + e);
    const int c = g_.cls(l, e);
    const uint64_t gb = (uint64_t) gu_rb_[(size_t) c] * (uint64_t) g_.n_ff, db = (uint64_t) d_rb_[(size_t) c] * (uint64_t) g_.n_embd;
    s[0] = {gate_off_[i], gb};
    s[1] = {up_off_[i], gb};
    s[2] = {down_off_[i], db};
}

// bytes [lo, hi) of the blob [gate | up | down] into dst + lo
static void pread_all(int fd, uint8_t* dst, uint64_t off, uint64_t n) {
    while (n) {
        const ssize_t r = ds41_sys::pread(fd, dst, (size_t) std::min<uint64_t>(n, 1u << 30), (int64_t) off);
        if (r <= 0) {
            std::fprintf(stderr, "Ds41FileBlobs: pread failed at %llu\n", (unsigned long long) off);
            std::abort();
        }
        dst += r;
        off += (uint64_t) r;
        n -= (uint64_t) r;
    }
}

void Ds41FileBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    Span s[3];
    spans(layer, expert, s);
    uint64_t at = 0;
    for (const Span& x : s) {
        pread_all(sfd(x.off), dst + at, sloc(x.off), x.bytes);
        at += x.bytes;
    }
}

void Ds41FileBlobs::read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const {
    Span s[3];
    spans(layer, expert, s);
    const uint64_t total = s[0].bytes + s[1].bytes + s[2].bytes;
    uint64_t lo = total * (uint64_t) part / (uint64_t) nparts, hi = total * (uint64_t) (part + 1) / (uint64_t) nparts;
    lo &= ~(uint64_t) 4095;
    if (part + 1 < nparts) hi &= ~(uint64_t) 4095;
    else hi = total;
    uint64_t at = 0;
    uint8_t* bb = part_direct_ && (direct_ok_ || mfd_ >= 0) ? take_bounce() : nullptr;
    // the mirror's share of the parts: part p goes there when floor((p+1)*share) > floor(p*share) - spread evenly,
    // and a 1-part read stays on the model file.  The mirror blob is contiguous: [lo, hi) is one aligned span there.
    if (mfd_ >= 0 && bb && nparts > 1 && ((part + 1) * mpshare_) / 1000 != (part * mpshare_) / 1000) {
        const uint64_t mo = moff_[(size_t) (layer * g_.n_experts + expert)];
        if (mo != ~0ull) {
            const uint64_t step = (uint64_t) (scratch_ - 8192) & ~(uint64_t) 4095;
            bool ok = true;
            for (uint64_t a = lo; a < hi && ok; a += step)
                ok = read_direct(mo + a, (size_t) std::min<uint64_t>(step, hi - a), dst + a, bb, scratch_, mfd_);
            if (ok) {
                give_bounce(bb);
                return;
            }
        }
    }
    for (const Span& x : s) {
        const uint64_t a = std::max(lo, at), b = std::min(hi, at + x.bytes);
        if (a < b && !(bb && read_direct(x.off + (a - at), (size_t) (b - a), dst + a, bb, scratch_)))
            pread_all(sfd(x.off), dst + a, sloc(x.off) + (a - at), b - a);
        at += x.bytes;
    }
    if (bb) give_bounce(bb);
}

std::vector<std::pair<int32_t, int32_t>> ds41_read_ranked(const std::string& path, std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    std::vector<std::pair<int32_t, int32_t>> r;
    if (!in) { err = "cannot open " + path; return r; }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    std::vector<int32_t> v((size_t) (bytes / 4));
    in.read((char*) v.data(), (std::streamsize) (v.size() * 4));
    for (size_t i = 0; i + 1 < v.size(); i += 2) r.push_back({v[i], v[i + 1]});
    return r;
}

}  // namespace strata::v41
