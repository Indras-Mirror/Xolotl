// tools/ds4/ds4_moe.hpp - the DeepSeek-V4 routed-expert TIER as a reusable library (slice ds4-moe).
//
// WHAT THIS IS.  `tools/ds4/moe_replay.cpp` proved the expert half of a DeepSeek-V4 decode on this machine
// (FINDINGS s10-s12): a VRAM cache of 6.75 MiB expert slots seeded from a routing profile, the CPU ExpertPool on
// the misses, a PCIe share of the misses computed on the GPU in parallel, a pinned host arena holding every
// expert, and predicted expert prefetch on a second stream under the dense work - measured 20.57 tok/s at 2150
// slots, pf-b 1.43, pcie 0.25 dithered.  That file is a HARNESS: it drives the pieces itself, per token, with a
// synthetic stand-in for the dense half and recorded routes instead of a router.
//
// `Ds41MoeTier` is the same machinery as one object, used once per layer by the real engine:
//
//     tier.init(gguf, cfg, err);                     // once, at startup
//     tier.seed_from_routes(routes_bin, err);        // the static profile, once
//     for each layer l of each token:
//         tier.prefetch(l, top16);                   // right after predict(l), before attention(l)
//         tier.run(l, ids6, w6, x, out);             // after router(l); out = sum_k w_k * expert_k(x)
//
// `run` is synchronous: the engine calls it per layer and needs `out` before the next one.  `prefetch` is not:
// it issues its DMAs on a second stream and returns immediately, so they overlap the dense work the caller is
// about to do between `prefetch` and `run`.
//
// TWO MODES.  `cfg.cpu_only` runs the whole expert half on the CPU pool: no ExpertCache, no PCIe share, no CUDA
// call of any kind, so the tier can be linked into a binary that must never touch the driver (this slice's gate,
// and machines without a GPU).  Otherwise the three tiers of the plan run concurrently and the weighted sum is
// assembled from all of them.
//
// THE INVARIANT, and the one thing to read before changing anything here:
//
//     for any (layer, ids, weights, x), run()'s output equals the weighted sum of the dequantised experts
//     computed in F32 - the only permitted difference being the expert kernels' own 8-bit activation rounding.
//
// It is tested by `tools/ds4/test_ds4_moe.cpp` against a dequant+F32 reference (weights through ggml's
// `to_float`), CPU-only, at rel < 1e-3 per layer and per expert.
//
// THE SWIGLU CLAMP.  DeepSeek-V4's graph clamps the expert pre-activations before the SwiGLU - gate -> min(gate, L),
// up -> clamp(up, -L, L), L = `swiglu_clamp_exp` = 10.0 (`tools/ds4/ds4_ref.cpp`, mirroring llama.cpp's DEEPSEEK4
// branch).  The native expert kernels apply it through `NativeFmt::swiglu_limit` (CPU: `native_gu_rows` and the
// AVX2/AVX-512 multi-token kernels) and `NativeExpertLayout::swiglu_limit` (CUDA: the grouped kernels' SwiGLU
// passes); `ds41_moe_geom_from_gguf` reads L into `Ds41MoeGeom::swiglu_limit` and the tier passes it to both.  Every
// other model leaves the limit at +inf, which is the unclamped arithmetic bit for bit.  On real activations L
// binds rarely (FINDINGS s13), so `test_ds4_moe` also reruns its real arms with L forced to 1.0 to check the path.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <mutex>
#include <vector>

namespace strata::v41 {

namespace cpu = strata::kernels::cpu;

// ================================ geometry ================================

/// The routed-expert half of one DeepSeek-V4-Flash layer.  The defaults are the real model's
/// (docs/ds4/GGUF_INVENTORY.txt): 43 layers x 256 experts, top-6, gate/up IQ2_XXS, down Q2_K, n_embd 4096,
/// n_ff 2048.  `ds41_moe_geom_from_gguf` reads them from metadata instead of trusting the defaults.
struct Ds41MoeGeom {
    int64_t n_embd = 4096;      ///< n_embd
    int64_t n_ff = 2048;        ///< n_ff of one routed expert
    int64_t n_layers = 43;      ///< layers that route
    int64_t n_experts = 256;    ///< experts per layer
    int64_t top_k = 6;          ///< experts routed per token
    int gu_type = 16;           ///< GGML_TYPE_IQ2_XXS (gate and up)
    int d_type = 10;            ///< GGML_TYPE_Q2_K (down)
    /// `deepseek4.swiglu_clamp_exp` (10.0 on every layer of the 0731 GGUF; one value - the tier has one format):
    /// gate -> min(gate, lim), up -> clamp(up, -lim, lim) before the SwiGLU.  +inf = no clamp.
    float swiglu_limit = std::numeric_limits<float>::infinity();
    /// PER-LAYER FORMATS (MiMo-V2.6-Flash GSQ-RCO picks a format per tensor: gate, up and down independently).
    /// Empty = every layer is `gu_type`/`gu_type`/`d_type` (DeepSeek-V4: the uniform path, unchanged).  Otherwise
    /// `n_layers` entries each, indexed by the MODEL's layer number; -1 = that layer has no routed experts (MiMo's
    /// dense layer 0).  `gu_type`/`d_type` then hold the first routed layer's types (reports only).
    std::vector<int> gate_types, up_types, down_types;

    /// PER-EXPERT FORMAT CLASSES (DeepSeek-V4.1 GSQ mixed tiers): eclass[l * n_experts + e] is expert e's class; class
    /// c has gate = up = cls_gu[c] and down = cls_d[c] (at most 9: Q2_0 / Q3_K / PTQ1_0 squared).  Empty = an
    /// expert's class is its LAYER (the per-layer / uniform geometry, exactly as before).  Every format, blob size and
    /// GPU layout below is per class; the VRAM LRU swaps only within a (layer, class).
    std::vector<int16_t> eclass;
    std::vector<int> cls_gu, cls_d;
    bool per_expert() const { return !eclass.empty(); }
    int64_t n_classes() const { return per_expert() ? (int64_t) cls_gu.size() : n_layers; }
    int cls(int64_t l, int64_t e) const { return per_expert() ? (int) eclass[(size_t) (l * n_experts + e)] : (int) l; }
    bool per_layer() const { return !gate_types.empty(); }
    bool routed(int64_t l) const { return l >= 0 && l < n_layers && (!per_layer() || gate_types[(size_t) l] >= 0); }
    int gate_type(int64_t l) const { return per_layer() ? gate_types[(size_t) l] : gu_type; }
    int up_type(int64_t l) const { return per_layer() ? up_types[(size_t) l] : gu_type; }
    int down_type(int64_t l) const { return per_layer() ? down_types[(size_t) l] : d_type; }
    /// Layers that route (n_layers for a uniform geometry).
    int64_t n_routed() const {
        if (!per_layer()) return n_layers;
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += routed(l);
        return n;
    }
};

/// Reads `n_embd / n_ff / n_layers / n_experts / top_k` from the GGUF's `<arch>.*` metadata (`general.architecture`:
/// deepseek4, mimo2, ...) and the expert quant types of every layer's `blk.L.ffn_{gate,up,down}_exps.weight`.  All
/// layers present and alike -> the uniform geometry (per-layer vectors empty), exactly as before; otherwise the
/// per-layer vectors.  Only the header region of the file is read; no tensor data is touched.
bool ds41_moe_geom_from_gguf(const std::string& gguf, Ds41MoeGeom& g, std::string& err);

/// The native blob layout for `g` (the three role slices back to back) and its byte size.  False with `err` when
/// ggml-cpu has no dot product for the pair at this geometry, i.e. when the CPU tier could not run at all.  For a
/// per-layer geometry this is the LARGEST routed layer's layout (what buffers are sized by).
bool ds41_moe_blob_layout(const Ds41MoeGeom& g, cpu::NativeFmt& f, std::string& err);
/// Class `c`'s layout (= ds41_moe_layer_layout(g, c) unless the geometry is per-expert).
bool ds41_moe_class_layout(const Ds41MoeGeom& g, int64_t c, cpu::NativeFmt& f, std::string& err);
/// Layer `l`'s own layout (any geometry).  False for a layer without routed experts.
bool ds41_moe_layer_layout(const Ds41MoeGeom& g, int64_t l, cpu::NativeFmt& f, std::string& err);

// ================================ blob sources ================================

/// Where one expert's bytes come from.  `read` is synchronous and const: the tier reads under a lock-free
/// per-thread split, and a source that is not reentrant is the caller's problem (both shipped ones are).
class Ds41BlobSource {
public:
    virtual ~Ds41BlobSource() = default;
    /// A short label for the startup report ("gguf mmap", "memory").
    virtual const char* kind() const = 0;
    /// Validates the source against the geometry (the roles exist, their types and strides are right).
    virtual bool open(std::string& err) = 0;
    /// Bytes of one expert blob: [gate rows | up rows | down rows].  The LARGEST layer's for a per-layer geometry.
    virtual size_t blob_bytes() const = 0;
    /// Bytes of one expert blob of layer `l` (`blob_bytes()` unless the formats differ per layer).
    virtual size_t blob_bytes(int64_t l) const { (void) l; return blob_bytes(); }
    /// Writes one expert's blob into `dst` (blob_bytes() large).
    virtual void read(int64_t layer, int64_t expert, uint8_t* dst) const = 0;
    /// Part `part` of `nparts` of the same blob: bytes [p0, p1) of it into dst + p0 (p0 = blob * part / nparts, 4 KiB
    /// rounded; the parts tile the blob), so several threads can read ONE blob.  Default: part 0 reads it all.
    virtual void read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const {
        if (part == 0) read(layer, expert, dst);
        (void) nparts;
    }
    /// The chunk-prefetch reader pool's read: the same bytes as `read`, through the fastest path the source has.  A
    /// source with a raw file layout overrides this to pread through an `O_DIRECT` twin fd via the aligned `scratch`
    /// bounce (`read_staged_scratch()` bytes per thread, 4096-aligned).  Measured on the V41-GSQ GGUF: the scattered
    /// 3 MiB expert pattern runs 2.25 GB/s buffered and 4.65 GB/s through O_DIRECT at 16 threads (buffered copies
    /// through the page cache; `dd` on the same file: 2.7 GB/s one stream, 5.1 GB/s at 16).  Default: `read`.
    virtual void read_staged(int64_t layer, int64_t expert, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes) const {
        (void) scratch;
        (void) scratch_bytes;
        read(layer, expert, dst);
    }
    /// Hint that `expert` of `layer` will be read soon (asynchronous kernel readahead into the page cache).  Default: no-op.
    virtual void advise(int64_t layer, int64_t expert) const { (void) layer; (void) expert; }
    /// Bytes one `read_staged` scratch bounce must have.  0 = `read_staged` ignores it.
    virtual size_t read_staged_scratch() const { return 0; }
};

/// The production source: the three per-role tensors of a `deepseek4` GGUF, mmap'd in place, so assembling one
/// blob is three memcpys of pages the OS already has.  Reading `N` experts touches `N * 6.75 MiB` of the file,
/// not the model.
class Ds41GgufBlobs : public Ds41BlobSource {
public:
    Ds41GgufBlobs(Ds41MoeGeom g, std::string gguf);
    ~Ds41GgufBlobs() override;
    Ds41GgufBlobs(const Ds41GgufBlobs&) = delete;
    Ds41GgufBlobs& operator=(const Ds41GgufBlobs&) = delete;

    const char* kind() const override { return "gguf mmap"; }
    bool open(std::string& err) override;
    size_t blob_bytes() const override { return blob_; }
    size_t blob_bytes(int64_t l) const override;
    void read(int64_t layer, int64_t expert, uint8_t* dst) const override;
    void read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const override;

private:
    /// Bytes [lo, hi) of role r's slice of (layer, expert) into dst (O_DIRECT, pread, then the mmap as fallbacks).
    void read_span(int r, int64_t layer, int64_t expert, size_t lo, size_t hi, uint8_t* dst) const;
    struct Impl;
    std::unique_ptr<Impl> im_;
    Ds41MoeGeom g_;
    std::string path_;
    size_t blob_ = 0;
};

/// An in-memory source: the gate's synthetic arms and anything that builds expert bytes itself.  `set` copies, so
/// the caller may free its buffer.  Sparse by (layer, expert) - a test materializes a handful of experts of an
/// 11,008-slot model, so the storage must not be sized for the whole table.  A pair never set reads as zeros;
/// `has()` is what a test asserts with, so reaching that fallback is a visible mistake rather than a silent one.
class Ds41MemoryBlobs : public Ds41BlobSource {
public:
    Ds41MemoryBlobs(Ds41MoeGeom g, size_t blob_bytes) : g_(g), blob_(blob_bytes) {}
    const char* kind() const override { return "memory"; }
    bool open(std::string& err) override;
    size_t blob_bytes() const override { return blob_; }
    void read(int64_t layer, int64_t expert, uint8_t* dst) const override;
    void set(int64_t layer, int64_t expert, const uint8_t* blob);
    bool has(int64_t layer, int64_t expert) const;
    int64_t count() const { return (int64_t) blobs_.size(); }

private:
    Ds41MoeGeom g_;
    size_t blob_ = 0;
    std::map<std::pair<int64_t, int64_t>, std::vector<uint8_t>> blobs_;
};

/// DeepSeek-V4.1 (`deepseek41`, tools/v41/v41_gguf.py): each expert's gate/up/down rows live in the stacked tensor of
/// its type (slot = its rank among that type's experts).  A blob is assembled from the three spans with preads (page
/// cache; no mmap of the 150 GiB file).  `geom()` is the per-expert-class geometry for the tier.
class Ds41FileBlobs : public Ds41BlobSource {
public:
    explicit Ds41FileBlobs(std::string gguf);
    ~Ds41FileBlobs() override;
    const char* kind() const override { return "deepseek41 gguf"; }
    bool open(std::string& err) override;
    size_t blob_bytes() const override { return max_; }
    size_t blob_bytes(int64_t l) const override;
    void read(int64_t layer, int64_t expert, uint8_t* dst) const override;
    void read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const override;
    void read_staged(int64_t layer, int64_t expert, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes) const override;
    size_t read_staged_scratch() const override { return scratch_; }
    void advise(int64_t layer, int64_t expert) const override;
    const Ds41MoeGeom& geom() const { return g_; }
    /// bytes of expert e of layer l (its class's blob)
    size_t expert_bytes(int64_t l, int64_t e) const;
    /// The mirror (STRATA_BLOB_MIRROR=<file>, written by `ds41_generate --build-mirror`): packed copies of the coldest
    /// experts on a SECOND drive.  read_part then reads a share of each mirrored blob's parts from it
    /// (STRATA_BLOB_MIRROR_SHARE, default 0.28), so both drives serve one blob at once.  Experts mirrored, 0 = none.
    /// 0.28 = 2 of 8 parts: the rb2 microbench (6.75 MiB blobs, 8 parts, 32 threads) gives the model drive 6.22 GB/s,
    /// the mirror 3.26, and the pair 8.29 at 2/8 vs 7.95 at 3/8 - the mirror's own ceiling, so a larger share only
    /// unloads the faster drive (8.13 at 1/4 with independent offsets; 3/8's 2% is inside the engine's noise).
    int64_t mirrored() const { return mirrored_; }
    /// Writes the mirror: `order` coldest first, until `budget` bytes.  Returns false with `err` on failure.
    bool build_mirror(const std::string& out, const std::vector<std::pair<int32_t, int32_t>>& order, uint64_t budget,
                      std::string& err) const;

private:
    struct Span { uint64_t off = 0, bytes = 0; };
    void spans(int64_t l, int64_t e, Span s[3]) const;
    /// One 4096-aligned `pread` of [off, off+n) through the O_DIRECT twin fd, sliced out of `scratch`.  False when
    /// there is no twin fd or the bounce is too small (the caller then falls back to the buffered `read`).
    bool read_direct(uint64_t off, size_t n, uint8_t* dst, uint8_t* scratch, size_t scratch_bytes, int fd = -1) const;
    bool open_mirror(const char* path, std::string& err);
    int mfd_ = -1;                        ///< the mirror (O_DIRECT), -1 = none
    std::vector<uint64_t> moff_;          ///< per (layer, expert): the blob's offset in the mirror, ~0 = not mirrored
    int64_t mirrored_ = 0;
    int mshare_ = 280;                    ///< per mille of a mirrored blob's parts read from the mirror
    /// read_part's aligned bounces (one per concurrent part, kept for the next call)
    uint8_t* take_bounce() const;
    void give_bounce(uint8_t* p) const;
    mutable std::mutex bmu_;
    mutable std::vector<uint8_t*> bfree_, ball_;
    bool part_direct_ = true;    ///< read_part through the O_DIRECT twin (STRATA_DECODE_DIRECT=0: buffered)
    std::string path_;
    int fd_ = -1;
    int dfd_ = -1;               ///< O_DIRECT twin of fd_ (-1 where the filesystem refused it)
    size_t scratch_ = 0;         ///< the aligned bounce one read_direct span needs
    Ds41MoeGeom g_;
    size_t max_ = 0;
    std::vector<uint64_t> base_;          ///< [layer * 3 * n_types...] see .cpp
    std::vector<int32_t> slot_gu_, slot_d_;
    std::vector<uint64_t> gate_off_, up_off_, down_off_;   ///< per (layer, expert): absolute file offsets
    std::vector<uint32_t> gu_rb_, d_rb_;                   ///< per class: row bytes of gate/up and of down
};

/// Experts ranked by calibration routing counts (tools/v41/engine/v41_rank.py writes int32 (layer, expert) pairs).
std::vector<std::pair<int32_t, int32_t>> ds41_read_ranked(const std::string& path, std::string& err);

// ================================ configuration ================================

/// The measured best arm of FINDINGS s12 as the defaults: 2150 slots (the 24 GiB card's real expert budget),
/// pcie 0.25 dithered, pf-b 1.43 experts/layer, and a host arena budgeted for every expert of the real model
/// (72.6 GiB).  `arena_gib` is a budget, not a promise: the arena fills in profile order up to it and everything
/// past it is read from the blob source on demand (the file tier).
struct Ds41MoeConfig {
    int64_t slots = 2150;         ///< VRAM expert slots (6.75 MiB each); 0 or `no_cache` = none
    double pcie_frac = 0.25;      ///< share of the (non-prefetched) misses computed on the GPU - single-token decode only
    /// The same share for the batched path (gpu_run_n_ex): prompt passes AND every DSpark/serve decode or verify call.
    /// < 0 = STRATA_TIER_BATCH_PCIE, else 1.0 (every arena miss over PCIe; measured best on 8-core dual-channel DDR4).
    /// Many-core / many-channel hosts want it low: 0.0 = +60% serve decode on a 24-core Threadripper (GitHub #3).
    double tier_batch_pcie = -1.0;
    int threads = 0;              ///< CPU pool workers; 0 = every physical core but the first
    bool pin = true;              ///< register the host arena with CUDA so the PCIe share DMAs out of it
    double arena_gib = 72.6;      ///< host arena budget; the file tier takes the rest
    double pf_b = 1.43;           ///< predicted prefetch budget, experts per layer
    bool dither = true;           ///< error-diffused rounding for both the PCIe share and the prefetch budget
    bool cpu_only = false;        ///< no CUDA at all: the pool computes every expert
    bool no_cache = false;        ///< GPU tier without a VRAM cache (every miss is CPU or PCIe)
    /// Uniform blobs (DS4): open the VRAM cache at the seed instead of at init, so prompt chunks (run_chunk) can use
    /// that VRAM first (build_arena_from_routes -> chunks -> release_chunk -> seed_from_routes).
    bool defer_cache = false;
    double mem_floor_gib = 3.0;   ///< refuse an arena that would leave less than this much MemAvailable
    double max_arena_gib = 60.0;  ///< hard ceiling on the arena, whatever `arena_gib` says
    /// VRAM budget for the expert cache in GiB, for geometries whose blob size varies per layer (MiMo): the seed
    /// takes the ranked profile's head until `slots` experts OR this many bytes, sizing each slot to its expert.
    /// 0 = `slots` x the largest blob.  Ignored for uniform geometries (DS4: `slots` alone, as before).
    double slot_gib = 0.0;
    /// Leave the experts the VRAM seed takes OUT of the host arena: a cache that never evicts never reads their arena
    /// copies again, so the bytes go to the next-ranked experts (fewer file-tier reads); with `vram_lru` an evicted
    /// expert is demoted into the arena slot its replacement vacates.  Off = the arena holds the ranking's head.
    bool arena_skip_resident = false;
    /// Decode: a file-tier expert that `run` had to read replaces the least-recently-used arena expert of its layer
    /// (same blob size), so the experts a conversation uses stay in RAM after their first use.  Off = static arena.
    bool arena_adapt = false;
    /// Prompt chunks (`run_chunk`): the streamed experts' products through llama.cpp's MMQ (int8 tensor cores; the
    /// activations rounded to MMQ's q8_1) instead of the grouped MMVQ kernel, on the layers whose gate/up/down types
    /// MMQ covers and that have no SwiGLU clamp.  Needs a build with the prompt MMQ path (`strata_mmq`, with
    /// STRATA_MMQ_KQUANTS for K-quants / MXFP4); otherwise, and for VRAM-resident experts, the MMVQ kernel runs.
    bool chunk_mmq = false;
    /// Prompt chunks: while layer l's experts compute (and the caller runs layer l+1's dense half), DMA layer l+1's
    /// arena experts into the other device half, which is then sized to a whole layer.  Off with arena_adapt.
    bool chunk_prestage = false;
    /// Prompt chunks: while layer l computes, a background reader pool fills a pinned host slab with layer l+1's
    /// FILE-TIER experts (neither VRAM-resident nor in the RAM arena), so run_chunk(l+1) DMAs straight out of the slab
    /// instead of reading on its own critical path.  Two slabs ping-pong; a layer that fits neither falls back to
    /// run_chunk's on-demand reader.  `chunk_prefetch_gib` caps the PAIR (0 = auto: the largest routed layer's file
    /// tier, capped by the host RAM MemAvailable can spare above `mem_floor_gib`).  Measured: the 4096-token prompt's
    /// 52.9 s file wait is what this moves off the critical path.
    bool chunk_prefetch = true;
    double chunk_prefetch_gib = 0.0;
    int cuda_device = 0;   ///< CUDA ordinal for this tier (layer split)
    /// REAP saliency (calibration only): every prompt chunk reads its experts' outputs back and accumulates, per
    /// (layer, expert), sum of routing weight x ||expert output||_2 and the token count (saliency_sum / saliency_count).
    bool saliency = false;
    /// REAP pruning: [n_layers * n_experts], nonzero = the expert is never routed (the dense side's selection bias
    /// masks it), so it takes no VRAM slot and no arena space - every ranking the tier seeds from skips it.  Empty = none.
    std::vector<char> exclude;
    /// Decode (run): a routed expert that is a VRAM miss and weighs less than `skip_miss` x the token's weight sum is
    /// dropped (not fetched, contributes 0).  Hits are never dropped.  0 = off.  A quality trade: measure ppl.
    float skip_miss = 0.0f;
    /// Decode (run): the same rule for the FILE TIER only (experts neither in VRAM nor in the RAM arena: an NVMe read
    /// each, ~2 ms).  0 = off.  A quality trade: measure ppl.
    float skip_file = 0.0f;
    /// Decode (MiMo and DS4): a PCIe-share miss (and a prefetched expert the routing used) takes
    /// the least-recently-used VRAM slot of its layer instead of a staging buffer, so the cache follows the
    /// conversation (route_probe sim, 1800 slots: static 40.5% held-out hit, LRU 61.6%; FINDINGS s16).  Same math per
    /// expert; only where the blob sits changes.  Off = the static seed, as before.
    bool vram_lru = false;
    /// A second CUDA device (its index in this process) that holds a static set of experts and computes them itself:
    /// the ranking right after the VRAM cache's head, up to `remote_gib` (0 = its free memory less
    /// `remote_reserve_gib`), kept out of the arena.  -1 = off.
    int remote_device = -1;
    double remote_gib = 0.0;
    double remote_reserve_gib = 1.5;
    uint64_t seed = 20261005;     ///< the synthetic arm's generator (unused by the tier itself)

    static constexpr int kPredW = 16;   ///< ranked predictions the caller passes to `prefetch`
    static constexpr int kMaxPf = 6;    ///< prefetch staging slots (the DMA window of one layer)
};

/// Per-token counters, split exactly as FINDINGS s12 reports them.  `hits` includes the prefetched misses that
/// the caller then routed to (they are computed from the staged blob on the GPU, like a resident slot).
struct Ds41MoeStats {
    int64_t hits = 0;               ///< served from a resident VRAM slot
    int64_t prefetched_useful = 0;  ///< routed misses whose blob the prefetch had already staged
    int64_t prefetch_issued = 0;    ///< DMAs the prefetch started (useful or not)
    int64_t prefetch_dummy = 0;     ///< guesses outside the arena (the DMA happens; the bytes are zeros)
    int64_t advised = 0;            ///< file-tier experts hinted for readahead (advise_file)
    int64_t spec_issued = 0;        ///< speculative file-tier reads started (spec_read)
    int64_t spec_hits = 0;          ///< file-tier misses served from a speculative read
    int64_t cpu = 0;                ///< computed by the CPU pool
    int64_t pcie = 0;               ///< misses DMA'd to the GPU and computed there
    int64_t file_tier = 0;          ///< reads served by the blob source instead of the arena
    double gap_ms = 0, hit_ms = 0, pcie_ms = 0, cpu_ms = 0, wall_ms = 0;
    double file_ms = 0;             ///< time spent reading file-tier blobs (inside run / prefetch)
    int64_t arena_swaps = 0;        ///< arena_adapt: file-tier experts moved into the arena
    int64_t vram_swaps = 0;         ///< vram_lru: misses / prefetches moved into a VRAM slot
    int64_t vram_demotes = 0;       ///< ...of which the victim went back to the arena (it had no arena copy)
    int64_t skipped = 0;            ///< skip_miss: low-weight misses dropped

    void add(const Ds41MoeStats& o);
    int64_t lookups() const { return hits + cpu + pcie + skipped; }
};

/// The tier's private state, defined in ds4_moe.cpp.  It carries the CUDA members in a CUDA build and none in a
/// CPU-only one, which is what keeps this header - and every binary linking it - free of a CUDA dependency.
struct Ds41MoeImpl;

/// The tier.  One instance serves every layer of a model (its cache is `n_layers x n_experts`), so the engine
/// holds one and calls `run` 43 times per token.
class Ds41MoeTier {
public:
    /// The most tokens one run_multi / run_multi_to_dev call takes (a depth-4 verify pass is 5-6 tokens; a lookup
    /// verify and the DSpark prompt tail are 16 - ONE call per layer at 16, so the experts the two halves share cross
    /// PCIe once instead of twice).  The pool's per-job cap (cpu::MAXT) stays 8: a CPU expert routed by more tokens
    /// is split into several jobs (kJobSplit in ds41_moe.cpp).
    static constexpr int kMaxMulti = 16;

    Ds41MoeTier();
    ~Ds41MoeTier();
    Ds41MoeTier(const Ds41MoeTier&) = delete;
    Ds41MoeTier& operator=(const Ds41MoeTier&) = delete;

    /// `gguf` convenience: builds a `Ds41GgufBlobs` over the file and takes ownership of it.
    bool init(const std::string& gguf, const Ds41MoeConfig& cfg, std::string& err);
    /// The general form.  `blobs` must outlive the tier.  In CUDA mode this allocates the device buffers and the
    /// VRAM cache; in `cpu_only` it allocates nothing but the CPU pool.  It does NOT build the host arena (52-73
    /// GiB of the real model): that is `build_arena`, which `seed_from_routes`/`seed_from_ranked` call, because
    /// the arena is ORDERED and building it twice would read the file twice.  Seed before the first `run`, or the
    /// tier runs the file tier for every expert (correct, slower).
    bool init(const Ds41MoeGeom& geom, Ds41BlobSource* blobs, const Ds41MoeConfig& cfg, std::string& err);

    /// Builds the host arena, in `ranked` order, up to the configured budget.  Called by the seed functions with
    /// the routing profile, or with every (layer, expert) in index order by a caller that has none; public so a
    /// test can force it.  The first build wins: a second call with a ranking after a ranked build is a no-op
    /// (the same rule the cache follows, where the first seed takes the free slots), so a re-seed re-fills the
    /// cache and never re-reads the arena.
    bool build_arena(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);
    /// The RAM arena alone, in the order seed_from_ranked gives it (that seed then only opens + fills the VRAM cache)
    bool build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);
    /// VRAM slot budget for a cache not opened yet (deferred seed: measured after the prompt freed its buffers)
    void set_slot_gib(double gib);
    /// Turn the VRAM expert cache on/off for the NEXT seed (off: the arena + file tier serve every miss - slower
    /// decode, no VRAM).  An ALREADY OPEN cache is not dropped by this; release_cache() does that.
    void set_cache_enabled(bool on);

    /// Ranks every (layer, expert) by routing frequency over the TRAIN half of a `ds4routes.bin` file
    /// (alternating 512-token blocks - the split `tools/ds4/route_skew.py` scores with), builds the host arena
    /// from it and fills the VRAM cache with the top `slots` of it.  The format is `moe_replay.cpp`'s:
    /// [512-token ubatch][layer][token][6] u16.
    bool seed_from_routes(const std::string& routes_bin, std::string& err);
    /// The same for route files with `batch_tokens`-token ubatches (tools/mimo/route_probe writes 2048) and only the
    /// routed layers in them (`Ds41MoeGeom::n_routed()` per token).
    bool seed_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err);
    /// The same fill from an already-ranked list (the arena and the gate both have one to hand).
    bool seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

    /// Issues, asynchronously, the DMAs for up to `pf_b` of `top16`'s experts that are not resident, into this
    /// layer's staging slots.  Call it after the router prediction for `layer` and BEFORE the layer's dense GPU
    /// work; `run(layer, ...)` consumes what it staged and clears the staging.  No-op in `cpu_only` and when
    /// `pf_b <= 0`.  `n` may be smaller than the caller's list; never larger than kPredW.
    void prefetch(int64_t layer, const int32_t* top16, int n = Ds41MoeConfig::kPredW);
    /// The multi-token verify pass's prefetch (call before attn_router(layer), so the DMAs run under its attention):
    /// `ids` = ranked candidate experts (best first, may repeat), the first `max_slots` (<= kMaxPf) that are neither
    /// VRAM-resident nor outside the arena are DMA'd into the prefetch slots on the prefetch stream; run_multi_to_dev
    /// computes a routed one from its slot (a hit).  Independent of pf_b.  A hint only: results do not depend on it.
    void prefetch_multi(int64_t layer, const int32_t* ids, int n, int max_slots);
    /// Decode file-tier readahead: for the predicted experts of `layer` that are neither VRAM-resident nor in the RAM
    /// arena, ask the kernel to read their bytes into the page cache now (posix_fadvise WILLNEED), so the tier's
    /// buffered read later is a page-cache copy.  The SSD is otherwise idle in decode, so a wrong guess costs ~nothing.
    void advise_file(int64_t layer, const int32_t* ids, int n);
    /// Decode speculative reads: start O_DIRECT reads of `ids` (a later layer's predicted experts) that are neither
    /// VRAM-resident nor in the arena into a pinned stash; that layer's run takes them from there.  A hint only.
    void spec_read(int64_t layer, const int32_t* ids, int n, bool d3 = false);

    /// out[i] = sum_k w6[k] * expert_{ids6[k]}(x)[i], with `x` on the host.  Synchronous.
    bool run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out);
    /// `run` with the weighted sum computed on the card and written to the device row `out_dev` (n_embd floats, e.g.
    /// the dense half's routed tensor): the same bits as `run`'s host sum, without its parts readback and upload.
    /// GPU tier only (false in cpu_only).
    bool run_to_dev(int64_t layer, const int32_t* ids6, const float* w6, const float* x, void* out_dev);
    /// The same with `x` already on the device (the engine's attention output).  CUDA modes only.
    bool run_dev(int64_t layer, const int32_t* ids6, const float* w6, const void* x_dev, float* out);

    void close();

    const Ds41MoeStats& stats() const;
    void reset_stats();
    void add_stats(const Ds41MoeStats& s);

    const Ds41MoeGeom& geom() const;
    const cpu::NativeFmt& fmt() const;
    const Ds41MoeConfig& config() const;
    const char* mode() const;          ///< "cpu-only" | "gpu" | "uninitialised"
    int64_t resident() const;          ///< experts admitted to the VRAM cache
    int64_t arena_experts() const;     ///< blobs the host arena holds
    int64_t file_tier() const;         ///< blobs the arena could not take
    double arena_gib() const;
    int64_t blob_bytes(int64_t layer) const;   ///< one expert's gate+up+down bytes in `layer` (0 if not routed)
    /// Whether `expert` of `layer` is resident in a VRAM slot right now (false in CPU-only mode).
    bool resident(int64_t layer, int64_t expert) const;
    /// Whether `expert` of `layer` is in the host arena or on the remote card - anywhere but the file tier (false in
    /// CPU-only mode and for the file tier).
    bool in_arena(int64_t layer, int64_t expert) const;
    /// Ds41MoeConfig::remote_device: whether the remote card holds `expert` of `layer`, and how many it holds.
    bool in_remote(int64_t layer, int64_t expert) const;
    int64_t remote_experts() const;
    double remote_gib() const;
    /// `nt` (1..4) tokens of one layer at once - draft verification: `ids`/`w` are nt*top_k (token-major), `x` and
    /// `out` nt*n_embd.  Each distinct expert is read/fetched once and applied to every token that routed to it.
    bool run_multi(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out);
    /// run_multi writing the nt * n_embd sums to device memory `out_dev` instead of the host (D3: the verify pass's
    /// per-token routed rows, one call per layer).  nt*top_k token-major ids/w, x nt*n_embd.  GPU tier only.
    bool run_multi_to_dev(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, void* out_dev);
    /// Measurement only (session 13): open a binary routing trace.  Writes a header snapshot (per (layer, expert):
    /// class, VRAM-resident, in arena), then one record per run_multi_to_dev call (the verify path): its distinct
    /// experts with the first pass row routing each and the class the call served it as.  `trace_mark` appends a
    /// caller record (the pass's acceptance).  Replayed by tools/v41/sim/route_sim.py.
    void trace_open(const std::string& path);
    void trace_mark(const float* v, int n);
    /// cudaProfilerStart / Stop (nsys --capture-range=cudaProfilerApi); a no-op without CUDA.
    static void profiler(bool on);
    /// A prompt chunk: `n` tokens (any count) of one layer, `ids`/`w` n*top_k token-major, `x`/`out` n*n_embd.  GPU
    /// tier: every expert on the card - resident from its slot, the rest streamed once per chunk (MiMo prefill;
    /// MIMO_CHUNK_STAGE_MIB sizes the two streaming halves, default 1024).  CPU tier: token by token.
    bool run_chunk(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out);
    /// run_chunk writing the n * n_embd sums to device memory `out_dev` (same device, e.g. the dense half's chunk
    /// tensor) instead of the host.  GPU tier only.
    bool run_chunk_dev(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev);
    /// The arena alone from a routing profile (the order `seed_from_routes` would give it, incl. arena_skip_resident),
    /// WITHOUT opening the VRAM cache - so prompt chunks can run first and `release_chunk` can give their buffers back
    /// before the cache takes the VRAM.  A later `seed_from_routes` keeps this arena and only seeds the cache.
    bool build_arena_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err);
    /// Frees run_chunk's device and pinned buffers (the next run_chunk allocates them again).
    void release_chunk();
    /// The conversation-adaptive arena (on by default; STRATA_ARENA_REBAL=0 turns it off) - routed-token counts per expert, a swap
    /// plan after each prompt (frequent file-tier experts in, rarely used arena experts out), executed by a reader thread
    /// in the idle NVMe time.  begin_prompt: before a prompt (pauses the reads, takes back every retired swap the reader
    /// has not started - the victim gets its slot back - and publishes finished ones); plan: after it (plans, resumes);
    /// tick with request_end at the end of a request (a large batch, so the gap until the next request is used), and
    /// every decode pass only with STRATA_ARENA_REBAL_DECODE=1 (s6: reads during decode cost it 5-19 %).  The prompt's
    /// logits do not depend on it (routing is exact).
    void rebalance_begin_prompt();
    void rebalance_plan();
    void rebalance_tick(bool request_end = false);
    /// Gives the layer-prestage region back (VRAM for the dense half when its pass buffer would not fit); prestage
    /// stays off until release_chunk.  False when there was none.
    bool release_prestage();
    /// Closes the VRAM expert cache and gives its device memory back (--serve: a multi-token prompt needs that VRAM
    /// for its pass / chunk buffers, as the one-shot deferred seed).  Every expert is then an arena / file-tier
    /// expert until the next seed_from_ranked, which reopens the cache (sized by set_slot_gib) and refills it; the
    /// arena is kept (an LRU-demoted expert stays in the arena slot it took).  No-op without a cache.
    void release_cache();
    /// Xolotl s6 (STRATA_SERVE_KEEP_CACHE=1; the cache opens segmented then): give back only the VRAM a small batch
    /// prompt needs instead of the whole cache.  Evicts the cache's TAIL slots (the lowest-ranked seeds) until
    /// `need_bytes` are free, remembering slot -> expert; returns the number evicted (0: it already fits), or -1 when
    /// the cache is not segmented (the caller then falls back to release_cache).
    int64_t shrink_cache_for(int64_t need_bytes, std::string& err);
    /// After that prompt: map the tail back and refill exactly the evicted experts into exactly their slots (from the
    /// RAM arena, else the file).  Returns the number refilled, or -1 with `err`.
    int64_t regrow_cache(std::string& err);
    static bool keep_cache_mode();
    /// REAP accumulators (Ds41MoeConfig::saliency), [layer * n_experts + expert]; empty when off.
    const std::vector<double>& saliency_sum() const;
    const std::vector<int64_t>& saliency_count() const;
    cpu::ExpertPool& pool();           ///< the CPU tier's worker pool (the gate's single-expert check uses it)

private:
    std::unique_ptr<Ds41MoeImpl> im_;
};

/// One expert through the CPU pool, in native format: `out` (n_embd floats) from `x`, using the tier's own
/// kernels.  Exposed because the gate needs the single-expert number as well as the weighted sum, and because it
/// is the only part of the CPU path that is worth testing on its own.
///
/// The SwiGLU is clamped per `f.swiglu_limit` (see the file header).
bool ds41_moe_cpu_expert(const cpu::NativeFmt& f, int64_t n_embd, int64_t n_ff, const uint8_t* blob,
                        const float* x, float* out, cpu::ExpertPool& pool);

}  // namespace strata::v41
