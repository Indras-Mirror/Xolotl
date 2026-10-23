// tools/v41/engine/ds41_remote.hpp - routed experts on a second GPU in a helper process (STRATA_DS41_REMOTE, opt-in).
//
// Why a process: on Windows (WDDM) a CUDA context on a second card caps the host memory one process can pin at
// ~56 GiB and breaks that card's context past it (measured: RTX 5090 + RTX 4080 over OCuLink, 128 GB).  The engine
// pins a 72 GiB arena, so the second card's work runs in ds41_helper, a process of its own; the cap is per process
// (a 5090-only process pinned 80 GiB while the helper held and rewrote 14 GiB on the 4080, no failures).
//
// The helper loads a set of experts into its card's VRAM (the next stretch of the routing profile after what the
// engine's card holds) and computes them for the engine, one request per layer call:
//   engine: fills x (nt rows of n_embd floats) and the entries (expert, token, output row), bumps `req`
//   helper: q8_1-quantizes x, runs native_expert_grouped per format class (the kernel and activations the engine's
//           card uses), writes one n_embd row per entry into `out`, sets `done = req`
// Round trip measured 100-130 us (5 tokens, 2-6 experts) through this layout, while a layer's call on the engine's
// card takes ~1.2 ms: the engine posts before its own launches and collects after them.
//
// One request in flight; the engine side is single-threaded (the tier's calls are sequential).
//
// Prompt chunks (run_chunk) use a second, larger segment `<name>_chunk` (Ds41RemoteChunk): the same exchange for up to
// `cap_tok` rows and `cap_ent` entries, so the helper's experts are not read from the file tier during a prompt.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::v41 {

struct Ds41RemoteShm {
    static constexpr uint32_t kMagic = 0x44533452;   // "DS4R"
    static constexpr uint32_t kVersion = 1;
    static constexpr int kMaxTok = 16;                // = Ds41MoeTier::kMaxMulti
    static constexpr int kMaxEnt = kMaxTok * 8;       // = ds41_moe.cpp's kMaxEnt
    static constexpr int64_t kMaxEmbd = 8192;
    static constexpr int64_t kMaxPairs = 128 * 1024;  // n_layers * n_experts

    uint32_t magic, version;
    int64_t n_layers, n_experts, n_embd;
    std::atomic<int32_t> ready;     // helper: 1 = loaded and serving, -1 = failed
    std::atomic<int32_t> stop;      // engine: 1 = the helper exits
    std::atomic<int64_t> req;       // engine: request number
    std::atomic<int64_t> done;      // helper: the last request finished
    std::atomic<int32_t> status;    // helper: 0 = ok, else the request failed (rows not valid)
    int32_t layer, nt, n_ent;
    int64_t held, held_bytes;       // helper: experts / bytes it holds
    int32_t ent_expert[kMaxEnt], ent_tok[kMaxEnt];
    alignas(64) uint8_t holds[kMaxPairs];             // [layer * n_experts + expert] != 0: the helper has it
    alignas(4096) float x[kMaxTok * kMaxEmbd];
    alignas(4096) float out[kMaxEnt * kMaxEmbd];      // row i = entry i
};
static_assert(std::atomic<int64_t>::is_always_lock_free, "shared-memory atomics must be lock-free");

/// The prompt-chunk exchange: header, then ent_expert[cap_ent], ent_tok[cap_ent], x[cap_tok * n_embd],
/// out[cap_ent * n_embd] at the offsets below (sized by the helper's --chunk-tok).
struct Ds41RemoteChunk {
    static constexpr uint32_t kMagic = 0x44533443;   // "DS4C"
    uint32_t magic;
    int64_t bytes, cap_tok, cap_ent, n_embd;
    std::atomic<int64_t> req, done;
    std::atomic<int32_t> status;
    int32_t layer, n, n_ent;
    static constexpr size_t kHead = 4096;
    static size_t size_for(int64_t cap_tok, int64_t cap_ent, int64_t n_embd) {
        return kHead + (size_t) cap_ent * 8 + 4096 + (size_t) (cap_tok + cap_ent) * (size_t) n_embd * 4;
    }
    int32_t* ent_expert() { return (int32_t*) ((uint8_t*) this + kHead); }
    int32_t* ent_tok() { return ent_expert() + cap_ent; }
    float* x() { return (float*) ((uint8_t*) this + ((kHead + (size_t) cap_ent * 8 + 4095) & ~(size_t) 4095)); }
    float* out() { return x() + (size_t) cap_tok * (size_t) n_embd; }
};

/// The mapping of one named segment (the helper creates it, the engine opens it).
class Ds41RemoteMap {
public:
    Ds41RemoteMap() = default;
    Ds41RemoteMap(const Ds41RemoteMap&) = delete;
    Ds41RemoteMap& operator=(const Ds41RemoteMap&) = delete;
    ~Ds41RemoteMap() { close(); }

    /// `bytes`: the size to create; 0 when opening = the whole segment (its creator's size)
    bool open(const std::string& name, bool create, std::string& err, size_t bytes = sizeof(Ds41RemoteShm)) {
        size_t n = bytes;
#ifdef _WIN32
        const std::string full = "Local\\" + name;
        h_ = create ? CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD) ((uint64_t) n >> 32),
                                         (DWORD) n, full.c_str())
                    : OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, full.c_str());
        if (!h_) { err = "shared memory " + full + ": error " + std::to_string(GetLastError()); return false; }
        p_ = MapViewOfFile(h_, FILE_MAP_ALL_ACCESS, 0, 0, create ? n : bytes);
        if (!p_) { err = "map " + full + ": error " + std::to_string(GetLastError()); close(); return false; }
#else
        name_ = "/" + name;
        create_ = create;
        const int fd = create ? shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600) : shm_open(name_.c_str(), O_RDWR, 0);
        if (fd < 0) { err = "shm_open " + name_ + " failed"; return false; }
        if (create && ftruncate(fd, (off_t) n) != 0) { ::close(fd); err = "ftruncate " + name_ + " failed"; return false; }
        struct stat sb;
        if (!create && n == 0) n = fstat(fd, &sb) == 0 ? (size_t) sb.st_size : 0;
        void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) { err = "mmap " + name_ + " failed"; return false; }
        p_ = p;
        n_ = n;
#endif
        return true;
    }
    void close() {
#ifdef _WIN32
        if (p_) UnmapViewOfFile(p_);
        if (h_) CloseHandle(h_);
        h_ = nullptr;
#else
        if (p_) munmap(p_, n_);
        if (create_ && !name_.empty()) shm_unlink(name_.c_str());
        name_.clear();
#endif
        p_ = nullptr;
    }
    Ds41RemoteShm* shm() const { return (Ds41RemoteShm*) p_; }
    void* data() const { return p_; }

private:
    void* p_ = nullptr;
#ifdef _WIN32
    HANDLE h_ = nullptr;
#else
    std::string name_;
    bool create_ = false;
    size_t n_ = 0;
#endif
};

inline void ds41_remote_pause() {
#ifdef _WIN32
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

}  // namespace strata::v41
