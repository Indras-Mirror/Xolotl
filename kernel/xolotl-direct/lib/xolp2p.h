// xolp2p.h - Xolotl direct NVMe -> VRAM reads (GPUDirect-Storage-class I/O on GeForce, Linux).
//
// Window: a CUDA VRAM region (VMM allocation) that the NVMe controller can DMA into.  It is exported to an RM object,
//   registered for third-party P2P through the RM 0x503C class (the step libcuda skips on GeForce), pinned to BAR1 by
//   /dev/p2pgate (HOLD) and handed to the kernel's PCI P2PDMA layer by /dev/xolp2p (MAP, mmap).  dma_target() is a
//   CPU-side mapping whose ONLY legal use is as the destination of an O_DIRECT read - never touch it with the CPU.
// RawFile: a model file read through its raw block device at its FIEMAP extents (ntfs3/exfat direct I/O cannot target
//   P2P pages; the raw block device can).  Read-only.
// Everything fails soft: open() returns null and a reason, and the engine keeps its host path.  Requires Linux >= 6.2,
// the p2pgate + xolp2p modules loaded (by hand), the NVIDIA driver version this was built against, a GPU + NVMe the
// kernel allows P2P between, and (for windows > ~190 MB) Resizable BAR.
#pragma once
#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xolp2p {

class Window {
public:
    // A window of `bytes` (rounded up to 2 MiB) on the CUDA context current on this thread.  Null + *why on failure.
    static std::unique_ptr<Window> open(size_t bytes, std::string* why);
    ~Window();
    CUdeviceptr dev() const { return dptr_; }        // where kernels read it
    uint8_t* dma_target() const { return win_; }     // O_DIRECT destination ONLY (never dereference)
    size_t size() const { return size_; }
    uint64_t bus() const { return bus_; }
private:
    Window() = default;
    CUmemGenericAllocationHandle memh_ = 0;
    CUdeviceptr dptr_ = 0;
    size_t size_ = 0;
    uint64_t bus_ = 0;
    uint8_t* win_ = nullptr;
    int fd_gpu_ = -1, fd_ctl_ = -1, fd_export_ = -1, fd_gate_ = -1, fd_xp_ = -1;
    uint32_t h_client_ = 0;
};

class RawFile {
public:
    // Null + *why when the file cannot be read raw (no FIEMAP, unaligned/encoded/inline extents, no raw device access).
    static std::unique_ptr<RawFile> open(const std::string& path, std::string* why);
    ~RawFile();
    // [off, off+len) of the file into dst.  off, len and dst must be 4 KiB aligned (O_DIRECT).  Splits at extents.
    bool read(uint64_t off, uint64_t len, uint8_t* dst, std::string* why) const;
    uint64_t file_size() const { return size_; }
    const std::string& device() const { return dev_; }
    size_t extents() const { return ex_.size(); }
private:
    struct Ext { uint64_t log, phys, len; };
    RawFile() = default;
    std::vector<Ext> ex_;
    uint64_t size_ = 0;
    int raw_ = -1;
    std::string dev_;
};

// Make NVMe writes into windows visible to later kernels (cuFlushGPUDirectRDMAWrites on the current context).
bool flush();

// One-shot self-test: a window of `bytes`, `bytes` of `path` at `off` read raw into it, compared with a plain read.
// true only if every byte matches.  Engines run it once at startup before trusting the path.
bool selftest(const std::string& path, uint64_t off, size_t bytes, std::string* report);

}  // namespace xolp2p
