// xolp2p.cpp - see xolp2p.h.  The RM registration recipe is ~/AI/gds-research/p2pgate/importgate.cu's (verified
// 2026-11-07: Step C, 3 x 64 MB of the DS4.1 gguf read NVMe -> VRAM at 6.31 GB/s, every byte identical).
#include "xolp2p.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <linux/fs.h>
#include <linux/fiemap.h>

#include <nvtypes.h>
#include <nvos.h>
#include <class/cl0000.h>
#include <class/cl0080.h>
#include <class/cl2080.h>
#include <class/cl503c.h>
#include <ctrl/ctrl503c.h>
#include <ctrl/ctrl0000/ctrl0000unix.h>

#ifndef XOLP2P_NV_VERSION
#error "build with -DXOLP2P_NV_VERSION=\"<the driver version whose SDK headers are on the include path>\""
#endif

namespace xolp2p {
namespace {

// the two kernel helpers (~/AI/gds-research/p2pgate, ~/AI/gds-research/xolp2p) - keep in sync with their sources
struct HoldReq { unsigned long long key, len; int status; unsigned int entries, page_size, contiguous; unsigned long long pa0, pa_last; };
#define P2PGATE_IOC_HOLD _IOWR('p', 3, struct HoldReq)
struct MapReq { unsigned long long bus, len; int status; unsigned int pages; unsigned long long bar_start, bar_len; };
#define XOLP2P_MAP _IOWR('x', 2, struct MapReq)

#define NV_IOCTL_MAGIC  'F'
#define NV_ESC_RM_ALLOC   0x2B
#define NV_ESC_RM_CONTROL 0x2A

int rm_alloc(int ctl, NvHandle root, NvHandle parent, NvV32 cls, void* params, NvU32 size, NvHandle* out) {
    NVOS64_PARAMETERS p; memset(&p, 0, sizeof p);
    p.hRoot = root; p.hObjectParent = parent; p.hClass = cls;
    p.pAllocParms = (NvP64) (uintptr_t) params; p.paramsSize = size;
    if (ioctl(ctl, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS64_PARAMETERS), &p) < 0) return -1;
    if (out) *out = p.hObjectNew;
    return (int) p.status;
}
int rm_control(int ctl, NvHandle client, NvHandle object, NvU32 cmd, void* params, NvU32 size) {
    NVOS54_PARAMETERS p; memset(&p, 0, sizeof p);
    p.hClient = client; p.hObject = object; p.cmd = cmd;
    p.params = (NvP64) (uintptr_t) params; p.paramsSize = size;
    if (ioctl(ctl, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &p) < 0) return -1;
    return (int) p.status;
}
bool fail(std::string* why, const std::string& s) { if (why) *why = s; return false; }
std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

// the RM ioctl ABI is version-locked: refuse unless the loaded driver is the one whose headers we were built with
bool driver_matches(std::string* why) {
    const std::string v = slurp("/proc/driver/nvidia/version");
    if (v.find(XOLP2P_NV_VERSION) == std::string::npos)
        return fail(why, std::string("NVIDIA driver is not ") + XOLP2P_NV_VERSION + " (rebuild xolp2p against the installed driver's SDK headers)");
    return true;
}
// /dev/nvidiaN minor of the CUDA device (from its PCI bus id)
int device_minor(CUdevice dev) {
    char bus[32] = {0};
    if (cuDeviceGetPCIBusId(bus, sizeof bus, dev) != CUDA_SUCCESS) return -1;
    std::string b(bus);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    if (b.size() > 12) b = b.substr(b.size() - 12);   // cuda prints 8 domain digits, /proc uses 4
    const std::string info = slurp("/proc/driver/nvidia/gpus/" + b + "/information");
    const size_t at = info.find("Device Minor:");
    return at == std::string::npos ? -1 : atoi(info.c_str() + at + 13);
}
std::atomic<uint64_t> g_key{0};

}  // namespace

std::unique_ptr<Window> Window::open(size_t bytes, std::string* why) {
    if (!driver_matches(why)) return nullptr;
    std::unique_ptr<Window> w(new Window());
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) { fail(why, "no current CUDA context"); return nullptr; }
    const int minor = device_minor(dev);
    if (minor < 0) { fail(why, "cannot map the CUDA device to /dev/nvidiaN"); return nullptr; }
    // ---- the CUDA side: an exportable VMM allocation, mapped for kernels ----
    CUmemAllocationProp ap; memset(&ap, 0, sizeof ap);
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ap.location.id = (int) dev;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || !gran) gran = 2u << 20;
    w->size_ = (bytes + gran - 1) / gran * gran;
    if (cuMemCreate(&w->memh_, w->size_, &ap, 0) != CUDA_SUCCESS) { w->memh_ = 0; fail(why, "cuMemCreate (exportable VRAM) failed"); return nullptr; }
    if (cuMemAddressReserve(&w->dptr_, w->size_, 0, 0, 0) != CUDA_SUCCESS) { w->dptr_ = 0; fail(why, "cuMemAddressReserve failed"); return nullptr; }
    if (cuMemMap(w->dptr_, w->size_, 0, w->memh_, 0) != CUDA_SUCCESS) { fail(why, "cuMemMap failed"); return nullptr; }
    CUmemAccessDesc ad; memset(&ad, 0, sizeof ad);
    ad.location = ap.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(w->dptr_, w->size_, &ad, 1) != CUDA_SUCCESS) { fail(why, "cuMemSetAccess failed"); return nullptr; }
    if (cuMemExportToShareableHandle((void*) &w->fd_export_, w->memh_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) != CUDA_SUCCESS) {
        w->fd_export_ = -1; fail(why, "cuMemExportToShareableHandle failed"); return nullptr; }
    // ---- RM: our own client imports the allocation and registers it for third-party P2P ----
    w->fd_gpu_ = ::open(("/dev/nvidia" + std::to_string(minor)).c_str(), O_RDWR | O_CLOEXEC);   // RM requires it open
    w->fd_ctl_ = ::open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (w->fd_gpu_ < 0 || w->fd_ctl_ < 0) { fail(why, "cannot open /dev/nvidiactl or /dev/nvidiaN"); return nullptr; }
    NvHandle hDevice = 0, hSub = 0, hP2P = 0, hClient = 0;
    if (rm_alloc(w->fd_ctl_, 0, 0, NV01_ROOT, nullptr, 0, &hClient) != 0) { fail(why, "RM client alloc failed"); return nullptr; }
    w->h_client_ = hClient;
    NV0080_ALLOC_PARAMETERS dp; memset(&dp, 0, sizeof dp); dp.deviceId = (NvU32) minor;
    NV2080_ALLOC_PARAMETERS sp; memset(&sp, 0, sizeof sp);
    if (rm_alloc(w->fd_ctl_, hClient, hClient, NV01_DEVICE_0, &dp, sizeof dp, &hDevice) != 0 ||
        rm_alloc(w->fd_ctl_, hClient, hDevice, NV20_SUBDEVICE_0, &sp, sizeof sp, &hSub) != 0) { fail(why, "RM device/subdevice alloc failed"); return nullptr; }
    NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS ip; memset(&ip, 0, sizeof ip);
    ip.fd = w->fd_export_;
    ip.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
    ip.object.data.rmObject.hDevice = hDevice; ip.object.data.rmObject.hParent = hDevice;
    int st = rm_control(w->fd_ctl_, hClient, hClient, NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD, &ip, sizeof ip);
    if (st) { fail(why, "RM import of the CUDA allocation failed (status " + std::to_string(st) + ")"); return nullptr; }
    NV503C_ALLOC_PARAMETERS tp; memset(&tp, 0, sizeof tp); tp.flags = NV503C_ALLOC_PARAMETERS_FLAGS_TYPE_BAR1;
    if (rm_alloc(w->fd_ctl_, hClient, hSub, NV50_THIRD_PARTY_P2P, &tp, sizeof tp, &hP2P) != 0) { fail(why, "RM third-party P2P object failed"); return nullptr; }
    const unsigned long long key = 0x7f0000000000ull + (g_key.fetch_add(1) << 36);   // a unique registration key per window
    NV503C_CTRL_REGISTER_VA_SPACE_PARAMS rv; memset(&rv, 0, sizeof rv);
    NV503C_CTRL_REGISTER_VIDMEM_PARAMS rmv; memset(&rmv, 0, sizeof rmv);
    rmv.hMemory = ip.object.data.rmObject.hObject; rmv.address = key; rmv.size = w->size_; rmv.offset = 0;
    NV503C_CTRL_REGISTER_PID_PARAMS rp; memset(&rp, 0, sizeof rp); rp.hClient = hClient;
    if ((st = rm_control(w->fd_ctl_, hClient, hP2P, NV503C_CTRL_CMD_REGISTER_VA_SPACE, &rv, sizeof rv)) ||
        (st = rm_control(w->fd_ctl_, hClient, hP2P, NV503C_CTRL_CMD_REGISTER_VIDMEM, &rmv, sizeof rmv)) ||
        (st = rm_control(w->fd_ctl_, hClient, hP2P, NV503C_CTRL_CMD_REGISTER_PID, &rp, sizeof rp))) {
        fail(why, "RM P2P registration failed (status " + std::to_string(st) + ")"); return nullptr; }
    // ---- kernel: pin to BAR1 (p2pgate HOLD), hand to P2PDMA and map (xolp2p) ----
    w->fd_gate_ = ::open("/dev/p2pgate", O_RDWR | O_CLOEXEC);
    w->fd_xp_ = ::open("/dev/xolp2p", O_RDWR | O_CLOEXEC);
    if (w->fd_gate_ < 0 || w->fd_xp_ < 0) { fail(why, "p2pgate/xolp2p modules not loaded (or no access: udev rule)"); return nullptr; }
    HoldReq h; memset(&h, 0, sizeof h); h.key = key; h.len = w->size_;
    if (ioctl(w->fd_gate_, P2PGATE_IOC_HOLD, &h) < 0 || h.status) { fail(why, "BAR1 pin (HOLD) failed: " + std::to_string(h.status)); return nullptr; }
    if (!h.contiguous) { fail(why, "BAR1 pages not contiguous"); return nullptr; }
    w->bus_ = h.pa0;
    MapReq m; memset(&m, 0, sizeof m); m.bus = h.pa0; m.len = w->size_;
    if (ioctl(w->fd_xp_, XOLP2P_MAP, &m) < 0) { fail(why, std::string("P2PDMA map failed: ") + strerror(errno)); return nullptr; }
    void* p = mmap(nullptr, w->size_, PROT_READ | PROT_WRITE, MAP_SHARED, w->fd_xp_, 0);
    if (p == MAP_FAILED) { fail(why, std::string("window mmap failed: ") + strerror(errno)); return nullptr; }
    w->win_ = (uint8_t*) p;
    return w;
}

Window::~Window() {
    // reverse order: the mapping, the P2PDMA pages, the BAR1 pin, the RM registration, then the CUDA allocation
    if (win_) munmap(win_, size_);
    if (fd_xp_ >= 0) ::close(fd_xp_);
    if (fd_gate_ >= 0) ::close(fd_gate_);
    if (fd_ctl_ >= 0) ::close(fd_ctl_);
    if (fd_gpu_ >= 0) ::close(fd_gpu_);
    if (fd_export_ >= 0) ::close(fd_export_);
    if (dptr_) { cuMemUnmap(dptr_, size_); cuMemAddressFree(dptr_, size_); }
    if (memh_) cuMemRelease(memh_);
}

std::unique_ptr<RawFile> RawFile::open(const std::string& path, std::string* why) {
    std::unique_ptr<RawFile> r(new RawFile());
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { fail(why, "open " + path + ": " + strerror(errno)); return nullptr; }
    struct stat sb;
    if (fstat(fd, &sb) < 0) { ::close(fd); fail(why, "fstat failed"); return nullptr; }
    r->size_ = (uint64_t) sb.st_size;
    // the extent map: every extent must be plain, 4 KiB aligned and on disk, and together cover the file with no holes
    const uint32_t bad = FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_ENCRYPTED |
                         FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DATA_INLINE | FIEMAP_EXTENT_DATA_TAIL | FIEMAP_EXTENT_UNWRITTEN;
    uint64_t start = 0, covered = 0;
    for (bool last = false; !last;) {
        const int N = 512;
        std::vector<char> buf(sizeof(struct fiemap) + N * sizeof(struct fiemap_extent));
        struct fiemap* fm = (struct fiemap*) buf.data();
        fm->fm_start = start; fm->fm_length = ~0ull; fm->fm_flags = 0; fm->fm_extent_count = N;
        if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) { ::close(fd); fail(why, std::string("FIEMAP unsupported: ") + strerror(errno)); return nullptr; }
        if (!fm->fm_mapped_extents) break;
        for (unsigned i = 0; i < fm->fm_mapped_extents; ++i) {
            const struct fiemap_extent& e = fm->fm_extents[i];
            if ((e.fe_flags & bad) || (e.fe_logical | e.fe_physical | e.fe_length) & 4095) {
                ::close(fd); fail(why, "an extent is not a plain aligned on-disk extent (flags 0x" + std::to_string(e.fe_flags) + ")"); return nullptr; }
            if (e.fe_logical != covered) { ::close(fd); fail(why, "the file has a hole at " + std::to_string(covered)); return nullptr; }
            r->ex_.push_back({e.fe_logical, e.fe_physical, e.fe_length});
            covered = e.fe_logical + e.fe_length;
            if (e.fe_flags & FIEMAP_EXTENT_LAST) last = true;
        }
        start = covered;
    }
    ::close(fd);
    if (covered < r->size_) { fail(why, "extents cover less than the file"); return nullptr; }
    // the raw block device the file lives on (a partition: FIEMAP physical offsets are partition-relative)
    char sys[64]; snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u/uevent", major(sb.st_dev), minor(sb.st_dev));
    const std::string ue = slurp(sys);
    const size_t at = ue.find("DEVNAME=");
    if (at == std::string::npos) { fail(why, std::string("no block device for ") + sys); return nullptr; }
    r->dev_ = "/dev/" + ue.substr(at + 8, ue.find('\n', at) - at - 8);
    r->raw_ = ::open(r->dev_.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);   // read-only, always
    if (r->raw_ < 0) { fail(why, "open " + r->dev_ + " read-only: " + strerror(errno) + " (group disk?)"); return nullptr; }
    return r;
}

RawFile::~RawFile() { if (raw_ >= 0) ::close(raw_); }

bool RawFile::read(uint64_t off, uint64_t len, uint8_t* dst, std::string* why) const {
    if ((off | len | (uintptr_t) dst) & 4095) return fail(why, "unaligned raw read");
    if (off + len > (size_ + 4095) / 4096 * 4096) return fail(why, "raw read past the end of the file");
    size_t i = std::upper_bound(ex_.begin(), ex_.end(), off, [](uint64_t o, const Ext& e) { return o < e.log; }) - ex_.begin();
    if (i) --i;
    uint64_t done = 0;
    while (done < len && i < ex_.size()) {
        const Ext& e = ex_[i];
        const uint64_t a = off + done, in = a - e.log, n = std::min(len - done, e.len - in);
        const ssize_t got = pread(raw_, dst + done, n, (off_t) (e.phys + in));
        if (got != (ssize_t) n) return fail(why, "raw pread " + std::to_string(n) + " B at " + std::to_string(e.phys + in) + ": " +
                                            (got < 0 ? strerror(errno) : "short read"));
        done += n; ++i;
    }
    return done == len || fail(why, "raw read ran out of extents");
}

bool flush() {
    return cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                      CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) == CUDA_SUCCESS;
}

bool selftest(const std::string& path, uint64_t off, size_t bytes, std::string* report) {
    std::string why;
    auto w = Window::open(bytes, &why);
    if (!w) return fail(report, "window: " + why);
    auto rf = RawFile::open(path, &why);
    if (!rf) return fail(report, "raw file: " + why);
    off &= ~4095ull; bytes &= ~4095ull;
    if (off + bytes > rf->file_size()) return fail(report, "selftest slice past the end of the file");
    std::vector<uint8_t> ref(bytes), back(bytes);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    const bool ok_ref = fd >= 0 && pread(fd, ref.data(), bytes, (off_t) off) == (ssize_t) bytes;
    if (fd >= 0) ::close(fd);
    if (!ok_ref) return fail(report, "reference read failed");
    if (cuMemsetD8(w->dev(), 0xA5, w->size()) != CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS) return fail(report, "memset failed");
    if (!rf->read(off, bytes, w->dma_target(), &why)) return fail(report, "raw read: " + why);
    if (!flush() || cuCtxSynchronize() != CUDA_SUCCESS || cuMemcpyDtoH(back.data(), w->dev(), bytes) != CUDA_SUCCESS)
        return fail(report, "flush/copy-back failed");
    size_t bad = 0;
    for (size_t i = 0; i < bytes; ++i) bad += back[i] != ref[i];
    if (report) *report = std::to_string(bytes) + " B of " + path + " @" + std::to_string(off) + " via " + rf->device() +
                          ": " + std::to_string(bad) + " bytes differ";
    return bad == 0;
}

}  // namespace xolp2p
