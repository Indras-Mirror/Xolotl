// tools/v41/engine/ds41_sys.hpp - the DeepSeek-V4.1 engine's file and memory calls, portable.
//
// Linux: thin forwards to the POSIX calls the engine always used (open/pread/pwrite/close/fstat/fsync/
// posix_fadvise/posix_memalign/aligned_alloc/setenv, /proc/meminfo), so its behaviour there is unchanged
// (STRATA_ZFS_ARC=1 opts in to counting the reclaimable OpenZFS ARC as available, mem_available_bytes).
// Windows: the same contract on Win32.
//   - Files are opened with CreateFileW + FILE_FLAG_OVERLAPPED, so positional reads from many threads run in
//     parallel instead of serialising on one file pointer; O_DIRECT maps to FILE_FLAG_NO_BUFFERING, whose
//     alignment duty (offset, length and buffer address multiples of the sector size) is the one the engine's
//     O_DIRECT paths already keep (4 KiB).  The handle is wrapped in a CRT descriptor, so ints flow through the
//     engine as before and close() releases it.
//   - Offsets are 64-bit throughout (MSVC's off_t is 32-bit): call sites pass int64_t.
//   - posix_fadvise has no Win32 counterpart for an unmapped file and is a no-op.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <malloc.h>   // alloca
#  include <cerrno>
using ssize_t = std::ptrdiff_t;
#  ifndef O_DIRECT
#    define O_DIRECT 0x20000000
#  endif
#  ifndef O_CLOEXEC
#    define O_CLOEXEC 0
#  endif
#  ifndef POSIX_FADV_NORMAL
#    define POSIX_FADV_NORMAL 0
#    define POSIX_FADV_RANDOM 1
#    define POSIX_FADV_SEQUENTIAL 2
#    define POSIX_FADV_WILLNEED 3
#    define POSIX_FADV_DONTNEED 4
#  endif
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <alloca.h>
#endif

namespace ds41_sys {

/// What the engine reads from fstat: size, and identity (to tell whether two descriptors are the same file).
struct Stat {
    uint64_t st_size = 0;
    uint64_t st_ino = 0;
    uint64_t st_dev = 0;
};

#ifdef _WIN32

inline std::wstring widen(const char* utf8) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    std::wstring w(n > 0 ? (size_t) n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), n);
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

inline HANDLE handle_of(int fd) { return (HANDLE) _get_osfhandle(fd); }

inline int open(const char* path, int flags, int /*mode*/ = 0) {
    const int acc = flags & (_O_WRONLY | _O_RDWR);
    const DWORD access = acc == _O_WRONLY ? GENERIC_WRITE : acc == _O_RDWR ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    DWORD create = OPEN_EXISTING;
    if ((flags & _O_CREAT) && (flags & _O_TRUNC)) create = CREATE_ALWAYS;
    else if (flags & _O_CREAT) create = OPEN_ALWAYS;
    else if (flags & _O_TRUNC) create = TRUNCATE_EXISTING;
    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (flags & O_DIRECT) attrs |= FILE_FLAG_NO_BUFFERING;
    else if (access == GENERIC_READ) attrs |= FILE_FLAG_RANDOM_ACCESS;
    HANDLE h = CreateFileW(widen(path).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, create, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        errno = (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) ? ENOENT
              : (e == ERROR_ACCESS_DENIED || e == ERROR_SHARING_VIOLATION) ? EACCES : EINVAL;
        return -1;
    }
    const int fd = _open_osfhandle((intptr_t) h, (acc == _O_WRONLY ? _O_WRONLY : acc == _O_RDWR ? _O_RDWR : _O_RDONLY) | _O_BINARY);
    if (fd < 0) { CloseHandle(h); errno = EMFILE; return -1; }
    return fd;
}

inline int close(int fd) { return _close(fd); }

/// One overlapped transfer at `off`, waited on with a per-thread event.  Returns bytes moved, 0 at end of file,
/// -1 on error (errno set).  Transfers above 1 GiB are split; a short transfer ends the loop like POSIX does.
inline ssize_t xfer(int fd, void* buf, size_t n, int64_t off, bool write) {
    HANDLE h = handle_of(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    thread_local HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    size_t done = 0;
    while (done < n) {
        const DWORD want = (DWORD) ((n - done) > (size_t) (1u << 30) ? (1u << 30) : (n - done));
        OVERLAPPED ov{};
        const uint64_t at = (uint64_t) off + done;
        ov.Offset = (DWORD) (at & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (at >> 32);
        ov.hEvent = ev;
        ResetEvent(ev);
        DWORD got = 0;
        BOOL ok = write ? WriteFile(h, (const uint8_t*) buf + done, want, nullptr, &ov)
                        : ReadFile(h, (uint8_t*) buf + done, want, nullptr, &ov);
        if (!ok) {
            const DWORD e = GetLastError();
            if (e == ERROR_IO_PENDING) ok = GetOverlappedResult(h, &ov, &got, TRUE);
            else if (e == ERROR_HANDLE_EOF) break;
            if (!ok) {
                if (GetLastError() == ERROR_HANDLE_EOF) break;
                if (done) break;
                errno = EIO;
                return -1;
            }
        } else if (!GetOverlappedResult(h, &ov, &got, FALSE)) {
            if (done) break;
            errno = EIO;
            return -1;
        }
        done += got;
        if (got < want) break;
    }
    return (ssize_t) done;
}

inline ssize_t pread(int fd, void* buf, size_t n, int64_t off) { return xfer(fd, buf, n, off, false); }
inline ssize_t pwrite(int fd, const void* buf, size_t n, int64_t off) { return xfer(fd, const_cast<void*>(buf), n, off, true); }

inline int fstat(int fd, Stat* st) {
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(handle_of(fd), &info)) { errno = EBADF; return -1; }
    st->st_size = ((uint64_t) info.nFileSizeHigh << 32) | info.nFileSizeLow;
    st->st_ino = ((uint64_t) info.nFileIndexHigh << 32) | info.nFileIndexLow;
    st->st_dev = info.dwVolumeSerialNumber;
    return 0;
}

inline int fsync(int fd) { return FlushFileBuffers(handle_of(fd)) ? 0 : -1; }

inline int posix_fadvise(int, int64_t, int64_t, int) { return 0; }

inline int posix_memalign(void** p, size_t align, size_t n) {
    *p = _aligned_malloc(n ? n : 1, align);
    return *p ? 0 : ENOMEM;
}
inline void* aligned_alloc(size_t align, size_t n) { return _aligned_malloc(n ? n : 1, align); }
inline void aligned_free(void* p) { _aligned_free(p); }

inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name)) return 0;
    return _putenv_s(name, value);
}

/// Physical memory the OS can hand out now (the available physical memory).
inline uint64_t mem_available_bytes() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (uint64_t) m.ullAvailPhys : 0;
}

#else  // POSIX: the calls the engine always made

inline int open(const char* path, int flags, int mode = 0) { return ::open(path, flags, mode); }
inline int close(int fd) { return ::close(fd); }
inline ssize_t pread(int fd, void* buf, size_t n, int64_t off) { return ::pread(fd, buf, n, (off_t) off); }
inline ssize_t pwrite(int fd, const void* buf, size_t n, int64_t off) { return ::pwrite(fd, buf, n, (off_t) off); }
inline int fstat(int fd, Stat* st) {
    struct stat s;
    if (::fstat(fd, &s) != 0) return -1;
    st->st_size = (uint64_t) s.st_size;
    st->st_ino = (uint64_t) s.st_ino;
    st->st_dev = (uint64_t) s.st_dev;
    return 0;
}
inline int fsync(int fd) { return ::fsync(fd); }
inline int posix_fadvise(int fd, int64_t off, int64_t len, int advice) {
    return ::posix_fadvise(fd, (off_t) off, (off_t) len, advice);
}
inline int posix_memalign(void** p, size_t align, size_t n) { return ::posix_memalign(p, align, n); }
inline void* aligned_alloc(size_t align, size_t n) { return std::aligned_alloc(align, n); }
inline void aligned_free(void* p) { std::free(p); }
inline int setenv(const char* name, const char* value, int overwrite) { return ::setenv(name, value, overwrite); }
/// The OpenZFS ARC the kernel can take back now: its evictable buffers (the MRU / MFU / uncached lists; the ghost lists
/// hold no data), at most down to the ARC's floor c_min.  0, with `why` set, without /proc/spl/kstat/zfs/arcstats (no
/// ZFS module loaded, not Linux) or when its fields are not the ones OpenZFS 2.x writes.
inline uint64_t zfs_arc_reclaimable_bytes(const char** why = nullptr) {
    std::FILE* f = std::fopen("/proc/spl/kstat/zfs/arcstats", "r");
    if (!f) { if (why) *why = "no /proc/spl/kstat/zfs/arcstats (ZFS module not loaded)"; return 0; }
    char line[256], name[64];
    unsigned type = 0;
    unsigned long long v = 0, size = 0, c_min = 0, evictable = 0;
    int seen = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::sscanf(line, "%63s %u %llu", name, &type, &v) != 3) continue;
        if (std::strcmp(name, "size") == 0) { size = v; seen |= 1; }
        else if (std::strcmp(name, "c_min") == 0) { c_min = v; seen |= 2; }
        else if (std::strstr(name, "_evictable_") && std::strstr(name, "ghost") == nullptr && std::strncmp(name, "anon_", 5) != 0) {
            evictable += v;   // mru_ / mfu_ / uncached_ x _data / _metadata
            seen |= 4;
        }
    }
    std::fclose(f);
    if (seen != 7) { if (why) *why = "arcstats lacks size / c_min / *_evictable_* (not an OpenZFS 2.x layout)"; return 0; }
    if (why) *why = size <= c_min ? "the ARC is at its floor c_min" : nullptr;
    return size <= c_min ? 0 : std::min<uint64_t>(evictable, size - c_min);
}

/// STRATA_ZFS_ARC=1 (opt-in, OpenZFS on Linux): whether mem_available_bytes() adds the reclaimable ARC.
inline bool zfs_arc_credit_on() {
    const char* e = std::getenv("STRATA_ZFS_ARC");
    return e && e[0] == '1';
}

/// Physical memory the OS can hand out now: MemAvailable.  With STRATA_ZFS_ARC=1 plus the ARC's reclaimable part:
/// MemAvailable leaves the ARC out although the kernel's shrinker takes it back under pressure, so on a ZFS box the
/// arena check (and the chunk-prefetch budget) saw too little and the ARC had to be capped by hand (zfs_arc_max).
/// Without the variable, or without ZFS, this is MemAvailable as before.
inline uint64_t mem_available_bytes() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", (unsigned long long*) &kb) == 1) break;
    std::fclose(f);
    const uint64_t avail = kb * 1024;
    if (avail == 0 || !zfs_arc_credit_on()) return avail;
    const char* why = nullptr;
    const uint64_t arc = zfs_arc_reclaimable_bytes(&why);
    std::fprintf(stderr, "STRATA_ZFS_ARC: MemAvailable %.1f GiB + reclaimable ZFS ARC %.1f GiB%s%s\n", avail / 1073741824.0,
                 arc / 1073741824.0, why ? " - " : "", why ? why : "");
    // OpenZFS 2.2 / 2.3 default zfs_arc_shrinker_limit=10000 (pages per shrinker call): the ARC then gives memory back
    // slowly, and pinning an arena sized from it can stall in reclaim.  2.4 defaults to 0 (no limit).
    static std::atomic<bool> warned{ false };
    if (std::FILE* s = std::fopen("/sys/module/zfs/parameters/zfs_arc_shrinker_limit", "r")) {
        long long lim = 0;
        if (std::fscanf(s, "%lld", &lim) == 1 && lim > 0 && arc > 0 && !warned.exchange(true))
            std::fprintf(stderr, "STRATA_ZFS_ARC: zfs_arc_shrinker_limit=%lld slows the ARC's shrinking; set it to 0, or cap "
                                 "zfs_arc_max instead\n", lim);
        std::fclose(s);
    }
    return avail + arc;
}

#endif

}  // namespace ds41_sys
