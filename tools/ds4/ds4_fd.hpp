// tools/ds4/ds4_fd.hpp - the fd API ds4_moe.cpp reads the shards with (open, close, pread), on Windows.
//
// They have names of their own, because `open` and `close` are member functions of classes in ds4_moe.cpp with other
// arities and MSVC expands a function-like macro after `::` as well: `#define open _open` cannot be used there.  On
// POSIX each name is the libc call.
//
// pread's contract is what a port has to keep: the offset travels with the request, so a read never depends on the
// file pointer.  The arena fill reads the same fds from 8 loader threads at once.  A seek + read pair races there (the
// first Windows port did that, and the expert bytes arrived scrambled); POSIX pread does not.  tools/ds4/pread_test.cpp
// checks both.
//
// One difference remains.  POSIX pread leaves the file pointer where it was; Windows' ReadFile moves it to the end of
// what it read, because the CRT fd's handle is a synchronous one.  Nothing in ds4_moe.cpp reads through that pointer.
//
// Offsets are int64_t throughout: MSVC's off_t is 32-bit and the shards run past 2 GiB.  The call sites cast to int64_t.
// A macro that renamed off_t would break the next header to declare it (<sys/types.h> does).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>

#if defined(_WIN32)
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX                // windows.h's min/max macros break std::min and std::max in what includes this
#endif
#include <windows.h>            // ReadFile, OVERLAPPED

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef long long ssize_t;      // POSIX ssize_t: windows.h declares it in basetsd.h; a TU without windows.h gets it here
#endif
#define O_CLOEXEC 0             // a Linux nicety; handles are not inherited unless they are made so
#define O_DIRECT 0              // no CRT flag for unbuffered I/O: buffered reads are as correct, only cached
// posix_memalign: what it returns goes to ds4_free, not to std::free.
#define posix_memalign(memptr, alignment, size) (*(memptr) = _aligned_malloc((size), (alignment)), *(memptr) ? 0 : ENOMEM)
#else
#include <unistd.h>
#endif

/// Frees what posix_memalign gave: _aligned_free on Windows (the pair of _aligned_malloc), free elsewhere.
static inline void ds4_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

static inline int ds4_open(const char* path, int flags) {
#if defined(_WIN32)
    return _open(path, (flags & ~O_CLOEXEC & ~O_DIRECT) | _O_BINARY);
#else
    return ::open(path, flags);
#endif
}

static inline int ds4_close(int fd) {
#if defined(_WIN32)
    return _close(fd);
#else
    return ::close(fd);
#endif
}

/// Up to n bytes at offset off into buf, whichever way the file pointer points.  The count read, 0 at the end of the
/// file, -1 on an error.
static inline ssize_t ds4_pread(int fd, void* buf, size_t n, int64_t off) {
    if (n == 0) return 0;
#if defined(_WIN32)
    // Windows' positional read: ReadFile with the offset in the OVERLAPPED.  The handle is the CRT fd's and
    // synchronous, so the call returns with the bytes in buf (and leaves the file pointer at off + the count read).
    // The Strata core's expert source reads its spans this way too (src/core/expert_source.cpp).
    OVERLAPPED ov{};
    ov.Offset = (DWORD) ((uint64_t) off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
    DWORD got = 0;
    if (!ReadFile((HANDLE) _get_osfhandle(fd), buf, (DWORD) (n > 0xFFFFFFFF ? 0xFFFFFFFF : n), &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return (ssize_t) got;   // the end of the file
        return -1;
    }
    return (ssize_t) got;
#else
    return ::pread(fd, buf, n, (off_t) off);
#endif
}
