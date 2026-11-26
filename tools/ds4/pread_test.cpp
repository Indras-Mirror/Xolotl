// tools/ds4/pread_test.cpp - ds4_fd.hpp's open, pread and close against seek + read, on a file that runs past 40 GiB.
//
// No shard needed.  The test makes a sparse file of 48 GiB under %TEMP% (or /tmp), writes 256 blocks of 1 MiB into it
// at fixed offsets (256 MiB of real data, some across 2 GiB and 4 GiB), and reads them back:
//   1. positional: every block through ds4_pread, byte for byte the block that was written, whichever way the file
//      pointer points (POSIX pread leaves the pointer alone; Windows' ReadFile moves it, which is reported, not
//      required);
//   2. concurrent: 8 threads on one fd, 2000 reads each of a block or a slice of one, every byte compared.  The same
//      reads through a seek + read pread are the control: that is the race a port must not have;
//   3. the ends: the last 4096 bytes of the file, a read at the very end, a read of 0 bytes.
// Exit 0 when every byte matched, 77 when the file system has no sparse files (nothing was run).
// POSIX runs it too: there it checks libc's pread, and this test.
#include "ds4_fd.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <winioctl.h>            // FSCTL_SET_SPARSE
#else
#include <unistd.h>
#endif

namespace {

constexpr size_t kBlock = size_t(1) << 20;            // 1 MiB
constexpr int kBlocks = 256;
constexpr uint64_t kFileEnd = (uint64_t) 48 << 30;    // the file's length: past 40 GiB, so offsets need 64 bits
constexpr int kThreads = 8;
constexpr int kReadsPerThread = 2000;

// The bytes of block b from `from` on: the same every time (xorshift32, seeded with b).
void fill(int b, size_t from, size_t n, uint8_t* dst) {
    uint32_t s = 0x811C9DC5u ^ (uint32_t) b * 2654435761u;
    const size_t upto = from + n;
    for (size_t at = 0; at < upto; at += 4) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        for (int k = 0; k < 4 && at + k < upto; ++k)
            if (at + k >= from) dst[at + k - from] = (uint8_t) (s >> (8 * k));
    }
}

std::vector<uint64_t> block_offsets() {
    const uint64_t fixed[] = { 0, kBlock + 4096,                      // the first two do not share bytes
                               ((uint64_t) 1 << 31) - kBlock / 2,     // across 2 GiB
                               ((uint64_t) 1 << 32) - kBlock / 2,     // across 4 GiB
                               ((uint64_t) 5 << 30) + 123, ((uint64_t) 20 << 30) + 999,
                               ((uint64_t) 40 << 30) + 32157 };
    std::vector<uint64_t> v(fixed, fixed + sizeof fixed / sizeof fixed[0]);
    for (int b = (int) v.size(); b < kBlocks; ++b) {
        uint64_t x = (uint64_t) b * 0x9E3779B97F4A7C15ull + 0x123456789ABCDEF0ull;   // scattered by a fixed sequence
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        uint64_t at = x % (kFileEnd - kBlock);
        for (size_t push = 0; push < v.size(); ++push) {          // clear of every block taken so far
            bool clear = true;
            for (uint64_t o : v)
                if (at + kBlock > o && o + kBlock > at) { clear = false; break; }
            if (clear) break;
            at = (at + kBlock) % (kFileEnd - kBlock);
        }
        v.push_back(at);
    }
    return v;
}

bool make_file(const std::string& path, const std::vector<uint64_t>& off, std::string& why) {
    std::vector<uint8_t> buf(kBlock);
#if defined(_WIN32)
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE | GENERIC_READ, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { why = "CreateFile, error " + std::to_string(GetLastError()); return false; }
    DWORD ret = 0;
    if (!DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ret, nullptr)) {
        why = "FSCTL_SET_SPARSE, error " + std::to_string(GetLastError())
              + ": the temp directory's file system has no sparse files";
        CloseHandle(h);
        DeleteFileA(path.c_str());
        return false;
    }
    LARGE_INTEGER end{};
    end.QuadPart = (LONGLONG) kFileEnd;
    bool ok = SetFilePointerEx(h, end, nullptr, FILE_BEGIN) && SetEndOfFile(h);
    for (int b = 0; ok && b < kBlocks; ++b) {
        fill(b, 0, kBlock, buf.data());
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off[b] & 0xFFFFFFFFu);
        ov.OffsetHigh = (DWORD) (off[b] >> 32);
        DWORD wrote = 0;
        ok = WriteFile(h, buf.data(), (DWORD) kBlock, &wrote, &ov) && wrote == kBlock;
    }
    if (!ok) why = "writing the blocks, error " + std::to_string(GetLastError());
    CloseHandle(h);
    return ok;
#else
    const int wfd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (wfd < 0) { why = "the file could not be created"; return false; }
    bool ok = ftruncate(wfd, (off_t) kFileEnd) == 0;
    for (int b = 0; ok && b < kBlocks; ++b) {
        fill(b, 0, kBlock, buf.data());
        ok = ::pwrite(wfd, buf.data(), kBlock, (off_t) off[b]) == (ssize_t) kBlock;
    }
    if (!ok) why = "writing the blocks failed";
    ::close(wfd);
    return ok;
#endif
}

void remove_file(const std::string& path) {
#if defined(_WIN32)
    DeleteFileA(path.c_str());
#else
    ::unlink(path.c_str());
#endif
}

int64_t pointer_pos(int fd) {
#if defined(_WIN32)
    LARGE_INTEGER at{};
    return SetFilePointerEx((HANDLE) _get_osfhandle(fd), LARGE_INTEGER{}, &at, FILE_CURRENT) ? at.QuadPart : -1;
#else
    return (int64_t) lseek(fd, 0, SEEK_CUR);
#endif
}

bool seek_to(int fd, uint64_t off) {
#if defined(_WIN32)
    return _lseeki64(fd, (int64_t) off, SEEK_SET) == (int64_t) off;
#else
    return lseek(fd, (off_t) off, SEEK_SET) == (off_t) off;
#endif
}

ssize_t read_here(int fd, uint8_t* dst, size_t n) {
#if defined(_WIN32)
    return (ssize_t) _read(fd, dst, (unsigned) (n > (size_t(1) << 30) ? (size_t(1) << 30) : n));
#else
    return ::read(fd, dst, n);
#endif
}

// What the first Windows port did: seek, then read.  One thread at a time it is right; from 8 threads it is not.
ssize_t seek_read_pread(int fd, void* buf, size_t n, int64_t off) {
    if (!seek_to(fd, (uint64_t) off)) return -1;
    return read_here(fd, (uint8_t*) buf, n);
}

/// Every thread on the same fd, each read compared byte for byte with the block written there.  The count of wrong reads.
template <class Pread>
long long concurrent(int fd, const std::vector<uint64_t>& off, const std::vector<std::vector<uint8_t>>& data, Pread pread) {
    std::atomic<long long> bad{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::vector<uint8_t> buf(kBlock);
            uint64_t x = 0x2545F4914F6CDD1Dull * (uint64_t) (t + 1);
            auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
            for (int r = 0; r < kReadsPerThread; ++r) {
                const int b = (int) (next() % kBlocks);
                const size_t from = (size_t) (next() % (kBlock / 2));
                const size_t n = kBlock / 2 + (size_t) (next() % (kBlock / 2 - from));
                const ssize_t got = pread(fd, buf.data(), n, (int64_t) (off[b] + from));
                if (got != (ssize_t) n || std::memcmp(buf.data(), data[b].data() + from, n) != 0) ++bad;
            }
        });
    }
    for (std::thread& th : threads) th.join();
    return bad;
}

}  // namespace

int main() {
    const std::vector<uint64_t> off = block_offsets();
    {
        std::vector<uint64_t> sorted = off;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] < sorted[i - 1] + kBlock) {
                std::printf("ds4_pread_test: two blocks share bytes; the offsets of this test are wrong\n");
                return 1;
            }
    }
#if defined(_WIN32)
    const char* dir = std::getenv("TEMP");
    const std::string path = std::string(dir && *dir ? dir : ".") + "/ds4_pread_test_" + std::to_string(_getpid()) + ".bin";
#else
    const std::string path = "/tmp/ds4_pread_test_" + std::to_string(getpid()) + ".bin";
#endif
    std::string why;
    if (!make_file(path, off, why)) {
        std::printf("ds4_pread_test: not run (%s)\n", why.c_str());
        return why.rfind("FSCTL_SET_SPARSE", 0) == 0 ? 77 : 1;
    }
    std::vector<std::vector<uint8_t>> data(kBlocks, std::vector<uint8_t>(kBlock));
    for (int b = 0; b < kBlocks; ++b) fill(b, 0, kBlock, data[b].data());

    bool failed = false;
    const int fd = ds4_open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::printf("ds4_pread_test: ds4_open failed\n");
        failed = true;
    } else {
        std::vector<uint8_t> buf(kBlock);
        // 1. positional, one thread: the bytes come from the offset given, wherever the pointer is.  POSIX pread leaves
        //    the pointer alone; Windows' ReadFile moves it to the end of what it read (a synchronous handle), which is
        //    reported here, not required: nothing in ds4_moe.cpp reads through the pointer.
        int wrong = 0, moved = 0;
        seek_to(fd, 12345);                                   // the pointer starts somewhere of its own
        for (int b = 0; b < kBlocks; ++b) {
            const int64_t before = pointer_pos(fd);
            const ssize_t got = ds4_pread(fd, buf.data(), kBlock, (int64_t) off[b]);
            if (pointer_pos(fd) != before) ++moved;
            const bool ok = got == (ssize_t) kBlock && std::memcmp(buf.data(), data[b].data(), kBlock) == 0;
            if (!ok && ++wrong <= 5)
                std::printf("  block %d at offset %llu: %s\n", b, (unsigned long long) off[b],
                            got != (ssize_t) kBlock ? "a short or failed read" : "wrong bytes");
        }
        const uint64_t top = *std::max_element(off.begin(), off.end());
        std::printf("positional, one thread: %d blocks of 1 MiB, offsets 0 to %llu: %d wrong; the file pointer moved in "
                    "%d of %d reads (%s)\n", kBlocks, (unsigned long long) top, wrong, moved, kBlocks,
                    moved ? "Windows: ReadFile moves it on a synchronous handle" : "left alone, as POSIX pread does");
        failed = failed || wrong != 0;
#if !defined(_WIN32)
        failed = failed || moved != 0;                        // pread's own promise
#endif

        // 2. concurrent, then the control
        const long long good = concurrent(fd, off, data, ds4_pread);
        std::printf("concurrent, %d threads on one fd, %d reads of a block or a slice of one: %lld wrong\n",
                    kThreads, kThreads * kReadsPerThread, good);
        failed = failed || good != 0;
        const long long control = concurrent(fd, off, data, seek_read_pread);
        std::printf("control, the same reads through seek + read: %lld wrong%s\n", control,
                    control ? "" : " (the two calls did not interleave in this run)");

        // 3. the ends
        const ssize_t tail = ds4_pread(fd, buf.data(), kBlock, (int64_t) kFileEnd - 4096);
        const ssize_t none = ds4_pread(fd, buf.data(), kBlock, (int64_t) kFileEnd);
        const ssize_t zero = ds4_pread(fd, buf.data(), 0, 0);
        const bool ends_ok = tail == 4096 && none == 0 && zero == 0;
        std::printf("the ends: the last 4096 bytes of a 48 GiB file read as %lld, at its very end %lld, 0 bytes %lld: %s\n",
                    (long long) tail, (long long) none, (long long) zero, ends_ok ? "as they should be" : "WRONG");
        failed = failed || !ends_ok;
        ds4_close(fd);
    }
    remove_file(path);
    std::printf(failed ? "ds4_pread_test: FAILED\n" : "ds4_pread_test: every byte matched\n");
    return failed ? 1 : 0;
}
