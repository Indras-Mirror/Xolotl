#pragma once
#include "host_executor.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::m31 {
// A bounded persistent copy pool. It owns no weight or staging memory.
// Its synchronous return is the lifetime barrier before the caller enqueues H2D.
class CpuPackSpans {
    static constexpr size_t stripe = 1024 * 1024;
    GpuHostExecutor executor;
    static GpuHostExecutor::Options options(unsigned threads) {
        if (!threads || threads > 16) throw std::invalid_argument("M31 pack spans: threads must be1..16");
        return {threads, 2, 1};
    }
    static bool overlaps(const void * x, size_t nx, const void * y, size_t ny) {
        if (!nx || !ny) return false;
        auto a = reinterpret_cast<uintptr_t>(x), b = reinterpret_cast<uintptr_t>(y);
        return a <= b ? b - a < nx : a - b < ny;
    }
public:
    explicit CpuPackSpans(unsigned threads = 1)
        : executor(options(threads)) {}
    // Span needs {void* destination; const void* source; size_t bytes;}, matching
    // PrefillUploads::CopySpan. Descriptors and storage stay borrowed until return.
    template<class Span> void copy(const Span * spans, size_t count) {
        if ((count && !spans) || count > 64) throw std::invalid_argument("M31 pack spans: invalid descriptor count");
        std::array<size_t, 65> prefix{};
        for (size_t i = 0; i < count; ++i) {
            const auto & s = spans[i];
            if (s.bytes && (!s.destination || !s.source)) throw std::invalid_argument("M31 pack spans: null storage");
            if (s.bytes > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(s.destination) ||
                s.bytes > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(s.source))
                throw std::overflow_error("M31 pack spans: address range overflow");
            prefix[i + 1] = prefix[i] + s.bytes / stripe + (s.bytes % stripe != 0);
            for (size_t j = 0; j <= i; ++j) {
                if ((i != j && overlaps(s.destination, s.bytes, spans[j].destination, spans[j].bytes)) ||
                    overlaps(s.destination, s.bytes, spans[j].source, spans[j].bytes) ||
                    overlaps(spans[j].destination, spans[j].bytes, s.source, s.bytes))
                    throw std::invalid_argument("M31 pack spans: overlapping source/destination");
            }
        }
        if (executor.options().threads == 1 || prefix[count] < 2) {
            for (size_t i = 0; i < count; ++i)
                if (spans[i].bytes) std::memcpy(spans[i].destination, spans[i].source, spans[i].bytes);
            return;
        }
        executor.parallel_count(prefix[count], [&](size_t begin, size_t end) {
            size_t role = 0;
            while (role < count && prefix[role + 1] <= begin) ++role;
            for (size_t index = begin; index < end; ++index) {
                while (role < count && prefix[role + 1] <= index) ++role;
                const auto & s = spans[role];
                const size_t offset = (index - prefix[role]) * stripe;
                std::memcpy(static_cast<uint8_t *>(s.destination) + offset,
                    static_cast<const uint8_t *>(s.source) + offset, std::min(stripe, s.bytes - offset));
            }
        });
    }
    GpuHostExecutor::Stats stats() const { return executor.stats(); }
};
}
