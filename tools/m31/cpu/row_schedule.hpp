#pragma once
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

namespace strata::m31 {
// How the fused CPU decode hands independent matrix rows to pool workers.
// STRATA_M31_DECODE_ROWS: unset, "0" or "auto" = guided batches of about
// remaining/(2*nth) rows (a multiple of 16, at least 16); "N" = fixed N-row
// batches; "static" = one equal contiguous slice per worker (the 4499fcbc
// partition). The selection is parsed once, outside the per-token path.
struct RowSchedule {
    enum class Mode { Guided, Fixed, Static };
    static constexpr const char * environment = "STRATA_M31_DECODE_ROWS";
    static constexpr size_t guided_granule = 16;
    static constexpr size_t max_fixed_rows = size_t(1) << 20;
    Mode mode = Mode::Guided;
    size_t rows = guided_granule;

    static RowSchedule parse(const char * text) {
        RowSchedule result;
        if (!text) return result;
        const std::string value(text);
        if (value == "0" || value == "auto") return result;
        if (value == "static") { result.mode = Mode::Static; result.rows = 0; return result; }
        size_t rows = 0;
        const char * end = value.data() + value.size();
        const auto parsed = std::from_chars(value.data(), end, rows);
        if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != end || rows < 1 || rows > max_fixed_rows)
            throw std::invalid_argument(std::string("M31 configuration: ") + environment +
                " must be unset, 0, auto, static or a row count 1..1048576");
        result.mode = Mode::Fixed;
        result.rows = rows;
        return result;
    }
    static RowSchedule from_environment() { return parse(std::getenv(environment)); }
    std::string describe() const {
        if (mode == Mode::Static) return "static";
        if (mode == Mode::Fixed) return "fixed rows=" + std::to_string(rows);
        return "auto (guided remaining/(2*nth), multiple of 16, min 16)";
    }
};

struct alignas(64) RowCounter { std::atomic<size_t> next{0}; };

// Calls rows(begin, end) on disjoint ranges that together cover [0, size)
// exactly once across all nth participants. Dynamic modes need the counter
// reset to zero before the participants start; every claim is one fetch_add.
template <class Function>
inline void schedule_rows(const RowSchedule & schedule, size_t size, int ith, int nth,
                          RowCounter & counter, Function && rows) {
    if (schedule.mode == RowSchedule::Mode::Static) {
        const size_t begin = size * size_t(ith) / size_t(nth), end = size * size_t(ith + 1) / size_t(nth);
        if (begin < end) rows(begin, end);
        return;
    }
    if (schedule.mode == RowSchedule::Mode::Fixed) {
        for (size_t begin = counter.next.fetch_add(schedule.rows, std::memory_order_relaxed);
             begin < size;
             begin = counter.next.fetch_add(schedule.rows, std::memory_order_relaxed))
            rows(begin, std::min(begin + schedule.rows, size));
        return;
    }
    // The guided batch size comes from a possibly stale load; the claim itself
    // is one fetch_add, so claimed ranges remain disjoint and complete.
    const size_t divisor = 2 * size_t(nth) * RowSchedule::guided_granule;
    for (;;) {
        const size_t seen = counter.next.load(std::memory_order_relaxed);
        if (seen >= size) return;
        const size_t step = std::max<size_t>(1, (size - seen) / divisor) * RowSchedule::guided_granule;
        const size_t begin = counter.next.fetch_add(step, std::memory_order_relaxed);
        if (begin >= size) return;
        rows(begin, std::min(begin + step, size));
    }
}
}
