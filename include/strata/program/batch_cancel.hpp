#pragma once
#include <atomic>
#include <charconv>
#include <cstdint>
#include <memory>
#include <string_view>
namespace strata::program {
struct BatchStop { int slot = -1; uint64_t epoch = 0; };
inline bool parse_batch_stop(std::string_view line, BatchStop& out) {
    if (!line.starts_with("BSTOP ")) return false;
    line.remove_prefix(6);
    const auto gap = line.find(' ');
    if (gap == line.npos) return false; // legacy command stays on the ordinary queue
    int slot = -1; uint64_t epoch = 0;
    auto a = std::from_chars(line.data(), line.data()+gap, slot);
    auto tail = line.substr(gap+1);
    auto b = std::from_chars(tail.data(), tail.data()+tail.size(), epoch);
    if (a.ec != std::errc{} || a.ptr != line.data()+gap || b.ec != std::errc{} ||
        b.ptr != tail.data()+tail.size() || slot < 0 || epoch == 0) return false;
    out = {slot, epoch}; return true;
}
inline uint64_t batch_cancel_epoch(std::string_view line) {
    constexpr std::string_view key = " cancel_epoch=";
    const auto at = line.find(key);
    if (at == line.npos) return 0;
    const auto first = at + key.size(), end = line.find(' ', first);
    const auto last = end == line.npos ? line.size() : end;
    uint64_t value = 0; auto r = std::from_chars(line.data()+first, line.data()+last, value);
    return r.ec == std::errc{} && r.ptr == line.data()+last ? value : 0;
}
// Only the scheduler publishes/retire owners; a reader only raises a monotonic
// request epoch. It may arrive before BGEN is dequeued. A stale old request can
// never overwrite a newer cancellation; only equality grants cancellation.
class BatchCancelMailbox {
    struct Slot { std::atomic<uint64_t> owner{0}, requested{0}, pending{0}; };
    size_t count_; std::unique_ptr<Slot[]> slots_;
public:
    explicit BatchCancelMailbox(size_t n):count_(n),slots_(new Slot[n]) {}
    bool valid(int slot) const { return slot >= 0 && static_cast<size_t>(slot) < count_; }
    void begin(int slot, uint64_t epoch) { if(valid(slot)) slots_[slot].owner.store(epoch); }
    void request(int slot, uint64_t epoch) {
        if(!valid(slot) || !epoch) return;
        auto& s=slots_[slot];
        // Keep the current owner's request independent of a future queued owner.
        // A refused future admission must not suppress cancellation of the active one.
        auto& target=(s.owner.load()==epoch ? s.requested : s.pending); auto old=target.load();
        while(old < epoch && !target.compare_exchange_weak(old,epoch)) {}
    }
    bool cancelled(int slot, uint64_t epoch) const {
        return valid(slot) && epoch && slots_[slot].owner.load()==epoch && (slots_[slot].requested.load()==epoch || slots_[slot].pending.load()==epoch);
    }
    void retire(int slot,uint64_t epoch) {
        if(valid(slot) && epoch) slots_[slot].owner.compare_exchange_strong(epoch,0);
    }
};
} // namespace strata::program
