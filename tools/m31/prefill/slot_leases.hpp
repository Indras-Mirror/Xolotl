#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
// Serialized owner-side bookkeeping, paired with the uploader's real stream
// events. A reserved write is unpublished until DMA completion is established;
// a read lease lives until its actual compute backend has completed.
class SlotLeases {
    struct State { bool writing = false, reading = false; };
    std::vector<State> slots_;
    State & slot(size_t index) {
        if(index>=slots_.size())throw std::logic_error("M31 slot lease: invalid slot");
        return slots_[index];
    }
public:
    void reset(size_t count) { slots_.assign(count,{}); }
    void reserve_write(size_t index) {
        auto & s=slot(index);
        if(s.writing||s.reading)throw std::logic_error("M31 slot lease: write overlaps an active reader/writer");
        s.writing=true;
    }
    void publish_write(size_t index) {
        auto & s=slot(index);
        if(!s.writing)throw std::logic_error("M31 slot lease: publish without a reserved write");
        s.writing=false;
    }
    void begin_read(size_t index) {
        auto & s=slot(index);
        if(s.writing||s.reading)throw std::logic_error("M31 slot lease: overlapping reader or unpublished weights");
        s.reading=true;
    }
    void finish_reads() noexcept { for(auto & s:slots_)s.reading=false; }
    bool writable(size_t index) const {
        if(index>=slots_.size())throw std::logic_error("M31 slot lease: invalid slot");
        return !slots_[index].writing&&!slots_[index].reading;
    }
};
} // namespace strata::m31
