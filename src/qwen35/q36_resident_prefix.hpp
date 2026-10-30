#pragma once
#include "q36_page_provenance.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata { namespace q36 { namespace resident {
// Metadata for ONE exclusively owned Engine, never a logical caller or host
// cache record. A generation identifies the current mapping's physical bytes.
struct Identity {
    uint64_t owner=0, context=0, generation=0;
    bool operator==(const Identity& other) const {
        return owner==other.owner && context==other.context && generation==other.generation;
    }
};
class Prefix {
    Identity identity_;
    std::vector<uint64_t> cells_;
    size_t valid_=0,page_cells_=0;
    std::vector<PageStamp> pages_;
    bool known_=false;
public:
    void invalidate() noexcept { known_=false;valid_=0; }
    size_t valid_end() const noexcept { return valid_; }
    // Call BEFORE a main or MTP KV write. Aborted/rolled-back writes never
    // promote this bound: only a later fenced teacher-prefill may publish.
    void before_write(size_t first) noexcept { valid_=std::min(valid_,first); }
    void publish(Identity identity,const std::vector<uint64_t>& cells,size_t valid_end,
                 const std::vector<PageStamp>& pages,size_t page_cells) {
        if(!identity.owner || !identity.context || !identity.generation || valid_end>cells.size() || !page_cells)
            throw std::invalid_argument("resident prefix publication bounds/identity");
        const size_t needed=valid_end/page_cells+(valid_end%page_cells!=0);
        if(pages.size()<needed)throw std::invalid_argument("resident prefix lacks page provenance");
        for(size_t i=0;i<needed;++i)if(!pages[i].valid())throw std::invalid_argument("resident page write is incomplete");
        cells_.assign(cells.begin(),cells.begin()+valid_end);
        pages_.assign(pages.begin(),pages.begin()+needed);
        page_cells_=page_cells;identity_=identity;valid_=valid_end;known_=true;
    }
    size_t reusable(Identity identity,const std::vector<uint64_t>& incoming,
                    size_t requested,size_t mapped_end,const std::vector<PageStamp>& selected,
                    size_t page_cells) const noexcept {
        if(!known_ || !(identity==identity_) || requested>mapped_end || requested>incoming.size() ||
           !page_cells || page_cells!=page_cells_)return 0;
        const size_t limit=std::min({requested,valid_,cells_.size()});
        size_t common=0;
        while(common<limit && cells_[common]==incoming[common])++common;
        for(size_t start=0;start<common;start+=page_cells) {
            const size_t page=start/page_cells;
            if(page>=pages_.size() || page>=selected.size() || !selected[page].valid() ||
               !(pages_[page]==selected[page]))return start;
        }
        return common;
    }
};
}}}
