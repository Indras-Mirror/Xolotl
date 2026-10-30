#pragma once
// Slow diagnostic comparator; never treats a hash alone as byte equality.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace strata { namespace q36 { namespace resident {
struct Difference {
    uint64_t bytes=0,different_bytes=0,different_cells=0;
    uint64_t gpu_hash=14695981039346656037ULL,ram_hash=14695981039346656037ULL;
    uint64_t first_cell=std::numeric_limits<uint64_t>::max(),last_cell=0;
    size_t first_byte_in_cell=0;
    void observe(uint64_t first,const uint8_t* gpu,const uint8_t* ram,size_t count,size_t cell_bytes) {
        if(!cell_bytes || count%cell_bytes || (count && (!gpu || !ram)))throw std::invalid_argument("resident verify comparison bounds");
        const size_t cells=count/cell_bytes;
        if(cells && first>std::numeric_limits<uint64_t>::max()-(cells-1))throw std::invalid_argument("resident verify cell overflow");
        for(size_t c=0;c<cells;++c) {
            bool changed=false;
            for(size_t b=0;b<cell_bytes;++b) {
                const size_t at=c*cell_bytes+b;
                gpu_hash=(gpu_hash^gpu[at])*1099511628211ULL;
                ram_hash=(ram_hash^ram[at])*1099511628211ULL;
                if(gpu[at]!=ram[at]) {
                    if(!different_bytes){first_cell=first+c;first_byte_in_cell=b;}
                    ++different_bytes;changed=true;
                }
            }
            if(changed){++different_cells;last_cell=first+c;}
        }
        bytes+=count;
    }
    bool same() const noexcept { return different_bytes==0; }
};
}}}
