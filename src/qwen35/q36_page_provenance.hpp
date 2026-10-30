#pragma once
#include <cstdint>
namespace strata { namespace q36 { namespace resident {
struct PageStamp {
    uint64_t page=0,cookie=0,version=0;
    bool valid()const noexcept{return page && cookie && version;}
    bool operator==(const PageStamp& other)const noexcept {
        return page==other.page && cookie==other.cookie && version==other.version;
    }
};
}}}
