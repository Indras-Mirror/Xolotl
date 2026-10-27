#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace strata::m31 {
// The index-score/pool kernels require strictly fewer than INT32_MAX cells.
// This is a correctness limit on every backend, independent of VRAM planning.
inline int msa_chunk_limit(int64_t position,int requested,int64_t index_heads,
                           int64_t block) {
    if(position<0||position>INT32_MAX||requested<0||index_heads<=0||block<=0||block>256||256%block)
        throw std::runtime_error("M31 invalid MSA chunk-limit geometry/request");
    const uint64_t alignment=uint64_t(std::max<int64_t>(256,block));
    auto fits=[&](int n){
        const uint64_t end=uint64_t(position)+uint64_t(n);
        const uint64_t keys=((end+alignment-1)/alignment)*alignment;
        uint64_t rows=uint64_t(n);
        // Divide sequentially rather than multiplying attacker-controlled
        // dimensions. floor(floor(limit/keys)/heads) is the same safe bound.
        return !keys||rows<=(uint64_t(INT32_MAX)-1)/keys/uint64_t(index_heads);
    };
    int low=0,high=requested;
    while(low<high){
        const int middle=low+(high-low)/2+1;
        if(fits(middle))low=middle;else high=middle-1;
    }
    return low;
}
}
