#pragma once
// CPU codec for one complete restorable Q36 checkpoint. Native buffers and
// ownership are deliberately absent from the wire format.
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata { namespace q36 { namespace persistent {
template<class Cleanup> struct ScopeExit {
    Cleanup cleanup;
    bool armed=true;
    ~ScopeExit() { if(armed)cleanup(); }
};
struct Checkpoint {
    std::vector<int32_t> tokens;
    std::vector<uint8_t> state, pages;
    const uint8_t* state_data=nullptr;
    const uint8_t* page_data=nullptr;
    size_t pages_bytes=0;
    uint64_t hash = 0;
    int priority = 0;
};
struct Writer {
    std::vector<uint8_t> bytes;
    void u64(uint64_t v) {
        const size_t at=bytes.size();
        if(at>bytes.max_size()-8)throw std::length_error("Q36 checkpoint length");
        bytes.resize(at+8);
        for(unsigned i=0;i<8;++i)bytes[at+i]=uint8_t(v>>(8*i));
    }
    void raw(const void* p,size_t n) {
        const size_t at=bytes.size();
        if(n>bytes.max_size()-at)throw std::length_error("Q36 checkpoint length");
        bytes.resize(at+n);
        if(n)std::memcpy(bytes.data()+at,p,n);
    }
};
inline bool size_bytes(size_t prefix,size_t count,size_t width,size_t& out) {
    if(width && count>(std::numeric_limits<size_t>::max()-prefix)/width)return false;
    out=prefix+count*width;return true;
}
struct Reader {
    const std::vector<uint8_t>& bytes;
    size_t at=0;
    bool u64(uint64_t& v) {
        if(bytes.size()-at<8)return false;
        v=0;for(unsigned i=0;i<8;++i)v|=uint64_t(bytes[at++])<<(8*i);return true;
    }
    bool raw(void* p,size_t n) {
        if(n>bytes.size()-at)return false;
        if(n)std::memcpy(p,bytes.data()+at,n);at+=n;return true;
    }
};
inline std::vector<uint8_t> encode(const Checkpoint& c,size_t page_bytes,size_t snapshot_bytes) {
    Writer w;w.u64(0x315043363351ULL);w.u64(c.tokens.size());w.u64(page_bytes);w.u64(snapshot_bytes);
    w.u64(c.hash);w.u64(uint64_t(c.priority));w.u64(c.pages.size());
    for(auto t:c.tokens)w.u64(uint32_t(t));
    w.raw(c.state.data(),c.state.size());w.raw(c.pages.data(),c.pages.size());return std::move(w.bytes);
}
inline bool decode(const std::vector<uint8_t>& bytes,const std::vector<uint64_t>& cells,
                   size_t page_bytes,size_t snapshot_bytes,size_t context,Checkpoint& result,bool copy=true) {
    Reader r{bytes};uint64_t magic,n,pb,sb,hash,priority,pages;
    if(!r.u64(magic)||magic!=0x315043363351ULL||!r.u64(n)||n==0||n>context||n!=cells.size()||
       !r.u64(pb)||pb!=page_bytes||!r.u64(sb)||sb!=snapshot_bytes||!r.u64(hash)||
       !r.u64(priority)||priority>3||!r.u64(pages)||!page_bytes)return false;
    const uint64_t count=(n+1023)/1024;
    if(count>std::numeric_limits<size_t>::max()/page_bytes||pages!=count*page_bytes||
       n>(bytes.size()-r.at)/8)return false;
    // Validate the entire length before any model-sized allocation.
    const size_t tail=bytes.size()-r.at-size_t(n)*8;
    if(snapshot_bytes>tail||pages!=tail-snapshot_bytes)return false;
    Checkpoint c;c.hash=hash;c.priority=int(priority);c.tokens.reserve(size_t(n));
    for(uint64_t i=0;i<n;++i){uint64_t t;if(!r.u64(t)||t>uint64_t(INT32_MAX))return false;c.tokens.push_back(int32_t(t));}
    c.state_data=bytes.data()+r.at;c.page_data=c.state_data+snapshot_bytes;c.pages_bytes=size_t(pages);
    if(copy) {
        c.state.assign(c.state_data,c.state_data+snapshot_bytes);
        c.pages.assign(c.page_data,c.page_data+c.pages_bytes);
    }
    result=std::move(c);return true;
}
}}}
