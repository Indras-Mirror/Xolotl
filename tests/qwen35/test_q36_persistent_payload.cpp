#include "../../src/qwen35/q36_persistent_payload.hpp"
#include <cstdio>
#include <cstdlib>

namespace p = strata::q36::persistent;
static void require(bool value,const char* why) {
    if(!value){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}
}
int main() {
    {
        const std::vector<uint8_t> source(3<<20,0x7b);
        p::Writer capture;size_t polls=0;
        require(!capture.capture(source.data(),source.size(),[&]{return ++polls==2;}),"mid-capture cancellation");
        require(capture.bytes.size()==size_t(1<<20) &&
                std::equal(capture.bytes.begin(),capture.bytes.end(),source.begin()),"only first bounded copy completed");
        p::Writer complete;
        require(complete.capture(source.data(),source.size(),[]{return false;}) &&
                complete.capture(nullptr,17,[]{return false;}),"uncancelled capture and padding");
        require(std::equal(source.begin(),source.end(),complete.bytes.begin()) &&
                std::all_of(complete.bytes.end()-17,complete.bytes.end(),[](uint8_t v){return v==0;}),"capture bytes and padding unchanged");
    }
    size_t count=0;
    require(p::size_bytes(40,10,4,count)&&count==80,"checked image length");
    require(!p::size_bytes(40,SIZE_MAX,4,count),"image length overflow rejected");
    constexpr size_t page=8192,state=17;
    p::Checkpoint original;
    std::vector<uint64_t> cells;
    for(int32_t i=0;i<1025;++i){original.tokens.push_back(i);cells.push_back(uint64_t(i));}
    original.state.assign(state,0x31);original.pages.assign(2*page,0x72);
    original.hash=123456;original.priority=3;
    const auto wire=p::encode(original,page,state);
    p::Checkpoint decoded;
    require(p::decode(wire,cells,page,state,2048,decoded),"round trip");
    require(decoded.tokens==original.tokens&&decoded.state==original.state&&decoded.pages==original.pages&&
            decoded.hash==original.hash&&decoded.priority==original.priority,"all checkpoint fields retained");
    p::Checkpoint view;
    require(p::decode(wire,cells,page,state,2048,view,false),"zero-copy restore view");
    require(view.state.empty()&&view.pages.empty()&&view.pages_bytes==original.pages.size()&&
            std::memcmp(view.state_data,original.state.data(),state)==0&&
            std::memcmp(view.page_data,original.pages.data(),original.pages.size())==0,"view references verified wire bytes");
    for(size_t end:{size_t(0),size_t(7),size_t(55),size_t(56),size_t(56+1025*8),wire.size()-1}) {
        const std::vector<uint8_t> truncated(wire.begin(),wire.begin()+end);
        require(!p::decode(truncated,cells,page,state,2048,decoded),"truncated state rejected");
    }
    auto bad=wire;bad.push_back(0);
    require(!p::decode(bad,cells,page,state,2048,decoded),"trailing bytes rejected");
    bad=wire;bad[0]^=1;
    require(!p::decode(bad,cells,page,state,2048,decoded),"foreign format rejected");
    bad=wire;bad[40]=4;
    require(!p::decode(bad,cells,page,state,2048,decoded),"invalid priority rejected");
    bad=wire;for(size_t i=8;i<16;++i)bad[i]=0xff;
    require(!p::decode(bad,cells,page,state,2048,decoded),"overflow count rejected before allocation");
    require(!p::decode(wire,cells,page+1,state,2048,decoded),"KV geometry mismatch");
    require(!p::decode(wire,cells,page,state+1,2048,decoded),"recurrent geometry mismatch");
    require(!p::decode(wire,cells,page,state,1024,decoded),"context mismatch");
    auto shorter=cells;shorter.pop_back();
    require(!p::decode(wire,shorter,page,state,2048,decoded),"catalogue count mismatch");
    std::puts("Q36 persistent payload CPU codec PASS (no inference)");
}
