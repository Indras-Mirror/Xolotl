#include "../../src/qwen35/q36_resident_prefix.hpp"
#include "../../src/qwen35/q36_kv_vmm.hpp"
#include <iostream>
#include <set>
#include <stdexcept>
#include <limits>
using strata::q36::resident::Identity;
// Stable synthetic page certificates isolate the older identity/write-bound
// checks. Real Hub cookies, write tickets and COW are tested separately.
class Prefix : public strata::q36::resident::Prefix {
    static std::vector<strata::q36::resident::PageStamp> stamps(size_t cells) {
        std::vector<strata::q36::resident::PageStamp> out;
        for(size_t i=0;i<cells/1024+(cells%1024!=0);++i)out.push_back({i+1,1,1});
        return out;
    }
public:
    void publish(Identity id,const std::vector<uint64_t>& cells,size_t end) {
        strata::q36::resident::Prefix::publish(id,cells,end,stamps(cells.size()),1024);
    }
    size_t reusable(Identity id,const std::vector<uint64_t>& cells,size_t requested,size_t mapped)const {
        return strata::q36::resident::Prefix::reusable(id,cells,requested,mapped,stamps(cells.size()),1024);
    }
};
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static std::vector<uint64_t> cells(size_t n){std::vector<uint64_t> v(n);for(size_t i=0;i<n;++i)v[i]=0x9000000000000000ULL+i;return v;}
struct MockDriver : strata::q36::kv::Driver {
 std::set<uintptr_t> maps;size_t calls=0,fail_map=SIZE_MAX;bool fail_unmap=false;
 size_t granularity()const override{return size_t(2)<<20;}
 bool reserve(uintptr_t& p,size_t,size_t)override{p=uintptr_t(1)<<40;return true;}
 bool map(uintptr_t p,size_t)override{if(calls++==fail_map)return false;return maps.insert(p).second;}
 bool unmap(uintptr_t p,size_t)override{if(fail_unmap)return false;return maps.erase(p)==1;}
 bool free_address(uintptr_t,size_t)override{return true;}
};
static void owned_prefix(){
 Prefix p;Identity a{11,21,31};auto v=cells(1000);check(p.reusable(a,v,1000,1000)==0,"unpublished must miss");p.publish(a,v,1000);
 check(p.reusable(a,v,1000,1000)==1000,"same identity exact prefix");
 for(auto wrong:std::vector<Identity>{{12,21,31},{11,22,31},{11,21,32}})check(p.reusable(wrong,v,1000,1000)==0,"owner/context/mapping isolation");
 auto grown=cells(2000);check(p.reusable(a,grown,1000,2000)==1000,"mapped growth preserves old physical prefix");check(p.reusable(a,grown,2000,2000)==1000,"cannot skip unpublished growth");
 check(p.reusable(a,grown,900,2000)==900,"snapshot requested bound cannot be exceeded");
 grown[407]^=0x1234;check(p.reusable(a,grown,1000,2000)==407,"changed image/cell identity must stop before mismatch");
 check(p.reusable(a,v,1001,2000)==0,"request beyond input rejected");check(p.reusable(a,v,1000,999)==0,"request beyond physical mapping rejected");
 check(p.reusable(a,v,SIZE_MAX,SIZE_MAX)==0,"oversized requested bound rejected without scan");
}
static void write_abort_and_fenced_republish(){
 Prefix p;Identity a{1,2,3};auto v=cells(1000);p.publish(a,v,1000);
 // A main/teacher/draft write dirties the same physical prefix even if its token
 // IDs happen to equal the old request. Cancellation cannot certify old bytes.
 p.before_write(800);check(p.valid_end()==800,"first write clamps");p.before_write(900);check(p.valid_end()==800,"later write cannot promote");
 check(p.reusable(a,v,1000,1000)==800,"aborted transaction only retains prior clean prefix");
 p.before_write(640);check(p.reusable(a,v,900,1000)==640,"second earlier write clamps again");
 auto fresh=v;fresh[700]^=0xbeef;p.publish(a,fresh,900);check(p.reusable(a,fresh,1000,1000)==900,"only fenced publication promotes committed PP");
 check(p.reusable(a,v,1000,1000)==700,"republication keeps new image/row identity");p.before_write(0);check(p.reusable(a,fresh,900,1000)==0,"write at zero invalidates all certified rows");
 p.publish(a,v,1000);p.invalidate();check(p.valid_end()==0&&p.reusable(a,v,1000,1000)==0,"checkpoint sleep invalidates physical hint");
 p.publish(a,v,0);check(p.reusable(a,v,1000,1000)==0,"valid empty publication");
}
static void invalid_publications(){
 Prefix p;auto v=cells(10);Identity a{1,2,3};p.publish(a,v,10);
 for(auto id:std::vector<Identity>{{0,2,3},{1,0,3},{1,2,0}}){bool threw=false;try{p.publish(id,v,10);}catch(const std::invalid_argument&){threw=true;}check(threw,"zero identity must reject");check(p.reusable(a,v,10,10)==10,"bad publication cannot destroy certified prefix");}
 bool threw=false;try{p.publish(a,v,11);}catch(const std::invalid_argument&){threw=true;}check(threw,"invalid publication bound rejects");check(p.reusable(a,v,10,10)==10,"bound failure atomicity");
}
static void actual_vmm_generation(){
 using strata::q36::kv::Budget;using strata::q36::kv::Arena;constexpr size_t MiB=size_t(1)<<20;MockDriver d;Budget b(512*MiB);Arena arena(b,d,2,262144,512);check(arena.init()&&arena.ensure(40000),"fake VMM admission");
 auto v=cells(80000);Identity current{17,29,arena.mapping_generation()};Prefix p;p.publish(current,v,40000);
 check(arena.ensure(80000),"grow physical mapping");check(arena.mapping_generation()==current.generation,"growth must preserve mapping generation");check(p.reusable({17,29,arena.mapping_generation()},v,40000,arena.committed_tokens())==40000,"growth retains certified old prefix");
 auto maps=d.maps;d.fail_map=d.calls;check(!arena.ensure(120000),"failed grow");check(d.maps==maps&&arena.mapping_generation()==current.generation,"failed map must leave old physical prefix valid");d.fail_map=SIZE_MAX;
 check(arena.trim(80000),"no-op trim");check(arena.mapping_generation()==current.generation,"no-op trim preserves generation");
 check(arena.trim(40000),"real trim");check(arena.mapping_generation()!=current.generation,"real trim bumps generation");check(p.reusable({17,29,arena.mapping_generation()},v,40000,arena.committed_tokens())==0,"old certificate cannot survive remapping identity");
 check(arena.ensure(80000),"regrow after trim");check(p.reusable({17,29,arena.mapping_generation()},v,40000,arena.committed_tokens())==0,"same address and same cell IDs cannot resurrect old certificate");
 // First failed unmap must already revoke the generation, not after success.
 const auto before=arena.mapping_generation();d.fail_unmap=true;check(!arena.trim(0),"failed trim");check(arena.mapping_generation()!=before,"failed destructive trim revokes identity before unmap");d.fail_unmap=false;
}
int main(){try{owned_prefix();write_abort_and_fenced_republish();invalid_publications();actual_vmm_generation();std::cout<<"resident prefix identity/write/cancel/bounds/VMM generation checks PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
