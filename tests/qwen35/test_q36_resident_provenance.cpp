#include "../../src/qwen35/q36_cache_hub.hpp"
#include "../../src/qwen35/q36_resident_prefix.hpp"
#include <array>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <future>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <vector>
using strata::q36::cache::Hub;
using strata::q36::cache::Conversation;
using strata::q36::resident::Identity;
using strata::q36::resident::PageStamp;
using strata::q36::resident::Prefix;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Fixture {
 Hub hub{4};std::array<std::array<uint8_t,64>,12> memory{};
 Fixture(){hub.track_page_versions=true;hub.page_bytes=64;hub.n_pages=memory.size();for(auto& p:memory)hub.free_pages.push_back(p.data());}
 uint8_t* ready(uint8_t value){auto*p=hub.take_page();check(p!=nullptr,"pool page available");auto ticket=hub.begin_page_write(p);std::memset(p,value,hub.page_bytes);check(hub.finish_page_write(ticket),"fenced write ready");return p;}
};
static std::vector<uint64_t> cells(size_t n){std::vector<uint64_t>v(n);for(size_t i=0;i<n;++i)v[i]=0xabc000+i;return v;}
static void changed_bytes_same_tokens_and_address(){
 Fixture f;auto v=cells(23);std::vector<uint8_t*> pages{f.ready(1),f.ready(2),f.ready(3)};auto original=f.hub.page_stamps(pages);Identity id{1,2,3};Prefix resident;resident.publish(id,v,23,original,8);
 check(resident.reusable(id,v,23,24,original,8)==23,"unchanged canonical pages reuse full prefix");
 // Real regression: CPU canonical bytes change within the last page, while
 // pointer, logical input cells, owner and mapping generation remain identical.
 auto ticket=f.hub.begin_page_write(pages[2]);check(!f.hub.page_stamp(pages[2]).valid(),"in-flight page cannot be certified");
 check(resident.reusable(id,v,23,24,f.hub.page_stamps(pages),8)==16,"writing last page clamps to preceding canonical pages");
 pages[2][17]^=0x5a;check(f.hub.finish_page_write(ticket),"partial write completion");auto changed=f.hub.page_stamps(pages);
 check(changed[2].page==original[2].page&&changed[2].cookie==original[2].cookie&&changed[2].version!=original[2].version,"ref1 same-address write changes content version");
 check(resident.reusable(id,v,23,24,changed,8)==16,"same token IDs never justify stale last-page skip");
 resident.before_write(19);check(resident.reusable(id,v,23,24,changed,8)==16,"GPU clamp alone does not bypass host provenance");
 resident.publish(id,v,23,changed,8);check(resident.reusable(id,v,23,24,changed,8)==23,"only fenced republish certifies new canonical page");
 auto wrong=changed;wrong[1]=changed[2];check(resident.reusable(id,v,23,24,wrong,8)==8,"stop at FIRST mismatching page, no holes");
 wrong=changed;wrong[2]={};check(resident.reusable(id,v,23,24,wrong,8)==16,"invalid stamp cannot be skipped");
 wrong.resize(1);check(resident.reusable(id,v,23,24,wrong,8)==8,"missing selected page is fail-closed");
 auto edited=v;edited[19]^=1;check(resident.reusable(id,edited,23,24,changed,8)==19,"cell mismatch still limits matching canonical page");
 check(resident.reusable({1,2,4},v,23,24,changed,8)==0,"trim/remap generation revokes page proof");
 check(resident.reusable({9,2,3},v,23,24,changed,8)==0,"another physical owner cannot borrow proof");
 check(resident.reusable(id,v,23,22,changed,8)==0,"mapped bounds guard");
 f.hub.release_pages(pages);
}
static void cookie_recycle_and_stale_completion(){
 Fixture f;auto*p=f.ready(7);auto old=f.hub.page_stamp(p);Prefix resident;Identity id{1,2,3};auto v=cells(8);resident.publish(id,v,8,{old},8);
 auto first=f.hub.begin_page_write(p);auto second=f.hub.begin_page_write(p);
 check(!f.hub.finish_page_write(first)&&!f.hub.page_stamp(p).valid(),"late prior completion cannot bless newer pending write");
 check(f.hub.finish_page_write(second),"current ticket completion");check(!f.hub.finish_page_write(first),"late old callback remains stale after completion");
 std::vector<uint8_t*> ownership{p};f.hub.release_pages(ownership);check(!f.hub.page_stamp(p).valid(),"free page has no readable provenance");
 auto*recycled=f.hub.take_page();check(recycled==p,"exercise actual freelist ABA reuse");auto current=f.hub.begin_page_write(recycled);std::memset(recycled,7,64);
 check(!f.hub.finish_page_write(second),"old ticket cannot complete recycled allocation");check(!f.hub.page_stamp(recycled).valid(),"ABA stale callback leaves new write unready");check(f.hub.finish_page_write(current),"new allocation completes");auto stamp=f.hub.page_stamp(recycled);
 check(stamp.page==old.page&&stamp.cookie!=old.cookie,"same address gets new allocation cookie");
 check(resident.reusable(id,v,8,8,{stamp},8)==0,"recycled address and same input IDs cannot resurrect proof");ownership={recycled};f.hub.release_pages(ownership);
}
static void cow_immutable_and_failure(){
 Fixture f;auto*src=f.ready(4);auto before=f.hub.page_stamp(src);std::vector<uint8_t*> original{src};f.hub.retain_pages(original);
 auto*dst=f.hub.cow_page(src,[&](uint8_t*to,const uint8_t*from,size_t n){check(!f.hub.page_stamp(to).valid(),"COW destination unpublished while callback runs");check(f.hub.page_stamp(src)==before,"COW source stays immutable while pinned");check(f.hub.page_refs.at(src)==3,"COW holds independent source pin");std::memcpy(to,from,n);return true;});
 check(dst&&dst!=src,"COW different owned buffer");auto dest=f.hub.page_stamp(dst);check(dest.valid()&&!(dest==before),"COW destination has fresh provenance");check(f.hub.page_stamp(src)==before&&std::memcmp(src,dst,64)==0,"COW does not change source stamp or bytes");
 auto old=Prefix{};auto v=cells(8);Identity id{1,2,3};old.publish(id,v,8,{before},8);check(old.reusable(id,v,8,8,{dest},8)==0,"conservative source identity required even for copied bytes");
 uint8_t*failed=nullptr;auto*result=f.hub.cow_page(src,[&](uint8_t*to,const uint8_t*,size_t){failed=to;to[0]=99;check(!f.hub.page_stamp(to).valid(),"failed-copy destination never ready");return false;});
 check(!result&&failed&&!f.hub.page_stamp(failed).valid(),"failed copy cannot publish a page");check(f.hub.page_stamp(src)==before&&f.hub.page_refs.at(src)==2,"failed copy releases only temporary source pin");
 bool threw=false;try{f.hub.cow_page(src,[&](uint8_t*,const uint8_t*,size_t)->bool{throw std::runtime_error("synthetic copy failure");});}catch(const std::runtime_error&){threw=true;}
 check(threw&&f.hub.page_refs.at(src)==2&&f.hub.page_stamp(src)==before,"throwing copy preserves source and releases pin");
 std::vector<uint8_t*> d{dst};f.hub.release_pages(d);f.hub.release_pages(original);original={src};f.hub.release_pages(original);
}
static void publication_and_leased_handoff(){
 Fixture f;auto*p=f.hub.take_page();auto pending=f.hub.begin_page_write(p);Conversation c;c.tok.assign(8,1);c.val=cells(8);c.mirrored=8;c.pages={p};
 bool refused=false;try{f.hub.publish(std::move(c));}catch(const std::logic_error&){refused=true;}check(refused&&f.hub.parked.empty(),"Hub cannot publish unfinished copy");
 Prefix bad;refused=false;try{bad.publish({1,2,3},c.val,8,f.hub.page_stamps(c.pages),8);}catch(const std::invalid_argument&){refused=true;}check(refused,"resident hint requires ready stamps");
 std::memset(p,8,64);check(f.hub.finish_page_write(pending),"fenced canonical publication");f.hub.publish(std::move(c));
 auto a=f.hub.checkout(0);auto b=f.hub.checkout(0);check(a&&b&&a.conv.pages==b.conv.pages,"two owners lease same immutable cache source");auto stable=f.hub.page_stamps(a.conv.pages);Prefix owned;owned.publish({11,22,33},a.conv.val,8,stable,8);
 refused=false;try{f.hub.begin_page_write(p);}catch(const std::logic_error&){refused=true;}check(refused,"leased/published aliases cannot be ref1 overwritten");
 check(owned.reusable({11,22,33},a.conv.val,8,8,f.hub.page_stamps(b.conv.pages),8)==8,"same physical owner retains immutable source proof");
 check(owned.reusable({12,22,33},a.conv.val,8,8,f.hub.page_stamps(b.conv.pages),8)==0,"second GPU owner must perform its own restore");
 Prefix other;other.publish({12,22,33},b.conv.val,8,stable,8);check(other.reusable({12,22,33},b.conv.val,8,8,stable,8)==8,"second owner can certify AFTER its own fenced restore");
 // Physical sleep invalidation remains necessary despite unchanged canonical RAM.
 owned.invalidate();check(owned.reusable({11,22,33},a.conv.val,8,8,stable,8)==0,"pause invalidates physical hint with stable host bytes");
}
static void blocked_cow_remains_unpublished_without_hub_stall(){
 Fixture f;auto*src=f.ready(6);std::vector<uint8_t*> hold{src};f.hub.retain_pages(hold);const auto before=f.hub.page_stamp(src);
 std::mutex gate;std::condition_variable cv;bool entered=false,release=false;uint8_t*pending=nullptr;uint8_t*done=nullptr;std::exception_ptr failure;
 std::thread writer([&]{try{done=f.hub.cow_page(src,[&](uint8_t*to,const uint8_t*from,size_t n){
   {std::unique_lock<std::mutex> lock(gate);pending=to;entered=true;cv.notify_all();cv.wait(lock,[&]{return release;});}
   std::memcpy(to,from,n);return true;
  });}catch(...){failure=std::current_exception();}});
 bool arrived=false;{std::unique_lock<std::mutex> lock(gate);arrived=cv.wait_for(lock,std::chrono::seconds(2),[&]{return entered;});}
 if(!arrived){{std::lock_guard<std::mutex> lock(gate);release=true;}cv.notify_all();writer.join();throw std::runtime_error("COW callback did not enter");}
 auto probe=std::async(std::launch::async,[&]{
  const bool metadata_ok=f.hub.page_stamp(src)==before&&!f.hub.page_stamp(pending).valid();
  bool denied=false;try{f.hub.begin_page_write(src);}catch(const std::logic_error&){denied=true;}
  return metadata_ok&&denied;
 });
 const bool not_stalled=probe.wait_for(std::chrono::seconds(1))==std::future_status::ready;
 {std::lock_guard<std::mutex> lock(gate);release=true;}cv.notify_all();writer.join();const bool probe_ok=probe.get();
 if(failure)std::rethrow_exception(failure);
 check(not_stalled&&probe_ok,"blocked COW must not hold global Hub metadata lock or certify destination");
 check(done&&f.hub.page_stamp(done).valid()&&f.hub.page_stamp(src)==before,"only completed COW destination gets certified");
 std::vector<uint8_t*> d{done};f.hub.release_pages(d);f.hub.release_pages(hold);hold={src};f.hub.release_pages(hold);
}
int main(){try{changed_bytes_same_tokens_and_address();cookie_recycle_and_stale_completion();cow_immutable_and_failure();publication_and_leased_handoff();blocked_cow_remains_unpublished_without_hub_stall();std::cout<<"resident provenance real-Hub partial-write/ABA/COW/inflight/lease checks PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
