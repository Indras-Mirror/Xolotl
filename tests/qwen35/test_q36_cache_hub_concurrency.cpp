// Actual Hub CPU concurrency, not CUDA math: blocked writes/copies stay unpublished,
// leases survive eviction/vector changes, six logical callers migrate across owners.
#include "../../src/qwen35/q36_cache_hub.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>
using namespace strata::q36::cache;
using namespace std::chrono_literals;
struct Gate {
 std::promise<void> entered,release;std::shared_future<void> go=release.get_future().share();
 void block(){entered.set_value();go.wait();}
};
struct Fixture {
 Hub hub{128};std::array<std::array<uint8_t,64>,128> snapshots{};std::array<std::array<uint8_t,128>,128> pages{};
 Fixture(){hub.snapshot_bytes=64;hub.page_bytes=128;hub.n_pages=pages.size();for(size_t i=0;i<128;++i){hub.snaps[i].host=snapshots[i].data();hub.free_pages.push_back(pages[i].data());}}
 Hub::Checkout get(uint64_t id){std::lock_guard<std::recursive_mutex> lock(hub.mutex);for(size_t i=0;i<hub.parked.size();++i)if(hub.parked[i].id==id)return hub.checkout(i);return {};}
 uint64_t publish(int caller,int revision,int owner=-1,uint8_t* page=nullptr){
  Hub::SnapshotWrite writer;
  {std::lock_guard<std::recursive_mutex> lock(hub.mutex);for(auto& sn:hub.snaps)if((writer=hub.reserve_snapshot(sn)))break;}
  assert(writer);std::memset(writer.data(),caller,64);writer.data()[1]=uint8_t(revision);
  assert(writer.commit(16,uint64_t(caller*100000+revision),0,[]{return true;}));
  Conversation c;c.tok.assign(16,caller);c.tok[1]=revision;c.val.assign(16,caller);c.val[1]=revision;c.mirrored=16;c.origin_slot=owner;
  c.pages.push_back(page?page:hub.take_page());assert(c.pages[0]);std::memset(c.pages[0],caller,128);c.pages[0][1]=uint8_t(revision);
  c.states.push_back(writer.get());hub.retain_states(c.states);return hub.publish(std::move(c));
 }
 void empty(){while(hub.evict_lru()){}assert(hub.parked.empty()&&hub.page_refs.empty()&&hub.free_pages.size()==128);for(auto& s:hub.snaps)assert(!s.refs&&!s.pins&&!s.writing);}
};
void blocked_snapshot(){
 Fixture f;auto root=f.publish(7,0);auto& h=f.hub;auto& pending=h.snaps[1];Gate copy,retain;std::promise<uint64_t> published;
 auto writer=std::async(std::launch::async,[&]{
  auto w=h.reserve_snapshot(pending);assert(w);std::memset(w.data(),23,32);
  assert(w.commit(16,230001,0,[&]{copy.block();std::memset(w.data()+32,23,32);return true;}));
  retain.block();Conversation c;c.tok.assign(16,23);c.val.assign(16,23);c.mirrored=16;c.pages.push_back(h.take_page());c.states.push_back(w.get());h.retain_states(c.states);published.set_value(h.publish(std::move(c)));
 });
 copy.entered.get_future().wait();
 auto observer=std::async(std::launch::async,[&]{
  auto lease=f.get(root);assert(lease&&lease.conv.pages[0][0]==7);
  {std::lock_guard<std::recursive_mutex> lock(h.mutex);assert(pending.writing&&!h.ready(pending));assert(!h.reserve_snapshot(pending));}
  bool refused=false;try{Hub::SnapPin pin(h,pending);}catch(const std::logic_error&){refused=true;}assert(refused);
  assert(!h.evict_lru());return true;
 });
 bool unlocked=observer.wait_for(1s)==std::future_status::ready;copy.release.set_value();assert(unlocked&&observer.get());
 retain.entered.get_future().wait();
 {std::lock_guard<std::recursive_mutex> lock(h.mutex);assert(h.ready(pending)&&pending.pins==1&&pending.refs==0);assert(!h.reserve_snapshot(pending));}
 Hub::SnapPin reader(h,pending);retain.release.set_value();writer.get();auto id=published.get_future().get();
 {auto c=f.get(id);assert(c);for(size_t i=0;i<64;++i)assert(c.conv.states[0]->host[i]==23);}
 while(h.evict_lru()){}assert(!h.reserve_snapshot(pending));reader.reset();{auto w=h.reserve_snapshot(pending);assert(w);}f.empty();
}
void blocked_cow(){
 Fixture f;auto id=f.publish(9,1);auto first=f.get(id);auto* source=first.conv.pages[0];Gate copy;
 auto copying=std::async(std::launch::async,[&]{return f.hub.cow_page(source,[&](uint8_t* dst,const uint8_t* src,size_t bytes){std::memcpy(dst,src,bytes/2);copy.block();std::memcpy(dst+bytes/2,src+bytes/2,bytes-bytes/2);return true;});});
 copy.entered.get_future().wait();first.reset();
 auto observer=std::async(std::launch::async,[&]{
  {auto other=f.get(id);assert(other&&other.conv.pages[0]==source&&source[0]==9&&source[1]==1);Hub::SnapPin pin(f.hub,*other.conv.states[0]);assert(!f.hub.evict_lru());}
  assert(f.hub.evict_lru());assert(!f.get(id));
  std::lock_guard<std::recursive_mutex> lock(f.hub.mutex);assert(f.hub.page_refs.at(source)==1);assert(std::find(f.hub.free_pages.begin(),f.hub.free_pages.end(),source)==f.hub.free_pages.end());return true;
 });
 bool unlocked=observer.wait_for(1s)==std::future_status::ready;copy.release.set_value();assert(unlocked&&observer.get());auto* dst=copying.get();assert(dst&&dst!=source&&dst[0]==9&&dst[1]==1);
 {std::lock_guard<std::recursive_mutex> lock(f.hub.mutex);assert(!f.hub.page_refs.count(source));}
 std::vector<uint8_t*> own{dst};f.hub.release_pages(own);f.empty();
}
void handoffs(size_t owners){
 Fixture f;std::array<uint64_t,6> current{};for(int c=0;c<6;++c)current[c]=f.publish(c,0);
 std::atomic<int> cross_owner{0};std::array<std::atomic<unsigned>,6> seen{};std::vector<std::thread> workers;
 for(size_t owner=0;owner<owners;++owner)workers.emplace_back([&,owner]{
  for(int round=0;round<12;++round){int caller=(round+int(owner))%6;Hub::Checkout lease;
   {std::lock_guard<std::recursive_mutex> lock(f.hub.mutex);lease=f.get(current[caller]);}
   assert(lease&&lease.conv.val[0]==uint64_t(caller));auto* source=lease.conv.pages[0];assert(source[0]==caller&&lease.conv.states[0]->host[0]==caller);
   if(lease.conv.origin_slot>=0&&lease.conv.origin_slot!=int(owner))++cross_owner;
   seen[caller].fetch_or(1u<<owner);uint8_t before=source[1];auto* dst=f.hub.cow_page(source,[](uint8_t* d,const uint8_t* s,size_t n){std::memcpy(d,s,n);return true;});assert(dst&&source[1]==before);
   int revision=1+round+int(owner)*16;auto id=f.publish(caller,revision,int(owner),dst);
   {std::lock_guard<std::recursive_mutex> lock(f.hub.mutex);current[caller]=id;while(f.hub.parked.size()>24)assert(f.hub.evict_lru());}
  }
 });
 for(auto& worker:workers)worker.join();
 for(int caller=0;caller<6;++caller){auto c=f.get(current[caller]);assert(c&&c.conv.val[0]==uint64_t(caller));assert(c.conv.pages[0][0]==caller&&c.conv.states[0]->host[0]==caller);assert(c.conv.pages[0][1]==c.conv.states[0]->host[1]);assert(seen[caller].load()==((1u<<owners)-1));}
 if(owners>1)assert(cross_owner>0);f.empty();
}
int main(){blocked_snapshot();blocked_cow();handoffs(1);handoffs(4);std::cout<<"GlobalCacheHub concurrency PASS: blocked writing/pins, COW+lookup+eviction,6 callers across1/4 owners\n";}
