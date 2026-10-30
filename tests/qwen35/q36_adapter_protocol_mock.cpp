// CPU-only protocol stand-in; includes REAL adapter, no CUDA/model/GPU.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
#define CK(x) do { if((x)!=0) std::abort(); } while(0)
int cudaMemGetInfo(size_t* f,size_t* t){*f=std::getenv("MOCK_LOW_VRAM")?0:(2ull<<30);*t=2ull<<30;return 0;}
int cudaSetDevice(int){return 0;} int cudaGetDevice(int* p){*p=0;return 0;} int cudaStreamSynchronize(int){return 0;}
struct Q36Channel { std::function<bool(std::string&)> read; std::function<void(const std::string&)> write; };
thread_local Q36Channel* q36_channel=nullptr;
std::vector<int> parse_ids(const char* s){std::vector<int> r; std::istringstream i(s); std::string t; while(std::getline(i,t,','))if(!t.empty())r.push_back(std::atoi(t.c_str()));return r;}
std::atomic<size_t> mock_used{0};
namespace strata::q36::kv { struct Budget {size_t cap(){return std::getenv("MOCK_SMALL_CAP")?1024:(std::getenv("MOCK_RECLAIM")?2400000:(1ull<<30));} size_t used(){return std::getenv("MOCK_PRESSURE")?cap():(std::getenv("MOCK_RECLAIM")?mock_used.load():0);}}; struct Driver {}; }
#include "../../src/qwen35/q36_cache_hub.hpp"
#include "../../src/qwen35/q36_resident_prefix.hpp"
std::atomic<int> mock_snap_allocations{0},mock_page_allocations{0};
struct Model{bool hybrid=false;struct{long long n_embd=2048;}g;};
struct Arena {size_t held=0;size_t committed_bytes(){return held;} size_t bytes_for(size_t n){return n*16;} bool trim(size_t){if(std::getenv("MOCK_RECLAIM"))mock_used.fetch_sub(held);held=0;return true;}};
struct Engine {
    bool resident_prefix_reuse=false, resident_prefix_verify=false;
    int64_t resident_ram_equal_end=0;
    strata::q36::resident::Prefix resident_prefix;
    std::shared_ptr<strata::q36::cache::Hub> cache_hub;std::shared_ptr<int> shared_prefill;Model* m=nullptr; int ctx=262144,tmax=512,st=0,mtp_d=3,max_snaps=0,batch_slot=-1; bool use_graph=false,dynamic_attention=false;unsigned long long batch_epoch=0;
    strata::q36::kv::Budget* kv_budget=nullptr; strata::q36::kv::Driver* kv_driver=nullptr;
    std::unique_ptr<Arena> kv_arena=std::make_unique<Arena>(); std::atomic<bool> cancellation{false}; std::atomic<bool>* cancel=&cancellation;
    std::function<bool(int,int64_t,int,bool,const float* const*)> batch_dispatch;
    std::function<bool(size_t)> batch_admit; std::function<void()> batch_idle,batch_begin;std::function<void(long long,size_t)> batch_parsed;
    void init(Model& model,int c,int t){m=&model;ctx=c;tmax=t;if(t!=512 && t!=64 && t!=256)throw std::runtime_error("unexpected prefill chunk");if(std::getenv("MOCK_REQUIRE_CHUNK") && t!=512 && t!=std::atoi(std::getenv("MOCK_REQUIRE_CHUNK")))throw std::runtime_error("owner init ignored requested prefill chunk");} void alloc_snaps(){if(cache_hub && max_snaps>0 && !cache_hub->snaps_initialized){cache_hub->snaps_initialized=true;cache_hub->snapshot_bytes=64;++mock_snap_allocations;}} void alloc_pages(size_t budget){if(cache_hub && budget>0 && !cache_hub->pages_initialized){cache_hub->pages_initialized=true;cache_hub->page_bytes=256;cache_hub->n_pages=budget/256;++mock_page_allocations;}}
    void reset(){} void warmup(){} void alloc_prefill_rollback(){}
    bool resize_attention(size_t){return true;}
    bool ensure_kv(size_t n){if(std::getenv("MOCK_MAP_FAIL"))return false;if(std::getenv("MOCK_RECLAIM")){size_t target=kv_arena->bytes_for(n),delta=target>kv_arena->held?target-kv_arena->held:0;if(delta>kv_budget->cap()-kv_budget->used())return false;mock_used.fetch_add(delta);kv_arena->held=std::max(target,kv_arena->held);}return true;}
    struct BatchGroup {Engine* state;int first,count;int64_t pos;bool verify;int heads;};
    bool batch_forward(const std::vector<BatchGroup>& groups,const float* const*,std::function<void(Engine*)> canceled){
        if(std::getenv("MOCK_EXECUTOR_THROW"))throw std::runtime_error("mock fatal executor failure");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for(auto& g:groups){if(std::getenv("MOCK_RECLAIM") && !g.state->kv_arena->held)throw std::runtime_error("active owner KV reclaimed");if(g.state->cancel->load())canceled(g.state);}
        return true;
    }
};
struct ImageCache {struct Img{int n=3,nx=1,ny=1;};std::recursive_mutex mutex;Img im;Img* find(uint64_t key){return key==42?&im:nullptr;}};
void serve(Engine& e,ImageCache&,int,size_t park_bytes,const std::string&){
    if(e.cache_hub && ((e.max_snaps>0 && !e.cache_hub->snaps_initialized) || (park_bytes>0 && !e.cache_hub->pages_initialized)))throw std::runtime_error("shared host pool was not initialized before workers");
    e.alloc_snaps();e.alloc_pages(park_bytes);
    Q36Channel* c=q36_channel; std::thread generating;
    std::string line;
    if(e.batch_slot==3 && std::getenv("MOCK_DELAY_READY"))std::this_thread::sleep_for(std::chrono::milliseconds(150));
    c->write("INFO mock=1\n"); c->write("READY 262144 stop\n");
    while(c->read(line)){
        if(line=="STOP"){e.cancel->store(true);continue;}
        if(generating.joinable()) generating.join();
        e.cancel->store(false);
        generating=std::thread([&,line]{
            int count=20;std::sscanf(line.c_str(),"GEN %d",&count);
            if(e.batch_begin)e.batch_begin();if(e.batch_parsed)e.batch_parsed(count,3);
            if(line.find("777")!=std::string::npos){c->write("MOCK cleaning\n");std::this_thread::sleep_for(std::chrono::milliseconds(30));c->write("ERR mock request failure\n");e.batch_idle();return;}
            c->write("MOCK admission\n");
            if((e.batch_slot==0 && std::getenv("MOCK_SLOW_ADMIT")) || (e.batch_slot==1 && std::getenv("MOCK_SLOW_ADMIT_SLOT1")))std::this_thread::sleep_for(std::chrono::milliseconds(150));
            if(e.batch_admit && !e.batch_admit(100000)){
                if(e.cancel->load())c->write("DONE 0 10 0 0 cancel 0 0 0\n");
                if(!e.cancel->load())c->write("ERR mock admission refused\n");
                e.batch_idle();
                return;
            }
            c->write("RESUME 7 0 0\n"); c->write("PP 3 10 1.5\n");
            int made=0;
            for(;made<count && !e.cancel->load();++made){
                if(!e.batch_dispatch(1,made,1,true,nullptr))break;
                c->write("T "+std::to_string(100+e.batch_slot)+"\n");
                if(made==2 && line.find("888")!=std::string::npos){c->write("ERR mock postadmit failure\n");std::this_thread::sleep_for(std::chrono::milliseconds(20));e.batch_idle();return;}
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
            }
            c->write("DONE "+std::to_string(made)+" 10 1.5 25 "+(e.cancel->load()?"cancel":"length")+" 0 0 7\n");
            // Busy must remain set until idle callback has done native cleanup.
            std::this_thread::sleep_for(std::chrono::milliseconds(15));e.batch_idle();
        });
    }
    e.cancel->store(true);if(generating.joinable())generating.join();
}
#include "../../src/qwen35/q36_parallel.hpp"
int main(){std::set_terminate([]{std::_Exit(86);});Model model;Engine scratch;scratch.init(model,262144,512);scratch.resident_prefix_reuse=std::getenv("MOCK_RESIDENT_PREFIX")!=nullptr;ImageCache images;strata::q36::kv::Budget budget;strata::q36::kv::Driver driver;int result=serve_parallel(scratch,images,"mock",std::getenv("MOCK_SLOTS")?std::atoi(std::getenv("MOCK_SLOTS")):4,1,std::getenv("MOCK_PREFILL_CHUNK")?std::atoi(std::getenv("MOCK_PREFILL_CHUNK")):64,std::getenv("MOCK_HOST_PARK")?1048576:0,std::getenv("MOCK_NO_SNAPS")?0:1,budget,driver,false);std::fprintf(stderr,"MOCK_HUB_ALLOCS snapshots=%d pages=%d\n",mock_snap_allocations.load(),mock_page_allocations.load());return result;}
