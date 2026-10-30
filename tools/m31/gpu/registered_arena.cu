#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#include "registered_arena.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "ggml-cuda.h"

namespace strata::m31 {
namespace {
using Clock=std::chrono::steady_clock;
void require(bool ok,const std::string &msg){if(!ok)throw std::runtime_error("M31 registered arena: "+msg);}
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(std::string("M31 registered arena: ")+cudaGetErrorString(s));}

}
struct GpuRegisteredArena::Impl {
    struct Item {Piece piece;bool registered=false,read_only=false;double seconds=0,end_seconds=0;};
    int device=-1;
    uintptr_t aligned_base=0;
    size_t chunk_bytes=0;
    Options options;
    std::vector<Item> items;
    std::vector<std::thread> workers;
    std::atomic<size_t> next{0}, completed{0};
    std::atomic<bool> failed{false}, halted{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    Clock::time_point start;
    // Deferred mode: source bytes still missing per piece, and filled pieces
    // waiting for a registration worker.
    std::unique_ptr<std::atomic<size_t>[]> missing;
    std::mutex queue_mutex;
    std::condition_variable queue_changed;
    std::deque<size_t> ready;
    size_t queued=0;
    bool stopping=false;
    void remember_error()noexcept{
        failed.store(true,std::memory_order_release);
        {std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=std::current_exception();}
        stop();
    }
    void stop()noexcept{
        halted.store(true,std::memory_order_release);
        {std::lock_guard<std::mutex> lock(queue_mutex);stopping=true;}
        queue_changed.notify_all();
    }
    // Next piece to register, or items.size() when no further piece will come.
    size_t take()noexcept{
        if(!options.deferred){
            if(halted.load(std::memory_order_acquire))return items.size();
            const size_t index=next.fetch_add(1);return index<items.size()?index:items.size();
        }
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_changed.wait(lock,[&]{return !ready.empty()||stopping||queued==items.size();});
        if(ready.empty()||failed.load(std::memory_order_acquire))return items.size();
        const size_t index=ready.front();ready.pop_front();
        return index;
    }
    void publish(size_t index)noexcept{
        bool last;
        {std::lock_guard<std::mutex> lock(queue_mutex);ready.push_back(index);last=++queued==items.size();}
        // The final publication also satisfies take()'s end-of-input predicate.
        // Wake every idle worker so finish() can join them after the queue drains.
        if(last)queue_changed.notify_all();else queue_changed.notify_one();
    }
    void join()noexcept{for(auto &t:workers)if(t.joinable())t.join();}
    void rollback()noexcept{
        for(auto &item:items)if(item.registered){
            const auto status=cudaHostUnregister(item.piece.registered_ptr);
            if(status!=cudaSuccess)std::fprintf(stderr,"M31 registered arena unregister failed: %s\n",cudaGetErrorString(status));
            if(status==cudaSuccess)item.registered=false;
        }
    }
    ~Impl(){stop();join();rollback();}
};
GpuRegisteredArena::GpuRegisteredArena(ggml_backend_t b,const void *p,size_t n)
    :GpuRegisteredArena(b,p,n,Options{}){}
GpuRegisteredArena::GpuRegisteredArena(ggml_backend_t backend,const void *data,size_t bytes,Options options)
    :base_(data),bytes_(bytes),impl_(std::make_unique<Impl>()){
    require(data&&bytes,"missing arena");
    const uintptr_t logical=reinterpret_cast<uintptr_t>(data);
    require(bytes<=std::numeric_limits<uintptr_t>::max()-logical,"address overflow");
    require(backend,"requires an explicit CUDA backend");
    auto dev=ggml_backend_get_device(backend);
    auto reg=dev?ggml_backend_dev_backend_reg(dev):nullptr;
    require(reg&&std::string(ggml_backend_reg_name(reg))=="CUDA","requires CUDA backend");
    require(options.limit_bytes>0&&bytes<=options.limit_bytes,"arena exceeds explicit registration budget");
    // The CUDA registry enumerates devices in CUDA ordinal order. Match the
    // actual ggml device handle rather than assuming CUDA_VISIBLE_DEVICES=0.
    for(size_t ordinal=0;ordinal<ggml_backend_reg_dev_count(reg);++ordinal)
        if(ggml_backend_reg_dev_get(reg,ordinal)==dev)impl_->device=int(ordinal);
    require(impl_->device>=0,"invalid CUDA backend device");
    ck(cudaSetDevice(impl_->device));
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const size_t page=system_info.dwPageSize;
#else
    const long page_result=sysconf(_SC_PAGESIZE);
    require(page_result>0,"page size lookup failed");
    const size_t page=size_t(page_result);
#endif
    require(page>0,"page size lookup failed");
    require(options.threads>=1&&options.threads<=64,"registration threads must be 1..64");
    require(options.chunk_bytes>=page&&options.chunk_bytes<=size_t(16)<<30&&options.chunk_bytes%page==0,
        "registration chunk must be page aligned and <=16GiB");
    const uintptr_t first=logical-logical%page;
    const uintptr_t end=logical+bytes;
    require(end<=std::numeric_limits<uintptr_t>::max()-(page-1),"end alignment overflow");
    const uintptr_t last=(end+page-1)/page*page;
    require(last-first<=options.limit_bytes,"page-aligned arena exceeds explicit registration budget");
    aligned_base_=first;chunk_bytes_=options.chunk_bytes;
    impl_->aligned_base=first;impl_->chunk_bytes=options.chunk_bytes;impl_->options=std::move(options);
    for(uintptr_t at=first;at<last;){
        const size_t count=std::min<size_t>(impl_->chunk_bytes,last-at);
        const uintptr_t source=std::max(at,logical),source_end=std::min(at+count,end);
        Impl::Item item;item.piece={impl_->items.size(),reinterpret_cast<void *>(at),count,reinterpret_cast<const void *>(source),size_t(source_end-source),size_t(source-logical)};
        impl_->items.push_back(item);at+=count;
    }
    stat_.bytes=bytes;stat_.registered_bytes=size_t(last-first);stat_.pieces=impl_->items.size();
    stat_.threads=unsigned(std::min<size_t>(impl_->options.threads,impl_->items.size()));
    if(impl_->options.deferred){
        impl_->missing.reset(new std::atomic<size_t>[impl_->items.size()]);
        for(size_t i=0;i<impl_->items.size();++i)impl_->missing[i].store(impl_->items[i].piece.source_bytes,std::memory_order_relaxed);
    }
    int ro=0;ck(cudaDeviceGetAttribute(&ro,cudaDevAttrHostRegisterReadOnlySupported,impl_->device));
    impl_->start=Clock::now();
    try{
        for(unsigned worker=0;worker<stat_.threads;++worker)impl_->workers.emplace_back([this,ro]{
            try{
                ck(cudaSetDevice(impl_->device));
                while(!impl_->failed.load(std::memory_order_acquire)){
                    while(impl_->options.may_register&&!impl_->halted.load(std::memory_order_acquire)&&!impl_->options.may_register())
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    const size_t index=impl_->take();if(index>=impl_->items.size())break;
                    auto &item=impl_->items[index];const auto begin=Clock::now();
                    unsigned flags=cudaHostRegisterPortable|(ro?cudaHostRegisterReadOnly:0);
                    auto status=cudaHostRegister(item.piece.registered_ptr,item.piece.registered_bytes,flags);
                    if(status==cudaErrorNotSupported&&ro){(void)cudaGetLastError();flags=cudaHostRegisterPortable;status=cudaHostRegister(item.piece.registered_ptr,item.piece.registered_bytes,flags);}
                    if(status!=cudaSuccess){(void)cudaGetLastError();ck(status);}
                    item.registered=true;item.read_only=(flags&cudaHostRegisterReadOnly)!=0;
                    item.seconds=std::chrono::duration<double>(Clock::now()-begin).count();
                    item.end_seconds=std::chrono::duration<double>(Clock::now()-impl_->start).count();
                    if(impl_->options.on_registered)impl_->options.on_registered(item.piece);
                    impl_->completed.fetch_add(1,std::memory_order_acq_rel);
                }
            }catch(...){impl_->remember_error();}
        });
    }catch(...){impl_->failed.store(true);impl_->stop();impl_->join();impl_->rollback();throw;}
    if(!impl_->options.asynchronous)finish();
}
void GpuRegisteredArena::filled(size_t offset,size_t bytes)noexcept{
    if(!impl_->options.deferred||!bytes)return;
    // Pieces partition [aligned_base, last) in chunk_bytes steps; source
    // offsets are relative to the logical base.
    const size_t shift=size_t(reinterpret_cast<uintptr_t>(base_)-aligned_base_);
    size_t at=offset;
    const size_t end=offset+bytes;
    while(at<end){
        const size_t index=(at+shift)/chunk_bytes_;
        if(index>=impl_->items.size())break;
        const auto &piece=impl_->items[index].piece;
        const size_t piece_end=piece.source_offset+piece.source_bytes;
        const size_t part=std::min(end,piece_end)-at;
        if(impl_->missing[index].fetch_sub(part,std::memory_order_acq_rel)==part)impl_->publish(index);
        at+=part;
    }
}
bool GpuRegisteredArena::settled()const noexcept{
    return ready_||impl_->failed.load(std::memory_order_acquire)||impl_->completed.load(std::memory_order_acquire)==impl_->items.size();
}
void GpuRegisteredArena::abort()noexcept{impl_->failed.store(true,std::memory_order_release);impl_->stop();}
void GpuRegisteredArena::finish(){
    if(ready_)return;
    // A deferred piece that was never completely filled cannot be published.
    if(impl_->options.deferred){std::lock_guard<std::mutex> lock(impl_->queue_mutex);if(impl_->queued!=impl_->items.size())impl_->failed.store(true);}
    if(impl_->failed.load())impl_->stop();
    const auto begin=Clock::now();impl_->join();stat_.finish_wait_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
    if(impl_->failed.load()){
        impl_->rollback();if(impl_->error)std::rethrow_exception(impl_->error);
        throw std::runtime_error("M31 registered arena: registration failed");
    }
    stat_.hardware_read_only=true;
    for(const auto &item:impl_->items){
        require(item.registered,"registration piece was not completed");
        stat_.hardware_read_only=stat_.hardware_read_only&&item.read_only;
        stat_.register_seconds=std::max(stat_.register_seconds,item.end_seconds);
        stat_.register_sum_seconds+=item.seconds;
    }
    ready_=true;
    std::fprintf(stderr,"M31 registered arena: bytes=%zu registered_bytes=%zu pieces=%zu threads=%u seconds=%.6f sum_seconds=%.6f finish_wait_seconds=%.6f hardware_read_only=%d immutable direct H2D\n",stat_.bytes,stat_.registered_bytes,stat_.pieces,stat_.threads,stat_.register_seconds,stat_.register_sum_seconds,stat_.finish_wait_seconds,int(stat_.hardware_read_only));
}
GpuRegisteredArena::~GpuRegisteredArena(){if(impl_){impl_->stop();impl_->join();}}
} // namespace strata::m31
