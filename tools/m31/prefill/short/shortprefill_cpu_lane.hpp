// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
namespace strata::m31 {
// One lazy persistent submission thread. The CPU executor it calls still owns
// its original persistent workers; this thread is its serialized master only.
class ShortPrefillCpuLane {
    std::mutex mutex_;std::condition_variable ready_,done_;
    std::function<void()> job_;std::exception_ptr error_;
    bool stop_=false,busy_=false;std::thread worker_;
    void run(){
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;){ready_.wait(lock,[&]{return stop_||bool(job_);});if(stop_)return;
            auto job=std::move(job_);lock.unlock();std::exception_ptr error;
            try{job();}catch(...){error=std::current_exception();}
            lock.lock();error_=error;busy_=false;done_.notify_all();}
    }
public:
    ShortPrefillCpuLane():worker_([this]{run();}){}
    ~ShortPrefillCpuLane(){drain();{std::lock_guard<std::mutex> lock(mutex_);stop_=true;}ready_.notify_one();worker_.join();}
    ShortPrefillCpuLane(const ShortPrefillCpuLane&)=delete;
    ShortPrefillCpuLane& operator=(const ShortPrefillCpuLane&)=delete;
    void submit(std::function<void()> job){
        if(!job)throw std::invalid_argument("empty short-prefill CPU job");
        {std::lock_guard<std::mutex> lock(mutex_);if(busy_)throw std::logic_error("overlapping CPU submissions");
            error_=nullptr;busy_=true;job_=std::move(job);}ready_.notify_one();
    }
    void wait(){std::exception_ptr error;{std::unique_lock<std::mutex> lock(mutex_);done_.wait(lock,[&]{return !busy_;});error=error_;error_=nullptr;}if(error)std::rethrow_exception(error);}
    void drain() noexcept {try{wait();}catch(...){}}
};
}
