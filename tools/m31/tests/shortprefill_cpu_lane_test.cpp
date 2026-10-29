// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "prefill/short/shortprefill_cpu_lane.hpp"
#include <stdexcept>
#define assert(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while(false)
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>
using namespace strata::m31;
int main(){
    ShortPrefillCpuLane lane;std::thread::id first,second;
    lane.submit([&]{first=std::this_thread::get_id();});lane.wait();
    lane.submit([&]{second=std::this_thread::get_id();});lane.wait();
    assert(first==second&&first!=std::this_thread::get_id());
    // A lane failure is delivered on join, and the persistent worker recovers.
    lane.submit([]{throw std::runtime_error("test");});bool threw=false;
    try{lane.wait();}catch(const std::runtime_error&){threw=true;}assert(threw);
    int value=0;lane.submit([&]{value=42;});lane.wait();assert(value==42);
    // The submitting thread can advance the independent DMA lane while CPU is
    // blocked; no work is deferred until wait().
    std::mutex mutex;std::condition_variable cv;bool started=false,release=false;
    lane.submit([&]{std::unique_lock<std::mutex> lock(mutex);started=true;cv.notify_one();cv.wait(lock,[&]{return release;});});
    {std::unique_lock<std::mutex> lock(mutex);assert(cv.wait_for(lock,std::chrono::seconds(5),[&]{return started;}));}
    threw=false;try{lane.submit([]{});}catch(const std::logic_error&){threw=true;}assert(threw);
    {std::lock_guard<std::mutex> lock(mutex);release=true;}cv.notify_one();lane.wait();
    // Exceptional stack unwinding drains a borrowed callback before locals die.
    bool finished=false;{ShortPrefillCpuLane local;local.submit([&]{finished=true;throw std::runtime_error("destructor");});}assert(finished);
    std::cout<<"shortprefill CPU lane PASS\n";
}
