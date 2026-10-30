#!/usr/bin/env python3
"""CPU-only regression using the actual CUDA source's queue methods.

Run under the project's build lock. --prepare-only emits C++ without compiling.
Outputs are retained in --out; this test never loads CUDA or a model.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--cxx', default='c++')
args = parser.parse_args()
source = (Path(__file__).resolve().parents[1] / 'gpu/registered_arena.cu').read_text()

def method(signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]

methods = '\n'.join(method(s) for s in ('size_t take()noexcept', 'void publish(size_t index)noexcept'))
program = r'''
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
using namespace std::chrono_literals;
struct ObservedCondition {
    std::condition_variable cv;
    std::atomic<unsigned> entered{0};
    template<class Predicate> void wait(std::unique_lock<std::mutex>& lock, Predicate pred) {
        ++entered;
        cv.wait(lock, pred);
    }
    void notify_one() { cv.notify_one(); }
    void notify_all() { cv.notify_all(); }
};
struct Queue {
    struct { bool deferred=true; } options;
    std::vector<int> items=std::vector<int>(1);
    std::atomic<bool> halted{false},failed{false};
    std::atomic<size_t> next{0};
    std::mutex queue_mutex;
    ObservedCondition queue_changed;
    std::deque<size_t> ready;
    size_t queued=0;
    bool stopping=false;
METHODS
};
int main() {
    Queue q;
    constexpr unsigned count=4;
    std::atomic<unsigned> done{0}, pieces{0};
    std::vector<std::thread> workers;
    for(unsigned i=0;i<count;++i)workers.emplace_back([&] {
        while(q.take()<q.items.size())++pieces;
        ++done;
    });
    auto deadline=std::chrono::steady_clock::now()+5s;
    while(q.queue_changed.entered<count && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(1ms);
    bool waiting=q.queue_changed.entered==count;
    // publish() acquires queue_mutex: all four waits have atomically released it.
    q.publish(0);
    deadline=std::chrono::steady_clock::now()+500ms;
    while(done<count && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(1ms);
    const unsigned completed=done.load();
    // Rescue failing implementations so the test reports failure instead of hanging.
    {std::lock_guard<std::mutex> lock(q.queue_mutex);q.stopping=true;}
    q.queue_changed.notify_all();
    for(auto& t:workers)t.join();
    const bool pass=waiting && completed==count && pieces==1;
    std::printf("waiting=%d terminated=%u/%u pieces=%u %s\n",waiting,completed,count,pieces.load(),pass?"PASS":"FAIL");
    return pass?0:1;
}
'''.replace('METHODS', methods)
args.out.mkdir(parents=True, exist_ok=True)
cpp = args.out / 'registered-arena-queue.cpp'
cpp.write_text(program)
if not args.prepare_only:
    exe = args.out / 'registered-arena-queue'
    subprocess.run([args.cxx, '-std=c++17', '-O2', '-pthread', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
else:
    print('Prepared actual-source queue regression; no compile or GPU launched:', cpp)
