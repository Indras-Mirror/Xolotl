// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "prefill/short/shortprefill_cost.hpp"
#include <stdexcept>
#define assert(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while(false)
#include <cmath>
#include <iostream>
using namespace strata::m31::shortprefill;
int main(){
    CostRates r;r.cpu_gops=2200; // Explicit fixture rate; independent of measured runtime defaults.
    std::vector<ExpertCost> x;
    // An idle layer has no work; resident work never needs CPU or PCIe.
    auto a=assign(x,r);assert(a.cpu.empty()&&a.cpu_seconds==0&&a.gpu_seconds==0);
    x={{100000000,2,1e9,true,true},{100000000,1,1e7,false,true}};
    a=assign(x,r);assert(a.cpu[0]==0&&a.cpu[1]==1&&a.gpu_cold_bytes==0);
    // Expensive CPU arithmetic keeps high-assignment experts on GPU, while
    // sparse experts fill the other lane. The split depends on costs, not n.
    x.clear();for(int i=0;i<16;++i)x.push_back({20000000,i<8?1:128,i<8?1e7:1e10,false,true});
    a=assign(x,r);size_t low=0,high=0;for(int i=0;i<16;++i)(i<8?low:high)+=a.cpu[size_t(i)];
    assert(low>0&&high==0);
    // The two-lane objective never exceeds the all-GPU prediction.
    double all_gpu=0;for(auto&e:x)all_gpu+=e.bytes/(r.pcie_gbps*1e9)+e.operations/(r.gpu_gops*1e9);
    assert(std::max(a.cpu_seconds,a.gpu_seconds)<=all_gpu);
    // Faster CPU changes the assignment, and cold-byte accounting is exact.
    auto fast=r;fast.cpu_gops=1e9;fast.cpu_gbps=1e9;auto b=assign(x,fast);
    assert(std::count(b.cpu.begin(),b.cpu.end(),uint8_t(1))>std::count(a.cpu.begin(),a.cpu.end(),uint8_t(1)));
    uint64_t cold=0;for(size_t i=0;i<x.size();++i)if(!b.cpu[i]&&!x[i].resident&&x[i].assignments)cold+=x[i].bytes;
    assert(cold==b.gpu_cold_bytes);
    // Zero-assignment resident entries do not create work, bad costs fail closed.
    x={{999,0,0,true,true}};a=assign(x,r);assert(a.gpu_seconds==0&&a.cpu_seconds==0);
    bool threw=false;try{auto bad=r;bad.cpu_gbps=0;assign(x,bad);}catch(const std::invalid_argument&){threw=true;}assert(threw);
    threw=false;try{x[0].assignments=-1;assign(x,r);}catch(const std::invalid_argument&){threw=true;}assert(threw);
    for(double bad:{double(NAN),double(INFINITY),-1.,0.}){threw=false;try{auto rate=r;rate.cpu_gops=bad;assign({},rate);}catch(const std::invalid_argument&){threw=true;}assert(threw);}
    x={{20,1,1e308,false,true}};threw=false;try{auto rate=r;rate.gpu_gops=1e-300;assign(x,rate);}catch(const std::invalid_argument&){threw=true;}assert(threw);
    // Equal candidates balance deterministically across both independent lanes.
    x.assign(16,ExpertCost{20000000,1,1e7,false,true});a=assign(x,r);
    assert(std::count(a.cpu.begin(),a.cpu.end(),uint8_t(1))>0);
    assert(std::count(a.cpu.begin(),a.cpu.end(),uint8_t(0))>0);
    auto again=assign(x,r);assert(again.cpu==a.cpu);
    std::cout<<"shortprefill cost PASS\n";
}
