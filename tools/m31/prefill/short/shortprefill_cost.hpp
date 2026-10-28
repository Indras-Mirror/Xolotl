// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
// Pure per-layer two-lane assignment. Complete experts, never token fragments,
// are moved so each lane retains the original histogram arithmetic contract.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
namespace strata::m31::shortprefill {
struct CostRates {
    double cpu_gbps=220, pcie_gbps=57, cpu_gops=6000, gpu_gops=60000;
    // NVFP4 has no CPU tiled kernel. This factor captures the measured loss;
    // rates are explicit tuning inputs, not hardware peak claims.
    double cpu_untiled_factor=1;

};
struct ExpertCost {size_t bytes=0;int assignments=0;double operations=0;bool resident=false,cpu_tiled=true;};
struct Assignment {std::vector<uint8_t> cpu;double cpu_seconds=0,gpu_seconds=0;uint64_t gpu_cold_bytes=0;};
inline Assignment assign(const std::vector<ExpertCost> &experts,const CostRates &r){
    for(double x:{r.cpu_gbps,r.pcie_gbps,r.cpu_gops,r.gpu_gops,r.cpu_untiled_factor})
        if(!std::isfinite(x)||x<=0)throw std::invalid_argument("invalid short-prefill cost rate");
    Assignment out;out.cpu.assign(experts.size(),0);
    struct Candidate {size_t index;double cpu,gpu,ratio;};std::vector<Candidate> candidates;
    for(size_t e=0;e<experts.size();++e){const auto &v=experts[e];
        if(v.assignments<0||!std::isfinite(v.operations)||v.operations<0)throw std::invalid_argument("invalid expert cost");
        if(!v.assignments)continue;
        const double gpu=(v.resident?0:double(v.bytes)/(r.pcie_gbps*1e9))+v.operations/(r.gpu_gops*1e9);
        if(!std::isfinite(gpu)||!std::isfinite(out.gpu_seconds+gpu))throw std::invalid_argument("nonfinite GPU estimate");
        out.gpu_seconds+=gpu;if(v.resident)continue;
        if(!v.bytes||out.gpu_cold_bytes>UINT64_MAX-v.bytes)throw std::invalid_argument("invalid expert byte count");
        out.gpu_cold_bytes+=v.bytes;
        const double cpu=double(v.bytes)/(r.cpu_gbps*1e9)+v.operations/(r.cpu_gops*1e9)*(v.cpu_tiled?1:r.cpu_untiled_factor);
        if(!std::isfinite(cpu)||!std::isfinite(cpu/gpu))throw std::invalid_argument("nonfinite CPU estimate");
        candidates.push_back({e,cpu,gpu,cpu/gpu});
    }
    // Low-assignment cold experts have the largest CPU benefit. Evaluate the
    // actual max of the lane costs after each move; a fixed histogram cutoff
    // cannot react to cache residency or different quant weight sizes.
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return a.ratio<b.ratio;});
    for(const auto &v:candidates){
        const double old=std::max(out.cpu_seconds,out.gpu_seconds);
        const double next=std::max(out.cpu_seconds+v.cpu,out.gpu_seconds-v.gpu);
        if(next<old){out.cpu[v.index]=1;out.cpu_seconds+=v.cpu;out.gpu_seconds=std::max(0.,out.gpu_seconds-v.gpu);out.gpu_cold_bytes-=experts[v.index].bytes;}
    }
    return out;
}
}
