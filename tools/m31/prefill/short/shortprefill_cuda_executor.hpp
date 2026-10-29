// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
#include "shortprefill_cuda_math_cpu.hpp"
#include "shortprefill_cuda_nvfp4_cpu.hpp"
#include "shortprefill_gpu_quant.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../cpu/task_barrier.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata::m31 {
// Native CUDA activation quantization on CPU. The Engine's existing persistent
// pool executes one custom graph node; there are no per-layer worker threads.
// Contributions retain the caller's compact route order and remain unreduced.
class ShortPrefillCudaExecutor {
public:
    struct Stats {
        uint64_t forwards=0,assignments=0,graph_builds=0,graph_hits=0;
        double setup_seconds=0,graph_build_seconds=0,pack_seconds=0,compute_seconds=0,read_seconds=0;
        double reduce_seconds=0;
        double correction_seconds=0; // included in compute_seconds
        uint64_t correction_groups=0;
        size_t retained_correction_bytes=0;
        uint64_t gpu_quant_calls=0,gpu_quant_h2d_bytes=0,gpu_quant_d2h_bytes=0;
        double gpu_quant_seconds=0;
        uint64_t gpu_quant_reserve_calls=0,gpu_quant_allocations=0,gpu_quant_runtime_allocations=0;
        double gpu_quant_reserve_seconds=0,gpu_quant_host_copy_seconds=0,gpu_quant_submit_seconds=0;
        double gpu_quant_wait_seconds=0,gpu_quant_unpack_seconds=0;
        size_t gpu_quant_reserved_max_values=0,gpu_quant_reserved_max_rows=0;
        int gpu_quant_stream_priority=0;
        double gate_thread_max_seconds=0,gate_thread_median_seconds=0;
        double down_thread_max_seconds=0,down_thread_median_seconds=0;
        size_t retained_graphs=0,retained_scratch_bytes=0;
    };
private:
    using Clock=std::chrono::steady_clock;
    struct Task {int role=0,expert=0;size_t first=0,last=0,assignment_first=0,assignment_last=0;};
    ggml_backend_t backend_;
    ggml_threadpool_t pool_;
    unsigned threads_;
    CpuTaskBarrier barrier_;
    ggml_context *context_=nullptr;
    ggml_gallocr_t allocator_=nullptr;
    ggml_cgraph *graph_=nullptr;
    ggml_tensor *output_=nullptr;
    size_t capacity_=0,embd_=0,ff_=0,n_=0,count_=0,used_=0;
    std::array<const ggml_tensor*,3> weight_{};
    std::array<const float*,3> scale_{};
    const float *input_=nullptr,*routes_weights_=nullptr;
    const int32_t *ids_=nullptr;
    bool reduce_=false;
    float *reduced_output_=nullptr;
    Clock::time_point reduction_started_{};
    std::atomic<bool> compute_failed_{false};
    std::vector<std::exception_ptr> worker_errors_;
    std::unique_ptr<ShortPrefillGpuQuantizer> gpu_quantizer_;
    bool gpu_quant_=false,gpu_oai_=false;
    bool default_gpu_exact_=false,retain_debug_hidden_=false;
    bool dynamic_schedule_=true,profile_tasks_=false;
    bool prepared_corrections_=true;
    size_t schedule_chunk_=4;
    std::atomic<size_t> gate_next_{0},down_next_{0};
    std::vector<double> gate_thread_seconds_,down_thread_seconds_;
    int active_threads_=0;
    std::vector<size_t> routes_;
    std::vector<std::vector<size_t>> experts_;
    std::vector<Task> gate_tasks_,down_tasks_;
    std::array<std::vector<cuda_math_cpu::Q8Mmq32>,3> quant_;
    std::array<std::vector<int32_t>,3> corrections_;
    std::array<std::vector<cuda_nvfp4_cpu::Activation>,3> nv_quant_;
    std::vector<float> gate_,up_,hidden_;
    Stats stats_;

    static void require(bool ok,const char *reason){if(!ok)throw std::invalid_argument(reason);}
    static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
    static std::pair<size_t,size_t> range(size_t size,int ith,int nth){
        return {size*size_t(ith)/size_t(nth),size*size_t(ith+1)/size_t(nth)};
    }
    static size_t product(size_t a,size_t b){
        require(!b||a<=std::numeric_limits<size_t>::max()/b,"M31 CUDA CPU executor size overflow");return a*b;
    }
    bool nv(int role)const{return weight_[size_t(role)]->type==GGML_TYPE_NVFP4;}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx2"))) static int32_t correction_avx2(const int8_t *values){
        const auto bytes=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
        const auto shifted=_mm256_xor_si256(bytes,_mm256_set1_epi8(char(0x80)));
        const auto sums=_mm256_sad_epu8(shifted,_mm256_setzero_si256());
        const auto halves=_mm_add_epi64(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1));
        const int64_t total=_mm_cvtsi128_si64(halves)+_mm_extract_epi64(halves,1);
        return int32_t(total-32*128)*128;
    }
#endif
    static int32_t correction(const int8_t *values){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        static const bool avx2=__builtin_cpu_supports("avx2");
        if(avx2)return correction_avx2(values);
#endif
        int32_t sum=0;for(size_t i=0;i<32;++i)sum+=int32_t(values[i]);return sum*128;
    }
    void prepare_corrections(int role,int ith,int nth){
        const auto &quant=quant_[size_t(role)];
        auto &output=corrections_[size_t(role)];
        const auto [first,last]=range(quant.size(),ith,nth);
        for(size_t group=first;group<last;++group)output[group]=correction(quant[group].qs);
    }
    void quantize(int role,const float *input,size_t columns,size_t row){
        if(nv(role))cuda_nvfp4_cpu::quantize(input,columns,nv_quant_[size_t(role)][row]);
        else cuda_math_cpu::quantize_mmq_d4_into(input,columns,quant_[size_t(role)].data()+row*(columns/32));
    }
    void quantize_gpu(int role,const float *input,size_t columns,size_t rows,bool oai=false){
        if(nv(role)){
            if(oai)gpu_quantizer_->oai_quantize_nvfp4(gate_.data(),up_.data(),columns,rows,nv_quant_[size_t(role)]);
            else gpu_quantizer_->quantize_nvfp4(input,columns,rows,nv_quant_[size_t(role)]);
        }else{
            if(oai)gpu_quantizer_->oai_quantize_q8(weight_[size_t(role)]->type,gate_.data(),up_.data(),columns,rows,quant_[size_t(role)]);
            else gpu_quantizer_->quantize_q8(weight_[size_t(role)]->type,input,columns,rows,quant_[size_t(role)]);
        }
    }
    void configure_gpu_quantizer(){
        gpu_quant_=default_gpu_exact_;
        gpu_oai_=default_gpu_exact_;
        if(gpu_quant_&&!gpu_quantizer_){
            require(ShortPrefillGpuQuantizer::available(),"M31 GPU activation quantization requires a CUDA build");
            gpu_quantizer_=std::make_unique<ShortPrefillGpuQuantizer>();
        }
    }
    void refresh_gpu_quant_stats(){
        if(!gpu_quantizer_)return;
        const auto &stats=gpu_quantizer_->stats();
        stats_.gpu_quant_calls=stats.calls;stats_.gpu_quant_h2d_bytes=stats.h2d_bytes;
        stats_.gpu_quant_d2h_bytes=stats.d2h_bytes;stats_.gpu_quant_seconds=stats.wall_seconds;
        stats_.gpu_quant_reserve_calls=stats.reserve_calls;stats_.gpu_quant_allocations=stats.allocations;
        stats_.gpu_quant_runtime_allocations=stats.runtime_allocations;stats_.gpu_quant_reserve_seconds=stats.reserve_seconds;
        stats_.gpu_quant_host_copy_seconds=stats.host_copy_seconds;stats_.gpu_quant_submit_seconds=stats.submit_seconds;
        stats_.gpu_quant_wait_seconds=stats.wait_seconds;stats_.gpu_quant_unpack_seconds=stats.unpack_seconds;
        stats_.gpu_quant_reserved_max_values=stats.reserved_max_values;stats_.gpu_quant_reserved_max_rows=stats.reserved_max_rows;
        stats_.gpu_quant_stream_priority=stats.stream_priority;
    }
    float dot(int role,const void *weight,size_t columns,size_t activation)const{
        const int source=role==1&&nv(0)==nv(1)?0:role;
        if(nv(role))return cuda_nvfp4_cpu::dot(weight,nv_quant_[size_t(source)][activation],columns);
        return cuda_math_cpu::dot_mmq(weight_[size_t(role)]->type,weight,columns,
            quant_[size_t(source)].data()+activation*(columns/32));
    }
    void matrix_task(const Task &task,float *destination,size_t columns,size_t rows){
        const auto *w=weight_[size_t(task.role)];
        const auto *base=static_cast<const uint8_t*>(w->data)+size_t(task.expert)*w->nb[2];
        const float scale=scale_[size_t(task.role)]?scale_[size_t(task.role)][size_t(task.expert)]:1.f;
        const auto &assignments=experts_[size_t(task.expert)];
        // Keep a small set of complete weight rows hot across this expert's
        // assignments. No expert is split across devices or rank reductions.
        {
            const size_t first=task.assignment_first,last=task.assignment_last;
            if(!nv(task.role)){
                const int source=task.role==1&&nv(0)==nv(1)?0:task.role;
                std::array<const cuda_math_cpu::Q8Mmq32*,16> activations{};
                std::array<const int32_t*,16> prepared{};
                for(size_t item=first;item<last;++item){
                    const size_t compact=assignments[item],route=routes_[compact];
                    const size_t activation=task.role==2?compact:route/used_;
                    activations[item-first]=quant_[size_t(source)].data()+activation*(columns/32);
                    if(prepared_corrections_)prepared[item-first]=corrections_[size_t(source)].data()+activation*(columns/32);
                }
                // The kernel writes every active [token,row] cell. Masked
                // row/token tails are never read by the scatter below.
                std::array<float,16*16> values;
                cuda_math_cpu::mul_mat_quantized_tile(w->type,base+task.first*w->nb[1],w->nb[1],
                    task.last-task.first,columns,activations.data(),last-first,values.data(),16,
                    prepared_corrections_?prepared.data():nullptr);
                for(size_t item=first;item<last;++item){const size_t compact=assignments[item],route=routes_[compact];
                    for(size_t row=task.first;row<task.last;++row){
                        float result=values[(item-first)*16+row-task.first];
                        if(scale_[size_t(task.role)])result*=scale;
                        if(task.role==2)result*=routes_weights_[route];
                        require(std::isfinite(result),"M31 CUDA CPU executor nonfinite matrix result");
                        destination[compact*rows+row]=result;
                    }
                }
                return;
            }
            for(size_t row=task.first;row<task.last;++row){
                const auto *weight=base+row*w->nb[1];
                for(size_t item=first;item<last;++item){
                    const size_t compact=assignments[item],route=routes_[compact];
                    const size_t activation=task.role==2?compact:route/used_;
                    float result=dot(task.role,weight,columns,activation);
                    if(scale_[size_t(task.role)])result*=scale;
                    if(task.role==2)result*=routes_weights_[route];
                    require(std::isfinite(result),"M31 CUDA CPU executor nonfinite matrix result");
                    destination[compact*rows+row]=result;
                }
            }
        }
    }
    static void compute(ggml_tensor *output,int ith,int nth,void *opaque){
        auto &self=*static_cast<ShortPrefillCudaExecutor*>(opaque);
        if(ith==0)self.active_threads_=nth;
        // No exception may escape a ggml worker while its peers await a
        // barrier. Each stage catches locally; all workers still visit every
        // barrier and the submitting thread rethrows after graph completion.
        auto guarded=[&](auto &&work){
            if(self.compute_failed_.load(std::memory_order_acquire))return;
            try{work();}catch(...){
                self.worker_errors_[size_t(ith)]=std::current_exception();
                self.compute_failed_.store(true,std::memory_order_release);
            }
        };
        auto [first,last]=range(self.n_,ith,nth);
        const bool same_input_format=self.nv(0)==self.nv(1);
        guarded([&]{
            if(self.gpu_quant_){
                if(ith==0){self.quantize_gpu(0,self.input_,self.embd_,self.n_);
                    if(!same_input_format)self.quantize_gpu(1,self.input_,self.embd_,self.n_);}
            }else for(size_t token=first;token<last;++token){
                self.quantize(0,self.input_+token*self.embd_,self.embd_,token);
                if(!same_input_format)self.quantize(1,self.input_+token*self.embd_,self.embd_,token);
            }
        });
        self.barrier_.wait(unsigned(nth));
        if(self.prepared_corrections_&&(!self.nv(0)||!self.nv(1))){
            const auto correction_started=Clock::now();
            guarded([&]{
                if(!self.nv(0))self.prepare_corrections(0,ith,nth);
                if(!same_input_format&&!self.nv(1))self.prepare_corrections(1,ith,nth);
            });
            self.barrier_.wait(unsigned(nth));
            if(ith==0)self.stats_.correction_seconds+=seconds(correction_started);
        }
        // Both IQ types use exactly the same MMQ D4 activation format.
        // Alias role1 via dot()'s source selection without copying token rows.
        std::tie(first,last)=range(self.gate_tasks_.size(),ith,nth);
        const auto gate_started=self.profile_tasks_?Clock::now():Clock::time_point{};
        guarded([&]{
            auto run=[&](size_t begin,size_t end){for(size_t task=begin;task<end;++task){
                const auto &work=self.gate_tasks_[task];
                self.matrix_task(work,work.role?self.up_.data():self.gate_.data(),self.embd_,self.ff_);
            }};
            if(self.dynamic_schedule_){
                for(;;){const size_t begin=self.gate_next_.fetch_add(self.schedule_chunk_,std::memory_order_relaxed);
                    if(begin>=self.gate_tasks_.size()||self.compute_failed_.load(std::memory_order_relaxed))break;
                    run(begin,std::min(begin+self.schedule_chunk_,self.gate_tasks_.size()));}
            }else run(first,last);
        });
        if(self.profile_tasks_)self.gate_thread_seconds_[size_t(ith)]=seconds(gate_started);
        self.barrier_.wait(unsigned(nth));
        std::tie(first,last)=range(self.hidden_.size(),ith,nth);
        guarded([&]{for(size_t item=first;item<last;++item){
            const float a=std::min(self.gate_[item],7.f),b=std::clamp(self.up_[item],-7.f,7.f);
            self.hidden_[item]=(a/(1.f+std::exp(1.702f*(-a))))*(b+1.f);
        }});
        self.barrier_.wait(unsigned(nth));
        std::tie(first,last)=range(self.count_,ith,nth);
        guarded([&]{
            if(self.gpu_quant_){
                if(ith==0)self.quantize_gpu(2,self.gpu_oai_?nullptr:self.hidden_.data(),self.ff_,self.count_,self.gpu_oai_);
            }else for(size_t compact=first;compact<last;++compact)
                self.quantize(2,self.hidden_.data()+compact*self.ff_,self.ff_,compact);
        });
        self.barrier_.wait(unsigned(nth));
        if(self.prepared_corrections_&&!self.nv(2)){
            const auto correction_started=Clock::now();
            guarded([&]{self.prepare_corrections(2,ith,nth);});
            self.barrier_.wait(unsigned(nth));
            if(ith==0)self.stats_.correction_seconds+=seconds(correction_started);
        }
        std::tie(first,last)=range(self.down_tasks_.size(),ith,nth);
        auto *destination=static_cast<float*>(output->data);
        const auto down_started=self.profile_tasks_?Clock::now():Clock::time_point{};
        guarded([&]{
            auto run=[&](size_t begin,size_t end){for(size_t task=begin;task<end;++task)
                self.matrix_task(self.down_tasks_[task],destination,self.ff_,self.embd_);};
            if(self.dynamic_schedule_){
                for(;;){const size_t begin=self.down_next_.fetch_add(self.schedule_chunk_,std::memory_order_relaxed);
                    if(begin>=self.down_tasks_.size()||self.compute_failed_.load(std::memory_order_relaxed))break;
                    run(begin,std::min(begin+self.schedule_chunk_,self.down_tasks_.size()));}
            }else run(first,last);
        });
        if(self.profile_tasks_)self.down_thread_seconds_[size_t(ith)]=seconds(down_started);
        if(self.reduce_){
            // Every weighted rank must be complete before any reduced row is
            // consumed. The destination is separate from the weighted graph
            // output, so independently scheduled token rows cannot alias.
            self.barrier_.wait(unsigned(nth));
            if(ith==0)self.reduction_started_=Clock::now();
            self.barrier_.wait(unsigned(nth));
            std::tie(first,last)=range(self.n_*self.embd_,ith,nth);
            guarded([&]{for(size_t item=first;item<last;++item){
                const size_t token=item/self.embd_,row=item%self.embd_;
                const size_t base=token*self.used_*self.embd_+row;
                float sum=destination[base];
                for(size_t rank=1;rank<self.used_;++rank)sum+=destination[base+rank*self.embd_];
                require(std::isfinite(sum),"M31 CUDA CPU executor nonfinite rank reduction");
                self.reduced_output_[item]=sum;
            }});
            self.barrier_.wait(unsigned(nth));
            if(ith==0)self.stats_.reduce_seconds+=seconds(self.reduction_started_);
        }
    }
    void release_graph(){
        if(allocator_)ggml_gallocr_free(allocator_);
        if(context_)ggml_free(context_);
        allocator_=nullptr;context_=nullptr;graph_=nullptr;output_=nullptr;capacity_=0;
    }
    void prepare_graph(size_t rows){
        if(graph_&&rows<=capacity_){++stats_.graph_hits;return;}
        const auto start=Clock::now();
        // A failed growth must preserve the previous usable graph. Publish new
        // context/allocator/output ownership only after allocation succeeds.
        auto *next_context=ggml_init({32*1024,nullptr,true});
        if(!next_context)throw std::runtime_error("M31 CUDA CPU executor context allocation failed");
        ggml_gallocr_t next_allocator=nullptr;
        try {
            auto *next_graph=ggml_new_graph_custom(next_context,16,false);
            auto *next_output=ggml_custom_4d(next_context,GGML_TYPE_F32,int64_t(embd_),int64_t(rows),1,1,
                nullptr,0,compute,GGML_N_TASKS_MAX,this);
            ggml_set_output(next_output);ggml_build_forward_expand(next_graph,next_output);
            next_allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if(!next_allocator||!ggml_gallocr_alloc_graph(next_allocator,next_graph))
                throw std::runtime_error("M31 CUDA CPU executor output allocation failed");
            release_graph();
            context_=next_context;allocator_=next_allocator;
            graph_=next_graph;output_=next_output;capacity_=rows;
        } catch (...) {
            if(next_allocator)ggml_gallocr_free(next_allocator);
            ggml_free(next_context);throw;
        }
        ++stats_.graph_builds;stats_.graph_build_seconds+=seconds(start);
        stats_.retained_graphs=1;stats_.retained_scratch_bytes=ggml_gallocr_get_buffer_size(allocator_,0);
    }

public:
    ShortPrefillCudaExecutor(ggml_backend_t backend,ggml_threadpool_t pool,unsigned threads,bool default_gpu_exact=false)
        :backend_(backend),pool_(pool),threads_(threads),default_gpu_exact_(default_gpu_exact){
        require(backend&&ggml_backend_is_cpu(backend)&&pool&&threads&&threads<=GGML_MAX_N_THREADS,
            "M31 CUDA CPU executor requires the persistent CPU backend/pool");
    }
    ~ShortPrefillCudaExecutor(){release_graph();}
    ShortPrefillCudaExecutor(const ShortPrefillCudaExecutor&)=delete;
    ShortPrefillCudaExecutor &operator=(const ShortPrefillCudaExecutor&)=delete;
    static bool supported(ggml_type type){return cuda_math_cpu::supported(type)||type==GGML_TYPE_NVFP4;}
    const Stats &stats()const{return stats_;}
    int active_threads()const{return active_threads_;}
    // Diagnostic snapshots, borrowed until the next forward. Compact route
    // order and [ff,route] layout are the same as the returned weighted terms.
    const std::vector<float> &gate_values()const{return gate_;}
    const std::vector<float> &up_values()const{return up_;}
    const std::vector<float> &hidden_values()const{return hidden_;}
    bool hidden_values_are_native()const{return !gpu_oai_;}
    bool gpu_quant_enabled()const{return gpu_quant_;}
    bool gpu_oai_enabled()const{return gpu_oai_;}
    // Called by Engine on the submitting thread before any layer uploads or
    // dense graphs. CUDA allocation/registration must not serialize the mixed
    // CPU/GPU lanes later in this append.
    void prepare_gpu_quant(size_t n,size_t embd,size_t used,size_t ff){
        require(n&&embd&&used&&ff,"M31 CUDA CPU executor invalid quantizer preparation geometry");
        configure_gpu_quantizer();
        prepared_corrections_=true;
        if(gpu_quant_){
            const size_t rows=product(n,used);
            gpu_quantizer_->reserve(std::max(product(n,embd),product(rows,ff)),rows);
            refresh_gpu_quant_stats();
        }
    }
    size_t gpu_auxiliary_bytes()const{return gpu_quantizer_?gpu_quantizer_->stats().device_bytes:0;}
    size_t gpu_auxiliary_pinned_bytes()const{return gpu_quantizer_?gpu_quantizer_->stats().pinned_bytes:0;}
    void forward(const ggml_tensor *wg,const ggml_tensor *wu,const ggml_tensor *wd,
        const float *gs,const float *us,const float *ds,const std::vector<float>&input,
        const std::vector<int32_t>&ids,const std::vector<float>&weights,
        const std::vector<size_t>&routes,size_t used,std::vector<float>&output,bool reduce=false){
        const auto start=Clock::now();
        require(wg&&wu&&wd&&wg->data&&wu->data&&wd->data&&used,"M31 CUDA CPU executor missing inputs");
        require(&input!=&output&&&weights!=&output,"M31 CUDA CPU executor aliased output");
        require(supported(wg->type)&&supported(wu->type)&&supported(wd->type),"M31 CUDA CPU executor unsupported quant");
        require(wg->ne[0]>0&&wg->ne[1]>0&&wg->ne[2]>0&&wg->ne[0]%32==0&&wg->ne[1]%32==0&&
            wu->ne[0]==wg->ne[0]&&wu->ne[1]==wg->ne[1]&&wd->ne[0]==wg->ne[1]&&wd->ne[1]==wg->ne[0]&&
            wu->ne[2]==wg->ne[2]&&wd->ne[2]==wg->ne[2],"M31 CUDA CPU executor invalid weight geometry");
        require(used<=size_t(wg->ne[2])&&wg->ne[3]==1&&wu->ne[3]==1&&wd->ne[3]==1,
            "M31 CUDA CPU executor invalid expert geometry");
        for(const auto *weight:{wg,wu,wd})require(weight->ne[0]%(weight->type==GGML_TYPE_NVFP4?64:256)==0,
            "M31 CUDA CPU executor invalid quant block width");
        require(!embd_||embd_==size_t(wg->ne[0]),"M31 CUDA CPU executor changed embedding size");
        embd_=size_t(wg->ne[0]);ff_=size_t(wg->ne[1]);used_=used;
        require(!input.empty()&&input.size()%embd_==0,"M31 CUDA CPU executor invalid input shape");
        n_=input.size()/embd_;count_=routes.size();
        for(const auto value:input)require(std::isfinite(value),"M31 CUDA CPU executor nonfinite input");
        require(ids.size()==product(n_,used)&&weights.size()==ids.size(),"M31 CUDA CPU executor invalid routes shape");
        for(size_t route=0;route<ids.size();++route){
            require(ids[route]>=0&&ids[route]<wg->ne[2]&&std::isfinite(weights[route]),"M31 CUDA CPU executor invalid route");
            for(size_t previous=route-route%used;previous<route;++previous)
                require(ids[route]!=ids[previous],"M31 CUDA CPU executor duplicate expert");
        }
        std::vector<uint8_t> seen(ids.size(),0);routes_=routes;
        experts_.clear();experts_.resize(size_t(wg->ne[2]));
        for(size_t compact=0;compact<count_;++compact){const size_t route=routes_[compact];
            require(route<ids.size()&&!seen[route],"M31 CUDA CPU executor invalid selected route");
            seen[route]=1;experts_[size_t(ids[route])].push_back(compact);}
        if(!count_){output.clear();return;}
        if(reduce){
            require(routes.size()==ids.size(),"M31 CUDA CPU reduction requires every route");
            for(size_t route=0;route<routes.size();++route)
                require(routes[route]==route,"M31 CUDA CPU reduction requires original route order");
        }
        reduce_=reduce;reduced_output_=nullptr;
        weight_={wg,wu,wd};scale_={gs,us,ds};input_=input.data();ids_=ids.data();routes_weights_=weights.data();
        configure_gpu_quantizer();
        prepared_corrections_=true;
        retain_debug_hidden_=!gpu_oai_;
        dynamic_schedule_=true;
        profile_tasks_=false;
        schedule_chunk_=4;
        for(const auto *scales:scale_)if(scales)for(size_t expert=0;expert<experts_.size();++expert)
            if(!experts_[expert].empty())require(std::isfinite(scales[expert]),"M31 CUDA CPU executor nonfinite expert scale");
        prepare_graph(count_);
        stats_.setup_seconds+=seconds(start);const auto pack_started=Clock::now();
        for(int role=0;role<3;++role){const size_t rows=role==2?count_:n_,cols=role==2?ff_:embd_;
            if(nv(role)){
                nv_quant_[size_t(role)].resize(rows);
                for(auto &activation:nv_quant_[size_t(role)]){
                    activation.scales.resize(cols/16);activation.values.resize(cols);
                }
            }
            else {
                quant_[size_t(role)].resize(product(rows,cols/32));
                if(prepared_corrections_)corrections_[size_t(role)].resize(quant_[size_t(role)].size());
            }}
        gate_.resize(product(count_,ff_));up_.resize(gate_.size());
        if(retain_debug_hidden_)hidden_.resize(gate_.size());else hidden_.clear();
        gate_tasks_.clear();down_tasks_.clear();
        for(size_t expert=0;expert<experts_.size();++expert)if(!experts_[expert].empty()){
            for(size_t assignment=0;assignment<experts_[expert].size();assignment+=16){
                const size_t end=std::min(assignment+16,experts_[expert].size());
                for(size_t row=0;row<ff_;row+=16)for(int role=0;role<2;++role)
                    gate_tasks_.push_back({role,int(expert),row,std::min(row+16,ff_),assignment,end});
                for(size_t row=0;row<embd_;row+=16)
                    down_tasks_.push_back({2,int(expert),row,std::min(row+16,embd_),assignment,end});
            }}
        stats_.pack_seconds+=seconds(pack_started);const auto compute_started=Clock::now();
        worker_errors_.assign(size_t(threads_),nullptr);compute_failed_.store(false,std::memory_order_relaxed);
        gate_next_.store(0,std::memory_order_relaxed);down_next_.store(0,std::memory_order_relaxed);
        if(profile_tasks_){gate_thread_seconds_.assign(size_t(threads_),0);down_thread_seconds_.assign(size_t(threads_),0);}
        if(reduce_){output.resize(product(n_,embd_));reduced_output_=output.data();}
        ggml_backend_cpu_set_n_threads(backend_,int(threads_));
        ggml_backend_cpu_set_threadpool(backend_,pool_);
        if(ggml_backend_graph_compute(backend_,graph_)!=GGML_STATUS_SUCCESS)throw std::runtime_error("M31 CUDA CPU executor compute failed");
        if(compute_failed_.load(std::memory_order_acquire)){
            for(const auto &error:worker_errors_)if(error)std::rethrow_exception(error);
            throw std::runtime_error("M31 CUDA CPU executor worker failed");
        }
        refresh_gpu_quant_stats();
        if(prepared_corrections_){
            if(!nv(0))stats_.correction_groups+=quant_[0].size();
            if(nv(0)!=nv(1)&&!nv(1))stats_.correction_groups+=quant_[1].size();
            if(!nv(2))stats_.correction_groups+=quant_[2].size();
        }
        stats_.retained_correction_bytes=0;
        for(const auto &buffer:corrections_)stats_.retained_correction_bytes+=buffer.capacity()*sizeof(int32_t);
        if(profile_tasks_){
            auto summarize=[&](const std::vector<double>&times){std::vector<double>sorted(times.begin(),times.begin()+active_threads_);
                std::sort(sorted.begin(),sorted.end());return std::make_pair(sorted.back(),sorted[sorted.size()/2]);};
            const auto gate=summarize(gate_thread_seconds_),down=summarize(down_thread_seconds_);
            stats_.gate_thread_max_seconds+=gate.first;stats_.gate_thread_median_seconds+=gate.second;
            stats_.down_thread_max_seconds+=down.first;stats_.down_thread_median_seconds+=down.second;
            std::fprintf(stderr,"M31 shortprefill_task_schedule mode=%s chunk=%zu threads=%d n=%zu routes=%zu gate_max_ms=%.3f gate_median_ms=%.3f down_max_ms=%.3f down_median_ms=%.3f\n",
                dynamic_schedule_?"dynamic":"static",schedule_chunk_,active_threads_,n_,count_,1000*gate.first,1000*gate.second,1000*down.first,1000*down.second);
        }
        stats_.compute_seconds+=seconds(compute_started);const auto read_started=Clock::now();
        if(!reduce_){output.resize(product(count_,embd_));std::memcpy(output.data(),output_->data,output.size()*sizeof(float));}
        stats_.read_seconds+=seconds(read_started);++stats_.forwards;stats_.assignments+=count_;
    }
};
} // namespace strata::m31
