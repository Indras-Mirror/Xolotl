// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "prefill/short/shortprefill_cuda_executor.hpp"
#include <iostream>
#include <random>

#ifdef M31_SHORT_ALLOC_WRAP
namespace { std::atomic<bool> fail_next_short_allocation{false}; }
extern "C" bool __real_ggml_gallocr_alloc_graph(ggml_gallocr_t,ggml_cgraph *);
extern "C" bool __wrap_ggml_gallocr_alloc_graph(ggml_gallocr_t allocator,ggml_cgraph *graph) {
    if(fail_next_short_allocation.exchange(false))return false;
    return __real_ggml_gallocr_alloc_graph(allocator,graph);
}
#endif
using namespace strata::m31;
namespace {
void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
struct Cpu {
    ggml_backend_t backend=ggml_backend_cpu_init();
    ggml_threadpool_t pool=nullptr;
    explicit Cpu(int threads){
        check(backend,"backend");ggml_backend_cpu_set_n_threads(backend,threads);
        auto params=ggml_threadpool_params_default(threads);params.poll=0;
        pool=ggml_threadpool_new(&params);check(pool,"pool");ggml_backend_cpu_set_threadpool(backend,pool);
    }
    ~Cpu(){ggml_backend_cpu_set_threadpool(backend,nullptr);ggml_threadpool_free(pool);ggml_backend_free(backend);}
};
struct Weights {
    ggml_context *context=ggml_init({64*1024,nullptr,true});
    ggml_backend_buffer_t buffer=nullptr;
    std::array<ggml_tensor*,3> tensors{};
    std::array<std::vector<float>,3> scales;
    Weights(ggml_backend_t backend,std::array<ggml_type,3> types){
        check(context,"weight context");
        for(size_t role=0;role<3;++role){tensors[role]=ggml_new_tensor_3d(context,types[role],256,256,4);
            scales[role]={.25f,.5f,.75f,1.f};}
        buffer=ggml_backend_alloc_ctx_tensors(context,backend);check(buffer,"weight buffer");
        std::mt19937 generator(195);std::uniform_real_distribution<float> distribution(-.2f,.2f);
        for(size_t role=0;role<3;++role){
            std::vector<float> source(size_t(ggml_nelements(tensors[role]))),imatrix(256,1.f);
            for(auto &value:source)value=distribution(generator);
            std::vector<uint8_t> packed(ggml_nbytes(tensors[role]));
            check(ggml_quantize_chunk(types[role],source.data(),packed.data(),0,int64_t(source.size()/256),256,imatrix.data())==packed.size(),"quantization");
            ggml_backend_tensor_set(tensors[role],packed.data(),0,packed.size());}
    }
    ~Weights(){if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);}
};
}
int main(){try{
    Cpu serial_cpu(1),parallel_cpu(4),mismatched_cpu(2);
    ShortPrefillCudaExecutor serial(serial_cpu.backend,serial_cpu.pool,1);
    ShortPrefillCudaExecutor parallel(parallel_cpu.backend,parallel_cpu.pool,4);
    ShortPrefillCudaExecutor mismatched(mismatched_cpu.backend,mismatched_cpu.pool,4);
    serial.prepare_gpu_quant(31,256,2,256);
    check(serial.gpu_auxiliary_bytes()==0&&serial.gpu_auxiliary_pinned_bytes()==0,
        "portable executor preparation allocated GPU memory");
    size_t compared=0;
    for(const auto types:{std::array{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS,GGML_TYPE_IQ3_XXS},
            std::array{GGML_TYPE_NVFP4,GGML_TYPE_IQ4_XS,GGML_TYPE_NVFP4},
            std::array{GGML_TYPE_NVFP4,GGML_TYPE_NVFP4,GGML_TYPE_NVFP4}}){
        Weights weights(parallel_cpu.backend,types);
        for(const size_t n:{1u,7u,8u,17u,31u,8u}){
            std::vector<float> input(n*256),rw(n*2);
            std::vector<int32_t> ids(n*2);
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.037f);
            for(size_t token=0;token<n;++token){ids[2*token]=int32_t(token%4);ids[2*token+1]=int32_t((token+1)%4);rw[2*token]=.375f;rw[2*token+1]=.625f;}
            for(const bool sparse:{false,true}){
                std::vector<size_t> selected;
                for(size_t route=0;route<ids.size();++route)if(!sparse||route%3)selected.push_back(route);
                std::reverse(selected.begin(),selected.end());std::vector<float> expected,actual;
                serial.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                    weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,selected,2,expected);
                check(serial.hidden_values().size()==selected.size()*256,
                    "portable CPU OAI hidden buffer missing");
                parallel.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                    weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,selected,2,actual);
                std::vector<float> fewer_workers;
                mismatched.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                    weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,selected,2,fewer_workers);
                check(mismatched.active_threads()==2,"fixture did not cap requested four workers to pool two");
                check(fewer_workers.size()==expected.size()&&
                    std::memcmp(fewer_workers.data(),expected.data(),expected.size()*sizeof(float))==0,
                    "actual worker count mismatch changed output");
                check(expected.size()==actual.size(),"output size");
                check(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))==0,"parallel differs from serial");
                for(const auto value:actual)check(std::isfinite(value),"nonfinite output");
                compared+=selected.size();
            }
            // Full-prompt rank reduction uses the same sequential rank order
            // for each row, while different rows execute on the persistent pool.
            std::vector<size_t> all_routes(ids.size());
            std::iota(all_routes.begin(),all_routes.end(),size_t(0));
            std::vector<float> weighted,reduced,expected_reduced(n*256),serial_reduced;
            serial.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,all_routes,2,weighted);
            for(size_t token=0;token<n;++token)for(size_t row=0;row<256;++row)
                expected_reduced[token*256+row]=weighted[token*2*256+row]+weighted[(token*2+1)*256+row];
            parallel.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,all_routes,2,reduced,true);
            serial.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,all_routes,2,serial_reduced,true);
            check(reduced.size()==expected_reduced.size(),"reduced output size");
            check(std::memcmp(reduced.data(),expected_reduced.data(),reduced.size()*sizeof(float))==0,"pooled rank reduction differs");
            check(std::memcmp(reduced.data(),serial_reduced.data(),reduced.size()*sizeof(float))==0,"serial rank reduction differs");
            if(n==8){
                std::vector<int32_t> ids4(n*4);std::vector<float> rw4(n*4);
                std::vector<size_t> routes4(n*4);
                for(size_t route=0;route<ids4.size();++route){ids4[route]=int32_t(route%4);rw4[route]=.125f*float(1+route%4);routes4[route]=route;}
                serial.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,
                    input,ids4,rw4,routes4,4,weighted);
                parallel.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,
                    input,ids4,rw4,routes4,4,reduced,true);
                for(size_t token=0;token<n;++token)for(size_t row=0;row<256;++row){
                    float sum=weighted[token*4*256+row];
                    for(size_t rank=1;rank<4;++rank)sum+=weighted[(token*4+rank)*256+row];
                    expected_reduced[token*256+row]=sum;
                }
                check(std::memcmp(reduced.data(),expected_reduced.data(),reduced.size()*sizeof(float))==0,"four-rank reduction order differs");
            }
            bool rejected=false;std::vector<float> output;
            try{parallel.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,input,ids,rw,{0,0},2,output);}
            catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"duplicate selected route accepted");
            if(n==1){
                const float saved=input[0];input[0]=std::numeric_limits<float>::quiet_NaN();rejected=false;
                try{parallel.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,input,ids,rw,{0},2,output);}
                catch(const std::invalid_argument&){rejected=true;}
                check(rejected,"nonfinite input accepted");input[0]=saved;
            }
            if(n==7&&types[0]==GGML_TYPE_IQ3_XXS){
                // A worker-side arithmetic error must join every barrier and
                // return an exception to the submitting thread. Restore this
                // fixture's original scale, then prove the pool still works.
                for(size_t role=0;role<3;++role) {
                    auto *bytes=static_cast<uint8_t*>(weights.tensors[role]->data);
                    uint16_t saved;std::memcpy(&saved,bytes,sizeof(saved));
                    const uint16_t nan_half=0x7e00;std::memcpy(bytes,&nan_half,sizeof(nan_half));
                    for(auto *executor : {&parallel,&mismatched}) {
                        rejected=false;
                        try{executor->forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,
                            input,ids,rw,all_routes,2,output,true);}
                        catch(const std::invalid_argument&){rejected=true;}
                        check(rejected,"worker arithmetic error was not returned after all barriers");
                    }
                    std::memcpy(bytes,&saved,sizeof(saved));
                    for(auto *executor : {&parallel,&mismatched}) {
                        executor->forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],
                            weights.scales[0].data(),weights.scales[1].data(),weights.scales[2].data(),input,ids,rw,all_routes,2,output,true);
                        check(output.size()==serial_reduced.size()&&std::memcmp(output.data(),serial_reduced.data(),output.size()*sizeof(float))==0,
                            "pool did not recover after worker error");
                    }
                }
            }
        }
    }
#ifdef M31_SHORT_ALLOC_WRAP
    {
        Weights weights(parallel_cpu.backend,{GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_XS});
        ShortPrefillCudaExecutor retry(mismatched_cpu.backend,mismatched_cpu.pool,4);
        ShortPrefillCudaExecutor fresh(serial_cpu.backend,serial_cpu.pool,1);
        std::vector<float> actual,expected;
        auto run=[&](auto &executor,size_t rows,auto &output) {
            std::vector<float> input(rows*256,.125f),rw(rows*2,.5f);
            std::vector<int32_t> ids(rows*2);std::vector<size_t> routes(rows*2);
            for(size_t route=0;route<ids.size();++route){ids[route]=int32_t(route%2);routes[route]=route;}
            executor.forward(weights.tensors[0],weights.tensors[1],weights.tensors[2],nullptr,nullptr,nullptr,
                input,ids,rw,routes,2,output);
        };
        // Two initial failures, a successful graph, then a failed growth. Old
        // graph reuse and a subsequent growth both preserve reference bytes.
        for(size_t rows : {2u,2u,4u}) {
            if(rows==4){run(retry,2,actual);run(fresh,2,expected);}
            fail_next_short_allocation.store(true);bool failed=false;
            try{run(retry,rows,actual);}catch(const std::runtime_error &error){
                failed=std::string(error.what()).find("output allocation failed")!=std::string::npos;
            }
            check(failed&&!fail_next_short_allocation.load(),"allocation seam did not reject graph creation/growth");
        }
        for(size_t rows : {2u,4u,4u}) {
            run(retry,rows,actual);run(fresh,rows,expected);
            check(actual.size()==expected.size()&&std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
                "allocation failure changed retry/reuse bytes");
        }
    }
#endif
    check(parallel.stats().graph_builds&&parallel.stats().graph_hits,"graph retention not exercised");
    check(parallel.stats().reduce_seconds>0,"pooled rank reduction not exercised");
    check(parallel.stats().correction_groups>0&&parallel.stats().correction_seconds>0,
        "prepared correction sidecars not exercised");
    std::cout<<"PASS CUDA-compatible CPU executor serial/parallel rows="<<compared<<" builds="<<parallel.stats().graph_builds<<" hits="<<parallel.stats().graph_hits<<'\n';
    return 0;
}catch(const std::exception &error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}}
