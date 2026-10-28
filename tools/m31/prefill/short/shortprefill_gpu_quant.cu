// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "shortprefill_gpu_quant.hpp"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/unary.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::m31 {
namespace {
void quant_check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("M31 activation CUDA quantizer: ")+cudaGetErrorString(status));}
void quant_require(bool value,const char *message){if(!value)throw std::invalid_argument(std::string("M31 activation CUDA quantizer: ")+message);}
size_t quant_product(size_t a,size_t b){quant_require(!b||a<=SIZE_MAX/b,"size overflow");return a*b;}
struct QuantBuffer {
    void *pointer=nullptr;size_t bytes=0;bool pinned=false;
    explicit QuantBuffer(bool host=false):pinned(host){}
    ~QuantBuffer(){if(pointer){if(pinned)cudaFreeHost(pointer);else cudaFree(pointer);}}
    bool reserve(size_t wanted){if(wanted<=bytes)return false;void *replacement=nullptr;
        if(pinned)quant_check(cudaHostAlloc(&replacement,wanted,cudaHostAllocDefault));else quant_check(cudaMalloc(&replacement,wanted));
        if(pointer){if(pinned)cudaFreeHost(pointer);else cudaFree(pointer);}pointer=replacement;bytes=wanted;return true;}
};
__global__ void quant_oai(const float *gate,const float *up,float *output,size_t count){
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<count)output[at]=ggml_cuda_op_swiglu_oai_single(gate[at],up[at]);
}
}
struct ShortPrefillGpuQuantizer::Impl {
    int device=0;cudaStream_t stream=nullptr;Stats stat;bool in_call=false;
    QuantBuffer input,up,hidden,packed,scale;
    QuantBuffer host_input{true},host_up{true},host_packed{true},host_scale{true};
    ~Impl(){if(stream){cudaSetDevice(device);cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
    void reserve_buffer(QuantBuffer &buffer,size_t bytes){
        if(bytes<=buffer.bytes)return;const auto start=Clock::now();
        if(buffer.reserve(bytes)){++stat.allocations;if(in_call)++stat.runtime_allocations;refresh_stats();}
        stat.reserve_seconds+=elapsed(start);
    }
    void reserve_all(size_t max_values,size_t max_rows){
        quant_require(max_values&&max_values<=268435456&&max_rows&&max_rows<=65536,"invalid reservation bounds");
        quant_check(cudaSetDevice(device));
        const size_t values=std::max(max_values,stat.reserved_max_values),rows=std::max(max_rows,stat.reserved_max_rows);
        const size_t bytes=quant_product(values,sizeof(float));
        const size_t padding=quant_product(MATRIX_ROW_PADDING-1,rows);
        quant_require(values<=SIZE_MAX-padding,"padded reservation overflow");
        const size_t blocks=(values+padding+QK8_1_MMQ-1)/QK8_1_MMQ;
        const size_t packed_bytes=quant_product(blocks,sizeof(block_q8_1_mmq));
        quant_require(bytes<=1024ull*1024*1024&&packed_bytes<=1024ull*1024*1024,"reservation buffer exceeds1GiB");
        // Every prior quantization call completes before return. This explicit
        // entry point is used before a layer starts either CPU work or DMA.
        reserve_buffer(input,bytes);reserve_buffer(up,bytes);reserve_buffer(hidden,bytes);
        reserve_buffer(host_input,bytes);reserve_buffer(host_up,bytes);
        reserve_buffer(packed,packed_bytes);reserve_buffer(host_packed,packed_bytes);
        reserve_buffer(scale,rows*sizeof(float));reserve_buffer(host_scale,rows*sizeof(float));
        stat.reserved_max_values=values;stat.reserved_max_rows=rows;++stat.reserve_calls;refresh_stats();
    }
    struct Drain {Impl *p;bool completed=false;~Drain(){if(!completed)cudaStreamSynchronize(p->stream);p->in_call=false;}};
    void copy_to_device(QuantBuffer &dst,QuantBuffer &host,const float *values,size_t bytes){
        const auto copy_start=Clock::now();std::memcpy(host.pointer,values,bytes);stat.host_copy_seconds+=elapsed(copy_start);
        const auto submit_start=Clock::now();quant_check(cudaMemcpyAsync(dst.pointer,host.pointer,bytes,cudaMemcpyHostToDevice,stream));
        stat.submit_seconds+=elapsed(submit_start);stat.h2d_bytes+=bytes;
    }

    const float *prepare(const float *values,const float *up_values,size_t columns,size_t rows){
        quant_require(values&&columns&&columns%32==0&&rows&&columns<=65536&&rows<=65536,"invalid input geometry");
        quant_check(cudaSetDevice(device));const size_t count=quant_product(columns,rows),bytes=quant_product(count,sizeof(float));
        quant_require(bytes<=1024ull*1024*1024,"input scratch exceeds1GiB");
        reserve_buffer(input,bytes);reserve_buffer(host_input,bytes);copy_to_device(input,host_input,values,bytes);
        if(!up_values)return static_cast<const float*>(input.pointer);
        reserve_buffer(up,bytes);reserve_buffer(hidden,bytes);reserve_buffer(host_up,bytes);copy_to_device(up,host_up,up_values,bytes);
        const auto submit_start=Clock::now();quant_oai<<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const float*>(input.pointer),static_cast<const float*>(up.pointer),static_cast<float*>(hidden.pointer),count);
        quant_check(cudaGetLastError());stat.submit_seconds+=elapsed(submit_start);return static_cast<const float*>(hidden.pointer);
    }
    void refresh_stats(){stat.device_bytes=input.bytes+up.bytes+hidden.bytes+packed.bytes+scale.bytes;
        stat.pinned_bytes=host_input.bytes+host_up.bytes+host_packed.bytes+host_scale.bytes;}
    void q8(ggml_type type,const float *values,const float *up_values,size_t columns,size_t rows,std::vector<cuda_math_cpu::Q8Mmq32>&output){
        quant_require(type==GGML_TYPE_IQ3_XXS||type==GGML_TYPE_IQ4_XS,"Q8 D4 supports IQ3_XXS/IQ4_XS only");
        quant_require(!in_call,"overlapping quantizer calls");in_call=true;
        const auto start=Clock::now();Drain drain{this};
        const float *device_input=prepare(values,up_values,columns,rows);
        const size_t padded=(columns+MATRIX_ROW_PADDING-1)/MATRIX_ROW_PADDING*MATRIX_ROW_PADDING;
        const size_t nbytes=quant_product(quant_product(padded/QK8_1_MMQ,rows),sizeof(block_q8_1_mmq));
        reserve_buffer(packed,nbytes);reserve_buffer(host_packed,nbytes);
        const auto submit_start=Clock::now();quantize_mmq_q8_1_cuda(device_input,nullptr,packed.pointer,type,int64_t(columns),int64_t(columns),int64_t(columns*rows),int64_t(columns*rows),int64_t(padded),int64_t(rows),1,1,stream);
        quant_check(cudaGetLastError());quant_check(cudaMemcpyAsync(host_packed.pointer,packed.pointer,nbytes,cudaMemcpyDeviceToHost,stream));
        stat.submit_seconds+=elapsed(submit_start);const auto wait_start=Clock::now();
        quant_check(cudaStreamSynchronize(stream));stat.wait_seconds+=elapsed(wait_start);drain.completed=true;stat.d2h_bytes+=nbytes;
        const auto unpack_start=Clock::now();output.resize(quant_product(columns/32,rows));const auto *blocks=static_cast<const block_q8_1_mmq*>(host_packed.pointer);
        for(size_t row=0;row<rows;++row)for(size_t group=0;group<columns/32;++group){const auto &block=blocks[(group/4)*rows+row];auto &q=output[row*(columns/32)+group];
            q.d=block.d4[group%4];std::memcpy(q.qs,block.qs+(group%4)*32,32);}
        stat.unpack_seconds+=elapsed(unpack_start);++stat.calls;stat.wall_seconds+=elapsed(start);refresh_stats();
    }
    void nvfp4(const float *values,const float *up_values,size_t columns,size_t rows,std::vector<cuda_nvfp4_cpu::Activation>&output){
        quant_require(columns%64==0,"NVFP4 columns must be a multiple of64");
        quant_require(!in_call,"overlapping quantizer calls");in_call=true;
        const auto start=Clock::now();Drain drain{this};
        const float *device_input=prepare(values,up_values,columns,rows);
        const size_t padded=(columns+MATRIX_ROW_PADDING-1)/MATRIX_ROW_PADDING*MATRIX_ROW_PADDING;
        const size_t nbytes=quant_product(quant_product(padded/QK_FP4_MMQ,rows),sizeof(block_fp4_mmq));
        reserve_buffer(packed,nbytes);reserve_buffer(host_packed,nbytes);reserve_buffer(scale,rows*sizeof(float));reserve_buffer(host_scale,rows*sizeof(float));
        const auto submit_start=Clock::now();quantize_mmq_fp4_cuda(device_input,nullptr,packed.pointer,static_cast<float*>(scale.pointer),GGML_TYPE_NVFP4,true,int64_t(columns),int64_t(columns),int64_t(columns*rows),int64_t(columns*rows),int64_t(padded),int64_t(rows),1,1,stream);
        quant_check(cudaGetLastError());quant_check(cudaMemcpyAsync(host_packed.pointer,packed.pointer,nbytes,cudaMemcpyDeviceToHost,stream));
        quant_check(cudaMemcpyAsync(host_scale.pointer,scale.pointer,rows*sizeof(float),cudaMemcpyDeviceToHost,stream));
        stat.submit_seconds+=elapsed(submit_start);const auto wait_start=Clock::now();
        quant_check(cudaStreamSynchronize(stream));stat.wait_seconds+=elapsed(wait_start);drain.completed=true;
        const auto unpack_start=Clock::now();stat.d2h_bytes+=nbytes+rows*sizeof(float);output.resize(rows);const auto *blocks=static_cast<const block_fp4_mmq*>(host_packed.pointer);
        for(size_t row=0;row<rows;++row){auto &a=output[row];a.row_scale=static_cast<const float*>(host_scale.pointer)[row];a.scales.resize(columns/16);a.values.resize(columns);
            for(size_t group=0;group<columns/16;++group){const auto &block=blocks[(group/16)*rows+row];const size_t sub=group%16;
                a.scales[group]=reinterpret_cast<const uint8_t*>(block.d4)[sub];const auto *q=reinterpret_cast<const uint8_t*>(block.qs)+sub*8;
                for(size_t i=0;i<8;++i){a.values[group*16+i]=cuda_nvfp4_cpu::fp4_values[q[i]&15];a.values[group*16+i+8]=cuda_nvfp4_cpu::fp4_values[q[i]>>4];}}}
        stat.unpack_seconds+=elapsed(unpack_start);++stat.calls;stat.wall_seconds+=elapsed(start);refresh_stats();
    }
};
bool ShortPrefillGpuQuantizer::available() noexcept{return true;}
ShortPrefillGpuQuantizer::ShortPrefillGpuQuantizer(int device):p_(std::make_unique<Impl>()){
    if(device<0)quant_check(cudaGetDevice(&device));
    p_->device=device;quant_check(cudaSetDevice(device));int lowest=0,highest=0;
    quant_check(cudaDeviceGetStreamPriorityRange(&lowest,&highest));
    quant_check(cudaStreamCreateWithPriority(&p_->stream,cudaStreamNonBlocking,highest));
    quant_check(cudaStreamGetPriority(p_->stream,&p_->stat.stream_priority));
}
ShortPrefillGpuQuantizer::~ShortPrefillGpuQuantizer()=default;
void ShortPrefillGpuQuantizer::reserve(size_t v,size_t r){quant_require(!p_->in_call,"reservation during an active quantization call");p_->reserve_all(v,r);}
void ShortPrefillGpuQuantizer::quantize_q8(ggml_type t,const float*x,size_t c,size_t r,std::vector<cuda_math_cpu::Q8Mmq32>&o){p_->q8(t,x,nullptr,c,r,o);}
void ShortPrefillGpuQuantizer::quantize_nvfp4(const float*x,size_t c,size_t r,std::vector<cuda_nvfp4_cpu::Activation>&o){p_->nvfp4(x,nullptr,c,r,o);}
void ShortPrefillGpuQuantizer::oai_quantize_q8(ggml_type t,const float*g,const float*u,size_t c,size_t r,std::vector<cuda_math_cpu::Q8Mmq32>&o){quant_require(u,"missing OAI up input");p_->q8(t,g,u,c,r,o);}
void ShortPrefillGpuQuantizer::oai_quantize_nvfp4(const float*g,const float*u,size_t c,size_t r,std::vector<cuda_nvfp4_cpu::Activation>&o){quant_require(u,"missing OAI up input");p_->nvfp4(g,u,c,r,o);}
const ShortPrefillGpuQuantizer::Stats &ShortPrefillGpuQuantizer::stats() const noexcept{return p_->stat;}
}
