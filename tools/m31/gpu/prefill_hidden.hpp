#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::m31 {
// Two engine-owned GPU hidden tensors. No graph metadata, FFN output, or
// workspace is borrowed across a layer boundary. The caller drains all graph
// and private FFN streams before prepare() changes descriptors or allocations.
class GpuPrefillHidden {
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * tensors_[2] = {nullptr,nullptr};
    int64_t columns_ = 0, rows_ = 0, capacity_rows_ = 0;
    size_t stride_ = 0;
    int current_ = 0;
    static void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
public:
    ~GpuPrefillHidden(){if(buffer_)ggml_backend_buffer_free(buffer_);if(context_)ggml_free(context_);}
    GpuPrefillHidden() = default;
    GpuPrefillHidden(const GpuPrefillHidden &) = delete;
    GpuPrefillHidden &operator=(const GpuPrefillHidden &) = delete;
    void prepare(ggml_backend_t backend,int64_t columns,int rows){
        check(backend&&columns>0&&rows>0,"M31 GPU prefill hidden: invalid geometry/backend");
        check(columns<=INT64_MAX/rows&&uint64_t(columns)*uint64_t(rows)<=SIZE_MAX/8,
              "M31 GPU prefill hidden: allocation overflow");
        check(!columns_||columns_==columns,"M31 GPU prefill hidden: embedding geometry changed");
        current_=0;
        if(buffer_&&context_&&tensors_[0]&&tensors_[1]&&rows_==rows&&columns_==columns)return;
        if(context_){ggml_free(context_);context_=nullptr;tensors_[0]=tensors_[1]=nullptr;}
        auto buft=ggml_backend_get_default_buffer_type(backend);
        if(!buffer_||rows>capacity_rows_){
            if(buffer_){ggml_backend_buffer_free(buffer_);buffer_=nullptr;}
            const size_t alignment=ggml_backend_buft_get_alignment(buft);
            const size_t raw=size_t(columns)*size_t(rows)*sizeof(float);
            check(alignment&&raw<=SIZE_MAX-(alignment-1),"M31 GPU prefill hidden: alignment overflow");
            stride_=(raw+alignment-1)/alignment*alignment;
            check(stride_<=SIZE_MAX/2,"M31 GPU prefill hidden: pair overflow");
            buffer_=ggml_backend_buft_alloc_buffer(buft,stride_*2);
            check(buffer_!=nullptr,"M31 GPU prefill hidden: buffer allocation failed");
            capacity_rows_=rows;
        }
        context_=ggml_init({64*1024,nullptr,true});
        check(context_!=nullptr,"M31 GPU prefill hidden: tensor metadata allocation failed");
        auto *base=static_cast<uint8_t*>(ggml_backend_buffer_get_base(buffer_));
        for(int i=0;i<2;++i){
            tensors_[i]=ggml_new_tensor_2d(context_,GGML_TYPE_F32,columns,rows);
            ggml_set_name(tensors_[i],i?"m31.prefill.hidden1":"m31.prefill.hidden0");
            check(ggml_backend_tensor_alloc(buffer_,tensors_[i],base+size_t(i)*stride_)==GGML_STATUS_SUCCESS,
                  "M31 GPU prefill hidden: tensor binding failed");
        }
        columns_=columns;rows_=rows;
    }
    ggml_tensor *current()const{return tensors_[current_];}
    ggml_tensor *next()const{return tensors_[1-current_];}
    void advance(){check(next()!=nullptr,"M31 GPU prefill hidden: uninitialized state");current_=1-current_;}
    void seed(const std::vector<float>&values){
        check(current()&&values.size()==size_t(columns_)*size_t(rows_),"M31 GPU prefill hidden: seed shape mismatch");
        ggml_backend_tensor_set(current(),values.data(),0,values.size()*sizeof(float));
    }
    void copy_and_advance(ggml_backend_t backend,ggml_tensor *source){
        check(source&&next()&&source->type==GGML_TYPE_F32&&ggml_is_contiguous(source)&&
              ggml_are_same_shape(source,next()),"M31 GPU prefill hidden: source shape mismatch");
        ggml_backend_tensor_copy_async(backend,backend,source,next());
        ggml_backend_synchronize(backend); // source workspace may be reused after this method
        advance();
    }
    size_t bytes()const{return buffer_?ggml_backend_buffer_get_size(buffer_):0;}
};
}
