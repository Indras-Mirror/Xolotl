// Included at namespace scope by m31_gpu_experts.cpp.
void GpuExperts::forward_device(int layer,int n,ggml_tensor *normalized,
    const std::vector<int32_t> &ids,const std::vector<float> &weights,
    ggml_tensor *shared,ggml_tensor *residual,ggml_tensor *destination){
    check(p_->ready&&p_->g.routed(layer),"uninitialized device FFN or invalid layer");
    check(n>0&&n<=p_->max_chunk,"invalid device FFN token count");
    const auto &g=p_->g;
    const size_t bytes=size_t(g.embd)*size_t(n)*sizeof(float);
    for(auto *tensor:{normalized,shared,residual,destination}){
        check(tensor&&tensor->type==GGML_TYPE_F32&&ggml_is_contiguous(tensor)&&ggml_nbytes(tensor)==bytes&&tensor->data,
              "device FFN needs contiguous F32 tensors of shape[embd,n]");
        auto *buffer=tensor->view_src?tensor->view_src->buffer:tensor->buffer;
        check(buffer&&!ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer)),"device FFN tensor must reside in GPU memory");
        check(ggml_backend_dev_supports_buft(p_->device,ggml_backend_buffer_get_type(buffer)),"device FFN tensor belongs to another device/backend");
    }
    const uintptr_t output=reinterpret_cast<uintptr_t>(destination->data);
    for(auto *source:{normalized,shared,residual}){
        const uintptr_t input=reinterpret_cast<uintptr_t>(source->data);
        check(output<=UINTPTR_MAX-bytes&&input<=UINTPTR_MAX-bytes&&
              (output+bytes<=input||input+bytes<=output),"device FFN output overlaps its inputs");
    }
    check(ids.size()==size_t(g.used)*size_t(n)&&weights.size()==ids.size(),"device FFN route geometry differs");
    std::vector<uint8_t> mask(size_t(g.experts),0);
    std::set<uint64_t> protected_keys;
    std::vector<int> selected;
    for(size_t i=0;i<ids.size();++i){
        check(ids[i]>=0&&ids[i]<g.experts&&std::isfinite(weights[i]),"invalid device FFN route");
        for(size_t previous=i-i%size_t(g.used);previous<i;++previous)check(ids[previous]!=ids[i],"duplicate device FFN top-k expert");
        mask[size_t(ids[i])]=1;
    }
    for(int e=0;e<g.experts;++e)if(mask[size_t(e)]){selected.push_back(e);protected_keys.insert(p_->key(layer,e));}
    check(selected.size()<=p_->stat.slot_count,"device FFN selected experts exceed cache capacity");

    try{
        // IDs/weights were read from this backend's dense graph. Drain any
        // remaining producer operations before the native expert graph borrows
        // normalized/shared/residual. No FFN F32 tensor is read to the host.
        ggml_backend_synchronize(p_->backend);
        p_->finish_prefetch();
        if(p_->uploads)p_->uploads->begin_batch();
        std::vector<int32_t> remap(size_t(g.experts),-1),mapped(ids.size());
        std::array<std::vector<float>,3> scales;
        for(auto &role:scales)role.assign(p_->stat.slot_count,1.f);
        for(int e:selected){
            const size_t slot=p_->load(layer,e,protected_keys);remap[size_t(e)]=int32_t(slot);
            for(size_t role=0;role<3;++role)scales[role][slot]=p_->layers[size_t(layer)].scales[role][size_t(e)];
        }
        for(size_t i=0;i<ids.size();++i)mapped[i]=remap[size_t(ids[i])];
        ggml_backend_event_t ready=nullptr;
        if(p_->uploads){p_->uploads->finish();ready=p_->uploads->completion_event();}
        const auto start=p_->profile?Impl::Clock::now():Impl::Clock::time_point{};
        check(p_->active_read_keys.empty(),"overlapping device expert consumers are unsupported");
        p_->publish_foreground();
        p_->acquire_readers(protected_keys);
        struct ReadersLease {Impl *owner;~ReadersLease(){owner->release_readers();}} readers{p_.get()};
        p_->execute_device(layer,n,normalized,mapped,ids,weights,scales,shared,residual,destination,ready,protected_keys);
        if(p_->profile)p_->stat.execute_seconds+=Impl::elapsed(start);
        ++p_->stat.forwards;
    }catch(...){
        p_->recover_uploads();
        throw;
    }
}
