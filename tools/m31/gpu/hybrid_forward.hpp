// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
// Included at namespace scope by gpu/experts.cpp.
void GpuExperts::forward_device_hybrid(int layer,int n,ggml_tensor *normalized,
    const std::vector<int32_t> &ids,const std::vector<float> &weights,
    ggml_tensor *shared,ggml_tensor *residual,ggml_tensor *destination,const CpuRoutes &cpu_routes){
    check(p_->ready&&p_->g.routed(layer)&&n>1&&n<=p_->max_chunk&&cpu_routes,
          "invalid hybrid prefill state");
    constexpr int upload_window=1;
    constexpr bool device_gather=true;
    const auto whole_start=Impl::Clock::now();
    const auto &g=p_->g;const auto &info=p_->layers[size_t(layer)];
    const size_t count=size_t(g.embd)*size_t(n),bytes=count*sizeof(float);
    for(auto *tensor:{normalized,shared,residual,destination}){
        check(tensor&&tensor->type==GGML_TYPE_F32&&ggml_is_contiguous(tensor)&&
            ggml_nbytes(tensor)==bytes&&tensor->data,"hybrid FFN tensor geometry differs");
        auto *buffer=tensor->view_src?tensor->view_src->buffer:tensor->buffer;
        check(buffer&&!ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer))&&
            ggml_backend_dev_supports_buft(p_->device,ggml_backend_buffer_get_type(buffer)),
            "hybrid FFN tensors must reside on the borrowed GPU");
    }
    const uintptr_t output_address=reinterpret_cast<uintptr_t>(destination->data);
    for(auto *source:{normalized,shared,residual}){
        const uintptr_t input_address=reinterpret_cast<uintptr_t>(source->data);
        check(output_address<=UINTPTR_MAX-bytes&&input_address<=UINTPTR_MAX-bytes&&
            (output_address+bytes<=input_address||input_address+bytes<=output_address),
            "hybrid output overlaps its inputs");
    }
    check(ids.size()==size_t(g.used)*size_t(n)&&weights.size()==ids.size(),"hybrid route geometry differs");
    std::vector<int> histogram(size_t(g.experts),0);
    for(size_t i=0;i<ids.size();++i){
        check(ids[i]>=0&&ids[i]<g.experts&&std::isfinite(weights[i]),"invalid hybrid route");
        for(size_t j=i-i%size_t(g.used);j<i;++j)check(ids[j]!=ids[i],"duplicate hybrid expert");
        ++histogram[size_t(ids[i])];
    }
    // Pure geometry/route validation above does not alter cache state. On an
    // execution failure this guard runs after CpuLease and ReadersLease have
    // joined the borrowed lanes, then restores the same upload state as the
    // native PR1 path so Engine::reset can recover.
    struct FailureGuard {Impl *p;bool complete=false;
        ~FailureGuard(){if(!complete)p->recover_uploads();}}failure{p_.get()};
    ggml_backend_synchronize(p_->backend);p_->finish_prefetch();
    std::vector<shortprefill::ExpertCost> costs(size_t(g.experts));
    const size_t expert_bytes=info.raw_bytes[0]+info.raw_bytes[1]+info.raw_bytes[2];
    for(int e=0;e<g.experts;++e){auto &c=costs[size_t(e)];c.bytes=expert_bytes;c.assignments=histogram[size_t(e)];
        c.operations=6.*double(g.embd)*double(g.expert_ff)*double(c.assignments);
        c.resident=p_->entries.count(p_->key(layer,e))!=0;
        c.cpu_tiled=c.assignments>=8&&info.types[0]!=GGML_TYPE_NVFP4&&info.types[1]!=GGML_TYPE_NVFP4&&info.types[2]!=GGML_TYPE_NVFP4;
    }
    const auto assignment=shortprefill::assign(costs,shortprefill::CostRates{});
    std::vector<int32_t> cpu_indices,gpu_indices;std::vector<int64_t> cpu_scatter,gpu_scatter;
    std::vector<int> gpu_experts;std::set<uint64_t> protected_keys;
    size_t cached_experts=0,cpu_experts=0;
    for(int e=0;e<g.experts;++e)if(histogram[size_t(e)]){
        if(assignment.cpu[size_t(e)])++cpu_experts;
        else{gpu_experts.push_back(e);protected_keys.insert(p_->key(layer,e));cached_experts+=costs[size_t(e)].resident;}
    }
    for(size_t i=0;i<ids.size();++i){if(assignment.cpu[size_t(ids[i])]){cpu_indices.push_back(int32_t(i));cpu_scatter.push_back(int64_t(i));}
        else{gpu_indices.push_back(int32_t(i));gpu_scatter.push_back(int64_t(i));}}
    // The full n>8 native CUDA MoE uses MMQ. A compact GPU subset must not
    // silently switch to MMVQ just because only1..8 assignments were resident.
    // Duplicate the last real route into private unused scatter rows; no new
    // expert is loaded, and only original rows enter the rank reduction.
    const size_t gpu_padded_rows=!gpu_indices.empty()&&gpu_indices.size()<9?9-gpu_indices.size():0;
    const size_t gpu_matrix_rows=gpu_indices.size()+gpu_padded_rows;
    for(size_t i=0;i<gpu_padded_rows;++i)gpu_scatter.push_back(int64_t(ids.size()+i));
    // Even an all-GPU cost decision must not launch a full next-layer transfer:
    // that layer's routing/cost decision has not happened yet.
    struct PrefetchGuard {Impl *p;bool old;~PrefetchGuard(){p->suppress_prefetch=old;}}prefetch_guard{p_.get(),p_->suppress_prefetch};
    p_->suppress_prefetch=true;
    if(cpu_indices.empty()){
        forward_device(layer,n,normalized,ids,weights,shared,residual,destination);
        ++p_->stat.short_hybrid_forwards;p_->stat.short_gpu_experts+=gpu_experts.size();
        p_->stat.short_cached_experts+=cached_experts;p_->stat.short_gpu_routes+=gpu_indices.size();
        p_->stat.short_cold_gpu_bytes+=assignment.gpu_cold_bytes;
        p_->stat.short_wall_seconds+=Impl::elapsed(whole_start);failure.complete=true;return;
    }
    // The all-GPU delegation above records its own routes. Only the mixed/CPU
    // branch records here, so demand counters and history are updated once.
    // Residency was decided after draining prefetch; this path never starts a
    // speculative next-layer transfer of its own.
    std::vector<float> host_input(count);ggml_backend_tensor_get(normalized,host_input.data(),0,bytes);
    std::vector<float> cpu_output;
    if(!p_->short_cpu_lane)p_->short_cpu_lane=std::make_unique<ShortPrefillCpuLane>();
    const auto cpu_submit=Impl::Clock::now();
    Impl::Clock::time_point cpu_begin{},cpu_end{};
    p_->short_cpu_lane->submit([&]{cpu_begin=Impl::Clock::now();
        cpu_routes(layer,n,host_input,ids,weights,cpu_indices,cpu_output);cpu_end=Impl::Clock::now();});
    struct CpuLease {ShortPrefillCpuLane *lane;~CpuLease(){lane->drain();}}cpu_lease{p_->short_cpu_lane.get()};
    const auto upload_begin=Impl::Clock::now();
    size_t upload_windows=0;double upload_throttle_seconds=0;
    std::vector<int32_t> remap(size_t(g.experts),-1);
    if(!gpu_experts.empty()){
        check(gpu_experts.size()<=p_->stat.slot_count,"hybrid GPU experts exceed cache slots");
        if(p_->uploads)p_->uploads->begin_batch();
        size_t outstanding_misses=0;
        for(int e:gpu_experts){
            const auto misses_before=p_->stat.misses;
            remap[size_t(e)]=int32_t(p_->load(layer,e,protected_keys));
            // A resident slot performs no DMA and must not consume the window.
            if(upload_window&&p_->uploads&&p_->stat.misses!=misses_before&&
               ++outstanding_misses>=size_t(upload_window)){
                const auto throttle_start=Impl::Clock::now();
                auto *ready=p_->uploads->checkpoint();
                // checkpoint has already enqueued the compute-stream wait.
                // Join this record on the host BEFORE another checkpoint can
                // re-record its borrowed ready event. All selected GPU keys
                // stay protected through the eventual graph completion.
                if(ready)ggml_backend_event_synchronize(ready);
                upload_throttle_seconds+=Impl::elapsed(throttle_start);
                ++upload_windows;outstanding_misses=0;
            }
        }
        if(p_->uploads){p_->uploads->finish();auto *event=p_->uploads->completion_event();if(event)ggml_backend_event_wait(p_->backend,event);}
    }
    p_->publish_foreground();
    check(p_->active_read_keys.empty(),"overlapping hybrid expert consumers are unsupported");
    p_->acquire_readers(protected_keys);
    struct ReadersLease {Impl *owner;~ReadersLease(){owner->release_readers();}} readers{p_.get()};
    const auto upload_submit_end=Impl::Clock::now();
    Context context(16*1024*1024);auto *c=context.value;
    auto *allocation_graph=ggml_new_graph_custom(c,8192,false);
    auto *gpu_graph=ggml_new_graph_custom(c,8192,false);
    auto *merge_graph=ggml_new_graph_custom(c,8192,false);
    struct Guard {ggml_backend_t backend;~Guard(){ggml_backend_synchronize(backend);clear_captures(backend);}}guard{p_->backend};
    auto *combined=ggml_new_tensor_2d(c,GGML_TYPE_F32,g.embd,int64_t(ids.size()+gpu_padded_rows));ggml_set_input(combined);ggml_set_output(combined);
    ggml_tensor *gpu_input=nullptr,*gpu_ids=nullptr,*gpu_rw=nullptr,*gpu_rows=nullptr,*gpu_token_rows=nullptr;
    std::array<ggml_tensor *,3> scale_inputs{};std::array<std::vector<float>,3> scale_values;
    ggml_tensor *gpu_output=nullptr;
    if(!gpu_indices.empty()){
        const int64_t m=int64_t(gpu_matrix_rows);
        if(device_gather){
            gpu_token_rows=ggml_new_tensor_1d(c,GGML_TYPE_I32,m);ggml_set_input(gpu_token_rows);
            auto *source=ggml_reshape_2d(c,p_->borrow_device(c,normalized),g.embd,n);
            gpu_input=ggml_reshape_3d(c,ggml_get_rows(c,source,gpu_token_rows),g.embd,1,m);
        }else{
            gpu_input=ggml_new_tensor_3d(c,GGML_TYPE_F32,g.embd,1,m);ggml_set_input(gpu_input);
        }
        gpu_ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,1,m);gpu_rw=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,1,m);
        gpu_rows=ggml_new_tensor_1d(c,GGML_TYPE_I64,m);
        for(auto *t:{gpu_ids,gpu_rw,gpu_rows})ggml_set_input(t);
        auto mm=[&](int role,ggml_tensor *activation){
            auto *w=p_->weight_view(c,layer,role);auto *out=ggml_mul_mat_id(c,w,activation,gpu_ids);
            if(w->type==GGML_TYPE_NVFP4)ggml_prec_set_acc(out,GGML_PREC_BF16);
            if(p_->stat.q8_activations)ggml_prec_set_src(out,GGML_PREC_Q8,1);
            if(info.has_scale[size_t(role)]){
                auto *s=ggml_new_tensor_1d(c,GGML_TYPE_F32,int64_t(p_->stat.slot_count));ggml_set_input(s);scale_inputs[size_t(role)]=s;
                scale_values[size_t(role)].assign(p_->stat.slot_count,1.f);
                for(int e:gpu_experts)scale_values[size_t(role)][size_t(remap[size_t(e)])]=info.scales[size_t(role)][size_t(e)];
                auto *repeat=ggml_repeat_4d(c,ggml_reshape_3d(c,s,1,int64_t(p_->stat.slot_count),1),1,int64_t(p_->stat.slot_count),m,1);
                out=ggml_mul(c,out,ggml_get_rows(c,repeat,gpu_ids));
            }return out;
        };
        auto *gate=mm(0,gpu_input),*up=mm(1,gpu_input);
        auto *activation=ggml_swiglu_oai(c,gate,up,1.702f,7.f);
        auto *weighted=ggml_mul(c,mm(2,activation),gpu_rw);
        gpu_output=ggml_set_rows(c,combined,ggml_reshape_2d(c,weighted,g.embd,m),gpu_rows);
        ggml_set_output(gpu_output);ggml_build_forward_expand(gpu_graph,gpu_output);
        ggml_build_forward_expand(allocation_graph,gpu_output);
    }
    auto *cpu_input=ggml_new_tensor_2d(c,GGML_TYPE_F32,g.embd,int64_t(cpu_indices.size()));ggml_set_input(cpu_input);
    auto *cpu_rows=ggml_new_tensor_1d(c,GGML_TYPE_I64,int64_t(cpu_indices.size()));ggml_set_input(cpu_rows);
    auto *merged=ggml_set_rows(c,combined,cpu_input,cpu_rows);
    auto *original_rows=gpu_padded_rows?ggml_view_2d(c,merged,g.embd,int64_t(ids.size()),merged->nb[1],0):merged;
    auto *ranked=ggml_reshape_3d(c,original_rows,g.embd,g.used,n);ggml_tensor *moe=nullptr;
    for(int rank=0;rank<g.used;++rank){auto *term=ggml_view_2d(c,ranked,g.embd,n,ranked->nb[2],size_t(rank)*ranked->nb[1]);moe=moe?ggml_add(c,moe,term):term;}
    auto *ff=ggml_add(c,ggml_cont(c,moe),ggml_reshape_2d(c,p_->borrow_device(c,shared),g.embd,n));
    auto *next=ggml_add(c,ff,ggml_reshape_2d(c,p_->borrow_device(c,residual),g.embd,n));
    auto *output=ggml_cpy(c,next,ggml_reshape_2d(c,p_->borrow_device(c,destination),g.embd,n));ggml_set_output(output);
    ggml_build_forward_expand(merge_graph,output);ggml_build_forward_expand(allocation_graph,output);
    for(int i=0;i<ggml_graph_n_nodes(allocation_graph);++i){auto *node=ggml_graph_node(allocation_graph,i);
        check(ggml_backend_dev_supports_op(p_->device,node),std::string("hybrid GPU does not support ")+ggml_op_name(node->op));}
    check(ggml_gallocr_reserve(p_->allocator,allocation_graph),"hybrid scratch reserve failed");
    check(ggml_gallocr_alloc_graph(p_->allocator,allocation_graph),"hybrid scratch allocation failed");
    p_->stat.scratch_bytes=std::max(p_->stat.scratch_bytes,ggml_gallocr_get_buffer_size(p_->allocator,0));
    const auto begin=Impl::Clock::now();
    if(!gpu_indices.empty()){
        std::vector<float> input,rw(gpu_matrix_rows);std::vector<int32_t> mapped(gpu_matrix_rows),token_rows;
        if(device_gather)token_rows.resize(gpu_matrix_rows);else input.resize(gpu_matrix_rows*size_t(g.embd));
        for(size_t i=0;i<gpu_matrix_rows;++i){const size_t route=size_t(gpu_indices[std::min(i,gpu_indices.size()-1)]),token=route/size_t(g.used);
            if(device_gather)token_rows[i]=int32_t(token);
            else std::memcpy(input.data()+i*size_t(g.embd),host_input.data()+token*size_t(g.embd),size_t(g.embd)*sizeof(float));
            mapped[i]=remap[size_t(ids[route])];rw[i]=weights[route];}
        if(device_gather)ggml_backend_tensor_set(gpu_token_rows,token_rows.data(),0,token_rows.size()*sizeof(int32_t));
        else ggml_backend_tensor_set(gpu_input,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(gpu_ids,mapped.data(),0,mapped.size()*sizeof(int32_t));
        ggml_backend_tensor_set(gpu_rw,rw.data(),0,rw.size()*sizeof(float));
        ggml_backend_tensor_set(gpu_rows,gpu_scatter.data(),0,gpu_scatter.size()*sizeof(int64_t));
        for(size_t role=0;role<3;++role)if(scale_inputs[role])ggml_backend_tensor_set(scale_inputs[role],scale_values[role].data(),0,scale_values[role].size()*sizeof(float));
        check(ggml_backend_graph_compute_async(p_->backend,gpu_graph)==GGML_STATUS_SUCCESS,"hybrid GPU compute failed");
    }
    // The persistent CPU master started before the first cold-weight upload.
    // Join its borrowed buffers before scatter/reduction or exception cleanup.
    p_->short_cpu_lane->wait();
    const double cpu_seconds=std::chrono::duration<double>(cpu_end-cpu_begin).count();
    check(cpu_output.size()==cpu_indices.size()*size_t(g.embd),"hybrid CPU callback returned invalid rows");
    ggml_backend_synchronize(p_->backend);
    const auto gpu_end=Impl::Clock::now();
    ggml_backend_tensor_set(cpu_input,cpu_output.data(),0,cpu_output.size()*sizeof(float));
    ggml_backend_tensor_set(cpu_rows,cpu_scatter.data(),0,cpu_scatter.size()*sizeof(int64_t));
    check(ggml_backend_graph_compute(p_->backend,merge_graph)==GGML_STATUS_SUCCESS,"hybrid rank reduction failed");
    ggml_backend_synchronize(p_->backend);++p_->stat.forwards;p_->stat.graph_batches+=gpu_indices.empty()?1:2;
    ++p_->stat.short_hybrid_forwards;p_->stat.short_cpu_experts+=cpu_experts;
    p_->stat.short_gpu_experts+=gpu_experts.size();p_->stat.short_cached_experts+=cached_experts;
    p_->stat.short_cpu_routes+=cpu_indices.size();p_->stat.short_gpu_routes+=gpu_indices.size();
    p_->stat.short_cold_gpu_bytes+=assignment.gpu_cold_bytes;p_->stat.short_cpu_seconds+=cpu_seconds;
    p_->stat.short_wall_seconds+=Impl::elapsed(whole_start);
    if(p_->profile)p_->stat.execute_seconds+=Impl::elapsed(begin);
    failure.complete=true;
    if(p_->profile)std::fprintf(stderr,"M31 shortprefill layer=%d n=%d cpu_experts=%zu gpu_experts=%zu cached_experts=%zu cpu_routes=%zu gpu_routes=%zu cold_gpu_bytes=%llu estimated_cpu_ms=%.3f estimated_gpu_ms=%.3f cpu_ms=%.3f execute_ms=%.3f whole_ms=%.3f cpu_submit_ms=%.3f cpu_begin_ms=%.3f cpu_end_ms=%.3f upload_begin_ms=%.3f upload_submit_end_ms=%.3f gpu_join_ms=%.3f upload_window=%d upload_windows=%zu upload_throttle_ms=%.3f device_gather=%d gpu_activation_h2d_bytes=%llu gather_index_h2d_bytes=%llu padded_rows=%zu\n",
        layer,n,cpu_experts,gpu_experts.size(),cached_experts,cpu_indices.size(),gpu_indices.size(),
        (unsigned long long)assignment.gpu_cold_bytes,1000*assignment.cpu_seconds,1000*assignment.gpu_seconds,1000*cpu_seconds,1000*Impl::elapsed(begin),1000*Impl::elapsed(whole_start),
        1000*std::chrono::duration<double>(cpu_submit-whole_start).count(),
        1000*std::chrono::duration<double>(cpu_begin-whole_start).count(),
        1000*std::chrono::duration<double>(cpu_end-whole_start).count(),
        1000*std::chrono::duration<double>(upload_begin-whole_start).count(),
        1000*std::chrono::duration<double>(upload_submit_end-whole_start).count(),
        1000*std::chrono::duration<double>(gpu_end-whole_start).count(),
        upload_window,upload_windows,1000*upload_throttle_seconds,int(device_gather),
        (unsigned long long)(device_gather?0:gpu_matrix_rows*size_t(g.embd)*sizeof(float)),
        (unsigned long long)(device_gather?gpu_matrix_rows*sizeof(int32_t):0),gpu_padded_rows);
}
