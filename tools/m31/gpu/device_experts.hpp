// Included inside GpuExperts::Impl. Borrowed dense descriptors and their buffers
// remain alive through this synchronously completing device-only expert call.
ggml_tensor * borrow_device(ggml_context * c,ggml_tensor * source) const {
    auto * leaf=ggml_dup_tensor(c,source);
    auto * buffer=source->view_src?source->view_src->buffer:source->buffer;
    check(buffer&&ggml_backend_tensor_alloc(buffer,leaf,source->data)==GGML_STATUS_SUCCESS,
          "cannot bind borrowed device tensor");
    ggml_set_input(leaf);
    return leaf;
}
void execute_device(int layer,int n,ggml_tensor * normalized,
                    const std::vector<int32_t> &mapped_ids,
                    const std::vector<int32_t> &original_ids,
                    const std::vector<float> &route_weights,
                    const std::array<std::vector<float>,3> &scales,
                    ggml_tensor *shared,ggml_tensor *residual,ggml_tensor *destination,
                    ggml_backend_event_t weights_ready,
                    const std::set<uint64_t> &protected_keys) {
    (void)original_ids;(void)weights_ready;
    Context context(8*1024*1024);
    auto *c=context.value;
    auto *graph=ggml_new_graph_custom(c,8192,false);
    struct Guard{ggml_backend_t backend;~Guard(){ggml_backend_synchronize(backend);clear_captures(backend);}}guard{backend};
    auto *input=ggml_reshape_3d(c,borrow_device(c,normalized),g.embd,1,n);
    auto *ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,g.used,n);
    auto *rw=ggml_new_tensor_3d(c,GGML_TYPE_F32,1,g.used,n);
    ggml_set_input(ids);ggml_set_input(rw);
    std::array<ggml_tensor *,3> scale_inputs{};
    const auto &info=layers[size_t(layer)];
    auto mm=[&](int role,ggml_tensor *activation){
        auto *weight=weight_view(c,layer,role);
        auto *out=ggml_mul_mat_id(c,weight,activation,ids);
        if(weight->type==GGML_TYPE_NVFP4)ggml_prec_set_acc(out,GGML_PREC_BF16);
        if(stat.q8_activations)ggml_prec_set_src(out,GGML_PREC_Q8,1);
        if(info.has_scale[size_t(role)]){
            auto *scale=ggml_new_tensor_1d(c,GGML_TYPE_F32,int64_t(stat.slot_count));
            ggml_set_input(scale);scale_inputs[size_t(role)]=scale;
            auto *s=ggml_repeat_4d(c,ggml_reshape_3d(c,scale,1,int64_t(stat.slot_count),1),1,int64_t(stat.slot_count),n,1);
            out=ggml_mul(c,out,ggml_get_rows(c,s,ids));
        }
        return out;
    };
    auto *gate=mm(0,input),*up=mm(1,input);
    auto *activation=ggml_swiglu_oai(c,gate,up,1.702f,7.f);
    auto *down=mm(2,activation);
    auto *weighted=ggml_mul(c,down,rw);
    ggml_tensor *moe=nullptr;
    for(int64_t rank=0;rank<g.used;++rank){
        auto *term=ggml_view_2d(c,weighted,g.embd,n,weighted->nb[2],size_t(rank)*weighted->nb[1]);
        moe=moe?ggml_add(c,moe,term):term;
    }
    moe=ggml_cont(c,moe);ggml_set_output(moe);
    // Preserve the CPU residual order and keep the old native MoE graph boundary.
    auto *ff=ggml_add(c,moe,ggml_reshape_2d(c,borrow_device(c,shared),g.embd,n));
    ggml_set_output(ff);
    auto *combined=ggml_add(c,ff,ggml_reshape_2d(c,borrow_device(c,residual),g.embd,n));
    auto *copy=ggml_cpy(c,combined,ggml_reshape_2d(c,borrow_device(c,destination),g.embd,n));
    ggml_set_output(copy);ggml_build_forward_expand(graph,copy);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto *node=ggml_graph_node(graph,i);check(ggml_backend_dev_supports_op(device,node),std::string("device FFN does not support ")+ggml_op_name(node->op));}
    check(ggml_gallocr_reserve(allocator,graph),"device FFN scratch reserve failed");
    check(ggml_gallocr_alloc_graph(allocator,graph),"device FFN scratch allocation failed");
    stat.scratch_bytes=std::max(stat.scratch_bytes,ggml_gallocr_get_buffer_size(allocator,0));
    ggml_backend_tensor_set(ids,mapped_ids.data(),0,mapped_ids.size()*sizeof(int32_t));
    ggml_backend_tensor_set(rw,route_weights.data(),0,route_weights.size()*sizeof(float));
    for(size_t role=0;role<3;++role)if(scale_inputs[role])ggml_backend_tensor_set(scale_inputs[role],scales[role].data(),0,stat.slot_count*sizeof(float));
    begin_prefetch(layer,n,protected_keys);
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"device FFN graph compute failed");
    ggml_backend_synchronize(backend);
    ++stat.graph_batches;
}
