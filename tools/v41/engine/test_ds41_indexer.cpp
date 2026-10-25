// test_ds41_indexer [--cuda] [--large]: old additive mask vs per-query visibility.
// Real V4.1 geometry (IH=32, ID=128, top-512), partial blocks, empty rows,
// padded/non-monotonic visibility, multiple streams and broadcast/strided nvis.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int check(ggml_backend_t be, int nk, int nq, int ns, bool broadcast, bool quant) {
    constexpr int D = 128, H = 32;
    ggml_init_params ip { ggml_tensor_overhead()*128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    auto * q = ggml_new_tensor_4d(c, GGML_TYPE_F32, D, H, nq, ns);
    auto * k = ggml_new_tensor_4d(c, quant ? GGML_TYPE_MXFP4 : GGML_TYPE_F32, D, 1, nk, ns);
    auto * w = ggml_new_tensor_4d(c, GGML_TYPE_F32, H, nq, 1, ns);
    auto * counts = ggml_new_tensor_4d(c, GGML_TYPE_F32, 2, nq, 1, broadcast ? 1 : ns);
    auto * nv = ggml_view_4d(c, counts, 1, nq, 1, counts->ne[3], counts->nb[1], counts->nb[2], counts->nb[3], 4);
    auto * mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, nk, nq, 1, counts->ne[3]);
    auto * old = ggml_lightning_indexer(c, q, k, w, mask);
    auto * fast = ggml_lightning_indexer(c, q, k, w, nv);
    ggml_set_output(old); ggml_set_output(fast);
    auto * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, old); ggml_build_forward_expand(gf, fast);
    auto al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(al, gf)) std::abort();
    std::mt19937 rng(1025);
    static const float mag[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    std::vector<float> qv(ggml_nelements(q)), kv((size_t)D*nk*ns), wv(ggml_nelements(w));
    for (size_t i=0; i<qv.size(); ++i) qv[i] = std::ldexp(mag[rng()%8]*(rng()%2 ? 1 : -1), -4 + (int)(i/32%4));
    for (size_t i=0; i<kv.size(); ++i) kv[i] = std::ldexp(mag[rng()%8]*(rng()%2 ? 1 : -1), -4 + (int)(i/32%4));
    for (auto & v : wv) v = ((int)(rng()%2001)-1000)*.00013f;
    std::vector<float> nvv(ggml_nelements(counts), -123);
    std::vector<ggml_fp16_t> mv(ggml_nelements(mask));
    const int edges[] = {0, 1, 31, 32, 33, 63, 64, 65, nk/2, nk-1, nk, nk+1};
    for (int s=0; s<counts->ne[3]; ++s) for (int t=0; t<nq; ++t) {
        float n = nq == 1 ? nk*.73f : (float)edges[(t+s)%12];
        nvv[2*(s*nq+t)+1] = n;
        for (int i=0; i<nk; ++i) mv[(s*nq+t)*nk+i] = ggml_fp32_to_fp16(i<n ? 0.0f : -INFINITY);
    }
    ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
    if (quant) {
        std::vector<unsigned char> packed(ggml_nbytes(k));
        ggml_quantize_chunk(GGML_TYPE_MXFP4,kv.data(),packed.data(),0,(int64_t)nk*ns,D,nullptr);
        ggml_backend_tensor_set(k,packed.data(),0,packed.size());
    } else ggml_backend_tensor_set(k,kv.data(),0,kv.size()*4);
    ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);
    ggml_backend_tensor_set(counts,nvv.data(),0,nvv.size()*4);
    ggml_backend_tensor_set(mask,mv.data(),0,mv.size()*2);
    auto t0=std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(be,gf) != GGML_STATUS_SUCCESS) std::abort();
    std::vector<float> a(ggml_nelements(old)), b(a.size());
    ggml_backend_tensor_get(old,a.data(),0,a.size()*4); ggml_backend_tensor_get(fast,b.data(),0,b.size()*4);
    size_t bad=0;
    for (size_t i=0;i<a.size();++i) if (std::memcmp(&a[i],&b[i],4)) {
        if (++bad<4) std::printf("  mismatch %zu old %.9g nvis %.9g\n",i,a[i],b[i]);
    }
    auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    std::printf("%s nk=%d nq=%d streams=%d broadcast=%d scores=%zu differ=%zu ms=%.3f\n", quant?"mxfp4":"f32",nk,nq,ns,broadcast,a.size(),bad,ms);
    ggml_gallocr_free(al); ggml_free(c);
    return bad ? 1 : 0;
}
int main(int argc,char **argv) {
    bool cuda=false,large=false;
    for(int i=1;i<argc;++i) {cuda |= std::string(argv[i])=="--cuda"; large |= std::string(argv[i])=="--large";}
    ggml_backend_t be=nullptr;
#if defined(GGML_USE_CUDA)
    if(cuda) be=ggml_backend_cuda_init(0);
#endif
    if(cuda && !be) return 77;
    if(!be) {be=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be,8);}
    int fails=0;
    for(bool quant:{false,true}) for(int nk:{1,31,32,33,63,64,65,512,1024}) {
        fails+=check(be,nk,13,2,false,quant);
        fails+=check(be,nk,3,2,true,quant);
    }
    if(large) for(int nk:{32768,131072,262144}) fails+=check(be,nk,128,1,true,true);
    ggml_backend_free(be);
    std::printf("%s\n",fails?"FAIL":"PASS: visibility scores match the additive-mask path bit for bit");
    return fails?1:0;
}
