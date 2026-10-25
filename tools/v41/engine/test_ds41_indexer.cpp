// test_ds41_indexer [--cuda] [--large] [--bench] [--replay FILE.idx ...]
// IH=32, ID=128, top-512. Visibility must match the legacy additive mask bitwise.
// The opt-in MXFP4 WMMA path is compared to vec, including actual GPU top-k sets.
// Replay files are captured by STRATA_INDEXER_DUMP (see INDEXER_FAST.md).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name,value ? value : "");
#else
    if(value) setenv(name,value,1); else unsetenv(name);
#endif
}
struct Input { ggml_type type; int64_t ne[4]; std::vector<uint8_t> bytes; };
struct Fixture { Input v[4]; std::string name; bool exact_wmma=false; };
static std::vector<uint8_t> bytes(const std::vector<float> & v) {
    std::vector<uint8_t> b(v.size()*4); std::memcpy(b.data(),v.data(),b.size()); return b;
}
static Fixture synthetic(int nk,int nq,int ns,bool broadcast,bool quant,int edge=0) {
    constexpr int D=128,H=32;
    Fixture f {{{GGML_TYPE_F32,{D,H,nq,ns},{}},
                {quant?GGML_TYPE_MXFP4:GGML_TYPE_F32,{D,1,nk,ns},{}},
                {GGML_TYPE_F32,{H,nq,1,ns},{}},
                {GGML_TYPE_F32,{1,nq,1,broadcast?1:ns},{}}},
               "synthetic nk="+std::to_string(nk)+" nq="+std::to_string(nq)+" ns="+std::to_string(ns)+
               " broadcast="+std::to_string(broadcast)+" quant="+std::to_string(quant)+" edge="+std::to_string(edge),edge!=0};
    std::mt19937 rng(1025); static const float mag[]={0,.5f,1,1.5f,2,3,4,6};
    std::vector<float> q((size_t)D*H*nq*ns),w((size_t)H*nq*ns),nv((size_t)nq*f.v[3].ne[3]);
    for(size_t i=0;i<q.size();++i) q[i]=std::ldexp(mag[rng()%8]*(rng()%2?1:-1),-4+(int)(i/32%4));
    if(edge>=1 && edge<=4) for(int s=0;s<ns;++s) for(int t=0;t<nq;++t)
        q[((size_t)s*nq+t)*D*H]=edge==1?.1f:edge==2?0x1p-25f:edge==3?0x1p16f:0x1p-24f;
    for(auto & a:w) a=((int)(rng()%2001)-1000)*.00013f;
    const int limits[]={0,1,31,32,33,63,64,65,nk/2,nk-1,nk,nk+1};
    for(int s=0;s<f.v[3].ne[3];++s) for(int t=0;t<nq;++t)
        nv[s*nq+t]=edge?float(nk):(nq==1?nk*.73f:float(limits[(s+t)%12]));
    f.v[0].bytes=bytes(q); f.v[2].bytes=bytes(w); f.v[3].bytes=bytes(nv);
    if(quant) {
        f.v[1].bytes.resize((size_t)nk*ns*4*17);
        for(size_t b=0;b<f.v[1].bytes.size()/17;++b) {
            f.v[1].bytes[b*17]=edge==5?113:edge==6?141:edge==7?0:uint8_t(123+b%4);
            for(int j=1;j<17;++j) f.v[1].bytes[b*17+j]=edge==7?0:uint8_t(rng()%256);
        }
    } else {
        std::vector<float> k((size_t)D*nk*ns);
        for(auto & a:k) a=std::ldexp(mag[rng()%8]*(rng()%2?1:-1),-3);
        f.v[1].bytes=bytes(k);
    }
    return f;
}
static Fixture replay(const char * path) {
    Fixture f; f.name=path;
    std::FILE * fp=std::fopen(path,"rb"); char magic[8];
    if(!fp || std::fread(magic,1,8,fp)!=8 || std::memcmp(magic,"DS41IDX1",8)) std::abort();
    for(auto & v:f.v) {
        int32_t type; uint64_t size;
        if(std::fread(&type,4,1,fp)!=1 || std::fread(v.ne,8,4,fp)!=4 || std::fread(&size,8,1,fp)!=1) std::abort();
        v.type=(ggml_type)type; v.bytes.resize(size);
        if(std::fread(v.bytes.data(),1,size,fp)!=size) std::abort();
    }
    std::fclose(fp); return f;
}
static int check(ggml_backend_t be,const Fixture & f,bool cuda,bool bench) {
    const int nk=(int)f.v[1].ne[2],nq=(int)f.v[0].ne[2],ns=(int)f.v[0].ne[3],top=std::min(512,nk);
    ggml_init_params ip {ggml_tensor_overhead()*256+4*ggml_graph_overhead(),nullptr,true};
    auto * c=ggml_init(ip); ggml_tensor * v[4];
    for(int j=0;j<4;++j) {v[j]=ggml_new_tensor(c,f.v[j].type,4,f.v[j].ne); if(ggml_nbytes(v[j])!=f.v[j].bytes.size()) std::abort();}
    // Exercise nvis views with nontrivial query/stream strides, as well as stream broadcast.
    auto * padded=ggml_new_tensor_4d(c,GGML_TYPE_F32,2,nq,1,v[3]->ne[3]);
    auto * nv=ggml_view_4d(c,padded,1,nq,1,padded->ne[3],padded->nb[1],padded->nb[2],padded->nb[3],4);
    auto * mask=ggml_new_tensor_4d(c,GGML_TYPE_F16,nk,nq,1,v[3]->ne[3]);
    // These inputs are reused across backend calls. INPUT controls allocation order;
    // OUTPUT also prevents gallocr from recycling their storage after the last reader.
    for(auto * t:{v[0],v[1],v[2],padded,mask}) {ggml_set_input(t);ggml_set_output(t);}
    auto * old=ggml_lightning_indexer(c,v[0],v[1],v[2],mask);
    auto * fast=ggml_lightning_indexer(c,v[0],v[1],v[2],nv);
    auto * ids=ggml_cont(c,ggml_top_k(c,fast,top));
    auto * selected_mask=ggml_log(c,ggml_step(c,ggml_add(c,ggml_scale(c,ggml_cast(c,ids,GGML_TYPE_F32),-1.0f),nv)));
    for(auto * t:{old,fast,ids,selected_mask}) ggml_set_output(t);
    auto * gf=ggml_new_graph(c); ggml_build_forward_expand(gf,old); ggml_build_forward_expand(gf,fast); ggml_build_forward_expand(gf,selected_mask);
    auto al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if(!ggml_gallocr_alloc_graph(al,gf)) std::abort();
    for(int j=0;j<3;++j) ggml_backend_tensor_set(v[j],f.v[j].bytes.data(),0,f.v[j].bytes.size());
    const float * nvp=(const float *)f.v[3].bytes.data();
    std::vector<float> nvv(ggml_nelements(padded),-123);
    std::vector<ggml_fp16_t> mv(ggml_nelements(mask));
    for(int s=0;s<v[3]->ne[3];++s) for(int t=0;t<nq;++t) {
        const float n=nvp[s*nq+t]; nvv[2*(s*nq+t)+1]=n;
        for(int i=0;i<nk;++i) mv[((size_t)s*nq+t)*nk+i]=ggml_fp32_to_fp16(i<n?0.0f:-INFINITY);
    }
    ggml_backend_tensor_set(padded,nvv.data(),0,nvv.size()*4); ggml_backend_tensor_set(mask,mv.data(),0,mv.size()*2);
    env("GGML_CUDA_INDEXER_VEC","1");
    if(ggml_backend_graph_compute(be,gf)!=GGML_STATUS_SUCCESS) std::abort();
    std::vector<float> a(ggml_nelements(old)),b(a.size()); std::vector<int32_t> ia((size_t)top*nq*ns),ib(ia.size());
    ggml_backend_tensor_get(old,a.data(),0,a.size()*4); ggml_backend_tensor_get(fast,b.data(),0,b.size()*4);
    ggml_backend_tensor_get(ids,ia.data(),0,ia.size()*4);
    size_t vis_bad=0; for(size_t i=0;i<a.size();++i) vis_bad+=std::memcmp(&a[i],&b[i],4)!=0;
    size_t score_bad=0,hits=0,total=0,rows_bad=0; double max_abs=0,max_rel=0,max_scaled=0;
    if(cuda && f.v[1].type==GGML_TYPE_MXFP4) {
        env("GGML_CUDA_INDEXER_VEC",nullptr); env("GGML_CUDA_MXFP4_WMMA","1");
        if(ggml_backend_graph_compute(be,gf)!=GGML_STATUS_SUCCESS) std::abort();
        ggml_backend_tensor_get(fast,b.data(),0,b.size()*4); ggml_backend_tensor_get(ids,ib.data(),0,ib.size()*4);
        for(size_t i=0;i<a.size();++i) {
            if(std::memcmp(&a[i],&b[i],4)) ++score_bad;
            if(std::isfinite(a[i]) && std::isfinite(b[i])) {double d=std::abs((double)a[i]-b[i]);max_abs=std::max(max_abs,d);max_rel=std::max(max_rel,d/std::max(1e-20,std::abs((double)a[i])));max_scaled=std::max(max_scaled,d/(1.+std::abs((double)a[i])));}
            else if(a[i]!=b[i]) {max_abs=INFINITY;}
        }
        for(int row=0;row<nq*ns;++row) {
            std::vector<int32_t> x(ia.begin()+(size_t)row*top,ia.begin()+(size_t)(row+1)*top);
            std::vector<int32_t> y(ib.begin()+(size_t)row*top,ib.begin()+(size_t)(row+1)*top);
            const float count=nvp[((row/nq)%v[3]->ne[3])*nq+row%nq];
            x.erase(std::remove_if(x.begin(),x.end(),[&](int i){return !(i<count);}),x.end());
            y.erase(std::remove_if(y.begin(),y.end(),[&](int i){return !(i<count);}),y.end());
            std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());
            std::vector<int32_t> common;std::set_intersection(x.begin(),x.end(),y.begin(),y.end(),std::back_inserter(common));
            hits+=common.size();total+=x.size();
            if(x!=y) {
                ++rows_bad;
                if(rows_bad<=8) {std::vector<int32_t> diff;std::set_symmetric_difference(x.begin(),x.end(),y.begin(),y.end(),std::back_inserter(diff));
                    std::printf("  top512 difference row=%d nvis=%.0f ids:",row,count);
                    for(size_t j=0;j<std::min<size_t>(diff.size(),8);++j) {int id=diff[j];std::printf(" %d(%.9g/%.9g)",id,a[(size_t)row*nk+id],b[(size_t)row*nk+id]);} std::puts("");}
            }
        }
    }
    std::vector<float> sm(ia.size());ggml_backend_tensor_get(selected_mask,sm.data(),0,sm.size()*4);
    const auto & selected=cuda && f.v[1].type==GGML_TYPE_MXFP4?ib:ia;
    size_t mask_bad=0;
    for(int row=0;row<nq*ns;++row)for(int j=0;j<top;++j) {
        const float count=nvp[((row/nq)%v[3]->ne[3])*nq+row%nq];
        const float expected=selected[(size_t)row*top+j]<count?0.0f:-INFINITY;
        mask_bad+=std::memcmp(&expected,&sm[(size_t)row*top+j],4)!=0;
    }
    std::printf("  selected_mask_diff=%zu/%zu\n",mask_bad,sm.size());
    int emin=255,emax=0;size_t unsafe_k=0,half_bad=0,subnormal_q=0;float qmin=INFINITY,qmax=0;
    const float * qp=(const float *)f.v[0].bytes.data();
    for(size_t i=0;i<f.v[0].bytes.size()/4;++i) {float x=std::abs(qp[i]); if(x){qmin=std::min(qmin,x);qmax=std::max(qmax,x);}half_bad+=ggml_fp16_to_fp32(ggml_fp32_to_fp16(qp[i]))!=qp[i];subnormal_q+=x>0 && x<0x1p-14f;}
    if(f.v[1].type==GGML_TYPE_MXFP4) for(int s=0;s<ns;++s) {
        int visible=0;for(int t=0;t<nq;++t) visible=std::max(visible,std::min(nk,(int)std::ceil(nvp[(s%v[3]->ne[3])*nq+t])));
        for(int i=0;i<visible*4;++i) {int e=f.v[1].bytes[((size_t)s*nk*4+i)*17];emin=std::min(emin,e);emax=std::max(emax,e);unsafe_k+=e<114||e>140;}
    }
    std::printf("%s scores=%zu visibility_diff=%zu wmma_diff=%zu max_abs=%.9g max_rel=%.9g top512=%zu/%zu (%.9f%%) rows_changed=%zu/%d\n",f.name.c_str(),a.size(),vis_bad,score_bad,max_abs,max_rel,hits,total,total?100.*hits/total:100.,rows_bad,nq*ns);
    std::printf("  Q_abs=[%.9g,%.9g] Q_half_inexact=%zu Q_subnormal=%zu K_e=[%d,%d] K_scale_pow2=[%d,%d] K_unsafe=%zu\n",qmin,qmax,half_bad,subnormal_q,emin,emax,emin-127,emax-127,unsafe_k);
    if(bench && cuda && f.v[1].type==GGML_TYPE_MXFP4) {
        auto * g_old=ggml_new_graph(c);ggml_build_forward_expand(g_old,old);
        auto * g_new=ggml_new_graph(c);ggml_build_forward_expand(g_new,fast);
        double ms[3]={};
        for(int mode=0;mode<3;++mode) {
            env("GGML_CUDA_INDEXER_VEC",mode==2?nullptr:"1");
            auto * graph=mode==0?g_old:g_new;
            ggml_backend_graph_compute(be,graph);ggml_backend_synchronize(be);
            auto t0=std::chrono::steady_clock::now();
            constexpr int repeats=10;
            for(int repeat=0;repeat<repeats;++repeat) {
                if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS) std::abort();
            }
            ggml_backend_synchronize(be);
            ms[mode]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/repeats;
        }
        std::printf("  indexer_compute_ms vec_mask=%.6f vec_nvis=%.6f wmma_nvis=%.6f (mean of 10, launch+sync included)\n",ms[0],ms[1],ms[2]);
    }
    bool fail=vis_bad || mask_bad || max_scaled>2e-5 || !std::isfinite(max_abs) || (f.exact_wmma && score_bad) || (total && (double)hits/total<.999);
    ggml_gallocr_free(al);ggml_free(c); return fail?1:0;
}
int main(int argc,char **argv) {
    bool cuda=false,large=false,bench=false;std::vector<std::string> files;
    for(int i=1;i<argc;++i) {std::string s=argv[i];if(s=="--cuda")cuda=true;else if(s=="--large")large=true;else if(s=="--bench")bench=true;else if(s!="--replay")files.push_back(s);}
    env("GGML_CUDA_DISABLE_GRAPHS","1"); // re-dispatch the same graph under both arithmetic paths
    ggml_backend_t be=nullptr;
#if defined(GGML_USE_CUDA)
    if(cuda)be=ggml_backend_cuda_init(0);
#endif
    if(cuda&&!be)return 77;if(!be){be=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(be,8);}
    int fails=0;
    if(files.empty()) {
        for(bool quant:{false,true})for(int nk:{1,31,32,33,63,64,65,512,1024}) {
            fails+=check(be,synthetic(nk,13,2,false,quant),cuda,bench); fails+=check(be,synthetic(nk,3,2,true,quant),cuda,bench);
        }
        if(cuda)for(int edge=1;edge<=7;++edge)fails+=check(be,synthetic(1024,3,1,true,true,edge),cuda,bench);
        if(large)for(int nk:{32768,131072,262144})fails+=check(be,synthetic(nk,128,1,true,true),cuda,bench);
    } else for(const auto & path:files)fails+=check(be,replay(path.c_str()),cuda,bench);
    ggml_backend_free(be);std::printf("%s\n",fails?"FAIL":"PASS");return fails?1:0;
}
