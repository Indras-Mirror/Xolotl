// Standalone numerical/speed comparison with q36's original FP16 cuBLAS path.
// q36-flash-check [activation.bin] (see STRATA_Q36_FA_DUMP in q36_engine.cu).
#include "strata/qwen35/flash.hpp"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)
#define BL(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s failed\n", #x); std::exit(1); } } while (0)
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { CK(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
};
__global__ void half_q(const float* q, __half* h, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half_rn(q[i]);
}
__device__ uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
template<class T> __global__ void random_data(T* data, size_t n, uint32_t seed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) data[i] = T((float(hash(uint32_t(i) ^ seed) >> 8) * 0x1p-24f - .5f) * 3.464101615f);
}
__global__ void softmax(const float* s, __half* p, int ld, int T, int P, int p0) {
    __shared__ float sh[256];
    const int row = blockIdx.y * T + blockIdx.x, lane = threadIdx.x, n = p0 + blockIdx.x + 1;
    s += size_t(row) * ld; p += size_t(row) * ld;
    float mx = -INFINITY;
    for (int i = lane; i < n; i += 256) mx = fmaxf(mx, s[i] * .0625f);
    sh[lane] = mx; __syncthreads();
    for (int d = 128; d; d /= 2) { if (lane < d) sh[lane] = fmaxf(sh[lane], sh[lane+d]); __syncthreads(); }
    mx = sh[0]; __syncthreads();
    float sum = 0.f;
    for (int i = lane; i < n; i += 256) sum += expf(s[i] * .0625f - mx);
    sh[lane] = sum; __syncthreads();
    for (int d = 128; d; d /= 2) { if (lane < d) sh[lane] += sh[lane+d]; __syncthreads(); }
    const float inv = 1.f / sh[0];
    for (int i = lane; i < P; i += 256) p[i] = __float2half(i < n ? expf(s[i] * .0625f - mx) * inv : 0.f);
}
__global__ void gate_mul(float* out, const float* gate, size_t n) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] /= 1.f + expf(-gate[i]);
}
struct Reference {
    cublasHandle_t handle{};
    cudaStream_t st;
    Buffer<float> scores{64u << 20};
    Buffer<__half> probs{64u << 20}, q16;
    Reference(int max_rows, cudaStream_t stream) : st(stream), q16(size_t(max_rows) * 256) {
        BL(cublasCreate(&handle)); BL(cublasSetStream(handle, st));
    }
    ~Reference() { cublasDestroy(handle); }
    void run(const float* q, const __half* k, const __half* v, const float* gate, float* out,
             int T, int H, int HKV, int P) {
        const int G = H / HKV, ld = (P + 7) / 8 * 8, p0 = P - T;
        const int tq = std::max(1, std::min(T, int((64u << 20) / (size_t(H) * ld))));
        const float one = 1.f, zero = 0.f;
        half_q<<<(size_t(T)*H*256+255)/256,256,0,st>>>(q,q16.p,size_t(T)*H*256);
        for (int t0 = 0; t0 < T; t0 += tq) {
            const int nt = std::min(tq, T-t0);
            for (int h = 0; h < HKV; ++h)
                BL(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,P,nt,256,&one,
                    k+size_t(h)*P*256,CUDA_R_16F,256,0,q16.p+(size_t(t0)*H+h*G)*256,CUDA_R_16F,H*256,256,
                    &zero,scores.p+size_t(h)*G*nt*ld,CUDA_R_32F,ld,(long long)nt*ld,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
            softmax<<<dim3(nt,H),256,0,st>>>(scores.p,probs.p,ld,nt,P,p0+t0);
            for (int h = 0; h < HKV; ++h)
                BL(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,256,nt,P,&one,
                    v+size_t(h)*P*256,CUDA_R_16F,256,0,probs.p+size_t(h)*G*nt*ld,CUDA_R_16F,ld,(long long)nt*ld,
                    &zero,out+(size_t(t0)*H+h*G)*256,CUDA_R_32F,H*256,256,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        }
        gate_mul<<<(size_t(T)*H*256+255)/256,256,0,st>>>(out,gate,size_t(T)*H*256);
    }
};
template<class F> float time_ms(F f, cudaStream_t st, int reps) {
    f(); CK(cudaStreamSynchronize(st));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a,st)); for (int r=0; r<reps; ++r) f(); CK(cudaEventRecord(b,st)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); return ms/reps;
}
int main(int argc, char** argv) {
    cudaStream_t st; CK(cudaStreamCreate(&st));
    int sms; CK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
    if (!strata::qwen35::flash_available()) return 2;
    bool fail = false;
    auto test = [&](int P, int H, int HKV, FILE* file, int maxT) {
        Buffer<float> q(size_t(maxT)*H*256), gate(size_t(maxT)*H*256), ref(size_t(maxT)*H*256), out(size_t(maxT)*H*256);
        Buffer<__half> k(size_t(P)*HKV*256), v(size_t(P)*HKV*256);
        Buffer<unsigned char> work(strata::qwen35::flash_workspace_bytes(maxT,H,HKV,P,sms));
        auto fill = [&](auto* p, size_t n, uint32_t seed) { random_data<<<(n+255)/256,256,0,st>>>(p,n,seed); };
        if (file) {
            auto read = [&](void* dst, size_t bytes) {
                std::vector<unsigned char> host(bytes);
                if (std::fread(host.data(),1,bytes,file) != bytes) std::exit(3);
                CK(cudaMemcpyAsync(dst,host.data(),bytes,cudaMemcpyHostToDevice,st)); CK(cudaStreamSynchronize(st));
            };
            read(q.p,size_t(maxT)*H*256*4); read(gate.p,size_t(maxT)*H*256*4);
            read(k.p,size_t(P)*HKV*256*2); read(v.p,size_t(P)*HKV*256*2);
        } else {
            fill(q.p,size_t(maxT)*H*256,1234); fill(gate.p,size_t(maxT)*H*256,5678);
            fill(k.p,size_t(P)*HKV*256,9911); fill(v.p,size_t(P)*HKV*256,1357);
        }
        Reference reference(maxT*H,st);
        for (int T : {9,100,512}) {
            if (T > maxT || T > P) continue;
            const float* qp = q.p + size_t(maxT-T)*H*256;
            const float* gp = gate.p + size_t(maxT-T)*H*256;
            const float old_ms = time_ms([&] { reference.run(qp,k.p,v.p,gp,ref.p,T,H,HKV,P); },st,2);
            const float new_ms = time_ms([&] { strata::qwen35::flash_prefill(qp,k.p,v.p,gp,out.p,work.p,T,H,HKV,P,P-T,sms,st); },st,5);
            const size_t n = size_t(T)*H*256;
            std::vector<float> a(n), b(n);
            CK(cudaMemcpy(a.data(),ref.p,n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(),out.p,n*4,cudaMemcpyDeviceToHost));
            double abs_sum=0, ref_sum=0, err2=0, ref2=0, elem=0, maxerr=0;
            for (size_t i=0; i<n; ++i) ref2 += double(a[i])*a[i];
            const double floor = std::sqrt(ref2/n)*1.e-3;
            for (size_t i=0; i<n; ++i) {
                const double e=std::abs(double(a[i])-b[i]);
                maxerr=std::max(maxerr,e); abs_sum+=e; ref_sum+=std::abs(a[i]); err2+=e*e;
                elem+=e/std::max(std::abs(double(a[i])),floor);
                if (!std::isfinite(b[i])) fail=true;
            }
            const double rel=abs_sum/ref_sum, rms=std::sqrt(err2/ref2);
            // Rounded normalized FP16 probabilities at 256K have subnormal quantization noise.
            if (rel > .015 || rms > .015) fail=true;
            std::printf("{\"data\":\"%s\",\"P\":%d,\"T\":%d,\"G\":%d,\"splits\":%d,\"max_abs\":%.9g,\"rel_l1\":%.9g,\"rel_rms\":%.9g,\"mean_element_rel\":%.9g,\"old_ms\":%.6f,\"fa_ms\":%.6f}\n",
                file?"real":"random",P,T,H/HKV,strata::qwen35::flash_splits(T,H,HKV,P,sms),maxerr,rel,rms,elem/n,old_ms,new_ms);
            std::fflush(stdout);
        }
    };
    if (argc > 1) {
        FILE* f=std::fopen(argv[1],"rb"); if (!f) return 3;
        int header[4]; if (std::fread(header,sizeof(header),1,f)!=1) return 3;
        test(header[3],header[1],header[2],f,header[0]); std::fclose(f);
    } else {
        for (int P : {512,4096,47000,131072,262144}) test(P,16,2,nullptr,512);
        for (int G : {1,2,4,16}) test(512,2*G,2,nullptr,512);
    }
    CK(cudaStreamDestroy(st)); return fail?1:0;
}
