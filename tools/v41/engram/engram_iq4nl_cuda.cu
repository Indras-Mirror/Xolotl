// tools/v41/engram/engram_iq4nl_cuda.cu - build: nvcc -O3 -fmad=false -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 engram_iq4nl_cuda.cu -o engram_iq4nl_cuda
// Verified byte-identical to ggml's CPU quantize_iq4_nl on 2,097,152 rows of layer 14; a full 384M-row layer takes ~6-8 min (disk-bound).
// Run one layer per GPU in parallel: engram_iq4nl_cuda engram-L14.safetensors 14 L14.gguf 0 & engram_iq4nl_cuda engram-L1.safetensors 1 L1.gguf 1
// then serve with --engram L1.gguf:L14.gguf (and optionally ds41_engram_hot pack --dir L1.gguf:L14.gguf for an IQ4 hot set).
// engram_iq4nl_cuda: DS4.1 Engram table (safetensors F8_E4M3 [rows,256] + F8_E8M0 [rows,8]) -> GGUF, one IQ4_NL tensor
// blk.<L>.engram_embd.weight [256, rows].  ggml's quantize_iq4_nl (quantize_row_iq4_nl_impl, super_block = block = 32,
// ntry 7, weights x^2, no imatrix) on the GPU: one thread per 32-value block.  Built with -fmad=false so the float
// arithmetic is ggml's operation for operation.
// usage: engram_iq4nl_cuda <in.safetensors> <layer> <out.gguf> [device=0] [max_rows=all] [--check N]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
__constant__ int8_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__constant__ float LUT8[256];
__constant__ float SC[256];
__device__ __forceinline__ int best_index(float x) {
    if (x <= KV[0]) return 0;
    if (x >= KV[15]) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) { int mav = (ml + mu) / 2; if (x < KV[mav]) mu = mav; else ml = mav; }
    return x - KV[mu - 1] < KV[mu] - x ? mu - 1 : mu;
}
__global__ void quant(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, uint8_t* __restrict__ out, int64_t nblk) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    float x[32], wt[32];
    const float sc = SC[s[b]];
    const uint8_t* wb = w + b * 32;
    for (int j = 0; j < 32; ++j) { float v = LUT8[wb[j]] * sc; if (!isfinite(v)) v = 0.f; x[j] = v; wt[j] = v * v; }
    float amax = 0, max = 0;
    for (int j = 0; j < 32; ++j) { float ax = fabsf(x[j]); if (ax > amax) { amax = ax; max = x[j]; } }
    uint8_t* o = out + b * 18;
    uint8_t L[32];
    float d = 0.f;
    if (amax < 1e-15f) {
        d = 0.f;
        for (int j = 0; j < 32; ++j) L[j] = 0;   // ggml: scales 0 -> dh = 0, L from id = 0 -> best_index(0)
        for (int j = 0; j < 32; ++j) L[j] = (uint8_t) best_index(0.f);
    } else {
        d = -max / KV[0];
        float id = 1 / d, sumqx = 0, sumq2 = 0;
        for (int j = 0; j < 32; ++j) { int l = best_index(id * x[j]); float q = KV[l]; float ww = wt[j]; sumqx += ww * q * x[j]; sumq2 += ww * q * q; }
        d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
        float best = d * sumqx;
        for (int itry = -7; itry <= 7; ++itry) {
            id = (itry + KV[0]) / max; sumqx = sumq2 = 0;
            for (int j = 0; j < 32; ++j) { int l = best_index(id * x[j]); float q = KV[l]; float ww = wt[j]; sumqx += ww * q * x[j]; sumq2 += ww * q * q; }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) { d = sumqx / sumq2; best = d * sumqx; }
        }
        const float idd = d ? 1 / d : 0;
        for (int j = 0; j < 32; ++j) L[j] = (uint8_t) best_index(idd * x[j]);
    }
    const __half h = __float2half_rn(d);
    memcpy(o, &h, 2);
    for (int j = 0; j < 16; ++j) o[2 + j] = L[j] | (L[16 + j] << 4);
}
static float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}
static bool st_find(int fd, const std::string& name, uint64_t& off, uint64_t& bytes) {
    uint64_t n; if (pread(fd, &n, 8, 0) != 8) return false;
    std::string h(n, 0); if (pread(fd, &h[0], n, 8) != (ssize_t) n) return false;
    size_t p = h.find("\"" + name + "\""); if (p == std::string::npos) return false;
    p = h.find("data_offsets", p); p = h.find('[', p);
    unsigned long long a, b; if (sscanf(h.c_str() + p, "[%llu,%llu]", &a, &b) != 2 && sscanf(h.c_str() + p, "[%llu, %llu]", &a, &b) != 2) return false;
    off = 8 + n + a; bytes = b - a; return true;
}
static void put(std::vector<uint8_t>& v, const void* p, size_t n) { const uint8_t* c = (const uint8_t*) p; v.insert(v.end(), c, c + n); }
static void put_str(std::vector<uint8_t>& v, const std::string& s) { uint64_t n = s.size(); put(v, &n, 8); put(v, s.data(), n); }
static bool pread_all(int fd, void* buf, size_t n, uint64_t off) { uint8_t* p = (uint8_t*) buf; while (n) { ssize_t r = pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; } return true; }
static bool pwrite_all(int fd, const void* buf, size_t n, uint64_t off) { const uint8_t* p = (const uint8_t*) buf; while (n) { ssize_t r = pwrite(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; } return true; }
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s in.safetensors layer out.gguf [device] [max_rows]\n", argv[0]); return 2; }
    const std::string in = argv[1], L = argv[2], out = argv[3];
    const int dev = argc > 4 ? atoi(argv[4]) : 0;
    CK(cudaSetDevice(dev));
    float lut[256], sct[256]; for (int i = 0; i < 256; ++i) { lut[i] = e4m3((uint8_t) i); sct[i] = std::ldexp(1.0f, i - 127); }
    CK(cudaMemcpyToSymbol(LUT8, lut, sizeof lut)); CK(cudaMemcpyToSymbol(SC, sct, sizeof sct));
    const int D = 256, RB = 144;
    int fi = open(in.c_str(), O_RDONLY); if (fi < 0) { perror(in.c_str()); return 1; }
    uint64_t woff, wb, soff, sb;
    if (!st_find(fi, "layers." + L + ".engram.embed.weight", woff, wb) || !st_find(fi, "layers." + L + ".engram.embed.scale", soff, sb)) { fprintf(stderr, "tensors not found\n"); return 1; }
    int64_t rows = (int64_t) (wb / D);
    if (sb != (uint64_t) rows * 8) { fprintf(stderr, "scale size mismatch\n"); return 1; }
    if (argc > 5 && atoll(argv[5]) > 0) rows = std::min<int64_t>(rows, atoll(argv[5]));
    std::vector<uint8_t> h; uint32_t magic = 0x46554747, ver = 3; uint64_t nt = 1, nkv = 1;
    put(h, &magic, 4); put(h, &ver, 4); put(h, &nt, 8); put(h, &nkv, 8);
    put_str(h, "general.name"); uint32_t ty_str = 8; put(h, &ty_str, 4); put_str(h, "DeepSeek-V4.1-Flash-1017 Engram layer " + L + " (IQ4_NL)");
    put_str(h, "blk." + L + ".engram_embd.weight"); uint32_t nd = 2; put(h, &nd, 4); uint64_t d0 = D, d1 = (uint64_t) rows; put(h, &d0, 8); put(h, &d1, 8);
    uint32_t ty = 20; put(h, &ty, 4); uint64_t toff = 0; put(h, &toff, 8);
    while (h.size() % 32) h.push_back(0);
    const uint64_t data0 = h.size();
    int fo = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fo < 0) { perror(out.c_str()); return 1; }
    if (!pwrite_all(fo, h.data(), h.size(), 0)) { perror("write"); return 1; }
    const int64_t CH = 1 << 22;   // 4M rows: 1 GiB in, 576 MiB out
    uint8_t *hw, *hs, *ho, *dw, *ds, *dout;
    CK(cudaMallocHost(&hw, (size_t) CH * D)); CK(cudaMallocHost(&hs, (size_t) CH * 8)); CK(cudaMallocHost(&ho, (size_t) CH * RB));
    CK(cudaMalloc(&dw, (size_t) CH * D)); CK(cudaMalloc(&ds, (size_t) CH * 8)); CK(cudaMalloc(&dout, (size_t) CH * RB));
    auto t0 = std::chrono::steady_clock::now(); double tio = 0, tgpu = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += CH) {
        const int64_t n = std::min(CH, rows - r0);
        auto a = std::chrono::steady_clock::now();
        if (!pread_all(fi, hw, (size_t) n * D, woff + (uint64_t) r0 * D) || !pread_all(fi, hs, (size_t) n * 8, soff + (uint64_t) r0 * 8)) { perror("read"); return 1; }
        posix_fadvise(fi, woff + (uint64_t) r0 * D, (off_t) n * D, POSIX_FADV_DONTNEED);
        posix_fadvise(fi, soff + (uint64_t) r0 * 8, (off_t) n * 8, POSIX_FADV_DONTNEED);
        auto b = std::chrono::steady_clock::now();
        CK(cudaMemcpy(dw, hw, (size_t) n * D, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, hs, (size_t) n * 8, cudaMemcpyHostToDevice));
        const int64_t nblk = n * 8;
        quant<<<(unsigned) ((nblk + 255) / 256), 256>>>(dw, ds, dout, nblk);
        CK(cudaGetLastError());
        CK(cudaMemcpy(ho, dout, (size_t) n * RB, cudaMemcpyDeviceToHost));
        auto c = std::chrono::steady_clock::now();
        if (!pwrite_all(fo, ho, (size_t) n * RB, data0 + (uint64_t) r0 * RB)) { perror("write"); return 1; }
        fdatasync(fo); posix_fadvise(fo, 0, 0, POSIX_FADV_DONTNEED);
        auto e = std::chrono::steady_clock::now();
        tio += std::chrono::duration<double>(b - a).count() + std::chrono::duration<double>(e - c).count();
        tgpu += std::chrono::duration<double>(c - b).count();
        const double s = std::chrono::duration<double>(e - t0).count();
        if ((r0 / CH) % 10 == 0 || r0 + n >= rows)
            fprintf(stderr, "L%s dev%d: %lld / %lld rows, %.0f s (io %.0f, gpu %.0f), %.2f M rows/s\n", L.c_str(), dev, (long long) (r0 + n), (long long) rows, s, tio, tgpu, (r0 + n) / s / 1e6);
    }
    close(fo); close(fi);
    printf("done %s: %lld rows\n", out.c_str(), (long long) rows);
}
