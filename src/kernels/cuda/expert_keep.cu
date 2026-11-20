// src/kernels/cuda/expert_keep.cu - STRATA_EXPERT_KEEP (see include/strata/kernels/expert_keep.hpp).
#include "strata/kernels/expert_keep.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#define strtok_r strtok_s   // MSVC has no POSIX strtok_r; same alias as src/dsv41/ds4_model.cu and ds4_packinfo.cpp
#endif

namespace strata::kernels {
namespace {

uint8_t* g_keep = nullptr;           // [n_layers][n_expert], 1 = kept; managed memory so any device can read it
int64_t g_layers = 0, g_ne = 0;
std::vector<char> g_layer_on;        // host copy: is this layer masked at all (no launch for unmasked layers)
std::vector<uint8_t> g_keep_host;    // host copy of the table (expert_keep_kept), filled by expert_keep_load
int64_t g_host_layers = 0, g_host_ne = 0;
bool g_loaded = false, g_load_ok = true;
int g_masked = 0;
int64_t g_kept_total = 0;

__global__ void expert_keep_kernel(float* __restrict__ logits, const uint8_t* __restrict__ keep, int64_t n_tok,
                                   int64_t ne) {
    const int64_t t = blockIdx.x;
    if (t >= n_tok) return;
    for (int64_t e = threadIdx.x; e < ne; e += blockDim.x)
        if (!keep[e]) logits[t * ne + e] = -1e30f;
}

}  // namespace

bool expert_keep_load(int64_t n_layers, int64_t n_expert) {
    if (g_loaded) return g_load_ok;
    g_loaded = true;
    g_load_ok = false;
    const char* path = std::getenv("STRATA_EXPERT_KEEP");
    if (path == nullptr || *path == '\0') return g_load_ok = true;
    if (n_layers <= 0 || n_expert <= 0) {
        std::fprintf(stderr, "STRATA_EXPERT_KEEP: this model has no MoE layers; ignored\n");
        return g_load_ok = true;
    }
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        std::fprintf(stderr, "STRATA_EXPERT_KEEP: cannot open %s\n", path);
        return false;
    }
    std::vector<uint8_t> keep((size_t) (n_layers * n_expert), 1);
    std::vector<char> on((size_t) n_layers, 0);
    char line[65536];
    int masked = 0;
    int64_t kept_total = 0;
    bool ok = true;
    while (ok && std::fgets(line, sizeof line, f) != nullptr) {
        if (char* hash = std::strchr(line, '#')) *hash = '\0';
        char* colon = std::strchr(line, ':');
        if (colon == nullptr) continue;
        *colon = '\0';
        const long l = std::strtol(line, nullptr, 10);
        if (l < 0 || l >= n_layers) {
            std::fprintf(stderr, "STRATA_EXPERT_KEEP: layer %ld is outside 0..%lld\n", l, (long long) n_layers - 1);
            ok = false;
            break;
        }
        uint8_t* row = keep.data() + l * n_expert;
        std::memset(row, 0, (size_t) n_expert);
        int kept = 0;
        char* save = nullptr;
        for (char* tok = strtok_r(colon + 1, " \t\r\n,", &save); tok; tok = strtok_r(nullptr, " \t\r\n,", &save)) {
            const long e = std::strtol(tok, nullptr, 10);
            if (e < 0 || e >= n_expert) {
                std::fprintf(stderr, "STRATA_EXPERT_KEEP: layer %ld: expert %ld is outside 0..%lld\n", l, e,
                             (long long) n_expert - 1);
                ok = false;
                break;
            }
            if (!row[e]) { row[e] = 1; ++kept; }
        }
        if (ok && kept < 10) {   // top-10 routing needs at least 10 experts left
            std::fprintf(stderr, "STRATA_EXPERT_KEEP: layer %ld keeps %d experts (need >= 10)\n", l, kept);
            ok = false;
        }
        if (ok && !on[(size_t) l]) { on[(size_t) l] = 1; ++masked; }
        kept_total += kept;
    }
    std::fclose(f);
    if (!ok) return false;
    g_keep_host.swap(keep);
    g_layer_on = on;
    g_host_layers = n_layers;
    g_host_ne = n_expert;
    g_masked = masked;
    g_kept_total = kept_total;
    return g_load_ok = true;
}

bool expert_keep_kept(int64_t layer, int64_t expert) {
    if (g_keep_host.empty() || layer < 0 || layer >= g_host_layers || expert < 0 || expert >= g_host_ne) return true;
    return g_keep_host[(size_t) (layer * g_host_ne + expert)] != 0;
}

bool expert_keep_init(int64_t n_layers, int64_t n_expert) {
    static bool done = false;
    if (done) return true;
    done = true;
    if (!expert_keep_load(n_layers, n_expert)) return false;
    if (g_keep_host.empty()) return true;   // unset (or no MoE layers): nothing to upload
    if (n_layers != g_host_layers || n_expert != g_host_ne) {
        std::fprintf(stderr, "STRATA_EXPERT_KEEP: loaded for %lldx%lld but initialised for %lldx%lld\n",
                     (long long) g_host_layers, (long long) g_host_ne, (long long) n_layers, (long long) n_expert);
        return false;
    }
    const char* path = std::getenv("STRATA_EXPERT_KEEP");
    const std::vector<uint8_t>& keep = g_keep_host;
    const int masked = g_masked;
    const int64_t kept_total = g_kept_total;
    if (cudaMallocManaged(&g_keep, keep.size()) != cudaSuccess) {
        std::fprintf(stderr, "STRATA_EXPERT_KEEP: cudaMallocManaged failed\n");
        g_keep = nullptr;
        return false;
    }
    std::memcpy(g_keep, keep.data(), keep.size());
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    g_layers = n_layers;
    g_ne = n_expert;
    std::fprintf(stderr, "[strata] STRATA_EXPERT_KEEP %s: %d of %lld layers masked, %.1f of %lld experts kept per masked layer\n",
                 path, masked, (long long) n_layers, masked ? (double) kept_total / masked : 0.0, (long long) n_expert);
    return true;
}

bool expert_keep_on() { return g_keep != nullptr; }

void expert_keep_apply(float* logits, int64_t n_tok, int64_t n_expert, int64_t layer, void* stream) {
    if (g_keep == nullptr || n_tok <= 0 || n_expert != g_ne || layer < 0 || layer >= g_layers ||
        !g_layer_on[(size_t) layer])
        return;
    expert_keep_kernel<<<(unsigned) n_tok, 256, 0, (cudaStream_t) stream>>>(logits, g_keep + layer * n_expert, n_tok,
                                                                             n_expert);
}

}  // namespace strata::kernels
