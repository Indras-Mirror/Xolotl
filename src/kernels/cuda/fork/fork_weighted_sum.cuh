// src/kernels/cuda/fork/fork_weighted_sum.cuh - weighted_rows_sum: the prompt chunk's top-k mix on the card (MiMo prefill).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// out[t][i] = sum_k w[t*K + k] * parts[(t*K + k)][i] - the prompt chunk's top-k mix on the card (MiMo prefill), so
// only n rows come back instead of n*K.  Summed in k order, like the host loop it replaces.
__global__ void weighted_rows_sum_kernel(const float* __restrict__ parts, const float* __restrict__ w, int K,
                                         int64_t H, int64_t n, float* __restrict__ out) {
    const int64_t total = n * H;
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = q / H, i = q - t * H;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += w[t * K + k] * parts[(t * K + k) * H + i];
        out[q] = acc;
    }
}

void weighted_rows_sum(const float* parts, const float* w, int K, int64_t H, int64_t n, float* out, void* stream) {
    const int64_t total = n * H;
    const unsigned blocks = (unsigned) std::min<int64_t>((total + 255) / 256, 65535);
    weighted_rows_sum_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, w, K, H, n, out);
    check("weighted_rows_sum");
}

// Decode (one token): out[i] = sum over k of w[k] * parts[k][i], k in routing order, accumulated in DOUBLE and rounded
// once - the host loop it replaces (ds41_moe gpu_run) bit for bit: a float x float product is exact in double, so
// fused or not the sum is the same; w[k] == 0 (a dropped miss: its row was never computed) is skipped, as there.
__global__ void decode_rows_sum_kernel(const float* __restrict__ parts, const float* __restrict__ w, int K, int64_t H,
                                       float* __restrict__ out) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < H; i += (int64_t) gridDim.x * blockDim.x) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            const float wk = w[k];
            if (wk == 0.0f) continue;
            s = __dadd_rn(s, __dmul_rn((double) wk, (double) parts[(int64_t) k * H + i]));
        }
        out[i] = (float) s;
    }
}

void decode_rows_sum(const float* parts, const float* w, int K, int64_t H, float* out, void* stream) {
    const unsigned blocks = (unsigned) std::min<int64_t>((H + 255) / 256, 1024);
    decode_rows_sum_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, w, K, H, out);
    check("decode_rows_sum");
}

// decode_rows_sum_kernel with row k taken from host_rows (pinned host memory, read over PCIe) when sel[k] != 0: the same
// values summed in the same order with the same double operations, so the same bits; w and sel are launch parameters.
__global__ void decode_rows_sum_sel_kernel(const float* __restrict__ parts, const float* __restrict__ host_rows,
                                           const RowsSumSel ws, int K, int64_t H, float* __restrict__ out) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < H; i += (int64_t) gridDim.x * blockDim.x) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            const float wk = ws.w[k];
            if (wk == 0.0f) continue;
            const float v = ws.sel[k] ? host_rows[(int64_t) k * H + i] : parts[(int64_t) k * H + i];
            s = __dadd_rn(s, __dmul_rn((double) wk, (double) v));
        }
        out[i] = (float) s;
    }
}

void decode_rows_sum_sel(const float* parts, const float* host_rows, const RowsSumSel& ws, int K, int64_t H, float* out,
                         void* stream) {
    const unsigned blocks = (unsigned) std::min<int64_t>((H + 255) / 256, 1024);
    decode_rows_sum_sel_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, host_rows, ws, K, H, out);
    check("decode_rows_sum_sel");
}

// STRATA_META_PULL (ds41 decode): a small host-to-device copy done by a kernel that reads mapped pinned memory, so it
// does not queue behind bulk DMAs on the copy engine.  4-byte words; dst / src 4-byte aligned.
__global__ void pull_from_host_kernel(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

void pull_from_host(void* dst, const void* src, size_t bytes, void* stream) {
    const int n = (int) (bytes / 4);
    if (n <= 0) return;
    const unsigned blocks = (unsigned) std::min((n + 255) / 256, 64);
    pull_from_host_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((uint32_t*) dst, (const uint32_t*) src, n);
    check("pull_from_host");
}

// STRATA_HIT_FAST (ds41 decode): one thread turns the device routing into the hits' group addresses.  row[e] is expert
// e's VRAM slot address (0: not resident); all K routed experts resident -> grp_ptr[k] = row[ids[k]] and *n_groups = K,
// else *n_groups = 0, and the grouped kernels and decode_rows_sum_gated that read it do nothing.
__global__ void decode_hit_dispatch_kernel(const int32_t* __restrict__ ids, const unsigned long long* __restrict__ row,
                                           int K, int n_experts, unsigned long long* __restrict__ grp_ptr,
                                           int32_t* __restrict__ n_groups) {
    if (threadIdx.x != 0) return;
    bool all = true;
    for (int k = 0; k < K; ++k) {
        const int32_t e = ids[k];
        const unsigned long long p = (e >= 0 && e < n_experts) ? row[e] : 0ull;
        grp_ptr[k] = p;
        all = all && p != 0ull;
    }
    *n_groups = all ? K : 0;
}

void decode_hit_dispatch(const int32_t* ids, const unsigned long long* row, int K, int n_experts,
                         unsigned long long* grp_ptr, int32_t* n_groups, void* stream) {
    decode_hit_dispatch_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, row, K, n_experts, grp_ptr, n_groups);
    check("decode_hit_dispatch");
}

// decode_rows_sum_sel_kernel with no pool rows and the weights in device memory, run only when *gate != 0: the same
// values summed in the same order with the same double operations, so the same bits.
__global__ void decode_rows_sum_gated_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                             const int32_t* __restrict__ gate, int K, int64_t H, float* __restrict__ out) {
    if (*gate == 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < H; i += (int64_t) gridDim.x * blockDim.x) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            const float wk = w[k];
            if (wk == 0.0f) continue;
            const float v = parts[(int64_t) k * H + i];
            s = __dadd_rn(s, __dmul_rn((double) wk, (double) v));
        }
        out[i] = (float) s;
    }
}

void decode_rows_sum_gated(const float* parts, const float* w, const int32_t* gate, int K, int64_t H, float* out,
                           void* stream) {
    const unsigned blocks = (unsigned) std::min<int64_t>((H + 255) / 256, 1024);
    decode_rows_sum_gated_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, w, gate, K, H, out);
    check("decode_rows_sum_gated");
}
