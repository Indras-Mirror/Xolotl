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
                                           int32_t* __restrict__ n_groups, int32_t* armed) {
    if (threadIdx.x != 0) return;
    bool all = true;
    if (armed) {   // STRATA_HIT_FAST=3: one launch per arming
        all = *(volatile int32_t*) armed != 0;
        *(volatile int32_t*) armed = 0;
    }
    for (int k = 0; k < K && (all || !armed); ++k) {
        const int32_t e = ids[k];
        const unsigned long long p = (e >= 0 && e < n_experts) ? row[e] : 0ull;
        grp_ptr[k] = p;
        all = all && p != 0ull;
    }
    *n_groups = all ? K : 0;
}

// STRATA_HIT_PAR: the same decision with lane k < K looking up routing k.  The row table and armed live in mapped pinned
// memory, so the loop above waits one bus round trip per read (K + 1 in a row); here they are all in flight at once.
// grp_ptr is written only when all K are resident (with *n_groups = 0 nothing reads it).
__global__ void decode_hit_dispatch_par_kernel(const int32_t* __restrict__ ids, const unsigned long long* __restrict__ row,
                                               int K, int n_experts, unsigned long long* __restrict__ grp_ptr,
                                               int32_t* __restrict__ n_groups, int32_t* armed) {
    const int lane = threadIdx.x;
    unsigned long long p = 0ull;
    if (lane < K) {
        const int32_t e = ids[lane];
        p = (e >= 0 && e < n_experts) ? row[e] : 0ull;
    }
    int32_t go = 1;
    if (armed && lane == 0) {
        go = *(volatile int32_t*) armed;
        *(volatile int32_t*) armed = 0;
    }
    go = __shfl_sync(0xFFFFFFFFu, go, 0);
    const bool all = go != 0 && __all_sync(0xFFFFFFFFu, lane >= K || p != 0ull);
    if (all && lane < K) grp_ptr[lane] = p;
    if (lane == 0) *n_groups = all ? K : 0;
}

// STRATA_HIT_PF: decode_hit_dispatch_par_kernel plus the layer's prefetched experts (HitPfList).  The list is read
// only when a lane missed, its stamp first and its entries after (the host writes the stamp last), one entry per lane;
// the lanes then match their expert against the entries by shuffles.  A prefetched expert's bytes are home when the
// copy queued behind the prefetch's DMAs has written the stamp into *ready.
__global__ void decode_hit_dispatch_pf_kernel(const int32_t* __restrict__ ids, const unsigned long long* __restrict__ row,
                                              int K, int n_experts, unsigned long long* __restrict__ grp_ptr,
                                              int32_t* __restrict__ n_groups, int32_t* armed, const HitPfList* pf,
                                              const int32_t* ready, int32_t* dec, long long wait_ns,
                                              int32_t* __restrict__ ent_dst, int32_t* __restrict__ sum_gate, int part,
                                              int32_t* __restrict__ stamp_dev, int32_t* err) {
    const unsigned full = 0xFFFFFFFFu;
    const int lane = threadIdx.x;
    int32_t e = -1;
    unsigned long long p = 0ull;
    if (lane < K) {
        e = ids[lane];
        p = (e >= 0 && e < n_experts) ? row[e] : 0ull;
    }
    int32_t s = 0;
    if (lane == 0) {
        s = *(volatile int32_t*) armed;
        *(volatile int32_t*) armed = 0;
    }
    s = __shfl_sync(full, s, 0);
    if (lane == 0) *stamp_dev = s;
    if (s == 0) {
        if (lane == 0) {
            *n_groups = 0;
            *sum_gate = 0;
        }
        return;
    }
    bool from_pf = false;
    if (__any_sync(full, lane < K && p == 0ull)) {
        int32_t ls = 0, ln = 0;
        if (lane == 0) {
            ls = *(volatile const int32_t*) &pf->stamp;
            ln = *(volatile const int32_t*) &pf->n;
        }
        ls = __shfl_sync(full, ls, 0);
        ln = __shfl_sync(full, ln, 0);
        if (ls == s) {
            int32_t le = -1;
            unsigned long long lp = 0ull;
            if (lane < kHitPfMax && lane < ln) {
                le = *(volatile const int32_t*) &pf->e[lane];
                lp = *(volatile const unsigned long long*) &pf->ptr[lane];
            }
            for (int j = 0; j < ln && j < kHitPfMax; ++j) {
                const int32_t ej = __shfl_sync(full, le, j);
                const unsigned long long pj = __shfl_sync(full, lp, j);
                if (lane < K && p == 0ull && ej >= 0 && e == ej) {
                    p = pj;
                    from_pf = true;
                }
            }
        }
    }
    const bool found = lane < K && p != 0ull;
    const bool all = __all_sync(full, lane >= K || found);
    unsigned m = 0;
    int ng = 0;
    if (all) {
        if (lane < K) {
            grp_ptr[lane] = p;
            ent_dst[lane] = lane;
        }
        ng = K;
        m = (1u << K) - 1u;
    } else if (part) {
        m = __ballot_sync(full, found);
        const int at = __popc(m & ((1u << lane) - 1u));
        if (found) {
            grp_ptr[at] = p;
            ent_dst[at] = lane;
        }
        ng = __popc(m);
    }
    if (lane == 0) {
        *n_groups = ng;
        *sum_gate = all ? K : 0;
        *(volatile int32_t*) &dec[1] = all ? K : 0;
        *(volatile int32_t*) &dec[2] = (int32_t) m;
        __threadfence_system();
        *(volatile int32_t*) &dec[0] = s;
    }
    // a prefetched expert taken: its bytes before the grouped launch behind this one reads them
    if (__any_sync(full, ((m >> lane) & 1u) && from_pf) && lane == 0) {
        unsigned long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        while (*(volatile const int32_t*) ready != s) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if ((long long) (t - t0) > wait_ns) {
                *(volatile int32_t*) err = s;
                __threadfence_system();
                break;
            }
        }
    }
}

void decode_hit_dispatch_pf(const int32_t* ids, const unsigned long long* row, int K, int n_experts,
                            unsigned long long* grp_ptr, int32_t* n_groups, void* stream, int32_t* armed,
                            const HitPfList* pf, const int32_t* ready, int32_t* dec, long long wait_ns,
                            int32_t* ent_dst, int32_t* sum_gate, bool part, int32_t* stamp_dev, int32_t* err) {
    decode_hit_dispatch_pf_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, row, K, n_experts, grp_ptr, n_groups, armed,
                                                                       pf, ready, dec, wait_ns, ent_dst, sum_gate,
                                                                       part ? 1 : 0, stamp_dev, err);
    check("decode_hit_dispatch_pf");
}

void decode_hit_dispatch(const int32_t* ids, const unsigned long long* row, int K, int n_experts,
                         unsigned long long* grp_ptr, int32_t* n_groups, void* stream, int32_t* armed, bool par) {
    if (par && K <= 32)
        decode_hit_dispatch_par_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, row, K, n_experts, grp_ptr, n_groups, armed);
    else
        decode_hit_dispatch_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, row, K, n_experts, grp_ptr, n_groups, armed);
    check("decode_hit_dispatch");
}

// decode_rows_sum_sel_kernel with no pool rows and the weights in device memory, run only when *gate != 0: the same
// values summed in the same order with the same double operations, so the same bits.
__global__ void decode_rows_sum_gated_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                             const int32_t* __restrict__ gate, int K, int64_t H, float* __restrict__ out,
                                             const int32_t* stamp, int32_t* done) {
    if (done && blockIdx.x == 0 && threadIdx.x == 0) {
        *(volatile int32_t*) done = *(volatile const int32_t*) stamp;
        __threadfence_system();
    }
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
                           void* stream, const int32_t* stamp, int32_t* done) {
    const unsigned blocks = (unsigned) std::min<int64_t>((H + 255) / 256, 1024);
    decode_rows_sum_gated_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(parts, w, gate, K, H, out, stamp, done);
    check("decode_rows_sum_gated");
}

// STRATA_RB_HOOK (ds41 decode): the words go to the host first; every thread's writes are fenced at system scope before
// the barrier, and thread 0 fences again before the flag, so the host sees the flag only after the words.
__global__ void push_to_host_kernel(const HostPush p) {
    for (int s = 0; s < p.n; ++s) {
        const uint32_t* __restrict__ a = (const uint32_t*) p.src[s];
        uint32_t* __restrict__ b = (uint32_t*) p.dst[s];
        for (int i = threadIdx.x; i < p.words[s]; i += blockDim.x) b[i] = a[i];
    }
    if (!p.flag) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile int32_t*) p.flag = 1;
    }
}

void push_to_host(const HostPush& p, void* stream) {
    if (p.n < 0 || p.n > kHostPushMax) return;
    push_to_host_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(p);
    check("push_to_host");
}

void* host_dev_ptr(void* host) {
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return d;
}
