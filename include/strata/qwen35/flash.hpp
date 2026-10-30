#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace strata::qwen35 {
// CUDA SM80+, HD=256, G in {1,2,4,8,16}. Queries/gates/output are [T,H,256],
// caches are FP16 [HKV,ctx,256]. Workspace belongs to the caller and is stream ordered.
bool flash_available();
int flash_splits(int T, int H, int HKV, int P, int sms);
size_t flash_workspace_bytes(int max_T, int H, int HKV, int ctx, int sms);
void flash_prefill(const float* q, const __half* k, const __half* v, const float* gate,
                   float* out, void* workspace, int T, int H, int HKV, int ctx, int p0,
                   int sms, cudaStream_t stream);
}  // namespace strata::qwen35
