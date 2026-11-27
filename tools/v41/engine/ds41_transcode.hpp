// ds41_transcode.hpp - PTQ1_0 (ternary, type 143) -> Q2_0 (type 42) on the device, value for value (see the .cu).
#pragma once

#include <cstddef>
#include <cstdint>

/// Bytes of the Q2_0 copy of `ptq1_bytes` of PTQ1_0 blocks (28 B / 128 weights -> 2 x 18 B).
size_t ds41_q2_0_bytes_from_ptq1(size_t ptq1_bytes);
/// Transcodes `ptq1_bytes` of PTQ1_0 blocks at `src` (device) into Q2_0 blocks at `dst` (device) on `stream`
/// (a cudaStream_t).  Whole matrices or rows: block b becomes Q2_0 blocks 2b and 2b+1.
void ds41_ptq1_to_q2_0(const void* src, void* dst, size_t ptq1_bytes, void* stream);
/// The same for `count` matrices of `ptq1_bytes` each: matrix c from src + c * src_stride to dst + c * dst_stride
/// (bytes) - one launch for a run of staged experts.
void ds41_ptq1_to_q2_0_strided(const void* src, size_t src_stride, void* dst, size_t dst_stride, size_t ptq1_bytes,
                               int count, void* stream);
/// dst[i] += src[i] for i < n (device floats) on `stream` - the remote tier's rows added to an out_dev call.
void ds41_add_f32(float* dst, const float* src, size_t n, void* stream);
/// dst[rows[i] * H + c] += src[i * H + c] for i < m, c < H (device; rows distinct) - the remote tier's compacted rows
/// (only the tokens that routed to the remote card) added to an out_dev call.
void ds41_add_rows_f32(float* dst, const float* src, const int32_t* rows, size_t m, size_t H, void* stream);
/// out[u * H + c] = sum_{j in [off[u], off[u+1])} w[j] * rows[pos[j] * H + c] for u < m (device; j order, no atomics) -
/// the remote tier's per-entry rows (in its sorted launch order) summed into its tokens' rows.
void ds41_gather_wsum_f32(const float* rows, const int32_t* off, const int32_t* pos, const float* w, size_t m, size_t H,
                          float* out, void* stream);
/// ds41_gather_wsum_f32 storing bf16 sums (round to nearest even) - STRATA_REMOTE_BF16_OUT.
void ds41_gather_wsum_bf16(const float* rows, const int32_t* off, const int32_t* pos, const float* w, size_t m, size_t H,
                           uint16_t* out, void* stream);
/// dst[i] = float(src[i]) for bf16 src (device) - STRATA_REMOTE_BF16_IN.
void ds41_bf16_to_f32(const uint16_t* src, float* dst, size_t n, void* stream);
