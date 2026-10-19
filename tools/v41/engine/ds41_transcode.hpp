// ds41_transcode.hpp - PTQ1_0 (ternary, type 143) -> Q2_0 (type 42) on the device, value for value (see the .cu).
#pragma once

#include <cstddef>

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
