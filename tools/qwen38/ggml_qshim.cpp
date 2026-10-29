// ggml_qshim.cpp - a tiny C ABI over ggml's own quantize/dequantize, for the XQ screener.
//
// The screener has to answer one question per (expert, candidate tier): how much does this
// expert's output move if I store it as that ggml type?  Doing that needs ggml's real quantisers,
// because the IQ types are lattice/codebook formats whose error depends on the imatrix weights -
// reimplementing them in numpy would silently measure a different format.
//
// No new kernels are introduced here.  Everything below is a thin wrapper over two existing,
// exported ggml entry points:
//
//   ggml_quantize_chunk()  - handles every ggml type, including the importance-weighted IQ ones
//   ggml_get_type_traits()->to_float - the generic dequantiser for any type
//
// The wrapper exists only because those take `enum ggml_type` and raw pointers, which ctypes can
// drive directly once the enum values are exported as functions (so the Python side never hardcodes
// ggml's numbering, which moves between revisions).
//
// Build (needs libggml from a CPU-only llama.cpp build):
//   g++ -O2 -fPIC -shared tools/qwen38/ggml_qshim.cpp \
//       -I<llama.cpp>/ggml/include -L<llama.cpp>/build-cpu-screen/bin -lggml -lggml-base \
//       -Wl,-rpath,<llama.cpp>/build-cpu-screen/bin -o /tmp/libgqshim.so
//
// All functions return 0 on success and a negative value on a type/shape error, so the Python side
// can fail loudly instead of reading uninitialised memory.

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <cstdio>

extern "C" {

/// Bytes one row of `n` weights occupies in `type`. 0 if the type is unknown or the shape is bad.
int64_t gq_row_bytes(int type, int64_t n) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    if (!t || t->blck_size <= 0) return 0;
    if (n % t->blck_size != 0) return -1;
    return (n / t->blck_size) * (int64_t) t->type_size;
}

/// Quantize `nrows` rows of `n_per_row` floats.  `imatrix` may be null (RTN); the IQ types need it.
/// Returns the number of bytes written, or -1.
int64_t gq_quantize(int type, const float * src, void * dst,
                    int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    if (!t || t->blck_size <= 0) return -1;
    if (n_per_row % t->blck_size != 0) return -1;
    return (int64_t) ggml_quantize_chunk((enum ggml_type) type, src, dst, 0, nrows, n_per_row, imatrix);
}

/// Dequantize `n` weights of `type` back to f32.  Returns 0 on success.
int gq_dequantize(int type, const void * src, float * dst, int64_t n) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    if (!t || t->to_float == nullptr) return -1;
    if (t->blck_size <= 0 || n % t->blck_size != 0) return -1;
    t->to_float(src, dst, n);
    return 0;
}

int      gq_type_blck_size(int type) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    return (t && t->blck_size > 0) ? (int) t->blck_size : -1;
}
const char * gq_type_name(int type) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    return t ? t->type_name : nullptr;
}
int gq_type_is_quantized(int type) {
    const ggml_type_traits * t = ggml_get_type_traits((enum ggml_type) type);
    return t ? (int) t->is_quantized : -1;
}

// the enum values this build actually has, so Python never hardcodes them
int gq_id(const char * name) {
    static const struct { const char * n; int v; } m[] = {
        {"f32", GGML_TYPE_F32}, {"f16", GGML_TYPE_F16}, {"bf16", GGML_TYPE_BF16},
        {"q2_0", GGML_TYPE_Q2_0},
        {"iq2_xxs", GGML_TYPE_IQ2_XXS}, {"iq2_xs", GGML_TYPE_IQ2_XS}, {"iq2_s", GGML_TYPE_IQ2_S},
        {"iq3_xxs", GGML_TYPE_IQ3_XXS}, {"iq3_s", GGML_TYPE_IQ3_S}, {"iq4_nl", GGML_TYPE_IQ4_NL},
        // in the published GSQ-RCO allocations (IQ4_XS on 181 tensors, IQ1_M on gate/up), so screenable here
        {"iq4_xs", GGML_TYPE_IQ4_XS}, {"iq1_s", GGML_TYPE_IQ1_S}, {"iq1_m", GGML_TYPE_IQ1_M},
        // 4-bit float formats.  Both have 64-only or 32-only blocks that DO divide qwen4exp's
        // n_ff = 640, so unlike every IQ2/IQ3/K-quant they are usable on the down projection:
        // NVFP4 = 64 values per block (4 E4M3 sub-scales), MXFP4 = 32 (one E8M0 scale).
        {"nvfp4", GGML_TYPE_NVFP4}, {"mxfp4", GGML_TYPE_MXFP4},
        {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K},
        {"q6_k", GGML_TYPE_Q6_K}, {"q8_0", GGML_TYPE_Q8_0},
    };
    for (const auto & e : m) {
        if (std::strcmp(e.n, name) == 0) return e.v;
    }
    return -1;
}

} // extern "C"
