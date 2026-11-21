// Fused decode ops for xolotl-inkling: GGML_OP_CUSTOM nodes with a CUDA implementation (ggml_cuda_custom_op, the
// xolotl ggml patch) and a CPU fallback.  Each replaces a chain of tiny ggml kernels with one, doing the same float
// operations in the same order, so the results match the op chains they replace.
#pragma once
#include "ggml.h"

// out = scale * (x + conv(state ++ x, w) [+ sum_lag win(lag) * dyn(lag, head)]), and the state takes the last K-1
// inputs (written in place).  x [C, T] contiguous, w [K, C], state = a [K-1, C] view of the slot's conv state,
// dyn = nullptr or [K * nh, T] (per-head input-dependent taps).  The chain it replaces: concat + cpy + ssm_conv
// (+ K x (cont transpose + mul + add)) + add + scale.
ggml_tensor* inkling_conv_step(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* state, ggml_tensor* dyn,
                               int K, int nh, float scale);

// SiTU on a gate|up tensor gu [2F, k, T]: beta*tanh(g/beta)*sigmoid(g) * (lb > 0 ? lb*tanh(u/lb) : u) -> [F, k, T].
// The chain it replaces: 2 conts + scale + tanh + scale + sigmoid + mul (+ scale + tanh + scale) + mul.
ggml_tensor* inkling_situ(ggml_context* c, ggml_tensor* gu, float beta, float lb);

// [E, n, T] from n tensors of [E, T] (contiguous F32, n <= GGML_MAX_SRC): row i = srcs[i]; nullptr if not applicable
ggml_tensor* inkling_stack(ggml_context* c, ggml_tensor* const* srcs, int n);

// true when the ggml build runs these on CUDA (the custom-op hook is present); else the engine keeps the op chains
bool inkling_fused_available();
