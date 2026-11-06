// include/strata/prefill/moe_iq4_w4a8.hpp - the routed experts' IQ4_XS / IQ4_NL prompt products on int8 tensor
// cores (sm_80+), a pipelined replacement for llama.cpp's MMQ tiling-mode launch with the same bits.
#pragma once

#include "strata/prefill/moe_mmq.hpp"

namespace strata::prefill::mmq {

/// Runs p (IQ4_XS or IQ4_NL weights, q8_1 activations as Context::run takes them, experts at w + z * expert_bytes or
/// tab->p[z]) and returns true; returns false without launching anything when it does not cover p (another type,
/// rows not a multiple of 128, a row table, or a launch MMQ would split along K with its stream-k fixup - those sums
/// are added in a different order), and the caller runs Context::run.  Every output it writes is MMQ's tiling-mode
/// sum bit for bit: per 32-value block the exact int32 dot, then sum += float(dot) * d_w * d_y in block order.
bool run_iq4_w4a8(const Product& p, void* stream);

/// STRATA_IQ4_W4A8: 0 = off (the default); 1 = Context::run tries run_iq4_w4a8 first for every product; 2 = for the
/// IQ4_NL ones only (the down products - where MMQ wastes the most: K 640 in 256-value iterations).
int iq4_w4a8_mode();

}  // namespace strata::prefill::mmq
