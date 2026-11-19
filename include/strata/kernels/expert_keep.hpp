// include/strata/kernels/expert_keep.hpp - STRATA_EXPERT_KEEP: a simulated expert prune (task profiles, 2026-11-19).
//
// STRATA_EXPERT_KEEP=<file> lists, per MoE layer, the experts that STAY: one line "<layer>: <id> <id> ..." (the same
// file as llama.cpp's LLAMA_EXPERT_KEEP; '#' starts a comment; a layer with no line keeps every expert). The pruned
// experts' router logits are set to -1e30 before the top-k, so the softmax, the selection and the renormalisation
// see only the kept experts - the same routing as a GGUF with those router rows dropped, on the original file.
// Unset: nothing is allocated and nothing is launched (the output is the build's own, bit for bit).
#pragma once
#include <cstdint>

namespace strata::kernels {

// Parse the file into a host table (no CUDA); idempotent. Call early (before the expert profile / complement are
// planned) so they can skip pruned experts. false = a bad file (the caller stops).
bool expert_keep_load(int64_t n_layers, int64_t n_expert);
// Host query: true unless the mask is loaded and prunes (layer, expert). Out-of-range (e.g. MTP layers) = kept.
bool expert_keep_kept(int64_t layer, int64_t expert);
// Load (if not yet) + upload once, before any CUDA graph is captured. false = a bad file (the caller stops).
bool expert_keep_init(int64_t n_layers, int64_t n_expert);
bool expert_keep_on();
// logits: [n_tok][n_expert] router logits of `layer`, on the device; masked in place on `stream`.
void expert_keep_apply(float* logits, int64_t n_tok, int64_t n_expert, int64_t layer, void* stream);

}  // namespace strata::kernels
