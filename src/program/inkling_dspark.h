// src/program/inkling_dspark.h - the DSpark block drafter for Inkling-2-Mini (GGUF arch `dflash`, 318 M params).
//
// The drafter sees the target's features at four layer-visit points (`dflash.target_layers = [5, 23, 41, 48]`),
// fuses them with a learned 4x4 softmax (row = drafter layer, row 3 = the markov head), and predicts a block of
// `dflash.block_size` (8) token positions in one pass: 3 dense layers of bidirectional attention over the committed
// context plus the block, then a 256-rank markov head that folds the LM head and runs on the host in fp32.
//
// THE CONVENTIONS (validated against a torch reference of the model card's MTP drafter, 6/7 accepted on p0):
//  * target_layers are LAYER-VISIT indices, not cache slots.  In xolotl-inkling the tensors are the dumped
//    `l_out-4` (visit 5), `l_out-28` (visit 23), `l_out-46` (visit 41) and the final normed `result_norm` (visit 48).
//  * block positions are T..T+7 (T = committed tokens), NOT the anchor's position.  Row 0 is
//    rms(embed(anchor), embed_norm); rows 1..7 are the mask_embd vector.  Context K/V holds every committed position
//    (the anchor included), rope'd at its own absolute position.
//  * drafter layers 0 and 2 are local (window 1025 = the python's `ck[max(0, anchor-1024):]`), layer 1 is global.
//  * markov: state = 0, prev = anchor; per slot i = 1..7 the 256-rank recurrence (fp32, host) writes
//    logits = 30*tanh((base_i + fold(latent))/30), and prev = argmax(logits[:200058]).
//
// The drafter's embeddings (token_embd, token_embd_norm) and LM head (output.weight) come from the TARGET GGUF.
//
// ---- the ~15-line integration into src/program/inkling_main.cpp -----------------------------------------------
//  1. flag: `--dspark <path>` -> std::string dspark_path (the drafter GGUF).  After `load(m, ...)`:
//         DsparkTarget tgt{m.tok_embd, m.tok_norm, m.output};
//         m.dspark = dspark_load(dspark_path, m.gpu, tgt, m.n_ctx);      // null when the flag is absent
//     (`m.tok_embd` is the host token table, `m.tok_norm` is token_embd_norm, `m.output` is output.weight.)
//  2. taps: build() gains a 5th output - the final normed features.  Immediately after
//         cur = g.rms(cur, m.output_norm);
//     add `ggml_set_output(cur);` and return it as `Step::result_norm`.  The other three taps are already the
//     existing dump taps: `s.taps[4]`, `s.taps[28]`, `s.taps[46]` (call forward() with taps != nullptr).  Each tap
//     row is [T][n_embd] f32 in token order.
//  3. after every forward() while drafting is on: gather the four tap rows for the chunk's T tokens into
//     `std::vector<float> ft((size_t) T * 4 * n_embd)` in tap order 5, 23, 41, 48, then
//         dspark_add_ctx(*m.dspark, ft.data(), T, p0);
//  4. before a verify forward at committed length T:
//         int32_t draft[7]; int n = dspark_draft(*m.dspark, last_committed_id, T, draft);
//     then run the verify forward on the prompt + draft and accept the prefix.  dspark_reset() whenever the
//     target state restarts (state_reset()).
// None of this needs the K/V or conv machinery of the target: the drafter keeps its own caches.
#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <vector>
#include <string>

struct ggml_tensor;
struct Dspark;

// The three target tensors the drafter borrows: the token table (host or GPU resident), its norm and the LM head.
struct DsparkTarget {
    ggml_tensor* embed = nullptr;       // token_embd.weight      [n_embd, n_vocab]
    ggml_tensor* embed_norm = nullptr;  // token_embd_norm.weight [n_embd] f32
    ggml_tensor* unembed = nullptr;     // output.weight          [n_embd, n_vocab_pad]
};

// Loads the 54-tensor drafter GGUF into VRAM on `gpu` (~0.65 GiB) plus its K/V caches, and folds the target LM head
// through the markov head's `dspark_hidden_out` once (the fold is what makes the per-slot markov bias one 256-wide
// matvec instead of a full 2304-wide LM matmul).  `unembed` must already be allocated on `gpu`.
// `prefix` empty: `path` is the drafter's own GGUF (general.architecture dflash).  Non-empty: the drafter is built
// into another GGUF (the model's own, tools/inkling/embed_dspark.py): its tensors are `<prefix><name>` and its
// `dflash.*` keys sit beside the model's keys; only those tensors are loaded.
Dspark* dspark_load(const std::string& path, ggml_backend_t gpu, const DsparkTarget& tgt, int64_t n_ctx,
                    const std::string& prefix = "");
// true when `path` carries a built-in drafter under `prefix` (metadata only, no tensor data is read)
bool dspark_in_gguf(const std::string& path, const std::string& prefix = "dspark.");
void dspark_free(Dspark* d);

// Forget every committed position (the target's state was reset).  The folded LM head survives.
void dspark_reset(Dspark& d);

// Forget committed positions >= p (the target rewound to p).  False when the drafter holds fewer than p positions
// (it then cannot draft for this context until a reset).
bool dspark_rewind(Dspark& d, int64_t p);
int64_t dspark_ctx_end(const Dspark& d);

// What a rewind cannot recover: the local layers' ring caches (later positions overwrite their cells) and the anchor
// taps.  The global layer is positional (cells < p stay valid until rewritten), so it is not copied.  The server keeps
// one per prompt checkpoint; restoring it puts the drafter exactly where it was at that position.
struct DsparkSnap {
    int64_t ctx_end = -1;
    std::vector<std::vector<uint8_t>> k, v;
    std::vector<float> anchor;
};
void dspark_snap(const Dspark& d, DsparkSnap& s);
void dspark_restore(Dspark& d, const DsparkSnap& s);

// Append the context K/V of committed positions p0 .. p0+T-1 from their target taps.  `taps` is
// [T][4][n_embd] f32, row-major, tap order = dflash.target_layers (5, 23, 41, 48).
void dspark_add_ctx(Dspark& d, const float* taps, int T, int64_t p0);

// Draft the block after T committed tokens.  `anchor` is token T-1 (the last committed id).  Writes up to 7 ids to
// `out` and returns how many (block - 1 = 7).  Requires add_ctx to have covered positions 0..T-1.
// Returns the confident count (XOLOTL_DSPARK_TAU: cumulative top-1 probability); with n_all the drafting continues
// past it and *n_all = every id written (out must hold 7).
int dspark_draft(Dspark& d, int32_t anchor, int64_t T, int32_t* out, int* n_all = nullptr);

// ------------------------------------------------------------------------------------------------ standalone gate
// `xolotl-inkling-dspark` (built from inkling_dspark.cpp with -DXOLOTL_DSPARK_TOOL) runs the drafter alone against a
// set of target taps on disk, so the block math can be gated without wiring the engine:
//   xolotl-inkling-dspark --target <inkling.gguf> --dspark <drafter.gguf> --taps <dir>
//                         --ids a,b,c,... --gen id,id,... [--anchor N]
// `--taps` holds the target taps for the prompt: l_out-4.f32, l_out-28.f32, l_out-46.f32 and result_norm.f32, each
// [T][n_embd] f32.  `--gen` is the target's greedy continuation (8 ids; the compared prefix is 7).
