// include/strata/core/dense_model.hpp - a DENSE qwen35 model on Strata's kernels: Qwen3.8-27B (unsloth GGUF).
//
// WHY THIS IS A SEPARATE RUNNER AND NOT A MODE OF `strata`.  The MoE engine is specialised end to end
// (check_architecture: `qwen4exp`, 2560 wide, 48 blocks; expert arena; QSA indexer; hyper-connections; PLE;
// n-gram table).  A dense model has none of it, so `strata` refuses its GGUF at the first guard.  What the two
// models DO share is the GDN mixer (S = 128, 16 QK heads, 48 V heads, 10240 conv channels, d_conv 4), the
// quantized GEMVs, RoPE and the attention output gate.  Those kernels are reused as they are.
//
// THE MODEL (qwen35, 64 layers + 1 MTP block in the file, n_embd 5120, n_ff 17408), per layer:
//
//     xn = rms_norm(x, attn_norm)
//     GDN layer  (3 of every 4):  qkv, z = attn_qkv xn, attn_gate xn
//                                 conv+silu -> l2(q), l2(k) -> alpha, beta = ssm_alpha xn, ssm_beta xn
//                                 gate = softplus(alpha + dt) * ssm_a ; beta = sigmoid(beta)
//                                 o = delta-rule(state, q, k, v) ; y = rms_norm(o) * ssm_norm * SiLU(z)
//                                 mix = ssm_out y
//     attention layer (1 of 4):   q_full, k, v = attn_q xn, attn_k xn, attn_v xn      q_full = [q | gate] per head
//                                 q, k = rope(rms_norm_head(q), rms_norm_head(k))       24 Q heads, 4 KV heads, 256
//                                 attn = softmax(q K^T / sqrt(256)) V ; attn *= sigmoid(gate)
//                                 mix = attn_output attn
//     x += mix
//     x += ffn_down( silu(ffn_gate xn2) * ffn_up xn2 ),   xn2 = rms_norm(x, post_attention_norm)
//
// then h = rms_norm(x, output_norm) and logits = output h.  The GGUF carries ONE extra block after the 64
// (qwen35.nextn_predict_layers = 1): the multi-token-prediction head.  It is NOT part of the trunk.
//
// COLUMNS.  run() takes up to kMaxCols consecutive tokens at once.  The GEMVs (which read every weight once per
// call, so they are where the time goes) then serve all the columns; the order-dependent parts (conv, delta-rule
// state, KV append) walk the columns one by one, so the result equals feeding the tokens through step() in
// sequence.  That is what makes both a fast prefill and speculative verification possible.
//
// MTP.  With `with_mtp` the block is loaded and the model can propose the next token (mtp_draft); the caller
// verifies it with a two-column run() that snapshots the recurrent state after column 0, and rollback() undoes
// column 1 if the proposal was wrong.  The MTP block keeps its own KV cache and is fed every position (mtp_ingest).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core
{

    struct DenseConfig
    {
        std::string arch; ///< general.architecture ("qwen35")
        int n_layer = 0;  ///< trunk layers (block_count - nextn_predict_layers)
        int n_nextn = 0;  ///< MTP blocks stored after the trunk
        int n_embd = 0, n_ff = 0, n_vocab = 0;
        int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
        int ssm_d_conv = 0, ssm_state = 0, ssm_k_heads = 0, ssm_v_heads = 0;
        /// Hyper-connections (the Flash-Next / Qwen4 gated residual): `hc` residual streams of `n_embd` and the
        /// rank of their down/up bottleneck. 0 = the plain pre-norm residual (Qwen3.8-27B, architecture qwen35).
        int hc = 0, hc_lr = 0;
        float rms_eps = 1e-6f, rope_base = 1e7f;
        int32_t eos_id = -1; ///< tokenizer.ggml.eos_token_id: the id that ends an answer
        int conv_channels() const { return 2 * ssm_state * ssm_k_heads + ssm_state * ssm_v_heads; }
        int value_dim() const { return ssm_state * ssm_v_heads; }
        int hc_dim() const { return hc * n_embd; }
        bool has_hc() const { return hc > 0; }
    };

    enum class Logits
    {
        None,
        Last,
        All
    };

    struct DenseOptions
    {
        bool mtp = false;       ///< load the MTP block and allow mtp_draft()
        bool mtp_force = false; ///< keep MTP even when the weights do not all fit in VRAM (otherwise it is switched off)
        int draft_max = 2;      ///< most tokens one draft proposes (1..kMaxCols-1); sizes the rollback snapshots
        /// KV cache storage, spelled as strata's own `--kv`: "fp16" (default), "int8" (8-bit, about 53% of fp16),
        /// "q4_0" (4-bit after a Hadamard rotation, about 28%) or "k8v4" (int8 keys, 4-bit values, about 40%).
        /// "f16", "q8_0" and "q4" are accepted too.  The 4-bit formats need head_dim 256.
        std::string kv = "fp16";
        /// xolotl: rollback snapshots (draft_max of them) without the MTP block, for an external drafter (DFlash2).
        bool spec_snapshots = false;
        /// xolotl: layers whose output rows run() and prefill() copy for an external drafter (DFlash2: its
        /// target_layer_ids, e.g. 5,19,33,47,61); empty = off.  See run_taps() / ring_taps().
        std::vector<int> taps;
        /// xolotl: MTP drafts score only these token ids (a frequency shortlist, NInfer's --lm-head-draft): the
        /// head rows are copied once into a smaller matrix, so each draft step reads a fraction of the 248K-row head.
        /// Verification keeps the full head, so the output is unchanged; a draft outside the list is just rejected.
        /// Empty = drafts use the full head.
        std::vector<int32_t> draft_vocab;
        int draft_vocab_extra = 16384;  ///< rows kept free after the shortlist for draft_vocab_extend()
        int64_t kv_resident = 0;        ///< cells per KV head kept in VRAM, the rest in host RAM (0 = all in VRAM, -1 = auto: what fits)
        int vram_reserve_mib = 0;       ///< VRAM left free for another program (the vision encoder) when budgeting
    };

    class DenseModel
    {
    public:
        static constexpr int kMaxCols = 8;
        /// Largest chunk prefill_setup() will size its scratch for.  The GEMVs of a chunk are read once for all of its
        /// tokens, so a bigger chunk is nearly free until the chunk's activations stop fitting comfortably in VRAM.
        static constexpr int64_t kPrefillMaxCols = 512;
        /// The chunk prefill_setup() uses when asked for 0.  The scratch for one chunk is a couple of hundred MiB,
        /// which is nothing next to the weights, and every extra token in a chunk is one more weight read avoided.
        static constexpr int64_t kPrefillDefaultCols = kPrefillMaxCols;

        DenseModel();
        ~DenseModel();
        DenseModel(const DenseModel &) = delete;
        DenseModel &operator=(const DenseModel &) = delete;

        /// Reads the GGUF (single file or the first shard of a split), checks every tensor it needs, uploads the
        /// weights and allocates the KV cache for `max_context` cells and the recurrent state.  `opt.mtp` also loads
        /// the multi-token-prediction block and `opt.draft_max` snapshots of the recurrent state (about 156 MB each)
        /// for rollback().  If the weights would not all fit in VRAM the MTP block is NOT loaded (has_mtp() stays
        /// false and a note is printed) unless `opt.mtp_force` is set: a draft is only worth making when no weight
        /// has to cross PCIe.
        bool load(const std::string &gguf_path, int64_t max_context, std::string &err, const DenseOptions &opt = {});

        const DenseConfig &config() const { return cfg_; }
        int64_t max_context() const { return max_context_; }
        uint64_t weight_bytes() const { return weight_bytes_; }
        /// Weights that did not fit in VRAM and sit in pinned host memory (read over PCIe each token).
        uint64_t host_weight_bytes() const { return host_bytes_; }
        uint64_t kv_bytes() const { return kv_bytes_; }
        bool has_mtp() const { return has_mtp_; }
        /// The KV cache format in use, as `--kv` spells it: fp16, int8, q4_0 or k8v4.
        const std::string &kv_type() const { return kv_name_; }
        int draft_max() const { return draft_max_; }

        /// Forgets the conversation: zeroes the recurrent state, the last hidden state and the position.
        void reset();
        /// Number of tokens the state has consumed (= the next position).
        int64_t position() const { return pos_; }

        /// Feeds `n` (1..kMaxCols) consecutive tokens starting at position().  `logits` selects which columns get the
        /// output head (Last: only the final column, in logits_col(0); All: column j in logits_col(j)).
        /// `keep_hidden` leaves the post-norm hidden state of every column for the MTP head.  `snapshot` saves the
        /// recurrent state after each of columns 0..n-2 (n - 1 <= draft_max) so rollback() can drop the columns after
        /// any of them.
        bool run(const int32_t *tokens, int n, Logits logits, bool keep_hidden, bool snapshot, std::string &err);
        /// One token (kept for the simple paths).
        bool step(int32_t token, bool want_logits, std::string &err)
        {
            return run(&token, 1, want_logits ? Logits::Last : Logits::None, false, false, err);
        }
        /// Undoes the columns after `keep_col` of the last run() (which must have snapshotted, and keep_col < n - 1):
        /// the state and position() go back to just after column `keep_col`.  O(1): snapshot buffers are swapped in.
        bool rollback(int keep_col, std::string &err);
        /// Blocks until everything queued has run.
        bool sync(std::string &err);

        float *logits_col(int j) const; ///< device pointer to n_vocab logits, valid after a run() with logits
        void *stream() const;

        // ---- batched prompt processing.
        //
        // run() can only serve kMaxCols columns, because every activation buffer is sized for that many and the GEMV
        // kernels take at most 8 columns.  A prompt is therefore fed 8 tokens at a time and every weight is read again
        // for each group of 8.  prefill() instead walks the prompt in chunks of prefill_chunk() tokens: the projections
        // go through llama.cpp's MMQ int8 tensor-core kernels over the whole chunk (the weights stay in their GGUF
        // blocks), and the order-dependent steps (the GDN conv and delta-rule state, the KV append, the causal mask)
        // walk the chunk's tokens inside one launch, so the result matches feeding the same tokens through run().
        /// Sizes the chunk scratch and turns the batched path on.  `chunk` is clamped to
        /// [kMaxCols, kPrefillMaxCols]; 0 asks for kPrefillDefaultCols.  Returns false, and leaves prefill() unusable,
        /// when a weight's GGUF type is not one MMQ covers or the scratch does not fit: the caller then uses run().
        bool prefill_setup(int64_t chunk, std::string &err);
        /// False until prefill_setup() has succeeded.
        bool prefill_ready() const;
        /// The chunk prefill_setup() settled on (0 when it has not run).
        int64_t prefill_chunk() const;
        /// Feeds `n` consecutive tokens at position() in chunks of prefill_chunk().  Same state changes as the same
        /// tokens through run(): position(), the recurrent state, the KV cache, and the MTP block (fed in kMaxCols
        /// groups, exactly as the prompt loop does today).  `logits` may be None or Last (logits_col(0)).  Requires
        /// prefill_ready(); the caller falls back to run() otherwise.
        bool prefill(const int32_t *tokens, int64_t n, Logits logits, std::string &err);
        // ---- conversation checkpoints: GDN state + conv history + last hidden, in pinned HOST memory (~154 MiB each).
        bool ckpt_init(int slots, std::string &err); // 0 = off
        int ckpt_slots() const { return ckpt_n_; }
        int64_t ckpt_pos(int slot) const;              // position held by the slot, -1 if empty
        bool ckpt_save(int slot, std::string &err);    // state after position() tokens; async, stream-ordered
        bool ckpt_restore(int slot, std::string &err); // position() becomes the slot's position
        void ckpt_drop(int slot);
        // ---- the MTP head (with_mtp only)
        /// Feeds the MTP block the pairs (tokens[j], hidden[j-1]) for j in [start, n) at positions pos0 + j, where
        /// hidden[-1] is the last hidden state before this run.  Precondition: run(tokens, n, ..., keep_hidden=true)
        /// has just executed at pos0 (= position() - n).  Afterwards the "last hidden" is column n-1's.
        bool mtp_ingest(const int32_t *tokens, int n, int pos0, int start, std::string &err);
        /// Proposes up to `max_n` tokens that follow `token`, which sits at `pos` and has not been fed to the trunk.
        /// The first goes through the MTP block as the pair (token, last hidden); each further one chains on the MTP
        /// block's own output (llama.cpp does the same for a single head).  Drafting stops early at the first token
        /// whose probability is below `p_min` (that token is not returned).  Greedy.  *n_out receives the count.
        bool mtp_draft(int32_t token, int pos, int max_n, float p_min, int32_t *out, int *n_out, std::string &err);
        /// xolotl: adds the tokens of `tokens` (a request's prompt) that the --draft-vocab shortlist lacks to the
        /// drafts' head, newest first, up to DenseOptions::draft_vocab_extra rows; replaces the previous request's.
        /// No-op without a shortlist.  Drafts only: verification is untouched.
        void draft_vocab_extend(const int32_t *tokens, int64_t n);
        /// Makes column `col` of the last run() the "last hidden" (used after a rollback).
        void set_last_hidden(int col);

        // ---- xolotl: an external drafter's inputs (DenseOptions::taps)
        static constexpr int kTapRing = 2048;
        int n_taps() const;
        /// Device [kMaxCols][n_taps * n_embd]: the tap layers' output rows of the last run() (column j = row j).
        const float *run_taps() const;
        /// Device [kTapRing][n_taps * n_embd]: prefill() writes the row of position p at p % kTapRing (the last
        /// kTapRing rows of every prompt chunk).
        const float *ring_taps() const;
        /// The token's embedding row (n_embd floats) on the host, dequantized from the model file.
        bool embed_host(int32_t token, float *out, std::string &err) const;
        /// The vision path (--vision): rows[c] is the n_embd-float embedding of cell c (an image row from strata-vision)
        /// or nullptr (the cell's token goes through the embedding table as usual), for cells 0..n-1.  run() and
        /// prefill() read it; the MTP drafter keeps the table row of the token (<|image_pad|>), as strata's does.  The
        /// array and the rows must stay alive while they are set; nullptr (the default) turns it off.
        void set_embd_rows(const float *const *rows, int64_t n);
        /// The output head as native_mmvq reads it: weights (device or mapped), type, rows (= n_vocab).
        const void *head_weights() const;
        int head_type() const;
        int64_t head_rows() const;

    private:
        struct Impl;
        DenseConfig cfg_;
        int64_t max_context_ = 0, pos_ = 0, snap_pos0_ = 0;
        int64_t pf_chunk_ = 0;
        bool pf_ready_ = false;
        int snap_cols_ = 0, draft_max_ = 0;
        bool snap_valid_ = false, has_mtp_ = false;
        int ckpt_n_ = 0;
        std::string kv_name_ = "fp16";
        uint64_t weight_bytes_ = 0, kv_bytes_ = 0, host_bytes_ = 0;
        std::unique_ptr<Impl> impl_;
    };

} // namespace strata::core
