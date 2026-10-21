// tools/v41/engine/ds41_dense.hpp - DeepSeek-V4.1-Flash (`deepseek41`): the dense (non-routed-expert) half.
//
// Math = DeepSeek's inference/model.py (tools/v41/engine/ds41_golden.py runs it on the GGUF's own weights; that is
// this file's gate).  The structure follows tools/ds4/ds4_dense (ggml graphs, CPU or CUDA backend, per-layer graphs
// built once and reused), but V4.1 differs enough that it is its own class:
//
//   - one PASS = n consecutive tokens at positions pos0..pos0+n-1 (n = 1: decode; n = a prompt chunk: prefill);
//     every graph is the same code for both (state rings are read as [ring | the pass's rows] and written back)
//   - hyper-connections single pass: a sublayer's hc_mixes give the NEXT sublayer's pre-mix (pre_state carries the
//     attention pre-mix from layer l's FFN to layer l+1)
//   - no per-head q norm; window K/V ring of 128; compressed KV (ratio 2 / 1) only on kv source layers, read by the
//     later layers of the same ratio; indexers on index source layers, their top-k reused by the layers after them;
//     layer `cand_layer` (20) picks 2048 candidate blocks of 8 that the later indexers rank inside
//   - attention = each query's own key set (its window + its top-k compressed rows) gathered and attended in one
//     flash-attn with the query as the batch axis (sinks added)
//   - Engram at layers 1 / 14: n-gram hash ids (host, from the token history) -> rows read from the HF shards
//     (Ds41Engram) -> wkv -> per-hc gated add, before the block
//
// Per layer per pass:  attn_router(l) -> [routed experts on fn] -> finish(l).  Then logits(...).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ds41_geometry.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct Ds41DenseConfig {
    ggml_backend_t backend = nullptr;   ///< NULL = create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx_max = 65536;            ///< positions the caches hold (compressed rows = ctx_max / ratio)
    int     max_pass = 1;               ///< tokens a pass may carry (>= 1; prefill chunk size = the sub-batch size S)
    /// Layer-major prompt batch (0 = off): the state tensors hold this many token columns and a prompt of up to
    /// `prompt_batch` tokens is walked layer by layer, one dense sub-batch of `max_pass` tokens at a time, with the
    /// experts run ONCE per layer over the whole batch (begin_batch / attn_router_span / finish_span).
    int     prompt_batch = 0;
    bool    skip_routed_experts = true; ///< do not load the routed experts (the tier owns them)
    /// Directories holding the HF shards with the Engram tables (model.safetensors.index.json in the first).
    /// Empty = no Engram (its add is skipped: a debugging mode, the model is then not the model).
    std::vector<std::string> engram_dirs;
    std::string engram_hot;             ///< hot-set pack (ds41_engram_hot pack); "" = STRATA_ENGRAM_HOT / none
    int     engram_threads = 32;        ///< parallel row reads (the disk's queue depth; cold peaks at 32-64)
    /// KV caches (window ring, compressed rows, indexer keys) element type: GGML_TYPE_F32 / BF16, or -1 = auto: BF16
    /// with act_quant (every cached value is then an FP8/FP4 grid point times a power-of-two / E4M3 scale: <= 7
    /// significant bits, exact in BF16 - half the VRAM of F32 at 512K), F32 without.
    int     kv_type = -1;
    /// model.py's activation rounding (ds41_ops.hpp): window K FP8, compressed KV FP4/E4M3, indexer q/k FP4.
    bool    act_quant = true;
    /// --kv-fp4 (needs act_quant, CUDA): the compressed KV as NVFP4 and the indexer keys as MXFP4 - their fake-quant
    /// grids exactly (e2m1 x E4M3 per 16 / x 2^k per 32), packed bit for bit by the vendored ggml-cuda's set_rows
    /// (cpy-utils.cuh *_ds41_block): 4.5 / 4.25 bits a value instead of BF16's 16 - ~3.5x less KV VRAM at long ctx.
    bool    kv_fp4 = false;
    /// --serve: start with max_pass (S) state columns and let set_columns() grow them for a layer-major batch and give
    /// them back after it (the prompt_batch columns are ~2.6 GiB at 8192 - expert slots while decoding).
    bool    elastic_cols = false;
    bool    vision = false;            ///< opt-in image embeddings, VL routing and Engram masking

    /// Batch slots (--batch-slots N): conversation states held at once, each of the full ctx_max - its window rings,
    /// ratio-2 compressor states, compressed KV + indexer keys, snapshot, park and Engram history.  Slot 0 is the one
    /// a single-slot engine has always had (n_slots = 1 = the engine as before, byte for byte).  Every single-sequence
    /// call works on the BOUND slot (bind_slot); begin_rows() runs rows of several slots in one pass.
    int     n_slots = 1;
    int64_t layer_lo = 0;   ///< layer split: first layer this instance runs (inclusive)
    int64_t layer_hi = -1;  ///< exclusive end; -1 = all layers in the file
    /// Tail stage with layer_hi == n_layer: skip output_norm / output.weight (they live on head_device).
    bool omit_output_head = false;
    /// Logits-only instance: output_norm + output.weight and a minimal x/pre state (no transformer layers).
    bool head_only = false;
    int cuda_device = -1;   ///< CUDA ordinal (for peer copies); -1 = unknown
    /// Layers below layer_lo whose KV / indexer state is copied from the previous pipeline stage at handoff.
    std::vector<int64_t> shadow_layers;
};

class Ds41Dense {
public:
    struct Impl;
    Ds41Dense();
    ~Ds41Dense();
    Ds41Dense(const Ds41Dense &) = delete;
    Ds41Dense & operator=(const Ds41Dense &) = delete;

    bool init(const std::string & gguf, const Ds41DenseConfig & cfg, std::string & err);
    const strata::v41::Ds41Geometry & geom() const;
    /// Vision was enabled at init and the model carries the per-layer VL routing biases.
    bool vision() const;
    /// This request's image span rows: `rows` is n_rows * n_embd floats, `pos` the n_rows absolute prompt positions
    /// (ascending) they overwrite.  They replace the token embeddings there, select the VL routing bias and mark the
    /// positions DEAD for Engram.  Call before the pass(es) / batch that feed those positions; clear with
    /// clear_vision() (a decode step after an image prompt carries no positions and needs no clearing, but a request
    /// with a different image does).  rows/pos must stay alive until those positions have been fed.
    void set_vision(const float * rows, const int64_t * pos, int n_rows);
    void clear_vision();
    /// Blocks present in the file (all 40 for the real model; a prefix for a partial test file).
    int64_t n_layer() const;
    ggml_backend_t backend() const;

    /// Start a pass: tokens tids[0..n) at positions pos0.., n <= max_pass.  Embeddings + Engram rows (the hash
    /// needs the 3 tokens before pos0: the history is kept by position, so a re-decoded position overwrites it).
    bool begin(const int * tids, int n, int pos0);

    // ---- batch slots (see Ds41DenseConfig::n_slots and docs/BATCHING.md)
    int n_slots() const;
    /// The bound slot: begin / begin_batch / spans / snapshot / restore / park / unpark / rearm / state_regions /
    /// pool_regions / set_engram_history all act on its state.  0 at init.
    int slot() const;
    /// Bind slot s (0 <= s < n_slots).  O(1): pointers only, no device work; the decode graphs of every slot stay
    /// cached (keyed by slot).  Call between passes, never inside one.
    bool bind_slot(int s);
    /// Exchange two slots' whole state (device tensors, snapshot, park, Engram history) - O(1), no copy: a
    /// conversation moves to another slot id.  The bound slot id stays bound (it now holds the other state).
    bool swap_slots(int a, int b);
    /// Device bytes one slot's state takes (rings + snapshot rings + compressor states + compressed KV + indexer keys).
    size_t slot_bytes() const;
    /// One row of a batch pass: token `tid` at position `pos` of slot `slot`'s conversation.
    struct Row { int slot = 0; int pos = 0; int tid = 0; };
    /// The most rows a batch pass takes (= Ds41MoeTier::kMaxMulti: one tier call per layer).
    static constexpr int kMaxRows = 16;
    /// Start a batch pass over rows of several slots: rows are grouped by slot (a slot's rows contiguous, at
    /// consecutive ascending positions; one slot per group), n <= kMaxRows and n <= the state columns.  A slot may
    /// carry one row (a decode step) or several (a piece of its prompt, or a verify).  Text only (no image rows).
    /// Then attn_router / predict_multi / finish / logits / read_state run as for a pass of n tokens: row i is pass
    /// column i.  Every slot's own window ring / compressor state / compressed rows / Engram history advance exactly
    /// as a solo pass of its rows would advance them; the numbers differ from a solo pass only by the row-batched
    /// matmuls (the same model, not the same bits).  The bound slot does not change.
    bool begin_rows(const Row * rows, int n);

    /// Layer-major prompt batch (cfg.prompt_batch > 0): the whole batch's inputs are prepared once - embeddings,
    /// positions, window / compressed / indexer index maps, Engram rows and the initial x/pre state - into the state
    /// tensors' columns 0..B-1; then attn_router_span / finish_span walk layer il's sub-batches in order.  B <=
    /// cfg.prompt_batch (the state columns) and B <= ctx_max - pos0.  Every cross-layer dependency of the sequential
    /// chunk path (the per-layer KV ring / compressed rows / indexer state advances in sub-batch order; layer l's
    /// residual and pre-mix are the columns the sub-batch used) is preserved, so the outputs match --prefill-chunk.
    bool begin_batch(const int * tids, int B, int pos0);
    /// Dense half of layer il for the batch's sub-batch starting at column col0 (n tokens, col0 % max_pass == 0):
    /// attention, hc, router, shared expert + the Engram add.  ids / wts / fn land in the batch buffers (batch_ids /
    /// batch_wts / batch_fn) at column col0.  Uses (and advances) layer il's own KV state in sub-batch order.
    bool attn_router_span(int il, int col0, int n);
    /// Encoder-only skip (enc-skip): build and run ONLY the persistent-KV half of layer il for the batch's sub-batch
    /// at column col0 - hc_pre / attn_norm, the raw window K/V ring write and, on a kv source layer, the compressed-KV
    /// row + indexer-key row + ratio-2 compressor state (all the state a LATER query can read across positions).  No
    /// attention output, no router, no shared expert, no hc_post / finish, no indexer selection / candidates: nothing
    /// that only ever flows into this layer's own output columns.  For prompt tokens whose layers past the replay
    /// window are never read (the model's cross-position reach is the 128-token window per layer, so the last
    /// `enc_replay` tokens are all that matters).  Writes no host ids/wts/fn, so batch_ids()/batch_wts()/batch_fn()
    /// are left untouched at these columns.
    bool attn_router_span_kv(int il, int col0, int n);
    /// The batch's host buffers filled by attn_router_span: ids / wts are n_used * B (token-major), fn is n_embd * B.
    const int32_t * batch_ids() const;
    const float * batch_wts() const;
    const float * batch_fn() const;
    int batch_size() const;
    /// finish for the batch's sub-batch at column col0: x_state[.., col0+n) = hc_post(shexp + routed, hap, post_f,
    /// comb_f).  `routed_host` is n_embd * B (token-major); call once per layer, before the sub-batches' finishes.
    bool upload_routed(const float * routed_host, int B);
    /// The routed-sum input tensor's device memory (columns of n_embd floats; null on the CPU backend): the tier can
    /// write a layer-major batch's routed sums straight into it (run_chunk_dev) instead of upload_routed's host trip.
    float * routed_dev();
    bool finish_span(int il, int col0, int n);
    /// STRATA_DENSE_PROF=1: print + reset the batch dense split (build / alloc / compute / Engram / readback) - diagnostic
    void prof_report(const char * tag);
    /// Give the prompt passes' compute buffer back to the device (rebuilt by the next multi-token pass): the deferred
    /// VRAM expert cache opens into it after the prompt.
    void release_pass();
    /// Called when a graph's buffer cannot be allocated; return true if it freed device memory (the allocation is
    /// then retried once).  E.g. the expert tier's prestage region.
    void set_oom_hook(std::function<bool()> hook);
    /// The pass-graph buffer (bytes) the widest span graph of a layer-major batch needs when its sub-batch of n columns
    /// ends at position pos_end: every layer kind's attention graph planned by a throwaway allocator (nothing is
    /// allocated), memoized on the visible compressed rows.  It grows with the context (the indexer's rows x queries).
    size_t span_graph_bytes(int pos_end, int n);
    /// Forget a narrowed span width: after a sub-batch's graph did not fit, attn_router_span keeps running narrower
    /// spans for the rest of the prompt; release_pass() or this resets it (the serve loop calls it per prompt).
    void reset_span_width();
    /// Selection-only routing bias for layer il (n_expert floats): added to the top-k score, not to the mixing weights.
    /// Soft pruning (-penalty on cold experts) / cache-aware routing (+bonus on VRAM-resident ones).  Default 0.
    void set_route_bias(int il, const float * bias);
    /// Smart router ranks (built when STRATA_RB_SMART=1 or STRATA_RB_PROTECT / STRATA_RB_POOL is set): the exact
    /// top-`protect` experts are always kept, and the bias only moves selections inside the exact top-`pool` (colibri's
    /// CACHE_ROUTE window).  0 = off; both 0 = the flat bias.  False when this engine has no smart router.
    bool set_route_select(int protect, int pool);
    /// Route meter (smart router only): since the last reset, the routed slots, how many of them differ from exact
    /// routing, and the share of the exact top-k's router weight the chosen experts kept (mean over tokens).
    struct RouteMeter { long long slots = 0, swaps = 0; double weight_kept = 1.0; bool on = false; };
    RouteMeter route_meter(bool reset);
    /// Layer il of the current pass: Engram (layers 1/14), attention, hc, ffn_norm, router, shared expert.
    /// ids / wts: n * n_used (token-major); *fn_host: n * n_embd floats (the routed experts' input, valid until the
    /// next call).  `fn_dev` (optional): the same activations on the backend.
    bool attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev = nullptr);
    /// Prefetch hint for layer il of a one-token pass (call before attn_router(il)): the layer's router applied to
    /// rms_norm(hc_pre(state, pre)) * ffn_norm.weight, + the selection bias, ranked; top_ids gets n_top (<= 16).
    /// A hint only: the result does not depend on it.
    bool predict(int il, int * top_ids, int n_top);
    /// predict() for EVERY column of the current pass (call before attn_router(il)): top_ids[t * 16 + i] = token t's
    /// i-th predicted expert of layer il (16 per token).  For the multi-token verify pass's expert prefetch; a hint only.
    bool predict_multi(int il, int * top_ids);
    /// routed_sum: n * n_embd floats, the routed experts' weighted sum (host).
    bool finish(int il, const float * routed_sum);
    /// Decode (n == 1) fusion: finish(il) and predict(next_il) in ONE cached graph - the router prediction for the next
    /// layer reads the state finish just wrote, so the result is identical to finish() then predict(), with one graph
    /// launch / sync / readback less per layer.  top_ids gets the next layer's top n_top (<= 16) predicted experts.
    bool finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top);
    /// finish_predict + the routing predictions of the `ahead` layers after next_il, from the same state, in the same
    /// graph: ahead_ids[a * 16 ..] = layer next_il + 1 + a's top 16 (-1 past the last layer).  For speculative
    /// file-tier reads; a hint only.
    bool finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                              int * ahead_ids);
    /// Logits of rows row0..row0+nrows-1 of the pass after the last layer's finish: nrows * n_vocab floats.
    bool logits(int row0, int nrows, float * out);
    /// Conversation reuse: every cache is position-indexed except the window ring and the ratio-2 compressor state
    /// ring, so a snapshot is those rings (~11 MB) + its position; restore() rewinds to it (decoding continues at
    /// snapshot_pos()).  Positions past a rewind point are overwritten when decoded again (rows past a query's
    /// visible count are never attended).
    bool snapshot(int pos);
    /// VRAM held by this half, by owner (weights on the device, KV state, columns, inputs, both graph allocators):
    /// one line for STRATA_VRAM_TRACE.
    std::string vram_ledger() const;
    bool restore();
    int snapshot_pos() const;
    /// Layer split: only run and retain KV for layers in [layer_lo, layer_hi).  head/logits require layer_hi == n_layer().
    void set_layer_range(int64_t layer_lo, int64_t layer_hi);
    int64_t layer_lo() const;
    int64_t layer_hi() const;
    bool owns_head() const;
    /// Current pass width / start position (set by begin / begin_batch / begin_*_continued).
    int pass_size() const;
    int pass_pos0() const;
    /// After a pipeline handoff: mirror the source pass layout so logits() row checks match (head-only and trunk).
    bool set_pass_layout(int n, int pos0);
    /// PCIe handoff between pipeline stages: per-column x_state [D,HC] then pre_state [HC], column-major (n columns).
    size_t handoff_bytes_per_column() const;
    bool export_handoff(int col0, int n, float * dst) const;
    bool import_handoff(int col0, int n, const float * src);
    /// Copy handoff columns from `src`: CUDA peer when available, else explicit pinned-host staging on `cuda_stream`.
    bool import_handoff_from(const Ds41Dense & src, int src_cuda_device, int dst_cuda_device, int col0, int n, void * cuda_stream,
                             std::vector<float> & pinned_host);
    /// Copy persistent / per-column layer state for a dependency layer from `src` (pipeline handoff).
    bool import_layer_cache_from(const Ds41Dense & src, int64_t global_layer, int col0, int n);
    bool import_cand_from(const Ds41Dense & src, int col0, int n);
    /// begin() without the embedding init graph; state columns must already hold the handoff (import_handoff).
    bool begin_continued(const int * tids, int n, int pos0);
    /// Layer-major batch after import_handoff: same span_cap / batch fields as begin_batch, no init graph.
    bool begin_batch_continued(const int * tids, int B, int pos0);
    /// Full dense+KV image for conversation parking (positions [0, positions)).
    bool capture_conversation(int64_t positions, std::vector<uint8_t> & out) const;
    bool restore_conversation(const std::vector<uint8_t> & in);
    /// Serve: keep a conversation through an UNRELATED request.  An unrelated request writes its own positions from 0, so
    /// it overwrites the start of every position-indexed cache (compressed KV, indexer keys) and the snapshot rings.
    /// park(n) copies the snapshot rings + the first n positions' rows of those caches (and their spare rows) to host
    /// memory; unpark() writes them back and re-arms the snapshot, so restore() resumes the parked conversation.  Valid
    /// only while no request has written past position n since park (the caller checks).
    bool park(int64_t positions);
    bool unpark();
    void drop_park();
    bool parked() const;
    /// A conversation's reusable device state at its snapshot, as regions of the device tensors: the snapshot rings
    /// (window K/V, ratio-2 compressor state) + rows [0, positions] of every compressed-KV / indexer-key cache and their
    /// spare rows - exactly what park() copies.  A conversation pool saves them right after snapshot(); writing them
    /// back and rearm(pos) + restore() resumes that conversation at pos.
    struct StateRegion { ggml_tensor * t; size_t off, bytes; };
    std::vector<StateRegion> state_regions(int64_t positions) const;
    /// The snapshot tensors now hold position `pos`'s state (a pool wrote them): restore() resumes there.
    void rearm(int pos);
    /// Engram history: the n-gram hash of a pass's first positions reads the tokens before it from the history, which
    /// is kept BY POSITION - another request that wrote those positions since leaves its own tokens there.  Rewrite
    /// [pos0, pos0 + n) with tids (DEAD where dead[i] != 0: an image position).  A no-op without Engram.
    void set_engram_history(int64_t pos0, int n, const int * tids, const uint8_t * dead);
    /// Conversation pool (ds41_conv_pool, ds41_conv_pool_dense.hpp): where a conversation's device state at a turn
    /// boundary lives.  `streams`: the position-indexed caches (compressed KV + indexer keys of every kv source layer;
    /// row r holds positions [r * ratio, (r + 1) * ratio), `rows` = cmax, the spare row excluded).  `regions`: the window
    /// rings and the ratio-2 compressor states - the *_snap copies (snap = true: what snapshot() / park() keep) or the
    /// live ones.  Pointers into the state buffer (device memory on CUDA); the sizes never change after init.
    struct PoolRegion {
        std::string name;
        void * data = nullptr;
        size_t row_bytes = 0, bytes = 0;
        int64_t rows = 0;
        int ratio = 0;
    };
    void pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const;
    /// Debug / gate taps: the hc state after the last finish (n * hc * n_embd floats, token-major) and the pre mix.
    bool read_state(std::vector<float> & x, std::vector<float> & pre);
    /// The hc state of state columns [c0, c0 + n) (token-major, n * hc * n_embd floats) - a layer-major batch's columns
    /// after finish_span (the serve DSpark seed reads its main_hidden there instead of re-running the tail as passes).
    bool read_state_cols(int c0, int n, std::vector<float> & x);
    /// Resize the state columns to max(S, min(nc, prompt_batch)) between passes (drops every cached graph); columns()
    /// is the current count.  A no-op when unchanged.
    bool set_columns(int nc);
    int columns() const;
    /// Gate tap: an index source layer's top-k rows of the current pass (n * K ints, token-major; K = min(cap, top-k))
    /// and their visibility mask; empty for other layers.
    bool read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K);
    /// The x state after begin (+ the Engram add of the layer `il`, when il is an Engram layer) - gate tap.
    const std::string & last_error() const;
    /// A loaded weight on the backend (nullptr if absent) - for sharing the trunk head with the DSpark draft stack.
    ggml_tensor * weight(const std::string & name) const;

private:
    std::unique_ptr<Impl> p_;
};
