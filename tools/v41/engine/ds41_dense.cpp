// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"
#include "ds41_embd.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <thread>
#include <sys/stat.h>

#ifdef _WIN32
#include <string.h>
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    bool failed = false;   // the last alloc failed: the gallocr holds a plan without a buffer (see alloc)
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        // A failed allocation leaves the gallocr PLANNED for that graph with NO buffer (ggml_gallocr_reserve_n frees
        // the old buffer, then the larger one does not fit).  The next alloc_graph of the same graph then finds the plan
        // current, skips the reserve and places the tensors in the null buffer - a segfault (2026-10-20: the oom_hook
        // retry on a 768K prompt).  So after a failure re-plan explicitly; the freed buffer also makes every graph
        // allocated before it stale (the epoch).
        if (failed && !ggml_gallocr_reserve(a, g)) { ++epoch; return false; }
        if (!ggml_gallocr_alloc_graph(a, g)) { failed = true; ++epoch; return false; }
        failed = false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

// The smart router (STRATA_RB_SMART=1, or STRATA_RB_PROTECT / STRATA_RB_POOL set): the protect / pool ranks come from
// the rknob tensor, so a request can change them (set_route_select) without a graph rebuild, and the router also
// writes the exact top-k for the route meter.  kRK = the deepest rank it can select.
constexpr int kRK = 32;
int env_rank(const char * k) {
    const char * e = std::getenv(k);
    return e ? std::max(0, std::min(kRK, std::atoi(e))) : 0;
}
bool smart_router() {
    static const bool on = env_on("STRATA_RB_SMART", false) || env_rank("STRATA_RB_PROTECT") > 0 || env_rank("STRATA_RB_POOL") > 0;
    return on;
}

}  // namespace

// ================================================================== Impl

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    int64_t layer_lo = 0, layer_hi = 0;   // [layer_lo, layer_hi) this instance runs and holds KV for
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    WStore w;
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * o_xids = nullptr;       // I32 [NU, NC]  route meter (smart router): the exact top-NU
    ggml_tensor * o_xwts = nullptr;       // [NU, NC]      ... and their normalized weights
    ggml_tensor * rknob = nullptr;        // F32 [kRK + 1, 2] smart router: protect / pool rank selectors (set_route_select)
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        // batch slots with a slab (Impl::slab): every slot's ring / compressed KV / indexer keys / compressor state of this
        // layer in ONE tensor, the slots' own tensors views of their region (a batch pass then writes and gathers every
        // row's in one op); null otherwise
        ggml_tensor * raw_all = nullptr, * comp_all = nullptr, * idxk_all = nullptr, * st_kv_all = nullptr, * st_sc_all = nullptr;
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
    std::vector<Layer> ly;
    // ---- batch slots: every slot's own conversation state.  The BOUND slot's lives in the members above (Layer::raw ..
    // idxk, snap_pos, parked, parked_pos) - which is what every single-sequence path reads, unchanged - and the other
    // slots' are parked here; bind_slot() exchanges them (pointers only).  slots[cur_slot] is stale while bound.
    struct SlotLayer {
        ggml_tensor * raw = nullptr, * raw_snap = nullptr, * st_kv = nullptr, * st_sc = nullptr, * st_kv_snap = nullptr,
                    * st_sc_snap = nullptr, * comp = nullptr, * idxk = nullptr;
    };
    struct Slot {
        std::vector<SlotLayer> ly;
        int snap_pos = -1, parked_pos = -1;
        std::vector<ParkedRows> parked;
    };
    std::vector<Slot> slots;
    int cur_slot = 0;
    bool slab = false;                  // the slot tensors are views of per-layer slabs (n_slots > 1, STRATA_SLOT_SLAB != 0)
    // a slot's region in the slabs (its tensors are views at region * their size; swap_slots moves the views, not data)
    int slot_region(int s) const {
        const SlotLayer T = slot_layer(s, 0);
        return T.raw && T.raw->view_src ? (int) (T.raw->view_offs / ggml_nbytes(T.raw)) : 0;
    }
    void stash_slot();                  // the bound members -> slots[cur_slot]
    void load_slot(int s);              // slots[s] -> the bound members; cur_slot = s
    void reset_graphs();                // drop every cached graph (their tensors' addresses changed) - set_columns too
    bool gctx_room() const;             // the cached-graph context can take another layer graph
    // a slot's layer tensors whether bound or not (the batch pass reads several slots at once)
    SlotLayer slot_layer(int s, size_t l) const {
        if (s == cur_slot) {
            const Layer & L = ly[l];
            return SlotLayer { L.raw, L.raw_snap, L.st_kv, L.st_sc, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
        }
        return slots[(size_t) s].ly[l];
    }

    std::vector<int64_t> shadow_layers;

    bool owns_layer(int64_t l) const {
        return l >= layer_lo && l < layer_hi;
    }
    bool active_layer(int64_t l) const {
        if (owns_layer(l)) return true;
        for (int64_t s : shadow_layers) if (s == l) return true;
        return false;
    }

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // a batch pass on slabs: the ratio-2 state rows it writes (their pass columns; i_sslot = their slab rows), and per
    // column the slab row its slot's compressed rows start at (F32: added to the top-k rows before the one gather)
    ggml_tensor * i_sscol = nullptr;      // I32 [2 * min(NC, kMaxRows)]
    ggml_tensor * i_cbase[3] = {};        // F32 [NC]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)
    ggml_tensor * zpad16 = nullptr;       // F16 [DH, 1] zeros: the flash-attn key padding (STRATA_ATTN_F16_GATHER)
    ggml_tensor * zidx = nullptr;         // I32 [kZidx] zeros: gathers that zero row once per pad slot
    // STRATA_ATTN_F16_GATHER=1 (CUDA): the attention's key rows are gathered straight into F16 and concatenated /
    // padded in F16, instead of gathered F32 -> concat F32 -> pad F32 -> cast F16.  Same values bit for bit (the
    // gather dequantizes in float and rounds with the same __float2half as the cast; concat and pad only move bits),
    // but the 512-query sub-batch's key tensors peak at ~0.74 GB instead of ~1.48 GB (the "1424 MiB" pass-graph
    // buffer of V41_START_HERE run20/21).
    bool f16_gather = false;

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- a batch pass (begin_rows): one group per slot - its columns [c, c + k), first position pos0, visible caps
    struct RowGroup { int slot; int64_t c, k, pos0; int64_t cap[3]; int reg = 0; };   // reg: the slot's slab region
    std::vector<RowGroup> rgroups;
    int64_t rows_nss = 0;                // a batch pass on slabs: the ratio-2 state rows it writes (i_sscol / i_sslot)
    bool rows_mode = false;
    // STRATA_ROWS_GRAPH_CACHE (default on, =0 off): a batch pass's graphs are cached by its LAYOUT (the groups' slot,
    // columns and visible caps): consecutive decode windows of the same slots reuse them (and ggml-cuda its CUDA graph
    // captures) instead of building / allocating 40 layer graphs a window.  rows_sig = the layout's key (0 = uncached).
    int64_t rows_sig = 0;
    std::map<int64_t, std::vector<int64_t>> rows_sigs;   // key -> the layout it stands for (a hash collision is not cached)
    std::vector<int> rows_slot;          // per row: its slot / position (the Engram reads)
    std::vector<int64_t> rows_pos;
    bool upload_rows(const Ds41Dense::Row * rows, int n);
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    // A sub-batch whose graph does not fit is run as narrower spans (attn_router_span): span_max is the width that
    // fitted, kept for the rest of the prompt (release_pass resets it); 0 = the full sub-batch.  alloc_failed: the last
    // compute() failed in the graph allocation (not in the compute) - the only failure a narrower span can cure.
    int64_t span_max = 0;
    bool alloc_failed = false;
    // The indexer's selection over a long context (build_attn): its [visible compressed rows x queries] score / mask
    // tensors are built for at most idx_tile elements at a time (query chunks); 0 = one piece.  Default 2^28 = a 512K
    // context's 512-query sub-batch, so every context up to 512K builds the very same graph as before.
    int64_t idx_tile = 0;
    std::map<std::array<int64_t, 3>, size_t> graph_bytes_memo;   // span_graph_bytes: (cap 1, cap 2, n) -> bytes
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_xids;       // route meter: o_xids / o_xwts readbacks, and the counters since route_meter(true)
    std::vector<float> host_xwts;
    long long rm_slots = 0, rm_swaps = 0, rm_rows = 0;
    double rm_kept = 0;
    const uint8_t * embd = nullptr;       // token_embd rows on the mmap (embd_type: F16 ours, Q5_K ISTA's)
    int embd_type = GGML_TYPE_F16;
    // ---- vision (DeepSeek-V4.1 image input): the request's image span rows, by absolute position.  When enabled,
    // set_vision() swaps in exp_probs_b_vl on those positions and
    // marks them DEAD for Engram.  Empty (n_vis = 0) for a text request.
    bool vision = false;                  // opt-in; requires per-layer exp_probs_b_vl.bias
    ggml_tensor * i_img = nullptr;        // F32 [NC]  1.0 at an image position, 0.0 elsewhere (the VL bias / gate mask)
    const float * vis_rows = nullptr;     // n_vis * D floats, already in prompt order
    const int64_t * vis_pos = nullptr;    // n_vis ascending absolute prompt positions
    int n_vis = 0;
    std::vector<std::vector<float>> eng_rows;
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    // slot: the batch slot whose state tensors the graph reads (the attention graphs; 0 for the others)
    struct GKey { int il, kind; int64_t cap, n; int slot = 0; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n, slot) < std::tie(o.il, o.kind, o.cap, o.n, o.slot); } };
    std::map<GKey, ggml_cgraph *> graphs;
    // pass graphs with 1 < n <= kPassGraphN (the DSpark verify widths: t + depth-4 drafts + a re-fed row), cached
    // like the n = 1 ones instead of built + allocated per layer per pass (DS41_PASS_GRAPHS=0: per pass, as before).
    // Their own node context, dropped wholesale when it runs low (the KV cap buckets move as the context grows).
    ggml_context * gctx_m = nullptr;
    std::map<GKey, ggml_cgraph *> mgraphs;
    long mgraph_resets = 0;
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs

    ~Impl() {
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (gctx_m) ggml_free(gctx_m);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // the Engram gate weight q_norm * k_norm [D, HC]: an upstream llama.cpp GGUF stores the product (engram_gate)
    ggml_tensor * engram_wqk(ggml_context * gc, int64_t il) const {
        const std::string g = "blk." + std::to_string(il) + ".engram_gate.weight";
        if (w.has(g)) return w.get(g);
        return ggml_mul(gc, BL(il, "engram_q_norm.weight"), BL(il, "engram_k_norm.weight"));
    }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    bool compute(ggml_cgraph * gf, bool cached);
    bool upload_inputs(const int * tids, int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    // the llama.cpp convention (ISTA-DASLab's GGUFs): its Engram names map to ours (ds41_canonical_name), its 3-D
    // attn_output_a [OGD, OL, OG] is our 2-D [OGD, OG*OL] (the same bytes), and its BF16 tensors are converted at load
    // to the type our file has there (fp32 router / Engram norms, F16 compressor / indexer) - BF16 is not an operand
    // type of every op the graph runs on them.  Quantized matrices (Q5_K ...) stay as they are: ggml multiplies them.
    const bool lc = ds41_is_llamacpp_gguf(model);
    auto canon = [&](const TensorInfo & ti) { return lc ? ds41_canonical_name(ti.name) : ti.name; };
    auto lc_type = [&](const TensorInfo & ti) -> int {   // -1 = keep the file's
        if (!lc) return -1;
        const std::string n = canon(ti);
        auto ends = [&](const char * x) { const size_t L = std::strlen(x); return n.size() >= L && n.compare(n.size() - L, L, x) == 0; };
        // upstream's engram_gate is an operand of an F32 multiply like the q_norm / k_norm it replaces: F32 whatever the
        // quantizer stored
        if (ends("engram_gate.weight")) return (ggml_type) ti.type == GGML_TYPE_F32 ? -1 : GGML_TYPE_F32;
        if ((ggml_type) ti.type != GGML_TYPE_BF16) return -1;
        return ends("ffn_gate_inp.weight") || ends("engram_q_norm.weight") || ends("engram_k_norm.weight") ? GGML_TYPE_F32 : GGML_TYPE_F16;
    };
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        int nd = (int) ti.shape.size();
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        const std::string cn = canon(ti);
        static const std::string oa = "attn_output_a.weight";
        if (lc && nd == 3 && cn.size() >= oa.size() && cn.compare(cn.size() - oa.size(), oa.size(), oa) == 0) {
            ne[1] *= ne[2];
            ne[2] = 1;
            nd = 2;
        }
        if (type < 0) type = lc_type(ti);
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), nd, ne);
        ggml_set_name(t, cn.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry
    auto blk_layer = [](const std::string & n) -> int64_t {
        if (n.rfind("blk.", 0) != 0) return -1;
        const size_t dot = n.find('.', 4);
        if (dot == std::string::npos) return -1;
        return std::atoll(n.substr(4, dot - 4).c_str());
    };
    auto in_dense_range = [&](const std::string & n) {
        if (im.cfg.head_only) return n == "output.weight" || n == "output_norm.weight";
        if (n == "token_embd.weight") return im.layer_lo == 0;
        if (n == "output.weight" || n == "output_norm.weight") return im.layer_hi == im.nl && !im.cfg.omit_output_head;
        const int64_t l = blk_layer(n);
        if (l < 0) return true;
        return l >= im.layer_lo && l < im.layer_hi;
    };
    // per_layer_token_embd: an upstream llama.cpp GGUF's Engram tables (~100+ GB) - Ds41Engram reads the tables
    auto skip = [&](const std::string & n) {
        return is_routed(n) || n == "token_embd.weight" || n == "per_layer_token_embd.weight" || !in_dense_range(n);
    };
    if (im.cpu) {
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti, (int) ti.type);   // (bound in place: the file's own type)
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[canon(ti)] = t;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        auto requant = [&](const TensorInfo & ti) {
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight");
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[canon(ti)] = make(ti, requant(ti) ? (int) rq : -1);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        // STRATA_DENSE_REQUANT_CACHE=<dir> (llm-server): the conversions below cost ~25 s a load (q6_K's reference
        // quantizer is CPU-bound; the reads are not the cost - measured with the source in the page cache).  Their
        // output is written once to <dir>/<model>.<key>.dense-rq, in the order this loop converts, and read back on
        // later loads.  The key: the model file's size and mtime, the requant / prediction types and the head switch;
        // a cache of the wrong total size is ignored (and rewritten).
        std::FILE * rq_in = nullptr;
        std::FILE * rq_out = nullptr;
        std::string rq_path, rq_tmp;
        if (const char * cd = std::getenv("STRATA_DENSE_REQUANT_CACHE"); cd && *cd && (rq != GGML_TYPE_COUNT || pq != GGML_TYPE_COUNT)) {
            size_t want = 0;
            for (size_t s = 0; s < model.size(); ++s)
                for (const auto & ti : model.shard(s).tensors()) {
                    if (skip(ti.name)) continue;
                    if (const std::string pn = pred_name(ti); !pn.empty()) want += ggml_nbytes(w.t[pn]);
                    if ((int) w.t[canon(ti)]->type != (int) ti.type) want += ggml_nbytes(w.t[canon(ti)]);
                }
            struct stat st {};
            const std::string mp = model.shard(0).path();
            if (want > 0 && ::stat(mp.c_str(), &st) == 0) {
                const size_t sl = mp.find_last_of('/');
                char key[160];
                std::snprintf(key, sizeof key, ".%llx.%llx.%s.%s.h%d.dense-rq", (unsigned long long) st.st_size,
                              (unsigned long long) st.st_mtime, rq != GGML_TYPE_COUNT ? ggml_type_name(rq) : "none",
                              pq != GGML_TYPE_COUNT ? ggml_type_name(pq) : "none", rq_head ? 1 : 0);
                rq_path = std::string(cd) + "/" + (sl == std::string::npos ? mp : mp.substr(sl + 1)) + key;
                struct stat cs {};
                if (::stat(rq_path.c_str(), &cs) == 0 && (size_t) cs.st_size == want) {
                    rq_in = std::fopen(rq_path.c_str(), "rb");
                } else {
                    rq_tmp = rq_path + ".tmp";
                    rq_out = std::fopen(rq_tmp.c_str(), "wb");
                    if (!rq_out) std::fprintf(stderr, "dense requant cache: cannot write %s\n", rq_tmp.c_str());
                }
            }
        }
        // rows of the file's tensor `ti` (bytes `src`) converted to t->type and uploaded; returns the bytes uploaded
        auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            if (rq_in) {
                if (std::fread(dst.data(), 1, dst.size(), rq_in) == dst.size()) {
                    ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
                    return dst.size();
                }
                std::fprintf(stderr, "dense requant cache: short read in %s - converting\n", rq_path.c_str());
                std::fclose(rq_in);
                rq_in = nullptr;
                // (a later tensor of this load is converted too: nothing below reads rq_in again)
            }
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                        else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            if (rq_out && std::fwrite(dst.data(), 1, dst.size(), rq_out) != dst.size()) {
                std::fclose(rq_out);
                rq_out = nullptr;
                std::remove(rq_tmp.c_str());
            }
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
            return dst.size();
        };
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[canon(ti)];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                rq_to += convert_upload(t, ti, src);
                ++rq_n;
                rq_from += (size_t) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
            }
        if (rq_n && rq == GGML_TYPE_COUNT)   // requant off: only type conversions (a llama.cpp GGUF's BF16 -> F32 / F16)
            std::fprintf(stderr, "dense convert: %zu matrices to the engine's types, %.2f -> %.2f GB\n", rq_n, rq_from / 1e9,
                         rq_to / 1e9);
        else if (rq_n) std::fprintf(stderr, "dense requant: %zu Q8_0 matrices -> %s, %.2f -> %.2f GB%s\n", rq_n, ggml_type_name(rq),
                               rq_from / 1e9, rq_to / 1e9, rq_in ? " (from the cache)" : rq_out ? " (cache written)" : "");
        if (rq_in) std::fclose(rq_in);
        if (rq_out) {
            const bool ok = std::fclose(rq_out) == 0 && std::rename(rq_tmp.c_str(), rq_path.c_str()) == 0;
            if (!ok) std::remove(rq_tmp.c_str());
        }
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
    }
    size_t si = 0;
    const TensorInfo * te = model.find("token_embd.weight", &si);
    if (!strata::v41::ds41_embd_type_ok((int) te->type)) { err = "token_embd type has no row reader"; return false; }
    im.embd = model.shard(si).tensor_data(*te);
    im.embd_type = (int) te->type;
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * indexer(ggml_tensor * q, ggml_tensor * k, ggml_tensor * w, ggml_tensor * nvis, int il, ggml_tensor ** vis) const {
        *vis = nullptr;
        ggml_tensor * m = nvis;
        if (!env_on("STRATA_INDEXER_NVIS", false)) { // reference composite for A/B checks
            const int64_t nk = k->ne[2], nq = q->ne[2];
            ggml_tensor * ramp = ggml_view_2d(c, im->neg_ramp, nk, 1, (size_t) nk * 4, 0);
            *vis = ggml_log(c, ggml_step(c, ggml_add(c,
                ggml_repeat(c, ramp, ggml_new_tensor_2d(c, GGML_TYPE_F32, nk, nq)), nvis)));
            m = ggml_cast(c, *vis, GGML_TYPE_F16);
        }
        ggml_tensor * sc = ggml_lightning_indexer(c, q, k, w, m);
        if (std::getenv("STRATA_INDEXER_DUMP") && m == nvis) {
            ggml_format_name(sc, "indexer.l%d", il);
            // Keep the input rows alive until compute() can export them for test_ds41_indexer --replay.
            for (ggml_tensor * t : {q, k, w, nvis}) ggml_set_output(t);
        }
        return sc;
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                  // [24, n]
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached) {
    Allo & a = cached ? allo_dec : allo_pass;
    // DS41_GRAPH_OPS=1 (diagnostic): the op mix of the first 200 distinct one-token graphs (count per op; matmul weight MiB)
    static const bool gops = std::getenv("DS41_GRAPH_OPS") != nullptr;
    static std::set<const ggml_cgraph *> gops_seen;
    if (gops && cached && gops_seen.size() < 200 && gops_seen.insert(gf).second) {
        std::map<std::string, int> cnt;
        double wmib = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            std::string k = ggml_op_desc(t);
            if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
                wmib += ggml_nbytes(t->src[0]) / 1048576.0;
                k += std::string("(") + ggml_type_name(t->src[0]->type) + ")";
            }
            ++cnt[k];
        }
        std::string s;
        for (const auto & kv : cnt) s += " " + kv.first + "x" + std::to_string(kv.second);
        std::fprintf(stderr, "graph ops: %d nodes, %.1f MiB matmul weights:%s\n", ggml_graph_n_nodes(gf), wmib, s.c_str());
    }
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    bool ok_alloc = a.alloc(gf, cached && env_on("DS41_GRAPH_REUSE", true));
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    alloc_failed = !ok_alloc;
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    // Opt-in input fixtures, one per indexer layer. No model outputs are changed.
    if (const char * dir = std::getenv("STRATA_INDEXER_DUMP")) {
        static std::set<std::string> saved;
        const int64_t min_kv = std::getenv("STRATA_INDEXER_DUMP_MIN_KV") ?
            std::atoll(std::getenv("STRATA_INDEXER_DUMP_MIN_KV")) : 65536;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_LIGHTNING_INDEXER || std::strncmp(t->name, "indexer.l", 9) ||
                t->src[1]->ne[2] < min_kv || t->src[3]->type != GGML_TYPE_F32) continue;
            const std::string path = std::string(dir) + "/" + t->name + ".idx";
            if (saved.count(path)) continue;
            bool contiguous = true;
            for (int j = 0; j < 4; ++j) contiguous &= ggml_is_contiguous(t->src[j]);
            if (!contiguous) continue;
            if (const char * fill = std::getenv("STRATA_INDEXER_DUMP_MIN_FILL")) {
                const ggml_tensor * nv = t->src[3];
                float last = 0;
                ggml_backend_tensor_get(nv, &last, (size_t) (nv->ne[1] - 1) * nv->nb[1], sizeof(last));
                if (last < std::atof(fill) * t->src[1]->ne[2]) continue;
            }
            std::FILE * f = std::fopen(path.c_str(), "wbx");
            if (!f) continue;
            bool ok = std::fwrite("DS41IDX1", 1, 8, f) == 8;
            for (int j = 0; j < 4 && ok; ++j) {
                const ggml_tensor * v = t->src[j];
                const int32_t type = (int32_t) v->type;
                const uint64_t size = ggml_nbytes(v);
                std::vector<uint8_t> bytes((size_t) size);
                ggml_backend_tensor_get(v, bytes.data(), 0, bytes.size());
                ok = std::fwrite(&type, 4, 1, f) == 1 && std::fwrite(v->ne, 8, 4, f) == 4 &&
                     std::fwrite(&size, 8, 1, f) == 1 && std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
            }
            ok = std::fclose(f) == 0 && ok;
            if (ok) { saved.insert(path); std::fprintf(stderr, "indexer fixture: %s\n", path.c_str()); }
            else { err = "indexer fixture write failed: " + path; return false; }
        }
    }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
// get_rows whose result is F16 (f16) or the usual F32.  ggml_get_rows always declares F32; the CUDA kernel writes any
// of F32 / F16 / BF16, dequantizing in float and rounding with the same conversion a later cast would use, so retyping
// the (not yet allocated) result is the gather + cast in one step without the F32 intermediate.  CUDA only.
static ggml_tensor * rows_as(ggml_context * gc, ggml_tensor * a, ggml_tensor * idx, bool f16) {
    ggml_tensor * t = ggml_get_rows(gc, a, idx);
    if (f16 && t->type == GGML_TYPE_F32) {
        t->type = GGML_TYPE_F16;
        t->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
    return t;
}

static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

// A batch pass's attention core on the slabs (Impl::slab): every row's window K/V goes into its slot's ring region and
// every query's SWA rows are gathered back in ONE op each; the compressor runs over every row (its state rows and
// compressed / indexer rows written by slab row); the indexer ranks per slot (its own rows and visible count), and the
// chosen rows of every query - its group's top-k, the rest of the longest group's padded with -inf - are gathered in one
// op and attended in ONE flash-attn.  The values are a solo pass's: the ring holds the rows the solo pass concatenates
// (the window's FP8 grid is exact in the ring type), and the -inf padding takes no part in the attention.
static ggml_tensor * attn_core_slab(Ds41Dense::Impl & im, ggml_context * gc, const B & b, int il, int64_t n, ggml_tensor * xn,
                                    ggml_tensor * qr, ggml_tensor * q, ggml_tensor * kv2, ggml_tensor * pos,
                                    std::vector<ggml_tensor *> & outs, std::vector<ggml_tensor *> & late) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const int64_t DH = im.DH, NH = im.NH, SWA = im.SWA;
    const int64_t r = L.ratio;
    const bool f16 = im.f16_gather && 255 * n <= im.zidx->ne[0];
    ggml_tensor * ring = ggml_set_rows(gc, L.raw_all, kv2, b.v1(im.i_wslot, n, 0));
    ggml_tensor * Kw = ggml_reshape_3d(gc, rows_as(gc, ring, b.v1(im.i_widx, SWA * n, 0), f16), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, 0);
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    if (r) {
        ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;
            if (r == 1) {
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv_all, skv, 1);                     // [DH, 2 NS + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc_all, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, 0);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                // each slot's state ring: its group's last min(k, 2) rows
                ggml_tensor * col = b.v1(im.i_sscol, im.rows_nss, 0), * dst = b.v1(im.i_sslot, im.rows_nss, 0);
                late.push_back(ggml_set_rows(gc, L.st_kv_all, ggml_get_rows(gc, skv, col), dst));
                late.push_back(ggml_set_rows(gc, L.st_sc_all, ggml_get_rows(gc, ssc, col), dst));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, 0);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, 0);   // slab rows
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk_all, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp_all, lr, cslot);
            outs.push_back(comp_w);
        }
        int64_t Kmax = 0;
        for (const auto & G : im.rgroups) if (G.cap[r] > 0) Kmax = std::max<int64_t>(Kmax, std::min<int64_t>(G.cap[r], im.ITOP));
        if (Kmax > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            ggml_tensor * comp = comp_w ? comp_w : KS.comp_all;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk_all;
            const L_t & IS = L.idx_src ? L : im.ly[(size_t) L.idxl];
            if (L.idx_src) {
                ggml_tensor * x_iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                x_iq = b.rope(ggml_reshape_3d(gc, x_iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) x_iq = fake_quant(gc, ggml_cont(gc, x_iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * x_iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                x_iw = ggml_scale(gc, x_iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));
                // every column's selection mask is -inf past its own group's top-k (graph order: before the groups write)
                outs.push_back(ggml_cpy(gc, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, Kmax, n), NEG_INF),
                                        ggml_view_2d(gc, L.selm, Kmax, n, L.selm->nb[1], 0)));
                for (const auto & G : im.rgroups) {
                    const int64_t gcap = G.cap[r], gk = G.k, gc0 = G.c;
                    if (gcap <= 0) continue;
                    const int64_t K = std::min<int64_t>(gcap, im.ITOP);
                    const size_t base = (size_t) G.reg * (size_t) (im.cmax[r] + 1);
                    ggml_tensor * iq = ggml_cont(gc, ggml_view_3d(gc, x_iq, im.ID, im.IH, gk, x_iq->nb[1], x_iq->nb[2], (size_t) gc0 * x_iq->nb[2]));
                    ggml_tensor * iw = ggml_cont(gc, ggml_view_2d(gc, x_iw, x_iw->ne[0], gk, x_iw->nb[1], (size_t) gc0 * x_iw->nb[1]));
                    ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, gcap, idxk->nb[1], idxk->nb[1], base * idxk->nb[1]);
                    const int64_t qc = im.idx_tile > 0 && gcap * gk > im.idx_tile ? std::max<int64_t>(1, im.idx_tile / gcap) : gk;
                    for (int64_t q0 = 0; q0 < gk; q0 += qc) {   // the query chunks of c8c7076a (one chunk up to a 512K context)
                        const int64_t nq = std::min(qc, gk - q0);
                        const bool wq = q0 == 0 && nq == gk;
                        const size_t c0q = (size_t) (gc0 + q0);
                        ggml_tensor * iq_c = wq ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                        ggml_tensor * iw_c = wq ? iw : ggml_view_2d(gc, iw, im.IH, nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                        ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, c0q * 4), 1, nq);
                        ggml_tensor * vis = nullptr;
                        ggml_tensor * sc = b.indexer(iq_c, ksrc, iw_c, nvs, il, &vis);
                        sc = ggml_reshape_2d(gc, sc, gcap, nq);
                        if (L.uses_cand) {
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                            ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                            m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                            m = ggml_view_2d(gc, m, gcap, nq, m->nb[2], 0);
                            sc = ggml_add(gc, sc, m);
                        }
                        if (L.cand_src) {
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            GGML_ASSERT(gcap % im.CBS == 0);
                            ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);
                            ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                            ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, c0q * 4), 1, nq);
                            ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                            bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                            ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));
                            outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                        }
                        ggml_tensor * s1 = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));
                        s1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s1, 1, K, nq),
                                                               ggml_argsort(gc, ggml_cast(gc, s1, GGML_TYPE_F32),
                                                                            GGML_SORT_ORDER_ASC)), K, nq);   // canonical order (build_attn)
                        ggml_tensor * m1 = vis ? ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, gcap, nq), s1), K, nq)
                                              : ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_scale(gc, ggml_cast(gc, s1, GGML_TYPE_F32), -1.0f), nvs)));
                        outs.push_back(ggml_cpy(gc, s1, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], c0q * L.sel->nb[1])));
                        outs.push_back(ggml_cpy(gc, m1, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], c0q * L.selm->nb[1])));
                    }
                }
            }
            // every query's rows: its group's top-k (relative to its slot's region) + the region's first slab row, then
            // ONE gather from the slab (the rows past a group's own top-k are masked)
            ggml_tensor * selv = ggml_cont(gc, ggml_view_2d(gc, IS.sel, Kmax, n, IS.sel->nb[1], 0));
            ggml_tensor * self = ggml_add(gc, ggml_cast(gc, selv, GGML_TYPE_F32), ggml_reshape_2d(gc, b.v1(im.i_cbase[r], n, 0), 1, n));
            ggml_tensor * sela = ggml_cast(gc, self, GGML_TYPE_I32);
            Kc = ggml_reshape_3d(gc, rows_as(gc, comp, ggml_reshape_1d(gc, sela, Kmax * n), f16), DH, Kmax, n);
            Mc = ggml_cont(gc, ggml_view_2d(gc, IS.selm, Kmax, n, IS.selm->nb[1], 0));
        }
    }
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                     // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;
    int64_t M = Kall->ne[1];
    if (!im.cpu && M % 256) {
        const int64_t pad = 256 - M % 256;
        // The F32 pad kernel clamps its grid at 65,535 rows (ggml-cuda/pad.cu: min(ne1, 65535)) and leaves every row
        // past that unwritten, so a slab pass whose visible key count crosses 65,535 would attend to uninitialised
        // device memory.  Same guard as the two sites below; past the limit build the pad rows explicitly (the same
        // zeros, no pad kernel).
        Kall = (f16 || M + pad > 65535)
             ? ggml_concat(gc, Kall, f16
                   ? ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * n, 0), true), DH, pad, n)
                   : ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, DH, pad, n), 0.0f), 1)
             : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, n)
                                                   : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    return ggml_reshape_3d(gc, o, DH, NH, n);
}

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        // The folded upstream gate uses the reference multiplication order and sign(0)=+1;
        // keep the older separate q/k graph's rounding unchanged.
        const bool folded_gate = im.w.has("blk." + std::to_string(il) + ".engram_gate.weight");
        ggml_tensor * wqk = im.engram_wqk(gc, il);   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, folded_gate ? ggml_mul(gc, ggml_mul(gc, hn, kn), wqk) : ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * sign = folded_gate ? ggml_scale_bias(gc, ggml_step(gc, ggml_neg(gc, dot)), -2.0f, 1.0f) : ggml_sgn(gc, dot);
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, sign, mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));   // gate * (1 - image)
        }
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
               &a_pre, &a_post, &a_comb);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));             // [D, n]
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = b.rope(q, pos, r);

    // ---- the attention core runs per GROUP: the pass's columns of one conversation (one batch slot) on that slot's own
    // state - its window ring, its compressor state, its compressed KV + indexer keys and its visible-row count.  A
    // single-sequence pass (begin / begin_batch spans) is ONE group of every column on the bound slot and builds the very
    // graph it always did (no views of the projections: `whole`); a batch pass (begin_rows) has one group per slot,
    // the projections / router / shared expert batched over all rows around it.
    struct Grp { int slot; int64_t c, k, cap; int64_t wslot_off, sslot_off; };
    std::vector<Grp> grps;
    if (im.rows_mode) {
        for (const auto & rg : im.rgroups) grps.push_back(Grp { rg.slot, rg.c, rg.k, r ? rg.cap[r] : 0, rg.c, 2 * rg.c });
    } else {
        grps.push_back(Grp { im.cur_slot, col0, n, cap, 0, 0 });
    }

    // ---- window K (= V) projection (every row); per group: [ring | the group's rows] -> each query's SWA rows
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    // the compressor's projections (kv source layers), every row: ratio 1 = the latent itself, ratio 2 = the state rows
    ggml_tensor * c_lat = nullptr, * c_skv = nullptr, * c_ssc = nullptr;
    // the indexer's query / weights (index source layers), every row
    ggml_tensor * x_iq = nullptr, * x_iw = nullptr;
    std::vector<ggml_tensor *> o_parts;
    if (im.rows_mode && im.slab) {   // the slabs: one write / gather / flash-attn for every row (attn_core_slab)
        o_parts.push_back(attn_core_slab(im, gc, b, il, n, xn, qr, q, kv2, pos, outs, late));
        grps.clear();
    }
    // a batch pass with several groups: their key sets (window + top-k rows, each padded to the longest) go into ONE
    // flash-attn over every row - per query its own keys either way, so only the -inf padding is added (STRATA_ROWS_FA=0:
    // one flash-attn per group)
    static const bool rows_fa = !(std::getenv("STRATA_ROWS_FA") && std::getenv("STRATA_ROWS_FA")[0] == '0');
    const bool merge_fa = rows_fa && im.rows_mode && grps.size() > 1;
    struct KPart { ggml_tensor * K, * M; int64_t k; bool f16; };
    std::vector<KPart> k_parts;
    for (const Grp & gp : grps) {
        const bool whole = grps.size() == 1 && gp.k == n;   // the single-sequence graph, exactly as before
        const int64_t gk = gp.k, gc0 = gp.c, rel = gp.c - col0;   // rel: the group's first column among this graph's n
        const Ds41Dense::Impl::SlotLayer SL = im.slot_layer(gp.slot, (size_t) il);
        auto colv = [&](ggml_tensor * t2) {   // the group's columns of a [ne0, n] tensor of this graph
            return whole ? t2 : ggml_cont(gc, ggml_view_2d(gc, t2, t2->ne[0], gk, t2->nb[1], (size_t) rel * t2->nb[1]));
        };
        ggml_tensor * gkv2 = colv(kv2);
        ggml_tensor * wsrc = ggml_concat(gc, SL.raw->type == GGML_TYPE_F32 ? SL.raw : ggml_cast(gc, SL.raw, GGML_TYPE_F32), gkv2, 1);     // [DH, RAW + k]
        // F16 key path for this graph only when the zero index covers its padding (255 pad rows x k queries at most)
        const bool f16 = im.f16_gather && 255 * gk <= im.zidx->ne[0];
        ggml_tensor * Kw = ggml_reshape_3d(gc, rows_as(gc, wsrc, b.v1(im.i_widx, SWA * gk, (size_t) gc0 * SWA * 4), f16), DH, SWA, gk);
        ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, gk, gc0);
        {
            const int64_t k = std::min<int64_t>(gk, RAW);
            ggml_tensor * last = ggml_view_2d(gc, gkv2, DH, k, gkv2->nb[1], (size_t) (gk - k) * gkv2->nb[1]);
            late.push_back(ggml_set_rows(gc, SL.raw, last, b.v1(im.i_wslot, k, (size_t) gp.wslot_off * 4)));
        }

        // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
        ggml_tensor * Kc = nullptr, * Mc = nullptr;
        ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this group's writes (later reads depend on them)
        const int64_t gcap = gp.cap;
        if (r) {
            if (L.kv_src) {
                ggml_tensor * latent = nullptr;                                                  // [DH, k]
                if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                    if (!c_lat) c_lat = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                    latent = colv(c_lat);
                } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                    if (!c_skv) {
                        c_skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                        c_ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                        ggml_mul_mat_set_prec(c_skv, GGML_PREC_F32);
                        ggml_mul_mat_set_prec(c_ssc, GGML_PREC_F32);
                    }
                    ggml_tensor * skv = colv(c_skv), * ssc = colv(c_ssc);
                    ggml_tensor * src_kv = ggml_concat(gc, SL.st_kv, skv, 1);                     // [DH, 2 + k]
                    ggml_tensor * src_sc = ggml_concat(gc, SL.st_sc, ssc, 1);
                    ggml_tensor * gi = b.v1(im.i_sidx, 2 * gk, (size_t) gc0 * 2 * 4);
                    ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, gk);
                    ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, gk);
                    vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, k]
                    vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                    latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, gk);
                    const int64_t k = std::min<int64_t>(gk, 2);
                    late.push_back(ggml_set_rows(gc, SL.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (gk - k) * skv->nb[1]),
                                                 b.v1(im.i_sslot, k, (size_t) gp.sslot_off * 4)));
                    late.push_back(ggml_set_rows(gc, SL.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (gk - k) * ssc->nb[1]),
                                                 b.v1(im.i_sslot, k, (size_t) gp.sslot_off * 4)));
                }
                latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
                ggml_tensor * cpos = b.v1(im.i_cpos[r], gk, (size_t) gc0 * 4);
                ggml_tensor * cslot = b.v1(im.i_cslot[r], gk, (size_t) gc0 * 4);
                // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
                ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
                ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, gk), cpos, r), im.ID, gk);
                if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
                idxk_w = ggml_set_rows(gc, SL.idxk, ik, cslot);
                outs.push_back(idxk_w);
                ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, gk), cpos, r), DH, gk);
                if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
                comp_w = ggml_set_rows(gc, SL.comp, lr, cslot);
                outs.push_back(comp_w);
            }
            if (gcap > 0) {
                const Ds41Dense::Impl::SlotLayer KS = im.slot_layer(gp.slot, (size_t) L.kvl);
                const int64_t K = std::min<int64_t>(gcap, im.ITOP);
                ggml_tensor * comp = comp_w ? comp_w : KS.comp;
                ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
                ggml_tensor * sel = nullptr, * selm = nullptr;
                if (L.idx_src) {
                    // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                    if (!x_iq) {
                        x_iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                        x_iq = b.rope(ggml_reshape_3d(gc, x_iq, im.ID, im.IH, n), pos, r);
                        if (im.cfg.act_quant) x_iq = fake_quant(gc, ggml_cont(gc, x_iq), Ds41Quant::FP4_POW2_32);
                        x_iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                        x_iw = ggml_scale(gc, x_iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                    }
                    ggml_tensor * iq = whole ? x_iq : ggml_cont(gc, ggml_view_3d(gc, x_iq, im.ID, im.IH, gk, x_iq->nb[1], x_iq->nb[2], (size_t) rel * x_iq->nb[2]));
                    ggml_tensor * iw = colv(x_iw);
                    ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, gcap, idxk->nb[1], idxk->nb[1], 0);
                    // The selection of the group's queries [q0, q0 + nq).  Its score / mask tensors are [cap, nq]: with
                    // cap = the visible compressed rows (a power of two) they grow with the context - a 1M context's
                    // ratio-1 layers hold ~18 bytes x 2^20 rows x 512 queries = a 9.3 GB graph buffer (2026-10-20: the
                    // 768K / 1M prompts' last batch, the first to run layers 21-39 past 2^19 rows, OOM'd on it).  Every
                    // op here is per query (the indexer kernel's grid is [kv blocks, queries], top-k / argsort / get_rows
                    // are per column), so the span's queries are taken in chunks of at most idx_tile score elements, each
                    // chunk's tensors dying before the next is built.  Up to idx_tile (default: a 512K context) the span
                    // is ONE chunk and the graph is the very one it always was.
                    auto select = [&](int64_t q0, int64_t nq, ggml_tensor ** sel_out, ggml_tensor ** selm_out) {
                        const bool wq = q0 == 0 && nq == gk;
                        const size_t c0q = (size_t) (gc0 + q0);
                        ggml_tensor * iq_c = wq ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                        ggml_tensor * iw_c = wq ? iw : ggml_view_2d(gc, iw, im.IH, nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                        // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                        ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, c0q * 4), 1, nq);
                        ggml_tensor * vis = nullptr;
                        ggml_tensor * sc = b.indexer(iq_c, ksrc, iw_c, nvs, il, &vis);   // [cap, nq]
                        sc = ggml_reshape_2d(gc, sc, gcap, nq);
                        if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                            ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                            m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                            m = ggml_view_2d(gc, m, gcap, nq, m->nb[2], 0);
                            sc = ggml_add(gc, sc, m);
                        }
                        if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            GGML_ASSERT(gcap % im.CBS == 0);
                            ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, nq]
                            // pin: + 1e30 at block (nvis - 1) / CBS of each query
                            ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                            ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, c0q * 4), 1, nq);
                            ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                            bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                            ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, nq]
                            outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                        }
                        ggml_tensor * s1 = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                          // I32 [K, nq]
                        // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                        // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                        // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                        // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                        // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                        // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                        s1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s1, 1, K, nq),
                                                               ggml_argsort(gc, ggml_cast(gc, s1, GGML_TYPE_F32),
                                                                            GGML_SORT_ORDER_ASC)), K, nq);
                        ggml_tensor * m1 = vis ? ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, gcap, nq), s1), K, nq)
                                              : ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_scale(gc, ggml_cast(gc, s1, GGML_TYPE_F32), -1.0f), nvs)));   // vis at sel
                        // the chunk's writes go to outs now, so the graph runs (and frees) this chunk before the next
                        outs.push_back(ggml_cpy(gc, s1, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], c0q * L.sel->nb[1])));
                        outs.push_back(ggml_cpy(gc, m1, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], c0q * L.selm->nb[1])));
                        *sel_out = s1;
                        *selm_out = m1;
                    };
                    const int64_t qc = im.idx_tile > 0 && gcap * gk > im.idx_tile ? std::max<int64_t>(1, im.idx_tile / gcap) : gk;
                    for (int64_t q0 = 0; q0 < gk; q0 += qc) {
                        ggml_tensor * s1 = nullptr, * m1 = nullptr;
                        select(q0, std::min(qc, gk - q0), &s1, &m1);
                        sel = sel ? ggml_concat(gc, sel, s1, 1) : s1;      // [K, k] for the gather below
                        selm = selm ? ggml_concat(gc, selm, m1, 1) : m1;
                    }
                } else {
                    const L_t & IS = im.ly[(size_t) L.idxl];
                    sel = ggml_view_2d(gc, IS.sel, K, gk, IS.sel->nb[1], (size_t) gc0 * IS.sel->nb[1]);
                    selm = ggml_view_2d(gc, IS.selm, K, gk, IS.selm->nb[1], (size_t) gc0 * IS.selm->nb[1]);
                }
                Kc = ggml_reshape_3d(gc, rows_as(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * gk)), f16), DH, K, gk);
                Mc = ggml_cont(gc, selm);
                // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
            }
        }

        // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
        ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, k]
        ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, k]
        if (merge_fa) { k_parts.push_back(KPart { Kall, Mall, gk, f16 }); continue; }
        int64_t M = Kall->ne[1];
        if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
            const int64_t pad = 256 - M % 256;
            // F16 has no CUDA pad: concat zeros gathered from the one zero row (F16 -> F16, exact) - the same zeros.
            // The F32 pad kernel clamps its grid at 65,535 rows (ggml-cuda/pad.cu: min(ne1, 65535)), so it leaves
            // every row past that unwritten: a prompt whose visible key count crosses 65,535 would attend to
            // uninitialised device memory.  Past the limit build the pad rows explicitly too - the same zeros, no
            // pad kernel.  (mab776's 251k-token 1017 run hit this on the other pad site; same class of bug.)
            Kall = (f16 || M + pad > 65535)
                 ? ggml_concat(gc, Kall, f16
                       ? ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * gk, 0), true), DH, pad, gk)
                       : ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, DH, pad, gk), 0.0f), 1)
                 : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
            Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, gk), NEG_INF), 0);
            M += pad;
        }
        ggml_tensor * gq = whole ? q : ggml_cont(gc, ggml_view_3d(gc, q, DH, NH, gk, q->nb[1], q->nb[2], (size_t) rel * q->nb[2]));
        ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, gk)
                                                       : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, gk), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, gq, DH, NH, 1, gk), 0, 2, 1, 3);    // [DH, 1, NH, k]
        ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, gk), GGML_TYPE_F16);
        ggml_tensor * og = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(og, im.BL(il, "attn_sinks.weight"));
        o_parts.push_back(ggml_reshape_3d(gc, og, DH, NH, gk));
    }
    if (merge_fa) {
        int64_t Mx = 0;
        for (const KPart & kp : k_parts) Mx = std::max<int64_t>(Mx, kp.K->ne[1]);
        if (!im.cpu && Mx % 256) Mx += 256 - Mx % 256;   // ggml-cuda's flash-attn for d 512: a multiple of 256 keys
        ggml_tensor * Kx = nullptr, * Mxk = nullptr;
        for (const KPart & kp : k_parts) {
            ggml_tensor * K = kp.K, * Mk = kp.M;
            const int64_t pad = Mx - K->ne[1];
            if (pad > 0) {
                K = (kp.f16 || Mx > 65535)   // the pad kernel's 65,535-row grid clamp - see the site above
                    ? ggml_concat(gc, K, kp.f16
                          ? ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * kp.k, 0), true), DH, pad, kp.k)
                          : ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, DH, pad, kp.k), 0.0f), 1)
                    : ggml_pad(gc, K, 0, (int) pad, 0, 0);
                Mk = ggml_concat(gc, Mk, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, kp.k), NEG_INF), 0);
            }
            if (Kx && K->type != Kx->type) K = ggml_cast(gc, K, Kx->type);
            Kx = Kx ? ggml_concat(gc, Kx, K, 2) : K;                                               // [DH, Mx, n]
            Mxk = Mxk ? ggml_concat(gc, Mxk, Mk, 1) : Mk;                                          // [Mx, n]
        }
        ggml_tensor * k4 = Kx->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kx, DH, Mx, 1, n)
                                                     : ggml_cast(gc, ggml_reshape_4d(gc, Kx, DH, Mx, 1, n), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);      // [DH, 1, NH, n]
        ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mxk, Mx, 1, 1, n), GGML_TYPE_F16);
        ggml_tensor * og = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(og, im.BL(il, "attn_sinks.weight"));
        o_parts.push_back(ggml_reshape_3d(gc, og, DH, NH, n));
    }
    ggml_tensor * o = o_parts[0];
    for (size_t i = 1; i < o_parts.size(); ++i) o = ggml_concat(gc, o, o_parts[i], 2);         // [DH, NH, n]
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);              // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);                       // [D, HC, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
               &f_pre, &f_post, &f_comb);
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));   // [D, n]

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
#if defined(STRATA_DSV41_ROUTE_OP)
    // -DSTRATA_DSV41_ROUTE_OP=ON (the ops are in tools/xolotl/ggml/xolotl-ggml.patch): the flat-bias router
    // below as two ops (GGML_OP_DSV41_ROUTE_SCORES / _WEIGHTS), the same steps bit for bit (test_ds41_route) - 12 fewer
    // tiny kernels per layer with --vision, 8 without.  The smart / margin selections keep the composite.
    // STRATA_ROUTE_COMPOSITE=1: the composite.
    static const bool route_composite = std::getenv("STRATA_ROUTE_COMPOSITE") != nullptr;
    const bool route_op = !route_composite && !smart_router() && rb_margin < 0.0f;
#else
    const bool route_op = false;
#endif
    ggml_tensor * sc = nullptr, * route_sel = nullptr;
#if defined(STRATA_DSV41_ROUTE_OP)
    if (route_op) {
        const bool vl = im.vision && im.i_img;
        ggml_tensor * rs = ggml_dsv41_route_scores(gc, rl, im.BL(il, "exp_probs_b.bias"),
                                                   vl ? im.BL(il, "exp_probs_b_vl.bias") : nullptr,
                                                   vl ? b.v1(im.i_img, n, (size_t) col0 * 4) : nullptr, im.ly[(size_t) il].rbias);
        sc = ggml_view_2d(gc, rs, im.NE, n, rs->nb[1], 0);                                     // [NE, n]
        route_sel = ggml_view_2d(gc, rs, im.NE, n, rs->nb[1], rs->nb[2]);
    }
#endif
    if (!route_op) sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                 // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    // vision: image positions select with exp_probs_b_vl.bias instead (model.py Gate: torch.where(image_mask, bias_vl, bias))
    ggml_tensor * bias_sel = im.BL(il, "exp_probs_b.bias");
    if (!route_op && im.vision && im.i_img) {
        ggml_tensor * bt = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b.bias"), im.NE, 1);
        ggml_tensor * bv = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b_vl.bias"), im.NE, 1);
        ggml_tensor * m = ggml_reshape_2d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, n);
        bias_sel = ggml_add(gc, ggml_mul(gc, ggml_repeat(gc, ggml_sub(gc, bv, bt), sc), m), bt);
    }
    ggml_tensor * s0 = route_op ? nullptr : ggml_add(gc, sc, bias_sel);                        // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    //   STRATA_RB_POOL=M     the bias only moves selections inside the exact top-M; every pick stays a top-M expert
    //                        (colibri's CACHE_ROUTE max-rank window, arXiv:2412.00099: PROTECT=2 POOL=12 there)
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    // PROTECT / POOL (and STRATA_RB_SMART=1) build them from rknob: P / M then change per request (set_route_select),
    // 0 = off, and P = M = 0 selects exactly as the flat bias.
    ggml_tensor * xids = nullptr;   // the exact top-NU (route meter)
    if (smart_router() || rb_margin >= 0.0f) {
        const int kt = smart_router() ? kRK : (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr, * pool = nullptr;
        if (smart_router()) {
            xids = ggml_cont(gc, ggml_view_2d(gc, ti, im.NU, n, ti->nb[1], 0));
            // rknob column c = a one-hot over the kRK exact ranks + an offset: the P-th / M-th exact score, or +-1e30 for
            // rank 0 (protect nothing / pool everything).  mul + sum_rows, not mul_mat: one nonzero term, exact in F32.
            auto thr = [&](int c) {
                ggml_tensor * oh = ggml_view_2d(gc, im.rknob, kRK, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1]);
                ggml_tensor * off = ggml_view_2d(gc, im.rknob, 1, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1] + (size_t) kRK * 4);
                return ggml_repeat(gc, ggml_add(gc, ggml_sum_rows(gc, ggml_mul(gc, tv, oh)), off), like);
            };
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(0)), 1.0f, 1e-6f));   // s0 >= the P-th exact score
            pool = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(1)), 1.0f, 1e-6f));      // s0 >= the M-th
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        if (pool) unprot = ggml_mul(gc, unprot, pool);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
        if (pool) sel = ggml_add(gc, sel, ggml_scale_bias(gc, pool, 1e4f, -1e4f));   // outside the pool: -1e4 (pool = 1: exactly 0)
    } else {
        sel = route_op ? route_sel : ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
    ggml_tensor * wts = nullptr;
#if defined(STRATA_DSV41_ROUTE_OP)
    if (route_op) wts = ggml_dsv41_route_weights(gc, sc, ids, G.norm_topk, (float) G.weights_scale);
#endif
    if (!wts) {
        wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
        if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
        wts = ggml_scale(gc, wts, (float) G.weights_scale);
    }

    // ---- shared expert (model.py Expert, clamped SwiGLU)
    const float lim = (float) G.swiglu_limit;
    ggml_tensor * up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    ggml_tensor * se = ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));

    // the layer's state writes.  -DSTRATA_DSV41_COPIES_OP=ON (the op is in
    // tools/xolotl/ggml/xolotl-ggml.patch): batched into GGML_OP_DSV41_COPIES (up to 5 per launch, the same bytes; they stay
    // where the ggml_cpy's were, after every read of the buffers).  STRATA_COPIES_SPLIT=1: one ggml_cpy each.
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> cps;
    cps.emplace_back(h, b.cols3(im.hap, D, HC, n, col0));
    cps.emplace_back(f_post, b.cols2(im.post_f, HC, n, col0));
    cps.emplace_back(ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0));
    cps.emplace_back(se, b.cols2(im.shexp, D, n, col0));
    cps.emplace_back(fn, b.cols2(im.o_fn, D, n, col0));
    cps.emplace_back(ids, b.cols2(im.o_ids, im.NU, n, col0));
    cps.emplace_back(wts, b.cols2(im.o_wts, im.NU, n, col0));
    if (xids) {   // route meter: the exact top-NU and their normalized weights
        ggml_tensor * xw = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), xids), im.NU, n);
        xw = ggml_div(gc, xw, ggml_scale_bias(gc, ggml_sum_rows(gc, xw), 1.0f, 1e-20f));
        cps.emplace_back(xids, b.cols2(im.o_xids, im.NU, n, col0));
        cps.emplace_back(xw, b.cols2(im.o_xwts, im.NU, n, col0));
    }
    cps.emplace_back(f_pre, b.cols2(im.pre_state, HC, n, col0));   // reads of pre_state happen above
#if defined(STRATA_DSV41_COPIES_OP)
    static const bool copies_split = std::getenv("STRATA_COPIES_SPLIT") != nullptr;
    if (!copies_split) {
        for (size_t i = 0; i < cps.size(); i += 5) {
            ggml_tensor * src[5], * dst[5];
            const int k = (int) std::min<size_t>(5, cps.size() - i);
            for (int j = 0; j < k; ++j) { src[j] = cps[i + (size_t) j].first; dst[j] = cps[i + (size_t) j].second; }
            outs.push_back(ggml_dsv41_copies(gc, src, dst, k));
        }
        cps.clear();
    }
#endif
    for (const auto & cp : cps) outs.push_back(ggml_cpy(gc, cp.first, cp.second));
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        // The folded upstream gate uses the reference multiplication order and sign(0)=+1;
        // keep the older separate q/k graph's rounding unchanged.
        const bool folded_gate = im.w.has("blk." + std::to_string(il) + ".engram_gate.weight");
        ggml_tensor * wqk = im.engram_wqk(gc, il);
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, folded_gate ? ggml_mul(gc, ggml_mul(gc, hn, kn), wqk) : ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * sign = folded_gate ? ggml_scale_bias(gc, ggml_step(gc, ggml_neg(gc, dot)), -2.0f, 1.0f) : ggml_sgn(gc, dot);
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, sign, mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));
        }
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    // the layers after next_il on the same state (speculative reads): their scores are stacked under next_il's so ONE
    // argsort gives all the rows and ONE readback fetches them ([16, 1 + ahead] I32)
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        if (il >= (int) im.ly.size()) break;
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, ra)), im.BL(il, "exp_probs_b.bias")),
                                    ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
    }
    *pout = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== cached pass graphs (1 < n <= kPassGraphN)

static constexpr int64_t kPassGraphN = 8;
static constexpr size_t kPassGraphCtx = (size_t) 256 << 20;

/// DS41_PASS_GRAPH_MIB (default 256): the arena the cached pass graphs are built in.  This - not a graph count -
/// is what "pass graph cache full" means: when the next graph does not fit, the cache is dropped and rebuilt.
/// A box whose launch overhead dominates (Volta sm_70, #60) can trade memory for fewer rebuilds.
static size_t pass_graph_ctx_bytes() {
    static const size_t v = [] {
        const char * e = std::getenv("DS41_PASS_GRAPH_MIB");
        const long mib = e ? std::atol(e) : 0;
        return (size_t) (mib > 0 ? mib : 256) << 20;
    }();
    return v;
}

static bool pass_graphs_on(int64_t n) {
    static const bool on = env_on("DS41_PASS_GRAPHS", true);
    return on && n > 1 && n <= kPassGraphN;
}

template <class F>
static ggml_cgraph * pass_graph(Ds41Dense::Impl & im, const Ds41Dense::Impl::GKey & key, F build) {
    auto it = im.mgraphs.find(key);
    if (it != im.mgraphs.end()) return it->second;
    // worst case of one graph: 8192 nodes' tensors + the graph struct
    const size_t need = ggml_tensor_overhead() * 8192 * 2 + ggml_graph_overhead_custom(8192, false);
    if (!im.gctx_m || ggml_get_mem_size(im.gctx_m) - ggml_used_mem(im.gctx_m) < need) {
        if (im.gctx_m) {
            for (const auto & kv : im.mgraphs) im.allo_dec.graph_epoch.erase(kv.second);
            std::fprintf(stderr, "ds41_dense: pass graph cache full (%zu graphs) - dropped\n", im.mgraphs.size());
            im.mgraphs.clear();
            ggml_free(im.gctx_m);
            ++im.mgraph_resets;
        }
        ggml_init_params gp = { pass_graph_ctx_bytes(), nullptr, true };
        im.gctx_m = ggml_init(gp);
    }
    return im.mgraphs.emplace(key, build(im.gctx_m)).first->second;
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }
bool Ds41Dense::vision() const { return p_->vision; }

void Ds41Dense::set_vision(const float * rows, const int64_t * pos, int n_rows) {
    Impl & im = *p_;
    im.vis_rows = rows;
    im.vis_pos = pos;
    im.n_vis = n_rows;
}

void Ds41Dense::clear_vision() {
    Impl & im = *p_;
    im.vis_rows = nullptr;
    im.vis_pos = nullptr;
    im.n_vis = 0;
}

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    if (smart_router()) {
        im.o_xids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
        im.o_xwts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    }
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        if (!L.idx_src) continue;
        L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC);
        L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC);
    }
    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_img = im.vision ? ggml_new_tensor_1d(c, GGML_TYPE_F32, NC) : nullptr;   // vision: the image-position mask
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    // a batch pass (begin_rows) keeps each group's 2 state slots at 2 x its first column
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::max<int64_t>(std::min<int64_t>(NC, 2), 2 * std::min<int64_t>(NC, Ds41Dense::kMaxRows)));
    im.i_sscol = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * std::min<int64_t>(NC, Ds41Dense::kMaxRows));
    for (int r = 1; r <= 2; ++r) im.i_cbase[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    for (int e = 0; e < 2; ++e) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    ggml_backend_buffer_clear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    ggml_backend_synchronize(im.backend);
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.reset_graphs();
    return alloc_cols(im, want, im.err);
}

void Ds41Dense::Impl::reset_graphs() {
    Impl & im = *this;
    im.graphs.clear();
    for (const auto & kv : im.mgraphs) im.allo_dec.graph_epoch.erase(kv.second);
    im.mgraphs.clear();
    if (im.gctx_m) { ggml_free(im.gctx_m); im.gctx_m = nullptr; }
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        al->failed = false;
        ++al->epoch;
    }
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
}

// Room for one more cached layer graph (a layer's attention graph is ~1-2K tensors; 8192 is the build_attn graph's
// node cap): with several slots each slot caches its own decode graphs, so the context can fill where one slot never
// did - then every cached graph is dropped and rebuilt lazily (as set_columns does) instead of overflowing it.
bool Ds41Dense::Impl::gctx_room() const {
    if (!gctx) return false;
    const size_t need = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false) * 2;
    return ggml_used_mem(gctx) + need <= ggml_get_mem_size(gctx);
}

void Ds41Dense::Impl::stash_slot() {
    Slot & S = slots[(size_t) cur_slot];
    S.ly.resize(ly.size());
    for (size_t l = 0; l < ly.size(); ++l) {
        const Layer & L = ly[l];
        S.ly[l] = SlotLayer { L.raw, L.raw_snap, L.st_kv, L.st_sc, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
    }
    S.snap_pos = snap_pos;
    S.parked_pos = parked_pos;
    S.parked = std::move(parked);
    parked.clear();
}

void Ds41Dense::Impl::load_slot(int s) {
    Slot & S = slots[(size_t) s];
    for (size_t l = 0; l < ly.size(); ++l) {
        Layer & L = ly[l];
        const SlotLayer & T = S.ly[l];
        L.raw = T.raw; L.raw_snap = T.raw_snap; L.st_kv = T.st_kv; L.st_sc = T.st_sc;
        L.st_kv_snap = T.st_kv_snap; L.st_sc_snap = T.st_sc_snap; L.comp = T.comp; L.idxk = T.idxk;
    }
    snap_pos = S.snap_pos;
    parked_pos = S.parked_pos;
    parked = std::move(S.parked);
    S.parked.clear();
    cur_slot = s;
    engram.select_history(s);
}

int Ds41Dense::n_slots() const { return (int) p_->slots.size(); }
int Ds41Dense::slot() const { return p_->cur_slot; }

bool Ds41Dense::bind_slot(int s) {
    Impl & im = *p_;
    if (s < 0 || s >= (int) im.slots.size()) { im.err = "bind_slot: no slot " + std::to_string(s); return false; }
    if (s == im.cur_slot) return true;
    if (im.eng_batch) { im.err = "bind_slot inside a pass"; return false; }
    im.stash_slot();
    im.load_slot(s);
    return true;
}

bool Ds41Dense::swap_slots(int a, int b) {
    Impl & im = *p_;
    const int n = (int) im.slots.size();
    if (a < 0 || b < 0 || a >= n || b >= n) { im.err = "swap_slots: bad slot"; return false; }
    if (a == b) return true;
    if (im.eng_batch) { im.err = "swap_slots inside a pass"; return false; }
    ggml_backend_synchronize(im.backend);
    const int cur = im.cur_slot;
    im.stash_slot();
    std::swap(im.slots[(size_t) a], im.slots[(size_t) b]);
    im.engram.swap_histories(a, b);   // the selected history (cur's) is swapped in place too
    im.load_slot(cur);
    im.reset_graphs();   // the cached attention graphs of slots a / b hold the other state's tensors now
    return true;
}

size_t Ds41Dense::slot_bytes() const {
    const Impl & im = *p_;
    size_t b = 0;
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const Impl::SlotLayer T = im.slot_layer(im.cur_slot, l);
        for (ggml_tensor * t : { T.raw, T.raw_snap, T.st_kv, T.st_sc, T.st_kv_snap, T.st_sc_snap, T.comp, T.idxk })
            if (t) b += ggml_nbytes(t);
    }
    return b;
}

int Ds41Dense::columns() const { return (int) p_->NC; }

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    // gate-only: run the first n layers (a faster CPU gate of the same code; the logits are not the model's then)
    if (const char * e = std::getenv("DS41_TEST_LAYERS"); e && std::atoi(e) > 0) im.nl = std::min<int64_t>(im.nl, std::atoi(e));
    if (cfg.head_only) {
        im.layer_lo = im.nl;
        im.layer_hi = im.nl;
    } else {
        im.layer_lo = std::max<int64_t>(0, cfg.layer_lo);
        im.layer_hi = cfg.layer_hi < 0 ? im.nl : std::min<int64_t>(cfg.layer_hi, im.nl);
        if (im.layer_lo >= im.layer_hi) { err = "invalid layer range"; return false; }
    }
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    // STRATA_INDEXER_TILE=<elements>: the indexer selection's per-graph piece (see build_attn); 0 = never split
    im.idx_tile = std::getenv("STRATA_INDEXER_TILE") ? std::max<long long>(0, std::atoll(std::getenv("STRATA_INDEXER_TILE")))
                                                     : (int64_t) 1 << 28;
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (!load_weights(im, err)) return false;
    im.engram.set_hot_path(cfg.engram_hot);
    im.vision = cfg.vision;
    if (im.vision) {
        for (int64_t l = 0; l < im.nl; ++l) {
            if (!im.w.has("blk." + std::to_string(l) + ".exp_probs_b_vl.bias")) {
                err = "--vision requires exp_probs_b_vl.bias in every layer";
                return false;
            }
        }
    }
    im.engram.set_hot_path(cfg.engram_hot);
    if (!cfg.head_only && !cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, cfg.engram_threads, err)) return false;

    im.shadow_layers = cfg.shadow_layers;
    // layers
    im.ly.resize(cfg.head_only ? 0 : (size_t) im.nl);
    for (int64_t l = 0; !cfg.head_only && l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        if (!im.active_layer(l)) continue;
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state.  A layer split (buster): a stage holds state only for the layers it runs or shadows
    // (active_layer); a head_only instance (output head + DSpark on its own card) holds none.
    if (!cfg.head_only && cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    constexpr int kSlotsMax = 64;
    int NS = cfg.head_only ? 1 : cfg.n_slots > 0 ? std::min(cfg.n_slots, kSlotsMax) : 8;   // <= 0: auto (below), at most 8
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (512 + 8 * im.nl * NS), nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
    static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
    const ggml_type comp_t = cfg.kv_fp4 && !(only && std::string(only) == "idx") ? GGML_TYPE_NVFP4 : kvt;
    const ggml_type idxk_t = cfg.kv_fp4 && !(only && std::string(only) == "comp") ? GGML_TYPE_MXFP4 : kvt;
    // one slot's state: the window ring + its snapshot, and on kv sources the compressed KV + indexer keys (+ the ratio-2
    // compressor state and its snapshot)
    size_t slot0 = 0;
    for (int64_t l = 0; l < im.nl; ++l) {
        const auto & L = im.ly[(size_t) l];
        if (cfg.head_only || !im.active_layer(l)) continue;
        slot0 += 2 * ggml_row_size(kvt, im.DH) * (size_t) im.RAW;
        if (!L.kv_src) continue;
        slot0 += (ggml_row_size(comp_t, im.DH) + ggml_row_size(idxk_t, im.ID)) * (size_t) (im.cmax[L.ratio] + 1);
        if (L.ratio == 2) slot0 += 4 * ggml_row_size(GGML_TYPE_F32, im.DH) * 2;
    }
    if (cfg.n_slots <= 0 && !cfg.head_only) {   // auto: as many as STRATA_BATCH_SLOTS_GIB (default 2 GiB) holds, 2..8
        const double gib = std::getenv("STRATA_BATCH_SLOTS_GIB") ? std::atof(std::getenv("STRATA_BATCH_SLOTS_GIB")) : 2.0;
        NS = (int) std::max<int64_t>(2, std::min<int64_t>(8, (int64_t) (gib * 1073741824.0 / (double) std::max<size_t>(1, slot0))));
    }
    // the slab: a batch pass gathers compressed rows by slab row = region * (cmax + 1) + row, computed in F32 - exact
    // below 2^24 rows (8 slots of 1M)
    im.slab = NS > 1 && !(std::getenv("STRATA_SLOT_SLAB") && std::getenv("STRATA_SLOT_SLAB")[0] == '0') &&
              (int64_t) NS * (im.cmax[1] + 1) < (int64_t) 1 << 24;
    if (NS > 1)
        std::fprintf(stderr, "batch slots: %d x %.1f MiB of conversation state (ctx %lld each) = %.2f GiB%s\n", NS,
                     slot0 / 1048576.0, (long long) cfg.ctx_max, (double) slot0 * NS / 1073741824.0, im.slab ? ", slabs" : "");
    im.slots.assign((size_t) NS, Impl::Slot());
    im.cur_slot = 0;
    for (int sl = 1; sl < NS; ++sl) im.slots[(size_t) sl].ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        if (cfg.head_only || !im.active_layer(l)) continue;   // another stage's layer: no state here
        if (!im.slab) {
            L.raw = T(kvt, { im.DH, im.RAW });
            L.raw_snap = T(kvt, { im.DH, im.RAW });
            L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
            if (L.kv_src) {
                L.comp = T(comp_t, { im.DH, im.cmax[L.ratio] + 1 });
                L.idxk = T(idxk_t, { im.ID, im.cmax[L.ratio] + 1 });
                if (L.ratio == 2) {
                    L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                }
            }
            continue;
        }
        L.rbias = T(GGML_TYPE_F32, { im.NE });
        const int64_t CR = im.cmax[L.ratio] + 1;
        L.raw_all = T(kvt, { im.DH, im.RAW * NS });
        if (L.kv_src) {
            L.comp_all = T(comp_t, { im.DH, CR * NS });
            L.idxk_all = T(idxk_t, { im.ID, CR * NS });
            if (L.ratio == 2) { L.st_kv_all = T(GGML_TYPE_F32, { im.DH, 2 * NS }); L.st_sc_all = T(GGML_TYPE_F32, { im.DH, 2 * NS }); }
        }
        auto rows = [&](ggml_tensor * all, int64_t per, int sl) {   // slot sl's region of a slab
            return all ? ggml_view_2d(im.sctx, all, all->ne[0], per, all->nb[1], (size_t) sl * (size_t) per * all->nb[1]) : nullptr;
        };
        for (int sl = 0; sl < NS; ++sl) {
            Impl::SlotLayer X;
            X.raw = rows(L.raw_all, im.RAW, sl);
            X.raw_snap = T(kvt, { im.DH, im.RAW });
            X.comp = rows(L.comp_all, CR, sl);
            X.idxk = rows(L.idxk_all, CR, sl);
            X.st_kv = rows(L.st_kv_all, 2, sl);
            X.st_sc = rows(L.st_sc_all, 2, sl);
            if (L.st_kv_all) { X.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 }); X.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 }); }
            if (sl == 0) {
                L.raw = X.raw; L.raw_snap = X.raw_snap; L.comp = X.comp; L.idxk = X.idxk;
                L.st_kv = X.st_kv; L.st_sc = X.st_sc; L.st_kv_snap = X.st_kv_snap; L.st_sc_snap = X.st_sc_snap;
            } else {
                im.slots[(size_t) sl].ly[(size_t) l] = X;
            }
        }
    }
    if (!cfg.head_only && smart_router()) im.rknob = T(GGML_TYPE_F32, { kRK + 1, 2 });
    // batch slots 1..NS-1 without a slab: the same per-conversation tensors again (slot 0 = the ones above)
    for (int sl = 1; sl < NS && !im.slab; ++sl) {
        auto & S = im.slots[(size_t) sl];
        for (int64_t l = 0; l < im.nl; ++l) {
            const auto & L = im.ly[(size_t) l];
            auto & X = S.ly[(size_t) l];
            auto like = [&](ggml_tensor * t) { return t ? T(t->type, { t->ne[0], t->ne[1] }) : nullptr; };
            X.raw = like(L.raw); X.raw_snap = like(L.raw_snap);
            X.comp = like(L.comp); X.idxk = like(L.idxk);
            X.st_kv = like(L.st_kv); X.st_sc = like(L.st_sc); X.st_kv_snap = like(L.st_kv_snap); X.st_sc_snap = like(L.st_sc_snap);
        }
    }
    if (!cfg.head_only) {   // (a head_only instance has no state tensors: an empty context allocates nothing)
        im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
        if (!im.sbuf) { err = "cannot allocate the decode state (" + std::to_string(NS) + " slots)"; return false; }
        ggml_backend_buffer_clear(im.sbuf, 0);
    }
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(L.st_sc, v.data(), 0, v.size() * 4); }
    for (int sl = 1; sl < NS; ++sl)
        for (auto & X : im.slots[(size_t) sl].ly)
            if (X.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(X.st_sc, v.data(), 0, v.size() * 4); }
    im.stash_slot();   // slots[0] = slot 0's tensors too (slot_layer / slot_bytes read every slot the same way)
    if (im.rknob) set_route_select(env_rank("STRATA_RB_PROTECT"), env_rank("STRATA_RB_POOL"));   // all-zero knobs would protect everything
    // inputs (skipped on head_only: only the pass columns hold x / pre for the logits)
    if (!cfg.head_only) {
        ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
        im.ictx = ggml_init(ipp);
        auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
        im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
        im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
        // default ON (CUDA) since 2026-10-18: 1017 8K prompt 239.8 -> 255.2 tok/s, +1.06 GiB min free VRAM, logits + tokens
        // identical (verify/s7/vram-f16.sh GATE A).  STRATA_ATTN_F16_GATHER=0 restores the F32 gather.
        im.f16_gather = !im.cpu && !(std::getenv("STRATA_ATTN_F16_GATHER") && std::getenv("STRATA_ATTN_F16_GATHER")[0] == '0');
        im.zidx = I(GGML_TYPE_I32, im.f16_gather ? 255 * 2048 : 1);   // ibuf is cleared to 0 below: all zeros
        if (im.f16_gather) im.zpad16 = I(GGML_TYPE_F16, im.DH, 1);
        im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
        if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
        ggml_backend_buffer_clear(im.ibuf, 0);
        {
            std::vector<float> v((size_t) im.neg_ramp->ne[0]);
            for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
            ggml_backend_tensor_set(im.neg_ramp, v.data(), 0, v.size() * 4);
            v.assign((size_t) im.blk_ramp->ne[0], 0);
            for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
            ggml_backend_tensor_set(im.blk_ramp, v.data(), 0, v.size() * 4);
        }
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!alloc_cols(im, im.NC, err)) return false;
    if (!cfg.head_only) {
        const int64_t D = im.D, NCM = im.NCmax;
        im.host_fn.resize((size_t) (D * NCM));
#ifdef DS41_HAVE_CUDA
        if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
        im.host_ids.resize((size_t) (im.NU * NCM));
        im.host_wts.resize((size_t) (im.NU * NCM));
        if (smart_router()) {
            im.host_xids.resize((size_t) (im.NU * NCM));
            im.host_xwts.resize((size_t) (im.NU * NCM));
        }
#ifdef DS41_HAVE_CUDA
        // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
        // pinned memory)
        if (!im.cpu) {
            ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
            ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
            if (!im.host_xids.empty()) {
                ggml_backend_cuda_register_host_buffer(im.host_xids.data(), im.host_xids.size() * 4);
                ggml_backend_cuda_register_host_buffer(im.host_xwts.data(), im.host_xwts.size() * 4);
            }
        }
#endif
    }
    return true;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    ggml_backend_tensor_set(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

bool Ds41Dense::set_route_select(int protect, int pool) {
    Impl & im = *p_;
    if (!im.rknob) return false;
    protect = std::max(0, std::min(protect, (int) im.NU));
    pool = pool <= 0 ? 0 : std::max((int) im.NU, std::min(pool, kRK));   // a pool smaller than top-k would force outsiders in
    std::vector<float> k((size_t) (kRK + 1) * 2, 0.0f);
    if (protect > 0) k[(size_t) protect - 1] = 1.0f; else k[kRK] = 1e30f;                          // column 0: protect
    if (pool > 0) k[(size_t) (kRK + 1 + pool - 1)] = 1.0f; else k[(size_t) (2 * kRK + 1)] = -1e30f;  // column 1: pool
    ggml_backend_tensor_set(im.rknob, k.data(), 0, k.size() * 4);
    return true;
}

Ds41Dense::RouteMeter Ds41Dense::route_meter(bool reset) {
    Impl & im = *p_;
    RouteMeter m;
    m.on = !im.host_xids.empty();
    m.slots = im.rm_slots;
    m.swaps = im.rm_swaps;
    m.weight_kept = im.rm_rows > 0 ? im.rm_kept / (double) im.rm_rows : 1.0;
    if (reset) { im.rm_slots = im.rm_swaps = im.rm_rows = 0; im.rm_kept = 0; }
    return m;
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::reset_span_width() { p_->span_max = 0; }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    im.allo_pass.failed = false;
    ++im.allo_pass.epoch;
    im.span_max = 0;   // the next prompt starts at full sub-batch width again (see attn_router_span)
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, im.CBS)), im.cmax[r]);
    }
}

size_t Ds41Dense::span_graph_bytes(int pos_end, int n) {
    Impl & im = *p_;
    n = (int) std::max<int64_t>(1, std::min<int64_t>(n, im.S));
    pos_end = (int) std::max<int64_t>(n, std::min<int64_t>(pos_end, im.cfg.ctx_max));
    int64_t caps[3];
    span_pass_caps(im, pos_end - n, n, caps);
    const std::array<int64_t, 3> key { caps[1], caps[2], n };
    auto it = im.graph_bytes_memo.find(key);
    if (it != im.graph_bytes_memo.end()) return it->second;
    // build every layer KIND's span graph (they differ in ratio, kv / index / candidate source, Engram) and let a
    // throwaway allocator plan it: ggml_gallocr_reserve_n_size sizes the buffer without allocating anything
    int64_t saved[3] = { im.pass_cap[0], im.pass_cap[1], im.pass_cap[2] };
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = caps[r];
    const bool saved_rows = im.rows_mode;   // a single-sequence span's graph (a batch pass before it left the flag set)
    im.rows_mode = false;
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    std::set<std::array<int64_t, 6>> kinds;
    size_t mx = 0;
    for (int64_t l = 0; l < im.nl; ++l) {
        const Impl::Layer & L = im.ly[(size_t) l];
        if (!kinds.insert({ L.ratio, L.kv_src, L.idx_src, L.cand_src, L.uses_cand, L.eslot >= 0 }).second) continue;
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_attn(im, gc, (int) l, n, 0);
        size_t sz = 0;
        ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, &sz);
        mx = std::max(mx, sz);
        ggml_free(gc);
    }
    ggml_gallocr_free(ga);
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = saved[r];
    im.rows_mode = saved_rows;
    im.graph_bytes_memo[key] = mx;
    return mx;
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
        ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    return true;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // embeddings (F16 rows on the mmap)
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        strata::v41::ds41_embd_row(im.embd, im.embd_type, D, tids[t], im.host_emb.data() + (size_t) (t * D));
    }
    // vision: overwrite the image span positions with the encoder's rows (model.py merge_image_embeddings) and mark
    // them, so the router picks exp_probs_b_vl there and Engram skips them.  vis_pos is ascending, the pass contiguous.
    std::vector<float> img;
    if (im.vision) {
        img.assign((size_t) n, 0.0f);
        for (int v = 0; v < im.n_vis; ++v) {
            const int64_t p = im.vis_pos[v];
            if (p < pos0 || p >= (int64_t) pos0 + n) continue;
            const int t = (int) (p - pos0);
            std::memcpy(im.host_emb.data() + (size_t) t * D, im.vis_rows + (size_t) v * D, (size_t) D * 4);
            img[(size_t) t] = 1.0f;
        }
        ggml_backend_tensor_set(im.i_img, img.data(), 0, (size_t) n * 4);
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    ggml_backend_tensor_set(im.i_pos, pos.data(), 0, (size_t) n * 4);
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    ggml_backend_tensor_set(im.i_widx, widx.data(), 0, widx.size() * 4);
    ggml_backend_tensor_set(im.i_wmask, wm.data(), 0, wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        ggml_backend_tensor_set(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            ggml_backend_tensor_set(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        ggml_backend_tensor_set(im.i_sidx, si.data(), 0, si.size() * 4);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) {
            if (!img.empty() && img[(size_t) t] > 0.0f) im.engram.set_dead(pos0 + t);   // an image span: n-grams never span it
            else im.engram.set_token(pos0 + t, tids[t]);
        }
        if (n <= 64) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    im.rows_mode = false;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    const bool cached = n == 1;
    const bool mcached = pass_graphs_on(n);
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 0, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 0, 0, 1 }, build_init(im, im.gctx, 1)).first;
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { -1, 0, 0, n }, [&](ggml_context * c) { return build_init(im, c, n); });
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_init(im, gc, n);
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    return ok;
}

// ================================================================== batch slots: a pass over rows of several slots

// The per-row inputs of a batch pass: as upload_inputs for each slot's group of rows (its own ring rows, its own
// [state ring | pass states], its own Engram history), the per-group ring / state slots at the group's first column.
bool Ds41Dense::Impl::upload_rows(const Ds41Dense::Row * rows, int n) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // i_widx / i_wslot / i_sidx / i_sslot are rewritten
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (rows[t].tid < 0 || rows[t].tid >= im.g.vocab) { im.err = "token id out of range"; return false; }
        strata::v41::ds41_embd_row(im.embd, im.embd_type, D, rows[t].tid, im.host_emb.data() + (size_t) (t * D));
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (im.vision && im.i_img) {   // text rows only
        std::vector<float> z((size_t) n, 0.0f);
        ggml_backend_tensor_set(im.i_img, z.data(), 0, (size_t) n * 4);
    }
    std::vector<int32_t> pos(n), widx((size_t) (SWA * n)), ws, si((size_t) (2 * n)), ss((size_t) (2 * n), 0), scol;
    std::vector<float> wm((size_t) (SWA * n));
    ws.assign((size_t) n, 0);
    // slabs (attn_core_slab): every index is a slab row - the ring region of the row's slot (its own rows are written to
    // the ring before the gather), the state region (or the pass's own states past 2 x n_slots), the compressed region
    const bool slab = im.slab;
    const int64_t NSS = (int64_t) im.slots.size();
    im.rows_nss = 0;
    for (const RowGroup & G : im.rgroups) {
        const int64_t p0 = G.pos0, k = G.k, rb = (int64_t) G.reg * RAW;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t t = G.c + i, p = p0 + i;
            pos[(size_t) t] = (int32_t) p;
            for (int64_t j = 0; j < SWA; ++j) {   // per query the SWA positions [p - SWA + 1, p]: [its slot's ring | its rows]
                const int64_t q = p - SWA + 1 + j;
                int32_t row = 0;
                float m = 0.0f;
                if (q < 0) m = NEG_INF;
                else if (slab) row = (int32_t) (rb + q % RAW);
                else if (q >= p0) row = (int32_t) (RAW + (q - p0));
                else row = (int32_t) (q % RAW);
                widx[(size_t) (t * SWA + j)] = row;
                wm[(size_t) (t * SWA + j)] = m;
            }
            const int64_t g0 = (p / 2) * 2;   // ratio-2 state rows of [its slot's state ring (2) | its pass states]
            auto srow = [&](int64_t q) {
                if (slab) return (int32_t) (q >= p0 ? 2 * NSS + G.c + (q - p0) : 2 * G.reg + q % 2);
                return (int32_t) (q >= p0 ? 2 + (q - p0) : q % 2);
            };
            si[(size_t) (2 * t)] = srow(g0);
            si[(size_t) (2 * t + 1)] = srow(p >= g0 + 1 ? g0 + 1 : g0);
            if (slab) ws[(size_t) t] = (int32_t) (rb + p % RAW);
        }
        const int64_t kw = std::min<int64_t>(k, RAW), ks = std::min<int64_t>(k, 2);
        if (slab) {
            for (int64_t j = 0; j < ks; ++j) {
                scol.push_back((int32_t) (G.c + k - ks + j));
                ss[(size_t) im.rows_nss++] = (int32_t) (2 * G.reg + (p0 + k - ks + j) % 2);
            }
            continue;
        }
        for (int64_t j = 0; j < kw; ++j) ws[(size_t) (G.c + j)] = (int32_t) ((p0 + k - kw + j) % RAW);
        for (int64_t j = 0; j < ks; ++j) ss[(size_t) (2 * G.c + j)] = (int32_t) ((p0 + k - ks + j) % 2);
    }
    if (slab && !scol.empty()) ggml_backend_tensor_set(im.i_sscol, scol.data(), 0, scol.size() * 4);
    ggml_backend_tensor_set(im.i_pos, pos.data(), 0, (size_t) n * 4);
    ggml_backend_tensor_set(im.i_widx, widx.data(), 0, widx.size() * 4);
    ggml_backend_tensor_set(im.i_wmask, wm.data(), 0, wm.size() * 4);
    ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, (size_t) std::min<int64_t>(n, im.i_wslot->ne[0]) * 4);
    ggml_backend_tensor_set(im.i_sidx, si.data(), 0, si.size() * 4);
    ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, (size_t) std::min<int64_t>(2 * n, im.i_sslot->ne[0]) * 4);
    for (int r = 1; r <= 2; ++r) {   // compressed rows per ratio: per position, as upload_inputs
        std::vector<float> nv(n), lb(n);
        std::vector<int32_t> cs(n), cp(n);
        std::vector<float> cb(slab ? n : 0);
        for (const RowGroup & G : im.rgroups)
            for (int64_t i = 0; i < G.k; ++i) {
                const int64_t t = G.c + i, p = pos[(size_t) t];
                const int64_t base = slab ? (int64_t) G.reg * (im.cmax[r] + 1) : 0;   // the slot's compressed region
                nv[t] = (float) ((p + 1) / r);
                const bool done = (p + 1) % r == 0;
                cs[t] = (int32_t) (base + (done ? p / r : im.cmax[r]));   // an open group writes the spare row
                cp[t] = (int32_t) ((p / r) * r);
                lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
                if (slab) cb[(size_t) t] = (float) base;
            }
        if (slab) ggml_backend_tensor_set(im.i_cbase[r], cb.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) ggml_backend_tensor_set(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
    }
    if (im.engram.enabled()) {   // each row hashes its own slot's history (written here first, as a solo pass does)
        if (!engram_ready(im)) return false;
        im.rows_slot.resize((size_t) n);
        im.rows_pos.resize((size_t) n);
        for (int t = 0; t < n; ++t) {
            im.engram.set_token_in(rows[t].slot, rows[t].pos, rows[t].tid);
            im.rows_slot[(size_t) t] = rows[t].slot;
            im.rows_pos[(size_t) t] = rows[t].pos;
        }
        if (!im.engram.rows_at(im.rows_slot.data(), im.rows_pos.data(), n, im.eng_rows, im.err)) return false;
        for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
            ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    }
    return true;
}

bool Ds41Dense::begin_rows(const Row * rows, int n) {
    Impl & im = *p_;
    if (n < 1 || n > kMaxRows || n > im.NC) { im.err = "begin_rows: 1.." + std::to_string(std::min<int64_t>(kMaxRows, im.NC)) + " rows"; return false; }
    if (im.eng_batch) { im.err = "begin_rows: a pass's Engram reads are pending"; return false; }
    im.rgroups.clear();
    for (int i = 0; i < n; ++i) {
        const Row & R = rows[i];
        if (R.slot < 0 || R.slot >= (int) im.slots.size() || R.pos < 0 || R.pos >= im.cfg.ctx_max) { im.err = "begin_rows: bad slot / position"; return false; }
        if (i > 0 && R.slot == rows[i - 1].slot) {
            if (R.pos != rows[i - 1].pos + 1) { im.err = "begin_rows: a slot's rows must be consecutive positions"; return false; }
            ++im.rgroups.back().k;
            continue;
        }
        for (const auto & G : im.rgroups)
            if (G.slot == R.slot) { im.err = "begin_rows: a slot's rows must be contiguous"; return false; }
        im.rgroups.push_back(Impl::RowGroup { R.slot, i, 1, R.pos, { 0, 0, 0 }, im.slot_region(R.slot) });
    }
    for (auto & G : im.rgroups) span_pass_caps(im, (int) G.pos0, (int) G.k, G.cap);
    im.rows_mode = true;
    im.rows_sig = 0;
    static const bool rows_cache = !(std::getenv("STRATA_ROWS_GRAPH_CACHE") && std::getenv("STRATA_ROWS_GRAPH_CACHE")[0] == '0');
    if (rows_cache) {
        std::vector<int64_t> sig;
        for (const auto & G : im.rgroups) { sig.push_back(G.slot); sig.push_back(G.c); sig.push_back(G.k); sig.push_back(G.cap[1]); sig.push_back(G.cap[2]); sig.push_back(G.reg); }
        uint64_t h = 1469598103934665603ull;
        for (int64_t v : sig) { h ^= (uint64_t) v; h *= 1099511628211ull; }
        int64_t key = (int64_t) (h & 0x3fffffffffffffffull) | 1;
        auto it = im.rows_sigs.find(key);
        if (it == im.rows_sigs.end()) im.rows_sigs.emplace(key, sig);
        else if (it->second != sig) key = 0;   // a collision: this layout runs uncached
        im.rows_sig = key;
    }
    im.pos0 = 0;
    im.n = n;
    im.pass_cap[0] = im.pass_cap[1] = im.pass_cap[2] = 0;   // per group (read_sel has nothing to show)
    if (!im.upload_rows(rows, n)) { im.rows_mode = false; return false; }
    // embeddings -> state, as begin()
    if (im.rows_sig) {
        const Impl::GKey key { -1, 10, 0, n, -1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();
            it = im.graphs.emplace(key, build_init(im, im.gctx, n)).first;
        }
        return im.compute(it->second, true);
    }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_init(im, gc, n);
    const bool ok = im.compute(gf, false);
    ggml_free(gc);
    return ok;
}

// ================================================================== the layer-major prompt batch

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    im.rows_mode = false;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_init(im, gc, sn, c0);
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
// A span is a whole sub-batch (col0 % S == 0) or, when a sub-batch's graph did not fit, a narrower piece of one
// (attn_router_span): a piece at an unaligned column gets the pass_cap of its own end position.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    if (col0 % im.S == 0) {
        const size_t si = (size_t) (col0 / im.S);
        if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
        im.pass_cap[0] = im.span_cap[si][0];
        im.pass_cap[1] = im.span_cap[si][1];
        im.pass_cap[2] = im.span_cap[si][2];
    } else {
        span_pass_caps(im, im.batch_pos0 + col0, n, im.pass_cap);
    }
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        ggml_backend_tensor_set(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        ggml_backend_tensor_set(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

// attn_router_span / attn_router_span_kv run their span as pieces of at most span_max columns (the whole span until a
// graph does not fit).  A piece whose GRAPH ALLOCATION fails - nothing has run, the state is untouched - is retried as
// halves down to kMinSpan columns, and span_max keeps the width that fitted for the rest of the prompt, so a tight
// card (a long context's graph, another process taking VRAM mid-prompt) makes the prompt a little slower instead of
// failing it.  The pieces walk the columns in order, as the sub-batches do, so every position-indexed state (window
// ring, compressor state, compressed rows, candidate blocks) advances exactly as before; the per-query arithmetic is
// the same, but the dense matmuls see fewer columns, so a split prompt is not bit-identical to an unsplit one (it only
// happens where the unsplit one would have failed).
static constexpr int kMinSpan = 16;
static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool run_spans(Ds41Dense::Impl & im, int il, int col0, int n, const std::function<bool(int, int)> & one) {
    int64_t w = im.span_max > 0 ? std::min<int64_t>(im.span_max, n) : n;
    for (int c = col0; c < col0 + n; ) {
        const int m = (int) std::min<int64_t>(w, (int64_t) col0 + n - c);
        im.alloc_failed = false;
        if (one(c, m)) { c += m; continue; }
        if (!im.alloc_failed || m <= kMinSpan) return false;
        w = std::max(kMinSpan, m / 2);
        im.span_max = w;
        std::fprintf(stderr, "ds41_dense: layer %d: the graph of a %d-column span did not fit - %lld-column spans for the rest "
                     "of this prompt\n", il, m, (long long) w);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_kv_one(im, il, c, m); });
}

static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;
}

// route meter: columns col0..col0+n-1 of the host ids against the exact top-k read beside them
static void meter_add(Ds41Dense::Impl & im, int64_t col0, int64_t n) {
    const int64_t K = im.NU;
    for (int64_t t = col0; t < col0 + n; ++t) {
        const int32_t * c = im.host_ids.data() + t * K, * x = im.host_xids.data() + t * K;
        const float * w = im.host_xwts.data() + t * K;
        for (int64_t j = 0; j < K; ++j) {
            bool kept = false;
            for (int64_t i = 0; i < K && !kept; ++i) kept = c[i] == x[j];
            if (kept) im.rm_kept += w[j]; else ++im.rm_swaps;
        }
        im.rm_slots += K;
        ++im.rm_rows;
    }
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_attn_one(im, il, c, m); });
}

static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // wait the pass's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_attn(im, gc, il, n, col0);
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    if (im.o_xids) {
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    }
    ggml_backend_synchronize(im.backend);
    if (im.o_xids) meter_add(im, col0, n);
    lap(4);
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    ggml_backend_tensor_set(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    const int64_t n = im.n;
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    // a batch pass's graph reads its rows' slots: cached by the pass's layout (rows_sig), or built per pass
    const bool cached = (n == 1 && !im.rows_mode) || (im.rows_mode && im.rows_sig != 0);
    // a single sequence's DSpark verify widths (1 < n <= 8): their own cache, keyed by the bound slot (the graph reads
    // that slot's rings / caches; swap_slots / bind_slot reset_graphs() drops it with the rest)
    const bool mcached = !im.rows_mode && pass_graphs_on(n);
    if (cached) {
        const Impl::GKey key = im.rows_mode ? Impl::GKey { il, 8, im.rows_sig, n, -1 } : Impl::GKey { il, 1, cap, 1, im.cur_slot };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();   // many slots' decode graphs: start the cache over (rebuilt lazily)
            it = im.graphs.emplace(key, build_attn(im, im.gctx, il, n)).first;
        }
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { il, 1, cap, n, im.cur_slot }, [&](ggml_context * c) { return build_attn(im, c, il, n); });
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n);
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    if (!ok) return false;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {
        ggml_backend_tensor_get_async(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        ggml_backend_tensor_get_async(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    ggml_backend_synchronize(im.backend);
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.predict_out[il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true)) return false;
    ggml_backend_tensor_get(o, top_ids, 0, sizeof(int32_t) * 16 * (size_t) n);
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    (void) il;
    const int64_t n = im.n;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1 || (im.rows_mode && im.rows_sig != 0);   // (a batch pass: by its width)
    std::vector<uint8_t> meta;
    const bool mcached = !im.rows_mode && pass_graphs_on(n);
    if (cached) {
        const Impl::GKey key = n == 1 ? Impl::GKey { -1, 2, 0, 1 } : Impl::GKey { -1, 9, 0, n, -1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (n > 1 && !im.gctx_room()) im.reset_graphs();
            it = im.graphs.emplace(key, build_finish(im, im.gctx, n)).first;
        }
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { -1, 2, 0, n }, [&](ggml_context * c) { return build_finish(im, c, n); });
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_finish(im, gc, n);
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    (void) il;
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.fpredict_out[next_il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao);
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    ggml_backend_tensor_get(ao[0], buf.data(), 0, buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    if (!owns_head()) { p_->err = "logits on a stage without the head"; return false; }
    Impl & im = *p_;
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    const bool ok = im.compute(gf, false);
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    ggml_backend_tensor_get(im.x_state, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_get(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

std::string Ds41Dense::vram_ledger() const {
    const Impl & im = *p_;
    auto gib = [](size_t b) { return (double) b / 1073741824.0; };
    auto dev = [](ggml_backend_buffer_t b) { return b && !ggml_backend_buffer_is_host(b) ? ggml_backend_buffer_get_size(b) : (size_t) 0; };
    size_t w = 0;
    for (auto b : im.w.bufs) w += dev(b);
    const size_t gd = im.allo_dec.a ? ggml_gallocr_get_buffer_size(im.allo_dec.a, 0) : 0;
    const size_t gp = im.allo_pass.a ? ggml_gallocr_get_buffer_size(im.allo_pass.a, 0) : 0;
    char line[320];
    std::snprintf(line, sizeof line, "dense weights %.2f, KV state %.2f, columns %.2f (NC %lld), inputs %.2f, graphs dec %.2f + "
                  "pass %.2f = %.2f GiB", gib(w), gib(dev(im.sbuf)), gib(dev(im.nbuf)), (long long) im.NC, gib(dev(im.ibuf)),
                  gib(gd), gib(gp), gib(w + dev(im.sbuf) + dev(im.nbuf) + dev(im.ibuf) + gd + gp));
    return line;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw, L.raw_snap);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv, L.st_kv_snap); ggml_backend_tensor_copy(L.st_sc, L.st_sc_snap); }
    }
    ggml_backend_synchronize(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv); ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc); }
    }
    ggml_backend_synchronize(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

std::vector<Ds41Dense::StateRegion> Ds41Dense::state_regions(int64_t positions) const {
    const Impl & im = *p_;
    std::vector<StateRegion> out;
    if (positions < 1) return out;
    for (const auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) out.push_back({ t, 0, ggml_nbytes(t) });
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            out.push_back({ t, 0, (size_t) n * t->nb[1] });
            out.push_back({ t, (size_t) (rows - 1) * t->nb[1], t->nb[1] });   // the spare row (open compressor groups write it)
        }
    }
    return out;
}

void Ds41Dense::rearm(int pos) { p_->snap_pos = pos; }

void Ds41Dense::set_engram_history(int64_t pos0, int n, const int * tids, const uint8_t * dead) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return;
    for (int i = 0; i < n; ++i) {
        if (pos0 + i < 0) continue;
        if (dead && dead[i]) im.engram.set_dead(pos0 + i);
        else im.engram.set_token(pos0 + i, tids[i]);
    }
}

bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    ggml_backend_synchronize(im.backend);
    im.parked.clear();
    for (const StateRegion & r : state_regions(positions)) {
        Impl::ParkedRows pr{ r.t, r.off, std::vector<uint8_t>(r.bytes) };
        ggml_backend_tensor_get(r.t, pr.b.data(), r.off, r.bytes);
        im.parked.push_back(std::move(pr));
    }
    im.parked_pos = im.snap_pos;
    return true;
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    for (auto & r : im.parked) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
    ggml_backend_synchronize(im.backend);
    im.snap_pos = im.parked_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

void Ds41Dense::pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const {
    const Impl & im = *p_;
    streams.clear();
    regions.clear();
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const auto & L = im.ly[l];
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            PoolRegion r;
            r.name = std::string(t == L.comp ? "comp" : "idxk") + std::to_string(l);
            r.data = t->data;
            r.row_bytes = t->nb[1];
            r.rows = t->ne[1] - 1;   // row cmax is the spare an open compressor group writes (a pass's scratch)
            r.ratio = (int) L.ratio;
            r.bytes = (size_t) r.rows * r.row_bytes;
            streams.push_back(r);
        }
    }
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const auto & L = im.ly[l];
        ggml_tensor * ts[3] = { snap ? L.raw_snap : L.raw, snap ? L.st_kv_snap : L.st_kv, snap ? L.st_sc_snap : L.st_sc };
        const char * nm[3] = { "raw", "st_kv", "st_sc" };
        for (int i = 0; i < 3; ++i) {
            if (!ts[i]) continue;
            PoolRegion r;
            r.name = std::string(nm[i]) + std::to_string(l);
            r.data = ts[i]->data;
            r.bytes = ggml_nbytes(ts[i]);
            regions.push_back(r);
        }
    }
}

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!L.idx_src || !L.ratio) return true;
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    ggml_backend_tensor_get(L.sel, s.data(), 0, s.size() * 4);
    ggml_backend_tensor_get(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}

void Ds41Dense::set_layer_range(int64_t layer_lo, int64_t layer_hi) {
    p_->layer_lo = layer_lo;
    p_->layer_hi = layer_hi < 0 ? p_->nl : layer_hi;
}

int64_t Ds41Dense::layer_lo() const { return p_->layer_lo; }
int64_t Ds41Dense::layer_hi() const { return p_->layer_hi; }
bool Ds41Dense::owns_head() const {
    const Impl & im = *p_;
    if (im.cfg.head_only) return true;
    return im.layer_hi == im.nl && !im.cfg.omit_output_head;
}

int Ds41Dense::pass_size() const { return p_->n; }
int Ds41Dense::pass_pos0() const { return p_->pos0; }

bool Ds41Dense::set_pass_layout(int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass layout"; return false; }
    im.n = n;
    im.pos0 = pos0;
    im.batch = n > 1 ? n : 0;
    return true;
}

size_t Ds41Dense::handoff_bytes_per_column() const {
    const Impl & im = *p_;
    return (size_t) (im.D * im.HC + im.HC) * sizeof(float);
}

bool Ds41Dense::export_handoff(int col0, int n, float * dst) const {
    const Impl & im = *p_;
    if (!dst || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    for (int c = 0; c < n; ++c) {
        ggml_backend_tensor_get(im.x_state, dst, (size_t) (col0 + c) * col_x, col_x);
        dst += (im.D * im.HC);
        ggml_backend_tensor_get(im.pre_state, dst, (size_t) (col0 + c) * col_p, col_p);
        dst += im.HC;
    }
    return true;
}

bool Ds41Dense::import_handoff(int col0, int n, const float * src) {
    Impl & im = *p_;
    if (!src || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    for (int c = 0; c < n; ++c) {
        ggml_backend_tensor_set(im.x_state, src, (size_t) (col0 + c) * col_x, col_x);
        src += (im.D * im.HC);
        ggml_backend_tensor_set(im.pre_state, src, (size_t) (col0 + c) * col_p, col_p);
        src += im.HC;
    }
    ggml_backend_synchronize(im.backend);
    return true;
}

static void copy_tensor_bytes(ggml_tensor * dst, const ggml_tensor * src, ggml_backend_t backend) {
    if (!dst || !src || ggml_nbytes(dst) != ggml_nbytes(src)) return;
    std::vector<uint8_t> buf(ggml_nbytes(src));
    ggml_backend_tensor_get(src, buf.data(), 0, buf.size());
    ggml_backend_tensor_set(dst, buf.data(), 0, buf.size());
}

static void copy_tensor_cols(ggml_tensor * dst, const ggml_tensor * src, int col0, int n, size_t col_bytes) {
    if (!dst || !src || col_bytes == 0) return;
    std::vector<uint8_t> buf(col_bytes);
    for (int c = 0; c < n; ++c) {
        const size_t off = (size_t) (col0 + c) * col_bytes;
        ggml_backend_tensor_get(src, buf.data(), off, col_bytes);
        ggml_backend_tensor_set(dst, buf.data(), off, col_bytes);
    }
}

bool Ds41Dense::import_layer_cache_from(const Ds41Dense & src, int64_t global_layer, int col0, int n) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (global_layer < 0 || global_layer >= im.nl || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    auto & L = im.ly[(size_t) global_layer];
    const auto & SL = sm.ly[(size_t) global_layer];
    ggml_backend_synchronize(sm.backend);
    if (L.sel && SL.sel) {
        const size_t col_bytes = (size_t) im.ITOP * 4;
        copy_tensor_cols(L.sel, SL.sel, col0, n, col_bytes);
        copy_tensor_cols(L.selm, SL.selm, col0, n, col_bytes);
    }
    if (L.comp && SL.comp) copy_tensor_bytes(L.comp, SL.comp, im.backend);
    if (L.idxk && SL.idxk) copy_tensor_bytes(L.idxk, SL.idxk, im.backend);
    if (L.raw && SL.raw) copy_tensor_bytes(L.raw, SL.raw, im.backend);
    if (L.st_kv && SL.st_kv) copy_tensor_bytes(L.st_kv, SL.st_kv, im.backend);
    if (L.st_sc && SL.st_sc) copy_tensor_bytes(L.st_sc, SL.st_sc, im.backend);
    ggml_backend_synchronize(im.backend);
    return true;
}

bool Ds41Dense::import_cand_from(const Ds41Dense & src, int col0, int n) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (!im.cand || !sm.cand || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    ggml_backend_synchronize(sm.backend);
    const size_t col_bytes = (size_t) im.CB * 4;
    copy_tensor_cols(im.cand, sm.cand, col0, n, col_bytes);
    ggml_backend_synchronize(im.backend);
    return true;
}

namespace {
enum Ds41HandoffMode { kHandoffSameDevice, kHandoffPeer, kHandoffPinnedHost };

#ifdef STRATA_ENABLE_CUDA
/// Same probe as Strata peer experts (#2666): peer access must move data both ways exactly.
bool ds41_p2p_copies_data(int dev_a, int dev_b) {
    constexpr size_t kN = 1 << 14;
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {dev_a, dev_b};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        cudaSetDevice(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        cudaSetDevice(dev[from]);
        if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        cudaSetDevice(dev[to]);
        if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        cudaSetDevice(dev[to]);
        if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { cudaSetDevice(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

Ds41HandoffMode ds41_handoff_mode(int src_dev, int dst_dev) {
    if (src_dev < 0 || dst_dev < 0) return kHandoffPinnedHost;
    if (src_dev == dst_dev) return kHandoffSameDevice;
    int ab = 0, ba = 0;
    cudaDeviceCanAccessPeer(&ab, src_dev, dst_dev);
    cudaDeviceCanAccessPeer(&ba, dst_dev, src_dev);
    if (!ab || !ba) return kHandoffPinnedHost;
    cudaSetDevice(dst_dev);
    cudaError_t e0 = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
    cudaSetDevice(src_dev);
    cudaError_t e1 = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
    if (e0 != cudaSuccess || e1 != cudaSuccess) return kHandoffPinnedHost;
    if (!ds41_p2p_copies_data(src_dev, dst_dev)) {
        cudaSetDevice(dst_dev);
        cudaDeviceDisablePeerAccess(src_dev);
        cudaGetLastError();
        cudaSetDevice(src_dev);
        cudaDeviceDisablePeerAccess(dst_dev);
        cudaGetLastError();
        return kHandoffPinnedHost;
    }
    return kHandoffPeer;
}

const char * ds41_handoff_mode_name(Ds41HandoffMode m) {
    switch (m) {
        case kHandoffSameDevice: return "same-device";
        case kHandoffPeer: return "peer";
        case kHandoffPinnedHost: return "pinned-host";
    }
    return "?";
}

bool ds41_try_enable_peer(int dst_dev, int src_dev) {
    cudaSetDevice(dst_dev);
    const cudaError_t e = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) return true;
    cudaGetLastError();
    return false;
}
#endif
}  // namespace

bool Ds41Dense::import_handoff_from(const Ds41Dense & src, int src_cuda_device, int dst_cuda_device, int col0, int n,
                                    void * cuda_stream, std::vector<float> & pinned_host) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (col0 < 0 || n < 1 || col0 + n > im.NC || col0 + n > sm.NC || !im.x_state || !sm.x_state) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    if (col_x != (size_t) (sm.D * sm.HC) * sizeof(float) || im.HC != sm.HC) return false;
    const size_t col_floats = col_x / sizeof(float) + col_p / sizeof(float);
    const size_t need = col_floats * (size_t) n;
    if (pinned_host.size() < need) {
        pinned_host.resize(need);
#ifdef STRATA_ENABLE_CUDA
        if (!pinned_host.empty()) ggml_backend_cuda_register_host_buffer(pinned_host.data(), pinned_host.size() * 4);
#endif
    }
#ifdef STRATA_ENABLE_CUDA
    if (src_cuda_device < 0) src_cuda_device = sm.cfg.cuda_device;
    if (dst_cuda_device < 0) dst_cuda_device = im.cfg.cuda_device;
    if (src_cuda_device < 0 || dst_cuda_device < 0) return false;
    cudaSetDevice(src_cuda_device);
    ggml_backend_synchronize(sm.backend);
    const Ds41HandoffMode mode = ds41_handoff_mode(src_cuda_device, dst_cuda_device);
    if (mode == kHandoffPinnedHost) {
        if (!src.export_handoff(col0, n, pinned_host.data())) return false;
        ggml_backend_synchronize(sm.backend);
        cudaSetDevice(dst_cuda_device);
        if (!import_handoff(col0, n, pinned_host.data())) return false;
        ggml_backend_synchronize(im.backend);
        if (cuda_stream) cudaStreamSynchronize((cudaStream_t) cuda_stream);
        return true;
    }
    const size_t off_x = (size_t) col0 * col_x;
    const size_t off_p = (size_t) col0 * col_p;
    const size_t bytes_x = (size_t) n * col_x;
    const size_t bytes_p = (size_t) n * col_p;
    cudaStream_t stream = cuda_stream ? (cudaStream_t) cuda_stream : nullptr;
    cudaSetDevice(dst_cuda_device);
    ggml_backend_synchronize(im.backend);
    if (mode == kHandoffPeer && !ds41_try_enable_peer(dst_cuda_device, src_cuda_device)) {
        if (!src.export_handoff(col0, n, pinned_host.data())) return false;
        ggml_backend_synchronize(sm.backend);
        if (!import_handoff(col0, n, pinned_host.data())) return false;
        ggml_backend_synchronize(im.backend);
        return true;
    }
    if (mode == kHandoffSameDevice) {
        if (cudaMemcpyAsync((uint8_t *) im.x_state->data + off_x, (uint8_t *) sm.x_state->data + off_x, bytes_x, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync((uint8_t *) im.pre_state->data + off_p, (uint8_t *) sm.pre_state->data + off_p, bytes_p, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
    } else {
        if (cudaMemcpyPeerAsync((uint8_t *) im.x_state->data + off_x, dst_cuda_device, (uint8_t *) sm.x_state->data + off_x, src_cuda_device,
                                bytes_x, stream) != cudaSuccess)
            return false;
        if (cudaMemcpyPeerAsync((uint8_t *) im.pre_state->data + off_p, dst_cuda_device, (uint8_t *) sm.pre_state->data + off_p,
                                src_cuda_device, bytes_p, stream) != cudaSuccess)
            return false;
    }
    if (stream) cudaStreamSynchronize(stream);
    else cudaDeviceSynchronize();
    return true;
#else
    if (!src.export_handoff(col0, n, pinned_host.data())) return false;
    return import_handoff(col0, n, pinned_host.data());
#endif
}

bool Ds41Dense::begin_continued(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    im.batch = 0;
    im.span_cap.clear();
    return im.upload_inputs(tids, n, pos0);
}

bool Ds41Dense::begin_batch_continued(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) {
        im.err = "bad batch";
        return false;
    }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0;
    im.batch_pos0 = pos0;
    im.batch = B;
    im.n = B;
    im.col0 = 0;
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    return true;
}

namespace {
constexpr char kConvMagic[] = "DS41CONV1";
void append_bytes(std::vector<uint8_t> & out, const void * p, size_t n) {
    const auto * b = static_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);
}
}  // namespace

bool Ds41Dense::capture_conversation(int64_t positions, std::vector<uint8_t> & out) const {
    Impl & im = *const_cast<Impl *>(p_.get());
    if (positions < 1 || im.snap_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    out.clear();
    append_bytes(out, kConvMagic, 8);
    const int64_t lo = im.layer_lo, hi = im.layer_hi;
    append_bytes(out, &lo, sizeof lo);
    append_bytes(out, &hi, sizeof hi);
    append_bytes(out, &positions, sizeof positions);
    append_bytes(out, &im.snap_pos, sizeof im.snap_pos);
    auto put_tensor = [&](ggml_tensor * t, size_t off, size_t n) {
        if (!t || n == 0) return;
        const size_t o = out.size();
        out.resize(o + n);
        ggml_backend_tensor_get(t, out.data() + o, off, n);
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put_tensor(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put_tensor(t, 0, (size_t) n * t->nb[1]);
            put_tensor(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);
        }
    }
    return true;
}

bool Ds41Dense::restore_conversation(const std::vector<uint8_t> & in) {
    if (in.size() < 8 + 4 * sizeof(int64_t) || std::memcmp(in.data(), kConvMagic, 8) != 0) return false;
    Impl & im = *p_;
    size_t o = 8;
    int64_t lo = 0, hi = 0, positions = 0, snap = 0;
    std::memcpy(&lo, in.data() + o, sizeof lo); o += sizeof lo;
    std::memcpy(&hi, in.data() + o, sizeof hi); o += sizeof hi;
    std::memcpy(&positions, in.data() + o, sizeof positions); o += sizeof positions;
    std::memcpy(&snap, in.data() + o, sizeof snap); o += sizeof snap;
    if (lo != im.layer_lo || hi != im.layer_hi) return false;
    auto get_tensor = [&](ggml_tensor * t, size_t n) -> bool {
        if (!t) return n == 0;
        if (o + n > in.size()) return false;
        ggml_backend_tensor_set(t, in.data() + o, 0, n);
        o += n;
        return true;
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (!get_tensor(t, t ? ggml_nbytes(t) : 0)) return false;
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            if (!get_tensor(t, (size_t) n * t->nb[1])) return false;
            if (!get_tensor(t, t->nb[1])) return false;
        }
        if (L.raw_snap) ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv_snap) {
            ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv);
            ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc);
        }
    }
    im.snap_pos = (int) snap;
    ggml_backend_synchronize(im.backend);
    return true;
}
