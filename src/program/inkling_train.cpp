// xolotl-inkling --train: LoRA training of Inkling-2-Mini on the served (quantised) GGUF, one GPU.
// Included at the end of inkling_main.cpp (it uses Model, G, load(), forward(), state_reset()).
//
// The model is a stream DAG: every [n_embd, T] stream tensor (the embedding, each slot visit's output) gets an id on
// the host. A slot visit reads its input stream, the depth-bank blocks before it and, in loop passes 2+, the previous
// pass's bank and output. Training = forward every visit (no grad) keeping the streams on the host, the head CE in
// position chunks, then one VJP graph per visit in reverse order (surrogate loss sum(out * g_out)) whose input
// gradients are ADDED to the producers' accumulators - every consumer of a stream runs after its producer, so a
// producer's gradient is complete when it is reached. Adapter gradients add over the three loop visits.
// Design + gates: docs/train/INKLING_TRAIN_PLAN.md.

#include "../../tools/train/lora_gguf.hpp"
#include "../../tools/train/metrics.hpp"

#include <random>

namespace {   // the engine's types live in an anonymous namespace too
namespace itrain {

using xolotl::train::LoraTensorPair;

// ---- one graph on the GPU backend: a no-alloc context, host inputs uploaded after allocation (QFN trainer's G1) -
static ggml_gallocr_t g_alloc = nullptr;
static size_t g_alloc_peak = 0;
struct G1 {
    ggml_context* c = nullptr;
    ggml_cgraph* gf = nullptr;
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
    explicit G1(size_t n_nodes = 1 << 16) {
        ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes * 2) + ggml_graph_overhead_custom(n_nodes, true), nullptr, true};
        c = ggml_init(ip);
        gf = ggml_new_graph_custom(c, n_nodes, true);
    }
    ~G1() { if (c) ggml_free(c); }
    G1(const G1&) = delete;
    ggml_tensor* f32(const float* d, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        ggml_tensor* t = ggml_new_tensor_3d(c, GGML_TYPE_F32, n0, n1, n2);
        ggml_set_input(t);
        up.emplace_back(t, std::vector<uint8_t>((const uint8_t*) d, (const uint8_t*) d + ggml_nbytes(t)));
        return t;
    }
    ggml_tensor* i32(const int32_t* d, int64_t n0, int64_t n1 = 1) {
        ggml_tensor* t = ggml_new_tensor_2d(c, GGML_TYPE_I32, n0, n1);
        ggml_set_input(t);
        up.emplace_back(t, std::vector<uint8_t>((const uint8_t*) d, (const uint8_t*) d + ggml_nbytes(t)));
        return t;
    }
    // grads_of: tensors whose gradient is read back after the compute
    void run(ggml_backend_t be, ggml_tensor* loss, const std::vector<ggml_tensor*>& outs, const std::vector<ggml_tensor*>& grads_of = {}) {
        for (ggml_tensor* o : outs) if (o) { ggml_set_output(o); ggml_build_forward_expand(gf, o); }
        if (loss) {
            ggml_set_loss(loss);
            ggml_build_forward_expand(gf, loss);
            ggml_build_backward_expand(c, gf, nullptr);
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
                if (ggml_tensor* acc = ggml_graph_get_grad_acc(gf, ggml_graph_node(gf, i))) ggml_set_input(acc);
            for (ggml_tensor* t : grads_of) {
                if (ggml_tensor* acc = ggml_graph_get_grad_acc(gf, t)) ggml_set_input(acc);
                if (ggml_tensor* gr = ggml_graph_get_grad(gf, t)) ggml_set_output(gr);
            }
        }
        if (!g_alloc) g_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(g_alloc, gf)) die("--train: graph allocation failed (VRAM)");
        const size_t bs = ggml_gallocr_get_buffer_size(g_alloc, 0);
        if (bs > g_alloc_peak) {
            g_alloc_peak = bs;
            std::fprintf(stderr, "[train] compute buffer grew to %.2f GiB (%d nodes)\n", bs / 1073741824.0, ggml_graph_n_nodes(gf));
        }
        for (auto& h : up) if (h.first->buffer) ggml_backend_tensor_set(h.first, h.second.data(), 0, h.second.size());
        if (loss) ggml_graph_reset(gf);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) die("--train: graph compute failed");
    }
    std::vector<float> grad(ggml_tensor* t) {   // zeros when the tensor got no gradient
        std::vector<float> v((size_t) ggml_nelements(t), 0.0f);
        if (ggml_tensor* g = ggml_graph_get_grad(gf, t)) if (g->buffer) ggml_backend_tensor_get(g, v.data(), 0, v.size() * 4);
        return v;
    }
};
static std::vector<float> fetch(ggml_tensor* t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
    return v;
}
static void addto(std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty()) { a = b; return; }
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

struct Args {
    std::string gguf, data, eval, out = "runs", run = "inkling-lora", save, targets = "q,k,v,o,gate,rel", init_from, resume, prefix_file;
    int rank = 16, steps = 0, warmup = 10, max_t = 2048, threads = 8, eval_every = 0, eval_n = 16, save_every = 0, seed = 0, ce_chunk = 256;
    int check = 0, fd = 0, overfit = 0, check_prefix = 0, max_prefix = 0, shuffle = 1;
    float alpha = 32.0f, lr = 1e-4f, clip = 1.0f, init_b = 0.0f, lr_min = 0.1f;
};

struct Window {
    std::vector<int32_t> ids; std::vector<uint8_t> mask;
    std::shared_ptr<const std::vector<int32_t>> prefix;   // W9: context before the window (no loss, carried as engine state)
    std::vector<int32_t> context;   // W9 with --prefix: the conversation tokens between the shared prompt and this window
};

// {"ids":[...], "mask":[...], ...} per line
static bool parse_int_array(const std::string& s, const char* key, std::vector<long long>& out) {
    const std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    p = s.find('[', p);
    if (p == std::string::npos) return false;
    const size_t e = s.find(']', p);
    out.clear();
    const char* q = s.c_str() + p + 1;
    const char* end = s.c_str() + e;
    while (q < end) {
        char* nx = nullptr;
        const long long v = std::strtoll(q, &nx, 10);
        if (nx == q) { ++q; continue; }
        out.push_back(v);
        q = nx;
    }
    return true;
}
static std::vector<Window> load_windows(const std::string& path, int max_t) {
    std::vector<Window> w;
    std::ifstream f(path);
    if (!f) die("--train: cannot read " + path);
    std::string line;
    std::vector<long long> ids, mask, pre;
    std::shared_ptr<const std::vector<int32_t>> last_pre;   // consecutive windows of one request share their prefix
    while (std::getline(f, line)) {
        if (line.empty() || !parse_int_array(line, "ids", ids)) continue;
        if (!parse_int_array(line, "mask", mask) || mask.size() != ids.size()) die("--train: a window without a matching mask in " + path);
        Window x;
        if (parse_int_array(line, "prefix", pre) && !pre.empty()) {   // a per-window context (an agent request: the whole conversation before the reply)
            if (!last_pre || last_pre->size() != pre.size() || !std::equal(pre.begin(), pre.end(), last_pre->begin(),
                                                                             [](long long a, int32_t b) { return a == b; })) {
                auto v = std::make_shared<std::vector<int32_t>>(pre.begin(), pre.end());
                last_pre = v;
            }
            x.prefix = last_pre;
        }
        if (parse_int_array(line, "context", pre) && !pre.empty()) x.context.assign(pre.begin(), pre.end());
        const size_t n = std::min<size_t>(ids.size(), (size_t) max_t);
        for (size_t i = 0; i < n; ++i) { x.ids.push_back((int32_t) ids[i]); x.mask.push_back(mask[i] ? 1 : 0); }
        int nl = 0;
        for (size_t i = 1; i < n; ++i) nl += x.mask[i];
        if (n >= 2 && nl > 0) w.push_back(std::move(x));
    }
    return w;
}

struct Adapter {
    int ip = 0;
    std::string tgt;
    ggml_tensor* W = nullptr;
    int64_t n_in = 0, n_out = 0;
    std::vector<float> A, B, gA, gB, mA, vA, mB, vB;
};

struct SlotIO {   // one slot visit of build(), with its stream ids
    int il = 0, ip = 0, in = 0, out = 0, bank_out = -1;
    std::vector<int> blocks, bank;
};

struct Trainer {
    Model& m;
    const Hp& h;
    Args a;
    ggml_backend_t be;
    std::vector<SlotIO> plan;
    int n_ids = 0, head_cur = 0;
    std::vector<int> head_blocks;
    std::vector<Adapter> ad;
    std::vector<std::vector<int>> ad_of;   // per physical layer
    float scale = 1.0f;

    Trainer(Model& mm, const Args& aa) : m(mm), h(mm.hp), a(aa), be(mm.gpu) {
        // the slot sequence of build(), on ids
        int cur = n_ids++;   // 0 = the embedding
        std::vector<int> blocks;
        auto visit = [&](int il, int ip, const std::vector<int>& bank, int bank_out) {
            SlotIO io;
            io.il = il; io.ip = ip; io.in = cur; io.blocks = blocks; io.bank = bank; io.bank_out = bank_out;
            io.out = n_ids++;
            plan.push_back(io);
            if (ip % h.block == 0) blocks.push_back(cur);   // layer() pushes its input as a new block
            cur = io.out;
        };
        const int start = h.n_pass > 1 ? h.loop_start : h.n_phys, end = h.n_pass > 1 ? h.loop_end : h.n_phys;
        for (int il = 0; il < start; ++il) visit(il, il, {}, -1);
        const int prelude = cur;
        const std::vector<int> prelude_blocks = blocks;
        for (int pass = 0; pass < h.n_pass && start < end; ++pass) {
            std::vector<int> bank;
            int bank_out = -1;
            if (pass > 0) {
                bank.assign(blocks.begin() + (long) prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ipy = start; ipy < end; ++ipy) {
                const int il = pass == 0 ? ipy : h.n_phys + (pass - 1) * (end - start) + (ipy - start);
                visit(il, ipy, bank, bank_out);
            }
        }
        for (int il = end; il < h.n_phys; ++il) visit(il, il, {}, -1);
        head_cur = cur;
        head_blocks = blocks;
        if ((int) plan.size() != h.n_slot) die("--train: slot plan has " + std::to_string(plan.size()) + " visits, model says " + std::to_string(h.n_slot));

        // adapters on base tensors
        std::mt19937 rng(a.seed);
        ad_of.assign(h.n_phys, {});
        std::vector<std::string> tg;
        { std::stringstream ss(a.targets); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) tg.push_back(t); }
        for (int ip = 0; ip < h.n_phys; ++ip) {
            const Layer& y = m.L[ip];
            for (const std::string& t : tg) {
                ggml_tensor* W = t == "q" ? y.wq : t == "k" ? y.wk : t == "v" ? y.wv : t == "o" ? y.wo : t == "gate" ? y.gate : t == "rel" ? y.rel : nullptr;
                if (!W) die("--train: unknown or missing target " + t + " on layer " + std::to_string(ip));
                Adapter x;
                x.ip = ip; x.tgt = t; x.W = W; x.n_in = W->ne[0]; x.n_out = W->ne[1];
                const size_t na = (size_t) x.n_in * a.rank, nb = (size_t) a.rank * x.n_out;
                x.A.resize(na); x.B.assign(nb, 0.0f);
                std::uniform_real_distribution<float> ua(-1.0f / std::sqrt((float) x.n_in), 1.0f / std::sqrt((float) x.n_in));
                for (float& v : x.A) v = ua(rng);
                if (a.init_b > 0) { std::normal_distribution<float> nb_(0.0f, a.init_b); for (float& v : x.B) v = nb_(rng); }
                x.gA.assign(na, 0.0f); x.gB.assign(nb, 0.0f);
                x.mA.assign(na, 0.0f); x.vA.assign(na, 0.0f); x.mB.assign(nb, 0.0f); x.vB.assign(nb, 0.0f);
                ad_of[ip].push_back((int) ad.size());
                ad.push_back(std::move(x));
            }
        }
        scale = a.alpha / (float) a.rank;
        size_t np = 0;
        for (const Adapter& x : ad) np += x.A.size() + x.B.size();
        std::fprintf(stderr, "[train] %zu adapters (%s, r %d, alpha %.0f), %.2f M params; %d slot visits, %d stream ids\n",
                     ad.size(), a.targets.c_str(), a.rank, a.alpha, np / 1e6, (int) plan.size(), n_ids);
    }

    // host masks / relative distances for cells = positions [0, T) (the chunk starts at 0, no carried prefix yet)
    // W9 prefix carry: the context before a window runs through the inference engine once (no gradient, base weights,
    // served kernels); each slot visit of the window then attends to that slot's cached K/V (global slots: every
    // prefix position = the first n_pre cache cells; sliding slots: the last n_swa-1 positions, gathered from the ring)
    // and starts its convolutions from the slot's conv state. Consecutive windows whose prefix extends the fed one
    // only prefill the new tokens (append-only, as the engine resumes requests).
    std::shared_ptr<const std::vector<int32_t>> global_prefix;   // --prefix: one context for every window (harness prompt)
    std::vector<int32_t> pre_fed;            // the ids the engine's state holds
    int64_t n_pre = 0, n_pre_swa = 0;        // the current window's prefix length / its visible sliding tail
    bool swa_full = false;                   // --check gate: sliding slots read the whole prefix (the engine's layout)
    std::vector<int32_t> swa_cells;          // ring cells of the sliding tail, in position order
    double prefill_s = 0;
    int64_t prefill_tok = 0;
    // the engine state right after the shared prompt (--prefix): every slot's conv state, the sliding rings and their
    // position map (global cells below the prompt length are never overwritten by later positions, so they need no
    // copy). The conv state is recurrent - without this every window would re-prefill the whole prompt.
    struct Snap { std::vector<int32_t> ids; ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
                  std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pairs; std::vector<int64_t> ring_pos; bool valid = false; } snap;
    void snap_take() {
        if (!snap.ctx) {
            ggml_init_params ip = {ggml_tensor_overhead() * (m.S.size() * 3 + 4), nullptr, true};
            snap.ctx = ggml_init(ip);
            for (Slot& z : m.S) {
                snap.pairs.push_back({z.conv, ggml_dup_tensor(snap.ctx, z.conv)});
                if (z.swa) { snap.pairs.push_back({z.k_cache, ggml_dup_tensor(snap.ctx, z.k_cache)}); snap.pairs.push_back({z.v_cache, ggml_dup_tensor(snap.ctx, z.v_cache)}); }
            }
            snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, m.gpu ? m.gpu : m.cpu);
            if (!snap.buf) die("--train: cannot allocate the prompt snapshot");
            std::fprintf(stderr, "[train] prompt snapshot: %.0f MiB\n", ggml_backend_buffer_get_size(snap.buf) / 1048576.0);
        }
        for (auto& [src, dst] : snap.pairs) ggml_backend_tensor_copy(src, dst);
        snap.ring_pos = m.ring_pos;
        snap.ids = pre_fed;
        snap.valid = true;
    }
    void snap_restore() {
        for (auto& [src, dst] : snap.pairs) ggml_backend_tensor_copy(dst, src);
        m.ring_pos = snap.ring_pos;
        pre_fed = snap.ids;
    }
    static bool starts_with(const std::vector<int32_t>& a, const std::vector<int32_t>& p) {
        return a.size() >= p.size() && std::equal(p.begin(), p.end(), a.begin());
    }
    void set_prefix(const std::vector<int32_t>* ids, bool served = true) {
        if (!ids || ids->empty()) { n_pre = n_pre_swa = 0; swa_cells.clear(); return; }
        // the shared prompt first: from the snapshot when the engine holds something else past it
        if (global_prefix && !global_prefix->empty() && starts_with(*ids, *global_prefix)) {
            size_t c = 0;
            while (c < pre_fed.size() && c < ids->size() && pre_fed[c] == (*ids)[c]) ++c;
            const bool rollback = c < pre_fed.size() && c >= global_prefix->size();   // diverges after the prompt
            const bool fresh = !starts_with(pre_fed, *global_prefix);
            if (rollback || fresh) {
                if (snap.valid) snap_restore();
                else { state_reset(m); pre_fed.clear(); prefill(*global_prefix, served); snap_take(); }
            }
        }
        if ((int64_t) ids->size() + 2 * std::min(a.max_t, 4096) > m.n_ctx) die("--train: a prefix of " + std::to_string(ids->size()) +
                                                               " tokens needs --max-prefix >= that (context " + std::to_string(m.n_ctx) + ")");
        size_t common = 0;
        while (common < pre_fed.size() && common < ids->size() && pre_fed[common] == (*ids)[common]) ++common;
        if (common < pre_fed.size()) { state_reset(m); pre_fed.clear(); common = 0; }
        if (common < ids->size()) prefill(*ids, served);
        n_pre = (int64_t) ids->size();
        // sliding slots read the last n_swa-1 prefix positions; the --check gate (swa_full) reads every prefix cell,
        // the far ones masked, in the engine's own cell order: the engine's f16-accumulated attention product depends
        // on where masked cells sit, so only the identical layout is bit-exact (one masked head cell: 2.9e-3)
        n_pre_swa = swa_full ? n_pre : std::min<int64_t>(n_pre, h.n_swa - 1);
        if (swa_full && n_pre + std::min(a.max_t, 4096) > m.ring) die("--check-prefix: the sliding ring is too small for the gate layout");
        swa_cells.assign((size_t) n_pre_swa, -1);
        for (int64_t j = 0; j < m.ring; ++j) {
            const int64_t pos = m.ring_pos[j];
            if (pos >= n_pre - n_pre_swa && pos < n_pre) swa_cells[(size_t) (pos - (n_pre - n_pre_swa))] = (int32_t) j;
        }
        for (int32_t c : swa_cells) if (c < 0) die("--train: a sliding-window prefix position is not in the ring");
    }

    // feed ids[pre_fed.size():] (pre_fed must be a prefix of ids) through the engine
    void prefill(const std::vector<int32_t>& ids, bool served) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t from = (int64_t) pre_fed.size();
        const bool was = inkling_train_mode();
        inkling_train_mode() = !served;    // served: the fused serve kernels, as the context is computed at serve time
        drop_graph(m);
        std::vector<float> lg;
        for (int64_t p = from; p < (int64_t) ids.size();) {
            const int T = chunk_len(m, p, (int64_t) ids.size() - p);
            forward(m, ids.data() + p, T, p, 1, lg);
            p += T;
        }
        drop_graph(m);
        inkling_train_mode() = was;
        pre_fed.assign(ids.begin(), ids.end());
        prefill_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        prefill_tok += (int64_t) ids.size() - from;
    }

    static int64_t cells_of(int64_t npre, int T) { return (npre + T + 255) / 256 * 256; }
    void chunk_inputs(G1& gg, Step& st, int T) {
        // cells: global = the n_pre prefix positions then the window; sliding = the n_pre_swa tail then the window
        auto mk = [&](int64_t npre, int64_t win, bool swa, ChunkIn& ci) {
            const int64_t cells = cells_of(npre, T);   // npre + T, padded with masked cells (j > npre + t) like the engine
            std::vector<float> mk_((size_t) cells * T);
            std::vector<int32_t> rp((size_t) cells * T);
            for (int t = 0; t < T; ++t)
                for (int64_t j = 0; j < cells; ++j) {
                    const int64_t d = (npre + t) - j;   // distance in positions (the tail is contiguous up to the window)
                    const size_t k = (size_t) t * cells + j;
                    const bool vis = d >= 0 && (!swa || d < win);
                    mk_[k] = vis ? 0.0f : -INFINITY;
                    rp[k] = swa ? (vis ? (int32_t) d : (int32_t) win) : (d >= 0 && d < h.rel_extent ? (int32_t) d : h.rel_extent);
                }
            ci.cells = cells;
            ci.mask = gg.f32(mk_.data(), cells, T); ci.rp = gg.i32(rp.data(), cells, T);
        };
        ChunkIn cf, cs;
        mk(n_pre, 0, false, cf);
        mk(n_pre_swa, h.n_swa, true, cs);
        st.ck_full = {cf};
        st.ck_swa = {cs};
    }
    ggml_tensor* tau_input(G1& gg, int T) {   // build()'s log-scaled positions
        std::vector<float> pf(T);
        for (int t = 0; t < T; ++t) pf[t] = (float) (n_pre + t);   // absolute positions (W9: after the prefix)
        ggml_tensor* posf = gg.f32(pf.data(), T);
        ggml_tensor* tau = ggml_scale_bias(gg.c, posf, 1.0f, 1.0f);
        tau = ggml_log(gg.c, ggml_clamp(gg.c, ggml_scale(gg.c, tau, 1.0f / (float) h.log_floor), 1.0f, INFINITY));
        return ggml_reshape_3d(gg.c, ggml_scale_bias(gg.c, tau, h.log_alpha, 1.0f), 1, 1, T);
    }

    // one slot visit as a graph; ins/in_ids = its stream inputs, abs = its adapters' (A, B) inputs in ad_of order
    ggml_tensor* slot_graph(G1& gg, Step& st, const SlotIO& io, int T, const std::vector<std::vector<float>>& X, bool vjp,
                            std::vector<ggml_tensor*>& ins, std::vector<int>& in_ids,
                            std::vector<std::pair<ggml_tensor*, ggml_tensor*>>& abs, std::unordered_map<const ggml_tensor*, G::Lora>& lmap) {
        chunk_inputs(gg, st, T);
        G g{m, h, gg.c, gg.gf, st, T, 0};
        g.train = true;
        if (n_pre > 0) {   // W9: the slot's conv state and cached K/V after the prefix (read in place, constants)
            Slot& z = m.S[(size_t) io.il];
            g.tconv = ggml_view_2d(gg.c, z.conv, h.n_conv - 1, h.n_chan(), z.conv->nb[1], 0);
            const int64_t hd = h.hd, nkv = h.n_head_kv;
            if (!h.swa[io.ip]) {
                g.tk_pre = ggml_cast(gg.c, ggml_view_3d(gg.c, z.k_cache, hd, nkv, n_pre, z.k_cache->nb[1], z.k_cache->nb[2], 0), GGML_TYPE_F32);
                g.tv_pre = ggml_cast(gg.c, ggml_view_3d(gg.c, z.v_cache, hd, nkv, n_pre, z.v_cache->nb[1], z.v_cache->nb[2], 0), GGML_TYPE_F32);
            } else {
                ggml_tensor* idx = gg.i32(swa_cells.data(), n_pre_swa);
                auto tail = [&](ggml_tensor* cache) {
                    ggml_tensor* rows = ggml_get_rows(gg.c, ggml_reshape_2d(gg.c, cache, hd * nkv, cache->ne[2]), idx);   // [hd*nkv, n] f32
                    return ggml_reshape_3d(gg.c, rows, hd, nkv, n_pre_swa);
                };
                g.tk_pre = tail(z.k_cache);
                g.tv_pre = tail(z.v_cache);
            }
        } else {
            std::vector<float> zc((size_t) (h.n_conv - 1) * h.n_chan(), 0.0f);
            g.tconv = ggml_reshape_2d(gg.c, gg.f32(zc.data(), h.n_conv - 1, h.n_chan()), h.n_conv - 1, h.n_chan());
        }
        {   // masked zero cells up to the engine's 256-multiple (see G::tpad)
            const int64_t npre = h.swa[io.ip] ? n_pre_swa : n_pre;
            const int64_t pad = cells_of(npre, T) - (npre + T);
            if (pad > 0) {
                std::vector<float> zp((size_t) h.hd * h.n_head_kv * pad, 0.0f);
                g.tpad = ggml_reshape_3d(gg.c, gg.f32(zp.data(), h.hd, h.n_head_kv, pad), h.hd, h.n_head_kv, pad);
            }
        }
        g.tau = tau_input(gg, T);
        auto in = [&](int id) {
            ggml_tensor* t = ggml_reshape_2d(gg.c, gg.f32(X[id].data(), h.n_embd, T), h.n_embd, T);
            if (vjp) ggml_set_param(t->view_src ? t->view_src : t);
            ins.push_back(t->view_src ? t->view_src : t);
            in_ids.push_back(id);
            return t;
        };
        ggml_tensor* cur = in(io.in);
        std::vector<ggml_tensor*> blocks;
        for (int b : io.blocks) blocks.push_back(in(b));
        ggml_tensor* msgs = nullptr;
        if (io.bank_out >= 0) {
            std::vector<ggml_tensor*> bank;
            for (int b : io.bank) bank.push_back(in(b));
            ggml_tensor* bo = in(io.bank_out);
            msgs = g.mix(bank, bo, m.L[io.ip].msg_query);
        }
        for (int k : ad_of[io.ip]) {
            Adapter& x = ad[k];
            ggml_tensor* ta = ggml_reshape_2d(gg.c, gg.f32(x.A.data(), x.n_in, a.rank), x.n_in, a.rank);
            ggml_tensor* tb = ggml_reshape_2d(gg.c, gg.f32(x.B.data(), a.rank, x.n_out), a.rank, x.n_out);
            if (vjp) { ggml_set_param(ta->view_src); ggml_set_param(tb->view_src); }
            lmap[x.W] = {ta, tb};
            abs.push_back({ta->view_src, tb->view_src});
        }
        g.lora = &lmap;
        g.lora_scale = scale;
        const size_t v = (size_t) (&io - plan.data());
        if (route_mode == 2) g.fixed_ids = gg.i32(route_ids.at(v).data(), h.n_used, T);
        if (route_mode == 1 && !vjp) g.ids_out = &route_rec;
        return g.layer(io.il, io.ip, cur, blocks, msgs);
    }

    // the head's normed stream [n_embd, T] from the last stream + the blocks
    ggml_tensor* head_graph(G1& gg, Step& st, int T, const std::vector<std::vector<float>>& X, bool vjp,
                            std::vector<ggml_tensor*>& ins, std::vector<int>& in_ids) {
        G g{m, h, gg.c, gg.gf, st, T, 0};
        g.train = true;
        auto in = [&](int id) {
            ggml_tensor* t = gg.f32(X[id].data(), h.n_embd, T);
            if (vjp) ggml_set_param(t);
            ins.push_back(t); in_ids.push_back(id);
            return ggml_reshape_2d(gg.c, t, h.n_embd, T);
        };
        ggml_tensor* cur = in(head_cur);
        std::vector<ggml_tensor*> blocks;
        for (int b : head_blocks) blocks.push_back(in(b));
        ggml_tensor* x = ggml_reshape_2d(gg.c, g.mix(blocks, cur, m.output_res_query), h.n_embd, T);
        return g.rms(x, m.output_norm);
    }

    // logits of rows xs [n_embd, n] (the served head: output weight, softcap, the real vocab)
    ggml_tensor* logits_of(ggml_context* c, ggml_tensor* xs) {
        ggml_tensor* l = ggml_mul_mat(c, m.output, xs);
        ggml_prec_set_acc(l, GGML_PREC_F32);
        if (h.softcap != 0.0f) l = ggml_scale(c, ggml_tanh(c, ggml_scale(c, l, 1.0f / h.softcap)), h.softcap);
        return ggml_cont(c, ggml_view_2d(c, l, h.n_vocab, l->ne[1], l->nb[1], 0));
    }

    std::vector<float> embed(const std::vector<int32_t>& ids) {
        const int64_t E = h.n_embd, T = (int64_t) ids.size();
        std::vector<float> raw((size_t) E * T);
        const size_t rb = ggml_row_size(m.tok_embd->type, E);
        std::vector<uint8_t> row(rb);
        const auto* tt = ggml_get_type_traits(m.tok_embd->type);
        for (int64_t t = 0; t < T; ++t) {
            ggml_backend_tensor_get(m.tok_embd, row.data(), (size_t) ids[t] * m.tok_embd->nb[1], rb);
            if (m.tok_embd->type == GGML_TYPE_F32) std::memcpy(&raw[(size_t) t * E], row.data(), rb);
            else tt->to_float(row.data(), &raw[(size_t) t * E], E);
        }
        G1 gg(4096);
        Step st;
        G g{m, h, gg.c, gg.gf, st, (int) T, 0};
        ggml_tensor* x = g.rms(gg.f32(raw.data(), E, T), m.tok_norm);
        gg.run(be, nullptr, {x});
        return fetch(x);
    }

    // forward every visit; X[id] = the streams. Returns the head's normed stream.
    std::vector<float> forward_streams(const std::vector<int32_t>& ids, std::vector<std::vector<float>>& X) {
        const int T = (int) ids.size();
        X.assign(n_ids, {});
        X[0] = embed(ids);
        for (const SlotIO& io : plan) {
            G1 gg;
            Step st;
            std::vector<ggml_tensor*> ins; std::vector<int> iid;
            std::vector<std::pair<ggml_tensor*, ggml_tensor*>> abs;
            std::unordered_map<const ggml_tensor*, G::Lora> lmap;
            route_rec = nullptr;
            ggml_tensor* out = slot_graph(gg, st, io, T, X, false, ins, iid, abs, lmap);
            gg.run(be, nullptr, {out, route_rec});
            X[io.out] = fetch(out);
            if (const char* dp = std::getenv("XOLOTL_INKLING_DUMP_STREAMS")) {   // debug: every slot's output stream
                static FILE* df = std::fopen(dp, "wb");
                if (df) { std::fwrite(X[io.out].data(), 4, X[io.out].size(), df); std::fflush(df); }
            }
            if (route_rec) {   // a strided view over the full argsort: read row by row
                std::vector<int32_t>& r = route_ids[(size_t) (&io - plan.data())];
                r.resize((size_t) h.n_used * T);
                for (int t = 0; t < T; ++t)
                    ggml_backend_tensor_get(route_rec, &r[(size_t) t * h.n_used], (size_t) t * route_rec->nb[1], (size_t) h.n_used * 4);
            }
        }
        G1 gg;
        Step st;
        std::vector<ggml_tensor*> ins; std::vector<int> iid;
        ggml_tensor* xf = head_graph(gg, st, T, X, false, ins, iid);
        gg.run(be, nullptr, {xf});
        return fetch(xf);
    }

    // mean CE over the loss positions (row t predicts ids[t+1] where mask[t+1]); dxf (optional) = d loss / d xf
    double head_ce(const Window& w, const std::vector<float>& xf, std::vector<float>* dxf) {
        const int T = (int) w.ids.size();
        const int64_t E = h.n_embd, V = h.n_vocab;
        std::vector<int> rows;
        for (int t = 0; t + 1 < T; ++t) if (w.mask[t + 1]) rows.push_back(t);
        const double N = (double) rows.size();
        if (dxf) dxf->assign((size_t) E * T, 0.0f);
        double loss = 0;
        for (size_t r0 = 0; r0 < rows.size(); r0 += a.ce_chunk) {
            const int n = (int) std::min<size_t>(a.ce_chunk, rows.size() - r0);
            std::vector<float> xs((size_t) E * n), lab((size_t) V * n, 0.0f);
            for (int i = 0; i < n; ++i) {
                const int t = rows[r0 + i];
                std::memcpy(&xs[(size_t) i * E], &xf[(size_t) t * E], E * sizeof(float));
                lab[(size_t) i * V + w.ids[t + 1]] = 1.0f;
            }
            G1 gg(4096);
            ggml_tensor* tx = gg.f32(xs.data(), E, n);
            if (dxf) ggml_set_param(tx);
            ggml_tensor* tl = gg.f32(lab.data(), V, n);
            ggml_tensor* ce = ggml_scale(gg.c, ggml_cross_entropy_loss(gg.c, logits_of(gg.c, ggml_reshape_2d(gg.c, tx, E, n)), ggml_reshape_2d(gg.c, tl, V, n)),
                                         (float) (n / N));
            gg.run(be, dxf ? ce : nullptr, {ce}, dxf ? std::vector<ggml_tensor*>{tx} : std::vector<ggml_tensor*>{});
            loss += fetch(ce)[0];
            if (dxf) {
                const std::vector<float> g = gg.grad(tx);
                for (int i = 0; i < n; ++i) std::memcpy(&(*dxf)[(size_t) rows[r0 + i] * E], &g[(size_t) i * E], E * sizeof(float));
            }
        }
        return loss;
    }

    // one window: loss, and (grads) the adapter gradients accumulated into ad[*].gA/gB
    double step_window(const Window& w, bool grads) {
        if (w.prefix) set_prefix(w.prefix.get());
        else if (global_prefix && !w.context.empty()) {   // the shared prompt + this window's conversation context
            std::vector<int32_t> full(*global_prefix);
            full.insert(full.end(), w.context.begin(), w.context.end());
            set_prefix(&full);
        } else set_prefix(global_prefix.get());
        const int T = (int) w.ids.size();
        std::vector<std::vector<float>> X;
        const std::vector<float> xf = forward_streams(w.ids, X);
        std::vector<float> dxf;
        const double loss = head_ce(w, xf, grads ? &dxf : nullptr);
        if (!grads) return loss;
        std::vector<std::vector<float>> GX(n_ids);
        {   // head VJP
            G1 gg;
            Step st;
            std::vector<ggml_tensor*> ins; std::vector<int> iid;
            ggml_tensor* x = head_graph(gg, st, T, X, true, ins, iid);
            ggml_tensor* go = gg.f32(dxf.data(), h.n_embd, T);
            ggml_tensor* sl = ggml_sum(gg.c, ggml_mul(gg.c, x, ggml_reshape_2d(gg.c, go, h.n_embd, T)));
            gg.run(be, sl, {}, ins);
            for (size_t i = 0; i < ins.size(); ++i) addto(GX[iid[i]], gg.grad(ins[i]));
        }
        for (int k = (int) plan.size() - 1; k >= 0; --k) {   // slot VJPs, reverse order
            const SlotIO& io = plan[k];
            if (GX[io.out].empty()) die("--train: slot " + std::to_string(io.il) + " has no output gradient (a dropped DAG edge)");
            G1 gg;
            Step st;
            std::vector<ggml_tensor*> ins; std::vector<int> iid;
            std::vector<std::pair<ggml_tensor*, ggml_tensor*>> abs;
            std::unordered_map<const ggml_tensor*, G::Lora> lmap;
            ggml_tensor* out = slot_graph(gg, st, io, T, X, true, ins, iid, abs, lmap);
            ggml_tensor* go = gg.f32(GX[io.out].data(), h.n_embd, T);
            ggml_tensor* sl = ggml_sum(gg.c, ggml_mul(gg.c, out, ggml_reshape_2d(gg.c, go, h.n_embd, T)));
            std::vector<ggml_tensor*> want = ins;
            for (auto& p : abs) { want.push_back(p.first); want.push_back(p.second); }
            gg.run(be, sl, {}, want);
            for (size_t i = 0; i < ins.size(); ++i) addto(GX[iid[i]], gg.grad(ins[i]));
            for (size_t j = 0; j < abs.size(); ++j) {
                Adapter& x = ad[ad_of[io.ip][j]];
                const std::vector<float> ga = gg.grad(abs[j].first), gb = gg.grad(abs[j].second);
                for (size_t i = 0; i < ga.size(); ++i) x.gA[i] += ga[i];
                for (size_t i = 0; i < gb.size(); ++i) x.gB[i] += gb[i];
            }
            GX[io.out].clear();
            GX[io.out].shrink_to_fit();
        }
        return loss;
    }

    int route_mode = 0;   // gates: 0 = free routing, 1 = record each visit's top-k ids, 2 = replay them
    std::vector<std::vector<int32_t>> route_ids;
    ggml_tensor* route_rec = nullptr;

    // --init-from <adapter.gguf>: start from a saved adapter (same targets + rank; alpha taken from the file)
    void init_from(const std::string& path) {
        ggml_context* c = nullptr;
        gguf_init_params gp = {false, &c};
        gguf_context* g = gguf_init_from_file(path.c_str(), gp);
        if (!g) die("--init-from: cannot read " + path);
        const int64_t ka = gguf_find_key(g, "adapter.lora.alpha");
        if (ka >= 0) { a.alpha = gguf_get_val_f32(g, ka); scale = a.alpha / (float) a.rank; }
        int n = 0;
        for (Adapter& x : ad) {
            const std::string base = ggml_get_name(x.W);
            ggml_tensor* ta = ggml_get_tensor(c, (base + ".lora_a").c_str());
            ggml_tensor* tb = ggml_get_tensor(c, (base + ".lora_b").c_str());
            if (!ta || !tb) die("--init-from: no adapter for " + base);
            if ((size_t) ggml_nelements(ta) != x.A.size() || (size_t) ggml_nelements(tb) != x.B.size()) die("--init-from: shape mismatch " + base);
            std::memcpy(x.A.data(), ta->data, x.A.size() * 4);
            std::memcpy(x.B.data(), tb->data, x.B.size() * 4);
            ++n;
        }
        std::fprintf(stderr, "[train] --init-from %s: %d adapters, alpha %.3g\n", path.c_str(), n, a.alpha);
        gguf_free(g);
        ggml_free(c);
    }

    void zero_grads() { for (Adapter& x : ad) { std::fill(x.gA.begin(), x.gA.end(), 0.0f); std::fill(x.gB.begin(), x.gB.end(), 0.0f); } }
    double grad_norm() const {
        double s = 0;
        for (const Adapter& x : ad) { for (float v : x.gA) s += (double) v * v; for (float v : x.gB) s += (double) v * v; }
        return std::sqrt(s);
    }
    int zero_grad_adapters(bool b_only) const {   // adapters whose gradient is exactly zero (A is 0 while B is 0 at init)
        int n = 0;
        for (const Adapter& x : ad) {
            bool zb = true, za = true;
            for (float v : x.gB) if (v != 0.0f) { zb = false; break; }
            for (float v : x.gA) if (v != 0.0f) { za = false; break; }
            if (zb || (!b_only && za)) ++n;
        }
        return n;
    }
    void adamw(long step, float lr) {
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        const double gn = grad_norm();
        const float cf = gn > a.clip ? (float) (a.clip / gn) : 1.0f;
        const float c1 = 1.0f - std::pow(b1, (float) step), c2 = 1.0f - std::pow(b2, (float) step);
        auto upd = [&](std::vector<float>& p, const std::vector<float>& g, std::vector<float>& mv, std::vector<float>& vv) {
            for (size_t i = 0; i < p.size(); ++i) {
                const float gi = g[i] * cf;
                mv[i] = b1 * mv[i] + (1 - b1) * gi;
                vv[i] = b2 * vv[i] + (1 - b2) * gi * gi;
                p[i] -= lr * (mv[i] / c1) / (std::sqrt(vv[i] / c2) + eps);
            }
        };
        for (Adapter& x : ad) { upd(x.A, x.gA, x.mA, x.vA); upd(x.B, x.gB, x.mB, x.vB); }
    }
    float lr_at(long step) const {
        if (step <= a.warmup) return a.lr * (float) step / (float) std::max(1, a.warmup);
        const double p = (double) (step - a.warmup) / std::max(1, a.steps - a.warmup);
        return (float) (a.lr * (a.lr_min + (1 - a.lr_min) * 0.5 * (1 + std::cos(M_PI * std::min(1.0, p)))));
    }
    // optimizer state next to an adapter: "<adapter>.state" = magic, version, step, then per adapter its name and the
    // AdamW moments (mA vA mB vB). With the adapter it resumes a run exactly (the data order is a function of seed + step).
    void save_state(const std::string& path, long step) const {
        const std::string tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) die("--train: cannot write " + tmp);
        const uint32_t hd[4] = {0x534C4958u /* "XILS" */, 1u, (uint32_t) ad.size(), (uint32_t) a.rank};
        const int64_t st = step;
        std::fwrite(hd, sizeof hd, 1, f); std::fwrite(&st, sizeof st, 1, f);
        for (const Adapter& x : ad) {
            const std::string n = ggml_get_name(x.W);
            const uint32_t ln = (uint32_t) n.size();
            std::fwrite(&ln, 4, 1, f); std::fwrite(n.data(), 1, ln, f);
            for (const std::vector<float>* v : {&x.mA, &x.vA, &x.mB, &x.vB}) std::fwrite(v->data(), 4, v->size(), f);
        }
        if (std::fclose(f) != 0 || std::rename(tmp.c_str(), path.c_str()) != 0) die("--train: cannot write " + path);
    }
    long load_state(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return -1;
        uint32_t hd[4]; int64_t st = 0;
        if (std::fread(hd, sizeof hd, 1, f) != 1 || hd[0] != 0x534C4958u || hd[1] != 1u || hd[2] != ad.size() || hd[3] != (uint32_t) a.rank ||
            std::fread(&st, sizeof st, 1, f) != 1) die("--resume: " + path + " does not match this run (adapters / rank)");
        for (Adapter& x : ad) {
            uint32_t ln = 0;
            std::string n;
            if (std::fread(&ln, 4, 1, f) != 1 || ln > 4096) die("--resume: bad state file " + path);
            n.resize(ln);
            if (std::fread(n.data(), 1, ln, f) != ln || n != ggml_get_name(x.W)) die("--resume: state file adapter order differs: " + n);
            for (std::vector<float>* v : {&x.mA, &x.vA, &x.mB, &x.vB})
                if (std::fread(v->data(), 4, v->size(), f) != v->size()) die("--resume: short state file " + path);
        }
        std::fclose(f);
        return (long) st;
    }
    // --save P [--save-every N]: P-step000400.gguf kept per checkpoint, P = the latest adapter, P.state = its optimizer state
    void checkpoint(long step) {
        const std::string stem = a.save.size() > 5 && a.save.compare(a.save.size() - 5, 5, ".gguf") == 0 ? a.save.substr(0, a.save.size() - 5) : a.save;
        char sfx[32];
        std::snprintf(sfx, sizeof sfx, "-step%06ld.gguf", step);
        save(stem + sfx);
        save(a.save + ".tmp");
        if (std::rename((a.save + ".tmp").c_str(), a.save.c_str()) != 0) die("--train: cannot write " + a.save);
        save_state(a.save + ".state", step);
        std::fprintf(stderr, "[train] checkpoint step %ld: %s%s (+ %s, %s.state)\n", step, stem.c_str(), sfx, a.save.c_str(), a.save.c_str());
    }
    void save(const std::string& path) {
        std::vector<LoraTensorPair> v;
        for (const Adapter& x : ad) {
            LoraTensorPair p;
            p.name = ggml_get_name(x.W); p.n_in = x.n_in; p.n_out = x.n_out; p.rank = a.rank; p.A = x.A; p.B = x.B;
            v.push_back(std::move(p));
        }
        std::string err;
        if (!xolotl::train::write_lora_gguf(path, "inkling2_loop", a.alpha, v, err)) die("--train: " + err);
        std::fprintf(stderr, "[train] adapter saved: %s (%zu tensors pairs)\n", path.c_str(), v.size());
    }
};

// ---- gates ---------------------------------------------------------------------------------------------------------
// --check T: the train graph chain (per-slot graphs, no cache) vs the engine's own forward over the same tokens
static int gate_check(Trainer& tr, const Window& w0, int T) {
    Model& m = tr.m;
    // --check-prefix P (W9): the first P ids are the prefix (carried engine state), the next T the window
    const size_t P = (size_t) std::max(0, tr.a.check_prefix);
    if (P + 2 > w0.ids.size()) die("--check-prefix: the window is too short");
    std::vector<int32_t> pre(w0.ids.begin(), w0.ids.begin() + P);
    std::vector<int32_t> ids(w0.ids.begin() + P, w0.ids.begin() + std::min<size_t>(P + T, w0.ids.size()));
    T = (int) ids.size();
    tr.swa_full = !std::getenv("XOLOTL_INKLING_CHECK_TAIL");   // =1: the training layout (sliding tail), to compare in f32
    tr.set_prefix(P ? &pre : nullptr, false);   // the reference (unfused) kernels on both sides
    std::vector<std::vector<float>> X;
    const std::vector<float> xf = tr.forward_streams(ids, X);
    G1 gg(4096);
    ggml_tensor* tx = gg.f32(xf.data(), tr.h.n_embd, T);
    ggml_tensor* lg = tr.logits_of(gg.c, ggml_reshape_2d(gg.c, tx, tr.h.n_embd, T));
    gg.run(tr.be, nullptr, {lg});
    const std::vector<float> a = fetch(lg);
    if (const char* lp = std::getenv("XOLOTL_INKLING_CHECK_LOGITS")) {   // the train chain's logits [T][n_vocab], for --dump compares
        if (FILE* f = std::fopen(lp, "wb")) { std::fwrite(a.data(), 4, a.size(), f); std::fclose(f); }
    }
    if (!tr.a.init_from.empty() && m.lora.empty()) load_lora(m, tr.a.init_from, 1.0f);   // W7: the engine's --lora path
    if (!P) state_reset(m);   // with a prefix the engine continues from the state the train side read
    std::vector<float> b;
    std::vector<std::vector<float>> taps;
    forward(m, ids.data(), T, (int64_t) P, T, b, &taps);
    for (const SlotIO& io : tr.plan) {   // per visit: the train chain's stream vs the engine's l_out
        if ((size_t) io.il >= taps.size() || taps[io.il].empty()) continue;
        const std::vector<float>& x = X[io.out];
        const std::vector<float>& y = taps[io.il];
        double num = 0, den = 0;
        for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) { num += (x[i] - y[i]) * (x[i] - y[i]); den += (double) y[i] * y[i]; }
        std::printf("  slot %2d (phys %2d%s%s): stream rel err %.3e\n", io.il, io.ip, tr.h.swa[io.ip] ? ", swa" : ", global",
                    io.bank_out >= 0 ? ", loop msg" : "", std::sqrt(num / std::max(den, 1e-30)));
    }
    const int64_t V = tr.h.n_vocab;
    const int64_t Vb = (int64_t) b.size() / T;
    double worst = 0, sum = 0;
    int agree = 0;
    for (int t = 0; t < T; ++t) {
        double num = 0, den = 0;
        int ia = 0, ib = 0;
        for (int64_t v = 0; v < V; ++v) {
            const double x = a[(size_t) t * V + v], y = b[(size_t) t * Vb + v];
            num += (x - y) * (x - y); den += y * y;
            if (x > a[(size_t) t * V + ia]) ia = (int) v;
            if (y > b[(size_t) t * Vb + ib]) ib = (int) v;
        }
        const double r = std::sqrt(num / std::max(den, 1e-30));
        worst = std::max(worst, r); sum += r; agree += ia == ib;
    }
    std::printf("CHECK T=%d prefix %zu: logits rel err mean %.3e worst %.3e, top-1 agree %d/%d\n", T, P, sum / T, worst, agree, T);
    std::fflush(stdout);
    return worst < 1e-2 && agree >= T - 1 ? 0 : 1;
}

// --fd: directional finite difference over every adapter parameter (B random so A matters): <g, d> vs FD at eps
static int gate_fd(Trainer& tr, const Window& w) {
    // Directional FD along the gradient itself: d = g_S / |g_S| over a subset S of the adapters, so the analytic
    // directional derivative is |g_S| (a random direction over ~10M params gives <g,d> ~ |g|/sqrt(N), lost in the
    // forward's noise). S = all adapters, then each target type; a gradient that is wrong for one target shows there.
    // XOLOTL_INKLING_FD_FREE_ROUTING=1 keeps routing live; default = the base forward's top-k ids replayed (a perturbed
    // forward that flips an expert choice makes the loss jump, which the gradient - routing held fixed - cannot see)
    const bool frozen = std::getenv("XOLOTL_INKLING_FD_FREE_ROUTING") == nullptr;
    tr.route_ids.assign(tr.plan.size(), {});
    tr.zero_grads();
    if (frozen) {
        tr.route_mode = 1;
        const double Lf = tr.step_window(w, false);
        tr.route_mode = 2;
        const double Lr = tr.step_window(w, false);
        std::printf("free-routing loss %.7f, replayed %.7f\n", Lf, Lr);
    }
    const double L0 = tr.step_window(w, true);
    std::printf("routing %s\n", frozen ? "frozen (base top-k replayed)" : "free");
    std::vector<std::vector<float>> gA(tr.ad.size()), gB(tr.ad.size());
    for (size_t k = 0; k < tr.ad.size(); ++k) { gA[k] = tr.ad[k].gA; gB[k] = tr.ad[k].gB; }
    // forward noise floor: the same parameters twice more
    const double L1 = tr.step_window(w, false), L2 = tr.step_window(w, false);
    std::printf("FD loss %.7f (repeat %.7f %.7f -> noise %.2e), |g| = %.4e, zero-grad adapters %d/%zu\n", L0, L1, L2,
                std::max(std::abs(L1 - L0), std::abs(L2 - L0)), tr.grad_norm(), tr.zero_grad_adapters(false), tr.ad.size());
    std::vector<std::string> sets = {"all"};
    for (const auto& x : tr.ad) if (std::find(sets.begin(), sets.end(), x.tgt) == sets.end()) sets.push_back(x.tgt);
    int fails = 0;
    for (const std::string& S : sets) {
        auto in = [&](size_t k) { return S == "all" || tr.ad[k].tgt == S; };
        double nrm = 0;
        for (size_t k = 0; k < tr.ad.size(); ++k) if (in(k)) {
            for (float v : gA[k]) nrm += (double) v * v;
            for (float v : gB[k]) nrm += (double) v * v;
        }
        nrm = std::sqrt(nrm);
        if (nrm == 0) { std::printf("  [%s] |g| = 0\n", S.c_str()); ++fails; continue; }
        auto shift = [&](double s) {
            const double c = s / nrm;
            for (size_t k = 0; k < tr.ad.size(); ++k) if (in(k)) {
                for (size_t i = 0; i < gA[k].size(); ++i) tr.ad[k].A[i] += (float) (c * gA[k][i]);
                for (size_t i = 0; i < gB[k].size(); ++i) tr.ad[k].B[i] += (float) (c * gB[k][i]);
            }
        };
        double best = 1e30;
        std::printf("  [%s] |g_S| = %.6e:", S.c_str(), nrm);
        for (double eps : {0.1, 0.03, 0.01, 0.003}) {
            shift(eps);
            const double Lp = tr.step_window(w, false);
            shift(-2 * eps);
            const double Lm = tr.step_window(w, false);
            shift(eps);
            const double fd = (Lp - Lm) / (2 * eps);
            const double rel = std::abs(fd - nrm) / nrm;
            best = std::min(best, rel);
            std::printf("  eps %g FD %.6e (%.1e)", eps, fd, rel);
        }
        std::printf("  -> best %.2e %s\n", best, best < 0.05 ? "OK" : "FAIL");
        std::fflush(stdout);
        fails += best >= 0.05;
    }
    return fails == 0 && tr.zero_grad_adapters(false) == 0 ? 0 : 1;
}

}  // namespace itrain
}  // namespace

int inkling_train_main(int argc, char** argv) {
    using namespace itrain;
    Args a;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) die("--train: " + k + " needs a value"); return argv[++i]; };
        if (k == "--data") a.data = v();
        else if (k == "--eval") a.eval = v();
        else if (k == "--out") a.out = v();
        else if (k == "--run") a.run = v();
        else if (k == "--save") a.save = v();
        else if (k == "--targets") a.targets = v();
        else if (k == "--rank") a.rank = std::atoi(v().c_str());
        else if (k == "--alpha") a.alpha = (float) std::atof(v().c_str());
        else if (k == "--lr") a.lr = (float) std::atof(v().c_str());
        else if (k == "--lr-min") a.lr_min = (float) std::atof(v().c_str());
        else if (k == "--steps") a.steps = std::atoi(v().c_str());
        else if (k == "--warmup") a.warmup = std::atoi(v().c_str());
        else if (k == "--max-t") a.max_t = std::atoi(v().c_str());
        else if (k == "--threads") a.threads = std::atoi(v().c_str());
        else if (k == "--eval-every") a.eval_every = std::atoi(v().c_str());
        else if (k == "--eval-n") a.eval_n = std::atoi(v().c_str());
        else if (k == "--save-every") a.save_every = std::atoi(v().c_str());
        else if (k == "--seed") a.seed = std::atoi(v().c_str());
        else if (k == "--clip") a.clip = (float) std::atof(v().c_str());
        else if (k == "--init-b") a.init_b = (float) std::atof(v().c_str());
        else if (k == "--ce-chunk") a.ce_chunk = std::max(16, std::atoi(v().c_str()));
        else if (k == "--check") a.check = std::atoi(v().c_str());
        else if (k == "--init-from") a.init_from = v();
        else if (k == "--resume") a.resume = v();
        else if (k == "--prefix") a.prefix_file = v();
        else if (k == "--max-prefix") a.max_prefix = std::atoi(v().c_str());
        else if (k == "--check-prefix") a.check_prefix = std::atoi(v().c_str());
        else if (k == "--no-shuffle") a.shuffle = 0;
        else if (k == "--fd") a.fd = 1;
        else if (k == "--overfit") a.overfit = std::atoi(v().c_str());
        else if (a.gguf.empty() && k[0] != '-') a.gguf = k;
        else die("--train: unknown option " + k);
    }
    if (a.gguf.empty() || a.data.empty()) {
        std::fprintf(stderr, "usage: xolotl-inkling --train <model.gguf> --data windows.jsonl [--eval heldout.jsonl] [--steps N]\n"
                             "       [--rank 16] [--alpha 32] [--lr 1e-4] [--targets q,k,v,o,gate,rel] [--max-t 2048] [--save a.gguf]\n"
                             "       gates: --check T [--init-from ADAPTER] | --fd [--init-b 0.01] | --overfit N\n"
                             "       checkpoints: --save P --save-every N (P-stepNNNNNN.gguf kept, P = latest, P.state = optimizer)\n"
                             "       --resume P continues a run (weights + optimizer state + step); SIGTERM/Ctrl-C checkpoints first\n");
        return 2;
    }
    // the reference graph only: every fused (custom) op off, controllers on their base tensors
    inkling_train_mode() = true;
    setenv("XOLOTL_INKLING_CTRL_REF", "1", 1);
    setenv("XOLOTL_INKLING_FUSED", "0", 1);
    setenv("XOLOTL_INKLING_PROMPT_ATTN", "0", 1);
    setenv("XOLOTL_INKLING_DECATTN", "0", 1);
    setenv("XOLOTL_INKLING_STACKQ", "0", 1);
    try {
        Model m;
        m.ubatch = std::max(512, std::min(a.max_t, 4096));   // prefill chunks / check windows (the ring holds n_swa + this)
        // --check-prefix: the gate's sliding slots read the whole prefix in ring order, so the ring must hold prefix + window
        // (load() sizes the ring from ubatch; the prefill chunks keep the normal ubatch, restored after the load)
        const int ub0 = m.ubatch;
        if (a.check && a.check_prefix > 0) m.ubatch = std::max<int>(m.ubatch, a.check_prefix + std::min(a.max_t, 4096));
        // the engine holds prefixes (prefill chunks) + a check window; windows themselves never enter its cache
        load(m, a.gguf, std::max<int64_t>(4096, (int64_t) a.max_prefix + std::max(a.check_prefix, 0) + 2 * std::min(a.max_t, 4096) + 512), a.threads, "", false);
        m.ubatch = ub0;
        if (a.check && a.check_prefix > 0) {   // gate: short prefill chunks (few attention chunks a slot over the big ring) and a
            m.ubatch = std::min(ub0, 1024);     // budget that keeps the check window's attention one softmax, like the train side
            m.attn_budget = std::max<int64_t>(m.attn_budget, ((int64_t) a.check_prefix + a.check + 255) / 256 * 256 * a.check * 32);
        }
        if (ggml_backend_buffer_is_host(m.output->buffer)) die("--train: the head weight is not on the GPU");
        std::vector<Window> tw = load_windows(a.data, a.max_t), ew;
        if (tw.empty()) die("--train: no windows in " + a.data);
        if (!a.eval.empty()) ew = load_windows(a.eval, a.max_t);
        std::fprintf(stderr, "[train] %zu train windows, %zu eval windows\n", tw.size(), ew.size());
        Trainer tr(m, a);
        if (!a.prefix_file.empty()) {   // {"tokens": [...]} (harness_build*.py prefix.json) or a bare id array
            std::ifstream pf(a.prefix_file);
            if (!pf) die("--prefix: cannot read " + a.prefix_file);
            std::stringstream ss; ss << pf.rdbuf();
            std::string js = ss.str();
            std::vector<long long> v;
            if (!parse_int_array(js, "tokens", v)) {   // a bare array
                if (!parse_int_array("{\"tokens\": " + js + "}", "tokens", v)) die("--prefix: no token ids in " + a.prefix_file);
            }
            tr.global_prefix = std::make_shared<std::vector<int32_t>>(v.begin(), v.end());
            if ((int64_t) v.size() > a.max_prefix) die("--prefix: " + std::to_string(v.size()) + " tokens; pass --max-prefix >= that");
            std::fprintf(stderr, "[train] --prefix %s: %zu tokens before every window\n", a.prefix_file.c_str(), v.size());
        }
        if (!a.init_from.empty()) tr.init_from(a.init_from);
        long start = 0;   // --resume <adapter.gguf>: its weights + "<adapter>.state" (AdamW moments, step); the run continues
        if (!a.resume.empty()) {
            tr.init_from(a.resume);
            start = tr.load_state(a.resume + ".state");
            if (start < 0) {   // a numbered checkpoint has no state: weights only, step from the name, moments restart at 0
                const size_t q = a.resume.rfind("-step");
                start = q == std::string::npos ? 0 : std::atol(a.resume.c_str() + q + 5);
                std::fprintf(stderr, "[train] --resume: no %s.state - weights only, AdamW moments restart, step %ld\n", a.resume.c_str(), start);
            } else {
                std::fprintf(stderr, "[train] --resume: step %ld (weights + optimizer state)\n", start);
            }
            if (start >= a.steps) die("--resume: the run already reached step " + std::to_string(start) + " (--steps " + std::to_string(a.steps) + ")");
        }
        if (a.check) return gate_check(tr, tw[0], a.check);
        if (a.fd) return gate_fd(tr, tw[0]);

        char cfg[1024];
        std::snprintf(cfg, sizeof cfg, "{\"model\": \"%s\", \"data\": \"%s\", \"rank\": %d, \"alpha\": %g, \"lr\": %g, \"targets\": \"%s\", \"max_t\": %d, \"steps\": %d}",
                      a.gguf.c_str(), a.data.c_str(), a.rank, a.alpha, a.lr, a.targets.c_str(), a.max_t, a.steps);
        xolotl::train::Metrics met(a.out, a.run, cfg);
        auto eval_loss = [&]() -> double {
            if (ew.empty()) return NAN;
            double s = 0;
            const int n = std::min<int>(a.eval_n, (int) ew.size());
            static const bool each = std::getenv("XOLOTL_INKLING_EVAL_EACH") != nullptr;   // gates: every window's loss
            for (int i = 0; i < n; ++i) {
                const double l = tr.step_window(ew[i], false);
                if (each) std::fprintf(stderr, "[train] eval window %d loss %.9f (prefill %lld tok, %.1f s so far)\n", i, l,
                                       (long long) tr.prefill_tok, tr.prefill_s);
                s += l;
            }
            return s / n;
        };
        if (a.overfit) { a.steps = a.overfit; tr.a.steps = a.overfit; }
        std::mt19937 rng(a.seed);
        std::vector<size_t> order(tw.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        if (a.shuffle) std::shuffle(order.begin(), order.end(), rng);   // --no-shuffle: file order (windows sharing a prefix stay together)
        const double e0 = eval_loss();
        if (std::isfinite(e0)) { std::fprintf(stderr, "[train] step %ld eval %.4f\n", start, e0); met.log(start, {{"eval_loss", e0}}); }
        double first = NAN;
        static volatile sig_atomic_t g_stop = 0;   // SIGTERM (systemctl stop) / SIGINT: checkpoint after this step, exit 0
        if (!a.save.empty()) { auto on_sig = [](int) { g_stop = 1; }; signal(SIGTERM, on_sig); signal(SIGINT, on_sig); }
        for (long step = start + 1; step <= a.steps; ++step) {
            const Window& w = a.overfit ? tw[0] : tw[order[(size_t) (step - 1) % order.size()]];
            const auto t0 = std::chrono::steady_clock::now();
            tr.zero_grads();
            const double loss = tr.step_window(w, true);
            if (!std::isfinite(first)) first = loss;
            const double gn = tr.grad_norm();
            const float lr = tr.lr_at(step);
            const int zb = tr.zero_grad_adapters(true);
            tr.adamw(step, lr);
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[train] step %ld loss %.4f gnorm %.3e lr %.2e T %zu %.1f s%s\n", step, loss, gn, lr, w.ids.size(), dt,
                         zb ? (" ZERO-GRAD-B " + std::to_string(zb)).c_str() : "");
            double el = NAN;
            if (a.eval_every && step % a.eval_every == 0) { el = eval_loss(); std::fprintf(stderr, "[train] step %ld eval %.4f\n", step, el); }
            met.log(step, {{"loss", loss}, {"eval_loss", el}, {"lr", lr}, {"gnorm", gn}, {"window_tokens", (double) w.ids.size()}, {"step_s", dt}});
            if (!a.save.empty() && ((a.save_every && step % a.save_every == 0) || step == a.steps || g_stop)) tr.checkpoint(step);
            if (g_stop) { std::fprintf(stderr, "[train] stopped by signal after step %ld (resume: --resume %s)\n", step, a.save.c_str()); return 0; }
        }
        if (a.overfit) {
            const double last = tr.step_window(tw[0], false);
            std::printf("OVERFIT %d steps: loss %.4f -> %.4f\n", a.overfit, first, last);
            return last < 0.7 * first ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xolotl-inkling --train: %s\n", e.what());
        return 1;
    }
    return 0;
}
