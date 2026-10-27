// MiniMax-M3.1 text engine. Architecture source: llama.cpp PR #31192,
// src/models/minimax-m3.cpp at 6b95a9908d2c7b242817a447708aec38e937e2c0 (MIT).
// Append-only, one sequence, F16 caches. This deliberately starts with stock ggml ops.
#include "m31_engine.hpp"
#include "m31_arena.hpp"
#include "gpu/experts.hpp"
#include "cpu/fused_decode.hpp"
#include "cpu/threadpool_config.hpp"
#include "prefill/plan.hpp"
#include "prefill/chunk_limit.hpp"
#include "gpu/prefill_hidden.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#if __has_include("ggml-impl.h")
#include "ggml-impl.h"
#define M31_ENGINE_GRAPH_UID 1
#endif
#endif

#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <utility>
#ifdef M31_SHARED_OVERLAP_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

namespace strata::m31 {
namespace {

void require(bool ok, const std::string & reason) {
    if (!ok) throw std::runtime_error("M31: " + reason);
}
int64_t padded(int64_t n, int64_t align) { return (n + align - 1) / align * align; }
// PR's MSA position axis and main KV window both have a minimum 256-cell pad.
// Matching it also keeps FA's kernel/tiling choice identical for numerical parity.
int64_t kv_length(int64_t n, const Geometry & g) { return padded(n, std::max<int64_t>(256, g.block)); }

struct Weights {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    std::map<std::string, ggml_tensor *> tensors;
    std::unique_ptr<ExpertArena> arena;
    ~Weights() {
        for (auto b : buffers) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        require(it != tensors.end(), "missing loaded tensor " + name);
        return it->second;
    }
    ggml_tensor * maybe(const std::string & name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
    void load(const GgufModel & model, ggml_backend_t backend, bool dense_gpu,
              const Geometry * geometry = nullptr, double arena_gib = 0, const Config & config = Config{}) {
        if (!dense_gpu && arena_gib > 0) {
            require(geometry != nullptr, "arena needs validated geometry");
            arena = std::make_unique<ExpertArena>();
            arena->init(model, *geometry, arena_gib, config);
        }
        ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "weight metadata allocation failed");
        for (size_t sh = 0; sh < model.size(); ++sh) {
            const auto & file = model.shard(sh);
            ggml_backend_buffer_t buffer = nullptr;
            if (!dense_gpu && !file.tensors().empty()) {
                const auto & first = file.tensors().front();
                uint8_t * base = const_cast<uint8_t *>(file.tensor_data(first)) - file.data_start() - first.offset;
                buffer = ggml_backend_cpu_buffer_from_ptr(base, size_t(file.file_size()));
                require(buffer != nullptr, "cannot wrap GGUF mapping");
                buffers.push_back(buffer);
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            for (const auto & ti : file.tensors()) {
                if (dense_gpu && (expert_tensor(ti.name) || ti.name == "token_embd.weight")) continue;
                require(model.in_bounds(ti, sh), "out-of-bounds weight " + ti.name);
                int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
                require(ti.shape.size() <= GGML_MAX_DIMS, "weight rank too large");
                for (size_t d = 0; d < ti.shape.size(); ++d) ne[d] = int64_t(ti.shape[d]);
                require(ti.type < GGML_TYPE_COUNT, "ggml does not know type " + ti.name);
                auto t = ggml_new_tensor(ctx, ggml_type(ti.type), int(ti.shape.size()), ne);
                ggml_set_name(t, ti.name.c_str());
                tensors.emplace(ti.name, t);
                if (!dense_gpu) {
                    const auto * packed = arena ? arena->find(ti.name) : nullptr;
                    if (packed) {
                        t->nb[2] = packed->expert_stride;
                        t->nb[3] = packed->expert_stride * size_t(t->ne[2]);
                    }
                    require(ggml_backend_tensor_alloc(packed ? arena->buffer() : buffer, t,
                                packed ? packed->data : const_cast<uint8_t *>(file.tensor_data(ti))) == GGML_STATUS_SUCCESS,
                            "cannot bind " + ti.name);
                }
            }
        }
        if (dense_gpu) {
            // Pack unchanged rows of compatible projections once at load. Original
            // named tensors alias their exact bytes; prefill retains its old matmuls.
            const auto buft = ggml_backend_get_default_buffer_type(backend);
            std::vector<ggml_tensor *> bundles;
            struct Part { ggml_tensor * bundle; size_t offset; };
            std::map<ggml_tensor *, Part> parts;
            bool concatenate = false;
#ifdef M31_ENGINE_CUDA
            concatenate = ggml_backend_is_cuda(backend);
#endif
            auto pack = [&](const std::string & key, std::initializer_list<std::string> names) {
                std::vector<ggml_tensor *> rows;
                int64_t total_rows = 0;
                for (const auto & name : names) {
                    auto t = maybe(name);
                    if (!t || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) ||
                        ggml_backend_buft_get_alloc_size(buft, t) != ggml_nbytes(t)) return;
                    if (!rows.empty() && (t->type != rows.front()->type || t->ne[0] != rows.front()->ne[0])) return;
                    rows.push_back(t); total_rows += t->ne[1];
                }
                auto bundle = ggml_new_tensor_2d(ctx, rows.front()->type, rows.front()->ne[0], total_rows);
                ggml_set_name(bundle, key.c_str());
                size_t offset = 0;
                for (auto t : rows) { parts.emplace(t, Part{bundle, offset}); offset += ggml_nbytes(t); }
                tensors.emplace(key, bundle); bundles.push_back(bundle);
            };
            if (concatenate && geometry) for (int l = 0; l < geometry->layers; ++l) {
                const auto prefix = "blk." + std::to_string(l) + ".";
                pack(prefix + "m31_qkv.weight", {prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight"});
                pack(prefix + "m31_index_qk.weight", {prefix + "indexer.q_proj.weight", prefix + "indexer.k_proj.weight"});
                const std::string suffix = geometry->routed(l) ? "_shexp" : "";
                pack(prefix + "m31_ffn_gu" + suffix + ".weight", {prefix + "ffn_gate" + suffix + ".weight", prefix + "ffn_up" + suffix + ".weight"});
            }
            std::vector<ggml_tensor *> allocation;
            for (auto t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
                if (!parts.count(t)) allocation.push_back(t);
            auto buffer = ggml_backend_buft_alloc_buffer_n(buft, allocation.data(), int(allocation.size()));
            require(buffer != nullptr, "cannot allocate GPU dense weights");
            buffers.push_back(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (const auto & [tensor, part] : parts)
                require(ggml_backend_tensor_alloc(part.bundle->buffer, tensor,
                    static_cast<uint8_t *>(part.bundle->data) + part.offset) == GGML_STATUS_SUCCESS,
                    "cannot bind packed projection rows");
            if (concatenate) std::fprintf(stderr, "M31 unchanged projection bundles: groups=%zu aliased_weights=%zu (decode only)\n", bundles.size(), parts.size());
            for (size_t sh = 0; sh < model.size(); ++sh)
                for (const auto & ti : model.shard(sh).tensors()) {
                    auto it = tensors.find(ti.name);
                    if (it != tensors.end())
                        ggml_backend_tensor_set(it->second, model.shard(sh).tensor_data(ti), 0, ggml_nbytes(it->second));
                }
        }
    }
};

struct Workspace {
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit Workspace(ggml_backend_t backend) {
        context = ggml_init({32 * 1024 * 1024, nullptr, true});
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!context || !allocator) {
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
            throw std::runtime_error("M31: workspace allocation failed");
        }
    }
    ~Workspace() {
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
    }
};

// Shared, immutable during a chunk: all layers have the same causal positions.
// Keep these outside transient graph scratch so the large mask is uploaded once.
struct AttentionInputs {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * positions = nullptr, * mask = nullptr, * bias = nullptr, * blocks = nullptr;
    int n = 0;
    int64_t nk = 0;
    ~AttentionInputs() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
    AttentionInputs(ggml_backend_t backend, const Geometry & g, int count, int64_t keys, bool f16_causal = false) : n(count), nk(keys) {
        context = ggml_init({64 * 1024, nullptr, true});
        require(context != nullptr, "shared attention input metadata allocation failed");
        positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, n);
        mask = ggml_new_tensor_3d(context, f16_causal ? GGML_TYPE_F16 : GGML_TYPE_F32, nk, 1, n);
        bias = ggml_new_tensor_3d(context, GGML_TYPE_F32, nk / g.block, 1, n);
        blocks = ggml_new_tensor_1d(context, GGML_TYPE_I32, nk);
        for (auto t : {positions, mask, bias, blocks}) ggml_set_input(t);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, ggml_backend_get_default_buffer_type(backend));
        if (!buffer) {
            ggml_free(context);
            throw std::runtime_error("M31: shared attention input allocation failed");
        }
    }
};

struct Graph {
    ggml_context * c = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;
    const Config & config;
    int64_t pos0 = 0;
    std::vector<std::pair<std::string, ggml_tensor *>> taps;
    bool borrowed = false;
    Graph(ggml_backend_t b, const Config & cfg, int64_t pos, Workspace * workspace = nullptr,
          size_t metadata_bytes = 32 * 1024 * 1024, size_t graph_nodes = 8192)
        : backend(b), config(cfg), pos0(pos), borrowed(workspace != nullptr) {
        if (workspace) {
            c = workspace->context;
            ggml_reset(c);
            allocator = workspace->allocator;
        } else {
            c = ggml_init({metadata_bytes, nullptr, true});
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        if (!c || !allocator) {
            if (!borrowed) {
                if (allocator) ggml_gallocr_free(allocator);
                if (c) ggml_free(c);
            }
            throw std::runtime_error("M31: graph allocation failed");
        }
        graph = ggml_new_graph_custom(c, graph_nodes, false);
    }
    ~Graph() {
#ifdef M31_ENGINE_CUDA
        // The patched backend keys captured graphs by node addresses. Those addresses can be
        // reused by the next transient context: discard captures BEFORE freeing their buffers.
        ggml_backend_cuda_graph_cache_clear(backend);
#endif
        if (!borrowed) {
            if (allocator) ggml_gallocr_free(allocator);
            if (c) ggml_free(c);
        }
    }
    ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        auto t = ggml_new_tensor_3d(c, type, n0, n1, n2);
        ggml_set_input(t);
        return t;
    }
    ggml_tensor * tap(ggml_tensor * t, const std::string & base, int l = -1) {
        if (config.tap) {
            const std::string name = l < 0 ? base : base + "-" + std::to_string(l);
            ggml_set_output(t);
            // A view's source can later be overwritten/reused; keep the view as an explicit graph output.
            ggml_build_forward_expand(graph, t);
            taps.emplace_back(name, t);
        }
        return t;
    }
    void allocate(std::initializer_list<ggml_tensor *> outputs) {
        for (auto t : outputs) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
        // MUL output fencing prevents the CUDA fusion regressions observed in the
        // model gates. The exact vendor cause is still unresolved.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL) ggml_set_output(node);
            }
        // A reused allocator can see equal node counts with different aliasing topology.
        // Replan each transient graph while retaining already large scratch buffers.
        if (borrowed) require(ggml_gallocr_reserve(allocator, graph), "compute reservation failed");
        require(ggml_gallocr_alloc_graph(allocator, graph), "compute allocation failed");
#ifdef M31_ENGINE_GRAPH_UID
        // Tensor addresses/strides and op parameters are immutable after allocation.
        // Only input VALUES change, so the CUDA backend can skip full node metadata scans.
        graph->uid = ggml_graph_next_uid();
#endif
    }
    void compute() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        // ggml_backend_graph_compute already synchronizes the backend.
        for (auto & t : taps) config.tap(t.first, pos0, t.second);
    }
    ggml_tensor * borrow(ggml_tensor * source) {
        require(source && ggml_is_contiguous(source), "expected contiguous external graph value");
        auto tensor = ggml_dup_tensor(c, source);
        auto buffer = source->view_src ? source->view_src->buffer : source->buffer;
        require(buffer && ggml_backend_tensor_alloc(buffer, tensor, source->data) == GGML_STATUS_SUCCESS,
                "cannot bind persistent graph value");
        ggml_set_input(tensor);
        return tensor; // leaf alias: never traverses or repeats the producing graph
    }
    static std::vector<float> read(ggml_tensor * t) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "expected contiguous F32 readback");
        std::vector<float> result(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
        return result;
    }
    static void read_into(ggml_tensor * t, std::vector<float> & result) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "expected contiguous F32 readback");
        result.resize(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    }
};

ggml_tensor * rms(Graph & b, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(b.c, ggml_rms_norm(b.c, x, eps), w);
}
ggml_tensor * ffn(Graph & b, const Weights & w, ggml_tensor * x, const std::string & prefix,
                  const std::string & suffix) {
    ggml_tensor * gate = nullptr, * up = nullptr;
    if (auto bundle = x->ne[1] == 1 && x->ne[2] == 1 ? w.maybe(prefix + "m31_ffn_gu" + suffix + ".weight") : nullptr) {
        auto joined = ggml_mul_mat(b.c, bundle, x);
        const auto rows = w.get(prefix + "ffn_gate" + suffix + ".weight")->ne[1];
        gate = ggml_view_2d(b.c, joined, rows, 1, joined->nb[1], 0);
        up = ggml_view_2d(b.c, joined, rows, 1, joined->nb[1], size_t(rows) * sizeof(float));
    } else {
        gate = ggml_mul_mat(b.c, w.get(prefix + "ffn_gate" + suffix + ".weight"), x);
        up = ggml_mul_mat(b.c, w.get(prefix + "ffn_up" + suffix + ".weight"), x);
    }
    // Pin FFN matmul inputs to avoid the observed large-batch fusion regression.
    // Marking only the terminal GLU output does not prevent that fusion.
    // Keep the already validated single-token graph and fusion exactly unchanged.
    if (x->ne[1] > 1) {
        ggml_set_output(gate); ggml_set_output(up);
    }
    auto h = ggml_swiglu_oai(b.c, gate, up, 1.702f, 7.0f);
    return ggml_mul_mat(b.c, w.get(prefix + "ffn_down" + suffix + ".weight"), h);
}

// minimax-m3.cpp::build_attn_msa_fa, single sequence, channel=GQA group.
ggml_tensor * grouped_fa(Graph & b, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                         ggml_tensor * mask, const Geometry & g, int n) {
    const int64_t group = g.heads / g.kv_heads;
    auto q4 = ggml_reshape_4d(b.c, q, g.head_dim, group, g.kv_heads, n);
    q4 = ggml_permute(b.c, q4, 0, 2, 3, 1);
    // This ggml CPU FA ignores F32 accumulation for F16 V and rounds its running
    // sum after every key. Preserve the F16 cache values but accumulate them in F32,
    // as requested below (CUDA already respects the accumulator precision).
    if (ggml_backend_is_cpu(b.backend)) v = ggml_cast(b.c, v, GGML_TYPE_F32);
    auto out = ggml_flash_attn_ext(b.c, q4, k, v, mask, 1.0f / std::sqrt(float(g.head_dim)), 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    out = ggml_cont(b.c, ggml_permute(b.c, out, 0, 1, 3, 2));
    return ggml_reshape_2d(b.c, out, g.head_dim * g.heads, n);
}

} // namespace

struct Engine::Impl {
    Config cfg;
    Geometry g;
    // Declaration order: model mappings outlive the borrowed tensor buffers in the stores.
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    bool own_backend = false;
    ggml_threadpool_t cpu_pool = nullptr;
    std::unique_ptr<Workspace> dense_workspace, cpu_workspace;
    std::unique_ptr<AttentionInputs> attention_inputs;
    std::unique_ptr<GpuPrefillHidden> prefill_hidden;
    bool gpu_prefill_chain = false, prefill_chain_active = false;
    struct RoutedGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * output = nullptr;
    };
    std::vector<std::unique_ptr<RoutedGraph>> cpu_decode_graphs;
    struct DenseGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * positions = nullptr, * mask = nullptr, * bias = nullptr;
        ggml_tensor * residual = nullptr, * norm = nullptr, * shared = nullptr;
        ggml_tensor * ids = nullptr, * weights = nullptr, * output = nullptr;
        int64_t nk = 0;
    };
    struct HeadGraph {
        std::unique_ptr<Graph> graph;
        ggml_tensor * input = nullptr, * output = nullptr;
    };
    std::vector<std::unique_ptr<DenseGraph>> dense_decode_graphs;
    HeadGraph head_decode_graph;
    std::vector<std::unique_ptr<HeadGraph>> shared_decode_graphs;
    bool cpu_shared_overlap = false, shared_inflight = false, shared_trace = false;
    bool gpu_decode_chain = false, dense_async_reads = false;
    bool decode_chain_active = false, native_gather = false, adaptive_prefill = false;
    uint64_t prefill_free_after_resident = 0;
    bool msa_topk_mask = false;
    std::unique_ptr<CpuFusedDecode> cpu_fused;
#ifdef M31_SHARED_OVERLAP_TESTING
    bool shared_failure_fired = false;
#endif
    struct Timings {
        double dense = 0, routed_setup = 0, routed_compute = 0, routed_read = 0, residual = 0;
        double dense_build = 0, dense_allocate = 0, dense_upload = 0, dense_compute = 0, dense_read = 0, dense_destroy = 0;
        double shared_enqueue = 0, shared_wait = 0, shared_read = 0;
        uint64_t shared_batches = 0;
    } timings;
    using Clock = std::chrono::steady_clock;
    static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
    Weights host, device;
    std::unique_ptr<GpuExperts> gpu_experts;
    const Weights * dense = nullptr;
    ggml_context * state = nullptr;
    ggml_backend_buffer_t state_buffer = nullptr;
    struct Cache { ggml_tensor * k = nullptr, * v = nullptr, * index = nullptr; };
    std::vector<Cache> cache;
    int64_t capacity = 0, pos = 0;
    std::shared_ptr<const void> checkpoint_owner = std::make_shared<uint8_t>(0);
    int last_n = 0;
    bool poisoned = false, ready = false;
    std::vector<float> hidden;
    // Opt-in host ownership only: every graph/backend operation stays unchanged.
    // Buffers are private to this serialized Engine and survive layer/reset boundaries.
    bool reuse_host_reads = false;
    struct HostReads {
        std::vector<float> input, residual, shared, weights, moe;
        std::vector<int32_t> ids;
        uint64_t reads = 0, reuses = 0, growths = 0, bytes = 0;
    } host_reads;
    void read_host(ggml_tensor * tensor, std::vector<float> & out) {
        const auto previous_capacity = out.capacity();
        Graph::read_into(tensor, out);
        ++host_reads.reads;
        host_reads.bytes += ggml_nbytes(tensor);
        if (out.capacity() > previous_capacity) ++host_reads.growths;
        else ++host_reads.reuses;
    }
    void read_async(ggml_tensor * tensor, std::vector<float> & out) {
        const auto previous_capacity = out.capacity();
        out.resize(size_t(ggml_nelements(tensor)));
        ggml_backend_tensor_get_async(backend, tensor, out.data(), 0, ggml_nbytes(tensor));
        if (reuse_host_reads) {
            ++host_reads.reads;
            host_reads.bytes += ggml_nbytes(tensor);
            if (out.capacity() > previous_capacity) ++host_reads.growths;
            else ++host_reads.reuses;
        }
    }
    size_t host_retained_bytes() const {
        return (host_reads.input.capacity() + host_reads.residual.capacity() + host_reads.shared.capacity() +
                host_reads.weights.capacity() + host_reads.moe.capacity()) * sizeof(float) +
               host_reads.ids.capacity() * sizeof(int32_t);
    }

    ~Impl() {
        drain_shared();
        if (gpu_experts && cfg.profile) {
            const auto & s = gpu_experts->stats();
            std::fprintf(stderr, "M31 expert cache: hits=%llu misses=%llu evictions=%llu upload_bytes=%llu resident=%zu peak=%zu scratch=%zu pinned=%d\n",
                         (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.evictions,
                         (unsigned long long)s.upload_bytes, s.resident_bytes, s.peak_resident_bytes, s.scratch_bytes, s.pinned_staging);
        }
        gpu_experts.reset();
        cpu_fused.reset();
        cpu_decode_graphs.clear();
        shared_decode_graphs.clear();
        dense_decode_graphs.clear();
        head_decode_graph.graph.reset();
        dense_workspace.reset();
        cpu_workspace.reset();
        attention_inputs.reset();
        prefill_hidden.reset();
        close_weights();
        if (state_buffer) ggml_backend_buffer_free(state_buffer);
        if (state) ggml_free(state);
        if (cpu_pool) {
            if (backend && ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_threadpool(backend, nullptr);
            if (cpu) ggml_backend_cpu_set_threadpool(cpu, nullptr);
        }
        if (own_backend && backend) ggml_backend_free(backend);
        if (cpu) ggml_backend_free(cpu);
        if (cpu_pool) ggml_threadpool_free(cpu_pool);
    }
    void close_weights() {
        for (Weights * w : {&device, &host}) {
            for (auto b : w->buffers) ggml_backend_buffer_free(b);
            w->buffers.clear();
            if (w->ctx) { ggml_free(w->ctx); w->ctx = nullptr; }
            w->tensors.clear();
            w->arena.reset();
        }
    }

    ggml_tensor * gather_half(Graph & b, ggml_tensor * source, ggml_tensor * rows) {
        if (!native_gather) return ggml_cast(b.c, ggml_get_rows(b.c, source, rows), GGML_TYPE_F16);
        // CUDA GET_ROWS already supports F16 destinations. Construct that native
        // op directly: the public helper currently hardcodes a F32 intermediate.
        require(source->ne[2] == rows->ne[1] && source->ne[3] == rows->ne[2] &&
                rows->ne[3] == 1 && rows->type == GGML_TYPE_I32, "invalid F16 gather shape");
        auto result = ggml_new_tensor_4d(b.c, GGML_TYPE_F16, source->ne[0], rows->ne[0], rows->ne[1], rows->ne[2]);
        result->op = GGML_OP_GET_ROWS;
        result->src[0] = source; result->src[1] = rows;
        return result;
    }

    ggml_tensor * attention(Graph & b, int l, int64_t start, int n, ggml_tensor * x,
                             ggml_tensor * positions, ggml_tensor * pos_mask, ggml_tensor * bias,
                             bool indexed_kv_write = false) {
        auto c = b.c;
        const std::string prefix = "blk." + std::to_string(l) + ".";
        auto wt = [&](const char * name) { return dense->get(prefix + name); };
        auto norm = b.tap(rms(b, x, wt("attn_norm.weight"), g.rms_eps), "attn_norm", l);
        ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr;
        if (auto bundle = n == 1 ? dense->maybe(prefix + "m31_qkv.weight") : nullptr) {
            auto joined = ggml_mul_mat(c, bundle, norm);
            auto slice = [&](int64_t rows, int64_t offset) { return ggml_view_2d(c, joined, rows, 1, joined->nb[1], size_t(offset) * sizeof(float)); };
            q = ggml_reshape_3d(c, slice(g.head_dim * g.heads, 0), g.head_dim, g.heads, 1);
            k = ggml_reshape_3d(c, slice(g.head_dim * g.kv_heads, g.head_dim * g.heads), g.head_dim, g.kv_heads, 1);
            v = ggml_reshape_3d(c, slice(g.head_dim * g.kv_heads, g.head_dim * (g.heads + g.kv_heads)), g.head_dim, g.kv_heads, 1);
        } else {
            q = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_q.weight"), norm), g.head_dim, g.heads, n);
            k = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_k.weight"), norm), g.head_dim, g.kv_heads, n);
            v = ggml_reshape_3d(c, ggml_mul_mat(c, wt("attn_v.weight"), norm), g.head_dim, g.kv_heads, n);
        }
        q = b.tap(rms(b, q, wt("attn_q_norm.weight"), g.rms_eps), "Qcur_normed", l);
        k = b.tap(rms(b, k, wt("attn_k_norm.weight"), g.rms_eps), "Kcur_normed", l);
        auto rope = [&](ggml_tensor * t) {
            return ggml_rope_ext(c, t, positions, nullptr, int(g.rope_dim), GGML_ROPE_TYPE_NEOX,
                                 int(g.context), g.rope_base, 1, 0, 1, 32, 1);
        };
        q = b.tap(rope(q), "Qcur", l);
        k = b.tap(rope(k), "Kcur", l);
        b.tap(v, "Vcur", l);
        ggml_tensor * iq = nullptr, * ik = nullptr;
        if (auto bundle = n == 1 ? dense->maybe(prefix + "m31_index_qk.weight") : nullptr) {
            auto joined = ggml_mul_mat(c, bundle, norm);
            iq = ggml_reshape_3d(c, ggml_view_2d(c, joined, g.index_dim * g.index_heads, 1, joined->nb[1], 0), g.index_dim, g.index_heads, 1);
            ik = ggml_reshape_3d(c, ggml_view_2d(c, joined, g.index_dim, 1, joined->nb[1], size_t(g.index_dim * g.index_heads) * sizeof(float)), g.index_dim, 1, 1);
        } else {
            iq = ggml_reshape_3d(c, ggml_mul_mat(c, wt("indexer.q_proj.weight"), norm), g.index_dim, g.index_heads, n);
            ik = ggml_reshape_3d(c, ggml_mul_mat(c, wt("indexer.k_proj.weight"), norm), g.index_dim, 1, n);
        }
        iq = rope(rms(b, iq, wt("indexer.q_norm.weight"), g.rms_eps));
        ik = rope(rms(b, ik, wt("indexer.k_norm.weight"), g.rms_eps));

        auto & kv = cache[size_t(l)];
        if (indexed_kv_write) {
            // Keep pointers and graph topology stable across decode positions. Only
            // the positions tensor contents change. F32 -> F16 conversion is the
            // same storage operation as CPY; cache row order/precision are unchanged.
            for (const auto & item : {std::make_pair(kv.k, k), std::make_pair(kv.v, v)}) {
                auto dst = ggml_reshape_2d(c, item.first, g.head_dim * g.kv_heads, capacity);
                auto src = ggml_reshape_2d(c, item.second, g.head_dim * g.kv_heads, n);
                ggml_build_forward_expand(b.graph, ggml_set_rows(c, dst, src, positions));
            }
            ggml_build_forward_expand(b.graph, ggml_set_rows(c, kv.index,
                ggml_reshape_2d(c, ik, g.index_dim, n), positions));
        } else {
            auto kdst = ggml_view_3d(c, kv.k, g.head_dim, g.kv_heads, n, kv.k->nb[1], kv.k->nb[2], size_t(start) * kv.k->nb[2]);
            auto vdst = ggml_view_3d(c, kv.v, g.head_dim, g.kv_heads, n, kv.v->nb[1], kv.v->nb[2], size_t(start) * kv.v->nb[2]);
            auto idst = ggml_view_2d(c, kv.index, g.index_dim, n, kv.index->nb[1], size_t(start) * kv.index->nb[1]);
            ggml_build_forward_expand(b.graph, ggml_cpy(c, k, kdst));
            ggml_build_forward_expand(b.graph, ggml_cpy(c, v, vdst));
            ggml_build_forward_expand(b.graph, ggml_cpy(c, ggml_reshape_2d(c, ik, g.index_dim, n), idst));
        }

        const int64_t nk = kv_length(start + n, g), blocks = nk / g.block;
        const int64_t top = std::min(g.top_blocks, blocks);
        auto ikv = ggml_view_2d(c, kv.index, g.index_dim, nk, kv.index->nb[1], 0);
        // PR's get_rows(pos->cell) converts the F16 cache to F32 before the dot.
        // An identity map still needs that conversion: a direct F16 mul_mat rounds IQ,
        // changing block selection near ties once more than top_blocks exist.
        ikv = ggml_cast(c, ikv, GGML_TYPE_F32);
        ggml_tensor * cell_block = nullptr; // cell-to-block map shared by all queries
        auto select_blocks = [&](ggml_tensor * query, ggml_tensor * mask, ggml_tensor * block_bias, int query_n) {
            require(nk * g.index_heads * query_n < INT32_MAX,
                    "MSA score exceeds int32 indexing; reduce the prefill chunk");
            auto score = ggml_mul_mat(c, ikv, ggml_reshape_2d(c, query, g.index_dim, g.index_heads * query_n));
            ggml_prec_set_acc(score, GGML_PREC_F32);
            score = ggml_reshape_3d(c, score, nk, g.index_heads, query_n);
            // The CUDA F32 ADD path expects a F32 second operand. Convert
            // only the current causal tile when its shared storage is F16.
            if (mask->type != GGML_TYPE_F32) mask = ggml_cast(c, mask, GGML_TYPE_F32);
            score = msa_topk_mask && n > 1 ? ggml_add_inplace(c, score, mask) : ggml_add(c, score, mask);
            auto bs = b.tap(ggml_pool_2d(c, score, GGML_OP_POOL_MAX, int(g.block), 1, int(g.block), 1, 0, 0), "msa_bs", l);
            return b.tap(ggml_top_k(c, ggml_add(c, bs, block_bias), int(top)), "msa_topk", l);
        };
        auto attend_queries = [&](ggml_tensor * q, ggml_tensor * iq, ggml_tensor * pos_mask,
                                  ggml_tensor * bias, int query_n) {
            ggml_tensor * idx = nullptr;
            idx = select_blocks(iq, pos_mask, bias, query_n);
            ggml_tensor * result = nullptr;
            if (n == 1) {
                // Match the oracle's score-sorted gather order, including future/padded rows masked at -inf.
                auto a = ggml_reshape_3d(c, ggml_scale(c, ggml_cast(c, idx, GGML_TYPE_F32), float(g.block)), 1, top, g.index_heads);
                auto tj = ggml_add(c, ggml_repeat_4d(c, a, g.block, top, g.index_heads, 1),
                                  ggml_reshape_3d(c, ggml_arange(c, 0, float(g.block), 1), g.block, 1, 1));
                auto tr = ggml_add(c, ggml_scale(c, tj, float(g.kv_heads)),
                                  ggml_reshape_3d(c, ggml_arange(c, 0, float(g.kv_heads), 1), 1, 1, g.kv_heads));
                auto tok = ggml_cast(c, ggml_reshape_1d(c, tj, g.block * top * g.index_heads), GGML_TYPE_I32);
                auto row = ggml_cast(c, ggml_reshape_1d(c, tr, g.block * top * g.index_heads), GGML_TYPE_I32);
                auto k3 = ggml_view_2d(c, kv.k, g.head_dim, nk * g.kv_heads, kv.k->nb[1], 0);
                auto v3 = ggml_view_2d(c, kv.v, g.head_dim, nk * g.kv_heads, kv.v->nb[1], 0);
                auto kg = gather_half(b, k3, row);
                auto vg = gather_half(b, v3, row);
                auto mask_rows = ggml_reshape_2d(c, pos_mask, 1, nk);
                auto mg = native_gather ? gather_half(b, mask_rows, tok) : ggml_get_rows(c, mask_rows, tok);
                auto mask = ggml_reshape_4d(c, mg, g.block * top, 1, 1, g.kv_heads);
                if (!native_gather) mask = ggml_cast(c, mask, GGML_TYPE_F16);
                result = grouped_fa(b, q,
                    ggml_reshape_4d(c, kg, g.head_dim, g.block * top, 1, g.kv_heads),
                    ggml_reshape_4d(c, vg, g.head_dim, g.block * top, 1, g.kv_heads), mask, g, query_n);
            } else {
                ggml_tensor * mask = nullptr;
                if (msa_topk_mask) {
                    // Keep the complete index GEMM/query geometry. Only materialize
                    // the final F16 causal mask, without expanded transpose copies.
                    auto causal = ggml_cast(c, ggml_reshape_4d(c, pos_mask, nk, query_n, 1, 1), GGML_TYPE_F16);
                    // Permute top-k axes directly; preserve its rank order.
                    auto picks = ggml_permute(c, idx, 0, 3, 1, 2); // [top,queries,1,index_heads]
                    auto block_map = ggml_reshape_2d(c,
                        ggml_cast(c, ggml_arange(c, 0, float(nk), 1), GGML_TYPE_I32), g.block, blocks);
                    mask = ggml_top_k_mask(c, causal, picks, nullptr, block_map, nullptr);
                } else {
                    // Scatter the chosen blocks into a mask, then expand each block to its 128 token positions.
                    auto ninf = ggml_cast(c, ggml_scale_bias(c, bias, 0, -1e30f), GGML_TYPE_F16);
                    ninf = ggml_repeat_4d(c, ninf, blocks, g.index_heads, query_n, 1);
                    auto zero = ggml_scale(c, ggml_cast(c, idx, GGML_TYPE_F32), 0);
                    auto bm = ggml_set_rows(c,
                        ggml_reshape_3d(c, ninf, 1, blocks, g.index_heads * query_n),
                        ggml_reshape_3d(c, zero, 1, top, g.index_heads * query_n),
                        ggml_reshape_2d(c, idx, top, g.index_heads * query_n));
                    bm = ggml_reshape_3d(c, bm, blocks, g.index_heads, query_n);
                    bm = ggml_cont(c, ggml_permute(c, bm, 0, 2, 1, 3)); // [blocks,query_n,heads]
                    b.tap(bm, "msa_block_mask", l);
                    auto rows = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, bm, blocks, query_n * g.index_heads)));
                    if (!cell_block) cell_block = attention_inputs ? attention_inputs->blocks : b.input(GGML_TYPE_I32, nk);
                    // This extra input is uploaded by the caller after graph allocation.
                    ggml_set_name(cell_block, "m31.cell_block");
                    cell_blocks_input = cell_block;
                    auto selected = native_gather ? gather_half(b, rows, cell_block) : ggml_get_rows(c, rows, cell_block);
                    selected = ggml_cont(c, ggml_transpose(c, selected));
                    selected = ggml_reshape_3d(c, selected, nk, query_n, g.index_heads);
                    auto causal = ggml_reshape_3d(c, pos_mask, nk, query_n, 1);
                    mask = ggml_add(c, selected, causal);
                    if (mask->type != GGML_TYPE_F16) mask = ggml_cast(c, mask, GGML_TYPE_F16);
                    mask = ggml_reshape_4d(c, mask, nk, query_n, 1, g.index_heads);
                }
                auto k4 = ggml_view_4d(c, kv.k, g.head_dim, g.kv_heads, nk, 1,
                                      kv.k->nb[1], kv.k->nb[2], kv.k->nb[3], 0);
                auto v4 = ggml_view_4d(c, kv.v, g.head_dim, g.kv_heads, nk, 1,
                                      kv.v->nb[1], kv.v->nb[2], kv.v->nb[3], 0);
                result = grouped_fa(b, q, ggml_permute(c, k4, 0, 3, 1, 2),
                                     ggml_permute(c, v4, 0, 3, 1, 2), mask, g, query_n);
            }
            return result;
        };
        ggml_tensor * result = nullptr;
        result = attend_queries(q, iq, pos_mask, bias, n);
        result = b.tap(result, "kqv_out", l);
        return ggml_mul_mat(c, wt("attn_output.weight"), result);
    }
    ggml_tensor * cell_blocks_input = nullptr; // per transient graph; never retained across a compute

    std::vector<float> routed(int l, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & ids, const std::vector<float> & weights) {
        if (gpu_experts && n >= cfg.gpu_experts_min_tokens) return gpu_experts->forward(l, n, x, ids, weights);
        std::vector<float> output;
        routed_cpu_into(l, n, x, ids, weights, output);
        return output;
    }

    void routed_cpu_into(int l, int n, const std::vector<float> & x,
                          const std::vector<int32_t> & ids, const std::vector<float> & weights,
                          std::vector<float> & output) {
        const auto setup_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (cpu_fused && n == 1 && !cfg.tap) {
            const std::string prefix = "blk." + std::to_string(l) + ".ffn_";
            auto scale = [&](const char * role) {
                auto tensor = host.maybe(prefix + role + "_exps.scale");
                return tensor ? static_cast<const float *>(tensor->data) : nullptr;
            };
            if (cfg.profile) timings.routed_setup += elapsed(setup_start);
            const auto compute_start = cfg.profile ? Clock::now() : Clock::time_point{};
            auto run = [&](auto & executor) {
                executor.forward(host.get(prefix + "gate_exps.weight"), host.get(prefix + "up_exps.weight"),
                    host.get(prefix + "down_exps.weight"), scale("gate"), scale("up"), scale("down"),
                    x, ids, weights, output);
            };
            run(*cpu_fused);
            if (cfg.profile) timings.routed_compute += elapsed(compute_start);
            return;
        }
        RoutedGraph temporary;
        RoutedGraph * cached = &temporary;
        if (cfg.cpu_decode_cache && n == 1 && !cfg.tap) {
            if (!cpu_decode_graphs[size_t(l)]) cpu_decode_graphs[size_t(l)] = std::make_unique<RoutedGraph>();
            cached = cpu_decode_graphs[size_t(l)].get();
        }
        if (!cached->graph) {
            // One token has fixed tensor shapes and only IDs/activation/route weights
            // vary. Keep the original ggml graph and its alias plan, not a new graph
            // with an assumed equivalent node count. Taps retain the uncached path.
            cached->graph = std::make_unique<Graph>(cpu, cfg, pos,
                cached == &temporary ? cpu_workspace.get() : nullptr,
                cached == &temporary ? 32 * 1024 * 1024 : 1024 * 1024,
                cached == &temporary ? 8192 : 512);
            build_routed(l, n, *cached);
        }
        auto & b = *cached->graph;
        ggml_backend_tensor_set(cached->input, x.data(), 0, ggml_nbytes(cached->input));
        ggml_backend_tensor_set(cached->ids, ids.data(), 0, ggml_nbytes(cached->ids));
        ggml_backend_tensor_set(cached->weights, weights.data(), 0, ggml_nbytes(cached->weights));
        if (cfg.profile) timings.routed_setup += elapsed(setup_start);
        const auto compute_start = cfg.profile ? Clock::now() : Clock::time_point{};
        b.compute();
        if (cfg.profile) timings.routed_compute += elapsed(compute_start);
        const auto read_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (reuse_host_reads) read_host(cached->output, output);
        else output = Graph::read(cached->output);
        if (cfg.profile) timings.routed_read += elapsed(read_start);
    }

    void build_routed(int l, int n, RoutedGraph & storage) {
        auto & b = *storage.graph;
        auto c = b.c;
        auto inp = b.input(GGML_TYPE_F32, g.embd, 1, n);
        auto id = b.input(GGML_TYPE_I32, g.used, n);
        auto rw = b.input(GGML_TYPE_F32, 1, g.used, n);
        const std::string prefix = "blk." + std::to_string(l) + ".ffn_";
        auto mm = [&](const char * role, ggml_tensor * activation) {
            auto w = host.get(prefix + role + "_exps.weight");
            auto out = ggml_mul_mat_id(c, w, activation, id);
            if (w->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(out, GGML_PREC_BF16);
            if (auto scale = host.maybe(prefix + role + "_exps.scale")) {
                auto s = ggml_repeat_4d(c, ggml_reshape_3d(c, scale, 1, g.experts, 1), 1, g.experts, n, 1);
                out = ggml_mul(c, out, ggml_get_rows(c, s, id));
            }
            return out;
        };
        auto gate = b.tap(mm("gate", inp), "ffn_moe_gate", l);
        auto up = b.tap(mm("up", inp), "ffn_moe_up", l);
        auto h = ggml_swiglu_oai(c, gate, up, 1.702f, 7.0f);
        auto down = b.tap(mm("down", h), "ffn_moe_down_scaled", l);
        auto weighted = ggml_mul(c, down, rw);
        // Match llama.cpp's sequential accumulation order, rather than sum_rows/tree reduction.
        ggml_tensor * result = nullptr;
        for (int64_t k = 0; k < g.used; ++k) {
            auto v = ggml_view_2d(c, weighted, g.embd, n, weighted->nb[2], size_t(k) * weighted->nb[1]);
            result = result ? ggml_add(c, result, v) : v;
        }
        result = b.tap(ggml_cont(c, result), "ffn_moe_out", l);
        b.allocate({result});
        storage.input = inp;
        storage.ids = id;
        storage.weights = rw;
        storage.output = result;
    }

    void drain_shared() noexcept {
        if (shared_inflight) {
            ggml_backend_synchronize(backend);
            shared_inflight = false;
        }
    }
    bool overlap_shared(int layer, int n) const {
        return cpu_shared_overlap && n == 1 && g.routed(layer) &&
            !(gpu_experts && n >= cfg.gpu_experts_min_tokens);
    }
    void enqueue_shared(int layer, const std::vector<float> & input, ggml_tensor * device_input = nullptr) {
        require(!shared_inflight, "a shared FFN graph is already in flight");
        auto & storage = shared_decode_graphs[size_t(layer)];
        if (!storage) storage = std::make_unique<HeadGraph>();
        if (!storage->graph) {
            // No views into the attention graph/workspace. These buffers and
            // tensor addresses remain owned until reset/destruction, after drain.
            storage->graph = std::make_unique<Graph>(backend, cfg, pos, nullptr, 1024 * 1024, 512);
            auto & b = *storage->graph;
            storage->input = b.input(GGML_TYPE_F32, g.embd, 1);
            storage->output = ffn(b, *dense, storage->input, "blk." + std::to_string(layer) + ".", "_shexp");
            b.allocate({storage->output});
        }
        if (device_input)
            ggml_backend_tensor_copy_async(backend, backend, device_input, storage->input);
        else ggml_backend_tensor_set(storage->input, input.data(), 0, ggml_nbytes(storage->input));
        // Set before submission: even a failed submission may have queued work.
        shared_inflight = true;
        require(ggml_backend_graph_compute_async(backend, storage->graph->graph) == GGML_STATUS_SUCCESS,
                "async shared FFN submission failed");
    }

    void clear_dense_decode_cache(bool release_shared = false) {
        // CUDA's captures reference tensor metadata and shared attention inputs.
        // Destroy them before either ownership group can be replaced/freed.
        drain_shared();
        if (release_shared) for (auto & graph : shared_decode_graphs) graph.reset();
        for (auto & graph : dense_decode_graphs) graph.reset();
        head_decode_graph.graph.reset();
        head_decode_graph.input = head_decode_graph.output = nullptr;
    }

    ggml_tensor * previous_hidden(Graph & b, int layer, ggml_tensor * moe_input) {
        auto & previous = *dense_decode_graphs[size_t(layer)];
        if (!g.routed(layer)) return b.borrow(previous.output);
        auto shared = previous.shared ? previous.shared : shared_decode_graphs[size_t(layer)]->output;
        // Keep the accepted rank sum then shared then residual F32 operation order.
        return ggml_add(b.c, ggml_add(b.c, moe_input, b.borrow(shared)), b.borrow(previous.residual));
    }

    void build_dense(int l, int n, DenseGraph & storage, bool stable_decode, bool split_shared) {
        const auto build_start = cfg.profile ? Clock::now() : Clock::time_point{};
        auto & b = *storage.graph;
        auto c = b.c;
        storage.input = !prefill_chain_active && (!decode_chain_active || l == 0 || g.routed(l - 1))
            ? b.input(GGML_TYPE_F32, g.embd, n) : nullptr;
        auto hidden_input = prefill_chain_active ? b.borrow(prefill_hidden->current())
            : (decode_chain_active && l > 0 ? previous_hidden(b, l - 1, storage.input) : storage.input);
        storage.nk = kv_length(pos + n, g);
        const int64_t nb = storage.nk / g.block;
        storage.positions = attention_inputs ? attention_inputs->positions : b.input(GGML_TYPE_I32, n);
        storage.mask = attention_inputs ? attention_inputs->mask : b.input(GGML_TYPE_F32, storage.nk, 1, n);
        storage.bias = attention_inputs ? attention_inputs->bias : b.input(GGML_TYPE_F32, nb, 1, n);
        cell_blocks_input = nullptr;
        auto a = attention(b, l, pos, n, hidden_input, storage.positions, storage.mask, storage.bias, stable_decode);
        auto ffn_inp = b.tap(ggml_add(c, a, hidden_input), "ffn_inp", l);
        const std::string prefix = "blk." + std::to_string(l) + ".";
        auto norm = b.tap(rms(b, ffn_inp, dense->get(prefix + "ffn_norm.weight"), g.rms_eps), "ffn_norm", l);
        auto allocate = [&](std::initializer_list<ggml_tensor *> outputs) {
            if (cfg.profile) timings.dense_build += elapsed(build_start);
            const auto alloc_start = cfg.profile ? Clock::now() : Clock::time_point{};
            b.allocate(outputs);
            if (cfg.profile) timings.dense_allocate += elapsed(alloc_start);
        };
        if (!g.routed(l)) {
            auto ff = b.tap(ffn(b, *dense, norm, prefix, ""), "ffn_out", l);
            storage.output = b.tap(ggml_add(c, ff, ffn_inp), "l_out", l);
            allocate({storage.output});
            return;
        }
        storage.residual = ffn_inp;
        storage.norm = norm;
        if (!split_shared)
            storage.shared = b.tap(ffn(b, *dense, norm, prefix, "_shexp"), "ffn_shexp", l);
        auto probs = ggml_sigmoid(c, ggml_mul_mat(c, dense->get(prefix + "ffn_gate_inp.weight"), norm));
        storage.ids = b.tap(ggml_top_k(c, ggml_add(c, probs, dense->get(prefix + "exp_probs_b.bias")), int(g.used)), "ffn_moe_topk", l);
        auto weight = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, g.experts, n), storage.ids);
        weight = ggml_reshape_2d(c, weight, g.used, n);
        if (g.norm_routes) {
            auto sum = ggml_clamp(c, ggml_sum_rows(c, weight), 6.103515625e-5f, INFINITY);
            weight = ggml_div(c, weight, sum);
        }
        if (g.route_scale != 1) weight = ggml_scale(c, weight, g.route_scale);
        storage.weights = weight;
        // Preserve the backend top-k rank order. Older ggml revisions may
        // require a contiguous pack before the routing metadata readback.
        // Keep the default correctness graph untouched; the opt-in allocation
        // path replans its alias topology after this extra integer-only node.
        if (cfg.reuse_allocations && !ggml_is_contiguous(storage.ids))
            storage.ids = ggml_cont(c, storage.ids);
        if (split_shared) allocate({storage.residual, storage.norm, storage.ids, storage.weights});
        else allocate({storage.residual, storage.norm, storage.shared, storage.ids, storage.weights});
    }

    void layer(int l, int n) {
        const auto dense_start = cfg.profile ? Clock::now() : Clock::time_point{};
        std::vector<float> local_input, local_residual, local_shared, local_weights;
        std::vector<int32_t> local_ids;
        auto & ffn_input = reuse_host_reads ? host_reads.input : local_input;
        auto & residual = reuse_host_reads ? host_reads.residual : local_residual;
        auto & shared = reuse_host_reads ? host_reads.shared : local_shared;
        auto & route_weights = reuse_host_reads ? host_reads.weights : local_weights;
        auto & ids = reuse_host_reads ? host_reads.ids : local_ids;
        struct DenseDrain {
            ggml_backend_t backend; bool pending;
            ~DenseDrain() { if (pending) ggml_backend_synchronize(backend); }
        } dense_drain{backend, dense_async_reads}; // drains before any local readback vector is freed
        const auto ids_capacity = ids.capacity();
        ids.resize(size_t(g.used * n));
        if (reuse_host_reads) {
            if (ids.capacity() > ids_capacity) ++host_reads.growths;
            else ++host_reads.reuses;
        }
        DenseGraph temporary;
        DenseGraph * storage = &temporary;
        const bool stable_decode = cfg.dense_decode_cache && n == 1 && !cfg.tap;
        const bool shared_overlap = overlap_shared(l, n);
        if (stable_decode) {
            if (!dense_decode_graphs[size_t(l)]) dense_decode_graphs[size_t(l)] = std::make_unique<DenseGraph>();
            storage = dense_decode_graphs[size_t(l)].get();
        }
        if (!storage->graph) {
            const auto create_start = cfg.profile ? Clock::now() : Clock::time_point{};
            storage->graph = std::make_unique<Graph>(backend, cfg, pos,
                stable_decode ? nullptr : dense_workspace.get(),
                stable_decode ? 4 * 1024 * 1024 : 32 * 1024 * 1024,
                stable_decode ? 2048 : 8192);
            if (cfg.profile) timings.dense_build += elapsed(create_start);
            build_dense(l, n, *storage, stable_decode, shared_overlap);
        }
        require(storage->nk == kv_length(pos + n, g), "stale dense decode bucket");
        auto & b = *storage->graph;
        const auto upload_start = cfg.profile ? Clock::now() : Clock::time_point{};
        upload_attention(b, storage->input, storage->positions, storage->mask, storage->bias,
                         n, storage->nk, storage->nk / g.block);
        if (cfg.profile) timings.dense_upload += elapsed(upload_start);
        const auto compute_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (dense_async_reads)
            require(ggml_backend_graph_compute_async(backend, b.graph) == GGML_STATUS_SUCCESS, "async dense graph failed");
        else b.compute();
        if (cfg.profile) timings.dense_compute += elapsed(compute_start);
        const auto read_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (!g.routed(l)) {
            if (prefill_chain_active) { /* copied to persistent GPU hidden before graph release */ }
            else if (decode_chain_active) { /* the next graph consumes this device output */ }
            else if (dense_async_reads) read_async(storage->output, hidden);
            else if (reuse_host_reads) read_host(storage->output, hidden);
            else hidden = Graph::read(storage->output);
        }
        else {
            if (prefill_chain_active) { /* norm/residual/shared stay in this live GPU graph */ }
            else if (dense_async_reads) {
                if (!decode_chain_active) read_async(storage->residual, residual);
                read_async(storage->norm, ffn_input);
                if (!decode_chain_active && !shared_overlap) read_async(storage->shared, shared);
            } else if (reuse_host_reads) {
                read_host(storage->residual, residual);
                read_host(storage->norm, ffn_input);
                if (!shared_overlap) read_host(storage->shared, shared);
            } else {
                residual = Graph::read(storage->residual);
                ffn_input = Graph::read(storage->norm);
                if (!shared_overlap) shared = Graph::read(storage->shared);
            }
            if (ggml_is_contiguous(storage->ids)) {
                if (dense_async_reads) ggml_backend_tensor_get_async(backend, storage->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
                else ggml_backend_tensor_get(storage->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            } else {
                for (int t = 0; t < n; ++t)
                    ggml_backend_tensor_get(storage->ids, ids.data() + t * g.used,
                        size_t(t) * storage->ids->nb[1], size_t(g.used) * sizeof(int32_t));
            }
            if (reuse_host_reads) {
                ++host_reads.reads;
                host_reads.bytes += ids.size() * sizeof(int32_t);
                if (dense_async_reads) read_async(storage->weights, route_weights);
                else read_host(storage->weights, route_weights);
            } else if (dense_async_reads) read_async(storage->weights, route_weights);
            else route_weights = Graph::read(storage->weights);
        }
        if (dense_async_reads) {
            // Dense leading layers have no host consumers; the same backend stream
            // carries their persistent outputs directly into the next layer.
            if (!decode_chain_active || g.routed(l)) ggml_backend_synchronize(backend);
            dense_drain.pending = false;
        }
        if (cfg.profile) timings.dense_read += elapsed(read_start);
        if (prefill_chain_active) {
            // ID/weight reads above join the backend. The current graph owns
            // norm/shared/residual until forward_device has joined its private
            // FFN stream; no tensor aliases survive its destruction below.
            double routed_seconds = 0;
            if (!g.routed(l)) prefill_hidden->copy_and_advance(backend, storage->output);
            else {
                require(gpu_experts && gpu_experts->native_gpu_math(), "GPU prefill chain lost its device expert executor");
                if (cfg.route_observer) cfg.route_observer(l,n,ids,route_weights);
                const auto routed_start = cfg.profile ? Clock::now() : Clock::time_point{};
                gpu_experts->forward_device(l,n,storage->norm,ids,route_weights,
                    storage->shared,storage->residual,prefill_hidden->next());
                if (cfg.profile) { routed_seconds = elapsed(routed_start); timings.routed_compute += routed_seconds; }
                prefill_hidden->advance();
            }
            const auto destroy_start = cfg.profile ? Clock::now() : Clock::time_point{};
            temporary.graph.reset();
            if (cfg.profile) { timings.dense_destroy += elapsed(destroy_start); timings.dense += elapsed(dense_start)-routed_seconds; }
            return;
        }
        const auto destroy_start = cfg.profile ? Clock::now() : Clock::time_point{};
        temporary.graph.reset(); // cached graphs/captures retain their buffers and tensor addresses
        if (cfg.profile) {
            timings.dense_destroy += elapsed(destroy_start);
            timings.dense += elapsed(dense_start);
        }
        if (!g.routed(l)) return;
        if (cfg.route_observer) cfg.route_observer(l, n, ids, route_weights);
        // The guard outlives submission and CPU work, and drains before any
        // failure/reset can release graph metadata, scratch, input or weights.
        struct SharedDrain {
            Impl * owner;
            ~SharedDrain() { owner->drain_shared(); }
        } shared_guard{this};
        const bool shared_timing = shared_overlap && (cfg.profile || shared_trace);
        const auto shared_submit = shared_timing ? Clock::now() : Clock::time_point{};
        if (shared_overlap) {
            enqueue_shared(l, ffn_input, decode_chain_active ? storage->norm : nullptr);
            if (cfg.profile) {
                timings.shared_enqueue += elapsed(shared_submit);
                ++timings.shared_batches;
            }
#ifdef M31_SHARED_OVERLAP_TESTING
            const char * inject = std::getenv("M31_CPU_SHARED_OVERLAP_FAIL_AFTER_LAUNCH");
            if (!shared_failure_fired && inject && std::string(inject) == "1") {
                shared_failure_fired = true;
                throw std::runtime_error("M31 injected failure after shared FFN launch");
            }
#endif
        }
        const auto shared_cpu_begin = shared_timing ? Clock::now() : Clock::time_point{};
#ifdef M31_SHARED_OVERLAP_NVTX
        if (shared_overlap) nvtxRangePushA("M31 CPU MoE while shared FFN is queued");
        struct NvtxPop {
            bool active;
            ~NvtxPop() { if (active) nvtxRangePop(); }
        } nvtx_cpu_range{shared_overlap};
#endif
        std::vector<float> local_moe;
        const std::vector<float> * moe_result = nullptr;
        if (reuse_host_reads) {
            if (gpu_experts && n >= cfg.gpu_experts_min_tokens)
                gpu_experts->forward_into(l, n, ffn_input, ids, route_weights, host_reads.moe);
            else routed_cpu_into(l, n, ffn_input, ids, route_weights, host_reads.moe);
            moe_result = &host_reads.moe;
        } else {
            // GPUExperts retains its existing owned-vector API and lifetime.
            local_moe = routed(l, n, ffn_input, ids, route_weights);
            moe_result = &local_moe;
        }
        const auto & moe = *moe_result;
        const auto shared_cpu_end = shared_timing ? Clock::now() : Clock::time_point{};
#ifdef M31_SHARED_OVERLAP_NVTX
        if (nvtx_cpu_range.active) { nvtxRangePop(); nvtx_cpu_range.active = false; }
#endif
        if (shared_overlap) {
            const auto wait_begin = shared_timing ? Clock::now() : Clock::time_point{};
            drain_shared();
            if (cfg.profile) timings.shared_wait += elapsed(wait_begin);
            const auto read_begin = shared_timing ? Clock::now() : Clock::time_point{};
            auto tensor = shared_decode_graphs[size_t(l)]->output;
            if (decode_chain_active) { /* next-layer residual consumes the device value */ }
            else if (reuse_host_reads) read_host(tensor, shared);
            else shared = Graph::read(tensor);
            if (cfg.profile) timings.shared_read += elapsed(read_begin);
            if (shared_trace) {
                auto ns = [](Clock::time_point time) { return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count(); };
                std::fprintf(stderr, "M31 shared-overlap pos=%lld layer=%d submit_ns=%lld cpu_begin_ns=%lld cpu_end_ns=%lld joined_ns=%lld\n",
                    (long long)pos, l, (long long)ns(shared_submit), (long long)ns(shared_cpu_begin),
                    (long long)ns(shared_cpu_end), (long long)ns(Clock::now()));
            }
        }
        const auto residual_start = cfg.profile ? Clock::now() : Clock::time_point{};
        if (decode_chain_active) {
            hidden.assign(moe.begin(), moe.end()); // only this MoE vector crosses H2D at the next layer
            if (cfg.profile) timings.residual += elapsed(residual_start);
            return;
        }
        if (cfg.cpu_residual && !cfg.tap) {
            hidden.resize(moe.size());
            for (size_t i = 0; i < moe.size(); ++i) {
                const float ff = moe[i] + shared[i];
                hidden[i] = ff + residual[i];
            }
            if (cfg.profile) timings.residual += elapsed(residual_start);
            return;
        }
        Graph residual_graph(backend, cfg, pos, dense_workspace.get());
        auto r = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto s = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto m = residual_graph.input(GGML_TYPE_F32, g.embd, n);
        auto ff = residual_graph.tap(ggml_add(residual_graph.c, m, s), "ffn_out", l);
        auto out = residual_graph.tap(ggml_add(residual_graph.c, ff, r), "l_out", l);
        residual_graph.allocate({out});
        ggml_backend_tensor_set(r, residual.data(), 0, ggml_nbytes(r));
        ggml_backend_tensor_set(s, shared.data(), 0, ggml_nbytes(s));
        ggml_backend_tensor_set(m, moe.data(), 0, ggml_nbytes(m));
        residual_graph.compute();
        if (reuse_host_reads) read_host(out, hidden);
        else hidden = Graph::read(out);
        if (cfg.profile) timings.residual += elapsed(residual_start);
    }

    void upload_attention(Graph &, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * mask,
                          ggml_tensor * bias, int n, int64_t nk, int64_t nb) {
        if (x) {
            if (dense_async_reads) ggml_backend_tensor_set_async(backend, x, hidden.data(), 0, ggml_nbytes(x));
            else ggml_backend_tensor_set(x, hidden.data(), 0, ggml_nbytes(x));
        }
        if (attention_inputs) return; // uploaded once before the first layer
        fill_attention(positions, mask, bias, cell_blocks_input, n, nk, nb);
    }

    void fill_attention(ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * bias,
                        ggml_tensor * cell_blocks, int n, int64_t nk, int64_t nb) {
        std::vector<int32_t> positions_h(size_t(n), 0);
        std::vector<float> mask_h, bias_h(size_t(nb * n), 0);
        std::vector<ggml_fp16_t> half_mask_h;
        if (mask->type == GGML_TYPE_F16) half_mask_h.assign(size_t(nk * n), ggml_fp32_to_fp16(-INFINITY));
        else mask_h.assign(size_t(nk * n), -INFINITY);
        for (int t = 0; t < n; ++t) {
            positions_h[size_t(t)] = int32_t(pos + t);
            if (mask->type == GGML_TYPE_F16)
                std::fill_n(half_mask_h.data() + size_t(t) * nk, size_t(pos + t + 1), ggml_fp16_t(0));
            else std::fill_n(mask_h.data() + size_t(t) * nk, size_t(pos + t + 1), 0);
            for (int64_t b = (pos + t) / g.block, local = 0; local < g.local_blocks && b >= 0; ++local, --b)
                bias_h[size_t(t) * nb + b] = 1e30f;
        }
        ggml_backend_tensor_set(positions, positions_h.data(), 0, ggml_nbytes(positions));
        ggml_backend_tensor_set(mask, mask->type == GGML_TYPE_F16 ? static_cast<const void *>(half_mask_h.data()) : static_cast<const void *>(mask_h.data()), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(bias, bias_h.data(), 0, ggml_nbytes(bias));
        if (cell_blocks) {
            std::vector<int32_t> blocks(size_t(nk), 0);
            for (int64_t i = 0; i < nk; ++i) blocks[size_t(i)] = int32_t(i / g.block);
            ggml_backend_tensor_set(cell_blocks, blocks.data(), 0, ggml_nbytes(cell_blocks));
        }
    }
};

Engine::Engine() : p_(new Impl) {}
Engine::~Engine() = default;

bool Engine::init(const std::string & path, const Config & requested, std::string & error) {
    try {
        Config cfg = requested;
        require(!p_->model, "init may only be called once");
        validate_config(cfg);
        bool cuda_requested = false;
#ifdef M31_ENGINE_CUDA
        cuda_requested = cfg.backend && ggml_backend_is_cuda(cfg.backend);
#endif
        require(cuda_requested || (!cfg.gpu_experts_gib && !cfg.register_arena),
                "GPU expert budgets and arena registration require a CUDA backend");
        require(cfg.threads > 0 && cfg.context > 0 && cfg.max_chunk > 0, "invalid configuration");
        require(std::isfinite(cfg.arena_gib) && cfg.arena_gib >= 0, "invalid arena budget");
        require(std::isfinite(cfg.gpu_experts_gib) && cfg.gpu_experts_gib >= 0, "invalid GPU expert budget");
        require(cfg.gpu_experts_min_tokens > 0 && cfg.gpu_experts_min_tokens <= 8192, "invalid GPU expert token threshold");
        require(!cfg.dense_decode_cache || (cfg.reuse_allocations && cfg.cpu_residual),
                "dense decode cache requires reuse-allocations and cpu-residual");
        p_->cfg = cfg;
        p_->reuse_host_reads = cfg.reuse_allocations && !cfg.tap;
        p_->model = std::make_unique<GgufModel>(gguf_split_paths(path));
        p_->g = read_geometry(*p_->model);
        if (cuda_requested && cfg.arena_gib == 0) {
            cfg.arena_gib = double(ExpertArena::required_bytes(*p_->model, p_->g)) / 1073741824.0;
            require(cfg.arena_gib > 0, "native GPU prefill requires routed expert weights");
            p_->cfg = cfg;
        }
        p_->cpu_decode_graphs.resize(size_t(p_->g.layers));
        p_->dense_decode_graphs.resize(size_t(p_->g.layers));
        p_->shared_decode_graphs.resize(size_t(p_->g.layers));
        require(cfg.context <= p_->g.context && cfg.context <= INT32_MAX, "context exceeds model maximum");
        p_->capacity = kv_length(cfg.context, p_->g);
        bool fused_supported = true;
        for (int l = int(p_->g.dense_lead); l < p_->g.layers; ++l)
            for (int type : {p_->g.gate_types[size_t(l)], p_->g.up_types[size_t(l)], p_->g.down_types[size_t(l)]})
                fused_supported = fused_supported && CpuFusedDecode::supported_type(ggml_type(type));
        require(!cfg.cpu_fused || (cfg.persistent_cpu_pool && fused_supported),
                "fused CPU decode requires a persistent pool and IQ3_XXS/IQ4_XS/NVFP4 experts");
        p_->cpu = ggml_backend_cpu_init();
        require(p_->cpu != nullptr, "cannot create CPU expert backend");
        ggml_backend_cpu_set_n_threads(p_->cpu, cfg.threads);
        p_->backend = cfg.backend;
        if (!p_->backend) {
            p_->backend = ggml_backend_cpu_init();
            p_->own_backend = true;
        }
        require(p_->backend != nullptr, "cannot create dense backend");
#ifdef M31_ENGINE_CUDA
        const bool cuda = ggml_backend_is_cuda(p_->backend);
        p_->cpu_shared_overlap = cuda && cfg.reuse_allocations && cfg.cpu_residual && cfg.dense_decode_cache && !cfg.tap;
        p_->gpu_decode_chain = cuda && cfg.dense_decode_cache && !cfg.tap;
        p_->dense_async_reads = cuda && cfg.reuse_allocations && !cfg.tap;
        p_->gpu_prefill_chain = cuda && !cfg.tap;
        p_->native_gather = cuda && !cfg.tap;
        p_->adaptive_prefill = cuda;
        p_->msa_topk_mask = cuda && !cfg.tap;
#endif
        if (ggml_backend_is_cpu(p_->backend)) ggml_backend_cpu_set_n_threads(p_->backend, cfg.threads);
        if (cfg.persistent_cpu_pool) {
            auto params = ggml_threadpool_params_default(cfg.threads);
            apply_cpu_pool_config(params, cfg.cpu_affinity, cfg.cpu_poll);
            p_->cpu_pool = ggml_threadpool_new(&params);
            require(p_->cpu_pool != nullptr, "cannot create persistent CPU pool");
            const auto affinity_count = std::count(std::begin(params.cpumask), std::end(params.cpumask), true);
            std::fprintf(stderr, "M31 CPU pool: threads=%d persistent=1 poll=%u strict=%d affinity_cpus=%zu\n",
                cfg.threads, unsigned(params.poll), int(params.strict_cpu), size_t(affinity_count));
            ggml_backend_cpu_set_threadpool(p_->cpu, p_->cpu_pool);
            if (ggml_backend_is_cpu(p_->backend)) ggml_backend_cpu_set_threadpool(p_->backend, p_->cpu_pool);
        } else std::fprintf(stderr, "M31 CPU pool: threads=%d persistent=0\n", cfg.threads);
        if (cfg.cpu_fused && !cfg.tap)
            p_->cpu_fused = std::make_unique<CpuFusedDecode>(p_->cpu, p_->cpu_pool, unsigned(cfg.threads));
        if (cfg.reuse_allocations) {
            p_->dense_workspace = std::make_unique<Workspace>(p_->backend);
            p_->cpu_workspace = std::make_unique<Workspace>(p_->cpu);
        }
        p_->host.load(*p_->model, p_->cpu, false, &p_->g, cfg.arena_gib, cfg);
        if (ggml_backend_is_cpu(p_->backend)) p_->dense = &p_->host;
        else { p_->device.load(*p_->model, p_->backend, true, &p_->g); p_->dense = &p_->device; }
        p_->state = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(p_->state != nullptr, "cache metadata allocation failed");
        const auto & g = p_->g;
        for (int64_t l = 0; l < g.layers; ++l) {
            Impl::Cache kv;
            kv.k = ggml_new_tensor_3d(p_->state, GGML_TYPE_F16, g.head_dim, g.kv_heads, p_->capacity);
            kv.v = ggml_new_tensor_3d(p_->state, GGML_TYPE_F16, g.head_dim, g.kv_heads, p_->capacity);
            kv.index = ggml_new_tensor_2d(p_->state, GGML_TYPE_F16, g.index_dim, p_->capacity);
            p_->cache.push_back(kv);
        }
        p_->state_buffer = ggml_backend_alloc_ctx_tensors_from_buft(p_->state, ggml_backend_get_default_buffer_type(p_->backend));
        require(p_->state_buffer != nullptr, "cannot allocate KV/index caches");
        ggml_backend_buffer_clear(p_->state_buffer, 0); // padded rows must contain finite values
        if (cuda_requested && cfg.gpu_experts_gib == 0) {
            cfg.gpu_experts_gib = double(GpuExperts::automatic_budget_bytes(*p_->model, p_->g, p_->backend)) / 1073741824.0;
            p_->cfg = cfg;
        }
        if (cfg.gpu_experts_gib > 0) {
            require(p_->host.arena != nullptr, "GPU expert cache requires an arena");
            p_->gpu_experts = std::make_unique<GpuExperts>();
            p_->gpu_experts->init(*p_->model, p_->g, *p_->host.arena, p_->backend, cfg.gpu_experts_gib, cfg.max_chunk, false, cfg);
        }
        if (p_->gpu_prefill_chain) {
            require(cfg.reuse_allocations && cfg.cpu_residual && !cfg.tap && p_->gpu_experts && p_->gpu_experts->native_gpu_math(),
                "GPU prefill chain requires reuse/cpu-residual/no-taps and native or fast GPU experts");
            p_->prefill_hidden = std::make_unique<GpuPrefillHidden>();
        }
        if (p_->adaptive_prefill) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(p_->backend), &free_bytes, &total_bytes);
            p_->prefill_free_after_resident = free_bytes;
            require(free_bytes > 0, "device did not report free VRAM for adaptive prefill");
        }
        p_->ready = true;
        std::fprintf(stderr, "M31 effective config: %s\n", effective_config(p_->cfg).c_str());
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

const Geometry & Engine::geometry() const { return p_->g; }
int64_t Engine::position() const { return p_->pos; }
int Engine::prefill_chunk(int requested, std::string & error) const {
    try {
        require(p_->ready && !p_->poisoned && requested > 0 && p_->pos < p_->cfg.context, "invalid prefill planning state");
        const int requested_limit = int(std::min<int64_t>({requested, p_->cfg.max_chunk, p_->cfg.context - p_->pos}));
        const int limit = msa_chunk_limit(p_->pos,requested_limit,p_->g.index_heads,p_->g.block);
        require(limit > 0,"MSA int32 indexing does not admit even one query at this position");
        if (!p_->adaptive_prefill || limit == 1) return limit;
        PrefillBudget budget;
        // Snapshot after resident weights/full KV/cache allocation credits scratch
        // retained by earlier chunks instead of subtracting that same memory twice.
        // The extra GiB is a conservative reserve for lazy native GPU buffers.
        budget.free_after_resident = p_->prefill_free_after_resident;
        budget.reserve_bytes = 1ULL << 30;
        budget.extra_scratch_bytes = 1ULL << 30;
        budget.max_chunk = limit;
        budget.gpu_prefill_chain = p_->gpu_prefill_chain && limit >= p_->cfg.gpu_experts_min_tokens;
        budget.retained_hidden_bytes = p_->prefill_hidden ? p_->prefill_hidden->bytes() : 0;
        budget.native_gpu_ffn = p_->gpu_experts && p_->gpu_experts->native_gpu_math();
        if (budget.native_gpu_ffn) budget.retained_routed_bytes = p_->gpu_experts->workspace_bytes();
        if (p_->dense_workspace)
            budget.retained_dense_bytes = ggml_gallocr_get_buffer_size(p_->dense_workspace->allocator, 0);
        budget.f16_mask = p_->native_gather;
        budget.top_k_mask = p_->msa_topk_mask;
        budget.inplace_score = p_->msa_topk_mask;
        budget.position = p_->pos; budget.remaining = limit; budget.context = p_->cfg.context;
        budget.streaming_slots = uint64_t(p_->g.experts);
        if (p_->gpu_experts) {
            const auto & stats = p_->gpu_experts->stats();
            budget.streaming_slots = stats.slot_count;
        }
        const auto plan = plan_prefill(p_->g, budget);
        require(plan.chunk > 0, "adaptive prefill refused: " + plan.reason);
        if (p_->cfg.profile) std::fprintf(stderr,
            "M31 prefill-plan pos=%lld requested=%d planned=%d keys=%lld initial_free=%llu scratch=%llu headroom=%llu\n",
            (long long)p_->pos, requested, plan.chunk, (long long)plan.padded_keys,
            (unsigned long long)budget.free_after_resident, (unsigned long long)plan.scratch_estimate,
            (unsigned long long)plan.headroom_bytes);
        return plan.chunk;
    } catch (const std::exception & exception) { error = exception.what(); return 0; }
}
bool Engine::rewind(int64_t position, std::string & error) {
    try {
        require(p_->ready && !p_->poisoned, "rewind requires a valid initialized engine");
        require(position >= 0 && position <= p_->pos, "rewind position exceeds valid KV prefix");
        require(position == p_->pos || position % 256 == 0,
                "partial prefix reuse requires a 256-token boundary");
        p_->poisoned = true;
        p_->clear_dense_decode_cache(true);
        ggml_backend_synchronize(p_->backend);
        p_->attention_inputs.reset();
        p_->pos = position; p_->last_n = 0; p_->hidden.clear();
        p_->decode_chain_active = p_->prefill_chain_active = false;
        p_->poisoned = false;
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

bool Engine::release_prefill_workspace(std::string & error) {
    try {
        require(p_->ready && !p_->poisoned, "workspace release requires valid state");
        p_->poisoned = true;
        p_->clear_dense_decode_cache(true);
        if (p_->gpu_experts) p_->gpu_experts->release_prefill_workspace();
        p_->attention_inputs.reset(); p_->dense_workspace.reset();
        if (p_->cfg.reuse_allocations) p_->dense_workspace = std::make_unique<Workspace>(p_->backend);
        p_->prefill_hidden.reset();
        if (p_->gpu_prefill_chain) p_->prefill_hidden = std::make_unique<GpuPrefillHidden>();
        p_->hidden.clear(); p_->last_n = 0;
        p_->decode_chain_active = p_->prefill_chain_active = false;
        p_->poisoned = false;
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

size_t Engine::prefix_checkpoint_bytes() const {
    if (!p_->ready || p_->poisoned || p_->pos <= 0) return 0;
    return size_t(p_->g.kv_bytes(p_->pos));
}

bool Engine::checkpoint_prefix(size_t maximum_bytes, PrefixCheckpointPtr & out, std::string & error) {
    try {
        require(p_->ready && !p_->poisoned && p_->pos > 0 && p_->last_n > 0,
                "prefix checkpoint requires a completed nonempty forward");
        const size_t total = prefix_checkpoint_bytes();
        require(total > 0 && total <= maximum_bytes, "prefix checkpoint exceeds host cache budget");
        auto checkpoint = std::shared_ptr<PrefixCheckpoint>(new PrefixCheckpoint);
        checkpoint->owner_ = p_->checkpoint_owner;
        checkpoint->position_ = p_->pos;
        checkpoint->bytes_.resize(total);
        ggml_backend_synchronize(p_->backend);
        size_t offset = 0;
        for (const auto & kv : p_->cache) {
            for (auto * tensor : {kv.k, kv.v, kv.index}) {
                require(tensor && tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor),
                        "prefix checkpoint requires contiguous F16 cache tensors");
                const size_t row_bytes = tensor == kv.index ? tensor->nb[1] : tensor->nb[2];
                const size_t bytes = row_bytes * size_t(p_->pos);
                require(bytes <= total - offset, "prefix checkpoint cache layout mismatch");
                ggml_backend_tensor_get(tensor, checkpoint->bytes_.data() + offset, 0, bytes);
                offset += bytes;
            }
        }
        require(offset == total, "prefix checkpoint cache byte count mismatch");
        out = std::move(checkpoint);
        return true;
    } catch (const std::exception & exception) { error = exception.what(); return false; }
}

bool Engine::restore_prefix(const PrefixCheckpointPtr & checkpoint, std::string & error) {
    bool started = false;
    try {
        require(p_->ready && checkpoint && checkpoint->owner_ == p_->checkpoint_owner,
                "prefix checkpoint belongs to another Engine");
        const int64_t position = checkpoint->position_;
        require(position > 0 && position <= p_->cfg.context &&
                checkpoint->bytes_.size() == size_t(p_->g.kv_bytes(position)),
                "invalid prefix checkpoint size/context");
        // Complete all validation before touching current KV or graph state.
        for (const auto & kv : p_->cache)
            for (auto * tensor : {kv.k, kv.v, kv.index})
                require(tensor && tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor),
                        "prefix restore requires contiguous F16 cache tensors");
        started = true;
        reset();
        p_->poisoned = true;
        ggml_backend_synchronize(p_->backend);
        size_t offset = 0;
        for (const auto & kv : p_->cache) {
            for (auto * tensor : {kv.k, kv.v, kv.index}) {
                const size_t bytes = (tensor == kv.index ? tensor->nb[1] : tensor->nb[2]) * size_t(position);
                ggml_backend_tensor_set(tensor, checkpoint->bytes_.data() + offset, 0, bytes);
                offset += bytes;
            }
        }
        ggml_backend_synchronize(p_->backend);
        p_->pos = position;
        p_->last_n = 0;
        p_->hidden.clear();
        p_->poisoned = false;
        return true;
    } catch (const std::exception & exception) {
        // A failed copy may have changed KV. A fresh prefill must start at zero.
        if (started) reset();
        error = exception.what();
        return false;
    }
}

void Engine::reset() {
    p_->clear_dense_decode_cache(true);
    // A failed pass can leave incomplete or non-finite future rows. Causal
    // -infinity masks cannot neutralize NaNs before MSA block selection.
    if (p_->poisoned) {
        // A failed lazy CPU graph build may have retained metadata without an
        // allocated output. Rebuild those graphs on the next request.
        for (auto & graph : p_->cpu_decode_graphs) graph.reset();
        if (p_->state_buffer) {
            ggml_backend_synchronize(p_->backend);
            ggml_backend_buffer_clear(p_->state_buffer, 0);
        }
    }
    p_->pos = 0; p_->last_n = 0; p_->hidden.clear(); p_->poisoned = false; p_->decode_chain_active = false;
    p_->prefill_chain_active = false;
}

bool Engine::forward(const int32_t * tokens, int n, std::string & error) {
    try {
        const auto prepare_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        require(p_->ready && !p_->poisoned, "not initialized or reset required after a failed pass");
        require(tokens && n > 0 && n <= p_->cfg.max_chunk && p_->pos + n <= p_->cfg.context, "invalid pass size/context");
        const auto & g = p_->g;
        require(msa_chunk_limit(p_->pos,n,g.index_heads,g.block)==n,
                "pass exceeds MSA int32 indexing; split it with prefill_chunk()");
        for (int i = 0; i < n; ++i) require(tokens[i] >= 0 && tokens[i] < g.vocab, "token id out of range");
        // Validation above has not touched KV, hidden, graph state or streams.
        // A caller can re-plan a rejected pass without resetting valid state.
        require(!p_->cfg.cancelled || !p_->cfg.cancelled(), "cancelled");
        p_->poisoned = true;
        p_->decode_chain_active = p_->gpu_decode_chain && n == 1;
        p_->prefill_chain_active = p_->gpu_prefill_chain && n > 1 && n >= p_->cfg.gpu_experts_min_tokens;
        auto emb = p_->host.get("token_embd.weight");
        p_->hidden.resize(size_t(g.embd * n));
        const auto * traits = ggml_get_type_traits(emb->type);
        require(emb->type == GGML_TYPE_F32 || traits->to_float, "no embedding dequantizer");
        for (int i = 0; i < n; ++i) {
            const auto * row = static_cast<const uint8_t *>(emb->data) + size_t(tokens[i]) * emb->nb[1];
            auto * dst = p_->hidden.data() + size_t(i) * g.embd;
            if (emb->type == GGML_TYPE_F32) std::memcpy(dst, row, size_t(g.embd) * sizeof(float));
            else traits->to_float(row, dst, g.embd);
        }
        if (p_->cfg.reuse_allocations) {
            const auto nk = kv_length(p_->pos + n, g);
            if (!p_->attention_inputs || p_->attention_inputs->n != n || p_->attention_inputs->nk != nk) {
                p_->clear_dense_decode_cache(); // references the old shared positions/mask/bias
                // Previous forward is complete. Release the old quadratic mask
                // before allocating its replacement, rather than holding both
                // until unique_ptr move-assignment finishes.
                p_->attention_inputs.reset();
                p_->attention_inputs = std::make_unique<AttentionInputs>(p_->backend, g, n, nk, false);
            }
            auto & inputs = *p_->attention_inputs;
            p_->fill_attention(inputs.positions, inputs.mask, inputs.bias, inputs.blocks, n, nk, nk / g.block);
        }
        if (p_->prefill_chain_active) {
            p_->prefill_hidden->prepare(p_->backend,g.embd,n);
            p_->prefill_hidden->seed(p_->hidden); // only the embedding input crosses H2D
        }
        const auto started = std::chrono::steady_clock::now();
        const auto previous_host_reads = p_->host_reads.reads;
        const auto previous_host_reuses = p_->host_reads.reuses;
        const auto previous_host_growths = p_->host_reads.growths;
        const auto previous_host_bytes = p_->host_reads.bytes;
        const double prepare_seconds = p_->cfg.profile ? Impl::elapsed(prepare_started) : 0;
        p_->timings = {};
        for (int l = 0; l < g.layers; ++l) {
            require(!p_->cfg.cancelled || !p_->cfg.cancelled(), "cancelled");
            p_->layer(l, n);
        }
        // The head consumes prefill_hidden directly. CPU hidden still contains
        // only the initial embeddings and must not be used for this forward.

        if (p_->cfg.profile) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::fprintf(stderr, "M31 forward pos=%lld n=%d seconds=%.6f tps=%.3f\n",
                         (long long)p_->pos, n, seconds, n / seconds);
            const auto & t = p_->timings;
            std::fprintf(stderr, "M31 stages pos=%lld n=%d prepare_ms=%.3f dense_ms=%.3f routed_setup_ms=%.3f routed_compute_ms=%.3f routed_read_ms=%.3f residual_ms=%.3f decode_cache=%d\n",
                         (long long)p_->pos, n, 1000*prepare_seconds, 1000*t.dense, 1000*t.routed_setup, 1000*t.routed_compute,
                         1000*t.routed_read, 1000*t.residual, int(p_->cfg.cpu_decode_cache));
            std::fprintf(stderr, "M31 dense-stages pos=%lld n=%d build_ms=%.3f allocate_ms=%.3f upload_ms=%.3f compute_ms=%.3f read_ms=%.3f destroy_ms=%.3f dense_cache=%d\n",
                         (long long)p_->pos, n, 1000*t.dense_build, 1000*t.dense_allocate, 1000*t.dense_upload,
                         1000*t.dense_compute, 1000*t.dense_read, 1000*t.dense_destroy, int(p_->cfg.dense_decode_cache));
            if (p_->reuse_host_reads) std::fprintf(stderr,
                "M31 host-reads pos=%lld n=%d reads=%llu reuses=%llu growths=%llu bytes=%llu retained_bytes=%zu\n",
                (long long)p_->pos, n,
                (unsigned long long)(p_->host_reads.reads-previous_host_reads),
                (unsigned long long)(p_->host_reads.reuses-previous_host_reuses),
                (unsigned long long)(p_->host_reads.growths-previous_host_growths),
                (unsigned long long)(p_->host_reads.bytes-previous_host_bytes), p_->host_retained_bytes());
            if (n > 1) {
                size_t free_bytes = 0, total_bytes = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(p_->backend), &free_bytes, &total_bytes);
                size_t dense_weights = 0;
                for (auto buffer : p_->device.buffers) dense_weights += ggml_backend_buffer_get_size(buffer);
                const auto * experts = p_->gpu_experts ? &p_->gpu_experts->stats() : nullptr;
                std::fprintf(stderr,
                    "M31 memory pos=%lld n=%d dense_workspace=%zu attention_inputs=%zu expert_scratch=%zu prefill_hidden=%zu kv=%zu expert_pool=%zu dense_weights=%zu backend_free=%zu backend_total=%zu\n",
                    (long long)p_->pos, n,
                    p_->dense_workspace ? ggml_gallocr_get_buffer_size(p_->dense_workspace->allocator, 0) : 0,
                    p_->attention_inputs ? ggml_backend_buffer_get_size(p_->attention_inputs->buffer) : 0,
                    experts ? experts->scratch_bytes : 0,
                    p_->prefill_hidden ? p_->prefill_hidden->bytes() : 0,
                    p_->state_buffer ? ggml_backend_buffer_get_size(p_->state_buffer) : 0,
                    experts ? experts->pool_bytes : 0, dense_weights, free_bytes, total_bytes);
            }
            if (p_->timings.shared_batches) std::fprintf(stderr,
                "M31 shared-stages pos=%lld n=%d batches=%llu enqueue_ms=%.3f wait_ms=%.3f read_ms=%.3f\n",
                (long long)p_->pos, n, (unsigned long long)p_->timings.shared_batches,
                1000*p_->timings.shared_enqueue, 1000*p_->timings.shared_wait, 1000*p_->timings.shared_read);
        }
        if (p_->gpu_experts && p_->cfg.profile) p_->gpu_experts->print_profile("forward");
        p_->pos += n;
        p_->last_n = n;
        p_->poisoned = false;
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
}

bool Engine::logits(int row, int count, std::vector<float> & out, std::string & error) {
    bool started = false;
    try {
        const auto head_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        require(p_->ready && !p_->poisoned && row >= 0 && count > 0 && row <= p_->last_n && count <= p_->last_n - row, "invalid logits rows/state");
        started = true;
        Impl::HeadGraph temporary;
        const bool stable_decode = p_->cfg.dense_decode_cache && count == 1 && p_->last_n == 1 && !p_->cfg.tap;
        auto & storage = stable_decode ? p_->head_decode_graph : temporary;
        if (!storage.graph) {
            storage.graph = std::make_unique<Graph>(p_->backend, p_->cfg, p_->pos - p_->last_n + row,
                stable_decode ? nullptr : p_->dense_workspace.get(),
                stable_decode ? 1024 * 1024 : 32 * 1024 * 1024,
                stable_decode ? 512 : 8192);
            auto & b = *storage.graph;
            storage.input = p_->prefill_chain_active ? nullptr : b.input(GGML_TYPE_F32, p_->g.embd, count);
            auto head_input = storage.input;
            if (p_->prefill_chain_active) {
                auto hidden = b.borrow(p_->prefill_hidden->current());
                head_input = ggml_view_2d(b.c,hidden,p_->g.embd,count,hidden->nb[1],size_t(row)*hidden->nb[1]);
            } else if (p_->decode_chain_active)
                head_input = p_->previous_hidden(b,int(p_->g.layers-1),storage.input);
            auto norm = b.tap(rms(b, head_input, p_->dense->get("output_norm.weight"), p_->g.rms_eps), "result_norm");
            storage.output = b.tap(ggml_mul_mat(b.c, p_->dense->get("output.weight"), norm), "result_output");
            b.allocate({storage.output});
        }
        auto & b = *storage.graph;
        if (storage.input)
            ggml_backend_tensor_set(storage.input, p_->hidden.data() + size_t(row) * p_->g.embd, 0, ggml_nbytes(storage.input));
        const double setup_seconds = p_->cfg.profile ? Impl::elapsed(head_started) : 0;
        const auto compute_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        b.compute();
        const double compute_seconds = p_->cfg.profile ? Impl::elapsed(compute_started) : 0;
        const auto read_started = p_->cfg.profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        out = Graph::read(storage.output);
        require(std::all_of(out.begin(), out.end(), [](float value) { return std::isfinite(value); }),
                "nonfinite logits");
        if (p_->cfg.profile) std::fprintf(stderr,
            "M31 head pos=%lld n=%d setup_ms=%.3f compute_ms=%.3f read_ms=%.3f\n",
            (long long)(p_->pos-p_->last_n+row), count, 1000*setup_seconds, 1000*compute_seconds, 1000*Impl::elapsed(read_started));
        return true;
    } catch (const std::exception & e) { if (started) p_->poisoned = true; error = e.what(); return false; }
}

} // namespace strata::m31
