#include "experts.hpp"
#include "../prefill/pipeline.hpp"
#include "../prefill/background.hpp"
#include "pack_spans.hpp"
#include "../prefill/slot_leases.hpp"
#include "../prefill/expert_budget.hpp"
#include "../prefill/short/shortprefill_cost.hpp"
#include "../prefill/short/shortprefill_cpu_lane.hpp"

#include "ggml-alloc.h"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef M31_REGISTERED_ARENA_CUDA
#include "registered_arena.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::m31 {
namespace {
void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("M31 GPU experts: " + message);
}
void clear_captures(ggml_backend_t backend) {
#ifdef M31_ENGINE_CUDA
    ggml_backend_cuda_graph_cache_clear(backend);
#else
    (void) backend;
#endif
}
size_t aligned(size_t value, size_t alignment) {
    check(alignment && value <= std::numeric_limits<size_t>::max() - (alignment - 1), "alignment overflow");
    return (value + alignment - 1) / alignment * alignment;
}
size_t common_multiple(size_t a, size_t b) {
    check(a && b, "invalid stride multiple");
    const size_t divided = a / std::gcd(a, b);
    check(divided <= std::numeric_limits<size_t>::max() / b, "stride multiple overflow");
    return divided * b;
}


struct Context {
    ggml_context * value = nullptr;
    explicit Context(size_t bytes) : value(ggml_init({bytes, nullptr, true})) {
        check(value != nullptr, "metadata allocation failed");
    }
    ~Context() { if (value) ggml_free(value); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
};
} // namespace

struct GpuExperts::Impl {
    using Clock = std::chrono::steady_clock;
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now()-start).count();
    }
    struct Layer {
        std::array<ggml_type, 3> types{};
        std::array<const ExpertArena::Slice *, 3> sources{};
        std::array<std::vector<float>, 3> scales;
        std::array<bool, 3> has_scale{};
        std::array<size_t, 3> offsets{}, raw_bytes{}, allocation_bytes{};
        size_t span = 0;
    };
    struct Entry {
        size_t slot = 0;
        uint64_t last_used = 0;
    };
    Geometry g;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t pool = nullptr;
    std::unique_ptr<CpuPackSpans> pack_pool;
    std::unique_ptr<PrefillUploads> uploads;
    std::unique_ptr<BackgroundPrefillUploads> background;
    PrefillUploads::Copier copier;
    struct PendingPrefetch { uint64_t key; size_t slot; };
    std::vector<PendingPrefetch> pending_prefetch;
    std::vector<std::vector<int>> prefetch_order;
    std::unique_ptr<Context> slot_metadata;
    ggml_tensor * pool_proxy = nullptr;
    std::vector<ggml_tensor *> slot_proxies;
    std::vector<size_t> free_slots;
    ggml_gallocr_t allocator = nullptr;
    std::vector<Layer> layers;
    std::map<uint64_t, Entry> entries;
    std::set<uint64_t> active_read_keys;
    SlotLeases leases;
    std::vector<size_t> foreground_writes;
    Stats stat;
    uint64_t clock = 0;
    int max_chunk = 0;
    bool ready = false;
    bool suppress_prefetch = false;
    std::unique_ptr<ShortPrefillCpuLane> short_cpu_lane;
    bool profile = false;
    bool compact_uploads = false;
    size_t prefetch_limit = 0;
    size_t upload_depth = 8, upload_streams = 1;
#ifdef M31_REGISTERED_ARENA_CUDA
    std::unique_ptr<GpuRegisteredArena> owned_registered_arena;
    const GpuRegisteredArena * registered_arena=nullptr;
#endif

    // Called only after all borrowed CPU jobs and GPU readers have joined.
    // Discard any partly published upload metadata; a reset may then retry.
    void recover_uploads() noexcept {
        if(background)background->drain();
        pending_prefetch.clear();
        if(!uploads)return;
        try {
            uploads.reset();
            ggml_backend_synchronize(backend);
            entries.clear();active_read_keys.clear();
            leases.reset(stat.slot_count);foreground_writes.clear();
            free_slots.clear();
            for(size_t slot=0;slot<stat.slot_count;++slot)
                free_slots.push_back(stat.slot_count-slot-1);
            stat.occupied_slots=0;
            uploads=std::make_unique<PrefillUploads>(backend,stat.slot_stride,upload_depth,
                true,256*1024*1024,copier,upload_streams,profile);
        } catch (...) {
            ready=false; // Re-creating the resources failed; reuse is unsafe.
        }
    }

    ~Impl() {
        // A worker borrows arena role spans and destination pool metadata.
        // Join it before any of those can be freed, including failure shutdown.
        short_cpu_lane.reset();
        background.reset();
        if (uploads) {
            const auto & u = uploads->stats();
            std::fprintf(stderr, "M31 prefill uploads: batches=%llu uploads=%llu upload_bytes=%llu staging_bytes=%zu reuses=%llu pinned=%d async=%d\n",
                         (unsigned long long)u.batches, (unsigned long long)u.uploads,
                         (unsigned long long)u.upload_bytes, u.staging_bytes,
                         (unsigned long long)u.staging_reuses, int(u.pinned), int(u.asynchronous));
            uploads.reset(); // Transfer/computation streams drain before slot tensors/pool die.
        }
#ifdef M31_REGISTERED_ARENA_CUDA
        // All DMA readers have drained; the borrowed RAM arena outlives us.
        owned_registered_arena.reset();
        registered_arena=nullptr;
#endif
        if (backend) {
            ggml_backend_synchronize(backend);
            clear_captures(backend);
        }
        if (allocator) ggml_gallocr_free(allocator);
        if (pool) ggml_backend_buffer_free(pool);
    }
    uint64_t key(int layer, int expert) const {
        return uint64_t(layer) * uint64_t(g.experts) + uint64_t(expert);
    }
    size_t reserve_slot(const std::set<uint64_t> & protected_keys, bool required) {
        if (!free_slots.empty()) {
            const size_t slot=free_slots.back();
            free_slots.pop_back();
            leases.reserve_write(slot);
            return slot;
        }
        auto victim=entries.end();
        for (auto it=entries.begin();it!=entries.end();++it)
            if (!protected_keys.count(it->first) && !active_read_keys.count(it->first) && leases.writable(it->second.slot) &&
                (victim==entries.end() || it->second.last_used<victim->second.last_used)) victim=it;
        if (victim==entries.end()) {
            check(!required, "all cache slots are protected; increase expert cache budget");
            return std::numeric_limits<size_t>::max();
        }
        const size_t slot=victim->second.slot;
        leases.reserve_write(slot);
        entries.erase(victim);
        ++stat.evictions;
        return slot;
    }
    void begin_prefetch(int layer, int n, const std::set<uint64_t> & current_keys) {

        // Speculative full-next-layer work is worthwhile only for a substantial
        // layer batch. Router/histogram/compute shape of the current layer is unchanged.
        if (suppress_prefetch || !background || n<128 || layer+1>=g.layers || !g.routed(layer+1)) return;
        check(pending_prefetch.empty() && !background->active(), "uncollected prefetch");
        const int next=layer+1;
        const auto & source=layers[size_t(next)];
        auto protected_keys=current_keys;
        // A warm next-layer entry must not be selected as another prefetch's victim.
        for (int expert=0;expert<g.experts;++expert)
            if (entries.count(key(next,expert))) protected_keys.insert(key(next,expert));
        std::vector<BackgroundPrefillUploads::Job> jobs;
        jobs.reserve(size_t(g.experts));
        for (const int expert:prefetch_order[size_t(next)]) {
            if (prefetch_limit && jobs.size()>=prefetch_limit) break;
            const auto id=key(next,expert);
            if (entries.count(id)) continue;
            const auto slot=reserve_slot(protected_keys,false);
            if (slot==std::numeric_limits<size_t>::max()) break;
            pending_prefetch.push_back({id,slot});
            BackgroundPrefillUploads::Job job;
            job.destination=slot_proxies[slot];
            for (size_t r=0;r<3;++r) job.spans[r]={source.offsets[r],
                source.sources[r]->data+size_t(expert)*source.sources[r]->expert_stride,
                source.raw_bytes[r],source.allocation_bytes[r]};
            jobs.push_back(job);
        }
        if (!jobs.empty()) {
#ifdef M31_REGISTERED_ARENA_CUDA
            background->start(std::move(jobs),compact_uploads,registered_arena);
#else
            background->start(std::move(jobs),compact_uploads);
#endif
        }
    }
    void finish_prefetch() {
        if (!background || !background->active()) return;
        const auto wait_start=profile?Clock::now():Clock::time_point{};
        const auto copied=background->wait(); // Throws before publishing partially written slots.
        if (profile) stat.prefetch_wait_seconds+=elapsed(wait_start);
        for (const auto & item:pending_prefetch) {
            leases.publish_write(item.slot); // background wait joined the real DMA stream
            entries.emplace(item.key,Entry{item.slot,++clock});
        }
        stat.misses+=pending_prefetch.size();
        stat.upload_bytes+=copied.upload_bytes;
        ++stat.prefetch_batches;
        stat.prefetched_experts+=pending_prefetch.size();
        stat.prefetch_bytes+=copied.upload_bytes;
        stat.prefetch_pack_seconds+=copied.pack_seconds;
        stat.prefetch_staging_wait_seconds+=copied.staging_wait_seconds;
        stat.prefetch_submit_seconds+=copied.submit_seconds;
        stat.occupied_slots=entries.size();
        pending_prefetch.clear();
    }
    size_t load(int layer, int expert, const std::set<uint64_t> & protected_keys) {
        const uint64_t id = key(layer, expert);
        auto found = entries.find(id);
        if (found != entries.end()) {
            found->second.last_used = ++clock;
            ++stat.hits;
            return found->second.slot;
        }
        const size_t slot=reserve_slot(protected_keys,true);
        foreground_writes.push_back(slot);
        const auto & source = layers[size_t(layer)];
        const auto upload_start = profile ? Clock::now() : Clock::time_point{};
        if (uploads) {
            // Identical original role bytes and zero padding, packed into a bounded
            // two-buffer ring while the preceding slot is transferred on its stream.
            const auto spans = {
                PrefillUploads::Span{source.offsets[0], source.sources[0]->data + size_t(expert)*source.sources[0]->expert_stride, source.raw_bytes[0], source.allocation_bytes[0]},
                PrefillUploads::Span{source.offsets[1], source.sources[1]->data + size_t(expert)*source.sources[1]->expert_stride, source.raw_bytes[1], source.allocation_bytes[1]},
                PrefillUploads::Span{source.offsets[2], source.sources[2]->data + size_t(expert)*source.sources[2]->expert_stride, source.raw_bytes[2], source.allocation_bytes[2]},
            };
#ifdef M31_REGISTERED_ARENA_CUDA
            if (registered_arena) uploads->enqueue_registered_spans(slot_proxies[slot], spans, *registered_arena);
            else
#endif
            if (compact_uploads) uploads->enqueue_spans(slot_proxies[slot], spans);
            /* spans retain role padding even when fixed-slot alignment is skipped. */
            else uploads->enqueue(slot_proxies[slot], {
                {source.offsets[0], source.sources[0]->data + size_t(expert)*source.sources[0]->expert_stride, source.raw_bytes[0]},
                {source.offsets[1], source.sources[1]->data + size_t(expert)*source.sources[1]->expert_stride, source.raw_bytes[1]},
                {source.offsets[2], source.sources[2]->data + size_t(expert)*source.sources[2]->expert_stride, source.raw_bytes[2]},
            });
        } else throw std::logic_error("M31 native uploader is missing");
        if (profile) stat.upload_submit_seconds += elapsed(upload_start);
        bool compact_transfer = uploads && compact_uploads;
#ifdef M31_REGISTERED_ARENA_CUDA
        compact_transfer = compact_transfer || bool(registered_arena);
#endif
        stat.upload_bytes += compact_transfer ?
            std::accumulate(source.allocation_bytes.begin(), source.allocation_bytes.end(), size_t(0)) : stat.slot_stride;
        ++stat.misses;
        entries.emplace(id, Entry{slot, ++clock});
        stat.occupied_slots = entries.size();
        return slot;
    }
    void publish_foreground() {
        // finish() has established copy-complete dependencies on the compute
        // backend. No new upload can reuse these slots until their read lease ends.
        for(size_t slot:foreground_writes)leases.publish_write(slot);
        foreground_writes.clear();
    }
    void acquire_readers(const std::set<uint64_t> & keys) {
        check(active_read_keys.empty(),"overlapping expert consumers are unsupported");
        for(uint64_t id:keys)leases.begin_read(entries.at(id).slot);
        active_read_keys=keys;
    }
    void release_readers() noexcept {
        // All native FFN paths and their exceptional guards synchronize this
        // actual backend before the ownership lease may be released.
        ggml_backend_synchronize(backend);
        leases.finish_reads();active_read_keys.clear();
    }
    ggml_tensor * weight_view(ggml_context * c, int layer, int role) const {
        const auto & info = layers[size_t(layer)];
        const int64_t columns = role == 2 ? g.expert_ff : g.embd;
        const int64_t rows = role == 2 ? g.embd : g.expert_ff;
        auto tensor = ggml_new_tensor_3d(c, info.types[size_t(role)], columns, rows, int64_t(stat.slot_count));
        tensor->nb[2] = stat.slot_stride;
        tensor->nb[3] = stat.slot_stride * stat.slot_count;
        check(stat.slot_stride % ggml_type_size(tensor->type) == 0, "CUDA expert stride is not an integral number of quant blocks");
        const size_t offset = info.offsets[size_t(role)];
        check(offset + ggml_nbytes(tensor) <= stat.pool_bytes, "weight view exceeds slot pool");
        check(offset + ggml_backend_buft_get_alloc_size(buffer_type, tensor) <= stat.pool_bytes,
              "weight view padding exceeds slot pool");
        // Explicit borrowed reinterpret view over an I8 byte-storage tensor. Calling
        // tensor_alloc() on a naked quant tensor would zero its trailing quant padding:
        // the last slot may contain another layer with different role offsets/types.
        // CUDA's view initializer deliberately skips that write. All required padding
        // was already zeroed inside the selected expert's own slot before its upload.
        tensor->view_src = pool_proxy;
        tensor->view_offs = offset;
        check(ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS, "cannot bind borrowed weight view");
        return tensor;
    }
    void execute(int layer, int n, const std::vector<float> & x,
                               const std::vector<int32_t> & mapped_ids,
                               const std::vector<int32_t> & original_ids,
                               const std::vector<float> & route_weights,
                               const std::array<std::vector<float>, 3> & scales,
                               std::vector<float> & output,
                               ggml_backend_event_t weights_ready = nullptr,
                               const std::set<uint64_t> * next_prefetch_keys = nullptr) {
        (void)original_ids;
        (void)weights_ready;
        (void)next_prefetch_keys;
        Context context(8 * 1024 * 1024);
        auto c = context.value;
        auto graph = ggml_new_graph_custom(c, 8192, false);
        struct CaptureGuard {
            ggml_backend_t backend;
            ~CaptureGuard() { ggml_backend_synchronize(backend); clear_captures(backend); }
        } guard{backend};
        auto input = ggml_new_tensor_3d(c, GGML_TYPE_F32, g.embd, 1, n);
        auto ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, g.used, n);
        auto rw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, g.used, n);
        for (auto tensor : {input, ids, rw}) ggml_set_input(tensor);
        std::array<ggml_tensor *, 3> scale_inputs{};
        const auto & info = layers[size_t(layer)];
        auto mm = [&](int role, ggml_tensor * activation) {
            auto weight = weight_view(c, layer, role);
            auto out = ggml_mul_mat_id(c, weight, activation, ids);
            if (weight->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(out, GGML_PREC_BF16);
            if (stat.q8_activations) ggml_prec_set_src(out, GGML_PREC_Q8, 1);
            if (info.has_scale[size_t(role)]) {
                auto scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, int64_t(stat.slot_count));
                ggml_set_input(scale);
                scale_inputs[size_t(role)] = scale;
                auto s = ggml_repeat_4d(c, ggml_reshape_3d(c, scale, 1, int64_t(stat.slot_count), 1),
                                      1, int64_t(stat.slot_count), n, 1);
                out = ggml_mul(c, out, ggml_get_rows(c, s, ids));
            }
            return out;
        };
        // Preserve the original [used,n] shape: GPU dispatch/activation quantization
        // must not change merely because an expert received fewer than nine tokens.
        auto gate=mm(0,input),up=mm(1,input);
        auto activation=ggml_swiglu_oai(c,gate,up,1.702f,7.0f);
        auto down = mm(2, activation);
        auto weighted = ggml_mul(c, down, rw);
        ggml_tensor * result = nullptr;
        for (int64_t rank = 0; rank < g.used; ++rank) {
            auto term = ggml_view_2d(c, weighted, g.embd, n, weighted->nb[2], size_t(rank) * weighted->nb[1]);
            result = result ? ggml_add(c, result, term) : term;
        }
        result = ggml_cont(c, result);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            check(ggml_backend_dev_supports_op(device, node), std::string("backend does not support ") + ggml_op_name(node->op));
        }
        // Graph shapes/scale topology change across layers. Rebuild the alias plan,
        // retaining sufficiently large existing buffers rather than allocating afresh.
        check(ggml_gallocr_reserve(allocator, graph), "graph scratch reserve failed");
        check(ggml_gallocr_alloc_graph(allocator, graph), "graph scratch allocation failed");
        stat.scratch_bytes = std::max(stat.scratch_bytes, ggml_gallocr_get_buffer_size(allocator, 0));
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, mapped_ids.data(), 0, mapped_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(rw, route_weights.data(), 0, route_weights.size() * sizeof(float));
        for (int role = 0; role < 3; ++role)
            if (scale_inputs[size_t(role)])
                ggml_backend_tensor_set(scale_inputs[size_t(role)], scales[size_t(role)].data(), 0, stat.slot_count * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
        output.resize(size_t(g.embd) * size_t(n));
        check(ggml_is_contiguous(result) && ggml_nelements(result) == int64_t(output.size()), "invalid final expert result layout");
        ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
        ++stat.graph_batches;
        return;
    }
#include "device_experts.hpp"
};

GpuExperts::GpuExperts() : p_(std::make_unique<Impl>()) {}
GpuExperts::~GpuExperts() = default;

uint64_t GpuExperts::automatic_budget_bytes(const GgufModel & model,const Geometry & g,
                                           ggml_backend_t backend) {
    check(backend && ggml_backend_get_device(backend),"automatic budget requires an explicit GPU backend");
    const auto device=ggml_backend_get_device(backend);
    check(ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU,"automatic budget requires a GPU device");
    const auto buft=ggml_backend_get_default_buffer_type(backend);
    const size_t alignment=ggml_backend_buft_get_alignment(buft);
    size_t stride_multiple=alignment,maximum_span=0;
    const char * roles[]={"gate","up","down"};
    Context metadata(64*1024);
    for(int layer=int(g.dense_lead);layer<g.layers;++layer) {
        ggml_reset(metadata.value);
        size_t span=0;
        for(int role=0;role<3;++role) {
            const std::string name=layer_name(layer,std::string("ffn_")+roles[role]+"_exps.weight");
            const auto * info=model.find(name);
            check(info&&info->type<GGML_TYPE_COUNT,"invalid automatic-budget expert tensor "+name);
            const auto type=ggml_type(info->type);
            auto * tensor=ggml_new_tensor_2d(metadata.value,type,role==2?g.expert_ff:g.embd,role==2?g.embd:g.expert_ff);
            const size_t bytes=ggml_backend_buft_get_alloc_size(buft,tensor);
            const size_t offset=aligned(span,alignment),extent=aligned(bytes,alignment);
            check(offset<=SIZE_MAX-extent,"automatic-budget slot overflow");
            span=offset+extent;
            stride_multiple=common_multiple(stride_multiple,ggml_type_size(type));
        }
        maximum_span=std::max(maximum_span,span);
    }
    const size_t stride=aligned(maximum_span,stride_multiple);
    size_t free_bytes=0,total_bytes=0;
    ggml_backend_dev_memory(device,&free_bytes,&total_bytes);
    return choose_expert_budget(free_bytes,stride,uint64_t(g.experts));
}

void GpuExperts::init(const GgufModel & model, const Geometry & g, const ExpertArena & arena,
                      ggml_backend_t backend, double cache_gib, int max_chunk, bool q8_activations, const Config & config) {
    check(!p_->ready && p_->backend == nullptr, "repeated initialization");
    check(backend != nullptr && ggml_backend_get_device(backend) != nullptr, "backend/device is missing");
    check(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU,
          "this opt-in path requires a GPU backend");
    check(std::isfinite(cache_gib) && cache_gib > 0 && cache_gib <= 512,
          "cache budget must be finite and positive (GiB)");
    check(max_chunk > 0 && max_chunk <= 32768 && g.embd > 0 && g.expert_ff > 0 && g.layers > g.dense_lead &&
          g.used > 0 && g.used <= g.experts && g.experts <= 4096 && g.layers <= 1024,
          "invalid geometry/chunk");
    check(arena.buffer() != nullptr && arena.layers().size() == size_t(g.layers), "complete RAM arena is required");
    p_->g = g;
    p_->backend = backend;
    p_->device = ggml_backend_get_device(backend);
    p_->buffer_type = ggml_backend_get_default_buffer_type(backend);
    p_->max_chunk = max_chunk;
    p_->profile = config.profile;
    p_->compact_uploads = true;
    p_->prefetch_limit = std::min<size_t>(config.prefetch_limit, size_t(g.experts));
    p_->upload_depth = config.upload_depth;
    p_->upload_streams = config.upload_streams;
    p_->prefetch_order.resize(size_t(g.layers));
    for (auto & order:p_->prefetch_order) {
        order.resize(size_t(g.experts));
        std::iota(order.begin(),order.end(),0);
    }
    p_->stat.budget_bytes = size_t(cache_gib * 1073741824.0);
    p_->stat.q8_activations = q8_activations;
    p_->layers.resize(size_t(g.layers));
    const size_t alignment = ggml_backend_buft_get_alignment(p_->buffer_type);
    size_t stride_multiple = alignment, maximum_span = 0;
    const char * role_names[] = {"gate", "up", "down"};
    Context metadata(64 * 1024);
    for (int l = 0; l < g.layers; ++l) {
        if (!g.routed(l)) continue;
        auto & layer = p_->layers[size_t(l)];
        ggml_reset(metadata.value);
        for (int r = 0; r < 3; ++r) {
            const std::string name = layer_name(l, std::string("ffn_") + role_names[r] + "_exps.weight");
            size_t shard = 0;
            const auto * info = model.find(name, &shard);
            const int64_t n0 = r == 2 ? g.expert_ff : g.embd;
            const int64_t n1 = r == 2 ? g.embd : g.expert_ff;
            const std::vector<uint64_t> shape = {uint64_t(n0), uint64_t(n1), uint64_t(g.experts)};
            check(info && info->shape == shape && info->type < GGML_TYPE_COUNT && model.in_bounds(*info, shard), "invalid weight " + name);
            const auto type = ggml_type(info->type);
            layer.types[size_t(r)] = type;
            layer.sources[size_t(r)] = arena.find(name);
            check(layer.sources[size_t(r)] != nullptr, "missing arena slice " + name);
            auto tensor = ggml_new_tensor_2d(metadata.value, type, n0, n1);
            layer.raw_bytes[size_t(r)] = ggml_nbytes(tensor);
            layer.allocation_bytes[size_t(r)] = ggml_backend_buft_get_alloc_size(p_->buffer_type, tensor);
            layer.offsets[size_t(r)] = aligned(layer.span, alignment);
            layer.span = layer.offsets[size_t(r)] + aligned(layer.allocation_bytes[size_t(r)], alignment);
            check(layer.raw_bytes[size_t(r)] <= layer.sources[size_t(r)]->expert_stride, "invalid arena stride");
            // CUDA casts nb[2]/type_size to a quant-block stride: rounding to 128
            // bytes alone silently truncates the expert offset for IQ3/IQ4/NVFP4.
            stride_multiple = common_multiple(stride_multiple, ggml_type_size(type));
            layer.scales[size_t(r)].assign(size_t(g.experts), 1.0f);
            const std::string scale_name = name.substr(0, name.size() - 6) + "scale";
            const auto * scales = model.find(scale_name, &shard);
            layer.has_scale[size_t(r)] = scales != nullptr;
            if (scales) {
                check(scales->type == GGML_TYPE_F32 && scales->shape == std::vector<uint64_t>{uint64_t(g.experts)} &&
                      model.in_bounds(*scales, shard), "invalid external scales " + scale_name);
                std::memcpy(layer.scales[size_t(r)].data(), model.shard(shard).tensor_data(*scales), size_t(g.experts) * sizeof(float));
            } else check(type != GGML_TYPE_NVFP4, "missing external NVFP4 scale");
            for (float scale : layer.scales[size_t(r)]) check(std::isfinite(scale), "non-finite external scale");
        }
        maximum_span = std::max(maximum_span, layer.span);
    }
    p_->stat.slot_stride = aligned(maximum_span, stride_multiple);
    p_->stat.minimum_budget_bytes = p_->stat.slot_stride;
    check(p_->stat.budget_bytes >= p_->stat.slot_stride,
          "cache budget must fit one complete expert slot: minimum bytes=" + std::to_string(p_->stat.slot_stride));
    const size_t total_experts = size_t(g.experts) * size_t(g.layers - g.dense_lead);
    p_->stat.slot_count = std::min(p_->stat.budget_bytes / p_->stat.slot_stride, total_experts);
    p_->leases.reset(p_->stat.slot_count);
    p_->stat.pool_bytes = p_->stat.slot_count * p_->stat.slot_stride;
    check(p_->stat.slot_count && p_->stat.slot_count <= size_t(std::numeric_limits<int32_t>::max()), "invalid slot count");
    p_->pool = ggml_backend_buft_alloc_buffer(p_->buffer_type, p_->stat.pool_bytes);
    check(p_->pool != nullptr, "fixed VRAM slot allocation failed (budget excludes dense/KV/graph scratch)");
    check(ggml_backend_buffer_get_size(p_->pool) <= p_->stat.budget_bytes, "backend allocated more than the requested cache budget");
    ggml_backend_buffer_set_usage(p_->pool, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    p_->stat.resident_bytes = ggml_backend_buffer_get_size(p_->pool);
    p_->stat.peak_resident_bytes = p_->stat.resident_bytes;
    p_->slot_metadata = std::make_unique<Context>((p_->stat.slot_count + 1) * ggml_tensor_overhead() + 4096);
    p_->pool_proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.pool_bytes));
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(p_->pool));
    check(ggml_backend_tensor_alloc(p_->pool, p_->pool_proxy, base) == GGML_STATUS_SUCCESS, "cannot bind pool byte tensor");
    for (size_t slot = 0; slot < p_->stat.slot_count; ++slot) {
        auto proxy = ggml_new_tensor_1d(p_->slot_metadata->value, GGML_TYPE_I8, int64_t(p_->stat.slot_stride));
        check(ggml_backend_tensor_alloc(p_->pool, proxy, base + slot * p_->stat.slot_stride) == GGML_STATUS_SUCCESS,
              "cannot bind slot byte tensor");
        p_->slot_proxies.push_back(proxy);
        p_->free_slots.push_back(p_->stat.slot_count - slot - 1);
    }
    p_->allocator = ggml_gallocr_new(p_->buffer_type);
    check(p_->allocator != nullptr, "graph allocator initialization failed");
    p_->pack_pool = std::make_unique<CpuPackSpans>(config.gpu_pack_threads);
    p_->copier = [owner=p_.get()](const PrefillUploads::CopySpan * spans,size_t n) { owner->pack_pool->copy(spans,n); };
    p_->uploads = std::make_unique<PrefillUploads>(backend,p_->stat.slot_stride,p_->upload_depth,true,256*1024*1024,p_->copier,p_->upload_streams,config.profile);
    p_->stat.pinned_staging = p_->uploads->stats().pinned;
#ifdef M31_REGISTERED_ARENA_CUDA
    if (config.register_arena) {
        p_->registered_arena=arena.registration();
        if (!p_->registered_arena) {
            GpuRegisteredArena::Options options;
            options.limit_bytes=config.register_limit_bytes;
            options.chunk_bytes=config.register_chunk_bytes;
            options.threads=config.register_threads;
            p_->owned_registered_arena=std::make_unique<GpuRegisteredArena>(backend,ggml_backend_buffer_get_base(arena.buffer()),arena.bytes(),options);
            p_->registered_arena=p_->owned_registered_arena.get();
        }
        check(p_->registered_arena->contains(ggml_backend_buffer_get_base(arena.buffer()),arena.bytes()), "RAM arena registration has not completed");
    }
#else
    check(!config.register_arena,"registered direct H2D is unavailable in this build");
#endif
    if (p_->prefetch_limit) p_->background=std::make_unique<BackgroundPrefillUploads>(backend,p_->stat.slot_stride,p_->copier,p_->upload_depth,p_->upload_streams,config.profile);
    if (p_->stat.slot_count < size_t(g.experts)*2) std::fprintf(stderr,
        "M31 GPU prefetch: %zu slots cannot retain two complete layers (%lld); speculative uploads will be bounded by free leases\n",
        p_->stat.slot_count,static_cast<long long>(g.experts*2));
    std::fprintf(stderr, "M31 GPU experts: fixed_slots=%zu slot_stride=%zu pool_bytes=%zu budget=%zu pinned=%d source_q8=%d (layer mul_mat_id; graph scratch separate)\n",
                 p_->stat.slot_count, p_->stat.slot_stride, p_->stat.pool_bytes, p_->stat.budget_bytes,
                 int(p_->stat.pinned_staging), int(p_->stat.q8_activations));
    p_->ready = true;

}

std::vector<float> GpuExperts::forward(int layer, int n, const std::vector<float> & x,
                                      const std::vector<int32_t> & ids, const std::vector<float> & weights) {
    std::vector<float> output;
    forward_into(layer,n,x,ids,weights,output);
    return output;
}

void GpuExperts::forward_into(int layer, int n, const std::vector<float> & x,
                              const std::vector<int32_t> & ids, const std::vector<float> & weights,
                              std::vector<float> & output) {
    check(&output!=&x && &output!=&weights,"output vector must not alias inputs");
    const auto prepare_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    check(p_->ready && p_->g.routed(layer), "uninitialized backend or invalid routed layer");
    check(n > 0 && n <= p_->max_chunk, "invalid token count");
    const auto & g = p_->g;
    check(x.size() == size_t(g.embd) * size_t(n) && ids.size() == size_t(g.used) * size_t(n) && weights.size() == ids.size(),
          "input dimensions differ");
    std::vector<uint8_t> selected_mask(size_t(g.experts), 0);
    std::vector<int> selected;
    std::set<uint64_t> protected_keys;
    for (int t = 0; t < n; ++t) for (int rank = 0; rank < g.used; ++rank) {
        const size_t route = size_t(t) * size_t(g.used) + size_t(rank);
        const int expert = ids[route];
        check(expert >= 0 && expert < g.experts && std::isfinite(weights[route]), "invalid route ID/weight");
        for (int j = 0; j < rank; ++j)
            check(ids[size_t(t) * size_t(g.used) + size_t(j)] != expert, "duplicate top-k expert");
        selected_mask[size_t(expert)] = 1;
    }
    // One cache-key insertion per expert, rather than per route. At large chunk
    // sizes most routes revisit the same 128 experts; preserve ascending order.
    for (int expert = 0; expert < g.experts; ++expert) if (selected_mask[size_t(expert)]) {
        selected.push_back(expert);
        protected_keys.insert(p_->key(layer, expert));
    }
    check(selected.size() <= p_->stat.slot_count,
          "layer needs " + std::to_string(selected.size()) + " distinct expert slots; cache has " + std::to_string(p_->stat.slot_count) +
          "; minimum batch cache bytes=" + std::to_string(selected.size() * p_->stat.slot_stride));
    if (p_->profile) p_->stat.prepare_seconds += Impl::elapsed(prepare_start);
    try {
        // Keep N+1 DMA in flight through its dense attention graph. Publish its
        // entries before any load/reservation can reuse a reserved destination.
        // A failed wait is handled by the same drain-and-discard recovery below.
        p_->finish_prefetch();
        if (p_->uploads) p_->uploads->begin_batch();
        std::vector<int32_t> expert_to_slot(size_t(g.experts), -1);
        std::array<std::vector<float>, 3> scales;
        for (auto & values : scales) values.assign(p_->stat.slot_count, 1.0f);
        for (int expert : selected) {
            const size_t slot = p_->load(layer, expert, protected_keys);
            expert_to_slot[size_t(expert)] = int32_t(slot);
            for (int role = 0; role < 3; ++role)
                scales[size_t(role)][slot] = p_->layers[size_t(layer)].scales[size_t(role)][size_t(expert)];
        }
        std::vector<int32_t> mapped_ids(ids.size());
        const auto map_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        for (size_t i = 0; i < ids.size(); ++i) mapped_ids[i] = expert_to_slot[size_t(ids[i])];
        if (p_->profile) p_->stat.prepare_seconds += Impl::elapsed(map_start);
        const auto wait_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        ggml_backend_event_t weights_ready = nullptr;
        if (p_->uploads) {
            p_->uploads->finish();
        }
        if (p_->profile) p_->stat.upload_wait_seconds += Impl::elapsed(wait_start);
        p_->begin_prefetch(layer,n,protected_keys);
        const auto execute_start = p_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
        check(p_->active_read_keys.empty(),"overlapping expert consumers are unsupported");
        p_->publish_foreground();
        p_->acquire_readers(protected_keys);
        struct ReadersLease {Impl *owner;~ReadersLease(){owner->release_readers();}} readers{p_.get()};
        // execute() drains its raw private stream on success and failure. Keep
        // its slot lease until that completion, independently of ggml's stream.
        p_->execute(layer, n, x, mapped_ids, ids, weights, scales, output, weights_ready,
                    nullptr);
        if (p_->profile) p_->stat.execute_seconds += Impl::elapsed(execute_start);
        ++p_->stat.forwards;
        return;
    } catch (...) {
        p_->recover_uploads();
        throw;
    }
}

#include "device_forward.hpp"



void GpuExperts::print_profile(const char * label) const {
    const auto & s = p_->stat;
    std::fprintf(stderr, "M31 GPU expert profile label=%s forwards=%llu hits=%llu misses=%llu bytes=%llu control_ms=%.3f pack_ms=%.3f upload_submit_ms=%.3f upload_wait_ms=%.3f execute_wall_ms=%.3f\n",
        label ? label : "live", (unsigned long long)s.forwards,
        (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.upload_bytes,
        1000*s.prepare_seconds, 1000*s.pack_seconds, 1000*s.upload_submit_seconds,
        1000*s.upload_wait_seconds, 1000*s.execute_seconds);
    std::fprintf(stderr, "M31 GPU prefetch profile label=%s batches=%llu experts=%llu bytes=%llu join_host_ms=%.3f pack_host_ms=%.3f staging_wait_host_ms=%.3f submit_host_ms=%.3f\n",
        label?label:"live",(unsigned long long)s.prefetch_batches,(unsigned long long)s.prefetched_experts,
        (unsigned long long)s.prefetch_bytes,1000*s.prefetch_wait_seconds,1000*s.prefetch_pack_seconds,
        1000*s.prefetch_staging_wait_seconds,1000*s.prefetch_submit_seconds);
    if (p_->uploads) {
        const auto & u = p_->uploads->stats();
        std::fprintf(stderr, "M31 GPU upload profile label=%s bytes=%llu pack_host_ms=%.3f staging_wait_host_ms=%.3f submit_host_ms=%.3f\n",
            label ? label : "live", (unsigned long long)u.upload_bytes,
            1000*u.pack_seconds, 1000*u.staging_wait_seconds, 1000*u.submit_seconds);
    }
}



const GpuExperts::Stats & GpuExperts::stats() const { return p_->stat; }


bool GpuExperts::native_gpu_math() const {
    if(!p_->ready)return false;
    return true;
}
size_t GpuExperts::workspace_bytes() const {return p_->stat.scratch_bytes;}



void GpuExperts::forward_device_no_prefetch(int layer,int n,ggml_tensor *normalized,
    const std::vector<int32_t> &ids,const std::vector<float> &weights,
    ggml_tensor *shared,ggml_tensor *residual,ggml_tensor *destination){
    struct Restore {Impl *p;bool old;~Restore(){p->suppress_prefetch=old;}}restore{p_.get(),p_->suppress_prefetch};
    p_->suppress_prefetch=true;
    forward_device(layer,n,normalized,ids,weights,shared,residual,destination);
}

#include "hybrid_forward.hpp"

void GpuExperts::release_prefill_workspace() {
    check(p_->ready && p_->active_read_keys.empty(), "cannot release active expert workspace");
    p_->finish_prefetch();
    if(p_->uploads)p_->uploads->synchronize();
    ggml_backend_synchronize(p_->backend);
    clear_captures(p_->backend);
    if(p_->allocator)ggml_gallocr_free(p_->allocator);
    p_->allocator=ggml_gallocr_new(p_->buffer_type);
    if(!p_->allocator){p_->ready=false;throw std::runtime_error("M31 GPU experts: allocator recreation failed");}
    p_->stat.scratch_bytes=0;
}
void GpuExperts::reset_stats() {
    // A caller may reset between arbitrary layer forwards. Complete speculative
    // uploads first so their counters belong wholly to the interval before reset.
    p_->finish_prefetch();
    const auto old = p_->stat;
    p_->stat = {};
    p_->stat.resident_bytes = old.resident_bytes;
    p_->stat.peak_resident_bytes = old.resident_bytes;
    p_->stat.budget_bytes = old.budget_bytes;
    p_->stat.minimum_budget_bytes = old.minimum_budget_bytes;
    p_->stat.scratch_bytes = old.scratch_bytes;
    p_->stat.pinned_staging = old.pinned_staging;
    p_->stat.q8_activations = old.q8_activations;
    p_->stat.slot_stride = old.slot_stride;
    p_->stat.slot_count = old.slot_count;
    p_->stat.pool_bytes = old.pool_bytes;
    p_->stat.occupied_slots = old.occupied_slots;
}
} // namespace strata::m31
