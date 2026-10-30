// Exact RAM layout of the original expert bytes and external scales.
#pragma once
#include "m31_geometry.hpp"
#include "m31_config.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gpu/registered_arena.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <memory>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::m31 {

class ExpertArena {
public:
    struct Slice { uint8_t * data = nullptr; size_t expert_stride = 0; };
    struct Layer {
        size_t offset = 0, stride = 0, tail_offset = 0;
        std::array<size_t, 3> role_offsets{}, role_bytes{};
        bool routed = false;
    };
    ~ExpertArena() {
        // Registration workers and DMA users must finish before source pages
        // disappear. GpuExperts is destroyed before its borrowed ExpertArena.
        registration_.reset();
        if (buffer_) ggml_backend_buffer_free(buffer_);
    }
    ExpertArena() = default;
    ExpertArena(const ExpertArena &) = delete;
    ExpertArena & operator=(const ExpertArena &) = delete;

    // Exact transport storage requirement, including per-expert alignment and
    // the external NVFP4 scale tail. Metadata only: no payload scan/allocation.
    static size_t required_bytes(const GgufModel & model, const Geometry & g) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        size_t total = 0;
        for (int l = 0; l < g.layers; ++l) {
            if (!g.routed(l)) continue;
            size_t payload = 0; bool tail = false;
            for (const char * role : {"gate", "up", "down"}) {
                const auto name = layer_name(l, std::string("ffn_") + role + "_exps.weight");
                const auto * tensor = model.find(name);
                if (!tensor || tensor->shape.size() != 3 || tensor->shape[2] != uint64_t(g.experts))
                    throw std::runtime_error("M31 arena: invalid weight " + name);
                const size_t row = ggml_row_size(ggml_type(tensor->type), int64_t(tensor->shape[0]));
                if (tensor->shape[1] > (SIZE_MAX - payload) / row)
                    throw std::runtime_error("M31 arena: required byte count overflow");
                payload += row * size_t(tensor->shape[1]);
                tail = tail || tensor->type == GGML_TYPE_NVFP4;
            }
            const size_t overhead = (tail ? 16 : 0) + alignment - 1;
            if (payload > SIZE_MAX - overhead)
                throw std::runtime_error("M31 arena: aligned byte count overflow");
            const size_t stride = (payload + overhead) / alignment * alignment;
            if (uint64_t(g.experts) > (SIZE_MAX - total) / stride)
                throw std::runtime_error("M31 arena: total byte count overflow");
            total += stride * size_t(g.experts);
        }
        return total;
    }

    void init(const GgufModel & model, const Geometry & g, double budget_gib, const Config & cfg) {
        if (buffer_ || !std::isfinite(budget_gib) || budget_gib <= 0)
            throw std::runtime_error("M31 arena: invalid/repeated initialization");
        const auto t0 = std::chrono::steady_clock::now();
        struct Source {
            std::string name;
            const uint8_t * data = nullptr, * scales = nullptr;
            bool nvfp4 = false;
        };
        std::vector<std::array<Source, 3>> sources(size_t(g.layers));
        layers_.resize(size_t(g.layers));
        const char * roles[] = {"gate", "up", "down"};
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        auto align = [alignment](size_t n) { return (n + alignment - 1) / alignment * alignment; };
        for (int l = 0; l < g.layers; ++l) {
            if (!g.routed(l)) continue;
            auto & layer = layers_[size_t(l)];
            layer.routed = true;
            layer.offset = bytes_;
            size_t role_end = 0;
            bool tail = false;
            for (int r = 0; r < 3; ++r) {
                auto & src = sources[size_t(l)][size_t(r)];
                src.name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                size_t shard = 0;
                const auto * ti = model.find(src.name, &shard);
                if (!ti || ti->shape.size() != 3 || ti->shape[2] != uint64_t(g.experts) || !model.in_bounds(*ti, shard))
                    throw std::runtime_error("M31 arena: invalid weight " + src.name);
                src.data = model.shard(shard).tensor_data(*ti);
                src.nvfp4 = ti->type == GGML_TYPE_NVFP4;
                layer.role_offsets[size_t(r)] = role_end;
                layer.role_bytes[size_t(r)] = ggml_row_size(ggml_type(ti->type), int64_t(ti->shape[0])) * ti->shape[1];
                role_end += layer.role_bytes[size_t(r)];
                if (src.nvfp4) {
                    const auto scale_name = src.name.substr(0, src.name.size() - 6) + "scale";
                    const auto * scale = model.find(scale_name, &shard);
                    if (!scale || scale->type != GGML_TYPE_F32 || scale->shape != std::vector<uint64_t>{uint64_t(g.experts)} ||
                        !model.in_bounds(*scale, shard))
                        throw std::runtime_error("M31 arena: missing/invalid scale " + scale_name);
                    src.scales = model.shard(shard).tensor_data(*scale);
                    tail = true;
                }
            }
            // The tail belongs to the packed blob, never to the source down-tensor stride.
            layer.tail_offset = tail ? role_end : 0;
            layer.stride = align(role_end + (tail ? 16 : 0));
            bytes_ += layer.stride * size_t(g.experts);
        }
        if (bytes_ != required_bytes(model, g))
            throw std::runtime_error("M31 arena: inconsistent storage plan");
        if (double(bytes_) > budget_gib * 1073741824.0)
            throw std::runtime_error("M31 arena: budget is smaller than the complete expert arena");
#if defined(__linux__)
        // Avoid relying on Linux overcommit when a complete arena is requested.
        // Page cache is reclaimable and is included in MemAvailable.
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        uint64_t kb = 0;
        while (meminfo >> key >> kb >> unit) {
            if (key == "MemAvailable:") {
                if (kb < (bytes_ + 1023) / 1024)
                    throw std::runtime_error("M31 arena: insufficient available RAM for the complete arena");
                break;
            }
        }
#endif
        if (cfg.register_arena) {
#ifdef M31_REGISTERED_ARENA_CUDA
            // Align an interior allocation, with its complete lifetime owned here.
            // The same C allocation works on Windows and POSIX systems.
            constexpr size_t page = 4096;
            if (bytes_ > SIZE_MAX - 2 * page)
                throw std::runtime_error("M31 arena: allocation alignment overflow");
            void * memory = std::malloc(bytes_ + 2 * page);
            if (!memory) throw std::runtime_error("M31 arena: host allocation failed");
            own_memory_.reset(memory);
            auto address = (reinterpret_cast<uintptr_t>(memory) + page - 1) & ~(uintptr_t(page) - 1);
            buffer_ = ggml_backend_cpu_buffer_from_ptr(reinterpret_cast<void *>(address), bytes_);
#else
            throw std::runtime_error("M31 arena: CUDA registration is unavailable in this build");
#endif
        } else buffer_ = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes_);
        if (!buffer_) throw std::runtime_error("M31 arena: CPU allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer_));
#ifdef M31_REGISTERED_ARENA_CUDA
        if (cfg.register_arena) {
            GpuRegisteredArena::Options options;
            options.asynchronous = true;
            options.deferred = true;
            options.limit_bytes = cfg.register_limit_bytes;
            options.chunk_bytes = cfg.register_chunk_bytes;
            options.threads = cfg.register_threads;
            registration_ = std::make_shared<GpuRegisteredArena>(cfg.backend, base, bytes_, std::move(options));
        }
#endif
        for (int l = 0; l < g.layers; ++l) {
            const auto & layer = layers_[size_t(l)];
            if (!layer.routed) continue;
            for (int r = 0; r < 3; ++r)
                slices_.emplace(sources[size_t(l)][size_t(r)].name,
                    Slice{base + layer.offset + layer.role_offsets[size_t(r)], layer.stride});
        }
        // One unit is one packed expert, in arena order. Units are disjoint and
        // their copies are independent, so the bytes equal the serial fill's.
        std::vector<std::pair<int, int>> units;
        units.reserve(size_t(g.experts) * size_t(g.layers));
        for (int l = 0; l < g.layers; ++l)
            if (layers_[size_t(l)].routed)
                for (int e = 0; e < g.experts; ++e) units.emplace_back(l, e);
        auto fill = [&](int l, int e) {
            const auto & layer = layers_[size_t(l)];
            auto * dst = base + layer.offset + size_t(e) * layer.stride;
            size_t raw_end = 0;
            float tail[4] = {1, 1, 1, 0};
            for (int r = 0; r < 3; ++r) {
                const auto & src = sources[size_t(l)][size_t(r)];
                const size_t size = layer.role_bytes[size_t(r)];
                std::memcpy(dst + layer.role_offsets[size_t(r)], src.data + size_t(e) * size, size);
                raw_end += size;
                if (src.scales) {
                    std::memcpy(&tail[r], src.scales + size_t(e) * sizeof(float), sizeof(float));
                    if (!std::isfinite(tail[r]))
                        throw std::runtime_error("M31 arena: invalid NVFP4 scale");
                }
            }
            std::memset(dst + raw_end, 0, layer.stride - raw_end);
            if (layer.tail_offset) std::memcpy(dst + layer.tail_offset, tail, sizeof tail);
#ifdef M31_REGISTERED_ARENA_CUDA
            if (registration_) registration_->filled(layer.offset + size_t(e) * layer.stride, layer.stride);
#endif
        };
        fill_threads_ = unsigned(std::max<size_t>(1, std::min<size_t>(cfg.arena_fill_threads, units.size())));
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto worker = [&] {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= units.size()) break;
                    fill(units[index].first, units[index].second);
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };
        {
            std::vector<std::thread> workers;
            try {
                for (unsigned t = 1; t < fill_threads_; ++t) workers.emplace_back(worker);
            } catch (...) {
                failed.store(true);
                for (auto & t : workers) t.join();
                throw;
            }
            worker();
            for (auto & t : workers) t.join();
        }
        fill_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (error) {
#ifdef M31_REGISTERED_ARENA_CUDA
            if (registration_) registration_->abort();
#endif
            registration_.reset();
            std::rethrow_exception(error);
        }
#ifdef M31_REGISTERED_ARENA_CUDA
        if(registration_)registration_->finish();
#endif
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "M31 RAM arena: bytes=%zu experts=%lld seconds=%.3f fill_seconds=%.3f fill_threads=%u (exact packed weights, %s)\n",
                     bytes_, (long long) (g.experts * (g.layers - g.dense_lead)), seconds, fill_seconds_, fill_threads_,
                     registration_?"each piece registered behind its fill":"pageable");
    }
    // Deep sleep: the CUDA context that pinned the arena goes away; the filled
    // RAM stays. Waking registers the resident pages for a new context.
    void release_registration() { registration_.reset(); }
    // One background thread registers piece by piece while may_register()
    // allows it; the aggregate pinning rate is the same as with several.
    void register_resident(ggml_backend_t backend, const Config & cfg, std::function<bool()> may_register) {
#ifdef M31_REGISTERED_ARENA_CUDA
        if (!cfg.register_arena || registration_) return;
        if (!buffer_) throw std::runtime_error("M31 arena: registration without a filled arena");
        GpuRegisteredArena::Options options;
        options.asynchronous = true;
        options.limit_bytes = cfg.register_limit_bytes;
        options.chunk_bytes = cfg.register_chunk_bytes;
        options.threads = 1;
        options.may_register = std::move(may_register);
        registration_ = std::make_shared<GpuRegisteredArena>(backend, ggml_backend_buffer_get_base(buffer_), bytes_, std::move(options));
#else
        (void) backend; (void) may_register;
        if (cfg.register_arena) throw std::runtime_error("M31 arena: CUDA registration is unavailable in this build");
#endif
    }
    bool registration_settled() const {
#ifdef M31_REGISTERED_ARENA_CUDA
        return !registration_ || registration_->settled();
#else
        return true;
#endif
    }
    void finish_registration() {
#ifdef M31_REGISTERED_ARENA_CUDA
        if (registration_) registration_->finish();
#endif
    }
    const Slice * find(const std::string & name) const {
        auto it = slices_.find(name);
        return it == slices_.end() ? nullptr : &it->second;
    }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    size_t bytes() const { return bytes_; }
    const std::vector<Layer> & layers() const { return layers_; }
    const GpuRegisteredArena * registration() const {return registration_.get();}
private:
    ggml_backend_buffer_t buffer_ = nullptr;
    size_t bytes_ = 0;
    std::vector<Layer> layers_;
    std::map<std::string, Slice> slices_;
    // Unconditional fields keep this header's class layout identical in CPU and
    // CUDA translation units. CPU-only builds never instantiate registration.
    std::shared_ptr<GpuRegisteredArena> registration_;
    double fill_seconds_ = 0;
    unsigned fill_threads_ = 0;
    std::unique_ptr<void,decltype(&std::free)> own_memory_{nullptr,&std::free};

};
} // namespace strata::m31
