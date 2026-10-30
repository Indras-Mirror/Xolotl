// Bounded original-byte upload staging. The borrowed compute backend and every
// destination tensor must outlive the uploader (including its destructor).
#pragma once
#include "ggml-backend.h"
#include "../gpu/registered_arena.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <utility>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31 {

class PrefillUploads {
public:
    struct Part { size_t offset; const void * data; size_t bytes; };
    struct CopySpan { void * destination; const void * source; size_t bytes; };
    // An optional owner-supplied synchronous copier may distribute disjoint spans
    // over a bounded persistent CPU pool. It MUST join all writes before return;
    // the uploader submits DMA immediately afterward. The owner outlives the ring.
    using Copier = std::function<void(const CopySpan *, size_t)>;
    // Only [offset, offset+extent) is written. bytes original weight bytes
    // are followed by zero padding through extent (the backend alloc size).
    // Gaps between spans and trailing slot alignment are deliberately untouched.
    struct Span { size_t offset; const void * data; size_t bytes; size_t extent; };
    struct Stats {
        uint64_t batches = 0, uploads = 0, packed_bytes = 0, upload_bytes = 0;
        uint64_t staging_reuses = 0, registered_uploads = 0, registered_source_bytes = 0;
        size_t staging_bytes = 0, depth = 0, copy_streams = 0;
        bool pinned = false, asynchronous = false;
        // Host wall times only; DMA execution is asynchronous. Collected only
        // when profiling is enabled; never interpreted as device timings.
        double pack_seconds = 0, staging_wait_seconds = 0, submit_seconds = 0;
    };
    // slot_bytes includes quant and alignment padding; each upload zeros all
    // padding before copying Parts. No host/device allocation occurs in enqueue.
    PrefillUploads(ggml_backend_t compute, size_t slot_bytes, size_t depth = 2,
                   bool allow_async = true, size_t staging_limit = 256 * 1024 * 1024,
                   Copier copier = {}, size_t stream_count = 1, bool profile = false)
        : compute_(compute), bytes_(slot_bytes), copier_(std::move(copier)) {
        require(compute && slot_bytes && depth && depth <= 16 && stream_count >= 1 && stream_count <= 4 &&
                slot_bytes <= staging_limit / depth, "invalid bounded staging budget");
        profile_ = profile;
        stat_.depth = depth;
        stat_.staging_bytes = slot_bytes * depth;
        stages_.resize(depth);
        const auto device = ggml_backend_get_device(compute);
        require(device != nullptr, "compute backend has no device");
        const auto type = ggml_backend_dev_type(device);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const auto host = ggml_backend_dev_host_buffer_type(device);
        try {
            if (allow_async && type == GGML_BACKEND_DEVICE_TYPE_GPU &&
                props.caps.async && props.caps.events && host) {
                transfers_.resize(std::min(stream_count, depth), nullptr);
                frontiers_.resize(transfers_.size(), nullptr);
                for (size_t i = 0; i < transfers_.size(); ++i) {
                    transfers_[i] = ggml_backend_dev_init(device, nullptr);
                    require(transfers_[i] != nullptr, "transfer backend initialization failed");
                    frontiers_[i] = ggml_backend_event_new(device);
                    require(frontiers_[i] != nullptr, "transfer frontier event allocation failed");
                }
                stat_.copy_streams = transfers_.size();
                ready_ = ggml_backend_event_new(device);
                readers_ = ggml_backend_event_new(device);
                require(ready_ && readers_, "batch event allocation failed");
                size_t index = 0;
                for (auto & stage : stages_) {
                    stage.transfer = transfers_[index++ % transfers_.size()];
                    stage.buffer = ggml_backend_buft_alloc_buffer(host, bytes_);
                    require(stage.buffer != nullptr, "pinned staging allocation failed");
                    stage.event = ggml_backend_event_new(device);
                    require(stage.event != nullptr, "staging event allocation failed");
                }
                stat_.pinned = stat_.asynchronous = true;
            } else {
                for (auto & stage : stages_) stage.fallback.resize(bytes_);
            }
        } catch (...) {
            release();
            throw;
        }
    }
    ~PrefillUploads() { release(); }
    PrefillUploads(const PrefillUploads &) = delete;
    PrefillUploads & operator=(const PrefillUploads &) = delete;

    // Call only after all previous-batch consumers have been queued on compute.
    // This records their completion and makes subsequent H2D writes wait for it;
    // destination slots cannot be overwritten under a previous consumer.
    void begin_batch() {
        require(!active_, "begin_batch while a batch is open");
        if (stat_.asynchronous) {
            ggml_backend_event_record(readers_, compute_);
            for (auto transfer : transfers_) ggml_backend_event_wait(transfer, readers_);
        } else {
            ggml_backend_synchronize(compute_);
        }
        active_ = true;
    }

    // The source Parts may be reused as soon as this returns. The staging ring
    // owns copies until H2D events complete. A destination is a complete slot.
    void enqueue(ggml_tensor * destination, std::initializer_list<Part> parts) {
        require(active_, "enqueue outside a batch");
        require(destination && ggml_is_contiguous(destination) && ggml_nbytes(destination) == bytes_,
                "destination must be one contiguous complete slot");
        for (auto a = parts.begin(); a != parts.end(); ++a) {
            require(a->offset <= bytes_ && a->bytes <= bytes_ - a->offset && (a->data || !a->bytes),
                    "invalid source part");
            for (auto b = parts.begin(); b != a; ++b)
                require(!a->bytes || !b->bytes || a->offset >= b->offset + b->bytes ||
                        b->offset >= a->offset + a->bytes, "overlapping source parts");
        }
        auto & stage = stages_[cursor_];
        if (stage.busy) {
            const auto wait_start = timestamp();
            // CPU packing must not race a DMA still reading this pinned buffer.
            ggml_backend_event_synchronize(stage.event);
            ++stat_.staging_reuses;
            account(stat_.staging_wait_seconds, wait_start);
        }
        auto * data = stage.buffer ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage.buffer))
                                  : stage.fallback.data();
        const auto pack_start = timestamp();
        std::memset(data, 0, bytes_);
        if (copier_) {
            require(parts.size() <= 16, "parallel copier accepts at most16 parts");
            std::array<CopySpan,16> copies;
            size_t count=0;
            for (const auto & part:parts) if (part.bytes)
                copies[count++]={data+part.offset,part.data,part.bytes};
            copier_(copies.data(),count); // synchronous CPU barrier before DMA
        } else for (const auto & part : parts) if (part.bytes)
            std::memcpy(data + part.offset, part.data, part.bytes);
        for (const auto & part:parts) stat_.packed_bytes += part.bytes;
        account(stat_.pack_seconds, pack_start);
        const auto submit_start = timestamp();
        if (stat_.asynchronous) {
            ggml_backend_tensor_set_async(stage.transfer, destination, data, 0, bytes_);
            ggml_backend_event_record(stage.event, stage.transfer);
            stage.busy = true;
        } else {
            ggml_backend_tensor_set(destination, data, 0, bytes_);
        }
        account(stat_.submit_seconds, submit_start);
        ++stat_.uploads;
        stat_.upload_bytes += bytes_;
        cursor_ = (cursor_ + 1) % stages_.size();
    }

    // Compact role upload. Each role retains its backend-readable quant padding,
    // while fixed-slot alignment gaps never cross PCIe. Callers must describe ALL
    // bytes the consumer can read: extent must be the backend allocation size,
    // not merely the raw tensor size. Old bytes outside these extents are not
    // meaningful weights and must never be consumed. Source lifetime, events and
    // destination overwrite protection are identical to enqueue().
    void enqueue_spans(ggml_tensor * destination, std::initializer_list<Span> spans) {
        require(active_, "enqueue_spans outside a batch");
        require(destination && ggml_is_contiguous(destination) && ggml_nbytes(destination) == bytes_,
                "destination must be one contiguous complete slot");
        size_t packed = 0;
        for (auto a = spans.begin(); a != spans.end(); ++a) {
            require(a->offset <= bytes_ && a->extent <= bytes_ - a->offset &&
                    a->bytes <= a->extent && (a->data || !a->bytes), "invalid upload span");
            require(a->extent <= bytes_ - packed, "compact spans exceed staging budget");
            packed += a->extent;
            for (auto b = spans.begin(); b != a; ++b)
                require(!a->extent || !b->extent || a->offset >= b->offset + b->extent ||
                        b->offset >= a->offset + a->extent, "overlapping upload spans");
        }
        auto & stage = stages_[cursor_];
        if (stage.busy) {
            const auto wait_start = timestamp();
            ggml_backend_event_synchronize(stage.event);
            ++stat_.staging_reuses;
            account(stat_.staging_wait_seconds, wait_start);
        }
        auto * data = stage.buffer ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage.buffer))
                                  : stage.fallback.data();
        size_t offset = 0;
        if (copier_) {
            require(spans.size() <= 16, "parallel copier accepts at most16 spans");
            const auto pack_start=timestamp();
            std::array<CopySpan,16> copies;
            size_t count=0;
            for (const auto & span:spans) {
                if (span.bytes) copies[count++]={data+offset,span.data,span.bytes};
                if (span.extent>span.bytes)
                    std::memset(data+offset+span.bytes,0,span.extent-span.bytes);
                offset+=span.extent;
            }
            copier_(copies.data(),count); // all role/chunk writes join before any DMA
            account(stat_.pack_seconds,pack_start);
            offset=0;
        }
        for (const auto & span : spans) {
            if (!span.extent) continue;
            if (!copier_) {
                const auto pack_start = timestamp();
                if (span.bytes) std::memcpy(data + offset, span.data, span.bytes);
                if (span.extent > span.bytes)
                    std::memset(data + offset + span.bytes, 0, span.extent - span.bytes);
                account(stat_.pack_seconds, pack_start);
            }
            const auto submit_start = timestamp();
            if (stat_.asynchronous)
                ggml_backend_tensor_set_async(stage.transfer, destination, data + offset, span.offset, span.extent);
            else
                ggml_backend_tensor_set(destination, data + offset, span.offset, span.extent);
            account(stat_.submit_seconds, submit_start);
            offset += span.extent;
            stat_.packed_bytes += span.bytes;
        }
        // One event covers every role DMA from this stage. A later ring reuse
        // may overwrite any of its compact spans only after this event completes.
        if (stat_.asynchronous && packed) {
            const auto submit_start = timestamp();
            ggml_backend_event_record(stage.event, stage.transfer);
            stage.busy = true;
            account(stat_.submit_seconds, submit_start);
        }
        ++stat_.uploads;
        stat_.upload_bytes += packed;
        cursor_ = (cursor_ + 1) % stages_.size();
    }

    // Direct DMA from a registered immutable arena. Unlike enqueue_spans(),
    // the original source must remain alive/unchanged until DMA completes. The
    // registration owner validates every source range and outlives this ring.
    // Only backend padding uses the pinned ring, which is fenced as usual.
    void enqueue_registered_spans(ggml_tensor * destination,
            std::initializer_list<Span> spans,const GpuRegisteredArena &registration) {
        require(active_ && stat_.asynchronous, "registered upload requires an open asynchronous batch");
        require(destination && ggml_is_contiguous(destination) && ggml_nbytes(destination)==bytes_,
                "destination must be one contiguous complete slot");
        size_t total=0,maximum_padding=0,raw=0;
        for(auto a=spans.begin();a!=spans.end();++a) {
            require(a->offset<=bytes_ && a->extent<=bytes_-a->offset && a->bytes<=a->extent,
                    "invalid registered upload span");
            require(!a->bytes || registration.contains(a->data,a->bytes), "source lies outside registered arena");
            require(a->extent<=bytes_-total, "registered spans exceed slot budget");
            for(auto b=spans.begin();b!=a;++b)
                require(!a->extent || !b->extent || a->offset>=b->offset+b->extent ||
                        b->offset>=a->offset+a->extent,"overlapping registered upload spans");
            total+=a->extent;raw+=a->bytes;
            maximum_padding=std::max(maximum_padding,a->extent-a->bytes);
        }
        auto &stage=stages_[cursor_];
        if(stage.busy) {
            const auto begin=timestamp();ggml_backend_event_synchronize(stage.event);
            ++stat_.staging_reuses;account(stat_.staging_wait_seconds,begin);
        }
        auto *zero=static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage.buffer));
        const auto pack_begin=timestamp();
        if(maximum_padding)std::memset(zero,0,maximum_padding);
        account(stat_.pack_seconds,pack_begin);
        const auto submit_begin=timestamp();
        for(const auto &span:spans) {
            size_t copied=0;
            while(copied<span.bytes) {
                const auto *source=static_cast<const uint8_t *>(span.data)+copied;
                const size_t part=registration.contiguous_bytes(source,span.bytes-copied);
                require(part>0,"registered span has no ready contiguous registration piece");
                ggml_backend_tensor_set_async(stage.transfer,destination,source,span.offset+copied,part);
                copied+=part;
            }
            if(span.extent>span.bytes)ggml_backend_tensor_set_async(stage.transfer,destination,zero,
                    span.offset+span.bytes,span.extent-span.bytes);
        }
        if(total){ggml_backend_event_record(stage.event,stage.transfer);stage.busy=true;}
        account(stat_.submit_seconds,submit_begin);
        ++stat_.uploads;++stat_.registered_uploads;
        stat_.upload_bytes+=total;stat_.registered_source_bytes+=raw;
        cursor_=(cursor_+1)%stages_.size();
    }

    // Publish all uploads queued so far while keeping this batch open. Consumers
    // may immediately run on compute while later enqueue calls upload DISJOINT
    // destination slots. The returned event is borrowed until uploader destruction;
    // an external CUDA-stream bridge must enqueue its wait before the next
    // checkpoint/finish, because the same event object is recorded again.
    // CPU/synchronous fallback returns nullptr: its writes have already completed.
    // This does not protect a same-batch slot from overwrite; the caller must not
    // reuse any published slot until that consumer has completed. begin_batch only
    // protects readers submitted on compute before begin_batch was called.
    ggml_backend_event_t checkpoint() {
        require(active_, "checkpoint outside a batch");
        if (stat_.asynchronous) {
            // Join every copy stream into stream0. Waiting on ready_ covers all
            // published role DMAs, including streams that stage reuse has not
            // yet synchronized. Events are recorded before their waits.
            for (size_t i = 1; i < transfers_.size(); ++i) {
                ggml_backend_event_record(frontiers_[i], transfers_[i]);
                ggml_backend_event_wait(transfers_[0], frontiers_[i]);
            }
            ggml_backend_event_record(ready_, transfers_[0]);
            ggml_backend_event_wait(compute_, ready_);
            return ready_;
        }
        return nullptr;
    }

    // Orders future compute after all uploads without a host synchronization.
    // Checkpoints permit upload/compute overlap only when the caller supplies
    // disjoint slots and a compute API that does not synchronize each subset.
    void finish() {
        require(active_, "finish outside a batch");
        checkpoint();
        active_ = false;
        ++stat_.batches;
    }
    // Borrowed completion fence of the last finished batch. A private CUDA
    // consumer enqueues its wait before this uploader starts/re-records another
    // batch. The owner still protects destination reuse until consumption ends.
    ggml_backend_event_t completion_event() const {
        require(!active_, "completion_event before finish");
        return stat_.asynchronous && stat_.batches ? ready_ : nullptr;
    }
    void synchronize() {
        require(!active_, "synchronize before finish");
        for (auto transfer : transfers_) if (transfer) ggml_backend_synchronize(transfer);
        ggml_backend_synchronize(compute_);
    }
    const Stats & stats() const { return stat_; }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point timestamp() const { return profile_ ? Clock::now() : Clock::time_point{}; }
    void account(double & total, Clock::time_point begin) const {
        if (profile_) total += std::chrono::duration<double>(Clock::now() - begin).count();
    }
    struct Stage {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        ggml_backend_t transfer = nullptr;
        std::vector<uint8_t> fallback;
        bool busy = false;
    };

    static void require(bool condition, const char * message) {
        if (!condition) throw std::runtime_error(std::string("M31 prefill uploads: ") + message);
    }
    void release() noexcept {
        for (auto transfer : transfers_) if (transfer) ggml_backend_synchronize(transfer);
        if (compute_) ggml_backend_synchronize(compute_);
        for (auto & stage : stages_) {
            if (stage.event) ggml_backend_event_free(stage.event);
            if (stage.buffer) ggml_backend_buffer_free(stage.buffer);
            stage.event = nullptr;
            stage.buffer = nullptr;
        }
        if (ready_) ggml_backend_event_free(ready_);
        if (readers_) ggml_backend_event_free(readers_);
        for (auto frontier : frontiers_) if (frontier) ggml_backend_event_free(frontier);
        for (auto transfer : transfers_) if (transfer) ggml_backend_free(transfer);
        frontiers_.clear();
        transfers_.clear();
        ready_ = readers_ = nullptr;
    }
    ggml_backend_t compute_ = nullptr;
    std::vector<ggml_backend_t> transfers_;
    std::vector<ggml_backend_event_t> frontiers_;
    ggml_backend_event_t ready_ = nullptr, readers_ = nullptr;
    size_t bytes_ = 0, cursor_ = 0;
    Copier copier_;
    std::vector<Stage> stages_;
    Stats stat_;
    bool active_ = false, profile_ = false;
};
} // namespace strata::m31
