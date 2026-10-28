// Exact-byte next-layer prefetch on a private backend and pinned ring.
#pragma once
#include "pipeline.hpp"
#include <array>
#include <future>
#include <memory>
#include <utility>

namespace strata::m31 {
class BackgroundPrefillUploads {
public:
    struct Job {
        ggml_tensor * destination = nullptr;
        std::array<PrefillUploads::Span, 3> spans{};
    };
    // The source backend is used only to discover its device. The private backend
    // and ring are initialized once by the first worker and reused thereafter.
    // This object must be drained/destroyed BEFORE any borrowed destination pool,
    // source arena, or tensor metadata dies. Serialized control by one owner only.
    BackgroundPrefillUploads(ggml_backend_t source, size_t slot_bytes, PrefillUploads::Copier copier = {}, size_t depth = 8, size_t streams = 1, bool profile = false)
        : device_(source ? ggml_backend_get_device(source) : nullptr), bytes_(slot_bytes), copier_(std::move(copier)), source_(source), depth_(depth), streams_(streams), profile_(profile) {
        require(device_ && bytes_, "missing source backend/device or slot bytes");
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device_, &props);
        if(props.caps.events) {
            readers_=ggml_backend_event_new(device_);
            require(readers_, "reader event allocation failed");
        }

    }
    ~BackgroundPrefillUploads() {
        drain();
        uploads_.reset();
        if (backend_) ggml_backend_free(backend_);
        if (readers_) ggml_backend_event_free(readers_);
    }
    BackgroundPrefillUploads(const BackgroundPrefillUploads &) = delete;
    BackgroundPrefillUploads & operator=(const BackgroundPrefillUploads &) = delete;

    // Jobs own descriptors; sources and destination tensors are borrowed and must
    // stay immutable/alive until wait(). Owner reserves DISJOINT slots before
    // start: no current compute or another uploader may read/write these slots.
    // Owner publishes its cache entries only after a successful wait().
    // Optional registration and its immutable arena sources outlive wait().
    void start(std::vector<Job> jobs, bool compact = true, const GpuRegisteredArena *registration = nullptr) {
        require(!active(), "start while previous work is uncollected");
        require(!jobs.empty() && jobs.size() <= 8192, "invalid bounded prefetch job count");
        // These slots were leased away from current/future readers by the owner.
        // The event belongs to the actual compute backend, never the private
        // uploader backend. It fences readers from completed earlier batches.
        if(readers_)ggml_backend_event_record(readers_,source_);
        else ggml_backend_synchronize(source_);
        future_ = std::async(std::launch::async, [this, jobs=std::move(jobs), compact, registration]() {
            if (!backend_) {
                backend_ = ggml_backend_dev_init(device_, nullptr);
                require(backend_ != nullptr, "private backend initialization failed");
            }
            if (!uploads_) uploads_ = std::make_unique<PrefillUploads>(backend_, bytes_, depth_, true, 256*1024*1024, copier_, streams_, profile_);
            const auto before = uploads_->stats();
            try {
                if(readers_)ggml_backend_event_wait(backend_,readers_);
                uploads_->begin_batch();
                for (const auto & job : jobs) {
                    const auto & a=job.spans[0], &b=job.spans[1], &c=job.spans[2];
                    if (registration) uploads_->enqueue_registered_spans(job.destination,{a,b,c},*registration);
                    else if (compact) uploads_->enqueue_spans(job.destination,{a,b,c});
                    else uploads_->enqueue(job.destination,{
                        {a.offset,a.data,a.bytes},{b.offset,b.data,b.bytes},{c.offset,c.data,c.bytes}});
                }
                uploads_->finish();
                uploads_->synchronize();
                auto result = uploads_->stats();
                result.batches -= before.batches;
                result.uploads -= before.uploads;
                result.packed_bytes -= before.packed_bytes;
                result.upload_bytes -= before.upload_bytes;
                result.staging_reuses -= before.staging_reuses;
                result.registered_uploads -= before.registered_uploads;
                result.registered_source_bytes -= before.registered_source_bytes;
                result.pack_seconds -= before.pack_seconds;
                result.staging_wait_seconds -= before.staging_wait_seconds;
                result.submit_seconds -= before.submit_seconds;
                return result;
            } catch (...) {
                // A partial DMA may have been queued. Drain it before propagating
                // failure, and recreate ring/event state on a later explicit retry.
                uploads_.reset();
                throw;
            }
        });
    }
    bool active() const noexcept { return future_.valid(); }
    PrefillUploads::Stats wait() {
        require(active(), "wait without started work");
        return future_.get(); // also joins on failure; the future is consumed
    }
    // Destructors/failure cleanup must not throw. Explicit wait propagates errors
    // so the cache owner can discard every reserved or partially uploaded entry.
    void drain() noexcept {
        if (active()) {
            try { (void)future_.get(); } catch (...) {}
        }
    }
private:
    static void require(bool ok, const char * message) {
        if (!ok) throw std::runtime_error(std::string("M31 background prefill: ")+message);
    }
    ggml_backend_dev_t device_ = nullptr;
    size_t bytes_ = 0;
    PrefillUploads::Copier copier_;
    ggml_backend_t backend_ = nullptr, source_ = nullptr;
    ggml_backend_event_t readers_ = nullptr;
    size_t depth_ = 8, streams_ = 1;
    bool profile_ = false;
    std::unique_ptr<PrefillUploads> uploads_;
    std::future<PrefillUploads::Stats> future_;
};
} // namespace strata::m31
