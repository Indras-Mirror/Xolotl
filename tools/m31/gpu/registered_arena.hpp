// Optional registration of the immutable, already resident expert RAM arena.
// The owner must drain all readers/uploaders before destroying registration,
// and destroy registration before freeing the borrowed source arena.
#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

namespace strata::m31 {
class GpuRegisteredArena {
public:
    struct Piece {
        size_t index=0;
        void * registered_ptr=nullptr;
        size_t registered_bytes=0;
        const void * source_ptr=nullptr;
        size_t source_bytes=0, source_offset=0;
    };
    struct Options {
        uint64_t limit_bytes=0; // required, explicit maximum pinned resident bytes
        size_t chunk_bytes=size_t(1)<<30;
        unsigned threads=4;
        bool asynchronous=false;
        // Runs on a registration worker. The source must be ready before this
        // callback reads it. DMA users drain their streams before destruction.
        std::function<void(const Piece &)> on_registered;
    };
    struct Stats {
        size_t bytes=0, registered_bytes=0, pieces=0;
        unsigned threads=0;
        double register_seconds=0, register_sum_seconds=0, finish_wait_seconds=0;
        bool hardware_read_only=false;
    };
    GpuRegisteredArena(ggml_backend_t backend,const void * data,size_t bytes);
    // Uses the device selected by the supplied CUDA backend. No process-global
    // device index or machine-local reservation protocol is assumed.
    GpuRegisteredArena(ggml_backend_t backend,const void * data,size_t bytes,Options options);
    ~GpuRegisteredArena();
    GpuRegisteredArena(const GpuRegisteredArena &)=delete;
    GpuRegisteredArena &operator=(const GpuRegisteredArena &)=delete;
    // Inline range validation keeps the generic uploader CPU-only compatible.
    bool contains(const void * data,size_t bytes)const noexcept {
        if (!ready_) return false;
        const auto begin=reinterpret_cast<uintptr_t>(base_);
        const auto source=reinterpret_cast<uintptr_t>(data);
        return source>=begin && source-begin<=bytes_ && bytes<=bytes_-(source-begin);
    }
    void finish(); // joins workers, publishes all pieces or throws after rollback
    bool ready() const noexcept {return ready_;}
    // Maximum logical source prefix contained in one physical registration.
    // A DMA spanning adjacent registered pieces must split at this boundary.
    size_t contiguous_bytes(const void * data,size_t requested) const noexcept {
        if (!contains(data,requested)||!requested||!chunk_bytes_) return 0;
        const size_t offset=size_t(reinterpret_cast<uintptr_t>(data)-aligned_base_);
        const size_t remaining=chunk_bytes_-offset%chunk_bytes_;
        return requested<remaining?requested:remaining;
    }
    const Stats &stats()const noexcept {return stat_;}
private:
    const void *base_=nullptr;
    size_t bytes_=0;
    uintptr_t aligned_base_=0;
    size_t chunk_bytes_=0;
    Stats stat_;
    bool ready_=false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
