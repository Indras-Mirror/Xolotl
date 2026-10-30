// Release of a CUDA device's primary context during an engine deep sleep.
#pragma once
#include <cstddef>
#include <string>

namespace strata::m31 {
// Every allocation, stream, registration and backend of this process on the
// device must already be freed. Resets the primary context, so the process
// holds no device memory until the next CUDA call creates a new context.
// Returns an empty string on success, otherwise the CUDA error.
std::string release_cuda_device(int device);

// Page-locks an existing, already resident host range for the current CUDA
// context while it lives, so large copies run at DMA speed. A failed
// registration leaves the range pageable; copies still work, only slower.
class PinnedHostRange {
public:
    PinnedHostRange(void * data, size_t bytes) noexcept;
    ~PinnedHostRange();
    PinnedHostRange(const PinnedHostRange &) = delete;
    PinnedHostRange & operator=(const PinnedHostRange &) = delete;
    bool pinned() const noexcept { return pinned_; }
private:
    void * data_ = nullptr;
    bool pinned_ = false;
};
} // namespace strata::m31
