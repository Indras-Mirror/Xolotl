// Request-owned vision rows. No llama/ggml ABI crosses the helper boundary.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::m31 {
struct MixedInput {
    std::vector<int32_t> tokens, visual_rows, positions;
    std::vector<float> embeddings;
    int visual_count = 0;
};
inline void require_vision_host() {
    const uint32_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
        throw std::runtime_error("vision unavailable: little-endian IEEE F32 host required");
}
// Common Strata GENI wire file: concatenated SVE1 records, each with five
// little-endian int32 fields (magic,n,nx,ny,width), followed by n*width F32.
// MiniMax uses ordinary linear positions; nx/ny do not alter its RoPE.
inline MixedInput load_vision_embeddings(const std::filesystem::path & path,
                                        const std::vector<int32_t> & ids, int width) {
    namespace fs = std::filesystem;
    auto fail = [](const char * why) { throw std::runtime_error(std::string("vision unavailable: ") + why); };
    require_vision_host();
    if (width <= 0 || ids.empty() || ids.size() > size_t(std::numeric_limits<int>::max())) fail("invalid embedding geometry");
    const auto visual_count = size_t(std::count(ids.begin(), ids.end(), int32_t(-1)));
    if (!visual_count) fail("GENI prompt has no visual rows");
    // Reject special files before opening: a FIFO/device must never block GENI.
    if (!fs::is_regular_file(path)) fail("embeddings file is missing or not regular");
    const auto bytes = fs::file_size(path);
    const uint64_t row_bytes = uint64_t(width) * sizeof(float);
    if (visual_count > std::numeric_limits<size_t>::max() / row_bytes ||
        visual_count > uint64_t(std::numeric_limits<std::streamsize>::max()) / row_bytes)
        fail("embeddings size overflow");
    const uint64_t payload = uint64_t(visual_count) * row_bytes;
    if (bytes < payload + 20 || bytes > payload + 20 * visual_count) fail("embeddings file size does not match prompt");
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open embeddings file");
    MixedInput out;
    out.tokens = ids;
    out.visual_rows.assign(ids.size(), -1);
    out.embeddings.resize(size_t(payload / sizeof(float)));
    out.visual_count = int(visual_count);
    size_t cursor = 0, visual = 0;
    uint64_t remaining = bytes;
    while (remaining) {
        if (remaining < 20) fail("short embeddings header");
        int32_t header[5];
        if (!file.read(reinterpret_cast<char *>(header), sizeof header)) fail("short embeddings header");
        remaining -= sizeof header;
        if (header[0] != 0x31455653 || header[1] < 1 || header[2] < 1 || header[3] < 1 ||
            header[4] != width || int64_t(header[2]) * header[3] != header[1] ||
            uint64_t(header[1]) > visual_count - visual) fail("invalid SVE1 header/count/width");
        const size_t count = size_t(header[1]);
        const uint64_t count_bytes = uint64_t(count) * row_bytes;
        if (count_bytes > remaining) fail("short embeddings payload");
        while (cursor < ids.size() && ids[cursor] != -1) ++cursor;
        if (count > ids.size() - cursor) fail("visual span exceeds prompt");
        for (size_t i = 0; i < count; ++i) {
            if (ids[cursor + i] != -1) fail("visual record crosses a text span");
            out.visual_rows[cursor + i] = int32_t(visual + i);
        }
        auto * first = out.embeddings.data() + visual * size_t(width);
        if (!file.read(reinterpret_cast<char *>(first), std::streamsize(count_bytes))) fail("short embeddings payload");
        for (size_t i = 0; i < count * size_t(width); ++i)
            if (!std::isfinite(first[i])) fail("non-finite visual embedding");
        cursor += count; visual += count; remaining -= count_bytes;
    }
    if (visual != visual_count) fail("missing visual records");
    if (file.peek() != std::char_traits<char>::eof()) fail("embeddings file changed during read");
    return out;
}
// Token IDs alone cannot identify a picture: every picture has -1 rows. Reuse
// stops at the first differing F32 row, even if a file name/source key is equal.
inline int reusable_visual_prefix(int candidate, const std::vector<int32_t> & ids,
                                  const MixedInput & previous, const MixedInput & current, int width) {
    for (int i = 0; i < candidate; ++i) if (ids[size_t(i)] == -1) {
        if (size_t(i) >= previous.visual_rows.size() || size_t(i) >= current.visual_rows.size()) return i / 256 * 256;
        const int a = previous.visual_rows[size_t(i)], b = current.visual_rows[size_t(i)];
        if (a < 0 || b < 0 || a >= previous.visual_count || b >= current.visual_count ||
            std::memcmp(previous.embeddings.data() + size_t(a) * width,
                        current.embeddings.data() + size_t(b) * width, size_t(width) * sizeof(float)))
            return i / 256 * 256;
    }
    return candidate;
}
} // namespace strata::m31
