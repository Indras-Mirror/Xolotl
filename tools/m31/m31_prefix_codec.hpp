// Host-only M31 prefix payload format. Checksums/catalogue identity are supplied
// by the shared store; this layer independently validates shapes and vision rows.
#pragma once
#include "m31_geometry.hpp"
#include "m31_vision.hpp"
#include <functional>
#include <optional>

namespace strata::m31::prefix_codec {
constexpr uint64_t magic = 0x31585031334d; // M31PX1
inline void u64(std::ostream & out, uint64_t n) { out.write(reinterpret_cast<const char *>(&n), 8); }
inline bool u64(std::istream & in, uint64_t & n) { return bool(in.read(reinterpret_cast<char *>(&n), 8)); }
struct Payload {
    std::vector<int32_t> tokens;
    MixedInput mixed;
    std::vector<uint8_t> kv;
};
inline bool write(std::ostream & out, const std::vector<int32_t> & tokens,
                  const MixedInput & mixed, const std::vector<uint8_t> & bytes) {
    require_vision_host();
    if (mixed.visual_rows.size() != tokens.size()) return false;
    u64(out, magic); u64(out, tokens.size()); u64(out, mixed.embeddings.size());
    u64(out, uint64_t(mixed.visual_count)); u64(out, bytes.size()); u64(out, 0);
    out.write(reinterpret_cast<const char *>(tokens.data()), std::streamsize(tokens.size() * 4));
    out.write(reinterpret_cast<const char *>(mixed.visual_rows.data()), std::streamsize(tokens.size() * 4));
    out.write(reinterpret_cast<const char *>(mixed.embeddings.data()), std::streamsize(mixed.embeddings.size() * 4));
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t n = std::min<size_t>(1 << 20, bytes.size() - offset);
        out.write(reinterpret_cast<const char *>(bytes.data() + offset), std::streamsize(n)); offset += n;
        if (!out) return false;
    }
    return bool(out);
}
inline std::optional<Payload> read(std::istream & in, uint64_t file_bytes, uint64_t maximum,
                                   const Geometry & g, const std::function<bool()> & cancelled = {}) {
    require_vision_host();
    if (file_bytes > maximum || file_bytes < 48 || (cancelled && cancelled())) return {};
    uint64_t tag, count, floats, visuals, kv, reserved;
    if (!u64(in, tag) || !u64(in, count) || !u64(in, floats) || !u64(in, visuals) || !u64(in, kv) || !u64(in, reserved) ||
        tag != magic || reserved || !count || count > uint64_t(g.context) ||
        count > (file_bytes - 48) / 8 || floats > (file_bytes - 48 - count * 8) / 4 || visuals > count ||
        floats != visuals * uint64_t(g.embd) || kv != g.kv_bytes(int64_t(count)) ||
        48 + count * 8 + floats * 4 + kv != file_bytes) return {};
    Payload image;
    image.tokens.resize(size_t(count)); image.mixed.visual_rows.resize(size_t(count));
    image.mixed.embeddings.resize(size_t(floats)); image.mixed.visual_count = int(visuals);
    if (!in.read(reinterpret_cast<char *>(image.tokens.data()), std::streamsize(count * 4)) ||
        !in.read(reinterpret_cast<char *>(image.mixed.visual_rows.data()), std::streamsize(count * 4)) ||
        !in.read(reinterpret_cast<char *>(image.mixed.embeddings.data()), std::streamsize(floats * 4))) return {};
    uint64_t seen = 0;
    for (size_t i = 0; i < count; ++i) {
        if (image.tokens[i] == -1) {
            if (seen >= visuals || image.mixed.visual_rows[i] != int32_t(seen++)) return {};
        } else if (image.tokens[i] < 0 || image.tokens[i] >= g.vocab || image.mixed.visual_rows[i] != -1) return {};
    }
    if (seen != visuals) return {};
    for (float value : image.mixed.embeddings) if (!std::isfinite(value)) return {};
    image.kv.resize(static_cast<size_t>(kv));
    for (size_t offset = 0; offset < image.kv.size();) {
        if (cancelled && cancelled()) return {};
        const size_t n = std::min<size_t>(1 << 20, image.kv.size() - offset);
        if (!in.read(reinterpret_cast<char *>(image.kv.data() + offset), std::streamsize(n))) return {};
        offset += n;
    }
    if (in.peek() != std::char_traits<char>::eof()) return {};
    return image;
}
} // namespace strata::m31::prefix_codec
