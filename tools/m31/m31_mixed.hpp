// Request-owned IPC for audited mtmd F32 rows; no llama/ggml ABI is shared.
#pragma once
#include "json.hpp"
#include "m31_vision.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::m31 {
// Files use little-endian IEEE F32/I32, as written by vision_batch.py.
// Fixed local filenames prevent a manifest from selecting unrelated files.
inline MixedInput load_mixed_manifest(const std::filesystem::path & manifest, int vocab, int64_t context) {
    namespace fs = std::filesystem;
    using J = nlohmann::json;
    auto fail = [](const std::string & reason) { throw std::runtime_error("mixed manifest: " + reason); };
    require_vision_host();
    if (vocab < 1 || context < 1 || context > 1048576) fail("invalid vocabulary/context bound");
    const auto path = fs::canonical(manifest);
    if (!fs::is_regular_file(path) || fs::file_size(path) > 16 * 1024 * 1024) fail("invalid manifest size");
    std::ifstream input(path);
    const auto m = J::parse(input);
    if (!m.is_object() || m.value("format", "") != "m31-vision-batch-v1" ||
        m.value("source_pr", "") != "6b95a9908d2c7b242817a447708aec38e937e2c0" ||
        !m.contains("encoded") || m.at("encoded") != true ||
        !m.contains("llm_inference") || m.at("llm_inference") != false)
        fail("unsupported format/source/encode receipt");
    auto integer = [&](const J & obj, const char * key, int64_t low, int64_t high) -> int64_t {
        if (!obj.contains(key) || !obj.at(key).is_number_integer()) fail(std::string("invalid integer ") + key);
        const auto value = obj.at(key).get<int64_t>();
        if (value < low || value > high) fail(std::string("out of range ") + key);
        return value;
    };
    if (integer(m, "n_embd", 6144, 6144) != 6144) fail("unsupported embedding width");
    const auto rows = integer(m, "rows", 1, context);
    MixedInput out;
    out.visual_count = int(integer(m, "visual_rows", 0, rows));
    auto read = [&](const char * name, auto & values, size_t count) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        const auto file = fs::canonical(path.parent_path() / name);
        if (count > std::numeric_limits<size_t>::max() / sizeof(T) ||
            count > uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(T)) fail("file size overflow");
        const auto bytes = count * sizeof(T);
        if (file.parent_path() != path.parent_path() || !fs::is_regular_file(file) || fs::file_size(file) != bytes)
            fail(std::string("invalid local file/size ") + name);
        values.resize(count);
        std::ifstream f(file, std::ios::binary);
        if (bytes) f.read(reinterpret_cast<char *>(values.data()), std::streamsize(bytes));
        if (!f) fail(std::string("short read ") + name);
    };
    read("text_ids.i32", out.tokens, size_t(rows));
    read("visual_rows.i32", out.visual_rows, size_t(rows));
    read("positions.i32", out.positions, size_t(rows));
    read("visual.f32", out.embeddings, size_t(out.visual_count) * 6144);
    int next_visual = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (out.positions[size_t(i)] != i) fail("decoder positions must be contiguous linear rows");
        const int token = out.tokens[size_t(i)], visual = out.visual_rows[size_t(i)];
        if (token == -1) {
            if (visual != next_visual || visual >= out.visual_count) fail("visual map must be complete and ordered");
            ++next_visual;
        } else if (token < 0 || token >= vocab || visual != -1) fail("invalid text ID/visual mapping");
    }
    if (next_visual != out.visual_count) fail("visual count mismatch");
    for (const float value : out.embeddings) if (!std::isfinite(value)) fail("non-finite embedding");
    if (!m.contains("chunks") || !m.at("chunks").is_array() || m.at("chunks").empty() ||
        m.at("chunks").size() > size_t(rows)) fail("invalid chunk list");
    int64_t first = 0, first_visual = 0, index = 0;
    for (const auto & chunk : m.at("chunks")) {
        if (first >= rows) fail("extra chunk");
        const auto count = integer(chunk, "rows", 1, rows - first);
        if (integer(chunk, "index", 0, rows) != index++ || integer(chunk, "row", 0, rows) != first ||
            integer(chunk, "position", 0, rows) != first || integer(chunk, "n_pos", 1, rows) != count)
            fail("chunk position discontinuity");
        const auto type = chunk.value("type", "");
        if (type == "text") {
            if (!chunk.contains("tokens") || !chunk.at("tokens").is_array() || chunk.at("tokens").size() != size_t(count))
                fail("invalid text chunk");
            for (int64_t j = 0; j < count; ++j) {
                const auto & token = chunk.at("tokens").at(size_t(j));
                if (!token.is_number_integer() || token.get<int64_t>() != out.tokens[size_t(first + j)] ||
                    out.tokens[size_t(first + j)] < 0) fail("text chunk disagrees with row IDs");
            }
        } else if (type == "image") {
            if (integer(chunk, "visual_row", 0, out.visual_count) != first_visual ||
                !chunk.contains("decoder_positions") || !chunk.at("decoder_positions").is_object() ||
                chunk.at("decoder_positions").size() != 3)
                fail("invalid linear visual span");
            const auto & decoder = chunk.at("decoder_positions");
            if (integer(decoder, "start", 0, rows) != first || integer(decoder, "step", 1, 1) != 1 ||
                integer(decoder, "axes", 1, 1) != 1) fail("invalid linear visual positions");
            for (int64_t j = 0; j < count; ++j)
                if (out.tokens[size_t(first + j)] != -1) fail("visual span disagrees with row IDs");
            first_visual += count;
        } else fail("unsupported chunk type");
        first += count;
    }
    if (first != rows || first_visual != out.visual_count) fail("chunks do not cover rows");
    return out;
}
} // namespace strata::m31
