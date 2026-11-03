// Host-only wire-format checks. This is not model execution or GPU validation.
#include "../m31_prefix_codec.hpp"
#include <cstring>
#include <iostream>
#include <sstream>

using namespace strata::m31;
namespace {
void check(bool value, const char * why) { if (!value) throw std::runtime_error(why); }
void field(std::string & bytes, size_t offset, uint64_t value) { std::memcpy(bytes.data() + offset, &value, 8); }
}
int main() {
    try {
        Geometry g; g.layers = 2; g.kv_heads = 1; g.head_dim = 2; g.index_dim = 2;
        g.embd = 3; g.vocab = 8; g.context = 1024;
        const std::vector<int32_t> ids{1, -1, -1, 3};
        MixedInput mixed; mixed.visual_count = 2; mixed.visual_rows = {-1, 0, 1, -1};
        mixed.embeddings = {0.25f, -0.0f, 1.5f, -2.0f, 4.0f, 0.0f};
        std::vector<uint8_t> kv(size_t(g.kv_bytes(ids.size())));
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = uint8_t(i * 17);
        std::ostringstream out(std::ios::binary);
        check(prefix_codec::write(out, ids, mixed, kv), "valid write failed");
        const std::string valid = out.str();
        auto read = [&](const std::string & value, uint64_t budget = 1 << 20,
                        const std::function<bool()> & cancelled = {}) {
            std::istringstream in(value, std::ios::binary);
            return prefix_codec::read(in, value.size(), budget, g, cancelled);
        };
        auto image = read(valid);
        check(bool(image), "roundtrip refused");
        check(image->tokens == ids && image->mixed.visual_rows == mixed.visual_rows &&
              image->mixed.visual_count == 2 && image->kv == kv, "roundtrip changed structure/KV bytes");
        check(image->mixed.embeddings.size() == mixed.embeddings.size() &&
              !std::memcmp(image->mixed.embeddings.data(), mixed.embeddings.data(), mixed.embeddings.size() * 4),
              "roundtrip changed actual F32 identity");
        for (size_t n = 0; n < valid.size(); ++n) check(!read(valid.substr(0, n)), "truncated payload accepted");
        check(!read(valid + "x"), "trailing payload accepted");
        check(!read(valid, valid.size() - 1), "byte budget ignored");
        for (const auto & edit : std::vector<std::pair<size_t, uint64_t>>{
                {0, 0}, {8, UINT64_MAX}, {16, UINT64_MAX}, {24, 3}, {32, 1}, {40, 1}}) {
            auto bad = valid; field(bad, edit.first, edit.second);
            check(!read(bad), "invalid header accepted");
        }
        {
            auto bad = valid; int32_t invalid = 8; std::memcpy(bad.data() + 48, &invalid, 4);
            check(!read(bad), "out-of-vocabulary token accepted");
        }
        {
            auto bad = valid; int32_t invalid = 1;
            std::memcpy(bad.data() + 48 + ids.size() * 4 + 4, &invalid, 4);
            check(!read(bad), "wrong visual row accepted");
        }
        {
            auto bad = valid; const uint32_t nan = 0x7fc00000;
            std::memcpy(bad.data() + 48 + ids.size() * 8, &nan, 4);
            check(!read(bad), "nonfinite image identity accepted");
        }
        int calls = 0;
        check(!read(valid, 1 << 20, [&] { return ++calls == 2; }), "cancel during payload read ignored");
        auto other = g; ++other.embd;
        std::istringstream in(valid, std::ios::binary);
        check(!prefix_codec::read(in, valid.size(), 1 << 20, other), "changed image geometry accepted");
        mixed.visual_rows.clear(); std::ostringstream invalid;
        check(!prefix_codec::write(invalid, ids, mixed, kv), "incomplete visual identity written");
        std::cout << "PASS M31 host-only prefix codec (roundtrip, malformed, budget, cancel)\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
