// tools/v41/engine/ds41_session_file.hpp - disk sessions for ds41_generate --serve: `SAVE <path>` / `RESTORE <path>`
// between requests, the protocol serve/server.py's Engine.session_file speaks (and llama-server's POST
// /slots/0?action=save|restore {"filename": ...} with --slot-save-path / the config's "slot_save_path").
//
// (local, K:/model/test, 2026-10-26.)  The file holds what a conversation-pool record holds: the live conversation at
// its TURN BOUNDARY (the snapshot Ds41Dense::snapshot() took at the end of the last prompt: window ring + ratio-2
// compressor state + every compressed-KV / indexer-key row before it = Ds41Dense::state_regions), the boundary's
// tokens, position identities (pictures) and last-row logits.  The boundary - not the end of the reply - is what
// the next turn's prompt starts with (the client re-renders the reply), so a restored session is resumed exactly as
// a pool record is: the engine matches the next prompt against the restored tokens and feeds only what follows.
//
// Layout (little endian): "DS41SES1" | u32 version | u64 fingerprint (model shards' + Engram files' sizes, layers,
// context) | i64 n | u32 regions | per region: u64 off, u64 bytes, u32 ggml type, i64 ne[4] | u32 ident flag | u64 logits
// | i32 tokens[n] | u64 ident[n] (flag 1) | f32 logits[] | region bytes ... | u64 FNV-1a of everything before it.
// Written to <path>.tmp and renamed over <path> only when complete; a restore reads and checks the whole file before
// the first device write, so a refused or damaged file leaves the live state as it was.
#pragma once

#include "ds41_dense.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace strata::v41 {

struct Ds41SessionImage {
    int64_t n = 0;
    std::vector<int> tokens;
    std::vector<uint64_t> ident;     // empty = all text
    std::vector<float> logits;
    std::vector<uint8_t> data;       // the regions' bytes, in state_regions(n) order
};

namespace sessf {
constexpr char kMagic[8] = { 'D', 'S', '4', '1', 'S', 'E', 'S', '1' };
constexpr uint32_t kVersion = 2;   // 2: regions identified by type + shape (1: by tensor name, which is not stable)

inline uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Writer {
    std::FILE * f = nullptr;
    uint64_t h = 1469598103934665603ull;
    size_t bytes = 0;
    bool ok = true;
    void put(const void * p, size_t n) {
        if (!ok || n == 0) return;
        h = fnv(p, n, h);
        ok = std::fwrite(p, 1, n, f) == n;
        bytes += n;
    }
    template <class T> void pod(const T & v) { put(&v, sizeof v); }
};

struct Reader {
    const std::vector<uint8_t> & b;
    size_t o = 0;
    bool ok = true;
    bool get(void * p, size_t n) {
        if (!ok || o + n > b.size()) { ok = false; return false; }
        std::memcpy(p, b.data() + o, n);
        o += n;
        return true;
    }
    template <class T> bool pod(T & v) { return get(&v, sizeof v); }
};

inline std::filesystem::path u8(const std::string & s) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(s.data()), s.size()));
}
}  // namespace sessf

/// The identity a file must match: the model's shard sizes, the Engram files' sizes, the layer count and the context
/// (the position caches' row counts follow from it).
inline uint64_t ds41_session_fingerprint(const std::vector<std::string> & files, int64_t n_layer, int64_t ctx) {
    uint64_t h = 1469598103934665603ull;
    for (const auto & f : files) {
        std::error_code ec;
        const uint64_t sz = (uint64_t) std::filesystem::file_size(sessf::u8(f), ec);
        const uint64_t v = ec ? ~0ull : sz;
        h = sessf::fnv(&v, sizeof v, h);
    }
    h = sessf::fnv(&n_layer, sizeof n_layer, h);
    h = sessf::fnv(&ctx, sizeof ctx, h);
    return h;
}

/// SAVE: the boundary n (the snapshot tensors hold it) to `path`.  kind = invalid / storage / io on failure.
inline bool ds41_session_save(const std::string & path, Ds41Dense & dense, uint64_t fp, const int * tokens,
                              const uint64_t * ident, size_t ident_n, int64_t n, const std::vector<float> & logits,
                              size_t & bytes, std::string & err, const char *& kind) {
    kind = "invalid";
    if (n < 1) { err = "no complete session to save"; return false; }
    const auto regs = dense.state_regions(n);
    if (regs.empty()) { err = "no state to save"; return false; }
    bool any_img = false;
    for (size_t i = 0; i < ident_n && i < (size_t) n; ++i) any_img |= ident[i] != 0;
    const std::string tmp = path + ".tmp";
    kind = "io";
#if defined(_WIN32)
    std::FILE * f = _wfopen(sessf::u8(tmp).c_str(), L"wb");
#else
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
#endif
    if (!f) { err = "cannot create " + tmp; return false; }
    std::vector<char> vb(16u << 20);
    std::setvbuf(f, vb.data(), _IOFBF, vb.size());
    sessf::Writer w{ f };
    w.put(sessf::kMagic, 8);
    w.pod(sessf::kVersion);
    w.pod(fp);
    w.pod(n);
    const uint32_t nr = (uint32_t) regs.size();
    w.pod(nr);
    for (const auto & g : regs) {   // a region's identity: offset, size, the tensor's type and shape (names are not
        const uint64_t off = g.off, nb = g.bytes;   // stable: unnamed state tensors get graph-order leaf_N names)
        const uint32_t ty = (uint32_t) g.t->type;
        w.pod(off); w.pod(nb); w.pod(ty);
        for (int d = 0; d < 4; ++d) w.pod(g.t->ne[d]);
    }
    const uint32_t flag = any_img ? 1u : 0u;
    w.pod(flag);
    const uint64_t nl = logits.size();
    w.pod(nl);
    w.put(tokens, (size_t) n * sizeof(int));
    if (any_img) {
        std::vector<uint64_t> id((size_t) n, 0);
        for (size_t i = 0; i < ident_n && i < (size_t) n; ++i) id[i] = ident[i];
        w.put(id.data(), id.size() * sizeof(uint64_t));
    }
    w.put(logits.data(), logits.size() * sizeof(float));
    ggml_backend_synchronize(dense.backend());
    std::vector<uint8_t> chunk;
    for (const auto & g : regs) {
        for (size_t done = 0; done < g.bytes && w.ok;) {
            const size_t k = std::min<size_t>(g.bytes - done, 64u << 20);
            chunk.resize(k);
            ggml_backend_tensor_get(g.t, chunk.data(), g.off + done, k);
            w.put(chunk.data(), k);
            done += k;
        }
    }
    const uint64_t sum = w.h;
    if (w.ok) w.ok = std::fwrite(&sum, 1, sizeof sum, f) == sizeof sum;
    const bool flushed = std::fflush(f) == 0;
    std::fclose(f);
    std::error_code ec;
    if (!w.ok || !flushed) {
        std::filesystem::remove(sessf::u8(tmp), ec);
        kind = "storage";
        err = "writing " + tmp + " failed (disk full?)";
        return false;
    }
    std::filesystem::rename(sessf::u8(tmp), sessf::u8(path), ec);   // replaces an existing file
    if (ec) {
        std::filesystem::remove(sessf::u8(tmp), ec);
        err = "cannot rename the new file over " + path;
        return false;
    }
    bytes = w.bytes + sizeof sum;
    return true;
}

/// RESTORE, step 1: read and check the whole file against this engine; no device write.
inline bool ds41_session_read(const std::string & path, const Ds41Dense & dense, uint64_t fp, Ds41SessionImage & img,
                              size_t & bytes, std::string & err, const char *& kind) {
    kind = "io";
    std::error_code ec;
    const uint64_t size = (uint64_t) std::filesystem::file_size(sessf::u8(path), ec);
    if (ec) { err = "cannot open " + path; return false; }
#if defined(_WIN32)
    std::FILE * f = _wfopen(sessf::u8(path).c_str(), L"rb");
#else
    std::FILE * f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> b;
    try {
        b.resize((size_t) size);
    } catch (const std::bad_alloc &) {
        std::fclose(f);
        kind = "memory";
        err = "not enough RAM to read the session";
        return false;
    }
    const bool rd = std::fread(b.data(), 1, b.size(), f) == b.size();
    std::fclose(f);
    if (!rd) { err = "reading " + path + " failed"; return false; }
    kind = "invalid";
    if (b.size() < 8 + sizeof(uint64_t) || std::memcmp(b.data(), sessf::kMagic, 8) != 0) {
        err = "not a DeepSeek V4.1 session file";
        return false;
    }
    uint64_t sum = 0;
    std::memcpy(&sum, b.data() + b.size() - sizeof sum, sizeof sum);
    if (sessf::fnv(b.data(), b.size() - sizeof sum, 1469598103934665603ull) != sum) {
        err = "the session file is damaged (checksum)";
        return false;
    }
    sessf::Reader r{ b };
    r.o = 8;
    uint32_t ver = 0;
    uint64_t ffp = 0;
    int64_t n = 0;
    uint32_t nr = 0;
    r.pod(ver); r.pod(ffp); r.pod(n); r.pod(nr);
    if (!r.ok || ver != sessf::kVersion) { err = "unsupported session file version"; return false; }
    if (ffp != fp) { err = "the session was saved by a different model / context size"; return false; }
    const auto regs = dense.state_regions(n);
    if (n < 1 || regs.size() != nr) { err = "the session does not match this engine's state layout"; return false; }
    size_t total = 0;
    for (size_t i = 0; i < regs.size(); ++i) {
        const auto & g = regs[i];
        uint64_t off = 0, nb = 0;
        uint32_t ty = 0;
        int64_t ne[4] = { 0, 0, 0, 0 };
        r.pod(off); r.pod(nb); r.pod(ty);
        for (int d = 0; d < 4; ++d) r.pod(ne[d]);
        if (!r.ok) { err = "truncated session file"; return false; }
        bool same = off == g.off && nb == g.bytes && ty == (uint32_t) g.t->type;
        for (int d = 0; d < 4; ++d) same = same && ne[d] == g.t->ne[d];
        if (!same) {
            err = "the session does not match this engine's state layout (region " + std::to_string(i) + ")";
            return false;
        }
        total += g.bytes;
    }
    uint32_t flag = 0;
    uint64_t nl = 0;
    r.pod(flag); r.pod(nl);
    if (!r.ok || nl > (1u << 26)) { err = "truncated session file"; return false; }
    img = Ds41SessionImage();
    img.n = n;
    img.tokens.resize((size_t) n);
    r.get(img.tokens.data(), (size_t) n * sizeof(int));
    if (flag) {
        img.ident.resize((size_t) n);
        r.get(img.ident.data(), (size_t) n * sizeof(uint64_t));
    }
    img.logits.resize((size_t) nl);
    r.get(img.logits.data(), (size_t) nl * sizeof(float));
    if (!r.ok || r.o + total + sizeof sum != b.size()) { err = "truncated session file"; return false; }
    img.data.assign(b.begin() + (std::ptrdiff_t) r.o, b.begin() + (std::ptrdiff_t) (r.o + total));
    bytes = b.size();
    return true;
}

/// RESTORE, step 2: the regions into the device (the snapshot tensors + position caches).  The caller then does
/// rearm(n) + restore() and takes the tokens as its history.
inline void ds41_session_apply(Ds41Dense & dense, const Ds41SessionImage & img) {
    const auto regs = dense.state_regions(img.n);
    ggml_backend_synchronize(dense.backend());
    size_t o = 0;
    for (const auto & g : regs) {
        ggml_backend_tensor_set(g.t, img.data.data() + o, g.off, g.bytes);
        o += g.bytes;
    }
    ggml_backend_synchronize(dense.backend());
}

}  // namespace strata::v41
