// tools/v41/engine/ds41_engram.cpp - see ds41_engram.hpp.
#include "ds41_engram.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#ifndef _WIN32
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/sysmacros.h>
#endif
#include <cstdlib>
#include <cstring>
#include "ds41_sys.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace strata::v41 {

namespace {

std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool exists(const std::string & p) { std::error_code ec; return std::filesystem::exists(p, ec); }

// "key": "value" in a JSON text (the HF index's weight_map; names are unique)
std::string json_str(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return {};
    p = js.find('"', js.find(':', p + pat.size()) + 1);
    const size_t q = js.find('"', p + 1);
    return p == std::string::npos || q == std::string::npos ? std::string() : js.substr(p + 1, q - p - 1);
}

// a safetensors tensor's absolute byte offset (8 + header length + data_offsets[0]) and its byte size
bool st_tensor(const std::string & path, const std::string & name, uint64_t & off, uint64_t & bytes, std::string & dtype) {
    const int fd = ds41_sys::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    uint64_t hl = 0;
    bool ok = ds41_sys::pread(fd, &hl, 8, 0) == 8 && hl < (1ull << 30);
    std::string h;
    if (ok) {
        h.resize(hl);
        ok = ds41_sys::pread(fd, h.data(), hl, 8) == (ssize_t) hl;
    }
    ds41_sys::close(fd);
    if (!ok) return false;
    const size_t p = h.find("\"" + name + "\"");
    if (p == std::string::npos) return false;
    const size_t e = h.find('}', p);
    const std::string ent = h.substr(p, e - p);
    dtype = json_str(ent, "dtype");
    const size_t d = ent.find("\"data_offsets\"");
    if (d == std::string::npos) return false;
    const size_t b = ent.find('[', d);
    char * end = nullptr;
    const uint64_t a0 = std::strtoull(ent.c_str() + b + 1, &end, 10);
    const uint64_t a1 = std::strtoull(std::strchr(end, ',') + 1, nullptr, 10);
    off = 8 + hl + a0;
    bytes = a1 - a0;
    return true;
}

// FP8 E4M3 (fn: no inf, S.1111.111 = NaN) -> float
float fp8_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = std::ldexp((float) m, -9);                 // subnormal: m/8 * 2^-6
    else if (e == 15 && m == 7) v = NAN;
    else v = std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}

}  // namespace

// ---- raw block-device reads for a fragmented table file ----------------------------------------------------------
// ntfs3 serves random reads into a heavily fragmented file badly (measured: a 101 GB table in 62,112 extents, written
// onto a 99%-full NVMe, 9K random 4 KiB reads/s vs 304K/s for a 1,226-extent file on the same partition).  When a
// table file has many extents (STRATA_ENGRAM_RAWDEV: auto = >= 1024 extents, 1 = always, 0 = never) its extent map is
// taken once with FIEMAP and rows are pread straight from the partition's block device (read-only; needs read access
// to the device, e.g. the `disk` group).  256 random spans are compared against the filesystem read at open - any
// mismatch turns it off.
struct Ds41Engram::RawMap {
    int dev = -1;
    std::vector<uint64_t> log, phys, len;
    ~RawMap() { if (dev >= 0) ds41_sys::close(dev); }
    ssize_t read(void * buf, size_t n, uint64_t off) const {
        size_t done = 0;
        while (done < n) {
            const uint64_t o = off + done;
            auto it = std::upper_bound(log.begin(), log.end(), o);
            if (it == log.begin()) return -1;
            const size_t i = (size_t) (it - log.begin()) - 1;
            if (o >= log[i] + len[i]) return -1;
            const size_t k = (size_t) std::min<uint64_t>(n - done, log[i] + len[i] - o);
            const ssize_t r = ds41_sys::pread(dev, (uint8_t *) buf + done, k, (int64_t) (phys[i] + (o - log[i])));
            if (r <= 0) return -1;
            done += (size_t) r;
        }
        return (ssize_t) done;
    }
};

static std::shared_ptr<Ds41Engram::RawMap> raw_map(const std::string & path, int fd) {
#ifdef _WIN32
    // FIEMAP + reading the partition's block device is Linux-only; Windows reads the table through NTFS, whose
    // random reads into a fragmented file are not the ntfs3 problem this works around.
    (void) path;
    (void) fd;
    return nullptr;
#else
    const char * env = std::getenv("STRATA_ENGRAM_RAWDEV");
    if (env && env[0] == '0') return nullptr;
    auto m = std::make_shared<Ds41Engram::RawMap>();
    const size_t NX = 512;
    std::vector<uint8_t> mem(sizeof(fiemap) + NX * sizeof(fiemap_extent));
    uint64_t cur = 0;
    for (;;) {
        std::fill(mem.begin(), mem.end(), 0);
        fiemap * fm = (fiemap *) mem.data();
        fm->fm_start = cur;
        fm->fm_length = ~0ull - cur;
        fm->fm_extent_count = NX;
        if (::ioctl(fd, FS_IOC_FIEMAP, fm) != 0 || fm->fm_mapped_extents == 0) break;
        bool last = false;
        for (uint32_t i = 0; i < fm->fm_mapped_extents; ++i) {
            const fiemap_extent & e = fm->fm_extents[i];
            if (e.fe_flags & (FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_INLINE |
                              FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_DATA_ENCRYPTED |
                              FIEMAP_EXTENT_UNWRITTEN))
                return nullptr;
            m->log.push_back(e.fe_logical); m->phys.push_back(e.fe_physical); m->len.push_back(e.fe_length);
            cur = e.fe_logical + e.fe_length;
            if (e.fe_flags & FIEMAP_EXTENT_LAST) last = true;
        }
        if (last) break;
    }
    const bool force = env && env[0] == '1';
    if (m->log.empty() || (!force && m->log.size() < 1024)) return nullptr;
    ds41_sys::Stat st;
    if (ds41_sys::fstat(fd, &st) != 0) return nullptr;
    char sys[96];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u/uevent", major(st.st_dev), minor(st.st_dev));
    std::ifstream ue(sys);
    std::string line, devname;
    while (std::getline(ue, line)) if (line.rfind("DEVNAME=", 0) == 0) devname = line.substr(8);
    if (devname.empty()) return nullptr;
    m->dev = ds41_sys::open(("/dev/" + devname).c_str(), O_RDONLY | O_CLOEXEC);
    if (m->dev < 0) { std::fprintf(stderr, "engram: %s: %zu extents, /dev/%s not readable - filesystem reads\n",
                                   path.c_str(), m->log.size(), devname.c_str()); return nullptr; }
    std::mt19937_64 rng(7);
    uint8_t a[4096], b[4096];
    const uint64_t size = (uint64_t) st.st_size;
    for (int k = 0; k < 256; ++k) {
        const uint64_t o = (rng() % (size > 4096 ? size - 4096 : 1)) & ~(uint64_t) 255;
        if (ds41_sys::pread(fd, a, sizeof a, (int64_t) o) != (ssize_t) sizeof a || m->read(b, sizeof b, o) != (ssize_t) sizeof b ||
            std::memcmp(a, b, sizeof a) != 0) {
            std::fprintf(stderr, "engram: %s: raw device check FAILED at %llu - filesystem reads\n", path.c_str(),
                         (unsigned long long) o);
            return nullptr;
        }
    }
    std::fprintf(stderr, "engram: %s: %zu extents -> raw reads via /dev/%s (256 spans verified)\n", path.c_str(),
                 m->log.size(), devname.c_str());
    return m;
#endif
}

static inline ssize_t tread(int fd, const std::shared_ptr<Ds41Engram::RawMap> & m, void * buf, size_t n, uint64_t off) {
    return m ? m->read(buf, n, off) : ds41_sys::pread(fd, buf, n, (int64_t) off);
}

Ds41Engram::~Ds41Engram() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto & t : workers_) t.join();
    for (auto & t : tables_) {
        if (t.fd_w >= 0) ds41_sys::close(t.fd_w);
        if (t.fd_s >= 0) ds41_sys::close(t.fd_s);
    }
}

void Ds41Engram::setup_hash(const Ds41Geometry & g) {
    g_ = &g;
    cols_ = (g.engram_ngram - 1) * g.engram_heads;
    dim_ = g.engram_dim;
    const int64_t nel = (int64_t) g.engram_layers.size();
    offsets_.assign((size_t) (nel * cols_), 0);
    for (int64_t e = 0; e < nel; ++e) {
        uint64_t acc = 0;
        for (int64_t c = 0; c < cols_; ++c) {
            offsets_[(size_t) (e * cols_ + c)] = (int64_t) acc;
            acc += g.engram_primes[(size_t) (e * cols_ + c)];
        }
    }
}

uint64_t Ds41Engram::geometry_fingerprint() const {
    uint64_t h = 0xcbf29ce484222325ull;   // FNV-1a over everything the hash and the row layout depend on
    auto mix = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    };
    const Ds41Geometry & g = *g_;
    const int64_t hdr[4] = { g.engram_heads, g.engram_dim, g.engram_ngram, g.engram_pad };
    mix(hdr, sizeof hdr);
    mix(g.engram_layers.data(), g.engram_layers.size() * sizeof(g.engram_layers[0]));
    mix(g.engram_rows.data(), g.engram_rows.size() * sizeof(g.engram_rows[0]));
    mix(g.engram_primes.data(), g.engram_primes.size() * sizeof(g.engram_primes[0]));
    mix(g.engram_mult.data(), g.engram_mult.size() * sizeof(g.engram_mult[0]));
    mix(g.engram_token_map.data(), g.engram_token_map.size() * sizeof(g.engram_token_map[0]));
    return h;
}

bool Ds41Engram::read_raw(int e, uint32_t row, uint8_t * dst) const {
    if (e < 0 || e >= (int) tables_.size()) return false;
    const Table & T = tables_[(size_t) e];
    if ((int64_t) row >= T.rows) return false;
    const uint64_t srsz = (uint64_t) dim_ / 32;
    return tread(T.fd_w, T.rm_w, dst, (size_t) dim_, T.off_w + (uint64_t) row * (uint64_t) dim_) == (ssize_t) dim_ &&
           tread(T.fd_s, T.rm_s, dst + dim_, (size_t) srsz, T.off_s + (uint64_t) row * srsz) == (ssize_t) srsz;
}

// Hot-set pack (written by ds41_engram_hot pack):
//   "XENGHOT1" | u32 version (1) | u32 slots | u32 dim | u32 scale bytes (dim/32) | u64 geometry fingerprint | u64 count
//   | u64 keys[count] (sorted ascending, slot << 32 | row) | raw rows[count][dim + dim/32] in key order.
// open() refuses a pack whose geometry fingerprint differs, and spot-checks 64 rows against the tables, so a pack built
// from another checkpoint's Engram (e.g. V4.1 vs 1017 - same geometry, different weights) is caught and skipped.
bool Ds41Engram::load_hot(const std::string & path, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    const int fd = ds41_sys::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open"; return false; }
    struct Hdr { char magic[8]; uint32_t version, slots, dim, srsz; uint64_t fp, count; } h {};
    const int64_t srsz = dim_ / 32, rsz = dim_ + srsz;
    bool ok = ds41_sys::pread(fd, &h, sizeof h, 0) == (ssize_t) sizeof h;
    if (ok && (std::memcmp(h.magic, "XENGHOT1", 8) != 0 || h.version != 1)) { err = "not a hot-set pack"; ok = false; }
    if (ok && (h.slots != tables_.size() || h.dim != (uint32_t) dim_ || h.srsz != (uint32_t) srsz)) { err = "shape mismatch"; ok = false; }
    if (ok && h.fp != geometry_fingerprint()) { err = "geometry fingerprint mismatch (built for another model)"; ok = false; }
    ds41_sys::Stat st;
    if (ok && (ds41_sys::fstat(fd, &st) != 0 || (uint64_t) st.st_size != sizeof h + h.count * (8 + (uint64_t) rsz))) { err = "truncated"; ok = false; }
    if (ok) {
        ds41_sys::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        hot_keys_.resize((size_t) h.count);
        hot_pool_.resize((size_t) h.count * (size_t) rsz);
        auto rd = [&](void * dst, uint64_t n, uint64_t off) {
            for (uint64_t done = 0; done < n;) {
                const ssize_t r = ds41_sys::pread(fd, (uint8_t *) dst + done, (size_t) std::min<uint64_t>(n - done, 1ull << 28), (int64_t) (off + done));
                if (r <= 0) return false;
                done += (uint64_t) r;
            }
            return true;
        };
        ok = rd(hot_keys_.data(), h.count * 8, sizeof h) && rd(hot_pool_.data(), hot_pool_.size(), sizeof h + h.count * 8);
        if (!ok) err = "read failed";
    }
    ds41_sys::close(fd);
    for (size_t i = 1; ok && i < hot_keys_.size(); ++i)
        if (hot_keys_[i] <= hot_keys_[i - 1]) { err = "keys not sorted"; ok = false; }
    for (size_t i = 0; ok && i < hot_keys_.size(); ++i) {
        const uint64_t e = hot_keys_[i] >> 32;
        if (e >= tables_.size() || (int64_t) (uint32_t) hot_keys_[i] >= tables_[(size_t) e].rows) { err = "row out of range"; ok = false; }
    }
    // the rows must be THIS checkpoint's: compare 64 spread entries with the tables
    std::vector<uint8_t> buf((size_t) rsz);
    for (int k = 0; ok && k < 64 && !hot_keys_.empty(); ++k) {
        const size_t i = hot_keys_.size() * (size_t) k / 64;
        if (!read_raw((int) (hot_keys_[i] >> 32), (uint32_t) hot_keys_[i], buf.data()) ||
            std::memcmp(buf.data(), &hot_pool_[i * (size_t) rsz], (size_t) rsz) != 0) { err = "rows differ from the tables (built from another checkpoint?)"; ok = false; }
    }
    if (!ok) { hot_keys_.clear(); hot_keys_.shrink_to_fit(); hot_pool_.clear(); hot_pool_.shrink_to_fit(); return false; }
    hot_begin_.assign(tables_.size() + 1, 0);
    for (size_t e = 0; e <= tables_.size(); ++e)
        hot_begin_[e] = (size_t) (std::lower_bound(hot_keys_.begin(), hot_keys_.end(), (uint64_t) e << 32) - hot_keys_.begin());
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "engram: hot set %zu rows (%.2f GiB) from %s in %.1f s (64 rows verified)\n", hot_keys_.size(),
                 (double) hot_pool_.size() / (1024.0 * 1024 * 1024), path.c_str(), dt);
    return true;
}

bool Ds41Engram::open(const Ds41Geometry & g, const std::vector<std::string> & dirs, int threads, std::string & err) {
    setup_hash(g);
    const int64_t nel = (int64_t) g.engram_layers.size();
    std::string index;
    for (const auto & d : dirs) {
        index = slurp(d + "/model.safetensors.index.json");
        if (!index.empty()) break;
    }
    if (index.empty()) { err = "engram: no model.safetensors.index.json in the given dirs"; return false; }
    for (int64_t e = 0; e < nel; ++e) {
        const std::string base = "layers." + std::to_string(g.engram_layers[(size_t) e]) + ".engram.embed.";
        Table t;
        for (int k = 0; k < 2; ++k) {
            const std::string name = base + (k ? "scale" : "weight");
            const std::string file = json_str(index, name);
            if (file.empty()) { err = "engram: " + name + " not in the index"; return false; }
            std::string path;
            for (const auto & d : dirs) if (exists(d + "/" + file)) { path = d + "/" + file; break; }
            if (path.empty()) { err = "engram: " + file + " not found in the given dirs"; return false; }
            uint64_t off = 0, bytes = 0;
            std::string dt;
            if (!st_tensor(path, name, off, bytes, dt)) { err = "engram: cannot read " + name + " from " + path; return false; }
            const int64_t rows = g.engram_rows[(size_t) e];
            if (bytes != (uint64_t) rows * (k ? (uint64_t) dim_ / 32 : (uint64_t) dim_) ||
                dt != (k ? "F8_E8M0" : "F8_E4M3")) {
                err = "engram: " + name + " has an unexpected dtype/size";
                return false;
            }
            const int fd = ds41_sys::open(path.c_str(), O_RDONLY);
            if (fd < 0) { err = "engram: cannot open " + path; return false; }
            ds41_sys::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
            (k ? t.fd_s : t.fd_w) = fd;
            (k ? t.off_s : t.off_w) = off;
            (k ? t.rm_s : t.rm_w) = raw_map(path, fd);
            t.rows = rows;
        }
        tables_.push_back(t);
    }
    for (int i = 0; i < std::max(1, threads); ++i)       // the pool is the disk's queue depth
        workers_.emplace_back([this] { pump(); });
    nthreads_ = (int64_t) workers_.size();
    {
        const char * env = std::getenv("STRATA_ENGRAM_CACHE_ROWS");
        cache_cap_ = env ? (size_t) std::atoll(env) : (size_t) 2000000;
        cache_pool_.assign(cache_cap_ * (size_t) (dim_ + dim_ / 32), 0);
        cache_idx_.reserve(cache_cap_);
        cache_used_ = 0;
    }
    {
        const char * env = std::getenv("STRATA_ENGRAM_HOT");
        const std::string hp = !hot_path_.empty() ? hot_path_ : env ? std::string(env) : std::string();
        std::string herr;
        if (!hp.empty() && hp != "0" && !load_hot(hp, herr))
            std::fprintf(stderr, "engram: hot set %s SKIPPED: %s\n", hp.c_str(), herr.c_str());
    }
    return true;
}

void Ds41Engram::set_token(int64_t pos, int tid) {
    if (pos >= (int64_t) hist_.size()) hist_.resize((size_t) pos + 1024, 0);
    hist_[(size_t) pos] = g_->engram_token_map[(size_t) tid];
}

// An image-span position (NgramHashState.DEAD): not a real compressed id, and it blocks the look-back (below).
static constexpr int64_t kEngramDead = -1;

void Ds41Engram::set_dead(int64_t pos) {
    if (pos >= (int64_t) hist_.size()) hist_.resize((size_t) pos + 1024, 0);
    hist_[(size_t) pos] = kEngramDead;
}

void Ds41Engram::set_dead_in(int s, int64_t pos) {
    std::vector<int64_t> & h = hist_of(s);
    if (pos >= (int64_t) h.size()) h.resize((size_t) pos + 1024, 0);
    h[(size_t) pos] = kEngramDead;
}

std::vector<int64_t> & Ds41Engram::hist_of(int s) {
    if (s == hist_cur_) return hist_;
    if (s >= (int) hist_store_.size()) hist_store_.resize((size_t) s + 1);
    return hist_store_[(size_t) s];
}

void Ds41Engram::select_history(int s) {
    if (s < 0 || s == hist_cur_) return;
    const int need = std::max(s, hist_cur_) + 1;
    if ((int) hist_store_.size() < need) hist_store_.resize((size_t) need);
    std::swap(hist_, hist_store_[(size_t) hist_cur_]);   // the current history back to its place ...
    std::swap(hist_, hist_store_[(size_t) s]);           // ... and slot s's in (its place keeps the empty placeholder)
    hist_cur_ = s;
}

void Ds41Engram::swap_histories(int a, int b) {
    if (a < 0 || b < 0 || a == b) return;
    std::swap(hist_of(a), hist_of(b));
}

void Ds41Engram::set_token_in(int s, int64_t pos, int tid) {
    std::vector<int64_t> & h = hist_of(s);
    if (pos >= (int64_t) h.size()) h.resize((size_t) pos + 1024, 0);
    h[(size_t) pos] = g_->engram_token_map[(size_t) tid];
}

void Ds41Engram::hash(int64_t pos, int e, int64_t * out) const { hash_in(hist_, pos, e, out); }

void Ds41Engram::hash_in(const std::vector<int64_t> & hist, int64_t pos, int e, int64_t * out) const {
    const int64_t N = g_->engram_ngram, H = g_->engram_heads;
    int64_t tok[8];
    // Look-back stops at the start of the sequence and at any DEAD token: once blocked, this and every earlier token
    // of the n-gram is the pad id (model.py engram.py forward: blocked is cumulative over the shifts).
    bool blocked = false;
    for (int64_t k = 0; k < N; ++k) {
        int64_t v = g_->engram_pad;
        if (pos - k < 0) blocked = true;
        else {
            const int64_t h = pos - k < (int64_t) hist.size() ? hist[(size_t) (pos - k)] : 0;
            if (h == kEngramDead) blocked = true;
            else if (!blocked) v = h;
        }
        tok[k] = v;
    }
    const int64_t * mult = &g_->engram_mult[(size_t) (e * N)];
    int64_t rolling = tok[0] * mult[0];
    for (int64_t i = 1; i < N; ++i) {
        rolling ^= tok[i] * mult[i];
        for (int64_t h = 0; h < H; ++h) {
            const int64_t c = (i - 1) * H + h;
            const int64_t p = (int64_t) g_->engram_primes[(size_t) (e * cols_ + c)];
            out[c] = rolling % p + offsets_[(size_t) (e * cols_ + c)];
        }
    }
}

bool Ds41Engram::rows_reference(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    out.resize((size_t) nel);
    std::vector<int64_t> ids((size_t) (nel * n * cols_));
    for (int64_t e = 0; e < nel; ++e) {
        out[(size_t) e].assign((size_t) (n * cols_ * dim_), 0.0f);
        for (int t = 0; t < n; ++t) hash(pos0 + t, (int) e, &ids[(size_t) ((e * n + t) * cols_)]);
    }
    const int64_t total = nel * n * cols_;
    std::atomic<int> bad { 0 };
    std::atomic<uint64_t> bytes { 0 };
    const std::function<void(int64_t, int64_t)> fn = [&](int64_t a, int64_t b) {
        uint8_t w[512], s[16];
        for (int64_t j = a; j < b; ++j) {
            const int64_t e = j / (n * cols_);
            const Table & T = tables_[(size_t) e];
            const int64_t r = ids[(size_t) j];
            if (r < 0 || r >= T.rows ||
                ds41_sys::pread(T.fd_w, w, (size_t) dim_, (int64_t) (T.off_w + (uint64_t) r * (uint64_t) dim_)) != (ssize_t) dim_ ||
                ds41_sys::pread(T.fd_s, s, (size_t) (dim_ / 32), (int64_t) (T.off_s + (uint64_t) r * (uint64_t) (dim_ / 32))) != (ssize_t) (dim_ / 32)) {
                bad.fetch_add(1);
                continue;
            }
            bytes.fetch_add((uint64_t) (dim_ + dim_ / 32));
            float * o = &out[(size_t) e][(size_t) ((j % (n * cols_)) * dim_)];
            for (int64_t q = 0; q < dim_ / 32; ++q) {
                const float sc = std::ldexp(1.0f, (int) s[q] - 127);
                for (int64_t i = 0; i < 32; ++i) o[q * 32 + i] = fp8_e4m3(w[q * 32 + i]) * sc;
            }
        }
    };
    parallel(total, fn);
    bytes_ += bytes.load();
    if (bad.load()) { err = "engram: " + std::to_string(bad.load()) + " row reads failed"; return false; }
    return true;
}

// One sorted, page-coalesced read: the unique rows uniq[u0 .. u0+count) all have their weight bytes inside one 4 KiB page
// (so [off_w, off_w+len_w) is the exact byte span, one pread), and their scale bytes span [off_s, off_s+len_s) (a second).
// The group dequantizes each row straight into its output slots - no intermediate buffer, no second pass.
struct Ds41Engram::ReadGroup {
    int64_t e = 0;                    // layer slot (indexes the batch's per-layer state)
    uint64_t off_w = 0, off_s = 0;    // weight / scale byte spans
    uint32_t len_w = 0, len_s = 0;
    uint32_t u0 = 0, count = 0;       // rows uniq[u0 .. u0+count)
};

struct Ds41Engram::Batch {
    int64_t n = 0;
    std::vector<std::vector<float>> * out = nullptr;  // [layer][n * cols * dim], the caller's storage
    std::vector<ReadGroup> groups;
    std::vector<std::vector<uint32_t>> uniq;          // [layer] sorted unique row ids
    std::vector<std::vector<uint64_t>> packed;         // [layer] sorted (row_id << 32 | slot): no-comparator sort
    std::vector<std::vector<uint32_t>> run_begin;     // [layer] unique row -> first sorted position (size u+1)
    // In-RAM rows (hot set / row cache) found by start(): dequantized by the pool together with the reads (was serial on
    // the caller: a fully cached 4096-token prompt chunk took ~0.45 s single-threaded, 2026-10-19).
    struct Hit { int32_t e; uint32_t k; const uint8_t * w; };
    std::vector<Hit> hits;
    std::shared_ptr<Task> read_task;
    std::atomic<int> bad { 0 };
    std::atomic<uint64_t> bytes { 0 };
};

void Ds41Engram::drop_cache() {
    // Evict in 4 GiB windows: one giant fadvise(0,0,DONTNEED) over a 94 GB file leaves the page cache in a state that
    // makes the next reads pathologically slow (measured 40K reads/s vs 158K/s after windowed eviction), so windows it is.
    for (const Table & t : tables_) {
        const int fd = t.fd_w >= 0 ? t.fd_w : t.fd_s;
        ds41_sys::Stat st;
        if (fd < 0 || ds41_sys::fstat(fd, &st) != 0) continue;
        const uint64_t win = 4ull << 30;
        for (uint64_t o = 0; o < (uint64_t) st.st_size; o += win)
            ds41_sys::posix_fadvise(fd, (int64_t) o, (int64_t) win, POSIX_FADV_DONTNEED);
        ds41_sys::Stat s2;
        if (t.fd_s >= 0 && ds41_sys::fstat(t.fd_s, &s2) == 0 && s2.st_ino != st.st_ino)   // scale in another shard
            for (uint64_t o = 0; o < (uint64_t) s2.st_size; o += win)
                ds41_sys::posix_fadvise(t.fd_s, (int64_t) o, (int64_t) win, POSIX_FADV_DONTNEED);
    }
}

std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    if (nel == 0) { err = "engram: no tables"; return nullptr; }
    if (n <= 0) { err = "engram: empty batch"; return nullptr; }
    const int64_t m = n * cols_;
    std::vector<int64_t> ids((size_t) (nel * m));
    for (int64_t e = 0; e < nel; ++e)
        for (int64_t t = 0; t < n; ++t) hash(pos0 + t, (int) e, &ids[(size_t) ((e * n + t) * cols_)]);
    return start_ids(ids, n, out, err);
}

bool Ds41Engram::rows_at(const int * slot, const int64_t * pos, int n, std::vector<std::vector<float>> & out, std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    if (nel == 0) { err = "engram: no tables"; return false; }
    if (n <= 0) { err = "engram: empty batch"; return false; }
    const int64_t m = n * cols_;
    std::vector<int64_t> ids((size_t) (nel * m));
    for (int64_t e = 0; e < nel; ++e)
        for (int64_t t = 0; t < n; ++t) hash_in(hist_of(slot[t]), pos[t], (int) e, &ids[(size_t) ((e * n + t) * cols_)]);
    auto b = start_ids(ids, n, out, err);
    if (!b) return false;
    return wait(b, err);
}

// The reads of n rows' hash ids (ids[(e * n + t) * cols + c], every Engram slot e): start() and rows_at().
std::shared_ptr<Ds41Engram::Batch> Ds41Engram::start_ids(std::vector<int64_t> & ids, int n, std::vector<std::vector<float>> & out,
                                                         std::string & err) {
    const int64_t nel = (int64_t) tables_.size();
    auto b = std::make_shared<Batch>();
    b->n = n;
    b->out = &out;
    const int64_t cols = cols_, dim = dim_, srsz = dim / 32, m = n * cols;

    out.resize((size_t) nel);
    b->uniq.resize((size_t) nel); b->packed.resize((size_t) nel); b->run_begin.resize((size_t) nel);
    for (int64_t e = 0; e < nel; ++e) {
        const Table & T = tables_[(size_t) e];
        out[(size_t) e].assign((size_t) (m * dim), 0.0f);   // reuses the caller's capacity - no per-pass allocation
        // pack (row_id << 32 | slot) so the sort is a plain integer sort - no comparator, no indirection
        auto & pk = b->packed[(size_t) e];
        pk.resize((size_t) m);
        const int64_t * id = &ids[(size_t) (e * m)];
        for (int64_t i = 0; i < m; ++i) pk[(size_t) i] = ((uint64_t) (uint32_t) id[i] << 32) | (uint32_t) i;
        std::sort(pk.begin(), pk.end());
        // walk the sorted order once: unique rows (ascending -> ascending file offsets) and each unique row's slots
        auto & uq = b->uniq[(size_t) e];
        auto & rb = b->run_begin[(size_t) e];
        for (int64_t p = 0; p < m; ++p) {
            const uint32_t r = (uint32_t) (pk[(size_t) p] >> 32);
            if ((uint64_t) r >= (uint64_t) T.rows) b->bad.fetch_add(1);
            if (rb.empty() || uq.back() != r) { uq.push_back(r); rb.push_back((uint32_t) p); }
        }
        rb.push_back((uint32_t) m);
        const size_t u = uq.size();
        const uint64_t bw = T.off_w, bs = T.off_s;
        // In-RAM rows (hot set first, then the row cache) are dequantized into their slots now and never read; the
        // read groups below span the remaining rows only.
        std::vector<char> hit(u, 0);
        auto emit = [&](size_t k, const uint8_t * w) {   // queued; the pool dequantizes it (rows are never rewritten)
            b->hits.push_back(Batch::Hit{ (int32_t) e, (uint32_t) k, w });
            hit[k] = 1;
        };
        if (!hot_keys_.empty()) {   // uq is ascending, so the search window only moves forward
            const uint64_t * hk = hot_keys_.data();
            size_t h = hot_begin_[(size_t) e];
            const size_t hend = hot_begin_[(size_t) e + 1];
            for (size_t k = 0; k < u && h < hend; ++k) {
                if ((uint64_t) uq[k] >= (uint64_t) T.rows) continue;
                const uint64_t key = ((uint64_t) e << 32) | uq[k];
                h = (size_t) (std::lower_bound(hk + h, hk + hend, key) - hk);
                if (h < hend && hk[h] == key) { ++hot_hits_; emit(k, &hot_pool_[h * (size_t) (dim + srsz)]); }
            }
        }
        if (cache_cap_) {
            std::lock_guard<std::mutex> lk(cache_mu_);   // a previous batch's reads may still be filling the cache
            for (size_t k = 0; k < u; ++k) {
                if (hit[k] || (uint64_t) uq[k] >= (uint64_t) T.rows) continue;
                auto it = cache_idx_.find(((uint64_t) e << 32) | uq[k]);
                if (it == cache_idx_.end()) { ++cache_miss_; continue; }
                ++cache_hits_;
                emit(k, &cache_pool_[(size_t) it->second * (size_t) (dim + srsz)]);
            }
        }
        size_t i = 0;
        while (i < u) {
            if ((uint64_t) uq[i] >= (uint64_t) T.rows || hit[i]) { ++i; continue; }   // out of range (bad set) / cached
            const uint64_t pw = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t page = pw & ~(uint64_t) 4095;
            size_t j = i + 1;
            while (j < u && (uint64_t) uq[j] < (uint64_t) T.rows && !hit[j] &&
                   ((bw + (uint64_t) uq[j] * (uint64_t) dim) & ~(uint64_t) 4095) == page) ++j;
            const uint64_t w0 = bw + (uint64_t) uq[i] * (uint64_t) dim;
            const uint64_t w1 = bw + (uint64_t) uq[j - 1] * (uint64_t) dim + (uint64_t) dim;
            const uint64_t s0 = bs + (uint64_t) uq[i] * (uint64_t) srsz;
            const uint64_t s1 = bs + (uint64_t) uq[j - 1] * (uint64_t) srsz + (uint64_t) srsz;
            b->groups.push_back(ReadGroup{ e, w0, s0, (uint32_t) (w1 - w0), (uint32_t) (s1 - s0), (uint32_t) i, (uint32_t) (j - i) });
            i = j;
        }
    }
    if (!b->groups.empty() || !b->hits.empty()) {
        auto t = std::make_shared<Task>();
        const int64_t ng = (int64_t) b->groups.size();
        constexpr int64_t HIT_CHUNK = 64;            // hits per work item
        t->n = ng + ((int64_t) b->hits.size() + HIT_CHUNK - 1) / HIT_CHUNK;
        Batch * bp = b.get();
        t->fn = [this, bp, dim, srsz, ng](int64_t a, int64_t c) {
            uint8_t bw[4096 + 512], bs[1024];
            auto & outv = *bp->out;
            for (int64_t g = a; g < c; ++g) {
                if (g >= ng) {                       // a chunk of in-RAM rows: dequantize into the first slot, copy to the rest
                    const size_t h0 = (size_t) (g - ng) * HIT_CHUNK, h1 = std::min(bp->hits.size(), h0 + HIT_CHUNK);
                    for (size_t h = h0; h < h1; ++h) {
                        const Batch::Hit & H = bp->hits[h];
                        const auto & pk = bp->packed[(size_t) H.e];
                        const auto & rb = bp->run_begin[(size_t) H.e];
                        float * oe = outv[(size_t) H.e].data();
                        const uint8_t * sc = H.w + dim;
                        float * o0 = oe + (size_t) (uint32_t) pk[rb[H.k]] * (size_t) dim;
                        for (int64_t q = 0; q < srsz; ++q) {
                            const float f = std::ldexp(1.0f, (int) sc[q] - 127);
                            for (int64_t x = 0; x < 32; ++x) o0[q * 32 + x] = fp8_e4m3(H.w[q * 32 + x]) * f;
                        }
                        for (uint32_t p = rb[H.k] + 1; p < rb[H.k + 1]; ++p)
                            std::memcpy(oe + (size_t) (uint32_t) pk[p] * (size_t) dim, o0, (size_t) dim * sizeof(float));
                    }
                    continue;
                }
                const ReadGroup & G = bp->groups[(size_t) g];
                const Table & T = tables_[(size_t) G.e];
                if (G.len_w > sizeof bw || G.len_s > sizeof bs ||
                    tread(T.fd_w, T.rm_w, bw, G.len_w, G.off_w) != (ssize_t) G.len_w ||
                    tread(T.fd_s, T.rm_s, bs, G.len_s, G.off_s) != (ssize_t) G.len_s) { bp->bad.fetch_add(1); continue; }
                bp->bytes.fetch_add((uint64_t) G.len_w + G.len_s);
                const auto & uq = bp->uniq[(size_t) G.e];
                const auto & pk = bp->packed[(size_t) G.e];
                const auto & rb = bp->run_begin[(size_t) G.e];
                float * oe = outv[(size_t) G.e].data();
                const uint32_t r0 = uq[G.u0];      // the group's first (lowest-offset) row: the buffer's origin
                for (uint32_t k = 0; k < G.count; ++k) {
                    const uint32_t uu = G.u0 + k;
                    const uint32_t row = uq[uu];
                    const uint8_t * w = bw + (uint64_t) (row - r0) * (uint64_t) dim;
                    const uint8_t * s = bs + (uint64_t) (row - r0) * (uint64_t) srsz;
                    if (cache_cap_ && cache_used_ < cache_cap_) {   // keep the raw row (fill-until-full)
                        std::lock_guard<std::mutex> lk(cache_mu_);
                        if (cache_used_ < cache_cap_ && cache_idx_.emplace(((uint64_t) G.e << 32) | row, (uint32_t) cache_used_).second) {
                            uint8_t * d = &cache_pool_[cache_used_ * (size_t) (dim + srsz)];
                            std::memcpy(d, w, (size_t) dim);
                            std::memcpy(d + dim, s, (size_t) srsz);
                            ++cache_used_;
                        }
                    }
                    const uint32_t p0 = rb[uu];
                    float * o0 = oe + (size_t) (uint32_t) pk[p0] * (size_t) dim;   // dequantize straight into the first slot
                    for (int64_t q = 0; q < srsz; ++q) {
                        const float sc = std::ldexp(1.0f, (int) s[q] - 127);
                        for (int64_t x = 0; x < 32; ++x) o0[q * 32 + x] = fp8_e4m3(w[q * 32 + x]) * sc;
                    }
                    for (uint32_t p = p0 + 1; p < rb[uu + 1]; ++p)               // ... and share it with the row's other slots
                        std::memcpy(oe + (size_t) (uint32_t) pk[p] * (size_t) dim, o0, (size_t) dim * sizeof(float));
                }
            }
        };
        b->read_task = t;
        if (nthreads_ <= 0 || t->n < 4) {            // tiny batch: run on the caller, no queue round-trip.  (Was n <= 2 *
                                                     // nthreads: a decode token's ~48 preads then ran serially, ~11 ms cold.)
            t->parts = 1;
            t->pending.store(1);
            t->fn(0, t->n);
            t->done();
        } else {
            fire(t, std::min<int64_t>(t->n, 8 * nthreads_));
        }
    }
    return b;
}

bool Ds41Engram::wait(const std::shared_ptr<Batch> & b, std::string & err) {
    if (!b) { err = "engram: null batch"; return false; }
    if (b->read_task) b->read_task->wait();
    reads_ += (uint64_t) b->groups.size();
    bytes_ += b->bytes.load();
    if (b->bad.load()) { err = "engram: " + std::to_string(b->bad.load()) + " row reads failed"; return false; }
    return true;
}

bool Ds41Engram::rows(int64_t pos0, int n, std::vector<std::vector<float>> & out, std::string & err) {
    auto b = start(pos0, n, out, err);
    if (!b) return false;
    return wait(b, err);
}

void Ds41Engram::pump() {
    for (;;) {
        std::function<void()> j;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            j = std::move(queue_.front());
            queue_.pop_front();
        }
        j();
    }
}

void Ds41Engram::task_run(const std::shared_ptr<Task> & t) {
    for (;;) {
        const int64_t k = t->next.fetch_add(1);
        if (k >= t->parts) break;
        const int64_t a = t->n * k / t->parts, b = t->n * (k + 1) / t->parts;
        if (a < b) t->fn(a, b);
    }
    t->done();
}

void Ds41Engram::fire(const std::shared_ptr<Task> & t, int64_t parts) {
    t->parts = std::max<int64_t>(1, std::min(parts, std::max<int64_t>(1, t->n)));
    t->next.store(0);
    const int64_t njobs = std::min<int64_t>(t->parts, std::max<int64_t>(1, nthreads_));
    t->pending.store((int) njobs);
    for (int64_t i = 0; i < njobs; ++i) {
        std::shared_ptr<Task> tt = t;
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back([this, tt] { task_run(tt); });
        cv_.notify_one();
    }
}

void Ds41Engram::run_sync(const std::shared_ptr<Task> & t, int64_t parts) {
    if (t->n <= 0) return;
    if (nthreads_ <= 0 || t->n < 4) {           // tiny batches (decode): stay on the caller, no queue latency
        t->parts = 1;
        t->next.store(0);
        t->pending.store(1);
        t->fn(0, t->n);
        t->done();
        return;
    }
    fire(t, parts);
    t->wait();
}

void Ds41Engram::parallel(int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
    if (n <= 0) return;
    if (workers_.empty() || n < 4) { fn(0, n); return; }
    auto t = std::make_shared<Task>();
    t->n = n;
    t->fn = fn;
    run_sync(t, 4 * (nthreads_ + 1));
}

}  // namespace strata::v41
