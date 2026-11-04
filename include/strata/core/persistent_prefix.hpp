// Opt-in host-only persistent exact-prefix catalogue. Payload formats belong to the native engines.
// A directory is one global quota domain, including all model identities and processes.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <sstream>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core::prefix {
namespace fs = std::filesystem;
// Opt-in lifecycle metadata for qualifying request-owned capture cancellation.
// This observes private capture work, never the asynchronous disk writer.
class CaptureTrace {
    bool enabled_ = std::getenv("STRATA_PREFIX_CAPTURE_TRACE") != nullptr;
    bool cancelled_ = false, finished_ = false;
    const char* engine_;
    const char* source_;
    int slot_;
    long long request_epoch_;
    uint64_t id_ = 0;
    void event(const char* name) const {
        if (!enabled_) return;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
#ifndef _WIN32
        const auto pid = ::getpid();
#else
        const int pid = 0;
#endif
        std::fprintf(stderr,"strata-prefix-capture: engine=%s source=%s pid=%lld slot=%d request_epoch=%lld capture=%llu monotonic_ns=%lld event=%s\n",
                     engine_,source_,(long long)pid,slot_,request_epoch_,(unsigned long long)id_,(long long)ns,name);
    }
public:
    CaptureTrace(const char* engine,const char* source,int slot,long long request_epoch=-1)
        : engine_(engine),source_(source),slot_(slot),request_epoch_(request_epoch) {
        if (enabled_) { static std::atomic<uint64_t> next{0}; id_=++next; event("begin"); }
    }
    CaptureTrace(const CaptureTrace&) = delete;
    CaptureTrace& operator=(const CaptureTrace&) = delete;
    bool cancelled(bool value) {
        if (value && !cancelled_) { cancelled_=true; event("cancel_observed"); }
        return value;
    }
    void finish(const char* outcome) { finished_=true; event(outcome); }
    ~CaptureTrace() { if (!finished_) event(cancelled_ ? "aborted" : "discarded"); }
};
struct Config {
    std::string directory;
    uint64_t disk_bytes = uint64_t(64) << 30, min_free_bytes = uint64_t(1) << 30;
    uint64_t queue_bytes = uint64_t(2) << 30;
    uint64_t min_restore_cells = 0;
    bool nonblocking_lookup = false;
    size_t max_entries = 4096;
    static Config from_env() {
        Config c;
        if (const char* s = std::getenv("STRATA_PREFIX_CACHE_DIR")) c.directory = s;
        auto read = [](const char* name, uint64_t& out) {
            if (const char* s = std::getenv(name)) {
                char* end = nullptr; errno = 0;
                const auto n = std::strtoull(s, &end, 10);
                bool digits = *s != 0;
                for (const char* p=s; *p; ++p) if (*p<'0' || *p>'9') digits=false;
                if (!errno && digits && end && !*end && n) out = n;
                else out = 0; // malformed explicit settings disable the tier
            }
        };
        read("STRATA_PREFIX_CACHE_BYTES", c.disk_bytes);
        read("STRATA_PREFIX_CACHE_MIN_FREE_BYTES", c.min_free_bytes);
        read("STRATA_PREFIX_CACHE_QUEUE_BYTES", c.queue_bytes);
        read("STRATA_PREFIX_CACHE_MIN_RESTORE_CELLS", c.min_restore_cells);
        if (const char* s = std::getenv("STRATA_PREFIX_CACHE_NONBLOCKING_LOOKUP"))
            c.nonblocking_lookup = std::string(s) == "1";
        return c;
    }
};
// A persistent entry belongs to this exact host file inventory, executable and settings.
// ctime detects ordinary same-size edits even if mtime is restored. This is not a content digest
// or an authentication boundary; model inputs must remain immutable while the engine is running.
inline std::string model_identity(const std::string& family, const std::string& settings,
                                  std::vector<std::string> files) {
#if defined(__linux__)
    files.push_back("/proc/self/exe");
    // Backend implementations may be shared libraries updated independently of the executable.
    std::ifstream maps("/proc/self/maps");
    if (!maps) return {};
    std::string line;
    while(std::getline(maps,line)) {
        const auto slash=line.find('/');
        if(slash==std::string::npos) continue;
        const auto path=line.substr(slash);
        if(path.find(".so")==std::string::npos) continue;
        if(path.size()>=10 && path.compare(path.size()-10,10," (deleted)")==0) return {};
        files.push_back(path);
    }
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());
    std::ostringstream out;
    out << "prefix-host-v1:" << family.size() << ':' << family << ':' << settings.size() << ':' << settings;
    if (const char* vision = std::getenv("STRATA_PREFIX_CACHE_VISION_ID"))
        out << ":vision-producer:" << std::strlen(vision) << ':' << vision;
    for (const auto& file : files) {
        struct stat st{};
        if (::stat(file.c_str(), &st) || !S_ISREG(st.st_mode)) return {};
        out << ':' << file.size() << ':' << file << ':' << st.st_dev << ':' << st.st_ino << ':' << st.st_size
            << ':' << st.st_mtim.tv_sec << ':' << st.st_mtim.tv_nsec << ':' << st.st_ctim.tv_sec << ':' << st.st_ctim.tv_nsec;
    }
    // Keep catalogue identity bounded even for packs containing thousands of files.
    // The inventory is local compatibility metadata, not an authentication digest.
    const auto inventory=out.str();
    uint64_t a=1469598103934665603ull,b=0x9e3779b97f4a7c15ull;
    for(unsigned char value:inventory) {
        a=(a^value)*1099511628211ull;
        b^=uint64_t(value)+0x9e3779b97f4a7c15ull+(b<<6)+(b>>2);
        b=(b<<27)|(b>>37);b*=0x94d049bb133111ebull;
    }
    auto avalanche=[](uint64_t value) {value^=value>>30;value*=0xbf58476d1ce4e5b9ull;
        value^=value>>27;value*=0x94d049bb133111ebull;return value^(value>>31);};
    std::ostringstream fingerprint;
    fingerprint << "host-inventory-v2/" << std::hex << avalanche(a) << '/' << avalanche(b);
    return fingerprint.str();
#else
    (void)family; (void)settings; (void)files; return {};
#endif
}
inline bool bound_image_producer() {
    const char* value=std::getenv("STRATA_PREFIX_CACHE_VISION_ID");
    const std::string prefix="frontend-vision-pipeline-v1:";
    if(!value || std::strlen(value)!=prefix.size()+64 || std::strncmp(value,prefix.c_str(),prefix.size()))return false;
    for(size_t i=prefix.size();value[i];++i)if(!((value[i]>='0'&&value[i]<='9')||(value[i]>='a'&&value[i]<='f')))return false;
    return true;
}
inline bool image_state_allowed(bool contains_images) { return !contains_images || bound_image_producer(); }
struct Record { std::shared_ptr<void> reservation; std::vector<uint64_t> cells; std::vector<uint8_t> payload; };
struct FileRecord {
    std::vector<uint64_t> cells;
    std::string path;
    // Keeps the global directory lease until the adapter has finished reading the immutable file.
    std::shared_ptr<void> lease;
};
struct Stats {
    std::atomic<uint64_t> hits{0}, misses{0}, writes{0}, failures{0}, dropped{0};
    std::atomic<uint64_t> restored_bytes{0}, saved_bytes{0}, restore_us{0}, save_us{0};
};
class Store {
public:
    // A claim owns one immutable capture through queue completion. Pending claims are
    // weakly cached: abandoning capture or a failed enqueue allows an immediate retry.
    struct CaptureClaim {
        // Destroy accounting last, after every buffer charged to the reservation.
        std::shared_ptr<void> accounting;
        std::string identity;
        std::vector<uint64_t> cells;
        std::string data, index;
        std::string data_stamp, index_stamp;
        bool committed = false, invalid = false;
    };
    using Capture = std::shared_ptr<CaptureClaim>;
    Capture claim_capture(const std::string& identity, const std::vector<uint64_t>& cells) {
        if (!enabled_ || identity.empty() || identity.size()>65536 || cells.empty() || cells.size()>max_cells) return {};
        std::lock_guard<std::mutex> guard(capture_mu_);
        size_t bytes = 0;
        for (auto it=captures_.begin(); it!=captures_.end();) {
            auto value=it->pending.lock();
            if (!value || value->invalid) {
                it=captures_.erase(it); continue;
            }
            if (value->identity==identity && value->cells==cells) {
                if(!value->committed || (file_stamp(value->data)==value->data_stamp && file_stamp(value->index)==value->index_stamp)) return {};
                value->invalid=true; it=captures_.erase(it); continue;
            }
            bytes += value->identity.size()+value->cells.size()*sizeof(uint64_t);
            ++it;
        }
        const size_t needed=identity.size()+cells.size()*sizeof(uint64_t);
        // The optimisation may miss when its bounded metadata budget is full;
        // persistence itself remains available through the ordinary enqueue API.
        // Claims retain a second exact key independently of queued metadata. Charge
        // it even when the bounded dedupe catalogue cannot track another slot.
        auto accounting=reserve(needed+sizeof(CaptureClaim)+2*config_.directory.size()+1024);
        if(!accounting) return {};
        auto value=std::make_shared<CaptureClaim>(); value->accounting=std::move(accounting);
        value->identity=identity; value->cells=cells;
        if (needed<=max_metadata && bytes<=max_metadata-needed && captures_.size()<config_.max_entries)
            captures_.push_back({value,{}});
        return value;
    }
    using Writer = std::function<bool(const std::string&, std::string&)>;
    explicit Store(Config c = Config::from_env()) : config_(std::move(c)) {
        Finally notice{[this] {
            if(!config_.directory.empty()) std::fprintf(stderr,"prefix-cache enabled=%d reason=%s disk_limit=%llu temporary_limit=%llu\n",
                int(enabled_),reason_.c_str(),(unsigned long long)config_.disk_bytes,(unsigned long long)config_.queue_bytes);
        }};
#if defined(__linux__)
        reason_="invalid limits";
        if (config_.directory.empty() || !config_.disk_bytes || !config_.min_free_bytes || !config_.queue_bytes || !config_.max_entries) return;
        reason_="directory unavailable or insecure (requires owner and mode 0700)";
        std::error_code ec;
        const bool created = fs::create_directories(config_.directory, ec);
        if (created && ::chmod(config_.directory.c_str(), 0700)) return;
        if (ec || fs::is_symlink(fs::symlink_status(config_.directory, ec))) return;
        // A private directory is required: these checksums are corruption checks, not authentication.
        struct stat st{};
        if (::stat(config_.directory.c_str(), &st) || st.st_uid != ::geteuid() || (st.st_mode & 0077)) return;
        auto lease = lock();
        reason_="shared directory lease unavailable";
        if (!lease) return;
        reason_="shared quota policy mismatch or unavailable";
        const fs::path policy = fs::path(config_.directory) / "quota";
        uint64_t budget = 0, reserve = 0, entries = 0;
        std::ifstream in(policy);
        if (in) {
            if (!(in >> budget >> reserve >> entries) || budget != config_.disk_bytes ||
                reserve != config_.min_free_bytes || entries != config_.max_entries) return;
        } else {
            std::ofstream out(policy); out << config_.disk_bytes << ' ' << config_.min_free_bytes << ' ' << config_.max_entries << '\n';
            out.close(); if (!out) return;
        }
        enabled_ = true; reason_="ready";
        try { worker_ = std::thread([this] { run(); }); }
        catch (...) { enabled_=false; reason_="background worker unavailable"; }
#else
        reason_="persistent storage unavailable on this platform";
#endif
    }
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;
    ~Store() {
        { std::lock_guard<std::mutex> g(queue_mu_); stopping_ = true; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    bool enabled() const { return enabled_; }
    const Stats& stats() const { return stats_; }
    void report(const char* event) const {
        if(!enabled_) return;
        std::fprintf(stderr,"prefix-cache event=%s hits=%llu misses=%llu writes=%llu errors=%llu dropped=%llu saved_bytes=%llu restored_bytes=%llu save_us=%llu restore_us=%llu temporary_bytes=%llu\n",
            event,(unsigned long long)stats_.hits.load(),(unsigned long long)stats_.misses.load(),
            (unsigned long long)stats_.writes.load(),(unsigned long long)stats_.failures.load(),
            (unsigned long long)stats_.dropped.load(),(unsigned long long)stats_.saved_bytes.load(),
            (unsigned long long)stats_.restored_bytes.load(),(unsigned long long)stats_.save_us.load(),
            (unsigned long long)stats_.restore_us.load(),(unsigned long long)used_->load());
    }
    bool can_enqueue(uint64_t bytes) {
        if (!enabled_ || bytes > config_.disk_bytes || bytes > config_.queue_bytes) return false;
        const auto used = used_->load();
        std::lock_guard<std::mutex> g(queue_mu_);
        return !stopping_ && queue_.size() < 16 && used <= config_.queue_bytes && bytes <= config_.queue_bytes - used;
    }
    // Reservations also bound immutable native snapshots held until request completion.
    std::shared_ptr<void> reserve(uint64_t bytes) {
        if (!enabled_ || bytes > config_.queue_bytes || bytes > config_.disk_bytes) return {};
        uint64_t old = used_->load();
        do { if (old > config_.queue_bytes || bytes > config_.queue_bytes-old) return {}; }
        while (!used_->compare_exchange_weak(old,old+bytes));
        try { return std::make_shared<Reservation>(used_, bytes); }
        catch (...) { *used_ -= bytes; return {}; }
    }
    bool enqueue(std::string identity, std::vector<uint64_t> cells, std::vector<uint8_t> payload) {
        const uint64_t n = payload.size();
        return enqueue_file(std::move(identity), std::move(cells), n,
            [data = std::move(payload)](const std::string& path, std::string&) {
                std::ofstream out(path, std::ios::binary);
                out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
                out.close(); return bool(out);
            });
    }
    bool enqueue_reserved(std::string identity, std::vector<uint64_t> cells, std::vector<uint8_t> payload,
                          std::shared_ptr<void> reservation, Capture capture = {}) {
        const auto n = payload.size();
        std::vector<std::vector<uint64_t>> keys; keys.push_back(std::move(cells));
        return enqueue_file_keys_reserved(std::move(identity),std::move(keys),n,
            [data=std::move(payload)](const std::string& path,std::string&) {
                std::ofstream out(path,std::ios::binary);
                out.write(reinterpret_cast<const char*>(data.data()),data.size()); out.close(); return bool(out);
            },std::move(reservation),std::move(capture));
    }
    bool enqueue_file(std::string identity, std::vector<uint64_t> cells, uint64_t expected_bytes, Writer writer) {
        std::vector<std::vector<uint64_t>> keys; keys.push_back(std::move(cells));
        return enqueue_file_keys(std::move(identity), std::move(keys), expected_bytes, std::move(writer));
    }
    bool enqueue_file_keys(std::string identity, std::vector<std::vector<uint64_t>> keys,
                           uint64_t expected_bytes, Writer writer) {
        return enqueue_file_keys_reserved(std::move(identity), std::move(keys), expected_bytes, std::move(writer), {});
    }
    bool enqueue_file_keys_reserved(std::string identity, std::vector<std::vector<uint64_t>> keys,
                           uint64_t expected_bytes, Writer writer, std::shared_ptr<void> reservation, Capture capture = {}) {
        bool accepted=false;
        Finally release_claim{[&] {
            if(capture && !accepted) { std::lock_guard<std::mutex> guard(capture_mu_); capture->invalid=true; }
        }};
        if(capture && (capture->identity!=identity || keys.size()!=1 || capture->cells!=keys.front())) return false;
        if (!enabled_ || identity.empty() || identity.size() > 65536 || keys.empty() || keys.size() > 256 || !writer) return false;
        uint64_t metadata = identity.size();
        for (const auto& k : keys) {
            if (k.empty() || k.size() > max_cells) return false;
            metadata += 8 + k.size() * 8;
        }
        if (metadata > max_metadata || expected_bytes > config_.disk_bytes || expected_bytes > config_.queue_bytes) {
            ++stats_.dropped; return false;
        }
        std::shared_ptr<void> extra;
        if (reservation) {
            auto r = std::static_pointer_cast<Reservation>(reservation);
            if (r->used != used_ || r->bytes < expected_bytes) return false;
            extra = reserve(metadata);
        } else reservation = reserve(expected_bytes + metadata);
        if (!reservation || (metadata && !extra && std::static_pointer_cast<Reservation>(reservation)->bytes < expected_bytes + metadata)) {
            ++stats_.dropped; return false;
        }
        bool unsubmitted = false;
        if (!std::static_pointer_cast<Reservation>(reservation)->submitted.compare_exchange_strong(unsubmitted,true)) return false;
        std::lock_guard<std::mutex> g(queue_mu_);
        if (stopping_ || queue_.size() >= 16) { ++stats_.dropped; return false; }
        queue_.push_back({std::move(identity), std::move(keys), expected_bytes, metadata, std::move(writer), std::move(reservation), std::move(extra), std::move(capture)});
        accepted=true; cv_.notify_one(); return true;
    }
    std::optional<FileRecord> find_file(const std::string& identity, const std::vector<uint64_t>& cells,
                size_t limit = std::numeric_limits<size_t>::max(), const std::function<bool()>& cancel = {}, size_t minimum_cells = 0, bool auxiliary = false) {
        try {
        if (!enabled_ || minimum_cells >= std::min(limit,cells.size()) || (cancel && cancel())) return {};
        // Restoring a large state to save a few tokens can cost more than prefill.
        // Auxiliary records (encoded images) have no token savings to estimate.
        const uint64_t gain = auxiliary ? 0 : config_.min_restore_cells;
        if (std::min(limit,cells.size()) - minimum_cells < gain) return {};
        const auto start = std::chrono::steady_clock::now();
        auto lease = lock(cancel, config_.nonblocking_lookup);
        if (!lease) { ++stats_.misses; return {}; }
        auto entries = scan(cancel);
        struct Choice { Entry* entry; const std::vector<uint64_t>* key; size_t count; };
        std::vector<Choice> choices;
        for (auto& e : entries) if (e.identity == identity) for (const auto& key : e.keys) {
            auto offer = [&](size_t n) {
                if (n > minimum_cells && n - minimum_cells >= gain && n <= limit && n <= cells.size() && std::equal(key.begin(),key.begin()+n,cells.begin()))
                    choices.push_back({&e,&key,n});
            };
            if (e.lengths.empty()) offer(key.size());
            else for (auto n : e.lengths) offer(n);
        }
        std::sort(choices.begin(),choices.end(),[](const Choice& a,const Choice& b){return a.count>b.count;});
        for (const auto& choice : choices) {
            if (cancel && cancel()) break;
            uint64_t bytes = 0, hash = 0;
            if (!hash_file(choice.entry->data, bytes, hash, cancel) || bytes != choice.entry->bytes || hash != choice.entry->hash) {
                invalidate_capture(choice.entry->data.string()); continue;
            }
            {
                std::lock_guard<std::mutex> guard(capture_mu_);
                const auto old_stamp=file_stamp(choice.entry->index.string());
                std::error_code ec; fs::last_write_time(choice.entry->index, fs::file_time_type::clock::now(), ec);
                for(auto& slot:captures_) if(auto value=slot.pending.lock())
                    if(value->committed && value->data==choice.entry->data.string() && value->index_stamp==old_stamp)
                        value->index_stamp=file_stamp(value->index);
            }
            ++stats_.hits; stats_.restored_bytes += bytes;
            stats_.restore_us += micros(start);
            report("lookup-hit");
            return FileRecord{std::vector<uint64_t>(choice.key->begin(),choice.key->begin()+choice.count), choice.entry->data.string(), std::move(lease)};
        }
        ++stats_.misses; stats_.restore_us += micros(start); report("lookup-miss"); return {};
        } catch (...) { ++stats_.failures; return {}; }
    }
    std::optional<Record> find(const std::string& identity, const std::vector<uint64_t>& cells,
                size_t limit = std::numeric_limits<size_t>::max(), const std::function<bool()>& cancel = {}, size_t minimum_cells = 0, bool auxiliary = false) {
        try {
        auto file = find_file(identity, cells, limit, cancel, minimum_cells, auxiliary);
        if (!file) return {};
        std::error_code ec; const auto n = fs::file_size(file->path, ec);
        if (ec || n > config_.queue_bytes || n > SIZE_MAX) return {};
        Record r; r.cells = std::move(file->cells); r.reservation = reserve(n);
        if (!r.reservation) return {};
        try { r.payload.resize(static_cast<size_t>(n)); } catch (...) { return {}; }
        std::ifstream in(file->path, std::ios::binary);
        size_t offset = 0;
        while (offset < r.payload.size()) {
            if (cancel && cancel()) return {};
            const size_t chunk = std::min<size_t>(1 << 20, r.payload.size() - offset);
            if (!in.read(reinterpret_cast<char*>(r.payload.data() + offset), chunk)) return {};
            offset += chunk;
        }
        return r;
        } catch (...) { ++stats_.failures; return {}; }
    }
    // Deterministic CPU checks and graceful shutdown may wait for the bounded queue.
    void flush() {
        std::unique_lock<std::mutex> g(queue_mu_);
        drained_.wait(g, [this] { return queue_.empty() && !active_; });
    }
private:
    struct Finally { std::function<void()> fn; ~Finally(){fn();} };
    std::string reason_="not configured";
    static constexpr size_t max_cells = 2 * 1024 * 1024, max_metadata = 32 * 1024 * 1024;
    struct Reservation {
        std::shared_ptr<std::atomic<uint64_t>> used;
        uint64_t bytes;
        std::atomic<bool> submitted{false};
        Reservation(std::shared_ptr<std::atomic<uint64_t>> u, uint64_t n) : used(std::move(u)), bytes(n) {}
        ~Reservation() { *used -= bytes; }
    };
    struct Job { std::string identity; std::vector<std::vector<uint64_t>> keys; uint64_t expected, metadata; Writer writer;
                 std::shared_ptr<void> reservation, extra; Capture capture; };
    struct Entry {
        fs::path index, data;
        std::string identity;
        std::vector<std::vector<uint64_t>> keys;
        std::vector<size_t> lengths;
        uint64_t bytes = 0, hash = 0;
        fs::file_time_type touched;
    };
    struct CaptureSlot { std::weak_ptr<CaptureClaim> pending; Capture committed; };
    std::mutex capture_mu_;
    std::vector<CaptureSlot> captures_;
    static std::string file_stamp(const std::string& path) {
#if defined(__linux__)
        struct stat st{};
        if (::lstat(path.c_str(), &st) || !S_ISREG(st.st_mode)) return {};
        std::ostringstream out;
        out<<st.st_dev<<':'<<st.st_ino<<':'<<st.st_size<<':'<<st.st_mtim.tv_sec<<':'<<st.st_mtim.tv_nsec
           <<':'<<st.st_ctim.tv_sec<<':'<<st.st_ctim.tv_nsec;
        return out.str();
#else
        (void)path; return {};
#endif
    }
    void invalidate_capture(const std::string& data) {
        std::lock_guard<std::mutex> guard(capture_mu_);
        for(auto& slot:captures_) if(auto value=slot.pending.lock()) if(value->data==data) {
            value->invalid=true; slot.committed.reset();
        }
    }
    Config config_;
    bool enabled_ = false, stopping_ = false, active_ = false;
    std::shared_ptr<std::atomic<uint64_t>> used_ = std::make_shared<std::atomic<uint64_t>>(0);
    std::mutex queue_mu_;
    std::condition_variable cv_, drained_;
    std::deque<Job> queue_;
    std::thread worker_;
    Stats stats_;
    static uint64_t micros(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    }
    std::shared_ptr<void> lock(const std::function<bool()>& cancel = {}, bool nonblocking = false) const {
#ifndef _WIN32
        const auto path = (fs::path(config_.directory) / "lock").string();
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) return {};
        while (::flock(fd, LOCK_EX | LOCK_NB)) {
            if (errno != EWOULDBLOCK || nonblocking || (cancel && cancel())) { ::close(fd); return {}; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return std::shared_ptr<void>(new int(fd), [](void* p) { auto* v = static_cast<int*>(p); ::flock(*v, LOCK_UN); ::close(*v); delete v; });
#else
        (void)cancel; return {};
#endif
    }
    static uint64_t hash_bytes(const void* ptr, size_t n, uint64_t h = 1469598103934665603ull) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h;
    }
    static bool regular(const fs::path& path) {
        std::error_code ec;
        return fs::is_regular_file(fs::symlink_status(path, ec)) && !ec && fs::hard_link_count(path, ec) == 1 && !ec;
    }
    static bool hash_file(const fs::path& path, uint64_t& bytes, uint64_t& hash, const std::function<bool()>& cancel = {}) {
        if (!regular(path)) return false;
        std::ifstream in(path, std::ios::binary);
        char buf[1 << 16]; bytes = 0; hash = 1469598103934665603ull;
        while (in) {
            if (cancel && cancel()) return false;
            in.read(buf, sizeof(buf)); const auto n = in.gcount();
            if (n > 0) { hash = hash_bytes(buf, n, hash); bytes += n; }
        }
        return in.eof();
    }
    static void u64(std::vector<uint8_t>& out, uint64_t n) {
        for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(n >> (8 * i)));
    }
    static bool read64(const std::vector<uint8_t>& in, size_t& p, uint64_t& n) {
        if (p > in.size() || in.size() - p < 8) return false;
        n = 0; for (unsigned i = 0; i < 8; ++i) n |= uint64_t(in[p++]) << (8 * i);
        return true;
    }
    std::vector<Entry> scan(const std::function<bool()>& cancel = {}) const {
        std::vector<Entry> entries;
        std::error_code ec;
        size_t examined = 0, metadata = 0;
        for (fs::directory_iterator it(config_.directory, ec), end; !ec && it != end; it.increment(ec)) {
            if ((cancel && cancel()) || ++examined > config_.max_entries * 4 + 64) break;
            const auto path = it->path();
            if (path.extension() != ".idx" || !regular(path)) continue;
            const auto size = fs::file_size(path, ec);
            if (ec || size > max_metadata || size < 56 || metadata + size > 64 * 1024 * 1024) continue;
            metadata += size;
            std::vector<uint8_t> raw(static_cast<size_t>(size));
            std::ifstream in(path, std::ios::binary);
            if (!in.read(reinterpret_cast<char*>(raw.data()), raw.size())) continue;
            size_t p = 0; uint64_t magic, ident, count, check;
            Entry e; e.index = path; e.data = path; e.data.replace_extension(".bin");
            if (!read64(raw,p,magic) || magic != 0x3158465054415253ull || !read64(raw,p,e.bytes) ||
                !read64(raw,p,e.hash) || !read64(raw,p,ident) || !read64(raw,p,count) || ident > 65536 || !count || (count & ~(uint64_t(1)<<63)) > 256 ||
                ident > raw.size() - p) continue;
            e.identity.assign(reinterpret_cast<char*>(raw.data()+p), ident); p += ident;
            bool ok = true;
            const bool compact = count >> 63;
            count &= ~(uint64_t(1)<<63);
            if (!count) continue;
            if (compact) {
                uint64_t n = 0;
                if (!read64(raw,p,n) || !n || n>max_cells || n>(raw.size()-p)/8) continue;
                std::vector<uint64_t> key(static_cast<size_t>(n));
                for (auto& cell:key) if (!read64(raw,p,cell)) ok=false;
                e.keys.push_back(std::move(key));
                for(uint64_t i=0;i<count;++i) { uint64_t length=0;
                    if(!read64(raw,p,length)||!length||length>n) {ok=false;break;}
                    e.lengths.push_back(static_cast<size_t>(length));
                }
            }
            for (uint64_t i = 0; !compact && i < count; ++i) {
                uint64_t n = 0;
                if (!read64(raw,p,n) || !n || n > max_cells || n > (raw.size()-p)/8) { ok = false; break; }
                std::vector<uint64_t> key(static_cast<size_t>(n));
                for (auto& cell : key) if (!read64(raw,p,cell)) ok = false;
                e.keys.push_back(std::move(key));
            }
            const size_t hashed = p;
            if (!ok || !read64(raw,p,check) || p != raw.size() || hash_bytes(raw.data(),hashed) != check || !regular(e.data)) continue;
            const auto actual = fs::file_size(e.data, ec);
            if (ec || actual != e.bytes) continue;
            e.touched = fs::last_write_time(path,ec);
            if (!ec) entries.push_back(std::move(e));
            if (entries.size() >= config_.max_entries) break;
        }
        return entries;
    }
    static bool sync_file(const fs::path& path) {
#ifndef _WIN32
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        const bool ok = ::fsync(fd) == 0; ::close(fd); return ok;
#else
        (void)path; return false;
#endif
    }
    bool save(Job& job) {
        auto lease = lock(); if (!lease) return false;
        auto entries = scan();
        std::error_code ec;
        uint64_t total = 0, catalogue = 0;
        const uint64_t required = job.expected + job.metadata + 56;
        // Include uncommitted/orphan payloads in the global byte quota too.
        std::vector<fs::path> garbage;
        for (fs::directory_iterator it(config_.directory,ec),end; !ec && it != end; it.increment(ec)) {
            if ((it->path().extension() == ".bin" || it->path().extension() == ".part" ||
                 it->path().extension() == ".idx" || it->path().extension() == ".meta") && regular(it->path())) {
                const auto size = it->file_size(ec); if (ec || size > UINT64_MAX-total) return false;
                total += size;
                if(it->path().extension()==".idx") catalogue += size;
                bool found = false; for (const auto& e : entries) if (e.data == it->path() || e.index == it->path()) { found = true; break; }
                if (!found) garbage.push_back(it->path());
            }
        }
        if (ec) return false;
        for (const auto& p : garbage) { const auto n = fs::file_size(p,ec); if (ec) return false; fs::remove(p,ec); if (ec) return false; total -= n; if(p.extension()==".idx") catalogue -= n; }
        std::sort(entries.begin(),entries.end(),[](const Entry& a,const Entry& b){ return a.touched < b.touched; });
        auto room = [&] {
            const auto space = fs::space(config_.directory,ec);
            return !ec && required <= config_.disk_bytes && total <= config_.disk_bytes-required &&
                catalogue <= 64*1024*1024 && job.metadata+56 <= 64*1024*1024-catalogue &&
                space.available >= config_.min_free_bytes && required <= space.available-config_.min_free_bytes && entries.size() < config_.max_entries;
        };
        while (!room() && !entries.empty()) {
            const auto e = entries.front(); entries.erase(entries.begin());
            const auto index_bytes = fs::file_size(e.index,ec); if(ec) return false;
            fs::remove(e.index,ec); if (ec) return false;
            fs::remove(e.data,ec); if (ec) return false;
            total -= e.bytes + index_bytes; catalogue -= index_bytes;
        }
        if (!room()) return false;
        static std::atomic<uint64_t> serial{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
#ifndef _WIN32
        const auto pid = ::getpid();
#else
        const int pid = 0;
#endif
        const std::string name = "p"+std::to_string(pid)+"-"+std::to_string(stamp)+"-"+std::to_string(++serial);
        const auto temp = fs::path(config_.directory)/(name+".part");
        const auto data = fs::path(config_.directory)/(name+".bin");
        const auto index = fs::path(config_.directory)/(name+".idx");
        const auto index_temp = fs::path(config_.directory)/(name+".meta");
        std::string error;
        bool written = false;
        try { written = job.writer(temp.string(),error); } catch (...) {}
        uint64_t bytes = 0, hash = 0;
        auto cleanup = [&] { std::error_code ignored; fs::remove(temp,ignored); fs::remove(data,ignored); fs::remove(index_temp,ignored); };
        if (!written || !hash_file(temp,bytes,hash) || bytes > job.expected || !sync_file(temp)) { cleanup(); return false; }
        // The callback cannot consume the free-space reserve or exceed the admitted quota.
        const auto space = fs::space(config_.directory,ec);
        if (ec || space.available < config_.min_free_bytes || job.metadata+56 > space.available-config_.min_free_bytes ||
            bytes+job.metadata+56 > config_.disk_bytes-total) { cleanup(); return false; }
        fs::rename(temp,data,ec); if (ec) { cleanup(); return false; }
        std::vector<uint8_t> meta;
        u64(meta,0x3158465054415253ull); u64(meta,bytes); u64(meta,hash); u64(meta,job.identity.size());
        const auto longest=std::max_element(job.keys.begin(),job.keys.end(),[](const auto& a,const auto& b){return a.size()<b.size();});
        bool compact=true;
        for(const auto& key:job.keys) if(!std::equal(key.begin(),key.end(),longest->begin())) compact=false;
        u64(meta,job.keys.size() | (compact ? uint64_t(1)<<63 : 0));
        meta.insert(meta.end(),job.identity.begin(),job.identity.end());
        if(compact) {
            u64(meta,longest->size()); for(auto cell:*longest)u64(meta,cell);
            for(const auto& key:job.keys)u64(meta,key.size());
        } else for (const auto& key : job.keys) { u64(meta,key.size()); for (auto cell:key) u64(meta,cell); }
        u64(meta,hash_bytes(meta.data(),meta.size()));
        { std::ofstream out(index_temp,std::ios::binary); out.write(reinterpret_cast<const char*>(meta.data()),meta.size()); out.close(); if (!out) { cleanup(); return false; } }
        if (!sync_file(index_temp)) { cleanup(); return false; }
        fs::rename(index_temp,index,ec); if (ec) { cleanup(); return false; }
        if (!sync_file(config_.directory)) return false;
        if(job.capture) {
            std::lock_guard<std::mutex> guard(capture_mu_);
            auto& value=*job.capture;
            value.data=data.string(); value.index=index.string();
            value.data_stamp=file_stamp(value.data); value.index_stamp=file_stamp(value.index);
            value.committed=!value.data_stamp.empty() && !value.index_stamp.empty();
            if(value.committed) for(auto& slot:captures_) if(slot.pending.lock()==job.capture) slot.committed=job.capture;
        }
        ++stats_.writes; stats_.saved_bytes += bytes; return true;
    }
    void run() {
        while (true) {
            Job job;
            { std::unique_lock<std::mutex> g(queue_mu_); cv_.wait(g,[this]{return stopping_ || !queue_.empty();});
              if (queue_.empty()) break;
              job = std::move(queue_.front()); queue_.pop_front(); active_ = true; }
            const auto start = std::chrono::steady_clock::now();
            bool ok = false; try { ok = save(job); } catch (...) {}
            if (!ok) {
                ++stats_.failures;
                if(job.capture) { std::lock_guard<std::mutex> guard(capture_mu_); job.capture->invalid=true; }
            }
            stats_.save_us += micros(start); report(ok ? "saved" : "save-miss");
            { std::lock_guard<std::mutex> g(queue_mu_); job.writer = nullptr; std::vector<std::vector<uint64_t>>().swap(job.keys); std::string().swap(job.identity);
              job.reservation.reset(); job.extra.reset(); job.capture.reset(); active_=false; if(queue_.empty()) drained_.notify_all(); }
        }
    }
};
} // namespace strata::core::prefix
