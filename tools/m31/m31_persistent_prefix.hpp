// M31's persistent adapter owns only F16 K/V/MSA-index rows and vision identity.
// It does not save live graph pointers, weights, expert residency or samplers.
#pragma once
#include "m31_engine.hpp"
#include "m31_vision.hpp"
#include "m31_prefix_codec.hpp"
#include "strata/core/persistent_prefix.hpp"
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>

namespace strata::m31 {
class PersistentPrefix {
    using Store = core::prefix::Store;
    struct Image {
        std::vector<int32_t> tokens;
        MixedInput mixed;
        Engine::PrefixCheckpointPtr checkpoint;
        uint64_t bytes() const {
            return 48 + tokens.size() * 8 + mixed.embeddings.size() * 4 + checkpoint->bytes();
        }
    };
    core::prefix::Config config_;
    Store store_;
    std::string identity_;
    uint64_t ram_budget_ = 0, ram_bytes_ = 0;
    std::deque<std::shared_ptr<const Image>> hot_;
    void report() const {
        const auto & s = store_.stats();
        std::cerr << "m31 prefix_cache disk_hits=" << s.hits.load() << " disk_misses=" << s.misses.load()
                  << " disk_writes=" << s.writes.load() << " failures=" << s.failures.load()
                  << " dropped=" << s.dropped.load() << " disk_read_bytes=" << s.restored_bytes.load()
                  << " disk_written_bytes=" << s.saved_bytes.load() << " disk_restore_us=" << s.restore_us.load()
                  << " disk_save_us=" << s.save_us.load() << " ram_bytes=" << ram_bytes_ << '\n';
    }
    static std::vector<uint64_t> cells(const std::vector<int32_t> & ids, const MixedInput & mixed, int width) {
        std::vector<uint64_t> out; out.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            uint64_t cell = uint32_t(ids[i]);
            if (ids[i] == -1 && i < mixed.visual_rows.size()) {
                const int row = mixed.visual_rows[i];
                if (row >= 0 && row < mixed.visual_count) {
                    // The index hash is only a filter; restore also compares
                    // every actual F32 byte before any live state is touched.
                    cell = 1469598103934665603ull;
                    const auto * bytes = reinterpret_cast<const uint8_t *>(mixed.embeddings.data() + size_t(row) * width);
                    for (size_t j = 0; j < size_t(width) * 4; ++j) cell = (cell ^ bytes[j]) * 1099511628211ull;
                    cell |= uint64_t(1) << 63;
                }
            }
            out.push_back(cell);
        }
        return out;
    }
    static int common(const Image & image, const std::vector<int32_t> & ids, const MixedInput & mixed, int width) {
        size_t n = 0, limit = std::min(image.tokens.size(), ids.size() - 1);
        while (n < limit && image.tokens[n] == ids[n]) ++n;
        int result = int(n / 256 * 256);
        if (result) result = reusable_visual_prefix(result, ids, image.mixed, mixed, width);
        return result;
    }
    static bool write(const std::string & path, const Image & image) {
        std::ofstream out(path, std::ios::binary);
        if (!prefix_codec::write(out, image.tokens, image.mixed, image.checkpoint->payload())) return false;
        out.close(); return bool(out);
    }
    static std::shared_ptr<Image> read(const std::string & path, Engine & engine, uint64_t maximum,
                                       const std::function<bool()> & cancelled) {
        std::error_code ec; const auto file_bytes = std::filesystem::file_size(path, ec);
        if (ec) return {};
        std::ifstream in(path, std::ios::binary);
        auto payload = prefix_codec::read(in, file_bytes, maximum, engine.geometry(), cancelled);
        if (!payload) return {};
        auto image = std::make_shared<Image>();
        image->tokens = std::move(payload->tokens); image->mixed = std::move(payload->mixed);
        std::string error;
        if (!engine.import_prefix(int64_t(image->tokens.size()), std::move(payload->kv), image->checkpoint, error)) return {};
        return image;
    }

public:
    PersistentPrefix(const std::string & model, const std::string & settings, const std::string & runtime)
        : config_(core::prefix::Config::from_env()), store_(config_) {
        if (!store_.enabled()) return;
        if (runtime.empty()) {
            std::cerr << "m31 prefix_cache enabled=0 reason=runtime_identity_unavailable\n";
            return;
        }
        // Preserve the existing last-live conversation cache. Extra RAM snapshots
        // are allocated only under an explicit, separately bounded budget.
        if (const char * value = std::getenv("STRATA_PREFIX_CACHE_RAM_BYTES")) {
            char * end = nullptr; errno = 0;
            const auto bytes = std::strtoull(value, &end, 10);
            if (!errno && *value && std::string(value).find_first_not_of("0123456789") == std::string::npos && end && !*end)
                ram_budget_ = bytes;
        }
        identity_ = core::prefix::model_identity("m31-f16-kv-index-v1", settings + " runtime=" + runtime, gguf_split_paths(model));
    }
    bool enabled() const { return store_.enabled() && !identity_.empty(); }
    int restore(Engine & engine, const std::vector<int32_t> & ids, const MixedInput & mixed,
                std::vector<int32_t> & history, MixedInput & history_mixed, int current,
                const std::function<bool()> & cancelled) {
      try {
        if (!enabled() || ids.size() < 257 || (cancelled && cancelled())) return current;
        // A last-live hit already covering every reusable boundary must not wait
        // for a background disk writer's directory lease.
        if (size_t(current) >= (ids.size() - 1) / 256 * 256) return current;
        const auto start = std::chrono::steady_clock::now();
        std::shared_ptr<const Image> best;
        int reused = current;
        bool disk = false;
        for (auto i = hot_.begin(); i != hot_.end(); ++i) {
            const int n = common(**i, ids, mixed, int(engine.geometry().embd));
            if (n > reused) { reused = n; best = *i; }
        }
        if (!best) {
            const auto key = cells(ids, mixed, int(engine.geometry().embd));
            auto file = store_.find_file(identity_, key, ids.size() - 1, cancelled, size_t(current));
            if (file) {
                std::error_code ec;
                const auto file_bytes = std::filesystem::file_size(file->path, ec);
                auto reservation = ec ? std::shared_ptr<void>{} : store_.reserve(file_bytes);
                if (!reservation) return current;
                auto image = read(file->path, engine, config_.queue_bytes, cancelled);
                // The indexed key must describe this payload, not another one.
                if (image && file->cells.size() <= image->tokens.size()) {
                    const auto actual = cells(image->tokens, image->mixed, int(engine.geometry().embd));
                    if (std::equal(file->cells.begin(), file->cells.end(), actual.begin())) {
                        const int n = std::min(int(file->cells.size()), common(*image, ids, mixed, int(engine.geometry().embd)));
                        if (n > reused) { reused = n; best = std::move(image); disk = true; }
                    }
                }
            }
        }
        if (!best || (cancelled && cancelled())) { report(); return current; }
        // Prepare ownership before uploading; an allocation failure is a miss
        // and must leave the current conversation untouched.
        std::vector<int32_t> restored_history(best->tokens.begin(), best->tokens.begin() + reused);
        MixedInput restored_mixed = best->mixed;
        std::string error;
        if (!engine.restore_prefix(best->checkpoint, error) || !engine.rewind(reused, error)) {
            history.clear(); history_mixed = {}; engine.reset(); return 0;
        }
        history.swap(restored_history);
        history_mixed = std::move(restored_mixed);
        auto hot = std::find(hot_.begin(), hot_.end(), best);
        if (hot != hot_.end()) { auto used = *hot; hot_.erase(hot); hot_.push_back(std::move(used)); }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        std::cerr << "m31 prefix_cache tier=" << (disk ? "disk" : "ram") << " restored_tokens=" << reused
                  << " restore_us=" << us << " bytes=" << best->checkpoint->bytes() << '\n';
        report();
        return reused;
      } catch (const std::exception &) { return current; }
    }
    void save(Engine & engine, const std::vector<int32_t> & history, const MixedInput & mixed,
              const std::function<bool()> & cancelled) {
      try {
        if (!enabled() || history.size() < 256 || engine.position() != int64_t(history.size()) ||
            (cancelled && cancelled())) return;
        const auto start = std::chrono::steady_clock::now();
        const uint64_t estimate = 48 + history.size() * 8 + mixed.embeddings.size() * 4 + engine.prefix_checkpoint_bytes();
        // Reserve before any immutable snapshot or index copies are allocated.
        // Indexes are limited below to at most 8 MiB per record.
        auto reservation = store_.reserve(estimate + (uint64_t(8) << 20) + history.size() * 8);
        if (!reservation) return;
        auto image = std::make_shared<Image>();
        std::string error;
        if (!engine.checkpoint_prefix(size_t(std::min(config_.queue_bytes, config_.disk_bytes)), image->checkpoint, error)) return;
        image->tokens = history;
        image->mixed.visual_rows = mixed.visual_rows;
        image->mixed.embeddings = mixed.embeddings;
        image->mixed.visual_count = mixed.visual_count;
        image->mixed.visual_rows.resize(history.size(), -1);
        if (cancelled && cancelled()) return;
        // Each index key names a prefix that M31 can rewind to exactly. All keys
        // share one immutable full-state payload, avoiding duplicate KV files.
        const auto all = cells(history, mixed, int(engine.geometry().embd));
        std::vector<std::vector<uint64_t>> keys;
        const size_t key_limit = std::max<size_t>(1, std::min<size_t>(255, (uint64_t(8) << 20) / (history.size() * 8)));
        const size_t stride = ((history.size() / 256 + key_limit - 1) / key_limit) * 256;
        for (size_t n = stride; n <= history.size(); n += stride) keys.emplace_back(all.begin(), all.begin() + n);
        const size_t last = history.size() / 256 * 256;
        if (keys.empty() || keys.back().size() != last) keys.emplace_back(all.begin(), all.begin() + last);
        const bool queued = store_.enqueue_file_keys_reserved(identity_, std::move(keys), image->bytes(),
            [image](const std::string & path, std::string &) { return write(path, *image); }, std::move(reservation));
        if (ram_budget_ && image->bytes() <= ram_budget_) {
            for (auto it = hot_.begin(); it != hot_.end();) {
                if ((*it)->tokens == image->tokens && (*it)->mixed.visual_rows == image->mixed.visual_rows &&
                    (*it)->mixed.embeddings.size() == image->mixed.embeddings.size() &&
                    ((*it)->mixed.embeddings.empty() || !std::memcmp((*it)->mixed.embeddings.data(),
                        image->mixed.embeddings.data(), image->mixed.embeddings.size() * sizeof(float)))) {
                    ram_bytes_ -= (*it)->bytes(); it = hot_.erase(it);
                } else ++it;
            }
            while (!hot_.empty() && image->bytes() > ram_budget_ - ram_bytes_) {
                ram_bytes_ -= hot_.front()->bytes(); hot_.pop_front();
            }
            ram_bytes_ += image->bytes(); hot_.push_back(image);
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        std::cerr << "m31 prefix_cache save_queued=" << int(queued) << " cached_tokens=" << history.size()
                  << " snapshot_us=" << us << " bytes=" << image->bytes() << '\n';
        report();
      } catch (const std::exception &) {
        // Host memory, storage and optional-cache errors cannot fail a request.
      }
    }
};
} // namespace strata::m31
