#pragma once

#include "ds41_pipeline.hpp"

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_memory.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::v41 {

/// DS41 host images for Strata-style --serve conversation caching (RAM budget + optional disk tier).
struct Ds41ConvImage {
    std::vector<int32_t> ids;
    std::vector<uint8_t> stage0;
    std::vector<std::vector<uint8_t>> later_stages;
    std::vector<strata::core::ConversationCheckpoint> checkpoints;
    int64_t snap_pos = -1;
    std::vector<float> snap_logits;
    size_t bytes() const;
};

class Ds41ConversationServe {
public:
    Ds41ConversationServe(int64_t cache_mib, int slots, int64_t min_tokens, int prompt_cache_every,
                          const std::string & disk_root);

    bool enabled() const { return cache_.enabled(); }
    bool wants(int64_t tokens) const { return cache_.wants(tokens); }
    size_t parked() const { return cache_.size(); }

    /// Park the live session before switching to an unrelated request.
    bool park_live(Ds41Pipeline & pipe, const std::vector<int> & live_ids, int snap_len, int64_t positions,
                   const std::vector<float> & snap_logits, std::string & err);

    /// Restore the best matching parked conversation; returns tokens restored (0 = miss).
    int try_restore(Ds41Pipeline & pipe, const std::vector<int> & prompt, std::vector<int> & live_out,
                    int & snap_len_out, std::vector<float> & snap_logits_out, std::string & err);

    void drop_all(Ds41Pipeline & pipe);

    /// After prefill, optionally append a prompt-cache checkpoint.
    void maybe_checkpoint(Ds41Pipeline & pipe, const std::vector<int> & ids, int64_t token_pos);

private:
    strata::core::ConversationCache cache_;
    int prompt_every_ = 0;
    std::string disk_root_;

    bool capture_image(Ds41Pipeline & pipe, int64_t positions, Ds41ConvImage & out, std::string & err) const;
    bool restore_image(Ds41Pipeline & pipe, const Ds41ConvImage & in, std::string & err) const;
    strata::core::SavedConversation pack(const Ds41ConvImage & img) const;
    bool unpack(const strata::core::SavedConversation & saved, Ds41ConvImage & out) const;
};

}  // namespace strata::v41
