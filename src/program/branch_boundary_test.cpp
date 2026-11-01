#include "strata/program/branch_boundary.hpp"
#include "strata/core/conversation_cache.hpp"

#include <climits>
#include <cstdio>

using namespace strata::core;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
SavedConversation saved(const std::vector<int32_t>& ids,
        const std::vector<ConversationImageKey>& images = {}, bool cvec = true) {
    SavedConversation out;
    out.live.ids = ids;
    out.live.imgs = images;
    out.cvec = cvec;
    return out;
}
}

int main() {
    using strata::program::branch_checkpoint_boundary;
    std::vector<int32_t> old(41000, 7);
    std::vector<int64_t> now(old.begin(), old.end());
    now[40629] = 9;
    const std::vector<ConversationImageKey> no_images;
    check(conversation_common_prefix(old, no_images, now, no_images) == 40629,
          "learn exact divergence inside a user message");
    check(conversation_common_prefix({}, no_images, now, no_images) == 0, "no old state");
    check(conversation_common_prefix(old, no_images, std::vector<int64_t>{}, no_images) == 0, "empty request");
    check(conversation_common_prefix(old, no_images, std::vector<int64_t>{7}, no_images) == 0,
          "last prompt token belongs to verification");
    check(conversation_common_prefix(old, no_images, std::vector<int64_t>{7,7}, no_images) == 1,
          "a shortened prefix stops before the last prompt token");
    check(conversation_common_prefix(std::vector<int32_t>{7,7}, no_images, now, no_images) == 2,
          "an append does not invent old cells");

    const std::vector<ConversationImageKey> images = {{100, 11}, {20000, 22}, {40000, 33}};
    check(conversation_common_prefix(old, images, now, images) == 40629, "same image keys preserve the text LCP");
    auto changed = images;
    changed[1].hash = 99;
    check(conversation_common_prefix(old, images, now, changed) == 20000,
          "changed image/grid clamps to its FIRST cell, not pad-token LCP");
    changed = images;
    changed[1].start = 20001;
    check(conversation_common_prefix(old, images, now, changed) == 20000, "moved image clamps to earlier start");
    check(conversation_common_prefix(old, images, now, no_images) == 100, "removed image clamps");
    check(conversation_common_prefix(old, no_images, now, images) == 100, "added image clamps");
    changed = images;
    changed.push_back({40700, 44});
    check(conversation_common_prefix(old, images, now, changed) == 40629, "image AFTER text divergence is irrelevant");
    changed = {{-1,99}};
    check(conversation_common_prefix(old, images, now, changed) == 0, "bad negative image start cannot yield a boundary");

    const int64_t resume = 37773;
    check(branch_checkpoint_boundary(40629, resume, (int64_t) now.size()) == 40629, "useful branch point");
    check(branch_checkpoint_boundary(40629, 40629, (int64_t) now.size()) == -1, "no extra read: do not checkpoint again");
    check(branch_checkpoint_boundary(40629, 40600, (int64_t) now.size()) == -1, "tiny replay does not pay for a snapshot");
    check(branch_checkpoint_boundary(8192, 8128, 9000) == 8192, "minimum prefix and minimum replay are inclusive");
    check(branch_checkpoint_boundary(8191, 0, 9000) == -1, "small shared prefix");
    check(branch_checkpoint_boundary(9000, 0, 9001) == -1, "no boundary at final verification token");
    check(branch_checkpoint_boundary(9001, 0, 9001) == -1, "no boundary outside request");
    check(branch_checkpoint_boundary(9000, -1, 10000) == -1, "negative resume");
    check(branch_checkpoint_boundary(9000, 0, 0) == -1, "invalid prompt length");
    check(branch_checkpoint_boundary(9000, 0, 10000, -1) == -1, "invalid prefix limit");
    check(branch_checkpoint_boundary(9000, 0, 10000, 0, 0) == -1, "invalid replay limit");
    check(branch_checkpoint_boundary(INT64_MAX - 1, INT64_MAX - 100, INT64_MAX) == -1,
          "extreme boundary rejected without overflow");
    check(branch_checkpoint_boundary(INT64_MAX - 2, INT64_MAX - 100, INT64_MAX) == INT64_MAX - 2,
          "extreme valid lengths avoid arithmetic overflow");

    ConversationCache cache(1 << 20, 4);
    check(cache.put(saved(old, images)), "park a previous branch");
    check(cache.common_prefix(now, images, true) == 40629, "A/B/A finds the parked semantic divergence");
    check(cache.common_prefix(now, images, false) == 0, "steering identity gates learned point");
    check(cache.common_prefix(now, images, true, 40629) == 0, "already reusable boundary gives no extra hint");
    check(cache.common_prefix(now, images, true, 37773) == 40629, "coarse reusable point can be refined");
    const size_t before_bytes = cache.bytes(), before_count = cache.size(), before_evictions = cache.evictions();
    check(cache.best(now, images, true).tokens == 0, "a hint is NOT a restorable running-state checkpoint");
    check(cache.bytes() == before_bytes && cache.size() == before_count && cache.evictions() == before_evictions,
          "learning only reads cache metadata; no budget or ownership mutations");
    auto first = cache.take(0);
    // A real GPU run must replay to this point and capture all running state.
    // This CPU policy test models that capture with a complete checkpoint token prefix.
    first.live.ids.assign(now.begin(), now.end());
    ConversationCheckpoint branch;
    branch.ids.assign(now.begin(), now.begin() + 40629);
    branch.imgs = images;
    first.checkpoints.push_back(std::move(branch));
    check(cache.put(std::move(first)), "park the newly captured branch");
    now[40629] = 10;
    check(cache.best(now, images, true).tokens == 40629, "second suffix edit reuses the learned checkpoint");
    now[40001] = 12;
    check(cache.common_prefix(now, images, true) == 40001, "moving divergence learns an earlier point");
    check(cache.best(now, images, true).tokens == 0, "a later learned checkpoint never authorizes an earlier rewind");
    changed = images;
    changed[2].hash = 77;
    check(cache.common_prefix(now, changed, true) == 40000, "new image moves learned point to image start");
    check(cache.best(now, changed, true).tokens == 0, "image identity still gates real snapshot reuse");
    cache.clear();
    check(cache.common_prefix(now, images, true) == 0, "dropped conversations cannot leave branch hints");
    std::printf("branch_boundary_test: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
