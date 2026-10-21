#pragma once

#include "ds41_pipeline.hpp"

#include <string>

namespace strata::v41 {

inline int ds41_stage_for_layer(const Ds41Pipeline & pipe, int global_layer) {
    for (int s = 0; s < pipe.n_stages(); ++s)
        if (global_layer >= (int) pipe.layer_lo(s) && global_layer < (int) pipe.layer_hi(s)) return s;
    return std::max(0, pipe.n_stages() - 1);
}

inline int ds41_split_at(const Ds41Pipeline & pipe, int n_layer) {
    return pipe.enabled() ? pipe.split_layer : n_layer;
}

inline Ds41Dense & ds41_dense_for(Ds41Pipeline & pipe, int global_layer) {
    if (!pipe.enabled()) return pipe.dense(0);
    return pipe.dense(ds41_stage_for_layer(pipe, global_layer));
}

inline Ds41MoeTier & ds41_tier_for(Ds41Pipeline & pipe, int global_layer) {
    if (!pipe.enabled()) return pipe.tier(0);
    return pipe.tier(ds41_stage_for_layer(pipe, global_layer));
}

inline Ds41Dense & ds41_head_dense(Ds41Pipeline & pipe) {
    if (pipe.head_offload()) return *pipe.head->dense;
    return pipe.enabled() ? pipe.dense(pipe.n_stages() - 1) : pipe.dense(0);
}

inline bool ds41_head_logits(Ds41Pipeline & pipe, int row0, int nrows, float * out, std::string & err) {
    return pipe.head_logits(row0, nrows, out, err);
}

inline bool ds41_sync_stages(Ds41Pipeline & pipe, int from_stage, int to_stage, int col0, int n, std::string & err) {
    if (to_stage <= from_stage) return true;
    for (int s = from_stage; s < to_stage; ++s)
        if (!pipe.sync_handoff_between(s, s + 1, col0, n, err)) return false;
    return true;
}

inline bool ds41_begin_stage_continued(Ds41Pipeline & pipe, int stage, const int * tids, int n, int pos0, bool layer_major_batch,
                                       std::string & err) {
    if (!pipe.enabled()) return true;
    Ds41Dense & d = pipe.dense(stage);
    const bool ok = layer_major_batch ? d.begin_batch_continued(tids, n, pos0) : d.begin_continued(tids, n, pos0);
    if (!ok) { err = d.last_error(); return false; }
    return true;
}

inline bool ds41_handoff_between_stages(Ds41Pipeline & pipe, int col0, int n, std::string & err) {
    return ds41_sync_stages(pipe, 0, 1, col0, n, err);
}

inline bool ds41_begin_stage1_continued(Ds41Pipeline & pipe, const int * tids, int n, int pos0, bool layer_major_batch,
                                        std::string & err) {
    return ds41_begin_stage_continued(pipe, 1, tids, n, pos0, layer_major_batch, err);
}

}  // namespace strata::v41
