// tools/v41/engine/ds41_conv_pool_dense.hpp - the conversation pool's layout and device pointers from a Ds41Dense
// (Ds41Dense::pool_regions).  Header-only so the pool itself stays free of ggml.
//
//   Ds41PoolLayout lay; Ds41PoolTargets snap, live;
//   ds41_pool_from_dense(dense, ctx_max, lay, snap, live);
//   pool.init(lay, cfg, err);
//   ... after a turn's snapshot():            pool.save(snap, hist, ids, snap_len, logits, n_vocab, err);
//   ... a request continuing a pool record:   pool.restore(m.rec, m.boundary, from, snap, rr, err); pool.wait(rr);
//                                             then the engine re-arms its snapshot at rr.pos and restore()s it.
#pragma once

#include "ds41_conv_pool.hpp"
#include "ds41_dense.hpp"

#include <algorithm>

namespace strata::v41 {

inline bool ds41_pool_from_dense(const Ds41Dense & d, int64_t ctx_max, Ds41PoolLayout & lay, Ds41PoolTargets & snap,
                                 Ds41PoolTargets & live) {
    std::vector<Ds41Dense::PoolRegion> st, rs, rl;
    d.pool_regions(true, st, rs);
    d.pool_regions(false, st, rl);   // the same streams again; only the regions differ
    lay = {};
    snap = {};
    live = {};
    int64_t maxp = ctx_max;
    for (const auto & s : st) {
        if (!s.data || !s.row_bytes || s.ratio < 1) return false;
        lay.streams.push_back({ s.name, s.row_bytes, s.ratio, s.rows });
        maxp = std::min<int64_t>(maxp, s.rows * s.ratio);
        snap.stream.push_back(s.data);
        live.stream.push_back(s.data);
    }
    if (rs.size() != rl.size()) return false;
    for (size_t i = 0; i < rs.size(); ++i) {
        if (!rs[i].data || !rl[i].data || rs[i].bytes != rl[i].bytes) return false;
        lay.regions.push_back({ rs[i].name, rs[i].bytes });
        snap.region.push_back(rs[i].data);
        live.region.push_back(rl[i].data);
    }
    lay.max_positions = maxp;
    snap.layout_fp = live.layout_fp = lay.fingerprint();
    return true;
}

}  // namespace strata::v41
