#!/usr/bin/env python3
"""tools/v41/sim/pf_sim.py - offline precision check for the decode prefetcher (measurement only).

The GPU profile of 2026-10-23 (V3, serve flags, 256-token decode) showed the expert tier is
PCIe-bound: 32.5 GiB host->device per 64 tokens, of which 28 GiB is prefetch DMA (2874 of 3328
large copies) and only 46% of the prefetched experts were routed at all.  The deployed predictor
is Ds41Dense::predict (a per-layer projection of the hidden state, top16) - it cannot be replayed
from a routing trace, so this script measures the cheap alternatives that CAN be:

    temporal  the same layer's routing one pass earlier (a per-(layer, expert) recency table)
    spatial   the previous layer's routing in the same pass (already computed when layer l starts)
    freq      the most-routed experts of that layer over the trace so far (a frequency prior)

for each, with the engine's own prefetch budget (pf_b experts per layer), reporting:
    precision  share of issued slots that were routed (the engine measured 45.6% on the live run)
    coverage   share of the layer's miss experts that were prefetched
    mib/token  bytes moved, at the blob sizes the engine actually copies

Usage: python3 pf_sim.py TRACE.bin [TRACE.bin ...] [--k 1] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from route_sim import load, HIT

HIT_CLS = HIT


def calls_by_pass(passes):
    """[(pass index, layer, routed ids, miss mask)] for single-row (decode) calls only."""
    out = []
    for pi, (v, calls) in enumerate(passes):
        for (layer, nt, u) in calls:
            if nt != 1:
                continue
            ids = u[:, 0].astype(int)
            cls = u[:, 2]
            out.append((pi, layer, ids, cls != HIT_CLS))
    return out


def evaluate(trace_path, k=1):
    L, X, snap, passes = load(trace_path)
    seq = calls_by_pass(passes)
    if not seq:
        return None
    # per-(pass, layer) -> dict expert -> is_miss
    by_pl = defaultdict(dict)
    for (pi, layer, ids, miss) in seq:
        d = by_pl[(pi, layer)]
        for e, m in zip(ids, miss):
            d[int(e)] = bool(m)

    freq = defaultdict(Counter)          # layer -> expert -> times routed (prefix only)
    stats = {p: {'issued': 0, 'useful': 0, 'misses': 0, 'covered': 0, 'passes': 0}
             for p in ('temporal', 'spatial', 'freq', 'oracle')}
    prev_routed = defaultdict(set)       # layer -> routed set of the previous pass
    cur_pass = {}                        # layer -> routed set of the pass being walked
    prev_pass = None

    for (pi, layer, ids, miss) in seq:
        if pi != prev_pass:
            cur_pass = {}
            prev_pass = pi
            stats['oracle']['passes'] += 1
        cur = by_pl[(pi, layer)]
        resident = {e for e, m in cur.items() if not m}
        routed = set(cur)
        nmiss = sum(1 for m in cur.values() if m)
        for p in stats:
            stats[p]['misses'] += nmiss

        cands = {}
        # temporal: same layer, previous pass
        cands['temporal'] = [e for e in prev_routed.get(layer, ()) if e not in resident]
        # spatial: the previous layer's routing in this same pass (already on the device)
        cands['spatial'] = [e for e in cur_pass.get(layer - 1, ()) if e not in resident]
        # freq: most-routed so far, excluding residents
        cands['freq'] = [e for e, _ in freq[layer].most_common() if e not in resident]
        # oracle: the layer's actual misses (an upper bound on any predictor)
        cands['oracle'] = [e for e, m in cur.items() if m]

        for p, c in cands.items():
            pick = c[:k]
            stats[p]['issued'] += len(pick)
            stats[p]['useful'] += sum(1 for e in pick if e in routed)
            stats[p]['covered'] += sum(1 for e in pick if e in routed and cur[e])

        # age the state
        prev_routed[layer] = routed
        cur_pass[layer] = routed
        for e in routed:
            freq[layer][e] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--k', type=int, default=1, help='prefetch budget per layer (pf_b ~ 1.4)')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    blob_mib = 9.8   # the 4-16 MiB copies the profile showed: mean expert blob
    out = {}
    for p in a.traces:
        r = evaluate(p, a.k)
        if r is None:
            print(f'{p}: no decode calls')
            continue
        print(f'\n=== {p}  (budget {a.k}/layer) ===')
        for name in ('temporal', 'spatial', 'freq', 'oracle'):
            s = r[name]
            prec = 100.0 * s['useful'] / s['issued'] if s['issued'] else 0.0
            cov = 100.0 * s['covered'] / s['misses'] if s['misses'] else 0.0
            print(f"  {name:9s} issued {s['issued']:6d}  useful {s['useful']:6d}  "
                  f"precision {prec:5.1f}%  miss coverage {cov:5.1f}%  "
                  f"{s['issued'] * blob_mib / max(1, s['misses'] or 1):5.2f} MiB/miss")
        out[p] = r
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
