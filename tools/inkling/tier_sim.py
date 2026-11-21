#!/usr/bin/env python3
"""tier_sim.py - miss-rate model for xolotl-inkling expert tiering (census-hot set + per-layer LRU cache).

Reads an `xolotl-inkling --route-log` file (XQRT, see route_skew.py) and replays the decode visits (T <= --max-t)
through a model of the planned tier:
  * per physical layer a static HOT set (resident in VRAM), chosen globally by census count per byte
    (layers whose experts are big - the parent's L20-23 at 15.9 MB - get fewer hot experts);
  * per physical layer an LRU cache of C slots for cold experts; a visit's picks must all be resident, a miss
    evicts the least recently used slot not picked in this visit and costs one expert copy over PCIe.
The census comes from --census: "prompt" (the prompt chunks of the same log; what an offline calibration gives),
"half" (the first half of the decode visits, replay the second half), or "oracle" (the replayed visits themselves).

For each expert-VRAM budget and cache size it prints hot fraction, misses per token, MB per token and an estimated
tok/s = 1 / (t_base + MB/BW + stage overhead). Per-expert bytes per physical layer come from --bytes (JSON written by
the gguf scan: {"parent": {"0": bytes, ...}, "xq": {...}}).
"""
import argparse, json, sys
from collections import OrderedDict, defaultdict
from route_skew import read_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bytes", required=True)
    ap.add_argument("--model", default="parent", help="key in --bytes (parent | xq)")
    ap.add_argument("--census", default="prompt", choices=["prompt", "half", "oracle"])
    ap.add_argument("--census-from", default=None, help="take the census from the decode visits of another route log")
    ap.add_argument("--budgets", default="12,14,16,17,18", help="GB of VRAM for experts (hot + cache)")
    ap.add_argument("--caches", default="0,8,16,32", help="LRU slots per physical layer")
    ap.add_argument("--max-t", type=int, default=8, help="replay visits with T <= this (decode / verify)")
    ap.add_argument("--base-tps", type=float, default=89.0, help="all-in-VRAM decode tok/s of this model")
    ap.add_argument("--bw", type=float, default=20.0, help="host->VRAM GB/s for the bulk copy")
    ap.add_argument("--stage-us", type=float, default=8.0, help="stage-op overhead per visit with a cold layer (us)")
    ap.add_argument("--n-expert", type=int, default=128)
    ap.add_argument("--rebal", default="0", help="adaptive hot set: re-rank every R tokens by decayed counts (0 = static); list ok")
    ap.add_argument("--decay", type=float, default=0.98, help="per-token decay of the adaptive counts")
    ap.add_argument("--max-swap", type=int, default=8, help="max hot-set swaps per layer per rebalance")
    a = ap.parse_args()

    eb = {int(k): float(v) for k, v in json.load(open(a.bytes))[a.model].items()}
    total = sum(eb.values()) * a.n_expert
    recs = list(read_records(a.log))
    dec = [r for r in recs if r[3] <= a.max_t]
    pro = [r for r in recs if r[3] > a.max_t]
    if not dec:
        sys.exit("no decode visits (T <= --max-t) in the log")
    # tokens = decode positions: count distinct (p0, T) of slot 0
    toks = sum(r[3] for r in dec if r[0] == 0)

    if a.census == "prompt":
        cen_src, replay = pro, dec
    elif a.census == "half":
        cut = len(dec) // 2
        cen_src, replay = dec[:cut], dec[cut:]
        toks = sum(r[3] for r in replay if r[0] == 0)
    else:
        cen_src, replay = dec, dec
    if a.census_from:
        cen_src = [r for r in read_records(a.census_from) if r[3] <= a.max_t]
        replay = dec; toks = sum(r[3] for r in dec if r[0] == 0)
    if not cen_src:
        sys.exit("empty census source")
    cnt = defaultdict(int)
    for slot, phys, pas, T, k, p0, ids, w in cen_src:
        for e in ids:
            cnt[(phys, e)] += 1
    layers = sorted(eb)
    order = sorted(((cnt.get((l, e), 0) / eb[l], l, e) for l in layers for e in range(a.n_expert)), reverse=True)

    print(f"{a.model}: experts {total / 1e9:.2f} GB | census {a.census_from or a.census} ({len(cen_src)} visits) | replay "
          f"{len(replay)} visits, {toks} tokens | base {a.base_tps} tok/s, {a.bw} GB/s")
    print(f"{'budget':>6} {'C':>3} {'R':>3} {'hot%':>5} {'cold GB':>7} {'miss/tok':>8} {'MB/tok':>7} {'est tok/s':>9} "
          f"{'swap/tok':>8}  worst layers (miss/tok)")
    for B in [float(x) for x in a.budgets.split(",")]:
        for C in [int(x) for x in a.caches.split(",")]:
            cache_bytes = sum(eb[l] * C for l in layers)
            room = B * 1e9 - cache_bytes
            if room <= 0:
                continue
            hot = set(); used = 0.0
            for _, l, e in order:
                if used + eb[l] > room:
                    continue
                hot.add((l, e)); used += eb[l]
            for R in [int(x) for x in a.rebal.split(",")]:
                r = replay_tier(a, layers, hot, cnt, len(cen_src), replay, C, R)
                miss, cold_visits, swaps = r
                mb = sum(miss[l] * eb[l] for l in layers) / 1e6 / toks
                mt = sum(miss.values()) / toks
                t = 1.0 / a.base_tps + mb / 1e3 / a.bw + cold_visits / toks * a.stage_us * 1e-6
                ws = " ".join(f"L{l}:{miss[l] / toks:.1f}" for l in sorted(layers, key=lambda l: -miss[l])[:3])
                print(f"{B:6.1f} {C:3d} {R:3d} {100 * len(hot) / (len(layers) * a.n_expert):5.1f} "
                      f"{(total - used) / 1e9:7.2f} {mt:8.2f} {mb:7.1f} {1 / t:9.1f} {swaps / toks:8.2f}  {ws}")


def replay_tier(a, layers, hot, cnt, n_cen, replay, C, R):
    """replay decode visits; R > 0: every R tokens swap up to --max-swap cold experts per layer whose decayed count beats
    the weakest hot one by 25 % (each swap = one copy, counted as a miss of that layer)."""
    hot_l = {l: set(e for (ll, e) in hot if ll == l) for l in layers}
    ema = {l: [0.0] * a.n_expert for l in layers}
    scale = len(layers) / max(1, n_cen)
    for (l, e), v in cnt.items():
        ema[l][e] = v * scale
    lru = {l: OrderedDict() for l in layers}
    miss = defaultdict(int); cold_visits = 0; swaps = 0; tok_i = 0
    for slot, phys, pas, T, k, p0, ids, w in replay:
        if slot == 0:
            tok_i += T
            if R:
                f = a.decay ** T
                for l in layers:
                    d = ema[l]
                    for e in range(a.n_expert): d[e] *= f
                if tok_i % R < T and tok_i >= R:
                    for l in layers:
                        d = ema[l]
                        hs = sorted(hot_l[l], key=lambda e: d[e])
                        cs = sorted((e for e in range(a.n_expert) if e not in hot_l[l]), key=lambda e: -d[e])
                        for i in range(min(a.max_swap, len(hs), len(cs))):
                            if d[cs[i]] <= d[hs[i]] * 1.25: break
                            hot_l[l].discard(hs[i]); hot_l[l].add(cs[i]); swaps += 1; miss[l] += 1
                            lru[l].pop(cs[i], None)
        if R:
            for e in ids: ema[phys][e] += 1.0
        picks = set(e for e in ids if e not in hot_l[phys])
        if not picks:
            continue
        cold_visits += 1
        c = lru[phys]
        for e in picks:
            if e in c:
                c.move_to_end(e); continue
            miss[phys] += 1
            if C == 0:
                continue
            if len(c) >= C:   # evict the LRU entry not picked in this visit
                for v in c:
                    if v not in picks:
                        del c[v]; break
                else:
                    continue   # every slot is picked this visit: stream it uncached
            c[e] = True
    return miss, cold_visits, swaps


if __name__ == "__main__":
    main()
