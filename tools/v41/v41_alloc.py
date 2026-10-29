#!/usr/bin/env python3
"""v41_alloc.py - mixed-tier allocation (RCO-lite) from v41_calib.py screening stats.

Each routed expert gets one tier (ptq1 / q2_0 / q3_k), or with --per-proj one tier for gate+up and one for down.
Cost of a choice = held-out output error ||f(x;W) - f(x;Q)||^2 (gate weight included, GPTQ) normalized by the
layer's held-out routed output energy, so every layer counts the same; --edge-weight boosts layers 0-1 and 37-39.
Budget = all-Q2_0 bytes of the experts covered (budget-neutral, V41_PLAN.md).  Solved by Lagrangian relaxation
(per-item argmin err + lam*bytes, bisection on lam), then a greedy pass spends what is left.
Guard rails: --max-ternary-frac per layer (default 0.5), --unseen-floor (tier floor for experts with < --min-eval
held-out tokens: their error is not measured, so they must not drift to ternary on no evidence).
  v41_alloc.py [--work DIR] [--layers 0-7] [--per-proj] [--out alloc.json]
"""
import argparse, glob, json, os, re
import numpy as np

# --- architecture table ---------------------------------------------------------------------------
# Everything the allocator needs to know about the model.  `ds4` is the original DeepSeek-V4.1-Flash
# geometry and its three grids.  `qwen4exp` is Qwen3.8-Flash-Next.  Tiers MUST be listed in increasing
# bytes: solve() promotes with sel+1 and demotes with sel-1.
#
# CORRECTED 2026-10-28: earlier this comment claimed the engine has "no PTQ1_0 kernel on that path"
# and therefore qwen4exp could never use ternary.  That is WRONG — the serving side fully supports it:
#   src/kernels/cuda/fork/fork_fmt.cuh:37  STRATA_GU_FMTS_FORK -> ... X(143)
#   src/kernels/cuda/fork/fork_fmt.cuh:40  STRATA_D_FMTS_FORK  -> ... X(143)
#   src/kernels/cuda/fork/fork_fmt.cuh:9   Fmt<143> { qk = 128, vec_dot_ptq1_0_q8_1 }
#   src/kernels/cpu/native_expert.cpp:29,222-234,333   CPU ptq1_0 path (incl. ptq1_0_rows_q8_0)
# PTQ1_0's block size is 128, which divides both n_embd 2560 (20) and n_ff 640 (5) — it is the ONLY
# ternary type that works on Flash-Next's down projection, which is exactly why V4 used it.
# What is missing is on the WRITE side: the vendored ggml enum stops at GGML_TYPE_COUNT 43, so
# llama-quantize cannot emit 143 (tools/v41/v41_formats.py:pack_ptq1_0 + v41_gguf.py can).  So
# "ptq1" joins the qwen4exp ladder when the mixed-layout writer lands, not before.
ARCH = {
    "ds4": dict(n_embd=5120, n_ff=2304, n_layers=40,
                tiers=["ptq1", "q2_0", "q3_k"],
                bpw={"ptq1": 1.75, "q2_0": 2.25, "q3_k": 3.4375}),
    "qwen4exp": dict(n_embd=2560, n_ff=640, n_layers=48,
                     # ptq1 is encoded by v41_formats, not the vendored ggml.
                     # gate/up may use it; edited down tensors may not (Q8_0
                     # and IQ4_NL are the only edit-preserving choices measured).
                     # Must stay in increasing bytes: the DP walks this list by index.
                     # q8_0 is the top rung so the DP can buy parent-identical gate/up
                     # for the few layers whose measured error justifies 4.9x ternary's
                     # bytes. It will not be chosen where the error gain is small.
                     #
                     # q2_0 is DELIBERATELY absent. Measured on this model (L16, real
                     # weights, ggml's own quantiser, verified against two ggml builds):
                     # q2_0 = 72.2% relative weight error at 2.25 bpw, while iq2_xs is
                     # 30.1% at 2.31. q2_0 is strictly dominated - the DP could only
                     # pick it to save 1% of bytes and lose 2.4x the accuracy.
                     tiers=["ptq1", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s",
                            "iq4_nl", "q8_0"],
                     # The down projection's row length is n_ff=640, and ggml's IQ2_*/IQ3_* types
                     # have block size 256 - 640 is not a multiple of 256, so those types CANNOT be
                     # written for down.  down's real ladder is Q2_0 (block 64) or IQ4_NL (block 32);
                     # anything else would be a tier the format cannot hold.  This is why the shipped
                     # third-party Flash-Next quants alternate exactly Q2_0 and IQ4_NL for down.
                     dn_tiers=["iq4_nl", "q8_0"],
                     pristine_dn_tiers=["q2_0", "iq4_nl", "q8_0"],
                     bpw={"ptq1": 1.75, "iq2_xxs": 2.0625, "q2_0": 2.25, "iq2_xs": 2.3125, "iq2_s": 2.5,
                          "iq3_xxs": 3.0625, "iq3_s": 3.4375, "iq4_nl": 4.5, "q8_0": 8.5}),
}
TIERS = list(ARCH["ds4"]["tiers"])
DN_TIERS = list(ARCH["ds4"]["tiers"])
BPW = dict(ARCH["ds4"]["bpw"])
GU, DN = 2 * 5120 * 2304, 5120 * 2304          # params per expert: gate+up, down
EDGE_HI = 37                                    # layers >= EDGE_HI get --edge-weight (ds4: 37-39)


def load(work, layers):
    out = {}
    for p in sorted(glob.glob(os.path.join(work, "stats", "L*.json"))):
        L = int(re.search(r"L(\d+)\.json", p).group(1))
        if layers is None or L in layers:
            out[L] = json.load(open(p))
    return out


def items_for(stats, a):
    """-> list of (layer, expert, part, [(tier, err_norm, bytes)...])."""
    items = []
    for L, j in stats.items():
        ex = j["experts"]
        energy = sum(s.get("y_energy", 0.0) for s in ex.values()) or 1.0
        w = a.edge_weight if (L <= 1 or L >= EDGE_HI) else 1.0
        for e, s in ex.items():
            # "Unseen" means too few held-out tokens, NOT "the combined e_<tier> key is
            # absent": an edited qwen run restricts the down ladder to iq4_nl/q8_0, so no
            # e_q2_0 is ever written and that test would mark all 512 experts unseen -
            # silently flooring them and excluding ternary gate/up from the whole file.
            unseen = s.get("n_eval", 0) < a.min_eval
            # Edited qwen down projections are restricted to formats that can
            # preserve the intervention. A pristine control run may opt out,
            # but then its gate/up and down still need separate ladders.
            dn_ladder = (DN_TIERS if a.edit_downs else list(ARCH[a.arch].get("pristine_dn_tiers", DN_TIERS))) if a.arch == "qwen4exp" else list(TIERS)
            if a.per_proj:
                parts = [("gu", GU, "egu_", TIERS), ("dn", DN, "edn_", dn_ladder)]
            else:
                parts = [("all", GU + DN, "e_", TIERS)]
            for part, n, key, ladder in parts:
                # the unseen floor is a bpw, so it lands correctly on a shorter ladder: pick the
                # cheapest tier of THIS part that is at least as large as the requested floor
                floor_bpw = BPW[a.unseen_floor]
                ch = []
                for t in ladder:
                    if unseen and BPW[t] < floor_bpw:
                        continue
                    if unseen:
                        err = 0.0
                    else:
                        if key + t not in s:
                            raise ValueError(f"L{L} expert {e} {part}: missing measurement {key + t}; refusing to score it as zero")
                        err = float(s[key + t])
                        if not np.isfinite(err) or err < 0:
                            raise ValueError(f"L{L} expert {e} {part}: invalid {key + t}={err}")
                    ch.append((t, w * err / energy, n * BPW[t] / 8))
                if not ch:                       # floor above this ladder: keep its top tier
                    t = ladder[-1]
                    ch = [(t, 0.0, n * BPW[t] / 8)]
                items.append((L, int(e), part, ch, unseen))
    return items


def solve(items, budget, a, layer_n):
    def pick(lam):
        sel = [min(range(len(ch)), key=lambda i: ch[i][1] + lam * ch[i][2]) for (_, _, _, ch, _) in items]
        return sel, sum(items[k][3][i][2] for k, i in enumerate(sel))
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 4
    for _ in range(60):
        mid = (lo + hi) / 2
        if pick(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    sel, used = pick(hi)
    # per-layer ternary cap: promote the cheapest-to-promote ternary picks to the next tier
    for L, nl in layer_n.items():
        tern = [k for k, it in enumerate(items) if it[0] == L and it[3][sel[k]][0] == TIERS[0]]
        cap = int(a.cap_at.get(L, a.max_ternary_frac) * nl * (2 if a.per_proj else 1))
        if len(tern) > cap:
            tern.sort(key=lambda k: items[k][3][sel[k] + 1][1] - items[k][3][sel[k]][1], reverse=True)
            for k in tern[: len(tern) - cap]:
                used += items[k][3][sel[k] + 1][2] - items[k][3][sel[k]][2]
                sel[k] += 1
    # cap may overshoot the budget: demote q3 picks with the smallest loss; else spend leftover greedily
    def gain(k, d):
        ch, i = items[k][3], sel[k]
        j = i + d
        if not 0 <= j < len(ch) or (d < 0 and ch[j][0] == TIERS[0]):
            return None
        return (ch[i][1] - ch[j][1]) / abs(ch[j][2] - ch[i][2])
    while used > budget:
        cand = [(g, k) for k in range(len(items)) if items[k][3][sel[k]][0] == TIERS[-1] and (g := gain(k, -1)) is not None]
        if not cand:
            break
        _, k = min(cand)
        used += items[k][3][sel[k] - 1][2] - items[k][3][sel[k]][2]; sel[k] -= 1
    while True:
        cand = sorted(((g, k) for k in range(len(items)) if (g := gain(k, +1)) is not None and g > 0), reverse=True)
        moved = False
        for g, k in cand:
            dbytes = items[k][3][sel[k] + 1][2] - items[k][3][sel[k]][2]
            if used + dbytes <= budget:
                used += dbytes; sel[k] += 1; moved = True
                break
        if not moved:
            break
    return sel, used


def layer_choices(stats, a):
    """Multiple-choice knapsack at the granularity a Qwen GGUF can serve.

    A layer's gate/up and down each have one tier. Each expert still supplies
    its own measured error; only the final combinatorial choice is per-layer.
    No majority-vote collapse and no unmeasured zero-cost tier.
    """
    import math
    layers = sorted(stats)
    choices = []
    for L in layers:
        ex = stats[L]["experts"]
        energy = sum(s.get("y_energy", 0.0) for s in ex.values()) or 1.0
        edge = a.edge_weight if L <= 1 or L >= EDGE_HI else 1.0
        dn_ladder = DN_TIERS if a.edit_downs else ARCH[a.arch].get("pristine_dn_tiers", DN_TIERS)
        options = []
        for gu in TIERS:
            for dn in dn_ladder:
                err = cost = 0.0
                for e, s in ex.items():
                    unseen = s.get("n_eval", 0) < a.min_eval
                    for part, tier, prefix, n in (("gu", gu, "egu_", GU), ("dn", dn, "edn_", DN)):
                        if unseen and BPW[tier] < BPW[a.unseen_floor]:
                            break
                        if unseen:
                            continue
                        key = prefix + tier
                        if key not in s or not np.isfinite(s[key]) or s[key] < 0:
                            raise ValueError(f"L{L} expert {e} {part}: missing/invalid {key}")
                        err += edge * float(s[key]) / energy
                        cost += n * BPW[tier] / 8
                    else:
                        continue
                    break
                else:
                    options.append({"gu": gu, "dn": dn, "err": err, "bytes": cost})
        if not options:
            raise ValueError(f"L{L}: no legal tier choices")
        choices.append(options)
    # Byte grid for the DP. A per-expert grid (one PTQ1 gate+up expert, 716800 B) is too
    # coarse: iq2_xxs costs 1.18 of that unit and ptq1 costs 1.00, so both would round to
    # one unit and the DP could not tell them apart. One MiB per layer is ~0.1% of a
    # layer's bytes, and every option is rounded UP so the plan can never exceed budget.
    unit = 1 << 20
    units = [[int(-(-o["bytes"] // unit)) for o in opts] for opts in choices]
    max_units = sum(max(u) for u in units)
    prev = np.full(max_units + 1, np.inf)
    prev[0] = 0.0
    back = []
    for L, opts in enumerate(choices):
        cur = np.full(max_units + 1, np.inf)
        parent = np.full(max_units + 1, -1, dtype=np.int32)
        for i, o in enumerate(opts):
            u = units[L][i]
            if u > max_units:
                continue
            v = prev[:max_units + 1 - u] + o["err"]
            better = v < cur[u:]
            cur[u:][better] = v[better]
            parent[u:][better] = i
        prev, back = cur, back + [parent]
    return layers, choices, units, prev, back, unit


def solve_layers(stats, a, budget):
    layers, choices, units, costs, back, unit = layer_choices(stats, a)
    limit = min(int(budget // unit), len(costs) - 1)
    feasible = [b for b in range(limit + 1) if np.isfinite(costs[b])]
    if not feasible:
        raise ValueError(f"budget {budget / 2**30:.3f} GiB cannot fund the cheapest legal layer choices")
    b = min(feasible, key=lambda x: costs[x])
    amap, used, err = {}, 0, costs[b]
    for i in range(len(layers) - 1, -1, -1):
        pick = back[i][b]
        if pick < 0:
            raise AssertionError("layer-choice DP backtrack failed")
        L = layers[i]
        o = choices[i][pick]
        amap[str(L)] = {str(e): {"gu": o["gu"], "dn": o["dn"]} for e in stats[L]["experts"]}
        used += o["bytes"]
        b -= units[i][pick]
    if b != 0:
        raise AssertionError(f"layer-choice DP residual budget units: {b}")
    if used > budget:
        raise AssertionError(f"layer-choice allocation exceeds budget: {used} > {budget}")
    return amap, used, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("V41_WORK", "/media/Crucial1TB/v41-calib"))
    ap.add_argument("--layers", default=None, help="e.g. 0-7")
    ap.add_argument("--per-proj", action="store_true")
    ap.add_argument("--edge-weight", type=float, default=2.0)
    ap.add_argument("--max-ternary-frac", type=float, default=0.5)
    ap.add_argument("--min-eval", type=int, default=8)
    ap.add_argument("--unseen-floor", default="q2_0",
                    choices=sorted({t for v in ARCH.values() for t in v["tiers"]}))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ternary-cap-at", default="", help="per-layer caps, e.g. 20-31,33-35=1.0;36-39=0.75")
    ap.add_argument("--budget-gib", type=float, default=0, help="expert bytes to spend (default: the all-Q2_0 count)")
    ap.add_argument("--arch", default="ds4", choices=sorted(ARCH),
                    help="model geometry + tier ladder (default ds4 = DeepSeek-V4.1-Flash)")
    ap.add_argument("--layer-choice", action="store_true",
                    help="qwen: solve a budget-aware knapsack over per-layer (gu,dn) choices; no post-hoc collapse")
    ap.add_argument("--edit-downs", action=argparse.BooleanOptionalAction, default=None,
                    help="protect edited qwen expert-down tensors (required for an ablated qwen run)")
    a = ap.parse_args()
    if a.arch == "qwen4exp":
        if not a.per_proj:
            raise SystemExit("qwen4exp requires --per-proj: edited down and unedited gate/up have different legal tiers")
        if a.edit_downs is None:
            raise SystemExit("qwen4exp requires an explicit --edit-downs or --no-edit-downs provenance decision")
        if not a.layer_choice:
            raise SystemExit("qwen4exp requires --layer-choice: no per-expert allocation followed by a lossy collapse")
    else:
        a.edit_downs = False
    ar = ARCH[a.arch]
    # These MUST be the module globals: items_for()/solve()/layer_choices() read them by
    # name. Assigning them as locals here (as this did) leaves the DeepSeek-V4 defaults in
    # place, so a qwen4exp run died at the first expert with KeyError: 'iq2_xxs'.
    global TIERS, BPW, DN_TIERS, GU, DN, EDGE_HI
    TIERS, BPW = list(ar["tiers"]), dict(ar["bpw"])
    # a model whose down projection cannot hold the full ladder declares a shorter one; every other
    # model keeps the old behaviour of one ladder for both projections
    DN_TIERS = list(ar.get("dn_tiers") or ar["tiers"])
    GU, DN = 2 * ar["n_embd"] * ar["n_ff"], ar["n_embd"] * ar["n_ff"]
    EDGE_HI = ar["n_layers"] - 3
    assert a.unseen_floor in TIERS, f"--unseen-floor {a.unseen_floor} is not a {a.arch} tier {TIERS}"
    layers = None
    if a.layers:
        x, y = map(int, a.layers.split("-"))
        layers = set(range(x, y + 1))
    stats = load(a.work, layers)
    assert stats, "no stats"
    items = items_for(stats, a)
    layer_n = {L: len(j["experts"]) for L, j in stats.items()}
    a.cap_at = {}
    for grp in filter(None, a.ternary_cap_at.split(";")):
        rng, v = grp.split("=")
        for part in rng.split(","):
            lo_, _, hi_ = part.partition("-")
            for L in range(int(lo_), int(hi_ or lo_) + 1):
                a.cap_at[L] = float(v)
    # Default budget = the cheapest legal build. This used to hardcode q2_0 as the reference
    # tier, which crashes on any ladder that does not offer it (qwen4exp no longer does).
    # ch[0] is already the cheapest tier of THAT part's ladder, so this is arch-independent.
    budget = sum(it[3][0][2] for it in items)
    if a.budget_gib:   # a smaller (or larger) file than the cheapest legal build
        budget = a.budget_gib * 2**30
    if a.layer_choice:
        amap, used, mixed_err = solve_layers(stats, a, budget)
        n_gu = sum(len(j["experts"]) for j in stats.values())
        # Print the whole ladder, per projection, as explicit values. The previous version
        # built a dict keyed only by "gu"/"dn" inside a comprehension over TIERS, so the last
        # tier won and the number labelled "PTQ1" was actually q8_0's (4.9x too large).
        def gib(part_tier):
            n = GU if part_tier[0] == "gu" else DN
            return n_gu * n * BPW[part_tier[1]] / 8 / 2**30
        print(f"EXPERT BYTE LEDGER for {n_gu} experts "
              f"(cheapest legal total = {budget / 2**30:.2f} GiB):")
        print("  gate/up: " + "  ".join(f"{t}={gib(('gu', t)):.2f}" for t in TIERS) + " GiB")
        print("  down   : " + "  ".join(f"{t}={gib(('dn', t)):.2f}" for t in DN_TIERS) + " GiB")
        for L in sorted(amap, key=int):
            tiers = next(iter(amap[L].values()))
            print(f"L{int(L):02d}: gu={tiers['gu']:8s} dn={tiers['dn']:7s} "
                  f"n={len(amap[L])} exact={sum(GU * BPW[tiers['gu']] + DN * BPW[tiers['dn']] for _ in amap[L]) / 8 / 2**30:.4f} GiB")
        q2_err = float("nan")
        if a.out:
            json.dump({"layers": sorted(stats), "per_proj": True, "layer_choice": True,
                       "budget_bytes": budget, "used_bytes": used, "proxy_err_mixed": mixed_err,
                       "edit_downs": a.edit_downs, "args": vars(a), "map": amap}, open(a.out, "w"), indent=1)
            print(f"wrote {a.out}: per-layer exact tiers, used {used / 2**30:.4f} GiB / budget {budget / 2**30:.4f} GiB, proxy {mixed_err:.6f}")
        return
    sel, used = solve(items, budget, a, layer_n)
    err = lambda t: sum(next(c[1] for c in it[3] if c[0] == t) for it in items if t in [c[0] for c in it[3]])
    mixed_err = sum(items[k][3][i][1] for k, i in enumerate(sel))
    q2_err = sum(next(c[1] for c in it[3] if c[0] == "q2_0") for it in items)
    print(f"layers {sorted(stats)}  items {len(items)}  budget {budget / 2**30:.2f} GiB (all-Q2_0)  used {used / 2**30:.2f} GiB")
    print(f"proxy error (sum of per-layer relative output error): all-Q2_0 {q2_err:.4f}  mixed {mixed_err:.4f}  "
          f"({100 * (mixed_err / q2_err - 1):+.1f}%)")
    hdr = " ".join(f"{t:>9}" for t in TIERS)
    print(f"{'L':>3} {hdr} {'unseen':>6} {'bpw':>6}   ratio(GPTQ/RTN)   rel err @q2")
    amap = {}
    for L in sorted(stats):
        ks = [k for k, it in enumerate(items) if it[0] == L]
        c = {t: sum(items[k][3][sel[k]][0] == t for k in ks) for t in TIERS}
        bits = sum(items[k][3][sel[k]][2] for k in ks) * 8 / sum((GU + DN) if items[k][2] == "all" else (GU if items[k][2] == "gu" else DN) for k in ks)
        ex = stats[L]["experts"].values()
        r = [np.nansum([s.get(f"e_{t}", np.nan) for s in ex]) / max(np.nansum([s.get(f"rtn_{t}", np.nan) for s in ex]), 1e-30) for t in TIERS]
        relq2 = np.nansum([s.get("e_q2_0", np.nan) for s in ex]) / max(sum(s.get("y_energy", 0) for s in ex), 1e-30)
        cols = " ".join(f"{c[t]:>9}" for t in TIERS)
        rat = " ".join(f"{v:.2f}" for v in r)
        print(f"{L:>3} {cols} {sum(items[k][4] for k in ks):>6} {bits:>6.3f}   {rat}   {relq2:.4f}")
        for k in ks:
            it = items[k]
            amap.setdefault(str(L), {}).setdefault(str(it[1]), {})[it[2]] = it[3][sel[k]][0]
    if a.out:
        json.dump({"layers": sorted(stats), "per_proj": a.per_proj, "budget_bytes": budget, "used_bytes": used,
                   "proxy_err_all_q2": q2_err, "proxy_err_mixed": mixed_err, "args": vars(a), "map": amap},
                  open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
