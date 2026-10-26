#!/usr/bin/env python3
"""lite_alloc.py - pick the Q3_K / Q2_0 map of HXQ3K-lite for a target expert size from lite_calib's stats.

Every routed expert has four classes (gate+up, down) in {Q3_K, Q2_0}^2 with their MEASURED held-out output errors
e33 / e23 / e32 / e22 (lite_calib: sum ||f(x;W) - f(x;Q)||^2 on held-out tokens, exactly that combination).  Cost of a
class defaults to the expert's relative held-out error times its share of the layer's routed tokens,
times --edge-weight for layers 0-1 and 37-39. This measures quantization damage weighted by usage;
routing frequency alone does not determine the precision.
Solved exactly per expert by Lagrangian relaxation (argmin cost + lam * bytes, bisection on lam to the byte budget),
then a greedy pass fills what is left with the cheapest remaining downgrade per byte.
Experts with fewer than --min-eval held-out tokens (their error is noise) get the layer's median relative cost per
class, weighted by their own usage under --cost use (or unweighted under the other objectives).

  lite_alloc.py --stats OUT/stats --gib 190 [--edge-weight 1.5] [--out map.json]
"""
import argparse, glob, hashlib, json, os, re

import numpy as np

D, FF, NE = 5120, 2304, 384
Q3 = D * FF // 256 * 110
Q2 = D * FF // 64 * 18
BYTES = {"33": 3 * Q3, "23": 2 * Q2 + Q3, "32": 2 * Q3 + Q2, "22": 3 * Q2}
CLS = ("33", "23", "32", "22")


def load(stats_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(stats_dir, "L*.json"))):
        m = re.search(r"L(\d+)\.json$", p)
        if m:
            out[int(m.group(1))] = json.load(open(p))["experts"]
    return out


def costs(stats, a):
    """-> C [L, 384, 4] cost per class, unseen mask [L, 384], protected mask [L, 384] (forced Q3_K).

    --cost use (default): own relative error times the expert's share of routed tokens.
    --cost rel: the expert's own relative held-out error e_c / y_energy, averaged over the layer's experts
      (every expert and every layer count the same).  The calibration found massive activations from layer ~15 on:
      one expert can hold 99.98 % of a layer's held-out output energy (energy per token up to 10^6 x the median), so
      the layer-energy normalization of v41_alloc (--cost energy) makes every other expert of such a layer look free.
    Massive experts (energy per held-out token > --massive x the layer median) are protected: Q3_K, no choice."""
    Ls = sorted(stats)
    C = np.zeros((len(Ls), NE, 4))
    uns = np.zeros((len(Ls), NE), bool)
    prot = np.zeros((len(Ls), NE), bool)
    for i, L in enumerate(Ls):
        ex = stats[L]
        E = sum(float(ex[str(e)].get("y_energy", 0.0)) for e in range(NE)) or 1.0
        w = a.edge_weight if (L <= 1 or L >= 37) else 1.0
        ept = np.array([float(ex[str(e)].get("y_energy", 0.0)) / max(ex[str(e)].get("n_eval", 0), 1) for e in range(NE)])
        med = np.median(ept[ept > 0]) if (ept > 0).any() else 1.0
        Ntok = sum(float(ex[str(e)].get("n", 0)) for e in range(NE)) or 1.0
        for e in range(NE):
            s = ex[str(e)]
            if s.get("n_eval", 0) < a.min_eval or "e33" not in s:
                uns[i, e] = True
                continue
            y = max(float(s.get("y_energy", 0.0)), 1e-30)
            if a.cost == "energy":
                C[i, e] = [w * float(s["e" + c]) / E for c in CLS]
            elif a.cost == "rel":
                C[i, e] = [w * float(s["e" + c]) / (y * NE) for c in CLS]
            else:   # "use": the expert's relative error x its share of the layer's routed tokens (expected error per token)
                C[i, e] = [w * float(s["e" + c]) / y * s.get("n", 0) / Ntok for c in CLS]
            if a.massive > 0 and ept[e] > a.massive * med:
                prot[i, e] = True
        seen = ~uns[i]
        if seen.any() and uns[i].any():
            if a.cost == "use":   # the layer's median relative error per class x the unseen expert's own (small) usage
                nn = np.array([float(ex[str(e)].get("n", 0)) for e in range(NE)])
                rel = C[i, seen] / np.maximum(nn[seen, None] / Ntok, 1e-30)
                C[i, uns[i]] = np.median(rel, axis=0)[None, :] * (np.maximum(nn[uns[i]], 1.0) / Ntok)[:, None]
            else:
                C[i, uns[i]] = np.median(C[i, seen], axis=0)
    return Ls, C, uns, prot


def solve(C, budget, prot=None):
    if C.size == 0 or not np.isfinite(C).all() or not np.isfinite(budget):
        raise ValueError("Need nonempty finite calibration costs and a finite byte budget")
    by = np.array([BYTES[c] for c in CLS], np.float64)
    n_protected = int(prot.sum()) if prot is not None else 0
    minimum = (C.shape[0] * C.shape[1] - n_protected) * BYTES["22"] + n_protected * BYTES["33"]
    if budget < minimum:
        raise ValueError(f"Budget {budget:g} bytes is below the feasible minimum {minimum} bytes")
    if prot is not None and prot.any():   # protected experts: only Q3_K/Q3_K is allowed
        C = C.copy()
        C[prot, 1:] = np.inf

    def pick(lam):
        k = np.argmin(C + lam * by[None, None, :], axis=-1)
        return k, by[k].sum()
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pick(mid)[1] > budget else (lo, mid)
    k, tot = pick(hi)
    # greedy: spend the slack on the cheapest upgrades (towards Q3_K) per byte that still fit
    slack = budget - tot
    while True:
        best = None
        cur = np.take_along_axis(C, k[..., None], -1)[..., 0]
        for j in range(4):
            db = by[j] - by[k]
            dc = cur - C[..., j]
            ok = (db > 0) & (db <= slack) & (dc > 0) & np.isfinite(C[..., j])
            if ok.any():
                r = np.where(ok, dc / np.maximum(db, 1), -1)
                idx = np.unravel_index(np.argmax(r), r.shape)
                if best is None or r[idx] > best[0]:
                    best = (r[idx], idx, j, db[idx])
        if best is None:
            break
        _, idx, j, db = best
        k[idx] = j
        slack -= db
    return k, by[k].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--gib", type=float, required=True, help="expert bytes budget, GiB (HXQ3K: 217.5)")
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--min-eval", type=int, default=16)
    ap.add_argument("--cost", choices=("use", "rel", "energy"), default="use")
    ap.add_argument("--massive", type=float, default=100.0, help="protect experts above this x median energy/token (0 = off)")
    ap.add_argument("--out")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    stats = load(a.stats)
    Ls, C, uns, prot = costs(stats, a)
    assert Ls == list(range(len(Ls))), f"layers {Ls}"
    k, tot = solve(C, a.gib * 2**30, prot)
    cost = np.take_along_axis(C, k[..., None], -1)[..., 0]
    base = C[..., 0].sum()
    allq2 = C[..., 3].sum()
    n = {c: int((k == j).sum()) for j, c in enumerate(CLS)}
    print(f"budget {a.gib} GiB -> experts {tot / 2**30:.2f} GiB; cost {a.cost}; classes gu/dn {n}; unseen {int(uns.sum())}, "
          f"protected (massive) {int(prot.sum())}")
    print(f"total relative error: chosen {cost.sum():.4f}, all-Q3_K {base:.4f} (+{cost.sum() - base:.4f}), all-Q2_0 {allq2:.4f}")
    per = [(L, (k[i] != 0).sum(), (cost[i].sum() - C[i, :, 0].sum())) for i, L in enumerate(Ls)]
    print("per layer (downgraded experts, added rel err): " + " ".join(f"L{L}:{d}/{x:.3f}" for L, d, x in per))
    if a.out:
        m = {"name": a.name or f"lite-{a.gib:g}", "gib_budget": a.gib, "experts_gib": round(tot / 2**30, 3),
             "edge_weight": a.edge_weight, "min_eval": a.min_eval, "cost": a.cost, "massive": a.massive,
             "protected": int(prot.sum()), "classes": n,
             "rel_err": {"chosen": float(cost.sum()), "all_q3k": float(base), "all_q2_0": float(allq2)},
             "layers": {str(L): {"gu": ["q2_0" if CLS[j][0] == "2" else "q3_k" for j in k[i]],
                                 "dn": ["q2_0" if CLS[j][1] == "2" else "q3_k" for j in k[i]]} for i, L in enumerate(Ls)}}
        m["sha"] = hashlib.sha256(json.dumps(m["layers"], sort_keys=True).encode()).hexdigest()[:16]
        json.dump(m, open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
