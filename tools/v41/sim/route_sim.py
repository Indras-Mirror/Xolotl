#!/usr/bin/env python3
"""tools/v41/sim/route_sim.py - replay ds41_generate's STRATA_ROUTE_TRACE (session 13, measurement only).

The trace (Ds41MoeTier::trace_open / trace_mark, one-shot DSpark decode):
  'H' L X, then L*X x {class, VRAM-resident, in-arena}   - the cache at the first decode pass
  'C' layer nt nu, then nu x {expert, first row, cls, from_pf}   - one run_multi_to_dev call (a verify pass layer);
       cls 0 HIT / 1 PCIE / 2 CPU / 3 FILEG (as the engine served it)
  'P' n, then n floats {m, k, acc, refeed, verify ms, p0..p4}  - the pass that the preceding 'C' records belong to

Two questions, both answered offline from the same trace:
  policy  - VRAM replacement: the engine's per-(layer, class) LRU with its 4-call admission window, replayed, then
            alternatives, then Belady's MIN (reads the trace ahead: the ceiling of ANY policy on this trace).
  prune   - in-pass draft pruning: the misses only the trailing draft rows cause, their cost (a regression of verify
            ms on the pass's misses), and what a trim rule at layer Lc would save vs the tokens it loses.

    python3 route_sim.py TRACE.bin [TRACE.bin ...] [--policy] [--prune]
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict

import numpy as np

HIT, PCIE, CPU, FILEG = 0, 1, 2, 3
PF = 4   # sim only: a miss the engine served from a D3 prefetch slot (its DMA ran under the attention; LRU-admitted)


def load(path):
    b = open(path, 'rb').read()
    a = np.frombuffer(b, dtype=np.int32)
    i = 0
    assert a[0] == ord('H'), 'not a route trace'
    L, X = int(a[1]), int(a[2])
    i = 3
    snap = a[i:i + L * X * 3].reshape(L, X, 3).copy()
    i += L * X * 3
    passes, calls = [], []
    while i < len(a):
        tag = a[i]
        if tag == ord('C'):
            layer, nt, nu = int(a[i + 1]), int(a[i + 2]), int(a[i + 3])
            u = a[i + 4:i + 4 + 4 * nu].reshape(nu, 4).copy()
            calls.append((layer, nt, u))
            i += 4 + 4 * nu
        elif tag == ord('P'):
            n = int(a[i + 1])
            v = np.frombuffer(b[(i + 2) * 4:(i + 2 + n) * 4], dtype=np.float32).copy()
            passes.append((v, calls))
            calls = []
            i += 2 + n
        else:
            raise SystemExit(f'{path}: bad tag {tag} at word {i}')
    return L, X, snap, passes


# ------------------------------------------------------------------------------------------------ policy replay
class Sim:
    """VRAM slots per (layer, class) + the RAM arena (exclusive tiers, as the engine: a VRAM victim is demoted into
    the arena slot its replacement vacates; a file-tier miss is read into the arena slot of the layer's
    least-recently-used arena expert of the same class - arena_adapt)."""

    def __init__(self, L, X, snap, policy, win=4, oracle_next=None, admit_file=False, decay=0.98):
        self.L, self.X = L, X
        self.cls = snap[:, :, 0]
        self.vram = snap[:, :, 1].astype(bool).copy()
        self.arena = snap[:, :, 2].astype(bool).copy()
        self.policy, self.win, self.nxt, self.admit_file, self.decay = policy, win, oracle_next, admit_file, decay
        self.tick = 0
        self.last = np.full((L, X), -1, np.int64)       # last use (call tick)
        self.last2 = np.full((L, X), -1, np.int64)      # second-to-last use (LRU-2)
        self.freq = np.zeros((L, X), np.float64)        # LFU-decay count
        self.vmiss = np.full((L, X), -10**9, np.int64)  # layer call count at the last miss (admission window)
        self.lcalls = np.zeros(L, np.int64)
        self.n = {HIT: 0, PCIE: 0, FILEG: 0, PF: 0}
        # SLRU (llama.cpp #30637): a VRAM expert hit again moves to the protected segment (<= `prot` of its
        # (layer, class) slots, overflow demoted back to probation by LRU); victims come from probation first
        self.prot = np.zeros((L, X), bool)
        self.prot_frac = 0.5
        self.cap = {}
        for l in range(L):
            for c in np.unique(self.cls[l]):
                self.cap[(l, int(c))] = int((self.vram[l] & (self.cls[l] == c)).sum())
        self.swaps = 0

    def victim(self, l, e, used, ci):
        cand = np.nonzero(self.vram[l] & (self.cls[l] == ci))[0]
        cand = [c for c in cand if c not in used]
        if not cand:
            return -1
        if self.policy == 'oracle':
            return max(cand, key=lambda c: self.nxt(l, c, self.tick))
        if self.policy == 'slru':
            prob = [c for c in cand if not self.prot[l, c]]
            return min(prob or cand, key=lambda c: self.last[l, c])
        if self.policy == 'lru2':
            return min(cand, key=lambda c: (self.last2[l, c], self.last[l, c]))
        if self.policy == 'lfu':
            return min(cand, key=lambda c: (self.freq[l, c], self.last[l, c]))
        return min(cand, key=lambda c: self.last[l, c])

    def call(self, l, experts, pf):
        self.tick += 1
        self.lcalls[l] += 1
        lc = self.lcalls[l]
        used = set(int(e) for e in experts)
        out = []
        for e, fp in zip(experts, pf):
            e = int(e)
            if self.vram[l, e]:
                out.append(HIT)
                if self.policy == 'slru' and not self.prot[l, e]:
                    self.prot[l, e] = True
                    ci = int(self.cls[l, e])
                    pm = np.nonzero(self.prot[l] & (self.cls[l] == ci))[0]
                    if len(pm) > self.prot_frac * self.cap[(l, ci)]:
                        d = min((c for c in pm if c != e), key=lambda c: self.last[l, c], default=-1)
                        if d >= 0:
                            self.prot[l, d] = False
                continue
            c = PCIE if self.arena[l, e] else FILEG
            if fp and c == PCIE:
                c = PF
            out.append(c)
            if c == FILEG:   # arena_adapt: into the arena slot of the layer's LRU same-class arena expert not routed now
                cand = np.nonzero(self.arena[l] & (self.cls[l] == self.cls[l, e]))[0]
                cand = [x for x in cand if x not in used]
                if cand:
                    v = min(cand, key=lambda x: self.last[l, x])
                    self.arena[l, v] = False
                    self.arena[l, e] = True
            # VRAM admission
            if c == FILEG and not self.admit_file:
                continue
            if self.policy == 'oracle':
                ok = True
            else:
                ok = lc - self.vmiss[l, e] <= self.win
            if not ok:
                continue
            v = self.victim(l, e, used, self.cls[l, e])
            if v < 0:
                continue
            if self.policy == 'oracle' and self.nxt(l, e, self.tick) >= self.nxt(l, v, self.tick):
                continue   # MIN with bypass: the incoming one is needed later than every candidate
            if self.policy == 'tinylfu' and self.freq[l, e] + 1.0 <= self.freq[l, v]:
                continue   # TinyLFU admission: a miss must be used more often than the LRU victim it would replace
            self.prot[l, v] = False
            self.prot[l, e] = False
            self.vram[l, v] = False
            self.arena[l, v] = True     # demoted into the slot e vacates
            self.vram[l, e] = True
            self.arena[l, e] = False
            self.swaps += 1
        for e, c in zip(experts, out):
            e = int(e)
            self.n[c] += 1
            if c != HIT:
                self.vmiss[l, e] = lc
            self.last2[l, e] = self.last[l, e]
            self.last[l, e] = self.tick
            self.freq[l, e] = self.freq[l, e] * self.decay + 1.0
        return out


def policy_report(traces, c_p, c_f):
    print('\n=== VRAM replacement policy (replay; misses per token; est. ms/token = pcie*%.2f + file*%.2f) ===' % (c_p, c_f))
    for name, (L, X, snap, passes) in traces:
        seq = [(l, u[:, 0], u[:, 3]) for _, calls in passes for (l, nt, u) in calls]
        obs = defaultdict(int)
        for _, calls in passes:
            for (l, nt, u) in calls:
                for c in u[:, 2]:
                    obs[int(c)] += 1
        toks = sum(int(v[2]) + 1 for v, _ in passes)
        # next-use index for the oracle: per (layer, expert) the sorted call ticks
        uses = defaultdict(list)
        for t, (l, ex, _) in enumerate(seq, start=1):
            for e in ex:
                uses[(l, int(e))].append(t)

        def nxt(l, e, tick):
            lst = uses.get((l, e))
            if not lst:
                return 10**12
            j = np.searchsorted(lst, tick, side='right')
            return lst[j] if j < len(lst) else 10**12

        print(f'\n[{name}] {len(passes)} passes, {toks} tokens, {len(seq)} calls; resident at start {int(snap[:,:,1].sum())}, '
              f'arena {int(snap[:,:,2].sum())}')
        print(f'  engine observed  : hit {obs[HIT]/toks:6.2f}  pcie {obs[PCIE]/toks:5.2f}  cpu {obs[CPU]/toks:4.2f}  '
              f'file {obs[FILEG]/toks:5.2f}   (VRAM hit {100*obs[HIT]/max(1,sum(obs.values())):.1f}%)')
        configs = [('lru', 4, False, .98), ('lru', 1, False, .98), ('lru', 8, False, .98), ('lru', 10**9, False, .98),
                   ('lru2', 4, False, .98), ('lfu', 4, False, .98), ('lfu', 4, False, .9), ('lfu', 4, False, .995),
                   ('tinylfu', 10**9, False, .98), ('tinylfu', 4, False, .98), ('tinylfu', 10**9, False, .995),
                   ('slru', 4, False, .98), ('slru', 10**9, False, .98),
                   ('oracle', 0, False, .98)]
        base = None
        for pol, win, af, dec in configs:
            s = Sim(L, X, snap, pol, win, nxt, af, dec)
            for l, ex, pf in seq:
                s.call(l, ex, pf)
            ms = (s.n[PCIE] * c_p + s.n[FILEG] * c_f) / toks
            if base is None:
                base = ms
            lab = f'{pol} win={win if win < 10**8 else "inf"}{" +file-admit" if af else ""}' + (f' d={dec}' if 'lfu' in pol else '')
            if pol == 'oracle':
                lab = 'ORACLE (MIN+bypass)' + (' +file-admit' if af else '')
            tot = s.n[HIT] + s.n[PCIE] + s.n[FILEG] + s.n[PF]
            print(f'  {lab:30s}: hit {s.n[HIT]/toks:6.2f}+pf {s.n[PF]/toks:4.2f}  pcie {s.n[PCIE]/toks:5.2f}  file {s.n[FILEG]/toks:5.2f}  '
                  f'(hit {100*(s.n[HIT]+s.n[PF])/tot:.1f}%)  swaps/tok {s.swaps/toks:5.1f}  miss-ms/tok {ms:5.2f} ({ms-base:+.2f})')


# ------------------------------------------------------------------------------------------------ prune
def pass_rows(v, calls, n_layer):
    """per layer, per first-row r: misses (pcie, file) and hits whose FIRST routing row is r"""
    m = int(v[0])
    pc = np.zeros((n_layer, m)); fi = np.zeros((n_layer, m)); hi = np.zeros((n_layer, m))
    for (l, nt, u) in calls:
        for e, r, c, pf in u:
            if r >= m:
                continue
            if c == HIT:
                hi[l, r] += 1
            elif c == FILEG:
                fi[l, r] += 1
            else:
                pc[l, r] += 1
    return pc, fi, hi


def prune_report(traces):
    rows = []
    for name, (L, X, snap, passes) in traces:
        for v, calls in passes:
            pc, fi, hi = pass_rows(v, calls, L)
            rows.append((name, v, pc, fi, hi))
    # regression: verify ms = a + b*m + c*pcie + d*file (+ hits)
    A = np.array([[1, r[1][0], r[2].sum(), r[3].sum()] for r in rows])
    y = np.array([r[1][4] for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a, b, c_p, c_f = coef
    print(f'\n=== verify ms regression over {len(rows)} passes: {a:.1f} + {b:.1f}*m + {c_p:.2f}*pcie + {c_f:.2f}*file  (R2 {r2:.3f})')
    for name, _ in traces:
        sel = [r for r in rows if r[0] == name]
        ms = np.mean([r[1][4] for r in sel]); toks = sum(int(r[1][2]) + 1 for r in sel)
        m = np.mean([r[1][0] for r in sel])
        pcs = np.mean([r[2].sum() for r in sel]); fis = np.mean([r[3].sum() for r in sel])
        print(f'  [{name}] {len(sel)} passes, mean m {m:.2f}, verify {ms:.1f} ms, pcie {pcs:.1f} file {fis:.1f} per pass, '
              f'{toks/len(sel):.2f} tokens/pass, {sum(r[1][4] for r in sel)/toks:.1f} verify-ms/token')

    def row_cost(pc, fi, r0, l0):
        """cost saved by dropping rows >= r0 from layer l0 on: their unique misses + the per-row term (b per row,
        prorated over the remaining layers)"""
        m = pc.shape[1]
        return (c_p * pc[l0:, r0:].sum() + c_f * fi[l0:, r0:].sum()) + b * (m - r0) * (L - l0) / L

    print('\n=== in-pass trim of the LAST row (verify-ms saved vs tokens lost), by decision layer Lc')
    print('    rule: drop the last draft row at layer Lc if its observed unique-miss cost so far x (40/Lc - 1) > '
          'theta x p_cum x (verify-ms/token)')
    for name, _ in traces + [('ALL', None)]:
        sel = [r for r in rows if name == 'ALL' or r[0] == name]
        tot_ms = sum(r[1][4] for r in sel); tot_tok = sum(int(r[1][2]) + 1 for r in sel)
        mpt = tot_ms / tot_tok
        elig = [r for r in sel if int(r[1][1]) >= 1]
        # oracle: drop the last row iff its full cost > its realised token value x ms/token (knows acceptance)
        o_save = o_lost = 0
        for name_, v, pc, fi, hi in elig:
            m, k, acc = int(v[0]), int(v[1]), int(v[2])
            cost = row_cost(pc, fi, m - 1, 0)
            lost = 1 if acc == k else 0
            if cost > lost * mpt:
                o_save += cost; o_lost += lost
        print(f'\n  [{name}] {len(sel)} passes, {tot_tok} tokens, {mpt:.1f} verify-ms/token; eligible (k>=1) {len(elig)}')
        print(f'    ORACLE (knows cost + acceptance): saves {o_save:.0f} ms, loses {o_lost} tokens -> '
              f'{(tot_ms - o_save)/(tot_tok - o_lost):.2f} ms/token ({100*((tot_ms - o_save)/(tot_tok - o_lost)/mpt-1):+.1f}%)')
        # the share of the last row's unique-miss cost by layer (how early is it visible)
        full = np.array([row_cost(pc, fi, int(v[0]) - 1, 0) for _, v, pc, fi, hi in elig])
        for Lc in (2, 4, 8, 12):
            obs = np.array([c_p * pc[:Lc, int(v[0]) - 1:].sum() + c_f * fi[:Lc, int(v[0]) - 1:].sum() for _, v, pc, fi, hi in elig])
            rest = np.array([row_cost(pc, fi, int(v[0]) - 1, Lc) for _, v, pc, fi, hi in elig])
            cc = np.corrcoef(obs, rest)[0, 1] if obs.std() > 0 and rest.std() > 0 else float('nan')
            best = None
            for theta in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                save = lost = 0
                for (_, v, pc, fi, hi), o, rs in zip(elig, obs, rest):
                    m, k, acc = int(v[0]), int(v[1]), int(v[2])
                    pcum = float(np.prod(v[5:5 + k]))
                    est = o * (L / Lc - 1) + b * (L - Lc) / L
                    if est > theta * pcum * mpt:
                        save += rs; lost += 1 if acc == k else 0
                new = (tot_ms - save) / max(1, tot_tok - lost)
                if best is None or new < best[0]:
                    best = (new, theta, save, lost)
            new, theta, save, lost = best
            print(f'    Lc={Lc:2d}: corr(observed, rest) {cc:.2f}; best theta {theta}: saves {save:.0f} ms, loses {lost} tokens -> '
                  f'{new:.2f} ms/token ({100*(new/mpt-1):+.1f}%)')
    return c_p, c_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--policy', action='store_true')
    ap.add_argument('--prune', action='store_true')
    ap.add_argument('--c-p', type=float, default=None)
    ap.add_argument('--c-f', type=float, default=None)
    a = ap.parse_args()
    traces = [(p.rsplit('/', 1)[-1].replace('.bin', ''), load(p)) for p in a.traces]
    c_p, c_f = 0.45, 2.3
    if a.prune or not a.policy:
        c_p, c_f = prune_report(traces)
    if a.c_p: c_p = a.c_p
    if a.c_f: c_f = a.c_f
    if a.policy or not a.prune:
        policy_report(traces, c_p, c_f)


if __name__ == '__main__':
    main()
