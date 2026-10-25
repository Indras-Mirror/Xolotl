#!/usr/bin/env python3
"""v41_rank_route_log.py - a --rank file from the engine's own routing: STRATA_ROUTE_LOG lines ("layer expert tier
count", one per routed expert of each batch pass) summed per (layer, expert), ordered as v41_rank.py orders its
calibration counts: routed tokens per BYTE of the expert's blob (gate + up + down at the layer's types), hottest first.

For a model without calibration stats (e.g. one converted by tools/v41/v41_from_llamacpp.py): run a corpus through
ds41_generate with STRATA_ROUTE_LOG set and --no-enc-skip (so every layer sees every token), then

    python v41_rank_route_log.py --log route.log --gguf model.gguf --out rank.bin
writes int32 (layer, expert) pairs and prints the cumulative routed share covered by the first N GiB.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tools/v41
from v41_from_llamacpp import BLOCK, Header  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, action="append", help="route log (repeatable)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = Header(a.gguf)
    NL, NE = h.get("deepseek41.num_hidden_layers"), h.get("deepseek41.n_routed_experts")
    D, FF = h.get("deepseek41.hidden_size"), h.get("deepseek41.moe_intermediate_size")
    types = {}
    for name, dims, t, off in h.tensors:
        for role in ("gate", "up", "down"):
            if name.endswith(f".ffn_{role}_exps.weight"):
                types[(int(name.split(".")[1]), role)] = t
    if len(types) != 3 * NL:
        sys.exit("expected one ffn_{gate,up,down}_exps tensor per layer (a uniform-per-layer file)")
    blob = [sum(D * FF * BLOCK[types[(L, r)]][0] // BLOCK[types[(L, r)]][1] for r in ("gate", "up", "down"))
            for L in range(NL)]
    n = np.zeros((NL, NE), np.float64)
    lines = 0
    for path in a.log:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 4:
                    n[int(p[0]), int(p[1])] += int(p[3])
                    lines += 1
    total = n.sum()
    seen = (n > 0).sum(axis=1)
    rows = sorted(((n[L, e] / blob[L], L, e) for L in range(NL) for e in range(NE)), key=lambda r: (-r[0], r[1], r[2]))
    np.array([(r[1], r[2]) for r in rows], np.int32).tofile(a.out)
    print(f"{lines} log lines, {total:.0f} routed (token, expert) picks = {total / NL / 6:.0f} tokens per layer at top-6; "
          f"experts seen per layer min {seen.min()} / max {seen.max()} of {NE}")
    cum_n, cum_b, marks = 0.0, 0.0, [8, 16, 24, 32, 44, 64, 100, 150]
    for r in rows:
        cum_n += r[0] * blob[r[1]]
        cum_b += blob[r[1]]
        while marks and cum_b >= marks[0] * 2**30:
            print(f"first {marks[0]:3d} GiB of experts cover {100 * cum_n / max(total, 1):5.1f}% of routed picks")
            marks.pop(0)
    print(f"wrote {len(rows)} pairs to {a.out}; all experts {cum_b / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
