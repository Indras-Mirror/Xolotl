#!/usr/bin/env python3
"""ernie_compare.py - compare a numpy reference dump (ernie_ref.py .npz) with a
llama.cpp PR dump (ernie_dump output directory).

    python3 tools/ernie/ernie_compare.py ref.npz dumpdir [--layers 3] [--quiet]

Prints, per captured tensor, the relative L2 of (llama - reference) and, for the
router, the fraction of (token, layer) slots whose top-10 id *sets* agree and
the max |dweight| over the matched experts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ggml tensor -> reference array layout.  The dump files are flat float32 in
# ggml order (ne0 fastest), so the numpy view is ne[::-1].
NAMES = [
    "hc_init",
    "l_last", "hc_attn_pre", "hc_ffn_pre", "attn_norm", "ffn_norm",
    "qr", "qr_norm", "q", "kv", "attn_hca", "attn_derope", "attn_wo_a", "attn_gate",
    "attn_out", "ffn_out", "ffn_shexp", "ffn_routed_up", "ffn_moe_probs",
    "ffn_moe_probs_split",
]
ROUTER = ["ffn_moe_topk", "ffn_moe_weights"]


def ggml_view(flat, ne, nt):
    """Flat ggml order (ne0 fastest) -> token-major numpy array."""
    ne = [int(x) for x in ne]
    while len(ne) > 1 and ne[-1] == 1:
        ne.pop()
    arr = flat[:int(np.prod(ne))].reshape(ne[::-1])
    if arr.shape[0] != nt and nt in arr.shape:
        # a token axis that ended up in the middle (e.g. attn_wo_a: [o, nt, group])
        arr = np.moveaxis(arr, arr.shape.index(nt), 0)
    return arr


def load_dump(dump_dir):
    with open(os.path.join(dump_dir, "manifest.json")) as f:
        man = json.load(f)
    nt = len(man.get("tokens") or [])
    out = {}
    seqs = {}
    for t in man["tensors"]:
        name = t["name"]
        flat = np.fromfile(os.path.join(dump_dir, t["file"]), dtype="<f4")
        arr = ggml_view(flat, t["ne"], nt)
        k = name
        seqs.setdefault(k, 0)
        seqs[k] += 1
        if seqs[k] > 1:
            k = f"{name}#{seqs[k]}"
        out[k] = arr
    out["__tokens"] = man.get("tokens")
    out["__flash_attn"] = man.get("flash_attn")
    return out


def rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("dump")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ref = np.load(args.ref)
    dump = load_dump(args.dump)
    layers = args.layers

    print(f"tokens  ref={len(ref['hc_init'])} dump={len(dump['__tokens'])} "
          f"flash_attn={dump['__flash_attn']}")
    worst = 0.0
    for il in range(-1, layers):
        rows = []
        for base in NAMES:
            name = base if il < 0 else f"{base}-{il}"
            if name not in ref or name not in dump:
                continue
            a = ref[name]
            b = dump[name]
            if a.shape != b.shape:
                try:
                    b = b.reshape(a.shape)
                except ValueError:
                    rows.append(f"{name:22s} SHAPE ref={a.shape} dump={b.shape}")
                    continue
            r = rel_l2(a, b)
            worst = max(worst, r if a.size > 64 else worst)
            rows.append(f"{name:22s} relL2 {r:9.3e}  max|d| {np.max(np.abs(a - b)):9.3e}")
        if not args.quiet or il == layers - 1:
            print(f"--- layer {il}" if il >= 0 else "--- root")
            for r in rows:
                print("   ", r)

    # router exactness.  The dumped I32 top-k tensor is not always trustworthy
    # (the small node's buffer can be recycled after its consumers run - the graph
    # keeps its own selection), so the authoritative comparison reconstructs the
    # selected ids from ffn_moe_probs + the gathered weights: w_j == probs[id_j]
    # exactly.  Direct id agreement is reported as well.
    print("--- router")
    slots_total = slots_same = slots_direct = 0
    dw_all = 0.0
    for il in range(layers):
        k_ids = f"ffn_moe_topk-{il}"
        k_w = f"ffn_moe_weights-{il}"
        k_p = f"ffn_moe_probs-{il}"
        if k_ids not in ref or k_ids not in dump:
            continue
        ids_ref = ref[k_ids].astype(np.int64)                     # [nt, 10]
        ids_dump = np.squeeze(dump[k_ids]).astype(np.int64)       # [nt, 10]
        w_ref = np.squeeze(ref[k_w]).astype(np.float64)           # [nt, 10]
        w_dump = np.squeeze(dump[k_w]).astype(np.float64)
        p_dump = np.squeeze(dump[k_p]).astype(np.float64)         # [nt, 512]
        same = direct = 0
        dw = 0.0
        for t in range(ids_ref.shape[0]):
            ids_llama = []
            ok = True
            for x in w_dump[t]:
                j = int(np.argmin(np.abs(p_dump[t] - x)))
                if abs(p_dump[t][j] - x) > 1e-6 * max(1.0, abs(x)):
                    ok = False
                    break
                ids_llama.append(j)
            if not ok:
                ids_llama = [int(i) for i in ids_dump[t]]
            if set(ids_llama) == set(int(i) for i in ids_ref[t]):
                same += 1
                a = {int(i): float(w) for i, w in zip(ids_ref[t], w_ref[t])}
                b = {int(i): float(w) for i, w in zip(ids_llama, w_dump[t])}
                for i in a:
                    dw = max(dw, abs(a[i] - b[i]))
            if set(int(i) for i in ids_dump[t]) == set(int(i) for i in ids_ref[t]):
                direct += 1
        slots_total += ids_ref.shape[0]
        slots_same += same
        slots_direct += direct
        dw_all = max(dw_all, dw)
        print(f"    layer {il}: top-10 sets identical {same}/{ids_ref.shape[0]} tokens "
              f"(direct dump tensor {direct}/{ids_ref.shape[0]}), max |dweight| {dw:.3e}")
    if slots_total:
        print(f"    overall: {slots_same}/{slots_total} = {100.0 * slots_same / slots_total:.2f}% "
              f"(direct ids {slots_direct}/{slots_total}), max |dweight| {dw_all:.3e}")
    print(f"worst relL2 {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
