#!/usr/bin/env python3
"""tools/v41/ternary/mesh_screen.py - meshapplied's ternary experts vs OUR shipped GSQ experts, on real FFN inputs.

Drives v41_calib.cmd_layer in --screen-only mode on a work dir that already holds a layer's input stream (default the
FP8 teacher's: /media/Crucial1TB/v41-teacher, h/L00..L40, 639 tokens - so ANY layer can be screened without
replaying the chain).  For every expert the routing reaches, on those tokens (fp32, routing weight and SwiGLU clamps
included, no act quant = weight damage only, as v41_calib.screen_expert):
  Y    = f(x; FP4 base)                                   the reference (base checkpoint experts)
  gsq  = f(x; our GSQ expert, read from V41-GSQ.gguf)      what we ship today
  mesh = f(x; meshapplied ternary, de-rotated)            if the pack has this expert as ternary (else n/a)
Errors ||f - Y||^2, whole expert and per projection group (gate+up swapped alone, down alone), plus ||Y||^2 and our
types (gu_type / d_type: 143 PTQ1_0, 42 Q2_0, 11 Q3_K).  Output: WORK/stats/LNN_mesh.json; `summary` aggregates.

The question it answers: on the experts we made TERNARY (PTQ1_0, same bytes as a repacked mesh expert) is mesh's
error lower -> a free quality swap; on the experts we made Q2_0 / Q3_K, is mesh ternary close enough that moving
them to ternary (2.25 / 3.44 -> 1.75 bpw) is worth the bytes.

    V41_WORK=/media/Crucial1TB/v41-teacher python3 mesh_screen.py layer 5 --pack /media/mal/NVME1TB/mesh-ternary
    python3 mesh_screen.py summary [--work DIR] [--layers 0-39]
GPU (the attention half + fp32 expert products); run it through tools/ds4/memguard.sh, never beside the engine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

GGUF = "/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/V41-GSQ.gguf"
TYPE_NAME = {143: "ptq1", 42: "q2_0", 11: "q3_k"}
BPW = {143: 1.75, 42: 2.25, 11: 3.4375}


class GsqExperts:
    """Our shipped experts, read straight from the GGUF at the offsets V41-GSQ.gguf.experts.json records."""

    def __init__(self, gguf=GGUF):
        import v41_formats as F
        self.F = F
        self.f = open(gguf, "rb")
        self.idx = json.load(open(gguf + ".experts.json"))["layers"]

    def types(self, L, e):
        lay = self.idx[str(L)]
        return lay["gu_type"][e], lay["d_type"][e]

    def proj(self, L, e, role):
        """role gate/up -> [2304, 5120], down -> [5120, 2304] float32."""
        F = self.F
        gu, dt = self.types(L, e)
        t = dt if role == "down" else gu
        rows, n = (5120, 2304) if role == "down" else (2304, 5120)
        nb = F.nbytes(t, (n, rows))
        off = self.idx[str(L)]["off"][str(e)][role]
        self.f.seek(off)
        raw = self.f.read(nb)
        return F.decode(t, raw, n * rows).reshape(rows, n)

    def weights(self, L, e):
        return self.proj(L, e, "gate"), self.proj(L, e, "up"), self.proj(L, e, "down")


def cmd_layer(a):
    import torch
    import v41_calib as C
    from mesh_pack import MeshLayer
    if a.work:
        C.WORK = a.work
    L = a.layer
    mesh = MeshLayer(a.pack, L)
    gsq = GsqExperts(a.gguf)
    dev = a.device
    rec = {}

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.float32)

    def hook(L_, ex, expert, xe, ge, fit, lim, out):
        W1 = C.K.dequant_fp4_weight(expert.w1.weight, expert.w1.scale).float()
        W3 = C.K.dequant_fp4_weight(expert.w3.weight, expert.w3.scale).float()
        W2 = C.K.dequant_fp4_weight(expert.w2.weight, expert.w2.scale).float()
        Y = C.expert_f(xe, W1, W3, W2, ge, lim)
        err = lambda q1, q3, q2: C.sq_err(xe, q1, q3, q2, ge, lim, Y)   # noqa: E731
        gu, dt = gsq.types(L_, ex)
        g1, g3, g2 = (T(w) for w in gsq.weights(L_, ex))
        st = {"n": int(len(xe)), "y": float(Y.square().sum()), "gu_type": gu, "d_type": dt,
              "gsq": err(g1, g3, g2), "gsq_gu": err(g1, g3, W2), "gsq_dn": err(W1, W3, g2)}
        if mesh.has(ex):
            m1, m3, m2 = (T(w) for w in mesh.derotated(ex))
            st.update(mesh=err(m1, m3, m2), mesh_gu=err(m1, m3, W2), mesh_dn=err(W1, W3, m2))
        rec[ex] = st
        return out, {}

    a.expert_hook = hook
    a.moe_done_hook = None
    a.screen_only = "mesh_raw"      # cmd_layer writes stats/LNN_mesh_raw.json and returns before the chain output
    a.no_gptq, a.gptq_experts, a.min_fit, a.damps, a.expert_stride, a.gc = True, 0, 32, [1.0], 1, False
    t0 = time.time()
    C.cmd_layer(a)
    out = os.path.join(C.WORK, "stats", f"L{L:02d}_mesh.json")
    json.dump({"layer": L, "pack": a.pack, "gguf": a.gguf, "seconds": round(time.time() - t0, 1),
               "n_mesh_ternary": len(mesh.ternary), "experts": rec}, open(out, "w"))
    print(f"layer {L}: {len(rec)} experts screened, {sum('mesh' in r for r in rec.values())} with a mesh copy -> {out}")


def parse_layers(s):
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def cmd_summary(a):
    """Per layer and overall: relative error (sum err / sum ||Y||^2) of ours vs mesh, split by OUR type, for the
    experts both have.  gate+up and down separately (each projection group has its own type)."""
    tot = {}
    for L in parse_layers(a.layers):
        p = os.path.join(a.work, "stats", f"L{L:02d}_mesh.json")
        if not os.path.exists(p):
            continue
        ex = json.load(open(p))["experts"]
        row = []
        for grp, tkey in (("gu", "gu_type"), ("dn", "d_type")):
            for t in (143, 42, 11):
                sel = [r for r in ex.values() if r.get(tkey) == t and f"mesh_{grp}" in r and r["n"] > 0]
                if not sel:
                    continue
                y = sum(r["y"] for r in sel)
                g = sum(r[f"gsq_{grp}"] for r in sel) / y
                m = sum(r[f"mesh_{grp}"] for r in sel) / y
                k = (grp, t)
                acc = tot.setdefault(k, [0.0, 0.0, 0.0, 0, 0])
                acc[0] += sum(r[f"gsq_{grp}"] for r in sel); acc[1] += sum(r[f"mesh_{grp}"] for r in sel)
                acc[2] += y; acc[3] += len(sel); acc[4] += sum(r[f"mesh_{grp}"] < r[f"gsq_{grp}"] for r in sel)
                row.append(f"{grp}:{TYPE_NAME[t]} n={len(sel)} ours {g:.4f} mesh {m:.4f} ({m / g:.2f}x)")
        print(f"L{L:02d}  " + " | ".join(row))
    print("\nALL layers (relative output error = sum ||f - Y||^2 / sum ||Y||^2; mesh/ours < 1 = mesh better):")
    for (grp, t), (g, m, y, n, wins) in sorted(tot.items()):
        print(f"  {'gate+up' if grp == 'gu' else 'down   '} where ours is {TYPE_NAME[t]:5s} ({BPW[t]} bpw): n={n:5d}  "
              f"ours {g / y:.4f}  mesh(1.75 bpw) {m / y:.4f}  ratio {m / g:.3f}  mesh wins {100 * wins / n:.0f}% of experts")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("layer")
    p.add_argument("layer", type=int)
    p.add_argument("--pack", required=True, help="dir with layer-LL.safetensors (+ manifest-layer-LL.json)")
    p.add_argument("--gguf", default=GGUF)
    p.add_argument("--work", default=os.environ.get("V41_WORK", "/media/Crucial1TB/v41-teacher"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=0)
    p = sub.add_parser("summary")
    p.add_argument("--work", default=os.environ.get("V41_WORK", "/media/Crucial1TB/v41-teacher"))
    p.add_argument("--layers", default="0-39")
    a = ap.parse_args()
    {"layer": cmd_layer, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
