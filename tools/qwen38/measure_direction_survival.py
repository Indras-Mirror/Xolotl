#!/usr/bin/env python3
"""How much of the ABLATION DIRECTION survives re-quantising the edited down tensor?

A norm-ratio ("||Yq-Ye|| / ||Ye-Yp||") answers "is the output close", which conflates the
edit with quantisation noise spread over all 2560 output columns. The ablation acts on ONE
direction: the adapter's B is shared across all 512 experts, so every expert's edit points
along the same u. What decides whether the ablation still works is the component of
(Y_quantised - Y_pristine) along u, compared with the edited model's component.

  keep = <(Yq - Yp), u> / <(Ye - Yp), u>    averaged over tokens, per expert, then per layer

keep = 1.00 -> the edit's effect is intact. keep = 0 -> quantisation erased it.
A value near 1 with a large position error is GOOD: the noise is orthogonal to u.

  measure_direction_survival.py --layers 2,16,32 --print
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v41"))
sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
import gguf  # noqa: E402
import q8_ablate as Q  # noqa: E402
import screen as S  # noqa: E402
from gguf import quants  # noqa: E402

MODEL = "/media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf"
DUMPS = "/media/mal/NVME1TB/qwen38/dumps"
ROLLBACK = "/home/mal/qwen38-pristine/edit-rollback"
ADAPTER = "/media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="2,16,32")
    ap.add_argument("--tiers", default="ptq1,q2_0,iq4_nl,nvfp4,mxfp4,q8_0")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out")
    a = ap.parse_args()

    G = S.GGML("/tmp/libgqshim_q4e.so")
    model = S.GGUFModel(MODEL)
    bdir = pathlib.Path(ROLLBACK)
    man = json.loads((bdir / "backup.json").read_text())
    idx = {t["name"]: t for t in man["tensors"]}
    blob = open(bdir / "tensors.bin", "rb")
    adapter = Q.pairs(ADAPTER)
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]

    print("keep = <(Yq-Yp),u> / <(Ye-Yp),u>   along the ablation direction u")
    print("1.00 = the edit's effect is intact; 0 = quantisation erased it\n")
    rows = []
    for L in [int(x) for x in a.layers.split(",") if x.strip()]:
        name = f"blk.{L}.ffn_down_exps.weight"
        if name not in idx or name not in adapter:
            print(f"L{L:02d} skip (not an edited tensor)")
            continue
        B = adapter[name]["b"]
        if B.ndim != 3 or B.shape[2] != 1:
            print(f"L{L:02d} skip (edit is not rank-1; no single direction)")
            continue
        u = B[0][:, 0].astype(np.float64)
        u /= np.linalg.norm(u)
        X, ids = S.read_dump(S.dump_files(DUMPS, L))
        r = idx[name]
        per = r["nbytes"] // 512
        allraw = Q.read_range(blob, r["blob_offset"], r["nbytes"])
        acc = {t: [] for t in tiers if G.row_bytes(t, 640) > 0}
        skipped = [t for t in tiers if G.row_bytes(t, 640) <= 0]
        nT = min(a.tokens, X.shape[0])
        U = ids[:nT]
        for e in np.unique(U):
            sel = np.where(U == e)[0]
            Xs = X[sel]
            Wg = model.expert(L, int(e), "gate_proj").astype(np.float32)
            Wu = model.expert(L, int(e), "up_proj").astype(np.float32)
            We = model.expert(L, int(e), "down_proj").astype(np.float32)
            Wp = quants.dequantize(
                np.frombuffer(allraw[int(e) * per:(int(e) + 1) * per], np.uint8),
                gguf.GGMLQuantizationType.Q8_0).astype(np.float32).reshape(2560, 640)
            H = S.swiglu(Xs @ Wg.T, Xs @ Wu.T)
            Yp = H @ Wp.T
            Ye = H @ We.T
            eu = (Ye - Yp) @ u
            den = float(np.sum(eu * eu))
            if den == 0.0:
                continue
            for t in acc:
                Wq, _ = G.quantize_dequantize(We, t, None)
                got = (H @ Wq.astype(np.float32).T - Yp) @ u
                acc[t].append(float(np.sum(got * eu) / den))
        line = "  ".join(f"{t}={np.mean(v):6.3f}" for t, v in acc.items() if v)
        print(f"L{L:02d}  {line}" + (f"   (skipped, not 640-legal: {skipped})" if skipped else ""))
        rows.append({"layer": L, "keep": {t: float(np.mean(v)) for t, v in acc.items() if v}})
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"schema": 1, "model": MODEL, "metric": "component along ablation direction u",
             "results": rows}, indent=1) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
