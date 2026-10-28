#!/usr/bin/env python3
"""How much of the abliteration survives re-quantising each tensor family?

V4's rule was "keep the sensitive experts near native, squash the redundant ones".
For Qwen4Exp the same question has to be asked per TENSOR FAMILY, because the
ablation is spread over tensors that the expert ladder does not cover:

  ffn_down_exps    routed experts   (has its own protected ladder)
  ffn_down_shexp   shared expert    always active, nothing damps its error
  ssm_out          linear attention (36 of 48 layers)
  attn_output      full attention   (12 of 48)
  ple_value        per-layer embedding path

For each family this reports, for each candidate write type:

  alpha_eff   fraction of the intended edit that survives
  signal_pct  how much of the realised change is the edit rather than quant noise

alpha ~1 means the ablation is intact; alpha ~0 means the tier erased it.
Runs against the PRISTINE copy in the rollback and the EDITED model, so the
answer is measured on the real adapter, not a synthetic perturbation.

  ablation_survival.py --model M.gguf --backup DIR --layers 16 --experts 16
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

FAMILIES = ("ffn_down_exps", "ffn_down_shexp", "ssm_out", "attn_output", "ple_value")
TIERS = ("iq4_nl", "q6_k", "q8_0")


def survival(base, edited, encoded):
    return Q.edit_survival(base, edited, encoded)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backup", required=True, help="tensor_backup.py dir covering the edit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="1,16,32,46")
    ap.add_argument("--experts", type=int, default=16, help="routed experts sampled per layer")
    ap.add_argument("--tiers", default=",".join(TIERS))
    a = ap.parse_args()

    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    G = S.GGML()
    man = json.loads((pathlib.Path(a.backup) / "backup.json").read_text())
    blob = open(pathlib.Path(a.backup) / "tensors.bin", "rb")
    idx = {t["name"]: t for t in man["tensors"]}
    reader = gguf.GGUFReader(a.model)
    ts = {t.name: t for t in reader.tensors}

    def pristine(name):
        r = idx[name]
        return Q.read_range(blob, r["blob_offset"], r["nbytes"])

    results = []
    for L in [int(x) for x in a.layers.split(",") if x.strip()]:
        for fam in FAMILIES:
            name = f"blk.{L}.{fam}.weight"
            if name not in idx:
                continue
            t = ts[name]
            shape = tuple(int(x) for x in t.shape)
            ed = Q.q8_rows(t)                       # edited, dequantised
            pr = Q.q8_rows(Q._BackupTensor(t, pristine(name)))   # pristine
            stacked = ed.ndim == 3
            exps = range(shape[2]) if stacked else [None]
            if stacked and a.experts < len(exps):
                pick = np.linspace(0, shape[2] - 1, a.experts).astype(int)
                exps = sorted(set(int(x) for x in pick))
            for tier in tiers:
                # A type whose block size does not divide the row length cannot be
                # written for this tensor at all (640 is not a multiple of 256).
                cols = int(shape[0])
                if tier != "ptq1" and G.row_bytes(tier, cols) <= 0:
                    print(f"L{L:02d} {fam:15s} {tier:7s} -- not representable at row {cols}",
                          flush=True)
                    continue
                acc = []
                for e in exps:
                    b = pr[e] if stacked else pr
                    x = ed[e] if stacked else ed
                    q, _ = G.quantize_dequantize(x.astype(np.float32), tier, None)
                    acc.append(survival(b.astype(np.float64), x.astype(np.float64),
                                        q.astype(np.float64)))
                al = float(np.mean([z["alpha_eff"] for z in acc]))
                sg = float(np.mean([z["signal_pct"] for z in acc]))
                rq = float(np.mean([z["requant_error_rel"] for z in acc]))
                edit_rel = float(np.mean([
                    np.linalg.norm((ed[e] if stacked else ed) - (pr[e] if stacked else pr))
                    / (np.linalg.norm(pr[e] if stacked else pr) or 1.0) for e in exps]))
                results.append({"layer": L, "family": fam, "tier": tier,
                                "n_sampled": len(list(exps)), "edit_rel": edit_rel,
                                "alpha_eff": al, "signal_pct": sg, "requant_error_rel": rq})
                print(f"L{L:02d} {fam:15s} {tier:7s} edit={edit_rel:.3%} "
                      f"alpha={al:.4f} signal={sg:6.2f}% requant={rq:.3%}", flush=True)
    pathlib.Path(a.out).write_text(json.dumps({"schema": 1, "model": a.model,
                                               "backup": a.backup,
                                               "layers": a.layers, "tiers": tiers,
                                               "results": results}, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
