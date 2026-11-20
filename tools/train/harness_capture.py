#!/usr/bin/env python3
"""Capture the served QFN's real activations for harness windows (the trainer's inputs).

usage: harness_capture.py <dataset-dir> <out-dir> [--n 50] [--seed 0] [--min-loss 256] [--keep-all K]

For each chosen window: tokens = prefix (20,135, shared) + window + one extra token (a native pack does not prefill
the prompt's last token - it is the first verify window), one `strata` run with the SERVED rope (YaRN x2, orig 262144)
and three taps:
  STRATA_KVA_CAPTURE (bf16), split 44      -> L44.bin: the residual entering layer 44, every position
  STRATA_QSA_DUMP, layer index 11 (= blk 47), last <window> positions -> qsa.bin: layer 47's selected cells per query
  STRATA_PREFILL_DUMP_R_ALL                -> final.bf16 (kept for the first K windows only: gate data)
Kept per window: <out>/w<i>/{x44.bf16 (window rows only), qsa.bin, meta.json}; the PREFIX rows once, from the first
window, in <out>/prefix_x44.bf16 (later windows' prefix rows are compared against it and must agree).
Needs the 4090 (~10-20 GB): announce on the relay first.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]   # the Xolotl checkout
NATIVE = os.environ.get("QFN_NATIVE") or sys.exit("set QFN_NATIVE (the QFN native GGUF, shard 1)")
PACK = os.environ.get("QFN_PACK") or sys.exit("set QFN_PACK (the QFN pack directory)")
ROW = 4 * 2560                     # hc * n_embd
QSA_LAYER_INDEX = 11               # blk 47 is the 12th full-attention layer ((il + 1) % 4 == 0)
PAD_TOKEN = 248046                 # any token: it is the verify window, never prefilled


def run(tokens, wdir: Path, n_window: int, keep_final: bool):
    wdir.mkdir(parents=True, exist_ok=True)
    tok_file = wdir / "tokens.txt"
    tok_file.write_text(",".join(str(t) for t in tokens))
    cap = wdir / "cap"
    cap.mkdir(exist_ok=True)
    for f in cap.glob("*"):
        f.unlink()
    qsa = wdir / "qsa.bin"
    if qsa.exists():
        qsa.unlink()
    env = dict(os.environ,
               STRATA_KVA_CAPTURE=str(cap), STRATA_KVA_SPLIT="44",
               STRATA_QSA_DUMP=str(qsa), STRATA_QSA_DUMP_LAYER=str(QSA_LAYER_INDEX),
               STRATA_QSA_DUMP_LAST=str(n_window),
               STRATA_PREFILL_CPU_SHARE="0", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    if keep_final:
        env["STRATA_PREFILL_DUMP_R_ALL"] = str(wdir / "final.bf16")
    args = [str(R / "build-xolotl/strata"), "--pack", PACK, "--native", NATIVE, "--ple-gguf", NATIVE, "--ple-io", "direct",
            "--expert-profile", str(R / "data/expert-profile.bin"), "--expert-cache", "auto",
            "--prefill", os.environ.get("HCAP_PREFILL", "1024"), "--spec", "2", "--max-context", "32768", "--kv", "int8",
            "--rope-scaling", "yarn", "--rope-scale", "2", "--yarn-orig-ctx", "262144",
            "--tokens-file", str(tok_file), "--max-new", "1", "--greedy"]
    t0 = time.time()
    with open(wdir / "strata.log", "w") as log:
        rc = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(R)).returncode
    return rc, time.time() - t0, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-loss", type=int, default=256)
    ap.add_argument("--keep-all", type=int, default=2, help="keep L45-47 + final for the first K windows (gates)")
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = json.loads((ds / "prefix.json").read_text())["tokens"]
    P = len(prefix)
    lines = (ds / "windows.jsonl").read_text().splitlines()
    rng = random.Random(a.seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    chosen = []
    for i in order:
        w = json.loads(lines[i])
        if sum(w["mask"]) >= a.min_loss:
            chosen.append((i, w))
        if len(chosen) >= a.n:
            break
    pref_path = out / "prefix_x44.bf16"
    done = 0
    for k, (i, w) in enumerate(chosen):
        wdir = out / f"w{i:05d}"
        if (wdir / "meta.json").exists():
            done += 1
            continue
        n_win = len(w["tokens"])
        toks = prefix + w["tokens"] + [PAD_TOKEN]
        rc, secs, cap = run(toks, wdir, n_win, keep_final=k < a.keep_all)
        x44 = cap / "L44.bin"
        if rc != 0 or not x44.exists():
            print(f"[cap] window {i}: strata rc {rc}, see {wdir}/strata.log", flush=True)
            continue
        raw = np.fromfile(x44, dtype=np.uint16)
        n_pos = raw.size // ROW
        if n_pos != P + n_win:
            print(f"[cap] window {i}: {n_pos} positions captured, expected {P + n_win}", flush=True)
            continue
        rows = raw.reshape(n_pos, ROW)
        if not pref_path.exists():
            rows[:P].tofile(pref_path)
            pdiff = 0.0
        else:
            ref = np.fromfile(pref_path, dtype=np.uint16).reshape(P, ROW)
            f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
            pdiff = float(np.abs(f(rows[:P]) - f(ref)).max())
        rows[P:].tofile(wdir / "x44.bf16")
        if k >= a.keep_all:
            for f in cap.glob("*"):
                f.unlink()
        else:   # gate data: the window rows of L45..L47 (+ final.bf16 above)
            for L in (45, 46, 47):
                p = cap / f"L{L}.bin"
                if p.exists():
                    r = np.fromfile(p, dtype=np.uint16).reshape(-1, ROW)
                    r[P:].tofile(wdir / f"x{L}.bf16")
                    p.unlink()
            fin = wdir / "final.bf16"
            if fin.exists():
                r = np.fromfile(fin, dtype=np.uint16).reshape(-1, ROW)
                r[P:].tofile(fin)
            x44.unlink(missing_ok=True)
        qn = (wdir / "qsa.bin").stat().st_size if (wdir / "qsa.bin").exists() else 0
        (wdir / "meta.json").write_text(json.dumps({"window": i, "conv": w["conv"], "n": n_win, "loss": sum(w["mask"]),
                                                    "secs": round(secs, 1), "prefix_max_abs_diff": pdiff,
                                                    "qsa_bytes": qn}))
        (wdir / "tokens.json").write_text(json.dumps({"tokens": w["tokens"], "mask": w["mask"]}))
        (wdir / "tokens.txt").unlink(missing_ok=True)
        done += 1
        pf = [l for l in (wdir / "strata.log").read_text().splitlines() if "prefill" in l and "tokens in" in l]
        print(f"[cap] {done}/{len(chosen)} window {i}: {n_win} tokens, {sum(w['mask'])} loss, {secs:.0f} s, "
              f"prefix diff {pdiff:.2e}, qsa {qn} B | {pf[0].strip() if pf else ''}", flush=True)


if __name__ == "__main__":
    main()
