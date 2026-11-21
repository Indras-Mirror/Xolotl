#!/usr/bin/env python3
"""parity.py - xolotl-inkling vs the llama.cpp reference golden dumps (S2/S3 gates).

  parity.py <engine binary> <gguf> <golden dir> [prompt ...]   (default prompts: p0 p1 p2)

For each prompt: runs `xolotl-inkling --dump` on golden/<p>/ids.txt, then compares slot by slot in EXECUTION order
(0-5, 6-17, 24-35, 36-47, 18-23) the relative error |ours - ref| / |ref| of l_out, and the logits: KL(ref || ours)
per position, top-1 agreement.  The first slot over --tol is where a bug starts.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ORDER = list(range(0, 18)) + list(range(24, 48)) + list(range(18, 24))
N_EMBD, N_VOCAB = 2304, 200058


def logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine"); ap.add_argument("gguf"); ap.add_argument("golden")
    ap.add_argument("prompts", nargs="*", default=["p0", "p1", "p2"])
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--extra", default="", help="extra engine args, space-separated")
    a = ap.parse_args()
    worst = 0.0
    for p in a.prompts:
        gd = os.path.join(a.golden, p)
        ids = open(os.path.join(gd, "ids.txt")).read().strip()
        n = len(ids.split(","))
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__))) as od:
            cmd = [a.engine, "--dump", a.gguf, "--ids", ids, "--out-dir", od] + (["--gen", str(a.gen)] if a.gen else []) + a.extra.split()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-4000:]); sys.exit(1)
            for ln in r.stderr.splitlines():
                if "tokens in" in ln or "tok/s" in ln: print("  ", ln)
            gen = [ln for ln in r.stdout.splitlines() if ln.startswith("GEN")]
            print(f"== {p}: {n} tokens")
            first_bad = None
            for d, s in enumerate(ORDER if os.path.exists(os.path.join(gd, "l_out-0.f32")) else []):
                ref = np.fromfile(os.path.join(gd, f"l_out-{s}.f32"), dtype=np.float32).reshape(n, N_EMBD)
                our = np.fromfile(os.path.join(od, f"l_out-{s}.f32"), dtype=np.float32).reshape(n, N_EMBD)
                rel = float(np.linalg.norm(our - ref) / (np.linalg.norm(ref) + 1e-12))
                worst = max(worst, rel)
                flag = ""
                if rel > a.tol and first_bad is None:
                    first_bad = s; flag = "  <-- first over tol"
                if d < 6 or d % 6 == 5 or flag or d == len(ORDER) - 1:
                    print(f"  depth {d:2d} slot {s:2d}: rel {rel:.2e}{flag}")
            ref = np.fromfile(os.path.join(gd, "logits.f32"), dtype=np.float32).reshape(n, N_VOCAB).astype(np.float64)
            our = np.fromfile(os.path.join(od, "logits.f32"), dtype=np.float32).reshape(n, N_VOCAB).astype(np.float64)
            lr, lo = logsoftmax(ref), logsoftmax(our)
            kl = (np.exp(lr) * (lr - lo)).sum(-1)
            top = (ref.argmax(-1) == our.argmax(-1)).mean()
            print(f"  logits: KL mean {kl.mean():.2e} p99 {np.percentile(kl, 99):.2e} max {kl.max():.2e}, top-1 {top * 100:.2f} %, last-pos ref top "
                  f"{int(ref[-1].argmax())} ours {int(our[-1].argmax())}")
            if gen: print("  ", gen[0][:400])
    print(f"worst slot rel {worst:.2e}")


if __name__ == "__main__":
    main()
