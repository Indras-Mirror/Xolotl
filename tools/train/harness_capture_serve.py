#!/usr/bin/env python3
"""Capture harness windows through ONE `strata --serve` engine (the fast path of harness_capture.py).

usage: harness_capture_serve.py <dataset-dir> <out-dir> [--n 50] [--seed 0] [--min-loss 256] [--keep-all K]
                                [--strata BIN] [--skip N]

harness_capture.py starts one engine per window and re-reads the whole 20,135-token Quetza prefix each time
(~35 s load + ~25 s prefill for ~1.5K new tokens). Here the engine loads once; serve's root checkpoint (a prompt read
from token 0 is checkpointed at its first turn boundary = the end of the system prompt = exactly the prefix) lets every
later request read only its window. Needs an engine with STRATA_KVA_CAPTURE_PER_REQ (lora-train, src/prefill/prefill.cpp):
each top-level read writes <cap>/r<seq>/{L44..L47.bin from ITS pos0, p0.txt, qsa.bin, final.bf16}.

Same window choice (seed, --min-loss) and the same output layout as harness_capture.py, so the trainer reads either:
<out>/w<i>/{x44.bf16, qsa.bin, tokens.json, meta.json} (+ x45/x46/x47/final.bf16 for the first K: gate data),
<out>/prefix_x44.bf16 once. Rows before the prefix end that a request did not read are never needed (the trainer
carries the prefix itself); a request whose read starts PAST the prefix end (a longer cached match) is reported and
skipped, never patched up.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]   # the Xolotl checkout
NATIVE = os.environ.get("QFN_NATIVE") or sys.exit("set QFN_NATIVE (the QFN native GGUF, shard 1)")
PACK = os.environ.get("QFN_PACK") or sys.exit("set QFN_PACK (the QFN pack directory)")
ROW = 4 * 2560
QSA_LAYER_INDEX = 11
PAD_TOKEN = 248046
STRATA = os.environ.get("STRATA_BIN", str(R / "build-xolotl/strata"))


def choose(ds: Path, n: int, seed: int, min_loss: int):   # == harness_capture.py's choice
    lines = (ds / "windows.jsonl").read_text().splitlines()
    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    chosen = []
    for i in order:
        w = json.loads(lines[i])
        if sum(w["mask"]) >= min_loss:
            chosen.append((i, w))
        if len(chosen) >= n:
            break
    return chosen


class Engine:
    def __init__(self, exe: str, cap: Path, keep_final: bool, log: Path):
        env = dict(os.environ,
                   STRATA_KVA_CAPTURE=str(cap), STRATA_KVA_CAPTURE_PER_REQ="1", STRATA_KVA_SPLIT="44",
                   STRATA_QSA_DUMP=str(cap / "unused-qsa.bin"), STRATA_QSA_DUMP_LAYER=str(QSA_LAYER_INDEX),
                   STRATA_QSA_DUMP_LAST="16384",   # every chunk of a window read (records carry absolute positions)
                   STRATA_PREFILL_CPU_SHARE="0", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        if keep_final:
            env["STRATA_PREFILL_DUMP_R_ALL"] = str(cap / "unused-final.bf16")
        args = [exe, "--serve", "--pack", PACK, "--native", NATIVE, "--ple-gguf", NATIVE, "--ple-io", "direct",
                "--expert-profile", str(R / "data/expert-profile.bin"), "--expert-cache", "auto",
                "--prefill", os.environ.get("HCAP_PREFILL", "1024"), "--spec", "2", "--max-context", "32768",
                "--kv", "int8", "--rope-scaling", "yarn", "--rope-scale", "2", "--yarn-orig-ctx", "262144",
                "--prompt-cache", "4"]
        self.log = open(log, "w")
        self.p = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  text=True, bufsize=1, cwd=str(R))
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited before READY (see {log})")
            self.log.write("<< " + line)
            if line.startswith("READY"):
                break

    def gen(self, ids):
        self.p.stdin.write("GEN 1 " + ",".join(str(int(t)) for t in ids) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died mid-request")
            if line.startswith("DONE"):
                return line.strip()
            if line.startswith("ERR"):
                raise RuntimeError(line.strip())

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.log.close()


def assemble(reads, layer_file: str, lo: int, hi: int):
    """Rows [lo, hi) of one per-layer capture from the request's read directories (each from its own pos0)."""
    out = np.zeros((hi - lo, ROW), dtype=np.uint16)
    have = np.zeros(hi - lo, dtype=bool)
    for d, p0 in reads:
        f = d / layer_file
        if not f.exists():
            continue
        r = np.fromfile(f, dtype=np.uint16).reshape(-1, ROW)
        a, b = max(lo, p0), min(hi, p0 + len(r))
        if a < b:
            out[a - lo:b - lo] = r[a - p0:b - p0]
            have[a - lo:b - lo] = True
    return out, bool(have.all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-loss", type=int, default=256)
    ap.add_argument("--keep-all", type=int, default=2)
    ap.add_argument("--skip", type=int, default=0, help="start at the k-th chosen window")
    ap.add_argument("--strata", default=STRATA)
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = json.loads((ds / "prefix.json").read_text())["tokens"]
    P = len(prefix)
    chosen = choose(ds, a.n, a.seed, a.min_loss)
    todo = [(k, i, w) for k, (i, w) in enumerate(chosen) if k >= a.skip and not (out / f"w{i:05d}" / "meta.json").exists()]
    print(f"[serve-cap] {len(chosen)} chosen, {len(todo)} to capture, prefix {P}", flush=True)
    if not todo:
        return
    cap = out / "_serve"
    shutil.rmtree(cap, ignore_errors=True)
    cap.mkdir()
    pref_path = out / "prefix_x44.bf16"
    t0 = time.time()
    eng = Engine(a.strata, cap, keep_final=a.keep_all > 0, log=out / "serve-engine.log")
    print(f"[serve-cap] engine READY in {time.time() - t0:.0f} s", flush=True)
    done = 0
    f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    try:
        for k, i, w in todo:
            wdir = out / f"w{i:05d}"
            wdir.mkdir(exist_ok=True)
            n_win = len(w["tokens"])
            before = {d.name for d in cap.glob("r*")}
            ts = time.time()
            eng.gen(prefix + w["tokens"] + [PAD_TOKEN])
            secs = time.time() - ts
            reads = []
            for d in sorted(cap.glob("r*")):
                if d.name in before:
                    continue
                p0 = int((d / "p0.txt").read_text().split()[0])
                reads.append((d, p0))
            starts = [p0 for _, p0 in reads]
            x44, ok = assemble(reads, "L44.bin", P, P + n_win)
            if not ok:
                print(f"[serve-cap] window {i}: reads start at {starts}, window rows [{P}, {P + n_win}) not all "
                      f"captured - skipped", flush=True)
                for d, _ in reads:
                    shutil.rmtree(d, ignore_errors=True)
                continue
            pdiff = None
            if min(starts) == 0:   # a cold read: the prefix rows too (stored once, or checked against the store)
                pre, pok = assemble(reads, "L44.bin", 0, P)
                if pok:
                    if not pref_path.exists():
                        pre.tofile(pref_path)
                        pdiff = 0.0
                    else:
                        ref = np.fromfile(pref_path, dtype=np.uint16).reshape(P, ROW)
                        pdiff = float(np.abs(f32(pre) - f32(ref)).max())
            x44.tofile(wdir / "x44.bf16")
            with open(wdir / "qsa.bin", "wb") as q:
                for d, _ in reads:
                    if (d / "qsa.bin").exists():
                        q.write((d / "qsa.bin").read_bytes())
            if k < a.keep_all:
                for L in (45, 46, 47):
                    r, lok = assemble(reads, f"L{L}.bin", P, P + n_win)
                    if lok:
                        r.tofile(wdir / f"x{L}.bf16")
                # final.bf16 rows are appended per read from ITS pos0, like the layer files
                r, fok = assemble(reads, "final.bf16", P, P + n_win)
                if fok:
                    r.tofile(wdir / "final.bf16")
            for d, _ in reads:
                shutil.rmtree(d, ignore_errors=True)
            qn = (wdir / "qsa.bin").stat().st_size
            (wdir / "meta.json").write_text(json.dumps({"window": i, "conv": w["conv"], "n": n_win, "loss": sum(w["mask"]),
                                                        "secs": round(secs, 2), "via": "serve", "read_starts": starts,
                                                        "prefix_max_abs_diff": pdiff, "qsa_bytes": qn}))
            (wdir / "tokens.json").write_text(json.dumps({"tokens": w["tokens"], "mask": w["mask"]}))
            done += 1
            print(f"[serve-cap] {done}/{len(todo)} window {i}: {n_win} tokens, {sum(w['mask'])} loss, {secs:.1f} s, "
                  f"reads from {starts}" + (f", prefix diff {pdiff:.2e}" if pdiff is not None else ""), flush=True)
    finally:
        eng.close()
        for f in cap.glob("unused-*"):
            f.unlink()
    print(f"[serve-cap] {done} windows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
