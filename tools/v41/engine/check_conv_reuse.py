#!/usr/bin/env python3
"""Conversation-reuse exactness check for ds41_generate --serve (CPU backend, real model).

Each scenario sends a sequence of GEN / GENI requests to one engine (the engine under test) and the same prompts,
each as a FRESH prefill, to a second engine (park off, a 1-token reset request between prompts).  Every request's
last-row logits (STRATA_SERVE_DUMP) are compared: reuse is exact when the reused request's logits match the fresh
prefill of the same prompt as closely as the batch-shape noise of a known-good continuation does.

  check_conv_reuse.py --bin build/ds41_generate --model M.gguf --engram A:B --tokenizer tokenizer.json \
      --scenario park_partial [--ref-bin B2] [--tail 20]

The prompts are token runs of the repo's README.md / HANDOFF.md / howto_eng.md (--tokenizer: the model's
tokenizer.json, or DS41_TOKENIZER_JSON).

Scenarios: park_partial (the STRATA_SERVE_PARK_PARTIAL resume 20 positions back, then 8 back), park16 (a partial
resume 8 back, within the window ring, then 21 back), engram_hist (a side request longer than the parked
conversation's end overwrites the Engram history), single (one growing text dialogue, a regenerate and a re-render),
abc (three conversations interleaved, two sharing a system prompt: the conversation pool), image (a dialogue with
synthetic SVE1 image rows and img_keys, a text side request, and the same tokens with another picture).
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

IMAGE_TOKEN = 129264
N_EMBD = 5120


class Engine:
    def __init__(self, binary, args, env, workdir, name, cpu=True, serve_flag=True):
        self.dump = Path(workdir) / (name + "-dump")
        self.dump.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(workdir) / (name + ".log")
        self.log = self.log_path.open("w")
        e = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "DS41_"))}
        if cpu:
            e.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", STRATA_ENGRAM_CACHE_ROWS="65536")
        e.update(env)
        e["STRATA_SERVE_DUMP"] = str(self.dump)
        self.p = subprocess.Popen([str(binary), *args] + (["--serve"] if serve_flag else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, env=e, bufsize=1)
        self.n = 0
        t0 = time.time()
        while True:
            ln = self.p.stdout.readline()
            if not ln:
                raise RuntimeError(f"{name}: engine exited at start; see {self.log_path}")
            if ln.startswith("READY"):
                break
        print(f"[{name}] ready in {time.time() - t0:.0f} s", flush=True)
        self.name = name

    def gen(self, ids, max_new=1, keys="", emb=None, stop_after=0):
        head = f"GENI {max_new}" if emb else f"GEN {max_new}"
        line = f"{head} temperature=0{(' ' + keys) if keys else ''} {(emb + ' ') if emb else ''}{','.join(map(str, ids))}\n"
        t0 = time.time()
        self.p.stdin.write(line)
        self.p.stdin.flush()
        out = {"resume": 0, "tokens": []}
        while True:
            ln = self.p.stdout.readline()
            if not ln:
                raise RuntimeError(f"{self.name}: engine died; see {self.log_path}")
            if ln.startswith("RESUME "):
                out["resume"] = int(ln.split()[1])
            elif ln.startswith("T "):
                out["tokens"].append(int(ln.split()[1]))
                if stop_after and len(out["tokens"]) == stop_after:
                    self.p.stdin.write("STOP\n")
                    self.p.stdin.flush()
            elif ln.startswith("ERR"):
                raise RuntimeError(f"{self.name}: {ln.strip()}")
            elif ln.startswith("DONE"):
                out["done"] = ln.split()
                out["prompt_ms"] = float(out["done"][3])   # DONE gen total prompt_ms dec_ms finish ...
                break
        out["sec"] = time.time() - t0
        out["n"] = self.n
        f = self.dump / f"req{self.n:03d}.f32"
        out["logits"] = np.fromfile(f, dtype=np.float32) if f.exists() else None
        self.n += 1
        return out

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.log.close()


NV = 129280


def tail_of(eng, r):
    f = eng.dump / f"req{r['n']:03d}.tail.f32"
    if not f.exists() or f.stat().st_size == 0:
        return None
    return np.fromfile(f, dtype=np.float32).reshape(-1, NV)


def cmp(a, b):
    """max |diff|, top-1 equal, KL(ref || test) in nats."""
    d = float(np.max(np.abs(a - b)))
    pa = np.exp(a - a.max()); pa /= pa.sum()
    lb = b - b.max(); lb = lb - math.log(np.exp(lb).sum())
    la = np.log(pa + 1e-30)
    kl = float(np.sum(pa * (la - lb)))
    return {"maxdiff": round(d, 5), "top1": bool(np.argmax(a) == np.argmax(b)), "kl": float(f"{kl:.3e}"),
            "bitexact": bool(np.array_equal(a, b))}


def corpus(seed_files, tokenizer_json):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    out = []
    for f in seed_files:
        out.append(tok.encode(Path(f).read_text(errors="replace")).ids)
    return out


def write_sve1(path, images):
    """images: list of (nx, ny, rows[n, N_EMBD] float32)."""
    with open(path, "wb") as f:
        for nx, ny, rows in images:
            f.write(struct.pack("<5i", 0x31455653, nx * ny, nx, ny, rows.shape[1]))
            f.write(rows.astype(np.float32).tobytes())


def build_steps(sc, out, tokenizer_json):
    """The scenario's requests: (label, ids, max_new, keys, embeddings file)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    texts = corpus([root / "README.md", root / "HANDOFF.md", root / "howto_eng.md"], tokenizer_json)
    T = [t for t in texts if len(t) >= 1000]
    if len(T) < 3:
        raise SystemExit("corpus too short")
    A, B, C = T[0], T[1], T[2]
    BOS = [0]
    steps = []
    if sc == "park_partial":
        a0 = BOS + A[:239]                         # 240 positions, parked by the side request
        b0 = BOS + B[:63]
        a1 = a0[:220] + A[500:520]                 # continues 220 of the parked 240 (>= 3/4): 20 back (> 16)
        a2 = a0[:232] + A[520:540]                 # continues 232 of A (8 back, <= 16); main: of the re-parked A' (20 back)
        steps = [("A 240", a0, 2, "", None), ("B side", b0, 2, "", None),
                 ("A' lcp 220 of 240 (20 back)", a1, 2, "", None),
                 ("B side 2", b0 + B[63:70], 2, "", None),
                 ("A'' lcp 232 of 240 (8 back)", a2, 2, "", None)]
    elif sc == "park16":
        a0 = BOS + A[:239]
        b0 = BOS + B[:63]
        a1 = a0[:232] + A[500:520]                 # continues 232 of the parked 240: 8 back (<= 16, even)
        a2 = a1[:231] + A[520:540]                 # 21 back from the re-parked 252 (> 16): no partial resume
        steps = [("A 240", a0, 2, "", None), ("B side", b0, 2, "", None),
                 ("A' lcp 232 of 240 (8 back)", a1, 2, "", None), ("B side 2", b0 + B[63:70], 2, "", None),
                 ("A'' lcp 231 of 252 (21 back)", a2, 2, "", None)]
    elif sc == "engram_hist":
        a0 = BOS + A[:239]
        b0 = BOS + B[:259]                         # the side request is LONGER than the parked conversation
        steps = [("A 240", a0, 2, "", None), ("B side 260 (longer)", b0, 2, "", None),
                 ("A + 20 (full unpark)", a0 + A[239:259], 2, "", None)]
    elif sc == "single":
        h = BOS + A[:199]
        for i in range(4):
            h = h + B[i * 30:(i * 30) + 20]
            steps.append((f"turn {i}", list(h), 3, "", None))
            h = h + B[i * 30 + 20:(i + 1) * 30]   # the "assistant reply" as the client re-renders it
        steps.append(("regen", list(steps[-1][1]), 3, "", None))
        steps.append(("re-render 12 back", steps[-1][1][:-12] + C[:16], 3, "", None))
    elif sc == "abc":
        sys_p = BOS + A[:119]
        convs = {"A": sys_p + B[:60], "B": sys_p + C[:60], "C": BOS + C[300:420]}
        src = {"A": B[60:], "B": C[60:], "C": C[420:]}
        k = {"A": 0, "B": 0, "C": 0}
        for c in ["A", "B", "C", "A", "B", "C", "A"]:
            convs[c] = convs[c] + src[c][k[c]:k[c] + 16]
            k[c] += 16
            steps.append((f"{c}{k[c] // 16}", list(convs[c]), 2, "", None))
    elif sc == "image":
        rng = np.random.default_rng(7)
        img1 = rng.standard_normal((8 * 8, N_EMBD)).astype(np.float32) * 0.02
        img2 = rng.standard_normal((6 * 6, N_EMBD)).astype(np.float32) * 0.02
        e1 = out / "img1.sve"; write_sve1(e1, [(8, 8, img1)])
        e12 = out / "img12.sve"; write_sve1(e12, [(8, 8, img1), (6, 6, img2)])
        e2 = out / "img2.sve"; write_sve1(e2, [(8, 8, img2[:36].repeat(2, axis=0)[:64])])
        k1, k2, k3 = "0123456789abcdef", "fedcba9876543210", "00000000deadbeef"
        p0 = BOS + A[:99] + [IMAGE_TOKEN] * 64 + A[100:120]
        p1 = p0 + A[120:136]
        p2 = p1 + A[136:150] + [IMAGE_TOKEN] * 36 + A[150:166]
        q0 = BOS + A[:99] + [IMAGE_TOKEN] * 64 + A[100:120] + A[120:136]   # the same tokens as p1, another picture
        steps = [("img turn 0", p0, 2, f"img_keys={k1}", str(e1)),
                 ("img turn 1 (+16)", p1, 2, f"img_keys={k1}", str(e1)),
                 ("text side", BOS + C[:99], 2, "", None),
                 ("img turn 2 (+2nd pic)", p2, 2, f"img_keys={k1},{k2}", str(e12)),
                 ("same tokens as turn 1, other pic", q0, 2, f"img_keys={k3}", str(e2)),
                 ("img turn 3 (+16)", p2 + A[166:182], 2, f"img_keys={k1},{k2}", str(e12))]
    else:
        raise SystemExit("unknown scenario " + sc)
    return steps



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ref-bin", help="the fresh-prefill engine (default: --bin)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engram", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DS41_TOKENIZER_JSON"),
                    help="the model's tokenizer.json (default: $DS41_TOKENIZER_JSON)")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env", action="append", default=[], help="K=V for the engine under test")
    ap.add_argument("--extra", default="", help="extra engine args (space separated) for both engines")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--no-ref", action="store_true", help="skip the fresh engine (decisions only)")
    ap.add_argument("--ref-all", action="store_true", help="a fresh reference for every step, not only the reused ones")
    ap.add_argument("--tail", type=int, default=0, help="STRATA_SERVE_DUMP_TAIL: compare the last N prompt rows too")
    a = ap.parse_args()
    if not a.tokenizer:
        ap.error("--tokenizer (or DS41_TOKENIZER_JSON) is required")
    out = Path(a.out or tempfile.mkdtemp(prefix="ds41-reuse-"))
    out.mkdir(parents=True, exist_ok=True)
    common = ["-m", a.model, "--engram", a.engram, "--backend", "cpu", "--experts", "cpu", "--arena-gib", "0",
              "--ctx", "8192", "--threads", str(a.threads), "--prompt-batch", "512", "--prefill-chunk", "64",
              "--stop", "1"] + (["--vision"] if a.vision else []) + (a.extra.split() if a.extra else [])
    env = dict(kv.split("=", 1) for kv in a.env)
    env.setdefault("STRATA_SERVE_PARK_MIN", "64")
    tail_env = {"STRATA_SERVE_DUMP_TAIL": str(a.tail)} if a.tail > 0 else {}
    env.update(tail_env)
    sc = a.scenario
    steps = build_steps(sc, out, a.tokenizer)
    eng = Engine(a.bin, common, env, out, "test")
    res = []
    try:
        for label, ids, mx, keys, emb in steps:
            r = eng.gen(ids, mx, keys, emb)
            res.append(r)
            print(f"[test] {label}: prompt {len(ids)} resume {r['resume']} tokens {r['tokens']} {r['sec']:.1f} s", flush=True)
    finally:
        eng.close()
    summary = {"scenario": sc, "bin": str(a.bin), "env": env, "steps": []}
    for (label, ids, *_), r in zip(steps, res):
        summary["steps"].append({"label": label, "prompt": len(ids), "resume": r["resume"], "tokens": r["tokens"],
                                 "sec": round(r["sec"], 1)})
    if not a.no_ref:
        ref = Engine(a.ref_bin or a.bin, common, {"STRATA_SERVE_PARK": "0", **tail_env}, out, "fresh")
        try:
            for (label, ids, mx, keys, emb), r, st in zip(steps, res, summary["steps"]):
                if r["resume"] == 0 and not a.ref_all:
                    continue                         # a fresh prefill in the engine under test too
                ref.gen([1], 1)                      # reset: nothing below is a continuation of the previous prompt
                f = ref.gen(ids, mx, keys, emb)
                if r["logits"] is not None and f["logits"] is not None:
                    st.update(cmp(f["logits"], r["logits"]))
                rt, ft = tail_of(eng, r), tail_of(ref, f)
                if rt is not None and ft is not None:
                    k = min(len(rt), len(ft))
                    rows = [cmp(ft[-k + i], rt[-k + i]) for i in range(k)]
                    st["tail_rows"] = k
                    st["tail_max_kl"] = max(x["kl"] for x in rows)
                    st["tail_first_kl"] = rows[0]["kl"]
                    st["tail_top1"] = sum(x["top1"] for x in rows)
                st.update(fresh_tokens=f["tokens"])
                print(f"[cmp] {label}: {st}", flush=True)
        finally:
            ref.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
