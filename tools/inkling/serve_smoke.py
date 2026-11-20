#!/usr/bin/env python3
"""serve_smoke.py <engine> <gguf> [ctx] - xolotl-inkling serve protocol: READY, GEN, append reuse, an 8K prefill."""
import subprocess, sys, time
eng, gguf = sys.argv[1], sys.argv[2]
ctx = sys.argv[3] if len(sys.argv) > 3 else "16384"
extra = sys.argv[4:]
p = subprocess.Popen([eng, "--serve", "--native", gguf, "--context", ctx] + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def until(pref):
    out = []
    while True:
        l = p.stdout.readline()
        if not l: raise SystemExit("engine died: " + "".join(out[-5:]))
        out.append(l)
        if l.startswith(pref) or l.startswith("ERR"): return out
t = time.time(); r = until("READY"); print(f"load {time.time()-t:.0f}s:", r[-2].strip()[:200], "|", r[-1].strip())
ids = open("/media/mal/NVME1TB/Xolotl/inkling-2-mini/golden/p1/ids.txt").read().strip()
def gen(ids, n, extra=""):
    p.stdin.write(f"GEN {n} {extra} {ids}\n"); p.stdin.flush(); o = until("DONE")
    toks = [l.split()[1] for l in o if l.startswith("T ")]; return toks, [l for l in o if l.startswith(("PP", "DONE"))]
toks, info = gen(ids, 48)
print("gen 1:", info[-1].strip())
toks2, info2 = gen(ids + "," + ",".join(toks[:-1]) + ",13", 16)   # appends: reuse = prompt + generated
print("append:", info2[-1].strip())
long = ",".join((ids.split(",") * 400)[:8192])
toks3, info3 = gen(long, 32)
print("8K prefill:", info3[-2].strip(), "|", info3[-1].strip())
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(30)
