#!/usr/bin/env python3
"""Drive `ds41_generate --serve` (through ds41-engine.sh: GPU lock, RAM cap) over its line protocol - speed and
exactness checks of the serve path without the HTTP server.

  drive_engine.py OUT.json LOG CTX REQS.json [extra engine args...]
  env DRIVE_ENV="K=V K=V"   extra environment for the engine (e.g. STRATA_SERVE_PASS_BIAS=0 STRATA_SERVE_DUMP=dir)

REQS.json: a list of {"file": ids.i32 [, "len": N] | "ids": [...] | "extend": [j, [extra ids]], "n": max_new
[, "stop_after": k]}; "extend" = request j's prompt + its generated tokens + the extra ids.  Greedy (temperature 0).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

DS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
G = "/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/V41-GSQ.gguf"
EN = "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"
R = "/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/rank.bin"

out_json, log, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
reqs = json.load(open(sys.argv[4]))
cmd = [f"{DS}/tools/v41/serve/ds41-engine.sh", "-m", G, "--backend", "cuda", "--experts", "gpu", "--engram", EN,
       "--rank", R, "--ctx", ctx, "--serve"] + sys.argv[5:]
env = dict(os.environ, STRATA_NO_LARGEPAGES="1")
for kv in os.environ.get("DRIVE_ENV", "").split():
    k, v = kv.split("=", 1)
    env[k] = v
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, env=env,
                     bufsize=1)
t0 = time.time()
while True:
    ln = p.stdout.readline()
    if not ln:
        sys.exit("engine died before READY (see log)")
    if ln.startswith("READY"):
        break
print(f"ready in {time.time() - t0:.0f} s", flush=True)
res = []
for i, r in enumerate(reqs):
    if "extend" in r:
        j, ext = r["extend"]
        ids = res[j]["prompt"] + res[j]["tokens"] + ext
    elif "file" in r:
        ids = np.fromfile(r["file"], np.int32)[: r.get("len", None)].tolist()
    else:
        ids = list(r["ids"])
    t = time.time()
    p.stdin.write(f"GEN {r['n']} temperature=0 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    toks, info = [], {}
    while True:
        ln = p.stdout.readline()
        if not ln:
            sys.exit("engine died mid-request")
        ln = ln.strip()
        if ln.startswith("T "):
            toks.append(int(ln[2:]))
            if r.get("stop_after") and len(toks) == r["stop_after"]:
                p.stdin.write("STOP\n")
                p.stdin.flush()
        elif ln.startswith("RESUME"):
            info["resume"] = int(ln.split()[1])
        elif ln.startswith("DONE") or ln.startswith("ERR"):
            info["end"] = ln
            break
    info["wall"] = round(time.time() - t, 2)
    res.append({"prompt": ids, "tokens": toks, **info})
    print(f"req {i}: prompt {len(ids)} -> {len(toks)} tokens, {info}", flush=True)
json.dump(res, open(out_json, "w"))
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=600)
