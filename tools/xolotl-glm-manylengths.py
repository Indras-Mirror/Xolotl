#!/usr/bin/env python3
"""Fill the GLM engine's graph cache: one-shot prompts of many DISTINCT lengths (no history, so every request is its
own chunk size n).  ensure_n() caches a graph set per n in the one fixed 512 MiB graph context, so this reaches the
2026-10-22 crash ("not enough space in the context's memory pool ... available 536870912" then exit 139) on the build
before the fix, and the "graph cache rebuilt" path on the fixed one.

Usage: xolotl-glm-manylengths.py [--url http://127.0.0.1:8081] [--count 40] [--lo 60] [--hi 3000]
"""
import argparse, json, sys, time, urllib.error, urllib.request

SENT = "The expert tier decides whether a blob is resident in VRAM or streamed from the model files. "

def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            return json.load(r)
    except Exception as e:
        return {"loaded": None, "error": str(e)}

def ask(url, text, max_tokens=8):
    body = json.dumps({"model": "glm-5.3-flash-maya-s",
                       "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        return True, round(time.time() - t0, 1), (d.get("usage") or {}).get("prompt_tokens")
    except urllib.error.HTTPError as e:
        return False, round(time.time() - t0, 1), f"HTTP {e.code}"
    except Exception as e:
        return False, round(time.time() - t0, 1), type(e).__name__

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--lo", type=int, default=60)
    ap.add_argument("--hi", type=int, default=3000)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    if not health(url).get("loaded"):
        print("engine not loaded", flush=True); return 2
    step = max(1, (a.hi - a.lo) // max(1, a.count - 1))
    for i in range(a.count):
        n = a.lo + i * step
        text = (SENT * (n * 4 // len(SENT) + 1))[: n * 4] + "\nReply with: ok"
        ok, secs, tok = ask(url, text)
        loaded = health(url).get("loaded")
        print(f"request {i + 1}/{a.count} (~{n} tokens): ok={ok} {secs}s prompt={tok} engine_loaded={loaded}", flush=True)
        if not ok or not loaded:
            print(f"STOP: the engine died or the request failed at ~{n} tokens", flush=True)
            return 1
    print(f"all {a.count} requests served", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
