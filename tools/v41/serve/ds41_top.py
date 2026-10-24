"""ds41_top.py - prefill and decode speed of a running server, in the terminal.

Reads the server's GET /metrics once a second (standard library only).  While a request runs, one status line shows
its phase: the prompt read so far and its rate, then the tokens generated and the decode rate.  When it finishes, a
line per request: prompt tokens (read / reused), prefill time and tok/s, output tokens, decode tok/s, DSpark drafts
accepted, and the expert cache hit rate.  It reads serve/server.py's metrics, so it works for a Qwen server too
(--url http://127.0.0.1:8080).

    python tools/v41/serve/ds41_top.py                    # http://127.0.0.1:8150
    python tools/v41/serve/ds41_top.py --url http://127.0.0.1:8082 --api-key KEY
"""
import argparse
import json
import sys
import time
import urllib.request


def get(url, key):
    req = urllib.request.Request(url + "/metrics", headers={"Accept": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def num(v, fmt="{:.1f}", none="-"):
    return none if v is None else fmt.format(v)


def request_line(q):
    read = q.get("prompt_read") or 0
    pms = q.get("prompt_ms") or 0
    pre = read / (pms / 1000.0) if read and pms else None
    off, acc = q.get("drafts_offered") or 0, q.get("drafts_accepted") or 0
    drafts = f"{acc}/{off} ({100.0 * acc / off:.0f}%)" if off else "-"
    hit = q.get("hit_rate")
    return (f"{time.strftime('%H:%M:%S', time.localtime(q.get('time') or time.time()))}  "
            f"prompt {read:>6} read +{q.get('reused') or 0:>6} reused  "
            f"prefill {pms / 1000.0:6.1f} s {num(pre, '{:6.0f}', '     -')} tok/s  "
            f"out {q.get('output_tokens') or 0:>5}  decode {num(q.get('decode_tok_s'), '{:5.1f}', '    -')} tok/s  "
            f"drafts {drafts:<12} hit {num(hit * 100 if hit is not None else None, '{:.0f}%')}  "
            f"{q.get('finish') or ''}")


def live_line(lv):
    st = lv.get("state") or "?"
    if st == "idle":
        return f"idle  (queued {lv.get('queued') or 0})"
    phase = lv.get("phase") or st
    el = num(lv.get("elapsed_s"), "{:.1f} s")
    if lv.get("prompt_total") and (lv.get("generated") or 0) == 0:
        return (f"{phase}: prompt {lv.get('prompt_read') or 0}/{lv.get('prompt_total')}  "
                f"{num(lv.get('prefill_tok_s_mean'), '{:.0f}')} tok/s  {el}")
    return (f"{phase}: generated {lv.get('generated') or 0}  now {num(lv.get('tok_s'))} tok/s  "
            f"mean {num(lv.get('tok_s_mean'))} tok/s  {el}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8150")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    seen = None   # finish time of the newest request already printed
    width = 0
    print(f"ds41_top: {url}/metrics every {a.interval:g} s (Ctrl+C to stop)")
    while True:
        try:
            m = get(url, a.api_key)
            reqs = sorted(m.get("requests") or [], key=lambda q: q.get("time") or 0)   # the server lists newest first
            if seen is None:   # first poll: show the last few, then only new ones
                new = reqs[-5:]
            else:
                new = [q for q in reqs if (q.get("time") or 0) > seen]
            if reqs:
                seen = max(seen or 0, max(q.get("time") or 0 for q in reqs))
            status = live_line(m.get("live") or {})
        except Exception as e:  # the server restarting or busy: keep polling
            new, status = [], f"no answer from {url} ({type(e).__name__})"
        if new:
            sys.stdout.write("\r" + " " * width + "\r")
            for q in new:
                print(request_line(q))
        sys.stdout.write("\r" + status.ljust(width))
        sys.stdout.flush()
        width = len(status)
        try:
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
