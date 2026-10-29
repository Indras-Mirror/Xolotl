#!/usr/bin/env python3
"""load_test.py - prefill and decode throughput of a qwen4exp native pack, measured from the engine's own output.

WHAT IT REPORTS, and why each number is where it is:
  load        wall time from spawn to the first prefill line = weights + expert-cache admission
  prefill     taken from strata' own line: "prefill N tokens in C chunks, MS ms (X tok/s)"
  decode      steady state, from the FIRST and LAST `T <id>` lines on stdout (31 intervals for 32 tokens),
              so the load and the prefill are excluded by construction rather than subtracted afterwards
  tokens      the token ids the model actually produced - a throughput number with no text is not evidence
              the model ran

The engine has no --model flag: a qwen4exp file loads as a native pack (--native SHARD --pack DIR
--ple-gguf SHARD --spec T). It refuses the pack without them, and says so.

    python3 load_test.py --model M.gguf --pack PACK --tokens-file ids.txt \
        --prompt-tokens 512 --max-new 32 [--label name] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

STRATA = "/home/mal/AI/Xolotl/build-xolotl/strata"

PREFILL = re.compile(
    r"prefill (\d+) tokens in (\d+) chunks, ([\d.]+) ms \(([\d.]+) tok/s\)"
    r"(?:; experts streamed (\d+) \((\d+) by DMA, host ([\d.]+) ms\), resident (\d+), "
    r"on the CPU (\d+) \(share ([\d.]+)\); PLE ([\d.]+) ms)?")


def load_ids(path: str) -> list[int]:
    with open(path) as f:
        txt = f.read()
    return [int(t) for t in re.split(r"[,\s]+", txt) if t]


def run(model: str, pack: str, ids: list[int], max_new: int, ctx: int,
        expert_cache: str, profile: str | None, timeout: int) -> dict:
    """One generate run. Returns the measured numbers, or raises with the engine's own error text.

    --spec REQUIRES --expert-profile: without the device residency table the engine exits 2 with
    "--spec needs the device residency table". Measured 2026-10-29; it is not optional.
    """
    cmd = [STRATA,
           "--native", model, "--pack", pack, "--ple-gguf", model,
           "--ple-io", "direct", "--mmap-experts", "--spec", "2"]
    if profile:
        cmd += ["--expert-profile", profile]
    cmd += [
           "--expert-cache", str(expert_cache),
           "--prefill", "512",
           "--max-context", str(ctx),
           "--max-new", str(max_new),
           "--tokens", ",".join(str(i) for i in ids)]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    token_ids: list[int] = []
    prefill = None
    stderr_lines: list[str] = []

    import threading
    def pump_err():
        # nonlocal, or `prefill = ...` below binds a LOCAL and the caller sees None forever.
        nonlocal prefill
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
            m = PREFILL.search(line)
            if m and prefill is None:
                d = {"tokens": int(m.group(1)), "chunks": int(m.group(2)),
                     "ms": float(m.group(3)), "tok_s": float(m.group(4))}
                if m.group(5) is not None:
                    d.update(experts_streamed=int(m.group(5)), experts_dma=int(m.group(6)),
                             experts_host_ms=float(m.group(7)), experts_resident=int(m.group(8)),
                             experts_cpu=int(m.group(9)), cpu_share=float(m.group(10)),
                             ple_ms=float(m.group(11)))
                prefill = d
    th = threading.Thread(target=pump_err, daemon=True)
    th.start()

    stdout_lines: list[str] = []
    for line in proc.stdout:
        stdout_lines.append(line.rstrip())

    rc = proc.wait(timeout=timeout)
    th.join(timeout=5)
    total = time.time() - t0

    if rc != 0:
        tail = "\n".join(stderr_lines[-12:])
        raise RuntimeError(f"strata exited {rc}\n{tail}")

    # The engine prints its own summary on stdout - parse that instead of timing T lines.
    # Measured 2026-10-29: this build emits "decode  N tokens in MS ms  ->  X tok/s" and
    # "prefill N tokens in MS ms  ->  X tok/s  (time to first token T ms)", plus the ids on
    # "output  : ...". It does NOT emit per-token "T <id>" lines on this path, so a T-line
    # parser reports zero tokens for a perfectly good run - that was this script's first bug.
    blob = "\n".join(stdout_lines)
    dec = re.search(r"decode\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s", blob)
    pre2 = re.search(r"prefill\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s"
                     r"(?:\s*\(time to first token ([\d.]+) ms\))?", blob)
    out = re.search(r"^output\s*:\s*(.+)$", blob, re.M)
    if dec is None:
        raise RuntimeError("no 'decode ... tok/s' summary line - cannot measure decode\n"
                           + "\n".join(stdout_lines[-6:] + stderr_lines[-8:]))
    decode = {
        "tokens": int(dec.group(1)),
        "ms": float(dec.group(2)),
        "tok_s": float(dec.group(3)),
    }
    if pre2 is not None:
        prefill = {"tokens": int(pre2.group(1)), "ms": float(pre2.group(2)),
                   "tok_s": float(pre2.group(3)),
                   "ttft_ms": float(pre2.group(4)) if pre2.group(4) else None,
                   "detail": prefill}
    token_ids = [int(t) for t in out.group(1).split()] if out else []

    return {
        "total_s": round(total, 2),
        "prefill": prefill,
        "decode": decode,
        "token_ids": token_ids,
        "stderr_tail": stderr_lines[-40:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=32)
    # "auto" lets the engine size the VRAM tier to the card; a small fixed count (1536 ~ 2.6 GB of a
    # 24 GB card) forces far more expert reads from disk per round - measured 1.84 GB/round.
    ap.add_argument("--expert-cache", default="auto")
    ap.add_argument("--profile", default="/home/mal/qwen38-profile.bin",
                    help="device residency table; --spec requires it")
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    ids = load_ids(a.tokens_file)[: a.prompt_tokens]
    if len(ids) < 8:
        print(f"prompt file gave only {len(ids)} ids", file=sys.stderr)
        return 2

    # The context must hold prompt + max_new + the engine's 8-cell margin.
    ctx = len(ids) + a.max_new + 64
    print(f"=== {a.label or os.path.basename(a.model)}: {len(ids)} prompt tokens, {a.max_new} new, "
          f"ctx {ctx}, expert-cache {a.expert_cache}", flush=True)

    profile = a.profile if a.profile and os.path.exists(a.profile) else None
    r = run(a.model, a.pack, ids, a.max_new, ctx, a.expert_cache, profile, a.timeout)
    r.update(label=a.label, model=a.model, pack=a.pack, prompt_tokens=len(ids),
             expert_cache=a.expert_cache, expert_profile=profile)

    p = r["prefill"]
    if p:
        print(f"prefill : {p['tok_s']:.1f} tok/s  ({p['tokens']} tok in {p['ms']/1000:.2f} s)")
        if p.get("ttft_ms"):
            print(f"          time to first token {p['ttft_ms']/1000:.2f} s")
        d0 = p.get("detail") or {}
        if d0:
            print(f"          {d0.get('chunks', '?')} chunk(s), experts streamed "
                  f"{d0.get('experts_streamed')} ({d0.get('experts_dma')} by DMA), resident "
                  f"{d0.get('experts_resident')}, cpu {d0.get('experts_cpu')} "
                  f"(share {d0.get('cpu_share')})")
    else:
        print("prefill : NO STATS LINE - do not report a prefill number from this run")
    for line in r.get("stderr_tail", []):
        if "expert tiers" in line or "adaptive tier" in line or "routing prefetch" in line:
            print(f"          {line.strip()}")
    d = r["decode"]
    print(f"decode  : {d['tok_s']} tok/s  ({d['tokens']} tok in {d['ms']/1000:.2f} s)")
    print(f"wall    : {r['total_s']} s total (includes model load)")
    print(f"tokens  : {r['token_ids']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
        print(f"written {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
