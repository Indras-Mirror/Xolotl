#!/usr/bin/env python3
"""tools/dense_bench.py - measure `xolotl-dense` without the HTTP server.

WHY.  Tuning the dense decode needs a number that comes back the same way every time.  Going through
serve/server.py adds the chat template, the sampler defaults and a websocket, and it can only be run by whoever
owns the running server.  This talks the engine's own line protocol directly, so the same command measures a
laptop and a CI box, and it works while another engine holds the GPU (it just has to wait for the card).

USAGE
    python tools/dense_bench.py --gguf <model.gguf> [--context 56000] [--kv q4_0]
        [--mtp --draft-max 2] [--prompt PATH] [--prompt-tokens 4000] [--max-new 256]
        [--repeats 3] [--no-prefill] [--env NAME=VALUE] [--json OUT.json]

The prompt is either a text file (--prompt) or, by default, a synthetic block of code-like text tokenized to
--prompt-tokens with the reference tokenizer (tools/strata_tokenizer.py).  Sampling is greedy (temperature 0) so
two runs are comparable token by token.

WHAT IT PRINTS.  One line per repeat with prompt tokens, PP tok/s, generated tokens, decode tok/s and the MTP
acceptance, then the mean decode tok/s.  --json writes the same records for diffing two builds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

EXE = "xolotl-dense.exe" if os.name == "nt" else "xolotl-dense"

# A prompt with real structure: the model has to read it, and its length is easy to control.  Repeated with
# variation so the prompt is not one giant n-gram the drafter could memorize.
SEED = '''
def rolling_window(series, width):
    """Yield the mean of a sliding window over a list of floats."""
    if width <= 0:
        raise ValueError("width must be positive")
    total = 0.0
    queue = []
    for i, value in enumerate(series):
        queue.append(value)
        total += value
        if len(queue) > width:
            total -= queue.pop(0)
        if len(queue) == width:
            yield i - width + 1, total / width


class EvictionCache:
    """Least-recently-used cache with a byte budget, not an entry budget."""

    def __init__(self, budget_bytes, cost=lambda v: len(v)):
        self.budget = budget_bytes
        self.cost = cost
        self.used = 0
        self.entries = {}

    def get(self, key):
        if key not in self.entries:
            return None
        value, size = self.entries.pop(key)
        self.entries[key] = (value, size)          # mark as recently used
        return value

    def put(self, key, value):
        size = self.cost(value)
        if size > self.budget:
            raise ValueError("value larger than the whole budget")
        while self.used + size > self.budget:
            oldest = next(iter(self.entries))
            _, freed = self.entries.pop(oldest)
            self.used -= freed
        self.entries[key] = (value, size)
        self.used += size

    def __len__(self):
        return len(self.entries)
'''


def build_prompt(tokenizer, target_tokens: int) -> list[int]:
    ids: list[int] = []
    block = SEED
    while len(ids) < target_tokens:
        ids = tokenizer.encode(block, parse_special=True)
        block += "\n\n" + SEED
    return ids[:target_tokens]


def load_tokenizer(gguf: Path, tokenizer_dir: Path | None):
    from strata_tokenizer import Tokenizer  # the reference implementation of record
    if tokenizer_dir and Path(tokenizer_dir).exists():
        for name in ("vocab.json", "merges.txt"):
            if not (Path(tokenizer_dir) / name).exists():
                raise SystemExit(f"tokenizer dir {tokenizer_dir} is missing {name}")
        return Tokenizer.from_gguf(str(Path(gguf)))
    return Tokenizer.from_gguf(str(Path(gguf)))


def run_once(exe: Path, args: list[str], prompt: list[int], max_new: int, env: dict,
             timeout: float) -> dict:
    proc = subprocess.Popen([str(exe)] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env)
    out: list[str] = []
    err: list[str] = []

    def pump(stream, sink):
        for line in stream:
            sink.append(line.rstrip("\n"))

    import threading
    to = threading.Thread(target=pump, args=(proc.stdout, out), daemon=True)
    te = threading.Thread(target=pump, args=(proc.stderr, err), daemon=True)
    to.start()
    te.start()

    ready = False
    t0 = time.time()
    while not ready and time.time() - t0 < timeout:
        if any(l.startswith("READY ") for l in out):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    if not ready:
        proc.kill()
        raise SystemExit("engine never became READY:\n" + "\n".join(err[-25:]))

    proc.stdin.write("GEN %d temperature=0 top_k=1 %s\n" % (max_new, ",".join(map(str, prompt))))
    proc.stdin.flush()

    done = None
    produced = 0
    while done is None and time.time() - t0 < timeout:
        if any(l.startswith("DONE ") for l in out):
            done = next(l for l in out if l.startswith("DONE "))
            break
        if any(l.startswith("ERR ") for l in out):
            proc.kill()
            raise SystemExit("engine error: " + next(l for l in out if l.startswith("ERR ")))
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    if done is None:
        proc.kill()
        raise SystemExit("no DONE within %.0f s" % timeout)

    produced, n_prompt, prompt_ms, decode_ms, finish, accepted, offered, reused = done.split()[1:9]
    rec = {
        "prompt_tokens": int(n_prompt),
        "reused": int(reused),
        "generated": int(produced),
        "prompt_ms": float(prompt_ms),
        "decode_ms": float(decode_ms),
        "finish": finish,
        "drafts_accepted": int(accepted),
        "drafts_offered": int(offered),
    }
    rec["pp_tok_s"] = 1000.0 * rec["prompt_tokens"] / rec["prompt_ms"] if rec["prompt_ms"] else 0.0
    rec["decode_tok_s"] = 1000.0 * rec["generated"] / rec["decode_ms"] if rec["decode_ms"] else 0.0
    # the per-cycle breakdown the engine prints to stderr, when MTP is on
    prof = [l for l in err if "speculative cycles" in l]
    if prof:
        rec["cycle_line"] = prof[-1]
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return rec, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--exe", default=str(ROOT / "build-dense" / EXE))
    ap.add_argument("--tokenizer", default="", help="tokenizer dir (default: read the GGUF itself)")
    ap.add_argument("--context", type=int, default=56000)
    ap.add_argument("--kv", default="q4_0")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--draft-max", type=int, default=2)
    ap.add_argument("--draft-min", type=int, default=1)
    ap.add_argument("--draft-p-min", type=float, default=0.0)
    ap.add_argument("--prompt", default="", help="text file used as the prompt")
    ap.add_argument("--prompt-tokens", type=int, default=4000)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--prompt-chunk", type=int, default=0)
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for the engine (STRATA_DENSE_PROF=1)")
    ap.add_argument("--json", default="", help="write the records here")
    ap.add_argument("--timeout", type=float, default=1200.0)
    a = ap.parse_args()

    gguf = Path(a.gguf)
    if not gguf.exists():
        raise SystemExit(f"no such GGUF: {gguf}")
    exe = Path(a.exe)
    if not exe.exists():
        raise SystemExit(f"no such engine: {exe} (build it with tools/dense_setup.py --build)")

    if a.prompt:
        text = Path(a.prompt).read_text(encoding="utf-8")
        tok = load_tokenizer(gguf, a.tokenizer or None)
        prompt = tok.encode(text, parse_special=True)[: a.prompt_tokens]
    else:
        tok = load_tokenizer(gguf, a.tokenizer or None)
        prompt = build_prompt(tok, a.prompt_tokens)

    args = ["--serve", "--native", str(gguf), "--context", str(a.context), "--kv", a.kv]
    if a.mtp:
        args += ["--mtp", "--draft-max", str(a.draft_max), "--draft-min", str(a.draft_min),
                 "--draft-p-min", str(a.draft_p_min)]
    if a.no_prefill:
        args += ["--no-prefill"]
    elif a.prompt_chunk:
        args += ["--prompt-chunk", str(a.prompt_chunk)]

    env = dict(os.environ)
    for e in a.env:
        k, _, v = e.partition("=")
        env[k] = v

    print(f"prompt {len(prompt)} tokens, max_new {a.max_new}, kv {a.kv}, mtp {int(a.mtp)} "
          f"(draft_max {a.draft_max}), context {a.context}", flush=True)

    records = []
    for i in range(a.repeats):
        rec, err = run_once(exe, args, prompt, a.max_new, env, a.timeout)
        records.append(rec)
        acc = f", drafts {rec['drafts_accepted']}/{rec['drafts_offered']}" if rec["drafts_offered"] else ""
        print(f"  run {i + 1}: PP {rec['pp_tok_s']:7.1f} tok/s ({rec['prompt_ms'] / 1000:6.1f} s), "
              f"decode {rec['decode_tok_s']:6.1f} tok/s ({rec['generated']} tok, "
              f"{rec['decode_ms'] / 1000:5.1f} s){acc}", flush=True)
        if "cycle_line" in rec:
            print(f"    {rec['cycle_line']}", flush=True)

    decodes = [r["decode_tok_s"] for r in records]
    pps = [r["pp_tok_s"] for r in records]
    if decodes:
        print(f"mean decode {sum(decodes) / len(decodes):.1f} tok/s, mean PP {sum(pps) / len(pps):.1f} tok/s")
    if a.json:
        Path(a.json).write_text(json.dumps({"args": args, "records": records}, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
