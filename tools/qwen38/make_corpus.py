#!/usr/bin/env python3
"""Build a plain-text calibration corpus for llama-imatrix from the local agent logs.

Why not tools/v41/build_corpus.py
---------------------------------
That builder renders every conversation through DeepSeek's chat template
(`tools/v41/dsv41_encoding.py`) so the calibration sees exactly what the DS4 server saw.  Two
things make it unusable for this model:

  1. its tokenizer path is DeepSeek's and transformers rejects `qwen4_exp` configs, and
  2. `dsv41_encoding.encode_messages` raises on any message carrying a DeepSeek image token
     (`_validate_no_image_sp_tokens`, dsv41_encoding.py:359), which the Qwen and pi session logs
     do contain.

`llama-imatrix` reads raw text and tokenises it with the model under test, so the template was
never load-bearing here - only the domain mix is.  This walks the same three session roots,
extracts plain text, and writes one .txt per pool plus a concatenated corpus.txt.

  python3 tools/qwen38/make_corpus.py [--max-tokens 2000000] [--out DIR]

Pools mirror the DS4 mix intent: quetza + claude-code + claude-data sessions give the
code/agentic half; whatever prose survives gives the rest.  Held-out: the last 10% of each pool
goes to <out>/heldout.txt and is NOT in corpus.txt.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import random
import sys

ROOTS = [
    ("~/.quetza-data", "quetza"),
    ("~/.claude/projects", "claude-code"),
    ("~/.claude-data", "claude-data"),
]
MIN_BYTES = 50_000


def _text_of(content):
    """Flatten a message content field to text, dropping non-text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif isinstance(b.get("content"), str) and b.get("type") == "tool_result":
                    out.append(b["content"])
        return "\n".join(out)
    return ""


def harvest(path: str) -> list[str]:
    """Every text message in one session log, in order. Tolerant of format drift."""
    chunks = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # claude-code / quetza: {"message": {"role","content"}} or {"type":"user","message":...}
                msg = rec.get("message") if isinstance(rec.get("message"), dict) else None
                if msg:
                    t = _text_of(msg.get("content"))
                    if t:
                        chunks.append(t)
                    continue
                # pi / simpler: {"role","content"}
                if "role" in rec and "content" in rec:
                    t = _text_of(rec["content"])
                    if t:
                        chunks.append(t)
    except OSError:
        return []
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--out", default="/media/mal/NVME1TB/qwen38/corpus")
    ap.add_argument("--chars-per-token", type=float, default=3.6,
                    help="rough budget only; llama-imatrix tokenises for real")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_chars = int(a.max_tokens * a.chars_per_token)

    pools: dict[str, list[str]] = {}
    for root, name in ROOTS:
        r = os.path.expanduser(root)
        files = [f for f in glob.glob(os.path.join(r, "**", "*.jsonl"), recursive=True)
                 if os.path.getsize(f) > MIN_BYTES]
        chunks: list[str] = []
        for f in files:
            chunks.extend(harvest(f))
        pools[name] = chunks
        print(f"  {name}: {len(files)} files -> {len(chunks)} text chunks, "
              f"{sum(len(c) for c in chunks)/1e6:.1f}M chars", file=sys.stderr)

    # interleave the pools so the corpus does not read as one workspace repeated
    rng = random.Random(20261027)
    merged: list[str] = []
    for chunks in pools.values():
        c = chunks[:]
        rng.shuffle(c)
        merged.extend(c)
    rng.shuffle(merged)

    # join until the character budget is reached, splitting at message boundaries
    body, used = [], 0
    for c in merged:
        if used >= budget_chars:
            break
        body.append(c)
        used += len(c) + 2

    # held-out = last 10% by chunk, kept out of the calibration file
    cut = int(len(body) * 0.9)
    calib, held = body[:cut], body[cut:]

    with open(out / "corpus.txt", "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(calib))
    with open(out / "heldout.txt", "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(held))

    print(f"wrote {out}/corpus.txt  {sum(len(c) for c in calib)/1e6:.1f}M chars "
          f"({len(calib)} chunks)")
    print(f"wrote {out}/heldout.txt {sum(len(c) for c in held)/1e6:.1f}M chars "
          f"({len(held)} chunks)")
    if not calib:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
