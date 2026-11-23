#!/usr/bin/env python3
"""Harness windows -> sub-windows behind a carried prompt (xolotl-inkling --train --prefix prefix.json, W9).

harness_build_inkling.py writes prefix-free windows (the conversation after the shared system prompt + tool schemas).
Trained behind that prompt, a window's attention spans prompt + window, so its length is capped by attention memory
(prompt x window): each window is cut into sub-windows of T = clamp(budget / (n_prompt + n_context), min_t, max_t)
tokens. A sub-window line carries "context" = the window's tokens before it (the trainer prepends the prompt), and
overlaps the previous piece by one token so every loss token is predicted.

Selection (all optional, combined with OR): --producer REGEX on the window's producer, --marker S (repeatable): the
window's loss text contains S. --max-windows caps the count (random, --seed). --heldout-every N sends every Nth chosen
window to eval.jsonl.

  INKLING_PACK=<pack> python3 harness_subwindows.py --windows harness/windows.jsonl --prefix harness/prefix.json \\
      --out harness-r2 --producer claude-opus --marker '"Edit"' --marker __global__ --budget 8e6
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_build_inkling import load_tokenizer, TOKDIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--prefix", required=True, help="prefix.json of the same build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--producer", default="", help="regex on the window's producer")
    ap.add_argument("--marker", action="append", default=[], help="keep windows whose loss text contains this")
    ap.add_argument("--budget", type=float, default=8e6)
    ap.add_argument("--min-t", type=int, default=64)
    ap.add_argument("--max-t", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--heldout-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_prompt = len(json.load(open(a.prefix))["tokens"])
    tok = load_tokenizer(TOKDIR) if a.marker else None
    prod = re.compile(a.producer) if a.producer else None
    chosen, why = [], {"producer": 0, "marker": 0}
    for line in open(a.windows):
        r = json.loads(line)
        keep = False
        if prod and prod.search(r.get("producer") or ""):
            keep = True
            why["producer"] += 1
        if not keep and tok:
            text = tok.decode([t for t, m in zip(r["ids"], r["mask"]) if m])
            if any(s in text for s in a.marker):
                keep = True
                why["marker"] += 1
        if keep or (not prod and not a.marker):
            chosen.append(r)
    rng = random.Random(a.seed)
    rng.shuffle(chosen)
    if a.max_windows:
        chosen = chosen[:a.max_windows]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ft, fe = open(out / "windows.jsonl", "w"), open(out / "eval.jsonl", "w")
    st = {"windows": len(chosen), "chosen_by": why, "sub_windows": 0, "eval_sub_windows": 0, "loss_tokens": 0,
          "eval_loss_tokens": 0, "n_prompt": n_prompt, "t_hist": {}}
    for k, r in enumerate(chosen):
        f = fe if a.heldout_every and k % a.heldout_every == 0 else ft
        ids, mask = r["ids"], r["mask"]
        s = 0
        while s < len(ids) - 1:
            T = int(max(a.min_t, min(a.max_t, a.budget // (n_prompt + s))))
            piece, pm = ids[s:s + T], mask[s:s + T]
            if sum(pm[1:]) > 0:
                f.write(json.dumps({"id": f"{r['id']}:{s}", "context": ids[:s], "ids": piece, "mask": pm},
                                   separators=(",", ":")) + "\n")
                key = "eval_" if f is fe else ""
                st[key + "sub_windows"] += 1
                st[key + "loss_tokens"] += sum(pm[1:])
                st["t_hist"][str(T)] = st["t_hist"].get(str(T), 0) + 1
            s += T - 1   # overlap one token: the next piece's first row predicts this piece's successor
    ft.close()
    fe.close()
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "t_hist"}, indent=1))


if __name__ == "__main__":
    main()
