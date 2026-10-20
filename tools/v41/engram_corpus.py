#!/usr/bin/env python3
"""tools/v41/engram_corpus.py - token corpus for the Engram hot set (tools/v41/engine/ds41_engram_hot.cpp).

Writes OUT/train.i32 and OUT/heldout.i32: int32 token ids, -1 between documents.  Sources: the rendered calibration
corpus (/media/Crucial1TB/v41-calib/corpus/*.jsonl - agentic sessions, agent, code, reasoning, language, translation)
plus NeelNanda/pile-10k (offline HF cache; StackExchange rendered as DeepSeek chat turns, as v41_calib.py prep does).
Every source holds out every 10th document (deterministic), so held-out text is never ranked.

  python3 tools/v41/engram_corpus.py --tokenizer /media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017 --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="checkpoint dir with the tokenizer files")
    ap.add_argument("--corpus", default="/media/Crucial1TB/v41-calib/corpus")
    ap.add_argument("--no-pile", action="store_true")
    ap.add_argument("--every", type=int, default=10, help="hold out every Nth document of each source")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    sources = {}
    for f in sorted(glob.glob(os.path.join(a.corpus, "*.jsonl"))):
        sources[os.path.basename(f)[:-6]] = [json.loads(l)["text"] for l in open(f)]
    if not a.no_pile:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        import datasets
        pile = datasets.load_dataset("NeelNanda/pile-10k", split="train")
        for r in pile:
            name, t = "pile_" + r["meta"]["pile_set_name"].split(" ")[0].lower(), r["text"]
            if name == "pile_stackexchange" and "\nA:\n" in t:   # as v41_calib.py prep: a non-thinking chat turn
                q, ans = t.split("\nA:\n", 1)
                t = f"<｜User｜>{q.replace('Q:', '', 1).strip()}<｜Assistant｜></think>{ans.strip()}<｜end▁of▁sentence｜>"
            sources.setdefault(name, []).append(t)

    os.makedirs(a.out, exist_ok=True)
    out = {"train": [], "heldout": []}
    stats = {}
    for name, docs in sources.items():
        ids = tok(docs, add_special_tokens=False)["input_ids"]
        for i, d in enumerate(ids):
            if not d:
                continue
            split = "heldout" if i % a.every == a.every - 1 else "train"
            out[split].append(np.asarray(d, dtype=np.int32))
            s = stats.setdefault(name, [0, 0, 0])
            s[0 if split == "train" else 1] += 1
            s[2] += len(d)
    for split, docs in out.items():
        sep = np.array([-1], dtype=np.int32)
        arr = np.concatenate([x for d in docs for x in (d, sep)]) if docs else np.zeros(0, np.int32)
        arr.tofile(os.path.join(a.out, f"{split}.i32"))
        print(f"{split}: {len(docs)} docs, {int((arr >= 0).sum())} tokens")
    for name, (tr, ho, nt) in sorted(stats.items()):
        print(f"  {name:28s} train {tr:5d}  held-out {ho:4d}  tokens {nt}")


if __name__ == "__main__":
    main()
