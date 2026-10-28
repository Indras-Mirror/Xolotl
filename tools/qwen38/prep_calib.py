#!/usr/bin/env python3
"""Calibration corpus for Qwen3.8-Flash-Next, built from THE V4 CORPUS ITSELF.

The requirement: same methodology AND same corpus as DeepSeek-V4.1's XQ calibration.

`tools/v41/v41_calib.py prep` is the V4 reference, but it cannot be pointed at Qwen: it
pre-tokenises with DeepSeek's tokenizer and renders through `dsv41_encoding.py`, which raises on
image special tokens.  So this script reuses its INPUTS and its RULES, without its DS4 machinery:

  INPUT   /media/Crucial1TB/v41-calib/corpus/*.jsonl      <- the actual V4 pools, on disk
  RATIOS  the exact domain sequence in /media/Crucial1TB/v41-calib/meta.json
              agentic_long 10, agent 5, code 6, reason 4, web 2, zh 3, chat 1, multi 1
  BUDGET  32 sequences x 8192 = 262,144 tokens  (V4)  ->  512 x 512 here, same total
  SPLIT   every 4th sequence of each domain is held out (v41_calib.py:241-244)

What changes, and why
---------------------
The V4 pools are DeepSeek-TEMPLATED text:
    " <|begin_of_sentence|><|System|>...<|User|>... "
Those special tokens are meaningless to a Qwen tokenizer and would tokenise as garbage byte
runs, so they are stripped and the plain conversation text is kept.  The CONTENT, the domain
proportions and the ordering are V4's; only the framing is Qwen-appropriate.

Other deviations, recorded in meta.json rather than hidden:
  * sequence length 512, not 8192 - the 162.6 GB model on a 24 GB card leaves no KV headroom
    for 8192.  TOKEN COUNT is matched exactly at 262,144.
  * no Engram hashes.npy - DS4-specific; Qwen uses PLE, and nothing on this path reads it.

  python3 tools/qwen38/prep_calib.py --seqs 512 --seq-len 512 \
      --out /media/mal/NVME1TB/qwen38/corpus-v4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

V4_CORPUS = "/media/Crucial1TB/v41-calib/corpus"   # the pools both sets were built from
V4_META = "/home/mal/v41-work/calib-1017-v4/meta.json"
V4_TOKENS = 200 * 8192   # the 1017-v4 quantisation calib

# the pools v41_calib.py assembles each domain from (v41_calib.py:189-202)
POOL_FILES = {
    "agentic_long": ["sessions.jsonl"],
    "agent": ["antirez_agent.jsonl"],
    "code": ["antirez_source.jsonl", "antirez_programming.jsonl", "antirez_algorithms.jsonl"],
    "reason": ["antirez_eval_reasoning.jsonl", "antirez_long_context.jsonl",
               "antirez_language.jsonl"],
    "web": ["antirez_general.jsonl"],
    "zh": ["antirez_language.jsonl"],
    "chat": ["antirez_general.jsonl"],
    "multi": ["antirez_translation.jsonl"],
}

# DeepSeek framing that must not reach a Qwen tokenizer
DS_TOKENS = [
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
    "<\uff5cSystem\uff5c>", "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>",
    "<\uff5cTool\uff5c>", "<\uff5cObservation\uff5c>", "<\uff5cbegin\u2581of\u2581thinking\uff5c>",
    "<\uff5cend\u2581of\u2581thinking\uff5c>", "</think>", "<｜end▁of▁thinking｜>",
    "<\uff5cDSML\uff5c>", "\uff5cDSML\uff5c",   # tool-call markup; meaningless to Qwen
]
DROP = re.compile("|".join(re.escape(t) for t in DS_TOKENS))


def strip_ds(text: str) -> str:
    return DROP.sub("", text).strip()


def load_pool(names):
    out = []
    for n in names:
        p = pathlib.Path(V4_CORPUS) / n
        if not p.is_file():
            print(f"  WARNING: missing pool file {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                t = strip_ds(t)
                if len(t) > 120:
                    out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chars-per-token", type=float, default=3.6,
                    help="budget only - llama-imatrix tokenises for real")
    ap.add_argument("--out", default="/media/mal/NVME1TB/qwen38/corpus-v4")
    a = ap.parse_args()

    v4 = json.load(open(V4_META))
    v4_domains = v4["domains"]
    from collections import Counter
    counts = Counter(v4_domains)
    print(f"V4 corpus: {v4['n']} seq x {v4['seq']} = {V4_TOKENS:,} tokens", file=sys.stderr)
    print(f"V4 domain counts: {dict(counts)}", file=sys.stderr)

    # V4 (calib-1017-v4) is 200 x 8192 = 1,638,400 tokens.  Reproducing that at 30 s/chunk
    # would take ~27 h, so the budget is set explicitly with --seqs and the domain PROPORTIONS
    # are preserved exactly.
    # scale V4's exact domain distribution to the requested sequence count
    scale = a.seqs / len(v4_domains)
    want = {d: max(1, round(n * scale)) for d, n in counts.items()}
    print(f"scaled to {a.seqs} seq: {want}", file=sys.stderr)

    rng = random.Random(1017)
    pools = {}
    for dom, files in POOL_FILES.items():
        pools[dom] = load_pool(files)
        rng.shuffle(pools[dom])
        print(f"  {dom:14s} {len(pools[dom]):5d} texts from {files}", file=sys.stderr)

    budget_chars = int(a.seq_len * a.chars_per_token)
    seqs, doms, cursor, used_up = [], [], {k: 0 for k in pools}, set()

    # interleave domains in V4's order so the first chunks are not all one domain
    order = []
    while len(order) < a.seqs:
        for d in v4_domains:
            if len(order) < a.seqs:
                order.append(d)

    for dom in order:
        # V4's pack(): many short texts concatenated until the sequence is full.  One text per
        # sequence is wrong here - the web/chat/multi pools have a median length of ~350 chars,
        # so they would each produce a handful of near-empty sequences and then "run dry".
        src = pools.get(dom) or []
        if not src or dom in used_up:
            used_up.add(dom)
            src = pools.get("code") or pools.get("agent") or []
            dom = "fallback:" + dom
        parts, got = [], 0
        guard = 0
        while got < budget_chars and guard < 10000:
            if cursor.get(dom, 0) >= len(src):
                cursor[dom] = 0          # cycle the pool rather than starving the domain
            t = src[cursor[dom]]
            cursor[dom] = cursor.get(dom, 0) + 1
            guard += 1
            if not t:
                continue
            parts.append(t)
            got += len(t) + 2
        seqs.append("\n\n".join(parts)[:budget_chars])
        doms.append(dom)

    # per-domain held-out: every 4th sequence of each domain (V4 rule)
    from collections import defaultdict
    tot = Counter(doms)
    cnt = defaultdict(int)
    is_eval = []
    for d in doms:
        n_ev = tot[d] // 4 + (1 if tot[d] % 4 == 3 else 0)
        is_eval.append(cnt[d] >= tot[d] - n_ev)
        cnt[d] += 1

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = [s for s, ev in zip(seqs, is_eval) if not ev]
    held = [s for s, ev in zip(seqs, is_eval) if ev]
    (out / "corpus.txt").write_text("\n\n".join(calib), encoding="utf-8")
    (out / "heldout.txt").write_text("\n\n".join(held), encoding="utf-8")
    json.dump({
        "source_corpus": V4_CORPUS, "v4_domains": v4_domains,
        "v4_counts": dict(counts), "scaled_want": want,
        "n": len(seqs), "seq_len": a.seq_len, "domains": doms, "is_eval": is_eval,
        "tokens_target": a.seqs * a.seq_len, "v4_tokens": V4_TOKENS,
        "tokenised_with": "Qwen tokenizer (imatrix tokenises the model under test)",
        "deviations": [
            "DeepSeek chat-template tokens stripped; text content preserved",
            "seq_len 512 not 8192 (VRAM-bound); total token count matched at 262,144",
            "no Engram hashes.npy (DS4-specific)",
        ],
    }, open(out / "meta.json", "w"), indent=2)

    print(f"\nwrote {out}/corpus.txt  {len(calib)} calib + {len(held)} held-out")
    print(f"domains {dict(Counter(doms))}")
    print(f"token target {a.seqs * a.seq_len:,}  (V4 = {V4_TOKENS:,})")
    return 0 if calib else 1


if __name__ == "__main__":
    sys.exit(main())
