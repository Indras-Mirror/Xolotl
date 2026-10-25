#!/usr/bin/env python3
"""lite_prep.py - the calibration corpus of the HXQ3K-lite build (Q3_K / Q2_0 mixed experts), v41_calib's prep format:
WORK/tokens.npy [N, S] int32, WORK/meta.json (domain + is_eval per sequence), WORK/hashes.npy (Engram n-gram ids).

Public sources only (no private transcripts):
  antirez/ds4 @ fc80bd69 imatrix prompts (agent / source / language / translation / eval_reasoning / ...), re-rendered
      with DeepSeek's V4.1 encoder by build_corpus.build_antirez();
  IlyaGusev/ru_turbo_saiga (Russian chats), rendered as V4.1 chat turns;
  NeelNanda/pile-10k (web / science / StackExchange);
  local C++/CUDA/Python sources of Xolotl (public, MIT).
Approximately one quarter of each domain is held out (the screening error is measured on those only), as v41_calib prep.

  V41_SRC=<checkpoint dir> V41_WORK=<work dir> python3 lite_prep.py --src DIR [--n 96] [--seq 2048]
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V41 = os.path.dirname(HERE)
sys.path.insert(0, V41)
import torch  # noqa: E402
import v41_calib as C  # noqa: E402
import build_corpus as BC  # noqa: E402
import dsv41_encoding as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="corpus-src dir (prompts.jsonl, ru_turbo_saiga.jsonl, pile10k.jsonl)")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    os.makedirs(C.WORK, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(C.MAIN)
    BOS, EOS = tok.bos_token_id, tok.eos_token_id
    S, rng = a.seq, np.random.default_rng(1017)

    BC.ANTIREZ = os.path.join(a.src, "prompts.jsonl")
    ant = BC.build_antirez()
    at = lambda *ks: [x["text"] for k in ks for x in ant.get(k, [])]

    ru = []
    for line in open(os.path.join(a.src, "ru_turbo_saiga.jsonl")):
        r = json.loads(line)
        msgs = [{"role": "assistant" if m["role"] == "bot" else m["role"], "content": m["content"]} for m in r["messages"]]
        if len(msgs) < 2 or msgs[0]["role"] not in ("user", "system"):
            continue
        try:
            ru.append(E.encode_messages(msgs, thinking_mode="chat"))
        except Exception:
            pass
    by = defaultdict(list)
    for line in open(os.path.join(a.src, "pile10k.jsonl")):
        r = json.loads(line)
        by[r["set"]].append(r["text"])

    def chat(t):   # StackExchange "Q: .. A: .." -> a V4.1 non-thinking chat turn (as v41_calib prep)
        if "\nA:\n" not in t:
            return None
        q, ans = t.split("\nA:\n", 1)
        q = q.replace("Q:\n", "", 1).strip()
        return f"<｜User｜>{q}<｜Assistant｜></think>{ans.strip()}<｜end▁of▁sentence｜>"
    web = [t for s in ("Pile-CC", "OpenWebText2", "Wikipedia (en)", "HackerNews", "FreeLaw", "USPTO Backgrounds") for t in by[s]]
    sci = [t for s in ("PubMed Abstracts", "PubMed Central", "ArXiv", "DM Mathematics") for t in by[s]]
    se = [c for c in map(chat, by["StackExchange"]) if c]
    code_local = []
    for pat in ("src/**/*.cpp", "src/**/*.cu", "tools/v41/engine/*.cpp", "tools/v41/serve/*.py", "tools/v41/*.py"):
        for p in sorted(glob.glob(os.path.join(V41, "..", "..", pat), recursive=True)):
            try:
                t = open(p, encoding="utf-8").read()
            except Exception:
                continue
            if len(t) > 3000:
                code_local.append(t)
    pools = {
        "agent": at("agent"),
        "code": at("source", "programming", "algorithms") + by["Github"][:300] + code_local,
        "reason": at("eval_reasoning", "long_context", "general") + sci,
        "ru": ru,
        "multi": at("language", "translation"),
        "web": web + se,
    }
    for k in pools:
        rng.shuffle(pools[k])
    want = {"agent": 22, "code": 20, "reason": 10, "ru": 16, "multi": 10, "web": 18}
    scale = a.n / sum(want.values())
    want = {k: max(1, round(v * scale)) for k, v in want.items()}

    def pack(texts, n):   # v41_calib prep's packing (BOS per sequence, a long document continues once)
        seqs, cur = [], [BOS]
        for t in texts:
            special = t.startswith("<｜User｜>") or t.startswith("<｜begin▁of▁sentence｜>")
            if t.startswith("<｜begin▁of▁sentence｜>"):
                t = t[len("<｜begin▁of▁sentence｜>"):]
            ids = tok.encode(t, add_special_tokens=False)[: S - 1]
            if not special:
                ids = ids + [EOS]
            while ids:
                room = S - len(cur)
                cur += ids[:room]; ids = ids[room:]
                if len(cur) == S:
                    seqs.append(cur); cur = [BOS]
                    if len(seqs) == n:
                        return seqs
                    ids = ids[: S // 2]
        return seqs
    seqs, doms = [], []
    for k, n in want.items():
        got = pack(pools[k], n)
        C.log(f"lite prep: {k}: {len(got)}/{n} sequences ({len(pools[k])} texts)")
        seqs += got; doms += [k] * len(got)
    seqs, doms = seqs[: a.n], doms[: a.n]
    tot, is_eval, cnt = Counter(doms), [], defaultdict(int)
    for d_ in doms:
        n_ev = tot[d_] // 4 + (1 if tot[d_] % 4 == 3 else 0)
        is_eval.append(cnt[d_] >= tot[d_] - n_ev); cnt[d_] += 1
    tokens = np.array(seqs, dtype=np.int32)
    np.save(C.wpath("tokens.npy"), tokens)
    json.dump({"n": len(seqs), "seq": S, "domains": doms, "is_eval": is_eval, "corpus": "lite: antirez+saiga+pile10k+xolotl-src"},
              open(C.wpath("meta.json"), "w"))
    args = C.model_args(len(seqs), S)
    layout = C.M.EngramLayout.from_args(args)
    hs = C.M.NgramHashState(args, layout, tok)
    with torch.inference_mode():
        hashes = hs(torch.from_numpy(tokens).long(), 0)
    np.save(C.wpath("hashes.npy"), hashes.numpy())
    C.log(f"lite prep: {tokens.shape} tokens, {sum(is_eval)} eval sequences ({dict(Counter(d for d, e in zip(doms, is_eval) if e))}), "
          f"hashes {tuple(hashes.shape)}")


if __name__ == "__main__":
    main()
