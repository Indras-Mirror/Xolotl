#!/usr/bin/env python3
"""Build the QuetzaCodetl HARNESS SFT set for Inkling-2-Mini from our own chat corpus.

Adapted from the QFN builder (lora-train/tools/train/harness_build.py, section 4.8), rendering through Inkling's own
serve path instead of a Jinja template:

    frontend.anthropic_to_messages(req) -> serve.inkling.render(messages, tools, reasoning_effort=...)

which is exactly what serve/server.py:render_prompt does for family "inkling2"/"inkling2_loop".  Inkling has no
generation prompt: the model writes its own `<|message_model|>` header, and one assistant turn is a list of blocks
(thinking / text / tool JSON) closed by `<|end_message|>`, then `<|content_model_end_sampling|>` (EOS 200006).

Tokenisation reproduces serve/server.py:encode_prompt: control-token texts inside message content are marked before the
render and encoded as plain text (frontend.mark_think_literals/unmark_think_literals), the per-request cache marker is
stripped, and the ids are the pack/GGUF tokenizer's (`inkling2` = GPT-4o o200k pre-tokenizer).

usage:
  python3 harness_build_inkling.py [--chats A,B] [--req REQ] [--out DIR] [--limit N]
         [--max-conv-tokens 2048] [--max-window-tokens 3200] [--heldout-frac 0.02] [--seed 0]

Outputs (DIR): prefix.json, windows.jsonl, stats.json, heldout.txt.
  prefix.json   {"tokens", "n", "chars", "sha256", "tools"}
  windows.jsonl {"id", "conv", "started", "source", "producer", "n", "n_loss", "ids", "mask"[, "plain"]}
                ids = the window's tokens AFTER the prefix; mask = 1 on assistant spans, by byte offset on ONE
                whole-window tokenisation; plain = char spans (relative to the body) encoded as text, or omitted.
  stats.json    counts and producers.
  heldout.txt   conversation ids held out (hash of seed:id, by conversation, never by window); they are NOT in
                windows.jsonl.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("INKLING_REPO", str(Path(__file__).resolve().parents[2])))   # the Xolotl checkout
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from serve import frontend, inkling  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

TOKDIR = Path(os.environ.get("INKLING_PACK", "inkling-2-mini/pack")) / "tokenizer"   # the converted pack's tokenizer dir
DEFAULT_CHATS = None   # --chats A.jsonl,B.jsonl: one conversation per line (messages + tool calls)
DEFAULT_REQ = None     # --req: the request your harness really sends (tools/train/harness_capture_server.py)
DEFAULT_PRODUCERS = r"deepseek|claude-opus|claude-fable|kimi|glm-5|qwen3\.7|qwen4-max"

EOS = inkling.EOS
HEADER = inkling.HEADER


def load_tokenizer(t: Path):
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for k, i in vocab.items():
        tokens[i] = k
    tc = json.loads((t / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((t / "token_type.json").read_text()), tc.get("pre") or "qwen35",
                        tc.get("special_ids") or None)


def to_anthropic(msg):
    """corpus message -> Anthropic message (None when nothing renderable is left). Same mapping as the QFN builder."""
    blocks = []
    for b in msg.get("content") or []:
        t = b.get("type")
        if t == "text" and b.get("text"):
            blocks.append({"type": "text", "text": b["text"]})
        elif t == "thinking" and (b.get("text") or "").strip():
            blocks.append({"type": "thinking", "thinking": b["text"], "signature": ""})
        elif t == "tool_call":
            blocks.append({"type": "tool_use", "id": b.get("id") or "call", "name": b.get("name") or "",
                           "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            blocks.append({"type": "tool_result", "tool_use_id": b.get("id") or "call",
                           "content": c if isinstance(c, str) else json.dumps(c), "is_error": bool(b.get("is_error"))})
        elif t == "attachment" and b.get("text") and msg["role"] == "user":
            blocks.append({"type": "text", "text": b["text"]})
    if not blocks:
        return None
    return {"role": msg["role"], "content": blocks}


def merge_roles(ms):
    """Consecutive same-role messages merged (the API requires alternation)."""
    out = []
    for m in ms:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + m["content"],
                       "_src": out[-1]["_src"] + m["_src"]}
        else:
            out.append(m)
    return out


class Builder:
    def __init__(self, req0, tok, tool_names, tool_params, producers, sources, literals):
        self.req0, self.tok = req0, tok
        self.tool_names, self.tool_params = tool_names, tool_params
        self.prod, self.sources, self.lits = producers, sources, literals

    def render(self, msgs):
        """(text, plain, base_req) — serve's render_prompt + the cache-marker and literal handling of encode_prompt."""
        req = {k: v for k, v in self.req0.items() if k != "messages"}
        req["messages"] = [{"role": m["role"], "content": m["content"]} for m in msgs]
        messages, tools, kw = frontend.anthropic_to_messages(req)
        marker = kw.pop(frontend.CACHE_MARK_KEY, None)
        effort = inkling.effort_from_kwargs(kw)
        marked, marked_tools, changed = frontend.mark_think_literals(messages, tools, self.lits)
        text = inkling.render(marked, marked_tools, reasoning_effort=effort)
        plain = ()
        if changed:
            text, plain = frontend.unmark_think_literals(text, self.lits)
        if marker and marker in text:
            raw, out, last = [], [], 0
            while True:
                i = text.find(marker, last)
                if i < 0:
                    break
                out.append(text[last:i])
                raw.append(i)
                last = i + len(marker)
            out.append(text[last:])
            text = "".join(out)
            plain = [(a - len(marker) * sum(o < a for o in raw), b - len(marker) * sum(o < b for o in raw))
                     for a, b in plain]
        return text, list(plain)


def holdout(cid: str, seed: int, frac: float) -> bool:
    """Deterministic by conversation id: stable when the corpus grows, never by window."""
    h = hashlib.sha256(f"{seed}:{cid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64 < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats", default=DEFAULT_CHATS, required=DEFAULT_CHATS is None)
    ap.add_argument("--req", default=DEFAULT_REQ, required=DEFAULT_REQ is None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-conv-tokens", type=int, default=2048)
    ap.add_argument("--max-window-tokens", type=int, default=3200)
    ap.add_argument("--producers", default=DEFAULT_PRODUCERS)
    ap.add_argument("--sources", default="quetza,quetza-conductor,claude-code,claude-data")
    ap.add_argument("--heldout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-errors", action="store_true",
                    help="mask an assistant turn whose tool call came back is_error (failed command, rejected edit, denial)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N conversations")
    ap.add_argument("--heldout-only", action="store_true",
                    help="render ONLY the held-out conversations (eval windows; use a different --out)")
    a = ap.parse_args()

    frontend.use_family("inkling2_loop")      # numeric effort 0.0..0.99 (frontend._FRACTION_EFFORT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    req0 = json.loads(Path(a.req).read_text())
    tool_names = {t["name"] for t in req0.get("tools") or []}
    tool_params = {t["name"]: set(((t.get("input_schema") or {}).get("properties") or {}).keys())
                   for t in req0.get("tools") or []}
    tok = load_tokenizer(TOKDIR)
    literals = frontend.literal_tags(getattr(tok, "control_tokens", ()))
    prod = re.compile(a.producers)
    local_q = re.compile(r"strata|xq|gsq|xolotl|abl|gguf|<synthetic>", re.I)
    sources = set(a.sources.split(","))
    b = Builder(req0, tok, tool_names, tool_params, prod, sources, literals)

    # the shared prefix: everything before the first user turn (tools block + system message + effort block)
    probe_text, _ = b.render([{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    cut = probe_text.find("<|message_user|>")
    if cut <= 0:
        raise SystemExit("[build] no <|message_user|> in the probe render")
    prefix_text = probe_text[:cut]
    prefix_ids = tok.encode(prefix_text, parse_special=True)
    (out / "prefix.json").write_text(json.dumps({
        "tokens": prefix_ids, "n": len(prefix_ids), "chars": len(prefix_text),
        "sha256": hashlib.sha256(prefix_text.encode()).hexdigest(), "tools": sorted(tool_names)}))
    print(f"[build] prefix: {len(prefix_text)} chars -> {len(prefix_ids)} tokens; tools {sorted(tool_names)}",
          flush=True)

    st = {"convs_seen": 0, "convs_used": 0, "convs_heldout": 0, "windows": 0, "tokens": 0, "loss_tokens": 0,
          "turns_trained": 0, "turns_masked_tool": 0, "turns_masked_producer": 0, "turns_masked_error": 0, "first_stripped": 0,
          "split_mismatch": 0, "prefix_mismatch": 0, "render_error": 0, "window_dropped_over_cap": 0,
          "window_no_span": 0, "prefix_n": len(prefix_ids), "heldout_frac": a.heldout_frac, "seed": a.seed, "heldout_only": a.heldout_only,
          "by_producer": {}, "heldout_ids": []}

    # --chats may name several builds (comma-separated): one conversation per id, the copy with the most messages
    best = {}
    for path in a.chats.split(","):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("source", "").split(":")[0] not in sources:
                    continue
                if r["id"] not in best or len(r.get("messages") or []) > len(best[r["id"]].get("messages") or []):
                    best[r["id"]] = r
    print(f"[build] {len(best)} conversations from {a.chats}", flush=True)

    wf = open(out / "windows.jsonl", "w")
    for r in sorted(best.values(), key=lambda x: x.get("started") or ""):
        if r.get("source") not in sources:
            continue
        ms = r.get("messages") or []
        if not ms or (ms[0].get("meta") or {}).get("isSidechain"):
            continue
        st["convs_seen"] += 1
        hold = holdout(r["id"], a.seed, a.heldout_frac)
        if hold:
            st["convs_heldout"] += 1
            st["heldout_ids"].append(r["id"])
            if not a.heldout_only:
                continue                         # held out by conversation id: never a training window
        elif a.heldout_only:
            continue
        conv = []
        for m in ms:
            am = to_anthropic(m)
            if am is not None:
                am["_src"] = [m]
                conv.append(am)
        conv = merge_roles(conv)
        if len(conv) < 2 or conv[0]["role"] != "user":
            continue

        def trainable(am):
            models = {s.get("model") or "" for s in am["_src"]}
            if not any(prod.search(x) and not local_q.search(x) for x in models):
                return "producer"
            calls = [blk for blk in am["content"] if blk["type"] == "tool_use"]
            if any(c["name"] not in tool_names for c in calls):
                return "tool"
            if any(set(c["input"]) - tool_params[c["name"]] for c in calls):
                return "tool"
            return None

        def errored(j):   # rest[j] is an assistant turn: did one of its tool calls come back is_error in rest[j + 1]?
            if not a.mask_errors or j + 1 >= len(rest):
                return None
            ids = {blk.get("id") for blk in rest[j]["content"] if blk["type"] == "tool_use"}
            if not ids:
                return None
            for blk in rest[j + 1]["content"]:
                if blk.get("type") == "tool_result" and blk.get("is_error") and (blk.get("tool_use_id") in ids or "call" in ids):
                    return "error"
            return None

        try:
            first, rest = conv[0], conv[1:]
            first_text, _ = b.render([first])
            if not first_text.startswith(prefix_text):
                st["prefix_mismatch"] += 1
                continue
            first_ids = len(tok.encode(first_text[len(prefix_text):], parse_special=True))
            if first_ids > 1024:
                typed = []
                for src in first["_src"]:
                    for blk in src.get("content") or []:
                        if blk.get("type") == "text" and blk.get("text"):
                            typed.append({"type": "text", "text": blk["text"]})
                if typed:
                    first = {"role": "user", "content": typed, "_src": first["_src"]}
                    ft, _ = b.render([first])
                    first_ids = len(tok.encode(ft[len(prefix_text):], parse_special=True))
                    st["first_stripped"] += 1
            budget = a.max_conv_tokens - first_ids
            if budget < 256:
                continue
            i, used, seq = 0, False, 0
            while i < len(rest):
                if rest[i]["role"] != "assistant":
                    i += 1
                    continue
                start = i
                seg_end, n_est = i, 0
                while seg_end < len(rest):
                    est = sum(len(json.dumps(blk)) for blk in rest[seg_end]["content"]) // 3 + 8
                    if seg_end > start and n_est + est > budget:
                        break
                    n_est += est
                    seg_end += 1
                # the char/3 estimate can undercount (CJK, base64, dense JSON): shrink the segment until it fits the cap
                while True:
                    msgs = [first] + rest[start:seg_end]
                    full, full_plain = b.render(msgs)
                    if not full.startswith(prefix_text):
                        st["prefix_mismatch"] += 1
                        i, seg_end = start + 1, None
                        break
                    spans, turn_models = [], []
                    for k in range(1, len(msgs)):
                        if msgs[k]["role"] != "assistant":
                            continue
                        why = trainable(msgs[k]) or errored(start + k - 1)
                        if why:
                            st["turns_masked_" + why] += 1
                            continue
                        g, _ = b.render(msgs[:k])
                        e, _ = b.render(msgs[:k + 1])
                        if not full.startswith(g) or not full.startswith(e) or len(e) <= len(g):
                            st["split_mismatch"] += 1
                            continue
                        # the turn's own end-of-sampling closes e (the render appends it to every assistant turn)
                        if e.endswith(EOS):
                            end = len(e)
                        else:
                            at = e.find(EOS, len(g))
                            end = at + len(EOS) if at >= 0 else len(e)
                        spans.append((len(g), end))
                        turn_models.append(next(iter({s.get("model") for s in msgs[k]["_src"]})))
                    if not spans:
                        st["window_no_span"] += 1
                        i, seg_end = seg_end, None
                        break
                    # tokenise the window ONCE (pieces can merge across a span edge) and give each token to the loss by
                    # its byte offset: a token is a loss token when it STARTS inside a span.
                    base = len(prefix_text)
                    body = full[base:]
                    body_plain = [(max(0, s - base), min(len(body), t - base)) for s, t in full_plain
                                  if t > base and s < len(full)]
                    ids = tok.encode(body, parse_special=True, plain=body_plain)
                    bspans = [(len(full[base:s0].encode("utf-8")), len(full[base:s1].encode("utf-8")))
                              for s0, s1 in spans]
                    mask, off = [], 0
                    for t in ids:
                        tb = tok.token_bytes(t)
                        mask.append(1 if any(b0 <= off < b1 for b0, b1 in bspans) else 0)
                        off += len(tb)
                    if off != len(body.encode("utf-8")):
                        st["split_mismatch"] += 1
                        i, seg_end = seg_end, None
                        break
                    if len(ids) <= a.max_window_tokens:
                        break
                    if seg_end - start <= 1:     # one assistant turn alone is over the cap: skip to the next turn
                        st["window_dropped_over_cap"] += 1
                        i, seg_end = start + 1, None
                        break
                    seg_end -= 1
                if seg_end is None:
                    continue
                i = seg_end
                for mdl in turn_models:
                    st["by_producer"][mdl] = st["by_producer"].get(mdl, 0) + 1
                producers = sorted(set(turn_models))
                st["turns_trained"] += len(spans)
                st["windows"] += 1
                st["tokens"] += len(ids)
                st["loss_tokens"] += sum(mask)
                used = True
                rec = {"id": f"{r['id']}:w{seq:04d}", "conv": r["id"], "started": r.get("started"),
                       "source": r.get("source"), "producer": producers[0] if len(producers) == 1
                       else ",".join(producers), "n": len(ids), "n_loss": sum(mask), "ids": ids, "mask": mask}
                if body_plain:
                    rec["plain"] = [[s, t] for s, t in body_plain]
                wf.write(json.dumps(rec) + "\n")
                seq += 1
            if used:
                st["convs_used"] += 1
        except Exception as exc:  # noqa: BLE001 - one odd conversation must not sink the build
            st["render_error"] += 1
            if st["render_error"] <= 5:
                print(f"[build] conv {r['id']} failed: {type(exc).__name__}: {exc}", flush=True)
        if a.limit and st["convs_seen"] >= a.limit:
            print(f"[build] --limit {a.limit} conversations reached", flush=True)
            break
        if st["convs_seen"] % 25 == 0:
            print(f"[build] {st['convs_seen']} convs, {st['windows']} windows, {st['loss_tokens']} loss tokens",
                  flush=True)
    wf.close()
    (out / "heldout.txt").write_text("".join(cid + "\n" for cid in st["heldout_ids"]))
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "heldout_ids"}, indent=1))


if __name__ == "__main__":
    main()
